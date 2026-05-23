"""MAPPO single-file (stile CleanRL) per MARLauder.

On-policy: rollout su N env paralleli x M agenti -> GAE per-agente -> PPO clip + value + entropy.
Policy condivisa (parameter-sharing) + critic centralizzato permutation-invariant (vedi models/networks.py).
Tutto su GPU. Ogni iterazione ricampiona mappe fresche (generalizzazione).

GATE Fase 5: con M=1 la copertura media deve salire sopra il baseline random.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/MARLauder/scripts")
from viz_util import render_marl  # noqa: E402
from rich.console import Group
from rich.live import Live
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table

from env.maps import load_split, sample_batch
from env.marl_env import MarlExploreEnv, EnvConfig
from models.networks import MarlActorCritic

OBS_KEYS = ["coords", "valid", "utility", "guidepost", "pos", "anchors", "anchor_mask",
            "teammate_rel", "teammate_known", "action_mask"]


def slim(obs: dict) -> dict:
    return {k: obs[k].detach() for k in OBS_KEYS}


def cat_obs(obs_list: list[dict]) -> dict:
    return {k: torch.cat([o[k] for o in obs_list], dim=0) for k in OBS_KEYS}


def index_obs(obs: dict, idx: torch.Tensor) -> dict:
    return {k: obs[k][idx] for k in OBS_KEYS}


def gae(rewards, values, dones, last_value, gamma, lam):
    """rewards/values/dones [T,N,M] (dones broadcast su M), last_value [N,M]. Ritorna adv,ret [T,N,M]."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        last_gae = delta + gamma * lam * nonterm * last_gae
        adv[t] = last_gae
    return adv, adv + values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--n-agents", type=int, default=1)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=128)
    ap.add_argument("--reward-scale", type=float, default=500.0)
    ap.add_argument("--runs-root", type=Path, default=Path("/workspace/MARLauder/runs/train"))
    ap.add_argument("--save-pct", type=int, default=20, help="salva ckpt+gif ogni N%% di iter")
    ap.add_argument("--eval-split", default="test/complex")
    ap.add_argument("--eval-steps", type=int, default=160)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda:0"
    split = load_split(args.split)
    N, M, T = args.n_envs, args.n_agents, args.rollout

    net = MarlActorCritic(K=21, a_max=64).to(dev)
    net = torch.compile(net)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    print(f"[mappo] split={args.split} N={N} M={M} T={T} | params={sum(p.numel() for p in net.parameters())/1e6:.2f}M")

    def new_env():
        gt, starts, free, _ = sample_batch(split, N)
        # cov_done=1.0 -> tutti gli env terminano insieme a max_steps (reset sincronizzato)
        return MarlExploreEnv(gt, free, starts,
                              EnvConfig(n_envs=N, n_agents=M, max_steps=args.max_steps,
                                        cov_done=1.0, reward_scale=args.reward_scale))

    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    # env PERSISTENTE attraverso le iterazioni: gli episodi durano max_steps interi,
    # reset (con mappe fresche) solo a fine episodio -> l'agente impara l'orizzonte lungo.
    # run dir timestampata, ckpt e gif vivono qui (niente sovrascrittura tra run)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.runs_root / f"M{M}_{args.split.replace('/', '_')}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}")

    # iter a cui salvare ckpt+gif (ogni save_pct%, incluso 100%)
    save_iters = sorted({int(args.iters * p / 100) - 1 for p in range(args.save_pct, 101, args.save_pct)} - {-1})

    eval_split = load_split(args.eval_split)

    def save_ckpt_and_gif(it: int, pct: int):
        ckpt_p = run_dir / f"ckpt_{pct:03d}.pth"
        torch.save({"net": net.state_dict(), "args": vars(args), "it": it, "pct": pct}, ckpt_p)
        # GIF di valutazione: 1 env eval, M agenti, policy attuale (sampling)
        gt, starts, free, _ = sample_batch(eval_split, 1)
        eenv = MarlExploreEnv(gt, free, starts,
                              EnvConfig(n_envs=1, n_agents=M, max_steps=args.eval_steps, cov_done=1.0))
        o = eenv.reset()
        net.eval()
        frames = []
        for s in range(args.eval_steps):
            with torch.no_grad(), amp():
                action, _, _, _ = net.act(o)
            o, _, _, ei = eenv.step(action)
            if s % 2 == 0:
                frames.append(render_marl(o["belief"][0], o["frontier_coarse"][0], o["fscale"],
                                          o["pos"][0], o["anchors"][0], o["anchor_mask"][0]))
        net.train()
        cov_eval = float(ei["coverage"][0])
        gif_p = run_dir / f"gif_{pct:03d}_cov{cov_eval*100:.0f}.gif"
        imageio.mimsave(gif_p, frames, fps=12)
        return cov_eval

    env = new_env()
    obs = env.reset()
    ep_cov = []          # copertura a fine episodio (metrica vera)
    cov_hist = []
    rows = []            # ultime righe metriche per la Table
    log_path = run_dir / "train.log"
    log_f = open(log_path, "w")
    log_f.write("it\tep_cov\tn_ep\tploss\tvloss\tent\tsps\n")

    progress = Progress(
        TextColumn("[bold cyan]MAPPO"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("/"),
        TimeRemainingColumn(),
        TextColumn("• [yellow]{task.fields[sps]} env-step/s"),
        TextColumn("• [green]ep_cov {task.fields[cov]}%"),
    )
    task_id = progress.add_task("train", total=args.iters, sps="-", cov="--.-")

    def make_table():
        t = Table(title=f"Last {len(rows[-12:])} iters  (M={M}, T={T}, N={N})",
                  header_style="bold magenta")
        for c in ("it", "ep_cov%", "#ep", "ploss", "vloss", "ent", "env-step/s"):
            t.add_column(c, justify="right")
        for r in rows[-12:]:
            t.add_row(*r)
        return t

    t_start = time.perf_counter()
    with Live(Group(progress, make_table()), refresh_per_second=4) as live:
        for it in range(args.iters):
            obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
            rew_buf = torch.zeros((T, N, M), device=dev)
            done_buf = torch.zeros((T, N, M), device=dev)

            for t in range(T):
                with torch.no_grad(), amp():
                    action, logp, ent, value = net.act(obs)
                obs_buf.append(slim(obs))
                act_buf.append(action); logp_buf.append(logp); val_buf.append(value.float())
                obs, reward, done, info = env.step(action)
                rew_buf[t] = reward
                done_buf[t] = done.float().unsqueeze(1).expand(-1, M)
                if bool(done.all()):
                    ep_cov.append(float(info["coverage"].mean()))
                    env = new_env()
                    obs = env.reset()

            with torch.no_grad(), amp():
                _, _, _, last_value = net.act(obs)
            last_value = last_value.float()
            values = torch.stack(val_buf)
            adv, ret = gae(rew_buf, values, done_buf, last_value, args.gamma, args.lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            big_obs = cat_obs(obs_buf)
            actions = torch.cat(act_buf, dim=0)
            old_logp = torch.cat(logp_buf, dim=0)
            adv_f = adv.reshape(T * N, M)
            ret_f = ret.reshape(T * N, M)
            ret_mean = ret_f.mean(); ret_std = ret_f.std() + 1e-8   # per value loss normalizzato
            R = T * N
            idx_all = torch.randperm(R, device=dev)
            mb = R // args.minibatches

            pl = vl = el = 0.0
            for _ in range(args.epochs):
                for s in range(0, R, mb):
                    mb_idx = idx_all[s:s + mb]
                    o = index_obs(big_obs, mb_idx)
                    with amp():
                        new_logp, ent, value = net.evaluate(o, actions[mb_idx])
                    new_logp = new_logp.float(); value = value.float(); ent = ent.float()
                    ratio = (new_logp - old_logp[mb_idx]).exp()
                    a = adv_f[mb_idx]
                    p_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                    v_loss = ((value - ret_f[mb_idx]) / ret_std).pow(2).mean()
                    e_loss = ent.mean()
                    loss = p_loss + args.vf_coef * v_loss - args.ent_coef * e_loss
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), args.max_grad)
                    opt.step()
                    pl += float(p_loss); vl += float(v_loss); el += float(e_loss)

            cov_hist.append(float(info["coverage"].mean()))
            n_upd = args.epochs * args.minibatches
            sps = (it + 1) * T * N / (time.perf_counter() - t_start)
            ep = float(np.mean(ep_cov[-20:])) if ep_cov else float("nan")
            rows.append((str(it), f"{ep*100:5.1f}" if ep_cov else "  -- ",
                         str(len(ep_cov)), f"{pl/n_upd:+.3f}",
                         f"{vl/n_upd:.3f}", f"{el/n_upd:.3f}", f"{sps:,.0f}"))
            log_f.write(f"{it}\t{ep:.4f}\t{len(ep_cov)}\t{pl/n_upd:.4f}\t"
                        f"{vl/n_upd:.4f}\t{el/n_upd:.4f}\t{sps:.0f}\n")
            log_f.flush()
            progress.update(task_id, advance=1, sps=f"{sps:,.0f}",
                            cov=f"{ep*100:5.1f}" if ep_cov else "--.-")
            live.update(Group(progress, make_table()))

            if it in save_iters:
                pct = int(round((it + 1) / args.iters * 100))
                ce = save_ckpt_and_gif(it, pct)
                rows.append(("--", f"ckpt {pct}%", "", "gif", f"cov_eval", f"{ce*100:.1f}%", "--"))
                live.update(Group(progress, make_table()))

    log_f.close()

    print(f"[run] artefatti in {run_dir}")
    if ep_cov:
        print(f"[done] ep_cov inizio {np.mean(ep_cov[:5])*100:.1f}% -> fine {np.mean(ep_cov[-5:])*100:.1f}% (#ep {len(ep_cov)})")


if __name__ == "__main__":
    main()
