"""Detect LIMIT CYCLES in a checkpoint's greedy (argmax) rollout.

The policy is memoryless by default (gru=False) and the env is near-deterministic, so
`deterministic=True` action selection can close an absorbing loop: once an agent walks a circuit
inside already-explored space, its map stops changing, its observation becomes periodic, and its
argmax action becomes periodic — forever. Sampling escapes such a loop with probability 1, which
is why every train/* metric (computed from SAMPLED rollouts, driver.py:291) stays flat while the
deterministic eval reports success_rate 0.000 and steps_to_90 pinned at the episode cap.

This measures that directly: per map and per agent, the smallest period p <= --max-period such
that pos[t] == pos[t-p] holds over the whole trailing window, plus the number of distinct cells
visited in that window.

    python scripts/cycle_diag.py --ckpt runs/R/ckpt_020.pt --split test/complex --n-maps 8
    python scripts/cycle_diag.py --ckpt runs/R/ckpt_020.pt --split test/complex --n-maps 8 --stochastic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split
from eval.ckpt_loader import load_model_from_ckpt
from train.driver import _eval_map_idxs


def smallest_period(track: np.ndarray, max_p: int) -> int:
    """track: [W, 2] trailing positions. Returns the smallest p in 1..max_p that repeats over the
    ENTIRE window, else 0. Exact equality is the right test: positions are recomputed from the
    same float ops every lap, so a true cycle repeats bit-for-bit."""
    W = track.shape[0]
    for p in range(1, min(max_p, W // 2) + 1):
        if np.array_equal(track[p:], track[:-p]):
            return p
    return 0


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test/complex")
    ap.add_argument("--n-maps", type=int, default=8)
    ap.add_argument("--steps", type=int, default=768)
    ap.add_argument("--window", type=int, default=200, help="trailing steps examined per episode")
    ap.add_argument("--max-period", type=int, default=32)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, env_peek = load_model_from_ckpt(args.ckpt, args.device, verbose=False)
    M = int(getattr(model, "M", 2))
    split = load_split(args.split, device=args.device)
    env_cfg = EnvCfg.from_ckpt_dict(dict(env_peek or {}), n_envs=1, n_agents=M,
                                    max_episode_steps=args.steps)
    env = Explorer(split, env_cfg, seed=0)
    model.eval()
    idxs = _eval_map_idxs(env, args.n_maps)
    mode = "SAMPLED" if args.stochastic else "ARGMAX"
    print(f"[cycle_diag] {args.ckpt.parent.name}/{args.ckpt.stem}  {mode}  split={args.split} "
          f"maps={len(idxs)} window={args.window}")

    n_cyc_agents = n_agents_tot = 0
    periods: list[int] = []
    uniq_all: list[int] = []
    rows = []
    ent_q: list[list[float]] = []
    top_q: list[list[float]] = []
    duty_q: list[list[float]] = []
    ov_q: list[list[float]] = []
    cov_q: list[list[float]] = []
    for midx in idxs:
        env.reload_map(env_idx=0, map_idx=int(midx))
        h_act, h_crit = model.init_hidden(1, args.device)
        obs = env.obs
        track: list[np.ndarray] = []
        ent_ep: list[float] = []
        top_ep: list[float] = []
        duty_ep: list[float] = []
        ov_ep: list[float] = []
        cov_ep: list[float] = []
        er = 0.0
        for _t in range(args.steps):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=args.device.startswith("cuda")):
                out = model.act(obs, h_act, h_crit, deterministic=not args.stochastic)
            # Decisiveness of the SAME states both checkpoints are scored on. Comparing this
            # across saved traces instead is confounded: traces sit on different maps and stop at
            # different lengths, and late-episode states are intrinsically more ambiguous.
            ent_ep.append(float(out["entropy"].float().mean().item()))
            top_ep.append(float(out["logits"].float().softmax(-1).max(-1).values.mean().item()))
            obs, _r, done, info = env.step(out["action"])
            duty_ep.append(float(info["metrics"]["comm_duty_cycle"].item()))
            ov_ep.append(float(info["metrics"]["sensing_overlap"].item()))
            h_act, h_crit = out["hidden_actor"], out["hidden_critic"]
            track.append(env.pos[0].detach().float().cpu().numpy().copy())   # [M, 2]
            er = float(info["explored_rate"][0].item())
            cov_ep.append(er)
            if bool(done[0].item()):
                break
        arr = np.stack(track, axis=0)                       # [T, M, 2]
        W = min(args.window, arr.shape[0])
        tail = arr[-W:]
        per_agent, per_uniq = [], []
        for a in range(M):
            p = smallest_period(tail[:, a, :], args.max_period)
            u = len(np.unique(tail[:, a, :], axis=0))
            per_agent.append(p)
            per_uniq.append(u)
            n_agents_tot += 1
            if p > 0:
                n_cyc_agents += 1
                periods.append(p)
            uniq_all.append(u)
        rows.append((int(midx), arr.shape[0], er, per_agent, per_uniq))
        q = max(1, len(ent_ep) // 4)
        ent_q.append([float(np.mean(ent_ep[i * q:(i + 1) * q])) for i in range(4)])
        top_q.append([float(np.mean(top_ep[i * q:(i + 1) * q])) for i in range(4)])
        duty_q.append([float(np.mean(duty_ep[i * q:(i + 1) * q])) for i in range(4)])
        ov_q.append([float(np.mean(ov_ep[i * q:(i + 1) * q])) for i in range(4)])
        # Coverage RATE per quartile (union explored gained per step) — the thing that stalls.
        # Reading it beside duty/overlap in the same quartile says whether the agents cluster
        # BEFORE progress dies (clustering is the cause) or only after (it is the consequence of
        # having nothing left but a few shared pockets).
        cov_q.append([(cov_ep[min((i + 1) * q, len(cov_ep)) - 1]
                       - cov_ep[i * q]) / float(q) for i in range(4)])

    print(f"{'map':>6} {'steps':>6} {'expl':>6}  period/agent            uniq-cells/agent")
    for midx, T, er, pa, pu in rows:
        print(f"{midx:>6} {T:>6} {er:>6.3f}  {str(pa):<22}  {pu}")
    frac = n_cyc_agents / max(1, n_agents_tot)
    print(f"\n  agents locked in a cycle: {n_cyc_agents}/{n_agents_tot} = {frac:.3f}")
    if periods:
        print(f"  cycle period: median {int(np.median(periods))}  min {min(periods)}  max {max(periods)}")
    print(f"  distinct cells in last {args.window} steps: median {int(np.median(uniq_all))}")
    eq = np.array(ent_q).mean(axis=0)
    tq = np.array(top_q).mean(axis=0)
    print(f"  policy entropy by episode quartile: " + " ".join(f"{v:.3f}" for v in eq)
          + f"   (whole-episode mean {eq.mean():.3f})")
    print(f"  top action prob  by episode quartile: " + " ".join(f"{v:.3f}" for v in tq)
          + f"   (whole-episode mean {tq.mean():.3f})")
    for label, arr in (("comm_duty", duty_q), ("sensing_overlap", ov_q)):
        v = np.array(arr).mean(axis=0)
        print(f"  {label:<16} by episode quartile: " + " ".join(f"{x:.3f}" for x in v))
    cv = np.array(cov_q).mean(axis=0)
    print(f"  {'coverage/step':<16} by episode quartile: " + " ".join(f"{x:.5f}" for x in cv))


if __name__ == "__main__":
    main()
