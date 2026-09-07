#!/usr/bin/env python3
"""Is the own-99% objective REACHABLE for this checkpoint, on this split, with this budget?

The question every `--done-mode own` run lives or dies on. v12 answered it by accident and too
late: the completion bonus never fired in 4M steps, so the objective it was nominally optimising
contributed exactly zero gradient, and nothing in the training log said so.

Run this BEFORE committing GPU hours — especially before a from-scratch run, where the honest
worry is that an early, bad policy never terminates and therefore never learns that terminating
is the point. Pointing it at an EARLY checkpoint (ckpt_020) answers that directly: if a policy
that weak already finishes some maps, a from-scratch run has a gradient from the start.

    python tools/probe_reachability.py --ckpt runs/<run>/ckpt_020.pt \
        --split train/easy --max-travel-frac 0.06 --n-envs 32

Deliberately NOT a comparison tool: no CSV, no IR2 columns. One number that matters (termination
rate) plus the distribution of what the runs actually spent.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from env.explorer import EnvCfg, Explorer  # noqa: E402
from env.maps import load_split  # noqa: E402
from eval.ckpt_loader import load_model_from_ckpt  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--n-agents", type=int, default=2)
    ap.add_argument("--max-travel-frac", type=float, default=0.06,
                    help="per-map travel budget, px per GT-free-px (0 = off)")
    ap.add_argument("--max-travel-px", type=float, default=0.0)
    ap.add_argument("--max-episode-steps", type=int, default=4000,
                    help="step safety net; must NOT bind, or the travel budget is not what is "
                         "being measured")
    ap.add_argument("--deterministic", action="store_true",
                    help="argmax actions (default: sample, which is what training actually does "
                         "and therefore what decides whether the bonus fires during training)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    split = load_split(args.split, device=args.device)
    peek = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    penv = (peek.get("cfg", {}) or {}).get("env", {}) or {}
    cfg = EnvCfg.from_ckpt_dict(
        penv, n_envs=args.n_envs, n_agents=args.n_agents,
        max_episode_steps=args.max_episode_steps,
        max_travel_px=args.max_travel_px, max_travel_frac=args.max_travel_frac,
        done_mode="own",                       # the whole point of the probe
    )
    env = Explorer(split, cfg, seed=0)
    model, _ = load_model_from_ckpt(args.ckpt, args.device, n_agents=args.n_agents, verbose=True)
    model.eval()

    K = args.n_envs
    h_act, h_crit = model.init_hidden(K, args.device)
    obs = env.obs
    active = torch.ones(K, dtype=torch.bool, device=args.device)
    travel = torch.zeros(K, args.n_agents, device=args.device)
    steps = torch.zeros(K, device=args.device)
    term = torch.zeros(K, device=args.device)
    own_min_final = torch.zeros(K, device=args.device)

    for _t in range(args.max_episode_steps):
        pos_prev = env.pos.clone()
        out = model.act(obs, h_act, h_crit, deterministic=args.deterministic)
        obs, _r, done, info = env.step(out["action"])
        h_act, h_crit = out["hidden_actor"], out["hidden_critic"]
        a = active.float()
        travel += (env.pos - pos_prev).norm(dim=-1) * a.unsqueeze(-1)
        steps += a
        newly = active & done
        term = torch.where(newly, info["terminated"].float(), term)
        own_min_final = torch.where(newly, info["own_cov"].min(dim=-1).values, own_min_final)
        active = active & ~done
        if not bool(active.any().item()):
            break
    # Envs still running at the step safety net never got a verdict — count them as failures and
    # say so, rather than silently dropping them.
    if bool(active.any().item()):
        own_min_final = torch.where(active, env._own_cov.min(dim=-1).values, own_min_final) \
            if hasattr(env, "_own_cov") else own_min_final
        print(f"[warn] {int(active.sum())} env(s) hit the {args.max_episode_steps}-step safety net "
              f"— raise --max-episode-steps, the travel budget was not the binding constraint")

    md = travel.max(dim=-1).values.tolist()
    om = own_min_final.tolist()
    rate = float(term.mean())
    print(f"\nckpt={args.ckpt}\nsplit={args.split}  n_envs={K}  M={args.n_agents}  "
          f"frac={args.max_travel_frac}  px={args.max_travel_px}  "
          f"actions={'argmax' if args.deterministic else 'sampled'}")
    print(f"\n  TERMINATION RATE (own-99% by every robot) : {rate:.2f}   "
          f"({int(term.sum())}/{K} episodes)")
    print(f"  weakest robot's own coverage at end       : "
          f"mean {st.mean(om):.3f}  min {min(om):.3f}  max {max(om):.3f}")
    print(f"  distance spent (max over robots, px)      : "
          f"mean {st.mean(md):.0f}  p50 {st.median(md):.0f}  max {max(md):.0f}")
    print(f"  steps                                     : mean {st.mean(steps.tolist()):.0f}")
    verdict = ("REACHABLE — the completion bonus fires, so a run starting from a policy this weak "
               "has a gradient toward the objective from iteration 1."
               if rate >= 0.15 else
               "NOT REACHABLE at this budget — the bonus is dead weight and --done-mode own would "
               "only lengthen episodes. Raise the budget, use an easier split first, or warm-start.")
    print(f"\n  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
