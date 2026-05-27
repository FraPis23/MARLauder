"""MAPPO training driver.

    python scripts/run_train.py --n-envs 128 --total-steps 5_000_000 --out runs/run_001
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from env.explorer import EnvCfg
from train.driver import TrainCfg, train
from train.mappo import MAPPOCfg


def main() -> None:
    ap = argparse.ArgumentParser()
    # --- what to train on ---
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/run_default"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    # --- scale ---
    ap.add_argument("--total-steps", type=int, default=5_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-agents", type=int, default=1,
                    help="Number of cooperative agents per env")
    ap.add_argument("--comm-range", type=float, default=120.0,
                    help="Communication range in pixels (0 = agents never communicate)")
    ap.add_argument("--rollout-len", type=int, default=128)
    ap.add_argument("--max-episode-steps", type=int, default=512)
    ap.add_argument("--minibatches", type=int, default=1,
                    help="PPO minibatches per epoch (must divide n-envs)")
    # --- learning ---
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    # --- flags ---
    ap.add_argument("--compile", action="store_true", help="torch.compile encoder (CUDA only)")
    ap.add_argument("--eval-on-ckpt", action="store_true",
                    help="Emit 2 eval GIFs at each milestone (25/50/75/100%%)")
    args = ap.parse_args()

    cfg = TrainCfg(
        split=args.split,
        out_dir=args.out,
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        n_agents=args.n_agents,
        rollout_len=args.rollout_len,
        device=args.device,
        seed=args.seed,
        compile=args.compile,
        eval_on_ckpt=args.eval_on_ckpt,
        eval_split=args.split,          # eval on same split as train
        env=EnvCfg(
            n_envs=args.n_envs,
            n_agents=args.n_agents,
            nr=16,                              # lattice spacing — 16px → N_max≈1200 nodes
            max_episode_steps=args.max_episode_steps,
            comm_range_px=args.comm_range,
        ),
        ppo=MAPPOCfg(
            ent_coef=args.ent_coef,
            n_minibatches=args.minibatches,
        ),
    )
    train(cfg, log_every=1)


if __name__ == "__main__":
    main()
