"""Step 7: tiny MAPPO smoke test. 2 envs × 32 rollout × 2 updates × 1 epoch.

    python scripts/07_smoke_mappo.py
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
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/step_07"))
    args = ap.parse_args()

    cfg = TrainCfg(
        split="train/easy",
        out_dir=args.out,
        total_steps=2 * 2 * 32,        # 2 updates × 2 envs × 32 steps
        n_envs=2,
        n_agents=1,
        rollout_len=32,
        d_hidden=64,
        n_heads=4,
        n_layers=2,
        lr_actor=3e-4,
        lr_critic=1e-3,
        device=args.device,
        env=EnvCfg(n_envs=2, n_agents=1, max_episode_steps=64),
        ppo=MAPPOCfg(k_epochs=1, tbptt_steps=8, use_amp=True),
    )
    train(cfg, log_every=1)


if __name__ == "__main__":
    main()
