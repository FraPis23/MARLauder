"""Random-policy baseline on a fixed map: explored rate vs MAPPO eval.

    python scripts/baseline_random.py --split test/complex --map-idx 0 --steps 96 --episodes 8
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
from env.maps import load_split, sample_batch


def reset_to_map(env: Explorer, split, map_idx: int, device: str) -> None:
    gt, starts, fc = sample_batch(split, 1, indices=np.array([map_idx]), seed=0, device=device)
    for i in range(env.N):
        env.world.gt_torch[i] = gt[0]
        env.world.occupancy_torch[i] = 0
        env.world.occupancy_logodds_torch[i] = 0.0
        env.starts[i] = starts[0]
        env.free_total[i] = float(fc[0])
    env.visited_step.fill_(-1)
    env.t.zero_()
    env.pos[:, :, 0] = float(starts[0, 1])
    env.pos[:, :, 1] = float(starts[0, 0])
    env.world.set_positions(env.pos)
    env.world.scan()
    env.last_union.copy_((env.world.occupancy_torch == 1).view(env.N, -1).float().sum(dim=-1))
    env._refresh_obs()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test/complex")
    ap.add_argument("--map-idx", type=int, default=0)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--nr", type=int, default=16)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    split = load_split(args.split, device=args.device)
    cfg = EnvCfg(n_envs=args.episodes, n_agents=1, nr=args.nr, max_episode_steps=args.steps + 1)
    env = Explorer(split, cfg, seed=args.seed)
    reset_to_map(env, split, args.map_idx, args.device)
    rng = np.random.default_rng(args.seed)
    explored_traj = []
    obs = env.obs
    for t in range(args.steps):
        mask = obs["action_mask"][:, 0].cpu().numpy()
        actions = np.zeros((args.episodes,), dtype=np.int64)
        for e in range(args.episodes):
            valid = np.where(mask[e])[0]
            actions[e] = int(rng.choice(valid)) if len(valid) > 0 else 0
        a_t = torch.from_numpy(actions).to(args.device).view(args.episodes, 1)
        obs, _, _, info = env.step(a_t)
        explored_traj.append(info["explored_rate"].cpu().numpy())
    explored_traj = np.stack(explored_traj, axis=0)        # [T, N]
    final = explored_traj[-1]
    print(f"[random] split={args.split} map={args.map_idx} steps={args.steps} episodes={args.episodes}")
    print(f"  final explored: mean={final.mean()*100:.2f}%  std={final.std()*100:.2f}%  "
          f"min={final.min()*100:.2f}%  max={final.max()*100:.2f}%")


if __name__ == "__main__":
    main()
