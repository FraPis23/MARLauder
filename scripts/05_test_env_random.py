"""Step 5 test: random-policy rollout. Saves a GIF per env.

Uses the probabilistic occupancy renderer (walls = dark navy, free = muted gray,
confidence grows with the number of LiDAR observations).

    python scripts/05_test_env_random.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import imageio.v2 as imageio
import numpy as np
import torch

from env.explorer import EnvCfg, Explorer
from env.frontier import compute_frontier
from env.maps import load_split
from eval.render import composite_frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/step_05"))
    args = ap.parse_args()

    split = load_split(args.split, device=args.device)
    cfg = EnvCfg(n_envs=args.n_envs, n_agents=1, max_episode_steps=args.steps + 1)
    env = Explorer(split, cfg, seed=args.seed)
    obs = env.obs
    print(f"[env] obs keys={list(obs.keys())}")
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:15s} {tuple(v.shape)} {v.dtype}")

    frames = [[] for _ in range(args.n_envs)]
    trails = [[] for _ in range(args.n_envs)]
    rng = np.random.default_rng(args.seed)
    for t in range(args.steps):
        mask = obs["action_mask"][:, 0].cpu().numpy()
        actions = np.zeros((args.n_envs,), dtype=np.int64)
        for e in range(args.n_envs):
            valid = np.where(mask[e])[0]
            actions[e] = int(rng.choice(valid)) if len(valid) > 0 else 0
        a_t = torch.from_numpy(actions).to(args.device).view(args.n_envs, 1)
        obs, reward, done, info = env.step(a_t)
        explored = info["explored_rate"].cpu().numpy()
        prob_np = env.world.occupancy_prob().cpu().numpy()
        gt_np = env.world.gt_torch.cpu().numpy()
        frontier_np = compute_frontier(env.world.occupancy_torch).cpu().numpy()
        nxy = obs["node_xy"][:, 0].cpu().numpy()
        nv = obs["node_valid"][:, 0].cpu().numpy()
        util = obs["utility"][:, 0].cpu().numpy()
        curr = obs["curr_idx"][:, 0].cpu().numpy()
        target = obs["guidepost_target"][:, 0].cpu().numpy()
        path_xy = obs["guidepost_path_xy"][:, 0].cpu().numpy()
        path_v = obs["guidepost_path_valid"][:, 0].cpu().numpy()
        pos = env.pos[:, 0].cpu().numpy()
        for e in range(args.n_envs):
            trails[e].append((float(pos[e, 0]), float(pos[e, 1])))
            tgt_flat = int(target[e])
            tgt_xy = (float(nxy[e, tgt_flat, 0]), float(nxy[e, tgt_flat, 1])) if tgt_flat != int(curr[e]) else None
            im = composite_frame(
                prob=prob_np[e], gt=gt_np[e], frontier=frontier_np[e],
                nxy=nxy[e], nv=nv[e], util=util[e], curr=int(curr[e]),
                agent_xy=(float(pos[e, 0]), float(pos[e, 1])),
                trail=trails[e][-32:], step=int(env.t[e]), explored=float(explored[e]),
                path_xy=path_xy[e], path_valid=path_v[e],
                target_xy=tgt_xy,
            )
            frames[e].append(np.array(im))
        if t % 16 == 0:
            r = reward[:, 0].cpu().numpy()
            print(f"[t={t:3d}] reward={r.tolist()}  explored={explored.tolist()}")

    args.out.mkdir(parents=True, exist_ok=True)
    for e in range(args.n_envs):
        path = args.out / f"random_env{e}.gif"
        imageio.mimsave(path, frames[e], duration=80, loop=0)
        print(f"[save] {path}  frames={len(frames[e])}")


if __name__ == "__main__":
    main()
