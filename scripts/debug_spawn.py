"""Audit spawn behavior across N random maps. G.5 diagnostic.

For each sampled map:
  - Load via sample_batch.
  - Call Explorer._spread_starts_graph.
  - Measure pixel distance between agents.
  - Report adjacency rate.

Usage:
    PYTHONPATH=. python scripts/debug_spawn.py --n-maps 1000 --n-agents 2 --split train/easy
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--n-maps", type=int, default=1000)
    ap.add_argument("--n-agents", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    split = load_split(args.split, device=args.device)
    env = Explorer(split, EnvCfg(n_envs=1, n_agents=args.n_agents), seed=args.seed)
    NR = env.cfg.nr
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, split.n, size=args.n_maps).astype(np.int64)

    non_adj = 0
    zero_dist = 0
    weird_maps: list[tuple[int, float]] = []
    dists: list[float] = []
    for idx in indices:
        env.reload_map(env_idx=0, map_idx=int(idx))
        pos = env.pos[0].cpu().numpy()    # [M, 2]
        # Worst-case pair distance across all agent pairs.
        max_d = 0.0
        for i in range(args.n_agents):
            for j in range(i + 1, args.n_agents):
                d = float(np.linalg.norm(pos[i] - pos[j]))
                if d > max_d:
                    max_d = d
        dists.append(max_d)
        if max_d > NR * 1.5:
            non_adj += 1
            if len(weird_maps) < 10:
                weird_maps.append((int(idx), max_d))
        if max_d < 1.0:
            zero_dist += 1

    d_arr = np.array(dists)
    print(f"Audited {args.n_maps} maps ({args.split}, M={args.n_agents}):")
    print(f"  min:  {d_arr.min():.2f} px")
    print(f"  max:  {d_arr.max():.2f} px")
    print(f"  mean: {d_arr.mean():.2f} px")
    print(f"  non-adjacent (>{NR*1.5:.0f} px): {non_adj}/{args.n_maps} ({100*non_adj/args.n_maps:.2f}%)")
    print(f"  zero-distance: {zero_dist}/{args.n_maps}")
    if weird_maps:
        print("  weird examples (map_idx, max_dist):")
        for w in weird_maps:
            print(f"    {w[0]}: {w[1]:.1f} px")


if __name__ == "__main__":
    main()
