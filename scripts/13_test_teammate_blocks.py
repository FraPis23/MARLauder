"""A known teammate's cell must be masked out of the action space, like a wall.

Checks, over a rollout: (a) whenever comm holds, no legal action points at the node the teammate stands
on; (b) no agent is ever left with zero legal actions by that masking; (c) two agents never end a step on
the same node while in contact.

    python scripts/13_test_teammate_blocks.py --split test/complex --map-idx 0 --steps 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test/complex")
    ap.add_argument("--map-idx", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    split = load_split(args.split, device=args.device)
    cfg = EnvCfg(n_agents=2, n_hops=6, comm_model="signal_strength", comm_range_px=120.0)
    env = Explorer(split, cfg, seed=args.map_idx)
    env.reload_map(env_idx=0, map_idx=args.map_idx)
    env.cfg.done_explored_thresh = 2.0
    env.cfg.max_episode_steps = args.steps + 5

    obs = env.obs
    leaks = stranded = collisions = comm_steps = 0
    for _ in range(args.steps):
        mask = obs["action_mask"][0]                       # [M, K]
        nbr_g = env._last_obs["curr_nbr_global"].view(env.N, env.M, -1)[0]      # [M, K] global node ids
        curr_g = env.curr_idx_global[0]                                          # [M]
        comm = env._last_obs.get("comm_mask")
        comm = comm[0] if comm is not None else torch.eye(env.M, dtype=torch.bool, device=env.dev)

        for i in range(env.M):
            if not mask[i].any():
                stranded += 1
            for j in range(env.M):
                if i == j or not bool(comm[i, j]):
                    continue
                comm_steps += 1
                if bool((mask[i] & (nbr_g[i] == curr_g[j])).any()):
                    leaks += 1

        act = torch.zeros((env.N, env.M), dtype=torch.long, device=env.dev)
        for i in range(env.M):
            ok = mask[i].nonzero()
            if ok.numel():
                act[0, i] = int(ok[torch.randint(0, ok.numel(), (1,))])
        obs, *_ = env.step(act)
        if int(env.curr_idx_global[0, 0]) == int(env.curr_idx_global[0, 1]) and bool(comm[0, 1]):
            collisions += 1

    print(f"steps={args.steps}  step-pairs in contact={comm_steps}")
    print(f"legal actions pointing at a known teammate's cell : {leaks}")
    print(f"agents left with no legal action                  : {stranded}")
    print(f"both agents on the same node while in contact     : {collisions}")
    print("RESULT:", "OK" if (leaks == 0 and stranded == 0) else "FAIL")


if __name__ == "__main__":
    main()
