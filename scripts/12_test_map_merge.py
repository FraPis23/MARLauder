"""Check the map merge on rendezvous: after comm, both agents' maps must show the UNION.

Compares, per agent and per step, the number of known cells in the rendered source (log-odds, what the
GIF paints) against the number in the occupancy grid (what the graph/node_valid is built from). If the
merge only lands in one of the two, the render and the graph disagree — the symptom being "when the
agents meet the graph appears but the map does not".

    python scripts/12_test_map_merge.py --split train/difficult --map-idx 50 --steps 220
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

import numpy as np

from env.explorer import EnvCfg, Explorer
from env.frontier import compute_frontier
from env.maps import load_split
from eval.render import shade_occupancy_prob, paint_frontier

_UNKNOWN, _FREE, _OBST = 0, 1, 2
_LO_FREE_TH, _LO_OCC_TH = 0.5, -0.5      # same thresholds world_warp uses to derive occupancy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/difficult")
    ap.add_argument("--map-idx", type=int, default=50)
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    split = load_split(args.split, device=args.device)
    cfg = EnvCfg(n_agents=2, n_hops=6, comm_model="signal_strength", comm_range_px=120.0, sensor_range_px=80.0)
    env = Explorer(split, cfg, seed=args.map_idx)
    env.reload_map(env_idx=0, map_idx=args.map_idx)
    env.cfg.done_explored_thresh = 2.0
    env.cfg.max_episode_steps = args.steps + 5

    obs = env.obs
    bad_steps, merges = [], []
    prev_known = [0] * env.M
    for t in range(args.steps):
        act = torch.zeros((env.N, env.M), dtype=torch.long, device=env.dev)
        valid = obs["action_mask"][0].bool()
        for a in range(env.M):
            ok = valid[a].nonzero()
            if ok.numel():
                act[0, a] = int(ok[torch.randint(0, ok.numel(), (1,))])
        obs, *_ = env.step(act)

        lo = env.world.occupancy_logodds_torch[0]                 # [M, H, W] — what the GIF renders
        occ = env.world.occupancy_torch[0]                        # [M, H, W] — what the graph uses
        for a in range(env.M):
            known_lo = int(((lo[a] > _LO_FREE_TH) | (lo[a] < _LO_OCC_TH)).sum())
            known_occ = int((occ[a] != _UNKNOWN).sum())
            if known_lo != known_occ:
                bad_steps.append((t, a, known_lo, known_occ))
            jump = known_occ - prev_known[a]
            if jump > 2000:                                       # a merge, not ordinary sensing
                merges.append((t, a, jump, known_lo - known_occ))
            prev_known[a] = known_occ

    print(f"steps={args.steps} agents={env.M}")
    print(f"log-odds vs occupancy disagreement: {len(bad_steps)} step-agent(s)")
    if bad_steps:
        for b in bad_steps[:5]:
            print(f"   t={b[0]} agent{b[1]}: known(log-odds)={b[2]} known(occupancy)={b[3]}  Δ={b[2]-b[3]}")
    print(f"merge events (known cells jumping >2000 in one step): {len(merges)}")
    for m in merges[:8]:
        print(f"   t={m[0]} agent{m[1]}: +{m[2]} cells, render/graph gap={m[3]}")
    # Pixel-level check on the RENDERED frame: every cell the occupancy grid calls FREE must come out
    # bright in the image the GIF is built from, and every frontier cell must sit on rendered-known ground.
    lo = env.world.occupancy_logodds_torch[0]
    occ = env.world.occupancy_torch[0]
    for a in range(env.M):
        prob = torch.sigmoid(lo[a]).cpu().numpy()
        fr = compute_frontier(occ[a:a + 1])[0].cpu().numpy()
        img = paint_frontier(shade_occupancy_prob(prob), fr)
        free = (occ[a] == _FREE).cpu().numpy()
        dark = img.sum(axis=2) <= 90
        free_dark = int((free & dark).sum())
        fr_dark = int((fr.astype(bool) & dark).sum())
        print(f"agent{a}: FREE cells rendered dark = {free_dark} | frontier cells on dark ground = {fr_dark}")
        if free_dark or fr_dark:
            ys, xs = np.nonzero(free & dark) if free_dark else np.nonzero(fr.astype(bool) & dark)
            print(f"   e.g. around (x={int(xs[0])}, y={int(ys[0])}) prob={float(prob[ys[0], xs[0]]):.3f} "
                  f"lo={float(lo[a][ys[0], xs[0]]):.2f}")
    print("RESULT:", "FAIL — render and graph see different maps" if bad_steps else "OK — render and graph agree")


if __name__ == "__main__":
    main()
