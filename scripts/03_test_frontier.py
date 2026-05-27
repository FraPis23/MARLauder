"""Step 3: frontier detection on top of probabilistic occupancy.

Frontier (red) ONLY appears on cells with categorical occupancy == FREE that
border UNKNOWN. Walls are dark navy and visually distinct.

    python scripts/03_test_frontier.py
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
from PIL import Image

from env.frontier import compute_frontier
from env.maps import load_split, sample_batch
from env.world_warp import WarpWorld
from eval.render import overlay_gt_hint, paint_agent, paint_frontier, shade_occupancy_prob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--map-idx", type=int, default=0)
    ap.add_argument("--sensor-range", type=float, default=60.0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/step_03/frontier.png"))
    args = ap.parse_args()

    split = load_split(args.split, device=args.device)
    gt, starts, _ = sample_batch(split, 1, indices=np.array([args.map_idx]), seed=0, device=args.device)
    world = WarpWorld(gt, n_agents=1, sensor_range=args.sensor_range, device=args.device)
    sr, sc = int(starts[0, 0]), int(starts[0, 1])
    pos_xy = torch.tensor([[sc, sr]], dtype=torch.float32, device=args.device)
    world.set_positions(pos_xy)
    world.scan()
    occupancy = world.occupancy_torch
    prob = world.occupancy_prob()
    frontier = compute_frontier(occupancy)

    # Verify: NO wall cell should be flagged frontier.
    overlap = int((frontier & (occupancy == 2)).sum().item())
    print(f"[check] frontier ∩ wall-occupancy = {overlap}  (must be 0)")
    print(f"[frontier] count={int(frontier.sum().item())}")

    rgb = shade_occupancy_prob(prob[0].cpu().numpy())
    rgb = overlay_gt_hint(rgb, gt[0].cpu().numpy(), prob[0].cpu().numpy())
    rgb = paint_frontier(rgb, frontier[0].cpu().numpy())
    im = Image.fromarray(rgb)
    paint_agent(im, (sc, sr))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    im.save(args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
