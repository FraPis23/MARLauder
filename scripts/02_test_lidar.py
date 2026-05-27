"""Step 2: Warp LiDAR + probabilistic occupancy on a real map.

Walls observed by LiDAR appear in dark navy. Unknown is near black. Free is muted
gray. Frontier red is *only* used for the frontier overlay (Step 3+) — walls are
NOT red here.

    python scripts/02_test_lidar.py --split train/easy --map-idx 0
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

from env.maps import load_split, sample_batch
from env.world_warp import WarpWorld
from eval.render import shade_occupancy_prob, overlay_gt_hint, paint_agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--map-idx", type=int, default=0)
    ap.add_argument("--sensor-range", type=float, default=60.0)
    ap.add_argument("--n-rays", type=int, default=720)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/step_02/occupancy.png"))
    args = ap.parse_args()

    split = load_split(args.split, device=args.device)
    gt, starts, _ = sample_batch(split, 1, indices=np.array([args.map_idx]), seed=0, device=args.device)
    world = WarpWorld(gt, n_agents=1, sensor_range=args.sensor_range, n_rays=args.n_rays, device=args.device)
    sr, sc = int(starts[0, 0]), int(starts[0, 1])
    pos_xy = torch.tensor([[sc, sr]], dtype=torch.float32, device=args.device)
    world.set_positions(pos_xy)
    world.scan()
    prob = world.occupancy_prob()
    cat = world.occupancy_torch
    print(f"[scan] cat unique={torch.unique(cat).tolist()}")
    print(f"[prob] min={prob.min().item():.3f}  max={prob.max().item():.3f}  mean={prob.mean().item():.3f}")

    prob_np = prob[0].cpu().numpy()
    gt_np = gt[0].cpu().numpy()
    rgb = shade_occupancy_prob(prob_np)
    rgb = overlay_gt_hint(rgb, gt_np, prob_np)
    im = Image.fromarray(rgb)
    paint_agent(im, (sc, sr))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    im.save(args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
