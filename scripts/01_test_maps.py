"""Step 1 test: load a split and save a 2×2 mosaic with start markers.

Run inside container:
    python scripts/01_test_maps.py --split train/easy
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
from PIL import Image, ImageDraw

from env.maps import FREE, OBSTACLE, load_split, sample_batch

C_FREE = (224, 226, 230)
C_OBST = (74, 80, 92)
C_START = (80, 130, 250)
C_PAD = (26, 26, 32)


def render(gt: torch.Tensor, start: torch.Tensor) -> Image.Image:
    H, W = gt.shape
    img = np.full((H, W, 3), C_PAD, dtype=np.uint8)
    gt_np = gt.cpu().numpy()
    img[gt_np == FREE] = C_FREE
    img[gt_np == OBSTACLE] = C_OBST
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    sr, sc = int(start[0]), int(start[1])
    r = 6
    dr.ellipse([sc - r, sr - r, sc + r, sr + r], fill=C_START, outline=(255, 255, 255), width=2)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/step_01/maps.png"))
    args = ap.parse_args()

    split = load_split(args.split, device=args.device)
    print(f"[split] {split.name}  n={split.n}  canvas={split.canvas}")
    gt, starts, fc = sample_batch(split, args.n, seed=args.seed)
    print(f"[batch] gt={tuple(gt.shape)} {gt.dtype} dev={gt.device}")
    print(f"[batch] starts={starts.tolist()}  free_counts={fc.tolist()}")

    # mosaic 2 wide
    cols = 2
    rows = (args.n + cols - 1) // cols
    H, W = split.canvas
    mosaic = Image.new("RGB", (cols * W, rows * H), color=C_PAD)
    for i in range(args.n):
        im = render(gt[i], starts[i])
        mosaic.paste(im, ((i % cols) * W, (i // cols) * H))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
