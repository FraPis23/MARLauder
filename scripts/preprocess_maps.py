"""One-off preprocessing of the IR2 DungeonMaps PNGs into GPU-ready tensors.

For each split (train/easy, train/difficult, test/*):
  - normalise channels (RGBA/RGB -> grayscale L)
  - binarise  raw > 150  ->  1 = free, 0 = obstacle  (IR2's import_ground_truth convention)
  - extract the start pixel (value == 208); if absent -> (-1, -1) and the env picks a free cell
    at runtime
  - pad to a common canvas (the split's max H, max W) with obstacle (0) at the border, map
    anchored top-left

Output in  data/<split>/ :
  maps.npy   uint8  [N, Hc, Wc]   (0 = obstacle, 1 = free)   memmapped, so no PNG decode in training
  meta.npz   starts[N,2] int16, valid_shapes[N,2] int16, free_counts[N] int32,
             canvas[2] int32, files (list of source filenames)

At runtime: np.load(mmap_mode='r') + torch.from_numpy(slice).to(cuda). No numpy in the training loop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


import numpy as np
from PIL import Image
from tqdm import tqdm

from paths import DATA_ROOT, IR2_ROOT

FREE_THRESHOLD = 150  # raw > 150 => free  (cfr env.py import_ground_truth)
START_VALUE = 208     # pixel marcatore start

DEFAULT_SRC = IR2_ROOT / "DungeonMaps"
DEFAULT_OUT = DATA_ROOT
SPLITS = ["train/easy", "train/difficult", "test/complex", "test/corridor", "test/hybrid"]


def list_pngs(split_dir: Path) -> list[Path]:
    return sorted(split_dir.glob("*.png"))


def load_raw(path: Path) -> np.ndarray:
    """PNG -> array grayscale uint8 (gestisce RGBA/RGB/L)."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def canvas_size(files: list[Path]) -> tuple[int, int]:
    """Max (H, W) sullo split senza decodificare i pixel (PIL .size = (W, H))."""
    h_max = w_max = 0
    for f in files:
        with Image.open(f) as im:
            w, h = im.size
        h_max, w_max = max(h_max, h), max(w_max, w)
    return h_max, w_max


def process_split(src: Path, out: Path, split: str) -> None:
    split_dir = src / split
    files = list_pngs(split_dir)
    if not files:
        print(f"[skip] {split}: no PNG found in {split_dir}")
        return

    hc, wc = canvas_size(files)
    n = len(files)
    out_dir = out / split
    out_dir.mkdir(parents=True, exist_ok=True)

    maps = np.lib.format.open_memmap(
        out_dir / "maps.npy", mode="w+", dtype=np.uint8, shape=(n, hc, wc)
    )
    starts = np.full((n, 2), -1, dtype=np.int16)        # (row, col) o (-1,-1)
    valid_shapes = np.zeros((n, 2), dtype=np.int16)     # (h, w) nativi
    free_counts = np.zeros((n,), dtype=np.int32)        # n. celle free (denom. copertura)

    for i, f in enumerate(tqdm(files, desc=split, unit="map")):
        raw = load_raw(f)
        h, w = raw.shape
        free = (raw > FREE_THRESHOLD).astype(np.uint8)

        maps[i, :h, :w] = free                          # resto del canvas = 0 (ostacolo)
        valid_shapes[i] = (h, w)
        free_counts[i] = int(free.sum())

        ys, xs = np.nonzero(raw == START_VALUE)
        if ys.size:
            starts[i] = (int(ys[0]), int(xs[0]))

    maps.flush()
    np.savez(
        out_dir / "meta.npz",
        starts=starts,
        valid_shapes=valid_shapes,
        free_counts=free_counts,
        canvas=np.array([hc, wc], dtype=np.int32),
        files=np.array([f.name for f in files]),
    )
    with_start = int((starts[:, 0] >= 0).sum())
    print(f"[ok] {split}: {n} maps | canvas {hc}x{wc} | start found {with_start}/{n} "
          f"| maps.npy ~{maps.nbytes/1e9:.2f} GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    args = ap.parse_args()

    for split in args.splits:
        process_split(args.src, args.out, split)


if __name__ == "__main__":
    main()
