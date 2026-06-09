"""Preprocessing una-tantum delle DungeonMaps IR2 -> tensori GPU-ready.

Per ogni split (train/easy, train/difficult, test/*):
  - normalizza canali (RGBA/RGB -> grayscale L)
  - binarizza  raw > 150  ->  1 = free, 0 = ostacolo  (convenzione IR2 import_ground_truth)
  - estrae lo start (pixel == 208); se assente -> (-1,-1), l'env scegliera una cella free a runtime
  - pad a canvas comune (max H, max W dello split) col bordo = ostacolo (0), mappa ancorata in alto-sx

Output in  MARLauder/data/<split>/ :
  maps.npy        uint8  [N, Hc, Wc]   (0=ostacolo, 1=free)   -> memmap, niente PNG-decode nel training
  meta.npz        starts[N,2] int16, valid_shapes[N,2] int16, free_counts[N] int32,
                  canvas[2] int32, files (lista nomi)

A runtime: np.load(mmap_mode='r') + torch.from_numpy(slice).to(cuda). Nessun numpy nel loop di training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

FREE_THRESHOLD = 150  # raw > 150 => free  (cfr env.py import_ground_truth)
START_VALUE = 208     # pixel marcatore start

DEFAULT_SRC = Path("/workspace/IR2-Multi-Robot-RL-Exploration/DungeonMaps")
DEFAULT_OUT = Path("/workspace/MARLauder/data")
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
        print(f"[skip] {split}: nessun png in {split_dir}")
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
    print(f"[ok] {split}: {n} mappe | canvas {hc}x{wc} | start trovato {with_start}/{n} "
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
