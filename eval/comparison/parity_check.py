"""Parity check: MARLauder .npy pack vs IR2 DungeonMaps PNGs.

Gate for the MARLauder-vs-IR2 comparison: both systems must see the SAME ground truth.
For each split, samples a few maps from the pack, finds the original PNG by filename
(meta.npz `files`), converts it with IR2's exact rule (grayscale > 150 = free, pixel 208
= start marker) and asserts:
  - identical free/obstacle mask on the valid region (pack is padded with obstacle);
  - pack start inside the PNG's 208-marker blob (MARLauder takes the first 208 pixel,
    IR2 env.py takes the 128th — both lie inside the same blob).

Usage (inside the marlauder container):
    python eval/comparison/parity_check.py [--n-per-split 3]
Exit code 0 = parity OK; non-zero = mismatch (comparison must NOT proceed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

MARL_DATA = Path("/workspace/MARLauder/data")
IR2_MAPS = Path("/workspace/IR2-Multi-Robot-RL-Exploration/DungeonMaps")

SPLITS = ["test/complex", "test/corridor", "test/hybrid", "train/easy", "train/difficult"]
FREE_THRESHOLD = 150
START_VALUE = 208


def png_to_gt(png_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """IR2 env.py import_ground_truth rule → (free mask uint8 [H,W], 208-blob mask)."""
    raw = imageio.imread(png_path)
    if raw.ndim == 3:                      # some PNGs may carry channels; IR2 reads as_gray
        raw = raw[..., 0]
    raw = raw.astype(int)
    if np.all(raw == 0):
        raw = raw * 255
    free = (raw > FREE_THRESHOLD).astype(np.uint8)
    start_blob = raw == START_VALUE
    return free, start_blob


def check_split(split: str, n_samples: int) -> list[str]:
    errors: list[str] = []
    sd = MARL_DATA / split
    maps = np.load(sd / "maps.npy", mmap_mode="r")
    meta = np.load(sd / "meta.npz")
    files, shapes, starts = meta["files"], meta["valid_shapes"], meta["starts"]
    n = maps.shape[0]
    idxs = sorted({0, n // 2, n - 1} | set(
        np.linspace(0, n - 1, n_samples, dtype=int).tolist()))
    for i in idxs:
        fname = str(files[i])
        png = IR2_MAPS / split / fname
        if not png.exists():
            errors.append(f"{split}[{i}] PNG mancante: {png}")
            continue
        free_png, blob = png_to_gt(png)
        h, w = int(shapes[i][0]), int(shapes[i][1])
        if free_png.shape != (h, w):
            errors.append(f"{split}[{i}] {fname}: shape PNG {free_png.shape} != valid_shape ({h},{w})")
            continue
        pack_region = np.asarray(maps[i, :h, :w])
        # NOTE: the 208 start blob is > 150, so it is FREE in both conversions.
        if not np.array_equal(pack_region, free_png):
            diff = int((pack_region != free_png).sum())
            errors.append(f"{split}[{i}] {fname}: mask mismatch su {diff} px")
        # padding beyond valid region must be obstacle
        if maps[i, h:, :].any() or maps[i, :, w:].any():
            errors.append(f"{split}[{i}] {fname}: padding non-ostacolo")
        # start: pack start (row,col) inside the PNG 208 blob
        r, c = int(starts[i][0]), int(starts[i][1])
        if r >= 0:
            if not blob.any():
                errors.append(f"{split}[{i}] {fname}: pack ha start ma PNG senza pixel 208")
            elif not blob[r, c]:
                errors.append(f"{split}[{i}] {fname}: start pack ({r},{c}) fuori dal blob 208")
        elif blob.any():
            errors.append(f"{split}[{i}] {fname}: PNG ha blob 208 ma pack start=(-1,-1)")
    print(f"[{split}] {len(idxs)} mappe controllate: " + ("OK" if not errors else f"{len(errors)} ERRORI"))
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-split", type=int, default=3)
    args = ap.parse_args()
    all_err: list[str] = []
    for split in SPLITS:
        if not (MARL_DATA / split / "maps.npy").exists():
            print(f"[{split}] SKIP (pack assente)")
            continue
        all_err += check_split(split, args.n_per_split)
    if all_err:
        print("\nPARITY FAIL:")
        for e in all_err:
            print(" -", e)
        sys.exit(1)
    print("\nPARITY OK — dataset identici, comparison può procedere.")


if __name__ == "__main__":
    main()
