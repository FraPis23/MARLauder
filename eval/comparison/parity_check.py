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
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import imageio.v2 as imageio

from paths import DATA_ROOT, IR2_ROOT

MARL_DATA = DATA_ROOT
IR2_MAPS = IR2_ROOT / "DungeonMaps"
IR2_COMPARISON_MAPS = IR2_ROOT / "comparison"
_COMPARISON_DIR = Path(__file__).resolve().parent

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
            errors.append(f"{split}[{i}] PNG missing: {png}")
            continue
        free_png, blob = png_to_gt(png)
        h, w = int(shapes[i][0]), int(shapes[i][1])
        if free_png.shape != (h, w):
            errors.append(f"{split}[{i}] {fname}: PNG shape {free_png.shape} != valid_shape ({h},{w})")
            continue
        pack_region = np.asarray(maps[i, :h, :w])
        # NOTE: the 208 start blob is > 150, so it is FREE in both conversions.
        if not np.array_equal(pack_region, free_png):
            diff = int((pack_region != free_png).sum())
            errors.append(f"{split}[{i}] {fname}: mask mismatch on {diff} px")
        # padding beyond valid region must be obstacle
        if maps[i, h:, :].any() or maps[i, :, w:].any():
            errors.append(f"{split}[{i}] {fname}: padding is not obstacle")
        # start: pack start (row,col) inside the PNG 208 blob
        r, c = int(starts[i][0]), int(starts[i][1])
        if r >= 0:
            if not blob.any():
                errors.append(f"{split}[{i}] {fname}: pack has a start but the PNG has no 208 pixel")
            elif not blob[r, c]:
                errors.append(f"{split}[{i}] {fname}: pack start ({r},{c}) is outside the 208 blob")
        elif blob.any():
            errors.append(f"{split}[{i}] {fname}: PNG has a 208 blob but pack start=(-1,-1)")
    print(f"[{split}] {len(idxs)} maps checked: " + ("OK" if not errors else f"{len(errors)} ERRORS"))
    return errors


def check_order(split: str) -> list[str]:
    """map_indices_{split}.json must list maps in the order IR2 actually runs them.

    THE GATE THIS FILE ORIGINALLY LACKED. Verifying that the two systems see the
    same MAPS is not enough — the comparison is PAIRED, so it also needs row i of the two CSVs to
    be the same map, and that depends on the ORDER of IR2's map list:

        self.map_list = os.listdir(self.map_dir)
        self.map_list.sort(reverse=True)            # IR2 env.py:34-35
        self.file_path = self.map_list[map_index]   # map_index == episode

    The json was ASCENDING while IR2 iterates DESCENDING, so episode 0 was 99.png on their side and
    1.png on ours. Per-cell means survived that (same 100 maps, different order); every paired
    statistic did not. See eval/comparison/PROTOCOL.md, section "Map order".
    """
    idx_file = _COMPARISON_DIR / f"map_indices_{split.split('/')[-1]}.json"
    map_dir = IR2_COMPARISON_MAPS / f"maps_{split.split('/')[-1]}"
    if not idx_file.exists() or not map_dir.exists():
        return []
    entries = json.loads(idx_file.read_text())["entries"]
    want = sorted(os.listdir(map_dir), reverse=True)          # exactly IR2 env.py:34-35
    got = [e["file"] for e in entries]
    if got == want:
        print(f"[{split}] map order matches IR2 (descending): OK")
        return []
    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), 0)
    hint = ""
    if got == sorted(want):
        hint = ("  <-- the json is in ASCENDING order; IR2 uses sort(reverse=True). "
                "This is exactly the pairing bug this check exists to catch.")
    return [f"{split}: map order DIFFERS from IR2. First difference at eps {first}: "
            f"json={got[first]} vs IR2={want[first]}.{hint}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-split", type=int, default=3)
    args = ap.parse_args()
    all_err: list[str] = []
    for split in ("test/hybrid", "test/corridor", "test/complex"):
        all_err += check_order(split)
    for split in SPLITS:
        if not (MARL_DATA / split / "maps.npy").exists():
            print(f"[{split}] SKIP (pack absent)")
            continue
        all_err += check_split(split, args.n_per_split)
    if all_err:
        print("\nPARITY FAIL:")
        for e in all_err:
            print(" -", e)
        sys.exit(1)
    print("\nPARITY OK — datasets are identical, the comparison may proceed.")


if __name__ == "__main__":
    main()
