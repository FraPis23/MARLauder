"""Split loader: memmap data/<split>/maps.npy → GPU tensors.

Convention (from scripts/preprocess_maps.py):
    maps[i]  uint8 [H, W]  — 0 = obstacle, 1 = free  (padded with obstacle)
    starts[i] int16 [2]    — (row, col); (-1, -1) means "pick a random free cell"

Public API:
    load_split(split, root='data', device='cuda:0') -> Split
    sample_batch(split, n, indices=None, device=...)
        -> (gt[n,H,W] uint8 on GPU, starts[n,2] int16 on GPU, free_counts[n] int32 on GPU)

Indexing convention everywhere in the project:
    pos = (x, y) = (col, row)   — matches env/world_warp.py vec2.
    grid[H, W]  — row-major (row, col).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_ROOT = Path("/workspace/MARLauder/data")
FREE = 1
OBSTACLE = 0


@dataclass
class Split:
    name: str
    gt: np.memmap          # uint8 [N, H, W] on host (memmap, read-only)
    starts: np.ndarray     # int16 [N, 2]  (row, col)
    valid_shapes: np.ndarray  # int16 [N, 2]
    free_counts: np.ndarray   # int32 [N]
    canvas: tuple[int, int]   # (H, W)
    files: np.ndarray         # [N] strings
    device: str

    @property
    def n(self) -> int:
        return int(self.gt.shape[0])


def load_split(split: str, root: Path | str = DEFAULT_ROOT, device: str = "cuda:0") -> Split:
    root = Path(root)
    sd = root / split
    maps_path = sd / "maps.npy"
    meta_path = sd / "meta.npz"
    if not maps_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"split '{split}' missing under {root}")
    gt = np.load(maps_path, mmap_mode="r")
    meta = np.load(meta_path)
    return Split(
        name=split,
        gt=gt,
        starts=meta["starts"],
        valid_shapes=meta["valid_shapes"],
        free_counts=meta["free_counts"],
        canvas=tuple(meta["canvas"].tolist()),
        files=meta["files"],
        device=device,
    )


def _pick_free_cell(gt_np: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Return (row, col) of a random FREE cell."""
    ys, xs = np.nonzero(gt_np == FREE)
    if ys.size == 0:
        return (0, 0)
    i = int(rng.integers(0, ys.size))
    return int(ys[i]), int(xs[i])


def sample_batch(
    split: Split,
    n: int,
    indices: np.ndarray | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull n maps + starts. If a start is (-1,-1) → pick random free cell.

    Returns:
        gt      uint8  [n, H, W]  on `device`
        starts  int16  [n, 2]     (row, col), on `device`
        free_counts int32 [n]    on `device`
    """
    dev = device or split.device
    rng = np.random.default_rng(seed)
    if indices is None:
        indices = rng.integers(0, split.n, size=n, dtype=np.int64)
    else:
        indices = np.asarray(indices, dtype=np.int64)
        assert indices.shape == (n,)
    H, W = split.canvas
    gt_np = np.empty((n, H, W), dtype=np.uint8)
    starts_np = np.empty((n, 2), dtype=np.int16)
    free_np = np.empty((n,), dtype=np.int32)
    for i, idx in enumerate(indices):
        m = np.asarray(split.gt[int(idx)])  # materializes uint8 [H, W]
        gt_np[i] = m
        sr, sc = int(split.starts[idx, 0]), int(split.starts[idx, 1])
        if sr < 0 or sc < 0:
            sr, sc = _pick_free_cell(m, rng)
        starts_np[i] = (sr, sc)
        free_np[i] = int(split.free_counts[idx])
    gt = torch.from_numpy(gt_np).contiguous().to(dev, non_blocking=True)
    starts = torch.from_numpy(starts_np).contiguous().to(dev, non_blocking=True)
    free_counts = torch.from_numpy(free_np).contiguous().to(dev, non_blocking=True)
    return gt, starts, free_counts
