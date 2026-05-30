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


class MultiSplit:
    """H.5 — Weighted union of multiple Split sources for curriculum training.

    `weights` is a list of float weights (must match `splits` length). At sample time,
    each map is drawn from a split chosen by the weight distribution.

    Mutable: `weights` can be updated each iter (e.g., curriculum ramp).
    """
    def __init__(self, splits: list["Split"], weights: list[float]) -> None:
        assert len(splits) == len(weights) and len(splits) >= 1
        canvas0 = splits[0].canvas
        device0 = splits[0].device
        for s in splits[1:]:
            if s.canvas != canvas0:
                raise ValueError(
                    f"MultiSplit canvases differ: '{splits[0].name}'={canvas0} vs "
                    f"'{s.name}'={s.canvas}. Curriculum requires same-canvas splits "
                    f"(Warp world allocates per H×W). Pre-process maps to common size first."
                )
            assert s.device == device0, "all splits must be on same device"
        self.splits = splits
        self.weights = list(weights)
        self.canvas = canvas0
        self.device = device0
        # Compose `n` as sum of sub-splits' n for queries; not meaningful for sampling.
        self.n = sum(s.n for s in splits)
        self.name = "+".join(s.name for s in splits)

    def set_weights(self, weights: list[float]) -> None:
        assert len(weights) == len(self.splits)
        self.weights = list(weights)

    def sample_one(self, rng: np.random.Generator) -> tuple["Split", int]:
        """Pick a (split, idx) sample."""
        w = np.array(self.weights, dtype=np.float64)
        w = w / w.sum()
        si = int(rng.choice(len(self.splits), p=w))
        sp = self.splits[si]
        idx = int(rng.integers(0, sp.n))
        return sp, idx


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
    split: "Split | MultiSplit",
    n: int,
    indices: np.ndarray | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull n maps + starts. If a start is (-1,-1) → pick random free cell.

    Accepts Split or MultiSplit (H.5 curriculum). When MultiSplit, each map drawn from
    a sub-split chosen by current weights. `indices` is ignored when MultiSplit (mapping
    indices to sub-splits is ambiguous; use single Split for indexed sampling).

    Returns:
        gt      uint8  [n, H, W]  on `device`
        starts  int16  [n, 2]     (row, col), on `device`
        free_counts int32 [n]    on `device`
    """
    dev = device or split.device
    rng = np.random.default_rng(seed)
    if isinstance(split, MultiSplit):
        assert indices is None, "MultiSplit does not support indexed sampling"
        H, W = split.canvas
        gt_np = np.empty((n, H, W), dtype=np.uint8)
        starts_np = np.empty((n, 2), dtype=np.int16)
        free_np = np.empty((n,), dtype=np.int32)
        for i in range(n):
            sp, idx = split.sample_one(rng)
            m = np.asarray(sp.gt[int(idx)])
            gt_np[i] = m
            sr, sc = int(sp.starts[idx, 0]), int(sp.starts[idx, 1])
            if sr < 0 or sc < 0:
                sr, sc = _pick_free_cell(m, rng)
            starts_np[i] = (sr, sc)
            free_np[i] = int(sp.free_counts[idx])
    else:
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
            m = np.asarray(split.gt[int(idx)])
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
