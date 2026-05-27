"""Vectorized frontier detection on GPU (pure torch, no Python loop, no skimage).

Convention from env.world_warp:
    occupancy: uint8 [N, H, W]   0 = UNKNOWN, 1 = FREE, 2 = OBSTACLE

Frontier definition (TOM-style):
    A FREE cell with 2..7 UNKNOWN neighbors in its 3x3 ring.
    (>=2: must border unknown space → reachable to expand exploration.
     <=7: not totally surrounded by unknown → otherwise an isolated speck.)

Public API:
    compute_frontier(occupancy) -> bool [N, H, W]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

UNKNOWN = 0
FREE = 1
OBSTACLE = 2

_KERNEL = torch.tensor(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


def compute_frontier(occupancy: torch.Tensor, min_unknown: int = 2, max_unknown: int = 7) -> torch.Tensor:
    """occupancy uint8 [N, H, W] → bool [N, H, W]: True at frontier cells."""
    assert occupancy.dim() == 3 and occupancy.dtype == torch.uint8
    dev = occupancy.device
    is_free = occupancy == FREE
    is_unknown = occupancy == UNKNOWN

    # Count UNKNOWN neighbors in 3x3 (excluding center).
    k = _KERNEL.to(dev)
    nu = F.conv2d(
        is_unknown.float().unsqueeze(1),    # [N, 1, H, W]
        k,
        padding=1,
    ).squeeze(1)                            # [N, H, W]

    frontier = is_free & (nu >= float(min_unknown)) & (nu <= float(max_unknown))
    return frontier
