"""Mondo occupancy-grid con sensore LIDAR 360 su GPU (NVIDIA Warp).

Vettorizzato su N mondi x M agenti. Belief CONDIVISA per mondo (comms perfette, baseline):
tutti gli agenti di un mondo scrivono nella stessa griglia belief. Ogni agente costruira'
poi il proprio ego-grafo da questa belief condivisa (vedi graph_lattice).

Convenzioni griglie (uint8):
  ground_truth : 0 = ostacolo, 1 = free
  belief       : 0 = unknown,  1 = free, 2 = ostacolo
Coordinate: posizione = vec2 (x=colonna, y=riga); griglia [mondo, riga, col].
"""
from __future__ import annotations

import torch
import warp as wp

UNKNOWN = wp.uint8(0)
FREE = wp.uint8(1)
OBSTACLE = wp.uint8(2)
GT_OBSTACLE = wp.uint8(0)


@wp.kernel
def _lidar_scan(
    ground_truth: wp.array3d(dtype=wp.uint8),   # [N, H, W]
    belief: wp.array3d(dtype=wp.uint8),         # [N, H, W] condivisa per mondo
    pos: wp.array2d(dtype=wp.vec2),             # [N, M]
    sensor_range: wp.float32,
    n_rays: wp.int32,
):
    e, a, r = wp.tid()                           # mondo, agente, raggio
    H = ground_truth.shape[1]
    W = ground_truth.shape[2]
    p = pos[e, a]

    ang = 2.0 * wp.pi * wp.float32(r) / wp.float32(n_rays)
    dx = wp.cos(ang)
    dy = wp.sin(ang)

    t = wp.float32(1.0)
    while t <= sensor_range:
        ix = wp.int32(p[0] + dx * t + 0.5)
        iy = wp.int32(p[1] + dy * t + 0.5)
        if ix < 0 or iy < 0 or ix >= W or iy >= H:
            break
        if ground_truth[e, iy, ix] == GT_OBSTACLE:
            belief[e, iy, ix] = OBSTACLE
            break
        belief[e, iy, ix] = FREE
        t += 1.0


@wp.kernel
def _mark_pos_free(belief: wp.array3d(dtype=wp.uint8), pos: wp.array2d(dtype=wp.vec2)):
    e, a = wp.tid()
    p = pos[e, a]
    belief[e, wp.int32(p[1] + 0.5), wp.int32(p[0] + 0.5)] = FREE


class WarpWorld:
    """N mondi x M agenti, belief condivisa per mondo, scansione LIDAR su GPU."""

    def __init__(self, ground_truth: torch.Tensor, n_agents: int = 1,
                 sensor_range: float = 80.0, n_rays: int = 720, device: str = "cuda:0") -> None:
        assert ground_truth.dtype == torch.uint8 and ground_truth.ndim == 3
        wp.init()
        self.device = device
        self.n, self.h, self.w = ground_truth.shape
        self.m = int(n_agents)
        self.sensor_range = float(sensor_range)
        self.n_rays = int(n_rays)

        self.gt_torch = ground_truth.contiguous().to(device)
        self.belief_torch = torch.zeros_like(self.gt_torch)        # [N,H,W] condivisa
        self.gt = wp.from_torch(self.gt_torch, dtype=wp.uint8)
        self.belief = wp.from_torch(self.belief_torch, dtype=wp.uint8)
        self.pos_torch = torch.zeros((self.n, self.m, 2), dtype=torch.float32, device=device)
        self.pos = wp.from_torch(self.pos_torch, dtype=wp.vec2)

    def set_positions(self, pos_xy: torch.Tensor) -> None:
        """pos_xy: [N,2] (M=1) oppure [N,M,2] (x=col, y=row)."""
        if pos_xy.ndim == 2:
            pos_xy = pos_xy.unsqueeze(1)
        self.pos_torch.copy_(pos_xy.to(self.device, torch.float32))

    def scan(self) -> torch.Tensor:
        wp.launch(_lidar_scan, dim=(self.n, self.m, self.n_rays),
                  inputs=[self.gt, self.belief, self.pos, self.sensor_range, self.n_rays],
                  device=self.device)
        wp.launch(_mark_pos_free, dim=(self.n, self.m),
                  inputs=[self.belief, self.pos], device=self.device)
        wp.synchronize()
        return self.belief_torch

    def reset_belief(self) -> None:
        self.belief_torch.zero_()
