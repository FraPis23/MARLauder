"""Occupancy grid with LIDAR-360 sensor on GPU (NVIDIA Warp), log-odds form.

Per-agent occupancy maps (v0.3): each agent maintains its own [H,W] log-odds grid.
Communication fuses maps via elementwise max (idempotent, no double-counting).

Warp constraint: wp.tid() supports max 3 dims. We fold N×M into a single batch
dim NM = N*M for all kernels. Python-facing API exposes [N, M, H, W] views.

Naming convention:
    `occupancy_*`  ← local sensor-observed map per agent.
    `belief_*`     ← reserved for future ToM module. NOT used here.
"""
from __future__ import annotations

import torch
import warp as wp

# Categorical values (uint8)
_UNKNOWN = 0
_FREE = 1
_OBSTACLE = 2

# Warp constants (kernel-facing)
UNKNOWN = wp.uint8(0)
FREE = wp.uint8(1)
OBSTACLE = wp.uint8(2)
GT_OBSTACLE = wp.uint8(0)

LO_FREE  = wp.float32(0.40)
LO_OCC   = wp.float32(-1.20)
LO_CLAMP = wp.float32(5.0)
LO_FREE_TH = wp.float32(0.40)
LO_OCC_TH  = wp.float32(-0.40)

# Python-float copies for torch ops
_LO_CLAMP   =  5.0
_LO_FREE_TH =  0.40
_LO_OCC_TH  = -0.40


@wp.kernel
def _lidar_scan_logodds(
    ground_truth: wp.array3d(dtype=wp.uint8),   # [N, H, W]
    logodds:      wp.array3d(dtype=wp.float32), # [NM, H, W]  NM = N*M
    pos:          wp.array1d(dtype=wp.vec2),    # [NM]
    n_agents:     wp.int32,
    sensor_range: wp.float32,
    n_rays:       wp.int32,
):
    nm, r = wp.tid()                             # (NM, n_rays)
    e = nm / n_agents                            # env index (integer div)
    H = ground_truth.shape[1]
    W = ground_truth.shape[2]
    p = pos[nm]

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
            wp.atomic_add(logodds, nm, iy, ix, LO_OCC)
            break
        wp.atomic_add(logodds, nm, iy, ix, LO_FREE)
        t += 1.0


@wp.kernel
def _mark_pos_free(
    logodds: wp.array3d(dtype=wp.float32),  # [NM, H, W]
    pos:     wp.array1d(dtype=wp.vec2),     # [NM]
):
    nm = wp.tid()
    p = pos[nm]
    iy = wp.int32(p[1] + 0.5)
    ix = wp.int32(p[0] + 0.5)
    wp.atomic_add(logodds, nm, iy, ix, LO_FREE)


@wp.kernel
def _clamp_logodds(logodds: wp.array3d(dtype=wp.float32)):  # [NM, H, W]
    nm, i, j = wp.tid()
    v = logodds[nm, i, j]
    if v >  LO_CLAMP:
        logodds[nm, i, j] =  LO_CLAMP
    if v < -LO_CLAMP:
        logodds[nm, i, j] = -LO_CLAMP


@wp.kernel
def _derive_categorical(
    logodds:   wp.array3d(dtype=wp.float32),  # [NM, H, W]
    occupancy: wp.array3d(dtype=wp.uint8),    # [NM, H, W]
):
    nm, i, j = wp.tid()
    v = logodds[nm, i, j]
    if v > LO_FREE_TH:
        occupancy[nm, i, j] = FREE
    elif v < LO_OCC_TH:
        occupancy[nm, i, j] = OBSTACLE
    else:
        occupancy[nm, i, j] = UNKNOWN


class WarpWorld:
    """N worlds × M agents, per-agent log-odds occupancy, GPU LiDAR."""

    def __init__(
        self,
        ground_truth: torch.Tensor,    # [N, H, W] uint8
        n_agents: int = 1,
        sensor_range: float = 80.0,
        n_rays: int = 720,
        device: str = "cuda:0",
    ) -> None:
        assert ground_truth.dtype == torch.uint8 and ground_truth.ndim == 3
        wp.init()
        self.device = device
        self.n, self.h, self.w = ground_truth.shape
        self.m = int(n_agents)
        self.nm = self.n * self.m
        self.sensor_range = float(sensor_range)
        self.n_rays = int(n_rays)

        self.gt_torch = ground_truth.contiguous().to(device)
        # Flat [NM, H, W] storage (contiguous → Warp-safe)
        self._logodds_flat = torch.zeros(
            (self.nm, self.h, self.w), dtype=torch.float32, device=device)
        self._occ_flat = torch.zeros(
            (self.nm, self.h, self.w), dtype=torch.uint8, device=device)
        self._pos_flat = torch.zeros(
            (self.nm, 2), dtype=torch.float32, device=device)

        # Python-facing [N, M, H, W] / [N, M, 2] views
        self.occupancy_logodds_torch = self._logodds_flat.view(self.n, self.m, self.h, self.w)
        self.occupancy_torch         = self._occ_flat.view(self.n, self.m, self.h, self.w)
        self.pos_torch               = self._pos_flat.view(self.n, self.m, 2)

        # Warp handles (point at flat storage)
        self.gt              = wp.from_torch(self.gt_torch,     dtype=wp.uint8)
        self._wp_logodds     = wp.from_torch(self._logodds_flat, dtype=wp.float32)
        self._wp_occ         = wp.from_torch(self._occ_flat,     dtype=wp.uint8)
        self._wp_pos         = wp.from_torch(self._pos_flat,     dtype=wp.vec2)

    def set_positions(self, pos_xy: torch.Tensor) -> None:
        """pos_xy: [N, M, 2] or [N, 2] (M=1)."""
        if pos_xy.ndim == 2:
            pos_xy = pos_xy.unsqueeze(1)
        self.pos_torch.copy_(pos_xy.to(self.device, torch.float32))

    def scan(self) -> torch.Tensor:
        """Run LiDAR for all agents, update per-agent log-odds. Returns [N,M,H,W] uint8."""
        wp.launch(_lidar_scan_logodds,
                  dim=(self.nm, self.n_rays),
                  inputs=[self.gt, self._wp_logodds, self._wp_pos,
                          self.m, self.sensor_range, self.n_rays],
                  device=self.device)
        wp.launch(_mark_pos_free,
                  dim=(self.nm,),
                  inputs=[self._wp_logodds, self._wp_pos],
                  device=self.device)
        wp.launch(_clamp_logodds,
                  dim=(self.nm, self.h, self.w),
                  inputs=[self._wp_logodds],
                  device=self.device)
        wp.launch(_derive_categorical,
                  dim=(self.nm, self.h, self.w),
                  inputs=[self._wp_logodds, self._wp_occ],
                  device=self.device)
        wp.synchronize()
        return self.occupancy_torch

    def fuse_maps(self, comm_mask: torch.Tensor) -> None:
        """Fuse per-agent log-odds for communicating pairs.

        comm_mask: [N, M, M] bool — True at (n,i,j) means agent i and j can comm in env n.

        Fusion: max-magnitude — pick lo with larger |lo|, keep its sign. Convention is
        positive lo = FREE, negative lo = OBSTACLE, |lo| ≈ 0 = UNKNOWN.

        Why not torch.max(lo_i, lo_j): max(0, -5) = 0 → UNKNOWN, dropping OBSTACLE evidence.
        Symptom: after rendezvous, cells that teammate observed as wall stop being OBSTACLE
        in my map (my lo=0, their lo=-5 → max=0). Surrounding cells become FREE via their
        positive lo. Then frontier (FREE cell with UNKNOWN neighbors) tints the former-wall
        UNKNOWN-now cells red on render.

        max-magnitude preserves both FREE and OBSTACLE evidence. Idempotent for repeated
        fusion (|lo| is monotonically non-decreasing under this op, capped by the clamp).
        """
        if self.m < 2:
            return
        lo = self.occupancy_logodds_torch   # [N, M, H, W]
        any_changed = False
        for i in range(self.m):
            for j in range(i + 1, self.m):
                can = comm_mask[:, i, j]    # [N] bool
                if not can.any():
                    continue
                any_changed = True
                mask = can.view(-1, 1, 1)   # broadcast over H, W
                lo_i = lo[:, i]
                lo_j = lo[:, j]
                use_i = lo_i.abs() >= lo_j.abs()
                merged = torch.where(use_i, lo_i, lo_j).clamp_(_LO_OCC_TH - 5, _LO_CLAMP)
                lo[:, i] = torch.where(mask, merged, lo[:, i])
                lo[:, j] = torch.where(mask, merged, lo[:, j])
        if any_changed:
            occ = self.occupancy_torch      # [N, M, H, W] — view of _occ_flat
            occ.fill_(_UNKNOWN)
            occ[lo > _LO_FREE_TH] = _FREE
            occ[lo < _LO_OCC_TH]  = _OBSTACLE

    def occupancy_prob(self) -> torch.Tensor:
        """Per-agent sigmoid(log-odds) → [N, M, H, W] float32."""
        return torch.sigmoid(self.occupancy_logodds_torch)

    def team_occupancy_prob(self) -> torch.Tensor:
        """Max log-odds across agents → sigmoid → [N, H, W]. Used for rendering."""
        return torch.sigmoid(self.occupancy_logodds_torch.max(dim=1).values)

    def team_occupancy(self) -> torch.Tensor:
        """Union categorical: FREE if any agent sees FREE → [N, H, W] uint8."""
        occ = self.occupancy_torch         # [N, M, H, W]
        result = torch.zeros(self.n, self.h, self.w, dtype=torch.uint8, device=self.device)
        result[(occ == _OBSTACLE).any(dim=1)] = _OBSTACLE
        result[(occ == _FREE).any(dim=1)]     = _FREE      # FREE wins over OBSTACLE
        return result

    def reset_occupancy(self) -> None:
        self._logodds_flat.zero_()
        self._occ_flat.zero_()
