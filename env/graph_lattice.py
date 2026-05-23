"""Grafo gerarchico a lattice fisso, su GPU (Warp + torch).

Due livelli (vedi ROADMAP):
  - BASSO: lattice ego-centrico K x K, denso, ancorato al robot (si muove con lui).
  - ALTO : anchor globali sparsi in coord assolute (centri di frontiera, capped+padded).

Per il livello basso si calcolano su GPU:
  - validita nodi (in-bounds + cella FREE nella belief)
  - edge-mask 8-vicini collision-free (LOS solo su celle FREE nella belief)
  - utility nodo (n. raggi che vedono una frontiera entro R) -> proxy di IR2 observable_frontiers

Convenzioni: belief 0=unknown,1=free,2=ostacolo. Coord = (x=col, y=row). Griglia [N,H,W].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import warp as wp

FREE = wp.uint8(1)
OBSTACLE = wp.uint8(2)
UNKNOWN_U8 = wp.uint8(0)

# 8 direzioni nel grid-index (di riga, dj colonna)
_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
N_DIR = 8


@wp.func
def _los_free(belief: wp.array3d(dtype=wp.uint8), e: wp.int32,
              x0: wp.float32, y0: wp.float32, x1: wp.float32, y1: wp.float32) -> wp.int32:
    """1 se il segmento (x0,y0)->(x1,y1) attraversa solo celle FREE nella belief, 0 altrimenti."""
    H = belief.shape[1]
    W = belief.shape[2]
    dx = x1 - x0
    dy = y1 - y0
    dist = wp.sqrt(dx * dx + dy * dy)
    n = wp.int32(dist) + 1
    inv = 1.0 / wp.float32(n)
    for s in range(1, n + 1):
        t = wp.float32(s) * inv
        ix = wp.int32(x0 + dx * t + 0.5)
        iy = wp.int32(y0 + dy * t + 0.5)
        if ix < 0 or iy < 0 or ix >= W or iy >= H:
            return 0
        if belief[e, iy, ix] != FREE:
            return 0
    return 1


@wp.kernel
def _node_valid(belief: wp.array3d(dtype=wp.uint8),
                env_idx: wp.array(dtype=wp.int32),    # [L] lattice -> mondo
                coords: wp.array2d(dtype=wp.vec2),    # [L, KK]
                valid: wp.array2d(dtype=wp.uint8)):
    e, k = wp.tid()
    b = env_idx[e]
    H = belief.shape[1]
    W = belief.shape[2]
    p = coords[e, k]
    ix = wp.int32(p[0] + 0.5)
    iy = wp.int32(p[1] + 0.5)
    if ix < 0 or iy < 0 or ix >= W or iy >= H:
        valid[e, k] = wp.uint8(0)
    elif belief[b, iy, ix] == FREE:
        valid[e, k] = wp.uint8(1)
    else:
        valid[e, k] = wp.uint8(0)


@wp.kernel
def _edge_mask(belief: wp.array3d(dtype=wp.uint8),
               env_idx: wp.array(dtype=wp.int32),
               coords: wp.array2d(dtype=wp.vec2),
               valid: wp.array2d(dtype=wp.uint8),
               diy: wp.array(dtype=wp.int32),
               dix: wp.array(dtype=wp.int32),
               K: wp.int32,
               edges: wp.array3d(dtype=wp.uint8)):    # [L, KK, N_DIR]
    e, k, d = wp.tid()
    b = env_idx[e]
    gi = k / K
    gj = k % K
    ni = gi + diy[d]
    nj = gj + dix[d]
    if ni < 0 or nj < 0 or ni >= K or nj >= K:
        edges[e, k, d] = wp.uint8(0)
        return
    nk = ni * K + nj
    if valid[e, k] == wp.uint8(0) or valid[e, nk] == wp.uint8(0):
        edges[e, k, d] = wp.uint8(0)
        return
    pa = coords[e, k]
    pb = coords[e, nk]
    edges[e, k, d] = wp.uint8(_los_free(belief, b, pa[0], pa[1], pb[0], pb[1]))


@wp.kernel
def _node_utility(belief: wp.array3d(dtype=wp.uint8),
                  frontier: wp.array3d(dtype=wp.uint8),   # [N,hc,wc] coarse (res = fscale)
                  env_idx: wp.array(dtype=wp.int32),
                  coords: wp.array2d(dtype=wp.vec2),
                  valid: wp.array2d(dtype=wp.uint8),
                  util_range: wp.float32,
                  n_uray: wp.int32,
                  fscale: wp.int32,
                  utility: wp.array2d(dtype=wp.int32)):
    e, k, r = wp.tid()
    if valid[e, k] == wp.uint8(0):
        return
    b = env_idx[e]
    H = belief.shape[1]
    W = belief.shape[2]
    HC = frontier.shape[1]
    WC = frontier.shape[2]
    p = coords[e, k]
    ang = 2.0 * wp.pi * wp.float32(r) / wp.float32(n_uray)
    dx = wp.cos(ang)
    dy = wp.sin(ang)
    t = wp.float32(1.0)
    while t <= util_range:
        ix = wp.int32(p[0] + dx * t + 0.5)
        iy = wp.int32(p[1] + dy * t + 0.5)
        if ix < 0 or iy < 0 or ix >= W or iy >= H:
            return
        fy = iy / fscale
        fx = ix / fscale
        if fy >= 0 and fx >= 0 and fy < HC and fx < WC:
            if frontier[b, fy, fx] == wp.uint8(1):
                wp.atomic_add(utility, e, k, 1)
                return
        if belief[b, iy, ix] != FREE:    # ostacolo o ignoto -> occluso
            return
        t += 1.0


@wp.kernel
def _frontier_coarse(belief: wp.array3d(dtype=wp.uint8), scale: wp.int32,
                     out: wp.array3d(dtype=wp.uint8)):
    """Vera frontiera full-res (cella FREE con vicino-4 UNKNOWN), output ridotto in coarse (OR).
    NON trapassa i muri: l'ostacolo tra free e unknown impedisce l'adiacenza diretta. out pre-azzerato."""
    e, y, x = wp.tid()
    if belief[e, y, x] != FREE:
        return
    H = belief.shape[1]
    W = belief.shape[2]
    is_f = False
    if y > 0 and belief[e, y - 1, x] == UNKNOWN_U8:
        is_f = True
    if y < H - 1 and belief[e, y + 1, x] == UNKNOWN_U8:
        is_f = True
    if x > 0 and belief[e, y, x - 1] == UNKNOWN_U8:
        is_f = True
    if x < W - 1 and belief[e, y, x + 1] == UNKNOWN_U8:
        is_f = True
    if is_f:
        out[e, y / scale, x / scale] = wp.uint8(1)


def frontier_coarse_warp(belief: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Frontier coarse [N,H//scale,W//scale] bool: vera adiacenza full-res ridotta a coarse (OR).
    Kernel Warp, niente bleed sui muri (a differenza del pooling coarse)."""
    wp.init()
    n, h, w = belief.shape
    hc, wc = h // scale, w // scale
    dev = belief.device
    out = torch.zeros((n, hc, wc), dtype=torch.uint8, device=dev)
    bel = wp.from_torch(belief.contiguous(), dtype=wp.uint8)
    wp.launch(_frontier_coarse, dim=(n, h, w),
              inputs=[bel, int(scale), wp.from_torch(out, dtype=wp.uint8)], device=str(dev))
    wp.synchronize()
    return out > 0


class EgoLattice:
    """Lattice ego-centrico K x K con edge-mask e utility su GPU.

    Gestisce L lattici (es. N mondi x M agenti). `env_idx[l]` mappa il lattice l al
    mondo (indice della belief condivisa). Per M=1, L=N e env_idx = identita'.
    """

    def __init__(self, n_lat: int, env_idx: torch.Tensor | None = None,
                 K: int = 21, spacing: float = 20.0,
                 util_range: float = 70.0, n_uray: int = 60, device: str = "cuda:0"):
        wp.init()
        self.n, self.K, self.KK = n_lat, K, K * K   # self.n = numero lattici L
        self.spacing = float(spacing)
        self.util_range = float(util_range)
        self.n_uray = int(n_uray)
        self.device = device

        if env_idx is None:
            env_idx = torch.arange(n_lat, dtype=torch.int32, device=device)
        self.env_idx_t = env_idx.to(device, torch.int32).contiguous()
        self._env_idx = wp.from_torch(self.env_idx_t, dtype=wp.int32)

        gi = torch.arange(K, device=device).repeat_interleave(K)
        gj = torch.arange(K, device=device).repeat(K)
        c = (K - 1) / 2.0
        self.offsets = torch.stack([(gj - c) * spacing, (gi - c) * spacing], dim=-1)  # [KK,2] (x,y)

        self.coords_t = torch.zeros((n_lat, self.KK, 2), dtype=torch.float32, device=device)
        self.valid_t = torch.zeros((n_lat, self.KK), dtype=torch.uint8, device=device)
        self.edges_t = torch.zeros((n_lat, self.KK, N_DIR), dtype=torch.uint8, device=device)
        self.util_t = torch.zeros((n_lat, self.KK), dtype=torch.int32, device=device)
        self._diy = wp.array([d[0] for d in _DIRS], dtype=wp.int32, device=device)
        self._dix = wp.array([d[1] for d in _DIRS], dtype=wp.int32, device=device)

    def build(self, pos_xy: torch.Tensor, belief: torch.Tensor, frontier: torch.Tensor,
              fscale: int = 1) -> dict:
        """pos_xy [L,2] (posa di ogni lattice), belief [N,H,W], frontier [N,hc,wc] (res=fscale).
        Ritorna dict coords/valid/edges/utility [L,...] su GPU. NB: buffer riusati in-place."""
        self.coords_t.copy_(pos_xy.unsqueeze(1) + self.offsets.unsqueeze(0))
        self.util_t.zero_()
        bel = wp.from_torch(belief.contiguous(), dtype=wp.uint8)
        fro = wp.from_torch(frontier.to(torch.uint8).contiguous(), dtype=wp.uint8)
        coords = wp.from_torch(self.coords_t, dtype=wp.vec2)
        valid = wp.from_torch(self.valid_t, dtype=wp.uint8)
        edges = wp.from_torch(self.edges_t, dtype=wp.uint8)
        util = wp.from_torch(self.util_t, dtype=wp.int32)

        wp.launch(_node_valid, dim=(self.n, self.KK),
                  inputs=[bel, self._env_idx, coords, valid], device=self.device)
        wp.launch(_edge_mask, dim=(self.n, self.KK, N_DIR),
                  inputs=[bel, self._env_idx, coords, valid, self._diy, self._dix, self.K, edges],
                  device=self.device)
        wp.launch(_node_utility, dim=(self.n, self.KK, self.n_uray),
                  inputs=[bel, fro, self._env_idx, coords, valid, self.util_range, self.n_uray,
                          int(fscale), util],
                  device=self.device)
        wp.synchronize()
        return {"coords": self.coords_t, "valid": self.valid_t,
                "edges": self.edges_t, "utility": self.util_t}


def build_anchors(centers: torch.Tensor, valid: torch.Tensor, count: torch.Tensor,
                  a_max: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Seleziona fino ad a_max centri di frontiera per i anchor globali, rankati per count.

    centers [N,C,2], valid [N,C], count [N,C] -> anchors [N,a_max,2], mask [N,a_max].
    Oltre a_max si droppano i centri con count piu basso (drop low-utility).
    """
    n, c, _ = centers.shape
    score = count * valid.float()                       # invalid -> 0
    a_max = min(a_max, c)
    top_val, top_idx = torch.topk(score, a_max, dim=1)  # [N,a_max]
    anchors = torch.gather(centers, 1, top_idx.unsqueeze(-1).expand(-1, -1, 2))
    mask = top_val > 0
    return anchors, mask
