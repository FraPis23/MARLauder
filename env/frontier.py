"""Frontier detection su GPU (torch), dalla belief grid.

Frontier = cella FREE adiacente ad almeno una cella UNKNOWN (cfr IR2 find_frontier,
bordo free/unexplored). Tutto vettorizzato su N mondi, niente numpy.

belief: 0=unknown, 1=free, 2=ostacolo.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

FREE = 1
UNKNOWN = 0


_NB4 = None


def _nb4(device):
    global _NB4
    if _NB4 is None or _NB4.device != device:
        _NB4 = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                            dtype=torch.float32, device=device).view(1, 1, 3, 3)
    return _NB4


def frontier_mask(belief: torch.Tensor) -> torch.Tensor:
    """belief [N,H,W] uint8 -> mask [N,H,W] bool: True dove cella free borda l'ignoto (full-res)."""
    free = (belief == FREE)
    unk = (belief == UNKNOWN).float().unsqueeze(1)        # [N,1,H,W]
    unk_dil = F.conv2d(unk, _nb4(belief.device), padding=1) > 0
    return free & unk_dil.squeeze(1)


def frontier_coarse(belief: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Frontier su mappa downsampled (res `scale`), come IR2 -> ~scale^2 volte piu veloce.

    Una cella-grossa e' frontiera se contiene free E confina con celle-grosse contenenti unknown.
    Ritorna mask coarse [N, H//scale, W//scale] bool. Le coord vanno riportate a full-res x scale.
    """
    free = (belief == FREE).float().unsqueeze(1)
    unk = (belief == UNKNOWN).float().unsqueeze(1)
    fc = F.max_pool2d(free, scale, scale)                 # [N,1,hc,wc] free presente
    uc = F.max_pool2d(unk, scale, scale)                  # unknown presente
    uc_dil = F.conv2d(uc, _nb4(belief.device), padding=1) > 0
    return ((fc > 0) & uc_dil).squeeze(1)


def frontier_centers(mask: torch.Tensor, stride: int = 8, min_count: int = 1,
                     coord_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Comprime la mask in 'centri' su griglia grossolana (per gli anchor globali).

    Ritorna:
      centers  [N, C, 2] float (x=col, y=row)  centroide dei pixel di frontiera nella cella
      valid    [N, C] bool
      count    [N, C] float  numero pixel di frontiera nella cella (per ranking anchor)
    dove C = (H//stride)*(W//stride). Un centro e' valido se la sua cella contiene
    >= min_count pixel di frontiera; la coord e' il centroide dei pixel di frontiera.
    """
    n, h, w = mask.shape
    m = mask.float().unsqueeze(1)                          # [N,1,H,W]
    # somma per cella + centroide via media pesata delle coordinate
    ys = torch.arange(h, device=mask.device).view(1, 1, h, 1).float()
    xs = torch.arange(w, device=mask.device).view(1, 1, 1, w).float()
    cnt = F.avg_pool2d(m, stride, stride) * stride * stride                    # [N,1,gh,gw]
    sy = F.avg_pool2d(m * ys, stride, stride) * stride * stride
    sx = F.avg_pool2d(m * xs, stride, stride) * stride * stride
    cnt_f = cnt.clamp(min=1e-6)
    cy = (sy / cnt_f).squeeze(1)                           # [N,gh,gw]
    cx = (sx / cnt_f).squeeze(1)
    cnt = cnt.squeeze(1)
    valid = cnt >= min_count
    centers = torch.stack([cx.reshape(n, -1), cy.reshape(n, -1)], dim=-1) * coord_scale  # [N,C,2]
    return centers, valid.reshape(n, -1), cnt.reshape(n, -1)
