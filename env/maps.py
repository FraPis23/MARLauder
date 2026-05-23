"""Loader delle mappe preprocessate (vedi scripts/preprocess_maps.py).

Legge i memmap uint8 [N,Hc,Wc] (0=ostacolo, 1=free) + meta, senza decodificare PNG.
Espone un campionamento di batch verso GPU per gli env vettorizzati.

Convenzioni:
  ground-truth : 0 = ostacolo, 1 = free (bordo padding = 0)
  start        : (row, col) int16, oppure (-1,-1) se assente -> scegliere free random a runtime
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATA_ROOT = Path("/workspace/MARLauder/data")


@dataclass
class MapSplit:
    maps: np.memmap            # [N, Hc, Wc] uint8
    starts: np.ndarray         # [N, 2] int16  (row, col) o (-1,-1)
    valid_shapes: np.ndarray   # [N, 2] int16  (h, w) nativi
    free_counts: np.ndarray    # [N] int32
    canvas: tuple[int, int]    # (Hc, Wc)
    files: np.ndarray          # [N] nomi file

    def __len__(self) -> int:
        return self.maps.shape[0]


def load_split(split: str, data_root: Path = DATA_ROOT) -> MapSplit:
    d = data_root / split
    maps = np.load(d / "maps.npy", mmap_mode="r")
    meta = np.load(d / "meta.npz", allow_pickle=True)
    hc, wc = (int(meta["canvas"][0]), int(meta["canvas"][1]))
    return MapSplit(
        maps=maps,
        starts=meta["starts"],
        valid_shapes=meta["valid_shapes"],
        free_counts=meta["free_counts"],
        canvas=(hc, wc),
        files=meta["files"],
    )


def sample_batch(
    split: MapSplit,
    n: int,
    device: torch.device | str = "cuda",
    generator: torch.Generator | None = None,
    indices: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Campiona n mappe -> tensori GPU.

    Ritorna:
      gt      [n, Hc, Wc] uint8  (0=ostacolo, 1=free) su device
      starts  [n, 2] int32  (row, col); -1 dove assente
      free    [n] int32  (n. celle free, denom. copertura)
      idx     [n] indici scelti (numpy)
    """
    if indices is None:
        if generator is not None:
            idx = torch.randint(len(split), (n,), generator=generator).numpy()
        else:
            idx = np.random.randint(0, len(split), size=n)
    else:
        idx = indices

    batch = np.ascontiguousarray(split.maps[idx])          # copia solo le n mappe servite
    gt = torch.from_numpy(batch).to(device, non_blocking=True)
    starts = torch.from_numpy(split.starts[idx].astype(np.int32)).to(device, non_blocking=True)
    free = torch.from_numpy(split.free_counts[idx].astype(np.int32)).to(device, non_blocking=True)
    return gt, starts, free, idx
