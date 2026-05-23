"""Comunicazione: connettivita' + provider delle posizioni dei compagni.

Astrazione pensata per il piano: ORA posizioni note (onniscienza), in FUTURO l'onniscienza
viene sostituita da un modulo di stima della posizione, dietro la stessa interfaccia, senza
toccare reti/env. La connettivita' fa da gating unico per mappe (Fase 7) e posizioni.

Baseline: connettivita' tutti-connessi -> posizioni vere note a tutti.
"""
from __future__ import annotations

import torch


def all_connected(n: int, m: int, device) -> torch.Tensor:
    """Connettivita' baseline [N,M,M] bool: tutti connessi (diagonale inclusa)."""
    return torch.ones((n, m, m), dtype=torch.bool, device=device)


class PositionProvider:
    """Fornisce le posizioni dei compagni alla policy, gated dalla connettivita'.

    Sostituisci `estimate` (default: posizioni vere) col modulo di predizione futuro.
    Ritorna:
      pos_rel [N,M,M,2]  posizione del compagno j relativa all'agente i (xj-xi, yj-yi)
      known   [N,M,M]    True se i conosce j (connesso). Diagonale (se stesso) sempre True.
    """

    def estimate(self, pos: torch.Tensor, connectivity: torch.Tensor) -> torch.Tensor:
        # baseline onnisciente: posizione vera. (futuro: predizione se disconnesso)
        return pos

    def __call__(self, pos: torch.Tensor, connectivity: torch.Tensor):
        # pos [N,M,2], connectivity [N,M,M]
        est = self.estimate(pos, connectivity)                    # [N,M,2]
        pos_rel = est.unsqueeze(1) - pos.unsqueeze(2)             # [N,M,M,2] (j relativo a i)
        known = connectivity.clone()
        return pos_rel, known
