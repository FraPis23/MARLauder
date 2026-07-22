"""Teammate-position belief — uniform expanding-zone estimator (v3, minimal).

The belief is a UNIFORM probability distribution (Σ=1) over the set of nodes the teammate could have
reached since the last contact — a geodesic ball that grows by ONE hop per step and never stops,
expanding through the known map AND onward into the unknown THROUGH frontiers (the optimistic
FREE∪UNKNOWN graph crosses over wherever a free node touches unknown). At a comm it collapses to a
point (the teammate's true node).

Deliberately minimal: no FOV carve, no diffusion/reservoir, no centroid — just expand + uniform.

State per (observer i, teammate j): a single mask `reached[N,M,M,N_max]`. As the zone grows, each
node's probability decreases (Σ=1 spread over more nodes).

One step (all gather-based, batch B=N·M, agents folded like the rest of _refresh_obs):
  COLLAPSE (comm)   → reached = {teammate's true node}
  BIRTH (first OOR) → reached = {last-known node}
  EXPAND one hop    → reached |= neighbours(reached) over edge_valid_optim
  p = UNIFORM Σ=1 over reached
"""

from __future__ import annotations

import torch


@torch.no_grad()
def update_teammate_belief(
    reached: torch.Tensor,      # [B, M, N_max] float mask (1 in the zone, 0 outside) — persistent state
    *,
    comm_mask: torch.Tensor,    # [B, M] bool — observer in comm with teammate j THIS step
    team_node: torch.Tensor,    # [B, M] long — j's lattice node (from last_known_pos; = truth on comm)
    nbr_idx: torch.Tensor,      # [N_max, K] long — static lattice neighbour indices (edge_idx_static, ≥0)
    edge_valid_optim: torch.Tensor,  # [B, N_max, K] bool — optimistic (FREE∪UNKNOWN) edge validity
    expand_per_step: int = 1,   # hops the zone grows per env step (1 = "one node per step")
    gate_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the wavefront one step. Returns (reached, p, alive).

    reached [B, M, N_max] float — updated zone mask (expands monotonically while out of comm).
    p       [B, M, N_max] float — UNIFORM posterior over the zone (Σ=1 per (i,j) where alive).
    alive   [B, M] bool         — the zone is non-empty.
    """
    B, M, N = reached.shape
    K = nbr_idx.shape[1]
    dev = reached.device

    idx_bmk = nbr_idx.clamp(min=0).view(1, 1, N * K).expand(B, M, N * K)          # [B, M, N·K]
    ev = edge_valid_optim.float().unsqueeze(1)                                    # [B, 1, N, K] (bcast M)

    # A one-hot on each teammate's node — COLLAPSE (comm) and BIRTH seed.
    delta = torch.zeros((B, M, N), dtype=reached.dtype, device=dev)
    delta.scatter_(2, team_node.clamp(0, N - 1).unsqueeze(-1), 1.0)              # [B, M, N]

    # COLLAPSE: certainty at contact.
    cm = comm_mask.unsqueeze(-1)                                                  # [B, M, 1]
    reached = torch.where(cm, delta, reached)
    # BIRTH: first time out-of-range with an empty zone → seed at the last-known node.
    empty = (reached.sum(-1, keepdim=True) <= gate_eps)
    reached = torch.where(empty & ~cm, delta, reached)

    # EXPAND: grow the reachable set by `expand_per_step` hops over the optimistic graph. A node joins
    # the zone if any of its valid optimistic neighbours is already in it (binary BFS wavefront).
    for _ in range(max(1, expand_per_step)):
        nbr_in = torch.gather(reached, 2, idx_bmk).view(B, M, N, K)               # [B, M, N, K] nbr∈zone?
        grow = (nbr_in * ev).sum(-1) > 0.5                                        # [B, M, N] has a nbr in
        reached = ((reached > 0.5) | grow).float()

    # p = UNIFORM Σ=1 over the zone (per-node prob falls as the zone grows).
    tot = reached.sum(-1, keepdim=True)                                          # [B, M, 1]
    alive = (tot.squeeze(-1) > gate_eps)                                          # [B, M]
    p = reached / tot.clamp(min=1.0)

    # COLLAPSE re-assert: comm slots are certain → force the clean delta.
    reached = torch.where(cm, delta, reached)
    p = torch.where(cm, delta, p)
    alive = alive | comm_mask

    # Self slot j==i carries no meaning — zero it so callers can reduce over j freely.
    b_agent = (torch.arange(B, device=dev) % M)                                   # observer id = b mod M
    is_self = (torch.arange(M, device=dev).view(1, M) == b_agent.view(B, 1))      # [B, M]
    z = is_self.unsqueeze(-1)
    reached = reached.masked_fill(z, 0.0)
    p = p.masked_fill(z, 0.0)
    alive = alive & ~is_self

    return reached, p, alive
