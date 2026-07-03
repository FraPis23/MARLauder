"""Masked GAT over a fixed 8-neighbor padded graph (no PyG).

Input per layer:
    x         [B, N, F]         node features (B = n_envs * n_agents)
    edge_idx  [B, N, K] long    neighbor flat index (-1 → padded). Padded entries
                                must be clamped before gathering; their attention
                                weight is set to -inf by `edge_valid`.
    edge_valid [B, N, K] bool   True where the edge is real.
    node_valid [B, N] bool      True where the node is real. Invalid nodes are
                                still embedded (zero in / zero out via masking).

Output of layer: x' [B, N, F_out]

Attention shaping (v0.7):
  A1 — learnable PER-HEAD temperature. The fixed 1/sqrt(D) scaling drove q·k so small
       (tiny input feats · /sqrt(32)) that the softmax stayed ~uniform and q/k received
       almost no gradient (L0 stayed frozen at init). A per-head learnable τ multiplies
       the score so each head can sharpen/soften on its own.
  A2 — PER-HEAD structural feature bias. Each head gets an additive score term computed
       from a FIXED SUBSET of the neighbor's RAW node features (geometry / utility /
       teammate), injecting the routing signal directly and bypassing the gradient-starved
       q/k path. This forces head specialization instead of the 4 heads collapsing onto
       one diffuse pattern.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.init_utils import apply_orthogonal


# Default per-head raw-feature groups (feat order: 0 x_rel, 1 y_rel, 2 utility, 3 age,
# 4 teammate_pot, 5 guidepost). Head 0→geometry, 1→utility, 2→teammate, rest free.
def default_head_feat_groups(n_heads: int, feat_dim: int) -> list[list[int]]:
    base = [[0, 1], [2], [4]]
    groups: list[list[int]] = []
    for h in range(n_heads):
        g = base[h] if h < len(base) else []
        groups.append([i for i in g if i < feat_dim])
    return groups


class MaskedGATLayer(nn.Module):
    """Single GAT-style layer with K=8 fixed neighbor padding.

    Adds a learnable per-head temperature (A1) and a per-head structural feature bias (A2)
    on the attention scores.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        feat_dim: int | None = None,
        head_feat_groups: list[list[int]] | None = None,
    ) -> None:
        super().__init__()
        assert out_dim % n_heads == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.q_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.o_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        # A1 — per-head learnable temperature. score := (q·k)/sqrt(D) · τ_h. τ = exp(log_tau),
        # clamped to [0.1, 10] so it can sharpen without runaway (τ→∞ collapses gradient again).
        self.log_tau = nn.Parameter(torch.zeros(n_heads))

        # A2 — per-head structural feature bias. bias_h(j) = Linear(raw_feat[j, group_h]) → scalar.
        # Heads with an empty group get no bias (pure learned attention).
        self.feat_dim = feat_dim
        if feat_dim is not None:
            self.head_feat_groups = (
                head_feat_groups if head_feat_groups is not None
                else default_head_feat_groups(n_heads, feat_dim)
            )
            self.bias_proj = nn.ModuleDict({
                str(h): nn.Linear(len(g), 1)
                for h, g in enumerate(self.head_feat_groups) if len(g) > 0
            })
        else:
            self.head_feat_groups = [[] for _ in range(n_heads)]
            self.bias_proj = nn.ModuleDict()

        # Inspector hook: when store_attn, keep the per-node neighbor-attention softmax of the
        # last forward (detached) so the viewer can show which neighbors a node aggregates from.
        # We ALSO stash `_last_contrib`: the real per-neighbor value-contribution magnitude to the
        # output embedding (‖c_j‖ through o_proj). This is the honest "how the H heads combine at
        # the end" number — the model concatenates heads and mixes them with o_proj, it never means
        # them, so a head-mean of attention is not a value the network uses. See forward().
        self.store_attn = False
        self._last_attn: torch.Tensor | None = None
        self._last_contrib: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,           # [B, N, F_embed]
        edge_idx: torch.Tensor,    # [B, N, K] long
        edge_valid: torch.Tensor,  # [B, N, K] bool
        node_valid: torch.Tensor,  # [B, N] bool
        node_feat: torch.Tensor | None = None,  # [B, N, F_raw] RAW feats for the structural bias
    ) -> torch.Tensor:
        B, N, _ = x.shape
        K = edge_idx.shape[-1]
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, D)             # [B, N, H, D]
        k = self.k_proj(x).view(B, N, H, D)
        v = self.v_proj(x).view(B, N, H, D)

        # Gather neighbor k and v: reshape to [B, N, H*D], gather over the N axis by neighbor
        # index, reshape back.
        safe_idx = edge_idx.clamp(min=0)                # [B, N, K]
        k_flat = k.reshape(B, N, H * D)
        v_flat = v.reshape(B, N, H * D)
        idx_flat = safe_idx.reshape(B, N * K).unsqueeze(-1).expand(B, N * K, H * D)
        k_nbr = torch.gather(k_flat, 1, idx_flat).view(B, N, K, H, D)            # [B, N, K, H, D]
        v_nbr = torch.gather(v_flat, 1, idx_flat).view(B, N, K, H, D)

        # Self-loop: a node also attends to ITSELF. Standard GAT (Veličković 2018) includes i in
        # its neighborhood N_i. We append a self slot (always valid) so the softmax is over the K
        # neighbors + self → the node learns how much to weight its own features vs neighbors',
        # AND a node with no valid neighbor still has a well-defined self-only output (no NaN, so
        # the old all-invalid fallback is gone).
        self_valid = torch.ones(B, N, 1, dtype=torch.bool, device=x.device)
        k_all = torch.cat([k_nbr, k.unsqueeze(2)], dim=2)                        # [B, N, K+1, H, D]
        v_all = torch.cat([v_nbr, v.unsqueeze(2)], dim=2)                        # [B, N, K+1, H, D]
        valid_all = torch.cat([edge_valid, self_valid], dim=2)                   # [B, N, K+1]

        # Attention scores: (q · k_all) / sqrt(D), then A1 per-head temperature.
        scores = (q.unsqueeze(2) * k_all).sum(dim=-1) / math.sqrt(D)             # [B, N, K+1, H]
        tau = self.log_tau.exp().clamp(0.1, 10.0)                                # [H]
        scores = scores * tau

        # A2 — per-head structural bias from the neighbor's RAW features (self slot included).
        # Build the full [B, N, K+1, H] bias then add once (compile-safe, no in-place index-set).
        if node_feat is not None and len(self.bias_proj) > 0:
            Fr = node_feat.shape[-1]
            feat_flat = node_feat.reshape(B, N, Fr)
            fidx = safe_idx.reshape(B, N * K).unsqueeze(-1).expand(B, N * K, Fr)
            feat_nbr = torch.gather(feat_flat, 1, fidx).view(B, N, K, Fr)        # [B, N, K, F_raw]
            feat_all = torch.cat([feat_nbr, node_feat.unsqueeze(2)], dim=2)      # [B, N, K+1, F_raw]
            bias_cols = []
            for h in range(H):
                key = str(h)
                if key not in self.bias_proj:
                    bias_cols.append(torch.zeros(B, N, K + 1, device=x.device, dtype=scores.dtype))
                else:
                    g = self.head_feat_groups[h]
                    bias_cols.append(self.bias_proj[key](feat_all[..., g]).squeeze(-1))  # [B, N, K+1]
            scores = scores + torch.stack(bias_cols, dim=-1)                     # [B, N, K+1, H]

        scores = scores.masked_fill(~valid_all.unsqueeze(-1), float("-inf"))
        attn = F.softmax(scores, dim=2)                                          # [B, N, K+1, H]
        attn = self.dropout(attn)

        if self.store_attn:
            self._last_attn = attn.detach()                                     # [B, N, K+1, H]
            # REAL per-neighbor value-contribution to the OUTPUT embedding, combining heads exactly
            # the way the model does (attn × value, then the o_proj mix — no head-mean). Because
            # o_proj is linear, out − bias = Σ_slot c_slot with
            #   c_slot = Σ_h attn[slot,h] · (O_h · v[slot,h]),  O_h = o_proj.weight[:, h·D:(h+1)·D].
            # ‖c_slot‖₂ is that neighbor/self's contribution magnitude to the 128-d embedding. This
            # is exact (not an estimate); eval-only, detached, does not touch training.
            with torch.no_grad():
                wv = attn.unsqueeze(-1) * v_all                                  # [B, N, K+1, H, D]
                O = self.o_proj.weight.view(self.out_dim, H, D)                  # [out_dim, H, D]
                c = torch.einsum("ohd,bnshd->bnso", O, wv)                       # [B, N, K+1, out_dim]
                self._last_contrib = c.norm(dim=-1).detach()                    # [B, N, K+1]
        out = (attn.unsqueeze(-1) * v_all).sum(dim=2)                            # [B, N, H, D]
        out = out.reshape(B, N, H * D)
        out = self.o_proj(out)                                                   # [B, N, out_dim]

        # Zero out features of invalid nodes (so they don't leak through residual).
        out = out * node_valid.unsqueeze(-1).float()
        return out


class GATEncoder(nn.Module):
    """2-layer GAT with residual + LayerNorm + GELU. Input proj from F_in → d."""

    def __init__(self, in_dim: int, d: int = 128, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d)
        # Layers take the RAW feature dim (in_dim) so the A2 structural bias reads the original
        # geometry/utility/teammate features, not the evolving embedding.
        self.layers = nn.ModuleList([
            MaskedGATLayer(d, d, n_heads=n_heads, feat_dim=in_dim) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.act = nn.GELU()
        self.last_attn: list | None = None   # inspector: per-layer [B,N,K+1,H] from last forward
        self.last_contrib: list | None = None  # inspector: per-layer [B,N,K+1] real value-contrib norm
        # MAPPO paper Tab.7 — orthogonal init across all GAT projections.
        apply_orthogonal(self)
        # A2 — start the structural-bias projections SMALL so they nudge (not dominate) early
        # training, then grow as the signal proves useful. (apply_orthogonal set them to gain √2.)
        for layer in self.layers:
            for lin in layer.bias_proj.values():
                nn.init.orthogonal_(lin.weight, 0.1)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)

    def forward(
        self,
        node_feat: torch.Tensor,   # [B, N, F_in]
        edge_idx: torch.Tensor,    # [B, N, K]
        edge_valid: torch.Tensor,  # [B, N, K]
        node_valid: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        h = self.input_proj(node_feat)
        h = h * node_valid.unsqueeze(-1).float()
        attns, contribs = [], []
        for layer, norm in zip(self.layers, self.norms):
            res = h
            h = layer(h, edge_idx, edge_valid, node_valid, node_feat=node_feat)
            if self.store_attn:
                attns.append(layer._last_attn)
                contribs.append(layer._last_contrib)
            h = norm(res + self.act(h))
            h = h * node_valid.unsqueeze(-1).float()
        self.last_attn = attns if self.store_attn else None
        self.last_contrib = contribs if self.store_attn else None
        return h

    def head_feat_groups(self) -> list[list[int]]:
        """Per-head raw-feature groups (A2 structural bias) — real config, used to label heads
        in the inspector (head0→geometry, head1→utility, head2→teammate, …). Read from layer 0."""
        return list(self.layers[0].head_feat_groups)

    @property
    def store_attn(self) -> bool:
        return getattr(self, "_store_attn", False)

    @store_attn.setter
    def store_attn(self, v: bool) -> None:
        self._store_attn = bool(v)
        for layer in self.layers:
            layer.store_attn = bool(v)
