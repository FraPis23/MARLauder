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
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedGATLayer(nn.Module):
    """Single GAT-style layer with K=8 fixed neighbor padding."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.0) -> None:
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

    def forward(
        self,
        x: torch.Tensor,           # [B, N, F]
        edge_idx: torch.Tensor,    # [B, N, K] long
        edge_valid: torch.Tensor,  # [B, N, K] bool
        node_valid: torch.Tensor,  # [B, N] bool
    ) -> torch.Tensor:
        B, N, _ = x.shape
        K = edge_idx.shape[-1]
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, D)             # [B, N, H, D]
        k = self.k_proj(x).view(B, N, H, D)
        v = self.v_proj(x).view(B, N, H, D)

        # Gather neighbor k and v.
        safe_idx = edge_idx.clamp(min=0)                # [B, N, K]
        # k: [B, N, H, D]; gather along dim=1 with [B, N*K, H, D] expand
        gather_idx = safe_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, K, H, D)  # [B, N, K, H, D]
        # We need to gather from k along dim=1 (the N axis), with index [B, N*K, ...]
        k_exp = k.unsqueeze(2).expand(B, N, K, H, D)    # placeholder, will overwrite via gather
        # easier route: reshape k to [B, N, H*D], gather over dim=1, reshape back.
        k_flat = k.reshape(B, N, H * D)
        v_flat = v.reshape(B, N, H * D)
        idx_flat = safe_idx.reshape(B, N * K).unsqueeze(-1).expand(B, N * K, H * D)
        k_nbr = torch.gather(k_flat, 1, idx_flat).view(B, N, K, H, D)            # [B, N, K, H, D]
        v_nbr = torch.gather(v_flat, 1, idx_flat).view(B, N, K, H, D)

        # Attention scores: (q · k_nbr) / sqrt(D)
        scores = (q.unsqueeze(2) * k_nbr).sum(dim=-1) / math.sqrt(D)             # [B, N, K, H]
        # Mask invalid edges → -inf.
        mask = edge_valid.unsqueeze(-1)                                          # [B, N, K, 1]
        scores = scores.masked_fill(~mask, float("-inf"))

        # If a row has all -inf (no valid neighbor), softmax → NaN. Detect & fall back to a self-attention-zero output (no aggregation).
        any_valid = edge_valid.any(dim=-1, keepdim=True).unsqueeze(-1)           # [B, N, 1, 1]
        scores = torch.where(any_valid.expand_as(scores), scores, torch.zeros_like(scores))
        attn = F.softmax(scores, dim=2)                                          # [B, N, K, H]
        attn = self.dropout(attn)
        attn = torch.where(any_valid.expand_as(attn), attn, torch.zeros_like(attn))

        out = (attn.unsqueeze(-1) * v_nbr).sum(dim=2)                            # [B, N, H, D]
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
        self.layers = nn.ModuleList([MaskedGATLayer(d, d, n_heads=n_heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.act = nn.GELU()

    def forward(
        self,
        node_feat: torch.Tensor,   # [B, N, F_in]
        edge_idx: torch.Tensor,    # [B, N, K]
        edge_valid: torch.Tensor,  # [B, N, K]
        node_valid: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        h = self.input_proj(node_feat)
        h = h * node_valid.unsqueeze(-1).float()
        for layer, norm in zip(self.layers, self.norms):
            res = h
            h = layer(h, edge_idx, edge_valid, node_valid)
            h = norm(res + self.act(h))
            h = h * node_valid.unsqueeze(-1).float()
        return h
