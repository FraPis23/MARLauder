"""MAPPO ActorCritic over the padded 8-neighbor graph.

Obs schema (from env.explorer.Explorer):
    node_feat        [N, M, N_max, F_in]
    edge_idx         [N, M, N_max, K]
    edge_valid       [N, M, N_max, K]
    node_valid       [N, M, N_max]
    curr_idx         [N, M]
    curr_nbr         [N, M, K]
    curr_nbr_valid   [N, M, K]
    action_mask      [N, M, K]

Actor: shared GATEncoder, gather curr_emb + nbr_embs per (env, agent), GRUCell,
PointerHead. Decentralized: each agent sees only its own graph.

Critic (CTDE): shared GATEncoder; gather curr_emb for each agent, concat across
M, MLP → GRUCell → scalar V(s)  [N].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from models.gat import GATEncoder

F_IN = 7
K = 8


class PointerHead(nn.Module):
    def __init__(self, d: int, init_guidepost_bias: float = 2.0) -> None:
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.d = d
        # Learnable scale for the guidepost-next-hop prior. Initialised positive so
        # untrained policy already prefers the shortest-path neighbor; gradient adjusts.
        self.guidepost_bias = nn.Parameter(torch.tensor(float(init_guidepost_bias)))

    def forward(
        self,
        query: torch.Tensor,           # [B, d]
        keys: torch.Tensor,            # [B, K, d]
        mask: torch.Tensor,            # [B, K] bool (True = valid)
        prior: torch.Tensor | None = None,  # [B, K] float in {0, 1}: 1 at guidepost next-hop slot
    ) -> torch.Tensor:
        q = self.q(query).unsqueeze(1)
        k = self.k(keys)
        scores = (q * k).sum(dim=-1) / math.sqrt(self.d)            # [B, K]
        if prior is not None:
            scores = scores + self.guidepost_bias * prior            # additive logit bias
        any_valid = mask.any(dim=-1, keepdim=True)
        mask_eff = torch.where(any_valid.expand_as(mask), mask, torch.ones_like(mask))
        scores = scores.masked_fill(~mask_eff, float("-inf"))
        return scores


class MarlActorCritic(nn.Module):
    def __init__(self, n_agents: int = 1, d: int = 128, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.M = n_agents
        self.d = d
        self.encoder = GATEncoder(in_dim=F_IN, d=d, n_heads=n_heads, n_layers=n_layers)
        # Actor head
        self.gru_actor = nn.GRUCell(d, d)
        self.pointer = PointerHead(d)
        # Critic head (CTDE)
        self.critic_pre = nn.Sequential(nn.Linear(n_agents * d, d), nn.GELU())
        self.gru_critic = nn.GRUCell(d, d)
        self.critic_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    # ---------------- helpers ----------------
    @staticmethod
    def _flatten_nm(t: torch.Tensor) -> torch.Tensor:
        """Merge first two dims (N, M) into B."""
        return t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])

    def _encode(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run shared encoder and gather curr_emb, nbr_embs per (env, agent).

        Returns:
            h_all      [N*M, N_max, d]
            curr_emb   [N*M, d]
            nbr_embs   [N*M, K, d]
        """
        node_feat = self._flatten_nm(obs["node_feat"])             # [B, N_max, F]
        edge_idx = self._flatten_nm(obs["edge_idx"])               # [B, N_max, K]
        edge_valid = self._flatten_nm(obs["edge_valid"])           # [B, N_max, K]
        node_valid = self._flatten_nm(obs["node_valid"])           # [B, N_max]

        h = self.encoder(node_feat, edge_idx, edge_valid, node_valid)   # [B, N_max, d]

        N, M = obs["curr_idx"].shape
        B = N * M
        curr_idx = obs["curr_idx"].reshape(B)                       # [B]
        curr_nbr = obs["curr_nbr"].reshape(B, K).clamp(min=0)       # [B, K]

        b_arange = torch.arange(B, device=h.device)
        curr_emb = h[b_arange, curr_idx]                            # [B, d]
        nbr_embs = h[b_arange.unsqueeze(-1).expand(B, K), curr_nbr]  # [B, K, d]
        return h, curr_emb, nbr_embs

    # ---------------- actor ----------------
    def act(
        self,
        obs: dict,
        hidden_actor: torch.Tensor,        # [N, M, d]
        hidden_critic: torch.Tensor,       # [N, d]
        deterministic: bool = False,
    ) -> dict:
        N, M = obs["curr_idx"].shape
        h_all, curr_emb, nbr_embs = self._encode(obs)
        # Actor GRU
        h_act_in = hidden_actor.reshape(N * M, self.d)
        h_act_out = self.gru_actor(curr_emb, h_act_in)              # [B, d]
        prior = obs.get("guidepost_nbr_bias")
        prior_flat = prior.reshape(N * M, K) if prior is not None else None
        logits = self.pointer(h_act_out, nbr_embs, obs["action_mask"].reshape(N * M, K), prior_flat)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()

        # Critic uses curr_emb of EACH agent → concat across M.
        curr_emb_per_agent = curr_emb.view(N, M, self.d)            # [N, M, d]
        joint = curr_emb_per_agent.reshape(N, M * self.d)
        joint = self.critic_pre(joint)                              # [N, d]
        h_crit_out = self.gru_critic(joint, hidden_critic)          # [N, d]
        value = self.critic_head(h_crit_out).squeeze(-1)            # [N]

        return {
            "action": action.view(N, M),
            "logp": logp.view(N, M),
            "entropy": entropy.view(N, M),
            "value": value,                                          # [N]
            "logits": logits.view(N, M, K),
            "hidden_actor": h_act_out.view(N, M, self.d),
            "hidden_critic": h_crit_out,
        }

    # ---------------- evaluate (for PPO update) ----------------
    def evaluate(
        self,
        obs: dict,
        action: torch.Tensor,              # [N, M]
        hidden_actor: torch.Tensor,        # [N, M, d]
        hidden_critic: torch.Tensor,       # [N, d]
    ) -> dict:
        N, M = obs["curr_idx"].shape
        _, curr_emb, nbr_embs = self._encode(obs)
        h_act_in = hidden_actor.reshape(N * M, self.d)
        h_act_out = self.gru_actor(curr_emb, h_act_in)
        prior = obs.get("guidepost_nbr_bias")
        prior_flat = prior.reshape(N * M, K) if prior is not None else None
        logits = self.pointer(h_act_out, nbr_embs, obs["action_mask"].reshape(N * M, K), prior_flat)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(action.reshape(N * M))
        entropy = dist.entropy()

        curr_emb_per_agent = curr_emb.view(N, M, self.d)
        joint = curr_emb_per_agent.reshape(N, M * self.d)
        joint = self.critic_pre(joint)
        h_crit_out = self.gru_critic(joint, hidden_critic)
        value = self.critic_head(h_crit_out).squeeze(-1)

        return {
            "logp": logp.view(N, M),
            "entropy": entropy.view(N, M),
            "value": value,
            "hidden_actor": h_act_out.view(N, M, self.d),
            "hidden_critic": h_crit_out,
        }

    def init_hidden(self, n_envs: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(n_envs, self.M, self.d, device=device),
            torch.zeros(n_envs, self.d, device=device),
        )

    # ---------------- chunked encode (v0.2 speedup) ----------------
    def encode_chunk(self, obs_chunk: dict) -> dict:
        """Run encoder ONCE over a whole TBPTT chunk.

        Each obs field has shape `[T, N, M, ...]`. Encoder is feed-forward (no
        temporal state) so we collapse `T*N*M` into a single batch dim.

        Returns dict with:
            curr_emb_chunk    [T, N, M, d]
            nbr_embs_chunk    [T, N, M, K, d]
        """
        node_feat = obs_chunk["node_feat"]                  # [T, N, M, N_max, F]
        T, N, M, N_max, F = node_feat.shape
        B = T * N * M
        nf  = node_feat.reshape(B, N_max, F)
        ei  = obs_chunk["edge_idx"].reshape(B, N_max, K)
        ev  = obs_chunk["edge_valid"].reshape(B, N_max, K)
        nv  = obs_chunk["node_valid"].reshape(B, N_max)

        h = self.encoder(nf, ei, ev, nv)                    # [B, N_max, d]

        curr_idx = obs_chunk["curr_idx"].reshape(B)
        curr_nbr = obs_chunk["curr_nbr"].reshape(B, K).clamp(min=0)
        b_arange = torch.arange(B, device=h.device)
        curr_emb = h[b_arange, curr_idx]                    # [B, d]
        nbr_embs = h[b_arange.unsqueeze(-1).expand(B, K), curr_nbr]  # [B, K, d]
        return {
            "curr_emb": curr_emb.view(T, N, M, self.d),
            "nbr_embs": nbr_embs.view(T, N, M, K, self.d),
        }

    def evaluate_step_from_enc(
        self,
        curr_emb: torch.Tensor,        # [N, M, d]
        nbr_embs: torch.Tensor,        # [N, M, K, d]
        action_mask: torch.Tensor,     # [N, M, K]
        action: torch.Tensor,          # [N, M]
        hidden_actor: torch.Tensor,    # [N, M, d]
        hidden_critic: torch.Tensor,   # [N, d]
        guidepost_nbr_bias: torch.Tensor | None = None,  # [N, M, K]
    ) -> dict:
        """One PPO-evaluate step given pre-encoded curr/nbr embeddings."""
        N, M, _ = curr_emb.shape
        B = N * M
        h_act_in = hidden_actor.reshape(B, self.d)
        h_act_out = self.gru_actor(curr_emb.reshape(B, self.d), h_act_in)
        prior_flat = guidepost_nbr_bias.reshape(B, K) if guidepost_nbr_bias is not None else None
        logits = self.pointer(h_act_out, nbr_embs.reshape(B, K, self.d),
                              action_mask.reshape(B, K), prior_flat)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(action.reshape(B))
        entropy = dist.entropy()

        joint = curr_emb.reshape(N, M * self.d)
        joint = self.critic_pre(joint)
        h_crit_out = self.gru_critic(joint, hidden_critic)
        value = self.critic_head(h_crit_out).squeeze(-1)
        return {
            "logp": logp.view(N, M),
            "entropy": entropy.view(N, M),
            "value": value,
            "hidden_actor": h_act_out.view(N, M, self.d),
            "hidden_critic": h_crit_out,
        }
