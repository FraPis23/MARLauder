"""Reti MAPPO per MARLauder: actor pointer-attention + critic centralizzato permutation-invariant.

Actor (per-agente, parameter-sharing):
  token = nodi ego-lattice (KK) + anchor globali (A) + compagni (M), con canale-livello e padding-mask.
  encoder self-attention -> dal nodo CENTRALE, pointer-attention sugli 8 vicini -> 8 logit + STAY.
Critic (centralizzato, M-agnostico):
  riassunto per agente (pool nodi ego validi) -> attention sugli agenti (permutation-invariant) ->
  valore per-agente [N,M].

L'osservazione arriva dal MarlExploreEnv (tensori GPU [N,M,...]).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_DIR = 8
N_ACT = 9          # 8 direzioni + STAY
TOK_FEAT = 7       # [rel_x, rel_y, utility_norm, guidepost, lvl0, lvl1, lvl2]
LVL_EGO, LVL_ANCHOR, LVL_MATE = 0, 1, 2

# offset grid 8 vicini (di, dj) — stesso ordine di env.graph_lattice._DIRS
_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class MarlActorCritic(nn.Module):
    def __init__(self, K: int = 21, a_max: int = 64, n_mates: int = 4,
                 d: int = 128, n_head: int = 8, n_layer: int = 2,
                 coord_scale: float = 1.0 / 1000.0, util_scale: float = 1.0 / 60.0,
                 guide_scale: float = 1.0):
        super().__init__()
        self.K, self.KK = K, K * K
        self.a_max = a_max
        self.coord_scale, self.util_scale, self.guide_scale = coord_scale, util_scale, guide_scale
        self.kc = (K // 2) * K + (K // 2)
        nk = [(K // 2 + di) * K + (K // 2 + dj) for (di, dj) in _DIRS]
        self.register_buffer("neighbor_idx", torch.tensor(nk, dtype=torch.long))

        self.embed = nn.Linear(TOK_FEAT, d)
        enc = nn.TransformerEncoderLayer(d, n_head, dim_feedforward=4 * d,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, n_layer)

        # pointer
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.stay_head = nn.Linear(d, 1)
        self.d = d

        # critic
        self.agent_attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.value_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    # ---------- costruzione token ----------
    def _tokens(self, obs: dict):
        coords = obs["coords"]                 # [N,M,KK,2]
        valid = obs["valid"].bool()            # [N,M,KK]
        util = obs["utility"].float()          # [N,M,KK]
        guide = obs["guidepost"].float()       # [N,M,KK]
        pos = obs["pos"]                       # [N,M,2]
        anchors = obs["anchors"]               # [N,A,2]
        amask = obs["anchor_mask"].bool()      # [N,A]
        mate_rel = obs["teammate_rel"]         # [N,M,M,2]
        mate_known = obs["teammate_known"].bool()  # [N,M,M]
        N, M, KK, _ = coords.shape
        A = anchors.shape[1]
        B = N * M
        dev = coords.device

        def feat(rel, u, g, lvl, mask):
            t = torch.zeros((*rel.shape[:-1], TOK_FEAT), device=dev)
            t[..., 0:2] = rel * self.coord_scale
            t[..., 2] = u * self.util_scale
            t[..., 3] = g * self.guide_scale   # guidepost [-1,1]
            t[..., 4 + lvl] = 1.0
            return t, mask

        # ego
        ego_rel = coords - pos.unsqueeze(2)                          # [N,M,KK,2]
        ego_t, ego_m = feat(ego_rel, util, guide, LVL_EGO, valid)
        # anchor (per env -> per agente)
        anc = anchors.unsqueeze(1).expand(N, M, A, 2)
        anc_rel = anc - pos.unsqueeze(2)
        _z_a = torch.zeros((N, M, A), device=dev)
        anc_t, anc_m = feat(anc_rel, _z_a, _z_a, LVL_ANCHOR, amask.unsqueeze(1).expand(N, M, A))
        # compagni (escludo se stesso via known gia' True su diagonale; lo manteniamo come token "self")
        _z_m = torch.zeros((N, M, M), device=dev)
        mate_t, mate_m = feat(mate_rel, _z_m, _z_m, LVL_MATE, mate_known)

        tokens = torch.cat([ego_t, anc_t, mate_t], dim=2).reshape(B, KK + A + M, TOK_FEAT)
        pad = ~torch.cat([ego_m, anc_m, mate_m], dim=2).reshape(B, KK + A + M)  # True = ignora
        return tokens, pad, N, M, KK, A

    # ---------- forward ----------
    def forward(self, obs: dict):
        tokens, pad, N, M, KK, A = self._tokens(obs)
        B = N * M
        h = self.embed(tokens)
        # evita righe tutte-pad (NaN in attention): garantisco il nodo centrale sempre valido
        pad = pad.clone()
        pad[:, self.kc] = False
        h = self.encoder(h, src_key_padding_mask=pad)               # [B,T,d]

        ego = h[:, :KK, :]                                          # [B,KK,d]
        h_c = ego[:, self.kc, :]                                    # [B,d] nodo centrale
        neigh = ego[:, self.neighbor_idx, :]                        # [B,8,d]

        q = self.q_proj(h_c).unsqueeze(1)                          # [B,1,d]
        k = self.k_proj(neigh)                                     # [B,8,d]
        ptr_logits = (q @ k.transpose(1, 2)).squeeze(1) / math.sqrt(self.d)  # [B,8]
        stay = self.stay_head(h_c)                                 # [B,1]
        logits = torch.cat([ptr_logits, stay], dim=-1)            # [B,9]

        amask = obs["action_mask"].bool().reshape(B, N_ACT)
        logits = logits.masked_fill(~amask, float("-inf"))

        # critic: riassunto per agente (mean sui nodi ego validi)
        ego_valid = obs["valid"].bool().reshape(B, KK).float()
        denom = ego_valid.sum(-1, keepdim=True).clamp(min=1.0)
        summ = (ego * ego_valid.unsqueeze(-1)).sum(1) / denom       # [B,d]
        summ = summ.reshape(N, M, self.d)
        glob, _ = self.agent_attn(summ, summ, summ)                # [N,M,d] permutation-invariant
        value = self.value_head(torch.cat([summ, glob], dim=-1)).squeeze(-1)  # [N,M]

        return logits.reshape(N, M, N_ACT), value

    # ---------- helper RL ----------
    def act(self, obs: dict, deterministic: bool = False):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs: dict, action: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value
