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
from models.init_utils import apply_orthogonal, orthogonal_

F_IN = 7
K = 8
CAND_FEAT_DIM = 8     # rel_x, rel_y, utility, euclid, min_team_dist, max_comm_gap, own_minus_team, team_alt_score


class PointerHead(nn.Module):
    """Phase A v2: pointer over K=8 neighbors. No guidepost-bias hijack.

    Coordination signal is now injected via the strategic head's `strategic_emb`
    which is concatenated into the actor's GRU input, NOT via additive logit bias.
    """
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.d = d

    def forward(
        self,
        query: torch.Tensor,           # [B, d]
        keys: torch.Tensor,            # [B, K, d]
        mask: torch.Tensor,            # [B, K] bool (True = valid)
    ) -> torch.Tensor:
        q = self.q(query).unsqueeze(1)
        k = self.k(keys)
        scores = (q * k).sum(dim=-1) / math.sqrt(self.d)            # [B, K]
        any_valid = mask.any(dim=-1, keepdim=True)
        mask_eff = torch.where(any_valid.expand_as(mask), mask, torch.ones_like(mask))
        # Finite large-negative (fp16-safe -1e4, matches StrategicHead): keeps Categorical
        # valid under fp16 / NaN-prone late training. softmax(-1e4)≈0 → masked slots get ~0
        # prob, identical to -inf for sampling/argmax (proven: fp32 prob diff 0.0). A
        # fully-masked row → all NEG → uniform softmax (no NaN). nan_to_num sanitizes any
        # NaN/inf leaked from the encoder before masking.
        NEG = -1.0e4
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0e4, neginf=NEG)
        scores = scores.masked_fill(~mask_eff, NEG)
        return scores


class StrategicHead(nn.Module):
    """Phase A v2 / A2: attention over top-K frontier candidates per agent.

    Returns:
      strategic_emb  [B, d]      — pooled candidate embedding (one-hot mix at forward,
                                    softmax mix at backward via straight-through trick).
      target_logits  [B, K_cand] — per-candidate score (raw).
      target_onehot  [B, K_cand] — STE one-hot used to pool kv. Differentiable wrt logits.

    Coordination is learned: candidate features include teammate-distance and
    comm-gap; the head learns to down-weight candidates near teammates.
    """
    def __init__(self, d: int, n_heads: int = 4, cand_feat_dim: int = CAND_FEAT_DIM) -> None:
        super().__init__()
        self.cand_proj = nn.Linear(cand_feat_dim, d)
        self.q_proj    = nn.Linear(d, d)
        self.mha       = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.score     = nn.Linear(d, 1)
        self.d = d

    def forward(
        self,
        curr_emb: torch.Tensor,           # [B, d]
        cand_feat: torch.Tensor,          # [B, K_cand, cand_feat_dim]
        cand_valid: torch.Tensor,         # [B, K_cand] bool
        gumbel_tau: float = 1.0,
        stored_choice: torch.Tensor | None = None,   # [B] long; if given, use as hard pick
        sample: bool = True,                          # False at eval → argmax (no Gumbel noise)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K_cand, _ = cand_feat.shape
        q  = self.q_proj(curr_emb).unsqueeze(1)                          # [B, 1, d]
        kv = self.cand_proj(cand_feat)                                   # [B, K_cand, d]
        any_valid = cand_valid.any(dim=-1, keepdim=True)                 # [B, 1]
        pad_mask  = ~cand_valid                                          # [B, K_cand]
        pad_mask  = torch.where(any_valid.expand_as(pad_mask), pad_mask, torch.zeros_like(pad_mask))
        attn_out, _ = self.mha(q, kv, kv, key_padding_mask=pad_mask)     # [B, 1, d]
        attn_out = attn_out.squeeze(1)                                    # [B, d]
        logits = self.score(kv + attn_out.unsqueeze(1)).squeeze(-1)       # [B, K_cand]
        # Half-safe sentinel (-65504 underflows fp16 ε of -inf already).
        NEG = -1.0e4
        logits = logits.masked_fill(~cand_valid, NEG)

        # Compute soft & hard one-hot. STE: forward=hard, backward=soft.
        # safe_logits: replace all-NEG rows (no valid cand) with zeros so softmax doesn't NaN.
        any_finite = (logits > NEG / 2).any(dim=-1, keepdim=True)
        safe_logits = torch.where(any_finite.expand_as(logits), logits, torch.zeros_like(logits))
        if sample:
            soft = torch.nn.functional.gumbel_softmax(safe_logits, tau=gumbel_tau, hard=False, dim=-1)
        else:
            soft = torch.softmax(safe_logits, dim=-1)
        if stored_choice is not None:
            hard = torch.nn.functional.one_hot(stored_choice.clamp(min=0), num_classes=K_cand).float()
        else:
            hard = torch.nn.functional.one_hot(soft.argmax(dim=-1), num_classes=K_cand).float()
        # Straight-through: hard at forward, gradient flows through soft.
        target_onehot = (hard - soft).detach() + soft                    # [B, K_cand]
        # Pool kv by one-hot mix.
        strategic_emb = (target_onehot.unsqueeze(-1) * kv).sum(dim=1)    # [B, d]
        strategic_emb = strategic_emb * any_valid.float()
        return strategic_emb, logits, target_onehot


class MarlActorCritic(nn.Module):
    def __init__(self, n_agents: int = 1, d: int = 128, n_heads: int = 4, n_layers: int = 2,
                 gumbel_tau: float = 1.0, path_bias_floor: float = 1.5) -> None:
        super().__init__()
        self.M = n_agents
        self.d = d
        self.gumbel_tau = gumbel_tau           # mutable; trainer anneals via attribute assignment
        self.encoder = GATEncoder(in_dim=F_IN, d=d, n_heads=n_heads, n_layers=n_layers)
        # Phase A v2: strategic head over top-K=16 candidates.
        self.strategic_head = StrategicHead(d, n_heads=n_heads, cand_feat_dim=CAND_FEAT_DIM)
        # Actor input = (curr_emb || strategic_emb || next_hop_dir_onehot[K] || prev_action[K]) → d
        self.actor_pre = nn.Linear(2 * d + 2 * K, d)
        self.gru_actor = nn.GRUCell(d, d)
        self.pointer = PointerHead(d)
        # I.3 — bias on action logits toward BF first-hop of strategic target.
        # effective = path_bias_floor (fixed) + softplus(path_bias_learn). The floor keeps a
        # persistent prior toward the model's own chosen target so the grid-utility pointer
        # signal cannot fully suppress target-following; learnable part adds headroom.
        self.register_buffer("path_bias_floor", torch.tensor(float(path_bias_floor)), persistent=False)
        self.path_bias_learn = nn.Parameter(torch.tensor(0.5))
        # Critic head (CTDE)
        self.critic_pre = nn.Sequential(nn.Linear(n_agents * d, d), nn.GELU())
        self.gru_critic = nn.GRUCell(d, d)
        self.critic_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        # Precomputed K_INDEX_TABLE for analytic next-hop slot from (sign(dy), sign(dx)).
        # Matches env.graph_lattice.NBR_OFFSETS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
        # Layout: kdir_table[dr+1, dc+1] = K-slot index in [0,7] for (dr, dc) ∈ {-1,0,1}².
        # Center (0, 0) → -1 (no movement).
        _t = torch.tensor([
            [0, 1, 2],     # dr=-1
            [3, -1, 4],    # dr= 0
            [5, 6, 7],     # dr= 1
        ], dtype=torch.long)
        self.register_buffer("_kdir_table", _t, persistent=False)

        # MAPPO paper Tab.7 — orthogonal init for every Linear/GRUCell, then
        # override output heads: policy logits near-uniform (small gain) for
        # healthy initial exploration; value head at unit gain.
        apply_orthogonal(self)
        orthogonal_(self.strategic_head.score, gain=0.01)   # strategic target logits
        orthogonal_(self.critic_head[-1], gain=1.0)         # V(s) output

    @property
    def path_bias(self) -> torch.Tensor:
        """I.3 — effective path-following bias: fixed floor + non-negative learnable extra."""
        return self.path_bias_floor + torch.nn.functional.softplus(self.path_bias_learn)

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
    def _next_hop_onehot(self, target_xy: torch.Tensor, pos: torch.Tensor, nr: float) -> torch.Tensor:
        """Analytic K=8 next-hop direction from agent pos toward chosen target xy.

        Quantize (target_xy - pos) / nr to {-1, 0, +1} per axis, look up K-slot,
        return one-hot [B, K=8]. When target_xy == pos (no movement direction), returns
        zeros (no slot lit) — bridge gap: don't bias toward any neighbor when at target.
        """
        diff = (target_xy - pos) / max(1.0, nr)                          # [B, 2]
        # Quantize to {-1, 0, +1} via threshold 0.5.
        sx = torch.where(diff[..., 0] >  0.5, torch.full_like(diff[..., 0], 1.0),
              torch.where(diff[..., 0] < -0.5, torch.full_like(diff[..., 0], -1.0),
                          torch.zeros_like(diff[..., 0])))
        sy = torch.where(diff[..., 1] >  0.5, torch.full_like(diff[..., 1], 1.0),
              torch.where(diff[..., 1] < -0.5, torch.full_like(diff[..., 1], -1.0),
                          torch.zeros_like(diff[..., 1])))
        dr = sy.long() + 1                                                # [B] in {0,1,2}
        dc = sx.long() + 1
        kidx = self._kdir_table[dr, dc]                                    # [B] in {-1..7}
        out = torch.zeros(target_xy.shape[0], K, dtype=target_xy.dtype, device=target_xy.device)
        valid = kidx >= 0
        safe_k = kidx.clamp(min=0)
        out.scatter_(1, safe_k.unsqueeze(-1), valid.float().unsqueeze(-1))
        return out

    def _strategic_and_actor_in(
        self,
        curr_emb: torch.Tensor,          # [B, d]
        cand_feat: torch.Tensor,         # [B, K_cand, F_cand]
        cand_valid: torch.Tensor,        # [B, K_cand]
        cand_xy: torch.Tensor,           # [B, K_cand, 2]
        pos: torch.Tensor,               # [B, 2]
        prev_action: torch.Tensor,       # [B, K=8] one-hot
        nr: float,
        stored_choice: torch.Tensor | None = None,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run strategic head + compute actor input. Returns actor_in, target_logits, target_idx."""
        strategic_emb, target_logits, target_onehot = self.strategic_head(
            curr_emb, cand_feat, cand_valid,
            gumbel_tau=self.gumbel_tau,
            stored_choice=stored_choice,
            sample=sample,
        )
        target_xy_chosen = (target_onehot.unsqueeze(-1) * cand_xy).sum(dim=1)   # [B, 2]
        next_hop_onehot = self._next_hop_onehot(target_xy_chosen, pos, nr)      # [B, K=8]
        actor_in = self.actor_pre(torch.cat(
            [curr_emb, strategic_emb, next_hop_onehot, prev_action], dim=-1))
        target_idx = target_onehot.argmax(dim=-1)
        return actor_in, target_logits, target_idx

    def act(
        self,
        obs: dict,
        hidden_actor: torch.Tensor,        # [N, M, d]
        hidden_critic: torch.Tensor,       # [N, d]
        deterministic: bool = False,
        nr: float = 16.0,
    ) -> dict:
        N, M = obs["curr_idx"].shape
        B = N * M
        h_all, curr_emb, nbr_embs = self._encode(obs)
        actor_in, target_logits, target_idx = self._strategic_and_actor_in(
            curr_emb,
            obs["cand_feat"].reshape(B, -1, CAND_FEAT_DIM),
            obs["cand_valid"].reshape(B, -1),
            obs["cand_xy"].reshape(B, -1, 2),
            obs["pos"].reshape(B, 2),
            obs["prev_action"].reshape(B, K),
            nr=nr,
            stored_choice=None,
            sample=(not deterministic),
        )
        h_act_in = hidden_actor.reshape(B, self.d)
        h_act_out = self.gru_actor(actor_in, h_act_in)              # [B, d]
        logits = self.pointer(h_act_out, nbr_embs, obs["action_mask"].reshape(B, K))
        # G.3.c — soft bias toward BF first-hop of strategic pick.
        cand_bf_first_hop = obs.get("cand_bf_first_hop")
        if cand_bf_first_hop is not None:
            cbfh = cand_bf_first_hop.reshape(B, -1, K)                                 # [B, K_cand, K=8]
            chosen_hop = torch.gather(
                cbfh, dim=1, index=target_idx.view(B, 1, 1).expand(-1, 1, K)
            ).squeeze(1)                                                                # [B, K=8]
            logits = logits + self.path_bias * chosen_hop
        # Guard: keep logits finite so Categorical never sees an all-(-inf)/NaN row.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()

        curr_emb_per_agent = curr_emb.view(N, M, self.d)
        joint = curr_emb_per_agent.reshape(N, M * self.d)
        joint = self.critic_pre(joint)
        h_crit_out = self.gru_critic(joint, hidden_critic)
        value = self.critic_head(h_crit_out).squeeze(-1)

        K_cand = target_logits.shape[-1]
        return {
            "action": action.view(N, M),
            "logp": logp.view(N, M),
            "entropy": entropy.view(N, M),
            "value": value,
            "logits": logits.view(N, M, K),
            "hidden_actor": h_act_out.view(N, M, self.d),
            "hidden_critic": h_crit_out,
            "target_choice": target_idx.view(N, M),                  # K-slot in cand list
            "target_logits": target_logits.view(N, M, K_cand),
        }

    # ---------------- evaluate (for PPO update) ----------------
    def evaluate(
        self,
        obs: dict,
        action: torch.Tensor,              # [N, M]
        hidden_actor: torch.Tensor,        # [N, M, d]
        hidden_critic: torch.Tensor,       # [N, d]
        stored_choice: torch.Tensor | None = None,   # [N, M] long, target K-slot from rollout
        nr: float = 16.0,
    ) -> dict:
        N, M = obs["curr_idx"].shape
        B = N * M
        _, curr_emb, nbr_embs = self._encode(obs)
        sc = stored_choice.reshape(B) if stored_choice is not None else None
        actor_in, _, _ = self._strategic_and_actor_in(
            curr_emb,
            obs["cand_feat"].reshape(B, -1, CAND_FEAT_DIM),
            obs["cand_valid"].reshape(B, -1),
            obs["cand_xy"].reshape(B, -1, 2),
            obs["pos"].reshape(B, 2),
            obs["prev_action"].reshape(B, K),
            nr=nr,
            stored_choice=sc,
            sample=False,
        )
        h_act_in = hidden_actor.reshape(B, self.d)
        h_act_out = self.gru_actor(actor_in, h_act_in)
        logits = self.pointer(h_act_out, nbr_embs, obs["action_mask"].reshape(B, K))
        # G.3.c — soft path bias on action logits, indexed by stored strategic pick.
        cand_bf_first_hop = obs.get("cand_bf_first_hop")
        if cand_bf_first_hop is not None and sc is not None:
            cbfh = cand_bf_first_hop.reshape(B, -1, K)
            chosen_hop = torch.gather(cbfh, dim=1, index=sc.view(B, 1, 1).expand(-1, 1, K)).squeeze(1)
            logits = logits + self.path_bias * chosen_hop
        # Guard: keep logits finite so Categorical never sees an all-(-inf)/NaN row.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(action.reshape(B))
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
        cand_feat: torch.Tensor,       # [N, M, K_cand, CAND_FEAT_DIM]
        cand_valid: torch.Tensor,      # [N, M, K_cand]
        cand_xy: torch.Tensor,         # [N, M, K_cand, 2]
        pos: torch.Tensor,             # [N, M, 2]
        prev_action: torch.Tensor,     # [N, M, K=8] one-hot
        stored_choice: torch.Tensor,   # [N, M] long — target K-slot stored at rollout
        cand_bf_first_hop: torch.Tensor | None = None,   # G.3.c [N, M, K_cand, K]
        nr: float = 16.0,
    ) -> dict:
        """One PPO-evaluate step given pre-encoded curr/nbr embeddings.

        Strategic head replays Gumbel-ST with the stored hard pick (STE forward = stored
        one-hot, backward = current softmax). Strategic head gradient flows through this.
        """
        N, M, _ = curr_emb.shape
        B = N * M
        curr_emb_flat = curr_emb.reshape(B, self.d)
        actor_in, _, _ = self._strategic_and_actor_in(
            curr_emb_flat,
            cand_feat.reshape(B, -1, CAND_FEAT_DIM),
            cand_valid.reshape(B, -1),
            cand_xy.reshape(B, -1, 2),
            pos.reshape(B, 2),
            prev_action.reshape(B, K),
            nr=nr,
            stored_choice=stored_choice.reshape(B),
            sample=False,                # no Gumbel noise at PPO update
        )
        h_act_in = hidden_actor.reshape(B, self.d)
        h_act_out = self.gru_actor(actor_in, h_act_in)
        logits = self.pointer(h_act_out, nbr_embs.reshape(B, K, self.d),
                              action_mask.reshape(B, K))
        # G.3.c — soft path bias using stored strategic pick.
        if cand_bf_first_hop is not None:
            cbfh = cand_bf_first_hop.reshape(B, -1, K)
            sc = stored_choice.reshape(B)
            chosen_hop = torch.gather(cbfh, dim=1, index=sc.view(B, 1, 1).expand(-1, 1, K)).squeeze(1)
            logits = logits + self.path_bias * chosen_hop
        # Guard: keep logits finite so Categorical never sees an all-(-inf)/NaN row.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
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
