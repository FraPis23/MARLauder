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
    guidepost_nbr_bias [N, M, K]   next-hop direction toward the analytic global target
    prev_action      [N, M, K]

Actor: shared GATEncoder, gather curr_emb + nbr_embs per (env, agent), GRUCell,
PointerHead. The high-level target is chosen analytically by the env (graph_lattice);
the actor is fed its first-hop direction (guidepost_nbr_bias) and does local control.
Decentralized: each agent sees only its own graph.

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

F_IN = 6
K = 8
# CTDE critic-only global state (value head, never seen by actors → no execution leak):
#   [explored_frac, t/max_steps, geo_pair, coverage_rate, redundancy, tgt_dist, idle_frac, imbalance].
#   1-3 fix non-stationarity + agent geometry; 4-8 (O2) add the team-coordination signal the actors
#   can't observe — redundancy in particular lets the critic explain the privileged novel_scan's
#   ~union drops → lower advantage variance. All ∈[0,1]. Built env-side in explorer._refresh_obs.
CRITIC_GLOBAL_DIM = 8


class PointerHead(nn.Module):
    """Pointer over K=8 neighbors. The coordination signal reaches the actor via the
    analytic guidepost next-hop direction (concatenated into the GRU input), NOT via an
    additive logit bias.
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
        # Finite large-negative (fp16-safe -1e4): keeps Categorical valid under fp16 / NaN-prone
        # late training. softmax(-1e4)≈0 → masked slots get ~0 prob, identical to -inf for
        # sampling/argmax. A fully-masked row → all NEG → uniform softmax (no NaN). nan_to_num
        # sanitizes any NaN/inf leaked from the encoder before masking.
        NEG = -1.0e4
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0e4, neginf=NEG)
        scores = scores.masked_fill(~mask_eff, NEG)
        return scores


class MarlActorCritic(nn.Module):
    def __init__(self, n_agents: int = 1, d: int = 128, n_heads: int = 4, n_layers: int = 2,
                 strategic_gate_eps: float = 0.0, use_gru: bool = True) -> None:
        super().__init__()
        self.M = n_agents
        self.d = d
        # GRU ablation switch. use_gru=False bypasses both recurrent cells: the actor query is the
        # feed-forward actor_in and the critic input is the feed-forward joint, with NO temporal
        # memory across steps. The GRUCell modules are still built (so their params exist for
        # checkpoint compatibility) but never called. Set at construction from cfg.use_gru.
        self.use_gru = bool(use_gru)
        # Inspector hook: when True, act() stashes the pointer-attention logits + the (purely
        # visual) guidepost first-hop direction into self._dbg_logits. The guidepost is NOT
        # added to the logits; the model must learn to follow it via node_feat[5] / the actor-
        # input direction. Off in training → zero cost.
        self.store_logit_components = False
        self._dbg_logits: dict | None = None
        # High-level guidepost GATE. The analytic global target is meant to be HIGH-LEVEL: a
        # distant frontier consulted only when the local ego window is exhausted. When
        # strategic_gate_eps > 0, the guidepost next-hop direction in the actor input is zeroed
        # per-agent on any step where max utility inside the ego window ≥ eps — i.e. while the
        # local GAT still sees nearby gain the actor climbs it greedily, and the analytic target
        # only steers the actor once no local utility remains. eps == 0 disables the gate (the
        # guidepost always influences). Mutable attribute so the driver/sweep can set it.
        self.strategic_gate_eps = float(strategic_gate_eps)
        self.encoder = GATEncoder(in_dim=F_IN, d=d, n_heads=n_heads, n_layers=n_layers)
        # Actor input = (curr_emb || prev_action[K]) → d. The analytic-target next-hop one-hot
        # was REMOVED: it handed the planner's decision to the actor in action space, inviting it
        # to just copy the guidepost. The route still reaches the actor as context via node_feat[5]
        # (guidepost path ribbon) + node_feat[2] (utility), processed by the GAT.
        self.actor_pre = nn.Linear(d + K, d)
        self.gru_actor = nn.GRUCell(d, d)
        self.pointer = PointerHead(d)
        # Critic head (CTDE)
        self.critic_pre = nn.Sequential(nn.Linear(n_agents * d + CRITIC_GLOBAL_DIM, d), nn.GELU())
        self.gru_critic = nn.GRUCell(d, d)
        self.critic_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

        # MAPPO paper Tab.7 — orthogonal init for every Linear/GRUCell, then
        # override the value head at unit gain.
        apply_orthogonal(self)
        orthogonal_(self.critic_head[-1], gain=1.0)         # V(s) output

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

    def _critic_in(self, curr_emb_per_agent: torch.Tensor, critic_global: torch.Tensor | None) -> torch.Tensor:
        """Concat joint agent embeddings [N, M·d] with the CTDE global vector [N, CRITIC_GLOBAL_DIM]
        (zeros if absent — back-compat / M=1)."""
        N = curr_emb_per_agent.shape[0]
        joint = curr_emb_per_agent.reshape(N, self.M * self.d)
        if critic_global is None:
            critic_global = joint.new_zeros(N, CRITIC_GLOBAL_DIM)
        return torch.cat([joint, critic_global], dim=-1)

    def _strategic_gate(self, obs: dict, B: int) -> torch.Tensor | None:
        """High-level gate: 1 where the ego window has NO local utility (→ follow the analytic
        guidepost), 0 where local gain remains (→ pure local GAT). Returns [B, 1] float, or None
        when disabled (strategic_gate_eps <= 0). Deterministic from obs, so act/evaluate agree.
        """
        if self.strategic_gate_eps <= 0.0:
            return None
        # feat[2] = utility on the windowed node_feat the model actually sees.
        util = self._flatten_nm(obs["node_feat"])[..., 2]          # [B, N_max_window]
        local_max = util.max(dim=-1).values                        # [B]
        return (local_max < self.strategic_gate_eps).float().unsqueeze(-1)   # [B, 1]

    # ---------------- actor ----------------
    def _actor_in(
        self,
        curr_emb: torch.Tensor,             # [B, d]
        prev_action: torch.Tensor,          # [B, K] one-hot
    ) -> torch.Tensor:
        """Build the actor GRU input: curr_emb || prev_action."""
        return self.actor_pre(torch.cat([curr_emb, prev_action], dim=-1))

    def _step_actor(self, actor_in: torch.Tensor, h_in: torch.Tensor) -> torch.Tensor:
        """Recurrent actor step, or feed-forward passthrough when use_gru is False (ablation)."""
        return self.gru_actor(actor_in, h_in) if self.use_gru else actor_in

    def _step_critic(self, joint: torch.Tensor, h_in: torch.Tensor) -> torch.Tensor:
        """Recurrent critic step, or feed-forward passthrough when use_gru is False (ablation)."""
        return self.gru_critic(joint, h_in) if self.use_gru else joint

    def act(
        self,
        obs: dict,
        hidden_actor: torch.Tensor,        # [N, M, d]
        hidden_critic: torch.Tensor,       # [N, d]
        deterministic: bool = False,
    ) -> dict:
        N, M = obs["curr_idx"].shape
        B = N * M
        h_all, curr_emb, nbr_embs = self._encode(obs)
        actor_in = self._actor_in(curr_emb, obs["prev_action"].reshape(B, K))
        h_act_in = hidden_actor.reshape(B, self.d)
        h_act_out = self._step_actor(actor_in, h_act_in)              # [B, d]
        pointer_logits = self.pointer(h_act_out, nbr_embs, obs["action_mask"].reshape(B, K))
        logits = pointer_logits
        # Inspector: stash the pointer logits + per-layer GAT neighbor attention.
        if self.store_logit_components:
            enc_attn = None
            if getattr(self.encoder, "last_attn", None) is not None:
                enc_attn = [a.view(N, M, a.shape[-3], a.shape[-2], a.shape[-1])
                            for a in self.encoder.last_attn]
            self._dbg_logits = {
                "pointer":   pointer_logits.detach().view(N, M, K),   # GAT pointer attention score
                "enc_attn":  enc_attn,                                 # GAT per-layer neighbor attention
            }
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
        joint = self.critic_pre(self._critic_in(curr_emb_per_agent, obs.get("critic_global")))
        h_crit_out = self._step_critic(joint, hidden_critic)
        value = self.critic_head(h_crit_out).squeeze(-1)

        return {
            "action": action.view(N, M),
            "logp": logp.view(N, M),
            "entropy": entropy.view(N, M),
            "value": value,
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
        B = N * M
        _, curr_emb, nbr_embs = self._encode(obs)
        actor_in = self._actor_in(curr_emb, obs["prev_action"].reshape(B, K))
        h_act_in = hidden_actor.reshape(B, self.d)
        h_act_out = self._step_actor(actor_in, h_act_in)
        logits = self.pointer(h_act_out, nbr_embs, obs["action_mask"].reshape(B, K))
        # Guard: keep logits finite so Categorical never sees an all-(-inf)/NaN row.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(action.reshape(B))
        entropy = dist.entropy()

        curr_emb_per_agent = curr_emb.view(N, M, self.d)
        joint = self.critic_pre(self._critic_in(curr_emb_per_agent, obs.get("critic_global")))
        h_crit_out = self._step_critic(joint, hidden_critic)
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
        prev_action: torch.Tensor,     # [N, M, K=8] one-hot
        critic_global: torch.Tensor | None = None,       # [N, CRITIC_GLOBAL_DIM] CTDE value-only state
    ) -> dict:
        """One PPO-evaluate step given pre-encoded curr/nbr embeddings."""
        N, M, _ = curr_emb.shape
        B = N * M
        curr_emb_flat = curr_emb.reshape(B, self.d)
        actor_in = self._actor_in(curr_emb_flat, prev_action.reshape(B, K))
        h_act_in = hidden_actor.reshape(B, self.d)
        h_act_out = self._step_actor(actor_in, h_act_in)
        logits = self.pointer(h_act_out, nbr_embs.reshape(B, K, self.d),
                              action_mask.reshape(B, K))
        # Guard: keep logits finite so Categorical never sees an all-(-inf)/NaN row.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(action.reshape(B))
        entropy = dist.entropy()

        joint = self.critic_pre(self._critic_in(curr_emb, critic_global))
        h_crit_out = self._step_critic(joint, hidden_critic)
        value = self.critic_head(h_crit_out).squeeze(-1)
        return {
            "logp": logp.view(N, M),
            "entropy": entropy.view(N, M),
            "value": value,
            "hidden_actor": h_act_out.view(N, M, self.d),
            "hidden_critic": h_crit_out,
        }
