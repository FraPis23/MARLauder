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
    prev_action      [N, M, K]

Actor: shared GATEncoder, gather curr_emb + nbr_embs per (env, agent), GRUCell,
PointerHead. No analytic target / guidepost: the actor steers purely from the ego-window
GAT features (local utility + the radar beyond-window channels feat[5]/feat[6]).
Decentralized: each agent sees only its own graph.

Critic (CTDE): shared GATEncoder; gather curr_emb for each agent, POOL across M
(mean ⊕ max, count-invariant) ⊕ critic_global, MLP → GRUCell → scalar V(s)  [N].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from models.gat import GATEncoder
from models.init_utils import apply_orthogonal, orthogonal_

F_IN = 7   # 0 x_rel, 1 y_rel, 2 utility, 3 age, 4 teammate_pot, 5 radar-util, 6 radar-teammate
K = 8
# CTDE critic-only global state (value head, never seen by actors → no execution leak):
#   [explored_frac, t/max_steps, geo_pair, coverage_rate, redundancy, idle_frac, imbalance].
# The pooled per-agent embeddings (mean⊕max, see _critic_in) are EGO-RELATIVE, so aggregating them
# keeps per-agent exploration CONTENT but not the team geometry — the RELATIONAL geometry the value
# head needs lives here as geo_pair (nearest-teammate GEODESIC distance /diam, translation-invariant,
# links the agents' positions: converging vs splitting). No absolute team position: V(s) for
# exploration must generalize across maps, and absolute coords would just let the critic overfit the
# training-map layouts. redundancy in particular lets the critic explain the privileged novel_scan's
# ~union drops → lower advantage variance. All ∈[0,1]. Built env-side in _refresh_obs.
CRITIC_GLOBAL_DIM = 7


def _mask_scores(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sanitize + mask K-way logits. Finite large-negative (fp16-safe -1e4): keeps Categorical
    valid under fp16 / NaN-prone late training. softmax(-1e4)≈0 → masked slots get ~0 prob,
    identical to -inf for sampling/argmax. A fully-masked row → all NEG → uniform softmax
    (no NaN). nan_to_num sanitizes any NaN/inf leaked from the encoder before masking."""
    any_valid = mask.any(dim=-1, keepdim=True)
    mask_eff = torch.where(any_valid.expand_as(mask), mask, torch.ones_like(mask))
    NEG = -1.0e4
    scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0e4, neginf=NEG)
    return scores.masked_fill(~mask_eff, NEG)


class PointerHead(nn.Module):
    """Pointer over K=8 neighbors: logit_k = (q·k_k)/sqrt(d)·τ + w_vf·vf_k. Beyond-window steering
    reaches the neighbor embeddings through the GAT radar channels (feat[5]/feat[6]); the analytic
    VALUE-FIELD enters as a per-neighbor logit bias (w_vf learnable, init 1 → the field shapes the
    policy from step 0; the net can amplify or unlearn it)."""
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.d = d
        # Learnable temperature (mirrors GAT A1). The fixed 1/sqrt(d) scaling with d=128 (sqrt≈11.3)
        # drove q·k so small that the pointer softmax stayed near-uniform → the agent picked
        # essentially at random among valid neighbors → wander / ping-pong loops. τ = exp(log_tau)
        # clamped to [0.1, 10] lets the head sharpen its neighbor ranking without runaway.
        # Init 0 → τ=1 → identity at start (back-compat: old ckpts miss this param, load fresh at 1).
        self.log_tau = nn.Parameter(torch.zeros(1))
        # Value-field logit-bias gain. vf ∈ [0,1] per neighbor → init 1.0 gives a meaningful
        # (but overridable) prior toward the highest-value branch from the first update.
        self.w_vf = nn.Parameter(torch.ones(1))

    def forward(
        self,
        query: torch.Tensor,           # [B, d]
        keys: torch.Tensor,            # [B, K, d]
        mask: torch.Tensor,            # [B, K] bool (True = valid)
        vf: torch.Tensor | None = None,  # [B, K] value-field ∈[0,1] (None = no bias)
    ) -> torch.Tensor:
        q = self.q(query).unsqueeze(1)
        k = self.k(keys)
        scores = (q * k).sum(dim=-1) / math.sqrt(self.d)            # [B, K]
        scores = scores * self.log_tau.exp().clamp(0.1, 10.0)       # A1-style learnable sharpening
        if vf is not None:
            scores = scores + self.w_vf * vf
        return _mask_scores(scores, mask)


class MarlActorCritic(nn.Module):
    def __init__(self, n_agents: int = 1, d: int = 128, n_heads: int = 4, n_layers: int = 2,
                 use_gru: bool = True, gat_actor: bool = True, gat_critic: bool = True) -> None:
        super().__init__()
        self.M = n_agents
        self.d = d
        # GAT ablation switches.
        #   gat_actor=False  → actor blind to the GAT: curr_emb replaced by zeros and the pointer
        #     (q·k over GAT neighbor embeddings) replaced by actor_head(h) + w_vf_direct·vf — the
        #     policy steers from the analytic value-field + prev_action + agent_scalars only.
        #   gat_critic=False → critic drops the GAT too: pooled per-agent embedding becomes
        #     critic_feat_emb (masked mean⊕max of RAW window node features, projected to d).
        #   Both False (--no-gat) → the encoder is NEVER RUN: full GAT removal, big speed/VRAM win.
        self.gat_actor = bool(gat_actor)
        self.gat_critic = bool(gat_critic)
        self.use_encoder = self.gat_actor or self.gat_critic
        # GRU ablation switch. use_gru=False bypasses both recurrent cells: the actor query is the
        # feed-forward actor_in and the critic input is the feed-forward joint, with NO temporal
        # memory across steps. The GRUCell modules are still built (so their params exist for
        # checkpoint compatibility) but never called. Set at construction from cfg.use_gru.
        self.use_gru = bool(use_gru)
        # Inspector hook: when True, act() stashes the pointer-attention logits into self._dbg_logits.
        # Off in training → zero cost.
        self.store_logit_components = False
        self._dbg_logits: dict | None = None
        self.encoder = GATEncoder(in_dim=F_IN, d=d, n_heads=n_heads, n_layers=n_layers)
        # Actor input = (curr_emb || prev_action[K] || agent_scalars[2]) → d. agent_scalars =
        # [∆M surplus-gate, staleness] (env-computed, execution-decentralized) let the policy DECIDE
        # when to rendezvous. Beyond-window spatial context still reaches the actor through the
        # GAT-processed node features (utility feat[2] + radar feat[5]/feat[6]).
        self.n_agent_scalars = 2
        # + K: the value-field [B, K] enters the actor trunk too (context for the GRU/MLP),
        # in addition to its per-neighbor logit bias in the pointer / actor_head.
        self.actor_pre = nn.Linear(d + K + self.n_agent_scalars + K, d)
        self.gru_actor = nn.GRUCell(d, d)
        self.pointer = PointerHead(d)
        # VF-only actor head (gat_actor=False): K logits from the trunk + direct vf bias.
        # Built unconditionally so both arms share one state_dict layout (warm-start across arms).
        self.actor_head = nn.Linear(d, K)
        self.w_vf_direct = nn.Parameter(torch.ones(1))
        # GAT-free critic embedding (gat_critic=False): masked mean⊕max of the raw window node
        # features → d. Built unconditionally (same state_dict either way).
        self.critic_feat_proj = nn.Linear(2 * F_IN, d)
        # Critic head (CTDE). Input = pooled team summary (mean ⊕ max over agents = 2·d, see
        # _critic_in) ⊕ CTDE global vector. The 2·d width is independent of n_agents → the critic
        # is count-invariant (unlocks IR2-style variable M without reshaping this weight).
        self.critic_pre = nn.Sequential(nn.Linear(2 * d + CRITIC_GLOBAL_DIM, d), nn.GELU())
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
        """Pool the per-agent embeddings [N, M, d] into a COUNT-INVARIANT team summary, then concat
        the CTDE global vector [N, CRITIC_GLOBAL_DIM] (zeros if absent — back-compat).

        Problem #1 fix. The old input concatenated the M embeddings → [N, M·d]. That is MAPPO's "CL"
        (concatenate-local) state — the worst-performing critic input in the MAPPO paper (Fig 3,
        "ineffective particularly in maps with many agents"): it bakes M into critic_pre's weight
        shape (blocks variable agent count / warm-start across M) and forces the value head to learn
        a permutation-SENSITIVE map of M redundant slots → noisy V → noisy advantages.

        Replace with symmetric pooling: mean ⊕ max over agents.
          mean_i = (1/M) Σ_a h_a     — the mean-field team state (MAPPO's recommended global term)
          max_i  = max_a h_a         — the "is ANY agent in state X" signal mean averages away
        Both are permutation-invariant and M-invariant (fixed 2·d width for any M), so the same
        critic serves any agent count. Unlike attention pooling there is NO softmax gate to get stuck
        in the uniform dead-zone we had to repair in the GAT/pointer heads — mean/max always pass
        gradient — so it learns from step 0 while sitting exactly on the proven mean-field baseline.
        (Attention-residual pooling is a later option; with M=2 it buys almost nothing.)
        """
        N = curr_emb_per_agent.shape[0]
        mean_h = curr_emb_per_agent.mean(dim=1)                     # [N, d]
        max_h = curr_emb_per_agent.max(dim=1).values               # [N, d]
        pooled = torch.cat([mean_h, max_h], dim=-1)                # [N, 2·d]
        if critic_global is None:
            critic_global = pooled.new_zeros(N, CRITIC_GLOBAL_DIM)
        return torch.cat([pooled, critic_global], dim=-1)          # [N, 2·d + G]

    # ---------------- actor ----------------
    def critic_feat_emb(self, node_feat: torch.Tensor, node_valid: torch.Tensor) -> torch.Tensor:
        """GAT-free per-agent embedding (gat_critic=False): masked mean ⊕ max of the raw window
        node features projected to d. Broadcasts over any leading dims:
        [..., N_max, F] + [..., N_max] → [..., d]."""
        nv = node_valid.unsqueeze(-1)
        cnt = nv.float().sum(dim=-2).clamp(min=1.0)
        mean = (node_feat * nv.float()).sum(dim=-2) / cnt
        mx = node_feat.masked_fill(~nv, -1.0e4).max(dim=-2).values
        mx = torch.where(node_valid.any(dim=-1, keepdim=True), mx, torch.zeros_like(mx))
        return self.critic_feat_proj(torch.cat([mean, mx], dim=-1))

    def _actor_in(
        self,
        curr_emb: torch.Tensor | None,      # [B, d] GAT embedding, or None (no-GAT-actor)
        prev_action: torch.Tensor,          # [B, K] one-hot
        agent_scalars: torch.Tensor,        # [B, 2] [∆M-gate, staleness]
        vf: torch.Tensor,                   # [B, K] value-field ∈[0,1]
    ) -> torch.Tensor:
        """Build the actor GRU input: curr_emb || prev_action || agent_scalars || value_field.
        gat_actor=False → curr_emb slot zeroed (VF-only actor)."""
        if not self.gat_actor or curr_emb is None:
            curr_emb = vf.new_zeros(vf.shape[0], self.d)
        return self.actor_pre(torch.cat([curr_emb, prev_action, agent_scalars, vf], dim=-1))

    def _actor_logits(
        self,
        h_act: torch.Tensor,                # [B, d] actor trunk output
        nbr_embs: torch.Tensor,             # [B, K, d] GAT neighbor embeddings
        mask: torch.Tensor,                 # [B, K] bool
        vf: torch.Tensor,                   # [B, K] value-field
    ) -> torch.Tensor:
        """K-way action logits. GAT arm: pointer q·k over neighbor embeddings + w_vf·vf.
        VF-only arm: actor_head(h) + w_vf_direct·vf — no GAT in the actor path."""
        if self.gat_actor:
            return self.pointer(h_act, nbr_embs, mask, vf)
        return _mask_scores(self.actor_head(h_act) + self.w_vf_direct * vf, mask)

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
        if self.use_encoder:
            h_all, curr_emb, nbr_embs = self._encode(obs)
        else:   # --no-gat: encoder never run; critic gets the raw-feature embedding instead.
            curr_emb, nbr_embs = None, None
        vf = obs["value_field"].reshape(B, K)
        actor_in = self._actor_in(curr_emb, obs["prev_action"].reshape(B, K),
                                   obs["agent_scalars"].reshape(B, self.n_agent_scalars), vf)
        h_act_in = hidden_actor.reshape(B, self.d)
        h_act_out = self._step_actor(actor_in, h_act_in)              # [B, d]
        pointer_logits = self._actor_logits(h_act_out, nbr_embs, obs["action_mask"].reshape(B, K), vf)
        logits = pointer_logits
        # Inspector: stash the pointer logits + per-layer GAT neighbor attention.
        if self.store_logit_components:
            enc_attn = None
            if getattr(self.encoder, "last_attn", None) is not None:
                enc_attn = [a.view(N, M, a.shape[-3], a.shape[-2], a.shape[-1])
                            for a in self.encoder.last_attn]
            enc_contrib = None
            if getattr(self.encoder, "last_contrib", None) is not None:
                enc_contrib = [c.view(N, M, c.shape[-2], c.shape[-1])
                               for c in self.encoder.last_contrib]
            self._dbg_logits = {
                "pointer":     pointer_logits.detach().view(N, M, K),  # GAT pointer attention score
                "enc_attn":    enc_attn,                               # GAT per-layer neighbor attention
                "enc_contrib": enc_contrib,                            # per-layer real value-contrib norm
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

        curr_emb_per_agent = (curr_emb.view(N, M, self.d) if self.gat_critic
                              else self.critic_feat_emb(obs["node_feat"], obs["node_valid"]))
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
        if self.use_encoder:
            _, curr_emb, nbr_embs = self._encode(obs)
        else:
            curr_emb, nbr_embs = None, None
        vf = obs["value_field"].reshape(B, K)
        actor_in = self._actor_in(curr_emb, obs["prev_action"].reshape(B, K),
                                   obs["agent_scalars"].reshape(B, self.n_agent_scalars), vf)
        h_act_in = hidden_actor.reshape(B, self.d)
        h_act_out = self._step_actor(actor_in, h_act_in)
        logits = self._actor_logits(h_act_out, nbr_embs, obs["action_mask"].reshape(B, K), vf)
        # Guard: keep logits finite so Categorical never sees an all-(-inf)/NaN row.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(action.reshape(B))
        entropy = dist.entropy()

        curr_emb_per_agent = (curr_emb.view(N, M, self.d) if self.gat_critic
                              else self.critic_feat_emb(obs["node_feat"], obs["node_valid"]))
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
        curr_emb: torch.Tensor,        # [N, M, d] GAT emb, or critic_feat_emb when --no-gat
        nbr_embs: torch.Tensor | None,  # [N, M, K, d]; None when --no-gat
        action_mask: torch.Tensor,     # [N, M, K]
        action: torch.Tensor,          # [N, M]
        hidden_actor: torch.Tensor,    # [N, M, d]
        hidden_critic: torch.Tensor,   # [N, d]
        prev_action: torch.Tensor,     # [N, M, K=8] one-hot
        agent_scalars: torch.Tensor,   # [N, M, 2] [∆M-gate, staleness]
        value_field: torch.Tensor,     # [N, M, K] per-first-step discounted utility ∈[0,1]
        critic_global: torch.Tensor | None = None,       # [N, CRITIC_GLOBAL_DIM] CTDE value-only state
    ) -> dict:
        """One PPO-evaluate step given pre-encoded curr/nbr embeddings."""
        N, M, _ = curr_emb.shape
        B = N * M
        vf = value_field.reshape(B, K)
        curr_emb_flat = curr_emb.reshape(B, self.d)
        actor_in = self._actor_in(curr_emb_flat, prev_action.reshape(B, K),
                                  agent_scalars.reshape(B, self.n_agent_scalars), vf)
        h_act_in = hidden_actor.reshape(B, self.d)
        h_act_out = self._step_actor(actor_in, h_act_in)
        logits = self._actor_logits(h_act_out,
                                    nbr_embs.reshape(B, K, self.d) if nbr_embs is not None else None,
                                    action_mask.reshape(B, K), vf)
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
