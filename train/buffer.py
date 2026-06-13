"""On-GPU rollout buffer for MAPPO.

Pre-allocates tensors for the obs dict, actions, log-probs, values, rewards,
dones and initial recurrent states. GAE-λ on the team-mean per-step reward
(matches TOM; identical to per-agent reward when M=1).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


_OBS_KEYS_FLOAT = ("node_feat",)
_OBS_KEYS_BOOL = ("node_valid", "edge_valid", "curr_nbr_valid", "action_mask")
_OBS_KEYS_LONG = ("edge_idx", "curr_idx", "curr_nbr")


@dataclass
class BufferStats:
    explored_rate_mean: float
    reward_mean: float
    return_mean: float


class Rollout:
    """Fixed-shape buffer for T steps × N envs × M agents."""

    def __init__(self, sample_obs: dict, T: int, N: int, M: int, d_hidden: int, device: str) -> None:
        self.T, self.N, self.M, self.d_hidden = T, N, M, d_hidden
        self.dev = device
        N_max = sample_obs["node_feat"].shape[2]
        F = sample_obs["node_feat"].shape[3]
        K = sample_obs["edge_idx"].shape[3]
        K_cand = sample_obs["cand_feat"].shape[2]
        F_cand = sample_obs["cand_feat"].shape[3]
        self.N_max, self.F, self.K = N_max, F, K
        self.K_cand, self.F_cand = K_cand, F_cand

        def _z(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        self.obs = {
            "node_feat":          _z((T, N, M, N_max, F),     torch.float32),
            "node_valid":         _z((T, N, M, N_max),        torch.bool),
            "edge_idx":           _z((T, N, M, N_max, K),     torch.long),
            "edge_valid":         _z((T, N, M, N_max, K),     torch.bool),
            "curr_idx":           _z((T, N, M),               torch.long),
            "curr_nbr":           _z((T, N, M, K),            torch.long),
            "curr_nbr_valid":     _z((T, N, M, K),            torch.bool),
            "action_mask":        _z((T, N, M, K),            torch.bool),
            # Phase A v2 — strategic head inputs.
            "cand_feat":          _z((T, N, M, K_cand, F_cand), torch.float32),
            "cand_valid":         _z((T, N, M, K_cand),        torch.bool),
            "cand_xy":            _z((T, N, M, K_cand, 2),     torch.float32),
            "cand_bf_first_hop":  _z((T, N, M, K_cand, K),     torch.float32),
            "pos":                _z((T, N, M, 2),             torch.float32),
            # Fix B — previous-action one-hot per agent.
            "prev_action":        _z((T, N, M, K),             torch.float32),
            # Phase 3 — guidepost first-hop one-hot (toward nearest frontier); used by the
            # single-pointer ablation (no strategic head) as the action-bias signal.
            "guidepost_nbr_bias": _z((T, N, M, K),             torch.float32),
        }
        self.actions       = _z((T, N, M), torch.long)
        # Phase A v2 — strategic K-slot chosen at rollout (for STE replay during PPO).
        self.target_choice = _z((T, N, M), torch.long)
        self.logp          = _z((T, N, M), torch.float32)
        self.values    = _z((T, N),    torch.float32)        # denormalized
        self.rewards   = _z((T, N, M), torch.float32)
        self.dones     = _z((T, N),    torch.bool)

        # initial recurrent states (state at the START of the buffer's first step)
        self.h_actor_init  = _z((N, M, d_hidden), torch.float32)
        self.h_critic_init = _z((N, d_hidden),    torch.float32)
        self.last_value    = _z((N,), torch.float32)         # V(s_T) for bootstrap

    def store(self, t: int, obs: dict, action: torch.Tensor, logp: torch.Tensor,
              value: torch.Tensor, reward: torch.Tensor, done: torch.Tensor,
              target_choice: torch.Tensor) -> None:
        for k in self.obs:
            self.obs[k][t].copy_(obs[k])
        self.actions[t].copy_(action)
        self.target_choice[t].copy_(target_choice)
        self.logp[t].copy_(logp)
        self.values[t].copy_(value)
        self.rewards[t].copy_(reward)
        self.dones[t].copy_(done)

    @torch.no_grad()
    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase D — per-agent GAE on shared CTDE value baseline.

        Returns:
            adv      [T, N, M]    per-agent advantage
            ret_team [T, N]       team-mean return (target for shared V)
        """
        T, N, M = self.T, self.N, self.M
        adv = torch.zeros((T, N, M), dtype=torch.float32, device=self.dev)
        gae = torch.zeros((N, M), dtype=torch.float32, device=self.dev)
        for t in reversed(range(T)):
            nonterm = (~self.dones[t]).float().unsqueeze(-1)              # [N, 1]
            next_v = (self.last_value if t == T - 1 else self.values[t + 1]).unsqueeze(-1).expand(-1, M)
            v_t    = self.values[t].unsqueeze(-1).expand(-1, M)
            delta = self.rewards[t] + gamma * next_v * nonterm - v_t       # [N, M]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        # Per-agent returns; value loss target is team mean (V is shared CTDE scalar).
        ret_per_agent = adv + self.values.unsqueeze(-1).expand(-1, -1, M)  # [T, N, M]
        ret_team = ret_per_agent.mean(dim=-1)                               # [T, N]
        return adv, ret_team

    def slice_step(self, t: int) -> dict:
        """Return an obs dict at time t with shape [N, M, ...] — ready for model forward."""
        return {k: v[t] for k, v in self.obs.items()}

    def slice_chunk(self, t0: int, t1: int) -> dict:
        """Stacked obs across a [t0, t1) chunk → tensors with leading dim (t1-t0)."""
        return {k: v[t0:t1] for k, v in self.obs.items()}
