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
        self.N_max, self.F, self.K = N_max, F, K

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
            "guidepost_nbr_bias": _z((T, N, M, K),            torch.float32),
        }
        self.actions   = _z((T, N, M), torch.long)
        self.logp      = _z((T, N, M), torch.float32)
        self.values    = _z((T, N),    torch.float32)        # denormalized
        self.rewards   = _z((T, N, M), torch.float32)
        self.dones     = _z((T, N),    torch.bool)

        # initial recurrent states (state at the START of the buffer's first step)
        self.h_actor_init  = _z((N, M, d_hidden), torch.float32)
        self.h_critic_init = _z((N, d_hidden),    torch.float32)
        self.last_value    = _z((N,), torch.float32)         # V(s_T) for bootstrap

    def store(self, t: int, obs: dict, action: torch.Tensor, logp: torch.Tensor,
              value: torch.Tensor, reward: torch.Tensor, done: torch.Tensor) -> None:
        for k in self.obs:
            self.obs[k][t].copy_(obs[k])
        self.actions[t].copy_(action)
        self.logp[t].copy_(logp)
        self.values[t].copy_(value)
        self.rewards[t].copy_(reward)
        self.dones[t].copy_(done)

    @torch.no_grad()
    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns advantages [T, N], returns [T, N]."""
        T, N = self.T, self.N
        team_r = self.rewards.mean(dim=-1)                   # [T, N]
        adv = torch.zeros((T, N), dtype=torch.float32, device=self.dev)
        gae = torch.zeros((N,), dtype=torch.float32, device=self.dev)
        for t in reversed(range(T)):
            nonterm = (~self.dones[t]).float()
            next_v = self.last_value if t == T - 1 else self.values[t + 1]
            delta = team_r[t] + gamma * next_v * nonterm - self.values[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + self.values
        return adv, ret

    def slice_step(self, t: int) -> dict:
        """Return an obs dict at time t with shape [N, M, ...] — ready for model forward."""
        return {k: v[t] for k, v in self.obs.items()}

    def slice_chunk(self, t0: int, t1: int) -> dict:
        """Stacked obs across a [t0, t1) chunk → tensors with leading dim (t1-t0)."""
        return {k: v[t0:t1] for k, v in self.obs.items()}
