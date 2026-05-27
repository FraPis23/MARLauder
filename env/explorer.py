"""Vectorized exploration environment, GPU-resident. v0.3: per-agent maps + comm.

State (all torch tensors on device):
    gt[N, H, W]                   uint8 — ground truth (0=obst, 1=free)
    occupancy[N, M, H, W]         uint8 — per-agent local map (v0.3: per-agent)
    occupancy_logodds[N, M, H, W] f32   — Bayesian log-odds per agent
    pos[N, M, 2]                  f32   — (x, y) world coords
    last_known_pos[N, M, M, 2]    f32   — agent i's last known position of agent j
    comm_mask[N, M, M]            bool  — who can communicate this step
    visited_step[N, M, N_max]     long  — last step node was curr, -1 if never
    t[N]                          long  — current step

Communication (v0.3):
    comm_range_px: Euclidean range threshold (pixels).
    LOS: sampled Bresenham check on gt (no comm through walls).
    On comm: fuse log-odds maps via elementwise max (idempotent).
    Positions exchanged: last_known_pos updated for visible agents.

step(action[N, M]):
    1. Move agents (linear interp + collision clamp).
    2. LiDAR scan (per-agent).
    3. Communication check + map fusion + last_known_pos update.
    4. Graph rebuild per agent + guidepost.
    5. Team reward = Δ(union of FREE across M agents) / total_free.

reset(indices): reload map, reset all per-agent state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from env.frontier import compute_frontier
from env.graph_lattice import GraphLattice
from env.maps import Split, sample_batch
from env.world_warp import WarpWorld

_FREE     = 1
_OBSTACLE = 2
GT_FREE   = 1
GT_OBST   = 0


@dataclass
class EnvCfg:
    n_envs: int = 8
    n_agents: int = 1
    nr: int = 16
    sensor_range_px: float = 60.0
    n_rays: int = 720
    utility_range_px: int = 30
    num_sim_steps: int = 5
    max_episode_steps: int = 512
    flood_max_iters: int = 200
    done_explored_thresh: float = 0.99
    comm_range_px: float = 120.0        # communication range (default 2× sensor_range)
    comm_los_samples: int = 40          # LOS line samples (Bresenham approx)


class Explorer:
    def __init__(self, split: Split, cfg: EnvCfg, seed: int = 0) -> None:
        self.split = split
        self.cfg = cfg
        self.dev = split.device
        self.H, self.W = split.canvas
        self.M = cfg.n_agents
        self.N = cfg.n_envs
        self.rng = np.random.default_rng(seed)

        gt, starts, fc = sample_batch(split, cfg.n_envs, seed=seed, device=self.dev)
        self.map_indices = torch.zeros(cfg.n_envs, dtype=torch.long, device=self.dev)
        self.starts = starts.clone()
        self.free_total = fc.clone().float()
        self.world = WarpWorld(
            gt,
            n_agents=cfg.n_agents,
            sensor_range=cfg.sensor_range_px,
            n_rays=cfg.n_rays,
            device=self.dev,
        )
        self.graph = GraphLattice(
            canvas=(self.H, self.W),
            nr=cfg.nr,
            sensor_range_px=cfg.sensor_range_px,
            utility_range_px=cfg.utility_range_px,
            collision_samples=5,
            flood_max_iters=cfg.flood_max_iters,
            device=self.dev,
        )
        self.N_max = self.graph.N_max
        self.P_max = self.graph.guidepost_path_max
        self.K = 8

        self.pos          = torch.zeros((self.N, self.M, 2),           dtype=torch.float32, device=self.dev)
        self.visited_step = torch.full((self.N, self.M, self.N_max), -1, dtype=torch.long,  device=self.dev)
        self.t            = torch.zeros(self.N,                        dtype=torch.long,    device=self.dev)
        self.last_union   = torch.zeros(self.N,                        dtype=torch.float32, device=self.dev)
        self.curr_idx     = torch.zeros((self.N, self.M),              dtype=torch.long,    device=self.dev)
        # last known position: agent i's knowledge of agent j's position
        self.last_known_pos = torch.zeros((self.N, self.M, self.M, 2), dtype=torch.float32, device=self.dev)
        self._last_obs: dict = {}

        self._reset_all()

    # ---------------------------------------------------------------------- #
    # public API                                                              #
    # ---------------------------------------------------------------------- #
    def reset(self) -> dict:
        self._reset_all()
        return self._last_obs

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
        """action: long [N, M] in [0, K). Returns (obs, reward[N,M], done[N], info)."""
        assert action.shape == (self.N, self.M)
        curr_nbr       = self._last_obs["curr_nbr"]
        curr_nbr_valid = self._last_obs["curr_nbr_valid"]
        chosen       = torch.gather(curr_nbr,       dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
        chosen_valid = torch.gather(curr_nbr_valid, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
        chosen = torch.where(chosen_valid, chosen, self.curr_idx).clamp(min=0)

        node_xy = self.graph.node_xy
        tgt_xy  = node_xy[chosen]   # [N, M, 2]

        K_sub = self.cfg.num_sim_steps
        min_agent_dist = float(self.cfg.nr)  # agents must stay >= 1 lattice spacing apart
        for s in range(1, K_sub + 1):
            t_frac  = float(s) / float(K_sub)
            sub_pos = self.pos * (1.0 - t_frac) + tgt_xy * t_frac
            # Wall collision: revert agents that hit an obstacle
            ix = sub_pos[..., 0].clamp(0, self.W - 1).long()
            iy = sub_pos[..., 1].clamp(0, self.H - 1).long()
            gt_at = self.world.gt_torch.view(self.N, -1).gather(
                1, (iy * self.W + ix).view(self.N, -1)
            ).view(self.N, self.M)
            collide_wall = (gt_at == GT_OBST)
            sub_pos = torch.where(collide_wall.unsqueeze(-1), self.pos, sub_pos)
            # Agent-agent collision: revert both agents when too close.
            # Hard env constraint — robots physically cannot overlap.
            if self.M > 1:
                for i in range(self.M):
                    for j in range(i + 1, self.M):
                        d = (sub_pos[:, i] - sub_pos[:, j]).norm(dim=-1)  # [N]
                        collide_pair = (d < min_agent_dist).unsqueeze(-1)  # [N, 1]
                        sub_pos[:, i] = torch.where(collide_pair, self.pos[:, i], sub_pos[:, i])
                        sub_pos[:, j] = torch.where(collide_pair, self.pos[:, j], sub_pos[:, j])
            self.pos = sub_pos
            self.world.set_positions(self.pos)
            self.world.scan()

        env_idx   = torch.arange(self.N, device=self.dev).view(self.N, 1).expand(-1, self.M)
        agent_idx = torch.arange(self.M, device=self.dev).view(1, self.M).expand(self.N, -1)
        self.visited_step[env_idx, agent_idx, chosen] = self.t.view(self.N, 1).expand(-1, self.M)
        self.t = self.t + 1

        # Communication: check range + LOS, fuse maps, update last_known_pos
        comm_mask = self._comm_check()
        self.world.fuse_maps(comm_mask)
        self._update_last_known_pos(comm_mask)

        # Reward: Δ(union FREE across agents) / total_free
        union_free = (self.world.occupancy_torch == _FREE).any(dim=1).view(self.N, -1).float().sum(-1)
        explored_rate = (union_free / self.free_total.clamp(min=1.0)).clamp(0, 1)
        delta = (union_free - self.last_union) / self.free_total.clamp(min=1.0)
        self.last_union = union_free
        team_reward = delta.clamp(min=0.0)
        reward = team_reward.unsqueeze(-1).expand(-1, self.M)

        self._refresh_obs(comm_mask)

        truncated  = self.t >= self.cfg.max_episode_steps
        terminated = explored_rate >= self.cfg.done_explored_thresh
        done = truncated | terminated
        info = {
            "explored_rate": explored_rate,
            "terminated":    terminated,
            "truncated":     truncated,
            "step":          self.t.clone(),
        }
        if bool(done.any().item()):
            idx = torch.nonzero(done, as_tuple=False).flatten().cpu().numpy().tolist()
            self._reset_envs(idx)
        return self._last_obs, reward, done, info

    @property
    def obs(self) -> dict:
        return self._last_obs

    # ---------------------------------------------------------------------- #
    # communication                                                           #
    # ---------------------------------------------------------------------- #
    def _comm_check(self) -> torch.Tensor:
        """Returns comm_mask[N, M, M] bool.

        True at (n,i,j): agent i and j can communicate in env n.
        Condition: Euclidean dist < comm_range_px AND LOS clear on GT.
        Diagonal is always True (self-comm).
        """
        N, M = self.N, self.M
        eye = torch.eye(M, dtype=torch.bool, device=self.dev).view(1, M, M).expand(N, -1, -1)
        comm_mask = eye.clone()
        if M < 2:
            return comm_mask

        comm_range = self.cfg.comm_range_px
        S = self.cfg.comm_los_samples
        gt = self.world.gt_torch   # [N, H, W]

        for i in range(M):
            for j in range(i + 1, M):
                pi   = self.pos[:, i, :]    # [N, 2]
                pj   = self.pos[:, j, :]    # [N, 2]
                diff = pj - pi
                dist = diff.norm(dim=-1)    # [N]
                in_range = dist < comm_range
                if not in_range.any():
                    continue

                # LOS: sample S points along segment, check GT for obstacles
                t_vals = torch.linspace(0.0, 1.0, S, device=self.dev)  # [S]
                pts = pi.unsqueeze(1) + t_vals.view(1, S, 1) * diff.unsqueeze(1)  # [N, S, 2]
                ix = pts[..., 0].clamp(0, self.W - 1).long()  # [N, S]
                iy = pts[..., 1].clamp(0, self.H - 1).long()
                n_idx  = torch.arange(N, device=self.dev).view(N, 1).expand(N, S)
                hit    = gt[n_idx, iy, ix] == GT_OBST         # [N, S]
                los_ok = ~hit.any(dim=-1)                      # [N]

                can = in_range & los_ok
                comm_mask[:, i, j] = can
                comm_mask[:, j, i] = can

        return comm_mask

    def _update_last_known_pos(self, comm_mask: torch.Tensor) -> None:
        """Update last_known_pos where agents can communicate."""
        for i in range(self.M):
            for j in range(self.M):
                can = comm_mask[:, i, j]   # [N]
                if not can.any():
                    continue
                new_pos = self.pos[:, j, :]   # [N, 2] — actual current position
                mask2d  = can.view(-1, 1)
                self.last_known_pos[:, i, j] = torch.where(
                    mask2d.expand(-1, 2), new_pos, self.last_known_pos[:, i, j]
                )

    # ---------------------------------------------------------------------- #
    # internals                                                               #
    # ---------------------------------------------------------------------- #
    def _spread_starts_graph(self, start_row: int, start_col: int) -> torch.Tensor:
        """Return M start positions [M, 2] (x=col, y=row) on distinct lattice nodes.

        Takes the M nearest graph nodes to the map start, sorted by distance.
        Guaranteed distinct (nodes are unique), guaranteed valid (nodes = free cells).
        Fully GPU — node_xy already on device, no CPU loop.
        """
        start_pos = torch.tensor([float(start_col), float(start_row)], device=self.dev)
        node_xy = self.graph.node_xy                                    # [N_max, 2]
        dist = (node_xy - start_pos.unsqueeze(0)).norm(dim=-1)          # [N_max]
        k = min(self.M, node_xy.shape[0])
        _, sorted_idx = torch.topk(dist, k=k, largest=False)            # k nearest
        out = torch.zeros(self.M, 2, dtype=torch.float32, device=self.dev)
        for i in range(self.M):
            out[i] = node_xy[sorted_idx[i % k]]                        # wraps if k < M
        return out

    def _reset_all(self) -> None:
        self._reset_envs(list(range(self.N)))

    def _reset_envs(self, idx: list[int]) -> None:
        if not idx:
            return
        n = len(idx)
        new_idx = self.rng.integers(0, self.split.n, size=n, dtype=np.int64)
        gt_new, starts_new, fc_new = sample_batch(
            self.split, n, indices=new_idx,
            seed=int(self.rng.integers(0, 1 << 31)), device=self.dev,
        )
        idx_t = torch.tensor(idx, dtype=torch.long, device=self.dev)
        self.world.gt_torch[idx_t]                    = gt_new
        self.world.occupancy_torch[idx_t]             = 0
        self.world.occupancy_logodds_torch[idx_t]     = 0.0
        self.free_total[idx_t]                        = fc_new.float()
        self.starts[idx_t]                            = starts_new
        self.visited_step[idx_t]                      = -1
        self.t[idx_t]                                 = 0

        for j_env, e in enumerate(idx):
            row0, col0 = int(starts_new[j_env, 0]), int(starts_new[j_env, 1])
            agent_pos = self._spread_starts_graph(row0, col0)           # [M, 2] on GPU
            self.pos[e] = agent_pos
            # All agents know all actual start positions (in comm range at reset)
            for ag in range(self.M):
                self.last_known_pos[e, :, ag] = agent_pos[ag]

        self.world.set_positions(self.pos)
        self.world.scan()
        union_free = (self.world.occupancy_torch[idx_t] == _FREE).any(dim=1).view(n, -1).float().sum(-1)
        self.last_union[idx_t] = union_free
        self._refresh_obs()

    def _refresh_obs(self, comm_mask: torch.Tensor | None = None) -> None:
        """Build per-agent obs from current per-agent occupancy + positions."""
        node_xy_list            = []
        node_valid_list         = []
        node_feat_list          = []
        edge_idx_list           = []
        edge_valid_list         = []
        curr_idx_list           = []
        curr_nbr_list           = []
        curr_nbr_valid_list     = []
        utility_list            = []
        guidepost_mask_list     = []
        guidepost_target_list   = []
        guidepost_path_xy_list  = []
        guidepost_path_valid_list = []
        guidepost_nbr_bias_list = []
        guidepost_next_hop_list = []

        for a in range(self.M):
            occ_a      = self.world.occupancy_torch[:, a, :, :]  # [N, H, W]
            frontier_a = compute_frontier(occ_a)
            info = self.graph.build(
                occupancy=occ_a,
                frontier=frontier_a,
                robot_xy=self.pos[:, a, :],
                visited_step=self.visited_step[:, a, :],
                current_step=int(self.t.max().item()),
            )
            self.graph.build_guidepost(info)
            node_xy_list.append(info["node_xy"])
            node_valid_list.append(info["node_valid"])
            node_feat_list.append(info["node_feat"])
            edge_idx_list.append(info["edge_idx"])
            edge_valid_list.append(info["edge_valid"])
            curr_idx_list.append(info["curr_idx"])
            curr_nbr_list.append(info["curr_nbr"])
            curr_nbr_valid_list.append(info["curr_nbr_valid"])
            utility_list.append(info["utility"])
            guidepost_mask_list.append(info["guidepost_mask"])
            guidepost_target_list.append(info["guidepost_target"])
            guidepost_path_xy_list.append(info["guidepost_path_xy"])
            guidepost_path_valid_list.append(info["guidepost_path_valid"])
            guidepost_nbr_bias_list.append(info["guidepost_nbr_bias"])
            guidepost_next_hop_list.append(info["guidepost_next_hop"])

        node_xy             = torch.stack(node_xy_list, dim=1)
        node_valid          = torch.stack(node_valid_list, dim=1)
        node_feat           = torch.stack(node_feat_list, dim=1)
        edge_idx            = torch.stack(edge_idx_list, dim=1)
        edge_valid          = torch.stack(edge_valid_list, dim=1)
        curr_idx            = torch.stack(curr_idx_list, dim=1)
        curr_nbr            = torch.stack(curr_nbr_list, dim=1)
        curr_nbr_valid      = torch.stack(curr_nbr_valid_list, dim=1)
        utility             = torch.stack(utility_list, dim=1)
        guidepost_mask      = torch.stack(guidepost_mask_list, dim=1)
        guidepost_target    = torch.stack(guidepost_target_list, dim=1)
        guidepost_path_xy   = torch.stack(guidepost_path_xy_list, dim=1)
        guidepost_path_valid = torch.stack(guidepost_path_valid_list, dim=1)
        guidepost_nbr_bias  = torch.stack(guidepost_nbr_bias_list, dim=1)
        guidepost_next_hop  = torch.stack(guidepost_next_hop_list, dim=1)

        # feat[5]: last-known position of teammate at nearest graph node.
        # For M=1: stays zero (no teammates). For M>1: mark the node nearest
        # to each agent's last-known position of each other agent.
        if self.M > 1:
            nx = self.graph.node_xy[:, 0].view(1, -1)  # [1, N_max]
            ny = self.graph.node_xy[:, 1].view(1, -1)
            occ_feat = torch.zeros(
                (self.N, self.M, self.N_max), dtype=torch.float32, device=self.dev)
            arange_n = torch.arange(self.N, device=self.dev)
            for a in range(self.M):
                for b in range(self.M):
                    if a == b:
                        continue
                    lkp = self.last_known_pos[:, a, b, :]   # [N, 2]
                    dx  = nx - lkp[:, 0:1]                  # [N, N_max]
                    dy  = ny - lkp[:, 1:2]
                    nearest = (dx * dx + dy * dy).argmin(dim=-1)  # [N]
                    occ_feat[arange_n, a, nearest] = 1.0
            node_feat[..., 5] = occ_feat

        self.curr_idx = curr_idx
        if comm_mask is None:
            comm_mask = torch.eye(
                self.M, dtype=torch.bool, device=self.dev
            ).view(1, self.M, self.M).expand(self.N, -1, -1)

        self._last_obs = {
            "node_xy":              node_xy,
            "node_valid":           node_valid,
            "node_feat":            node_feat,
            "edge_idx":             edge_idx,
            "edge_valid":           edge_valid,
            "curr_idx":             curr_idx,
            "curr_nbr":             curr_nbr,
            "curr_nbr_valid":       curr_nbr_valid,
            "action_mask":          curr_nbr_valid,
            "utility":              utility,
            "guidepost_mask":       guidepost_mask,
            "guidepost_target":     guidepost_target,
            "guidepost_path_xy":    guidepost_path_xy,
            "guidepost_path_valid": guidepost_path_valid,
            "guidepost_nbr_bias":   guidepost_nbr_bias,
            "guidepost_next_hop":   guidepost_next_hop,
            "pos":                  self.pos.clone(),
            "comm_mask":            comm_mask,
            "last_known_pos":       self.last_known_pos.clone(),
        }
