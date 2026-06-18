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

from dataclasses import dataclass, fields

import numpy as np
import torch

from env.frontier import compute_frontier
from env.graph_lattice import GraphLattice
from env.maps import Split, sample_batch
from env.world_warp import WarpWorld

_UNKNOWN  = 0
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
    visit_age_window: int = 16               # feat[3] recency horizon (steps): walked node ramps 0→1 freshness
    # Analytic global-target selection (deterministic). score = util/(1+β·d_curr/NR)·(1+λ·w·rdv).
    target_beta: float = 1.0                 # β: distance discount (per NR node-unit)
    target_lambda: float = 1.0               # λ: rendezvous pull strength (0 = pure exploration)
    rdv_offer_frac: float = 0.15             # offer saturates (w→1) at this fraction of map cells gained since sync
    target_keep_margin: float = 0.2          # commit: keep last target unless a new one beats it by >this fraction
    progress_reward_coef: float = 0.3        # PBRS-style: reward per node-unit closer to the committed target
    analytic_target: bool = True             # env owns the deterministic global target (vs learned StrategicHead)
    num_sim_steps: int = 5
    max_episode_steps: int = 512
    flood_max_iters: int = 200
    done_explored_thresh: float = 0.99
    comm_range_px: float = 120.0        # communication range (default 2× sensor_range)
    comm_los_samples: int = 40          # LOS line samples (Bresenham approx)
    step_penalty_coef: float = 0.1     # total step penalty over episode = coef (scaled by 1/max_steps)
    completion_bonus: float = 10.0     # reward given at the terminal step when explored >= threshold
    n_hops: int = 2                     # ego-centric encoder window radius (window_side = 2·n_hops + 3)
    # Phase D — per-agent reward shaping (lattice-level set ops, baselined at last comm).
    scan_reward_weight: float = 1.0         # α_scan: cells I LiDAR-scanned this step
    # v2 reward — privileged novel-scan credit (IR2-style r_f): pay only cells the agent
    # scanned that are NEW to the TEAM UNION map. Follower scanning a leader's wake earns 0
    # → removes the chase/free-ride incentive at the source. Training-only privileged signal
    # (CTDE); the deployed actor never sees the union. Replaces scan_self in the reward;
    # scan_self stays as a logged diagnostic.
    novel_scan_weight: float = 1.0          # α_novel
    # Dense-term normalization: ~one sensor disk worth of lattice nodes per productive step.
    # The old /N_max (≈1200) crushed dense terms to O(0.001) vs completion bonus 10.
    scan_norm_nodes: float = 50.0
    team_reward_weight: float = 0.3         # β: shared Δunion (cooperation anchor)
    give_bonus_coef: float = 1.5            # ζ_give: NEW cells I bring to teammate at rendezvous
    recv_bonus_coef: float = 0.5            # ζ_recv: NEW cells I get from teammate at rendezvous
    overlap_penalty_coef: float = 3.0       # η_lap: cells we BOTH scanned independently since last comm
    revisit_penalty_coef: float = 0.05      # γ: penalty per step on a node visited in last W steps
    revisit_window: int = 8                 # W: lookback window for revisit detection
    # Stall penalty — heavy cost for standing still (no net displacement this step). Catches
    # collision-revert holds AND invalid/curr-node picks. Pressures agents to reroute /
    # separate instead of deadlocking. δ_stall ≫ revisit so standing still is "heavily penalized".
    stall_penalty_coef: float = 0.1         # δ_stall
    # Objective second-guessing penalty (graph-tree). Fires when the agent flips the
    # BF-from-curr first-hop BRANCH toward its strategic target while the previous target
    # was still reachable + unreached (B+D). Same-direction target shifts (frontier
    # receding down the same branch) cost 0; only genuine mid-route fork-flips are taxed.
    # v2: 0.05 → 0.01 (sweep v1 showed it dominating the dense reward 10–50×) and the
    # caller now passes the ARGMAX strategic intent, not the Gumbel-sampled pick.
    target_switch_penalty_coef: float = 0.05    # δ_obj: B+D graph-tree branch-flip commitment
    # penalty. Raised 0.01→0.05 — it was lowered only because the Gumbel-SAMPLED target jittered;
    # it now receives the argmax intent, so it can punish uncommitted back-and-forth on the tree.
    # G.4.a — amplify cand_own_minus_team feature (yield signal). Higher scale → faster
    # learning of yielding behavior (smooth, no oscillation).
    cand_own_minus_team_scale: float = 3.0
    # J.1 — yield-gated target bonus (reactive position-driven division). Rewards the agent
    # for CHOOSING a target frontier it is closer to than the teammate (= "mine"), using the
    # live teammate position (cand_min_team_dist ← last_known_pos). It is a one-sided PULL:
    # clamp(team_dist − own_dist, 0) ≥ 0, never negative → no repulsion, no ping-pong. With
    # one viable frontier the closer agent is paid and the farther agent is simply NOT paid
    # (0, not penalized) so it still goes (novel_scan dominates) → no idle. With two frontiers
    # each agent is closer to a different one → both paid → reactive division emerges. Static
    # split can't fake it: the bonus tracks where the teammate ACTUALLY is. 0.0 = off.
    target_yield_weight: float = 0.0        # α_yield
    # G.4.b — per-step proximity penalty when teammate is too close (anti-chase).
    # ELIMINATED (default 0.0): this raw-distance reflex was the limit-cycle driver — it
    # over-corrected into ping-pong and punished BOTH agents converging on the last frontier
    # (deadlock). novel_scan already pays 0 for scanning team-known cells, so the anti-chase
    # incentive is present without it. Kept as a flag for ablation.
    proximity_penalty_coef: float = 0.0     # ε_prox: reward subtracted per step when close
    proximity_penalty_radius_px: float = -1.0   # <=0 = sensor_range_px
    # A2 — bypass distance / LOS check in comm: every step every agent communicates.
    force_full_comm: bool = False
    # Debug — persistent teammate-position awareness (positions only, not maps).
    # When True, last_known_pos and t_last_comm always reflect actual current pos,
    # regardless of comm_mask. Map fusion still gated by comm_mask. Used to isolate
    # whether chase/weird-movement bugs come from stale lkp or elsewhere. Remove later.
    force_full_pos_sharing: bool = False
    # H.4 — persistent occupancy sharing (debug only). When True, world.fuse_maps fires
    # with all-True mask every step → maps continuously synchronized. Set ops give/recv/
    # overlap derived from fully-synced maps. Distinct from force_full_comm (which
    # short-circuits _comm_check); this directly overrides the comm_mask used in fusion
    # and reward set ops, without changing what cand_max_comm_gap reports.
    force_full_occupancy_sharing: bool = False
    # Phase A v2 / A1 — top-K frontier candidates per agent for strategic head.
    top_k_candidates: int = 16

    @classmethod
    def from_ckpt_dict(cls, d: dict, **overrides) -> "EnvCfg":
        """I.2 — reconstruct EnvCfg from a saved cfg["env"] dict, applying overrides.

        Filters `d` to valid EnvCfg fields so unknown / stale keys are ignored. Ensures
        eval mirrors training comm/sharing/feature config (force flags, top_k, n_hops...).
        """
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in valid}
        kwargs.update(overrides)
        return cls(**kwargs)


class Explorer:
    def __init__(self, split: Split, cfg: EnvCfg, seed: int = 0) -> None:
        self.split = split
        self.cfg = cfg
        self.dev = split.device
        self.H, self.W = split.canvas
        self.M = cfg.n_agents
        self.N = cfg.n_envs
        # Map RNG always fresh entropy (independent of cfg.seed). cfg.seed governs torch
        # RNG only (action sampling, init reproducibility for training stability).
        self.rng = np.random.default_rng()
        init_seed = int(self.rng.integers(0, 1 << 31))
        gt, starts, fc = sample_batch(split, cfg.n_envs, seed=init_seed, device=self.dev)
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
            n_hops=cfg.n_hops,
            visit_age_window=cfg.visit_age_window,
            # Optimistic teammate-BF graph only when it can matter: M>1 AND maps actually
            # diverge. Under force_full_occupancy_sharing (Stage 1) maps are identical every
            # step → teammate is always on a known-FREE node → the FREE graph already reaches
            # it → building the optimistic edge set every step is pure overhead. Skip it.
            build_optim_graph=(self.M > 1 and not cfg.force_full_occupancy_sharing),
            device=self.dev,
        )
        self.N_max = self.graph.N_max
        # Render-only: obs ships just the ego window. When True, _refresh_obs stashes the
        # full-graph per-agent utility/validity in self._render_global for the GIF. Eval sets it.
        self.store_render_global = False
        self._render_global: dict | None = None
        self.P_max = self.graph.guidepost_path_max
        self.K = 8

        self.pos          = torch.zeros((self.N, self.M, 2),           dtype=torch.float32, device=self.dev)
        self.visited_step = torch.full((self.N, self.M, self.N_max), -1, dtype=torch.long,  device=self.dev)
        self.t            = torch.zeros(self.N,                        dtype=torch.long,    device=self.dev)
        self.last_union   = torch.zeros(self.N,                        dtype=torch.float32, device=self.dev)
        self.curr_idx     = torch.zeros((self.N, self.M),              dtype=torch.long,    device=self.dev)
        self.curr_idx_global = torch.zeros((self.N, self.M),           dtype=torch.long,    device=self.dev)
        # last known position: agent i's knowledge of agent j's position
        self.last_known_pos = torch.zeros((self.N, self.M, self.M, 2), dtype=torch.float32, device=self.dev)
        # Target-claim: last_known_target[i, j] = global node agent i believes j is heading to
        # (j's strategic target). Updated comm-gated alongside last_known_pos; -1 = unknown.
        # Used to deconflict targets at rendezvous (lowest agent-ID keeps its target).
        self.last_known_target = torch.full((self.N, self.M, self.M), -1, dtype=torch.long, device=self.dev)
        # B1-redo: per-agent guidepost cache for warm-start BF from target.
        # _target_prev[N, M]: previous step's target node per agent. -1 = no cache.
        # _dist_prev[N, M, N_max]: previous step's BF dist (rooted at target). +inf = cold.
        self._target_prev = torch.full((self.N, self.M), -1, dtype=torch.long, device=self.dev)
        self._dist_prev   = torch.full(
            (self.N, self.M, self.N_max), float("inf"), dtype=torch.float32, device=self.dev,
        )
        # Step of last comm event between agents a and j (per env). Used by Phase A v2
        # strategic head as the cand_max_comm_gap feature (uncertainty over teammate pos).
        # Reset to t=0 at episode start (since _reset_envs writes actual start positions
        # into last_known_pos, so all pairs are "freshly in comm" at t=0).
        self.t_last_comm = torch.zeros((self.N, self.M, self.M), dtype=torch.long, device=self.dev)
        # Fix B — previous action K-slot per agent. -1 at reset → zero one-hot.
        self.last_action = torch.full((self.N, self.M), -1, dtype=torch.long, device=self.dev)
        # Collision tiebreak — per-episode randomized priority key per (env, agent). Lower
        # key wins (advances); higher key yields (holds). Re-drawn each reset → no systematic
        # role bias. Decentralized: derivable from a shared per-episode seed at deploy.
        self._collision_key = torch.rand((self.N, self.M), device=self.dev)
        # Objective second-guessing — previous step's strategic target (global node idx).
        # -1 = none yet (episode start). Reset on episode done. Used by the B+D branch-flip
        # penalty to detect mid-route fork switches in the BF-from-curr tree.
        self._prev_target_node = torch.full((self.N, self.M), -1, dtype=torch.long, device=self.dev)
        # Committed BF-tree first-hop branch (slot 0..7) of last step's target. Fed back as the
        # cand_prev_branch_match observation so the (feedforward) strategic head sees its own
        # last direction → learned commitment, no hysteresis. -1 = none yet.
        self._prev_branch = torch.full((self.N, self.M), -1, dtype=torch.long, device=self.dev)
        # Phase 1 (interrupting options) — consecutive steps the SAME strategic target node has
        # been committed. Fed to the strategic head as committed_steps so it can force a re-pick
        # at the max-horizon cap (prevents pursuing an unreachable-but-still-candidate target
        # forever). Reset to 0 when the committed node changes; reset to 0 on episode done.
        self._steps_on_option = torch.zeros((self.N, self.M), dtype=torch.long, device=self.dev)
        # Phase 1b — True for an agent on the step a map fusion delivered it NEW cells. The
        # strategic target is already correct after fusion (re-scored on the fused map); the
        # observed failure is TACTICAL — stale GRU navigation momentum carries the agent into
        # the teammate's just-received (now-explored) region instead of following its target.
        # This flag marks "world changed, re-plan from here"; wiring (GRU-hidden refresh vs
        # path-bias boost vs obs feature) is decided AFTER instrumentation. Gated on cells
        # ACTUALLY received (not mere comm range) so already-synced neighbors don't trigger.
        self._comm_event = torch.zeros((self.N, self.M), dtype=torch.bool, device=self.dev)
        # Option A — BF-from-curr cache. Warm-start when curr unchanged step-to-step.
        self._curr_prev = torch.full((self.N, self.M), -1, dtype=torch.long, device=self.dev)
        self._dist_curr_prev = torch.full(
            (self.N, self.M, self.N_max), float("inf"), dtype=torch.float32, device=self.dev,
        )
        # H.3 — BF-from-teammate cache per (env, agent, teammate). Warm-start when lkp_node
        # unchanged. Mem: N·M·M·N_max·4B = 614 KB at M=2 / 9.8 MB at M=8.
        self._team_node_prev = torch.full(
            (self.N, self.M, self.M), -1, dtype=torch.long, device=self.dev,
        )
        self._dist_team_prev = torch.full(
            (self.N, self.M, self.M, self.N_max), float("inf"), dtype=torch.float32, device=self.dev,
        )
        # Rendezvous: agent i's explored-cell count at its last sync with agent j. offer_ij =
        # (i's explored now − this) = fresh map i holds that j hasn't received. Updated comm-gated.
        self._own_expl_at_comm = torch.zeros((self.N, self.M, self.M), dtype=torch.float32, device=self.dev)
        # Phase D — lattice-level per-agent free count after last step (post-fusion).
        self.last_own_free_node = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        # v2 reward — privileged team-union FREE-node mask (last step) + per-agent post-fusion
        # own mask (last step), for novel-scan attribution. Episode accumulator of novel cells
        # per agent feeds the contribution-share metrics.
        self.union_node_mask = torch.zeros((self.N, self.N_max), dtype=torch.bool, device=self.dev)
        self.own_node_mask_prev = torch.zeros((self.N, self.M, self.N_max), dtype=torch.bool, device=self.dev)
        self.novel_cells_ep = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        # Per-pair last-meeting node-level FREE mask (union snapshot at last comm event).
        # [N, M, M, N_max] bool ≈ 154 KB at default (N=32, M=2, N_max≈1200).
        self.last_meeting_node_mask = torch.zeros(
            (self.N, self.M, self.M, self.N_max), dtype=torch.bool, device=self.dev,
        )
        # Precompute lattice→pixel flat index for fast node-level FREE extraction.
        nx = self.graph.node_xy[:, 0].long().clamp(0, self.W - 1)
        ny = self.graph.node_xy[:, 1].long().clamp(0, self.H - 1)
        self._node_flat_idx = (ny * self.W + nx).long()                                  # [N_max]
        self._last_obs: dict = {}

        self._reset_all()

    # ---------------------------------------------------------------------- #
    # public API                                                              #
    # ---------------------------------------------------------------------- #
    def reset(self) -> dict:
        self._reset_all()
        return self._last_obs

    @torch.no_grad()
    def step(
        self, action: torch.Tensor, target_choice: torch.Tensor | None = None,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
        """action: long [N, M] in [0, K). Returns (obs, reward[N,M], done[N], info).

        target_choice: optional long [N, M] = the strategic head's chosen K-slot in the
        candidate list (from `model.act`). When provided, enables the objective
        second-guessing penalty (B+D branch-flip on the BF-from-curr tree). None (eval /
        baseline) → penalty off.
        """
        assert action.shape == (self.N, self.M)
        # Phase C: action decode uses GLOBAL curr_nbr (model picked K-slot from local
        # window's curr_nbr_local, but env needs global flat idx to compute world coords
        # and update visited_step).
        curr_nbr_global = self._last_obs["curr_nbr_global"]            # [N, M, K]
        curr_nbr_valid  = self._last_obs["curr_nbr_valid"]             # [N, M, K] (local-edge validity)
        chosen       = torch.gather(curr_nbr_global, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
        chosen_valid = torch.gather(curr_nbr_valid,  dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
        # Invalid action → stay put on the GLOBAL current node. self.curr_idx is the LOCAL
        # window-center constant (≈window²/2) and must NOT be used as a global index here —
        # doing so teleported agents to node_xy[that_constant].
        chosen = torch.where(chosen_valid, chosen, self.curr_idx_global).clamp(min=0)

        node_xy = self.graph.node_xy
        tgt_xy  = node_xy[chosen]   # [N, M, 2]

        # Stall detection — snapshot pre-move position; compared after the sub-step loop.
        pos_entry = self.pos.clone()                                   # [N, M, 2]

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
            # Agent-agent collision: asymmetric yield. Robots physically cannot overlap, but
            # reverting BOTH deadlocks adjacent agents contesting the same cell. Instead the
            # lower-priority agent (higher _collision_key) yields (holds prev pos) while the
            # winner advances. The winner reverts too only if it is STILL within min_dist of
            # the loser's hold cell (true blockage, e.g. loser sits on the only path).
            if self.M > 1:
                key = self._collision_key                              # [N, M] lower = wins
                for i in range(self.M):
                    for j in range(i + 1, self.M):
                        d = (sub_pos[:, i] - sub_pos[:, j]).norm(dim=-1)   # [N]
                        collide = (d < min_agent_dist)                     # [N]
                        i_wins = key[:, i] <= key[:, j]                    # [N]; tie → i (deterministic)
                        i_loser = (collide & ~i_wins).unsqueeze(-1)        # [N, 1]
                        j_loser = (collide & i_wins).unsqueeze(-1)
                        sub_pos[:, i] = torch.where(i_loser, self.pos[:, i], sub_pos[:, i])
                        sub_pos[:, j] = torch.where(j_loser, self.pos[:, j], sub_pos[:, j])
                        # Winner still blocked by loser's hold cell → revert winner too.
                        d2 = (sub_pos[:, i] - sub_pos[:, j]).norm(dim=-1)  # [N]
                        still = (collide & (d2 < min_agent_dist)).unsqueeze(-1)
                        sub_pos[:, i] = torch.where(still, self.pos[:, i], sub_pos[:, i])
                        sub_pos[:, j] = torch.where(still, self.pos[:, j], sub_pos[:, j])
            self.pos = sub_pos
            self.world.set_positions(self.pos)
            self.world.scan()

        env_idx   = torch.arange(self.N, device=self.dev).view(self.N, 1).expand(-1, self.M)
        agent_idx = torch.arange(self.M, device=self.dev).view(1, self.M).expand(self.N, -1)
        # Phase D — snapshot prior visited_step for the chosen node BEFORE update, for revisit detection.
        self._prev_visit_for_revisit = self.visited_step[env_idx, agent_idx, chosen].clone()        # [N, M]
        self.visited_step[env_idx, agent_idx, chosen] = self.t.view(self.N, 1).expand(-1, self.M)
        self.t = self.t + 1
        # Fix B: remember last action K-slot for next obs.
        self.last_action = action.clone()

        # ------ Objective second-guessing penalty (B+D, graph-tree) ------------
        # All quantities below live in the PRE-step frame (tree rooted at the node the
        # agent occupied when it made this decision) → self._last_obs + the cached
        # bf_dist_from_curr (self._dist_curr_prev), both built by the previous _refresh_obs.
        target_switch_pen = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        yield_bonus = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        # Analytic-target mode: the env owns the (deterministic) global target, computed in
        # _refresh_obs. The model's target_choice is a sentinel → skip the learned-head
        # switch-penalty / yield / commitment bookkeeping (it would read garbage from slot 0).
        if target_choice is not None and not self.cfg.analytic_target:
            obs0      = self._last_obs
            cand_idx0 = obs0["cand_idx"]                                  # [N, M, K_cand] global node
            cbfh0     = obs0["cand_bf_first_hop"]                         # [N, M, K_cand, K=8] one-hot
            parent0   = obs0["bf_parent_from_curr"]                       # [N, M, N_max]
            curr_g    = obs0["curr_idx_global"]                          # [N, M]
            nbr_g     = obs0["curr_nbr_global"]                          # [N, M, K=8]
            K_cand    = cand_idx0.shape[-1]
            tc        = target_choice.clamp(0, K_cand - 1)               # [N, M]
            # Current target node + its first-hop branch (slot 0..7).
            g_t = torch.gather(cand_idx0, 2, tc.unsqueeze(-1)).squeeze(-1)               # [N, M]
            # J.2 — yield-gated target bonus = OWNERSHIP × exploration VALUE (no distance
            # magnitude). cand_feat[...,6] = own_minus_team (scaled, clamped [-1,1]): NEGATIVE
            # = I'm closer than the live teammate → "mine". cand_feat[...,2] = cand_utility ∈
            # [0,1] = exploration value of the frontier.
            #   bonus = utility(target) · σ(k·(−own_minus_team − margin))
            # The σ is a SMOOTH ownership gate (≈1 when clearly mine, ≈0 when teammate is
            # closer). Crucially the magnitude is the frontier's UTILITY, not its nearness —
            # so the greedy policy can't game it by grabbing the nearest cell (the J.1 flaw
            # that collapsed coverage). One-sided: never negative → no repulsion. Single
            # frontier → margin unmet for the farther agent → 0 (not penalized) → novel_scan
            # still pulls it in → no idle.
            if self.M > 1 and self.cfg.target_yield_weight > 0.0:
                cf0 = obs0["cand_feat"]
                omt_chosen = torch.gather(cf0[..., 6], 2, tc.unsqueeze(-1)).squeeze(-1)   # [N, M]
                util_chosen = torch.gather(cf0[..., 2], 2, tc.unsqueeze(-1)).squeeze(-1)  # [N, M]
                own_gate = torch.sigmoid(8.0 * (-omt_chosen - 0.05))                      # [N, M] ∈(0,1)
                yield_bonus = util_chosen * own_gate * (g_t >= 0).float()                 # [N, M]
            fh_t = torch.gather(
                cbfh0, 2, tc.view(self.N, self.M, 1, 1).expand(self.N, self.M, 1, self.K)
            ).squeeze(2)                                                                 # [N, M, K=8]
            branch_t       = fh_t.argmax(-1)                                             # [N, M]
            branch_t_valid = fh_t.sum(-1) > 0                                            # [N, M]
            # Previous target node — re-derive its first-hop branch in the CURRENT tree.
            g_prev = self._prev_target_node                                              # [N, M]
            cur = g_prev.clamp(min=0)
            for _walk in range(self.N_max):
                par = torch.gather(parent0, 2, cur.unsqueeze(-1)).squeeze(-1)            # [N, M]
                stop = (par == curr_g) | (par < 0)
                cur = torch.where(stop, cur, par)
                if bool(stop.all().item()):
                    break
            match_prev = (cur.unsqueeze(-1) == nbr_g)                                    # [N, M, K=8]
            branch_prev       = match_prev.float().argmax(-1)                            # [N, M]
            branch_prev_valid = match_prev.any(-1)                                       # [N, M]
            # D gate: only penalize while prev target was still pursuable.
            bf_dist = self._dist_curr_prev                                               # [N, M, N_max]
            d_prev  = torch.gather(bf_dist, 2, g_prev.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            reached_thresh   = float(self.cfg.nr) * 1.5
            prev_exists      = g_prev >= 0
            prev_reached     = (curr_g == g_prev) | (d_prev <= reached_thresh)
            prev_unreachable = ~torch.isfinite(d_prev)
            prev_pursuable = (prev_exists & ~prev_reached & ~prev_unreachable
                              & branch_prev_valid & branch_t_valid)
            flip = branch_t != branch_prev
            target_switch_pen = (prev_pursuable & flip).float()                          # [N, M]
            # Phase 1 — track consecutive steps on the same committed target node. A valid new
            # pick (g_t >= 0) that equals the carried node → increment; a change → reset to 0.
            node_changed = (g_t >= 0) & (g_t != self._prev_target_node)
            self._steps_on_option = torch.where(
                g_t >= 0,
                torch.where(node_changed, torch.zeros_like(self._steps_on_option),
                            self._steps_on_option + 1),
                self._steps_on_option,
            )
            # Carry target forward; keep prev on a transient invalid pick (g_t < 0).
            self._prev_target_node = torch.where(g_t >= 0, g_t, self._prev_target_node)
            # Carry committed branch forward for the cand_prev_branch_match observation.
            self._prev_branch = torch.where(branch_t_valid, branch_t, self._prev_branch)

        # ------ Phase D — node-level set-op reward, baselined at last comm ------
        # Snapshot post-scan, pre-fusion node-level FREE per agent.
        N_max = self.N_max
        occ_pre_flat = self.world.occupancy_torch.view(self.N, self.M, -1)                # [N, M, H*W]
        free_node_pre = occ_pre_flat[:, :, self._node_flat_idx] == _FREE                  # [N, M, N_max]

        # Communication: check range + LOS, fuse maps, update last_known_pos
        comm_mask = self._comm_check()
        # H.4 — when persistent occupancy sharing enabled, override comm_mask used for
        # map fusion AND reward set ops to all-True. Keeps cand_max_comm_gap intact
        # (that uses t_last_comm which only updates via real or pos-share comm).
        if self.cfg.force_full_occupancy_sharing:
            comm_mask = torch.ones_like(comm_mask)
        self.world.fuse_maps(comm_mask)
        self._update_last_known_pos(comm_mask)

        # Post-fusion node-level FREE.
        occ_post_flat = self.world.occupancy_torch.view(self.N, self.M, -1)
        free_node_post = occ_post_flat[:, :, self._node_flat_idx] == _FREE                # [N, M, N_max]

        # Phase 1b — map-exchange event: a node became FREE for agent i via FUSION (not its own
        # scan) this step → it received teammate map data. Marks "world changed, re-plan from here"
        # for the strategic/tactical re-planning signal (wiring decided after instrumentation).
        self._comm_event = (free_node_post & ~free_node_pre).any(dim=-1)                  # [N, M]

        # team_delta (pixel-level, for completion check + cooperative term).
        union_free = (self.world.occupancy_torch == _FREE).any(dim=1).view(self.N, -1).float().sum(-1)
        explored_rate = (union_free / self.free_total.clamp(min=1.0)).clamp(0, 1)
        team_delta = ((union_free - self.last_union) / self.free_total.clamp(min=1.0)).clamp(min=0.0)
        self.last_union = union_free

        # scan_self_delta: cells I LiDAR-scanned this step (node level, pre-fusion).
        # v2: DIAGNOSTIC ONLY — no longer in the reward (novel_scan replaces it).
        denom = float(max(1, N_max))
        own_free_post_scan_node = free_node_pre.float().sum(-1)                            # [N, M]
        scan_self_delta = ((own_free_post_scan_node - self.last_own_free_node) / denom).clamp(min=0.0)
        # Update last_own_free_node to post-fusion (next step's baseline).
        self.last_own_free_node = free_node_post.float().sum(-1)

        # ------ v2 — privileged novel-scan credit + node-level team delta -------
        # novel[a] = cells a scanned THIS STEP that were new to the TEAM UNION map.
        # my_new: vs my own post-fusion map of last step (so cells received via fusion
        # don't count as "scanned by me"). Both-scan-same-new-cell ties credit both
        # (simultaneous discovery — rare, acceptable).
        scan_norm = float(max(1.0, self.cfg.scan_norm_nodes))
        union_prev = self.union_node_mask                                                  # [N, N_max]
        my_new = free_node_pre & ~self.own_node_mask_prev                                  # [N, M, N_max]
        novel_count = (my_new & ~union_prev.unsqueeze(1)).float().sum(-1)                  # [N, M]
        novel_scan = novel_count / scan_norm
        self.novel_cells_ep = self.novel_cells_ep + novel_count
        # Node-level team delta on the SAME normalization (β term; pixel team_delta stays
        # for explored_rate / completion below).
        union_now = union_prev | free_node_post.any(dim=1)                                 # [N, N_max]
        team_delta_node = (union_now & ~union_prev).float().sum(-1) / scan_norm            # [N]
        self.union_node_mask = union_now
        self.own_node_mask_prev = free_node_post

        # Per-pair: contribution / reception / overlap with last-meeting baseline.
        give_bonus  = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        recv_bonus  = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        overlap_pen = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        if self.M > 1:
            for i in range(self.M):
                for j in range(self.M):
                    if i == j:
                        continue
                    comm_ij = comm_mask[:, i, j]                                           # [N]
                    if not comm_ij.any():
                        continue
                    baseline = self.last_meeting_node_mask[:, i, j]                        # [N, N_max]
                    M_i = free_node_pre[:, i]
                    M_j = free_node_pre[:, j]
                    my_new = M_i & ~baseline                                               # cells I scanned since
                    j_new  = M_j & ~baseline
                    contribution_to_j = (my_new & ~M_j).float().sum(-1)                    # NEW I bring
                    reception_from_j  = (j_new  & ~M_i).float().sum(-1)                    # NEW j brings
                    new_overlap_ij    = (my_new & j_new).float().sum(-1)                   # both scanned same
                    comm_f = comm_ij.float()
                    give_bonus[:, i]  = give_bonus[:, i]  + (contribution_to_j / denom) * comm_f
                    recv_bonus[:, i]  = recv_bonus[:, i]  + (reception_from_j  / denom) * comm_f
                    overlap_pen[:, i] = overlap_pen[:, i] + (new_overlap_ij    / denom) * comm_f

            # Update last_meeting_node_mask for pairs that just communicated.
            # Post-fusion union is the new baseline. After max-magnitude fusion both agents
            # have same map → use free_node_post[:, i] as canonical union snapshot.
            for i in range(self.M):
                for j in range(self.M):
                    if i == j:
                        continue
                    mask_ij = comm_mask[:, i, j].view(self.N, 1).expand(self.N, N_max)
                    self.last_meeting_node_mask[:, i, j] = torch.where(
                        mask_ij, free_node_post[:, i], self.last_meeting_node_mask[:, i, j]
                    )

        # revisit_pen: chosen node revisited within last W steps by same agent.
        # Graduated by recency: penalty = (W − age)/W ∈ (0, 1] so tighter loops hurt more.
        W_rev = max(1, int(self.cfg.revisit_window))
        prev_visit_for_chosen = self._prev_visit_for_revisit                                # [N, M]
        t_now_per_m = (self.t - 1).view(self.N, 1).expand(self.N, self.M)                   # [N, M]
        age = (t_now_per_m - prev_visit_for_chosen).clamp(min=0)                            # [N, M]
        is_recent_revisit = (prev_visit_for_chosen >= 0) & (age < W_rev)
        revisit_pen = is_recent_revisit.float() * ((W_rev - age).clamp(min=0).float() / W_rev)

        # G.4.b — per-step proximity penalty when teammate within sensor_range AND in comm.
        # Gated by comm_mask → "visible teammate too close" only. Decentralized.
        proximity_pen = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        if self.M > 1 and self.cfg.proximity_penalty_coef > 0:
            prox_r = self.cfg.proximity_penalty_radius_px
            if prox_r <= 0:
                prox_r = self.cfg.sensor_range_px
            for i in range(self.M):
                for j in range(self.M):
                    if i == j:
                        continue
                    d = (self.pos[:, i] - self.pos[:, j]).norm(dim=-1)              # [N]
                    too_close = (d < prox_r).float() * comm_mask[:, i, j].float()    # [N]
                    proximity_pen[:, i] = proximity_pen[:, i] + too_close

        # Stall penalty — no net displacement this step (collision-revert hold or
        # invalid/curr-node pick). step_disp also feeds the coverage-efficiency metric.
        step_disp = (self.pos - pos_entry).norm(dim=-1)                  # [N, M]

        # Progress shaping toward the committed global target (the missing "follow utility"
        # signal — base reward only pays novel_scan on NEW cells, so explored-area moves had no
        # gradient → loops). Euclidean closing distance to _prev_target_node (the target the
        # agent acted on this step), in node units, capped. Telescopes to ~0 on round trips
        # (no hacking) and aligns with coverage. analytic_target only.
        progress_reward = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        if self.cfg.analytic_target and self.cfg.progress_reward_coef > 0.0:
            tgt_node = self._prev_target_node                            # [N, M] committed target
            tgt_xy = self.graph.node_xy[tgt_node.clamp(min=0)]           # [N, M, 2]
            valid_t = (tgt_node >= 0) & (tgt_node != self.curr_idx_global)
            d_prev = (pos_entry - tgt_xy).norm(dim=-1)                   # [N, M]
            d_now  = (self.pos - tgt_xy).norm(dim=-1)                    # [N, M]
            progress_reward = ((d_prev - d_now) / float(self.cfg.nr)).clamp(-2.0, 2.0) * valid_t.float()
        stall_pen = (step_disp < float(self.cfg.nr) * 0.5).float()       # [N, M]

        terminated_now = explored_rate >= self.cfg.done_explored_thresh
        step_penalty = self.cfg.step_penalty_coef / max(1, self.cfg.max_episode_steps)
        a_novel = self.cfg.novel_scan_weight
        beta    = self.cfg.team_reward_weight
        z_give  = self.cfg.give_bonus_coef
        z_recv  = self.cfg.recv_bonus_coef
        eta_lap = self.cfg.overlap_penalty_coef
        gamma   = self.cfg.revisit_penalty_coef
        eps_prox = self.cfg.proximity_penalty_coef
        delta_obj = self.cfg.target_switch_penalty_coef
        delta_stall = self.cfg.stall_penalty_coef
        a_yield = self.cfg.target_yield_weight
        reward = (a_novel * novel_scan
                  + beta  * team_delta_node.unsqueeze(-1)
                  + z_give * give_bonus
                  + z_recv * recv_bonus
                  + a_yield * yield_bonus
                  - eta_lap * overlap_pen
                  - gamma   * revisit_pen
                  - eps_prox * proximity_pen
                  - delta_obj * target_switch_pen
                  - delta_stall * stall_pen
                  + self.cfg.progress_reward_coef * progress_reward
                  + terminated_now.float().unsqueeze(-1) * self.cfg.completion_bonus
                  - step_penalty)

        # ---- Telemetry: per-step means of each reward COMPONENT (signed contribution) ----
        # For W&B + tuning. Means over [N, M]. Cheap; detached scalars.
        reward_terms = {
            "novel":         (a_novel * novel_scan).mean(),
            "scan_self_diag": scan_self_delta.mean(),     # diagnostic only, not in reward
            "team":          (beta * team_delta_node).mean(),
            "give":          (z_give * give_bonus).mean(),
            "recv":          (z_recv * recv_bonus).mean(),
            "yield":         (a_yield * yield_bonus).mean(),
            "overlap":       (-eta_lap * overlap_pen).mean(),
            "revisit":       (-gamma * revisit_pen).mean(),
            "proximity":     (-eps_prox * proximity_pen).mean(),
            "target_switch": (-delta_obj * target_switch_pen).mean(),
            "stall":         (-delta_stall * stall_pen).mean(),
            "progress":      (self.cfg.progress_reward_coef * progress_reward).mean(),
        }

        # ---- Exploration-quality metrics (per-step scalars; driver aggregates) ----
        # redundancy = (Σ_a own_free − union_free) / union_free  (overlap; low = good).
        # MUST use PRE-fusion per-agent maps: post-fusion both in-comm agents share an
        # identical map → own_sum ≈ M·union → redundancy pinned near M−1 (measures map
        # sharing, not redundant exploration). free_node_pre is each agent's own holdings
        # before this step's fusion, so it reflects genuine independent coverage divergence.
        own_free_sum = free_node_pre.float().sum(-1).sum(-1)                                 # [N]
        union_node_pre = free_node_pre.any(dim=1).float().sum(-1)                            # [N]
        redundancy = ((own_free_sum - union_node_pre) / union_node_pre.clamp(min=1.0))       # [N]
        # mean pairwise inter-agent distance / canvas_diag (separation; chase = low).
        canvas_diag = float((self.H ** 2 + self.W ** 2) ** 0.5)
        if self.M > 1:
            pd = torch.cdist(self.pos, self.pos)                                             # [N, M, M]
            triu = torch.triu(torch.ones(self.M, self.M, device=self.dev), diagonal=1).bool()
            mean_pair_dist = (pd[:, triu].mean(-1) / canvas_diag)                             # [N]
        else:
            mean_pair_dist = torch.zeros(self.N, device=self.dev)
        # comm_duty_cycle: fraction of pairs currently in comm (off-diagonal mean).
        # Persistent ≈1.0 = chase signature. sensing_overlap: pair LiDAR disks physically
        # overlap (dist < 2·sensor_range) — MARVEL's overlap ratio; immune to fusion history.
        if self.M > 1:
            offdiag = ~torch.eye(self.M, dtype=torch.bool, device=self.dev)
            comm_duty = comm_mask[:, offdiag].float().mean()
            sens_overlap = (pd[:, triu] < 2.0 * self.cfg.sensor_range_px).float().mean()
        else:
            comm_duty = torch.zeros((), device=self.dev)
            sens_overlap = torch.zeros((), device=self.dev)
        metrics = {
            "redundancy":     redundancy.mean(),
            "stall_rate":     stall_pen.mean(),
            "revisit_rate":   is_recent_revisit.float().mean(),
            "mean_pair_dist": mean_pair_dist.mean(),
            "comm_duty_cycle":     comm_duty,
            "sensing_overlap":     sens_overlap,
            "team_delta_sum": team_delta.sum(),                   # Σ_N Δunion frac this step (efficiency num)
            "step_disp_sum":  step_disp.sum(),                    # Σ_{N,M} displacement px (efficiency denom)
            # Per-step work proxy: fraction of envs where ALL agents found ≥1 union-new cell
            # this step (training trend for the alternation/idle problem; eval/concurrency is
            # the windowed, authoritative version).
            "both_active":    (novel_count > 0).all(dim=1).float().mean() if self.M > 1
                              else (novel_count > 0).float().mean(),
        }

        self._refresh_obs(comm_mask)

        truncated  = self.t >= self.cfg.max_episode_steps
        terminated = explored_rate >= self.cfg.done_explored_thresh
        done = truncated | terminated
        info = {
            "explored_rate": explored_rate,
            "terminated":    terminated,
            "truncated":     truncated,
            "step":          self.t.clone(),
            "reward_terms":  reward_terms,
            "metrics":       metrics,
            # Per-agent union-new cells found so far this episode [N, M] — snapshot taken
            # BEFORE auto-reset so episode-end contribution shares are readable at done.
            "novel_cells_ep": self.novel_cells_ep.clone(),
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
        if self.cfg.force_full_comm:
            return torch.ones((N, M, M), dtype=torch.bool, device=self.dev)

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
        """Update last_known_pos (point belief) and t_last_comm (σ-inflation timer).

        Normally gated by comm_mask. If cfg.force_full_pos_sharing is True, ALL pairs
        get fresh pos every step (debug only — used to test if chase/weird-movement
        bugs come from stale lkp). Map fusion remains comm-gated regardless.
        """
        t_now = self.t                                   # [N]
        # Post-fusion explored-cell count per agent — snapshot at sync drives the offer metric.
        own_expl = (self.world.occupancy_torch != _UNKNOWN).view(self.N, self.M, -1).sum(-1).float()  # [N, M]
        if self.cfg.force_full_pos_sharing:
            # Update all (i, j) pairs unconditionally to actual positions.
            for i in range(self.M):
                for j in range(self.M):
                    self.last_known_pos[:, i, j] = self.pos[:, j, :]
                    self.last_known_target[:, i, j] = self._prev_target_node[:, j]
                    self.t_last_comm[:, i, j] = t_now
                    self._own_expl_at_comm[:, i, j] = own_expl[:, i]
            return
        for i in range(self.M):
            for j in range(self.M):
                can = comm_mask[:, i, j]                # [N]
                if not can.any():
                    continue
                new_pos = self.pos[:, j, :]             # [N, 2] — actual current position
                mask2d  = can.view(-1, 1)
                self.last_known_pos[:, i, j] = torch.where(
                    mask2d.expand(-1, 2), new_pos, self.last_known_pos[:, i, j]
                )
                self.last_known_target[:, i, j] = torch.where(
                    can, self._prev_target_node[:, j], self.last_known_target[:, i, j]
                )
                self.t_last_comm[:, i, j] = torch.where(
                    can, t_now, self.t_last_comm[:, i, j]
                )
                # Snapshot i's explored count at this sync (offer baseline toward j).
                self._own_expl_at_comm[:, i, j] = torch.where(
                    can, own_expl[:, i], self._own_expl_at_comm[:, i, j]
                )

    # ---------------------------------------------------------------------- #
    # internals                                                               #
    # ---------------------------------------------------------------------- #
    def _spread_starts_graph(self, start_row: int, start_col: int, env_idx: int = 0) -> torch.Tensor:
        """Return M start positions [M, 2] on lattice-adjacent FREE nodes.

        Strategy:
          1. Anchor = nearest FREE lattice node to (start_row, start_col).
          2. For agents 2..M: pick from anchor's 8 graph neighbors (one lattice hop away),
             requiring (a) neighbor node FREE on GT, (b) segment from anchor to neighbor
             contains no obstacle pixel (S=5 sample points).
          3. If fewer than M-1 valid graph-neighbors of anchor: extend search to anchor's
             2-hop ring. Final fallback: reuse anchor (will trigger collision-revert at
             step 0 — better than placing agent on a wall).

        Prevents the prior bug where two agents could be picked at lattice nodes close
        in Euclidean distance but separated by a wall — those agents start on opposite
        sides of an obstacle, never see each other, never share maps.
        """
        from env.graph_lattice import NBR_OFFSETS

        dev = self.dev
        H, W = self.H, self.W
        LH, LW = self.graph.LH, self.graph.LW
        NR = self.cfg.nr
        gt = self.world.gt_torch[env_idx]                                # [H, W]
        node_xy = self.graph.node_xy                                     # [N_max, 2]

        # Anchor: nearest FREE lattice node to start.
        start_pos = torch.tensor([float(start_col), float(start_row)], device=dev)
        nx = node_xy[:, 0].long().clamp(0, W - 1)
        ny = node_xy[:, 1].long().clamp(0, H - 1)
        node_free = gt[ny, nx] == GT_FREE                                # [N_max]
        dist = (node_xy - start_pos.unsqueeze(0)).norm(dim=-1)
        dist_masked = torch.where(node_free, dist, torch.full_like(dist, float("inf")))
        anchor_flat = int(dist_masked.argmin().item())

        chosen: list[int] = [anchor_flat]

        if self.M > 1:
            # BFS over 8-conn lattice from anchor. Edge (u,v) passable iff:
            #   - v is FREE on GT (node_free[v])
            #   - segment u→v has no obstacle (S=5 samples on a one-hop segment is reliable)
            # Guarantees chosen nodes lie in anchor's connected FREE component.
            node_free_list = node_free.cpu().tolist()
            nx_arr = node_xy[:, 0].cpu().tolist()
            ny_arr = node_xy[:, 1].cpu().tolist()
            gt_cpu = gt.cpu()

            def segment_clear(ax: float, ay: float, bx: float, by: float) -> bool:
                S = 5
                for s in range(1, S + 1):
                    t = s / (S + 1.0)
                    sx = int(round(ax + t * (bx - ax)))
                    sy = int(round(ay + t * (by - ay)))
                    sx = max(0, min(W - 1, sx))
                    sy = max(0, min(H - 1, sy))
                    if int(gt_cpu[sy, sx].item()) != GT_FREE:
                        return False
                return True

            from collections import deque
            visited: set[int] = {anchor_flat}
            order: list[int] = []
            q: deque[int] = deque([anchor_flat])
            while q and len(order) < self.M * 4:
                u = q.popleft()
                u_li, u_lj = u // LW, u % LW
                ux, uy = nx_arr[u], ny_arr[u]
                for dr, dc in NBR_OFFSETS:
                    v_li, v_lj = u_li + dr, u_lj + dc
                    if not (0 <= v_li < LH and 0 <= v_lj < LW):
                        continue
                    v = v_li * LW + v_lj
                    if v in visited or not node_free_list[v]:
                        continue
                    if not segment_clear(ux, uy, nx_arr[v], ny_arr[v]):
                        continue
                    visited.add(v)
                    order.append(v)
                    q.append(v)

            # Pick first M-1 from BFS order (closest in graph hops from anchor).
            for v in order:
                if len(chosen) >= self.M:
                    break
                chosen.append(v)
            # Last-resort fallback: pad with anchor (collision-revert at step 0).
            while len(chosen) < self.M:
                chosen.append(anchor_flat)

        out = torch.zeros(self.M, 2, dtype=torch.float32, device=dev)
        for i, flat in enumerate(chosen):
            out[i] = node_xy[flat]
        return out

    def _spawn_degenerate(self, agent_pos: torch.Tensor) -> bool:
        """True if any two of the M start positions are co-located (< nr·0.5 apart) —
        i.e. `_spread_starts_graph` fell back to the anchor for lack of adjacent FREE nodes."""
        if self.M < 2:
            return False
        pd = torch.cdist(agent_pos.unsqueeze(0), agent_pos.unsqueeze(0))[0]   # [M, M]
        eye = torch.eye(self.M, dtype=torch.bool, device=agent_pos.device)
        return bool((pd[~eye] < float(self.cfg.nr) * 0.5).any().item())

    def _reset_all(self) -> None:
        self._reset_envs(list(range(self.N)))

    def reload_map(self, env_idx: int, map_idx: int) -> None:
        """G.1 — load specific map into env slot `env_idx` and do a FULL reset.

        Used by eval scripts so all stale state (BF cache, comm timers, reward baselines,
        last_meeting_node_mask, etc.) is cleared. Previously eval scripts only reset a
        subset → corrupted BF warm-start + strategic features.
        """
        gt_new, starts_new, fc_new = sample_batch(
            self.split, 1, indices=np.array([map_idx]),
            seed=int(self.rng.integers(0, 1 << 31)), device=self.dev,
        )
        # Overwrite the slot's map ingredients, then call _reset_envs which sets up
        # all per-agent state from scratch — but _reset_envs draws RANDOM map idx,
        # so we splice the requested map in afterwards.
        # Simpler path: invoke the shared reset path with this env_idx, then overwrite.
        self._reset_envs([env_idx])
        # Replace the random map _reset_envs used with the requested one.
        idx_t = torch.tensor([env_idx], dtype=torch.long, device=self.dev)
        self.world.gt_torch[idx_t]                    = gt_new
        self.world.occupancy_torch[idx_t]             = 0
        self.world.occupancy_logodds_torch[idx_t]     = 0.0
        self.free_total[idx_t]                        = fc_new.float()
        self.starts[idx_t]                            = starts_new
        self.visited_step[idx_t]                      = -1
        self.t[idx_t]                                 = 0
        self._target_prev[idx_t]                      = -1
        self._dist_prev[idx_t]                        = float("inf")
        self._curr_prev[idx_t]                        = -1
        self._dist_curr_prev[idx_t]                   = float("inf")
        self.t_last_comm[idx_t]                       = 0
        self.last_action[idx_t]                       = -1
        self._collision_key[idx_t]                    = torch.rand((1, self.M), device=self.dev)
        self._prev_target_node[idx_t]                 = -1
        self._prev_branch[idx_t]                      = -1
        self._steps_on_option[idx_t]                  = 0
        self._comm_event[idx_t]                       = False
        # Place agents using new map's start.
        row0, col0 = int(starts_new[0, 0]), int(starts_new[0, 1])
        agent_pos = self._spread_starts_graph(row0, col0, env_idx=env_idx)
        self.pos[env_idx] = agent_pos
        for ag in range(self.M):
            self.last_known_pos[env_idx, :, ag] = agent_pos[ag]
        self.last_known_target[env_idx] = -1
        self.world.set_positions(self.pos)
        self.world.scan()
        union_free = (self.world.occupancy_torch[idx_t] == _FREE).any(dim=1).view(1, -1).float().sum(-1)
        self.last_union[idx_t] = union_free
        # Lattice-level reward state.
        occ_flat = self.world.occupancy_torch[idx_t].view(1, self.M, -1)
        free_node = occ_flat[:, :, self._node_flat_idx] == _FREE
        self.last_own_free_node[idx_t] = free_node.float().sum(-1)
        self.own_node_mask_prev[idx_t] = free_node
        self.union_node_mask[idx_t]    = free_node.any(dim=1)
        self.novel_cells_ep[idx_t]     = 0.0
        union_node = free_node.any(dim=1, keepdim=False)
        self.last_meeting_node_mask[idx_t] = union_node.view(1, 1, 1, self.N_max).expand(
            -1, self.M, self.M, -1).contiguous()
        self._refresh_obs()

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
        # Reset guidepost cache: cold-start BF on next obs refresh.
        self._target_prev[idx_t]                      = -1
        self._dist_prev[idx_t]                        = float("inf")
        # Reset BF-from-curr cache.
        self._curr_prev[idx_t]                        = -1
        self._dist_curr_prev[idx_t]                   = float("inf")
        # H.3 — reset BF-from-teammate cache.
        self._team_node_prev[idx_t]                   = -1
        self._dist_team_prev[idx_t]                   = float("inf")
        # Reset comm-gap timer: at reset, last_known_pos is set to actual start positions
        # (see loop below), so all pairs are "freshly in comm" at t=0.
        self.t_last_comm[idx_t]                       = 0
        self.last_action[idx_t]                       = -1
        # Re-draw collision priority + clear objective-commitment memory.
        self._collision_key[idx_t]                    = torch.rand((n, self.M), device=self.dev)
        self._prev_target_node[idx_t]                 = -1
        self._prev_branch[idx_t]                      = -1
        self._steps_on_option[idx_t]                  = 0
        self._comm_event[idx_t]                       = False

        for j_env, e in enumerate(idx):
            row0, col0 = int(starts_new[j_env, 0]), int(starts_new[j_env, 1])
            agent_pos = self._spread_starts_graph(row0, col0, env_idx=e)  # [M, 2] on GPU
            # Reject degenerate spawns — if a map cannot fit M adjacent FREE nodes,
            # `_spread_starts_graph` pads with the anchor → agents CO-LOCATED → instant
            # collision/stall. Resample a different map (≤8 tries) instead. On train/easy
            # M=2 this never fires (audited 0%); matters on dense / M>2 splits.
            if self.M > 1:
                tries = 0
                while tries < 8 and self._spawn_degenerate(agent_pos):
                    tries += 1
                    r = int(self.rng.integers(0, self.split.n))
                    g2, s2, f2 = sample_batch(
                        self.split, 1, indices=np.array([r]),
                        seed=int(self.rng.integers(0, 1 << 31)), device=self.dev,
                    )
                    et = torch.tensor([e], dtype=torch.long, device=self.dev)
                    self.world.gt_torch[et]                = g2
                    self.world.occupancy_torch[et]         = 0
                    self.world.occupancy_logodds_torch[et] = 0.0
                    self.free_total[et]                    = f2.float()
                    self.starts[et]                        = s2
                    self.map_indices[et]                   = r
                    row0, col0 = int(s2[0, 0]), int(s2[0, 1])
                    agent_pos = self._spread_starts_graph(row0, col0, env_idx=e)
            self.pos[e] = agent_pos
            # All agents know all actual start positions (in comm range at reset)
            for ag in range(self.M):
                self.last_known_pos[e, :, ag] = agent_pos[ag]
            self.last_known_target[e] = -1

        self.world.set_positions(self.pos)
        self.world.scan()
        union_free = (self.world.occupancy_torch[idx_t] == _FREE).any(dim=1).view(n, -1).float().sum(-1)
        self.last_union[idx_t] = union_free
        # Phase D — lattice-level per-agent free count after first scan + per-pair baseline.
        occ_flat_reset = self.world.occupancy_torch[idx_t].view(n, self.M, -1)              # [n, M, H*W]
        free_node_reset = occ_flat_reset[:, :, self._node_flat_idx] == _FREE                # [n, M, N_max]
        self.last_own_free_node[idx_t] = free_node_reset.float().sum(-1)
        # v2 — novel-scan baselines: spawn scans are baseline, not credited.
        self.own_node_mask_prev[idx_t] = free_node_reset
        self.union_node_mask[idx_t]    = free_node_reset.any(dim=1)
        self.novel_cells_ep[idx_t]     = 0.0
        # Baseline per pair = node-level UNION of all agents' initial scans.
        # Treats reset as a "virtual rendezvous" where everyone knows what's been scanned.
        union_node_reset = free_node_reset.any(dim=1, keepdim=False)                        # [n, N_max]
        self.last_meeting_node_mask[idx_t] = union_node_reset.view(n, 1, 1, self.N_max).expand(
            -1, self.M, self.M, -1).contiguous()
        self._refresh_obs()

    def _refresh_obs(self, comm_mask: torch.Tensor | None = None) -> None:
        """Build per-agent obs from current per-agent occupancy + positions.

        Phase C: encoder consumes ego-centric subgraph windows, not the full lattice.
        Pass 1: build global graph + guidepost per agent (still need global for utility
                integral image, flood-fill reachability, target selection).
        Pass 2: cross-agent feat[5] (teammate_pos) — writes to global node_feat.
        Pass 3: extract local (2·n_hops + 3)² window per agent; this is what the model sees.
        """
        # ---- Pass 1: build global infos + warm-started target-rooted BF ----
        # Phase A v2 (A1): also extract top-K frontier candidates per agent for strategic head.
        # Argmax target selection here is the LEGACY path (used until Phase A v2 / A2 lands
        # the strategic head that will pick target from cand_*).
        K_cand = int(self.cfg.top_k_candidates)
        infos: list[dict] = []
        cand_list: list[dict] = []
        for a in range(self.M):
            occ_a      = self.world.occupancy_torch[:, a, :, :]
            frontier_a = compute_frontier(occ_a)
            info = self.graph.build(
                occupancy=occ_a,
                frontier=frontier_a,
                robot_xy=self.pos[:, a, :],
                visited_step=self.visited_step[:, a, :],
                current_step=int(self.t.max().item()),
            )
            # ---- BF FROM curr (target-INDEPENDENT) → path length to every node. Moved BEFORE
            # target selection so the analytic selector can score frontiers by true reach cost.
            curr_same = (info["curr_idx"] == self._curr_prev[:, a]).unsqueeze(-1)
            curr_dist_init = torch.where(
                curr_same.expand(-1, self.N_max),
                self._dist_curr_prev[:, a, :],
                torch.full_like(self._dist_curr_prev[:, a, :], float("inf")),
            )
            bf_dist_from_curr, bf_parent_from_curr = self.graph.bf_from_target(
                info, target=info["curr_idx"], dist_init=curr_dist_init,
            )
            self._curr_prev[:, a]      = info["curr_idx"]
            self._dist_curr_prev[:, a] = bf_dist_from_curr
            info["bf_dist_from_curr"]  = bf_dist_from_curr
            info["bf_parent_from_curr"] = bf_parent_from_curr   # [N, N_max] predecessor on path from curr
            info["_bf_parent_for_first_hop"] = bf_parent_from_curr   # used after cand extraction
            # H.3 — BF FROM each teammate's last-known position (in agent a's own map).
            # Per teammate j != a: lkp_node = floor(lkp[a, j] / NR), then BF rooted there.
            # Warm-startable. Result stored in info["bf_dist_team"][N, M, N_max] with self-slot
            # left as +inf (unused).
            if self.M > 1:
                LH, LW = self.graph.LH, self.graph.LW
                NR = float(self.cfg.nr)
                lkp_a = self.last_known_pos[:, a]                                          # [N, M, 2]
                lj_t = (lkp_a[..., 0] / NR).long().clamp(0, LW - 1)
                li_t = (lkp_a[..., 1] / NR).long().clamp(0, LH - 1)
                team_node_a = li_t * LW + lj_t                                              # [N, M]
                bf_dist_team_per_j = torch.full(
                    (self.N, self.M, self.N_max), float("inf"),
                    dtype=torch.float32, device=self.dev,
                )
                for j in range(self.M):
                    if j == a:
                        continue
                    target_j = team_node_a[:, j]                                            # [N]
                    same_j = (target_j == self._team_node_prev[:, a, j]).unsqueeze(-1)
                    team_dist_init = torch.where(
                        same_j.expand(-1, self.N_max),
                        self._dist_team_prev[:, a, j],
                        torch.full_like(self._dist_team_prev[:, a, j], float("inf")),
                    )
                    dist_j, _ = self.graph.bf_from_target(
                        info, target=target_j, dist_init=team_dist_init,
                        # Optimistic (UNKNOWN-passable) graph: teammate usually sits in
                        # this agent's unexplored region; the FREE graph would return +inf
                        # and silence the coordination channel. None-safe (M==1 → key
                        # absent → FREE graph). See graph_lattice.build() step 3b.
                        edge_valid=info.get("edge_valid_optim"),
                    )
                    self._team_node_prev[:, a, j] = target_j
                    self._dist_team_prev[:, a, j] = dist_j
                    bf_dist_team_per_j[:, j] = dist_j
                info["bf_dist_team"] = bf_dist_team_per_j                                   # [N, M, N_max]
            # ---- Analytic GLOBAL TARGET (deterministic) — nearest-richest frontier, with a
            # rendezvous pull toward the teammate I owe the most fresh map. offer≈0 → pure
            # exploration; offer high → drift the chosen frontier toward that teammate so we
            # converge for a map exchange (self-limiting: fusion on contact resets offer→0).
            d_team_star = None
            w_offer = None
            if self.M > 1:
                # offer_j = explored cells I gained since I last synced with j (j lacks them).
                own_expl_now = (occ_a != _UNKNOWN).view(self.N, -1).sum(-1).float()         # [N]
                offer_j = (own_expl_now.unsqueeze(1) - self._own_expl_at_comm[:, a, :]).clamp(min=0.0)  # [N, M]
                offer_j[:, a] = -1.0                                                        # mask self slot
                j_star = offer_j.argmax(dim=1)                                              # [N] teammate owed most
                offer_star = offer_j.gather(1, j_star.unsqueeze(1)).squeeze(1).clamp(min=0.0)  # [N]
                offer_scale = max(1.0, self.cfg.rdv_offer_frac * float(self.H * self.W))
                w_offer = (offer_star / offer_scale).clamp(0.0, 1.0)
                d_team_star = info["bf_dist_team"].gather(
                    1, j_star.view(self.N, 1, 1).expand(-1, 1, self.N_max)
                ).squeeze(1)                                                               # [N, N_max]
            new_target = self.graph.select_target_analytic(
                info["utility"], info["node_valid"], bf_dist_from_curr,
                d_team=d_team_star, w=w_offer,
                beta=float(self.cfg.target_beta), lam=float(self.cfg.target_lambda),
                prev_target=self._prev_target_node[:, a], curr_idx=info["curr_idx"],
                keep_margin=float(self.cfg.target_keep_margin),
            )
            # Guidepost (BF path) to the analytic target — warm-started on unchanged target.
            target_same = (new_target == self._target_prev[:, a]).unsqueeze(-1)            # [N, 1]
            dist_init = torch.where(
                target_same.expand(-1, self.N_max),
                self._dist_prev[:, a, :],
                torch.full_like(self._dist_prev[:, a, :], float("inf")),
            )
            self.graph.build_guidepost_v2(info, target=new_target, dist_init=dist_init)
            self._target_prev[:, a] = new_target
            self._dist_prev[:, a, :] = info["guidepost_dist"]
            # Analytic mode: env owns the committed target → record it for last_known_target
            # (teammates' belief of where I'm headed) and commitment-step tracking.
            if self.cfg.analytic_target:
                same_tgt = (new_target == self._prev_target_node[:, a])
                self._steps_on_option[:, a] = torch.where(
                    same_tgt, self._steps_on_option[:, a] + 1,
                    torch.zeros_like(self._steps_on_option[:, a]),
                )
                self._prev_target_node[:, a] = new_target
            infos.append(info)
            cand_a = self.graph.extract_topk_candidates(
                info["utility"], info["node_valid"], curr_xy=self.pos[:, a, :], K=K_cand,
                bf_dist=bf_dist_from_curr,
            )
            # G.3.c — for each cand, compute K=8 first-hop slot via BF parent walk.
            # Walk parent from cand_idx back; node whose parent==curr is first hop.
            cand_idx_a = cand_a["cand_idx"]                                      # [N, K_cand]
            curr_idx_a = info["curr_idx"]                                        # [N]
            parent_a = bf_parent_from_curr                                       # [N, N_max]
            curr_nbr_a_global = info["curr_nbr"]                                 # [N, K=8] global flat idx (-1 padded)
            cur = cand_idx_a.clamp(min=0)                                        # [N, K_cand]
            # Walk: stop when parent[cur] == curr OR parent[cur] < 0.
            for _walk in range(self.N_max):
                par = torch.gather(parent_a, dim=1, index=cur)                   # [N, K_cand]
                reached = (par == curr_idx_a.unsqueeze(-1))                      # cur is first-hop
                invalid = (par < 0)
                stop = reached | invalid
                cur = torch.where(stop, cur, par)
                if bool(stop.all().item()):
                    break
            first_hop_node = cur                                                  # [N, K_cand]
            # Match against curr's K=8 neighbors to get K-slot.
            match = (first_hop_node.unsqueeze(-1) == curr_nbr_a_global.unsqueeze(1))  # [N, K_cand, K=8]
            any_match = match.any(dim=-1)                                         # [N, K_cand]
            slot = match.float().argmax(dim=-1)                                   # [N, K_cand]
            first_hop_onehot = torch.nn.functional.one_hot(slot, num_classes=self.K).float()
            first_hop_onehot = first_hop_onehot * (any_match.float().unsqueeze(-1) * cand_a["cand_valid"].float().unsqueeze(-1))
            cand_a["cand_bf_first_hop_onehot"] = first_hop_onehot                 # [N, K_cand, K=8]

            # H.3 — per-cand BF dist FROM each teammate j.
            # H.2 — joint distribution alternative score (M-agnostic).
            if self.M > 1:
                cand_valid_f = cand_a["cand_valid"].float()                                  # [N, K_cand]
                cand_util_a = cand_a["cand_utility"]                                          # [N, K_cand]
                safe_idx = cand_idx_a.clamp(min=0)
                # Per-teammate BF dist to cand.
                cand_team_bf_per_j = torch.full(
                    (self.N, self.M, K_cand), float("inf"),
                    dtype=torch.float32, device=self.dev,
                )
                alt_per_j = torch.zeros(
                    (self.N, self.M, K_cand), dtype=torch.float32, device=self.dev,
                )
                for j in range(self.M):
                    if j == a:
                        continue
                    dist_j_full = info["bf_dist_team"][:, j]                                  # [N, N_max]
                    d_per_cand = torch.gather(dist_j_full, dim=1, index=safe_idx)             # [N, K_cand]
                    d_per_cand = torch.where(
                        torch.isfinite(d_per_cand), d_per_cand, torch.full_like(d_per_cand, 1.0e6)
                    )
                    cand_team_bf_per_j[:, j] = d_per_cand * cand_valid_f
                    # Teammate j's score for each cand:
                    team_score_j = cand_util_a / (1.0 + d_per_cand / float(self.cfg.nr))      # [N, K_cand]
                    team_score_j = team_score_j * cand_valid_f
                    # Best alternative: max over k' != k. Use top-2 trick.
                    top2_vals, top2_idx = team_score_j.topk(2, dim=-1)                        # [N, 2]
                    # For each k, best_alt = top2_vals[..., 0] if k != top2_idx[..., 0] else top2_vals[..., 1]
                    arange_k = torch.arange(K_cand, device=self.dev).view(1, K_cand).expand(self.N, K_cand)
                    is_top1 = (arange_k == top2_idx[..., 0:1])                                # [N, K_cand]
                    best_alt = torch.where(is_top1, top2_vals[..., 1:2], top2_vals[..., 0:1]).squeeze(-1)
                    alt_per_j[:, j] = (best_alt - team_score_j).clamp(min=0.0) * cand_valid_f
                # Min over teammates of BF dist (self slot already +inf from init).
                cand_min_team_bf_dist = cand_team_bf_per_j.min(dim=1).values                   # [N, K_cand]
                cand_min_team_bf_dist = torch.where(
                    torch.isfinite(cand_min_team_bf_dist),
                    cand_min_team_bf_dist,
                    torch.zeros_like(cand_min_team_bf_dist),
                )
                # Mean over teammates of alt score (excludes self via 0-init for j==a).
                # Number of valid teammates = M - 1.
                alt_score_mean = alt_per_j.sum(dim=1) / max(1, self.M - 1)                     # [N, K_cand]
                cand_a["cand_min_team_bf_dist"] = cand_min_team_bf_dist
                cand_a["cand_team_alt_score"] = alt_score_mean
            else:
                cand_a["cand_min_team_bf_dist"] = torch.zeros(
                    (self.N, K_cand), dtype=torch.float32, device=self.dev,
                )
                cand_a["cand_team_alt_score"] = torch.zeros(
                    (self.N, K_cand), dtype=torch.float32, device=self.dev,
                )
            cand_list.append(cand_a)

        # ---- Pass 2: cross-agent feat[4] = teammate-proximity POTENTIAL on GLOBAL node_feat ----
        # The old feat lit a SINGLE one-hot node nearest the teammate's last-known position.
        # Once agents split (the whole point of the task) the teammate sits outside the agent's
        # (2·n_hops+3)² ego window, so after window extraction feat[5] ≡ 0 — dead by construction,
        # and a one-hot carries near-zero gradient even when alive. Replace with a DENSE field:
        # exp(-d / scale) of the BF geodesic distance from the teammate's lkp to each node
        # (info["bf_dist_team"], built in Pass 1 over the optimistic FREE∪UNKNOWN graph so it
        # stays finite when the teammate is in unexplored space). Always informative inside the
        # window, gradient-rich, wall-aware, and pointing toward the teammate.
        if self.M > 1:
            scale_px = max(1.0, 4.0 * float(self.cfg.nr))
            for a in range(self.M):
                # min over teammates (self slot left at +inf in Pass 1) → nearest teammate.
                d_min = infos[a]["bf_dist_team"].min(dim=1).values            # [N, N_max]
                pot = torch.exp(-d_min / scale_px)                            # +inf → 0
                pot = torch.nan_to_num(pot, nan=0.0, posinf=0.0, neginf=0.0)
                infos[a]["node_feat"][..., 4] = pot * infos[a]["node_valid"].float()

        # ---- Render-global stash (eval/debug only) — full-graph utility/validity for the GIF,
        # since obs ships only the ego window. Gated so training pays nothing. ----
        if self.store_render_global:
            self._render_global = {
                "node_xy":    self.graph.node_xy,                                              # [N_max, 2] static
                "edge_idx":   self.graph.edge_idx_static,                                      # [N_max, K] static
                "window_idx_table": self.graph.window_idx_table,                               # [N_max, W²] global idx (-1 pad)
                "utility":    torch.stack([infos[a]["utility"]    for a in range(self.M)], 1),  # [N, M, N_max]
                "node_valid": torch.stack([infos[a]["node_valid"] for a in range(self.M)], 1),  # [N, M, N_max]
                "edge_valid": torch.stack([infos[a]["edge_valid"] for a in range(self.M)], 1),  # [N, M, N_max, K]
                "curr_idx":   torch.stack([infos[a]["curr_idx"]   for a in range(self.M)], 1),  # [N, M] GLOBAL node
            }

        # ---- Pass 3: extract per-agent local window ----
        node_xy_list = []
        node_valid_list = []
        node_feat_list = []
        edge_idx_list = []
        edge_valid_list = []
        curr_idx_list = []
        curr_nbr_list = []
        curr_nbr_valid_list = []
        utility_list = []
        curr_nbr_global_list = []
        local_to_global_list = []
        guidepost_target_list = []
        guidepost_target_xy_list = []
        guidepost_path_xy_list = []
        guidepost_path_valid_list = []
        guidepost_nbr_bias_list = []
        guidepost_next_hop_list = []
        # G.3.b — BF parent from curr, for render-time path walk to strategic target.
        bf_parent_list = []
        curr_idx_global_list = []

        for a in range(self.M):
            info = infos[a]
            local = self.graph.extract_local_window(info)
            node_xy_list.append(local["node_xy_local"])
            node_valid_list.append(local["node_valid_local"])
            node_feat_list.append(local["node_feat_local"])
            edge_idx_list.append(local["edge_idx_local"])
            edge_valid_list.append(local["edge_valid_local"])
            curr_idx_list.append(local["curr_idx_local"])
            curr_nbr_list.append(local["curr_nbr_local"])
            curr_nbr_valid_list.append(local["curr_nbr_valid_local"])
            utility_list.append(local["utility_local"])
            curr_nbr_global_list.append(local["curr_nbr_global"])
            local_to_global_list.append(local["local_to_global"])
            guidepost_target_list.append(info["guidepost_target"])
            guidepost_target_xy_list.append(info["guidepost_target_xy"])
            guidepost_path_xy_list.append(info["guidepost_path_xy"])
            guidepost_path_valid_list.append(info["guidepost_path_valid"])
            guidepost_nbr_bias_list.append(info["guidepost_nbr_bias"])
            guidepost_next_hop_list.append(info["guidepost_next_hop"])
            bf_parent_list.append(info["bf_parent_from_curr"])
            curr_idx_global_list.append(info["curr_idx"])

        # ---- Phase A v2 / A1: assemble candidate tensors + teammate features ----
        # Stack per-agent cand dicts into [N, M, K, ...] tensors.
        cand_idx_all     = torch.stack([c["cand_idx"]     for c in cand_list], dim=1)  # [N, M, K]
        cand_bf_first_hop_all = torch.stack(
            [c["cand_bf_first_hop_onehot"] for c in cand_list], dim=1
        )  # [N, M, K_cand, K=8]
        cand_xy_all      = torch.stack([c["cand_xy"]      for c in cand_list], dim=1)  # [N, M, K, 2]
        cand_utility_all = torch.stack([c["cand_utility"] for c in cand_list], dim=1)  # [N, M, K]
        cand_valid_all   = torch.stack([c["cand_valid"]   for c in cand_list], dim=1)  # [N, M, K]
        cand_rel_xy_all  = torch.stack([c["cand_rel_xy"]  for c in cand_list], dim=1)  # [N, M, K, 2]
        cand_euclid_all  = torch.stack([c["cand_euclid"]  for c in cand_list], dim=1)  # [N, M, K]
        # H.3 — BF min-team-dist replaces euclidean. H.2 — alt score for joint distribution.
        cand_min_team_dist = torch.stack([c["cand_min_team_bf_dist"] for c in cand_list], dim=1)  # [N, M, K]
        cand_team_alt_all  = torch.stack([c["cand_team_alt_score"]   for c in cand_list], dim=1)  # [N, M, K]

        # ---- Target-claim deconfliction (in-comm, lowest agent-ID priority) ----
        # If a LOWER-ID teammate j<i that is in comm has claimed a target node that also
        # appears among agent i's candidates, mask that candidate so i picks a DIFFERENT
        # frontier — UNLESS it is i's only option (single-frontier guard: both must commit,
        # never back down). Decentralized: uses comm-gated last_known_target only.
        if self.M > 1 and comm_mask is not None:
            lower = torch.tril(torch.ones(self.M, self.M, dtype=torch.bool, device=self.dev), -1)  # [i,j] True if j<i
            active = (lower.view(1, self.M, self.M) & comm_mask
                      & (self.last_known_target >= 0))                                # [N, M, M]
            claimed = torch.where(active, self.last_known_target,
                                  torch.full_like(self.last_known_target, -1))        # [N, M, M]
            match = (cand_idx_all.unsqueeze(2) == claimed.unsqueeze(-1)).any(dim=2)    # [N, M, K]
            proposed = match & cand_valid_all                                          # only valid slots
            keep_ok = (cand_valid_all.sum(-1) - proposed.sum(-1)) >= 1                 # leave >=1 valid
            final_mask = proposed & keep_ok.unsqueeze(-1)
            cand_valid_all = cand_valid_all & ~final_mask                              # higher-ID diverts

        # Comm-gap feature: max over j != a of (t - t_last_comm[a, j]).
        if self.M > 1:
            eye = torch.eye(self.M, dtype=torch.bool, device=self.dev).view(1, self.M, self.M)
            gap = (self.t.view(self.N, 1, 1) - self.t_last_comm).clamp(min=0).float()  # [N, M, M]
            gap = torch.where(eye, torch.full_like(gap, -1.0), gap)
            cand_max_comm_gap_per_agent = gap.max(dim=-1).values.clamp(min=0.0)         # [N, M]
            cand_max_comm_gap = cand_max_comm_gap_per_agent.unsqueeze(-1).expand(-1, -1, K_cand)
        else:
            cand_max_comm_gap  = torch.zeros_like(cand_euclid_all)

        # Fix A: per-cand "am I closer than nearest teammate?" signal.
        # own_dist = ||cand - own_pos|| (already cand_euclid_all).
        # delta = own_dist - cand_min_team_dist. Negative = I'm closer → I should take it.
        # Positive = teammate closer → I should yield. Decentralized (each uses own lkp).
        canvas_diag = float((self.H ** 2 + self.W ** 2) ** 0.5)
        # G.4.a — own_dist uses BF (cand_euclid_all = bf_dist_from_curr[cand] post-Option A);
        # team_dist remains euclidean to lkp. Amplified by scale before clamp to give the
        # strategic head a stronger yield signal (faster learning, no hard threshold).
        cand_own_minus_team_raw = (cand_euclid_all - cand_min_team_dist) / canvas_diag
        cand_own_minus_team = cand_own_minus_team_raw * float(self.cfg.cand_own_minus_team_scale)
        # prev_branch_match: 1.0 if this candidate's BF first-hop branch equals the branch the
        # agent committed to last step → gives the feedforward strategic head memory of its own
        # direction so it stops re-thrashing the target every step (the deterministic limit cycle).
        cand_branch = cand_bf_first_hop_all.argmax(-1)                           # [N, M, K]
        cand_branch_valid = cand_bf_first_hop_all.sum(-1) > 0                    # [N, M, K]
        prev_branch_match = (
            (cand_branch == self._prev_branch.unsqueeze(-1))
            & cand_branch_valid
            & (self._prev_branch >= 0).unsqueeze(-1)
        ).float()                                                                # [N, M, K]
        cand_feat_all = torch.stack([
            cand_rel_xy_all[..., 0] / canvas_diag,
            cand_rel_xy_all[..., 1] / canvas_diag,
            cand_utility_all,                                       # already in [0, 1]
            cand_euclid_all / canvas_diag,
            cand_min_team_dist / canvas_diag,                       # H.3 — BF dist (was euclid)
            (cand_max_comm_gap / max(1.0, float(self.cfg.max_episode_steps))).clamp(max=1.0),
            cand_own_minus_team.clamp(-1.0, 1.0),                   # Fix A — yielding signal
            cand_team_alt_all.clamp(0.0, 1.0),                       # H.2 — joint alt score
            prev_branch_match,                                       # direction commitment memory
        ], dim=-1)                                                  # [N, M, K, 9]
        # Mask out invalid candidate slots' features (zero them).
        cand_feat_all = cand_feat_all * cand_valid_all.unsqueeze(-1).float()

        node_xy              = torch.stack(node_xy_list, dim=1)
        node_valid           = torch.stack(node_valid_list, dim=1)
        node_feat            = torch.stack(node_feat_list, dim=1)
        edge_idx             = torch.stack(edge_idx_list, dim=1)
        edge_valid           = torch.stack(edge_valid_list, dim=1)
        curr_idx             = torch.stack(curr_idx_list, dim=1)
        curr_nbr             = torch.stack(curr_nbr_list, dim=1)
        curr_nbr_valid       = torch.stack(curr_nbr_valid_list, dim=1)
        utility              = torch.stack(utility_list, dim=1)
        curr_nbr_global      = torch.stack(curr_nbr_global_list, dim=1)
        local_to_global      = torch.stack(local_to_global_list, dim=1)
        guidepost_target     = torch.stack(guidepost_target_list, dim=1)
        guidepost_target_xy  = torch.stack(guidepost_target_xy_list, dim=1)
        guidepost_path_xy    = torch.stack(guidepost_path_xy_list, dim=1)
        guidepost_path_valid = torch.stack(guidepost_path_valid_list, dim=1)
        guidepost_nbr_bias   = torch.stack(guidepost_nbr_bias_list, dim=1)
        guidepost_next_hop   = torch.stack(guidepost_next_hop_list, dim=1)
        bf_parent_from_curr  = torch.stack(bf_parent_list, dim=1)         # [N, M, N_max]
        curr_idx_global      = torch.stack(curr_idx_global_list, dim=1)   # [N, M]

        self.curr_idx = curr_idx
        self.curr_idx_global = curr_idx_global   # [N, M] real lattice node — invalid-action fallback
        if comm_mask is None:
            comm_mask = torch.eye(
                self.M, dtype=torch.bool, device=self.dev
            ).view(1, self.M, self.M).expand(self.N, -1, -1)

        self._last_obs = {
            "node_xy":              node_xy,            # LOCAL [N, M, W², 2]
            "node_valid":           node_valid,         # LOCAL [N, M, W²]
            "node_feat":            node_feat,          # LOCAL [N, M, W², F]
            "edge_idx":             edge_idx,           # LOCAL [N, M, W², K]
            "edge_valid":           edge_valid,         # LOCAL [N, M, W², K]
            "curr_idx":             curr_idx,           # LOCAL [N, M] = constant window center
            "curr_nbr":             curr_nbr,           # LOCAL [N, M, K]
            "curr_nbr_valid":       curr_nbr_valid,
            "action_mask":          curr_nbr_valid,
            "utility":              utility,            # LOCAL [N, M, W²]
            "curr_nbr_global":      curr_nbr_global,    # GLOBAL [N, M, K] — for env.step action decode
            "local_to_global":      local_to_global,    # [N, M, W²] global flat idx (or -1) per local slot
            "guidepost_target":     guidepost_target,
            "guidepost_target_xy":  guidepost_target_xy,    # GLOBAL world coords [N, M, 2]
            "guidepost_path_xy":    guidepost_path_xy,
            "guidepost_path_valid": guidepost_path_valid,
            "guidepost_nbr_bias":   guidepost_nbr_bias,
            "guidepost_next_hop":   guidepost_next_hop,
            "bf_parent_from_curr":  bf_parent_from_curr,    # G.3.b — [N, M, N_max] for render path
            "curr_idx_global":      curr_idx_global,        # G.3.b — [N, M] global flat idx for walk anchor
            "pos":                  self.pos.clone(),
            "comm_mask":            comm_mask,
            "last_known_pos":       self.last_known_pos.clone(),
            # Phase A v2 / A1: strategic-head inputs.
            "cand_idx":             cand_idx_all,       # GLOBAL flat idx  [N, M, K]
            "cand_xy":              cand_xy_all,        # world coords     [N, M, K, 2]
            "cand_utility":         cand_utility_all,   # raw utility      [N, M, K]
            "cand_valid":           cand_valid_all,     # bool             [N, M, K]
            "cand_feat":            cand_feat_all,      # [N, M, K, 7] normalized features
            "cand_bf_first_hop":    cand_bf_first_hop_all,  # G.3.c [N, M, K_cand, K=8]
            # Fix B: previous action one-hot per agent.
            "prev_action":          self._prev_action_onehot(),    # [N, M, K=8] float
            # Phase 1 (interrupting options) — strategic-head commitment state.
            "committed_node":       self._prev_target_node.clone(),    # [N, M] GLOBAL node last committed (-1 none)
            "committed_steps":      self._steps_on_option.clone(),     # [N, M] consecutive steps on that node
            "comm_event":           self._comm_event.clone(),          # [N, M] received teammate map cells this step
        }

    def _prev_action_onehot(self) -> torch.Tensor:
        """One-hot [N, M, K=8] of last_action. Zero everywhere when last_action == -1."""
        K = self.K
        out = torch.zeros((self.N, self.M, K), dtype=torch.float32, device=self.dev)
        valid = self.last_action >= 0
        safe = self.last_action.clamp(min=0)
        out.scatter_(2, safe.unsqueeze(-1), valid.float().unsqueeze(-1))
        return out
