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
    sensor_range_px: float = 80.0   # realistic 2D-LiDAR reach (matches IR2 SENSOR_RANGE)
    n_rays: int = 720
    utility_range_px: int = 30
    visit_age_window: int = 16               # feat[3] recency horizon (steps): walked node ramps 0→1 freshness
    # Analytic global-target selection (deterministic). score = util/(1+β·d_curr/NR)·(1+λ·w·rdv).
    target_beta: float = 1.0                 # β: distance discount (per NR node-unit)
    target_lambda: float = 1.0               # λ: rendezvous pull strength (0 = pure exploration)
    rdv_offer_frac: float = 0.15             # offer saturates (w→1) at this fraction of map cells gained since sync
    target_keep_margin: float = 0.2          # commit: keep last target unless a new one beats it by >this fraction
    # Separation / ownership term (analytic target): down-weight frontiers a teammate is closer to,
    # so agents spread to different regions (division of labor). own_gate = σ(k·(adv−m)), adv =
    # (d_team_min − d_curr)/NR in hops; sep_mult = 1 − w·(1−own_gate). w=0 → off (today's behavior).
    target_sep_weight: float = 0.5           # strength ∈[0,1]; 0 disables
    target_sep_from_offer: bool = False      # if True: sep_w = (1−w_offer) per-env (separate when I have
                                             # nothing fresh to give; converge when I do). Overrides the
                                             # fixed target_sep_weight scalar. M>1 only.
    target_sep_k: float = 2.0                # sigmoid sharpness (per hop)
    target_sep_margin: float = 0.0           # hops a teammate must be closer before I yield
    analytic_target: bool = True             # env owns the deterministic global target (always True since the StrategicHead was removed; kept as a guard for the bookkeeping branches)
    # Which env-owned target rule to use when analytic_target=True:
    #   "analytic" — nearest-richest frontier scored by util/(1+β·d)·rendezvous·separation (default).
    #   "nearest"  — the closest reachable frontier by BF path distance (greedy-nearest, no scoring,
    #                no rendezvous/separation). The guidepost just points at the nearest unexplored
    #                boundary. Ignored when analytic_target=False (learned StrategicHead).
    target_kind: str = "analytic"
    # Bellman-Ford caps for the guidepost / BF distance fields (0 = auto from canvas size). Raise
    # guidepost_iters if a very long maze geodesic still truncates (auto = N_max already covers it).
    guidepost_iters: int = 0
    guidepost_path_max: int = 0
    num_sim_steps: int = 5
    max_episode_steps: int = 512
    flood_max_iters: int = 200
    done_explored_thresh: float = 0.99
    comm_range_px: float = 120.0        # [comm_model="los" only] hard Euclidean cutoff (px)
    comm_los_samples: int = 40          # line samples along the a→b segment (Bresenham approx)
    # --- Communication model ---
    # "los"             : legacy — connect iff dist < comm_range_px AND no GT obstacle on the segment (hard block).
    # "signal_strength" : realistic log-distance path-loss radio model (IR2 / hal-03365129). Walls ATTENUATE
    #                     (γ_obst) rather than hard-block; free space uses γ. Connect iff received power
    #                     P_R = P_T − PL > ss_thresh. Per-episode shadowing noise (X_g, K) is resampled at
    #                     each env reset → comm range varies episode-to-episode (domain randomization), exactly
    #                     like IR2. No fixed comm_range_px is used in this mode.
    # Default "los" for back-compat: pre-SS checkpoints lack this key, so from_ckpt_dict falls
    # back here and their eval still mirrors how they trained. run_train.py's CLI default is
    # "signal_strength", so NEW trainings opt into the realistic model and persist it in the ckpt.
    comm_model: str = "los"
    ss_p_t: float = -20.0               # tx power (dBm)
    ss_thresh: float = -70.0            # rx sensitivity threshold (dBm): connect iff P_R > this
    ss_gamma: float = 2.0               # path-loss exponent, free space
    ss_gamma_obst: float = 4.0          # path-loss exponent, through obstacle cells
    ss_dist_o: float = 35.0             # reference distance (px) for the free-space term
    ss_pl_o: float = 31.0               # path loss (dB) at the reference distance
    ss_xg_min: float = 0.0              # free-space shadowing noise X_g ~ U[min,max], per episode
    ss_xg_max: float = 13.0
    ss_k_min: float = 0.0               # obstacle shadowing noise K ~ U[min,max], per episode
    ss_k_max: float = 13.0
    # Per-move traversal cost, charged PER LATTICE-EDGE LENGTH (not amortized over the episode):
    # an AXIAL step costs `step_penalty_coef`, a DIAGONAL step costs `step_penalty_coef·√2` (it
    # covers √2× the distance / takes √2× the time). Sized to be comparable to the dense terms
    # (novel ~0.04-0.1 per productive step) so movement has a real price → shorter paths preferred,
    # diagonals only taken when they actually cover more ground, and every loop step bleeds cost.
    step_penalty_coef: float = 0.015   # axial-step cost in reward units (diagonal = ·√2)
    completion_bonus: float = 10.0     # reward given at the terminal step when explored >= threshold
    n_hops: int = 6                     # ego-centric encoder window radius (window_side = 2·n_hops + 3); GAT n_layers tied to this
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
    team_reward_weight: float = 0.0         # β: REMOVED from reward (double-counted novel cells already
    #                                         paid via novel_scan → reintroduced free-ride). Kept as
    #                                         a dead knob for back-compat; not summed into the reward.
    # ζ_give/ζ_recv/η_lap are now in scan_norm units (set-op normalized by scan_norm, not N_max) →
    # rescaled ~24× down from the old /N_max defaults (1.5/0.5/3.0) to preserve the train/easy
    # magnitude, now map-size independent.
    give_bonus_coef: float = 0.06           # ζ_give: NEW cells I bring to teammate at rendezvous
    recv_bonus_coef: float = 0.02           # ζ_recv: NEW cells I get from teammate at rendezvous
    overlap_penalty_coef: float = 0.12      # η_lap: cells we BOTH scanned independently since last comm
    revisit_penalty_coef: float = 0.10      # γ: penalty per step on a node visited in last W steps.
                                            # Raised 0.05→0.10 (2×): a tight 2-node ping-pong (age=2,
                                            # graduated ≈0.75 → 0.075/step) now costs ≈1.5× a novel step,
                                            # so cycling is clearly worse than any explored-area shuffle.
                                            # Ceiling ~0.15; >0.2 over-corrects (punishes legit backtrack
                                            # out of a dead-end room + the old both-agents ping-pong bug).
    revisit_window: int = 8                 # W: lookback window for revisit detection
    # RADAR (feat[6/7]) travel-cost discount per hop beyond the ego-window horizon. Lower = more
    # myopic (only just-beyond mass matters); higher = far mass carries further. See build_radar.
    radar_gamma: float = 0.92
    # Ablation: zero the guidepost channel (node_feat[5]) so the policy sees no analytic route —
    # tests whether the radar out-of-window channels subsume the guidepost.
    disable_guidepost: bool = False
    # Stall penalty — heavy cost for standing still (no net displacement this step). Catches
    # collision-revert holds AND invalid/curr-node picks. Pressures agents to reroute /
    # separate instead of deadlocking. δ_stall ≫ revisit so standing still is "heavily penalized".
    stall_penalty_coef: float = 0.1         # δ_stall
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
            guidepost_iters=(cfg.guidepost_iters or None),
            guidepost_path_max=(cfg.guidepost_path_max or None),
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
        self._dbg_reward: dict | None = None   # per-agent reward components (inspector)
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
        # -1 = none yet (episode start). Reset on episode done. Tracks the committed analytic
        # target node per agent (drives the progress-reward telescoping + commitment bookkeeping).
        self._prev_target_node = torch.full((self.N, self.M), -1, dtype=torch.long, device=self.dev)
        # Consecutive steps the SAME analytic target node has been committed. Reset to 0 when the
        # committed node changes; reset to 0 on episode done.
        self._steps_on_option = torch.zeros((self.N, self.M), dtype=torch.long, device=self.dev)
        # Phase 1b — True for an agent on the step a map fusion delivered it NEW cells. The
        # strategic target is already correct after fusion (re-scored on the fused map); the
        # observed failure is TACTICAL — stale GRU navigation momentum carries the agent into
        # the teammate's just-received (now-explored) region instead of following its target.
        # This flag marks "world changed, re-plan from here"; wiring (GRU-hidden refresh vs
        # path-bias boost vs obs feature) is decided AFTER instrumentation. Gated on cells
        # ACTUALLY received (not mere comm range) so already-synced neighbors don't trigger.
        self._comm_event = torch.zeros((self.N, self.M), dtype=torch.bool, device=self.dev)
        # Signal-strength comm: per-episode shadowing noise (X_g free, K obstacle), [N] on GPU.
        # Resampled at every env reset (domain randomization à la IR2). Allocated here, filled below.
        self._ss_xg = torch.zeros(self.N, dtype=torch.float32, device=self.dev)
        self._ss_k  = torch.zeros(self.N, dtype=torch.float32, device=self.dev)
        self._resample_ss_noise(torch.arange(self.N, device=self.dev))
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
        # CTDE critic_global extras: prev-step union-explored frac (coverage_rate derivative) and
        # this-step simple-idle mask (agent scanned no team-new cells). Set each step before the
        # critic_global build; init here so reset()'s first _refresh_obs reads valid tensors.
        self._prev_expl_frac = torch.zeros((self.N,), dtype=torch.float32, device=self.dev)
        self._idle_now = torch.zeros((self.N, self.M), dtype=torch.bool, device=self.dev)
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
        self, action: torch.Tensor,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
        """action: long [N, M] in [0, K). Returns (obs, reward[N,M], done[N], info)."""
        assert action.shape == (self.N, self.M)
        # 1. Decode K-slot pick → global target node + world coords.
        chosen, tgt_xy = self._decode_action(action)

        # 2. Move agents toward the target (interp + collision resolution) and re-scan.
        #    Stall detection snapshots the pre-move position for the displacement check below.
        pos_entry = self.pos.clone()                                   # [N, M, 2]
        self._move_and_scan(tgt_xy)

        env_idx   = torch.arange(self.N, device=self.dev).view(self.N, 1).expand(-1, self.M)
        agent_idx = torch.arange(self.M, device=self.dev).view(1, self.M).expand(self.N, -1)
        # Phase D — snapshot prior visited_step for the chosen node BEFORE update, for revisit detection.
        self._prev_visit_for_revisit = self.visited_step[env_idx, agent_idx, chosen].clone()        # [N, M]
        self.visited_step[env_idx, agent_idx, chosen] = self.t.view(self.N, 1).expand(-1, self.M)
        self.t = self.t + 1
        # Fix B: remember last action K-slot for next obs.
        self.last_action = action.clone()

        # ====================================================================== #
        # 4. Compute reward ingredients (each term below is one summand of the   #
        #    final reward; comm/fusion happens first so set-ops see fused maps).  #
        # ====================================================================== #
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
        # Simple idle (critic_global feature): agent scanned no team-new cells this step. Coarser
        # than the 3-clause refined idle (counts productive transit as idle) but enough as a
        # descriptive critic signal — no penalty semantics, no extra BF flood.
        self._idle_now = novel_count <= 0.0                                                # [N, M] bool
        # Advance the union mask (needed next step for novel-scan attribution). The β·team_delta
        # reward term was REMOVED: a union-new cell is already paid once via novel_scan to its
        # discoverer; adding the shared union-delta to everyone double-counted it and reintroduced
        # the free-ride that novel_scan exists to kill.
        union_now = union_prev | free_node_post.any(dim=1)                                 # [N, N_max]
        self.union_node_mask = union_now
        self.own_node_mask_prev = free_node_post

        # Per-pair contribution / reception / overlap, baselined at last meeting.
        give_bonus, recv_bonus, overlap_pen = self._setop_rewards(
            comm_mask, free_node_pre, free_node_post,
        )

        # revisit_pen: chosen node revisited within last W steps by same agent.
        # Graduated by recency: penalty = (W − age)/W ∈ (0, 1] so tighter loops hurt more.
        W_rev = max(1, int(self.cfg.revisit_window))
        prev_visit_for_chosen = self._prev_visit_for_revisit                                # [N, M]
        t_now_per_m = (self.t - 1).view(self.N, 1).expand(self.N, self.M)                   # [N, M]
        age = (t_now_per_m - prev_visit_for_chosen).clamp(min=0)                            # [N, M]
        is_recent_revisit = (prev_visit_for_chosen >= 0) & (age < W_rev)
        revisit_pen = is_recent_revisit.float() * ((W_rev - age).clamp(min=0).float() / W_rev)

        # G.4.b — per-step proximity penalty (anti-chase) when a visible teammate is too close.
        proximity_pen = self._proximity_penalty(comm_mask)

        # Stall penalty — no net displacement this step (collision-revert hold or
        # invalid/curr-node pick). step_disp also feeds the coverage-efficiency metric.
        step_disp = (self.pos - pos_entry).norm(dim=-1)                  # [N, M]

        # progress_reward REMOVED (2026-06-30): it shaped the agent toward the analytic
        # committed target (d_prev−d_new over the target-rooted BF field) → soft-forced the
        # policy to FOLLOW the selector instead of learning the criterion from observations.
        # The dead-zone gradient it provided (explored-area moves earn 0 novel_scan) is now
        # the policy's job via the in-window utility field (node_feat[2]) + GRU memory; the
        # beyond-window blind spot is addressed in OBSERVATION (coarse global frontier channel),
        # not by bribing toward a target. revisit_pen remains the local anti-loop signal.
        stall_pen = (step_disp < float(self.cfg.nr) * 0.5).float()       # [N, M]

        # ---- 5. Assemble per-agent reward (weighted sum of the terms above) ----
        terminated_now = explored_rate >= self.cfg.done_explored_thresh
        # Explicit per-step movement cost proportional to the lattice edge length of the chosen
        # move: an AXIAL step costs `step_cost`, a DIAGONAL step costs `step_cost·√2` (edge_len is
        # NR for axial, NR·√2 for diagonal). Charged by the chosen action's edge length; an
        # invalid / no-move pick costs 0 (standing still is handled by the stall penalty).
        NR = float(self.cfg.nr)
        step_cost = self.cfg.step_penalty_coef                                       # per-axial-step cost
        move_len = self.graph.edge_len[action]                                       # [N, M] px
        chosen_valid_sp = torch.gather(
            self._last_obs["curr_nbr_valid"], -1, action.unsqueeze(-1)
        ).squeeze(-1)                                                                 # [N, M] bool
        step_penalty = step_cost * (move_len / NR) * chosen_valid_sp.float()         # [N, M]
        a_novel = self.cfg.novel_scan_weight
        z_give  = self.cfg.give_bonus_coef
        z_recv  = self.cfg.recv_bonus_coef
        eta_lap = self.cfg.overlap_penalty_coef
        gamma   = self.cfg.revisit_penalty_coef
        eps_prox = self.cfg.proximity_penalty_coef
        delta_stall = self.cfg.stall_penalty_coef
        reward = (a_novel * novel_scan
                  + z_give * give_bonus
                  + z_recv * recv_bonus
                  - eta_lap * overlap_pen
                  - gamma   * revisit_pen
                  - eps_prox * proximity_pen
                  - delta_stall * stall_pen
                  + terminated_now.float().unsqueeze(-1) * self.cfg.completion_bonus
                  - step_penalty)

        # ---- Telemetry: per-step means of each reward COMPONENT (signed contribution) ----
        # For W&B + tuning. Means over [N, M]. Cheap; detached scalars.
        reward_terms = {
            "novel":         (a_novel * novel_scan).mean(),
            "scan_self_diag": scan_self_delta.mean(),     # diagnostic only, not in reward
            "give":          (z_give * give_bonus).mean(),
            "recv":          (z_recv * recv_bonus).mean(),
            "overlap":       (-eta_lap * overlap_pen).mean(),
            "revisit":       (-gamma * revisit_pen).mean(),
            "proximity":     (-eps_prox * proximity_pen).mean(),
            "stall":         (-delta_stall * stall_pen).mean(),
            "step":          (-step_penalty).mean(),
        }
        # Per-agent signed reward components [N, M] for the step-through inspector (eval/trace only).
        if self.store_render_global:
            self._dbg_reward = {
                "total":         reward.detach(),
                "novel":         (a_novel * novel_scan).detach(),
                "give":          (z_give * give_bonus).detach(),
                "recv":          (z_recv * recv_bonus).detach(),
                "overlap":       (-eta_lap * overlap_pen).detach(),
                "revisit":       (-gamma * revisit_pen).detach(),
                "stall":         (-delta_stall * stall_pen).detach(),
            }

        # ---- Exploration-quality metrics (per-step scalars; driver aggregates) ----
        metrics = self._compute_metrics(
            free_node_pre, comm_mask, team_delta, step_disp,
            stall_pen, is_recent_revisit, novel_count,
        )

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

    # ---------------------------------------------------------------------- #
    # step helpers (each called once per step; pure refactor, no perf impact) #
    # ---------------------------------------------------------------------- #
    def _decode_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map each agent's chosen K-slot to a GLOBAL target node + world coords.

        Phase C: the model picked the K-slot from the LOCAL window's curr_nbr, but the
        env needs the GLOBAL flat index to compute world coords and update visited_step.
        Returns (chosen[N, M] global node, tgt_xy[N, M, 2] world coords).
        """
        curr_nbr_global = self._last_obs["curr_nbr_global"]            # [N, M, K]
        curr_nbr_valid  = self._last_obs["curr_nbr_valid"]             # [N, M, K] (local-edge validity)
        chosen       = torch.gather(curr_nbr_global, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
        chosen_valid = torch.gather(curr_nbr_valid,  dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
        # Invalid action → stay put on the GLOBAL current node. self.curr_idx is the LOCAL
        # window-center constant (≈window²/2) and must NOT be used as a global index here —
        # doing so teleported agents to node_xy[that_constant].
        chosen = torch.where(chosen_valid, chosen, self.curr_idx_global).clamp(min=0)
        tgt_xy = self.graph.node_xy[chosen]   # [N, M, 2]
        return chosen, tgt_xy

    def _move_and_scan(self, tgt_xy: torch.Tensor) -> None:
        """Interpolate agents toward tgt_xy over num_sim_steps, resolving wall and
        agent-agent collisions each sub-step, then LiDAR-scan. Mutates self.pos."""
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
            # lower-priority agent yields (holds prev pos) while the winner advances. The winner
            # reverts too only if it is STILL within min_dist of the loser's hold cell (true
            # blockage, e.g. loser sits on the only path).
            #
            # Priority = WHO ARRIVES FIRST: the agent closer to the contested meeting point
            # (shorter remaining travel) wins the cell — an axial mover beats a diagonal mover
            # aiming at the same node. Only when both are within eps of the meeting point (true
            # geometric tie) does the per-episode random _collision_key break it.
            if self.M > 1:
                key = self._collision_key                              # [N, M] lower = wins
                tie_eps = min_agent_dist * 0.1                         # ~1/10 lattice = "same dist"
                for i in range(self.M):
                    for j in range(i + 1, self.M):
                        d = (sub_pos[:, i] - sub_pos[:, j]).norm(dim=-1)   # [N]
                        collide = (d < min_agent_dist)                     # [N]
                        meet = 0.5 * (sub_pos[:, i] + sub_pos[:, j])       # contested point [N, 2]
                        di = (self.pos[:, i] - meet).norm(dim=-1)          # remaining travel of i
                        dj = (self.pos[:, j] - meet).norm(dim=-1)          # remaining travel of j
                        dist_tie = (di - dj).abs() <= tie_eps             # equal-distance → random
                        i_wins = torch.where(dist_tie,                     # closer arrives first
                                             key[:, i] <= key[:, j],       # tie: random priority
                                             di <= dj)                     # else: nearer wins
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

    def _setop_rewards(
        self, comm_mask: torch.Tensor, free_node_pre: torch.Tensor, free_node_post: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-pair rendezvous set-op terms, baselined at last meeting. Returns
        (give_bonus, recv_bonus, overlap_pen), each [N, M]. Updates last_meeting_node_mask
        for pairs that communicated this step (post-fusion union becomes the new baseline)."""
        N_max = self.N_max
        # Normalize by scan_norm (≈ one sensor-disk of nodes), SAME unit as novel_scan/team —
        # so give/recv/overlap are comparable to the per-step terms AND map-size INDEPENDENT.
        # (Was /N_max → ~24× smaller on train/easy AND scaling with map size; the coefs were
        # silently compensating. Coefs were re-scaled accordingly in EnvCfg.)
        denom = float(max(1.0, self.cfg.scan_norm_nodes))
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
        return give_bonus, recv_bonus, overlap_pen

    def _proximity_penalty(self, comm_mask: torch.Tensor) -> torch.Tensor:
        """G.4.b — per-step proximity penalty [N, M] when a VISIBLE teammate is within
        sensor_range (anti-chase). Gated by comm_mask → only counts in-comm teammates.
        Decentralized. Zero unless proximity_penalty_coef > 0."""
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
        return proximity_pen

    def _compute_metrics(
        self, free_node_pre, comm_mask, team_delta, step_disp,
        stall_pen, is_recent_revisit, novel_count,
    ) -> dict:
        """Exploration-quality telemetry (per-step scalars; driver aggregates). Diagnostic
        only — never enters the reward."""
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
        return {
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

    @property
    def obs(self) -> dict:
        return self._last_obs

    # ---------------------------------------------------------------------- #
    # communication                                                           #
    # ---------------------------------------------------------------------- #
    def _resample_ss_noise(self, idx_t: torch.Tensor) -> None:
        """Draw per-episode log-normal shadowing noise (X_g free, K obstacle) for the given
        env rows, uniform in [min, max] (IR2). Fully on GPU. No-op when the SS model is off
        (cheap; keeps reset branch-free)."""
        n = idx_t.numel()
        if n == 0:
            return
        c = self.cfg
        self._ss_xg[idx_t] = c.ss_xg_min + (c.ss_xg_max - c.ss_xg_min) * torch.rand(n, device=self.dev)
        self._ss_k[idx_t]  = c.ss_k_min  + (c.ss_k_max  - c.ss_k_min)  * torch.rand(n, device=self.dev)

    def _comm_check(self) -> torch.Tensor:
        """Returns comm_mask[N, M, M] bool — True at (n,i,j) iff agents i,j can communicate.

        Two models (cfg.comm_model):
          "los"             : Euclidean dist < comm_range_px AND no GT obstacle on the segment.
          "signal_strength" : log-distance path-loss radio model (IR2 / hal-03365129). The a→b
                              segment is split into free vs obstacle length; path loss is
                              PL = PL_o + [γ_obst·10·log10(d_obst) + K]·(d_obst>0)
                                        + [γ·10·log10(d_free/d_o) + X_g]·(d_free≥d_o);
                              connect iff P_R = P_T − PL > thresh. Walls attenuate, not hard-block.
        Diagonal always True (self-comm).
        """
        N, M = self.N, self.M
        eye = torch.eye(M, dtype=torch.bool, device=self.dev).view(1, M, M).expand(N, -1, -1)
        comm_mask = eye.clone()
        if M < 2:
            return comm_mask
        if self.cfg.force_full_comm:
            return torch.ones((N, M, M), dtype=torch.bool, device=self.dev)

        c = self.cfg
        ss = (c.comm_model == "signal_strength")
        comm_range = c.comm_range_px
        S = c.comm_los_samples
        gt = self.world.gt_torch   # [N, H, W]
        n_idx = torch.arange(N, device=self.dev).view(N, 1).expand(N, S)
        t_vals = torch.linspace(0.0, 1.0, S, device=self.dev)  # [S]

        for i in range(M):
            for j in range(i + 1, M):
                pi   = self.pos[:, i, :]    # [N, 2]
                pj   = self.pos[:, j, :]    # [N, 2]
                diff = pj - pi
                dist = diff.norm(dim=-1)    # [N] total segment length (px)

                # Sample S points along the segment; classify GT obstacle vs free.
                pts = pi.unsqueeze(1) + t_vals.view(1, S, 1) * diff.unsqueeze(1)  # [N, S, 2]
                ix = pts[..., 0].clamp(0, self.W - 1).long()  # [N, S]
                iy = pts[..., 1].clamp(0, self.H - 1).long()
                obst = gt[n_idx, iy, ix] == GT_OBST           # [N, S] True where wall

                if ss:
                    # Fraction of the path inside obstacle → split into free/obstacle distance.
                    frac_obst = obst.float().mean(dim=-1)             # [N]
                    d_obst = frac_obst * dist                         # [N]
                    d_free = (dist - d_obst).clamp(min=0.0)           # [N]
                    pl = pi.new_full((N,), c.ss_pl_o)                 # [N] path loss (dB)
                    has_obst = d_obst > 0.0
                    pl = pl + torch.where(
                        has_obst,
                        10.0 * c.ss_gamma_obst * torch.log10(d_obst.clamp(min=1.0)) + self._ss_k,
                        pl.new_zeros(N),
                    )
                    far_free = d_free >= c.ss_dist_o
                    pl = pl + torch.where(
                        far_free,
                        10.0 * c.ss_gamma * torch.log10((d_free / c.ss_dist_o).clamp(min=1.0)) + self._ss_xg,
                        pl.new_zeros(N),
                    )
                    p_r = c.ss_p_t - pl                               # [N] received power (dBm)
                    can = p_r > c.ss_thresh
                else:
                    # Legacy LOS: in-range AND no obstacle on the segment (hard block).
                    can = (dist < comm_range) & ~obst.any(dim=-1)

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
        self._steps_on_option[idx_t]                  = 0
        self._comm_event[idx_t]                       = False
        self._resample_ss_noise(idx_t)
        # Place agents using new map's start.
        row0, col0 = int(starts_new[0, 0]), int(starts_new[0, 1])
        agent_pos = self._spread_starts_graph(row0, col0, env_idx=env_idx)
        self.pos[env_idx] = agent_pos
        for ag in range(self.M):
            self.last_known_pos[env_idx, :, ag] = agent_pos[ag]
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
        self._steps_on_option[idx_t]                  = 0
        self._comm_event[idx_t]                       = False
        self._resample_ss_noise(idx_t)

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

    # ---------------------------------------------------------------------- #
    # obs helpers (per-agent; called inside the Pass-1 loop of _refresh_obs)  #
    # ---------------------------------------------------------------------- #
    def _bf_from_teammates(self, info: dict, a: int) -> None:
        """H.3 — BF FROM each teammate's last-known position, in agent a's own map.

        Per teammate j != a: lkp_node = floor(lkp[a, j] / NR), then BF rooted there over the
        OPTIMISTIC (UNKNOWN-passable) graph — the teammate usually sits in this agent's
        unexplored region, where the FREE graph would return +inf and silence the
        coordination channel. Warm-started on an unchanged teammate node. Writes
        info["bf_dist_team"][N, M, N_max], self-slot left at +inf (unused).
        """
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
                edge_valid=info.get("edge_valid_optim"),   # None-safe; optimistic graph
            )
            self._team_node_prev[:, a, j] = target_j
            self._dist_team_prev[:, a, j] = dist_j
            bf_dist_team_per_j[:, j] = dist_j
        info["bf_dist_team"] = bf_dist_team_per_j                                   # [N, M, N_max]

    def _refresh_obs(self, comm_mask: torch.Tensor | None = None) -> None:
        """Build per-agent obs from current per-agent occupancy + positions.

        Phase C: encoder consumes ego-centric subgraph windows, not the full lattice.
        Pass 1: build global graph + analytic target + guidepost per agent.
        Pass 2: cross-agent feat[4] (teammate-proximity potential) — writes to global node_feat.
        Pass 3: extract local (2·n_hops + 3)² window per agent; this is what the model sees.
        """
        # ---- Pass 1: build global infos + warm-started target-rooted BF ----
        infos: list[dict] = []
        pref_targets: list[torch.Tensor] = []     # per-agent PRE-deconfliction analytic target
        dteam_list: list = []                      # per-agent rendezvous anchor (re-select arg)
        w_list: list = []                          # per-agent offer weight (re-select arg)
        dteam_min_list: list = []                  # per-agent NEAREST-teammate BF dist (separation arg)
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
            # ---- RADAR (feat[6] b_util, feat[7] b_teammate): compress the world BEYOND the ego
            # window onto the geodesic receptive-horizon nodes so the feed-forward policy gets a
            # heading toward far exploration mass / teammates (anti-loop without recurrence). Uses
            # the FREE-graph BF just computed. teammate_src = each OTHER agent's last-known node.
            if self.M > 1:
                others = [j for j in range(self.M) if j != a]
                lkp = self.last_known_pos[:, a, others, :]                          # [N, M-1, 2]
                lx = (lkp[..., 0] / float(self.graph.NR)).long().clamp(0, self.graph.LW - 1)
                ly = (lkp[..., 1] / float(self.graph.NR)).long().clamp(0, self.graph.LH - 1)
                teammate_src = ly * self.graph.LW + lx                              # [N, M-1]
            else:
                teammate_src = None
            b_util, b_team = self.graph.build_radar(
                info, teammate_src=teammate_src, gamma_r=float(self.cfg.radar_gamma),
            )
            info["node_feat"][..., 6] = b_util
            info["node_feat"][..., 7] = b_team
            # H.3 — BF from each teammate's last-known position → info["bf_dist_team"].
            if self.M > 1:
                self._bf_from_teammates(info, a)
            # ---- Analytic GLOBAL TARGET (deterministic) — nearest-richest frontier, with a
            # rendezvous pull toward the teammate I owe the most fresh map. offer≈0 → pure
            # exploration; offer high → drift the chosen frontier toward that teammate so we
            # converge for a map exchange (self-limiting: fusion on contact resets offer→0).
            d_team_star = None
            w_offer = None
            d_team_min = None
            # Rendezvous / separation anchors feed ONLY select_target_analytic. In "nearest" the
            # selector ignores them → skip the whole offer/d_team computation there. bf_dist_team
            # (built above) is kept regardless: it still feeds feat[4] team_pot in Pass 2.
            if self.M > 1 and self.cfg.target_kind != "nearest":
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
                # NEAREST teammate BF reach per node (self slot = +inf in Pass 1) — ownership signal.
                d_team_min = info["bf_dist_team"].min(dim=1).values                        # [N, N_max]
            # Preferred analytic target (PRE-deconfliction). Guidepost + commitment bookkeeping
            # are DEFERRED until after the intra-step target deconfliction below, so they reflect
            # the FINAL target (a teammate may out-rank this one and force a re-pick).
            sep_w_a = (1.0 - w_offer) if (self.cfg.target_sep_from_offer and w_offer is not None) \
                      else float(self.cfg.target_sep_weight)
            if self.cfg.target_kind == "nearest":
                # Greedy-nearest frontier by BF distance — no scoring / rendezvous / separation.
                pref_target = self.graph.select_target_nearest_bf(
                    info["util_raw"], info["node_valid"], bf_dist_from_curr,
                    prev_target=self._prev_target_node[:, a], curr_idx=info["curr_idx"],
                    keep_margin=float(self.cfg.target_keep_margin),
                )
            else:
                pref_target = self.graph.select_target_analytic(
                    info["util_raw"], info["node_valid"], bf_dist_from_curr,
                    d_team=d_team_star, w=w_offer,
                    beta=float(self.cfg.target_beta), lam=float(self.cfg.target_lambda),
                    prev_target=self._prev_target_node[:, a], curr_idx=info["curr_idx"],
                    keep_margin=float(self.cfg.target_keep_margin),
                    d_team_min=d_team_min, sep_w=sep_w_a,
                    sep_k=float(self.cfg.target_sep_k), sep_m=float(self.cfg.target_sep_margin),
                )
            pref_targets.append(pref_target)
            dteam_list.append(d_team_star)
            w_list.append(w_offer)
            dteam_min_list.append(d_team_min)
            infos.append(info)

        # ---- Intra-step target deconfliction (arrival-time priority, lower-index tiebreak) ----
        # Two agents that picked the SAME global target deadlock (both route to one node, collide,
        # mutual-revert, never re-scan → frontier persists → re-pick same target forever). Resolve
        # HERE, intra-step (so no 1-step-stale claim): the agent with the smaller BF reach cost to
        # the contested node keeps it (arrives first); ties broken by lower index. The loser drops
        # that node and re-selects a different frontier — UNLESS it has no other frontier (then both
        # commit, never back down). Analytic mode only (env owns the target).
        final_targets = [t.clone() for t in pref_targets]
        if self.M > 1 and self.cfg.analytic_target:
            exclude = [torch.zeros((self.N, self.N_max), dtype=torch.bool, device=self.dev)
                       for _ in range(self.M)]

            def _arrival(a: int, g: torch.Tensor) -> torch.Tensor:
                # BF reach cost agent a → node g (+inf if unreachable). [N]
                return infos[a]["bf_dist_from_curr"].gather(
                    1, g.clamp(min=0).unsqueeze(1)).squeeze(1)

            for _pass in range(self.M):           # bounded; cascades settle within M passes
                for a in range(self.M):
                    for b in range(self.M):
                        if a == b:
                            continue
                        ta, tb = final_targets[a], final_targets[b]
                        same = (ta == tb) & (ta >= 0)
                        ca, cb = _arrival(a, ta), _arrival(b, tb)
                        # a yields to b: b strictly closer, or equal cost and b lower index.
                        a_yields = same & ((cb < ca) | ((cb == ca) & (b < a)))
                        add = torch.zeros_like(exclude[a])
                        add.scatter_(1, ta.clamp(min=0).unsqueeze(1), True)
                        trial = exclude[a] | (add & a_yields.unsqueeze(1))
                        sep_w_a = (1.0 - w_list[a]) if (self.cfg.target_sep_from_offer and w_list[a] is not None) \
                                  else float(self.cfg.target_sep_weight)
                        if self.cfg.target_kind == "nearest":
                            re_t = self.graph.select_target_nearest_bf(
                                infos[a]["util_raw"], infos[a]["node_valid"],
                                infos[a]["bf_dist_from_curr"],
                                prev_target=self._prev_target_node[:, a], curr_idx=infos[a]["curr_idx"],
                                keep_margin=float(self.cfg.target_keep_margin), exclude=trial,
                            )
                        else:
                            re_t = self.graph.select_target_analytic(
                                infos[a]["util_raw"], infos[a]["node_valid"],
                                infos[a]["bf_dist_from_curr"],
                                d_team=dteam_list[a], w=w_list[a],
                                beta=float(self.cfg.target_beta), lam=float(self.cfg.target_lambda),
                                prev_target=self._prev_target_node[:, a], curr_idx=infos[a]["curr_idx"],
                                keep_margin=float(self.cfg.target_keep_margin), exclude=trial,
                                d_team_min=dteam_min_list[a], sep_w=sep_w_a,
                                sep_k=float(self.cfg.target_sep_k), sep_m=float(self.cfg.target_sep_margin),
                            )
                        # Single-frontier guard: only yield if the re-pick is a REAL other frontier
                        # (not the curr-node fallback) — else both commit, never back down.
                        do_yield = a_yields & (re_t != infos[a]["curr_idx"])
                        exclude[a] = exclude[a] | (add & do_yield.unsqueeze(1))
                        final_targets[a] = torch.where(do_yield, re_t, final_targets[a])

        # ---- Guidepost + commitment bookkeeping on the FINAL (deconflicted) target ----
        for a in range(self.M):
            info = infos[a]
            new_target = final_targets[a]
            target_same = (new_target == self._target_prev[:, a]).unsqueeze(-1)            # [N, 1]
            dist_init = torch.where(
                target_same.expand(-1, self.N_max),
                self._dist_prev[:, a, :],
                torch.full_like(self._dist_prev[:, a, :], float("inf")),
            )
            self.graph.build_guidepost_v2(info, target=new_target, dist_init=dist_init)
            self._target_prev[:, a] = new_target
            self._dist_prev[:, a, :] = info["guidepost_dist"]
            # Analytic mode: env owns the committed target → advance commitment-step tracking.
            if self.cfg.analytic_target:
                same_tgt = (new_target == self._prev_target_node[:, a])
                self._steps_on_option[:, a] = torch.where(
                    same_tgt, self._steps_on_option[:, a] + 1,
                    torch.zeros_like(self._steps_on_option[:, a]),
                )
                self._prev_target_node[:, a] = new_target

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

        # Guidepost ablation: blank the guidepost channel so the model sees no analytic route.
        if self.cfg.disable_guidepost:
            for a in range(self.M):
                infos[a]["node_feat"][..., 5] = 0.0

        # ---- Render-global stash (eval/debug only) — full-graph utility/validity for the GIF,
        # since obs ships only the ego window. Gated so training pays nothing. ----
        if self.store_render_global:
            # comm_mask is None on the reset-time refresh (the None→eye default runs later);
            # fall back to self-only so the stash never dereferences None.
            cm_stash = comm_mask if comm_mask is not None else torch.eye(
                self.M, dtype=torch.bool, device=self.dev).unsqueeze(0).expand(self.N, self.M, self.M)
            self._render_global = {
                "node_xy":    self.graph.node_xy,                                              # [N_max, 2] static
                "edge_idx":   self.graph.edge_idx_static,                                      # [N_max, K] static
                "window_idx_table": self.graph.window_idx_table,                               # [N_max, W²] global idx (-1 pad)
                "utility":    torch.stack([infos[a]["utility"]    for a in range(self.M)], 1),  # [N, M, N_max]
                # Utility decomposition (boundary-pixel ribbon vs revealable-volume) per node.
                "util_boundary": torch.stack([infos[a]["util_boundary"] for a in range(self.M)], 1),  # [N, M, N_max]
                "util_volume":   torch.stack([infos[a]["util_volume"]   for a in range(self.M)], 1),  # [N, M, N_max]
                "node_valid": torch.stack([infos[a]["node_valid"] for a in range(self.M)], 1),  # [N, M, N_max]
                "edge_valid": torch.stack([infos[a]["edge_valid"] for a in range(self.M)], 1),  # [N, M, N_max, K]
                "curr_idx":   torch.stack([infos[a]["curr_idx"]   for a in range(self.M)], 1),  # [N, M] GLOBAL node
                # Full global node features (0 x_rel,1 y_rel,2 utility,3 age,4 team_pot,5 guidepost)
                # + the analytic global target — for the step-through decision inspector.
                "node_feat":  torch.stack([infos[a]["node_feat"]  for a in range(self.M)], 1),  # [N, M, N_max, F]
                "target":     self._prev_target_node.clone(),                                   # [N, M] GLOBAL target node
                "guidepost_dist": torch.stack([infos[a]["guidepost_dist"] for a in range(self.M)], 1),  # [N, M, N_max]
                # Inspector: teammate visibility. pos = ground-truth xy; last_known_pos[i,j] =
                # i's belief of j (fresh when comm, else stale estimate); comm_mask[i,j] = i&j
                # exchanging this step (→ belief == truth). Lets the viewer draw known vs guessed.
                "pos":            self.pos.clone(),                                              # [N, M, 2]
                "last_known_pos": self.last_known_pos.clone(),                                   # [N, M, M, 2]
                "comm_mask":      cm_stash.clone(),                                              # [N, M, M] bool
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

        # ---- CTDE critic-only global state [N, 8] (value head only; actors never see it) ----
        #   [explored_frac, t/T, geo_pair, coverage_rate, redundancy, tgt_dist, idle_frac, imbalance]
        # All ∈[0,1]. First three fix non-stationarity + agent geometry; the five extras (O2) feed
        # the value head the team-coordination signal the actors can't observe — notably redundancy,
        # which lets the critic explain the privileged novel_scan's ~union drops (lower adv variance).
        diam = float((self.graph.LH + self.graph.LW) * self.cfg.nr)
        T_max = float(max(1, self.cfg.max_episode_steps))
        union_free = (self.world.occupancy_torch == _FREE).any(dim=1).view(self.N, -1).sum(-1).float()
        explored_frac = (union_free / self.free_total.clamp(min=1.0)).clamp(0.0, 1.0)               # [N]
        t_frac = (self.t.float() / T_max).clamp(0.0, 1.0)                                            # [N]
        # geo_pair: nearest-teammate geodesic, mean over agents, /diam. REUSES Pass-1 bf_dist_team.
        if self.M > 1:
            geo_per_agent = []
            for a in range(self.M):
                bt = infos[a]["bf_dist_team"]                                                        # [N, M, N_max]
                ca = infos[a]["curr_idx"]                                                            # [N]
                d_at = bt.gather(2, ca.view(self.N, 1, 1).expand(-1, self.M, 1)).squeeze(-1)         # [N, M] teammate j → a's node
                geo_per_agent.append(d_at.min(dim=1).values)                                         # [N] nearest teammate
            geo = torch.stack(geo_per_agent, dim=1)                                                  # [N, M]
            geo = torch.where(torch.isfinite(geo), geo, torch.full_like(geo, diam))
            geo_pair = (geo.mean(dim=1) / max(1.0, diam)).clamp(0.0, 1.0)                            # [N]
        else:
            geo_pair = torch.zeros(self.N, device=self.dev)
        # coverage_rate: union-explored growth THIS step, scaled to "fraction-of-map per episode"
        # units (Δ·T), clamped [0,5]→[0,1]. Distinguishes "still progressing" from "stalled late".
        # On a per-env reset explored drops → Δ<0 → clamp(min=0) reads 0 (no spurious spike).
        cov_rate = ((explored_frac - self._prev_expl_frac) * T_max).clamp(0.0, 5.0) / 5.0           # [N]
        # redundancy: team double-coverage. (Σ_a own_free − union_free)/union_free per teammate ∈[0,1].
        own_free_a = (self.world.occupancy_torch == _FREE).view(self.N, self.M, -1).sum(-1).float()  # [N, M]
        if self.M > 1:
            redundancy = (((own_free_a.sum(1) - union_free) / union_free.clamp(min=1.0)) / (self.M - 1)).clamp(0.0, 1.0)
        else:
            redundancy = torch.zeros(self.N, device=self.dev)
        # tgt_dist: BF distance from each agent's node to its committed analytic target, mean, /diam.
        # Descriptive only (critic input, not reward) → no "follow the selector" forcing.
        d_tgt = self._dist_prev.gather(2, self.curr_idx_global.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # [N, M]
        d_tgt = torch.where(torch.isfinite(d_tgt), d_tgt, torch.full_like(d_tgt, diam))
        tgt_dist = (d_tgt.mean(dim=1) / max(1.0, diam)).clamp(0.0, 1.0)                              # [N]
        # idle_frac: fraction of agents that scanned no team-new cells this step (simple idle).
        idle_frac = self._idle_now.float().mean(dim=1)                                              # [N]
        # imbalance: contribution skew, normalized so 1 = one agent did everything, 0 = even split.
        own_expl_a = (self.world.occupancy_torch != _UNKNOWN).view(self.N, self.M, -1).sum(-1).float()  # [N, M]
        if self.M > 1:
            share = own_expl_a / own_expl_a.sum(1, keepdim=True).clamp(min=1.0)                      # [N, M]
            imbalance = ((share.max(dim=1).values - 1.0 / self.M) / (1.0 - 1.0 / self.M)).clamp(0.0, 1.0)
        else:
            imbalance = torch.zeros(self.N, device=self.dev)
        critic_global = torch.stack(
            [explored_frac, t_frac, geo_pair, cov_rate, redundancy, tgt_dist, idle_frac, imbalance],
            dim=-1,
        )                                                                                            # [N, 8]
        self._prev_expl_frac = explored_frac.detach()

        self._last_obs = {
            "critic_global":        critic_global,      # CTDE value-only [N, 8]
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
            # Previous action one-hot per agent.
            "prev_action":          self._prev_action_onehot(),    # [N, M, K=8] float
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
