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
    4. Graph rebuild per agent + radar boundary summary.
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
    # Bellman-Ford iteration caps for the from-curr BF distance field (feeds the radar channels) and
    # its path buffer (0 = auto from canvas size; auto = N_max already covers any maze geodesic).
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
    # v2 reward — privileged novel-scan credit (IR2-style r_f): pay only cells the agent
    # scanned that are NEW to the TEAM UNION map. Follower scanning a leader's wake earns 0
    # → removes the chase/free-ride incentive at the source. Training-only privileged signal
    # (CTDE); the deployed actor never sees the union. Replaces scan_self in the reward;
    # scan_self stays as a logged diagnostic.
    novel_scan_weight: float = 1.0          # α_novel
    # Dense-term normalization: ~one sensor disk worth of lattice nodes per productive step.
    # The old /N_max (≈1200) crushed dense terms to O(0.001) vs completion bonus 10.
    scan_norm_nodes: float = 50.0
    # ---- Dense RENDEZVOUS term (IR2 r_s spirit). Rewards NET geodesic approach toward the teammate
    # I owe fresh map, gated by that surplus ∆M. NO separation term (privileged novel_scan already
    # zeroes redundant co-scanning → agents spread without a proximity penalty → no "fear the only
    # path" — a single shared frontier keeps BOTH agents heading down it together, nothing here
    # pushes them apart). Telescoping toward the teammate's FIXED last-known pos between comms:
    # oscillation cancels, and at comm ∆M→0 kills the gate so the lkp-jump is never paid → no
    # flip/hover farming.
    # w calibrated against the OTHER dense terms, not picked in isolation: φ is normalized by
    # nr·scan_norm_nodes (see _refresh_obs), so one real hop of approach is Δφ = 1/scan_norm_nodes
    # = 0.02 — the SAME "one sensor-disk" unit novel/revisit/stall are already denominated in.
    # w=2.5 → a full-gate (g=1) approach step pays 2.5·0.02=0.05, same order as
    # revisit_penalty_coef/stall_penalty_coef (0.10) and the low end of novel (~0.04-0.1): a real
    # consideration, not noise, but never bigger than a full novel-scan credit — exploration still
    # wins a frontier-vs-teammate tug of war when both are on the table.
    rdv_dense_weight: float = 2.5           # w: strength of g·(φ_prev−φ_now). 0 disables. M>1 only.
    # ABLATION — blind the ACTOR to teammates: zeroes agent_scalars [∆M-gate, staleness],
    # feat[4] (teammate-proximity potential) and feat[6] (radar teammate). Map fusion at comm,
    # the rdv reward gate and the privileged critic (geo_pair) are untouched. Pure-exploration
    # test: no approach/avoid reasoning toward teammates possible from the actor's inputs.
    teammate_obs: bool = True
    # VALUE-FIELD obs (anti-loop): per-first-step discounted utility mass over the BF tree from
    # curr — V_k = Σ γ^hops·utility over the branch leaving through neighbor k, max-normalized to
    # [0,1]. One comparable scalar per action ("how much is down each exit, distance included"),
    # so near-weak vs far-strong frontier choices are resolved analytically instead of asking the
    # GAT to integrate window + radar. Fed to the actor (obs["value_field"] [N, M, K]).
    vf_gamma: float = 0.97                  # per-hop discount of utility mass (mirrors radar_gamma)
    rdv_offer_frac: float = 0.15            # gate saturates (g→1) when the map gained since last sync
    #                                         reaches this fraction of the OWN map size AT that sync
    #                                         (relative growth, floored by scan_norm_nodes). Also the ∆M obs norm.
    #                                         SUPERSEDED by rdv_frac_max/min/b0 below (kept only as the
    #                                         ∆M obs norm / scan_norm_nodes floor reference).
    # ---- Content-driven required-surplus fraction, itself decaying with the BASELINE size (how much
    # map was already known/shared AT the last sync) — NOT with elapsed time or sync count. First
    # rendezvous (baseline tiny, map barely known) demands rdv_frac_max surplus; once the shared
    # baseline is already a big chunk of the map, the SAME relative fraction would mean an enormous
    # absolute surplus, so the required fraction decays toward rdv_frac_min as baseline grows:
    #   frac(b) = frac_min + (frac_max-frac_min)·exp(-b/b0),   b = baseline / (H·W)  (∈[0,1])
    rdv_frac_max: float = 0.60              # required surplus fraction when baseline ≈ 0 (first sync)
    rdv_frac_min: float = 0.10              # floor as baseline → whole map
    rdv_frac_b0: float = 0.15               # decay scale in baseline-fraction-of-map units
    # ---- Staleness URGENCY — secondary additive nudge on top of the content-driven gate above.
    # Content (surplus vs the decaying frac) is what COMMANDS g; staleness just adds a small,
    # capped push so a very long separation nudges toward meeting even with a modest surplus.
    rdv_urgency_weight: float = 0.25        # max additive boost to g from pure staleness
    rdv_urgency_T: float = 200.0            # steps of staleness to reach the full urgency boost
    revisit_penalty_coef: float = 0.10      # γ: penalty per step on a node visited in last W steps.
                                            # Raised 0.05→0.10 (2×): a tight 2-node ping-pong (age=2,
                                            # graduated ≈0.75 → 0.075/step) now costs ≈1.5× a novel step,
                                            # so cycling is clearly worse than any explored-area shuffle.
                                            # Ceiling ~0.15; >0.2 over-corrects (punishes legit backtrack
                                            # out of a dead-end room + the old both-agents ping-pong bug).
    revisit_window: int = 16                # W: lookback window for revisit detection. 8→16
                                            # (2026-07-15): the freshly-scanned trail stays "hot"
                                            # longer, so early returns onto it read as revisits.
    # Cumulative revisit STREAK (v0.9 anti ping-pong lungo). Counts steps landing on
    # recently-visited nodes (age < W); the existing graduated revisit penalty is multiplied by
    # 1 + β_rev·(streak−1), UNCAPPED — insisting inside the recency window grows linearly without
    # limit. No per-node visit count: once outside the window there is no memory, so a legitimate
    # future pass through old ground costs nothing.
    # DECAY not reset (2026-07-15): landing on a non-recent node DECAYS the streak by
    # revisit_streak_decay instead of zeroing it — a single high-age hop between two recent nodes
    # can no longer launder the whole streak (the A(recent)-B(old)-A(recent) exploit).
    revisit_streak_beta: float = 0.5        # β_rev: per-consecutive-revisit multiplier growth. 0 disables.
    revisit_streak_decay: float = 0.5       # subtracted per NON-recent landing (0.5 → forgiving one
                                            # accumulated revisit takes two genuinely-new steps).
    # RADAR (feat[5/6]) travel-cost discount per hop beyond the ego-window horizon. Lower = more
    # myopic (only just-beyond mass matters); higher = far mass carries further. See build_radar.
    # Stall diagnosis 2026-07-09 (traces m240/m160): at 0.92 a frontier 45 hops beyond the horizon
    # contributes 0.92^45/8 ≈ 0.4%/node — mute → agents stall next to geodesic-far frontiers.
    # 0.97 + norm 3 lifts the same node to ~8%. Defaults stay at the OLD values for ckpt
    # back-compat (from_ckpt_dict fills absent keys from here); new runs pass the CLI flags.
    radar_gamma: float = 0.92
    # RADAR b_util normalization divisor (build_radar util_norm). Lower = less squashing → far
    # frontier mass survives; b_util still clamped to [0,1].
    radar_util_norm: float = 8.0
    # Stall penalty — heavy cost for standing still (no net displacement this step). Catches
    # collision-revert holds AND invalid/curr-node picks. Pressures agents to reroute /
    # separate instead of deadlocking. δ_stall ≫ revisit so standing still is "heavily penalized".
    stall_penalty_coef: float = 0.1         # δ_stall
    # Cumulative stall STREAK (v0.9): consecutive no-displacement steps multiply δ_stall by
    # 1 + β·(streak−1), clamped to stall_streak_cap (unlike the revisit streak, capped: a hard
    # physical block shouldn't nuke the return). Reset on the first real displacement.
    stall_streak_beta: float = 0.5          # β: per-consecutive-stall multiplier growth. 0 disables.
    stall_streak_cap: float = 4.0           # max multiplier on δ_stall
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
        # Static teammate table: _others_idx[a] = the M-1 indices j != a. Drives the batched
        # Pass-1 radar teammate_src + teammate BF (agents folded into the batch dim).
        self._others_idx = torch.tensor(
            [[j for j in range(self.M) if j != a] for a in range(self.M)],
            dtype=torch.long, device=self.dev,
        ).view(self.M, max(0, self.M - 1))
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
        # Rendezvous term: previous step's φ (geodesic curr→owed-teammate /diam) per agent, for the
        # telescoping reward. +inf = cold (first post-reset step → that step's rdv masked to 0).
        self._rdv_phi_prev = torch.full((self.N, self.M), float("inf"), dtype=torch.float32, device=self.dev)
        self._rdv_gate = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        # v0.9 cumulative-penalty streaks (consecutive stalls / consecutive recent-revisits).
        self._stall_streak = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        self._revisit_streak = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        self._geo_curr_team = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
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

        # 1b. SAME-TARGET-NODE arbitration (v0.9 anti-deadlock). When two agents decode the SAME
        # global node, the sub-step physics used to freeze BOTH (winner-blocked revert). Resolve
        # at the ACTION level instead: the agent with the SHORTER chosen edge wins (axial NR beats
        # diagonal NR·√2 — it arrives first); a length tie breaks RANDOMLY PER-STEP (fair over
        # time, unlike the per-episode _collision_key). The loser is forced to hold (target =
        # own current node) → it takes the stall penalty, learning not to contest; the winner
        # proceeds untouched.
        if self.M > 1:
            chosen_len = self.graph.edge_len[action]                       # [N, M] px of chosen edge
            for i in range(self.M):
                for j in range(i + 1, self.M):
                    same = chosen[:, i] == chosen[:, j]                    # [N]
                    if not bool(same.any()):
                        continue
                    li, lj = chosen_len[:, i], chosen_len[:, j]
                    tie = (li - lj).abs() < 1e-3
                    rnd = torch.rand(self.N, device=self.dev) < 0.5
                    i_wins = torch.where(tie, rnd, li <= lj)               # shorter edge arrives first
                    i_loses = (same & ~i_wins)
                    j_loses = (same & i_wins)
                    for a, loses in ((i, i_loses), (j, j_loses)):
                        chosen[:, a] = torch.where(loses, self.curr_idx_global[:, a], chosen[:, a])
                        tgt_xy[:, a] = torch.where(loses.unsqueeze(-1), self.pos[:, a], tgt_xy[:, a])

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

        # revisit_pen: chosen node revisited within last W steps by same agent.
        # Graduated by recency: penalty = (W − age)/W ∈ (0, 1] so tighter loops hurt more.
        W_rev = max(1, int(self.cfg.revisit_window))
        prev_visit_for_chosen = self._prev_visit_for_revisit                                # [N, M]
        t_now_per_m = (self.t - 1).view(self.N, 1).expand(self.N, self.M)                   # [N, M]
        age = (t_now_per_m - prev_visit_for_chosen).clamp(min=0)                            # [N, M]
        is_recent_revisit = (prev_visit_for_chosen >= 0) & (age < W_rev)
        revisit_pen = is_recent_revisit.float() * ((W_rev - age).clamp(min=0).float() / W_rev)
        # v0.9 — cumulative revisit STREAK: steps landing on recent (age<W) nodes multiply the
        # graduated penalty by 1 + β_rev·(streak−1), UNCAPPED. One isolated pass costs as before
        # (mult=1); sustained ping-pong inside the window grows linearly.
        # DECAY not hard-reset (2026-07-15): a non-recent landing subtracts revisit_streak_decay —
        # alternating one old node between recent ones no longer zeroes the debt; only a sustained
        # run over genuinely old/new ground works it off.
        self._revisit_streak = torch.where(
            is_recent_revisit,
            self._revisit_streak + 1.0,
            (self._revisit_streak - float(self.cfg.revisit_streak_decay)).clamp(min=0.0))
        beta_rev = float(self.cfg.revisit_streak_beta)
        revisit_mult = 1.0 + beta_rev * (self._revisit_streak - 1.0).clamp(min=0.0)
        revisit_pen = revisit_pen * revisit_mult

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
        # v0.9 — cumulative stall STREAK: consecutive stalls multiply δ_stall by
        # 1 + β·(streak−1), clamped to stall_streak_cap. Reset on first real displacement.
        self._stall_streak = (self._stall_streak + 1.0) * stall_pen
        stall_mult = (1.0 + float(self.cfg.stall_streak_beta)
                      * (self._stall_streak - 1.0).clamp(min=0.0)).clamp(max=float(self.cfg.stall_streak_cap))
        stall_pen = stall_pen * stall_mult

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
        gamma   = self.cfg.revisit_penalty_coef
        delta_stall = self.cfg.stall_penalty_coef
        reward = (a_novel * novel_scan
                  - gamma   * revisit_pen
                  - delta_stall * stall_pen
                  + terminated_now.float().unsqueeze(-1) * self.cfg.completion_bonus
                  - step_penalty)
        # NOTE: the dense RENDEZVOUS term (rdv_dense) is added AFTER _refresh_obs below, because it
        # needs the fresh geodesic-to-teammate field + surplus gate that _refresh_obs computes.

        # ---- Telemetry: per-step means of each reward COMPONENT (signed contribution) ----
        # For W&B + tuning. Means over [N, M]. Cheap; detached scalars. (rdv added post-refresh.)
        reward_terms = {
            "novel":         (a_novel * novel_scan).mean(),
            "scan_self_diag": scan_self_delta.mean(),     # diagnostic only, not in reward
            "revisit":       (-gamma * revisit_pen).mean(),
            "stall":         (-delta_stall * stall_pen).mean(),
            "step":          (-step_penalty).mean(),
            # v0.9 streak diagnostics (not reward summands; the multipliers are already
            # folded into revisit/stall above).
            "stall_streak":   self._stall_streak.mean(),
            "revisit_streak": self._revisit_streak.mean(),
        }
        # Per-agent signed reward components [N, M] for the step-through inspector (eval/trace only).
        if self.store_render_global:
            self._dbg_reward = {
                "total":         reward.detach(),   # rdv folded in after _refresh_obs
                "novel":         (a_novel * novel_scan).detach(),
                "revisit":       (-gamma * revisit_pen).detach(),
                "stall":         (-delta_stall * stall_pen).detach(),
            }

        # ---- Exploration-quality metrics (per-step scalars; driver aggregates) ----
        metrics = self._compute_metrics(
            free_node_pre, comm_mask, team_delta, step_disp,
            stall_pen, is_recent_revisit, novel_count,
        )

        self._refresh_obs(comm_mask)

        # ---- Dense RENDEZVOUS reward (needs the post-refresh geodesic-to-teammate + surplus gate).
        # rdv = w · g · (φ_prev − φ_now): reward NET geodesic approach toward the owed teammate,
        # gated by surplus g. φ_prev held from the previous step (per env/agent); +inf on the first
        # post-reset step → masked to 0. Telescoping toward the FIXED lkp between comms → oscillation
        # cancels; at comm g→0 kills the lkp-jump credit → farm-safe.
        rdv_w = float(self.cfg.rdv_dense_weight)
        if self.M > 1 and rdv_w > 0.0:
            phi_now = self._geo_curr_team                                    # [N, M] ∈[0,1]
            valid_prev = torch.isfinite(self._rdv_phi_prev)                  # [N, M]
            delta_phi = torch.where(valid_prev, self._rdv_phi_prev - phi_now,
                                    torch.zeros_like(phi_now))
            rdv_dense = rdv_w * self._rdv_gate * delta_phi                   # [N, M]
            reward = reward + rdv_dense
            self._rdv_phi_prev = phi_now
            reward_terms["rdv"] = rdv_dense.mean()
            if self.store_render_global and self._dbg_reward is not None:
                self._dbg_reward["rdv"] = rdv_dense.detach()
                self._dbg_reward["total"] = reward.detach()

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
                        # Winner still inside min_dist of the loser's hold cell: v0.9 — instead of
                        # reverting BOTH (mutual freeze every step in corridors/swaps), the winner
                        # advances PARTIALLY: pushed radially out of the loser's hold to exactly
                        # min_dist. Progress is made every sub-step, the deadlock can't latch.
                        # Degenerate d2≈0 (perfect overlap) → fall back to the winner's prev pos.
                        d2 = (sub_pos[:, i] - sub_pos[:, j]).norm(dim=-1)  # [N]
                        still = collide & (d2 < min_agent_dist)            # [N]
                        w_pos = torch.where(i_wins.unsqueeze(-1), sub_pos[:, i], sub_pos[:, j])
                        l_pos = torch.where(i_wins.unsqueeze(-1), sub_pos[:, j], sub_pos[:, i])
                        w_prev = torch.where(i_wins.unsqueeze(-1), self.pos[:, i], self.pos[:, j])
                        delta = w_pos - l_pos
                        safe_d = d2.clamp(min=1e-6).unsqueeze(-1)
                        pushed = l_pos + delta / safe_d * min_agent_dist   # winner at ring min_dist
                        pushed = torch.where((d2 > 1e-6).unsqueeze(-1), pushed, w_prev)
                        # Never push into a wall: obstacle at the pushed pixel → winner holds prev.
                        px = pushed[:, 0].clamp(0, self.W - 1).long()
                        py = pushed[:, 1].clamp(0, self.H - 1).long()
                        gt_p = self.world.gt_torch.view(self.N, -1).gather(
                            1, (py * self.W + px).view(self.N, 1)).view(self.N)
                        pushed = torch.where((gt_p != GT_OBST).unsqueeze(-1), pushed, w_prev)
                        new_i = torch.where(i_wins.unsqueeze(-1), pushed, sub_pos[:, i])
                        new_j = torch.where(i_wins.unsqueeze(-1), sub_pos[:, j], pushed)
                        sub_pos[:, i] = torch.where(still.unsqueeze(-1), new_i, sub_pos[:, i])
                        sub_pos[:, j] = torch.where(still.unsqueeze(-1), new_j, sub_pos[:, j])
            self.pos = sub_pos
            self.world.set_positions(self.pos)
            self.world.scan()

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

        Used by eval scripts so all stale state (BF cache, comm timers, rendezvous φ cache,
        etc.) is cleared. Previously eval scripts only reset a subset → corrupted BF warm-start.
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
        self._curr_prev[idx_t]                        = -1
        self._dist_curr_prev[idx_t]                   = float("inf")
        self.t_last_comm[idx_t]                       = 0
        self._own_expl_at_comm[idx_t]                 = 0.0   # baseline reset → first comm re-snaps it
        self.last_action[idx_t]                       = -1
        self._collision_key[idx_t]                    = torch.rand((1, self.M), device=self.dev)
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
        self._rdv_phi_prev[idx_t]      = float("inf")
        self._stall_streak[idx_t]      = 0.0
        self._revisit_streak[idx_t]    = 0.0
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
        # Reset BF-from-curr cache.
        self._curr_prev[idx_t]                        = -1
        self._dist_curr_prev[idx_t]                   = float("inf")
        # H.3 — reset BF-from-teammate cache.
        self._team_node_prev[idx_t]                   = -1
        self._dist_team_prev[idx_t]                   = float("inf")
        # Reset comm-gap timer: at reset, last_known_pos is set to actual start positions
        # (see loop below), so all pairs are "freshly in comm" at t=0.
        self.t_last_comm[idx_t]                       = 0
        self._own_expl_at_comm[idx_t]                 = 0.0   # baseline reset → first comm re-snaps it
        self.last_action[idx_t]                       = -1
        # Re-draw collision priority.
        self._collision_key[idx_t]                    = torch.rand((n, self.M), device=self.dev)
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
        self._rdv_phi_prev[idx_t]      = float("inf")
        self._stall_streak[idx_t]      = 0.0
        self._revisit_streak[idx_t]    = 0.0
        self._refresh_obs()

    # ---------------------------------------------------------------------- #
    # obs helpers (batched over N·M; called from Pass 1 of _refresh_obs)      #
    # ---------------------------------------------------------------------- #
    def _bf_from_teammates(self, info: dict) -> None:
        """H.3 — BF FROM each teammate's last-known position, in the observing agent's own map.

        `info` is the Pass-1 BATCHED build: leading dim B = N·M, batch element b = env b//M,
        agent a = b % M. lkp_node = floor(lkp[a, j] / NR), then BF rooted there over the
        OPTIMISTIC (UNKNOWN-passable) graph — the teammate usually sits in the observer's
        unexplored region, where the FREE graph would return +inf and silence the coordination
        channel. One BF call per teammate SLOT r (M-1 total) covers the whole batch; each call
        is warm-started on an unchanged teammate node from the per-(a, j) cache. Writes
        info["bf_dist_team"][B, M, N_max], self-slot left at +inf (unused).
        """
        B = self.N * self.M
        LH, LW = self.graph.LH, self.graph.LW
        NR = float(self.cfg.nr)
        lkp = self.last_known_pos.reshape(B, self.M, 2)                             # [B, M, 2]
        lj_t = (lkp[..., 0] / NR).long().clamp(0, LW - 1)
        li_t = (lkp[..., 1] / NR).long().clamp(0, LH - 1)
        team_node = li_t * LW + lj_t                                                # [B, M]
        # Views into the caches (contiguous → reshape shares storage; in-place writes persist).
        team_node_prev = self._team_node_prev.reshape(B, self.M)
        dist_team_prev = self._dist_team_prev.reshape(B, self.M, self.N_max)
        bf_dist_team = torch.full(
            (B, self.M, self.N_max), float("inf"),
            dtype=torch.float32, device=self.dev,
        )
        b_arange = torch.arange(B, device=self.dev)
        for r in range(self.M - 1):
            # Teammate index per batch element — depends only on a = b % M ([N,M]→B is n-major).
            j_flat = self._others_idx[:, r].repeat(self.N)                          # [B]
            target_j = team_node[b_arange, j_flat]                                  # [B]
            same_j = (target_j == team_node_prev[b_arange, j_flat]).unsqueeze(-1)
            prev_j = dist_team_prev[b_arange, j_flat]                               # [B, N_max]
            team_dist_init = torch.where(
                same_j.expand(-1, self.N_max),
                prev_j,
                torch.full_like(prev_j, float("inf")),
            )
            dist_j, _ = self.graph.bf_from_target(
                info, target=target_j, dist_init=team_dist_init,
                edge_valid=info.get("edge_valid_optim"),   # None-safe; optimistic graph
            )
            team_node_prev[b_arange, j_flat] = target_j
            dist_team_prev[b_arange, j_flat] = dist_j
            bf_dist_team[b_arange, j_flat] = dist_j
        info["bf_dist_team"] = bf_dist_team                                         # [B, M, N_max]

    def _refresh_obs(self, comm_mask: torch.Tensor | None = None) -> None:
        """Build per-agent obs from current per-agent occupancy + positions.

        Phase C: encoder consumes ego-centric subgraph windows, not the full lattice.
        Pass 1: build global graph + BF-from-curr + radar (feat[5]/feat[6]) per agent.
        Pass 2: cross-agent feat[4] (teammate-proximity potential) — writes to global node_feat.
        Pass 3: extract local (2·n_hops + 3)² window per agent; this is what the model sees.
        """
        # ---- Pass 1 (BATCHED over N·M): build + warm-started BF-from-curr (feeds the radar) ----
        # Every GraphLattice op is batch-agnostic on its leading dim, so the M agents are folded
        # into the batch (B = N·M, b = env·M + agent): build/BF/radar/window each run ONCE
        # instead of M times — identical math, 1/M the kernel launches.
        B = self.N * self.M
        occ_b = self.world.occupancy_torch.reshape(B, self.H, self.W)
        frontier_b = compute_frontier(occ_b)
        info = self.graph.build(
            occupancy=occ_b,
            frontier=frontier_b,
            robot_xy=self.pos.reshape(B, 2),
            visited_step=self.visited_step.reshape(B, self.N_max),
            current_step=int(self.t.max().item()),
        )
        # ---- BF FROM curr (target-INDEPENDENT) → path length to every node.
        curr_prev = self._curr_prev.reshape(B)
        dist_curr_prev = self._dist_curr_prev.reshape(B, self.N_max)
        curr_same = (info["curr_idx"] == curr_prev).unsqueeze(-1)
        curr_dist_init = torch.where(
            curr_same.expand(-1, self.N_max),
            dist_curr_prev,
            torch.full_like(dist_curr_prev, float("inf")),
        )
        bf_dist_from_curr, bf_parent_from_curr = self.graph.bf_from_target(
            info, target=info["curr_idx"], dist_init=curr_dist_init,
        )
        self._curr_prev.copy_(info["curr_idx"].view(self.N, self.M))
        self._dist_curr_prev.copy_(bf_dist_from_curr.view(self.N, self.M, self.N_max))
        info["bf_dist_from_curr"]  = bf_dist_from_curr
        info["bf_parent_from_curr"] = bf_parent_from_curr   # [B, N_max] predecessor on path from curr
        # ---- VALUE-FIELD [B, K]: discounted utility mass per first-step branch (see EnvCfg.vf_gamma).
        vf = self.graph.value_field(info, gamma_vf=float(self.cfg.vf_gamma))
        self._vf = vf.view(self.N, self.M, self.K)
        # ---- RADAR (feat[5] b_util, feat[6] b_teammate): compress the world BEYOND the ego
        # window onto the geodesic receptive-horizon nodes so the feed-forward policy gets a
        # heading toward far exploration mass / teammates (anti-loop without recurrence). Uses
        # the FREE-graph BF just computed. teammate_src = each OTHER agent's last-known node.
        # teammate_obs=False (ablation) → src None → b_team stays zero (actor blind to teammates).
        if self.M > 1 and self.cfg.teammate_obs:
            ar = torch.arange(self.M, device=self.dev).view(self.M, 1)
            lkp = self.last_known_pos[:, ar, self._others_idx, :]               # [N, M, M-1, 2]
            lx = (lkp[..., 0] / float(self.graph.NR)).long().clamp(0, self.graph.LW - 1)
            ly = (lkp[..., 1] / float(self.graph.NR)).long().clamp(0, self.graph.LH - 1)
            teammate_src = (ly * self.graph.LW + lx).reshape(B, self.M - 1)     # [B, M-1]
        else:
            teammate_src = None
        b_util, b_team = self.graph.build_radar(
            info, teammate_src=teammate_src, gamma_r=float(self.cfg.radar_gamma),
            util_norm=float(self.cfg.radar_util_norm),
        )
        info["node_feat"][..., 5] = b_util
        info["node_feat"][..., 6] = b_team
        # H.3 — BF from each teammate's last-known position → info["bf_dist_team"] [B, M, N_max].
        if self.M > 1:
            self._bf_from_teammates(info)

        # ---- Pass 2: cross-agent feat[4] = teammate-proximity POTENTIAL on GLOBAL node_feat ----
        # DENSE field exp(-d / scale) of the BF geodesic distance from the teammate's last-known
        # position to each node (info["bf_dist_team"], built in Pass 1 over the optimistic
        # FREE∪UNKNOWN graph so it stays finite when the teammate is in unexplored space).
        # Gradient-rich, wall-aware, and pointing toward the teammate.
        # teammate_obs=False (ablation) → skip the write, feat[4] stays zero (node_feat is
        # zero-alloc'd every build). bf_dist_team itself is still built: critic geo_pair +
        # rdv φ need it and both are reward/critic-side, not actor obs.
        if self.M > 1 and self.cfg.teammate_obs:
            scale_px = max(1.0, 4.0 * float(self.cfg.nr))
            # min over teammates (self slot left at +inf in Pass 1) → nearest teammate.
            d_min = info["bf_dist_team"].min(dim=1).values                    # [B, N_max]
            pot = torch.exp(-d_min / scale_px)                                # +inf → 0
            pot = torch.nan_to_num(pot, nan=0.0, posinf=0.0, neginf=0.0)
            info["node_feat"][..., 4] = pot * info["node_valid"].float()

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
                "utility":    info["utility"].view(self.N, self.M, self.N_max),                # [N, M, N_max]
                # Utility decomposition (boundary-pixel ribbon vs revealable-volume) per node.
                "util_boundary": info["util_boundary"].view(self.N, self.M, self.N_max),       # [N, M, N_max]
                "util_volume":   info["util_volume"].view(self.N, self.M, self.N_max),         # [N, M, N_max]
                "node_valid": info["node_valid"].view(self.N, self.M, self.N_max),             # [N, M, N_max]
                "edge_valid": info["edge_valid"].view(self.N, self.M, self.N_max, -1),         # [N, M, N_max, K]
                "curr_idx":   info["curr_idx"].view(self.N, self.M),                           # [N, M] GLOBAL node
                # Full global node features (0 x_rel,1 y_rel,2 utility,3 age,4 team_pot,
                # 5 radar-util,6 radar-teammate) — for the step-through decision inspector.
                "node_feat":  info["node_feat"].view(self.N, self.M, self.N_max, -1),          # [N, M, N_max, F]
                # Inspector: teammate visibility. pos = ground-truth xy; last_known_pos[i,j] =
                # i's belief of j (fresh when comm, else stale estimate); comm_mask[i,j] = i&j
                # exchanging this step (→ belief == truth). Lets the viewer draw known vs guessed.
                "pos":            self.pos.clone(),                                              # [N, M, 2]
                "last_known_pos": self.last_known_pos.clone(),                                   # [N, M, M, 2]
                "comm_mask":      cm_stash.clone(),                                              # [N, M, M] bool
                # Inspector: per-first-step value-field (what the actor sees as obs["value_field"]).
                "value_field":    self._vf.clone(),                                              # [N, M, K]
            }

        # ---- Pass 3: extract local windows — ONE batched call, unfold [B, ...] → [N, M, ...] ----
        local = self.graph.extract_local_window(info)
        W2 = self.graph.window_size
        node_xy              = local["node_xy_local"].view(self.N, self.M, W2, 2)
        node_valid           = local["node_valid_local"].view(self.N, self.M, W2)
        node_feat            = local["node_feat_local"].view(self.N, self.M, W2, -1)
        edge_idx             = local["edge_idx_local"].view(self.N, self.M, W2, -1)
        edge_valid           = local["edge_valid_local"].view(self.N, self.M, W2, -1)
        curr_idx             = local["curr_idx_local"].view(self.N, self.M)
        curr_nbr             = local["curr_nbr_local"].view(self.N, self.M, -1)
        curr_nbr_valid       = local["curr_nbr_valid_local"].view(self.N, self.M, -1)
        utility              = local["utility_local"].view(self.N, self.M, W2)
        curr_nbr_global      = local["curr_nbr_global"].view(self.N, self.M, -1)
        local_to_global      = local["local_to_global"].view(self.N, self.M, W2)
        curr_idx_global      = info["curr_idx"].view(self.N, self.M)   # [N, M]

        self.curr_idx = curr_idx
        self.curr_idx_global = curr_idx_global   # [N, M] real lattice node — invalid-action fallback
        if comm_mask is None:
            comm_mask = torch.eye(
                self.M, dtype=torch.bool, device=self.dev
            ).view(1, self.M, self.M).expand(self.N, -1, -1)

        # ---- CTDE critic-only global state [N, 7] (value head only; actors never see it) ----
        #   [explored_frac, t/T, geo_pair, coverage_rate, redundancy, idle_frac, imbalance]
        # All ∈[0,1]. The pooled per-agent embeddings the critic also gets are EGO-RELATIVE, so they
        # carry per-agent exploration CONTENT but not the team geometry — the RELATIONAL geometry
        # lives here as geo_pair (nearest-teammate GEODESIC distance /diam, translation-invariant).
        # No absolute team position: V(s) must generalize across maps. The rest feed the value head
        # team-coordination signal the actors can't observe — notably redundancy, which explains the
        # privileged novel_scan's ~union drops (lower adv variance).
        diam = float((self.graph.LH + self.graph.LW) * self.cfg.nr)
        T_max = float(max(1, self.cfg.max_episode_steps))
        union_free = (self.world.occupancy_torch == _FREE).any(dim=1).view(self.N, -1).sum(-1).float()
        explored_frac = (union_free / self.free_total.clamp(min=1.0)).clamp(0.0, 1.0)               # [N]
        t_frac = (self.t.float() / T_max).clamp(0.0, 1.0)                                            # [N]
        # geo_pair: nearest-teammate geodesic, mean over agents, /diam. REUSES Pass-1 bf_dist_team.
        if self.M > 1:
            bt = info["bf_dist_team"].view(self.N, self.M, self.M, self.N_max)                       # [N, a, j, N_max]
            ca = info["curr_idx"].view(self.N, self.M)                                               # [N, a]
            d_at = bt.gather(3, ca.view(self.N, self.M, 1, 1).expand(-1, -1, self.M, 1)).squeeze(-1) # [N, a, j] teammate j → a's node
            geo = d_at.min(dim=2).values                                                             # [N, M] nearest teammate
            geo = torch.where(torch.isfinite(geo), geo, torch.full_like(geo, diam))
            geo_pair = (geo.mean(dim=1) / max(1.0, diam)).clamp(0.0, 1.0)                            # [N]
            # φ (reward-only) per agent = geodesic curr→nearest-teammate, normalized by
            # nr·scan_norm_nodes — the SAME "one sensor-disk" physical unit every other dense reward
            # term (novel/revisit/stall) is denominated in — NOT /diam. /diam is map-SIZE dependent
            # (inversely: a bigger map crushes Δφ per step, a smaller map inflates it), the opposite
            # of invariant. nr·scan_norm_nodes is a fixed physical constant (sensor/step scale), so
            # one real hop of approach is always worth the same Δφ regardless of map size. geo_pair
            # above (critic-only CTDE feature) intentionally keeps /diam — it wants "fraction of
            # THIS map traversed", a different, legitimately map-relative quantity.
            phi_norm_px = max(1.0, float(self.cfg.nr) * float(self.cfg.scan_norm_nodes))
            self._geo_curr_team = (geo / phi_norm_px).clamp(0.0, 1.0)                                # [N, M]
        else:
            geo_pair = torch.zeros(self.N, device=self.dev)
            self._geo_curr_team = torch.zeros((self.N, self.M), device=self.dev)
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
        # idle_frac: fraction of agents that scanned no team-new cells this step (simple idle).
        idle_frac = self._idle_now.float().mean(dim=1)                                              # [N]
        # imbalance: contribution skew, normalized so 1 = one agent did everything, 0 = even split.
        own_expl_a = (self.world.occupancy_torch != _UNKNOWN).view(self.N, self.M, -1).sum(-1).float()  # [N, M]
        if self.M > 1:
            share = own_expl_a / own_expl_a.sum(1, keepdim=True).clamp(min=1.0)                      # [N, M]
            imbalance = ((share.max(dim=1).values - 1.0 / self.M) / (1.0 - 1.0 / self.M)).clamp(0.0, 1.0)
        else:
            imbalance = torch.zeros(self.N, device=self.dev)
        # ---- RENDEZVOUS RAW OBS + gate (per-agent scalars, execution-decentralized). Given to the
        # actor as agent_scalars = [∆M_norm, staleness_norm] so the policy can DECIDE when to
        # rendezvous; the SAME ∆M gate scales the dense reward. ∆M_a = surplus (cells I mapped that
        # the teammate I owe most lacks) since our last sync, normalized by the map I HAD at that sync
        # (relative growth, not a fraction of the whole canvas); staleness = steps since that sync.
        if self.M > 1:
            offer = (own_expl_a.unsqueeze(2) - self._own_expl_at_comm).clamp(min=0.0)                # [N, M, M]
            eye = torch.eye(self.M, dtype=torch.bool, device=self.dev).view(1, self.M, self.M)
            offer = offer.masked_fill(eye, -1.0)                                                     # mask self slot
            j_star = offer.argmax(dim=2)                                                             # [N, M] teammate owed most
            offer_max = offer.gather(2, j_star.unsqueeze(2)).squeeze(2).clamp(min=0.0)               # [N, M]
            # Gate on RELATIVE growth: surplus / (frac · own map size AT THE LAST SYNC with j_star),
            # not a fixed fraction of the whole canvas. g→1 when I have grown my known map by
            # rdv_offer_frac SINCE we last met — i.e. I now hold a meaningful fraction of NEW content
            # that teammate j_star lacks, so meeting is worth it. Floored by scan_norm_nodes (≈ one
            # sensor disk) so an early / near-empty baseline can't blow the ratio up.
            baseline = self._own_expl_at_comm.gather(2, j_star.unsqueeze(2)).squeeze(2)              # [N, M]
            # Required surplus fraction DECAYS with the baseline itself (map already shared at the
            # last sync), not with time/count — see EnvCfg.rdv_frac_max/min/b0 above.
            total_cells = float(self.H * self.W)
            b_frac = (baseline / total_cells).clamp(0.0, 1.0)
            frac = float(self.cfg.rdv_frac_min) + (float(self.cfg.rdv_frac_max) - float(self.cfg.rdv_frac_min)) \
                   * torch.exp(-b_frac / float(self.cfg.rdv_frac_b0))
            scale = (frac * baseline).clamp(min=float(self.cfg.scan_norm_nodes))
            g_content = (offer_max / scale).clamp(0.0, 1.0)                                          # [N, M] content-driven gate
            last_comm = self.t_last_comm.gather(2, j_star.unsqueeze(2)).squeeze(2).float()           # [N, M]
            staleness = ((self.t.view(self.N, 1).float() - last_comm).clamp(min=0.0) / T_max).clamp(0.0, 1.0)
            # Secondary urgency nudge from raw staleness (steps, not the T_max-normalized one above) —
            # small, capped, never overrides the content-driven gate on its own.
            dt = (self.t.view(self.N, 1).float() - last_comm).clamp(min=0.0)                         # [N, M] steps
            urgency = (dt / float(self.cfg.rdv_urgency_T)).clamp(0.0, 1.0)
            g = (g_content + float(self.cfg.rdv_urgency_weight) * urgency).clamp(0.0, 1.0)
            self._rdv_gate = g
            # teammate_obs=False (ablation): gate g still feeds the rdv REWARD above, but the
            # actor's [∆M-gate, staleness] scalars are zeroed — no "when to rendezvous" signal.
            if self.cfg.teammate_obs:
                agent_scalars = torch.stack([g, staleness], dim=-1)                                  # [N, M, 2]
            else:
                agent_scalars = torch.zeros((self.N, self.M, 2), device=self.dev)
        else:
            self._rdv_gate = torch.zeros((self.N, self.M), device=self.dev)
            agent_scalars = torch.zeros((self.N, self.M, 2), device=self.dev)
        critic_global = torch.stack(
            [explored_frac, t_frac, geo_pair, cov_rate, redundancy, idle_frac, imbalance],
            dim=-1,
        )                                                                                            # [N, 7]
        self._prev_expl_frac = explored_frac.detach()

        self._last_obs = {
            "critic_global":        critic_global,      # CTDE value-only [N, 7]
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
            "curr_idx_global":      curr_idx_global,        # [N, M] global flat idx — invalid-action fallback
            "pos":                  self.pos.clone(),
            "comm_mask":            comm_mask,
            "last_known_pos":       self.last_known_pos.clone(),
            # Previous action one-hot per agent.
            "prev_action":          self._prev_action_onehot(),    # [N, M, K=8] float
            # Rendezvous raw obs: [∆M surplus-gate, staleness] per agent (actor input).
            "agent_scalars":        agent_scalars,                 # [N, M, 2] float
            # Value-field: per-first-step discounted utility mass, max-normalized (actor input).
            "value_field":          self._vf,                      # [N, M, K] float ∈[0,1]
        }

    def _prev_action_onehot(self) -> torch.Tensor:
        """One-hot [N, M, K=8] of last_action. Zero everywhere when last_action == -1."""
        K = self.K
        out = torch.zeros((self.N, self.M, K), dtype=torch.float32, device=self.dev)
        valid = self.last_action >= 0
        safe = self.last_action.clamp(min=0)
        out.scatter_(2, safe.unsqueeze(-1), valid.float().unsqueeze(-1))
        return out
