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

import math
import os
import numpy as np
import torch

_PF_DEBUG = bool(os.environ.get("PF_DEBUG"))

from env.frontier import compute_frontier
from env.graph_lattice import GraphLattice, NBR_OFFSETS
from env.maps import Split, sample_batch
from env.teammate_belief import update_teammate_belief
from env.teammate_belief_pathfront import advance_pathfront, freeze_hypotheses
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
    # TRAVEL BUDGET (px, 0.0 = OFF → step cap alone, so every existing checkpoint replays exactly
    # as it trained). When > 0 the episode also truncates once the FARTHEST-travelled robot has
    # covered this much ground. This is the only horizon that means the same thing on both sides
    # of the IR2 comparison: an IR2 "step" is a waypoint teleport of unbounded length, ours is one
    # lattice hop of ≤ nr·√2 px, so equal step caps are NOT equal budgets — at IR2's native caps we
    # get ~100% of their measured travel on hybrid but only ~50% on corridor and ~43% on complex,
    # which is exactly the pattern of where we pass and fail. A distance budget is also the honest
    # physical constraint for a real robot (battery / mission time).
    max_travel_px: float = 0.0
    # Same budget, expressed PER MAP as px-travelled per GT-free-pixel (0.0 = off). Takes
    # precedence over max_travel_px when both are set. This is the form TRAINING needs:
    # train/difficult spans 3.9x in free area between its p50 (128k px) and p90 (495k px) maps, so
    # one fixed px budget is simultaneously generous on half the split and starving on the other
    # half — and the starved half is exactly where the completion bonus has to fire for the own-99%
    # objective to have a gradient. Scale for calibration, measured on the v12 policy in Stage 0:
    # it actually spent 0.0217 px/free-px on hybrid, 0.0311 on complex, 0.0374 on corridor.
    max_travel_frac: float = 0.0
    flood_max_iters: int = 200
    done_explored_thresh: float = 0.99
    # Episode-termination rule (what counts as "the map is done"):
    #   "union" : the TEAM union map reaches the threshold. Legacy MARLauder rule and the default,
    #             so every existing checkpoint replays exactly as it trained.
    #   "own"   : EVERY agent's OWN map reaches the threshold — the IR2 rule (their env.check_done,
    #             which loops over robots and requires each one's private belief >= 99% of the GT
    #             free area). Sharing is then part of the task: the union being complete is not
    #             enough, the information has to have reached each robot. REQUIRED for the IR2
    #             comparison, because IR2's `success` column IS this termination flag.
    done_mode: str = "union"
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
    # MULTI-HOP RELAY. The comm check is pairwise: with A—B—C (B in range of both, A and C out of
    # range of each other) A and C exchanged nothing, because no consumer of comm_mask ever closed
    # it transitively. IR2 does (recursive DFS over the comm graph, env.py:424-446, then one merged
    # belief written to every member of the flock, env.py:239-248) and that is the behaviour this
    # restores: A, B and C hold the same map, the same teammate positions and the same staleness
    # clocks after a single step, symmetrically.
    #
    # Default False on PURPOSE. from_ckpt_dict restores the checkpoint's env cfg and a key absent
    # from an old ckpt falls back to the dataclass default, so every pre-v20 checkpoint, trace and
    # eval keeps reproducing exactly what it trained under. train_args.py defaults the CLI to True,
    # so new runs get the relay and `--no-comm-relay` is the control arm.
    #
    # The sync REWARD deliberately stays on the direct mask: at M=4 the reward budget is the known
    # cause of v18's failure, and moving its scale in the same change as the connectivity fix would
    # leave v19 with no valid control.
    comm_relay: bool = False
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
    # Pay only the APPROACH half: Δφ clamped to ≥0, so moving away from the teammate is never taxed.
    # Two things make this safe, neither of which is "it stays telescoping" — it does not:
    #   1. NOTHING IS LOST. The term was never potential-based shaping to begin with. PBRS needs
    #      γ·Φ(s')−Φ(s) with Φ a pure function of state; this is g·(Φ_prev−Φ_now) with a
    #      state-dependent, time-varying gate g and no γ. The policy-invariance guarantee was
    #      already gone, so clamping forfeits nothing that was actually held.
    #   2. THE OSCILLATION IS NOT FARMABLE while w < step_penalty_coef·scan_norm_nodes = 0.75.
    #      One approach hop pays at most w·(1/scan_norm_nodes) = w·0.02 (0.002 at the shipped
    #      w=0.10); the hop itself costs step_penalty_coef = 0.015, plus the revisit penalty on the
    #      way back. Approach→retreat→approach loses 7.5× what it earns. Raise w past 0.75 and this
    #      inverts into free money — the bound is the flag's whole safety argument.
    # WHY: measured over v15, `reward/rdv` was −0.20/episode — a NET TAX on the divergence that
    # exploration requires. The positive half is the shaping that was wanted; the negative half was
    # a standing penalty on leaving.
    rdv_clamp_pos: bool = False
    # Inspector/trace only: suppress BUDGET truncation (step cap + travel budget) without zeroing the
    # budget itself, so `travel_frac` in agent_scalars stays on-distribution while capture_trace runs
    # an episode past its natural end. Zeroing max_travel_frac (the old way) also zeroed the obs.
    ignore_budget_truncation: bool = False
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
    # ---- FRONTIER-DIVERSITY overlap tensor (CTDE, training only). When True, _refresh_obs also
    # emits obs["div_overlap"] [N, M, M, K, K]: how much DISCOUNTED FRONTIER MASS agent i's exit k
    # and agent j's exit l lead to IN COMMON, on the shared lattice index. The training loss then
    # contracts it with the two policies (see MAPPOCfg.div_weight) to price two agents committing
    # to the same work.
    # WHY not a penalty on the local logits: that port is v17/J.1/J.2 and it failed for a reason —
    # the logits are over the 8 neighbours of the agent's OWN node, so two agents 300 px apart
    # share no action index and the penalty is identically zero exactly when the duplication is
    # being decided. Frontiers are a SHARED index; branches are not.
    # WHY not a reward term: v18 died by moving the reward budget at M=4 (novel/completion
    # inverted). An auxiliary policy loss leaves the return, the advantage and the critic alone.
    # Default False so every pre-existing checkpoint, trace and eval reproduces unchanged.
    div_overlap: bool = False
    # ---- ATTRIBUTION PARITY (eval only). Recompute the per-agent union-new credit under IR2's
    # OWN two accounting rules, alongside ours, so "is our contribution imbalance a real behaviour
    # difference or an artefact of how the credit is counted?" gets a number instead of an argument.
    #   ours   : every agent is compared against the PREVIOUS step's union, simultaneously, so a
    #            cell two agents scan in the same step is credited to BOTH (shares are then
    #            renormalised). Multi-claimant, one attribution event per lattice hop (<=22.63 px).
    #   IR2    : compat/env.py:163-166 loops over robots and folds each one into the merged belief
    #            BEFORE the next is measured, so every pixel has exactly ONE claimant; and it only
    #            senses at hop endpoints, so credit lands in chunks of one graph edge.
    # Two extra accumulators isolate the two rules: `seq` = single-claimant at OUR cadence,
    # `ir2` = single-claimant at IR2's stride (Explorer.attr_stride_px, per map).
    attr_ir2_parity: bool = False
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
    # WHAT MAKES A MEETING URGENT. "time" (legacy) ramps on steps-since-last-sync, so the gate opens
    # merely because the two have been apart a while — which pulls them together in the MIDDLE of an
    # episode, when they should still be splitting. "budget" ramps on the fraction of the episode
    # budget already spent, so the pull appears only as the deadline approaches. That matches the
    # actual objective: under done_mode="own" the terminal rendezvous is what completes both maps,
    # and there is no reason to pay for it early. Uses the same travel_frac the actor now observes.
    # NOTE this changes what `g` means, and `g` is BOTH agent_scalars[0] and the rdv reward gate — a
    # warm start across a mode switch shifts that input's distribution. Deliberate at a phase
    # boundary (the split/cap/budget all change there anyway); do not flip it mid-phase.
    rdv_urgency_mode: str = "time"          # "time" | "budget"
    rdv_urgency_start: float = 0.5          # budget mode: fraction of budget spent before urgency starts
    # ---- IDLE-CONTACT penalty: the cost of STAYING in radio contact with nothing to exchange.
    # WHY. Measured on v16: after the first paid sync the pair closes from 472 to 342 px, sensor
    # overlap goes 0.089 -> 0.247 and comm duty 0.030 -> 0.120. The cause is a self-reinforcing loop:
    # contact fuses the maps, identical maps produce identical utility fields, identical fields pick
    # the same frontier, co-location produces more contact. Nothing in the reward priced the LINGER —
    # novel merely declines to pay the follower, and a zero is not a penalty.
    # WHY sync_paid AND NOT the g gate. `offer` is computed in _refresh_obs, i.e. AFTER fusion has
    # already reset _own_expl_at_comm, so g is ~0 on EVERY contact step by construction (measured:
    # mean 0.026 on rising edges and on sustained contact alike) — gating on g would punish the
    # legitimate meeting exactly as hard as the tether. `sync_paid` is the pre-fusion rising edge that
    # actually delivered map, so it isolates the productive step. Measured: 81.8% of contact steps are
    # sustained rather than paying, and contact bursts average 3.1 steps.
    # The late-episode exemption reuses the budget urgency ramp, so the terminal rendezvous — which
    # done_mode="own" REQUIRES — is never taxed.
    # Training-time shaping only: the radio physics stays exactly IR2's (signal_strength, ss_thresh
    # -70, sensor 80), so the comparison is untouched.
    comm_idle_pen: float = 0.0
    # ---- SYNC-EVENT reward (the OBJECTIVE term rendezvous was missing). rdv_dense above is
    # TELESCOPING shaping: its net payoff over a full separate→approach→meet cycle is only
    # w·g·φ_sep (measured φ_sep≈0.45 → 0.045 at w=0.10, 1.13 even at w=2.5) against a measured
    # 1.7-1.9 detour cost, so no dense weight can make meeting worth it — past w≈2 it just turns
    # into a chase term. What pays for a rendezvous has to be the EXCHANGE itself:
    #
    #   give_ij = |M_i \ M_j| on the PRE-fusion maps  (nodes I deliver to j at this contact)
    #   recv_ij = give_ji
    #   r_i += ζ_g · Σ_j paid_ij · (give_ij + ρ·recv_ij) / scan_norm_nodes
    #
    # paid_ij fires only on the RISING EDGE of comm and only if ≥ sync_min_gap steps have passed
    # since the last PAID sync with j. Both guards are load-bearing:
    #   * rising edge — with ss_thresh=-70 the free-space comm radius is 150-310 px while the two
    #     80 px LiDAR disks stop overlapping at 160 px, so "walk in parallel at the comm boundary"
    #     has CONTINUOUS comm and DISJOINT sensing: it is the reward-maximal degenerate strategy
    #     under any per-step transfer reward. Paying only the rising edge makes a permanent tether
    #     earn exactly zero after t=0.
    #   * sync_min_gap — kills flicker (step in/out of range). The contact still FUSES; only the
    #     payment and the t_last_paid_sync update are suppressed.
    # Frequency-farming is impossible by conservation: give is a SET difference on monotone maps,
    # so syncing at t1 (set A) then t2 (set B) pays |A|+|B|, exactly what syncing only at t2 pays
    # (|A∪B|). More syncs never pay more; never syncing forfeits the lot. Re-gifting is impossible
    # too — post-fusion M_i \ M_j = ∅.
    # ζ_g calibration: an agent scans ≈865 nodes/episode (novel_scan sum 17.3 × scan_norm 50), so
    # ~200 steps apart leaves a ≈340-node = 6.8 scan_norm-unit surplus. At ζ_g=0.25, ρ=0.5 a sync
    # pays 0.25·(6.8+0.5·6.8) = 2.55 against the 1.7-1.9 detour cost (margin ~1.4×), and the
    # per-episode ceiling is 0.25·1.5·17.3 = 6.5 vs novel 17.3 → the exchange is worth 37.5% of
    # everything found since parting, never more than exploration itself.
    # ρ<1 keeps give dominant: a meeting needs BOTH agents to move, so recv has to be positive
    # (else the map-poor agent evades while the rich one chases), but a lazy agent's recv is large
    # precisely because it explored nothing — ρ=0.5 pays for showing up without paying free-riding.
    sync_give_weight: float = 0.0           # ζ_g (0 = OFF, pre-v11 behavior). Per scan_norm_nodes delivered.
    sync_recv_ratio: float = 0.5            # ρ: ζ_recv = ρ·ζ_g
    sync_min_gap: int = 32                  # steps since the last PAID sync before a contact pays again
    # M-SCALING of the sync bonus: ζ_g_eff = ζ_g · (2/M)^sync_weight_m_scale.
    # 0.0 = OFF (exact no-op at every M, so every pre-existing checkpoint reproduces bit-for-bit).
    # The exponent form is (2/M)^a, so it is identically 1 at M=2 for ANY exponent — the whole M=2
    # history is untouched by construction and only M>2 runs can move.
    # WHY: encounters are not M-invariant. MEASURED v16 (M=2) vs v19 (M=4), realized reward shares
    # over the whole run: sync 5.5% -> 9.8% (x1.78) and metric/sync_rate 0.0068 -> 0.0206 (x3.03),
    # while novel is already at parity (27.2% vs 27.3%, thanks to --novel-scan-weight 1.65) and rdv
    # is 0.6% vs 0.7% — negligible at both M, so DO NOT bother scaling rdv, there is nothing there.
    # a=1 gives 0.5 at M=4; the share-matching target is 5.5/9.8 = 0.56, and the realized share
    # falls further than the weight because syncing less is also a behavioural change (feedback),
    # so do not expect share ∝ weight.
    # RISK TO WATCH: under done_mode=own the episode ends only when the WEAKEST robot holds 99%, and
    # at M=4 each robot needs the maps of THREE others, not one. Cut this too far and
    # eval/own_coverage_final + eval/sync_gap go first — they are the abort signal, not eval/score.
    sync_weight_m_scale: float = 0.0
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
    # Cap on the revisit multiplier, mirroring stall_streak_cap. DEFAULT inf = the uncapped legacy
    # behavior, so v10 and earlier stay bit-comparable. Measured on v10/test-complex: streak peaks
    # at 89 → multiplier 45 → 4.05 reward/step, and the per-episode revisit sum runs to −44 against
    # novel +17 on loop-prone maps. That tail is pure return VARIANCE: it can bury a ~2.5 sync lump
    # in the advantage noise. Set 4.0 (same as stall) if reward/revisit p95 still dominates.
    revisit_streak_cap: float = float("inf")
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
    # and reward set ops, leaving t_last_comm (→ the rendezvous staleness scalar) untouched.
    force_full_occupancy_sharing: bool = False
    # ---- TEAMMATE BELIEF (uniform geodesic wavefront — see env/teammate_belief.py). Replaces the
    # static exp(-BF_dist/scale) teammate_pot (feat[4]) with a UNIFORM possible-location zone that
    # grows one hop/step over the optimistic (FREE∪UNKNOWN) lattice from the last-known node — through
    # the known map and into the unknown via frontiers — holes out in the agent's current sensor FOV
    # (Feature A), and collapses to a delta on comm. feat[4] (plateau) and feat[6]/φ (aimed at the
    # zone CENTROID) derive from it; critic geo_pair uses the teammate's true pos. See _refresh_obs.
    use_teammate_belief: bool = True        # False → legacy exp(-d_min/scale) feat[4], lkp-based radar/φ
    belief_expand_per_step: int = 1         # hops the zone grows per step (1 = "one node per step")
    belief_gate_eps: float = 1e-6           # zone empty below this → teammate "lost" (radar feat[6]/φ off)
    # belief_mode: "uniform" = the geodesic ball above; "pathfront" = two-phase hypothesis model
    # (BF particles lkp→frontier clusters, weighted utility/dist, then per-frontier uniform bloom —
    # see env/teammate_belief_pathfront.py). Frozen at comm-break, cap pf_max_frontiers clusters.
    belief_mode: str = "uniform"
    pf_max_frontiers: int = 6
    # A pathfront belief FRONTIER node = free node with ≥ this many UNKNOWN 8-neighbours (and ≤7). ≥2 (the
    # exploration-frontier default) floods thin corridors — every corridor cell borders the unknown behind
    # its walls → the whole corridor is one connected "frontier" → 1 giant cluster whose centroid sits in
    # the middle. Requiring more unknown neighbours keeps only genuine OPENINGS into large unknown regions,
    # so distinct openings become distinct clusters and a point departs toward each.
    # Kept only as a MINIMUM GUARD (≥1 unknown 8-neighbour). It used to be 4, to stop thin-corridor
    # interior cells chaining into one giant cluster — but it measures the wrong thing: the count
    # collapses as soon as the observer gets CLOSE and reveals a few neighbours, while the node's
    # actual ribbon into the unknown is still there. Measured on test/hybrid #1, node #362 between
    # t=73 and t=74: seed 0.9002 → 0.5428 (still 61% of its ribbon, 27× the utility gate) and it was
    # dropped anyway, purely because its unknown-neighbour count fell under 4 — the belief moved to
    # node #809, seed 0.1439, whose only merit was being far enough away to still be surrounded by
    # unknown. The anti-chaining job is done properly by pf_frontier_min_util below, which measures
    # what is left to reveal. Sweep on the same episode (frontier clusters median/max · belief
    # sitting on real openings): 4/0.02 → 5/9 · 0.5302 ; 1/0.02 → 6/18 · 0.4265 ; 1/0.10 → 6/14 ·
    # 0.6800. The giant cluster does not come back, and the belief lands on openings far more often.
    pf_frontier_min_unknown: int = 1
    # ...AND a real frontier ribbon behind it. The unknown-neighbour count alone marks a node as an
    # opening whenever the cells past it are unknown — including when they are unknown because a
    # WALL is in the way and they will never be revealed from anywhere. Those nodes are permanent
    # false openings: walking over them does not clear them (the count never drops), so the belief
    # absorbs on them forever. Measured on test/hybrid #1 at t=60: 29 of the 32 nodes flagged as
    # frontiers had util_boundary = 0.000, i.e. a PRE-DIFFUSION seed of exactly 0.0 — no ribbon at
    # all — and they were holding 0.2442 of the belief. `util_raw` (= f_ind, the seed, nonzero only
    # on true frontier nodes) is the honest test, and it also removes what the user called "la punta
    # dell'angolo di una stanza": a corner tip has a negligible ribbon, so a negligible seed. The
    # DIFFUSED utility is NOT usable here — a corner tip next to a real opening inherits its value.
    # STRICTLY POSITIVE, not a tuned threshold. The bug is nodes with EXACTLY zero ribbon, and
    # `> 0` is the whole fix. A threshold of 0.10 was calibrated on one test/hybrid episode, where
    # genuine openings scored 0.14-0.90 — it does NOT generalise: on train/easy the same gate cut
    # the openings from ~20 per step to ~5 and, because a hypothesis needs a frontier target, took
    # the belief's own liveness with it. Measured with a fixed policy over 4 easy maps, 1600 steps
    # (scripts/pf_obs_diag.py), against the v13 belief as control — frontier nodes · belief alive
    # while out of comm · feat[4] teammate_pot nonzero · mass on real openings:
    #   v13 belief, no gate   21.1 · 99.4% · 24.7% · 0.184
    #   gate 0.10 (shipped)    5.5 · 81.0% ·  5.6% · 0.542   <- v14's 60-iteration regression
    #   gate 0.02              6.1 · 84.0% · 14.9% · 0.682
    #   gate 1e-6              6.3 · 82.8% · 24.6% · 0.686   <- full observation coverage restored
    # feat[4] is the potential the policy navigates the teammate by; starving it from a quarter of
    # the known nodes to a twentieth is what cost v14 the run, not any of the belief logic fixes
    # (the same measurement shows those cost nothing: 19.3 · 99.4% · 24.6% at gate 0).
    pf_frontier_min_util: float = 1e-6
    # pathfront phase-2 absorbing-diffusion knobs (β_F = min(gain·utility(F), β_max) per-step lock rate).
    belief_absorb_gain: float = 1.0
    belief_beta_max: float = 0.9
    belief_diffuse_lambda: float = 0.5   # fraction of a node's live mass that hops out per step
    # radar_team_source: what feat[6] (b_team, teammate RADAR) sources beyond the ego window.
    # "lkp" (default, legacy) = point-source at each teammate's LAST-KNOWN node, decayed by hops
    # beyond the horizon — independent of belief_mode, blind to where the belief filter thinks the
    # teammate actually is once it has moved off the lkp. "belief" = mass-transport of the belief
    # FIELD itself (self._belief_p, Σ=1 per teammate — works with either belief_mode, but only
    # "pathfront" gives it structure beyond a symmetric ball): every beyond-horizon node's belief
    # mass is routed down its BF-parent chain to its horizon gateway node and discounted by the same
    # gamma_r**hops travel-cost decay as feat[5] b_util, so distant belief mass fades exactly like
    # distant utility mass. Requires use_teammate_belief=True; silently falls back to "lkp" otherwise
    # (see _refresh_obs).
    radar_team_source: str = "lkp"
    # A/B DETERMINISM. The map stream is drawn from `Explorer.rng`, which is deliberately seeded
    # from fresh OS entropy so every run sees different maps (diversity is the point during normal
    # training). That makes two runs with the same --seed train on DIFFERENT maps, which is fatal
    # when comparing two reward configurations. None = legacy fresh-entropy behavior; an int makes
    # the map stream reproducible so an A/B differs only by the thing under test.
    map_seed: int | None = None

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
        # Map RNG: fresh entropy by default (independent of cfg.seed, so every run sees different
        # maps — diversity is the point during normal training). cfg.map_seed pins it instead, which
        # is what an A/B needs: otherwise two runs with the same --seed still train on different
        # maps and the comparison measures map luck as much as the change under test.
        self.rng = np.random.default_rng(cfg.map_seed)
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
        # RAW rendezvous state, per env/agent, for the inspector: every factor that goes into
        # rdv = w · g · (φ_prev − φ_now) and every input the gate g is built from. Populated only
        # when store_render_global is on (traces/eval), so training pays nothing for it. Without
        # this the inspector shows the rdv reward as a bare number with no way to tell WHICH factor
        # made it what it is — a zero from g=0, from Δφ=0, or from the two cancelling.
        self._rdv_dbg: dict | None = None
        # φ evaluated at the K candidate moves [N, M, K] (inspector only; None during training).
        self._phi_nbr: torch.Tensor | None = None
        # Per-teammate φ, [N, M, M] at curr and [N, M, M, K] at the candidate moves. Kept whole so
        # the rendezvous term can select the teammate `g` is actually about (j_star) instead of the
        # nearest one — see the geo block in _refresh_obs. The _nbr one is inspector-only.
        self._phi_all: torch.Tensor | None = None
        self._phi_nbr_all: torch.Tensor | None = None
        self._curr_nbr_valid_dbg: torch.Tensor | None = None
        self.P_max = self.graph.guidepost_path_max
        self.K = 8

        self.pos          = torch.zeros((self.N, self.M, 2),           dtype=torch.float32, device=self.dev)
        self.visited_step = torch.full((self.N, self.M, self.N_max), -1, dtype=torch.long,  device=self.dev)
        self.t            = torch.zeros(self.N,                        dtype=torch.long,    device=self.dev)
        # Cumulative px travelled per robot this episode (EnvCfg.max_travel_px budget + the
        # max_dist column of the IR2 comparison). Reset with self.t at every episode boundary.
        self.travel_px    = torch.zeros((self.N, self.M),              dtype=torch.float32, device=self.dev)
        # PER-ENV travel budget override, [N] px (comparison v2, PROTOCOL_V2_DISTANZA.md §7.1).
        # None = fall back to the cfg-level budgets, so every existing run and eval is unchanged.
        # Set by eval_comparison.py to IR2's per-map distance D_k, which is a DIFFERENT number for
        # every map in the batch and therefore cannot be expressed as an EnvCfg scalar.
        self.travel_budget_px: torch.Tensor | None = None
        self.last_union   = torch.zeros(self.N,                        dtype=torch.float32, device=self.dev)
        self.curr_idx     = torch.zeros((self.N, self.M),              dtype=torch.long,    device=self.dev)
        self.curr_idx_global = torch.zeros((self.N, self.M),           dtype=torch.long,    device=self.dev)
        # last known position: agent i's knowledge of agent j's position
        self.last_known_pos = torch.zeros((self.N, self.M, self.M, 2), dtype=torch.float32, device=self.dev)
        # Step of last comm event between agents a and j (per env). Single consumer today: the
        # rendezvous gate in _refresh_obs, where t − t_last_comm becomes the `staleness` actor
        # scalar and the rdv_urgency nudge. (It used to feed the StrategicHead's cand_max_comm_gap
        # candidate feature; that head was deleted 2026-06-29.)
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
        # Per-step occupancy-count cache: (own_free [N,M], union_free [N], own_known [N,M]) filled by
        # step() right after fusion and consumed+cleared by _refresh_obs. None = "recompute", which
        # is what every reset path (and this constructor's first _refresh_obs) takes.
        self._occ_counts: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        # Ray-sample parameters for the LOS/comm-footprint checks. Constant for the env's lifetime;
        # was re-allocated by torch.linspace on every step.
        self._los_t = torch.linspace(0.0, 1.0, int(cfg.comm_los_samples), device=self.dev)
        self._ss_xg = torch.zeros(self.N, dtype=torch.float32, device=self.dev)
        self._ss_k  = torch.zeros(self.N, dtype=torch.float32, device=self.dev)
        self.reseed_channel_noise(int(seed) + 12345)   # own stream, decoupled from action sampling
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
        # ---- TEAMMATE BELIEF state (see EnvCfg.use_teammate_belief / env/teammate_belief.py).
        # belief_reached = UNIFORM possible-location zone per (observer i, teammate j): a geodesic-ball
        # mask that grows one hop/step over the optimistic graph from the last-known node, holes out in
        # the current sensor FOV, and collapses to a delta on comm. Self slot j==i unused. Single tensor
        # [N, M, M, N_max] (the old mobile/reservoir/cleared triple is gone with the diffusion model).
        self.belief_reached = torch.zeros((self.N, self.M, self.M, self.N_max), dtype=torch.float32, device=self.dev)
        # Orthogonal-neighbour mask over the K lattice slots (|dr|+|dc|==1). The known→unknown FRONTIER
        # crossing is restricted to these ("archi generabili" — a diagonal free→unknown cuts a corner and
        # is not a path a robot could generate); known-zone and unknown-interior stay fully 8-connected.
        self._orth_k = torch.tensor([abs(dr) + abs(dc) == 1 for (dr, dc) in NBR_OFFSETS],
                                    dtype=torch.bool, device=self.dev)         # [K]
        # ---- PATHFRONT belief state (EnvCfg.belief_mode == "pathfront"; env/teammate_belief_pathfront.py).
        # Per (observer i, teammate j) frozen at comm-break: up to Kf frontier-cluster hypotheses, each a
        # BF particle lkp→frontier that then blooms. _comm_prev tracks last step's comm to detect the break.
        Kf = int(self.cfg.pf_max_frontiers)
        Lmax = int(self.graph.guidepost_path_max)
        self._pf_Kf, self._pf_Lmax = Kf, Lmax
        self.pf_front_node = torch.full((self.N, self.M, self.M, Kf), -1, dtype=torch.long, device=self.dev)
        self.pf_weight     = torch.zeros((self.N, self.M, self.M, Kf), dtype=torch.float32, device=self.dev)
        self.pf_dist       = torch.zeros((self.N, self.M, self.M, Kf), dtype=torch.long, device=self.dev)
        self.pf_path       = torch.full((self.N, self.M, self.M, Kf, Lmax), -1, dtype=torch.long, device=self.dev)
        # KNOWN-graph absorbing-diffusion fields: live (mobile) + acc (locked on frontiers) + per-hyp seeded.
        self.pf_live       = torch.zeros((self.N, self.M, self.M, self.N_max), dtype=torch.float32, device=self.dev)
        self.pf_acc        = torch.zeros((self.N, self.M, self.M, self.N_max), dtype=torch.float32, device=self.dev)
        self.pf_seeded     = torch.zeros((self.N, self.M, self.M, Kf), dtype=torch.bool, device=self.dev)
        self.pf_born       = torch.zeros((self.N, self.M, self.M), dtype=torch.bool, device=self.dev)
        # step index at which the hypotheses froze (comm-break), so transit s = t - pf_t0 starts at 0 →
        # the FIRST out-of-comm frame shows the dots AT lkp, then they visibly depart hop by hop.
        self.pf_t0         = torch.zeros((self.N, self.M, self.M), dtype=torch.long, device=self.dev)
        self._comm_prev    = torch.zeros((self.N, self.M, self.M), dtype=torch.bool, device=self.dev)
        # Static teammate table: _others_idx[a] = the M-1 indices j != a. Drives the batched
        # Pass-1 radar teammate_src + teammate BF (agents folded into the batch dim).
        self._others_idx = torch.tensor(
            [[j for j in range(self.M) if j != a] for a in range(self.M)],
            dtype=torch.long, device=self.dev,
        ).view(self.M, max(0, self.M - 1))
        # Rendezvous: agent i's explored-cell count at its last sync with agent j. offer_ij =
        # (i's explored now − this) = fresh map i holds that j hasn't received. Updated comm-gated.
        self._own_expl_at_comm = torch.zeros((self.N, self.M, self.M), dtype=torch.float32, device=self.dev)
        # SYNC-EVENT reward state. Deliberately NOT reusing self._comm_prev above: that one is only
        # written inside _pathfront_belief, so it is stale whenever belief_mode != "pathfront" or
        # the belief filter is off — a rising-edge test against it would misfire silently.
        self._comm_prev_sync = torch.zeros((self.N, self.M, self.M), dtype=torch.bool, device=self.dev)
        # Step of the last PAID sync per ordered pair, initialized to 0 = "we synced at spawn".
        # That is literally true — the agents start adjacent, in comm, holding the same map — and it
        # makes the t=0 rising edge unpayable, so no episode opens with a free (if small) handout
        # for the slightly different spawn scans. First payable contact is at t ≥ sync_min_gap.
        self._sync_t_last_paid = torch.zeros(
            (self.N, self.M, self.M), dtype=torch.long, device=self.dev)
        # Phase D — lattice-level per-agent free count after last step (post-fusion).
        self.last_own_free_node = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        # v2 reward — privileged team-union FREE-node mask (last step) + per-agent post-fusion
        # own mask (last step), for novel-scan attribution. Episode accumulator of novel cells
        # per agent feeds the contribution-share metrics.
        self.union_node_mask = torch.zeros((self.N, self.N_max), dtype=torch.bool, device=self.dev)
        self.own_node_mask_prev = torch.zeros((self.N, self.M, self.N_max), dtype=torch.bool, device=self.dev)
        self.novel_cells_ep = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        # Attribution-parity accumulators + their private unions (see EnvCfg.attr_ir2_parity).
        self.novel_cells_seq_ep = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        self.novel_cells_ir2_ep = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        self._seq_union = torch.zeros((self.N, self.N_max), dtype=torch.bool, device=self.dev)
        self._attr_union = torch.zeros((self.N, self.N_max), dtype=torch.bool, device=self.dev)
        self._attr_last_px = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.dev)
        # Per-map attribution stride in px, set from outside like travel_budget_px. None = fall
        # back to one hop, i.e. our own cadence (which makes `ir2` collapse onto `seq`).
        self.attr_stride_px: torch.Tensor | None = None
        # CTDE critic_global extras: prev-step union-explored frac (coverage_rate derivative) and
        # this-step simple-idle mask (agent scanned no team-new cells). Set each step before the
        # critic_global build; init here so reset()'s first _refresh_obs reads valid tensors.
        self._prev_expl_frac = torch.zeros((self.N,), dtype=torch.float32, device=self.dev)
        self._idle_now = torch.zeros((self.N, self.M), dtype=torch.bool, device=self.dev)
        # Per-agent idle regime, 0=productive 1=redundant 2=transit (see _compute_metrics).
        self._idle_bucket = torch.zeros((self.N, self.M), dtype=torch.int8, device=self.dev)
        self._idle_flags = torch.zeros((self.N, self.M), dtype=torch.int8, device=self.dev)
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
        # map fusion AND reward set ops to all-True. t_last_comm is left alone (it only advances
        # on a real or pos-share comm), so the rendezvous staleness scalar stays meaningful.
        if self.cfg.force_full_occupancy_sharing:
            comm_mask = torch.ones_like(comm_mask)
        # comm_mask is the DIRECT radio link; comm_group is the connected component it belongs to
        # (EnvCfg.comm_relay). Everything that is STATE or OBSERVATION uses the group — the whole
        # flock is one radio network, so a relayed map, position and staleness clock are as real as
        # a direct one. The sync REWARD stays on the direct mask: see EnvCfg.comm_relay.
        comm_group = self._comm_closure(comm_mask) if self.cfg.comm_relay else comm_mask
        # SYNC-EVENT reward — MUST be computed here, on the PRE-fusion maps: one line later the
        # fusion makes M_i == M_j and the set difference is identically empty. Also updates
        # _comm_prev_sync / _sync_t_last_paid, so it has to run exactly once per step.
        sync_give, sync_recv, sync_paid = self._sync_rewards(comm_mask, free_node_pre)
        # fuse_maps walks pairs in-place, so a transitively-closed mask converges in ONE pass:
        # every j is paired with M-1, which already holds the component's union after (0, M-1).
        # (That in-place ordering is also why the pre-relay code leaked a partial one-hop relay —
        # the higher-index leaf got the far map in the same step, the lower-index one a step late.)
        self.world.fuse_maps(comm_group)
        # ---- ONE pass over the post-fusion occupancy. Occupancy does not change again until the
        # end-of-step auto-reset, and _update_last_known_pos / _refresh_obs both need the same three
        # counts, so they are computed here once and cached. Before this they were recomputed
        # independently in three places: 4 full [N, M, H, W] comparisons and 5 reductions per step
        # instead of 2 and 3 (at N=32 that is ~130M redundant element-comparisons every step).
        self._occ_counts = self._count_occupancy()
        own_free_px, union_free, own_known_px = self._occ_counts
        self._update_last_known_pos(comm_group)

        # Post-fusion node-level FREE.
        occ_post_flat = self.world.occupancy_torch.view(self.N, self.M, -1)
        free_node_post = occ_post_flat[:, :, self._node_flat_idx] == _FREE                # [N, M, N_max]

        # team_delta (pixel-level, for completion check + cooperative term).
        explored_rate = (union_free / self.free_total.clamp(min=1.0)).clamp(0, 1)
        # own_cov: what EACH robot actually holds, same denominator as explored_rate. explored_rate
        # is the privileged team UNION — a training-time construct no deployed robot ever has. The
        # gap between them is exactly the map that only exists because we took a union, i.e. the map
        # the agents failed to exchange. Nothing in the reward or the score looked at it before v11.
        own_cov = (own_free_px / self.free_total.clamp(min=1.0).unsqueeze(1)).clamp(0.0, 1.0)  # [N, M]
        # The fraction the termination test looks at (cfg.done_mode). "union" is the legacy rule and
        # is literally explored_rate, so the default path is unchanged; "own" is the IR2 rule — the
        # WEAKEST robot's private map, so the episode ends only once every robot holds the map.
        done_frac = explored_rate if self.cfg.done_mode == "union" else own_cov.amin(dim=1)   # [N]
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
        my_new_count = my_new.float().sum(-1)                                              # [N, M]
        novel_count = (my_new & ~union_prev.unsqueeze(1)).float().sum(-1)                  # [N, M]
        novel_scan = novel_count / scan_norm
        self.novel_cells_ep = self.novel_cells_ep + novel_count
        # Simple idle (critic_global feature): agent scanned no team-new cells this step. Coarser
        # than the 3-clause refined idle (counts productive transit as idle) but enough as a
        # descriptive critic signal — no penalty semantics, no extra BF flood.
        self._idle_now = novel_count <= 0.0                                                # [N, M] bool
        # idle_frac collapses TWO regimes with opposite fixes; _compute_metrics splits them using
        # my_new_count (how much MY lidar added, regardless of who else already had it):
        #   my_new_count == 0                    → TRANSIT   — walking through space I already
        #                                          mapped. A routing/assignment problem, nothing
        #                                          to do with teammates.
        #   my_new_count > 0, novel_count == 0   → REDUNDANT — I scanned real ground a teammate
        #                                          already held. An INFORMATION problem: out of
        #                                          comm the agent has no way to know.
        # novel_count is measured against the PRIVILEGED union, which holds more of the map at any
        # t the more agents there are — so idle_frac is not behaviourally comparable across M even
        # though it is a plain mean. coverage_per_dist is.
        self._my_new_count = my_new_count
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
        revisit_mult = (1.0 + beta_rev * (self._revisit_streak - 1.0).clamp(min=0.0)).clamp(
            max=float(self.cfg.revisit_streak_cap))          # cap = inf by default (legacy uncapped)
        revisit_pen = revisit_pen * revisit_mult

        # Stall penalty — no net displacement this step (collision-revert hold or
        # invalid/curr-node pick). step_disp also feeds the coverage-efficiency metric.
        step_disp = (self.pos - pos_entry).norm(dim=-1)                  # [N, M]
        # Cumulative distance travelled per robot (px) — the episode's PHYSICAL budget, and the
        # only unit in which our horizon is comparable to IR2's. Their "step" is a waypoint
        # teleport of arbitrary length; ours is one lattice hop (≤ nr·√2 = 22.63 px), so capping
        # both at the same step count hands them 2-2.3x our travel on corridor/complex. This is
        # also exactly IR2's headline metric (max over robots of distance travelled).
        self.travel_px = self.travel_px + step_disp                      # [N, M]

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
        terminated_now = done_frac >= self.cfg.done_explored_thresh
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
        # SYNC-EVENT payoff, already normalized by scan_norm_nodes inside _sync_rewards.
        z_sync = float(self.cfg.sync_give_weight)
        if self.cfg.sync_weight_m_scale != 0.0 and self.M != 2:
            # (2/M)^a — identically 1 at M=2, so M=2 runs are untouched. See EnvCfg.
            z_sync *= (2.0 / float(self.M)) ** float(self.cfg.sync_weight_m_scale)
        sync_bonus = z_sync * (sync_give + float(self.cfg.sync_recv_ratio) * sync_recv)   # [N, M]
        # IDLE-CONTACT penalty (see EnvCfg.comm_idle_pen): charged for BEING in contact on a step
        # that did not actually deliver map. Free on the paying step, and free near the deadline so
        # the terminal rendezvous is never taxed. travel_px is already updated for this step above,
        # so the budget ramp here matches the one _refresh_obs feeds the actor as travel_frac.
        comm_idle = None
        if self.M > 1 and float(self.cfg.comm_idle_pen) > 0.0:
            eye_ci = torch.eye(self.M, dtype=torch.bool, device=self.dev).view(1, self.M, self.M)
            in_contact = (comm_group & ~eye_ci).any(dim=2).float()                        # [N, M]
            _bud = self.budget_px()
            if _bud is not None:
                tfrac = (self.travel_px / _bud.unsqueeze(1)).clamp(0.0, 1.0)
            else:
                tfrac = torch.zeros_like(in_contact)
            t0_ci = float(self.cfg.rdv_urgency_start)
            late = ((tfrac - t0_ci) / max(1e-6, 1.0 - t0_ci)).clamp(0.0, 1.0)              # [N, M]
            comm_idle = (float(self.cfg.comm_idle_pen) * in_contact
                         * (1.0 - sync_paid.float()) * (1.0 - late))                      # [N, M]
        reward = (a_novel * novel_scan
                  - gamma   * revisit_pen
                  - delta_stall * stall_pen
                  + sync_bonus
                  + terminated_now.float().unsqueeze(-1) * self.cfg.completion_bonus
                  - step_penalty)
        if comm_idle is not None:
            reward = reward - comm_idle
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
            "sync":          sync_bonus.mean(),
            # Identically 0 unless --comm-idle-pen is set. When it is, this is the whole story of
            # whether the term bit: compare it against reward/novel, and watch metric/comm_duty_cycle.
            "comm_idle":     (-comm_idle).mean() if comm_idle is not None
                             else torch.zeros((), device=self.dev),
            # The terminal payout was the ONE summand with no telemetry, so "did the completion
            # bonus ever fire?" could only be answered indirectly via eval/success_rate — which is
            # measured on the eval suite, not on the training distribution. Under done_mode="own"
            # that is the difference between a hard objective and an unreachable one: v12 ran 4M
            # steps with this term identically zero and nothing in the training log said so.
            "completion":    (terminated_now.float().unsqueeze(-1)
                              * self.cfg.completion_bonus).mean(),
            # Diagnostics (not reward summands): raw map delivered/received per step in
            # scan_norm units, and the paid-sync rate — n_syncs/episode is their time-sum.
            "sync_give_diag": sync_give.mean(),
            "sync_recv_diag": sync_recv.mean(),
            "sync_events":    sync_paid.mean(),
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
                "sync":          sync_bonus.detach(),
                # step + completion were the two summands the inspector never showed, so its rows
                # silently failed to add up to the `total` row printed right under them — by exactly
                # (completion − step). Both are per-agent [N, M] like the rest.
                "step":          (-step_penalty).detach(),
                # expand, not unsqueeze: termination is per-ENV [N], and it only BROADCASTS to
                # [N, M] inside the reward sum. The inspector indexes [0, a] per agent, so a [N, 1]
                # entry here reads fine for agent 0 and throws for every other agent.
                "completion":    (terminated_now.float().unsqueeze(-1)
                                  * self.cfg.completion_bonus).expand(-1, self.M).detach(),
                "comm_idle":     ((-comm_idle) if comm_idle is not None
                                  else torch.zeros_like(step_penalty)).detach(),
            }

        # ---- Exploration-quality metrics (per-step scalars; driver aggregates) ----
        metrics = self._compute_metrics(
            free_node_pre, comm_mask, team_delta, step_disp,
            stall_pen, is_recent_revisit, novel_count, own_cov, explored_rate, sync_paid,
            my_new_count, comm_group,
        )

        self._refresh_obs(comm_group)

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
            delta_phi_raw = delta_phi
            if self.cfg.rdv_clamp_pos:
                delta_phi = delta_phi.clamp(min=0.0)                         # see EnvCfg.rdv_clamp_pos
            rdv_dense = rdv_w * self._rdv_gate * delta_phi                   # [N, M]
            reward = reward + rdv_dense
            if self.store_render_global and self._rdv_dbg is not None:
                # captured BEFORE _rdv_phi_prev is overwritten below. `valid` = 0 on the first
                # post-reset step, where φ_prev is +inf and the term is masked to exactly 0 — that
                # is not "the agent did not approach", it is "there is nothing to compare against".
                self._rdv_dbg.update({
                    "w": rdv_w, "phi": phi_now.detach(),
                    "phi_prev": torch.where(valid_prev, self._rdv_phi_prev,
                                            torch.full_like(phi_now, float("nan"))).detach(),
                    # BOTH: dphi is what got paid, dphi_raw is the geometry. Recording only the
                    # post-clamp value would make a clamped run look like the agent never moved away
                    # from the teammate at all, which is the opposite of what the clamp is there to
                    # reveal. Their difference IS the tax the clamp removed.
                    "dphi": delta_phi.detach(), "dphi_raw": delta_phi_raw.detach(),
                    "clamped": bool(self.cfg.rdv_clamp_pos),
                    "valid": valid_prev.detach(),
                    "rdv": rdv_dense.detach(),
                })
            self._rdv_phi_prev = phi_now
            reward_terms["rdv"] = rdv_dense.mean()
            if self.store_render_global and self._dbg_reward is not None:
                self._dbg_reward["rdv"] = rdv_dense.detach()
                self._dbg_reward["total"] = reward.detach()

        # Episode budget: step count AND (when enabled) travel distance. The step cap stays as a
        # safety net — a policy that stalls forever burns no distance and would otherwise never
        # truncate. max over robots, matching IR2's max_dist: the mission ends when the FIRST
        # robot exhausts its budget, not the average one.
        truncated  = self.t >= self.cfg.max_episode_steps
        _budget = self.budget_px()                             # None = step cap only
        if _budget is not None:
            truncated = truncated | (self.travel_px.amax(dim=1) >= _budget)
        if self.cfg.ignore_budget_truncation:
            # Trace/inspector only. Suppress the truncation but NOT the budget itself, so
            # agent_scalars[2] (travel_frac) keeps the value the policy trained against. The old
            # capture_trace trick — zeroing max_travel_frac/max_travel_px — would have zeroed the
            # observation too, quietly running every trace off-distribution.
            truncated = torch.zeros_like(truncated)
        terminated = done_frac >= self.cfg.done_explored_thresh
        done = truncated | terminated
        if self.cfg.attr_ir2_parity:
            self._attribute_ir2_parity(free_node_pre, union_prev, done)
        info = {
            "explored_rate": explored_rate,
            "terminated":    terminated,
            "truncated":     truncated,
            "travel_px":     self.travel_px.clone(),   # [N, M] cumulative px, pre-auto-reset
            # [N, M, 2] agent positions, ALSO pre-auto-reset. Anything measured from `env.pos`
            # after step() returns reads the RESPAWNED positions on the step an episode ends,
            # because _reset_envs runs at the tail of step(). That silently turned the last step
            # of every episode into a 200-800 px teleport in the comparison harness's distance
            # accumulator. Read positions from here, not from env.pos.
            "pos":           self.pos.clone(),
            "step":          self.t.clone(),
            "reward_terms":  reward_terms,
            "metrics":       metrics,
            # Per-agent union-new cells found so far this episode [N, M] — snapshot taken
            # BEFORE auto-reset so episode-end contribution shares are readable at done.
            "novel_cells_ep": self.novel_cells_ep.clone(),
            # Attribution-parity twins (EnvCfg.attr_ir2_parity); zeros when the flag is off.
            "novel_cells_seq_ep": self.novel_cells_seq_ep.clone(),
            "novel_cells_ir2_ep": self.novel_cells_ir2_ep.clone(),
            # Per-agent OWN-map coverage [N, M] and paid-sync flags [N, M], top-level (not inside
            # "metrics", whose values are reduced to scalars) because the eval suites need them
            # PER ENV: eval_best.py runs K maps as K parallel envs.
            "own_cov":       own_cov,
            "sync_paid":     sync_paid,
            # Per-agent idle regime this step [N, M] int8: 0=productive, 1=redundant, 2=transit.
            # info["metrics"] carries the same split already reduced to scalars over N and M, which
            # cannot answer "is ONE agent carrying the idle" — scripts/idle_diag.py needs the raw
            # per-agent codes to cross them with contact / travel_frac / episode phase.
            "idle_bucket":   self._idle_bucket.clone(),
            "idle_flags":    self._idle_flags.clone(),   # bit0 stalled, bit1 revisit, bit2 contact
            # Per-env "is any pair in contact this step" — comm_duty_cycle in info["metrics"] is
            # already reduced to a scalar over N, and the batched eval scores K maps as K envs.
            "comm_any":      (comm_mask.sum(dim=(1, 2)) > self.M) if self.M > 1
                             else torch.zeros(self.N, dtype=torch.bool, device=self.dev),
            # Full pairwise contact mask [N, M, M], pre-auto-reset. comm_any collapses it to a
            # bool; the IR2 comparison needs the graph itself to decide whether the M>2 team is a
            # single connected "flock" at the final step (their connectivity metric is transitive,
            # so any-pair-in-contact is not the same question).
            "comm_mask":     comm_mask,
            # The transitive closure of the above (== comm_mask when cfg.comm_relay is off). Kept
            # as a SEPARATE key so no existing consumer of comm_mask — eval_comparison.py,
            # eval_best.py, eval/trace.py, the inspector — silently changes what it measures.
            "comm_group":    comm_group,
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
        stall_pen, is_recent_revisit, novel_count, own_cov, explored_rate, sync_paid,
        my_new_count, comm_group,
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
            # comm_duty stays on the DIRECT mask — it is the radio duty cycle and the tether
            # detector, and every past run's value means that. comm_group_duty is its transitive
            # twin; comm_connected is IR2's connectivity_rate (env.py:386): the whole team in one
            # flock, which is the semantics scripts/eval_comparison.py already reports.
            comm_duty = comm_mask[:, offdiag].float().mean()
            comm_group_duty = comm_group[:, offdiag].float().mean()
            comm_connected = comm_group[:, 0, :].all(dim=-1).float().mean()
            sens_overlap = (pd[:, triu] < 2.0 * self.cfg.sensor_range_px).float().mean()
        else:
            comm_duty = torch.zeros((), device=self.dev)
            comm_group_duty = torch.zeros((), device=self.dev)
            comm_connected = torch.zeros((), device=self.dev)
            sens_overlap = torch.zeros((), device=self.dev)
        # ---- idle decomposition. self._idle_now is "scanned nothing TEAM-new"; my_new_count is
        # "how much MY lidar added". The two together separate an assignment failure (transit) from
        # an information failure (redundant), which have opposite fixes.
        idle_transit   = my_new_count <= 0.0                                                 # [N, M]
        idle_redundant = (my_new_count > 0.0) & self._idle_now                               # [N, M]
        if self.M > 1:
            eye_m = torch.eye(self.M, dtype=torch.bool, device=self.dev).view(1, self.M, self.M)
            # Group, not direct link: the question these crosses answer is "could this agent have
            # KNOWN the ground was already the team's", and a relayed link informs it just as well.
            in_contact = (comm_group & ~eye_m).any(dim=2)                                    # [N, M]
        else:
            in_contact = torch.zeros_like(idle_transit)
        # Same budget ramp the actor observes as travel_frac (see the comm_idle block in step()).
        _bud_m = self.budget_px()
        if _bud_m is not None:
            tfrac_m = (self.travel_px / _bud_m.unsqueeze(1)).clamp(0.0, 1.0)                  # [N, M]
        else:
            tfrac_m = torch.zeros_like(self.travel_px)
        # Per-agent bucket code for offline diagnostics (the metrics above are means over N and M,
        # so they cannot answer "is one agent carrying the idle"): 0=productive, 1=redundant,
        # 2=transit. Exported in info["idle_bucket"], with the conditioning bits alongside it in
        # info["idle_flags"] (bit0 stalled, bit1 recent-revisit, bit2 in contact) so a diagnostic
        # can cross them per agent instead of re-deriving them from tensors step() does not export.
        self._idle_bucket = (idle_transit.to(torch.int8) * 2
                             + idle_redundant.to(torch.int8))                                # [N, M]
        self._idle_flags = ((stall_pen > 0).to(torch.int8)
                            + is_recent_revisit.to(torch.int8) * 2
                            + in_contact.to(torch.int8) * 4)                                 # [N, M]
        return {
            "redundancy":     redundancy.mean(),
            "stall_rate":     stall_pen.mean(),
            "revisit_rate":   is_recent_revisit.float().mean(),
            "mean_pair_dist": mean_pair_dist.mean(),
            "comm_duty_cycle":     comm_duty,
            "comm_group_duty":     comm_group_duty,
            "comm_connected":      comm_connected,
            "sensing_overlap":     sens_overlap,
            "team_delta_sum": team_delta.sum(),                   # Σ_N Δunion frac this step (efficiency num)
            "step_disp_sum":  step_disp.sum(),                    # Σ_{N,M} displacement px (efficiency denom)
            # Per-step work proxy: fraction of envs where ALL agents found ≥1 union-new cell
            # this step (training trend for the alternation/idle problem; eval/concurrency is
            # the windowed, authoritative version).
            "both_active":    (novel_count > 0).all(dim=1).float().mean() if self.M > 1
                              else (novel_count > 0).float().mean(),
            # OWN-map coverage (what each robot actually holds) vs the privileged union.
            # own_cov_gap is the headline coordination number: the map that exists only because
            # training takes a union, i.e. the map the agents never managed to exchange.
            "own_cov_mean":   own_cov.mean(),
            "own_cov_min":    own_cov.min(dim=1).values.mean(),
            "own_cov_gap":    (explored_rate - own_cov.min(dim=1).values).clamp(min=0.0).mean(),
            # Paid sync-event rate per agent-step; × episode length = syncs/episode.
            "sync_rate":      sync_paid.mean(),
            # Still computed for telemetry although they left critic_global in v11.
            "idle_frac":      self._idle_now.float().mean(),
            # ---- idle_frac split into DISJOINT regimes (see the comment at the novel-scan block).
            # transit + redundant == idle_frac and productive + idle_frac == 1, exactly — the
            # historical metric is unchanged and stays comparable with every past run.
            "idle_transit":   idle_transit.float().mean(),
            "idle_redundant": idle_redundant.float().mean(),
            # Redundant scanning while OUT of comm is the share of the waste the agent could not
            # possibly have avoided: it had no way to learn that ground was already the team's.
            # This is the number the multi-hop relay (EnvCfg.comm_relay) is expected to move.
            "idle_redundant_nocomm": (idle_redundant & ~in_contact).float().mean(),
            "idle_transit_nocomm":   (idle_transit & ~in_contact).float().mean(),
            # Immobile-while-idle: the 3-clause split says WHY nothing was found, stall_rate says
            # whether the agent was even moving. Cross them so "parked" is separable from "walking
            # over old ground" without re-running a diagnostic.
            "idle_stalled":   (self._idle_now & (stall_pen > 0)).float().mean(),
            "idle_revisit":   (self._idle_now & is_recent_revisit).float().mean(),
            # Mean travel-budget fraction spent, so the split can be read against the deadline.
            "travel_frac_mean": tfrac_m.mean(),
        }

    @property
    def obs(self) -> dict:
        return self._last_obs

    # ---------------------------------------------------------------------- #
    # communication                                                           #
    # ---------------------------------------------------------------------- #
    def _sync_rewards(
        self, comm_mask: torch.Tensor, free_node_pre: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SYNC-EVENT reward ingredients. Returns (give, recv, paid), each [N, M].

        give[i] = Σ_j paid_ij · |M_i \\ M_j| / scan_norm_nodes — map I DELIVER to teammates at a
        paid contact this step; recv[i] the mirror; paid[i] ∈ {0,1} = "i was paid by someone".
        Caller multiplies by ζ_g and ρ (see EnvCfg.sync_give_weight).

        MUST be called on PRE-fusion maps: after `fuse_maps` the two agents hold the same map and
        every set difference is empty.

        A contact pays only when BOTH hold:
          * RISING EDGE — comm_mask & ~comm_prev. A pair that never breaks contact is paid at most
            once, on the first edge; combined with the spawn guard below (t_last_paid starts at 0,
            so t=0 is inside the min gap) a permanent tether earns exactly ZERO. Without this, the
            optimal policy is a comm-boundary tether: with ss_thresh=-70 the free-space radio
            radius is 150-310 px while the two 80 px LiDAR disks separate at 160 px, so a pair can
            hold continuous comm with disjoint sensing and bill their entire novel_scan every step.
          * MIN GAP — ≥ sync_min_gap steps since the last PAID sync with that same teammate. Kills
            range flicker. The contact still FUSES normally; only the payment is suppressed.

        Not farmable by frequency: give is a SET difference over monotone maps, so syncing at t1
        (delivering A) and t2 (delivering B) pays |A|+|B| — exactly what syncing only at t2 pays
        (|A∪B|, disjoint). Extra syncs never pay more; skipping them only forfeits. And post-fusion
        M_i \\ M_j = ∅, so the same node can never be delivered twice on the same pair.
        """
        N, M = self.N, self.M
        zeros = torch.zeros((N, M), dtype=torch.float32, device=self.dev)
        if M < 2:
            return zeros, zeros, zeros
        eye = torch.eye(M, dtype=torch.bool, device=self.dev).view(1, M, M)
        rise = comm_mask & ~self._comm_prev_sync & ~eye                        # [N, M, M]
        gap = self.t.view(N, 1, 1) - self._sync_t_last_paid                    # [N, M, M] steps
        paid = rise & (gap >= int(self.cfg.sync_min_gap))
        self._comm_prev_sync.copy_(comm_mask)
        if not bool(paid.any()):
            return zeros, zeros, zeros
        # Only now materialize the [N, M, M, N_max] set difference (~0.5 MB at M=2, N=32): a paid
        # sync happens at most a handful of times per episode, so this is effectively free — unlike
        # the deleted per-step _setop_rewards, which paid it on every single step.
        M_i = free_node_pre.unsqueeze(2)                                       # [N, M, 1, N_max]
        M_j = free_node_pre.unsqueeze(1)                                       # [N, 1, M, N_max]
        give_ij = (M_i & ~M_j).sum(-1).float()                                 # [N, M, M] nodes
        recv_ij = give_ij.transpose(1, 2)                                      # what j delivers to i
        paid_f = paid.float()
        denom = float(max(1.0, self.cfg.scan_norm_nodes))
        self._sync_t_last_paid = torch.where(
            paid, self.t.view(N, 1, 1).expand_as(paid), self._sync_t_last_paid)
        return ((give_ij * paid_f).sum(2) / denom,
                (recv_ij * paid_f).sum(2) / denom,
                paid.any(dim=2).float())

    def reseed_map_rng(self, seed: int) -> None:
        """Reset the MAP/SPAWN RNG stream. Call at the start of an eval suite, for the same reason
        reseed_channel_noise exists — and it is the bigger of the two effects.

        `self.rng` is seeded once at construction and then ADVANCES on every reload_map/_reset_envs,
        so a persistent eval env scores the same maps from DIFFERENT SPAWN POSITIONS at every tick.
        Measured on v19 ckpt_stop, three back-to-back suites over the same 32 test/complex maps:
        success_rate 0.781 / 0.844 / 0.750 and eval/score 0.367 / 0.388 / 0.384 — about +-9 points
        of success from spawn luck alone, on identical weights. eval/score is the ONLY writer of
        ckpt_best, so without this the best-checkpoint pick is substantially a lottery and two
        ticks of the same run are not comparable.
        """
        self.rng = np.random.default_rng(int(seed))

    def reseed_channel_noise(self, seed: int) -> None:
        """Reset the radio-shadowing RNG stream. Call at the start of an eval suite so the comm
        range each episode draws is a function of the map only, not of how many actions the policy
        happened to sample before it — otherwise eval/comm_duty and eval/sensing_overlap (the
        tether detectors, and the noisiest metrics we have) differ between two evaluations of the
        SAME checkpoint. Measured drift before this: ~30% on duty/sync counts."""
        self._ss_gen = torch.Generator(device=self.dev)
        self._ss_gen.manual_seed(int(seed))

    def _resample_ss_noise(self, idx_t: torch.Tensor) -> None:
        """Draw per-episode log-normal shadowing noise (X_g free, K obstacle) for the given
        env rows, uniform in [min, max] (IR2). Fully on GPU. No-op when the SS model is off
        (cheap; keeps reset branch-free).

        Uses a DEDICATED generator: the global torch stream is interleaved with the policy's
        Categorical.sample(), so drawing from it made the comm-noise sequence depend on the policy."""
        n = idx_t.numel()
        if n == 0:
            return
        c = self.cfg
        g = self._ss_gen
        self._ss_xg[idx_t] = c.ss_xg_min + (c.ss_xg_max - c.ss_xg_min) * torch.rand(n, device=self.dev, generator=g)
        self._ss_k[idx_t]  = c.ss_k_min  + (c.ss_k_max  - c.ss_k_min)  * torch.rand(n, device=self.dev, generator=g)

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
        t_vals = self._los_t                                   # [S] cached at construction

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

    @torch.no_grad()
    def _attribute_ir2_parity(self, free_node_pre: torch.Tensor, union_prev: torch.Tensor,
                              done: torch.Tensor) -> None:
        """Re-credit this step's discoveries under IR2's accounting, in two separable arms.

        `free_node_pre` [N, M, N_max] is each agent's OWN free-node mask this step, pre-fusion —
        the same quantity IR2 reads as all_robot_belief[id][id] == 255. `union_prev` is our shared
        previous-step union, used only to seed nothing: both arms carry their own running union so
        the two rules cannot contaminate each other or our headline number.

        ARM `seq`: strictly IR2's claim rule at OUR cadence — loop the agents in index order and
        fold each into the running union BEFORE measuring the next, so a cell scanned by two
        agents in the same step goes to the LOWER index only. Isolates "one claimant vs many".

        ARM `ir2`: the same claim rule, but an agent is only measured once it has travelled
        `attr_stride_px` since its last measurement — IR2 senses at hop endpoints, so its credit
        arrives one graph edge at a time rather than one lattice hop at a time. Isolates
        "coarse vs fine crediting". Every agent of a finishing episode is flushed on its done step,
        because IR2 credits its final step too and dropping the tail would bias whoever happened
        to be mid-stride.

        Both arms are pure measurement: nothing here feeds a reward, an observation or a
        termination test.
        """
        M = self.M
        stride = self.attr_stride_px
        if stride is None:
            stride = torch.full((self.N,), float(self.graph.NR), device=self.dev)
        trig = (self.travel_px - self._attr_last_px) >= stride.unsqueeze(1)          # [N, M]
        trig = trig | done.unsqueeze(1)
        for m in range(M):
            own = free_node_pre[:, m]                                                # [N, N_max]
            # --- seq: every step, single claimant, index order
            self.novel_cells_seq_ep[:, m] += (own & ~self._seq_union).sum(-1).float()
            self._seq_union |= own
            # --- ir2: same, but gated on the agent's own stride
            t = trig[:, m]
            self.novel_cells_ir2_ep[:, m] += ((own & ~self._attr_union).sum(-1).float()
                                              * t.float())
            self._attr_union |= own & t.unsqueeze(-1)
            self._attr_last_px[:, m] = torch.where(t, self.travel_px[:, m],
                                                   self._attr_last_px[:, m])
        _ = union_prev          # kept in the signature: the arms deliberately do NOT read it

    @torch.no_grad()
    def _branch_overlap(self, label: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """[N, M, M, K, K] shared discounted-frontier mass between agent i's exit k and j's exit l.

        `label` [B, N_max] is the first-step branch each lattice node hangs off in the agent's own
        BF tree and `mass` [B, N_max] is gamma_vf^hops * utility there (both from
        graph.value_field(..., return_branch=True)). Scattering mass onto the branch axis gives
        w[n, m, k, v] = the discounted frontier mass exit k of agent m leads to at NODE v; the
        node axis is the LATTICE index, which every agent shares, so

            O[n, i, j, k, l] = sum_v w[n, i, k, v] * w[n, j, l, v]

        is "if i goes k and j goes l, how much of the same ground are they going for".

        Each agent is normalised to unit total mass FIRST, so O is scale-free: an agent standing in
        a utility-rich region does not dominate the term simply by having more mass, and the value
        is comparable across steps, maps and M. An agent whose reachable utility is zero (the
        `desert` case value_field already signals) contributes exactly 0 rather than a NaN.

        Cost at N=32, M=4, K=8, N_max=961: a [32, 32, 961] x [32, 961, 32] batched matmul, ~2 GFLOP
        per env-step, and a transient [N, M, K, N_max] of 3.9 MB. Kept under no_grad: the tensor is
        a fixed property of the state, and the loss differentiates only through the policies.
        """
        N, M, K, V = self.N, self.M, self.K, self.N_max
        lab = label.view(N, M, V)
        m = mass.view(N, M, V).clamp(min=0.0)
        m = m / m.sum(dim=-1, keepdim=True).clamp(min=1e-6)                 # per-agent unit mass
        w = torch.zeros((N, M, K, V), dtype=m.dtype, device=m.device)
        w.scatter_(2, lab.clamp(min=0).unsqueeze(2), m.unsqueeze(2))
        # scatter_ wrote every branch row for unreachable nodes (label -1 -> row 0); zero them.
        w = w * (lab >= 0).unsqueeze(2)
        wf = w.reshape(N, M * K, V)
        return torch.bmm(wf, wf.transpose(1, 2)).view(N, M, K, M, K).permute(0, 1, 3, 2, 4)

    def _comm_closure(self, comm_mask: torch.Tensor) -> torch.Tensor:
        """Transitive closure of the comm graph → [N, M, M] bool, True at (n,i,j) iff i and j are
        in the same connected component ("flock" in IR2, env.py:424-446 — a recursive DFS there,
        boolean matrix squaring here because M is tiny and this runs every step of every env).

        _comm_check already sets the diagonal, so `reach` starts reflexive and each squaring
        doubles the hop radius: ceil(log2(M)) rounds saturate reachability (2 matmuls on [N,4,4]
        at M=4). Below M=3 the closure is the identity, hence the exact early return.
        """
        if self.M < 3:
            return comm_mask
        reach = comm_mask
        for _ in range(int(math.ceil(math.log2(self.M)))):
            reach = (reach.float() @ reach.float()) > 0
        return reach

    def _count_occupancy(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three occupancy reductions everything downstream needs, from ONE pass.

        Returns (own_free [N, M], union_free [N], own_known [N, M]) as float pixel counts:
        per-agent FREE cells, FREE cells held by at least one agent, and per-agent non-UNKNOWN
        cells. Split out so step() can compute them once right after fusion and hand the result
        to _update_last_known_pos and _refresh_obs via self._occ_counts, instead of each
        recomputing its own `== _FREE` / `!= _UNKNOWN` over the whole [N, M, H, W] canvas.
        """
        occ = self.world.occupancy_torch
        free_px = occ == _FREE                                                       # [N, M, H, W]
        own_free = free_px.view(self.N, self.M, -1).sum(-1).float()                  # [N, M]
        union_free = free_px.any(dim=1).view(self.N, -1).sum(-1).float()             # [N]
        own_known = (occ != _UNKNOWN).view(self.N, self.M, -1).sum(-1).float()       # [N, M]
        return own_free, union_free, own_known

    def _update_last_known_pos(self, comm_mask: torch.Tensor) -> None:
        """Update last_known_pos (point estimate), t_last_comm (staleness clock) and
        _own_expl_at_comm (the rendezvous surplus baseline) on every pair that is in contact.

        "σ-inflation timer" in the old wording referred to the v0.6 Gaussian belief predictor,
        which no longer exists — the uncertainty over a teammate's position now lives in the
        pathfront belief field (env/teammate_belief_pathfront.py), and t_last_comm is just a clock.

        Normally gated by comm_mask. If cfg.force_full_pos_sharing is True, ALL pairs
        get fresh pos every step (debug only — used to test if chase/weird-movement
        bugs come from stale lkp). Map fusion remains comm-gated regardless.
        """
        t_now = self.t                                   # [N]
        # Post-fusion explored-cell count per agent — snapshot at sync drives the offer metric.
        # step() has already counted it for this step (self._occ_counts); recompute only on the
        # reset paths, which call this without going through step().
        own_expl = (self._occ_counts[2] if self._occ_counts is not None
                    else self._count_occupancy()[2])                                        # [N, M]
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

    def _snap_to_lattice(self, pts_xy, env_idx: int) -> torch.Tensor:
        """Map M arbitrary (x, y) pixel positions onto the nearest FREE lattice nodes → [M, 2].

        Used to pin MARLauder's agents to IR2's actual start positions (PROTOCOL_V2_DISTANZA.md
        §6.2). The two systems discretise space differently — IR2 puts robot i on the i-th node of
        its own k-NN graph, we move on a fixed lattice — so NO pixel is a valid node in both. This
        gets as close as the discretisation allows; the residual offset is recorded in
        `last_start_offset_px` and REPORTED rather than assumed away.

        Distinct nodes are enforced: two IR2 robots can be closer together than one lattice cell,
        and collapsing them onto the same node would spawn co-located agents — the degenerate case
        `_spawn_degenerate` exists to reject, which stalls both at step 0.
        """
        dev = self.dev
        gt = self.world.gt_torch[env_idx]                                # [H, W]
        node_xy = self.graph.node_xy                                     # [N_max, 2] as (x, y)
        nx = node_xy[:, 0].long().clamp(0, self.W - 1)
        ny = node_xy[:, 1].long().clamp(0, self.H - 1)
        node_free = gt[ny, nx] == GT_FREE                                # [N_max]
        pts = torch.as_tensor(pts_xy, dtype=torch.float32, device=dev).view(-1, 2)
        out = torch.zeros(self.M, 2, dtype=torch.float32, device=dev)
        off = torch.zeros(self.M, dtype=torch.float32, device=dev)
        taken: list[int] = []
        for i in range(self.M):
            d = (node_xy - pts[i].unsqueeze(0)).norm(dim=-1)
            d = torch.where(node_free, d, torch.full_like(d, float("inf")))
            for t in taken:
                d[t] = float("inf")
            j = int(d.argmin().item())
            taken.append(j)
            out[i] = node_xy[j]
            off[i] = d[j]
        self.last_start_offset_px = off
        return out

    def budget_px(self) -> torch.Tensor | None:
        """Per-env travel budget in px, [N], or None when the episode is capped by steps alone.

        ONE definition, consumed by all four places that used to re-derive it independently:
        the truncation test, the actor's `travel_frac` observation, the comm-idle penalty ramp,
        and the idle-bucket metric. They must agree — a policy whose observed budget disagrees
        with the budget that actually truncates it is being evaluated off-distribution, and under
        `rdv_urgency_mode="budget"` the rendezvous pull is driven by exactly that observation.

        Precedence: explicit per-env override > max_travel_frac (per-map, training) > max_travel_px
        (flat). Unchanged from the previous inline logic apart from the override.
        """
        if self.travel_budget_px is not None:
            return self.travel_budget_px
        if self.cfg.max_travel_frac > 0.0:
            return (self.cfg.max_travel_frac * self.free_total).clamp(min=1.0)
        if self.cfg.max_travel_px > 0.0:
            return torch.full((self.N,), max(1.0, float(self.cfg.max_travel_px)),
                              dtype=torch.float32, device=self.dev)
        return None

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

    def reload_map(self, env_idx: int, map_idx: int, start_override=None) -> None:
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
        self.travel_px[idx_t]                         = 0.0
        self._curr_prev[idx_t]                        = -1
        self._dist_curr_prev[idx_t]                   = float("inf")
        self.t_last_comm[idx_t]                       = 0
        # Baseline re-seeded from the post-spawn-scan map further below (they ARE in comm at spawn):
        # leaving it at 0 made offer = own_expl − 0 = the whole spawn scan, scale = clamp(0,min=50),
        # so the actor's rendezvous gate read g=1.0 ("maximum urgency") on the first observed step
        # while the two agents stood on adjacent nodes.
        self._own_expl_at_comm[idx_t]                 = 0.0
        self._comm_prev_sync[idx_t]                   = False
        self._sync_t_last_paid[idx_t]                 = 0      # "synced at spawn" (same map, in comm)
        # Teammate belief zone — born fresh at first OOR (all-zero = "not yet born").
        self.belief_reached[idx_t]                    = 0.0
        self.pf_front_node[idx_t] = -1; self.pf_weight[idx_t] = 0.0; self.pf_dist[idx_t] = 0
        self.pf_path[idx_t] = -1; self.pf_live[idx_t] = 0.0; self.pf_acc[idx_t] = 0.0
        self.pf_seeded[idx_t] = False; self.pf_born[idx_t] = False
        self.pf_t0[idx_t] = 0
        self._comm_prev[idx_t] = False
        self.last_action[idx_t]                       = -1
        self._collision_key[idx_t]                    = torch.rand((1, self.M), device=self.dev)
        self._resample_ss_noise(idx_t)
        # Place agents using new map's start, or pin them to externally supplied positions.
        if start_override is not None:
            # comparison v2 §6.2 — IR2's own per-robot start positions, snapped to our lattice.
            agent_pos = self._snap_to_lattice(start_override, env_idx)
        else:
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
        self.novel_cells_seq_ep[idx_t] = 0.0
        self.novel_cells_ir2_ep[idx_t] = 0.0
        self._seq_union[idx_t]         = False
        self._attr_union[idx_t]        = False
        self._attr_last_px[idx_t]      = 0.0
        self._rdv_phi_prev[idx_t]      = float("inf")
        self._stall_streak[idx_t]      = 0.0
        self._revisit_streak[idx_t]    = 0.0
        self._seed_own_expl_at_comm(idx_t)
        self._refresh_obs()

    def _seed_own_expl_at_comm(self, idx_t: torch.Tensor) -> None:
        """Seed the rendezvous-gate baseline with the POST-SPAWN-SCAN explored count.

        At spawn the agents stand adjacent and are in comm, so their "map owed to the teammate"
        is zero by definition. Leaving the baseline at 0 instead made offer = own_expl − 0 = the
        entire spawn scan and scale = clamp(frac·0, min=scan_norm_nodes) = 50, so the gate saturated
        at g=1.0 on the very first observed step — the actor was told to rendezvous with the agent
        standing next to it. Mirrors the comm-gated update in _update_last_known_pos."""
        own_expl = (self.world.occupancy_torch[idx_t] != _UNKNOWN).view(
            idx_t.numel(), self.M, -1).sum(-1).float()                       # [n, M]
        self._own_expl_at_comm[idx_t] = own_expl.unsqueeze(2).expand(-1, -1, self.M).clone()

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
        self.travel_px[idx_t]                         = 0.0
        # Reset BF-from-curr cache.
        self._curr_prev[idx_t]                        = -1
        self._dist_curr_prev[idx_t]                   = float("inf")
        # H.3 — reset BF-from-teammate cache.
        self._team_node_prev[idx_t]                   = -1
        self._dist_team_prev[idx_t]                   = float("inf")
        # Teammate belief zone — born fresh at first OOR (all-zero = "not yet born").
        self.belief_reached[idx_t]                    = 0.0
        self.pf_front_node[idx_t] = -1; self.pf_weight[idx_t] = 0.0; self.pf_dist[idx_t] = 0
        self.pf_path[idx_t] = -1; self.pf_live[idx_t] = 0.0; self.pf_acc[idx_t] = 0.0
        self.pf_seeded[idx_t] = False; self.pf_born[idx_t] = False
        self.pf_t0[idx_t] = 0
        self._comm_prev[idx_t] = False
        # Reset comm-gap timer: at reset, last_known_pos is set to actual start positions
        # (see loop below), so all pairs are "freshly in comm" at t=0.
        self.t_last_comm[idx_t]                       = 0
        # Baseline re-seeded from the post-spawn-scan map further below (they ARE in comm at spawn):
        # leaving it at 0 made offer = own_expl − 0 = the whole spawn scan, scale = clamp(0,min=50),
        # so the actor's rendezvous gate read g=1.0 ("maximum urgency") on the first observed step
        # while the two agents stood on adjacent nodes.
        self._own_expl_at_comm[idx_t]                 = 0.0
        self._comm_prev_sync[idx_t]                   = False
        self._sync_t_last_paid[idx_t]                 = 0      # "synced at spawn" (same map, in comm)
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
        self.novel_cells_seq_ep[idx_t] = 0.0
        self.novel_cells_ir2_ep[idx_t] = 0.0
        self._seq_union[idx_t]         = False
        self._attr_union[idx_t]        = False
        self._attr_last_px[idx_t]      = 0.0
        self._rdv_phi_prev[idx_t]      = float("inf")
        self._stall_streak[idx_t]      = 0.0
        self._revisit_streak[idx_t]    = 0.0
        self._seed_own_expl_at_comm(idx_t)
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

    def _pathfront_belief(self, B, info, team_node, comm_bm, frontier_node, edge_free, seen_nodes,
):
        """PATHFRONT belief (EnvCfg.belief_mode=='pathfront'). Two phases on the KNOWN graph: BF particles
        lkp→frontier (transit), then absorbing diffusion (diffuse inward + frontiers lock β=utility, with
        release when a frontier is explored). Sets self._belief_p [B,M,N_max] / _belief_alive [B,M].
        State in self.pf_* (reshaped [B,M,...], observer folded into B, dim1 = teammate j)."""
        M, Kf, Lmax, N = self.M, self._pf_Kf, self._pf_Lmax, self.N_max
        dev = self.dev
        pf_front = self.pf_front_node.reshape(B, M, Kf)
        pf_w     = self.pf_weight.reshape(B, M, Kf)
        pf_d     = self.pf_dist.reshape(B, M, Kf)
        pf_pth   = self.pf_path.reshape(B, M, Kf, Lmax)
        pf_live  = self.pf_live.reshape(B, M, N)
        pf_acc   = self.pf_acc.reshape(B, M, N)
        pf_seed  = self.pf_seeded.reshape(B, M, Kf)
        pf_born  = self.pf_born.reshape(B, M)
        pf_t0    = self.pf_t0.reshape(B, M)
        comm_prev = self._comm_prev.reshape(B, M)
        utility = info["utility"]                                       # [B, N] live utility → absorb β
        eidx = self.graph.edge_idx_static
        NR = float(self.graph.NR)
        t_b = self.t.repeat_interleave(M).view(B, 1)                    # env step per observer-row
        observer_id = torch.arange(B, device=dev) % M                   # [B]
        break_bm = comm_prev & ~comm_bm                                 # [B, M] comm just lost
        self._belief_p = torch.zeros((B, M, N), dtype=torch.float32, device=dev)
        self._belief_transit = torch.zeros((B, M, N), dtype=torch.float32, device=dev)
        self._belief_alive = torch.zeros((B, M), dtype=torch.bool, device=dev)
        for m in range(M):
            active = (observer_id != m)                                 # skip self slot
            lkp_m = team_node[:, m].clamp(0, N - 1)                     # [B]
            frz = break_bm[:, m] & active
            if bool(frz.any()):
                # FREEZE ON THE SAME SET advance_pathfront calls an opening — the current
                # frontier. Two sites with two definitions of one set is how hypotheses ended up
                # born already refuted.
                # Transit BF over the KNOWN-FREE graph (edge_free), NOT the optimistic
                # unknown-passable one: the geodesic lkp→frontier must stay in known space
                # (frontier nodes are known-free, same component as lkp), so the transit point
                # follows real corridors instead of shortcutting through unknown / across walls.
                # Frontiers reachable only via unknown get dist=inf → discarded (known-only model).
                dist_m, parent_m = self.graph.bf_from_target(
                    info, target=lkp_m, edge_valid=edge_free)
                fidx = frz.nonzero(as_tuple=True)[0]
                fn, w, dh, pth = freeze_hypotheses(
                    lkp_node=lkp_m[fidx], opening=frontier_node[fidx],
                    dist=dist_m[fidx], parent=parent_m[fidx], utility=utility[fidx],
                    node_xy=self.graph.node_xy, node_spacing=NR, Kf=Kf, Lmax=Lmax)
                pf_front[fidx, m] = fn; pf_w[fidx, m] = w; pf_d[fidx, m] = dh; pf_pth[fidx, m] = pth
                pf_live[fidx, m] = 0.0; pf_acc[fidx, m] = 0.0; pf_seed[fidx, m] = False
                pf_born[fidx, m] = True
                # backdated by one hop: by the time comm is OBSERVED lost this step, the teammate has
                # already taken the step that broke it → s=1 (one hop off lkp) at the freeze step itself,
                # not s=0 (sitting on lkp, indistinguishable from the marker until the step after).
                pf_t0[fidx, m] = t_b[fidx, 0] - 1
                if _PF_DEBUG:
                    for r, b0 in enumerate(fidx.tolist()):
                        print(f"[PF] obs{b0//self.M} tm{m} t={int(t_b[b0,0])}: "
                              f"frontier={int(frontier_node[b0].sum())}n "
                              f"used={int((fn[r]>=0).sum())} dist_h={dh[r][fn[r]>=0].tolist()}")
            s_m = (t_b[:, 0] - pf_t0[:, m]).clamp(min=0)                # [B] hops since THIS freeze (s=0 at lkp)
            live_m, acc_m, seed_m, p_m, alive_m, tviz_m, w_m = advance_pathfront(
                pf_live[:, m], pf_acc[:, m], pf_seed[:, m],
                front_node=pf_front[:, m], weight=pf_w[:, m], dist_h=pf_d[:, m], path=pf_pth[:, m],
                step=s_m, frontier_node=frontier_node, utility=utility, edge_free=edge_free,
                nbr_idx=eidx, absorb_gain=float(self.cfg.belief_absorb_gain),
                beta_max=float(self.cfg.belief_beta_max),
                diffuse_lambda=float(self.cfg.belief_diffuse_lambda),
                seen=seen_nodes, just_frozen=frz)
            pf_live[:, m] = live_m; pf_acc[:, m] = acc_m; pf_seed[:, m] = seed_m; pf_w[:, m] = w_m
            use = pf_born[:, m] & active & ~comm_bm[:, m]
            self._belief_p[:, m] = torch.where(use.view(B, 1), p_m, self._belief_p[:, m])
            self._belief_transit[:, m] = torch.where(use.view(B, 1), tviz_m, self._belief_transit[:, m])
            self._belief_alive[:, m] = use & alive_m
            collapse = comm_bm[:, m] & active
            if bool(collapse.any()):
                delta = torch.zeros((B, N), device=dev)
                delta.scatter_(1, lkp_m.view(B, 1), 1.0)
                self._belief_p[:, m] = torch.where(collapse.view(B, 1), delta, self._belief_p[:, m])
                self._belief_alive[:, m] = self._belief_alive[:, m] | collapse
                cidx = collapse.nonzero(as_tuple=True)[0]
                pf_born[cidx, m] = False; pf_w[cidx, m] = 0.0; pf_front[cidx, m] = -1
                pf_live[cidx, m] = 0.0; pf_acc[cidx, m] = 0.0; pf_seed[cidx, m] = False
        comm_prev.copy_(comm_bm)                                        # remember for next-step break test

    def _refresh_obs(self, comm_mask: torch.Tensor | None = None) -> None:
        """Build per-agent obs from current per-agent occupancy + positions.

        Phase C: encoder consumes ego-centric subgraph windows, not the full lattice.
        Pass 1: build global graph + BF-from-curr + radar (feat[5]/feat[6]) per agent.
        Pass 2: cross-agent feat[4] (teammate-proximity potential) — writes to global node_feat.
        Pass 3: extract local (2·n_hops + 3)² window per agent; this is what the model sees.

        NOTE `comm_mask` here is the CONNECTIVITY mask, which step() passes as the transitively
        closed one (comm_group) whenever EnvCfg.comm_relay is set — everything observational is a
        question of what the agent can know, and a relayed link answers it the same as a direct
        one. It equals the direct mask when the relay is off. The reset path still passes None.
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
        # Diagnostics-only mirror of the GLOBAL utility field, kept per agent like _dist_curr_prev.
        # _last_obs carries only the LOCAL window (half-width n_hops*NR px), which cannot answer
        # "was the target the agent walked 400 px to actually worth more than the one it left
        # behind" — the destination is outside the window by construction.
        self._utility_global = info["utility"].view(self.N, self.M, self.N_max)
        # (re)set every refresh; see EnvCfg.div_overlap
        # ---- VALUE-FIELD [B, K]: discounted utility mass per first-step branch (see EnvCfg.vf_gamma).
        if self.cfg.div_overlap and self.M > 1:
            vf, _vf_label, _vf_mass = self.graph.value_field(
                info, gamma_vf=float(self.cfg.vf_gamma), return_branch=True)
            self._div_ov = self._branch_overlap(_vf_label, _vf_mass)          # [N, M, M, K, K]
        else:
            vf = self.graph.value_field(info, gamma_vf=float(self.cfg.vf_gamma))
            self._div_ov = None
        self._vf = vf.view(self.N, self.M, self.K)
        # ---- TEAMMATE BELIEF FILTER (Pass 1.5): advance the graph-native Bayesian estimate of
        # each teammate's node, then DERIVE feat[4] (potential) and the radar teammate_src from it.
        # team_node[B,M] = observer's belief of each teammate's node (= truth on comm). Reused by
        # the belief filter, _bf_from_teammates and (below) the geo/φ terms.
        lkp_all = self.last_known_pos.reshape(B, self.M, 2)                     # [B, M, 2]
        _lx = (lkp_all[..., 0] / float(self.graph.NR)).long().clamp(0, self.graph.LW - 1)
        _ly = (lkp_all[..., 1] / float(self.graph.NR)).long().clamp(0, self.graph.LH - 1)
        team_node = (_ly * self.graph.LW + _lx)                                 # [B, M] long
        self._belief_p = None       # [B, M, N_max] uniform posterior (set below when the filter runs)
        self._belief_transit = None # [B, M, N_max] pathfront transit-dot markers (viz only)
        self._belief_alive = None   # [B, M] bool
        belief_on = (self.M > 1 and self.cfg.use_teammate_belief
                     and info.get("edge_valid_optim") is not None)
        if belief_on:
            comm_bm = (comm_mask if comm_mask is not None else torch.eye(
                self.M, dtype=torch.bool, device=self.dev).view(1, self.M, self.M).expand(self.N, -1, -1)
            ).reshape(B, self.M)                                               # observer i ↔ teammate j
            # HYBRID expansion graph = KNOWN-FREE edges (edge_free: both endpoints known-free, no robot-
            # reachability gate — the belief BFS's from the SEED, so it must also fill known-free pockets
            # that are disconnected from the robot in the free graph yet re-entered from the unknown) ∪
            # OPTIMISTIC edges touching an unknown node, split by zone: the known→unknown FRONTIER crossing
            # (exactly one endpoint unknown) uses ONLY orthogonal edges ("archi generabili" — a diagonal
            # free→unknown cuts a corner, not a generatable path), while the unknown INTERIOR (both endpoints
            # unknown) stays 8-connected (walls invisible → all neighbours reachable). So: known respects
            # walls, exit only through real frontier edges, spread freely in the unknown, AND re-enter known
            # pockets from the far side.
            occ_nodes = self.world.occupancy_torch.view(B, -1)[:, self._node_flat_idx]   # [B, N_max]
            node_unknown = (occ_nodes == _UNKNOWN)                             # [B, N_max]
            nbr_unknown = torch.gather(
                node_unknown, 1, self.graph.edge_idx_static.clamp(min=0).view(1, -1).expand(B, -1)
            ).view(B, self.N_max, -1)                                          # [B, N_max, K]
            src_unknown = node_unknown.unsqueeze(-1)                          # [B, N_max, 1]
            crossing    = src_unknown ^ nbr_unknown                          # [B, N_max, K] free↔unknown
            internal_u  = src_unknown & nbr_unknown                          # [B, N_max, K] unknown↔unknown
            optim_ok    = internal_u | (crossing & self._orth_k.view(1, 1, -1))
            known_free  = info["edge_free"] if info.get("edge_free") is not None else info["edge_valid"]
            expand_edge = known_free | (info["edge_valid_optim"] & optim_ok)
            if self.cfg.belief_mode == "pathfront":
                # frontier NODES = known-FREE nodes that are genuine OPENINGS into the unknown: FREE with
                # ≥ pf_frontier_min_unknown UNKNOWN 8-neighbours (and ≤7). MUST be FREE (a wall touching
                # unknown is not a frontier). The ≥min_unknown gate (vs the ≥1-orthogonal default) drops
                # thin-corridor interior cells — which border the unknown behind their walls and would
                # otherwise chain the whole corridor into ONE cluster whose centroid lands mid-map — so
                # distinct openings stay distinct clusters and a transit point departs toward each.
                node_free = (occ_nodes == _FREE)                               # [B, N_max]
                n_unk = nbr_unknown.sum(-1)                                     # [B, N_max] unknown 8-nbrs
                min_unk = int(self.cfg.pf_frontier_min_unknown)
                frontier_node = node_free & (n_unk >= min_unk) & (n_unk <= 7)  # [B, N_max] openings only
                # ...and it must have something to actually reveal. See pf_frontier_min_util: the
                # count alone keeps every node whose unknown neighbours sit BEHIND A WALL, which no
                # amount of walking ever clears. `util_raw` is the pre-diffusion seed, nonzero only
                # where a real known-free→unknown ribbon exists. Applied HERE, at the single place
                # `frontier_node` is built, so every consumer agrees: cluster freezing, absorption,
                # the push target, the `acc`-immunity rule, and the `fr` flag in the trace.
                min_futil = float(self.cfg.pf_frontier_min_util)
                if min_futil > 0.0 and info.get("util_raw") is not None:
                    frontier_node = frontier_node & (info["util_raw"] >= min_futil)
                # NEGATIVE-EVIDENCE footprint: nodes where, if the teammate stood there, COMM WOULD HAVE
                # FIRED — the exact same criterion _comm_check applies to the true teammate position,
                # now evaluated against every known-free node. Previously this used the MAPPING lidar's
                # geodesic reachability within sensor_range_px: a different radius (80px vs comm's own
                # comm_range_px, 40px in this trace) AND a different model (walking distance around
                # corners vs straight-line sight) — a node just behind a nearby wall is a short walk
                # away (lidar → "seen") but comm-blind (LOS blocked), so a correct nearby hypothesis got
                # killed as "checked empty" when comm could never have detected it there at all.
                own_xy = self.pos.reshape(B, 2)                                     # [B, 2]
                node_xy = self.graph.node_xy                                       # [N_max, 2]
                diff = node_xy.view(1, self.N_max, 2) - own_xy.view(B, 1, 2)        # [B, N_max, 2]
                eucl = diff.pow(2).sum(-1).sqrt()                                   # [B, N_max]
                Sl = int(self.cfg.comm_los_samples)
                t_vals = self._los_t                                                # [Sl] cached
                # x and y are built SEPARATELY on purpose. The obvious form materializes the full
                # ray-sample tensor `pts = own + t·diff` at [B, N_max, Sl, 2] and then slices the
                # two channels out of it — at B=64, N_max=3969, Sl=40 that is a 81 MB float
                # allocation every single step, on top of the two 81 MB int64 index tensors it
                # feeds. Splitting the channels drops the intermediate entirely; the arithmetic per
                # element (own + t·diff, same dtype, same order) is unchanged.
                ix = (own_xy[:, 0].view(B, 1, 1)
                      + t_vals.view(1, 1, Sl) * diff[..., 0].unsqueeze(-1)).clamp(0, self.W - 1).long()
                iy = (own_xy[:, 1].view(B, 1, 1)
                      + t_vals.view(1, 1, Sl) * diff[..., 1].unsqueeze(-1)).clamp(0, self.H - 1).long()
                e_idx = (torch.arange(B, device=self.dev) // self.M).view(B, 1, 1).expand(B, self.N_max, Sl)
                obst = self.world.gt_torch[e_idx, iy, ix] == GT_OBST               # [B, N_max, Sl]
                if self.cfg.comm_model == "signal_strength":
                    c = self.cfg
                    frac_obst = obst.float().mean(dim=-1)                          # [B, N_max]
                    d_obst = frac_obst * eucl
                    d_free = (eucl - d_obst).clamp(min=0.0)
                    ss_xg = self._ss_xg.repeat_interleave(self.M).view(B, 1)        # env→row (b=env·M+agent)
                    ss_k  = self._ss_k.repeat_interleave(self.M).view(B, 1)
                    pl = eucl.new_full((B, self.N_max), c.ss_pl_o)
                    has_obst = d_obst > 0.0
                    pl = pl + torch.where(has_obst, 10.0 * c.ss_gamma_obst
                                          * torch.log10(d_obst.clamp(min=1.0)) + ss_k, torch.zeros_like(pl))
                    far_free = d_free >= c.ss_dist_o
                    pl = pl + torch.where(far_free, 10.0 * c.ss_gamma
                                          * torch.log10((d_free / c.ss_dist_o).clamp(min=1.0)) + ss_xg,
                                          torch.zeros_like(pl))
                    would_comm = (c.ss_p_t - pl) > c.ss_thresh
                else:
                    would_comm = (eucl < float(self.cfg.comm_range_px)) & ~obst.any(dim=-1)
                # RELAY. `would_comm` above is "if the teammate stood on that node, would *I* hear
                # him" — evaluated observer→node, a second, independent copy of the pairwise comm
                # predicate. Under multi-hop that under-claims: a teammate parked inside a
                # GROUP-MATE's radio footprint is just as audible to me, so leaving this direct
                # would let the belief keep mass on ground the flock has already ruled out.
                # OR the footprint over the observer's component. Exactly a no-op when the closure
                # is the identity (comm_relay off, or M<3), since the diagonal is always set.
                if self.cfg.comm_relay and self.M > 2 and comm_mask is not None:
                    wc = would_comm.view(self.N, self.M, -1)                       # [N, M, N_max]
                    would_comm = (wc.unsqueeze(1) & comm_mask.unsqueeze(-1)).any(dim=2).view(B, -1)
                seen_nodes = node_free & would_comm
                self._pf_seen = seen_nodes                                       # [B, N_max] for the trace
                # The openings themselves, exported alongside `seen`: "belief mass on ground that
                # is inside comm range and is NOT an opening" is the one check that settles whether
                # the model is behaving, and it cannot be evaluated from `seen` alone.
                self._pf_frontier = frontier_node                                # [B, N_max] for the trace
                self._pathfront_belief(B, info, team_node, comm_bm, frontier_node, known_free.bool(),
                                       seen_nodes)
            else:
                reached_new, p_bel, alive = update_teammate_belief(
                    self.belief_reached.reshape(B, self.M, self.N_max),
                    comm_mask=comm_bm,
                    team_node=team_node,
                    nbr_idx=self.graph.edge_idx_static,
                    edge_valid_optim=expand_edge,   # hybrid: strict in known map, optimistic touching unknown
                    expand_per_step=int(self.cfg.belief_expand_per_step),
                    gate_eps=float(self.cfg.belief_gate_eps),
                )
                self.belief_reached.copy_(reached_new.view(self.N, self.M, self.M, self.N_max))
                self._belief_p = p_bel               # [B, M, N_max] uniform Σ=1 over the expanding zone
                self._belief_alive = alive           # [B, M]

        # ---- RADAR (feat[5] b_util, feat[6] b_teammate): compress the world BEYOND the ego window
        # onto the geodesic receptive-horizon nodes. teammate_src = each OTHER agent's last-known node
        # (legacy, lkp-based — radar_team_source="lkp", the default; φ/geo_pair always stay on this
        # path regardless). teammate_obs=False (ablation) → src None → b_team all-zero (actor blind).
        if self.M > 1 and self.cfg.teammate_obs:
            lkp = self.last_known_pos[:, torch.arange(self.M, device=self.dev).view(self.M, 1),
                                      self._others_idx, :]                     # [N, M, M-1, 2]
            lx = (lkp[..., 0] / float(self.graph.NR)).long().clamp(0, self.graph.LW - 1)
            ly = (lkp[..., 1] / float(self.graph.NR)).long().clamp(0, self.graph.LH - 1)
            teammate_src = (ly * self.graph.LW + lx).reshape(B, self.M - 1)
        else:
            teammate_src = None
        # radar_team_source="belief": swap the point-source lkp for the belief FIELD itself
        # (self._belief_p, set above by either belief_mode). Falls back to the lkp path above
        # when the belief filter didn't run this step (belief_on False — teammate_belief stays
        # None and build_radar takes the teammate_src branch), so this never blinds feat[6].
        teammate_belief = None
        if (self.M > 1 and self.cfg.teammate_obs
                and self.cfg.radar_team_source == "belief" and belief_on):
            observer_id = torch.arange(B, device=self.dev) % self.M
            others = self._others_idx[observer_id]                            # [B, M-1]
            teammate_belief = torch.gather(
                self._belief_p, 1, others.unsqueeze(-1).expand(-1, -1, self.N_max))  # [B, M-1, N_max]
        b_util, b_team = self.graph.build_radar(
            info, teammate_src=teammate_src, teammate_belief=teammate_belief,
            gamma_r=float(self.cfg.radar_gamma), util_norm=float(self.cfg.radar_util_norm),
        )
        info["node_feat"][..., 5] = b_util
        info["node_feat"][..., 6] = b_team
        # H.3 — BF from each teammate's last-known position → info["bf_dist_team"] [B, M, N_max].
        # Still built: legacy feat[4]/φ/geo_pair fall back to it when the belief filter is off.
        if self.M > 1:
            self._bf_from_teammates(info)

        # ---- Pass 2: feat[4] teammate-proximity POTENTIAL on GLOBAL node_feat ----
        # With the belief ON: feat[4] = max over teammates of the per-teammate zone, PEAK-NORMALIZED
        # → a 0..1 plateau over the possible-location set (masked to node_valid, so only the known
        # part shows to the actor; the unknown part lives in the raw `bel` viz).
        # With the belief OFF (legacy): dense exp(-d_min/scale) of the BF geodesic to the nearest
        # teammate's last-known node.
        # teammate_obs=False (ablation) → skip the write, feat[4] stays zero.
        if self.M > 1 and self.cfg.teammate_obs:
            if belief_on:
                # PEAK NORMALIZATION (v11 fix). _belief_p is a Σ=1 probability field, so each node
                # carries ~1/|zone| — and the zone grows with the separation, meaning the channel
                # FADES exactly when the teammate is hardest to find. Measured on v10: in-window
                # peak 0.07-0.26 and the channel entirely absent in 39-43% of steps, against
                # utility at 0.44-0.48 present in ~90%. The docstring here always claimed a
                # peak-normalized plateau; the code never did it, so switching the belief filter on
                # silently shrank feat[4] relative to the legacy exp(-d/scale) path below (∈[0,1]).
                # Normalize by the GLOBAL peak, not the in-window one: if the belief mode sits far
                # outside the window the in-window values SHOULD stay small — that is the
                # information "he is probably not here".
                # PER-TEAMMATE, BEFORE the max over teammates. `_belief_p` is Σ=1 PER TEAMMATE, so
                # each row already lives on its own scale: a teammate lost 5 steps ago has a sharp
                # peak, one lost 200 steps ago a wide shallow plateau. Normalizing AFTER the max
                # divides every teammate by the sharpest one's peak, which at M>2 erases exactly the
                # teammates that are hardest to find — the ones the channel exists for. At M=2 there
                # is a single non-self row (the self slot is all-zero, masked by comm_bm), so
                # max-then-normalize and normalize-then-max are identical: this is a no-op for every
                # 2-agent run and a fix for M>2.
                pot = self._belief_p / self._belief_p.amax(
                    dim=-1, keepdim=True).clamp(min=1e-8)                         # [B, M, N_max] each ∈[0,1]
                pot = pot.amax(dim=1)                                             # [B, N_max] plateau ∈[0,1]
            else:
                scale_px = max(1.0, 4.0 * float(self.cfg.nr))
                d_min = info["bf_dist_team"].min(dim=1).values                    # [B, N_max]
                pot = torch.exp(-d_min / scale_px)
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
                # nodes where comm WOULD have fired this step = ground the observer has proven empty.
                # No belief may sit here; dumped so the trace/inspector can check it directly.
                "belief_seen": (self._pf_seen.view(self.N, self.M, self.N_max).clone()
                                if getattr(self, "_pf_seen", None) is not None else None),
                "belief_frontier": (self._pf_frontier.view(self.N, self.M, self.N_max).clone()
                                    if getattr(self, "_pf_frontier", None) is not None else None),
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
                # Teammate belief posterior p[N, a, j, N_max] (Σ=1 per (a,j) where alive) + alive
                # mask — for the belief heatmap viz (scripts/viz_belief.py). None when filter off.
                "belief_p": (self._belief_p.view(self.N, self.M, self.M, self.N_max).clone()
                             if self._belief_p is not None else None),
                # Pathfront transit dots (uniform 1.0 markers, viz only) — all Kf travelling hypotheses,
                # so the viewer sees every dot depart lkp→frontier even when weights concentrate on one.
                "belief_transit": (self._belief_transit.view(self.N, self.M, self.M, self.N_max).clone()
                                   if self._belief_transit is not None else None),
                "belief_alive": (self._belief_alive.view(self.N, self.M, self.M).clone()
                                 if self._belief_alive is not None else None),
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

        # ---- A KNOWN TEAMMATE'S CELL IS AN OBSTACLE ----------------------------------------------
        # A move onto the node a teammate is standing on is masked out of the action space, exactly like
        # a wall. Only while comm holds: out of contact the observer does not know where he is and has
        # nothing to mask against. This removes the same-node collision at the source instead of leaving
        # it to the step-time arbitration, and stops the policy spending probability on a cell it cannot
        # take. Guarded: if this would leave an agent with no legal move at all (a teammate plugging the
        # only way out of a dead end), the mask is left alone for that agent — being stuck is worse.
        if self.M > 1:
            others = self._others_idx                                          # [M, M-1] j != i
            tm_node = curr_idx_global.unsqueeze(1).expand(-1, self.M, -1)      # [N, M, M] j's node
            tm_node = torch.gather(tm_node, 2, others.unsqueeze(0).expand(self.N, -1, -1))
            tm_known = torch.gather(comm_mask, 2, others.unsqueeze(0).expand(self.N, -1, -1))
            blocked = ((curr_nbr_global.unsqueeze(2) == tm_node.unsqueeze(-1))  # [N, M, M-1, K]
                       & tm_known.unsqueeze(-1)).any(dim=2)                     # [N, M, K]
            masked = curr_nbr_valid & (~blocked)
            keep_old = ~masked.any(dim=-1, keepdim=True)                        # would strand the agent
            curr_nbr_valid = torch.where(keep_old, curr_nbr_valid, masked)

        # ---- CTDE critic-only global state [N, 7] (value head only; actors never see it) ----
        #   [explored_frac, t/T, geo_pair, coverage_rate, redundancy, sync_surplus, sync_staleness]
        # All ∈[0,1]. The pooled per-agent embeddings the critic also gets are EGO-RELATIVE, so they
        # carry per-agent exploration CONTENT but not the team geometry — the RELATIONAL geometry
        # lives here as geo_pair (nearest-teammate GEODESIC distance /diam, translation-invariant).
        # No absolute team position: V(s) must generalize across maps. The rest feed the value head
        # team-coordination signal the actors can't observe — notably redundancy, which explains the
        # privileged novel_scan's ~union drops (lower adv variance).
        # v11: idle_frac and imbalance were REPLACED IN PLACE by sync_surplus/sync_staleness (dim
        # stays 7 → no state_dict change, warm-start keeps working). Rationale: critic_global should
        # hold what predicts the RETURN. imbalance appears only in eval/score, which is not a reward;
        # idle_frac is a 1-step binary mean already covered by cov_rate + redundancy. Neither
        # predicts return. The sync pair does: without it V(s) cannot represent "we are about to gain
        # a lot by meeting", so the approach move's advantage stays ≈0 and the sync reward has
        # nothing to bootstrap through. Both are still computed and logged as metric/*.
        diam = float((self.graph.LH + self.graph.LW) * self.cfg.nr)
        T_max = float(max(1, self.cfg.max_episode_steps))
        # Occupancy counts: reuse step()'s single pass when we were called from step(), recompute on
        # the reset paths. CONSUMED here — cleared right after, so a later _refresh_obs (e.g. the one
        # at the tail of an auto-reset, after world.scan() has rewritten occupancy) can never read a
        # stale count.
        _counts = self._occ_counts if self._occ_counts is not None else self._count_occupancy()
        self._occ_counts = None
        own_free_a, union_free, own_expl_a = _counts             # [N, M], [N], [N, M]
        explored_frac = (union_free / self.free_total.clamp(min=1.0)).clamp(0.0, 1.0)               # [N]
        t_frac = (self.t.float() / T_max).clamp(0.0, 1.0)                                            # [N]
        # geo_pair (CTDE critic) + φ (reward): nearest-teammate GEODESIC from each agent's curr to the
        # teammate's last-known node, over the optimistic BF (info["bf_dist_team"]). Legacy lkp-based —
        # v3 belief drives ONLY feat[4]; radar/φ/geo_pair stay on this path for now.
        phi_norm_px = max(1.0, float(self.cfg.nr) * float(self.cfg.scan_norm_nodes))
        if self.M > 1:
            bt = info["bf_dist_team"].view(self.N, self.M, self.M, self.N_max)                       # [N, a, j, N_max]
            ca = info["curr_idx"].view(self.N, self.M)                                               # [N, a]
            d_at = bt.gather(3, ca.view(self.N, self.M, 1, 1).expand(-1, -1, self.M, 1)).squeeze(-1) # [N, a, j]
            geo = d_at.min(dim=2).values                                                             # [N, a] nearest teammate
            geo = torch.where(torch.isfinite(geo), geo, torch.full_like(geo, diam))
            geo_pair = (geo.mean(dim=1) / max(1.0, diam)).clamp(0.0, 1.0)                            # [N]
            # φ TOWARD THE TEAMMATE THE GATE IS ABOUT. `g` is built from the surplus owed to
            # `j_star` (the teammate owed the most map, computed further down), so φ must measure
            # the distance to THAT teammate. Reducing by min-over-teammates instead means that at
            # M>2 the gate can open because A3 is owed half the map while the dense term pays for
            # approaching A1 — two different robots, one reward. Keep the full per-teammate φ here
            # and select it with j_star once the offer is known; `geo_pair` (CTDE critic) stays on
            # the min, where "how spread is the team" is the intended reading. At M=2 the only
            # finite slot IS j_star (the self slot is left at +inf by _bf_from_teammates), so the
            # gather below reproduces the min exactly — no-op for every 2-agent run.
            self._phi_all = (d_at.where(torch.isfinite(d_at), torch.full_like(d_at, diam))
                             / phi_norm_px).clamp(0.0, 1.0)                                          # [N, a, j]
            self._geo_curr_team = (geo / phi_norm_px).clamp(0.0, 1.0)                                # [N, M] (min; re-aimed below)
            # φ at each of the K CANDIDATE moves, not just at curr — inspector only, so it is gated
            # on store_render_global and costs training exactly nothing. This is what makes the rdv
            # gate legible: `g` alone says "the gate is hot", it never says WHICH move the gate is
            # paying for. With this, rdv_preview[k] = w·g·(φ_curr − φ_nbr[k]) is the actual reward
            # the term would hand out for taking neighbour k, readable next to that action's logit.
            if self.store_render_global:
                nb = curr_nbr_global.clamp(min=0)                                                    # [N, M, K]
                K_ = nb.shape[-1]
                d_nb = bt.gather(3, nb.view(self.N, self.M, 1, K_).expand(-1, -1, self.M, -1))       # [N, a, j, K]
                d_nb = torch.where(torch.isfinite(d_nb), d_nb, torch.full_like(d_nb, diam))
                # Per-teammate, same reason as _phi_all above; selected by j_star with the reward.
                self._phi_nbr_all = (d_nb / phi_norm_px).clamp(0.0, 1.0)                             # [N, a, j, K]
                # Invalid neighbour slots carry a meaningless index; blank them rather than show a
                # number that looks like a real option.
                self._curr_nbr_valid_dbg = curr_nbr_valid.bool()                                     # [N, M, K]
                self._phi_nbr = torch.where(self._curr_nbr_valid_dbg,
                                            self._phi_nbr_all.min(dim=2).values,
                                            torch.full((), float("nan"), device=self.dev))           # [N, M, K]
            else:
                self._phi_nbr = None
                self._phi_nbr_all = None
                self._curr_nbr_valid_dbg = None
        else:
            geo_pair = torch.zeros(self.N, device=self.dev)
            self._geo_curr_team = torch.zeros((self.N, self.M), device=self.dev)
            self._phi_all = None
            self._phi_nbr_all = None
            self._curr_nbr_valid_dbg = None
        # coverage_rate: union-explored growth THIS step, scaled to "fraction-of-map per episode"
        # units (Δ·T), clamped [0,5]→[0,1]. Distinguishes "still progressing" from "stalled late".
        # On a per-env reset explored drops → Δ<0 → clamp(min=0) reads 0 (no spurious spike).
        cov_rate = ((explored_frac - self._prev_expl_frac) * T_max).clamp(0.0, 5.0) / 5.0           # [N]
        # redundancy: team double-coverage. (Σ_a own_free − union_free)/union_free per teammate ∈[0,1].
        # own_free_a / union_free come from the single occupancy pass above.
        if self.M > 1:
            redundancy = (((own_free_a.sum(1) - union_free) / union_free.clamp(min=1.0)) / (self.M - 1)).clamp(0.0, 1.0)
        else:
            redundancy = torch.zeros(self.N, device=self.dev)
        # idle_frac: fraction of agents that scanned no team-new cells this step (simple idle).
        # NOT in critic_global anymore (see the stack below) — kept for the metrics dict.
        idle_frac = self._idle_now.float().mean(dim=1)                                              # [N]
        # own_cov: per-agent OWN-map coverage — the fraction of the map THAT ROBOT actually holds,
        # as opposed to explored_frac which is the privileged team UNION. Their gap is the map that
        # exists only because a training-time union was taken; no deployed robot ever has it. Same
        # denominator as explored_frac so the two are directly comparable.
        own_cov = (own_free_a / self.free_total.clamp(min=1.0).unsqueeze(1)).clamp(0.0, 1.0)        # [N, M]
        self._own_cov = own_cov
        # sync_surplus (CTDE critic): how much map is PENDING exchange, 0 = every agent holds the
        # whole union (just synced), 1 = the agents' maps are disjoint. Derived from counts already
        # computed above — no [N,M,M,N_max] set-difference materialization per step:
        #   mean_a |M_a| / |∪M| ∈ [1/M, 1]  →  rescale so 0 = fully shared, 1 = fully disjoint.
        # This is what lets V(s) represent "we are about to gain a lot by meeting"; without it the
        # advantage of an approach move is ≈0 and the sync reward has nothing to bootstrap through.
        if self.M > 1:
            shared_frac = (own_free_a.mean(dim=1) / union_free.clamp(min=1.0)).clamp(1.0 / self.M, 1.0)
            sync_surplus = ((1.0 - shared_frac) / (1.0 - 1.0 / self.M)).clamp(0.0, 1.0)              # [N]
            # sync_staleness: steps since the most stale PAID sync, normalized. Pairs with
            # sync_surplus as "how much is owed" × "how long it has been owed".
            eye_mm = torch.eye(self.M, dtype=torch.bool, device=self.dev).view(1, self.M, self.M)
            since = (self.t.view(self.N, 1, 1) - self._sync_t_last_paid).float()
            since = since.masked_fill(eye_mm, 0.0)
            sync_staleness = (since.amax(dim=(1, 2)) / T_max).clamp(0.0, 1.0)                        # [N]
        else:
            sync_surplus = torch.zeros(self.N, device=self.dev)
            sync_staleness = torch.zeros(self.N, device=self.dev)
        # imbalance: contribution skew, normalized so 1 = one agent did everything, 0 = even split.
        # own_expl_a (non-UNKNOWN pixels per agent) also comes from the single occupancy pass above.
        if self.M > 1:
            share = own_expl_a / own_expl_a.sum(1, keepdim=True).clamp(min=1.0)                      # [N, M]
            imbalance = ((share.max(dim=1).values - 1.0 / self.M) / (1.0 - 1.0 / self.M)).clamp(0.0, 1.0)
        else:
            imbalance = torch.zeros(self.N, device=self.dev)
        # ---- BUDGET CONSUMED, per agent ∈[0,1] — agent_scalars[2]. THE DEADLINE THE ACTOR COULD NOT
        # SEE. An episode ends on whichever of the two criteria binds first (see the truncation block
        # in step()): the step cap, or the travel budget. Until v16 neither reached the actor —
        # critic_global carried t_frac, but the actor had nothing, and under a travel budget t_frac
        # does not even predict the end (train/difficult spans 3.9x in free area p50→p90, so a p50
        # episode truncates at t_frac≈0.23 and a p90 one at ≈0.90). A policy that cannot perceive its
        # own deadline cannot choose to be brisk about it, which is most of what "the agents stopped
        # going straight" was. max() of the two ratios = progress toward whichever binds FIRST, so
        # this stays the honest signal under either criterion alone or both together.
        t_ratio = (self.t.float() / T_max).clamp(0.0, 1.0).view(self.N, 1).expand(self.N, self.M)
        _budget_obs = self.budget_px()
        if _budget_obs is not None:
            travel_frac = (self.travel_px / _budget_obs.unsqueeze(1)).clamp(0.0, 1.0).maximum(t_ratio)
        else:
            travel_frac = t_ratio                                                                     # step cap only
        # ---- RENDEZVOUS RAW OBS + gate (per-agent scalars, execution-decentralized). Given to the
        # actor as agent_scalars so the policy can DECIDE when to rendezvous; the SAME ∆M gate scales
        # the dense reward. ∆M_a = surplus (cells I mapped that the teammate I owe most lacks) since
        # our last sync, normalized by the map I HAD at that sync (relative growth, not a fraction of
        # the whole canvas); staleness = steps since that sync.
        if self.M > 1:
            offer = (own_expl_a.unsqueeze(2) - self._own_expl_at_comm).clamp(min=0.0)                # [N, M, M]
            eye = torch.eye(self.M, dtype=torch.bool, device=self.dev).view(1, self.M, self.M)
            offer = offer.masked_fill(eye, -1.0)                                                     # mask self slot
            j_star = offer.argmax(dim=2)                                                             # [N, M] teammate owed most
            offer_max = offer.gather(2, j_star.unsqueeze(2)).squeeze(2).clamp(min=0.0)               # [N, M]
            # RE-AIM φ ONTO j_star (see the _phi_all comment in the geo block above). The dense rdv
            # term is w·g·(φ_prev − φ_now) and `g` is entirely about j_star, so φ has to be the
            # distance to the same robot or the two halves of the term describe different targets.
            # No-op at M=2 (one finite slot). NOTE this makes φ piecewise: if j_star switches
            # between steps, φ_prev and φ_now measure different teammates and Δφ is meaningless for
            # that one step — the same discontinuity the lkp jump already has, and `g` collapses on
            # a switch anyway (the new j_star's surplus starts from its own baseline).
            self._geo_curr_team = self._phi_all.gather(2, j_star.unsqueeze(2)).squeeze(2)            # [N, M]
            if self._phi_nbr_all is not None:
                K_dbg = self._phi_nbr_all.shape[-1]
                self._phi_nbr = torch.where(
                    self._curr_nbr_valid_dbg,
                    self._phi_nbr_all.gather(
                        2, j_star.view(self.N, self.M, 1, 1).expand(-1, -1, 1, K_dbg)).squeeze(2),
                    torch.full((), float("nan"), device=self.dev))                                   # [N, M, K]
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
            dt = (self.t.view(self.N, 1).float() - last_comm).clamp(min=0.0)                         # [N, M] steps
            # OBSERVED staleness — "how long since we last synced", agent_scalars[1]. It used to be
            # dt/T_max, which was wrong twice over: at T_max=2048 one step moved it by 0.0005 (below
            # the noise the policy can act on), and T_max CHANGES between curriculum phases, so a
            # warm start silently rescaled the input while the weights reading it stayed put.
            # rdv_urgency_T is a fixed physical scale, so this is now phase- and trace-invariant.
            staleness = (dt / float(self.cfg.rdv_urgency_T)).clamp(0.0, 1.0)                         # [N, M]
            # GATE urgency — a small, capped nudge on top of the content-driven gate; it never
            # overrides g_content on its own. Kept SEPARATE from the staleness obs above: they
            # answer different questions ("how long apart" vs "how late is it"), and collapsing them
            # would both hide staleness from the policy and duplicate travel_frac.
            if self.cfg.rdv_urgency_mode == "budget":
                t0 = float(self.cfg.rdv_urgency_start)
                urgency = ((travel_frac - t0) / max(1e-6, 1.0 - t0)).clamp(0.0, 1.0)                 # [N, M]
            else:
                urgency = staleness
            g = (g_content + float(self.cfg.rdv_urgency_weight) * urgency).clamp(0.0, 1.0)
            self._rdv_gate = g
            if self.store_render_global:
                # Everything the gate is made of, unnormalised where it is a count (offer,
                # baseline, scale are PIXELS; dt is STEPS), so the inspector can show the actual
                # arithmetic rather than a single opaque g.
                self._rdv_dbg = {
                    "g": g.detach(), "g_content": g_content.detach(),
                    "urgency": urgency.detach(),
                    "urgency_w": float(self.cfg.rdv_urgency_weight),
                    "urgency_mode": str(self.cfg.rdv_urgency_mode),
                    "offer": offer_max.detach(), "baseline": baseline.detach(),
                    "frac": frac.detach(), "scale": scale.detach(),
                    "j_star": j_star.detach(), "dt": dt.detach(),
                    "staleness": staleness.detach(),
                    # Per-candidate-move payout of the rdv term (see _phi_nbr above). NaN on
                    # invalid slots. Read it against the same step's logits to see whether the gate
                    # is actually steering the choice or just decorating it.
                    "phi_nbr": (self._phi_nbr.detach() if self._phi_nbr is not None else None),
                    "rdv_preview": (
                        (float(self.cfg.rdv_dense_weight) * g.unsqueeze(-1)
                         * (self._geo_curr_team.unsqueeze(-1) - self._phi_nbr)).detach()
                        if self._phi_nbr is not None else None),
                }
            # contact: am I in comm with ANY teammate right now? comm_mask has been in the obs dict
            # since v1 but no model ever read it, so "we are talking" reached the policy only
            # indirectly, through staleness collapsing to 0. One unambiguous bit is cheaper than
            # asking the trunk to infer an edge from a level.
            contact = (comm_mask & ~eye).any(dim=2).float()                                          # [N, M]
            # offer_frac: the surplus MAGNITUDE. g cannot carry it — it is clamped to 1 twice over
            # (g_content above, then g), so past offer_max ≥ scale "I owe him a sensor disk" and
            # "I owe him half the map" are the same number to both the actor and the reward. Same
            # denominator as explored_frac/own_cov, so it reads on the map-fraction scale the policy
            # already sees elsewhere, and it cannot saturate: offer ≤ own map ≤ free_total.
            offer_frac = (offer_max / self.free_total.clamp(min=1.0).unsqueeze(1)).clamp(0.0, 1.0)   # [N, M]
            # teammate_obs=False (ablation): gate g still feeds the rdv REWARD above, but every
            # TEAMMATE-derived scalar is zeroed — no "when to rendezvous" signal. travel_frac stays
            # live: my own remaining budget is not information about a teammate, and blinding the
            # ablation arm to its own deadline would confound the ablation with a horizon change.
            if self.cfg.teammate_obs:
                agent_scalars = torch.stack([g, staleness, travel_frac, contact, offer_frac], dim=-1)
            else:
                zc = torch.zeros_like(travel_frac)
                agent_scalars = torch.stack([zc, zc, travel_frac, zc, zc], dim=-1)
        else:
            self._rdv_gate = torch.zeros((self.N, self.M), device=self.dev)
            zc = torch.zeros_like(travel_frac)
            agent_scalars = torch.stack([zc, zc, travel_frac, zc, zc], dim=-1)                       # [N, M, 5]
        critic_global = torch.stack(
            [explored_frac, t_frac, geo_pair, cov_rate, redundancy, sync_surplus, sync_staleness],
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
            # Per-agent actor scalars (see AGENT_SCALAR_DIM in models/actor_critic.py for the
            # authoritative order): [g, staleness, travel_frac, contact, offer_frac].
            "agent_scalars":        agent_scalars,                 # [N, M, 5] float
            # Value-field: per-first-step discounted utility mass, max-normalized (actor input).
            "value_field":          self._vf,                      # [N, M, K] float ∈[0,1]
        }
        # CTDE, TRAINING ONLY. Absent from the dict entirely when cfg.div_overlap is False, so the
        # rollout buffer never allocates it, the update never looks for it, and every checkpoint
        # made before this existed replays byte-identical. Privileged by construction: it is built
        # from every agent's BF tree at once, which no single agent can see.
        if self._div_ov is not None:
            self._last_obs["div_overlap"] = self._div_ov           # [N, M, M, K, K]

    def _prev_action_onehot(self) -> torch.Tensor:
        """One-hot [N, M, K=8] of last_action. Zero everywhere when last_action == -1."""
        K = self.K
        out = torch.zeros((self.N, self.M, K), dtype=torch.float32, device=self.dev)
        valid = self.last_action >= 0
        safe = self.last_action.clamp(min=0)
        out.scatter_(2, safe.unsqueeze(-1), valid.float().unsqueeze(-1))
        return out
