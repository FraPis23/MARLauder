# MARLauder Development Log

Session-based log of design decisions, architectural understanding, observed problems, and proposed/applied fixes. Append entries as the project evolves. Newest at top.

---

## Session 2026-07-13/14 — value-field, ablation no-GAT, fix deadlock, penalità cumulative (v0.9)

**Ablation pure-explore (chiusa).** `--rdv-weight 0` + nuovo `--no-teammate-obs` (azzera
agent_scalars, feat[4] teammate_pot, feat[6] radar-teammate; fusione mappe e critic CTDE intatti):
i loop PERSISTONO → causa = dithering tra frontiere (vicina-debole vs lontana-forte, window vs
radar), non il rendezvous.

**VALUE-FIELD (implementato).** `GraphLattice.value_field()`: albero BF-from-curr partizionato per
PRIMO passo (label propagation sui parent), V_k = Σ γ^hops·utility del ramo k, max-norm [0,1].
Riusa bf_dist/parent già calcolati (zero BF extra). `EnvCfg.vf_gamma=0.97` (`--vf-gamma`).
obs["value_field"] [N,M,K] → actor_pre (trunk) + bias `w_vf·V_k` nel PointerHead (learnable,
init 1). Plumbing: buffer, mappo, ckpt_loader. **Ckpt vecchi incompatibili** (actor_pre d+K+2+K).

**Flag GAT.** `--no-gat-actor` (actor VF-only, GAT solo critic — nessuno speedup) e `--no-gat`
(encoder MAI eseguito; critic = mean⊕max node_feat grezze → critic_feat_proj; mappo salta
encode_chunk): **4.5× sps (754 vs 169), VRAM 2.1 vs 8.3 GB** a 32 env.

**Run vfonly (no-GAT, rdv 0, blind)** easy 2M → difficult 4M: ep_end ~85% train, eval_best
test/complex @512/32: best ckpt_080 score +0.031, succ 0%, idle ~0.75. Policy decisa (ent 0.20).
**Decisione utente: la GAT resta**; prossimo train = modello completo (teammate obs + rdv 0.10)
sopra GAT+VF.

**Fix deadlock stesso-nodo (v0.9, `env/explorer.py`).** La priorità geometrica esisteva già in
`_move_and_scan` (vince chi ha meno strada residua, pareggio → chiave random); il freeze veniva
dal ramo "winner blocked" (revert di ENTRAMBI). Fix doppio:
1. **Arbitraggio a livello azione** (step(), post-decode): stesso nodo target → vince l'edge più
   corto (assiale NR batte diagonale NR·√2), pareggio → random PER-STEP; il perdente è forzato a
   hold (prende la stall penalty — insegna a non contendere), il vincitore procede libero.
2. **Ramo blocked**: il vincitore avanza parzialmente fino al ring min_dist attorno al perdente
   (guard anti-muro, fallback prev pos) invece del revert totale di entrambi.
Test: 35/35 contese assiale-vs-diagonale risolte al vincitore giusto, mai entrambi mossi/fermi,
tie fairness 217/434 (50%).

**Penalità cumulative (v0.9).** Due streak per-agente (reset con gli env):
- **Stallo consecutivo**: δ_stall × (1+β·(streak−1)) clamp a cap. `--stall-streak-beta 0.5`,
  `--stall-streak-cap 4`. Test: -0.1 → -0.4 in 7 step, cap tenuto, reset al movimento.
- **Revisit recenti consecutivi**: la rampa graduata esistente × (1+β_rev·(streak−1)) SENZA cap e
  SENZA conteggio per-nodo (fuori dalla finestra W=8 nessuna memoria → passaggi futuri legittimi
  gratis). `--revisit-streak-beta 0.5`. Test ping-pong A↔B: streak 1→7, penalità ×4 lineare.
Telemetria: reward_terms stall_streak/revisit_streak.

**Docs.** `docs/architecture.html` → v0.9 (box value_field, BUS value_field[8], actor_pre,
w_vf·V_k nel pointer, nota flag ablation). Nuove pipeline: `pipeline_noRdv.sh`,
`pipeline_vfonly.sh` (--no-gat), `pipeline_v09.sh` (modello completo + eval_best in coda).

---

## Session 2026-07-09 — perf: batched-agents env build + bf16 AMP (+41% sps)

Two of the four proposed optimizations applied (user approved #1/#2; #3 checkpointing deferred
until VRAM is the binder, #4 compile candidate next):

1. **Agents batched into the batch dim in `_refresh_obs`** (`env/explorer.py`). All GraphLattice
   ops are batch-agnostic on their leading dim, so Pass 1 (build + BF-from-curr + radar +
   teammate BF), Pass 2 (feat[4] potential) and Pass 3 (window extract) now run ONCE on
   B = N·M instead of M sequential calls. `_bf_from_teammates` rewritten batched (one BF per
   teammate SLOT, M-1 total, warm-start caches preserved via reshape views); new static
   `_others_idx [M, M-1]` table. Render stash uses `[B]→[N,M]` views instead of stacks.
   **Verified bit-identical obs/rewards** vs the pre-refactor code on a seeded 30-step +
   snapshot harness (gotcha found on the way: `Explorer.rng = np.random.default_rng()` at
   explorer.py:163 is UNSEEDED → spawn/maps differ run-to-run; the harness patches it).
   env.step 17.9 → 11.5 ms (N=4, M=2, easy).

2. **bf16 AMP** (`train/mappo.py`, `train/driver.py`). `AMP_DTYPE = bfloat16` when supported
   (Ampere+; fp16+GradScaler fallback else). PPO update autocast now bf16 — same fp32 exponent
   range → value-target spikes can't overflow → GradScaler disabled (passthrough), the fp16
   NaN-collapse failure mode is structurally gone (guard kept). Rollout collection, bootstrap
   and `_run_eval_suite` forward passes now also run under autocast (previously fp32);
   log_softmax stays fp32 under autocast so rollout logp matches update-side numerics.

**Benchmark** (train/difficult, 32 env, M=2, rollout 256, tbptt 8, 12 iters): ~130 sps clean
iters / 99-133 avg vs 92 avg with the old code on the identical config (488-iter pipeline log)
→ **+41%**. 4M-step difficult stage: ~12.1 h → ~8.6 h. Learning curves healthy (v_loss ↓,
ep_end 24→64% in 12 iters, kl/clip normal, zero nan_skips). Smoke: trace_episode + inspector OK
on the batched env.

**Stuck-agent diagnosis (traces, no code change yet).** User observed agents stalling in
rooms/corridors ~40-50 geodesic nodes from remaining frontiers, intermittently. Stall-window
scan of the ckpt_best traces confirms the radar hypothesis: m240 agent0 stalls steps 308-384
with max feat[5] b_util = **0.013** (radar mute: 0.92^45/8 ≈ 0.004/node) while an in-window
utility 0.58 sits across a wall (Euclid-near, geodesic-far → attractive nuisance). m160 stalls
at bu 0.05/0.14. m320's 80% is time-limit, not stalling. Intermittency explained: b_util is a
mass SUM — big far rooms stay visible, thin far frontiers vanish. **Planned fix (approved
direction): radar_gamma 0.92→0.97 + util_norm 8→3 (~20× signal at 45 hops), then a short A/B
retrain vs control; sweep only if the A/B confirms but under-delivers. Step penalty explicitly
rejected (informational problem, not motivational).** M>2 postponed until after the radar fix
(critic mean⊕max is already count-invariant → M=2 ckpt warm-starts M=3/4).

---

## Session 2026-07-08 — dead-code sweep + runs cleanup (during v0.8 training)

Dead code REMOVED (verified: syntax + CPU env/model/buffer smoke; the running v0.8 pipeline is
unaffected — its code is already in memory):
- `train/buffer.py`: `BufferStats`, `slice_step`, `slice_chunk`, `_OBS_KEYS_*` constants, and the
  duplicate `curr_nbr_valid` buffer entry (it IS `action_mask` env-side; update reads action_mask).
- `train/mappo.py`: `_slice_obs`.
- `env/world_warp.py`: `reset_occupancy`, `team_occupancy`, `team_occupancy_prob`.
- `env/explorer.py`: dead cfg knobs `scan_reward_weight` + `team_reward_weight` (with `--scan-weight`
  / `--team-weight` CLI); dead obs keys `comm_event` (Phase-1b instrumentation never wired, incl.
  `self._comm_event`) and `bf_parent_from_curr` (the [N,M,N_max] per-step stack — the per-agent
  `info["bf_parent_from_curr"]` stays, the radar needs it). `EnvCfg.from_ckpt_dict` filters unknown
  keys → old ckpts still load.
- `eval/ckpt_loader.py` + `run_eval.py` + `eval_final.py`: legacy `path_bias`→`path_bias_learn`
  remap shims (those ckpts are incompatible since v0.8 anyway).
- `scripts/debug_single.sh` deleted (used --no-strategic-head/--div-coef/--path-bias-floor — all gone).
- `viz/inspector.html` + `eval/trace.py`: last guidepost/target/gate UI leftovers removed.
- `runs/`: deleted all stopped (ckpt_stop.pt) + mid-truncated runs (~4.3GB freed, 25 dirs).
  Kept: all COMPLETE runs, the RUNNING rdv pair, step_0x/trace167 artifacts, runs/profile (1.7GB,
  flagged for user decision).

---

## Session 2026-07-07 — v0.8: critic pooling, analytic-target/guidepost REMOVED, dense rendezvous reward

Big refactor. All verified in container `marlauder` (syntax, env step, model forward, eval GIF,
inspector trace, multi-iter run_train) and a real easy→difficult training pipeline was launched.

**Problem #1 — critic input.** The CTDE critic took MAPPO's CL/concat-local state
`Linear(M·d + G, d)` (worst input in the MAPPO paper: bakes M into weights, noisy V → noisy
advantages). Replaced with symmetric **mean ⊕ max pooling** over agents → `Linear(2·d + G, d)`,
count-invariant (same weights for any M). Chose mean+max over attention pooling deliberately: no
softmax dead-zone (the bug we had to repair in GAT + pointer), always passes gradient, sits on the
proven mean-field baseline from step 0.

**Analytic target + guidepost — FULLY REMOVED.**
- `node_feat` F_IN 8→7: `0 x_rel,1 y_rel,2 utility,3 age,4 teammate_pot,5 radar-util,6 radar-teammate`
  (guidepost channel gone; radar reindexed 6/7→5/6). GAT head-3 A2 bias reassigned to the radar
  channels [5,6] — head 3 is now the beyond-window steering head that replaces the guidepost's role.
- `graph_lattice.py`: deleted `build_guidepost`, `build_guidepost_v2`, `select_target_*`,
  `analytic_next_hop` (~400 lines). Kept `bf_from_target` + `build_radar` (radar still uses the
  from-curr BF; `guidepost_iters` kept as a generic BF iteration cap).
- `explorer.py`: removed Pass-1 target selection/deconfliction/guidepost bookkeeping,
  `_prev_target_node`/`_dist_prev`/`_target_prev`/`_steps_on_option`, obs `guidepost_*`/`target`
  keys, config `target_*`/`analytic_target`/`target_kind`/`disable_guidepost`. Kept `curr_idx_global`
  (invalid-action fallback) + `bf_dist_team` (feeds geo_pair, radar, feat[4] team_pot).
- `actor_critic.py`: dropped `strategic_gate_eps` + `_strategic_gate`. Plumbing cleaned in
  driver/run_train/train_args. eval consumers fixed (rollout/trace/render/ckpt_loader/run_eval/
  eval_final).

**critic_global 8→7:** dropped `tgt_dist` (was descriptive-only, needed the analytic target).
Considered adding absolute team centroid cx,cy but DROPPED it — absolute position makes V overfit
map layouts; `geo_pair` (nearest-teammate geodesic, translation-invariant) already gives the
relational link. Final: `[explored_frac, t/T, geo_pair, cov_rate, redundancy, idle_frac, imbalance]`.

**Reward — dense rendezvous, no separation (IR2 r_s spirit).**
- REMOVED give_bonus, recv_bonus (sparse rendezvous), overlap_pen (redundant — novel_scan union
  already zeroes redundant co-scan), proximity_pen (this WAS the "fear the teammate" term). Deleted
  `_setop_rewards`, `_proximity_penalty`, `last_meeting_node_mask`.
- ADDED `rdv_dense = w·g·(φ_prev − φ_now)`: telescoping toward the owed teammate's FIXED last-known
  pos, gated by surplus `g = clamp(∆M/(rdv_offer_frac·H·W), 0, 1)`. φ = geodesic curr→nearest-
  teammate /diam. Farm-safe: oscillation cancels, at comm ∆M→0 kills the lkp-jump credit, hover
  gives Δφ=0. NO separation penalty (privileged novel_scan does the spreading → no "fear the only
  path", per the design constraint).
- Reward now: `novel_scan − revisit − stall + completion − step + rdv_dense`.

**Observations — raw rendezvous ingredients.** Added `agent_scalars` [N,M,2] = [∆M surplus-gate,
staleness] fed to the ACTOR input (`actor_pre = Linear(d+K+2)`; plumbed through buffer + mappo like
prev_action). The policy DECIDES when to rendezvous — not a precooked weight.

**VRAM / launch geometry (measured on the 12GB 4080 laptop).** OOM driver = the `encode_chunk`
update peak (∝ tbptt·n_envs·n_layers·window), which is MAP-INDEPENDENT (model always sees the fixed
225-node ego window). So easy and difficult cap at the SAME ~40 envs; difficult is slightly heavier
env-side (bigger grids) so ≤ easy, never more. `tbptt=8` halves the update peak vs 16 (which caps at
a razor-edge 24 envs). Chose **32 env / tbptt=8 / rollout=256 / n-hops=6 / minibatches=1** →
~8.4GB (easy) / ~9–11GB (difficult), safe headroom. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Launched:** `pipeline_rdv.sh` — easy (128-step episodes, 2M steps) → difficult (384-step episodes,
4M steps, warm-start via `--init-ckpt`). Fresh train required (F_IN + actor_pre + critic_pre shapes
changed → no old-ckpt warm-start). New sweep `sweep_rdv.yaml` (rdv-weight/offer-frac/novel/revisit/
ent). DELETED obsolete: `scripts/04_test_graph.py`, `05_test_env_random.py`, and 5 stale sweeps
(sweep.yaml, sweep_div, sweep_stage2, sweep_stage0_commit, sweep_v06_belief).

---

## Session 2026-06-29 / 06-30 — StrategicHead removal, analytic-only, reward & critic overhaul

Large cleanup + simplification pass. Goal: a genuinely *learned* policy — env supplies an analytic guidepost as context, but no longer hard-forces the agent toward it. **All pre-2026-06-29 checkpoints are broken (actor/critic input dims changed) → full retrain required. 5 sweep yamls reference deleted flags → obsolete.**

### Done — architecture cleanup (06-29)

- **StrategicHead fully deleted** + everything used only by it: `CAND_FEAT_DIM`, all `cand_*` (feat/valid/xy/idx/bf_first_hop) extraction + obs keys, Gumbel-ST, commitment knobs (`switch_margin`/`max_steps_on_option`/`disable_strategic`), `target_choice/logits/argmax`, `_diversity_loss`+`div_coef`, reward terms `target_switch`+`yield`, env `_objective_switch_and_yield`/`_zero_cand`/`_candidate_*`, graph `extract_topk_candidates`. Code is now **analytic-target-only**: env's `graph_lattice` picks the global target; actor does local control. `--target-mode` choices now `analytic|nearest` (no `learned`).
- **next_hop_dir removed** from actor input → `actor_pre = Linear(d+K → d)` (was d+2K). Route still reaches the actor as context via `node_feat[5]` (guidepost ribbon) + `node_feat[2]` (utility) through the GAT; the explicit one-hot was redundant with message-passing.
- **GAT self-loop added** (`gat.py MaskedGATLayer`): node attends to itself (K+1 slots, self always valid) → removed the all-invalid NaN fallback. Standard GAT (Veličković 2018).
- **Dead `k_exp` line removed** (`gat.py`).
- **Explicit per-step movement cost** (`explorer.py`): axial = `step_cost`, diagonal = `step_cost·√2` (via `graph.edge_len[action]/NR`), invalid/no-move = 0. New `reward_terms["step"]`.
- Deleted head-only diag scripts: `measure_thrash`, `diag_decouple`, `diag_path_head`, `diag_division`, `diag_actor_pre`, `diag_loops`, stale `profile_step`. Architecture diagram: `docs/architecture.html`.

### Done — reward study (06-29 → 06-30)

- **`in_comm` dropped** from `critic_global` (derivable from `geo_pair`) → was 4-dim, briefly 3.
- **`team_delta` (β) removed from reward** — double-counted novel cells already paid by `novel_scan` (reintroduced the free-ride novel_scan exists to kill). `team_reward_weight` now a dead knob (0); `team_delta` kept as a metric only.
- **Set-op normalization fixed** — give/recv/overlap were `/N_max` (~24× smaller than novel AND map-size dependent) → now `/scan_norm` (map-independent units). Coefs rescaled to preserve train/easy magnitude: give 1.5→0.06, recv 0.5→0.02, overlap 3.0→0.12. `scan_norm_nodes=50` ≈ one sensor disk of lattice nodes (range 80px / NR 16 = 5 rings → π·5²≈78, minus occlusion/overlap ≈ 50).
- **`progress_reward` REMOVED (06-30)** — it shaped `(d_prev−d_new)` toward the committed analytic target → soft-forced the policy to *follow the selector* instead of learning the criterion. Deleted block + reward-sum term + telemetry + `progress_reward_coef` (EnvCfg / run_train / `--progress-reward-coef`). BF machinery (`_dist_prev`/`_prev_target_node`) kept — still used by guidepost commitment.
  - **Decided AGAINST analytic-PBRS** `γφ(s')−φ(s)` as a replacement: when the analytic target flips, φ changes basis → telescoping breaks → flip-farming (the exact bug the old BF-field code fought). The resulting beyond-window blindness is to be solved in **observation** (coarse global frontier channel, O1 — NOT yet done), not by bribing toward a target.
- **`novel_scan` credit note**: it's privileged (the `& ~union_prev` subtraction uses team-union info the actor never observes → CTDE-legit). The controllable part ("go to MY visible frontier") is learnable; the unobservable overlap-variance is meant to be absorbed by the centralized critic baseline — which is *why* `critic_global` now carries redundancy/coverage (below).
- **`completion_bonus=10` verified, left unchanged (R4)**: discounted-to-start `10·0.99^T` ≈ 0.18–1.3, but ≈ 9.9 at the finish-decision step (dominant) and ≈ half the dense return undiscounted. Raising it would inflate value-target variance (rare +10 outlier vs ~0.05/step → Welford std blows up → shrinks normalized dense gradients).

### Done — critic_global extension (O2, 06-30)

`critic_global` **3 → 8** (`CRITIC_GLOBAL_DIM`), all features ∈[0,1], M=1 guards → 0, built env-side in `_refresh_obs`. History-state critic (GRU) kept — unbiased per Amato CTDE §4.6.

| # | feature | formula | role |
|---|---|---|---|
| 1 | explored_frac | union_free / free_total | non-stationarity |
| 2 | t_frac | t / max_steps | non-stationarity |
| 3 | geo_pair | nearest-teammate BF, mean, /diam | agent geometry (reuses Pass-1 `bf_dist_team`) |
| 4 | coverage_rate | `Δexplored·T` clamp[0,5]/5 | progressing vs stalled (derivative) |
| 5 | redundancy | `(Σ own_free − union)/union/(M−1)` | **lets critic explain novel_scan's ~union drops → lower adv variance** |
| 6 | tgt_dist | BF agent→committed analytic target, mean, /diam | descriptive critic input — **no penalty/forcing** |
| 7 | idle_frac | `(novel_count≤0)` mean over agents — *simple* idle | coordination health, no extra BF flood |
| 8 | imbalance | `(max_share − 1/M)/(1 − 1/M)` | contribution skew |

Critic distance is **descriptive-only** (a value-baseline input, never a reward term) → the "penalized forever for leaving the best frontier" failure mode does not apply (that only bites if distance enters the *reward*, which is exactly what `progress_reward` did and why it was cut). Chose *simple* idle (novel_count≤0) over the 3-clause refined idle to avoid an extra nearest-frontier BF flood; accepts that productive transit reads as "idle" — acceptable for a descriptive feature.

### Implementation

- `models/actor_critic.py`: StrategicHead class + `CAND_FEAT_DIM` gone; `actor_pre = Linear(d+K)`; `_strategic_gate`/`strategic_gate_eps` vestigial (trace.py compat); `CRITIC_GLOBAL_DIM = 8`.
- `models/gat.py`: self-loop (K+1 slots), `k_exp` deleted.
- `env/explorer.py`: reward block (step penalty, team_delta/progress removed, set-op `/scan_norm`); `critic_global` 8-feature build in `_refresh_obs`; new state `_prev_expl_frac` + `_idle_now` (init in `__init__`, reset auto-handled via `_refresh_obs`, per-env reset → Δ<0 → clamp0 → no spike); EnvCfg knobs removed/rescaled.
- `train/buffer.py`, `train/mappo.py`, `train/driver.py`: cand/target_choice/div_loss wiring removed; buffer `critic_global` auto-sizes from `sample_obs`.
- `scripts/train_args.py`, `scripts/run_train.py`, `eval/rollout.py`, `eval/trace.py`: strategic + progress flags removed; analytic-target rendering.
- **Smoke-verified on GPU** (marlauder container, M=1 and M=2): env build, `critic_global` finite ∈[0,1] with correct M=1 guards, 8-step rollout, `model.act`, `model.evaluate`+backward (gradients finite), `critic_pre.in_features = 2·d+8 = 264`.

### Open

- **O1 — coarse global frontier/explored obs channel** (low-res): fills the beyond-window blindness reopened by removing `progress_reward`, keeping the policy *learned* (give it the info, don't bribe a direction). NOT done.
- **SRU** (Spatially-Enhanced Recurrent Unit, arXiv 2506.05997): separate-branch decision pending. Motivated only if learned spatial memory of left-behind frontiers is needed beyond O1; needs TC-dropout / DML or underperforms.
- **Dead knobs** still present (left in place, low priority): `scan_reward_weight`/`--scan-weight` (scan_self is diagnostic-only, not in reward), `proximity_penalty_coef=0`.
- **5 obsolete sweep yamls** reference removed flags (`--top-k`, `--target-switch-pen`, `--switch-margin`, `--div-coef`, etc.): `sweep.yaml`, `sweep_stage2.yaml`, `sweep_div.yaml`, `sweep_stage0_commit.yaml`, `sweep_v06_belief.yaml` — delete vs rewrite undecided.

---

## Session 2026-06-26 — Realistic sensor + signal-strength comm model

### Done

Ported IR2's realistic comms + bumped sensor range, to make the physical layer verisimilar.

- **Sensor range** `60 → 80` px (EnvCfg `sensor_range_px` default; `--sensor-range`). Matches IR2 `SENSOR_RANGE`. Fixed, not randomized (IR2 doesn't randomize the sensor). Old checkpoints restore their saved `60` via `from_ckpt_dict` → eval unaffected.
- **Comm model** — new `comm_model` flag:
  - `"los"` (EnvCfg default, back-compat): legacy hard `dist < comm_range_px` AND no GT obstacle on the segment.
  - `"signal_strength"` (run_train CLI default): log-distance path-loss radio (IR2 / hal-03365129). Segment split into free vs obstacle length; `PL = PL_o + [10·γ_obst·log10(d_obst)+K]·(d_obst>0) + [10·γ·log10(d_free/d_o)+X_g]·(d_free≥d_o)`; connect iff `P_R = P_T − PL > ss_thresh`. **Walls attenuate (γ_obst=4) instead of hard-blocking.**
- **Per-episode domain randomization**: shadowing noise `X_g ~ U[0,13]` (free), `K ~ U[0,13]` (obstacle) resampled per env reset (`_resample_ss_noise`, GPU). Effective free-space range ≈ 69–311 px depending on draw (centered ~150, near old fixed 120); walls cut it hard.
- Default-`los` in EnvCfg but default-`signal_strength` in the CLI keeps old-ckpt eval faithful while new trainings opt in and persist `comm_model` in their ckpt.

### Implementation

`env/explorer.py`: EnvCfg fields (`comm_model`, `ss_*`); `_resample_ss_noise(idx_t)` GPU helper; `_comm_check` SS branch (fully vectorized over `[N]`, reuses the S-sample segment trace); `_ss_xg/_ss_k` `[N]` buffers allocated in `__init__`, resampled in both reset paths.
`scripts/run_train.py`: `--comm-model`, `--sensor-range`, `--ss-thresh`; wired into EnvCfg.

### Open

- SS params (`PL_o=31`, `d_o=35`, thresholds) are IR2 pixel-scale values dropped onto MARLauder's 1000px/16NR scale — sane (range ~150px) but worth a sweep on `--ss-thresh` if comm too generous/strict.

---

## Session 2026-05-27 (d) — Phase C: ego-centric subgraph encoder

### Done

Encoder now consumes a per-agent `(2·n_hops + 3)²` window centered on the agent's current node, instead of the full `N_max ≈ 1200` lattice. Same receptive field for the model (`n_layers` lattice hops) at `n_hops = 2`, but with ~24× fewer node embeddings per forward pass.

### Also done in this push

- **SPS measurement fix** (`train/driver.py`): the wall-clock denominator now excludes milestone-eval time. Previous `sps_iter = steps_per_iter / (t_end - t_prev)` included eval GIF emission between iters because `t_prev = t_end` was set BEFORE the eval block. Replaced with `sps_iter = steps_per_iter / iter_time` (collect+update only) and `sps_all = total_env_steps / total_train_time` (sum of iter_time, eval-free).

### Implementation

`env/graph_lattice.py`:
- `n_hops` constructor param. Precomputes:
  - `window_offsets [W², 2]`, `window_idx_table [N_max, W²]` (global flat idx per window cell), `window_local_edge_table [W², K]` (local-window edge layout).
- `extract_local_window(info)` — gathers per-env window views from global tensors. Edge indices in returned dict are LOCAL (∈ [0, W²)). Carries `local_to_global [N, W²]` and `curr_nbr_global [N, K]` for downstream use.

`env/explorer.py`:
- `EnvCfg.n_hops` knob.
- `_refresh_obs` refactored to 3-pass: (1) build global + warm-started target-rooted BF per agent; (2) cross-agent `feat[5]` on GLOBAL `node_feat`; (3) `extract_local_window` per agent, stack across M.
- `step()` action decode uses `obs["curr_nbr_global"]` (env needs global flat idx to compute world coords and update `visited_step`).

`models/gat.py`, `models/actor_critic.py`: **unchanged**. Both already accept `[B, N, F]` for arbitrary N. The encoder simply sees a smaller graph.

`train/driver.py`:
- `TrainCfg.n_hops`. `_normalize_cfg` ties `cfg.n_layers = cfg.n_hops` so GAT depth matches the window radius (boundary nodes get one extra ring as padding via the `+3` in window side).

`scripts/run_train.py`: `--n-hops` flag.
`scripts/eval_final.py`: reads `n_hops` from saved cfg so eval matches training window.
`eval/rollout.py`: `target_xy` extracted from new `obs["guidepost_target_xy"]` field (global world coords). The old `node_xy[guidepost_target]` would index out-of-bounds since `node_xy` is local-window now.

### Notes on `feat[5]` (teammate_pos)

Teammate's nearest GLOBAL lattice node is marked on the GLOBAL `node_feat` before window extraction. If that node falls inside the local window: `feat[5] = 1` at the local slot. If outside: teammate info "lost" in this view. Acceptable: when teammate is many hops away, the only useful action is "explore more, rendezvous later" — exact position offers little local guidance.

### Measured (same command: `--n-agents 2 --n-envs 64 --total-steps 30000 --max-episode-steps 512 --minibatches 4 --compile`)

| Config | avg sps | iter-3 sps | explored end (3 iters) |
|---|---|---|---|
| baseline (B1-redo, full graph, 2 layers) | 334 | 333 | ~38% |
| **C n_hops=2 (49 nodes, 2 layers)** | **596 (+78%)** | **723** | **43.5%** |
| **C n_hops=6 (225 nodes, 6 layers)** | **412 (+23%)** | **523** | **49.6%** |

n_hops=2 is pure speed gain (same receptive field as v0.3). n_hops=6 trades some speed for a 6-hop receptive field — visible quality lift even in 3 iters (49.6% vs 43.5% explored).

### Compatibility notes

- Old checkpoints (pre-C) load fine — encoder weights are shape-agnostic over the graph-size dim. But behaviorally they were trained with full-graph context; feeding them window-only at eval will deviate slightly from their training distribution. Not a problem for new trainings.
- `MarlActorCritic.encode_chunk` (PPO update path) inherits the new local-shape automatically — same `node_feat [T, N, M, W², F]` flow.

### What's next

User verifies. After approval, proceed to Phase A (target diversification — coordination unblock).

---

## Session 2026-05-27 (c) — Phase B1 redo: BF FROM target with warm-start

### Done

Replaced `curr`-rooted BF with **target-rooted BF** that warm-starts from the previous step's distances.

Why the swap: in an undirected graph with symmetric edge costs, shortest path curr→target equals shortest path target→curr reversed. `next_hop` derived from either rooting is identical. But target moves MUCH slower than curr — between steps, target usually unchanged (or shifts one lattice cell), so previous step's `dist` rooted at target is mostly correct. Warm-starting cuts typical convergence from ~30 iters to a few.

### Implementation

`env/graph_lattice.py`:
- `K_INDEX_TABLE[3, 3]` — (sign(d_li), sign(d_lj)) → K-slot. For analytic direction.
- `bf_from_target(info, target, dist_init)` — **overwrite-mode** BF (`dist = best_vals` not `min(dist, best_vals)`). Handles staleness from invalidated edges correctly because each iter recomputes dist[v] from current neighbor dist. Forces `dist[target] = 0` each iter to anchor. Early-exit via `torch.equal(prev_dist, dist)` (1 sync per call, modern Ada cost ~10 μs).
- `select_target_no_bf(utility, node_valid)` — target = argmax(utility · node_valid). No BF needed for selection because flood-fill in `build()` already ensures node_valid implies reachability from curr.
- `analytic_next_hop(curr, target, edge_valid)` — O(1) direction from `sign(target_li - curr_li, target_lj - curr_lj)`. Returns k-slot + per-env "first edge clear" bool. Not used as a BF-skip in batched mode (with N=64, P(all envs clear) ≈ 0); kept for future per-env masked execution.
- `build_guidepost_v2(info, target, dist_init)` — orchestrator. Calls BF, reconstructs path by walking parent from curr, builds mask + path_xy + next_hop + guidepost_nbr_bias. Writes same info-dict keys as the old `build_guidepost` so downstream code unchanged.

`env/explorer.py`:
- `_target_prev [N, M]` and `_dist_prev [N, M, N_max]` cache state, init in `__init__`, reset on env reset.
- `_refresh_obs` per agent now: select target → cache check → warm-start dist_init → `build_guidepost_v2`. Cache updated each step.

### Measured (user command: `--n-agents 2 --n-envs 64 --total-steps 200_000 --max-episode-steps 512 --minibatches 4 --compile`)

| Variant | avg sps |
|---|---|
| baseline (curr-rooted BF, early-exit) | 320 |
| B2 alone (no early-exit) | 320 (net wash) |
| **B1 redo (target-rooted BF + warm-start + early-exit)** | **334** (+4.4%) |

Modest lift, matching the prediction that BF is only ~4% of step time. Encoder is the real bottleneck.

### Not implemented (and why)

**O(1) analytic skip-BF-entirely**: in batched mode with N=64 envs, the probability that ALL envs have a clear analytic direction is ≈ 0, so the "skip BF if all clear" check never fires. Per-env masked execution (run BF only on envs where analytic failed) is possible via boolean indexing but adds gather/scatter overhead that washes the BF saving. Kept `analytic_next_hop` method for future use (single-env eval, or as a sanity-check on next_hop direction).

### What's next

User verifies. After approval, proceed to Phase C (ego-centric subgraph encoder — the actual bottleneck).

---

## Session 2026-05-27 (b) — Phase B abandoned: BF not the bottleneck

### Outcome

Phase B (BF speedup) reverted. Net sps change ≈ 0. BF is not the bottleneck.

### What was tried

**B2 — Drop GPU sync from BF early-exit.**

Removed the `bool(update.any().item())` check that synced every 8 iters. Hypothesis: ~10 syncs/call × 75 μs = 750 μs stall per call, eliminated by running all 78 iters unconditionally.

User measured before B2: 320 sps at N=64, M=2.
After B2: 320 sps unchanged.

**Postmortem analysis.** Recalibrated cost estimates:
- BF iter cost is dominated by kernel-launch overhead (~50 μs/iter) at N=64, not raw FLOPs.
- Original with early-exit: ~30 iters × 50 μs + 10 syncs × ~10 μs (modern Ada drivers overlap better than 75 μs) ≈ 1.6 ms/call.
- After B2: 78 iters × 50 μs ≈ 3.9 ms/call. **Slightly worse.**
- M=2 → BF cost ≈ 7.8 ms/step. Total step ~200 ms → BF is ~4% of total. Even a perfect BF rewrite buys < 5% sps.

The original sync estimate (~75 μs/sync) was based on dated CUDA reference numbers; modern PyTorch + Ada drivers overlap syncs with queued kernels much better, making the early-exit check effectively free.

### Why B1 (warm-start) is also a dead end

The dev_log claim "warm-start saves iters because most of the graph is unchanged between steps" was wrong. BF computes `dist` from a single source = `curr_idx`. Every step the agent moves → `curr_idx` changes → all dist values change. Previous step's `dist` (relative to old source) gives zero useful prior for current step's BF (relative to new source).

For warm-start to actually help, we'd need all-pairs shortest paths cached — O(V³) precompute, infeasible. Or a static-source formulation, which contradicts the agent-centric design.

### Revert

`env/graph_lattice.py:312-322` restored to the original early-exit form. No other files touched in B.

### What's actually the bottleneck

Encoder forward pass dominates. With N_max=1200, B=N·M=128, n_layers=2, d=128:
- QKV projections: 3 × 128 × 1200 × 128² ≈ 7.5 GFLOPs per layer.
- 2 layers + softmax + gather + aggregate ≈ 25 GFLOPs per encoder forward.
- At RTX 5080 ~50 TFLOPs effective: ~0.5 ms per forward.
- But ~80% of nodes are invalid (zeroed) in early training → ~80% wasted FLOPs.

**Phase C (ego-centric subgraph) targets this directly.** Window of (2·n_hops + 3)² ≈ 49 nodes at n_hops=2 vs 1200 full lattice → 24× reduction in encoder FLOPs. Big sps potential.

### What's next

Skip remainder of Phase B. Proceed to Phase C.

---

## Session 2026-05-27 — Architecture deep-dive + reward and graph fixes

### Project state recap

MARLauder v0.3 is a GPU-resident MARL exploration system:

- Vectorized LiDAR simulation in Warp (`env/world_warp.py`), per-agent occupancy maps `[N, M, H, W]` stored flat as `[N·M, H, W]` to fit Warp's max 3-dim kernel indexing.
- 8-neighbor regular lattice graph (`env/graph_lattice.py`), node spacing `NR=16` px → `N_max=1200` nodes on 480×640 canvas.
- Intermittent communication: euclidean distance + Bresenham LOS check on GT. Map fusion via elementwise max of log-odds (idempotent). Position sharing via `last_known_pos[N, M, M, 2]`.
- Per-agent graph build → GAT (2 layers, d=128) → GRU → PointerHead. Critic centralized over concatenated `curr_emb` of all agents.
- MAPPO with TBPTT (16 steps), GAE-λ, value normalization, AMP fp16, torch.compile on encoder.
- Reward: team union-FREE delta (shared across agents).

### Architectural understanding (key findings)

#### Node identification
- Lattice positions are deterministic: `node_xy[k] = ((lj+0.5)·NR, (li+0.5)·NR)` with `k = li·LW + lj`.
- Nearest node from world position is **analytic O(1)**: `lj = floor(x/NR), li = floor(y/NR)`. No QuadTree, no argmin needed. TOM used QuadTree because their nodes were dynamic; ours are a fixed grid.
- The flood-fill in `build()` defines `node_valid` (reachable from robot through 8-connected FREE cells). Disconnected free pockets behind walls are correctly excluded.

#### Edges
- `edge_idx[N, N_max, K=8]` is a lookup table: for each node `k`, the flat index of its K=8 neighbors, or `-1` if no edge.
- `-1` means either: (a) geometrically out of canvas, (b) endpoint invalid, (c) edge segment collides with obstacle (verified by S=5 sample collision check along the segment).
- `edge_len[K]` = `[NR√2, NR, NR√2, NR, NR, NR√2, NR, NR√2]` — diagonals cost √2 more than axials. Used by BF for true Euclidean shortest path.

#### Bellman-Ford guidepost
- Called per agent per step in `_refresh_obs()`, AFTER move, scan, comm fusion, reward computation.
- Source = `curr_idx[n]` (agent's current node). Computes `dist[n, k]` = shortest weighted path from curr to every node.
- Target = `argmax(utility · node_valid · reachable)`. The "most exploration-valuable reachable node," potentially many hops away.
- **Only one bit reaches the policy**: `next_hop[n]` = the first neighbor of curr on the path to target. Converted to `guidepost_nbr_bias[N, K]` (one-hot over K=8 directions), added directly to PointerHead logits as a learnable-scaled prior.
- Full path (`guidepost_path_xy`) is render-only.
- Iterations: up to `LH + LW + 8 ≈ 78`, with early-exit check every 8 iters (causes GPU↔CPU sync via `.any().item()`).
- Per agent per step: `78 × N × N_max × K ≈ 24M` element ops batched on GPU.

#### Network data flow
- `obs` flattens `(N, M) → B` so each agent in each env is an independent graph in the batch.
- GAT input: `[B, N_max, 7]` → 2 layers → `[B, N_max, 128]`.
- Attention is **local**: each node attends only to its K=8 lattice neighbors. After 2 layers, each node's embedding encodes its 2-hop Chebyshev neighborhood (~25 nodes).
- Only 9 vectors survive per agent: `curr_emb [d]` + `nbr_embs [K, d]`. The other 1191 embeddings are discarded.
- PointerHead = scaled dot-product over K=8 candidate neighbors. Not aggregation attention; "attention as selection". One logit per move direction.
- Critic: concat `curr_emb` of all M agents → Linear → GRUCell → scalar value V(s).

#### 7 node features (in order)
| i | name | meaning |
|---|---|---|
| 0 | x_rel | (node_x − curr_x) / (W/2) |
| 1 | y_rel | (node_y − curr_y) / (H/2) |
| 2 | utility | frontier density in window |
| 3 | visited | 1 if ever visited |
| 4 | recency | last_visit_step / current_step |
| 5 | teammate_pos | 1 at node nearest to teammate last-known pos |
| 6 | guidepost | 1 if on BF shortest path to target |

### Observed problems

1. **Eval agents stuck**: in some GIFs, both agents walk a few steps then freeze. Root cause: when both agents pick targets via the same `argmax(utility)` and converge on the same destination node, the agent-agent collision rule (`d < NR` → both revert) traps them. Deterministic argmax + symmetric observations + collision → infinite loop. Confirmed by user.
2. **Maps not random in eval**: `eval_final.py` defaulted `--seed=42`, producing identical map indices each run. Fixed (now defaults to None = system entropy).
3. **Wall placement on reset**: `_spread_starts_graph` picked M nearest lattice nodes regardless of validity; if a node landed on a wall pixel, the agent spawned there and was stuck. Fixed (now filters to nodes with `gt == FREE` before topk).
4. **Agents not separating**: with shared team reward and identical observations (when in comm range), agents converge on identical actions. No incentive to spatially diverge.
5. **Target switching frequently** (after reward change): observed in a fresh short training. Likely due to new reward signal (step penalty + completion bonus) — policy still adapting. Not a bug.

### Reward design (current state)

**Implemented this session**:
```
reward = max(0, delta_union_free)            # coverage increment
       + terminated · completion_bonus        # one-shot bonus at terminal step (default +10.0)
       − step_penalty_coef / max_steps        # constant per-step pressure (default 0.1/512)
```

Calibration: with total_free ≈ 150k cells and a typical good step discovering ~100 cells, `delta ≈ 0.00065`. Step penalty `0.000195` is ~30% of a mediocre step. Tune `step_penalty_coef` based on observed behavior.

**Deferred**:
- Overlap/anti-redundancy penalty: needs per-agent scan-area tracking.
- Rendezvous reward: postponed until baseline coordination works.
- Per-agent differentiation: currently all M agents receive the same team reward.

### Q&A summary (this session)

#### Q: Is the graph rebuilt every step? Why not incremental?

Yes, full rebuild every step (`build()` + `build_guidepost()`). Not incremental because:
- Full rebuild is already sub-millisecond per env on RTX 5080.
- Incremental update is bug-prone: tracking which edges changed, propagating BF distances correctly, keeping per-agent maps consistent.
- Diminishing returns: with N=32 batched, full rebuild ≈ 5ms total; incremental might shave 3ms. Not worth complexity until other bottlenecks resolved.
- Warm-start BF (item 6 in schedule) captures most of the incremental benefit with much less complexity.

#### Q: Why 8 ops per node in BF? Isn't it enough to check which neighbor is on the path to source?

Confusion of direction. BF computes `dist[v]` = shortest path FROM source TO v. To compute `dist[v]` for each node v, we don't know in advance which incoming neighbor gives the minimum — must check all K=8 and take the min. This is per BF iteration, per node.

Once dist is converged, path reconstruction (target → curr via parent pointers) follows only ONE pointer per node. That's the cheap part. The 8x is intrinsic to BF on an 8-connected graph.

#### Q: Why does GPU sync require CPU?

`update.any().item()` in the early-exit check converts a GPU scalar to Python int. The Python `if` statement runs on CPU, so the value must come back. To return a value, GPU must finish all queued kernels first (drain its pipeline). That stall is the sync cost (~50-100 μs each).

GPU-only alternative: drop the early-exit check entirely, always run all 78 iterations. Result is identical (BF converges within max iters; extra passes are no-ops). Total cost depends on whether the saved sync time exceeds the cost of extra iters — likely yes for early-converging cases (most steps after initial exploration).

#### Q: Can we do O(1) analytic direction with warm-start BF fallback?

Yes — best of both worlds:
1. Compute analytic next direction from `(curr_li, curr_lj) → (target_li, target_lj)`: `dir = (sign(dli), sign(dlj))`.
2. Trace the analytic staircase path from curr to target, checking `edge_valid` along the way.
3. If all edges valid → use analytic direction (one hop, O(path_length) edge checks, fully parallel).
4. If any edge invalid → fall back to warm-start BF (init dist from previous step → 2-5 iters convergence).

This addresses the concern about not committing to a path with mid-route obstacles: we validate the full analytic path before using it. Cost: O(path_length × N) edge_valid lookups per step (negligible), plus rare BF fallback. Expected average BF iter count drops from ~30 to ~5.

#### Q: Multi-agent target collision — what's the best fix for M > 2?

**Option B (clustering)** scales fine:
- top-K (K = 3·M) high-utility nodes: O(N_max log K), trivial.
- K-means with M centroids on K points: <0.1 ms for M ≤ 16.
- Hungarian assignment (agent → cluster by BF distance): O(M³), trivial up to M=16.
- BF distances are already computed.

Total overhead at M=8: ~0.1 ms/step. Scales cleanly to M=16+.

For probabilistic teammate positions (when out of comm range, future ToM module): replace point estimate `last_known_pos` with a position distribution. For target diversification, use the expected position. The clustering algorithm itself doesn't change.

#### Q: Should occupancy be shared in early phases (debugging)?

Yes for sanity baseline. Add `force_full_comm: bool = False` to `EnvCfg`. When True, override `_comm_check` to return all-ones. Lets us train with god-mode comm to verify the policy can learn coordination at all. If it can't even with full info, the bottleneck is elsewhere (reward shaping, architecture, target diversity).

### Ego-centric subgraph design

Current GAT processes all 1200 nodes per agent per step, but only 9 embeddings (`curr_emb` + `nbr_embs`) are actually used downstream. The other 1191 embeddings are computed and discarded. Waste scales with `N_max` and `n_layers`.

Proposal: extract a `(2·n_hops + 3) × (2·n_hops + 3)` window centered on `curr_idx` per agent. Run GAT only on this window. Receptive field for `curr_emb` matches the global-graph version with same `n_layers`.

| n_hops | side | nodes | FLOPs (B=64) | Speedup |
|---|---|---|---|---|
| 2 | 7 | 49 | ~100M | 24× |
| 4 | 11 | 121 | ~500M | 5× |
| 6 | 15 | 225 | ~1.4B | 1.8× |

The `+3` (not `+1`) in `2·n_hops + 3` comes from needing one extra ring of nodes at the window boundary so the boundary nodes' attention propagation completes correctly. Without it, boundary node embeddings are degraded which corrupts `curr_emb` indirectly.

Configurable via `n_hops` parameter. Match `MarlActorCritic.n_layers = n_hops`. Precompute window index table `[N_max, window_size]` once at `__init__`.

### Updated schedule (status board)

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | O(1) nearest node | ✅ Done | `graph_lattice.py` analytic floor-divide |
| 2 | Step penalty + completion bonus | ✅ Done | `explorer.py` reward block |
| 3 | Start position FREE-node filter | ✅ Done | `_spread_starts_graph` checks gt |
| 4 | Eval random map sampling | ✅ Done | `eval_final.py` `--seed=None` default |
| 5 | Target diversification (option B: top-K + cluster + Hungarian) | Next | Single change in `build_guidepost` argmax. Biggest behavioral fix for "agents not separating" and "stuck collision" |
| 6 | O(1) analytic direction + warm-start BF fallback | Next | Replaces full BF with cheap path-trace; BF only when path blocked |
| 7 | Force-full-comm debug flag | Next | Quick |
| 8 | Ego-centric subgraph (configurable `n_hops`) | High | Big GAT speedup, bigger receptive field option |
| 9 | Overlap penalty | Medium | Anti-redundancy. Requires per-agent scan tracking |
| 10 | GPU-only BF convergence (drop `.item()` sync) | Medium | Performance polish |
| 11 | Network architecture review vs IR2 | Medium | Read-only analysis of `model.py` |
| 12 | Rendezvous reward | Later | After 5-9 stable |

### Acceptance criteria to track

- After #5: eval GIFs show agents heading to distinct frontier regions, no stuck-in-place behavior.
- After #6: profile shows BF average iter count drops from ~30 to ~5; training sps improves by 10-20%.
- After #8: GAT forward pass <0.3 ms (vs current ~2 ms); receptive field tunable.
- After #11: documented diff between MARLauder and IR2 architectures + decision on what to port.

---

## Earlier sessions (compressed)

### v0.3 — Multi-agent intermittent comm
- Per-agent occupancy maps `[N, M, H, W]`, stored flat for Warp compatibility.
- Comm check: Euclidean range + LOS sampling. Map fusion via max log-odds.
- `last_known_pos` tracks each agent's view of other agents.
- Agent-agent collision: hard env constraint, both revert if `d < NR`.
- CLI parameter reduction: only user-facing flags exposed; internals hardcoded.
- Eval rendering: per-agent side-by-side panels (each shows that agent's own occupancy map). Comm link drawn between agents when in range.

### v0.2 — Guidepost + speedup
- Bellman-Ford guidepost on GPU (replaces feat[6] placeholder).
- Diagonal edge cost √2.
- Batched encoder over T chunks (`encode_chunk`).
- `torch.compile` on encoder.
- Renaming `belief*` → `occupancy*` (belief reserved for future ToM).

### v0.1 — Baseline
- GPU-vectorized LiDAR via Warp.
- 8-neighbor lattice graph replacing TOM NodeManager/QuadTree.
- Masked GAT + GRU + PointerHead.
- MAPPO with GAE+TBPTT.
- ~252 sps single-agent baseline; trained from 5-10% (random) to 14-25% explored on `train/easy`.
