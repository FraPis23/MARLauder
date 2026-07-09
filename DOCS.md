# MARLauder — Module Map, Parameters, Commands

GPU-vectorized graph MAPPO for cooperative exploration. Multi-agent with intermittent
signal-strength communication. A genuinely **learned** policy: no analytic guidepost, no
hand-picked target, no strategic candidate head — the actor steers purely from the ego-window
GAT features (local frontier utility + the beyond-window "radar" channels) plus two rendezvous
scalars, through a GRU and a pointer action head. Per-agent privileged novel-scan reward + dense
rendezvous economy; count-invariant CTDE critic (mean⊕max pooling).

---

## 1. Module map

```
MARLauder/
├── env/                Simulation: world, sensors, graph, frontier, env loop
├── models/             Networks: GAT encoder, ActorCritic, value normalizer
├── train/              MAPPO trainer: buffer, update, driver
├── eval/               Eval: deterministic rollout + trace + GIF renderer
├── scripts/            CLI entrypoints + step-by-step tests
├── viz/                inspector.html (attention + reward step-through)
├── docs/               architecture.html (pipeline diagram)
├── data/               Preprocessed map tensors (uint8 memmap + meta.npz)
├── docker/             Dockerfile + compose for the runtime image
└── DOCS.md             This file
```

### env/

| File | Purpose |
|---|---|
| `world_warp.py` | GPU LiDAR via NVIDIA Warp. Maintains **per-agent** `occupancy_torch [N, M, H, W]` (stored flat as `[N·M, H, W]` for Warp's max-3-dim kernel indexing) and `occupancy_logodds_torch`. `fuse_maps(comm_mask)` merges connected pairs via elementwise **max-magnitude** log-odds (keeps OBSTACLE evidence a plain `max` would drop). **Self-cell FREE invariant** (`_mark_pos_free`): the lidar loop starts at t=1.0 so it never marks the robot's own cell; the kernel stamps a 3×3 footprint at `2·LO_FREE` so the origin clears the strict `v > LO_FREE_TH` test. Bug history (2026-06-11): a single `LO_FREE` add landed exactly at threshold → origin stayed UNKNOWN → current graph node invalid → 0 legal moves → invalid-action teleport, masked by ~92% coverage. Pre-fix sweeps invalid. |
| `maps.py` | Load preprocessed `data/<split>/maps.npy` + `meta.npz`; sample N maps to GPU. `MultiSplit` (weighted union of splits, curriculum). |
| `frontier.py` | Torch conv2d frontier detector. `compute_frontier(occupancy)` → bool [N,H,W]. Frontier = FREE cell with 2..7 UNKNOWN neighbors. |
| `graph_lattice.py` | Core graph manager. 8-neighbor lattice on free cells, reachability flood-fill, collision-checked edges, integral-image utility. `bf_from_target(info, target, dist_init)` — overwrite-mode warm-startable Bellman-Ford from any source (BF-from-curr, BF-from-teammate). `build_radar(info, teammate_src, gamma_r)` — compresses the known world BEYOND the ego window onto the geodesic horizon gateway nodes → `b_util` (feat[5]) + `b_teammate` (feat[6]). `extract_local_window(info)` — slices the `(2·n_hops+3)²` ego window per agent. `curr_idx` from O(1) floor-divide. |
| `explorer.py` | Vectorized environment. Per-agent occupancy + positions + `last_known_pos[N,M,M,2]` + `t_last_comm` (staleness timer) + `visited_step` + `_own_expl_at_comm` (surplus baseline) + BF warm-start caches + `_rdv_phi_prev`/`_rdv_gate`. `step(action)`: sub-step LiDAR move, wall revert + **asymmetric agent-agent collision** (lower-priority agent yields via per-episode `_collision_key`), `_comm_check` (signal-strength or LOS), `fuse_maps`, reward assembly, `_refresh_obs`, then the dense `rdv_dense` term (needs the post-refresh geodesic-to-teammate field). `_refresh_obs` (3-pass, agents batched into B=N·M): build graph + BF-from-curr + BF-from-teammate + `build_radar`; cross-agent feat[4] teammate potential; `extract_local_window`; also builds `critic_global[7]` and `agent_scalars[N,M,2]`. `reload_map` for eval. `EnvCfg.from_ckpt_dict(d, **overrides)` rebuilds cfg from a checkpoint (filters unknown keys → old ckpts load). |
| `teammate_belief.py` | Teammate-state belief scaffold (last-known pos + staleness σ-inflation). |

### models/

| File | Purpose |
|---|---|
| `gat.py` | `MaskedGATLayer` + `GATEncoder`. Multi-head attention over K=8 padded neighbors, self-loop included. Pure torch — no PyG. Two learnable shaping terms on the attention scores: **A1** per-head temperature `τ_h` (`score := q·k/√D · τ_h`, clamped [0.1,10] — fixes the near-uniform softmax that made the agent pick neighbors at random), and **A2** per-head structural feature-bias `bias_h(j) = Linear(raw_feat[j, group_h]) → scalar`. Default A2 groups (entity split): `H0 [2,5]` explore (utility + its radar), `H1 [4,6]` rendezvous (teammate + its radar), `H2 [3]` recency, `H3 [5,6]` beyond-window steering. Runs on the ego window, not the full lattice. |
| `actor_critic.py` | `MarlActorCritic`. Shared ego-centric GAT encoder → per-agent `curr_emb` + `nbr_embs[K]`. **Actor** (decentralized): `actor_pre` over `[curr_emb ‖ prev_action[K] ‖ agent_scalars[2]]` → `gru_actor` (GRUCell) → `PointerHead` — a scaled dot-product `logit_k = (q·k_k)/√d · τ` over the K=8 neighbor embeddings, finite-masked + NaN-guarded, learnable temperature τ. **Critic** (CTDE): pool per-agent `curr_emb` across M via **mean⊕max** (`2·d`, count-invariant) ‖ `critic_global[7]` → `critic_pre` MLP → `gru_critic` → V scalar. **Default feed-forward** (both GRUCells bypassed); `--gru` opts into temporal memory (the modules always exist for ckpt compat). `encode_chunk()` batches the encoder across T for the MAPPO update. |
| `value_normalizer.py` | Welford online mean/var. Critic predicts normalized V, GAE uses denormalized. |

### train/

| File | Purpose |
|---|---|
| `buffer.py` | Pre-allocated rollout `[T, N, M, ...]` for obs (incl. `agent_scalars`, `prev_action`), action, logp, value, reward, done. **Per-agent** GAE-λ with a shared CTDE value baseline (`compute_gae` → advantages `[T,N,M]`, returns `[T,N]` team-mean V target). |
| `mappo.py` | PPO update. One encoder call per TBPTT chunk (`encode_chunk`), then GRU re-roll per timestep (`evaluate_step_from_enc`). Per-agent advantages. PPO clip, clipped value MSE (Huber δ=10), entropy bonus. Minibatching over N. **bf16 AMP** (GradScaler disabled — bf16 has fp32 exponent range, so the fp16 NaN-collapse mode is structurally gone). |
| `driver.py` | Main loop. `TrainCfg` defaults, rollout collection, ppo_update, milestones (25/50/75/100 %), throughput logging, optional `torch.compile`, optional curriculum (fixed or eval-score-gated), `_run_eval_suite` on the fixed 32-map `EVAL_MAP_IDX`. `_normalize_cfg` ties `n_layers = n_hops`. Logs `ep_end`. |

### eval/

| File | Purpose |
|---|---|
| `render.py` | Palette + painters (`shade_occupancy_prob`, `paint_frontier`, `paint_graph`, `paint_agent`, `paint_comm_link`, `composite_frame`, `hstack_frames`). Per-agent colors from `C_AGENTS`. |
| `rollout.py` | `EvalRollout`: deterministic single-episode play; one panel per agent (own occupancy + frontier + graph + comm-link), hstacked. |
| `trace.py` | Step-through episode trace for the inspector (per-step obs, action, per-agent reward components, GAT attention). |
| `ckpt_loader.py` | Infers architecture (`n_layers`/d/heads/agents/`use_gru`) from the checkpoint state dict. |

### scripts/

| File | Use |
|---|---|
| `run_train.py` | Full training entrypoint (`train_args.py` defines the grouped CLI). |
| `run_eval.py` | Load ckpt → deterministic episode on one map → GIF. Reads env cfg from ckpt; `--force-full-*` override. |
| `eval_final.py` | Batch eval on N random maps (or `--map-idx`). Infers architecture from the checkpoint. |
| `eval_best.py` | Score every milestone ckpt on the fixed 32-map suite → write `ckpt_best.pt`. |
| `trace_episode.py` | Emit an inspector trace JSON. |
| `baseline_random.py` | Random-policy explored-rate (sanity vs MAPPO eval). |
| `01_test_*.py … 07_*.py` | Step-by-step component tests (maps, lidar, frontier, model shapes, smoke MAPPO). |

---

## 2. Data flow (one rollout iteration)

```
data/<split>/maps.npy  (memmap, uint8 [N, H, W])
   │  env.maps.sample_batch  (N maps to GPU)
   ▼
WarpWorld.gt_torch  +  occupancy_logodds_torch  +  occupancy_torch
   │ env.step(action) for t in [0, T):
   │   1. decode action (K=8 slot) via curr_nbr_global → target node world coord
   │   2. path-follow K_sub sub-steps: Warp LiDAR per sub-step (+ asymmetric collision)
   │   3. _comm_check + fuse_maps + update last_known_pos / t_last_comm
   │   4. per-agent reward: novel_scan − revisit − stall + completion − step
   ▼
   │ _refresh_obs (agents batched into B = N·M):
   │   compute_frontier(occupancy)                        (torch conv2d)
   │   GraphLattice.build()                               (flood-fill + collision + utility)
   │   bf_from_target(curr)  → bf_dist_from_curr          (warm-started)
   │   bf_from_target(teammate lkp, edge_valid_optim)     (FREE∪UNKNOWN graph)
   │   build_radar(teammate_src) → feat[5] b_util, feat[6] b_teammate
   │   feat[4] teammate potential (cross-agent, global)
   │   extract_local_window → ego window (2·n_hops+3)²
   │   critic_global[7] + agent_scalars[N,M,2]=[∆M gate, staleness]
   │   5. rdv_dense = w · g · (φ_prev − φ_now)   (added to reward, post-refresh)
   ▼
obs dict [N, M, ...] → MarlActorCritic.act(obs, h_act, h_crit)
                          ├── ego-centric GAT encoder → curr_emb, nbr_embs
                          ├── actor_pre([curr_emb ‖ prev_action ‖ agent_scalars])
                          │     → GRU → PointerHead → action
                          └── mean⊕max pool ‖ critic_global → critic_pre → GRU → V(s)
   │ buffer.store(t, obs, action, logp, value, reward, done)
   │ after T steps: compute_gae → per-agent adv [T,N,M], team-mean returns [T,N]
   ▼ MAPPO update (k_epochs × n_minibatches × T/tbptt_steps chunks)
        ├── encode_chunk(chunk_obs)  ← ONE pass per chunk
        ├── for tt: GRU + pointer + critic re-roll
        └── optimizer.step()
   ▼ next rollout
```

---

## 3. Graph node features (F_IN = 7)

| Idx | Name | Meaning | Range |
|---|---|---|---|
| 0 | `x_rel` | `(node.x − curr.x) / win_half` — **EGO-scale** (window half-extent), so in-window coords span the full range | [-1, +1] |
| 1 | `y_rel` | `(node.y − curr.y) / win_half` | [-1, +1] |
| 2 | `utility` | **info-gain** — estimated UNKNOWN area revealed on arrival (unknown cells in a `sensor_range_px` disk / disk area), diffused along valid edges. Captures big rooms behind small openings | [0, 1] |
| 3 | `age` | **stationary recency**: `clamp((step − last_visit)/visit_age_window, 0, 1)`; never-visited = 1 (cold/re-explorable), just-walked = 0 (avoid backtrack). `visit_age_window` default 16 | [0, 1] |
| 4 | `teammate_pot` | **BF teammate-proximity POTENTIAL** — dense, wall-aware, points toward the nearest teammate's last-known position (in-window). Zero for M=1 | [0, 1] |
| 5 | `radar-util` (`b_util`) | **RADAR** — beyond-window utility mass routed geodesically onto the horizon gateway nodes (0 elsewhere). Far-exploration heading | [0, 1] |
| 6 | `radar-teammate` (`b_teammate`) | **RADAR** — beyond-window teammate direction routed onto the same gateway nodes | [0, 1] |

Invalid nodes have their feature row zeroed. Edges to invalid neighbors are masked in GAT attention. The encoder runs on the ego window `(2·n_hops+3)²` centered on `curr`, not the full lattice.

**Utility (info-gain, wall-aware)**: seeds the diffusion with estimated information gain — the count of UNKNOWN cells inside a `sensor_range_px` disk around each node (one-scan lookahead, integral image), normalized to a fraction. Then h=⌈UR/NR⌉ rounds of graph diffusion along **collision-checked `edge_valid` edges** (mass flows only through passable edges → walls block by construction), normalized by 2^h. A small frontier opening onto a big unknown component scores high.

**RADAR (`build_radar`)**: replaces the removed analytic guidepost. Nodes BEYOND the receptive horizon (`D_h = n_hops·NR` px from curr) route their mass DOWN the BF parent chain to their first gateway node at/inside the horizon — obstacle-aware, the path bends around walls; never a straight-line projection through a wall. Weight `= γ_r^(hops beyond horizon)` (travel-cost discount, `--radar-gamma` 0.92), normalized by `--radar-util-norm` (8.0). Gives a feed-forward-friendly heading toward far exploration mass / far teammates for an agent that runs out of local utility, instead of stalling in a loop.

---

## 4. Reward

Per-agent, lattice-level, in **map-independent units** (`scan_norm_nodes=50` ≈ one sensor disk, not /N_max≈1200 — so shaping is O(0.1), not O(0.001), vs the completion bonus).

```
# Per step, per agent a:
novel_scan[a]  = |cells a scanned this step ∧ ¬union_prev| / scan_norm      # NEW TO THE TEAM UNION
revisit_pen[a] = (W − age)/W  if chosen node visited within last W steps     # graduated by recency
stall_pen[a]   = 1 if ‖pos_after − pos_before‖ < nr·0.5                      # no net displacement
step_penalty   = step_cost · (edge_len / NR)                                 # axial=1, diagonal=√2

# Dense rendezvous (M>1, added after _refresh_obs):
g              = clamp(∆M / (rdv_offer_frac · own_map_at_last_sync), 0, 1)   # RELATIVE growth I owe the teammate
φ              = geodesic(curr → owed-teammate lkp) / diam
rdv_dense[a]   = w · g · (φ_prev − φ_now)                                    # NET geodesic approach

# Final reward:
reward[a] = α · novel_scan[a]
          − γ · revisit_pen[a]
          − δ_stall · stall_pen[a]
          + 1{explored ≥ 0.99} · completion_bonus
          − step_penalty
          + rdv_dense[a]
```

Defaults: `α=1.0`, `γ=0.10`, `δ_stall=0.1`, `completion_bonus=10.0`, `step_penalty_coef=0.015`, `w(rdv)=0.10`, `rdv_offer_frac=0.15`, `W=8`.

**Privileged novel-scan credit (IR2-style `r_f`)**: pays only cells **new to the team union** — a follower scanning a leader's wake earns 0, so splitting up is the highest-paying policy by construction. Privileged (training-only, CTDE; the deployed actor never sees the union). Both-scan-same-cell ties credit both (simultaneous discovery). `scan_self_delta` remains as the logged diagnostic `reward/scan_self_diag`. **There is deliberately NO separation / proximity penalty** — the design constraint is that novel-scan does the spreading, so agents never "fear the only path".

**Dense rendezvous (`rdv_dense`)**: telescoping toward the owed teammate's FIXED last-known position, gated by **relative map growth** `g` — the cells I mapped that the teammate I owe most still lacks (`∆M = own_expl − _own_expl_at_comm`), as a fraction of the map I already had when we last met (`rdv_offer_frac · own_map_at_last_sync`, floored by `scan_norm_nodes`). So `g→1` = "I have grown my known map by `rdv_offer_frac` since we last met → enough NEW content to be worth sharing", independent of canvas size. Farm-safe: oscillation cancels, and at comm `∆M→0` kills the gate so the last-known-position jump is never paid; a hover gives `Δφ=0`. **The SAME gate `g` and a normalized `staleness` are fed to the actor as `agent_scalars`** so the policy DECIDES when to rendezvous — the reward and the observation share the trigger, no precooked weight.

**Stall penalty**: physical no-progress detector — snapshot `pos` at the top of `step()`, compare after the sub-step loop. Catches both collision-revert holds and invalid/curr-node picks. Heavily weighted (`δ_stall=0.1`) to break deadlocks and force reroute/separation.

**Decentralization**: every term is computed from agent-local state (own occupancy, own visited, own `last_known_pos`/staleness) or via comm-gated set ops. The privileged team-union subtraction in `novel_scan` is the only CTDE-only signal — its unobservable overlap variance is absorbed by the centralized critic baseline (which is why `critic_global` carries `redundancy`).

**Debug full-sharing flags** (training-only sanity, NOT deployment): `--force-full-comm` (every pair connects), `--force-full-pos-sharing` (fresh teammate positions), `--force-full-occupancy-sharing` (maps fused every step). Saved in the ckpt cfg and propagated to eval.

---

## 5. Training parameters (CLI flags of `scripts/run_train.py`)

Flags are grouped by `add_argument_group` (the group title shows in the launch banner). Full source: `scripts/train_args.py`.

**Run / scale**

| Flag | Default | Note |
|---|---|---|
| `--split` | `train/easy` | `train/{easy,difficult}`, `test/{complex,corridor,hybrid}` |
| `--stage` | none | `easy`/`difficult` shorthand for the two-stage pipeline |
| `--out` | auto | Ckpts `ckpt_{025,050,075,100}.pt` + `final.pt` (carry `cfg`); with `--eval-on-ckpt` also eval GIFs |
| `--seed` | `0` | torch RNG (actions, init). Map sampling RNG is independent (fresh entropy) |
| `--device` | `cuda:0` | Or `cpu` (slow; AMP/Warp disabled) |
| `--total-steps` | `5_000_000` | Total env transitions |
| `--n-envs` | `16` | Parallel envs. Must be divisible by `--minibatches` |
| `--n-agents` | `1` | Cooperative agents per env |
| `--rollout-len` | `128` | T per PPO update. Set ≥ `--max-episode-steps` to populate `ep_end` |
| `--max-episode-steps` | `512` | Episode truncation |
| `--minibatches` | `1` | **Keep at 1** (MAPPO paper Suggestion 3: 4 minibatches fails while 1 is best on 22/23 maps) |
| `--n-hops` | `6` | Ego-window radius. Window = (2·n_hops+3)². `n_layers` tied to this |

**Sensing & communication**

| Flag | Default | Note |
|---|---|---|
| `--comm-model` | `signal_strength` | Path-loss radio (walls attenuate) or `los` (hard Euclidean + Bresenham LOS) |
| `--comm-range` | `120.0` | LOS-mode cutoff (px). Ignored in signal-strength mode |
| `--sensor-range` | `80.0` | LiDAR reach (px, matches IR2 SENSOR_RANGE) |
| `--ss-thresh` | `-70.0` | rx sensitivity (dBm): connect iff `P_R > this` |
| `--force-full-comm` / `--force-full-pos-sharing` / `--force-full-occupancy-sharing` | off | Debug sharing |

**Reward shaping**

| Flag | Default | Note |
|---|---|---|
| `--novel-scan-weight` | `1.0` | α: privileged team-union novel-scan credit |
| `--rdv-weight` | `0.10` | w: dense rendezvous strength |
| `--rdv-offer-frac` | `0.15` | relative map growth since last sync (fraction of the own map AT that sync) at which the gate `g` saturates; also normalizes the `∆M` obs |
| `--revisit-pen` / `--revisit-window` | `0.05` / `8` | γ: revisit penalty (graduated) / lookback W |
| `--stall-pen` | `0.1` | δ_stall: standing-still penalty |
| `--radar-gamma` / `--radar-util-norm` | `0.92` / `8.0` | beyond-window travel-discount / mass normalizer |

**Model ablation & warm-start**

| Flag | Default | Note |
|---|---|---|
| `--gru` | off | Enable GRU temporal memory (default is feed-forward, both GRUCells bypassed) |
| `--init-ckpt` | none | Warm-start from a checkpoint (stage-2 of the pipeline) |

**PPO / learning**

| Flag | Default | Note |
|---|---|---|
| `--lr` | `3e-4` | Adam LR |
| `--ent-coef` | `0.01` | Entropy bonus |
| `--clip-eps` | `0.15` | PPO clip ε |
| `--k-epochs` | `4` | PPO epochs per rollout (reduce to 2 if KL > 0.02) |
| `--max-grad-norm` | `2.0` | Global gradient clip |
| `--gae-lambda` / `--gamma` | `0.95` / `0.99` | GAE λ / discount |
| `--vf-coef` | `0.5` | Value loss weight |
| `--tbptt-steps` | `16` | TBPTT chunk length |

**Curriculum / eval scoring / runtime**

| Flag | Default | Note |
|---|---|---|
| `--curriculum` / `--curriculum-gated` | off | Ramp easy→difficult (fixed schedule / eval-score-gated) |
| `--curriculum-stage-splits` / `-stage-steps` / `-gate-score` / `-min-stage-iters` | — | Gated-curriculum config |
| `--score-w-imbalance` / `--score-w-overlap` / `--score-w-idle` | `0.5` / `0.25` / `0.25` | Eval-suite score weights |
| `--compile` | off | `torch.compile` the encoder |
| `--eval-on-ckpt` / `--eval-steps` / `--eval-n-maps` / `--eval-map-idx` | off / `-1` / `2` / `-1` | Milestone eval GIFs |
| `--wandb` (+ project/entity/group/run-name/mode/tags) | off | Weights & Biases |

---

## 6. Recommended training command

The canonical run is the two-stage `pipeline_rdv.sh` (easy learns to MOVE with short 128-step episodes, difficult uses 384-step episodes warm-started via `--init-ckpt`). Shared block:

```bash
COMMON="--n-envs 32 --n-agents 2 --rollout-len 256 --n-hops 6 --tbptt-steps 8 \
        --minibatches 1 --k-epochs 4 --rdv-weight 0.10 --eval-on-ckpt"

# Stage 1 — easy
python scripts/run_train.py --split train/easy --max-episode-steps 128 \
    --total-steps 2000000 $COMMON --out runs/run_easy

# Stage 2 — difficult (warm-start)
python scripts/run_train.py --split train/difficult --max-episode-steps 384 \
    --total-steps 4000000 $COMMON --init-ckpt runs/run_easy/final.pt --out runs/run_difficult
```

**VRAM / launch geometry (12 GB 4080 laptop).** The OOM driver is the `encode_chunk` update peak (∝ tbptt·n_envs·n_layers·window), which is MAP-INDEPENDENT (the model always sees the fixed 225-node ego window at n-hops=6). So easy and difficult cap at the SAME ~40 envs. `tbptt=8` halves the update peak vs 16. **32 env / tbptt=8 / rollout=256 / n-hops=6 / minibatches=1** → ~8.4 GB (easy) / ~9–11 GB (difficult). Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Smoke run (~1 min):

```bash
python scripts/run_train.py --split train/easy --total-steps 40000 \
    --n-envs 8 --n-agents 2 --rollout-len 64 --max-episode-steps 64 --out runs/smoke
```

---

## 7. Evaluation commands

### Batch eval on N random maps — `eval_final.py`

```bash
python scripts/eval_final.py runs/run_difficult/final.pt --split train/difficult --n-maps 5 --steps 512
```

Architecture (`n_agents`, `d`, `n_heads`, `n_layers`, `use_gru`) is inferred from the checkpoint. Handles `torch.compile` checkpoints (strips `encoder._orig_mod.`). Outputs `eval_map{idx:05d}.gif` per map + summary stats. `--map-idx N [N ...]` pins maps; `--seed` defaults to system entropy.

### Best-checkpoint selection — `eval_best.py`

Scores every milestone ckpt on the fixed 32-map suite and writes `ckpt_best.pt`. The web inspector exposes a "Find best ckpt" action.

### Single map by index — `run_eval.py`

```bash
python scripts/run_eval.py --ckpt runs/run_difficult/final.pt --split train/difficult \
    --map-idx 9580 --n-agents 2 --steps 256 --out runs/run_difficult/eval_map9580.gif
```

Env cfg is read from the checkpoint. Add `--force-full-occupancy-sharing` / `--force-full-pos-sharing` to force sharing at eval.

### Eval rendering

Each frame is a horizontal stack of **M panels** (one per agent): that agent's own occupancy (sigmoid of log-odds), its own frontier, the ego-centric lattice (nodes cyan→orange by utility, current-node yellow ring), the agent + trail, other agents as ghosts, a green comm-link line whenever the pair is connected, and a `[A0] t=N explored=X.X%` bar.

### Random-policy baseline (sanity)

```bash
python scripts/baseline_random.py --split test/complex --map-idx 0 --steps 512 --episodes 16 --nr 16
```

A trained policy should beat the random baseline by ≥ 2×.

---

## 8. Diagnostics — what good and bad training look like

Per-iter log line:

```
[it   N/T] ep_end=XX.X%(ended=K)  pg=±0.0NNN  v=N.NNNN  ent=N.NNN  kl=±0.0NNN  clip=N.N%  sps=NNN(NNNavg) coll=NNN upd=NNN
```

| Metric | Healthy | Warning sign |
|---|---|---|
| `ep_end` | mean explored at the terminal step of episodes that ENDED this iter (`ended=K`). Grows over iters. `n/a` until ≥1 completes | flat near random after 100+ iters |
| `pg` | small negative (-0.005..-0.02) | always positive / huge swings |
| `v` | drops then plateaus | climbing / stuck |
| `ent` | decays smoothly | crashes to ~0 (collapse) |
| `kl` | < 0.02 | > 0.1 (clip ineffective) |
| `clip` | 5-20% | > 50% (lr too high) or 0% (too low) |
| `coll`/`upd` sps | flat across run | dropping (memory pressure / recompile / oom) |

**ep_end populated only when episodes finish in the rollout** — match `rollout-len ≥ max-episode-steps` for a number every iter.

**bf16 AMP**: the update autocast is bf16 (same fp32 exponent range as fp32 → value-target spikes can't overflow → the fp16 NaN-collapse mode is structurally gone; GradScaler disabled). A finite large-negative logit mask + `nan_to_num` still guard the pointer against a one-step spike.

Exploration-quality metrics (logged to W&B): `metric/{redundancy, stall_rate, revisit_rate, mean_pair_dist, coverage_per_dist, steps_to_50, steps_to_90}`, per-term `reward/*`, per-agent `info["novel_cells_ep"]`.

---

## 9. Architecture summary

```
                          ┌──────────────────────────┐
                          │  Warp LiDAR (GPU)        │   n_rays per agent
                          │  PER-AGENT log-odds      │   occupancy[N,M,H,W]
                          └────────┬─────────────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │  _comm_check (signal-strength)   │
                  │  fuse_maps (max-magnitude)       │   comm_mask[N,M,M]
                  │  update last_known_pos / t_comm  │
                  └────────────────┬────────────────┘
                                   │  (agents batched into B = N·M)
                  ┌────────────────▼────────────────┐
                  │   frontier (torch conv2d)        │
                  │   graph_lattice.build            │
                  │   bf_from_target(curr)           │
                  │   bf_from_target(teammate lkp)   │
                  │   build_radar → feat[5], feat[6] │
                  │   feat[4] teammate potential     │
                  │   extract_local_window (ego)     │
                  └─────────────┬───────────────────┘
                                │ node_feat[N,M,W²,7], edge_idx, masks,
                                │ prev_action, agent_scalars[∆M-gate, staleness], critic_global[7]
                                ▼
   ┌──────────── Ego-centric GAT Encoder (per (env, agent))  ────────────────┐
   │  window (2·n_hops+3)²; Linear(7→d); MaskedGATLayer × n_layers            │
   │  A1 per-head temperature · A2 per-head feature-bias groups               │
   │  curr_emb [N·M, d]   nbr_embs [N·M, K=8, d]                              │
   └───────────────┬─────────────────────────────────────┬──────────────────┘
        decentralized actor                       centralized critic (CTDE)
   ┌─── per agent ─▼─────────────────────────┐  ┌─ per env ─▼───────────────┐
   │ actor_pre([curr_emb ‖ prev_action        │  │ mean⊕max pool over M (2·d) │
   │   ‖ agent_scalars[∆M-gate, staleness]])   │  │ ‖ critic_global[7]         │
   │ GRUCell → PointerHead(nbr_embs, mask)     │  │ → critic_pre MLP           │
   │   logit_k = (q·k_k)/√d · τ                 │  │ GRUCell → Linear → V(s)    │
   │ (finite mask + NaN guard) → action        │  └────────────────────────────┘
   └──────────────────────────────────────────┘
                   │  env.step → per-agent reward / done / next obs
            ┌──────▼───────────────────┐
            │ MAPPO update             │
            │  per-agent GAE-λ         │
            │  shared CTDE V baseline  │
            │  PPO clip ε=0.15, bf16   │
            │  TBPTT chunks, encode_chunk
            └──────────────────────────┘
```

Invariants:
- Encoder weights shared actor↔critic — both gradients flow back.
- Actor decentralized: each agent sees only its own ego window + its own `agent_scalars`.
- Critic count-invariant: mean⊕max pool over agents → same weights for any M (unlocks M warm-start).
- Per-agent advantages (GAE) against a single shared V; returns target = team-mean.
- Hidden states zeroed at episode resets via `(1 − done)` mask (only when `--gru`).
- All obs tensors live on GPU; no host roundtrips during rollout.
- Edge length: axial `NR`, diagonal `NR·√2` — used in all Bellman-Ford calls.
- Encoder called ONCE per TBPTT chunk.

---

## 10. Roadmap

| Ver | Goal | Status |
|---|---|---|
| v0.1 | Single-agent baseline (Warp LiDAR + lattice graph + GAT + MAPPO) | ✓ |
| v0.2 | Bellman-Ford guidepost, diagonal cost, MAPPO speedup | ✓ (guidepost later removed) |
| v0.3 | Multi-agent intermittent comm, per-agent maps, per-agent eval render, O(1) curr_idx | ✓ |
| v0.4–v0.7 | StrategicHead / analytic-target / path-bias experiments | ✓ then **REMOVED** — a genuinely learned policy |
| v0.8 | Critic mean⊕max pooling; analytic target & guidepost DELETED (F_IN 8→7); dense rendezvous reward + `agent_scalars`; realistic signal-strength comm (sensor 80px); ego-window radar channels | ✓ (current) |
| — | Perf: batched-agent env build + bf16 (+41% sps); best-ckpt selection; entity-split GAT heads (`[[2,5],[4,6],[3],[5,6]]`) | ✓ |
| next | Radar-gain fix (far-field mute at long range → `radar-gamma 0.97 / util-norm 3`); rendezvous under-experience with M=2 (surplus gate rarely fires); M>2 warm-start | open |

See [dev_log.md](dev_log.md) for the design-decision context behind each version and the current open problems.

---

## 11. Currently hardcoded (knobs not on the CLI)

Edit the dataclass to change these.

### `env.explorer.EnvCfg`

| Name | Default | Effect |
|---|---|---|
| `nr` | `16` | Lattice spacing (px). N_max scales as `(H/nr)·(W/nr)` |
| `sensor_range_px` | `80.0` | LiDAR range (px). Overridable via `--sensor-range` |
| `n_rays` | `720` | LiDAR ray count per scan |
| `utility_range_px` | `30` | Diffusion horizon (px) for the info-gain utility |
| `visit_age_window` | `16` | Recency horizon for feat[3] `age` |
| `num_sim_steps` | `5` | LiDAR sub-steps per high-level step |
| `flood_max_iters` | `200` | Max flood-fill iterations for `node_valid` |
| `done_explored_thresh` | `0.99` | Episode `terminated` threshold |
| `comm_los_samples` | `40` | Bresenham samples for the LOS comm check |
| `scan_norm_nodes` | `50.0` | Dense-reward normalizer (≈ one sensor disk of nodes) |
| `step_penalty_coef` | `0.015` | Per-axial-step movement cost (diagonal ·√2) |
| `completion_bonus` | `10.0` | One-shot terminal reward |
| `ss_*` | IR2 scale | Signal-strength path-loss params (`ss_p_t=-20`, `ss_pl_o=31`, `ss_dist_o=35`, `ss_gamma=2`, `ss_gamma_obst=4`, shadowing `X_g,K ~ U[0,13]`) |

### `train.driver.TrainCfg`

| Name | Default | Effect |
|---|---|---|
| `d_hidden` | `128` | Encoder + GRU hidden width |
| `n_heads` | `4` | GAT attention heads (must divide `d_hidden`) |
| `n_hops` / `n_layers` | `6` / `6` | Ego-window radius; `n_layers` tied to `n_hops` in `_normalize_cfg` |
| `use_gru` | `False` | GRU memory OFF by default (feed-forward); enable via `--gru` |
| `eval_every` | `10` | Iters between eval-suite runs |

### `train.mappo.MAPPOCfg`

| Name | Default | Effect |
|---|---|---|
| `clip_eps` | `0.15` | PPO clip ε (`--clip-eps`) |
| `vf_coef` | `0.5` | Value loss weight (`--vf-coef`) |
| `ent_coef` | `0.01` | Entropy bonus (`--ent-coef`) |
| `k_epochs` | `4` | PPO epochs per rollout (`--k-epochs`) |
| `tbptt_steps` | `16` | TBPTT chunk length (`--tbptt-steps`) |
| `n_minibatches` | `1` | PPO minibatches (`--minibatches`) |
| `gamma` / `lam` | `0.99` / `0.95` | Discount / GAE λ |
| `clip_vloss` | `True` | Clipped value loss (MAPPO paper §3.3) |
| `huber_delta` | `10.0` | Value-loss Huber delta (paper Tab.7); `0.0` = squared error |

**Weight init**: all `nn.Linear` / `nn.GRUCell` use orthogonal init (gain √2), policy logits gain 0.01, value head gain 1.0 — MAPPO paper Tab.7 (`models/init_utils.py`).

---

## 12. Weights & Biases + hyperparameter sweeps

`wandb` is in `requirements.txt`. Logging is **off by default** — pass `--wandb`.

**Per-iter logging**: `train/{pg_loss,v_loss,entropy,kl,clipfrac}`, `perf/{sps,coll_sps,upd_sps}`, `explore/{ep_end,ep_end_n}`, `reward/*` (per-term signed contributions), `metric/*` (exploration quality). `wandb.init(config=…)` flattens the full `TrainCfg`.

**Fixed eval suite (the sweep's scoring source)**: every `eval_every=10` iters, `_run_eval_suite` runs the policy **deterministically** on the fixed `EVAL_MAP_IDX` (32 maps evenly spaced over the big splits — same exam for every run/machine) in a persistent 1-env Explorer mirroring the training cfg. Logs `eval/{coverage_auc, contrib_imbalance, sensing_overlap, comm_duty, success_rate, steps_to_90, score, score_std}`.

`eval/score` = mean of per-map `coverage_auc − w_imb·(contrib_imbalance/(1−1/M)) − w_ov·sensing_overlap` (imbalance normalized to [0,1] before weighting). Weights via `--score-w-imbalance` (0.5) / `--score-w-overlap` (0.25) / `--score-w-idle` (0.25). AUC pads early success with the final explored-rate so finishing sooner scores strictly higher. These maps are **validation** — final reporting must use fresh random maps / `test/*`.

**Sweep** (`sweep_rdv.yaml`): tunes the rendezvous economy + core reward — `rdv-weight`, `rdv-offer-frac`, `novel-scan-weight`, `revisit-pen`, `ent-coef`. MAPPO is frozen (sweep history: no MAPPO signal, k=6 KL blowups). Param keys are the exact dashed CLI flags.

```bash
docker exec -it marlauder bash -lc 'cd /workspace/MARLauder && wandb login && wandb sweep sweep_rdv.yaml'
docker exec -it marlauder bash -lc 'cd /workspace/MARLauder && wandb agent <ENTITY/PROJECT/SWEEP_ID>'
```
