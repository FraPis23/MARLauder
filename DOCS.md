# MARLauder — Module Map, Parameters, Commands

GPU-vectorized graph MAPPO for cooperative exploration. Multi-agent with
intermittent line-of-sight communication. v0.3 (per-agent occupancy maps,
comm + map fusion, per-agent eval render, step-penalty + completion-bonus
reward).

---

## 1. Module map

```
MARLauder/
├── env/                Simulation: world, sensors, graph, frontier, env loop
├── models/             Networks: GAT encoder, ActorCritic, value normalizer
├── train/              MAPPO trainer: buffer, update, driver
├── eval/               Eval: deterministic rollout + GIF renderer
├── scripts/            CLI entrypoints + step-by-step tests
├── data/               Preprocessed map tensors (uint8 memmap + meta.npz)
├── docker/             Dockerfile + compose for the runtime image
└── DOCS.md             This file
```

### env/

| File | Purpose |
|---|---|
| `world_warp.py` | GPU LiDAR via NVIDIA Warp. Maintains **per-agent** `occupancy_torch [N, M, H, W]` (stored flat as `[N·M, H, W]` for Warp's max-3-dim kernel indexing) and `occupancy_logodds_torch`. Exposes `team_occupancy()` (union across agents) and `fuse_maps(comm_mask)` (elementwise max log-odds where two agents are in comm). |
| `maps.py` | Load preprocessed `data/<split>/maps.npy` + `meta.npz`. Sample N maps to GPU. |
| `frontier.py` | Torch conv2d frontier detector. `compute_frontier(occupancy)` → bool [N,H,W]. Frontier = FREE cell with 2..7 UNKNOWN neighbors. |
| `graph_lattice.py` | Core graph manager. Builds 8-neighbor lattice on free cells, reachability flood-fill, collision-checked edges, integral-image utility, and **Bellman-Ford guidepost** (Dijkstra path to nearest high-utility node, edge weights = `NR` axial / `NR·√2` diagonal). `curr_idx` from O(1) floor-divide. |
| `explorer.py` | Vectorized environment. Per-agent occupancy + positions + `last_known_pos[N, M, M, 2]` + `visited_step`. `step(action)`: move with sub-step LiDAR, wall + agent-agent collision revert, `_comm_check` (Euclidean range + Bresenham LOS), `fuse_maps`, reward, `_refresh_obs` (per-agent graph build + guidepost). `_spread_starts_graph` places M agents on distinct nearest FREE lattice nodes. |
| `teammate_belief.py` | Stub for v0.7 ToM teammate-state estimator. Unused in v0.3. |

### models/

| File | Purpose |
|---|---|
| `gat.py` | `MaskedGATLayer` + `GATEncoder`. Multi-head attention over K=8 padded neighbors. Pure torch — no PyG. Accepts any leading batch dim. |
| `actor_critic.py` | `MarlActorCritic`. Shared encoder, actor (GRUCell + PointerHead over K=8 neighbors), critic (CTDE: concat per-agent curr_emb → MLP → GRUCell → V scalar). `encode_chunk()` batches encoder across T for MAPPO speedup. |
| `value_normalizer.py` | Welford online mean/var. Critic predicts normalized V, GAE uses denormalized. |

### train/

| File | Purpose |
|---|---|
| `buffer.py` | Pre-allocated rollout `[T, N, M, ...]` for obs/action/logp/value/reward/done. GAE-λ on team-mean reward. |
| `mappo.py` | PPO update. One encoder call per TBPTT chunk (`encode_chunk`), then GRU re-roll per timestep. PPO clip, value MSE, entropy bonus. Minibatching over N. AMP fp16. Single grad clipper + scaler. |
| `driver.py` | Main loop. `TrainCfg` defaults, rollout collection, ppo_update calls, milestones (25/50/75/100 %), throughput logging (sps, collect-sps, update-sps), optional `torch.compile` on encoder. |

### eval/

| File | Purpose |
|---|---|
| `render.py` | Palette + painters: `shade_occupancy_prob`, `paint_frontier`, `paint_graph`, `paint_path`, `paint_target`, `paint_agent`, `paint_comm_link` (green line between agents in comm range), `composite_frame` orchestrator, `hstack_frames` (concatenate per-agent panels horizontally). Per-agent colors from `C_AGENTS` palette. |
| `rollout.py` | `EvalRollout`: deterministic single-episode play. Builds **one panel per agent** showing that agent's own occupancy + frontier + graph + guidepost + comm-link, hstacks them into a wide frame. |

### scripts/

| File | Use |
|---|---|
| `01_test_maps.py` | Load split, render 2×2 mosaic with start markers. |
| `02_test_lidar.py` | Single LiDAR scan from start. Verifies probabilistic occupancy. |
| `03_test_frontier.py` | Single scan + frontier overlay. Verifies frontier ∩ wall = 0. |
| `04_test_graph.py` | Build graph + guidepost after short walk. Renders edges, utility-colored nodes, amber path, target ring. Prints diag/axial = √2. |
| `05_test_env_random.py` | Random policy rollout over N envs, per-env GIF (full render). |
| `06_test_model_shapes.py` | Instantiate model, forward + backward, assert grad flows. |
| `07_smoke_mappo.py` | Tiny PPO run (2 envs × 32 steps × 2 updates × 1 epoch). |
| `baseline_random.py` | Random-policy explored-rate on a fixed map (sanity vs MAPPO eval). |
| `run_train.py` | Full training entrypoint. |
| `run_eval.py` | Load ckpt → deterministic episode on one specific map → GIF. Requires explicit `--d-hidden`, `--n-heads`, `--n-layers`. |
| `eval_final.py` | Batch eval on N random maps. Infers architecture from checkpoint weights (no need to pass widths). Strips `encoder._orig_mod.` keys from torch.compile checkpoints. |

---

## 2. Data flow (one rollout iteration)

```
data/<split>/maps.npy  (memmap, uint8 [N, H, W])
   │
   ▼  env.maps.sample_batch  (N maps to GPU)
WarpWorld.gt_torch  +  WarpWorld.occupancy_logodds_torch  +  WarpWorld.occupancy_torch
   │
   │ env.step(action) for t in [0, T):
   │   1. decode action via curr_nbr  →  target node world coord
   │   2. path-follow K_sub sub-steps:  Warp LiDAR per sub-step
   │   3. compute_frontier(occupancy)            (torch conv2d)
   │   4. GraphLattice.build()                   (flood-fill + collision check + utility)
   │      GraphLattice.build_guidepost()         (Bellman-Ford with edge_len)
   │   5. reward = Δ(union FREE) / total_free   (clamped ≥ 0)
   ▼
obs dict [N, M, ...] → MarlActorCritic.act(obs, h_act, h_crit)
                          ├── encoder (shared GAT) → curr_emb, nbr_embs
                          ├── actor GRU + PointerHead → action sample
                          └── critic GRU → V(s)
   │
   ▼ buffer.store(t, obs, action, logp, value, reward, done)
   │
   │ after T steps, compute_gae(rewards, values) → advantages, returns
   │
   ▼ MAPPO update (k_epochs × n_minibatches × T/tbptt_steps chunks)
        │
        ├── encode_chunk(chunk_obs)  ← ONE pass per chunk
        ├── for tt in chunk_len: GRU + pointer + critic head, accumulate PPO loss
        └── optimizer.step()
   │
   ▼ next rollout
```

---

## 3. Graph node features (NODE_INPUT_DIM = 7)

| Idx | Name | Meaning | Range |
|---|---|---|---|
| 0 | `x_rel` | `(node.x - curr.x) / (max(H,W)/2)` | [-1, +1] |
| 1 | `y_rel` | `(node.y - curr.y) / (max(H,W)/2)` | [-1, +1] |
| 2 | `utility_norm` | # frontier cells inside `(2·UR+1)²` window around node, / area | [0, 1] |
| 3 | `visited` | 1 if this node was ever `curr_idx` for this agent, else 0 | {0, 1} |
| 4 | `last_visit_norm` | `last_visit_step / current_step` | [0, 1] |
| 5 | `teammate_pos` | 1.0 at the lattice node nearest to **each teammate's last-known position** (M>1 only). At reset all teammates are co-located (in comm range); thereafter updates only when the pair is in range + LOS. Zero for M=1. | {0, 1} |
| 6 | `guidepost` | 1 if node lies on Bellman-Ford shortest path from curr to nearest high-utility reachable node | {0, 1} |

Invalid nodes have feature row zeroed. Edges to invalid neighbors masked in GAT attention.

Utility computed via integral image of the frontier mask (one prefix sum + 4 corner gathers per node), square window approximation of disk, edge-clipped (border nodes underestimate — acceptable).

---

## 4. Reward (v0.3)

```
union_free_t   = |{cell : team-occupancy_t[cell] == FREE}|      # union across M agents
discovery_t    = max(0, (union_free_t − union_free_{t−1}) / gt_total_free)
completion_t   = (explored_rate_t ≥ done_explored_thresh) · completion_bonus    # one-shot
step_penalty   = step_penalty_coef / max_episode_steps                          # constant
team_reward_t  = discovery_t + completion_t − step_penalty
reward[e, a]   = team_reward_t                                                  # broadcast to all M agents
```

Defaults (from `EnvCfg`): `step_penalty_coef = 0.1`, `completion_bonus = 10.0`, `done_explored_thresh = 0.99`, `max_episode_steps = 512`.

Calibration: with `max_episode_steps=512`, step penalty is `~0.000195` per step (total budget `0.1` per unterminated episode). A typical good step discovers ~50–150 cells on `train/easy` for a delta of ~0.0003–0.001 — step penalty is ~20–65% of a mediocre step's reward, creating time pressure without drowning the discovery signal.

Properties:
- Single scalar per env per step (shared by all agents).
- Discovery is computed on the **team union** of FREE cells, so two agents scanning the same area count once (implicit anti-redundancy).
- Episode done if `explored_rate ≥ done_explored_thresh` (`terminated`) or `t ≥ max_episode_steps` (`truncated`). Completion bonus fires only on `terminated`, not `truncated`.
- M=1: discovery equals "Δ own occupancy free / total free".
- M>1: dual-agent on same zone → no double-count, but team reward is shared so agents have no direct incentive to diverge spatially. Diversification work is tracked in `dev_log.md`.

---

## 5. Training parameters (CLI flags of `scripts/run_train.py`)

| Flag | Default | Range / Note |
|---|---|---|
| `--split` | `train/easy` | `train/easy`, `train/difficult`, `test/{complex,corridor,hybrid}` |
| `--out` | `runs/run_default` | Output dir. Ckpts at `ckpt_{025,050,075,100}.pt` + `final.pt`. With `--eval-on-ckpt` also `eval_ckpt_{pct}_m{0,1}.gif` on random maps |
| `--seed` | `0` | RNG seed for split sampling, eval map selection, policy init |
| `--device` | `cuda:0` | Or `cpu` (CPU is slow; AMP/Warp disabled) |
| `--total-steps` | `5_000_000` | Total env transitions (`n_envs × rollout_len × iters`) |
| `--n-envs` | `16` | Parallel envs. Must be divisible by `--minibatches` |
| `--n-agents` | `1` | Number of cooperative agents per env |
| `--comm-range` | `120.0` | Communication range (px). Set 0 to disable comm entirely |
| `--rollout-len` | `128` | T per PPO update |
| `--max-episode-steps` | `512` | Episode truncation (typically ≥ rollout-len) |
| `--minibatches` | `1` | PPO minibatches per epoch. Must divide `n-envs`. 4 is a good default at `n-envs=32` |
| `--lr` | `3e-4` | Adam learning rate for actor |
| `--ent-coef` | `0.01` | Entropy bonus weight |
| `--compile` | off | `torch.compile` the encoder (~2× update speedup) |
| `--eval-on-ckpt` | off | Emit 2 eval GIFs on random maps at each milestone (25/50/75/100%) |

**On `--seed`**: training uses the given integer to seed `torch.manual_seed`, env split sampling, and the milestone-eval random map picker. For `eval_final.py`, `--seed` defaults to `None` (system entropy → different maps each run); pass an integer for reproducible map selection.

Knobs not on the CLI (lattice spacing, GAT width, PPO clip, etc.): see [§11](#11-currently-hardcoded-knobs-not-on-the-cli).

---

## 6. Recommended training command (v0.3)

```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. python scripts/run_train.py \
    --split train/easy \
    --total-steps 5_000_000 \
    --n-envs 32 --n-agents 2 \
    --comm-range 120 \
    --rollout-len 128 --max-episode-steps 512 \
    --minibatches 4 \
    --lr 3e-4 --ent-coef 0.01 \
    --compile --eval-on-ckpt \
    --seed 0 \
    --out /workspace/MARLauder/runs/run_v03'
```

Smoke run (1 min, just to verify pipeline):

```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. python scripts/run_train.py \
    --split train/easy --total-steps 40000 \
    --n-envs 8 --n-agents 2 \
    --rollout-len 64 --max-episode-steps 128 \
    --out /workspace/MARLauder/runs/smoke'
```

---

## 7. Evaluation commands

### Batch eval on N random maps — `eval_final.py`

```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. python scripts/eval_final.py \
    /workspace/MARLauder/runs/run_v03/final.pt \
    --split train/easy --n-maps 5 --steps 512'
```

| Flag | Default | Meaning |
|---|---|---|
| `<ckpt>` (positional) | — | Path to `final.pt` or `ckpt_*.pt` |
| `--split` | `train/easy` | Map split to eval on |
| `--n-maps` | `5` | Number of random maps to eval |
| `--steps` | `512` | Max episode steps |
| `--seed` | `None` | RNG for map sampling. `None` → system entropy (random each run). Pass an integer for reproducibility |
| `--out` | ckpt dir | Output dir for GIFs (defaults next to checkpoint) |
| `--device` | `cuda:0` | Device |

Architecture (`n_agents`, `d_hidden`, `n_heads`, `n_layers`) inferred automatically from checkpoint weights. Handles `torch.compile` checkpoints (strips `encoder._orig_mod.` key prefix). Outputs `eval_map{idx:05d}.gif` per map + summary stats (mean/std/min/max explored).

### Single map with explicit architecture — `run_eval.py`

```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. python scripts/run_eval.py \
    --ckpt /workspace/MARLauder/runs/run_v03/final.pt \
    --split test/complex --map-idx 0 \
    --n-agents 2 \
    --d-hidden 128 --n-heads 4 --n-layers 2 \
    --steps 512 \
    --out /workspace/MARLauder/runs/run_v03/eval_ckpt_100.gif'
```

Use when you need to override architecture or pick a specific map index.

### Eval rendering

Each frame is a horizontal stack of **M panels** (one per agent). Each panel shows:

- That agent's personal occupancy map (sigmoid of log-odds).
- That agent's own frontier (red tint).
- Lattice graph: nodes colored cyan→orange by utility, current-node yellow ring.
- Guidepost target (amber ring) + shortest-path polyline (amber).
- The agent itself (filled dot in its assigned color from `C_AGENTS`) + trail.
- All other agents (smaller dots in their own colors) as "ghosts" on this agent's panel.
- Green line between agents whenever both are within `comm_range_px` AND have LOS clear.
- Top-left text bar: `[A0] t=N explored=X.X%` (per-agent label + global step + team explored fraction).

### Random-policy baseline (sanity)

```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. python scripts/baseline_random.py \
    --split test/complex --map-idx 0 --steps 512 --episodes 16 --nr 16'
```

Trained policy on `train/easy` should beat the random baseline by ≥ 2× to count as "learning".

---

## 8. Diagnostics — what good and bad training look like

Per-iter log line:

```
[it   N/T] explored avg=XX.X% end=YY.Y%  pg=±0.0NNN  v=N.NNNN  ent=N.NNN  kl=±0.0NNN  clip=N.N%  sps=NNN(NNNavg) coll=NNNN upd=NNN
```

| Metric | Healthy | Warning sign |
|---|---|---|
| `explored end` | grows over iters | flat near random (~10%) after 100+ iters |
| `pg_loss` | small negative (-0.005..-0.02) | always positive, or huge swings |
| `v_loss` | drops then plateaus around 0.3-0.8 | climbing, or stuck at 1.0 (normalizer not adapting) |
| `entropy` | drops smoothly 2.0 → ~1.0 | crashes to ~0 within 5 iters (policy collapsed) |
| `kl` | < 0.02 | > 0.1 (clip ineffective, too many epochs) |
| `clip` | 5-20% | > 50% (lr too high) or 0% (lr too low) |
| `coll sps` | flat across run | dropping (memory pressure, graph blow-up) |
| `upd sps` | flat | dropping (recompile, oom) |

Random baseline final explored on `train/easy` map 7 (96 high-level steps,
NR=16, n-rays=720, sensor-range=60): **5.95% ± 2.2%**. A trained policy should
beat this by ≥ 2× to count as "learning".

**v0.3 note**: the reward now includes a constant step penalty and a one-shot
completion bonus. `pg_loss` magnitude is similar to v0.2, but the per-step
return baseline shifts (mean reward per step is lower due to the penalty;
episode-total return jumps at completion). If you see `v_loss` plateauing higher
than v0.2, that is expected — the value function now has to predict the
completion bonus.

---

## 9. Architecture summary

```
                                  ┌──────────────────────────┐
                                  │  Warp LiDAR (GPU)        │   n-rays per agent
                                  │  PER-AGENT log-odds      │   occupancy[N,M,H,W]
                                  └────────┬─────────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │  _comm_check (range + LOS)       │
                          │  fuse_maps (max log-odds)        │   comm_mask[N,M,M]
                          │  update last_known_pos           │
                          └────────────────┬────────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │  per-agent loop:                 │
                          │   frontier (torch conv2d)        │
                          │   graph_lattice.build            │
                          │   graph_lattice.build_guidepost  │
                          └─────────────┬───────────────────┘
                                        │ node_feat[N,M,N_max,7], edge_idx, masks,
                                        │ guidepost_target, guidepost_path_xy,
                                        │ guidepost_nbr_bias, last_known_pos
                                        ▼
   ┌─────────────────  Shared Encoder (per (env, agent))  ──────────────────┐
   │   Linear(7 → d)                                                         │
   │   MaskedGATLayer × n_layers  (heads=4, residual + LayerNorm + GELU)     │
   │   h_all [N·M, N_max, d]                                                 │
   │   gather curr_emb [N·M, d]   gather nbr_embs [N·M, K=8, d]              │
   └───────────────┬─────────────────────────────────────┬──────────────────┘
                   │                                     │
        decentralized actor                       centralized critic (CTDE)
                   │                                     │
   ┌─── per agent ─▼───────────────┐    ┌───── per env ─▼─────────────────┐
   │ GRUCell(curr_emb, h_act_prev) │    │ concat curr_emb over M agents    │
   │ PointerHead(query, nbr_embs,  │    │   → Linear(M·d → d) + GELU       │
   │   action_mask) → logits[K=8]  │    │ GRUCell(joint, h_crit_prev)      │
   │ Categorical sample → action   │    │ Linear(d → d/2) → GELU           │
   └───────────────────────────────┘    │ Linear(d/2 → 1)  V(s) ∈ ℝ         │
                   │                    └──────────────────────────────────┘
                   ▼
              env.step(action)  →  reward / done / next obs
                   │
                   ▼  buffer.store
            ┌──────────────────┐
            │ MAPPO update     │
            │  GAE-λ team mean │
            │  PPO clip ε=0.2  │
            │  TBPTT chunks=16 │
            │  AMP fp16        │
            │  Welford V-norm  │
            │  encode_chunk    │
            │   ─ one enc/chunk│
            │  n_minibatches   │
            │   ─ over N axis  │
            └──────────────────┘
```

Invariants:
- Encoder weights shared actor↔critic — both gradients flow back.
- Actor decentralized: each agent sees only its own padded graph + curr.
- Critic centralized: concatenates per-agent curr_emb → joint state V(s).
- Hidden states zeroed at episode resets via `(1 - done)` mask.
- All obs tensors live on GPU; no host roundtrips during rollout.
- `N_max = (H/NR)·(W/NR)` fixed → pure pad+mask, no PyG dynamic batching.
- Edge length: axial `NR`, diagonal `NR·√2` — used in Bellman-Ford guidepost.
- v0.2: encoder called ONCE per TBPTT chunk, not once per timestep.

---

## 10. Roadmap

| Ver | Goal | Status |
|---|---|---|
| v0.1 | Single-agent baseline (Warp LiDAR + lattice graph + GAT + MAPPO) | ✓ |
| v0.2 | Terminology rename, Bellman-Ford guidepost, diagonal cost, MAPPO speedup, docs | ✓ |
| v0.3 | Multi-agent intermittent comm, per-agent maps, per-agent eval render, step-penalty + completion-bonus reward, O(1) curr_idx, FREE-only start placement, random-seed eval | ✓ (current) |
| v0.4 | Target diversification (top-K + clustering + Hungarian), warm-start Bellman-Ford, ego-centric subgraph encoder | planned |
| v0.5 | Curriculum train/easy → train/difficult | planned |
| v0.6 | Eval suite: per-split curves, TB logger, milestone GIF auto-generation, rendezvous reward | planned |
| v0.7 | ToM `teammate_belief.py` module (probabilistic teammate-state estimator + belief merge on rendezvous) | planned |

v0.8 (hierarchical L2 graph) explicitly **out of scope** for this rewrite.

See [dev_log.md](dev_log.md) for design-decision context behind each version.

---

## 11. Currently hardcoded (knobs not on the CLI)

These defaults live in dataclass definitions and are not exposed as CLI flags. Edit the dataclass to change them.

### `env.explorer.EnvCfg`

| Name | Default | Effect |
|---|---|---|
| `nr` | `16` | Lattice spacing (px). Smaller → denser graph (more nodes, more memory). N_max scales as `(H/nr)·(W/nr)` |
| `sensor_range_px` | `60.0` | LiDAR range (px) |
| `n_rays` | `720` | LiDAR ray count per scan |
| `utility_range_px` | `30` | Half-window size (px) for frontier-density utility integral |
| `num_sim_steps` | `5` | LiDAR sub-steps per high-level step (collision check granularity) |
| `flood_max_iters` | `200` | Max iterations of flood-fill in `node_valid` |
| `done_explored_thresh` | `0.99` | Episode `terminated` threshold (fraction of GT-free seen) |
| `comm_los_samples` | `40` | Number of Bresenham samples for line-of-sight comm check |
| `step_penalty_coef` | `0.1` | Total step penalty budget per episode (penalty = coef / max_episode_steps) |
| `completion_bonus` | `10.0` | One-shot reward at terminal step when `terminated` |

### `train.driver.TrainCfg`

| Name | Default | Effect |
|---|---|---|
| `d_hidden` | `128` | Encoder + GRU hidden width |
| `n_heads` | `4` | GAT attention heads (must divide `d_hidden`) |
| `n_layers` | `2` | GAT layers (receptive field = `n_layers` hops) |
| `lr_actor` | `3e-4` | Adam LR for actor (overridden by `--lr` CLI) |
| `lr_critic` | `1e-3` | Adam LR for critic |
| `eval_steps` | `256` | Episode length for `--eval-on-ckpt` GIFs |
| `eval_n_maps` | `2` | Number of eval GIFs emitted per milestone |
| `eval_map_idx` | `-1` | `-1` = random map each time. Set to a non-negative int to fix |

### `train.mappo.MAPPOCfg`

| Name | Default | Effect |
|---|---|---|
| `clip_eps` | `0.2` | PPO clip ε |
| `vf_coef` | `0.5` | Value loss weight |
| `ent_coef` | `0.01` | Entropy bonus weight (overridden by `--ent-coef` CLI) |
| `k_epochs` | `4` | PPO epochs per rollout. Reduce to 2 if KL > 0.02 |
| `tbptt_steps` | `16` | TBPTT chunk length for hidden-state truncation |
| `n_minibatches` | `1` | PPO minibatches per epoch (overridden by `--minibatches` CLI). Must divide `n-envs` |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE λ |
| `grad_clip` | `0.5` | Global gradient clip norm |

To change any of these, edit the dataclass directly. There is no in-place override — values are stamped into the checkpoint's `cfg` field at save time.
