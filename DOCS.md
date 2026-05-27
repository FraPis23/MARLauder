# MARLauder — Module Map, Parameters, Commands

GPU-vectorized graph MAPPO for cooperative exploration. Single agent baseline,
N-agent ready. v0.2 (post terminology rename, guidepost, render upgrade, MAPPO
speedup).

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
| `world_warp.py` | GPU LiDAR via NVIDIA Warp. Maintains `occupancy_torch` (uint8: 0=UNKNOWN, 1=FREE, 2=OBSTACLE) and `occupancy_logodds_torch` (f32 Bayesian log-odds). Categorical derived by threshold. `occupancy_prob()` → sigmoid for render. |
| `maps.py` | Load preprocessed `data/<split>/maps.npy` + `meta.npz`. Sample N maps to GPU. |
| `frontier.py` | Torch conv2d frontier detector. `compute_frontier(occupancy)` → bool [N,H,W]. Frontier = FREE cell with 2..7 UNKNOWN neighbors. |
| `graph_lattice.py` | Core graph manager. Builds 8-neighbor lattice on free cells, reachability flood-fill, collision-checked edges, integral-image utility, and **Bellman-Ford guidepost** (Dijkstra path to nearest high-utility node, edge weights = `NR` axial / `NR·√2` diagonal). |
| `explorer.py` | Vectorized environment. Owns occupancy + position + visited state. `step(action)` decodes K=8 pointer action, follows path with sub-step LiDAR, computes reward (Δ FREE union / total free), refreshes obs. |
| `teammate_belief.py` | Stub for v0.4+ ToM teammate-state estimator. Unused in v0.2. |

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
| `render.py` | Palette + painters: `shade_occupancy_prob` (continuous), `paint_frontier` (soft red tint), `paint_graph` (nodes colored by utility), `paint_path` (amber polyline = guidepost), `paint_target` (amber ring), `paint_agent` (blue + trail), `composite_frame` orchestrator. |
| `rollout.py` | `EvalRollout`: deterministic single-episode play, collects frames + stats. Pulls guidepost target + path from obs each step. |

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
| `run_eval.py` | Load ckpt → deterministic episode → GIF. |

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
| 5 | `prob_occupied` | 1 if **another** agent's nearest node = this node (M>1 only) | {0, 1} |
| 6 | `guidepost` | 1 if node lies on Bellman-Ford shortest path from curr to nearest high-utility reachable node | {0, 1} |

Invalid nodes have feature row zeroed. Edges to invalid neighbors masked in GAT attention.

Utility computed via integral image of the frontier mask (one prefix sum + 4 corner gathers per node), square window approximation of disk, edge-clipped (border nodes underestimate — acceptable).

---

## 4. Reward

```
union_free_t  = |{cell : occupancy_t[cell] == FREE}|            # shared per env
team_reward_t = max(0, (union_free_t − union_free_{t−1}) / gt_total_free)
reward[e, a]  = team_reward_t                                    # broadcast to all M agents
```

- Single scalar per env per step. Critic predicts V(s) global.
- Non-negative: no penalty for revisiting cells.
- Bounded ≤ 1 cumulative per episode.
- M=1: equals "Δ own occupancy free / total free".
- M>1: dual-agent on same zone → no double-count, incentive to diverge.
- Episode done if `explored_rate ≥ done_explored_thresh` or `t ≥ max_episode_steps`.

---

## 5. Training parameters (CLI flags of `scripts/run_train.py`)

| Flag | Default | Range / Note |
|---|---|---|
| `--split` | `train/easy` | `train/easy`, `train/difficult`, `test/{complex,corridor,hybrid}` |
| `--out` | `runs/run_default` | Output dir. Ckpts at `ckpt_{025,050,075,100}.pt` + `final.pt` |
| `--total-steps` | 5_000_000 | Total env transitions (`n_envs * rollout_len * n_iters`) |
| `--n-envs` | 16 | Parallel envs. Must be divisible by `--minibatches`. |
| `--n-agents` | 1 | M. Use 1 for v0.2 baseline. |
| `--rollout-len` | 128 | T per PPO update. |
| `--max-episode-steps` | 128 | Episode truncation. ≤ rollout-len recommended. |
| `--nr` | 16 | Lattice spacing (px). Smaller = denser graph + more memory. |
| `--sensor-range` | 60 | LiDAR range (px). |
| `--d-hidden` | 128 | Encoder + GRU hidden width. |
| `--n-heads` | 4 | GAT attention heads. |
| `--n-layers` | 2 | GAT layers (each with residual+LayerNorm+GELU). |
| `--lr` | 3e-4 | Adam learning rate (shared actor+critic). |
| `--clip-eps` | 0.2 | PPO clip ε. |
| `--ent-coef` | 0.01 | Entropy bonus weight. |
| `--k-epochs` | 4 | PPO epochs per rollout. 2 OK if KL < 0.01. |
| `--tbptt-steps` | 16 | TBPTT chunk length. |
| `--minibatches` | 1 | PPO minibatches per epoch. Must divide n_envs. 4 good at n_envs=32. |
| `--compile` | off | torch.compile the encoder. ~2.3× update speedup. |
| `--seed` | 0 | RNG seed (split sampling + policy). |
| `--device` | `cuda:0` | Or `cpu` (no AMP, slow). |
| `--no-amp` | off | Disable fp16 autocast + GradScaler. |

---

## 6. Recommended training command (v0.2)

```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. python scripts/run_train.py \
    --split train/easy \
    --total-steps 5_000_000 \
    --n-envs 32 \
    --rollout-len 128 \
    --max-episode-steps 128 \
    --nr 16 \
    --sensor-range 60 \
    --d-hidden 128 \
    --n-heads 4 \
    --n-layers 2 \
    --lr 3e-4 \
    --clip-eps 0.2 \
    --ent-coef 0.01 \
    --k-epochs 4 \
    --tbptt-steps 16 \
    --minibatches 4 \
    --compile \
    --seed 0 \
    --out /workspace/MARLauder/runs/run_v02'
```

Smoke run (1 min, just to verify pipeline):

```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. python scripts/run_train.py \
    --split train/easy --total-steps 40000 --n-envs 32 --rollout-len 64 \
    --max-episode-steps 64 --nr 16 --d-hidden 96 --compile --minibatches 4 \
    --out /workspace/MARLauder/runs/smoke'
```

---

## 7. Evaluation command

```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. python scripts/run_eval.py \
    --ckpt /workspace/MARLauder/runs/run_v02/ckpt_100.pt \
    --split test/complex --map-idx 0 \
    --d-hidden 128 --n-heads 4 --n-layers 2 \
    --steps 128 \
    --out /workspace/MARLauder/runs/run_v02/eval_ckpt_100.gif'
```

GIF shows: probabilistic occupancy bg, frontier soft red tint, lattice nodes
(cyan→orange by utility), curr ring (yellow), guidepost target ring + amber
shortest-path polyline, agent dot + trail, step + explored-rate text bar.

Random baseline on same map (sanity):

```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. python scripts/baseline_random.py \
    --split test/complex --map-idx 0 --steps 128 --episodes 16 --nr 16'
```

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

---

## 9. Architecture summary

```
                                  ┌──────────────────────┐
                                  │  Warp LiDAR (GPU)    │   n-rays per agent
                                  │  per-env log-odds    │
                                  └────────┬─────────────┘
                                           │ occupancy[N,H,W] uint8 + log-odds f32
                                           ▼
                          ┌────────────────────────────────┐
                          │  frontier (torch conv2d)        │
                          │  + graph_lattice.build           │
                          │  + graph_lattice.build_guidepost │
                          └─────────────┬──────────────────┘
                                        │ node_feat[N,M,N_max,7], edge_idx, masks,
                                        │ guidepost_target, guidepost_path_xy
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

| Ver | Goal |
|---|---|
| v0.1 ✓ | Single-agent baseline working end-to-end |
| v0.2 ✓ | Rename, guidepost, target+path render, MAPPO speedup, docs |
| v0.3 | Reward shaping experiments, longer training, validation curves |
| v0.4 | M=2 cooperative, validate union reward + prob_occupied feat[5] |
| v0.5 | Curriculum train/easy → train/difficult |
| v0.6 | Eval suite: per-split curves, TB logger, 4-milestone GIFs auto-generated |
| v0.7 | ToM `teammate_belief.py` module (rendezvous-driven belief merge + teammate-state encoder) |

v0.8 (hierarchical L2) explicitly **out of scope** for this rewrite.
