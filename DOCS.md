# MARLauder — Module Map, Parameters, Commands

GPU-vectorized graph MAPPO for cooperative exploration. Multi-agent with
intermittent line-of-sight communication. v0.4 (per-agent occupancy maps,
comm + map fusion, per-agent eval render, **per-agent set-op reward**,
**strategic frontier-attention head** with Gumbel-ST, **ego-centric encoder**,
**BF-from-curr + BF-from-teammate** for true-cost candidate ranking, joint
exploration-distribution feature, learnable+floored path-following bias,
anti-chase signals, ramping curriculum, debug full-sharing flags).

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
| `world_warp.py` | GPU LiDAR via NVIDIA Warp. Maintains **per-agent** `occupancy_torch [N, M, H, W]` (stored flat as `[N·M, H, W]` for Warp's max-3-dim kernel indexing) and `occupancy_logodds_torch`. Exposes `team_occupancy()` (union across agents) and `fuse_maps(comm_mask)` (elementwise **max-magnitude** log-odds where two agents are in comm — keeps OBSTACLE evidence). **Self-cell FREE invariant** (`_mark_pos_free`): the lidar loop starts at t=1.0 so it never marks the robot's own cell; the kernel stamps a 3×3 footprint at `2·LO_FREE` so the cell clears the strict `v > LO_FREE_TH` test. **Bug history (2026-06-11):** a single `LO_FREE` add landed at exactly the threshold → origin stayed UNKNOWN → the agent's current graph node was invalid (`node_valid` floods FROM the robot cell) → 0 legal moves → every action hit the invalid-action fallback and **teleported**. Coverage still hit ~92% via teleport-bouncing, masking it. ALL pre-fix sweeps (`uelh3hs3`, `odft9txk`, `wyhlx0ki`) are invalid. |
| `maps.py` | Load preprocessed `data/<split>/maps.npy` + `meta.npz`; sample N maps to GPU. `MultiSplit` wrapper (weighted union of splits, used by curriculum). `sample_batch` accepts `Split` or `MultiSplit`. |
| `frontier.py` | Torch conv2d frontier detector. `compute_frontier(occupancy)` → bool [N,H,W]. Frontier = FREE cell with 2..7 UNKNOWN neighbors. |
| `graph_lattice.py` | Core graph manager. 8-neighbor lattice on free cells, reachability flood-fill, collision-checked edges, integral-image utility. `bf_from_target(info, target, dist_init)` — overwrite-mode warm-startable Bellman-Ford from any source node (used for guidepost target, BF-from-curr, BF-from-teammate). `extract_topk_candidates(util, valid, curr_xy, K, bf_dist)` — top-K frontier candidates with BF distances. `build_guidepost_v2`. `curr_idx` from O(1) floor-divide. |
| `explorer.py` | Vectorized environment. Per-agent occupancy + positions + `last_known_pos[N,M,M,2]` + `visited_step` + reward-baseline caches (`last_meeting_node_mask`, `last_own_free_node`) + BF warm-start caches (`_dist_curr_prev`, `_dist_team_prev`). `step(action)`: sub-step LiDAR move, wall revert + **asymmetric agent-agent collision** (lower-priority agent yields via per-episode `_collision_key`; winner advances, reverts too only if still blocked), `_comm_check` (Euclidean range + Bresenham LOS), `fuse_maps`, **per-agent set-op reward** (incl. objective second-guessing penalty via `_prev_target_node` + `target_choice`), `_refresh_obs`. `_refresh_obs` builds per-agent graph + BF-from-curr + BF-from-teammate + top-K candidates + strategic features. `reload_map(env_idx, map_idx)` does a full reset for eval. `_spread_starts_graph` places M agents on adjacent FREE lattice nodes (BFS + segment-clear). `EnvCfg.from_ckpt_dict(d, **overrides)` rebuilds cfg from a saved checkpoint dict. |
| `teammate_belief.py` | Stub for v0.7 ToM teammate-state estimator. Unused. |

### models/

| File | Purpose |
|---|---|
| `gat.py` | `MaskedGATLayer` + `GATEncoder`. Multi-head attention over K=8 padded neighbors. Pure torch — no PyG. Accepts any leading batch dim. Runs on the ego-centric window `(2·n_hops+3)²`, not the full lattice. |
| `actor_critic.py` | `MarlActorCritic`. Shared ego-centric GAT encoder; `StrategicHead` (MHA over top-K candidates → Gumbel-ST pick → `strategic_emb` + `target_idx`); actor (`actor_pre` over `[curr_emb ‖ strategic_emb ‖ next_hop_onehot ‖ prev_action]` → GRUCell → PointerHead over K=8 neighbors, with finite-masked logits + NaN guard); learnable+floored `path_bias` adds a soft prior on the BF first-hop of the strategic pick; critic (CTDE: concat per-agent curr_emb → MLP → GRUCell → V scalar). `encode_chunk()` batches encoder across T for MAPPO speedup. |
| `value_normalizer.py` | Welford online mean/var. Critic predicts normalized V, GAE uses denormalized. |

### train/

| File | Purpose |
|---|---|
| `buffer.py` | Pre-allocated rollout `[T, N, M, ...]` for obs (incl. `cand_*`, `prev_action`, `cand_bf_first_hop`), action, `target_choice`, logp, value, reward, done. **Per-agent** GAE-λ with a shared CTDE value baseline (`compute_gae` → advantages `[T,N,M]`, returns `[T,N]` team-mean for the V target). |
| `mappo.py` | PPO update. One encoder call per TBPTT chunk (`encode_chunk`), then GRU re-roll per timestep replaying the stored strategic pick (`target_choice`) through `evaluate_step_from_enc`. Per-agent advantages. PPO clip, value MSE, entropy bonus. Minibatching over N. AMP fp16. Single grad clipper + scaler. |
| `driver.py` | Main loop. `TrainCfg` defaults, rollout collection, ppo_update, milestones (25/50/75/100 %), throughput logging, optional `torch.compile`, optional ramping curriculum (`_curriculum_weights`), `_emit_eval_gif` (uses `reload_map` + `EnvCfg.from_ckpt_dict`). Logs `ep_end` (mean explored at terminal step over episodes that ended this iter). |

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
| `run_eval.py` | Load ckpt → deterministic episode on one map → GIF. Reads FULL env cfg from ckpt (`n_hops`, `top_k`, force flags) via `EnvCfg.from_ckpt_dict`; `--force-full-*` CLI override. Pass `--d-hidden`/`--n-heads`/`--n-layers` to match net. |
| `eval_final.py` | Batch eval on N random maps (or `--map-idx`). Infers architecture from checkpoint weights; reads env cfg from ckpt. Strips `encoder._orig_mod.` from torch.compile checkpoints; remaps legacy `path_bias`→`path_bias_learn`. |
| `debug_spawn.py` | Audit spawn adjacency over N maps (reports % non-adjacent, min/max/mean agent distance). |

---

## 2. Data flow (one rollout iteration)

```
data/<split>/maps.npy  (memmap, uint8 [N, H, W])
   │
   ▼  env.maps.sample_batch  (N maps to GPU)
WarpWorld.gt_torch  +  WarpWorld.occupancy_logodds_torch  +  WarpWorld.occupancy_torch
   │
   │ env.step(action) for t in [0, T):
   │   1. decode action (K=8 slot) via curr_nbr_global → target node world coord
   │   2. path-follow K_sub sub-steps:  Warp LiDAR per sub-step
   │   3. _comm_check + fuse_maps + update last_known_pos (force flags may override mask)
   │   4. per-agent set-op reward (scan/team/give/recv/overlap/revisit/proximity/...)
   ▼
   │ _refresh_obs per agent:
   │   compute_frontier(occupancy)                 (torch conv2d)
   │   GraphLattice.build()                        (flood-fill + collision + utility)
   │   bf_from_target(curr)  → bf_dist_from_curr   (warm-started)
   │   bf_from_target(teammate lkp, edge_valid=edge_valid_optim) → bf_dist_team (per teammate)
   │   extract_topk_candidates(util, valid, curr_xy, K=16, bf_dist)
   │   build cand_feat[N,M,K,8] + cand_bf_first_hop + ego-centric window
   ▼
obs dict [N, M, ...] → MarlActorCritic.act(obs, h_act, h_crit)
                          ├── ego-centric GAT encoder → curr_emb, nbr_embs
                          ├── StrategicHead(curr_emb, cand_feat) → Gumbel-ST target pick
                          ├── actor_pre([curr_emb‖strategic_emb‖next_hop‖prev_action])
                          │     → GRU → PointerHead (+ path_bias·first_hop) → action
                          └── CTDE critic GRU → V(s)
   │
   ▼ buffer.store(t, obs, action, target_choice, logp, value, reward, done)
   │
   │ after T steps, compute_gae → per-agent advantages [T,N,M], team-mean returns [T,N]
   │
   ▼ MAPPO update (k_epochs × n_minibatches × T/tbptt_steps chunks)
        │
        ├── encode_chunk(chunk_obs)  ← ONE pass per chunk
        ├── for tt: replay stored target_choice → strategic STE → GRU + pointer + critic
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

Invalid nodes have feature row zeroed. Edges to invalid neighbors masked in GAT attention. The GAT encoder runs on the ego-centric window `(2·n_hops+3)²` centered on `curr`, not the full lattice.

**Utility (v2, wall-aware)**: the old (2·UR+1)² raw integral window counted frontier pixels
**through walls** (a node beside a wall scored frontier in the corridor on the other side,
poisoning the GAT feature, the strategic candidate ranking and the guidepost argmax). Now:
(a) per-node frontier indicator from a tiny ~NR/2-px window (integral image, leak negligible),
(b) h=⌈UR/NR⌉=2 rounds of graph diffusion of that indicator along **collision-checked
`edge_valid` edges** — frontier mass only flows through passable edges, so walls block it by
construction. Normalized to [0,1] by 2^h; typical max ≈ 0.1.

### 3.1 Strategic candidate features (CAND_FEAT_DIM = 8)

Separate from the GAT node features above. For each agent, the top-K (`--top-k`, default 16) reachable frontier candidates are extracted globally (not windowed) and fed to the `StrategicHead`. Per-candidate feature vector:

| Idx | Name | Meaning |
|---|---|---|
| 0,1 | `rel_xy` | candidate world position − agent position, / canvas_diag |
| 2 | `utility` | candidate frontier-density utility ∈ [0, 1] |
| 3 | `bf_dist` | BF shortest-path distance from curr to candidate (wall-aware), / canvas_diag |
| 4 | `min_team_bf_dist` | min over teammates of BF dist FROM teammate to candidate (in my map), / canvas_diag |
| 5 | `max_comm_gap` | steps since last comm with the most-stale teammate, / max_episode_steps |
| 6 | `own_minus_team` | (my bf_dist − min teammate dist) × `--yield-scale`, clamped [-1,1]. Positive = teammate closer → yield. |
| 7 | `team_alt_score` | mean over teammates of "teammate's best alternative − this cand's value" (H.2 joint distribution). High = teammate has other good options → I can take this. |

The head outputs `target_logits[K]` (Gumbel-ST → one-hot pick) and a pooled `strategic_emb`. `cand_bf_first_hop[K, 8]` one-hot maps the chosen candidate to its first K=8 lattice hop (for the `path_bias` action prior).

---

## 4. Reward (v0.4 — per-agent set-op formulation)

Each agent receives an **independent** scalar reward. Set ops are on the lattice (N_max ≈ 1200 nodes per env), baselined at the last comm event between each pair.

```
# Per step, per agent a:
scan_self_delta[a]   = (#FREE nodes I LiDAR-scanned this step) / N_max        # post-scan, pre-fusion
team_delta           = Δ(union FREE pixels) / total_free                       # cooperation anchor

# Per pair (i, j) gated by comm_mask[i, j] (rendezvous event):
B_ij = M_i ∧ ¬last_meeting_node_mask[i, j]    # cells i scanned since last meeting with j
B_ji = M_j ∧ ¬last_meeting_node_mask[i, j]
give[i]    = |B_ij ∧ ¬M_j| / N_max             # NEW cells I bring to j (j doesn't have)
recv[i]    = |B_ji ∧ ¬M_i| / N_max             # NEW cells I receive from j
overlap[i] = |B_ij ∧ B_ji|  / N_max             # we BOTH scanned same area since last meeting

# Anti-loop / anti-chase / anti-stall:
revisit_pen[a]   = (W − age)/W  if chosen node visited within last W steps (graduated by recency)
proximity_pen[a] = 1 if teammate within sensor_range AND visible (comm)
stall_pen[a]     = 1 if ‖pos_after − pos_before‖ < nr·0.5  (no net displacement this step)

# Objective second-guessing (graph-tree, B+D):
target_switch_pen[a] = 1 if  branch(g_t) ≠ branch(g_{t-1})  AND  g_{t-1} still pursuable
                       branch(g) = first-hop slot off curr toward g in the BF-from-curr tree
                       pursuable = reachable (bf_dist finite) AND not reached (bf_dist > 1.5·nr)

# v2 — privileged novel-scan credit (IR2-style r_f). union = team union FREE-node mask:
novel_scan[a] = |cells a scanned this step ∧ ¬union_prev| / scan_norm_nodes(=50)
team_delta_node = |union_now ∧ ¬union_prev| / scan_norm_nodes

# Final reward (v2):
reward[a] = α_novel · novel_scan[a]
          + β     · team_delta_node
          + ζ_give · Σ_j give[a]
          + ζ_recv · Σ_j recv[a]
          − η_lap  · Σ_j overlap[a]
          − γ      · revisit_pen[a]
          − ε_prox · proximity_pen[a]
          − δ_obj  · target_switch_pen[a]
          − δ_stall· stall_pen[a]
          + 1{terminated} · completion_bonus
          − step_penalty
```

Defaults: `α_novel=1.0`, `β=0.3`, `ζ_give=1.5`, `ζ_recv=0.5`, `η_lap=3.0`, `γ=0.05`, `ε_prox=0.05`, `δ_obj=0.01`, `δ_stall=0.1`, `W=8`, `completion_bonus=10.0`, `step_penalty_coef=0.1`.

**v2 novel-scan credit**: pays only cells **new to the team union** — a follower scanning a
leader's wake earns 0, so splitting up is the highest-paying policy by construction (the old
`scan_self + shared team_delta` paid the follower a β commission on the leader's work →
chasing was rational). Privileged (training-only, CTDE; deployed actor never sees the union).
`scan_self_delta` remains as the logged diagnostic `reward/scan_self_diag`. Both-scan-same-cell
ties credit both agents (simultaneous discovery). Spawn scans are baseline, not credited.
Relationship to give/recv/overlap: those are the **rendezvous accounting** (fire at comm
events, baselined at last meeting); novel-scan is **instantaneous credit at scan time** —
complementary, and under `--force-full-occupancy-sharing` (per-step fusion) the rendezvous
terms degenerate to ≈0 leaving novel-scan as the only per-agent channel (exploited by the
Stage-1 sweep, §12). Dense normalization `scan_norm_nodes=50` (≈ one sensor disk) replaces
/N_max≈1200 so shaping is O(0.1), not O(0.001), vs the completion bonus.

**v2 target_switch**: env receives the strategic head's **argmax intent** (`target_argmax`
from `model.act`), not the Gumbel-sampled `target_choice` — sampling noise no longer counts
as second-guessing. Default coef 0.05→0.01 (sweep v1: sampled variant dominated the dense
reward 10–50×, `reward/target_switch` = −0.007…−0.054/step vs scan +0.0004…+0.0018).

**Stall penalty (anti-standing-still)**: physical no-progress detector — snapshot `pos` at the
top of `step()`, compare after the sub-step loop; `‖Δpos‖ < nr·0.5` → `stall_pen=1`. Catches
BOTH collision-revert holds (asymmetric-collision loser) and invalid/curr-node picks. Heavily
weighted (`δ_stall=0.1` ≫ old revisit) to break chase/standoff deadlocks and push agents to
reroute / separate. `revisit_pen` is now **graduated** (`(W−age)/W`) so tighter loops cost
more; `ε_prox` raised 0.005→0.05. These plus `δ_stall` are the primary anti-degenerate knobs
the W&B sweep tunes (§12).

**Objective second-guessing penalty (B+D, graph-tree)**: treats the BF-from-curr parent tree (`bf_parent_from_curr`) as the exploration tree. The strategic target `g_t` lives in some first-hop branch off the current node; `branch(g)` = walk `bf_parent` from `g` back to a curr-neighbor. Penalty fires only when the committed branch **flips** AND the previous target was still **reachable + unreached** (D gate via cached `bf_dist_from_curr`). A target that shifts *forward along the same branch* (receding frontier) keeps the same first-hop → **0 penalty**, regardless of how far it jumped — the term keys on graph *direction*, not node identity. Reaching / invalidating the old target opens free re-selection. Computed in the PRE-step frame (tree rooted where the decision was made), from `self._last_obs` + `self._prev_target_node`. Requires `target_choice` plumbed from `model.act` → `env.step`; eval/baseline pass nothing → term off. Agent-local → decentralized, real-robot-safe.

**Why per-agent**:
- Phase A v2's strategic head needs per-agent gradient signal to differentiate yielding.
- Set-op decomposition gives the policy credit for the right behavior: scan-self, bring info, receive info, avoid overlap, avoid backtrack, avoid chasing.

**Decentralization**: each term computed from agent-local state (own occupancy, own visited, own last_known_pos) or via comm-gated set ops on rendezvous events.

**Optimistic teammate-distance graph** (`edge_valid_optim`, built in `graph_lattice.build()` when M>1): BF-from-teammate is rooted at the teammate's `last_known_pos` in the agent's OWN map. Once agents split, the teammate usually sits in the agent's UNKNOWN region; on the FREE graph (`node_valid` = `occ==FREE` ∧ reachable) that node is invalid/disconnected, so `bf_from_target` returns `+inf` for every candidate and the `team_alt_score` / `cand_own_minus_team` coordination signals go silent — exactly when map-sharing is off. Fix: a second edge graph flooded through FREE∪UNKNOWN (`occ != OBSTACLE`) from the robot cell, reusing the same `!= OBSTACLE` collision check. The teammate BF passes this graph via `bf_from_target(..., edge_valid=info["edge_valid_optim"])`. Result: exact geodesic through known-free space, ≈Euclidean through unknown, and `+inf` ONLY when a KNOWN wall separates the pair (pure Euclidean would lie there). Used for the teammate-distance heuristic only; the agent's own navigation BF (`bf_from_curr`, target pick) stays on the FREE graph. `bf_from_target` reads geometric edge indices (`edge_idx_static`) masked by the active `edge_valid` so both graphs resolve neighbours correctly.

**Last-meeting baseline**: `last_meeting_node_mask[i, j]` snapshots the post-fusion union at the most recent comm between i and j. Set ops are over "new scans since last meeting", so initial co-spawn overlap doesn't keep firing.

**v0.4 anti-chase signals**:
- `overlap` penalty fires at every comm event with overlap (default `η_lap=3.0`).
- ~~`proximity` penalty~~ **ELIMINATED (default `ε_prox=0.0`, 2026-06-12)**: the raw-distance
  per-step reflex was the **stalemate/ping-pong driver** — it over-corrected into back-and-forth
  and punished BOTH agents converging on the last frontier (deadlock). `novel_scan` already pays
  0 for team-known cells, so anti-chase is covered without it. Flag kept for ablation; optional
  fallback is a *productivity gate* (`*= novel_scan<=0`) so only freeloading proximity is penalized.
- **`target_switch_pen` raised 0.01→0.05** (B+D graph-tree branch-flip): commitment to a
  direction; now on argmax intent so safe to strengthen → punishes uncommitted back-and-forth.
- **Target-claim at rendezvous** (`last_known_target`, comm-gated): when two agents are in comm,
  the **lowest agent-ID keeps its target**; a higher-ID agent whose candidate set contains a
  lower-ID teammate's claimed target has that candidate **masked** (`cand_valid→False`) so it
  diverts — UNLESS it is the agent's only option (**single-frontier guard**: both commit, never
  back down). Decentralized (only acts in comm range); out-of-comm division → Phase 2 surplus obs.
- `cand_own_minus_team` feature (amplified by `--yield-scale 3.0`) → "yield to closer agent".
- `team_alt_score` feature (H.2) → "take a frontier the teammate has good alternatives for".
- `path_bias` (fixed floor `--path-bias-floor 1.5` + learnable extra) keeps the actor following the strategic pick's BF first-hop so grid-utility doesn't fully dominate.

**Debug full-sharing flags** (training-only sanity, NOT for deployment):
- `--force-full-pos-sharing` — teammate positions always fresh (decouples lkp from comm).
- `--force-full-occupancy-sharing` — maps fused every step regardless of comm range.
Both are saved in the checkpoint cfg and propagated to eval so renders reflect training behavior.

**v0.3 → v0.4 migration**: the old single-scalar team reward is replaced by the per-agent set-op reward. Per-agent advantages flow through GAE against a shared CTDE value baseline.

---

## 5. Training parameters (CLI flags of `scripts/run_train.py`)

| Flag | Default | Range / Note |
|---|---|---|
| `--split` | `train/easy` | `train/easy`, `train/difficult`, `test/{complex,corridor,hybrid}` |
| `--out` | `runs/run_default` | Output dir. Ckpts at `ckpt_{025,050,075,100}.pt` + `final.pt` (all carry `cfg`). With `--eval-on-ckpt` also `eval_ckpt_{pct}_m{0,1}.gif` on random maps |
| `--seed` | `-1` | torch RNG (action sampling, init). `-1` = time-based. Map sampling RNG is independent (fresh entropy) so maps differ each run |
| `--device` | `cuda:0` | Or `cpu` (CPU is slow; AMP/Warp disabled) |
| `--total-steps` | `5_000_000` | Total env transitions (`n_envs × rollout_len × iters`) |
| `--n-envs` | `16` | Parallel envs. Must be divisible by `--minibatches` |
| `--n-agents` | `1` | Number of cooperative agents per env |
| `--comm-range` | `120.0` | Communication range (px). Set 0 to disable comm entirely |
| `--rollout-len` | `128` | T per PPO update |
| `--max-episode-steps` | `512` | Episode truncation (typically ≥ rollout-len) |
| `--minibatches` | `1` | PPO minibatches per epoch. Must divide `n-envs`. **Keep at 1** (max 2): MAPPO paper Suggestion 3 (Fig.5b) shows 4 minibatches fails to solve maps while 1 is best on 22/23 — avoid splitting the batch |
| `--lr` | `3e-4` | Adam learning rate for actor |
| `--ent-coef` | `0.01` | Entropy bonus weight |
| `--compile` | off | `torch.compile` the encoder (~2× update speedup) |
| `--eval-on-ckpt` | off | Emit 2 eval GIFs on random maps at each milestone (25/50/75/100%) |
| `--eval-steps` | `-1` | Episode length for eval-on-ckpt GIFs. `-1` aligns with `--max-episode-steps` (G.2) |
| `--n-hops` | `2` | Ego-centric encoder window radius. Window = (2·n_hops+3)². n_layers tied to this |
| `--top-k` | `16` | Top-K frontier candidates per agent for strategic attention head (Phase A v2) |
| `--force-full-comm` | off | Debug: bypass dist/LOS check; every pair communicates every step |
| `--force-full-pos-sharing` | off | Debug: persistent teammate-position awareness (positions only, maps still comm-gated) |
| `--force-full-occupancy-sharing` | off | Debug: maps fused every step (occupancy synced across agents) |
| `--curriculum` | off | Ramp train/easy → train/difficult mix (0–30% easy, 30–60% 70/30, 60–100% 50/50). Requires same-canvas splits |
| `--eval-split` | = `--split` | Split for eval-on-ckpt GIFs (defaults to `test/complex` when `--curriculum`) |
| `--scan-weight` | `1.0` | (v2: diagnostic only — scan_self no longer in the reward) |
| `--novel-scan-weight` | `1.0` | α_novel: privileged team-union novel-scan credit (v2 core reward term) |
| `--team-weight` | `0.3` | β: shared Δunion coef (cooperation anchor) |
| `--give-bonus` | `1.5` | ζ_give: NEW cells brought to teammate at comm |
| `--recv-bonus` | `0.5` | ζ_recv: NEW cells received at comm |
| `--overlap-pen` | `3.0` | η_lap: redundant parallel-scan penalty |
| `--revisit-pen` | `0.05` | γ: revisit penalty per step (graduated by recency) |
| `--revisit-window` | `8` | W: revisit lookback steps |
| `--yield-scale` | `3.0` | amplify `cand_own_minus_team` yield feature |
| `--proximity-pen` | `0.05` | per-step penalty when teammate visible within sensor_range |
| `--target-switch-pen` | `0.01` | δ_obj: objective second-guessing penalty (v2: argmax intent, default lowered from 0.05) |
| `--stall-pen` | `0.1` | δ_stall: heavy penalty for standing still (no net displacement this step) |
| `--path-bias-floor` | `1.5` | fixed floor on target-following bias (actor logits toward strategic pick's BF first-hop) |
| `--clip-eps` | `0.2` | PPO clip ε (→ `MAPPOCfg`) |
| `--k-epochs` | `4` | PPO epochs per rollout |
| `--gae-lambda` | `0.95` | GAE λ |
| `--gamma` | `0.99` | discount factor |
| `--vf-coef` | `0.5` | value loss weight |
| `--tbptt-steps` | `16` | TBPTT chunk length |
| `--lr` | `3e-4` | Adam LR (now wired to `lr_actor`) |
| `--wandb` | off | Log to Weights & Biases. `--wandb-project/-entity/-group/-run-name/-mode/-tags` |

**On `--seed`**: seeds torch RNG (action sampling, init) only. Map sampling RNG uses fresh entropy each run, so training and eval see different maps regardless of `--seed`. `eval_final.py --seed N` accepts an int for reproducible map selection.

MAPPO knobs (`clip-eps`, `k-epochs`, `gae-lambda`, `gamma`, `vf-coef`, `tbptt-steps`) and the
reward coefs above are all exposed precisely so the W&B sweep (§12) can tune them. Remaining
hardcoded knobs (lattice spacing, GAT width, ...): see [§11](#11-currently-hardcoded-knobs-not-on-the-cli).

---

## 6. Recommended training command (v0.4)

Set `--rollout-len ≥ --max-episode-steps` so full episodes complete inside a rollout and `ep_end` is populated each iter. `--max-episode-steps` should be large enough to actually explore (256+ on 480×640).

```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. python scripts/run_train.py \
    --split train/easy \
    --total-steps 5_000_000 \
    --n-envs 64 --n-agents 2 \
    --comm-range 120 \
    --rollout-len 256 --max-episode-steps 256 \
    --minibatches 1 \
    --lr 3e-4 --ent-coef 0.01 \
    --path-bias-floor 1.5 \
    --compile --eval-on-ckpt \
    --out /workspace/MARLauder/runs/run_v04'
```

Debug sanity (perfect info — verify coordination ceiling):

```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. python scripts/run_train.py \
    --split train/easy --total-steps 5_000_000 \
    --n-envs 64 --n-agents 2 \
    --rollout-len 256 --max-episode-steps 256 --eval-on-ckpt \
    --force-full-pos-sharing --force-full-occupancy-sharing \
    --out /workspace/MARLauder/runs/run_v04_godmode'
```

Smoke run (~1 min, verify pipeline):

```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. python scripts/run_train.py \
    --split train/easy --total-steps 40000 \
    --n-envs 8 --n-agents 2 \
    --rollout-len 64 --max-episode-steps 64 \
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

### Single map by index — `run_eval.py`

```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. python scripts/run_eval.py \
    --ckpt /workspace/MARLauder/runs/run_v04/final.pt \
    --split train/easy --map-idx 9580 \
    --n-agents 2 \
    --d-hidden 128 --n-heads 4 --n-layers 2 \
    --steps 256 \
    --out /workspace/MARLauder/runs/run_v04/eval_map9580.gif'
```

Env cfg (`n_hops`, `top_k`, force flags) is read from the checkpoint. To force persistent sharing on a checkpoint trained without it, add `--force-full-occupancy-sharing` / `--force-full-pos-sharing`. Use a milestone `ckpt_*.pt` or a v0.4 `final.pt` — both carry `cfg`; older `final.pt` lacked it (use the CLI override flags then).

### Eval rendering

Each frame is a horizontal stack of **M panels** (one per agent). Each panel shows:

- That agent's personal occupancy map (sigmoid of log-odds). Under `--force-full-occupancy-sharing` all panels render the same fused map.
- That agent's own frontier (red tint).
- Ego-centric lattice graph: nodes colored cyan→orange by utility, current-node yellow ring.
- **Strategic head's chosen target** (amber ring) + the **correct BF path** from curr to it (amber polyline). This is what the policy actually pursues — not the legacy env-argmax target.
- The agent itself (filled dot in its `C_AGENTS` color) + trail.
- Other agents as "ghosts" (smaller dots).
- Green line between agents whenever both are within `comm_range_px` AND have LOS clear.
- Top-left text bar: `[A0] t=N explored=X.X%`.

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
[it   N/T] ep_end=XX.X%(ended=K)  pg=±0.0NNN  v=N.NNNN  ent=N.NNN  kl=±0.0NNN  clip=N.N%  redun=N.NN stall=N% pair=N.NN sps=NNN(NNNavg)
```

| Metric | Meaning / Healthy | Warning sign |
|---|---|---|
| `ep_end` | mean explored fraction at the terminal step of all episodes that ENDED this iter. `ended=K` = how many episodes that was. Grows over iters. `n/a` until ≥1 episode completes (set `rollout-len ≥ max-episode-steps`). | flat near random after 100+ iters |
| `pg` | small negative (-0.005..-0.02) | always positive, or huge swings |
| `v` | drops then plateaus | climbing, or stuck (normalizer not adapting) |
| `ent` | decays smoothly | crashes to ~0 within a few iters (collapse) |
| `kl` | < 0.02 | > 0.1 (clip ineffective) |
| `clip` | 5-20% | > 50% (lr too high) or 0% (too low) |
| `redun` | redundancy `(Σ own_free − union)/union`. Lower over training = agents stop overlapping | stays high / rises (chasing, scanning same area) |
| `stall` | fraction of steps with no net displacement. Should fall toward 0 | stays high (deadlocks / standing still) |
| `pair` | mean pairwise inter-agent distance / canvas-diag. Rises as agents separate | stays low (chasing / clustered) |
| `coll`/`upd` sps | flat across run | dropping (memory pressure / recompile / oom) |

Full exploration-quality metric set (logged to W&B, §12): `metric/redundancy`,
`metric/stall_rate`, `metric/revisit_rate`, `metric/mean_pair_dist`,
`metric/coverage_per_dist`, `metric/steps_to_50`, `metric/steps_to_90`, plus per-term reward
contributions under `reward/*` and the composite `explore/efficiency`.

**ep_end populated only when episodes finish in the rollout.** With `rollout-len < max-episode-steps`, most iters show `n/a` (only 99%-threshold completions land). Match them for a number every iter.

**fp16 late-training NaN guard**: at very low entropy a logit row could go all-`-inf`/NaN and crash `Categorical`. v0.4 masks with a finite large-negative and `nan_to_num`-guards the logits, so a one-step spike no longer kills the run. Frequent guard activation still signals instability — lower `--lr` or raise `--ent-coef` late.

Random baseline final explored on `train/easy` (96 steps, NR=16, rays=720, sensor=60): **~6%**. Trained policy should beat ≥ 2×.

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
                          │   bf_from_target(curr)           │
                          │   bf_from_target(teammate lkp)   │
                          │   extract_topk_candidates(K=16)  │
                          └─────────────┬───────────────────┘
                                        │ node_feat[N,M,N_max,7] (ego-window), edge_idx, masks,
                                        │ cand_feat[N,M,K,8], cand_bf_first_hop,
                                        │ prev_action, last_known_pos
                                        ▼
   ┌──────────── Ego-centric GAT Encoder (per (env, agent))  ────────────────┐
   │   window (2·n_hops+3)²; Linear(7→d); MaskedGATLayer × n_layers           │
   │   curr_emb [N·M, d]   nbr_embs [N·M, K=8, d]                             │
   └───────────────┬─────────────────────────────────────┬──────────────────┘
                   │                                     │
        decentralized actor                       centralized critic (CTDE)
                   │                                     │
   ┌─── per agent ─▼─────────────────────────┐  ┌─ per env ─▼───────────────┐
   │ StrategicHead(curr_emb, cand_feat[K,8])  │  │ concat curr_emb over M     │
   │   → target_logits[K] → Gumbel-ST pick    │  │ → Linear(M·d→d)+GELU       │
   │   → strategic_emb, target_idx            │  │ GRUCell(joint, h_prev)     │
   │ actor_pre([curr_emb‖strategic_emb        │  │ Linear→GELU→Linear→V(s)    │
   │   ‖next_hop_onehot‖prev_action])         │  └────────────────────────────┘
   │ GRUCell → PointerHead(nbr_embs, mask)    │
   │   logits[K=8] + path_bias·first_hop      │
   │ (finite mask + NaN guard) → action       │
   └──────────────────────────────────────────┘
                   │
                   ▼  env.step(action) → per-agent reward / done / next obs
                   ▼  buffer.store(.., target_choice, ..)
            ┌──────────────────────────┐
            │ MAPPO update             │
            │  per-agent GAE-λ         │
            │  shared CTDE V baseline  │
            │  replay stored pick (STE)│
            │  PPO clip ε=0.2, AMP fp16│
            │  TBPTT chunks, encode_chunk
            └──────────────────────────┘
```

Invariants:
- Encoder weights shared actor↔critic — both gradients flow back.
- Actor decentralized: each agent sees only its own ego-window + its own candidate set.
- Strategic head shared across agents but operates per-agent (per-agent inputs).
- Critic centralized: concatenates per-agent curr_emb → joint state V(s).
- Per-agent advantages (GAE) against a single shared V; returns target = team-mean.
- Hidden states zeroed at episode resets via `(1 - done)` mask.
- All obs tensors live on GPU; no host roundtrips during rollout.
- `N_max = (H/NR)·(W/NR)` fixed → pure pad+mask, no PyG dynamic batching.
- Edge length: axial `NR`, diagonal `NR·√2` — used in all Bellman-Ford calls.
- Encoder called ONCE per TBPTT chunk; strategic pick replayed via stored `target_choice`.

---

## 10. Roadmap

| Ver | Goal | Status |
|---|---|---|
| v0.1 | Single-agent baseline (Warp LiDAR + lattice graph + GAT + MAPPO) | ✓ |
| v0.2 | Terminology rename, Bellman-Ford guidepost, diagonal cost, MAPPO speedup, docs | ✓ |
| v0.3 | Multi-agent intermittent comm, per-agent maps, per-agent eval render, step-penalty + completion-bonus reward, O(1) curr_idx, FREE-only start placement, random-seed eval | ✓ |
| v0.4 | Phase A v2 strategic frontier-attention head, Phase B BF target-rooted + warm-start, Phase C ego-centric encoder (n_hops), Phase D per-agent set-op reward, Option A BF-from-curr for cand ranking, G.3 strategic-pick render + BF path bias, G.4 anti-chase (yield-scale + proximity + overlap), random map RNG | ✓ (current) |
| v0.5 | Curriculum train/easy → train/difficult (ramp scaffold landed in v0.4; **blocked** until splits share a canvas — easy=480×640, difficult=1000×1000) | partial |
| v0.6 | Eval suite: per-split curves, TB logger, milestone GIF auto-generation, voluntary-rendezvous reward | planned |
| v0.7 | ToM `teammate_belief.py` module (probabilistic teammate-state estimator + belief merge on rendezvous; replaces point `lkp` in candidate features) | planned |

v0.8 (hierarchical L2 graph) explicitly **out of scope** for this rewrite.

**Curriculum note**: `--curriculum` + `MultiSplit` are implemented and ramp the easy/difficult mix over iters, but `MultiSplit` raises if the two splits differ in canvas size. To enable, pre-process maps to a common H×W first.

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
| `proximity_penalty_radius_px` | `-1` | Proximity penalty radius. `-1` = sensor_range_px |

(`cand_own_minus_team_scale`, `top_k_candidates`, force flags, reward weights → exposed on the CLI; see §5.)

### `train.driver.TrainCfg`

| Name | Default | Effect |
|---|---|---|
| `d_hidden` | `128` | Encoder + GRU hidden width |
| `n_heads` | `4` | GAT attention heads (must divide `d_hidden`) |
| `n_layers` | `2` | GAT layers; tied to `n_hops` in `_normalize_cfg` |
| `lr_actor` | `3e-4` | Adam LR (overridden by `--lr`) |
| `lr_critic` | `1e-3` | (unused; single optimizer) |
| `eval_steps` | `-1` | Eval-on-ckpt episode length; `-1` = `max_episode_steps` (set via `--eval-steps`) |
| `eval_n_maps` | `2` | Number of eval GIFs per milestone |
| `eval_map_idx` | `-1` | `-1` = random map each time |
| `path_bias_floor` | `1.5` | Fixed floor on path-following bias (set via `--path-bias-floor`) |
| `curriculum` | `False` | Ramp easy/difficult mix (set via `--curriculum`) |
| `curriculum_splits` | `("train/easy","train/difficult")` | Splits used when curriculum on |

### `train.mappo.MAPPOCfg`

| Name | Default | Effect |
|---|---|---|
| `clip_eps` | `0.2` | PPO clip ε (`--clip-eps`) |
| `vf_coef` | `0.5` | Value loss weight (`--vf-coef`) |
| `ent_coef` | `0.01` | Entropy bonus weight (`--ent-coef`) |
| `k_epochs` | `4` | PPO epochs per rollout (`--k-epochs`). Reduce to 2 if KL > 0.02 |
| `tbptt_steps` | `16` | TBPTT chunk length for hidden-state truncation (`--tbptt-steps`) |
| `n_minibatches` | `1` | PPO minibatches per epoch (`--minibatches`). Must divide `n-envs` |
| `gamma` | `0.99` | Discount factor (`--gamma`) |
| `lam` | `0.95` | GAE λ (`--gae-lambda`) |
| `grad_clip` | `0.5` | Global gradient clip norm |
| `clip_vloss` | `True` | Clipped value loss (max of unclipped and `V_old±clip_eps` error). MAPPO paper §3.3 / Alg.1 |
| `huber_delta` | `10.0` | Value-loss Huber delta (paper Tab.7). `0.0` = squared error. Robust to return-spike outliers |

**Weight init**: all `nn.Linear` / `nn.GRUCell` use orthogonal init (gain √2), policy/strategic logits gain 0.01, value head gain 1.0 — MAPPO paper Tab.7 (`models/init_utils.py`). Was torch-default before.

---

## 12. Weights & Biases + hyperparameter sweeps

`wandb` is in `requirements.txt` (pip-installable in the running container if the image
predates it). Logging is **off by default** — pass `--wandb` (no network otherwise).

**Per-iter logging** (`train.driver.train`, guarded import → silent no-op if `--wandb` off or
package missing): `train/{pg_loss,v_loss,entropy,kl,clipfrac}`, `perf/{sps,coll_sps,upd_sps}`,
`explore/{ep_end,ep_end_n,efficiency}`, `reward/*` (per-term signed contributions from
`info["reward_terms"]`), `metric/*` (exploration quality from `info["metrics"]`, aggregated
over the rollout in `collect_rollout`). `wandb.init(config=…)` flattens the full `TrainCfg`
(incl. `env` + `ppo`).

**Fixed eval suite (v2 — the sweep's scoring source)**: every `TrainCfg.eval_every=10` iters,
`_run_eval_suite` runs the policy **deterministically** on the 8 hardcoded validation maps
`EVAL_MAP_IDX` (same for every run/machine — "same exam") in a persistent 1-env Explorer that
mirrors the training cfg (force flags incl.). Logs `eval/{coverage_auc, contrib_imbalance,
contrib_imbalance_norm, fairness, sensing_overlap, comm_duty, success_rate, steps_to_90,
score, score_std}`.
**D2 — equity rebalance**: `eval/score` is the mean of per-map scores, each
`= coverage_auc − w_imb·(contrib_imbalance / (1−1/M)) − w_ov·sensing_overlap`. The
imbalance is **normalized to [0,1]** before weighting (raw `contrib_imbalance` spans only
`[0, 1−1/M]` ≈ [0, 0.5] for M=2, so at the old weight it was a near-free rider and the
sweep optimized ~90% raw coverage — that's why a mechanical `proximity_pen` won). Weights
exposed as `--score-w-imbalance` (0.5) / `--score-w-overlap` (0.25) flags. `eval/fairness`
= Jain index over per-agent novel shares (1 = perfectly equal) — scale-robust cooperation
read. `eval/score_std` = cross-map spread of per-map score → exposes map-luck/noise
(motivates seed replicas, D5). AUC pads early success with the final explored_rate so
finishing sooner scores strictly higher. `contrib_imbalance` = `max_a(novel share) − 1/M`
from `info["novel_cells_ep"]`. These 8 maps are validation — final reporting must use fresh
random maps / `test/*` splits. Cost ≈ 5% wall time.

Legacy rollout-based `explore/efficiency` (= ep_end − 0.5·redundancy − 0.5·stall_rate) is
still logged but is NOT the sweep target anymore (it saturated in sweep v1).

**Exploration metrics** (`env.explorer.step` → `info["metrics"]`, all GPU):

| Key | Definition |
|---|---|
| `redundancy` | `(Σ_a own_free − union_free)/union_free` on **PRE-fusion** per-agent maps (overlap; low good). Pre-fusion is essential — post-fusion in-comm agents share an identical map → metric pinned at M−1 |
| `stall_rate` | mean of `stall_pen` (fraction of steps with no net displacement) |
| `revisit_rate` | fraction of steps revisiting a node within `W` |
| `mean_pair_dist` | mean pairwise `‖pos_i−pos_j‖` / canvas-diag (separation; chase = low). Clean — unaffected by fusion |
| `coverage_per_dist` | `Σ team_delta / Σ step_disp` (Δunion per pixel travelled; efficiency, size-invariant) |
| `steps_to_50` / `steps_to_90` | first step each env crosses 50/90% coverage this rollout (raw speed) |
| `steps_to_50_per_kfree` / `_90_per_kfree` | above ÷ (free cells / 1000) → **map-size-normalized** speed (comparable across maps with different free area) |

Residual caveat on `redundancy`: even pre-fusion, each agent's map carries cells fused in *past* rendezvous, so the metric reflects current map *divergence* rather than pure independent-scan overlap. `mean_pair_dist` + `coverage_per_dist` are the cleaner, fusion-free chase/efficiency signals — cross-check all three.

**Sweeps v2 — two stages, MAPPO frozen** (sweep v1 over 17 dims produced no signal: saturated
metric, dominated reward, no MAPPO separation, k_epochs=6 KL blowups). Both stages maximize
`eval/score`; Hyperband `min_iter: 3` counts eval-suite logs (≈ train iter 30). Frozen MAPPO:
lr 3e-4, ent 0.005, clip 0.2, k 4, λ 0.95, γ 0.99, mb 4, tbptt 16.

1. **`sweep.yaml` — Stage 1, division of labor under perfect information.** Trials run with
   `--force-full-occupancy-sharing --force-full-pos-sharing` (rendezvous terms degenerate →
   novel-scan is the only per-agent credit). Swept: `novel-scan-weight [0.5,2]`,
   `team-weight [0,0.5]`, `proximity-pen [0,0.2]`, `target-switch-pen [0,0.05]`,
   `n-hops {2,4,6}` (window 49/121/225 nodes, n_layers tied; peak VRAM measured 6.5 GB at
   h=6 / n-envs 64 — fits the 12 GB 4080).
2. **`sweep_stage2.yaml` — Stage 2, rendezvous economy at comm-range 120.** Paste Stage-1
   winners into the marked command block; sweep only `overlap-pen [0.5,3]` (lowered — novel
   scan already zero-pays redundant cells), `give-bonus [0.5,3]`, `recv-bonus [0,1.5]`.
   `comm_duty_cycle` becomes meaningful here (Stage 1 pins it at 1.0).

Param keys are the exact dashed CLI flags (W&B emits `--<key>=<value>`).

```bash
docker exec -it marlauder bash -lc 'cd /workspace/MARLauder && wandb login && wandb sweep sweep.yaml'
docker exec -it marlauder bash -lc 'cd /workspace/MARLauder && wandb agent <ENTITY/PROJECT/SWEEP_ID>'
```

To change any of these, edit the dataclass directly. There is no in-place override — values are stamped into the checkpoint's `cfg` field at save time.
