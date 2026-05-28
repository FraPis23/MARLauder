# MARLauder Development Log

Session-based log of design decisions, architectural understanding, observed problems, and proposed/applied fixes. Append entries as the project evolves. Newest at top.

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
