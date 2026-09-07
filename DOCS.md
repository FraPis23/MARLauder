# MARLauder — reference

Module map, data flow, observation schema, reward, and the full parameter reference.
For installation and the headline results, see [README.md](README.md); for the evaluation
protocol against IR2, see [eval/comparison/PROTOCOL.md](eval/comparison/PROTOCOL.md);
for a visual overview of the pipeline, open [docs/architecture.html](docs/architecture.html).

---

## 1. Module map

```
MARLauder/
├── env/         Simulation: Warp world, sensors, lattice graph, frontier, teammate belief, env loop
├── models/      Networks: masked GAT encoder, actor-critic, value normalizer
├── train/       MAPPO: rollout buffer, PPO update, training driver
├── eval/        Deterministic rollout, inspector trace, GIF rendering, IR2 comparison
├── scripts/     CLI entrypoints
├── tests/       Property tests (see tests/README.md)
├── tools/       Offline diagnostics (see tools/README.md)
├── pipelines/   End-to-end reproduction recipes (see pipelines/README.md)
├── viz/         Web dashboard + step-through decision inspector
├── docs/        architecture.html — the pipeline diagram
├── data/        Preprocessed map packs (uint8 memmap + meta.npz per split; gitignored)
└── docker/      Dockerfile for the runtime image
```

### `env/`

| File | Purpose |
|---|---|
| `world_warp.py` | GPU LiDAR via NVIDIA Warp. Maintains **per-agent** `occupancy_torch [N, M, H, W]` (stored flat as `[N·M, H, W]`, since Warp kernels index at most 3 dimensions) and its log-odds source. `fuse_maps(comm_mask)` merges connected pairs by elementwise **max-magnitude**, which keeps OBSTACLE evidence that a plain `max` would drop. `_mark_pos_free` stamps a 3×3 footprint at `2·LO_FREE` so the robot's own cell is reliably FREE — a single `LO_FREE` add lands exactly on the strict `> LO_FREE_TH` threshold, leaving the current graph node invalid and the agent with no legal moves. |
| `maps.py` | Loads `data/<split>/maps.npy` + `meta.npz` and samples batches to the GPU. `MultiSplit` is a weighted union of splits for the ramped curriculum. |
| `frontier.py` | `compute_frontier(occupancy)` → bool `[N,H,W]`. A frontier cell is FREE with 2..7 UNKNOWN neighbours: ≥2 so it genuinely borders unknown space, ≤7 so an isolated speck does not qualify. |
| `graph_lattice.py` | 8-neighbour lattice on free cells, reachability flood-fill, collision-checked edges, integral-image utility. `bf_from_target` is an overwrite-mode, warm-startable Bellman-Ford from any source. `build_radar` compresses the known world *beyond* the ego window onto its geodesic horizon. `value_field` partitions the BF tree by first step. `extract_local_window` slices the `(2·n_hops+3)²` ego window. |
| `teammate_belief.py` | Uniform expanding-zone belief: a geodesic ball that grows one hop per step from the last-known node and collapses to a point at contact. |
| `teammate_belief_pathfront.py` | The default belief model. Hypotheses freeze at comm break, one per frontier opening; each travels its BF geodesic (TRANSIT), then evolves by absorbing diffusion on the known graph. Lives only on KNOWN-free nodes. |
| `explorer.py` | The environment. `EnvCfg` holds every simulation and reward parameter; `step()` and `_refresh_obs()` are described in §2. |

### `models/`

| File | Purpose |
|---|---|
| `gat.py` | `MaskedGATLayer` + `GATEncoder`. Multi-head attention over K=8 padded neighbours plus a self-loop, in plain torch (no PyG). Two learnable shaping terms: **A1**, a per-head temperature `τ_h` clamped to [0.1, 10], which fixes the near-uniform softmax the fixed `1/√D` scaling produced; and **A2**, a per-head structural bias computed from a fixed subset of the neighbour's RAW features, injecting the routing signal past the gradient-starved q/k path. Default head groups: `H0 [2,5]` explore, `H1 [4,6]` rendezvous, `H2 [3]` recency, `H3 [5,6]` far-field. |
| `actor_critic.py` | `MarlActorCritic`. One shared encoder feeds a **decentralized actor** (pointer over the K neighbour embeddings, biased by the value field) and a **centralized critic** (mean⊕max pooling over agents, so the same weights serve any M). Both GRUCells exist for checkpoint compatibility but are bypassed unless `--gru`. `encode_chunk` batches the encoder across a whole TBPTT chunk. |
| `value_normalizer.py` | Welford online mean/variance. The critic predicts normalized values; GAE uses denormalized ones. Non-finite samples are dropped, since one would permanently poison the running statistics. |
| `init_utils.py` | Orthogonal initialization (MAPPO paper Table 7): gain √2 on hidden layers, 0.01 on policy logits, 1.0 on the value head. |

### `train/`

| File | Purpose |
|---|---|
| `buffer.py` | Pre-allocated rollout `[T, N, M, ...]`. **Per-agent** GAE-λ against a shared CTDE value baseline: advantages `[T,N,M]`, returns `[T,N]` (team-mean V target). |
| `mappo.py` | The PPO update. One encoder call per TBPTT chunk, then a per-timestep re-roll. PPO clip, `V_old`-clipped Huber value loss, entropy bonus, optional frontier-diversity auxiliary loss, **bf16 AMP** (GradScaler disabled — bf16 has fp32 exponent range, so the fp16 NaN-collapse mode is structurally absent). Reports `explained_var`, which unlike `v_loss` is scale-free and therefore comparable between M=2 and M=4 runs. |
| `driver.py` | `TrainCfg`, the main loop, milestone checkpoints, `ckpt_best` tracking, the deterministic eval suite, `metrics.jsonl`, the web control channel, and the curriculum paths. |

### `eval/`

| File | Purpose |
|---|---|
| `rollout.py` | Deterministic single-episode play; one rendered panel per agent, horizontally stacked. |
| `trace.py` | Full per-step decision trace for the web inspector: observations, logits, per-agent reward components, rendezvous factors, teammate belief, and the real per-layer GAT attention. |
| `ckpt_loader.py` | Infers the architecture from the checkpoint itself. **Mandatory** in anything that loads a checkpoint: a 6-layer checkpoint silently evaluated at 2 layers scored 1–4 % explored instead of 56 %, with no crash to flag it. |
| `render.py` | Palette and painters used by both the GIFs and the step tests. |
| `comparison/` | The IR2 comparison: the frozen `PROTOCOL.md`, the map index files, the dataset parity gate, and the aggregator. |

---

## 2. Data flow — one rollout iteration

```
data/<split>/maps.npy  (memmap, uint8 [N, H, W])
   │  env.maps.sample_batch → N maps on the GPU
   ▼
WarpWorld: per-agent log-odds + categorical occupancy
   │ env.step(action) for t in [0, T):
   │   1. decode the K=8 slot → target node; arbitrate agents picking the same node
   │   2. path-follow K_sub sub-steps, LiDAR each one, wall revert + asymmetric collision
   │   3. comm check → _sync_rewards (PRE-fusion) → fuse_maps → last_known_pos / staleness
   │   4. per-agent reward: novel_scan − revisit − stall + sync + completion − step_penalty
   ▼
   │ _refresh_obs (agents batched into B = N·M):
   │   compute_frontier(occupancy)                    torch conv2d
   │   GraphLattice.build()                           flood-fill + collision + utility
   │   bf_from_target(curr)                           warm-started
   │   bf_from_target(teammate)                       over the optimistic FREE∪UNKNOWN graph
   │   teammate belief (pathfront or uniform)         → feat[4]
   │   build_radar                                    → feat[5] b_util, feat[6] b_teammate
   │   value_field                                    → obs["value_field"] [N,M,K]
   │   extract_local_window                           → the (2·n_hops+3)² ego window
   │   critic_global[7] + agent_scalars[5]
   │   5. rdv_dense = w · g · (φ_prev − φ_now)        added post-refresh
   ▼
obs dict [N, M, ...] → MarlActorCritic.act(obs, h_act, h_crit)
                          ├── shared ego-centric GAT encoder → curr_emb, nbr_embs
                          ├── actor: (curr_emb ‖ prev_action ‖ value_field ‖ agent_scalars)
                          │          → PointerHead → action
                          └── critic: mean⊕max pool over M ‖ critic_global → V(s)
   │ buffer.store(...)
   │ after T steps: compute_gae → per-agent adv [T,N,M], team-mean returns [T,N]
   ▼ MAPPO update (k_epochs × n_minibatches × T/tbptt_steps chunks)
        ├── encode_chunk(chunk_obs)   ← ONE encoder pass per chunk
        ├── per timestep: pointer + critic re-roll
        └── optimizer.step()
```

---

## 3. Node features (`F_IN = 7`)

| Idx | Name | Meaning | Range |
|---|---|---|---|
| 0 | `x_rel` | `(node.x − curr.x) / win_half` — **ego-scaled** by the window half-extent, so in-window coordinates span the full range. Normalizing by the half-map instead squashed them to ~±0.15, where geometry drowned under the binary features at the input layer. | [−1, +1] |
| 1 | `y_rel` | as above, vertically | [−1, +1] |
| 2 | `utility` | **Frontier-gated information gain.** `seed = frontier_ribbon × (FLOOR + (1−FLOOR)·unknown_volume)`, then diffused along collision-checked edges so walls block by construction. The frontier gate is what makes it sharp and frontier-anchored; the volume multiplier is what makes a small opening onto a big unknown room score high. | [0, 1] |
| 3 | `age` | **Stationary recency**: `clamp((step − last_visit)/visit_age_window, 0, 1)`; never-visited nodes read 1 (cold, re-explorable), a just-walked node reads 0. | [0, 1] |
| 4 | `teammate_pot` | Teammate-proximity potential derived from the **belief field**, peak-normalized per teammate before the max over teammates (so at M>2 a hard-to-find teammate is not erased by a sharply-located one). Zero at M=1. | [0, 1] |
| 5 | `radar-util` (`b_util`) | **RADAR** — exploration mass beyond the ego window, routed geodesically down the BF parent chain onto the horizon gateway nodes, discounted by `radar_gamma^hops`. Obstacle-aware: the path bends around walls, never projects through them. Squashed as `m/(m+util_norm)`, which is monotone over the whole range and never saturates. | [0, 1] |
| 6 | `radar-teammate` (`b_teammate`) | The same transport applied to the teammate **belief field** (`--radar-team-source belief`) rather than to a point at the last-known position. Row-normalized into a directional distribution. | [0, 1] |

Invalid nodes have their feature row zeroed, and edges to them are masked in attention. The encoder
runs on the ego window, never on the full lattice.

## 3b. The other observation tensors

**`agent_scalars [N, M, 5]`** — per-agent, and every entry is something a deployed robot could
compute for itself (order is authoritative, from `AGENT_SCALAR_DIM` in `models/actor_critic.py`):

| Idx | Name | Meaning |
|---|---|---|
| 0 | `g` | The surplus gate ∈[0,1] — *the same* gate that scales the dense rendezvous reward, so the policy's trigger and the reward's trigger are one quantity. |
| 1 | `staleness` | Steps since the last sync with the owed teammate, over `rdv_urgency_T`. A fixed physical scale, not `max_episode_steps`, which changes between phases and would silently rescale the input across a warm start. |
| 2 | `travel_frac` | Episode budget consumed, `max(travel_px/budget, t/T_max)` — progress toward whichever stop criterion binds first. Without it the actor cannot perceive its own deadline. |
| 3 | `contact` | 1 while in comm with any teammate. |
| 4 | `offer_frac` | Surplus owed, as a fraction of the map. `g` cannot carry the magnitude: it is clamped to 1 twice over, so past saturation "I owe him a sensor disk" and "I owe him half the map" are the same number. |

`agent_scalars` sits **last** in the actor concatenation on purpose: the warm-start widening path in
`train/driver.py` copies a narrower checkpoint weight into the *leading* columns, so any block after
it would be silently re-aimed at the new scalar columns the next time this width grows.

**`critic_global [N, 7]`** — value head only, never seen by an actor:
`[explored_frac, t/T, geo_pair, coverage_rate, redundancy, sync_surplus, sync_staleness]`.
The pooled per-agent embeddings are ego-relative, so they carry exploration *content* but not team
geometry; the relational geometry lives here as `geo_pair` (nearest-teammate geodesic / diameter,
translation-invariant). `sync_surplus` and `sync_staleness` are what let V(s) represent "we are
about to gain a lot by meeting" — without them the advantage of an approach move is ≈0 and the sync
reward has nothing to bootstrap through. There is deliberately no absolute team position: V(s) has
to generalize across maps.

**`value_field [N, M, K]`** — for each of the K exits, the discounted utility mass reachable down
that branch of the BF tree, `V_k = Σ γ_vf^hops · utility`, max-normalized. One comparable scalar per
action, so "near and weak" versus "far and strong" is resolved analytically instead of asking the
encoder to integrate the window and the radar. It enters the actor trunk *and* biases the pointer
logits through a learnable `w_vf` (initialised to 1, so it steers from step 0 and the network may
amplify or unlearn it).

---

## 4. Reward

Per agent, at lattice level, in map-independent units (`scan_norm_nodes = 50` ≈ one sensor disk).

```
novel_scan[a]  = |cells a scanned this step ∧ ¬union_prev| / scan_norm   # NEW TO THE TEAM UNION
revisit_pen[a] = (W − age)/W  if the chosen node was visited within W steps, × a streak multiplier
stall_pen[a]   = 1 if ‖pos_after − pos_before‖ < nr·0.5, × a capped streak multiplier
step_penalty   = step_cost · (edge_len / NR)                             # axial 1, diagonal √2

# Dense rendezvous (M>1, added after _refresh_obs):
g              = clamp(surplus / (frac(baseline) · baseline), 0, 1) + urgency_nudge
φ              = geodesic(curr → the owed teammate) / (nr · scan_norm)
rdv_dense[a]   = w · g · (φ_prev − φ_now)          # optionally clamped to the approach half

# Sync event (M>1, computed on the PRE-fusion maps):
give_ij        = |M_i \ M_j| / scan_norm
paid_ij        = rising_edge(comm_ij) AND (t − t_last_paid_sync_ij ≥ sync_min_gap)
sync[a]        = ζ_g · Σ_j paid_aj · (give_aj + ρ · give_ja)

reward[a] = α·novel_scan − γ·revisit_pen − δ_stall·stall_pen + sync
          + 1{done} · completion_bonus − step_penalty + rdv_dense
```

**Privileged novel-scan credit.** An agent is paid only for cells new to the **team union**, so a
follower scanning a leader's wake earns exactly zero and splitting up is the highest-paying policy
by construction. This is the one CTDE-only signal: privileged at training time, never visible to the
deployed actor. Its unobservable overlap variance is absorbed by the centralized value baseline —
which is why `critic_global` carries `redundancy`. There is deliberately **no separation or
proximity penalty**: novel-scan does the spreading, so agents never "fear the only corridor".

**Sync-event reward — the objective term for rendezvous.** `rdv_dense` is telescoping shaping: the
net payoff of a whole separate→approach→meet cycle is only `w·g·φ_sep` (0.045 at `w=0.10`, and
still just 1.13 at `w=2.5`) against a measured 1.7–1.9 detour cost, so no dense weight can make
meeting worth it — past `w≈2` it simply becomes a chase term. What pays for a rendezvous has to be
the exchange itself. Two guards make it farm-proof:

* **Rising edge only.** With `ss_thresh = -70` the free-space radio radius is 150–310 px while two
  80 px LiDAR disks stop overlapping at 160 px, so "walk in parallel at the comm boundary" gives
  *continuous comm with disjoint sensing* — the reward-maximal degenerate strategy under any
  per-step transfer reward. Paying only the rising edge makes a permanent tether earn zero.
* **`sync_min_gap`.** Kills range flicker. The contact still fuses; only the payment is suppressed.

Frequency-farming is impossible by conservation: `give` is a set difference over monotone maps, so
syncing at t1 and then t2 pays exactly what syncing only at t2 pays. Re-gifting is impossible too,
since post-fusion `M_i \ M_j = ∅`. Calibration at `ζ_g=0.25, ρ=0.5`: ≈2.55 per sync after ~200 steps
apart, against a 1.7–1.9 detour cost, with a per-episode ceiling of 6.5 versus `novel` 17.3 — meeting
is worth 37.5 % of everything found since parting, never more than exploration itself. `ρ<1` keeps
`give` dominant: a meeting needs *both* agents to move, so `recv` must be positive, but a lazy
agent's `recv` is large precisely because it explored nothing. These properties are pinned by
`tests/11_test_sync_reward.py`.

**M-scaling of the sync bonus.** `ζ_g_eff = ζ_g · (2/M)^a`, identically 1 at M=2 for any exponent,
so no M=2 result can move. Encounters are not M-invariant: measured between the M=2 and M=4 runs,
the realized sync reward share went 5.5 % → 9.8 % and the sync rate ×3.0, while `novel` was already
at parity. The risk to watch is that under `done_mode=own` each robot needs the maps of M−1 others;
`eval/own_coverage_final` and `eval/sync_gap` are the abort signals, not `eval/score`.

**The rendezvous gate `g`.** Content-driven, and the *required* fraction itself decays with how much
map was already shared at the last sync: `frac(b) = frac_min + (frac_max−frac_min)·exp(−b/b0)`. The
first rendezvous (baseline tiny) demands a large relative surplus; once the shared baseline is
already most of the map, the same relative fraction would mean an enormous absolute surplus, so the
requirement relaxes. On top sits a small capped urgency nudge. With `--rdv-urgency-mode budget` that
nudge ramps on the fraction of the *episode budget* spent rather than on time apart, so the pull
appears near the deadline instead of mid-episode when the agents should still be splitting.

**Decentralization.** Every term is computed from agent-local state or from comm-gated set
operations. The team-union subtraction in `novel_scan` is the only privileged signal, and it is
training-only.

---

## 5. Communication

Default `comm_model = signal_strength`: a log-distance path-loss radio. The segment between two
agents is split into free and obstacle length; walls **attenuate** (`γ_obst = 4`) rather than block,
and per-episode shadowing noise is resampled at each reset. Two agents connect iff the received
power `P_R = P_T − PL` exceeds `ss_thresh`. A legacy `los` model (hard Euclidean cutoff plus a
Bresenham line-of-sight test) remains available.

**Multi-hop relay** (`comm_relay`, on by default for new runs). The comm check is pairwise, so with
A—B—C, A and C used to exchange nothing. The relay closes the mask transitively: everything that is
*state* or *observation* uses the connected component, because the whole flock is one radio network
and a relayed map is as real as a direct one. The sync **reward** deliberately stays on the direct
link. Note `fuse_maps` walks pairs in place, so a transitively-closed mask converges in one pass.

On connection, in the same step: the per-agent log-odds maps are fused by max-magnitude,
`last_known_pos` is overwritten with the true current position, and the pair's staleness timer resets.

Agent–agent collision is resolved env-side. Priority is *who arrives first* — the agent with less
remaining travel to the contested point wins, so an axial mover beats a diagonal one aiming at the
same node; only a true geometric tie falls back to a per-episode random key, redrawn each episode so
there is no systematic role bias. The winner is pushed radially out to exactly `min_dist` rather than
reverted, so progress is made every sub-step and the deadlock cannot latch.

---

## 6. Training

See [pipelines/README.md](pipelines/README.md) for the full four-stage recipe. A single stage:

```bash
PYTHONPATH=. python scripts/run_train.py \
    --split train/difficult --max-episode-steps 768 --max-travel-frac 0.030 \
    --n-envs 32 --n-agents 4 --rollout-len 256 --n-hops 6 --tbptt-steps 8 \
    --minibatches 1 --k-epochs 4 --gamma 0.998 \
    --belief-mode pathfront --radar-team-source belief \
    --radar-gamma 0.97 --radar-util-norm 3 --done-mode own \
    --rdv-weight 0.10 --rdv-clamp-pos --rdv-urgency-mode budget \
    --sync-weight 0.25 --sync-weight-m-scale 1.0 \
    --eval-suite-splits test/complex --eval-on-ckpt \
    --out runs/my_run
```

**Launch geometry and VRAM.** The OOM driver is the `encode_chunk` update peak
(∝ `tbptt · n_envs · n_layers · window`), which is **map-independent** — the model always sees the
same 225-node ego window at `n_hops=6` — so easy and difficult splits cap at the same environment
count. 32 envs at M=4 with 6 hops fit in 15.5 GiB by a hair; export
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or the first backward pass fails on
fragmentation. Halving `tbptt` from 16 to 8 halves the update peak.

Set `--rollout-len ≥ --max-episode-steps` if you want `ep_end` populated every iteration; otherwise
episodes straddle the buffer boundary. `_normalize_cfg` clamps `max_episode_steps` up to
`rollout_len` and prints a warning when it does — `params.json` records what you asked for,
`metrics.jsonl` records what actually ran.

### Console output

```
[it   N/T] ep_end=XX.X%(ended=K)  pg=±0.0NNN  v=N.NNNN  ent=N.NNN  kl=±0.0NNN
           clip=N.N%  ev=±N.NN  redun=N.NN stall=N% pair=N.NN sync=+N.NNNN(N.N/k)
           ownGap=N.NNN sps=NNN(NNNavg)
```

| Metric | Healthy | Warning sign |
|---|---|---|
| `ep_end` | mean explored at the terminal step of episodes that ended this iteration; grows over time | flat near random after 100+ iterations |
| `pg` | small negative, −0.005 to −0.02 | always positive, or large swings |
| `v` | drops, then plateaus | climbing, or stuck |
| `ent` | decays smoothly | crashes toward 0 (collapse) |
| `kl` | < 0.02 | > 0.1 (the clip is ineffective) |
| `clip` | 5–20 % | > 50 % (lr too high) or 0 % (too low) |
| `ev` | explained variance → 1 | ≤ 0: the advantages are mostly noise |
| `ownGap` | shrinking | growing: the union is complete but the robots are not sharing |
| `sps` | flat across the run | dropping (memory pressure, recompilation, OOM) |

### Where the numbers go

Every iteration writes one JSON row to `runs/<run>/metrics.jsonl`, whether or not W&B is enabled;
`scripts/analyze_run.py` reads it (`--compare`, `--reward-budget`, `--own-coverage`, `--csv`). W&B
is off by default and logs the same object when `--wandb` is passed.

### The eval suite

Every `--eval-every` iterations the policy is run **deterministically** on a fixed set of
evenly-spaced maps per split, in a persistent 1-environment mirror of the training config. Both the
radio-shadowing and the map/spawn RNG streams are re-pinned first: without that, the same checkpoint
scores ±9 points of success rate from spawn luck alone, and `eval/score` is the only writer of
`ckpt_best.pt`.

`eval/score` = `coverage_auc − w_imb·(contrib_imbalance/(1−1/M)) − w_ov·sensing_overlap −
w_idle·idle_rate_max`, averaged per map. Coverage AUC pads an early success with the final explored
rate, so finishing sooner scores strictly higher. Reported alongside but **not** in the score:
`score_own`, `own_coverage_auc/final`, `sync_gap`, `n_syncs`, `fairness` (Jain), `concurrency`,
`success_rate`, `steps_to_90`, `score_std`.

These maps are **validation**. Final reporting uses the frozen protocol in
[eval/comparison/PROTOCOL.md](eval/comparison/PROTOCOL.md).

---

## 7. Evaluation commands

```bash
# batch eval on N random maps → one GIF per map
PYTHONPATH=. python scripts/eval_final.py runs/<run>/final.pt --split test/complex --n-maps 5

# score an explicit list of checkpoints under identical settings, with a noise floor
PYTHONPATH=. python scripts/score_ckpts.py --ckpt runs/<run>/ckpt_0{40,60,80}.pt \
    --splits train/difficult,test/complex --repeats 3 --n-agents 4

# pick the best milestone checkpoint of a finished run → writes <run>/ckpt_best.pt
PYTHONPATH=. python scripts/eval_best.py --run runs/<run>

# a single map, with an inspector trace
PYTHONPATH=. python scripts/trace_episode.py --ckpt runs/<run>/ckpt_best.pt \
    --split test/hybrid --map-idx 167 --out runs/<run>

# random-policy sanity baseline — a trained policy should beat it by ≥2×
PYTHONPATH=. python scripts/baseline_random.py --split test/complex --map-idx 0 --steps 512
```

Architecture (`n_agents`, `d`, `n_heads`, `n_layers`, `use_gru`, the GAT ablations) is always
inferred from the checkpoint by `eval/ckpt_loader.py`; `torch.compile` key prefixes are stripped
automatically.

### The web inspector

`viz/web_server.py` (started automatically by `docker compose up`) serves a dashboard at
`http://localhost:8080/` listing every run, with a launch form auto-built from the CLI parser, live
console output, and per-run controls. Each run with a captured trace exposes a step-through
inspector: the map, per-node observation channels, the teammate belief field, the real per-layer
per-head GAT attention, and the reward decomposed into its terms for the step being viewed.

---

## 8. Parameter reference

Every flag of `scripts/run_train.py`, grouped exactly as `add_argument_group` defines them (the
group title is also the web launch form's section label).

**Run**

| Flag | Default | Meaning |
|---|---|---|
| `--split` | `train/easy` | map split to train on (when --stage is not used) |
| `--stage` | `—` | IR2-style MANUAL curriculum: pick one stage and train only on it (no auto-advance). Overrides --split and --max-episode-steps to the IR2 coupling (easy=train/easy@196 steps, … |
| `--out` | `—` | output run dir. Omit → auto-create runs/<run-name\|run>_<timestamp> so every training gets its own fresh folder (no manual --out each time). |
| `--force` | `off` | overwrite an existing --out directory without asking. Only matters when --out names an existing dir; auto-named runs never collide. |
| `--seed` | `0` | random seed (torch: action sampling, init) |
| `--map-seed` | `—` | Seed the MAP stream too. Default None = fresh OS entropy every run (map diversity), which means two runs with the same --seed still see different maps — fine for training, fatal for an … |
| `--device` | `cuda:0` | torch device (cuda:0 or cpu) |

**Scale & episode**

| Flag | Default | Meaning |
|---|---|---|
| `--total-steps` | `5000000` | total env steps to train for |
| `--n-envs` | `16` | parallel environments |
| `--n-agents` | `1` | Number of cooperative agents per env |
| `--rollout-len` | `128` | rollout length per PPO iteration |
| `--max-episode-steps` | `512` | max steps per episode |
| `--max-travel-px` | `0.0` | Episode travel budget in px (0 = off, step cap only). Truncates once the FARTHEST-travelled robot has covered this much ground. One of our steps is a single lattice hop (<=22.63px) while … |
| `--max-travel-frac` | `0.0` | Travel budget PER MAP, as px travelled per GT-free-pixel (0 = off; overrides --max-travel-px when both are set). Use this for TRAINING: train/difficult spans 3.9x in free area p50->p90, … |
| `--done-mode` | `union` | What ends an episode. 'union' = the TEAM union map hits 99% (legacy MARLauder). 'own' = EVERY agent's OWN map hits 99% — the IR2 rule (their env.check_done), which makes sharing part of … |
| `--minibatches` | `1` | PPO minibatches per epoch (must divide n-envs) |
| `--n-hops` | `6` | Ego-centric encoder window radius. Window side = 2·n_hops + 3 (49 nodes at 2, 121 at 4, 225 at 6). GAT n_layers tied to n_hops (default 6 = 6-layer GAT, 6-hop receptive field). |

**Sensing & communication**

| Flag | Default | Meaning |
|---|---|---|
| `--comm-range` | `120.0` | [comm-model=los] hard Euclidean comm cutoff in pixels (0 = agents never communicate) |
| `--comm-model` | `signal_strength` | Comm model: 'signal_strength' = realistic path-loss radio (walls attenuate, per-episode noise); 'los' = legacy hard range+LOS |
| `--sensor-range` | `80.0` | LiDAR sensor range in pixels (realistic 2D-LiDAR reach) |
| `--ss-thresh` | `-70.0` | [comm-model=signal_strength] rx sensitivity (dBm): connect iff received power > this. Lower = longer comm range |
| `--no-comm-relay` | `on` | Exchange state only over a DIRECT link. Default is multi-hop relay: with A-B-C, A and C share maps, positions and staleness through B, as IR2 does (connected components of the comm … |
| `--force-full-comm` | `off` | A2 debug: bypass dist/LOS check; every pair communicates every step |
| `--force-full-pos-sharing` | `off` | Debug: persistent teammate-position awareness (positions only, maps still comm-gated) |
| `--force-full-occupancy-sharing` | `off` | H.4 debug: persistent map fusion every step (occupancy synced across agents) |
| `--no-teammate-obs` | `off` | ABLATION: blind the actor to teammates — zeroes agent_scalars [∆M-gate, staleness], feat[4] teammate-proximity potential and feat[6] radar-teammate. Map fusion at comm, rdv reward gate … |

**Curriculum**

| Flag | Default | Meaning |
|---|---|---|
| `--curriculum` | `off` | H.5: train on easy + difficult with ramping mix (0-30% all-easy, 30-60% 70/30, 60-100% 50/50) |
| `--curriculum-gated` | `off` | Performance-gated curriculum (split-SWAP): train on --curriculum-stage-splits one at a time, advancing to the next only when the eval suite score clears --curriculum-gate-score (after … |
| `--curriculum-stage-splits` | `train/easy,train/difficult` | comma-separated split sequence for gated curriculum (easy→hard). Env+buffer rebuilt on each advance |
| `--curriculum-stage-steps` | `196,384` | comma-separated per-stage max episode length (IR2 values: easy=196, difficult=384; bigger maps need longer episodes). Empty = same --max-episode-steps for all stages. Must match … |
| `--curriculum-gate-score` | `0.5` | eval/score threshold to advance to the next curriculum stage |
| `--curriculum-min-stage-iters` | `20` | min iters on a stage before a gated advance is allowed (anti-noise dwell) |
| `--eval-split` | `—` | H.5: eval split for eval-on-ckpt (default = --split or test/complex when curriculum) |
| `--eval-suite-splits` | `—` | comma-separated splits for the eval suite (e.g. train/difficult,test/complex). NOT 'extra': this REPLACES the default single suite on the training split, so listing only test splits … |

**Reward shaping**

| Flag | Default | Meaning |
|---|---|---|
| `--novel-scan-weight` | `1.0` | α_novel: privileged team-union novel-scan credit (v2 core reward) |
| `--rdv-weight` | `1.0` | w: dense RENDEZVOUS reward = w·g·(φ_prev−φ_now), g=surplus gate. At w=1.0 a full-gate approach hop pays 1.0·0.02=0.020 against a 0.015-0.021 step_penalty, i.e. it exactly REBATES the … |
| `--rdv-offer-frac` | `0.15` | Rendezvous gate saturates (g→1) when the map gained since last sync reaches this fraction of the OWN map size AT that sync (relative growth, floored by scan_norm_nodes); also normalizes … |
| `--rdv-clamp-pos` | `off` | Pay only the APPROACH half of the rdv term: Δφ clamped to ≥0, so moving AWAY from the teammate is never taxed. Measured on v15, reward/rdv was −0.20/episode — a standing tax on exactly … |
| `--div-weight` | `0.0` | FRONTIER-DIVERSITY auxiliary ACTOR loss (0 = off, exact no-op). Prices two agents committing to the same work: E[shared discounted frontier mass down the exits their policies pick], … |
| `--comm-idle-pen` | `0.0` | Cost of STAYING in radio contact on a step that delivered no map. Free on the step a sync is actually PAID, and free near the deadline (same budget ramp as --rdv-urgency-mode budget) so … |
| `--rdv-urgency-mode` | `time` | What makes a rendezvous urgent. 'time' (legacy) ramps on steps-since-last-sync, so the gate opens merely because the two have been apart — pulling them together mid-episode, when they … |
| `--rdv-urgency-start` | `0.5` | budget mode only: fraction of the episode budget spent before urgency starts ramping (0.5 = explore for the first half, then meeting becomes progressively worth more). Ignored when … |
| `--rdv-urgency-T` | `200.0` | Steps of separation at which the rendezvous urgency nudge saturates. ALSO the normalizer of the staleness ACTOR OBS (was max_episode_steps, which made one step worth 0.0005 at T=2048 and … |
| `--completion-bonus` | `10.0` | Terminal payout when the done criterion fires. Under --done-mode own this is the ONLY term paying for the actual objective, and it had no flag at all before v16. Raise it if … |
| `--step-penalty` | `0.015` | Per-axial-step movement cost (diagonal costs ·√2), charged per lattice-edge length. The direct price of hesitation: raise it to buy directness, at the risk of the agent preferring to … |
| `--sync-weight` | `0.0` | ζ_g: SYNC-EVENT reward = ζ_g·(give + ρ·recv)/scan_norm_nodes, paid on the RISING EDGE of comm only, and only ≥ --sync-min-gap steps after the last paid sync. give = \|my map \ his map\| … |
| `--sync-recv-ratio` | `0.5` | ρ: recv is paid at ρ·ζ_g so BOTH agents gain from meeting (else the map-poor one evades while the rich one chases), while give stays dominant so free-riding on recv doesn't pay |
| `--sync-min-gap` | `32` | Steps since the last PAID sync required for a contact to pay again. Kills comm-boundary flicker; the contact still FUSES, only the payment is suppressed |
| `--revisit-pen` | `0.05` | γ: revisit penalty per step (graduated by recency) |
| `--revisit-window` | `16` | W: revisit lookback steps (8→16 2026-07-15: freshly-scanned trail stays hot longer) |
| `--stall-pen` | `0.1` | δ_stall: heavy penalty for standing still (no net displacement this step) |
| `--stall-streak-beta` | `0.5` | v0.9 cumulative stall: consecutive stalls multiply δ_stall by 1+β·(streak−1), clamped to --stall-streak-cap. 0 disables |
| `--stall-streak-cap` | `4.0` | v0.9: max multiplier on δ_stall for consecutive stalls |
| `--revisit-streak-beta` | `0.5` | v0.9 cumulative revisit: landings on recent (age<W) nodes multiply the graduated revisit penalty by 1+β_rev·(streak−1), UNCAPPED. 0 disables |
| `--revisit-streak-decay` | `0.5` | v0.9.1: a NON-recent landing subtracts this from the revisit streak instead of zeroing it — one high-age hop can't launder the debt; working it off takes a sustained run on new/old ground |
| `--revisit-streak-cap` | `inf` | Max multiplier on the graduated revisit penalty (mirrors --stall-streak-cap). Default inf = legacy uncapped. Measured on v10: streak peaks at 89 → ×45 → 4.05 reward/step and a −44 … |
| `--radar-gamma` | `0.92` | RADAR feat[5/6] per-hop discount beyond the ego-window horizon. 0.92 mutes frontiers ~45+ hops out (0.4%/node); 0.97 keeps them visible (~8% with --radar-util-norm 3) |
| `--radar-util-norm` | `8.0` | RADAR b_util normalization divisor (lower = far frontier mass squashed less) |
| `--belief-mode` | `uniform` | teammate-position belief model used post-comm-break: 'uniform' geodesic ball (old default) vs 'pathfront' two-phase hypothesis model |
| `--pf-frontier-min-unknown` | `1` | pathfront: minimum UNKNOWN 8-neighbours for a node to count as an opening. Guard only — the real gate is --pf-frontier-min-util. Was 4, which dropped large openings the observer had … |
| `--pf-frontier-min-util` | `1e-06` | pathfront: minimum PRE-diffusion utility seed (util_raw = ribbon x volume) for a node to count as an opening. This is what removes wall-adjacent nodes whose 'unknown' is unreachable, and … |
| `--radar-team-source` | `lkp` | feat[6] RADAR teammate source beyond the ego window: 'lkp' (old default) decays a point at each teammate's last-known node; 'belief' mass-transports the belief FIELD itself (same gamma_r … |

**Model ablations & warm-start**

| Flag | Default | Meaning |
|---|---|---|
| `--gru` | `off` | Enable GRU temporal memory in actor+critic. Default OFF: the model runs feed-forward (both GRUCells bypassed) |
| `--no-gru` | `off` | Force GRU OFF (redundant with the default; kept for back-compat / explicitness). Overrides --gru |
| `--no-gat-actor` | `off` | ABLATION: VF-only actor — steers from the analytic value-field (+prev_action/agent_scalars) only; curr_emb zeroed, pointer replaced by actor_head(h)+w_vf·vf. GAT still runs for the CTDE … |
| `--no-gat` | `off` | ABLATION: NO GAT AT ALL — encoder never run. Actor as --no-gat-actor (VF-only); critic embedding = masked mean⊕max of raw window node features projected to d (+ critic_global). Big … |
| `--vf-gamma` | `0.97` | Value-field per-hop discount: V_k = Σ γ^hops·utility over the BF branch leaving through neighbor k (max-normalized to [0,1], actor obs + pointer logit bias) |
| `--init-ckpt` | `—` | Warm-start: load model + value-norm from this .pt at startup (optimizer stays fresh). Use to relaunch a new stage (easy→difficult) at a different --n-envs in a fresh process (avoids the … |

**Eval scoring weights**

| Flag | Default | Meaning |
|---|---|---|
| `--score-w-imbalance` | `0.5` | eval/score weight on NORMALIZED contrib_imbalance (equity; D2: now on [0,1] imb so equity is a first-class term, not a free rider) |
| `--score-w-overlap` | `0.25` | eval/score weight on sensing_overlap (redundant sensing) |
| `--score-w-idle` | `0.25` | eval/score weight on idle_rate_max (laziest agent idle-step fraction) → selects for BOTH agents actively exploring (no idle/turn-taking) |

**PPO / learning**

| Flag | Default | Meaning |
|---|---|---|
| `--lr` | `0.0003` | learning rate |
| `--sync-weight-m-scale` | `0.0` | Scale the sync bonus by (2/M)^THIS. 0 = off (exact no-op at every M). Identically 1 at M=2 for any exponent, so M=2 history cannot move. MEASURED motivation: v16 M=2 vs v19 M=4 realized … |
| `--ent-coef` | `0.01` | entropy bonus coefficient |
| `--diag-grad` | `off` | Log train/g_pg, train/g_ent and their ratio: \|\|grad\|\| of the policy-gradient term vs of the entropy bonus, measured with two extra backward passes on one chunk per iteration. Comparing … |
| `--clip-eps` | `0.15` | PPO clip ε (≤0.2; 0.15 default — this task is more non-stationary than the paper's benchmarks) |
| `--k-epochs` | `4` | PPO epochs per rollout (keep low: intra-episode obs shift + dense shaping = high non-stationarity) |
| `--max-grad-norm` | `2.0` | gradient clip norm (paper 10.0; 2.0 here — dense shaping spikes gradients) |
| `--gae-lambda` | `0.95` | GAE λ |
| `--gamma` | `0.99` | discount factor |
| `--vf-coef` | `0.5` | value loss weight |
| `--tbptt-steps` | `16` | TBPTT chunk length |

**Runtime & checkpointing**

| Flag | Default | Meaning |
|---|---|---|
| `--compile` | `off` | torch.compile encoder (CUDA only) |
| `--no-milestone-ckpt` | `off` | Disable the automatic 20/40/60/80/100% checkpoints. Use with the web dashboard's on-demand 'checkpoint + eval' button to avoid useless ckpts. |
| `--eval-on-ckpt` | `off` | Emit eval GIFs + inspector traces at each milestone checkpoint (20/40/60/80/100%; --eval-n-maps per milestone) |
| `--eval-every` | `10` | Iterations between eval-suite ticks. The suite is 32 maps x full episodes on ONE env and renders nothing: measured on v19 (M=4, 768 steps) it costs ~19 min, i.e. 27% of wall time at 10. … |
| `--eval-steps` | `-1` | G.2: episode length for eval-on-ckpt GIFs/traces. -1 = same as --max-episode-steps |
| `--trace-steps` | `512` | HARD CAP on the episode length of the milestone GIF + inspector trace (NOT the eval suite, which still runs full --eval-steps episodes and is what picks ckpt_best). eval/trace.py builds … |
| `--eval-n-maps` | `2` | GIFs + decision traces per milestone |
| `--eval-map-idx` | `-1` | fixed eval map (-1 = random each milestone) |

**Weights & Biases**

| Flag | Default | Meaning |
|---|---|---|
| `--wandb` | `off` | log metrics to Weights & Biases |
| `--wandb-project` | `marlauder` | W&B project |
| `--wandb-entity` | `—` | W&B entity |
| `--wandb-group` | `—` | W&B group |
| `--wandb-run-name` | `—` | W&B run name (also seeds the auto run-dir name) |
| `--wandb-mode` | `online` | W&B mode |
| `--wandb-tags` | `[]` | W&B tags |

### Hardcoded knobs (not on the CLI)

Edit the dataclass to change these.

**`env.explorer.EnvCfg`**

| Name | Default | Effect |
|---|---|---|
| `nr` | `16` | Lattice spacing (px). `N_max` scales as `(H/nr)·(W/nr)` |
| `n_rays` | `720` | LiDAR rays per scan |
| `utility_range_px` | `30` | Diffusion horizon for the info-gain utility |
| `visit_age_window` | `16` | Recency horizon for feat[3] |
| `num_sim_steps` | `5` | LiDAR sub-steps per high-level step |
| `flood_max_iters` | `200` | Reachability flood-fill cap |
| `done_explored_thresh` | `0.99` | Termination threshold |
| `comm_los_samples` | `40` | Samples along the a→b segment for the comm check |
| `scan_norm_nodes` | `50.0` | Dense-reward normalizer (≈ one sensor disk of nodes) |
| `rdv_frac_max/min/b0` | `0.60 / 0.10 / 0.15` | The decaying required-surplus fraction (§4) |
| `rdv_urgency_weight` | `0.25` | Cap on the staleness/budget nudge to the gate |
| `belief_absorb_gain`, `belief_beta_max`, `belief_diffuse_lambda` | `1.0`, `0.9`, `0.5` | Pathfront phase-2 absorbing diffusion |
| `pf_max_frontiers` | `6` | Hypothesis cap at comm break |
| `ss_*` | IR2 scale | Path-loss parameters: `ss_p_t=-20`, `ss_pl_o=31`, `ss_dist_o=35`, `ss_gamma=2`, `ss_gamma_obst=4`, shadowing `X_g,K ~ U[0,13]` |

**`train.driver.TrainCfg`**

| Name | Default | Effect |
|---|---|---|
| `d_hidden` | `128` | Encoder and GRU width |
| `n_heads` | `4` | GAT heads (must divide `d_hidden`) |
| `lr_critic` | `5e-4` | Faster than the actor: the value target is highly non-stationary here |
| `n_layers` | tied to `n_hops` | Set in `_normalize_cfg` so the receptive field covers the window |

**`train.mappo.MAPPOCfg`**: `clip_vloss=True` (paper §3.3) and `huber_delta=10.0` (paper Table 7);
`0.0` selects plain squared error.

### Two names that are historical

`EnvCfg.guidepost_iters` and `guidepost_path_max` no longer have anything to do with a guidepost —
that analytic component was removed. They are now plain Bellman-Ford loop bounds and are still very
much live. The names are kept because they are **persisted in checkpoints**: renaming them would
make `from_ckpt_dict` silently fall back to the dataclass defaults for every existing run.

---

## 9. Invariants

- Encoder weights are shared between actor and critic; both gradients flow back into it.
- The actor is decentralized: each agent sees only its own ego window and its own scalars.
- The critic is count-invariant (mean⊕max pooling), so the same weights serve any M — this is what
  makes the M=4 lift zero-shot.
- Advantages are per-agent, against a single shared value baseline; the value target is the team mean.
- Hidden states are zeroed at episode boundaries via a `(1 − done)` mask — only relevant under `--gru`.
- All observation tensors stay on the GPU; there is no host round-trip during a rollout.
- Edge length is `NR` axial and `NR·√2` diagonal, in every Bellman-Ford call and in the step penalty.
- The encoder is called exactly once per TBPTT chunk.
- `act()` (rollout, sampling) and `evaluate_step_from_enc()` (update, replay) must compute the same
  thing in two regimes.
