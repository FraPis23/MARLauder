# MARLauder

GPU-vectorized multi-agent reinforcement learning for autonomous map exploration.

Per-agent occupancy maps, intermittent line-of-sight communication, 8-neighbor lattice graph, masked GAT actor-critic with shortest-path guidepost, MAPPO training. Designed to run end-to-end on a single GPU with no CPU roundtrip during the rollout.

For deep architectural detail, see [DOCS.md](DOCS.md). For session-by-session design notes, see [dev_log.md](dev_log.md).

---

## Features

- GPU-resident simulation: Warp LiDAR + torch graph build + frontier detection + Bellman-Ford on GPU, all batched across N envs.
- Multi-agent (1..M) with intermittent communication: agents fuse occupancy maps when within `comm_range_px` AND with clear line of sight on ground truth.
- Per-agent occupancy maps `[N, M, H, W]` stored flat for Warp 3-dim kernel compatibility.
- 8-neighbor regular lattice graph (NR=16 px → ~1200 nodes on 480×640), flood-fill reachability, collision-checked edges.
- **Ego-centric GAT encoder** (configurable `n_hops` window) — restricts attention to a `(2·n_hops+3)²` window around curr.
- **Strategic frontier-attention head** — per-agent MHA over top-K=16 global frontier candidates with BF-distance, teammate-distance, comm-gap, yield, and joint-distribution (`team_alt_score`) features. Gumbel-ST discrete pick. Replaces the previous hard `guidepost_nbr_bias` hijack.
- **BF-from-curr + BF-from-teammate** for true wall-aware shortest-path candidate distances (warm-started). Plus a floored+learnable `path_bias` soft prior on action logits toward the BF first-hop of the strategic pick (floor keeps it from collapsing).
- **Per-agent set-op reward** (v0.4): per-agent scan-delta, give/recv at rendezvous, overlap penalty (lattice set-op baselined at last meeting), revisit + proximity penalty. Per-agent GAE advantages, shared CTDE value baseline.
- **Debug full-sharing flags** + **ramping curriculum** scaffold. fp16 NaN-guarded action logits.
- Centralized-training-decentralized-execution (CTDE) critic.
- MAPPO with GAE-λ (per-agent advantages with shared V baseline), value normalization, AMP fp16, TBPTT (chunked encoder forward), torch.compile on encoder.
- Per-agent side-by-side eval rendering: each panel shows that agent's own occupancy, frontier, graph, **strategic target + correct BF path**, trail, and green comm-link line drawn whenever the pair is in range.

---

## Quick start

### Prerequisites

- NVIDIA GPU (tested on RTX 4080, RTX 5080).
- CUDA 12.x driver on the host.
- Docker + nvidia-container-toolkit (Compose plugin v2+).

### Build and run the container

```bash
cd MARLauder
docker compose build
docker compose up -d
docker exec -it marlauder bash
```

The container mounts the parent directory at `/workspace` so `MARLauder/` and adjacent repos (e.g. `IR2-Multi-Robot-RL-Exploration/`) are both reachable.

### Smoke train (~1 min)

```bash
PYTHONPATH=. python scripts/run_train.py \
    --split train/easy --total-steps 40000 \
    --n-envs 8 --n-agents 2 \
    --rollout-len 64 --max-episode-steps 64 \
    --out /workspace/MARLauder/runs/smoke
```

Verifies the full pipeline boots, runs a few PPO updates, writes a checkpoint, and emits an eval GIF.

### Full training run

See [DOCS.md §6](DOCS.md) for the canonical command. Set `--rollout-len ≥ --max-episode-steps` so episodes complete inside a rollout (populates `ep_end`). Summary:

```bash
PYTHONPATH=. python scripts/run_train.py \
    --split train/easy \
    --total-steps 5_000_000 \
    --n-envs 64 --n-agents 2 \
    --comm-range 120 \
    --rollout-len 256 --max-episode-steps 256 \
    --minibatches 4 \
    --lr 3e-4 --ent-coef 0.01 \
    --path-bias-floor 1.5 \
    --compile --eval-on-ckpt \
    --out /workspace/MARLauder/runs/run_v04
```

### Eval a checkpoint on N random maps

```bash
PYTHONPATH=. python scripts/eval_final.py \
    /workspace/MARLauder/runs/run_v03/final.pt \
    --split train/easy --n-maps 5 --steps 512
```

GIFs land next to the checkpoint as `eval_map00892.gif`, `eval_map04388.gif`, ...
Each is a side-by-side panel per agent. Architecture is inferred from the checkpoint weights — no need to pass `--d-hidden`/`--n-heads`/`--n-layers`.

Pin specific maps via `--map-idx N [N ...]` (single or list, overrides `--n-maps`/`--seed`), e.g. `--map-idx 9580 1234`. Without `--map-idx`, indices are drawn from `--seed`.

### Eval one specific map by index

```bash
PYTHONPATH=. python scripts/run_eval.py \
    --ckpt /workspace/MARLauder/runs/run_v04/final.pt \
    --split train/easy --map-idx 9580 \
    --n-agents 2 \
    --d-hidden 128 --n-heads 4 --n-layers 2 \
    --steps 256 \
    --out /workspace/MARLauder/runs/run_v04/eval_map9580.gif
```

`--map-idx N` selects map at index `N` (0-indexed). Start positions placed via `_spread_starts_graph` (lattice-adjacent FREE nodes, BFS + segment-clear, same as training). Env cfg (`n_hops`, `top_k`, comm/force flags) is read from the checkpoint via `EnvCfg.from_ckpt_dict`. Add `--force-full-occupancy-sharing` / `--force-full-pos-sharing` to force persistent sharing at eval. Pass `--d-hidden`/`--n-heads`/`--n-layers` to match the trained network.

The rendered amber target/path is the **strategic head's chosen frontier + its correct BF path** — the node the policy actually pursues.

---

## Project structure

```
MARLauder/
├── env/                Simulation: Warp world, sensors, lattice graph, frontier, env loop
│   ├── world_warp.py       Per-agent occupancy maps + LiDAR kernels (Warp)
│   ├── explorer.py         Vectorized env: step, comm check, map fusion, reward
│   ├── graph_lattice.py    8-neighbor graph build + Bellman-Ford guidepost
│   ├── frontier.py         conv2d frontier detector
│   └── maps.py             Map split loading + batched sampling
├── models/             Networks
│   ├── gat.py              MaskedGATLayer + GATEncoder
│   ├── actor_critic.py     MarlActorCritic (shared encoder + actor + CTDE critic)
│   └── value_normalizer.py Welford running mean/var for V normalization
├── train/              MAPPO trainer
│   ├── buffer.py           Rollout buffer + GAE-λ
│   ├── mappo.py            PPO update with chunked encoder forward
│   └── driver.py           TrainCfg, main loop, milestone checkpoints, eval GIF on ckpt
├── eval/               Inference + visualization
│   ├── rollout.py          Deterministic rollout, per-agent frame builder
│   └── render.py           Palette + painters + composite_frame + hstack_frames
├── scripts/            CLI entrypoints
│   ├── run_train.py        Training
│   ├── run_eval.py         Single-map eval
│   ├── eval_final.py       Batch eval on N random maps
│   ├── baseline_random.py  Random-policy sanity baseline
│   └── 0[1-7]_test_*.py    Step-by-step component tests
├── data/               Preprocessed map tensors (memmap + meta.npz per split)
├── docker/             Dockerfile (base: PyTorch CUDA 12.8)
├── docker-compose.yml
├── DOCS.md             Deep reference (modules, data flow, parameters, diagnostics)
├── dev_log.md          Session-by-session design notes
└── README.md           This file
```

---

## Training output

A single iteration prints a dense one-liner:

```
[train] iters=152  steps/iter=16384  total≈2,490,368
[it    1/152] ep_end=  9.3%(ended= 64)  pg=-0.0123  v=0.7521  ent=2.063  kl=+0.0041  clip=8.3%  sps=540(540avg) coll=687 upd=2514
[it    2/152] ep_end= 11.1%(ended= 64)  pg=-0.0118  v=0.7204  ent=2.041  kl=+0.0045  clip=9.1%  sps=560(550avg) coll=690 upd=2520
...
[ckpt] /workspace/MARLauder/runs/run_v04/ckpt_025.pt
[eval] /workspace/MARLauder/runs/run_v04/eval_ckpt_025_m0.gif  map=12  final_explored=42.1%  frames=256
...
[done] /workspace/MARLauder/runs/run_v04/final.pt
```

Field meanings:

| Field | What it means |
|---|---|
| `ep_end` | Mean explored fraction at the terminal step of all episodes that ENDED this iter. `ended=K` = how many episodes that was. `n/a` until ≥1 completes — set `--rollout-len ≥ --max-episode-steps` so episodes finish inside a rollout. |
| `pg` | PPO policy gradient loss. Small negative (-0.005 to -0.02) is healthy. |
| `v` | Value loss. Drops then plateaus. |
| `ent` | Action-distribution entropy. Should decay smoothly, not collapse. |
| `kl` | KL between old/new policy. Stays < 0.02 with `clip-eps=0.2`. |
| `clip` | Fraction of samples hit by the PPO clip. Healthy: 5–20%. |
| `sps` | Total env steps per wall-clock second (current iter / running average). |
| `coll` / `upd` | Per-phase sps — rollout collection vs PPO update. |

Full diagnostic table including warning signs: [DOCS.md §8](DOCS.md).

Checkpoints are written at 25%, 50%, 75%, 100% of total iterations as `ckpt_025.pt`, ..., `ckpt_100.pt` plus a final `final.pt`. With `--eval-on-ckpt`, two eval GIFs are also produced at each milestone on randomly sampled maps.

---

## Common CLI flags

The most-used `run_train.py` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--split` | `train/easy` | Map split: `train/easy`, `train/difficult`, `test/{complex,corridor,hybrid}` |
| `--out` | `runs/run_default` | Output directory for checkpoints + eval GIFs |
| `--seed` | `-1` | torch RNG (actions, init). Maps use independent fresh-entropy RNG each run |
| `--total-steps` | `5_000_000` | Total environment transitions (`n_envs × rollout_len × iters`) |
| `--n-envs` | `16` | Parallel envs. Must be divisible by `--minibatches` |
| `--n-agents` | `1` | Number of cooperative agents per env |
| `--comm-range` | `120.0` | Communication range in pixels (0 = never communicate) |
| `--rollout-len` | `128` | Steps per PPO update. Set ≥ `--max-episode-steps` to populate `ep_end` |
| `--max-episode-steps` | `512` | Episode truncation |
| `--n-hops` | `2` | Ego-centric encoder window radius |
| `--top-k` | `16` | Strategic-head frontier candidates per agent |
| `--path-bias-floor` | `1.5` | Floor on target-following bias |
| `--yield-scale` / `--overlap-pen` / `--proximity-pen` | `3.0` / `3.0` / `0.005` | Anti-chase knobs |
| `--force-full-pos-sharing` / `--force-full-occupancy-sharing` | off | Debug: perfect teammate positions / synced maps |
| `--curriculum` | off | Ramp easy→difficult mix (needs same-canvas splits) |
| `--compile` | off | `torch.compile` the encoder (~2× update speedup) |
| `--eval-on-ckpt` | off | Emit 2 eval GIFs at each milestone |

Full parameter reference + reward weights: [DOCS.md §5](DOCS.md). Hardcoded knobs: [DOCS.md §11](DOCS.md).

`eval_final.py` infers architecture from the checkpoint weights and reads env cfg from the checkpoint. `run_eval.py` / `eval_final.py` `--seed` defaults to system entropy (different maps each run).

---

## Reward (v0.4)

Per-agent set-op reward, lattice-level, baselined at last comm event between each pair:

```
reward[a] = α_scan · scan_self_delta[a]      # cells I LiDAR-scanned this step (node level)
          + β     · team_delta                # Δunion FREE (cooperation anchor)
          + ζ_give · give[a]                  # NEW cells I bring to teammate at comm
          + ζ_recv · recv[a]                  # NEW cells I receive at comm
          − η_lap  · overlap[a]               # we BOTH scanned same area since last meeting
          − γ      · revisit_pen[a]           # node visited within last W=8 steps
          − ε_prox · proximity_pen[a]         # teammate within sensor_range AND visible
          + 1{terminated} · completion_bonus
          − step_penalty
```

Defaults: `α_scan=1.0, β=0.3, ζ_give=1.5, ζ_recv=0.5, η_lap=3.0, γ=0.02, ε_prox=0.005, completion_bonus=10.0, step_penalty_coef=0.1`.

Each agent gets its **own** scalar reward. GAE computes per-agent advantages with a shared CTDE value baseline. Anti-chase comes from `overlap`/`proximity` penalties plus the strategic head's `cand_own_minus_team` (yield) and `team_alt_score` (joint distribution) features — all smooth, no hard thresholds. A floored+learnable `path_bias` keeps the actor following its chosen target so grid-utility doesn't dominate.

Full reward derivation + decentralization properties: [DOCS.md §4](DOCS.md).

---

## Multi-agent communication

Two agents `i` and `j` can communicate at step `t` if both:

1. `‖pos[i] − pos[j]‖ < comm_range_px`
2. Sampled Bresenham line on ground truth between `pos[i]` and `pos[j]` contains no obstacle cell (default 40 samples).

When the condition holds, on the same step:

- Per-agent occupancy log-odds maps are fused via elementwise **max-magnitude** (`where(|lo_i|≥|lo_j|, lo_i, lo_j)`) — preserves OBSTACLE evidence (negative lo) that a plain `max` would drop. Idempotent across consecutive steps in range.
- `last_known_pos[i][j]` is overwritten with the actual current position of agent `j` (and vice versa).

When out of range, agents drift with their own partial maps. The `last_known_pos` entries become stale until the next rendezvous. Feature `node_feat[..., 5]` exposes a one-hot at the lattice node nearest each known teammate position.

Agent–agent collision is enforced at the env level: if a planned sub-step move would bring two agents within `nr` pixels, both revert to their previous positions. Same as wall collision.

---

## Architecture (brief)

```
GT map → Warp LiDAR per agent → per-agent log-odds → torch occupancy categorical
   → conv2d frontier → graph_lattice.build (nodes + edges + utility integral image)
   → BF-from-curr + BF-from-teammate → extract_topk_candidates (K=16, wall-aware dist)
   → obs dict [N, M, ...]
   → ego-centric GAT encoder → StrategicHead (Gumbel-ST target pick)
   → actor GRU + PointerHead (+path_bias toward target's BF first-hop) → action
                                                  └→ CTDE critic GRU → V(s)
```

Full pipeline and shapes: [DOCS.md §9](DOCS.md).

---

## Roadmap

| Version | Goal | Status |
|---|---|---|
| v0.1 | Single-agent baseline (Warp LiDAR + lattice graph + GAT + MAPPO) | ✓ |
| v0.2 | Terminology cleanup, Bellman-Ford guidepost, diagonal cost, MAPPO speedup | ✓ |
| v0.3 | Multi-agent intermittent communication, per-agent maps, per-agent eval render | ✓ |
| v0.4 | Strategic frontier-attention head (Gumbel-ST), ego-centric encoder, BF-from-curr + BF-from-teammate cand ranking, joint-distribution feature, floored path-bias, per-agent set-op reward, anti-chase signals, debug full-sharing flags, curriculum scaffold | ✓ (current) |
| v0.5 | Curriculum train/easy → train/difficult (scaffold landed; blocked on differing canvas sizes) | Partial |
| v0.6 | Eval suite: per-split curves, TB logger, voluntary-rendezvous reward | Planned |
| v0.7 | ToM teammate-belief module (probabilistic teammate-state encoder, replaces point lkp in cand features) | Planned |

v0.8 (hierarchical L2 graph) explicitly out of scope for this rewrite.
