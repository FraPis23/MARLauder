# MARLauder

GPU-vectorized multi-agent reinforcement learning for autonomous map exploration.

Per-agent occupancy maps, intermittent line-of-sight communication, 8-neighbor lattice graph, masked GAT actor-critic with shortest-path guidepost, MAPPO training. Designed to run end-to-end on a single GPU with no CPU roundtrip during the rollout.

For deep architectural detail, see [DOCS.md](DOCS.md). For session-by-session design notes, see [dev_log.md](dev_log.md).

---

## Features

- GPU-resident simulation: Warp LiDAR + torch graph build + frontier detection + Bellman-Ford guidepost, all batched across N envs.
- Multi-agent (1..M) with intermittent communication: agents fuse occupancy maps when within `comm_range_px` AND with clear line of sight on ground truth.
- Per-agent occupancy maps `[N, M, H, W]` stored flat for Warp 3-dim kernel compatibility, exposed as `team_occupancy()` (union) and per-agent views.
- 8-neighbor regular lattice graph (NR=16 px → ~1200 nodes on 480×640), flood-fill reachability, collision-checked edges.
- Masked GAT encoder (2 layers, d=128, 4 heads) → GRUCell actor → PointerHead over K=8 directions.
- Centralized-training-decentralized-execution (CTDE) critic: shared encoder + per-agent `curr_emb` concatenation → joint state value V(s).
- Bellman-Ford guidepost on GPU: shortest path with axial cost NR, diagonal cost NR·√2; feeds the policy as both a node feature and a logit-bias prior on the next neighbor.
- MAPPO with GAE-λ, value normalization, AMP fp16, TBPTT (chunked encoder forward), torch.compile on encoder.
- Per-agent side-by-side eval rendering: each panel shows that agent's own occupancy, frontier, graph, target, trail, and green comm-link line drawn whenever the pair is in range.

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
    --rollout-len 64 --max-episode-steps 128 \
    --out /workspace/MARLauder/runs/smoke
```

Verifies the full pipeline boots, runs a few PPO updates, writes a checkpoint, and emits an eval GIF.

### Full training run

See [DOCS.md §6](DOCS.md) for the canonical command. Summary:

```bash
PYTHONPATH=. python scripts/run_train.py \
    --split train/easy \
    --total-steps 5_000_000 \
    --n-envs 32 --n-agents 2 \
    --comm-range 120 \
    --rollout-len 128 --max-episode-steps 512 \
    --minibatches 4 \
    --lr 3e-4 --ent-coef 0.01 \
    --compile --eval-on-ckpt \
    --seed 0 \
    --out /workspace/MARLauder/runs/run_v03
```

### Eval a checkpoint on N random maps

```bash
PYTHONPATH=. python scripts/eval_final.py \
    /workspace/MARLauder/runs/run_v03/final.pt \
    --split train/easy --n-maps 5 --steps 512
```

GIFs land next to the checkpoint as `eval_map00892.gif`, `eval_map04388.gif`, ...
Each is a side-by-side panel per agent. Architecture is inferred from the checkpoint weights — no need to pass `--d-hidden`/`--n-heads`/`--n-layers`.

### Eval one specific map with explicit architecture

```bash
PYTHONPATH=. python scripts/run_eval.py \
    --ckpt /workspace/MARLauder/runs/run_v03/final.pt \
    --split test/complex --map-idx 7 \
    --d-hidden 128 --n-heads 4 --n-layers 2 \
    --steps 512 \
    --out /workspace/MARLauder/runs/run_v03/eval_map7.gif
```

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
[load] split=train/easy  ngt=2451  H=480 W=640
[train] iters=1220  steps/iter=4096  total≈4,997,120
[it    1/1220] explored avg= 5.4% end= 9.3%  pg=-0.0123  v=0.7521  ent=2.063  kl=+0.0041  clip=8.3%  sps=178(178avg) coll=512 upd=240
[it    2/1220] explored avg= 6.1% end=10.1%  pg=-0.0118  v=0.7204  ent=2.041  kl=+0.0045  clip=9.1%  sps=183(180avg) coll=510 upd=241
...
[ckpt] /workspace/MARLauder/runs/run_v03/ckpt_025.pt
[eval] /workspace/MARLauder/runs/run_v03/eval_ckpt_025_m0.gif  map=12  final_explored=42.1%  frames=287
[eval] /workspace/MARLauder/runs/run_v03/eval_ckpt_025_m1.gif  map=84  final_explored=39.7%  frames=265
...
[done] /workspace/MARLauder/runs/run_v03/final.pt
```

Field meanings:

| Field | What it means |
|---|---|
| `explored avg` / `end` | Mean explored fraction during the rollout and at the final step. Watch this grow with iters. |
| `pg` | PPO policy gradient loss. Small negative (-0.005 to -0.02) is healthy. |
| `v` | Value loss. Drops then plateaus around 0.3–0.8. |
| `ent` | Entropy of the action distribution. Should decay smoothly, not collapse. |
| `kl` | KL divergence between old and new policy. Stays < 0.02 with `clip-eps=0.2`. |
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
| `--seed` | `0` | RNG seed for split sampling, policy initialization, episode randomness |
| `--total-steps` | `5_000_000` | Total environment transitions (`n_envs × rollout_len × iters`) |
| `--n-envs` | `16` | Parallel envs. Must be divisible by `--minibatches` |
| `--n-agents` | `1` | Number of cooperative agents per env |
| `--comm-range` | `120.0` | Communication range in pixels (0 = never communicate) |
| `--max-episode-steps` | `512` | Episode truncation |
| `--compile` | off | `torch.compile` the encoder (~2× update speedup) |
| `--eval-on-ckpt` | off | Emit 2 eval GIFs on random maps at each milestone |

Full parameter reference: [DOCS.md §5](DOCS.md). For knobs not exposed on the CLI (lattice spacing, GAT width, PPO clip, etc.), see [DOCS.md §11](DOCS.md).

`eval_final.py` reads architecture (n_agents, d_hidden, n_heads, n_layers) directly from the checkpoint state dict. `--seed` defaults to system entropy (different maps each run).

---

## Reward

```
reward_team = max(0, Δ(union FREE) / total_free)         # discovery delta — shared across agents
            + 1{terminated} · completion_bonus            # one-shot at episode end (default +10.0)
            − step_penalty_coef / max_episode_steps       # per-step time pressure (default 0.1 / max_steps)
```

The discovery term rewards expansion of the team's collective known-free region. The completion bonus fires when `explored_rate ≥ done_explored_thresh` (default 0.99). The step penalty creates time pressure: with defaults at `max_episode_steps=512`, each step costs ~0.000195, totalling 0.1 over a full unterminated episode.

All M agents receive the same scalar reward (team reward).

---

## Multi-agent communication

Two agents `i` and `j` can communicate at step `t` if both:

1. `‖pos[i] − pos[j]‖ < comm_range_px`
2. Sampled Bresenham line on ground truth between `pos[i]` and `pos[j]` contains no obstacle cell (default 40 samples).

When the condition holds, on the same step:

- Per-agent occupancy log-odds maps are fused via elementwise `max(lo_i, lo_j)`. Idempotent across consecutive steps in range (no double-counting).
- `last_known_pos[i][j]` is overwritten with the actual current position of agent `j` (and vice versa).

When out of range, agents drift with their own partial maps. The `last_known_pos` entries become stale until the next rendezvous. Feature `node_feat[..., 5]` exposes a one-hot at the lattice node nearest each known teammate position.

Agent–agent collision is enforced at the env level: if a planned sub-step move would bring two agents within `nr` pixels, both revert to their previous positions. Same as wall collision.

---

## Architecture (brief)

```
GT map → Warp LiDAR per agent → per-agent log-odds → torch occupancy categorical
         → conv2d frontier → graph_lattice.build (nodes + edges + utility integral image)
         → graph_lattice.build_guidepost (Bellman-Ford from curr to argmax-utility target)
         → obs dict [N, M, ...] → encoder (shared GAT) → actor GRU + PointerHead → action
                                                       └→ CTDE critic GRU → V(s)
```

Full pipeline and shapes: [DOCS.md §9](DOCS.md).

---

## Roadmap

| Version | Goal | Status |
|---|---|---|
| v0.1 | Single-agent baseline (Warp LiDAR + lattice graph + GAT + MAPPO) | ✓ |
| v0.2 | Terminology cleanup, Bellman-Ford guidepost, diagonal cost, MAPPO speedup | ✓ |
| v0.3 | Multi-agent intermittent communication, per-agent maps, per-agent eval render | ✓ (current) |
| v0.4 | Target diversification, reward shaping, ego-centric subgraph | In progress |
| v0.5 | Curriculum train/easy → train/difficult | Planned |
| v0.6 | Eval suite: per-split curves, TB logger, milestone GIF auto-generation | Planned |
| v0.7 | ToM teammate-belief module (probabilistic teammate-state encoder) | Planned |

v0.8 (hierarchical L2 graph) explicitly out of scope for this rewrite.
