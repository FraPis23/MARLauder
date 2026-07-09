# MARLauder

GPU-vectorized multi-agent reinforcement learning for autonomous map exploration.

Per-agent occupancy maps, intermittent signal-strength communication, 8-neighbor lattice graph, masked GAT actor-critic with GRU memory and a pointer action head, MAPPO training. A learned policy — no analytic guidepost or hand-picked target: the agent steers purely from the ego-window graph features (local frontier utility + beyond-window "radar" channels) plus two rendezvous scalars. Designed to run end-to-end on a single GPU with no CPU roundtrip during the rollout.

For deep architectural detail, see [DOCS.md](DOCS.md). For session-by-session design notes, see [dev_log.md](dev_log.md).

---

## Features

- GPU-resident simulation: Warp LiDAR + torch graph build + frontier detection + Bellman-Ford on GPU, all batched across N envs (agents folded into the batch dim — one build for N·M).
- Multi-agent (1..M) with intermittent communication: agents fuse occupancy maps when they can connect. Default comm model is a **signal-strength path-loss radio** (walls attenuate, per-episode shadowing noise), with a legacy hard line-of-sight model available.
- Per-agent occupancy maps `[N, M, H, W]` stored flat for Warp 3-dim kernel compatibility.
- 8-neighbor regular lattice graph (NR=16 px), flood-fill reachability, collision-checked edges, diagonal cost √2.
- **Ego-centric GAT encoder** (configurable `n_hops` window) — restricts attention to a `(2·n_hops+3)²` window around the agent's current node. Per-head learnable temperature (A1) and per-head structural feature-bias groups (A2) specialize heads on explore / rendezvous / recency / beyond-window steering.
- **RADAR beyond-window channels** — `build_radar` compresses the known world *beyond* the ego window onto the geodesic horizon nodes (obstacle-aware, routed down the BF parent chain), giving a feed-forward-friendly heading toward far exploration mass and far teammates. Replaces the removed analytic guidepost.
- **Dense rendezvous economy** — `rdv_dense` reward (telescoping geodesic approach to the owed teammate, gated by the map-surplus you owe) plus two actor observations `[∆M surplus-gate, staleness]` that let the policy DECIDE when to meet. No separation penalty — the privileged novel-scan credit already spreads agents.
- **Per-agent privileged novel-scan reward** — an agent is paid only for cells new to the *team union*, so splitting up is the highest-paying policy by construction. Per-agent GAE advantages, count-invariant (mean⊕max pooled) CTDE value baseline.
- **Feed-forward actor + critic by default**; optional GRU temporal memory via `--gru` (both GRUCells still exist for checkpoint compatibility).
- Debug full-sharing / full-comm flags + gated easy→difficult curriculum scaffold. bf16 AMP, TBPTT chunked encoder forward, optional `torch.compile`.
- Per-agent side-by-side eval rendering: each panel shows that agent's own occupancy, frontier, graph, trail, and a green comm-link line drawn whenever the pair is connected. Interactive step-through inspector with REAL per-layer/head GAT attention.

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

The container mounts the parent directory at `/workspace` so `MARLauder/` and adjacent repos are both reachable.

### Smoke train (~1 min)

```bash
PYTHONPATH=. python scripts/run_train.py \
    --split train/easy --total-steps 40000 \
    --n-envs 8 --n-agents 2 \
    --rollout-len 64 --max-episode-steps 64 \
    --out /workspace/MARLauder/runs/smoke
```

Verifies the full pipeline boots, runs a few PPO updates, writes a checkpoint, and emits an eval GIF.

### Full training pipeline

The canonical two-stage run is `pipeline_rdv.sh` (easy → difficult warm-start). Summary of one stage:

```bash
PYTHONPATH=. python scripts/run_train.py \
    --split train/easy \
    --total-steps 2_000_000 \
    --n-envs 32 --n-agents 2 \
    --rollout-len 256 --max-episode-steps 128 \
    --n-hops 6 --tbptt-steps 8 --minibatches 1 --k-epochs 4 \
    --rdv-weight 0.10 \
    --eval-on-ckpt \
    --out /workspace/MARLauder/runs/run_easy
```

Set `--rollout-len ≥ --max-episode-steps` so episodes complete inside a rollout (populates `ep_end`). See [DOCS.md §6](DOCS.md) for the canonical command and the VRAM/launch geometry.

### Eval a checkpoint on N random maps

```bash
PYTHONPATH=. python scripts/eval_final.py \
    /workspace/MARLauder/runs/run_easy/final.pt \
    --split train/easy --n-maps 5 --steps 512
```

GIFs land next to the checkpoint as `eval_map00892.gif`, ... Each is a side-by-side panel per agent. Architecture (d, heads, layers, agents, `use_gru`) is inferred from the checkpoint weights — no need to pass it. Pin maps via `--map-idx N [N ...]`.

### Best-checkpoint selection

`eval_best.py` scores every milestone checkpoint on a fixed 32-map suite and writes `ckpt_best.pt`. The web inspector exposes a "Find best ckpt" action.

---

## Project structure

```
MARLauder/
├── env/                Simulation: Warp world, sensors, lattice graph, frontier, env loop
│   ├── world_warp.py       Per-agent occupancy maps + LiDAR kernels (Warp)
│   ├── explorer.py         Vectorized env: step, comm check, map fusion, reward, obs build
│   ├── graph_lattice.py    8-neighbor graph build + Bellman-Ford + build_radar
│   ├── teammate_belief.py  Teammate-state belief scaffold (last-known pos + staleness)
│   ├── frontier.py         conv2d frontier detector
│   └── maps.py             Map split loading + batched sampling
├── models/             Networks
│   ├── gat.py              MaskedGATLayer (A1 temp + A2 feat-bias) + GATEncoder
│   ├── actor_critic.py     MarlActorCritic (shared encoder + GRU actor/critic + PointerHead)
│   └── value_normalizer.py Welford running mean/var for V normalization
├── train/              MAPPO trainer
│   ├── buffer.py           Rollout buffer + GAE-λ
│   ├── mappo.py            PPO update with chunked encoder forward + TBPTT
│   └── driver.py           TrainCfg, main loop, milestone checkpoints, eval on ckpt
├── eval/               Inference + visualization
│   ├── rollout.py          Deterministic rollout, per-agent frame builder
│   ├── trace.py            Step-through episode trace for the inspector
│   ├── ckpt_loader.py      Architecture inference from a checkpoint
│   └── render.py           Palette + painters + composite_frame + hstack_frames
├── scripts/            CLI entrypoints
│   ├── run_train.py        Training
│   ├── train_args.py       CLI flag definitions (grouped)
│   ├── run_eval.py         Single-map eval
│   ├── eval_final.py       Batch eval on N random maps
│   ├── eval_best.py        Best-checkpoint selection on the fixed suite
│   └── trace_episode.py    Emit an inspector trace
├── viz/                inspector.html (attention + reward step-through), index.html
├── docs/               architecture.html (pipeline diagram)
├── data/               Preprocessed map tensors (memmap + meta.npz per split)
├── docker/             Dockerfile (base: PyTorch CUDA)
├── DOCS.md             Deep reference (modules, data flow, parameters, diagnostics)
├── dev_log.md          Session-by-session design notes
└── README.md           This file
```

---

## Training output

A single iteration prints a dense one-liner:

```
[it    1/152] ep_end=  9.3%(ended= 64)  pg=-0.0123  v=0.7521  ent=2.063  kl=+0.0041  clip=8.3%  sps=130(130avg) coll=687 upd=2514
```

| Field | What it means |
|---|---|
| `ep_end` | Mean explored fraction at the terminal step of all episodes that ENDED this iter. `ended=K` = how many. `n/a` until ≥1 completes — set `--rollout-len ≥ --max-episode-steps`. |
| `pg` | PPO policy-gradient loss. Small negative (-0.005 to -0.02) is healthy. |
| `v` | Value loss. Drops then plateaus. |
| `ent` | Action-distribution entropy. Should decay smoothly, not collapse. |
| `kl` | KL between old/new policy. Stays < 0.02 with `clip-eps=0.15`. |
| `clip` | Fraction of samples hit by the PPO clip. Healthy: 5–20%. |
| `sps` | Total env steps per wall-clock second (current iter / running average). |
| `coll` / `upd` | Per-phase sps — rollout collection vs PPO update. |

Full diagnostic table: [DOCS.md §8](DOCS.md). Checkpoints written at 25/50/75/100% as `ckpt_025.pt`..`ckpt_100.pt` + `final.pt`; with `--eval-on-ckpt`, eval GIFs at each milestone.

---

## Common CLI flags

Most-used `run_train.py` flags (full list: `scripts/train_args.py`, grouped by section):

| Flag | Default | Meaning |
|---|---|---|
| `--split` | `train/easy` | Map split: `train/{easy,difficult}`, `test/{complex,corridor,hybrid}` |
| `--out` | auto | Output directory for checkpoints + eval GIFs |
| `--seed` | `0` | torch RNG (actions, init). Maps use independent fresh-entropy RNG each run |
| `--total-steps` | `5_000_000` | Total environment transitions |
| `--n-envs` | `16` | Parallel envs. Must be divisible by `--minibatches` |
| `--n-agents` | `1` | Cooperative agents per env |
| `--rollout-len` / `--max-episode-steps` | `128` / `512` | Steps per PPO update / episode truncation |
| `--minibatches` / `--tbptt-steps` | `1` / `16` | PPO minibatches / truncated-BPTT chunk length |
| `--n-hops` | `6` | Ego-centric encoder window radius (GAT depth ties to it) |
| `--comm-model` | `signal_strength` | `signal_strength` (path-loss radio) or `los` (hard line-of-sight) |
| `--comm-range` | `120.0` | LOS-mode Euclidean cutoff (px). Ignored in signal-strength mode |
| `--sensor-range` / `--ss-thresh` | `80.0` / `-70.0` | LiDAR range (px) / rx sensitivity (dBm) |
| `--novel-scan-weight` | `1.0` | α: privileged team-union novel-scan credit (core reward) |
| `--rdv-weight` / `--rdv-offer-frac` | `0.10` / `0.15` | Dense rendezvous strength / surplus at which the gate saturates |
| `--revisit-pen` / `--revisit-window` / `--stall-pen` | `0.05` / `8` / `0.1` | Anti-loop (graduated) / window / anti-standing-still |
| `--radar-gamma` / `--radar-util-norm` | `0.92` / `8.0` | Beyond-window radar travel-discount / mass normalizer |
| `--gru` | off | Enable GRU temporal memory in actor+critic (default is feed-forward) |
| `--init-ckpt` | none | Warm-start from a checkpoint (stage-2 of the pipeline) |
| `--lr` / `--ent-coef` / `--clip-eps` / `--k-epochs` / `--gae-lambda` / `--gamma` / `--vf-coef` | `3e-4` / `0.01` / `0.15` / `4` / `0.95` / `0.99` / `0.5` | MAPPO knobs |
| `--force-full-comm` / `--force-full-pos-sharing` / `--force-full-occupancy-sharing` | off | Debug: perfect comm / teammate positions / synced maps |
| `--curriculum` / `--curriculum-gated` | off | Ramp easy→difficult (fixed schedule / eval-score-gated) |
| `--compile` / `--eval-on-ckpt` | off | `torch.compile` the encoder / emit eval GIFs at milestones |
| `--wandb` (+ project/entity/group/run-name/mode/tags) | off | Log to Weights & Biases |

Full parameter reference + hardcoded knobs: [DOCS.md §5](DOCS.md), [DOCS.md §11](DOCS.md).

---

## Reward

Per-agent, lattice-level, in map-independent units (`scan_norm_nodes=50` ≈ one sensor disk):

```
reward[a] = α · novel_scan[a]              # cells I scanned that are NEW TO THE TEAM UNION (privileged)
          − γ · revisit_pen[a]             # node visited within last W=8 steps (graduated by recency)
          − δ_stall · stall_pen[a]         # no net displacement this step
          + 1{explored ≥ 99%} · completion_bonus
          − step_penalty                   # axial = step_cost, diagonal = ·√2
          + w · g · (φ_prev − φ_now)       # rdv_dense: net geodesic approach to the owed teammate, gated by surplus g
```

Defaults: `α=1.0, γ=0.10, δ_stall=0.1, completion_bonus=10.0, step_penalty_coef=0.015, w(rdv)=0.10, rdv_offer_frac=0.15`.

**Privileged novel-scan credit (IR2-style `r_f`)**: an agent earns scan reward only for cells **new to the team union map** — a follower scanning a leader's wake earns exactly 0, so splitting up is the highest-paying policy by construction. Privileged, training-only (CTDE); the deployed actor never sees the union. There is **no separation / proximity penalty** — the design constraint is that novel-scan does the spreading, so agents never "fear the only path".

**Dense rendezvous (`rdv_dense`)**: telescoping toward the owed teammate's FIXED last-known position, gated by **relative map growth** `g = clamp(∆M / (rdv_offer_frac · own_map_at_last_sync), 0, 1)` — the cells I mapped that the teammate I owe most still lacks, as a fraction of the map I already had when we last met (floored by one sensor disk). So `g→1` means "I have grown my known map by `rdv_offer_frac` since we last met → I hold enough NEW content to be worth sharing". Farm-safe: oscillation cancels, and at comm `∆M→0` kills the gate so the last-known-position jump is never paid. The SAME gate `g` (plus a normalized `staleness`) is fed to the actor as `agent_scalars` so the policy decides *when* to rendezvous — the reward and the observation share the trigger.

Full reward derivation + decentralization properties: [DOCS.md §4](DOCS.md).

---

## Multi-agent communication

Default `comm_model = signal_strength`: a log-distance path-loss radio. The segment between two agents is split into free vs obstacle length; walls **attenuate** (γ_obst=4) rather than hard-block; per-episode shadowing noise is resampled each reset. Agents connect iff received power `P_R = P_T − PL > ss_thresh`. A legacy `los` model (hard Euclidean `comm_range_px` + Bresenham LOS on ground truth) is available.

On connection, on the same step:

- Per-agent occupancy log-odds maps are fused via elementwise **max-magnitude** — preserves OBSTACLE evidence a plain `max` would drop. Idempotent while connected.
- `last_known_pos[i][j]` is overwritten with the actual current position of agent `j` (and vice versa); the pair's staleness timer resets.

When out of range, agents drift with their own partial maps; `last_known_pos` goes stale. The teammate's direction — both in-window (`teammate_pot`) and beyond-window (radar `b_teammate`) — reaches the policy through the graph features.

Agent–agent collision is enforced env-side: if a sub-step move would bring two agents within `nr` px, the lower-priority agent (higher per-episode `_collision_key`) yields and holds; the winner reverts only if still blocked. Priority is re-drawn each episode (no systematic role bias; decentralized via a shared per-episode seed).

---

## Architecture (brief)

```
GT map → Warp LiDAR per agent → per-agent log-odds → torch occupancy
   → conv2d frontier → graph_lattice.build (nodes + edges + utility integral image)
   → BF-from-curr + BF-from-teammate → build_radar (beyond-window channels) → extract_local_window
   → obs dict [N, M, ...]
   → ego-centric GAT encoder (A1 temp + A2 head feat-bias)
   → actor: (curr_emb ‖ prev_action ‖ agent_scalars[∆M-gate, staleness]) → GRU → PointerHead → action
   → critic (CTDE): mean⊕max pool over agents ‖ critic_global[7] → GRU → V(s)
```

Node features `F_IN = 7`: `0 x_rel, 1 y_rel, 2 utility, 3 age, 4 teammate_pot, 5 radar-util, 6 radar-teammate`.
GAT A2 head feature-bias groups: `H0 [2,5]` explore · `H1 [4,6]` rendezvous · `H2 [3]` recency · `H3 [5,6]` beyond-window steering.
Full pipeline and shapes: [DOCS.md §9](DOCS.md).

---

## Roadmap

| Version | Goal | Status |
|---|---|---|
| v0.1 | Single-agent baseline (Warp LiDAR + lattice graph + GAT + MAPPO) | ✓ |
| v0.2 | Bellman-Ford guidepost, diagonal cost, MAPPO speedup | ✓ (guidepost later removed) |
| v0.3 | Multi-agent intermittent communication, per-agent maps, per-agent eval render | ✓ |
| v0.4–v0.7 | StrategicHead / analytic-target experiments | ✓ then **removed** (analytic target/guidepost/path-bias/strategic-head all deleted; a genuinely learned policy) |
| v0.8 | Critic mean⊕max pooling, analytic target & guidepost REMOVED (F_IN 8→7), dense rendezvous reward + `agent_scalars`, realistic signal-strength comm, ego-window radar channels | ✓ (current) |
| — | Perf: batched-agent env build + bf16 (+41% sps); best-ckpt selection; entity-split GAT heads | ✓ |
| next | Radar-gain fix (mute far-field at long range), rendezvous under-experience with M=2, M>2 warm-start | In progress |

See [dev_log.md](dev_log.md) for the full session history and the current open problems.
