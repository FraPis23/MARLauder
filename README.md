# MARLauder

GPU-resident multi-agent reinforcement learning for cooperative map exploration under
intermittent communication.

A team of robots with 2D LiDAR explores an unknown occupancy map. Each robot keeps its **own**
map and its **own** belief about where its teammates are; they exchange maps only when a
realistic path-loss radio actually connects them. The policy is genuinely learned — there is no
analytic frontier assignment, no planner, no hand-picked target — and the whole simulator lives on
the GPU, so training runs end to end with no host round-trip inside a step.

---

## What is in here

- **A learned, decentralized exploration policy.** An ego-centric masked-GAT encoder feeds a
  pointer actor over the 8 lattice neighbours. Beyond-window context reaches it as *observations*
  (a geodesically-routed "radar" of far exploration mass and far teammates, plus a per-direction
  value field), never as an imposed target.
- **A rendezvous economy instead of a rendezvous heuristic.** A privileged team-union novel-scan
  reward makes splitting up the highest-paying policy by construction; a sync-event reward pays for
  the map actually exchanged when two robots meet. Both the reward gate and the actor's own
  observation are literally the same scalar, so the policy decides *when* to meet.
- **A teammate-position belief that survives comm loss.** When contact breaks, hypotheses freeze on
  the frontier openings, travel their geodesics, and then diffuse — and that field, not a stale
  last-known point, is what the policy navigates by.
- **A count-invariant CTDE critic.** Mean⊕max pooling over agents means the same weights serve any
  team size, which is what makes the M=4 result a zero-shot lift from an M=2 policy.
- **A frozen, adversarially-fair evaluation protocol** against the IR2 baseline
  ([eval/comparison/PROTOCOL.md](eval/comparison/PROTOCOL.md)), with a bit-for-bit dataset parity
  gate and paired per-map statistics.

---

## Results

Evaluated against IR2 on 100 fixed maps per split, for teams of 2 and 4 robots, under the frozen
protocol. Each map's travel budget is **IR2's own distance on that same map**, so the two systems
are held to the same physical constraint; `explored` is IR2's own definition (the mean over robots
of each robot's *private* map, not the union) and `success` is their termination rule (every robot
holds ≥ 99 % of the map in its own belief).

| Split | M | Travel budget (px) | MARLauder travel | Budget used | IR2 explored | **MARLauder explored** | IR2 success | **MARLauder success** |
|---|---|---|---|---|---|---|---|---|
| hybrid   | 2 | 3422  | 1858  | **54 %** | 0.997 | 0.996 | 1.00 | 0.97 |
| hybrid   | 4 | 2413  | 1458  | **60 %** | 0.999 | 0.999 | 1.00 | 0.97 |
| corridor | 2 | 7204  | 3746  | **52 %** | 0.976 | **0.998** | 0.76 | **0.97** |
| corridor | 4 | 5352  | 3152  | **59 %** | 0.992 | **0.999** | 0.93 | **0.98** |
| complex  | 2 | 16966 | 11470 | **68 %** | 0.975 | **0.983** | 0.72 | **0.85** |
| complex  | 4 | 13403 | 8566  | **64 %** | 0.982 | 0.984 | 0.80 | 0.82 |

Read it as: **given the distance IR2 needed, MARLauder finishes on roughly half to two-thirds of
it**, at equal or better coverage, with the largest success-rate gains exactly where the baseline
struggles (corridor and complex). The two hybrid cells are the honest exception — the maps are easy
enough that IR2 already succeeds on every episode, and we give up 3 points of success rate there.

Reproduce the table with:

```bash
CKPT=runs/<run>/ckpt_best.pt bash pipelines/eval_ir2_comparison.sh
```

Per-episode CSVs for the released checkpoint are in `eval/comparison/results/`;
`eval/comparison/analyze.py` prints the per-cell table with paired Wilcoxon tests. Never average
across splits — they are different problems with different caps, and the aggregator has no code
path that produces such a mean.

---

## Installation

### Docker (recommended)

```bash
docker compose build
docker compose up -d
docker compose exec marlauder bash
```

The container mounts the **parent** directory at `/workspace`, so a sibling checkout of the IR2
baseline repository is visible for the comparison. The web dashboard starts automatically on
<http://localhost:8080/>.

### Bare metal

Needs an NVIDIA GPU with a CUDA 12.x driver (Blackwell requires driver ≥ 570).

```bash
pip install -r requirements.txt
export PYTHONPATH=.
```

Paths resolve relative to the repository root (see `paths.py`) and can be redirected with
`MARLAUDER_DATA`, `MARLAUDER_RUNS` and `IR2_ROOT`.

### Map data

The map packs are not in the repository (~9 GB, and derived). Regenerate them from the DungeonMaps
PNGs:

```bash
python scripts/preprocess_maps.py --src <path-to>/DungeonMaps --out data
python eval/comparison/parity_check.py     # gate: our packs must match the PNGs bit for bit
```

---

## Quick start

Verify the toolchain (torch and Warp on the same GPU, zero-copy interop):

```bash
PYTHONPATH=. python tests/00_test_toolchain.py
```

Smoke train, about a minute — boots the full pipeline, runs a few PPO updates, writes a checkpoint:

```bash
PYTHONPATH=. python scripts/run_train.py \
    --split train/easy --total-steps 40000 \
    --n-envs 8 --n-agents 2 --rollout-len 64 --max-episode-steps 64 \
    --out runs/smoke
```

Evaluate it and render one GIF per map:

```bash
PYTHONPATH=. python scripts/eval_final.py runs/smoke/final.pt \
    --split train/easy --n-maps 3 --steps 256
```

Run the property tests:

```bash
bash tests/run_all.sh
```

Reproduce the released policy end to end (~69 h on a 16 GB GPU) — see
[pipelines/README.md](pipelines/README.md) for what each of the four stages does and why:

```bash
bash pipelines/train_full.sh
```

---

## Project structure

```
MARLauder/
├── env/                Simulation
│   ├── world_warp.py       Per-agent occupancy + Warp LiDAR kernels
│   ├── explorer.py         The vectorized environment: step, comm, fusion, reward, obs
│   ├── graph_lattice.py    8-neighbour graph, Bellman-Ford, radar, value field, ego window
│   ├── teammate_belief*.py Uniform and pathfront teammate-position belief models
│   ├── frontier.py         conv2d frontier detector
│   └── maps.py             Split loading and batched sampling
├── models/             Masked GAT encoder, actor-critic, orthogonal init, value normalizer
├── train/              Rollout buffer + GAE, MAPPO update, training driver
├── eval/               Deterministic rollout, inspector trace, rendering
│   └── comparison/     PROTOCOL.md, map indices, parity gate, aggregator, result CSVs
├── scripts/            CLI entrypoints (training, evaluation, tracing, preprocessing)
├── tests/              Property tests — see tests/README.md
├── tools/              Offline diagnostics — see tools/README.md
├── pipelines/          Reproduction recipes — see pipelines/README.md
├── viz/                Web dashboard + step-through decision inspector
├── docs/               architecture.html — the pipeline diagram
├── paths.py            Repository-relative path resolution
├── DOCS.md             Full reference: modules, obs schema, reward, every CLI flag
└── README.md           This file
```

---

## Understanding a run

Every iteration appends one JSON row to `runs/<run>/metrics.jsonl` — independently of Weights &
Biases, which is off unless `--wandb` is passed.

```bash
python scripts/analyze_run.py runs/<run>                    # summary
python scripts/analyze_run.py runs/<run> --reward-budget    # which term owns the return
python scripts/analyze_run.py runs/a runs/b --compare       # one row per run
```

The web inspector steps through a real episode showing, for each decision: the observation channels
per node, the teammate belief field, the **real** per-layer per-head GAT attention, the action
logits, and the reward broken into its terms. It is served at
`http://localhost:8080/<run>/inspector.html` for any run that has captured a trace.

The most-used training flags are below; the complete reference is in
[DOCS.md §8](DOCS.md), and the architecture diagram in
[docs/architecture.html](docs/architecture.html).

| Flag | Default | Meaning |
|---|---|---|
| `--split` | `train/easy` | `train/{easy,difficult}`, `test/{complex,corridor,hybrid}` |
| `--n-envs` / `--n-agents` | `16` / `1` | Parallel environments / cooperative robots per environment |
| `--rollout-len` / `--max-episode-steps` | `128` / `512` | Steps per PPO update / episode truncation |
| `--max-travel-frac` | `0` | Travel budget per map, as px travelled per free px. The binding horizon |
| `--done-mode` | `union` | `own` = every robot at 99 % of its own map (the IR2 rule) |
| `--n-hops` | `6` | Ego-window radius; the GAT depth is tied to it |
| `--comm-model` / `--ss-thresh` | `signal_strength` / `-70` | Path-loss radio / receiver sensitivity (dBm) |
| `--novel-scan-weight` | `1.0` | The privileged team-union exploration credit — the core reward |
| `--sync-weight` / `--sync-min-gap` | `0` / `32` | Sync-event payoff for map actually exchanged / anti-flicker gap |
| `--rdv-weight` / `--rdv-urgency-mode` | `1.0` / `time` | Dense rendezvous shaping / what makes a meeting urgent |
| `--belief-mode` / `--radar-team-source` | `uniform` / `lkp` | Teammate belief model / what feat[6] transports |
| `--init-ckpt` | — | Warm start from a checkpoint (how the stages are chained) |
| `--gru` | off | Enable temporal memory (the model is feed-forward by default) |

---

## Citation

If you use this code, please cite the accompanying paper. See `CITATION.cff`.

## License

MIT — see [LICENSE](LICENSE).
