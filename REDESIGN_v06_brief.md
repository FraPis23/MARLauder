# MARLauder v0.6 — Redesign Brief (cold-start context)

You are implementing an architecture change in **MARLauder**, a GPU-based decentralized
multi-agent exploration system. You have **no prior context** — this file is your full
brief. Read the actual code before trusting any file/function name here: this brief was
written from notes that may be slightly stale. Where it says "verify", grep first.

---

## 0. Mission (one sentence)

Replace the failed per-step hierarchical "navigation head" with a **two-layer
controller** (reactive local navigation + gated analytical far-target assignment) plus a
**supervised teammate-position belief predictor**, keeping everything GPU-batched and
fully decentralized.

You may implement this **by modifying the existing structure OR by rewriting** — that
decision is yours to make after you read the code (see §6). Both are acceptable.

---

## 1. What this project is

- Decentralized MARL exploration of 2D occupancy maps (DungeonMaps-style). N agents with
  LiDAR explore an unknown map; goal = cover it fast, without redundant overlap, with the
  work fairly split.
- **All-GPU pipeline**: NVIDIA Warp LiDAR kernels + PyTorch, vectorized over
  `[N_envs, M_agents]`. There is **no Python per-agent loop** in the hot path — everything
  is batched tensor ops. Any new logic you add MUST also be batched this way.
- **Decentralized**: each agent acts only on **its own** beliefs (own occupancy map, own
  graph, possibly-stale teammate info). No agent reads another's private state at action
  time. Training is CTDE (centralized critic / privileged reward is allowed at train time
  only, never at action time).
- Trained with MAPPO (PPO, AMP fp16). Sweeps via W&B.
- Runs inside a Docker container. Host GPU: RTX 5080 Laptop 16GB.
  Run pattern:
  ```
  docker exec <CONTAINER_ID> bash -lc 'cd /workspace/MARLauder && python scripts/run_train.py ...'
  ```
  Find the container with `docker ps`. The repo is mounted at `/workspace/MARLauder`.
- **Constraint**: do NOT `apt install` new system packages. `pip install` inside the
  container is fine if a Python dep is genuinely needed (prefer not to add deps).

## 2. File map (verify against actual tree)

| file | role |
|---|---|
| `env/world_warp.py` | Warp LiDAR kernels; per-agent occupancy fusion; `expected_gain(cand_xy, cand_valid, n_rays)` raycast info-gain |
| `env/graph_lattice.py` | 8-neighbour lattice graph; Bellman-Ford: `bf_from_target` / `bf_from_curr` / teammate-BF; node-feature build; `build_optim_graph` (optimistic FREE∪UNKNOWN flood) |
| `env/explorer.py` | vectorized environment: `EnvCfg`, `step`, `_refresh_obs`, reward terms, comm/gap, candidate frontiers |
| `models/actor_critic.py` | `MarlActorCritic`: ego-centric GAT encoder, `PointerHead` (low-level over K candidates), `StrategicHead` (the high-level head being removed), `disable_strategic` flag + `guidepost_nbr_bias` (the no-head path), `bind_graph` |
| `train/mappo.py` | MAPPO update; `_diversity_loss` (J.3 division); `evaluate_step_from_enc` |
| `train/buffer.py` | rollout storage; `compute_gae`; `compute_gae_hi` (high-level GAE — to be removed) |
| `train/driver.py` | training loop, logging, W&B, eval orchestration |
| `eval/rollout.py`, `eval/render.py` | deterministic eval rollouts + GIF rendering |
| `scripts/run_train.py` | CLI entrypoint + all flags |
| `DOCS.md`, `README.md`, `dev_log.md` | docs — keep updated |

## 3. Why we are changing it (the evidence — do not re-litigate)

The previous design made a learned `StrategicHead` choose a **per-step navigation
target** that the low-level pointer was rewarded to follow (`follow_coef`, `hi_coef`,
high-level critic `V_hi`, SMDP GAE `compute_gae_hi`, option commitment `_steps_on_option`).

A W&B sweep (8 trials) proved this is the wrong abstraction:
- Best runs only **match** a no-head local-greedy baseline (`coverage_auc ≈ 0.48`), never
  beat it; the best runs are the ones with the **least** head influence (lowest `hi_coef`).
- The head ignores the target info-gain feature (`head_target_gain_ratio → ~0`).
- **4 of 8 runs crashed** (NaN) at high `hi_coef` — the high-level PPO path is unstable.

Conclusion: **local-greedy navigation already owns coverage.** A learned per-step
navigator cannot beat it and is fragile. So we remove it and put learning where it
actually helps: predicting teammates under degraded communication, for coordination.

There is already a validated **no-head path** in the code: the `disable_strategic` /
`guidepost_nbr_bias` mode (pointer biased toward the BF first-hop to the nearest
frontier). In past runs it **beat** the head (coverage 95–100%, success 0.75). This is
your Layer 0 — you are extending it, not inventing it.

## 4. Target architecture

Three parts. Keep them cleanly separated.

### Layer 0 — reactive local navigation (always on)
- Per-step motion = greedy on local/egocentric utility (the existing `disable_strategic`
  pointer biased by `guidepost_nbr_bias`). The high-level head is **never** in the
  per-step loop.
- **In-grid / same-room separation stays reward-driven**: keep the **J.3 diversity loss**
  (`_diversity_loss`, `--div-coef`) and overlap/proximity penalties. These already make
  two nearby agents fan out (validated: `target_yield 0.04→0.56`, `overlap 0.29→0.19`).
  Do not remove them.

### Layer 1 — gated analytical far-target assignment
- **Trigger**: engage only when local utility is exhausted, i.e.
  `max over K candidates of candidate_utility < epsilon` (a utility floor, NOT "all cells
  visited"), optionally sustained for T steps. While local utility is productive, Layer 1
  is dormant and Layer 0 runs alone.
- **Assignment** (per agent, decentralized, on the agent's own beliefs). Score each
  global frontier `f`:
  ```
  score(f) = expected_gain(f) / (d_self(f) + 1)  -  beta * spread(f)
  ```
  - `expected_gain(f)`: reuse `world_warp.expected_gain` (LiDAR raycast info-gain).
  - `d_self(f)`: own Bellman-Ford geodesic distance (reuse `bf_from_*`).
  - `spread(f)`: coordination penalty — high if a teammate is closer to `f`, or is
    predicted to be heading there (uses the belief from §4.3 and teammate-BF distance).
  - Pick `argmax_f score(f)` **per agent**. **Do NOT use Hungarian / any joint optimal
    assignment** — it does not batch on GPU and is centralized. Decentralized greedy
    argmax is both correct and trivially parallel.
- **Execute**: path-follow to the chosen frontier via BF first-hop (pure navigation, no
  learned head per step). The moment local utility recovers, drop back to Layer 0.

### Layer 3 — supervised teammate-position belief predictor (the learned contribution)
- A network head that predicts **where a teammate is** when its position is stale/unknown.
- **Output a distribution, not a point**: a softmax **heatmap over lattice nodes**.
  (Acceptable v1 simplification: predict a 2D Gaussian `mu, Sigma` trained with Gaussian
  NLL — easier, but cannot represent multi-modal belief. Prefer the heatmap.)
- **Inputs**: `{teammate last-known node, time-since-last-comm Δt, last known velocity,
  shared-map embedding}`. **Reuse the existing GAT encoder** as the map backbone.
- **Training = supervised, not RL.** Ground-truth teammate positions are available at
  train time (the `force_full_pos_sharing` mode). Train the predictor with cross-entropy
  (heatmap) or Gaussian-NLL against the true teammate node. This sidesteps the credit
  assignment that sank the old head.
- **Create Δt gaps**: randomly mask teammate comm for intervals during training while
  STILL logging ground truth as the label. (The `--comm-range` mechanism already drops
  comm spatially — leverage it.)
- **Avoid distribution shift**: train the Layer-1 assignment on the **predicted** belief,
  not on ground truth (privileged→predicted annealing / DAgger-style). Start on GT, anneal
  onto predictions.
- **Coordination consumes the FULL distribution, never the argmax**:
  `E_belief[dist(team, f)] = belief[N,M,Nodes] · dist[Nodes,Frontiers]` (a batched
  matmul). Integrating over the belief hedges against confident-wrong predictions (a point
  prediction would collide both agents onto the same frontier).

### Two division regimes — the core design idea, keep it intact
| range | mechanism |
|---|---|
| same room / in-grid | **reward + J.3 div-loss** (dense gradient pushes agents apart) |
| far / out-of-grid | **Layer-1 analytical assignment + belief** (discrete allocation; no reward gradient can reach a far target) |

## 5. Reuse vs remove

**REUSE (do not break — Layer 1 / belief / Layer 0 depend on these):**
- Warp LiDAR + occupancy fusion; `expected_gain`.
- Lattice graph + all Bellman-Ford (`bf_from_*`, `build_optim_graph`).
- GAT encoder, `PointerHead`, the `disable_strategic` / `guidepost_nbr_bias` path.
- J.3 `_diversity_loss` + `--div-coef`; novel-scan reward; overlap/proximity penalties.
- Candidate-frontier machinery, `target_logits` (div-loss needs them), teammate-BF.
- Metrics, eval suite, W&B, `EnvCfg` plumbing.

**REMOVE — but LAST, only after the new path is validated (so you can A/B against it):**
- `StrategicHead` as a per-step navigator; high-level critic `V_hi` / `critic_hi`;
  `compute_gae_hi`; the follow-reward block; CLI/cfg `--follow-coef`, `--hi-coef`,
  `--hi-ent-coef`, `follow_taper_nodes`; option commitment `_steps_on_option` /
  `--max-steps-on-option` / `--switch-margin`.
- These touch ~9 files (model/env/buffer/mappo/driver/scripts). Map every reference
  before deleting; remove only what the new path does not use.

## 6. Scratch vs modify — your call

Read the code first, then decide. Guidance:
- **Strong default: modify in place, starting from the `disable_strategic` path.** The
  GPU env, Warp kernels, BF, reward shaping, div-loss, metrics, eval, and the many fixed
  env bugs (teleport, spawn, NaN) are validated and orthogonal to the head. Rewriting risks
  resurrecting them.
- **Build forward, delete last**: keep the dead head behind its flag while you add Layer 1
  + the predictor; delete the high-level PPO only once the new controller is validated.
- A full rewrite is acceptable ONLY if you find the existing structure genuinely blocks the
  new design — if you go that route, still reuse `env/` and the Warp/BF/graph modules
  verbatim; the env is not the problem.

## 7. Constraints (hard)
- Everything batched over `[N_envs, M_agents]` on GPU. No per-agent Python loops in the
  step/obs/assignment hot path.
- Decentralized at action time: an agent uses only its own beliefs + (stale/predicted)
  teammate info. Privileged info is train-time-only (CTDE).
- No new apt packages. Avoid new pip deps if possible.
- fp16 AMP is on — guard new losses against NaN (finite-check before optimizer step; the
  codebase already has a `nan_skips` pattern — follow it).

## 8. Acceptance criteria
1. **Smoke**: a short training run completes with no shape errors and finite losses, e.g.
   ```
   python scripts/run_train.py --split train/difficult --n-agents 2 \
     --total-steps 300000 --n-envs 32 --max-episode-steps 256 \
     --div-coef 0.1 --eval-on-ckpt --out runs/v06_smoke
   ```
2. **Layer 0 regression**: with Layer 1 disabled, coverage matches the old
   `disable_strategic` baseline (coverage_auc ≈ 0.48+, no crash).
3. **Layer 1 helps on far targets**: on a map with a frontier outside the egocentric
   window, the trigger fires and the agent path-follows to the assigned far frontier
   (visible in a GIF), and `sensing_overlap` does not increase vs Layer-0-only.
4. **Belief predictor learns**: its supervised loss decreases; predicted teammate node
   accuracy beats a naive "last-known position" baseline as Δt grows.
5. **Ablation ladder** wired so it can be run later:
   `no-belief / naive-Gaussian / learned-belief / privileged(GT, upper bound)` vs
   `--comm-range`. Expect learned to cut overlap/redundancy as comm degrades.

## 9. Out of scope (do not do now)
- Recursive belief (teammate's motion depends on its belief of you) — v1 is one-step
  theory-of-mind only. Note as future work.
- Joint/Hungarian assignment (centralized, doesn't batch).
- Re-tuning the old head.

## 10. When done
- Update `DOCS.md`, `README.md`, and prepend a `dev_log.md` entry describing the v0.6
  architecture, what was removed, what was added, and smoke results.
- Report: which approach you took (modify vs rewrite) and why, the new CLI flags, the
  files changed, and the smoke-run output.
