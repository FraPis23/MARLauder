# MARLauder — Agent Handoff (2026-06-12)

You are taking over a MARL cooperative-exploration project (PhD thesis: beat IR2 on DungeonMaps
with an all-GPU pipeline). Decentralized, GPU-vectorized (Warp LiDAR + torch graph/BF), MAPPO,
2 agents. Authoritative docs: `DOCS.md`, `README.md`. Memory index for deeper history:
`~/.claude/projects/-home-ivancist-Documents-MARLauder-Dir/memory/`.

Runs in a Docker container named **`marlauder`** (`docker exec marlauder bash -lc '...'`,
workdir `/workspace/MARLauder`, which is the host repo, volume-mounted). Two training machines:
**RTX 4080 12 GB** and **RTX 5080 16 GB** (the 12 GB binds VRAM choices). W&B project
`marlauder-sweep2`, entity `ivancist`. **Never run a train/eval on the same GPU as a live
sweep** — it halves SPS for both.

---

## THE OPEN PROBLEM (top priority — not yet fixed/verified)

**Agents stall / ping-pong: the deterministic policy oscillates between 2 cells, constantly
re-picking its strategic target, and explores ~nothing.** User requirement: *both agents must
always be working* — never freeze, never just take turns; equal work = both contribute.

### Hard evidence (do not re-litigate; reproduce if unsure)
- Sweep `gh3qj58r` (24 trials, 2M steps each, **valid** env): **train coverage `explore/ep_end`
  0.93–0.99** but **deterministic `eval/coverage_auc` 0.51–0.71**, `eval/steps_to_90`→232,
  several `eval/success`=0. Big train↔eval gap.
- Direct measurement on a checkpoint (`scripts/measure_thrash.py`): **DET target-change=0.99/step,
  explored=5%**; **STO target-change=0.12/step, covers map**. Both move ~18px/step (1 hop).
- GIF (position-only 80k run): `explored=5.0%` at t=1, t=129 **and t=256** — confined to spawn room.

### Root cause (confirmed in code)
The **strategic head is feedforward & memoryless** — `StrategicHead.forward(curr_emb, cand_feat,
cand_valid)` in `models/actor_critic.py` (no hidden state, no previous target). It re-`argmax`es
a target every step from the current obs only. Deterministically this is a **2-cell limit cycle**:
move → obs shifts → candidate scores flip → target flips back → move back → repeat. The GRU
(`gru_actor`) is **downstream** (smooths the action, not the target). Action-sampling noise breaks
the cycle, so **training (stochastic) looks great while deterministic eval is frozen**. The
`target_switch_pen` reward (branch-flip penalty) can't fix it: it only acts in stochastic
training where the thrash doesn't occur, so the net never learns to avoid it.

---

## CHANGES I MADE THIS SESSION — **IMPLEMENTED BUT NOT BEHAVIORALLY VERIFIED**

Treat ALL of these as untested until you run the verification below. They compile and pass
trivial functional checks only.

1. **`cand_prev_branch_match` observation feature (THE fix for the thrash)** — `env/explorer.py`
   `_refresh_obs`: a 9th candidate feature, 1.0 if a candidate's BF first-hop branch equals the
   branch the agent committed to last step. New state `self._prev_branch [N,M]` (set in the
   `target_switch` block, reset per episode). `CAND_FEAT_DIM` 8→9 in `models/actor_critic.py`.
   Goal: give the feedforward head memory of its own direction → learned commitment (NOT
   hysteresis; obs still update freely). **Unverified that it reduces the 0.99 thrash.**
2. **Eval now passes the target** to `step()` — `train/driver.py:265` (eval suite) and
   `eval/rollout.py:91` (gif) now call `step(action, target_choice=out["target_argmax"])`. Needed
   so `_prev_branch` and the branch penalty are live in deterministic eval. **Unverified.**
3. **`proximity_pen` ELIMINATED** (`EnvCfg.proximity_penalty_coef` default 0.0, `--proximity-pen`
   default 0). It was a raw-distance reflex that drove inter-agent ping-pong and the
   single-frontier deadlock; `novel_scan` already pays 0 for team-known cells so anti-chase
   survives. Fallback documented (productivity gate). **Unverified that removal doesn't cause clumping.**
4. **`target_switch_penalty_coef` raised 0.01→0.05** (graph-tree branch-flip commitment; safe now
   it's on argmax intent, not Gumbel sample). **Unverified.**
5. **Target-claim at rendezvous** — `env/explorer.py`: `self.last_known_target [N,M,M]`
   (comm-gated, updated in `_update_last_known_pos`), and in `_refresh_obs` a higher-ID agent's
   candidate that equals a **lower-ID** in-comm teammate's claimed target is **masked**
   (`cand_valid→False`) UNLESS it's the agent's only option (**single-frontier guard**). A static
   unit test confirmed the mask fires; **not verified in a trained policy.**
6. **`k_epochs` 4→2** in `sweep.yaml` + `sweep_stage2.yaml` — fixes observed `train/clipfrac`
   0.21–0.26 (>0.2 clip): `mb=1`×`k=4` reused the batch 4× → policy drift → heavy clipping.
7. **`minibatch=1`, `n_envs=32`** in both sweeps (paper Suggestion 3; user confirmed 32 fits the
   4080). **NaN guard** in `train/mappo.py` (skip optimizer step on non-finite loss/grad;
   `train/nan_skips` logged) — fixed 3 diverged runs in `gh3qj58r`; not stress-tested since.
8. New tool: `scripts/measure_thrash.py` (the verification harness).

### Earlier-session changes (verified by unit/smoke, NOT by a full decentralized train)
- **Env teleport bug FIXED** (was catastrophic): `world_warp.py _mark_pos_free` now stamps a 3×3
  footprint at 2·LO_FREE so the robot's own cell is FREE (a single LO_FREE landed exactly on the
  strict `>` threshold → cell stayed UNKNOWN → graph collapsed → every action teleported via the
  fallback). Also `explorer.py:253` invalid-action fallback now uses `curr_idx_global` (real
  node), not the LOCAL window-center constant. Verified: 1-hop moves, co-located spawn.
- **D1 optimistic teammate-BF** (`graph_lattice.py build_optim_graph` + `bf_from_target` edge
  override): teammate-distance BF over a FREE∪UNKNOWN edge set so a teammate in your unexplored
  region is still reachable. Flood removed (was a big SPS regressor); connectivity handled by BF.
- **D2 eval/score rebalance** (`train/driver.py`): score = mean per-map
  `auc − w_imb·(imb/(1−1/M)) − w_ov·ov` (imbalance normalized to [0,1]); added `eval/fairness`
  (Jain), `eval/score_std`. Flags `--score-w-imbalance/--score-w-overlap`.

**ALL prior sweeps (`uelh3hs3`, `odft9txk`, `wyhlx0ki`) trained on the broken teleport env →
INVALID. `gh3qj58r` is the first valid sweep but predates the thrash fix.**

---

## YOUR PLAN

### Step 0 — VERIFY the thrash fix (gate everything on this)
Train a short test (position-only, n_hops=2 for speed, k=2, maps vary automatically across the
10k train/easy each reset), then measure. GPU must be free (no sweep running).
```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && python -u scripts/run_train.py \
  --split train/easy --n-envs 32 --n-agents 2 --force-full-pos-sharing --comm-range 120 \
  --n-hops 2 --novel-scan-weight 1.50 --team-weight 0.405 --proximity-pen 0.0 \
  --target-switch-pen 0.05 --minibatches 1 --k-epochs 2 --rollout-len 256 \
  --max-episode-steps 256 --total-steps 300000 2>&1 | tee runs/train.log'

docker exec marlauder bash -lc 'cd /workspace/MARLauder && \
  python scripts/measure_thrash.py --ckpt runs/run_default/final.pt --map-idx 120'
```
**PASS = DETERMINISTIC target-change drops from ~0.99 toward ~0.1 AND explored ≫ 5%.**
Also render GIFs and eyeball behavior on several maps (10k available, vary the idx):
```bash
docker exec marlauder bash -lc 'cd /workspace/MARLauder && bash scripts/gif.sh 120 256'
# also 1543, 2877, 4012, 5530, 7211, 8650, 9904 — open runs/latest.gif on the host
```
Look for: both agents moving with purpose, committing to a frontier, NOT oscillating in place,
NOT both freezing on the last frontier. `explored` must climb across the episode.

**If it still thrashes** (target-change stays high): the obs feature alone is insufficient.
Escalate to (a) feed the actor GRU hidden into the strategic head (make target selection
recurrent), or (b) a macro-action (env executes BFS toward the chosen target for H steps so the
target isn't re-picked every step). Do NOT use hard hysteresis — the user rejected it
("observations change until the target is reached"). Also sanity-check `clipfrac` < ~0.15 in
`runs/train.log`; if higher, drop `lr` 3e-4→1e-4.

### Step 1 — only after Step 0 passes: the sweep
User prefers **position-only** sharing (occupancy NOT shared, position shared) — under full
occupancy sharing the give/recv/overlap exchange terms are inert, so map-exchange can't be
evaluated. So use `sweep_stage2.yaml` (already position-only: `--force-full-pos-sharing`,
`--comm-range 120`, no occupancy sharing), which currently also fixes the labor-division coefs to
the `gh3qj58r` winner (summer-6) and sweeps give/recv/overlap. **Decide with the user**: either
re-sweep the labor-division dims in this position-only regime (edit `sweep_stage2.yaml` params to
add novel/team/target-switch/n-hops), or run `sweep.yaml` (currently still full-occupancy Stage 1
— would need `--force-full-occupancy-sharing` removed if you want position-only there too).
```bash
docker exec -it marlauder bash -lc 'cd /workspace/MARLauder && wandb sweep sweep_stage2.yaml'
docker exec -it marlauder bash -lc 'cd /workspace/MARLauder && wandb agent --count 12 ivancist/marlauder-sweep2/<ID>'  # both machines
```
Monitor the **W&B dashboard** (not tail): `eval/score`, `eval/fairness` (both work), `eval/score_std`
(map-luck), `train/nan_skips` (≈0), `train/clipfrac` (<~0.15). The eval metric is the
deterministic 8-map suite (`EVAL_MAP_IDX` in `train/driver.py`) — it is only trustworthy once the
thrash fix passes Step 0 (otherwise it measures the limit cycle, not the reward).

### Step 2 — deferred (don't start until 0+1 done)
- IR2 **map-surplus observation `s_ij`** for OUT-OF-COMM division (the target-claim only works in
  comm range). Per-candidate "info I uniquely hold vs teammate toward this frontier."
- Realistic comm: drop `--force-full-pos-sharing` → comm-gated `last_known_pos` → estimate
  teammate position; `comm_duty` becomes meaningful.
- Seed replicas (3 seeds) on finalists; final long (5M) train of the winner; `test/*` splits.

---

## RULES / GOTCHAS
- **GPU contention**: never train/eval while a sweep runs on the same box.
- **Buffered logs**: piping `python` stdout buffers; use `python -u ... | tee runs/train.log` to
  watch live. The W&B dashboard is the right monitor for sweeps.
- **Checkpoints overwrite** `runs/run_default/*.pt` every run — copy any you want to keep.
- **Obs-space change = from-scratch retrain** (CAND_FEAT_DIM is now 9; old checkpoints won't load
  cleanly — `measure_thrash.py` uses `strict=False` and warns).
- **VRAM**: `mb=1, h6` peaks ~12.7 GB → at the 4080's edge; h2 is far lighter and fine for tests.
- **Don't trust a single eval endpoint** — `gh3qj58r` scores were partly phase-luck because of the
  oscillation; that's exactly what Step 0 must fix.
- The user wants **evidence, not claims**: every assertion about behavior must be backed by a
  measurement (`measure_thrash.py`) or a GIF, not by "it should work" or "more steps will fix it."
