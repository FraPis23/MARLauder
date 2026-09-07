# MARLauder vs IR2 — evaluation protocol

This document is the authority for the comparison. It is referenced by `env/explorer.py`,
`scripts/eval_comparison.py`, `eval/comparison/parity_check.py` and `eval/comparison/analyze.py`;
where those files and this document disagree, this document is wrong and should be corrected.

The protocol is **frozen**: it was fixed before the MARLauder side was ever run, and none of the
choices below are free parameters at reporting time.

---

## 1. The principle

Every discretionary choice is resolved **in IR2's favour**. The metrics are IR2's own, computed
their way; the episode caps are theirs; the termination rule is theirs; the ground truth is
verified bit-for-bit identical. Where our system is disadvantaged by that, it is disadvantaged.

---

## 2. Dataset

- **100 maps per test split** (`complex`, `corridor`, `hybrid`), fixed **by filename** in
  `map_indices_{split}.json`. Each entry carries `pack_idx` (the index in the MARLauder `.npy`
  pack) and `file` (the IR2 PNG), so the same physical map is addressable from both systems.
- **Parity is a gate, not an assumption.** `parity_check.py` re-derives the ground truth from the
  original PNG with IR2's own rule (grayscale `> 150` = free; pixel value `208` = start marker) and
  asserts an identical free/obstacle mask over the valid region, obstacle-only padding outside it,
  and a pack start inside the PNG's 208 blob. Exit code 0 is required before the comparison runs.

### Map order

`map_indices_{split}.json` must list maps in the order **IR2 actually runs them**:

```python
self.map_list = os.listdir(self.map_dir)
self.map_list.sort(reverse=True)            # IR2 env.py:34-35 — DESCENDING
self.file_path = self.map_list[map_index]   # map_index == episode
```

Seeing the same *set* of maps is not enough: the comparison is **paired**, so row *i* of the two
CSVs has to be the same map. An ascending list against IR2's descending iteration makes episode 0
`99.png` on their side and `1.png` on ours — per-cell means survive that, every paired statistic
does not. `parity_check.check_order` enforces it.

---

## 3. Agent counts

**M = 2 and M = 4.** MARLauder's M=4 is **zero-shot**: the critic pools over agents with
mean⊕max, so it is count-invariant and the same weights serve any M. Should a fine-tuned M=4 run
ever be reported, it must be labelled as a separate run and never substituted for the zero-shot one.

---

## 4. Episode budget

Two variants have been run. They answer different questions and must never be mixed in one table.

### v1 — native IR2 step caps

`hybrid` 196, `corridor` 196, `complex` 384 steps, from IR2's `test_parameter.py`
(`IR2_CAPS` in `scripts/eval_comparison.py`). No distance budget.

**Steps are not comparable between the systems.** An IR2 step is a waypoint decision plus an A\*
traverse of arbitrary length; a MARLauder step is one lattice hop of at most `nr·√2` = 22.63 px.
At equal step caps MARLauder spends roughly 45 % of the metres IR2 spends on `complex`
(7 684 px against 16 966), so nearly every complex episode ends truncated rather than solved.
Measured on one checkpoint: at the native complex cap, `complex_M2` scores success 0.01; given a
16 966 px budget it scores 0.90. v1 is therefore **conservative toward MARLauder by construction**.

### v2 — per-map distance budget

Each map's budget is **IR2's own `max_dist` on that same map** — the max over robots of cumulative
travel, which is exactly the reduction our truncation applies (`travel_px.amax(dim=1)`), so the two
systems are held to the same physical quantity rather than to two things that share a name. The
step cap is demoted to a safety net against a stalling policy.

The budget also reaches the actor: `Explorer.budget_px()` feeds `agent_scalars[2]` (`travel_frac`),
and the released policy trains with `--rdv-urgency-mode budget`, so the rendezvous pull ramps on
the observed fraction of budget spent. A budget the agent cannot see is a budget it cannot plan
against.

v2 additionally pins the agents to **IR2's own start positions**, snapped to the nearest free
lattice node. No pixel is a valid node in both discretisations, so the residual offset is measured
per episode and **reported** in the `start_offset_{mean,max}_px` columns rather than assumed away.

---

## 5. Termination

**`done_mode="own"`, forced inside `eval_comparison.py`.**

MARLauder's default is the team **union** at ≥ 99 %; IR2 stops when **every robot** holds ≥ 99 % in
its **own** belief (`env.check_done`, a loop over robots), and their `success` column *is* that
flag (`test_multi_robot_worker.py:122`). Under the union rule the exchange is optional — the union
is complete whether or not the map ever reached the other robot — so scoring MARLauder that way
would answer a strictly easier question.

The `--done-mode own` flag exists for training too, but it is **not a neutral change**: under the
own rule the completion bonus almost never fires on hard maps, and `novel_scan` pays for cells new
to the *union*, so an agent earns nothing for the last stretch of its *own* map. Aligning training
requires making the reward own-based as well, not just the stopping rule.

---

## 6. Metrics — IR2's native columns

CSV header: `eps,num_robots,max_dist,steps,explored,success,connectivity`.

| column | definition |
|---|---|
| `max_dist` | **The headline.** Max over robots of cumulative distance travelled (px) at episode end. The one temporal quantity that means the same thing on both sides. |
| `steps` | Recorded for completeness. **Not comparable across systems** (see §4); `analyze.py` deliberately reports no p-value for it. |
| `explored` | **Not the union.** IR2's `evaluate_team_exploration_rate` (`env.py:624-630`) is the **mean over agents** of `evaluate_exploration_rate(a)`, which reads `all_robot_belief[a][a]` — each robot's *own* map. Reporting the union instead would hand us roughly 8 points that do not exist. |
| `success` | Every robot holds ≥ 99 % of the map in its own belief. Sharing is part of the task. |
| `connectivity` | End-of-episode boolean: all robots in one connected "flock" in the comm graph, transitively (multi-hop). Anyone outside the largest flock is broken; two equal-largest flocks means everyone is broken. IR2 `env.py:205-221, 386`. |
| `explored_union` | MARLauder-only **appendix** column. Explicit so the union and the per-robot mean can never be confused again — never a substitute for `explored`. |

### Appendix columns (behaviour diagnostics)

`success` cannot distinguish coordinated exploration (split up, then deliberately meet to exchange)
from the degenerate solution (never separate, so the two maps coincide for free and no rendezvous
is ever needed). The degenerate solution scores a perfect `success` while demonstrating none of the
coordination being claimed. These columns exist to tell them apart:

`pair_dist_mean_px`, `pair_dist_max_px` (cross-system comparable — pixels, not canvas fractions,
because our canvas is zero-padded to a fixed per-split size and IR2 reads the unpadded PNG),
`comm_duty`, `sensing_overlap`, `n_syncs`, `own_gap_final`, `contrib_imbalance`.

### Attribution parity

`contrib_imbalance` is sensitive to *how* credit is counted, not only to behaviour. Two extra
columns recompute it under IR2's own accounting so the question gets a number instead of an
argument (`EnvCfg.attr_ir2_parity`, enabled with `--attr-parity`):

- `contrib_imbalance_seq` — IR2's single-claimant rule at **our** cadence: agents are folded into
  the running union in index order, so a cell scanned by two agents on the same step is credited to
  the lower index only. Isolates *one claimant vs many*.
- `contrib_imbalance_ir2` — the same rule at **IR2's** stride: they sense at hop endpoints, so
  credit arrives one graph edge at a time rather than one lattice hop at a time. Isolates
  *coarse vs fine crediting*.

Both are pure measurement: neither feeds a reward, an observation or a termination test.

---

## 7. Already identical — nothing to correct

- **Sensor range**: 80 px on both sides.
- **Radio model**: both run the log-distance path-loss model with `P_T = -20`, `thresh = -70`,
  `γ = 2` free / `4` through obstacles, `d₀ = 35`, `PL₀ = 31`, and shadowing `X_g, K ~ U[0,13]`
  resampled per episode. (IR2's `PROXIMITY_COMMS_RANGE` is dead code under
  `USE_SIGNAL_STRENGTH_NOT_PROXIMITY = True`.)
- **Coverage denominator**: ground-truth free pixels.

---

## 8. Statistics

Mean ± std per cell, plus a **paired Wilcoxon signed-rank** test on the per-map differences
(`max_dist`, `explored`, and the appendix columns). Pairing is available because row *i* of both
CSVs is the same map (§2).

**Never average across splits.** `hybrid`, `corridor` and `complex` are different problems with
different episode caps; a grand mean over them is not a quantity. `analyze.py` enforces this by
construction — it has no code path that produces one.

`analyze.py` implements Wilcoxon without scipy (tie-corrected normal approximation, zero
differences dropped, no continuity correction) because the runtime image does not ship scipy and
the paired test is not optional. With n = 100 the two agree to ~1e-12. The tie correction is
load-bearing rather than a formality: `success` and `connectivity` are booleans, so their
differences are all in {−1, 0, +1} and form one enormous tie group; without the correction sigma is
overstated and every boolean p-value comes out too large — the direction that silently hides a
real effect.

---

## 9. Running it

```bash
# 0. gate — datasets must be bit-identical, and in IR2's order
python eval/comparison/parity_check.py

# 1. MARLauder side -> eval/comparison/results/marlauder_{split}_M{M}_{tag}.csv
CKPT=runs/<run>/ckpt_best.pt TAG=<tag> bash pipelines/eval_ir2_comparison.sh
```

The IR2 side is produced in the IR2 checkout (see `$IR2_ROOT/comparison/`) and its per-episode
CSVs are read from `$IR2_ROOT/comparison/results/ir2_{split}_M{M}.csv`. Set `IR2_ROOT` if that
repository is not a sibling of this one.

## 10. Reference — IR2 baseline

Pretrained IR2 `model/stage2`, 600 episodes, 0 A\* skips:

| cell | max_dist | steps | explored | success | connectivity |
|---|---|---|---|---|---|
| hybrid_M2   | 3422 ± 928  | 88.7  | 0.997 | 1.00 | 0.89 |
| hybrid_M4   | 2413 ± 467  | 48.4  | 0.999 | 1.00 | 0.60 |
| corridor_M2 | 7204 ± 2297 | 141.9 | 0.976 | 0.76 | 0.66 |
| corridor_M4 | 5352 ± 2276 | 92.5  | 0.992 | 0.93 | 0.56 |
| complex_M2  | 16966 ± 4532| 277.8 | 0.975 | 0.72 | 0.59 |
| complex_M4  | 13403 ± 5768| 203.5 | 0.982 | 0.80 | 0.27 |
