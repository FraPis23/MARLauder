#!/bin/bash
# v1.9 — M=4, REWARD BUDGET RE-BALANCED + TRAVEL BUDGET THAT ACTUALLY BITES.
# Control = v18 (runs/v18_m4_20260809_125724, stopped at it 270/732). Same warm start, same M,
# same everything else; two deliberate changes, both aimed at ONE measured cause.
#
# WHY v18 WAS STOPPED. 270 iterations bought nothing and cost a little. The best checkpoint was
# effectively the ZERO-SHOT transfer from v16 M=2:
#     eval/success_rate     0.88 (first eval) -> 0.75      eval/steps_to_90   310 -> ~370
#     eval/own_cov_final    0.99             -> 0.97      metric/own_cov_gap 0.255 -> 0.252 (flat)
#     metric/redundancy /3  0.640 -> 0.658 (RISING; v16 M=2 FELL 0.607 -> 0.563 in the same window)
# And v16's own full run proves the window matters: redundancy and steps_to_90 are decided inside
# the first ~200 iterations and then FROZEN for 500 more (redundancy by 100-it bins: 0.684 0.571
# 0.584 0.557 0.542 0.555 0.534; steps_to_90: 304 372 360 359 352 358 348 — it never returns to the
# 304 it started at). Waiting for v18 to fix itself was not supported by any precedent in this repo.
#
# THE MEASURED CAUSE — the per-agent reward budget INVERTED when M went 2 -> 4.
# Per-step reward, last 10 iterations of each run:
#     term         v18 M=4     v16 M=2    ratio
#     novel        0.01602     0.02639    x0.61     <- being the DISCOVERER pays 39% less
#     completion   0.03271     0.02234    x1.46     <- ENDING the episode pays 46% more
#     sync         0.00918     0.00543    x1.69     <- EXCHANGING pays 69% more (6 pairs, not 1)
#     revisit      -0.01148   -0.01118    x1.03
#     step         -0.01770   -0.01783    x0.99
# Budget SHARES: M=2 was novel 30% / completion 26%. M=4 is completion 37% / novel 18%. The ordering
# flipped. It is arithmetic, not a bug: novel_scan pays union-new cells to the DISCOVERER only, and
# the same union now splits four ways, while completion is a flat 10.0 paid to everyone and under
# done_mode=own collecting it requires four maps at 99% -> exchange -> proximity. The reward budget
# itself now pays more for standing near a teammate than for being the one who finds new ground.
# That is why redundancy RISES at M=4 while it FELL at M=2.
#
# CHANGE 1 — `--novel-scan-weight 1.65`. Restores the MEASURED per-agent share (0.02639/0.01602 =
# 1.65), not a theoretical 1/M factor: the union grows somewhat faster with four robots, so the
# per-agent share is not exactly 1/M and the empirical ratio is the honest number. Zero code change,
# it is a plain multiplier (EnvCfg.novel_scan_weight, default 1.0). NOTE this is deliberately NOT a
# no-op at M=2 — the M=2 recipe keeps 1.0, because 1.0 is what produced v16's working budget.
#
# CHANGE 2 — `--max-travel-frac 0.030` (was 0.040). MEASURED knee, v18's own ckpt_stop, M=4,
# train/difficult, 32 envs, sampled actions:
#     frac    terminate   weakest own_cov   dist p50   steps
#     0.040     0.84          0.968           4244      308     <- v18: the budget never bit
#     0.035     0.59          0.930           4087      315
#     0.030     0.59          0.898           3863      280     <- same gradient, 6% less distance
# The knee sits between 0.040 and 0.035 and then PLATEAUS, so 0.030 buys real pressure at no cost in
# gradient availability: the completion bonus still fires on 59% of episodes, far above
# probe_reachability's 0.15 "dead objective" threshold, so this is NOT the v12 failure mode.
# Note what the probe does NOT show: tightening did not make THIS (fixed) policy faster — p50
# distance barely moved. That is expected; the probe cannot adapt. The point of training under a
# budget that bites is that the policy adapts, which is exactly what v16's own diagnosis said
# ("a budget that never bites stops training the thing being measured").
# On IR2 fairness: 0.030 is strictly WORSE for us than 0.040, so it stays on the allowed side. Four
# robots at 0.030 still get 1.5x the team distance two robots had at 0.040.
#
# NO CHANGE 3 — the step cap STAYS AT 768. A first attempt at this run used 512 and that was wrong;
# it was killed at it 7 and relaunched. The reason, computed from the splits' own free_counts and
# the MEASURED travel rate of 20.5 px/step (mean distance / mean steps was 20.5 at BOTH frac 0.040
# and 0.030, so it is a property of the policy, not of the budget):
#
#   train/difficult   free px  p10 102k  p50 128k  p90 495k   (6000 maps)
#     frac 0.030 -> budget gives  p10 150   p50 187   p90 724  steps
#     a 512 cap bites on 33.3% of maps · a 768 cap bites on 3.6%
#   test/complex      free px  p10 410k  p50 468k  p90 521k   (500 maps, THE EVAL SPLIT)
#     frac 0.030 -> budget gives  p10 601   p50 685   p90 763  steps
#     a 512 cap bites on 99.6% of maps · a 768 cap bites on 7.2%
#
# So a FLAT 512 does nothing on two thirds of the training maps (the budget already stops them at
# 187 steps) and amputates the big third — no added pressure where episodes are short, no gradient
# where they are long. That is the v12 mistake in miniature, and it is precisely what
# `max_travel_frac` exists to prevent: train/difficult spans 3.9x in free area, so only a budget
# that SCALES with the map can restrict every map by the same proportion.
# Worse, `--eval-steps 512` would have censored 99.6% of test/complex, making success_rate
# incomparable with v18's 768 — the same measurement bug this repo already documented on v12
# ("a measurement cap below the training budget silently reintroduces starvation in the
# measurement"). The cap's ONLY job is catching an agent that burns steps without burning distance;
# 768 does that and nothing else.
# THE RESTRICTION IS THE BUDGET, expressed in the right currency: on the median train/difficult map,
# 0.040 -> 0.030 cuts the available steps from 250 to 187, i.e. -25%, on every map proportionally.
#
# NOT CHANGED, deliberately: no comm_idle_pen (v17 refuted — pricing contact buys contact-avoidance,
# syncs/ep 5.06->2.25 and own_cov 0.974->0.889). The M>2 fixes (per-teammate peak-normalized feat[4],
# phi gathered at j_star, per-agent belief hue) are in the code and are exact no-ops at M=2.
#
# STILL OPEN, not attempted here: the cross-agent DIVERSITY LOSS (J.3, 2026-06-13). It was measured
# to work at M=2 (target_yield 0.04->0.56, auc 0.715 vs 0.651, overlap 0.19 vs 0.29) and then DELETED
# when the StrategicHead was killed on 2026-06-17 — `div_coef` / `_diversity_loss` no longer exist
# anywhere in the repo. It is the structurally right lever (a penalty on the POLICY DISTRIBUTION,
# adding no reward mass, so it cannot be traded off against exploration and cannot teach the
# contact-avoidance that sank v17), but re-implementing it on the current pointer architecture is
# real work, not a revert. Do that only if v19 still shows redundancy flat-or-rising.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
INIT_CKPT=${INIT_CKPT:-runs/v16_difficult_20260803_135950/final.pt}
OUT=${OUT:-runs/v19_m4_${TS}}

if [ ! -f "${INIT_CKPT}" ]; then
  echo "ERRORE: checkpoint di warm start non trovato: ${INIT_CKPT}"; exit 1
fi

# SAME warm start as v18 (v16 phase 2, M=2) on purpose: that makes v18 the control and the two
# changes above the only difference. Expect `missing=0 unexpected=0` and NO "DROPPED" line.
python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.030 \
  --n-agents 4 \
  --novel-scan-weight 1.65 \
  --total-steps 6000000 \
  --rdv-urgency-mode budget --rdv-urgency-start 0.5 \
  --n-envs 32 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 \
  --k-epochs 4 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront \
  --radar-team-source belief --map-seed 0 \
  --pf-frontier-min-unknown 1 --pf-frontier-min-util 1e-6 \
  --done-mode own \
  --gamma 0.998 \
  --rdv-weight 0.10 --rdv-clamp-pos --rdv-urgency-T 256 \
  --sync-weight 0.25 --sync-recv-ratio 0.5 --sync-min-gap 32 \
  --trace-steps 512 \
  --eval-suite-splits test/complex \
  --eval-on-ckpt --eval-split test/complex --eval-steps 768 \
  --init-ckpt ${INIT_CKPT} \
  --out ${OUT}
echo "V19_DONE final=${OUT}/final.pt best=${OUT}/ckpt_best.pt"

# DA GUARDARE, contro runs/v18_m4_20260809_125724 (stesso warm start, stessa M).
# The decision window is the FIRST ~200 iterations — that is where v16 made all of its progress and
# where v18 made none. Do not wait for 732 to judge this.
#   reward/novel vs reward/completion  -> novel must be back ABOVE completion in |share|. If it is
#                                         not, 1.65 was too small and the cause is unaddressed.
#   metric/redundancy /(M-1)           -> v18 0.640 -> 0.658 (RISING). Must FALL. This is the run's
#                                         whole purpose; console `redun=` already prints it divided.
#   eval/steps_to_90                   -> v18 310 (zero-shot) -> ~370. Must beat 310, i.e. actually
#                                         beat the untrained transfer, which v18 never did.
#   eval/success_rate                  -> v18 0.88 zero-shot -> 0.75. Expect a DIP at the start (the
#                                         budget is 25% tighter) then recovery. If it is still under
#                                         0.60 by iteration 200, 0.030 was too tight -> go to 0.035.
#   reward/completion                  -> must NOT go to zero. That is the v12 failure and it means
#                                         the objective became unreachable at this budget.
#   metric/own_cov_gap                 -> v18 flat at 0.25 for 270 iterations. Any real fall is the
#                                         first evidence that division of labour is being learned.
