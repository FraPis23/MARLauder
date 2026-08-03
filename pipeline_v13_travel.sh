#!/bin/bash
# v1.3 — FROM SCRATCH under a TRAVEL BUDGET episode criterion. The run that produces the policy
# for the IR2 comparison.
#
# WHY THIS EXISTS. v12 was killed at 68% because eval/success_rate sat at 0.00 for all 58 eval
# blocks. That diagnosis was half wrong. Stage 0 re-evaluated the SAME v12 ckpt_best changing only
# the stop criterion — step cap → equal TRAVEL distance — and success went 0.01 → 0.90 on complex
# and 0.09 → 1.00 on corridor without touching a single weight. The policy was not broken; it was
# being switched off mid-mission.
#
# THE UNIT ERROR. One MARLauder action is one lattice hop: nr=16px, so ≤22.63px diagonal. One IR2
# "step" is a WAYPOINT DECISION — the robot teleports to the chosen node and the LiDAR fires only
# on arrival (IR2 env.py:139-142), so their step covers arbitrary distance. Capping both systems at
# the same STEP COUNT is therefore not the same budget at all:
#     cell          our cap    our travel   IR2's actual travel   ratio
#     hybrid_M2     196 steps     ~3.6k px          3422 px       ~1.00
#     corridor_M2   196 steps     ~3.6k px          7204 px       ~0.50
#     complex_M2    384 steps     ~7.2k px         16966 px       ~0.43
# which predicts exactly where we passed and failed. Distance is the unit that means the same thing
# on both sides — and it is IR2's own headline metric, and the honest physical constraint for a
# robot (battery / mission time). eval/comparison/README.md had already frozen "il confronto
# temporale si fa sulla DISTANZA"; --max-travel-frac makes it operative instead of declared.
#
# WHY --max-travel-frac AND NOT --max-travel-px FOR TRAINING. train/difficult spans 3.9x in free
# area between its p50 (128k px) and p90 (495k px) maps. One flat px budget is simultaneously
# generous on half the split and starving on the other half — and the starved half is precisely
# where the completion bonus has to fire for the own-99% objective to have any gradient. A budget
# of `frac × GT-free-pixels` scales with the job. (train/easy is uniform, p50 = p90 = 128k, so the
# distinction only bites on difficult. Use the flat --max-travel-px for the comparison cells.)
#
# WHY frac = 0.06, MEASURED NOT GUESSED. scripts/probe_reachability.py, sampled actions (sampling
# is what training does, so it is what decides whether the bonus actually fires):
#   budget axis, early ckpt_020 on easy:  0.06→0.09   0.10→0.25   0.15→0.22   0.25→0.25
#   training axis, frac 0.06 on easy:     ckpt_020→0.09   ckpt_060→0.38   ckpt_100→0.81
#   difficult, entering with end-of-easy policy, frac 0.06:  0.88 (own_min mean 0.992)
# Two conclusions. (1) The termination rate SATURATES past frac 0.10, so a larger budget buys
# longer episodes, not more learning — the binding constraint above 0.10 is the policy. (2) At 0.06
# the signal bootstraps on its own during the easy phase (0.09 → 0.81), so from-scratch has a real
# gradient; and the budget stays ~1.6x the trained policy's actual spend, which keeps distance a
# genuine cost to minimise. That last point is why 0.06 beats 0.10: distance IS the headline metric,
# and a budget that never bites stops training the thing being measured.
#
# DELIBERATELY NOT CHANGED (user's calls, recorded so nobody "fixes" them later):
#   * revisit streak stays UNCAPPED. It is the strongest term in the budget report and it punishes
#     the backtracking that rendezvous requires — watch `analyze_run.py --reward-budget`, but the
#     decision to leave it alone is deliberate, not an oversight.
#   * --rdv-weight 0.10 (v10's value), NOT the CLI default 1.0, and NOT IR2's effective 1.0. IR2
#     rewards rendezvous explicitly at full weight (env.py:158 `+ new_pose_rendezvous_util`, a
#     map-delta field laid along the A* route to the teammate's believed position) AND feeds it as
#     an observation channel. We are entitled to match that, and are choosing not to: our numbers
#     are already better, so the comparison is stronger without copying their shaping.
#   * NO own-coverage reward term. Stage 0 showed the horizon was the dominant cause, so the
#     horizon fix ships alone — one change, one experiment. Measure before adding a second.
#   * --sync-weight 0. v11 proved the sync-EVENT bonus backfires (nSync 2.6→2.2, auc 0.605→0.556).
#
# STEP CAPS ARE SAFETY NETS ONLY, sized so the TRAVEL budget is what actually binds — including in
# the eval suite, whose T is eval_env.cfg.max_episode_steps. Getting this wrong is how v12 hid its
# own failure: a step cap below the budget silently reintroduces starvation in the measurement.
#   easy      budget 128k×0.06 =  7680px ≈  415 hops  → cap 1024
#   difficult budget 540k×0.06 = 32400px ≈ 1750 hops  → cap 2048
#   eval hybrid  108k×0.06 =  6480px ≈  350 hops (fits 1024)
#   eval complex 467k×0.06 = 28020px ≈ 1515 hops (fits 2048)
#
# --eval-suite-splits is set PER PHASE, on a split where success is actually visible. v12 pinned it
# to test/complex for both phases, so under done_mode=own there was no way to see whether the bonus
# was reachable anywhere — the measurement design hid the problem as much as the horizon did.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=$(date +%Y%m%d_%H%M%S)
EASY_OUT=runs/v13_travel_easy_${TS}
DIFF_OUT=runs/v13_travel_difficult_${TS}

COMMON="--n-envs 32 --n-agents 2 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 \
  --k-epochs 4 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront \
  --radar-team-source belief --map-seed 0 \
  --done-mode own --max-travel-frac 0.06 --sync-weight 0 --rdv-weight 0.10"

echo "############ PHASE 1: EASY  (from scratch, travel budget 0.06, 2M steps) ############"
python scripts/run_train.py --split train/easy --max-episode-steps 1024 --total-steps 2000000 \
  $COMMON --eval-suite-splits test/hybrid \
  --eval-on-ckpt --eval-split test/hybrid --eval-steps 1024 \
  --out ${EASY_OUT}
echo "PHASE1_DONE easy_final=${EASY_OUT}/final.pt"

echo "############ PHASE 2: DIFFICULT (4M steps, warm-start from phase 1) ############"
python scripts/run_train.py --split train/difficult --max-episode-steps 2048 --total-steps 4000000 \
  $COMMON --eval-suite-splits test/complex \
  --eval-on-ckpt --eval-split test/complex --eval-steps 2048 \
  --init-ckpt ${EASY_OUT}/final.pt \
  --out ${DIFF_OUT}
echo "PHASE2_DONE diff_final=${DIFF_OUT}/final.pt"
echo "PIPELINE_DONE best=${DIFF_OUT}/ckpt_best.pt"

echo
echo "==================================================================================="
echo "Read the run WITHOUT W&B (metrics.jsonl is written every iteration now):"
echo "  python scripts/analyze_run.py ${EASY_OUT} ${DIFF_OUT} --compare"
echo "  python scripts/analyze_run.py ${DIFF_OUT} --reward-budget --own-coverage"
echo
echo "Then the comparison, at a NON-BINDING flat travel budget per cell:"
echo "  python scripts/eval_comparison.py --ckpt ${DIFF_OUT}/ckpt_best.pt \\"
echo "      --splits hybrid --agents 2 4 --max-travel-px 8000  --tag v13"
echo "  python scripts/eval_comparison.py --ckpt ${DIFF_OUT}/ckpt_best.pt \\"
echo "      --splits corridor --agents 2 4 --max-travel-px 14000 --tag v13"
echo "  python scripts/eval_comparison.py --ckpt ${DIFF_OUT}/ckpt_best.pt \\"
echo "      --splits complex --agents 2 4 --max-travel-px 30000 --tag v13"
echo "  python eval/comparison/analyze.py --tag v13"
echo
echo "TO BEAT — v12's ckpt_best under the same protocol (max_dist LOWER is better, rest HIGHER),"
echo "and IR2's frozen baseline (eval/comparison/README.md):"
echo "  cell          v12 max_dist  expl  succ  conn  |  IR2 max_dist  expl  succ  conn"
echo "  hybrid_M2            2343  0.998  1.00  0.97  |         3422  0.997  1.00  0.89"
echo "  hybrid_M4            1841  0.999  1.00  0.65  |         2413  0.999  1.00  0.60"
echo "  corridor_M2          4781  0.998  1.00  0.94  |         7204  0.976  0.76  0.66"
echo "  corridor_M4          3830  1.000  1.00  0.69  |         5352  0.992  0.93  0.56"
echo "  complex_M2          14536  0.994  0.98  0.83  |        16966  0.975  0.72  0.59"
echo "  complex_M4          10671  0.998  1.00  0.49  |        13403  0.982  0.80  0.27"
echo
echo "WATCH in metrics.jsonl (analyze_run.py --reward-budget / --own-coverage):"
echo "  reward/completion    NEW telemetry. If it is identically 0 the objective is unreachable"
echo "                       again and the run is v12 all over — stop immediately."
echo "  eval/success_rate    must climb. The probe says 0.81 is reachable on easy by end of phase 1."
echo "  metric/own_cov_gap   union minus the weakest robot. THE number this run exists to shrink."
echo "  reward/revisit       per-episode total, printed next to reward/novel. Uncapped by choice."
echo "==================================================================================="
