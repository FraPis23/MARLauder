#!/bin/bash
# v1.2 — v10 FROM SCRATCH, trained under the IR2 termination rule.
#
# WHY THIS EXISTS. v11/R2 tried to buy rendezvous with a sync-event reward and failed: nSync went
# DOWN (2.6 -> 2.2), sync_gap went UP (0.087 -> 0.149) and exploration regressed (auc 0.605 ->
# 0.556). Three eval blocks during that run moved monotonically — the better it explored, the less
# it met. That is the expected outcome once you notice the objective never needed the exchange:
# novel_scan pays cells that are new to the TEAM UNION, and the episode ended when the UNION hit
# 99%, so a map that never reached the other robot cost nothing. Paying a bonus on top of an
# objective that rewards dispersion just gets outvoted.
#
# So this run stops bribing and changes the objective instead. --done-mode own is IR2's rule
# (env.check_done): the episode ends when EVERY robot holds 99% of the map in its OWN belief. Now
# the exchange is not optional — it is the only way to finish, collect completion_bonus=10, and
# stop paying step_penalty. And it is the same rule the comparison measures, so what gets trained
# is what gets scored (IR2's `success` column IS this flag, test_multi_robot_worker.py:122).
#
# WHAT IS OFF, AND WHY EXPLICITLY RATHER THAN BY DEFAULT:
#   --sync-weight 0   the v11 sync-event reward. Default is already 0; passed anyway so the run's
#                     params.json records the intent instead of leaving it to a default.
#   --rdv-weight 0.10 v10's value. THIS ONE MUST BE PASSED: the v11 CLI default is 1.0, so omitting
#                     it silently trains a different config from v10. Kept at 0.10 rather than 0
#                     because under --done-mode own the team still has to physically meet, and a
#                     small approach shaping is the only dense signal pointing that way.
#
# WHAT STAYS FROM v11 (no flag exists to remove it; that is deliberate):
#   * feat[4] peak-normalization, feat[6] row-normalization, b_util soft squash. These are not
#     rendezvous features, they are fixes to normalization the docstrings already promised. The v10
#     policy evaluated under this code scores auc 0.605 vs 0.589 under intact v10 code, so they do
#     not cost anything.
#   * critic_global slots sync_surplus / sync_staleness. Under --done-mode own these predict the
#     return MORE directly than the idle_frac / imbalance they replaced: "how much map is still
#     pending exchange" is literally the distance to termination.
#
# WATCH reward/revisit. The revisit streak is UNCAPPED (1 + 0.5·(streak−1), measured streak 89 →
# ×45, a −44 per-episode tail against novel +17). Under the own rule the team MUST backtrack to
# meet, and backtracking is what that penalty punishes — the reward structure now fights the
# termination rule. If reward/revisit p95 dominates the return in train.log, rerun with
# --revisit-streak-cap 4.0 appended to COMMON.
#
# REACHABILITY CHECK (the one measurement that exists): the v10 checkpoint, trained under the UNION
# rule and evaluated under the OWN rule on test/hybrid @196 steps, already reaches own-99% on 99 of
# 100 maps. So completion_bonus does fire on the easier splits. On train/difficult @384 it is
# UNTESTED — if eval/success_rate sits at ~0 through phase 2, the bonus is dead weight there and
# the stage needs either more steps per episode or an own-coverage reward term.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=$(date +%Y%m%d_%H%M%S)
EASY_OUT=runs/v12_ir2_easy_${TS}
DIFF_OUT=runs/v12_ir2_difficult_${TS}

# --eval-suite-splits: v10 did NOT set this, and --eval-split only redirects the milestone GIF —
# the in-training eval SUITE (and therefore best-ckpt selection) ran on train_eval_name = cfg.split,
# i.e. the TRAINING maps. That is why v10 reported eval/score +0.369 while the same checkpoint
# measures auc 0.605 on test/complex. Set explicitly so the best ckpt is chosen on held-out maps.
# --map-seed 0: without it the map stream is fresh OS entropy, so two runs with the same --seed
# still see different maps and any comparison measures map luck.
COMMON="--n-envs 32 --n-agents 2 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 \
  --k-epochs 4 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront \
  --radar-team-source belief --map-seed 0 \
  --done-mode own --sync-weight 0 --rdv-weight 0.10 \
  --eval-on-ckpt --eval-split test/complex --eval-suite-splits test/complex --eval-steps 512"

echo "############ PHASE 1: EASY  (128-step episodes, 2M steps) ############"
python scripts/run_train.py --split train/easy --max-episode-steps 128 --total-steps 2000000 \
  $COMMON --out ${EASY_OUT}
echo "PHASE1_DONE easy_final=${EASY_OUT}/final.pt"

echo "############ PHASE 2: DIFFICULT (384-step episodes, 4M steps, warm-start) ############"
python scripts/run_train.py --split train/difficult --max-episode-steps 384 --total-steps 4000000 \
  $COMMON --init-ckpt ${EASY_OUT}/final.pt --out ${DIFF_OUT}
echo "PHASE2_DONE diff_final=${DIFF_OUT}/final.pt"

echo "############ BEST-CKPT EVAL (test/complex, 512 steps, 32 maps) ############"
python scripts/eval_best.py --run ${DIFF_OUT} --split test/complex --steps 512 --n-maps 32
echo "PIPELINE_DONE best=${DIFF_OUT}/ckpt_best.pt"

echo
echo "==================================================================================="
echo "Then the IR2 comparison, on the SAME rule this was trained under:"
echo "  python scripts/eval_comparison.py --ckpt ${DIFF_OUT}/ckpt_best.pt --tag v12"
echo "  python eval/comparison/analyze.py --tag v12"
echo
echo "IR2 baseline to beat (mean over 100 maps/cell; max_dist LOWER is better, rest HIGHER):"
echo "  cell           max_dist  steps  explored  success  conn"
echo "  hybrid_M2          3422   88.7     0.997     1.00  0.89"
echo "  hybrid_M4          2413   48.4     0.999     1.00  0.60"
echo "  corridor_M2        7204  141.9     0.976     0.76  0.66"
echo "  corridor_M4        5352   92.5     0.992     0.93  0.56"
echo "  complex_M2        16966  277.8     0.975     0.72  0.59"
echo "  complex_M4        13403  203.5     0.982     0.80  0.27"
echo
echo "In-training numbers to watch in train.log / W&B:"
echo "  eval/success_rate    now = 'every robot got 99% of the map'. ~0 through phase 2 means the"
echo "                       completion_bonus never fires and the own rule is only lengthening"
echo "                       episodes — stop and add an own-coverage reward term."
echo "  metric/own_cov_gap   union minus the weakest robot. THE number this run exists to shrink."
echo "  reward/revisit       if its p95 dominates, rerun with --revisit-streak-cap 4.0"
echo "==================================================================================="
