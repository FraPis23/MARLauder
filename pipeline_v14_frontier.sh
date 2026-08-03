#!/bin/bash
# v1.4 — v13's training setup, unchanged, on a REPAIRED pathfront belief.
#
# WHY THIS EXISTS. v13 was stopped 72 iters into phase 2 because the teammate belief was visibly
# wrong in the inspector. It has been rebuilt step by step against scripts/pf_scenarios.py (8
# scripted cases, Sigma p = 1.0000 on every step of every page) and then verified on real maps with
# the trained policy. NOTHING about the training recipe changed: same budget, same rewards, same
# caps, same schedule as pipeline_v13_travel.sh. The only difference is what the policy SEES in
# feat[4]/feat[6], because --radar-team-source belief transports the belief field itself.
#
# WHAT WAS ACTUALLY WRONG (each fixed and measured, one change at a time):
#   * a hypothesis was killed the moment COMM reached its doorway, not when the door was looked
#     through — `tgt_seen` now `seen & ~frontier`. Whole zones were being written off unseen.
#   * mass cleared inside the comm blob was spread over the entire surviving field, draining the
#     branch the robot was walking into its sibling; it now walks to the NEAREST opening.
#   * a spent hypothesis (target no longer an opening) still accepted redistributed mass and
#     teleported it across the map; `arrived` now requires `tgt_front`.
#   * two hard mass bugs: a Sigma p 1.0 -> 0.0 leak into already-seeded hypothesis weights, and a
#     gate_eps-guarded division that turned a 0.249 field into [-44.6, +11.4] while Sigma p still
#     read 1.0000 because the halves cancelled.
#   * the terminal corner (nothing left to claim) now equidistributes over known-but-unheard ground
#     instead of parking mass on refuted floor; only a truly empty map deletes it (alive = False).
#
# THE CHANGE THAT MATTERS MOST FOR TRAINING — what counts as an OPENING.
#   pf_frontier_min_unknown 4 -> 1   and   pf_frontier_min_util 0 -> 1e-6 (NEW gate, STRICTLY > 0).
# The old rule was the unknown-8-neighbour COUNT alone. That keeps every node whose unknown
# neighbours sit BEHIND A WALL — nothing ever reveals them, so walking over the node never clears
# it and the belief absorbs there forever. Measured, test/hybrid #1 at t=60: 29 of 32 nodes flagged
# as frontiers had a pre-diffusion seed of EXACTLY 0.0, holding 0.2442 of the belief. The same
# count also drops REAL openings as soon as the robot gets close enough to reveal a few of their
# neighbours: node #362, seed 0.5428 (61% of its ribbon still there), dropped between t=73 and
# t=74, and the belief moved to node #809, seed 0.1439, whose only merit was being far away.
#
# THE GATE IS STRICTLY-POSITIVE AND NOT A TUNED THRESHOLD — this cost a run, so it is written down.
# v14's FIRST attempt shipped 0.10, calibrated on that one test/hybrid episode (where genuine
# openings scored 0.14-0.90). It did not generalise, and the run was stopped at iteration 60 after
# being below v13 on all six eval blocks (auc 0.464 vs 0.763, ownAUC 0.363 vs 0.712, succ 0.00 vs
# 0.22 at it=60). The cause, measured with a FIXED policy over 4 easy maps / 1600 steps
# (scripts/pf_obs_diag.py) — frontier nodes . belief alive out of comm . feat[4] nonzero . mass on
# real openings:
#   v13 belief, no gate    21.1 . 99.4% . 24.7% . 0.184     <- control
#   new belief, no gate    19.3 . 99.4% . 24.6% . 0.257     <- the belief FIXES cost nothing
#   gate 0.10 (v14 att.1)   5.5 . 81.0% .  5.6% . 0.542     <- the regression
#   gate 0.02               6.1 . 84.0% . 14.9% . 0.682
#   gate 1e-6 (THIS RUN)    6.3 . 82.8% . 24.6% . 0.686     <- coverage restored, quality kept
# feat[4] is the BF potential the policy navigates toward the teammate by. Starving it from a
# quarter of the known nodes to a twentieth is what cost the run. `> 0` removes exactly the
# zero-ribbon nodes, which is the entire bug, and nothing else.
# STILL OPEN, not addressed here: belief liveness is 82.8% vs the control's 99.4%. With fewer
# openings a hypothesis more often has no frontier target left, so the terminal corner fires. One
# change per run — measure this one first.
#
# Both gates are CLI flags now and are passed explicitly below, so params.json records them. They
# used to be EnvCfg-only defaults, which made two runs with different frontier semantics
# indistinguishable after the fact.
#
# EVERY OTHER DECISION IS v13's AND IS DELIBERATE — do not "fix" these:
#   * --max-travel-frac 0.06 (measured, not guessed: see pipeline_v13_travel.sh's header).
#   * revisit streak stays UNCAPPED.
#   * --rdv-weight 0.10, not the CLI default 1.0.
#   * NO own-coverage reward term. One change per experiment.
#   * --sync-weight 0 (v11 proved the sync-EVENT bonus backfires).
#   * step caps are safety nets sized so the TRAVEL budget is what binds: easy 1024, difficult 2048.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=$(date +%Y%m%d_%H%M%S)
EASY_OUT=runs/v14_frontier_easy_${TS}
DIFF_OUT=runs/v14_frontier_difficult_${TS}

COMMON="--n-envs 32 --n-agents 2 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 \
  --k-epochs 4 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront \
  --radar-team-source belief --map-seed 0 \
  --pf-frontier-min-unknown 1 --pf-frontier-min-util 1e-6 \
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
echo "  python scripts/analyze_run.py ${EASY_OUT} ${DIFF_OUT} --compare"
echo "  python scripts/analyze_run.py ${DIFF_OUT} --reward-budget --own-coverage"
echo
echo "v13's numbers on the SAME recipe with the OLD belief — this is the honest control:"
echo "  easy      2.0M steps  score 0.3892  auc 0.9243  succ 0.9375  ownAUC 0.8984  ownGap 0.1579"
echo "  difficult 0.59M steps score 0.3809  auc 0.7633  succ 0.5625  ownAUC 0.7057  ownGap 0.1048"
echo "  (phase 2 was STOPPED at 72/488 iters, so only its early trajectory is comparable)"
echo
echo "WATCH:  reward/completion  identically 0 => objective unreachable, stop (v12's failure mode)"
echo "        eval/success_rate  must climb; 0.81 is reachable on easy by end of phase 1"
echo "        metric/own_cov_gap THE number this run exists to shrink"
echo "==================================================================================="
