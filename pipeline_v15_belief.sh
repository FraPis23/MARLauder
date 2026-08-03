#!/bin/bash
# v1.5 — v14's training recipe, UNCHANGED, on the cleaned-up pathfront belief.
#
# The COMMON block below is byte-identical to pipeline_v14_frontier.sh. Nothing about the training
# setup moved: same budget, same rewards, same caps, same schedule, same frontier gates. Read that
# file's header for why each of those values is what it is — none of it is restated here.
#
# WHY A NEW NAME. v14 died at iteration 98/488 of phase 2 (OOM, an external cause, not the recipe).
# Since it was launched the belief module itself changed, so a rerun under the v14 name would put
# two DIFFERENT beliefs in run dirs distinguishable only by timestamp — the exact failure v14's own
# header warns about for the frontier gates. What changed in env/teammate_belief_pathfront.py:
#   * cluster_frontiers DELETED. It collapsed a 12-17-node frontier arc into ONE centroid
#     representative, so the transit ring could not exist: measured t=62 hybrid, 20 frontier nodes
#     -> 4 components -> 1 dot. It had been added back when pf_frontier_min_unknown was 4; at 1 it
#     produces exactly the giant cluster it was meant to prevent.
#   * freeze_hypotheses now picks ONE hypothesis PER OPENING NODE by spaced greedy on
#     utility/dist(lkp->f), forbidding a new pick within min_sep_hops=3 of one already taken. Plain
#     top-Kf does not work: distance in the denominator selects the Kf nearest openings, which all
#     arrive on the step they are born. Result on hybrid a1: 2 -> 4 dots, tail [4,3,2,1,1,0].
#   * the `sensed` gate is GONE. freeze and the kill test now use the SAME opening set
#     (= the current frontier). The gate was structurally wrong, not mistuned: a freshly revealed
#     frontier is ALWAYS inside the sensor footprint that revealed it, so `frontier & ~sensed`
#     excluded every opening the robot had just created. Measured on the bench: 02 t=9, mass 0.510
#     teleported 14 cells past an unentered door; 03 t=22, the whole 0.4826 crossed to the sibling
#     fork the step its own fork opened.
#   * the two competing push implementations (nearest-opening watershed, global attract field) are
#     replaced by ONE local bounded walk, _spread_to_openings: freed mass walks known-free edges
#     with weights (utility + push_floor) and is absorbed where it touches an opening. Conserves
#     mass node-for-node (landed + boxed == mass).
# Verified on scripts/pf_scenarios.py: 9 pages, Sigma p = 1.0000 on every step, and
# "seen & not frontier" = 0.0000 on every step (no mass on ground the radio already covers).
#
# SUCCESS = 99% PER AGENT. --done-mode own + cfg.done_explored_thresh 0.99 (EnvCfg default, no CLI
# flag) => done_frac = own_cov.amin(dim=1), i.e. EVERY agent's own map must reach 99%, not the team
# union. This is the IR2 rule and the number the comparison is against.
#
# KNOWN AND DELIBERATELY NOT ADDRESSED IN THIS RUN (one change per run):
#   * 35-55% of the belief mass sits on non-frontier known floor (the isotropic diffusion in
#     section 4). Removing it is a model change, not a fix.
#   * release of a consumed opening's mass is LOCAL, so it goes to whoever is adjacent rather than
#     being split over the live openings by u/dist. Visible on 03b: two openings of identical
#     utility 0.056 split 0.11 / 0.68.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=$(date +%Y%m%d_%H%M%S)
EASY_OUT=runs/v15_belief_easy_${TS}
DIFF_OUT=runs/v15_belief_difficult_${TS}

COMMON="--n-envs 32 --n-agents 2 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 \
  --k-epochs 4 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront \
  --radar-team-source belief --map-seed 0 \
  --pf-frontier-min-unknown 1 --pf-frontier-min-util 1e-6 \
  --done-mode own --max-travel-frac 0.06 --sync-weight 0 --rdv-weight 0.10 \
  --trace-steps 512"

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

# --- 2026-08-01 ------------------------------------------------------------------------------
# --trace-steps 512 added AFTER the first full launch. Phase 1 completed; phase 2 was OOM-KILLED
# by the HOST kernel at it=98/488 (anon-rss 27.3 GB of 30 GB), exactly at the 20% milestone, right
# after "[ckpt] ckpt_020.pt / [eval] ...gif / [trace] ckpt_020_m0 steps=1813". v14 died at the same
# iteration for the same reason — earlier attributed to concurrent traces I was running, which was
# wrong: nothing else was running this time.
# eval/trace.py builds the entire episode in Python objects (per-step records + 2 x 1813 RGB frames
# at 500x500x3) and serialises at the end: 3.1 GB of JSON on disk, ~25 GB live. v10 never hit this
# because its episodes were 384 steps and its traces 748 MB. The training loop itself is untouched
# and stayed on GPU throughout (74 sps, 7.5/16 GiB VRAM, flat host RAM for 98 iterations).
# The cap is on the milestone GIF + inspector trace ONLY. The eval SUITE still runs full-length
# episodes, so eval/score and ckpt_best selection are unchanged and stay comparable to v13/v14.
