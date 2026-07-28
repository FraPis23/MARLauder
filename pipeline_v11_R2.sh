#!/bin/bash
# v1.1 / run R2 — THE RENDEZVOUS RUN.
#
# v10 mapped well (union 0.76-0.82 on test/complex@512) but never met on purpose: 8 mostly
# accidental contacts per episode, comm duty 9-10%, longest gap 268 of 512 steps, and a measured
# rendezvous reward of −0.04 per episode against novel +17.3. Three independent causes, all fixed
# here:
#
#   1. NO OBJECTIVE TERM FOR THE EXCHANGE. rdv_dense is telescoping shaping — its net payoff over a
#      full separate→approach→meet cycle is only w·g·φ_sep (0.045 at the shipped w=0.10, and still
#      just 1.13 even at the "calibrated" w=2.5) against a measured 1.7-1.9 detour cost. Raising w
#      cannot fix that; past w≈2 it becomes a chase term. --sync-weight adds the missing objective:
#      ζ_g·(give + ρ·recv)/scan_norm paid on the RISING EDGE of comm, ≈2.55 for a sync after ~200
#      steps apart. Farm-proof by construction (see env/explorer.py::_sync_rewards and
#      scripts/14_test_sync_reward.py: tether pays once, min-gap kills flicker, and total give is
#      conserved so syncing more often never pays more).
#   2. THE ACTOR COULD BARELY SEE THE TEAMMATE. feat[4] was the Σ=1 belief with no peak
#      normalization (the docstring always claimed one), so its amplitude fell as the possible-
#      location zone grew — faintest exactly when the two had been apart longest. Measured in-window
#      peak 0.07-0.26 and the channel absent in 39-43% of steps, vs utility 0.44-0.48 present ~90%.
#      Now peak-normalized; feat[6] row-normalized; b_util given a non-saturating squash.
#   3. THE CRITIC COULD NOT PRICE A MEETING. critic_global had the teammate DISTANCE but nothing
#      about pending map surplus, so V(s) could not represent "we are about to gain a lot by
#      meeting" and an approach move's advantage stayed ≈0. idle_frac/imbalance (neither predicts
#      return) swapped in place for sync_surplus/sync_staleness — dim stays 7, so warm-start works.
#
# Also: --rdv-weight 1.0 makes a full-gate approach hop exactly reward-neutral against the step
# penalty (it rebates travel), leaving the meet-vs-explore call to the sync payoff; --map-seed 0
# pins the map stream so this is comparable to a baseline arm.
#
# NOT changed here, deliberately: the anti-loop penalties. The revisit streak is still UNCAPPED
# (measured peak 89 → ×45 → a −44 per-episode tail against novel +17). If the eval below shows
# reward/revisit p95 still dominating, rerun this exact script with --revisit-streak-cap 4.0
# appended to COMMON — the flag exists, it is off by default so R2 stays comparable to v10.
#
# WARM START: this runs the DIFFICULT stage only, from the v10 easy checkpoint (~2h instead of 7h).
# EASY_CKPT must exist on the training machine.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=$(date +%Y%m%d_%H%M%S)
OUT=runs/v11_R2_${TS}
EASY_CKPT=${EASY_CKPT:-runs/v10_easy_20260724_073601/final.pt}

if [ ! -f "${EASY_CKPT}" ]; then
  echo "ERROR: warm-start checkpoint not found: ${EASY_CKPT}"
  echo "Copy it from the v10 run, or set EASY_CKPT=<path>, or run the full easy stage first"
  echo "(2M steps on train/easy at 128-step episodes) — that turns a 2h run into ~7h."
  exit 1
fi

# NOTE --eval-suite-splits: in v10 this was NOT set, and --eval-split only redirects the milestone
# GIF/trace — the in-training eval SUITE (and therefore best-ckpt selection) runs on
# `train_eval_name = cfg.split`, i.e. the TRAINING maps (train/driver.py:598-614). So v10's reported
# eval/score=+0.369 / auc=0.739 are train/difficult numbers; the same checkpoint measures auc=0.605
# on test/complex. Setting the suite split explicitly makes best-ckpt selection happen on held-out
# maps, which is what the final eval_best reports anyway.
COMMON="--n-envs 32 --n-agents 2 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 \
  --k-epochs 4 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront \
  --radar-team-source belief --map-seed 0 \
  --sync-weight 0.25 --sync-recv-ratio 0.5 --sync-min-gap 32 --rdv-weight 1.0 \
  --eval-on-ckpt --eval-split test/complex --eval-suite-splits test/complex --eval-steps 512"

echo "############ R2: DIFFICULT (384-step episodes, 4M steps, warm-start) ############"
python scripts/run_train.py --split train/difficult --max-episode-steps 384 --total-steps 4000000 \
  $COMMON --init-ckpt ${EASY_CKPT} --out ${OUT}
echo "R2_DONE final=${OUT}/final.pt"

echo "############ BEST-CKPT EVAL (test/complex, 512 steps, 32 maps) ############"
python scripts/eval_best.py --run ${OUT} --split test/complex --steps 512 --n-maps 32
echo "PIPELINE_DONE best=${OUT}/ckpt_best.pt"
echo
echo "==================================================================================="
echo "Send back the last eval_best table row. BASELINE measured with the SAME command on"
echo "v10 ckpt_best (test/complex, 512 steps, 32 maps):"
echo
echo "  ckpt      score   ±std    auc  succ  idle   imbN  ownAUC syncGap nSync duty maxGap scoreOwn"
echo "  v10      +0.283  0.213  0.605  0.31  0.67  0.214   0.442   0.087   2.6 0.10    327   +0.120"
echo
echo "What decides R2:"
echo "  nSync   >= 5      (baseline 2.6)   — did they start meeting ON PURPOSE?"
echo "  syncGap <= 0.045  (baseline 0.087) — map the weakest robot still lacks; halve it"
echo "  ownAUC  >= 0.50   (baseline 0.442) — the real payoff: what each robot comes home with"
echo "  auc     >= 0.585  (baseline 0.605) — exploration must not regress more than 2pp"
echo "  duty    <= 0.11   (baseline 0.10)  — HARD GATE. If comm duty climbs they tethered:"
echo "                                       raise --sync-min-gap to 64 and rerun"
echo
echo "If reward/revisit p95 in train.log still dominates the return, rerun with"
echo "  --revisit-streak-cap 4.0   appended to COMMON (flag exists, off by default)."
echo "==================================================================================="
