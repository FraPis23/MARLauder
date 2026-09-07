#!/usr/bin/env bash
# NULL CONTROL for the sync-bonus M-scaling — identical to stage 4 except --sync-weight-m-scale 0.0.
#
# WHY IT EXISTS. Stage 4 beat its stage-3 control on test/complex (score 0.4445+-0.0104 against
# 0.4281+-0.0157; steps_to_90 278.6+-1.6 against 300.8+-25.5, i.e. -7.4%) and, more importantly,
# passed unharmed through the window in which stage 3 had collapsed: stage 3 at it=90 (fifty
# iterations after its warm-start point) drops to success_rate 0.000 for ~50 iterations and never
# fully recovers, while stage 4, restarted from exactly that checkpoint, is positive on 90 of 110
# ticks in the same window with no collapse.
#
# BUT STAGE 4 CHANGED TWO THINGS AT ONCE:
#   (1) the reward      — the sync bonus scaled by (2/M)^1, i.e. halved at M=4;
#   (2) the optimizer   — it restarts from ckpt_best@40 with Adam ZEROED (--init-ckpt loads the
#       weights, not the optimizer state), so the moments are lost and the LR is at full value.
# Either one alone explains "it did not collapse". With a single arm the cause is not attributable,
# and that is exactly the kind of claim this project has already had to retract once (the relay
# "costs 25% of success" — the null control alone accounted for -11% of it).
#
# WHAT THIS RUN ISOLATES. Same starting checkpoint, same Adam restart, same seed, same maps, same
# iteration count, same eval suite (test/complex, the only one stage 4 had — note that
# --eval-suite-splits REPLACES the training split's suite rather than adding to it, so adding
# train/difficult here would break the pairing with stage 4). The only difference is a = 0.0, which
# returns the sync weight to the unscaled stage-3 value.
#
# HOW TO READ THE RESULT, decided BEFORE looking:
#   - the control COLLAPSES in the it 40-60 window (eval/success_rate -> ~0, eval/score below the
#     starting point)  -> it is the reward. Normalising the sync bonus prevents the collapse.
#                         A defensible claim.
#   - the control does NOT collapse -> it is the optimizer restart (or at any rate not the reward).
#                         Stage 4 remains better by measurement (-7.4% steps_to_90 stands on its
#                         own, measured over 3 repeats), but the mechanism has to be rewritten as
#                         "restarting Adam from a good checkpoint avoids the collapse".
#   In both cases n = 1 episode per arm: stage 3's collapse is ONE event. A single non-replication
#   does not refute it, it weakens it.
#
# GPU MEMORY: 32 envs at M=4 with 6 hops fit in 15.47 GiB by a hair, and stage 4's first backward
# went OOM through fragmentation (1.61 GiB reserved and unallocated). expandable_segments is
# required. Do NOT lower --n-envs and do NOT raise --minibatches: that would change the batch and
# break the pairing with stage 4, which is the only reason this run exists.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
INIT_CKPT=${INIT_CKPT:?set INIT_CKPT=<stage-3 ckpt_best, the same warm start stage 4 used>}
OUT=${OUT:-runs/ablation_sync_scaling_${TS}}

if [ ! -f "${INIT_CKPT}" ]; then
  echo "ERROR: warm-start checkpoint not found: ${INIT_CKPT}"; exit 1
fi

# Line for line identical to stage 4's saved command, except --sync-weight-m-scale (1.0 -> 0.0).
# 0.0 is also the dataclass default; it is explicit here because the value IS the experiment.
python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.030 \
  --n-agents 4 \
  --novel-scan-weight 1.65 \
  --total-steps 850000 \
  --rdv-urgency-mode budget --rdv-urgency-start 0.5 \
  --n-envs 32 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 \
  --k-epochs 4 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront \
  --radar-team-source belief --map-seed 0 \
  --pf-frontier-min-unknown 1 --pf-frontier-min-util 1e-6 \
  --done-mode own \
  --gamma 0.998 \
  --rdv-weight 0.10 --rdv-clamp-pos --rdv-urgency-T 256 \
  --sync-weight 0.25 --sync-recv-ratio 0.5 --sync-min-gap 32 \
  --sync-weight-m-scale 0.0 \
  --trace-steps 512 \
  --eval-suite-splits test/complex \
  --eval-on-ckpt --eval-split test/complex --eval-steps 768 --eval-every 10 \
  --init-ckpt ${INIT_CKPT} \
  --out ${OUT}
echo "DONE  final=${OUT}/final.pt  best=${OUT}/ckpt_best.pt"

# WHAT TO LOOK AT, against the stage-4 run (same init, same restart, same everything else).
#   eval/success_rate over it 40-60  -> THE measurement. Stage 3 went to 0.000 there; stage 4 did not.
#   eval/score                       -> stage 4 peaked at 0.4386 at it=80. Does the control reach it?
#   reward/sync (share)              -> stage 4 landed at 5.2% of the reward budget (target: the
#                                       5.5% seen at M=2). Here it must return to ~9-10%, as in
#                                       stage 3. If it does not, the flag never reached the env:
#                                       check params.json.
#   metric/own_cov_gap, eval/own_coverage_final -> the known risk of scaling sync down under
#                                       done_mode=own is paying for it in map sharing. Stage 4 did
#                                       not pay (own_cov 0.9958). If the control does BETTER here,
#                                       stage 4's -7.4% steps_to_90 had a price it was hiding.
#
# Final comparison always with the same instrument, never by eye on training metrics:
#   python scripts/score_ckpts.py --ckpt <stage4>/ckpt_best.pt ${OUT}/ckpt_best.pt \
#       --splits train/difficult,test/complex --repeats 3
