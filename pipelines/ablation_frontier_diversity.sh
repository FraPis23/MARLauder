#!/usr/bin/env bash
# Frontier-diversity ablation — stage 4 repeated IDENTICALLY, with --div-weight 1.0 added.
#
# A clean A/B against the released stage-4 run: same --init-ckpt, same 850000 steps, same COMMON
# block, same eval suite. The only difference is the auxiliary frontier-level diversity loss
# (train/mappo.py, MAPPOCfg.div_weight).
#
# WEIGHT: 1.0 is measured, not guessed — on the smoke run div_loss sits at ~0.004-0.006 against a
# pg_loss of ~0.002-0.019, i.e. the two terms are comparable in magnitude.
#
# comm_relay: the stage-3 checkpoint predates the flag (absent from its params.json), but
# train_args.py sets set_defaults(comm_relay=True), so this run starts with the relay ON, exactly
# as stage 4 did.
#
# MEMORY: 15562 MiB of 16303 measured at this configuration with --div-weight active. Do not raise
# --n-envs and do not lower --minibatches: both change the batch and the numbers stop being
# comparable with stage 4.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
INIT_CKPT=${INIT_CKPT:?set INIT_CKPT=<stage-3 ckpt_best, the same warm start stage 4 used>}
OUT=${OUT:-runs/ablation_frontier_diversity_${TS}}

COMMON="--n-envs 32 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 --k-epochs 4 \
 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront --radar-team-source belief \
 --map-seed 0 --pf-frontier-min-unknown 1 --pf-frontier-min-util 1e-6 --done-mode own \
 --gamma 0.998 --rdv-weight 0.10 --rdv-clamp-pos --rdv-urgency-T 256 \
 --sync-weight 0.25 --sync-recv-ratio 0.5 --sync-min-gap 32 --trace-steps 512 --eval-on-ckpt"

python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.030 \
  --n-agents 4 --novel-scan-weight 1.65 --total-steps 850000 \
  --rdv-urgency-mode budget --rdv-urgency-start 0.5 ${COMMON} \
  --sync-weight-m-scale 1.0 \
  --div-weight 1.0 \
  --eval-suite-splits test/complex --eval-split test/complex --eval-steps 768 --eval-every 10 \
  --init-ckpt ${INIT_CKPT} \
  --out ${OUT}

echo "DONE -> ${OUT}"
