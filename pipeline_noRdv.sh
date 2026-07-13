#!/bin/bash
# ABLATION: pure exploration — rendezvous reward OFF (--rdv-weight 0) AND actor blind to
# teammates (--no-teammate-obs: agent_scalars, feat[4] team potential, feat[6] radar-teammate
# all zeroed). No approach/avoid reasoning toward teammates possible. Map fusion at comm and
# the privileged critic (geo_pair) are unchanged. Tests whether loops come from rdv shaping
# and/or stale-lkp teammate repulsion.
# Eval at each milestone: test/complex, 512-step episodes.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=$(date +%Y%m%d_%H%M%S)
EASY_OUT=runs/noRdv_easy_${TS}
DIFF_OUT=runs/noRdv_difficult_${TS}
COMMON="--n-envs 32 --n-agents 2 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 --k-epochs 4 --rdv-weight 0.0 --no-teammate-obs --radar-gamma 0.97 --radar-util-norm 3 --eval-on-ckpt --eval-split test/complex --eval-steps 512"

echo "############ PHASE 1: EASY  (128-step episodes, 2M steps) ############"
python scripts/run_train.py --split train/easy --max-episode-steps 128 --total-steps 2000000 \
  $COMMON --out ${EASY_OUT}
echo "PHASE1_DONE easy_final=${EASY_OUT}/final.pt"

echo "############ PHASE 2: DIFFICULT (384-step episodes, 4M steps, warm-start) ############"
python scripts/run_train.py --split train/difficult --max-episode-steps 384 --total-steps 4000000 \
  $COMMON --init-ckpt ${EASY_OUT}/final.pt --out ${DIFF_OUT}
echo "PIPELINE_DONE diff_final=${DIFF_OUT}/final.pt"
