#!/bin/bash
# RADAR A/B arm: same easy->difficult pipeline as pipeline_rdv.sh, but with the far-frontier
# radar fix: --radar-gamma 0.97 --radar-util-norm 3 (stall diagnosis 2026-07-09: at 0.92/8 a
# frontier 45 hops beyond the horizon contributes ~0.4%/node -> agents stall; 0.97/3 -> ~8%).
# Control arm = runs/rdv_difficult_20260707_203151 (ckpt_best, radar 0.92/8).
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=$(date +%Y%m%d_%H%M%S)
EASY_OUT=runs/radar097_easy_${TS}
DIFF_OUT=runs/radar097_difficult_${TS}
COMMON="--n-envs 32 --n-agents 2 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 --k-epochs 4 --rdv-weight 0.10 --radar-gamma 0.97 --radar-util-norm 3 --eval-on-ckpt"

echo "############ PHASE 1: EASY  (128-step episodes, 2M steps) ############"
python scripts/run_train.py --split train/easy --max-episode-steps 128 --total-steps 2000000 \
  $COMMON --out ${EASY_OUT}
echo "PHASE1_DONE easy_final=${EASY_OUT}/final.pt"

echo "############ PHASE 2: DIFFICULT (384-step episodes, 4M steps, warm-start) ############"
python scripts/run_train.py --split train/difficult --max-episode-steps 384 --total-steps 4000000 \
  $COMMON --init-ckpt ${EASY_OUT}/final.pt --out ${DIFF_OUT}
echo "PIPELINE_DONE diff_final=${DIFF_OUT}/final.pt"
