#!/bin/bash
# v0.9 FULL MODEL: GAT + value-field + teammate obs (feat[4]/[6], agent_scalars) + rdv reward,
# with the v0.9 env fixes: same-target-node arbitration (loser holds, winner moves),
# partial-advance winner-blocked resolution, cumulative stall streak (capped 4×) and
# cumulative recent-revisit streak (uncapped). Eval each milestone + final best-ckpt
# selection on test/complex @ 512 steps, 32 maps.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=$(date +%Y%m%d_%H%M%S)
EASY_OUT=runs/v09_easy_${TS}
DIFF_OUT=runs/v09_difficult_${TS}
COMMON="--n-envs 32 --n-agents 2 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 --k-epochs 4 --rdv-weight 0.10 --radar-gamma 0.97 --radar-util-norm 3 --eval-on-ckpt --eval-split test/complex --eval-steps 512"

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
