#!/usr/bin/env bash
# Full four-stage reproduction of the M=4 policy used in the IR2 comparison.
#
# The released policy is NOT an independent training run: it is the last of four chained stages,
# each warm-started from the previous one. This script puts the four exact commands back in order,
# reconstructed from the `cmd` field saved in each stage's params.json rather than retyped.
#
#   stage 1  easy         train/easy       M=2   from scratch     244 it     5.2 h
#   stage 2  difficult    train/difficult  M=2   <- 1/final.pt    732 it    37.8 h
#   stage 3  m4           train/difficult  M=4   <- 2/final.pt    ~60 it    ~8 h    (see NOTE 1)
#   stage 4  sync-scaled  train/difficult  M=4   <- 3/ckpt_best   103 it    17.7 h
#                                                        TOTAL   ~69 h on a 16 GB RTX 5080
#
# NOTE 1 — STAGE 3 IS DELIBERATELY SHORTENED, AND THIS IS NOT A SHORTCUT. The original run was
# launched with --total-steps 6000000 (732 it), ran to iteration 274 and was stopped by hand. The
# only artifact stage 4 consumes is its `ckpt_best`, which sits at iteration 40. The eval ticks of
# that original run (it 10..60) were .402 .432 .432 **.438** .433 .434 — the maximum is at it=40
# whether you stop at 60 or run to 274, so 60 iterations produce THE SAME ckpt_best in 8 hours
# instead of 44. It also stays clear of the region where stage 3 collapses (it 90-140, success
# rate -> 0.000). The --total-steps below is therefore 500000 (~61 it), NOT the 6000000 of the
# saved command.
#
# NOTE 2 — THIS REPLICATION IS NOT BIT-EXACT, AND CANNOT BE. `torch.manual_seed(0)` is set
# (train/driver.py) but `cudnn.benchmark = True` picks algorithms by timing them, and
# `use_deterministic_algorithms` is not enabled. Two identical runs diverge. What to expect is a
# STATISTICALLY equivalent policy, not identical weights.
#
# NOTE 3 — REDO THE SELECTION, DO NOT INHERIT IT. `ckpt_best` is chosen from the maximum of a
# SINGLE eval tick, and the ticks near stage 4's peak sit inside the noise: .433 (it60) .421 (it70)
# .439 (it80) — a spread of .006 against a measured .015 noise floor on eval/score. On a rerun the
# peak will almost certainly land on a different iteration of the same plateau. After stage 4:
#
#   python scripts/score_ckpts.py --ckpt <OUT4>/ckpt_0{40,60,80,100}.pt <OUT4>/ckpt_best.pt \
#       --splits train/difficult,test/complex --repeats 3 --n-agents 4 \
#       --max-travel-frac 0.040 --comm-relay
#
# and keep the best on the OUTCOME BLOCK (coverage_auc, steps_to_90, success, own_cov), NOT on
# eval/score: the composite is dominated by contrib_imbalance_norm, which does not exist in the
# IR2 CSVs and therefore cannot arbitrate the comparison.
#
# NOTE 4 — MEMORY. 32 envs at M=4 with 6 hops fit in 15.47 GiB by a hair. Stages 3 and 4 go OOM on
# the first backward without expandable_segments. Do NOT lower --n-envs and do NOT raise
# --minibatches: both change the batch, and the numbers stop being comparable with the published
# ones.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
R=${R:-runs/replication_${TS}}

COMMON="--n-envs 32 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 --k-epochs 4 \
 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront --radar-team-source belief \
 --map-seed 0 --pf-frontier-min-unknown 1 --pf-frontier-min-util 1e-6 --done-mode own \
 --gamma 0.998 --rdv-weight 0.10 --rdv-clamp-pos --rdv-urgency-T 256 \
 --sync-weight 0.25 --sync-recv-ratio 0.5 --sync-min-gap 32 --trace-steps 512 --eval-on-ckpt"

echo "=== stage 1/4 — easy, M=2, from scratch"
python scripts/run_train.py --split train/easy --max-episode-steps 256 --max-travel-frac 0.040 \
  --total-steps 2000000 --n-agents 2 ${COMMON} \
  --eval-suite-splits test/hybrid --eval-split test/hybrid --eval-steps 256 \
  --out ${R}/s1_easy

echo "=== stage 2/4 — difficult, M=2, warm-started from stage 1"
python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.040 \
  --total-steps 6000000 --n-agents 2 --rdv-urgency-mode budget --rdv-urgency-start 0.5 ${COMMON} \
  --eval-suite-splits test/complex --eval-split test/complex --eval-steps 768 \
  --init-ckpt ${R}/s1_easy/final.pt \
  --out ${R}/s2_difficult

echo "=== stage 3/4 — M=4, shortened: only its ckpt_best@40 is consumed (see NOTE 1)"
python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.030 \
  --n-agents 4 --novel-scan-weight 1.65 --total-steps 500000 \
  --rdv-urgency-mode budget --rdv-urgency-start 0.5 ${COMMON} \
  --eval-suite-splits test/complex --eval-split test/complex --eval-steps 768 --eval-every 10 \
  --init-ckpt ${R}/s2_difficult/final.pt \
  --out ${R}/s3_m4

echo "=== stage 4/4 — M=4 with the sync bonus scaled by (2/M)^1"
python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.030 \
  --n-agents 4 --novel-scan-weight 1.65 --total-steps 850000 \
  --rdv-urgency-mode budget --rdv-urgency-start 0.5 ${COMMON} \
  --sync-weight-m-scale 1.0 \
  --eval-suite-splits test/complex --eval-split test/complex --eval-steps 768 --eval-every 10 \
  --init-ckpt ${R}/s3_m4/ckpt_best.pt \
  --out ${R}/s4_sync_scaled

echo "REPLICATION COMPLETE. Candidate artifact: ${R}/s4_sync_scaled/ckpt_best.pt"
echo "NOW REDO THE SELECTION (NOTE 3) — do not trust a ckpt_best taken from a single eval tick."

# REFERENCE — what a successful replication has to reproduce. Measured on the released checkpoint
# with score_ckpts.py --n-agents 4 --max-travel-frac 0.040 --comm-relay, 32 maps, 3 repeats:
#
#   test/complex     coverage_auc .8209   steps_to_90 282.5   success .979   own_cov .9953
#   train/difficult  coverage_auc .9015   steps_to_90 154.9   success .969   own_cov .9957
#
# A successful replication falls INSIDE the noise of these (score +-.015; success +-12.6 points at
# two sigma, since it is a binomial over 32 maps).
