#!/usr/bin/env bash
# Debug substrate — single autonomous agent, attention (StrategicHead) OFF, GAT-only @ 6 hops,
# guidepost bias ON, minimal coverage+anti-loop reward. Purpose: hunt env bugs in isolation
# before any cooperative / strategic-head work. See plan: substrato minimo per caccia ai bug.
#
# Usage:
#   scripts/debug_single.sh                      # full 500k debug run -> runs/dbg_m1
#   scripts/debug_single.sh --total-steps 40000 --n-envs 8 --rollout-len 64 \
#       --max-episode-steps 64 --out /workspace/MARLauder/runs/smoke_m1   # 1-min smoke
# Extra args are forwarded to run_train.py and override the defaults below.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=. python scripts/run_train.py \
    --split train/easy \
    --n-agents 1 --comm-range 0 \
    --no-strategic-head --n-hops 6 \
    --n-envs 16 --rollout-len 256 --max-episode-steps 256 \
    --total-steps 500000 \
    --novel-scan-weight 1 --stall-pen 0.1 --revisit-pen 0.05 \
    --team-weight 0 --give-bonus 0 --recv-bonus 0 --overlap-pen 0 \
    --proximity-pen 0 --target-switch-pen 0 --target-yield-weight 0 --div-coef 0 \
    --path-bias-floor 1.5 \
    --eval-on-ckpt \
    --out /workspace/MARLauder/runs/dbg_m1 \
    "$@"
