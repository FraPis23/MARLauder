#!/usr/bin/env bash
# Render a behavior GIF from the most recent checkpoint. Run anytime (even mid-sweep).
#   docker exec marlauder bash -lc 'cd /workspace/MARLauder && bash scripts/gif.sh'
# Output: runs/latest.gif  (volume-mounted → open it on the host).
#
# Optional args:  bash scripts/gif.sh <map_idx> <steps>
set -e
cd "$(dirname "$0")/.."

MAP_IDX="${1:-120}"     # which fixed map (try 1543, 2877, 4012, 5530, 7211, 8650, 9904 too)
STEPS="${2:-256}"

# Newest checkpoint under runs/ (any trial). Half-written files are skipped by torch.load.
CKPT="$(find runs -name '*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
if [ -z "$CKPT" ]; then echo "no checkpoint found under runs/"; exit 1; fi
echo "[gif] checkpoint: $CKPT  map=$MAP_IDX  steps=$STEPS"

python scripts/run_eval.py \
  --ckpt "$CKPT" \
  --split train/easy \
  --map-idx "$MAP_IDX" \
  --n-agents 2 \
  --steps "$STEPS" \
  --force-full-pos-sharing \
  --out runs/latest.gif

echo "[gif] wrote runs/latest.gif  (open on host: MARLauder/runs/latest.gif)"
