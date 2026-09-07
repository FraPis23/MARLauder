#!/usr/bin/env bash
# MARLauder side of the comparison against IR2 — FROZEN PROTOCOL, native IR2 conditions.
#
# No distance budget: --max-travel-px stays 0 and IR2's own native step caps apply
# (hybrid 196, corridor 196, complex 384, from their test_parameter.py). The rule is to evaluate
# under exactly what IR2 was evaluated under; where that disadvantages us, it disadvantages us.
#
# KNOWN AND ACCEPTED COST. An IR2 step is a waypoint decision plus an A* traverse of arbitrary
# length; a MARLauder step is one lattice hop of at most 22.63 px. At equal step caps MARLauder
# spends ~45% of the metres IR2 spends on `complex` (7684 px against 16966), so almost every
# complex episode ends truncated at the cap rather than by success. Measured on the same
# checkpoint: at the native complex cap, complex_M2 scores success 0.01; with a 16966 px budget it
# scores 0.90. The comparison below is therefore CONSERVATIVE toward MARLauder, by choice.
#
# What is NOT a free parameter here — it comes from the protocol, eval/comparison/PROTOCOL.md:
#   - the 100 maps per split are fixed in map_indices_{split}.json, with .npy/PNG parity verified
#   - done_mode="own" is forced inside eval_comparison.py: IR2 terminates when EVERY robot holds
#     >=99% in its OWN belief, whereas MARLauder defaults to the team union (an easier question)
#   - `explored` is the MEAN OVER AGENTS of each agent's own belief, not the union (the union is
#     carried alongside, in the extra explored_union column)
#
# --comm-relay is redundant for the released checkpoint (it already carries comm_relay=True in its
# saved cfg) but stays explicit: it makes the command reproducible on any checkpoint and aligns the
# radio physics with IR2, which relays multi-hop.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

CKPT=${CKPT:?set CKPT=<path to the checkpoint to evaluate>}
TAG=${TAG:-v20_final}

python scripts/eval_comparison.py \
  --ckpt "${CKPT}" \
  --splits hybrid corridor complex \
  --agents 2 4 \
  --comm-relay \
  --tag "${TAG}"

python eval/comparison/analyze.py --tag "${TAG}"
