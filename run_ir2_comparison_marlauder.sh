#!/bin/bash
# Lato MARLauder del confronto con IR2 — PROTOCOLLO CONGELATO, condizioni IR2 native.
#
# Decisione utente 2026-08-19: "Usa esattamente cio' che e' stato usato per IR2, se siamo
# svantaggiati e' meglio. I cap servivano per il train, nel test dobbiamo adattarci a loro."
# Quindi NESSUN budget di distanza: --max-travel-px resta 0 e valgono i cap in step nativi di
# IR2 (IR2_CAPS = hybrid 196, corridor 196, complex 384, dai loro test_parameter.py).
#
# COSTO NOTO E ACCETTATO. Uno step IR2 e' una decisione di waypoint piu' il tragitto A* di
# lunghezza arbitraria; uno step MARLauder e' un hop di lattice <= 22.63 px. A parita' di cap in
# step MARLauder spende ~45% dei metri che spende IR2 su complex (7684 px contro 16966), quindi
# quasi ogni episodio complex finisce troncato al muro e non per successo. Misurato sullo stesso
# checkpoint (run s0, luglio): a cap nativo complex_M2 fa success 0.01, con budget 16966 px fa
# 0.90. Il confronto qui sotto e' percio' CONSERVATIVO verso MARLauder, per scelta.
#
# Cosa NON e' un parametro libero (viene dal protocollo, eval/comparison/PROTOCOL.md):
#   - le 100 mappe per split sono fissate in map_indices_{split}.json, parity .npy/PNG verificata
#   - done_mode="own" e' forzato dentro eval_comparison.py: IR2 termina quando OGNI robot ha
#     >=99% nella PROPRIA belief, MARLauder di default sull'unione (domanda piu' facile)
#   - `explored` e' la MEDIA sugli agenti della belief propria, non l'unione (l'unione va solo
#     nella colonna extra explored_union)
#
# --comm-relay e' ridondante per questo checkpoint (v20 ha gia' comm_relay=True nel suo cfg) ma
# resta esplicito: rende il comando riproducibile su qualunque ckpt e allinea la fisica radio a
# IR2, che rilancia multi-hop (env.py:424).
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
CKPT=${CKPT:-runs/v20_syncM_20260816_000500/ckpt_best.pt}   # v20, iterazione 80 — artefatto scelto
TAG=${TAG:-v20_final}

python scripts/eval_comparison.py \
  --ckpt "${CKPT}" \
  --splits hybrid corridor complex \
  --agents 2 4 \
  --comm-relay \
  --tag "${TAG}"

python eval/comparison/analyze.py --tag "${TAG}"
