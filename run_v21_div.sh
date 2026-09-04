#!/bin/bash
# v21_div — stadio 4 di v20, IDENTICO, con la sola aggiunta di --div-weight 1.0.
#
# E' un A/B pulito contro runs/v20_syncM_20260816_000500: stesso init-ckpt (v19_m4/ckpt_best),
# stessi 850000 step, stessa COMMON, stesso eval. L'unica differenza e' la loss ausiliaria di
# diversita' a livello di frontiera (docs/contrib_imbalance_diagnosi_2026-08-20.md sezione 8).
#
# PESO: 1.0 misurato, non tirato a indovinare — allo smoke div_loss ~0.004-0.006 contro pg_loss
# ~0.002-0.019, cioe' i due termini sono comparabili.
#
# comm_relay: v19 e' pre-flag (assente nel suo params.json) ma train_args ha
# set_defaults(comm_relay=True), quindi questo run parte con relay ON esattamente come v20.
#
# MEMORIA: 15562 MiB / 16303 misurati a questa config con --div-weight attivo. Non alzare --n-envs
# e non abbassare --minibatches: cambiano il batch e i numeri non sono piu' confrontabili con v20.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
OUT=runs/v21_div_20260821_104706
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
  --init-ckpt runs/v19_m4_20260811_122528/ckpt_best.pt \
  --out ${OUT}

echo "V21 DONE -> ${OUT}"
