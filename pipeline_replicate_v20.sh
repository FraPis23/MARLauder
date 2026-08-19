#!/bin/bash
# REPLICA DA ZERO dell'artefatto M=4 usato nel confronto con IR2:
#   runs/v20_syncM_20260816_000500/ckpt_best.pt  (iterazione 80)
#
# v20 NON e' un training indipendente: e' l'ultimo di quattro stadi incatenati, ognuno warm-started
# dal precedente. Questo script rimette in fila i quattro comandi esatti, ricostruiti dal campo
# `cmd` salvato in ciascun params.json (non riscritti a mano).
#
#   stadio 1  v16_easy        train/easy       M=2   da zero          244 it    5.2 h
#   stadio 2  v16_difficult   train/difficult  M=2   <- 1/final.pt    732 it   37.8 h
#   stadio 3  v19_m4          train/difficult  M=4   <- 2/final.pt    ~60 it   ~8 h   (vedi NOTA 1)
#   stadio 4  v20_syncM       train/difficult  M=4   <- 3/ckpt_best   103 it   17.7 h
#                                                            TOTALE   ~69 h
#
# NOTA 1 — lo stadio 3 e' ACCORCIATO di proposito, e non e' una scorciatoia. Il run originale fu
# lanciato con --total-steps 6000000 (732 it), girato fino a 274 e FERMATO A MANO; l'unico artefatto
# che lo stadio 4 consuma e' il suo `ckpt_best`, che sta all'iterazione 40. I tick di eval del run
# originale (it 10..60): .402 .432 .432 **.438** .433 .434 — il massimo e' it=40 sia fermandosi a 60
# sia andando fino a 274, quindi 60 iterazioni producono LO STESSO ckpt_best in 8 ore invece di 44.
# In piu' si resta fuori dalla regione in cui v19 collassa (it 90-140, success -> 0.000).
# Il --total-steps qui sotto e' quindi 500000 (~61 it), NON i 6000000 del comando salvato.
#
# NOTA 2 — QUESTA REPLICA NON E' BIT-EXACT, E NON PUO' ESSERLO. `torch.manual_seed(0)` e' impostato
# (driver.py:580) ma `cudnn.benchmark = True` (driver.py:587) sceglie gli algoritmi cronometrandoli,
# e non c'e' `use_deterministic_algorithms`. Due run identici divergono. Cosa aspettarsi:
# una policy STATISTICAMENTE equivalente, non gli stessi pesi.
#
# NOTA 3 — LA SELEZIONE VA RIFATTA, non ereditata. `ckpt_best` e' scelto dal massimo di UN SINGOLO
# tick di eval, e i tick vicini al picco di v20 stanno dentro il rumore: .433 (it60) .421 (it70)
# .439 (it80), spread .006 contro un rumore misurato di .015 su eval/score. Su un rerun il picco
# cadra' quasi certamente su un'altra iterazione dello stesso plateau. Dopo lo stadio 4:
#   python scripts/score_ckpts.py --ckpt <OUT4>/ckpt_0{40,60,80,100}.pt <OUT4>/ckpt_best.pt \
#       --splits train/difficult,test/complex --repeats 3 --n-agents 4 \
#       --max-travel-frac 0.040 --comm-relay
# e si tiene il migliore sul BLOCCO ESITI (coverage_auc, steps_to_90, success, own_cov), NON su
# eval/score: il composito e' dominato da contrib_imbalance_norm, che nel CSV di IR2 non esiste.
#
# NOTA 4 — 32 env a M=4 con 6 hop stanno in 15.47 GiB per un soffio. Gli stadi 3 e 4 vanno in OOM
# al primo backward senza expandable_segments. Non abbassare --n-envs e non alzare --minibatches:
# cambiano il batch e i numeri non sono piu' confrontabili con quelli in tesi.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
R=${R:-runs/repl_${TS}}
COMMON="--n-envs 32 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 --k-epochs 4 \
 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront --radar-team-source belief \
 --map-seed 0 --pf-frontier-min-unknown 1 --pf-frontier-min-util 1e-6 --done-mode own \
 --gamma 0.998 --rdv-weight 0.10 --rdv-clamp-pos --rdv-urgency-T 256 \
 --sync-weight 0.25 --sync-recv-ratio 0.5 --sync-min-gap 32 --trace-steps 512 --eval-on-ckpt"

echo "=== stadio 1/4 — v16_easy (da zero, M=2)"
python scripts/run_train.py --split train/easy --max-episode-steps 256 --max-travel-frac 0.040 \
  --total-steps 2000000 --n-agents 2 ${COMMON} \
  --eval-suite-splits test/hybrid --eval-split test/hybrid --eval-steps 256 \
  --out ${R}/s1_easy

echo "=== stadio 2/4 — v16_difficult (M=2)"
python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.040 \
  --total-steps 6000000 --n-agents 2 --rdv-urgency-mode budget --rdv-urgency-start 0.5 ${COMMON} \
  --eval-suite-splits test/complex --eval-split test/complex --eval-steps 768 \
  --init-ckpt ${R}/s1_easy/final.pt \
  --out ${R}/s2_difficult

echo "=== stadio 3/4 — v19_m4 (M=4, accorciato: serve solo ckpt_best@40 — vedi NOTA 1)"
python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.030 \
  --n-agents 4 --novel-scan-weight 1.65 --total-steps 500000 \
  --rdv-urgency-mode budget --rdv-urgency-start 0.5 ${COMMON} \
  --eval-suite-splits test/complex --eval-split test/complex --eval-steps 768 --eval-every 10 \
  --init-ckpt ${R}/s2_difficult/final.pt \
  --out ${R}/s3_m4

echo "=== stadio 4/4 — v20_syncM (M=4, sync scalato (2/M)^1)"
python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.030 \
  --n-agents 4 --novel-scan-weight 1.65 --total-steps 850000 \
  --rdv-urgency-mode budget --rdv-urgency-start 0.5 ${COMMON} \
  --sync-weight-m-scale 1.0 \
  --eval-suite-splits test/complex --eval-split test/complex --eval-steps 768 --eval-every 10 \
  --init-ckpt ${R}/s3_m4/ckpt_best.pt \
  --out ${R}/s4_v20

echo "REPLICA COMPLETA. Artefatto candidato: ${R}/s4_v20/ckpt_best.pt"
echo "ORA RIFAI LA SELEZIONE (NOTA 3) — non fidarti di ckpt_best preso da un tick singolo."

# RIFERIMENTO — cosa deve riprodurre, misurato su ckpt_best originale con
# score_ckpts.py --n-agents 4 --max-travel-frac 0.040 --comm-relay, 32 mappe, 3 ripetizioni:
#   test/complex     coverage_auc .8209   steps_to_90 282.5   success .979   own_cov .9953
#   train/difficult  coverage_auc .9015   steps_to_90 154.9   success .969   own_cov .9957
# Una replica riuscita cade DENTRO il rumore di queste (score +-.015, success +-12.6 punti a 2 sigma).
