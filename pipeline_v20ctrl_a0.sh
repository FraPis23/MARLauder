#!/bin/bash
# v2.0 CONTROLLO NULLO — identico a v20 tranne `--sync-weight-m-scale 0.0`.
#
# PERCHE' ESISTE. v20 (runs/v20_syncM_20260816_000500) ha vinto su test/complex contro il suo
# controllo v19 (score 0.4445+-0.0104 vs 0.4281+-0.0157, steps_to_90 278.6+-1.6 vs 300.8+-25.5,
# -7.4%) e soprattutto ha ATTRAVERSATO INDENNE la finestra in cui v19 era crollato: v19 a it=90
# (cioe' 50 iterazioni dopo il punto di warm start) va a success_rate 0.000 per ~50 iterazioni e
# non recupera mai del tutto; v20, ripartito esattamente da quel ckpt, in quella stessa finestra
# ha cum 90/110 positivi e nessun crollo.
#
# MA v20 HA CAMBIATO DUE COSE INSIEME:
#   (1) la reward     — il bonus di sync scalato (2/M)^1, cioe' dimezzato a M=4;
#   (2) l'ottimizzatore — riparte da ckpt_best@40 con Adam AZZERATO (init_ckpt carica i pesi, non
#       lo stato dell'ottimizzatore), momenti e second-moment persi, LR di nuovo pieno.
# Ognuna delle due basta a spiegare "non e' crollato". Con un solo braccio la causa non e'
# attribuibile: e' esattamente il tipo di claim che questa repo ha gia' dovuto ritrattare
# (relay "costa 25% di success" -> il controllo nullo da solo dava -11%).
#
# COSA ISOLA QUESTO RUN. Stesso ckpt di partenza, stesso restart di Adam, stesso seed, stesse
# mappe, stesso numero di iterazioni, stessa suite di eval (test/complex, la sola che v20 aveva —
# `--eval-suite-splits` SOSTITUISCE la suite dello split di training, non la aggiunge, quindi
# aggiungere train/difficult qui romperebbe il pareggiamento con v20). L'unica differenza e'
# a = 0.0, che riporta il peso del sync a quello non scalato di v19.
#
# LETTURA DEL RISULTATO, decisa PRIMA di guardare:
#   - il controllo CROLLA nella finestra it 40-60 (eval/success_rate -> ~0, eval/score sotto il
#     punto di partenza)      -> e' la reward. La normalizzazione del sync previene il crollo.
#                                Claim difendibile in tesi.
#   - il controllo NON crolla -> e' il restart dell'ottimizzatore (o comunque non e' la reward).
#                                v20 resta migliore in misura (-7.4% steps_to_90 sta in piedi da
#                                solo, e' misurato con 3 ripetizioni), ma il meccanismo va
#                                riscritto: "riavviare Adam da un buon ckpt evita il crollo".
#   In entrambi i casi il numero di episodi e' n=1 per braccio: il crollo di v19 e' UN evento.
#   Una singola non-replica non lo smentisce, lo indebolisce.
#
# ATTENZIONE ALLA MEMORIA GPU: 32 env a M=4 con 6 hop stanno in 15.47 GiB per un soffio e il primo
# backward di v20 e' andato in OOM per frammentazione (1.61 GiB riservati e non allocati). Serve
# expandable_segments. NON abbassare --n-envs e NON alzare --minibatches: cambierebbero il batch e
# romperebbero il pareggiamento con v20, che e' l'unico motivo per cui questo run esiste.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
INIT_CKPT=${INIT_CKPT:-runs/v19_m4_20260811_122528/ckpt_best.pt}
OUT=${OUT:-runs/v20ctrl_a0_${TS}}

if [ ! -f "${INIT_CKPT}" ]; then
  echo "ERRORE: checkpoint di warm start non trovato: ${INIT_CKPT}"; exit 1
fi

# Riga per riga identica al cmd salvato in runs/v20_syncM_20260816_000500/params.json,
# tranne --sync-weight-m-scale (1.0 -> 0.0). 0.0 e' anche il default della dataclass; e'
# esplicito qui perche' il valore E' l'esperimento.
python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.030 \
  --n-agents 4 \
  --novel-scan-weight 1.65 \
  --total-steps 850000 \
  --rdv-urgency-mode budget --rdv-urgency-start 0.5 \
  --n-envs 32 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 \
  --k-epochs 4 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront \
  --radar-team-source belief --map-seed 0 \
  --pf-frontier-min-unknown 1 --pf-frontier-min-util 1e-6 \
  --done-mode own \
  --gamma 0.998 \
  --rdv-weight 0.10 --rdv-clamp-pos --rdv-urgency-T 256 \
  --sync-weight 0.25 --sync-recv-ratio 0.5 --sync-min-gap 32 \
  --sync-weight-m-scale 0.0 \
  --trace-steps 512 \
  --eval-suite-splits test/complex \
  --eval-on-ckpt --eval-split test/complex --eval-steps 768 --eval-every 10 \
  --init-ckpt ${INIT_CKPT} \
  --out ${OUT}
echo "V20CTRL_DONE final=${OUT}/final.pt best=${OUT}/ckpt_best.pt"

# DA GUARDARE, contro runs/v20_syncM_20260816_000500 (stesso init, stesso restart, stesso tutto).
#   eval/success_rate nelle it 40-60  -> LA misura. v19 andava a 0.000 li'; v20 no.
#   eval/score                        -> v20 picco 0.4386 a it=80. Il controllo lo raggiunge?
#   reward/sync (share)               -> v20 e' atterrato al 5.2% del budget (target: il 5.5% di
#                                        M=2). Qui deve tornare ~9-10%, come v19. Se non lo fa,
#                                        il flag non e' arrivato all'env: verificare params.json.
#   metric/own_cov_gap, eval/own_coverage_final -> il rischio noto dello scalare il sync sotto
#                                        done_mode=own e' pagare in condivisione mappe. v20 non ha
#                                        pagato (own_cov 0.9958). Se il controllo fa MEGLIO qui,
#                                        il -7.4% di steps_to_90 ha un prezzo che v20 nascondeva.
# Confronto finale con la STESSA armatura, mai a occhio sulle metriche di training:
#   python scripts/score_ckpts.py --ckpt runs/v20_syncM_20260816_000500/ckpt_best.pt \
#       ${OUT}/ckpt_best.pt --splits train/difficult,test/complex --repeats 3
