#!/bin/bash
# v1.7 — PREZZO DEL CONTATTO INUTILE. Una sola modifica rispetto alla fase 2 di v16.
#
# IL FENOMENO, misurato sui ckpt v16 (test/complex, 16 env, policy deterministica), prima e dopo il
# PRIMO sync pagato:
#     fase 2 (sync .25, frac .040): dist 472 -> 342 px   overlap 0.089 -> 0.247   comm 0.030 -> 0.120
#     fase 3 (sync .50, frac .050): dist 442 -> 288 px   overlap 0.106 -> 0.378   comm 0.032 -> 0.201
# Dopo il primo incontro si avvicinano e la sovrapposizione dei sensori TRIPLICA. La fase 3 peggiora
# tutto del ~70%: quelle due modifiche (sync 0.5, budget 0.050) vanno considerate SBAGLIATE e il
# checkpoint di riferimento resta quello di FASE 2 (own_cov_final 0.974 vs 0.939, succ 0.698 vs 0.667).
#
# IL MECCANISMO. Un anello che si auto-rinforza: il contatto FONDE le mappe -> mappe identiche
# producono campi di utilita' identici -> campi identici scelgono la stessa frontiera -> co-locazione
# -> altro contatto. Niente nel reward prezzava il RESTARE attaccati: novel semplicemente non paga chi
# segue, e uno zero non e' una penalita'.
#
# COSA NON E' IL PROBLEMA — tutte ipotesi TESTATE E SCARTATE con misura, non ripercorrerle:
#   * novel indebolito: NO. Peso 1.0 invariato, 0.0264/step (v15: 0.0246), 31.5% del budget di fase 2
#     — il termine piu' grande, sopra completion (26.7%).
#   * doppio credito a chi co-scansiona: NO. Rapporto credito/crescita-unione = 1.0076, scoperta
#     simultanea 0.8%. Chi segue prende davvero 0.
#   * belief piatta / poco informativa: NO. Entropia 1.12 -> 2.70 e nodi efficaci 5.4 -> 31.4 (su 3844)
#     al crescere del tempo dall'ultimo contatto, p_max 0.66 -> 0.37. Alzare la nitidezza
#     (temperatura < 1) la collasserebbe su un nodo solo, distruggendo la rappresentazione
#     dell'incertezza.
#   * gate g morto: NO. Durante la SEPARAZIONE sale a 0.99 e offer_frac arriva a 0.41 (41% di mappa
#     esclusiva). Sembrava morto solo perche' era stato misurato sui passi IN CONTATTO, dove `offer`
#     e' zero per costruzione: viene calcolato in _refresh_obs, DOPO che la fusione ha resettato
#     _own_expl_at_comm.
#   * geometria radio (LOS al posto di signal_strength): SCARTATA per correttezza del confronto. Il
#     nostro default e' gia' il port esatto di IR2 (parameter.py: SS_P_T -20, SS_THRESH -70,
#     SS_GAMMA 2/4, SS_DIST_O 35, SS_PL_O 31, SS_XG/SS_K ~ U[0,13], SENSOR_RANGE 80,
#     USE_SIGNAL_STRENGTH_NOT_PROXIMITY=True). La banda del tether (radio 150-310 px contro dischi
#     LiDAR che si separano a 160) e' la geometria DI IR2, non una nostra svista: i loro agenti hanno
#     lo stesso spazio per sfruttarla. La fisica non si tocca.
#
# LA MODIFICA — solo reward di training, che e' l'unica cosa legittimamente nostra:
#     --comm-idle-pen 0.03
#     pen = coef * in_contact * (1 - sync_paid) * (1 - rampa_budget)
#   * GRATIS sul passo in cui lo scambio viene davvero PAGATO -> "tocca, scambia, riparti".
#   * GRATIS verso la scadenza (stessa rampa di --rdv-urgency-mode budget, da 0.5) -> il rendezvous
#     finale, che done_mode=own RICHIEDE, non e' mai tassato.
#   * COSTA su tutto il resto: contatto prolungato e ri-contatto dentro il min_gap.
# Perche' sync_paid e NON il gate g: g e' ~0 su OGNI passo in contatto per costruzione (misurato:
# media 0.026 sia sui fronti di salita sia sul contatto prolungato), quindi punirebbe l'incontro
# legittimo esattamente quanto il tether. sync_paid e' il fronte di salita pre-fusione che ha
# consegnato mappa davvero.
# Taratura: l'81.8% dei passi in contatto e' prolungato (raffiche da 3.1 passi in media). A coef 0.03
# risultano tassati il 95% dei passi in contatto, penalita' media 0.00188/step = 5.9% di novel in
# aggregato — ma al momento della decisione restare costa 0.03 E rinuncia a ~0.026 di novel.
#
# WARM START dalla fase 1 di v16, la STESSA da cui e' partita la fase 2 di v16: l'unica differenza fra
# questo run e quello e' --comm-idle-pen. Confronto pulito, una variabile sola.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
EASY_CKPT=${EASY_CKPT:-runs/v16_easy_20260803_135950/final.pt}
OUT=${OUT:-runs/v17_commidle_${TS}}

if [ ! -f "${EASY_CKPT}" ]; then
  echo "ERRORE: checkpoint di warm start non trovato: ${EASY_CKPT}"; exit 1
fi

python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.040 \
  --total-steps 6000000 \
  --comm-idle-pen 0.03 \
  --rdv-urgency-mode budget --rdv-urgency-start 0.5 \
  --n-envs 32 --n-agents 2 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 \
  --k-epochs 4 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront \
  --radar-team-source belief --map-seed 0 \
  --pf-frontier-min-unknown 1 --pf-frontier-min-util 1e-6 \
  --done-mode own \
  --gamma 0.998 \
  --rdv-weight 0.10 --rdv-clamp-pos --rdv-urgency-T 256 \
  --sync-weight 0.25 --sync-recv-ratio 0.5 --sync-min-gap 32 \
  --trace-steps 512 \
  --eval-suite-splits test/complex \
  --eval-on-ckpt --eval-split test/complex --eval-steps 768 \
  --init-ckpt ${EASY_CKPT} \
  --out ${OUT}
echo "V17_DONE final=${OUT}/final.pt best=${OUT}/ckpt_best.pt"

# DA GUARDARE, contro runs/v16_difficult_20260803_135950 (fase 2):
#   metric/sensing_overlap   v16 0.189  -> DEVE SCENDERE. E' la ridondanza, l'obiettivo del run.
#   metric/comm_duty_cycle   v16 0.063  -> deve scendere.
#   eval/n_syncs             v16 4.66   -> IL RISCHIO. Se crolla sotto ~1.5 gli agenti stanno evitando
#                                          il contatto per non pagare: abbassare coef a 0.015, non
#                                          alzare altri termini.
#   eval/own_coverage_final  v16 0.974  -> non deve peggiorare. Sotto 0.90 = penalty troppo forte.
#   reward/comm_idle                    -> nuovo. Se e' ~0 il termine non morde (contatto gia' raro);
#                                          se supera |reward/sync| domina lo scambio.
#   reward/completion        v16 0.0223 -> se va a zero e' il fallimento di v12 (obiettivo irraggiungibile).
