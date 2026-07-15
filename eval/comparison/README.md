# Comparison MARLauder vs IR2 — protocollo e stato

Documento AUTOSUFFICIENTE: da qui si riprende il confronto senza altro contesto.
Stato al 2026-07-15: **lato IR2 COMPLETO (baseline definitiva sotto). Lato MARLauder DA FARE
quando l'utente sceglie il checkpoint** (in training: pipeline_v09 sul PC potente).

## Protocollo (deciso e congelato)

- **Mappe**: 100 per test set, FISSATE per nome file in `map_indices_{complex,corridor,hybrid}.json`
  (in questa dir; `pack_idx` = indice nel pack .npy MARLauder, `file` = PNG IR2). Stesse mappe
  per entrambi i sistemi. Parity dataset VERIFICATA bit-a-bit (`parity_check.py`, gate passato).
- **Agent count**: M=2 e M=4 (MARLauder M=4 = zero-shot, critic count-invariant; se degenera,
  eventuale finetune M=4 va tenuto come run separata etichettata, mai sostituito allo zero-shot).
- **Cap episodio**: nativi IR2 — 196 step (hybrid, corridor), 384 (complex). ATTENZIONE: gli
  "step" NON sono confrontabili tra i sistemi (IR2 = decisione waypoint + tragitto A*;
  MARLauder = 1 hop di lattice ≤16·√2 px). Il confronto temporale si fa sulla DISTANZA.
- **Metriche = quelle NATIVE IR2** (scelta utente), CSV con colonne
  `eps,num_robots,max_dist,steps,explored,success,connectivity`:
  - `max_dist` (headline): max tra i robot della distanza percorsa cumulata (px) a fine episodio.
  - `explored`: explored rate a fine episodio (unione team, come IR2 evaluate_team_exploration_rate).
  - `success`: criterio IR2 — OGNI robot ha ≥99% della mappa nella PROPRIA belief
    (IR2 env.py:600-603). NON è l'unione: la condivisione è parte del task.
  - `connectivity`: booleano di fine episodio — tutti i robot in un unico "flock" connesso nel
    grafo comm (link segnale>soglia, transitività multi-hop; fuori dal flock massimo = broken;
    due flock massimi a pari merito = tutti broken). IR2 env.py:205-221, 386.
  - MARLauder può aggiungere la colonna extra `explored_union` (appendice), senza sostituire nulla.
- **Statistica**: mean±std per set (stile paper IR2) + Wilcoxon signed-rank APPAIATO per mappa
  su max_dist/explored (stessi indici mappa). MAI medie tra set.

## Baseline IR2 (DONE 2026-07-14 — pretrained model/stage2, 600 episodi, 0 skip A*)

| cella | max_dist | steps | explored | success | connectivity |
|---|---|---|---|---|---|
| hybrid_M2 | 3422±928 | 88.7 | 0.997 | 1.00 | 0.89 |
| hybrid_M4 | 2413±467 | 48.4 | 0.999 | 1.00 | 0.60 |
| corridor_M2 | 7204±2297 | 141.9 | 0.976 | 0.76 | 0.66 |
| corridor_M4 | 5352±2276 | 92.5 | 0.992 | 0.93 | 0.56 |
| complex_M2 | 16966±4532 | 277.8 | 0.975 | 0.72 | 0.59 |
| complex_M4 | 13403±5768 | 203.5 | 0.982 | 0.80 | 0.27 |

CSV grezzi: `../../../IR2-Multi-Robot-RL-Exploration/comparison/results/ir2_{split}_M{M}.csv`.

## Com'è stato prodotto il lato IR2 (repro)

Tutto in `IR2-.../comparison/` — file originali IR2 INTATTI:
- `{split}_M{2,4}/test_parameter.py`: varianti config (solo NUM_ROBOTS/set/output cambiati) +
  COPIA di test_driver.py (NON symlink: sys.path[0] risolve i symlink e vincerebbe il
  test_parameter nativo del repo).
- `maps_{split}/`: le 100 mappe come symlink RELATIVI ai PNG (assoluti host si rompono nel
  container). `Env(map_index=episodio)` + `map_list` di soli 100 file → ogni mappa 1 volta.
- `compat/`: sensor.py (itemset) ed env.py (np.lib.pad) patchati per numpy 2.0, shadow via
  `PYTHONPATH=compat:repo`.
- Deps container (ephemeral! reinstallare se il container viene ricreato):
  `pip install -r comparison/requirements_extra.txt` (scikit-image, scikit-learn, matplotlib, ray, pandas).
- Run: `docker exec marlauder bash /workspace/IR2-Multi-Robot-RL-Exploration/comparison/run_ir2_comparison.sh`
  (~4h, NUM_META_AGENT=4). Log: `comparison/run_full.log`.

## Lato MARLauder — DA FARE quando l'utente indica il checkpoint

Costruire `MARLauder/scripts/eval_comparison.py`:
1. Carica il ckpt con `eval/ckpt_loader.load_model_from_ckpt` (auto-arch: n_layers/gru/gat dal ckpt).
2. Per ogni split e M∈{2,4}: env sui 100 `pack_idx` di `map_indices_{split}.json`
   (`Explorer.reload_map(env_idx, map_idx)` resetta TUTTO lo stato), `max_episode_steps` =
   cap nativo IR2 (196/384), rollout deterministico.
3. Per episodio, emette la riga CSV formato IR2:
   - `max_dist`: accumula ‖Δpos‖ per agente a ogni step (env.pos, px), max finale sugli agenti.
   - `explored`: unione (già in env: explored_rate).
   - `success`: PER-ROBOT — per ogni agente `(occupancy_torch[n,a]==FREE).sum() / free_total ≥ 0.99`
     per tutti gli a (occupancy per-agente già esistente [N,M,H,W]).
   - `connectivity`: flock unico sul comm_mask dell'ULTIMO step (M=2: comm_mask[i,j] diretto;
     M=4: componenti connesse transitive, tutti nello stesso flock massimo).
   - extra: `explored_union` (= explored, ridondante ma esplicito).
4. Output: `eval/comparison/results/marlauder_{split}_M{M}.csv` (creare `results/`).
5. Aggregatore `analyze.py`: tabella per set/M dai CSV dei due sistemi + Wilcoxon appaiato
   (per mappa, via ordine indici) su max_dist/explored.

Nota M=4: `load_model_from_ckpt(..., n_agents=4)` sovrascrive M (pesi identici, architettura
count-invariant); l'env va costruito con n_agents=4.

## Contesto storico

Baseline interna MARLauder (suite diversa, NON confrontabile con questa): eval_best su
test/complex @512. Diario completo: `MARLauder/dev_log.md` (sessioni 2026-07-13/14/15).
