# Come dare contesto globale all'agente senza il target analitico
### Sintesi della discussione — 2026-06-25

Documento per allineamento col team. Riassume il ragionamento dal momento in cui
abbiamo iniziato a parlare di **comprimere le info del grafo sul bordo della finestra
locale**, fino alla proposta corrente (campo di valore propagato sull'albero BF).
Obiettivo di fondo: rendere il progetto **più MARL puro**, riducendo la dipendenza dal
target globale analitico, **rispettando il vincolo di env GPU-only a tensori fissi**.

---

## 0. Contesto / punto di partenza

L'agente MARLauder osserva solo una **finestra ego-centrica locale** (`n_hops=2` →
finestra `(2·2+3)² = 49` nodi). Tutto ciò che sta **oltre la finestra è invisibile**.

L'unico canale che inietta "dove andare" da fuori finestra è il **target globale
analitico** + il **guidepost** (feature di nodo n.5 = path BF dal target committato).

**Diagnosi condivisa:** il target analitico è una *stampella*. L'RL si riduce a
**seguire il guidepost** (path-following locale); la decisione difficile — quale
frontiera nel mondo puntare — la risolve un'**euristica**, non la policy. È una
decomposizione gerarchica valida (planner classico + controllo appreso) ma **non è
MARL puro**: l'agente è dipendente dalla guida e non impara il ragionamento globale.

---

## 1. Idea iniziale: comprimere le info sul bordo della finestra

**Proposta:** mentre l'agente esplora, comprimere progressivamente i nodi che si lascia
alle spalle. Sulla frontiera info ad alta risoluzione, poco dietro meno, più dietro
ancora meno. Quando un nodo rientra nella finestra locale, ripristinarne le info vere.
Variante: **due grafi sovrapposti** — uno processato (compresso), uno intatto.

### Punto debole fatale: lo "sbroglio al ritorno"
Una compressione **latente lossy** sbatte contro un trade-off **fondamentale**:
*compressione lossy ↔ ricostruibilità*. Al ritorno su una zona compressa:
- se **lossy** → hai perso dettaglio → ri-esplori (spreco) o l'agente cicla;
- se **lossless** → non è vera compressione.

### Perché NON serve farla (osservazione chiave)
Nel nostro env il problema **non si pone**, perché:
- Il **lattice globale completo è già un tensore fisso e persistente**
  (`node_feat [N, N_max, 6]`), **mai distrutto**.
- Quindi *niente viene compresso in modo irreversibile*. Tornare indietro = ri-ritagliare
  la finestra alla nuova posizione: i dati erano sempre lì.
- I "**due grafi sovrapposti**" che immaginavamo **esistono già**: il lattice intatto
  (persistente) + la vista processata (il crop-finestra dell'encoder). Il "ripristino"
  è automatico, non serve logica di restore.

**Conclusione:** istinto giusto (fovea locale + periferia riassunta), **macchinario
sbagliato**. La compressione latente per-nodo con restore è lavoro inutile e fragile, e
risolve un non-problema. → **Scartata.**

---

## 2. Confronto con IR2 (come fa chi NON usa target analitico)

IR2 (linea ARIADNE, IROS 2024) **non ha target analitico**. Usa:
- **Grafo globale sparso** come osservazione: tutti i nodi del grafo esplorato.
- **Encoder ad attenzione a 6 layer sull'INTERO grafo** → l'info delle frontiere lontane
  si propaga fino al nodo corrente (`model.py: encode_graph`).
- **Decoder**: il nodo corrente fa cross-attention su tutta la memoria-grafo → contesto
  globale.
- **Pointer** sui vicini: l'azione è un vicino, ma gli embedding contengono già il
  contesto globale. Scelta **appresa**, non euristica.
- Grafo tenuto compatto con **sparsificazione strutturata**: pota i nodi a utility~0,
  tiene i **centri-frontiera** e i path A* di collegamento (`parameter.py`).

**Lezione:** la stampella analitica di MARLauder esiste *perché* abbiamo scelto la
finestra ego-locale. IR2 risolve a monte con la **rappresentazione = grafo globale
sparso**. La loro sparsificazione strutturata è la "compressione giusta": compatta ma
**topologicamente lossless** → il ritorno è gratis (il nodo è ancora lì).

### Perché NON possiamo copiare IR2 pari pari
**Vincolo del progetto: env GPU-only, vettorizzato su N env paralleli a tensori fissi.**
Il grafo sparso dinamico di IR2 (numero di nodi variabile, sparsificazione A*,
graph-merger) è **incompatibile**: rompe il batching a dimensione fissa. → Serve una
soluzione **fixed-size, interamente tensorizzata, batchabile su GPU**.

---

## 3. Asset già presenti nel progetto (a dimensione fissa)

Riscoperta importante: **l'info globale ce l'abbiamo già in forma fissa.** Due strutture:

1. **Lattice globale completo** — `node_feat [N, N_max, 6]`. Persistente, fisso.
   Le 6 feature di nodo: `0 x_rel, 1 y_rel, 2 utility(diffusa), 3 age, 4 teammate_pot,
   5 guidepost`.

2. **Top-K candidati frontiera** — `cand_feat [N, M, K=16, 9]`
   (`extract_topk_candidates`). I 16 nodi a **utility più alta** tra quelli
   **raggiungibili** (flood-fill FREE), su scala **globale**, dimensione **fissa**.
   Le 9 feature per candidato:
   `0-1 rel_x/y, 2 utility, 3 dist_BF, 4 min_team_dist, 5 max_comm_gap,
    6 own_minus_team (ownership/yield), 7 team_alt_score, 8 prev_branch_match`.
   Le feature 4-6 sono **già** consapevolezza per-frontiera del teammate.

### Il "doppio strato" del progetto (policy gerarchica)
- **Strato strategico** = quale **frontiera** globale puntare → top-K candidati +
  `StrategicHead` (attenzione sui K candidati). **OGGI DISABILITATO**: con
  `analytic_target=True` il target lo sceglie l'env (`select_target_analytic`) e la
  StrategicHead è bypassata. I `cand_feat` si calcolano ma **non alimentano nulla** →
  infrastruttura dormiente.
- **Strato tattico** = quale **vicino** (8) fare come prossimo passo → finestra GAT +
  `PointerHead`. Sempre attivo.

**Implicazione:** lo strato globale fixed-size **esiste già** (top-K + 9 feature incl.
teammate). Non serve inventare una rappresentazione nuova.

---

## 4. Opzioni emerse (tutte fixed-size / GPU-compatibili)

| # | Opzione | Cosa fa | Pro | Contro |
|---|---------|---------|-----|--------|
| 1 | **Riattivare la StrategicHead** sui top-K | La policy SCEGLIE la frontiera (target appreso) al posto dell'euristica | MARL puro; infrastruttura già pronta; toglie la stampella | Retrain; capire **perché** fu disabilitata (rischio thrashing/coverage) |
| 2 | **Canale coarse-global foveato** | Lattice poolato su griglia fissa (es. 16×16) come "periferia" + finestra full-res | È la "compressione graduata" fatta **statica**; niente restore; piccolo cambio | Aggiunge un canale; risoluzione periferica grossolana |
| 3 | **Campo di valore sull'albero BF** (proposta corrente) | Propaga la utility delle frontiere lungo i cammini minimi → campo direzionale nella finestra | Vedi §5 | Vedi §5 |

Le tre **non si escludono**.

---

## 5. Proposta corrente: campo di valore propagato sull'albero BF

**Idea:** per ogni frontiera, propagare la sua utility lungo il **cammino minimo BF
dall'agente**, così ogni nodo porta "**quanto valore-frontiera è raggiungibile passando
di qui**". Stesso schema dal lato del **teammate** (posizione nota). È un *navigation
value field* (parente di Value-Iteration-Networks, ma calcolato analiticamente).

### Perché NON è una forzatura peggiore — è più morbida
- **guidepost (feat5)** odierno: BF da **UN solo** target (la scelta analitica) → mostra
  il path di **una** frontiera committata. Steer forte + commitment rigido = **questa è
  la vera stampella**.
- **utility diffusa (feat2)**: isotropa, si spande uguale in tutte le direzioni → non sa
  quale **direzione** porta al valore.
- **Campo di valore proposto**: **multi-frontiera**, pesato per utility, lungo i **path
  veri** → mostra il valore in tutte le direzioni, l'agente **sceglie**. Niente
  commitment a un target singolo.

→ Rimuove la parte forzante (il target imposto), tiene solo un **campo direzionale di
valore** che l'agente legge nella finestra. **Più MARL, non meno**: resta un prior
geometrico calcolato, ma è un'**osservazione**, non un vincolo d'azione — la policy
impara a usarlo. Generalizza il guidepost (1→K target) ed è strettamente più informativo
della diffusione isotropa (direzionale, wall-aware, segue i corridoi).

### Costo computazionale: minimo, già parallelo
Il pezzo costoso (la BF) **lo calcoliamo già**:
- `bf_dist_from_curr` + `bf_parent_from_curr` = **albero dei cammini minimi dall'agente**
  (UNA BF single-source, non una per frontiera).
- `bf_dist_team` = BF dalle posizioni note dei teammate. Già pronto.

La propagazione = **somma sui sotto-alberi** lungo i parent pointer:
```
val[v] = utility[v]                 # le frontiere sono le "foglie ricche"
ripeti depth volte:
    scatter_add  val[v] → val[parent[v]]     # il valore fluisce verso curr
```
Ogni nodo accumula la utility delle frontiere nel suo sotto-albero. È **una
accumulazione iterata sull'albero**, stessa classe di costo della **diffusione che già
facciamo**: `O(depth · N_max)` per env, **scatter_add vettoriale → pienamente
GPU-parallelo, dimensione fissa**. Trascurabile. Idem per il teammate (secondo albero,
già calcolato).

### Raffinamenti di design
1. **Sconto per distanza**: pesa la foglia con `utility/(1+β·dist)` o decadi durante la
   risalita → valore **scontato** (frontiere lontane pesano meno). Coerente con
   `explore` analitico.
2. **(Nuova idea, da includere) Canale distanza-nodo→frontiera**: oltre al valore
   accumulato, esporre per ogni nodo la **distanza alla frontiera** lungo il path. Rende
   il campo **più interpretabile** e la scelta più **consapevole** (l'agente distingue
   "molto valore ma lontano" da "poco valore ma vicino"), invece di avere solo un valore
   aggregato.
3. **Somma-sottoalbero vs marca-path**: la somma-sottoalbero è migliore — dà al
   **punto di biforcazione** (il curr) il **contrasto tra i rami**: quale vicino porta a
   più valore. È esattamente ciò che serve al pointer tattico.
4. **Lato teammate**: stesso campo sul suo albero → "quanto valore è meglio coperto da
   lui" = **divisione del lavoro** come campo; oppure marca il path verso di lui per il
   **rendezvous**. Dà attrazione/repulsione **spaziale**, non solo prossimità (feat4).

### Canali nuovi proposti (in `node_feat`)
- `reachable_value_self` — valore-frontiera scontato raggiungibile attraverso il nodo
  (albero dell'agente).
- `dist_to_frontier_self` — distanza al miglior valore lungo il path (interpretabilità).
- `reachable_value_team` — analogo sull'albero del teammate (divisione del lavoro).

### Cautele oneste
- Resta un **prior geometrico calcolato** (l'agente non deriva la BF da sé). Accettabile
  come **osservazione**; la forzatura era il **target rigido**, che questo **toglie**.
- **Sovrappone feat2 (diffusa)**: questo è la versione **direzionale e migliore** → si
  può **rimpiazzare** la diffusione isotropa, non sommarla.
- Vicino al curr il campo è grande (tutto è "oltre"); il segnale utile è il **contrasto
  tra rami fratelli**, non il valore assoluto → eventualmente normalizzare per-vicino.
- Toglie la stampella → durante il **riapprendimento la coverage calerà**; il valore vero
  (ragionamento globale emergente) si misura solo a fine retrain. È una scelta di
  ricerca, non un bugfix.

---

## 6. Raccomandazione

Dentro il vincolo GPU-fixed, il **campo di valore sull'albero BF (Opzione 3)** è
probabilmente il modo migliore di passare l'info globale alla finestra **senza target
analitico e senza grafo dinamico**:
- è una **feature** → nessun cambio all'architettura dell'encoder;
- **toglie il commitment rigido a un target** (la vera stampella);
- **costa pochissimo** e riusa BF + pattern di propagazione già presenti;
- **fixed-size, parallelo**;
- **soft** → avvicina al MARL puro mantenendo il prior geometrico utile.

Possibile percorso a passi:
1. Aggiungere i canali `reachable_value_self` (+ `dist_to_frontier_self`) e validarli su
   hybrid/easy (a parità di tutto) per vedere se sostituiscono il guidepost senza
   crollo.
2. Aggiungere il lato teammate (`reachable_value_team`) per la divisione del lavoro.
3. Solo dopo, valutare se **togliere del tutto il target analitico** (e/o riattivare la
   StrategicHead, Opzione 1) per la scelta pienamente appresa.

---

## Appendice — decisioni in breve (cosa sì, cosa no, perché)

| Idea | Esito | Perché |
|------|-------|--------|
| Compressione latente per-nodo + restore | ❌ No | Trade-off lossy↔recuperabilità; "sbroglio al ritorno" insolubile; **e** il lattice è già persistente → non serve |
| "Due grafi sovrapposti" | ⚪ Già esiste | Lattice intatto persistente + vista finestra: nessuna macchina da costruire |
| Grafo globale sparso dinamico (IR2 puro) | ❌ No (così com'è) | Numero nodi variabile → incompatibile con env GPU a tensori fissi |
| Riattivare StrategicHead sui top-K (Opz.1) | 🟡 Possibile dopo | MARL puro, infrastruttura pronta; ma capire prima perché fu disabilitata; retrain |
| Canale coarse-global foveato (Opz.2) | 🟡 Complementare | La "compressione graduata" fatta statica; semplice; periferia grossolana |
| **Campo di valore su albero BF (Opz.3)** | ✅ **Raccomandata** | Soft (no target rigido), cheap, parallelo, fixed-size, riusa BF; più MARL |
| Sconto distanza + canale dist→frontiera | ✅ Incluso nel design | Interpretabilità e scelta consapevole valore-vs-distanza |
