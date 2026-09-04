# Diagnosi di `contrib_imbalance` — Passo 0 (split a 3 vie di `idle_frac`)

**Data:** 2026-08-20 · **Training lanciati: nessuno** · **Log grezzo:** `MARLauder/idle_diag_v2.log`

Perche': sotto protocollo v2 sia v20 sia v16 perdono contro IR2 su `contrib_imbalance` in 6 celle
su 6, e il divario si allarga a M=4 (hybrid .206 -> .572). Il piano prevedeva di spaccare
`idle_frac` in due regimi disgiunti con fix **opposti** prima di scegliere un intervento.

## 0. Cosa e' stato modificato per poterlo misurare

Lo split a 3 vie era gia' in `explorer.py:1197-1257` e `scripts/idle_diag.py` esisteva gia'.
Mancavano tre cose, tutte aggiunte a `idle_diag.py`:

1. **Ranking per contributo dentro l'episodio.** `by_agent` e' per SLOT, e uno slot non e' un ruolo:
   l'agente che porta l'episodio e' un indice diverso su ogni mappa, quindi la media per slot lava
   via l'effetto. `contrib_imbalance` e' per-episodio, quindi lo sono anche i bucket ora.
2. **`--ir2-dir`** — condizioni protocollo v2 (100 mappe del confronto, start di IR2, budget `D_k`).
   Il budget guida `travel_frac`, che guida l'urgency del rendezvous, che e' cio' che tira insieme
   gli agenti: diagnosticare con un altro budget diagnostica un'altra policy. Riusa
   `_load_ir2_reference` di `eval_comparison` invece di copiarla.
3. **Gate esplicito** contro i CSV del confronto (`max_dist`, `steps`, `contrib_imbalance`).

### DIFETTO TROVATO — `max_episode_steps` non e' solo un cap

Il gate non passava: 1408 px / 72.1 step / imb .579 contro i 1456 / 74.5 / .615 del CSV, stesso
ckpt e stesse mappe. Due cause, entrambe reali e entrambe fuori da questo script:

- **autocast bf16.** `idle_diag` girava l'attore in bf16, `eval_comparison` in piena precisione.
  Cambia l'argmax sui quasi-pareggi abbastanza spesso da spostare la traiettoria.
- **`max_episode_steps` entra nell'OSSERVAZIONE.** `explorer.py:2345` normalizza il tempo
  d'episodio con `T_max = max_episode_steps`. Il diag usava 768 (dal ckpt), il confronto girava la
  cella al suo safety-net cap (462 su hybrid M=4). Stesso checkpoint, stesso budget, **orologio
  diverso in input all'attore -> policy diversa.**

> Vale per qualunque script che carichi un ckpt con un cap diverso da quello di training
> (`eval_ckpt.py`, `pf_scenarios.py`, `score_ckpts.py`): sta valutando una policy che vede un
> orologio che non ha mai visto. Non era documentato da nessuna parte.

Dopo il fix il gate passa su tutte e 12 le celle: `max_dist` entro 1.3%, `contrib_imbalance` entro
0.015 assoluti. Il diag gira gli stessi episodi dei CSV.

## 1. Risultato principale — TRANSIT domina ovunque

Quote su tutti gli agent-step, v20, condizioni v2, 100 mappe per cella:

| cella | productive | redundant | transit | idle_frac |
|---|---|---|---|---|
| hybrid_M2   | .434 | .090 | **.477** | .566 |
| hybrid_M4   | .286 | .092 | **.622** | .714 |
| corridor_M2 | .401 | .166 | **.433** | .599 |
| corridor_M4 | .242 | .206 | **.552** | .758 |
| complex_M2  | .344 | .116 | **.539** | .656 |
| complex_M4  | .238 | .166 | **.595** | .762 |

v16 e' praticamente sovrapponibile (max scarto .024 su qualunque cella). **Non e' una regressione
di v20: e' strutturale della famiglia.**

Il gate della Fase 2 del piano voleva scegliere fra "ridondanza dominante -> informazione" e
"transito dominante -> assegnazione". Il transito e' 2.6-6.9x la ridondanza. **E' assegnazione.**

## 2. Il gradiente di contributo E' il gradiente di transito

Agenti ordinati DENTRO ogni episodio per quota di contributo (v20):

| cella | rango | quota | productive | redundant | transit |
|---|---|---|---|---|---|
| hybrid_M4 | top | .489 | .526 | .024 | .450 |
| | #1 | .285 | .328 | .064 | .608 |
| | #2 | .162 | .196 | .117 | .687 |
| | **low** | **.064** | .093 | .164 | **.743** |
| corridor_M4 | top | .537 | .516 | .039 | .445 |
| | **low** | **.039** | .040 | **.315** | .644 |
| complex_M4 | top | .401 | .366 | .065 | .569 |
| | **low** | **.119** | .122 | .269 | .609 |

L'agente che contribuisce meno **non e' fermo e non sta duplicando**: sta camminando. Su hybrid
fa il 6.4% del lavoro con il 74.3% dei suoi passi in transito.

Eccezione da registrare: su **corridor** la ridondanza dell'ultimo agente e' .315, quasi il doppio
di hybrid. Corridor e' l'unica cella dove i due problemi coesistono.

## 3. Il transito NON e' vagabondaggio — ed e' quasi sempre razionale

- frontiera raggiungibile nel **99.3-100%** dei passi di transito; gli 0.7% senza frontiera sono
  agenti con `own_cov = 1.000`, cioe' localmente finiti. **Nessun agente e' bloccato o stranded.**
- `cos(turn)` fra passi consecutivi **0.645-0.794**, inversioni 4.8-6.9%, rettilineita' 0.82.
  Cammini impegnati, non oscillazione.
- Le corse lunghe (>=10 passi) sono il 26-37% delle corse ma **76-86% di tutti i passi di transito**.
- **Il motivo del cammino lungo:** `u(nearest)` p50 = .09-.15 contro `u(best)` p50 = .49-.75.
  La frontiera piu' vicina e' >=90% buona come la migliore solo nello **0.7%** dei casi (hybrid),
  1.1% (complex), 22.4% (corridor). Gli agenti **saltano correttamente frontiere spazzatura** e
  commutano verso quelle vere.
- Firma coerente nel progress: dentro la finestra ego (<96 px) il progress e' **-0.49 / -0.75**
  (si allontanano dalla frontiera vicina), oltre 2x finestra e' **+0.40 / +0.53**.

Costo residuo misurato: cammino percorso / distanza alla frontiera migliore = **1.6x** su hybrid,
**1.34x** su corridor, **0.76x** su complex. Inefficienza modesta, non il problema.

## 4. Il meccanismo vero: in contatto commutano INSIEME

| cella | in-comm productive | in-comm transit | out-of-comm productive | out-of-comm transit |
|---|---|---|---|---|
| hybrid_M4 | .214 | **.721** | .372 | .504 |
| corridor_M4 | .128 | **.738** | .332 | .405 |
| complex_M2 | .162 | **.818** | .369 | .502 |
| complex_M4 | .111 | **.812** | .291 | .505 |

In contatto radio la produttivita' **si dimezza** e il transito sale di 20-30 punti. Ma il modo
NON e' ridondanza: `redundant-while-out-of-comm` e' solo .056-.160 di tutti gli agent-step.

Questa e' la versione precisa del loop gia' annotato in `explorer.py:229-232`:

> *"contact fuses the maps, identical maps produce identical utility fields, identical fields pick
> the same frontier, co-location produces more contact."*

Con il relay acceso le belief si allineano PIU' in fretta, quindi i campi di utility coincidono di
piu', quindi **commutano verso la stessa frontiera lontana**. Non riscansionano lo stesso terreno
(il relay lo impedisce): **ri-percorrono lo stesso terreno**. E la distanza e' esattamente cio' che
il budget addebita.

Torna con tutto il resto:
- v20 (relay ON) e' significativamente PEGGIORE di v16 su `contrib_imbalance` su hybrid
  (M=2 p=.012, M=4 p=.003), pur essendo migliore su connettivita';
- `r(imbalance, pair_dist)` e' **negativa** (corridor_M4 -0.398): piu' si separano, meno sbilancio;
- `r(imbalance, comm_duty)` e' positiva a M=4.

## 5. Conseguenze sui fix

**Escluso: piu' informazione.** Il relay c'e' gia' ed e' acceso; la ridondanza e' il regime
minoritario; nessun agente e' stranded. Non e' un problema di sapere cosa ha il team.

**Escluso: diversity loss sui logit locali.** E' il port gia' fallito in v17/J.1/J.2. Qui la
divergenza da rompere e' a livello di FRONTIERA scelta, a 200-500 px di distanza; una penalita'
sugli 8 vicini del nodo proprio e' zero esattamente quando serve.

**Candidato 1 — `comm_idle_pen`, gia' implementato, `0.0` in v20 e assente in v16.**
La sua condizione di scatto e' "in contatto senza scambiare", che e' ESATTAMENTE la riga della
tabella §4 dove la produttivita' crolla. Costo: il solo stadio 4 (17.7 h) warm-started da
`v19_m4/ckpt_best`.
> TRAPPOLA: l'esenzione di fine episodio riusa la rampa di urgency del budget, e sotto v2 quella
> rampa non spara quasi mai (finiamo al ~54% di `D_k` -> `urgency ~ 0.09`). Il rendezvous
> terminale, che `done_mode=own` richiede, verrebbe tassato. Va riagganciata a qualcosa che
> sopravviva al cambio di budget PRIMA di lanciare.

**Candidato 2 — diversita' a livello di frontiera** (proiezione dei logit lungo l'albero BF per
ottenere P(agente -> frontiera), ~150-200 righe). E' il fix di principio del meccanismo §4.

**Candidato 3, da misurare per primo perche' costa minuti — `vf_gamma` (0.97).** E' il discount
per hop sulla massa di utility, cioe' il parametro che decide quanto lontano vale la pena andare.
E' il manico diretto sulla lunghezza delle commute, ed e' uno scalare.

## 6. Limite onesto di questa diagnosi

Il transito e' in gran parte razionale (§3): le frontiere buone sono poche e lontane, e sotto un
budget di distanza qualcuno DEVE commutare. Una parte dello sbilancio e' irriducibile. Quello che
IR2 fa diversamente non e' scegliere meglio — e' che la stessa commute le costa **1 step** (salto
di grafo fino a 160 px) invece dei nostri ~15 hop di lattice, quindi i suoi agenti arrivano e
contribuiscono dentro lo stesso episodio. Aspettarsi di annullare il divario e' irrealistico;
ridurlo attaccando la co-locazione (§4) no.

---

# Passo 1 — ablazioni a costo minuti (2026-08-21)

Tutte su **hybrid M=4**, la cella col divario massimo (IR2 .206 contro i nostri .569), 100 mappe,
condizioni protocollo v2. CSV in `eval/comparison/_sweep_gamma/`, **fuori** dai risultati
pubblicati. Tutte OFF-DISTRIBUTION: misurano la sensibilita' del comportamento appreso a un
ingresso, non cosa farebbe una policy riaddestrata li'.

## 7.0 Rumore di fondo, misurato

Rieseguita la cella con il codice rifattorizzato e **nessun override** (percorso di codice
identico, `env_overrides=None`):

| | NOOP | pubblicato | Δ |
|---|---|---|---|
| max_dist | 1465.1 | 1457.8 | +0.5% |
| success | .98 | .97 | +.01 |
| connectivity | .77 | .72 | **+.05** |
| contrib_imbalance | .5689 | .5721 | −.003 |

Puro non-determinismo GPU. **Soglia sotto la quale nulla e' un segnale: max_dist ±0.5%,
connectivity ±0.05, success ±0.01, imbalance ±0.003.** Nota che `connectivity` e' binaria per
episodio: 5 flip su 100 dalla sola perturbazione numerica, quindi la soglia pratica su quella
colonna e' piu' alta di quanto suggerisca un Wilcoxon.

## 7.1 L'orizzonte NON e' la leva — `vf_gamma` / `radar_gamma` sono morti

| arm | contrib_imbalance | Δ vs NOOP | p | max_dist | Δ | conn |
|---|---|---|---|---|---|---|
| NOOP (0.97/0.97) | .569 | — | — | 1465 | — | .77 |
| vf_gamma 0.99 | .577 | +.008 | .038 | 1428 | −37 | .74 |
| vf_gamma 0.90 | .590 | +.021 | .024 | 1421 | −44 | .76 |
| vf_gamma 0.80 | .561 | −.008 | .72 | 1428 | −38 | .72 |
| vf_gamma 0.60 | .555 | −.014 | .52 | 1444 | −22 | .67 |
| radar_gamma 0.90 | .568 | −.001 | .73 | 1490 | +25 | .74 |
| radar_gamma 0.80 | .574 | +.005 | .70 | **1554** | **+89** | .67 |
| entrambi 0.80 | .585 | +.016 | .37 | **1543** | **+78** | .71 |

Su un range di γ da 0.60 a 0.99 — da "vede solo il proprio naso" (0.80^20hop = 0.012) a
"vede tutta la mappa" (0.99^20 = 0.82) — `contrib_imbalance` si muove al massimo di **±0.02**
contro un divario di **0.363**, e **non monotonicamente**. Le uniche variazioni significative
sono peggioramenti.

Abbassare `radar_gamma` costa distanza (+89 px, p<1e-4) e connettivita' (−0.10) senza comprare
nulla sull'equita'. **Candidato 3 chiuso.**

Il motivo, col senno di poi, e' il meccanismo stesso: se gli agenti sono co-locati e diventano
entrambi miopi, la frontiera vicina e' **la stessa** per tutti e due. La miopia non rompe la
simmetria, la stringe.

## 7.2 Sonda causale sull'asse della CONDIVISIONE

Tre punti a parita' di tutto il resto, per testare "belief identiche → utility identiche →
stessa frontiera".

| arm | contrib_imbalance | Δ vs protocollo | p | max_dist | Δ | comm_duty |
|---|---|---|---|---|---|---|
| **meno** condivisione (`--no-comm-relay`) | .576 | +.008 | .56 | 1455 | −10 | .274 |
| protocollo (relay ON) | .569 | — | — | 1465 | — | .275 |
| **occupancy condivisa sempre** | **.633** | **+.064** | **.0058** | 1213 | −252 | 1.000 |
| comm totale (mappe **+ posizioni**) | .554 | −.015 | .22 | 1097 | −368 | 1.000 |

Tre letture, in ordine di importanza:

1. **La simmetria delle belief E' causale.** Forzare occupancy identica alza l'imbalance di
   +0.064, p=.006. Ed e' **sottostimato**: quell'arm accorcia anche gli episodi del 17%, e
   l'imbalance CRESCE con la distanza (correlazione positiva in 11 celle su 12), quindi
   l'effetto vero e' piu' grande del misurato.
2. **Ma il relay non e' il colpevole.** Spegnerlo non cambia nulla (+.008, p=.56). La lettura
   "v20 e' peggiore di v16 perche' ha il relay" era **confusa dal training diverso**, non causale.
   Correggo quella conclusione di ieri.
3. **La posizione del compagno cancella l'effetto.** Con mappe identiche *e* posizioni sempre
   note l'imbalance torna al livello base (−.015, n.s.) nonostante episodi ancora piu' corti.

**Diagnosi raffinata:** non e' avere la stessa mappa a far scegliere la stessa frontiera. E'
avere la stessa mappa **senza sapere che il compagno ci sta gia' andando**. Il tie-break e'
l'informazione sull'altro, non la diversita' della propria mappa.

Coerente con i numeri del Passo 0: `comm_duty` a hybrid M=4 e' .275, cioe' **gli agenti sono
fuori contatto nel 72% dei passi** e la posizione nota del compagno e' stantia quasi sempre.
L'attore ha `last_known_pos`, ma non ha la sua **destinazione**.

## 7.3 Dove restano i soldi

- ~~Orizzonte (`vf_gamma`, `radar_gamma`)~~ — chiuso da §7.1.
- ~~Piu' informazione sulla MAPPA~~ — chiuso dal Passo 0 (ridondanza minoritaria) e da §7.2 riga 1.
- **Informazione sull'INTENTO del compagno** (dove sta andando, non dov'era) — e' l'unica leva
  che in §7.2 ha spento l'effetto. Costo: modifica di osservazione + retrain.
- **Diversita' a livello di FRONTIERA** (proiezione dei logit lungo l'albero BF, ~150-200 righe) —
  insegna a priori a spartirsi, senza dipendere da informazione viva. Complementare alla
  precedente, non alternativa.
- **`comm_idle_pen`** — non testabile a eval (e' un termine di reward). Resta plausibile ma dopo
  §7.2 la sua motivazione e' piu' debole: spinge alla separazione FISICA, mentre la sonda dice che
  conta l'informazione sull'altro, non la distanza da lui.

---

# Passo 2 — diversita' a livello di FRONTIERA, implementata (2026-08-21)

Nessun training lanciato. Codice + test; il retrain e' una decisione separata.

## 8.1 Il termine

Loss ausiliaria dell'ATTORE, bilineare nelle policy di due agenti su uno spazio che condividono:

    L = media_su_coppie_ordinate  sum_{k,l}  pi_i(k) . O[i,j,k,l] . pi_j(l)

`O[i,j,k,l]` = massa di frontiera scontata che l'uscita k di i e l'uscita l di j raggiungono IN
COMUNE. Costruita da cio' che `value_field` gia' calcolava e buttava via nello scatter finale:

    label[v]      il primo ramo da cui il nodo v pende nell'albero BF dell'agente
    mass[v]       gamma_vf^hops(v) . utility(v)
    w[m,k,v]      = mass[v] se label[v]==k, altrimenti 0     (massa per uscita, PER NODO)
    O[i,j,k,l]    = sum_v w[i,k,v] . w[j,l,v]

L'asse `v` e' l'indice del LATTICE, che ogni agente condivide. L'indice di ramo no — ed e'
esattamente il motivo per cui il port v17/J.1/J.2 valeva zero: due agenti a 300 px non condividono
nessun indice d'azione, quindi la penalita' era nulla proprio quando la duplicazione veniva decisa.

Tre proprieta' volute:

- **AUTO-ESTINGUENTE.** Insiemi di frontiere disgiunti -> overlap 0 -> gradiente 0. Smette di
  spingere quando la squadra si e' divisa davvero; non resta un termine che tira a vuoto.
- **LOSS, NON REWARD.** Ritorno, advantage e target del critic intatti. E' l'errore che ha ucciso
  v18 (budget del reward spostato a M=4, novel/completion invertiti).
- **SCALE-FREE.** Ogni agente normalizzato a massa unitaria PRIMA del prodotto: chi sta in una
  zona ricca di utility non domina il termine solo per avere piu' massa, e il numero resta
  comparabile fra passi, mappe e M.

## 8.2 File

| file | cosa |
|---|---|
| `env/graph_lattice.py` | `value_field(..., return_branch=True)` restituisce anche `label` e `mass`. Costo zero: esistevano gia'. |
| `env/explorer.py` | `EnvCfg.div_overlap` (default False) + `_branch_overlap()`; emette `obs["div_overlap"]` [N,M,M,K,K] |
| `models/actor_critic.py` | `evaluate_step_from_enc` restituisce `probs` (la loss e' bilineare in DUE policy, `logp` dell'azione campionata non basta) |
| `train/buffer.py` | allocazione CONDIZIONATA alla presenza della chiave in `sample_obs` |
| `train/mappo.py` | `MAPPOCfg.div_weight` (default 0.0) + il termine nell'actor loss + stat `div_loss` |
| `scripts/train_args.py`, `run_train.py`, `train/driver.py` | `--div-weight`, wiring, logging |

Con `--div-weight 0` (default) l'env non emette la chiave, il buffer non alloca, l'update non la
cerca: **no-op esatto**, ogni checkpoint esistente rigioca identico.

## 8.3 Test — `scripts/17_test_frontier_div.py`, tutti verdi

```
ok 1  OFF: key absent -> buffer allocates nothing, update never sees it
ok 2  shape (4, 3, 3, 8, 8), symmetric under (i,k)<->(j,l) to 0.0e+00
ok 3  arithmetic exact (shared 0.500 / 0.500, disjoint pair 0), invariant to a x100 mass rescale
ok 3b Cauchy-Schwarz holds for all 36 pairs, all finite
ok 4  identical state (same node + same belief): cross/self = 1.00000..1.00000
      same pair spawned apart: 0.4650..0.6752 -> the metric discriminates
ok 5  descent moves the shared pair 0.0156 -> 0.0017
```

Due premesse SBAGLIATE trovate scrivendo i test, entrambe da ricordare:

1. **La riga self NON somma a 1**, somma a `sum_v m_v^2` (indice di Simpson): la normalizzazione e'
   sulla massa, non sul suo quadrato. Sostituita con i controlli che contano davvero — caso
   sintetico con risposta calcolata a mano, invarianza a un riscalamento x100 (prova DIRETTA
   della normalizzazione), Cauchy-Schwarz su mappa vera.
2. **Co-locazione NON e' stato identico.** Ogni agente porta la PROPRIA occupancy, seminata al
   proprio spawn: due agenti sullo stesso nodo hanno ancora mappe diverse, quindi utility diverse,
   quindi alberi diversi. `force_full_occupancy_sharing` non serve qui — agisce dentro `step()` al
   momento della fusione e non tocca un env appena resettato. Servono posizione E occupancy
   copiate. Con entrambe: cross/self = 1.00000 esatto.

## 8.4 Smoke di training

Config identica allo stadio 4 di v20 (32 env, M=4, 6 hop, rollout 256), `--div-weight 1.0`:

```
[it 1] pg=+0.0075  div=0.0055  ent=1.562  ... [done] final.pt
```

- **Memoria: 15562 MiB su 16303**, sotto i 15.47 GiB che la nota di v20 registra come limite. I
  +33 MB di buffer ci stanno. Non e' un margine su cui alzare `--n-envs`.
- **Scala:** `div_loss` ~0.004-0.006 contro `pg_loss` ~0.002-0.019. A peso 1.0 i due sono
  comparabili -> **1.0 e' il punto di partenza sensato**, non un numero da tarare al buio.

FALSO ALLARME REGISTRATO: un primo smoke a 32 env aveva prodotto `ckpt_stop.pt` e un traceback, e
l'avevo letto come un OOM di fine run. Era il mio `timeout 900` della shell: `status.json` dice
`elapsed 896.4 s`, `state "stopped"`, `iter 3/3`, `progress 100.0`, e `ckpt_stop` e' scritto dal
solo handler SIGTERM/atexit (`driver.py:751-760`, "Stop requested (web Stop button / docker stop /
Ctrl-C)"). Nessun difetto. Con timeout adeguato lo stesso config chiude su `final.pt`, eval-suite
e trace inclusi.

## 8.5 Prossimo passo, non fatto

Stadio 4 (~18 h) con `--div-weight 1.0`, warm-start da `v19_m4/ckpt_best`, poi confronto sotto
protocollo v2 contro `v20_v2` sulle stesse 100 mappe. Da leggere con il rumore di fondo di §7.0:
max_dist +-0.5%, connectivity +-0.05, imbalance +-0.003.

---

# Passo 3 — v21 (diversita' di frontiera) e la prova di parita' di ATTRIBUZIONE (2026-08-22)

## 9. v21: la diversita' di frontiera NON funziona

`runs/v21_div_20260821_104706` — stadio 4 identico a v20 (stesso `init-ckpt` v19_m4/ckpt_best,
stessi 850k step, stessa COMMON), unica differenza `--div-weight 1.0`. 103/103 iterazioni, 18h45.

Protocollo v2, hybrid_M4, appaiato con v20 sulle stesse 100 mappe:

| ckpt | contrib_imbalance | Δ vs v20 | p | max_dist | Δ | success | conn |
|---|---|---|---|---|---|---|---|
| v20 | .569 | — | — | 1465 | — | .98 | .77 |
| v21 it20 (`ckpt_best`) | .569 | −.000 | .93 | 1454 | −11 | .99 | .68 |
| v21 it40 | .613 | **+.044** | .012 | 1646 | **+181** | .91 | .44 |
| v21 it100 | .598 | +.030 | .15 | 1822 | **+357** | .76 | .28 |
| v21 it103 (`final`) | .615 | **+.047** | .022 | 1785 | **+320** | .79 | .35 |

**Dose-risposta monotona nella direzione sbagliata.** Non e' il peso da tarare: piu' il termine si
allena, peggio vanno imbalance, distanza, success e connettivita' insieme.

Il termine ha fatto meccanicamente cio' che prometteva — `sensing_overlap` .573→.457, `comm_duty`
.275→.178, `pair_dist_mean` 154→183 px: gli agenti si sparpagliano davvero. Ma sparpagliarsi non
riequilibra i contributi. `idle_diag` su `final` dice dove sono finiti i passi:

| | v20 | v21 final |
|---|---|---|
| productive | .286 | **.238** |
| redundant | .092 | .096 |
| transit | .622 | **.666** |
| quota agente piu' basso | .064 | .057 |
| transit agente piu' basso | .743 | **.774** |

La ridondanza e' **piatta**: non c'era duplicazione da togliere. Il termine ha convertito lavoro
produttivo in TRANSITO. Separare gli agenti allunga i tragitti, e chi pesca la regione lontana
perde l'episodio camminandoci — esattamente il meccanismo del Passo 0, amplificato.

Ipotesi falsificata. `--div-weight` resta nel codice (default 0.0, no-op esatto) come ablazione
documentata.

## 10. E' comportamento o e' contabilita'? — la prova

Domanda: IR2 tiene l'imbalance piu' basso perche' si comporta meglio, o perche' il credito viene
contato in modo diverso? Due regole differiscono davvero:

| | rivendicatore | cadenza |
|---|---|---|
| **IR2** (`compat/env.py:163-166`) | loop sui robot, l'unione e' aggiornata DOPO ciascuno -> **uno solo** per pixel | a fine hop di grafo, ~38-58 px |
| **noi** (`explorer.py:773`) | tutti confrontati con l'unione del passo PRECEDENTE -> una cella scansionata da due agenti nello stesso passo va a **entrambi** | ogni hop di lattice, ≤22.63 px |

Nota che la nostra regola **equalizza**: il doppio credito avvicina le quote, e da noi succede
spesso (`sensing_overlap` .573 a hybrid M=4). Il nostro numero e' gia' un limite inferiore.

`--attr-parity` ricalcola l'imbalance sui NOSTRI stessi episodi sotto le regole di IR2, in due arm
separabili: `SEQ` = rivendicatore unico alla nostra cadenza, `IR2A` = rivendicatore unico **e**
stride di IR2. Lo stride e' preso per mappa come `max_dist / steps` del loro CSV: e' un LIMITE
SUPERIORE del loro stride medio (max_dist e' il robot piu' veloce), quindi rende l'ipotesi
"e' solo contabilita'" piu' facile da confermare. Un risultato nullo sotto questo bias e' il
risultato conservativo.

| cella | IR2 | OURS | SEQ | IR2A | divario | correzione SEQ | correzione IR2A | % del divario | p |
|---|---|---|---|---|---|---|---|---|---|
| hybrid_M2 | .139 | .305 | .304 | .316 | +.167 | −.0011 | **+.0110** | −6.6% | .18 |
| hybrid_M4 | .206 | .569 | .567 | .550 | +.363 | −.0016 | −.0192 | 5.3% | .36 |
| corridor_M2 | .253 | .347 | .330 | .319 | +.095 | −.0176 | −.0283 | **29.9%** | .0002 |
| corridor_M4 | .397 | .696 | .684 | .668 | +.299 | −.0118 | −.0286 | 9.6% | .022 |

**Risposta: no, non e' contabilita'.** Adottare INTEGRALMENTE le regole di IR2 sposta il nostro
numero di **al massimo 0.029 in assoluto**, in ogni cella, contro divari di 0.095-0.363. Su
hybrid_M2 lo sposta perfino nella direzione sbagliata. La quota corretta e' significativa solo
dove il divario e' gia' piccolo (corridor_M2: 30% di 0.095, cioe' 0.028).

Il rivendicatore unico da solo (`SEQ`) non vale quasi nulla (≤0.018): la cadenza conta piu' della
regola di rivendicazione, e comunque poco.

### 10.1 Cosa questa prova NON copre — il limite onesto

Lo stride RITARDA il credito, ma il nostro agente di testa ha comunque spazzato quei corridoi in
transito e il credito gli arriva, solo a blocchi. I robot IR2 **non scansionano affatto durante il
salto**: quei pixel restano ignoti e vengono presi da chiunque ci passi dopo, il che sparpaglia il
credito fra i robot per costruzione.

Quella e' una differenza del MODELLO DI SENSING, non della contabilita' — e' gia' dichiarata nel
protocollo (§5) e non e' emulabile senza cambiare il sensore, a quel punto non e' piu' una
ri-misura degli stessi episodi ma un altro sistema. Resta la spiegazione candidata piu' probabile
per il residuo, e non e' testata.

### 10.2 Conclusione

Con circa il 90% del divario non spiegato dalla contabilita', e con tre interventi indipendenti
(orizzonte, condivisione, diversita' di frontiera) che lo hanno mosso di ±0.06 al massimo, la
lettura difendibile e' quella del Passo 0: il divario e' il prezzo della **granularita' del passo**.
Uno step IR2 e' un salto fino a 160 px, il nostro un hop ≤22.63: il loro robot in ritardo raggiunge
il lavoro lontano in 1-3 step e contribuisce, il nostro ci mette ~18 hop e l'episodio finisce prima.

Il numero che regge in tesi non e' l'imbalance ma la sua conseguenza operativa, la tabella di
scaling M=2→M=4 (§3 del Passo 1): hybrid 19.1% contro 26.8% di IR2 — deficit reale; complex 22.7%
contro 18.2% — **noi meglio**. Accanto ai 6/6 sulla distanza con p<1e-4.
