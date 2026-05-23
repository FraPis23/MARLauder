# MARLauder — MARL decentralizzato per esplorazione (GPU-based)

## Context

Tesi: esplorazione multi-robot con **MARL decentralizzato**, ispirata a IR2 (IROS 2024,
`IR2-Multi-Robot-RL-Exploration/`) ma con tre cambi sostanziali rispetto allo stato dell'arte:

| Aspetto | IR2 (riferimento) | MARLauder (obiettivo) |
|---|---|---|
| Algoritmo | SAC off-policy, replay buffer, double-Q, entropy auto | **MAPPO** on-policy, CTDE, GAE |
| Compute env | numpy/scipy/sklearn su **CPU**, parallelismo Ray (32 worker) | **tutto-GPU**: Warp + PyTorch, env vettorizzati |
| Grafo | topologico dinamico (merge/prune/sparsify, KDTree CPU) | **lattice multi-risoluzione fisso** (tensoriale, GPU) |
| Coordinamento | policy condivisa, CTDE implicito | decentralizzato, policy condivisa + critic centralizzato |
| Comms | signal-strength path-loss + rendezvous implicito | **baseline comms perfette**, vincolo aggiunto dopo |

Obiettivo finale: agenti con LIDAR 360°, ciascuno con il **proprio grafo/belief**, che
collaborano e fanno **rendezvous**. **IR2 NON è un template da replicare**: è il **target di
performance SOTA** da eguagliare o superare a fine progetto. Il design può/deve divergere
liberamente (lattice, MAPPO, tutto-GPU) finché batte le metriche IR2 sulle stesse `DungeonMaps`.

Stack: **PyTorch bare-metal + CleanRL (stile single-file) + NVIDIA Warp**.

> Nota: questa scaletta va anche salvata come `MARLauder/ROADMAP.md` (deliverable di progetto) —
> prima azione della Fase 0.

---

## Librerie usate

| Libreria | Uso |
|---|---|
| `warp-lang` (NVIDIA Warp) | kernel GPU: raycasting LIDAR, update occupancy grid, frontier conv, collision-check edge |
| `torch` (PyTorch) | reti actor/critic, loop MAPPO, autograd; interop zero-copy con Warp |
| `numpy` | preparazione host-side, caricamento mappe |
| `scikit-image` | `block_reduce` (downsample mappe), I/O immagini come in IR2 |
| `Pillow` / `imageio` | caricare `DungeonMaps`, salvare gif/frame |
| `matplotlib` | plot benchmark, visualizzazione grafo/copertura, gif |
| `tensorboard` (o `wandb`) | logging curve training (return, copertura, KL, loss) |
| `omegaconf` / `pyyaml` | config esperimenti |
| `einops` | reshape tensori per attention (leggibilità) |
| `tqdm` | progress bar |

Note:
- **CleanRL** usato come *stile* (single-file), non come dipendenza runtime; niente `gymnasium`
  perché l'env è custom e vettorizzato su GPU.
- **Ray NON usato** (a differenza di IR2): il parallelismo viene dalla vettorizzazione GPU, non da worker CPU.

### Decisioni prese con l'utente
- Grafo = **lattice multi-risoluzione fisso** (nodi statici, edge-mask dinamica via collision-check GPU).
- Comms = **perfette come baseline**; signal-strength/vincolo aggiunto in fase finale per il confronto IR2.
- Confronto **diretto con IR2**: stesse mappe, stesse metriche (% esplorato vs distanza, overlap, tempo rendezvous).

---

## Critica / opinioni sullo stack (lette prima di procedere)

1. **Grafo dinamico su GPU = trappola.** Le graph-ops di IR2 (merge/prune/A*/KDTree) sono
   irregolari e sequenziali, mal mappate su GPU. Il lattice fisso le elimina: nodi statici,
   solo l'**edge-mask** (collision-free?) cambia ogni step ed è calcolabile in parallelo con Warp.
   Scelta giusta per restare tutto-GPU.
2. **MAPPO + env vettorizzati GPU = accoppiata coerente.** On-policy esige throughput → tanti env
   paralleli (ricetta Isaac Gym). È il motivo per cui Warp ha senso qui.
3. **CleanRL non è plug-and-play** con obs a grafo e n. agenti variabile. Si usa lo *stile*
   (single-file, niente astrazioni, riproducibile per tesi), non il codice as-is.
4. **Tensione comms.** Baseline perfetto valida in fretta, ma il valore scientifico di IR2 è il
   vincolo di comunicazione. Tenere il vincolo come fase esplicita e non opzionale per il confronto.
5. **Numero agenti variabile** (IR2: 3–5): su GPU conviene **M fisso per batch** + padding/masking,
   o curriculum su M. Decidere presto per non rifare i tensori.

---

## Decisioni di design (risposte ai dubbi)

### Variare il numero di agenti M (DECISO: permutation-invariant)
- **Actor**: opera sull'osservazione del singolo agente → parameter sharing → **agnostico a M**.
  Robot omogenei → policy condivisa + agent-id nell'osservazione.
- **Critic centralizzato**: **permutation-invariant** (attention/pooling sugli agenti, NON concat) →
  input agnostico a M, generalizza a M mai visti, abilita curriculum su M.
- **Buffer/env**: shape con M_max + padding/mask, mai M hardcoded.
- **Conseguenza**: variare M = **solo retrain** (spesso zero-shot).
- Alternative scartate: concat (M cablato, retrain strutturale); M_max+padding senza attention
  (tetto rigido, non permutation-invariant).

### Grafo gerarchico a 2 livelli (risolve dimensione-fissa vs spazio ignoto)
Due assi distinti: **gerarchia** (livelli di dettaglio) e **ancoraggio** (assoluto vs relativo).
Si combinano così — un livello per ciascun ancoraggio:

| Livello | Scopo | Ancoraggio | Densità | Capacità |
|---|---|---|---|---|
| **Basso (locale)** | navigazione fine, azione immediata | **ego-centrico** (finestra K×K mobile col robot) | denso | fissa (es. 21×21) |
| **Alto (globale)** | frontiere lontane, pose compagni, rendezvous | **assoluto** (anchor sparsi in coord mondo) | sparso | fissa, capped (es. 64) |

- Info globali → livello alto; info specifiche → livello basso. Risolve "spazio ignoto": il livello alto
  è sparso+capped (non cresce con la mappa), il basso è ego-centrico (indipendente dalla dimensione mondo).
- **L'attention NON riceve un grafo che cresce all'infinito**: i due livelli sono uniti in **un unico
  tensore paddato** `[N_max, d]` con un **canale-feature che marca il livello**; una sola attention
  mascherata gira sull'insieme. Capacità fissa (cfr IR2 `NODE_PADDING_SIZE=360`); i nodi si **attivano
  dinamicamente** man mano che la mappa è scoperta (mask 0→1), gli slot vuoti restano padding.
- Azione = **pointer sui vicini del nodo corrente nel livello fine** (movimento locale); il livello alto
  informa encoding/value (dove dirigersi a lungo raggio).
- **Niente merge/prune dinamico** (a differenza di IR2): capacità fissa + mask. Oltre capacità del livello
  alto → drop anchor a utility più bassa.
- Upgrade futuro opzionale: due encoder separati + cross-attention locale→globale (più espressivo).
  Iniziare col set unificato (stile `PolicyNet` IR2).

## Concetti riusabili da IR2 (non reinventare)

- `sensor.py::sensor_work` — modello LIDAR 360° (720 raggi, 0.5°, range 80px). → portare in kernel Warp.
- `env.py::find_frontier` — frontier = bordo free(255)/unknown(127), 8-vicini. → conv 2D su GPU.
- `node.py` — utility nodo = #frontier in line-of-sight entro raggio. → batch su lattice.
- `env.py::merge_beliefs` — merge map element-wise (max). → per comms/rendezvous.
- `model.py::PolicyNet` — encoder attention + **pointer** su vicini del nodo corrente; `INPUT_DIM=6`,
  `EMBEDDING_DIM=128`, `K_SIZE=30`, `NODE_PADDING_SIZE=360`. → riusare schema, riscrivere per MAPPO.
- `ss_realistic_model.py` — path-loss (P_T=-20, thr=-70, γ=2, γ_obst=4). → solo fase comms finale.
- `parameter.py` — costanti (SENSOR_RANGE=80, GAMMA=0.995, scaling factor). → punto di partenza valori.
- `DungeonMaps/` — mappe per training e per il confronto IR2.

---

## Scaletta sequenziale (ogni fase ha un GATE di validazione)

### Fase 0 — Scaffold & ambiente
- Salva questa scaletta come `MARLauder/ROADMAP.md` (deliverable).
- Struttura repo `MARLauder/` (`env/`, `models/`, `train/`, `configs/`, `eval/`).
- Verifica toolchain: CUDA + `warp-lang` + PyTorch sulla stessa GPU; smoke-test kernel Warp banale.
- **GATE:** kernel Warp gira su GPU, tensori Torch↔Warp si scambiano senza copia host.

### Fase 1 — Mondo occupancy-grid + LIDAR su GPU (Warp, no RL)
- Carica `DungeonMaps` come tensori GPU (ground-truth: 1=ostacolo, 255=free, 127=unknown).
- Kernel Warp di raycasting 360° → aggiorna belief grid di **un** agente.
- **GATE:** belief riempita correttamente vs ground-truth; confronto visivo con `sensor_work` IR2; tempo/step misurato.

### Fase 2 — Lattice ego-centrico + anchor globali + frontier + utility + edge-mask (GPU)
- Lattice 2 livelli: **fine ego-window** (K×K mobile attorno al robot, capacità fissa) +
  **globale sparso di anchor** (visitati/frontier, capped + padded, padding-mask).
- Nodi **attivati dinamicamente** man mano che la mappa è scoperta (mask, non ricostruzione).
- Frontier detection (conv GPU) + utility per nodo (line-of-sight batch).
- Edge-mask collision-free su edge del lattice (kernel Warp), ricalcolata dal belief.
- **GATE:** grafo sovrapposto alla mappa coerente; utility sensate; ego-window si ri-ancora col moto;
  capacità nodi mai superata (drop low-utility funziona); connettività verificata su mappa nota.

### Fase 3 — Env vettorizzato N_env × M_agenti (perfect comms)
- API `reset()/step()` interamente su GPU; M fisso per batch (padding/mask per assenti).
- Belief condivisa/merge (comms perfette) tra agenti dello stesso env.
- Reward stile IR2: `new_frontiers + explore_util − dist_penalty` (+ team reward).
- **GATE:** rollout con policy random; copertura cresce; nessun NaN; throughput (step/s) loggato e accettabile.

### Fase 4 — Reti policy/critic (PyTorch), M-agnostiche
- Actor: encoder attention + **pointer** sui vicini del nodo corrente (riuso schema `PolicyNet`),
  parameter sharing → agnostico a M.
- Critic **centralizzato permutation-invariant** (attention/pooling sugli agenti, NON concat) →
  input agnostico a M; riceve stato globale (riassunto mappa merge + set pose agenti).
- Tutto batched su GPU, masking corretto (node/edge padding **e** agent padding a M_max).
- **GATE:** forward con shape corrette a M variabile; maschere rispettate; gradiente integro; gira batched su GPU.

### Fase 5 — Loop MAPPO single-agent (CleanRL-style)
- Single-file: rollout on-policy + GAE + clip PPO + value loss + entropy.
- M=1 su poche mappe.
- **GATE:** impara a esplorare; curva copertura sopra random; loss/KL stabili.

### Fase 6 — Scale a M agenti decentralizzati (policy condivisa, critic centralizzato)
- Decentralized execution; ogni agente decide dal **proprio** sub-grafo locale.
- **GATE:** copertura multi-agente > single; suddivisione spazio emergente; training stabile su più mappe.

### Fase 7 — Rendezvous + vincolo di comunicazione
- Map-merging solo tra agenti connessi; **rendezvous utility layer** (riuso idea IR2 `generate_rendezvous_utility_layer`).
- Vincolo comms: prima range/line-of-sight semplice, poi `ss_realistic_model` su GPU.
- **GATE:** rendezvous implicito emerge (agenti si riconnettono per condividere); performance sotto vincolo misurata.

### Fase 8 — Benchmark vs IR2 (target SOTA da battere)
- IR2 usato solo come **riferimento di performance**, non come vincolo di design.
- Stesse `DungeonMaps`, stesse metriche: % esplorato vs distanza percorsa, overlap mappe, tempo a rendezvous.
- Eval greedy + script confronto e plot.
- **GATE:** MARLauder **eguaglia o supera** i numeri IR2 su almeno le metriche chiave; tabelle/plot riproducibili.

---

## File principali da creare (MARLauder/)

- `env/world_warp.py` — occupancy grid + raycasting LIDAR (kernel Warp).
- `env/lattice.py` — costruzione lattice multi-risoluzione + edge-mask GPU.
- `env/frontier.py` — frontier detection + utility nodi (GPU).
- `env/marl_env.py` — env vettorizzato N_env×M, reset/step, reward, comms.
- `models/networks.py` — actor pointer-attention + critic centralizzato.
- `train/mappo.py` — loop MAPPO single-file (stile CleanRL).
- `configs/` — iper-parametri (partire da `parameter.py` IR2).
- `eval/benchmark_ir2.py` — metriche e confronto su DungeonMaps.

---

## Verifica end-to-end (a regime)

1. `python train/mappo.py --config configs/marl.yaml` → curve copertura/return salgono, training stabile.
2. Eval greedy su mappe held-out: copertura ~completa entro budget distanza.
3. `python eval/benchmark_ir2.py` → plot % esplorato vs distanza e tabella metriche MARLauder vs IR2.
4. Profilo throughput: confermare che l'env gira interamente su GPU (no colli di bottiglia host nel loop).
