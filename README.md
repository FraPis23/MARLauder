# MARLauder

MARL **decentralizzato** per **esplorazione multi-robot**, interamente su **GPU**.
Ispirato a [IR2](https://ir2-explore.github.io/) (IROS 2024) ma con stack e algoritmo diversi;
IR2 è il **target di performance SOTA** da eguagliare/superare, non un template da copiare.

| Aspetto | IR2 | MARLauder |
|---|---|---|
| Algoritmo | SAC off-policy | **MAPPO** on-policy, CTDE |
| Compute | numpy/scipy CPU + Ray | **tutto-GPU**: PyTorch + NVIDIA Warp, env vettorizzati |
| Grafo | topologico dinamico (merge/prune) | **lattice gerarchico fisso** (ego-centrico + anchor globali) |
| Comms | signal-strength | **perfette** (baseline); vincolo aggiunto dopo |

Roadmap completa: [ROADMAP.md](ROADMAP.md).

## Stack
- **NVIDIA Warp** — kernel GPU: raycasting LIDAR, occupancy grid, edge-mask, utility nodi.
- **PyTorch (cu128)** — reti, MAPPO; interop zero-copy con Warp.
- **CleanRL** (stile single-file) — riferimento per il loop di training.
- Tutto in **Docker**, immagine compatibile **RTX 4080 (sm_89)** e **RTX 5080 (sm_120)**.

## Setup (Docker)
```bash
scripts/build.sh        # build immagine (torch 2.7.1 cu128 + warp 1.13)
scripts/up.sh           # avvia container in background
scripts/shell.sh        # apri una shell (eseguibile più volte = più terminali)
scripts/smoke.sh        # GATE Fase 0: torch+warp su GPU, interop zero-copy
scripts/down.sh         # ferma il container
```
Il container monta l'intero `Thesis_Project` → vede sia `MARLauder/` sia le `DungeonMaps` di IR2.
Dentro il container imposta `PYTHONPATH=/workspace/MARLauder` per gli script.

## Dati (mappe)
Le `DungeonMaps` di IR2 vengono preprocessate **una volta** in tensori GPU-ready (memmap), così
il training non decodifica PNG né usa numpy nel loop:
```bash
python scripts/preprocess_maps.py        # -> data/<split>/{maps.npy, meta.npz}
```
Output (~9 GB, **gitignored**): `data/{train/easy, train/difficult, test/{complex,corridor,hybrid}}`.
Convenzione ground-truth: `0=ostacolo, 1=free`; start dal pixel `208` (fallback: cella free random).

## Stato avanzamento

| Fase | Contenuto | Stato |
|---|---|---|
| 0 | Scaffold + Docker (4080/5080) + toolchain | ✅ |
| 1 | Occupancy-grid + LIDAR 360° su GPU (Warp) | ✅ |
| 2 | Lattice gerarchico + frontier + utility + edge-mask | ✅ |
| 3 | Env vettorizzato N_env×M (comms perfette, anti-collisione) | ✅ |
| 4 | Reti actor-pointer + critic centralizzato (M-agnostico) | ✅ |
| 5 | MAPPO single-agent (CleanRL-style) | ☐ |
| 6 | Scale a M agenti decentralizzati | ☐ |
| 7 | Rendezvous + vincolo comunicazione | ☐ |
| 8 | Benchmark vs IR2 | ☐ |

## Struttura
```
env/
  maps.py          loader memmap mappe preprocessate -> GPU
  world_warp.py    WarpWorld: occupancy grid + LIDAR 360° (N mondi x M agenti)
  frontier.py      frontier detection (conv GPU) + centri (anchor)
  graph_lattice.py EgoLattice: validità/edge-mask/utility nodi; anchor globali
  comms.py         connettività + PositionProvider (onniscienza ora, predittore in futuro)
  marl_env.py      MarlExploreEnv: reset/step vettorizzato, spawn, anti-collisione, reward
models/
  networks.py      MarlActorCritic: actor pointer-attention + critic permutation-invariant
train/             loop MAPPO (Fase 5+)
eval/              benchmark IR2 (Fase 8)
scripts/           build/up/shell/down, preprocess, test_*, make_gif
data/              mappe preprocessate (gitignored)
runs/              output: gif, viz, log (gitignored)
```

## Design (decisioni chiave)
- **Grafo gerarchico**: livello basso = lattice **ego-centrico** K×K denso (si muove col robot);
  livello alto = **anchor globali** sparsi (centri di frontiera, capped+padded). Capacità fissa,
  niente merge/prune dinamico; nodi si attivano via mask man mano che la mappa è scoperta.
- **Agenti M-agnostico**: actor a parameter-sharing; critic centralizzato permutation-invariant
  (attention/pooling sugli agenti) → variare M = solo retrain.
- **Comms perfette (baseline)**: belief condivisa per mondo; ogni agente ha però il proprio ego-grafo.
- **Spawn**: agenti nella stessa stanza, su nodi distinti.
- **Collisioni**: proprietà dell'env — **hard-mask** (zero collisioni garantite) **+ piccola penalty**
  (la policy impara anche a non volerle).
- **Comms / posizioni**: gli agenti conoscono solo le **posizioni** dei compagni (non le osservazioni),
  fornite dal `PositionProvider` (baseline onnisciente). In futuro l'onniscienza è sostituita da un
  modulo di **stima posizione** dietro la stessa interfaccia, gated dalla `connectivity`. La belief
  resta condivisa nel baseline e diventa **per-agente in Fase 7** (le shape delle reti non cambiano →
  nessun rework, solo retraining); la comunicazione imperfetta agirà su mappe e posizioni insieme.

## Validazione (GATE per fase)
```bash
python scripts/smoke_test.py     # Fase 0
python scripts/test_lidar.py     # Fase 1  (+ viz)
python scripts/test_graph.py     # Fase 2  (+ viz nodi/edge/frontiere/anchor)
python scripts/test_env.py       # Fase 3  (+ GIF multi-agente)
python scripts/make_gif.py       # GIF random-walk singolo agente
```
Output visivi in `runs/`.

## Performance
Throughput env (random policy, test/complex 1000×1000, 4 agenti):

| N_env | env-step/s | ms/batch-step |
|---|---|---|
| 256 | ~10.3k | 24.7 |
| 512 | ~10.8k | 47.6 |

Ottimizzazioni applicate:
- **frontier downsampled res-4** (come IR2): utility legge la frontier coarse con indicizzazione `/scale`.
- **kernel Warp** per la frontier: calcola la **vera adiacenza full-res** (free con vicino-4 unknown) e
  riduce in coarse (OR). Importante: NON marca i muri come frontiera (il pooling coarse lo faceva →
  utility/anchor falsati sui muri). ~2.4× rispetto al baseline iniziale, e corretto.

TODO perf (rendimenti decrescenti, rinviati): update frontier **incrementale** (solo intorno agli agenti),
vettorizzare i loop di risoluzione collisioni. Da valutare se il training lo richiede.
# MARLauder
