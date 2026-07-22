# Teammate-belief expansion demos

Visual tests for the teammate-position belief (`env/teammate_belief.py`) — the uniform
geodesic-ball estimator that replaced the old `teammate_pot`. The belief is a UNIFORM Σ=1
distribution over every node the teammate could have reached since last contact: born at the
last-known node when comm breaks, then **expanding one hop per env step** over the optimistic
(known-free ∪ frontier→unknown) graph. It collapses to a point at each contact.

## Expansion rules (verified)

1. **Known zone** — spread only over `edge_valid` (known-free, 8-connected): stops at known walls. ✅
2. **Unknown interior** — spread over the optimistic graph, 8-connected (walls invisible → all neighbours). ✅
3. **Frontier (known→unknown)** — cross ONLY through orthogonal edges ("archi generabili"); a diagonal
   free→unknown cuts a corner and is not a generatable path. Fixed 2026-07-21 in
   `env/explorer.py` (`optim_ok = internal_u | (crossing & _orth_k)`): frontier diagonal crossings
   312 → 0, orthogonal 172 kept, unknown-internal diagonals untouched (13932). ✅
4. **Re-entry** — the known term uses `edge_free` (known-free edges WITHOUT the robot-rooted
   `node_valid` gate, added to `graph_lattice.build`), because the belief BFS's from the SEED, not the
   robot. So a known-free POCKET sealed off from the seed inside the known map but reachable by going
   out into the unknown and back gets filled from the far side. Fixed 2026-07-21. ✅

## GIFs

### Controlled cases — `scripts/viz_belief_cases.py` (big synthetic map, exact frontier count)
- **`belief_case_known.gif`** — map fully KNOWN, two serpentine walls: belief winds top→right-gap→
  middle→left-gap→bottom, routing AROUND walls (rule 1, no unknown involved).
- **`belief_case_one.gif`** — one known room, wall with ONE gap: belief fills the room then exits the
  single frontier and floods the unknown.
- **`belief_case_three.gif`** — same room, THREE separate gaps: three simultaneous frontier exits.
- **`belief_case_reentry.gif`** — left room (seed) + a right known-free POCKET sealed off by a full
  wall, both touching unknown. Belief fills left, goes out top/bottom into the unknown, wraps AROUND the
  wall, and re-enters the pocket from the far side (rule 4). Without `edge_free` the pocket stayed empty.
- **`belief_frontier.gif`** (`scripts/viz_belief_frontier.py`) — frozen partial real map; last frame
  draws every active expansion edge colored by type (green known-free / orange frontier-cross / blue unknown).

### PATHFRONT model — KNOWN-graph only (`env/teammate_belief_pathfront.py`, `belief_mode='pathfront'`)
Two phases, all on **known-free nodes** (never unknown — the policy can't read unknown nodes and a
deployment map has no fixed size for them): from the last-known node (✕), up to 6 frontier-cluster
hypotheses freeze; each is a BF particle that TRANSITS the geodesic lkp→frontier (weight =
utility/distance), then on arrival its mass enters an **absorbing diffusion**: it diffuses one hop
INWARD each step over the known-free graph, and each frontier node LOCKS a fraction
`β_F = min(gain·utility(F), β_max)` of its live mass into an accumulator (β = utility → high-utility
frontier locks fast = "teammate pushed out & far"; dead-end frontier keeps mass in the interior). When
a frontier is later EXPLORED it stops being a frontier → its accumulator RELEASES and flows on to the
new outer frontier. Σ p = 1 always; blooms/overlaps sum automatically (one merged live+acc field).

Both GIFs use a **classic heatmap on a FIXED scale 0→`--vmax`** (custom black→light-yellow→red ramp,
NOT per-frame normalised) with the **map visible** underneath (free grey / wall dark red / unknown
near-black); `--frame-ms` sets speed.
- **`belief_pathfront.gif`** (`scripts/viz_belief_pathfront.py`) — FROZEN partial disk map: watch the
  gradient rise toward the frontiers + probability peaks accumulate ON the frontier nodes; interior
  empties; nothing in the unknown.
- **`belief_pathfront_move.gif`** (`scripts/viz_belief_pathfront_move.py`) — the OBSERVER MOVES and
  explores past its frontiers: the accumulated mass RELEASES from explored frontiers and CHASES the
  new outer frontiers as the known map grows. cyan = observer, lime = teammate truth, ✕ = last-known.
Knobs: `--absorb-gain` (β gain), `--beta-max`, `--diffuse-lambda`, `--vmax`. Alt. to the uniform model.

### Real map, partially revealed — `scripts/viz_belief_real.py`
- **`belief_real.gif`** — a real dataset map (train/difficult#5) with a KNOWN disk (R=260px) revealed
  around the centre (true free/wall layout inside, unknown outside). Belief seeded centrally: fills the
  known disk routing AROUND the real internal walls, then leaks out through the natural frontiers
  (green rings) and floods the unknown. The realistic version of the three synthetic cases.

### Free-run demos

- **`belief_expand_clean.gif`** — purpose-built "watch it bloom". Stationary observer a0,
  teammate a1 walks away. Nodes colored by the step they ENTERED the zone (birth-step gradient),
  so you see concentric wavefronts: dark = seeded early (near last-known ✕), bright = just reached
  (the growing frontier). Teammate truth = lime dot, observer = cyan ring.
  Generated by `scripts/viz_belief_expand.py`.

- **`belief_model_difficult.gif`** — the real thing, 2-panel (each agent's belief of the other)
  driven by a trained policy (`test/ckpt_100.pt`) on a difficult map. Belief overlaid on the
  actual explored occupancy/frontier. Watch it start as a point IN COMM, then spread as the
  agents separate. Generated by `scripts/viz_belief.py`.

## Regenerate

```bash
# three controlled cases (fully-known / one frontier / three frontiers)
docker exec marlauder python /workspace/MARLauder/scripts/viz_belief_cases.py \
    --out-dir /workspace/MARLauder/test/belief --hops 120

# real map, known disk around centre + natural frontiers
docker exec marlauder python /workspace/MARLauder/scripts/viz_belief_real.py \
    --split train/difficult --map-idx 5 --radius 260 --hops 100 \
    --out /workspace/MARLauder/test/belief/belief_real.gif

# pathfront two-phase hypothesis model
docker exec marlauder python /workspace/MARLauder/scripts/viz_belief_pathfront.py \
    --split train/difficult --map-idx 5 --radius 300 --steps 90 \
    --out /workspace/MARLauder/test/belief/belief_pathfront.gif

# clean bloom (no checkpoint needed)
docker exec marlauder python /workspace/MARLauder/scripts/viz_belief_expand.py \
    --split train/difficult --map-idx 30 --steps 150 --comm-range 40 \
    --out /workspace/MARLauder/test/belief/belief_expand_clean.gif

# policy-driven, on-map
docker exec marlauder python /workspace/MARLauder/scripts/viz_belief.py \
    --policy model --ckpt /workspace/MARLauder/test/ckpt_100.pt \
    --split train/difficult --map-idx 30 --steps 220 --comm-range 45 \
    --out /workspace/MARLauder/test/belief/belief_model_difficult.gif
```

Lower `--comm-range` → contact breaks sooner → belief is born and expands earlier.
