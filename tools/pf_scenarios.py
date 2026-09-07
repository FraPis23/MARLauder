#!/usr/bin/env python3
"""Hand-scripted scenarios for the pathfront teammate belief — one case per page, step by step.

    python tools/pf_scenarios.py                 # all scenarios → runs/pf_scenarios/
    python tools/pf_scenarios.py --only 03       # just one

No policy, no RL episode, no GPU: a small ASCII map, an observer whose moves ARE the script, and
the belief module driven on exactly the same inputs the real env feeds it. That is the point —
every number and every pixel on those pages is a property of `env/teammate_belief_pathfront.py`
alone, so a disagreement about behaviour can be settled by pointing at a step.

The env's asymmetry is reproduced faithfully because it is where the interesting bugs live:
  * SENSING radius reveals the map (small);
  * COMM radius produces `seen` = "he would have been detected if he stood here" (LARGER), which
    is why `seen` routinely swallows a whole frontier cluster the sensor has not even reached.
Both are BFS over free cells, so neither passes through a wall.

Read a page like this: the dot (yellow ring) is a hypothesis still travelling one hop per step; red
is probability; blue tint is `seen` this step; hatched is still unknown. `Σp` must be 1.0000 on
every single step of every scenario — if it is not, nothing else on the page means anything.
"""
from __future__ import annotations

import argparse
import re
import html
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from env.teammate_belief_pathfront import advance_pathfront, freeze_hypotheses  # noqa: E402


def load_impl(path: Path):
    """Run the bench against a DIFFERENT copy of the belief module (e.g. the frozen reference).

        git show b3dce57a:env/teammate_belief_pathfront.py > /tmp/ref.py
        python tools/pf_scenarios.py --impl /tmp/ref.py --tag 00_reference

    Comparing two tags page by page is the whole point of the bench: one change, one tag, one diff.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("pf_alt", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.advance_pathfront, mod.freeze_hypotheses

# 8-neighbour lattice, same as env/graph_lattice.py
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
KMAX = len(DIRS)


# --------------------------------------------------------------------------------------- world
class World:
    """Free/wall grid + revealed-so-far mask. Cells are lattice nodes; index = r*W + c."""

    def __init__(self, rows: list[str]):
        self.rows = rows
        self.H, self.W = len(rows), len(rows[0])
        self.N = self.H * self.W
        self.free = torch.zeros(self.N, dtype=torch.bool)
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                if ch != "#":
                    self.free[r * self.W + c] = True
        self.known = torch.zeros(self.N, dtype=torch.bool)
        self.nbr = torch.zeros((self.N, KMAX), dtype=torch.long)
        self.geo = torch.zeros((self.N, KMAX), dtype=torch.bool)   # geometric free-free adjacency
        for n in range(self.N):
            r, c = divmod(n, self.W)
            for k, (dr, dc) in enumerate(DIRS):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < self.H and 0 <= cc < self.W):
                    continue
                m = rr * self.W + cc
                self.nbr[n, k] = m
                if not (self.free[n] and self.free[m]):
                    continue
                if dr and dc:                     # no corner cutting, same as the real lattice
                    if not (self.free[r * self.W + cc] and self.free[rr * self.W + c]):
                        continue
                self.geo[n, k] = True
        self.xy = torch.stack([
            torch.arange(self.N) % self.W, torch.arange(self.N) // self.W], dim=1).float()

    def ball(self, src: int, radius: int, within_known: bool = False) -> torch.Tensor:
        """Cells reachable from `src` in ≤radius hops over free cells (never through a wall)."""
        out = torch.zeros(self.N, dtype=torch.bool)
        out[src] = True
        frontier = [src]
        for _ in range(radius):
            nxt = []
            for n in frontier:
                for k in range(KMAX):
                    if not self.geo[n, k]:
                        continue
                    m = int(self.nbr[n, k])
                    if within_known and not self.known[m]:
                        continue
                    if not out[m]:
                        out[m] = True
                        nxt.append(m)
            frontier = nxt
        return out & self.free

    def hops_from(self, src: int) -> torch.Tensor:
        """Hop distance from `src` over KNOWN-free cells — how far the observer's ear reaches."""
        BIG = float(self.N)
        d = torch.full((self.N,), BIG)
        d[src] = 0.0
        cur, h = [src], 0
        while cur:
            h += 1
            nxt = []
            for n in cur:
                for k in range(KMAX):
                    if not self.geo[n, k]:
                        continue
                    m = int(self.nbr[n, k])
                    if self.known[m] and d[m] >= BIG:
                        d[m] = float(h)
                        nxt.append(m)
            cur = nxt
        return d

    def reveal(self, src: int, radius: int) -> None:
        self.known |= self.ball(src, radius)

    def edge_free(self) -> torch.Tensor:
        kn = self.known & self.free
        return (self.geo & kn.view(self.N, 1) & kn[self.nbr]).unsqueeze(0)      # [1, N, K]

    def frontier(self) -> torch.Tensor:
        """Known-free cell touching something not yet known — the env's definition."""
        kn = self.known & self.free
        unk_nbr = torch.zeros(self.N, dtype=torch.bool)
        for k in range(KMAX):
            unk_nbr |= self.geo[:, k] & (~self.known[self.nbr[:, k]])
        return (kn & unk_nbr).unsqueeze(0)                                       # [1, N]

    def utility(self, decay: float = 0.55, hops: int = 3, r_sense: int = 2) -> torch.Tensor:
        """Frontier gain diffused a few hops along known-free edges, as `info['utility']` is.

        The SEED is the env's own formula (env/graph_lattice.py): f_ind = ribbon x (FLOOR +
        (1-FLOOR) x volume), where `ribbon` is the fraction of the node's neighbours that are
        free-but-unknown and `volume` is the fraction of a sensor-sized ball around it that is
        still unknown. It used to be a flat 1.0 on every frontier node, which made every opening
        in this bench EQUALLY GOOD — so no scenario here could express "a big opening and a poor
        one", and any rule that weighs openings by quality was untestable by construction.
        """
        FLOOR = 0.25
        fr = self.frontier()[0]
        u = torch.zeros(self.N)
        for n in fr.nonzero().flatten().tolist():
            nb = self.geo[n].nonzero().flatten().tolist()          # free neighbours only
            if not nb:
                continue
            ribbon = sum(1 for k in nb if not bool(self.known[int(self.nbr[n, k])])) / len(nb)
            ball = self.ball(n, r_sense)                            # free cells within reach
            vol = float((ball & ~self.known).sum()) / max(1, int(ball.sum()))
            u[n] = ribbon * (FLOOR + (1.0 - FLOOR) * vol)
        ef = self.edge_free()[0]
        for _ in range(hops):
            best = torch.zeros_like(u)
            for k in range(KMAX):
                best = torch.maximum(best, torch.where(ef[:, k], u[self.nbr[:, k]] * decay,
                                                       torch.zeros_like(u)))
            u = torch.maximum(u, best)
        return u.unsqueeze(0).clamp(0.0, 1.0)                                    # [1, N]


def bf(world: World, src: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Hop distance + parent-toward-src over the KNOWN-free graph (what freeze_hypotheses wants)."""
    INF = float("inf")
    dist = torch.full((world.N,), INF)
    par = torch.full((world.N,), -1, dtype=torch.long)
    dist[src] = 0.0
    ef = world.edge_free()[0]
    frontier = [src]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for n in frontier:
            for k in range(KMAX):
                if not ef[n, k]:
                    continue
                m = int(world.nbr[n, k])
                if dist[m] == INF:
                    dist[m] = float(d)
                    par[m] = n
                    nxt.append(m)
        frontier = nxt
    return dist.unsqueeze(0), par.unsqueeze(0)


# ------------------------------------------------------------------------------------ scenarios
class Scenario:
    def __init__(self, name: str, title: str, why: str, rows: list[str], *,
                 lkp: tuple[int, int], obs_path: list[tuple[int, int]],
                 truth: list[tuple[int, int]] | None = None,
                 r_sense: int = 3, r_comm: int = 6, kf: int = 6,
                 prereveal: list[tuple[int, int, int, int]] | None = None,
                 regions: dict[str, list[tuple[int, int]]] | None = None):
        self.name, self.title, self.why = name, title, why
        self.rows = rows
        self.lkp, self.obs_path, self.truth = lkp, obs_path, truth
        self.r_sense, self.r_comm, self.kf = r_sense, r_comm, kf
        self.prereveal = prereveal or []
        self.regions = regions or {}


def _n(world: World, rc: tuple[int, int]) -> int:
    return rc[0] * world.W + rc[1]


MAPS = {}

H, W = 11, 26
CORRIDOR = (1, 1, 1, 24)


def grid(opens: list[tuple[int, int, int, int]]) -> list[str]:
    """Walls everywhere except the given inclusive (r0, r1, c0, c1) rectangles."""
    rows = [["#"] * W for _ in range(H)]
    for r0, r1, c0, c1 in opens:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                rows[r][c] = "."
    return ["".join(r) for r in rows]


def rect(r0: int, r1: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


# Every scenario shares one shape, because it is the shape the real env produces: a corridor both
# robots have ALREADY mapped (so the belief has a known graph to live on and real frontiers to bet
# on), with rooms still unknown behind doors off it. `lkp` is always outside the observer's comm
# ball at t=0 — that is what "contact was just lost" means, and putting it inside instead makes
# every scenario degenerate into "everything dies at the freeze step".

# 1 ─ plain transit: three doors, observer parked out of reach of all of them.
MAPS["01_transit"] = Scenario(
    "01_transit", "Transit — one hop per step, no mass in the field before arrival",
    "Three unexplored doors off a mapped corridor, and an observer parked far away who never sees "
    "any of them. Nothing but the pacing rule is on show: each dot must advance exactly one node "
    "per step, and the field must stay empty until a dot actually lands.",
    grid([CORRIDOR, (2, 2, 5, 5), (3, 4, 4, 6), (2, 2, 11, 11), (3, 4, 10, 12),
          (2, 2, 17, 17), (3, 4, 16, 18)]),
    lkp=(1, 1), obs_path=[(1, 23)] * 22, r_sense=2, r_comm=4,
    prereveal=[CORRIDOR],
    regions={"door A (c5)": rect(1, 4, 4, 6), "door B (c11)": rect(1, 4, 10, 12),
             "door C (c17)": rect(1, 4, 16, 18)})

# 2 ─ the push: the observer walks THROUGH the door the belief bet on, revealing more behind it.
MAPS["02_push_behind"] = Scenario(
    "02_push_behind", "Push — the door is looked through, new ground opens behind it",
    "One door, and the belief has already landed on it. The observer then walks in and finds a "
    "further passage. The door stops being a frontier and a new one appears deeper: the belief "
    "must FOLLOW it outward, not evaporate and not stay stuck on the old node.",
    # A THIRD door at c=20 that the observer never goes near. Without it the observer ends up
    # having revealed the entire map, no frontier exists anywhere, and the page silently turns
    # into the degenerate `08_no_target_left` corner instead of testing the push.
    grid([CORRIDOR, (2, 2, 6, 6), (3, 5, 4, 8), (6, 6, 8, 8), (7, 9, 6, 12),
          (2, 2, 20, 20), (3, 5, 18, 22)]),
    lkp=(1, 12),
    # The observer starts LEFT of the door and holds until the dot has landed on it; then it walks
    # in. It never goes right of c=6, so the far door at c=20 is never seen and its hypothesis
    # stays alive — which is the whole point: when the near zone runs out there has to be a real
    # surviving target for its mass to go to.
    obs_path=[(1, 1)] * 7 + [(1, c) for c in range(2, 7)]
             + [(2, 6), (3, 6), (4, 6), (5, 6), (5, 7), (5, 8), (6, 8), (7, 8), (7, 9), (7, 10),
                (7, 11), (7, 12)] + [(7, 12)] * 5,
    r_sense=2, r_comm=4, prereveal=[CORRIDOR],
    regions={"first room": rect(2, 5, 4, 8), "room behind it": rect(6, 9, 6, 12),
             "far door (c20)": rect(1, 5, 18, 22), "corridor": rect(1, 1, 1, 24)})

# 3 ─ bifurcation: the room behind the door forks. The observer walks ONE fork to its end.
MAPS["03_bifurcation"] = Scenario(
    "03_bifurcation", "Bifurcation — one fork exhausted, the sibling must take its mass",
    "Behind the door the space forks left and right. The observer walks the LEFT fork to the very "
    "end and finds nothing. Everything that fork was holding belongs to the RIGHT fork — the same "
    "zone — and must not end up somewhere else on the map.",
    grid([CORRIDOR, (2, 2, 6, 6), (3, 4, 4, 8), (5, 9, 4, 5), (5, 9, 7, 8)]),
    lkp=(1, 14),
    obs_path=[(1, c) for c in range(24, 5, -1)] + [(2, 6), (3, 6), (4, 6), (4, 5), (5, 5),
                                                   (6, 5), (7, 5), (8, 5), (9, 5), (9, 4),
                                                   (9, 4), (9, 4)],
    r_sense=2, r_comm=4, prereveal=[CORRIDOR],
    regions={"left fork": rect(5, 9, 4, 5), "right fork": rect(5, 9, 7, 8),
             "shared room": rect(2, 4, 4, 8)})

# 3b ─ the case the nearest-opening rule cannot arbitrate: TWO IDENTICAL DOORS, one near, one far.
#      Both are plain doorways off a room both robots have already mapped, with unexplored space
#      behind each and NO WAY TO SEE HOW MUCH — so their utility is the same by construction, and
#      the ONLY thing separating them is distance. 03 is the symmetric control (equal distance,
#      equal prize → 50/50); this is equal prize, unequal distance. Any rule that weighs openings by
#      QUALITY is a no-op here, which is exactly what makes the page decisive: what is left to
#      argue about is whether distance alone should take everything.
MAPS["03b_two_doors_near_far"] = Scenario(
    "03b_two_doors_near_far",
    "Two identical pockets off one room, one near the way in and one far",
    "ONLY THE CORRIDOR IS KNOWN. Below it, through a single neck, lies a room nobody has entered, "
    "and two identical pockets hang off that room — one right by the way in, one at the far end. "
    "The observer starts at the opposite end of the corridor and walks the whole way back, so the "
    "room and both pockets are revealed BY EXPLORING: the two doors are frontiers the map "
    "generates on its own, not ground pre-marked as known. They are the same size and neither can "
    "be seen through, so nothing separates them but distance. The question: when the belief is "
    "sitting in the room and both pockets open up, does the near one take everything or do they "
    "split?",
    grid([CORRIDOR, (2, 2, 5, 5), (3, 6, 4, 17),
          (7, 7, 6, 6), (8, 9, 5, 8), (7, 7, 15, 15), (8, 9, 14, 17)]),
    lkp=(1, 3),
    # The observer starts far away (so the ring departs and is visible), then walks the corridor
    # back to the neck and sweeps the room left-to-right, which is what makes the two pockets
    # appear. It never enters either pocket — the page is about the split, not about consuming one.
    obs_path=[(1, 23)] * 3 + [(1, c) for c in range(22, 4, -1)]
             + [(2, 5), (3, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12),
                (4, 13), (4, 14), (4, 15), (4, 16)] + [(4, 16)] * 6,
    r_sense=2, r_comm=4, prereveal=[CORRIDOR],
    regions={"near pocket": rect(6, 9, 5, 8), "far pocket": rect(6, 9, 14, 17),
             "room": rect(3, 6, 4, 17)})

# 4 ─ containment: two doors far apart. Sweeping one must not move the other one's probability.
MAPS["04_two_zones"] = Scenario(
    "04_two_zones", "Containment — sweeping one zone must not move the other's mass",
    "Two doors at opposite ends of the corridor, sharing nothing but the corridor. The observer "
    "sweeps the LEFT room completely. What that frees belongs to the left zone's own remaining "
    "ground first, and only reaches the right door once the left zone is genuinely spent.",
    grid([CORRIDOR, (2, 2, 5, 5), (3, 6, 3, 7), (2, 2, 19, 19), (3, 6, 17, 21)]),
    lkp=(1, 12),
    # `lkp` sits 7 hops from BOTH representatives, so the two dots depart symmetrically and the
    # page is about containment and nothing else. The observer HOLDS those 7 steps so the left dot
    # LANDS on its door before he walks in — without the hold he intercepts it in the corridor and
    # the page turns into 05 (interception) instead.
    obs_path=[(1, 1)] * 7 + [(1, c) for c in range(2, 6)]
             + [(2, 5), (3, 5), (3, 3), (4, 3), (5, 3), (6, 3),
                (6, 5), (6, 7), (5, 7), (4, 7), (3, 7), (3, 7)],
    r_sense=2, r_comm=4, prereveal=[CORRIDOR],
    # Regions start at ROW 1: the frontier a zone's mass actually sits on before the room is
    # entered is the corridor cell in front of its door, so a region that starts at row 2 reads
    # 0.0000 for a zone holding half the belief.
    regions={"left zone": rect(1, 6, 3, 7), "right zone": rect(1, 6, 17, 21),
             "middle corridor": rect(1, 1, 8, 16)})

# 5 ─ interception: the observer is parked ON the corridor the dot has to cross.
MAPS["05_ring_intercept"] = Scenario(
    "05_ring_intercept", "Interception — the dot walks into the observer and must die there",
    "Two doors, one on each side of `lkp`, and an observer parked between `lkp` and the RIGHT one. "
    "The right dot has to cross him to get there: the step it enters comm range and nobody is "
    "detected, that route is refuted and the dot must vanish ON THE SPOT — either the teammate "
    "came this way, in which case the radio has him, or he did not. Its weight belongs to the "
    "LEFT dot, which is heading away from the observer and is untouched by any of this.",
    # A door at c=2 as well, so there IS a surviving hypothesis. With only the far door the kill
    # has nowhere to send the freed mass, the no-survivor fallback puts it back down on the
    # corridor behind the dot, and it sits there on explored non-frontier floor for the rest of
    # the run (measured: 1.0 parked on (1,6) from step 7 to the end) — the degenerate corner, not
    # the interception this page is for. That corner is 08's job.
    grid([CORRIDOR, (2, 2, 2, 2), (3, 5, 1, 4), (2, 2, 20, 20), (3, 5, 18, 22)]),
    lkp=(1, 5), obs_path=[(1, 12)] * 20, r_sense=2, r_comm=4, prereveal=[CORRIDOR],
    regions={"left door (c2)": rect(1, 5, 1, 4), "right door (c20)": rect(1, 5, 18, 22),
             "corridor": rect(1, 1, 5, 17)})

# 6 ─ the freeze happens with the target door ALREADY inside the comm blob.
MAPS["06_target_in_comm_blob"] = Scenario(
    "06_target_in_comm_blob", "Frozen with the target door already inside the comm radius",
    "Contact is lost close to a door, so at the very first step that door is already inside "
    "`seen`. This is the most common real configuration, not an edge case: `lkp` sits on the comm "
    "boundary BY DEFINITION, because being outside it is what losing contact means.",
    # The door mouth is at (1,9): comm (radius 4 from the observer at (1,5)) reaches it, the SENSOR
    # (radius 2) does not, and `lkp` at (1,10) is one cell further out — the only geometry in which
    # the target is in the blob at the freeze step AND the dot does not have to cross the blob to
    # get there (dist_h = 1, so it arrives on the freeze step itself). With the door BETWEEN the
    # observer and lkp the dot's route runs through comm and `cur_seen` kills it, which is
    # interception — that is 05's page, and this one silently duplicated it.
    grid([CORRIDOR, (2, 2, 9, 9), (3, 5, 7, 11), (2, 2, 20, 20), (3, 5, 18, 22)]),
    # lkp is FOUR cells past the door, not one. The route lkp->door must not itself run through the
    # comm ball, or the dot is killed in transit by `cur_seen` and the page silently becomes 05's
    # interception test instead. From (1,12) the path is (1,11),(1,10),(1,9): only the last cell,
    # the TARGET, is inside comm — which is exactly the configuration this page is about.
    lkp=(1, 12),
    # park long enough to show the near door HOLDING its mass while inside comm, then walk in and
    # look through it, so the second half of the page shows the release following the frontier.
    obs_path=[(1, 5)] * 12 + [(1, 6), (1, 7), (1, 8), (1, 9), (2, 9), (3, 9)] + [(3, 9)] * 4,
    r_sense=2, r_comm=4, prereveal=[CORRIDOR],
    # rects MUST include the corridor row: the frontier node IS the corridor cell in front of the
    # neck, and a rect starting at row 2 reads a flat 0.0000 while the whole belief sits above it.
    regions={"near door (c7-11)": rect(1, 5, 7, 11), "far door (c20)": rect(1, 5, 18, 22),
             "corridor between": rect(1, 1, 12, 17)})

# 7 ─ exhaustion: one door, a closed pocket, and the observer walks all of it.
MAPS["07_zone_exhausted"] = Scenario(
    "07_zone_exhausted", "Exhaustion — one zone walked to the end, another door still open",
    "Two doors. The observer sweeps the LEFT pocket completely and finds nothing; the RIGHT door "
    "is never approached. The rule: push while there is unexplored ground adjacent to keep pushing "
    "into, and once there is not, DELETE — the freed mass belongs to the target that is still "
    "open. Nothing may be left resting on ground the observer is standing in.",
    grid([CORRIDOR, (2, 2, 6, 6), (3, 5, 4, 8), (2, 2, 20, 20), (3, 5, 18, 22)]),
    # lkp sits 3 hops from the left door and 5 from the observer: the dot LANDS before the
    # observer gets there, which is the transition this page exists to show. Put lkp further and
    # the hypothesis is killed in transit instead, and the page tests nothing.
    lkp=(1, 9),
    # The observer HOLDS for four steps so the dot lands on the left door first. Without the hold
    # its comm radius reaches the door at t=1 and kills the hypothesis in transit, and the page
    # ends up testing interception instead of exhaustion.
    obs_path=[(1, 1)] * 4 + [(1, c) for c in range(2, 7)]
             + [(2, 6), (3, 6), (3, 4), (4, 4), (5, 4), (5, 6), (5, 8), (4, 8), (3, 8)]
             + [(3, 8)] * 6,
    r_sense=2, r_comm=4, prereveal=[CORRIDOR],
    regions={"left pocket": rect(2, 5, 4, 8), "right door (c20)": rect(1, 5, 18, 22),
             "corridor": rect(1, 1, 1, 24)})

# 8 ─ the degenerate corner: the observer clears the ONLY zone and the map has nothing else left.
MAPS["08_no_target_left"] = Scenario(
    "08_no_target_left", "Nothing left anywhere — the honest answer is not 'under my feet'",
    "One door, one pocket, and the observer walks all of it. Afterwards there is no frontier "
    "anywhere on the map and no hypothesis alive. Every node is either explored or proven empty, "
    "so any probability the model still displays is somewhere it has already been shown he is "
    "not. This page exists to make that corner visible rather than let it hide in a real episode.",
    grid([CORRIDOR, (2, 2, 6, 6), (3, 5, 4, 8)]),
    lkp=(1, 16),
    # He then WALKS BACK OUT and down to the dead end at (1,1). Without that last leg he stays
    # parked in the pocket, the corridor west of the door never enters comm range, and the mass
    # simply diffuses there for the rest of the run — correct behaviour, but the page could not
    # show the corner it exists for, because the belief was never actually cornered.
    obs_path=[(1, c) for c in range(24, 5, -1)] + [(2, 6), (3, 6), (3, 4), (4, 4), (5, 4),
                                                   (5, 6), (5, 8), (4, 8), (3, 8), (3, 7), (3, 6),
                                                   (2, 6)] + [(1, c) for c in range(6, 0, -1)]
             + [(1, 1)] * 4,
    r_sense=2, r_comm=4, prereveal=[CORRIDOR],
    regions={"pocket": rect(2, 5, 4, 8), "corridor": rect(1, 1, 1, 24),
             "dead end (c1-5)": rect(1, 1, 1, 5)})


# --------------------------------------------------------------------------------------- render
CELL = 22


def svg(world: World, p: torch.Tensor, seen: torch.Tensor, fr: torch.Tensor,
        dots: list[int], obs: int, truth: int | None) -> str:
    W, H = world.W * CELL, world.H * CELL
    pm = float(p.max()) if float(p.max()) > 0 else 1.0
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" class="grid">',
           f'<rect width="{W}" height="{H}" fill="#0d1117"/>']
    for n in range(world.N):
        r, c = divmod(n, world.W)
        x, y = c * CELL, r * CELL
        if not world.free[n]:
            fill = "#000000"
        elif not world.known[n]:
            fill = "#161b22"                                   # unknown
        elif seen[n]:
            fill = "#123047"                                   # proven empty this step
        else:
            fill = "#2b3138"                                   # known free
        out.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" fill="{fill}" '
                   f'stroke="#0d1117" stroke-width="1"/>')
        v = float(p[n])
        if v > 1e-6:
            a = min(1.0, 0.15 + 0.85 * (v / pm) ** 0.5)
            s = CELL - 4
            out.append(f'<rect x="{x + 2}" y="{y + 2}" width="{s}" height="{s}" rx="3" '
                       f'fill="#ff4d4d" fill-opacity="{a:.3f}"/>')
        if fr[n]:
            out.append(f'<rect x="{x + 1}" y="{y + 1}" width="{CELL - 2}" height="{CELL - 2}" '
                       f'fill="none" stroke="#f0c674" stroke-width="1.5"/>')
    for n in dots:
        r, c = divmod(n, world.W)
        out.append(f'<circle cx="{c * CELL + CELL / 2}" cy="{r * CELL + CELL / 2}" r="{CELL / 3}" '
                   f'fill="none" stroke="#ffd866" stroke-width="2.5"/>')
    if truth is not None:
        r, c = divmod(truth, world.W)
        out.append(f'<circle cx="{c * CELL + CELL / 2}" cy="{r * CELL + CELL / 2}" r="{CELL / 4}" '
                   f'fill="#4dd0e1"/>')
    r, c = divmod(obs, world.W)
    out.append(f'<circle cx="{c * CELL + CELL / 2}" cy="{r * CELL + CELL / 2}" r="{CELL / 3.2}" '
               f'fill="#7bd88f"/>')
    out.append("</svg>")
    return "".join(out)


PAGE_CSS = """
:root{color-scheme:dark light}
body{margin:0;padding:24px;font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
     background:#0d1117;color:#c9d1d9}
h1{font-size:19px;margin:0 0 4px}
p.why{max-width:70ch;color:#8b949e;margin:0 0 18px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:0 0 18px;color:#8b949e;font-size:12px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;
          vertical-align:-1px}
.steps{display:flex;flex-wrap:wrap;gap:16px}
.step{background:#11161d;border:1px solid #21262d;border-radius:8px;padding:10px}
.step h3{margin:0 0 6px;font-size:12px;font-weight:600;color:#8b949e}
.grid{display:block;border-radius:4px}
table{border-collapse:collapse;font-size:11px;margin-top:6px}
td{padding:1px 7px 1px 0;white-space:nowrap}
td.k{color:#8b949e}
.bad{color:#ff6b6b;font-weight:700}
a{color:#58a6ff}
@media(prefers-color-scheme:light){body{background:#fff;color:#24292f}
 .step{background:#f6f8fa;border-color:#d0d7de}}
:root[data-theme=light] body{background:#fff;color:#24292f}
:root[data-theme=dark] body{background:#0d1117;color:#c9d1d9}
"""

LEGEND = ('<div class="legend">'
          '<span><i style="background:#ff4d4d"></i>p (teammate belief)</span>'
          '<span><i style="background:#123047"></i>seen — proven empty this step</span>'
          '<span><i style="background:#161b22"></i>unknown</span>'
          '<span><i style="border:1.5px solid #f0c674;background:none"></i>frontier</span>'
          '<span><i style="border:2px solid #ffd866;border-radius:50%;background:none"></i>'
          'transit dot</span>'
          '<span><i style="background:#7bd88f;border-radius:50%"></i>observer</span>'
          '</div>')


# ------------------------------------------------------------------------------------------ run

def ascii_frame(world: World, p: torch.Tensor, seen: torch.Tensor, fr: torch.Tensor,
                dots: list[int], obs: int, lkp: int) -> list[str]:
    """One step as a picture. Reading a column of these IS the test — a table of numbers hides
    exactly the things that go wrong here (mass appearing where nothing was next to it, a branch
    emptying while it still has frontier, a ring that never departs).

        #  wall            ?  unknown            .  known-free, nothing
        ^  frontier, no mass                     o  observer      L  teammate last-known
        A..J  mass ON a frontier   (A ~0.05 ... J ~1.0)   <- this is where mass belongs
        1..9  mass NOT on a frontier                      <- suspicious unless it is a * dot
        *  transit dot (a hypothesis still walking its BF path)
    A cell inside the observer's comm blob is shown in lowercase when it is a frontier (v) — mass
    on `v` is legitimate ("he is beyond it"); a DIGIT inside the blob is the one thing that is
    always wrong.
    """
    out = []
    for r in range(world.H):
        line = []
        for c in range(world.W):
            n = r * world.W + c
            m = float(p[n])
            if not bool(world.free[n]):
                line.append("#"); continue
            if n == obs:
                line.append("o"); continue
            if n in dots:
                line.append("*"); continue
            if not bool(world.known[n]):
                line.append("?"); continue
            if m >= 0.005:
                lvl = min(9, int(m * 10))
                line.append(chr(ord("A") + lvl) if bool(fr[n]) else str(lvl if lvl else 1))
            elif bool(fr[n]):
                line.append("v" if bool(seen[n]) else "^")
            elif n == lkp:
                line.append("L")
            else:
                line.append("." if not bool(seen[n]) else ",")
        out.append("".join(line))
    return out

def run_scenario(sc: Scenario, out_dir: Path, verbose: bool = False,
                 ascii_seq: bool = False) -> dict:
    world = World(sc.rows)
    lkp = _n(world, sc.lkp)
    obs0 = _n(world, sc.obs_path[0])
    for r0, r1, c0, c1 in sc.prereveal:                # ground both robots have already mapped
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                world.known[r * world.W + c] = True
    world.reveal(lkp, sc.r_sense + 1)
    world.reveal(obs0, sc.r_sense)
    if bool(world.ball(obs0, sc.r_comm)[lkp]):
        print(f"    [!] {sc.name}: lkp is INSIDE the observer's comm ball at t=0 — contact would "
              f"not be lost there, and every hypothesis dies at the freeze step. Move them apart.")

    Kf, Lmax = sc.kf, 128
    live = torch.zeros((1, world.N))
    acc = torch.zeros((1, world.N))
    seeded = torch.zeros((1, Kf), dtype=torch.bool)
    front_node = weight = dist_h = path = None
    frames, rows_report = [], []
    sum_min, sum_max = 1e9, -1e9

    for t, rc in enumerate(sc.obs_path):
        obs = _n(world, rc)
        world.reveal(obs, sc.r_sense)
        ef, fr, u = world.edge_free(), world.frontier(), world.utility(r_sense=sc.r_sense)
        seen = (world.ball(obs, sc.r_comm) & world.known).unsqueeze(0)

        if t == 0:                                        # contact is lost HERE → freeze
            dist, par = bf(world, lkp)
            # the SAME opening set advance_pathfront uses: the current frontier.
            front_node, weight, dist_h, path = freeze_hypotheses(
                lkp_node=torch.tensor([lkp]), opening=fr, dist=dist, parent=par,
                utility=u, node_xy=world.xy, node_spacing=1.0, Kf=Kf, Lmax=Lmax)

        live, acc, seeded, p, alive, tviz, weight = advance_pathfront(
            live, acc, seeded, front_node=front_node, weight=weight, dist_h=dist_h, path=path,
            step=torch.tensor([t]), frontier_node=fr, utility=u, edge_free=ef,
            nbr_idx=world.nbr, seen=seen,
            just_frozen=torch.tensor([t == 0]))

        tot = float(p.sum())
        sum_min, sum_max = min(sum_min, tot), max(sum_max, tot)
        dots = tviz[0].nonzero().flatten().tolist()
        # count HYPOTHESES, not distinct nodes: several dots share a node while their routes
        # still overlap, and "1 dot" for three live hypotheses reads like a bug that is not there.
        n_travel = int(((front_node >= 0) & (torch.tensor([[t]]) < dist_h) & (~seeded)).sum())
        truth = _n(world, sc.truth[min(t, len(sc.truth) - 1)]) if sc.truth else None
        stats = {
            "Σp": f"{tot:.4f}",
            "on frontiers": f"{float((p[0] * fr[0].float()).sum()):.4f}",
            "in `seen`": f"{float((p[0] * seen[0].float()).sum()):.4f}",
            # the number that decides the argument: probability on ground that is inside the comm
            # radius AND is not an opening. There is no reading under which that is legitimate.
            "seen & not frontier": f"{float((p[0] * seen[0].float() * (~fr[0]).float()).sum()):.4f}",
            "still travelling": str(n_travel),
            "live / acc": f"{float(live.sum()):.3f} / {float(acc.sum()):.3f}",
            # where the peak is, and whether it is on an opening. A peak on a NON-frontier node
            # inside the observed region is the signature of freed mass being dumped somewhere
            # instead of spread over the ground it was already on.
            # peak location + whether the observer can hear that cell. A peak the radio would
            # already have caught is the single clearest sign the belief is claiming something
            # contradicted by evidence.
            "peak in comm": f"{'YES' if bool(seen[0, int(p[0].argmax())]) else 'no'}",
            "peak": f"{float(p[0].max()):.4f}@"
                    f"{divmod(int(p[0].argmax()), world.W)}"
                    f"{'F' if bool(fr[0, int(p[0].argmax())]) else '-'}",
        }
        for rname, cells in sc.regions.items():
            idx = [_n(world, c) for c in cells]
            stats[rname] = f"{float(p[0, idx].sum()):.4f}"
        rows_report.append((t, stats))
        if ascii_seq:
            print(f"--- {sc.name}  t={t}  obs={rc}  Sp={tot:.4f}  onFrontier={stats['on frontiers']}"
                  f"  seen&notFrontier={stats['seen & not frontier']}  travelling={n_travel}")
            for ln in ascii_frame(world, p[0], seen[0], fr[0], dots, obs, lkp):
                print("    " + ln)
        frames.append((t, svg(world, p[0], seen[0], fr[0], dots, obs, truth), stats))

    # ---- page
    stamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [f"<h1>{html.escape(sc.title)}</h1>",
             f"<p class='why' style='color:#6e7681'>rebuilt {stamp}</p>",
             f'<p class="why">{html.escape(sc.why)}</p>', LEGEND,
             '<p><a href="index.html">&larr; all scenarios</a></p>', '<div class="steps">']
    for t, s, stats in frames:
        rowsh = "".join(
            f'<tr><td class="k">{html.escape(k)}</td>'
            f'<td class="{"bad" if k == "Σp" and abs(float(v) - 1) > 5e-4 else ""}">'
            f'{html.escape(v)}</td></tr>'
            for k, v in stats.items())
        parts.append(f'<div class="step"><h3>step {t}</h3>{s}<table>{rowsh}</table></div>')
    parts.append("</div>")
    (out_dir / f"{sc.name}.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>{html.escape(sc.title)}</title>"
        f"<style>{PAGE_CSS}</style>" + "".join(parts), encoding="utf-8")

    if verbose:
        keys = list(rows_report[0][1].keys())
        print("    step  " + "  ".join(f"{k:>14s}" for k in keys))
        for t, st in rows_report:
            print(f"    {t:4d}  " + "  ".join(f"{st[k]:>14s}" for k in keys))
    return {"name": sc.name, "title": sc.title, "steps": len(frames),
            "sum_min": sum_min, "sum_max": sum_max}


def main() -> None:
    stamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=_REPO / "runs" / "pf_scenarios")
    ap.add_argument("--tag", default=None,
                    help="OPTIONAL subfolder. Omit it (the default) and the pages are rewritten in "
                         "place, so the URL never changes and there is only ever one thing to "
                         "look at. Pass a tag only when a side-by-side is explicitly wanted, e.g. "
                         "--impl /tmp/ref_pf.py --tag reference.")
    ap.add_argument("--only", default=None, help="substring of the scenario name")
    ap.add_argument("--ascii", action="store_true",
                    help="print every step as a picture (see ascii_frame for the legend)")
    ap.add_argument("--verbose", action="store_true", help="print the per-step table")
    ap.add_argument("--impl", type=Path, default=None,
                    help="path to an alternative copy of the belief module")
    args = ap.parse_args()
    if args.impl is not None:
        global advance_pathfront, freeze_hypotheses
        advance_pathfront, freeze_hypotheses = load_impl(args.impl)
    if args.tag:
        args.out = args.out / args.tag
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    for name, sc in MAPS.items():
        if args.only and args.only not in name:
            continue
        res = run_scenario(sc, args.out, args.verbose, args.ascii)
        ok = abs(res["sum_min"] - 1) < 5e-4 and abs(res["sum_max"] - 1) < 5e-4
        res["ok"] = ok
        results.append(res)
        print(f"[pf] {name:24s} steps={res['steps']:3d}  "
              f"Σp∈[{res['sum_min']:.6f},{res['sum_max']:.6f}]  {'ok' if ok else 'MASS LEAK'}")

    # The index lists every page PRESENT in the folder, not just the ones this invocation
    # rebuilt: `--only 07` used to overwrite it with a single entry and silently unlink the other
    # six pages, which still existed on disk.
    by_name = {r["name"]: r for r in results}
    for name, sc in MAPS.items():
        if name in by_name or not (args.out / f"{name}.html").exists():
            continue
        by_name[name] = {"name": name, "title": sc.title, "steps": None,
                         "sum_min": None, "sum_max": None, "ok": True}
    results = [by_name[n] for n in MAPS if n in by_name]
    # Pages written by OTHER tools into the same folder are picked up too, so a real-map run
    # can be read next to the hand-scripted scenarios instead of living at a URL of its own.
    # Listed last, by filename, with no stats of ours to report.
    for f in sorted(args.out.glob("*.html")):
        if f.stem == "index" or f.stem in by_name:
            continue
        m = re.search(r"<title>(.*?)</title>", f.read_text(encoding="utf-8"))
        results.append({"name": f.stem, "title": html.unescape(m.group(1)) if m else f.stem,
                        "steps": None, "sum_min": None, "sum_max": None, "ok": True})

    items = "".join(
        f'<li><a href="{r["name"]}.html">{html.escape(r["title"])}</a> '
        + (f'<span style="color:#8b949e">— {r["steps"]} steps, '
           f'Σp∈[{r["sum_min"]:.4f},{r["sum_max"]:.4f}]</span>'
           if r["steps"] is not None
           else '<span style="color:#8b949e">— not rebuilt this run</span>')
        + ("" if r["ok"] else " <b class=bad>MASS LEAK</b>") + "</li>"
        for r in results)
    (args.out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>pathfront scenarios</title>"
        f"<style>{PAGE_CSS}</style><h1>pathfront belief — scripted scenarios"
        + (f" <span style='color:#8b949e'>[{html.escape(args.tag)}]</span>" if args.tag else "")
        + f"</h1><p class='why' style='color:#6e7681'>rebuilt {stamp}</p>"
        '<p class="why">One page per case. The observer\'s moves are scripted, so everything '
        "shown is a property of the belief model alone. Sensing radius reveals the map; the "
        "larger COMM radius is what produces <code>seen</code>.</p>"
        f"<ul>{items}</ul>", encoding="utf-8")
    print(f"\n[pf] {args.out}/index.html")


if __name__ == "__main__":
    main()
