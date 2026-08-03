"""Teammate-position belief — "pathfront" two-phase hypothesis model, KNOWN-graph only.

Alternative to the uniform expanding-ball (`env/teammate_belief.py`). Models the teammate as having
headed toward one of the known openings, weighted by that opening's attractiveness (utility over
distance). The belief lives ONLY on KNOWN-free nodes — never on unknown ones — because the policy
cannot read unknown nodes and a deployment map has no fixed size for them.

Two phases (hypotheses frozen at the moment contact is lost, ONE PER OPENING NODE):
  1. TRANSIT — the RING. Each hypothesis is a point travelling the geodesic (BF) lkp→F_i, one hop
     per step, carrying w_i (Σ w_i = 1). Together they are the probability moving outward along the
     Bellman-Ford tree toward the openings, weighted by distance and by the opening's utility —
     i.e. a simulation of where the teammate walked.
  2. ABSORBING DIFFUSION — on arrival, w_i is injected on F_i, then evolves on the KNOWN-free graph
     by DIFFUSE one hop inward + OPENING ABSORPTION (each opening locks β = min(gain·utility, β_max)
     of its live mass). When an opening is later consumed, its accumulator RELEASES and is spread
     over the openings the look revealed, so the belief follows the frontier outward. Σ p = 1.

Diffusion and absorption are LINEAR, so all hypotheses share ONE live field and ONE accumulator;
overlaps sum automatically. Weights matter only at injection time.

State held by the caller (Explorer). Frozen per set (cap Kf hypotheses, path cap Lmax):
  front_node [P,Kf] long · weight [P,Kf] · dist_h [P,Kf] long · path [P,Kf,Lmax] long (FRONT→lkp).
Live per set: live [P,N] · acc [P,N] · seeded [P,Kf] bool (hypothesis already injected).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def freeze_hypotheses(
    lkp_node: torch.Tensor,        # [P] long — teammate last-known node (BF source)
    opening: torch.Tensor,         # [P, N] bool — LIVE openings (the same set advance_pathfront uses)
    dist: torch.Tensor,            # [P, N] float — BF cost (px) from lkp
    parent: torch.Tensor,          # [P, N] long — BF parent toward lkp (-1 root/none)
    utility: torch.Tensor,         # [P, N] float ∈[0,1]
    node_xy: torch.Tensor,         # [N, 2] float — node pixel coords (spacing between hypotheses)
    node_spacing: float,           # NR px per hop (distance→hops)
    Kf: int = 6,
    Lmax: int = 256,
    min_sep_hops: float = 3.0,     # two hypotheses must be at least this many hops apart
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Freeze up to Kf hypotheses on OPENING NODES, spaced apart. Returns (front_node, weight, dist_h, path).

    weight_i = utility(F_i) / distance(lkp→F_i), normalised Σ=1 over the kept openings; a uniform
    fallback when every kept opening has zero utility, so mass can never vanish here.

    SPACED GREEDY, NOT CONNECTED COMPONENTS. This used to label the frontier's components and send
    one point toward each component's centroid. That was built when a "frontier" needed 4 unknown
    8-neighbours, which kept components small; at the current threshold (1) the frontier of a real
    map is one long connected arc, so 12-17 nodes collapsed into a single dot and there was no ring
    left to move — measured on test/hybrid #1: 20 frontier nodes → 4 components → 1 surviving dot,
    against 9 live openings.
    Plain top-Kf by utility/distance is not the answer either: distance is in the DENOMINATOR, so
    it picks the Kf nearest openings, they all sit 1-2 hops from lkp, they arrive on the step they
    are born and no ring ever departs. Picking greedily by the same score while forbidding a new
    hypothesis within `min_sep_hops` of one already taken keeps the ranking rule the model has
    always used and spreads the hypotheses over genuinely distinct doors — which is what the
    clustering was for, in 12 lines instead of 60 and without depending on frontier connectivity.
    """
    P, N = opening.shape
    dev = opening.device
    reachable = opening & torch.isfinite(dist)
    # utility / distance, the model's own attractiveness rule.
    score = utility.clamp(min=0.0) / dist.clamp(min=float(node_spacing))
    sep2 = float(min_sep_hops * node_spacing) ** 2
    xy = node_xy.view(1, N, 2)                                    # shared across rows: same graph
    front_node = torch.full((P, Kf), -1, dtype=torch.long, device=dev)
    avail = reachable.clone()
    for k in range(Kf):
        # `-1` on everything unavailable, so any available opening outranks it — including one whose
        # utility is exactly 0, which is still a real opening and must not be skipped for a wall.
        sc = torch.where(avail, score, torch.full_like(score, -1.0))
        best = sc.argmax(dim=1)                                   # [P]
        ok = torch.gather(sc, 1, best.unsqueeze(1)).squeeze(1) >= 0.0
        front_node[:, k] = torch.where(ok, best, torch.full_like(best, -1))
        # index `node_xy` DIRECTLY: gather along dim 1 would require the index to match `xy`'s
        # batch dim, which is 1 here (one graph for every row). That only ever worked because the
        # bench runs with P=1; with P>1 it raises "Size does not match at dimension 0".
        d2 = (xy - node_xy[best].view(P, 1, 2)).pow(2).sum(-1)    # [P, N]
        avail = avail & ((d2 > sep2) | ~ok.unsqueeze(1))
    used = front_node >= 0
    w = torch.gather(score, 1, front_node.clamp(min=0)) * used.float()
    wsum = w.sum(dim=1, keepdim=True)
    n_used = used.float().sum(dim=1, keepdim=True)
    weight = torch.where(wsum > eps, w / wsum.clamp(min=eps),
                         used.float() / n_used.clamp(min=1.0))

    # path FRONT→lkp by following parent pointers (parent points toward the BF source = lkp). dist_h is
    # the hop index at which the path reaches lkp → path[0..dist_h] are all valid nodes, so the transit
    # lookup path[dist_h - s] is always a real node.
    path = torch.full((P, Kf, Lmax), -1, dtype=torch.long, device=dev)
    used = front_node >= 0
    dist_h = torch.full((P, Kf), Lmax - 1, dtype=torch.long, device=dev)
    cur = front_node.clamp(min=0)                                 # [P, Kf]
    path[:, :, 0] = torch.where(used, cur, torch.full_like(cur, -1))
    at_lkp = used & (cur == lkp_node.unsqueeze(1))
    dist_h = torch.where(at_lkp, torch.zeros_like(dist_h), dist_h)
    for t in range(1, Lmax):
        par = torch.gather(parent, 1, cur)                       # [P, Kf] parent of cur toward lkp
        stop = (~used) | (cur == lkp_node.unsqueeze(1)) | (par < 0)
        nxt = torch.where(stop, cur, par)
        path[:, :, t] = torch.where(used & ~stop, nxt, torch.full_like(nxt, -1))
        now = used & ~at_lkp & (nxt == lkp_node.unsqueeze(1))     # first arrival at lkp
        dist_h = torch.where(now, torch.full_like(dist_h, t), dist_h)
        at_lkp = at_lkp | (nxt == lkp_node.unsqueeze(1))
        cur = nxt.clamp(min=0)
        if bool((~stop).sum().item() == 0):
            break
    return front_node, weight, dist_h, path


@torch.no_grad()
def advance_pathfront(
    live: torch.Tensor,            # [P, N] float — LIVE (mobile) mass on known-free nodes (mutated)
    acc: torch.Tensor,             # [P, N] float — ACCUMULATED (locked) mass on frontier nodes (mutated)
    seeded: torch.Tensor,          # [P, Kf] bool — hypothesis already injected into `live`
    *,
    front_node: torch.Tensor,      # [P, Kf] long (-1 unused)
    weight: torch.Tensor,          # [P, Kf] float
    dist_h: torch.Tensor,          # [P, Kf] long — arrival step
    path: torch.Tensor,            # [P, Kf, Lmax] long — FRONT→lkp sequence
    step: torch.Tensor,            # [P] long — hops since last contact (s)
    frontier_node: torch.Tensor,   # [P, N] bool — CURRENT frontier nodes (live, recomputed each step)
    utility: torch.Tensor,         # [P, N] float ∈[0,1] — CURRENT node utility (drives absorb rate β)
    edge_free: torch.Tensor,       # [P, N, K] bool — KNOWN-free graph (diffusion edges, no unknown)
    nbr_idx: torch.Tensor,         # [N, K] long
    absorb_gain: float = 1.0,      # β_F = min(absorb_gain · utility(F), beta_max)
    beta_max: float = 0.9,
    diffuse_lambda: float = 0.5,   # fraction of a node's live mass that hops out per step
    push_iters: int = 16,          # hop budget for the spread walk. `seen` is the comm radius, so
                                   # the region a freed unit may have to cross is a few nodes deep.
    push_floor: float = 0.02,      # utility floor in the push weights, so fully-explored floor is
                                   # CROSSABLE — at 0 a corridor of utility 0 has no legal exit
    seen: torch.Tensor | None = None,   # [P, N] bool — nodes the observer has checked empty THIS step
    just_frozen: torch.Tensor | None = None,  # [P] bool — hypothesis (re)frozen THIS call, s already
                                               # backdated by the caller to look one hop travelled; skip
                                               # the current-point negative-evidence test for these rows
                                               # ONLY on this call (trivially near lkp, not new evidence)
    gate_eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One step of the absorbing diffusion. Returns (live, acc, seeded, p, alive, transit_viz). Σ p = 1,
    all mass on known nodes. p = live + acc + transit-point-masses (hypotheses still travelling to their
    frontier). `transit_viz` [P,N] marks EVERY still-travelling hypothesis's current point with 1.0
    (uniform, NOT probability-weighted) so the viz can show all Kf dots depart even when the utility/
    distance weights concentrate mass on one cluster."""
    P, N = live.shape
    Kf = front_node.shape[1]
    K = nbr_idx.shape[1]
    dev = live.device
    Lmax = path.shape[2]
    used = front_node >= 0                                        # [P, Kf]
    s = step.view(P, 1)                                          # [P, 1]

    # An OPENING is simply a CURRENT frontier node. There is no separate "has the sensor looked
    # through it" test, and there must not be: a frontier the sensor has swept but whose far side is
    # still unknown is exactly the node the belief has to move ONTO. A freshly revealed frontier is
    # always inside the sensor footprint that revealed it, so gating on that footprint makes the set
    # of legal targets systematically exclude the new frontier and the belief can never follow the
    # frontier outward. Measured on the bench when such a gate was tried: 02_push_behind t=9, the
    # observer two cells short of an unentered door, the room behind it still unknown, and the
    # door's 0.510 teleported 14 cells to the far door; 03_bifurcation t=22, the sweep created
    # (6,4)/(6,5) in the left fork and both were excluded, so the fork lost its whole 0.4826 to its
    # sibling while it still had unexplored ground. A node stops being an anchor when it stops being
    # a FRONTIER — i.e. when the ground behind it is actually known. That is the same test
    # `tgt_seen`, `keep_acc` and `stray_m` all use, and one test is the point.
    opening = frontier_node

    # Transit-point node for hypotheses still travelling (one hop/step lkp→front_node). Computed once,
    # up front, so BOTH the negative-evidence kill test below AND the viz/mass placement at step 5 agree
    # on where each dot actually is THIS step — otherwise a dot can be tested against `seen` at its
    # frontier target while being drawn somewhere else entirely.
    s_idx = (dist_h - step.view(P, 1)).clamp(0, Lmax - 1)
    pnode = torch.gather(path, 2, s_idx.unsqueeze(-1)).squeeze(-1)                # [P, Kf]
    pnode = torch.where(pnode >= 0, pnode, front_node)

    nbr_flat_p = nbr_idx.clamp(min=0).view(1, N * K).expand(P, -1)
    u_nbr_p = torch.gather(utility, 1, nbr_flat_p).view(P, N, K).clamp(min=0.0)

    # Per-neighbour push weight: how attractive the step onto that neighbour is. `push_floor` keeps
    # fully-explored floor CROSSABLE — at 0 a corridor of utility 0 would have no legal exit.
    w_push = edge_free.float() * (u_nbr_p + push_floor)
    Z_push = w_push.sum(-1)
    ok_push = Z_push > gate_eps

    def _spread_to_openings(mass):
        """SPREAD `mass` over the openings around it — at most `push_iters` hops, ∝ utility.

        A local, bounded random walk on the known-free graph: at each hop the moving mass splits
        over its neighbours ∝ (utility + push_floor), and whatever lands on an opening stays there.
        That is the rule as stated: spread onto the frontiers next to where the mass was, and push
        only while there is unexplored ground adjacent.

        NOT a global rule, deliberately. Two global variants were tried and both fail in mirror
        image, measured on test/hybrid #1: a watershed to the NEAREST opening gave the two BEST
        openings on one wall exactly 0.000 (u0.76 and u0.84 at t=26) and let a u0.12 corner outrank
        a u0.43 door on complex; an attractiveness field ∝ utility·λ^dist did the reverse, stranding
        the two openings sitting DIRECTLY above the mass at 0.000 (u0.27, u0.24 at t=27) while a
        node four hops away took 0.473 — the visible "reappears out of nowhere" jump. Both answer
        "where is the best opening on the map"; the question is "where can he have gone from here".

        Returns (landed, signed_leftover, boxed). `landed + boxed == mass` node-for-node; the signed
        leftover is only ever legal to SUM — reading it as a distribution is what once turned a
        0.249 field into [−44.6, +11.4].
        """
        resid = mass * (~opening).float()
        landed = mass * opening.float()
        boxed = torch.zeros_like(mass)
        for _ in range(push_iters):
            boxed = boxed + resid * (~ok_push).float()            # boxed in → give up on it
            resid = resid * ok_push.float()
            moved = torch.zeros_like(live).scatter_add(
                1, nbr_flat_p,
                ((resid / Z_push.clamp(min=gate_eps)).unsqueeze(-1) * w_push).reshape(P, N * K))
            landed = landed + moved * opening.float()
            resid = moved * (~opening).float()
            if not bool((resid > gate_eps).any()):
                break
        boxed = boxed + resid                                     # hop budget exhausted mid-walk
        return landed, mass - landed, boxed

    # 0) NEGATIVE EVIDENCE: the observer checked `seen` nodes this step and the teammate was NOT there
    #    (else comm would have collapsed the belief). Zero the belief mass on those nodes, then PUSH the
    #    removed mass back onto the SURVIVING hypotheses (∝ their expected utility) — NOT straight onto
    #    the destination frontiers. Each survivor takes its share on its CURRENT form: one still
    #    TRAVELLING → its carried weight grows (brighter dot + bigger bloom on arrival); one already
    #    ARRIVED/expanding → its frontier's live field grows (stronger expansion). Mass-conserving.
    if seen is not None:
        seen_f = seen.float()
        keep = 1.0 - seen_f
        # `acc` ON A CURRENT FRONTIER IS NOT A CLAIM ABOUT THAT NODE. It is the mass that has
        # already gone THROUGH the opening, into unknown space this model has no nodes for — the
        # frontier node is only its proxy. `seen` is the COMM radius, several times the sensing
        # radius, so walking up to an opening puts it inside `seen` long before anything behind it
        # has been looked at; erasing there deletes "he is somewhere beyond that door" on evidence
        # that only says "he is not standing in the doorway". Measured on 02_push_behind: at the
        # step the observer entered, mass on frontiers went 0.9888 -> 0.0000 and 0.977 of the
        # belief was pushed BACKWARDS into the corridor it had already cleared.
        # The immunity is also what makes "exhaust the zone before deleting it" true rather than
        # aspirational: this mass becomes ordinary evidence again only once the opening is actually
        # looked through, at which point it is no longer a frontier, section 2 releases it, and
        # section 4 carries it one hop at a time toward whatever opening replaced it.
        keep_acc = 1.0 - seen_f * (~opening).float()
        removed = live * seen_f + acc * (1.0 - keep_acc)          # mass sitting on checked nodes
        live = live * keep
        acc = acc * keep_acc

        # PUSH WHILE THERE IS UNEXPLORED GROUND ADJACENT TO PUSH INTO; ONLY THEN DELETE.
        # The mass just removed is not evidence that he is gone, it is evidence that he is FURTHER
        # ON, so it is carried rather than deleted. It may land ONLY ON AN OPENING: letting it stop
        # on the first node that merely happened to be out of comm range put probability in the
        # middle of explored rooms, and funnelled a whole zone's freed mass onto one arbitrary
        # interior cell of a neighbouring room.
        # The walk crosses only nodes the observer has JUST proven empty and stops at the first
        # opening it reaches, so it cannot enter unexplored ground and cannot overtake a transit
        # dot. Whatever reaches no opening at all falls through to the spread below.
        landed, removed, _ = _spread_to_openings(removed)
        live = live + landed

        # NO OPENING REACHABLE — the zone is spent, and THAT is when the mass is deleted: it falls
        # through to the redistribution below, which hands it to the hypotheses that are still
        # alive. An earlier version parked it here instead, spread over the ground it had just been
        # cleared from; that keeps probability on explored floor inside the comm radius for the
        # rest of the episode (measured on 02_push_behind: 0.5705 frozen there from step 21 while
        # the one door still open in the map stayed at 0.4243), which is precisely the claim the
        # radio has already refuted. Nothing is added here on purpose — `removed` simply survives
        # into `dM`.
        # A hypothesis is falsified either when its TARGET frontier is already checked, OR the moment
        # its CURRENT transit point enters `seen` — the observer's own advance can check the exact
        # corridor cell a dot is passing through well before it ever reaches the frontier. Without the
        # latter test the dot visibly TRAVELS THROUGH checked-empty ground instead of vanishing there.
        # Excludes rows frozen THIS call: their point is trivially near lkp (comm only just broke
        # there) — testing it would kill every hypothesis at birth regardless of which way it departs.
        jf = (torch.zeros((P, 1), dtype=torch.bool, device=dev) if just_frozen is None
              else just_frozen.view(P, 1))
        # ...but ONLY once the target has stopped being an opening. `seen` is the COMM radius, so
        # standing anywhere in the corridor puts the doorway of an unentered room inside it, and
        # killing on that alone deletes "he is beyond that door" on evidence that only says "he is
        # not standing IN the door" — the same mistake `keep_acc`, `ef_out` and `stray_m` already
        # refuse to make. Measured on 04_two_zones: the left cluster's representative sat 4 hops
        # from the observer at t=0, so the hypothesis died at the freeze step and the ring never
        # split at all — the whole containment case was untestable. The real falsification is the
        # door being LOOKED THROUGH: it is then no longer a frontier, and this fires.
        tgt_front = torch.gather(opening, 1, front_node.clamp(min=0))       # [P, Kf]
        tgt_seen = used & torch.gather(seen, 1, front_node.clamp(min=0)) & (~tgt_front)
        cur_seen = used & (s < dist_h) & (~jf) & torch.gather(seen, 1, pnode.clamp(min=0))
        falsified = tgt_seen | cur_seen
        kill = falsified & (~seeded)
        dM = removed.sum(dim=1) + (kill.float() * weight).sum(dim=1)              # [P] mass to re-place
        seeded = seeded | kill
        # split dM over the survivors ∝ expected utility at their representative frontier — falling
        # back to a UNIFORM split over the same (real, still-alive) survivors when their utility-weight
        # is degenerate (e.g. every survivor's target frontier currently has ~0 utility). Utility being
        # low is not the same as having no survivor: the freed mass must stay with the hypotheses that
        # are actually still alive, never skip past them onto some unrelated frontier elsewhere on the
        # map just because this local direction scores low on utility right now.
        u_h = torch.gather(utility, 1, front_node.clamp(min=0))                   # [P, Kf]
        moving = used & (s < dist_h) & (~seeded) & (~falsified)                   # survivors travelling
        # An ARRIVED survivor is only a legal deposit point while its target is STILL AN OPENING.
        # `front_node` is frozen at comm-break; once the observer walks through that door it is
        # ordinary explored floor, and scattering onto it teleports mass there from anywhere on
        # the map — measured on 08_no_target_left at t=36: 1.0 sitting on the dead end at (1,1)
        # jumped in one step to (1,6) and the pocket behind it, five hops, ground the observer had
        # already cleared. "He is further in" is not this hypothesis's job any more; it is the
        # live field's, via section 2's release and the push, one hop per step like everything else.
        arrived = used & (s >= dist_h) & (~falsified) & tgt_front                 # target still an opening
        survivor = moving | arrived
        n_surv = survivor.float().sum(dim=1, keepdim=True)                        # [P, 1]
        U = (u_h * survivor.float()).sum(dim=1, keepdim=True)                     # [P, 1]
        share = torch.where(
            U > gate_eps, u_h * survivor.float() / U.clamp(min=gate_eps),
            torch.where(n_surv > gate_eps, survivor.float() / n_surv.clamp(min=gate_eps),
                        torch.zeros_like(u_h)))
        add_h = dM.unsqueeze(1) * share                                           # [P, Kf] per-hyp mass
        weight = weight + torch.where(moving, add_h, torch.zeros_like(add_h))     # travelling → weight
        live = live.scatter_add(1, front_node.clamp(min=0),
                                torch.where(arrived, add_h, torch.zeros_like(add_h)))  # arrived → live
        # fallback (truly NO surviving hypothesis left, not merely a low-utility one): put dM back
        # EXACTLY where it just evaporated from — nodes `removed` just zeroed (real live/acc positions),
        # plus the CURRENT transit point `pnode` of whichever hypothesis was `kill`ed this call (where
        # its dot actually IS right now) — NOT its target `front_node` (the distant frontier it was still
        # travelling toward). Scattering onto `front_node` teleports mass onto a node that may have zero
        # neighbouring precursor (a real, confirmed regression: belief appeared on nodes with no adjacent
        # probability the step before). `pnode` is real-world-consistent — it moved one hop/step same as
        # every other hypothesis, so mass re-entering there is adjacent to where it was a moment ago.
        # The ordinary DIFFUSE (step 4) / ABSORB (step 3) machinery below already runs every step and
        # will carry this back out one hop at a time, same pacing every other hypothesis obeys — so it
        # can never skip a node or show probability on ground the diffusion hasn't actually reached yet.
        # `origin` sums to exactly dM by construction (removed.sum() + kill·weight.sum()), so it's a
        # valid non-empty distribution whenever dM>0 — no separate empty-map fallback needed.
        no_surv = (n_surv <= gate_eps).squeeze(1)                                 # [P]
        if bool(no_surv.any()):
            # `removed` is what the push could NOT place, and the push subtracts what it moved
            # node by node, so individual entries are negative wherever mass left. Feeding that
            # straight in as a distribution is what blew this up: the negatives nearly cancel the
            # positives, the sum lands next to zero, and dividing by it — guarded only by
            # `gate_eps` — turned a field of 0.249 into one spanning -44.6 to +11.4 in a single
            # step (Σp stayed 1.0000 the whole time, because the two halves cancel).
            origin = removed.clamp(min=0.0) + torch.zeros_like(removed).scatter_add(
                1, pnode.clamp(min=0), kill.float() * weight)
            Zorig = origin.sum(dim=1, keepdim=True)
            add_back = torch.where(
                Zorig > 1e-6,
                (dM * no_surv.float()).unsqueeze(1) * origin / Zorig.clamp(min=1e-6),
                torch.zeros_like(origin))
            # `origin` can itself BE the node that just got proven seen (the degenerate case: the last
            # hypothesis's only real mass was sitting exactly where negative evidence just fired — e.g.
            # an arrived hypothesis's frontier gets checked the same step every rival hypothesis also
            # dies). Depositing straight back there would re-violate "never show mass on proven-empty
            # ground" the instant it lands (confirmed via full-episode scan: node showing live mass with
            # seen=True, sourced from exactly this line). Relay that portion ONE hop onto its unseen
            # neighbours instead — same single-hop-per-step rule as section 4's diffuse, never a forced
            # multi-hop escape.
            if seen is not None:
                stuck = add_back * seen_f
                add_back = add_back * keep
                nbr_flat_o = nbr_idx.clamp(min=0).view(1, N * K).expand(P, -1)
                nbr_seen_o = torch.gather(seen, 1, nbr_flat_o).view(P, N, K)
                route_o = edge_free & (~nbr_seen_o)
                deg_o = route_o.float().sum(-1)
                has_o = deg_o > gate_eps
                share_o = torch.where(
                    has_o.unsqueeze(-1),
                    stuck.unsqueeze(-1) * route_o.float() / deg_o.clamp(min=1.0).unsqueeze(-1),
                    torch.zeros((P, N, K), dtype=torch.float32, device=dev))
                relay = torch.zeros_like(live)
                relay.scatter_add_(1, nbr_flat_o, share_o.reshape(P, N * K))
                add_back = add_back + relay + stuck * (~has_o).float()  # fully-boxed-in node: stays put
            live = live + add_back

    # 1) INJECT newly-arrived hypotheses: add w_i onto frontier node F_i (once).
    arrived = used & (s >= dist_h)
    newly = arrived & ~seeded                                    # [P, Kf]
    if newly.any():
        # ...ONTO AN OPENING, not onto whatever node was an opening at comm-break. `front_node` is
        # frozen, the hypothesis takes dist_h steps to reach it, and by then the observer has often
        # walked past and looked through it. Injecting there and leaving it puts a lump of belief on
        # ordinary explored floor, where section 3 cannot lock it and only the isotropic diffuse can
        # move it — measured on test/complex #1, t=25-35: 1.0000 of the belief smeared over 20 → 58
        # non-opening nodes with utility 0.00, with ten live openings on the map holding nothing.
        # The push carries it to the openings that replaced the frozen one, which is what "he is
        # further in" means. It is a no-op when the frozen target is still an opening.
        inj = torch.zeros_like(live).scatter_add(1, front_node.clamp(min=0), newly.float() * weight)
        inj_land, _, inj_boxed = _spread_to_openings(inj)
        live = live + inj_land + inj_boxed
        seeded = seeded | newly

    # 2) RELEASE stale accumulators: a node that is no longer an opening (looked through / explored
    #    beyond) unlocks its accumulated mass — and that mass is PUSHED to the openings that replaced
    #    it, not dropped into the live field to diffuse isotropically. The claim being released is
    #    "he is beyond this door"; the door has now been looked through, so the claim becomes "he is
    #    beyond whatever the look revealed", which is a specific set of nodes and not a fog around
    #    the old door. Whatever reaches no opening at all stays exactly where it was (`boxed`), so
    #    this is mass-conserving and never invents a destination.
    stale = (acc > gate_eps) & ~opening                    # [P, N]
    rel = torch.where(stale, acc, torch.zeros_like(acc))
    acc = torch.where(stale, torch.zeros_like(acc), acc)
    rel_land, _, rel_boxed = _spread_to_openings(rel)
    live = live + rel_land + rel_boxed

    # 3) ABSORB at current frontiers: lock β_F = min(gain·utility, β_max) of live mass (β = utility).
    beta = (absorb_gain * utility).clamp(0.0, beta_max) * opening.float()   # [P, N]
    lock = beta * live
    acc = acc + lock
    live = live - lock

    # 4) DIFFUSE live mass one hop INWARD over the known-free graph (mass-conserving push). Never routes
    #    onto a neighbour the observer has already proven empty THIS step (`seen`) — a node simply keeps
    #    its share instead of being forced onto checked ground. The negative evidence in section 0 is
    #    what actually zeroes a node's mass, and it only fires once THAT node itself becomes seen — which
    #    happens one hop later, as diffusion naturally carries the mass forward. This keeps propagation
    #    at exactly one hop per real step in both directions: outward via this masked diffuse, and
    #    "backward" (off proven-empty ground) via section 0 next call — never a multi-hop jump forced
    #    within a single step.
    #
    #    The out-mask and in-mask are NOT the same tensor, on purpose. `edge_free` is a pairwise
    #    (direction-independent) predicate, so the original unmasked code could reuse one `ef` for both
    #    the per-source degree/share AND the gather-based inflow sum — a node's own row doubled as its
    #    reciprocal edge's validity. `seen` breaks that: "is my neighbour seen" (out-mask, gates what a
    #    SENDER will route to) and "am I seen" (in-mask, gates what a RECEIVER may accept) are different
    #    single-node predicates, not a shared pairwise one. Reusing the out-mask for inflow gathering
    #    checks the wrong endpoint — it blocks "my neighbour is seen" instead of "I am seen", which lets
    #    mass keep flowing INTO seen nodes from any not-yet-seen sender (a confirmed leak: verified via
    #    instrumentation, small live mass accumulating step over step on nodes marked `seen`, sourced
    #    purely from ordinary diffuse inflow). Two separate masks fixes it while staying exactly
    #    conservative (per-edge: out-mask at the sender's slot and in-mask at the receiver's reciprocal
    #    slot are provably equal, since both reduce to `edge_free[i,k] & ~seen[receiver]`).
    ef_out = edge_free.float()                                    # [P, N, K] — sender-side: don't route to a seen neighbour
    if seen is not None:
        nbr_flat_d = nbr_idx.clamp(min=0).view(1, N * K).expand(P, -1)
        nbr_seen_d = torch.gather(seen, 1, nbr_flat_d).view(P, N, K)
        nbr_front_d = torch.gather(opening, 1, nbr_flat_d).view(P, N, K)
        # ...but a FRONTIER inside `seen` is still a legal destination, for the same reason: mass
        # arriving there means "beyond that opening", not "standing on that node". Without this,
        # the mass released by a consumed frontier cannot reach the opening that replaced it
        # whenever that replacement is inside the observer's comm radius — which it always is,
        # because the observer is standing right there — so it can only go backwards.
        ef_out = ef_out * (~nbr_seen_d | nbr_front_d).float()
    deg_raw = ef_out.sum(-1)                                      # [P, N]
    has_nbr = deg_raw > 0
    share = diffuse_lambda * live / deg_raw.clamp(min=1.0)       # amount sent to EACH known-free, unseen-target nbr
    nbr = nbr_idx.clamp(min=0).view(1, N * K).expand(P, -1)
    ef_in = edge_free.float()                                     # [P, N, K] — receiver-side: don't accept mass if I'm seen
    if seen is not None:
        ef_in = ef_in * (~seen | opening).float().unsqueeze(-1)
    inflow = (torch.gather(share, 1, nbr).view(P, N, K) * ef_in).sum(-1)          # [P, N]
    outflow = torch.where(has_nbr, diffuse_lambda * live, torch.zeros_like(live))
    live = live - outflow + inflow

    # 4b) NEGATIVE EVIDENCE, ONE LAST TIME — the same erase as section 0, applied at the END.
    #     Section 0 clears the checked nodes, and then everything after it is free to put mass
    #     straight back onto them: the survivor redistribution deposits on `front_node`, which is
    #     the frontier frozen at comm-break and is ordinary explored floor by the time the observer
    #     walks in; the no-survivor fallback restores onto `origin`; the diffuse step lands on a
    #     node that was a frontier when it left and is not one any more. All three put probability
    #     inside the comm radius on ground that is not an opening — visible in 03_bifurcation,
    #     steps 19-23, as the red cells with no yellow outline right under the observer.
    #     A frontier keeps its mass (it stands for the unknown behind it). Everything else inside
    #     the blob is cleared, and what is cleared is handed to the field that survives, in
    #     proportion to it — the same "spread over what is still standing" the rest of the model
    #     uses. Nothing new is decided here; the rule is simply applied where it was being undone.
    if seen is not None:
        stray_m = (seen & (~opening)).float()
        stray = (live + acc) * stray_m
        live = live - live * stray_m
        acc = acc - acc * stray_m
        # SAME PUSH AS SECTION 0, not a global spread. Handing this mass to the whole surviving
        # field in proportion drains the branch the observer is currently walking into its sibling
        # a little every step — measured on 03_bifurcation, the left fork went 0.4871 → 0.3220 →
        # 0.1919 while the right fork rose to 0.7676, before the left one was anywhere near
        # exhausted. Pushed to the NEAREST opening instead, it follows the observer to the bottom
        # of the branch he is clearing, and only moves to the sibling once that branch has no
        # opening left at all — which is what the step-25 behaviour already showed is right.
        got, stray, _ = _spread_to_openings(stray)
        live = live + got
        # What still found no opening is spread over the field that survives, in proportion to it.
        # Weights are the NON-NEGATIVE field and the guard is a real threshold, not `gate_eps`:
        # scaling `live` and `acc` separately by amt/Σ(live+acc) turns any small cancellation
        # between them into an enormous multiplier (measured: live/acc = -40.011 / +41.011 at one
        # step, with Σp still reading 1.0000 because the two cancel). All of it goes into `live`;
        # section 3 will lock whatever part of it belongs on a frontier at the next step anyway.
        w_k = (live + acc).clamp(min=0.0)
        Zk = w_k.sum(dim=1, keepdim=True)
        amt = stray.sum(dim=1, keepdim=True)
        ok_k = Zk > 1e-6
        live = live + torch.where(ok_k, amt * w_k / Zk.clamp(min=1e-6), torch.zeros_like(live))
        # nothing at all left standing: the travelling dots are the only thing that still means
        # something, so the mass rides them; if there are none either, it stays where it was.
        # ONLY the dots that are genuinely STILL TRAVELLING may be ridden. `weight` is kept for
        # every slot, including hypotheses that were injected long ago or killed, and section 1
        # only ever injects `arrived & ~seeded` — so weight added to a seeded slot is written into
        # storage nothing reads again. That is a straight mass leak, and it is not hypothetical:
        # on 08_no_target_left, at the step the observer finally corners the last of the belief,
        # Σp went 1.0000 → 0.0000 in one step because the only slot left was seeded 12 steps back.
        alive_h = used & (s < dist_h) & (~seeded)                                 # [P, Kf]
        w_ride = weight * alive_h.float()
        w_tot = w_ride.sum(dim=1, keepdim=True)
        ride = (~ok_k) & (w_tot > gate_eps)
        weight = weight + torch.where(ride, amt * w_ride / w_tot.clamp(min=gate_eps),
                                      torch.zeros_like(weight))
        # NOTHING LEFT ANYWHERE — no opening on the map, no surviving field, no travelling dot.
        # The mass may NOT be put back where it came from: that node is inside the comm radius and
        # is not an opening, which is the one state the radio has already refuted (08's t=32-40 had
        # 1.0 pinned on the dead end with the observer standing on it, `seen & not frontier` =
        # 1.0000). Spread it uniformly over the KNOWN-FREE ground the observer cannot currently
        # hear — the only ground left that the evidence does not contradict. From there the
        # ordinary machinery takes over: the diffuse carries it one hop per step and section 3
        # locks whatever reaches an opening, so it re-concentrates on the frontiers by itself as
        # soon as the map grows one again. Nothing special-cased, just a legal starting point.
        known_free = edge_free.any(-1)                                # [P, N] node has a known-free edge
        dest = known_free & (~seen)
        n_dest = dest.float().sum(dim=1, keepdim=True)
        spread = (~ok_k) & (~ride) & (n_dest > gate_eps)
        live = live + torch.where(spread, amt * dest.float() / n_dest.clamp(min=1.0),
                                  torch.zeros_like(live))
        # The terminal corner: every known node is inside the comm radius AND there is not one
        # opening left on the map. No position at all is allowed by the evidence, so the mass is
        # DELETED — Σp goes to 0 and `alive` reports False below. That is "there is no belief",
        # which is the honest answer; it is NOT "he is nowhere", and the caller must not read a
        # zero field as a claim about the map.

    # 5) TRANSIT point masses for hypotheses still travelling (s < dist_h). Killed hypotheses (target
    #    or current point already checked empty) are marked seeded → excluded so they neither render
    #    nor inject. `pnode` computed up top so this agrees exactly with the kill test in step 0.
    transit = used & (s < dist_h) & (~seeded)
    p = live + acc
    transit_viz = torch.zeros((P, N), dtype=torch.float32, device=dev)   # uniform dots (viz only)
    if transit.any():
        add = torch.zeros((P, N), dtype=torch.float32, device=dev)
        add.scatter_add_(1, pnode.clamp(min=0), transit.float() * weight)
        p = p + add
        # uniform-brightness marker for EVERY travelling dot (scatter 1.0, overlaps stay 1.0).
        transit_viz.scatter_(1, pnode.clamp(min=0), transit.float())

    # A row with hypotheses but an EMPTY field is not alive: the terminal corner in section 4b
    # deletes the mass rather than claim refuted ground, and a Σp=0 field handed to the policy as
    # if it were a belief is worse than no belief at all.
    alive = used.any(dim=1) & (p.sum(dim=1) > gate_eps)          # [P]
    # `weight` is MUTABLE state now: negative-evidence redistribution grows the carried weight of
    # travelling survivors, so it must be persisted by the caller (else the extra mass evaporates next
    # step → Σp leaks). Returned last for backward-compatible unpacking.
    return live, acc, seeded, p, alive, transit_viz, weight
