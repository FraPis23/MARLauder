"""Teammate-position belief — "pathfront" two-phase hypothesis model, KNOWN-graph only.

Alternative to the uniform expanding-ball (`env/teammate_belief.py`). Models the teammate as having
committed to head toward ONE of the known frontiers, weighted by that frontier's attractiveness
(utility/distance). Crucially the belief lives ONLY on KNOWN-free nodes — never on unknown nodes —
because the policy cannot read unknown nodes and a deployment map has no fixed size for them.

Two phases (hypotheses frozen at the moment contact is lost, one per frontier cluster):
  1. TRANSIT — a point travels the geodesic (BF) lkp→frontier_i, one hop/step, carrying w_i (Σ w_i=1).
  2. ABSORBING DIFFUSION — on arrival, w_i is injected on frontier node F_i, then evolves on the
     KNOWN-free graph by: DIFFUSE one hop inward (mass-conserving) + FRONTIER ABSORPTION — each
     frontier node locks a fraction β_F = min(gain·utility(F), β_max) of its live mass into an
     accumulator (β = utility: high-utility frontier → teammate likely pushed out & far → fast lock;
     dead-end frontier → mass lingers in the known interior). When a frontier is later EXPLORED (no
     longer a frontier), its accumulator RELEASES back into the live field → flows to the new outer
     frontier: the belief "chases" the frontier outward as the map grows. Σ p = 1, all on known nodes.

Because diffusion+absorption are LINEAR, all hypotheses are merged into ONE live field + ONE
accumulator field (weights matter only at injection time); overlaps sum automatically.

State held by the caller (Explorer). Frozen per set (cap Kf clusters, path cap Lmax):
  front_node [P,Kf] long · weight [P,Kf] · dist_h [P,Kf] long · path [P,Kf,Lmax] long (FRONT→lkp).
Live per set: live [P,N] · acc [P,N] · seeded [P,Kf] bool (hypothesis already injected).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def cluster_frontiers(
    frontier_node: torch.Tensor,   # [P, N] bool — node is a frontier (known-free touching unknown)
    nbr_idx: torch.Tensor,         # [N, K] long — static lattice neighbours (≥0 padded)
    edge_ok: torch.Tensor,         # [P, N, K] bool — connectivity between frontier nodes (known-free edges)
    max_iters: int = 64,
) -> torch.Tensor:
    """Label frontier nodes by connected component (min node-index in the component). Non-frontier → -1.

    Label propagation: each frontier node keeps the minimum id it can see through frontier↔frontier
    edges; converges to the component's min index. Batched over P.
    """
    P, N = frontier_node.shape
    K = nbr_idx.shape[1]
    dev = frontier_node.device
    BIG = N + 1
    idx = torch.arange(N, device=dev).view(1, N).expand(P, N)
    label = torch.where(frontier_node, idx, torch.full_like(idx, BIG))
    nbr = nbr_idx.clamp(min=0).view(1, N, K).expand(P, N, K)
    fr_nbr = torch.gather(frontier_node, 1, nbr.reshape(P, N * K)).view(P, N, K)
    conn = edge_ok & fr_nbr & frontier_node.unsqueeze(-1)          # both-frontier known-free edge
    for _ in range(max_iters):
        nl = torch.gather(label, 1, nbr.reshape(P, N * K)).view(P, N, K)
        nl = torch.where(conn, nl, torch.full_like(nl, BIG))
        new = torch.minimum(label, nl.amin(dim=-1))
        new = torch.where(frontier_node, new, torch.full_like(new, BIG))
        if torch.equal(new, label):
            break
        label = new
    return torch.where(frontier_node, label, torch.full_like(label, -1))


@torch.no_grad()
def freeze_hypotheses(
    lkp_node: torch.Tensor,        # [P] long — teammate last-known node (BF source)
    frontier_node: torch.Tensor,   # [P, N] bool
    dist: torch.Tensor,            # [P, N] float — BF cost (px) from lkp over the optimistic graph
    parent: torch.Tensor,          # [P, N] long — BF parent toward lkp (-1 root/none)
    utility: torch.Tensor,         # [P, N] float ∈[0,1]
    nbr_idx: torch.Tensor,         # [N, K] long
    edge_ok: torch.Tensor,         # [P, N, K] bool — frontier connectivity (known-free edges)
    node_spacing: float,           # NR px per hop (distance→hops)
    node_xy: torch.Tensor,         # [N, 2] float — node pixel coords (for cluster CENTRE)
    Kf: int = 6,
    Lmax: int = 256,
    eps: float = 1e-6,
    min_util: float = 0.0,         # skip clusters whose representative utility is below this
    min_cluster: int = 0,          # skip clusters made of fewer than this many frontier nodes
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Freeze up to Kf frontier-cluster hypotheses. Returns (front_node, weight, dist_h, path).

    representative = the frontier node nearest the cluster's CENTROID (its geometric centre) — so the
    transit point heads toward the middle of each frontier region, not its closest edge. weight_i =
    utility(rep_i) / distance(lkp→rep_i), normalised Σ=1 over the kept (finite-distance) clusters.
    """
    P, N = frontier_node.shape
    dev = frontier_node.device
    label = cluster_frontiers(frontier_node, nbr_idx, edge_ok)     # [P, N] component id (-1 non-frontier)
    reachable = frontier_node & torch.isfinite(dist)

    front_node = torch.full((P, Kf), -1, dtype=torch.long, device=dev)
    weight = torch.zeros((P, Kf), dtype=torch.float32, device=dev)

    for p in range(P):                                            # freeze is rare (only at comm-break)
        rp = reachable[p]
        if not rp.any():
            continue
        comps = torch.unique(label[p][rp])
        reps, ws = [], []
        for c in comps.tolist():
            m = rp & (label[p] == c)                             # reachable frontier nodes of this cluster
            if min_cluster > 0 and int(m.sum()) < min_cluster:
                continue                                          # sliver, not a real opening
            cxy = node_xy[m].mean(dim=0)                          # cluster CENTROID (px)
            d2c = ((node_xy - cxy) ** 2).sum(-1)                  # dist² to centroid, per node
            d2c = torch.where(m, d2c, torch.full_like(d2c, float("inf")))
            rep = int(d2c.argmin())                              # frontier node nearest the centre
            if min_util > 0.0 and float(utility[p, rep]) < min_util:
                continue                                          # dead opening, no hypothesis
            d_px = float(dist[p, rep])                            # BF cost lkp→rep (for weight/path)
            reps.append(rep)
            ws.append(float(utility[p, rep]) / max(d_px, node_spacing))   # weight = utility / distance
        order = sorted(range(len(reps)), key=lambda k: ws[k], reverse=True)[:Kf]
        wsum = sum(ws[k] for k in order)
        n_kept = len(order)
        for slot, k in enumerate(order):
            front_node[p, slot] = reps[k]
            # uniform fallback when every kept frontier has 0 utility → mass must not vanish.
            weight[p, slot] = (ws[k] / wsum) if wsum > eps else (1.0 / max(1, n_kept))

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
    diffuse_lambda: float = 0.5,   # fraction of a node's live mass that flows out per step
    unlock_frac: float = 0.35,     # fraction of the locked frontier mass returned to the flow each step.
                                   # With a small value the absorption pins essentially everything on the
                                   # openings and nothing ever propagates back into the mapped area.
    flow_back: float = 0.6,       # weight multiplier for DOWNHILL edges (toward less-unexplored ground):
                                   # 0 pins the belief on the frontier, 1 makes the flow fully isotropic
    dead_drain: float = 0.06,      # per-step fraction of probability stranded in fully-explored dead space
                                   # handed back to the redistribution (0 = off, keeps the old behaviour)
    evac_iters: int = 24,          # passes for walking in-view mass out to the boundary of what is seen
    transit_deposit: float = 0.2,  # fraction of a travelling dot's weight dropped into the field each step
                                   # (the spreading wake behind the moving point; 0 = pure point, old model)
    transit_iters: int = 4,        # relay passes for mass that came to rest on a node in view: transit
                                   # through what the observer watches is allowed, resting there is not
    flow_floor: float = 0.02,      # utility floor in the flow weights: 0 would forbid crossing fully
                                   # explored ground at all, stranding the belief on its own frontier
    door_min_util: float = 0.05,   # a node is a "way out" (mass on it = "he is BEYOND here") only while
                                   # this much utility is left to reveal behind it
    prune_util: float = 0.0,       # >0 → mass behind an opening whose utility fell below this is dropped
                                   # into a sink (Σp < 1 = "teammate lost") instead of redistributed
    newly_free: torch.Tensor | None = None,   # [P, N] bool — nodes revealed known-free THIS step: what
                                   # the PUSH hands frontier mass to, before any deletion
    escape_node: torch.Tensor | None = None,  # [P, N] bool — nodes with a generatable edge into the
                                   # unknown. Laxer than frontier_node (which stays the clustering
                                   # criterion) so a thin corridor's leading edge counts too. None → frontier_node.
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

    # Transit-point node for hypotheses still travelling (one hop/step lkp→front_node). Computed once,
    # up front, so BOTH the negative-evidence kill test below AND the viz/mass placement at step 5 agree
    # on where each dot actually is THIS step — otherwise a dot can be tested against `seen` at its
    # frontier target while being drawn somewhere else entirely.
    s_idx = (dist_h - step.view(P, 1)).clamp(0, Lmax - 1)
    pnode = torch.gather(path, 2, s_idx.unsqueeze(-1)).squeeze(-1)                # [P, Kf]
    pnode = torch.where(pnode >= 0, pnode, front_node)

    # A DOOR is a node the teammate could still have walked off the known map through: a generatable
    # edge into the unknown AND enough utility left behind it to be worth anything. The utility gate is
    # what stops a whisker of sealed unknown from making a node permanently exempt below (confirmed:
    # m50/agent1 t=114-118, mass surviving on a utility-0.004 node with the observer standing on it).
    door = (frontier_node if escape_node is None else escape_node) & (utility > door_min_util)
    closed_front = (door & (utility < prune_util)) if prune_util > 0.0 else torch.zeros_like(door)
    open_front = door & (~closed_front)

    # 0a) PUSH — BEFORE ANY DELETION, on purpose. Mass on a door means "he is somewhere in the unknown
    #     beyond it". The moment that unknown becomes known, the mass belongs on the ground just revealed,
    #     so every node holding mass hands it to the nodes revealed THIS step next to it, keeping a share
    #     only while there is still unknown behind it. One hop, onto ground that did not exist a step ago.
    #     Pushing first lets the mass move and THEN be judged: the part that landed inside comm range dies,
    #     the part on the new outer boundary survives and keeps receding. Deleting first would kill the
    #     hypothesis before it ever moved.
    if newly_free is not None:
        nbr_flat_p = nbr_idx.clamp(min=0).view(1, N * K).expand(P, -1)
        nbr_new = torch.gather(newly_free, 1, nbr_flat_p).view(P, N, K) & edge_free
        k_new = nbr_new.float().sum(-1)
        field = live + acc
        pushes = (k_new > 0) & (field > gate_eps)
        denom = (k_new + door.float()).clamp(min=1.0)
        unit = torch.where(pushes, field / denom, torch.zeros_like(field))
        live = live - torch.where(pushes, live, torch.zeros_like(live))
        acc = acc - torch.where(pushes, acc, torch.zeros_like(acc))
        acc = acc + unit * door.float()
        share_p = unit.unsqueeze(-1) * nbr_new.float()
        live = live + torch.zeros_like(live).scatter_add(1, nbr_flat_p, share_p.reshape(P, N * K))

    # 0b) RELEASE stale accumulators — also before the deletion. A node stops being a way out exactly when
    #     the observer finishes revealing what was behind it, i.e. while the observer is still next to it.
    #     Unlocking there and then lets this step's evidence judge that mass: it meant "he is beyond here",
    #     the beyond is now known and empty, so it dies. Releasing afterwards leaves it as ordinary mass in
    #     a corridor the observer walks out of next step — never testable again, and then fed by the
    #     redistribution (m50/agent1 t=119: 0.30 + 0.21 piling up behind the corner on zero-utility nodes).
    stale = (acc > gate_eps) & ~open_front
    live = live + torch.where(stale, acc, torch.zeros_like(acc))
    acc = torch.where(stale, torch.zeros_like(acc), acc)

    # 0) NEGATIVE EVIDENCE, RESOLVED BY EVACUATION — NOT by deletion and NOT by exemption.
    #    `seen` is "comm would have fired if he stood here", the same test that produces real contact, so
    #    a node in it is PROVEN EMPTY: no probability may rest there, not even on an opening the observer
    #    is looking straight at. But proven-empty is not proof he is gone — it is proof he is FURTHER ON.
    #    So that mass is not erased, it WALKS: one hop per pass over the known-free graph, biased by
    #    utility, and it FREEZES the instant it reaches a node the observer cannot see. In a corridor the
    #    observer is sweeping, every intermediate node is visible and only the far end is not, so the mass
    #    ends up exactly there — pushed to the frontier, which is the whole point. Where the escape has
    #    several mouths the shares split across them, so a frontier breaking into n branches breaks the
    #    belief into n branches by itself.
    #
    #    This is bounded and local by construction: mass only ever moves across the visible region, one
    #    hop at a time, and stops at its boundary — it can never stream on to some unrelated frontier
    #    across the map. Mass with no unseen way out at all (fully enclosed by what the observer sees) is
    #    the only mass that is genuinely falsified, and only that goes to the redistribution.
    if seen is not None:
        seen_f = seen.float()
        nbr_e = nbr_idx.clamp(min=0).view(1, N * K).expand(P, -1)
        nbr_seen_e = torch.gather(seen, 1, nbr_e).view(P, N, K)
        w_e = edge_free.float() * (torch.gather(utility, 1, nbr_e).view(P, N, K).clamp(min=0.0) + flow_floor)
        # Where the walk is allowed to STOP: ground the observer cannot see, OR a door — the last known
        # node before the unknown. A door is a legitimate resting place even in plain view, because mass
        # there means "he is past this opening", which seeing the node cannot contradict; it is also the
        # only thing that keeps a swept dead-end corridor from having nowhere to put its belief at all
        # (its far end opens onto unknown, and unknown is not a node we may use). Everything else in view
        # is proven empty and the walk continues through it.
        # A resting place must be plausible, not merely out of sight: either a DOOR (last known node before
        # the unknown) or unseen ground that still has something to reveal. Stopping at the first unseen
        # node regardless is what put belief BEHIND the agent — the instant it turns a corner, the node it
        # just left stops being visible and catches everything walking out (m50 t=118/159, 0.26-0.31 stuck
        # on util-0.02 ground already cleared, unable to leave afterwards because the flow only goes
        # uphill). Dead ground is transit, never a destination.
        u_nbr_e = torch.gather(utility, 1, nbr_e).view(P, N, K)
        stop_ok = (torch.gather(open_front, 1, nbr_e).view(P, N, K)
                   | ((~nbr_seen_e) & (u_nbr_e >= door_min_util)))
        w_out = w_e * stop_ok.float()                             # exits: freeze there
        w_on = w_e * (~stop_ok).float()                           # proven-empty ground: keep walking
        walking = (live + acc) * seen_f                           # everything in view has to leave
        live = live * (1.0 - seen_f)
        acc = acc * (1.0 - seen_f)
        for _ in range(evac_iters):
            if float(walking.sum()) <= gate_eps:
                break
            Zo, Zn = w_out.sum(-1), w_on.sum(-1)
            Z = Zo + Zn
            movable = (Z > gate_eps).float()
            flow_e = walking * movable
            walking = walking - flow_e                            # what stays is boxed in
            unit_e = (flow_e / Z.clamp(min=gate_eps)).unsqueeze(-1)
            live = live + torch.zeros_like(live).scatter_add(
                1, nbr_e, (unit_e * w_out).reshape(P, N * K))      # reached unseen ground → frozen
            walking = walking + torch.zeros_like(live).scatter_add(
                1, nbr_e, (unit_e * w_on).reshape(P, N * K))       # still in view → next pass
        removed = walking                                         # never found a way out → falsified
        walking = None
        # DEAD GROUND DRAINS. Probability that has ended up deep in fully-explored space — its own node and
        # every neighbour below the door threshold, so there is nothing to reveal anywhere around it — is
        # not a place a teammate who is himself exploring would be. It cannot be walked out either (no
        # gradient to follow), so a fraction of it is handed back to the redistribution each step and ends
        # up on the openings that are still plausible. Draining a fraction rather than all of it keeps the
        # travelling dot's wake visible for a few steps (it fades instead of vanishing), while making it
        # impossible for belief to sit behind the agent on cleared ground for the rest of the episode.
        u_nbr_d = torch.gather(utility, 1, nbr_idx.clamp(min=0).view(1, N * K).expand(P, -1)).view(P, N, K)
        u_nbr_d = torch.where(edge_free, u_nbr_d, torch.zeros_like(u_nbr_d)).amax(dim=-1)
        dead_node = (utility < door_min_util) & (u_nbr_d < door_min_util)
        drained = dead_drain * (live + acc) * dead_node.float()
        live = live - dead_drain * live * dead_node.float()
        acc = acc - dead_drain * acc * dead_node.float()
        removed = removed + drained

        # A hypothesis is falsified either when its TARGET frontier is already checked, OR the moment
        # its CURRENT transit point enters `seen` — the observer's own advance can check the exact
        # corridor cell a dot is passing through well before it ever reaches the frontier. Without the
        # latter test the dot visibly TRAVELS THROUGH checked-empty ground instead of vanishing there.
        # Excludes rows frozen THIS call: their point is trivially near lkp (comm only just broke
        # there) — testing it would kill every hypothesis at birth regardless of which way it departs.
        jf = (torch.zeros((P, 1), dtype=torch.bool, device=dev) if just_frozen is None
              else just_frozen.view(P, 1))
        tgt_seen = used & torch.gather(seen & (~open_front), 1, front_node.clamp(min=0))   # [P, Kf]
        cur_seen = used & (s < dist_h) & (~jf) & torch.gather(seen, 1, pnode.clamp(min=0))
        tgt_closed = used & torch.gather(closed_front, 1, front_node.clamp(min=0))
        falsified = tgt_seen | cur_seen | tgt_closed
        kill = falsified & (~seeded)
        dM = removed.sum(dim=1) + (kill.float() * weight).sum(dim=1)              # [P] mass to re-place
        seeded = seeded | kill
        # Re-place dM WHERE THE SURVIVING PROBABILITY ALREADY IS — a rescaling, never a scatter onto a
        # destination frontier (that teleported mass ~90 px in one step onto a node whose neighbours held
        # nothing the step before). Bayes only asks for a renormalisation: P(x | not in seen) ∝ P(x)·1[x∉seen].
        # Only PLAUSIBLE probability may grow: mass on a live way out, or still travelling. Mass stranded on
        # fully-explored ground the observer merely cannot see right now keeps what it has but must not soak
        # up a dead hypothesis — one crumb behind a corner otherwise ends up holding the whole belief on a
        # zero-utility node (m50/agent1 t=120, b=1.00 on (88,296), held for 60 steps). With nothing plausible
        # left, RE-SEED beyond the openings the map still has, ∝ their utility: everything reachable has been
        # checked, so he has to be past one of the remaining ways out.
        moving = used & (s < dist_h) & (~seeded) & (~falsified)                   # survivors travelling
        plaus = open_front.float()
        S_door = ((live + acc) * plaus).sum(dim=1, keepdim=True)                  # [P, 1]
        W_move = (weight * moving.float()).sum(dim=1, keepdim=True)               # [P, 1]
        surv_mass = S_door + W_move
        has_surv = surv_mass > gate_eps
        grow = dM.unsqueeze(1) / surv_mass.clamp(min=gate_eps)
        live = live + torch.where(has_surv, live * plaus * grow, torch.zeros_like(live))
        acc = acc + torch.where(has_surv, acc * plaus * grow, torch.zeros_like(acc))
        weight = weight + torch.where(has_surv & moving, weight * grow, torch.zeros_like(weight))
        u_door = utility * plaus
        U_door = u_door.sum(dim=1, keepdim=True)
        reseed_ok = (~has_surv) & (U_door > gate_eps) & (prune_util <= 0.0)
        live = live + torch.where(reseed_ok, dM.unsqueeze(1) * u_door / U_door.clamp(min=gate_eps),
                                  torch.zeros_like(live))
        has_surv = has_surv | reseed_ok
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
        # With pruning ON this resurrection is skipped: no plausible carrier and no opening means the
        # honest answer is "I no longer know where he is", so dM goes to the sink and Σp drops below 1.
        no_surv = (~has_surv).squeeze(1)                                          # [P]
        if prune_util > 0.0:
            no_surv = torch.zeros_like(no_surv)
        if bool(no_surv.any()):
            # Last resort: no plausible carrier anywhere. Spread dM over the known-free ground the
            # observer canNOT see — never back onto proven-empty nodes, which is the one thing the
            # evacuation above exists to prevent. If even that is empty the map is fully in view, and the
            # mass simply stays where it was boxed in.
            unseen_free = (edge_free.any(-1) & (~seen)).float() if seen is not None else edge_free.any(-1).float()
            Zf = unseen_free.sum(dim=1, keepdim=True)
            ok_f = (Zf > gate_eps) & no_surv.unsqueeze(1)
            live = live + torch.where(ok_f, dM.unsqueeze(1) * unseen_free / Zf.clamp(min=gate_eps),
                                      torch.zeros_like(live))
            live = live + torch.where((~ok_f) & no_surv.unsqueeze(1), removed, torch.zeros_like(live))

    # 1) INJECT newly-arrived hypotheses: add w_i onto frontier node F_i (once).
    arrived = used & (s >= dist_h)
    newly = arrived & ~seeded                                    # [P, Kf]
    if newly.any():
        live = live.scatter_add(1, front_node.clamp(min=0), newly.float() * weight)
        seeded = seeded | newly

    # 2) DRAIN dead openings (prune_util > 0, inert otherwise): an opening onto a sliver nobody would
    #    travel to keeps no belief — if the observer could get nothing out of that pocket, neither could
    #    the teammate. The mass leaves the field for good instead of standing as a confident wrong peak.
    if prune_util > 0.0:
        acc = torch.where(closed_front, torch.zeros_like(acc), acc)

    # 2b) UNLOCK a slice of every accumulator. Locked mass sits on ONE opening forever, so when the map
    #     grows and a different opening becomes the attractive one, the belief cannot follow (m50/agent1
    #     t≥127: pushed north, never able to reach the frontier to the east that the observer then took).
    #     Returning a fraction to the live field each step lets the utility-biased flow carry it there,
    #     while the absorption below immediately re-locks most of it — so the belief still SITS on the
    #     frontier, it just stops being welded to whichever opening it reached first.
    if unlock_frac > 0.0:
        moved_up = unlock_frac * acc
        acc = acc - moved_up
        live = live + moved_up

    # 3) ABSORB at current ways out: lock β_F = min(gain·utility, β_max) of live mass (β = utility).
    beta = (absorb_gain * utility).clamp(0.0, beta_max) * open_front.float()      # [P, N]
    lock = beta * live
    acc = acc + lock
    live = live - lock

    # 4) FLOW one hop, BIASED TOWARD THE UNEXPLORED. The teammate keeps moving, so the belief moves too —
    #    but not isotropically. Plain diffusion spreads mass equally in all directions, which leaks it
    #    backwards into corridor already swept and still crawls to the opening one branch over far too
    #    slowly. Weighting each neighbour by utility makes the same one-hop transport run ALONG the
    #    frontier and TOWARD the unexplored, so a hypothesis spreads over the opening it reached and can
    #    migrate to the next one instead of sitting as a single point.
    #
    #    Never routes onto a node the observer can see (sender-side out-mask) and never lets a seen node
    #    accept inflow (receiver-side in-mask) — different single-node predicates, but per edge both reduce
    #    to edge_free[i,k] & ~seen[receiver], so the step stays exactly mass-conserving.
    nbr = nbr_idx.clamp(min=0).view(1, N * K).expand(P, -1)
    u_nbr_f = torch.gather(utility, 1, nbr).view(P, N, K).clamp(min=0.0)
    # UPHILL ONLY. A neighbour may receive mass only if it has at least as much left to reveal as the node
    # sending it. Without this gate the flow drifts BACKWARDS into corridor the observer already swept and
    # no longer watches — ground with utility 0, nothing to explore, where a teammate who is himself
    # exploring would not be (confirmed: m50 t=118/159, belief re-appearing behind the agent on util-0.00
    # nodes it had already cleared). Uphill keeps the same one-hop transport but points it at the
    # unexplored: along the frontier ribbon (equal utility) and outward (higher), never back into the void.
    # The uphill gate binds only on LIVE ground. A node whose own utility is already below the door
    # threshold is dead ground: mass there must be free to leave in any direction, otherwise a residual
    # sliver that happens to be a local maximum of utility becomes a trap — everything flows in, nothing
    # flows out, and the belief sits behind the agent for the rest of the episode (m50/agent0, (488,376),
    # utility 0.02, holding 0.31 from t≈100 to the end). Weighting still favours whatever has more left to
    # reveal, so escaping mass drifts toward the openings rather than wandering.
    alive_node = (utility >= door_min_util).unsqueeze(-1)
    uphill = (u_nbr_f >= (utility.unsqueeze(-1) - gate_eps)) | (~alive_node)
    # Downhill edges are damped, not forbidden: `flow_back` of the normal weight. With 0 the belief ends
    # up entirely pinned on the openings, which reads as "he can only be past a frontier" — but he is also
    # free to walk back into the mapped area, so a slice of the probability propagates inward every step.
    # The dead-ground drain below still stops that slice from piling up in cleared pockets.
    flow_w = edge_free.float() * torch.where(uphill, torch.ones_like(u_nbr_f),
                                             torch.full_like(u_nbr_f, flow_back)) * (u_nbr_f + flow_floor)
    if seen is not None:
        nbr_seen = torch.gather(seen, 1, nbr).view(P, N, K)
        flow_w = flow_w * (~nbr_seen).float()                    # out-mask: don't send to a seen node
    Wd = flow_w.sum(-1)
    has_nbr = Wd > gate_eps
    share = diffuse_lambda * live.unsqueeze(-1) * flow_w / Wd.clamp(min=gate_eps).unsqueeze(-1)
    inflow = torch.zeros_like(live).scatter_add(1, nbr, share.reshape(P, N * K))   # in-mask implicit: no seen slot
    outflow = torch.where(has_nbr, diffuse_lambda * live, torch.zeros_like(live))
    live = live - outflow + inflow

    # 4c) A TRAVELLING DOT SPREADS ONLY ONTO GROUND JUST REVEALED NEXT TO IT — never continuously. It is
    #     a concentrated hypothesis heading for its frontier and it must stay concentrated until it gets
    #     there, otherwise the field starts blooming before the dots have "exploded" on the frontier, which
    #     is wrong (m50 t=21, t≈195, t≈217). But when the observer uncovers ground right beside the dot,
    #     the teammate could be on that new ground, so the dot hands it a share — same one-slot-each split
    #     the push uses (self + one per revealed neighbour). No revelation next to it → nothing moves.
    if newly_free is not None and bool((used & (s < dist_h) & (~seeded)).any()):
        trav = (used & (s < dist_h) & (~seeded)).float()                          # [P, Kf]
        pn = pnode.clamp(min=0)                                                   # [P, Kf]
        nbr_of_p = nbr_idx.clamp(min=0)[pn]                                       # [P, Kf, K]
        new_at_p = torch.gather(nbr_new.reshape(P, N * K), 1,
                                (pn.unsqueeze(-1) * K + torch.arange(K, device=dev).view(1, 1, K)
                                 ).reshape(P, Kf * K)).view(P, Kf, K)             # [P, Kf, K] bool
        k_at_p = new_at_p.float().sum(-1)                                         # [P, Kf]
        share_t = trav * weight / (k_at_p + 1.0).clamp(min=1.0)                   # per-slot
        give = torch.where(k_at_p > 0, share_t * k_at_p, torch.zeros_like(share_t))
        live = live.scatter_add(1, nbr_of_p.reshape(P, Kf * K),
                                (share_t.unsqueeze(-1) * new_at_p.float()).reshape(P, Kf * K))
        weight = weight - give

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

    alive = used.any(dim=1)                                      # [P]
    # `weight` is MUTABLE state now: negative-evidence redistribution grows the carried weight of
    # travelling survivors, so it must be persisted by the caller (else the extra mass evaporates next
    # step → Σp leaks). Returned last for backward-compatible unpacking.
    return live, acc, seeded, p, alive, transit_viz, weight
