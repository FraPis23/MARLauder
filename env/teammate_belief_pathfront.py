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
            cxy = node_xy[m].mean(dim=0)                          # cluster CENTROID (px)
            d2c = ((node_xy - cxy) ** 2).sum(-1)                  # dist² to centroid, per node
            d2c = torch.where(m, d2c, torch.full_like(d2c, float("inf")))
            rep = int(d2c.argmin())                              # frontier node nearest the centre
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
    diffuse_lambda: float = 0.5,   # fraction of a node's live mass that hops out per step
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

    # 0) NEGATIVE EVIDENCE: the observer checked `seen` nodes this step and the teammate was NOT there
    #    (else comm would have collapsed the belief). Zero the belief mass on those nodes, then PUSH the
    #    removed mass back onto the SURVIVING hypotheses (∝ their expected utility) — NOT straight onto
    #    the destination frontiers. Each survivor takes its share on its CURRENT form: one still
    #    TRAVELLING → its carried weight grows (brighter dot + bigger bloom on arrival); one already
    #    ARRIVED/expanding → its frontier's live field grows (stronger expansion). Mass-conserving.
    if seen is not None:
        seen_f = seen.float()
        keep = 1.0 - seen_f
        removed = (live + acc) * seen_f                          # mass sitting on checked nodes
        live = live * keep
        acc = acc * keep
        # A hypothesis is falsified either when its TARGET frontier is already checked, OR the moment
        # its CURRENT transit point enters `seen` — the observer's own advance can check the exact
        # corridor cell a dot is passing through well before it ever reaches the frontier. Without the
        # latter test the dot visibly TRAVELS THROUGH checked-empty ground instead of vanishing there.
        # Excludes rows frozen THIS call: their point is trivially near lkp (comm only just broke
        # there) — testing it would kill every hypothesis at birth regardless of which way it departs.
        jf = (torch.zeros((P, 1), dtype=torch.bool, device=dev) if just_frozen is None
              else just_frozen.view(P, 1))
        tgt_seen = used & torch.gather(seen, 1, front_node.clamp(min=0))          # [P, Kf]
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
        arrived = used & (s >= dist_h) & (~falsified)                             # survivors expanding (frontier unseen)
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
            origin = removed + torch.zeros_like(removed).scatter_add(
                1, pnode.clamp(min=0), kill.float() * weight)
            add_back = (dM * no_surv.float()).unsqueeze(1) * (
                origin / origin.sum(dim=1, keepdim=True).clamp(min=gate_eps))
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
        live = live.scatter_add(1, front_node.clamp(min=0), newly.float() * weight)
        seeded = seeded | newly

    # 2) RELEASE stale accumulators: a node that is no longer a frontier (explored beyond) unlocks its
    #    accumulated mass back into the live field → it will diffuse on toward the new outer frontier.
    stale = (acc > gate_eps) & ~frontier_node                    # [P, N]
    live = live + torch.where(stale, acc, torch.zeros_like(acc))
    acc = torch.where(stale, torch.zeros_like(acc), acc)

    # 3) ABSORB at current frontiers: lock β_F = min(gain·utility, β_max) of live mass (β = utility).
    beta = (absorb_gain * utility).clamp(0.0, beta_max) * frontier_node.float()   # [P, N]
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
        ef_out = ef_out * (~nbr_seen_d).float()
    deg_raw = ef_out.sum(-1)                                      # [P, N]
    has_nbr = deg_raw > 0
    share = diffuse_lambda * live / deg_raw.clamp(min=1.0)       # amount sent to EACH known-free, unseen-target nbr
    nbr = nbr_idx.clamp(min=0).view(1, N * K).expand(P, -1)
    ef_in = edge_free.float()                                     # [P, N, K] — receiver-side: don't accept mass if I'm seen
    if seen is not None:
        ef_in = ef_in * (~seen).float().unsqueeze(-1)
    inflow = (torch.gather(share, 1, nbr).view(P, N, K) * ef_in).sum(-1)          # [P, N]
    outflow = torch.where(has_nbr, diffuse_lambda * live, torch.zeros_like(live))
    live = live - outflow + inflow

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
