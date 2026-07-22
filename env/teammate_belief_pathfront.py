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
        # transit hypotheses whose TARGET frontier is already checked → falsified: stop them injecting
        # and recycle their carried weight into the redistribution pool.
        tgt_seen = used & torch.gather(seen, 1, front_node.clamp(min=0))          # [P, Kf]
        kill = tgt_seen & (~seeded)
        dM = removed.sum(dim=1) + (kill.float() * weight).sum(dim=1)              # [P] mass to re-place
        seeded = seeded | kill
        # split dM over the survivors ∝ expected utility at their representative frontier.
        u_h = torch.gather(utility, 1, front_node.clamp(min=0))                   # [P, Kf]
        moving = used & (s < dist_h) & (~seeded) & (~tgt_seen)                    # survivors travelling
        arrived = used & (s >= dist_h) & (~tgt_seen)                              # survivors expanding (frontier unseen)
        survivor = moving | arrived
        U = (u_h * survivor.float()).sum(dim=1, keepdim=True)                     # [P, 1]
        share = torch.where(U > gate_eps, u_h * survivor.float() / U.clamp(min=gate_eps),
                            torch.zeros_like(u_h))
        add_h = dM.unsqueeze(1) * share                                           # [P, Kf] per-hyp mass
        weight = weight + torch.where(moving, add_h, torch.zeros_like(add_h))     # travelling → weight
        live = live.scatter_add(1, front_node.clamp(min=0),
                                torch.where(arrived, add_h, torch.zeros_like(add_h)))  # arrived → live
        # fallback (no surviving hypothesis with utility): place dM via a cascade that is guaranteed
        # non-empty, so mass is NEVER lost and Σ stays 1. Priority: unseen frontiers ∝ utility → unseen
        # frontiers uniform → any frontier uniform → any known-free node uniform (always non-empty while
        # the belief is alive).
        no_surv = (U <= gate_eps).squeeze(1)                                      # [P]
        if bool(no_surv.any()):
            unseen_front = (frontier_node & (~seen)).float()
            cands = (utility * unseen_front, unseen_front, frontier_node.float(),
                     edge_free.any(dim=-1).float())
            dest = torch.zeros_like(live)
            filled = torch.zeros((P, 1), dtype=torch.bool, device=dev)
            for c in cands:
                cs = c.sum(dim=1, keepdim=True)
                ok = cs > gate_eps
                take = ok & (~filled)
                dest = torch.where(take, c / cs.clamp(min=gate_eps), dest)
                filled = filled | ok
            live = live + (dM * no_surv.float()).unsqueeze(1) * dest

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

    # 4) DIFFUSE live mass one hop INWARD over the known-free graph (mass-conserving push).
    ef = edge_free.float()                                       # [P, N, K]
    deg_raw = ef.sum(-1)                                         # [P, N]
    has_nbr = deg_raw > 0
    share = diffuse_lambda * live / deg_raw.clamp(min=1.0)       # amount sent to EACH known-free nbr
    nbr = nbr_idx.clamp(min=0).view(1, N * K).expand(P, -1)
    inflow = (torch.gather(share, 1, nbr).view(P, N, K) * ef).sum(-1)             # [P, N]
    outflow = torch.where(has_nbr, diffuse_lambda * live, torch.zeros_like(live))
    live = live - outflow + inflow

    # 5) TRANSIT point masses for hypotheses still travelling (s < dist_h). Killed hypotheses (target
    #    already checked empty) are marked seeded → excluded so they neither render nor inject.
    transit = used & (s < dist_h) & (~seeded)
    p = live + acc
    transit_viz = torch.zeros((P, N), dtype=torch.float32, device=dev)   # uniform dots (viz only)
    if transit.any():
        s_idx = (dist_h - step.view(P, 1)).clamp(0, Lmax - 1)
        pnode = torch.gather(path, 2, s_idx.unsqueeze(-1)).squeeze(-1)            # [P, Kf]
        pnode = torch.where(pnode >= 0, pnode, front_node)
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
