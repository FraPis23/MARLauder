"""GPU-vectorized 8-neighbor lattice graph (replaces TOM NodeManager + QuadTree).

Lattice
-------
Nodes live on a regular grid spaced by `NR` pixels. For canvas (H, W):
    LH = H // NR,  LW = W // NR
    flat node idx  k = li * LW + lj   (li = row in lattice, lj = col)
    world (x, y) = ((lj + 0.5) * NR, (li + 0.5) * NR)

Active mask per env per step:
    node_valid[k] = occupancy at world(k) is FREE  AND  k is reachable from the
                    robot through 8-connected FREE lattice cells.

Edges: between a node and any of its 8 lattice neighbors, valid iff both
endpoints are active AND the straight segment is collision-free in `occupancy`
(sampled at S points). Edges stored per source as `[N_max, K=8]` in canonical
order matching the action space:
    0:NW  1:N  2:NE  3:W  4:E  5:SW  6:S  7:SE

Edge lengths (precomputed [K]):
    axial = NR,  diagonal = NR·√2 — used by `build_guidepost` for true Dijkstra.

Per-step compute is O(LH * LW * K) plus a flood-fill (~diameter conv2d iters)
plus an optional Bellman-Ford for guidepost. All ops batched over n_envs.

Public API
----------
    GraphLattice(canvas, nr=8, sensor_range_px=60, utility_range_px=30,
                 collision_samples=5, device=...)
    graph.build(occupancy, frontier, robot_xy_world, visited_step, current_step)
        -> info dict (see return at end of build)
    graph.build_guidepost(info)
        -> augments info in-place with guidepost_mask, guidepost_target,
           guidepost_path_idx, guidepost_path_valid, guidepost_path_xy
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

UNKNOWN = 0
FREE = 1
OBSTACLE = 2

# Lattice neighbor offsets in (dr, dc) canonical order — also the action space.
NBR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
)
K = 8


class GraphLattice:
    def __init__(
        self,
        canvas: tuple[int, int],
        nr: int = 8,
        sensor_range_px: float = 60.0,
        utility_range_px: int = 30,
        collision_samples: int = 5,
        flood_max_iters: int = 200,
        guidepost_iters: int | None = None,
        guidepost_path_max: int | None = None,
        n_hops: int = 2,
        build_optim_graph: bool = False,
        visit_age_window: int = 16,
        device: str = "cuda:0",
    ) -> None:
        self.device = device
        # When True, build() also emits an "optimistic" edge graph (UNKNOWN treated
        # passable) for teammate-distance BF — see build() step 3b. Only needed M>1.
        self.build_optim_graph = bool(build_optim_graph)
        self.H, self.W = canvas
        self.NR = int(nr)
        self.LH = self.H // self.NR
        self.LW = self.W // self.NR
        self.N_max = self.LH * self.LW
        self.UR = int(utility_range_px)
        self.S = int(collision_samples)
        self.flood_max_iters = int(flood_max_iters)
        self.SR = float(sensor_range_px)
        # feat[3] age horizon (steps): recency window over which a walked node's "freshness"
        # ramps from 0 (just left) back to 1 (cold / re-explorable). Stationary, unlike the
        # old last_visit/current_step.
        self.visit_age_window = max(1, int(visit_age_window))
        # Bellman-Ford iteration upper bound. The OLD bound (Manhattan diameter ~LH+LW) is only
        # valid in open maps: in a corridor/maze the geodesic hop count from agent→target winds
        # far past LH+LW, so BF-from-target never propagated back to curr → parent[curr]<0 →
        # next_hop=curr → the guidepost (and bf_dist used for scoring/progress) silently broke and
        # the global target "disappeared". The true upper bound on a simple path is N_max hops, so
        # use that. bf_from_target early-exits the instant dist stops changing (every-8-iter check),
        # so on open maps the cost is unchanged (~diameter iters); the higher cap only spends extra
        # iters on the long maze geodesics that actually need them. Reachable targets always converge.
        self.guidepost_iters = int(guidepost_iters) if guidepost_iters else int(self.N_max)
        # Path-reconstruction length cap (number of edges on a path). Drives a Python loop +
        # the rendered guidepost line / feat[5] mask (NOT next_hop, which reads parent[curr]
        # directly), so it can stay bounded well below N_max. Doubled vs the old LH+LW+4 so long
        # corridor paths render without truncation, capped so the loop never blows up on big maps.
        self.guidepost_path_max = (
            int(guidepost_path_max) if guidepost_path_max
            else int(min(self.N_max, 2 * (self.LH + self.LW) + 8))
        )

        dev = device
        # Lattice indices (li, lj) per flat node.
        li = torch.arange(self.LH, device=dev).view(self.LH, 1).expand(self.LH, self.LW)
        lj = torch.arange(self.LW, device=dev).view(1, self.LW).expand(self.LH, self.LW)
        self.li_flat = li.reshape(-1)                  # [N_max]
        self.lj_flat = lj.reshape(-1)                  # [N_max]
        # World coordinates of each node, centered in lattice cell.
        nx = (self.lj_flat.float() + 0.5) * self.NR    # [N_max]
        ny = (self.li_flat.float() + 0.5) * self.NR    # [N_max]
        self.node_xy = torch.stack([nx, ny], dim=-1)   # [N_max, 2]  (x=col, y=row)

        # Edge target flat-index (regardless of validity), -1 if out of bounds.
        edge_idx_flat = torch.full((self.N_max, K), -1, dtype=torch.long, device=dev)
        for k, (dr, dc) in enumerate(NBR_OFFSETS):
            tgt_li = self.li_flat + dr
            tgt_lj = self.lj_flat + dc
            in_bounds = (tgt_li >= 0) & (tgt_li < self.LH) & (tgt_lj >= 0) & (tgt_lj < self.LW)
            edge_idx_flat[:, k] = torch.where(
                in_bounds,
                tgt_li * self.LW + tgt_lj,
                torch.full_like(tgt_li, -1),
            )
        self.edge_idx_static = edge_idx_flat            # [N_max, K]

        # Collision-sample offsets along each edge in world coords, S samples.
        # For sample s in [0, S), point = node + (s+1)/(S+1) * (dx, dy)  (endpoints excluded).
        nbr_dx = torch.tensor([dc * self.NR for (dr, dc) in NBR_OFFSETS], dtype=torch.float32, device=dev)
        nbr_dy = torch.tensor([dr * self.NR for (dr, dc) in NBR_OFFSETS], dtype=torch.float32, device=dev)
        t = torch.arange(1, self.S + 1, dtype=torch.float32, device=dev) / (self.S + 1.0)  # [S]
        self.sample_dx = nbr_dx.view(K, 1) * t.view(1, self.S)  # [K, S]
        self.sample_dy = nbr_dy.view(K, 1) * t.view(1, self.S)  # [K, S]
        # Precomputed edge length (Euclidean, in pixels).  Axial = NR, Diagonal = NR·√2.
        self.edge_len = torch.sqrt(nbr_dx ** 2 + nbr_dy ** 2)   # [K]

        # 3x3 dilation kernel for flood fill on lattice.
        self._dilate_k = torch.ones(1, 1, 3, 3, dtype=torch.float32, device=dev)

        # K_INDEX_TABLE[d_li+1, d_lj+1] → K-slot index for analytic direction.
        # Maps (sign(target_li - curr_li), sign(target_lj - curr_lj)) ∈ {-1,0,1}² to
        # the K=8 slot in NBR_OFFSETS. Diagonal (0,0) → -1 (curr == target, no move).
        # NBR_OFFSETS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
        # i.e. (NW,N,NE,W,E,SW,S,SE).
        k_table = torch.full((3, 3), -1, dtype=torch.long, device=dev)
        for k_idx, (dr, dc) in enumerate(NBR_OFFSETS):
            k_table[dr + 1, dc + 1] = k_idx
        self.K_INDEX_TABLE = k_table                              # [3, 3]

        # ----- Phase C: ego-centric subgraph window precomputation ----- #
        # Window side = (2 * n_hops + 3), one extra ring beyond the n_hops receptive
        # field so boundary nodes still aggregate from valid neighbors in the last
        # GAT layer (boundary nodes whose neighbors fall outside the window have
        # those edges masked, degrading them — keep them as padding only).
        self.n_hops = int(n_hops)
        self.window_side = 2 * self.n_hops + 3
        self.window_size = self.window_side * self.window_side
        self.window_radius = self.window_side // 2                # = n_hops + 1
        self.window_local_center = self.window_size // 2          # center index, constant

        # window_offsets[w_local, :] = (d_li, d_lj) in [-radius, +radius], row-major.
        offs_li = torch.arange(-self.window_radius, self.window_radius + 1, device=dev)
        offs_lj = torch.arange(-self.window_radius, self.window_radius + 1, device=dev)
        oli, olj = torch.meshgrid(offs_li, offs_lj, indexing="ij")
        self.window_offsets = torch.stack([oli.reshape(-1), olj.reshape(-1)], dim=-1)   # [W², 2]

        # window_idx_table[k, w_local] = GLOBAL flat idx of node at offset (d_li, d_lj)
        # from cell k. -1 if out of lattice bounds.
        k_li_col = self.li_flat.view(self.N_max, 1)                # [N_max, 1]
        k_lj_col = self.lj_flat.view(self.N_max, 1)
        off_li_row = self.window_offsets[:, 0].view(1, self.window_size)
        off_lj_row = self.window_offsets[:, 1].view(1, self.window_size)
        gli = k_li_col + off_li_row                                # [N_max, W²]
        glj = k_lj_col + off_lj_row
        in_bounds = (gli >= 0) & (gli < self.LH) & (glj >= 0) & (glj < self.LW)
        self.window_idx_table = torch.where(
            in_bounds, gli * self.LW + glj, torch.full_like(gli, -1),
        )                                                          # [N_max, W²]

        # window_local_edge_table[w_local, k] = LOCAL idx (in [0, W²)) of the K-th
        # neighbor of the window cell at w_local. -1 if neighbor outside the window.
        # Independent of the lattice cell k; depends only on window geometry.
        local_edge = torch.full((self.window_size, K), -1, dtype=torch.long, device=dev)
        for w_local in range(self.window_size):
            d_li = int(self.window_offsets[w_local, 0].item())
            d_lj = int(self.window_offsets[w_local, 1].item())
            for k_idx, (dr, dc) in enumerate(NBR_OFFSETS):
                nbr_li = d_li + dr
                nbr_lj = d_lj + dc
                if -self.window_radius <= nbr_li <= self.window_radius and \
                   -self.window_radius <= nbr_lj <= self.window_radius:
                    nbr_local = (nbr_li + self.window_radius) * self.window_side \
                              + (nbr_lj + self.window_radius)
                    local_edge[w_local, k_idx] = nbr_local
        self.window_local_edge_table = local_edge                  # [W², K]

    # ---------------------------------------------------------------------- #
    # main build                                                              #
    # ---------------------------------------------------------------------- #
    def build(
        self,
        occupancy: torch.Tensor,     # uint8 [N, H, W]
        frontier: torch.Tensor,      # bool  [N, H, W]
        robot_xy: torch.Tensor,      # float [N, 2]  (x, y) world
        visited_step: torch.Tensor,  # long  [N, N_max]  step index of last visit (-1 if unvisited)
        current_step: int,
    ) -> dict[str, torch.Tensor]:
        assert occupancy.dim() == 3
        N, H, W = occupancy.shape
        assert (H, W) == (self.H, self.W)
        dev = occupancy.device

        # 1) Sample occupancy at each lattice node → is_free_lattice[N, LH, LW]
        node_x = self.node_xy[:, 0].long().clamp(0, W - 1)   # [N_max]
        node_y = self.node_xy[:, 1].long().clamp(0, H - 1)
        flat = node_y * W + node_x                            # [N_max]
        occ_flat = occupancy.view(N, -1)                      # [N, H*W]
        node_occ = occ_flat[:, flat]                          # [N, N_max]
        is_free_node = node_occ == FREE                       # [N, N_max]
        is_free_lat = is_free_node.view(N, self.LH, self.LW)

        # 2) Edge geometry + collision masks — computed BEFORE the flood so reachability runs
        #    over the SAME strict FREE edge set the BF/guidepost use. (Old build flooded a 3×3
        #    node-center dilation that ignored edge collision → node_valid a SUPERSET of
        #    BF-reachable → targets node-reachable but edge-unreachable stranded the guidepost.)
        edge_idx = self.edge_idx_static.unsqueeze(0).expand(N, -1, -1).contiguous()  # [N, N_max, K]
        nbr_idx_safe = edge_idx.clamp(min=0)                  # [-1 → 0] for gather
        nbr_valid_geom = edge_idx >= 0
        # Collision: sample S points along each edge in world coords.
        nx_all = self.node_xy[:, 0].view(1, self.N_max, 1, 1)    # [1, N_max, 1, 1]
        ny_all = self.node_xy[:, 1].view(1, self.N_max, 1, 1)
        sx = (nx_all + self.sample_dx.view(1, 1, K, self.S)).clamp(0, W - 1).long()
        sy = (ny_all + self.sample_dy.view(1, 1, K, self.S)).clamp(0, H - 1).long()
        sample_lin = (sy * W + sx).expand(N, -1, -1, -1)         # [N, N_max, K, S]
        occ_samples = torch.gather(occ_flat, 1, sample_lin.reshape(N, -1)).view(N, self.N_max, K, self.S)
        # collision_free       — no KNOWN OBSTACLE on segment (UNKNOWN allowed): OPTIMISTIC graph
        #                        (3b), which must reach into unexplored space for teammate BF.
        # collision_free_known — segment is ALL KNOWN-FREE (no UNKNOWN, no OBSTACLE): the FREE
        #                        graph. The permissive mask let two free nodes connect across an
        #                        unmapped gap → guidepost/target BF + GAT routed THROUGH walls /
        #                        unknown. The FREE graph must only follow confirmed-open space.
        collision_free       = (occ_samples != OBSTACLE).all(dim=-1)   # [N, N_max, K]
        collision_free_known = (occ_samples == FREE).all(dim=-1)       # [N, N_max, K]

        # 3) Reachability flood over the strict FREE edge set (both endpoints free + segment
        #    known-free) → node_valid == BF-reachable, by construction.
        is_free_nbr = torch.gather(is_free_node, 1, nbr_idx_safe.view(N, -1)).view(N, self.N_max, K)
        trav = (is_free_node.unsqueeze(-1) & is_free_nbr & nbr_valid_geom & collision_free_known)  # [N,N_max,K]
        trav_f = trav.float()
        rx = robot_xy[:, 0].clamp(0, W - 1).long()
        ry = robot_xy[:, 1].clamp(0, H - 1).long()
        rli = (ry // self.NR).clamp(0, self.LH - 1)
        rlj = (rx // self.NR).clamp(0, self.LW - 1)
        reach = torch.zeros((N, self.N_max), dtype=torch.float32, device=dev)
        reach[torch.arange(N, device=dev), rli * self.LW + rlj] = 1.0
        reach = reach * is_free_node.float()                  # seed must itself be a free node
        for _ in range(self.flood_max_iters):
            nbr_reach = (torch.gather(reach, 1, nbr_idx_safe.view(N, -1)).view(N, self.N_max, K) * trav_f).amax(dim=-1)
            new = torch.maximum(reach, nbr_reach)
            if torch.equal(new, reach):
                break
            reach = new
        node_valid = reach > 0                                # [N, N_max] == BF-reachable

        # 4) Edge validity over the strict FREE set (both endpoints now reachable).
        nbr_node_valid = torch.gather(node_valid, 1, nbr_idx_safe.view(N, -1)).view(N, self.N_max, K)
        endpoints_valid = node_valid.unsqueeze(-1) & nbr_node_valid & nbr_valid_geom
        edge_valid = endpoints_valid & collision_free_known
        edge_idx = torch.where(edge_valid, edge_idx, torch.full_like(edge_idx, -1))

        # 3b) OPTIMISTIC traversable graph (UNKNOWN treated passable) — for teammate-
        # distance BF ONLY. A teammate that has split off usually sits in THIS agent's
        # UNKNOWN region; on the FREE graph above its node is invalid/disconnected and
        # bf_from_target returns +inf for every candidate, silencing the coordination
        # channel exactly when map-sharing is off. Flooding through FREE∪UNKNOWN from the
        # robot gives the honest partial-knowledge geodesic: exact through known-free
        # space, ≈Euclidean through unknown, and +inf only when a KNOWN wall separates
        # them (correct — Euclidean would lie there). collision_free is reused as-is
        # (already != OBSTACLE → edges crossing UNKNOWN already pass it). Gated by
        # build_optim_graph (set True only for M>1) so single-agent pays nothing.
        edge_valid_optim = None
        if self.build_optim_graph:
            # Optimistic node validity = simply "not a KNOWN obstacle" (FREE∪UNKNOWN). NO
            # reachability flood: the teammate-rooted BF over this edge set already returns
            # +inf for nodes its source cannot reach, so connectivity is handled for free.
            # The earlier flood expanded through FREE∪UNKNOWN (≈the whole map early in an
            # episode) → ~diameter iterations every build/agent/step → the dominant cost.
            # collision_free (already != OBSTACLE) still blocks edges crossing KNOWN walls,
            # so optimism cannot leak across seen obstacles.
            is_trav_node = (node_occ != OBSTACLE)                                 # [N, N_max]
            nbr_trav = torch.gather(
                is_trav_node, dim=1, index=nbr_idx_safe.view(N, -1)
            ).view(N, self.N_max, K)
            endpoints_valid_o = is_trav_node.unsqueeze(-1) & nbr_trav & nbr_valid_geom
            edge_valid_optim = endpoints_valid_o & collision_free                 # [N, N_max, K]

        # 4) Utility — FRONTIER-GATED INFORMATION GAIN (v4).
        #
        # v3 tried to seed utility with raw UNKNOWN-area in a sensor-range disk. Three failures:
        #   - disk radius (=SR=60px) ≫ node spacing (NR=16px) → adjacent boxes overlap ~87% →
        #     utility a flat blob, NO local gradient → the actor cannot tell neighbors apart
        #     (loops in rooms);
        #   - the box counted unknown BEHIND walls (no occlusion) → dead-ends scored like real
        #     frontiers;
        #   - utility>0 almost everywhere → top-K candidates / guidepost argmax no longer
        #     frontier-anchored → targets land in dead-ends / interior.
        #
        # v4 keeps the "volume behind the boundary" idea but GATES it by the frontier indicator:
        #   seed = frontier_ribbon(small window, sharp)  ×  (FRONTIER_FLOOR + (1-FLOOR)·volume)
        #   - frontier_ribbon: count of frontier pixels in a tiny ~NR/2 window → sharp, nonzero
        #     ONLY on frontier nodes → restores the local gradient and frontier-anchored targets.
        #     A frontier is a FREE cell adjacent to UNKNOWN; a wall between free and unknown makes
        #     the cell adjacent to OBSTACLE (not a frontier) → the behind-wall occled mass is
        #     rejected at the gate. Fixes the dead-end-scores-high bug.
        #   - volume: fraction of a sensor-range disk that is UNKNOWN (the v3 term), now only a
        #     MULTIPLIER on genuine frontier nodes → "big room behind a small door" still wins,
        #     interior/dead-end nodes stay 0.
        #   - FRONTIER_FLOOR keeps small-room frontiers as valid (utility>0) candidates.
        # Then the same wall-aware edge diffusion spreads the gradient toward the frontier.
        FRONTIER_FLOOR = 0.25
        def _box_count(ii_img: torch.Tensor, r: int) -> torch.Tensor:
            x0 = (node_x - r).clamp(0, W); x1 = (node_x + r + 1).clamp(0, W)
            y0 = (node_y - r).clamp(0, H); y1 = (node_y + r + 1).clamp(0, H)
            def g(ys, xs):
                lin = (ys * (W + 1) + xs).view(1, self.N_max).expand(N, -1)
                return torch.gather(ii_img.view(N, -1), 1, lin)
            return g(y1, x1) - g(y0, x1) - g(y1, x0) + g(y0, x0)        # [N, N_max]
        # Frontier ribbon — sharp, frontier-anchored (true free/unknown boundary, occlusion-safe).
        # r_fr = NR (full node cell + one ring): a border free node's center can sit > NR/2 px
        # from the frontier pixels (they lie in the gap toward unknown) — NR/2 missed them and
        # zeroed genuine border nodes. NR catches the node's own cell and the adjacent boundary.
        ii_fr = F.pad(frontier.float(), (1, 0, 1, 0)).cumsum(-1).cumsum(-2)
        r_fr = max(2, self.NR)
        fr_frac = (_box_count(ii_fr, r_fr) / float(2 * r_fr + 1)).clamp(0, 1)   # ribbon length proxy
        # Revealable volume — fraction of the sensor disk that is UNKNOWN.
        ii_unk = F.pad((occupancy == UNKNOWN).float(), (1, 0, 1, 0)).cumsum(-1).cumsum(-2)
        r_v = max(2, int(self.SR))
        vol = (_box_count(ii_unk, r_v) / (math.pi * float(r_v) ** 2)).clamp(0, 1)
        f_ind = (fr_frac * (FRONTIER_FLOOR + (1.0 - FRONTIER_FLOOR) * vol)) * node_valid.float()
        # Graph diffusion along valid edges only — spreads the frontier gradient inward.
        h_diff = max(1, round(self.UR / self.NR))
        u = f_ind
        ev_f = edge_valid.float()                                  # [N, N_max, K]
        deg = ev_f.sum(-1).clamp(min=1.0)                          # [N, N_max]
        for _ in range(h_diff):
            u_nbr = torch.gather(u, 1, nbr_idx_safe.view(N, -1)).view(N, self.N_max, K)
            u = u + (u_nbr * ev_f).sum(-1) / deg
        # Normalize: h rounds of (self + nbr-mean) bound u by 2^h · max(f_ind) = 2^h.
        utility_norm = (u / float(2 ** h_diff)).clamp(0, 1)
        utility_norm = utility_norm * node_valid.float()

        # 5) curr_idx: O(1) analytic lookup.
        # Node center at (lj+0.5)*NR → nearest column = floor(x/NR).
        # Agents always land on lattice node positions (_spread_starts_graph places
        # them on node_xy; step() moves them to chosen node coords). Floor-divide
        # is therefore exact — no argmin search needed.
        lj_curr = (robot_xy[:, 0] / float(self.NR)).long().clamp(0, self.LW - 1)
        li_curr = (robot_xy[:, 1] / float(self.NR)).long().clamp(0, self.LH - 1)
        curr_idx = li_curr * self.LW + lj_curr                   # [N]

        # 6) Per-node features (8 dims). Channels 0-5 filled here / by Explorer; 6-7 are the
        #    RADAR boundary-summary channels (b_util, b_teammate), written by Explorer.build_radar
        #    onto the geodesic receptive-horizon nodes (0 everywhere else). See build_radar().
        node_feat = torch.zeros((N, self.N_max, 8), dtype=torch.float32, device=dev)
        curr_xy = self.node_xy[curr_idx]                          # [N, 2]
        # EGO-SCALE normalization. The GAT only ever sees the (2·n_hops+3)² window centered on
        # curr, so the relative coords of in-window nodes span at most ±window_radius·NR px.
        # Normalizing by the half-MAP (old: 0.5·max(H,W)) squashed them into ~±0.15 — geometry
        # drowned by the {0,1} binary feats at the input Linear. Normalize by the window
        # half-extent so in-window x_rel/y_rel actually use the full [-1, +1] range.
        win_half = float(self.window_radius * self.NR)
        node_feat[..., 0] = (self.node_xy[:, 0].view(1, -1) - curr_xy[:, 0:1]) / win_half  # x_rel
        node_feat[..., 1] = (self.node_xy[:, 1].view(1, -1) - curr_xy[:, 1:2]) / win_half  # y_rel
        node_feat[..., 2] = utility_norm
        # feat[3] AGE — stationary recency. 0 = just walked (avoid backtrack), ramps to 1 over
        # visit_age_window steps; never-visited nodes start cold at 1 (re-explorable). Replaces
        # the old visited{0,1} + last_visit/current_step pair (the latter was non-stationary —
        # same node's value drifted every step — and redundant with the actor GRU + utility).
        delta = (float(current_step) - visited_step.float()).clamp(min=0.0)
        age = (delta / float(self.visit_age_window)).clamp(0.0, 1.0)
        age = torch.where(visited_step < 0, torch.ones_like(age), age)
        node_feat[..., 3] = age
        # feat[4] teammate BF-proximity potential (M>1, written by Explorer)
        # feat[5] guidepost (filled by build_guidepost if called)
        # feat[6] b_util  — RADAR: geodesically-routed out-of-window utility mass (Explorer)
        # feat[7] b_teammate — RADAR: geodesically-routed out-of-window teammate direction (Explorer)
        node_feat = node_feat * node_valid.unsqueeze(-1).float()

        # 7) curr_nbr gather.
        curr_nbr = torch.gather(
            edge_idx, dim=1,
            index=curr_idx.view(N, 1, 1).expand(-1, 1, K),
        ).squeeze(1)
        curr_nbr_valid = curr_nbr >= 0

        return {
            "node_xy": self.node_xy.unsqueeze(0).expand(N, -1, -1).contiguous(),
            "node_valid": node_valid,
            "node_feat": node_feat,
            "edge_idx": edge_idx,
            "edge_valid": edge_valid,
            "edge_len": self.edge_len,                            # static [K]
            "curr_idx": curr_idx,
            "curr_nbr": curr_nbr,
            "curr_nbr_valid": curr_nbr_valid,
            "utility": utility_norm,
            # util_raw = f_ind: PRE-diffusion frontier seed (nonzero ONLY on true frontier nodes).
            # Experimental: feed to select_target_analytic instead of the diffused utility_norm.
            "util_raw": f_ind,
            # Inspector decomposition of the utility seed (pre-diffusion, per node) — these are
            # references to tensors already computed above, so they add ZERO training compute
            # (only read when store_render_global, and the trace iterates valid nodes only):
            #   util_boundary = fr_frac  → frontier-ribbon length (BOUNDARY pixels)
            #   util_volume   = vol      → fraction of sensor disk that is UNKNOWN (DETECTABLE cells)
            # f_ind = fr_frac · (FRONTIER_FLOOR + (1−FLOOR)·vol); utility_norm diffuses f_ind.
            "util_boundary": fr_frac,
            "util_volume":   vol,
            "edge_valid_optim": edge_valid_optim,   # [N,N_max,K] or None (M==1 / flag off)
        }

    # ---------------------------------------------------------------------- #
    # guidepost (Bellman-Ford on GPU)                                         #
    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def build_guidepost(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Batched Bellman-Ford with edge_len weights from curr_idx.

        Mutates `info` in-place to add:
            guidepost_mask     bool  [N, N_max]      nodes lying on shortest path
            guidepost_target   long  [N]             flat idx of best target (utility argmax over reachable)
            guidepost_path_idx long  [N, P_max]      from-curr-to-target node indices, -1 padded
            guidepost_path_valid bool [N, P_max]
            guidepost_path_xy  float [N, P_max, 2]   world coords for render (NaN-padded)

        Also writes the mask into `info['node_feat'][..., 5]` to populate feat[5].
        """
        edge_idx = info["edge_idx"]        # [N, N_max, K]
        edge_valid = info["edge_valid"]    # [N, N_max, K]
        node_valid = info["node_valid"]    # [N, N_max]
        utility = info["utility"]          # [N, N_max]
        curr_idx = info["curr_idx"]        # [N]
        node_xy = self.node_xy             # [N_max, 2]

        N = edge_idx.shape[0]
        N_max = self.N_max
        K_ = K
        dev = edge_idx.device
        INF = float("inf")

        # dist[N, N_max] init INF, dist[curr] = 0.
        dist = torch.full((N, N_max), INF, dtype=torch.float32, device=dev)
        arange_N = torch.arange(N, device=dev)
        dist[arange_N, curr_idx] = 0.0
        parent = torch.full((N, N_max), -1, dtype=torch.long, device=dev)

        edge_len = self.edge_len.view(1, 1, K_)                          # [1, 1, K]
        edge_idx_safe = edge_idx.clamp(min=0)                             # gather-safe

        # Iterate Bellman-Ford relaxations. Process the relaxation as:
        #   for each node v, look at its 8 neighbors u; candidate dist via u = dist[u] + w(u, v)
        # The static `edge_idx[v, k]` here means "what is the neighbor u? edge from v to that u".
        # Since the graph is undirected with symmetric weights and same K layout,
        # using the candidate `dist[u] + edge_len[k]` is equivalent to relaxing v through u.
        for it in range(self.guidepost_iters):
            # Gather dist of each neighbor.
            nbr_dist = torch.gather(
                dist, dim=1, index=edge_idx_safe.view(N, -1)
            ).view(N, N_max, K_)                                          # [N, N_max, K]
            cand = nbr_dist + edge_len                                    # [N, N_max, K]
            cand = cand.masked_fill(~edge_valid, INF)
            # best per node v over K neighbors
            best_vals, best_k = cand.min(dim=-1)                          # [N, N_max]
            # update where best < dist
            update = best_vals < dist
            dist = torch.where(update, best_vals, dist)
            # parent[v] = edge_idx[v, best_k]
            best_parent = torch.gather(edge_idx, dim=2, index=best_k.unsqueeze(-1)).squeeze(-1)
            parent = torch.where(update, best_parent, parent)
            # cheap early exit (avoid sync every iter)
            if it >= 4 and (it % 8 == 0):
                if not bool(update.any().item()):
                    break

        # Pick target = argmax(utility) over reachable + valid nodes.
        reachable = (dist < INF) & node_valid                             # [N, N_max]
        util_for_target = torch.where(reachable, utility, torch.full_like(utility, -1.0))
        target = util_for_target.argmax(dim=-1)                           # [N]
        # If all unreachable / zero utility, target == curr (no-op).
        no_util = (util_for_target.max(dim=-1).values <= 0)
        target = torch.where(no_util, curr_idx, target)

        # Reconstruct path by walking parent pointers from target → curr.
        P_max = self.guidepost_path_max
        path_idx = torch.full((N, P_max), -1, dtype=torch.long, device=dev)
        path_valid = torch.zeros((N, P_max), dtype=torch.bool, device=dev)
        # Start at target; first slot in path_idx[:, 0] is target.
        cur = target.clone()
        active = torch.ones(N, dtype=torch.bool, device=dev) & (target != curr_idx)
        for p in range(P_max):
            path_idx[:, p] = torch.where(active, cur, torch.full_like(cur, -1))
            path_valid[:, p] = active
            # advance: if cur == curr_idx, stop after recording this slot.
            reached_curr = (cur == curr_idx)
            # next node = parent[cur]; deactivate if cur reached curr_idx or parent == -1
            par = parent[arange_N, cur]
            stop = reached_curr | (par < 0)
            active = active & ~stop
            cur = torch.where(stop, cur, par)

        # Always include curr_idx as final waypoint if path has length.
        # The loop already records cur == curr_idx when reached_curr triggers.

        # Build mask of nodes on path.
        guidepost_mask = torch.zeros((N, N_max), dtype=torch.bool, device=dev)
        # scatter at path_idx where valid
        safe_pi = path_idx.clamp(min=0)
        mask_v = path_valid
        guidepost_mask.scatter_(1, safe_pi, mask_v)
        # Also include curr itself.
        guidepost_mask[arange_N, curr_idx] = True

        # Path xy for render. NaN-padded.
        path_xy = torch.full((N, P_max, 2), float("nan"), dtype=torch.float32, device=dev)
        safe_xy = node_xy[safe_pi]                                        # [N, P_max, 2]
        path_xy = torch.where(path_valid.unsqueeze(-1), safe_xy, path_xy)

        # next_hop[N]: the neighbor of curr_idx that lies on the shortest path
        # toward target. Identify as the node whose parent == curr_idx among path nodes.
        # path_idx walks target → ... → curr; the slot immediately before curr_idx
        # is the next hop. Find it by scanning path_idx for entries with parent == curr_idx.
        next_hop = curr_idx.clone()                                       # default: stay put
        for p in range(P_max):
            slot = path_idx[:, p]                                          # [N]
            slot_safe = slot.clamp(min=0)
            par = parent[arange_N, slot_safe]                              # parent of slot node
            is_next = (par == curr_idx) & (slot >= 0) & path_valid[:, p]
            # only overwrite where we haven't found a next_hop yet (stay put → curr_idx)
            need = (next_hop == curr_idx) & is_next
            next_hop = torch.where(need, slot, next_hop)

        # Build per-K bias for pointer: 1.0 at the K-slot whose neighbor flat-idx == next_hop.
        # curr_nbr is [N, K] from build().
        curr_nbr = info["curr_nbr"]                                        # [N, K]
        guidepost_nbr_bias = (curr_nbr == next_hop.unsqueeze(-1)).float()  # [N, K]
        # If next_hop == curr (no path), bias all zero.
        any_match = guidepost_nbr_bias.sum(dim=-1, keepdim=True) > 0
        guidepost_nbr_bias = guidepost_nbr_bias * any_match.float()

        # Write feat[6] guidepost.
        info["node_feat"][..., 5] = guidepost_mask.float()
        info["guidepost_mask"] = guidepost_mask
        info["guidepost_target"] = target
        info["guidepost_path_idx"] = path_idx
        info["guidepost_path_valid"] = path_valid
        info["guidepost_path_xy"] = path_xy
        info["guidepost_dist"] = dist
        info["guidepost_next_hop"] = next_hop                              # [N]
        info["guidepost_nbr_bias"] = guidepost_nbr_bias                    # [N, K]
        return info

    # ---------------------------------------------------------------------- #
    # B1 redo: BF from TARGET + warm-start + overwrite-mode + early exit     #
    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def bf_from_target(
        self,
        info: dict[str, torch.Tensor],
        target: torch.Tensor,                    # [N] long, BF source = target node
        dist_init: torch.Tensor | None = None,   # [N, N_max] warm-start dist (else +inf)
        max_iters: int | None = None,
        edge_valid: torch.Tensor | None = None,  # override edge set (else info["edge_valid"])
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Overwrite-mode Bellman-Ford from `target` over the lattice.

        Why overwrite-mode (`dist = best_vals` not `min(dist, best_vals)`):
            With monotonic-min BF + warm-start, stale-low dist values from a previous
            step persist even when the edge that achieved them is now invalid (e.g.
            a newly-scanned wall blocked the path). Overwrite-mode recomputes dist[v]
            from scratch each iter using current neighbor distances, so stale values
            propagate out within a few iterations. Forces dist[target] = 0 each iter
            to anchor the source.

        Returns:
            dist   [N, N_max] f32 — shortest path cost FROM target TO each node.
            parent [N, N_max] long — neighbor index used to reach each node (parent
                                     in the shortest-path tree rooted at target).
        """
        # Use the GEOMETRIC edge indices (not info["edge_idx"], which build() overwrote to
        # -1 wherever the FREE edge_valid is False). For valid FREE edges the two are
        # identical, so FREE-graph BF is unchanged; but the optimistic override has True
        # edges where info["edge_idx"] is -1, and gathering through -1 (clamped→0) would
        # point at the wrong neighbour. The geometric table + the active edge_valid mask
        # is the only consistent source for both graphs.
        N = info["edge_idx"].shape[0]
        edge_idx = self.edge_idx_static.unsqueeze(0).expand(N, -1, -1)   # [N, N_max, K]
        # Optional override: teammate-distance BF passes the optimistic (UNKNOWN-passable)
        # graph; None → the FREE graph, identical to prior behaviour.
        edge_valid = info["edge_valid"] if edge_valid is None else edge_valid  # [N, N_max, K]

        N_max = self.N_max
        K_ = K
        dev = edge_idx.device
        INF = float("inf")
        max_iters = max_iters if max_iters is not None else self.guidepost_iters

        # Init dist
        if dist_init is not None:
            dist = dist_init.clone()
        else:
            dist = torch.full((N, N_max), INF, dtype=torch.float32, device=dev)
        arange_N = torch.arange(N, device=dev)
        dist[arange_N, target] = 0.0
        parent = torch.full((N, N_max), -1, dtype=torch.long, device=dev)

        edge_len = self.edge_len.view(1, 1, K_)
        edge_idx_safe = edge_idx.clamp(min=0)

        for it in range(max_iters):
            nbr_dist = torch.gather(
                dist, dim=1, index=edge_idx_safe.view(N, -1)
            ).view(N, N_max, K_)
            cand = nbr_dist + edge_len
            cand = cand.masked_fill(~edge_valid, INF)
            best_vals, best_k = cand.min(dim=-1)                          # [N, N_max]
            # Overwrite (not monotonic min): handles stale dist_init correctly.
            prev_dist = dist
            dist = best_vals
            dist[arange_N, target] = 0.0                                  # force source
            best_parent = torch.gather(edge_idx, dim=2, index=best_k.unsqueeze(-1)).squeeze(-1)
            parent = best_parent
            # Early exit when converged (sync ~10 μs on modern Ada, cheap).
            if it >= 1 and (it % 8 == 0):
                if bool(torch.equal(prev_dist, dist)):
                    break

        return dist, parent

    @torch.no_grad()
    def build_radar(
        self,
        info: dict[str, torch.Tensor],
        teammate_src: torch.Tensor | None = None,   # [N, T] long: teammate last-known node idxs (-1 = none)
        gamma_r: float = 0.92,
        util_norm: float = 8.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RADAR boundary summary: compress the KNOWN world BEYOND the ego window onto the
        geodesic receptive-horizon nodes, so a feed-forward (no-recurrence) policy still gets a
        direction toward far exploration mass / teammates.

        Reuses the FREE-graph BF from curr already in info (bf_dist_from_curr / bf_parent_from_curr):
          - horizon = nodes within D_h = n_hops·NR px of curr (⇒ ≤ n_hops graph hops even along an
            all-axial path, so inside the neighbours' n_layers-hop receptive field);
          - a node BEYOND the horizon routes its mass DOWN its parent chain to the first node
            at/inside the horizon (its traversable GATEWAY) — obstacle-aware, the path bends around
            walls; never a straight-line projection through a wall;
          - weight = gamma_r ** (hops beyond horizon) — a travel-cost discount, so nearer-beyond
            mass dominates. Normalised by a FIXED constant (stationary — no per-env max).

        Returns (b_util, b_teammate), each [N, N_max], nonzero only on horizon gateway nodes.
        """
        dev = info["node_feat"].device
        N = info["node_feat"].shape[0]
        dist = info["bf_dist_from_curr"]                  # [N, N_max] px cost, +inf if unreachable
        parent = info["bf_parent_from_curr"]              # [N, N_max] predecessor toward curr (-1 root/none)
        utility = info["utility"]                         # [N, N_max] ∈ [0, 1]
        node_valid = info["node_valid"]
        NR = float(self.NR)
        D_h = float(self.n_hops) * NR                     # receptive horizon radius in px
        gamma = torch.as_tensor(float(gamma_r), device=dev)

        reachable = torch.isfinite(dist) & node_valid
        beyond = reachable & (dist > D_h)                 # [N, N_max]

        # Gateway: walk parent pointers until dist ≤ D_h (or the chain roots at -1). Pointer-follow;
        # `need.any()` early-exits at ≈graph-diameter iterations (checked every 8, like the BF), NOT
        # the N_max cap. Nodes already inside the horizon map to themselves (their mass is 0 anyway).
        g = torch.arange(self.N_max, device=dev).unsqueeze(0).expand(N, -1).clone()   # [N, N_max]
        for it in range(self.guidepost_iters):
            dist_g = torch.gather(dist, 1, g)
            par_g = torch.gather(parent, 1, g)
            need = (dist_g > D_h) & (par_g >= 0)
            g = torch.where(need, par_g.clamp(min=0), g)
            if it % 8 == 0 and not bool(need.any().item()):
                break

        # Travel-cost discount over the hops BEYOND the horizon (px excess / NR ≈ graph hops).
        excess_hops = (dist - D_h).clamp(min=0.0) / NR
        w = torch.pow(gamma, excess_hops)                                             # [N, N_max]

        b_util = torch.zeros((N, self.N_max), device=dev)
        src_w = torch.where(beyond, utility * w, torch.zeros_like(utility))           # only beyond mass
        b_util.scatter_add_(1, g, src_w)
        b_util = (b_util / float(util_norm)).clamp(0.0, 1.0)

        b_team = torch.zeros((N, self.N_max), device=dev)
        if teammate_src is not None and teammate_src.numel() > 0:
            T = teammate_src.shape[1]
            for t in range(T):
                src = teammate_src[:, t]                                              # [N], -1 = none
                src_safe = src.clamp(min=0)
                d_src = torch.gather(dist, 1, src_safe.unsqueeze(1)).squeeze(1)       # [N]
                g_src = torch.gather(g, 1, src_safe.unsqueeze(1)).squeeze(1)          # [N] gateway
                w_src = torch.pow(gamma, (d_src - D_h).clamp(min=0.0) / NR)
                use = (src >= 0) & torch.isfinite(d_src) & (d_src > D_h)              # only beyond-horizon teammates
                b_team.scatter_add_(1, g_src.unsqueeze(1), (use.float() * w_src).unsqueeze(1))
        b_team = b_team.clamp(0.0, 1.0)
        return b_util, b_team

    @torch.no_grad()
    def analytic_next_hop(
        self,
        curr_idx: torch.Tensor,                  # [N] flat node idx
        target: torch.Tensor,                    # [N] flat node idx
        edge_valid: torch.Tensor,                # [N, N_max, K]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """O(1) analytic direction from curr toward target on the lattice.

        Returns:
            k_analytic  [N] long  — K-slot of the immediate-step direction toward
                                    target. -1 when curr == target.
            edge_clear  [N] bool  — True if the immediate edge (curr, k_analytic)
                                    is valid. Does NOT trace the full path; only
                                    checks the first hop. (Full-path tracing would
                                    require a Python loop and lose the O(1) cost.)
        """
        N = curr_idx.shape[0]
        dev = curr_idx.device
        arange_N = torch.arange(N, device=dev)
        curr_li = curr_idx // self.LW
        curr_lj = curr_idx % self.LW
        tgt_li = target // self.LW
        tgt_lj = target % self.LW
        d_li = torch.sign(tgt_li - curr_li).long()                        # {-1, 0, +1}
        d_lj = torch.sign(tgt_lj - curr_lj).long()
        k_analytic = self.K_INDEX_TABLE[d_li + 1, d_lj + 1]               # [N], -1 if curr==target
        # Check edge_valid[arange, curr_idx, k_analytic]. k_analytic == -1 needs masking.
        k_safe = k_analytic.clamp(min=0)
        edge_at = edge_valid[arange_N, curr_idx, k_safe]                  # [N]
        edge_clear = edge_at & (k_analytic >= 0)
        return k_analytic, edge_clear

    @torch.no_grad()
    def select_target_no_bf(
        self,
        utility: torch.Tensor,        # [N, N_max]
        node_valid: torch.Tensor,     # [N, N_max]
    ) -> torch.Tensor:
        """Target = argmax(utility · node_valid). node_valid is set by flood-fill from
        curr in build() and already filters reachability, so a 'valid' node IS reachable
        from curr through known-FREE space. No BF needed for target selection."""
        masked = torch.where(node_valid, utility, torch.full_like(utility, -1.0))
        target = masked.argmax(dim=-1)                                     # [N]
        return target

    def select_target_analytic(
        self,
        utility: torch.Tensor,        # [N, N_max]  (frontier-gated, >0 only near frontiers)
        node_valid: torch.Tensor,     # [N, N_max]  reachable-through-FREE mask
        bf_dist_curr: torch.Tensor,   # [N, N_max]  BF path length curr→node (+inf unreachable)
        d_team: torch.Tensor | None = None,   # [N, N_max] BF dist teammate→node (rendezvous anchor)
        w: torch.Tensor | None = None,        # [N] offer weight ∈[0,1] (how much fresh map I owe)
        beta: float = 1.0,            # distance discount (per NR node-unit)
        lam: float = 1.0,             # rendezvous pull strength
        prev_target: torch.Tensor | None = None,  # [N] last step's committed target (-1 none)
        curr_idx: torch.Tensor | None = None,      # [N] curr node — safe fallback when no frontier
        keep_margin: float = 0.2,     # commit: keep prev unless best beats it by >this fraction
        exclude: torch.Tensor | None = None,  # [N, N_max] bool: nodes a teammate out-ranked → drop
        d_team_min: torch.Tensor | None = None,  # [N, N_max] BF dist NEAREST teammate→node (ownership)
        sep_w: float | torch.Tensor = 0.0,   # separation strength ∈[0,1] (0 = off). [N] tensor → per-env
                                             # (e.g. 1−w: separate when nothing fresh to offer)
        sep_k: float = 2.0,           # ownership-gate sigmoid sharpness (per hop)
        sep_m: float = 0.0,           # margin: hops a teammate must be closer before I yield
    ) -> torch.Tensor:
        """Deterministic global target = the frontier that maximizes

            explore(c) = utility(c) / (1 + β·BF_dist(curr→c)/NR)            # nearest-richest
            score(c)   = explore(c) · (1 + λ·w·rdv(c)),  rdv(c)=1/(1+β·d_team(c)/NR)

        w≈0 → pure exploration; w high → among comparable frontiers prefer the one toward the
        teammate I owe the most fresh map → explore-while-converging for a map-exchange
        rendezvous. Candidates restricted to reachable nodes with utility>0 (frontier-anchored).
        """
        nr = float(self.NR)
        dc = bf_dist_curr / nr
        dc = torch.where(torch.isfinite(dc), dc, torch.full_like(dc, 1.0e6))
        explore = utility / (1.0 + beta * dc)
        score = explore
        if d_team is not None and w is not None:
            dt = d_team / nr
            dt = torch.where(torch.isfinite(dt), dt, torch.full_like(dt, 1.0e6))
            rdv = 1.0 / (1.0 + beta * dt)
            score = explore * (1.0 + lam * w.unsqueeze(-1) * rdv)
        # Separation / ownership: down-weight frontiers a teammate is closer to (division of labor).
        # adv > 0 → I'm closer (mine) → gate≈1; adv < 0 → teammate closer → gate≈0 → score scaled
        # down by (1 − sep_w). One-sided (never negative → no repulsion); single-frontier still wins
        # the argmax (only down-weighted, not zeroed → no idle).
        sep_is_tensor = isinstance(sep_w, torch.Tensor)
        sep_active = sep_is_tensor or sep_w > 0.0
        if d_team_min is not None and sep_active:
            dtm = d_team_min / nr
            dtm = torch.where(torch.isfinite(dtm), dtm, torch.full_like(dtm, 1.0e6))
            adv = dtm - dc                                                 # hops; >0 = mine
            own_gate = torch.sigmoid(sep_k * (adv - sep_m))               # ∈(0,1)
            sw = sep_w.unsqueeze(-1) if sep_is_tensor else sep_w           # [N,1] broadcast over nodes
            score = score * (1.0 - sw * (1.0 - own_gate))
        elig = node_valid & (utility > 0.0)
        if curr_idx is not None:
            # Exclude the node the agent is ON: even if it's still a frontier (unknown within NR
            # due to range/occlusion), staying re-scans nothing — the agent must MOVE toward the
            # unknown. Zero distance (dc=0) would otherwise let curr win the argmax → self-target.
            node_idx = torch.arange(self.N_max, device=utility.device).view(1, -1)
            elig = elig & (node_idx != curr_idx.unsqueeze(1))
        if exclude is not None:
            elig = elig & ~exclude                                         # drop teammate-claimed nodes
        masked = torch.where(elig, score, torch.full_like(score, 0.0))     # 0 = ineligible (scores>0)
        best_idx = masked.argmax(dim=-1)                                   # [N]
        best_val = masked.gather(1, best_idx.unsqueeze(1)).squeeze(1)      # [N]
        has_elig = elig.any(dim=-1)                                        # [N]
        # Commitment / hysteresis: keep last step's target unless a candidate beats it by
        # > keep_margin (fraction) — stops the 1-step argmax flip that ping-pongs the guidepost.
        target = best_idx
        if prev_target is not None:
            pv = prev_target.clamp(min=0)
            prev_elig = torch.gather(elig, 1, pv.unsqueeze(1)).squeeze(1) & (prev_target >= 0)
            prev_val = masked.gather(1, pv.unsqueeze(1)).squeeze(1)
            keep = prev_elig & (best_val <= prev_val * (1.0 + keep_margin))
            target = torch.where(keep, pv, best_idx)
        # Safe fallback: no eligible frontier at all → stay on curr (guidepost→self→zero bias),
        # NEVER node 0 (which renders as an off-map top-left target).
        if curr_idx is not None:
            target = torch.where(has_elig, target, curr_idx)
        return target                                                     # [N]

    def select_target_nearest_bf(
        self,
        utility: torch.Tensor,        # [N, N_max]  (frontier-gated, >0 only near frontiers)
        node_valid: torch.Tensor,     # [N, N_max]  reachable-through-FREE mask
        bf_dist_curr: torch.Tensor,   # [N, N_max]  BF path length curr→node (+inf unreachable)
        prev_target: torch.Tensor | None = None,   # [N] last step's committed target (-1 none)
        curr_idx: torch.Tensor | None = None,      # [N] curr node — safe fallback when no frontier
        keep_margin: float = 0.2,     # commit: keep prev unless a new frontier is >this fraction CLOSER
        exclude: torch.Tensor | None = None,       # [N, N_max] bool: nodes a teammate claimed → drop
    ) -> torch.Tensor:
        """Global target = the reachable frontier with the SMALLEST BF path distance from curr.

        Pure greedy-nearest on the wall-aware BF field — ignores utility magnitude, rendezvous
        and separation (those are the analytic selector's job). The guidepost then just points the
        agent at the closest unexplored boundary it can actually reach. Same eligibility set as the
        analytic path (reachable, utility>0, not the curr node, not teammate-claimed) and the same
        fallback (stay on curr when no frontier) so downstream code is unchanged.
        """
        INF = float("inf")
        elig = node_valid & (utility > 0.0)
        if curr_idx is not None:
            node_idx = torch.arange(self.N_max, device=utility.device).view(1, -1)
            elig = elig & (node_idx != curr_idx.unsqueeze(1))          # must MOVE off curr
        if exclude is not None:
            elig = elig & ~exclude
        dist = torch.where(elig, bf_dist_curr, torch.full_like(bf_dist_curr, INF))
        best_idx = dist.argmin(dim=-1)                                  # [N] nearest reachable frontier
        best_val = dist.gather(1, best_idx.unsqueeze(1)).squeeze(1)     # [N]
        has_elig = elig.any(dim=-1)                                     # [N]
        # Commitment / hysteresis: keep last step's target unless a candidate is closer by
        # > keep_margin (fraction) — stops the 1-step argmin flip when two frontiers tie in reach.
        target = best_idx
        if prev_target is not None:
            pv = prev_target.clamp(min=0)
            prev_elig = torch.gather(elig, 1, pv.unsqueeze(1)).squeeze(1) & (prev_target >= 0)
            prev_val = dist.gather(1, pv.unsqueeze(1)).squeeze(1)
            # smaller dist is better → keep prev unless the new nearest beats it by the margin.
            keep = prev_elig & (best_val >= prev_val * (1.0 - keep_margin))
            target = torch.where(keep, pv, best_idx)
        # Safe fallback: no eligible frontier → stay on curr (never node 0 / off-map top-left).
        if curr_idx is not None:
            target = torch.where(has_elig, target, curr_idx)
        return target                                                     # [N]

    @torch.no_grad()
    def build_guidepost_v2(
        self,
        info: dict[str, torch.Tensor],
        target: torch.Tensor,                    # [N] target flat idx
        dist_init: torch.Tensor | None = None,   # [N, N_max] warm-start dist
    ) -> dict[str, torch.Tensor]:
        """B1-redo guidepost: BF FROM target (not curr), overwrite-mode, warm-startable.

        Writes the same info-dict fields as `build_guidepost`. Downstream code does
        not change. The semantic change is: `parent[v]` now points TOWARD target (not
        toward curr), so the path is reconstructed by walking parent from curr.

        Args:
            info: dict from `build()` (must contain edge_idx, edge_valid, node_valid,
                  utility, curr_idx, curr_nbr, curr_nbr_valid, node_xy).
            target: [N] flat node idx of the BF source (= long-horizon goal).
            dist_init: optional warm-start [N, N_max] from a previous step. When the
                       target is unchanged across steps, this dramatically cuts BF iters.

        Writes to `info`:
            guidepost_mask        bool  [N, N_max]
            guidepost_target      long  [N]
            guidepost_path_idx    long  [N, P_max]      curr → ... → target
            guidepost_path_valid  bool  [N, P_max]
            guidepost_path_xy     float [N, P_max, 2]
            guidepost_dist        float [N, N_max]      for next-step warm-start
            guidepost_next_hop    long  [N]
            guidepost_nbr_bias    float [N, K]
        Also fills `info["node_feat"][..., 5]`.
        """
        edge_idx = info["edge_idx"]
        edge_valid = info["edge_valid"]
        node_valid = info["node_valid"]
        curr_idx = info["curr_idx"]
        curr_nbr = info["curr_nbr"]              # [N, K] flat indices, -1 padded
        curr_nbr_valid = info["curr_nbr_valid"]  # [N, K]
        node_xy = self.node_xy

        N = edge_idx.shape[0]
        N_max = self.N_max
        K_ = K
        P_max = self.guidepost_path_max
        dev = edge_idx.device
        INF = float("inf")
        arange_N = torch.arange(N, device=dev)

        # 1) BF FROM target with warm-start. Returns dist[N, N_max], parent[N, N_max].
        dist, parent = self.bf_from_target(info, target=target, dist_init=dist_init)

        # 2) next_hop = parent[curr]. If parent[curr] < 0 (curr == target or unreachable),
        #    stay put (next_hop = curr).
        par_at_curr = parent[arange_N, curr_idx]                          # [N]
        next_hop = torch.where(par_at_curr >= 0, par_at_curr, curr_idx)

        # 3) Reconstruct path by walking parent from curr toward target.
        path_idx = torch.full((N, P_max), -1, dtype=torch.long, device=dev)
        path_valid = torch.zeros((N, P_max), dtype=torch.bool, device=dev)
        cur = curr_idx.clone()
        active = torch.ones(N, dtype=torch.bool, device=dev)
        for p in range(P_max):
            path_idx[:, p] = torch.where(active, cur, torch.full_like(cur, -1))
            path_valid[:, p] = active
            reached_target = (cur == target)
            par = parent[arange_N, cur]
            stop = reached_target | (par < 0)
            active = active & ~stop
            cur = torch.where(stop, cur, par)

        # 4) Mask
        guidepost_mask = torch.zeros((N, N_max), dtype=torch.bool, device=dev)
        safe_pi = path_idx.clamp(min=0)
        guidepost_mask.scatter_(1, safe_pi, path_valid)
        guidepost_mask[arange_N, curr_idx] = True

        # 5) path_xy for render
        path_xy = torch.full((N, P_max, 2), float("nan"), dtype=torch.float32, device=dev)
        safe_xy = node_xy[safe_pi]
        path_xy = torch.where(path_valid.unsqueeze(-1), safe_xy, path_xy)

        # 6) guidepost_nbr_bias: one-hot over K at the slot matching next_hop
        guidepost_nbr_bias = (curr_nbr == next_hop.unsqueeze(-1)).float()
        any_match = guidepost_nbr_bias.sum(dim=-1, keepdim=True) > 0
        guidepost_nbr_bias = guidepost_nbr_bias * any_match.float()

        # 7) Write feat[6] + info
        info["node_feat"][..., 5] = guidepost_mask.float()
        info["guidepost_mask"] = guidepost_mask
        info["guidepost_target"] = target
        info["guidepost_path_idx"] = path_idx
        info["guidepost_path_valid"] = path_valid
        info["guidepost_path_xy"] = path_xy
        info["guidepost_dist"] = dist
        info["guidepost_next_hop"] = next_hop
        info["guidepost_nbr_bias"] = guidepost_nbr_bias
        # Target world coords (render uses this — node_xy in obs is local window now).
        info["guidepost_target_xy"] = node_xy[target]                 # [N, 2]
        return info

    # ---------------------------------------------------------------------- #
    # Phase C: extract ego-centric subgraph window centered on curr_idx       #
    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def extract_local_window(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Slice a (2·n_hops + 3)² window out of the global graph, per env.

        The encoder runs on the window instead of the full N_max-node lattice.
        Receptive field of `n_layers` GAT layers stacked on the window equals
        `n_layers` lattice hops outward from the center — same as on the full
        graph, but with vastly fewer wasted FLOPs.

        Returns a NEW dict with local-window views. Edge indices in this dict
        are LOCAL indices into [0, window_size). The global-flat `curr_nbr`
        (needed by env.step to decode actions) is retained as `curr_nbr_global`.

        All other fields produced by `build()` / `build_guidepost_v2()` are kept
        from `info` for downstream use (visited_step, guidepost_path_xy in global
        world coords, etc.). The caller decides which subset to expose in obs.
        """
        edge_idx_global = info["edge_idx"]                     # [N, N_max, K]
        edge_valid_global = info["edge_valid"]                 # [N, N_max, K]
        node_valid_global = info["node_valid"]                 # [N, N_max]
        node_feat_global = info["node_feat"]                   # [N, N_max, F]
        node_xy_global = info["node_xy"]                       # [N, N_max, 2]
        utility_global = info["utility"]                       # [N, N_max]
        curr_idx = info["curr_idx"]                            # [N]
        curr_nbr_global = info["curr_nbr"]                     # [N, K]

        N = curr_idx.shape[0]
        W2 = self.window_size
        K_ = K
        F_ = node_feat_global.shape[-1]
        dev = curr_idx.device

        # Per-env local→global map: window_idx_table indexed by curr_idx.
        l2g = self.window_idx_table[curr_idx]                  # [N, W²]
        in_window = l2g >= 0                                   # [N, W²]
        safe_g = l2g.clamp(min=0)                              # [N, W²], -1 → 0 for gather safety

        # Gather node-level fields
        local_node_xy = torch.gather(
            node_xy_global, 1, safe_g.unsqueeze(-1).expand(-1, -1, 2),
        )                                                      # [N, W², 2]
        local_node_valid = torch.gather(node_valid_global, 1, safe_g) & in_window  # [N, W²]
        local_node_feat = torch.gather(
            node_feat_global, 1, safe_g.unsqueeze(-1).expand(-1, -1, F_),
        )
        local_node_feat = local_node_feat * in_window.unsqueeze(-1).float()        # zero OOB
        local_utility = torch.gather(utility_global, 1, safe_g) * in_window.float()

        # Edge indices are LOCAL (broadcast static table)
        local_edge_idx = self.window_local_edge_table.unsqueeze(0).expand(N, -1, -1).contiguous()
        # Edge validity: combine global edge_valid AND neighbor-in-window AND own-in-window
        local_edge_valid_global = torch.gather(
            edge_valid_global, 1, safe_g.unsqueeze(-1).expand(-1, -1, K_),
        )                                                      # [N, W², K]
        nbr_in_window = local_edge_idx >= 0                    # [N, W², K]
        own_in_window = in_window.unsqueeze(-1)                # [N, W², 1]
        local_edge_valid = local_edge_valid_global & nbr_in_window & own_in_window

        # curr is window center
        local_curr_idx = torch.full((N,), self.window_local_center, dtype=torch.long, device=dev)
        local_curr_nbr = self.window_local_edge_table[self.window_local_center].view(1, K_).expand(N, K_).contiguous()
        local_curr_nbr_valid = torch.gather(
            local_edge_valid, 1, local_curr_idx.view(N, 1, 1).expand(-1, 1, K_),
        ).squeeze(1)                                           # [N, K]

        return {
            "node_xy_local": local_node_xy,
            "node_valid_local": local_node_valid,
            "node_feat_local": local_node_feat,
            "edge_idx_local": local_edge_idx,
            "edge_valid_local": local_edge_valid,
            "curr_idx_local": local_curr_idx,
            "curr_nbr_local": local_curr_nbr,
            "curr_nbr_valid_local": local_curr_nbr_valid,
            "utility_local": local_utility,
            "local_to_global": l2g,           # [N, W²], -1 padded
            "curr_nbr_global": curr_nbr_global,
        }
