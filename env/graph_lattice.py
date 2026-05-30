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
        device: str = "cuda:0",
    ) -> None:
        self.device = device
        self.H, self.W = canvas
        self.NR = int(nr)
        self.LH = self.H // self.NR
        self.LW = self.W // self.NR
        self.N_max = self.LH * self.LW
        self.UR = int(utility_range_px)
        self.S = int(collision_samples)
        self.flood_max_iters = int(flood_max_iters)
        self.SR = float(sensor_range_px)
        # Bellman-Ford iteration upper bound: Manhattan diameter ~ LH+LW; pad a bit.
        self.guidepost_iters = int(guidepost_iters) if guidepost_iters else int(self.LH + self.LW + 8)
        # Path-reconstruction length cap (number of edges on a path).
        self.guidepost_path_max = int(guidepost_path_max) if guidepost_path_max else int(self.LH + self.LW + 4)

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

        # 2) Flood-fill from robot's lattice cell, restricted to is_free_lat.
        rx = robot_xy[:, 0].clamp(0, W - 1).long()
        ry = robot_xy[:, 1].clamp(0, H - 1).long()
        rli = (ry // self.NR).clamp(0, self.LH - 1)
        rlj = (rx // self.NR).clamp(0, self.LW - 1)
        seed = torch.zeros((N, self.LH, self.LW), dtype=torch.float32, device=dev)
        seed[torch.arange(N, device=dev), rli, rlj] = 1.0
        seed = seed * is_free_lat.float()                     # seed must itself be free; if not, will stay zero
        reach = seed
        for _ in range(self.flood_max_iters):
            dil = F.conv2d(reach.unsqueeze(1), self._dilate_k, padding=1).squeeze(1)
            new = (dil > 0).float() * is_free_lat.float()
            if torch.equal(new, reach):
                break
            reach = new
        node_valid = (reach.view(N, self.N_max) > 0)          # [N, N_max] bool

        # 3) Edges: edge target idx, validity = both endpoints valid + collision-free.
        edge_idx = self.edge_idx_static.unsqueeze(0).expand(N, -1, -1).contiguous()  # [N, N_max, K]
        nbr_idx_safe = edge_idx.clamp(min=0)                  # [-1 → 0] for gather
        nbr_valid_geom = edge_idx >= 0
        # Gather neighbor validity.
        nbr_node_valid = torch.gather(
            node_valid, dim=1, index=nbr_idx_safe.view(N, -1)
        ).view(N, self.N_max, K)
        endpoints_valid = node_valid.unsqueeze(-1) & nbr_node_valid & nbr_valid_geom

        # Collision check: sample S points along each edge in world coords.
        nx_all = self.node_xy[:, 0].view(1, self.N_max, 1, 1)    # [1, N_max, 1, 1]
        ny_all = self.node_xy[:, 1].view(1, self.N_max, 1, 1)
        sx = nx_all + self.sample_dx.view(1, 1, K, self.S)       # [1, N_max, K, S]
        sy = ny_all + self.sample_dy.view(1, 1, K, self.S)
        sx = sx.clamp(0, W - 1).long()
        sy = sy.clamp(0, H - 1).long()
        sample_lin = sy * W + sx                                  # [1, N_max, K, S]
        sample_lin = sample_lin.expand(N, -1, -1, -1)             # [N, N_max, K, S]
        occ_samples = torch.gather(occ_flat, 1, sample_lin.reshape(N, -1)).view(N, self.N_max, K, self.S)
        collision_free = (occ_samples != OBSTACLE).all(dim=-1)    # [N, N_max, K]

        edge_valid = endpoints_valid & collision_free
        edge_idx = torch.where(edge_valid, edge_idx, torch.full_like(edge_idx, -1))

        # 4) Utility via integral image of frontier (square window approximation of disk).
        fr = frontier.float()                                     # [N, H, W]
        ii = F.pad(fr, (1, 0, 1, 0)).cumsum(-1).cumsum(-2)        # [N, H+1, W+1]
        UR = self.UR
        x0 = (node_x - UR).clamp(0, W)
        x1 = (node_x + UR + 1).clamp(0, W)
        y0 = (node_y - UR).clamp(0, H)
        y1 = (node_y + UR + 1).clamp(0, H)
        def _gather_ii(ys, xs):
            lin = ys * (W + 1) + xs
            lin = lin.view(1, self.N_max).expand(N, -1)
            return torch.gather(ii.view(N, -1), 1, lin)
        util = (_gather_ii(y1, x1) - _gather_ii(y0, x1)
                - _gather_ii(y1, x0) + _gather_ii(y0, x0))
        util_max = float((2 * UR + 1) ** 2)
        utility_norm = (util / util_max).clamp(0, 1)
        utility_norm = utility_norm * node_valid.float()

        # 5) curr_idx: O(1) analytic lookup.
        # Node center at (lj+0.5)*NR → nearest column = floor(x/NR).
        # Agents always land on lattice node positions (_spread_starts_graph places
        # them on node_xy; step() moves them to chosen node coords). Floor-divide
        # is therefore exact — no argmin search needed.
        lj_curr = (robot_xy[:, 0] / float(self.NR)).long().clamp(0, self.LW - 1)
        li_curr = (robot_xy[:, 1] / float(self.NR)).long().clamp(0, self.LH - 1)
        curr_idx = li_curr * self.LW + lj_curr                   # [N]

        # 6) Per-node features (7 dims).
        node_feat = torch.zeros((N, self.N_max, 7), dtype=torch.float32, device=dev)
        curr_xy = self.node_xy[curr_idx]                          # [N, 2]
        win_half = 0.5 * max(self.H, self.W)
        node_feat[..., 0] = (self.node_xy[:, 0].view(1, -1) - curr_xy[:, 0:1]) / win_half  # x_rel
        node_feat[..., 1] = (self.node_xy[:, 1].view(1, -1) - curr_xy[:, 1:2]) / win_half  # y_rel
        node_feat[..., 2] = utility_norm
        visited = (visited_step >= 0).float()
        node_feat[..., 3] = visited
        last_norm = torch.where(
            visited.bool(),
            visited_step.float() / max(1.0, float(current_step)),
            torch.zeros_like(visited),
        ).clamp(0, 1)
        node_feat[..., 4] = last_norm
        # feat[5] prob_occupied (M>1, written by Explorer)
        # feat[6] guidepost (filled by build_guidepost if called)
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

        Also writes the mask into `info['node_feat'][..., 6]` to populate feat[6].
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
        info["node_feat"][..., 6] = guidepost_mask.float()
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
        edge_idx = info["edge_idx"]              # [N, N_max, K]
        edge_valid = info["edge_valid"]          # [N, N_max, K]

        N = edge_idx.shape[0]
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
    def extract_topk_candidates(
        self,
        utility: torch.Tensor,        # [N, N_max]
        node_valid: torch.Tensor,     # [N, N_max]
        curr_xy: torch.Tensor,        # [N, 2] world coords of agent
        K: int = 16,
        bf_dist: torch.Tensor | None = None,   # [N, N_max] BF dist from curr; replaces euclid
    ) -> dict[str, torch.Tensor]:
        """Phase A v2 / A1: top-K frontier candidates per env, ranked by valid utility.

        Returns dict with:
            cand_idx       long  [N, K]      flat global node idx; -1 where slot has no valid candidate
            cand_xy        f32   [N, K, 2]   world coords (zeros where invalid)
            cand_utility   f32   [N, K]      raw utility values (0 where invalid)
            cand_valid     bool  [N, K]      True where cand_idx[k] >= 0
            cand_rel_xy    f32   [N, K, 2]   (cand_xy − curr_xy) (zeros where invalid)
            cand_euclid    f32   [N, K]      ‖cand_xy − curr_xy‖ (0 where invalid)

        Reachability is already encoded in node_valid (flood-fill from curr through
        known-FREE space). Selected targets are guaranteed reachable; wall-bumping
        at the strategic level is impossible by construction.
        """
        N, N_max = utility.shape
        # Mask unreachable nodes with sentinel below 0 so they never enter top-K.
        SENTINEL = float("-inf")
        masked = torch.where(node_valid, utility, torch.full_like(utility, SENTINEL))
        topk_vals, topk_idx = masked.topk(K, dim=-1)                              # [N, K]
        valid = torch.isfinite(topk_vals) & (topk_vals > 0.0)                     # frontier must have utility>0
        # Use 0-fill (not -1) for the gather; rely on valid mask downstream.
        safe_idx = torch.where(valid, topk_idx, torch.zeros_like(topk_idx))       # [N, K]
        xy = self.node_xy[safe_idx]                                               # [N, K, 2]
        xy = xy * valid.unsqueeze(-1).float()                                     # zero out invalid slots
        cand_idx_out = torch.where(valid, topk_idx, torch.full_like(topk_idx, -1))
        cand_util = torch.where(valid, topk_vals, torch.zeros_like(topk_vals))
        rel = xy - curr_xy.unsqueeze(1)                                           # [N, K, 2]
        rel = rel * valid.unsqueeze(-1).float()
        if bf_dist is not None:
            # Option A: replace straight-line euclid with BF shortest-path distance
            # through known-FREE space. Wall-aware. Same magnitude as euclid in open
            # space, much larger in maze around obstacles.
            d_per_cand = torch.gather(bf_dist, dim=1, index=safe_idx)              # [N, K]
            # Unreachable nodes have +inf; clamp to a large finite value for stability.
            d_per_cand = torch.where(torch.isfinite(d_per_cand), d_per_cand,
                                      torch.full_like(d_per_cand, 1.0e6))
            euclid = d_per_cand * valid.float()                                    # [N, K]
        else:
            euclid = rel.norm(dim=-1) * valid.float()                              # [N, K]
        return {
            "cand_idx":     cand_idx_out,
            "cand_xy":      xy,
            "cand_utility": cand_util,
            "cand_valid":   valid,
            "cand_rel_xy":  rel,
            "cand_euclid":  euclid,
        }

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
        Also fills `info["node_feat"][..., 6]`.
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
        info["node_feat"][..., 6] = guidepost_mask.float()
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
