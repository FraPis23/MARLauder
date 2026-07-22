"""Belief expansion — three controlled cases on a big synthetic map.

Observer agent on one side, the belief (last-known of the OTHER agent) seeded elsewhere. We hand-
build the occupancy so the number of frontiers is EXACT, then expand the belief hop-by-hop over the
graph (same rule as env: known 8-conn respecting walls · frontier crossing ORTHOGONAL only · unknown
interior 8-conn) and render the wavefront colored by the hop it entered.

  known  — map fully KNOWN (no unknown at all): pure known-zone expansion, routes around walls.
  one    — one KNOWN room, a wall with ONE gap → belief fills the room then exits the single frontier.
  three  — same room, THREE separate gaps → three simultaneous frontier exits.

    python scripts/viz_belief_cases.py --out-dir test/belief

Produces belief_case_known.gif / _one.gif / _three.gif.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from env.explorer import EnvCfg, Explorer
from env.graph_lattice import NBR_OFFSETS
from env.maps import load_split
from env.frontier import compute_frontier

_UNK, _FREE, _OBST = 0, 1, 2


def build_occ(case: str, H: int, W: int) -> np.ndarray:
    """Return uint8 occupancy [H, W] in {0 unk, 1 free, 2 obst} for the requested case."""
    occ = np.full((H, W), _FREE, np.uint8)
    if case == "known":
        # fully known: free everywhere + two serpentine bars (offset gaps, kept connected → the belief
        # must wind top→right-gap→middle→left-gap→bottom, so we watch it route AROUND known walls).
        t = max(6, W // 90)                                   # wall thickness
        occ[H // 3 - t:H // 3 + t, 0:int(W * 0.70)] = _OBST   # upper bar from LEFT edge, gap on the right
        occ[2 * H // 3 - t:2 * H // 3 + t, int(W * 0.30):W] = _OBST   # lower bar from RIGHT edge, gap on left
        return occ
    # frontier cases: left KNOWN room, vertical wall near mid, UNKNOWN to the right.
    wall_x0, wall_x1 = int(W * 0.46), int(W * 0.50)
    occ[:, wall_x1:] = _UNK                                   # everything right of the wall = unknown
    occ[:, wall_x0:wall_x1] = _OBST                           # the dividing wall
    if case == "one":
        gaps = [(int(H * 0.44), int(H * 0.56))]
    elif case == "three":
        gaps = [(int(H * 0.10), int(H * 0.20)),
                (int(H * 0.45), int(H * 0.55)),
                (int(H * 0.80), int(H * 0.90))]
    else:
        raise ValueError(case)
    for y0, y1 in gaps:                                       # carve FREE passages through the wall
        occ[y0:y1, wall_x0:wall_x1] = _FREE                  # the gap is known-free, touching unknown
    return occ


def build_occ_reentry(H: int, W: int) -> np.ndarray:
    """Seed's LEFT room and a RIGHT known-free POCKET, sealed apart by a full wall (no gap between them
    in the known map), but both touching UNKNOWN (top/bottom strips + the pocket's right side). The
    pocket is unreachable from the seed/robot through free space → only re-entry from the unknown fills
    it. Demonstrates 'from outside it comes back into the interior if a path exists'."""
    t = max(6, W // 90)
    occ = np.full((H, W), _FREE, np.uint8)
    occ[:int(H * 0.15), :] = _UNK                            # top unknown strip (frontier for both rooms)
    occ[int(H * 0.85):, :] = _UNK                            # bottom unknown strip
    occ[:, int(W * 0.75):] = _UNK                            # unknown to the right of the pocket
    occ[int(H * 0.15):int(H * 0.85), int(W * 0.45):int(W * 0.45) + t] = _OBST   # sealing wall (no gap)
    return occ


def render(node_xy, node_cat, reached, birth, hop, seed_xy, obs_xy, title, extent):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_facecolor("#050509")
    col = np.where(node_cat == _FREE, "#3b3b52",
          np.where(node_cat == _OBST, "#5a1414", "#0e0e18"))
    ax.scatter(node_xy[:, 0], node_xy[:, 1], s=7, c=col, zorder=1)
    zi = np.where(reached)[0]
    if zi.size:
        age = hop - birth[zi]
        ax.scatter(node_xy[zi, 0], node_xy[zi, 1], s=20, c=age, cmap="plasma_r",
                   vmin=0, vmax=max(1, hop), zorder=3)
    ax.scatter(*obs_xy, s=150, facecolors="none", edgecolors="#22d3ee", linewidths=2.4, zorder=5)
    ax.scatter(*seed_xy, s=140, c="white", marker="x", linewidths=2.4, zorder=6)
    ax.set_title(f"{title}   hop={hop}   zone={int(reached.sum())} nodes", color="w", fontsize=11)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[3], extent[2])
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    fr = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
    plt.close(fig)
    return fr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/difficult", help="only used to size the canvas/graph")
    ap.add_argument("--map-idx", type=int, default=0)
    ap.add_argument("--hops", type=int, default=90)
    ap.add_argument("--out-dir", type=Path, default=Path("test/belief"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = EnvCfg.from_ckpt_dict({}, n_envs=1, n_agents=2, max_episode_steps=8,
                                use_teammate_belief=True, comm_model="los", comm_range_px=1e6)
    split = load_split(args.split, device=args.device)
    env = Explorer(split, cfg, seed=int(args.map_idx))
    env.reload_map(env_idx=0, map_idx=int(args.map_idx))
    H, W = env.H, env.W
    dev = args.device
    node_xy = env.graph.node_xy.cpu().numpy()
    eidx = env.graph.edge_idx_static                                     # [N_max, K]
    nbr = eidx.clamp(min=0)
    orth_k = torch.tensor([abs(dr) + abs(dc) == 1 for (dr, dc) in NBR_OFFSETS],
                          dtype=torch.bool, device=dev)                  # [K]
    nfx = node_xy[:, 0].astype(int).clip(0, W - 1)
    nfy = node_xy[:, 1].astype(int).clip(0, H - 1)

    def nearest_node(x, y):
        return int(((node_xy[:, 0] - x) ** 2 + (node_xy[:, 1] - y) ** 2).argmin())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for case in ("known", "one", "three", "reentry"):
        occ_np = build_occ_reentry(H, W) if case == "reentry" else build_occ(case, H, W)
        occ = torch.from_numpy(occ_np).to(dev).view(1, H, W)
        node_cat = occ_np[nfy, nfx]                                     # [N_max]

        # observer + belief seed: both in the KNOWN-free region, far apart.
        if case == "known":
            obs_x, obs_y = W * 0.90, H * 0.90
            seed_x, seed_y = W * 0.06, H * 0.06
        elif case == "reentry":
            obs_x, obs_y = W * 0.20, H * 0.50           # robot in the LEFT room (seals node_valid there)
            seed_x, seed_y = W * 0.20, H * 0.50         # belief born in the left room too
        else:
            obs_x, obs_y = W * 0.10, H * 0.85
            seed_x, seed_y = W * 0.10, H * 0.12
        # snap to a free node
        free_nodes = np.where(node_cat == _FREE)[0]
        def snap(x, y):
            fx, fy = node_xy[free_nodes, 0], node_xy[free_nodes, 1]
            return int(free_nodes[((fx - x) ** 2 + (fy - y) ** 2).argmin()])
        obs_node = snap(obs_x, obs_y)
        seed_node = snap(seed_x, seed_y)
        obs_xy = node_xy[obs_node]; seed_xy = node_xy[seed_node]

        fro = compute_frontier(occ)
        robot_xy = torch.tensor([[obs_xy[0], obs_xy[1]]], dtype=torch.float32, device=dev)
        vstep = torch.full((1, env.N_max), -1, dtype=torch.long, device=dev)
        info = env.graph.build(occ, fro, robot_xy, vstep, current_step=0)
        ev = info["edge_free"][0]                                       # known-free, no robot gate (belief)
        eo = info["edge_valid_optim"][0]
        assert eo is not None

        node_unknown = torch.tensor(node_cat == _UNK, device=dev)      # [N_max]
        nbr_unknown = node_unknown[nbr]                                # [N_max, K]
        crossing = node_unknown.unsqueeze(-1) ^ nbr_unknown
        internal_u = node_unknown.unsqueeze(-1) & nbr_unknown
        optim_ok = internal_u | (crossing & orth_k.view(1, -1))
        expand = ev | (eo & optim_ok)                                  # [N_max, K]

        reached = torch.zeros(env.N_max, dtype=torch.bool, device=dev)
        reached[seed_node] = True
        birth = np.full(env.N_max, -1, np.int32); birth[seed_node] = 0
        pad = 20
        extent = (node_xy[:, 0].min() - pad, node_xy[:, 0].max() + pad,
                  node_xy[:, 1].min() - pad, node_xy[:, 1].max() + pad)
        title = {"known": "MAPPA TUTTA NOTA", "one": "UNA FRONTIERA",
                 "three": "TRE FRONTIERE", "reentry": "RIENTRO NELLA TASCA"}[case]
        frames = [render(node_xy, node_cat, reached.cpu().numpy(), birth, 0, seed_xy, obs_xy, title, extent)]
        for hop in range(1, args.hops + 1):
            grow = (reached[nbr] & expand).any(dim=-1)
            newly = grow & ~reached
            birth[newly.cpu().numpy()] = hop
            reached = reached | grow
            frames.append(render(node_xy, node_cat, reached.cpu().numpy(), birth, hop,
                                 seed_xy, obs_xy, title, extent))
            if not newly.any():
                break
        frames += [frames[-1]] * 10
        out = args.out_dir / f"belief_case_{case}.gif"
        imageio.mimsave(out, frames, duration=110, loop=0)
        print(f"[save] {out}  ({len(frames)} frames)  seed={seed_node} obs={obs_node} "
              f"free={int((node_cat==_FREE).sum())} unk={int((node_cat==_UNK).sum())}")


if __name__ == "__main__":
    main()
