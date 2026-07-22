"""Belief expansion on a REAL dataset map, partially revealed.

Reveal a KNOWN disk (radius R) around the centre of a real map — inside it occupancy = the true
free/wall layout, outside = UNKNOWN. The belief is seeded at the centre and expands with the env's
3-zone rule (known 8-conn respecting real walls · frontier crossing ORTHOGONAL only · unknown 8-conn).
Where the disk boundary crosses open space we get natural frontiers scattered around; where it hits a
wall there is none. Nodes are colored by the hop they entered the zone (dark seed → bright frontier);
frontier nodes (known-free touching unknown) are ringed so you can see where the belief leaks out.

    python scripts/viz_belief_real.py --split train/difficult --map-idx 30 --radius 220 \
        --hops 110 --out test/belief/belief_real.gif
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/difficult")
    ap.add_argument("--map-idx", type=int, default=30)
    ap.add_argument("--radius", type=float, default=220.0, help="known-disk radius in px")
    ap.add_argument("--hops", type=int, default=110)
    ap.add_argument("--out", type=Path, default=Path("test/belief/belief_real.gif"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    cfg = EnvCfg.from_ckpt_dict({}, n_envs=1, n_agents=2, max_episode_steps=8,
                                use_teammate_belief=True, comm_model="los", comm_range_px=1e6)
    split = load_split(args.split, device=dev)
    env = Explorer(split, cfg, seed=int(args.map_idx))
    env.reload_map(env_idx=0, map_idx=int(args.map_idx))
    H, W = env.H, env.W
    gt_free = (env.world.gt_torch[0] == 1)                              # [H, W] bool, True = free

    # ---- reveal a known disk around a free cell near the centre --------------------------------
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    cy, cx = H // 2, W // 2
    # snap centre onto the nearest free pixel so the seed sits in open space
    fy, fx = torch.where(gt_free)
    d2 = (fy - cy) ** 2 + (fx - cx) ** 2
    k = int(d2.argmin())
    cy, cx = int(fy[k]), int(fx[k])
    known = ((yy - cy) ** 2 + (xx - cx) ** 2) <= args.radius ** 2       # [H, W] disk
    occ = torch.full((H, W), _UNK, dtype=torch.uint8, device=dev)
    occ[known & gt_free] = _FREE
    occ[known & ~gt_free] = _OBST
    occ = occ.view(1, H, W)

    node_xy = env.graph.node_xy.cpu().numpy()
    eidx = env.graph.edge_idx_static
    nbr = eidx.clamp(min=0)
    nfx = node_xy[:, 0].astype(int).clip(0, W - 1)
    nfy = node_xy[:, 1].astype(int).clip(0, H - 1)
    node_cat = occ[0].cpu().numpy()[nfy, nfx]                           # [N_max]

    # seed = free node nearest the disk centre
    free_nodes = np.where(node_cat == _FREE)[0]
    seed_node = int(free_nodes[((node_xy[free_nodes, 0] - cx) ** 2 +
                                (node_xy[free_nodes, 1] - cy) ** 2).argmin()])
    seed_xy = node_xy[seed_node]

    fro = compute_frontier(occ)
    robot_xy = torch.tensor([[float(seed_xy[0]), float(seed_xy[1])]], device=dev)
    vstep = torch.full((1, env.N_max), -1, dtype=torch.long, device=dev)
    info = env.graph.build(occ, fro, robot_xy, vstep, current_step=0)
    ev = info["edge_free"][0]                                           # known-free, no robot gate (belief)
    eo = info["edge_valid_optim"][0]
    assert eo is not None

    orth_k = torch.tensor([abs(dr) + abs(dc) == 1 for (dr, dc) in NBR_OFFSETS],
                          dtype=torch.bool, device=dev)
    node_unknown = torch.tensor(node_cat == _UNK, device=dev)
    nbr_unknown = node_unknown[nbr]
    crossing = node_unknown.unsqueeze(-1) ^ nbr_unknown
    internal_u = node_unknown.unsqueeze(-1) & nbr_unknown
    optim_ok = internal_u | (crossing & orth_k.view(1, -1))
    expand = ev | (eo & optim_ok)

    # frontier NODES = known-free with an orthogonal crossing available (where belief actually exits)
    frontier_node = ((~node_unknown) & (crossing & orth_k.view(1, -1)).any(-1)).cpu().numpy()
    fr_xy = node_xy[frontier_node]
    print(f"known-free={int((node_cat==_FREE).sum())} obst={int((node_cat==_OBST).sum())} "
          f"unk={int((node_cat==_UNK).sum())} frontier-nodes={int(frontier_node.sum())}")

    reached = torch.zeros(env.N_max, dtype=torch.bool, device=dev)
    reached[seed_node] = True
    birth = np.full(env.N_max, -1, np.int32); birth[seed_node] = 0
    pad = 20
    extent = (node_xy[:, 0].min() - pad, node_xy[:, 0].max() + pad,
              node_xy[:, 1].min() - pad, node_xy[:, 1].max() + pad)

    def render(hop):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.set_facecolor("#050509")
        col = np.where(node_cat == _FREE, "#3b3b52",
              np.where(node_cat == _OBST, "#5a1414", "#0e0e18"))
        ax.scatter(node_xy[:, 0], node_xy[:, 1], s=7, c=col, zorder=1)
        if fr_xy.size:
            ax.scatter(fr_xy[:, 0], fr_xy[:, 1], s=42, facecolors="none",
                       edgecolors="#10b981", linewidths=1.0, zorder=2)      # frontier ring
        rj = reached.cpu().numpy(); zi = np.where(rj)[0]
        if zi.size:
            ax.scatter(node_xy[zi, 0], node_xy[zi, 1], s=20, c=hop - birth[zi],
                       cmap="plasma_r", vmin=0, vmax=max(1, hop), zorder=3)
        ax.scatter(*seed_xy, s=150, c="white", marker="x", linewidths=2.4, zorder=6)
        ax.set_title(f"REAL map {args.split}#{args.map_idx}  R={int(args.radius)}px   "
                     f"hop={hop}  zone={int(rj.sum())}   (green ring = frontier)",
                     color="w", fontsize=10)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[3], extent[2])
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        fr = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
        plt.close(fig)
        return fr

    frames = [render(0)]
    for hop in range(1, args.hops + 1):
        grow = (reached[nbr] & expand).any(dim=-1)
        newly = grow & ~reached
        birth[newly.cpu().numpy()] = hop
        reached = reached | grow
        frames.append(render(hop))
        if not newly.any():
            break
    frames += [frames[-1]] * 12

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, duration=110, loop=0)
    print(f"[save] {args.out}  ({len(frames)} frames)  seed={seed_node} centre=({cx},{cy})")


if __name__ == "__main__":
    main()
