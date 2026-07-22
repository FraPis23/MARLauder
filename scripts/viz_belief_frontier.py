"""Belief FRONTIER diagnostic — verify the 3 expansion rules on a FROZEN partial map.

Unlike viz_belief_expand (which blooms over an almost-entirely-UNKNOWN map, so it just floods),
this first EXPLORES a chunk of the map (agents kept in comm, so we only build a known region with
real walls + a real frontier), then FREEZES and expands the belief hop-by-hop over that fixed
partial graph. That isolates the three rules we care about:

  (1) KNOWN zone   → spread only over edge_valid (known-free edges): STOPS at known walls.
  (2) UNKNOWN zone → spread over the optimistic graph (walls invisible): all 8 neighbours.
  (3) FRONTIER     → cross known→unknown ONLY through generatable edges (collision-free, from a
                     real frontier free-node), never indiscriminately across a known wall.

Render: node occupancy backdrop (free = grey, KNOWN wall = dark red, unknown = near-black),
belief zone colored by the hop it entered (birth step: dark core → bright frontier). The LAST
frame additionally draws every ACTIVE expansion edge inside the zone, colored by type:
  green  = known-free edge (rule 1)   orange = frontier crossing (rule 3)   blue = unknown-internal (rule 2)

    python scripts/viz_belief_frontier.py --split train/difficult --map-idx 30 \
        --explore-steps 90 --hops 60 --out test/belief/belief_frontier.gif
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
from matplotlib.collections import LineCollection
import numpy as np
import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split
from env.frontier import compute_frontier

_UNKNOWN, _FREE, _OBST = 0, 1, 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/difficult")
    ap.add_argument("--map-idx", type=int, default=30)
    ap.add_argument("--explore-steps", type=int, default=90)
    ap.add_argument("--hops", type=int, default=60)
    ap.add_argument("--observer", type=int, default=0)
    ap.add_argument("--seed-at", choices=["observer", "teammate"], default="observer",
                    help="which agent's node seeds the frozen belief bloom")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("test/belief/belief_frontier.gif"))
    args = ap.parse_args()

    ag = args.observer
    # keep agents IN COMM during exploration (huge range) — we only want the KNOWN map they build.
    cfg = EnvCfg.from_ckpt_dict({}, n_envs=1, n_agents=2, max_episode_steps=args.explore_steps + 2,
                                use_teammate_belief=True, comm_model="los", comm_range_px=1e6)
    split = load_split(args.split, device=args.device)
    env = Explorer(split, cfg, seed=int(args.map_idx))
    env.store_render_global = True
    env.reload_map(env_idx=0, map_idx=int(args.map_idx))
    obs = env._last_obs
    e = 0
    K = env.graph.edge_idx_static.shape[1]
    node_xy = env.graph.node_xy.cpu().numpy()
    eidx = env.graph.edge_idx_static                                   # [N_max, K]
    last_act = [-1, -1]

    def greedy(obs):
        vf = obs["value_field"][0].clone(); am = obs["action_mask"][0].bool()
        bias = torch.zeros_like(vf); bias[0, :K // 2] += .03; bias[1, K // 2:] += .03
        for a in range(2):
            if last_act[a] >= 0:
                am[a, K - 1 - last_act[a]] = False
        act = torch.where(am, vf + bias, torch.full_like(vf, -1e9)).argmax(-1)
        for a in range(2):
            last_act[a] = int(act[a])
        return act.view(1, 2)

    # ---- EXPLORE to build a partial known map --------------------------------------------------
    for _ in range(args.explore_steps):
        obs, _, done, _ = env.step(greedy(obs))
        if bool(done[e]):
            break

    # ---- FREEZE: rebuild the graph at the current partial occupancy for the observer -----------
    occ = env.world.occupancy_torch[e:e + 1, ag].contiguous()         # [1, H, W]
    fro = compute_frontier(occ)                                       # [1, H, W] bool
    robot_xy = env.pos[e:e + 1, ag]                                   # [1, 2]
    vstep = torch.full((1, env.N_max), -1, dtype=torch.long, device=args.device)
    info = env.graph.build(occ, fro, robot_xy, vstep, current_step=0)
    edge_valid = info["edge_free"][0]                                 # [N_max, K] known-free, no robot gate
    edge_optim = info["edge_valid_optim"][0]                          # [N_max, K] optimistic
    assert edge_optim is not None, "optim graph off — need M>1 / build_optim_graph"

    # node occupancy category (for backdrop + touches_unknown), sampled at node centers
    nx = node_xy[:, 0].astype(int).clip(0, env.W - 1)
    ny = node_xy[:, 1].astype(int).clip(0, env.H - 1)
    node_cat = occ[0].cpu().numpy()[ny, nx]                           # [N_max] {0 unk,1 free,2 obst}
    node_unknown = torch.tensor(node_cat == _UNKNOWN, device=args.device)   # [N_max]

    # ---- expand_edge EXACTLY as explorer._refresh_obs builds it (lines ~1224-1230) -------------
    nbr_unknown = node_unknown[eidx.clamp(min=0)]                     # [N_max, K]
    touches_unknown = node_unknown.unsqueeze(-1) | nbr_unknown        # [N_max, K]
    expand_edge = edge_valid | (edge_optim & touches_unknown)         # [N_max, K]

    # edge-type label for drawing: 1 known-free, 2 frontier-cross, 3 unknown-internal
    src_unk = node_unknown.unsqueeze(-1).expand(-1, K)
    dst_unk = nbr_unknown
    etype = torch.zeros_like(expand_edge, dtype=torch.long)
    etype = torch.where(expand_edge & ~src_unk & ~dst_unk, torch.tensor(1, device=args.device), etype)  # both known
    etype = torch.where(expand_edge & (src_unk ^ dst_unk), torch.tensor(2, device=args.device), etype)  # crossing
    etype = torch.where(expand_edge & src_unk & dst_unk, torch.tensor(3, device=args.device), etype)    # both unknown

    # ---- manual hop-by-hop wavefront from the chosen seed --------------------------------------
    seed_node = int(env.curr_idx_global[e, ag if args.seed_at == "observer" else 1 - ag])
    reached = torch.zeros(env.N_max, dtype=torch.bool, device=args.device)
    reached[seed_node] = True
    birth = np.full(env.N_max, -1, dtype=np.int32)
    birth[seed_node] = 0
    ev = expand_edge
    nbr = eidx.clamp(min=0)

    frames = []
    seed_xy = node_xy[seed_node]

    def render(hop, draw_edges=False):
        fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=110)
        ax.set_facecolor("#050509")
        # backdrop: occupancy per node
        col = np.where(node_cat == _FREE, "#3b3b52",
              np.where(node_cat == _OBST, "#5a1414", "#0e0e18"))
        ax.scatter(node_xy[:, 0], node_xy[:, 1], s=6, c=col, zorder=1)
        rj = reached.cpu().numpy()
        zi = np.where(rj)[0]
        if zi.size:
            age = hop - birth[zi]
            ax.scatter(node_xy[zi, 0], node_xy[zi, 1], s=18, c=age, cmap="plasma_r",
                       vmin=0, vmax=max(1, hop), zorder=3)
        if draw_edges:
            segs, cols = [], []
            cmap = {1: "#22c55e", 2: "#f59e0b", 3: "#3b82f6"}
            et = etype.cpu().numpy(); nb = nbr.cpu().numpy()
            for n in zi:                                             # only edges inside the zone
                for k in range(K):
                    m = nb[n, k]
                    if et[n, k] > 0 and rj[m] and n < m:            # dedupe
                        segs.append([node_xy[n], node_xy[m]])
                        cols.append(cmap[int(et[n, k])])
            if segs:
                ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.1, zorder=2, alpha=.8))
        ax.scatter(*seed_xy, s=130, c="white", marker="x", linewidths=2.2, zorder=6)
        ttl = f"frozen belief bloom  hop={hop}  zone={int(rj.sum())} nodes"
        if draw_edges:
            ttl += "   edges: green=known-free  orange=frontier-cross  blue=unknown"
        ax.set_title(ttl, color="w", fontsize=10)
        pad = 25
        ax.set_xlim(node_xy[:, 0].min() - pad, node_xy[:, 0].max() + pad)
        ax.set_ylim(node_xy[:, 1].max() + pad, node_xy[:, 1].min() - pad)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        fr = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
        plt.close(fig)
        return fr

    frames.append(render(0))
    for hop in range(1, args.hops + 1):
        nbr_in = reached[nbr]                                        # [N_max, K] neighbour in zone?
        grow = (nbr_in & ev).any(dim=-1)                            # node joins if a valid nbr is in zone
        newly = grow & ~reached
        birth[newly.cpu().numpy()] = hop
        reached = reached | grow
        frames.append(render(hop))
    frames.append(render(args.hops, draw_edges=True))               # final frame with edges
    frames += [frames[-1]] * 12                                     # hold the edge frame

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, duration=120, loop=0)
    print(f"[save] {args.out}  ({len(frames)} frames)  seed_node={seed_node} "
          f"known-free-nodes={int((node_cat==_FREE).sum())} unknown={int((node_cat==_UNKNOWN).sum())}")


if __name__ == "__main__":
    main()
