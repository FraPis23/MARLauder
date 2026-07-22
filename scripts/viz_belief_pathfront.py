"""Belief "pathfront" — visual correspondence of the two-phase hypothesis model.

Reveals a KNOWN disk on a real map, seeds the belief at the centre (teammate last-known position),
freezes up to Kf frontier-cluster hypotheses, then animates:
  TRANSIT — one point per hypothesis travels the BF geodesic lkp→frontier (marker size ∝ weight);
  BLOOM   — on arrival the uniform zone expands from that frontier, weighted by the hypothesis prob.
Each hypothesis has its own colour; the last-known node is a white ✕, cluster frontiers are ringed.

    python scripts/viz_belief_pathfront.py --split train/difficult --map-idx 5 --radius 300 \
        --steps 90 --out test/belief/belief_pathfront.gif
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
from matplotlib.colors import LinearSegmentedColormap

from env.explorer import EnvCfg, Explorer
from env.graph_lattice import NBR_OFFSETS
from env.maps import load_split
from env.frontier import compute_frontier
from env.teammate_belief_pathfront import freeze_hypotheses, advance_pathfront

_UNK, _FREE, _OBST = 0, 1, 2
# Classic heat ramp on a FIXED 0→1 probability scale: 0 = black (off), low prob = light yellow,
# high prob = intense red. Control points are placed at the low end because a Σ=1 belief spread over
# thousands of nodes has small per-node values — this keeps the diffuse bloom mass visible (yellow)
# while concentrated/overlapping mass reads orange→red. Endpoints honour "0=black, 1=red".
_REDHEAT = LinearSegmentedColormap.from_list("redheat", [
    (0.000, "#000000"),  # 0     → black (no probability)
    (0.003, "#fff2a0"),  # ~0.3% → light yellow (a single diffuse bloom node is this faint)
    (0.02,  "#ffb020"),  # ~2%   → amber
    (0.10,  "#e23010"),  # ~10%  → red
    (1.00,  "#7a0000"),  # 1.0   → intense dark red (a near-certain node)
])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/difficult")
    ap.add_argument("--map-idx", type=int, default=5)
    ap.add_argument("--radius", type=float, default=300.0)
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--kf", type=int, default=6)
    ap.add_argument("--frame-ms", type=float, default=180.0, help="ms per frame (higher = slower gif)")
    ap.add_argument("--vmax", type=float, default=0.02,
                    help="FIXED top of the probability colour scale (lower → low probs more visible)")
    ap.add_argument("--absorb-gain", type=float, default=1.0, help="β_F = min(gain·utility, beta_max)")
    ap.add_argument("--beta-max", type=float, default=0.9)
    ap.add_argument("--diffuse-lambda", type=float, default=0.5)
    ap.add_argument("--min-unknown", type=int, default=4,
                    help="frontier = FREE node with ≥ this many unknown 8-nbrs (keeps openings, drops corridor interiors)")
    ap.add_argument("--out", type=Path, default=Path("test/belief/belief_pathfront.gif"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    cfg = EnvCfg.from_ckpt_dict({}, n_envs=1, n_agents=2, max_episode_steps=8,
                                use_teammate_belief=True, comm_model="los", comm_range_px=1e6)
    env = Explorer(load_split(args.split, device=dev), cfg, seed=int(args.map_idx))
    env.reload_map(env_idx=0, map_idx=int(args.map_idx))
    H, W, N_max = env.H, env.W, env.N_max
    gt_free = (env.world.gt_torch[0] == 1)

    # ---- reveal known disk around a central free pixel -----------------------------------------
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    fy, fx = torch.where(gt_free)
    k0 = int(((fy - H // 2) ** 2 + (fx - W // 2) ** 2).argmin())
    cy, cx = int(fy[k0]), int(fx[k0])
    known = ((yy - cy) ** 2 + (xx - cx) ** 2) <= args.radius ** 2
    occ = torch.full((H, W), _UNK, dtype=torch.uint8, device=dev)
    occ[known & gt_free] = _FREE
    occ[known & ~gt_free] = _OBST
    occ = occ.view(1, H, W)

    node_xy = env.graph.node_xy.cpu().numpy()
    eidx = env.graph.edge_idx_static
    nbr = eidx.clamp(min=0)
    nfx = node_xy[:, 0].astype(int).clip(0, W - 1)
    nfy = node_xy[:, 1].astype(int).clip(0, H - 1)
    node_cat = occ[0].cpu().numpy()[nfy, nfx]
    free_nodes = np.where(node_cat == _FREE)[0]
    lkp = int(free_nodes[((node_xy[free_nodes, 0] - cx) ** 2 + (node_xy[free_nodes, 1] - cy) ** 2).argmin()])
    lkp_xy = node_xy[lkp]

    # ---- graph build + BF from lkp (optimistic) + the 4-rule expand graph ----------------------
    fro = compute_frontier(occ)
    info = env.graph.build(occ, fro, torch.tensor([[float(lkp_xy[0]), float(lkp_xy[1])]], device=dev),
                           torch.full((1, N_max), -1, dtype=torch.long, device=dev), 0)
    edge_free = info["edge_free"]                                   # [1, N, K]
    edge_optim = info["edge_valid_optim"]                          # [1, N, K]
    utility = info["utility"]                                      # [1, N]
    # transit BF over the KNOWN-FREE graph (not the optimistic unknown-passable one): the geodesic
    # lkp→frontier must stay inside the known disk so the transit point follows real corridors and
    # doesn't shortcut across walls / through the unknown (matches env/explorer.py:_pathfront_belief).
    dist, parent = env.graph.bf_from_target(info, target=torch.tensor([lkp], device=dev),
                                            edge_valid=edge_free.bool())  # [1, N]

    orth_k = torch.tensor([abs(dr) + abs(dc) == 1 for (dr, dc) in NBR_OFFSETS], dtype=torch.bool, device=dev)
    node_unknown = torch.tensor(node_cat == _UNK, device=dev).view(1, N_max)
    nbr_unknown = node_unknown[0][nbr].view(1, N_max, -1)
    crossing = node_unknown.unsqueeze(-1) ^ nbr_unknown
    internal_u = node_unknown.unsqueeze(-1) & nbr_unknown
    optim_ok = internal_u | (crossing & orth_k.view(1, 1, -1))
    expand_edge = edge_free.bool() | (edge_optim & optim_ok)       # [1, N, K]

    # frontier = known-FREE OPENINGS into the unknown: FREE with ≥min_unknown UNKNOWN 8-neighbours (≤7).
    # The threshold (vs ≥1) drops thin-corridor interior cells that would chain into one giant cluster;
    # distinct openings stay distinct clusters. Matches env/explorer.py pf_frontier_min_unknown.
    node_free_t = torch.tensor(node_cat == _FREE, device=dev).view(1, N_max)
    n_unk_t = nbr_unknown.sum(-1)                                   # [1, N] unknown 8-neighbours
    frontier_node = (node_free_t & (n_unk_t >= args.min_unknown) & (n_unk_t <= 7))  # [1, N] openings

    # ---- FREEZE hypotheses ---------------------------------------------------------------------
    front_node, weight, dist_h, path = freeze_hypotheses(
        lkp_node=torch.tensor([lkp], device=dev),
        frontier_node=frontier_node, dist=dist, parent=parent, utility=utility,
        nbr_idx=eidx, edge_ok=edge_free.bool(), node_spacing=float(env.graph.NR),
        node_xy=env.graph.node_xy, Kf=args.kf, Lmax=int(env.graph.guidepost_path_max))
    used = ((front_node[0] >= 0) & (weight[0] > 1e-6)).cpu().numpy()   # only hypotheses carrying prob
    print(f"hypotheses(w>0): {int(used.sum())}  weights={weight[0].cpu().numpy().round(3)}  "
          f"dist_h(hops)={dist_h[0].cpu().numpy()}")

    live = torch.zeros((1, N_max), dtype=torch.float32, device=dev)
    acc = torch.zeros((1, N_max), dtype=torch.float32, device=dev)
    seeded = torch.zeros((1, args.kf), dtype=torch.bool, device=dev)
    fr_xy = node_xy[front_node[0].clamp(min=0).cpu().numpy()]

    def render(s, pv, psum, tviz):
        """CLASSIC heatmap of the summed posterior p on a FIXED (rescaled) 0→vmax probability scale —
        consistent across frames, not per-frame normalised. 0 = black/backdrop, low = light yellow,
        high = red; vmax<1 rescales so the diffuse blooms are visible. The underlying MAP stays visible
        under the heat: free = grey, wall = dark red, unknown = near-black; probability is overlaid only
        where p>0. Overlapping blooms sum → hotter."""
        fig, ax = plt.subplots(figsize=(8.8, 8), dpi=100)
        ax.set_facecolor("#07070c")
        # MAP backdrop (all nodes) — stays visible wherever there is no belief mass.
        col = np.where(node_cat == _FREE, "#2b2b3d",
              np.where(node_cat == _OBST, "#59201c", "#12121c"))
        ax.scatter(node_xy[:, 0], node_xy[:, 1], s=10, c=col, zorder=1)
        for i in range(args.kf):
            if used[i]:
                ax.scatter(*fr_xy[i], s=70, facecolors="none", edgecolors="#3f5166",
                           linewidths=1.0, zorder=2)
        hot = np.where(pv > 1e-6)[0]                                  # overlay heat only where p>0
        sc = ax.scatter(node_xy[hot, 0], node_xy[hot, 1], s=15, c=pv[hot], cmap=_REDHEAT,
                        vmin=0.0, vmax=float(args.vmax), zorder=3)    # FIXED rescaled 0→vmax
        # TRANSIT dots: every still-travelling hypothesis at UNIFORM brightness (cyan ring), so all
        # Kf points are visible departing lkp→frontier even when the prob-weights concentrate on one.
        tp = np.where(tviz > 0.5)[0]
        if tp.size:
            ax.scatter(node_xy[tp, 0], node_xy[tp, 1], s=95, facecolors="none",
                       edgecolors="#25f0ff", linewidths=2.2, zorder=6)
        ax.scatter(*lkp_xy, s=140, c="white", marker="x", linewidths=2.2, zorder=7)
        n_arr = int(((dist_h[0] <= s) & (front_node[0] >= 0)).sum())
        ax.set_title(f"PATHFRONT  step={s}   Σp={psum:.3f}   p_max={pv.max():.3f}  (scale 0–{args.vmax})   "
                     f"{int(used.sum())} hyp ({n_arr} bloom, {int(used.sum()) - n_arr} transit)",
                     color="w", fontsize=9)
        pad = 20
        ax.set_xlim(node_xy[:, 0].min() - pad, node_xy[:, 0].max() + pad)
        ax.set_ylim(node_xy[:, 1].max() + pad, node_xy[:, 1].min() - pad)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(f"probability (FIXED 0→{args.vmax}: black=0, yellow=low, red≥{args.vmax})",
                     color="w", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="w", labelsize=7)
        plt.setp(cb.ax.get_yticklabels(), color="w")
        fig.tight_layout(); fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        fr = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
        plt.close(fig)
        return fr

    frames = []
    w_np = weight[0].cpu().numpy()
    for s in range(0, args.steps + 1):
        step_t = torch.tensor([s], device=dev)
        live, acc, seeded, p, alive, tviz, weight = advance_pathfront(
            live, acc, seeded, front_node=front_node, weight=weight, dist_h=dist_h, path=path,
            step=step_t, frontier_node=frontier_node, utility=utility, edge_free=edge_free.bool(),
            nbr_idx=eidx, absorb_gain=args.absorb_gain, beta_max=args.beta_max,
            diffuse_lambda=args.diffuse_lambda)
        frames.append(render(s, p[0].cpu().numpy(), float(p.sum()), tviz[0].cpu().numpy()))

        if s in (max(1, int(dist_h[0].max())), args.steps):
            print(f"  step {s}: Σlive={float(live.sum()):.3f} Σacc={float(acc.sum()):.3f} "
                  f"Σp={float(p.sum()):.4f}  (weights={w_np.round(3)})  "
                  f"acc-on-frontiers={float((acc[0] * frontier_node[0].float()).sum()):.3f}")
    frames += [frames[-1]] * 20

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, duration=float(args.frame_ms), loop=0)
    print(f"[save] {args.out}  ({len(frames)} frames @ {args.frame_ms}ms)  lkp={lkp} "
          f"centre=({cx},{cy}) Σp={float(p.sum()):.3f}")


if __name__ == "__main__":
    main()
