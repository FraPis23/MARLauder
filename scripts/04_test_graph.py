"""Step 4 test: build the lattice graph after a few LiDAR scans and render.

Drops a single agent at the start, moves it on a short manual L-path (a few
sub-steps), then builds the graph and saves a PNG with:
  - occupancy background
  - frontier overlay (light red)
  - graph nodes (cyan = valid, dim = dead, BIG = curr)
  - edges (thin gray)
  - agent (blue)

    python scripts/04_test_graph.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch
from PIL import Image, ImageDraw

from env.frontier import compute_frontier
from env.graph_lattice import GraphLattice
from env.maps import load_split, sample_batch
from env.world_warp import WarpWorld

C_FREE_GT = (224, 226, 230)
C_OBST_GT = (74, 80, 92)
C_BELIEF_FREE = (160, 200, 230)
C_BELIEF_OBST = (235, 120, 110)
C_UNKNOWN_DIM = (35, 35, 45)
C_FRONTIER = (255, 90, 90)
C_AGENT = (80, 130, 250)
C_NODE_ACTIVE = (40, 200, 230)
C_NODE_DEAD = (60, 70, 80)
C_NODE_CURR = (255, 200, 60)
C_EDGE = (120, 130, 140)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--map-idx", type=int, default=0)
    ap.add_argument("--nr", type=int, default=8)
    ap.add_argument("--sensor-range", type=float, default=60.0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=12, help="number of LiDAR scans (random walk)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/step_04/graph.png"))
    args = ap.parse_args()

    split = load_split(args.split, device=args.device)
    gt, starts, _ = sample_batch(split, 1, indices=np.array([args.map_idx]), seed=0, device=args.device)
    H, W = split.canvas
    world = WarpWorld(gt, n_agents=1, sensor_range=args.sensor_range, device=args.device)

    # short random walk (cells) just to expand the occupancy — graph quality is the test.
    rng = np.random.default_rng(args.seed)
    sr, sc = int(starts[0, 0]), int(starts[0, 1])
    pos = torch.tensor([[sc, sr]], dtype=torch.float32, device=args.device)
    world.set_positions(pos)
    world.scan()
    for _ in range(args.steps):
        dx, dy = int(rng.integers(-20, 21)), int(rng.integers(-20, 21))
        nx = int(np.clip(pos[0, 0].item() + dx, 1, W - 2))
        ny = int(np.clip(pos[0, 1].item() + dy, 1, H - 2))
        # only commit if target cell is FREE in GT (so we are simulating a valid robot move)
        if int(gt[0, ny, nx].item()) == 1:
            pos[0, 0] = nx
            pos[0, 1] = ny
            world.set_positions(pos)
            world.scan()

    occupancy = world.occupancy_torch
    frontier = compute_frontier(occupancy)
    graph = GraphLattice(canvas=(H, W), nr=args.nr, device=args.device)
    visited = torch.full((1, graph.N_max), -1, dtype=torch.long, device=args.device)
    info = graph.build(occupancy, frontier, pos, visited, current_step=0)
    graph.build_guidepost(info)

    nv = int(info["node_valid"].sum())
    ne = int(info["edge_valid"].sum())
    print(f"[graph] LH×LW={graph.LH}×{graph.LW}={graph.N_max}  active nodes={nv}  edges={ne}")
    print(f"[graph] curr_idx={int(info['curr_idx'][0])}  curr_nbr={info['curr_nbr'][0].tolist()}")
    print(f"[graph] curr_nbr_valid={info['curr_nbr_valid'][0].tolist()}")

    gt_np = gt[0].cpu().numpy()
    bel_np = occupancy[0].cpu().numpy()
    fr_np = frontier[0].cpu().numpy()
    nv_np = info["node_valid"][0].cpu().numpy()
    nxy = info["node_xy"][0].cpu().numpy()
    eidx = info["edge_idx"][0].cpu().numpy()
    evalid = info["edge_valid"][0].cpu().numpy()
    util_np = info["utility"][0].cpu().numpy()
    curr = int(info["curr_idx"][0])

    img = np.empty((H, W, 3), dtype=np.uint8)
    img[gt_np == 1] = C_FREE_GT
    img[gt_np == 0] = C_OBST_GT
    img[bel_np == 0] = C_UNKNOWN_DIM
    img[bel_np == 1] = C_BELIEF_FREE
    img[bel_np == 2] = C_BELIEF_OBST
    img[fr_np] = C_FRONTIER
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)

    # Edges first (under nodes).
    for k_node in range(graph.N_max):
        if not nv_np[k_node]:
            continue
        x0, y0 = float(nxy[k_node, 0]), float(nxy[k_node, 1])
        for k in range(8):
            if not evalid[k_node, k]:
                continue
            tgt = int(eidx[k_node, k])
            if tgt < k_node:                           # draw each edge once
                continue
            x1, y1 = float(nxy[tgt, 0]), float(nxy[tgt, 1])
            dr.line([(x0, y0), (x1, y1)], fill=C_EDGE, width=1)

    # Nodes: color blend cyan→orange by utility.
    for k_node in range(graph.N_max):
        x, y = float(nxy[k_node, 0]), float(nxy[k_node, 1])
        if not nv_np[k_node]:
            col = C_NODE_DEAD
            rad = 1
        else:
            u = float(util_np[k_node])
            col = (
                int(C_NODE_ACTIVE[0] * (1 - u) + 255 * u),
                int(C_NODE_ACTIVE[1] * (1 - u) + 140 * u),
                int(C_NODE_ACTIVE[2] * (1 - u) + 50  * u),
            )
            rad = 2
        dr.ellipse([x - rad, y - rad, x + rad, y + rad], fill=col)

    # Guidepost path (amber polyline) + target ring.
    path_xy = info["guidepost_path_xy"][0].cpu().numpy()
    path_v = info["guidepost_path_valid"][0].cpu().numpy()
    pts = [(float(path_xy[p, 0]), float(path_xy[p, 1])) for p in range(path_xy.shape[0]) if bool(path_v[p])]
    if len(pts) >= 2:
        dr.line(pts, fill=(255, 180, 40), width=3)
    tgt_idx = int(info["guidepost_target"][0])
    if tgt_idx != curr:
        tx, ty = float(nxy[tgt_idx, 0]), float(nxy[tgt_idx, 1])
        dr.ellipse([tx - 9, ty - 9, tx + 9, ty + 9], outline=(255, 230, 60), width=3)
        dr.ellipse([tx - 2, ty - 2, tx + 2, ty + 2], fill=(255, 230, 60))

    # Current node.
    cx, cy = float(nxy[curr, 0]), float(nxy[curr, 1])
    dr.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=C_NODE_CURR, width=2)

    # Agent.
    ax, ay = float(pos[0, 0]), float(pos[0, 1])
    dr.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill=C_AGENT, outline=(255, 255, 255), width=1)

    # Diagonal-cost sanity check.
    el = info["edge_len"]
    print(f"[guidepost] target_idx={tgt_idx}  dist={float(info['guidepost_dist'][0, tgt_idx]):.2f} px")
    print(f"[edge_len] axial={el[1]:.2f}  diag={el[0]:.2f}  ratio={el[0]/el[1]:.4f} (expect ≈1.4142)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    im.save(args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
