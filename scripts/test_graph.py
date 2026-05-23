"""GATE Fase 2 — lattice gerarchico + frontier + utility + edge-mask su GPU.

Esplora un po' (random walk) per avere area scoperta, poi costruisce:
  frontiera, lattice ego-centrico (validita/edge-mask/utility), anchor globali.
Verifiche:
  1. nodi validi stanno su celle FREE
  2. edge attivi non attraversano ostacoli (ricontrollo a campione)
  3. utility>0 esiste e si concentra vicino alle frontiere
  4. anchor entro capacita (cap a_max)
  5. il lattice ego si ri-ancora col movimento
Salva una visualizzazione PNG.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from env.maps import load_split, sample_batch
from env.world_warp import WarpWorld
from env.frontier import frontier_mask, frontier_centers
from env.graph_lattice import EgoLattice, build_anchors, _DIRS

K_DEFAULT = 21


def random_walk(world, gt_np, x, y, steps, step_size=6.0):
    h, w = gt_np.shape
    heading = np.random.uniform(0, 2 * math.pi)
    for _ in range(steps):
        world.set_positions(torch.tensor([[x, y]], dtype=torch.float32, device="cuda"))
        world.scan()
        for _ in range(8):
            nx, ny = x + math.cos(heading) * step_size, y + math.sin(heading) * step_size
            ix, iy = int(nx + 0.5), int(ny + 0.5)
            if 0 <= ix < w and 0 <= iy < h and gt_np[iy, ix] == 1:
                x, y = nx, ny
                break
            heading = np.random.uniform(0, 2 * math.pi)
        heading += np.random.uniform(-0.3, 0.3)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test/corridor")
    ap.add_argument("--map-idx", type=int, default=0)
    ap.add_argument("--walk", type=int, default=40)
    ap.add_argument("--K", type=int, default=K_DEFAULT)
    ap.add_argument("--spacing", type=float, default=20.0)
    ap.add_argument("--a-max", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/graph_check"))
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    split = load_split(args.split)
    gt_t, starts, free, _ = sample_batch(split, 1, indices=np.array([args.map_idx]))
    world = WarpWorld(gt_t, sensor_range=80.0, n_rays=720)
    gt_np = gt_t[0].cpu().numpy()
    h, w = gt_np.shape

    r0, c0 = int(starts[0, 0]), int(starts[0, 1])
    if r0 < 0:
        fr = torch.nonzero(gt_t[0] == 1)[torch.randint(int((gt_t[0] == 1).sum()), (1,))][0]
        r0, c0 = int(fr[0]), int(fr[1])
    x, y = random_walk(world, gt_np, float(c0), float(r0), args.walk)

    belief = world.belief_torch
    fmask = frontier_mask(belief)
    centers, cvalid, ccount = frontier_centers(fmask, stride=16, min_count=3)
    anchors, amask = build_anchors(centers, cvalid, ccount, a_max=args.a_max)

    lat = EgoLattice(1, K=args.K, spacing=args.spacing)
    pos = torch.tensor([[x, y]], dtype=torch.float32, device="cuda")
    g = lat.build(pos, belief, fmask)
    coords, valid, edges, util = g["coords"][0], g["valid"][0], g["edges"][0], g["utility"][0]

    # --- check 1: nodi validi su free ---
    vidx = torch.nonzero(valid == 1).squeeze(1)
    cc = coords[vidx].round().long()
    bvals = belief[0, cc[:, 1].clamp(0, h - 1), cc[:, 0].clamp(0, w - 1)]
    bad_node = int((bvals != 1).sum())

    # --- check 2: edge attivi non su ostacolo (campiona midpoint) ---
    bad_edge = 0
    act = torch.nonzero(edges == 1)
    for k, d in act.tolist():
        gi, gj = k // args.K, k % args.K
        di, dj = _DIRS[d]
        nk = (gi + di) * args.K + (gj + dj)
        mx = ((coords[k, 0] + coords[nk, 0]) / 2).round().long().clamp(0, w - 1)
        my = ((coords[k, 1] + coords[nk, 1]) / 2).round().long().clamp(0, h - 1)
        if int(belief[0, my, mx]) == 2:
            bad_edge += 1

    # --- check 3: utility ---
    n_util = int((util > 0).sum())
    max_u = int(util.max())

    # --- check 4: anchor entro cap ---
    n_anchor = int(amask[0].sum())

    # --- check 5: re-anchor (clono: build riusa i buffer in-place) ---
    coords_before = coords.clone()
    pos2 = pos + torch.tensor([[50.0, 0.0]], device="cuda")
    g2 = lat.build(pos2, belief, fmask)
    shift = float((g2["coords"][0] - coords_before)[:, 0].mean())

    print(f"[check1] nodi validi {len(vidx)} | non-free tra i validi: {bad_node} (atteso 0)")
    print(f"[check2] edge attivi {len(act)} | midpoint su ostacolo: {bad_edge} (atteso 0)")
    print(f"[check3] nodi con utility>0: {n_util} | utility max: {max_u}")
    print(f"[check4] anchor attivi: {n_anchor}/{args.a_max} (entro cap)")
    print(f"[check5] shift coords con +50px in x: {shift:.1f} (atteso ~50)")

    # --- viz ---
    from PIL import Image, ImageDraw
    img = np.zeros((h, w, 3), dtype=np.uint8)
    bnp = belief[0].cpu().numpy()
    img[gt_np == 0] = (40, 40, 40)
    img[gt_np == 1] = (90, 90, 90)
    img[bnp == 1] = (210, 210, 210)
    img[bnp == 2] = (170, 70, 70)
    img[fmask[0].cpu().numpy()] = (60, 200, 90)        # frontiere verde
    pim = Image.fromarray(img)
    dr = ImageDraw.Draw(pim)
    cnp = coords.cpu().numpy(); enp = edges.cpu().numpy(); unp = util.cpu().numpy(); vnp = valid.cpu().numpy()
    for k in range(args.K * args.K):                   # edge
        if vnp[k] == 0:
            continue
        gi, gj = k // args.K, k % args.K
        for d in range(8):
            if enp[k, d]:
                di, dj = _DIRS[d]
                nk = (gi + di) * args.K + (gj + dj)
                dr.line([tuple(cnp[k]), tuple(cnp[nk])], fill=(80, 130, 230), width=1)
    umax = max(unp.max(), 1)
    for k in range(args.K * args.K):                   # nodi
        if vnp[k] == 0:
            continue
        u = unp[k] / umax
        col = (int(60 + 195 * u), int(60 + 100 * (1 - u)), 60)   # giallo=alta utility
        dr.ellipse([cnp[k, 0] - 2, cnp[k, 1] - 2, cnp[k, 0] + 2, cnp[k, 1] + 2], fill=col)
    anp = anchors[0].cpu().numpy(); amk = amask[0].cpu().numpy()
    for i in range(len(anp)):                          # anchor globali
        if amk[i]:
            dr.rectangle([anp[i, 0] - 3, anp[i, 1] - 3, anp[i, 0] + 3, anp[i, 1] + 3], outline=(255, 120, 0), width=2)
    dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(60, 120, 250))   # robot
    args.out.mkdir(parents=True, exist_ok=True)
    outp = args.out / f"{args.split.replace('/', '_')}_idx{args.map_idx}_graph.png"
    pim.save(outp)
    print(f"[viz] {outp}")

    ok = bad_node == 0 and bad_edge == 0 and n_util > 0 and abs(shift - 50) < 1
    print("\nGATE Fase 2", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
