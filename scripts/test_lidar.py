"""GATE Fase 1 — verifica raycasting LIDAR 360 + occupancy grid su GPU.

Carica mappe da uno split, posiziona il robot allo start (o cella free random),
esegue la scansione Warp e:
  - verifica coerenza belief vs ground-truth (nessun free dove c'e ostacolo)
  - calcola la copertura da un singolo scan
  - misura il throughput (scan/s) su N mondi paralleli
  - salva una visualizzazione PNG (gt + belief + robot) per ispezione visiva
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from env.maps import load_split, sample_batch
from env.world_warp import WarpWorld


def resolve_starts(gt: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    """starts (row,col) int32, -1 se assente. Ritorna pos (x=col,y=row) float [N,2].
    Dove assente, sceglie una cella free random."""
    n = gt.shape[0]
    pos = torch.zeros((n, 2), dtype=torch.float32, device=gt.device)
    for i in range(n):
        r, c = int(starts[i, 0]), int(starts[i, 1])
        if r < 0:
            free = torch.nonzero(gt[i] == 1, as_tuple=False)
            sel = free[torch.randint(len(free), (1,), device=gt.device)][0]
            r, c = int(sel[0]), int(sel[1])
        pos[i, 0] = float(c)   # x = col
        pos[i, 1] = float(r)   # y = row
    return pos


def save_viz(gt: torch.Tensor, belief: torch.Tensor, pos: torch.Tensor, out: Path) -> None:
    from PIL import Image
    g = gt.cpu().numpy()
    b = belief.cpu().numpy()
    h, w = g.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[g == 0] = (40, 40, 40)        # ostacolo vero -> grigio scuro
    img[g == 1] = (90, 90, 90)        # free vero non ancora visto
    img[b == 1] = (220, 220, 220)     # free osservato -> chiaro
    img[b == 2] = (200, 60, 60)       # ostacolo osservato -> rosso
    # robot
    px, py = int(pos[0]), int(pos[1])
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            yy, xx = py + dy, px + dx
            if 0 <= yy < h and 0 <= xx < w:
                img[yy, xx] = (60, 120, 240)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test/corridor")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--sensor-range", type=float, default=80.0)
    ap.add_argument("--n-rays", type=int, default=720)
    ap.add_argument("--iters", type=int, default=50, help="ripetizioni per il timing")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/lidar_check"))
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    split = load_split(args.split)
    print(f"split {args.split}: {len(split)} mappe, canvas {split.canvas}")

    gt, starts, free, idx = sample_batch(split, args.n)
    world = WarpWorld(gt, sensor_range=args.sensor_range, n_rays=args.n_rays)
    pos = resolve_starts(gt, starts)
    world.set_positions(pos)

    belief = world.scan()

    # 1) coerenza: nessuna cella marcata FREE dove il gt e ostacolo
    bad = int(((belief == 1) & (gt == 0)).sum())
    # ostacoli osservati devono essere ostacoli veri
    bad_obs = int(((belief == 2) & (gt == 1)).sum())
    print(f"[check] free su ostacolo: {bad} | ostacolo osservato su free: {bad_obs} (attesi 0,0)")

    # 2) copertura single-scan
    seen_free = (belief == 1).sum(dim=(1, 2)).float()
    cov = (seen_free / free.float().clamp(min=1)).mean().item()
    print(f"[cov] copertura media da 1 scan: {cov*100:.1f}% delle celle free")

    # 3) throughput
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        world.reset_belief()
        world.scan()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = args.n * args.iters / dt
    print(f"[perf] {args.n} mondi x {args.iters} scan in {dt:.3f}s -> {sps:,.0f} scan/s "
          f"({args.n_rays} raggi, range {args.sensor_range:.0f}px)")

    # 4) viz mappa 0
    world.reset_belief(); world.scan()
    out = args.out / f"{args.split.replace('/', '_')}_map0.png"
    save_viz(gt[0], world.belief_torch[0], pos[0], out)
    print(f"[viz] salvata {out}")

    ok = (bad == 0 and bad_obs == 0)
    print("\nGATE Fase 1", "PASS" if ok else "FAIL",
          "- raycasting coerente, occupancy grid su GPU funziona." if ok else "- incoerenze belief/gt!")


if __name__ == "__main__":
    main()
