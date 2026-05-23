"""Primo riscontro visivo: robot in random-walk con belief LIDAR che si accumula.

Nessuna logica/RL. Serve solo a vedere l'ambiente in movimento e la mappa che si scopre.
Output GIF in runs/gifs/.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from env.maps import load_split, sample_batch
from env.world_warp import WarpWorld


def pick_start(gt: torch.Tensor, start: torch.Tensor) -> tuple[float, float]:
    r, c = int(start[0]), int(start[1])
    if r < 0:
        free = torch.nonzero(gt == 1, as_tuple=False)
        sel = free[torch.randint(len(free), (1,))][0]
        r, c = int(sel[0]), int(sel[1])
    return float(c), float(r)   # (x, y)


def render(gt_np, belief_np, trail, pos, fr) -> np.ndarray:
    h, w = gt_np.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[gt_np == 0] = (40, 40, 40)
    img[gt_np == 1] = (95, 95, 95)
    img[belief_np == 1] = (225, 225, 225)
    img[belief_np == 2] = (200, 70, 70)
    for (tx, ty) in trail:
        if 0 <= ty < h and 0 <= tx < w:
            img[ty, tx] = (70, 160, 250)
    px, py = int(pos[0]), int(pos[1])
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            yy, xx = py + dy, px + dx
            if 0 <= yy < h and 0 <= xx < w:
                img[yy, xx] = (60, 120, 240)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test/corridor")
    ap.add_argument("--map-idx", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--step-size", type=float, default=6.0)
    ap.add_argument("--sensor-range", type=float, default=80.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/gifs"))
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    split = load_split(args.split)
    gt_t, starts, free, _ = sample_batch(split, 1, indices=np.array([args.map_idx]))
    world = WarpWorld(gt_t, sensor_range=args.sensor_range, n_rays=720)

    x, y = pick_start(gt_t[0], starts[0])
    heading = np.random.uniform(0, 2 * math.pi)
    gt_np = gt_t[0].cpu().numpy()
    h, w = gt_np.shape

    def is_free(xx, yy) -> bool:
        ix, iy = int(xx + 0.5), int(yy + 0.5)
        return 0 <= ix < w and 0 <= iy < h and gt_np[iy, ix] == 1

    trail: list[tuple[int, int]] = []
    frames = []
    for step in range(args.steps):
        world.set_positions(torch.tensor([[x, y]], dtype=torch.float32, device="cuda"))
        world.scan()
        trail.append((int(x), int(y)))

        # random walk con momentum: prova ad avanzare, se muro ruota
        moved = False
        for _ in range(8):
            nx = x + math.cos(heading) * args.step_size
            ny = y + math.sin(heading) * args.step_size
            if is_free(nx, ny):
                x, y = nx, ny
                moved = True
                break
            heading = np.random.uniform(0, 2 * math.pi)
        heading += np.random.uniform(-0.3, 0.3)   # piccola deriva
        if not moved:
            heading = np.random.uniform(0, 2 * math.pi)

        if step % 2 == 0:   # 1 frame ogni 2 step (gif piu leggera)
            frames.append(render(gt_np, world.belief_torch[0].cpu().numpy(), trail, (x, y), step))

    seen = int((world.belief_torch[0] == 1).sum())
    cov = seen / max(int(free[0]), 1)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.split.replace('/', '_')}_idx{args.map_idx}_walk.gif"
    imageio.mimsave(out, frames, fps=args.fps)
    print(f"[gif] {out} | {len(frames)} frame | copertura finale {cov*100:.1f}%")


if __name__ == "__main__":
    main()
