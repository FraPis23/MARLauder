"""Side-by-side PNG strip of the teammate belief BEFORE vs AFTER the frontier-push fixes.

Reads two already-captured traces of the SAME map/seed/policy (so the trajectory is identical and only
the belief differs) and paints, for a handful of steps, every known node as a dot: grey = no belief,
heat = belief probability. The observer is marked with a ring, so you can see whether the mass rides
the exploration front ahead of it (fixed) or is erased and re-drawn elsewhere (old).

    python scripts/viz_belief_push_compare.py --old runs/A/traces/<tag> --new runs/B/traces/<tag> \
        --agent 1 --steps 92,96,100,104,108 --out runs/B/push_compare.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

BEL_VMAX = 0.5          # fixed colour scale: p ≥ this is full red
DOT = 3                 # node dot radius (px, in output scale)


def heat(v: float) -> tuple[int, int, int]:
    """black→yellow→red ramp on a FIXED scale, matching the web inspector's belief field."""
    t = max(0.0, min(1.0, v / BEL_VMAX))
    if t <= 0.0:
        return (58, 62, 74)                                   # grey: known node, no belief
    if t < 0.5:
        u = t / 0.5
        return (int(40 + 215 * u), int(40 + 200 * u), 30)     # dark → yellow
    u = (t - 0.5) / 0.5
    return (255, int(240 - 200 * u), int(30 + 20 * u))        # yellow → red


def panel(step: dict, agent: int, bbox, scale: float, size) -> Image.Image:
    x0, y0 = bbox[0], bbox[1]
    img = Image.new("RGB", size, (18, 20, 26))
    d = ImageDraw.Draw(img)
    a = step["agents"][agent]
    nodes = sorted(a["nodes"], key=lambda n: n.get("bel", 0.0))   # belief drawn last = on top
    for n in nodes:
        px = (n["x"] - x0) * scale
        py = (n["y"] - y0) * scale
        b = n.get("bel", 0.0) or 0.0
        r = DOT + (2 if b > 1e-4 else 0)
        d.ellipse([px - r, py - r, px + r, py + r], fill=heat(b))
    ax = (a["pos"][0] - x0) * scale
    ay = (a["pos"][1] - y0) * scale
    d.ellipse([ax - 7, ay - 7, ax + 7, ay + 7], outline=(80, 220, 255), width=3)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True, help="trace dir of the OLD behaviour")
    ap.add_argument("--new", type=Path, required=True, help="trace dir of the FIXED behaviour")
    ap.add_argument("--agent", type=int, default=1)
    ap.add_argument("--steps", default="92,96,100,104,108")
    ap.add_argument("--cell", type=int, default=300, help="panel size in px")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    tr = {k: json.loads((p / "trace.json").read_text())
          for k, p in (("old", args.old), ("new", args.new))}

    # Common bbox over both traces (identical map, but be safe) so panels are directly comparable.
    xs, ys = [], []
    for t in tr.values():
        for st in t["steps"]:
            for n in st["agents"][args.agent]["nodes"]:
                xs.append(n["x"]); ys.append(n["y"])
    pad = 20.0
    bbox = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    scale = args.cell / max(w, h)
    size = (int(w * scale), int(h * scale))

    lab_h = 26
    sheet = Image.new("RGB", (size[0] * len(steps), (size[1] + lab_h) * 2), (10, 11, 15))
    dr = ImageDraw.Draw(sheet)
    for row, key in enumerate(("old", "new")):
        for col, ti in enumerate(steps):
            st = tr[key]["steps"][ti]
            p = panel(st, args.agent, bbox, scale, size)
            ox, oy = col * size[0], row * (size[1] + lab_h) + lab_h
            sheet.paste(p, (ox, oy))
            tag = ("PRIMA" if key == "old" else "DOPO ") + f"  t={st['t']}"
            dr.text((ox + 8, oy - lab_h + 6), tag, fill=(200, 210, 225))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"[viz] wrote {args.out}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
