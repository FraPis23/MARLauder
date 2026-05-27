"""Shared rendering primitives — used by step tests and the eval GIF.

Occupancy is shaded by probability (sigmoid(log-odds)):
    p ≈ 0.0  → dark navy        (confidently OCCUPIED — observed wall)
    p ≈ 0.5  → very dark gray   (UNKNOWN)
    p ≈ 1.0  → muted gray       (confidently FREE — observed open)

Walls are NOT red, so they cannot be confused with frontiers.
Frontier cells appear as a soft red tint on top of the FREE-shaded background.
Guidepost target = amber ring; the path leading to it = amber polyline.

Public API
----------
    shade_occupancy_prob(prob_np) -> RGB uint8 [H, W, 3]
    overlay_gt_hint(rgb, gt, prob) -> RGB uint8 (faint outline of true walls under UNK)
    paint_frontier(rgb, frontier_np)
    paint_path(im, path_xy, path_valid, color, width)
    paint_target(im, target_xy, color, ring)
    paint_graph(im, nxy, nv, util, curr, draw_edges=False, eidx=None, evalid=None)
    paint_agent(im, xy, trail)
    composite_frame(...) -> Image
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

# Palette
C_OCC_DEEP = np.array([24, 26, 34], dtype=np.float32)
C_UNK = np.array([18, 18, 24], dtype=np.float32)
C_FREE_LIGHT = np.array([120, 128, 140], dtype=np.float32)
C_GT_WALL_HINT = np.array([42, 46, 56], dtype=np.uint8)
C_FRONTIER = (255, 90, 90)
C_AGENT = (90, 160, 255)
C_NODE_ACTIVE = (60, 220, 240)
C_NODE_DEAD = (70, 78, 90)
C_NODE_CURR = (255, 215, 60)
C_EDGE = (90, 100, 115)
C_TRAIL = (150, 200, 255)
C_PATH = (255, 180, 40)        # amber polyline
C_TARGET = (255, 230, 60)      # bright amber ring
C_COMM_LINK = (80, 240, 120)   # green comm line between agents
# Per-agent colors (index into this list)
C_AGENTS = [(90, 160, 255), (90, 220, 100), (255, 140, 60), (220, 90, 255)]
C_TRAILS = [(150, 200, 255), (150, 240, 170), (255, 200, 130), (240, 170, 255)]


def shade_occupancy_prob(prob: np.ndarray) -> np.ndarray:
    """prob [H, W] in [0, 1] -> RGB uint8 [H, W, 3]."""
    p = np.clip(prob, 0.0, 1.0).astype(np.float32)
    out = np.empty((*p.shape, 3), dtype=np.float32)
    mask_low = p < 0.5
    t_low = (p * 2.0)[..., None]
    t_high = ((p - 0.5) * 2.0)[..., None]
    out_low = C_OCC_DEEP * (1 - t_low) + C_UNK * t_low
    out_high = C_UNK * (1 - t_high) + C_FREE_LIGHT * t_high
    out[mask_low] = out_low[mask_low]
    out[~mask_low] = out_high[~mask_low]
    return out.clip(0, 255).astype(np.uint8)


def overlay_gt_hint(rgb: np.ndarray, gt: np.ndarray, prob: np.ndarray, alpha: float = 0.18) -> np.ndarray:
    is_wall = gt == 0
    is_unk = (prob > 0.4) & (prob < 0.6)
    mask = (is_wall & is_unk)[..., None]
    rgb = rgb.astype(np.float32)
    rgb = np.where(mask, rgb * (1 - alpha) + C_GT_WALL_HINT.astype(np.float32) * alpha, rgb)
    return rgb.clip(0, 255).astype(np.uint8)


def paint_frontier(rgb: np.ndarray, frontier: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    tint = np.array(C_FRONTIER, dtype=np.float32)
    mask = frontier[..., None].astype(np.float32)
    rgb = rgb * (1 - mask * alpha) + tint * (mask * alpha)
    return rgb.clip(0, 255).astype(np.uint8)


def paint_path(
    im: Image.Image,
    path_xy: np.ndarray | None,
    path_valid: np.ndarray | None,
    color: tuple[int, int, int] = C_PATH,
    width: int = 3,
) -> Image.Image:
    """Draw the guidepost polyline. path_xy [P, 2], path_valid [P] bool.
    Path goes target → ... → curr; valid entries are drawn as a polyline."""
    if path_xy is None or path_valid is None:
        return im
    pts = [(float(path_xy[p, 0]), float(path_xy[p, 1])) for p in range(path_xy.shape[0]) if bool(path_valid[p])]
    if len(pts) >= 2:
        ImageDraw.Draw(im).line(pts, fill=color, width=width)
    return im


def paint_target(
    im: Image.Image,
    target_xy: tuple[float, float] | None,
    color: tuple[int, int, int] = C_TARGET,
    ring: int = 9,
) -> Image.Image:
    if target_xy is None:
        return im
    x, y = target_xy
    dr = ImageDraw.Draw(im)
    dr.ellipse([x - ring, y - ring, x + ring, y + ring], outline=color, width=3)
    dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
    return im


def paint_graph(
    im: Image.Image,
    nxy: np.ndarray,
    nv: np.ndarray,
    util: np.ndarray,
    curr: int,
    draw_edges: bool = False,
    eidx: np.ndarray | None = None,
    evalid: np.ndarray | None = None,
) -> Image.Image:
    dr = ImageDraw.Draw(im)
    if draw_edges and eidx is not None and evalid is not None:
        for k_node in range(nxy.shape[0]):
            if not nv[k_node]:
                continue
            x0, y0 = float(nxy[k_node, 0]), float(nxy[k_node, 1])
            for kk in range(8):
                if not evalid[k_node, kk]:
                    continue
                tgt = int(eidx[k_node, kk])
                if tgt < k_node:
                    continue
                x1, y1 = float(nxy[tgt, 0]), float(nxy[tgt, 1])
                dr.line([(x0, y0), (x1, y1)], fill=C_EDGE, width=1)
    for k_node in range(nxy.shape[0]):
        x, y = float(nxy[k_node, 0]), float(nxy[k_node, 1])
        if not nv[k_node]:
            continue
        u = float(util[k_node])
        col = (
            int(C_NODE_ACTIVE[0] * (1 - u) + 255 * u),
            int(C_NODE_ACTIVE[1] * (1 - u) + 140 * u),
            int(C_NODE_ACTIVE[2] * (1 - u) + 50 * u),
        )
        dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col, outline=(0, 0, 0))
    cx, cy = float(nxy[curr, 0]), float(nxy[curr, 1])
    dr.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=C_NODE_CURR, width=2)
    return im


def paint_agent(
    im: Image.Image,
    xy: tuple[float, float],
    trail: Iterable[tuple[float, float]] | None = None,
    color: tuple[int, int, int] = C_AGENTS[0],
    trail_color: tuple[int, int, int] = C_TRAILS[0],
) -> Image.Image:
    dr = ImageDraw.Draw(im)
    if trail is not None:
        trail = list(trail)
        if len(trail) > 1:
            dr.line(trail, fill=trail_color, width=2)
    x, y = xy
    dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color, outline=(255, 255, 255), width=1)
    return im


def paint_comm_link(
    im: Image.Image,
    xy0: tuple[float, float],
    xy1: tuple[float, float],
    color: tuple[int, int, int] = C_COMM_LINK,
    width: int = 2,
) -> Image.Image:
    """Draw line between two agents when they are in communication range."""
    ImageDraw.Draw(im).line([xy0, xy1], fill=color, width=width)
    return im


def composite_frame(
    prob: np.ndarray,
    gt: np.ndarray,
    frontier: np.ndarray,
    nxy: np.ndarray,
    nv: np.ndarray,
    util: np.ndarray,
    curr: int,
    agent_xy: tuple[float, float],
    trail: list[tuple[float, float]] | None,
    step: int,
    explored: float,
    draw_edges: bool = False,
    eidx: np.ndarray | None = None,
    evalid: np.ndarray | None = None,
    path_xy: np.ndarray | None = None,
    path_valid: np.ndarray | None = None,
    target_xy: tuple[float, float] | None = None,
    # Multi-agent extras (optional)
    extra_agents_xy: list[tuple[float, float]] | None = None,
    extra_agents_trails: list[list[tuple[float, float]]] | None = None,
    extra_agent_indices: list[int] | None = None,   # absolute agent indices for colors
    comm_links: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    agent_idx: int = 0,          # which agent this panel belongs to
    agent_label: str = "",       # displayed in text bar
) -> Image.Image:
    rgb = shade_occupancy_prob(prob)
    rgb = paint_frontier(rgb, frontier)
    im = Image.fromarray(rgb)
    # draw order: edges → comm links → path → nodes → target → agents → text
    paint_graph(im, nxy, nv, util, curr, draw_edges, eidx, evalid)
    if comm_links:
        for xy0, xy1 in comm_links:
            paint_comm_link(im, xy0, xy1)
    paint_path(im, path_xy, path_valid)
    paint_target(im, target_xy)
    # Extra agents drawn first (behind main agent)
    if extra_agents_xy:
        for ag_i, axy in enumerate(extra_agents_xy):
            atrail = extra_agents_trails[ag_i] if extra_agents_trails else None
            abs_idx = extra_agent_indices[ag_i] if extra_agent_indices else (ag_i + 1)
            paint_agent(im, axy, atrail,
                        color=C_AGENTS[abs_idx % len(C_AGENTS)],
                        trail_color=C_TRAILS[abs_idx % len(C_TRAILS)])
    paint_agent(im, agent_xy, trail,
                color=C_AGENTS[agent_idx % len(C_AGENTS)],
                trail_color=C_TRAILS[agent_idx % len(C_TRAILS)])
    dr = ImageDraw.Draw(im)
    H, W = prob.shape
    dr.rectangle([(0, 0), (W, 16)], fill=(0, 0, 0))
    label = f"[{agent_label}] " if agent_label else ""
    dr.text((4, 2), f"{label}t={step}  explored={explored * 100:.1f}%", fill=(255, 255, 255))
    return im


def hstack_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-agent frames horizontally into one wide image."""
    return np.concatenate(frames, axis=1)


# Back-compat alias for old callers.
shade_belief_prob = shade_occupancy_prob
