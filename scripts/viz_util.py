"""Renderer condiviso per GIF/PNG (stile pulito, PIL)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

# palette
C_UNKNOWN = (28, 28, 34)
C_FREE = (224, 226, 230)
C_OBST = (74, 80, 92)
C_FRONTIER = (70, 200, 110)
AGENT_COLORS = [(70, 130, 250), (240, 80, 80), (90, 210, 120), (240, 200, 60),
                (200, 100, 240), (60, 215, 215), (245, 140, 60), (170, 170, 255)]


def base_image(belief: torch.Tensor, fcoarse: torch.Tensor | None, fscale: int) -> Image.Image:
    b = belief.cpu().numpy()
    h, w = b.shape
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[b == 0] = C_UNKNOWN
    img[b == 1] = C_FREE
    img[b == 2] = C_OBST
    if fcoarse is not None:
        fr = fcoarse.cpu().numpy().repeat(fscale, 0).repeat(fscale, 1)[:h, :w]
        img[fr] = C_FRONTIER
    return Image.fromarray(img)


def draw_agents(im: Image.Image, pos: torch.Tensor, r: int = 6, trails=None) -> None:
    dr = ImageDraw.Draw(im)
    if trails is not None:
        for i, tr in enumerate(trails):
            col = AGENT_COLORS[i % len(AGENT_COLORS)]
            for (tx, ty) in tr:
                dr.ellipse([tx - 1, ty - 1, tx + 1, ty + 1], fill=col)
    pn = pos.cpu().numpy()
    for i, (px, py) in enumerate(pn):
        col = AGENT_COLORS[i % len(AGENT_COLORS)]
        dr.ellipse([px - r, py - r, px + r, py + r], fill=col, outline=(255, 255, 255), width=2)


def draw_anchors(im: Image.Image, anchors: torch.Tensor, mask: torch.Tensor) -> None:
    dr = ImageDraw.Draw(im)
    a = anchors.cpu().numpy(); mk = mask.cpu().numpy()
    for i in range(len(a)):
        if mk[i]:
            dr.ellipse([a[i, 0] - 4, a[i, 1] - 4, a[i, 0] + 4, a[i, 1] + 4],
                       outline=(255, 150, 40), width=2)


def render_marl(belief, fcoarse, fscale, pos, anchors=None, anchor_mask=None,
                trails=None, scale_up: int = 1) -> np.ndarray:
    im = base_image(belief, fcoarse, fscale)
    if anchors is not None:
        draw_anchors(im, anchors, anchor_mask)
    draw_agents(im, pos, trails=trails)
    if scale_up > 1:
        im = im.resize((im.width * scale_up, im.height * scale_up), Image.NEAREST)
    return np.asarray(im)
