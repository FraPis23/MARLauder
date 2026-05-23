"""Carica un checkpoint MAPPO e genera una GIF della policy addestrata (greedy o sampling).

Uso: python scripts/eval_gif.py --ckpt runs/ckpt/mappo_1ag.pth --split test/complex --n-agents 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from env.maps import load_split, sample_batch
from env.marl_env import MarlExploreEnv, EnvConfig
from models.networks import MarlActorCritic
from viz_util import render_marl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test/complex")
    ap.add_argument("--n-agents", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--map-idx", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/gifs"))
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda:0"
    split = load_split(args.split)
    net = MarlActorCritic(K=21, a_max=64).to(dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev)["net"])
    net.eval()

    gt, starts, free, _ = sample_batch(split, 1, indices=np.array([args.map_idx]))
    env = MarlExploreEnv(gt, free, starts,
                         EnvConfig(n_envs=1, n_agents=args.n_agents, max_steps=args.steps + 10))
    obs = env.reset()
    frames = []
    for step in range(args.steps):
        with torch.no_grad():
            action, _, _, _ = net.act(obs, deterministic=args.deterministic)
        obs, r, d, info = env.step(action)
        if step % 2 == 0:
            frames.append(render_marl(obs["belief"][0], obs["frontier_coarse"][0], obs["fscale"],
                                      obs["pos"][0], obs["anchors"][0], obs["anchor_mask"][0]))
    args.out.mkdir(parents=True, exist_ok=True)
    outp = args.out / f"trained_{args.split.replace('/', '_')}_{args.n_agents}ag.gif"
    imageio.mimsave(outp, frames, fps=12)
    print(f"[gif] {outp} | copertura finale {float(info['coverage'][0])*100:.1f}%")


if __name__ == "__main__":
    main()
