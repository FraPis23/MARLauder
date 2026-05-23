"""GATE Fase 3 — env MARL vettorizzato N_env x M_agenti (comms perfette, random policy).

Verifiche:
  1. spawn: agenti nella stessa stanza, su nodi distinti
  2. nessuna collisione inter-agente lungo tutto il rollout
  3. la copertura cresce nel tempo (random policy)
  4. throughput env-step/s
Genera anche una GIF multi-agente (env 0) per validazione visiva.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from env.maps import load_split, sample_batch
from env.marl_env import MarlExploreEnv, EnvConfig, N_ACT
from viz_util import render_marl


def sample_actions(mask: torch.Tensor) -> torch.Tensor:
    n, m, a = mask.shape
    probs = mask.float()
    probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-6)
    return torch.multinomial(probs.reshape(n * m, a), 1).reshape(n, m)


def count_collisions(pos: torch.Tensor) -> int:
    n, m, _ = pos.shape
    p = pos.round()
    diff = (p.unsqueeze(2) - p.unsqueeze(1)).abs().sum(-1)        # [N,M,M]
    eye = torch.eye(m, dtype=torch.bool, device=pos.device).unsqueeze(0)
    return int(((diff < 1.0) & ~eye).sum() // 2)


def frame(obs):
    return render_marl(obs["belief"][0], obs["frontier_coarse"][0], obs["fscale"],
                       obs["pos"][0], obs["anchors"][0], obs["anchor_mask"][0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test/complex")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/gifs"))
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    split = load_split(args.split)
    gt, starts, free, _ = sample_batch(split, args.n_envs)
    cfg = EnvConfig(n_envs=args.n_envs, n_agents=args.n_agents, max_steps=args.steps + 10)
    env = MarlExploreEnv(gt, free, starts, cfg)

    obs = env.reset()

    # check 1: spawn distinti in env 0
    p0 = obs["pos"][0].round()
    dd = (p0.unsqueeze(1) - p0.unsqueeze(0)).abs().sum(-1)
    eye = torch.eye(args.n_agents, dtype=torch.bool, device=p0.device)
    spawn_dups = int(((dd < 1.0) & ~eye).sum() // 2)
    print(f"[check1] spawn env0: {args.n_agents} agenti | nodi coincidenti: {spawn_dups} (atteso 0)")

    cov0 = float(env._explored()[0] / free[0])
    frames = [frame(obs)]
    total_coll = 0

    torch.cuda.synchronize(); t0 = time.perf_counter()
    for step in range(args.steps):
        actions = sample_actions(obs["action_mask"])
        obs, reward, done, info = env.step(actions)
        total_coll += count_collisions(obs["pos"])
        if step % 2 == 0:
            frames.append(frame(obs))
    torch.cuda.synchronize(); dt = time.perf_counter() - t0

    cov = info["coverage"]
    print(f"[check2] collisioni totali nel rollout: {total_coll} (atteso 0)")
    print(f"[check3] copertura env0: {cov0*100:.1f}% -> {float(cov[0])*100:.1f}% "
          f"| media batch {float(cov.mean())*100:.1f}%")
    sps = args.n_envs * args.steps / dt
    print(f"[check4] {args.n_envs} env x {args.steps} step in {dt:.2f}s -> {sps:,.0f} env-step/s "
          f"({args.n_agents} agenti)")

    args.out.mkdir(parents=True, exist_ok=True)
    outp = args.out / f"marl_{args.split.replace('/', '_')}_{args.n_agents}ag.gif"
    imageio.mimsave(outp, frames, fps=12)
    print(f"[gif] {outp} | {len(frames)} frame")

    ok = spawn_dups == 0 and total_coll == 0 and float(cov.mean()) > cov0
    print("\nGATE Fase 3", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
