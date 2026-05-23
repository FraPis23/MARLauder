"""GATE Fase 4 — reti actor-pointer + critic permutation-invariant.

Verifiche:
  1. forward: shape logits [N,M,9], value [N,M]
  2. masking: azioni campionate sempre dentro action_mask; logit mascherati = -inf
  3. backward: gradiente finito sui parametri
  4. M-agnostico: stessa rete gira con M=3 e M=6
Genera una GIF con la policy (NON addestrata, solo wiring) per validazione visiva.
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


def make_env(split, N, M):
    gt, starts, free, _ = sample_batch(split, N)
    return MarlExploreEnv(gt, free, starts, EnvConfig(n_envs=N, n_agents=M, max_steps=300))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test/complex")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/gifs"))
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda:0"
    split = load_split(args.split)

    net = MarlActorCritic(K=21, a_max=64).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[net] parametri: {n_params/1e6:.2f}M")

    env = make_env(split, args.n_envs, args.n_agents)
    obs = env.reset()

    # 1) forward
    logits, value = net(obs)
    print(f"[check1] logits {tuple(logits.shape)} | value {tuple(value.shape)} "
          f"(atteso [{args.n_envs},{args.n_agents},9] e [{args.n_envs},{args.n_agents}])")

    # 2) masking
    action, logp, ent, val = net.act(obs)
    mask = obs["action_mask"].bool()
    chosen_ok = bool(mask.gather(-1, action.unsqueeze(-1)).all())
    masked_neg_inf = bool(torch.isinf(logits[~mask]).all() and (logits[~mask] < 0).all())
    print(f"[check2] azioni dentro mask: {chosen_ok} | logit mascherati = -inf: {masked_neg_inf}")

    # 3) backward
    loss = -(logp.mean()) + val.pow(2).mean() - 0.01 * ent.mean()
    net.zero_grad(); loss.backward()
    g = net.embed.weight.grad
    grad_ok = g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0
    print(f"[check3] gradiente finito e non nullo: {bool(grad_ok)}")

    # 4) M-agnostico: stessa rete, M diversi
    ok_m = True
    for M in (3, 6):
        e = make_env(split, 4, M)
        o = e.reset()
        with torch.no_grad():
            lg, v = net(o)
        ok_m &= tuple(lg.shape) == (4, M, 9) and tuple(v.shape) == (4, M)
    print(f"[check4] stessa rete con M=3 e M=6: {ok_m}")

    # GIF (policy non addestrata)
    frames = []
    obs = env.reset()
    for step in range(args.steps):
        with torch.no_grad():
            action, _, _, _ = net.act(obs)
        obs, r, d, info = env.step(action)
        if step % 2 == 0:
            frames.append(render_marl(obs["belief"][0], obs["frontier_coarse"][0], obs["fscale"],
                                      obs["pos"][0], obs["anchors"][0], obs["anchor_mask"][0]))
    args.out.mkdir(parents=True, exist_ok=True)
    outp = args.out / f"net_{args.split.replace('/', '_')}_{args.n_agents}ag.gif"
    imageio.mimsave(outp, frames, fps=12)
    print(f"[gif] {outp} | copertura batch {float(info['coverage'].mean())*100:.1f}%")

    ok = (tuple(logits.shape) == (args.n_envs, args.n_agents, 9) and chosen_ok
          and masked_neg_inf and grad_ok and ok_m)
    print("\nGATE Fase 4", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
