"""Step 6 test: instantiate MarlActorCritic, push a real obs through, check shapes + grads.

    python scripts/06_test_model_shapes.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split
from models.actor_critic import MarlActorCritic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-agents", type=int, default=1)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    split = load_split(args.split, device=args.device)
    cfg = EnvCfg(n_envs=args.n_envs, n_agents=args.n_agents)
    env = Explorer(split, cfg, seed=0)
    obs = env.obs

    model = MarlActorCritic(n_agents=args.n_agents, d=128, n_heads=4, n_layers=2).to(args.device)
    h_act, h_crit = model.init_hidden(args.n_envs, args.device)
    out = model.act(obs, h_act, h_crit, deterministic=False)
    print("[act]")
    for k, v in out.items():
        print(f"  {k:15s} {tuple(v.shape) if isinstance(v, torch.Tensor) else v}")

    # backward sanity: pretend loss = -value.mean() + logp.mean()
    loss = -out["value"].mean() + out["logp"].mean() - out["entropy"].mean()
    loss.backward()
    n_grad = sum(int(p.grad is not None) for p in model.parameters())
    n_total = sum(1 for _ in model.parameters())
    print(f"[backward] loss={loss.item():.4f}  params_with_grad={n_grad}/{n_total}")

    # Re-evaluate same obs with the action — checks shape parity for PPO update path.
    model.zero_grad()
    h_act2, h_crit2 = model.init_hidden(args.n_envs, args.device)
    ev = model.evaluate(obs, out["action"], h_act2, h_crit2)
    print("[evaluate]")
    for k, v in ev.items():
        print(f"  {k:15s} {tuple(v.shape)}")
    print("OK")


if __name__ == "__main__":
    main()
