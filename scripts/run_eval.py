"""Load a checkpoint, run a deterministic episode, save a GIF.

    python scripts/eval.py --ckpt runs/train_default/final.pt \\
        --split test/complex --map-idx 0 --out runs/eval.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import imageio.v2 as imageio
import numpy as np
import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split, sample_batch
from eval.rollout import EvalCfg, EvalRollout
from models.actor_critic import MarlActorCritic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test/complex")
    ap.add_argument("--map-idx", type=int, default=0)
    ap.add_argument("--n-agents", type=int, default=1)
    ap.add_argument("--d-hidden", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--draw-edges", action="store_true", default=True)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/eval.gif"))
    args = ap.parse_args()

    split = load_split(args.split, device=args.device)
    env_cfg = EnvCfg(n_envs=1, n_agents=args.n_agents, max_episode_steps=args.steps + 1)
    env = Explorer(split, env_cfg, seed=int(args.map_idx))
    # Force the fixed map idx.
    gt_new, starts_new, fc_new = sample_batch(split, 1, indices=np.array([args.map_idx]),
                                              seed=0, device=args.device)
    env.world.gt_torch.copy_(gt_new)
    env.world.occupancy_torch.zero_()
    env.world.occupancy_logodds_torch.zero_()
    env.starts.copy_(starts_new)
    env.free_total.copy_(fc_new.float())
    env.visited_step.fill_(-1)
    env.t.zero_()
    env.pos[:, :, 0] = float(starts_new[0, 1])
    env.pos[:, :, 1] = float(starts_new[0, 0])
    env.world.set_positions(env.pos)
    env.world.scan()
    env.last_union.copy_((env.world.occupancy_torch == 1).view(1, -1).float().sum(dim=-1))
    env._refresh_obs()

    model = MarlActorCritic(n_agents=args.n_agents, d=args.d_hidden,
                            n_heads=args.n_heads, n_layers=args.n_layers).to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    sd = ckpt["model"]
    # Strip torch.compile prefix if present.
    sd = {k.replace("encoder._orig_mod.", "encoder."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print(f"[load] {args.ckpt}  iter={ckpt.get('iter', '?')}")

    rollout = EvalRollout(env, model, EvalCfg(max_steps=args.steps, env_idx=0,
                                              deterministic=args.deterministic,
                                              draw_edges=args.draw_edges))
    frames, stats = rollout.run()
    print(f"[rollout] frames={stats['n_frames']}  final_explored={stats['final_explored']*100:.1f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, duration=80, loop=0)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
