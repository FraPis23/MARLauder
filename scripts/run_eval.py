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
    ap.add_argument("--force-full-occupancy-sharing", action="store_true",
                    help="I.2: force persistent map fusion at eval (override ckpt)")
    ap.add_argument("--force-full-pos-sharing", action="store_true",
                    help="I.2: force persistent teammate-pos awareness at eval (override ckpt)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/eval.gif"))
    args = ap.parse_args()

    # Peek at ckpt to recover FULL env cfg (n_hops, top_k, force flags, ...).
    ckpt_peek = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_peek = ckpt_peek.get("cfg", {})
    env_peek = cfg_peek.get("env", {}) if isinstance(cfg_peek, dict) else {}

    # I.2 — CLI overrides let you force sharing at eval even if ckpt was trained without.
    overrides = dict(n_envs=1, n_agents=args.n_agents, max_episode_steps=args.steps + 1)
    if args.force_full_occupancy_sharing:
        overrides["force_full_occupancy_sharing"] = True
    if args.force_full_pos_sharing:
        overrides["force_full_pos_sharing"] = True

    split = load_split(args.split, device=args.device)
    env_cfg = EnvCfg.from_ckpt_dict(env_peek, **overrides)
    env = Explorer(split, env_cfg, seed=int(args.map_idx))
    # G.1 — full reset for specific map. Avoids stale BF cache / strategic features.
    env.reload_map(env_idx=0, map_idx=int(args.map_idx))

    model = MarlActorCritic(n_agents=args.n_agents, d=args.d_hidden,
                            n_heads=args.n_heads, n_layers=args.n_layers).to(args.device)
    ckpt = ckpt_peek  # already loaded above
    sd = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in ckpt["model"].items()}
    # Strip torch.compile prefix if present.
    sd = {k.replace("encoder._orig_mod.", "encoder."): v for k, v in sd.items()}
    # I.3 — old checkpoints store `path_bias`; new model uses `path_bias_learn`. Remap + tolerant.
    if "path_bias" in sd and "path_bias_learn" not in sd:
        sd["path_bias_learn"] = sd.pop("path_bias")
    model.load_state_dict(sd, strict=False)
    # Restore high-level strategic gate + target mode from the training cfg (mutable attrs).
    if isinstance(cfg_peek, dict):
        model.strategic_gate_eps = float(cfg_peek.get("strategic_gate_eps", 0.0))
    model.target_mode = "analytic" if env_peek.get("analytic_target", True) else "learned"
    model.eval()
    print(f"[load] {args.ckpt}  iter={ckpt.get('iter', '?')}  strategic_gate_eps={model.strategic_gate_eps}")

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
