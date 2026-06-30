"""Standalone evaluation of a SAVED checkpoint → GIFs + step-through traces (web inspector).

Runs independently of training (works on stopped/done runs too). For each of N evenly-spaced
maps on the chosen split it renders a GIF and captures an inspector trace into the run dir, so
the inspector unlocks. Driven by the web dashboard's "Evaluate checkpoint" button.

    python scripts/eval_ckpt.py --ckpt runs/run/ckpt_stop.pt --split test/hybrid \\
        --n-maps 3 --out runs/run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split
from eval.rollout import EvalCfg, EvalRollout
from eval.trace import capture_trace
from models.actor_critic import MarlActorCritic


def _load_model(ckpt_path: Path, n_agents: int, d_hidden: int, n_heads: int,
                n_layers: int, device: str):
    """Mirror trace_episode.py's tolerant checkpoint loader."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_peek = ckpt.get("cfg", {})
    env_peek = cfg_peek.get("env", {}) if isinstance(cfg_peek, dict) else {}
    model = MarlActorCritic(n_agents=n_agents, d=d_hidden,
                            n_heads=n_heads, n_layers=n_layers).to(device)
    sd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ckpt["model"].items()}
    sd = {k.replace("encoder._orig_mod.", "encoder."): v for k, v in sd.items()}
    if "path_bias" in sd and "path_bias_learn" not in sd:
        sd["path_bias_learn"] = sd.pop("path_bias")
    msd = model.state_dict()
    for k in [k for k in sd if k in msd and msd[k].shape != sd[k].shape]:
        del sd[k]
    model.load_state_dict(sd, strict=False)
    if isinstance(cfg_peek, dict):
        model.strategic_gate_eps = float(cfg_peek.get("strategic_gate_eps", 0.0))
    model.eval()
    return model, env_peek


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test/hybrid")
    ap.add_argument("--n-maps", type=int, default=3, help="how many evenly-spaced maps to evaluate")
    ap.add_argument("--n-agents", type=int, default=2)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--d-hidden", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True, help="run dir to write GIFs + traces into")
    args = ap.parse_args()

    model, env_peek = _load_model(args.ckpt, args.n_agents, args.d_hidden,
                                  args.n_heads, args.n_layers, args.device)
    split = load_split(args.split, device=args.device)
    n = int(getattr(split, "n", 0)) or 1
    k = max(1, min(args.n_maps, n))
    idxs = [int(i) for i in np.linspace(0, n - 1, k).round().astype(int)]
    short = args.split.split("/")[-1]
    stem = args.ckpt.stem

    env_cfg = EnvCfg.from_ckpt_dict(env_peek or {}, n_envs=1, n_agents=args.n_agents,
                                    max_episode_steps=args.steps + 1)
    env = Explorer(split, env_cfg, seed=0)

    print(f"[eval_ckpt] ckpt={stem} split={args.split} maps={idxs} steps={args.steps}")
    import imageio.v2 as imageio
    for gi, midx in enumerate(idxs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tag = f"{stem}_{short}_m{gi}"
        # GIF
        try:
            env.reload_map(env_idx=0, map_idx=int(midx))
            roll = EvalRollout(env, model, EvalCfg(max_steps=args.steps, env_idx=0,
                                                   deterministic=True, draw_edges=True))
            frames, stats = roll.run()
            gif = args.out / f"eval_{tag}.gif"
            imageio.mimsave(gif, frames, duration=80, loop=0)
            print(f"[eval_ckpt] gif {gif.name} explored={stats['final_explored']*100:.1f}%")
        except Exception as exc:
            print(f"[eval_ckpt] gif {tag} skipped ({exc})")
        # Inspector trace (also copies inspector.html into the run dir)
        try:
            capture_trace(model, split, env_peek or {}, args.n_agents,
                          int(midx), args.steps, args.out, tag, args.device)
        except Exception as exc:
            print(f"[eval_ckpt] trace {tag} skipped ({exc})")
    print(f"[eval_ckpt] DONE → http://localhost:8080/{args.out.name}/inspector.html")


if __name__ == "__main__":
    main()
