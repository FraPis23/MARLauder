"""Capture a per-step decision trace of one episode → web inspector data (JSON + map PNGs).

    python scripts/trace_episode.py --ckpt runs/run/ckpt_020.pt --split test/hybrid \\
        --map-idx 167 --steps 120 --out runs/run

Writes runs/run/traces/<tag>/ + traces/index.json + inspector.html. Then:
    cd runs/run && python -m http.server 8000  → http://localhost:8000/inspector.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from env.maps import load_split
from eval.trace import capture_trace
from models.actor_critic import MarlActorCritic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test/hybrid")
    ap.add_argument("--map-idx", type=int, default=0)
    ap.add_argument("--n-agents", type=int, default=2)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--d-hidden", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/trace_run"))
    ap.add_argument("--tag", default=None, help="trace name (default ckpt+map)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_peek = ckpt.get("cfg", {})
    env_peek = cfg_peek.get("env", {}) if isinstance(cfg_peek, dict) else {}

    model = MarlActorCritic(n_agents=args.n_agents, d=args.d_hidden,
                            n_heads=args.n_heads, n_layers=args.n_layers).to(args.device)
    sd = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in ckpt["model"].items()}
    sd = {k.replace("encoder._orig_mod.", "encoder."): v for k, v in sd.items()}
    if "path_bias" in sd and "path_bias_learn" not in sd:
        sd["path_bias_learn"] = sd.pop("path_bias")
    msd = model.state_dict()
    for k in [k for k in sd if k in msd and msd[k].shape != sd[k].shape]:
        del sd[k]
    model.load_state_dict(sd, strict=False)
    if isinstance(cfg_peek, dict):
        model.strategic_gate_eps = float(cfg_peek.get("strategic_gate_eps", 0.0))
    model.target_mode = "analytic" if env_peek.get("analytic_target", True) else "learned"

    split = load_split(args.split, device=args.device)
    tag = args.tag or f"{args.ckpt.stem}_m{args.map_idx}"
    capture_trace(model, split, env_peek, args.n_agents, args.map_idx,
                  args.steps, args.out, tag, args.device)
    print(f"[view] cd {args.out} && python -m http.server 8000  → http://localhost:8000/inspector.html")


if __name__ == "__main__":
    main()
