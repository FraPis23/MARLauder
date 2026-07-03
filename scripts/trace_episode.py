"""Capture a per-step decision trace of one episode → web inspector data (JSON + map PNGs).

    python scripts/trace_episode.py --ckpt runs/run/ckpt_020.pt --split test/hybrid \\
        --map-idx 167 --steps 120 --out runs/run

Writes runs/run/traces/<tag>/ + traces/index.json + inspector.html.
In Docker the web server is already running on :8080; just open:
    http://localhost:8080/run/inspector.html
Outside Docker: cd runs/run && python -m http.server 8080
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
from eval.ckpt_loader import load_model_from_ckpt
from eval.trace import capture_trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test/hybrid")
    ap.add_argument("--map-idx", type=int, default=0)
    ap.add_argument("--n-agents", type=int, default=None, help="default: from ckpt cfg, else 2")
    ap.add_argument("--steps", type=int, default=120)
    # Architecture args default to None → auto-detected from the checkpoint so the traced model
    # EXACTLY matches the trained one (critical: a layer/head/dim mismatch loads wrong weights and
    # every number in the trace, attention included, becomes fiction). CLI value overrides.
    ap.add_argument("--d-hidden", type=int, default=None, help="default: from ckpt")
    ap.add_argument("--n-heads", type=int, default=None, help="default: from ckpt")
    ap.add_argument("--n-layers", type=int, default=None, help="default: from ckpt")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/trace_run"))
    ap.add_argument("--tag", default=None, help="trace name (default ckpt+map)")
    ap.add_argument("--comm-gated-pos", action="store_true",
                    help="Override force_full_pos_sharing=False so teammate last-known pos freezes "
                         "between comm contacts (real behavior, not the debug real-time pos)")
    args = ap.parse_args()

    model, env_peek = load_model_from_ckpt(args.ckpt, args.device, n_agents=args.n_agents,
                                           d_hidden=args.d_hidden, n_heads=args.n_heads,
                                           n_layers=args.n_layers)
    n_agents = int(model.M)

    if args.comm_gated_pos and isinstance(env_peek, dict):
        env_peek = {**env_peek, "force_full_pos_sharing": False}

    split = load_split(args.split, device=args.device)
    tag = args.tag or f"{args.ckpt.stem}_m{args.map_idx}"
    capture_trace(model, split, env_peek, n_agents, args.map_idx,
                  args.steps, args.out, tag, args.device)
    print(f"[view] http://localhost:8080/{args.out.name}/inspector.html  (Docker already serves runs/ on :8080)")


if __name__ == "__main__":
    main()
