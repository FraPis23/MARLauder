"""Capture ONE inspector trace under the frozen PROTOCOL v2 conditions (distance budget).

Same conditions as `scripts/eval_comparison.py --ir2-dir ...` produced the published
`*_v2.csv` cells with: per-map travel budget D_k = IR2's own max_dist on that map, IR2's
start positions, `done_mode="own"`, `max_travel_frac=0`, relay pinned, and the step cap
demoted to the cell-wide safety net (so `max_episode_steps` — which the actor OBSERVES —
is the same number the CSV run used).

    python scripts/trace_v2_demo.py --ckpt runs/<run>/ckpt_best.pt --split complex \
        --agents 2 --eps 79 --out runs/demo_v20_complex_v2

Row `--eps i` is map `map_indices_<split>.json["entries"][i]["pack_idx"]`, i.e. the same
episode index as row i of both comparison CSVs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from env.explorer import Explorer
from env.maps import load_split
from eval.ckpt_loader import load_model_from_ckpt
from eval.trace import capture_trace

_COMPARISON_DIR = _REPO / "eval" / "comparison"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="complex", choices=["hybrid", "corridor", "complex"])
    ap.add_argument("--agents", type=int, required=True)
    ap.add_argument("--eps", type=int, required=True, help="row index in map_indices_<split>.json")
    ap.add_argument("--ir2-dir", type=Path,
                    default=Path("/workspace/IR2-Multi-Robot-RL-Exploration/comparison/results"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    M = args.agents
    entries = json.loads((_COMPARISON_DIR / f"map_indices_{args.split}.json").read_text())["entries"]
    pack_idx = int(entries[args.eps]["pack_idx"])

    rows = sorted(csv.DictReader((args.ir2_dir / f"ir2_{args.split}_M{M}.csv").open()),
                  key=lambda r: int(r["eps"]))
    budgets = [float(r["max_dist"]) for r in rows]
    d_k = budgets[args.eps]
    # cell-wide safety net, identical to eval_comparison.py's PROTOCOL v2 branch. It is also the
    # actor's episode-clock normaliser (max_episode_steps is an observation), so it must match.
    cap = int(math.ceil(max(budgets) / 16.0)) * 2
    starts = json.loads((args.ir2_dir / f"starts_{args.split}_M{M}.json").read_text())[str(args.eps)]["starts"]
    if len(starts) != M:
        sys.exit(f"starts for eps {args.eps} have {len(starts)} entries, expected M={M}")

    model, penv = load_model_from_ckpt(args.ckpt, args.device, n_agents=M, verbose=True)
    penv = dict(penv or {})
    penv["comm_relay"] = True        # pinned, as in the published run
    penv["done_mode"] = "own"        # IR2's stopping rule
    penv["max_travel_frac"] = 0.0    # the training budget must NOT come back (PROTOCOL v2 §2)
    penv["max_travel_px"] = float(d_k)   # n_envs=1 → the flat budget IS the per-map D_k

    # capture_trace builds and resets the env itself; the only thing it cannot pass through is the
    # IR2 start override, so inject it here for this process only.
    _orig_reload = Explorer.reload_map

    def _reload_with_starts(self, env_idx, map_idx, start_override=None):
        return _orig_reload(self, env_idx, map_idx, start_override=starts)

    Explorer.reload_map = _reload_with_starts

    split = load_split(f"test/{args.split}", device=args.device)
    tag = args.tag or f"v2_{args.split}_M{M}_eps{args.eps}"
    print(f"[v2] {tag}: map pack_idx={pack_idx} D_k={d_k:.0f}px cap={cap} starts={starts}",
          flush=True)
    try:
        meta = capture_trace(model, split, penv, M, pack_idx, cap - 1, args.out, tag, args.device)
    finally:
        Explorer.reload_map = _orig_reload
    print(f"[v2] done: steps={meta['n_steps']} completed={meta['completed']} "
          f"explored={meta['final_explored']}", flush=True)


if __name__ == "__main__":
    main()
