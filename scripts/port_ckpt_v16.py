#!/usr/bin/env python3
"""Port a pre-v16 checkpoint into the v16 actor layout, losslessly.

v16 changed the actor trunk input in two ways at once:
  * agent_scalars grew 2 -> 5 ([g, staleness] -> [g, staleness, travel_frac, contact, offer_frac])
  * the concat order moved agent_scalars to the END, so future width bumps stay warm-start-safe

    pre-v16   actor_pre.weight [d, d + K + 2 + K]   = [curr_emb | prev_action | agent_scalars | vf]
    v16       actor_pre.weight [d, d + K + K + 5]   = [curr_emb | prev_action | vf | agent_scalars]

Every generic loader in the repo handles this by DROPPING actor_pre.weight (eval/ckpt_loader.py) or
by copying the old columns into the leading ones (train/driver.py's widening path). Both are wrong
here for the same reason: the blocks moved, so leading-column copy lands the old `vf` weights on the
new scalar slots, and dropping leaves the actor trunk randomly initialized while the log still says
"warm-started". A probe or eval run against such a checkpoint measures noise — which is exactly what
happened when the v16 budget sweep reported termination rates of 0.16 and 0.00 for identical settings.

The layouts are both fully known, so the map is exact rather than heuristic. The three NEW scalar
columns are zero-initialized, which makes them precise no-ops: the ported model reproduces the
original policy bit-for-bit until those inputs are trained on.

    python scripts/port_ckpt_v16.py --in runs/v15_.../ckpt_best.pt --out /tmp/v15_ported.pt

This is a compatibility shim for MEASUREMENT (probing, eval, comparison arms against older runs).
It is NOT the way to start v16 training: the plan is a from-scratch phase 1, and porting a v15
policy in would carry over exactly the behaviour v16 exists to change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from models.actor_critic import AGENT_SCALAR_DIM, K  # noqa: E402

OLD_SCALARS = 2   # [g, staleness]


def port_actor_pre(w_old: torch.Tensor, d: int) -> torch.Tensor:
    """[d, d+K+2+K] (…|scalars|vf) -> [d, d+K+K+AGENT_SCALAR_DIM] (…|vf|scalars)."""
    head = d + K                                   # curr_emb || prev_action — unmoved
    exp_old = head + OLD_SCALARS + K
    if w_old.shape[1] != exp_old:
        raise SystemExit(f"actor_pre.weight is {tuple(w_old.shape)}, expected [*, {exp_old}] for a "
                         f"pre-v16 checkpoint with d={d}, K={K}. Already ported?")
    w_new = w_old.new_zeros(w_old.shape[0], head + K + AGENT_SCALAR_DIM)
    w_new[:, :head] = w_old[:, :head]                                    # curr_emb || prev_action
    w_new[:, head:head + K] = w_old[:, head + OLD_SCALARS:exp_old]       # vf, moved earlier
    w_new[:, head + K:head + K + OLD_SCALARS] = w_old[:, head:head + OLD_SCALARS]   # g, staleness
    # remaining AGENT_SCALAR_DIM-OLD_SCALARS columns stay 0 → the new scalars start as no-ops
    return w_new


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ck["model"]
    key = next((k for k in sd if k.endswith("actor_pre.weight")), None)
    if key is None:
        raise SystemExit("no actor_pre.weight in this checkpoint")
    w_old = sd[key]
    d = w_old.shape[0]
    sd[key] = port_actor_pre(w_old, d)
    print(f"[port] {key}: {tuple(w_old.shape)} -> {tuple(sd[key].shape)}  (d={d}, K={K}, "
          f"scalars {OLD_SCALARS}->{AGENT_SCALAR_DIM}, new columns zeroed)")
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.dst)
    print(f"[port] wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
