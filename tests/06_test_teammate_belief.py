"""Unit test for the uniform expanding-zone teammate belief (env/teammate_belief.py, v3).

Checks: COLLAPSE at comm → single node; while out of comm the zone grows one hop/step (radius = t);
the belief is UNIFORM and sums to 1; the zone expands through a frontier into UNKNOWN nodes.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from env.teammate_belief import update_teammate_belief

dev = "cuda" if torch.cuda.is_available() else "cpu"


def make_line_graph(N, K=8, dev="cpu"):
    """Chain 0-1-2-…-N-1 in a K-slot neighbour table (slot0=left, slot1=right)."""
    nbr = torch.zeros((N, K), dtype=torch.long, device=dev)
    ev = torch.zeros((N, K), dtype=torch.bool, device=dev)
    for i in range(N):
        if i > 0:
            nbr[i, 0] = i - 1; ev[i, 0] = True
        if i < N - 1:
            nbr[i, 1] = i + 1; ev[i, 1] = True
    return nbr, ev


def main():
    N, M, Bf = 21, 2, 1
    nbr, ev1 = make_line_graph(N, dev=dev)
    ev = ev1.unsqueeze(0).expand(Bf, N, nbr.shape[1])
    reached = torch.zeros((Bf, M, N), device=dev)
    team_node = torch.zeros((Bf, M), dtype=torch.long, device=dev)
    team_node[:, 1] = 10   # teammate slot 1 last-known at the middle node

    def step(comm):
        nonlocal reached
        cmask = torch.zeros((Bf, M), dtype=torch.bool, device=dev)
        cmask[:, 1] = comm
        reached, p, alive = update_teammate_belief(
            reached, comm_mask=cmask, team_node=team_node, nbr_idx=nbr,
            edge_valid_optim=ev, expand_per_step=1)
        return p, alive

    ok = True

    # (1) COLLAPSE → single node at 10.
    p, alive = step(comm=True)
    bel = p[0, 1]
    c1 = int((bel > 1e-6).sum()) == 1 and int(bel.argmax()) == 10 and abs(float(bel.sum()) - 1) < 1e-5
    print(f"[1] COLLAPSE → 1 node @ {int(bel.argmax())} (want 10), Σp={float(bel.sum()):.4f}  {'OK' if c1 else 'FAIL'}")
    ok &= c1

    # (2) EXPAND one hop/step OOR → after t steps the zone spans nodes [10-t, 10+t] = 2t+1 nodes,
    #     UNIFORM and Σ=1.
    for t in range(1, 6):
        p, alive = step(comm=False)
    bel = p[0, 1]
    zone = (bel > 1e-6)
    nz = int(zone.sum())
    lo, hi = int(zone.nonzero().min()), int(zone.nonzero().max())
    vals = bel[zone]
    uniform = float(vals.max() - vals.min()) < 1e-6
    c2 = nz == 2 * 5 + 1 and lo == 5 and hi == 15 and uniform and abs(float(bel.sum()) - 1) < 1e-5
    print(f"[2] EXPAND 5 steps → {nz} nodi (want 11), range [{lo},{hi}] (want [5,15]), "
          f"uniforme={uniform}, Σp={float(bel.sum()):.4f}  {'OK' if c2 else 'FAIL'}")
    ok &= c2

    # (3) per-node prob decays as the zone grows (Σ=1 spread over more nodes).
    p1, _ = step(comm=True)          # collapse → 1 node, p=1
    v_start = float(p1[0, 1].max())
    for _ in range(3):
        p2, _ = step(comm=False)
    v_grown = float(p2[0, 1][p2[0, 1] > 1e-6].max())
    c3 = v_start > v_grown and abs(v_grown - 1.0 / 7) < 1e-5   # 3 hops → 7 nodes → 1/7 each
    print(f"[3] prob cala: start={v_start:.3f} → dopo 3 step={v_grown:.3f} (want 1/7≈0.143)  {'OK' if c3 else 'FAIL'}")
    ok &= c3

    print("\nRESULT:", "ALL OK" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
