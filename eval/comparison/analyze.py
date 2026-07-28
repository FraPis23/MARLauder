"""Aggregate the MARLauder-vs-IR2 comparison: per-cell tables + paired Wilcoxon, from the CSVs.

Reads IR2's results (../../../IR2-Multi-Robot-RL-Exploration/comparison/results/ir2_{split}_M{M}.csv,
produced 2026-07-14) and MARLauder's (results/marlauder_{split}_M{M}[_{tag}].csv, produced by
scripts/eval_comparison.py) and reports them side by side.

Two rules from the frozen protocol, enforced here rather than left to the reader:

  * NEVER average across splits. hybrid, corridor and complex are different problems with
    different episode caps (196/196/384); a grand mean over them is not a quantity.
  * The test is PAIRED. Row i of both CSVs is the same map — eval_comparison.py writes rows in
    map_indices_{split}.json order and IR2 was driven with Env(map_index=episode) over the same
    100-file list — so a Wilcoxon signed-rank on the per-map differences is available, and it is
    far more sensitive than comparing two means with std ±30%.

`max_dist` is the headline: `steps` is NOT comparable between the systems (an IR2 step is a
waypoint decision plus its A* traverse; a MARLauder step is one ≤NR·√2 px lattice hop), while
metres of robot travel are the same physical quantity on both sides.

    python eval/comparison/analyze.py [--tag v10]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_MARL_RESULTS = _HERE / "results"
_IR2_RESULTS = _HERE.parent.parent.parent / "IR2-Multi-Robot-RL-Exploration" / "comparison" / "results"

SPLITS = ["hybrid", "corridor", "complex"]
AGENTS = [2, 4]


def _read(path: Path) -> dict[str, np.ndarray]:
    """CSV → column arrays. IR2 writes booleans as 'True'/'False', we write 1/0 — both land as float."""
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    out: dict[str, np.ndarray] = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            v = r[k]
            vals.append(1.0 if v == "True" else 0.0 if v == "False" else float(v))
        out[k] = np.asarray(vals, dtype=float)
    return out


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Paired signed-rank on a−b. Returns (p, note); note flags why a test could not run."""
    d = a - b
    if np.all(d == 0):
        return float("nan"), "identical"
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float("nan"), "scipy missing"
    return float(wilcoxon(a, b).pvalue), ""


def _fmt(v: float, prec: int = 3) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.{prec}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix of the MARLauder CSVs, e.g. --tag v10")
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--agents", nargs="+", type=int, default=AGENTS)
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    print("=" * 108)
    print("MARLauder vs IR2 — mean±std per cell, n=100 maps. NEVER average across splits.")
    print("max_dist is the headline (px of robot travel). `steps` is NOT comparable across systems.")
    print("explored = IR2's evaluate_team_exploration_rate = MEAN over robots of each robot's OWN")
    print("map (not the union). success = every robot >= 99% of its own map. Both lower-is-better")
    print("for max_dist, higher-is-better for the rest.")
    print("=" * 108)

    for split in args.splits:
        for M in args.agents:
            ir2_f = _IR2_RESULTS / f"ir2_{split}_M{M}.csv"
            mar_f = _MARL_RESULTS / f"marlauder_{split}_M{M}{suffix}.csv"
            cell = f"{split}_M{M}"
            if not ir2_f.is_file():
                print(f"\n[{cell}] IR2 csv missing: {ir2_f}")
                continue
            if not mar_f.is_file():
                print(f"\n[{cell}] MARLauder csv missing: {mar_f.name} — run scripts/eval_comparison.py")
                continue
            ir2, mar = _read(ir2_f), _read(mar_f)
            n = min(len(ir2["eps"]), len(mar["eps"]))
            if len(ir2["eps"]) != len(mar["eps"]):
                print(f"\n[{cell}] WARNING: row counts differ "
                      f"(IR2 {len(ir2['eps'])} vs MARLauder {len(mar['eps'])}) — pairing the first {n}")

            print(f"\n[{cell}]  n={n}")
            print(f"  {'metric':<16}{'IR2':>22}{'MARLauder':>22}{'Δ (MARL−IR2)':>16}{'Wilcoxon p':>14}")
            for key, prec in (("max_dist", 0), ("steps", 1), ("explored", 3),
                              ("success", 2), ("connectivity", 2)):
                if key not in mar:
                    continue
                a, b = mar[key][:n], ir2[key][:n]
                # steps is reported for completeness but is a different unit on each side, so no
                # significance test is offered for it — a p-value there would invite a false read.
                p, note = _wilcoxon(a, b) if key != "steps" else (float("nan"), "not comparable")
                pcol = note if note else _fmt(p, 4)
                print(f"  {key:<16}{b.mean():>14.{prec}f}±{b.std():<7.{prec}f}"
                      f"{a.mean():>14.{prec}f}±{a.std():<7.{prec}f}"
                      f"{a.mean() - b.mean():>+16.{prec}f}{pcol:>14}")
            if "explored_union" in mar:
                # Appendix only. The union is what MARLauder's own objective optimizes, so it is the
                # flattering number; it has no IR2 counterpart and must never stand in for `explored`.
                print(f"  {'(explored_union':<16}{'—':>22}"
                      f"{mar['explored_union'][:n].mean():>14.3f}±{mar['explored_union'][:n].std():<7.3f}"
                      f"{'appendix only)':>32}")

    print("\n" + "=" * 108)


if __name__ == "__main__":
    main()
