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
import math
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


def _wilcoxon_numpy(d: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value on the paired differences, without scipy.

    The container this runs in is ephemeral and scipy is not part of the image, so relying on the
    import meant every p-value in the table printed "scipy missing" — and the paired test is not
    optional, it is what the frozen protocol asks for (PROTOCOL.md: "Wilcoxon signed-rank APPAIATO
    per mappa"). This reproduces scipy.stats.wilcoxon's default path for our n: zero differences
    dropped ("wilcox" handling), average ranks on |d|, tie-corrected normal approximation, and NO
    continuity correction — scipy's mode="auto" already switches to the same normal approximation
    above n=25 and correction defaults to False, so with n=100 maps the two agree to ~1e-12.

    The tie correction matters here rather than being a formality: `success` and `connectivity` are
    booleans, so their differences are all in {-1, 0, +1} and |d| is one enormous tie group. Without
    the sum(t^3-t)/48 term sigma is overstated and every boolean p-value comes out too large — i.e.
    the direction that silently hides a real effect.
    """
    n = d.size
    if n == 0:
        return float("nan")
    absd = np.abs(d)
    order = np.argsort(absd, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and absd[order[j + 1]] == absd[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0                      # average of ranks i+1..j+1
        ranks[order[i:j + 1]] = avg
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1
    w_plus = ranks[d > 0].sum()
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return float("nan")
    z = (w_plus - mu) / math.sqrt(var)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))   # two-sided


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Paired signed-rank on a−b. Returns (p, note); note flags why a test could not run."""
    d = a - b
    if np.all(d == 0):
        return float("nan"), "identical"
    d = d[d != 0]
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return _wilcoxon_numpy(d), ""
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
                              ("success", 2), ("connectivity", 2),
                              # --- PROTOCOL v2 behaviour columns. Same formula on both sides
                              # (PROTOCOL_V2_DISTANZA.md §7.2), so these ARE cross-system
                              # comparable — unlike `steps`. They exist because `success` cannot
                              # tell coordinated exploration (split up, then deliberately meet to
                              # exchange) from the degenerate solution (never separate, so the two
                              # maps coincide for free and no rendezvous is ever needed); the
                              # degenerate one scores a perfect `success` while demonstrating none
                              # of the coordination the thesis claims.
                              ("contrib_imbalance", 3), ("own_gap_final", 3),
                              ("comm_duty", 3), ("sensing_overlap", 3),
                              ("pair_dist_mean_px", 1), ("pair_dist_max_px", 1),
                              ("n_contacts", 1)):
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
