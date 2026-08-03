#!/usr/bin/env python3
"""Offline analysis of `runs/<run>/metrics.jsonl` — the W&B-free reporting path.

STDLIB ONLY in the text path, on purpose: this has to run on the HOST (where numpy is not
installed), not just inside the container. matplotlib is imported lazily and only for --plot;
its absence degrades to ASCII sparklines rather than failing.

    python scripts/analyze_run.py runs/<run>                       # summary
    python scripts/analyze_run.py runs/<run> --reward-budget       # which term owns the return
    python scripts/analyze_run.py runs/<run> --own-coverage        # progress toward own-99%
    python scripts/analyze_run.py runs/a runs/b --compare          # one row per run
    python scripts/analyze_run.py runs/<run> --keys reward/novel metric/own_cov_gap
    python scripts/analyze_run.py runs/<run> --csv out.csv         # wide sparse CSV
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# Reward-term taxonomy. `reward_terms` in env/explorer.py mixes actual summands of the reward with
# pure diagnostics; averaging them together (or summing them) would be meaningless, so the split
# is explicit here and drives --reward-budget.
SUMMANDS = ["novel", "revisit", "stall", "step", "sync", "rdv", "completion", "own_cov"]
DIAGNOSTICS = ["scan_self_diag", "sync_give_diag", "sync_recv_diag", "sync_events",
               "stall_streak", "revisit_streak"]

SUMMARY_KEYS = [
    "explore/ep_end", "explore/efficiency", "train/kl", "train/entropy", "train/clipfrac",
    "metric/own_cov_mean", "metric/own_cov_min", "metric/own_cov_gap",
    "metric/redundancy", "metric/sensing_overlap", "metric/comm_duty_cycle",
    "eval/score", "eval/coverage_auc", "eval/own_coverage_auc", "eval/own_coverage_final",
    "eval/success_rate", "eval/steps_to_90", "eval/n_syncs", "eval/sync_gap",
]
OWN_COV_KEYS = ["eval/success_rate", "eval/own_coverage_final", "eval/own_coverage_auc",
                "eval/coverage_auc", "eval/n_syncs", "eval/sync_gap"]

_BLOCKS = "▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------------------------------------
# Loading

def load(run_dir: Path) -> dict:
    """Read metrics.jsonl. Tolerant by design: the writer is line-buffered append and the run is
    routinely SIGTERM'd, so the final line can be torn. A bad line is counted and skipped, never
    fatal — and the count is reported, because silently dropping data is how you end up trusting
    a truncated curve."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Runs started before the metrics writer landed can be "
                         f"backfilled with scripts/parse_train_log.py.")
    meta, iters, ondemand, events, bad = {}, [], [], [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                bad += 1
                continue
            kind = row.get("kind")
            if kind == "meta":
                meta = row
            elif kind == "iter":
                iters.append(row)
            elif kind == "eval_ondemand":
                ondemand.append(row)
            elif kind == "event":
                events.append(row)
    return {"dir": run_dir, "meta": meta, "iters": iters,
            "ondemand": ondemand, "events": events, "bad_lines": bad}


def series(rows: list[dict], key: str) -> list[tuple[int, float]]:
    """(env_steps, value) pairs where `key` is present and numeric. Eval keys are sparse — they
    exist only on eval iterations — so callers must never assume alignment with the iter index."""
    out = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append((int(r.get("env_steps", r.get("iter", 0))), float(v)))
    return out


# ---------------------------------------------------------------------------------------------
# Rendering helpers

def spark(vals: list[float], width: int = 56) -> str:
    """Bucketed ASCII sparkline. Buckets by MEAN so a single spike cannot dominate the shape."""
    if not vals:
        return ""
    if len(vals) > width:
        step = len(vals) / width
        vals = [sum(vals[int(i * step):max(int((i + 1) * step), int(i * step) + 1)])
                / max(1, len(vals[int(i * step):max(int((i + 1) * step), int(i * step) + 1)]))
                for i in range(width)]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return _BLOCKS[0] * len(vals)
    return "".join(_BLOCKS[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def fmt(v, w: int = 9, prec: int = 4) -> str:
    if v is None:
        return "—".rjust(w)
    if isinstance(v, float):
        if not math.isfinite(v):
            return "—".rjust(w)
        if abs(v) >= 1e5 or (v != 0 and abs(v) < 1e-3):
            return f"{v:>{w}.2e}"
        return f"{v:>{w}.{prec}f}"
    return f"{v:>{w}}"


def table(headers: list[str], rows: list[list], widths: list[int] | None = None) -> str:
    widths = widths or [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
                        for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) if i == 0 else h.rjust(w) for i, (h, w) in enumerate(zip(headers, widths)))
    sep = "  ".join("-" * w for w in widths)
    body = [
        "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w)
                  for i, (c, w) in enumerate(zip(r, widths)))
        for r in rows
    ]
    return "\n".join([line, sep] + body)


def _hdr(title: str) -> str:
    return f"\n{'=' * 92}\n{title}\n{'=' * 92}"


# ---------------------------------------------------------------------------------------------
# Reports

def report_summary(run: dict) -> None:
    meta, iters = run["meta"], run["iters"]
    print(_hdr(f"RUN  {run['dir']}"))
    if meta.get("source") == "train_log_backfill":
        print("!! BACKFILLED from a text log — INCOMPLETE. "
              f"missing: {', '.join(meta.get('missing_keys', [])) or 'n/a'}")
        if meta.get("low_precision_keys"):
            print(f"!! low precision (printed rounded): {', '.join(meta['low_precision_keys'])}")
    if run["bad_lines"]:
        print(f"!! {run['bad_lines']} unparseable line(s) skipped (torn final line is expected "
              f"after a SIGTERM)")
    if not iters:
        print("no iteration rows")
        return
    last = iters[-1]
    steps = last.get("env_steps", 0)
    wall = last.get("wall") or 0.0
    print(f"split={meta.get('split','?')}  n_agents={meta.get('n_agents','?')}  "
          f"n_envs={meta.get('n_envs','?')}  rollout_len={meta.get('rollout_len','?')}")
    # The EFFECTIVE episode length, which params.json does not record (driver clamps it up to
    # rollout_len after argparse has already written the file).
    mes = meta.get("max_episode_steps", "?")
    req = meta.get("max_episode_steps_requested")
    clamp = f" (requested {req}, CLAMPED UP to rollout_len)" if req and req != mes else ""
    # Backfilled runs have no wall clock (never printed); fall back to the mean of the per-iter
    # sps the log did print rather than showing a fabricated 0.
    if wall:
        pace = f"wall={wall/3600:.2f}h  mean_sps={steps/wall:.0f}"
    else:
        sps = [v for _, v in series(iters, "perf/sps")]
        pace = f"wall=—  mean_sps={sum(sps)/len(sps):.0f} (from per-iter sps)" if sps else "wall=—"
    print(f"max_episode_steps={mes} (effective){clamp}  "
          f"iters={len(iters)}/{meta.get('n_iters','?')}  env_steps={steps:,}  {pace}")
    done = [e for e in run["events"] if e.get("event") in ("done", "stopped")]
    if done:
        e = done[-1]
        extra = (f"  best_score={e['best_eval_score']:+.3f}@it{e['best_eval_iter']}"
                 if e.get("best_eval_score") is not None else "")
        print(f"state={e['event']}{extra}")

    rows = []
    for k in SUMMARY_KEYS:
        s = series(iters, k)
        if not s:
            continue
        vals = [v for _, v in s]
        tail = vals[-10:]
        best_i = max(range(len(vals)), key=lambda i: vals[i])
        rows.append([k, fmt(vals[0]), fmt(vals[-1]), fmt(sum(tail) / len(tail)),
                     fmt(vals[best_i]), f"{s[best_i][0]:,}", spark(vals)])
    print()
    print(table(["key", "first", "last", "last10", "best", "best@steps", "trend"], rows))


def report_reward_budget(run: dict) -> None:
    """Which term actually owns the return.

    The per-episode column is the one that matters: `agg` values are per-step per-agent means, and
    a −0.11/step penalty against a +0.04/step credit reads as "comparable" until you multiply by
    the episode length. This is the report that decides whether a shaping term is doing anything.
    """
    iters, meta = run["iters"], run["meta"]
    ep_len = meta.get("max_episode_steps") or 0
    print(_hdr(f"REWARD BUDGET  {run['dir']}   (per-episode = per-step × {ep_len} steps)"))
    if not ep_len:
        print("no max_episode_steps in meta — per-episode column unavailable")
    tail = iters[-10:] if len(iters) >= 10 else iters

    def mean_of(key):
        s = series(tail, key)
        return sum(v for _, v in s) / len(s) if s else None

    present = [(t, mean_of(f"reward/{t}")) for t in SUMMANDS]
    present = [(t, v) for t, v in present if v is not None]
    total_abs = sum(abs(v) for _, v in present) or 1.0
    net = sum(v for _, v in present)

    rows = []
    for t, v in sorted(present, key=lambda kv: -abs(kv[1])):
        s = series(iters, f"reward/{t}")
        rows.append([t, fmt(v, 10, 5), fmt(v * ep_len, 10, 2) if ep_len else "—",
                     f"{100*abs(v)/total_abs:5.1f}%", spark([x for _, x in s], 40)])
    rows.append(["— NET —", fmt(net, 10, 5), fmt(net * ep_len, 10, 2) if ep_len else "—", "", ""])
    print(f"(mean over the last {len(tail)} iteration(s))\n")
    print(table(["term", "per-step", "per-episode", "|share|", "trend"], rows))

    diag = [(t, mean_of(f"reward/{t}")) for t in DIAGNOSTICS]
    diag = [(t, v) for t, v in diag if v is not None]
    if diag:
        print("\ndiagnostics (NOT reward summands):")
        print(table(["term", "per-step", "per-episode"],
                    [[t, fmt(v, 10, 5), fmt(v * ep_len, 10, 2) if ep_len else "—"]
                     for t, v in diag]))

    # The two pre-registered failure modes of this project, checked automatically.
    comp = dict(present).get("completion")
    if comp is not None and abs(comp) < 1e-12:
        print("\n!! reward/completion is IDENTICALLY ZERO — the terminal bonus never fired. Under "
              "--done-mode own that means the objective was unreachable and there was no gradient "
              "toward it at all.")
    rev, nov = dict(present).get("revisit"), dict(present).get("novel")
    if rev is not None and nov and abs(rev) > 0.7 * abs(nov):
        print(f"\n!! reward/revisit ({rev*ep_len:+.1f}/ep) exceeds 0.7x |reward/novel| "
              f"({nov*ep_len:+.1f}/ep) — the anti-loop penalty is competing with exploration. "
              f"If the objective requires backtracking, rerun with --revisit-streak-cap 4.0.")


def report_own_coverage(run: dict) -> None:
    iters = run["iters"]
    print(_hdr(f"OWN COVERAGE  {run['dir']}"))
    rows = []
    for k in ["metric/own_cov_mean", "metric/own_cov_min", "metric/own_cov_gap"]:
        s = series(iters, k)
        if s:
            vals = [v for _, v in s]
            rows.append([k, fmt(vals[0]), fmt(vals[-1]), spark(vals)])
    if rows:
        print("training (per-iteration):\n")
        print(table(["key", "first", "last", "trend"], rows))

    ev = [r for r in iters if any(k.startswith("eval/") for k in r)]
    if ev:
        print("\neval suite (one row per eval tick):\n")
        hdr = ["env_steps"] + [k.replace("eval/", "") for k in OWN_COV_KEYS]
        body = [[f"{r.get('env_steps',0):,}"] + [fmt(r.get(k), 8) for k in OWN_COV_KEYS]
                for r in ev]
        print(table(hdr, body))
        succ = [r.get("eval/success_rate") for r in ev if r.get("eval/success_rate") is not None]
        if succ and max(succ) == 0.0:
            print(f"\n!! eval/success_rate is 0.00 in ALL {len(succ)} eval blocks — no episode "
                  f"ever met the termination criterion on this suite.")

    if run["ondemand"]:
        print("\non-demand evals:\n")
        hdr = ["tag", "split", "n"] + [k.replace("eval/", "") for k in OWN_COV_KEYS]
        body = [[r.get("tag", ""), r.get("split", ""), r.get("n_maps", "")]
                + [fmt(r.get(k), 8) for k in OWN_COV_KEYS] for r in run["ondemand"]]
        print(table(hdr, body))


def report_keys(run: dict, keys: list[str]) -> None:
    print(_hdr(f"KEYS  {run['dir']}"))
    rows = []
    for k in keys:
        s = series(run["iters"], k) or series(run["ondemand"], k)
        if not s:
            rows.append([k, "—", "—", "—", "(absent)"])
            continue
        vals = [v for _, v in s]
        rows.append([k, fmt(vals[0]), fmt(vals[-1]), f"{len(vals):>5}", spark(vals)])
    print(table(["key", "first", "last", "n", "trend"], rows))


def report_compare(runs: list[dict], keys: list[str]) -> None:
    print(_hdr("COMPARE  (last value per key)"))
    hdr = ["run", "iters", "env_steps"] + [k.split("/")[-1] for k in keys]
    body = []
    for run in runs:
        iters = run["iters"]
        if not iters:
            continue
        row = [run["dir"].name, len(iters), f"{iters[-1].get('env_steps',0):,}"]
        for k in keys:
            s = series(iters, k)
            row.append(fmt(s[-1][1], 8) if s else "—")
        body.append(row)
    print(table(hdr, body))


def write_csv(run: dict, out: Path) -> None:
    """Wide sparse CSV: union of all keys, one row per iteration. Blank where a key is absent on
    that iteration (which is the normal state of every eval/* column)."""
    iters = run["iters"]
    keys, seen = [], set()
    for r in iters:
        for k in r:
            if k not in seen and k != "kind":
                seen.add(k)
                keys.append(k)
    keys.sort(key=lambda k: (k not in ("iter", "env_steps", "wall"), k))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in iters:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[csv] {out}  ({len(iters)} rows x {len(keys)} cols)")


def make_plots(run: dict, keys: list[str], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available — the ASCII trends above carry the same data.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in keys:
        s = series(run["iters"], k)
        if not s:
            continue
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot([x for x, _ in s], [v for _, v in s], lw=1.2)
        ax.set_xlabel("env steps")
        ax.set_ylabel(k)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"{k.replace('/', '_')}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"[plot] {p}")


# ---------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories containing metrics.jsonl")
    ap.add_argument("--summary", action="store_true", help="(default when nothing else is asked)")
    ap.add_argument("--reward-budget", action="store_true",
                    help="per-term per-episode magnitudes and share of the return")
    ap.add_argument("--own-coverage", action="store_true",
                    help="own_cov trajectory + eval success/own-coverage per eval tick")
    ap.add_argument("--compare", action="store_true", help="one row per run")
    ap.add_argument("--keys", nargs="*", default=None, help="explicit keys to report/plot")
    ap.add_argument("--csv", type=Path, default=None, help="write a wide sparse CSV (single run)")
    ap.add_argument("--plot", action="store_true", help="PNG curves (needs matplotlib)")
    ap.add_argument("--out-dir", type=Path, default=None, help="where --plot writes (default <run>/figs)")
    args = ap.parse_args()

    loaded = [load(r) for r in args.runs]
    any_report = args.reward_budget or args.own_coverage or args.compare or args.keys

    if args.compare:
        report_compare(loaded, args.keys or ["eval/score", "eval/coverage_auc",
                                             "eval/success_rate", "eval/own_coverage_auc",
                                             "metric/own_cov_gap", "explore/ep_end"])
    for run in loaded:
        if args.summary or not any_report:
            report_summary(run)
        if args.reward_budget:
            report_reward_budget(run)
        if args.own_coverage:
            report_own_coverage(run)
        if args.keys and not args.compare:
            report_keys(run, args.keys)
        if args.csv:
            write_csv(run, args.csv if len(loaded) == 1 else
                      args.csv.with_name(f"{run['dir'].name}_{args.csv.name}"))
        if args.plot:
            make_plots(run, args.keys or SUMMARY_KEYS, args.out_dir or (run["dir"] / "figs"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
