#!/usr/bin/env python3
"""Backfill `metrics.jsonl` from a console training log — for runs that predate the metrics writer.

DELIBERATELY SEPARATE from scripts/analyze_run.py. Its regexes are welded to the current print
format (`train/driver.py` [it …] and [evalsuite …] lines); every run started after the metrics
writer landed produces the real thing, so this becomes dead weight and should not drag the
long-lived analysis tool's format-independence down with it.

The output is LOSSY and says so: the `meta` row carries source/recovered/missing/low-precision
fields, and analyze_run.py prints a warning banner for any run marked this way. Only 6 of the 29
per-iteration `agg` keys and 15 of the 18 `eval/*` keys survive being printed.

    python scripts/parse_train_log.py runs_pipeline_v12_ir2.log
    python scripts/parse_train_log.py <log> --out runs/<dir>/metrics.jsonl   # single segment
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from jsonio import jsonable  # noqa: E402

# --- the per-iteration console line (train/driver.py, the `[it …]` print) --------------------
RE_ITER = re.compile(
    r"\[it\s+(?P<it>\d+)/(?P<n_iters>\d+)\]\s+"
    r"(?:ep_end=\s*(?P<ep_end>[\d.]+)%\(ended=\s*(?P<ended>\d+)\)|ep_end=\s*n/a\s*)\s+"
    r"pg=(?P<pg>[+-][\d.]+)\s+v=(?P<v>[\d.]+)\s+"
    r"ent=(?P<ent>[\d.]+)\s+kl=(?P<kl>[+-][\d.]+)\s+"
    r"clip=(?P<clip>[\d.]+)%\s+"
    r"redun=(?P<redun>[\d.]+)\s+stall=(?P<stall>[\d.]+)%\s+"
    r"pair=(?P<pair>[\d.]+)\s+"
    r"sync=(?P<sync>[+-][\d.]+)\((?P<sync_rate>[\d.]+)/k\)\s+"
    r"ownGap=(?P<owngap>[\d.]+)\s+"
    r"sps=(?P<sps>[\d.]+)\(")

# --- the eval-suite line ---------------------------------------------------------------------
RE_EVAL = re.compile(
    r"\[evalsuite it=\s*(?P<it>\d+)\]\s+score=(?P<score>[+-][\d.]+)±(?P<score_std>[\d.]+)\s+"
    r"auc=(?P<auc>[\d.]+)\s+imbN=(?P<imbn>[\d.]+)\s+fair=(?P<fair>[\d.]+)\s+"
    r"conc=(?P<conc>[\d.]+)\s+idle=(?P<idle>[\d.]+)\s+ov=(?P<ov>[\d.]+)\s+duty=(?P<duty>[\d.]+)\s+"
    r"succ=(?P<succ>[\d.]+)\s+s90=(?P<s90>[\d.]+)\s+nSync=(?P<nsync>[\d.]+)\s+"
    r"syncGap=(?P<syncgap>[\d.]+)\s+ownAUC=(?P<ownauc>[\d.]+)\s+scoreOwn=(?P<scoreown>[+-][\d.]+)")
RE_EVAL_SPLITS = re.compile(r"\[([a-z0-9_]+)=([+-][\d.]+)\]|\s([a-z0-9_]+)=([+-][\d.]+)\]")

RE_BANNER = re.compile(r"\[train\] iters=(?P<n_iters>\d+)\s+steps/iter=(?P<spi>\d+)")
RE_CKPT = re.compile(r"\[ckpt\] (?P<path>runs/(?P<dir>[^/]+)/\S+)")
RE_BEST = re.compile(r"\[best\] new best eval/score=(?P<score>[+-][\d.]+) at it=(?P<it>\d+)")
RE_OUTDIR = re.compile(r"--out\s+(?P<out>\S+)")
RE_PHASE_DONE = re.compile(r"PHASE\d_DONE\s+\S+=(?P<path>runs/(?P<dir>[^/]+)/\S+)")
RE_CONTROL_EVAL = re.compile(
    r"\[control eval tag=(?P<tag>\S+)\]\s+(?P<split>\S+)\(n=(?P<n>\d+)\):\s+"
    r"score=(?P<score>[+-][\d.]+)\s+auc=(?P<auc>[\d.]+)\s+succ=(?P<succ>[\d.]+)\s+"
    r"idle=(?P<idle>[\d.]+)")

RECOVERED = [
    "explore/ep_end", "explore/ep_end_n", "train/pg_loss", "train/v_loss", "train/entropy",
    "train/kl", "train/clipfrac", "metric/redundancy", "metric/stall_rate",
    "metric/mean_pair_dist", "reward/sync", "metric/sync_rate", "metric/own_cov_gap", "perf/sps",
    "eval/score", "eval/score_std", "eval/coverage_auc", "eval/contrib_imbalance_norm",
    "eval/fairness", "eval/concurrency", "eval/idle_rate_max", "eval/sensing_overlap",
    "eval/comm_duty", "eval/success_rate", "eval/steps_to_90", "eval/n_syncs", "eval/sync_gap",
    "eval/own_coverage_auc", "eval/score_own",
]
MISSING = [
    "eval/own_coverage_final", "eval/max_comm_gap", "eval/contrib_imbalance",
    "reward/novel", "reward/revisit", "reward/stall", "reward/step", "reward/rdv",
    "reward/completion", "reward/scan_self_diag", "reward/sync_give_diag", "reward/sync_recv_diag",
    "reward/sync_events", "reward/stall_streak", "reward/revisit_streak",
    "metric/revisit_rate", "metric/comm_duty_cycle", "metric/sensing_overlap",
    "metric/both_active", "metric/own_cov_mean", "metric/own_cov_min", "metric/idle_frac",
    "metric/coverage_per_dist", "metric/steps_to_50", "metric/steps_to_90",
    "metric/steps_to_50_per_kfree", "metric/steps_to_90_per_kfree",
    "explore/efficiency", "perf/coll_sps", "perf/upd_sps", "train/nan_skips",
]
# `stall` is printed with {:.0f}% — two significant figures at best, and 0% for anything under
# half a percent. Never compare a backfilled stall_rate against a natively-logged one.
LOW_PRECISION = ["metric/stall_rate"]


def _iter_row(m: re.Match, steps_per_iter: int) -> dict:
    it = int(m.group("it"))
    ep_end = m.group("ep_end")
    row = {
        "kind": "iter", "iter": it, "env_steps": it * steps_per_iter,
        "train/pg_loss": float(m.group("pg")), "train/v_loss": float(m.group("v")),
        "train/entropy": float(m.group("ent")), "train/kl": float(m.group("kl")),
        "train/clipfrac": float(m.group("clip")) / 100.0,
        "metric/redundancy": float(m.group("redun")),
        "metric/stall_rate": float(m.group("stall")) / 100.0,
        "metric/mean_pair_dist": float(m.group("pair")),
        "reward/sync": float(m.group("sync")),
        "metric/sync_rate": float(m.group("sync_rate")) / 1000.0,
        "metric/own_cov_gap": float(m.group("owngap")),
        "perf/sps": float(m.group("sps")),
    }
    if ep_end is not None:
        row["explore/ep_end"] = float(ep_end) / 100.0
        row["explore/ep_end_n"] = int(m.group("ended"))
    return row


def _eval_keys(m: re.Match) -> dict:
    return {
        "eval/score": float(m.group("score")), "eval/score_std": float(m.group("score_std")),
        "eval/coverage_auc": float(m.group("auc")),
        "eval/contrib_imbalance_norm": float(m.group("imbn")),
        "eval/fairness": float(m.group("fair")), "eval/concurrency": float(m.group("conc")),
        "eval/idle_rate_max": float(m.group("idle")),
        "eval/sensing_overlap": float(m.group("ov")), "eval/comm_duty": float(m.group("duty")),
        "eval/success_rate": float(m.group("succ")), "eval/steps_to_90": float(m.group("s90")),
        "eval/n_syncs": float(m.group("nsync")), "eval/sync_gap": float(m.group("syncgap")),
        "eval/own_coverage_auc": float(m.group("ownauc")),
        "eval/score_own": float(m.group("scoreown")),
    }


def parse(log_path: Path) -> list[dict]:
    """Split the log into segments (one per `[train] iters=` banner) and parse each.

    A pipeline log holds several runs back to back with no run tag on the `[it …]` lines, so the
    destination run dir has to be recovered from the surrounding `[ckpt] runs/<dir>/…`,
    `PHASEn_DONE …=runs/<dir>/…` and `--out runs/<dir>` lines instead.
    """
    segments: list[dict] = []
    cur: dict | None = None
    pending_out: str | None = None

    for line in log_path.read_text(errors="replace").splitlines():
        mo = RE_OUTDIR.search(line)
        if mo:
            pending_out = Path(mo.group("out")).name

        mb = RE_BANNER.search(line)
        if mb:
            cur = {"n_iters": int(mb.group("n_iters")), "steps_per_iter": int(mb.group("spi")),
                   "dir": pending_out, "rows": [], "pending_eval": {}}
            segments.append(cur)
            continue
        if cur is None:
            continue

        me = RE_EVAL.search(line)
        if me:
            keys = _eval_keys(me)
            tail = line.split("scoreOwn=")[-1]
            if "[" in tail:                       # multi-split suffix: [complex=+0.42 hybrid=+0.1]
                for short, val in re.findall(r"([a-z0-9_]+)=([+-][\d.]+)", tail.split("[", 1)[1]):
                    keys[f"eval/{short}/score"] = float(val)
            cur["pending_eval"][int(me.group("it"))] = keys
            continue

        mc = RE_CONTROL_EVAL.search(line)
        if mc:
            cur["rows"].append({
                "kind": "eval_ondemand", "tag": mc.group("tag"), "split": mc.group("split"),
                "n_maps": int(mc.group("n")), "eval/score": float(mc.group("score")),
                "eval/coverage_auc": float(mc.group("auc")),
                "eval/success_rate": float(mc.group("succ")),
                "eval/idle_rate_max": float(mc.group("idle"))})
            continue

        mk = RE_CKPT.search(line)
        if mk:
            cur["dir"] = cur["dir"] or mk.group("dir")
            cur["rows"].append({"kind": "event", "event": "ckpt", "path": mk.group("path")})
            continue

        mbest = RE_BEST.search(line)
        if mbest:
            cur["rows"].append({"kind": "event", "event": "best", "iter": int(mbest.group("it")),
                                "score": float(mbest.group("score"))})
            continue

        mp = RE_PHASE_DONE.search(line)
        if mp:
            cur["dir"] = cur["dir"] or mp.group("dir")
            continue

        mi = RE_ITER.search(line)
        if mi:
            cur["rows"].append(_iter_row(mi, cur["steps_per_iter"]))

    # Fold eval blocks into their iteration row in a POST-PASS. The [evalsuite] line is printed
    # AFTER the [it] line of the same iteration (train/driver.py prints the iter summary first,
    # then runs the eval suite), so it cannot be merged on the fly. Matching by iteration number
    # rather than by adjacency also survives any interleaved [ckpt]/[best]/GIF output.
    for seg in segments:
        by_iter = {r["iter"]: r for r in seg["rows"] if r.get("kind") == "iter"}
        for it, keys in seg["pending_eval"].items():
            if it in by_iter:
                by_iter[it].update(keys)
            else:                       # eval with no surviving iter line (truncated log)
                seg["rows"].append({"kind": "iter", "iter": it,
                                    "env_steps": it * seg["steps_per_iter"], **keys})
        seg["rows"].sort(key=lambda r: (r.get("iter", 1 << 30), r.get("kind") != "iter"))

    return segments


def _cfg_from_params(run_dir: Path) -> dict:
    """Recover the run config from params.json, which the console log never prints.

    max_episode_steps is corrected for the clamp the driver applies AFTER argparse writes the
    file (`max(max_episode_steps, rollout_len)`) — params.json records the requested value, not
    the executed one, and every per-episode figure downstream depends on the executed one.
    """
    p = run_dir / "params.json"
    if not p.exists():
        return {}
    try:
        params = json.loads(p.read_text()).get("params", {})
    except Exception:
        return {}
    out = {k: params[k] for k in ("split", "n_agents", "n_envs", "rollout_len") if k in params}
    mes, rl = params.get("max_episode_steps"), params.get("rollout_len")
    if mes is not None:
        out["max_episode_steps"] = max(int(mes), int(rl)) if rl else int(mes)
        out["max_episode_steps_requested"] = int(mes)
    return out


def write_segment(seg: dict, out: Path, log_path: Path) -> None:
    iters = [r for r in seg["rows"] if r.get("kind") == "iter"]
    meta = {
        "kind": "meta", "schema": 1, "source": "train_log_backfill",
        "source_log": str(log_path), "run": out.parent.name,
        **_cfg_from_params(out.parent),
        "n_iters": seg["n_iters"], "steps_per_iter": seg["steps_per_iter"],
        "recovered_keys": RECOVERED, "missing_keys": MISSING,
        "low_precision_keys": LOW_PRECISION,
        "note": ("Reconstructed from console text. env_steps = iter x steps_per_iter. "
                 "max_episode_steps unknown (not printed) so --reward-budget cannot show "
                 "per-episode totals; reward/* components other than sync were never printed."),
    }
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(jsonable(meta), separators=(",", ":")) + "\n")
        for r in seg["rows"]:
            fh.write(json.dumps(jsonable(r), separators=(",", ":")) + "\n")
    n_eval = sum(1 for r in iters if any(k.startswith("eval/") for k in r))
    print(f"[backfill] {out}  iters={len(iters)}  eval_ticks={n_eval}  "
          f"events={sum(1 for r in seg['rows'] if r.get('kind') == 'event')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="single output path (only valid when the log holds one run)")
    ap.add_argument("--runs-dir", type=Path, default=_REPO / "runs")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing metrics.jsonl (refused by default when it was "
                         "written natively — backfilled data must never silently replace real data)")
    args = ap.parse_args()

    segments = [s for s in parse(args.log) if s["rows"]]
    if not segments:
        print("no [train] banner / [it ...] lines found — is this a training log?")
        return 1
    if args.out and len(segments) > 1:
        print(f"log holds {len(segments)} runs; --out takes exactly one. Omit it to write each "
              f"run's own dir.")
        return 1

    rc = 0
    for seg in segments:
        if args.out:
            out = args.out
        elif seg["dir"]:
            out = args.runs_dir / seg["dir"] / "metrics.jsonl"
        else:
            print(f"[backfill] SKIP segment ({len(seg['rows'])} rows): could not determine the run "
                  f"dir (no [ckpt]/--out/PHASE_DONE line). Re-run with --out to place it.")
            rc = 1
            continue
        if not out.parent.exists():
            print(f"[backfill] SKIP {out}: {out.parent} does not exist")
            rc = 1
            continue
        if out.exists() and not args.force:
            try:
                first = json.loads(out.open(encoding="utf-8").readline())
            except Exception:
                first = {}
            if first.get("source") != "train_log_backfill":
                print(f"[backfill] REFUSING to overwrite natively-written {out} (use --force)")
                rc = 1
                continue
        write_segment(seg, out, args.log)
    return rc


if __name__ == "__main__":
    sys.exit(main())
