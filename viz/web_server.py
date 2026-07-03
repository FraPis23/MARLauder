#!/usr/bin/env python3
"""MARLauder web interface server.

Serves runs/ as static files + GET /api/runs for run discovery.
Run from the repo root (working_dir = /workspace/MARLauder):
    python viz/web_server.py [--port 8080]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

_REPO    = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))   # so `import scripts.train_args` resolves (torch-free)
RUNS_DIR = _REPO / "runs"
VIZ_DIR  = _REPO / "viz"
DATA_DIR = _REPO / "data"
_HOST    = socket.gethostname()


def _scan_splits() -> list[str]:
    """Available map splits = data/<group>/<name>/ dirs holding a maps.npy. Feeds the
    dashboard's on-demand-eval map-type dropdown."""
    out: list[str] = []
    if not DATA_DIR.exists():
        return out
    for grp in sorted(DATA_DIR.iterdir()):
        if not grp.is_dir():
            continue
        for sp in sorted(grp.iterdir()):
            if sp.is_dir() and (sp / "maps.npy").exists():
                out.append(f"{grp.name}/{sp.name}")
    return out


# ---- Training launcher: web form → spawned run_train.py process -------------------------
# The form is auto-built from scripts.train_args.schema() — EVERY flag, with its default,
# choices, and help (tooltip). Last-used values persist server-side (survive browser close AND
# power-off) in runs/.launch_params.json; the defaults themselves come from the schema.
LAUNCH_PARAMS_FILE = RUNS_DIR / ".launch_params.json"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")   # run-folder name allow-list (rename + launch)


def _arg_schema() -> list[dict]:
    from scripts.train_args import schema
    return schema()


def _schema_defaults() -> dict:
    """{dest: default} for every flag + the free-form extra_args field."""
    d = {f["dest"]: f["default"] for f in _arg_schema() if f["dest"] != "out"}
    d["extra_args"] = ""
    return d


def _load_launch_params() -> dict:
    try:
        saved = json.loads(LAUNCH_PARAMS_FILE.read_text())
    except Exception:
        saved = {}
    return {**_schema_defaults(), **(saved if isinstance(saved, dict) else {})}


def _save_launch_params(params: dict) -> None:
    tmp = LAUNCH_PARAMS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(params, indent=2))
    tmp.replace(LAUNCH_PARAMS_FILE)


def _build_cmd(p: dict) -> list[str]:
    """Map the web form's params to a run_train.py argv, generically from the schema. `-u` =
    unbuffered stdout so the live log streams to the browser without block-buffering. `--out`
    is appended by the launcher; only NON-default values are emitted to keep the command short."""
    import shlex
    cmd = ["python", "-u", "scripts/run_train.py"]
    for f in _arg_schema():
        dest, flag, kind = f["dest"], f["flag"], f["kind"]
        if dest == "out" or dest not in p:
            continue
        v = p[dest]
        if kind == "bool":
            if v:
                cmd.append(flag)
        else:
            sv = "" if v is None else str(v).strip()
            if sv == "":
                continue
            d = f["default"]
            if kind in ("int", "float"):
                try:
                    if float(v) == float(d):
                        continue        # numeric default (1 == 1.0) → skip
                except (TypeError, ValueError):
                    pass
            elif sv == str(d):
                continue                # string default → skip
            cmd += [flag, sv]
    extra = str(p.get("extra_args", "") or "").strip()
    if extra:
        cmd += shlex.split(extra)
    return cmd


def _next_auto_name(base: str) -> str:
    """base_<timestamp>, suffixed _2/_3/... on the (rare) chance that exact name is already
    taken — e.g. two launches within the same second. Never returns an existing dir, so
    auto-named runs (no explicit name typed) can never silently collide."""
    import time
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{base}_{stamp}"
    n = 2
    while (RUNS_DIR / name).exists():
        name = f"{base}_{stamp}_{n}"
        n += 1
    return name


def _launch_training(params: dict) -> dict:
    """Spawn a detached run_train.py. stdout+stderr → the run dir's train.log.
    start_new_session detaches it so it outlives the web server.

    Naming: `out_name` in params is an explicit folder name the user typed (from the launch
    form's "Run name" field) — if it already exists, this refuses UNLESS `force` is set, so the
    frontend can catch the {"error": "exists"} and re-ask/re-send with force after a JS confirm.
    Leaving out_name blank auto-generates a name that's guaranteed not to collide (no
    confirmation needed — nothing to overwrite)."""
    base = (str(params.get("wandb_run_name", "") or "run").strip().replace("/", "_")) or "run"
    requested = str(params.get("out_name", "") or "").strip()
    force = bool(params.get("force"))
    if requested:
        if not _SAFE_NAME.match(requested):
            return {"ok": False, "error": "run name must be non-empty and use only letters/digits/._-"}
        run = requested
    else:
        run = _next_auto_name(base)
    run_dir = RUNS_DIR / run
    if run_dir.exists() and not force:
        return {"ok": False, "error": "exists", "run": run}
    import subprocess
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(params) + ["--out", str(run_dir)]
    # Persist this run's exact params alongside its outputs (audit / reproduce).
    (run_dir / "params.json").write_text(json.dumps(
        {"params": params, "cmd": " ".join(cmd)}, indent=2))
    logf = open(run_dir / "train.log", "wb")
    logf.write(("$ " + " ".join(cmd) + "\n\n").encode())
    logf.flush()
    # Redirecting stdout to a real (non-tty) file here means run_train.py's own tee (which only
    # kicks in when stdout is a tty — i.e. a bare CLI launch) stays a no-op, so the file is never
    # written twice.
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=str(_REPO), stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)
    # Persist as "last used" for form prefill — MINUS the per-launch decisions (a specific
    # out_name / a force-overwrite confirmation must never silently apply to the NEXT launch).
    _save_launch_params({k: v for k, v in params.items() if k not in ("out_name", "force")})
    return {"ok": True, "run": run, "pid": proc.pid, "cmd": " ".join(cmd)}


def _read_log(run: str, offset: int) -> dict:
    """Incremental tail of a run's train.log from `offset` bytes → for the live console."""
    p = (RUNS_DIR / run / "train.log").resolve()
    if RUNS_DIR.resolve() not in p.parents or not p.exists():
        return {"data": "", "offset": offset, "exists": False}
    size = p.stat().st_size
    if offset > size:        # file shrank/rotated → restart from the top
        offset = 0
    with open(p, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    return {"data": chunk.decode("utf-8", "replace"), "offset": offset + len(chunk), "exists": True}


def _list_ckpts(run: str) -> list[str]:
    """Saved policy checkpoints in a run dir (newest first) → the eval-checkpoint picker."""
    d = (RUNS_DIR / run).resolve()
    if RUNS_DIR.resolve() not in d.parents or not d.is_dir():
        return []
    pts = sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in pts]


def _eval_ckpt(run: str, ckpt: str, split: str, n_maps: int, steps: int) -> dict:
    """Spawn a standalone eval of one saved checkpoint → GIFs + traces (unlocks inspector).
    Independent of training, so it works on stopped/done runs too."""
    import subprocess
    d = (RUNS_DIR / run).resolve()
    if RUNS_DIR.resolve() not in d.parents or not d.is_dir():
        return {"ok": False, "error": "invalid run"}
    ck = (d / ckpt).resolve()
    if ck.parent != d or not ck.exists() or ck.suffix != ".pt":
        return {"ok": False, "error": "invalid checkpoint"}
    # n_agents from the run's params.json when available (else 2).
    n_agents = 2
    try:
        n_agents = int(json.loads((d / "params.json").read_text())["params"].get("n_agents", 2))
    except Exception:
        pass
    cmd = ["python", "-u", "scripts/eval_ckpt.py",
           "--ckpt", str(ck), "--split", str(split),
           "--n-maps", str(int(n_maps)), "--steps", str(int(steps)),
           "--n-agents", str(n_agents), "--out", str(d)]
    logf = open(d / "eval.log", "ab")
    logf.write(("\n$ " + " ".join(cmd) + "\n").encode())
    logf.flush()
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=str(_REPO), stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)
    return {"ok": True, "pid": proc.pid, "cmd": " ".join(cmd)}


def _run_state(run: str) -> str | None:
    for r in _scan_runs():
        if r["name"] == run:
            return r["state"]
    return None


def _rename_run(old: str, new: str) -> dict:
    """Rename a finished run's directory. Refused for 'training' (an active process still holds
    paths/handles into the old dir) — safe for done/stopped/empty."""
    if not new or not _SAFE_NAME.match(new):
        return {"ok": False, "error": "new name must be non-empty and use only letters/digits/._-"}
    old_dir = (RUNS_DIR / old).resolve()
    if RUNS_DIR.resolve() not in old_dir.parents or not old_dir.is_dir():
        return {"ok": False, "error": "invalid run"}
    state = _run_state(old)
    if state == "training":
        return {"ok": False, "error": "can't rename a run that's still training — stop it first"}
    new_dir = RUNS_DIR / new
    if new_dir.exists():
        return {"ok": False, "error": f"'{new}' already exists"}
    old_dir.rename(new_dir)
    return {"ok": True, "name": new}


def _save_note(run: str, text: str) -> dict:
    d = (RUNS_DIR / run).resolve()
    if RUNS_DIR.resolve() not in d.parents or not d.is_dir():
        return {"ok": False, "error": "invalid run"}
    tmp = d / "notes.json.tmp"
    tmp.write_text(json.dumps({"text": text}))
    tmp.replace(d / "notes.json")
    return {"ok": True}


def _stop_run(run: str) -> dict:
    """Graceful stop+save: SIGTERM the run's training PID. The driver's signal handler saves
    the policy (ckpt_stop.pt) before exiting. PID is read from the run's status.json."""
    p = (RUNS_DIR / run / "status.json").resolve()
    if RUNS_DIR.resolve() not in p.parents or not p.exists():
        return {"ok": False, "error": "no status.json"}
    try:
        pid = int(json.loads(p.read_text()).get("pid", 0))
    except Exception as exc:
        return {"ok": False, "error": f"bad status.json ({exc})"}
    if pid <= 0:
        return {"ok": False, "error": "no pid"}
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"ok": False, "error": "process already gone"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "pid": pid, "msg": "SIGTERM sent; policy saved as ckpt_stop on exit"}


def _pid_alive(pid: int) -> bool:
    """True if process `pid` is currently running. POSIX: signal 0 probes without killing."""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False          # no such process → dead
    except PermissionError:
        return True           # exists but owned by another user → still alive
    except OSError:
        return False
    return True


def _scan_runs() -> list[dict]:
    runs: list[dict] = []
    if not RUNS_DIR.exists():
        return runs
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        has_traces    = (d / "traces" / "index.json").exists()
        has_final     = (d / "final.pt").exists()

        ckpts     = sorted(d.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
        last_ckpt = ckpts[-1].stem if ckpts else None
        pt_files  = list(d.glob("*.pt"))
        last_mtime = max((p.stat().st_mtime for p in pt_files), default=d.stat().st_mtime)

        n_traces = 0
        if has_traces:
            try:
                idx = json.loads((d / "traces" / "index.json").read_text())
                n_traces = len(idx)
            except Exception:
                pass

        # Primary signal: event-driven status.json (start / milestone / exit) from
        # train/driver.py. Liveness between events comes from the recorded PID, not a
        # timer — so a slow iteration never looks "stopped", and a hard kill (kill -9,
        # which writes no exit event) is still caught because the PID is gone.
        status: dict | None = None
        status_file = d / "status.json"
        if status_file.exists():
            try:
                status = json.loads(status_file.read_text())
            except Exception:
                pass

        if has_final or (status and status.get("state") == "done"):
            state = "done"
        elif status and status.get("state") == "stopped":
            state = "stopped"     # clean exit event recorded an interruption
        elif status and status.get("state") == "training":
            # Trust the PID only when the run is on THIS host (same container/namespace).
            same_host = status.get("host") == _HOST
            if same_host and not _pid_alive(int(status.get("pid", 0))):
                state = "stopped"  # process vanished without an exit event (hard kill)
            else:
                state = "training"
        elif pt_files:
            state = "stopped"     # legacy run (no status.json) but has checkpoints
        else:
            state = "empty"

        notes = ""
        notes_file = d / "notes.json"
        if notes_file.exists():
            try:
                notes = json.loads(notes_file.read_text()).get("text", "")
            except Exception:
                pass

        runs.append({
            "name":          d.name,
            "state":         state,
            "has_traces":    has_traces,
            "has_log":       (d / "train.log").exists(),
            "has_params":    (d / "params.json").exists(),
            "has_ckpts":     bool(pt_files),
            "last_ckpt":     last_ckpt,
            "last_mtime":    last_mtime,
            "n_traces":      n_traces,
            "status":        status,   # full status.json (progress %, ep_end, etc.) or null
            "notes":         notes,
        })

    _order = {"training": 0, "stopped": 1, "done": 2, "empty": 3}
    runs.sort(key=lambda r: (_order[r["state"]], -r["last_mtime"]))
    return runs


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(RUNS_DIR), **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/runs":
            self._api_runs()
        elif path == "/api/splits":
            self._send_json(_scan_splits())
        elif path == "/api/ckpts":
            qs = dict(p.split("=", 1) for p in urlparse(self.path).query.split("&") if "=" in p)
            self._send_json(_list_ckpts(qs.get("run", "")))
        elif path == "/api/argschema":
            self._send_json(_arg_schema())
        elif path == "/api/defaults":
            self._send_json(_schema_defaults())
        elif path == "/api/launch_params":
            self._send_json(_load_launch_params())
        elif path == "/api/log":
            qs = dict(p.split("=", 1) for p in urlparse(self.path).query.split("&") if "=" in p)
            try:
                offset = int(qs.get("offset", 0))
            except ValueError:
                offset = 0
            self._send_json(_read_log(qs.get("run", ""), offset))
        elif path in ("/", "/index.html"):
            self._serve_file(VIZ_DIR / "index.html", "text/html; charset=utf-8")
        elif path.endswith("/inspector.html"):
            # Served canonically here (not the per-run copy trace.py used to drop into each run
            # dir) so every run always gets the CURRENT inspector, never one frozen at whatever
            # viz/inspector.html looked like the last time that run's traces were captured.
            self._serve_file(VIZ_DIR / "inspector.html", "text/html; charset=utf-8")
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/launch":
            self._api_launch()
            return
        if parsed.path == "/api/stop":
            qs = dict(p.split("=", 1) for p in parsed.query.split("&") if "=" in p)
            self._send_json(_stop_run(qs.get("run", "")))
            return
        if parsed.path == "/api/rename":
            qs = dict(p.split("=", 1) for p in parsed.query.split("&") if "=" in p)
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception as exc:
                self.send_error(400, f"Bad body: {exc}")
                return
            self._send_json(_rename_run(qs.get("run", ""), str(body.get("new_name", "")).strip()))
            return
        if parsed.path == "/api/notes":
            qs = dict(p.split("=", 1) for p in parsed.query.split("&") if "=" in p)
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception as exc:
                self.send_error(400, f"Bad body: {exc}")
                return
            self._send_json(_save_note(qs.get("run", ""), str(body.get("text", ""))))
            return
        if parsed.path == "/api/eval":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception as exc:
                self.send_error(400, f"Bad body: {exc}")
                return
            self._send_json(_eval_ckpt(
                str(body.get("run", "")), str(body.get("ckpt", "")),
                str(body.get("split", "test/hybrid")),
                int(body.get("n_maps", 3)), int(body.get("steps", 256))))
            return
        if parsed.path != "/api/control":
            self.send_error(404, "Unknown endpoint")
            return
        qs = dict(p.split("=", 1) for p in parsed.query.split("&") if "=" in p)
        run = qs.get("run", "")
        run_dir = (RUNS_DIR / run).resolve()
        # Path-traversal guard: the resolved dir must live under RUNS_DIR and exist.
        if RUNS_DIR.resolve() not in run_dir.parents or not run_dir.is_dir():
            self.send_error(400, "Invalid run")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            cmd = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self.send_error(400, f"Bad body: {exc}")
            return
        if not isinstance(cmd, dict) or cmd.get("cmd") not in ("ckpt_eval", "switch_stage"):
            self.send_error(400, "cmd must be ckpt_eval or switch_stage")
            return
        # Atomic write so the driver never reads a partial command.
        tmp = run_dir / "control.json.tmp"
        tmp.write_text(json.dumps(cmd))
        tmp.replace(run_dir / "control.json")
        self._send_json({"ok": True, "queued": cmd})

    def _api_launch(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            params = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self.send_error(400, f"Bad body: {exc}")
            return
        if not isinstance(params, dict):
            self.send_error(400, "Body must be a JSON object of params")
            return
        try:
            result = _launch_training(params)
        except Exception as exc:
            self.send_error(500, f"Launch failed: {exc}")
            return
        self._send_json(result)

    def _api_runs(self):
        self._send_json(_scan_runs())

    def _send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, ctype: str):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404, f"Not found: {path.name}")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass   # suppress per-request noise; training stdout stays clean


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[web] http://0.0.0.0:{args.port}/  (runs → {RUNS_DIR})")
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
