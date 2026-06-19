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
import socket
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

_REPO    = Path(__file__).resolve().parent.parent
RUNS_DIR = _REPO / "runs"
VIZ_DIR  = _REPO / "viz"
_HOST    = socket.gethostname()


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
        has_inspector = (d / "inspector.html").exists()
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

        runs.append({
            "name":          d.name,
            "state":         state,
            "has_inspector": has_inspector,
            "has_traces":    has_traces,
            "last_ckpt":     last_ckpt,
            "last_mtime":    last_mtime,
            "n_traces":      n_traces,
            "status":        status,   # full status.json (progress %, ep_end, etc.) or null
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
        elif path in ("/", "/index.html"):
            self._serve_file(VIZ_DIR / "index.html", "text/html; charset=utf-8")
        else:
            super().do_GET()

    def _api_runs(self):
        body = json.dumps(_scan_runs()).encode()
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
