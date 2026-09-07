"""Repository-relative path resolution.

Every default path in this project used to be the container-absolute `/workspace/MARLauder/...`,
which made the code runnable only inside the Docker image. The three roots below resolve from the
location of this file instead, so a bare `git clone` works as-is, and each can still be overridden
by an environment variable for a container, a cluster job, or a scratch filesystem.

The Docker image mounts the repository at `/workspace/MARLauder`, so inside the container these
resolve to exactly the paths they were hardcoded to before — nothing changes there.

    MARLAUDER_DATA   preprocessed map packs      (default: <repo>/data)
    MARLAUDER_RUNS   checkpoints, logs, traces   (default: <repo>/runs)
    IR2_ROOT         the IR2 baseline repository (default: <repo>/../IR2-Multi-Robot-RL-Exploration)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_ROOT = Path(os.environ.get("MARLAUDER_DATA") or REPO_ROOT / "data")
RUNS_ROOT = Path(os.environ.get("MARLAUDER_RUNS") or REPO_ROOT / "runs")

# The IR2 baseline is a SEPARATE repository (the ground-truth DungeonMaps PNGs and the published
# per-episode CSVs live there). Only the comparison harness needs it; everything else runs without.
IR2_ROOT = Path(os.environ.get("IR2_ROOT") or REPO_ROOT.parent / "IR2-Multi-Robot-RL-Exploration")
