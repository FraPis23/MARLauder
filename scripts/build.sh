#!/usr/bin/env bash
# Build the MARLauder Docker image.
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose build "$@"
