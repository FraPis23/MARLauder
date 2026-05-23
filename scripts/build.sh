#!/usr/bin/env bash
# Build dell'immagine Docker MARLauder.
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose build "$@"
