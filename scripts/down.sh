#!/usr/bin/env bash
# Ferma e rimuove il container.
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose down
