#!/usr/bin/env bash
# Stop and remove the container.
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose down
