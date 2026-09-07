#!/usr/bin/env bash
# Start the container detached. It stays alive so several shells can attach.
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose up -d
echo "Container 'marlauder' is up. Open a shell with: scripts/shell.sh"
