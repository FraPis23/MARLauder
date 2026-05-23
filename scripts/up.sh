#!/usr/bin/env bash
# Avvia il container in background (detached). Resta vivo per attach multipli.
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose up -d
echo "Container 'marlauder' attivo. Apri un terminale con: scripts/shell.sh"
