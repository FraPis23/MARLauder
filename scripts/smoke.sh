#!/usr/bin/env bash
# GATE Fase 0: verifica che torch+warp vedano la GPU e scambino tensori senza copia host.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "$(docker compose ps -q marlauder 2>/dev/null)" ]; then
  docker compose up -d
fi
docker compose exec marlauder python scripts/smoke_test.py
