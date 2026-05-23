#!/usr/bin/env bash
# Apre una shell bash nel container in esecuzione.
# Eseguibile piu volte in parallelo => piu terminali sullo stesso container.
# Avvia prima il container se non e attivo.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "$(docker compose ps -q marlauder 2>/dev/null)" ]; then
  echo "Container non attivo, lo avvio..."
  docker compose up -d
fi
docker compose exec marlauder bash
