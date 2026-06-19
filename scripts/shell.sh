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
# GPU passthrough injects the host's render/video group GIDs as supplementary groups;
# they have no /etc/group entry inside the container, so `groups` warns "cannot find
# name for group ID <n>". Name any unnamed GID, then drop into an interactive shell.
docker compose exec marlauder bash -c '
  for g in $(id -G); do
    getent group "$g" >/dev/null 2>&1 || groupadd -g "$g" "hostgrp$g" 2>/dev/null || true
  done
  exec bash
'
