#!/usr/bin/env bash
# Open a bash shell in the running container.
# Can be run several times in parallel => several terminals on the same container.
# Starts the container first if it is not already up.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "$(docker compose ps -q marlauder 2>/dev/null)" ]; then
  echo "Container not running, starting it..."
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
