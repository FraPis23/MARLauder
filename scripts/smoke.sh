#!/usr/bin/env bash
# Toolchain gate: check that torch and Warp both see the GPU and exchange tensors
# without a host copy.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "$(docker compose ps -q marlauder 2>/dev/null)" ]; then
  docker compose up -d
fi
docker compose exec marlauder python tests/00_test_toolchain.py
