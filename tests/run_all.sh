#!/usr/bin/env bash
# Run the whole test suite from the repository root. Stops at the first failure.
#   bash tests/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:."
for t in tests/[0-9][0-9]_test_*.py; do
  echo "=============================== $t"
  python "$t"
done
echo "=============================== all tests passed"
