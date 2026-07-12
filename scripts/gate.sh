#!/usr/bin/env bash
# Merge gate: run before every commit of implementation work.
# 1. lint  2. full test suite  3. golden-trace regression check
set -uo pipefail
cd "$(dirname "$0")/.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

fail=0

echo "=== ruff ==="
uv run ruff check . || fail=1

echo "=== tests ==="
uv run python run_tests.py || fail=1

echo "=== golden traces ==="
uv run python scripts/golden_traces.py --check || fail=1

if [ "$fail" -ne 0 ]; then
    echo "GATE: FAIL"
    exit 1
fi
echo "GATE: PASS"
