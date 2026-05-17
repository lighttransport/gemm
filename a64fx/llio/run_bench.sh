#!/bin/bash
# run_bench.sh -- end-to-end LLIO read benchmark on $PJM_LOCALTMP.
#
# Assumes:
#   - bench_fopen / bench_mmap already built (run `make` first)
#   - copy_bench.sh has been run, or a single file lives at $TARGET
#
# Usage:
#   run_bench.sh [TARGET_FILE]
#     TARGET_FILE defaults to
#       $PJM_LOCALTMP/llio_bench/Qwen3.5-9B-UD-Q4_K_XL.gguf
#       (smallest non-mmproj file → ~6 GB, fits comfortably in 87 GB)
#
# Set BENCH_PY=0 to skip the Python passes.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_BASE="${PJM_LOCALTMP:-/tmp}/llio_bench"
TARGET="${1:-$DEFAULT_BASE/Qwen3.5-9B-UD-Q4_K_XL.gguf}"
BENCH_PY="${BENCH_PY:-1}"

if [[ ! -f "$TARGET" ]]; then
    echo "ERROR: target file not found: $TARGET" >&2
    echo "       run copy_bench.sh first" >&2
    exit 1
fi

echo "============================================================"
echo "llio read bench"
echo "host:   $(hostname)"
echo "date:   $(date -u +%FT%TZ)"
echo "target: $TARGET"
echo "size:   $(stat -c %s "$TARGET") bytes"
echo "============================================================"
echo

run() {
    echo "----- $* -----"
    "$@"
    echo
}

# C: fopen
if [[ -x "$SCRIPT_DIR/bench_fopen" ]]; then
    run "$SCRIPT_DIR/bench_fopen" "$TARGET" 1048576 2
    run "$SCRIPT_DIR/bench_fopen" "$TARGET" 16777216 2
else
    echo "skip bench_fopen (not built; run \`make\`)"
fi

# C: mmap
if [[ -x "$SCRIPT_DIR/bench_mmap" ]]; then
    run "$SCRIPT_DIR/bench_mmap" "$TARGET" seq8 2 0
    run "$SCRIPT_DIR/bench_mmap" "$TARGET" page 2 0
    run "$SCRIPT_DIR/bench_mmap" "$TARGET" seq8 2 1   # MAP_POPULATE
else
    echo "skip bench_mmap (not built; run \`make\`)"
fi

if [[ "$BENCH_PY" != "0" ]]; then
    run python3 "$SCRIPT_DIR/bench_fopen.py" "$TARGET" 1048576 2 os
    run python3 "$SCRIPT_DIR/bench_mmap.py"  "$TARGET" page    2 0
fi
