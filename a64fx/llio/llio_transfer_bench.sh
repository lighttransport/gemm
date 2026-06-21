#!/bin/bash
# llio_transfer_bench.sh -- benchmark `llio_transfer` (LLIO L2 common-file
# distribution) vs. plain NFS-source reads.
#
# What it does:
#   1. (optional) llio_transfer --purge to drop any existing common-file cache
#   2. time `llio_transfer --sync <files>` (pre-stages to L2 cache,
#      file path unchanged)
#   3. run bench_fopen against the ORIGINAL path -- reads now hit the
#      SSD-backed L2 cache instead of Lustre
#
# IMPORTANT (per llio_transfer(1)):
#   - Do NOT open / cp / stat the file before calling llio_transfer in this
#     job, or the L2 cache may already hold stale data and the transfer
#     will fail.
#   - Common files are auto-deleted at job end; use --purge to free space
#     mid-job.
#
# Usage:
#   llio_transfer_bench.sh [SRC_DIR] [file1 file2 ...]
#
# Defaults:
#   SRC_DIR=$HOME/models/qwen35/9b
#   files = Qwen3.5-9B-UD-Q4_K_XL.gguf  (single, smallest non-mmproj file)

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="${1:-$HOME/models/qwen35/9b}"
shift || true
if (( $# == 0 )); then
    FILES=( Qwen3.5-9B-UD-Q4_K_XL.gguf )
else
    FILES=( "$@" )
fi

if ! command -v llio_transfer >/dev/null 2>&1; then
    echo "ERROR: llio_transfer not on PATH (only available in pjsub jobs)" >&2
    exit 1
fi

echo "============================================================"
echo "llio_transfer bench"
echo "host: $(hostname)  date: $(date -u +%FT%TZ)"
echo "src:  $SRC_DIR"
echo "files: ${FILES[*]}"
echo "============================================================"

# Build absolute paths.
declare -a PATHS
for f in "${FILES[@]}"; do
    PATHS+=( "$SRC_DIR/$f" )
done

echo
echo "### purge any existing common-file cache (ignore errors) ###"
llio_transfer --purge "${PATHS[@]}" 2>&1 || true

echo
echo "### llio_transfer --sync (time it) ###"
total_bytes=0
for p in "${PATHS[@]}"; do
    if [[ -f "$p" ]]; then
        total_bytes=$(( total_bytes + $(stat -c %s "$p") ))
    fi
done
# NB: the `stat` above opens the inode but not the data; per the manual
# that should be safe, but if `llio_transfer --sync` errors with
# "cache already exists", remove the stat loop and trust the file list.

t0=$(date +%s.%N)
llio_transfer --sync "${PATHS[@]}"
rc=$?
t1=$(date +%s.%N)
dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b - a}')
echo "llio_transfer --sync rc=$rc  time=${dt}s  bytes=${total_bytes}"
if (( total_bytes > 0 )); then
    awk -v n="$total_bytes" -v t="$dt" \
        'BEGIN{printf "transfer bandwidth: %.3f MB/s (%.3f MiB/s)\n",
               n/1e6/t, n/1048576/t}'
fi

if (( rc != 0 )); then
    echo "WARNING: llio_transfer reported errors; results below may not"
    echo "         reflect L2-cached access. Re-run in a fresh job."
fi

echo
echo "### bench_fopen on original path (now backed by L2 cache) ###"
for p in "${PATHS[@]}"; do
    if [[ ! -x "$SCRIPT_DIR/bench_fopen" ]]; then
        echo "bench_fopen not built; skipping" >&2
        break
    fi
    echo
    echo "--- $p ---"
    "$SCRIPT_DIR/bench_fopen" "$p" 16777216 2
done

echo
echo "### bench_mmap on original path ###"
for p in "${PATHS[@]}"; do
    if [[ ! -x "$SCRIPT_DIR/bench_mmap" ]]; then
        echo "bench_mmap not built; skipping" >&2
        break
    fi
    echo
    echo "--- $p ---"
    "$SCRIPT_DIR/bench_mmap" "$p" seq8 2 0
done

echo
echo "### cleanup ###"
llio_transfer --purge "${PATHS[@]}" 2>&1 || true
echo "done."
