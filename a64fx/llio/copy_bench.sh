#!/bin/bash
# copy_bench.sh -- time `cp` from a source dir to $DST (default $PJM_LOCALTMP)
#
# Usage:
#   copy_bench.sh [SRC_DIR] [DST_DIR]
#
# Defaults:
#   SRC_DIR=$HOME/models/qwen35/9b
#   DST_DIR=$PJM_LOCALTMP/llio_bench
#
# Reports per-file wall-clock cp time and bandwidth (file-size / dt) in
# MB/s and MiB/s. Uses `cp --reflink=never` to defeat any CoW-style
# tricks the underlying FS might play.

set -eu

SRC_DIR="${1:-$HOME/models/qwen35/9b}"
DST_BASE="${2:-${PJM_LOCALTMP:-/tmp}}"
DST_DIR="${DST_BASE%/}/llio_bench"

if [[ -z "${PJM_LOCALTMP:-}" && -z "${2:-}" ]]; then
    echo "WARNING: \$PJM_LOCALTMP not set; falling back to /tmp" >&2
    echo "         (request LLIO via pjsub --llio localtmp-size=...Gi)" >&2
fi

if [[ ! -d "$SRC_DIR" ]]; then
    echo "ERROR: source dir does not exist: $SRC_DIR" >&2
    exit 1
fi

mkdir -p "$DST_DIR"

echo "src: $SRC_DIR"
echo "dst: $DST_DIR"
echo "host: $(hostname)  date: $(date -u +%FT%TZ)"

# Try to report destination capacity.
df -h "$DST_DIR" 2>/dev/null || true
echo

# File list. Override by editing or by exporting FILES_OVERRIDE="a b c".
FILES=(
    mmproj-F32.gguf
    mmproj-BF16.gguf
    mmproj-F16.gguf
    Qwen3.5-9B-UD-Q4_K_XL.gguf
    Qwen3.5-9B-UD-Q8_K_XL.gguf
    Qwen3.5-9B-BF16.gguf
)
if [[ -n "${FILES_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    FILES=( ${FILES_OVERRIDE} )
fi

# Header
printf "%-32s %14s %10s %12s %12s\n" "file" "size(bytes)" "time(s)" "MB/s" "MiB/s"
printf "%-32s %14s %10s %12s %12s\n" "----" "-----------" "-------" "----" "-----"

total_bytes=0
total_dt=0
for f in "${FILES[@]}"; do
    src="$SRC_DIR/$f"
    if [[ ! -f "$src" ]]; then
        printf "%-32s  (missing in src)\n" "$f"
        continue
    fi
    bytes=$(stat -c %s "$src")
    # Wipe stale dst so the cp actually copies.
    rm -f "$DST_DIR/$f"
    # Warm cache hint: keep this honest. We measure the *source*-read +
    # *dst*-write cost together, which is what real first-touch costs.
    t0=$(date +%s.%N)
    cp --reflink=never "$src" "$DST_DIR/$f"
    sync   # flush to dst before stopping the clock
    t1=$(date +%s.%N)
    dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.6f", b - a}')
    mb=$(awk -v n="$bytes" -v t="$dt" 'BEGIN{printf "%.3f", n / 1e6 / t}')
    mib=$(awk -v n="$bytes" -v t="$dt" 'BEGIN{printf "%.3f", n / 1048576 / t}')
    printf "%-32s %14d %10s %12s %12s\n" "$f" "$bytes" "$dt" "$mb" "$mib"
    total_bytes=$(( total_bytes + bytes ))
    total_dt=$(awk -v a="$total_dt" -v b="$dt" 'BEGIN{printf "%.6f", a + b}')
done

printf "%-32s %14d %10s %12s %12s\n" "TOTAL" "$total_bytes" "$total_dt" \
    "$(awk -v n="$total_bytes" -v t="$total_dt" 'BEGIN{printf "%.3f", n/1e6/t}')" \
    "$(awk -v n="$total_bytes" -v t="$total_dt" 'BEGIN{printf "%.3f", n/1048576/t}')"

echo
echo "destination contents:"
ls -lh "$DST_DIR"
