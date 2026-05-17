#!/bin/bash
# probe_sharedtmp_limit.sh -- empirically find the largest file that
# `llio_transfer --sync` accepts for the current sharedtmp allocation.
#
# Usage:
#   probe_sharedtmp_limit.sh [WORKDIR] [SIZES_GIB...]
#
# Defaults:
#   WORKDIR = ./synth (under a Lustre-backed path; must be LLIO-cached)
#   SIZES_GIB = 4 6 7 8 10 12 16
#
# Notes:
#   - Each probe creates a fresh random-data file with dd (the LLIO
#     cn-cache caches recent writes, so we sleep 150 s before --sync
#     to let dirty pages flush; otherwise --sync returns "File was
#     already cached"). dd cost is ~12 s/GiB.
#   - Between probes we --purge and sleep so the common-file cache
#     drains (release is asynchronous; we measured 60 s + sleep as
#     adequate to keep the next probe honest).
#   - Sparse files (truncate -s) are rejected by llio_transfer
#     ("System error(to be no longer common file)"). Use real data.

set -u

WORKDIR="${1:-./synth}"
shift || true
if (( $# == 0 )); then
    SIZES=( 4 6 7 8 10 12 16 )
else
    SIZES=( "$@" )
fi

mkdir -p "$WORKDIR"

if ! command -v llio_transfer >/dev/null 2>&1; then
    echo "ERROR: llio_transfer not on PATH (need pjsub job)" >&2
    exit 1
fi

printf "%-8s %-10s %-8s %s\n" "size_GiB" "result" "time_s" "note"

for sz in "${SIZES[@]}"; do
    f="$WORKDIR/probe_${sz}g.bin"
    rm -f "$f"
    dd if=/dev/urandom of="$f" bs=4M count=$(( sz * 256 )) status=none
    sync
    # Wait for cn-cache to flush so the next --sync doesn't trip
    # "File was already cached".
    sleep 150
    llio_transfer --purge "$f" >/dev/null 2>&1
    t0=$(date +%s.%N)
    out=$(llio_transfer --sync "$f" 2>&1)
    rc=$?
    t1=$(date +%s.%N)
    dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b - a}')
    if (( rc == 0 )); then
        result="OK"
    else
        if [[ "$out" == *"Not enough disk space"* ]]; then
            result="NO_SPACE"
        elif [[ "$out" == *"already cached"* ]]; then
            result="CACHE_DIRTY"   # retry with longer sleep
        else
            result="ERR_${rc}"
        fi
    fi
    printf "%-8d %-10s %-8s %s\n" "$sz" "$result" "$dt" "$out"
    llio_transfer --purge "$f" >/dev/null 2>&1
    rm -f "$f"
    # Let the common-file cache drain before the next probe.
    sleep 60
done
