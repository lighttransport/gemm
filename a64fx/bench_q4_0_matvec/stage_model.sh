#!/bin/sh
# Chunked staging of the 12B BF16 model to node-local /local, 1 GiB at a time.
#
# WHY: a single `cp` of the 24 GB model accumulates ~24 GB of DIRTY page-cache
# pages -> writeback pressure -> kswapd thrashes -> the INTERACTIVE session HANGS
# (and can OOM the agent). Copying 1 GiB then fsync()'ing keeps dirty pages bounded
# to ~1 GiB so kswapd never storms. NEVER `cp` the whole model in an interactive
# session; always use this (or run it detached / from a batch job).
#
# Usage: sh stage_model.sh   (idempotent; skips if already fully staged)
set -e
SRC=${SRC:-/home/u14346/models/gemma4/12b/gemma-4-12b-it-BF16.gguf}
DST=${DST:-/local/u14346/gemma-4-12b-it-BF16.gguf}
mkdir -p "$(dirname "$DST")"
SZ=$(stat -c%s "$SRC")
if [ -e "$DST" ] && [ "$(stat -c%s "$DST" 2>/dev/null)" = "$SZ" ]; then
    echo "already staged: $DST ($SZ bytes)"; exit 0
fi
CHUNKS=$(( (SZ + 1073741823) / 1073741824 ))   # ceil(SZ / 1 GiB)
echo "staging $SZ bytes in $CHUNKS x 1 GiB chunks (fsync per chunk)..."
i=0
while [ "$i" -lt "$CHUNKS" ]; do
    # bs=1M -> skip/seek are in MiB; each chunk = 1024 MiB = 1 GiB.
    # conv=notrunc,fsync: don't truncate dst, flush to disk after each chunk so
    # dirty pages stay ~1 GiB (no writeback storm / kswapd hang).
    dd if="$SRC" of="$DST" bs=1M count=1024 skip=$((i*1024)) seek=$((i*1024)) \
       conv=notrunc,fsync status=none
    i=$((i+1))
    echo "  chunk $i/$CHUNKS done"
    sleep 1   # let writeback/reclaim settle between chunks
done
echo "staged: $DST"
ls -la "$DST" | awk '{print $5, $NF}'
