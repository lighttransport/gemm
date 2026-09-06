#!/bin/bash
# Generic GGUF split-shard staging helper.
# Given one shard path (or single-file GGUF), copy all shards to node-local storage.
# Usage:
#   stage_gguf_shards.sh /home/.../Model-00001-of-00002.gguf /local/models

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <src-shard-or-single.gguf> <dst-dir>" >&2
    exit 1
fi

SRC_SHARD1=$1
DST_DIR=$2

if [ ! -f "$SRC_SHARD1" ]; then
    echo "stage_gguf_shards: source not found: $SRC_SHARD1" >&2
    exit 2
fi

mkdir -p "$DST_DIR"

src_dir=$(cd "$(dirname "$SRC_SHARD1")" && pwd)
fname=$(basename "$SRC_SHARD1")

shards=("$src_dir/$fname")
if [[ "$fname" =~ ^(.*)-([0-9]+)-of-([0-9]+)\.gguf$ ]]; then
    pfx="${BASH_REMATCH[1]}"
    idx_w=${#BASH_REMATCH[2]}
    tot_raw="${BASH_REMATCH[3]}"
    tot_w=${#tot_raw}
    tot=$((10#${tot_raw}))
    total_tag=$(printf "%0${tot_w}d" "$tot")
    shards=()
    for ((i = 1; i <= tot; i++)); do
        idx=$(printf "%0${idx_w}d" "$i")
        shards+=("$src_dir/$pfx-$idx-of-$total_tag.gguf")
    done
fi

for src in "${shards[@]}"; do
    if [ ! -f "$src" ]; then
        echo "stage_gguf_shards: missing shard $src" >&2
        exit 3
    fi
done

for src in "${shards[@]}"; do
    base=$(basename "$src")
    dst="$DST_DIR/$base"
    want=$(stat -c '%s' "$src")
    if [ -f "$dst" ] && [ "$(stat -c '%s' "$dst" 2>/dev/null)" = "$want" ]; then
        echo "[$(hostname)] reuse $dst ($((want / 1024 / 1024)) MiB)"
        continue
    fi
    t0=$(date +%s)
    tmp="$dst.partial.$$"
    rm -f "$tmp"

    # Model files are commonly tens of GiB while an A64FX node has 32 GiB HBM.
    # A plain cp lets clean source and dirty destination pages accumulate in the
    # page cache.  Copy aligned whole-MiB blocks with direct I/O, then handle the
    # (sub-MiB) tail normally before the atomic publish.
    mib=$((1024 * 1024))
    whole=$((want / mib))
    tail=$((want % mib))
    if [ "$whole" -gt 0 ]; then
        dd if="$src" of="$tmp" bs="$mib" count="$whole" \
            iflag=direct,fullblock oflag=direct status=none
    fi
    if [ "$tail" -gt 0 ]; then
        dd if="$src" of="$tmp" bs="$mib" skip="$whole" seek="$whole" \
            count=1 iflag=fullblock conv=notrunc,fsync status=none
    else
        sync -f "$tmp"
    fi
    mv -f "$tmp" "$dst"
    got=$(stat -c '%s' "$dst" 2>/dev/null)
    t1=$(date +%s)
    if [ "$got" != "$want" ]; then
        echo "stage_gguf_shards: size mismatch src=$want dst=${got:-0} src=$src dst=$dst" >&2
        exit 4
    fi
    dt=$((t1 - t0))
    echo "[$(hostname)] copied $(basename "$src") to $dst in ${dt}s ($((want / 1024 / 1024)) MiB)"
done
