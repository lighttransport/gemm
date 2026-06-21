#!/bin/bash
# Generic GGUF split-shard staging helper.
# Given one shard path (or single-file GGUF), copy all shards to shared local dir.
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
    cp "$src" "$dst"
    got=$(stat -c '%s' "$dst" 2>/dev/null)
    t1=$(date +%s)
    if [ "$got" != "$want" ]; then
        echo "stage_gguf_shards: size mismatch src=$want dst=${got:-0} src=$src dst=$dst" >&2
        exit 4
    fi
    dt=$((t1 - t0))
    echo "[$(hostname)] copied $(basename "$src") to $dst in ${dt}s ($((want / 1024 / 1024)) MiB)"
done
