#!/usr/bin/env bash
# Prepare Qwen3.6-27B BF16 shard package for multinode runs.
#
# This script:
#  - finds the available shard family from a source file or directory,
#  - copies all shards into the package root (default:
#    ~/models/qwen36/27b/<N>nodes, N defaults to 12),
#  - verifies byte-level size for each copied shard.
#
# It does not split or quantize GGUF payloads. It only prepares the shard set.

set -euo pipefail

ROOT="${HOME}/models/qwen36/27b"
PKG_FALLBACK_NODES="${QWEN27B_PREPARE_NODES:-12}"

SRC_INPUT="${1:-$ROOT}"
DST_ROOT="${2:-$ROOT/${PKG_FALLBACK_NODES}nodes}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

die() {
    echo "prepare_27b_shards: $*" >&2
    exit 1
}

is_shard_name() {
    local f="$1"
    [[ "$f" =~ -[0-9]+-of-[0-9]+\.gguf$ ]]
}

collect_shards_from_path() {
    local path="$1"
    local dir base
    if [ -f "$path" ]; then
        if [[ "$(basename "$path")" == *.gguf && "$(basename "$path")" == *-of-* ]]; then
            local fname dir total width idx
            dir="$(cd "$(dirname "$path")" && pwd)"
            fname="$(basename "$path")"
            if [[ "$fname" =~ ^(.*)-([0-9]+)-of-([0-9]+)\.gguf$ ]]; then
                local pfx="${BASH_REMATCH[1]}"
                local i_width=${#BASH_REMATCH[2]}
                local total_raw="${BASH_REMATCH[3]}"
                local total=$((10#$total_raw))
                local total_w=${#total_raw}
                local shards=()
                for ((i = 1; i <= total; i++)); do
                    local idxf
                    idxf="$(printf "%0${i_width}d" "$i")"
                    shards+=("$dir/$pfx-$idxf-of-$(printf "%0${total_w}d" "$total").gguf")
                done
                SHARDS=("${shards[@]}")
                return 0
            fi
        fi
        # Single-file GGUF input (no split suffix) -> copy single file.
        SHARDS=("$path")
        return 0
    fi
    if [ ! -d "$path" ]; then
        die "input not found: $path"
    fi

    mapfile -t shards < <(find "$path" -maxdepth 3 -type f -name '*.gguf' | sort)
    if [ "${#shards[@]}" -eq 0 ]; then
        die "no GGUF files found in $path"
    fi

    # Prefer split-collection if present; detect a family from a first shard.
    local fam="" total=0 width=0 total_width=0
    for candidate in "${shards[@]}"; do
        base="$(basename "$candidate")"
        if [[ "$base" =~ ^(.*)-([0-9]+)-of-([0-9]+)\.gguf$ ]]; then
            if [ "${BASH_REMATCH[2]}" = "00001" ] || [ "${BASH_REMATCH[2]}" = "1" ]; then
                fam="${BASH_REMATCH[1]}"
                width="${#BASH_REMATCH[2]}"
                total="${BASH_REMATCH[3]}"
                total_width="${#BASH_REMATCH[3]}"
                break
            fi
        fi
    done

    if [ -n "$fam" ]; then
        local collected=()
        for ((i = 1; i <= total; i++)); do
            local idxf path_i
            idxf="$(printf "%0${width}d" "$i")"
            path_i="$path/$fam-$idxf-of-$(printf "%0${total_width}d" "$total").gguf"
            [ -f "$path_i" ] && collected+=("$path_i")
        done
        if [ "${#collected[@]}" -gt 0 ]; then
            SHARDS=("${collected[@]}")
            return 0
        fi
    fi

    SHARDS=("${shards[@]}")
}

copy_shards() {
    local -a arr=("$@")
    mkdir -p "$DST_ROOT"
    if [ ! -d "$DST_ROOT" ]; then
        die "destination is not a directory: $DST_ROOT"
    fi

    for src in "${arr[@]}"; do
        [ -f "$src" ] || die "missing shard $src"
        local base dst src_sz dst_sz
        base="$(basename "$src")"
        dst="$DST_ROOT/$base"
        src_sz="$(stat -c '%s' "$src")"

        if [ -f "$dst" ] && [ "$(stat -c '%s' "$dst" 2>/dev/null || echo 0)" = "$src_sz" ]; then
            log "reuse $dst ($(numfmt --to=iec "$src_sz"))"
            continue
        fi

        log "copying $base ($(numfmt --to=iec "$src_sz")) -> $DST_ROOT"
        local t0 t1 dt mbps
        t0="$(date +%s)"
        if ! cp "$src" "$dst"; then
            die "copy failed: $src -> $dst"
        fi
        t1="$(date +%s)"
        dt=$((t1 - t0))
        dst_sz="$(stat -c '%s' "$dst" 2>/dev/null || echo 0)"
        if [ "$dst_sz" != "$src_sz" ]; then
            die "size mismatch after copy: src=$src_sz dst=$dst_sz ($src -> $dst)"
        fi
        if [ "$dt" -gt 0 ]; then
            mbps="$(awk -v b="$src_sz" -v s="$dt" 'BEGIN{printf "%.2f", (b/1024/1024)/s}')"
            log "copied $(numfmt --to=iec "$src_sz") in ${dt}s @ ${mbps} MiB/s"
        else
            log "copied $(numfmt --to=iec "$src_sz") in <1s"
        fi
    done
}

collect_shards_from_path "$SRC_INPUT"

if [ "${#SHARDS[@]}" -eq 0 ]; then
    die "no shards resolved from input: $SRC_INPUT"
fi

log "source input=$SRC_INPUT"
log "destination root=$DST_ROOT"
log "resolved $((${#SHARDS[@]})) shard(s)"
copy_shards "${SHARDS[@]}"

log "ready: $(for s in "${SHARDS[@]}"; do printf ' %s' "$(basename "$s")"; done)"
cat <<EOF > "$DST_ROOT/.manifest"
src_inputs=(${SHARDS[*]})
dst_root=$DST_ROOT
prepared_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

log "written manifest: $DST_ROOT/.manifest"
