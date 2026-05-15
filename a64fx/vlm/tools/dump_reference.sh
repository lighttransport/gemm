#!/bin/sh
# dump_reference.sh - rebuild cpu/vlm with -DVLM_DUMP_REFERENCE=1 and dump
# the scalar vision encoder tensors for tensor_diff to consume.
#
# Usage:
#   ./tools/dump_reference.sh <model.gguf> <mmproj.gguf> <image> <out_dir>
#
# Env overrides:
#   CC=gcc | clang   (default: gcc)
#   IMG_SIZE=384     (passed to test_vision via 4th positional)
#
# Notes:
#   - The dump hook is triggered by VLM_DUMP_DIR (set by this script).
#   - cpu/vlm/Makefile has -mavx2/-mfma in its default CFLAGS; we override
#     CFLAGS so the -DVLM_DUMP_REFERENCE flag survives on aarch64.

set -eu

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <model.gguf> <mmproj.gguf> <image> <out_dir>" >&2
    exit 2
fi

MODEL="$1"
MMPROJ="$2"
IMAGE="$3"
OUT_DIR="$4"
CC="${CC:-gcc}"

HERE="$(cd "$(dirname "$0")" && pwd)"
CPU_VLM="$HERE/../../../cpu/vlm"
BIN="$CPU_VLM/test_vision"

mkdir -p "$OUT_DIR"

echo "[ref] building cpu/vlm with -DVLM_DUMP_REFERENCE=1 (CC=$CC)..."
make -C "$CPU_VLM" COMPILER="$CC" \
    CFLAGS="-O2 -DVLM_DUMP_REFERENCE=1 -Wall -Wextra" >/dev/null

if [ ! -x "$BIN" ]; then
    echo "[ref] cpu/vlm did not produce $BIN" >&2
    exit 1
fi

echo "[ref] dumping to $OUT_DIR..."
VLM_DUMP_DIR="$OUT_DIR" "$BIN" "$MODEL" "$MMPROJ" "$IMAGE"

if [ ! -f "$OUT_DIR/manifest.txt" ]; then
    echo "[ref] no manifest.txt produced — dump hook may not have triggered" >&2
    exit 1
fi

echo "[ref] done. Tensors in $OUT_DIR; manifest has $(wc -l < "$OUT_DIR/manifest.txt") entries."
