#!/usr/bin/env bash
# Regenerate AOT Triton spconv hsacos for tex_dec on RX 9070 XT (gfx1201).
#
# What this does:
#   1. (optional) wipes ~/.triton/cache so leftover entries don't collide.
#   2. Runs the PyTorch tex_dec reference pipeline once. That triggers
#      FlexGEMM/Triton to compile the 9 spconv shapes into ~/.triton/cache.
#   3. Calls extract_hsacos.py to pick the matching kernels and copy them
#      under triton_aot/kernels/<tag>/{kernel.hsaco,kernel.json}.
#
# Prereqs (on the GPU box, gfx1201 only):
#   - ROCm 6.4+ with PyTorch ROCm + Triton 3.6 (the project's .venv works).
#   - FlexGEMM checkout at rdna4/trellis2/deps/FlexGEMM (already vendored).
#   - TRELLIS.2 weights cached under $HF_HOME (microsoft/TRELLIS.2-4B).
#   - A populated tex_dec dump dir with tex_slat_{feats,coords}.npy + cache_*.npy
#     (re-use /mnt/disk1/tmp/tex_knight_r512_fresh, or generate via the same
#     gen_stage2_ref.py without --skip-dit).
#
# Usage:
#   ./build_aot.sh [DUMP_DIR]
#     DUMP_DIR defaults to /mnt/disk1/tmp/tex_knight_r512_fresh
#
# Output: rdna4/trellis2/triton_aot/kernels/<shape_tag>/kernel.{hsaco,json}
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
DUMP_DIR="${1:-/mnt/disk1/tmp/tex_knight_r512_fresh}"
GEN_SCRIPT="$REPO_ROOT/ref/trellis2/gen_stage2_ref.py"

if ! command -v rocminfo >/dev/null 2>&1; then
    echo "warning: rocminfo not found; not verifying gpu arch" >&2
else
    arch="$(rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx[0-9]/ {print $2; exit}')"
    if [[ "$arch" != "gfx1201" ]]; then
        echo "error: this script targets gfx1201 (RX 9070 XT); found $arch" >&2
        echo "       AOT hsacos are arch-specific — re-tune autotune table for other GPUs" >&2
        exit 1
    fi
fi

if [[ ! -d "$DUMP_DIR" ]]; then
    echo "error: tex_dec dump dir not found: $DUMP_DIR" >&2
    echo "       create it first via: cd $REPO_ROOT/ref/trellis2 && python gen_stage2_ref.py ..." >&2
    exit 1
fi
if [[ ! -f "$DUMP_DIR/tex_slat_feats.npy" || ! -f "$DUMP_DIR/tex_slat_coords.npy" ]]; then
    echo "error: $DUMP_DIR is missing tex_slat_{feats,coords}.npy" >&2
    exit 1
fi
if [[ ! -f "$GEN_SCRIPT" ]]; then
    echo "error: bench script missing: $GEN_SCRIPT" >&2
    exit 1
fi

echo "==[1/3] wiping ~/.triton/cache spconv entries..."
TRITON_CACHE="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
if [[ -d "$TRITON_CACHE" ]]; then
    # only purge spconv-related dirs to keep unrelated kernels cached
    find "$TRITON_CACHE" -maxdepth 2 \
        -name 'sparse_submanifold_conv_fwd_masked_implicit_gemm*kernel.hsaco' \
        -printf '%h\n' | sort -u | xargs -r rm -rf
fi

echo "==[2/3] running PyTorch tex_dec to populate Triton cache..."
PY="${PYTHON:-python3}"
cd "$REPO_ROOT/ref/trellis2"
PYTHONPATH="${REPO_ROOT}/rdna4/trellis2/deps/FlexGEMM:${REPO_ROOT}/cpu/trellis2/trellis2_repo:${PYTHONPATH:-}" \
    "$PY" gen_stage2_ref.py \
        --skip-dit \
        --output-dir "$DUMP_DIR" \
        --resolution 512 \
        --no-image-out \
        2>&1 | tail -20 || {
    echo "note: gen_stage2_ref.py failed — that's ok if it errored AFTER decoding" >&2
    echo "      (image-out shim isn't needed for AOT extraction)" >&2
}

echo "==[3/3] extracting hsacos to $HERE/kernels/..."
"$PY" "$HERE/extract_hsacos.py"

echo
echo "done. files under $HERE/kernels/ are ready to commit."
echo "verify via: cd $HERE && gcc -O2 -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ \\"
echo "    -o test_bridge test_bridge.c -L/opt/rocm/lib -lamdhip64 && ./test_bridge"
