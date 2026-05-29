#!/usr/bin/env bash
# Dump CUDA-reference texgen artefacts (post-UV mesh, BVH inputs,
# attr volume, intermediate rast/mask/valid_pos/face_id/uvw, attrs)
# for diffing against the rdna4 cumesh_xatlas shim.
# See: rdna4/trellis2/docs/verify_texgen.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: $VENV_PY missing. Run ./setup_cuda.sh first." >&2
    exit 1
fi

IMAGE="${IMAGE:-$SCRIPT_DIR/../../cpu/trellis2/trellis2_repo/assets/example_image/T.png}"
DINOV3="${DINOV3:-/mnt/disk01/models/dinov3-vitl16/model.safetensors}"
MODEL_ROOT="${MODEL_ROOT:-$SCRIPT_DIR/model_root}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/verify-dumps}"
TEXSIZE="${TEXSIZE:-1024}"
SEED="${SEED:-42}"

mkdir -p "$OUTDIR"

"$VENV_PY" "$SCRIPT_DIR/dump_texgen.py" \
    --image "$IMAGE" \
    --model-root "$MODEL_ROOT" \
    --dinov3 "$DINOV3" \
    --output-dir "$OUTDIR" \
    --texture-size "$TEXSIZE" \
    --seed "$SEED"

echo "Texgen dumps + textured_cuda.glb -> $OUTDIR"
