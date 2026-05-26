#!/usr/bin/env bash
# Dump CUDA-reference intermediates for diffing against the ROCm path.
# Outputs land in cuda/trellis2/verify-dumps/ (override via OUTDIR).
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
SEED="${SEED:-42}"
# Shape-decoder mesh resolutions to extract (the SC-VAE net runs identically;
# only flexible_dual_grid_to_mesh grid_size varies). e.g. DECODER_RES=64,128,256,512
DECODER_RES="${DECODER_RES:-512}"
DUMP_PER_STEP_FLAG=""
if [[ "${DUMP_PER_STEP:-0}" == "1" ]]; then
    DUMP_PER_STEP_FLAG="--dump-per-step"
fi
# Per-DiT-block + per-decoder-layer dumps into verify-dumps/per_layer/ (large; f32).
DUMP_PER_BLOCK_FLAG=""
if [[ "${DUMP_PER_BLOCK:-0}" == "1" ]]; then
    DUMP_PER_BLOCK_FLAG="--dump-per-block"
fi

mkdir -p "$OUTDIR"

"$VENV_PY" "$SCRIPT_DIR/dump_ground_truth.py" \
    --image "$IMAGE" \
    --model-root "$MODEL_ROOT" \
    --dinov3 "$DINOV3" \
    --output-dir "$OUTDIR" \
    --pipeline-type 512 \
    --seed "$SEED" \
    --decoder-res "$DECODER_RES" \
    $DUMP_PER_STEP_FLAG \
    $DUMP_PER_BLOCK_FLAG

echo "Dumps -> $OUTDIR (manifest.json describes each entry)"
