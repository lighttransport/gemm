#!/usr/bin/env bash
# CUDA reference: image -> mesh + texture (textured GLB).
#
# Defaults sized for 16 GB VRAM (RTX 5060 Ti, sm_120):
#   PIPELINE_TYPE=512  (skip 1024 cascade — needs >24 GB)
#   TEXTURE_SIZE=1024
#
# Override via env:
#   IMAGE         conditioning image (RGBA preferred)
#   DINOV3        local timm DINOv3 ViT-L/16 safetensors
#   MODEL_ROOT    dir containing pipeline.json + ckpts/
#   OUTDIR        output dir
#   PIPELINE_TYPE 512 | 1024 | 1024_cascade | 1536_cascade
#   TEXTURE_SIZE  baked texture resolution
#   SEED          torch.manual_seed
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
OUTDIR="${OUTDIR:-/tmp/t2_cuda_e2e}"
PIPELINE_TYPE="${PIPELINE_TYPE:-512}"
TEXTURE_SIZE="${TEXTURE_SIZE:-1024}"
SEED="${SEED:-42}"

mkdir -p "$OUTDIR"

"$VENV_PY" "$SCRIPT_DIR/gen_image_to_3d.py" \
    --image "$IMAGE" \
    --model-root "$MODEL_ROOT" \
    --dinov3 "$DINOV3" \
    --output-dir "$OUTDIR" \
    --pipeline-type "$PIPELINE_TYPE" \
    --texture-size "$TEXTURE_SIZE" \
    --seed "$SEED"

echo "Outputs: $OUTDIR/{mesh.obj, textured.glb, basecolor.png, metalrough.png}"
