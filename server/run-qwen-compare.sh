#!/bin/sh
# Launch the Qwen-Image side-by-side compare server.
#
# Usage:
#   VARIANTS="q4_0:/mnt/disk01/models/qwen-image,\
#   fp8:/mnt/disk01/models/qwen-image-st" \
#       ./server/run-qwen-compare.sh
#
# A variant entry is "name:PATH" or "name:DITFILE@ROOT" where:
#   - PATH = a model DIRECTORY containing:
#       <dir>/diffusion_models/*.{safetensors,gguf}  (or diffusion-models/)
#       <dir>/vae/*.safetensors
#       <dir>/text_encoders/* or text-encoder/*
#   - PATH = a direct .safetensors/.gguf DIT FILE (vae/encoder auto-detected from the
#     model root = the dit's grandparent dir). Lets several dits in one
#     diffusion-models/ dir register as distinct variants sharing vae/encoder.
#   - DITFILE@ROOT = a dit file whose vae/encoder come from a DIFFERENT model ROOT.
#     Used to run the canonical FP8 dit (in qwen-image-st/, which has only a
#     .safetensors encoder) with the shared qwen-image gguf encoder + VAE.
#
# The /qwen-image-fp4 demo compares native FP4 (W4A4 OMMA) against FP8:
#   VARIANTS="fp4:/mnt/disk01/models/qwen-image/diffusion-models/qwen-image-fp4-omma.safetensors,\
#   fp8:/mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors@/mnt/disk01/models/qwen-image,\
#   fp8-repack:/mnt/disk01/models/qwen-image/diffusion-models/qwen-image-fp4-repack-fp8.safetensors" \
#       ./server/run-qwen-compare.sh
# (If VARIANTS is unset, those three are used automatically when present.)
#
# Other env:
#   DEFAULT_VARIANT  variant name shown first in dropdown (default: first entry)
#   PYTHON           python interpreter (default: python3)
#   REF_PYTHON       python used for pytorch shell-out (default: same as PYTHON)
#   HOST             bind host (default: 0.0.0.0)
#   PORT             server port (default: 8085)
#   LOG_DIR          where server log lands (default: ./logs)
#   DISABLE          comma-sep: ours-cpu,ours-cuda,pytorch
#
# Ctrl-C stops the server.

set -eu

PYTHON="${PYTHON:-python3}"
REF_PYTHON="${REF_PYTHON:-$PYTHON}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8085}"
LOG_DIR="${LOG_DIR:-./logs}"
DISABLE="${DISABLE:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPU_BIN="${CPU_BIN:-$REPO_ROOT/cpu/qwen_image/test_qwen_image}"
CUDA_BIN="${CUDA_BIN:-$REPO_ROOT/cuda/qimg/test_cuda_qimg}"
REF_SCRIPT="${REF_SCRIPT:-$REPO_ROOT/ref/qwen_image/gen_diffusers_reference.py}"

die() { echo "error: $*" >&2; exit 1; }

[ -x "$CPU_BIN" ]  || echo "warn: $CPU_BIN not built (ours-cpu will fail)" >&2
[ -x "$CUDA_BIN" ] || echo "note: $CUDA_BIN not built (ours-cuda will be unavailable)" >&2

# Default to the FP4-vs-FP8 comparison when VARIANTS is unset and the dits exist.
#   fp4        = native W4A4 OMMA (Nunchaku NVFP4)
#   fp8        = canonical ComfyUI FP8 (qwen_image_fp8_e4m3fn), run with the shared
#                qwen-image gguf encoder + VAE via @ROOT (its own dir has only a .safetensors enc)
#   fp8-repack = Nunchaku FP4 -> dense FP8 (Stage-1)
if [ -z "${VARIANTS:-}" ]; then
    QROOT=/mnt/disk01/models/qwen-image
    QDM=$QROOT/diffusion-models
    FP8_CANON=/mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors
    DEF=""
    [ -f "$QDM/qwen-image-fp4-omma.safetensors" ]       && DEF="fp4:$QDM/qwen-image-fp4-omma.safetensors"
    [ -f "$FP8_CANON" ]                                 && DEF="${DEF:+$DEF,}fp8:$FP8_CANON@$QROOT"
    [ -f "$QDM/qwen-image-fp4-repack-fp8.safetensors" ] && DEF="${DEF:+$DEF,}fp8-repack:$QDM/qwen-image-fp4-repack-fp8.safetensors"
    [ -n "$DEF" ] && VARIANTS="$DEF" && echo "[run-qwen-compare] VARIANTS unset; defaulting to: $VARIANTS"
fi
[ -n "${VARIANTS:-}" ] || die "set VARIANTS=name:dir-or-ditfile,name:... (see header for the FP4 example)"

SERVER_ARGS="--variants $VARIANTS"
[ -n "${DEFAULT_VARIANT:-}" ] && SERVER_ARGS="$SERVER_ARGS --default-variant $DEFAULT_VARIANT"

mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
LOG="$LOG_DIR/qwen-compare.log"

echo "[run-qwen-compare] -> http://$HOST:$PORT   (log: $LOG)"
echo "    VARIANTS=$VARIANTS"
echo "    CPU_BIN=$CPU_BIN"
echo "    CUDA_BIN=$CUDA_BIN"
echo "    REF_SCRIPT=$REF_SCRIPT  REF_PYTHON=$REF_PYTHON"
echo "    DISABLE=${DISABLE:-<none>}"
echo

# shellcheck disable=SC2086
exec "$PYTHON" "$REPO_ROOT/ref/qwen_image/qwen_image_server.py" \
    --host "$HOST" --port "$PORT" \
    --cpu-bin "$CPU_BIN" --cuda-bin "$CUDA_BIN" \
    --ref-script "$REF_SCRIPT" --ref-python "$REF_PYTHON" \
    --web-root "$REPO_ROOT/web" \
    --disable "$DISABLE" \
    $SERVER_ARGS \
    2>&1 | tee "$LOG"
