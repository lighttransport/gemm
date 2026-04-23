#!/bin/sh
# Launch the Qwen-Image side-by-side compare server.
#
# Usage:
#   VARIANTS="q4_0:/mnt/disk01/models/qwen-image,\
#   fp8:/mnt/disk01/models/qwen-image-st" \
#       ./server/run-qwen-compare.sh
#
# Each variant directory must contain:
#   <dir>/diffusion_models/*.{safetensors,gguf}  (or diffusion-models/)
#   <dir>/vae/*.safetensors
#   <dir>/text_encoders/* or text-encoder/*
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
CUDA_BIN="${CUDA_BIN:-$REPO_ROOT/cuda/qwen_image/test_cuda_qwen_image}"
REF_SCRIPT="${REF_SCRIPT:-$REPO_ROOT/ref/qwen_image/gen_diffusers_reference.py}"

die() { echo "error: $*" >&2; exit 1; }

[ -x "$CPU_BIN" ]  || echo "warn: $CPU_BIN not built (ours-cpu will fail)" >&2
[ -x "$CUDA_BIN" ] || echo "note: $CUDA_BIN not built (ours-cuda will be unavailable)" >&2

[ -n "${VARIANTS:-}" ] || die "set VARIANTS=name:dir,name:dir,..."

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
