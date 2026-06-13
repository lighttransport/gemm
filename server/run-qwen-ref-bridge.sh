#!/bin/sh
# Launch the PyTorch/ComfyUI FP8 reference bridge for Qwen-Image, then point the
# C diffusion-server at it so the web demo gets a "pytorch (comfyui fp8)" column.
#
# 1) Start the bridge (loads FP8 weights via ComfyUI internals, stays resident):
#       COMFYUI_DIR=/mnt/disk01/ComfyUI \
#       MODEL_DIR=/mnt/disk01/models/qwen-image-st \
#           ./server/run-qwen-ref-bridge.sh
#
# 2) Start diffusion-server with --qwen-ref-url http://127.0.0.1:8189
#    (or env QWEN_IMAGE_REF_URL). The web UI then shows a pytorch backend
#    checkbox; tick it alongside hip to compare.
#
# Single 16 GB GPU: the HIP backend holds the GPU resident, so run the PyTorch
# reference at a different time (not concurrently) or set BRIDGE_DEVICE.
set -eu

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8189}"
COMFYUI_DIR="${COMFYUI_DIR:-/mnt/disk01/ComfyUI}"
MODEL_DIR="${MODEL_DIR:-/mnt/disk01/models/qwen-image-st}"
PYTHON="${PYTHON:-python3}"
EXTRA="${EXTRA:-}"            # e.g. EXTRA="--fast" for fp8 matrix-mult, or --preload

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BRIDGE="$REPO_ROOT/ref/qwen_image/qwen_ref_bridge.py"

[ -d "$COMFYUI_DIR" ] || echo "warn: COMFYUI_DIR not found: $COMFYUI_DIR" >&2
[ -d "$MODEL_DIR" ]   || echo "warn: MODEL_DIR not found: $MODEL_DIR" >&2

echo "[run-qwen-ref-bridge] http://$HOST:$PORT"
echo "    COMFYUI_DIR=$COMFYUI_DIR"
echo "    MODEL_DIR=$MODEL_DIR"
echo "    -> set diffusion-server --qwen-ref-url http://$HOST:$PORT"
echo

# shellcheck disable=SC2086
exec "$PYTHON" "$BRIDGE" \
    --host "$HOST" --port "$PORT" \
    --comfyui-dir "$COMFYUI_DIR" \
    --model-dir "$MODEL_DIR" \
    $EXTRA
