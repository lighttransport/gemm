#!/bin/sh
# Launch the DA3 side-by-side compare server.
#
# Two launch modes:
#
#   1) Single variant (legacy):
#        MODEL_DIR=/path/to/da3-small ./server/run-da3-compare.sh
#
#   2) Multi-variant (one server, pick via the page dropdown):
#        VARIANTS="small:/mnt/disk01/models/da3-small,base:/mnt/disk01/models/da3-base,\
#        large:/mnt/disk01/models/da3-large,large-1.1:/mnt/disk01/models/da3-large-1.1,\
#        giant:/mnt/disk01/models/da3-giant,giant-1.1:/mnt/disk01/models/da3-giant-1.1,\
#        nested-1.0:/mnt/disk01/models/da3nested-giant-large,\
#        nested-1.1:/mnt/disk01/models/da3nested-giant-large-1.1" \
#           ./server/run-da3-compare.sh
#
# Other env:
#   DEFAULT_VARIANT  variant name shown first in dropdown (default: first entry)
#   PYTHON           python interpreter (default: python3)
#   HOST             bind host (default: 0.0.0.0)
#   PORT             server port (default: 8083)
#   DEVICE           torch device (default: cuda)
#   LOG_DIR          where server log lands (default: ./logs)
#   DISABLE          comma-sep: ours-cpu,ours-cuda,pytorch
#   TORCH_UNLOAD_AFTER=1  drop pytorch model + empty CUDA cache after every
#                         infer (avoids OOM sharing VRAM with ours-cuda)
#
# Ctrl-C stops the server.

set -eu

PYTHON="${PYTHON:-python3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8083}"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="${LOG_DIR:-./logs}"
DISABLE="${DISABLE:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPU_BIN="${CPU_BIN:-$REPO_ROOT/cpu/da3/test_da3}"
CUDA_BIN="${CUDA_BIN:-$REPO_ROOT/cuda/da3/test_cuda_da3}"

die() { echo "error: $*" >&2; exit 1; }

[ -x "$CPU_BIN" ]  || echo "warn: $CPU_BIN not built (ours-cpu will fail)" >&2
[ -x "$CUDA_BIN" ] || echo "warn: $CUDA_BIN not built (ours-cuda will fail)" >&2

SERVER_ARGS=""
if [ -n "${VARIANTS:-}" ]; then
    SERVER_ARGS="--variants $VARIANTS"
    [ -n "${DEFAULT_VARIANT:-}" ] && SERVER_ARGS="$SERVER_ARGS --default-variant $DEFAULT_VARIANT"
elif [ -n "${MODEL_DIR:-}" ]; then
    CKPT="${CKPT:-$MODEL_DIR/model.safetensors}"
    [ -f "$CKPT" ] || die "missing $CKPT"
    SERVER_ARGS="--model-dir $MODEL_DIR --ckpt $CKPT"
else
    die "set MODEL_DIR=... or VARIANTS=name:dir,name:dir,..."
fi

mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
LOG="$LOG_DIR/da3-compare.log"

echo "[run-da3-compare] -> http://$HOST:$PORT   (log: $LOG)"
[ -n "${VARIANTS:-}" ]  && echo "    VARIANTS=$VARIANTS"
[ -n "${MODEL_DIR:-}" ] && echo "    MODEL_DIR=$MODEL_DIR"
echo "    CPU_BIN=$CPU_BIN"
echo "    CUDA_BIN=$CUDA_BIN"
echo "    DEVICE=$DEVICE  DISABLE=${DISABLE:-<none>}  TORCH_UNLOAD_AFTER=${TORCH_UNLOAD_AFTER:-0}"
echo

# shellcheck disable=SC2086
exec "$PYTHON" "$REPO_ROOT/ref/da3/da3_server.py" \
    --host "$HOST" --port "$PORT" \
    --cpu-bin "$CPU_BIN" --cuda-bin "$CUDA_BIN" \
    --device "$DEVICE" \
    --web-root "$REPO_ROOT/web" \
    --disable "$DISABLE" \
    $SERVER_ARGS \
    2>&1 | tee "$LOG"
