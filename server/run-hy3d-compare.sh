#!/bin/sh
# Launch the Hunyuan3D-2.1 side-by-side compare server.
#
# Usage:
#   VARIANTS="v2_1:/mnt/disk01/models/Hunyuan3D-2.1" \
#       ./server/run-hy3d-compare.sh
#
# Each variant directory contains:
#   <dir>/conditioner.safetensors
#   <dir>/model.safetensors
#   <dir>/vae.safetensors
#
# Other env:
#   DEFAULT_VARIANT  variant name shown first (default: first entry)
#   PYTHON           python interpreter (default: python3)
#   REF_PYTHON       python used for pytorch shell-out (default: same as PYTHON)
#   HOST             bind host (default: 0.0.0.0)
#   PORT             server port (default: 8087)
#   LOG_DIR          where server log lands (default: ./logs)
#   DISABLE          comma-sep: ours-cpu,ours-cuda,pytorch
#
# Ctrl-C stops the server.

set -eu

PYTHON="${PYTHON:-python3}"
REF_PYTHON="${REF_PYTHON:-$PYTHON}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8087}"
LOG_DIR="${LOG_DIR:-./logs}"
DISABLE="${DISABLE:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPU_BIN="${CPU_BIN:-$REPO_ROOT/cpu/hy3d/test_hy3d}"
CUDA_BIN="${CUDA_BIN:-$REPO_ROOT/cuda/hy3d/test_cuda_hy3d}"
REF_SCRIPT="${REF_SCRIPT:-$REPO_ROOT/ref/hy3d/run_full_pipeline.py}"

die() { echo "error: $*" >&2; exit 1; }

[ -n "${VARIANTS:-}" ] || die "set VARIANTS=name:dir,name:dir,..."

SERVER_ARGS="--variants $VARIANTS"
[ -n "${DEFAULT_VARIANT:-}" ] && SERVER_ARGS="$SERVER_ARGS --default-variant $DEFAULT_VARIANT"

mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
LOG="$LOG_DIR/hy3d-compare.log"

echo "[run-hy3d-compare] -> http://$HOST:$PORT   (log: $LOG)"
echo "    VARIANTS=$VARIANTS"
echo "    CPU_BIN=$CPU_BIN"
echo "    CUDA_BIN=$CUDA_BIN"
echo "    REF_SCRIPT=$REF_SCRIPT  REF_PYTHON=$REF_PYTHON"
echo "    DISABLE=${DISABLE:-<none>}"
echo

# shellcheck disable=SC2086
exec "$PYTHON" "$REPO_ROOT/ref/hy3d/hy3d_server.py" \
    --host "$HOST" --port "$PORT" \
    --cpu-bin "$CPU_BIN" --cuda-bin "$CUDA_BIN" \
    --ref-script "$REF_SCRIPT" --ref-python "$REF_PYTHON" \
    --web-root "$REPO_ROOT/web" \
    --disable "$DISABLE" \
    $SERVER_ARGS \
    2>&1 | tee "$LOG"
