#!/bin/sh
# Launch the TRELLIS.2 side-by-side compare server (Stage 1 structure mesh).
#
# Usage:
#   VARIANTS="default=/mnt/disk01/models/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors:/mnt/disk01/models/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors:/mnt/disk01/models/dinov3-vitl16/model.safetensors" \
#       ./server/run-trellis2-compare.sh
#
# Variant spec:
#   name=<stage1>:<decoder>:<dinov3>
#
# Other env:
#   DEFAULT_VARIANT  variant name shown first (default: first entry)
#   PYTHON           python interpreter (default: python3)
#   REF_PYTHON       python used for pytorch shell-out (default: same as PYTHON)
#   HOST             bind host (default: 0.0.0.0)
#   PORT             server port (default: 8088)
#   LOG_DIR          where server log lands (default: ./logs)
#   DISABLE          comma-sep backends to skip. Default 'ours-cpu' (slow).
#                    Use DISABLE='' to enable all advertised backends.
#
# Ctrl-C stops the server.

set -eu

PYTHON="${PYTHON:-python3}"
REF_PYTHON="${REF_PYTHON:-$PYTHON}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8088}"
LOG_DIR="${LOG_DIR:-./logs}"
DISABLE="${DISABLE-ours-cpu}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPU_BIN="${CPU_BIN:-$REPO_ROOT/cpu/trellis2/test_trellis2}"
CUDA_BIN="${CUDA_BIN:-$REPO_ROOT/cuda/trellis2/test_cuda_trellis2}"
REF_SCRIPT="${REF_SCRIPT:-$REPO_ROOT/ref/trellis2/gen_stage1_ref.py}"

die() { echo "error: $*" >&2; exit 1; }

[ -n "${VARIANTS:-}" ] || die "set VARIANTS=name=stage1:decoder:dinov3,..."

SERVER_ARGS="--variants $VARIANTS"
[ -n "${DEFAULT_VARIANT:-}" ] && SERVER_ARGS="$SERVER_ARGS --default-variant $DEFAULT_VARIANT"

mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
LOG="$LOG_DIR/trellis2-compare.log"

echo "[run-trellis2-compare] -> http://$HOST:$PORT   (log: $LOG)"
echo "    VARIANTS=$VARIANTS"
echo "    CPU_BIN=$CPU_BIN"
echo "    CUDA_BIN=$CUDA_BIN"
echo "    REF_SCRIPT=$REF_SCRIPT  REF_PYTHON=$REF_PYTHON"
echo "    DISABLE=${DISABLE:-<none>}"
echo

# shellcheck disable=SC2086
exec "$PYTHON" "$REPO_ROOT/ref/trellis2/trellis2_server.py" \
    --host "$HOST" --port "$PORT" \
    --cpu-bin "$CPU_BIN" --cuda-bin "$CUDA_BIN" \
    --ref-script "$REF_SCRIPT" --ref-python "$REF_PYTHON" \
    --web-root "$REPO_ROOT/web" \
    --disable "$DISABLE" \
    $SERVER_ARGS \
    2>&1 | tee "$LOG"
