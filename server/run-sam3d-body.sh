#!/bin/sh
# Launch the SAM 3D Body compare server.
#
# Usage:
#   ./server/run-sam3d-body.sh
#
# Env:
#   PYTHON         python interpreter (default: python3)
#   HOST           bind host (default: 0.0.0.0)
#   PORT           server port (default: 8088)
#   SFT            sam3d_body safetensors dir
#                  (default: /mnt/disk01/models/sam3d-body/safetensors)
#   MHR            MHR assets dir (default: $SFT)
#   RT_DETR_MODEL  RT-DETR-S safetensors for --auto-bbox
#                  (default: /mnt/disk01/models/rt_detr_s/model.safetensors)
#   CPU_BIN        cpu/sam3d_body/test_sam3d_body
#   CUDA_BIN       cuda/sam3d_body/test_cuda_sam3d_body
#   DISABLE        comma-sep: ours-cpu,ours-cuda
#   LOG_DIR        log directory (default: ./logs)
#
# Ctrl-C stops the server.

set -eu

PYTHON="${PYTHON:-python3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8088}"
LOG_DIR="${LOG_DIR:-./logs}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SFT="${SFT:-/mnt/disk01/models/sam3d-body/safetensors}"
MHR="${MHR:-$SFT}"
RT_DETR_MODEL="${RT_DETR_MODEL:-/mnt/disk01/models/rt_detr_s/model.safetensors}"
CPU_BIN="${CPU_BIN:-$REPO_ROOT/cpu/sam3d_body/test_sam3d_body}"
CUDA_BIN="${CUDA_BIN:-$REPO_ROOT/cuda/sam3d_body/test_cuda_sam3d_body}"
DISABLE="${DISABLE:-}"

mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
LOG="$LOG_DIR/sam3d-body.log"

echo "[run-sam3d-body] -> http://$HOST:$PORT   (log: $LOG)"
echo "    SFT=$SFT  MHR=$MHR"
echo "    RT_DETR_MODEL=$RT_DETR_MODEL"
echo "    CPU_BIN=$CPU_BIN  CUDA_BIN=$CUDA_BIN"
echo "    DISABLE=${DISABLE:-<none>}"
echo

exec "$PYTHON" "$REPO_ROOT/ref/sam3d-body/sam3d_body_server.py" \
    --host "$HOST" --port "$PORT" \
    --safetensors-dir "$SFT" \
    --mhr-assets "$MHR" \
    --rt-detr-model "$RT_DETR_MODEL" \
    --cpu-bin "$CPU_BIN" \
    --cuda-bin "$CUDA_BIN" \
    --disable "$DISABLE" \
    --web-root "$REPO_ROOT/web" \
    2>&1 | tee "$LOG"
