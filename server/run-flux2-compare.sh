#!/bin/sh
# Launch the Flux.2-klein side-by-side compare server.
#
# Usage:
#   VARIANTS="distilled:/mnt/disk01/models/klein2-4b,\
#   base:/mnt/disk01/models/klein2-4b" \
#       TOK=/mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf \
#       ./server/run-flux2-compare.sh
#
# Each variant directory is a klein2-4b model root with:
#   <dir>/diffusion_models/*-base-*fp8.safetensors    (base variant)
#   <dir>/diffusion_models/*-klein-*fp8.safetensors   (distilled variant)
#   <dir>/vae/flux2-vae.safetensors
#   <dir>/text_encoder/   (sharded Qwen3-VL safetensors)
#
# Other env:
#   TOK              Qwen3-VL GGUF used for tokenizer vocab (required)
#   DEFAULT_VARIANT  variant name shown first (default: first entry)
#   PYTHON           python interpreter (default: python3)
#   REF_PYTHON       python used for pytorch shell-out (default: same as PYTHON)
#   HOST             bind host (default: 0.0.0.0)
#   PORT             server port (default: 8086)
#   LOG_DIR          where server log lands (default: ./logs)
#   DISABLE          comma-sep: ours-cpu,ours-cuda,ours-hip,pytorch
#
# Ctrl-C stops the server.

set -eu

PYTHON="${PYTHON:-python3}"
REF_PYTHON="${REF_PYTHON:-$PYTHON}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8086}"
LOG_DIR="${LOG_DIR:-./logs}"
DISABLE="${DISABLE:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPU_BIN="${CPU_BIN:-$REPO_ROOT/cpu/flux2/test_flux2}"
CUDA_BIN="${CUDA_BIN:-$REPO_ROOT/cuda/flux2/test_cuda_flux2}"
HIP_BIN="${HIP_BIN:-$REPO_ROOT/rdna4/flux2/test_hip_flux2}"
REF_SCRIPT="${REF_SCRIPT:-$REPO_ROOT/ref/flux2_klein/gen_reference.py}"

die() { echo "error: $*" >&2; exit 1; }

[ -x "$CPU_BIN" ]  || echo "warn: $CPU_BIN not built (ours-cpu will fail)" >&2
[ -x "$CUDA_BIN" ] || echo "note: $CUDA_BIN not built (ours-cuda will be unavailable)" >&2
[ -x "$HIP_BIN" ]  || echo "note: $HIP_BIN not built (ours-hip will be unavailable)" >&2

[ -n "${VARIANTS:-}" ] || die "set VARIANTS=name:dir,name:dir,..."
[ -n "${TOK:-}" ] || die "set TOK=/path/to/Qwen3-VL-*.gguf (tokenizer vocab)"

SERVER_ARGS="--variants $VARIANTS --tok $TOK"
[ -n "${DEFAULT_VARIANT:-}" ] && SERVER_ARGS="$SERVER_ARGS --default-variant $DEFAULT_VARIANT"

mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
LOG="$LOG_DIR/flux2-compare.log"

echo "[run-flux2-compare] -> http://$HOST:$PORT   (log: $LOG)"
echo "    VARIANTS=$VARIANTS"
echo "    TOK=$TOK"
echo "    CPU_BIN=$CPU_BIN"
echo "    CUDA_BIN=$CUDA_BIN"
echo "    HIP_BIN=$HIP_BIN"
echo "    REF_SCRIPT=$REF_SCRIPT  REF_PYTHON=$REF_PYTHON"
echo "    DISABLE=${DISABLE:-<none>}"
echo

# shellcheck disable=SC2086
exec "$PYTHON" "$REPO_ROOT/ref/flux2_klein/flux2_server.py" \
    --host "$HOST" --port "$PORT" \
    --cpu-bin "$CPU_BIN" --cuda-bin "$CUDA_BIN" --hip-bin "$HIP_BIN" \
    --ref-script "$REF_SCRIPT" --ref-python "$REF_PYTHON" \
    --web-root "$REPO_ROOT/web" \
    --disable "$DISABLE" \
    $SERVER_ARGS \
    2>&1 | tee "$LOG"
