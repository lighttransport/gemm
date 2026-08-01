#!/bin/sh
# Launch the LLM / VLM side-by-side compare server.
#
# Variant spec (comma-separated):
#   name=<kind>:<gguf>:<mmproj_or_empty>:<hf_path_or_empty>
#
#   kind    : llm  (text-only) | vlm  (image + text)
#   gguf    : GGUF for the C runner
#   mmproj  : vision projector GGUF — required for vlm, empty for llm
#   hf_path : HF model dir/hub id for the pytorch reference (empty if absent)
#
# A trailing `:family=qwen` forces the qwen-family VLM binary; otherwise the
# family is detected from the GGUF header (gemma4 vs qwen). Only used for vlm.
#
# Examples:
#   VARIANTS="\
#qwen35-4b=llm:/mnt/disk01/models/qwen3-5/4b/Qwen3.5-4B-UD-Q8_K_XL.gguf::Qwen/Qwen3.5-4B,\
#qwen35-9b=llm:/mnt/disk01/models/qwen3-5/9b/Qwen3.5-9B-UD-Q8_K_XL.gguf::,\
#gemma4-4b=vlm:/mnt/disk01/models/gemma4/4b/gemma-4-E4B-it-UD-Q8_K_XL.gguf:/mnt/disk01/models/gemma4/4b/mmproj-F16.gguf:google/gemma-4-4b-it,\
#qwen36-27b=vlm:/mnt/disk01/models/qwen36/27b/Qwen3.6-27B-UD-Q2_K_XL.gguf:/mnt/disk01/models/qwen36/27b/mmproj-F16.gguf:unsloth/Qwen3.6-27B" \
#     ./server/run-llm-compare.sh
#
# Other env:
#   DEFAULT_VARIANT  variant name shown first (default: first entry)
#   PYTHON           python interpreter (default: python3)
#   REF_PYTHON       python used for pytorch shell-out (default: same as PYTHON)
#   HOST             bind host (default: 0.0.0.0)
#   PORT             server port (default: 8089)
#   OURS_CPU_THREADS number of threads for the CPU LLM runner (default 8)
#   BUDGET           gemma4 reasoning-token budget (default 32)
#   LOG_DIR          where server log lands (default: ./logs)
#   DISABLE          comma-sep backends to skip (default: empty).
#                    Names: ours-cpu-llm, ours-cpu-vlm, ours-cuda-vlm,
#                           ours-cpu-qwen-vlm, ours-cuda-qwen-vlm, pytorch-cuda
#
# Ctrl-C stops the server.

set -eu

PYTHON="${PYTHON:-python3}"
REF_PYTHON="${REF_PYTHON:-$PYTHON}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8089}"
LOG_DIR="${LOG_DIR:-./logs}"
DISABLE="${DISABLE:-}"
OURS_CPU_THREADS="${OURS_CPU_THREADS:-8}"
BUDGET="${BUDGET:-32}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPU_LLM_BIN="${CPU_LLM_BIN:-$REPO_ROOT/common/test_transformer}"
CPU_VLM_BIN="${CPU_VLM_BIN:-$REPO_ROOT/cpu/vlm/test_gemma4_vision}"
CUDA_VLM_BIN="${CUDA_VLM_BIN:-$REPO_ROOT/cuda/vlm/test_cuda_gemma4_vlm}"
CPU_QWEN_VLM_BIN="${CPU_QWEN_VLM_BIN:-$REPO_ROOT/cpu/vlm/test_vision}"
CUDA_QWEN_VLM_BIN="${CUDA_QWEN_VLM_BIN:-}"
REF_SCRIPT="${REF_SCRIPT:-$REPO_ROOT/ref/llm/llm_ref.py}"

die() { echo "error: $*" >&2; exit 1; }

[ -n "${VARIANTS:-}" ] || die "set VARIANTS=name=kind:gguf:mmproj:hf,... (see header)"

SERVER_ARGS="--variants $VARIANTS"
[ -n "${DEFAULT_VARIANT:-}" ] && SERVER_ARGS="$SERVER_ARGS --default-variant $DEFAULT_VARIANT"

mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
LOG="$LOG_DIR/llm-compare.log"

echo "[run-llm-compare] -> http://$HOST:$PORT   (log: $LOG)"
echo "    VARIANTS=$VARIANTS"
echo "    CPU_LLM_BIN=$CPU_LLM_BIN"
echo "    CPU_VLM_BIN=$CPU_VLM_BIN"
echo "    CUDA_VLM_BIN=$CUDA_VLM_BIN"
echo "    CPU_QWEN_VLM_BIN=$CPU_QWEN_VLM_BIN"
echo "    CUDA_QWEN_VLM_BIN=${CUDA_QWEN_VLM_BIN:-<none>}"
echo "    REF_SCRIPT=$REF_SCRIPT  REF_PYTHON=$REF_PYTHON"
echo "    OURS_CPU_THREADS=$OURS_CPU_THREADS  BUDGET=$BUDGET"
echo "    DISABLE=${DISABLE:-<none>}"
echo

# shellcheck disable=SC2086
exec "$PYTHON" "$REPO_ROOT/ref/llm/llm_server.py" \
    --host "$HOST" --port "$PORT" \
    --cpu-llm-bin  "$CPU_LLM_BIN" \
    --cpu-vlm-bin  "$CPU_VLM_BIN" \
    --cuda-vlm-bin "$CUDA_VLM_BIN" \
    --cpu-qwen-vlm-bin  "$CPU_QWEN_VLM_BIN" \
    --cuda-qwen-vlm-bin "$CUDA_QWEN_VLM_BIN" \
    --ref-script "$REF_SCRIPT" --ref-python "$REF_PYTHON" \
    --web-root "$REPO_ROOT/web" \
    --n-threads "$OURS_CPU_THREADS" \
    --budget "$BUDGET" \
    --disable "$DISABLE" \
    $SERVER_ARGS \
    2>&1 | tee "$LOG"
