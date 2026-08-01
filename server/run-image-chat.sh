#!/bin/sh
# Launch the sidecar for the image-chat page (web/image_chat.html, served at
# /image_chat on the Python sidecar). Thin wrapper over run-llm-compare.sh
# that only wires up a single Qwen-family VLM variant — image_chat.html
# auto-picks the first Qwen VLM variant anyway.
#
# Required env:
#   QWEN_GGUF     path to the Qwen VLM GGUF (main model)
#   QWEN_MMPROJ   path to the Qwen vision projector GGUF
#
# Optional env:
#   QWEN_HF       HF path for the pytorch-cuda fallback (default: empty)
#   VARIANT_NAME  variant id shown in /v1/models (default: qwen36-27b)
#   PORT          listen port (default: 8089)
#   HOST          listen host (default: 0.0.0.0)
#   DISABLE       comma-sep backends to skip. Default disables the non-Qwen
#                 backends so /v1/models lists only ours-{cpu,cuda}-qwen-vlm
#                 (+ pytorch-cuda if QWEN_HF is set).
#   LOG_DIR       where the sidecar log lands (default: ./logs)
#
# Any other env var honored by run-llm-compare.sh is forwarded verbatim
# (OURS_CPU_THREADS, REF_PYTHON, CPU_QWEN_VLM_BIN, CUDA_QWEN_VLM_BIN, ...).
#
# Example:
#   QWEN_GGUF=/mnt/disk01/models/qwen36/27b/Qwen3.6-27B-UD-Q2_K_XL.gguf \
#   QWEN_MMPROJ=/mnt/disk01/models/qwen36/27b/mmproj-F16.gguf \
#   QWEN_HF=unsloth/Qwen3.6-27B \
#       ./server/run-image-chat.sh
#
# Then open: http://localhost:8089/image_chat
#            (or http://localhost:8080/image_chat via the C server)

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

VARIANT_NAME="${VARIANT_NAME:-qwen36-27b}"
QWEN_HF="${QWEN_HF:-}"

: "${QWEN_GGUF:?set QWEN_GGUF=/path/to/qwen-vlm.gguf}"
: "${QWEN_MMPROJ:?set QWEN_MMPROJ=/path/to/mmproj-F16.gguf}"

# Trim the backend list to just the Qwen VLM path. The user can override by
# exporting DISABLE explicitly before invoking this script.
DEFAULT_DISABLE="ours-cpu-llm,ours-cpu-vlm,ours-cuda-vlm"
export DISABLE="${DISABLE:-$DEFAULT_DISABLE}"

# Build the VARIANTS spec. Trailing `:` leaves hf_path empty when QWEN_HF is
# unset, which the sidecar parses as "no pytorch-cuda reference".
export VARIANTS="${VARIANT_NAME}=vlm:${QWEN_GGUF}:${QWEN_MMPROJ}:${QWEN_HF}"
export DEFAULT_VARIANT="${DEFAULT_VARIANT:-$VARIANT_NAME}"

PORT_SHOWN="${PORT:-8089}"
echo "[run-image-chat] image-chat demo"
echo "    variant : ${VARIANT_NAME}"
echo "    gguf    : ${QWEN_GGUF}"
echo "    mmproj  : ${QWEN_MMPROJ}"
echo "    hf      : ${QWEN_HF:-<none>}"
echo "    disable : ${DISABLE}"
echo
echo "  open:  http://localhost:${PORT_SHOWN}/image_chat"
echo "         http://localhost:8080/image_chat   (via C diffusion-server)"
echo

exec "$SCRIPT_DIR/run-llm-compare.sh"
