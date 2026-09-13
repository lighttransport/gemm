#!/bin/sh
# Start llmgr with the generic llama.cpp-backed Qwen3.6 adapter selected.
# Set QWEN36_MODEL and LLMGR_LLAMA_SERVER (or put llama-server on PATH).
set -eu

: "${QWEN36_MODEL:?set QWEN36_MODEL to a GGUF model path}"
export LLMGR_DEFAULT_MODEL=qwen36
exec "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/run_llmgr.sh" "$@"
