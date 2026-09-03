#!/usr/bin/env bash
set -euo pipefail

# Start the Qwen3.8-Flash-Next (q38fn) runner as a local OpenAI-compatible
# endpoint for Codex and other coding-agent clients.
runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${QWEN38_MODEL:-/mnt/nvme01/models/q38nf/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
# 64K KV plus server scratch needs more headroom than the short-context bench.
# Override to 8192 on a larger card when maximizing MoE hit rate.
cache_mb="${QWEN38_MOE_CACHE_MB:-4096}"
context="${QWEN38_CONTEXT:-65536}"
port="${QWEN38_API_PORT:-8080}"
host="${QWEN38_API_HOST:-127.0.0.1}"
max_output="${QWEN38_MAX_OUTPUT:-4096}"
cpu_lib="${LLM_MOE_CPU_LIB:-/mnt/nvme02/work/llama.cpp/build-codex-hetero-dev2/bin/libggml-cpu.so.0.22.0}"
if [[ ! -r "${cpu_lib}" ]]; then
    cpu_lib=""
fi

if [[ ! -r "${model}" ]]; then
    echo "q38fn model not found: ${model}" >&2
    echo "Set QWEN38_MODEL to the first GGUF shard." >&2
    exit 1
fi

exec env \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
    LLM_MOE_REGISTER_HOST="${LLM_MOE_REGISTER_HOST:-1}" \
    LLM_MOE_COPY_PIPELINE="${LLM_MOE_COPY_PIPELINE:-1}" \
    LLM_MOE_LFU_CACHE="${LLM_MOE_LFU_CACHE:-1}" \
    LLM_MOE_STREAM_SLOTS="${LLM_MOE_STREAM_SLOTS:-4}" \
    LLM_MOE_CPU_LIB="${cpu_lib}" \
    LLM_MOE_CPU_PREFILL_MAX_COUNT="${LLM_MOE_CPU_PREFILL_MAX_COUNT:-2}" \
    LLM_MOE_CPU_PREFILL_MAX_JOBS="${LLM_MOE_CPU_PREFILL_MAX_JOBS:-160}" \
    LLM_BMAX="${LLM_BMAX:-128}" \
    python3 "${runner_dir}/codex_server.py" "${model}" \
    --runner "${runner_dir}/test_hip_llm" \
    --context "${context}" \
    --max-output "${max_output}" \
    --port "${port}" \
    --host "${host}" \
    --moe-cache-mb "${cache_mb}" \
    --coding "$@"
