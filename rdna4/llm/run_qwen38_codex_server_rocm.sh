#!/usr/bin/env bash
set -euo pipefail

# Start the Qwen3.8-Flash-Next (q38fn) runner as a local OpenAI-compatible
# endpoint for Codex and other coding-agent clients.
runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${QWEN38_MODEL:-/mnt/nvme01/models/q38nf/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
# 8.0 GiB is the largest verified cache on a 16 GiB RX 9070 XT with 64K KV
# when the server's 128-token batch scratch is used. It keeps up to 89
# experts/layer resident; the old 4 GiB default kept only 28 and
# reduced decode from ~28 to ~18 tok/s through repeated PCIe uploads.
cache_mb="${QWEN38_MOE_CACHE_MB:-8000}"
context="${QWEN38_CONTEXT:-65536}"
port="${QWEN38_API_PORT:-8080}"
host="${QWEN38_API_HOST:-127.0.0.1}"
# The HTTP shim emits the first SSE event after generation completes. Keep the
# interactive default bounded; raise QWEN38_MAX_OUTPUT for long code patches.
max_output="${QWEN38_MAX_OUTPUT:-512}"
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
    OMP_PROC_BIND="${OMP_PROC_BIND:-close}" \
    OMP_PLACES="${OMP_PLACES:-cores}" \
    LLM_MOE_REGISTER_HOST="${LLM_MOE_REGISTER_HOST:-1}" \
    LLM_MOE_COPY_PIPELINE="${LLM_MOE_COPY_PIPELINE:-1}" \
    LLM_MOE_LFU_CACHE="${LLM_MOE_LFU_CACHE:-0}" \
    LLM_MOE_CPU_DECODE_MISSES="${LLM_MOE_CPU_DECODE_MISSES:-1}" \
    LLM_MOE_CPU_REFILLS_PER_LAYER="${LLM_MOE_CPU_REFILLS_PER_LAYER:-1}" \
    LLM_MOE_CPU_MIN_WEIGHT="${LLM_MOE_CPU_MIN_WEIGHT:-1}" \
    LLM_QWEN4_DELAYED_CACHE="${LLM_QWEN4_DELAYED_CACHE:-1}" \
    LLM_QWEN4_DELAYED_REFILL_INTERVAL="${LLM_QWEN4_DELAYED_REFILL_INTERVAL:-2}" \
    LLM_HC_GRAPHS="${LLM_HC_GRAPHS:-1}" \
    LLM_QWEN_PRE_GRAPHS="${LLM_QWEN_PRE_GRAPHS:-1}" \
    LLM_MOE_STREAM_SLOTS="${LLM_MOE_STREAM_SLOTS:-4}" \
    LLM_MOE_CPU_LIB="${cpu_lib}" \
    LLM_MOE_CPU_PREFILL_MAX_COUNT="${LLM_MOE_CPU_PREFILL_MAX_COUNT:-2}" \
    LLM_MOE_CPU_PREFILL_MAX_JOBS="${LLM_MOE_CPU_PREFILL_MAX_JOBS:-160}" \
    LLM_BMAX="${LLM_BMAX:-128}" \
    LLM_MOE_GROUPED_PREFILL="${LLM_MOE_GROUPED_PREFILL:-0}" \
    python3 "${runner_dir}/codex_server.py" "${model}" \
    --runner "${runner_dir}/test_hip_llm" \
    --context "${context}" \
    --max-output "${max_output}" \
    --port "${port}" \
    --host "${host}" \
    --moe-cache-mb "${cache_mb}" \
    --coding "$@"
