#!/usr/bin/env bash
set -euo pipefail

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${QWEN38_MODEL:-/mnt/nvme01/models/q38nf/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
cache_mb="${QWEN38_MOE_CACHE_MB:-8192}"
cpu_lib="${LLM_MOE_CPU_LIB:-/mnt/nvme02/work/llama.cpp/build-codex-hetero-dev2/bin/libggml-cpu.so.0.22.0}"
if [[ ! -r "${cpu_lib}" ]]; then
    cpu_lib=""
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
    "${runner_dir}/test_hip_llm" "${model}" \
    --gpu-only-bench --coding --moe-cache-mb "${cache_mb}" "$@"
