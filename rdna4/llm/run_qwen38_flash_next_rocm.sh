#!/usr/bin/env bash
set -euo pipefail

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${QWEN38_MODEL:-/mnt/nvme01/models/q38nf/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
cache_mb="${QWEN38_MOE_CACHE_MB:-8192}"

exec env \
    LLM_MOE_REGISTER_HOST="${LLM_MOE_REGISTER_HOST:-1}" \
    LLM_MOE_COPY_PIPELINE="${LLM_MOE_COPY_PIPELINE:-1}" \
    LLM_MOE_LFU_CACHE="${LLM_MOE_LFU_CACHE:-1}" \
    "${runner_dir}/test_hip_llm" "${model}" \
    --gpu-only-bench --coding --moe-cache-mb "${cache_mb}" "$@"
