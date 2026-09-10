#!/usr/bin/env bash
set -euo pipefail

# Diagnostic sub-32K target profile.  This intentionally enables the
# parity-gated Qwen4 batched SSM/attention path and exact CPU cold misses; it
# is not a serving default until coding-prompt parity is complete.
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${QWEN38_MODEL:-/mnt/nvme01/models/q38nf/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
max_seq="${QWEN38_SUB32_CONTEXT:-8192}"
prefill="${QWEN38_SUB32_PREFILL:-1024}"
decode="${QWEN38_SUB32_DECODE:-64}"
cache_mb="${QWEN38_SUB32_CACHE_MB:-8500}"
log_file="${QWEN38_SUB32_LOG:-${root_dir}/tmp/qwen38_sub32_target.log}"
cpu_lib="${LLM_MOE_CPU_LIB:-/mnt/nvme02/work/llama.cpp/build-codex-hetero-dev2/bin/libggml-cpu.so.0.22.0}"
prompt="${QWEN38_SUB32_PROMPT:-}"
bmax="${LLM_BMAX:-1024}"
multi_chunk_force="${LLM_QWEN4_BATCH_MULTI_CHUNK_FORCE:-}"
if [[ -z "${multi_chunk_force}" ]]; then
    # The native single-batch path is faster and more stable at or below BMAX;
    # force the stateful multi-chunk dispatcher only for longer requests.
    if (( prefill > bmax )); then multi_chunk_force=1; else multi_chunk_force=0; fi
fi

[[ -r "${model}" ]] || { echo "model not readable: ${model}" >&2; exit 2; }
[[ -x "${root_dir}/test_hip_llm" ]] || { echo "build test_hip_llm first" >&2; exit 2; }
[[ -r /dev/kfd && -w /dev/kfd ]] || { echo "AMD KFD access unavailable" >&2; exit 2; }
mkdir -p "${root_dir}/tmp"
extra_args=()
if [[ -n "${prompt}" ]]; then extra_args+=( -t "${prompt}" ); fi
if [[ "${QWEN38_PREFILL_STAGING:-0}" != 0 ]]; then
    extra_args+=( --qwen4-prefill-staging )
fi

# Host routing is slower than GPU top-k but deterministic; GPU top-k caused
# run-to-run route/hash changes on gfx1201 in this parity experiment.
env \
    LLM_MOE_CPU_LIB="${cpu_lib}" \
    LLM_MOE_CPU_DECODE_MISSES="${LLM_MOE_CPU_DECODE_MISSES:-1}" \
    LLM_QWEN4_EXACT_CPU_MISSES="${LLM_QWEN4_EXACT_CPU_MISSES:-1}" \
    LLM_MOE_CPU_MIN_WEIGHT="${LLM_MOE_CPU_MIN_WEIGHT:-0.0}" \
    LLM_QWEN4_EXACT_CPU_MIN_WEIGHT="${LLM_QWEN4_EXACT_CPU_MIN_WEIGHT:-}" \
    LLM_QWEN4_KV_QUANT=none \
    LLM_MOE_CACHE_MB="${cache_mb}" \
    LLM_BMAX="${bmax}" \
    LLM_MOE_CHUNK="${LLM_MOE_CHUNK:-${bmax}}" \
    LLM_MOE_REGISTER_HOST=1 \
    LLM_MOE_COPY_PIPELINE=1 \
    LLM_MOE_STREAM_SLOTS="${LLM_MOE_STREAM_SLOTS:-2}" \
    LLM_QWEN4_PREFILL_GPU_TOPK="${LLM_QWEN4_PREFILL_GPU_TOPK:-0}" \
    LLM_QWEN4_EXACT_GPU_TOPK="${LLM_QWEN4_EXACT_GPU_TOPK:-0}" \
    LLM_QWEN4_BATCH=1 \
    LLM_QWEN4_BATCH_STATEFUL=1 \
    LLM_QWEN4_BATCH_MULTI_CHUNK="${LLM_QWEN4_BATCH_MULTI_CHUNK:-1}" \
    LLM_QWEN4_BATCH_MULTI_CHUNK_FORCE="${multi_chunk_force}" \
    LLM_QWEN4_BATCH_SSM=1 \
    LLM_QWEN4_BATCH_ATTN_MAX_LAYER=47 \
    LLM_QWEN4_NATIVE_BATCH_QKV=1 \
    LLM_SSM_BATCH_Q6K=1 \
    LLM_SSM_BATCH_CONV=1 \
    LLM_SSM_BATCH_RECURRENCE=1 \
    LLM_SSM_BATCH_PARITY=1 \
    LLM_SSM_BATCH_WARP=0 \
    LLM_QWEN4_PREFILL_CACHE_BALANCE="${LLM_QWEN4_PREFILL_CACHE_BALANCE:-1}" \
    LLM_MOE_GROUPED_PREFILL="${LLM_MOE_GROUPED_PREFILL:-0}" \
    LLM_HC_GRAPHS=0 \
    LLM_QWEN_PRE_GRAPHS=0 \
    LLM_QWEN4_PRE_GRAPHS=0 \
    LLM_PLAN_PREWARM=0 \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-17}" \
    "${root_dir}/test_hip_llm" "${model}" -s "${max_seq}" \
    --gpu-only-bench --bench -n "${prefill}" --prefill-len "${prefill}" \
    --decode "${decode}" --moe-cache-mb "${cache_mb}" "${extra_args[@]}" >"${log_file}" 2>&1

grep -E 'Prefill:|Decode:|End-to-end:|sequence hash=|Result:' "${log_file}"
grep -q 'Result: PASS' "${log_file}"
echo "sub-32K diagnostic PASS: ${log_file}"
