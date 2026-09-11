#!/usr/bin/env bash
set -euo pipefail

# Reproducible quality-safe 256K smoke benchmark for the RX 9070 XT.
# Logs intentionally stay in the repository-local tmp directory.
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${QWEN38_MODEL:-/mnt/nvme01/models/q38nf/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
log_dir="${root_dir}/tmp"
log_file="${QWEN38_BENCH_LOG:-${log_dir}/qwen38_256k_smoke.log}"
prefill="${QWEN38_BENCH_PREFILL:-8}"
decode="${QWEN38_BENCH_DECODE:-8}"
bench_timeout="${QWEN38_BENCH_TIMEOUT:-600}"
prefill_topk="${LLM_QWEN4_PREFILL_GPU_TOPK:-1}"
kv_quant="${LLM_QWEN4_KV_QUANT:-i8}"
case "${kv_quant}" in
    i8|fp8|none|f16) ;;
    *) echo "LLM_QWEN4_KV_QUANT must be i8, fp8, f16, or none" >&2; exit 2 ;;
esac
if [[ "${LLM_QWEN4_BATCH:-0}" == "1" && -z "${LLM_QWEN4_PREFILL_GPU_TOPK+x}" ]]; then
    prefill_topk=0
fi

mkdir -p "${log_dir}"
[[ "${bench_timeout}" =~ ^[1-9][0-9]*$ ]] || {
    echo "QWEN38_BENCH_TIMEOUT must be a positive integer number of seconds" >&2
    exit 2
}
[[ -r "${model}" ]] || { echo "model not readable: ${model}" >&2; exit 2; }
[[ -x "${root_dir}/test_hip_llm" ]] || {
    echo "build ${root_dir}/test_hip_llm first" >&2
    exit 2
}
[[ -r /dev/kfd && -w /dev/kfd ]] || {
    echo "AMD KFD access unavailable: expose readable/writable /dev/kfd" >&2
    exit 2
}
render_ok=0
for node in /dev/dri/renderD*; do
    if [[ -r "${node}" && -w "${node}" ]]; then render_ok=1; break; fi
done
[[ "${render_ok}" -eq 1 ]] || {
    echo "AMD render-node access unavailable" >&2
    exit 2
}

# A second long-running ROCm client can consume VRAM or leave MES queues in a
# reset-prone state. Do not kill it, but make benchmark results attributable.
if command -v fuser >/dev/null 2>&1; then
    kfd_users="$(fuser /dev/kfd 2>/dev/null || true)"
    if [[ -n "${kfd_users//[[:space:]]/}" ]]; then
        echo "warning: /dev/kfd is already in use by: ${kfd_users}" >&2
        if [[ "${QWEN38_BENCH_REQUIRE_EXCLUSIVE_GPU:-0}" != "0" ]]; then
            echo "256K smoke refused: GPU contention (set QWEN38_BENCH_REQUIRE_EXCLUSIVE_GPU=0 to override)" >&2
            exit 2
        fi
    fi
fi

# 5.9 GiB is the largest reproducible scalar 16-GiB budget at max context
# with the transient graph/plan allocations disabled below.
timeout --foreground "${bench_timeout}s" env \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
    LLM_QWEN4_KV_QUANT="${kv_quant}" \
    LLM_MOE_CACHE_MB="${LLM_MOE_CACHE_MB:-5900}" \
    LLM_BMAX="${LLM_BMAX:-512}" \
    LLM_HC_GRAPHS="${LLM_HC_GRAPHS:-0}" \
    LLM_QWEN_PRE_GRAPHS="${LLM_QWEN_PRE_GRAPHS:-0}" \
    LLM_QWEN4_PRE_GRAPHS="${LLM_QWEN4_PRE_GRAPHS:-0}" \
    LLM_PLAN_PREWARM="${LLM_PLAN_PREWARM:-0}" \
    LLM_MOE_COPY_PIPELINE="${LLM_MOE_COPY_PIPELINE:-0}" \
    LLM_QWEN4_BATCH="${LLM_QWEN4_BATCH:-0}" \
    LLM_QWEN4_PREFILL_GPU_TOPK="${prefill_topk}" \
    LLM_QWEN4_BATCH_ATTN_MAX_LAYER="${LLM_QWEN4_BATCH_ATTN_MAX_LAYER:-2}" \
    LLM_QWEN4_BATCH_MULTI_CHUNK="${LLM_QWEN4_BATCH_MULTI_CHUNK:-0}" \
    LLM_QWEN4_BATCH_STATEFUL="${LLM_QWEN4_BATCH_STATEFUL:-0}" \
    "${root_dir}/test_hip_llm" "${model}" \
        -s 262144 --gpu-only-bench --bench -n "${prefill}" \
        --prefill-len "${prefill}" --decode "${decode}" \
        --moe-cache-mb "${LLM_MOE_CACHE_MB:-5900}" >"${log_file}" 2>&1

grep -q 'weights loaded successfully' "${log_file}"
if grep -Eq 'Failed to (load weights to GPU|init HIP runner)|ROCm device unavailable|HIP error at' "${log_file}"; then
    echo "256K smoke failed: GPU initialization/load error is present in ${log_file}" >&2
    exit 1
fi
grep -q 'Result: PASS' "${log_file}"
grep -E 'Qwen4 KV cache:|Prefill:|Decode:|End-to-end:' "${log_file}"
if [[ -n "${QWEN38_EXPECT_HASH:-}" ]]; then
    grep -q "sequence hash=${QWEN38_EXPECT_HASH}" "${log_file}" || {
        echo "256K quality hash mismatch (expected ${QWEN38_EXPECT_HASH})" >&2
        exit 1
    }
fi
echo "256K smoke PASS: kv=${kv_quant} log=${log_file}"
