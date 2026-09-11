#!/usr/bin/env bash
set -euo pipefail

# Diagnostic sub-32K target profile. This intentionally enables the
# parity-gated Qwen4 batched SSM/attention path and exact CPU cold misses; it
# is not a serving default until coding-prompt parity is complete. The caller
# owns prompt templating; test_hip_llm tokenizes the supplied bytes verbatim.
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="${QWEN38_RUNNER:-${root_dir}/test_hip_llm}"
model="${QWEN38_MODEL:-/mnt/nvme01/models/q38nf/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
max_seq="${QWEN38_SUB32_CONTEXT:-8192}"
prefill="${QWEN38_SUB32_PREFILL:-1024}"
decode="${QWEN38_SUB32_DECODE:-64}"
cache_mb="${QWEN38_SUB32_CACHE_MB:-8500}"
log_file="${QWEN38_SUB32_LOG:-${root_dir}/tmp/qwen38_sub32_target.log}"
cpu_lib="${LLM_MOE_CPU_LIB:-/mnt/nvme02/work/llama.cpp/build-codex-hetero-dev2/bin/libggml-cpu.so.0.22.0}"
prompt="${QWEN38_SUB32_PROMPT:-}"
prompt_file="${QWEN38_SUB32_PROMPT_FILE:-}"
bmax="${LLM_BMAX:-1024}"
multi_chunk_force="${LLM_QWEN4_BATCH_MULTI_CHUNK_FORCE:-}"
no_pad="${QWEN38_SUB32_NO_PAD:-0}"
stream_chunk="${LLM_BENCH_STREAM_CHUNK:-}"
publish_chunk="${LLM_BENCH_STREAM_PUBLISH_CHUNK:-}"
run_timeout="${QWEN38_SUB32_TIMEOUT:-900}"
if ! [[ "${run_timeout}" =~ ^[0-9]+$ ]] || (( run_timeout < 1 )); then
    run_timeout=900
fi
qwen_batch="${LLM_QWEN4_BATCH:-1}"
qwen_batch_stateful="${LLM_QWEN4_BATCH_STATEFUL:-1}"
qwen_batch_multi="${LLM_QWEN4_BATCH_MULTI_CHUNK:-1}"
batch_ssm="${QWEN38_SUB32_BATCH_SSM:-1}"
batch_attn_max="${QWEN38_SUB32_BATCH_ATTN_MAX_LAYER:-47}"
if [[ -z "${stream_chunk}" && "${prefill}" -gt 2048 ]]; then
    # A single 4K+ dispatch can exhaust gfx1201 scratch or hit the unstable
    # large-request stateful path.  Use the serving-shaped bounded stream by
    # default; callers can set LLM_BENCH_STREAM_CHUNK=0 to force an A/B run.
    if [[ "${LLM_QWEN4_PREFILL_COPY_PIPELINE:-0}" != "0" ]]; then
        stream_chunk=2048
    else
        stream_chunk=512
    fi
fi
if [[ -z "${publish_chunk}" &&
      "${LLM_QWEN4_PREFILL_COPY_PIPELINE:-0}" != "0" ]]; then
    publish_chunk=1
fi
if [[ -z "${LLM_QWEN4_BATCH+x}" && "${prefill}" -gt 2048 ]]; then
    # Until the stateful batched dispatcher is repaired for 4K+, keep the
    # automatic streamed profile on the scalar recurrent path. Callers can
    # explicitly opt back into batched A/B testing with LLM_QWEN4_BATCH=1.
    qwen_batch=0
    qwen_batch_stateful=0
    qwen_batch_multi=0
fi
if [[ -z "${multi_chunk_force}" ]]; then
    # The native single-batch path is faster and more stable at or below BMAX;
    # force the stateful multi-chunk dispatcher only for longer requests.
    if (( prefill > bmax )); then multi_chunk_force=1; else multi_chunk_force=0; fi
fi

if [[ "${QWEN38_DRY_RUN:-0}" != 0 ]]; then
    echo "sub32 dispatch: runner=${runner} prefill=${prefill} BMAX=${bmax} stream_chunk=${stream_chunk:-single} batch=${qwen_batch} stateful=${qwen_batch_stateful} multi=${qwen_batch_multi} cache_mb=${cache_mb}"
    exit 0
fi

if [[ -n "${prompt_file}" ]]; then
    [[ -r "${prompt_file}" ]] || { echo "prompt file not readable: ${prompt_file}" >&2; exit 2; }
fi

[[ -r "${model}" ]] || { echo "model not readable: ${model}" >&2; exit 2; }
[[ -x "${runner}" ]] || { echo "runner not executable: ${runner}" >&2; exit 2; }
[[ -r /dev/kfd && -w /dev/kfd ]] || { echo "AMD KFD access unavailable" >&2; exit 2; }
if command -v rocm-smi >/dev/null 2>&1; then
    # rocm-smi can transiently return nonzero while amdgpu is recovering from
    # a prior disposable kernel experiment.  This is advisory telemetry, not
    # a reason to abort before the benchmark creates its diagnostic log.
    vram_report="$(rocm-smi --showmeminfo vram 2>/dev/null || true)"
    vram_used="$(awk '/VRAM Total Used Memory/ { print $NF; exit }' <<<"${vram_report}")"
    if [[ "${vram_used}" =~ ^[0-9]+$ ]] && (( vram_used > 12000000000 )); then
        echo "AMD VRAM busy (${vram_used} bytes); refusing a crash-prone benchmark load" >&2
        exit 3
    fi
fi
mkdir -p "${root_dir}/tmp"
echo "sub32 dispatch: prefill=${prefill} BMAX=${bmax} stream_chunk=${stream_chunk:-single} batch=${qwen_batch} stateful=${qwen_batch_stateful} multi=${qwen_batch_multi} cache_mb=${cache_mb}" >&2
extra_args=()
if [[ -n "${prompt_file}" ]]; then
    extra_args+=( --prompt-file "${prompt_file}" )
elif [[ -n "${prompt}" ]]; then
    extra_args+=( -t "${prompt}" )
fi
if [[ "${QWEN38_SUB32_CODING:-0}" != 0 ]]; then extra_args+=( --coding ); fi
if [[ "${QWEN38_PREFILL_STAGING:-0}" != 0 ]]; then
    extra_args+=( --qwen4-prefill-staging )
fi
if [[ -n "${LLM_QWEN4_PREFILL_STAGE_MB:-}" ]]; then
    extra_args+=( --qwen4-prefill-stage-mb "${LLM_QWEN4_PREFILL_STAGE_MB}" )
fi

# Host routing is slower than GPU top-k but deterministic; GPU top-k caused
# run-to-run route/hash changes on gfx1201 in this parity experiment.
bench_args=("${runner}" "${model}" -s "${max_seq}" \
    --gpu-only-bench --bench -n "${prefill}" \
    --decode "${decode}" --moe-cache-mb "${cache_mb}" "${extra_args[@]}")
if [[ "${no_pad}" == 0 ]]; then
    bench_args+=(--prefill-len "${prefill}")
fi
# Keep the diagnostic pipeline enabled by default, but honor an explicit zero
# so direct-copy controls are real A/B measurements.
env \
    LLM_MOE_CPU_LIB="${cpu_lib}" \
    LLM_MOE_CPU_DECODE_MISSES="${LLM_MOE_CPU_DECODE_MISSES:-1}" \
    LLM_QWEN4_EXACT_CPU_MISSES="${LLM_QWEN4_EXACT_CPU_MISSES:-1}" \
    LLM_MOE_CPU_MIN_WEIGHT="${LLM_MOE_CPU_MIN_WEIGHT:-0.0}" \
    LLM_QWEN4_EXACT_CPU_MIN_WEIGHT="${LLM_QWEN4_EXACT_CPU_MIN_WEIGHT:-}" \
    LLM_QWEN4_KV_QUANT="${LLM_QWEN4_KV_QUANT:-none}" \
    LLM_MOE_CACHE_MB="${cache_mb}" \
    LLM_BMAX="${bmax}" \
    LLM_MOE_CHUNK="${LLM_MOE_CHUNK:-${bmax}}" \
    LLM_MOE_REGISTER_HOST=1 \
    LLM_MOE_COPY_PIPELINE="${LLM_MOE_COPY_PIPELINE:-1}" \
    LLM_MOE_STREAM_SLOTS="${LLM_MOE_STREAM_SLOTS:-2}" \
    LLM_QWEN4_PREFILL_GPU_TOPK="${LLM_QWEN4_PREFILL_GPU_TOPK:-0}" \
    LLM_QWEN4_EXACT_GPU_TOPK="${LLM_QWEN4_EXACT_GPU_TOPK:-0}" \
    LLM_QWEN4_BATCH="${qwen_batch}" \
    LLM_QWEN4_BATCH_STATEFUL="${qwen_batch_stateful}" \
    LLM_QWEN4_BATCH_MULTI_CHUNK="${qwen_batch_multi}" \
    LLM_QWEN4_BATCH_MULTI_CHUNK_FORCE="${multi_chunk_force}" \
    LLM_QWEN4_BATCH_SSM="${batch_ssm}" \
    LLM_QWEN4_BATCH_ATTN_MAX_LAYER="${batch_attn_max}" \
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
    LLM_BENCH_STREAM_CHUNK="${stream_chunk}" \
    LLM_BENCH_STREAM_PUBLISH_CHUNK="${publish_chunk}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-17}" \
    timeout --foreground "${run_timeout}s" "${bench_args[@]}" >"${log_file}" 2>&1

grep -E 'Prefill:|Decode:|End-to-end:|sequence hash=|Result:' "${log_file}"
grep -q 'Result: PASS' "${log_file}"
echo "sub-32K diagnostic PASS: ${log_file}"
