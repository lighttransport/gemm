#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B GSQ-RCO IQ2_XS (qwen35) launcher for 16-GiB RDNA4 cards.
#
# The optimized profile stores the 16 full-attention K/V pairs as Q8 K/Q4 V.
# On the 16-GiB RX 9070 XT, 53,248 tokens was validated with 538 MiB free
# after model, KV, and decode work-buffer allocation (measured 504 MiB in the
# latest long-context run; leave margin for allocator variation).
# Set QWEN38_GSQ_ALLOW_UNSAFE_CONTEXT=1 only when deliberately testing a
# larger card or a separately patched KV implementation.

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${QWEN38_MODEL:-/mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf}"
vram_profile="${QWEN38_VRAM_PROFILE:-16g}"
safe_context="${QWEN38_GSQ_SAFE_CONTEXT:-53248}"
allow_unsafe="${QWEN38_GSQ_ALLOW_UNSAFE_CONTEXT:-0}"

[[ -r "${model}" ]] || {
    echo "Qwen3.8 GSQ model not found: ${model}" >&2
    echo "Set QWEN38_MODEL to the GGUF path." >&2
    exit 2
}
[[ "${safe_context}" =~ ^[1-9][0-9]*$ ]] || {
    echo "QWEN38_GSQ_SAFE_CONTEXT must be a positive integer" >&2
    exit 2
}
case "${vram_profile}" in
    16g) ;;
    24g|32g) safe_context="${QWEN38_GSQ_SAFE_CONTEXT:-262144}" ;;
    *) echo "unknown QWEN38_VRAM_PROFILE=${vram_profile} (use 16g, 24g, or 32g)" >&2; exit 2 ;;
esac

args=()
requested_context=0
for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    if [[ "${arg}" == "-s" || "${arg}" == "--max-seq-len" ]]; then
        j=$((i + 1))
        if (( j <= $# )); then
            requested_context="${!j}"
            args+=("${arg}" "${!j}")
            ((i++))
            continue
        fi
    fi
    args+=("${arg}")
done
if (( requested_context == 0 )); then
    requested_context="${safe_context}"
    args+=("-s" "${requested_context}")
fi
[[ "${requested_context}" =~ ^[1-9][0-9]*$ ]] || {
    echo "context after -s/--max-seq-len must be a positive integer" >&2
    exit 2
}
selected_context="${requested_context}"
if [[ "${vram_profile}" == "16g" && "${allow_unsafe}" == "0" ]] &&
   (( requested_context > safe_context )); then
    echo "Qwen3.8 GSQ: clamping context ${requested_context} to ${safe_context} for the 16-GiB profile" >&2
    selected_context="${safe_context}"
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[i]}" == "-s" || "${args[i]}" == "--max-seq-len" ]]; then
            args[i+1]="${safe_context}"
            break
        fi
    done
fi

perf_profile="${QWEN38_GSQ_PERF:-0}"
fast_prefill="${QWEN38_GSQ_FAST_PREFILL:-0}"
fast_all_iq2="${QWEN38_GSQ_FAST_ALL_IQ2:-0}"
fast_iq2_max_layer="${QWEN38_GSQ_FAST_IQ2_MAX_LAYER:-${LLM_QWEN35_IQ2_BF16_MAX_LAYER:-}}"
fast_prefill_pins=""
if [[ "${fast_prefill}" != "0" ]]; then
    # gfx1201's first hipBLASLt heuristic is not consistently the fastest for
    # the large BF16 shapes used by the IQ2_XS fallback.  These pins were
    # measured with M=2048 on the RX 9070 XT; users can override them.
    fast_prefill_pins="2048x17408x5120:110451,2048x5120x17408:110451,2048x5120x6144:110451,2048x10240x5120:110451,2048x1024x5120:110451"
fi
perf_bmax=512
if [[ "${perf_profile}" != "0" && "${selected_context}" -lt 40000 ]]; then
    # At 32K the Qwen3.8 Q8/Q4 KV allocation leaves enough headroom for the
    # validated 2048-row hybrid scratch set.  4096 rows faults gfx1201, while
    # at the 53K 16-GiB ceiling even 2048 leaves too little margin, so retain
    # the validated 512-row tile there.
    perf_bmax=2048
fi
selected_bmax="${LLM_BMAX:-${perf_bmax}}"
if [[ "${vram_profile}" == "16g" && "${allow_unsafe}" == "0" ]]; then
    bmax_cap=512
    if (( selected_context < 40000 )); then bmax_cap=2048; fi
    if [[ "${selected_bmax}" =~ ^[1-9][0-9]*$ ]] &&
       (( selected_bmax > bmax_cap )); then
        echo "Qwen3.8 GSQ: clamping batch size ${selected_bmax} to ${bmax_cap} for the 16-GiB profile" >&2
        selected_bmax="${bmax_cap}"
    fi
fi
if [[ "${QWEN38_DRY_RUN:-0}" != "0" ]]; then
    printf 'q38gsq profile: model=%s vram=%s requested_context=%s selected_context=%s safe_context=%s bmax=%s\n' \
        "${model}" "${vram_profile}" "${requested_context}" "${selected_context}" "${safe_context}" "${selected_bmax}"
    exit 0
fi

# IQ2_XXS/S BF16 dequant is a separate approximate mode: it clears the
# 300 tok/s long-context target, but changes later-token numerics. Keep it
# explicit so the normal fast profile retains its existing quality gate.
exec env QWEN38_MODEL="${model}" \
    LLM_BMAX="${selected_bmax}" \
    LLM_BENCH_STREAM_CHUNK="${LLM_BENCH_STREAM_CHUNK:-${selected_bmax}}" \
    LLM_MW_THREADS="${LLM_MW_THREADS:-64}" \
    LLM_DECODE_DP4A2="${LLM_DECODE_DP4A2:-1}" \
    LLM_QWEN35_NATIVE_IQ2_BATCH="${LLM_QWEN35_NATIVE_IQ2_BATCH:-1}" \
    LLM_QWEN35_NATIVE_IQ2_DP4A_BATCH="${LLM_QWEN35_NATIVE_IQ2_DP4A_BATCH:-1}" \
    LLM_QWEN35_NATIVE_IQ4_DP4A_BATCH="${LLM_QWEN35_NATIVE_IQ4_DP4A_BATCH:-1}" \
    LLM_QWEN35_NATIVE_IQ1S_DP4A_BATCH="${LLM_QWEN35_NATIVE_IQ1S_DP4A_BATCH:-1}" \
    LLM_QWEN35_NATIVE_IQ1M_DP4A_BATCH="${LLM_QWEN35_NATIVE_IQ1M_DP4A_BATCH:-1}" \
    LLM_QWEN35_NATIVE_IQ3_DP4A_BATCH="${LLM_QWEN35_NATIVE_IQ3_DP4A_BATCH:-1}" \
    LLM_QWEN35_NATIVE_IQ3XXS_BATCH="${LLM_QWEN35_NATIVE_IQ3XXS_BATCH:-1}" \
    LLM_QWEN35_IQ2_MMQ_SINGLE="${LLM_QWEN35_IQ2_MMQ_SINGLE:-${perf_profile}}" \
    LLM_QWEN35_IQ2_BF16_GEMM="${LLM_QWEN35_IQ2_BF16_GEMM:-${fast_prefill}}" \
    LLM_QWEN35_IQ2XXS_BF16_GEMM="${LLM_QWEN35_IQ2XXS_BF16_GEMM:-${fast_all_iq2}}" \
    LLM_QWEN35_IQ2S_BF16_GEMM="${LLM_QWEN35_IQ2S_BF16_GEMM:-${fast_all_iq2}}" \
    LLM_QWEN35_IQ2_BF16_MAX_LAYER="${fast_iq2_max_layer}" \
    LLM_QWEN35_IQ2XXS_MMQ_FUSED="${LLM_QWEN35_IQ2XXS_MMQ_FUSED:-0}" \
    LLM_QWEN35_IQ2XS_MMQ_FUSED="${LLM_QWEN35_IQ2XS_MMQ_FUSED:-0}" \
    LLM_QWEN35_IQ2S_MMQ_FUSED="${LLM_QWEN35_IQ2S_MMQ_FUSED:-0}" \
    LLM_QWEN35_IQ3XXS_Q8_WMMA="${LLM_QWEN35_IQ3XXS_Q8_WMMA:-${perf_profile}}" \
    LLM_QWEN35_IQ3_Q8_WMMA="${LLM_QWEN35_IQ3_Q8_WMMA:-${perf_profile}}" \
    LLM_QWEN35_IQ1S_Q8_WMMA="${LLM_QWEN35_IQ1S_Q8_WMMA:-${perf_profile}}" \
    LLM_QWEN35_IQ1M_Q8_WMMA="${LLM_QWEN35_IQ1M_Q8_WMMA:-${perf_profile}}" \
    LLM_ATTN_DECODE_Q8Q4_WARPRED="${LLM_ATTN_DECODE_Q8Q4_WARPRED:-${perf_profile}}" \
    LLM_ATTN_DECODE_Q8Q4_SPLIT="${LLM_ATTN_DECODE_Q8Q4_SPLIT:-${perf_profile}}" \
    LLM_ATTN_DECODE_Q8Q4_SPLIT_CHUNK="${LLM_ATTN_DECODE_Q8Q4_SPLIT_CHUNK:-1024}" \
    LLM_ATTN_DECODE_Q8Q4_DP4A="${LLM_ATTN_DECODE_Q8Q4_DP4A:-${perf_profile}}" \
    LLM_ATTN_DECODE_Q8Q4_THREADS="${LLM_ATTN_DECODE_Q8Q4_THREADS:-512}" \
    LLM_ATTN_DECODE_Q8Q4_VECV="${LLM_ATTN_DECODE_Q8Q4_VECV:-${perf_profile}}" \
    LLM_QWEN35_BF16_FFN_ONLY="${LLM_QWEN35_BF16_FFN_ONLY:-${perf_profile}}" \
    LLM_QWEN35_BATCH_SSM="${LLM_QWEN35_BATCH_SSM:-1}" \
    LLM_QWEN35_BATCH_SSM_FAST="${LLM_QWEN35_BATCH_SSM_FAST:-1}" \
    LLM_SSM_BATCH_CONV="${LLM_SSM_BATCH_CONV:-1}" \
    LLM_SSM_BATCH_PREP_FUSE="${LLM_SSM_BATCH_PREP_FUSE:-0}" \
    LLM_SSM_BATCH_RECURRENCE="${LLM_SSM_BATCH_RECURRENCE:-1}" \
    MM_BLASLT_ALGO_PINS="${MM_BLASLT_ALGO_PINS:-${fast_prefill_pins}}" \
    "${runner_dir}/run_qwen38_flash_next_rocm.sh" \
    --kv-cache q8q4 --decode-kernels auto --decode-layout auto \
    --decode-layout-budget-mib "${QWEN38_DECODE_LAYOUT_BUDGET_MIB:-1792}" \
    --qwen35-batched-prefill --ubatch "${QWEN38_GSQ_UBATCH:-512}" \
    "${args[@]}"
