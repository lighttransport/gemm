#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B GSQ-RCO IQ2_XS (qwen35) launcher for 16-GiB RDNA4 cards.
#
# The optimized profile stores the 16 full-attention K/V pairs as F16.  At
# 35,840 tokens this consumes 2.19 GiB, leaving the other half of the former
# FP32 allocation available for decode layouts and work buffers.
# Set QWEN38_GSQ_ALLOW_UNSAFE_CONTEXT=1 only when deliberately testing a
# larger card or a separately patched KV implementation.

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${QWEN38_MODEL:-/mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf}"
vram_profile="${QWEN38_VRAM_PROFILE:-16g}"
safe_context="${QWEN38_GSQ_SAFE_CONTEXT:-35840}"
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

if [[ "${QWEN38_DRY_RUN:-0}" != "0" ]]; then
    printf 'q38gsq profile: model=%s vram=%s requested_context=%s selected_context=%s safe_context=%s\n' \
        "${model}" "${vram_profile}" "${requested_context}" "${selected_context}" "${safe_context}"
    exit 0
fi

exec env QWEN38_MODEL="${model}" \
    LLM_MW_THREADS="${LLM_MW_THREADS:-64}" \
    LLM_DECODE_DP4A2="${LLM_DECODE_DP4A2:-1}" \
    "${runner_dir}/run_qwen38_flash_next_rocm.sh" \
    --kv-cache f16 --decode-kernels auto --decode-layout auto \
    --decode-layout-budget-mib "${QWEN38_DECODE_LAYOUT_BUDGET_MIB:-1792}" \
    "${args[@]}"
