#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_dir="${root_dir}/tmp/long-context-gate"
depth="${QWEN38_LONG_DEPTH:-65536}"
context="${QWEN38_LONG_CONTEXT:-66560}"
decode="${QWEN38_LONG_DECODE:-256}"
repeats="${QWEN38_LONG_REPEATS:-3}"
prefill_floor="${QWEN38_LONG_PREFILL_FLOOR:-400}"
decode_floor="${QWEN38_LONG_DECODE_FLOOR:-32}"
mode="${QWEN38_LONG_MODE:-target}"

mkdir -p "${log_dir}"
[[ -x "${root_dir}/test_hip_llm" ]] || {
    echo "build ${root_dir}/test_hip_llm first" >&2
    exit 2
}

run_gate() {
    local label="$1"
    shift
    local log_file="${log_dir}/${label}.log"
    "${root_dir}/run_qwen38_gsq_rocm.sh" \
        --gpu-only-bench -n 1 -s "${context}" --ubatch 512 \
        --kv-cache q8q8 --qwen35-prefill-bf16 --qwen35-decode-graph \
        --qwen35-native-q8-prefill --qwen35-native-mmvq \
        --sampling-profile llama --temp 0 --seed 42 \
        --decode "${decode}" --bench-ignore-eos --bench-depth "${depth}" \
        --bench-repeat "${repeats}" "$@" >"${log_file}" 2>&1
    python3 "${root_dir}/check_qwen38_long_context.py" "${log_file}" \
        --label "${label}" --depth "${depth}" --repeats "${repeats}" \
        --prefill-floor "${prefill_floor}" --decode-floor "${decode_floor}"
}

case "${mode}" in
    target)
        run_gate target
        ;;
    dflash2|both)
        [[ -n "${QWEN38_DFLASH2_MODEL:-}" ]] || {
            echo "QWEN38_DFLASH2_MODEL is required for ${mode} mode" >&2
            exit 2
        }
        [[ "${mode}" == "dflash2" ]] || run_gate target
        run_gate dflash2 --qwen35-dflash2 "${QWEN38_DFLASH2_MODEL}" \
            --qwen35-dflash2-draft "${QWEN38_DFLASH2_DRAFT:-7}"
        ;;
    *)
        echo "QWEN38_LONG_MODE must be target, dflash2, or both" >&2
        exit 2
        ;;
esac
