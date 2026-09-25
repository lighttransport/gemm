#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_dir="${root_dir}/tmp"
log_file="${QWEN38_GSQ_BENCH_LOG:-${log_dir}/qwen38_gsq_iq2_decode.log}"
context="${QWEN38_GSQ_BENCH_CONTEXT:-53248}"
prefill="${QWEN38_GSQ_BENCH_PREFILL:-512}"
decode="${QWEN38_GSQ_BENCH_DECODE:-64}"
repeats="${QWEN38_GSQ_BENCH_REPEATS:-3}"
target="${QWEN38_GSQ_DECODE_TARGET:-30}"
run_timeout="${QWEN38_GSQ_BENCH_TIMEOUT:-1800}"

mkdir -p "${log_dir}"
[[ -x "${root_dir}/test_hip_llm" ]] || {
    echo "build ${root_dir}/test_hip_llm first" >&2
    exit 2
}

 : >"${log_file}"
for ((rep = 1; rep <= repeats; rep++)); do
    timeout --foreground "${run_timeout}s" \
        "${root_dir}/run_qwen38_gsq_rocm.sh" \
        -s "${context}" --bench -n "${prefill}" \
        --prefill-len "${prefill}" --decode "${decode}" \
        --bench-repeat 1 >>"${log_file}" 2>&1
done

grep -q 'Result: PASS' "${log_file}"
mapfile -t hashes < <(grep -oE 'sequence hash=[0-9a-f]+' "${log_file}" | sed 's/.*=//')
if (( ${#hashes[@]} != repeats )) ||
   [[ "$(printf '%s\n' "${hashes[@]}" | sort -u | wc -l)" -ne 1 ]]; then
    echo "GSQ decode gate FAIL: output was incomplete or nondeterministic" >&2
    exit 1
fi

decode_tps="$(awk '/^Decode:/ { for (i=1; i<=NF; ++i) if ($i == "->") {
    value=$(i+1)+0; if (!seen || value < minimum) minimum=value; seen=1
} } END { if (seen) printf "%.2f\n", minimum }' "${log_file}")"
[[ -n "${decode_tps}" ]] || {
    echo "GSQ decode gate FAIL: decode throughput missing from ${log_file}" >&2
    exit 1
}
awk -v got="${decode_tps}" -v want="${target}" 'BEGIN { exit !(got + 0 >= want + 0) }' || {
    echo "GSQ decode gate FAIL: ${decode_tps} tok/s < ${target} tok/s" >&2
    exit 1
}

grep -E 'clamping qwen35 context|VRAM|Prefill:|Decode:|End-to-end:' "${log_file}" || true
echo "GSQ decode gate PASS: ${decode_tps} tok/s >= ${target} tok/s; hash=${hashes[0]}"
