#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rows="${IQ2_BW_ROWS:-17408}"
cols="${IQ2_BW_COLS:-5120}"
iters="${IQ2_BW_ITERS:-300}"
repeats="${IQ2_BW_REPEATS:-11}"
peak_gbs="${IQ2_BW_PEAK_GBS:-640}"
target_pct="${IQ2_BW_TARGET_PCT:-90}"

[[ -x "${root_dir}/test_hip_llm" ]] || {
    echo "build ${root_dir}/test_hip_llm first" >&2
    exit 2
}
(( cols % 256 == 0 )) || {
    echo "IQ2_BW_COLS must be divisible by 256" >&2
    exit 2
}

output="$(env LLM_DECODE_DP4A=1 LLM_MW_THREADS=64 \
    LLM_BENCH_QMV_PREQUANT=1 \
    "${root_dir}/test_hip_llm" --bench-quant-matvec \
    IQ2_S "${rows}" "${cols}" "${iters}" "${repeats}" 2>&1)"
printf '%s\n' "${output}"
ms="$(awk '/^quant_matvec IQ2_S / {for (i=1;i<=NF;i++) if ($i=="median") {print $(i+1); exit}}' <<<"${output}")"
[[ -n "${ms}" ]] || { echo "IQ2 bandwidth gate FAIL: timing missing" >&2; exit 1; }

# IQ2_S: 82 compressed bytes per 256 weights, plus one activation read and
# one output write. This is the compulsory-traffic convention used by the gate.
read -r bandwidth pct <<<"$(awk -v r="${rows}" -v c="${cols}" -v ms="${ms}" -v peak="${peak_gbs}" \
    'BEGIN {bytes=r*(c/256)*82+c*4+r*4; bw=bytes/(ms/1000)/1e9; printf "%.2f %.2f",bw,100*bw/peak}')"
printf 'IQ2_S compulsory bandwidth: %.2f GB/s (%.2f%% of %.0f GB/s)\n' \
    "${bandwidth}" "${pct}" "${peak_gbs}"
awk -v got="${pct}" -v want="${target_pct}" 'BEGIN {exit !(got >= want)}' || {
    echo "IQ2 bandwidth gate FAIL: ${pct}% < ${target_pct}%" >&2
    exit 1
}
echo "IQ2 bandwidth gate PASS: ${pct}% >= ${target_pct}%"
