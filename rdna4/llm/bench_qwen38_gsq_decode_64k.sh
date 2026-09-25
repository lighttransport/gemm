#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_dir="${root_dir}/tmp"
log_file="${QWEN38_GSQ_64K_LOG:-${log_dir}/qwen38_gsq_iq2_decode_64k_random.log}"
depth="${QWEN38_GSQ_64K_DEPTH:-65536}"
context="${QWEN38_GSQ_64K_CONTEXT:-66560}"
decode="${QWEN38_GSQ_64K_DECODE:-512}"
repeats="${QWEN38_GSQ_64K_REPEATS:-3}"
floor_tps="${QWEN38_GSQ_64K_FLOOR_TPS:-32.0}"
prefill_floor_tps="${QWEN38_GSQ_64K_PREFILL_FLOOR_TPS:-400.0}"

mkdir -p "${log_dir}"
[[ -x "${root_dir}/test_hip_llm" ]] || {
    echo "build ${root_dir}/test_hip_llm first" >&2
    exit 2
}

"${root_dir}/run_qwen38_gsq_rocm.sh" \
    --gpu-only-bench -n 1 -s "${context}" --ubatch 512 \
    --kv-cache q8q8 --qwen35-prefill-bf16 --qwen35-decode-graph \
    --qwen35-native-q8-prefill --qwen35-native-mmvq \
    --sampling-profile llama --temp 0 --seed 42 \
    --decode "${decode}" --bench-ignore-eos --bench-depth "${depth}" \
    --bench-repeat "${repeats}" >"${log_file}" 2>&1

grep -q 'Result: PASS' "${log_file}"
grep -q "Depth prefill: ${depth} random tokens .*seed=1" "${log_file}" || {
    echo "64K decode gate FAIL: random-token depth preparation missing" >&2
    exit 1
}
prefill_tps="$(awk '/^Depth prefill:/ { for (i=1; i<=NF; ++i) if ($i == "->") {
    print $(i+1)+0; exit
} }' "${log_file}")"
[[ -n "${prefill_tps}" ]] || {
    echo "64K prefill gate FAIL: throughput missing from ${log_file}" >&2
    exit 1
}
awk -v got="${prefill_tps}" -v want="${prefill_floor_tps}" \
    'BEGIN { exit !(got + 0 >= want + 0) }' || {
    echo "64K prefill gate FAIL: ${prefill_tps} tok/s < ${prefill_floor_tps} tok/s" >&2
    exit 1
}
mapfile -t hashes < <(grep -oE 'sequence hash=[0-9a-f]+' "${log_file}" | sed 's/.*=//')
if (( ${#hashes[@]} != repeats )) ||
   [[ "$(printf '%s\n' "${hashes[@]}" | sort -u | wc -l)" -ne 1 ]]; then
    echo "64K decode gate FAIL: output was incomplete or nondeterministic" >&2
    exit 1
fi

minimum="$(awk '/^Decode:/ { for (i=1; i<=NF; ++i) if ($i == "->") {
    value=$(i+1)+0; if (!seen || value < minimum) minimum=value; seen=1
} } END { if (seen) printf "%.2f\n", minimum }' "${log_file}")"
[[ -n "${minimum}" ]] || {
    echo "64K decode gate FAIL: throughput missing from ${log_file}" >&2
    exit 1
}
awk -v got="${minimum}" -v want="${floor_tps}" 'BEGIN { exit !(got + 0 >= want + 0) }' || {
    echo "64K decode gate FAIL: ${minimum} tok/s < ${floor_tps} tok/s" >&2
    exit 1
}

grep -E '^Depth prefill: [0-9]+ random tokens|^=== Bench:|^Decode:|^VRAM:' "${log_file}"
echo "64K gate PASS: prefill ${prefill_tps} tok/s >= ${prefill_floor_tps} tok/s; decode ${minimum} tok/s >= ${floor_tps} tok/s; hash=${hashes[0]}"
