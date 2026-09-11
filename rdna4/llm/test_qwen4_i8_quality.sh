#!/bin/sh
# Compare the explicit I8 KV profile with the F16 exact-MTP control.
# This is a real-GPU regression (not part of the GPU-free profile tests).
set -eu

root_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
runner=${QWEN4_RUNNER:-$root_dir/rdna4/llm/test_hip_llm}
model=${QWEN38_MODEL:?set QWEN38_MODEL to the target GGUF}
sidecar=${QWEN38_MTP:?set QWEN38_MTP to the NextN sidecar GGUF}
context=${QWEN4_I8_CONTEXT:-4096}
decode=${QWEN4_I8_QUALITY_DECODE:-64}
cache=${QWEN4_I8_MOE_CACHE_MB:-6000}
prompt=${QWEN4_I8_PROMPT:-'Implement a bounded LRU cache in C with integer keys and values. Explain the eviction policy and provide insertion, lookup, and deletion functions.'}
mkdir -p "$root_dir/tmp"

run_one() {
    quant=$1
    out=$root_dir/tmp/qwen4_i8_quality_${quant}.log
    LLM_QWEN4_KV_QUANT=$quant timeout 600s "$runner" "$model" \
      --gpu-only-bench -s "$context" --moe-cache-mb "$cache" \
      --qwen4-mtp "$sidecar" --qwen4-mtp-draft 4 \
      --qwen4-mtp-cache-mb 128 --qwen4-mtp-verify scalar \
      -t "$prompt" --decode "$decode" >"$out" 2>&1
    grep -q 'Result: PASS' "$out" || { echo "${quant}: FAIL (see $out)" >&2; return 1; }
    hash=$(sed -n 's/.*sequence hash=\([0-9a-f]*\).*/\1/p' "$out" | tail -1)
    rate=$(sed -n 's/^Decode:.*-> \([0-9.]*\) tok\/s.*/\1/p' "$out" | tail -1)
    e2e=$(sed -n 's/^End-to-end:.*-> \([0-9.]*\) tok\/s.*/\1/p' "$out" | tail -1)
    printf '%s hash=%s decode_tok_s=%s e2e_tok_s=%s log=%s\n' "$quant" "$hash" "$rate" "$e2e" "$out"
}

i8_line=$(run_one i8)
fp8_line=
if [ "${QWEN4_I8_INCLUDE_FP8:-0}" != 0 ]; then
    fp8_line=$(run_one fp8)
fi
f16_line=$(run_one none)
printf '%s\n' "$i8_line"
[ -z "$fp8_line" ] || printf '%s\n' "$fp8_line"
printf '%s\n' "$f16_line"
if [ "$decode" -le 4 ]; then
    i8_hash=$(printf '%s\n' "$i8_line" | sed -n 's/.*hash=\([^ ]*\).*/\1/p')
    f16_hash=$(printf '%s\n' "$f16_line" | sed -n 's/.*hash=\([^ ]*\).*/\1/p')
    [ "$i8_hash" = "$f16_hash" ] || {
        echo "short exact parity failed: I8=$i8_hash F16=$f16_hash" >&2
        exit 1
    }
    if [ -n "$fp8_line" ]; then
        fp8_hash=$(printf '%s\n' "$fp8_line" | sed -n 's/.*hash=\([^ ]*\).*/\1/p')
        [ "$fp8_hash" = "$f16_hash" ] || {
            echo "short FP8 parity failed: FP8=$fp8_hash F16=$f16_hash" >&2
            exit 1
        }
    fi
else
    echo "long-horizon hashes are diagnostic; I8 may drift from F16 after quantization accumulation" >&2
fi
