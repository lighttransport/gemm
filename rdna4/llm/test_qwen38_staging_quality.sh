#!/usr/bin/env bash
set -uo pipefail
# Matched scalar F16 vs staged greedy-hash checks; requires the real model.
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${root_dir}/../.." && pwd)"
cd "${repo_dir}"
export TMPDIR="$PWD/rdna4/llm/tmp"
export QWEN38_TARGET_CONTEXT=8192 QWEN38_TARGET_REPEATS=2 QWEN38_TARGET_REQUIRE_EXCLUSIVE_GPU=1
export LLM_BENCH_WARMUP=0 LLM_BMAX=4096 QWEN38_MOE_CACHE_MB=4000
export LLM_MOE_REGISTER_HOST=1 QWEN38_TARGET_CPU_EXPERTS=0
export LLM_QWEN4_APPROX_DECODE=0 LLM_QWEN4_DEVICE_HITS_ONLY=0 LLM_QWEN4_STAGE_PROMOTE=0
mkdir -p rdna4/llm/tmp/staging_quality
head -c 9000 common/gguf_loader.h > rdna4/llm/tmp/staging_quality/target.txt
printf '%s\n' 'Write a C function int clamp(int x, int lo, int hi) that clamps x to the inclusive range. Return only compilable C code.' > rdna4/llm/tmp/staging_quality/coding.txt
printf '%s\n' 'Compute 37 * 19 + 248. Explain the calculation step by step and give the final integer.' > rdna4/llm/tmp/staging_quality/arithmetic.txt
printf '%s\n' 'Explain why leaves change color in autumn in a short paragraph for a curious twelve-year-old.' > rdna4/llm/tmp/staging_quality/prose.txt
printf '%s\n' '日本語で、雨が降った後に虹が見える理由を、小学生にも分かるように説明してください。' > rdna4/llm/tmp/staging_quality/japanese.txt
status=0
for name in target coding arithmetic prose japanese; do
    if [[ "$name" == target ]]; then
        export QWEN38_TARGET_PROMPT_FILE=rdna4/llm/tmp/staging_quality/target.txt
        export QWEN38_TARGET_PREFILL=4096 QWEN38_TARGET_DECODE=64
    else
        export QWEN38_TARGET_PROMPT_FILE="rdna4/llm/tmp/staging_quality/${name}.txt"
        export QWEN38_TARGET_PREFILL=128 QWEN38_TARGET_DECODE=16
    fi
    unset QWEN38_TARGET_EXPECTED_FIRST_TOKEN QWEN38_TARGET_EXPECTED_HASH
    export QWEN38_TARGET_PROFILE=scalar-exact LLM_MOE_COPY_PIPELINE=0 LLM_QWEN4_BATCH=0
    export QWEN38_TARGET_LOG="rdna4/llm/tmp/staging_quality/${name}_scalar.log"
    echo "=== QUALITY ${name}: scalar reference ==="
    if ! ./rdna4/llm/bench_qwen38_target.sh; then
        echo "=== QUALITY ${name}: reference failed ==="; status=1; continue
    fi
    QWEN38_TARGET_EXPECTED_FIRST_TOKEN=$(sed -n 's/.*First decoded token id=\([0-9]*\).*/\1/p' "$QWEN38_TARGET_LOG" | head -1)
    QWEN38_TARGET_EXPECTED_HASH=$(sed -n 's/.*sequence hash=\([0-9a-f]*\).*/\1/p' "$QWEN38_TARGET_LOG" | head -1)
    export QWEN38_TARGET_EXPECTED_FIRST_TOKEN QWEN38_TARGET_EXPECTED_HASH
    export QWEN38_TARGET_PROFILE=batch4k-stage LLM_MOE_COPY_PIPELINE=1 LLM_QWEN4_BATCH=1
    export QWEN38_TARGET_LOG="rdna4/llm/tmp/staging_quality/${name}_staged.log"
    echo "=== QUALITY ${name}: staged vs reference ==="
    ./rdna4/llm/bench_qwen38_target.sh
    rc=$?
    echo "=== QUALITY ${name}: rc=${rc} ==="
    (( rc == 0 )) || status=1
done
exit "$status"
