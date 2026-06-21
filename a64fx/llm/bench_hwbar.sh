#!/bin/bash
# A/B: flat SEV/WFE barrier vs A64FX hardware barrier (TF_HW_BARRIER=1).
# Same prompt/seed/threads; reports prefill + decode tok/s. Run each config
# REPS times and keep the best (max tok/s) to suppress OS noise.
set -u
cd "$(dirname "$0")"
M=${MODEL:-/local/models/Qwen3.5-9B-BF16.gguf}
KV=${KV:-f16}
GEN=${GEN:-256}
REPS=${REPS:-2}
OUT=/tmp/bench_hwbar.out
: > "$OUT"

# ~3500-token prompt (same construction as bench_9b_kv.sh).
HEAD=$(cat <<'TXT'
You are a senior systems engineer writing a long, detailed technical report.
Write a comprehensive multi-section essay of at least 500 words on the topic
"Designing memory-bandwidth-bound kernels for the Fujitsu A64FX processor".
Cover the HBM2 subsystem, SVE, the four-CMG NUMA layout, FP32/BF16/FP16
throughput, blocking guidelines, and fapp profiling. Background notes:

TXT
)
BODY=$(cat ../doc/FP16_GEMM_CEILING.md ../doc/llm-decode-optimization.md ../doc/fapp_pmu_profiling.md ../doc/a64fx_sector_cache_pragma.md 2>/dev/null)
TAIL=$'\n\nBegin the essay now with a level-1 heading, then six numbered sections.'
PROMPT="${HEAD}${BODY}${TAIL}"

run() {
    local label="$1"; shift
    local best_dec=0 best_pre=0
    for r in $(seq 1 "$REPS"); do
        local log
        log=$("$@" ./build/llm_runner "$M" --prompt "$PROMPT" \
              --max-gen "$GEN" --max-seq 20480 --llm-threads 48 \
              --kv-dtype "$KV" --mmap --seed 42 2>&1)
        local pre dec
        pre=$(echo "$log" | grep -oE 'prefill total:.*\(.* ([0-9.]+) tok/s\)' | grep -oE '[0-9.]+ tok/s' | grep -oE '[0-9.]+' | tail -1)
        dec=$(echo "$log" | grep -oE 'gen: .* \(([0-9.]+) tok/s\)' | grep -oE '[0-9.]+ tok/s' | grep -oE '[0-9.]+' | tail -1)
        echo "  [$label r$r] prefill=${pre:-?} decode=${dec:-?} tok/s" | tee -a "$OUT"
        awk "BEGIN{exit !(${dec:-0} > ${best_dec})}" && best_dec=${dec:-0}
        awk "BEGIN{exit !(${pre:-0} > ${best_pre})}" && best_pre=${pre:-0}
    done
    echo "== $label BEST: prefill=$best_pre decode=$best_dec tok/s ==" | tee -a "$OUT"
}

echo "model=$M KV=$KV GEN=$GEN REPS=$REPS  $(date)" | tee -a "$OUT"
run "flat   " env TF_HW_BARRIER=0
run "hwbar  " env TF_HW_BARRIER=1
echo "--- full log in $OUT ---"
