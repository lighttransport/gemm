#!/bin/bash
# Phase 2 decode A/B on the 11 no0 nodes (claude node excluded via VCOORD).
#   A_perop  : TP_DECODE_PERSIST=0 -> per-op block loop (~320 dispatches/tok), the proven baseline
#   B_persist: TP_DECODE_PERSIST=1 -> persistent worker (1 dispatch/tok, parallel SSM) w/ ported TP hooks
# Correctness: TP_DUMP_TOKENS dumps the rank0 generated token-id stream; A and B MUST be byte-identical
#   (greedy argmax, TP_IGNORE_EOS=1, fixed TP_MAXGEN -> deterministic, same length).
# Speed: decode tok/s from the rank0 run file + per_tok from tp_perf_rank00.txt.
# Model already staged to /local (SKIP_STAGE=1); topo already correct (SKIP_TOPO=1).
set -u
cd "$(dirname "$0")"
LOG=phase2_ab.log
: > "$LOG"
GEN=${GEN:-64}
PROMPT=${PROMPT:-"Explain what a transformer neural network is in two sentences."}

echo "=== rebuild tp_runner (CC=fcc OPENMP=1) ===" | tee -a "$LOG"
make tp_runner CC=fcc OPENMP=1 >>"$LOG" 2>&1 && echo "build_run_ok" | tee -a "$LOG" || { echo "BUILD FAILED" | tee -a "$LOG"; exit 1; }

run_one() {
    local tag="$1" persist="$2"
    echo "############ RUN $tag (TP_DECODE_PERSIST=$persist) ($(date +%H:%M:%S)) ############" | tee -a "$LOG"
    rm -f tp_run_*.txt tp_load_rank*.txt tp_perf_rank*.txt tp_stderr_rank*.txt tp_tokens_rank00.txt
    env VCOORD=vcoord_no0.txt SKIP_STAGE=1 SKIP_TOPO=1 QWEN27B_PREPARE=0 \
        NP=11 TP_PREFILL_ONLY=0 TP_IGNORE_EOS=1 TP_MAXGEN="$GEN" TP_DUMP_TOKENS=1 \
        TP_PROMPT="$PROMPT" TP_DECODE_PERSIST="$persist" \
        ./run_tp_27b.sh >>"$LOG" 2>&1
    echo "---- RUN $tag exit=$? ----" | tee -a "$LOG"
    shopt -s nullglob
    local r=( tp_run_*.txt )
    [ ${#r[@]} -gt 0 ] && cp "${r[0]}" "run_${tag}.txt"
    [ -f tp_perf_rank00.txt ]    && cp tp_perf_rank00.txt    "perf_${tag}.txt"
    [ -f tp_tokens_rank00.txt ]  && cp tp_tokens_rank00.txt  "tokens_${tag}.txt"
    shopt -u nullglob
    echo ">>> $tag decode line:" | tee -a "$LOG"
    [ -f "run_${tag}.txt" ]  && grep -h "decode(" "run_${tag}.txt"  | tee -a "$LOG"
    [ -f "perf_${tag}.txt" ] && grep -h "decode:" "perf_${tag}.txt" | tee -a "$LOG"
    echo | tee -a "$LOG"
}

run_one A_perop   0
run_one B_persist 1

echo "==================== SUMMARY ====================" | tee -a "$LOG"
for tag in A_perop B_persist; do
    echo "[$tag]" | tee -a "$LOG"
    [ -f "run_${tag}.txt" ]  && grep -h "decode(" "run_${tag}.txt"  | tee -a "$LOG"
    [ -f "perf_${tag}.txt" ] && grep -h "decode:" "perf_${tag}.txt" | tee -a "$LOG"
done

echo "==================== TOKEN PARITY ====================" | tee -a "$LOG"
if [ -f tokens_A_perop.txt ] && [ -f tokens_B_persist.txt ]; then
    na=$(wc -l < tokens_A_perop.txt); nb=$(wc -l < tokens_B_persist.txt)
    echo "tokens: A=$na B=$nb" | tee -a "$LOG"
    if diff -q tokens_A_perop.txt tokens_B_persist.txt >/dev/null; then
        echo "TOKEN PARITY: IDENTICAL ($na tokens byte-for-byte)" | tee -a "$LOG"
    else
        echo "TOKEN PARITY: !!! MISMATCH !!! (first diffs:)" | tee -a "$LOG"
        diff tokens_A_perop.txt tokens_B_persist.txt | head -20 | tee -a "$LOG"
    fi
else
    echo "TOKEN PARITY: missing token dump(s)" | tee -a "$LOG"
fi
echo "DONE $(date +%H:%M:%S)" | tee -a "$LOG"
