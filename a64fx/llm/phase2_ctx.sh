#!/bin/bash
# Phase 2 decode A/B at LONG context (synthetic prompt) on the 11 no0 nodes.
# Short-ctx (P=20) already showed A=39.66 B=41.90 tok/s, byte-identical. This sweeps
# realistic context to confirm decode stays >= the 20 tok/s target and to measure the
# persistent-worker delta where KV attention is non-trivial (SSM layers stay O(1)/tok).
#   A_perop  : TP_DECODE_PERSIST=0   B_persist: TP_DECODE_PERSIST=1
# TP_SYNTH_TOKENS=CTX builds a CTX-length KV via batched prefill, then decodes GEN tokens
# at positions CTX..CTX+GEN. Greedy argmax over synthetic IDs is deterministic -> token parity holds.
set -u
cd "$(dirname "$0")"
LOG=phase2_ctx.log
: > "$LOG"
GEN=${GEN:-64}
CTXS=${CTXS:-"1024 4096 8192"}

echo "=== rebuild tp_runner (CC=fcc OPENMP=1) ===" | tee -a "$LOG"
make tp_runner CC=fcc OPENMP=1 >>"$LOG" 2>&1 && echo "build_run_ok" | tee -a "$LOG" || { echo "BUILD FAILED" | tee -a "$LOG"; exit 1; }

run_one() {
    local tag="$1" persist="$2" ctx="$3"
    echo "######## RUN $tag ctx=$ctx persist=$persist ($(date +%H:%M:%S)) ########" | tee -a "$LOG"
    rm -f tp_run_*.txt tp_perf_rank*.txt tp_tokens_rank00.txt
    env VCOORD=vcoord_no0.txt SKIP_STAGE=1 SKIP_TOPO=1 QWEN27B_PREPARE=0 \
        NP=11 TP_PREFILL_ONLY=0 TP_IGNORE_EOS=1 TP_MAXGEN="$GEN" TP_DUMP_TOKENS=1 \
        TP_SYNTH_TOKENS="$ctx" TP_MAXSEQ=$((ctx + GEN + 32)) TP_DECODE_PERSIST="$persist" \
        ./run_tp_27b.sh >>"$LOG" 2>&1
    echo "---- $tag exit=$? ----" | tee -a "$LOG"
    shopt -s nullglob
    local r=( tp_run_*.txt )
    [ ${#r[@]} -gt 0 ] && cp "${r[0]}" "run_${tag}_c${ctx}.txt"
    [ -f tp_perf_rank00.txt ]   && cp tp_perf_rank00.txt   "perf_${tag}_c${ctx}.txt"
    [ -f tp_tokens_rank00.txt ] && cp tp_tokens_rank00.txt "tokens_${tag}_c${ctx}.txt"
    shopt -u nullglob
}

for ctx in $CTXS; do
    run_one A_perop   0 "$ctx"
    run_one B_persist 1 "$ctx"
    echo "==== ctx=$ctx ====" | tee -a "$LOG"
    for tag in A_perop B_persist; do
        echo "[$tag c$ctx]" | tee -a "$LOG"
        [ -f "run_${tag}_c${ctx}.txt" ]  && grep -h "decode(" "run_${tag}_c${ctx}.txt"  | tee -a "$LOG"
        [ -f "perf_${tag}_c${ctx}.txt" ] && grep -h "decode:" "perf_${tag}_c${ctx}.txt" | tee -a "$LOG"
    done
    if [ -f "tokens_A_perop_c${ctx}.txt" ] && [ -f "tokens_B_persist_c${ctx}.txt" ]; then
        if diff -q "tokens_A_perop_c${ctx}.txt" "tokens_B_persist_c${ctx}.txt" >/dev/null; then
            echo "PARITY c$ctx: IDENTICAL ($(wc -l < tokens_A_perop_c${ctx}.txt) tok)" | tee -a "$LOG"
        else
            echo "PARITY c$ctx: !!! MISMATCH !!!" | tee -a "$LOG"
            diff "tokens_A_perop_c${ctx}.txt" "tokens_B_persist_c${ctx}.txt" | head -10 | tee -a "$LOG"
        fi
    fi
    echo | tee -a "$LOG"
done
echo "DONE $(date +%H:%M:%S)" | tee -a "$LOG"
