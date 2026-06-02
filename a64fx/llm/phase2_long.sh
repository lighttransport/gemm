#!/bin/bash
# Phase 2 long-context decode curve (persistent worker, default ON) at ctx 4096/8192.
# ctx 1024 already done (A=26.64 B=27.27, parity identical). The batched-prefill that
# BUILDS the KV context issues thousands of all-reduces at large M and can hit an
# intermittent uTofu dropped-Put deadlock (tp_ar bcast timeout want=N+1 got=N), so each
# context is retried up to 3x and uses TP_AR_BATCH=512 (fewer chunks -> fewer reduces).
# Goal: the real decode tok/s vs context, to see where it crosses the 20 tok/s target.
set -u
cd "$(dirname "$0")"
LOG=phase2_long.log
: > "$LOG"
GEN=${GEN:-64}
CTXS=${CTXS:-"4096 8192"}
RETRIES=${RETRIES:-3}

echo "=== rebuild tp_runner (CC=fcc OPENMP=1) ===" | tee -a "$LOG"
make tp_runner CC=fcc OPENMP=1 >>"$LOG" 2>&1 && echo "build_run_ok" | tee -a "$LOG" || { echo "BUILD FAILED" | tee -a "$LOG"; exit 1; }

run_persist() {
    local ctx="$1"
    local ok=0
    for try in $(seq 1 "$RETRIES"); do
        echo "######## ctx=$ctx persist=1 try=$try ($(date +%H:%M:%S)) ########" | tee -a "$LOG"
        rm -f tp_run_*.txt tp_perf_rank*.txt tp_tokens_rank00.txt tp_stderr_rank*.txt
        env VCOORD=vcoord_no0.txt SKIP_STAGE=1 SKIP_TOPO=1 QWEN27B_PREPARE=0 \
            NP=11 TP_PREFILL_ONLY=0 TP_IGNORE_EOS=1 TP_MAXGEN="$GEN" TP_AR_BATCH=512 \
            TP_SYNTH_TOKENS="$ctx" TP_MAXSEQ=$((ctx + GEN + 32)) TP_DECODE_PERSIST=1 \
            ./run_tp_27b.sh >>"$LOG" 2>&1
        echo "---- ctx=$ctx try=$try exit=$? ----" | tee -a "$LOG"
        shopt -s nullglob
        local r=( tp_run_*.txt )
        [ ${#r[@]} -gt 0 ] && cp "${r[0]}" "run_persist_c${ctx}.txt"
        [ -f tp_perf_rank00.txt ] && cp tp_perf_rank00.txt "perf_persist_c${ctx}.txt"
        shopt -u nullglob
        if [ -f "run_persist_c${ctx}.txt" ] && grep -q "decode(" "run_persist_c${ctx}.txt"; then
            echo "[ctx=$ctx OK on try $try]" | tee -a "$LOG"
            grep -h "decode(" "run_persist_c${ctx}.txt" | tee -a "$LOG"
            grep -h "decode:" "perf_persist_c${ctx}.txt" 2>/dev/null | head -1 | tee -a "$LOG"
            ok=1; break
        else
            echo "[ctx=$ctx try $try FAILED — checking flake]" | tee -a "$LOG"
            grep -h "tp_ar:.*timeout\|poll_tcq rc\|SIGSEGV\|died" tp_stderr_rank0*.txt 2>/dev/null | head -3 | tee -a "$LOG"
        fi
    done
    [ "$ok" = 0 ] && echo "[ctx=$ctx GAVE UP after $RETRIES tries]" | tee -a "$LOG"
}

for ctx in $CTXS; do run_persist "$ctx"; echo | tee -a "$LOG"; done

echo "==================== CURVE (persistent, default ON) ====================" | tee -a "$LOG"
echo "ctx~50  : 41.90 tok/s   (prior phase2_ab)" | tee -a "$LOG"
echo "ctx1024 : 27.27 tok/s   (prior phase2_ctx)" | tee -a "$LOG"
for ctx in $CTXS; do
    echo -n "ctx$ctx : " | tee -a "$LOG"
    [ -f "run_persist_c${ctx}.txt" ] && grep -h "decode(" "run_persist_c${ctx}.txt" | tail -1 | tee -a "$LOG" || echo "(no result)" | tee -a "$LOG"
done
echo "DONE $(date +%H:%M:%S)" | tee -a "$LOG"
