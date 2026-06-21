#!/bin/bash
# Workstream A correctness guard: prove the robust AR path (drain+civac, default ON)
# yields a BYTE-IDENTICAL token stream vs the original passive spin (TP_AR_ROBUST=0).
# Short synthetic ctx so it is fast; the reduce math + civac logic are M-independent,
# so short ctx fully exercises the robust read path's correctness.
set -u
cd "$(dirname "$0")"
LOG=wsA_parity.log
: > "$LOG"
CTX=${CTX:-64}
GEN=${GEN:-48}
log() { echo "$@" | tee -a "$LOG"; }

run() {
    local robust="$1" out="$2"
    rm -f tp_tokens_rank00.txt tp_run_*.txt tp_stderr_rank*.txt
    env VCOORD=vcoord_no0.txt SKIP_STAGE=1 SKIP_TOPO=1 QWEN27B_PREPARE=0 \
        NP=11 TP_PREFILL_ONLY=0 TP_IGNORE_EOS=1 TP_MAXGEN="$GEN" \
        TP_AR_BATCH=128 TP_AR_ROBUST="$robust" TP_DUMP_TOKENS=1 \
        TP_SYNTH_TOKENS="$CTX" TP_MAXSEQ=$((CTX + GEN + 32)) TP_DECODE_PERSIST=1 \
        ./run_tp_27b.sh >>"$LOG" 2>&1
    cp tp_tokens_rank00.txt "$out" 2>/dev/null || echo "NO_TOKENS" > "$out"
}

log "=== WSA token-parity A/B $(date +%H:%M:%S) ctx=$CTX gen=$GEN ==="
make tp_runner CC=fcc OPENMP=1 >>"$LOG" 2>&1 && log "build_ok" || { log "BUILD FAILED"; log "WSAP_DONE"; exit 1; }

log "--- run A: TP_AR_ROBUST=0 (original passive spin) ---"
run 0 toks_robust0.txt
log "--- run B: TP_AR_ROBUST=1 (drain+civac, default) ---"
run 1 toks_robust1.txt

log "=== PARITY RESULT ==="
if diff -q toks_robust0.txt toks_robust1.txt >/dev/null 2>&1; then
    log "PARITY: IDENTICAL ($(wc -l < toks_robust1.txt) token lines)"
else
    log "PARITY: DIFFER"
    diff toks_robust0.txt toks_robust1.txt | head -10 | tee -a "$LOG"
fi
log "WSAP_DONE $(date +%H:%M:%S)"
