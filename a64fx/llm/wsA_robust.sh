#!/bin/bash
# Workstream A validation — all-reduce dropped-Put robust fix (TP_AR_ROBUST).
#
# Loops the EXACT config that previously deadlocked once — ctx=4096 with
# TP_AR_BATCH=128 (NOT the 512 crutch) — and counts clean passes vs the
# `tp_ar: rank R ... timeout want=N+1 got=N` flake. With the fix (per-recv MRQ
# drain + civac, default ON) we expect N/N clean. Optional A/B: ROBUST=0 reverts
# to the prior passive spin (the flaky baseline).
#
# Funnels everything to ONE log with a final sentinel; the caller reads once.
set -u
cd "$(dirname "$0")"
LOG=${LOG:-wsA_robust.log}
: > "$LOG"
ITERS=${ITERS:-10}
CTX=${CTX:-4096}
GEN=${GEN:-64}
BATCH=${BATCH:-128}
ROBUST=${ROBUST:-1}

log() { echo "$@" | tee -a "$LOG"; }

log "=== Workstream A robust-AR validation $(date +%H:%M:%S) ==="
log "ctx=$CTX gen=$GEN TP_AR_BATCH=$BATCH TP_AR_ROBUST=$ROBUST iters=$ITERS"
log "=== rebuild tp_runner (CC=fcc OPENMP=1) ==="
if make tp_runner CC=fcc OPENMP=1 >>"$LOG" 2>&1; then log "build_ok"; else log "BUILD FAILED"; log "WSA_DONE"; exit 1; fi

clean=0; flake=0; other=0
for i in $(seq 1 "$ITERS"); do
    log "######## iter=$i/$ITERS ctx=$CTX batch=$BATCH robust=$ROBUST ($(date +%H:%M:%S)) ########"
    rm -f tp_run_*.txt tp_perf_rank*.txt tp_tokens_rank00.txt tp_stderr_rank*.txt
    env VCOORD=vcoord_no0.txt SKIP_STAGE=1 SKIP_TOPO=1 QWEN27B_PREPARE=0 \
        NP=11 TP_PREFILL_ONLY=0 TP_IGNORE_EOS=1 TP_MAXGEN="$GEN" \
        TP_AR_BATCH="$BATCH" TP_AR_ROBUST="$ROBUST" \
        TP_SYNTH_TOKENS="$CTX" TP_MAXSEQ=$((CTX + GEN + 32)) TP_DECODE_PERSIST=1 \
        ./run_tp_27b.sh >>"$LOG" 2>&1
    rc=$?
    shopt -s nullglob
    r=( tp_run_*.txt )
    res=""
    [ ${#r[@]} -gt 0 ] && res=$(grep -h "decode(" "${r[0]}" 2>/dev/null | tail -1)
    shopt -u nullglob
    if [ -n "$res" ]; then
        clean=$((clean+1)); log "[iter=$i CLEAN] $res"
    else
        tmo=$(grep -h "tp_ar:.*timeout" tp_stderr_rank*.txt 2>/dev/null | head -1)
        if [ -n "$tmo" ]; then
            flake=$((flake+1)); log "[iter=$i FLAKE(timeout)] rc=$rc $tmo"
        else
            other=$((other+1)); log "[iter=$i OTHER-FAIL] rc=$rc"
            grep -h "SIGSEGV\|poll_tcq rc\|utofu_put rc\|died\|BUILD FAILED" tp_stderr_rank*.txt 2>/dev/null | head -3 | tee -a "$LOG"
        fi
    fi
done

log "==================== WSA SUMMARY ===================="
log "ctx=$CTX batch=$BATCH robust=$ROBUST  clean=$clean flake=$flake other=$other / $ITERS"
log "WSA_DONE $(date +%H:%M:%S)"
