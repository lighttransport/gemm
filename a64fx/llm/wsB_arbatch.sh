#!/bin/bash
# Workstream B tuning: AR is ~36% of M=1000 prefill (co-dominant w/ SSM) and was
# invisible before. It chunks the M*n_embd reduce by max_count = n_embd*TP_AR_BATCH;
# at batch=128 a 1000-token prefill splits each all-reduce into 8 chunks, each paying
# full recursive-doubling latency. Workstream A made larger chunks safe. Sweep
# TP_AR_BATCH and measure prefill tok/s + the per-phase ar bucket. bf16 payload keeps
# the per-chunk Put < 16MiB cap; decode is unaffected (always one n_embd chunk).
set -u
cd "$(dirname "$0")"
LOG=${LOG:-wsB_arbatch.log}
: > "$LOG"
M=${M:-1000}
BATCHES=${BATCHES:-"128 256 512 1024"}
log() { echo "$@" | tee -a "$LOG"; }

log "=== WSB AR-batch sweep $(date +%H:%M:%S)  M=$M batches=[$BATCHES] ==="
make tp_runner CC=fcc OPENMP=1 >>"$LOG" 2>&1 && log build_ok || { log "BUILD FAILED"; log "WSBB_DONE"; exit 1; }

for B in $BATCHES; do
    log "######## M=$M TP_AR_BATCH=$B ($(date +%H:%M:%S)) ########"
    rm -f tp_run_*.txt tp_perf_rank*.txt tp_stderr_rank*.txt tp_prefill_rank*.txt
    env VCOORD=vcoord_no0.txt SKIP_STAGE=1 SKIP_TOPO=1 QWEN27B_PREPARE=0 \
        NP=11 TP_PREFILL_ONLY=1 TP_MAXGEN=0 TF_PREFILL_PROF=1 TP_AR_BATCH="$B" \
        TP_SYNTH_TOKENS="$M" TP_MAXSEQ=$((M + 64)) \
        ./run_tp_27b.sh >>"$LOG" 2>&1
    pf=$(grep -hE "prefill-only:|prefill\([0-9]+ tok\)" tp_run_*.txt tp_stderr_rank00.txt 2>/dev/null | head -1)
    ph=$(grep -h "TF_PREFILL_PROF" tp_stderr_rank00.txt 2>/dev/null | tail -1)
    rg=$(grep -h "tp_ar region:" tp_stderr_rank00.txt 2>/dev/null | head -1)
    log "[B=$B] $pf"
    log "[B=$B] $ph"
    log "[B=$B] $rg"
done

log "==================== WSBB SUMMARY ===================="
grep -hE "^\[B=[0-9]+\] prefill-only|^\[B=[0-9]+\]   \[TF_PREFILL" "$LOG"
log "WSBB_DONE $(date +%H:%M:%S)"
