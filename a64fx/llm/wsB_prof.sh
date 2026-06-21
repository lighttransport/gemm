#!/bin/bash
# Workstream B measure-first: per-phase prefill profiling for Qwen3.6-27B TP=11.
# Runs prefill-ONLY at several M with TF_PREFILL_PROF=1 (new split: attn/ssm/ffn/ar)
# and reports, per M: overall prefill tok/s + the per-phase seconds breakdown.
# This tells us which bucket to tune before touching any kernel. Funnels to one log
# + sentinel; caller reads once.
set -u
cd "$(dirname "$0")"
LOG=${LOG:-wsB_prof.log}
: > "$LOG"
MS=${MS:-"128 512 1000"}
log() { echo "$@" | tee -a "$LOG"; }

log "=== WSB prefill per-phase profile $(date +%H:%M:%S)  M=[$MS] ==="
log "=== rebuild tp_runner (CC=fcc OPENMP=1) ==="
make tp_runner CC=fcc OPENMP=1 >>"$LOG" 2>&1 && log build_ok || { log "BUILD FAILED"; log "WSB_DONE"; exit 1; }

for M in $MS; do
    log "######## M=$M prefill-only ($(date +%H:%M:%S)) ########"
    rm -f tp_run_*.txt tp_perf_rank*.txt tp_stderr_rank*.txt tp_prefill_rank*.txt
    env VCOORD=vcoord_no0.txt SKIP_STAGE=1 SKIP_TOPO=1 QWEN27B_PREPARE=0 \
        NP=11 TP_PREFILL_ONLY=1 TP_MAXGEN=0 TF_PREFILL_PROF=1 \
        TP_SYNTH_TOKENS="$M" TP_MAXSEQ=$((M + 64)) \
        ./run_tp_27b.sh >>"$LOG" 2>&1
    # overall prefill tok/s (rank0 logmsg) + the per-phase split (rank0 stderr)
    pf=$(grep -hE "prefill-only:|prefill\([0-9]+ tok\)" tp_run_*.txt tp_stderr_rank00.txt 2>/dev/null | head -1)
    ph=$(grep -h "TF_PREFILL_PROF" tp_stderr_rank00.txt 2>/dev/null | tail -1)
    log "[M=$M] $pf"
    log "[M=$M] $ph"
done

log "==================== WSB SUMMARY ===================="
grep -hE "^\[M=" "$LOG"
log "WSB_DONE $(date +%H:%M:%S)"
