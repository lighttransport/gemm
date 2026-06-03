#!/bin/bash
# Workstream C deliverable: decode per-token cost vs context, attn_pp OFF vs ON.
# Robust against the orphaned-org/mpiexec stray problem: each run is watchdogged,
# and teardown KILLS the reparented launcher (PPID=1 org/mpiexec) + remote-sweeps
# + gaps, so back-to-back runs never wedge on saturated coordinates.
# Timing is valid regardless of token sense (attn does identical work).
set -u
cd "$(dirname "$0")"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
OUT=attn_pp_curve2.out; : > "$OUT"
GEN=${GEN:-14}; CAP=${CAP:-300}
CTXS=${CTXS:-"4096 8192"}
say(){ echo "$@" | tee -a "$OUT"; }

teardown(){
    pkill -9 -f 'org/mpiexec.*tp_runner'  >/dev/null 2>&1
    pkill -9 -f 'plexec.*tp_runner'       >/dev/null 2>&1
    pkill -9 -f 'run_tp_27b'              >/dev/null 2>&1
    sleep 2
    # remote sweep now that launcher slots are free
    timeout 40 mpiexec -np 11 -vcoordfile vcoord_no0.txt sh -c \
        'pkill -9 -f build/tp_runner >/dev/null 2>&1; true' >/dev/null 2>&1
    sleep 4
}

measure(){  # $1=ctx $2=pp $3=label
    local ctx=$1 pp=$2 lbl=$3
    rm -f tp_curve_rank00.txt
    say "=== measure $lbl ctx=$ctx TF_ATTN_PP=$pp $(date +%H:%M:%S) ==="
    timeout "$CAP" env VCOORD=vcoord_no0.txt SKIP_STAGE=1 SKIP_TOPO=1 QWEN27B_PREPARE=0 \
        NP=11 TP_MAXGEN=$GEN TP_IGNORE_EOS=1 TF_ATTN_PP=$pp \
        TP_SYNTH_TOKENS=$ctx TP_MAXSEQ=$((ctx+GEN+8)) \
        ./run_tp_27b.sh >>"$OUT" 2>&1
    local rc=$?
    local rows; rows=$(wc -l < tp_curve_rank00.txt 2>/dev/null || echo 0)
    # average decode rows, skip first 2 (warm); col3=fwd_ms col4=comm_ms
    local ms; ms=$(awk 'NR>2{f+=$3;c+=$4;n++} END{if(n)printf "%.3f %.3f",f/n,c/n; else print "NA NA"}' tp_curve_rank00.txt 2>/dev/null)
    local fwd=${ms% *}; local comm=${ms#* }
    local tps; tps=$(awk -v m="$fwd" 'BEGIN{if(m+0>0)printf "%.2f",1000.0/m; else print "NA"}')
    say "CURVE $lbl ctx=$ctx rc=$rc rows=$rows fwd_ms=$fwd comm_ms=$comm tok_s=$tps"
    cp -f tp_curve_rank00.txt "curve_${lbl}_${ctx}.txt" 2>/dev/null
    teardown
}

say "=== build (mine, attn_pp) $(date +%H:%M:%S) ==="
make tp_runner CC=fcc OPENMP=1 >>"$OUT" 2>&1 && say BUILD_OK
teardown   # clean slate
for ctx in $CTXS; do
    measure "$ctx" 0 OFF
    measure "$ctx" 1 ON
done
say "=== SUMMARY ==="
grep '^CURVE' "$OUT" | tee -a "$OUT"
say "CURVE2_DONE $(date +%H:%M:%S)"
