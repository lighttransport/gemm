#!/bin/bash
# Workstream C perf: decode per-token cost vs context, attn_pp OFF vs ON.
# Synthetic prefill of CTX tokens, then decode ~GEN tokens; tp_curve_rank00.txt
# records fwd_ms per generated token at pos>=CTX. We average the decode rows to
# get steady-state ms/tok at that context, and report tok/s = 1000/ms_tok.
# Goal: ctx4096 & ctx8192 cross 20 tok/s with attn_pp ON (C0 projected ~34/~26).
set -u
cd "$(dirname "$0")"
LOG=${LOG:-attn_pp_curve.log}
: > "$LOG"
GEN=${GEN:-20}
CTXS=${CTXS:-"4096 8192"}
COMMON=(VCOORD=vcoord_no0.txt SKIP_STAGE=1 SKIP_TOPO=1 QWEN27B_PREPARE=0 NP=11
        TP_MAXGEN=$GEN TP_IGNORE_EOS=1)

measure () {  # $1=ctx $2=pp_val $3=label
    local ctx=$1 pp=$2 lbl=$3
    rm -f tp_curve_rank00.txt
    # synth prefill = ctx tokens; cache must hold ctx+GEN
    env "${COMMON[@]}" TF_ATTN_PP=$pp TP_SYNTH_TOKENS=$ctx \
        TP_MAXSEQ=$((ctx+GEN+8)) ./run_tp_27b.sh >>"$LOG" 2>&1
    # average decode-row fwd_ms (col3), skip first 2 rows (warm)
    local ms; ms=$(awk 'NR>3{s+=$3;n++} END{if(n)printf "%.3f",s/n; else print "NA"}' tp_curve_rank00.txt)
    local tps; tps=$(awk -v m="$ms" 'BEGIN{if(m+0>0)printf "%.2f",1000.0/m; else print "NA"}')
    printf "CURVE %s ctx=%-5d ms_tok=%-8s tok_s=%s\n" "$lbl" "$ctx" "$ms" "$tps" | tee -a "$LOG"
}

for ctx in $CTXS; do
    measure "$ctx" 0 OFF
    measure "$ctx" 1 ON
done
echo "CURVE_DONE $(date +%H:%M:%S)" | tee -a "$LOG"
