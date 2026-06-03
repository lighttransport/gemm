#!/bin/bash
# Workstream C validation: token-identity A/B for position-parallel attention.
# A = TF_ATTN_PP=0 (head-parallel baseline), B = TF_ATTN_PP=1 (position-parallel).
# m/l fp32 + exact-argmax reduction => generated token stream MUST be byte-identical.
# Weights already on /local; topo already placed. Funnel to ONE log + sentinel.
set -u
cd "$(dirname "$0")"
LOG=${LOG:-attn_pp_ab.log}
: > "$LOG"
COMMON=(VCOORD=vcoord_no0.txt SKIP_STAGE=1 SKIP_TOPO=1 QWEN27B_PREPARE=0 NP=11
        TP_DUMP_TOKENS=1 TP_MAXGEN=${TP_MAXGEN:-24} TP_IGNORE_EOS=${TP_IGNORE_EOS:-1}
        TP_PROMPT=${TP_PROMPT:-"Hello, who are you?"})

run_one () {   # $1 = label, $2 = TF_ATTN_PP value
    echo "=== run $1 (TF_ATTN_PP=$2) $(date +%H:%M:%S) ===" | tee -a "$LOG"
    rm -f tp_tokens_rank00.txt
    env "${COMMON[@]}" TF_ATTN_PP=$2 ./run_tp_27b.sh >>"$LOG" 2>&1
    cp -f tp_tokens_rank00.txt "tokens_$1.txt" 2>/dev/null || echo "NO TOKENS $1" | tee -a "$LOG"
    echo "--- run $1 tokens ($(wc -l < tokens_$1.txt 2>/dev/null) lines) ---" | tee -a "$LOG"
}

run_one A 0
run_one B 1

echo "=== DIFF A vs B ===" | tee -a "$LOG"
if diff -q tokens_A.txt tokens_B.txt >/dev/null 2>&1; then
    echo "TOKENS_IDENTICAL PASS" | tee -a "$LOG"
else
    echo "TOKENS_DIFFER FAIL" | tee -a "$LOG"
    diff tokens_A.txt tokens_B.txt | head -40 | tee -a "$LOG"
fi
echo "AB_DONE $(date +%H:%M:%S)" | tee -a "$LOG"
