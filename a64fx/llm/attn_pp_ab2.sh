#!/bin/bash
# Robust A/B token-identity for position-parallel attention (direct mpiexec; model
# already on /local, topo already fresh). Watchdog per run + teardown gap + stray
# sweep between runs to avoid the back-to-back uTofu load-stage race that wedged
# rank 5 for 40 min. A=TF_ATTN_PP=0 (head-parallel), B=TF_ATTN_PP=1 (position-par).
set -u
cd "$(dirname "$0")"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
LOG=attn_pp_ab2.log; : > "$LOG"
CAP=${CAP:-240}; MAXGEN=${MAXGEN:-24}
MODEL_LOCAL=/local/qwen36/27b/Qwen3.6-27B-BF16-00001-of-00002.gguf
BIN="$PWD/build/tp_runner"
export GGUF_LAZY_MMAP=1 LLM_THREADS=48 OMP_NUM_THREADS=48
export TP_PROMPT="Hello, who are you?" TP_MAXGEN=$MAXGEN TP_DUMP_TOKENS=1
export TP_IGNORE_EOS=1 QWEN27B_LOCAL_DIR=/local/qwen36/27b

echo "=== build $(date +%H:%M:%S) ===" | tee -a "$LOG"
make tp_runner CC=fcc OPENMP=1 >>"$LOG" 2>&1 && echo BUILD_OK | tee -a "$LOG"

sweep () { mpiexec -np 11 -vcoordfile vcoord_no0.txt sh -c \
    'pkill -9 -f build/tp_runner >/dev/null 2>&1; true' >/dev/null 2>&1; }

run_one () {  # $1=label $2=PP
    echo "=== run $1 (TF_ATTN_PP=$2) $(date +%H:%M:%S) ===" | tee -a "$LOG"
    rm -f tp_tokens_rank00.txt tp_load_rank*.txt
    TF_ATTN_PP=$2 timeout "$CAP" mpiexec -np 11 -vcoordfile vcoord_no0.txt "$BIN" "$MODEL_LOCAL" >>"$LOG" 2>&1
    local rc=$?
    local lf; lf=$(ls tp_load_rank*.txt 2>/dev/null | wc -l)
    local tk; tk=$(wc -l < tp_tokens_rank00.txt 2>/dev/null || echo 0)
    echo "run $1: rc=$rc load=$lf/11 tokens=$tk (124=hang)" | tee -a "$LOG"
    cp -f tp_tokens_rank00.txt "tokens_$1.txt" 2>/dev/null
    echo "$rc"
}

rc1=$(run_one A 0 | tail -1)
sweep; sleep 8
rc2=$(run_one B 1 | tail -1)

echo "=== DIFF ===" | tee -a "$LOG"
if [ -s tokens_A.txt ] && [ -s tokens_B.txt ] && diff -q tokens_A.txt tokens_B.txt >/dev/null 2>&1; then
    echo "TOKENS_IDENTICAL PASS ($(wc -l < tokens_A.txt) tokens)" | tee -a "$LOG"
else
    echo "TOKENS_DIFFER_OR_MISSING FAIL" | tee -a "$LOG"
    echo "A=$(wc -l < tokens_A.txt 2>/dev/null) B=$(wc -l < tokens_B.txt 2>/dev/null)" | tee -a "$LOG"
    diff tokens_A.txt tokens_B.txt 2>&1 | head -30 | tee -a "$LOG"
fi
echo "AB2_DONE $(date +%H:%M:%S)" | tee -a "$LOG"
