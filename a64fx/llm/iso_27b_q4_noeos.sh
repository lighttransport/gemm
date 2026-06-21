#!/bin/bash
# Continue past EOS + per-step logit dump: proves the single-node 27B forward is
# coherent (healthy non-degenerate logits => NOT the source of TP=11's "!!!!").
cd "$(dirname "$0")"
OUT=iso_27b_q4_noeos.out; : > "$OUT"
M=/local/q4/Qwen3.6-27B-Q4_0.gguf
echo "=== 27B Q4 no-eos-stop $(date +%H:%M:%S) ===" | tee -a "$OUT"
timeout 900 env OMP_NUM_THREADS=48 TF_NO_PANEL=1 TF_NO_BF16_PV=1 \
    TF_DUMP_LOGITS=1 TF_NO_EOS_STOP=1 \
    ./build/llm_runner "$M" \
    --prompt "The capital of France is" --max-gen 12 --llm-threads 48 \
    >>"$OUT" 2>&1
echo "RC=$?" | tee -a "$OUT"
echo "ISO27B_Q4_NOEOS_DONE $(date +%H:%M:%S)" | tee -a "$OUT"
