#!/bin/bash
# 27B single-node (NO TP) ground truth via 4-bit Q4_0 (~16GB) -> fits RESIDENT in
# 32GB HBM, so NO --mmap, NO page-fault thrash (the BF16 54GB model thrashed and
# hung the node). Q4_0 of a working model is still coherent, so this cleanly tests
# whether the 27B FORWARD is fine or degenerate, independent of TP.
#   TF_DUMP_LOGITS per-step:
#     nan/inf>0     -> numerical overflow in the 27B forward (TP-independent)
#     all_equal=YES -> uniform logits (structural degeneracy, TP-independent)
#     sane text     -> forward FINE -> "!!!!" is a TP=11-only bug
# TF_NO_PANEL/TF_NO_BF16_PV keep weights as Q4_0 (no 54GB bf16 repack -> no OOM).
cd "$(dirname "$0")"
OUT=iso_27b_q4.out; : > "$OUT"
M=/local/q4/Qwen3.6-27B-Q4_0.gguf
echo "=== 27B single-node Q4_0 isolation $(date +%H:%M:%S) ===" | tee -a "$OUT"
echo "model=$M" | tee -a "$OUT"
timeout 900 env OMP_NUM_THREADS=48 TF_NO_PANEL=1 TF_NO_BF16_PV=1 TF_DUMP_LOGITS=1 \
    ./build/llm_runner "$M" \
    --prompt "The capital of France is" --max-gen 16 --llm-threads 48 \
    >>"$OUT" 2>&1
echo "RC=$?" | tee -a "$OUT"
echo "ISO27B_Q4_DONE $(date +%H:%M:%S)" | tee -a "$OUT"
