#!/bin/bash
# KV-cache precision quality sweep (1 node, 4 real layers incl. 1 MoE/MSA layer).
# Same synthetic forward (identical seed) under three KV-cache precisions:
#   bf16 (default, 8-bit mantissa)  |  fp16 (M3_KV_FP16, 11-bit, the quality reference)
#   |  int4 (M3_INT4_KV, +/-7 per head-group, the 1M-context lever).
# Long prefill (2500 > 18*128=2304) so MSA block-selection (idx cache) is exercised too.
# Compares final ||x|| + last argmax + NaN -> quantifies whether int4 KV degrades quality.
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_kv_quality_1n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p
LLM="$REPO/a64fx/llm"; M3="$REPO/a64fx/m3"
cd "$M3" || exit 2
export OMP_NUM_THREADS=12 LLM_THREADS=12

echo "=== KV quality sweep 1n: $(date) ==="
make -C "$LLM" m3_stage CC=fcc OPENMP=1 >/dev/null || { echo FATAL build stage; exit 3; }
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -DM3_IMPL -D_GNU_SOURCE \
    -I "$REPO/common" -I "$LLM" -o "$LLM/build/m3_real_test" "$M3/m3_real_test.c" -lm \
    || { echo FATAL build real_test; exit 3; }

echo "--- stage 4 layers m3-fp8 ($(date)) ---"
M3_MODEL_DIR=$HOME/models/m3-fp8 M3_NSHARDS=31 M3_EP_RANK=0 M3_EP_SIZE=1 \
  M3_STAGE_LAYERS=4 M3_SHARD_LIMIT=2 M3_STAGE_DIR=/local/m3fp8 "$LLM/build/m3_stage" 2>&1 | tail -1

run(){ # $1=label  $2..=env
  echo "  [$1]"
  env "${@:2}" M3_STAGE_DIR=/local/m3fp8 M3_LAYERS=4 M3_MAXPOS=4096 M3_PREFILL=2500 M3_DECODE=8 \
    M3_MSA=1 OMP_NUM_THREADS=12 "$LLM/build/m3_real_test" 2>&1 | grep -iE "arena|last argmax|NaN|OK|FAIL"
}
echo "--- KV precision sweep (||x|| / argmax should match across precisions if quant is benign) ---"
run "bf16 KV (baseline)"  M3_INT4_KV=0 M3_KV_FP16=0
run "fp16 KV (hi-fidelity)" M3_INT4_KV=0 M3_KV_FP16=1
run "int4 KV (1M lever)"  M3_INT4_KV=1 M3_KV_FP16=0
echo "=== done $(date) ==="
