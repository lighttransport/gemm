#!/bin/bash
# Native 1-node: (1) MXFP8 multi-stream N=1 vs N=8 throughput (decode-once kernel),
# (2) KV-cache precision quality sweep bf16/fp16/int4 (same forward, identical seed).
# Prefill kept short (dense attention is O(n^2); int4 K/V error already shows per-token);
# MSA-idx int4 is exercised functionally in the CP 6200-prefill job. One staging, fast runs.
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_kv_mstream_1n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:25:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p
LLM="$REPO/a64fx/llm"; M3="$REPO/a64fx/m3"
cd "$M3" || exit 2
export OMP_NUM_THREADS=12 LLM_THREADS=12

echo "=== KV+mstream 1n: $(date) ==="
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -I "$REPO/common" -I "$LLM" \
    -o /tmp/m3_mxfp8_test "$M3/m3_mxfp8_test.c" -lm || { echo FATAL build test; exit 3; }
echo "--- (1) decode-once kernel ---"; N=6144 /tmp/m3_mxfp8_test | tail -1; N=3072 /tmp/m3_mxfp8_test | tail -1
make -C "$LLM" m3_stage m3_ep_runner CC=fcc OPENMP=1 >/dev/null || { echo FATAL build; exit 3; }
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -DM3_IMPL -D_GNU_SOURCE \
    -I "$REPO/common" -I "$LLM" -o "$LLM/build/m3_real_test" "$M3/m3_real_test.c" -lm \
    || { echo FATAL build real_test; exit 3; }

echo "--- stage 4 layers m3-fp8 ($(date)) ---"
M3_MODEL_DIR=$HOME/models/m3-fp8 M3_NSHARDS=31 M3_EP_RANK=0 M3_EP_SIZE=1 \
  M3_STAGE_LAYERS=4 M3_SHARD_LIMIT=2 M3_STAGE_DIR=/local/m3fp8 "$LLM/build/m3_stage" 2>&1 | tail -1

echo "--- (2) MXFP8 multi-stream: N=1 vs N=8 (real, 4 layers) ($(date)) ---"
for NS in 1 8; do echo "  M3_MSTREAM=$NS:"
  M3_REAL=1 M3_STAGE_DIR=/local/m3fp8 M3_EP_SIZE=1 M3_TP=0 M3_MSA=1 \
    M3_LAYERS=4 M3_MAXPOS=256 M3_PREFILL=8 M3_DECODE=24 M3_MSTREAM=$NS \
    mpiexec -np 1 "$LLM/build/m3_ep_runner" 2>&1 | grep -iE "MSTREAM|AGG|per-stream|^.*decode:|NaN|FATAL" | head -6
done

echo "--- (3) KV precision quality: bf16 vs fp16 vs int4 (prefill 320, identical seed) ($(date)) ---"
for KV in "bf16 M3_INT4_KV=0 M3_KV_FP16=0" "fp16 M3_INT4_KV=0 M3_KV_FP16=1" "int4 M3_INT4_KV=1 M3_KV_FP16=0"; do
  set -- $KV; lbl=$1; shift; echo "  [$lbl]"
  env "$@" M3_STAGE_DIR=/local/m3fp8 M3_LAYERS=4 M3_MAXPOS=512 M3_PREFILL=320 M3_DECODE=8 M3_MSA=1 \
    OMP_NUM_THREADS=12 "$LLM/build/m3_real_test" 2>&1 | grep -iE "arena|last argmax|NaN"
done
echo "=== done $(date) ==="
