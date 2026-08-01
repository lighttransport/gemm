#!/bin/bash
# MXFP8 multi-stream decode-once microbench (1 node, no MPI -> immune to the 1907 flake).
# Measures m3_gemm_mxfp8 per-stream throughput at N=1 vs 2/4/8 on real expert + attn shapes.
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_mxgemm_bench_1n.sh'
#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p; LLM="$REPO/a64fx/llm"; M3="$REPO/a64fx/m3"
export OMP_NUM_THREADS=12
echo "=== MXFP8 decode-once microbench: $(date) ==="
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -DM3_IMPL -D_GNU_SOURCE \
    -I "$REPO/common" -I "$LLM" -o /tmp/mxg "$M3/m3_mxgemm_bench.c" -lm || { echo FATAL build; exit 3; }
echo "--- expert w1/w3 shape rows=3072 cols=6144 ---"; OMP_NUM_THREADS=12 /tmp/mxg 3072 6144 300
echo "--- expert w2 shape rows=6144 cols=3072 ---";    OMP_NUM_THREADS=12 /tmp/mxg 6144 3072 300
echo "--- attn wq shape rows=8192 cols=6144 ---";      OMP_NUM_THREADS=12 /tmp/mxg 8192 6144 300
echo "=== done $(date) ==="
