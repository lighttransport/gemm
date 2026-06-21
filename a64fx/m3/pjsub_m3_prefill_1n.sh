#!/bin/bash
# Lever-1 chunked batched prefill validation (1 node, no MPI -> immune to 1907).
# Stage 4 real layers, run m3_real_test token-serial (M3_PCHUNK=0) vs chunked (M3_PCHUNK=128)
# on the SAME synthetic sequence: prefill argmax + ||x|| must MATCH (chunked is mathematically
# identical to token-serial), and chunked prefill tok/s should be much higher (batched GEMMs +
# one reduce/layer/chunk + amortized FP8 decode). prefill=512 (4 chunks of 128).
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_prefill_1n.sh'
#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:25:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p; LLM="$REPO/a64fx/llm"; M3="$REPO/a64fx/m3"
cd "$M3" || exit 2; export OMP_NUM_THREADS=12 LLM_THREADS=12
echo "=== prefill-chunk 1n: $(date) ==="
make -C "$LLM" m3_stage CC=fcc OPENMP=1 >/dev/null || { echo FATAL build stage; exit 3; }
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -DM3_IMPL -D_GNU_SOURCE \
    -I "$REPO/common" -I "$LLM" -o "$LLM/build/m3_real_test" "$M3/m3_real_test.c" -lm || { echo FATAL build; exit 3; }
echo "--- stage 4 layers ($(date)) ---"
M3_MODEL_DIR=$HOME/models/m3-fp8 M3_NSHARDS=31 M3_EP_RANK=0 M3_EP_SIZE=1 \
  M3_STAGE_LAYERS=4 M3_SHARD_LIMIT=2 M3_STAGE_DIR=/local/m3fp8 "$LLM/build/m3_stage" 2>&1 | tail -1
echo "--- token-serial (baseline) ---"
M3_STAGE_DIR=/local/m3fp8 M3_LAYERS=4 M3_MAXPOS=1024 M3_PREFILL=512 M3_DECODE=0 M3_PCHUNK=0 \
  "$LLM/build/m3_real_test" 2>&1 | grep -iE "prefill|argmax|NaN|OK"
echo "--- chunked M=128 ---"
M3_STAGE_DIR=/local/m3fp8 M3_LAYERS=4 M3_MAXPOS=1024 M3_PREFILL=512 M3_DECODE=0 M3_PCHUNK=128 \
  "$LLM/build/m3_real_test" 2>&1 | grep -iE "prefill|argmax|NaN|OK"
echo "=== done $(date) ==="
