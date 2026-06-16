#!/bin/bash
# Validate + benchmark the batched-parallel MSA select (1 node, 4 real layers incl. 1 MoE/MSA).
# prefill 2600 (> keep*128=2304 -> MSA top-k ACTIVE). token-serial (M3_PCHUNK=0, sequential
# m3_msa_select) vs chunked M=256 (m3_msa_prefill_select, parallel) must give the SAME argmax;
# chunked at 48 threads should be far faster (the O(pos) block scoring now spreads over all CMGs).
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_msaopt_1n.sh'
#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p; LLM="$REPO/a64fx/llm"; M3="$REPO/a64fx/m3"
cd "$M3" || exit 2; export LLM_THREADS=48
echo "=== MSA-opt validate 1n: $(date) ==="
make -C "$LLM" m3_stage CC=fcc OPENMP=1 >/dev/null || { echo FATAL; exit 3; }
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -DM3_IMPL -D_GNU_SOURCE \
    -I "$REPO/common" -I "$LLM" -o "$LLM/build/m3_real_test" "$M3/m3_real_test.c" -lm || { echo FATAL; exit 3; }
echo "--- stage 4 layers ($(date)) ---"
M3_MODEL_DIR=$HOME/models/m3-fp8 M3_NSHARDS=31 M3_EP_RANK=0 M3_EP_SIZE=1 \
  M3_STAGE_LAYERS=4 M3_SHARD_LIMIT=2 M3_STAGE_DIR=/local/m3fp8 "$LLM/build/m3_stage" 2>&1 | tail -1
run(){ echo "  [$1]"; env "${@:2}" M3_STAGE_DIR=/local/m3fp8 M3_LAYERS=4 M3_MAXPOS=4096 M3_PREFILL=2600 M3_DECODE=0 M3_MSA=1 \
    "$LLM/build/m3_real_test" 2>&1 | grep -iE "prefill |argmax|NaN"; }
echo "--- token-serial baseline (slow; sequential MSA select), 12 threads ---"
run "serial" OMP_NUM_THREADS=12 M3_PCHUNK=0
echo "--- chunked M=256 + parallel MSA select, 48 threads (argmax must match serial) ---"
run "chunked-t48" OMP_NUM_THREADS=48 M3_PCHUNK=256
echo "--- chunked M=256, 12 threads (apples-to-apples vs serial) ---"
run "chunked-t12" OMP_NUM_THREADS=12 M3_PCHUNK=256
echo "=== done $(date) ==="
