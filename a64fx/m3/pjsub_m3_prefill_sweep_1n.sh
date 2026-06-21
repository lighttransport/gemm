#!/bin/bash
# Prefill optimization probe (1 node, no MPI -> immune to 1907, runs immediately).
# Characterizes chunked prefill: chunk-size sweep, thread/CMG scaling, int4-KV composition.
# All on 4 real layers, prefill 512. argmax must stay 3051 (== token-serial) for every config.
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_prefill_sweep_1n.sh'
#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:25:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p; LLM="$REPO/a64fx/llm"; M3="$REPO/a64fx/m3"
cd "$M3" || exit 2; export LLM_THREADS=12
echo "=== prefill sweep 1n: $(date) ==="
make -C "$LLM" m3_stage CC=fcc OPENMP=1 >/dev/null || { echo FATAL; exit 3; }
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -DM3_IMPL -D_GNU_SOURCE \
    -I "$REPO/common" -I "$LLM" -o "$LLM/build/m3_real_test" "$M3/m3_real_test.c" -lm || { echo FATAL; exit 3; }
echo "--- stage 4 layers ($(date)) ---"
M3_MODEL_DIR=$HOME/models/m3-fp8 M3_NSHARDS=31 M3_EP_RANK=0 M3_EP_SIZE=1 \
  M3_STAGE_LAYERS=4 M3_SHARD_LIMIT=2 M3_STAGE_DIR=/local/m3fp8 "$LLM/build/m3_stage" 2>&1 | tail -1
run(){ # $1=label  rest=env
  echo "  [$1]"; env "${@:2}" M3_STAGE_DIR=/local/m3fp8 M3_LAYERS=4 M3_MAXPOS=1024 M3_PREFILL=512 M3_DECODE=0 \
    "$LLM/build/m3_real_test" 2>&1 | grep -iE "prefill |argmax|NaN" | head -3; }
echo "--- chunk-size sweep (12 threads / 1 CMG) ---"
run "serial"   OMP_NUM_THREADS=12 M3_PCHUNK=0
run "M=64"     OMP_NUM_THREADS=12 M3_PCHUNK=64
run "M=128"    OMP_NUM_THREADS=12 M3_PCHUNK=128
run "M=256"    OMP_NUM_THREADS=12 M3_PCHUNK=256
run "M=512"    OMP_NUM_THREADS=12 M3_PCHUNK=512
echo "--- thread/CMG scaling (M=256) ---"
run "M=256 t=24" OMP_NUM_THREADS=24 M3_PCHUNK=256
run "M=256 t=48" OMP_NUM_THREADS=48 M3_PCHUNK=256
echo "--- int4-KV + chunked (M=256) ---"
run "M=256 int4" OMP_NUM_THREADS=12 M3_PCHUNK=256 M3_INT4_KV=1
echo "=== done $(date) ==="
