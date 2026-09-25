#!/usr/bin/env bash
# Six- and twelve-stage FP4 prefill sweep on a fresh twelve-node allocation.
# Submit from the Fugaku frontend with: pjsub --no-check-directory pjsub_pp12_v2.sh
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=12,elapse=02:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=12"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -euo pipefail

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$HOME/work/gemm/qwen38-27b"
module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
unset OPAL_PREFIX
export Q38P_STAGE_SOURCE="$PWD/tmp/q38-lowbit-20260924/hw-51893515/fp4-v1.image"
test -s "$Q38P_STAGE_SOURCE"

echo "q38p_pp12: job=$PJM_JOBID stage start $(date)"
mpiexec -n 12 sh -c '
    mkdir -p /local/q38/bin /local/q38/final /local/q38/tmp
    test -s /local/q38/fp4.image ||
      dd if="$Q38P_STAGE_SOURCE" of=/local/q38/fp4.image bs=8M iflag=direct oflag=direct status=none
    test -s /local/q38/fp4.image
'
cp tmp/q38-fast-final/fp4-f32-1024-ref.log /local/q38/ref-f32.log
echo "q38p_pp12: stage done $(date)"

bash a64fx/llm/q38p/build_pp.sh
echo "q38p_pp12: build done $(date)"
bash a64fx/llm/q38p/run_pp.sh 6 8 1024 160 1
bash a64fx/llm/q38p/run_pp.sh 12 8 1024 160 1
echo "q38p_pp12: done $(date)"
