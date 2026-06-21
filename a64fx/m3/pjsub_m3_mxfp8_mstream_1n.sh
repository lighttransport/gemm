#!/bin/bash
# Native 1-node MXFP8 multi-stream validation + throughput:
#  (1) m3_mxfp8_test  -> decode-once (bf16-tile) kernel == scalar ref (correctness)
#  (2) stage 4 layers of ~/models/m3-fp8 to /local, run m3_ep_runner (np=1, M3_REAL=1,
#      M3_LAYERS=4) at M3_MSTREAM=1 and M3_MSTREAM=8 -> per-stream + aggregate decode tok/s.
# The decode-once tile amortizes the FP8 LUT-gather decode across the N streams (the win).
#
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_mxfp8_mstream_1n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:20:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4p
LLM="$REPO/a64fx/llm"; M3="$REPO/a64fx/m3"
cd "$M3" || exit 2
export OMP_NUM_THREADS=12 LLM_THREADS=12

echo "=== MXFP8 mstream 1n: $(date) ==="
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -I "$REPO/common" -I "$LLM" \
    -o /tmp/m3_mxfp8_test "$M3/m3_mxfp8_test.c" -lm || { echo "FATAL build test"; exit 3; }
echo "--- (1) decode-once kernel correctness ---"
N=6144 /tmp/m3_mxfp8_test
N=3072 /tmp/m3_mxfp8_test

make -C "$LLM" m3_stage m3_ep_runner CC=fcc OPENMP=1 >/dev/null || { echo "FATAL build runner"; exit 3; }

echo "--- (2) stage 4 layers of m3-fp8 ---"
M3_MODEL_DIR=$HOME/models/m3-fp8 M3_NSHARDS=31 M3_EP_RANK=0 M3_EP_SIZE=1 \
  M3_STAGE_LAYERS=4 M3_SHARD_LIMIT=2 M3_STAGE_DIR=/local/m3fp8 \
  "$LLM/build/m3_stage" 2>&1 | tail -2

for NS in 1 8; do
  echo "--- (2) M3_MSTREAM=$NS real MXFP8 decode (4 layers) ---"
  M3_REAL=1 M3_STAGE_DIR=/local/m3fp8 M3_EP_SIZE=1 M3_TP=0 M3_MSA=1 \
    M3_LAYERS=4 M3_MAXPOS=256 M3_PREFILL=8 M3_DECODE=24 M3_MSTREAM=$NS \
    mpiexec -np 1 "$LLM/build/m3_ep_runner" 2>&1 | grep -iE "mstream|decode|tok/s|aggregate|arena|NaN|SENTINEL|FATAL" | head -20
done
echo "=== done $(date) ==="
