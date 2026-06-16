#!/bin/bash
# Native 1-node validation of (a) int4-KV and (b) MXFP8 multi-stream throughput.
# Stage 4 layers (0-2 dense + 3 MoE/MSA) of ~/models/m3-fp8 once, then:
#  (1) m3_mxfp8_test          -> decode-once kernel bit-exact
#  (2) m3_real_test bf16-KV vs int4-KV, long prefill (>2304 -> MSA triggers) -> ||x|| match + NaN
#  (3) m3_ep_runner MXFP8 M3_MSTREAM=1 vs 8 -> per-stream + aggregate decode tok/s (decode-once win)
# Submit: ssh fugaku 'cd ~/work/gemm/ds4p && pjsub --no-check-directory a64fx/m3/pjsub_m3_kv_mstream_1n.sh'

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
  M3_STAGE_LAYERS=4 M3_SHARD_LIMIT=2 M3_STAGE_DIR=/local/m3fp8 "$LLM/build/m3_stage" 2>&1 | tail -2

echo "--- (2) int4-KV coherence: bf16 vs int4 (MAXPOS=4096, prefill 2500 -> MSA on) ---"
for I4 in 0 1; do
  echo "  M3_INT4_KV=$I4:"
  M3_STAGE_DIR=/local/m3fp8 M3_LAYERS=4 M3_MAXPOS=4096 M3_PREFILL=2500 M3_DECODE=8 M3_MSA=1 \
    M3_INT4_KV=$I4 OMP_NUM_THREADS=12 "$LLM/build/m3_real_test" 2>&1 | grep -iE "load OK|arena|prefill|decode|argmax|NaN|OK"
done

echo "--- (3) MXFP8 multi-stream: N=1 vs N=8 (real, 4 layers) ---"
for NS in 1 8; do
  echo "  M3_MSTREAM=$NS:"
  M3_REAL=1 M3_STAGE_DIR=/local/m3fp8 M3_EP_SIZE=1 M3_TP=0 M3_MSA=1 \
    M3_LAYERS=4 M3_MAXPOS=256 M3_PREFILL=8 M3_DECODE=24 M3_MSTREAM=$NS \
    mpiexec -np 1 "$LLM/build/m3_ep_runner" 2>&1 | grep -iE "MSTREAM|AGG|per-stream|^.*decode:|tok/s|NaN|FATAL" | head -8
done
echo "=== done $(date) ==="
