#!/bin/bash
# GLM-5.2 partial real-weight stager test. This validates safetensors reading,
# tensor classification, expert ownership, manifest/blob output, and /local I/O
# using already downloaded shards. It does not run the real-weight loader.

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=1,elapse=01:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=1"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"
GLM5="$REPO/a64fx/glm5"
cd "$GLM5" || exit 2
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-1}
export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm5.2}
export GLM5_STAGE_DIR=/local/glm5
export GLM5_NSHARDS=282
export GLM5_EP_SIZE=$NP
export GLM5_STATUS_DIR="$GLM5"
export GLM5_STAGE_LAYERS=${GLM5_STAGE_LAYERS:-4}
export GLM5_SHARD_LIMIT=${GLM5_SHARD_LIMIT:-120}

echo "=== GLM5.2 partial stage 1n: layers=$GLM5_STAGE_LAYERS shard_limit=$GLM5_SHARD_LIMIT model=$GLM5_MODEL_DIR job=${PJM_JOBID:-?} ==="
date

[ -s "$GLM5_MODEL_DIR/model-00001-of-00282.safetensors" ] || { echo "FATAL: missing shard 1"; exit 2; }
[ -s "$GLM5_MODEL_DIR/model-00079-of-00282.safetensors" ] || { echo "FATAL: missing shard 79 for layer-3 MoE"; exit 2; }

rm -f glm5_stage_rank*.txt
make -C "$LLM" glm5_stage CC=fcc OPENMP=1 >/dev/null || exit 3
mpiexec -np "$NP" "$LLM/build/glm5_stage" || exit 4

echo "--- stage status ---"
cat "$GLM5"/glm5_stage_rank*.txt 2>/dev/null | sort
echo "=== partial stage done $(date) ==="
