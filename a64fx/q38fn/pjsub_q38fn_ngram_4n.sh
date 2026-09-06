#!/bin/bash
# PJM-launched four-rank Qwen n-gram staging and resident uTofu pipeline probe.
#
# This must be submitted as a normal MPI PJM job.  A bash-over-HTTP interactive
# shell is already a PJM/plexec process, so launching mpiexec from that shell is
# rejected by Fugaku (PLE 0008).  Keep staging and the resident probe in this
# one job because /local is node-private and is wiped at job start/end.
#
#   pjsub --no-check-directory -x Q38FN_PIPE_ITERS=1000 \
#     a64fx/q38fn/pjsub_q38fn_ngram_4n.sh

#PJM -g hp250467
#PJM -L "rscgrp=small,node=4,elapse=06:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=4"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j

set -u

REPO=${Q38FN_REPO:-$HOME/work/gemm/glm53f}
Q38FN="$REPO/q38fn"
MODEL=${Q38FN_MODEL:-$HOME/models/q38fn/bf16}
NP=${PJM_MPI_PROC:-4}
JOB=${PJM_JOBID:-manual_$$}
WORK="$REPO/q38fn/runs/ngram_${JOB}"
STAGE="/local/$USER/q38fn-ngram-${JOB}"
TOPO="$WORK/tofu_topo.txt"

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
# The LLVM module's mpiclang wrapper can point at an unavailable wrapper-data
# file in batch environments.  TCSDS mpifcc is usable after clang mode is
# selected explicitly; clang mode is also required for C11 atomics.
module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
export Q38FN_NGRAM_STAGE_BASE="$STAGE"
export Q38FN_NGRAM_STAGE_LIMIT=${Q38FN_NGRAM_STAGE_LIMIT:-128}
export Q38FN_STAGE_THREADS=${Q38FN_STAGE_THREADS:-4}
export Q38FN_UTOFU_OWNER_CREDITS=${Q38FN_UTOFU_OWNER_CREDITS:-2}
export Q38FN_UTOFU_SERVICE_THREADS=${Q38FN_UTOFU_SERVICE_THREADS:-2}
export Q38FN_UTOFU_IMPLICIT_ACK=${Q38FN_UTOFU_IMPLICIT_ACK:-1}
export Q38FN_PIPE_ITERS=${Q38FN_PIPE_ITERS:-1000}
export Q38FN_PIPE_WORKERS=${Q38FN_PIPE_WORKERS:-8}
export Q38FN_PIPE_WINDOW=${Q38FN_PIPE_WINDOW:-8}
export Q38FN_RESULT_PREFIX="$WORK/result"

mkdir -p "$WORK" || exit 2
cd "$REPO" || exit 2
echo "=== Q38FN n-gram 4-node resident probe job=$JOB np=$NP ==="
echo "model=$MODEL stage=$STAGE work=$WORK"
date

make -C a64fx/utofu-tests tofu_topo_helper MPICC=mpifcc \
    MPICFLAGS="-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall" >/dev/null || exit 3
make -C q38fn ngram_stage_mpi utofu_pipeline_probe_mpi MPICC=mpifcc CC=fcc \
    CFLAGS="-Nclang -O2 -Wall -Wextra -Wpedantic" >/dev/null || exit 3

cd "$WORK" || exit 2
rm -f "$TOPO"
mpiexec -np "$NP" "$REPO/a64fx/utofu-tests/tofu_topo_helper" || exit 4
test "$(grep -vc '^#' "$TOPO" 2>/dev/null || echo 0)" -ge "$NP" || exit 4

echo "--- staging $(date) ---"
mpiexec -np "$NP" -of-proc "$WORK/stage.rank" \
    "$Q38FN/ngram_stage_mpi" "$MODEL" "$STAGE" "$NP" || exit 5

echo "--- resident pipeline $(date) ---"
mpiexec -np "$NP" -of-proc "$WORK/pipe.rank" \
    "$Q38FN/utofu_pipeline_probe_mpi" "$MODEL" "$TOPO" "$NP" \
    "$Q38FN_PIPE_ITERS" "$Q38FN_PIPE_WORKERS" "$Q38FN_PIPE_WINDOW" resident || exit 6

echo "=== result files ==="
for f in "$WORK"/result.*; do test -f "$f" && { echo "--- $f"; tail -8 "$f"; }; done
echo "=== Q38FN n-gram probe done $(date) ==="
