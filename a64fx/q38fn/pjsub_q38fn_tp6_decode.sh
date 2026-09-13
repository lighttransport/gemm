#!/bin/bash
# Top-level PJM TP6 decode benchmark.  Submit this script with pjsub; do not
# invoke it through bash-over-HTTP, whose shell is already a plexec rank.
#
#   pjsub --no-check-directory a64fx/q38fn/pjsub_q38fn_tp6_decode.sh

#PJM -g hp250467
#PJM -L "rscgrp=small,node=6,elapse=12:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=6"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j

set -u
REPO=${Q38FN_REPO:-$HOME/work/gemm/glm53f}
MODEL=${Q38FN_MODEL:-$HOME/models/q38fn/bf16}
JOB=${PJM_JOBID:-manual_$$}
WORK="$REPO/q38fn/runs/tp6_decode_${JOB}"
STAGE=${Q38FN_STAGE:-/local/$USER/q38fn-tp6-${JOB}}
TOPO="$WORK/tofu_topo.txt"
NP=${PJM_MPI_PROC:-6}

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
mkdir -p "$WORK" || exit 2
cd "$REPO" || exit 2

make -C a64fx/utofu-tests tofu_topo_helper MPICC=mpifcc \
    MPICFLAGS="-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast" || exit 3
make -C q38fn q38fn_tp6_stage_mpi q38fn_tp6_runner_utofu MPICC=mpifcc CC=fcc \
    CFLAGS="-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp" || exit 3

cd "$WORK" || exit 2
rm -f "$TOPO"
mpiexec -np "$NP" "$REPO/a64fx/utofu-tests/tofu_topo_helper" || exit 4
test "$(grep -vc '^#' "$TOPO" 2>/dev/null || echo 0)" -ge "$NP" || exit 4

echo "=== TP6 stage job=$JOB stage=$STAGE ==="
cd "$REPO" || exit 2
Q38FN_TP_STAGE_LAYERS=48 Q38FN_TP_STAGE_Q8=${Q38FN_TP_STAGE_Q8:-1} \
  mpiexec -np "$NP" -of-proc "$WORK/stage.rank" \
  "$REPO/q38fn/q38fn_tp6_stage_mpi" "$MODEL" "$STAGE" || exit 5

echo "=== TP6 decode loader=${Q38FN_TP_FILE_BACKED:-resident} ==="
export TOFU_TOPO_PATH="$TOPO"
export Q38FN_TP_GROUP_DOWN=${Q38FN_TP_GROUP_DOWN:-1}
export Q38FN_TP_PROFILE=${Q38FN_TP_PROFILE:-1}
export Q38FN_TP_TRACE_DIR=${Q38FN_TP_TRACE_DIR:-$STAGE/trace}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-39}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}
mpiexec -np "$NP" -of-proc "$WORK/decode.rank" \
  "$REPO/q38fn/q38fn_tp6_runner_utofu" "$MODEL" --local-base "$STAGE" \
  --prompt "${Q38FN_PROMPT:-Write a complete C11 function that computes Fibonacci numbers iteratively and returns uint64_t. Return only code.}" \
  --max-gen "${Q38FN_MAX_GEN:-32}" --max-seq "${Q38FN_MAX_SEQ:-256}" || exit 6

echo "=== TP6 result files ==="
for f in "$WORK"/decode.rank.*; do test -f "$f" && { echo "--- $f"; tail -20 "$f"; }; done
