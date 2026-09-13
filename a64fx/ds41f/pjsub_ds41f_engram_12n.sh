#!/bin/bash
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=12,elapse=06:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=12"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j

set -eu

REPO=${DS41F_REPO:-$HOME/work/gemm/glm53f}
MODEL=${DS41F_MODEL:-$HOME/models/ds41f}
JOB=${PJM_JOBID:-manual_$$}
STAGE=/local/$USER/ds41f-engram-$JOB
RUN=$REPO/a64fx/ds41f/runs/$JOB
NP=${PJM_MPI_PROC:-12}

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
mkdir -p "$RUN"
cd "$REPO"

make -C a64fx/utofu-tests tofu_topo_helper MPICC=mpifcc \
  MPICFLAGS="-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall"
make -C a64fx/ds41f ds41f_stage A64FX_CC=fcc \
  CFLAGS="-Nclang -O2 -Wall -Wextra -Wpedantic -std=c11"
make -C a64fx/ds41f ds41f_bench_utofu A64FX_CC=fcc

echo "=== DS41F Engram 12-node job=$JOB model=$MODEL stage=$STAGE ==="
date

rm -f "$RUN/tofu_topo.txt"
cd "$RUN"
for try in $(seq 1 40); do
  if mpiexec -np "$NP" "$REPO/a64fx/utofu-tests/tofu_topo_helper" \
       >"$RUN/topo.$try.log" 2>&1 &&
     [ "$(grep -vc '^#' "$RUN/tofu_topo.txt" 2>/dev/null || echo 0)" -ge "$NP" ]; then
    break
  fi
  sleep 12
done
test "$(grep -vc '^#' "$RUN/tofu_topo.txt" 2>/dev/null || echo 0)" -ge "$NP"

cd "$REPO"

echo "--- staging ---"
if [ -n "${DS41F_META_SCRIPT:-}" ]; then
  mpiexec -np "$NP" python3 "$REPO/a64fx/ds41f/make_token_map.py" \
    --tokenizer "$MODEL/tokenizer.json" --output "$STAGE/engram_meta.bin"
else
  echo "DS41F_META_SCRIPT not set; storage/uTofu benchmark skips hash-ID validation"
fi
mpiexec -np "$NP" -of-proc "$RUN/stage.rank" \
  "$REPO/a64fx/ds41f/ds41f_stage" --model-dir "$MODEL" \
  --stage-dir "$STAGE" --ranks "$NP"

echo "--- local owner benchmark ---"
DS41F_ITERS=${DS41F_ITERS:-10000} DS41F_WORKERS=${DS41F_WORKERS:-8}
for mode in local remote; do
  args=(--stage-dir "$STAGE" --ranks "$NP" --iters "$DS41F_ITERS" \
        --workers "$DS41F_WORKERS")
  [ "$mode" = remote ] && args+=(--remote --topology "$RUN/tofu_topo.txt")
  mpiexec -np "$NP" -of-proc "$RUN/$mode.rank" \
    "$REPO/a64fx/ds41f/ds41f_bench_utofu" "${args[@]}"
done

echo "=== DS41F Engram done job=$JOB ==="
date
