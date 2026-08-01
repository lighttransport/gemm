#!/bin/bash
# Fast synthetic continuous-batch decode test for the GLM-5.2 runner. This avoids
# weight staging and validates request scheduling, runtime clones, and profiling.

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=1,elapse=01:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=1"
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"
UTOFU="$REPO/a64fx/utofu-tests"
GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-1}
PROMPTS=${GLM5_CBATCH_PROMPTS:-$GLM5/cbatch_agentic_prompts.txt}
WORK="$GLM5/cbatch_run_${PJM_JOBID:-manual}_synth1n"
mkdir -p "$WORK" || exit 2
cd "$WORK" || exit 2
export GLM5_LAYERS=${GLM5_LAYERS:-4}
export GLM5_EXPERTS=${GLM5_EXPERTS:-8}
export GLM5_MAXPOS=${GLM5_MAXPOS:-128}
export GLM5_MAX_NEW=${GLM5_MAX_NEW:-16}
export GLM5_CBATCH_SLOTS=${GLM5_CBATCH_SLOTS:-2}
export LLM_THREADS=${LLM_THREADS:-12}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$LLM_THREADS}

echo "=== GLM5.2 synthetic cbatch 1n: layers=$GLM5_LAYERS experts=$GLM5_EXPERTS slots=$GLM5_CBATCH_SLOTS max_new=$GLM5_MAX_NEW job=${PJM_JOBID:-?} ==="
echo "workdir=$WORK"
date
test -s "$PROMPTS" || { echo "FATAL: missing $PROMPTS"; exit 2; }

rm -f tofu_topo.txt glm5_ep_*.txt glm5_ep_rank00.txt glm5_cbatch_synth_*.txt
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" || exit 3
GLM5_REAL=0 GLM5_CBATCH_PROMPTS="$PROMPTS" GLM5_CBATCH_OUT_PREFIX="$WORK/glm5_cbatch_synth" \
  mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: synthetic cbatch"; exit 5; }

echo "--- rank0 log ---"
cat glm5_ep_rank00.txt 2>/dev/null
echo "=== synthetic cbatch done $(date) ==="
