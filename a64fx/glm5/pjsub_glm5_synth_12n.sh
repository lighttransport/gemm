#!/bin/bash
# GLM-5.2 synthetic 12-node EP/TP smoke. Does not read model weights; exercises
# uTofu topology, TP collectives, EP routing, and short forward lockstep.

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=1x12:torus,elapse=01:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=12"
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"
UTOFU="$REPO/a64fx/utofu-tests"
GLM5="$REPO/a64fx/glm5"
cd "$GLM5" || exit 2
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-12}
export GLM5_TP=${GLM5_TP:-1}
export GLM5_MSA=${GLM5_MSA:-0}
export GLM5_LAYERS=${GLM5_LAYERS:-4}
export GLM5_MAXPOS=${GLM5_MAXPOS:-64}
export GLM5_PREFILL=${GLM5_PREFILL:-2}
export GLM5_DECODE=${GLM5_DECODE:-2}
export LLM_THREADS=${LLM_THREADS:-12}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$LLM_THREADS}

echo "=== GLM5.2 synthetic 12n: NP=$NP layers=$GLM5_LAYERS maxpos=$GLM5_MAXPOS job=${PJM_JOBID:-?} ==="
date

rm -f tofu_topo.txt glm5_ep_*.txt glm5_ep_rank00.txt
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

topo_ok=0
for t in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then
        topo_ok=1
        break
    fi
    echo "[synth12] topo try $t failed; retry"
    sleep 3
done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed"; exit 4; }

GLM5_REAL=0 mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || exit 5

echo "--- rank0 log ---"
cat glm5_ep_rank00.txt 2>/dev/null
echo "=== synth12 done $(date) ==="
