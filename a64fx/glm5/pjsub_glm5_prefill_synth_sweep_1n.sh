#!/bin/bash
# GLM-5.2 1-node synthetic-token prefill sweep. Keeps the synthetic model small
# and sweeps chunk/thread settings to expose prefill amortization limits quickly.

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=1,elapse=01:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=1"
#PJM --llio localtmp-size=10Gi
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"
UTOFU="$REPO/a64fx/utofu-tests"
GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

WORK="$GLM5/prefill_sweep_${PJM_JOBID:-manual}_synth1n"
mkdir -p "$WORK" || exit 2
cd "$WORK" || exit 2

export GLM5_REAL=0
export GLM5_LAYERS=${GLM5_LAYERS:-4}
export GLM5_EXPERTS=${GLM5_EXPERTS:-8}
export GLM5_EP_SIZE=1
export GLM5_STATUS_DIR="$WORK"
export GLM5_MAXPOS=${GLM5_MAXPOS:-512}
export GLM5_PREFILL_SYNTH=${GLM5_PREFILL_SYNTH:-256}
export GLM5_PREFILL_ONLY=1
export GLM5_PREFILL_ROLLING=${GLM5_PREFILL_ROLLING:-0}
export GLM5_MSA=${GLM5_MSA:-0}

PCHUNKS=${GLM5_PCHUNK_SWEEP:-16 32 64 128}
THREADS=${GLM5_THREAD_SWEEP:-48}

echo "=== GLM5.2 synthetic prefill sweep 1n: layers=$GLM5_LAYERS experts=$GLM5_EXPERTS synth=$GLM5_PREFILL_SYNTH maxpos=$GLM5_MAXPOS job=${PJM_JOBID:-?} ==="
echo "workdir=$WORK chunks=[$PCHUNKS] threads=[$THREADS]"
date

make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

for th in $THREADS; do
    for pc in $PCHUNKS; do
        echo "--- synthetic prefill threads=$th pchunk=$pc ($(date)) ---"
        rm -f tofu_topo.txt glm5_ep_*.txt glm5_ep_rank00.txt
        mpiexec -np 1 "$UTOFU/tofu_topo_helper" || { echo "FATAL: topo threads=$th pchunk=$pc"; exit 4; }
        LLM_THREADS=$th OMP_NUM_THREADS=$th GLM5_PCHUNK=$pc \
          mpiexec -np 1 "$LLM/build/glm5_ep_runner" || { echo "FATAL: threads=$th pchunk=$pc"; exit 5; }
        cp glm5_ep_rank00.txt "prefill_rank00_t${th}_pc${pc}.txt" 2>/dev/null || true
        grep -E "prefill_synth:|PROFILE prefill_synth|SENTINEL" "prefill_rank00_t${th}_pc${pc}.txt" 2>/dev/null || true
    done
done

echo "=== synthetic prefill sweep done $(date) ==="
