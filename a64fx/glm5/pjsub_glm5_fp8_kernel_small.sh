#!/bin/bash
# Small-node GLM-5.2 FP8 real-weight kernel benchmark.
# Submit with command-line resources, for example:
#   pjsub -L "rscgrp=small-s2,node=4,elapse=01:30:00" --mpi "proc=4" pjsub_glm5_fp8_kernel_small.sh

#PJM -g hp250467
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"
UTOFU="$REPO/a64fx/utofu-tests"
GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-1}
JOB_TAG=${PJM_JOBID:-manual_$$}
RUN_DIR="$GLM5/kernel_small_${JOB_TAG}_${NP}n_fp8${GLM5_FP8_8ROW:-1}"

export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm52-fp8}
export GLM5_STAGE_DIR=${GLM5_STAGE_DIR:-/local/glm5_kernel_${JOB_TAG}}
export GLM5_NSHARDS=${GLM5_NSHARDS:-141}
export GLM5_STAGE_LAYERS=${GLM5_STAGE_LAYERS:-4}
export GLM5_LAYERS=${GLM5_LAYERS:-$GLM5_STAGE_LAYERS}
export GLM5_EP_SIZE=$NP
export GLM5_STATUS_DIR="$RUN_DIR"
export GLM5_TP=${GLM5_TP:-1}
export GLM5_TP_SHARED=${GLM5_TP_SHARED:-0}
export GLM5_MSA=${GLM5_MSA:-0}
export GLM5_MAXPOS=${GLM5_MAXPOS:-256}
export GLM5_PREFILL=${GLM5_PREFILL:-16}
export GLM5_DECODE=${GLM5_DECODE:-8}
export GLM5_FP8_8ROW=${GLM5_FP8_8ROW:-1}
export LLM_THREADS=${LLM_THREADS:-12}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$LLM_THREADS}

echo "=== GLM5.2 FP8 kernel small: NP=$NP layers=$GLM5_LAYERS stage_layers=$GLM5_STAGE_LAYERS fp8_8row=$GLM5_FP8_8ROW job=${PJM_JOBID:-?} ==="
date

"$GLM5/check_glm5_model.sh" "$GLM5_MODEL_DIR" --tokenizer || exit 2

mkdir -p "$RUN_DIR" || exit 2
cd "$RUN_DIR" || exit 2
rm -f tofu_topo.txt glm5_stage_rank*.txt glm5_ep_*.txt glm5_ep_rank00.txt
make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null || exit 3

topo_ok=0
for t in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then
        topo_ok=1
        break
    fi
    echo "[kernel-small] topo try $t failed; retry"
    sleep 3
done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed"; exit 3; }
echo "topo OK ($(grep -vc '^#' tofu_topo.txt 2>/dev/null || echo 0) ranks)"

echo "--- staging FP8 model -> $GLM5_STAGE_DIR ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/glm5_stage" || { echo "FATAL: stage failed"; exit 4; }
echo "staged ranks: $(ls "$RUN_DIR"/glm5_stage_rank*.txt 2>/dev/null | wc -l)/$NP"
cat "$RUN_DIR"/glm5_stage_rank*.txt 2>/dev/null | sort | tail -12

echo "--- run GLM5_REAL=1 synthetic prefill/decode ($(date)) ---"
GLM5_REAL=1 mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: run failed"; exit 5; }
echo "--- rank0 log ---"
cat "$RUN_DIR"/glm5_ep_rank00.txt 2>/dev/null
echo "=== kernel small done $(date) ==="
