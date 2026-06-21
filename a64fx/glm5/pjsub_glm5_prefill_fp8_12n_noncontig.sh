#!/bin/bash
# GLM-5.2 FP8 full-layer chunked-prefill benchmark on 12 non-contiguous
# A64FX nodes. This usually queues faster than torus placement but may pay
# higher uTofu communication cost.

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=12,elapse=03:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=12"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"
UTOFU="$REPO/a64fx/utofu-tests"
GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-12}
STAGE_LAYERS=${GLM5_STAGE_LAYERS:-0}
RUN_LAYERS=${GLM5_LAYERS:-0}
PREFILL_SYNTH=${GLM5_PREFILL_SYNTH:-1024}
PCHUNKS=${GLM5_PCHUNK_SWEEP:-64}
THREADS=${GLM5_THREAD_SWEEP:-12}
JOB_TAG=${PJM_JOBID:-manual_$$}
WORK="$GLM5/prefill_fp8_run_${JOB_TAG}_12n_noncontig"

export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm52-fp8}
export GLM5_STAGE_DIR=${GLM5_STAGE_DIR:-/local/glm5_fp8_prefill_$JOB_TAG}
export GLM5_NSHARDS=141
export GLM5_EP_SIZE=$NP
export GLM5_STATUS_DIR="$WORK"
export GLM5_TP=${GLM5_TP:-1}
export GLM5_TP_SHARED=${GLM5_TP_SHARED:-0}
export GLM5_MSA=${GLM5_MSA:-1}
export GLM5_MSA_BLOCK_REP=${GLM5_MSA_BLOCK_REP:-1}
export GLM5_CP=${GLM5_CP:-1}
export GLM5_INT4_KV=${GLM5_INT4_KV:-1}
export GLM5_MAXPOS=${GLM5_MAXPOS:-1048576}
export GLM5_DENSE_ATTN_WINDOW=${GLM5_DENSE_ATTN_WINDOW:-4096}
export GLM5_SPARSE_ATTN_WINDOW=${GLM5_SPARSE_ATTN_WINDOW:-0}
export GLM5_PREFILL_ONLY=1
export GLM5_PREFILL_SYNTH=$PREFILL_SYNTH
export GLM5_PREFILL_ROLLING=${GLM5_PREFILL_ROLLING:-1}
export GLM5_ABSORB_ATTN=${GLM5_ABSORB_ATTN:-1}
export GLM5_ABSORB_SVE_DOT=${GLM5_ABSORB_SVE_DOT:-1}
export GLM5_PCHUNK=${GLM5_PCHUNK:-64}
export GLM5_TOKENIZER=${GLM5_TOKENIZER:-$GLM5_MODEL_DIR/tokenizer.json}

echo "=== GLM5.2 FP8 prefill 12n noncontig: NP=$NP layers=$RUN_LAYERS synth=$PREFILL_SYNTH maxpos=$GLM5_MAXPOS model=$GLM5_MODEL_DIR job=${PJM_JOBID:-?} ==="
echo "workdir=$WORK stage_dir=$GLM5_STAGE_DIR chunks=[$PCHUNKS] threads=[$THREADS] cp=$GLM5_CP msa=$GLM5_MSA int4=$GLM5_INT4_KV tp=$GLM5_TP tp_shared=$GLM5_TP_SHARED ar_tokens=${GLM5_AR_TOKENS:-auto} tp_ar_bf16=${TP_AR_BF16:-0}"
date

"$GLM5/check_glm5_model.sh" "$GLM5_MODEL_DIR" --tokenizer || exit 2
mkdir -p "$WORK" || exit 2
cd "$WORK" || exit 2
rm -f tofu_topo.txt glm5_stage_rank*.txt glm5_ep_*.txt glm5_ep_rank00.txt

make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null || exit 3

topo_ok=0
for t in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then
        topo_ok=1
        break
    fi
    echo "[prefill-fp8-12-noncontig] topo try $t failed; retry"
    sleep 3
done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed"; exit 3; }
echo "topo OK ($(grep -vc '^#' tofu_topo.txt 2>/dev/null || echo 0) ranks)"
sed -n '1,40p' tofu_topo.txt
if [ "${GLM5_TOPO_ONLY:-0}" = 1 ]; then
    echo "=== prefill-fp8-12-noncontig topo-only done $(date) ==="
    exit 0
fi

echo "--- staging FP8 model -> $GLM5_STAGE_DIR ($(date)) ---"
GLM5_STAGE_LAYERS=$STAGE_LAYERS mpiexec -np "$NP" "$LLM/build/glm5_stage" || { echo "FATAL: stage"; exit 4; }
echo "staged ranks: $(ls "$WORK"/glm5_stage_rank*.txt 2>/dev/null | wc -l)/$NP ($(date))"
cat "$WORK"/glm5_stage_rank*.txt 2>/dev/null | sort | tail -12

for th in $THREADS; do
    for pc in $PCHUNKS; do
        echo "--- prefill FP8 run threads=$th pchunk=$pc ($(date)) ---"
        rm -f glm5_ep_*.txt glm5_ep_rank00.txt
        LLM_THREADS=$th OMP_NUM_THREADS=$th GLM5_PCHUNK=$pc GLM5_REAL=1 GLM5_LAYERS=$RUN_LAYERS \
          mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: prefill threads=$th pchunk=$pc"; exit 5; }
        cp glm5_ep_rank00.txt "prefill_rank00_t${th}_pc${pc}.txt" 2>/dev/null || true
        grep -E "prefill_synth:|prefill_progress:|PROFILE prefill_synth|SENTINEL" "prefill_rank00_t${th}_pc${pc}.txt" 2>/dev/null || true
    done
done

echo "=== prefill-fp8-12 noncontig done $(date) ==="
