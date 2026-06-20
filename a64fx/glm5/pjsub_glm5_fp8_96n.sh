#!/bin/bash
# GLM-5.2 FP8 real-weight run on 96 A64FX nodes, 8x12 torus.
# Stages ~/models/glm52-fp8 (141 safetensor shards) to /local/glm5,
# then runs the GLM5_REAL=1 EP+TP forward with synthetic activations.

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=8x12:torus,elapse=03:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"
UTOFU="$REPO/a64fx/utofu-tests"
GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-96}
STAGE_LAYERS=${GLM5_STAGE_LAYERS:-0}      # 0 = full 78 main layers
RUN_LAYERS=${GLM5_LAYERS:-0}
JOB_TAG=${PJM_JOBID:-manual_$$}
RUN_DIR="$GLM5/output.$JOB_TAG"

export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm52-fp8}
export GLM5_STAGE_DIR=${GLM5_STAGE_DIR:-/local/glm5_$JOB_TAG}
export GLM5_NSHARDS=141
export GLM5_EP_SIZE=$NP
export GLM5_STATUS_DIR="$RUN_DIR"
export GLM5_TP=${GLM5_TP:-1}
# 96-way sharding gives 21/22 shared-expert columns per rank; MXFP8 scale blocks
# are 32 columns, so keep shared experts replicated unless explicitly overridden.
export GLM5_TP_SHARED=${GLM5_TP_SHARED:-0}
export GLM5_MSA=${GLM5_MSA:-0}
export GLM5_MAXPOS=${GLM5_MAXPOS:-512}
export GLM5_PREFILL=${GLM5_PREFILL:-4}
export GLM5_DECODE=${GLM5_DECODE:-8}
export GLM5_PROMPT_IDS=${GLM5_PROMPT_IDS:-$GLM5/prompt_ids_fp8_short.txt}
export GLM5_GEN_OUT=${GLM5_GEN_OUT:-$RUN_DIR/gen_ids.txt}
export GLM5_MAX_NEW=${GLM5_MAX_NEW:-8}
export GLM5_TOKENIZER=${GLM5_TOKENIZER:-$GLM5_MODEL_DIR/tokenizer.json}
export LLM_THREADS=${LLM_THREADS:-12}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$LLM_THREADS}

echo "=== GLM5.2 FP8 real 96n: NP=$NP stage_layers=$STAGE_LAYERS run_layers=$RUN_LAYERS model=$GLM5_MODEL_DIR job=${PJM_JOBID:-?} ==="
date

"$GLM5/check_glm5_model.sh" "$GLM5_MODEL_DIR" --tokenizer || exit 2
test -s "$GLM5_PROMPT_IDS" || { echo "FATAL: missing GLM5_PROMPT_IDS=$GLM5_PROMPT_IDS"; exit 2; }

mkdir -p "$RUN_DIR" || exit 2
cd "$RUN_DIR" || exit 2
rm -f tofu_topo.txt glm5_stage_rank*.txt glm5_ep_*.txt glm5_ep_rank00.txt
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

topo_ok=0
for t in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then
        topo_ok=1
        break
    fi
    echo "[fp8-96] topo try $t failed; retry"
    sleep 3
done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed"; exit 3; }
echo "topo OK ($(wc -l < tofu_topo.txt) rows)"

echo "--- staging FP8 model -> $GLM5_STAGE_DIR ($(date)) ---"
GLM5_STAGE_LAYERS=$STAGE_LAYERS mpiexec -np "$NP" "$LLM/build/glm5_stage" || { echo "FATAL: stage failed"; exit 4; }
echo "--- stage status ($(date)) ---"
cat "$RUN_DIR"/glm5_stage_rank*.txt 2>/dev/null | sort | tail -12
echo "staged ranks: $(ls "$RUN_DIR"/glm5_stage_rank*.txt 2>/dev/null | wc -l)/$NP"

echo "--- run GLM5_REAL=1 FP8 ($(date)) ---"
GLM5_REAL=1 GLM5_LAYERS=$RUN_LAYERS mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: run failed"; exit 5; }
echo "--- rank0 log ---"
cat "$RUN_DIR"/glm5_ep_rank00.txt 2>/dev/null
echo "--- gen ids ---"
cat "$GLM5_GEN_OUT" 2>/dev/null || true
echo "--- decoded ---"
python3 "$GLM5/glm5_tokenizer.py" decode-file "$GLM5_GEN_OUT" 2>/dev/null || true
echo "=== fp8 96n done $(date) ==="
