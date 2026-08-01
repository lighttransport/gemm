#!/bin/bash
# GLM-5.2 12-node continuous-batch generation smoke. Partial-layer output is
# for scheduler/perf plumbing only; meaningful text requires the 192-node full run.

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=1x12:torus,elapse=01:10:00"
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
STAGE_LAYERS=${GLM5_STAGE_LAYERS:-12}
RUN_LAYERS=${GLM5_LAYERS:-$STAGE_LAYERS}
PROMPTS=${GLM5_CBATCH_PROMPTS:-$GLM5/cbatch_agentic_prompts.txt}
WORK="$GLM5/cbatch_run_${PJM_JOBID:-manual}_12n"
mkdir -p "$WORK" || exit 2
cd "$WORK" || exit 2

export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm5.2}
export GLM5_STAGE_DIR=/local/glm5
export GLM5_NSHARDS=282
export GLM5_EP_SIZE=$NP
export GLM5_STATUS_DIR="$WORK"
export GLM5_TP=${GLM5_TP:-1}
export GLM5_MSA=${GLM5_MSA:-0}
export GLM5_MAXPOS=${GLM5_MAXPOS:-192}
export GLM5_MAX_NEW=${GLM5_MAX_NEW:-32}
export GLM5_CBATCH_SLOTS=${GLM5_CBATCH_SLOTS:-4}
export LLM_THREADS=${LLM_THREADS:-12}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$LLM_THREADS}

echo "=== GLM5.2 cbatch 12n: NP=$NP stage_layers=$STAGE_LAYERS run_layers=$RUN_LAYERS slots=$GLM5_CBATCH_SLOTS max_new=$GLM5_MAX_NEW job=${PJM_JOBID:-?} ==="
echo "workdir=$WORK"
date

"$GLM5/check_glm5_model.sh" "$GLM5_MODEL_DIR" --tokenizer || exit 2
test -s "$PROMPTS" || { echo "FATAL: missing $PROMPTS"; exit 2; }

rm -f tofu_topo.txt glm5_stage_rank*.txt glm5_ep_*.txt glm5_ep_rank00.txt glm5_cbatch_gen_*.txt
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

topo_ok=0
for t in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then
        topo_ok=1
        break
    fi
    echo "[cbatch12] topo try $t failed; retry"
    sleep 3
done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed"; exit 3; }

echo "--- staging ($(date)) ---"
GLM5_STAGE_LAYERS=$STAGE_LAYERS mpiexec -np "$NP" "$LLM/build/glm5_stage" || { echo "FATAL: stage"; exit 4; }
echo "staged ranks: $(ls "$WORK"/glm5_stage_rank*.txt 2>/dev/null | wc -l)/$NP ($(date))"

echo "--- continuous batch generate ($(date)) ---"
GLM5_REAL=1 GLM5_LAYERS=$RUN_LAYERS \
  GLM5_CBATCH_PROMPTS="$PROMPTS" GLM5_CBATCH_OUT_PREFIX="$WORK/glm5_cbatch_gen" \
  mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: cbatch"; exit 5; }

echo "--- rank0 log ---"
cat glm5_ep_rank00.txt 2>/dev/null
echo "--- decoded outputs ---"
for f in glm5_cbatch_gen_*.txt; do
    [ -s "$f" ] || continue
    echo "### $f"
    python3 "$GLM5/glm5_tokenizer.py" decode-file "$f" 2>/dev/null || true
done
echo "=== cbatch12 done $(date) ==="
