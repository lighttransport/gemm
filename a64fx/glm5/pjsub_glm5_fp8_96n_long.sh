#!/bin/bash
# GLM-5.2 FP8 long-context generation validation on 96 A64FX nodes.
# Builds a 1K+ token prompt from prompt_fp8_1k.txt, requests 1024 output tokens,
# and reports prefill/decode throughput plus decoded text.

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
JOB_TAG=${PJM_JOBID:-manual_$$}
RUN_DIR="$GLM5/output.$JOB_TAG"

export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm52-fp8}
export GLM5_STAGE_DIR=${GLM5_STAGE_DIR:-/local/glm5_$JOB_TAG}
export GLM5_NSHARDS=141
export GLM5_EP_SIZE=$NP
export GLM5_STATUS_DIR="$RUN_DIR"
export GLM5_TP=${GLM5_TP:-1}
export GLM5_TP_SHARED=${GLM5_TP_SHARED:-0}
export GLM5_MSA=${GLM5_MSA:-0}
export GLM5_MAXPOS=${GLM5_MAXPOS:-2304}
export GLM5_MAX_NEW=${GLM5_MAX_NEW:-1024}
export GLM5_MIN_NEW=${GLM5_MIN_NEW:-1024}
export GLM5_PCHUNK=${GLM5_PCHUNK:-64}
export GLM5_TOKENIZER=${GLM5_TOKENIZER:-$GLM5_MODEL_DIR/tokenizer.json}
export LLM_THREADS=${LLM_THREADS:-12}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$LLM_THREADS}

echo "=== GLM5.2 FP8 long gen 96n: NP=$NP maxpos=$GLM5_MAXPOS max_new=$GLM5_MAX_NEW model=$GLM5_MODEL_DIR job=${PJM_JOBID:-?} ==="
date

"$GLM5/check_glm5_model.sh" "$GLM5_MODEL_DIR" --tokenizer || exit 2
test -s "$GLM5/prompt_fp8_1k.txt" || { echo "FATAL: missing prompt_fp8_1k.txt"; exit 2; }

mkdir -p "$RUN_DIR" || exit 2
cd "$RUN_DIR" || exit 2
rm -f tofu_topo.txt glm5_stage_rank*.txt glm5_ep_*.txt glm5_ep_rank00.txt gen_ids.txt prompt_ids_long.txt
python3 "$GLM5/glm5_tokenizer.py" chat-file "$GLM5/prompt_fp8_1k.txt" > prompt_ids_long.txt || exit 2
PROMPT_TOK=$(wc -w < prompt_ids_long.txt)
export GLM5_PROMPT_IDS="$RUN_DIR/prompt_ids_long.txt"
export GLM5_GEN_OUT="$RUN_DIR/gen_ids.txt"
echo "prompt tokens: $PROMPT_TOK"
[ "$PROMPT_TOK" -ge 1024 ] || { echo "FATAL: prompt is below 1024 tokens"; exit 2; }

make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

topo_ok=0
for t in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then
        topo_ok=1
        break
    fi
    echo "[fp8-long] topo try $t failed; retry"
    sleep 3
done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed"; exit 3; }
echo "topo OK ($(wc -l < tofu_topo.txt) rows)"

echo "--- staging FP8 model -> $GLM5_STAGE_DIR ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/glm5_stage" || { echo "FATAL: stage failed"; exit 4; }
echo "--- stage status ($(date)) ---"
cat "$RUN_DIR"/glm5_stage_rank*.txt 2>/dev/null | sort | tail -12
echo "staged ranks: $(ls "$RUN_DIR"/glm5_stage_rank*.txt 2>/dev/null | wc -l)/$NP"

echo "--- run long GLM5_REAL=1 FP8 ($(date)) ---"
GLM5_REAL=1 mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: run failed"; exit 5; }
echo "--- rank0 log ---"
cat "$RUN_DIR"/glm5_ep_rank00.txt 2>/dev/null
echo "--- gen ids count ---"
wc -w "$GLM5_GEN_OUT" 2>/dev/null || true
echo "--- decoded head ---"
python3 "$GLM5/glm5_tokenizer.py" decode-file "$GLM5_GEN_OUT" 2>/dev/null | sed -n '1,80p' || true
echo "=== fp8 long 96n done $(date) ==="
