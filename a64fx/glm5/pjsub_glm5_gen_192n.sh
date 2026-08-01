#!/bin/bash
# GLM-5.2 (text) REAL-WEIGHT GENERATION on 192 A64FX nodes, 16x12 torus. Stage the
# full model to /local/glm5, then run GLM5_REAL=1 in gen-mode: embed the prompt token ids
# (a64fx/glm5/prompt_ids.txt), prefill, greedy-decode GLM5_MAX_NEW tokens, write the
# generated ids to a64fx/glm5/gen_ids.txt. Detokenize offline:
#   python3 a64fx/glm5/glm5_tokenizer.py decode-file a64fx/glm5/gen_ids.txt
#
# Submit: ssh fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory a64fx/glm5/pjsub_glm5_gen_192n.sh'

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=16x12:torus,elapse=03:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=192"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
cd "$GLM5" || exit 2
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-192}
STAGE_LAYERS=${GLM5_STAGE_LAYERS:-0}; RUN_LAYERS=${GLM5_LAYERS:-0}
export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm5.2} GLM5_STAGE_DIR=/local/glm5
export GLM5_NSHARDS=282 GLM5_EP_SIZE=$NP GLM5_STATUS_DIR="$GLM5"
export GLM5_TP=${GLM5_TP:-1} GLM5_MSA=${GLM5_MSA:-0}
export GLM5_MAXPOS=${GLM5_MAXPOS:-256} GLM5_MAX_NEW=${GLM5_MAX_NEW:-48}
export LLM_THREADS=${LLM_THREADS:-12} OMP_NUM_THREADS=${LLM_THREADS:-12}   # 12 = 1 CMG (sweet spot)

echo "=== GLM5.2 REAL gen 192n (16x12): NP=$NP layers=$RUN_LAYERS prompt=$(cat "$GLM5/prompt_ids.txt") job=${PJM_JOBID:-?} ==="
date
"$GLM5/check_glm5_model.sh" "$GLM5_MODEL_DIR" --tokenizer || exit 2
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

topo_ok=0
for t in 1 2 3 4 5; do rm -f tofu_topo.txt
    if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(wc -l < tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then topo_ok=1; break; fi
    echo "[gen192] topo try $t failed (1907?); retry"; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed"; exit 3; }

echo "--- staging ($(date)) ---"
GLM5_STAGE_LAYERS=$STAGE_LAYERS mpiexec -np "$NP" "$LLM/build/glm5_stage" || { echo "FATAL: stage"; exit 4; }
echo "staged ranks: $(ls "$GLM5"/glm5_stage_rank*.txt 2>/dev/null | wc -l)/$NP ($(date))"

echo "--- generate ($(date)) ---"
GLM5_REAL=1 GLM5_LAYERS=$RUN_LAYERS \
  GLM5_PROMPT_IDS="$GLM5/prompt_ids.txt" GLM5_GEN_OUT="$GLM5/gen_ids.txt" \
  mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: gen"; exit 5; }
echo "--- rank0 log ---"; cat glm5_ep_rank00.txt 2>/dev/null
echo "--- gen ids ---"; cat "$GLM5/gen_ids.txt" 2>/dev/null
echo "=== done $(date) ==="
