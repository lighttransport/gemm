#!/bin/bash
# FP8-mitigation eval: prefill a short prompt + greedy-generate, ONE runner call, single config.
# Run 3x (bf16 / FP8 / mixed) with the same prompt; diff the gen.ids to measure how close each
# sits to the bf16 reference. Config via env:
#   GLM5_MODEL_DIR (glm5.2=bf16 | glm52-fp8=FP8), GLM5_NSHARDS (282|141),
#   GLM5_BF16_LAYERS (optional, e.g. "0-2,75-77" -> mixed precision from GLM5_BF16_DIR).
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=96,elapse=01:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-96}
JOB_TAG=${PJM_JOBID:-manual_$$}; WORK="$GLM5/eval_run_${JOB_TAG}"
TOK=${GLM5_PROMPT_TOKENS:-$HOME/glm5_eval_fact.bin}
NPROMPT=${GLM5_NPROMPT:-25}; GEN=${GLM5_GEN_NEW:-48}
export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm52-fp8}
export GLM5_STAGE_DIR=${GLM5_STAGE_DIR:-/local/glm5_eval_$JOB_TAG}
export GLM5_NSHARDS=${GLM5_NSHARDS:-141}
export GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1 GLM5_STATUS_DIR="$WORK"
export GLM5_TP=1 GLM5_TP_SHARED=1 GLM5_MAXPOS=${GLM5_MAXPOS:-512} GLM5_KV_BUDGET_GB=0
export GLM5_PREFILL_ONLY=1 GLM5_PREFILL_ROLLING=1 GLM5_ABSORB_ATTN=1 GLM5_ABSORB_SVE_DOT=1
export GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
th=${GLM5_THREAD_SWEEP:-12}; pc=${GLM5_PCHUNK_SWEEP:-32}
CFG="model=$GLM5_MODEL_DIR bf16_layers=${GLM5_BF16_LAYERS:-none}"
RUNENV="LLM_THREADS=$th OMP_NUM_THREADS=$th GLM5_PCHUNK=$pc GLM5_REAL=1 GLM5_LAYERS=0 GLM5_COMM_OVERLAP=1 TP_AR_BF16=1 GLM5_PROMPT_TOKENS=$TOK"

echo "=== FP8-mix eval: NP=$NP prompt=$NPROMPT gen=$GEN $CFG job=${PJM_JOBID:-?} ==="
date; mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null || exit 3
topo_ok=0; for t in 1 2 3 4 5; do rm -f tofu_topo.txt
  mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null||echo 0)" -ge "$NP" ] && { topo_ok=1; break; }; sleep 3; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo"; exit 3; }; echo "topo OK"
echo "--- staging ($(date)) $CFG ---"
mpiexec -np "$NP" "$LLM/build/glm5_stage" 2>"$WORK/stage_stderr.txt" || { echo "FATAL: stage"; tail -20 "$WORK/stage_stderr.txt"; exit 4; }
grep -hE "mixed-precision" "$WORK"/../pjsub_glm5_fp8mix_eval_96n.sh.${PJM_JOBID}.out 2>/dev/null || true
echo "staged $(ls "$WORK"/glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP  rank00: $(head -1 "$WORK"/glm5_stage_rank00.txt)"

echo "--- prefill $NPROMPT + greedy $GEN ($(date)) ---"; rm -f glm5_ep_rank00.txt
eval $RUNENV GLM5_PREFILL_SYNTH=$NPROMPT GLM5_GEN_NEW=$GEN GLM5_GEN_OUT=$WORK/gen.ids mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: run"; exit 5; }
grep -hE "prefill_synth:|gen: [0-9]|GEN_IDS|SENTINEL" glm5_ep_rank00.txt | head
echo "=== GEN ($CFG) ==="
[ -s "$WORK/gen.ids" ] && { echo "IDS: $(cat "$WORK/gen.ids")"; python3 "$GLM5/glm5_tokenizer.py" decode-file "$WORK/gen.ids"; }
echo "=== eval done $(date) ==="
