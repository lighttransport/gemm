#!/bin/bash
# Validate decode SAMPLING + repetition penalty on the int8 model, 96 nodes.
# Same prompt run twice: (A) greedy baseline [expect the 0,0,0 collapse], (B) sampled
# [temp/top_p/rep_pen, expect coherent text]. Checks NaN=0 and cross-rank token agreement
# (lockstep: replicated head + shared rng => identical sampled token on every rank).
#PJM -g hp250467
#PJM -L "rscgrp=small,node=384,elapse=01:40:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=384"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-384}; JOB_TAG=${PJM_JOBID:-manual_$$}; WORK="$GLM5/sampval_bf16_run_${JOB_TAG}"
PROMPT=${GLM5_PROMPT:-$HOME/eval_prompts/algorithm.bin}; GEN=${GLM5_GEN_NEW:-140}
export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm5.2}
export GLM5_STAGE_DIR=${GLM5_STAGE_DIR:-/local/glm5_int8_$JOB_TAG}
export GLM5_NSHARDS=${GLM5_NSHARDS:-282} GLM5_STAGE_LAYERS=78
export GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1 GLM5_STATUS_DIR="$WORK"
export GLM5_TP=1 GLM5_MAXPOS=${GLM5_MAXPOS:-2304} GLM5_KV_BUDGET_GB=16
export GLM5_AR_HARD_CAP=2048 GLM5_AR_AUTO_CAP=2048
export GLM5_PREFILL_ONLY=1 GLM5_PREFILL_ROLLING=1 GLM5_ABSORB_ATTN=1 GLM5_ABSORB_SVE_DOT=1
export GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
th=48; ptok=$(( $(stat -c%s "$PROMPT") / 4 ))
RUNENV="LLM_THREADS=$th OMP_NUM_THREADS=$th GLM5_PCHUNK=256 GLM5_REAL=1 GLM5_LAYERS=78 GLM5_COMM_OVERLAP=1 TP_AR_BF16=1 GLM5_PROMPT_TOKENS=$PROMPT GLM5_PREFILL_SYNTH=$ptok GLM5_GEN_NEW=$GEN"
echo "=== SAMPLING VALIDATION int8 NP=$NP prompt=$PROMPT ($ptok tok) gen=$GEN job=${PJM_JOBID:-?} ==="
date; mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null || exit 3
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt
  mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" 2>/dev/null && [ "$(grep -vc "^#" tofu_topo.txt 2>/dev/null||echo 0)" -ge "$NP" ] && { topo_ok=1; break; }; sleep 12; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo"; exit 3; }; echo "topo OK"
mpiexec -np "$NP" "$LLM/build/glm5_stage" 2>"$WORK/stage_stderr.txt" || { echo "FATAL: stage"; tail -20 "$WORK/stage_stderr.txt"; exit 4; }
echo "staged $(ls "$WORK"/glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP"
runone(){ # $1=label  rest=extra env
  local label=$1; shift; echo "===== RUN $label ($(date)) ====="; rm -f glm5_ep_rank*.txt
  eval $RUNENV "$@" GLM5_GEN_OUT=$WORK/${label}.ids mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: $label"; return 1; }
  grep -hE "sampling ON|gen: [0-9]+ tokens|prefill_synth:|NaNs=" glm5_ep_rank00.txt | tail -4
  # cross-rank lockstep: last sampled/argmax token must match on all ranks
  u=$(grep -hoE "argmax=[0-9]+" glm5_ep_perf_rank*.txt 2>/dev/null | sort -u | wc -l)
  echo "lockstep: distinct last-token across ranks = ${u:-NA} (1 == ok)"
  if [ -s "$WORK/${label}.ids" ]; then echo "--- $label TEXT ---"; python3 "$GLM5/glm5_tokenizer.py" decode-file "$WORK/${label}.ids" | tee "$WORK/${label}.txt" | head -40; echo "--- end ---"; fi
}
runone greedy
runone sampled GLM5_TEMP=0.7 GLM5_TOPP=0.95 GLM5_REP_PEN=1.1 GLM5_SEED=12345
echo "=== sampling validation done $(date) ==="
