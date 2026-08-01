#!/bin/bash
# GLM-5.2 bf16 frontier-eval, low-concurrency + DECODE-AWARE: one 384n job, stage once, 7 tasks
# SEQUENTIALLY. Decode at 384n is ~0.16 tok/s, so GEN is small and a TIME-GUARD skips any task
# that cannot finish in the remaining wall-clock. gen.ids is flushed incrementally (partial safe).
#PJM -g hp250467
#PJM -L "rscgrp=small,node=384,elapse=03:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=384"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-384}; JOB_TAG=${PJM_JOBID:-manual_$$}
WORK="$GLM5/evalseq_run_${JOB_TAG}"; GEN=${GLM5_GEN_NEW:-256}
TOK_PER_S=0.16          # measured 384n single-stream decode rate
BUDGET_S=12600; MARGIN_S=540   # 3:30 elapse, leave 9 min tail for final detok/exit
export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm5.2}
export GLM5_STAGE_DIR=${GLM5_STAGE_DIR:-/local/glm5_bf16_$JOB_TAG}
export GLM5_NSHARDS=${GLM5_NSHARDS:-282}
export GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1 GLM5_STATUS_DIR="$WORK"
export GLM5_TP=1 GLM5_TP_SHARED=1 GLM5_MAXPOS=${GLM5_MAXPOS:-4096} GLM5_KV_BUDGET_GB=${GLM5_KV_BUDGET_GB:-16}
export GLM5_PREFILL_ONLY=1 GLM5_PREFILL_ROLLING=1 GLM5_ABSORB_ATTN=1 GLM5_ABSORB_SVE_DOT=1
export GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
th=48; pc=256
RUNENV="LLM_THREADS=$th OMP_NUM_THREADS=$th GLM5_PCHUNK=$pc GLM5_REAL=1 GLM5_LAYERS=0 GLM5_COMM_OVERLAP=1 TP_AR_BF16=1"
TASKS="coding debugging algorithm math systemdesign logic physics"
EST_S=$(awk -v g=$GEN -v r=$TOK_PER_S "BEGIN{printf \"%d\", g/r + 150}")   # per-task: gen + prefill/teardown
T0=$(date +%s)
echo "=== GLM5.2 eval SEQ: NP=$NP gen=$GEN (~${EST_S}s/task @ $TOK_PER_S tok/s) tasks=[$TASKS] job=${PJM_JOBID:-?} ==="
date; mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null || exit 3
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt
  mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" && [ "$(grep -vc "^#" tofu_topo.txt 2>/dev/null||echo 0)" -ge "$NP" ] && { topo_ok=1; break; }; sleep 12; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo"; exit 3; }; echo "topo OK"
echo "--- staging ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/glm5_stage" 2>"$WORK/stage_stderr.txt" || { echo "FATAL: stage"; tail -20 "$WORK/stage_stderr.txt"; exit 4; }
echo "staged $(ls "$WORK"/glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP ($(date))"
for task in $TASKS; do
  now=$(date +%s); used=$((now - T0)); rem=$((BUDGET_S - used - MARGIN_S))
  if [ "$rem" -lt "$EST_S" ]; then echo "### TIME-GUARD: skip $task (rem=${rem}s < est=${EST_S}s)"; continue; fi
  bin=$HOME/eval_prompts/$task.bin
  [ -s "$bin" ] || { echo "### SKIP $task (no $bin)"; continue; }
  ptok=$(( $(stat -c%s "$bin") / 4 ))
  echo "### TASK $task: prompt=$ptok gen=$GEN rem=${rem}s ($(date)) ###"; rm -f glm5_ep_rank00.txt
  eval $RUNENV GLM5_PROMPT_TOKENS=$bin GLM5_PREFILL_SYNTH=$ptok GLM5_GEN_NEW=$GEN GLM5_GEN_OUT=$WORK/${task}.ids \
    mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: run $task (continuing)"; continue; }
  cp glm5_ep_rank00.txt "rank00_${task}.txt" 2>/dev/null || true
  grep -hE "gen: [0-9]+ tokens|argmax=" "rank00_${task}.txt" 2>/dev/null | tail -2
  if [ -s "$WORK/${task}.ids" ]; then
    echo "--- $task OUTPUT (detok) ---"
    python3 "$GLM5/glm5_tokenizer.py" decode-file "$WORK/${task}.ids" | tee "$WORK/${task}.out.txt" | head -60
    echo "--- end $task ---"
  fi
done
echo "=== eval SEQ done $(date) ==="
