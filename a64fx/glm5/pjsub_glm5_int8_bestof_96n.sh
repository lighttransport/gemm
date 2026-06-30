#!/bin/bash
# Meaningful int8 workload (post router-fix): best-of-N sampling eval. Each job runs the 8 frontier
# prompts with sampling (seed = job id -> diverse samples), GEN tokens each, detok+save per prompt,
# time-guarded to fill ~5h. Run N copies for a best-of-N corpus + to burn the node-hour budget.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=96,elapse=05:15:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1; LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=96; JOB_TAG=${PJM_JOBID:-manual_$$}; WORK="$GLM5/bestof_run_${JOB_TAG}"
GEN=${GLM5_GEN_NEW:-512}; SEED=${GLM5_SEED:-${PJM_JOBID:-12345}}
TOK_PER_S=0.20; BUDGET_S=18900; MARGIN_S=600; EST_S=$(awk -v g=$GEN -v r=$TOK_PER_S "BEGIN{printf \"%d\", g/r+150}")
export GLM5_MODEL_DIR=$HOME/models/glm52-int8 GLM5_NSHARDS=233 GLM5_STAGE_LAYERS=78
export GLM5_STAGE_DIR=/local/glm5_int8_$JOB_TAG GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1 GLM5_STATUS_DIR="$WORK"
export GLM5_TP=1 GLM5_MAXPOS=2304 GLM5_KV_BUDGET_GB=16 GLM5_AR_HARD_CAP=2048 GLM5_AR_AUTO_CAP=2048
export GLM5_PREFILL_ONLY=1 GLM5_PREFILL_ROLLING=1 GLM5_ABSORB_ATTN=1 GLM5_ABSORB_SVE_DOT=1
export GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
# sampling ON (the fixed int8 + the new sampler)
RUNENV="LLM_THREADS=48 OMP_NUM_THREADS=48 GLM5_PCHUNK=256 GLM5_REAL=1 GLM5_LAYERS=78 GLM5_COMM_OVERLAP=1 TP_AR_BF16=1 GLM5_TEMP=0.7 GLM5_TOPP=0.95 GLM5_REP_PEN=1.1 GLM5_SEED=$SEED"
TASKS="coding debugging algorithm math systemdesign logic physics analysis"
T0=$(date +%s)
echo "=== INT8 best-of-N: NP=$NP gen=$GEN seed=$SEED tasks=[$TASKS] job=${PJM_JOBID:-?} ==="; date
mkdir -p "$WORK"||exit 2; cd "$WORK"||exit 2
make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null||exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null||exit 3
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt
  mpiexec -np $NP "$UTOFU/tofu_topo_helper" 2>/dev/null && [ "$(grep -vc "^#" tofu_topo.txt 2>/dev/null||echo 0)" -ge $NP ] && { topo_ok=1; break; }; sleep 12; done
[ $topo_ok = 1 ]||{ echo "FATAL topo"; exit 3; }; echo "topo OK"
mpiexec -np $NP "$LLM/build/glm5_stage" 2>stage.err||{ echo FATAL stage; tail stage.err; exit 4; }
echo "staged $(ls glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP ($(date))"
round=0
while :; do round=$((round+1))
  for task in $TASKS; do
    now=$(date +%s); rem=$((BUDGET_S-(now-T0)-MARGIN_S))
    [ "$rem" -lt "$EST_S" ] && { echo "### TIME-GUARD stop (rem=${rem}s) after $((round-1)) full rounds"; echo "=== bestof done $(date) ==="; exit 0; }
    bin=$HOME/eval_prompts/$task.bin; [ -s "$bin" ]||continue
    ptok=$(( $(stat -c%s "$bin")/4 )); out="r${round}_${task}"
    echo "### $out prompt=$ptok gen=$GEN rem=${rem}s ($(date)) ###"; rm -f glm5_ep_rank00.txt
    eval $RUNENV GLM5_PROMPT_TOKENS=$bin GLM5_PREFILL_SYNTH=$ptok GLM5_GEN_NEW=$GEN GLM5_GEN_OUT=$WORK/${out}.ids \
      mpiexec -np $NP "$LLM/build/glm5_ep_runner" 2>/dev/null || { echo "FATAL run $out"; continue; }
    [ -s "$WORK/${out}.ids" ] && python3 "$GLM5/glm5_tokenizer.py" decode-file "$WORK/${out}.ids" > "$WORK/${out}.txt" 2>/dev/null
    echo "  -> $(wc -c <"$WORK/${out}.txt" 2>/dev/null||echo 0) chars"
  done
done
