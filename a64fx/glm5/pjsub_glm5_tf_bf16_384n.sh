#!/bin/bash
# Teacher-forcing accuracy of the INT8 forward (token-at-a-time prefill) on a real prompt.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=384,elapse=01:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=384"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1; LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=384; JOB_TAG=${PJM_JOBID:-manual_$$}; WORK="$GLM5/tfbf16_run_${JOB_TAG}"
export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm5.2}
export GLM5_STAGE_DIR=/local/glm5_tf_$JOB_TAG GLM5_NSHARDS=${GLM5_NSHARDS:-282} GLM5_STAGE_LAYERS=78
export GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1 GLM5_STATUS_DIR="$WORK"
export GLM5_TP=1 GLM5_MAXPOS=2304 GLM5_KV_BUDGET_GB=16 GLM5_AR_HARD_CAP=2048 GLM5_AR_AUTO_CAP=2048
export GLM5_ABSORB_ATTN=1 GLM5_ABSORB_SVE_DOT=1 GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
RUNENV="LLM_THREADS=48 OMP_NUM_THREADS=48 GLM5_PCHUNK=0 GLM5_REAL=1 GLM5_LAYERS=78 GLM5_COMM_OVERLAP=1 TP_AR_BF16=1 GLM5_TF_CHECK=1 GLM5_PROMPT_IDS=$HOME/tf_algorithm.ids GLM5_MAX_NEW=4"
echo "=== BF16 TF check NP=$NP model=$GLM5_MODEL_DIR job=${PJM_JOBID:-?} ==="; date; mkdir -p "$WORK"||exit 2; cd "$WORK"||exit 2
make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null||exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null||exit 3
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt
  mpiexec -np $NP "$UTOFU/tofu_topo_helper" 2>/dev/null && [ "$(grep -vc "^#" tofu_topo.txt 2>/dev/null||echo 0)" -ge $NP ] && { topo_ok=1; break; }; sleep 12; done
[ $topo_ok = 1 ]||{ echo "FATAL topo"; exit 3; }
mpiexec -np $NP "$LLM/build/glm5_stage" 2>stage.err||{ echo FATAL stage; tail stage.err; exit 4; }
echo "staged $(ls glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP"
eval $RUNENV mpiexec -np $NP "$LLM/build/glm5_ep_runner" || { echo "FATAL run"; exit 5; }
echo "=== TF done $(date) ==="
