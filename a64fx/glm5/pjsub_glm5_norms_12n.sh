#!/bin/bash
# Per-layer hidden-state norm dump, int8 vs bf16, 12 nodes, 6 layers. The first layer where int8
# ||x|| / ||attn_out|| / ||x_postffn|| departs from bf16 localizes the broken int8 component.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=12,elapse=00:40:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=12"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1; LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=12; JOB_TAG=${PJM_JOBID:-manual_$$}; WORK="$GLM5/norms_run_${JOB_TAG}"
export GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1 GLM5_STATUS_DIR="$WORK"
export GLM5_TP=1 GLM5_MAXPOS=512 GLM5_KV_BUDGET_GB=4 GLM5_AR_HARD_CAP=2048 GLM5_AR_AUTO_CAP=2048
export GLM5_ABSORB_ATTN=1 GLM5_ABSORB_SVE_DOT=1 GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
COMMON="LLM_THREADS=48 OMP_NUM_THREADS=48 GLM5_PCHUNK=0 GLM5_REAL=1 GLM5_LAYERS=6 GLM5_COMM_OVERLAP=1 TP_AR_BF16=1 GLM5_DUMP_NORMS=1 GLM5_PROMPT_IDS=$HOME/tf_tiny.ids GLM5_MAX_NEW=1"
date; mkdir -p "$WORK"||exit 2; cd "$WORK"||exit 2
make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null||exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fccpx OPENMP=1 >/dev/null||exit 3
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt
  mpiexec -np $NP "$UTOFU/tofu_topo_helper" 2>/dev/null && [ "$(grep -vc "^#" tofu_topo.txt 2>/dev/null||echo 0)" -ge $NP ] && { topo_ok=1; break; }; sleep 12; done
[ $topo_ok = 1 ]||{ echo "FATAL topo"; exit 3; }; echo "topo OK"
runmodel(){ local tag=$1 md=$2 ns=$3
  export GLM5_MODEL_DIR=$md GLM5_NSHARDS=$ns GLM5_STAGE_DIR=/local/glm5_${tag}_$JOB_TAG GLM5_STAGE_LAYERS=6
  GLM5_LAYERS=6 mpiexec -np $NP "$LLM/build/glm5_stage" 2>${tag}_stage.err || { echo "FATAL stage $tag"; return 1; }
  echo "##### $tag staged $(ls glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP #####"
  rm -f glm5_ep_stderr_rank00.txt
  eval $COMMON mpiexec -np $NP "$LLM/build/glm5_ep_runner" >/dev/null 2>&1 || echo "run $tag rc=$?"
  cp glm5_ep_stderr_rank00.txt ${tag}_norms.txt 2>/dev/null
  echo "=== $tag LNORM (pos 0) ==="; grep -aE "LNORM" ${tag}_norms.txt 2>/dev/null | head -18
}
runmodel int8 $HOME/models/glm52-int8 233
runmodel bf16 $HOME/models/glm5.2 282
echo "=== norms done $(date) ==="
