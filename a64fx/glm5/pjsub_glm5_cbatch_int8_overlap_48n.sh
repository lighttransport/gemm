#!/bin/bash
# K2 — compute/comm OVERLAP A/B, int8 full model @48n (memory-safe for replicated shared: 20 GB/node).
# The MoE routed-AR overlap (glm5_impl.h:1581) only engages when the shared expert is REPLICATED
# (!tp_sh); GLM5_TP_SHARED defaults to GLM5_TP=1, so it must be forced off. 3 arms isolate it:
#   A baseline    : TP_SHARED=1 OVERLAP=0  (shipped: sharded shared, no overlap)
#   B replicated  : TP_SHARED=0 OVERLAP=0  (replicated shared, no overlap -> the replication COST)
#   C overlap     : TP_SHARED=0 OVERLAP=1  (routed-AR hidden behind shared GEMMs -> the BENEFIT)
# overlap gain = C vs B; net vs shipped = C vs A. All arms GLM5_DOT_ACC4=1 (shipped attn kernel).

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=48,elapse=01:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=48"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-48}; JOB_TAG=${PJM_JOBID:-manual_$$}
WORK="$GLM5/overlap_run_${JOB_TAG}_${NP}n"; NL=78
PROMPTS="$GLM5/cbatch_agentic_prompts.txt"
export GLM5_MODEL_DIR=$HOME/models/glm52-int8 GLM5_STAGE_DIR=/local/glm5_int8_$JOB_TAG
export GLM5_NSHARDS=233 GLM5_STAGE_LAYERS=$NL
export GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1 GLM5_STATUS_DIR="$WORK"
export GLM5_TP=1 GLM5_MSA=0 GLM5_MAXPOS=2048 GLM5_CBATCH_SLOTS=8 GLM5_MAX_NEW=24
export GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
export LLM_THREADS=48 OMP_NUM_THREADS=48
echo "=== GLM5.2 int8 OVERLAP A/B NP=$NP job=${PJM_JOBID:-?} ==="; date
mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
"$GLM5/check_glm5_model.sh" "$GLM5_MODEL_DIR" --tokenizer || exit 2
test -s "$PROMPTS" || { echo "FATAL: prompts"; exit 2; }
rm -f tofu_topo.txt glm5_stage_rank*.txt glm5_ep_*.txt
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt
  mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" 2>/dev/null && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null||echo 0)" -ge "$NP" ] && { topo_ok=1; echo "topo try $t"; break; }; sleep 12; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo"; exit 3; }
echo "--- staging ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/glm5_stage" 2>"$WORK/stage_stderr.txt" || { echo "FATAL: stage"; tail -20 "$WORK/stage_stderr.txt"; exit 4; }
echo "staged $(ls "$WORK"/glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP ($(date))"

run_arm(){ local tag=$1 tpsh=$2 ovl=$3; echo "--- ARM $tag: TP_SHARED=$tpsh COMM_OVERLAP=$ovl ($(date)) ---"; rm -f glm5_ep_rank00.txt
  GLM5_TP_SHARED=$tpsh GLM5_COMM_OVERLAP=$ovl GLM5_DOT_ACC4=1 GLM5_REAL=1 GLM5_LAYERS=$NL GLM5_BATCH_DECODE=0 \
    GLM5_CBATCH_PROMPTS="$PROMPTS" GLM5_CBATCH_OUT_PREFIX="$WORK/${tag}" \
    mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || echo "WARN: arm $tag rc=$?"
  grep -hE "cbatch: service decode|PROFILE cbatch_total (wall|attn|shared|router|experts)|AR/MoE|overlap" glm5_ep_rank00.txt 2>/dev/null
  mv glm5_ep_rank00.txt "arm_${tag}_rank00.txt" 2>/dev/null
}
run_arm A 1 0
run_arm B 0 0
run_arm C 0 1
echo "--- token identity across arms (A vs C should MATCH) ---"
for f in "$WORK"/A_*.txt; do c="$WORK/C_${f##*/A_}"; cmp -s "$f" "$c" || echo "TOKEN DIFF A vs C: ${f##*/}"; done
echo "SENTINEL glm5_overlap_${NP}n=done"; date
