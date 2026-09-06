#!/bin/bash
# COMBINED-CONFIG run @32n: the shippable decode stack, cumulative A/B (one 78L stage).
#   A baseline (shipped: NUMA off, bd=0, w8a16)  ->  B +NUMA  ->  C +batching  ->  D +int16-GEMM (FULL).
# Each arm is one runner invocation. A->D = the combined multiplier; the deltas isolate each lever.
# NUMA = OMP pin (OMP_PROC_BIND/PLACES, launch-time) + interleave (--numa 1 = set_mempolicy, in runner);
# arm A forces --numa 0 for a true CMG0-default baseline (the runner defaults --numa on now).
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=32,elapse=01:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=32"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=32; JOB_TAG=${PJM_JOBID:-manual_$$}; WORK="$GLM5/combined_run_${JOB_TAG}_32n"; NL=78
export GLM5_MODEL_DIR=$HOME/models/glm52-int8 GLM5_STAGE_DIR=/local/glm5_int8_$JOB_TAG
export GLM5_NSHARDS=233 GLM5_STAGE_LAYERS=$NL GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1 GLM5_STATUS_DIR="$WORK"
export GLM5_TP=1 GLM5_MSA=0 GLM5_MAXPOS=1024 GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
PROMPTS="$GLM5/cbatch_prompts_32.txt"
echo "=== COMBINED @${NP}n job=${PJM_JOBID:-?} ==="; date
mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
"$GLM5/check_glm5_model.sh" "$GLM5_MODEL_DIR" --tokenizer || exit 2
rm -f tofu_topo.txt glm5_stage_rank*.txt glm5_ep_*.txt
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt
  mpiexec -np $NP "$UTOFU/tofu_topo_helper" 2>/dev/null && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null||echo 0)" -ge $NP ] && { topo_ok=1; echo "topo try $t"; break; }; sleep 12; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo"; exit 3; }
echo "--- staging 78L ($(date)) ---"
mpiexec -np $NP "$LLM/build/glm5_stage" 2>stage.err || { echo "FATAL: stage"; tail -20 stage.err; exit 4; }
echo "staged $(ls glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP ($(date))"

# arm: tag, numa(0/1), bd(0/1), gemm_sdot(0/2)
run_arm(){ local tag=$1 numa=$2 bd=$3 gsd=$4; echo "--- ARM $tag: numa=$numa bd=$bd gemm_sdot=$gsd ($(date)) ---"; rm -f glm5_ep_rank00.txt
  local pin="" wrap="" th=48
  if [ "$numa" = 1 ]; then pin="OMP_PROC_BIND=close OMP_PLACES=cores"; wrap="numactl --interleave=all"; fi
  env $pin LLM_THREADS=$th OMP_NUM_THREADS=$th \
      GLM5_REAL=1 GLM5_LAYERS=$NL GLM5_BATCH_DECODE=$bd GLM5_GEMM_SDOT=$gsd \
      GLM5_CBATCH_SLOTS=8 GLM5_MAX_NEW=24 GLM5_CBATCH_PROMPTS="$PROMPTS" GLM5_CBATCH_OUT_PREFIX="$WORK/${tag}" \
      mpiexec -np $NP $wrap "$LLM/build/glm5_ep_runner" --numa $numa | grep -hE 'service decode'
  cp glm5_ep_rank00.txt "arm_${tag}_rank00.txt" 2>/dev/null
  grep -hE "cbatch: service decode" "arm_${tag}_rank00.txt" 2>/dev/null | head -1
}
run_arm A 0 0 0     # shipped baseline (no NUMA, per-slot, w8a16)
run_arm B 1 0 0     # + NUMA
run_arm C 1 1 0     # + batching (bd=1)
run_arm D 1 1 2     # + int16 GEMM (FULL STACK)

echo "--- token identity: D (full stack) vs A (baseline) — expect near-identical (NUMA exact, int16/batch near-lossless) ---"
diffs=0; for f in "$WORK"/A_0*.txt; do b="$WORK/D_$(basename "$f" | sed s/A_//)"; [ -f "$b" ] && { cmp -s "$f" "$b" || diffs=$((diffs+1)); }; done; echo "req files differing (A vs D): $diffs"
echo "SENTINEL glm5_combined_32n=done"; date
