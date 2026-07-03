#!/bin/bash
# GLM-5.2 int8 decode — UNIFIED production launcher. Stages the model once, runs the runner via CLI
# flags (not env soup), with NUMA on by default (the 1.40x bit-identical lever: in-runner
# set_mempolicy interleave + OMP thread pinning). Supersedes the cbatch_int8_{32,48,96}n / _numa /
# _batch one-offs — set the knobs below (or override at submit: `NODES=48 BATCH=1 pjsub ...` won't
# work through pjsub, so copy+edit node=/proc=/NODES together, or submit interactively).
#
# Knobs (env at top only; everything the RUNNER needs is passed as --flags):
#   NODES(=32) LAYERS(=78) SLOTS(=8) MAXNEW(=24) MAXPOS(=2048) BATCH(=0) SDOT(=0) OVERLAP(=0) NUMA(=1)

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=32,elapse=01:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=32"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
: "${NODES:=${PJM_MPI_PROC:-32}}"; : "${LAYERS:=78}"; : "${SLOTS:=8}"; : "${MAXNEW:=24}"
: "${MAXPOS:=2048}"; : "${BATCH:=0}"; : "${SDOT:=0}"; : "${OVERLAP:=0}"; : "${NUMA:=1}"
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=$NODES; JOB_TAG=${PJM_JOBID:-manual_$$}; WORK="$GLM5/decode_run_${JOB_TAG}_${NP}n"
MODEL=${GLM5_MODEL_DIR:-$HOME/models/glm52-int8}; STAGE=/local/glm5_int8_$JOB_TAG
PROMPTS=${PROMPTS:-$GLM5/cbatch_agentic_prompts.txt}
# staging env (glm5_stage is env-driven); the RUNNER is driven by --flags below.
export GLM5_MODEL_DIR=$MODEL GLM5_STAGE_DIR=$STAGE GLM5_NSHARDS=233 GLM5_STAGE_LAYERS=$LAYERS
export GLM5_EP_SIZE=$NP GLM5_STATUS_DIR="$WORK" GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
# NUMA thread-pin (OMP reads affinity at init -> must be launch-time env; the interleave half is
# landed in the runner's --numa). OVERLAP needs one core free for the pinned comm-driver.
TH=48; [ "$OVERLAP" = 1 ] && TH=47
[ "$NUMA" = 1 ] && export OMP_PROC_BIND=close OMP_PLACES=cores
export LLM_THREADS=$TH OMP_NUM_THREADS=$TH
echo "=== GLM5.2 decode: NP=$NP L=$LAYERS slots=$SLOTS batch=$BATCH sdot=$SDOT overlap=$OVERLAP numa=$NUMA th=$TH job=${PJM_JOBID:-?} ==="
date; mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
"$GLM5/check_glm5_model.sh" "$MODEL" --tokenizer || exit 2
test -s "$PROMPTS" || { echo "FATAL: prompts $PROMPTS"; exit 2; }

rm -f tofu_topo.txt glm5_stage_rank*.txt glm5_ep_*.txt
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt
  mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" 2>/dev/null && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null||echo 0)" -ge "$NP" ] && { topo_ok=1; echo "topo try $t"; break; }; sleep 12; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo"; exit 3; }
echo "--- staging ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/glm5_stage" 2>"$WORK/stage_stderr.txt" || { echo "FATAL: stage"; tail -20 "$WORK/stage_stderr.txt"; exit 4; }
echo "staged $(ls "$WORK"/glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP ($(date))"

# ---- the run: ALL runtime config as CLI flags (NUMA interleave via --numa; batching/overlap/sdot too)
ARGS="--real 1 --numa $NUMA --model $MODEL --layers $LAYERS --tp 1 --ep-size $NP \
  --maxpos $MAXPOS --slots $SLOTS --max-new $MAXNEW --prompts $PROMPTS \
  --batch-decode $BATCH --sdot $SDOT --gen-out $WORK/gen"
[ "$OVERLAP" = 1 ] && ARGS="$ARGS --overlap 1 --tp-shared 0"
echo "--- run: glm5_ep_runner $ARGS ($(date)) ---"; rm -f glm5_ep_rank00.txt
mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" $ARGS || echo "WARN: run rc=$?"
grep -hE "cbatch:|PROFILE cbatch_total|comm-driver pinned|SENTINEL|NaN" glm5_ep_rank00.txt 2>/dev/null
echo "SENTINEL glm5_decode_${NP}n=done"; date
