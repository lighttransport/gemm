#!/bin/bash
# DeepSeek-V4-Flash (ds4f) — UNIFIED pjsub launcher. Stages the fp8+fp4 weights once to node-local
# /local, then runs ds4f_ep_runner via CLI --flags (NUMA on by default: the 1.40x bit-identical decode
# lever = in-runner set_mempolicy(MPOL_INTERLEAVE) + OMP thread pinning). Mirrors pjsub_glm5_decode.sh.
#
# Unlike the interactive run_ds4f_*11n.sh (which reserve 1 node for the login/claude control process),
# this is a BATCH job: ALL $NODES are EP ranks (e % NODES == rank). Edit #PJM node=/proc= to match NODES.
#
# Knobs (env at top only; the RUNNER is driven by --flags):
#   NODES(=11) MODE(=decode) PREFILL(=8) MAXGEN(=32) MAXPOS(=4096)
#   CP(=0) INT8_KV(=0) MTP(=0) NUMA(=1) PROMPT_IDS(=)  (set PROMPT_IDS=file for real-prompt gen)
# MODE:
#   prefill  — long prompt, 1 decode step; measures time-to-first-token. Run ABOVE the node floor.
#   decode   — short prompt, many decode steps; the --preset decode bundle. Run AT the node floor.
#   serve    — prefill+decode; --mtp/--cp for speculative + long-ctx. Node count pinned by KV.
# Node floor (weights 9.02 GB replicated + 150.59/N experts, <=27 GB): decode ~11n; see ds4f_sim.py.

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=11,elapse=01:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=11"
#PJM --llio localtmp-size=48Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u
: "${NODES:=${PJM_MPI_PROC:-11}}"; : "${MODE:=decode}"
: "${PREFILL:=8}"; : "${MAXGEN:=32}"; : "${MAXPOS:=4096}"
: "${CP:=0}"; : "${INT8_KV:=0}"; : "${MTP:=0}"; : "${NUMA:=1}"; : "${PROMPT_IDS:=}"
REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=$NODES; JOB_TAG=${PJM_JOBID:-manual_$$}; WORK="$LLM/ds4f_run_${JOB_TAG}_${NP}n"
MODEL=${DS4F_MODEL_DIR:-$HOME/models/ds4f}; STAGE=/local/ds4f_$JOB_TAG

# ---- staging env (ds4f_stage is env-driven; the RUNNER is driven by --flags below) ----
export DS4F_MODEL_DIR=$MODEL DS4F_STAGE_DIR=$STAGE DS4F_NSHARDS=46 DS4F_EP_SIZE=$NP
export DS4F_STATUS_DIR="$WORK" DS4F_STAGE_FLUSH_GB=${DS4F_STAGE_FLUSH_GB:-2}
# NUMA thread-pin (OMP reads affinity at init -> must be launch-time env; the interleave half is in the
# runner's --numa). The stager does no compute, so pinning only matters for the run.
[ "$NUMA" = 1 ] && export OMP_PROC_BIND=close OMP_PLACES=cores
export LLM_THREADS=48 OMP_NUM_THREADS=48
echo "=== ds4f (DeepSeek-V4-Flash): NP=$NP mode=$MODE prefill=$PREFILL maxgen=$MAXGEN maxpos=$MAXPOS cp=$CP int8_kv=$INT8_KV mtp=$MTP numa=$NUMA job=${PJM_JOBID:-?} ==="
date; mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
test -d "$MODEL" || { echo "FATAL: model dir $MODEL"; exit 2; }

rm -f tofu_topo.txt ds4f_stage_rank*.txt ds4f_ep_*.txt
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" ds4f_stage ds4f_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
# topo retry: coll-select can race the system rule file on cold ranks
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt
  mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" 2>/dev/null && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null||echo 0)" -ge "$NP" ] && { topo_ok=1; echo "topo try $t"; break; }; sleep 12; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo"; exit 3; }

echo "--- staging fp8+fp4 weights ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/ds4f_stage" 2>"$WORK/stage_stderr.txt" || { echo "FATAL: stage"; tail -20 "$WORK/stage_stderr.txt"; exit 4; }
echo "staged $(ls "$WORK"/ds4f_stage_rank*.txt 2>/dev/null|wc -l)/$NP ($(date))"

# ---- the run: ALL runtime config as CLI flags (NUMA interleave via --numa) ----
COMMON="--real 1 --numa $NUMA --model $MODEL --stage-dir $STAGE --ep-size $NP --nshards 46 \
  --maxpos $MAXPOS --int8-kv $INT8_KV --cp $CP"
case "$MODE" in
  prefill) ARGS="$COMMON --prefill $PREFILL --max-gen 1 --tierb2 1 --sparse 1" ;;
  decode)  ARGS="$COMMON --preset decode --prefill $PREFILL --max-gen $MAXGEN" ;;
  serve)   ARGS="$COMMON --preset decode --mtp $MTP --prefill $PREFILL --max-gen $MAXGEN" ;;
  *) echo "FATAL: unknown MODE=$MODE (prefill|decode|serve)"; exit 2 ;;
esac
[ -n "$PROMPT_IDS" ] && ARGS="$ARGS --prompt-ids $PROMPT_IDS --gen-out $WORK/gen_ids.txt"
echo "--- run: ds4f_ep_runner $ARGS ($(date)) ---"; rm -f ds4f_ep_rank00.txt
mpiexec -np "$NP" "$LLM/build/ds4f_ep_runner" $ARGS || echo "WARN: run rc=$?"
grep -hE "prefill|decode|tok/s|arena|RSS|NUMA interleave|NaN|argmax" ds4f_ep_rank00.txt ds4f_ep_perf_rank00.txt 2>/dev/null | head -30
echo "SENTINEL ds4f_${MODE}_${NP}n=done"; date
