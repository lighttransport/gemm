#!/bin/bash
# GLM-5.2 INT8 DECODE — NUMA + w8a8-SDOT lever A/B (real weights, 32 A64FX nodes, short ctx).
# Isolates the two decode levers found by native kernel microbench (2026-07-03):
#   (1) NUMA-local weight placement — the default A64FX large-page policy is `prepage`, which pins
#       all malloc'd weights to CMG0 (doc a64fx_hpc_extensions_and_hugepage_tuning.md §6.2). With
#       OMP_NUM_THREADS=48 and no affinity, 48 threads cross-CMG-read CMG0 at the ~100 GB/s inter-CMG
#       limit. Fix = pin threads (OMP_PROC_BIND/PLACES) + interleave weights across the 4 CMGs
#       (`numactl --interleave=all`) -> ~495-739 GB/s. Microbench matvec: 197 -> 592-671 Gop/s (~3x).
#       This is the undiagnosed "OMP-48 regression" in common/glm5_impl.h:362-365.
#   (2) w8a8 SDOT M=1 matvec (GLM5_MV_SDOT=1, glm5_mv_int8_sdot) — replaces the convert-bound w8a16
#       matvec; +1.27-1.46x on top once NUMA-local (compute-bound regime). rms ~2e-4, argmax-safe.
# Arms (bd=0 single-stream decode = the M=1 lever target), each an independent runner invocation
# after ONE staging:  A baseline | B +NUMA(pin+interleave) | C +NUMA+SDOT | plus D bd=1 batched+NUMA+SDOT.
# Derived from pjsub_glm5_cbatch_int8_32n.sh (same staging/model/config).

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=32,elapse=01:20:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=32"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-32}
JOB_TAG=${PJM_JOBID:-manual_$$}
WORK="$GLM5/cbatch_int8_numa_run_${JOB_TAG}_${NP}n"
NL=${GLM5_NL:-78}
PROMPTS=${GLM5_CBATCH_PROMPTS:-$GLM5/cbatch_agentic_prompts.txt}

export GLM5_MODEL_DIR=${GLM5_MODEL_DIR:-$HOME/models/glm52-int8}
export GLM5_STAGE_DIR=${GLM5_STAGE_DIR:-/local/glm5_int8_$JOB_TAG}
export GLM5_NSHARDS=${GLM5_NSHARDS:-233}
export GLM5_STAGE_LAYERS=$NL
export GLM5_EP_SIZE=$NP GLM5_PREFILL_GROUPS=1 GLM5_STATUS_DIR="$WORK"
export GLM5_TP=${GLM5_TP:-1} GLM5_MSA=${GLM5_MSA:-0}
export GLM5_MAXPOS=${GLM5_MAXPOS:-2048}
export GLM5_CBATCH_SLOTS=${GLM5_CBATCH_SLOTS:-8}
export GLM5_MAX_NEW=${GLM5_MAX_NEW:-24}
export GLM5_TOKENIZER=$HOME/models/glm5.2/tokenizer.json
export LLM_THREADS=${LLM_THREADS:-48} OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}

echo "=== GLM5.2 INT8 NUMA/SDOT A/B: NP=$NP layers=$NL slots=$GLM5_CBATCH_SLOTS maxpos=$GLM5_MAXPOS job=${PJM_JOBID:-?} ==="
date; mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2

"$GLM5/check_glm5_model.sh" "$GLM5_MODEL_DIR" --tokenizer || exit 2
test -s "$PROMPTS" || { echo "FATAL: missing $PROMPTS"; exit 2; }

rm -f tofu_topo.txt glm5_stage_rank*.txt glm5_ep_*.txt
make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_stage glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

topo_ok=0
for t in $(seq 1 40); do
    rm -f tofu_topo.txt
    if mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" 2>/dev/null && \
       [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null || echo 0)" -ge "$NP" ]; then
        topo_ok=1; echo "topo window on try $t"; break
    fi
    echo "[cbatch_int8_numa] topo try $t failed; retry"; sleep 12
done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper failed"; exit 3; }

echo "--- staging full int8 ($(date)) ---"
mpiexec -np "$NP" "$LLM/build/glm5_stage" 2>"$WORK/stage_stderr.txt" || { echo "FATAL: stage"; tail -20 "$WORK/stage_stderr.txt"; exit 4; }
echo "staged $(ls "$WORK"/glm5_stage_rank*.txt 2>/dev/null|wc -l)/$NP ($(date))"

# one arm: tag, BD, extra-env (NUMA pin/interleave, SDOT). NUMA pin = OMP_PROC_BIND/PLACES passed
# via env; interleave = numactl wrapper on the per-rank binary.
run_arm(){ local tag=$1 bd=$2 numa=$3 sdot=$4
  echo "--- ARM $tag: bd=$bd numa=$numa sdot=$sdot ($(date)) ---"; rm -f glm5_ep_rank00.txt
  local pin="" wrap=""
  if [ "$numa" = 1 ]; then pin="OMP_PROC_BIND=close OMP_PLACES=cores"; wrap="numactl --interleave=all"; fi
  env $pin GLM5_BATCH_DECODE=$bd GLM5_MV_SDOT=$sdot GLM5_REAL=1 GLM5_LAYERS=$NL \
    GLM5_CBATCH_PROMPTS="$PROMPTS" GLM5_CBATCH_OUT_PREFIX="$WORK/${tag}" \
    mpiexec -np "$NP" $wrap "$LLM/build/glm5_ep_runner" || echo "WARN: arm $tag rc=$?"
  grep -hE "cbatch:|PROFILE cbatch_total|SENTINEL|NaN" glm5_ep_rank00.txt 2>/dev/null
  mv glm5_ep_rank00.txt "arm_${tag}_rank00.txt" 2>/dev/null
}
run_arm A 0 0 0     # baseline single-stream (shipped)
run_arm B 0 1 0     # + NUMA pin+interleave
run_arm C 0 1 1     # + NUMA + w8a8 SDOT
run_arm D 1 1 1     # batched decode + NUMA + SDOT (full stack)

echo "--- token-stream identity (A is reference; B must MATCH, C/D may differ by SDOT rounding) ---"
for X in B C D; do for f in "$WORK"/A_*.txt; do o="$WORK/${X}_${f##*/A_}"; [ -f "$o" ] && { cmp -s "$f" "$o" || echo "TOKEN DIFF A vs $X: ${f##*/A_}"; }; done; done
echo "SENTINEL glm5_cbatch_int8_numa_${NP}n=done"; date
