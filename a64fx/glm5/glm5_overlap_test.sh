#!/bin/bash
# GLM5.2 compute/comm OVERLAP reproducer — SYNTHETIC weights, 2..12 nodes, no staging (fast/interactive).
# Reproduces the K2 (job 49420218) failure: GLM5_COMM_OVERLAP=1 in the decode path is ~56x SLOWER because
# the comm-driver thread + N OMP threads oversubscribe the N cores. Also tests the fix hypothesis
# (free one core for the comm thread: OMP_NUM_THREADS = cores-1). Runs the REAL glm5_ep_runner overlap
# path (comm_driver + ar_async_start/ar_wait, glm5_impl.h:1581), just with a tiny synthetic MoE model.
#
# Interactive (recommended): grab an allocation, then run this with NP = node count. E.g.
#   pjsub --interact -g hp250467 -L "rscgrp=int,node=4,elapse=00:30:00" --mpi "proc=4"
#   NP=4 bash a64fx/glm5/glm5_overlap_test.sh
# Batch: pjsub a64fx/glm5/pjsub_glm5_overlap_test_4n.sh  (a thin wrapper that calls this).
set -u
: "${NP:=${PJM_MPI_PROC:-4}}"                 # ranks = nodes (1 rank/node). Override: NP=8 bash ...
: "${CORES:=48}"                              # A64FX compute cores/node (comm thread would be the (CORES+1)th)
REPO=${REPO:-$HOME/work/gemm/glm5-1}
LLM="$REPO/a64fx/llm"; UTOFU="$REPO/a64fx/utofu-tests"; GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
WORK="$GLM5/overlap_test_${PJM_JOBID:-int}_${NP}n"; mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
echo "=== GLM5.2 overlap test: NP=$NP cores/node=$CORES  $(date) ==="

# tiny synthetic MoE model: enough MoE layers + experts + a shared expert to exercise the routed-AR overlap.
export GLM5_REAL=0 GLM5_DUMMY=1 GLM5_LAYERS=8 GLM5_EXPERTS=32 GLM5_MAXPOS=128
export GLM5_TP=1 GLM5_PREFILL_GROUPS=1 GLM5_EP_SIZE=$NP GLM5_STATUS_DIR="$WORK"
export GLM5_CBATCH_SLOTS=4 GLM5_MAX_NEW=16 GLM5_BATCH_DECODE=0    # per-slot decode -> clean comm% + hits the overlap path

make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3
# synthetic prompts
PROMPTS="$WORK/prompts.txt"; : > "$PROMPTS"
for i in 0 1 2 3; do echo "$((i+2)) $((i+3)) $((i+5)) $((i+7)) $((i+11))" >> "$PROMPTS"; done
export GLM5_CBATCH_PROMPTS="$PROMPTS"

# Fujitsu MPI coll-select init window: retry topo until it writes >= NP lines.
topo_ok=0; for t in $(seq 1 20); do rm -f tofu_topo.txt
  mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" 2>/dev/null && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null||echo 0)" -ge "$NP" ] && { topo_ok=1; echo "topo try $t ok"; break; }; sleep 8; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo helper"; exit 3; }

# arm: tag, TP_SHARED, COMM_OVERLAP, OMP_NUM_THREADS, PIN(0/1 = pin OMP pool to cores; the runner
# auto-pins the comm-driver to the highest cpuset core, so PIN=1 frees that core for it)
run_arm(){ local tag=$1 tpsh=$2 ovl=$3 omp=$4 pin=${5:-0}
  local pinenv=""; [ "$pin" = 1 ] && pinenv="OMP_PROC_BIND=close OMP_PLACES=cores"
  echo "--- ARM $tag: TP_SHARED=$tpsh COMM_OVERLAP=$ovl OMP=$omp PIN=$pin ($(date)) ---"; rm -f glm5_ep_rank00.txt
  env $pinenv GLM5_TP_SHARED=$tpsh GLM5_COMM_OVERLAP=$ovl LLM_THREADS=$omp OMP_NUM_THREADS=$omp \
    GLM5_CBATCH_OUT_PREFIX="$WORK/${tag}" \
    timeout 600 mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || echo "WARN: arm $tag rc=$? (timeout=BROKEN)"
  grep -hE "cbatch: service decode|comm-driver pinned|SENTINEL glm5_cbatch" glm5_ep_rank00.txt 2>/dev/null
  mv glm5_ep_rank00.txt "arm_${tag}_rank00.txt" 2>/dev/null
}
run_arm A 1 0 $CORES       0        # baseline: shipped (sharded shared, no overlap)
run_arm B 0 0 $CORES       0        # replicated shared, no overlap    -> replication COST
run_arm C 0 1 $CORES       0        # overlap, OMP=all cores, unpinned  -> reproduce the 56x SLOWDOWN
run_arm E 0 1 $((CORES-1)) 0        # overlap, OMP=cores-1, UNPINNED (the old "fix": free a core but comm floats)
run_arm D 0 1 $((CORES-1)) 1        # overlap, OMP=cores-1, PINNED      -> the REAL fix (comm gets its own core)
run_arm F 0 0 $((CORES-1)) 1        # no-overlap, OMP=cores-1, pinned   -> control: cost of losing 1 compute core

echo "--- token identity (A is the reference) ---"
for f in "$WORK"/A_*.txt; do for X in B C D E F; do o="$WORK/${X}_${f##*/A_}"; [ -f "$o" ] && { cmp -s "$f" "$o" || echo "TOKEN DIFF A vs $X: ${f##*/}"; }; done; done
echo "=== overlap test done $(date) ==="
echo "READ: A=baseline tok/s. C should be ~10-50x slower (repro). If D ~= A speed -> oversubscription confirmed,"
echo "      fix = OMP=cores-1 (+ pin comm thread). If D still slow -> deeper bug. Check TOKEN DIFF for correctness."
