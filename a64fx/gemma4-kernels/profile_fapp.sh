#!/bin/bash
# Profile a standalone kernel microbench with fapp across all 17 PA event groups.
# Mirrors the clair llm-guided-opt profiling flow.
#
# Usage:  bash profile_fapp.sh <bin_fj> "<args>" [outdir]
#   e.g.  bash profile_fapp.sh q8v2_profile_fj "16 200 0"
#         bash profile_fapp.sh q8v2_profile_fj "16 200 1"   # per-row variant
#         bash profile_fapp.sh fp16_profile_fj "384 200"
#         bash profile_fapp.sh attn_profile_fj "256 384 512"
#
# Produces <outdir>/pa{1..17}.csv. Use fewer reps than the native run (fapp adds
# overhead and each PA group reruns the binary). Pin to one core for clean per-core
# CPUPA numbers (single-thread kernels); attn uses OMP so set OMP_NUM_THREADS=1 too.
set -e
BIN=${1:?need binary, e.g. q8v2_profile_fj}
ARGS=${2:-"16 200 0"}
OUT=${3:-prof_${BIN}}
REGION=$(basename "$BIN")

command -v fapp >/dev/null || { echo "fapp not found (login/compute node only)"; exit 1; }
[ -x "./$BIN" ] || make "$BIN"

mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_PROC_BIND=close OMP_PLACES=cores
echo "=== fapp profiling ./$BIN $ARGS  (OMP_NUM_THREADS=$OMP_NUM_THREADS) ==="
for i in $(seq 1 17); do
    echo "  pa${i}/17..."
    rm -rf "$OUT/rep${i}"
    fapp -C -d "$OUT/rep${i}" -Hevent=pa${i} ./"$BIN" $ARGS >/dev/null
done
echo "Exporting CSVs..."
for i in $(seq 1 17); do
    fapppx -A -d "$OUT/rep${i}" -Icpupa -tcsv -o "$OUT/pa${i}.csv"
done
echo "=== Done. CSVs in $OUT/pa{1..17}.csv (region '$REGION') ==="
echo "pa1 = statistics (cycles, insns, GFLOPS); pa6/pa7 = cache; see fapp_pmu_profiling.md"
