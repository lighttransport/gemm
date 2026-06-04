#!/bin/bash
# Run the FP8 fast-decode bench on ONE clean (non-origin) node via mpiexec, so
# the 48-thread measurement is not contaminated by the co-located claude session
# on relative coord (0,0,0). Aggregates all results into ONE shared-FS file.
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"
OUT="$LLM_DIR/fp8fast_clean.log"; : > "$OUT"
BIN="/local/ds4f_fp8fast"   # node-local path; mpiexec target node builds its own? no -> build to shared
BIN="$LLM_DIR/build/ds4f_fp8fast"

# build to shared FS so the remote node sees the same binary
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -I../../common \
    -o "$BIN" ds4f_fp8_fast_bench.c -lm

# single clean node: pick a non-origin coord present in the alloc
VC="$LLM_DIR/vc_fp8fast.txt"; echo "(1,1,1)" > "$VC"

export OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores
run() { # label ftz K NROW ITERS
    local lbl="$1" ftz="$2" K="$3" N="$4" IT="$5"
    echo "===== $lbl (ftz=$ftz K=$K NROW=$N) =====" >> "$OUT"
    DS4F_FTZ=$ftz mpiexec -np 1 -vcoordfile "$VC" \
        sh -c "DS4F_FTZ=$ftz OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores '$BIN' $K $N $IT >> '$OUT' 2>&1" || echo "(run failed)" >> "$OUT"
}
# matvec-SIZE sweep at K=4096 (FTZ on, so magic-D is valid): map FP8 vs bf16
# crossover across the model's actual dense row-counts (wkv=512, wq_a=1024,
# shared=2048, wo_b=4096, wo_a=8192, wq_b=32768).
for N in 512 1024 2048 4096 8192 16384 32768; do
    run "K4096 N$N" 1 4096 "$N" 120
done
echo "=== DONE ===" >> "$OUT"
cat "$OUT"
