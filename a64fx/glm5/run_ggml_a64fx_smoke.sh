#!/bin/bash
set -euo pipefail

SRC=${LLAMA_SRC:-$HOME/work/llama.cpp}
BUILD=${LLAMA_BUILD:-/local/glm53f-llama-a64fx}
OUT=${OUT:-/local/glm53f-validation-${PJM_JOBID:-manual}}
REPO=${REPO:-$HOME/work/gemm/glm53f}
TMPDIR=${TMPDIR:-$OUT/tmp}

mkdir -p "$OUT" "$TMPDIR"
export TMPDIR
FCC -Nclang -O3 -std=c++17 -fopenmp \
    -I"$SRC/ggml/src/../include" \
    "$REPO/a64fx/glm5/ggml_a64fx_smoke.cpp" \
    -L"$BUILD/bin" -lggml -lggml-cpu -lggml-base \
    -Wl,-rpath,"$BUILD/bin" \
    -o "$OUT/ggml_a64fx_smoke"

file "$OUT/ggml_a64fx_smoke"
"$OUT/ggml_a64fx_smoke" 2>&1 | tee "$OUT/ggml_a64fx_smoke.log"
