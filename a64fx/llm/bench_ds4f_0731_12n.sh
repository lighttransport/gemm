#!/bin/bash
# Comparable before/after benchmark for the K3-compatible DS4F runtime polish.
# Run after run_ds4f_0731_stage_12n.sh inside the same 12-node allocation.
set -euo pipefail

LLM_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LLM_DIR"
ROOT=${RESULT_ROOT:-$LLM_DIR/runs/ds4f-0731-bench-${PJM_JOBID:-manual-$$}}
STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4f-0731-${PJM_JOBID:-manual}}
PREFILL=${DS4F_PREFILL:-64}
MAXGEN=${DS4F_MAXGEN:-32}
MAXPOS=${DS4F_MAXPOS:-256}

run_case() {
    local tag=$1 spins=$2 robust=$3 out="$ROOT/$1"
    mkdir -p "$out"
    echo "=== case=$tag poll_spins=$spins robust=$robust ==="
    DS4F_COMM_POLL_SPINS="$spins" DS4F_COMM_ROBUST="$robust" \
    DS4F_PREFILL="$PREFILL" DS4F_MAXGEN="$MAXGEN" DS4F_MAXPOS="$MAXPOS" \
    DS4F_RUN_TAG="ds4f-0731-$tag" DS4F_STAGE_DIR="$STAGE_DIR" \
    RESULT_DIR="$out" ./run_ds4f_0731_12n.sh > "$out/outer.log" 2>&1
}

mkdir -p "$ROOT"
run_case baseline 1 0
run_case polished 4 1
python3 report_ds4f_0731.py "$ROOT/baseline" "$ROOT/polished" | tee "$ROOT/decision.txt"
echo "DS4F_0731_BENCH_PASS root=$ROOT"
