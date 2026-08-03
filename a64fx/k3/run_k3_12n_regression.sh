#!/bin/bash
# Model-free K3/uTofu regression matrix for an interactive 12-node allocation.
# This intentionally exercises startup/teardown repeatedly and keeps payloads
# small enough to run safely while debugging a hardware barrier failure.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
UTOFU="$REPO/a64fx/utofu-tests"
NODES=12
BARRIER_ITERS=${K3_BARRIER_ITERS:-128}
LONG_BARRIER_ITERS=${K3_LONG_BARRIER_ITERS:-512}
AR_COUNT=${K3_AR_COUNT:-32}
AR_REPS=${K3_AR_REPS:-32}
DROP_EVERY=${K3_AR_DROP_EVERY:-17}
JOB_TAG=${PJM_JOBID:-manual-$$}
ROOT="$SCRIPT_DIR/logs/k3-12n-regression-$JOB_TAG"
RUN_BARRIER=1
RUN_AR=1

usage() {
    cat >&2 <<EOF
usage: $0 [options]
  --result-dir DIR       output directory (must not already exist)
  --barrier-iters N      normal barrier stress length (default: $BARRIER_ITERS)
  --long-barrier-iters N long barrier stress length (default: $LONG_BARRIER_ITERS)
  --ar-count N           all-reduce element count (default: $AR_COUNT)
  --ar-reps N             all-reduce repetitions per case (default: $AR_REPS)
  --drop-every N         ACK case drops every Nth payload Put (default: $DROP_EVERY)
  --barrier-only         skip all-reduce cases
  --ar-only               skip barrier cases
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --result-dir) ROOT=$2; shift 2;;
        --barrier-iters) BARRIER_ITERS=$2; shift 2;;
        --long-barrier-iters) LONG_BARRIER_ITERS=$2; shift 2;;
        --ar-count) AR_COUNT=$2; shift 2;;
        --ar-reps) AR_REPS=$2; shift 2;;
        --drop-every) DROP_EVERY=$2; shift 2;;
        --barrier-only) RUN_AR=0; shift;;
        --ar-only) RUN_BARRIER=0; shift;;
        -h|--help) usage; exit 0;;
        *) echo "$0: unknown option: $1" >&2; usage; exit 2;;
    esac
done

is_uint() { case "$1" in ''|*[!0-9]*) return 1;; esac; }
for value_name in BARRIER_ITERS LONG_BARRIER_ITERS AR_COUNT AR_REPS DROP_EVERY; do
    value=${!value_name}
    if ! is_uint "$value"; then
        echo "$0: $value_name must be a positive integer" >&2
        exit 2
    fi
done
if [ "$BARRIER_ITERS" -lt 1 ] || [ "$BARRIER_ITERS" -gt 100000 ] ||
   [ "$LONG_BARRIER_ITERS" -lt 1 ] || [ "$LONG_BARRIER_ITERS" -gt 100000 ]; then
    echo "$0: barrier iteration counts must be in 1..100000" >&2
    exit 2
fi
if [ "$AR_COUNT" -lt 1 ] || [ "$AR_COUNT" -gt 4096 ] ||
   [ "$AR_REPS" -lt 1 ] || [ "$AR_REPS" -gt 100000 ] || [ "$DROP_EVERY" -lt 1 ]; then
    echo "$0: invalid all-reduce test size/repetition/drop value" >&2
    exit 2
fi

ROOT=$(realpath -m "$ROOT")
if [ -n "${PJM_MPI_PROC:-}" ] && [ "$PJM_MPI_PROC" -ne "$NODES" ]; then
    echo "$0: this harness requires PJM_MPI_PROC=$NODES (got $PJM_MPI_PROC)" >&2
    exit 3
fi
if [ -e "$ROOT" ]; then
    echo "$0: result root already exists: $ROOT" >&2
    exit 2
fi

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
mkdir -p "$ROOT"
K3_RUNNER="$SCRIPT_DIR/k3_full_runner"
TOPO_HELPER="$UTOFU/tofu_topo_helper"
AR_TEST="$UTOFU/tp_ar_ack_test"
TOPO="$ROOT/tofu_topo.txt"
SUMMARY="$ROOT/summary.log"

echo "K3_12N_REGRESSION_BEGIN nodes=$NODES root=$ROOT" | tee "$SUMMARY"
make -C "$SCRIPT_DIR" full-runner >/dev/null
make -C "$UTOFU" tofu_topo_helper tp_ar_ack_test >/dev/null

cd "$ROOT"
mpiexec -np "$NODES" "$TOPO_HELPER" >topology.log 2>&1
if [ "$(grep -vc '^#' "$TOPO")" -ne "$NODES" ]; then
    echo "$0: topology discovery did not produce $NODES rows" >&2
    exit 4
fi
echo "topology PASS rows=$NODES" | tee -a "$SUMMARY"

run_barrier_case() {
    local name=$1
    local iters=$2
    local case_dir="$ROOT/barrier-$name"
    local prefix="$case_dir/rank"
    mkdir -p "$case_dir"
    echo "barrier BEGIN case=$name iterations=$iters" | tee -a "$SUMMARY"
    mpiexec -np "$NODES" -of-proc "$prefix" "$K3_RUNNER" \
        --mode barrier --nodes "$NODES" --topo "$TOPO" --barrier-iters "$iters" \
        >"$case_dir/launcher.log" 2>&1
    if ! grep -hF "K3FULL_BARRIER PASS nodes=$NODES iterations=$iters " "$prefix".* \
        >"$case_dir/result.log"; then
        echo "barrier FAIL case=$name; rank logs are in $case_dir" >&2
        exit 10
    fi
    echo "barrier PASS case=$name iterations=$iters" | tee -a "$SUMMARY"
}

run_ar_case() {
    local name=$1
    local ack=$2
    local drop=$3
    local deterministic=$4
    local a2a=$5
    local bf16=$6
    local ar2d=$7
    local case_dir="$ROOT/ar-$name"
    local prefix="$case_dir/rank"
    mkdir -p "$case_dir"
    echo "allreduce BEGIN case=$name ack=$ack drop=$drop deterministic=$deterministic a2a=$a2a bf16=$bf16 2d=$ar2d" \
        | tee -a "$SUMMARY"
    if ! (
        cd "$case_dir"
        TOFU_TOPO_PATH="$TOPO" COUNT="$AR_COUNT" REPS="$AR_REPS" \
        TP_AR_ACK="$ack" TP_AR_DROP="$drop" TP_AR_DETERMINISTIC="$deterministic" \
        TP_AR_A2A="$a2a" TP_AR_BF16="$bf16" TP_AR_2D="$ar2d" \
        mpiexec -np "$NODES" -of-proc "$prefix" "$AR_TEST" >launcher.log 2>&1
    ); then
        echo "allreduce FAIL case=$name; rank logs are in $case_dir" >&2
        exit 11
    fi
    if ! grep -hF 'RESULT: PASS' "$case_dir/tp_ar_ack_result.txt" >"$case_dir/result.log"; then
        echo "allreduce FAIL case=$name; result is in $case_dir" >&2
        exit 12
    fi
    echo "allreduce PASS case=$name" | tee -a "$SUMMARY"
}

if [ "$RUN_BARRIER" -eq 1 ]; then
    # Separate launches cover registration/cleanup boundaries as well as a
    # long run. The rank skew is generated inside k3_full_runner.
    run_barrier_case startup 1
    run_barrier_case short 8
    run_barrier_case normal "$BARRIER_ITERS"
    run_barrier_case long "$LONG_BARRIER_ITERS"
fi

if [ "$RUN_AR" -eq 1 ]; then
    # Every case checks exact integer SUM and MAX results. These cases select
    # independent transport/reduction paths without loading model weights.
    run_ar_case flat 0 0 0 0 0 0
    run_ar_case deterministic 0 0 1 0 0 0
    run_ar_case a2a 0 0 0 1 0 0
    run_ar_case bf16 0 0 0 0 1 0
    run_ar_case two-level 0 0 0 0 0 3
    run_ar_case ack-drop 1 "$DROP_EVERY" 0 0 0 0
fi

echo "K3_12N_REGRESSION PASS nodes=$NODES root=$ROOT" | tee -a "$SUMMARY"
