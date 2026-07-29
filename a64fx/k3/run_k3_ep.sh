#!/bin/bash
# Run inside a PJM allocation. Operational settings are CLI-only; environment
# is reserved for PJM rank discovery and the OpenMP/XOS runtime.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
UTOFU="$REPO/a64fx/utofu-tests"
MODE=dummy
NODES=${PJM_MPI_PROC:-96}
LAYERS=1
TOKENS=2
THREADS=48
LAYER=1
EXPERTS=0-15
CHUNK_MIB=8
MODEL_DIR="$HOME/models/kimi-k3"
JOB_TAG=${PJM_JOBID:-manual-$$}
STAGE_DIR="/local/$USER/k3-runner-$JOB_TAG"
RESULT_DIR="$SCRIPT_DIR/logs/run-$JOB_TAG"

usage() {
    cat >&2 <<EOF
usage: $0 [--mode dummy|real] [--nodes N] [--layers N] [--tokens N]
          [--threads N] [--layer N] [--experts LIST] [--chunk-mib N]
          [--model-dir DIR] [--stage-dir DIR] [--result-dir DIR]
EOF
}
need_value() { if (( $# < 2 )); then echo "$0: missing value for $1" >&2; usage; exit 2; fi; }
while (( $# )); do
    case "$1" in
        --mode) need_value "$@"; MODE=$2; shift 2;;
        --nodes) need_value "$@"; NODES=$2; shift 2;;
        --layers) need_value "$@"; LAYERS=$2; shift 2;;
        --tokens) need_value "$@"; TOKENS=$2; shift 2;;
        --threads) need_value "$@"; THREADS=$2; shift 2;;
        --layer) need_value "$@"; LAYER=$2; shift 2;;
        --experts) need_value "$@"; EXPERTS=$2; shift 2;;
        --chunk-mib) need_value "$@"; CHUNK_MIB=$2; shift 2;;
        --model-dir) need_value "$@"; MODEL_DIR=$2; shift 2;;
        --stage-dir) need_value "$@"; STAGE_DIR=$2; shift 2;;
        --result-dir) need_value "$@"; RESULT_DIR=$2; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "$0: unknown argument: $1" >&2; usage; exit 2;;
    esac
done
case "$MODE" in dummy|real) ;; *) echo "$0: --mode must be dummy or real" >&2; exit 2;; esac
for value in "$NODES" "$LAYERS" "$TOKENS" "$THREADS" "$LAYER" "$CHUNK_MIB"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "$0: numeric options must be integers" >&2; exit 2; }
done
(( NODES > 0 && LAYERS > 0 && TOKENS > 0 && THREADS > 0 && THREADS <= 48 && CHUNK_MIB > 0 )) || {
    echo "$0: invalid numeric option range" >&2; exit 2; }
(( 3072 % NODES == 0 && (3072 / NODES) % 32 == 0 )) || {
    echo "$0: node count $NODES does not preserve native 32-channel MXFP4 groups (it must divide 96)" >&2; exit 2; }
if [[ -n "${PJM_MPI_PROC:-}" && "$NODES" -ne "$PJM_MPI_PROC" ]]; then
    echo "$0: --nodes $NODES differs from allocation process count $PJM_MPI_PROC" >&2
    exit 2
fi
if [[ -e "$RESULT_DIR" ]]; then echo "$0: result directory already exists: $RESULT_DIR" >&2; exit 2; fi
mkdir -p "$RESULT_DIR"

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
export OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
make -C "$UTOFU" tofu_topo_helper >/dev/null
make -C "$SCRIPT_DIR" runner >/dev/null

cd "$RESULT_DIR"
topology_ok=0
for attempt in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    if mpiexec -np "$NODES" "$UTOFU/tofu_topo_helper" &&
       [[ $(grep -vc '^#' tofu_topo.txt 2>/dev/null || true) -eq "$NODES" ]]; then
        topology_ok=1
        break
    fi
    echo "topology discovery attempt $attempt/5 failed" >&2
    sleep 2
done
(( topology_ok == 1 )) || { echo "$0: topology discovery failed" >&2; exit 3; }

if [[ "$MODE" == real ]]; then
    if [[ ! -d "$MODEL_DIR" ]]; then echo "$0: model directory is missing: $MODEL_DIR" >&2; exit 4; fi
    mpiexec -np "$NODES" -of-proc "$RESULT_DIR/stage" \
        "$SCRIPT_DIR/run_k3_stage_rank.sh" "$SCRIPT_DIR" "$NODES" "$MODEL_DIR" \
        "$STAGE_DIR" "$LAYER" "$EXPERTS" "$CHUNK_MIB"
    staged=$(find "$RESULT_DIR" -maxdepth 1 -name 'stage.*' -type f | wc -l)
    echo "stage launch output files: $staged/$NODES"
fi

set +e
mpiexec -np "$NODES" -of-proc "$RESULT_DIR/rank" \
    "$SCRIPT_DIR/k3_ep_runner" --mode "$MODE" --nodes "$NODES" \
    --layers "$LAYERS" --tokens "$TOKENS" --threads "$THREADS" --layer "$LAYER" \
    --stage-dir "$STAGE_DIR" --status-dir "$RESULT_DIR" --topo "$RESULT_DIR/tofu_topo.txt"
runner_rc=$?
set -e

passes=$(grep -l 'state=pass' "$RESULT_DIR"/k3_rank*.status 2>/dev/null | wc -l || true)
grep -hE 'K3_RUN|K3_RESULT|FATAL|timeout|failed' "$RESULT_DIR"/rank.* 2>/dev/null || true
echo "K3 distributed result: rc=$runner_rc pass_markers=$passes/$NODES results=$RESULT_DIR"

# Rank-local storage is job-scoped and is wiped by the scheduler. Deliberately
# leave it untouched here: automatic recursive cleanup of a caller-supplied
# --stage-dir is unsafe, and retaining it helps diagnose a failed run.
(( runner_rc == 0 && passes == NODES )) || exit 5
