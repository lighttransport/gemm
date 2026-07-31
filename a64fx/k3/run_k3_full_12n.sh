#!/bin/bash
# Direct interactive 12-node K3 C11 debug harness.  This script deliberately
# does not contain PJM directives and never submits a batch job.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
UTOFU="$REPO/a64fx/utofu-tests"
MODEL_DIR=${K3_MODEL_DIR:-${HOME}/models/kimi-k3}
MODE=layer12
LAYER_INDEX=1
THREADS=${K3_THREADS:-48}
PREFILL_TOKENS=32
NEW_TOKENS=0
PREFILL_CHUNK=1
INPUT_SEED=0x4b33444542554701
JOB_TAG=${PJM_JOBID:-manual-$$}
ROOT="$SCRIPT_DIR/logs/full-debug-12n-$JOB_TAG"
STAGE_DIR="/local/$USER/k3-full-debug-12n-$JOB_TAG"
PROMPT_IDS=

usage() {
    cat >&2 <<EOF
usage: $0 [options]
  --mode layer12|synthetic12
  --layer-index N       checkpoint layer index, 0..92 (default: 1)
  --model-dir DIR
  --result-dir DIR
  --stage-dir DIR
  --prompt-ids FILE
  --prefill-tokens N
  --new-tokens N
  --prefill-chunk N
  --threads N
  --input-seed N
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --mode) MODE=$2; shift 2;;
        --layer-index) LAYER_INDEX=$2; shift 2;;
        --model-dir) MODEL_DIR=$2; shift 2;;
        --result-dir) ROOT=$2; shift 2;;
        --stage-dir) STAGE_DIR=$2; shift 2;;
        --prompt-ids) PROMPT_IDS=$2; shift 2;;
        --prefill-tokens) PREFILL_TOKENS=$2; shift 2;;
        --new-tokens) NEW_TOKENS=$2; shift 2;;
        --prefill-chunk) PREFILL_CHUNK=$2; shift 2;;
        --threads) THREADS=$2; shift 2;;
        --input-seed) INPUT_SEED=$2; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "$0: unknown option $1" >&2; usage; exit 2;;
    esac
done

case "$MODE" in layer12|synthetic12) ;; *) echo "$0: invalid --mode $MODE" >&2; exit 2;; esac
case "$LAYER_INDEX" in ''|*[!0-9]*) echo "$0: invalid --layer-index" >&2; exit 2;; esac
if [ "$LAYER_INDEX" -gt 92 ]; then echo "$0: layer index must be 0..92" >&2; exit 2; fi
ROOT=$(realpath -m "$ROOT")
if [ -n "${PJM_MPI_PROC:-}" ] && [ "$PJM_MPI_PROC" -ne 12 ]; then
    echo "$0: this harness requires the current allocation to expose 12 MPI processes" >&2
    exit 3
fi
if [ -e "$ROOT" ]; then echo "$0: result root already exists: $ROOT" >&2; exit 2; fi
if [ ! -d "$MODEL_DIR" ]; then echo "$0: missing model directory: $MODEL_DIR" >&2; exit 4; fi

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
export OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
mkdir -p "$ROOT"

echo "K3_FULL_DEBUG_BEGIN mode=$MODE nodes=12 layer_index=$LAYER_INDEX root=$ROOT"
make -C "$SCRIPT_DIR" full-runner >/dev/null
make -C "$UTOFU" tofu_topo_helper >/dev/null

cd "$ROOT"
mpiexec -np 12 "$UTOFU/tofu_topo_helper" >topology.log
mv tofu_topo.txt topology.txt
if [ "$(grep -vc '^#' topology.txt)" -ne 12 ]; then
    echo "$0: topology discovery did not produce 12 ranks" >&2
    exit 4
fi

mkdir -p "$STAGE_DIR"
mpiexec -np 12 -of-proc "$ROOT/stage.rank" sh -c \
    "exec '$SCRIPT_DIR/run_k3_full_stage_rank.sh' '$MODEL_DIR' '$STAGE_DIR' 12 \${PMIX_RANK:-\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:?no MPI rank}}} layer12 '$LAYER_INDEX'"

args=(--mode "$MODE" --nodes 12 --threads "$THREADS" --stage-dir "$STAGE_DIR"
      --topo "$ROOT/topology.txt" --output "$ROOT/output.txt"
      --real-layer-index "$LAYER_INDEX" --input-seed "$INPUT_SEED"
      --prefill-tokens "$PREFILL_TOKENS" --new-tokens "$NEW_TOKENS"
      --prefill-chunk "$PREFILL_CHUNK" --max-seq 4096)
if [ -n "$PROMPT_IDS" ]; then args+=(--prompt-ids "$PROMPT_IDS"); fi

set +e
mpiexec -np 12 -of-proc "$ROOT/run.rank" "$SCRIPT_DIR/k3_full_runner" "${args[@]}"
runner_rc=$?
set -e
if [ "$runner_rc" -ne 0 ]; then
    echo "K3_FULL_DEBUG status=FAIL rc=$runner_rc root=$ROOT" >&2
    exit "$runner_rc"
fi

python3 "$SCRIPT_DIR/validate_k3_full_output.py" "$ROOT/output.txt" \
    --nodes 12 --mode "$MODE" | tee "$ROOT/validation.txt"
printf 'K3_FULL_DEBUG status=PASS mode=%s layer_index=%s output=%s stage_dir=%s\n' \
    "$MODE" "$LAYER_INDEX" "$ROOT/output.txt" "$STAGE_DIR"
