#!/bin/bash
# Real IQ1/IQ2 layer measurement on the current interactive 12-node job.
# This deliberately performs no submission and streams each rank's GGUF slice.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
FORMAT=${K3_IQ_FORMAT:-iq1}
MODEL_DIR=${K3_IQ_MODEL_DIR:-}
LAYER=${K3_IQ_LAYER:-1}
THREADS=${K3_IQ_THREADS:-}
REPS=${K3_IQ_REPS:-10}
PACKED=${K3_QUANT_PACKED:-1}
NIBBLE=${K3_QUANT_PACKED_NIBBLE:-1}
KERNEL=${K3_QUANT_KERNEL:-sve-a16}
JOB_TAG=${PJM_JOBID:-manual-$$}
STAGE_DIR=${K3_IQ_STAGE_DIR:-/local/$USER/k3-${FORMAT}-12n-${JOB_TAG}}
RESULT_DIR=${K3_IQ_RESULT_DIR:-$SCRIPT_DIR/logs/iq-${FORMAT}-12n-${JOB_TAG}}

usage() { echo "usage: $0 [--format iq1|q2] [--model-dir DIR] [--layer N] [--threads N] [--reps N]" >&2; }
while (($#)); do
    case "$1" in
        --format) FORMAT=$2; shift 2;;
        --model-dir) MODEL_DIR=$2; shift 2;;
        --layer) LAYER=$2; shift 2;;
        --threads) THREADS=$2; shift 2;;
        --reps) REPS=$2; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "$0: unknown option $1" >&2; usage; exit 2;;
    esac
done
case "$FORMAT" in
    iq1) DEFAULT_MODEL="$HOME/models/k3/iq1"; THREADS=${THREADS:-40};;
    q2) DEFAULT_MODEL="$HOME/models/k3/q2"; THREADS=${THREADS:-36};;
    *) echo "$0: format must be iq1 or q2" >&2; exit 2;;
esac
MODEL_DIR=${MODEL_DIR:-$DEFAULT_MODEL}
[[ "$LAYER" =~ ^[0-9]+$ && "$THREADS" =~ ^[0-9]+$ && "$REPS" =~ ^[0-9]+$ ]] || { echo "$0: numeric options must be integers" >&2; exit 2; }
if [[ -n "${PJM_MPI_PROC:-}" && "$PJM_MPI_PROC" -ne 12 ]]; then echo "$0: requires PJM_MPI_PROC=12" >&2; exit 3; fi
[[ -d "$MODEL_DIR" ]] || { echo "$0: missing model directory $MODEL_DIR" >&2; exit 4; }
if [[ -e "$RESULT_DIR" ]]; then echo "$0: result directory exists: $RESULT_DIR" >&2; exit 2; fi
mkdir -p "$RESULT_DIR"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
export OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores
export OMP_WAIT_POLICY=active KMP_BLOCKTIME=infinite

echo "K3_IQ12_BEGIN format=$FORMAT layer=$LAYER nodes=12 threads=$THREADS reps=$REPS"
mkdir -p "$STAGE_DIR"
mpiexec -np 12 -of-proc "$RESULT_DIR/stage" sh -c \
    "r=\${PMIX_RANK:-\${OMPI_COMM_WORLD_RANK:-\${PJM_MPI_RANK:?no rank}}}; exec python3 '$SCRIPT_DIR/k3_gguf_stage.py' --model-dir '$MODEL_DIR' --output-dir '$STAGE_DIR' --rank \$r --nodes 12 --layer-index '$LAYER' --format '$FORMAT' --no-global --force"

make -C "$SCRIPT_DIR" k3_gguf_layer_bench >/dev/null
mpiexec -np 12 -of-proc "$RESULT_DIR/bench" sh -c \
    "r=\${PMIX_RANK:-\${OMPI_COMM_WORLD_RANK:-\${PJM_MPI_RANK:?no rank}}}; \
     export OMP_NUM_THREADS='$THREADS' OMP_PROC_BIND=close OMP_PLACES=cores K3_QUANT_KERNEL='$KERNEL' \
            K3_QUANT_PACKED='$PACKED' K3_QUANT_PACKED_NIBBLE='$NIBBLE'; \
     exec '$SCRIPT_DIR/k3_gguf_layer_bench' '$STAGE_DIR'/rank\$(printf '%03d' \$r).manifest \
          '$STAGE_DIR'/rank\$(printf '%03d' \$r).blob '$REPS'"

cat "$RESULT_DIR"/bench.* >"$RESULT_DIR/bench.log"
grep 'K3_QBENCH_LAYER' "$RESULT_DIR/bench.log" | sort -k6,6n >"$RESULT_DIR/layer-summary.txt"
echo "K3_IQ12_RESULT format=$FORMAT stage=$STAGE_DIR result=$RESULT_DIR"
cat "$RESULT_DIR/layer-summary.txt"
