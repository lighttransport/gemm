#!/bin/bash
# Live 12-node validation of rank-local IQ intermediate-channel slices.
# This intentionally performs no job submission and does not run the full
# graph yet; it validates compressed TP ownership before MPI integration.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
FORMAT=${K3_IQ_FORMAT:-iq1}
MODEL_DIR=${K3_IQ_MODEL_DIR:-}
LAYER=${K3_IQ_LAYER:-1}
EXPERTS=${K3_IQ_TP_EXPERTS:-all}
REPS=${K3_IQ_TP_REPS:-1}
THREADS=${K3_IQ_THREADS:-}
WARMUP=${K3_IQ_TP_WARMUP:-8}
TOPO=${K3_IQ_TOPO:-}
UTOFU=${K3_UTOFU_DIR:-$SCRIPT_DIR/../utofu-tests}
JOB_TAG=${PJM_JOBID:-manual-$$}
STAGE_DIR=${K3_IQ_TP_STAGE_DIR:-/local/$USER/k3-${FORMAT}-tp12-${JOB_TAG}}
RESULT_DIR=${K3_IQ_TP_RESULT_DIR:-$SCRIPT_DIR/logs/iq-${FORMAT}-tp12-${JOB_TAG}}

usage() { echo "usage: $0 [--format iq1|q2] [--model-dir DIR] [--layer N] [--experts LIST]" >&2; }
while (($#)); do
    case "$1" in
        --format) FORMAT=$2; shift 2;;
        --model-dir) MODEL_DIR=$2; shift 2;;
        --layer) LAYER=$2; shift 2;;
        --experts) EXPERTS=$2; shift 2;;
        --reps) REPS=$2; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "$0: unknown option $1" >&2; usage; exit 2;;
    esac
done
case "$FORMAT" in
    # IQ1's packed 16-row path benefits from four additional compute threads;
    # 44 was repeatable at 0.869 ms/token on the live 12-node stage.
    iq1) DEFAULT_MODEL="$HOME/models/k3/iq1"; THREADS=${THREADS:-44};;
    q2) DEFAULT_MODEL="$HOME/models/k3/q2"; THREADS=${THREADS:-36};;
    *) echo "$0: format must be iq1 or q2" >&2; exit 2;;
esac
MODEL_DIR=${MODEL_DIR:-$DEFAULT_MODEL}
[[ -d "$MODEL_DIR" ]] || { echo "$0: missing model directory $MODEL_DIR" >&2; exit 4; }
if [[ -n "${PJM_MPI_PROC:-}" && "$PJM_MPI_PROC" -ne 12 ]]; then
    echo "$0: requires PJM_MPI_PROC=12" >&2; exit 3
fi
if [[ -e "$RESULT_DIR" ]]; then
    echo "$0: result directory exists: $RESULT_DIR" >&2; exit 2
fi
mkdir -p "$RESULT_DIR" "$STAGE_DIR"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
export OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores
export OMP_WAIT_POLICY=active KMP_BLOCKTIME=infinite

if [[ -z "$TOPO" ]]; then
    TOPO="$RESULT_DIR/topology.txt"
    make -C "$UTOFU" tofu_topo_helper >/dev/null
    (cd "$RESULT_DIR" && mpiexec -np 12 "$UTOFU/tofu_topo_helper" >topology.log)
    mv "$RESULT_DIR/tofu_topo.txt" "$TOPO"
    [[ "$(grep -vc '^#' "$TOPO")" -eq 12 ]] || {
        echo "$0: topology discovery did not produce 12 ranks" >&2; exit 3;
    }
fi

echo "K3_IQTP12_BEGIN format=$FORMAT layer=$LAYER experts=$EXPERTS reps=$REPS"
mpiexec -np 12 -of-proc "$RESULT_DIR/stage" sh -c \
    "r=\${PMIX_RANK:-\${OMPI_COMM_WORLD_RANK:-\${PJM_MPI_RANK:?no rank}}}; \
     exec python3 '$SCRIPT_DIR/k3_gguf_expert_tp_stage.py' --model-dir '$MODEL_DIR' \
       --output-dir '$STAGE_DIR' --rank \$r --nodes 12 --layer '$LAYER' \
       --experts '$EXPERTS' --format '$FORMAT' --force"

make -C "$SCRIPT_DIR" k3_ep_runner >/dev/null
mpiexec -np 12 -of-proc "$RESULT_DIR/runner" sh -c \
    "exec '$SCRIPT_DIR/k3_ep_runner' --nodes 12 --tp-nodes 12 \
       --layers 1 --tokens '$REPS' --layer '$LAYER' --threads '$THREADS' \
       --ar-groups auto --topo '$TOPO' --status-dir '$RESULT_DIR' \
       --iq-stage-dir '$STAGE_DIR' --iq-warmup-tokens '$WARMUP' --profile"

cat "$RESULT_DIR"/runner.* >"$RESULT_DIR/runner.log"
count=$(grep -c 'K3_IQTP_RUN .*status=PASS' "$RESULT_DIR/runner.log" || true)
[[ "$count" -eq 12 ]] || { echo "K3_IQTP12_FAIL pass_ranks=$count/12" >&2; exit 5; }
echo "K3_IQTP12_RESULT format=$FORMAT stage=$STAGE_DIR result=$RESULT_DIR pass_ranks=$count/12"
cat "$RESULT_DIR/runner.log"
