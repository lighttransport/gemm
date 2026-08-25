#!/bin/bash
# Native full KDA+MoE layer validation for IQ1/IQ2 in the current 12-node job.
# This script never submits a batch job.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
UTOFU="$REPO/a64fx/utofu-tests"
FORMAT=${K3_IQ_FORMAT:-iq1}
MODEL_DIR=${K3_IQ_MODEL_DIR:-}
LAYER=${K3_IQ_LAYER:-1}
THREADS=${K3_IQ_THREADS:-}
TOKENS=${K3_IQ_TOKENS:-64}
JOB_TAG=${PJM_JOBID:-manual-$$}
STAGE_DIR=${K3_IQ_STAGE_DIR:-/local/$USER/k3-${FORMAT}-native-etp-layer12-${JOB_TAG}}
RESULT_DIR=${K3_IQ_RESULT_DIR:-$SCRIPT_DIR/logs/iq-${FORMAT}-native-etp-layer12-${JOB_TAG}}

usage() {
    echo "usage: $0 [--format iq1|q2] [--model-dir DIR] [--layer N] [--threads N] [--tokens N]" >&2
}
while (($#)); do
    case "$1" in
        --format) FORMAT=$2; shift 2;;
        --model-dir) MODEL_DIR=$2; shift 2;;
        --layer) LAYER=$2; shift 2;;
        --threads) THREADS=$2; shift 2;;
        --tokens) TOKENS=$2; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "$0: unknown option $1" >&2; usage; exit 2;;
    esac
done
case "$FORMAT" in
    iq1) MODEL_DIR=${MODEL_DIR:-$HOME/models/k3/iq1}; THREADS=${THREADS:-48};;
    q2) MODEL_DIR=${MODEL_DIR:-$HOME/models/k3/q2}; THREADS=${THREADS:-40};;
    *) echo "$0: format must be iq1 or q2" >&2; exit 2;;
esac
[[ "$LAYER" =~ ^[0-9]+$ && "$THREADS" =~ ^[0-9]+$ && "$TOKENS" =~ ^[0-9]+$ ]] || {
    echo "$0: numeric options must be integers" >&2; exit 2;
}
if [[ "$LAYER" -eq 0 ]]; then
    echo "$0: expert-TP IQ validation requires a MoE layer (layer > 0)" >&2
    exit 2
fi
[[ -d "$MODEL_DIR" ]] || { echo "$0: missing model directory $MODEL_DIR" >&2; exit 4; }
if [[ -e "$RESULT_DIR" ]]; then echo "$0: result directory exists: $RESULT_DIR" >&2; exit 2; fi

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
export OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores
export OMP_WAIT_POLICY=active KMP_BLOCKTIME=infinite
export K3_PREFILL_PROJ_THREADS=${K3_PREFILL_PROJ_THREADS:-48}
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
mkdir -p "$RESULT_DIR" "$STAGE_DIR"
"$SCRIPT_DIR/k3_setup_python.sh"
make -C "$SCRIPT_DIR" full-runner >/dev/null
make -C "$UTOFU" tofu_topo_helper >/dev/null

cd "$RESULT_DIR"
mpiexec -np 12 -of-proc "$RESULT_DIR/topology.rank" "$UTOFU/tofu_topo_helper"
mv tofu_topo.txt topology.txt
[[ $(grep -vc '^#' topology.txt) -eq 12 ]] || {
    echo "$0: topology discovery did not return 12 ranks" >&2; exit 4;
}

mpiexec -np 12 -of-proc "$RESULT_DIR/stage.rank" sh -c \
    "rank=\${PMIX_RANK:-\${OMPI_COMM_WORLD_RANK:-\${PJM_MPI_RANK:-\${PMI_RANK:?no rank}}}}; \
     exec '$SCRIPT_DIR/k3_python.sh' '$SCRIPT_DIR/k3_gguf_native_stage.py' \
          '$MODEL_DIR' --nodes 12 --rank \"\$rank\" --layer '$LAYER' \
          --output-dir '$STAGE_DIR' --expert-tp --force"

mpiexec -np 12 -of-proc "$RESULT_DIR/run.rank" "$SCRIPT_DIR/k3_full_runner" \
    --mode layer12 --nodes 12 --threads "$THREADS" --stage-dir "$STAGE_DIR" \
    --topo "$RESULT_DIR/topology.txt" --output "$RESULT_DIR/output.txt" \
    --profile "$RESULT_DIR/profile.txt" --real-layer-index "$LAYER" \
    --input-seed 0x4b33444542554701 --prefill-tokens "$TOKENS" \
    --new-tokens 0 --prefill-chunk 1 --max-seq 4096 \
    --comm-deterministic 1 --comm-bf16 0 --comm-robust 2 \
    --comm-poll-spins 4 --comm-a2a 0 --comm-a2a-max 8192 \
    --prefetch-mib 0 --ar-groups 2

"$SCRIPT_DIR/k3_python.sh" "$SCRIPT_DIR/validate_k3_full_output.py" \
    "$RESULT_DIR/output.txt" --nodes 12 --mode layer12 | tee "$RESULT_DIR/validation.txt"
grep -E '^(layer=|layer_ms_rank_max|phase=)' "$RESULT_DIR/profile.txt" >"$RESULT_DIR/profile-summary.txt"
echo "K3_IQ_FULL_LAYER PASS format=$FORMAT layer=$LAYER threads=$THREADS tokens=$TOKENS stage=$STAGE_DIR result=$RESULT_DIR"
grep '^layer=' "$RESULT_DIR/profile-summary.txt" | grep "layer=$LAYER "
