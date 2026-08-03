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
BARRIER_ITERS=${K3_BARRIER_ITERS:-128}
COMM_DETERMINISTIC=${K3_COMM_DETERMINISTIC:-1}
COMM_BF16=${K3_COMM_BF16:-0}
COMM_ROBUST=${K3_COMM_ROBUST:-2}
COMM_POLL_SPINS=${K3_COMM_POLL_SPINS:-4}
COMM_A2A=${K3_COMM_A2A:-0}
COMM_A2A_MAX=${K3_COMM_A2A_MAX:-8192}
PREFETCH_MIB=${K3_PREFETCH_MIB:-16}
PROFILE=${K3_PROFILE:-0}
PREFILL_TOKENS=32
NEW_TOKENS=0
PREFILL_CHUNK=1
AR_GROUPS=${K3_AR_GROUPS:-2}
EXPERT_TP=${K3_EXPERT_TP:-0}
MOE_SHARD_LAYOUT=${K3_MOE_SHARD_LAYOUT:-replicated}
Q8_MODE=0
Q8_QUALITY_GATE=${K3_Q8_QUALITY_GATE:-1}
Q8_STAGE_DIR=
INPUT_SEED=0x4b33444542554701
JOB_TAG=${PJM_JOBID:-manual-$$}
ROOT="$SCRIPT_DIR/logs/full-debug-12n-$JOB_TAG"
STAGE_DIR="/local/$USER/k3-full-debug-12n-$JOB_TAG"
PROMPT_IDS=

usage() {
    cat >&2 <<EOF
usage: $0 [options]
  --mode layer12|synthetic12
  --barrier-iters N      uTofu barrier preflight iterations (default: 128)
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
  --expert-tp            stage every expert's TP slice for the fused-MoE probe
  --q8                   quality-gated Q8W16-convert the staged probe image
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --mode) MODE=$2; shift 2;;
        --barrier-iters) BARRIER_ITERS=$2; shift 2;;
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
        --ar-groups) AR_GROUPS=$2; shift 2;;
        --expert-tp) EXPERT_TP=1; shift;;
        --q8) Q8_MODE=1; shift;;
        -h|--help) usage; exit 0;;
        *) echo "$0: unknown option $1" >&2; usage; exit 2;;
    esac
done

case "$MODE" in layer12|synthetic12) ;; *) echo "$0: invalid --mode $MODE" >&2; exit 2;; esac
case "$BARRIER_ITERS" in ''|*[!0-9]*) echo "$0: invalid --barrier-iters" >&2; exit 2;; esac
if [ "$BARRIER_ITERS" -lt 1 ] || [ "$BARRIER_ITERS" -gt 100000 ]; then
    echo "$0: --barrier-iters must be in 1..100000" >&2
    exit 2
fi
case "$LAYER_INDEX" in ''|*[!0-9]*) echo "$0: invalid --layer-index" >&2; exit 2;; esac
if [ "$LAYER_INDEX" -gt 92 ]; then echo "$0: layer index must be 0..92" >&2; exit 2; fi
ROOT=$(realpath -m "$ROOT")
if [ -n "${PJM_MPI_PROC:-}" ] && [ "$PJM_MPI_PROC" -ne 12 ]; then
    echo "$0: this harness requires the current allocation to expose 12 MPI processes" >&2
    exit 3
fi
if [ -e "$ROOT" ]; then echo "$0: result root already exists: $ROOT" >&2; exit 2; fi
if [ ! -d "$MODEL_DIR" ]; then echo "$0: missing model directory: $MODEL_DIR" >&2; exit 4; fi
if [ "$Q8_MODE" -eq 1 ] && [ "$EXPERT_TP" -ne 1 ]; then
    echo "$0: --q8 requires --expert-tp" >&2
    exit 2
fi
if [ "$Q8_MODE" -eq 1 ]; then
    Q8_STAGE_DIR=${K3_Q8_STAGE_DIR:-${STAGE_DIR}-q8}
fi
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
export OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
mkdir -p "$ROOT"
"$SCRIPT_DIR/k3_setup_python.sh"

echo "K3_FULL_DEBUG_BEGIN mode=$MODE nodes=12 layer_index=$LAYER_INDEX root=$ROOT expert_tp=$EXPERT_TP"
echo "K3_FULL_CONFIG threads=$THREADS barrier_iters=$BARRIER_ITERS comm_deterministic=$COMM_DETERMINISTIC comm_bf16=$COMM_BF16 comm_robust=$COMM_ROBUST comm_poll_spins=$COMM_POLL_SPINS comm_a2a=$COMM_A2A comm_a2a_max=$COMM_A2A_MAX prefetch_mib=$PREFETCH_MIB ar_groups=$AR_GROUPS profile=$PROFILE moe_shard_layout=$MOE_SHARD_LAYOUT"
make -C "$SCRIPT_DIR" full-runner >/dev/null
make -C "$UTOFU" tofu_topo_helper >/dev/null

cd "$ROOT"
mpiexec -np 12 "$UTOFU/tofu_topo_helper" >topology.log
mv tofu_topo.txt topology.txt
if [ "$(grep -vc '^#' topology.txt)" -ne 12 ]; then
    echo "$0: topology discovery did not produce 12 ranks" >&2
    exit 4
fi

echo "K3_FULL_BARRIER_BEGIN nodes=12 iterations=$BARRIER_ITERS"
BARRIER_PREFIX="$ROOT/barrier.rank"
mpiexec -np 12 -of-proc "$BARRIER_PREFIX" "$SCRIPT_DIR/k3_full_runner" \
    --mode barrier --nodes 12 --topo "$ROOT/topology.txt" \
    --barrier-iters "$BARRIER_ITERS"
grep -h 'K3FULL_BARRIER' "$BARRIER_PREFIX".* >"$ROOT/barrier.log"
echo "K3_FULL_BARRIER_END status=PASS nodes=12 iterations=$BARRIER_ITERS"

mkdir -p "$STAGE_DIR"
if [ -s "$STAGE_DIR/rank000.manifest" ]; then
    echo "K3_FULL_REUSE_STAGE dir=$STAGE_DIR"
else
    mpiexec -np 12 -of-proc "$ROOT/stage.rank" sh -c \
        "K3_EXPERT_TP=$EXPERT_TP K3_MOE_SHARD_LAYOUT=$MOE_SHARD_LAYOUT exec '$SCRIPT_DIR/run_k3_full_stage_rank.sh' '$MODEL_DIR' '$STAGE_DIR' 12 \${PMIX_RANK:-\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:?no MPI rank}}} layer12 '$LAYER_INDEX'"
fi

if [ "$Q8_MODE" -eq 1 ]; then
    mkdir -p "$Q8_STAGE_DIR"
    make -C "$SCRIPT_DIR" full-convert >/dev/null
    Q8_QUALITY_FLAG=
    if [ "$Q8_QUALITY_GATE" -eq 1 ]; then Q8_QUALITY_FLAG=--quality-gate; fi
    mpiexec -np 12 -of-proc "$ROOT/convert.rank" sh -c \
        "rank=\${PMIX_RANK:-\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:?no rank}}}; blob='$Q8_STAGE_DIR'/rank\$(printf '%03d' \$rank).blob; manifest='$Q8_STAGE_DIR'/rank\$(printf '%03d' \$rank).manifest; if [ -s \$blob ] && [ -s \$manifest ]; then echo K3_Q8_REUSE rank=\$rank; else '$SCRIPT_DIR/k3_full_convert' --input-dir '$STAGE_DIR' --output-dir '$Q8_STAGE_DIR' --rank \$rank --nodes 12 --mode mixed-q8w16-expert-tp-$MOE_SHARD_LAYOUT --force $Q8_QUALITY_FLAG; fi"
    STAGE_DIR=$Q8_STAGE_DIR
fi

args=(--mode "$MODE" --nodes 12 --threads "$THREADS" --stage-dir "$STAGE_DIR"
      --topo "$ROOT/topology.txt" --output "$ROOT/output.txt"
      --real-layer-index "$LAYER_INDEX" --input-seed "$INPUT_SEED"
      --prefill-tokens "$PREFILL_TOKENS" --new-tokens "$NEW_TOKENS"
      --prefill-chunk "$PREFILL_CHUNK" --max-seq 4096
      --comm-deterministic "$COMM_DETERMINISTIC" --comm-bf16 "$COMM_BF16"
      --comm-robust "$COMM_ROBUST" --comm-poll-spins "$COMM_POLL_SPINS"
      --comm-a2a "$COMM_A2A" --comm-a2a-max "$COMM_A2A_MAX"
      --prefetch-mib "$PREFETCH_MIB"
      --ar-groups "$AR_GROUPS")
if [ "$PROFILE" -ne 0 ]; then args+=(--profile "$ROOT/profile.txt"); fi
if [ -n "$PROMPT_IDS" ]; then args+=(--prompt-ids "$PROMPT_IDS"); fi

set +e
mpiexec -np 12 -of-proc "$ROOT/run.rank" "$SCRIPT_DIR/k3_full_runner" "${args[@]}"
runner_rc=$?
set -e
if [ "$runner_rc" -ne 0 ]; then
    echo "K3_FULL_DEBUG status=FAIL rc=$runner_rc root=$ROOT" >&2
    exit "$runner_rc"
fi

"$SCRIPT_DIR/k3_python.sh" "$SCRIPT_DIR/validate_k3_full_output.py" "$ROOT/output.txt" \
    --nodes 12 --mode "$MODE" | tee "$ROOT/validation.txt"
printf 'K3_FULL_DEBUG status=PASS mode=%s layer_index=%s output=%s stage_dir=%s\n' \
    "$MODE" "$LAYER_INDEX" "$ROOT/output.txt" "$STAGE_DIR"
