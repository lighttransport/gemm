#!/bin/bash
# Prepare the durable full-96 mixed Q8W16 artifact from a 12-node allocation.
# Each physical rank owns eight logical rank images and processes one tensor at
# a time, so no 24-GiB checkpoint copy or whole-model conversion is performed.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
MODEL_DIR=${K3_MODEL_DIR:-${HOME}/models/kimi-k3}
CHECKPOINT_ID=${K3_CHECKPOINT_ID:-$(basename "$MODEL_DIR")}
ARTIFACT_ROOT=${K3_Q8_ARTIFACT_ROOT:-$REPO/artifacts/k3-full96-q8w16/$CHECKPOINT_ID}
ARTIFACT_ROOT_SET=0
NODES=96
PHYSICAL_NODES=12
CHUNK_MIB=${K3_STAGE_CHUNK_MIB:-8}
QUALITY_GATE=${K3_Q8_QUALITY_GATE:-1}
FORCE=0
CONVERT_THREADS=${K3_CONVERT_THREADS:-8}
MOE_SHARD_LAYOUT=${K3_MOE_SHARD_LAYOUT:-replicated}

usage() {
    cat >&2 <<EOF
usage: $0 [--model-dir DIR] [--artifact-root DIR] [--checkpoint-id ID]
          [--chunk-mib N] [--moe-shard-layout replicated|row-aligned]
          [--no-quality-gate] [--force]
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model-dir) MODEL_DIR=$2; shift 2;;
        --artifact-root) ARTIFACT_ROOT=$2; ARTIFACT_ROOT_SET=1; shift 2;;
        --checkpoint-id) CHECKPOINT_ID=$2; shift 2;;
        --chunk-mib) CHUNK_MIB=$2; shift 2;;
        --moe-shard-layout) MOE_SHARD_LAYOUT=$2; shift 2;;
        --no-quality-gate) QUALITY_GATE=0; shift;;
        --force) FORCE=1; shift;;
        -h|--help) usage; exit 0;;
        *) echo "$0: unknown option $1" >&2; usage; exit 2;;
    esac
done

if [ "$ARTIFACT_ROOT_SET" -eq 0 ] && [ -z "${K3_Q8_ARTIFACT_ROOT:-}" ]; then
    ARTIFACT_ROOT="$REPO/artifacts/k3-full96-q8w16/$CHECKPOINT_ID"
fi

if [ -n "${PJM_MPI_PROC:-}" ] && [ "$PJM_MPI_PROC" -ne "$PHYSICAL_NODES" ]; then
    echo "$0: requires the active 12-node allocation (PJM_MPI_PROC=$PJM_MPI_PROC)" >&2
    exit 3
fi
if [ ! -d "$MODEL_DIR" ]; then
    echo "$0: missing model directory: $MODEL_DIR" >&2
    exit 4
fi
if [ "$CHUNK_MIB" -lt 1 ] || [ "$CHUNK_MIB" -gt 64 ]; then
    echo "$0: --chunk-mib must be in 1..64" >&2
    exit 2
fi
case "$MOE_SHARD_LAYOUT" in
    replicated|row-aligned) ;;
    *) echo "$0: invalid --moe-shard-layout $MOE_SHARD_LAYOUT" >&2; exit 2;;
esac

ARTIFACT_ROOT=$(realpath -m "$ARTIFACT_ROOT")
if [ "$MOE_SHARD_LAYOUT" = replicated ]; then
    NATIVE_DIR="$ARTIFACT_ROOT/native-full96-expert-tp"
    MIXED_DIR="$ARTIFACT_ROOT/mixed-q8w16-expert-tp"
else
    NATIVE_DIR="$ARTIFACT_ROOT/native-full96-expert-tp-$MOE_SHARD_LAYOUT"
    MIXED_DIR="$ARTIFACT_ROOT/mixed-q8w16-expert-tp-$MOE_SHARD_LAYOUT"
fi
mkdir -p "$NATIVE_DIR" "$MIXED_DIR"

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
export K3_PYTHON="${K3_PYTHON:-$SCRIPT_DIR/.venv-$(uname -m)/bin/python}"
export CHUNK_MIB
export MODEL_DIR NATIVE_DIR MIXED_DIR QUALITY_GATE FORCE SCRIPT_DIR MOE_SHARD_LAYOUT
export OMP_NUM_THREADS="$CONVERT_THREADS" OMP_DYNAMIC=false

make -C "$SCRIPT_DIR" full-convert >/dev/null

echo "K3_Q8_PREPARE_BEGIN model=$MODEL_DIR artifact=$ARTIFACT_ROOT logical_nodes=$NODES moe_shard_layout=$MOE_SHARD_LAYOUT"
mpiexec -np "$PHYSICAL_NODES" -of-proc "$ARTIFACT_ROOT/prepare.rank" sh -c '
    set -euo pipefail
    physical=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:?no MPI rank}}}
    for ((logical=physical; logical<96; logical+=12)); do
        native_blob="$NATIVE_DIR/rank$(printf "%03d" "$logical").blob"
        native_manifest="$NATIVE_DIR/rank$(printf "%03d" "$logical").manifest"
        mixed_blob="$MIXED_DIR/rank$(printf "%03d" "$logical").blob"
        mixed_manifest="$MIXED_DIR/rank$(printf "%03d" "$logical").manifest"
        if [ "$FORCE" -eq 0 ] && [ -s "$native_blob" ] && [ -s "$native_manifest" ]; then
            echo "K3_Q8_REUSE native rank=$logical"
        else
            K3_EXPERT_TP=1 K3_MOE_SHARD_LAYOUT="$MOE_SHARD_LAYOUT" "$SCRIPT_DIR/run_k3_full_stage_rank.sh" \
                "$MODEL_DIR" "$NATIVE_DIR" 96 "$logical" full96
        fi
        if [ "$FORCE" -eq 0 ] && [ -s "$mixed_blob" ] && [ -s "$mixed_manifest" ]; then
            echo "K3_Q8_REUSE mixed rank=$logical"
        else
            quality_args=()
            # One quality gate per physical node is enough to cover the
            # replicated routed matrices while keeping preparation bounded.
            if [ "$QUALITY_GATE" -eq 1 ] && [ "$logical" -eq "$physical" ]; then
                quality_args+=(--quality-gate)
            fi
            args=(--input-dir "$NATIVE_DIR" --output-dir "$MIXED_DIR" \
                  --rank "$logical" --nodes 96 --mode mixed-q8w16-expert-tp-$MOE_SHARD_LAYOUT)
            if [ "$FORCE" -eq 1 ]; then args+=(--force); fi
            "$SCRIPT_DIR/k3_full_convert" "${args[@]}" "${quality_args[@]}"
        fi
    done
'

for ((rank=0; rank<NODES; ++rank)); do
    test -s "$MIXED_DIR/rank$(printf '%03d' "$rank").blob"
    test -s "$MIXED_DIR/rank$(printf '%03d' "$rank").manifest"
done
printf 'K3_Q8_PREPARE status=PASS artifact=%s native=%s mixed=%s ranks=%d quality_gate=%d moe_shard_layout=%s\n' \
    "$ARTIFACT_ROOT" "$NATIVE_DIR" "$MIXED_DIR" "$NODES" "$QUALITY_GATE" "$MOE_SHARD_LAYOUT" \
    | tee "$ARTIFACT_ROOT/READY"
