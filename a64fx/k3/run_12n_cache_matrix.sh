#!/bin/bash
set -euo pipefail

# Reproducible 12-node prefix-cache lifecycle for the dummy K3 runner.
# Override variables when exercising another validated topology or dtype.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

NODES=${NODES:-12}
TP_NODES=${TP_NODES:-12}
LAYER=${LAYER:-1}
LAYERS=${LAYERS:-3}
THREADS=${THREADS:-8}
KDA_THREADS=${KDA_THREADS:-2}
CACHE_TOKENS=${CACHE_TOKENS:-64}
TOKENS=${TOKENS:-128}
DTYPE=${DTYPE:-bf16}
AR_GROUPS=${AR_GROUPS:-auto}
COMM_ROBUST=${COMM_ROBUST:-1}
COMM_ACK=${COMM_ACK:-1}
COMM_DETERMINISTIC=${COMM_DETERMINISTIC:-1}
COMM_POLL_SPINS=${COMM_POLL_SPINS:-4}
PREFETCH_MIB=${PREFETCH_MIB:-1}
PREFETCH_THREADS=${PREFETCH_THREADS:-4}
RESULT_ROOT=${RESULT_ROOT:-"$REPO_ROOT/a64fx/k3/logs/cache-matrix-$(date +%s)"}
case "$RESULT_ROOT" in
    /*) ;;
    *) RESULT_ROOT="$REPO_ROOT/$RESULT_ROOT" ;;
esac

case "$DTYPE" in
    bf16) CACHE_FLAG=--mla-cache-bf16 ;;
    fp32) CACHE_FLAG=--mla-cache-fp32 ;;
    *) echo "DTYPE must be bf16 or fp32" >&2; exit 2 ;;
esac

RUNNER=${RUNNER:-$SCRIPT_DIR/run_k3_ep.sh}
COMMON=(
    --mode dummy --nodes "$NODES" --tp-nodes "$TP_NODES"
    --layer "$LAYER" --layers "$LAYERS" --threads "$THREADS"
    --kda-threads "$KDA_THREADS" --ar-groups "$AR_GROUPS"
    --comm-robust "$COMM_ROBUST" --comm-ack "$COMM_ACK"
    --comm-deterministic "$COMM_DETERMINISTIC"
    --comm-poll-spins "$COMM_POLL_SPINS"
    --prefetch-mib "$PREFETCH_MIB" --prefetch-threads "$PREFETCH_THREADS"
)

mkdir -p "$RESULT_ROOT"
CACHE="$RESULT_ROOT/cache"

echo "K3_MATRIX_BEGIN nodes=$NODES tp_nodes=$TP_NODES dtype=$DTYPE cache_tokens=$CACHE_TOKENS tokens=$TOKENS results=$RESULT_ROOT"
run_phase() {
    local phase=$1 expected_tokens=$2
    shift 2
    local phase_dir="$RESULT_ROOT/$phase"
    local phase_log="$RESULT_ROOT/.${phase}.runner.log"

    if ! bash "$RUNNER" "$@" --result-dir "$phase_dir" 2>&1 | tee "$phase_log"; then
        echo "K3_MATRIX_PHASE_FAIL phase=$phase reason=runner" >&2
        return 5
    fi
    grep -Eq "K3 distributed result: rc=0 pass_markers=${NODES}/${NODES} " "$phase_log" || {
        echo "K3_MATRIX_PHASE_FAIL phase=$phase reason=rank-markers" >&2
        return 5
    }
    mv "$phase_log" "$phase_dir/runner.log"
    awk -v expected="$NODES" -v tokens="$expected_tokens" '
        /K3_RESULT status=PASS/ {
            results++
            if ($0 !~ /disagreement=0[.]000e[+]00/) bad_disagreement++
            if ($0 !~ ("tokens_completed=" tokens "([[:space:]]|$)")) bad_tokens++
        }
        END {
            if (results < 1 || bad_disagreement || bad_tokens) exit 1
        }
    ' "$phase_dir"/rank.*.* || {
        echo "K3_MATRIX_PHASE_FAIL phase=$phase reason=result-invariant" >&2
        return 5
    }
}

check_cache_shards() {
    local phase=$1 shards
    shards=$(find "$CACHE" -maxdepth 1 -type f -name 'k3_ep_cache_*.bin' 2>/dev/null | wc -l)
    if [ "$shards" -ne "$NODES" ]; then
        echo "K3_MATRIX_PHASE_FAIL phase=$phase reason=cache-shards count=$shards expected=$NODES" >&2
        return 5
    fi
}

run_phase save "$CACHE_TOKENS" \
    "${COMMON[@]}" "$CACHE_FLAG" --tokens "$CACHE_TOKENS" \
    --cache-save "$CACHE"
check_cache_shards save
run_phase inplace "$TOKENS" \
    "${COMMON[@]}" "$CACHE_FLAG" --tokens "$TOKENS" \
    --cache-load "$CACHE" --cache-save "$CACHE"
check_cache_shards inplace
run_phase verify 0 \
    "${COMMON[@]}" "$CACHE_FLAG" --tokens "$TOKENS" \
    --cache-load "$CACHE"
echo "K3_MATRIX_END status=PASS results=$RESULT_ROOT"
