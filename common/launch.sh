#!/bin/bash
# launch.sh - Launch N processes with custom comm rank assignment
#
# Usage: ./launch.sh N ./program [args...]
#
# Example:
#   ./launch.sh 2 ./test_transformer_parallel model.gguf --pp 2 "Hello" 16
#   ./launch.sh 4 ./test_transformer_parallel model.gguf --tp 2 --dp 2 "Hello" 16

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 N program [args...]"
    echo "  N = number of processes"
    exit 1
fi

N=$1
shift
PROGRAM="$1"
shift

MASTER_ADDR="${COMM_MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${COMM_MASTER_PORT:-29500}"

PIDS=()

for ((i=0; i<N; i++)); do
    "$PROGRAM" "$@" \
        --comm-rank $i \
        --comm-nranks $N \
        --comm-addr "$MASTER_ADDR" \
        --comm-port "$MASTER_PORT" &
    PIDS+=($!)
done

# Wait for all and propagate exit code
EXIT_CODE=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        EXIT_CODE=1
    fi
done

exit $EXIT_CODE
