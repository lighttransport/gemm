#!/bin/bash
# Keep a resident inference rank from exhausting the node's HBM.
set -uo pipefail
"$@" &
rank_pid=$!
(
    minimum=9223372036854775807
    while true; do
        available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        if [[ "$available" =~ ^[0-9]+$ ]]; then
            (( available < minimum )) && minimum=$available
            if (( available < 2097152 )); then
                echo "GLM53F_MEMORY_ABORT host=$(hostname) available_kib=$available" >&2
                kill -TERM "$rank_pid" 2>/dev/null
                break
            fi
        fi
        kill -0 "$rank_pid" 2>/dev/null || break
        sleep 5
    done
    echo "GLM53F_MEMORY_MIN host=$(hostname) available_kib=$minimum" >&2
) &
guard_pid=$!
trap 'kill -TERM "$rank_pid" "$guard_pid" 2>/dev/null' TERM INT
wait "$rank_pid"
rank_status=$?
wait "$guard_pid"
exit "$rank_status"
