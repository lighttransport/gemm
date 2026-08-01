#!/bin/bash
# Client/frontend-side supervisor for the bash-over-http local forward.
#
# open_bash_http_local_tunnel.sh creates a single ControlMaster `ssh -L` from the
# local box to the pinned Fugaku frontend (loginN). ControlPersist keeps it up,
# but a dropped master (network blip, frontend bounce, laptop sleep) is NOT
# auto-recreated. This watcher monitors the master and re-establishes it on drop,
# up to MAX_RETRY CONSECUTIVE failures before giving up (a healthy check resets
# the counter). It is the client-side counterpart to the job-side reverse-tunnel
# supervisor in pjsub_bash_http_1n.sh.
#
# Run it in the background or a tmux pane, e.g.:
#   nohup tools/watch_bash_http_local_tunnel.sh > /tmp/ds4p-bash-http/watch.log 2>&1 &
#
# Honors LOGIN_NODE / REMOTE (same as open_/submit) and MAX_RETRY / MONITOR_INTERVAL.

set -uo pipefail   # not -e: transient ssh failures must not kill the watcher

HERE=$(cd "$(dirname "$0")" && pwd)

LOGIN_NODE=${LOGIN_NODE:-1}
CONTROL_DIR=${CONTROL_DIR:-/tmp/ds4p-bash-http}
CONTROL_PATH=${CONTROL_PATH:-$CONTROL_DIR/cm-%r@%h:%p}
STATE_FILE=${STATE_FILE:-$CONTROL_DIR/state.env}
MAX_RETRY=${MAX_RETRY:-10}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-15}

# Resolve the pinned frontend: explicit REMOTE= > state.env REMOTE > loginN default.
REMOTE_ENV=${REMOTE:-}
if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
fi
REMOTE=${REMOTE_ENV:-${REMOTE:-login${LOGIN_NODE}.fugaku.r-ccs.riken.jp}}

ts() { date -u +%FT%TZ; }
master_alive() { ssh -o ControlPath="$CONTROL_PATH" -O check "$REMOTE" >/dev/null 2>&1; }

echo "$(ts) watching local forward to $REMOTE (max_retry=$MAX_RETRY, interval=${MONITOR_INTERVAL}s)"

# Ensure the forward exists at startup.
if ! master_alive; then
    LOGIN_NODE="$LOGIN_NODE" REMOTE="$REMOTE" CONTROL_DIR="$CONTROL_DIR" \
        "$HERE/open_bash_http_local_tunnel.sh" >/dev/null 2>&1 || true
fi

fails=0
while true; do
    if master_alive; then
        fails=0
    else
        fails=$((fails + 1))
        if (( fails > MAX_RETRY )); then
            echo "$(ts) local forward reconnect failed ${MAX_RETRY} consecutive times; giving up" >&2
            exit 1
        fi
        echo "$(ts) local forward down; reconnect attempt ${fails}/${MAX_RETRY}"
        # Clear any stale master socket, then re-create (open_ is idempotent).
        ssh -o ControlPath="$CONTROL_PATH" -O exit "$REMOTE" >/dev/null 2>&1 || true
        LOGIN_NODE="$LOGIN_NODE" REMOTE="$REMOTE" CONTROL_DIR="$CONTROL_DIR" \
            "$HERE/open_bash_http_local_tunnel.sh" >/dev/null 2>&1 || true
        if master_alive; then
            echo "$(ts) local forward re-established"
            fails=0
        fi
    fi
    sleep "$MONITOR_INTERVAL"
done
