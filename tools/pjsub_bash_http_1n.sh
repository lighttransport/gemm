#!/bin/bash
# One-node Fugaku batch job that exposes bash-over-http from the compute node
# back to a pinned frontend through an SSH reverse tunnel.
#
# The reverse tunnel is SUPERVISED: if it drops (network blip, frontend bounce),
# it is re-established — re-selecting a reachable frontend target each time — up
# to MAX_RETRY consecutive failures before the job gives up. The job stays up
# until PJM's elapse limit (small rscgrp allows up to 72:00:00) or, if
# KEEPALIVE_SECONDS>0, that many seconds after the bridge first comes up.

#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:20:00"
#PJM -L "freq=2000,eco_state=0"
#PJM -j

set -uo pipefail   # not -e: the supervisor loop relies on non-zero returns

# Frontend to reverse-tunnel back to. LOGIN_NODE (1..8) -> loginN.fugaku.r-ccs.riken.jp;
# submit_bash_http_1n_over_ssh.sh overrides FRONTEND_* explicitly via pjsub -x.
LOGIN_NODE=${LOGIN_NODE:-1}
FRONTEND_HOST=${FRONTEND_HOST:-login${LOGIN_NODE}.fugaku.r-ccs.riken.jp}
FRONTEND_SSH_TARGET=${FRONTEND_SSH_TARGET:-$FRONTEND_HOST}
FRONTEND_SSH_TARGETS=${FRONTEND_SSH_TARGETS:-$FRONTEND_SSH_TARGET}
FRONTEND_PORT=${FRONTEND_PORT:-21364}
SERVER_PORT=${SERVER_PORT:-21264}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
WORKDIR=${WORKDIR:-$HOME/work/gemm/ds4p}
TOKEN=${TOKEN:-${DS4P_BASH_HTTP_TOKEN:?set DS4P_BASH_HTTP_TOKEN (bash-over-http bearer token) before submitting}}
LOGDIR=${LOGDIR:-$WORKDIR/tools/bash_http_job}
SSH_KNOWN_HOSTS=${SSH_KNOWN_HOSTS:-$LOGDIR/known_hosts}

# Reconnection / lifetime knobs.
MAX_RETRY=${MAX_RETRY:-10}                # consecutive reverse-tunnel reconnect failures before giving up
MONITOR_INTERVAL=${MONITOR_INTERVAL:-15}  # seconds between supervisor checks
HEALTH_EVERY=${HEALTH_EVERY:-8}           # deep (through-frontend) health probe every N ticks
KEEPALIVE_SECONDS=${KEEPALIVE_SECONDS:-0} # >0: self-exit after this long; 0: run until PJM elapse

mkdir -p "$LOGDIR"
cd "$WORKDIR" || { echo "ERROR cannot cd to WORKDIR=$WORKDIR"; exit 1; }

SERVER_LOG="$LOGDIR/server.${PJM_JOBID:-nojob}.log"
TUNNEL_LOG="$LOGDIR/tunnel.${PJM_JOBID:-nojob}.log"
RUNTIME_ENV="$LOGDIR/runtime.${PJM_JOBID:-nojob}.env"

echo "=== bash-over-http batch job ==="
echo "jobid=${PJM_JOBID:-unknown}"
echo "compute_host=$(hostname)"
echo "frontend_host=$FRONTEND_HOST"
echo "frontend_ssh_target=$FRONTEND_SSH_TARGET"
echo "frontend_ssh_targets=$FRONTEND_SSH_TARGETS"
echo "frontend_port=$FRONTEND_PORT"
echo "server_port=$SERVER_PORT"
echo "workdir=$WORKDIR"
echo "logdir=$LOGDIR"
echo "max_retry=$MAX_RETRY monitor_interval=${MONITOR_INTERVAL}s keepalive_seconds=$KEEPALIVE_SECONDS"

SSH_OPTS=(
    -o BatchMode=yes
    -o ConnectTimeout=5
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -o IdentitiesOnly=yes
    -o StrictHostKeyChecking=accept-new
    -o UserKnownHostsFile="$SSH_KNOWN_HOSTS"
)

TUNNEL_PID=
SERVER_PID=

cleanup() {
    set +e
    [[ -n "$TUNNEL_PID" ]] && { kill "$TUNNEL_PID" 2>/dev/null; wait "$TUNNEL_PID" 2>/dev/null; }
    [[ -n "$SERVER_PID" ]] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
}
trap cleanup EXIT

ts() { date -u +%FT%TZ; }

# Probe the local server directly (compute loopback).
server_healthy() {
    python3 - "$SERVER_PORT" "$TOKEN" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
port = int(sys.argv[1]); token = sys.argv[2]
req = urllib.request.Request("http://127.0.0.1:%d/health" % port,
                             headers={"Authorization": "Bearer " + token})
with urllib.request.urlopen(req, timeout=2.0) as r:
    sys.exit(0 if json.loads(r.read().decode()).get("ok") else 1)
PY
}

# Deep probe: from the frontend, curl the forwarded port (verifies the reverse tunnel end to end).
tunnel_healthy() {
    ssh "${SSH_OPTS[@]}" "$FRONTEND_SSH_TARGET" \
        "python3 - '$FRONTEND_PORT' '$TOKEN' <<'PY'
import json, sys, urllib.request
port = int(sys.argv[1]); token = sys.argv[2]
req = urllib.request.Request('http://127.0.0.1:%d/health' % port,
                             headers={'Authorization': 'Bearer ' + token})
with urllib.request.urlopen(req, timeout=5.0) as r:
    sys.exit(0 if json.loads(r.read().decode()).get('ok') else 1)
PY" >/dev/null 2>&1
}

# Pick a reachable frontend ssh target from the (comma-split) candidate list; sets FRONTEND_SSH_TARGET.
select_frontend_target() {
    local target
    for target in $(printf '%s\n' "$FRONTEND_SSH_TARGETS" | tr ',' ' '); do
        if ssh "${SSH_OPTS[@]}" "$target" true >/dev/null 2>&1; then
            FRONTEND_SSH_TARGET=$target
            return 0
        fi
    done
    return 1
}

# (Re)launch the reverse tunnel as a background ssh -R; reaps any prior one. Sets TUNNEL_PID.
open_reverse_tunnel() {
    [[ -n "$TUNNEL_PID" ]] && { kill "$TUNNEL_PID" 2>/dev/null; wait "$TUNNEL_PID" 2>/dev/null; }
    ssh -N "${SSH_OPTS[@]}" \
        -R 127.0.0.1:${FRONTEND_PORT}:127.0.0.1:${SERVER_PORT} \
        "$FRONTEND_SSH_TARGET" >>"$TUNNEL_LOG" 2>&1 &
    TUNNEL_PID=$!
}

# --- start the bash-over-http server on the compute node ---
python3 tools/bash_http_server.py \
    --host "$SERVER_HOST" --port "$SERVER_PORT" --token "$TOKEN" --verbose \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 50); do server_healthy && break; sleep 0.2; done
if ! server_healthy; then
    echo "ERROR bash_http_server did not come up"
    tail -20 "$SERVER_LOG" 2>/dev/null || true
    exit 1
fi

# --- bring up the reverse tunnel (retry the initial connect up to MAX_RETRY) ---
established=0
for attempt in $(seq 1 "$MAX_RETRY"); do
    if ! select_frontend_target; then
        echo "$(ts) no reachable frontend ssh target (attempt $attempt/$MAX_RETRY)"
        sleep "$MONITOR_INTERVAL"; continue
    fi
    open_reverse_tunnel
    sleep 3
    if kill -0 "$TUNNEL_PID" 2>/dev/null && tunnel_healthy; then
        established=1; break
    fi
    echo "$(ts) reverse tunnel not healthy via $FRONTEND_SSH_TARGET (attempt $attempt/$MAX_RETRY)"
    tail -3 "$TUNNEL_LOG" 2>/dev/null || true
    sleep "$MONITOR_INTERVAL"
done

if [[ "$established" -ne 1 ]]; then
    echo "ERROR could not establish reverse tunnel after $MAX_RETRY attempts"
    exit 1
fi

cat >"$RUNTIME_ENV" <<EOF
JOBID=${PJM_JOBID:-unknown}
COMPUTE_HOST=$(hostname)
FRONTEND_HOST=$FRONTEND_HOST
FRONTEND_SSH_TARGET=$FRONTEND_SSH_TARGET
FRONTEND_SSH_TARGETS=$FRONTEND_SSH_TARGETS
FRONTEND_PORT=$FRONTEND_PORT
SERVER_PORT=$SERVER_PORT
TOKEN=$TOKEN
SERVER_LOG=$SERVER_LOG
TUNNEL_LOG=$TUNNEL_LOG
EOF

echo "SENTINEL bash_http_batch_ready=OK"
echo "runtime_env=$RUNTIME_ENV"
echo "server_log=$SERVER_LOG"
echo "tunnel_log=$TUNNEL_LOG"

# --- supervisor: keep the reverse tunnel up, reconnect on drop ---
# `fails` counts CONSECUTIVE failed reconnects; a healthy re-establish resets it.
# > MAX_RETRY consecutive failures -> give up. Runs until PJM elapse (or KEEPALIVE_SECONDS).
start_ts=$(date +%s)
fails=0
ticks=0
while true; do
    if [[ "$KEEPALIVE_SECONDS" -gt 0 ]] && (( $(date +%s) - start_ts >= KEEPALIVE_SECONDS )); then
        echo "$(ts) keepalive window ${KEEPALIVE_SECONDS}s elapsed; exiting cleanly"
        break
    fi

    sleep "$MONITOR_INTERVAL"
    ticks=$((ticks + 1))

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "$(ts) ERROR bash_http_server died; exiting"
        exit 1
    fi

    down=0
    kill -0 "$TUNNEL_PID" 2>/dev/null || down=1
    if [[ "$down" -eq 0 ]] && (( ticks % HEALTH_EVERY == 0 )); then
        tunnel_healthy || { echo "$(ts) deep health probe failed; tunnel considered down"; down=1; }
    fi
    [[ "$down" -eq 0 ]] && continue

    fails=$((fails + 1))
    if (( fails > MAX_RETRY )); then
        echo "$(ts) ERROR reverse tunnel reconnect failed ${MAX_RETRY} consecutive times; giving up"
        exit 1
    fi
    echo "$(ts) reverse tunnel down; reconnect attempt ${fails}/${MAX_RETRY}"
    if select_frontend_target; then
        open_reverse_tunnel
        sleep 3
        if kill -0 "$TUNNEL_PID" 2>/dev/null && tunnel_healthy; then
            echo "$(ts) reverse tunnel re-established via ${FRONTEND_SSH_TARGET}"
            fails=0
        else
            kill "$TUNNEL_PID" 2>/dev/null || true; wait "$TUNNEL_PID" 2>/dev/null || true
        fi
    else
        echo "$(ts) no reachable frontend target during reconnect"
    fi
done
