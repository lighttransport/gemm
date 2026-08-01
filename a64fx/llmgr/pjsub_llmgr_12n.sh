#!/bin/bash
# 12-node Fugaku batch job that exposes the llmgr control port back to a
# frontend through a supervised SSH reverse tunnel.
#
#   pjsub --no-check-directory a64fx/llmgr/pjsub_llmgr_12n.sh
#
# Then, on the frontend that was tunnelled to:
#   curl localhost:21374/health
#
# Auth is off by default (loopback bind, private fabric). To require a bearer
# token:  LLMGR_TOKEN=... pjsub -x LLMGR_TOKEN ...
#
# The tunnel is SUPERVISED: if it drops (network blip, frontend bounce) it is
# re-established -- re-selecting a reachable frontend each time -- up to
# MAX_RETRY consecutive failures. Structure follows tools/pjsub_bash_http_1n.sh,
# which is the proven version of this dance; the differences are node=12, the
# llmgr server command, and the /health probe shape.
#
# NB: frontend targets MUST be FQDNs. From a compute node, `login1` and
# `fn01sv03` do not resolve; `login1.fugaku.r-ccs.riken.jp` does.

#PJM -g hp250467
#PJM -L "rscgrp=small,node=12,elapse=03:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=12"
#PJM --llio localtmp-size=80Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j

set -uo pipefail   # not -e: the supervisor loop relies on non-zero returns

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"

LOGIN_NODE=${LOGIN_NODE:-1}
FRONTEND_HOST=${FRONTEND_HOST:-login${LOGIN_NODE}.fugaku.r-ccs.riken.jp}
FRONTEND_SSH_TARGET=${FRONTEND_SSH_TARGET:-$FRONTEND_HOST}
# Comma-separated fallbacks: a frontend can be down or full.
FRONTEND_SSH_TARGETS=${FRONTEND_SSH_TARGETS:-$FRONTEND_SSH_TARGET,login2.fugaku.r-ccs.riken.jp,login3.fugaku.r-ccs.riken.jp}
FRONTEND_PORT=${FRONTEND_PORT:-21374}
SERVER_PORT=${SERVER_PORT:-21274}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
REPO=${REPO:-/home/u14346/work/gemm/glm5-1}
TOKEN=${LLMGR_TOKEN:-}   # empty => no auth (loopback + private fabric)
LOGDIR=${LOGDIR:-$REPO/a64fx/llmgr/logs}
SSH_KNOWN_HOSTS=${SSH_KNOWN_HOSTS:-$LOGDIR/known_hosts}

MAX_RETRY=${MAX_RETRY:-10}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-15}
HEALTH_EVERY=${HEALTH_EVERY:-8}
KEEPALIVE_SECONDS=${KEEPALIVE_SECONDS:-0}   # >0: self-exit after this long

mkdir -p "$LOGDIR"
cd "$REPO" || { echo "ERROR cannot cd to REPO=$REPO"; exit 1; }

SERVER_LOG="$LOGDIR/llmgr.${PJM_JOBID:-nojob}.log"
TUNNEL_LOG="$LOGDIR/tunnel.${PJM_JOBID:-nojob}.log"
RUNTIME_ENV="$LOGDIR/runtime.${PJM_JOBID:-nojob}.env"

echo "=== llmgr batch job ==="
echo "jobid=${PJM_JOBID:-unknown} nodes=${PJM_NODE:-?} compute_host=$(hostname)"
echo "frontend=$FRONTEND_HOST port=$FRONTEND_PORT -> server 127.0.0.1:$SERVER_PORT"
echo "repo=$REPO logdir=$LOGDIR"

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
    # Ask llmgr to stop its runners before the job dies, so no mpiexec tree is
    # left holding nodes at teardown.
    CURL_AUTH=()
    [[ -n "$TOKEN" ]] && CURL_AUTH=(-H "Authorization: Bearer $TOKEN")
    [[ -n "$SERVER_PID" ]] && curl -fsS -m 30 -X POST \
        "${CURL_AUTH[@]+"${CURL_AUTH[@]}"}" \
        "http://127.0.0.1:$SERVER_PORT/shutdown" >/dev/null 2>&1
    sleep 2
    [[ -n "$TUNNEL_PID" ]] && { kill "$TUNNEL_PID" 2>/dev/null; wait "$TUNNEL_PID" 2>/dev/null; }
    [[ -n "$SERVER_PID" ]] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
}
trap cleanup EXIT

ts() { date -u +%FT%TZ; }

probe() {   # probe <port> -- HTTP /health on the given loopback port
    python3 - "$1" "$TOKEN" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
port, token = int(sys.argv[1]), sys.argv[2]
hdrs = {"Authorization": "Bearer " + token} if token else {}
req = urllib.request.Request("http://127.0.0.1:%d/health" % port, headers=hdrs)
with urllib.request.urlopen(req, timeout=5.0) as r:
    sys.exit(0 if json.loads(r.read().decode()).get("ok") else 1)
PY
}

server_healthy() { probe "$SERVER_PORT"; }

# Deep probe: from the frontend, hit the forwarded port (end-to-end check).
tunnel_healthy() {
    ssh "${SSH_OPTS[@]}" "$FRONTEND_SSH_TARGET" \
        "python3 - '$FRONTEND_PORT' '$TOKEN' <<'PY'
import json, sys, urllib.request
port, token = int(sys.argv[1]), sys.argv[2]
hdrs = {'Authorization': 'Bearer ' + token} if token else {}
req = urllib.request.Request('http://127.0.0.1:%d/health' % port, headers=hdrs)
with urllib.request.urlopen(req, timeout=8.0) as r:
    sys.exit(0 if json.loads(r.read().decode()).get('ok') else 1)
PY" >/dev/null 2>&1
}

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

open_reverse_tunnel() {
    [[ -n "$TUNNEL_PID" ]] && { kill "$TUNNEL_PID" 2>/dev/null; wait "$TUNNEL_PID" 2>/dev/null; }
    ssh -N "${SSH_OPTS[@]}" \
        -R 127.0.0.1:${FRONTEND_PORT}:127.0.0.1:${SERVER_PORT} \
        "$FRONTEND_SSH_TARGET" >>"$TUNNEL_LOG" 2>&1 &
    TUNNEL_PID=$!
}

# --- start llmgr on this (head) node ---
SRV=(python3 -u "$REPO/a64fx/llmgr/llmgr_server.py"
     --host "$SERVER_HOST" --port "$SERVER_PORT" --verbose)
[[ -n "$TOKEN" ]] && SRV+=(--token "$TOKEN")
"${SRV[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 60); do server_healthy && break; sleep 0.5; done
if ! server_healthy; then
    echo "ERROR llmgr did not come up"; tail -30 "$SERVER_LOG" 2>/dev/null; exit 1
fi
echo "$(ts) llmgr up (pid $SERVER_PID)"

# --- bring up the reverse tunnel ---
established=0
for attempt in $(seq 1 "$MAX_RETRY"); do
    if ! select_frontend_target; then
        echo "$(ts) no reachable frontend (attempt $attempt/$MAX_RETRY)"
        sleep "$MONITOR_INTERVAL"; continue
    fi
    open_reverse_tunnel
    sleep 3
    if kill -0 "$TUNNEL_PID" 2>/dev/null && tunnel_healthy; then
        established=1; break
    fi
    echo "$(ts) tunnel not healthy via $FRONTEND_SSH_TARGET (attempt $attempt/$MAX_RETRY)"
    tail -3 "$TUNNEL_LOG" 2>/dev/null
    sleep "$MONITOR_INTERVAL"
done
[[ "$established" -eq 1 ]] || { echo "ERROR no reverse tunnel after $MAX_RETRY attempts"; exit 1; }

cat >"$RUNTIME_ENV" <<EOF
JOBID=${PJM_JOBID:-unknown}
COMPUTE_HOST=$(hostname)
NODES=${PJM_NODE:-}
FRONTEND_HOST=$FRONTEND_HOST
FRONTEND_SSH_TARGET=$FRONTEND_SSH_TARGET
FRONTEND_PORT=$FRONTEND_PORT
SERVER_PORT=$SERVER_PORT
SERVER_LOG=$SERVER_LOG
TUNNEL_LOG=$TUNNEL_LOG
EOF

echo "SENTINEL llmgr_batch_ready=OK"
if [[ -n "$TOKEN" ]]; then
  echo "on ${FRONTEND_SSH_TARGET}: curl -H 'Authorization: Bearer <token>' localhost:${FRONTEND_PORT}/health"
else
  echo "on ${FRONTEND_SSH_TARGET}: curl localhost:${FRONTEND_PORT}/health"
fi
echo "runtime_env=$RUNTIME_ENV server_log=$SERVER_LOG tunnel_log=$TUNNEL_LOG"

# --- supervisor loop ---
start_ts=$(date +%s); fails=0; ticks=0
while true; do
    if [[ "$KEEPALIVE_SECONDS" -gt 0 ]] && (( $(date +%s) - start_ts >= KEEPALIVE_SECONDS )); then
        echo "$(ts) keepalive window ${KEEPALIVE_SECONDS}s elapsed; exiting cleanly"; break
    fi
    sleep "$MONITOR_INTERVAL"; ticks=$((ticks + 1))

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "$(ts) ERROR llmgr died; exiting"; tail -30 "$SERVER_LOG"; exit 1
    fi

    down=0
    kill -0 "$TUNNEL_PID" 2>/dev/null || down=1
    if [[ "$down" -eq 0 ]] && (( ticks % HEALTH_EVERY == 0 )); then
        tunnel_healthy || { echo "$(ts) deep probe failed; tunnel considered down"; down=1; }
    fi
    [[ "$down" -eq 0 ]] && continue

    fails=$((fails + 1))
    if (( fails > MAX_RETRY )); then
        echo "$(ts) ERROR tunnel reconnect failed ${MAX_RETRY} consecutive times; giving up"; exit 1
    fi
    echo "$(ts) tunnel down; reconnect ${fails}/${MAX_RETRY}"
    if select_frontend_target; then
        open_reverse_tunnel; sleep 3
        if kill -0 "$TUNNEL_PID" 2>/dev/null && tunnel_healthy; then
            echo "$(ts) tunnel re-established via ${FRONTEND_SSH_TARGET}"; fails=0
        else
            kill "$TUNNEL_PID" 2>/dev/null; wait "$TUNNEL_PID" 2>/dev/null
        fi
    else
        echo "$(ts) no reachable frontend during reconnect"
    fi
done
