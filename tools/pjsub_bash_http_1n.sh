#!/bin/bash
# One-node Fugaku batch job that exposes bash-over-http from the compute node
# back to a pinned frontend through an SSH reverse tunnel.

#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:20:00"
#PJM -L "freq=2000,eco_state=0"
#PJM -j

set -euo pipefail

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

mkdir -p "$LOGDIR"
cd "$WORKDIR"

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

cleanup() {
    set +e
    if [[ -n "${TUNNEL_PID:-}" ]]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
    fi
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

python3 tools/bash_http_server.py \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --token "$TOKEN" \
    --verbose \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 50); do
    if python3 - "$SERVER_PORT" "$TOKEN" <<'PY'
import json
import sys
import urllib.request

port = int(sys.argv[1])
token = sys.argv[2]
req = urllib.request.Request(
    "http://127.0.0.1:%d/health" % port,
    headers={"Authorization": "Bearer " + token},
)
with urllib.request.urlopen(req, timeout=2.0) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    if not data.get("ok"):
        raise SystemExit(1)
PY
    then
        break
    fi
    sleep 0.2
done

SELECTED_FRONTEND_TARGET=
for target in $(printf '%s\n' "$FRONTEND_SSH_TARGETS" | tr ',' ' '); do
    if ssh "${SSH_OPTS[@]}" "$target" true >/dev/null 2>&1; then
        SELECTED_FRONTEND_TARGET=$target
        break
    fi
done

if [[ -z "$SELECTED_FRONTEND_TARGET" ]]; then
    echo "ERROR no reachable frontend ssh target"
    exit 1
fi

FRONTEND_SSH_TARGET=$SELECTED_FRONTEND_TARGET

ssh -N \
    "${SSH_OPTS[@]}" \
    -R 127.0.0.1:${FRONTEND_PORT}:127.0.0.1:${SERVER_PORT} \
    "$FRONTEND_SSH_TARGET" \
    >"$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

sleep 2
if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "ERROR reverse tunnel failed"
    cat "$TUNNEL_LOG" || true
    exit 1
fi

if ! ssh "${SSH_OPTS[@]}" "$FRONTEND_SSH_TARGET" \
    "python3 - '$FRONTEND_PORT' '$TOKEN' <<'PY'
import json
import sys
import urllib.request

port = int(sys.argv[1])
token = sys.argv[2]
req = urllib.request.Request(
    'http://127.0.0.1:%d/health' % port,
    headers={'Authorization': 'Bearer ' + token},
)
with urllib.request.urlopen(req, timeout=5.0) as resp:
    data = json.loads(resp.read().decode('utf-8'))
    if not data.get('ok'):
        raise SystemExit(1)
PY"; then
    echo "ERROR reverse tunnel health probe failed"
    cat "$TUNNEL_LOG" || true
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

# Keep the job alive long enough for a frontend-side test client session.
sleep 600
