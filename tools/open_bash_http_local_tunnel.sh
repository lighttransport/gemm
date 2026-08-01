#!/bin/bash
set -euo pipefail

# Fugaku frontend login node to pin (1..8) -> loginN.fugaku.r-ccs.riken.jp.
# Set this (or REMOTE) before launch to choose the frontend; the node is recorded
# in state.env so submit/close follow it.
LOGIN_NODE=${LOGIN_NODE:-1}
case "$LOGIN_NODE" in [1-8]) ;; *) echo "LOGIN_NODE must be 1..8 (got '$LOGIN_NODE')" >&2; exit 2 ;; esac
REMOTE=${REMOTE:-login${LOGIN_NODE}.fugaku.r-ccs.riken.jp}
LOCAL_PORT=${LOCAL_PORT:-21364}
REMOTE_PORT=${REMOTE_PORT:-21364}
CONTROL_DIR=${CONTROL_DIR:-/tmp/ds4p-bash-http}
CONTROL_PATH=${CONTROL_PATH:-$CONTROL_DIR/cm-%r@%h:%p}
STATE_FILE=${STATE_FILE:-$CONTROL_DIR/state.env}

mkdir -p "$CONTROL_DIR"

if ssh -o ControlPath="$CONTROL_PATH" -O check "$REMOTE" >/dev/null 2>&1; then
    :
else
    ssh -MNf \
        -o ControlMaster=yes \
        -o ControlPersist=yes \
        -o ExitOnForwardFailure=yes \
        -o IdentitiesOnly=yes \
        -o StrictHostKeyChecking=accept-new \
        -o ControlPath="$CONTROL_PATH" \
        -L 127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT} \
        "$REMOTE"
fi

REMOTE_HOST=$(ssh -o IdentitiesOnly=yes -o ControlPath="$CONTROL_PATH" "$REMOTE" 'hostname')
REMOTE_IPV4=$(ssh -o IdentitiesOnly=yes -o ControlPath="$CONTROL_PATH" "$REMOTE" "hostname -I 2>/dev/null | awk '{print \$1}'")

cat >"$STATE_FILE" <<EOF
REMOTE=$REMOTE
REMOTE_HOST=$REMOTE_HOST
REMOTE_IPV4=$REMOTE_IPV4
LOCAL_PORT=$LOCAL_PORT
REMOTE_PORT=$REMOTE_PORT
CONTROL_PATH=$CONTROL_PATH
STATE_FILE=$STATE_FILE
EOF

echo "REMOTE_HOST=$REMOTE_HOST"
echo "REMOTE_IPV4=$REMOTE_IPV4"
echo "LOCAL_URL=http://127.0.0.1:${LOCAL_PORT}"
echo "STATE_FILE=$STATE_FILE"
