#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../../.." && pwd)
eval "$(python3 "$HERE/config.py" --project-dir "$REPO_ROOT" --shell)"

# Pin the local forward to the configured SSH target. The repository's
# `fugaku1` alias resolves to login1; set REMOTE to use another login host.
LOGIN_NODE=${LOGIN_NODE:-1}
case "$LOGIN_NODE" in [1-8]) ;; *) echo "LOGIN_NODE must be 1..8 (got '$LOGIN_NODE')" >&2; exit 2 ;; esac
REMOTE=${REMOTE:-${BASH_HTTP_REMOTE:-fugaku1}}
LOCAL_PORT=${LOCAL_PORT:-${BASH_HTTP_LOCAL_PORT:-42386}}
REMOTE_PORT=${REMOTE_PORT:-${BASH_HTTP_REMOTE_PORT:-32386}}
PORT_OFFSET=${PORT_OFFSET:-${BASH_HTTP_PORT_OFFSET:-0}}
case "$PORT_OFFSET" in
    ''|*[!0-9]*) echo "PORT_OFFSET must be a non-negative integer" >&2; exit 2 ;;
esac
LOCAL_PORT=$((LOCAL_PORT + PORT_OFFSET))
REMOTE_PORT=$((REMOTE_PORT + PORT_OFFSET))
if [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
    CONTROL_ROOT=$XDG_RUNTIME_DIR
elif [[ -d /local ]]; then
    CONTROL_ROOT=/local
else
    CONTROL_ROOT=tmp
fi
if [[ -n "${CONTROL_DIR:-}" || -n "${BASH_HTTP_CONTROL_DIR:-}" ]]; then
    CONTROL_DIR=${CONTROL_DIR:-$BASH_HTTP_CONTROL_DIR}
else
    CONTROL_DIR=$CONTROL_ROOT/clair-bash-http-${USER}
    # Each offset needs its own ControlMaster: an existing master cannot
    # acquire a new -L listener after it has been created.
    if (( PORT_OFFSET > 0 )); then
        CONTROL_DIR=${CONTROL_DIR}-port-$((LOCAL_PORT))
    fi
fi
CONTROL_PATH=${CONTROL_PATH:-$CONTROL_DIR/cm-%r@%h:%p}
STATE_FILE=${STATE_FILE:-$CONTROL_DIR/state.env}

umask 077
mkdir -p "$CONTROL_DIR"
chmod 700 "$CONTROL_DIR"

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

chmod 600 "$STATE_FILE"

echo "REMOTE_HOST=$REMOTE_HOST"
echo "REMOTE_IPV4=$REMOTE_IPV4"
echo "LOCAL_URL=http://127.0.0.1:${LOCAL_PORT}"
echo "STATE_FILE=$STATE_FILE"
