#!/bin/bash
set -euo pipefail

LOGIN_NODE=${LOGIN_NODE:-1}
if [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
    CONTROL_ROOT=$XDG_RUNTIME_DIR
elif [[ -d /local ]]; then
    CONTROL_ROOT=/local
else
    CONTROL_ROOT=tmp
fi
CONTROL_DIR=${CONTROL_DIR:-$CONTROL_ROOT/clair-bash-http-${USER}}
CONTROL_PATH=${CONTROL_PATH:-$CONTROL_DIR/cm-%r@%h:%p}
STATE_FILE=${STATE_FILE:-$CONTROL_DIR/state.env}

# Close the master open_…sh actually created (REMOTE recorded in state.env);
# explicit REMOTE= wins, else fall back to the fugaku1 SSH alias.
REMOTE_ENV=${REMOTE:-}
if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
fi
REMOTE=${REMOTE_ENV:-${REMOTE:-fugaku1}}

ssh -o IdentitiesOnly=yes -o ControlPath="$CONTROL_PATH" -O exit "$REMOTE"
