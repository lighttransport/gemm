#!/bin/bash
set -euo pipefail

LOGIN_NODE=${LOGIN_NODE:-1}
CONTROL_DIR=${CONTROL_DIR:-/tmp/ds4p-bash-http}
CONTROL_PATH=${CONTROL_PATH:-$CONTROL_DIR/cm-%r@%h:%p}
STATE_FILE=${STATE_FILE:-$CONTROL_DIR/state.env}

# Close the master open_…sh actually created (REMOTE recorded in state.env);
# explicit REMOTE= wins, else fall back to loginN.fugaku.r-ccs.riken.jp.
REMOTE_ENV=${REMOTE:-}
if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
fi
REMOTE=${REMOTE_ENV:-${REMOTE:-login${LOGIN_NODE}.fugaku.r-ccs.riken.jp}}

ssh -o IdentitiesOnly=yes -o ControlPath="$CONTROL_PATH" -O exit "$REMOTE"
