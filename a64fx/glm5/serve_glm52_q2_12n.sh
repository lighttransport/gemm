#!/bin/bash
# Restage/verify the 12-node model, then expose serialized HTTP generation.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "${GLM52_HTTP_NO_STAGE:-0}" != 1 ]; then
    "$HERE/run_glm52_q2_12n.sh" check --stable-outputs --no-enforce
fi
exec python3 "$HERE/glm52_http_server.py" "$@"
