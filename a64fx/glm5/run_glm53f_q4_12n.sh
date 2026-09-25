#!/bin/bash
# Compatibility entry point; use run_glm53f_12n.sh for new workflows.
set -euo pipefail
export GLM53F_QUANT=q4
exec bash "$(dirname "$0")/run_glm53f_12n.sh" run "$@"
