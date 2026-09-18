#!/bin/bash
set -euo pipefail
REPO=${REPO:-$PWD}
OUT=${OUT:-/local/glm53f-validation-${PJM_JOBID:-manual}}
MODEL=${MODEL:-$HOME/models/glm53f}
mkdir -p "$OUT"
export TMPDIR="${TMPDIR:-$OUT/tmp}"
mkdir -p "$TMPDIR"
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast \
  -I"$REPO/common" "$REPO/a64fx/glm5/glm53f_router_dispatch_probe.c" \
  -lm -o "$OUT/glm53f_router_dispatch_probe"
"$OUT/glm53f_router_dispatch_probe" "$MODEL"
