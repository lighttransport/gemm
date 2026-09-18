#!/bin/bash
set -euo pipefail

REPO=${WORKDIR:-$HOME/work/gemm/glm53f}
MODEL=${MODEL:-$HOME/models/glm53f}
OUT=${OUT:-/local/glm53f-validation-${PJM_JOBID:-manual}}
mkdir -p "$OUT"
cd "$REPO"

if [ -r /proc/meminfo ]; then
    avail=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    [ "$avail" -ge 6291456 ] || {
        echo "MemAvailable below 6 GiB: ${avail} KiB" >&2
        exit 3
    }
fi
python3 a64fx/glm5/glm53f_validation.py manifest "$MODEL" >"$OUT/manifest.txt"
printf 'model=%s\nout=%s\njob=%s\nhost=%s\n' \
    "$MODEL" "$OUT" "${PJM_JOBID:-unknown}" "$(hostname)" >"$OUT/job.env"
printf 'k_norm_eps=1e-6\nindex_topk=2048\nkpool=4\nexperts=288\ntop_k=8\nrouted_scale=2.5\n' >"$OUT/contract.txt"
echo "GLM53F_VALIDATION_JOB_READY job=${PJM_JOBID:-unknown} host=$(hostname) out=$OUT"
exec bash -i
