#!/bin/sh
# Single-A64FX, CPU-only validation.  No MPI/uTofu and no full-model staging.
set -eu

REPO=${REPO:-$HOME/work/gemm/glm53f}
MODEL=${MODEL:-$HOME/models/glm53f}
OUT=${OUT:-${LOCAL_DIR:-/local}/glm53f-validation-${PJM_JOBID:-manual}}
THREADS=${OMP_NUM_THREADS:-48}

case "$OUT" in /local/*|*/tmp/*|"$REPO"/tmp/*) ;; *) echo "OUT must be /local or repo tmp" >&2; exit 2;; esac
mkdir -p "$OUT"
if [ -r /proc/meminfo ]; then
    avail=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    [ "$avail" -ge 6291456 ] || { echo "MemAvailable below 6 GiB" >&2; exit 3; }
fi

python3 "$REPO/a64fx/glm5/glm53f_validation.py" manifest "$MODEL" >"$OUT/manifest.txt"

# The exact sequential oracle is intentionally selected by the caller.  This
# wrapper provides the bounded preflight and stable output locations used by
# both the reference executable and llama-debug.
printf 'out=%s\nmodel=%s\nthreads=%s\n' "$OUT" "$MODEL" "$THREADS" >"$OUT/run.env"
printf 'k_norm_eps=1e-6\nindex_topk=2048\nkpool=4\nexperts=288\ntop_k=8\nrouted_scale=2.5\n' >"$OUT/contract.txt"
if [ "$#" -gt 0 ]; then
    "$@" >"$OUT/reference.stdout" 2>"$OUT/reference.stderr"
fi
echo "GLM53F_VALIDATION_PREFLIGHT PASS out=$OUT"
