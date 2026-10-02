#!/bin/bash
# Native images only: does not construct or retain an inference model.
set -euo pipefail
if [ "$#" -lt 4 ]; then
    echo "usage: $0 GGUF /local/IMAGE_ROOT BIN_DIR LOG_DIR [--pipeline-cuts A,B]" >&2
    exit 2
fi
gguf=$1
image_root=$(realpath -m "$2")
bin_dir=$(realpath -m "$3")
log_dir=$(realpath -m "$4")
shift 4
case "$image_root" in /local/?*) ;; *) echo "image root must be below /local" >&2; exit 2;; esac
[ -f "$gguf" ] || { echo "missing GGUF: $gguf" >&2; exit 2; }
[ "${PJM_NODE:-0}" = 12 ] && [ "${PJM_MPI_PROC:-0}" = 12 ] || { echo "requires a twelve-node allocation" >&2; exit 2; }
options=(--parallel-layout pp3-tp4)
if [ "$#" != 0 ]; then
    [ "$#" = 2 ] && [ "$1" = --pipeline-cuts ] || { echo "only --pipeline-cuts A,B is accepted" >&2; exit 2; }
    options+=("$@")
fi
# Never launch staging beside a resident model or another native MPI job.
if ps -u "$USER" -o args= | grep -Eq '[m]piwrapp mpiexec -n 12|[o]rg/mpiexec -n 12'; then
    echo "refusing overlapping MPI job" >&2; exit 2
fi
mkdir -p "$log_dir"
for component in routed dense shared kda sparse embed head core; do
    binary=$bin_dir/glm53f_pp_${component}_stage
    [ -x "$binary" ] || { echo "missing executable: $binary" >&2; exit 2; }
done
mpiexec -n 12 mkdir -p "$image_root"
for component in routed dense shared kda sparse embed head core; do
    prefix=$log_dir/pp-stage-$component
    # Distinct logs are needed to attribute every rank to this staging run.
    if compgen -G "$prefix.*" >/dev/null; then echo "existing staging logs: $prefix" >&2; exit 2; fi
    mpiexec -n 12 -of-proc "$prefix" "$bin_dir/glm53f_pp_${component}_stage" \
        "$gguf" "$image_root/$component" "${options[@]}"
    for rank in {0..11}; do
        files=("$prefix".*."$rank")
        [ "${#files[@]}" = 1 ] && [ -f "${files[0]}" ] || { echo "missing or ambiguous rank log: $component rank=$rank" >&2; exit 2; }
        # Constructors verify layout, ownership, shape and payload hashes.
        # MPI success plus the component's terminal sentinel qualifies staging.
        grep -Eq 'SENTINEL .*=(OK|REUSE|SKIP)|GLM53F_PP_CORE_STAGE .* PASS' "${files[0]}" || {
            echo "missing staging sentinel: $component rank=$rank" >&2; exit 2;
        }
    done
    sha256sum "$bin_dir/glm53f_pp_${component}_stage"
done
echo GLM53F_PP_STAGE_PASS
