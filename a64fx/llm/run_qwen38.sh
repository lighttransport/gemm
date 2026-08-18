#!/bin/sh
set -eu

usage() {
    echo "usage: $0 --model PATH --mode single|pp|tp --nodes N [runner options]" >&2
    exit 2
}

model= mode= nodes=
rest=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) [ "$#" -ge 2 ] || usage; model=$2; shift 2 ;;
        --mode) [ "$#" -ge 2 ] || usage; mode=$2; shift 2 ;;
        --nodes) [ "$#" -ge 2 ] || usage; nodes=$2; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done
[ -n "$model" ] && [ -n "$mode" ] && [ -n "$nodes" ] || usage
[ -r "$model" ] || { echo "model is not readable: $model" >&2; exit 1; }
case "$nodes" in *[!0-9]*|'') usage;; esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
threads=${LLM_THREADS:-48}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$threads}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

case "$mode" in
    single)
        [ "$nodes" -eq 1 ] || { echo "single mode requires --nodes 1" >&2; exit 2; }
        make -C "$here" qwen38_runner CC=fcc OPENMP=1
        exec "$here/build/qwen38_runner" "$model" --threads "$threads" "$@"
        ;;
    pp)
        [ "$nodes" -ge 2 ] || { echo "pp mode requires at least two nodes" >&2; exit 2; }
        [ "${PJM_MPI_PROC:-$nodes}" -ge "$nodes" ] || { echo "allocation has fewer than $nodes ranks" >&2; exit 2; }
        make -C "$here" qwen38_pp_runner CC=fcc OPENMP=1
        export OPAL_PREFIX=/opt/FJSVxtclanga/tcsds-1.2.43
        exec mpiexec -np "$nodes" "$here/build/qwen38_pp_runner" "$model" --threads "$threads" "$@"
        ;;
    tp)
        echo "Qwen3.8 TP is not available: the repository TP slicing API must be restored before this mode is safe." >&2
        exit 3
        ;;
    *) usage ;;
esac
