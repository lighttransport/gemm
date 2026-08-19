#!/bin/sh
set -eu

usage() {
    echo "usage: $0 --model PATH --mode single|pp|tp --nodes N [--stage-dir DIR] [runner options]" >&2
    exit 2
}

model= mode= nodes= stage_dir=
rest=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) [ "$#" -ge 2 ] || usage; model=$2; shift 2 ;;
        --mode) [ "$#" -ge 2 ] || usage; mode=$2; shift 2 ;;
        --nodes) [ "$#" -ge 2 ] || usage; nodes=$2; shift 2 ;;
        --stage-dir) [ "$#" -ge 2 ] || usage; stage_dir=$2; shift 2 ;;
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

if [ -n "$stage_dir" ]; then
    staged_model=$stage_dir/$(basename -- "$model")
    if [ "$nodes" -eq 1 ]; then
        "$here/stage_gguf_shards.sh" "$model" "$stage_dir"
    else
        export OPAL_PREFIX=/opt/FJSVxtclanga/tcsds-1.2.43
        mpiexec -np "$nodes" "$here/stage_gguf_shards.sh" "$model" "$stage_dir"
    fi
    model=$staged_model
fi

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
        [ "$nodes" -ge 2 ] || { echo "tp mode requires at least two nodes" >&2; exit 2; }
        prompt=Hello; maxgen=16; maxseq=512; speck=0; token_id=
        while [ "$#" -gt 0 ]; do
            case "$1" in
                --prompt) prompt=$2; shift 2 ;;
                --max-gen) maxgen=$2; shift 2 ;;
                --max-seq) maxseq=$2; shift 2 ;;
                --spec-k) speck=$2; shift 2 ;;
                --token-id) token_id=$2; shift 2 ;;
                *) echo "unsupported TP option: $1" >&2; exit 2 ;;
            esac
        done
        make -C "$here" tp_runner CC=fcc OPENMP=1
        (cd "$here" && OPAL_PREFIX=/opt/FJSVxtclanga/tcsds-1.2.43 \
            mpiexec -np "$nodes" ../utofu-tests/tofu_topo_helper)
        export TP_PROMPT=$prompt TP_MAXGEN=$maxgen TP_MAXSEQ=$maxseq TP_SPEC_K=$speck
        export TP_PREFILL_GEMM=${TP_PREFILL_GEMM:-0} TF_NO_PANEL=${TF_NO_PANEL:-1}
        if [ -n "$token_id" ]; then
            export TP_SYNTH_TOKENS=1 TP_SYNTH_TOKEN_ID=$token_id
        fi
        cd "$here"
        exec mpiexec -np "$nodes" ./build/tp_runner "$model"
        ;;
    *) usage ;;
esac
