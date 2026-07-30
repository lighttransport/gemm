#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
nodes=${PJM_NODE:-1}
job_tag=${PJM_JOBID:-$$}
result_dir="$HOME/.cache/k3-etp-results-$job_tag"
model_dir="$HOME/models/kimi-k3"
layer=1
experts=16
threads=48
prefill=0

usage() {
    echo "usage: $0 [--nodes N] [--result-dir DIR] [--model-dir DIR]" >&2
    echo "          [--layer N] [--experts N] [--threads N] [--prefill]" >&2
}
need_value() {
    if [ "$#" -lt 2 ]; then
        echo "$0: missing value for $1" >&2
        usage
        exit 2
    fi
}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --nodes) need_value "$@"; nodes=$2; shift 2 ;;
        --result-dir) need_value "$@"; result_dir=$2; shift 2 ;;
        --model-dir) need_value "$@"; model_dir=$2; shift 2 ;;
        --layer) need_value "$@"; layer=$2; shift 2 ;;
        --experts) need_value "$@"; experts=$2; shift 2 ;;
        --threads) need_value "$@"; threads=$2; shift 2 ;;
        --prefill) prefill=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "$0: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done
for value in "$nodes" "$layer" "$experts" "$threads"; do
    case "$value" in ''|*[!0-9]*) echo "$0: numeric options must be integers" >&2; exit 2 ;; esac
done
if [ "$nodes" -eq 0 ] || [ "$experts" -eq 0 ] || [ "$threads" -eq 0 ]; then
    echo "$0: nodes, experts, and threads must be greater than zero" >&2
    exit 2
fi

result_prefix="$result_dir/result"
created=0
cleanup() {
    if [ "$created" = 1 ] && [ "${K3_KEEP_RESULTS:-0}" != 1 ]; then rm -rf -- "$result_dir"; fi
}
trap cleanup EXIT HUP INT TERM

if [ -e "$result_dir" ]; then
    echo "$0: result directory already exists: $result_dir" >&2
    exit 2
fi
mkdir -p "$result_dir"
created=1
make -C "$script_dir" k3_moe_probe
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
mpiexec -n "$nodes" -of-proc "$result_prefix" \
    -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
    -x XOS_MMM_L_PAGING_POLICY=demand:demand:demand \
    "$script_dir/run_expert_tp_probe_rank.sh" "$script_dir" "$nodes" "$job_tag" \
    "$model_dir" "$layer" "$experts" "$threads" "$prefill"

passes=$(grep -l 'K3 expert-TP probe: PASS' "$result_prefix".* | wc -l)
if [ "$passes" -ne "$nodes" ]; then
    echo "K3 expert-TP probe: FAIL ($passes/$nodes ranks passed)" >&2
    exit 1
fi
if [ "$prefill" -eq 1 ]; then grep 'PROBE expert-tp-prefill' "$result_prefix".*; else grep 'PROBE expert-tp-selected' "$result_prefix".*; fi
echo "K3 multi-node expert-TP probe: PASS ($passes/$nodes ranks)"
