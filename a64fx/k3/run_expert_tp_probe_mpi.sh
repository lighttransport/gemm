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
logical_tp=0
logical_waves=1

usage() {
    echo "usage: $0 [--nodes N] [--result-dir DIR] [--model-dir DIR]" >&2
    echo "          [--layer N] [--experts N] [--threads N] [--prefill]" >&2
    echo "          [--logical-tp N] [--logical-waves N]" >&2
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
        --logical-tp) need_value "$@"; logical_tp=$2; shift 2 ;;
        --logical-waves) need_value "$@"; logical_waves=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "$0: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done
for value in "$nodes" "$layer" "$experts" "$threads" "$logical_tp" "$logical_waves"; do
    case "$value" in ''|*[!0-9]*) echo "$0: numeric options must be integers" >&2; exit 2 ;; esac
done
if [ "$nodes" -eq 0 ] || [ "$experts" -eq 0 ] || [ "$experts" -gt 16 ] ||
   [ "$threads" -eq 0 ] || [ "$threads" -gt 48 ] || [ "$logical_waves" -eq 0 ]; then
    echo "$0: require nodes>0, experts=1..16, threads=1..48, and logical-waves>0" >&2
    exit 2
fi
if [ "$logical_tp" -eq 0 ]; then
    if [ "$prefill" -eq 1 ]; then logical_tp=96; else logical_tp=$nodes; fi
fi
if [ "$logical_tp" -lt "$nodes" ] || [ "$logical_tp" -gt 96 ] ||
   [ $((logical_tp % nodes)) -ne 0 ]; then
    echo "$0: --logical-tp must be a multiple of --nodes in [nodes,96]" >&2
    exit 2
fi
max_waves=$((logical_tp / nodes))
if [ "$logical_waves" -gt "$max_waves" ]; then
    echo "$0: --logical-waves cannot exceed logical-tp/nodes ($max_waves)" >&2
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
    "$model_dir" "$layer" "$experts" "$threads" "$prefill" "$logical_tp" "$logical_waves"

passes=$(grep -l 'K3 expert-TP probe: PASS' "$result_prefix".* | wc -l)
sample_passes=$(grep -h -c 'K3 expert-TP probe: PASS' "$result_prefix".* | awk '{s+=$1} END{print s+0}')
expected_passes=$((nodes * logical_waves))
if [ "$passes" -ne "$nodes" ] || [ "$sample_passes" -ne "$expected_passes" ]; then
    echo "K3 expert-TP probe: FAIL ($passes/$nodes rank files, $sample_passes/$expected_passes samples passed)" >&2
    exit 1
fi
if [ "$prefill" -eq 1 ]; then grep 'PROBE expert-tp-prefill' "$result_prefix".*; else grep 'PROBE expert-tp-selected' "$result_prefix".*; fi
echo "K3 multi-node expert-TP probe: PASS ($passes/$nodes ranks, $sample_passes/$expected_passes logical samples, TP=$logical_tp)"
