#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
nodes=${K3_PROBE_NODES:-${PJM_NODE:-1}}
job_tag=${PJM_JOBID:-$$}
result_dir=${K3_MPI_RESULT_DIR:-$HOME/.cache/k3-moe-results-$job_tag}
result_prefix="$result_dir/result"
cleanup() {
    if [ "${K3_KEEP_RESULTS:-0}" != 1 ]; then rm -rf -- "$result_dir"; fi
}
trap cleanup EXIT HUP INT TERM

test ! -e "$result_dir"
mkdir -p "$result_dir"
make -C "$script_dir" k3_moe_probe
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
mpiexec -n "$nodes" -of-proc "$result_prefix" \
    -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
    -x XOS_MMM_L_PAGING_POLICY=demand:demand:demand \
    "$script_dir/run_moe_probe_rank.sh" "$script_dir" "$nodes" "$job_tag"

passes=$(grep -l 'K3 MoE probe: PASS' "$result_prefix".* | wc -l)
if [ "$passes" -ne "$nodes" ]; then
    echo "K3 12-node MoE probe: FAIL ($passes/$nodes ranks passed)" >&2
    exit 1
fi
for result in "$result_prefix".*; do
    echo "===== $result ====="
    cat "$result"
done
echo "K3 multi-node MoE probe: PASS ($passes/$nodes ranks)"
