#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
nodes=${K3_PROBE_NODES:-${PJM_NODE:-1}}
job_tag=${PJM_JOBID:-$$}
stage_dir=${K3_MPI_STAGE_DIR:-$HOME/.cache/k3-kda-probe-$job_tag}
result_prefix="$stage_dir/result"
cleanup() {
    if [ "${K3_KEEP_PROBE:-0}" != 1 ]; then rm -rf -- "$stage_dir"; fi
}
trap cleanup EXIT HUP INT TERM

test ! -e "$stage_dir"
mkdir -p "$stage_dir"
make -C "$script_dir" k3_kda_probe
PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/k3_kda_stage.py" \
    --model-dir "${K3_MODEL_DIR:-$HOME/models/kimi-k3}" \
    --output-dir "$stage_dir" --layer "${K3_KDA_LAYER:-0}" \
    --head "${K3_KDA_HEAD:-0}"

stem="layer$(printf '%02d' "${K3_KDA_LAYER:-0}")_head$(printf '%02d' "${K3_KDA_HEAD:-0}")"
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
mpiexec -n "$nodes" -of-proc "$result_prefix" \
    -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
    -x XOS_MMM_L_PAGING_POLICY=demand:demand:demand \
    "$script_dir/k3_kda_probe" "$stage_dir/$stem.blob" "$stage_dir/$stem.manifest"

for result in "$result_prefix".*; do
    echo "===== $result ====="
    cat "$result"
done
