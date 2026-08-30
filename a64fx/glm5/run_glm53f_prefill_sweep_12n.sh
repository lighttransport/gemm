#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
test "${PJM_MPI_PROC:-0}" -eq 12
job=${PJM_JOBID:?}
model=${GLM53F_MODEL_DIR:-$HOME/models/glm53f}
routed=${GLM53F_STAGE_DIR:-/local/glm53f-target-routed-$job}
shared=${GLM53F_SHARED_STAGE_DIR:-/local/glm53f-target-shared-$job}
core=${GLM53F_REPACK_STAGE_DIR:-/local/glm53f-target-core-$job}
log=${GLM53F_PREFILL_LOG:-prefill_sweep_$job.log}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-47}
export OMP_DYNAMIC=false OMP_WAIT_POLICY=active
export OMP_PROC_BIND=close OMP_PLACES=cores
export GLM53F_REPACK_DIR=$core
# Older trace-built core images can omit late-layer tensors.  Missing tensors
# are read once during construction into their anonymous HBM allocations;
# steady-state prefill never streams them from shared storage.
export GLM53F_REPACK_REQUIRE=${GLM53F_REPACK_REQUIRE:-0}
export GLM53F_PROFILE=1

rm -f tofu_topo.txt
mpiexec -np 12 ../utofu-tests/tofu_topo_helper
test "$(grep -vc '^#' tofu_topo.txt)" -eq 12
export GLM53F_UTOFU=1 TOFU_TOPO_PATH=$PWD/tofu_topo.txt

echo "=== batch correctness $(date) ===" | tee -a "$log"
mpiexec -np 12 ./glm53f_target_batch_check_12n \
    "$model" "$routed" "$shared" 2>&1 | tee -a "$log"

for chunk in 1 2 4 5; do
    echo "=== prefill positions=64 chunk=$chunk $(date) ===" | tee -a "$log"
    mpiexec -np 12 ./glm53f_prefill_12n \
        "$model" "$routed" "$shared" 64 "$chunk" 2>&1 | tee -a "$log"
done
echo "SENTINEL glm53f_prefill_sweep_12n=OK" | tee -a "$log"
