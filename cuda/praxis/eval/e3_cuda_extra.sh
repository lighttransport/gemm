#!/bin/bash
# E3 (CUDA side of the HIP-vs-CUDA validation, fresh seeds 112-135; the HIP side is /mnt/disk01/tmp/praxis_sim_hip/batches.sh jobs X_<task>_<first>).
# Same 'fast' variant as e1.sh A runs. Run ONLY when the local GPU is free (no other CUDA sweep).  bash cuda/praxis/eval/e3_cuda_extra.sh
source /home/syoyo/work/gemm/praxis/cuda/praxis/eval/common.sh
SEEDS="episodes.seeds=[112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135]"
for t in lift_cube cube_on_plate move_block; do fast X_$t $t mixed; done      # -> $OUT/X_<task> (OUT=/mnt/disk01/tmp/praxis_sim2 by default)
