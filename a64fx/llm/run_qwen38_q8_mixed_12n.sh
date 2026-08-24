#!/bin/bash
# One resident-weight lifetime: PP3xTP4 prefill for three requests, transpose
# runtime state in memory, then three concurrent TP4 BF16-PV decode replicas.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-/home/u14346/models/qwen38/27b/Qwen3.8-27B-Q8_0.gguf}
STAGE=${TP_STAGE_DIR:-/local/u14346/qwen38-q8-tp4}
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/bin"
export PJM_MPI_PROC=12 TP_STAGE_DIR=$STAGE
export LLM_THREADS=${LLM_THREADS:-48} OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread} OMP_PLACES=${OMP_PLACES:-cores}
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive} KMP_BLOCKTIME=${KMP_BLOCKTIME:-0}
export OMP_DYNAMIC=${OMP_DYNAMIC:-false}
export NUMA_DISTRIBUTE=${NUMA_DISTRIBUTE:-1} NUMA_N_CMGS=${NUMA_N_CMGS:-4}
export NUMA_CMG_BUDGET_GB=${NUMA_CMG_BUDGET_GB:-7}
export NUMA_ALIGNMENT=${NUMA_ALIGNMENT:-2097152}
export TF_LOAD_KEEPCACHE=0 TF_NO_PANEL=1 TP_STAGE_BF16_PV=0
export TF_KEEP_BF16_SRC=1
export TF_SSM_FUSED_DOTS=${TF_SSM_FUSED_DOTS:-1}
export TF_SILU_SVE=${TF_SILU_SVE:-1}
export TF_BF16PV_PREFETCH=${TF_BF16PV_PREFETCH:-8}
export TP_Q8_BF16_RESERVE_GB=${TP_Q8_BF16_RESERVE_GB:-3}
unset GGUF_LAZY_MMAP TF_FORCE_MMAP

cd "$HERE"
make qwen38_mixed_runner CC=fcc OPENMP=1
make -C ../utofu-tests tofu_topo_helper >/dev/null
mpiexec -np 12 ../utofu-tests/tofu_topo_helper
exec mpiexec -np 12 ./build/qwen38_mixed_runner "$MODEL"
