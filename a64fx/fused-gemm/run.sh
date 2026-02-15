#!/bin/bash
# Run script for fused GEMM benchmark on A64FX Fugaku

# Set thread count (12 cores per CMG, 4 CMGs per node = 48 cores)
# For single CMG testing:
export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# For full node (48 cores):
# export OMP_NUM_THREADS=48

# Disable NUMA auto-interleave for L2 residency
export FLIB_SCCR_CNTL=FALSE

echo "=== Fused GEMM Benchmark ==="
echo "Threads: $OMP_NUM_THREADS"
echo "Date: $(date)"
echo ""

./bench_fused
