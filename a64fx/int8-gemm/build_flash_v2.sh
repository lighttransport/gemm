#!/bin/bash
# Build script for Fused FlashAttention V2
# Run this on Fugaku compute node

set -e

echo "Building Fused FlashAttention V2..."

# Clean previous builds
make clean

# Build the fused FlashAttention V2 benchmark
make bench_flash_fused_v2

echo "Build complete!"
echo "Binary: bench_flash_fused_v2"
echo ""
echo "Usage: ./bench_flash_fused_v2 [L] [head_dim] [num_iters]"
echo "Example: ./bench_flash_fused_v2 1024 128 5"
echo "Example: ./bench_flash_fused_v2 4096 128 5"
