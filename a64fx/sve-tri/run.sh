#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Building..."
make clean
make

echo ""
echo "Running benchmark (single core)..."
OMP_NUM_THREADS=1 ./bench
