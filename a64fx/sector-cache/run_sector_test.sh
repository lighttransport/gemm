#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00"
#PJM -g hp250467
#PJM --no-check-directory
#PJM -o sector_test.out
#PJM -e sector_test.err

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/sector-cache

echo "=== Building Sector Cache Test ==="
make clean
make

echo ""
echo "=== Running Sector Cache Test ==="
./bench_sector_cache

echo ""
echo "=== Done ==="
