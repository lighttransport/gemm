#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00"
#PJM -g hp250467
#PJM -o sector_v2.out
#PJM -e sector_v2.err

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/sector-cache

echo "=== Building Improved Sector Cache Test ==="
fcc -Nclang -Ofast -march=armv8.2-a+sve -ffj-ocl -ffj-no-largepage \
    -o bench_sector_v2 bench_sector_v2.c -lm

echo ""
echo "=== Running Improved Sector Cache Test ==="
./bench_sector_v2

echo ""
echo "=== Running with fapp profiling ==="
rm -rf fapp_sector_v2
fapp -C -d fapp_sector_v2 -Ihwm ./bench_sector_v2

echo ""
echo "=== fapp Performance Report ==="
fapppx -A -d fapp_sector_v2 -Icpupa

echo ""
echo "=== Done ==="
