#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:15:00"
#PJM -g hp250467
#PJM -o fapp_simple.out
#PJM -e fapp_simple.err

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/sector-cache

echo "=== Building Sector Cache Test ==="
make clean
make

if [ ! -f bench_sector_cache ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo ""
echo "=== Running without fapp first (baseline) ==="
./bench_sector_cache

echo ""
echo "=== Running with fapp pa1 (Cache events) ==="
rm -rf fapp_pa1
fapp -C -d fapp_pa1 -Hpa=1 ./bench_sector_cache

echo ""
echo "=== fapp pa1 Report ==="
fapppx -A -d fapp_pa1 -Icpupa

echo ""
echo "=== Running with fapp pa2 (more detailed) ==="
rm -rf fapp_pa2
fapp -C -d fapp_pa2 -Hpa=2 ./bench_sector_cache

echo ""
echo "=== fapp pa2 Report ==="
fapppx -A -d fapp_pa2 -Icpupa

echo ""
echo "=== Done ==="
