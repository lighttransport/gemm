#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:15:00"
#PJM -g hp250467
#PJM -o fapp_v2.out
#PJM -e fapp_v2.err

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/sector-cache

echo "=== Check fapp options ==="
fapp -h 2>&1 | head -50

echo ""
echo "=== Running with basic fapp (default profiling) ==="
rm -rf fapp_default
fapp -C -d fapp_default ./bench_sector_cache

echo ""
echo "=== fapp default Report ==="
fapppx -A -d fapp_default -Icpupa 2>&1 || fapppx -A -d fapp_default 2>&1

echo ""
echo "=== Done ==="
