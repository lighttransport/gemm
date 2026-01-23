#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:15:00"
#PJM -g hp250467
#PJM --no-check-directory
#PJM -o fapp_cache.out
#PJM -e fapp_cache.err

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/sector-cache

echo "=== Building Sector Cache Test with fapp ==="
make clean
make fapp

echo ""
echo "=== Running with fapp Cache Profiling ==="

# Clean up old profiling data
rm -rf fapp_cache_output

# Run with cache event profiling
# Cache event includes L1/L2 cache hit/miss information
fapp -C -d fapp_cache_output -Hevent=Cache ./bench_sector_cache_fapp

echo ""
echo "=== fapp Cache Profile Report ==="
fapppx -A -d fapp_cache_output -Icpu -tcsv -o cache_report.csv

# Display cache report
echo ""
echo "=== Cache Report (CSV) ==="
cat cache_report.csv

echo ""
echo "=== Human Readable Report ==="
fapppx -A -d fapp_cache_output -Icpu

echo ""
echo "=== Done ==="
