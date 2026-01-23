#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:20:00"
#PJM -g hp250467
#PJM -o fapp_detailed.out
#PJM -e fapp_detailed.err

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/sector-cache

echo "=== Building Sector Cache Test with fapp ==="
make clean
make fapp

if [ ! -f bench_sector_cache_fapp ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo ""
echo "=== fapp Detailed Cache Profiling ==="
echo "This measures L1 cache sector behavior with tagged pointers"
echo ""

# Clean up old profiling data
rm -rf fapp_output

# Run with fapp - use pa1 for cache events (Performance Analysis level 1)
echo "=== Running with fapp pa1 (Cache events) ==="
fapp -C -d fapp_output -Hpa=1 ./bench_sector_cache_fapp

echo ""
echo "=== fapp Report ==="
fapppx -A -d fapp_output -Icpupa -tcsv -o cache_report.csv
echo ""
echo "CSV Report:"
cat cache_report.csv

echo ""
echo "=== Human Readable Report ==="
fapppx -A -d fapp_output -Icpupa

echo ""
echo "==================================================="
echo "=== Key Metrics to Check ==="
echo "==================================================="
echo "Look for these PMU events:"
echo "  - L1D_CACHE_REFILL: L1 data cache line refills (misses)"
echo "  - L1D_CACHE: L1 data cache accesses"
echo "  - L2D_CACHE_REFILL: L2 cache refills from memory"
echo ""
echo "Compare these metrics between the profiled regions:"
echo "  - streaming_no_hint vs streaming_sector1"
echo "  - reuse_no_hint vs reuse_sector0"
echo "  - mixed_no_hint vs mixed_with_hint (most important!)"
echo ""

echo "=== Done ==="
