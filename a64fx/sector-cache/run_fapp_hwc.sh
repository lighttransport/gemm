#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:15:00"
#PJM -g hp250467
#PJM -o fapp_hwc.out
#PJM -e fapp_hwc.err

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/sector-cache

echo "=== Check fapppx options ==="
fapppx -h 2>&1 | head -100

echo ""
echo "=== Running with fapp hardware counters ==="
rm -rf fapp_hwc

# Try -Ihwm for hardware monitor
fapp -C -d fapp_hwc -Ihwm ./bench_sector_cache

echo ""
echo "=== fapp HWC Report with cpupa ==="
fapppx -A -d fapp_hwc -Icpupa 2>&1

echo ""
echo "=== fapp HWC Report with hwm ==="
fapppx -A -d fapp_hwc -Ihwm 2>&1

echo ""
echo "=== fapp HWC Report default ==="
fapppx -A -d fapp_hwc 2>&1

echo ""
echo "=== Done ==="
