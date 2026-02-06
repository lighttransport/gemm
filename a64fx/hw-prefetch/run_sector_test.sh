#!/bin/bash
#PJM -g hp250467
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:05:00"
#PJM --no-check-directory
#PJM -j
#PJM -S

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/hw-prefetch

echo "=== Build on compute node ==="
./build_sector_test.sh native
echo ""

echo "=== Run sector cache test ==="
./test_sector_cache
echo ""

echo "=== Assembly analysis ==="
echo ""
echo "Key functions and their sector cache instructions:"
echo ""
for func in test_sector_stream_vs_resident test_nosector_stream_vs_resident \
            test_gemm_sector test_gemm_nosector \
            test_sector_config_1_3 test_sector_config_2_2 test_sector_config_3_1; do
    has_sector=$(grep -c "__jwe_xset_sccr" test_sector_cache.s 2>/dev/null || echo "0")
    has_orr=$(grep -c "orr.*72057594037927936" test_sector_cache.s 2>/dev/null || echo "0")
    echo "  $func: sccr_calls=$has_sector, tagged_addrs=$has_orr"
done
echo ""
echo "=== Done ==="
