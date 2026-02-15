#!/bin/bash
#PJM -g hp250467
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:05:00"
#PJM -j
#PJM -S

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/hw-prefetch

echo "=== Build on compute node ==="
./build_sector_test.sh native
echo ""

# Required runtime environment variables for sector cache:
#   FLIB_HPCFUNC=TRUE  - enables HPC function features (tag addressing, SCCR)
#                        writes '1' to /sys/kernel/xos_hpc/hwpf_uaccess
#   FLIB_SCCR_CNTL=TRUE - enables sector cache control register management
#   FLIB_L1_SCCR_CNTL=FALSE - disable L1 SCCR fallback
export FLIB_HPCFUNC=TRUE
export FLIB_SCCR_CNTL=TRUE
export FLIB_L1_SCCR_CNTL=FALSE

echo "=== Runtime environment ==="
echo "  FLIB_HPCFUNC=$FLIB_HPCFUNC"
echo "  FLIB_SCCR_CNTL=$FLIB_SCCR_CNTL"
echo "  FLIB_L1_SCCR_CNTL=$FLIB_L1_SCCR_CNTL"
echo ""

echo "=== Check /sys/kernel/xos_hpc/hwpf_uaccess ==="
if [ -f /sys/kernel/xos_hpc/hwpf_uaccess ]; then
    echo "  hwpf_uaccess = $(cat /sys/kernel/xos_hpc/hwpf_uaccess)"
else
    echo "  hwpf_uaccess: file not found"
fi
echo ""

echo "=== Run sector cache test ==="
./test_sector_cache
echo ""

echo "=== Assembly analysis ==="
echo ""
echo "Sector cache instructions in assembly:"
sccr_calls=$(grep -c "__jwe_xset_sccr" test_sector_cache.s 2>/dev/null || echo "0")
tagged_addrs=$(grep -c "orr.*72057594037927936" test_sector_cache.s 2>/dev/null || echo "0")
echo "  __jwe_xset_sccr calls: $sccr_calls"
echo "  ORR tagged addresses (bit 56): $tagged_addrs"
echo ""
echo "=== Done ==="
