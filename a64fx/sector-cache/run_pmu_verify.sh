#!/bin/bash
# A64FX Sector Cache Tag Bit PMU Verification Script
#
# Runs test_sector_tag_pmu with fapp to capture SC3 (sector utilization)
# and SC1 (L1 cache performance) PMU events.
#
# Usage (direct on compute node):
#   ./run_pmu_verify.sh
#
# Usage (batch via pjsub):
#   Uncomment PJM headers below, then:
#   pjsub -g hp250467 run_pmu_verify.sh

## --- PJM headers (uncomment for batch submission) ---
##PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00"
##PJM -j
##PJM -S

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Output directories
SC3_DIR=prof_sc3
SC1_DIR=prof_sc1

echo "============================================"
echo " A64FX Sector Cache Tag Bit PMU Verification"
echo "============================================"
echo ""
echo "Working directory: $(pwd)"
echo "Date: $(date)"
echo ""

# ------------------------------------------------------------------
# Phase 0: Build
# ------------------------------------------------------------------
echo "=== Phase 0: Build ==="
make pmu
make pmu_fapp
echo ""

# ------------------------------------------------------------------
# Phase 1: Cycle-count baseline (no fapp)
# ------------------------------------------------------------------
echo "=== Phase 1: Cycle-count baseline (no fapp) ==="
echo ""
./test_sector_tag_pmu
echo ""

# ------------------------------------------------------------------
# Phase 2: fapp with SC3 events (sector utilization -- the key test)
# ------------------------------------------------------------------
echo "=== Phase 2: fapp SC3 events (sector utilization) ==="
echo ""
echo "Events:"
echo "  0x0011  EA_OTHER_CORE_RETIREMENT_STALL (cycle reference)"
echo "  0x0240  L1_PIPE0_VAL_IU_TAG_ADRS       (pipe0 tagged loads)"
echo "  0x0241  L1_PIPE1_VAL_IU_TAG_ADRS       (pipe1 tagged loads)"
echo "  0x0250  L1_PIPE0_VAL_IU_TAG_ADRS_SCE   (pipe0 SCE=1)"
echo "  0x0252  L1_PIPE1_VAL_IU_TAG_ADRS_SCE   (pipe1 SCE=1)"
echo "  0x02a0  L1_PIPE0_VAL_IU_NOT_SEC0       (pipe0 sector_id!=0)"
echo "  0x02a1  L1_PIPE1_VAL_IU_NOT_SEC0       (pipe1 sector_id!=0)"
echo "  0x0260  L1_PIPE0_VAL_IU_TAG_ADRS_PFE   (pipe0 PFE=1)"
echo ""

rm -rf "$SC3_DIR"
fapp -C -d "$SC3_DIR" -Icpupa \
  -Hevent_raw=0x0011,0x0240,0x0241,0x0250,0x0252,0x02a0,0x02a1,0x0260,method=fast,mode=user \
  ./test_sector_tag_pmu_fapp

echo ""
echo "Exporting SC3 results..."
fapppx -A -Icpupa -ttext -o prof_sc3.txt -d "$SC3_DIR"
fapppx -A -Icpupa -tcsv  -o prof_sc3.csv -d "$SC3_DIR"
echo "  -> prof_sc3.txt, prof_sc3.csv"
echo ""

echo "--- SC3 Results ---"
cat prof_sc3.txt
echo ""

# ------------------------------------------------------------------
# Phase 3: fapp with SC1 events (L1 cache performance)
# ------------------------------------------------------------------
echo "=== Phase 3: fapp SC1 events (L1 cache performance) ==="
echo ""
echo "Events:"
echo "  0x0011  EA_OTHER_CORE_RETIREMENT_STALL"
echo "  0x0008  INST_COMMIT_FP              (FP instructions committed)"
echo "  0x0004  L1D_CACHE_MISS              (L1D cache misses)"
echo "  0x0003  L1D_CACHE                   (L1D cache accesses)"
echo "  0x0200  L1_PIPE0_VAL_IU             (pipe0 valid IU ops)"
echo "  0x0015  CPU_CYCLES                  (cycle counter)"
echo "  0x0184  L2D_CACHE_REFILL_DM_WRITE   (L2 dirty refill)"
echo "  0x0180  L2D_CACHE_REFILL_DM_READ    (L2 demand read refill)"
echo ""

rm -rf "$SC1_DIR"
fapp -C -d "$SC1_DIR" -Icpupa \
  -Hevent_raw=0x0011,0x0008,0x0004,0x0003,0x0200,0x0015,0x0184,0x0180,method=fast,mode=user \
  ./test_sector_tag_pmu_fapp

echo ""
echo "Exporting SC1 results..."
fapppx -A -Icpupa -ttext -o prof_sc1.txt -d "$SC1_DIR"
fapppx -A -Icpupa -tcsv  -o prof_sc1.csv -d "$SC1_DIR"
echo "  -> prof_sc1.txt, prof_sc1.csv"
echo ""

echo "--- SC1 Results ---"
cat prof_sc1.txt
echo ""

# ------------------------------------------------------------------
# Interpretation Guide
# ------------------------------------------------------------------
echo "============================================"
echo " Interpretation Guide"
echo "============================================"
echo ""
echo "SC3 ratios per fapp region:"
echo "  SCE_usage_ratio  = (0x0250 + 0x0252) / (0x0240 + 0x0241)"
echo "  NOT_SEC0_ratio   = (0x02a0 + 0x02a1) / (0x0240 + 0x0241)"
echo ""
echo "Expected results:"
echo "  Region    SCE_ratio    NOT_SEC0_ratio   L1 bypass"
echo "  tag_0x0   ~0%          ~0%              baseline"
echo "  tag_0x2   ~0%          >50%             ~1.0x"
echo "  tag_0xA   >50%         >50%             ~1.0x"
echo "  tag_0xB   >50%         >50%             ~1.5-2.0x"
echo ""
echo "PASS criteria:"
echo "  - tag_0xA, tag_0xB: SCE > 50%"
echo "  - tag_0x2, tag_0xA, tag_0xB: NOT_SEC0 > 50%"
echo "  - tag_0x0: both near 0%"
echo "  - tag_0xB: cycle ratio ~1.5-2.0x vs tag_0x0"
echo ""
echo "FAIL indicators:"
echo "  - All regions ~0%: compiler stripped tags (check FORCE_PTR, check asm)"
echo "  - tag_0xB same speed as tag_0x0: tags not reaching hardware"
echo "  - Only NOT_SEC0 fires (not SCE): bit layout differs from model"
echo ""
echo "To check assembly:"
echo "  fcc -Nclang -O2 -march=armv8.2-a+sve -ffj-no-largepage -S test_sector_tag_pmu.c"
echo ""
echo "Done. $(date)"
