#!/bin/bash
# A64FX Sector Cache L1 Way-Conflict — fapp PMU Profiling
#
# Measures L1D cache miss/hit and sector cache tag events
# to verify sector cache partitioning effect.
#
# Usage (direct on compute node):
#   FLIB_SCCR_CNTL=TRUE FLIB_L1_SCCR_CNTL=TRUE ./run_fapp_sector_conflict.sh
#
# Usage (batch):
#   pjsub -g hp250467 -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00" \
#     --no-check-directory run_fapp_sector_conflict.sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export FLIB_SCCR_CNTL=TRUE
export FLIB_L1_SCCR_CNTL=TRUE

echo "============================================"
echo " Sector Cache L1 Way-Conflict PMU Profiling"
echo "============================================"
echo ""
echo "Working directory: $(pwd)"
echo "Date: $(date)"
echo ""
echo "FLIB_SCCR_CNTL=$FLIB_SCCR_CNTL"
echo "FLIB_L1_SCCR_CNTL=$FLIB_L1_SCCR_CNTL"
echo ""

# ------------------------------------------------------------------
# Phase 0: Baseline (no fapp, verify sector cache works)
# ------------------------------------------------------------------
echo "=== Phase 0: Baseline (no fapp) ==="
./test_sector_l1_conflict
echo ""

# ------------------------------------------------------------------
# Phase 1: fapp — L1/L2 Cache Performance Events
# ------------------------------------------------------------------
echo "=== Phase 1: fapp L1/L2 Cache Events ==="
echo ""
echo "Events:"
echo "  0x0011  CPU_CYCLES"
echo "  0x0003  L1D_CACHE_REFILL      (L1D misses → refill from L2)"
echo "  0x0004  L1D_CACHE             (L1D accesses)"
echo "  0x0016  L2D_CACHE             (L2 accesses)"
echo "  0x0017  L2D_CACHE_REFILL      (L2 misses → refill from memory)"
echo "  0x0015  L1D_CACHE_WB          (L1D writebacks)"
echo "  0x0008  INST_RETIRED          (instructions retired)"
echo "  0x0049  L1D_CACHE_REFILL_PRF  (L1D prefetch refills)"
echo ""

PROF_CACHE=prof_conflict_cache
rm -rf "$PROF_CACHE"
fapp -C -d "$PROF_CACHE" -Icpupa \
  -Hevent_raw=0x0011,0x0003,0x0004,0x0016,0x0017,0x0015,0x0008,0x0049,method=fast,mode=user \
  ./test_sector_l1_conflict_fapp

echo ""
echo "Exporting cache results..."
fapppx -A -Icpupa -tcsv  -o prof_conflict_cache.csv  -d "$PROF_CACHE"
fapppx -A -Icpupa -ttext -o prof_conflict_cache.txt  -d "$PROF_CACHE"
echo ""

echo "--- L1/L2 Cache Results ---"
cat prof_conflict_cache.txt
echo ""

# ------------------------------------------------------------------
# Phase 2: fapp — Sector Cache Tag Events
# ------------------------------------------------------------------
echo "=== Phase 2: fapp Sector Cache Tag Events ==="
echo ""
echo "Events:"
echo "  0x0011  CPU_CYCLES"
echo "  0x0240  L1_PIPE0_VAL_IU_TAG_ADRS      (pipe0 tagged loads)"
echo "  0x0241  L1_PIPE1_VAL_IU_TAG_ADRS      (pipe1 tagged loads)"
echo "  0x02a0  L1_PIPE0_VAL_IU_NOT_SEC0      (pipe0 sector!=0)"
echo "  0x02a1  L1_PIPE1_VAL_IU_NOT_SEC0      (pipe1 sector!=0)"
echo "  0x0250  L1_PIPE0_VAL_IU_TAG_ADRS_SCE  (pipe0 SCE=1)"
echo "  0x0252  L1_PIPE1_VAL_IU_TAG_ADRS_SCE  (pipe1 SCE=1)"
echo "  0x0260  L1_PIPE0_VAL_IU_TAG_ADRS_PFE  (pipe0 PFE=1)"
echo ""

PROF_TAG=prof_conflict_tag
rm -rf "$PROF_TAG"
fapp -C -d "$PROF_TAG" -Icpupa \
  -Hevent_raw=0x0011,0x0240,0x0241,0x02a0,0x02a1,0x0250,0x0252,0x0260,method=fast,mode=user \
  ./test_sector_l1_conflict_fapp

echo ""
echo "Exporting tag results..."
fapppx -A -Icpupa -tcsv  -o prof_conflict_tag.csv  -d "$PROF_TAG"
fapppx -A -Icpupa -ttext -o prof_conflict_tag.txt  -d "$PROF_TAG"
echo ""

echo "--- Sector Cache Tag Results ---"
cat prof_conflict_tag.txt
echo ""

# ------------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------------
echo "============================================"
echo " Interpretation Guide"
echo "============================================"
echo ""
echo "Phase 1 (Cache Events) — key metric: L1D_CACHE_REFILL (0x0003)"
echo "  nohint_all: high L1D_CACHE_REFILL → keep data evicted, reloads from L2"
echo "  sector_all: low  L1D_CACHE_REFILL → keep data stays in L1 sector 0"
echo "  Expected:   nohint_refill >> sector_refill (sector cache prevents eviction)"
echo ""
echo "Phase 2 (Tag Events) — key metric: NOT_SEC0 (0x02a0 + 0x02a1)"
echo "  nohint_all: NOT_SEC0 ≈ 0 (no tags, all loads use default sector 0)"
echo "  sector_all: NOT_SEC0 > 0 (evict loads tagged to sector 1)"
echo "  Also: SCE (0x0250 + 0x0252) should be high in sector_all"
echo ""
echo "Done. $(date)"
