#!/bin/bash
# fapp PMU profiling for L1D cache set partitioning benchmark
#
# Submit with:
#   pjsub -g hp250467 -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00" --no-check-directory run_fapp.sh
set -e

cd "$(dirname "$0")"

make clean
make fapp

# SC1: Basic L1D events (8 counters)
#   CPU_CYCLES(0x0011), INST_RETIRED(0x0008),
#   L1D_CACHE(0x0004), L1D_CACHE_REFILL(0x0003),
#   L1D_CACHE_REFILL_DM(0x0200), L1D_CACHE_WB(0x0015),
#   LD_COMP_WAIT(0x0184), L1_MISS_WAIT(0x0208)
fapp -C -d ./fapp_sc1 -Icpupa \
  -Hevent_raw=0x0011,0x0008,0x0004,0x0003,0x0200,0x0015,0x0184,0x0208 \
  ./bench_cache_partition

# SC2: Prefetch detail events (4 counters)
#   L1D_CACHE_REFILL_HWPRF(0x0202), L1D_CACHE_REFILL_PRF(0x0049),
#   LD_COMP_WAIT_L2_MISS(0x0180), LD_COMP_WAIT_L1_MISS(0x0182)
fapp -C -d ./fapp_sc2 -Icpupa \
  -Hevent_raw=0x0202,0x0049,0x0180,0x0182 \
  ./bench_cache_partition

# Output results (text + csv)
fapppx -A -Icpupa -ttext -o fapp_sc1_result.txt -d ./fapp_sc1
fapppx -A -Icpupa -ttext -o fapp_sc2_result.txt -d ./fapp_sc2
fapppx -A -Icpupa -tcsv  -o fapp_sc1_result.csv -d ./fapp_sc1
fapppx -A -Icpupa -tcsv  -o fapp_sc2_result.csv -d ./fapp_sc2

echo "=== fapp profiling complete ==="
echo "Results: fapp_sc1_result.{txt,csv}  fapp_sc2_result.{txt,csv}"
