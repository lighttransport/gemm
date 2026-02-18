#!/bin/bash
#
# Run PMU example on compute node.
#
# Native run (this host is A64FX):
#   ./run_pmu_example.sh
#
# Submit via pjsub (from login node):
#   pjsub -g hp250467 -L "rscgrp=small,node=1,elapse=00:05:00" run_pmu_example.sh

# Pin to single core for accurate measurements
export OMP_NUM_THREADS=1
export FLIB_FASTOMP=FALSE

echo "=== PMU Counter Library Test ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CPU:  $(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2)"
echo ""

./pmu_example
