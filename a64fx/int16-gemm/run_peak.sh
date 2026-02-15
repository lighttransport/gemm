#!/bin/bash
cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int16-gemm
echo "=== INT16 SDOT Peak Test ==="
echo "Host: $(hostname)"
./bench_peak 1000000
echo "=== Done ==="
