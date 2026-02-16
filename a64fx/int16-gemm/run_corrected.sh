#!/bin/bash
cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int16-gemm
echo "=== INT16 SDOT Benchmark (Corrected Ops Counting) ==="
echo "Host: $(hostname)"
echo ""
echo "=== Main Benchmark (ASM version) ==="
./bench_int16_sdot_asm 1000000
echo ""
echo "=== Peak Test ==="
./bench_peak 1000000
echo "=== Done ==="
