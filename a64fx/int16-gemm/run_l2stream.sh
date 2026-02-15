#!/bin/bash
cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int16-gemm
echo "=== INT16 SDOT L2 Streaming Benchmark ==="
echo "Host: $(hostname)"
echo ""
./bench_int16_l2stream 512 10000
echo "=== Done ==="
