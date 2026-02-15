#!/bin/bash
echo "=== INT8 vs INT16 L2 Streaming Comparison ==="
echo "Host: $(hostname)"
echo ""

echo "=========================================="
echo "INT8 SDOT L2 Streaming"
echo "=========================================="
cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-gemm
./bench_int8_l2stream 512 10000

echo ""
echo "=========================================="
echo "INT16 SDOT L2 Streaming"
echo "=========================================="
cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int16-gemm
./bench_int16_l2stream 512 10000

echo ""
echo "=== Done ==="
