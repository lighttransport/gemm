#!/bin/bash
cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/fp8-conv
export OMP_NUM_THREADS=1
./bench_fp8_gemm
