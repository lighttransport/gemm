#!/bin/bash

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/fp8-conv

# Build
make bench_opt_convert

# Run
./bench_opt_convert
