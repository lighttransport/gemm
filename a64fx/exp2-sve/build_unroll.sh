#!/bin/bash
# Build all unroll variants

CC="fcc -Nclang"
CFLAGS="-O3 -march=armv8.2-a+sve -fopenmp"

set -x

# Assemble all kernels
$CC $CFLAGS -c exp2_fexpa_opt.S -o exp2_fexpa_opt.o
$CC $CFLAGS -c exp2_fexpa_u8.S -o exp2_fexpa_u8.o
$CC $CFLAGS -c exp2_fexpa_u16.S -o exp2_fexpa_u16.o
$CC $CFLAGS -c exp2_fexpa_pipe.S -o exp2_fexpa_pipe.o

# Build benchmark
$CC $CFLAGS bench_unroll.c \
    exp2_fexpa_opt.o \
    exp2_fexpa_u8.o \
    exp2_fexpa_u16.o \
    exp2_fexpa_pipe.o \
    -lm -o bench_unroll

echo "Build complete: bench_unroll"
