#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00"
#PJM -g hp250467
#PJM -j
#PJM -S

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-cmg

export OMP_NUM_THREADS=12
export FLIB_FASTOMP=TRUE

echo "=============================================="
echo "Tile Configuration Comparison v3 (Fixed)"
echo "=============================================="
echo ""

# Compile all kernels
fcc -c -Nclang -mcpu=a64fx+sve micro_kernel_6x4_ooo_v5.S -o micro_kernel_6x4_ooo_v5.o
fcc -c -Nclang -mcpu=a64fx+sve micro_kernel_7x3_ooo.S -o micro_kernel_7x3_ooo.o
fcc -c -Nclang -mcpu=a64fx+sve micro_kernel_8x3_v2.S -o micro_kernel_8x3_v2.o

cat > bench_tile_compare_v3.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#define NUM_CORES 12
#define CPU_FREQ_GHZ 2.0

// Kernel declarations
extern void micro_kernel_6x4_ooo_v5(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t, int64_t);
extern void micro_kernel_7x3_ooo(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t, int64_t);
extern void micro_kernel_8x3_v2(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t, int64_t);

typedef void (*kernel_fn)(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t, int64_t);

static void* fast_alloc(size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, 256, size);
    if (ptr) memset(ptr, 0, size);
    return ptr;
}

double bench_kernel(const char* name, kernel_fn kernel, int MR, int NR,
                    int64_t M, int64_t N_per_core, int64_t K, int niters) {
    omp_set_num_threads(NUM_CORES);

    int num_vecs = NR / 16;
    int64_t M_tiles = M / MR;
    int64_t N_tiles = N_per_core / NR;
    int64_t K4 = K / 4;
    int64_t C_stride = num_vecs * 16 * sizeof(int32_t);
    int64_t Bp_tile_size = K4 * num_vecs * 64;
    int64_t Ap_tile_size = K4 * MR * 4;

    size_t Ap_size = M_tiles * Ap_tile_size;
    size_t Bp_size = N_tiles * Bp_tile_size;
    size_t C_size = M_tiles * N_tiles * MR * num_vecs * 16 * sizeof(int32_t);

    // Per-core allocations - OUTSIDE parallel region
    int8_t* Ap_per_core[NUM_CORES];
    int8_t* Bp_per_core[NUM_CORES];
    int32_t* C_per_core[NUM_CORES];

    for (int i = 0; i < NUM_CORES; i++) {
        Ap_per_core[i] = (int8_t*)fast_alloc(Ap_size);
        Bp_per_core[i] = (int8_t*)fast_alloc(Bp_size);
        C_per_core[i] = (int32_t*)fast_alloc(C_size);
        memset(Ap_per_core[i], 1, Ap_size);
        memset(Bp_per_core[i], 1, Bp_size);
    }

    // Warmup
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int64_t mt = 0; mt < M_tiles; mt++) {
            for (int64_t nt = 0; nt < N_tiles; nt++) {
                kernel(
                    Ap_per_core[tid] + mt * Ap_tile_size,
                    Bp_per_core[tid] + nt * Bp_tile_size,
                    C_per_core[tid] + (mt * N_tiles + nt) * MR * num_vecs * 16,
                    K, 0, C_stride);
            }
        }
    }

    double t0 = omp_get_wtime();
    for (int iter = 0; iter < niters; iter++) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for (int64_t mt = 0; mt < M_tiles; mt++) {
                for (int64_t nt = 0; nt < N_tiles; nt++) {
                    kernel(
                        Ap_per_core[tid] + mt * Ap_tile_size,
                        Bp_per_core[tid] + nt * Bp_tile_size,
                        C_per_core[tid] + (mt * N_tiles + nt) * MR * num_vecs * 16,
                        K, 0, C_stride);
                }
            }
        }
    }
    double t = omp_get_wtime() - t0;

    for (int i = 0; i < NUM_CORES; i++) {
        free(Ap_per_core[i]);
        free(Bp_per_core[i]);
        free(C_per_core[i]);
    }

    // Calculate SDOT/cycle using same formula as successful benchmark
    int64_t total_sdot = (int64_t)M * N_per_core * K / 64 * NUM_CORES * niters;
    double cycles = t * CPU_FREQ_GHZ * 1e9;
    double sdot_per_cycle = (double)total_sdot / cycles;
    double efficiency = sdot_per_cycle / 24.0 * 100.0;

    printf("%-10s M=%-4ld N/c=%-4ld K=%-3ld: %.2f SDOT/cyc (%.1f%%)\n",
           name, M, N_per_core, K, sdot_per_cycle, efficiency);

    return sdot_per_cycle;
}

int main() {
    omp_set_num_threads(NUM_CORES);
    double peak = 24.0;

    printf("=== Tile Configuration Comparison ===\n");
    printf("Peak = %.1f SDOT/cycle for 12-core CMG\n\n", peak);

    // Best config from previous analysis: M=672, N=640/core, K=512
    int64_t M = 672;
    int64_t K = 512;

    printf("=== Test 1: Optimal config (M=672, K=512) ===\n");
    // For 6x4: N=640 (10 tiles of 64)
    // For 7x3: Use M=672 works (672/7=96), N=624 (13 tiles of 48)
    // For 8x3: M=672 works (672/8=84), N=624
    int64_t N_6x4 = 640;
    int64_t N_x3 = 624;

    bench_kernel("6x4_v5", micro_kernel_6x4_ooo_v5, 6, 64, M, N_6x4, K, 50);
    bench_kernel("7x3_ooo", micro_kernel_7x3_ooo, 7, 48, M, N_x3, K, 50);
    bench_kernel("8x3_v2", micro_kernel_8x3_v2, 8, 48, M, N_x3, K, 50);

    printf("\n=== Test 2: Fair comparison (same N=576) ===\n");
    // N=576 is divisible by both 64 and 48
    int64_t N_fair = 576;
    bench_kernel("6x4_v5", micro_kernel_6x4_ooo_v5, 6, 64, M, N_fair, K, 50);
    bench_kernel("7x3_ooo", micro_kernel_7x3_ooo, 7, 48, M, N_fair, K, 50);
    bench_kernel("8x3_v2", micro_kernel_8x3_v2, 8, 48, M, N_fair, K, 50);

    printf("\n=== Test 3: Consistency check for 6x4 (5 runs) ===\n");
    for (int i = 0; i < 5; i++) {
        double sdot = bench_kernel("6x4_v5", micro_kernel_6x4_ooo_v5, 6, 64, M, N_6x4, K, 50);
    }

    printf("\n=== Test 4: Consistency check for 8x3 (5 runs) ===\n");
    for (int i = 0; i < 5; i++) {
        double sdot = bench_kernel("8x3_v2", micro_kernel_8x3_v2, 8, 48, M, N_x3, K, 50);
    }

    printf("\n=== Summary ===\n");
    printf("Config   MR×NR  SDOTs/K  Register Allocation\n");
    printf("------   -----  -------  -------------------\n");
    printf("6x4_v5   6×64   24       z0-23(acc) z24-29(A) z30-31(B)\n");
    printf("7x3_ooo  7×48   21       z0-20(acc) z21-27(A) z28-30(B) z31(spare)\n");
    printf("8x3_v2   8×48   24       z0-23(acc) z24-27(A) z28-30(B) [2-phase]\n");

    return 0;
}
CEOF

fcc -O3 -Nclang -mcpu=a64fx+sve -fopenmp bench_tile_compare_v3.c \
    micro_kernel_6x4_ooo_v5.o micro_kernel_7x3_ooo.o micro_kernel_8x3_v2.o \
    -o bench_tile_compare_v3

./bench_tile_compare_v3
