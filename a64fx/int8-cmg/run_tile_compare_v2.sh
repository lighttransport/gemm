#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00"
#PJM -g hp250467
#PJM -j
#PJM -S

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-cmg

export OMP_NUM_THREADS=12

echo "=============================================="
echo "Tile Configuration Comparison v2 (Per-Core)"
echo "=============================================="
echo ""
echo "Testing: 6x4, 7x3, 8x3 with per-core memory allocation"
echo ""

# Compile all kernels
fcc -c -Nclang -mcpu=a64fx+sve micro_kernel_6x4_ooo_v5.S -o micro_kernel_6x4_ooo_v5.o
fcc -c -Nclang -mcpu=a64fx+sve micro_kernel_7x3_ooo.S -o micro_kernel_7x3_ooo.o
fcc -c -Nclang -mcpu=a64fx+sve micro_kernel_8x3_v2.S -o micro_kernel_8x3_v2.o

cat > bench_tile_compare_v2.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

#define NUM_CORES 12

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

static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Per-core benchmark with optimal memory layout
double bench_kernel_percore(const char* name, kernel_fn kernel, int MR, int NR,
                            int64_t M, int64_t N_per_core, int64_t K, int niters) {
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

    // Per-core allocations
    int8_t* Ap[NUM_CORES];
    int8_t* Bp[NUM_CORES];
    int32_t* Cp[NUM_CORES];

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Ap[tid] = (int8_t*)fast_alloc(Ap_size);
        Bp[tid] = (int8_t*)fast_alloc(Bp_size);
        Cp[tid] = (int32_t*)fast_alloc(C_size);

        // Initialize with thread-specific pattern
        unsigned int seed = 42 + tid;
        for (size_t i = 0; i < Ap_size; i++) Ap[tid][i] = (int8_t)((rand_r(&seed) % 64) - 32);
        for (size_t i = 0; i < Bp_size; i++) Bp[tid][i] = (int8_t)((rand_r(&seed) % 64) - 32);
    }

    // Warmup
    for (int w = 0; w < 5; w++) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for (int64_t nt = 0; nt < N_tiles; nt++) {
                const int8_t* Bpp = Bp[tid] + nt * Bp_tile_size;
                for (int64_t mt = 0; mt < M_tiles; mt++) {
                    const int8_t* App = Ap[tid] + mt * Ap_tile_size;
                    int32_t* Cptr = Cp[tid] + (mt * N_tiles + nt) * MR * num_vecs * 16;
                    kernel(App, Bpp, Cptr, K, 0, C_stride);
                }
            }
        }
    }

    double t0 = get_time();
    for (int iter = 0; iter < niters; iter++) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for (int64_t nt = 0; nt < N_tiles; nt++) {
                const int8_t* Bpp = Bp[tid] + nt * Bp_tile_size;
                for (int64_t mt = 0; mt < M_tiles; mt++) {
                    const int8_t* App = Ap[tid] + mt * Ap_tile_size;
                    int32_t* Cptr = Cp[tid] + (mt * N_tiles + nt) * MR * num_vecs * 16;
                    kernel(App, Bpp, Cptr, K, 0, C_stride);
                }
            }
        }
    }
    double elapsed = get_time() - t0;

    // SDOTs per tile = (K/4) * MR * num_vecs
    double sdots_per_tile = (double)K / 4.0 * MR * num_vecs;
    double tiles_per_core = M_tiles * N_tiles;
    double total_sdots = sdots_per_tile * tiles_per_core * NUM_CORES * niters;
    double sdot_per_cycle = total_sdots / (elapsed * 2e9);  // Total for CMG
    double efficiency = sdot_per_cycle / 24.0 * 100.0;      // Peak = 24 for 12 cores
    double gops = total_sdots * 64.0 / 1e9 / elapsed;       // 64 ops per SDOT

    printf("%-12s M=%-4ld N/c=%-4ld K=%-4ld: %.2f SDOT/cyc (%.1f%%), %.0f GOPS\n",
           name, M, N_per_core, K, sdot_per_cycle, efficiency, gops);

    // Cleanup
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        free(Ap[tid]); free(Bp[tid]); free(Cp[tid]);
    }

    return efficiency;
}

int main() {
    omp_set_num_threads(NUM_CORES);

    printf("=== Tile Configuration Comparison (Per-Core Memory) ===\n");
    printf("Peak = 24 SDOT/cycle for 12-core CMG\n\n");

    // Best config from previous analysis: M=672, N=640/core, K=512
    int64_t M = 672;
    int64_t K = 512;

    // For 6x4: NR=64, N_per_core should be divisible by 64
    // For 7x3/8x3: NR=48, N_per_core should be divisible by 48
    // LCM(48, 64) = 192, so use N that's divisible by 192

    printf("=== Test 1: Original optimal config (B=320KB/core) ===\n");
    // 6x4: N=640 -> B = 640 * K/4 * 4 = 640 * 512 / 4 * 4 = 327680 bytes = 320KB
    // x3:  N=624 -> B = 624 * K/4 * 4 = 624 * 512 / 4 * 4 = 319488 bytes = 312KB
    int64_t N_6x4_1 = 640;   // 10 tiles of 64
    int64_t N_x3_1 = 624;    // 13 tiles of 48

    bench_kernel_percore("6x4_v5", micro_kernel_6x4_ooo_v5, 6, 64, M, N_6x4_1, K, 100);
    bench_kernel_percore("7x3_ooo", micro_kernel_7x3_ooo, 7, 48, M, N_x3_1, K, 100);
    bench_kernel_percore("8x3_v2", micro_kernel_8x3_v2, 8, 48, M, N_x3_1, K, 100);

    printf("\n=== Test 2: Same N (fair comparison) ===\n");
    // Use N=576 which is divisible by both 64 and 48
    // 6x4: 576/64 = 9 tiles, 7x3/8x3: 576/48 = 12 tiles
    int64_t N_fair = 576;

    bench_kernel_percore("6x4_v5", micro_kernel_6x4_ooo_v5, 6, 64, M, N_fair, K, 100);
    bench_kernel_percore("7x3_ooo", micro_kernel_7x3_ooo, 7, 48, M, N_fair, K, 100);
    bench_kernel_percore("8x3_v2", micro_kernel_8x3_v2, 8, 48, M, N_fair, K, 100);

    printf("\n=== Test 3: Larger N (B=384KB/core) ===\n");
    // 6x4: N=768 -> B = 384KB
    // x3:  N=768 -> B = 768 * 512 / 4 * 4 = 393216 = 384KB
    int64_t N_6x4_3 = 768;
    int64_t N_x3_3 = 768;

    bench_kernel_percore("6x4_v5", micro_kernel_6x4_ooo_v5, 6, 64, M, N_6x4_3, K, 100);
    bench_kernel_percore("7x3_ooo", micro_kernel_7x3_ooo, 7, 48, M, N_x3_3, K, 100);
    bench_kernel_percore("8x3_v2", micro_kernel_8x3_v2, 8, 48, M, N_x3_3, K, 100);

    printf("\n=== Test 4: K sweep at optimal N ===\n");
    int64_t K_vals[] = {256, 384, 512, 768};
    for (int i = 0; i < 4; i++) {
        printf("--- K=%ld ---\n", K_vals[i]);
        bench_kernel_percore("6x4_v5", micro_kernel_6x4_ooo_v5, 6, 64, M, N_6x4_1, K_vals[i], 100);
        bench_kernel_percore("7x3_ooo", micro_kernel_7x3_ooo, 7, 48, M, N_x3_1, K_vals[i], 100);
        bench_kernel_percore("8x3_v2", micro_kernel_8x3_v2, 8, 48, M, N_x3_1, K_vals[i], 100);
        printf("\n");
    }

    printf("=== Summary ===\n");
    printf("Config   MR×NR  Accum  SDOTs/K  Notes\n");
    printf("------   -----  -----  -------  -----\n");
    printf("6x4_v5   6×64   24     24       2x K-unroll, best for large N\n");
    printf("7x3_ooo  7×48   21     21       More M rows, fewer B loads\n");
    printf("8x3_v2   8×48   24     24       2-phase (8 A reloads per K)\n");

    return 0;
}
CEOF

fcc -O3 -Nclang -mcpu=a64fx+sve -fopenmp bench_tile_compare_v2.c \
    micro_kernel_6x4_ooo_v5.o micro_kernel_7x3_ooo.o micro_kernel_8x3_v2.o \
    -o bench_tile_compare_v2

echo ""
./bench_tile_compare_v2
