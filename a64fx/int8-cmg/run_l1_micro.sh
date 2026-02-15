#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:15:00"
#PJM -g hp250467
#PJM -j
#PJM -S

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-cmg

export OMP_NUM_THREADS=12
export FLIB_FASTOMP=TRUE

echo "=============================================="
echo "L1 Micro-Blocking for 90%+ Efficiency"
echo "=============================================="
echo ""
echo "Strategy: Keep working set in L1 (64KB per core)"
echo "Process many small tiles rather than few large tiles"
echo ""

fcc -c -Nclang -mcpu=a64fx+sve micro_kernel_6x4_ooo_v5.S -o micro_kernel_6x4_ooo_v5.o

cat > bench_l1_micro.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#define NUM_CORES 12
#define MR 6
#define NR 64
#define CPU_FREQ_GHZ 2.0

extern void micro_kernel_6x4_ooo_v5(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t, int64_t);

static void* fast_alloc(size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, 256, size);
    if (ptr) memset(ptr, 0, size);
    return ptr;
}

// L1-sized micro-blocking: Keep A and B tiles in L1
// L1 = 64KB, so target ~40KB working set
// A tile = K/4 × 24 bytes, B tile = K/4 × 256 bytes
// For K=512: A=3KB, B=32KB, Total=35KB (fits!)
double bench_l1_micro(int64_t total_M, int64_t total_N, int64_t K,
                      int64_t micro_M, int64_t micro_N, int niters) {
    omp_set_num_threads(NUM_CORES);

    int64_t K4 = K / 4;
    int64_t C_stride = 4 * 16 * sizeof(int32_t);
    int64_t Bp_tile_size = K4 * 256;
    int64_t Ap_tile_size = K4 * MR * 4;

    int64_t micro_M_tiles = micro_M / MR;
    int64_t micro_N_tiles = micro_N / NR;
    int64_t num_M_blocks = total_M / micro_M;
    int64_t num_N_blocks = total_N / micro_N;

    // Per-core full GEMM allocation
    int64_t M_tiles = total_M / MR;
    int64_t N_tiles = total_N / NR;
    size_t Ap_size = M_tiles * Ap_tile_size;
    size_t Bp_size = N_tiles * Bp_tile_size;
    size_t C_size = M_tiles * N_tiles * MR * NR * sizeof(int32_t);

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

    // Warmup with micro-blocking pattern
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int64_t mb = 0; mb < num_M_blocks; mb++) {
            for (int64_t nb = 0; nb < num_N_blocks; nb++) {
                // Process micro-block
                for (int64_t mt = 0; mt < micro_M_tiles; mt++) {
                    int64_t global_mt = mb * micro_M_tiles + mt;
                    for (int64_t nt = 0; nt < micro_N_tiles; nt++) {
                        int64_t global_nt = nb * micro_N_tiles + nt;
                        micro_kernel_6x4_ooo_v5(
                            Ap_per_core[tid] + global_mt * Ap_tile_size,
                            Bp_per_core[tid] + global_nt * Bp_tile_size,
                            C_per_core[tid] + (global_mt * N_tiles + global_nt) * MR * NR,
                            K, 0, C_stride);
                    }
                }
            }
        }
    }

    double t0 = omp_get_wtime();
    for (int iter = 0; iter < niters; iter++) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            // Micro-blocking: process small blocks that fit in L1
            for (int64_t mb = 0; mb < num_M_blocks; mb++) {
                for (int64_t nb = 0; nb < num_N_blocks; nb++) {
                    // Each micro-block: micro_M_tiles × micro_N_tiles kernels
                    // N-inner for B reuse within micro-block
                    for (int64_t nt = 0; nt < micro_N_tiles; nt++) {
                        int64_t global_nt = nb * micro_N_tiles + nt;
                        for (int64_t mt = 0; mt < micro_M_tiles; mt++) {
                            int64_t global_mt = mb * micro_M_tiles + mt;
                            micro_kernel_6x4_ooo_v5(
                                Ap_per_core[tid] + global_mt * Ap_tile_size,
                                Bp_per_core[tid] + global_nt * Bp_tile_size,
                                C_per_core[tid] + (global_mt * N_tiles + global_nt) * MR * NR,
                                K, 0, C_stride);
                        }
                    }
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

    int64_t total_sdot = (int64_t)total_M * total_N * K / 64 * NUM_CORES * niters;
    double cycles = t * CPU_FREQ_GHZ * 1e9;
    return (double)total_sdot / cycles;
}

// Standard flat loop for comparison
double bench_flat(int64_t M, int64_t N, int64_t K, int niters) {
    omp_set_num_threads(NUM_CORES);

    int64_t M_tiles = M / MR;
    int64_t N_tiles = N / NR;
    int64_t K4 = K / 4;
    int64_t C_stride = 4 * 16 * sizeof(int32_t);
    int64_t Bp_tile_size = K4 * 256;
    int64_t Ap_tile_size = K4 * MR * 4;

    size_t Ap_size = M_tiles * Ap_tile_size;
    size_t Bp_size = N_tiles * Bp_tile_size;
    size_t C_size = M_tiles * N_tiles * MR * NR * sizeof(int32_t);

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
                micro_kernel_6x4_ooo_v5(
                    Ap_per_core[tid] + mt * Ap_tile_size,
                    Bp_per_core[tid] + nt * Bp_tile_size,
                    C_per_core[tid] + (mt * N_tiles + nt) * MR * NR,
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
                    micro_kernel_6x4_ooo_v5(
                        Ap_per_core[tid] + mt * Ap_tile_size,
                        Bp_per_core[tid] + nt * Bp_tile_size,
                        C_per_core[tid] + (mt * N_tiles + nt) * MR * NR,
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

    int64_t total_sdot = (int64_t)M * N * K / 64 * NUM_CORES * niters;
    double cycles = t * CPU_FREQ_GHZ * 1e9;
    return (double)total_sdot / cycles;
}

int main() {
    double peak = 24.0;

    printf("=== Working Set Analysis ===\n");
    printf("K=512: A_tile=3KB, B_tile=32KB, Total=35KB (fits in 64KB L1)\n");
    printf("K=256: A_tile=1.5KB, B_tile=16KB, Total=17.5KB\n\n");

    printf("=== Test 1: Different micro-block sizes (K=512) ===\n");
    printf("Total: M=672, N=640\n");
    printf("uM    uN    uA(KB)  uB(KB)  Total   Flat     Micro\n");
    printf("----  ----  ------  ------  ------  ------   ------\n");

    // Different micro-block sizes
    struct { int64_t uM, uN; } blocks[] = {
        {6, 64},      // Single tile
        {12, 128},    // 2x2 tiles
        {24, 192},    // 4x3 tiles
        {48, 256},    // 8x4 tiles
        {96, 320},    // 16x5 tiles
        {168, 640},   // Full N, partial M
    };

    for (int i = 0; i < 6; i++) {
        int64_t uM = blocks[i].uM;
        int64_t uN = blocks[i].uN;
        size_t uA = (uM / MR) * 128 * 24;    // K/4=128, 24 bytes per K-group per M-tile
        size_t uB = (uN / NR) * 128 * 256;   // K/4=128, 256 bytes per K-group per N-tile

        double flat = bench_flat(672, 640, 512, 30);
        double micro = bench_l1_micro(672, 640, 512, uM, uN, 30);

        printf("%4ld  %4ld  %6.1f  %6.1f  %6.1f  %5.1f%%   %5.1f%%\n",
               uM, uN, uA/1024.0, uB/1024.0, (uA+uB)/1024.0,
               flat/peak*100, micro/peak*100);
    }

    printf("\n=== Test 2: Optimal K for micro-blocking ===\n");
    printf("Total: M=672, N=640, uM=48, uN=256\n");
    printf("K      A(KB)  B(KB)  Total   Micro Eff\n");
    printf("----   -----  -----  ------  ---------\n");

    int64_t K_vals[] = {128, 256, 384, 512, 768};
    for (int i = 0; i < 5; i++) {
        int64_t K = K_vals[i];
        int64_t K4 = K / 4;
        size_t uA = 8 * K4 * 24;   // 8 M-tiles
        size_t uB = 4 * K4 * 256;  // 4 N-tiles

        double micro = bench_l1_micro(672, 640, K, 48, 256, 30);

        printf("%4ld   %5.1f  %5.1f  %6.1f  %5.1f%%\n",
               K, uA/1024.0, uB/1024.0, (uA+uB)/1024.0, micro/peak*100);
    }

    printf("\n=== Test 3: Best configuration search ===\n");
    printf("M      N      K      uM    uN    Flat    Micro\n");
    printf("----   ----   ----   ----  ----  ------  ------\n");

    struct { int64_t M, N, K, uM, uN; } configs[] = {
        {672, 640, 512, 48, 256},
        {672, 640, 512, 24, 128},
        {672, 640, 256, 48, 256},
        {672, 640, 256, 24, 192},
        {480, 512, 512, 48, 256},
        {480, 512, 256, 48, 256},
        {384, 640, 512, 48, 320},
        {384, 640, 256, 48, 320},
    };

    double best_eff = 0;
    int best_idx = 0;

    for (int i = 0; i < 8; i++) {
        double flat = bench_flat(configs[i].M, configs[i].N, configs[i].K, 30);
        double micro = bench_l1_micro(configs[i].M, configs[i].N, configs[i].K,
                                      configs[i].uM, configs[i].uN, 30);

        printf("%4ld   %4ld   %4ld   %4ld  %4ld  %5.1f%%   %5.1f%%\n",
               configs[i].M, configs[i].N, configs[i].K,
               configs[i].uM, configs[i].uN,
               flat/peak*100, micro/peak*100);

        if (micro > best_eff) {
            best_eff = micro;
            best_idx = i;
        }
    }

    printf("\n=== Best Micro-Block Verification ===\n");
    printf("M=%ld, N=%ld, K=%ld, uM=%ld, uN=%ld\n",
           configs[best_idx].M, configs[best_idx].N, configs[best_idx].K,
           configs[best_idx].uM, configs[best_idx].uN);

    for (int i = 0; i < 10; i++) {
        double eff = bench_l1_micro(configs[best_idx].M, configs[best_idx].N,
                                     configs[best_idx].K,
                                     configs[best_idx].uM, configs[best_idx].uN, 50);
        printf("Run %2d: %.3f SDOT/cyc (%.1f%%)\n", i+1, eff, eff/peak*100);
    }

    return 0;
}
CEOF

fcc -O3 -Nclang -mcpu=a64fx+sve -fopenmp bench_l1_micro.c \
    micro_kernel_6x4_ooo_v5.o -o bench_l1_micro

./bench_l1_micro
