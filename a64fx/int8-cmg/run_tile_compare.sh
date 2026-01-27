#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00"
#PJM -g hp250467
#PJM -j
#PJM -S

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-cmg

export OMP_NUM_THREADS=12

echo "=============================================="
echo "Microkernel Tile Configuration Comparison"
echo "=============================================="
echo ""
echo "Configurations:"
echo "  6x4: 24 SDOTs/K-group, NR=64, minimal spare regs"
echo "  7x3: 21 SDOTs/K-group, NR=48, 1 spare reg"
echo "  8x3: 24 SDOTs/K-group, NR=48, 2-phase approach"
echo ""

# Compile all kernels
fcc -c -Nclang -mcpu=a64fx+sve micro_kernel_6x4_ooo_v5.S -o micro_kernel_6x4_ooo_v5.o
fcc -c -Nclang -mcpu=a64fx+sve micro_kernel_7x3_ooo.S -o micro_kernel_7x3_ooo.o
fcc -c -Nclang -mcpu=a64fx+sve micro_kernel_8x3_v2.S -o micro_kernel_8x3_v2.o

cat > bench_tile_compare.c << 'CEOF'
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

// Reference implementation for verification
void reference_kernel(const int8_t* Ap, const int8_t* Bp, int32_t* C,
                      int MR, int NR, int64_t K) {
    int num_vecs = NR / 16;  // Each vector is 16 int32s
    memset(C, 0, MR * num_vecs * 16 * sizeof(int32_t));
    int64_t K_groups = K / 4;

    for (int64_t kg = 0; kg < K_groups; kg++) {
        const int8_t* Ak = Ap + kg * MR * 4;
        const int8_t* Bk = Bp + kg * num_vecs * 64;  // 64 bytes per vector

        for (int row = 0; row < MR; row++) {
            const int8_t* a_row = Ak + row * 4;
            for (int col_grp = 0; col_grp < num_vecs; col_grp++) {
                const int8_t* b_col = Bk + col_grp * 64;
                int32_t* c_out = C + row * num_vecs * 16 + col_grp * 16;
                for (int i = 0; i < 16; i++) {
                    int32_t sum = 0;
                    for (int k = 0; k < 4; k++) {
                        sum += (int32_t)a_row[k] * (int32_t)b_col[i * 4 + k];
                    }
                    c_out[i] += sum;
                }
            }
        }
    }
}

int verify_kernel(const char* name, kernel_fn kernel, int MR, int NR, int64_t K) {
    int num_vecs = NR / 16;
    int64_t K4 = K / 4;
    int64_t Ap_size = K4 * MR * 4;
    int64_t Bp_size = K4 * num_vecs * 64;
    int64_t C_size = MR * num_vecs * 16;
    int64_t C_stride = num_vecs * 16 * sizeof(int32_t);

    int8_t* Ap = (int8_t*)fast_alloc(Ap_size);
    int8_t* Bp = (int8_t*)fast_alloc(Bp_size);
    int32_t* C_ref = (int32_t*)fast_alloc(C_size * sizeof(int32_t));
    int32_t* C_test = (int32_t*)fast_alloc(C_size * sizeof(int32_t));

    srand(42);
    for (size_t i = 0; i < Ap_size; i++) Ap[i] = (int8_t)((rand() % 16) - 8);
    for (size_t i = 0; i < Bp_size; i++) Bp[i] = (int8_t)((rand() % 16) - 8);

    reference_kernel(Ap, Bp, C_ref, MR, NR, K);
    memset(C_test, 0, C_size * sizeof(int32_t));
    kernel(Ap, Bp, C_test, K, 0, C_stride);

    int errors = 0;
    for (int i = 0; i < C_size; i++) {
        if (C_ref[i] != C_test[i]) {
            if (errors < 3) {
                printf("  %s K=%ld: mismatch at %d: ref=%d, test=%d\n",
                       name, K, i, C_ref[i], C_test[i]);
            }
            errors++;
        }
    }

    free(Ap); free(Bp); free(C_ref); free(C_test);
    return errors;
}

void bench_kernel(const char* name, kernel_fn kernel, int MR, int NR,
                  int64_t M, int64_t N, int64_t K, int niters) {
    int num_vecs = NR / 16;
    int64_t M_tiles = M / MR;
    int64_t N_tiles = N / NR;
    int64_t K4 = K / 4;
    int64_t C_stride = num_vecs * 16 * sizeof(int32_t);
    int64_t Bp_tile_size = K4 * num_vecs * 64;
    int64_t Ap_tile_size = K4 * MR * 4;

    size_t Ap_size = M_tiles * Ap_tile_size;
    size_t Bp_size = N_tiles * Bp_tile_size;
    size_t C_size = M_tiles * N_tiles * MR * num_vecs * 16 * sizeof(int32_t);

    int8_t* Ap = (int8_t*)fast_alloc(Ap_size);
    int8_t* Bp = (int8_t*)fast_alloc(Bp_size);
    int32_t* C = (int32_t*)fast_alloc(C_size);

    srand(42);
    for (size_t i = 0; i < Ap_size; i++) Ap[i] = (int8_t)((rand() % 64) - 32);
    for (size_t i = 0; i < Bp_size; i++) Bp[i] = (int8_t)((rand() % 64) - 32);

    // Warmup
    for (int w = 0; w < 5; w++) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int64_t nt_per = (N_tiles + NUM_CORES - 1) / NUM_CORES;
            int64_t nt_start = tid * nt_per;
            int64_t nt_end = (nt_start + nt_per < N_tiles) ? nt_start + nt_per : N_tiles;
            for (int64_t nt = nt_start; nt < nt_end; nt++) {
                const int8_t* Bpp = Bp + nt * Bp_tile_size;
                for (int64_t mt = 0; mt < M_tiles; mt++) {
                    const int8_t* App = Ap + mt * Ap_tile_size;
                    int32_t* Cp = C + (mt * N_tiles + nt) * MR * num_vecs * 16;
                    kernel(App, Bpp, Cp, K, 0, C_stride);
                }
            }
        }
    }

    double t0 = get_time();
    for (int iter = 0; iter < niters; iter++) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int64_t nt_per = (N_tiles + NUM_CORES - 1) / NUM_CORES;
            int64_t nt_start = tid * nt_per;
            int64_t nt_end = (nt_start + nt_per < N_tiles) ? nt_start + nt_per : N_tiles;
            for (int64_t nt = nt_start; nt < nt_end; nt++) {
                const int8_t* Bpp = Bp + nt * Bp_tile_size;
                for (int64_t mt = 0; mt < M_tiles; mt++) {
                    const int8_t* App = Ap + mt * Ap_tile_size;
                    int32_t* Cp = C + (mt * N_tiles + nt) * MR * num_vecs * 16;
                    kernel(App, Bpp, Cp, K, 0, C_stride);
                }
            }
        }
    }
    double elapsed = get_time() - t0;

    // SDOTs per tile = (K/4) * MR * num_vecs
    double sdots_per_tile = (double)K / 4.0 * MR * num_vecs;
    double total_tiles = M_tiles * N_tiles;
    double total_sdots = sdots_per_tile * total_tiles * niters;
    double sdot_per_cycle = total_sdots / (elapsed * 2e9) / NUM_CORES;
    double efficiency = sdot_per_cycle / 2.0 * 100.0;
    double gops = total_sdots * 64.0 / 1e9 / elapsed;  // 64 ops per SDOT

    printf("%-12s K=%-4ld: %.3f SDOT/cyc (%.1f%%), %.0f GOPS\n",
           name, K, sdot_per_cycle, efficiency, gops);

    free(Ap); free(Bp); free(C);
}

int main() {
    omp_set_num_threads(NUM_CORES);

    printf("=== Correctness Check ===\n");
    int64_t K_test[] = {16, 64, 128, 256, 512};
    for (int i = 0; i < 5; i++) {
        int64_t K = K_test[i];
        int err_6x4 = verify_kernel("6x4_v5", micro_kernel_6x4_ooo_v5, 6, 64, K);
        int err_7x3 = verify_kernel("7x3_ooo", micro_kernel_7x3_ooo, 7, 48, K);
        int err_8x3 = verify_kernel("8x3_v2", micro_kernel_8x3_v2, 8, 48, K);
        printf("K=%3ld: 6x4=%s, 7x3=%s, 8x3=%s\n", K,
               err_6x4 == 0 ? "PASS" : "FAIL",
               err_7x3 == 0 ? "PASS" : "FAIL",
               err_8x3 == 0 ? "PASS" : "FAIL");
    }

    printf("\n=== Performance Comparison ===\n");
    printf("Cores=%d\n\n", NUM_CORES);

    // Test with optimal config found earlier: M=672, N=640/core, K=512
    // For 6x4: M_tiles = 672/6 = 112, N_tiles = 7680/64 = 120
    // For 7x3: M_tiles = 672/7 = 96, N_tiles = 7680/48 = 160 (use M=672 -> not divisible, use M=672-7*10=602? use 700=7*100)
    // For 8x3: M_tiles = 672/8 = 84, N_tiles = 7680/48 = 160

    // Use M divisible by lcm(6,7,8) = 168, M = 672 = 4*168
    int64_t M = 672;
    int64_t N_6x4 = 7680;   // 120 tiles of 64
    int64_t N_x3 = 7680;    // 160 tiles of 48

    printf("=== Optimal Config: M=%ld, N/core=%ld, K=512 ===\n\n", M, N_6x4/NUM_CORES);

    // Compare at K=512 (best from previous analysis)
    int64_t K = 512;
    printf("--- K=%ld (best config) ---\n", K);
    bench_kernel("6x4_v5", micro_kernel_6x4_ooo_v5, 6, 64, M, N_6x4, K, 100);
    bench_kernel("7x3_ooo", micro_kernel_7x3_ooo, 7, 48, M, N_x3, K, 100);
    bench_kernel("8x3_v2", micro_kernel_8x3_v2, 8, 48, M, N_x3, K, 100);
    printf("\n");

    // Also test at K=256 and K=384
    int64_t K_perf[] = {256, 384};
    for (int i = 0; i < 2; i++) {
        int64_t Kp = K_perf[i];
        printf("--- K=%ld ---\n", Kp);
        bench_kernel("6x4_v5", micro_kernel_6x4_ooo_v5, 6, 64, M, N_6x4, Kp, 100);
        bench_kernel("7x3_ooo", micro_kernel_7x3_ooo, 7, 48, M, N_x3, Kp, 100);
        bench_kernel("8x3_v2", micro_kernel_8x3_v2, 8, 48, M, N_x3, Kp, 100);
        printf("\n");
    }

    // Summary table
    printf("=== Summary ===\n");
    printf("Config   MR×NR  SDOTs/K  Reg Alloc        Notes\n");
    printf("------   -----  -------  --------         -----\n");
    printf("6x4_v5   6×64   24       24+6+2=32        Best overall (90.5%%)\n");
    printf("7x3_ooo  7×48   21       21+7+3+1=32      1 spare reg\n");
    printf("8x3_v2   8×48   24       24+4+3=31        2-phase (4 rows each)\n");

    return 0;
}
CEOF

fcc -O3 -Nclang -mcpu=a64fx+sve -fopenmp bench_tile_compare.c \
    micro_kernel_6x4_ooo_v5.o micro_kernel_7x3_ooo.o micro_kernel_8x3_v2.o \
    -o bench_tile_compare

echo ""
./bench_tile_compare
