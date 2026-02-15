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
extern void micro_kernel_6x4_ooo_v6(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t, int64_t);

static void* fast_alloc(size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, 256, size);
    if (ptr) memset(ptr, 0, size);
    return ptr;
}

typedef void (*kernel_fn)(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t, int64_t);

// N-outer loop (baseline)
double bench_n_outer(kernel_fn kernel, const int8_t* Ap, const int8_t* Bp, int32_t* C,
                     int64_t M_tiles, int64_t N_tiles, int64_t K,
                     int64_t Ap_tile_size, int64_t Bp_tile_size, int64_t C_stride, int niters) {
    double t0 = omp_get_wtime();
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
                    int32_t* Cp = C + (mt * N_tiles + nt) * MR * 4 * 16;
                    kernel(App, Bpp, Cp, K, 0, C_stride);
                }
            }
        }
    }
    return omp_get_wtime() - t0;
}

// M-inner loop (A stays in L1)
double bench_m_inner(kernel_fn kernel, const int8_t* Ap, const int8_t* Bp, int32_t* C,
                     int64_t M_tiles, int64_t N_tiles, int64_t K,
                     int64_t Ap_tile_size, int64_t Bp_tile_size, int64_t C_stride, int niters) {
    double t0 = omp_get_wtime();
    for (int iter = 0; iter < niters; iter++) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int64_t nt_per = (N_tiles + NUM_CORES - 1) / NUM_CORES;
            int64_t nt_start = tid * nt_per;
            int64_t nt_end = (nt_start + nt_per < N_tiles) ? nt_start + nt_per : N_tiles;
            
            for (int64_t mt = 0; mt < M_tiles; mt++) {
                const int8_t* App = Ap + mt * Ap_tile_size;
                for (int64_t nt = nt_start; nt < nt_end; nt++) {
                    const int8_t* Bpp = Bp + nt * Bp_tile_size;
                    int32_t* Cp = C + (mt * N_tiles + nt) * MR * 4 * 16;
                    kernel(App, Bpp, Cp, K, 0, C_stride);
                }
            }
        }
    }
    return omp_get_wtime() - t0;
}

int main() {
    omp_set_num_threads(NUM_CORES);

    int64_t M = 84, N = 8192, K = 512;
    int niters = 100;

    int64_t M_tiles = M / MR;
    int64_t N_tiles = N / NR;
    int64_t K4 = K / 4;
    int64_t C_stride = 4 * 16 * sizeof(int32_t);
    int64_t Bp_tile_size = K4 * 256;
    int64_t Ap_tile_size = K4 * MR * 4;

    size_t Ap_size = M_tiles * Ap_tile_size;
    size_t Bp_size = N_tiles * Bp_tile_size;
    size_t C_size = M_tiles * N_tiles * MR * 4 * 16 * sizeof(int32_t);

    int8_t* Ap = (int8_t*)fast_alloc(Ap_size);
    int8_t* Bp = (int8_t*)fast_alloc(Bp_size);
    int32_t* C = (int32_t*)fast_alloc(C_size);

    srand(42);
    for (size_t i = 0; i < Ap_size; i++) Ap[i] = (int8_t)((rand() % 64) - 32);
    for (size_t i = 0; i < Bp_size; i++) Bp[i] = (int8_t)((rand() % 64) - 32);

    printf("M=%ld, N=%ld, K=%ld, iterations=%d\n", M, N, K, niters);
    printf("A size: %.1f KB, B size: %.1f MB\n", Ap_size/1024.0, Bp_size/1e6);
    printf("\n");

    // CORRECT formula: M * N * K / 64 (16 outputs per SDOT, 4 K per SDOT)
    // Each SDOT instruction: 64 int8 products contributing to 16 int32 outputs
    int64_t total_sdot = (int64_t)M * N * K / 64 * niters;
    // Peak = 2 SDOT/cycle per core = 24 SDOT/cycle for 12 cores
    double peak = 2.0 * NUM_CORES;
    
    printf("Total SDOT instructions: %ld (%.1fM)\n", total_sdot, total_sdot/1e6);
    printf("Peak throughput: %.1f SDOT/cycle (12 cores)\n", peak);
    printf("\n");

    // Warmup
    for (int w = 0; w < 3; w++) {
        bench_n_outer(micro_kernel_6x4_ooo_v5, Ap, Bp, C, M_tiles, N_tiles, K,
                      Ap_tile_size, Bp_tile_size, C_stride, 1);
    }

    printf("%-35s  %8s  %10s\n", "Version", "SDOT/cyc", "Efficiency");
    printf("%-35s  %8s  %10s\n", "-----------------------------------", "--------", "----------");

    double t, cycles, sdot_cyc, eff;
    
    // v5 N-outer
    t = bench_n_outer(micro_kernel_6x4_ooo_v5, Ap, Bp, C, M_tiles, N_tiles, K,
                      Ap_tile_size, Bp_tile_size, C_stride, niters);
    cycles = t * CPU_FREQ_GHZ * 1e9;
    sdot_cyc = (double)total_sdot / cycles;
    eff = sdot_cyc / peak * 100;
    printf("v5 (sector B) + N-outer               %6.3f    %7.1f%%\n", sdot_cyc, eff);

    // v5 M-inner
    t = bench_m_inner(micro_kernel_6x4_ooo_v5, Ap, Bp, C, M_tiles, N_tiles, K,
                      Ap_tile_size, Bp_tile_size, C_stride, niters);
    cycles = t * CPU_FREQ_GHZ * 1e9;
    sdot_cyc = (double)total_sdot / cycles;
    eff = sdot_cyc / peak * 100;
    printf("v5 (sector B) + M-inner               %6.3f    %7.1f%%\n", sdot_cyc, eff);

    // v6 N-outer
    t = bench_n_outer(micro_kernel_6x4_ooo_v6, Ap, Bp, C, M_tiles, N_tiles, K,
                      Ap_tile_size, Bp_tile_size, C_stride, niters);
    cycles = t * CPU_FREQ_GHZ * 1e9;
    sdot_cyc = (double)total_sdot / cycles;
    eff = sdot_cyc / peak * 100;
    printf("v6 (sector A) + N-outer               %6.3f    %7.1f%%\n", sdot_cyc, eff);

    // v6 M-inner  
    t = bench_m_inner(micro_kernel_6x4_ooo_v6, Ap, Bp, C, M_tiles, N_tiles, K,
                      Ap_tile_size, Bp_tile_size, C_stride, niters);
    cycles = t * CPU_FREQ_GHZ * 1e9;
    sdot_cyc = (double)total_sdot / cycles;
    eff = sdot_cyc / peak * 100;
    printf("v6 (sector A) + M-inner               %6.3f    %7.1f%%\n", sdot_cyc, eff);

    printf("\n");
    printf("Analysis:\n");
    printf("  N-outer: B stays in L1 per N-tile, A streams\n");
    printf("  M-inner: A stays in L1 per M-tile, B streams\n");
    printf("  Sector hint controls L2 caching behavior\n");

    free(Ap); free(Bp); free(C);
    return 0;
}
