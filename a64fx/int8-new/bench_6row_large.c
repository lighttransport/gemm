// bench_6row_large.c - Large benchmark comparing 5-row vs 6-row D=512 kernels
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define D 512

static inline uint64_t rdtsc(void) {
    uint64_t t;
    __asm__ volatile("mrs %0, CNTVCT_EL0" : "=r"(t));
    return t;
}

static inline uint64_t get_timer_freq(void) {
    uint64_t f;
    __asm__ volatile("mrs %0, CNTFRQ_EL0" : "=r"(f));
    return f;
}

extern void kernel_fused_d512_5row(const int8_t*, const int8_t*, const int8_t*, int32_t*);
extern void kernel_fused_d512_6row(const int8_t*, const int8_t*, const int8_t*, int32_t*);

void benchmark(int M, int L, uint64_t freq,
               int8_t* Q, int8_t* K_int, int8_t* V_t, int32_t* O) {
    int N_chunks = L / 64;
    int K_chunk_size = 128 * 64 * 4;   // D=512: 128 d_groups
    int V_chunk_size = 8 * 16 * 64 * 4; // D=512: 8 d_tiles

    int M_tiles_5 = M / 5;
    int M_tiles_6 = M / 6;
    int warmup = 2;
    int reps = 5;

    double total_flops = 2.0 * 2.0 * M * L * D;  // Q@K^T and P@V

    printf("\n=== M=%d, L=%d, D=%d ===\n", M, L, D);
    printf("5-row: %d M-tiles, 6-row: %d M-tiles\n", M_tiles_5, M_tiles_6);

    // Warmup 5-row
    for (int w = 0; w < warmup; w++) {
        memset(O, 0, M * D * sizeof(int32_t));
        for (int mt = 0; mt < M_tiles_5; mt++) {
            for (int nc = 0; nc < N_chunks; nc++) {
                kernel_fused_d512_5row(Q + mt * 5 * D,
                                        K_int + nc * K_chunk_size,
                                        V_t + nc * V_chunk_size,
                                        O + mt * 5 * D);
            }
        }
    }

    // Benchmark 5-row
    uint64_t best_5 = UINT64_MAX;
    for (int r = 0; r < reps; r++) {
        memset(O, 0, M * D * sizeof(int32_t));
        uint64_t t0 = rdtsc();
        for (int mt = 0; mt < M_tiles_5; mt++) {
            for (int nc = 0; nc < N_chunks; nc++) {
                kernel_fused_d512_5row(Q + mt * 5 * D,
                                        K_int + nc * K_chunk_size,
                                        V_t + nc * V_chunk_size,
                                        O + mt * 5 * D);
            }
        }
        uint64_t t1 = rdtsc();
        if (t1 - t0 < best_5) best_5 = t1 - t0;
    }

    // Warmup 6-row
    for (int w = 0; w < warmup; w++) {
        memset(O, 0, M * D * sizeof(int32_t));
        for (int mt = 0; mt < M_tiles_6; mt++) {
            for (int nc = 0; nc < N_chunks; nc++) {
                kernel_fused_d512_6row(Q + mt * 6 * D,
                                        K_int + nc * K_chunk_size,
                                        V_t + nc * V_chunk_size,
                                        O + mt * 6 * D);
            }
        }
    }

    // Benchmark 6-row
    uint64_t best_6 = UINT64_MAX;
    for (int r = 0; r < reps; r++) {
        memset(O, 0, M * D * sizeof(int32_t));
        uint64_t t0 = rdtsc();
        for (int mt = 0; mt < M_tiles_6; mt++) {
            for (int nc = 0; nc < N_chunks; nc++) {
                kernel_fused_d512_6row(Q + mt * 6 * D,
                                        K_int + nc * K_chunk_size,
                                        V_t + nc * V_chunk_size,
                                        O + mt * 6 * D);
            }
        }
        uint64_t t1 = rdtsc();
        if (t1 - t0 < best_6) best_6 = t1 - t0;
    }

    double sec_5 = (double)best_5 / freq;
    double sec_6 = (double)best_6 / freq;
    double gflops_5 = total_flops / sec_5 / 1e9;
    double gflops_6 = total_flops / sec_6 / 1e9;

    // Peak is 128 GOPS at 2 GHz (2 SDOT/cycle * 4 lanes * 4 ops * 2000 MHz * 2 FPU)
    double peak_gops = 128.0;
    double eff_5 = gflops_5 / peak_gops * 100;
    double eff_6 = gflops_6 / peak_gops * 100;

    int calls_5 = M_tiles_5 * N_chunks;
    int calls_6 = M_tiles_6 * N_chunks;

    printf("\nKernel                     Cycles     GFLOPS       Eff%%\n");
    printf("-------------------- ------------ ---------- ----------\n");
    printf("5-row                %12lu %10.1f %9.1f%%\n",
           best_5, gflops_5, eff_5);
    printf("6-row                %12lu %10.1f %9.1f%%\n",
           best_6, gflops_6, eff_6);

    printf("\nSpeedup 6-row/5-row: %.2fx\n", sec_5 / sec_6);
    printf("Cycles/call: 5-row=%lu, 6-row=%lu\n",
           best_5/calls_5, best_6/calls_6);

    // Per-call analysis
    // 5-row D=512: 5*128*64*4=163840 SDOTs for Q@K^T, 5*8*64*16*4=163840 for P@V
    // Total: 327680 SDOTs per call
    // 6-row D=512: 6*128*64*4=196608 SDOTs for Q@K^T, 6*8*64*16*4=196608 for P@V
    // Total: 393216 SDOTs per call
    int sdots_5 = 5 * 128 * 64 * 4 + 5 * 8 * 64 * 16 * 4;
    int sdots_6 = 6 * 128 * 64 * 4 + 6 * 8 * 64 * 16 * 4;
    printf("SDOTs/call: 5-row=%d, 6-row=%d\n", sdots_5/4, sdots_6/4);
}

int main() {
    printf("================================================================\n");
    printf("5-row vs 6-row Kernel Benchmark (D=512)\n");
    printf("Load latency: 11 cycles\n");
    printf("================================================================\n");

    uint64_t freq = get_timer_freq();
    printf("Timer freq: %lu Hz\n", freq);

    // M=1800 is divisible by both 5 and 6 (LCM=30)
    int M = 1800;
    int L_values[] = {4096, 8192};
    int num_L = 2;

    int max_L = 8192;
    int N_chunks_max = max_L / 64;
    int K_chunk_size = 128 * 64 * 4;   // D=512: 128 d_groups
    int V_chunk_size = 8 * 16 * 64 * 4; // D=512: 8 d_tiles

    printf("\nAllocating memory...\n");
    int8_t* Q = aligned_alloc(64, M * D);
    int8_t* K_int = aligned_alloc(64, N_chunks_max * K_chunk_size);
    int8_t* V_t = aligned_alloc(64, N_chunks_max * V_chunk_size);
    int32_t* O = aligned_alloc(64, M * D * sizeof(int32_t));

    if (!Q || !K_int || !V_t || !O) {
        printf("Allocation failed!\n");
        return 1;
    }

    printf("Initializing with random data...\n");
    for (size_t i = 0; i < M * D; i++) Q[i] = rand() % 256 - 128;
    for (size_t i = 0; i < (size_t)N_chunks_max * K_chunk_size; i++) K_int[i] = rand() % 256 - 128;
    for (size_t i = 0; i < (size_t)N_chunks_max * V_chunk_size; i++) V_t[i] = rand() % 256 - 128;

    for (int i = 0; i < num_L; i++) {
        benchmark(M, L_values[i], freq, Q, K_int, V_t, O);
    }

    free(Q); free(K_int); free(V_t); free(O);

    printf("\n================================================================\n");
    printf("Analysis:\n");
    printf("- 5-row: 20 SDOTs / 9 loads = 2.22 SDOT/load\n");
    printf("- 6-row: 24 SDOTs / 10 loads = 2.40 SDOT/load (8%% better)\n");
    printf("- Need ~22 SDOTs to fully hide 11-cycle latency (2 SDOT/cycle)\n");
    printf("================================================================\n");

    return 0;
}
