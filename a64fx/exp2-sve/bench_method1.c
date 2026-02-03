/*
 * Clean benchmark for Method 1: exp2_fast + pack + 8x3 GEMM
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

extern void exp2_fast_rows(const int32_t* S, float* P, int M, int Nc,
                           float scale, float neg_max, int ld_s, int ld_p);
extern void micro_kernel_fp32_8x3_unroll4(const float* A, const float* B,
                                           float* C, int K, int alpha_flag, int ldc);

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

/* Pack row-major P[M][Nc] to column-major P_packed[Nc][M] */
static void pack_p_colmajor(const float* P, float* P_packed, int M, int Nc) {
    for (int k = 0; k < Nc; k++) {
        for (int m = 0; m < M; m++) {
            P_packed[k * M + m] = P[m * Nc + k];
        }
    }
}

int main(int argc, char* argv[]) {
    int Nc = (argc > 1) ? atoi(argv[1]) : 256;
    int D = (argc > 2) ? atoi(argv[2]) : 48;
    int iterations = (argc > 3) ? atoi(argv[3]) : 2000;

    const int M = 8;
    const int n_tiles = (D + 47) / 48;
    const int D_padded = n_tiles * 48;

    printf("M=%d, Nc=%d, D=%d, iters=%d\n", M, Nc, D, iterations);

    /* Allocate */
    int32_t* S = aligned_alloc(256, M * Nc * sizeof(int32_t));
    float* P = aligned_alloc(256, M * Nc * sizeof(float));
    float* P_packed = aligned_alloc(256, Nc * M * sizeof(float));
    float* V = aligned_alloc(256, Nc * D_padded * sizeof(float));
    float* O = aligned_alloc(256, M * D_padded * sizeof(float));

    float scale = 1.0f / 8.0f;
    float neg_max = -5.0f;

    srand(42);
    for (int i = 0; i < M * Nc; i++) S[i] = (rand() % 100) - 50;
    for (int i = 0; i < Nc * D_padded; i++) V[i] = 0.01f;

    /* Timer params */
    double timer_freq = 100e6;
    double cpu_freq = 2.0e9;
    double fp32_peak = 128.0;
    uint64_t total_ops = (uint64_t)M * Nc * D * 2;

    int ld_c = D_padded;
    uint64_t start, end, elapsed;
    double cycles, gflops;

    /* Warmup */
    for (int i = 0; i < 100; i++) {
        exp2_fast_rows(S, P, M, Nc, scale, neg_max, Nc, Nc);
        pack_p_colmajor(P, P_packed, M, Nc);
        for (int t = 0; t < n_tiles; t++)
            micro_kernel_fp32_8x3_unroll4(P_packed, V + t*48, O + t*48, Nc, 0, ld_c);
    }

    /* Full pipeline */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_fast_rows(S, P, M, Nc, scale, neg_max, Nc, Nc);
        pack_p_colmajor(P, P_packed, M, Nc);
        for (int t = 0; t < n_tiles; t++)
            micro_kernel_fp32_8x3_unroll4(P_packed, V + t*48, O + t*48, Nc, 0, ld_c);
    }
    end = get_cycles();
    elapsed = end - start;
    cycles = (double)elapsed * (cpu_freq / timer_freq) / iterations;
    gflops = (double)total_ops * iterations / ((double)elapsed / timer_freq) / 1e9;
    printf("Full:     %.0f cycles, %.2f GFLOPS (%.1f%% peak)\n", cycles, gflops, gflops/fp32_peak*100);

    /* exp2 only */
    start = get_cycles();
    for (int i = 0; i < iterations; i++)
        exp2_fast_rows(S, P, M, Nc, scale, neg_max, Nc, Nc);
    end = get_cycles();
    double exp2_cycles = (double)(end - start) * (cpu_freq / timer_freq) / iterations;
    printf("exp2:     %.0f cycles (%.2f cycles/elem)\n", exp2_cycles, exp2_cycles / (M*Nc));

    /* pack only */
    start = get_cycles();
    for (int i = 0; i < iterations; i++)
        pack_p_colmajor(P, P_packed, M, Nc);
    end = get_cycles();
    double pack_cycles = (double)(end - start) * (cpu_freq / timer_freq) / iterations;
    printf("pack:     %.0f cycles (%.2f cycles/elem)\n", pack_cycles, pack_cycles / (M*Nc));

    /* GEMM only */
    start = get_cycles();
    for (int i = 0; i < iterations; i++)
        for (int t = 0; t < n_tiles; t++)
            micro_kernel_fp32_8x3_unroll4(P_packed, V + t*48, O + t*48, Nc, 0, ld_c);
    end = get_cycles();
    double gemm_cycles = (double)(end - start) * (cpu_freq / timer_freq) / iterations;
    double gemm_gflops = (double)total_ops * iterations / ((double)(end - start) / timer_freq) / 1e9;
    printf("GEMM:     %.0f cycles, %.2f GFLOPS (%.1f%% peak)\n", gemm_cycles, gemm_gflops, gemm_gflops/fp32_peak*100);

    printf("\nOverhead: exp2=%.1f%%, pack=%.1f%% of GEMM\n",
           exp2_cycles/gemm_cycles*100, pack_cycles/gemm_cycles*100);
    printf("Sum of phases: %.0f cycles (measured full: %.0f)\n",
           exp2_cycles + pack_cycles + gemm_cycles, cycles);

    free(S); free(P); free(P_packed); free(V); free(O);
    return 0;
}
