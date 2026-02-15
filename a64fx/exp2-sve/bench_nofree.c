#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

extern void exp2_flash_tiled_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

extern void exp2_flash_ld1rw_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

static inline uint64_t rdcycle(void) {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t rdfreq(void) {
    uint64_t val;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

int main(int argc, char** argv) {
    const int M = 4;
    const int Nc = argc > 1 ? atoi(argv[1]) : 64;
    const int D = 64;
    const int ld_s = Nc;
    const int ld_v = D * 4;
    const int ld_o = D * 4;

    printf("=== FlashAttention LD1RW Benchmark ===\n");
    printf("M=%d, Nc=%d, D=%d\n\n", M, Nc, D);

    int32_t* S = malloc(M * Nc * sizeof(int32_t));
    float* V = malloc(Nc * D * sizeof(float));
    float* O = malloc(M * D * sizeof(float));
    float* P = malloc(M * Nc * sizeof(float));

    for (int i = 0; i < M * Nc; i++) S[i] = (rand() % 100);
    for (int i = 0; i < Nc * D; i++) V[i] = 0.01f;

    float scale = 0.1f;
    float max_val = 5.0f;

    const int warmup = 10;
    const int iters = 100;
    uint64_t start, end;
    uint64_t freq = rdfreq();
    double total_flops = 2.0 * M * Nc * D;

    /* Test 1: exp2_flash_tiled_4x4 */
    memset(O, 0, M * D * sizeof(float));
    for (int i = 0; i < warmup; i++) {
        exp2_flash_tiled_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }

    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        exp2_flash_tiled_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }
    end = rdcycle();

    double cycles = (double)(end - start) / iters;
    double gflops = total_flops / cycles * (freq / 1e9);
    printf("exp2_flash_tiled_4x4: %.1f cycles, %.2f GFLOPS (%.1f%% peak)\n",
           cycles, gflops, gflops / 128.0 * 100);

    /* Test 2: exp2_flash_ld1rw_4x4 */
    memset(O, 0, M * D * sizeof(float));
    for (int i = 0; i < warmup; i++) {
        exp2_flash_ld1rw_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }

    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        exp2_flash_ld1rw_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }
    end = rdcycle();

    cycles = (double)(end - start) / iters;
    gflops = total_flops / cycles * (freq / 1e9);
    printf("exp2_flash_ld1rw_4x4: %.1f cycles, %.2f GFLOPS (%.1f%% peak)\n",
           cycles, gflops, gflops / 128.0 * 100);

    printf("\nP[0:4] = %.4f, %.4f, %.4f, %.4f\n", P[0], P[1], P[2], P[3]);

    // Don't free - let OS clean up to avoid corruption issue
    return 0;
}
