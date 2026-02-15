#include <stdio.h>
#include <stdint.h>

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

// Static buffers
#define NC_MAX 256
static int32_t S[4*NC_MAX] __attribute__((aligned(64)));
static float V[NC_MAX*64] __attribute__((aligned(64)));
static float O[4*64] __attribute__((aligned(64)));
static float P[4*NC_MAX] __attribute__((aligned(64)));

int main(int argc, char** argv) {
    int M = 4;
    int Nc = argc > 1 ? atoi(argv[1]) : 64;
    if (Nc > NC_MAX) Nc = NC_MAX;
    int D = 64;
    int ld_s = Nc;
    int ld_v = D * 4;
    int ld_o = D * 4;

    printf("=== FlashAttention LD1RW Benchmark ===\n");
    printf("M=%d, Nc=%d, D=%d\n\n", M, Nc, D);

    for (int i = 0; i < M * Nc; i++) S[i] = i % 100;
    for (int i = 0; i < Nc * D; i++) V[i] = 0.01f;

    float scale = 0.1f;
    float max_val = 5.0f;

    const int warmup = 100;
    const int iters = 1000;
    uint64_t start, end;
    uint64_t freq = rdfreq();
    double total_flops = 2.0 * M * Nc * D;

    /* Test 1: exp2_flash_tiled_4x4 */
    for (int i = 0; i < M * D; i++) O[i] = 0;
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
    for (int i = 0; i < M * D; i++) O[i] = 0;
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

    return 0;
}
