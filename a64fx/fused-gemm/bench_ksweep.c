/*
 * K-sweep: NOEPI vs NOEPI4 (fp16 C) L1-resident efficiency ceiling
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <arm_sve.h>

#define MR 12
#define NR 64

extern void micro_kernel_fp16_12x2_noepi(
    const _Float16*, const _Float16*, _Float16*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_noepi_accum(
    const _Float16*, const _Float16*, _Float16*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_noepi4(
    const _Float16*, const _Float16*, _Float16*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_noepi4_accum(
    const _Float16*, const _Float16*, _Float16*, int64_t, int64_t, int64_t);

typedef void (*kernel_fn)(const _Float16*, const _Float16*, _Float16*, int64_t, int64_t, int64_t);

static inline uint64_t rdtick(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
}
static inline uint64_t tick_freq(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v;
}

static void bench_kernel(const char *name, kernel_fn init_fn, kernel_fn accum_fn, int Kc)
{
    int warmup = 500;
    int iters = 10000;

    size_t a_sz = ((size_t)Kc * MR * 2 + 255) & ~(size_t)255;
    size_t b_sz = ((size_t)Kc * NR * 2 + 255) & ~(size_t)255;
    size_t c_sz = ((size_t)MR * NR * 2 + 255) & ~(size_t)255;

    _Float16 *A = (_Float16 *)aligned_alloc(256, a_sz);
    _Float16 *B = (_Float16 *)aligned_alloc(256, b_sz);
    _Float16 *C = (_Float16 *)aligned_alloc(256, c_sz);

    srand(42);
    for (size_t i = 0; i < (size_t)Kc * MR; i++)
        A[i] = (_Float16)(0.01f * ((float)rand() / (float)RAND_MAX));
    for (size_t i = 0; i < (size_t)Kc * NR; i++)
        B[i] = (_Float16)((float)rand() / (float)RAND_MAX - 0.5f);

    double flops = 2.0 * MR * Kc * NR;
    int64_t ldc = (int64_t)(NR * sizeof(_Float16));

    for (int r = 0; r < warmup; r++) {
        memset(C, 0, c_sz);
        init_fn(A, B, C, (int64_t)Kc, 0, ldc);
    }

    double t2s = 1.0 / (double)tick_freq();
    double gf_init, gf_accum;
    {
        uint64_t t0 = rdtick();
        for (int r = 0; r < iters; r++)
            init_fn(A, B, C, (int64_t)Kc, 0, ldc);
        uint64_t t1 = rdtick();
        gf_init = flops * iters / ((double)(t1 - t0) * t2s) / 1e9;
    }
    memset(C, 0, c_sz);
    init_fn(A, B, C, (int64_t)Kc, 0, ldc);
    {
        uint64_t t0 = rdtick();
        for (int r = 0; r < iters; r++)
            accum_fn(A, B, C, (int64_t)Kc, 0, ldc);
        uint64_t t1 = rdtick();
        gf_accum = flops * iters / ((double)(t1 - t0) * t2s) / 1e9;
    }

    size_t total = (size_t)Kc * MR * 2 + (size_t)Kc * NR * 2 + MR * NR * 2;
    printf("  %-8s K=%3d: INIT %.1f GF (%.1f%%)  ACCUM %.1f GF (%.1f%%)  [%zuB]\n",
           name, Kc, gf_init, gf_init/256*100, gf_accum, gf_accum/256*100, total);

    free(A); free(B); free(C);
}

int main(void)
{
    {
        uint64_t fpcr;
        __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
        fpcr |= (1UL << 19);
        __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
    }

    printf("=== NOEPI vs NOEPI4 (fp16 C, L1-resident) ===\n");
    printf("MR=%d NR=%d peak=256GF L1D=64KB\n\n", MR, NR);

    int Ks[] = {128, 192, 256, 320, 384};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    for (int ki = 0; ki < nK; ki++) {
        int Kc = Ks[ki];
        bench_kernel("NOEPI",  micro_kernel_fp16_12x2_noepi,  micro_kernel_fp16_12x2_noepi_accum,  Kc);
        bench_kernel("NOEPI4", micro_kernel_fp16_12x2_noepi4, micro_kernel_fp16_12x2_noepi4_accum, Kc);
        printf("\n");
    }

    return 0;
}
