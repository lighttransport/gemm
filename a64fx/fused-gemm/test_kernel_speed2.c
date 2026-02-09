/* Direct kernel throughput test - check for denormal slowdown */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <fenv.h>

typedef void (*gemm_fp16_fn)(const _Float16*, const _Float16*, float*,
                              int64_t, int64_t, int64_t);
typedef void (*gemm_fp32_fn)(const float*, const float*, float*,
                              int64_t, int64_t, int64_t);

extern void micro_kernel_fp16_12x2_swp(const _Float16*, const _Float16*, float*,
                                        int64_t, int64_t, int64_t);
extern void micro_kernel_fp32_12x2_swp(const float*, const float*, float*,
                                        int64_t, int64_t, int64_t);

static inline uint64_t rdtsc(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
}
static inline uint64_t rdfreq(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v;
}

static void bench_fp16(const char *label, _Float16 a_val, _Float16 b_val, int Kc) {
    double cpu_ghz = 2.0, timer_ghz = (double)rdfreq() / 1e9;
    int MR = 12, NR = 64;

    _Float16 *A = aligned_alloc(256, (size_t)Kc * MR * sizeof(_Float16));
    _Float16 *B = aligned_alloc(256, (size_t)Kc * NR * sizeof(_Float16));
    float *C = aligned_alloc(256, (size_t)MR * NR * sizeof(float));

    for (int i = 0; i < Kc * MR; i++) A[i] = a_val;
    for (int i = 0; i < Kc * NR; i++) B[i] = b_val;
    int64_t ldc = NR * sizeof(float);

    int REPS = 10000;
    memset(C, 0, MR * NR * 4);
    for (int i = 0; i < 100; i++)
        micro_kernel_fp16_12x2_swp(A, B, C, (int64_t)Kc, 0, ldc);

    uint64_t t0 = rdtsc();
    for (int r = 0; r < REPS; r++)
        micro_kernel_fp16_12x2_swp(A, B, C, (int64_t)Kc, 0, ldc);
    uint64_t t1 = rdtsc();

    double cycles = (double)(t1 - t0) * (cpu_ghz / timer_ghz);
    double cy_call = cycles / REPS;
    double flops = 2.0 * MR * Kc * NR;
    double total_fmla = flops / 64.0;
    double fmla_cy = total_fmla / cy_call;
    double gf = flops * REPS / (cycles / cpu_ghz) / 1e9;

    /* Check C for denormals */
    int n_denorm = 0, n_zero = 0;
    for (int i = 0; i < MR * NR; i++) {
        float v = C[i];
        if (v == 0.0f) n_zero++;
        else if (fabsf(v) < 6.1e-5f) n_denorm++;
    }

    printf("%-18s Kc=%3d: %6.0f cy  %5.2f FMLA/cy  %6.1f GF  C[0]=%.4e  denorm=%d zero=%d\n",
           label, Kc, cy_call, fmla_cy, gf, C[0], n_denorm, n_zero);

    free(A); free(B); free(C);
}

int main(void) {
    /* Check FPCR status */
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    printf("FPCR = 0x%016lx\n", fpcr);
    printf("  FZ  (bit 24) = %lu (flush-to-zero)\n", (fpcr >> 24) & 1);
    printf("  FZ16(bit 19) = %lu (flush-to-zero fp16)\n", (fpcr >> 19) & 1);
    printf("  AH  (bit 1)  = %lu (alt half-precision)\n", (fpcr >> 1) & 1);
    printf("\n");

    printf("=== FP16 Kernel: denormal investigation ===\n\n");
    printf("--- Kc=256, varying A/B values ---\n");
    bench_fp16("a=0.001 b=0.001",  (_Float16)0.001f, (_Float16)0.001f, 256);
    bench_fp16("a=0.01  b=0.01",   (_Float16)0.01f,  (_Float16)0.01f,  256);
    bench_fp16("a=0.1   b=0.1",    (_Float16)0.1f,   (_Float16)0.1f,   256);
    bench_fp16("a=1.0   b=0.001",  (_Float16)1.0f,   (_Float16)0.001f, 256);
    bench_fp16("a=1.0   b=1.0",    (_Float16)1.0f,   (_Float16)1.0f,   256);

    /* Now enable FZ16 (flush-to-zero for fp16) and retest */
    printf("\n--- Enable FZ16 (flush fp16 denorms to zero) ---\n");
    fpcr |= (1UL << 19);  /* Set FZ16 bit */
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    printf("FPCR = 0x%016lx  FZ16=%lu\n\n", fpcr, (fpcr >> 19) & 1);

    bench_fp16("FZ16 a=0.001",     (_Float16)0.001f, (_Float16)0.001f, 256);
    bench_fp16("FZ16 a=0.01",      (_Float16)0.01f,  (_Float16)0.01f,  256);
    bench_fp16("FZ16 a=1.0",       (_Float16)1.0f,   (_Float16)1.0f,   256);

    printf("\n--- Kc sweep with a=1.0, b=1.0 ---\n");
    bench_fp16("a=1.0", (_Float16)1.0f, (_Float16)1.0f, 16);
    bench_fp16("a=1.0", (_Float16)1.0f, (_Float16)1.0f, 32);
    bench_fp16("a=1.0", (_Float16)1.0f, (_Float16)1.0f, 64);
    bench_fp16("a=1.0", (_Float16)1.0f, (_Float16)1.0f, 128);
    bench_fp16("a=1.0", (_Float16)1.0f, (_Float16)1.0f, 256);

    return 0;
}
