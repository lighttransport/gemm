/*
 * FP16 12x2 kernel comparison: SWP vs SWP4 vs SPLITB vs NOEPI
 * NOEPI = no fp16→fp32 conversion epilogue, stores fp16 directly.
 * Measures the conversion overhead ceiling.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <arm_sve.h>

#define MR 12
#define NR 64

/* SWP kernels (2K unrolled, fp32 C) */
extern void micro_kernel_fp16_12x2_swp(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_swp_accum(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);

/* SWP4 kernels (4K unrolled, fp32 C) */
extern void micro_kernel_fp16_12x2_swp4(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_swp4_accum(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);

/* NOEPI kernels (2K unrolled, fp16 C — no conversion) */
extern void micro_kernel_fp16_12x2_noepi(
    const _Float16*, const _Float16*, _Float16*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_noepi_accum(
    const _Float16*, const _Float16*, _Float16*, int64_t, int64_t, int64_t);

typedef void (*kernel_fn)(const void*, const void*, void*, int64_t, int64_t, int64_t);

static inline uint64_t rdtick(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
}
static inline uint64_t tick_freq(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v;
}

/*
 * L1-resident single tile benchmark (fp32 C)
 */
static void bench_l1_fp32(const char *name, kernel_fn init_fn, kernel_fn accum_fn, int Kc)
{
    int warmup = 500;
    int iters = 10000;

    size_t a_sz = ((size_t)Kc * MR * sizeof(_Float16) + 255) & ~(size_t)255;
    size_t b_sz = ((size_t)Kc * NR * sizeof(_Float16) + 255) & ~(size_t)255;
    size_t c_sz = ((size_t)MR * NR * sizeof(float) + 255) & ~(size_t)255;

    _Float16 *A = (_Float16 *)aligned_alloc(256, a_sz);
    _Float16 *B = (_Float16 *)aligned_alloc(256, b_sz);
    float *C = (float *)aligned_alloc(256, c_sz);

    srand(42);
    for (size_t i = 0; i < (size_t)Kc * MR; i++)
        A[i] = (_Float16)(0.01f * ((float)rand() / (float)RAND_MAX));
    for (size_t i = 0; i < (size_t)Kc * NR; i++)
        B[i] = (_Float16)((float)rand() / (float)RAND_MAX - 0.5f);

    double flops_per_call = 2.0 * MR * Kc * NR;
    int64_t ldc = (int64_t)(NR * sizeof(float));

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
        gf_init = flops_per_call * iters / ((double)(t1 - t0) * t2s) / 1e9;
    }
    memset(C, 0, c_sz);
    init_fn(A, B, C, (int64_t)Kc, 0, ldc);
    {
        uint64_t t0 = rdtick();
        for (int r = 0; r < iters; r++)
            accum_fn(A, B, C, (int64_t)Kc, 0, ldc);
        uint64_t t1 = rdtick();
        gf_accum = flops_per_call * iters / ((double)(t1 - t0) * t2s) / 1e9;
    }

    printf("  %-8s K=%3d: INIT %.1f GF (%.1f%%)  ACCUM %.1f GF (%.1f%%)\n",
           name, Kc, gf_init, gf_init/256*100, gf_accum, gf_accum/256*100);

    free(A); free(B); free(C);
}

/*
 * L1-resident single tile benchmark (fp16 C — no conversion)
 */
static void bench_l1_fp16(const char *name, kernel_fn init_fn, kernel_fn accum_fn, int Kc)
{
    int warmup = 500;
    int iters = 10000;

    size_t a_sz = ((size_t)Kc * MR * sizeof(_Float16) + 255) & ~(size_t)255;
    size_t b_sz = ((size_t)Kc * NR * sizeof(_Float16) + 255) & ~(size_t)255;
    size_t c_sz = ((size_t)MR * NR * sizeof(_Float16) + 255) & ~(size_t)255;

    _Float16 *A = (_Float16 *)aligned_alloc(256, a_sz);
    _Float16 *B = (_Float16 *)aligned_alloc(256, b_sz);
    _Float16 *C = (_Float16 *)aligned_alloc(256, c_sz);

    srand(42);
    for (size_t i = 0; i < (size_t)Kc * MR; i++)
        A[i] = (_Float16)(0.01f * ((float)rand() / (float)RAND_MAX));
    for (size_t i = 0; i < (size_t)Kc * NR; i++)
        B[i] = (_Float16)((float)rand() / (float)RAND_MAX - 0.5f);

    double flops_per_call = 2.0 * MR * Kc * NR;
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
        gf_init = flops_per_call * iters / ((double)(t1 - t0) * t2s) / 1e9;
    }
    memset(C, 0, c_sz);
    init_fn(A, B, C, (int64_t)Kc, 0, ldc);
    {
        uint64_t t0 = rdtick();
        for (int r = 0; r < iters; r++)
            accum_fn(A, B, C, (int64_t)Kc, 0, ldc);
        uint64_t t1 = rdtick();
        gf_accum = flops_per_call * iters / ((double)(t1 - t0) * t2s) / 1e9;
    }

    printf("  %-8s K=%3d: INIT %.1f GF (%.1f%%)  ACCUM %.1f GF (%.1f%%)\n",
           name, Kc, gf_init, gf_init/256*100, gf_accum, gf_accum/256*100);

    free(A); free(B); free(C);
}

/*
 * Full GEMM streaming (fp32 C)
 */
static void bench_gemm_fp32(const char *name, kernel_fn init_fn, kernel_fn accum_fn,
                             int Kc, int L, int d_out)
{
    int n_tiles = d_out / NR;
    int n_kblocks = L / Kc;
    int64_t ldo_bytes = (int64_t)d_out * sizeof(float);
    int warmup = 2, iters = 5;

    size_t pp_sz = ((size_t)L * MR * sizeof(_Float16) + 255) & ~(size_t)255;
    size_t vp_sz = ((size_t)n_kblocks * n_tiles * Kc * NR * sizeof(_Float16) + 255) & ~(size_t)255;
    size_t o_sz = ((size_t)MR * d_out * sizeof(float) + 255) & ~(size_t)255;

    _Float16 *Pp = (_Float16 *)aligned_alloc(256, pp_sz);
    _Float16 *Vp = (_Float16 *)aligned_alloc(256, vp_sz);
    float *O = (float *)aligned_alloc(256, o_sz);

    srand(42);
    for (size_t i = 0; i < (size_t)L * MR; i++)
        Pp[i] = (_Float16)(0.001f * ((float)rand() / (float)RAND_MAX));
    for (size_t i = 0; i < (size_t)n_kblocks * n_tiles * Kc * NR; i++)
        Vp[i] = (_Float16)((float)rand() / (float)RAND_MAX - 0.5f);

    double flops = 2.0 * MR * (double)L * d_out;

    for (int r = 0; r < warmup; r++) {
        memset(O, 0, o_sz);
        for (int kc = 0; kc < L; kc += Kc) {
            int kb = kc / Kc;
            const _Float16 *Pp_blk = Pp + kc * MR;
            for (int t = 0; t < n_tiles; t++) {
                const _Float16 *Vpt = Vp + ((size_t)kb * n_tiles + t) * Kc * NR;
                if (kc == 0)
                    init_fn(Pp_blk, Vpt, O + t * NR, (int64_t)Kc, 0, ldo_bytes);
                else
                    accum_fn(Pp_blk, Vpt, O + t * NR, (int64_t)Kc, 0, ldo_bytes);
            }
        }
    }

    double t2s = 1.0 / (double)tick_freq();
    double best_gf = 0;
    for (int r = 0; r < iters; r++) {
        memset(O, 0, o_sz);
        uint64_t t0 = rdtick();
        for (int kc = 0; kc < L; kc += Kc) {
            int kb = kc / Kc;
            const _Float16 *Pp_blk = Pp + kc * MR;
            for (int t = 0; t < n_tiles; t++) {
                const _Float16 *Vpt = Vp + ((size_t)kb * n_tiles + t) * Kc * NR;
                if (kc == 0)
                    init_fn(Pp_blk, Vpt, O + t * NR, (int64_t)Kc, 0, ldo_bytes);
                else
                    accum_fn(Pp_blk, Vpt, O + t * NR, (int64_t)Kc, 0, ldo_bytes);
            }
        }
        uint64_t t1 = rdtick();
        double gf = flops / ((double)(t1 - t0) * t2s) / 1e9;
        if (gf > best_gf) best_gf = gf;
    }
    printf("  %-8s K=%3d L=%5d: %.1f GF (%.1f%%)\n",
           name, Kc, L, best_gf, best_gf/256*100);

    free(Pp); free(Vp); free(O);
}

/*
 * Full GEMM streaming (fp16 C — no conversion)
 */
static void bench_gemm_fp16(const char *name, kernel_fn init_fn, kernel_fn accum_fn,
                             int Kc, int L, int d_out)
{
    int n_tiles = d_out / NR;
    int n_kblocks = L / Kc;
    int64_t ldo_bytes = (int64_t)d_out * sizeof(_Float16);
    int warmup = 2, iters = 5;

    size_t pp_sz = ((size_t)L * MR * sizeof(_Float16) + 255) & ~(size_t)255;
    size_t vp_sz = ((size_t)n_kblocks * n_tiles * Kc * NR * sizeof(_Float16) + 255) & ~(size_t)255;
    size_t o_sz = ((size_t)MR * d_out * sizeof(_Float16) + 255) & ~(size_t)255;

    _Float16 *Pp = (_Float16 *)aligned_alloc(256, pp_sz);
    _Float16 *Vp = (_Float16 *)aligned_alloc(256, vp_sz);
    _Float16 *O = (_Float16 *)aligned_alloc(256, o_sz);

    srand(42);
    for (size_t i = 0; i < (size_t)L * MR; i++)
        Pp[i] = (_Float16)(0.001f * ((float)rand() / (float)RAND_MAX));
    for (size_t i = 0; i < (size_t)n_kblocks * n_tiles * Kc * NR; i++)
        Vp[i] = (_Float16)((float)rand() / (float)RAND_MAX - 0.5f);

    double flops = 2.0 * MR * (double)L * d_out;

    for (int r = 0; r < warmup; r++) {
        memset(O, 0, o_sz);
        for (int kc = 0; kc < L; kc += Kc) {
            int kb = kc / Kc;
            const _Float16 *Pp_blk = Pp + kc * MR;
            for (int t = 0; t < n_tiles; t++) {
                const _Float16 *Vpt = Vp + ((size_t)kb * n_tiles + t) * Kc * NR;
                if (kc == 0)
                    init_fn(Pp_blk, Vpt, O + t * NR, (int64_t)Kc, 0, ldo_bytes);
                else
                    accum_fn(Pp_blk, Vpt, O + t * NR, (int64_t)Kc, 0, ldo_bytes);
            }
        }
    }

    double t2s = 1.0 / (double)tick_freq();
    double best_gf = 0;
    for (int r = 0; r < iters; r++) {
        memset(O, 0, o_sz);
        uint64_t t0 = rdtick();
        for (int kc = 0; kc < L; kc += Kc) {
            int kb = kc / Kc;
            const _Float16 *Pp_blk = Pp + kc * MR;
            for (int t = 0; t < n_tiles; t++) {
                const _Float16 *Vpt = Vp + ((size_t)kb * n_tiles + t) * Kc * NR;
                if (kc == 0)
                    init_fn(Pp_blk, Vpt, O + t * NR, (int64_t)Kc, 0, ldo_bytes);
                else
                    accum_fn(Pp_blk, Vpt, O + t * NR, (int64_t)Kc, 0, ldo_bytes);
            }
        }
        uint64_t t1 = rdtick();
        double gf = flops / ((double)(t1 - t0) * t2s) / 1e9;
        if (gf > best_gf) best_gf = gf;
    }
    printf("  %-8s K=%3d L=%5d: %.1f GF (%.1f%%)\n",
           name, Kc, L, best_gf, best_gf/256*100);

    free(Pp); free(Vp); free(O);
}

int main(void)
{
    /* Enable FZ16 */
    {
        uint64_t fpcr;
        __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
        fpcr |= (1UL << 19);
        __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
    }

    printf("=== FP16 12x2 kernel comparison: conversion overhead analysis ===\n");
    printf("MR=%d, NR=%d, peak=256 GF\n", MR, NR);
    printf("SWP/SWP4 = fp32 C (with uunpk+fcvt epilogue)\n");
    printf("NOEPI    = fp16 C (direct st1h, no conversion)\n\n");

    printf("--- L1-resident single tile ---\n");
    bench_l1_fp32("SWP",    (kernel_fn)micro_kernel_fp16_12x2_swp,    (kernel_fn)micro_kernel_fp16_12x2_swp_accum,    128);
    bench_l1_fp32("SWP4",   (kernel_fn)micro_kernel_fp16_12x2_swp4,   (kernel_fn)micro_kernel_fp16_12x2_swp4_accum,   128);
    bench_l1_fp16("NOEPI",  (kernel_fn)micro_kernel_fp16_12x2_noepi,  (kernel_fn)micro_kernel_fp16_12x2_noepi_accum,  128);
    printf("\n");
    bench_l1_fp32("SWP",    (kernel_fn)micro_kernel_fp16_12x2_swp,    (kernel_fn)micro_kernel_fp16_12x2_swp_accum,    256);
    bench_l1_fp32("SWP4",   (kernel_fn)micro_kernel_fp16_12x2_swp4,   (kernel_fn)micro_kernel_fp16_12x2_swp4_accum,   256);
    bench_l1_fp16("NOEPI",  (kernel_fn)micro_kernel_fp16_12x2_noepi,  (kernel_fn)micro_kernel_fp16_12x2_noepi_accum,  256);

    printf("\n--- Full GEMM streaming (d=256) ---\n");
    bench_gemm_fp32("SWP",    (kernel_fn)micro_kernel_fp16_12x2_swp,    (kernel_fn)micro_kernel_fp16_12x2_swp_accum,    256, 32768, 256);
    bench_gemm_fp32("SWP4",   (kernel_fn)micro_kernel_fp16_12x2_swp4,   (kernel_fn)micro_kernel_fp16_12x2_swp4_accum,   256, 32768, 256);
    bench_gemm_fp16("NOEPI",  (kernel_fn)micro_kernel_fp16_12x2_noepi,  (kernel_fn)micro_kernel_fp16_12x2_noepi_accum,  256, 32768, 256);

    return 0;
}
