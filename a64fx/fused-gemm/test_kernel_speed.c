/* Direct kernel throughput: call kernel repeatedly with L1-resident data */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

typedef void (*gemm_fp32_fn)(const float*, const float*, float*,
                              int64_t, int64_t, int64_t);
typedef void (*gemm_fp16_fn)(const _Float16*, const _Float16*, float*,
                              int64_t, int64_t, int64_t);

extern void micro_kernel_fp32_12x2_swp(const float*, const float*, float*,
                                        int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_swp(const _Float16*, const _Float16*, float*,
                                        int64_t, int64_t, int64_t);

static inline uint64_t rdtsc(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
}
static inline uint64_t rdfreq(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v;
}

int main(void) {
    double freq = (double)rdfreq();
    double cpu_ghz = 2.0;
    double timer_ghz = freq / 1e9;

    /* Kernel parameters */
    int MR = 12;
    int Kc = 256;    /* K dimension */
    int NR32 = 32;   /* fp32: 2 SVE vecs × 16 */
    int NR16 = 64;   /* fp16: 2 SVE vecs × 32 */

    /* Allocate L1-resident buffers */
    /* fp32: A = MR×Kc×4 = 12KB, B = NR×Kc×4 = 32KB, C = MR×NR×4 = 1.5KB */
    float *A32 = aligned_alloc(256, (size_t)Kc * MR * sizeof(float));
    float *B32 = aligned_alloc(256, (size_t)Kc * NR32 * sizeof(float));
    float *C32 = aligned_alloc(256, (size_t)MR * NR32 * sizeof(float));

    /* fp16: A = MR×Kc×2 = 6KB, B = NR×Kc×2 = 32KB, C = MR×NR×4 = 3KB */
    _Float16 *A16 = aligned_alloc(256, (size_t)Kc * MR * sizeof(_Float16));
    _Float16 *B16 = aligned_alloc(256, (size_t)Kc * NR16 * sizeof(_Float16));
    float *C16 = aligned_alloc(256, (size_t)MR * NR16 * sizeof(float));

    /* Initialize with small values */
    for (int i = 0; i < Kc * MR; i++) {
        A32[i] = 0.001f * (i % 7);
        A16[i] = (_Float16)(0.001f * (i % 7));
    }
    for (int i = 0; i < Kc * NR32; i++) B32[i] = 0.001f * (i % 11);
    for (int i = 0; i < Kc * NR16; i++) B16[i] = (_Float16)(0.001f * (i % 11));

    /* A is packed as K×MR (row = k, col = m), each row = MR elements */
    /* B is packed as K×NR (row = k, col = n), each row = NR elements */
    int64_t ldc32 = NR32 * sizeof(float);  /* C row stride in bytes */
    int64_t ldc16 = NR16 * sizeof(float);  /* C row stride in bytes (fp32 output) */

    int REPS = 10000;
    double flops32 = 2.0 * MR * (double)Kc * NR32;
    double flops16 = 2.0 * MR * (double)Kc * NR16;

    printf("=== Single-Tile Kernel Throughput (L1-resident, Kc=%d) ===\n", Kc);
    printf("FP32: MR=%d NR=%d, A=%.1fKB B=%.1fKB C=%.1fKB, FLOPs/call=%g\n",
           MR, NR32, Kc*MR*4.0/1024, Kc*NR32*4.0/1024, MR*NR32*4.0/1024, flops32);
    printf("FP16: MR=%d NR=%d, A=%.1fKB B=%.1fKB C=%.1fKB, FLOPs/call=%g\n",
           MR, NR16, Kc*MR*2.0/1024, Kc*NR16*2.0/1024, MR*NR16*4.0/1024, flops16);
    printf("\n");

    /* ─── FP32 kernel ─── */
    {
        /* Warmup */
        for (int i = 0; i < 100; i++) {
            memset(C32, 0, MR * NR32 * 4);
            micro_kernel_fp32_12x2_swp(A32, B32, C32, (int64_t)Kc, 0, ldc32);
        }
        uint64_t t0 = rdtsc();
        for (int r = 0; r < REPS; r++) {
            memset(C32, 0, MR * NR32 * 4);
            micro_kernel_fp32_12x2_swp(A32, B32, C32, (int64_t)Kc, 0, ldc32);
        }
        uint64_t t1 = rdtsc();
        double cycles = (double)(t1 - t0) * (cpu_ghz / timer_ghz);
        double cy_per_call = cycles / REPS;
        double gf = flops32 * REPS / (cycles / cpu_ghz) / 1e9;
        double fmla_total = flops32 / 32.0; /* 16 elem × 2 = 32 FLOPs per FMLA */
        printf("FP32: %.0f cy/call  %.1f GF  (%.2f FMLA/cy, of %.0f total FMLAs)\n",
               cy_per_call, gf, fmla_total / cy_per_call, fmla_total);
    }

    /* ─── FP16 kernel ─── */
    {
        for (int i = 0; i < 100; i++) {
            memset(C16, 0, MR * NR16 * 4);
            micro_kernel_fp16_12x2_swp(A16, B16, C16, (int64_t)Kc, 0, ldc16);
        }
        uint64_t t0 = rdtsc();
        for (int r = 0; r < REPS; r++) {
            memset(C16, 0, MR * NR16 * 4);
            micro_kernel_fp16_12x2_swp(A16, B16, C16, (int64_t)Kc, 0, ldc16);
        }
        uint64_t t1 = rdtsc();
        double cycles = (double)(t1 - t0) * (cpu_ghz / timer_ghz);
        double cy_per_call = cycles / REPS;
        double gf = flops16 * REPS / (cycles / cpu_ghz) / 1e9;
        double fmla_total = flops16 / 64.0; /* 32 elem × 2 = 64 FLOPs per FMLA */
        printf("FP16: %.0f cy/call  %.1f GF  (%.2f FMLA/cy, of %.0f total FMLAs)\n",
               cy_per_call, gf, fmla_total / cy_per_call, fmla_total);
    }

    /* ─── FP16 kernel WITHOUT memset (avoid memset overhead) ─── */
    {
        memset(C16, 0, MR * NR16 * 4);
        for (int i = 0; i < 100; i++)
            micro_kernel_fp16_12x2_swp(A16, B16, C16, (int64_t)Kc, 0, ldc16);
        uint64_t t0 = rdtsc();
        for (int r = 0; r < REPS; r++)
            micro_kernel_fp16_12x2_swp(A16, B16, C16, (int64_t)Kc, 0, ldc16);
        uint64_t t1 = rdtsc();
        double cycles = (double)(t1 - t0) * (cpu_ghz / timer_ghz);
        double cy_per_call = cycles / REPS;
        double gf = flops16 * REPS / (cycles / cpu_ghz) / 1e9;
        double fmla_total = flops16 / 64.0;
        printf("FP16 (no memset): %.0f cy/call  %.1f GF  (%.2f FMLA/cy)\n",
               cy_per_call, gf, fmla_total / cy_per_call);
    }

    /* ─── FP32 kernel WITHOUT memset ─── */
    {
        memset(C32, 0, MR * NR32 * 4);
        for (int i = 0; i < 100; i++)
            micro_kernel_fp32_12x2_swp(A32, B32, C32, (int64_t)Kc, 0, ldc32);
        uint64_t t0 = rdtsc();
        for (int r = 0; r < REPS; r++)
            micro_kernel_fp32_12x2_swp(A32, B32, C32, (int64_t)Kc, 0, ldc32);
        uint64_t t1 = rdtsc();
        double cycles = (double)(t1 - t0) * (cpu_ghz / timer_ghz);
        double cy_per_call = cycles / REPS;
        double gf = flops32 * REPS / (cycles / cpu_ghz) / 1e9;
        double fmla_total = flops32 / 32.0;
        printf("FP32 (no memset): %.0f cy/call  %.1f GF  (%.2f FMLA/cy)\n",
               cy_per_call, gf, fmla_total / cy_per_call);
    }

    /* ─── Vary Kc to see scaling ─── */
    printf("\n=== Kc sweep (no memset, single tile) ===\n");
    int Kcs[] = {16, 32, 64, 128, 256};
    for (int ki = 0; ki < 5; ki++) {
        int kc = Kcs[ki];
        double fl32 = 2.0 * MR * kc * NR32;
        double fl16 = 2.0 * MR * kc * NR16;
        int reps = 50000;

        /* fp32 */
        memset(C32, 0, MR * NR32 * 4);
        for (int i = 0; i < 100; i++)
            micro_kernel_fp32_12x2_swp(A32, B32, C32, (int64_t)kc, 0, ldc32);
        uint64_t t0 = rdtsc();
        for (int r = 0; r < reps; r++)
            micro_kernel_fp32_12x2_swp(A32, B32, C32, (int64_t)kc, 0, ldc32);
        uint64_t t1 = rdtsc();
        double cy32 = (double)(t1 - t0) * (cpu_ghz / timer_ghz);
        double gf32 = fl32 * reps / (cy32 / cpu_ghz) / 1e9;
        double fmla_cy32 = (fl32 / 32.0) / (cy32 / reps);

        /* fp16 */
        memset(C16, 0, MR * NR16 * 4);
        for (int i = 0; i < 100; i++)
            micro_kernel_fp16_12x2_swp(A16, B16, C16, (int64_t)kc, 0, ldc16);
        t0 = rdtsc();
        for (int r = 0; r < reps; r++)
            micro_kernel_fp16_12x2_swp(A16, B16, C16, (int64_t)kc, 0, ldc16);
        t1 = rdtsc();
        double cy16 = (double)(t1 - t0) * (cpu_ghz / timer_ghz);
        double gf16 = fl16 * reps / (cy16 / cpu_ghz) / 1e9;
        double fmla_cy16 = (fl16 / 64.0) / (cy16 / reps);

        printf("Kc=%3d  fp32: %6.1f GF (%4.2f FMLA/cy %5.0f cy)  fp16: %6.1f GF (%4.2f FMLA/cy %5.0f cy)  ratio=%.2f\n",
               kc, gf32, fmla_cy32, cy32/reps, gf16, fmla_cy16, cy16/reps, gf16/gf32);
    }

    free(A32); free(B32); free(C32);
    free(A16); free(B16); free(C16);
    return 0;
}
