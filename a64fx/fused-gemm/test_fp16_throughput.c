#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <arm_sve.h>

extern void bench_fmla_fp32(uint64_t iters);
extern void bench_fmla_fp16(uint64_t iters);
extern void bench_fmla_fp16_withld(uint64_t iters, void *dummy);
extern void bench_fmla_fp32_withld(uint64_t iters, void *dummy);

static inline uint64_t rdtsc(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

static inline uint64_t rdfreq(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
}

int main(void) {
    uint64_t freq = rdfreq();
    double cpu_ghz = (double)freq / 1e9;
    printf("Timer freq: %lu (%.3f GHz)\n\n", freq, cpu_ghz);

    // Dummy buffer for ld1rh/ld1rw tests (stays in L1)
    float dummy[64] __attribute__((aligned(256)));
    for (int i = 0; i < 64; i++) dummy[i] = 1.0f;

    uint64_t ITERS = 10000000;

    // ─── Test 1: Pure fp32 FMLA ───
    {
        bench_fmla_fp32(100); // warmup
        uint64_t t0 = rdtsc();
        bench_fmla_fp32(ITERS);
        uint64_t t1 = rdtsc();
        double cycles = (double)(t1 - t0) * (2.0 / cpu_ghz); // scale to 2GHz core clock
        double fmla_per_iter = 24.0;
        double total_fmla = fmla_per_iter * ITERS;
        double fmla_per_cy = total_fmla / cycles;
        double flops_per_cy = fmla_per_cy * 16 * 2; // 16 fp32 elements * 2 (mul+add)
        printf("FP32 pure FMLA:  %.1f M iters in %.0f cycles => %.2f FMLA/cy  (%.0f FLOP/cy = %.1f GF)\n",
               ITERS/1e6, cycles, fmla_per_cy, flops_per_cy, flops_per_cy * 2.0);
    }

    // ─── Test 2: Pure fp16 FMLA ───
    {
        bench_fmla_fp16(100);
        uint64_t t0 = rdtsc();
        bench_fmla_fp16(ITERS);
        uint64_t t1 = rdtsc();
        double cycles = (double)(t1 - t0) * (2.0 / cpu_ghz);
        double fmla_per_iter = 24.0;
        double total_fmla = fmla_per_iter * ITERS;
        double fmla_per_cy = total_fmla / cycles;
        double flops_per_cy = fmla_per_cy * 32 * 2; // 32 fp16 elements * 2
        printf("FP16 pure FMLA:  %.1f M iters in %.0f cycles => %.2f FMLA/cy  (%.0f FLOP/cy = %.1f GF)\n",
               ITERS/1e6, cycles, fmla_per_cy, flops_per_cy, flops_per_cy * 2.0);
    }

    // ─── Test 3: fp32 FMLA + ld1rw ───
    {
        bench_fmla_fp32_withld(100, dummy);
        uint64_t t0 = rdtsc();
        bench_fmla_fp32_withld(ITERS, dummy);
        uint64_t t1 = rdtsc();
        double cycles = (double)(t1 - t0) * (2.0 / cpu_ghz);
        double fmla_per_iter = 24.0;
        double total_fmla = fmla_per_iter * ITERS;
        double fmla_per_cy = total_fmla / cycles;
        double flops_per_cy = fmla_per_cy * 16 * 2;
        printf("FP32 FMLA+ld1rw: %.1f M iters in %.0f cycles => %.2f FMLA/cy  (%.0f FLOP/cy = %.1f GF)\n",
               ITERS/1e6, cycles, fmla_per_cy, flops_per_cy, flops_per_cy * 2.0);
    }

    // ─── Test 4: fp16 FMLA + ld1rh ───
    {
        bench_fmla_fp16_withld(100, dummy);
        uint64_t t0 = rdtsc();
        bench_fmla_fp16_withld(ITERS, dummy);
        uint64_t t1 = rdtsc();
        double cycles = (double)(t1 - t0) * (2.0 / cpu_ghz);
        double fmla_per_iter = 24.0;
        double total_fmla = fmla_per_iter * ITERS;
        double fmla_per_cy = total_fmla / cycles;
        double flops_per_cy = fmla_per_cy * 32 * 2;
        printf("FP16 FMLA+ld1rh: %.1f M iters in %.0f cycles => %.2f FMLA/cy  (%.0f FLOP/cy = %.1f GF)\n",
               ITERS/1e6, cycles, fmla_per_cy, flops_per_cy, flops_per_cy * 2.0);
    }

    return 0;
}
