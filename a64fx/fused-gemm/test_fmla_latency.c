#include <stdio.h>
#include <stdint.h>

extern void bench_fp32_lat1(uint64_t);
extern void bench_fp16_lat1(uint64_t);
extern void bench_fp32_lat9(uint64_t);
extern void bench_fp16_lat9(uint64_t);
extern void bench_fp32_lat12(uint64_t);
extern void bench_fp16_lat12(uint64_t);
extern void bench_fp32_lat18(uint64_t);
extern void bench_fp16_lat18(uint64_t);

static inline uint64_t rdtsc(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
}
static inline uint64_t rdfreq(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v;
}

typedef void (*bench_fn)(uint64_t);

static void run(const char *label, bench_fn fn, int n_accum, int n_fmla, int elem, uint64_t iters) {
    double cpu_ghz = 2.0;
    double timer_ghz = (double)rdfreq() / 1e9;
    fn(100);
    uint64_t t0 = rdtsc();
    fn(iters);
    uint64_t t1 = rdtsc();
    double cycles = (double)(t1 - t0) * (cpu_ghz / timer_ghz);
    double total = (double)n_fmla * iters;
    double per_cy = total / cycles;
    // Distance between same-accum reuses in instructions
    // (n_accum FMLAs between reuses within body)
    double reuse_dist_insn = (double)n_accum;
    // At 2 FMLA/cy throughput, how many cycles between reuses
    double reuse_dist_cy = reuse_dist_insn / 2.0;
    // Inferred latency from 1-accum test
    double inferred_lat = (n_accum == 1) ? 1.0 / per_cy : reuse_dist_cy;

    printf("%-6s %2d accum (%2d fmla): %5.2f FMLA/cy  %5.0f FLOP/cy",
           label, n_accum, n_fmla, per_cy, per_cy * elem * 2);
    if (n_accum == 1)
        printf("  => LATENCY = %.1f cy\n", 1.0 / per_cy);
    else
        printf("  (reuse dist=%.0f insn)\n", reuse_dist_insn);
}

int main(void) {
    uint64_t ITERS = 5000000;
    printf("=== FMLA Latency/Throughput Sweep ===\n");
    printf("Tests: vary # independent accumulators to find latency\n");
    printf("Throughput limit: 2 FMLA/cy. Latency limit: 1/lat FMLA/cy with 1 accum\n\n");

    printf("--- 1 accumulator (pure latency chain) ---\n");
    run("fp32", bench_fp32_lat1,  1, 8,  16, ITERS);
    run("fp16", bench_fp16_lat1,  1, 8,  32, ITERS);

    printf("\n--- 9 accumulators (18 FMLAs, each used 2x) ---\n");
    run("fp32", bench_fp32_lat9,  9, 18, 16, ITERS);
    run("fp16", bench_fp16_lat9,  9, 18, 32, ITERS);

    printf("\n--- 12 accumulators (24 FMLAs, each used 2x) ---\n");
    run("fp32", bench_fp32_lat12, 12, 24, 16, ITERS);
    run("fp16", bench_fp16_lat12, 12, 24, 32, ITERS);

    printf("\n--- 18 accumulators (18 FMLAs, each used 1x) ---\n");
    run("fp32", bench_fp32_lat18, 18, 18, 16, ITERS);
    run("fp16", bench_fp16_lat18, 18, 18, 32, ITERS);

    return 0;
}
