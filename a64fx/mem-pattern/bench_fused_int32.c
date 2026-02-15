// bench_fused_int32.c
// Fused Q@K^T -> S (INT32) -> P@V with proper tiling

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

static inline uint64_t rdtsc(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// Baseline: Sequential phases
void fused_baseline(
    const int8_t* Q,      // [4, 256]
    const int8_t* K_int,  // [64, 64, 4] interleaved
    int32_t* S,           // [4, 64]
    const int8_t* V_t,    // [16, 256, 4] interleaved
    int32_t* O            // [4, 256]
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // ============================================================
        // Phase 1: Q@K^T -> S[4,64] INT32
        // ============================================================
        "mov x0, %[q]\n\t"
        "add x1, x0, #256\n\t"
        "add x2, x0, #512\n\t"
        "add x3, x0, #768\n\t"
        "mov x4, %[k]\n\t"

        "dup z24.s, #0\n\t"
        "dup z25.s, #0\n\t"
        "dup z26.s, #0\n\t"
        "dup z27.s, #0\n\t"

        "mov x5, #16\n\t"
        "1:\n\t"
        // K loads
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"

        "ld1b {z8.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z9.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x4, #7, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #4]\n\t"
        "ld1rw {z13.s}, p1/z, [x1, #4]\n\t"
        "ld1rw {z14.s}, p1/z, [x2, #4]\n\t"
        "ld1rw {z15.s}, p1/z, [x3, #4]\n\t"

        "add x4, x4, #512\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0, #8]\n\t"
        "ld1rw {z5.s}, p1/z, [x1, #8]\n\t"
        "ld1rw {z6.s}, p1/z, [x2, #8]\n\t"
        "ld1rw {z7.s}, p1/z, [x3, #8]\n\t"

        "ld1b {z8.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z9.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x4, #7, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #12]\n\t"
        "ld1rw {z13.s}, p1/z, [x1, #12]\n\t"
        "ld1rw {z14.s}, p1/z, [x2, #12]\n\t"
        "ld1rw {z15.s}, p1/z, [x3, #12]\n\t"

        "add x4, x4, #512\n\t"
        "add x0, x0, #16\n\t"
        "add x1, x1, #16\n\t"
        "add x2, x2, #16\n\t"
        "add x3, x3, #16\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Store S[4,64] = 4 rows x 64 int32 = 4 x 256 bytes
        // But we only have 1 vector per row in accumulator, so store 16 int32 per row
        "mov x6, %[s]\n\t"
        "st1w {z24.s}, p1, [x6]\n\t"
        "add x6, x6, #64\n\t"
        "st1w {z25.s}, p1, [x6]\n\t"
        "add x6, x6, #64\n\t"
        "st1w {z26.s}, p1, [x6]\n\t"
        "add x6, x6, #64\n\t"
        "st1w {z27.s}, p1, [x6]\n\t"

        // ============================================================
        // Phase 2: P@V -> O[4,256] INT32
        // ============================================================
        "mov x7, %[v]\n\t"
        "mov x8, %[o]\n\t"
        "mov x9, #4\n\t"      // 4 D-tiles

        "2:\n\t"
        // P row pointers
        "mov x0, %[s]\n\t"
        "add x1, x0, #64\n\t"
        "add x2, x0, #128\n\t"
        "add x3, x0, #192\n\t"

        "dup z24.s, #0\n\t"
        "dup z25.s, #0\n\t"
        "dup z26.s, #0\n\t"
        "dup z27.s, #0\n\t"

        "mov x10, x7\n\t"
        "mov x5, #4\n\t"

        "3:\n\t"
        "ld1b {z0.b}, p0/z, [x10]\n\t"
        "ld1b {z1.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"

        "add x10, x10, #1024\n\t"

        "ld1b {z8.b}, p0/z, [x10]\n\t"
        "ld1b {z9.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #4]\n\t"
        "ld1rw {z13.s}, p1/z, [x1, #4]\n\t"
        "ld1rw {z14.s}, p1/z, [x2, #4]\n\t"
        "ld1rw {z15.s}, p1/z, [x3, #4]\n\t"

        "add x10, x10, #1024\n\t"

        "ld1b {z0.b}, p0/z, [x10]\n\t"
        "ld1b {z1.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0, #8]\n\t"
        "ld1rw {z5.s}, p1/z, [x1, #8]\n\t"
        "ld1rw {z6.s}, p1/z, [x2, #8]\n\t"
        "ld1rw {z7.s}, p1/z, [x3, #8]\n\t"

        "add x10, x10, #1024\n\t"

        "ld1b {z8.b}, p0/z, [x10]\n\t"
        "ld1b {z9.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #12]\n\t"
        "ld1rw {z13.s}, p1/z, [x1, #12]\n\t"
        "ld1rw {z14.s}, p1/z, [x2, #12]\n\t"
        "ld1rw {z15.s}, p1/z, [x3, #12]\n\t"

        "add x10, x10, #1024\n\t"
        "add x0, x0, #16\n\t"
        "add x1, x1, #16\n\t"
        "add x2, x2, #16\n\t"
        "add x3, x3, #16\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 3b\n\t"

        // Store O tile
        "st1w {z24.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z25.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z26.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z27.s}, p1, [x8]\n\t"
        "sub x8, x8, #3008\n\t"

        "add x7, x7, #256\n\t"
        "subs x9, x9, #1\n\t"
        "b.ne 2b\n\t"
        :
        : [q] "r"(Q), [k] "r"(K_int), [s] "r"(S), [v] "r"(V_t), [o] "r"(O)
        : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z24", "z25", "z26", "z27",
          "p0", "p1", "memory"
    );
}

typedef void (*kernel_fn)(const int8_t*, const int8_t*, int32_t*, const int8_t*, int32_t*);

void bench(const char* name, kernel_fn fn,
           const int8_t* Q, const int8_t* K, int32_t* S, const int8_t* V, int32_t* O,
           int iters, int warmup) {
    for (int i = 0; i < warmup; i++) fn(Q, K, S, V, O);

    uint64_t start = rdtsc();
    for (int i = 0; i < iters; i++) fn(Q, K, S, V, O);
    uint64_t end = rdtsc();

    double ticks = (double)(end - start) / iters;
    double cycles = ticks * 20.0;

    // Memory traffic:
    // Phase 1: K=16KB + Q=1KB loaded, S=256B stored (only 1 vector per row)
    // Phase 2: V=16KB + S=256B loaded, O=1KB stored (4 tiles x 4 rows x 64B)
    size_t bytes_loaded = 16384 + 1024 + 16384 + 256;
    size_t bytes_stored = 256 + 1024;

    double load_bw = bytes_loaded / cycles;
    double store_bw = bytes_stored / cycles;
    double eff = 100.0 * load_bw / 128.0;

    printf("%-20s %8.1f cy  %6.1f ld  %5.1f st  %5.1f%%\n",
           name, cycles, load_bw, store_bw, eff);
}

int main() {
    printf("==============================================\n");
    printf("Fused INT32 Attention Memory Access\n");
    printf("==============================================\n");
    printf("Q@K^T: [4,256] @ [256,64] -> S[4,64] INT32\n");
    printf("P@V:   [4,64] @ [64,256] -> O[4,256] INT32\n\n");

    printf("Memory traffic (simplified - 1 tile per row):\n");
    printf("  Load:  K(16KB) + Q(1KB) + V(16KB) + S(256B) = 33.25KB\n");
    printf("  Store: S(256B) + O(1KB) = 1.25KB\n\n");

    double theo_load = 34048.0 / 128.0;
    printf("Theoretical min: %.1f cycles (load-bound)\n\n", theo_load);

    int8_t* Q = (int8_t*)aligned_alloc(256, 1024);
    int8_t* K = (int8_t*)aligned_alloc(256, 16384);
    int32_t* S = (int32_t*)aligned_alloc(256, 1024);
    int8_t* V = (int8_t*)aligned_alloc(256, 16384);
    int32_t* O = (int32_t*)aligned_alloc(256, 4096);

    memset(Q, 1, 1024);
    memset(K, 2, 16384);
    memset(S, 0, 1024);
    memset(V, 3, 16384);
    memset(O, 0, 4096);

    printf("%-20s %8s      %6s      %5s      %s\n", "Kernel", "Cycles", "Ld B/cy", "St B/cy", "Eff");
    printf("-------------------- --------      ------      -----      -----\n");

    bench("baseline", fused_baseline, Q, K, S, V, O, 10000, 1000);

    free(Q); free(K); free(S); free(V); free(O);
    return 0;
}
