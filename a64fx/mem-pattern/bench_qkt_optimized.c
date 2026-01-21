// bench_qkt_optimized.c
// Optimized Q@K^T with ld1rw unroll4 + immediate offset

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

// Original Q@K^T kernel (baseline)
void qkt_original(const int8_t* Q, const int8_t* K_int, int32_t* S) {
    svbool_t pg = svptrue_b8();
    svbool_t pg32 = svptrue_b32();

    const int8_t* k_ptr = K_int;
    const int8_t* q_ptr = Q;

    svint32_t acc0 = svdup_s32(0);

    for (int d = 0; d < 64; d++) {
        // K loads
        svint8_t k0 = svld1_s8(pg, k_ptr);
        svint8_t k1 = svld1_s8(pg, k_ptr + 64);
        svint8_t k2 = svld1_s8(pg, k_ptr + 128);
        svint8_t k3 = svld1_s8(pg, k_ptr + 192);
        k_ptr += 256;

        // Q loads (broadcast) - inline asm
        svint32_t q0, q1, q2, q3;
        __asm__ volatile(
            "ld1rw {%0.s}, %4/z, [%5]\n\t"
            "ld1rw {%1.s}, %4/z, [%6]\n\t"
            "ld1rw {%2.s}, %4/z, [%7]\n\t"
            "ld1rw {%3.s}, %4/z, [%8]"
            : "=w"(q0), "=w"(q1), "=w"(q2), "=w"(q3)
            : "Upl"(pg32), "r"(q_ptr), "r"(q_ptr+256), "r"(q_ptr+512), "r"(q_ptr+768)
            : "memory"
        );
        q_ptr += 4;

        // Dummy use
        __asm__ volatile("" :: "w"(k0), "w"(k1), "w"(k2), "w"(k3));
        __asm__ volatile("" :: "w"(q0), "w"(q1), "w"(q2), "w"(q3));
    }

    svst1_s32(pg32, S, acc0);
}

// Optimized Q@K^T with unrolled ld1rw + immediate offset (ASM)
void qkt_optimized(const int8_t* Q, const int8_t* K_int, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Set up Q row pointers
        "mov x5, %[q]\n\t"            // Q row 0
        "add x6, x5, #256\n\t"        // Q row 1
        "add x7, x5, #512\n\t"        // Q row 2
        "add x8, x5, #768\n\t"        // Q row 3

        // K pointer
        "mov x9, %[k]\n\t"

        // Loop counter: 64/4 = 16 iterations
        "mov x10, #16\n\t"

        // Zero accumulators
        "dup z16.s, #0\n\t"

        "1:\n\t"
        // ===== Unrolled iteration 0 =====
        // K loads (4x64B = 256B)
        "ld1b {z0.b}, p0/z, [x9]\n\t"
        "ld1b {z1.b}, p0/z, [x9, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x9, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x9, #3, mul vl]\n\t"
        // Q loads with immediate offset
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "ld1rw {z5.s}, p1/z, [x6]\n\t"
        "ld1rw {z6.s}, p1/z, [x7]\n\t"
        "ld1rw {z7.s}, p1/z, [x8]\n\t"

        // ===== Unrolled iteration 1 =====
        "ld1b {z8.b}, p0/z, [x9, #4, mul vl]\n\t"
        "ld1b {z9.b}, p0/z, [x9, #5, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x9, #6, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x9, #7, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x5, #4]\n\t"
        "ld1rw {z13.s}, p1/z, [x6, #4]\n\t"
        "ld1rw {z14.s}, p1/z, [x7, #4]\n\t"
        "ld1rw {z15.s}, p1/z, [x8, #4]\n\t"

        // ===== Unrolled iteration 2 =====
        "add x9, x9, #512\n\t"        // K += 8 * 64B = 512
        "ld1b {z0.b}, p0/z, [x9]\n\t"
        "ld1b {z1.b}, p0/z, [x9, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x9, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x9, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5, #8]\n\t"
        "ld1rw {z5.s}, p1/z, [x6, #8]\n\t"
        "ld1rw {z6.s}, p1/z, [x7, #8]\n\t"
        "ld1rw {z7.s}, p1/z, [x8, #8]\n\t"

        // ===== Unrolled iteration 3 =====
        "ld1b {z8.b}, p0/z, [x9, #4, mul vl]\n\t"
        "ld1b {z9.b}, p0/z, [x9, #5, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x9, #6, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x9, #7, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x5, #12]\n\t"
        "ld1rw {z13.s}, p1/z, [x6, #12]\n\t"
        "ld1rw {z14.s}, p1/z, [x7, #12]\n\t"
        "ld1rw {z15.s}, p1/z, [x8, #12]\n\t"

        // Advance pointers
        "add x9, x9, #512\n\t"        // K += 8 * 64B
        "add x5, x5, #16\n\t"         // Q += 4 * 4B
        "add x6, x6, #16\n\t"
        "add x7, x7, #16\n\t"
        "add x8, x8, #16\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Store result
        "st1w {z16.s}, p1, [%[s]]\n\t"
        :
        : [q] "r"(Q), [k] "r"(K_int), [s] "r"(S)
        : "x5", "x6", "x7", "x8", "x9", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16",
          "p0", "p1", "memory"
    );
}

typedef void (*kernel_fn)(const int8_t*, const int8_t*, int32_t*);

void bench(const char* name, kernel_fn fn, const int8_t* Q, const int8_t* K, int32_t* S, int iters, int warmup) {
    for (int i = 0; i < warmup; i++) fn(Q, K, S);

    uint64_t start = rdtsc();
    for (int i = 0; i < iters; i++) fn(Q, K, S);
    uint64_t end = rdtsc();

    double ticks = (double)(end - start) / iters;
    double cycles = ticks * 20.0;

    // Memory traffic: K=16KB, Q=1KB loaded
    size_t bytes = 16384 + 1024;
    double bw = bytes / cycles;
    double eff = 100.0 * bw / 128.0;

    printf("%-30s %8.1f cycles  %6.1f B/cy  %5.1f%%\n", name, cycles, bw, eff);
}

int main(int argc, char** argv) {
    int iters = 10000;
    int warmup = 1000;

    printf("==============================================\n");
    printf("Q@K^T Optimized Kernel Comparison\n");
    printf("==============================================\n");
    printf("Q[4,256] @ K[256,64] -> S[4,64]\n");
    printf("Memory: K=16KB + Q=1KB = 17KB (fits in L1)\n");
    printf("Theoretical min: %.1f cycles (128 B/cy)\n\n", 17408.0 / 128.0);

    int8_t* Q = (int8_t*)aligned_alloc(256, 1024);
    int8_t* K = (int8_t*)aligned_alloc(256, 16384);
    int32_t* S = (int32_t*)aligned_alloc(256, 256);

    memset(Q, 1, 1024);
    memset(K, 2, 16384);

    printf("%-30s %8s        %6s       %s\n", "Kernel", "Cycles", "BW", "Eff");
    printf("------------------------------ --------        ------       -----\n");

    bench("original (C)", qkt_original, Q, K, S, iters, warmup);
    bench("optimized (ASM unroll4)", qkt_optimized, Q, K, S, iters, warmup);

    printf("\n");
    printf("Improvement target: 2-3x faster\n");

    free(Q);
    free(K);
    free(S);
    return 0;
}
