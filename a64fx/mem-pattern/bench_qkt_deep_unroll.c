// bench_qkt_deep_unroll.c
// Deep unrolled Q@K^T - 8x unroll, interleaved K/Q loads

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

// Baseline: 4x unroll (16 loop iterations)
void qkt_unroll4(const int8_t* Q, const int8_t* K_int, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        "mov x5, %[q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x8, x5, #768\n\t"
        "mov x9, %[k]\n\t"
        "mov x10, #16\n\t"   // 64/4 = 16 iterations

        "1:\n\t"
        // Iter 0: K[0:3], Q[0]
        "ld1b {z0.b}, p0/z, [x9]\n\t"
        "ld1b {z1.b}, p0/z, [x9, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x9, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x9, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "ld1rw {z5.s}, p1/z, [x6]\n\t"
        "ld1rw {z6.s}, p1/z, [x7]\n\t"
        "ld1rw {z7.s}, p1/z, [x8]\n\t"

        // Iter 1: K[4:7], Q[1]
        "ld1b {z8.b}, p0/z, [x9, #4, mul vl]\n\t"
        "ld1b {z9.b}, p0/z, [x9, #5, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x9, #6, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x9, #7, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x5, #4]\n\t"
        "ld1rw {z13.s}, p1/z, [x6, #4]\n\t"
        "ld1rw {z14.s}, p1/z, [x7, #4]\n\t"
        "ld1rw {z15.s}, p1/z, [x8, #4]\n\t"

        // Iter 2: K[8:11], Q[2]
        "add x9, x9, #512\n\t"
        "ld1b {z0.b}, p0/z, [x9]\n\t"
        "ld1b {z1.b}, p0/z, [x9, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x9, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x9, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5, #8]\n\t"
        "ld1rw {z5.s}, p1/z, [x6, #8]\n\t"
        "ld1rw {z6.s}, p1/z, [x7, #8]\n\t"
        "ld1rw {z7.s}, p1/z, [x8, #8]\n\t"

        // Iter 3: K[12:15], Q[3]
        "ld1b {z8.b}, p0/z, [x9, #4, mul vl]\n\t"
        "ld1b {z9.b}, p0/z, [x9, #5, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x9, #6, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x9, #7, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x5, #12]\n\t"
        "ld1rw {z13.s}, p1/z, [x6, #12]\n\t"
        "ld1rw {z14.s}, p1/z, [x7, #12]\n\t"
        "ld1rw {z15.s}, p1/z, [x8, #12]\n\t"

        "add x9, x9, #512\n\t"
        "add x5, x5, #16\n\t"
        "add x6, x6, #16\n\t"
        "add x7, x7, #16\n\t"
        "add x8, x8, #16\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"
        :
        : [q] "r"(Q), [k] "r"(K_int), [s] "r"(S)
        : "x5", "x6", "x7", "x8", "x9", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "p0", "p1", "memory"
    );
}

// 8x unroll (8 loop iterations)
void qkt_unroll8(const int8_t* Q, const int8_t* K_int, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        "mov x5, %[q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x8, x5, #768\n\t"
        "mov x9, %[k]\n\t"
        "mov x10, #8\n\t"    // 64/8 = 8 iterations

        "1:\n\t"
        // Iters 0-1
        "ld1b {z0.b}, p0/z, [x9]\n\t"
        "ld1b {z1.b}, p0/z, [x9, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x9, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x9, #3, mul vl]\n\t"
        "ld1rw {z16.s}, p1/z, [x5]\n\t"
        "ld1rw {z17.s}, p1/z, [x6]\n\t"
        "ld1rw {z18.s}, p1/z, [x7]\n\t"
        "ld1rw {z19.s}, p1/z, [x8]\n\t"

        "ld1b {z4.b}, p0/z, [x9, #4, mul vl]\n\t"
        "ld1b {z5.b}, p0/z, [x9, #5, mul vl]\n\t"
        "ld1b {z6.b}, p0/z, [x9, #6, mul vl]\n\t"
        "ld1b {z7.b}, p0/z, [x9, #7, mul vl]\n\t"
        "ld1rw {z20.s}, p1/z, [x5, #4]\n\t"
        "ld1rw {z21.s}, p1/z, [x6, #4]\n\t"
        "ld1rw {z22.s}, p1/z, [x7, #4]\n\t"
        "ld1rw {z23.s}, p1/z, [x8, #4]\n\t"

        // Iters 2-3
        "add x9, x9, #512\n\t"
        "ld1b {z0.b}, p0/z, [x9]\n\t"
        "ld1b {z1.b}, p0/z, [x9, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x9, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x9, #3, mul vl]\n\t"
        "ld1rw {z24.s}, p1/z, [x5, #8]\n\t"
        "ld1rw {z25.s}, p1/z, [x6, #8]\n\t"
        "ld1rw {z26.s}, p1/z, [x7, #8]\n\t"
        "ld1rw {z27.s}, p1/z, [x8, #8]\n\t"

        "ld1b {z4.b}, p0/z, [x9, #4, mul vl]\n\t"
        "ld1b {z5.b}, p0/z, [x9, #5, mul vl]\n\t"
        "ld1b {z6.b}, p0/z, [x9, #6, mul vl]\n\t"
        "ld1b {z7.b}, p0/z, [x9, #7, mul vl]\n\t"
        "ld1rw {z28.s}, p1/z, [x5, #12]\n\t"
        "ld1rw {z29.s}, p1/z, [x6, #12]\n\t"
        "ld1rw {z30.s}, p1/z, [x7, #12]\n\t"
        "ld1rw {z31.s}, p1/z, [x8, #12]\n\t"

        // Iters 4-5
        "add x9, x9, #512\n\t"
        "ld1b {z0.b}, p0/z, [x9]\n\t"
        "ld1b {z1.b}, p0/z, [x9, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x9, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x9, #3, mul vl]\n\t"
        "ld1rw {z16.s}, p1/z, [x5, #16]\n\t"
        "ld1rw {z17.s}, p1/z, [x6, #16]\n\t"
        "ld1rw {z18.s}, p1/z, [x7, #16]\n\t"
        "ld1rw {z19.s}, p1/z, [x8, #16]\n\t"

        "ld1b {z4.b}, p0/z, [x9, #4, mul vl]\n\t"
        "ld1b {z5.b}, p0/z, [x9, #5, mul vl]\n\t"
        "ld1b {z6.b}, p0/z, [x9, #6, mul vl]\n\t"
        "ld1b {z7.b}, p0/z, [x9, #7, mul vl]\n\t"
        "ld1rw {z20.s}, p1/z, [x5, #20]\n\t"
        "ld1rw {z21.s}, p1/z, [x6, #20]\n\t"
        "ld1rw {z22.s}, p1/z, [x7, #20]\n\t"
        "ld1rw {z23.s}, p1/z, [x8, #20]\n\t"

        // Iters 6-7
        "add x9, x9, #512\n\t"
        "ld1b {z0.b}, p0/z, [x9]\n\t"
        "ld1b {z1.b}, p0/z, [x9, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x9, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x9, #3, mul vl]\n\t"
        "ld1rw {z24.s}, p1/z, [x5, #24]\n\t"
        "ld1rw {z25.s}, p1/z, [x6, #24]\n\t"
        "ld1rw {z26.s}, p1/z, [x7, #24]\n\t"
        "ld1rw {z27.s}, p1/z, [x8, #24]\n\t"

        "ld1b {z4.b}, p0/z, [x9, #4, mul vl]\n\t"
        "ld1b {z5.b}, p0/z, [x9, #5, mul vl]\n\t"
        "ld1b {z6.b}, p0/z, [x9, #6, mul vl]\n\t"
        "ld1b {z7.b}, p0/z, [x9, #7, mul vl]\n\t"
        "ld1rw {z28.s}, p1/z, [x5, #28]\n\t"
        "ld1rw {z29.s}, p1/z, [x6, #28]\n\t"
        "ld1rw {z30.s}, p1/z, [x7, #28]\n\t"
        "ld1rw {z31.s}, p1/z, [x8, #28]\n\t"

        "add x9, x9, #512\n\t"
        "add x5, x5, #32\n\t"
        "add x6, x6, #32\n\t"
        "add x7, x7, #32\n\t"
        "add x8, x8, #32\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"
        :
        : [q] "r"(Q), [k] "r"(K_int), [s] "r"(S)
        : "x5", "x6", "x7", "x8", "x9", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1", "memory"
    );
}

// Interleaved: K and Q loads mixed better
void qkt_interleaved(const int8_t* Q, const int8_t* K_int, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        "mov x5, %[q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x8, x5, #768\n\t"
        "mov x9, %[k]\n\t"
        "mov x10, #16\n\t"

        "1:\n\t"
        // Interleave K and Q loads to hide latency
        "ld1b {z0.b}, p0/z, [x9]\n\t"
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "ld1b {z1.b}, p0/z, [x9, #1, mul vl]\n\t"
        "ld1rw {z5.s}, p1/z, [x6]\n\t"
        "ld1b {z2.b}, p0/z, [x9, #2, mul vl]\n\t"
        "ld1rw {z6.s}, p1/z, [x7]\n\t"
        "ld1b {z3.b}, p0/z, [x9, #3, mul vl]\n\t"
        "ld1rw {z7.s}, p1/z, [x8]\n\t"

        "ld1b {z8.b}, p0/z, [x9, #4, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x5, #4]\n\t"
        "ld1b {z9.b}, p0/z, [x9, #5, mul vl]\n\t"
        "ld1rw {z13.s}, p1/z, [x6, #4]\n\t"
        "ld1b {z10.b}, p0/z, [x9, #6, mul vl]\n\t"
        "ld1rw {z14.s}, p1/z, [x7, #4]\n\t"
        "ld1b {z11.b}, p0/z, [x9, #7, mul vl]\n\t"
        "ld1rw {z15.s}, p1/z, [x8, #4]\n\t"

        "add x9, x9, #512\n\t"

        "ld1b {z0.b}, p0/z, [x9]\n\t"
        "ld1rw {z4.s}, p1/z, [x5, #8]\n\t"
        "ld1b {z1.b}, p0/z, [x9, #1, mul vl]\n\t"
        "ld1rw {z5.s}, p1/z, [x6, #8]\n\t"
        "ld1b {z2.b}, p0/z, [x9, #2, mul vl]\n\t"
        "ld1rw {z6.s}, p1/z, [x7, #8]\n\t"
        "ld1b {z3.b}, p0/z, [x9, #3, mul vl]\n\t"
        "ld1rw {z7.s}, p1/z, [x8, #8]\n\t"

        "ld1b {z8.b}, p0/z, [x9, #4, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x5, #12]\n\t"
        "ld1b {z9.b}, p0/z, [x9, #5, mul vl]\n\t"
        "ld1rw {z13.s}, p1/z, [x6, #12]\n\t"
        "ld1b {z10.b}, p0/z, [x9, #6, mul vl]\n\t"
        "ld1rw {z14.s}, p1/z, [x7, #12]\n\t"
        "ld1b {z11.b}, p0/z, [x9, #7, mul vl]\n\t"
        "ld1rw {z15.s}, p1/z, [x8, #12]\n\t"

        "add x9, x9, #512\n\t"
        "add x5, x5, #16\n\t"
        "add x6, x6, #16\n\t"
        "add x7, x7, #16\n\t"
        "add x8, x8, #16\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"
        :
        : [q] "r"(Q), [k] "r"(K_int), [s] "r"(S)
        : "x5", "x6", "x7", "x8", "x9", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
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
    double bw = 17408.0 / cycles;
    double eff = 100.0 * bw / 128.0;

    printf("%-25s %8.1f cy  %6.1f B/cy  %5.1f%%\n", name, cycles, bw, eff);
}

int main() {
    printf("==============================================\n");
    printf("Q@K^T Deep Unroll Comparison\n");
    printf("==============================================\n");
    printf("Theoretical min: 136 cycles (128 B/cy)\n\n");

    int8_t* Q = (int8_t*)aligned_alloc(256, 1024);
    int8_t* K = (int8_t*)aligned_alloc(256, 16384);
    int32_t* S = (int32_t*)aligned_alloc(256, 256);
    memset(Q, 1, 1024);
    memset(K, 2, 16384);

    printf("%-25s %8s      %6s       %s\n", "Kernel", "Cycles", "BW", "Eff");
    printf("------------------------- --------      ------       -----\n");

    bench("unroll4 (16 iter)", qkt_unroll4, Q, K, S, 10000, 1000);
    bench("unroll8 (8 iter)", qkt_unroll8, Q, K, S, 10000, 1000);
    bench("interleaved K/Q", qkt_interleaved, Q, K, S, 10000, 1000);

    free(Q); free(K); free(S);
    return 0;
}
