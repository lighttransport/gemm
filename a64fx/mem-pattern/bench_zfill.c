// bench_zfill.c
// Test DC ZVA (zero cache line) to avoid read-for-ownership on stores

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

// Baseline: Store without zfill
// S is 4 rows x 16 int32 = 256 bytes = 1 cache line
void store_baseline(int32_t* S) {
    __asm__ volatile(
        "ptrue p1.s\n\t"
        "dup z0.s, #1\n\t"
        "dup z1.s, #2\n\t"
        "dup z2.s, #3\n\t"
        "dup z3.s, #4\n\t"

        // Store 4 vectors = 256 bytes = 1 cache line
        "st1w {z0.s}, p1, [%[s]]\n\t"
        "st1w {z1.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z2.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z3.s}, p1, [%[s], #3, mul vl]\n\t"
        :
        : [s] "r"(S)
        : "z0", "z1", "z2", "z3", "p1", "memory"
    );
}

// With DC ZVA before stores
void store_zfill(int32_t* S) {
    __asm__ volatile(
        "ptrue p1.s\n\t"
        "dup z0.s, #1\n\t"
        "dup z1.s, #2\n\t"
        "dup z2.s, #3\n\t"
        "dup z3.s, #4\n\t"

        // Zero the cache line first (256 bytes)
        "dc zva, %[s]\n\t"

        // Store 4 vectors = 256 bytes = 1 cache line
        "st1w {z0.s}, p1, [%[s]]\n\t"
        "st1w {z1.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z2.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z3.s}, p1, [%[s], #3, mul vl]\n\t"
        :
        : [s] "r"(S)
        : "z0", "z1", "z2", "z3", "p1", "memory"
    );
}

// Multiple cache lines without zfill
void store_multi_baseline(int32_t* S, int ncachelines) {
    __asm__ volatile(
        "ptrue p1.s\n\t"
        "dup z0.s, #1\n\t"
        "dup z1.s, #2\n\t"
        "dup z2.s, #3\n\t"
        "dup z3.s, #4\n\t"

        "mov x0, %[s]\n\t"
        "mov x1, %[n]\n\t"

        "1:\n\t"
        "st1w {z0.s}, p1, [x0]\n\t"
        "st1w {z1.s}, p1, [x0, #1, mul vl]\n\t"
        "st1w {z2.s}, p1, [x0, #2, mul vl]\n\t"
        "st1w {z3.s}, p1, [x0, #3, mul vl]\n\t"
        "add x0, x0, #256\n\t"
        "subs x1, x1, #1\n\t"
        "b.ne 1b\n\t"
        :
        : [s] "r"(S), [n] "r"((uint64_t)ncachelines)
        : "x0", "x1", "z0", "z1", "z2", "z3", "p1", "memory"
    );
}

// Multiple cache lines with zfill
void store_multi_zfill(int32_t* S, int ncachelines) {
    __asm__ volatile(
        "ptrue p1.s\n\t"
        "dup z0.s, #1\n\t"
        "dup z1.s, #2\n\t"
        "dup z2.s, #3\n\t"
        "dup z3.s, #4\n\t"

        "mov x0, %[s]\n\t"
        "mov x1, %[n]\n\t"

        "1:\n\t"
        "dc zva, x0\n\t"
        "st1w {z0.s}, p1, [x0]\n\t"
        "st1w {z1.s}, p1, [x0, #1, mul vl]\n\t"
        "st1w {z2.s}, p1, [x0, #2, mul vl]\n\t"
        "st1w {z3.s}, p1, [x0, #3, mul vl]\n\t"
        "add x0, x0, #256\n\t"
        "subs x1, x1, #1\n\t"
        "b.ne 1b\n\t"
        :
        : [s] "r"(S), [n] "r"((uint64_t)ncachelines)
        : "x0", "x1", "z0", "z1", "z2", "z3", "p1", "memory"
    );
}

// Fused baseline: Q@K^T phase 1 with normal stores
void fused_qkt_baseline(
    const int8_t* Q,
    const int8_t* K,
    int32_t* S
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

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

        // Store S: 4 vectors = 256 bytes = 1 cache line
        "st1w {z24.s}, p1, [%[s]]\n\t"
        "st1w {z25.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z26.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z27.s}, p1, [%[s], #3, mul vl]\n\t"
        :
        : [q] "r"(Q), [k] "r"(K), [s] "r"(S)
        : "x0", "x1", "x2", "x3", "x4", "x5",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z24", "z25", "z26", "z27",
          "p0", "p1", "memory"
    );
}

// Fused with zfill before S store
void fused_qkt_zfill(
    const int8_t* Q,
    const int8_t* K,
    int32_t* S
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

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

        // DC ZVA before stores (256-byte cache line)
        "dc zva, %[s]\n\t"

        // Store S: 4 vectors = 256 bytes = 1 cache line
        "st1w {z24.s}, p1, [%[s]]\n\t"
        "st1w {z25.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z26.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z27.s}, p1, [%[s], #3, mul vl]\n\t"
        :
        : [q] "r"(Q), [k] "r"(K), [s] "r"(S)
        : "x0", "x1", "x2", "x3", "x4", "x5",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z24", "z25", "z26", "z27",
          "p0", "p1", "memory"
    );
}

// Full fused kernel with zfill for both S and O
void fused_full_zfill(
    const int8_t* Q,
    const int8_t* K,
    int32_t* S,
    const int8_t* V,
    int32_t* O
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Phase 1: Q@K^T
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

        // DC ZVA + Store S
        "dc zva, %[s]\n\t"
        "st1w {z24.s}, p1, [%[s]]\n\t"
        "st1w {z25.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z26.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z27.s}, p1, [%[s], #3, mul vl]\n\t"

        // Phase 2: P@V (4 D-tiles)
        "mov x7, %[v]\n\t"
        "mov x8, %[o]\n\t"
        "mov x9, #4\n\t"

        "2:\n\t"
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

        // DC ZVA for O (each row = 256B = 1 cache line)
        "mov x11, x8\n\t"
        "dc zva, x11\n\t"
        "add x11, x11, #1024\n\t"
        "dc zva, x11\n\t"
        "add x11, x11, #1024\n\t"
        "dc zva, x11\n\t"
        "add x11, x11, #1024\n\t"
        "dc zva, x11\n\t"

        // Store O
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
        : [q] "r"(Q), [k] "r"(K), [s] "r"(S), [v] "r"(V), [o] "r"(O)
        : "x0", "x1", "x2", "x3", "x4", "x5", "x7", "x8", "x9", "x10", "x11",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z24", "z25", "z26", "z27",
          "p0", "p1", "memory"
    );
}

// Full fused kernel without zfill
void fused_full_baseline(
    const int8_t* Q,
    const int8_t* K,
    int32_t* S,
    const int8_t* V,
    int32_t* O
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Phase 1: Q@K^T
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

        // Store S (no zfill)
        "st1w {z24.s}, p1, [%[s]]\n\t"
        "st1w {z25.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z26.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z27.s}, p1, [%[s], #3, mul vl]\n\t"

        // Phase 2: P@V
        "mov x7, %[v]\n\t"
        "mov x8, %[o]\n\t"
        "mov x9, #4\n\t"

        "2:\n\t"
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

        // Store O (no zfill)
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
        : [q] "r"(Q), [k] "r"(K), [s] "r"(S), [v] "r"(V), [o] "r"(O)
        : "x0", "x1", "x2", "x3", "x4", "x5", "x7", "x8", "x9", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z24", "z25", "z26", "z27",
          "p0", "p1", "memory"
    );
}

int main() {
    printf("==============================================\n");
    printf("DC ZVA (Zero Cache Line) Store Optimization\n");
    printf("==============================================\n");
    printf("A64FX cache line: 256 bytes\n");
    printf("4 x st1w (64B) = 1 cache line\n\n");

    // Allocate buffers
    int32_t* S_warm = (int32_t*)aligned_alloc(256, 4096);

    int8_t* Q = (int8_t*)aligned_alloc(256, 1024);
    int8_t* K = (int8_t*)aligned_alloc(256, 16384);
    int8_t* V = (int8_t*)aligned_alloc(256, 16384);
    int32_t* O = (int32_t*)aligned_alloc(256, 4096);

    memset(Q, 1, 1024);
    memset(K, 2, 16384);
    memset(V, 3, 16384);

    int iters = 10000;
    int warmup = 1000;
    uint64_t start, end;
    double ticks, cycles;

    printf("=== Pure Store Tests ===\n");
    printf("%-25s %8s %8s\n", "Test", "Cycles", "B/cy");
    printf("------------------------------------------------\n");

    // Warm cache store baseline
    for (int i = 0; i < warmup; i++) store_baseline(S_warm);
    start = rdtsc();
    for (int i = 0; i < iters; i++) store_baseline(S_warm);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f %8.1f\n", "warm_baseline", cycles, 256.0 / cycles);

    // Warm cache store zfill
    for (int i = 0; i < warmup; i++) store_zfill(S_warm);
    start = rdtsc();
    for (int i = 0; i < iters; i++) store_zfill(S_warm);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f %8.1f\n", "warm_zfill", cycles, 256.0 / cycles);

    // Multi-cacheline tests (16 cache lines = 4KB)
    int ncl = 16;
    printf("\n=== Multi-cacheline (16 x 256B = 4KB) ===\n");

    for (int i = 0; i < warmup; i++) store_multi_baseline(S_warm, ncl);
    start = rdtsc();
    for (int i = 0; i < iters; i++) store_multi_baseline(S_warm, ncl);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f %8.1f\n", "multi_baseline", cycles, (ncl * 256.0) / cycles);

    for (int i = 0; i < warmup; i++) store_multi_zfill(S_warm, ncl);
    start = rdtsc();
    for (int i = 0; i < iters; i++) store_multi_zfill(S_warm, ncl);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f %8.1f\n", "multi_zfill", cycles, (ncl * 256.0) / cycles);

    printf("\n=== Q@K^T Phase Only ===\n");
    printf("Load: K(16KB) + Q(1KB) = 17KB, Store: S(256B)\n");
    printf("Theoretical: 136 cy (load-bound)\n\n");

    for (int i = 0; i < warmup; i++) fused_qkt_baseline(Q, K, S_warm);
    start = rdtsc();
    for (int i = 0; i < iters; i++) fused_qkt_baseline(Q, K, S_warm);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f cy  %5.1f ld B/cy\n", "qkt_baseline", cycles, 17408.0 / cycles);

    for (int i = 0; i < warmup; i++) fused_qkt_zfill(Q, K, S_warm);
    start = rdtsc();
    for (int i = 0; i < iters; i++) fused_qkt_zfill(Q, K, S_warm);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f cy  %5.1f ld B/cy\n", "qkt_zfill", cycles, 17408.0 / cycles);

    printf("\n=== Full Fused Attention ===\n");
    printf("Load: K(16KB) + Q(1KB) + V(16KB) + S(256B) = 33.25KB\n");
    printf("Store: S(256B) + O(1KB) = 1.25KB\n");
    printf("Theoretical: 266 cy (load-bound)\n\n");

    for (int i = 0; i < warmup; i++) fused_full_baseline(Q, K, S_warm, V, O);
    start = rdtsc();
    for (int i = 0; i < iters; i++) fused_full_baseline(Q, K, S_warm, V, O);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f cy  %5.1f ld B/cy  %.1f%%\n",
           "fused_baseline", cycles, 34048.0 / cycles, 100.0 * 266.0 / cycles);

    for (int i = 0; i < warmup; i++) fused_full_zfill(Q, K, S_warm, V, O);
    start = rdtsc();
    for (int i = 0; i < iters; i++) fused_full_zfill(Q, K, S_warm, V, O);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f cy  %5.1f ld B/cy  %.1f%%\n",
           "fused_zfill", cycles, 34048.0 / cycles, 100.0 * 266.0 / cycles);

    free(S_warm);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
