// bench_ld1rw_opt.c
// Test optimized Q broadcast loads

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

#define USE_RESULT(v) __asm__ volatile("" :: "w"(v) : "memory")

// Original: ld1rw with inline asm, pointer arithmetic each iteration
static inline svint32_t ld1rw_ptr(svbool_t pg, const int32_t* ptr) {
    svint32_t result;
    __asm__ volatile(
        "ld1rw {%0.s}, %1/z, [%2]"
        : "=w"(result)
        : "Upl"(pg), "r"(ptr)
        : "memory"
    );
    return result;
}

// Test 1: Original - pointer arithmetic per load (C version)
void test_original(const int8_t* Q, int n_iters) {
    svbool_t pg = svptrue_b32();
    const int8_t* q = Q;

    for (int i = 0; i < n_iters; i++) {
        svint32_t q0 = ld1rw_ptr(pg, (const int32_t*)q);
        svint32_t q1 = ld1rw_ptr(pg, (const int32_t*)(q + 256));
        svint32_t q2 = ld1rw_ptr(pg, (const int32_t*)(q + 512));
        svint32_t q3 = ld1rw_ptr(pg, (const int32_t*)(q + 768));
        q += 4;
        USE_RESULT(q0); USE_RESULT(q1); USE_RESULT(q2); USE_RESULT(q3);
    }
}

// Test 2: Pre-computed row bases with post-increment (ASM)
void test_precomputed_postinc(const int8_t* Q, int n_iters) {
    __asm__ volatile(
        "ptrue p0.s\n\t"
        "mov x9, %[q]\n\t"            // Row 0 base
        "add x10, x9, #256\n\t"       // Row 1 base
        "add x11, x9, #512\n\t"       // Row 2 base
        "add x12, x9, #768\n\t"       // Row 3 base
        "mov x13, %[n]\n\t"
        "1:\n\t"
        "ld1rw {z0.s}, p0/z, [x9]\n\t"
        "add x9, x9, #4\n\t"
        "ld1rw {z1.s}, p0/z, [x10]\n\t"
        "add x10, x10, #4\n\t"
        "ld1rw {z2.s}, p0/z, [x11]\n\t"
        "add x11, x11, #4\n\t"
        "ld1rw {z3.s}, p0/z, [x12]\n\t"
        "add x12, x12, #4\n\t"
        "subs x13, x13, #1\n\t"
        "b.ne 1b\n\t"
        :
        : [q] "r"(Q), [n] "r"((uint64_t)n_iters)
        : "x9", "x10", "x11", "x12", "x13",
          "z0", "z1", "z2", "z3", "p0", "memory"
    );
}

// Test 3: Scalar ldr + dup (avoid ld1rw entirely)
void test_ldr_dup(const int8_t* Q, int n_iters) {
    __asm__ volatile(
        "ptrue p0.s\n\t"
        "mov x9, %[q]\n\t"            // Row 0
        "add x10, x9, #256\n\t"       // Row 1
        "add x11, x9, #512\n\t"       // Row 2
        "add x12, x9, #768\n\t"       // Row 3
        "mov x13, %[n]\n\t"
        "1:\n\t"
        "ldr w14, [x9], #4\n\t"       // Load + post-inc
        "ldr w15, [x10], #4\n\t"
        "ldr w16, [x11], #4\n\t"
        "ldr w17, [x12], #4\n\t"
        "dup z0.s, w14\n\t"           // Broadcast
        "dup z1.s, w15\n\t"
        "dup z2.s, w16\n\t"
        "dup z3.s, w17\n\t"
        "subs x13, x13, #1\n\t"
        "b.ne 1b\n\t"
        :
        : [q] "r"(Q), [n] "r"((uint64_t)n_iters)
        : "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17",
          "z0", "z1", "z2", "z3", "p0", "memory"
    );
}

// Test 4: Unrolled 4x with pre-computed bases
void test_unroll4_precomputed(const int8_t* Q, int n_iters) {
    __asm__ volatile(
        "ptrue p0.s\n\t"
        "mov x9, %[q]\n\t"
        "add x10, x9, #256\n\t"
        "add x11, x9, #512\n\t"
        "add x12, x9, #768\n\t"
        "mov x13, %[n]\n\t"
        "1:\n\t"
        // Iter 0
        "ld1rw {z0.s}, p0/z, [x9]\n\t"
        "ld1rw {z1.s}, p0/z, [x10]\n\t"
        "ld1rw {z2.s}, p0/z, [x11]\n\t"
        "ld1rw {z3.s}, p0/z, [x12]\n\t"
        // Iter 1
        "ld1rw {z4.s}, p0/z, [x9, #4]\n\t"
        "ld1rw {z5.s}, p0/z, [x10, #4]\n\t"
        "ld1rw {z6.s}, p0/z, [x11, #4]\n\t"
        "ld1rw {z7.s}, p0/z, [x12, #4]\n\t"
        // Iter 2
        "ld1rw {z8.s}, p0/z, [x9, #8]\n\t"
        "ld1rw {z9.s}, p0/z, [x10, #8]\n\t"
        "ld1rw {z10.s}, p0/z, [x11, #8]\n\t"
        "ld1rw {z11.s}, p0/z, [x12, #8]\n\t"
        // Iter 3
        "ld1rw {z12.s}, p0/z, [x9, #12]\n\t"
        "ld1rw {z13.s}, p0/z, [x10, #12]\n\t"
        "ld1rw {z14.s}, p0/z, [x11, #12]\n\t"
        "ld1rw {z15.s}, p0/z, [x12, #12]\n\t"
        // Advance
        "add x9, x9, #16\n\t"
        "add x10, x10, #16\n\t"
        "add x11, x11, #16\n\t"
        "add x12, x12, #16\n\t"
        "subs x13, x13, #4\n\t"
        "b.ne 1b\n\t"
        :
        : [q] "r"(Q), [n] "r"((uint64_t)n_iters)
        : "x9", "x10", "x11", "x12", "x13",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "p0", "memory"
    );
}

// Test 5: ldr + dup unrolled 4x
void test_ldr_dup_unroll4(const int8_t* Q, int n_iters) {
    __asm__ volatile(
        "ptrue p0.s\n\t"
        "mov x0, %[q]\n\t"
        "add x1, x0, #256\n\t"
        "add x2, x0, #512\n\t"
        "add x3, x0, #768\n\t"
        "mov x4, %[n]\n\t"
        "1:\n\t"
        // Load 4 columns from each row
        "ldp w5, w6, [x0], #16\n\t"   // Row 0: cols 0,1 + post-inc 16
        "ldp w7, w8, [x0, #-8]\n\t"   // Row 0: cols 2,3 (offset from new x0)
        "ldp w9, w10, [x1], #16\n\t"  // Row 1
        "ldp w11, w12, [x1, #-8]\n\t"
        "ldp w13, w14, [x2], #16\n\t" // Row 2
        "ldp w15, w16, [x2, #-8]\n\t"
        "ldp w17, w18, [x3], #16\n\t" // Row 3
        "ldp w19, w20, [x3, #-8]\n\t"
        // Broadcast
        "dup z0.s, w5\n\t"
        "dup z1.s, w9\n\t"
        "dup z2.s, w13\n\t"
        "dup z3.s, w17\n\t"
        "dup z4.s, w6\n\t"
        "dup z5.s, w10\n\t"
        "dup z6.s, w14\n\t"
        "dup z7.s, w18\n\t"
        "dup z8.s, w7\n\t"
        "dup z9.s, w11\n\t"
        "dup z10.s, w15\n\t"
        "dup z11.s, w19\n\t"
        "dup z12.s, w8\n\t"
        "dup z13.s, w12\n\t"
        "dup z14.s, w16\n\t"
        "dup z15.s, w20\n\t"
        "subs x4, x4, #4\n\t"
        "b.ne 1b\n\t"
        :
        : [q] "r"(Q), [n] "r"((uint64_t)n_iters)
        : "x0", "x1", "x2", "x3", "x4",
          "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12",
          "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "p0", "memory"
    );
}

typedef void (*test_fn)(const int8_t*, int);

void bench(const char* name, test_fn fn, const int8_t* Q, int n_iters, int outer_iters) {
    // Warmup
    for (int i = 0; i < 100; i++) fn(Q, n_iters);

    uint64_t start = rdtsc();
    for (int i = 0; i < outer_iters; i++) fn(Q, n_iters);
    uint64_t end = rdtsc();

    double ticks = (double)(end - start) / outer_iters;
    double cycles = ticks * 20.0;
    int n_loads = n_iters * 4;
    double cy_per_load = cycles / n_loads;

    printf("%-35s %8.1f cy  %5.2f cy/ld\n", name, cycles, cy_per_load);
}

int main(int argc, char** argv) {
    int n_iters = 64;
    int outer_iters = 10000;

    printf("==============================================\n");
    printf("ld1rw vs ldr+dup Optimization Tests\n");
    printf("==============================================\n");
    printf("64 iterations x 4 rows = 256 broadcast loads\n\n");

    int8_t* Q = (int8_t*)aligned_alloc(256, 1024);
    memset(Q, 1, 1024);

    printf("%-35s %8s    %s\n", "Test", "Cycles", "cy/load");
    printf("----------------------------------- --------    -------\n");

    bench("1. original (C ptr arith)", test_original, Q, n_iters, outer_iters);
    bench("2. precomputed base + post-inc", test_precomputed_postinc, Q, n_iters, outer_iters);
    bench("3. ldr + dup", test_ldr_dup, Q, n_iters, outer_iters);
    bench("4. ld1rw unroll4 + imm offset", test_unroll4_precomputed, Q, n_iters, outer_iters);
    bench("5. ldp + dup unroll4", test_ldr_dup_unroll4, Q, n_iters, outer_iters);

    printf("\n");
    printf("Target: ~0.5 cy/load (match 2 loads/cycle throughput)\n");

    free(Q);
    return 0;
}
