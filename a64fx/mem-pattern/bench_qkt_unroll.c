// bench_qkt_unroll.c
// Q@K^T with loop unrolling to reduce overhead

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

#define TIMER_FREQ_HZ 100000000UL
#define CPU_FREQ_HZ   2000000000UL

static inline uint64_t rdtsc(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline svint32_t sve_ld1rw_s32(svbool_t pg, const int32_t* ptr) {
    svint32_t result;
    __asm__ volatile(
        "ld1rw {%0.s}, %1/z, [%2]"
        : "=w"(result)
        : "Upl"(pg), "r"(ptr)
        : "memory"
    );
    return result;
}

#define NOP_BLOCK_24() __asm__ volatile(      \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop"             \
    ::: "memory")

#define USE_RESULT(v) __asm__ volatile("" :: "w"(v) : "memory")

// Original: 64 iterations
void qkt_loop64(const int8_t* Q, const int8_t* K_int, int32_t* S) {
    svbool_t pg = svptrue_b8();
    svbool_t pg32 = svptrue_b32();

    const int8_t* k_ptr = K_int;
    const int8_t* q_ptr = Q;

    for (int d = 0; d < 64; d++) {
        svint8_t k0 = svld1_s8(pg, k_ptr);
        svint8_t k1 = svld1_s8(pg, k_ptr + 64);
        svint8_t k2 = svld1_s8(pg, k_ptr + 128);
        svint8_t k3 = svld1_s8(pg, k_ptr + 192);
        k_ptr += 256;

        svint32_t q0 = sve_ld1rw_s32(pg32, (const int32_t*)q_ptr);
        svint32_t q1 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 256));
        svint32_t q2 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 512));
        svint32_t q3 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 768));
        q_ptr += 4;

        USE_RESULT(k0); USE_RESULT(k1); USE_RESULT(k2); USE_RESULT(k3);
        USE_RESULT(q0); USE_RESULT(q1); USE_RESULT(q2); USE_RESULT(q3);
        NOP_BLOCK_24();
    }

    svst1_s32(pg32, S, svdup_s32(0));
}

// Unroll 2x: 32 iterations
void qkt_unroll2(const int8_t* Q, const int8_t* K_int, int32_t* S) {
    svbool_t pg = svptrue_b8();
    svbool_t pg32 = svptrue_b32();

    const int8_t* k_ptr = K_int;
    const int8_t* q_ptr = Q;

    for (int d = 0; d < 32; d++) {
        // Iteration 0
        svint8_t k0 = svld1_s8(pg, k_ptr);
        svint8_t k1 = svld1_s8(pg, k_ptr + 64);
        svint8_t k2 = svld1_s8(pg, k_ptr + 128);
        svint8_t k3 = svld1_s8(pg, k_ptr + 192);

        svint32_t q0 = sve_ld1rw_s32(pg32, (const int32_t*)q_ptr);
        svint32_t q1 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 256));
        svint32_t q2 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 512));
        svint32_t q3 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 768));

        USE_RESULT(k0); USE_RESULT(k1); USE_RESULT(k2); USE_RESULT(k3);
        USE_RESULT(q0); USE_RESULT(q1); USE_RESULT(q2); USE_RESULT(q3);
        NOP_BLOCK_24();

        // Iteration 1
        svint8_t k4 = svld1_s8(pg, k_ptr + 256);
        svint8_t k5 = svld1_s8(pg, k_ptr + 320);
        svint8_t k6 = svld1_s8(pg, k_ptr + 384);
        svint8_t k7 = svld1_s8(pg, k_ptr + 448);
        k_ptr += 512;

        svint32_t q4 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 4));
        svint32_t q5 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 260));
        svint32_t q6 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 516));
        svint32_t q7 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 772));
        q_ptr += 8;

        USE_RESULT(k4); USE_RESULT(k5); USE_RESULT(k6); USE_RESULT(k7);
        USE_RESULT(q4); USE_RESULT(q5); USE_RESULT(q6); USE_RESULT(q7);
        NOP_BLOCK_24();
    }

    svst1_s32(pg32, S, svdup_s32(0));
}

// Unroll 4x: 16 iterations
void qkt_unroll4(const int8_t* Q, const int8_t* K_int, int32_t* S) {
    svbool_t pg = svptrue_b8();
    svbool_t pg32 = svptrue_b32();

    const int8_t* k_ptr = K_int;
    const int8_t* q_ptr = Q;

    for (int d = 0; d < 16; d++) {
        // 4 iterations unrolled
        for (int u = 0; u < 4; u++) {
            svint8_t k0 = svld1_s8(pg, k_ptr);
            svint8_t k1 = svld1_s8(pg, k_ptr + 64);
            svint8_t k2 = svld1_s8(pg, k_ptr + 128);
            svint8_t k3 = svld1_s8(pg, k_ptr + 192);
            k_ptr += 256;

            svint32_t q0 = sve_ld1rw_s32(pg32, (const int32_t*)q_ptr);
            svint32_t q1 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 256));
            svint32_t q2 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 512));
            svint32_t q3 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 768));
            q_ptr += 4;

            USE_RESULT(k0); USE_RESULT(k1); USE_RESULT(k2); USE_RESULT(k3);
            USE_RESULT(q0); USE_RESULT(q1); USE_RESULT(q2); USE_RESULT(q3);
            NOP_BLOCK_24();
        }
    }

    svst1_s32(pg32, S, svdup_s32(0));
}

// Fully unrolled (no loop)
void qkt_full_unroll(const int8_t* Q, const int8_t* K_int, int32_t* S) {
    svbool_t pg = svptrue_b8();
    svbool_t pg32 = svptrue_b32();

    #define LOAD_ITER(d) do { \
        svint8_t k0 = svld1_s8(pg, K_int + (d)*256); \
        svint8_t k1 = svld1_s8(pg, K_int + (d)*256 + 64); \
        svint8_t k2 = svld1_s8(pg, K_int + (d)*256 + 128); \
        svint8_t k3 = svld1_s8(pg, K_int + (d)*256 + 192); \
        svint32_t q0 = sve_ld1rw_s32(pg32, (const int32_t*)(Q + (d)*4)); \
        svint32_t q1 = sve_ld1rw_s32(pg32, (const int32_t*)(Q + 256 + (d)*4)); \
        svint32_t q2 = sve_ld1rw_s32(pg32, (const int32_t*)(Q + 512 + (d)*4)); \
        svint32_t q3 = sve_ld1rw_s32(pg32, (const int32_t*)(Q + 768 + (d)*4)); \
        USE_RESULT(k0); USE_RESULT(k1); USE_RESULT(k2); USE_RESULT(k3); \
        USE_RESULT(q0); USE_RESULT(q1); USE_RESULT(q2); USE_RESULT(q3); \
        NOP_BLOCK_24(); \
    } while(0)

    // Unroll all 64 iterations
    LOAD_ITER(0);  LOAD_ITER(1);  LOAD_ITER(2);  LOAD_ITER(3);
    LOAD_ITER(4);  LOAD_ITER(5);  LOAD_ITER(6);  LOAD_ITER(7);
    LOAD_ITER(8);  LOAD_ITER(9);  LOAD_ITER(10); LOAD_ITER(11);
    LOAD_ITER(12); LOAD_ITER(13); LOAD_ITER(14); LOAD_ITER(15);
    LOAD_ITER(16); LOAD_ITER(17); LOAD_ITER(18); LOAD_ITER(19);
    LOAD_ITER(20); LOAD_ITER(21); LOAD_ITER(22); LOAD_ITER(23);
    LOAD_ITER(24); LOAD_ITER(25); LOAD_ITER(26); LOAD_ITER(27);
    LOAD_ITER(28); LOAD_ITER(29); LOAD_ITER(30); LOAD_ITER(31);
    LOAD_ITER(32); LOAD_ITER(33); LOAD_ITER(34); LOAD_ITER(35);
    LOAD_ITER(36); LOAD_ITER(37); LOAD_ITER(38); LOAD_ITER(39);
    LOAD_ITER(40); LOAD_ITER(41); LOAD_ITER(42); LOAD_ITER(43);
    LOAD_ITER(44); LOAD_ITER(45); LOAD_ITER(46); LOAD_ITER(47);
    LOAD_ITER(48); LOAD_ITER(49); LOAD_ITER(50); LOAD_ITER(51);
    LOAD_ITER(52); LOAD_ITER(53); LOAD_ITER(54); LOAD_ITER(55);
    LOAD_ITER(56); LOAD_ITER(57); LOAD_ITER(58); LOAD_ITER(59);
    LOAD_ITER(60); LOAD_ITER(61); LOAD_ITER(62); LOAD_ITER(63);

    svst1_s32(pg32, S, svdup_s32(0));
    #undef LOAD_ITER
}

typedef void (*kernel_fn)(const int8_t*, const int8_t*, int32_t*);

void bench(const char* name, kernel_fn fn, const int8_t* Q, const int8_t* K, int32_t* S, int iters, int warmup) {
    for (int i = 0; i < warmup; i++) fn(Q, K, S);

    uint64_t start = rdtsc();
    for (int i = 0; i < iters; i++) fn(Q, K, S);
    uint64_t end = rdtsc();

    double ticks = (double)(end - start) / iters;
    double cycles = ticks * 20.0;
    double load_bw = 17408.0 / cycles;
    double eff = 100.0 * load_bw / 128.0;

    printf("%-20s %8.1f cycles  %6.1f B/cy  %5.1f%%\n", name, cycles, load_bw, eff);
}

int main(int argc, char** argv) {
    int iters = 10000;
    int warmup = 1000;

    if (argc > 1) iters = atoi(argv[1]);

    printf("==============================================\n");
    printf("Q@K^T Loop Unrolling Comparison\n");
    printf("==============================================\n");
    printf("Iterations: %d\n", iters);
    printf("Theoretical: 136 cycles (17408B / 128 B/cy)\n\n");

    int8_t* Q = (int8_t*)aligned_alloc(256, 1024);
    int8_t* K = (int8_t*)aligned_alloc(256, 16384);
    int32_t* S = (int32_t*)aligned_alloc(256, 256);

    memset(Q, 1, 1024);
    memset(K, 2, 16384);

    printf("%-20s %8s        %6s       %s\n", "Kernel", "Cycles", "BW", "Eff");
    printf("-------------------- --------        ------       -----\n");

    bench("loop64 (baseline)", qkt_loop64, Q, K, S, iters, warmup);
    bench("unroll2 (32 iter)", qkt_unroll2, Q, K, S, iters, warmup);
    bench("unroll4 (16 iter)", qkt_unroll4, Q, K, S, iters, warmup);
    bench("full_unroll (0 iter)", qkt_full_unroll, Q, K, S, iters, warmup);

    free(Q);
    free(K);
    free(S);

    return 0;
}
