// bench_qkt_only.c
// Isolated Q@K^T memory access pattern only
// MR=4, D=256, N=64 -> S[4,64]

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

// 24 NOPs to hide L1 latency
#define NOP_BLOCK_24() __asm__ volatile(      \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop"             \
    ::: "memory")

#define USE_RESULT(v) __asm__ volatile("" :: "w"(v) : "memory")

// Q@K^T only: [4,256] @ [256,64] -> [4,64]
// 64 iterations, per iter: 4x ld1b K (256B), 4x ld1rw Q (16B)
void qkt_mem_access(
    const int8_t* restrict Q,       // [4, 256]
    const int8_t* restrict K_int,   // [64, 64, 4] interleaved
    int32_t* restrict S             // [4, 64] output
) {
    svbool_t pg = svptrue_b8();
    svbool_t pg32 = svptrue_b32();

    const int D_GROUPS = 64;  // D/4
    const int Q_STRIDE = 256; // Q row stride

    svint32_t acc0 = svdup_s32(0);
    svint32_t acc1 = svdup_s32(0);
    svint32_t acc2 = svdup_s32(0);
    svint32_t acc3 = svdup_s32(0);

    const int8_t* k_ptr = K_int;
    const int8_t* q_ptr = Q;

    for (int d = 0; d < D_GROUPS; d++) {
        // K loads: 4x ld1b (64 bytes each) = 256 bytes
        svint8_t k0 = svld1_s8(pg, k_ptr);
        svint8_t k1 = svld1_s8(pg, k_ptr + 64);
        svint8_t k2 = svld1_s8(pg, k_ptr + 128);
        svint8_t k3 = svld1_s8(pg, k_ptr + 192);
        k_ptr += 256;

        // Q loads: 4x ld1rw (4 bytes each, broadcast)
        svint32_t q0 = sve_ld1rw_s32(pg32, (const int32_t*)q_ptr);
        svint32_t q1 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + Q_STRIDE));
        svint32_t q2 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + Q_STRIDE*2));
        svint32_t q3 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + Q_STRIDE*3));
        q_ptr += 4;

        USE_RESULT(k0); USE_RESULT(k1); USE_RESULT(k2); USE_RESULT(k3);
        USE_RESULT(q0); USE_RESULT(q1); USE_RESULT(q2); USE_RESULT(q3);

        // 24 NOPs to hide L1 latency
        NOP_BLOCK_24();
    }

    // Store S[4, 64] = 4 rows x 1 vector = 256 bytes
    svst1_s32(pg32, S, acc0);
    svst1_s32(pg32, S + 16, acc1);
    svst1_s32(pg32, S + 32, acc2);
    svst1_s32(pg32, S + 48, acc3);
}

int main(int argc, char** argv) {
    int iters = 10000;
    int warmup = 1000;

    if (argc > 1) iters = atoi(argv[1]);

    printf("==============================================\n");
    printf("Q@K^T Only Memory Access Pattern\n");
    printf("==============================================\n");
    printf("Iterations: %d, Warmup: %d\n\n", iters, warmup);

    // Memory layout
    printf("=== Memory Layout ===\n");
    printf("  Q[4,256]:    1024 bytes (1 KB)\n");
    printf("  K_int[64,64,4]: 16384 bytes (16 KB)\n");
    printf("  S[4,64]:     256 bytes (output)\n");
    printf("  Total:       17664 bytes (17.2 KB) - fits in L1\n\n");

    // Memory traffic per kernel
    printf("=== Memory Traffic ===\n");
    printf("  K loads: 64 iters x 256B = 16384 bytes\n");
    printf("  Q loads: 64 iters x 16B  = 1024 bytes\n");
    printf("  S stores: 256 bytes\n");
    printf("  Total loaded: 17408 bytes\n");
    printf("  Total stored: 256 bytes\n\n");

    // Theoretical minimum
    double load_cycles = 17408.0 / 128.0;  // 128 B/cy peak
    double store_cycles = 256.0 / 64.0;    // 64 B/cy peak
    printf("=== Theoretical Minimum ===\n");
    printf("  Load-bound:  %.1f cycles (128 B/cy)\n", load_cycles);
    printf("  Store-bound: %.1f cycles (64 B/cy)\n", store_cycles);
    printf("  Limiting:    %.1f cycles\n\n", load_cycles > store_cycles ? load_cycles : store_cycles);

    // Allocate
    int8_t* Q = (int8_t*)aligned_alloc(256, 1024);
    int8_t* K_int = (int8_t*)aligned_alloc(256, 16384);
    int32_t* S = (int32_t*)aligned_alloc(256, 256);

    memset(Q, 1, 1024);
    memset(K_int, 2, 16384);
    memset(S, 0, 256);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        qkt_mem_access(Q, K_int, S);
    }

    // Benchmark
    uint64_t start = rdtsc();
    for (int i = 0; i < iters; i++) {
        qkt_mem_access(Q, K_int, S);
    }
    uint64_t end = rdtsc();

    double ticks = (double)(end - start) / iters;
    double cycles = ticks * 20.0;  // 100MHz -> 2000MHz

    double load_bw = 17408.0 / cycles;
    double efficiency = 100.0 * load_bw / 128.0;

    printf("=== Results ===\n");
    printf("  Timer ticks:    %.1f\n", ticks);
    printf("  CPU cycles:     %.1f\n", cycles);
    printf("  Time/kernel:    %.1f ns\n", ticks * 10.0);  // 100MHz = 10ns/tick
    printf("\n");
    printf("  Load BW:        %.1f byte/cycle (peak: 128)\n", load_bw);
    printf("  Efficiency:     %.1f%%\n", efficiency);
    printf("  vs theoretical: %.2fx\n", cycles / load_cycles);

    free(Q);
    free(K_int);
    free(S);

    return 0;
}
