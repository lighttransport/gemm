// bench_l2_pattern.c
// L2 cache performance test - larger working set to exceed L1 (64KB)
//
// Test configurations:
// - L1 test: Working set ~34KB (fits in 64KB L1)
// - L2 test: Working set ~256KB (exceeds L1, fits in L2)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

// Timer frequency (A64FX cntvct_el0 runs at 100MHz)
#define TIMER_FREQ_HZ 100000000UL
#define CPU_FREQ_HZ   2000000000UL

static inline uint64_t rdtsc(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// Inline ld1rw for broadcast load
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

// 24 NOPs to hide L1 latency (11 cycles)
#define NOP_BLOCK_24() __asm__ volatile(      \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop\n\t"         \
    "nop\n\t nop\n\t nop\n\t nop"             \
    ::: "memory")

// Prevent compiler from optimizing away loads
#define USE_RESULT(v) __asm__ volatile("" :: "w"(v) : "memory")

// Memory access pattern - parameterized for different cache levels
// MR = 4 rows of Q/P
// D = head dimension (256)
// N = sequence length (number of K/V columns)
void mem_access_pattern(
    const int8_t* restrict Q,       // [MR, D]
    const int8_t* restrict K_int,   // [D/4, N, 4] interleaved
    const int8_t* restrict V_t,     // [N/4, D, 4] interleaved
    int32_t* restrict O,            // [MR, D]
    int D,                          // Head dimension
    int N                           // Sequence length (N columns)
) {
    svbool_t pg = svptrue_b8();
    svbool_t pg32 = svptrue_b32();

    const int MR = 4;
    const int D_GROUPS = D / 4;       // D/4 for SDOT groups
    const int N_GROUPS = N / 4;       // N/4 for SDOT groups
    const int D_TILES = D / 64;       // D/64 for 64-byte output tiles

    // Stack buffer for P[MR, N] (up to 4KB for N=1024)
    int8_t* P = (int8_t*)aligned_alloc(64, MR * N);

    svint32_t acc0 = svdup_s32(0);
    svint32_t acc1 = svdup_s32(0);
    svint32_t acc2 = svdup_s32(0);
    svint32_t acc3 = svdup_s32(0);

    // ========================================================================
    // PHASE 1: Q @ K^T memory access pattern
    // Per iteration: 4x ld1b K (256B), 4x ld1rw Q (16B)
    // ========================================================================

    const int8_t* k_ptr = K_int;
    const int8_t* q_ptr = Q;
    const int K_STRIDE = N * 4;  // K_int stride per d_group = N * 4 bytes

    for (int d = 0; d < D_GROUPS; d++) {
        // K loads: 4x ld1b (64 bytes each) = 256 bytes
        svint8_t k0 = svld1_s8(pg, k_ptr);
        svint8_t k1 = svld1_s8(pg, k_ptr + 64);
        svint8_t k2 = svld1_s8(pg, k_ptr + 128);
        svint8_t k3 = svld1_s8(pg, k_ptr + 192);
        k_ptr += K_STRIDE;

        // Q loads: 4x ld1rw (4 bytes each, broadcast)
        svint32_t q0 = sve_ld1rw_s32(pg32, (const int32_t*)q_ptr);
        svint32_t q1 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + D));
        svint32_t q2 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + D*2));
        svint32_t q3 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + D*3));
        q_ptr += 4;

        USE_RESULT(k0); USE_RESULT(k1); USE_RESULT(k2); USE_RESULT(k3);
        USE_RESULT(q0); USE_RESULT(q1); USE_RESULT(q2); USE_RESULT(q3);

        // 24 NOPs to hide L1 latency
        NOP_BLOCK_24();
    }

    // ========================================================================
    // PHASE 2: Quantize - Store P[MR, N] to scratch
    // ========================================================================

    for (int i = 0; i < MR * N; i += 64) {
        svst1_s8(pg, P + i, svdup_s8(0));
    }

    // ========================================================================
    // PHASE 3: P @ V memory access pattern
    // ========================================================================

    int32_t* o_base = O;
    const int8_t* v_base = V_t;
    const int V_STRIDE = D * 4;  // V_t stride per N_group = D * 4 bytes

    for (int d_tile = 0; d_tile < D_TILES; d_tile++) {
        const int8_t* p_ptr = P;
        const int8_t* v_ptr = v_base;

        for (int n_grp = 0; n_grp < N_GROUPS; n_grp++) {
            // V loads: 4x ld1b (64 bytes each) = 256 bytes
            svint8_t v0 = svld1_s8(pg, v_ptr);
            svint8_t v1 = svld1_s8(pg, v_ptr + 64);
            svint8_t v2 = svld1_s8(pg, v_ptr + 128);
            svint8_t v3 = svld1_s8(pg, v_ptr + 192);
            v_ptr += V_STRIDE;

            // P loads: 4x ld1rw (4 bytes each, broadcast)
            svint32_t p0 = sve_ld1rw_s32(pg32, (const int32_t*)p_ptr);
            svint32_t p1 = sve_ld1rw_s32(pg32, (const int32_t*)(p_ptr + N));
            svint32_t p2 = sve_ld1rw_s32(pg32, (const int32_t*)(p_ptr + N*2));
            svint32_t p3 = sve_ld1rw_s32(pg32, (const int32_t*)(p_ptr + N*3));
            p_ptr += 4;

            USE_RESULT(v0); USE_RESULT(v1); USE_RESULT(v2); USE_RESULT(v3);
            USE_RESULT(p0); USE_RESULT(p1); USE_RESULT(p2); USE_RESULT(p3);

            // 24 NOPs to hide L1 latency
            NOP_BLOCK_24();
        }

        // O stores
        int32_t* o_ptr = o_base;
        for (int row = 0; row < MR; row++) {
            svst1_s32(pg32, o_ptr, acc0);
            svst1_s32(pg32, o_ptr + 16, acc1);
            svst1_s32(pg32, o_ptr + 32, acc2);
            svst1_s32(pg32, o_ptr + 48, acc3);
            o_ptr += D;
        }

        o_base += 16;
        v_base += 256;
    }

    free(P);
}

typedef struct {
    const char* name;
    int D;              // Head dimension
    int N;              // Sequence length
    size_t q_bytes;
    size_t k_bytes;
    size_t v_bytes;
    size_t p_bytes;
    size_t o_bytes;
    size_t total_bytes;
} TestConfig;

void print_config(TestConfig* cfg) {
    printf("  %s: D=%d, N=%d\n", cfg->name, cfg->D, cfg->N);
    printf("    Q[4,%d]:    %6zu bytes\n", cfg->D, cfg->q_bytes);
    printf("    K_int:      %6zu bytes\n", cfg->k_bytes);
    printf("    V_t:        %6zu bytes\n", cfg->v_bytes);
    printf("    P[4,%d]:    %6zu bytes\n", cfg->N, cfg->p_bytes);
    printf("    O[4,%d]:    %6zu bytes\n", cfg->D, cfg->o_bytes);
    printf("    Total:      %6zu bytes (%.1f KB)\n", cfg->total_bytes, cfg->total_bytes / 1024.0);

    if (cfg->total_bytes <= 64*1024) {
        printf("    Cache:      L1 resident (< 64KB)\n");
    } else if (cfg->total_bytes <= 8*1024*1024) {
        printf("    Cache:      L2 resident (64KB - 8MB)\n");
    } else {
        printf("    Cache:      Memory spill (> 8MB)\n");
    }
}

int main(int argc, char** argv) {
    int iters = 1000;
    int warmup = 100;

    if (argc > 1) iters = atoi(argv[1]);
    if (argc > 2) warmup = atoi(argv[2]);

    printf("==============================================\n");
    printf("L1 vs L2 Cache Memory Access Pattern Test\n");
    printf("==============================================\n");
    printf("Iterations: %d, Warmup: %d\n\n", iters, warmup);

    // Test configurations
    TestConfig configs[] = {
        // L1 test: D=256, N=64 -> ~34KB (fits in L1)
        {"L1_small", 256, 64, 0, 0, 0, 0, 0, 0},

        // L1 edge: D=256, N=128 -> ~70KB (just over L1)
        {"L1_edge", 256, 128, 0, 0, 0, 0, 0, 0},

        // L2 test: D=256, N=256 -> ~140KB (L2)
        {"L2_small", 256, 256, 0, 0, 0, 0, 0, 0},

        // L2 test: D=256, N=512 -> ~280KB (L2)
        {"L2_medium", 256, 512, 0, 0, 0, 0, 0, 0},

        // L2 test: D=256, N=1024 -> ~560KB (L2)
        {"L2_large", 256, 1024, 0, 0, 0, 0, 0, 0},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    // Calculate buffer sizes
    for (int c = 0; c < num_configs; c++) {
        TestConfig* cfg = &configs[c];
        cfg->q_bytes = 4 * cfg->D;                      // Q[4, D]
        cfg->k_bytes = (cfg->D/4) * cfg->N * 4;         // K_int[D/4, N, 4]
        cfg->v_bytes = (cfg->N/4) * cfg->D * 4;         // V_t[N/4, D, 4]
        cfg->p_bytes = 4 * cfg->N;                      // P[4, N]
        cfg->o_bytes = 4 * cfg->D * sizeof(int32_t);    // O[4, D] int32
        cfg->total_bytes = cfg->q_bytes + cfg->k_bytes + cfg->v_bytes + cfg->p_bytes + cfg->o_bytes;
    }

    printf("=== Test Configurations ===\n\n");
    for (int c = 0; c < num_configs; c++) {
        print_config(&configs[c]);
        printf("\n");
    }

    printf("=== Benchmark Results ===\n\n");
    printf("%-12s %8s %10s %10s %10s %10s %10s\n",
           "Config", "N", "Cycles", "Load B/cy", "Peak B/cy", "Efficiency", "L1/L2");
    printf("------------ -------- ---------- ---------- ---------- ---------- ----------\n");

    double cycle_scale = (double)CPU_FREQ_HZ / TIMER_FREQ_HZ;

    for (int c = 0; c < num_configs; c++) {
        TestConfig* cfg = &configs[c];

        // Allocate buffers
        int8_t* Q = (int8_t*)aligned_alloc(256, cfg->q_bytes);
        int8_t* K_int = (int8_t*)aligned_alloc(256, cfg->k_bytes);
        int8_t* V_t = (int8_t*)aligned_alloc(256, cfg->v_bytes);
        int32_t* O = (int32_t*)aligned_alloc(256, cfg->o_bytes);

        // Initialize with pattern
        memset(Q, 1, cfg->q_bytes);
        memset(K_int, 2, cfg->k_bytes);
        memset(V_t, 3, cfg->v_bytes);
        memset(O, 0, cfg->o_bytes);

        // Warmup
        for (int i = 0; i < warmup; i++) {
            mem_access_pattern(Q, K_int, V_t, O, cfg->D, cfg->N);
        }

        // Benchmark
        uint64_t start = rdtsc();
        for (int i = 0; i < iters; i++) {
            mem_access_pattern(Q, K_int, V_t, O, cfg->D, cfg->N);
        }
        uint64_t end = rdtsc();

        double ticks = (double)(end - start) / iters;
        double cycles = ticks * cycle_scale;

        // Calculate memory traffic
        // Phase 1: K loads + Q loads per iteration
        size_t phase1_loads = (cfg->D/4) * (256 + 16);
        // Phase 3: V loads + P loads per iteration
        size_t phase3_loads = (cfg->D/64) * (cfg->N/4) * (256 + 16);
        size_t total_loads = phase1_loads + phase3_loads;

        double load_bw = total_loads / cycles;
        double peak_bw = (cfg->total_bytes <= 64*1024) ? 128.0 : 42.6;  // L1 vs L2
        double efficiency = 100.0 * load_bw / peak_bw;
        const char* cache_level = (cfg->total_bytes <= 64*1024) ? "L1" : "L2";

        printf("%-12s %8d %10.0f %10.1f %10.1f %9.1f%% %10s\n",
               cfg->name, cfg->N, cycles, load_bw, peak_bw, efficiency, cache_level);

        free(Q);
        free(K_int);
        free(V_t);
        free(O);
    }

    printf("\n");
    printf("=== Analysis ===\n");
    printf("L1 peak bandwidth: 128 bytes/cycle (2x 64B load pipes)\n");
    printf("L2 peak bandwidth: ~42.6 bytes/cycle/core (shared per CMG)\n");
    printf("L1 latency: 11 cycles\n");
    printf("L2 latency: 27-37 cycles\n");
    printf("\n");
    printf("Expected:\n");
    printf("  - L1 configs should show higher efficiency\n");
    printf("  - L2 configs will be limited by L2 bandwidth (~42 B/cy)\n");
    printf("  - Larger N increases working set, triggers L2 access\n");

    return 0;
}
