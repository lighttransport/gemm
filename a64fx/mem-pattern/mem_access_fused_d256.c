// mem_access_fused_d256.c
// C intrinsics version of memory-access-only fused GEMM kernel
//
// Uses SVE intrinsics to perform the same memory access pattern as
// the ASM kernel, with NOPs replacing compute (SDOT) instructions.
//
// This version is useful for:
// 1. Comparing compiler-generated code vs hand-written ASM
// 2. Easier modification for different analysis scenarios
// 3. Validating address calculations

#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>

// 24 NOPs to hide L1 latency (11 cycles)
// A64FX issues 4 instructions/cycle, 24 NOPs = 6 cycles of work
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

// Inline ld1rw: load 32-bit scalar and replicate across all lanes
// SVE ACLE doesn't have direct intrinsic for ld1rw, so use inline asm
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

// Memory layout constants (matching kernel_fused_d256_4row.S)
#define MR 4            // M-tile rows
#define D_DIM 256       // Head dimension
#define N_CHUNK 64      // N-tile columns
#define D_GROUPS 64     // D/4 for SDOT groups
#define N_GROUPS 16     // N_CHUNK/4 for SDOT groups
#define D_TILES 4       // D_DIM/64 for output tiles

// Q row stride = D = 256 bytes
// K_int stride per d_group = N_CHUNK * 4 = 256 bytes
// V_t_int stride per N_group = D_DIM * 4 = 1024 bytes
// P row stride = N_CHUNK = 64 bytes
// O row stride = D_DIM * sizeof(int32_t) = 1024 bytes

void mem_access_fused_d256_c(
    const int8_t* restrict Q,       // [4, 256] row-major
    const int8_t* restrict K_int,   // [64, 64, 4] interleaved
    const int8_t* restrict V_t,     // [16, 256, 4] interleaved
    int32_t* restrict O             // [4, 256] int32
) {
    svbool_t pg = svptrue_b8();
    svbool_t pg32 = svptrue_b32();

    // Stack buffer for P[4, 64] = 256 bytes
    int8_t P[256] __attribute__((aligned(64)));

    // Initialize dummy accumulators (for register pressure)
    svint32_t acc0 = svdup_s32(0);
    svint32_t acc1 = svdup_s32(0);
    svint32_t acc2 = svdup_s32(0);
    svint32_t acc3 = svdup_s32(0);

    // ========================================================================
    // PHASE 1: Q @ K^T memory access pattern
    // 64 iterations: 4x ld1b K (256B), 4x ld1rw Q (16B)
    // ========================================================================

    const int8_t* k_ptr = K_int;
    const int8_t* q_ptr = Q;

    for (int d = 0; d < D_GROUPS; d++) {
        // K loads: 4x ld1b (64 bytes each) = 256 bytes
        svint8_t k0 = svld1_s8(pg, k_ptr);
        svint8_t k1 = svld1_s8(pg, k_ptr + 64);
        svint8_t k2 = svld1_s8(pg, k_ptr + 128);
        svint8_t k3 = svld1_s8(pg, k_ptr + 192);
        k_ptr += 256;  // K_int stride per d_group

        // Q loads: 4x ld1rw (4 bytes each, broadcasted) = 16 bytes
        svint32_t q0 = sve_ld1rw_s32(pg32, (const int32_t*)q_ptr);
        svint32_t q1 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 256));
        svint32_t q2 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 512));
        svint32_t q3 = sve_ld1rw_s32(pg32, (const int32_t*)(q_ptr + 768));
        q_ptr += 4;  // Q advances by 4 bytes

        // Prevent optimizer from removing loads
        USE_RESULT(k0); USE_RESULT(k1); USE_RESULT(k2); USE_RESULT(k3);
        USE_RESULT(q0); USE_RESULT(q1); USE_RESULT(q2); USE_RESULT(q3);

        // 24 NOPs to hide L1 latency (11 cycles)
        NOP_BLOCK_24();
    }

    // ========================================================================
    // PHASE 2: Quantize (simplified, just stores)
    // Store 4x 64 bytes = 256 bytes to stack
    // ========================================================================

    svst1_s8(pg, P, svdup_s8(0));
    svst1_s8(pg, P + 64, svdup_s8(0));
    svst1_s8(pg, P + 128, svdup_s8(0));
    svst1_s8(pg, P + 192, svdup_s8(0));

    // ========================================================================
    // PHASE 3: P @ V memory access pattern
    // 4 D-tiles x 16 N_groups: 4x ld1b V (256B), 4x ld1rw P (16B), st1w O
    // ========================================================================

    int32_t* o_base = O;
    const int8_t* v_base = V_t;

    for (int d_tile = 0; d_tile < D_TILES; d_tile++) {
        const int8_t* p_ptr = P;
        const int8_t* v_ptr = v_base;

        for (int n_grp = 0; n_grp < N_GROUPS; n_grp++) {
            // V loads: 4x ld1b (64 bytes each) = 256 bytes
            svint8_t v0 = svld1_s8(pg, v_ptr);
            svint8_t v1 = svld1_s8(pg, v_ptr + 64);
            svint8_t v2 = svld1_s8(pg, v_ptr + 128);
            svint8_t v3 = svld1_s8(pg, v_ptr + 192);
            v_ptr += 1024;  // V_t_int stride per N_group

            // P loads: 4x ld1rw (4 bytes each, broadcasted) = 16 bytes
            svint32_t p0 = sve_ld1rw_s32(pg32, (const int32_t*)p_ptr);
            svint32_t p1 = sve_ld1rw_s32(pg32, (const int32_t*)(p_ptr + 64));
            svint32_t p2 = sve_ld1rw_s32(pg32, (const int32_t*)(p_ptr + 128));
            svint32_t p3 = sve_ld1rw_s32(pg32, (const int32_t*)(p_ptr + 192));
            p_ptr += 4;  // P advances by 4 bytes

            // Prevent optimizer from removing loads
            USE_RESULT(v0); USE_RESULT(v1); USE_RESULT(v2); USE_RESULT(v3);
            USE_RESULT(p0); USE_RESULT(p1); USE_RESULT(p2); USE_RESULT(p3);

            // 24 NOPs to hide L1 latency (11 cycles)
            NOP_BLOCK_24();
        }

        // O stores: 4 rows x 4 vectors = 16x st1w (64 bytes each) = 1024 bytes
        int32_t* o_ptr = o_base;
        svst1_s32(pg32, o_ptr, acc0);
        svst1_s32(pg32, o_ptr + 16, acc1);
        svst1_s32(pg32, o_ptr + 32, acc2);
        svst1_s32(pg32, o_ptr + 48, acc3);

        o_ptr = o_base + 256;  // Row 1 (O row stride = 1024 bytes = 256 int32)
        svst1_s32(pg32, o_ptr, acc0);
        svst1_s32(pg32, o_ptr + 16, acc1);
        svst1_s32(pg32, o_ptr + 32, acc2);
        svst1_s32(pg32, o_ptr + 48, acc3);

        o_ptr = o_base + 512;  // Row 2
        svst1_s32(pg32, o_ptr, acc0);
        svst1_s32(pg32, o_ptr + 16, acc1);
        svst1_s32(pg32, o_ptr + 32, acc2);
        svst1_s32(pg32, o_ptr + 48, acc3);

        o_ptr = o_base + 768;  // Row 3
        svst1_s32(pg32, o_ptr, acc0);
        svst1_s32(pg32, o_ptr + 16, acc1);
        svst1_s32(pg32, o_ptr + 32, acc2);
        svst1_s32(pg32, o_ptr + 48, acc3);

        // Advance to next D-tile
        o_base += 16;    // O advances by 64 bytes = 16 int32
        v_base += 256;   // V_t_int advances by 256 bytes
    }
}

// Debug version that prints addresses for verification
void mem_access_fused_d256_debug(
    const int8_t* Q,
    const int8_t* K_int,
    const int8_t* V_t,
    int32_t* O,
    int verbose
) {
    if (verbose) {
        // Print memory layout info
        printf("Memory Layout Analysis:\n");
        printf("  Q base:     %p\n", (void*)Q);
        printf("  K_int base: %p\n", (void*)K_int);
        printf("  V_t base:   %p\n", (void*)V_t);
        printf("  O base:     %p\n", (void*)O);
        printf("\nPhase 1 (Q@K^T) - first 3 iterations:\n");

        const int8_t* k_ptr = K_int;
        const int8_t* q_ptr = Q;
        for (int d = 0; d < 3; d++) {
            printf("  d=%d: K=[%p, %p, %p, %p] Q=[%p, %p, %p, %p]\n",
                   d,
                   (void*)k_ptr, (void*)(k_ptr+64), (void*)(k_ptr+128), (void*)(k_ptr+192),
                   (void*)q_ptr, (void*)(q_ptr+256), (void*)(q_ptr+512), (void*)(q_ptr+768));
            k_ptr += 256;
            q_ptr += 4;
        }

        printf("\nPhase 3 (P@V) - first D-tile, first 3 N_groups:\n");
        const int8_t* v_ptr = V_t;
        for (int n = 0; n < 3; n++) {
            printf("  n=%d: V=[%p, %p, %p, %p]\n",
                   n,
                   (void*)v_ptr, (void*)(v_ptr+64), (void*)(v_ptr+128), (void*)(v_ptr+192));
            v_ptr += 1024;
        }
    }

    // Run the actual kernel
    mem_access_fused_d256_c(Q, K_int, V_t, O);
}
