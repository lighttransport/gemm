/* bf16_gemm.c — A64FX BF16-storage, FP32-compute SVE GEMM.
 *
 * A64FX has no BFMMLA. Strategy: store B as BF16 (half the bytes), load with
 * LD1H{Z.S} + LSL #16 → FP32 in registers, accumulate FP32.
 *
 * Microtile: MR=8 rows × NR=48 cols (3 SVE vec lanes) — same shape as the
 * fp32 8x3_unroll4 asm kernel so we can compare apples-to-apples. K is
 * unrolled by 4; loads are bunched at the top of each K-block so the
 * compiler can schedule them ahead of the FMAs.
 *
 * SVE types are sizeless → cannot live in C arrays. Accumulators are 24
 * named locals; the FMA pass is macro-expanded over 8 rows.
 */
#include "bf16_gemm.h"
#include "fused_gemm.h"   /* MR=8, NR=48, pack_A_fp32, packed_A_size */

#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void f32_to_bf16(const float *src, uint16_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &src[i], 4);
        uint32_t lsb     = (u >> 16) & 1u;
        uint32_t rounder = 0x7FFFu + lsb;
        u += rounder;
        dst[i] = (uint16_t)(u >> 16);
    }
}

void f32_to_bf16_buf(const float *src, uint16_t *dst, size_t n) {
    f32_to_bf16(src, dst, n);
}

static inline svfloat32_t svld1_bf16_f32(svbool_t pg, const uint16_t *p) {
    svuint32_t raw = svld1uh_u32(pg, p);
    return svreinterpret_f32_u32(svlsl_n_u32_x(pg, raw, 16));
}

#define FMA_ROW(M_IDX) \
    do { \
        float av0 = A[(size_t)(M_IDX) * lda + (k + 0)]; \
        float av1 = A[(size_t)(M_IDX) * lda + (k + 1)]; \
        float av2 = A[(size_t)(M_IDX) * lda + (k + 2)]; \
        float av3 = A[(size_t)(M_IDX) * lda + (k + 3)]; \
        a##M_IDX##0 = svmla_n_f32_x(pg,  a##M_IDX##0, b0_0, av0); \
        a##M_IDX##1 = svmla_n_f32_x(pg1, a##M_IDX##1, b1_0, av0); \
        a##M_IDX##2 = svmla_n_f32_x(pg2, a##M_IDX##2, b2_0, av0); \
        a##M_IDX##0 = svmla_n_f32_x(pg,  a##M_IDX##0, b0_1, av1); \
        a##M_IDX##1 = svmla_n_f32_x(pg1, a##M_IDX##1, b1_1, av1); \
        a##M_IDX##2 = svmla_n_f32_x(pg2, a##M_IDX##2, b2_1, av1); \
        a##M_IDX##0 = svmla_n_f32_x(pg,  a##M_IDX##0, b0_2, av2); \
        a##M_IDX##1 = svmla_n_f32_x(pg1, a##M_IDX##1, b1_2, av2); \
        a##M_IDX##2 = svmla_n_f32_x(pg2, a##M_IDX##2, b2_2, av2); \
        a##M_IDX##0 = svmla_n_f32_x(pg,  a##M_IDX##0, b0_3, av3); \
        a##M_IDX##1 = svmla_n_f32_x(pg1, a##M_IDX##1, b1_3, av3); \
        a##M_IDX##2 = svmla_n_f32_x(pg2, a##M_IDX##2, b2_3, av3); \
    } while (0)

#define FMA_ROW_1(M_IDX) \
    do { \
        float av0 = A[(size_t)(M_IDX) * lda + k]; \
        a##M_IDX##0 = svmla_n_f32_x(pg,  a##M_IDX##0, b0_0, av0); \
        a##M_IDX##1 = svmla_n_f32_x(pg1, a##M_IDX##1, b1_0, av0); \
        a##M_IDX##2 = svmla_n_f32_x(pg2, a##M_IDX##2, b2_0, av0); \
    } while (0)

static void microtile_8x3vl(
    int K,
    const float    *A,  int lda,
    const uint16_t *BT, int ldb,
    float          *C,  int ldc,
    int n_tail)
{
    const svbool_t pg = svptrue_b32();
    const int VL = (int)svcntw();

    svbool_t pg1 = pg, pg2 = pg;
    if (n_tail < 3 * VL) {
        int t1 = n_tail - VL;       if (t1 < 0) t1 = 0;
        int t2 = n_tail - 2 * VL;   if (t2 < 0) t2 = 0;
        pg1 = svwhilelt_b32_s32(0, t1);
        pg2 = svwhilelt_b32_s32(0, t2);
    }

    svfloat32_t a00 = svdup_f32(0.0f), a01 = svdup_f32(0.0f), a02 = svdup_f32(0.0f);
    svfloat32_t a10 = svdup_f32(0.0f), a11 = svdup_f32(0.0f), a12 = svdup_f32(0.0f);
    svfloat32_t a20 = svdup_f32(0.0f), a21 = svdup_f32(0.0f), a22 = svdup_f32(0.0f);
    svfloat32_t a30 = svdup_f32(0.0f), a31 = svdup_f32(0.0f), a32 = svdup_f32(0.0f);
    svfloat32_t a40 = svdup_f32(0.0f), a41 = svdup_f32(0.0f), a42 = svdup_f32(0.0f);
    svfloat32_t a50 = svdup_f32(0.0f), a51 = svdup_f32(0.0f), a52 = svdup_f32(0.0f);
    svfloat32_t a60 = svdup_f32(0.0f), a61 = svdup_f32(0.0f), a62 = svdup_f32(0.0f);
    svfloat32_t a70 = svdup_f32(0.0f), a71 = svdup_f32(0.0f), a72 = svdup_f32(0.0f);

    int K4 = (K / 4) * 4;
    int k = 0;
    for (; k < K4; k += 4) {
        svfloat32_t b0_0 = svld1_bf16_f32(pg,  BT + (size_t)(k + 0) * ldb);
        svfloat32_t b1_0 = svld1_bf16_f32(pg1, BT + (size_t)(k + 0) * ldb + VL);
        svfloat32_t b2_0 = svld1_bf16_f32(pg2, BT + (size_t)(k + 0) * ldb + 2 * VL);
        svfloat32_t b0_1 = svld1_bf16_f32(pg,  BT + (size_t)(k + 1) * ldb);
        svfloat32_t b1_1 = svld1_bf16_f32(pg1, BT + (size_t)(k + 1) * ldb + VL);
        svfloat32_t b2_1 = svld1_bf16_f32(pg2, BT + (size_t)(k + 1) * ldb + 2 * VL);
        svfloat32_t b0_2 = svld1_bf16_f32(pg,  BT + (size_t)(k + 2) * ldb);
        svfloat32_t b1_2 = svld1_bf16_f32(pg1, BT + (size_t)(k + 2) * ldb + VL);
        svfloat32_t b2_2 = svld1_bf16_f32(pg2, BT + (size_t)(k + 2) * ldb + 2 * VL);
        svfloat32_t b0_3 = svld1_bf16_f32(pg,  BT + (size_t)(k + 3) * ldb);
        svfloat32_t b1_3 = svld1_bf16_f32(pg1, BT + (size_t)(k + 3) * ldb + VL);
        svfloat32_t b2_3 = svld1_bf16_f32(pg2, BT + (size_t)(k + 3) * ldb + 2 * VL);

        FMA_ROW(0); FMA_ROW(1); FMA_ROW(2); FMA_ROW(3);
        FMA_ROW(4); FMA_ROW(5); FMA_ROW(6); FMA_ROW(7);
    }
    for (; k < K; k++) {
        svfloat32_t b0_0 = svld1_bf16_f32(pg,  BT + (size_t)k * ldb);
        svfloat32_t b1_0 = svld1_bf16_f32(pg1, BT + (size_t)k * ldb + VL);
        svfloat32_t b2_0 = svld1_bf16_f32(pg2, BT + (size_t)k * ldb + 2 * VL);
        FMA_ROW_1(0); FMA_ROW_1(1); FMA_ROW_1(2); FMA_ROW_1(3);
        FMA_ROW_1(4); FMA_ROW_1(5); FMA_ROW_1(6); FMA_ROW_1(7);
    }

    #define STORE_ROW(M_IDX) \
        do { \
            svst1_f32(pg,  C + (size_t)(M_IDX) * ldc,          a##M_IDX##0); \
            svst1_f32(pg1, C + (size_t)(M_IDX) * ldc + VL,     a##M_IDX##1); \
            svst1_f32(pg2, C + (size_t)(M_IDX) * ldc + 2 * VL, a##M_IDX##2); \
        } while (0)
    STORE_ROW(0); STORE_ROW(1); STORE_ROW(2); STORE_ROW(3);
    STORE_ROW(4); STORE_ROW(5); STORE_ROW(6); STORE_ROW(7);
    #undef STORE_ROW
}

#undef FMA_ROW
#undef FMA_ROW_1

static void row_tail_1xN(
    int K, int N,
    const float    *A,
    const uint16_t *BT, int ldb,
    float          *C)
{
    const svbool_t pg = svptrue_b32();
    const int VL = (int)svcntw();
    for (int n = 0; n < N; n += VL) {
        svbool_t p = (n + VL <= N) ? pg : svwhilelt_b32_s32(n, N);
        svfloat32_t acc = svdup_f32(0.0f);
        for (int k = 0; k < K; k++) {
            svfloat32_t b = svld1_bf16_f32(p, BT + (size_t)k * ldb + n);
            acc = svmla_n_f32_x(p, acc, b, A[k]);
        }
        svst1_f32(p, C + n, acc);
    }
}

void gemm_bf16_BT(int M, int K, int N,
                  const float    *A,  int lda,
                  const uint16_t *BT, int ldb,
                  float          *C,  int ldc)
{
    const int VL = (int)svcntw();
    const int mr = 8;
    const int nr = 3 * VL;

    int M8 = (M / mr) * mr;
    int N_blocks = (N + nr - 1) / nr;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int m0 = 0; m0 < M8; m0 += mr) {
        for (int nb = 0; nb < N_blocks; nb++) {
            int n0 = nb * nr;
            int n_tail = (n0 + nr <= N) ? nr : (N - n0);
            microtile_8x3vl(K,
                            A  + (size_t)m0 * lda, lda,
                            BT + (size_t)n0,       ldb,
                            C  + (size_t)m0 * ldc + n0, ldc,
                            n_tail);
        }
    }

    if (M8 < M) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int m = M8; m < M; m++) {
            row_tail_1xN(K, N, A + (size_t)m * lda, BT, ldb, C + (size_t)m * ldc);
        }
    }
}

/* ────────────────────────────────────────────────────────────────────────
 * Packed-B BF16 GEMM (uses asm microkernel micro_kernel_bf16B_8x3_unroll4)
 * ──────────────────────────────────────────────────────────────────────── */

extern size_t packed_A_size(int M, int K);   /* from pack_matrices.c */

extern void micro_kernel_bf16B_8x3_unroll4(
    const float    *A_packed,
    const uint16_t *B_packed_bf16,
    float          *C,
    int64_t K,
    int64_t unused,
    int64_t ldc_bytes);

extern void micro_kernel_bf16B_8x3_unroll4_pv(
    const float    *A_packed,
    const uint16_t *B_packed_pv,
    float          *C,
    int64_t K,
    int64_t unused,
    int64_t ldc_bytes);

size_t packed_B_bf16_size(int K, int N) {
    int N_blocks  = (N + NR - 1) / NR;
    int K_rounded = ((K + 3) / 4) * 4;
    return (size_t)N_blocks * K_rounded * NR * sizeof(uint16_t);
}

void pack_B_bf16(int K, int N,
                 const uint16_t *BT, int ldb,
                 uint16_t *BTP)
{
    const int N_blocks  = (N + NR - 1) / NR;
    const int K_rounded = ((K + 3) / 4) * 4;
    for (int nb = 0; nb < N_blocks; nb++) {
        int n_start = nb * NR;
        int n_count = (n_start + NR <= N) ? NR : N - n_start;
        uint16_t *dst = BTP + (size_t)nb * K_rounded * NR;
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < n_count; n++) {
                dst[(size_t)k * NR + n] = BT[(size_t)k * ldb + n_start + n];
            }
            for (int n = n_count; n < NR; n++) dst[(size_t)k * NR + n] = 0;
        }
        for (int k = K; k < K_rounded; k++) {
            for (int n = 0; n < NR; n++) dst[(size_t)k * NR + n] = 0;
        }
    }
}

void gemm_bf16_BTP(int M, int K, int N,
                   const float    *A,    int lda,
                   const uint16_t *BTP,
                   float          *C,    int ldc)
{
    const int K_rounded = ((K + 3) / 4) * 4;
    const int M_blocks  = (M + MR - 1) / MR;
    const int N_blocks  = (N + NR - 1) / NR;

    size_t A_packed_bytes = packed_A_size(M, K);
    float *A_packed = pack_A_get_scratch(A_packed_bytes);
    if (!A_packed) {
        fprintf(stderr, "gemm_bf16_BTP: failed to alloc A_packed (%zu bytes)\n",
                A_packed_bytes);
        return;
    }
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        /* Parallel A-pack + C-zero, then compute — one region so the
         * pack/zero scale with the thread count instead of running
         * serially before the GEMM. */
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (int mb = 0; mb < M_blocks; mb++)
            pack_A_fp32_block(mb, M, K, K_rounded, A, lda, A_packed);
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int m = 0; m < M; m++) memset(C + (size_t)m * ldc, 0, N * sizeof(float));

#ifdef _OPENMP
        #pragma omp for collapse(2) schedule(static)
#endif
        for (int mb = 0; mb < M_blocks; mb++) {
            for (int nb = 0; nb < N_blocks; nb++) {
                int m_start = mb * MR;
                int n_start = nb * NR;
                int m_count = (m_start + MR <= M) ? MR : M - m_start;
                int n_count = (n_start + NR <= N) ? NR : N - n_start;

                const float    *A_tile = A_packed + (size_t)mb * K_rounded * MR;
                const uint16_t *B_tile = BTP      + (size_t)nb * K_rounded * NR;

                if (m_count == MR && n_count == NR) {
                    float *C_tile = C + (size_t)m_start * ldc + n_start;
                    micro_kernel_bf16B_8x3_unroll4(
                        A_tile, B_tile, C_tile,
                        (int64_t)K_rounded, 0,
                        (int64_t)ldc * sizeof(float));
                } else {
                    float local_buf[MR * NR] __attribute__((aligned(64)));
                    micro_kernel_bf16B_8x3_unroll4(
                        A_tile, B_tile, local_buf,
                        (int64_t)K_rounded, 0,
                        (int64_t)NR * sizeof(float));
                    for (int m = 0; m < m_count; m++) {
                        for (int n = 0; n < n_count; n++) {
                            C[(size_t)(m_start + m) * ldc + n_start + n] =
                                local_buf[m * NR + n];
                        }
                    }
                }
            }
        }
    } /* omp parallel */

    /* A_packed is a persistent grow-only scratch — do not free here. */
}

/* ────────────────────────────────────────────────────────────────────────
 * Pair-interleaved variant (uses micro_kernel_bf16B_8x3_unroll4_pv).
 *
 * Same MR×NR microtile; B repacked so two consecutive K-rows are
 * interleaved at halfword granularity within each 16-col chunk. The
 * asm kernel reads BF16 directly into the upper 16 bits of FP32 lanes
 * via ld1h{.h} + p_odd — no LSL on the FLA pipe.
 *
 * The pv allocation reserves a 64-byte prefix so the asm kernel's
 * "base - 2 bytes" k_even load on the first chunk of every N-block
 * dereferences valid memory. The caller passes the allocation base;
 * we add PV_PREFIX_BYTES internally.
 * ──────────────────────────────────────────────────────────────────────── */

#define PV_PREFIX_BYTES 64

size_t packed_B_bf16_pv_size(int K, int N) {
    return packed_B_bf16_size(K, N) + PV_PREFIX_BYTES;
}

/* Pair-interleaved BTP. Per N-block, K processed in pairs of 2.
 * Per K-pair: 3 chunks × 32 HW. Within chunk c (cols 16c..16c+15):
 *   HW[2i + 0] = BT[k_even][16c + i]
 *   HW[2i + 1] = BT[k_odd ][16c + i]
 * BTP_alloc is the allocation base; the layout starts at
 * BTP_alloc + PV_PREFIX_BYTES. Bytes 0..PV_PREFIX_BYTES-1 are zeroed. */
void pack_B_bf16_pv(int K, int N,
                    const uint16_t *BT, int ldb,
                    uint16_t *BTP_alloc)
{
    memset(BTP_alloc, 0, PV_PREFIX_BYTES);
    uint16_t *BTP = (uint16_t *)((uint8_t *)BTP_alloc + PV_PREFIX_BYTES);

    const int N_blocks  = (N + NR - 1) / NR;
    const int K_rounded = ((K + 3) / 4) * 4;
    for (int nb = 0; nb < N_blocks; nb++) {
        int n_start = nb * NR;
        int n_count = (n_start + NR <= N) ? NR : N - n_start;
        uint16_t *dst = BTP + (size_t)nb * K_rounded * NR;
        for (int kp = 0; kp < K_rounded; kp += 2) {
            int k0 = kp, k1 = kp + 1;
            for (int c = 0; c < 3; c++) {
                uint16_t *chunk = dst + (size_t)(kp / 2) * (NR * 2) + c * 32;
                for (int i = 0; i < 16; i++) {
                    int col = c * 16 + i;
                    uint16_t v0 = 0, v1 = 0;
                    if (k0 < K && col < n_count)
                        v0 = BT[(size_t)k0 * ldb + n_start + col];
                    if (k1 < K && col < n_count)
                        v1 = BT[(size_t)k1 * ldb + n_start + col];
                    chunk[2 * i + 0] = v0;
                    chunk[2 * i + 1] = v1;
                }
            }
        }
    }
}

void gemm_bf16_BTP_pv(int M, int K, int N,
                      const float    *A,    int lda,
                      const uint16_t *BTP_alloc,
                      float          *C,    int ldc)
{
    const uint16_t *BTP =
        (const uint16_t *)((const uint8_t *)BTP_alloc + PV_PREFIX_BYTES);

    const int K_rounded = ((K + 3) / 4) * 4;
    const int M_blocks  = (M + MR - 1) / MR;
    const int N_blocks  = (N + NR - 1) / NR;

    size_t A_packed_bytes = packed_A_size(M, K);
    float *A_packed = pack_A_get_scratch(A_packed_bytes);
    if (!A_packed) {
        fprintf(stderr, "gemm_bf16_BTP_pv: failed to alloc A_packed (%zu bytes)\n",
                A_packed_bytes);
        return;
    }
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (int mb = 0; mb < M_blocks; mb++)
            pack_A_fp32_block(mb, M, K, K_rounded, A, lda, A_packed);
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int m = 0; m < M; m++) memset(C + (size_t)m * ldc, 0, N * sizeof(float));

#ifdef _OPENMP
        #pragma omp for collapse(2) schedule(static)
#endif
        for (int mb = 0; mb < M_blocks; mb++) {
            for (int nb = 0; nb < N_blocks; nb++) {
                int m_start = mb * MR;
                int n_start = nb * NR;
                int m_count = (m_start + MR <= M) ? MR : M - m_start;
                int n_count = (n_start + NR <= N) ? NR : N - n_start;

                const float    *A_tile = A_packed + (size_t)mb * K_rounded * MR;
                const uint16_t *B_tile = BTP      + (size_t)nb * K_rounded * NR;

                if (m_count == MR && n_count == NR) {
                    float *C_tile = C + (size_t)m_start * ldc + n_start;
                    micro_kernel_bf16B_8x3_unroll4_pv(
                        A_tile, B_tile, C_tile,
                        (int64_t)K_rounded, 0,
                        (int64_t)ldc * sizeof(float));
                } else {
                    float local_buf[MR * NR] __attribute__((aligned(64)));
                    micro_kernel_bf16B_8x3_unroll4_pv(
                        A_tile, B_tile, local_buf,
                        (int64_t)K_rounded, 0,
                        (int64_t)NR * sizeof(float));
                    for (int m = 0; m < m_count; m++) {
                        for (int n = 0; n < n_count; n++) {
                            C[(size_t)(m_start + m) * ldc + n_start + n] =
                                local_buf[m * NR + n];
                        }
                    }
                }
            }
        }
    }
}

void gemm_bf16_BTP_cmg_pv(int M, int K, int N,
                          const float    *A, int lda,
                          const uint16_t * const BTP_alloc_repl[],
                          int n_cmgs,
                          float          *C, int ldc)
{
    if (n_cmgs <= 0 || !BTP_alloc_repl || !BTP_alloc_repl[0]) {
        gemm_bf16_BTP_pv(M, K, N, A, lda,
                         BTP_alloc_repl ? BTP_alloc_repl[0] : NULL, C, ldc);
        return;
    }

    const int K_rounded = ((K + 3) / 4) * 4;
    const int M_blocks  = (M + MR - 1) / MR;
    const int N_blocks  = (N + NR - 1) / NR;

    size_t A_packed_bytes = packed_A_size(M, K);
    float *A_packed = pack_A_get_scratch(A_packed_bytes);
    if (!A_packed) {
        fprintf(stderr, "gemm_bf16_BTP_cmg_pv: failed to alloc A_packed (%zu bytes)\n",
                A_packed_bytes);
        return;
    }
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int tid  = omp_get_thread_num();
        int nthr = omp_get_num_threads();
#else
        int tid = 0, nthr = 1;
#endif
        int my_cmg = (nthr > 0) ? (tid * n_cmgs / nthr) : 0;
        if (my_cmg >= n_cmgs) my_cmg = n_cmgs - 1;
        const uint16_t *alloc_local = BTP_alloc_repl[my_cmg];
        if (!alloc_local) alloc_local = BTP_alloc_repl[0];
        const uint16_t *BTP_local =
            (const uint16_t *)((const uint8_t *)alloc_local + PV_PREFIX_BYTES);

#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (int mb = 0; mb < M_blocks; mb++)
            pack_A_fp32_block(mb, M, K, K_rounded, A, lda, A_packed);
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int m = 0; m < M; m++) memset(C + (size_t)m * ldc, 0, N * sizeof(float));

#ifdef _OPENMP
        #pragma omp for collapse(2) schedule(static)
#endif
        for (int mb = 0; mb < M_blocks; mb++) {
            for (int nb = 0; nb < N_blocks; nb++) {
                int m_start = mb * MR;
                int n_start = nb * NR;
                int m_count = (m_start + MR <= M) ? MR : M - m_start;
                int n_count = (n_start + NR <= N) ? NR : N - n_start;

                const float    *A_tile = A_packed + (size_t)mb * K_rounded * MR;
                const uint16_t *B_tile = BTP_local + (size_t)nb * K_rounded * NR;

                if (m_count == MR && n_count == NR) {
                    float *C_tile = C + (size_t)m_start * ldc + n_start;
                    micro_kernel_bf16B_8x3_unroll4_pv(
                        A_tile, B_tile, C_tile,
                        (int64_t)K_rounded, 0,
                        (int64_t)ldc * sizeof(float));
                } else {
                    float local_buf[MR * NR] __attribute__((aligned(64)));
                    micro_kernel_bf16B_8x3_unroll4_pv(
                        A_tile, B_tile, local_buf,
                        (int64_t)K_rounded, 0,
                        (int64_t)NR * sizeof(float));
                    for (int m = 0; m < m_count; m++) {
                        for (int n = 0; n < n_count; n++) {
                            C[(size_t)(m_start + m) * ldc + n_start + n] =
                                local_buf[m * NR + n];
                        }
                    }
                }
            }
        }
    }
}

/* CMG-aware twin of gemm_bf16_BTP (see gemm_fp16_BTP_cmg comment). */
void gemm_bf16_BTP_cmg(int M, int K, int N,
                       const float    *A, int lda,
                       const uint16_t * const BTP_repl[],
                       int n_cmgs,
                       float          *C, int ldc)
{
    if (n_cmgs <= 0 || !BTP_repl || !BTP_repl[0]) {
        gemm_bf16_BTP(M, K, N, A, lda, BTP_repl ? BTP_repl[0] : NULL, C, ldc);
        return;
    }

    const int K_rounded = ((K + 3) / 4) * 4;
    const int M_blocks  = (M + MR - 1) / MR;
    const int N_blocks  = (N + NR - 1) / NR;

    size_t A_packed_bytes = packed_A_size(M, K);
    float *A_packed = pack_A_get_scratch(A_packed_bytes);
    if (!A_packed) {
        fprintf(stderr, "gemm_bf16_BTP_cmg: failed to alloc A_packed (%zu bytes)\n",
                A_packed_bytes);
        return;
    }
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int tid  = omp_get_thread_num();
        int nthr = omp_get_num_threads();
#else
        int tid = 0, nthr = 1;
#endif
        int my_cmg = (nthr > 0) ? (tid * n_cmgs / nthr) : 0;
        if (my_cmg >= n_cmgs) my_cmg = n_cmgs - 1;
        const uint16_t *BTP_local = BTP_repl[my_cmg];
        if (!BTP_local) BTP_local = BTP_repl[0];

        /* Parallel A-pack + C-zero before the compute loop. */
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (int mb = 0; mb < M_blocks; mb++)
            pack_A_fp32_block(mb, M, K, K_rounded, A, lda, A_packed);
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int m = 0; m < M; m++) memset(C + (size_t)m * ldc, 0, N * sizeof(float));

#ifdef _OPENMP
        #pragma omp for collapse(2) schedule(static)
#endif
        for (int mb = 0; mb < M_blocks; mb++) {
            for (int nb = 0; nb < N_blocks; nb++) {
                int m_start = mb * MR;
                int n_start = nb * NR;
                int m_count = (m_start + MR <= M) ? MR : M - m_start;
                int n_count = (n_start + NR <= N) ? NR : N - n_start;

                const float    *A_tile = A_packed + (size_t)mb * K_rounded * MR;
                const uint16_t *B_tile = BTP_local + (size_t)nb * K_rounded * NR;

                if (m_count == MR && n_count == NR) {
                    float *C_tile = C + (size_t)m_start * ldc + n_start;
                    micro_kernel_bf16B_8x3_unroll4(
                        A_tile, B_tile, C_tile,
                        (int64_t)K_rounded, 0,
                        (int64_t)ldc * sizeof(float));
                } else {
                    float local_buf[MR * NR] __attribute__((aligned(64)));
                    micro_kernel_bf16B_8x3_unroll4(
                        A_tile, B_tile, local_buf,
                        (int64_t)K_rounded, 0,
                        (int64_t)NR * sizeof(float));
                    for (int m = 0; m < m_count; m++) {
                        for (int n = 0; n < n_count; n++) {
                            C[(size_t)(m_start + m) * ldc + n_start + n] =
                                local_buf[m * NR + n];
                        }
                    }
                }
            }
        }
    }

    /* A_packed is a persistent grow-only scratch — do not free here. */
}
