/* fp16_gemm.c — A64FX FP16-storage, FP32-compute SVE GEMM.
 *
 * Same microtile shape as bf16_gemm (8x48, K-unroll 4). The only difference
 * is the load: LD1H + FCVT Z.S, Pg/M, Z.H (saturating fp16→fp32) vs LSL #16.
 *
 * NB: A64FX FP16 denormals trap to microcode (~100+ cy/op). Caller must
 * call set_fpcr_fz16() once before the encode loop (vit_a64fx_encode already
 * does this guard via sve_math.h when fp16 is engaged).
 */
#include "fp16_gemm.h"
#include "fused_gemm.h"   /* MR, MR=8, NR=48, pack_A_fp32, packed_A_size */

#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── FP32 → FP16 via SVE FCVT (saturating, RNE) ──────────────────────── */
void f32_to_fp16_buf(const float *src, uint16_t *dst, size_t n) {
    const svbool_t pg = svptrue_b32();
    size_t i = 0;
    while (i < n) {
        svbool_t p = svwhilelt_b32_u64(i, (uint64_t)n);
        svfloat32_t v = svld1_f32(p, src + i);
        svfloat16_t h = svcvt_f16_f32_x(p, v);
        svst1h_u32(p, dst + i, svreinterpret_u32_f16(h));
        i += (size_t)svcntw();
    }
    (void)pg;
}

/* ── FP16 → FP32 load: LD1H {Z.S} + FCVT Z.S, Pg/M, Z.H ──────────────── */
static inline svfloat32_t svld1_fp16_f32(svbool_t pg, const uint16_t *p) {
    svuint32_t raw = svld1uh_u32(pg, p);
    return svcvt_f32_f16_x(pg, svreinterpret_f16_u32(raw));
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
        int t1 = n_tail - VL;     if (t1 < 0) t1 = 0;
        int t2 = n_tail - 2 * VL; if (t2 < 0) t2 = 0;
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
        svfloat32_t b0_0 = svld1_fp16_f32(pg,  BT + (size_t)(k + 0) * ldb);
        svfloat32_t b1_0 = svld1_fp16_f32(pg1, BT + (size_t)(k + 0) * ldb + VL);
        svfloat32_t b2_0 = svld1_fp16_f32(pg2, BT + (size_t)(k + 0) * ldb + 2 * VL);
        svfloat32_t b0_1 = svld1_fp16_f32(pg,  BT + (size_t)(k + 1) * ldb);
        svfloat32_t b1_1 = svld1_fp16_f32(pg1, BT + (size_t)(k + 1) * ldb + VL);
        svfloat32_t b2_1 = svld1_fp16_f32(pg2, BT + (size_t)(k + 1) * ldb + 2 * VL);
        svfloat32_t b0_2 = svld1_fp16_f32(pg,  BT + (size_t)(k + 2) * ldb);
        svfloat32_t b1_2 = svld1_fp16_f32(pg1, BT + (size_t)(k + 2) * ldb + VL);
        svfloat32_t b2_2 = svld1_fp16_f32(pg2, BT + (size_t)(k + 2) * ldb + 2 * VL);
        svfloat32_t b0_3 = svld1_fp16_f32(pg,  BT + (size_t)(k + 3) * ldb);
        svfloat32_t b1_3 = svld1_fp16_f32(pg1, BT + (size_t)(k + 3) * ldb + VL);
        svfloat32_t b2_3 = svld1_fp16_f32(pg2, BT + (size_t)(k + 3) * ldb + 2 * VL);

        FMA_ROW(0); FMA_ROW(1); FMA_ROW(2); FMA_ROW(3);
        FMA_ROW(4); FMA_ROW(5); FMA_ROW(6); FMA_ROW(7);
    }
    for (; k < K; k++) {
        svfloat32_t b0_0 = svld1_fp16_f32(pg,  BT + (size_t)k * ldb);
        svfloat32_t b1_0 = svld1_fp16_f32(pg1, BT + (size_t)k * ldb + VL);
        svfloat32_t b2_0 = svld1_fp16_f32(pg2, BT + (size_t)k * ldb + 2 * VL);
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
            svfloat32_t b = svld1_fp16_f32(p, BT + (size_t)k * ldb + n);
            acc = svmla_n_f32_x(p, acc, b, A[k]);
        }
        svst1_f32(p, C + n, acc);
    }
}

void gemm_fp16_BT(int M, int K, int N,
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
 * Packed-B FP16 GEMM (uses asm microkernel micro_kernel_fp16B_8x3_unroll4)
 * ──────────────────────────────────────────────────────────────────────── */

extern size_t packed_A_size(int M, int K);   /* from pack_matrices.c */

extern void micro_kernel_fp16B_8x3_unroll4(
    const float    *A_packed,
    const uint16_t *B_packed_fp16,
    float          *C,
    int64_t K,
    int64_t unused,
    int64_t ldc_bytes);

size_t packed_B_fp16_size(int K, int N) {
    int N_blocks  = (N + NR - 1) / NR;
    int K_rounded = ((K + 3) / 4) * 4;
    return (size_t)N_blocks * K_rounded * NR * sizeof(uint16_t);
}

/* Pack BT[K,N] fp16 row-major (ldb=N) into [N_blocks][K_rounded][NR] fp16.
 * Layout mirrors pack_B_fp32; partial NR-tail and K-tail are zero-padded. */
void pack_B_fp16(int K, int N,
                 const uint16_t *BT, int ldb,
                 uint16_t *BTP)
{
    const int N_blocks  = (N + NR - 1) / NR;
    const int K_rounded = ((K + 3) / 4) * 4;
    for (int nb = 0; nb < N_blocks; nb++) {
        int n_start = nb * NR;
        int n_count = (n_start + NR < N) ? NR : N - n_start;
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

/* GEMM C[M,N] = A[M,K] @ BT^T (i.e. BT was packed from row-major BT[K,N]).
 * A is FP32, BTP is the pre-packed FP16, C is FP32. Initializes C to zero. */
void gemm_fp16_BTP(int M, int K, int N,
                   const float    *A,    int lda,
                   const uint16_t *BTP,
                   float          *C,    int ldc)
{
    const int K_rounded = ((K + 3) / 4) * 4;
    const int M_blocks  = (M + MR - 1) / MR;
    const int N_blocks  = (N + NR - 1) / NR;

    /* Pack A per call (broadcast format). */
    size_t A_packed_bytes = packed_A_size(M, K);
    float *A_packed = pack_A_get_scratch(A_packed_bytes);
    if (!A_packed) {
        fprintf(stderr, "gemm_fp16_BTP: failed to alloc A_packed (%zu bytes)\n",
                A_packed_bytes);
        return;
    }

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        /* Parallel A-pack + C-zero, then the compute loop — all in one
         * region so the pack/zero scale with the thread count instead of
         * running serially before the GEMM. */
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
                micro_kernel_fp16B_8x3_unroll4(
                    A_tile, B_tile, C_tile,
                    (int64_t)K_rounded, 0,
                    (int64_t)ldc * sizeof(float));
            } else {
                /* Edge tile: kernel writes a full MR×NR block; spill to a
                 * local buffer then copy the valid portion. */
                float local_buf[MR * NR] __attribute__((aligned(64)));
                micro_kernel_fp16B_8x3_unroll4(
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

/* CMG-aware twin of gemm_fp16_BTP — same compute, but each OMP thread reads
 * from its CMG-local replica of the packed weights to avoid cross-CMG ring
 * traffic. Thread→CMG mapping is `tid * n_cmgs / nthreads` (caller pins). */
void gemm_fp16_BTP_cmg(int M, int K, int N,
                       const float    *A, int lda,
                       const uint16_t * const BTP_repl[],
                       int n_cmgs,
                       float          *C, int ldc)
{
    if (n_cmgs <= 0 || !BTP_repl || !BTP_repl[0]) {
        /* Fallback to single-replica path. */
        gemm_fp16_BTP(M, K, N, A, lda, BTP_repl ? BTP_repl[0] : NULL, C, ldc);
        return;
    }

    const int K_rounded = ((K + 3) / 4) * 4;
    const int M_blocks  = (M + MR - 1) / MR;
    const int N_blocks  = (N + NR - 1) / NR;

    size_t A_packed_bytes = packed_A_size(M, K);
    float *A_packed = pack_A_get_scratch(A_packed_bytes);
    if (!A_packed) {
        fprintf(stderr, "gemm_fp16_BTP_cmg: failed to alloc A_packed (%zu bytes)\n",
                A_packed_bytes);
        return;
    }
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int tid     = omp_get_thread_num();
        int nthr    = omp_get_num_threads();
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
                    micro_kernel_fp16B_8x3_unroll4(
                        A_tile, B_tile, C_tile,
                        (int64_t)K_rounded, 0,
                        (int64_t)ldc * sizeof(float));
                } else {
                    float local_buf[MR * NR] __attribute__((aligned(64)));
                    micro_kernel_fp16B_8x3_unroll4(
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
