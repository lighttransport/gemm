/* fp32_gemm.c — A64FX FP32-storage SVE GEMM with pre-packed B (BTP layout).
 *
 * Mirror of gemm_fp16_BTP / gemm_bf16_BTP. B is pre-packed into
 *   [N_blocks][K_rounded][NR]
 * once at cache build time, so the per-call cost collapses to:
 *   - parallel pack_A_fp32 (broadcast format, MR-block partitioned)
 *   - parallel memset(C)
 *   - parallel compute via micro_kernel_fp32_8x3_unroll4
 * all inside a single `#pragma omp parallel` region so no fork/join between
 * the three phases.
 *
 * Layout is bit-for-bit identical to what pack_B_fp32 already produces; we
 * just reuse that buffer permanently instead of allocating & repacking on
 * every GEMM call (the M6.2 fix that lifted the fp16 path was missing for
 * fp32, so fp32 was 5-13× slower than fp16 — purely due to B-pack overhead).
 */
#include "fp32_gemm.h"
#include "fused_gemm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

extern void micro_kernel_fp32_8x3_unroll4(
    const float *A_packed,
    const float *B_packed,
    float       *C,
    int64_t      K,
    int64_t      unused,
    int64_t      ldc_bytes);

extern size_t packed_A_size(int M, int K);
extern size_t packed_B_size(int K, int N);  /* same layout for FP32 BTP */

size_t packed_B_fp32_size(int K, int N) {
    return packed_B_size(K, N);
}

void gemm_fp32_BTP(int M, int K, int N,
                   const float *A,   int lda,
                   const float *BTP,
                   float       *C,   int ldc)
{
    const int K_rounded = ((K + 3) / 4) * 4;
    const int M_blocks  = (M + MR - 1) / MR;
    const int N_blocks  = (N + NR - 1) / NR;

    size_t A_packed_bytes = packed_A_size(M, K);
    float *A_packed = pack_A_get_scratch(A_packed_bytes);
    if (!A_packed) {
        fprintf(stderr, "gemm_fp32_BTP: failed to alloc A_packed (%zu bytes)\n",
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

                const float *A_tile = A_packed + (size_t)mb * K_rounded * MR;
                const float *B_tile = BTP      + (size_t)nb * K_rounded * NR;

                if (m_count == MR && n_count == NR) {
                    float *C_tile = C + (size_t)m_start * ldc + n_start;
                    micro_kernel_fp32_8x3_unroll4(
                        A_tile, B_tile, C_tile,
                        (int64_t)K_rounded, 0,
                        (int64_t)ldc * sizeof(float));
                } else {
                    float local_buf[MR * NR] __attribute__((aligned(64)));
                    micro_kernel_fp32_8x3_unroll4(
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

void gemm_fp32_BTP_cmg(int M, int K, int N,
                       const float *A, int lda,
                       const float * const BTP_repl[],
                       int n_cmgs,
                       float       *C, int ldc)
{
    if (n_cmgs <= 0 || !BTP_repl || !BTP_repl[0]) {
        gemm_fp32_BTP(M, K, N, A, lda, BTP_repl ? BTP_repl[0] : NULL, C, ldc);
        return;
    }

    const int K_rounded = ((K + 3) / 4) * 4;
    const int M_blocks  = (M + MR - 1) / MR;
    const int N_blocks  = (N + NR - 1) / NR;

    size_t A_packed_bytes = packed_A_size(M, K);
    float *A_packed = pack_A_get_scratch(A_packed_bytes);
    if (!A_packed) {
        fprintf(stderr, "gemm_fp32_BTP_cmg: failed to alloc A_packed (%zu bytes)\n",
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
        const float *BTP_local = BTP_repl[my_cmg];
        if (!BTP_local) BTP_local = BTP_repl[0];

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

                const float *A_tile = A_packed  + (size_t)mb * K_rounded * MR;
                const float *B_tile = BTP_local + (size_t)nb * K_rounded * NR;

                if (m_count == MR && n_count == NR) {
                    float *C_tile = C + (size_t)m_start * ldc + n_start;
                    micro_kernel_fp32_8x3_unroll4(
                        A_tile, B_tile, C_tile,
                        (int64_t)K_rounded, 0,
                        (int64_t)ldc * sizeof(float));
                } else {
                    float local_buf[MR * NR] __attribute__((aligned(64)));
                    micro_kernel_fp32_8x3_unroll4(
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
