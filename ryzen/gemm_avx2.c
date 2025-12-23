/*
 * AVX2/FMA optimized GEMM for AMD Zen2 (Ryzen 9 3950X)
 *
 * Optimization strategies:
 *   - 6x16 micro-kernel (6 rows x 16 cols, using 12 YMM registers for C)
 *   - Register blocking to maximize FMA throughput
 *   - Cache blocking for L1/L2 efficiency
 *   - Prefetching for memory latency hiding
 *
 * Register allocation for 6x16 micro-kernel:
 *   - ymm0-ymm11: 6x2 = 12 accumulators for C tile
 *   - ymm12-ymm13: B values (2 x 8 floats = 16 columns)
 *   - ymm14-ymm15: A values (broadcast)
 */

#include "gemm_avx2.h"
#include <immintrin.h>
#include <string.h>
#include <stdlib.h>

/* Micro-kernel tile sizes */
#define MR 6   /* rows of A processed per micro-kernel */
#define NR 16  /* cols of B processed per micro-kernel (2 YMM registers) */

/* Cache blocking parameters tuned for Zen2 */
#define MC 72   /* rows of A per L2 block (multiple of MR) */
#define NC 256  /* cols of B per L3 block (multiple of NR) */
#define KC 256  /* depth of A/B panels */

/* Alignment for AVX2 (32 bytes) */
#define ALIGN_SIZE 32

static inline void *aligned_malloc(size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, ALIGN_SIZE, size) != 0) {
        return NULL;
    }
    return ptr;
}

/*
 * 6x16 micro-kernel: C[6x16] += A[6xK] * B[Kx16]
 *
 * This is the innermost computation. Uses all 16 YMM registers:
 *   - 12 registers for the 6x16 accumulator tile (6 rows x 2 YMM per row)
 *   - 2 registers for B panel columns
 *   - 2 registers for broadcasting A values
 */
void sgemm_kernel_6x16(
    size_t K,
    const float *A_panel,  /* 6 x K, packed column-major */
    const float *B_panel,  /* K x 16, packed row-major */
    float *C, size_t ldc,
    float alpha, float beta
) {
    /* Accumulators for 6x16 tile (6 rows, 2 YMM registers per row) */
    __m256 c00, c01;  /* row 0 */
    __m256 c10, c11;  /* row 1 */
    __m256 c20, c21;  /* row 2 */
    __m256 c30, c31;  /* row 3 */
    __m256 c40, c41;  /* row 4 */
    __m256 c50, c51;  /* row 5 */

    /* Initialize accumulators to zero */
    c00 = _mm256_setzero_ps(); c01 = _mm256_setzero_ps();
    c10 = _mm256_setzero_ps(); c11 = _mm256_setzero_ps();
    c20 = _mm256_setzero_ps(); c21 = _mm256_setzero_ps();
    c30 = _mm256_setzero_ps(); c31 = _mm256_setzero_ps();
    c40 = _mm256_setzero_ps(); c41 = _mm256_setzero_ps();
    c50 = _mm256_setzero_ps(); c51 = _mm256_setzero_ps();

    /* Main K loop - unroll by 4 for better instruction-level parallelism */
    size_t k = 0;
    for (; k + 3 < K; k += 4) {
        /* Prefetch next iteration's data */
        _mm_prefetch((const char *)(B_panel + 64), _MM_HINT_T0);
        _mm_prefetch((const char *)(A_panel + 24), _MM_HINT_T0);

        /* Iteration 0 */
        {
            __m256 b0 = _mm256_load_ps(B_panel);       /* B[k, 0:7] */
            __m256 b1 = _mm256_load_ps(B_panel + 8);   /* B[k, 8:15] */

            __m256 a0 = _mm256_broadcast_ss(A_panel + 0);
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);

            __m256 a1 = _mm256_broadcast_ss(A_panel + 1);
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);

            __m256 a2 = _mm256_broadcast_ss(A_panel + 2);
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);

            __m256 a3 = _mm256_broadcast_ss(A_panel + 3);
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);

            __m256 a4 = _mm256_broadcast_ss(A_panel + 4);
            c40 = _mm256_fmadd_ps(a4, b0, c40);
            c41 = _mm256_fmadd_ps(a4, b1, c41);

            __m256 a5 = _mm256_broadcast_ss(A_panel + 5);
            c50 = _mm256_fmadd_ps(a5, b0, c50);
            c51 = _mm256_fmadd_ps(a5, b1, c51);
        }

        /* Iteration 1 */
        {
            __m256 b0 = _mm256_load_ps(B_panel + 16);
            __m256 b1 = _mm256_load_ps(B_panel + 24);

            __m256 a0 = _mm256_broadcast_ss(A_panel + 6);
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);

            __m256 a1 = _mm256_broadcast_ss(A_panel + 7);
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);

            __m256 a2 = _mm256_broadcast_ss(A_panel + 8);
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);

            __m256 a3 = _mm256_broadcast_ss(A_panel + 9);
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);

            __m256 a4 = _mm256_broadcast_ss(A_panel + 10);
            c40 = _mm256_fmadd_ps(a4, b0, c40);
            c41 = _mm256_fmadd_ps(a4, b1, c41);

            __m256 a5 = _mm256_broadcast_ss(A_panel + 11);
            c50 = _mm256_fmadd_ps(a5, b0, c50);
            c51 = _mm256_fmadd_ps(a5, b1, c51);
        }

        /* Iteration 2 */
        {
            __m256 b0 = _mm256_load_ps(B_panel + 32);
            __m256 b1 = _mm256_load_ps(B_panel + 40);

            __m256 a0 = _mm256_broadcast_ss(A_panel + 12);
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);

            __m256 a1 = _mm256_broadcast_ss(A_panel + 13);
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);

            __m256 a2 = _mm256_broadcast_ss(A_panel + 14);
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);

            __m256 a3 = _mm256_broadcast_ss(A_panel + 15);
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);

            __m256 a4 = _mm256_broadcast_ss(A_panel + 16);
            c40 = _mm256_fmadd_ps(a4, b0, c40);
            c41 = _mm256_fmadd_ps(a4, b1, c41);

            __m256 a5 = _mm256_broadcast_ss(A_panel + 17);
            c50 = _mm256_fmadd_ps(a5, b0, c50);
            c51 = _mm256_fmadd_ps(a5, b1, c51);
        }

        /* Iteration 3 */
        {
            __m256 b0 = _mm256_load_ps(B_panel + 48);
            __m256 b1 = _mm256_load_ps(B_panel + 56);

            __m256 a0 = _mm256_broadcast_ss(A_panel + 18);
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);

            __m256 a1 = _mm256_broadcast_ss(A_panel + 19);
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);

            __m256 a2 = _mm256_broadcast_ss(A_panel + 20);
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);

            __m256 a3 = _mm256_broadcast_ss(A_panel + 21);
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);

            __m256 a4 = _mm256_broadcast_ss(A_panel + 22);
            c40 = _mm256_fmadd_ps(a4, b0, c40);
            c41 = _mm256_fmadd_ps(a4, b1, c41);

            __m256 a5 = _mm256_broadcast_ss(A_panel + 23);
            c50 = _mm256_fmadd_ps(a5, b0, c50);
            c51 = _mm256_fmadd_ps(a5, b1, c51);
        }

        A_panel += MR * 4;
        B_panel += NR * 4;
    }

    /* Handle remaining K iterations */
    for (; k < K; k++) {
        __m256 b0 = _mm256_load_ps(B_panel);
        __m256 b1 = _mm256_load_ps(B_panel + 8);

        __m256 a0 = _mm256_broadcast_ss(A_panel + 0);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        __m256 a1 = _mm256_broadcast_ss(A_panel + 1);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);

        __m256 a2 = _mm256_broadcast_ss(A_panel + 2);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);

        __m256 a3 = _mm256_broadcast_ss(A_panel + 3);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);

        __m256 a4 = _mm256_broadcast_ss(A_panel + 4);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);

        __m256 a5 = _mm256_broadcast_ss(A_panel + 5);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);

        A_panel += MR;
        B_panel += NR;
    }

    /* Apply alpha scaling */
    __m256 valpha = _mm256_set1_ps(alpha);
    c00 = _mm256_mul_ps(c00, valpha);
    c01 = _mm256_mul_ps(c01, valpha);
    c10 = _mm256_mul_ps(c10, valpha);
    c11 = _mm256_mul_ps(c11, valpha);
    c20 = _mm256_mul_ps(c20, valpha);
    c21 = _mm256_mul_ps(c21, valpha);
    c30 = _mm256_mul_ps(c30, valpha);
    c31 = _mm256_mul_ps(c31, valpha);
    c40 = _mm256_mul_ps(c40, valpha);
    c41 = _mm256_mul_ps(c41, valpha);
    c50 = _mm256_mul_ps(c50, valpha);
    c51 = _mm256_mul_ps(c51, valpha);

    /* Store results: C = alpha * A * B + beta * C */
    if (beta != 0.0f) {
        __m256 vbeta = _mm256_set1_ps(beta);

        /* Row 0 */
        __m256 old0 = _mm256_loadu_ps(C);
        __m256 old1 = _mm256_loadu_ps(C + 8);
        _mm256_storeu_ps(C, _mm256_fmadd_ps(vbeta, old0, c00));
        _mm256_storeu_ps(C + 8, _mm256_fmadd_ps(vbeta, old1, c01));

        /* Row 1 */
        old0 = _mm256_loadu_ps(C + ldc);
        old1 = _mm256_loadu_ps(C + ldc + 8);
        _mm256_storeu_ps(C + ldc, _mm256_fmadd_ps(vbeta, old0, c10));
        _mm256_storeu_ps(C + ldc + 8, _mm256_fmadd_ps(vbeta, old1, c11));

        /* Row 2 */
        old0 = _mm256_loadu_ps(C + 2*ldc);
        old1 = _mm256_loadu_ps(C + 2*ldc + 8);
        _mm256_storeu_ps(C + 2*ldc, _mm256_fmadd_ps(vbeta, old0, c20));
        _mm256_storeu_ps(C + 2*ldc + 8, _mm256_fmadd_ps(vbeta, old1, c21));

        /* Row 3 */
        old0 = _mm256_loadu_ps(C + 3*ldc);
        old1 = _mm256_loadu_ps(C + 3*ldc + 8);
        _mm256_storeu_ps(C + 3*ldc, _mm256_fmadd_ps(vbeta, old0, c30));
        _mm256_storeu_ps(C + 3*ldc + 8, _mm256_fmadd_ps(vbeta, old1, c31));

        /* Row 4 */
        old0 = _mm256_loadu_ps(C + 4*ldc);
        old1 = _mm256_loadu_ps(C + 4*ldc + 8);
        _mm256_storeu_ps(C + 4*ldc, _mm256_fmadd_ps(vbeta, old0, c40));
        _mm256_storeu_ps(C + 4*ldc + 8, _mm256_fmadd_ps(vbeta, old1, c41));

        /* Row 5 */
        old0 = _mm256_loadu_ps(C + 5*ldc);
        old1 = _mm256_loadu_ps(C + 5*ldc + 8);
        _mm256_storeu_ps(C + 5*ldc, _mm256_fmadd_ps(vbeta, old0, c50));
        _mm256_storeu_ps(C + 5*ldc + 8, _mm256_fmadd_ps(vbeta, old1, c51));
    } else {
        /* beta == 0: just store the results */
        _mm256_storeu_ps(C, c00);
        _mm256_storeu_ps(C + 8, c01);
        _mm256_storeu_ps(C + ldc, c10);
        _mm256_storeu_ps(C + ldc + 8, c11);
        _mm256_storeu_ps(C + 2*ldc, c20);
        _mm256_storeu_ps(C + 2*ldc + 8, c21);
        _mm256_storeu_ps(C + 3*ldc, c30);
        _mm256_storeu_ps(C + 3*ldc + 8, c31);
        _mm256_storeu_ps(C + 4*ldc, c40);
        _mm256_storeu_ps(C + 4*ldc + 8, c41);
        _mm256_storeu_ps(C + 5*ldc, c50);
        _mm256_storeu_ps(C + 5*ldc + 8, c51);
    }
}

/*
 * Pack A matrix into column-major panel format for efficient micro-kernel access
 * A[M x K] -> A_packed[MC x KC], stored as MR x KC strips
 */
static void pack_A(
    size_t M, size_t K,
    const float *A, size_t lda,
    float *A_packed
) {
    for (size_t i = 0; i < M; i += MR) {
        size_t mr = (i + MR <= M) ? MR : (M - i);
        for (size_t k = 0; k < K; k++) {
            for (size_t ii = 0; ii < mr; ii++) {
                *A_packed++ = A[(i + ii) * lda + k];
            }
            /* Pad with zeros if necessary */
            for (size_t ii = mr; ii < MR; ii++) {
                *A_packed++ = 0.0f;
            }
        }
    }
}

/*
 * Pack B matrix into row-major panel format for efficient micro-kernel access
 * B[K x N] -> B_packed[KC x NC], stored as KC x NR strips
 */
static void pack_B(
    size_t K, size_t N,
    const float *B, size_t ldb,
    float *B_packed
) {
    for (size_t j = 0; j < N; j += NR) {
        size_t nr = (j + NR <= N) ? NR : (N - j);
        for (size_t k = 0; k < K; k++) {
            for (size_t jj = 0; jj < nr; jj++) {
                *B_packed++ = B[k * ldb + j + jj];
            }
            /* Pad with zeros if necessary */
            for (size_t jj = nr; jj < NR; jj++) {
                *B_packed++ = 0.0f;
            }
        }
    }
}

/*
 * Handle edge cases for the micro-kernel when M or N is not a multiple of MR/NR
 */
static void sgemm_kernel_edge(
    size_t m, size_t n, size_t K,
    const float *A_panel,
    const float *B_panel,
    float *C, size_t ldc,
    float alpha, float beta
) {
    /* Use a temporary buffer for edge cases */
    float C_tmp[MR * NR] __attribute__((aligned(32)));

    /* Initialize temp buffer */
    memset(C_tmp, 0, sizeof(C_tmp));

    /* Run micro-kernel on temp buffer */
    sgemm_kernel_6x16(K, A_panel, B_panel, C_tmp, NR, 1.0f, 0.0f);

    /* Copy relevant portion to C with alpha/beta scaling */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            C[i * ldc + j] = alpha * C_tmp[i * NR + j] + beta * C[i * ldc + j];
        }
    }
}

/*
 * Main SGEMM routine with cache blocking
 * C = alpha * A * B + beta * C
 */
void sgemm_avx2(
    size_t M, size_t N, size_t K,
    float alpha,
    const float *A, size_t lda,
    const float *B, size_t ldb,
    float beta,
    float *C, size_t ldc
) {
    /* Allocate packing buffers */
    float *A_packed = (float *)aligned_malloc(MC * KC * sizeof(float));
    float *B_packed = (float *)aligned_malloc(KC * NC * sizeof(float));

    if (!A_packed || !B_packed) {
        /* Fallback to naive if allocation fails */
        free(A_packed);
        free(B_packed);
        sgemm_naive(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    /* Scale C by beta if needed (handle beta != 1 case) */
    if (beta != 1.0f) {
        if (beta == 0.0f) {
            for (size_t i = 0; i < M; i++) {
                memset(C + i * ldc, 0, N * sizeof(float));
            }
        } else {
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    C[i * ldc + j] *= beta;
                }
            }
        }
        beta = 1.0f;  /* Now we'll just accumulate */
    }

    /* Loop over N in blocks of NC */
    for (size_t jc = 0; jc < N; jc += NC) {
        size_t nc = (jc + NC <= N) ? NC : (N - jc);

        /* Loop over K in blocks of KC */
        for (size_t pc = 0; pc < K; pc += KC) {
            size_t kc = (pc + KC <= K) ? KC : (K - pc);

            /* Pack B panel: B[pc:pc+kc, jc:jc+nc] */
            pack_B(kc, nc, B + pc * ldb + jc, ldb, B_packed);

            /* Loop over M in blocks of MC */
            for (size_t ic = 0; ic < M; ic += MC) {
                size_t mc = (ic + MC <= M) ? MC : (M - ic);

                /* Pack A panel: A[ic:ic+mc, pc:pc+kc] */
                pack_A(mc, kc, A + ic * lda + pc, lda, A_packed);

                /* Compute C[ic:ic+mc, jc:jc+nc] += A_packed * B_packed */
                float *A_ptr = A_packed;
                for (size_t i = 0; i < mc; i += MR) {
                    size_t mr = (i + MR <= mc) ? MR : (mc - i);
                    float *B_ptr = B_packed;

                    for (size_t j = 0; j < nc; j += NR) {
                        size_t nr = (j + NR <= nc) ? NR : (nc - j);
                        float *C_ptr = C + (ic + i) * ldc + (jc + j);

                        if (mr == MR && nr == NR) {
                            /* Full micro-kernel */
                            sgemm_kernel_6x16(kc, A_ptr, B_ptr, C_ptr, ldc, alpha, 1.0f);
                        } else {
                            /* Edge case */
                            sgemm_kernel_edge(mr, nr, kc, A_ptr, B_ptr, C_ptr, ldc, alpha, 1.0f);
                        }

                        B_ptr += kc * NR;
                    }
                    A_ptr += kc * MR;
                }
            }
        }
    }

    free(A_packed);
    free(B_packed);
}

/*
 * Naive reference implementation for correctness verification
 */
void sgemm_naive(
    size_t M, size_t N, size_t K,
    float alpha,
    const float *A, size_t lda,
    const float *B, size_t ldb,
    float beta,
    float *C, size_t ldc
) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}
