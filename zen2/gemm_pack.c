#include "gemm_pack.h"
#include <string.h>
#include <immintrin.h>

/*
 * Optimized packing routines using SSE/AVX2 in-register transpose.
 *
 * pack_a: 4×4 SSE transpose for rows 0-3, paired 64-bit stores for rows 4-5
 * pack_b: 8×8 AVX2 transpose, processing 16 rows in two groups of 8
 */

/* ------------------------------------------------------------------ */
/*  8×8 float transpose using AVX2                                    */
/*  Input:  r0..r7 each contain 8 floats (one source row)             */
/*  Output: d0..d7 each contain 8 floats (one transposed column)      */
/* ------------------------------------------------------------------ */
static inline void transpose_8x8_ps(
    __m256 r0, __m256 r1, __m256 r2, __m256 r3,
    __m256 r4, __m256 r5, __m256 r6, __m256 r7,
    __m256 *d0, __m256 *d1, __m256 *d2, __m256 *d3,
    __m256 *d4, __m256 *d5, __m256 *d6, __m256 *d7)
{
    /* Phase 1: interleave pairs within 128-bit lanes */
    __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    /* Phase 2: interleave quads within 128-bit lanes */
    __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    /* Phase 3: cross 128-bit lane permute */
    *d0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    *d1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    *d2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    *d3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    *d4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    *d5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    *d6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    *d7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

/* ------------------------------------------------------------------ */
/*  pack_a: A[mc×K] row-major → column-panel A_pack[k*MR + m]        */
/*                                                                     */
/*  Full tile (mc=6): SSE 4×4 transpose for rows 0-3 + paired         */
/*  64-bit stores for rows 4-5.  Processes 4 K-values per iteration.  */
/* ------------------------------------------------------------------ */
void pack_a(const float *A, int lda, float *A_pack, int mc, int K)
{
    if (mc < MR) {
        /* Edge tile: scalar with zero-pad */
        memset(A_pack, 0, (size_t)MR * K * sizeof(float));
        for (int m = 0; m < mc; m++) {
            const float *src = A + (size_t)m * lda;
            for (int k = 0; k < K; k++) {
                A_pack[k * MR + m] = src[k];
            }
        }
        return;
    }

    /* Full tile: mc == MR == 6 */
    const float *r0 = A;
    const float *r1 = A + lda;
    const float *r2 = A + 2 * (size_t)lda;
    const float *r3 = A + 3 * (size_t)lda;
    const float *r4 = A + 4 * (size_t)lda;
    const float *r5 = A + 5 * (size_t)lda;

    int k = 0;
    for (; k + 4 <= K; k += 4) {
        /* Load 4 consecutive K-values from each of 6 rows */
        __m128 a0 = _mm_loadu_ps(r0 + k);
        __m128 a1 = _mm_loadu_ps(r1 + k);
        __m128 a2 = _mm_loadu_ps(r2 + k);
        __m128 a3 = _mm_loadu_ps(r3 + k);
        __m128 a4 = _mm_loadu_ps(r4 + k);
        __m128 a5 = _mm_loadu_ps(r5 + k);

        /* 4×4 transpose of rows 0-3 */
        __m128 t0 = _mm_unpacklo_ps(a0, a1); /* a0[0] a1[0] a0[1] a1[1] */
        __m128 t1 = _mm_unpackhi_ps(a0, a1); /* a0[2] a1[2] a0[3] a1[3] */
        __m128 t2 = _mm_unpacklo_ps(a2, a3); /* a2[0] a3[0] a2[1] a3[1] */
        __m128 t3 = _mm_unpackhi_ps(a2, a3); /* a2[2] a3[2] a2[3] a3[3] */

        /* Column vectors for rows 0-3 */
        __m128 c0 = _mm_movelh_ps(t0, t2);   /* a0[0] a1[0] a2[0] a3[0] */
        __m128 c1 = _mm_movehl_ps(t2, t0);   /* a0[1] a1[1] a2[1] a3[1] */
        __m128 c2 = _mm_movelh_ps(t1, t3);   /* a0[2] a1[2] a2[2] a3[2] */
        __m128 c3 = _mm_movehl_ps(t3, t1);   /* a0[3] a1[3] a2[3] a3[3] */

        /* Interleave rows 4-5 into pairs for 64-bit stores */
        __m128 p45lo = _mm_unpacklo_ps(a4, a5); /* a4[0] a5[0] a4[1] a5[1] */
        __m128 p45hi = _mm_unpackhi_ps(a4, a5); /* a4[2] a5[2] a4[3] a5[3] */

        float *dst = A_pack + (size_t)k * MR;

        /* K-step 0: [a0[0] a1[0] a2[0] a3[0]] + [a4[0] a5[0]] */
        _mm_storeu_ps(dst, c0);
        _mm_storel_pi((__m64 *)(dst + 4), p45lo);

        /* K-step 1: [a0[1] a1[1] a2[1] a3[1]] + [a4[1] a5[1]] */
        _mm_storeu_ps(dst + MR, c1);
        _mm_storeh_pi((__m64 *)(dst + MR + 4), p45lo);

        /* K-step 2: [a0[2] a1[2] a2[2] a3[2]] + [a4[2] a5[2]] */
        _mm_storeu_ps(dst + 2 * MR, c2);
        _mm_storel_pi((__m64 *)(dst + 2 * MR + 4), p45hi);

        /* K-step 3: [a0[3] a1[3] a2[3] a3[3]] + [a4[3] a5[3]] */
        _mm_storeu_ps(dst + 3 * MR, c3);
        _mm_storeh_pi((__m64 *)(dst + 3 * MR + 4), p45hi);
    }

    /* K remainder (0-3 iterations) */
    for (; k < K; k++) {
        float *dst = A_pack + (size_t)k * MR;
        dst[0] = r0[k];
        dst[1] = r1[k];
        dst[2] = r2[k];
        dst[3] = r3[k];
        dst[4] = r4[k];
        dst[5] = r5[k];
    }
}

/* ------------------------------------------------------------------ */
/*  pack_b: B[nc×K] row-major → row-panel B_pack[k*NR + n]           */
/*                                                                     */
/*  Full tile (nc=16): AVX2 8×8 transpose, two groups of 8 rows.      */
/*  Processes 8 K-values per iteration.                                */
/*  Output: each K-step is exactly one 64-byte cache line.             */
/* ------------------------------------------------------------------ */
void pack_b(const float *B, int ldb, float *B_pack, int nc, int K)
{
    if (nc < NR) {
        /* Edge tile: scalar with zero-pad */
        memset(B_pack, 0, (size_t)NR * K * sizeof(float));
        for (int n = 0; n < nc; n++) {
            const float *src = B + (size_t)n * ldb;
            for (int k = 0; k < K; k++) {
                B_pack[k * NR + n] = src[k];
            }
        }
        return;
    }

    /* Full tile: nc == NR == 16 */
    size_t s = (size_t)ldb;

    /* Group 1: rows 0-7 → B_pack[k*16 + 0..7] */
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 r0 = _mm256_loadu_ps(B + 0 * s + k);
        __m256 r1 = _mm256_loadu_ps(B + 1 * s + k);
        __m256 r2 = _mm256_loadu_ps(B + 2 * s + k);
        __m256 r3 = _mm256_loadu_ps(B + 3 * s + k);
        __m256 r4 = _mm256_loadu_ps(B + 4 * s + k);
        __m256 r5 = _mm256_loadu_ps(B + 5 * s + k);
        __m256 r6 = _mm256_loadu_ps(B + 6 * s + k);
        __m256 r7 = _mm256_loadu_ps(B + 7 * s + k);

        __m256 d0, d1, d2, d3, d4, d5, d6, d7;
        transpose_8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7,
                          &d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);

        /* d[j] = [B[0][k+j], B[1][k+j], ..., B[7][k+j]]
         * Store to first 8 floats of each K-step's NR-wide slot */
        float *dst = B_pack + (size_t)k * NR;
        _mm256_storeu_ps(dst + 0 * NR, d0);
        _mm256_storeu_ps(dst + 1 * NR, d1);
        _mm256_storeu_ps(dst + 2 * NR, d2);
        _mm256_storeu_ps(dst + 3 * NR, d3);
        _mm256_storeu_ps(dst + 4 * NR, d4);
        _mm256_storeu_ps(dst + 5 * NR, d5);
        _mm256_storeu_ps(dst + 6 * NR, d6);
        _mm256_storeu_ps(dst + 7 * NR, d7);
    }
    /* K remainder for group 1 */
    for (int kk = k; kk < K; kk++) {
        for (int n = 0; n < 8; n++)
            B_pack[(size_t)kk * NR + n] = B[n * s + kk];
    }

    /* Group 2: rows 8-15 → B_pack[k*16 + 8..15] */
    k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 r0 = _mm256_loadu_ps(B +  8 * s + k);
        __m256 r1 = _mm256_loadu_ps(B +  9 * s + k);
        __m256 r2 = _mm256_loadu_ps(B + 10 * s + k);
        __m256 r3 = _mm256_loadu_ps(B + 11 * s + k);
        __m256 r4 = _mm256_loadu_ps(B + 12 * s + k);
        __m256 r5 = _mm256_loadu_ps(B + 13 * s + k);
        __m256 r6 = _mm256_loadu_ps(B + 14 * s + k);
        __m256 r7 = _mm256_loadu_ps(B + 15 * s + k);

        __m256 d0, d1, d2, d3, d4, d5, d6, d7;
        transpose_8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7,
                          &d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);

        /* Store to second 8 floats of each K-step's NR-wide slot */
        float *dst = B_pack + (size_t)k * NR + 8;
        _mm256_storeu_ps(dst + 0 * NR, d0);
        _mm256_storeu_ps(dst + 1 * NR, d1);
        _mm256_storeu_ps(dst + 2 * NR, d2);
        _mm256_storeu_ps(dst + 3 * NR, d3);
        _mm256_storeu_ps(dst + 4 * NR, d4);
        _mm256_storeu_ps(dst + 5 * NR, d5);
        _mm256_storeu_ps(dst + 6 * NR, d6);
        _mm256_storeu_ps(dst + 7 * NR, d7);
    }
    /* K remainder for group 2 */
    for (int kk = k; kk < K; kk++) {
        for (int n = 8; n < 16; n++)
            B_pack[(size_t)kk * NR + n] = B[n * s + kk];
    }
}
