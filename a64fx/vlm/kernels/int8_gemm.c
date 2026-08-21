// int8_gemm.c — W8A8 int8 SDOT GEMM for the a64fx VLM.
//
// Weights: quantized to int8 per OUTPUT-channel (per-n) at cache-build time
// (take_int8_packed), pre-packed for the 6x4 kernel.
// Activations: quantized to int8 per-tensor (one scale per GEMM call).
// Accumulate: int32, then dequantized to fp32 (C = C_i32 * a_scale * w_scale[n]).
//
// The 6x4 kernel (kernel_6x4_int8.S) computes C += A[6][256] x B[64][256]^T
// (accumulating variant, so K can be summed over K/256 chunks). nb-outer
// schedule: each thread owns an N-slice, streams small A across M-tiles, reads
// the (int8, half-size) W once.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_sve.h>
#include <omp.h>

#define I8_MR 6
#define I8_NR 64
#define I8_KC 256

extern void kernel_int8_6x4_256(const int8_t *Apack, const int8_t *Bpack,
                                int32_t *C, int ldc);

static void *xal(size_t n) {
    void *p = aligned_alloc(64, (n + 63) & ~(size_t)63);
    if (!p) fprintf(stderr, "int8_gemm: OOM %zu\n", n), exit(1);
    return p;
}

/* ── packing (copied from int8-new/gemm_pack.c) ───────────────────────── */
static void pack_A_6x256(const int8_t *A, int lda, int8_t *Apack, int M) {
    for (int m0 = 0; m0 < M; m0 += I8_MR) {
        int mr = (m0 + I8_MR <= M) ? I8_MR : (M - m0);
        for (int m = 0; m < I8_MR; m++)
            for (int k = 0; k < I8_KC; k++)
                *Apack++ = (m < mr) ? A[(size_t)(m0 + m) * lda + k] : 0;
    }
}
static void pack_B_64x256(const int8_t *B, int ldb, int8_t *Bpack, int N) {
    for (int n0 = 0; n0 < N; n0 += I8_NR) {
        int nr = (n0 + I8_NR <= N) ? I8_NR : (N - n0);
        for (int k = 0; k < I8_KC; k += 4)
            for (int vec = 0; vec < 4; vec++)
                for (int col = 0; col < 16; col++) {
                    int n = n0 + vec * 16 + col;
                    for (int kk = 0; kk < 4; kk++)
                        *Bpack++ = (n < N && k + kk < I8_KC) ? B[(size_t)n * ldb + k + kk] : 0;
                }
    }
}
static size_t packed_int8_B_size(int K, int N) {
    int NB = (N + I8_NR - 1) / I8_NR, KC = (K + I8_KC - 1) / I8_KC;
    return (size_t)NB * KC * I8_NR * I8_KC;
}

/* Quantize a [K][N] fp32 BT (BT[k][n] = W[n][k]) to int8 and pre-pack.
 * Per-n (output-channel) scale. Takes ownership of *pbt (frees it).
 * Returns the packed B buffer; *scale_out = per-n scale (length NP, padded). */
int8_t *take_int8_packed(float **pbt, int K, int N, float **scale_out) {
    const float *BT = *pbt;
    int NP = (N + I8_NR - 1) / I8_NR * I8_NR;
    int NB = NP / I8_NR, KC = (K + I8_KC - 1) / I8_KC;

    float *scale = (float *)xal((size_t)NP * 4);
    for (int n = 0; n < N; n++) {
        float mx = 0;
        for (int k = 0; k < K; k++) { float v = fabsf(BT[(size_t)k * N + n]); if (v > mx) mx = v; }
        scale[n] = mx > 1e-9f ? mx / 127.0f : 1.0f;
    }
    for (int n = N; n < NP; n++) scale[n] = 1.0f;

    int8_t *B8 = (int8_t *)xal((size_t)NP * K);   // [NP][K], rows [N,NP) zero
    for (int n = 0; n < N; n++) {
        float inv = 1.0f / scale[n];
        for (int k = 0; k < K; k++) {
            int v = (int)lroundf(BT[(size_t)k * N + n] * inv);
            if (v > 127) v = 127; if (v < -127) v = -127;
            B8[(size_t)n * K + k] = (int8_t)v;
        }
    }
    int8_t *Bpack = (int8_t *)xal(packed_int8_B_size(K, N));
    for (int nb = 0; nb < NB; nb++)
        for (int kc = 0; kc < KC; kc++)
            pack_B_64x256(B8 + (size_t)(nb * I8_NR) * K + kc * I8_KC, K,
                          Bpack + (size_t)(nb * KC + kc) * I8_NR * I8_KC,
                          N - nb * I8_NR);
    free(B8);
    free(*pbt); *pbt = NULL;
    *scale_out = scale;
    return Bpack;
}

/* C[M][N] (fp32) = X[M][K] (fp32) * W[N][K]^T (int8)  +  (dequant only; bias
 * is added by the caller via add_bias_mt). W is pre-packed int8 with per-n scale. */
void gemm_int8_BTP(int M, int K, int N, const float *X, int lda,
                   const int8_t *Bpack, const float *w_scale,
                   float *Y, int ldc) {
    if (K % I8_KC) { fprintf(stderr, "gemm_int8_BTP: K=%d not multiple of %d\n", K, I8_KC); return; }
    int MB = (M + I8_MR - 1) / I8_MR;
    int NP = (N + I8_NR - 1) / I8_NR * I8_NR;
    int NB = NP / I8_NR;
    int KC = K / I8_KC;

    // 1+2) per-ROW activation quantize (each row has its own scale -> no global
    // max barrier, more accurate than per-tensor). One parallel region over rows.
    float *a_scale = (float *)xal((size_t)M * 4);
    int8_t *X8 = (int8_t *)xal((size_t)M * K);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float *xr = X + (size_t)m * lda;
        float mx = 0;
        for (int k = 0; k < K; k++) { float v = fabsf(xr[k]); if (v > mx) mx = v; }
        float inv = (mx > 1e-9f) ? (127.0f / mx) : 1.0f;
        a_scale[m] = (mx > 1e-9f) ? (mx / 127.0f) : 1.0f;
        for (int k = 0; k < K; k++) {
            int v = (int)lroundf(xr[k] * inv);
            if (v > 127) v = 127; if (v < -127) v = -127;
            X8[(size_t)m * K + k] = (int8_t)v;
        }
    }

    // 3) pack A per (mb, kc)
    int8_t *Apack = (int8_t *)xal((size_t)MB * KC * I8_MR * I8_KC);
    // Pack exactly ONE 6x256 tile per (mb,kc) — passing more rows would write
    // past the tile into the next (mb,kc) slot and race across threads.
    #pragma omp parallel for schedule(static)
    for (int mb = 0; mb < MB; mb++)
        for (int kc = 0; kc < KC; kc++) {
            int mr = (M - mb * I8_MR < I8_MR) ? (M - mb * I8_MR) : I8_MR;
            pack_A_6x256(X8 + (size_t)(mb * I8_MR) * K + kc * I8_KC, K,
                         Apack + (size_t)(mb * KC + kc) * I8_MR * I8_KC, mr);
        }

    // 4) GEMM (nb-outer, accumulating over K-chunks) into int32 C [MB*6][NB*64]
    int32_t *C = (int32_t *)xal((size_t)(MB * I8_MR) * NP * 4);
    memset(C, 0, (size_t)(MB * I8_MR) * NP * 4);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nb = 0; nb < NB; nb++)
        for (int mb = 0; mb < MB; mb++) {
            int32_t *Ct = C + (size_t)(mb * I8_MR) * NP + (nb * I8_NR);
            for (int kc = 0; kc < KC; kc++)
                kernel_int8_6x4_256(Apack + (size_t)(mb * KC + kc) * I8_MR * I8_KC,
                                    Bpack + (size_t)(nb * KC + kc) * I8_NR * I8_KC,
                                    Ct, NP * 4);
        }

    // 5) dequant Y[m][n] = C[m][n] * a_scale[m] * w_scale[n]  (valid [M][N])
    const svbool_t pg = svptrue_b32();
    const int VL = (int)svcntw();
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int32_t *cr = C + (size_t)m * NP;
        float *yr = Y + (size_t)m * ldc;
        svfloat32_t va = svdup_f32(a_scale[m]);
        for (int n = 0; n < N; n += VL) {
            svbool_t pg2 = svwhilelt_b32_s32(n, N);
            svfloat32_t zf = svcvt_f32_x(pg2, svld1_s32(pg2, cr + n));
            svfloat32_t zr = svmul_f32_x(pg2, zf, svmul_f32_x(pg2, va, svld1_f32(pg2, w_scale + n)));
            svst1_f32(pg2, yr + n, zr);
        }
    }
    free(C); free(X8); free(Apack); free(a_scale);
}


/* ─────────────────────────────────────────────────────────────────────────
 * INT16 path. A64FX has NO 16-bit dot product / int16 FMA (FEAT_SVE_FP16
 * absent) and this binutils 2.30 doesn't even assemble SMLA/SMLAL/SMULL, so
 * int16 is done by splitting each int16 into a high int8 + low int8 and
 * expanding into int8 SDOT GEMMs.
 *
 *   x16 = xhi*256 + xlo_s + 128   (xhi = x16>>8, xlo_s = (x16&0xff)-128)
 *
 *   Σ_k a16*b16  =  2^16*Σahi*bhi + 2^8*Σahi*blo_s + 2^8*Σalo_s*bhi
 *                 + 2^15*Σahi + 2^7*Σalo_s + 2^15*Σbhi + 2^7*Σblo_s + 2^14*K
 *   (the alo_s*blo_s term is dropped: a sqrt(K) random walk vs the K*signal,
 *    ~1e-5 relative for the GEMM sum.)
 *
 * = 3 int8 GEMMs + 2 per-row reductions (A) + 2 per-col reductions (W,
 *   precomputed) + an int64 combine + dequant.
 * ───────────────────────────────────────────────────────────────────────── */
size_t packed_int16_B_size(int K, int N) { return 2 * packed_int8_B_size(K, N); }

/* Quantize a [K][N] fp32 BT to int16 per-n, split hi/lo, pre-pack BOTH halves
 * (contiguous: [hi][lo]), precompute per-col sums of each half. Takes
 * ownership of *pbt. *scale_out = per-n scale (length NP). *colsum_out =
 * int32[2*NP]: [0,NP)=Σ_k bhi, [NP,2NP)=Σ_k blo. Returns the hi block; the
 * lo block is at  ret + packed_int8_B_size(K,N). */
int8_t *take_int16_packed(float **pbt, int K, int N, float **scale_out,
                          int32_t **colsum_out) {
    const float *BT = *pbt;
    int NP = (N + I8_NR - 1) / I8_NR * I8_NR;
    int NB = NP / I8_NR, KC = (K + I8_KC - 1) / I8_KC;
    size_t pb1 = packed_int8_B_size(K, N);

    float *scale = (float *)xal((size_t)NP * 4);
    for (int n = 0; n < N; n++) {
        float mx = 0;
        for (int k = 0; k < K; k++) { float v = fabsf(BT[(size_t)k * N + n]); if (v > mx) mx = v; }
        scale[n] = mx > 1e-9f ? mx / 32767.0f : 1.0f;
    }
    for (int n = N; n < NP; n++) scale[n] = 1.0f;

    int16_t *W16 = (int16_t *)xal((size_t)NP * K * 2);
    for (int n = 0; n < N; n++) {
        float inv = 1.0f / scale[n];
        for (int k = 0; k < K; k++) {
            int v = (int)lroundf(BT[(size_t)k * N + n] * inv);
            if (v > 32767) v = 32767; if (v < -32768) v = -32768;
            W16[(size_t)n * K + k] = (int16_t)v;
        }
    }
    int8_t *Bhi = (int8_t *)xal((size_t)NP * K), *Blo = (int8_t *)xal((size_t)NP * K);
    int32_t *colsum = (int32_t *)xal((size_t)2 * NP * 4);
    for (int n = 0; n < N; n++) {
        int32_t shi = 0, slo = 0;
        for (int k = 0; k < K; k++) {
            int16_t w = W16[(size_t)n * K + k];
            int hi = (int)(w >> 8), lo = (int)(w & 0xFF) - 128;
            Bhi[(size_t)n * K + k] = (int8_t)hi;
            Blo[(size_t)n * K + k] = (int8_t)lo;
            shi += hi; slo += lo;
        }
        colsum[n] = shi; colsum[NP + n] = slo;
    }
    int8_t *Bpack = (int8_t *)xal(2 * pb1);
    int8_t *Bpack_hi = Bpack, *Bpack_lo = Bpack + pb1;
    for (int nb = 0; nb < NB; nb++)
        for (int kc = 0; kc < KC; kc++) {
            pack_B_64x256(Bhi + (size_t)(nb * I8_NR) * K + kc * I8_KC, K,
                          Bpack_hi + (size_t)(nb * KC + kc) * I8_NR * I8_KC, N - nb * I8_NR);
            pack_B_64x256(Blo + (size_t)(nb * I8_NR) * K + kc * I8_KC, K,
                          Bpack_lo + (size_t)(nb * KC + kc) * I8_NR * I8_KC, N - nb * I8_NR);
        }
    free(Bhi); free(Blo); free(W16);
    free(*pbt); *pbt = NULL;
    *scale_out = scale;
    *colsum_out = colsum;
    return Bpack;
}

/* Y[M][N] = X[M][K] * W[N][K]^T with W pre-quantized int16 (take_int16_packed),
 * A quantized int16 per-row. X, Y are fp32. bias added by the caller. */
void gemm_int16_BTP(int M, int K, int N, const float *X, int lda,
                    const int8_t *Bpack, const float *w_scale, const int32_t *colsum,
                    float *Y, int ldc) {
    if (K % I8_KC) { fprintf(stderr, "gemm_int16_BTP: K=%d not multiple of %d\n", K, I8_KC); return; }
    size_t pb1 = packed_int8_B_size(K, N);
    const int8_t *Bpack_hi = Bpack, *Bpack_lo = Bpack + pb1;
    int MB = (M + I8_MR - 1) / I8_MR;
    int NP = (N + I8_NR - 1) / I8_NR * I8_NR;
    int NB = NP / I8_NR, KC = K / I8_KC;
    int NP2 = (N + I8_NR - 1) / I8_NR * I8_NR;
    (void)NP2;

    // 1) quantize A to int16 per-row + split hi/lo + row sums
    float *a_scale = (float *)xal((size_t)M * 4);
    int8_t *Ahi = (int8_t *)xal((size_t)M * K), *Alo = (int8_t *)xal((size_t)M * K);
    int32_t *rowhi = (int32_t *)xal((size_t)M * 4), *rowlo = (int32_t *)xal((size_t)M * 4);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float *xr = X + (size_t)m * lda;
        float mx = 0;
        for (int k = 0; k < K; k++) { float v = fabsf(xr[k]); if (v > mx) mx = v; }
        float inv = (mx > 1e-9f) ? (32767.0f / mx) : 1.0f;
        a_scale[m] = (mx > 1e-9f) ? (mx / 32767.0f) : 1.0f;
        int32_t shi = 0, slo = 0;
        for (int k = 0; k < K; k++) {
            int v = (int)lroundf(xr[k] * inv);
            if (v > 32767) v = 32767; if (v < -32768) v = -32768;
            int16_t w = (int16_t)v;
            int hi = (int)(w >> 8), lo = (int)(w & 0xFF) - 128;
            Ahi[(size_t)m * K + k] = (int8_t)hi;
            Alo[(size_t)m * K + k] = (int8_t)lo;
            shi += hi; slo += lo;
        }
        rowhi[m] = shi; rowlo[m] = slo;
    }

    // 2) pack A hi/lo per (mb,kc) -- one 6-row tile each
    size_t aptile = (size_t)I8_MR * I8_KC;
    int8_t *Apack_hi = (int8_t *)xal((size_t)MB * KC * aptile);
    int8_t *Apack_lo = (int8_t *)xal((size_t)MB * KC * aptile);
    #pragma omp parallel for schedule(static)
    for (int mb = 0; mb < MB; mb++)
        for (int kc = 0; kc < KC; kc++) {
            int mr = (M - mb * I8_MR < I8_MR) ? (M - mb * I8_MR) : I8_MR;
            pack_A_6x256(Ahi + (size_t)(mb * I8_MR) * K + kc * I8_KC, K,
                         Apack_hi + (size_t)(mb * KC + kc) * aptile, mr);
            pack_A_6x256(Alo + (size_t)(mb * I8_MR) * K + kc * I8_KC, K,
                         Apack_lo + (size_t)(mb * KC + kc) * aptile, mr);
        }

    // 3) 3 GEMMs (nb-outer, accumulating over K-chunks)
    size_t csize = (size_t)(MB * I8_MR) * NP;
    int32_t *Chi = (int32_t *)xal(csize * 4), *Chl = (int32_t *)xal(csize * 4);
    int32_t *Clh = (int32_t *)xal(csize * 4);
    memset(Chi, 0, csize * 4); memset(Chl, 0, csize * 4); memset(Clh, 0, csize * 4);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nb = 0; nb < NB; nb++)
        for (int mb = 0; mb < MB; mb++) {
            int32_t *Chit = Chi + (size_t)(mb * I8_MR) * NP + (nb * I8_NR);
            int32_t *Chlt = Chl + (size_t)(mb * I8_MR) * NP + (nb * I8_NR);
            int32_t *Clht = Clh + (size_t)(mb * I8_MR) * NP + (nb * I8_NR);
            for (int kc = 0; kc < KC; kc++) {
                const int8_t *ah = Apack_hi + (size_t)(mb * KC + kc) * aptile;
                const int8_t *al = Apack_lo + (size_t)(mb * KC + kc) * aptile;
                kernel_int8_6x4_256(ah, Bpack_hi + (size_t)(nb * KC + kc) * I8_NR * I8_KC, Chit, NP * 4);
                kernel_int8_6x4_256(ah, Bpack_lo + (size_t)(nb * KC + kc) * I8_NR * I8_KC, Chlt, NP * 4);
                kernel_int8_6x4_256(al, Bpack_hi + (size_t)(nb * KC + kc) * I8_NR * I8_KC, Clht, NP * 4);
            }
        }

    // 4) int64 combine + dequant (valid [M][N]). SVE has no int64 accumulate, so
    //    the (few) per-element int64 combine steps are scalar; the GEMMs (the bulk
    //    of the cost) are fully SVE int8.
    const int64_t K14 = (int64_t)16384 * K;   // 2^14 * K
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int32_t *chi = Chi + (size_t)m * NP, *chl = Chl + (size_t)m * NP, *clh = Clh + (size_t)m * NP;
        float *yr = Y + (size_t)m * ldc;
        const int64_t ra = (int64_t)32768 * rowhi[m] + (int64_t)128 * rowlo[m];
        const float as = a_scale[m];
        for (int n = 0; n < N; n++) {
            int64_t c = (int64_t)65536 * chi[n] + (int64_t)256 * chl[n] + (int64_t)256 * clh[n]
                      + ra + (int64_t)32768 * colsum[n] + (int64_t)128 * colsum[NP + n] + K14;
            yr[n] = (float)c * as * w_scale[n];
        }
    }
    free(a_scale); free(Ahi); free(Alo); free(rowhi); free(rowlo);
    free(Apack_hi); free(Apack_lo); free(Chi); free(Chl); free(Clh);
}
