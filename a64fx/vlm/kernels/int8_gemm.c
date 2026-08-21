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
