#ifndef VLM_INT8_GEMM_H
#define VLM_INT8_GEMM_H
#include <stdint.h>
// Quantize a [K][N] fp32 BT (BT[k][n]=W[n][k]) to int8 + pre-pack; per-n scale.
// Takes ownership of *pbt (frees it). Returns packed B; *scale_out = [NP] scale.
int8_t *take_int8_packed(float **pbt, int K, int N, float **scale_out);
// C[M][N] = X[M][K] * W^T (int8), dequantized to fp32 (bias added by caller).
void gemm_int8_BTP(int M, int K, int N, const float *X, int lda,
                   const int8_t *Bpack, const float *w_scale,
                   float *Y, int ldc);
// Fused variant (GEMM + dequant in one kernel, no int32 C buffer). Same result
// as gemm_int8_BTP; use when the separate dequant pass / C traffic matters.
void gemm_int8_BTP_fused(int M, int K, int N, const float *X, int lda,
                         const int8_t *Bpack, const float *w_scale,
                         float *Y, int ldc);

// ── int16 (hi/lo int8 split; A64FX has no 16-bit dot product / int16 FMA) ─
size_t packed_int16_B_size(int K, int N);
// Quantize a [K][N] fp32 BT to int16 per-n, split hi/lo, pre-pack both halves
// (contiguous [hi][lo]) + per-col sums. Takes ownership of *pbt. *scale_out =
// per-n scale (length NP). *colsum_out = int32[2*NP]: [0,NP)=Σ bhi, [NP,2NP)=Σ blo.
int8_t *take_int16_packed(float **pbt, int K, int N, float **scale_out,
                          int32_t **colsum_out);
// C[M][N] = X * W^T, W pre-quantized int16, A quantized int16 per-row.
void gemm_int16_BTP(int M, int K, int N, const float *X, int lda,
                    const int8_t *Bpack, const float *w_scale, const int32_t *colsum,
                    float *Y, int ldc);
#endif
