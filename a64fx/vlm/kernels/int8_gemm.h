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
#endif
