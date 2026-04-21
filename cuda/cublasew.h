/*
 * cublasew.h - Minimal dynamic cuBLAS loader without CUDA SDK headers
 *
 * Provides a small wrapper for row-major GEMMs used by local CUDA runners.
 * Falls back cleanly when libcublas is unavailable.
 */
#ifndef CUBLASEW_H
#define CUBLASEW_H

#include "cuew.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cublasew_context cublasew_context;

int cublasewInit(void);
int cublasewCreate(cublasew_context **out, CUstream stream);
void cublasewDestroy(cublasew_context *ctx);
int cublasewSetStream(cublasew_context *ctx, CUstream stream);

/* Row-major Y[m, n_out] = X[m, n_in] * W[n_out, n_in]^T */
int cublasew_gemm_f32_rowmajor_nt(cublasew_context *ctx,
                                  CUdeviceptr d_Y,
                                  CUdeviceptr d_W_f32,
                                  CUdeviceptr d_X_f32,
                                  int n_tok,
                                  int n_out,
                                  int n_in);

/* Row-major Y[m, n_out] = X[m, n_in] * W[n_out, n_in]^T
 * W is FP16, X/Y are FP32, compute is FP32.
 */
int cublasew_gemm_f16_f32_rowmajor_nt(cublasew_context *ctx,
                                      CUdeviceptr d_Y,
                                      CUdeviceptr d_W_f16,
                                      CUdeviceptr d_X_f32,
                                      int n_tok,
                                      int n_out,
                                      int n_in);

/* Row-major Y[m, n_out] = X[m, n_in] * W[n_out, n_in]^T
 * W and X are FP16, Y is FP32, compute is FP32.
 * Use when mixed F16×F32 is not supported (Blackwell).
 */
int cublasew_gemm_f16_f16_f32_rowmajor_nt(cublasew_context *ctx,
                                           CUdeviceptr d_Y,
                                           CUdeviceptr d_W_f16,
                                           CUdeviceptr d_X_f16,
                                           int n_tok,
                                           int n_out,
                                           int n_in);

#ifdef __cplusplus
}
#endif

#endif
