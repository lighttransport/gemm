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

/* Row-major Y[m, n_out] = X[m, n_in] * W[n_out, n_in]^T
 * W and X are BF16, Y is FP32, compute is FP32.
 */
int cublasew_gemm_bf16_bf16_f32_rowmajor_nt(cublasew_context *ctx,
                                             CUdeviceptr d_Y,
                                             CUdeviceptr d_W_bf16,
                                             CUdeviceptr d_X_bf16,
                                             int n_tok,
                                             int n_out,
                                             int n_in);

/* cuBLAS-LT availability check: returns 0 if libcublasLt was loaded and
 * an Lt handle exists in the context. Otherwise -1.
 */
int cublasew_lt_available(cublasew_context *ctx);

/* Row-major Y[m, n_out] = X[m, n_in] * W[n_out, n_in]^T using cuBLAS-LT
 * FP8 e4m3 matmul. W and X are FP8 e4m3, Y is BF16, compute is FP32.
 *
 *   d_w_scale_f32, d_x_scale_f32: per-tensor scales (device pointers,
 *   single float each); D = (W * X^T) * w_scale * x_scale.
 *   d_bias_bf16: optional BF16 bias (length n_out, device pointer); pass 0
 *   to skip.
 *
 * Returns 0 on success, -1 on failure (caller should fall back).
 */
int cublasew_gemm_fp8_e4m3_bf16out_rowmajor_nt(cublasew_context *ctx,
                                               CUdeviceptr d_Y_bf16,
                                               CUdeviceptr d_W_e4m3,
                                               CUdeviceptr d_X_e4m3,
                                               CUdeviceptr d_w_scale_f32,
                                               CUdeviceptr d_x_scale_f32,
                                               CUdeviceptr d_bias_bf16,
                                               int n_tok,
                                               int n_out,
                                               int n_in);

/* Same as above but FP32 output (no bias). */
int cublasew_gemm_fp8_e4m3_f32out_rowmajor_nt(cublasew_context *ctx,
                                              CUdeviceptr d_Y_f32,
                                              CUdeviceptr d_W_e4m3,
                                              CUdeviceptr d_X_e4m3,
                                              CUdeviceptr d_w_scale_f32,
                                              CUdeviceptr d_x_scale_f32,
                                              int n_tok,
                                              int n_out,
                                              int n_in);

#ifdef __cplusplus
}
#endif

#endif
