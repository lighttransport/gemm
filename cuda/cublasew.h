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

/* F32 GEMM precision tiers (highest precision first):
 *   - cublasew_gemm_f32_pedantic_*  : strict IEEE FP32, never TF32 — for
 *                                     reference-parity comparisons.
 *   - cublasew_gemm_f32_rowmajor_nt : default cublasSgemm (no TF32 unless the
 *                                     gate below opts in).
 *   - cublasew_gemm_f32_tf32_*      : TF32 tensor-core path, ~2-4x faster,
 *                                     ~1e-3 relative error.
 * cublasew_set_tf32() or env CUBLASEW_ALLOW_TF32=1 lets the *plain* f32 NT path
 * route to the TF32 tensor cores. Default is off (exact SGEMM, no behavior
 * change for existing callers). */
void cublasew_set_tf32(cublasew_context *ctx, int enable);

/* Row-major Y[m, n_out] = X[m, n_in] * W[n_out, n_in]^T */
int cublasew_gemm_f32_rowmajor_nt(cublasew_context *ctx,
                                  CUdeviceptr d_Y,
                                  CUdeviceptr d_W_f32,
                                  CUdeviceptr d_X_f32,
                                  int n_tok,
                                  int n_out,
                                  int n_in);

/* Same layout as cublasew_gemm_f32_rowmajor_nt, but forces the TF32
 * tensor-core compute path (CUBLAS_COMPUTE_32F_FAST_TF32). Faster, ~1e-3 rel. */
int cublasew_gemm_f32_tf32_rowmajor_nt(cublasew_context *ctx,
                                       CUdeviceptr d_Y,
                                       CUdeviceptr d_W_f32,
                                       CUdeviceptr d_X_f32,
                                       int n_tok,
                                       int n_out,
                                       int n_in);

/* Same layout as above, but accumulates into an existing Y with beta=1. */
int cublasew_gemm_f32_rowmajor_nt_beta1(cublasew_context *ctx,
                                        CUdeviceptr d_Y,
                                        CUdeviceptr d_W_f32,
                                        CUdeviceptr d_X_f32,
                                        int n_tok,
                                        int n_out,
                                        int n_in);

/* cuBLAS-LT variant of row-major NT with beta=1 into an existing Y. */
int cublasew_gemm_f32_lt_rowmajor_nt_beta1(cublasew_context *ctx,
                                           CUdeviceptr d_Y,
                                           CUdeviceptr d_W_f32,
                                           CUdeviceptr d_X_f32,
                                           int n_tok,
                                           int n_out,
                                           int n_in);

/* Same layout as above, but uses cuBLAS-LT with a F32 bias epilogue. This
 * matches PyTorch CUDA linear() more closely than SGEMM followed by a separate
 * bias add for small-output SC-VAE projections.
 */
int cublasew_gemm_f32_lt_bias_rowmajor_nt(cublasew_context *ctx,
                                          CUdeviceptr d_Y,
                                          CUdeviceptr d_W_f32,
                                          CUdeviceptr d_X_f32,
                                          CUdeviceptr d_bias_f32,
                                          int n_tok,
                                          int n_out,
                                          int n_in);

/* Same row-major NT layout, but no bias epilogue. Useful for matching
 * PyTorch CUDA matmul() on SC-VAE sparse-conv packed rows.
 */
int cublasew_gemm_f32_lt_rowmajor_nt(cublasew_context *ctx,
                                     CUdeviceptr d_Y,
                                     CUdeviceptr d_W_f32,
                                     CUdeviceptr d_X_f32,
                                     int n_tok,
                                     int n_out,
                                     int n_in);

/* Same layout as above, but requests pedantic FP32 compute via cublasGemmEx.
 * This avoids TF32/fast-math paths for strict reference comparisons.
 */
int cublasew_gemm_f32_pedantic_rowmajor_nt(cublasew_context *ctx,
                                           CUdeviceptr d_Y,
                                           CUdeviceptr d_W_f32,
                                           CUdeviceptr d_X_f32,
                                           int n_tok,
                                           int n_out,
                                           int n_in);

int cublasew_gemm_f32_pedantic_rowmajor_nt_strided(cublasew_context *ctx,
                                                   CUdeviceptr d_Y,
                                                   CUdeviceptr d_W_f32,
                                                   int ld_w,
                                                   CUdeviceptr d_X_f32,
                                                   int ld_x,
                                                   int n_tok,
                                                   int n_out,
                                                   int n_in);

/* INT8 W8A8 GEMM: int8 x int8 -> int32 (CUBLAS_COMPUTE_32I), library-tuned AND
 * bit-exact (order-independent int32 accumulate). Y[n_tok,n_out] int32 =
 * Xq[n_tok,n_in] @ Wq[n_out,n_in]^T. Dequant to bf16 is a separate kernel. */
int cublasew_gemm_int8_s32_rowmajor_nt(cublasew_context *ctx,
                                       CUdeviceptr d_Yi32,
                                       CUdeviceptr d_Wq,
                                       CUdeviceptr d_Xq,
                                       int n_tok,
                                       int n_out,
                                       int n_in);

int cublasew_gemm_int8_s32_rowmajor_nt_strided(cublasew_context *ctx,
                                               CUdeviceptr d_Yi32,
                                               CUdeviceptr d_Wq,
                                               int ld_w,
                                               CUdeviceptr d_Xq,
                                               int ld_x,
                                               int n_tok,
                                               int n_out,
                                               int n_in);

/* Row-major Y[m, n_out] = A[m, n_in] * B[n_in, n_out].
 * `ld_y` is the row stride of Y in floats, allowing writes into an
 * interleaved parent matrix.
 */
int cublasew_gemm_f32_pedantic_rowmajor_nn(cublasew_context *ctx,
                                           CUdeviceptr d_Y,
                                           int ld_y,
                                           CUdeviceptr d_A_f32,
                                           CUdeviceptr d_B_f32,
                                           int n_tok,
                                           int n_out,
                                           int n_in);

int cublasew_gemm_f32_rowmajor_nn(cublasew_context *ctx,
                                  CUdeviceptr d_Y,
                                  int ld_y,
                                  CUdeviceptr d_A_f32,
                                  CUdeviceptr d_B_f32,
                                  int n_tok,
                                  int n_out,
                                  int n_in);

int cublasew_gemm_f32_pedantic_rowmajor_nn_stridedB(cublasew_context *ctx,
                                                    CUdeviceptr d_Y,
                                                    int ld_y,
                                                    CUdeviceptr d_A_f32,
                                                    CUdeviceptr d_B_f32,
                                                    int ld_b,
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

/* Strided-batched F16×F16→F32 GEMM for MoE all-expert matmul.
 * For each batch e: Y[e, n_tok, n_out] = X[n_tok, n_in] @ W[e, n_out, n_in]^T
 * X is shared across all batches (strideB = 0).
 * Returns 0 on success, -1 if cuBLAS strided-batched API not available. */
int cublasew_gemm_f16_f16_f32_strided_batched(cublasew_context *ctx,
                                               CUdeviceptr d_Y,
                                               CUdeviceptr d_W_f16,
                                               CUdeviceptr d_X_f16,
                                               int n_tok,
                                               int n_out,
                                               int n_in,
                                               int batch);

/* Row-major Y[n_tok, n_out] = A[n_tok, n_in] * B[n_in, n_out].
 * A and B are FP16, Y is FP32, compute is FP32. `ld_y` is Y's row stride in
 * floats (allows writing into an interleaved parent matrix). Tensor-core path
 * for the attention P*V product on Blackwell.
 */
int cublasew_gemm_f16_f16_f32_rowmajor_nn(cublasew_context *ctx,
                                          CUdeviceptr d_Y,
                                          int ld_y,
                                          CUdeviceptr d_A_f16,
                                          CUdeviceptr d_B_f16,
                                          int n_tok,
                                          int n_out,
                                          int n_in);

/* Row-major Y[n_tok, n_out] = X[n_tok, n_in] * W[n_out, n_in]^T + bias, with an
 * optional tanh-GELU on (Y + bias), fused into the cuBLAS-LT epilogue. W and X
 * are FP16, bias FP32, compute FP32. `gelu` != 0 selects GELU_BIAS. `y_f16` != 0
 * makes the output D FP16 (d_Y points at an FP16 buffer); 0 keeps it FP32. The
 * bias stays FP32 in both cases. Returns -1 (no side effects) if cuBLAS-LT is
 * unavailable so callers can fall back. */
int cublasew_gemm_f16_f16_f32_lt_bias_rowmajor_nt(cublasew_context *ctx,
                                                  CUdeviceptr d_Y,
                                                  CUdeviceptr d_W_f16,
                                                  CUdeviceptr d_X_f16,
                                                  CUdeviceptr d_bias_f32,
                                                  int gelu,
                                                  int y_f16,
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

/* Row-major Y[n_tok, n_out] = X[n_tok, n_in] * W[n_out, n_in]^T + bias, with an
 * optional tanh-GELU on (Y + bias), fused into the cuBLAS-LT epilogue. W and X
 * are BF16, bias FP32, compute FP32. `gelu` != 0 selects GELU_BIAS.
 * `y_f16` != 0 makes the output D FP16 (d_Y points at an FP16 buffer); 0 keeps
 * it FP32. The bias stays FP32 in both cases. Returns -1 (no side effects) if
 * cuBLAS-LT is unavailable so callers can fall back. */
int cublasew_gemm_bf16_bf16_f32_lt_bias_rowmajor_nt(cublasew_context *ctx,
                                                    CUdeviceptr d_Y,
                                                    CUdeviceptr d_W_bf16,
                                                    CUdeviceptr d_X_bf16,
                                                    CUdeviceptr d_bias_f32,
                                                    int gelu,
                                                    int y_f16,
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

/* Per-row (per-output-channel) FP8 weight scaling, runnable on consumer
 * GeForce Blackwell (sm_120), FP32 output.
 *
 *   d_w_scale_vec_f32: a [n_out] FP32 device vector — one scale s[o] per output
 *   channel (row of W). Each weight row is FP8-quantized with its own scale, so
 *   channels with wide per-channel dynamic range keep precision a single
 *   per-tensor scale would crush. d_x_scale_f32 stays a per-tensor B scalar.
 *
 * Consumer GeForce cuBLAS-LT FP8 only supports a per-tensor A scale (vector A
 * scale modes are NOT_SUPPORTED at the heuristic stage). Since s[o] factors out
 * of the dot-product sum, this runs a per-tensor FP8 matmul with no A scale into
 * an F32 D, then applies the per-row scale as a kernel-free cublasSdgmm row-scale
 * (C = diag(s) * D, in place). Output is FP32 (Sdgmm is F32-typed; cublasew adds
 * no conversion kernels). No bias epilogue. Returns -1 if cublasSdgmm is
 * unavailable or the FP8 F32-out matmul is rejected, so the caller can fall back
 * to per-tensor.
 */
int cublasew_gemm_fp8_e4m3_f32out_wperrow_rowmajor_nt(cublasew_context *ctx,
                                                      CUdeviceptr d_Y_f32,
                                                      CUdeviceptr d_W_e4m3,
                                                      CUdeviceptr d_X_e4m3,
                                                      CUdeviceptr d_w_scale_vec_f32,
                                                      CUdeviceptr d_x_scale_f32,
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
