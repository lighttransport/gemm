/*
 * mm_blaslt_bridge.h - C-callable multi-shape hipBLASLt BF16 GEMM bridge.
 *
 * Provides a per-(M,N,K) plan cache. All plans use:
 *   X row-major [M,K] BF16, W row-major [N,K] BF16, Y row-major [M,N] F32
 *   computing Y = X * W^T (no bias, no epilogue in v1).
 * hipBLASLt sees this as col-major Y^T[N,M] = W[N,K] * X^T[K,M], so
 *   A=W (op=T, [K,N] col-major), B=X (op=N, [K,M] col-major), C=D=Y^T [N,M].
 *
 * Generalized form of rdna4/vlm/mm0_hipblaslt_bridge.cpp.
 */
#ifndef MM_BLASLT_BRIDGE_H
#define MM_BLASLT_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Lazily creates the hipBLASLt handle on first call. Returns 0 on success. */
int  mm_blaslt_init(void);

/* Run Y = X * W^T for the given (M,N,K). On first call for a new shape, queries
 * the heuristic and caches the plan + workspace. d_y/d_w/d_x are device ptrs.
 * stream is a hipStream_t (cast to void*); pass NULL for default stream.
 * Returns 0 on success, -1 on error. */
int  mm_blaslt_run_bf16(void *d_y_f32, const void *d_w_bf16,
                       const void *d_x_bf16,
                       int M, int N, int K, void *stream);

/* Same as mm_blaslt_run_bf16 but with fused F32 bias epilogue: Y = X*W^T + bias[N].
 * bias may be NULL — in that case behavior matches mm_blaslt_run_bf16. */
int  mm_blaslt_run_bf16_bias(void *d_y_f32, const void *d_w_bf16,
                             const void *d_x_bf16, const void *d_bias_f32,
                             int M, int N, int K, void *stream);

/* Fused bias + residual: D = X*W^T + bias + C, where C is F32 [M,N] (row-major).
 * beta=1.0 internally; pass d_c=d_y to do an in-place "Y += X*W^T + bias".
 * Equivalent to mm_blaslt_run_bf16_bias followed by a residual add. */
int  mm_blaslt_run_bf16_bias_residual(void *d_y_f32, const void *d_c_f32,
                                      const void *d_w_bf16,
                                      const void *d_x_bf16,
                                      const void *d_bias_f32,
                                      int M, int N, int K, void *stream);

/* Fused bias + GELU with BF16 D output: D[bf16] = GELU(X*W^T + bias).
 * Writes BF16 directly into d_y_bf16 (no F32 D buffer). Useful before the
 * next BF16 GEMM (e.g. mlp_fc2) — eliminates the standalone gelu_pack pass. */
int  mm_blaslt_run_bf16_bias_gelu_bf16d(void *d_y_bf16, const void *d_w_bf16,
                                         const void *d_x_bf16,
                                         const void *d_bias_f32,
                                         int M, int N, int K, void *stream);

/* Bias-only with BF16 D output: D[bf16] = X*W^T + bias.  Bias is required. */
int  mm_blaslt_run_bf16_bias_bf16d(void *d_y_bf16, const void *d_w_bf16,
                                    const void *d_x_bf16,
                                    const void *d_bias_f32,
                                    int M, int N, int K, void *stream);

/* Tear down all cached plans + handle. Safe to call multiple times. */
void mm_blaslt_destroy(void);

#ifdef __cplusplus
}
#endif

#endif /* MM_BLASLT_BRIDGE_H */
