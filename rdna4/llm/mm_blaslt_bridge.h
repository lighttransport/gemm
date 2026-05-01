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

/* Tear down all cached plans + handle. Safe to call multiple times. */
void mm_blaslt_destroy(void);

#ifdef __cplusplus
}
#endif

#endif /* MM_BLASLT_BRIDGE_H */
