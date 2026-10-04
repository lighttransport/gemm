// SPDX-License-Identifier: MIT
// Optional cuDNN SDPA bridge for H3 DiT attention. Loaded with dlopen only when
// requested; cuDNN 9 and cudart are themselves discovered at runtime.
#ifndef PIXAL3D_H3_CUDNN_BRIDGE_H
#define PIXAL3D_H3_CUDNN_BRIDGE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Returns 1. */
typedef int (*h3_cudnn_abi_fn)(void);
/* Loads cuDNN 9 from `library`, or when NULL from the default loader search path and
 * fixed system locations (no environment variables are read).
 * Returns 0 on success and writes the loaded cuDNN path/version into info. */
typedef int (*h3_cudnn_init_fn)(const char *library, char *info, size_t capacity);
/* Workspace bytes for a BF16 [1, heads, rows, dim] non-causal forward, or -1. */
typedef long long (*h3_cudnn_workspace_fn)(int rows, int heads, int dim, char *error,
                                            size_t capacity);
/* O = softmax(Q K^T * scale) V for BF16 [1, heads, rows, dim] contiguous tensors. */
typedef int (*h3_cudnn_attention_fn)(void *out, const void *q, const void *k, const void *v,
                                     int rows, int heads, int dim, float scale, void *workspace,
                                     void *stream, char *error, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
