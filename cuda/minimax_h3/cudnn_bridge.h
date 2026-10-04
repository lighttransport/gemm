// SPDX-License-Identifier: MIT
// Optional cuDNN SDPA bridge for H3 DiT attention. Loaded with dlopen only when
// requested; cuDNN 9 and cudart are themselves discovered at runtime.
#ifndef PIXAL3D_H3_CUDNN_BRIDGE_H
#define PIXAL3D_H3_CUDNN_BRIDGE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Returns 2 (ABI 2 adds the fp16 selector). Shared by MiniMax H3 (BF16) and
 * HunyuanVideo 1.5 (FP16). */
typedef int (*h3_cudnn_abi_fn)(void);
/* Loads cuDNN 9 from `library`, or when NULL from the default loader search path and
 * fixed system locations (no environment variables are read).
 * Returns 0 on success and writes the loaded cuDNN path/version into info. */
typedef int (*h3_cudnn_init_fn)(const char *library, char *info, size_t capacity);
/* Workspace bytes for a [1, heads, rows, dim] non-causal forward (fp16 != 0: FP16,
 * otherwise BF16), or -1. */
typedef long long (*h3_cudnn_workspace_fn)(int rows, int heads, int dim, int fp16, char *error,
                                            size_t capacity);
/* O = softmax(Q K^T * scale) V for [1, heads, rows, dim] contiguous FP16/BF16 tensors. */
typedef int (*h3_cudnn_attention_fn)(void *out, const void *q, const void *k, const void *v,
                                     int rows, int heads, int dim, int fp16, float scale,
                                     void *workspace, void *stream, char *error, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
