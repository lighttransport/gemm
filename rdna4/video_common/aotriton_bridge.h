// SPDX-License-Identifier: MIT
#ifndef PIXAL3D_VIDEO_AOTRITON_BRIDGE_H
#define PIXAL3D_VIDEO_AOTRITON_BRIDGE_H
#include <stdint.h>

// Optional standalone AOTriton adapter. Operands and output are BF16 (kind=1)
// or FP16 (kind=2), in row/head/channel order. LSE is FP32 [heads, rows].
// This ABI keeps HIP and AOTriton SDK headers out of the native runner build.
typedef int (*video_aotriton_forward_fn)(uint64_t q, uint64_t k, uint64_t v, uint64_t out,
                                         uint64_t lse, int rows, int heads, int kv_heads, int dim,
                                         int kind, float scale, void *stream);
#ifdef __cplusplus
extern "C" {
#endif
int video_aotriton_bridge_abi(void);
int video_aotriton_forward(uint64_t q, uint64_t k, uint64_t v, uint64_t out, uint64_t lse, int rows,
                           int heads, int kv_heads, int dim, int kind, float scale, void *stream);
// Head/channel-packed inputs [heads, rows, dim], row/head/channel output.
int video_aotriton_forward_heads(uint64_t q, uint64_t k, uint64_t v, uint64_t out, uint64_t lse,
                                 int rows, int heads, int kv_heads, int dim, int kind, float scale,
                                 void *stream);
#ifdef __cplusplus
}
#endif
#endif
