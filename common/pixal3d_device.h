/* Private, versioned resident plugin protocol. SPDX-License-Identifier: MIT */
#ifndef PIXAL3D_DEVICE_H
#define PIXAL3D_DEVICE_H
#include <stddef.h>
#include <stdint.h>
#define PX_DEVICE_ABI 2
enum px_device_op {
    PX_LINEAR,
    PX_NORM,
    PX_RMS,
    PX_ROPE,
    PX_PART,
    PX_ROUND,
    PX_GELU,
    PX_SILU,
    PX_ADD,
    PX_MODULATE,
    PX_RESIDUAL,
    PX_ATTENTION,
    PX_GATHER,
    PX_C2S,
    PX_SKIP,
    PX_ROPE2,
    PX_SCALE_ADD,
    PX_CONV2_GATHER,
    PX_GROUP_SILU,
    PX_AVERAGE,
    PX_JOIN,
    PX_NAF_SAMPLE,
    PX_ROPE_PHASE
};
/* Handles refer to plugin-owned Buffer objects, not raw device pointers.
 * Activations are F32; precision selects rounding (0 F32, 1 BF16, 2 F16).
 * LINEAR: n rows, c outputs, k inputs, offset output-row offset; w is packed
 * in the requested precision and b is F32. ATTENTION: n queries, k keys,
 * heads heads, c head width; x/w/v are Q/K/V in token-major layout.
 * PART selects offset of k interleaved channel slices. NORM normalizes c
 * channels; RMS/ROPE use k heads. ROPE extra=1 reads cached F32 phases
 * [tokens,c]; ROPE_PHASE builds these from I32 [tokens,4] coordinates. ROPE2 is in-place (n grid, heads, c width,
 * offset prefix). GATHER uses n output rows, c=27*channels, offset source row,
 * extra selects dense [channels,27] or sparse [27,channels] ordering.
 * The remaining operations are internal pointwise/conditioning commands.
 * Coordinate maps are I32 stored in four-byte buffers. */
struct px_device_command {
    int op, precision, n, c, k, heads, offset, extra;
    float epsilon;
    void *out, *x, *w, *b, *v;
};
struct px_device_metrics {
    uint64_t uploads, downloads, allocations, gemms, mma_gemms, attentions, mma_attentions;
    double kernel_ms;
    uint64_t effective_budget_bytes, active_bytes, pooled_bytes, peak_active_bytes, largest_allocation_bytes;
};
#endif
