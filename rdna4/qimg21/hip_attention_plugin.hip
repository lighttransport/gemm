/* Segment-prefix WMMA attention plus causal text-row correction.
 * Each image group attends to all preceding rows and its entire own group;
 * interleaved text rows retain their triangular mask. SPDX-License-Identifier: MIT */
#include <hip/hip_runtime.h>
#include <cstring>

__device__ float round_bfloat(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7f800000u) != 0x7f800000u)
        u += 0x7fffU + ((u >> 16) & 1U);
    return __uint_as_float(u & 0xffff0000u);
}

/* The same validated gfx12 WMMA kernel used by Pixal3D. */
#include "../../tmp/pixal3d/wmma_attention.inc"

__global__ void q21_causal_text_prefix(float *out, const float *q,
                                       const float *k, const float *v,
                                       int text_rows, int heads,
                                       const int *image_ids) {
    int h = blockIdx.x, qi = blockIdx.y, lane = threadIdx.x;
    if (qi >= text_rows || h >= heads || (image_ids && image_ids[qi] >= 0)) return;
    int stride = heads * 128;
    float qr[4], acc[4] = {0, 0, 0, 0};
    for (int e = 0; e < 4; e++) qr[e] = q[(long)qi * stride + h * 128 + lane + 32 * e];
    float mx = -1e30f;
    for (int t = 0; t <= qi; t++) {
        float dot = 0;
        for (int e = 0; e < 4; e++) dot += qr[e] * k[(long)t * stride + h * 128 + lane + 32 * e];
        for (int z = 16; z; z >>= 1) dot += __shfl_xor(dot, z);
        mx = fmaxf(mx, dot * 0.08838834764831845f);
    }
    float den = 0;
    for (int t = 0; t <= qi; t++) {
        float dot = 0;
        for (int e = 0; e < 4; e++) dot += qr[e] * k[(long)t * stride + h * 128 + lane + 32 * e];
        for (int z = 16; z; z >>= 1) dot += __shfl_xor(dot, z);
        float p = expf(dot * 0.08838834764831845f - mx);
        den += p;
        for (int e = 0; e < 4; e++) acc[e] += p * v[(long)t * stride + h * 128 + lane + 32 * e];
    }
    for (int e = 0; e < 4; e++)
        out[(long)qi * stride + h * 128 + lane + 32 * e] = acc[e] / den;
}

extern "C" int q21_hip_fused_attention(float *out, const float *q,
                                         const float *k, const float *v,
                                         int tokens, int prefix, int heads,
                                         void *stream_handle,
                                         const int *segment_ends, int segments,
                                         const int *image_ids) {
    if (!out || !q || !k || !v || tokens < 1 || prefix < 0 || prefix > tokens ||
        heads < 1 || heads > 64) return hipErrorInvalidValue;
    if (segments < 0 || (segments && (!segment_ends || !image_ids ||
                                      segment_ends[segments - 1] != tokens)))
        return hipErrorInvalidValue;
    hipStream_t stream = static_cast<hipStream_t>(stream_handle);
    /* Later image groups are computed first. Recomputing shorter prefixes
     * then corrects earlier groups without overwriting later output rows. */
    int previous = tokens + 1;
    for (int i = segments ? segments - 1 : 0; i >= 0; i--) {
        int end = segments ? segment_ends[i] : tokens;
        if (end < 1 || end >= previous) return hipErrorInvalidValue;
        previous = end;
        px_trellis_attention<<<dim3(heads, (end + 63) / 64), 128, 0, stream>>>(
            out, q, k, v, end, heads, 128, nullptr, 0);
        hipError_t rc = hipGetLastError();
        if (rc != hipSuccess) return rc;
        if (!segments) break;
    }
    int text_rows = image_ids ? tokens : prefix;
    if (text_rows) {
        q21_causal_text_prefix<<<dim3(heads, text_rows), 32, 0, stream>>>(
            out, q, k, v, text_rows, heads, image_ids);
        return hipGetLastError();
    }
    return hipSuccess;
}
