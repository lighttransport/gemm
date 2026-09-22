/* Noncausal image-query WMMA attention plus causal text-prefix correction.
 * The image rows of Qwen Image 2.1 attend to all text and image keys; only
 * the text prefix needs the triangular mask. SPDX-License-Identifier: MIT */
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
                                       int prefix, int heads) {
    int h = blockIdx.x, qi = blockIdx.y, lane = threadIdx.x;
    if (qi >= prefix || h >= heads) return;
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
                                         void *stream_handle) {
    if (!out || !q || !k || !v || tokens < 1 || prefix < 0 || prefix > tokens ||
        heads < 1 || heads > 64) return hipErrorInvalidValue;
    hipStream_t stream = static_cast<hipStream_t>(stream_handle);
    px_trellis_attention<<<dim3(heads, (tokens + 63) / 64), 128, 0, stream>>>(
        out, q, k, v, tokens, heads, 128, nullptr, 0);
    hipError_t rc = hipGetLastError();
    if (rc != hipSuccess) return rc;
    if (prefix) {
        q21_causal_text_prefix<<<dim3(heads, prefix), 32, 0, stream>>>(
            out, q, k, v, prefix, heads);
        rc = hipGetLastError();
    }
    return rc;
}
