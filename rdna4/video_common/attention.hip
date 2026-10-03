// SPDX-License-Identifier: MIT
// Bounded online softmax, 128 query rows x 64 keys. Transposed vector LDS slots
// keep WMMA fragments contiguous; Q remains in registers throughout the loop.
#include <hip/hip_runtime.h>
typedef float va_f8 __attribute__((ext_vector_type(8)));
typedef short va_b8 __attribute__((ext_vector_type(8)));
typedef _Float16 va_h8 __attribute__((ext_vector_type(8)));
template <bool BF> struct va_type;
template <> struct va_type<true> {
    using scalar = short;
    using vector = va_b8;
};
template <> struct va_type<false> {
    using scalar = _Float16;
    using vector = va_h8;
};
template <bool BF> __device__ __forceinline__ typename va_type<BF>::scalar va_cast(float v) {
    if constexpr (BF) {
        unsigned u = __float_as_uint(v);
        if ((u & 0x7f800000u) != 0x7f800000u)
            u += 0x7fff + ((u >> 16) & 1);
        return short(u >> 16);
    } else
        return _Float16(v);
}
template <bool BF>
__device__ __forceinline__ va_f8 va_mma(typename va_type<BF>::vector a,
                                        typename va_type<BF>::vector b, va_f8 c) {
    if constexpr (BF)
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    else
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
template <bool BF, int HD, bool BASE2 = false>
__device__ __forceinline__ void va_attention(float *out, const float *q, const float *k,
                                             const float *v, int rows, int heads, float scale,
                                             int kv_heads, int causal) {
    if constexpr (BASE2)
        scale *= 1.4426950408889634f;
    using element = typename va_type<BF>::scalar;
    using fragment = typename va_type<BF>::vector;
    constexpr int BK = 64, WAVES = 8, BQ = WAVES * 16;
    int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, half = lane >> 4, col = lane & 15;
    int head = blockIdx.x, q0 = blockIdx.y * BQ + wave * 16, kh = head / (heads / kv_heads);
    __shared__ fragment keys[(HD / 8) * BK];
    __shared__ element values[HD * (BK + 4)];
    __shared__ element probabilities[WAVES * 16 * (BK + 4)];
    element *prob = probabilities + wave * 16 * (BK + 4);
    fragment qr[HD / 16];
#pragma unroll
    for (int a = 0; a < HD / 16; a++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            int row = q0 + col, d = a * 16 + half * 8 + j;
            qr[a][j] = va_cast<BF>(row < rows ? q[((long)row * heads + head) * HD + d] : 0.f);
        }
    }
    va_f8 acc[HD / 16];
#pragma unroll
    for (int a = 0; a < HD / 16; a++)
        acc[a] = va_f8{};
    float maximum[8], denominator[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        maximum[j] = -1e30f;
        denominator[j] = 0.f;
    }
    for (int base = 0; base < rows; base += BK) {
        for (int i = tid; i < BK * (HD / 8); i += 256) {
            int r = i / (HD / 8), d = (i % (HD / 8)) * 8, row = base + r;
            fragment key;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                long at = ((long)row * kv_heads + kh) * HD + d + j;
                key[j] = va_cast<BF>(row < rows ? k[at] : 0.f);
                values[(d + j) * (BK + 4) + r] = va_cast<BF>(row < rows ? v[at] : 0.f);
            }
            keys[(d / 8) * BK + r] = key;
        }
        __syncthreads();
        va_f8 score[BK / 16];
#pragma unroll
        for (int b = 0; b < BK / 16; b++)
            score[b] = va_f8{};
#pragma unroll
        for (int a = 0; a < HD / 16; a++) {
#pragma unroll
            for (int b = 0; b < BK / 16; b++)
                score[b] = va_mma<BF>(qr[a], keys[(a * 2 + half) * BK + b * 16 + col], score[b]);
        }
        float alpha[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float mx = -1e30f;
#pragma unroll
            for (int b = 0; b < BK / 16; b++) {
                int row = base + b * 16 + col;
                float s = score[b][j] * scale;
                if (row >= rows || (causal && row > q0 + half * 8 + j))
                    s = -1e30f;
                score[b][j] = s;
                mx = fmaxf(mx, s);
            }
            mx = fmaxf(mx, __shfl_xor(mx, 1));
            mx = fmaxf(mx, __shfl_xor(mx, 2));
            mx = fmaxf(mx, __shfl_xor(mx, 4));
            mx = fmaxf(mx, __shfl_xor(mx, 8));
            float next = fmaxf(maximum[j], mx);
            alpha[j] =
                BASE2 ? __builtin_amdgcn_exp2f(maximum[j] - next) : __expf(maximum[j] - next);
            float sum = 0;
#pragma unroll
            for (int b = 0; b < BK / 16; b++) {
                int row = base + b * 16 + col;
                float p = (row >= rows || (causal && row > q0 + half * 8 + j))
                              ? 0.f
                              : (BASE2 ? __builtin_amdgcn_exp2f(score[b][j] - next)
                                       : __expf(score[b][j] - next));
                sum += p;
                prob[(half * 8 + j) * (BK + 4) + b * 16 + col] = va_cast<BF>(p);
            }
            if constexpr (BASE2) {
                // Match the reference online denominator: reduce each key tile
                // from widest shuffle to narrowest before its fused rescaling.
                sum += __shfl_xor(sum, 8);
                sum += __shfl_xor(sum, 4);
                sum += __shfl_xor(sum, 2);
                sum += __shfl_xor(sum, 1);
                denominator[j] = fmaf(denominator[j], alpha[j], sum);
            } else {
                denominator[j] = denominator[j] * alpha[j] + sum;
            }
            maximum[j] = next;
        }
#pragma unroll
        for (int a = 0; a < HD / 16; a++) {
#pragma unroll
            for (int j = 0; j < 8; j++)
                acc[a][j] *= alpha[j];
        }
        __syncthreads();
#pragma unroll
        for (int b = 0; b < BK / 16; b++) {
            fragment p;
#pragma unroll
            for (int j = 0; j < 8; j++)
                p[j] = prob[col * (BK + 4) + b * 16 + half * 8 + j];
#pragma unroll
            for (int a = 0; a < HD / 16; a++) {
                fragment value;
#pragma unroll
                for (int j = 0; j < 8; j++)
                    value[j] = values[(a * 16 + col) * (BK + 4) + b * 16 + half * 8 + j];
                acc[a] = va_mma<BF>(p, value, acc[a]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
        float sum = denominator[j];
        if constexpr (!BASE2) {
            sum += __shfl_xor(sum, 1);
            sum += __shfl_xor(sum, 2);
            sum += __shfl_xor(sum, 4);
            sum += __shfl_xor(sum, 8);
        }
        int row = q0 + half * 8 + j;
#pragma unroll
        for (int a = 0; a < HD / 16; a++)
            if (row < rows)
                out[((long)row * heads + head) * HD + a * 16 + col] =
                    sum > 0 ? (BASE2 ? acc[a][j] * (1.f / sum) : acc[a][j] / sum) : 0.f;
    }
}
#define VA_WRAPPER(name, bf, hd)                                                                   \
    extern "C" __global__ __launch_bounds__(256, 1) void name(                                     \
        float *o, const float *q, const float *k, const float *v, int n, int h, int d, float s,    \
        int kh, int causal) {                                                                      \
        (void)d;                                                                                   \
        va_attention<bf, hd>(o, q, k, v, n, h, s, kh, causal);                                     \
    }
VA_WRAPPER(video_attention_tile_bf16, true, 128)
VA_WRAPPER(video_attention_tile_bf16_64, true, 64)
VA_WRAPPER(video_attention_tile_f16, false, 128)
VA_WRAPPER(video_attention_tile_f16_64, false, 64)
#undef VA_WRAPPER

// H3 uses the base-2 path; keep Hunyuan and FP16 arithmetic unchanged.
extern "C" __global__ __launch_bounds__(256, 1) void video_attention_tile_bf16_exp2(
    float *o, const float *q, const float *k, const float *v, int n, int h, int d, float s, int kh,
    int causal) {
    (void)d;
    va_attention<true, 128, true>(o, q, k, v, n, h, s, kh, causal);
}
