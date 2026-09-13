/* SPDX-License-Identifier: MIT
 * Original CUDA/HIP network kernels. Compiled by NVRTC/HIPRTC, not a framework.
 * Inference rounds operands to BF16. Training decomposes each FP32 operand
 * into three BF16 components and uses six products with separate corrections.
 * Default matrix accumulators and trainable state stay FP32. Explicit RDNA4
 * experiments in gn_rdna4.cuh also implement BF16/INT32/INT64 accumulators.
 */
#if defined(GN_HIP)
#ifndef __HIPCC_RTC__
#include <hip/hip_runtime.h>
#endif
typedef unsigned short ushort8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
#else
#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif
#endif
#if !defined(__CUDACC_RTC__) && !defined(__HIPCC_RTC__)
#include <math.h>
#else
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#endif
extern "C" {
__device__ unsigned short bf(float x) {
    union {
        float f;
        unsigned u;
    } v;
    v.f = x;
    if ((v.u & 0x7fffffffU) > 0x7f800000U)
        return (unsigned short)((v.u >> 16) | 64);
    return (unsigned short)((v.u + 0x7fffU + ((v.u >> 16) & 1)) >> 16);
}
__device__ unsigned short hf(float x) {
    union {
        float f;
        unsigned u;
    } v;
    v.f = x;
    unsigned sign = (v.u >> 16) & 0x8000, bits = v.u & 0x7fffffff;
    int exponent = (int)(bits >> 23) - 127 + 15;
    unsigned mantissa = bits & 0x7fffff;
    if ((bits >> 23) == 255)
        return (unsigned short)(sign | (mantissa ? 0x7e00 : 0x7c00));
    if (exponent >= 31)
        return (unsigned short)(sign | 0x7c00);
    if (exponent <= 0) {
        if (exponent < -10)
            return (unsigned short)sign;
        mantissa |= 0x800000;
        int shift = 14 - exponent;
        unsigned value = mantissa >> shift, remainder = mantissa & ((1u << shift) - 1);
        unsigned halfway = 1u << (shift - 1);
        value += remainder > halfway || (remainder == halfway && (value & 1));
        return (unsigned short)(sign | value);
    }
    unsigned value = ((unsigned)exponent << 10) | (mantissa >> 13), remainder = mantissa & 0x1fff;
    value += remainder > 0x1000 || (remainder == 0x1000 && (value & 1));
    return (unsigned short)(sign | value);
}
__device__ float at(const float *p, int r, int c, int rows, int cols, int trans) {
    if (r >= rows || c >= cols)
        return 0;
    return trans ? p[c * rows + r] : p[r * cols + c];
}
__device__ float unbf(unsigned short x) { return __uint_as_float((unsigned)x << 16); }
#if !defined(GN_HIP)
__device__ __forceinline__ void mma(float &d0, float &d1, float &d2, float &d3, const unsigned *av,
                                    const unsigned *bv) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(bv[0]), "r"(bv[1]));
}
#endif
__global__ void gn_mm(float *y, const float *a, const float *b, int M, int N, int K, int ta, int tb,
                      int add, int precise) {
    int lane = threadIdx.x;
#if defined(GN_HIP)
    int row0 = blockIdx.y * 16, col0 = blockIdx.x * 16, ix = lane & 15, half = lane >> 4;
    float8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    float8 correction = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < K; k += 16) {
        ushort8 av, bv, al, bl, all, bll;
        for (int i = 0; i < 8; i++) {
            float va = at(a, row0 + ix, k + half * 8 + i, M, K, ta);
            float vb = at(b, k + half * 8 + i, col0 + ix, K, N, tb);
            av[i] = bf(va);
            bv[i] = bf(vb);
            al[i] = bf(va - unbf(av[i]));
            bl[i] = bf(vb - unbf(bv[i]));
            all[i] = bf((va - unbf(av[i])) - unbf(al[i]));
            bll[i] = bf((vb - unbf(bv[i])) - unbf(bl[i]));
        }
        if (precise) {
            correction = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(al, bv, correction);
            correction = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, bl, correction);
            correction = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(al, bl, correction);
            correction = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(all, bv, correction);
            correction = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, bll, correction);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, bv, acc);
    }
    for (int i = 0; i < 8; i++) {
        int r = row0 + half * 8 + i, c = col0 + ix;
        if (r < M && c < N)
            y[r * N + c] = (acc[i] + correction[i]) + (add ? y[r * N + c] : 0);
    }
#else
    int row0 = blockIdx.y * 16, col0 = blockIdx.x * 8, g = lane >> 2, t = lane & 3;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    float l0 = 0, l1 = 0, l2 = 0, l3 = 0;
    for (int k = 0; k < K; k += 16) {
        unsigned av[4], bv[2], al[4], bl[2], all[4], bll[2];
        for (int q = 0; q < 4; q++) {
            int r = row0 + g + (q % 2) * 8, c = k + t * 2 + (q / 2) * 8;
            float x0 = at(a, r, c, M, K, ta), x1 = at(a, r, c + 1, M, K, ta);
            unsigned short h0 = bf(x0), h1 = bf(x1);
            av[q] = (unsigned)h0 | ((unsigned)h1 << 16);
            al[q] = (unsigned)bf(x0 - unbf(h0)) | ((unsigned)bf(x1 - unbf(h1)) << 16);
            all[q] = (unsigned)bf((x0 - unbf(h0)) - unbf((unsigned short)al[q])) |
                     ((unsigned)bf((x1 - unbf(h1)) - unbf((unsigned short)(al[q] >> 16))) << 16);
        }
        for (int q = 0; q < 2; q++) {
            int r = k + t * 2 + q * 8, c = col0 + g;
            float x0 = at(b, r, c, K, N, tb), x1 = at(b, r + 1, c, K, N, tb);
            unsigned short h0 = bf(x0), h1 = bf(x1);
            bv[q] = (unsigned)h0 | ((unsigned)h1 << 16);
            bl[q] = (unsigned)bf(x0 - unbf(h0)) | ((unsigned)bf(x1 - unbf(h1)) << 16);
            bll[q] = (unsigned)bf((x0 - unbf(h0)) - unbf((unsigned short)bl[q])) |
                     ((unsigned)bf((x1 - unbf(h1)) - unbf((unsigned short)(bl[q] >> 16))) << 16);
        }
        /* Three BF16 components per operand, six products (i+j<=2).
         * Keep corrections separate from the high-product accumulator so
         * adding small products does not repeatedly lose their low bits. */
        if (precise) {
            mma(l0, l1, l2, l3, al, bv);
            mma(l0, l1, l2, l3, av, bl);
            mma(l0, l1, l2, l3, al, bl);
            mma(l0, l1, l2, l3, all, bv);
            mma(l0, l1, l2, l3, av, bll);
        }
        mma(d0, d1, d2, d3, av, bv);
    }
    int r = row0 + g, c = col0 + t * 2;
    d0 += l0;
    d1 += l1;
    d2 += l2;
    d3 += l3;
    if (r < M && c < N)
        y[r * N + c] = d0 + (add ? y[r * N + c] : 0);
    if (r < M && c + 1 < N)
        y[r * N + c + 1] = d1 + (add ? y[r * N + c + 1] : 0);
    if (r + 8 < M && c < N)
        y[(r + 8) * N + c] = d2 + (add ? y[(r + 8) * N + c] : 0);
    if (r + 8 < M && c + 1 < N)
        y[(r + 8) * N + c + 1] = d3 + (add ? y[(r + 8) * N + c + 1] : 0);
#endif
}
__global__ void gn_mm_fp32(float *y, const float *a, const float *b, int M, int N, int K, int ta,
                           int tb, int add, int precise) {
    (void)precise;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N)
        return;
    int r = i / N, c = i % N;
    float sum = 0;
    for (int k = 0; k < K; k++)
        sum += at(a, r, k, M, K, ta) * at(b, k, c, K, N, tb);
    y[i] = sum + (add ? y[i] : 0);
}
__global__ void gn_columns(float *col, const float *x, int R, int C, int side, int kernel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, K = kernel * kernel * C;
    if (i >= R * K)
        return;
    int row = i / K, k = i % K, ch = k % C, dx = (k / C) % kernel, dy = k / (C * kernel),
        S = side * side;
    int yy = (row % S) / side + dy - kernel / 2, xx = row % side + dx - kernel / 2;
    col[i] = (yy < 0 || xx < 0 || yy >= side || xx >= side)
                 ? 0
                 : x[((row / S) * S + yy * side + xx) * C + ch];
}
__global__ void gn_uncolumns(float *dx, const float *col, int R, int C, int side, int kernel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R * C)
        return;
    int row = i / C, ch = i % C, S = side * side, yy = (row % S) / side, xx = row % side,
        K = kernel * kernel * C;
    float v = 0;
    for (int ky = 0; ky < kernel; ky++)
        for (int kx = 0; kx < kernel; kx++) {
            int oy = yy - ky + kernel / 2, ox = xx - kx + kernel / 2;
            if (oy < 0 || ox < 0 || oy >= side || ox >= side)
                continue;
            v += col[((row / S * S + oy * side + ox) * K) + (ky * kernel + kx) * C + ch];
        }
    dx[i] += v;
}
__global__ void gn_bias(float *y, const float *bias, int R, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < R * C)
        y[i] += bias[i % C];
}
__global__ void gn_bias_back(float *db, const float *dy, int R, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    float v = 0;
    for (int r = 0; r < R; r++)
        v += dy[r * C + c];
    db[c] += v;
}
__global__ void gn_point(float *y, float *dx, float *db, const float *x, const float *b,
                         const float *dy, int count, int op, int back) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    if (!back) {
        y[i] = op == 3   ? x[i] + b[i]
               : op == 4 ? x[i] * b[i]
               : op == 5 ? fmaxf(0, x[i])
                         : x[i] / (1 + expf(-x[i]));
        return;
    }
    if (op == 3) {
        dx[i] += dy[i];
        db[i] += dy[i];
    } else if (op == 4) {
        dx[i] += dy[i] * b[i];
        db[i] += dy[i] * x[i];
    } else if (op == 5)
        dx[i] += x[i] > 0 ? dy[i] : 0;
    else {
        float s = 1 / (1 + expf(-x[i]));
        dx[i] += dy[i] * s * (1 + x[i] * (1 - s));
    }
}
__global__ void gn_norm(float *y, float *aux, float *mean, float *var, const float *x,
                        const float *w, const float *bias, int R, int C, int layer, int training) {
    int group = blockIdx.x * blockDim.x + threadIdx.x, G = layer ? R : C, N = layer ? C : R;
    if (group >= G)
        return;
    double mu = 0, v = 0;
    if (layer || training) {
        for (int i = 0; i < N; i++)
            mu += x[layer ? group * C + i : i * C + group];
        mu /= N;
        for (int i = 0; i < N; i++) {
            double d = x[layer ? group * C + i : i * C + group] - mu;
            v += d * d;
        }
        v /= N;
        if (!layer) {
            mean[group] = .9f * mean[group] + .1f * (float)mu;
            var[group] = .9f * var[group] + .1f * (float)(N > 1 ? v * N / (N - 1) : v);
        }
    } else {
        mu = mean[group];
        v = var[group];
    }
    float inv = rsqrtf((float)v + 1e-5f);
    aux[group] = (float)mu;
    aux[G + group] = inv;
    for (int i = 0; i < N; i++) {
        int idx = layer ? group * C + i : i * C + group, c = layer ? i : group;
        y[idx] = (x[idx] - (float)mu) * inv * w[c] + bias[c];
    }
}
__global__ void gn_norm_back(float *dx, float *dw, float *db, const float *x, const float *dy,
                             const float *w, const float *aux, int R, int C, int layer) {
    int group = blockIdx.x * blockDim.x + threadIdx.x, G = layer ? R : C, N = layer ? C : R;
    if (group >= G)
        return;
    float mu = aux[group], inv = aux[G + group];
    double sum = 0, prod = 0;
    for (int i = 0; i < N; i++) {
        int j = layer ? group * C + i : i * C + group;
        float d = dy[j] * (layer ? w[i] : 1);
        sum += d;
        prod += d * (x[j] - mu) * inv;
    }
    if (!layer) {
        dw[group] += (float)prod;
        db[group] += (float)sum;
    }
    for (int i = 0; i < N; i++) {
        int j = layer ? group * C + i : i * C + group;
        float d = dy[j] * (layer ? w[i] : 1);
        dx[j] += inv * (layer ? 1 : w[group]) *
                 (d - (float)(sum / N) - (x[j] - mu) * inv * (float)(prod / N));
    }
}
__global__ void gn_layer_param(float *dw, float *db, const float *x, const float *dy,
                               const float *aux, int R, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    double a = 0, b = 0;
    for (int r = 0; r < R; r++) {
        a += dy[r * C + c] * (x[r * C + c] - aux[r]) * aux[R + r];
        b += dy[r * C + c];
    }
    dw[c] += (float)a;
    db[c] += (float)b;
}
__global__ void gn_attention(float *y, float *prob, const float *x, const float *bias, int B,
                             int side, int C, int D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x, S = side * side, H = C / D;
    if (t >= B * H * S)
        return;
    int i = t % S, h = (t / S) % H, b = t / (S * H), span = 2 * side - 1;
    float scale = rsqrtf((float)D), top = -INFINITY;
    for (int j = 0; j < S; j++) {
        float dot = 0;
        for (int d = 0; d < D; d++)
            dot += x[(b * S + i) * 3 * C + h * D + d] * x[(b * S + j) * 3 * C + C + h * D + d];
        int rel = (i / side + side - 1 - j / side) * span + (i % side + side - 1 - j % side);
        float p = dot * scale + bias[rel * H + h];
        prob[t * S + j] = p;
        top = fmaxf(top, p);
    }
    double sum = 0;
    for (int j = 0; j < S; j++) {
        float p = expf(prob[t * S + j] - top);
        prob[t * S + j] = p;
        sum += p;
    }
    for (int j = 0; j < S; j++)
        prob[t * S + j] /= (float)sum;
    for (int d = 0; d < D; d++) {
        float v = 0;
        for (int j = 0; j < S; j++)
            v += prob[t * S + j] * x[(b * S + j) * 3 * C + 2 * C + h * D + d];
        y[(b * S + i) * C + h * D + d] = v;
    }
}
__global__ void gn_attention_back(float *dx, float *db, const float *x, const float *dy,
                                  const float *prob, int B, int side, int C, int D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x, S = side * side, H = C / D;
    if (t >= B * H * S)
        return;
    int i = t % S, h = (t / S) % H, b = t / (S * H), span = 2 * side - 1;
    float dp[361], scale = rsqrtf((float)D);
    double dot = 0;
    for (int j = 0; j < S; j++) {
        float v = 0;
        for (int d = 0; d < D; d++) {
            float g = dy[(b * S + i) * C + h * D + d];
            v += g * x[(b * S + j) * 3 * C + 2 * C + h * D + d];
            atomicAdd(&dx[(b * S + j) * 3 * C + 2 * C + h * D + d], prob[t * S + j] * g);
        }
        dp[j] = v;
        dot += v * prob[t * S + j];
    }
    for (int j = 0; j < S; j++) {
        float ds = prob[t * S + j] * (dp[j] - (float)dot);
        int rel = (i / side + side - 1 - j / side) * span + (i % side + side - 1 - j % side);
        atomicAdd(&db[rel * H + h], ds);
        for (int d = 0; d < D; d++) {
            int q = (b * S + i) * 3 * C + h * D + d, k = (b * S + j) * 3 * C + C + h * D + d;
            atomicAdd(&dx[q], ds * scale * x[k]);
            atomicAdd(&dx[k], ds * scale * x[q]);
        }
    }
}
__global__ void gn_loss(float *dp, float *dv, float *loss, const float *p, const float *v,
                        const float *target, const unsigned *label, int B, int A) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;
    float top = -INFINITY;
    for (int i = 0; i < A; i++)
        if (target[b * A + i] >= 0)
            top = fmaxf(top, p[b * A + i]);
    double sum = 0;
    for (int i = 0; i < A; i++)
        if (target[b * A + i] >= 0)
            sum += exp((double)p[b * A + i] - top);
    double logz = log(sum) + top, lp = 0;
    for (int i = 0; i < A; i++)
        if (target[b * A + i] >= 0) {
            dp[b * A + i] = (float)exp(p[b * A + i] - logz) - target[b * A + i];
            lp += target[b * A + i] * (logz - p[b * A + i]);
        }
    top = fmaxf(v[b * 3], fmaxf(v[b * 3 + 1], v[b * 3 + 2]));
    sum = 0;
    for (int i = 0; i < 3; i++)
        sum += exp((double)v[b * 3 + i] - top);
    logz = log(sum) + top;
    for (int i = 0; i < 3; i++)
        dv[b * 3 + i] = (float)exp(v[b * 3 + i] - logz) - (i == (int)label[b]);
    loss[b * 2] = (float)lp;
    loss[b * 2 + 1] = (float)(logz - v[b * 3 + label[b]]);
}
__global__ void gn_grad_norm(float *norm, const float *g, int count, float inv) {
    __shared__ float sums[256];
    int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    float v = i < count ? g[i] * inv : 0;
    if (!isfinite(v))
        atomicExch(norm + 1, 1);
    sums[t] = v * v;
    __syncthreads();
    for (int d = 128; d > 0; d /= 2) {
        if (t < d)
            sums[t] += sums[t + d];
        __syncthreads();
    }
    if (t == 0)
        atomicAdd(norm, sums[0]);
}
__global__ void gn_adam(float *x, float *m, float *v, float *g, int count, float scale, float lr,
                        float decay, float b1, float b2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    float d = g[i] * scale;
    float a = .9f * m[i] + .1f * d, b = .999f * v[i] + .001f * d * d;
    m[i] = a;
    v[i] = b;
    x[i] -= lr * ((a / b1) / (sqrtf(b / b2) + 1e-8f) + decay * x[i]);
    g[i] = 0;
}
}
#include "gn_sm120.cuh"
#include "gn_rdna4.cuh"
