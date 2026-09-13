/* SPDX-License-Identifier: MIT
 * Original shared parallel operations and sm120 tiled kernels.
 * A and B are packed as [component, row, padded K]; B is logically transposed.
 */
template <typename T> __device__ T gn_block_sum(T v) {
    __shared__ T sums[256];
    __syncthreads();
    int t = threadIdx.x;
    sums[t] = v;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (t < d)
            sums[t] += sums[t + d];
        __syncthreads();
    }
    return sums[0];
}
__device__ float gn_block_max(float v) {
    __shared__ float maxima[256];
    int t = threadIdx.x;
    maxima[t] = v;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (t < d)
            maxima[t] = fmaxf(maxima[t], maxima[t + d]);
        __syncthreads();
    }
    return maxima[0];
}
extern "C" __global__ void gn_norm_parallel(float *y, float *aux, float *mean, float *var,
                                            const float *x, const float *w, const float *bias,
                                            int R, int C, int layer, int training) {
    int group = blockIdx.x, t = threadIdx.x, N = layer ? C : R, G = layer ? R : C;
    double mu = 0, v = 0;
    if (layer || training) {
        for (int i = t; i < N; i += 256)
            mu += x[layer ? group * C + i : i * C + group];
        mu = gn_block_sum(mu) / N;
        for (int i = t; i < N; i += 256) {
            double d = x[layer ? group * C + i : i * C + group] - mu;
            v += d * d;
        }
        v = gn_block_sum(v) / N;
        if (!layer && t == 0) {
            mean[group] = .9f * mean[group] + .1f * (float)mu;
            var[group] = .9f * var[group] + .1f * (float)(N > 1 ? v * N / (N - 1) : v);
        }
    } else {
        mu = mean[group];
        v = var[group];
    }
    float inv = rsqrtf((float)v + 1e-5f);
    if (!t) {
        aux[group] = (float)mu;
        aux[G + group] = inv;
    }
    for (int i = t; i < N; i += 256) {
        int idx = layer ? group * C + i : i * C + group, c = layer ? i : group;
        y[idx] = (x[idx] - (float)mu) * inv * w[c] + bias[c];
    }
}
extern "C" __global__ void gn_norm_back_parallel(float *dx, float *dw, float *db, const float *x,
                                                 const float *dy, const float *w, const float *aux,
                                                 int R, int C, int layer) {
    int group = blockIdx.x, t = threadIdx.x, N = layer ? C : R, G = layer ? R : C;
    float mu = aux[group], inv = aux[G + group];
    double sum = 0, prod = 0;
    for (int i = t; i < N; i += 256) {
        int j = layer ? group * C + i : i * C + group;
        float d = dy[j] * (layer ? w[i] : 1);
        sum += d;
        prod += d * (x[j] - mu) * inv;
    }
    sum = gn_block_sum(sum);
    prod = gn_block_sum(prod);
    if (!layer && !t) {
        dw[group] += (float)prod;
        db[group] += (float)sum;
    }
    for (int i = t; i < N; i += 256) {
        int j = layer ? group * C + i : i * C + group;
        float d = dy[j] * (layer ? w[i] : 1);
        dx[j] += inv * (layer ? 1 : w[group]) *
                 (d - (float)(sum / N) - (x[j] - mu) * inv * (float)(prod / N));
    }
}
extern "C" __global__ void gn_layer_param_parallel(float *dw, float *db, const float *x,
                                                   const float *dy, const float *aux, int R,
                                                   int C) {
    int c = blockIdx.x, t = threadIdx.x;
    double a = 0, b = 0;
    for (int r = t; r < R; r += 256) {
        a += dy[r * C + c] * (x[r * C + c] - aux[r]) * aux[R + r];
        b += dy[r * C + c];
    }
    a = gn_block_sum(a);
    b = gn_block_sum(b);
    if (!t) {
        dw[c] += (float)a;
        db[c] += (float)b;
    }
}
extern "C" __global__ void gn_loss_parallel(float *dp, float *dv, float *loss, const float *p,
                                            const float *v, const float *target,
                                            const unsigned *label, int B, int A) {
    int b = blockIdx.x, t = threadIdx.x;
    float top = -INFINITY;
    for (int i = t; i < A; i += 256)
        if (target[b * A + i] >= 0)
            top = fmaxf(top, p[b * A + i]);
    top = gn_block_max(top);
    double sum = 0, lp = 0;
    for (int i = t; i < A; i += 256)
        if (target[b * A + i] >= 0)
            sum += exp((double)p[b * A + i] - top);
    sum = gn_block_sum(sum);
    double logz = log(sum) + top;
    for (int i = t; i < A; i += 256) {
        int j = b * A + i;
        dp[j] = target[j] >= 0 ? (float)exp(p[j] - logz) - target[j] : 0;
        if (target[j] >= 0)
            lp += target[j] * (logz - p[j]);
    }
    lp = gn_block_sum(lp);
    if (!t) {
        loss[b * 2] = (float)lp;
        top = fmaxf(v[b * 3], fmaxf(v[b * 3 + 1], v[b * 3 + 2]));
        sum = 0;
        for (int i = 0; i < 3; i++)
            sum += exp((double)v[b * 3 + i] - top);
        logz = log(sum) + top;
        for (int i = 0; i < 3; i++)
            dv[b * 3 + i] = (float)exp(v[b * 3 + i] - logz) - (i == (int)label[b]);
        loss[b * 2 + 1] = (float)(logz - v[b * 3 + label[b]]);
    }
    (void)B;
}
extern "C" __global__ void gn_pack_bf16(unsigned short *out, const float *in, int R, int K,
                                        int trans, int precise) {
    __shared__ float tile[32][33];
    int t = threadIdx.x, r0 = blockIdx.y * 32, k0 = blockIdx.x * 32;
    int stride = (K + 31) & ~31;
#if defined(GN_HIP)
    if (!trans) {
        int r = r0 + t / 8, k = k0 + t % 8 * 4;
        if (r < R) {
            float4 x = {};
            if (!(K & 3) && k + 3 < K)
                x = *reinterpret_cast<const float4 *>(in + r * K + k);
            else {
                x.x = k < K ? in[r * K + k] : 0;
                x.y = k + 1 < K ? in[r * K + k + 1] : 0;
                x.z = k + 2 < K ? in[r * K + k + 2] : 0;
                x.w = k + 3 < K ? in[r * K + k + 3] : 0;
            }
            ushort4 high = {bf(x.x), bf(x.y), bf(x.z), bf(x.w)};
            *reinterpret_cast<ushort4 *>(out + r * stride + k) = high;
            if (precise) {
                float4 residual = {x.x - unbf(high.x), x.y - unbf(high.y), x.z - unbf(high.z),
                                   x.w - unbf(high.w)};
                ushort4 low = {bf(residual.x), bf(residual.y), bf(residual.z), bf(residual.w)};
                *reinterpret_cast<ushort4 *>(out + (R + r) * stride + k) = low;
                if (precise != 2) {
                    ushort4 lowest = {bf(residual.x - unbf(low.x)), bf(residual.y - unbf(low.y)),
                                      bf(residual.z - unbf(low.z)), bf(residual.w - unbf(low.w))};
                    *reinterpret_cast<ushort4 *>(out + (2 * R + r) * stride + k) = lowest;
                }
            }
        }
        return;
    }
#endif
    for (int i = t; i < 1024; i += 256) {
        int r = i / 32, k = i % 32;
        int sr = trans ? r0 + k : r0 + r, sk = trans ? k0 + r : k0 + k;
        tile[r][k] = sr < R && sk < K ? in[trans ? sk * R + sr : sr * K + sk] : 0;
    }
    __syncthreads();
    for (int i = t; i < 1024; i += 256) {
        int r = r0 + i / 32, k = k0 + i % 32;
        if (r >= R)
            continue;
        float x = trans ? tile[i % 32][i / 32] : tile[i / 32][i % 32];
        unsigned short h = bf(x);
        out[r * stride + k] = h;
        if (precise) {
            float l = x - unbf(h);
            unsigned short lo = bf(l);
            out[(R + r) * stride + k] = lo;
            if (precise != 2)
                out[(2 * R + r) * stride + k] = bf(l - unbf(lo));
        }
    }
}
/* Emit im2col directly in the packed BF16 layout consumed by the tensor-core
 * kernels, avoiding a full FP32 column write and subsequent pack read. */
extern "C" __global__ void gn_columns_bf16(unsigned short *out, const float *x, int R, int C,
                                           int side, int kernel, int precise) {
    int vector = blockIdx.x * blockDim.x + threadIdx.x;
    int K = C * kernel * kernel, stride = (K + 31) & ~31, vectors = stride / 4;
    if (vector >= R * vectors)
        return;
    int row = vector / vectors, k = vector % vectors * 4;
    int S = side * side, position = row % S;
    int yy = position / side + k / (C * kernel) - kernel / 2;
    int xx = position % side + k / C % kernel - kernel / 2;
    float4 value = {};
    if (k + 3 < K && yy >= 0 && xx >= 0 && yy < side && xx < side)
        value = *reinterpret_cast<const float4 *>(
            x + (row / S * S + yy * side + xx) * C + k % C);
    ushort4 high = {bf(value.x), bf(value.y), bf(value.z), bf(value.w)};
    *reinterpret_cast<ushort4 *>(out + row * stride + k) = high;
    if (precise) {
        ushort4 low = {bf(value.x - unbf(high.x)), bf(value.y - unbf(high.y)),
                       bf(value.z - unbf(high.z)), bf(value.w - unbf(high.w))};
        *reinterpret_cast<ushort4 *>(out + (R + row) * stride + k) = low;
    }
}
extern "C" __global__ void gn_pack_fp16(unsigned short *out, const float *in, int R, int K,
                                        int trans, int precise) {
    __shared__ float tile[32][33];
    int t = threadIdx.x, r0 = blockIdx.y * 32, k0 = blockIdx.x * 32;
    int stride = (K + 31) & ~31;
#if defined(GN_HIP)
    if (!trans) {
        int r = r0 + t / 8, k = k0 + t % 8 * 4;
        if (r < R) {
            float4 x = {};
            if (!(K & 3) && k + 3 < K)
                x = *reinterpret_cast<const float4 *>(in + r * K + k);
            else {
                x.x = k < K ? in[r * K + k] : 0;
                x.y = k + 1 < K ? in[r * K + k + 1] : 0;
                x.z = k + 2 < K ? in[r * K + k + 2] : 0;
                x.w = k + 3 < K ? in[r * K + k + 3] : 0;
            }
            ushort4 packed = {hf(x.x), hf(x.y), hf(x.z), hf(x.w)};
            *reinterpret_cast<ushort4 *>(out + r * stride + k) = packed;
        }
        return;
    }
#endif
    for (int i = t; i < 1024; i += 256) {
        int r = i / 32, k = i % 32;
        int sr = trans ? r0 + k : r0 + r, sk = trans ? k0 + r : k0 + k;
        tile[r][k] = sr < R && sk < K ? in[trans ? sk * R + sr : sr * K + sk] : 0;
    }
    __syncthreads();
    for (int i = t; i < 1024; i += 256) {
        int r = r0 + i / 32, k = k0 + i % 32;
        if (r < R)
            out[r * stride + k] = hf(trans ? tile[i % 32][i / 32] : tile[i / 32][i % 32]);
    }
    (void)precise;
}

/* One CTA owns a query. Separate Q/K/V reductions avoid contended float
 * atomics and the old per-thread 361-entry stack array in backward. */
extern "C" __global__ void gn_attention_parallel(float *y, float *prob, const float *x,
                                                 const float *bias, int B, int side, int C, int D) {
#if !defined(GN_HIP)
    if (side == 9 && D == 32) {
        __shared__ float keys[81][32], values[81][32], scores[8][81];
        int t = threadIdx.x, warp = t / 32, lane = t % 32, H = C / 32;
        int group = blockIdx.x % 11, h = blockIdx.x / 11 % H, b = blockIdx.x / (11 * H);
        int query = group * 8 + warp;
        for (int z = t; z < 81 * 64; z += 256) {
            int j = z / 64, q = z % 64, d = q % 32;
            int offset = (b * 81 + j) * 3 * C + h * 32 + d;
            if (q < 32)
                keys[j][d] = x[offset + C];
            else
                values[j][d] = x[offset + 2 * C];
        }
        __syncthreads();
        float q = query < 81 ? x[(b * 81 + query) * 3 * C + h * 32 + lane] : 0;
        for (int j = 0; j < 81; j++) {
            float dot = q * keys[j][lane];
            for (int delta = 16; delta; delta /= 2)
                dot += __shfl_down_sync(0xffffffffu, dot, delta);
            if (!lane && query < 81) {
                int rel = (query / 9 + 8 - j / 9) * 17 + query % 9 + 8 - j % 9;
                scores[warp][j] = dot * .1767766952966369f + bias[rel * H + h];
            }
        }
        if (!lane && query < 81) {
            float top = -INFINITY;
            for (int j = 0; j < 81; j++)
                top = fmaxf(top, scores[warp][j]);
            double sum = 0;
            for (int j = 0; j < 81; j++) {
                scores[warp][j] = expf(scores[warp][j] - top);
                sum += scores[warp][j];
            }
            int qi = (b * H + h) * 81 + query;
            for (int j = 0; j < 81; j++) {
                scores[warp][j] /= (float)sum;
                prob[qi * 81 + j] = scores[warp][j];
            }
        }
        __syncwarp();
        if (query < 81) {
            float out = 0;
            for (int j = 0; j < 81; j++)
                out += scores[warp][j] * values[j][lane];
            y[(b * 81 + query) * C + h * 32 + lane] = out;
        }
        (void)B;
        return;
    }
#endif
    int S = side * side, H = C / D, query = blockIdx.x, t = threadIdx.x;
    int i = query % S, h = (query / S) % H, b = query / (S * H), span = 2 * side - 1;
    float scale = rsqrtf((float)D), top = -INFINITY;
    for (int j = t; j < S; j += 256) {
        float dot = 0;
        for (int d = 0; d < D; d++)
            dot += x[(b * S + i) * 3 * C + h * D + d] * x[(b * S + j) * 3 * C + C + h * D + d];
        int rel = (i / side + side - 1 - j / side) * span + (i % side + side - 1 - j % side);
        float p = dot * scale + bias[rel * H + h];
        prob[query * S + j] = p;
        top = fmaxf(top, p);
    }
    top = gn_block_max(top);
    double sum = 0;
    for (int j = t; j < S; j += 256) {
        float p = expf(prob[query * S + j] - top);
        prob[query * S + j] = p;
        sum += p;
    }
    sum = gn_block_sum(sum);
    for (int j = t; j < S; j += 256)
        prob[query * S + j] /= (float)sum;
    __syncthreads();
    for (int d = t; d < D; d += 256) {
        float v = 0;
        for (int j = 0; j < S; j++)
            v += prob[query * S + j] * x[(b * S + j) * 3 * C + 2 * C + h * D + d];
        y[(b * S + i) * C + h * D + d] = v;
    }
    (void)B;
}
extern "C" __global__ void gn_attention_scores_back(float *ds, const float *x, const float *dy,
                                                    const float *prob, int B, int side, int C,
                                                    int D) {
    int S = side * side, H = C / D, query = blockIdx.x, t = threadIdx.x;
    int i = query % S, h = (query / S) % H, b = query / (S * H);
    double dot = 0;
    for (int j = t; j < S; j += 256) {
        float dp = 0;
        for (int d = 0; d < D; d++)
            dp += dy[(b * S + i) * C + h * D + d] * x[(b * S + j) * 3 * C + 2 * C + h * D + d];
        ds[query * S + j] = dp;
        dot += dp * prob[query * S + j];
    }
    dot = gn_block_sum(dot);
    for (int j = t; j < S; j += 256)
        ds[query * S + j] = prob[query * S + j] * (ds[query * S + j] - (float)dot);
    (void)B;
}
extern "C" __global__ void gn_attention_qkv_back(float *dx, const float *x, const float *dy,
                                                 const float *prob, const float *ds, int B,
                                                 int side, int C, int D) {
    int index = blockIdx.x * blockDim.x + threadIdx.x, S = side * side, H = C / D;
    if (index >= B * S * C)
        return;
    int d = index % C, h = d / D, i = (index / C) % S, b = index / (S * C);
    int base = (b * H + h) * S * S;
    float dq = 0, dk = 0, dv = 0, scale = rsqrtf((float)D);
    for (int j = 0; j < S; j++) {
        dq += (ds[base + i * S + j] * scale) * x[(b * S + j) * 3 * C + C + d];
        dk += (ds[base + j * S + i] * scale) * x[(b * S + j) * 3 * C + d];
        dv += prob[base + j * S + i] * dy[(b * S + j) * C + d];
    }
    dx[(b * S + i) * 3 * C + d] += dq;
    dx[(b * S + i) * 3 * C + C + d] += dk;
    dx[(b * S + i) * 3 * C + 2 * C + d] += dv;
}
extern "C" __global__ void gn_attention_bias_back(float *db, const float *ds, int B, int side,
                                                  int C, int D) {
    int S = side * side, H = C / D, span = 2 * side - 1, h = blockIdx.x % H, rel = blockIdx.x / H;
    int oy = rel / span - side + 1, ox = rel % span - side + 1;
    double sum = 0;
    for (int r = threadIdx.x; r < B * S; r += 256) {
        int i = r % S, jy = i / side - oy, jx = i % side - ox;
        if (jy >= 0 && jy < side && jx >= 0 && jx < side)
            sum += ds[((r / S) * H + h) * S * S + i * S + jy * side + jx];
    }
    sum = gn_block_sum(sum);
    if (!threadIdx.x)
        db[rel * H + h] += (float)sum;
}

#if !defined(GN_HIP)
__device__ __forceinline__ void gn_ld_a(unsigned *v, const unsigned short *p) {
    unsigned address = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                 : "r"(address));
}
__device__ __forceinline__ void gn_ld_b(unsigned *v, const unsigned short *p) {
    unsigned address = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(v[0]), "=r"(v[1])
                 : "r"(address));
}
__device__ __forceinline__ void gn_stage16(unsigned short *dst, const unsigned short *src,
                                           int valid) {
    unsigned address = (unsigned)__cvta_generic_to_shared(dst);
    int bytes = valid ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(src),
                 "r"(bytes));
}
template <bool Precise, int Products = 6>
__device__ void gn_tiled_body(float *y, const unsigned short *a, const unsigned short *b, int M,
                              int N, int K, int add) {
    /* Training: four 16x32 warps and a 32x64 CTA improve wave occupancy on
     * 1296-row shogi batches. Inference uses a 64x64 CTA. */
    constexpr int MR = Precise ? 1 : 2, Threads = 128, BM = Precise ? 32 : 64;
    constexpr int Planes = Precise ? (Products == 3 ? 2 : 3) : 1;
    __shared__ unsigned short sa[Planes][BM][40];
    __shared__ unsigned short sb[Planes][64][40];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int wr = (warp / 2) * (16 * MR), wc = (warp % 2) * 32;
    int r0 = blockIdx.y * BM, c0 = blockIdx.x * 64, stride = (K + 31) & ~31;
    float high[MR][4][4] = {}, low[MR][4][4] = {};
    for (int k0 = 0; k0 < stride; k0 += 32) {
#pragma unroll
        for (int p = 0; p < Planes; p++) {
            for (int i = tid; i < 64 * 4; i += Threads) {
                int r = i / 4, k = 8 * (i % 4);
                const unsigned short *ap =
                    a + (r0 + r < M ? (p * M + r0 + r) * stride + k0 + k : 0);
                const unsigned short *bp =
                    b + (c0 + r < N ? (p * N + c0 + r) * stride + k0 + k : 0);
                if (r < BM)
                    gn_stage16(&sa[p][r][k], ap, r0 + r < M);
                gn_stage16(&sb[p][r][k], bp, c0 + r < N);
            }
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            unsigned av[Planes][MR][4], bv[Planes][4][2];
#pragma unroll
            for (int p = 0; p < Planes; p++) {
#pragma unroll
                for (int i = 0; i < MR; i++)
                    gn_ld_a(av[p][i], &sa[p][wr + 16 * i + lane % 16][kk + (lane / 16) * 8]);
#pragma unroll
                for (int j = 0; j < 4; j++)
                    gn_ld_b(bv[p][j], &sb[p][wc + 8 * j + lane % 8][kk + ((lane / 8) % 2) * 8]);
            }
#pragma unroll
            for (int product = 0; product < (Precise ? Products : 1); product++) {
                /* Interleave independent output fragments before revisiting
                 * the same correction accumulator. Keep each dot's order. */
                int full_product = Products == 3 && product == 2 ? 5 : product;
                int ac = full_product == 0 || full_product == 2 ? 1
                         : full_product == 3                     ? 2
                                                                 : 0;
                int bc = full_product == 1 || full_product == 2 ? 1
                         : full_product == 4                     ? 2
                                                                 : 0;
                if (!Precise)
                    ac = bc = 0;
#pragma unroll
                for (int i = 0; i < MR; i++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float *acc = Precise && full_product < 5 ? low[i][j] : high[i][j];
                        mma(acc[0], acc[1], acc[2], acc[3], av[ac][i], bv[bc][j]);
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < MR; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int q = 0; q < 4; q++) {
                int r = r0 + wr + i * 16 + lane / 4 + (q / 2) * 8;
                int c = c0 + wc + j * 8 + (lane % 4) * 2 + q % 2;
                if (r < M && c < N)
                    y[r * N + c] = (high[i][j][q] + low[i][j][q]) + (add ? y[r * N + c] : 0);
            }
        }
    }
}
extern "C" __global__ __launch_bounds__(128) void gn_mm_tiled(float *y, const unsigned short *a,
                                                              const unsigned short *b, int M, int N,
                                                              int K, int add) {
    gn_tiled_body<true>(y, a, b, M, N, K, add);
}
extern "C" __global__ __launch_bounds__(128) void gn_mm_tiled_fast(float *y,
                                                                   const unsigned short *a,
                                                                   const unsigned short *b, int M,
                                                                   int N, int K, int add) {
    gn_tiled_body<false>(y, a, b, M, N, K, add);
}
extern "C" __global__ __launch_bounds__(128) void gn_mm_bf16x3(float *y,
                                                               const unsigned short *a,
                                                               const unsigned short *b, int M,
                                                               int N, int K, int add) {
    gn_tiled_body<true, 3>(y, a, b, M, N, K, add);
}

/* Experimental integer operand training: FP32 master state and dequantization.
 * "16" is a signed, balanced two-byte decomposition, NOT native INT16 MMA.
 * q = 256*hi + lo in [-32639,32639]. Row scales are recomputed every call.
 */
extern "C" __global__ void gn_pack_integer(signed char *out, float *scales, const float *in, int R,
                                           int K, int trans, int bits) {
    int r = blockIdx.x, t = threadIdx.x, stride = (K + 31) & ~31;
    float peak = 0;
    for (int k = t; k < K; k += 256)
        peak = fmaxf(peak, fabsf(in[trans ? k * R + r : r * K + k]));
    peak = gn_block_max(peak);
    float scale = peak > 0 ? fmaxf(peak / (bits == 8 ? 127 : 32639), 1e-30f) : 1;
    if (!t)
        scales[r] = scale;
    for (int k = t; k < stride; k += 256) {
        float x = k < K ? in[trans ? k * R + r : r * K + k] : 0;
        int q = __float2int_rn(x / scale), bound = bits == 8 ? 127 : 32639;
        q = max(-bound, min(bound, q));
        int hi = bits == 8 ? q : (q + 128) >> 8;
        out[r * stride + k] = (signed char)hi;
        if (bits == 16)
            out[(R + r) * stride + k] = (signed char)(q - 256 * hi);
    }
}
__device__ __forceinline__ void gn_imma(int *d, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
extern "C" __global__ void gn_mm_integer(float *y, const signed char *a, const signed char *b,
                                         const float *as, const float *bs, int M, int N, int K,
                                         int add, int bits) {
    int lane = threadIdx.x, group = lane / 4, t = lane % 4;
    int r0 = blockIdx.y * 16, c0 = blockIdx.x * 8, stride = (K + 31) & ~31;
    float h[4] = {}, l[4] = {};
    for (int k = 0; k < stride; k += 32) {
        unsigned av[2][4], bv[2][2];
#pragma unroll
        for (int p = 0; p < 2; p++) {
            if (p && bits == 8)
                break;
#pragma unroll
            for (int q = 0; q < 4; q++) {
                int r = r0 + group + (q % 2) * 8, c = k + t * 4 + (q / 2) * 16;
                av[p][q] = r < M ? *(const unsigned *)&a[(p * M + r) * stride + c] : 0;
            }
#pragma unroll
            for (int q = 0; q < 2; q++) {
                int c = c0 + group;
                bv[p][q] =
                    c < N ? *(const unsigned *)&b[(p * N + c) * stride + k + t * 4 + q * 16] : 0;
            }
        }
        /* Reset INT32 accumulators every K=32: even signed endpoints cannot
         * overflow. Do not silently wrap an unbounded INT32 reduction. */
        int hh[4] = {}, hl[4] = {}, lh[4] = {}, ll[4] = {};
        gn_imma(hh, av[0], bv[0]);
        if (bits == 16) {
            gn_imma(hl, av[0], bv[1]);
            gn_imma(lh, av[1], bv[0]);
            gn_imma(ll, av[1], bv[1]);
        }
#pragma unroll
        for (int q = 0; q < 4; q++) {
            h[q] += (float)hh[q];
            if (bits == 16)
                l[q] += (float)(hl[q] + lh[q]) * 256 + (float)ll[q];
        }
    }
#pragma unroll
    for (int q = 0; q < 4; q++) {
        int r = r0 + group + (q / 2) * 8, c = c0 + t * 2 + q % 2;
        if (r < M && c < N) {
            float value = bits == 16 ? h[q] * 65536 + l[q] : h[q];
            y[r * N + c] = value * as[r] * bs[c] + (add ? y[r * N + c] : 0);
        }
    }
}
#endif
