/* SPDX-License-Identifier: MIT
 * Original gfx1201 wave32 WMMA kernels. Packed [component, row, Kpad32]
 * operands are shared across four waves, with independent high/correction
 * accumulators to preserve the six-product training precision contract.
 */
#if defined(GN_HIP)
/* Coalesced transpose before rowwise integer scaling. A reused FP32 scratch
 * buffer holds an exact bitwise layout change, not an FP32 dot/reduction. */
extern "C" __global__ void gn_transpose_rows(float *out, const float *in, int R, int K) {
    __shared__ float tile[32][33];
    int r0 = blockIdx.y * 32, k0 = blockIdx.x * 32, t = threadIdx.x;
    for (int i = t; i < 1024; i += 256) {
        int r = i / 32, k = i % 32;
        tile[r][k] = r0 + k < R && k0 + r < K ? in[(k0 + r) * R + r0 + k] : 0;
    }
    __syncthreads();
    for (int i = t; i < 1024; i += 256) {
        int r = r0 + i / 32, k = k0 + i % 32;
        if (r < R && k < K)
            out[r * K + k] = tile[i % 32][i / 32];
    }
}
/* Default 81-token, 32-wide head: eight waves own eight queries and share one
 * staged K/V tile. Wave leaders use the CPU softmax order. */
extern "C" __global__ __launch_bounds__(384) void
gn_attention_81(float *y, float *prob, const float *x, const float *bias, int B, int side, int C,
                int D) {
    union Storage {
        struct {
            unsigned short q[2][32][32], k[2][96][32];
            float score[32][96];
        } qk;
        struct {
            unsigned short p[2][32][96], v[2][32][96];
        } pv;
    };
    __shared__ Storage s;
    int t = threadIdx.x, wave = t / 32, lane = t % 32, ix = lane & 15, half = lane / 16;
    int H = C / 32, groups = 3, group = blockIdx.x % groups;
    int h = blockIdx.x / groups % H, b = blockIdx.x / (groups * H), q0 = group * 32;
    for (int z = t; z < 32 * 32; z += 384) {
        int i = z / 32, d = z % 32, query = q0 + i;
        float value = query < 81 ? x[(b * 81 + query) * 3 * C + h * 32 + d] : 0;
        unsigned short high = bf(value);
        s.qk.q[0][i][d] = high;
        s.qk.q[1][i][d] = bf(value - unbf(high));
    }
    for (int z = t; z < 96 * 32; z += 384) {
        int j = z / 32, d = z % 32;
        float value = j < 81 ? x[(b * 81 + j) * 3 * C + C + h * 32 + d] : 0;
        unsigned short high = bf(value);
        s.qk.k[0][j][d] = high;
        s.qk.k[1][j][d] = bf(value - unbf(high));
    }
    __syncthreads();
    float8 acc = {};
    int query_tile = wave / 6, key_tile = wave % 6;
    ushort8 qh = *reinterpret_cast<ushort8 *>(&s.qk.q[0][query_tile * 16 + ix][half * 8]);
    ushort8 ql = *reinterpret_cast<ushort8 *>(&s.qk.q[1][query_tile * 16 + ix][half * 8]);
    ushort8 kh = *reinterpret_cast<ushort8 *>(&s.qk.k[0][key_tile * 16 + ix][half * 8]);
    ushort8 kl = *reinterpret_cast<ushort8 *>(&s.qk.k[1][key_tile * 16 + ix][half * 8]);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(qh, kh, acc);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ql, kh, acc);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(qh, kl, acc);
    qh = *reinterpret_cast<ushort8 *>(&s.qk.q[0][query_tile * 16 + ix][16 + half * 8]);
    ql = *reinterpret_cast<ushort8 *>(&s.qk.q[1][query_tile * 16 + ix][16 + half * 8]);
    kh = *reinterpret_cast<ushort8 *>(&s.qk.k[0][key_tile * 16 + ix][16 + half * 8]);
    kl = *reinterpret_cast<ushort8 *>(&s.qk.k[1][key_tile * 16 + ix][16 + half * 8]);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(qh, kh, acc);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ql, kh, acc);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(qh, kl, acc);
    for (int r = 0; r < 8; r++)
        s.qk.score[query_tile * 16 + half * 8 + r][key_tile * 16 + ix] = acc[r];
    __syncthreads();
    if (t < 32 && q0 + t < 81) {
        int query = q0 + t;
        float top = -INFINITY;
        for (int j = 0; j < 81; j++) {
            int rel = (query / 9 + 8 - j / 9) * 17 + query % 9 + 8 - j % 9;
            float value = s.qk.score[t][j] * rsqrtf((float)D) + bias[rel * H + h];
            s.qk.score[t][j] = value;
            top = fmaxf(top, value);
        }
        double sum = 0;
        for (int j = 0; j < 81; j++) {
            s.qk.score[t][j] = expf(s.qk.score[t][j] - top);
            sum += s.qk.score[t][j];
        }
        int qi = (b * H + h) * 81 + query;
        for (int j = 0; j < 81; j++) {
            s.qk.score[t][j] /= (float)sum;
            prob[qi * 81 + j] = s.qk.score[t][j];
        }
    }
    __syncthreads();
    for (int z = t; z < 32 * 96; z += 384) {
        int i = z / 96, j = z % 96;
        float value = q0 + i < 81 && j < 81 ? s.qk.score[i][j] : 0;
        unsigned short high = bf(value);
        s.pv.p[0][i][j] = high;
        s.pv.p[1][i][j] = bf(value - unbf(high));
    }
    __syncthreads();
    for (int z = t; z < 32 * 96; z += 384) {
        int d = z / 96, j = z % 96;
        float value = j < 81 ? x[(b * 81 + j) * 3 * C + 2 * C + h * 32 + d] : 0;
        unsigned short high = bf(value);
        s.pv.v[0][d][j] = high;
        s.pv.v[1][d][j] = bf(value - unbf(high));
    }
    __syncthreads();
    if (wave < 4) {
        int pv_query_tile = wave / 2, value_tile = wave % 2;
        float8 out = {};
        for (int k = 0; k < 96; k += 16) {
            ushort8 ph =
                *reinterpret_cast<ushort8 *>(&s.pv.p[0][pv_query_tile * 16 + ix][k + half * 8]);
            ushort8 pl =
                *reinterpret_cast<ushort8 *>(&s.pv.p[1][pv_query_tile * 16 + ix][k + half * 8]);
            ushort8 vh =
                *reinterpret_cast<ushort8 *>(&s.pv.v[0][value_tile * 16 + ix][k + half * 8]);
            ushort8 vl =
                *reinterpret_cast<ushort8 *>(&s.pv.v[1][value_tile * 16 + ix][k + half * 8]);
            out = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ph, vh, out);
            out = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(pl, vh, out);
            out = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ph, vl, out);
        }
        for (int r = 0; r < 8; r++) {
            int query = q0 + pv_query_tile * 16 + half * 8 + r, d = value_tile * 16 + ix;
            if (query < 81)
                y[(b * 81 + query) * C + h * 32 + d] = out[r];
        }
    }
    (void)B;
    (void)side;
}
/* Matching grouped backward score kernel. Eight query waves share V, reducing
 * CTA count and redundant global reads; per-query FP32 dp order and serial
 * double softmax-Jacobian dot match the CPU reference. */
extern "C" __global__ void gn_attention_scores_back_81(float *ds, const float *x, const float *dy,
                                                       const float *prob, int B, int side, int C,
                                                       int D) {
    __shared__ unsigned short dys[2][32][32], values[2][96][32];
    __shared__ float scores[32][96];
    int t = threadIdx.x, wave = t / 32, lane = t % 32, ix = lane & 15, half = lane / 16;
    int H = C / 32, groups = 3;
    int group = blockIdx.x % groups, h = blockIdx.x / groups % H;
    int b = blockIdx.x / (groups * H), q0 = group * 32;
    for (int z = t; z < 32 * 32; z += 384) {
        int i = z / 32, d = z % 32, query = q0 + i;
        float value = query < 81 ? dy[(b * 81 + query) * C + h * 32 + d] : 0;
        unsigned short high = bf(value);
        dys[0][i][d] = high;
        dys[1][i][d] = bf(value - unbf(high));
    }
    for (int z = t; z < 96 * 32; z += 384) {
        int j = z / 32, d = z % 32;
        float value = j < 81 ? x[(b * 81 + j) * 3 * C + 2 * C + h * 32 + d] : 0;
        unsigned short high = bf(value);
        values[0][j][d] = high;
        values[1][j][d] = bf(value - unbf(high));
    }
    __syncthreads();
    float8 acc = {};
    int query_tile = wave / 6, key_tile = wave % 6;
    ushort8 dh = *reinterpret_cast<ushort8 *>(&dys[0][query_tile * 16 + ix][half * 8]);
    ushort8 dl = *reinterpret_cast<ushort8 *>(&dys[1][query_tile * 16 + ix][half * 8]);
    ushort8 vh = *reinterpret_cast<ushort8 *>(&values[0][key_tile * 16 + ix][half * 8]);
    ushort8 vl = *reinterpret_cast<ushort8 *>(&values[1][key_tile * 16 + ix][half * 8]);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(dh, vh, acc);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(dl, vh, acc);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(dh, vl, acc);
    dh = *reinterpret_cast<ushort8 *>(&dys[0][query_tile * 16 + ix][16 + half * 8]);
    dl = *reinterpret_cast<ushort8 *>(&dys[1][query_tile * 16 + ix][16 + half * 8]);
    vh = *reinterpret_cast<ushort8 *>(&values[0][key_tile * 16 + ix][16 + half * 8]);
    vl = *reinterpret_cast<ushort8 *>(&values[1][key_tile * 16 + ix][16 + half * 8]);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(dh, vh, acc);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(dl, vh, acc);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(dh, vl, acc);
    for (int r = 0; r < 8; r++)
        scores[query_tile * 16 + half * 8 + r][key_tile * 16 + ix] = acc[r];
    __syncthreads();
    if (t < 32 && q0 + t < 81) {
        int query = (b * H + h) * 81 + q0 + t;
        double dot = 0;
        for (int j = 0; j < 81; j++)
            dot += scores[t][j] * prob[query * 81 + j];
        for (int j = 0; j < 81; j++)
            ds[query * 81 + j] = prob[query * 81 + j] * (scores[t][j] - (float)dot);
    }
    (void)B;
    (void)side;
    (void)D;
}
extern "C" __global__ __launch_bounds__(128) void gn_attention_qkv_back_81(
    float *dx, const float *x, const float *dy, const float *prob, const float *ds, int B, int side,
    int C, int D) {
    __shared__ unsigned short a[2][16][96], v[2][32][96];
    int t = threadIdx.x, wave = t / 32, lane = t % 32, ix = lane & 15, half = lane / 16;
    int H = C / 32, groups = 6, group = blockIdx.x % groups;
    int h = blockIdx.x / groups % H, b = blockIdx.x / (groups * H), row0 = group * 16;
    int base = (b * H + h) * 81 * 81;
    for (int phase = 0; phase < 3; phase++) {
        for (int z = t; z < 16 * 96; z += 128) {
            int r = z / 96, k = z % 96, row = row0 + r;
            float value = 0;
            if (row < 81 && k < 81)
                value = phase == 0   ? ds[base + row * 81 + k] * .1767766952966369f
                        : phase == 1 ? ds[base + k * 81 + row] * .1767766952966369f
                                     : prob[base + k * 81 + row];
            unsigned short high = bf(value);
            a[0][r][k] = high;
            a[1][r][k] = bf(value - unbf(high));
        }
        for (int z = t; z < 32 * 96; z += 128) {
            int d = z / 96, k = z % 96;
            float value = 0;
            if (k < 81)
                value = phase == 0   ? x[(b * 81 + k) * 3 * C + C + h * 32 + d]
                        : phase == 1 ? x[(b * 81 + k) * 3 * C + h * 32 + d]
                                     : dy[(b * 81 + k) * C + h * 32 + d];
            unsigned short high = bf(value);
            v[0][d][k] = high;
            v[1][d][k] = bf(value - unbf(high));
        }
        __syncthreads();
        if (wave < 2) {
            float8 out = {};
            for (int k = 0; k < 96; k += 16) {
                ushort8 ah = *reinterpret_cast<ushort8 *>(&a[0][ix][k + half * 8]);
                ushort8 al = *reinterpret_cast<ushort8 *>(&a[1][ix][k + half * 8]);
                ushort8 vh = *reinterpret_cast<ushort8 *>(&v[0][wave * 16 + ix][k + half * 8]);
                ushort8 vl = *reinterpret_cast<ushort8 *>(&v[1][wave * 16 + ix][k + half * 8]);
                out = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ah, vh, out);
                out = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(al, vh, out);
                out = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ah, vl, out);
            }
            for (int r = 0; r < 8; r++) {
                int row = row0 + half * 8 + r, d = wave * 16 + ix;
                if (row < 81)
                    dx[(b * 81 + row) * 3 * C + phase * C + h * 32 + d] += out[r];
            }
        }
        __syncthreads();
    }
    (void)B;
    (void)side;
    (void)D;
}
/* Generate packed im2col directly: no FP32 column buffer or second read.
 * Constant default-network dimensions strength-reduce integer indexing. */
template <int Channels = 0, int Side = 0, int Kernel = 0, bool Half = false>
__device__ __forceinline__ void gn_columns_pack(unsigned short *out, const float *x, int R, int C,
                                                int side, int kernel, int precise) {
    C = Channels ? Channels : C;
    side = Side ? Side : side;
    kernel = Kernel ? Kernel : kernel;
    int i = blockIdx.x * blockDim.x + threadIdx.x, K = C * kernel * kernel;
    int stride = (K + 31) & ~31;
    if (i >= R * stride)
        return;
    int row = i / stride, k = i % stride, S = side * side;
    int yy = row % S / side + k / (C * kernel) - kernel / 2;
    int xx = row % side + k / C % kernel - kernel / 2;
    float v = k < K && yy >= 0 && xx >= 0 && yy < side && xx < side
                  ? x[(row / S * S + yy * side + xx) * C + k % C]
                  : 0;
    unsigned short h = Half ? hf(v) : bf(v);
    out[i] = h;
    if (precise && !Half) {
        float residual = v - unbf(h);
        unsigned short low = bf(residual);
        out[R * stride + i] = low;
        if (precise != 2)
            out[2 * R * stride + i] = bf(residual - unbf(low));
    }
}
/* C256 keeps every four-channel vector within one spatial tap. Vector packing
 * amortizes row/tap address arithmetic and emits naturally aligned 64-bit
 * high/residual stores without changing the packed matrix layout. */
__device__ __forceinline__ void gn_columns_pack4_256(unsigned short *out, const float *x, int R,
                                                     int side, int kernel, int precise) {
    int vector = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int C = 256, Side = 9, Kernel = 3, K = C * Kernel * Kernel;
    constexpr int Vectors = K / 4;
    if (vector >= R * Vectors)
        return;
    int row = vector / Vectors, k = vector % Vectors * 4, position = row % (Side * Side);
    int yy = position / Side + k / (C * Kernel) - Kernel / 2;
    int xx = position % Side + k / C % Kernel - Kernel / 2;
    float4 value = {};
    if (yy >= 0 && xx >= 0 && yy < Side && xx < Side)
        value = *reinterpret_cast<const float4 *>(
            x + (row / (Side * Side) * (Side * Side) + yy * Side + xx) * C + k % C);
    ushort4 high = {bf(value.x), bf(value.y), bf(value.z), bf(value.w)};
    *reinterpret_cast<ushort4 *>(out + row * K + k) = high;
    if (precise) {
        ushort4 low = {bf(value.x - unbf(high.x)), bf(value.y - unbf(high.y)),
                       bf(value.z - unbf(high.z)), bf(value.w - unbf(high.w))};
        *reinterpret_cast<ushort4 *>(out + R * K + row * K + k) = low;
    }
    (void)side;
    (void)kernel;
}
extern "C" __global__ void gn_columns_fp16(unsigned short *out, const float *x, int R, int C,
                                           int side, int kernel, int precise) {
    if (side == 9 && C == 256 && kernel == 3)
        gn_columns_pack<256, 9, 3, true>(out, x, R, C, side, kernel, precise);
    else if (side == 9 && C == 80 && kernel == 5)
        gn_columns_pack<80, 9, 5, true>(out, x, R, C, side, kernel, precise);
    else
        gn_columns_pack<0, 0, 0, true>(out, x, R, C, side, kernel, precise);
}
extern "C" __global__ void gn_columns_bf16(unsigned short *out, const float *x, int R, int C,
                                           int side, int kernel, int precise) {
    if (side == 9 && C == 256 && kernel == 3)
        gn_columns_pack4_256(out, x, R, side, kernel, precise);
    else if (side == 9 && C == 80 && kernel == 5)
        gn_columns_pack<80, 9, 5>(out, x, R, C, side, kernel, precise);
    else
        gn_columns_pack<>(out, x, R, C, side, kernel, precise);
}
/* Backward dW needs im2col transposed. Gather a channel-coalesced tile
 * straight from NHWC, then transpose/round in LDS. */
template <int Channels = 0, int Side = 0, int Kernel = 0, bool Half = false>
__device__ __forceinline__ void gn_columns_pack_back(unsigned short *out, const float *x, int R,
                                                     int C, int side, int kernel, int precise) {
    C = Channels ? Channels : C;
    side = Side ? Side : side;
    kernel = Kernel ? Kernel : kernel;
    __shared__ float tile[32][33];
    int t = threadIdx.x, k0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    int K = C * kernel * kernel, stride = (R + 31) & ~31, S = side * side;
    for (int i = t; i < 1024; i += 256) {
        int r = r0 + i / 32, k = k0 + i % 32;
        int yy = r % S / side + k / (C * kernel) - kernel / 2;
        int xx = r % side + k / C % kernel - kernel / 2;
        tile[i / 32][i % 32] = r < R && k < K && yy >= 0 && xx >= 0 && yy < side && xx < side
                                   ? x[(r / S * S + yy * side + xx) * C + k % C]
                                   : 0;
    }
    __syncthreads();
    for (int i = t; i < 1024; i += 256) {
        int k = k0 + i / 32, r = r0 + i % 32;
        if (k >= K)
            continue;
        float v = tile[i % 32][i / 32];
        unsigned short high = Half ? hf(v) : bf(v);
        out[k * stride + r] = high;
        if (precise && !Half) {
            float residual = v - unbf(high);
            unsigned short low = bf(residual);
            out[(K + k) * stride + r] = low;
            if (precise != 2)
                out[(2 * K + k) * stride + r] = bf(residual - unbf(low));
        }
    }
}
__device__ __forceinline__ void gn_columns_pack_back4_fp16_256(unsigned short *out, const float *x,
                                                              int R) {
    __shared__ float tile[32][33];
    int t = threadIdx.x, k0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    constexpr int C = 256, Side = 9, Kernel = 3, K = C * Kernel * Kernel;
    int rr = t / 8, u = t % 8 * 4, row = r0 + rr, k = k0 + u;
    float4 value = {};
    int position = row % (Side * Side);
    int yy = position / Side + k / (C * Kernel) - Kernel / 2;
    int xx = position % Side + k / C % Kernel - Kernel / 2;
    if (row < R && k + 3 < K && yy >= 0 && xx >= 0 && yy < Side && xx < Side)
        value = *reinterpret_cast<const float4 *>(
            x + (row / (Side * Side) * (Side * Side) + yy * Side + xx) * C + k % C);
    tile[rr][u] = value.x;
    tile[rr][u + 1] = value.y;
    tile[rr][u + 2] = value.z;
    tile[rr][u + 3] = value.w;
    __syncthreads();
    int kk = k0 + t / 8, r = r0 + t % 8 * 4, stride = (R + 31) & ~31;
    if (kk < K) {
        ushort4 packed = {hf(tile[t % 8 * 4][t / 8]), hf(tile[t % 8 * 4 + 1][t / 8]),
                          hf(tile[t % 8 * 4 + 2][t / 8]), hf(tile[t % 8 * 4 + 3][t / 8])};
        *reinterpret_cast<ushort4 *>(out + kk * stride + r) = packed;
    }
}
extern "C" __global__ void gn_uncolumns4_256(float *dx, const float *col, int R, int C, int side,
                                             int kernel) {
    int vector = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int Channels = 256, Side = 9, Kernel = 3, K = Channels * Kernel * Kernel;
    if (vector >= R * Channels / 4)
        return;
    int row = vector / (Channels / 4), channel = vector % (Channels / 4) * 4;
    int position = row % (Side * Side), yy = position / Side, xx = position % Side;
    float4 sum = {};
    for (int ky = 0; ky < Kernel; ky++)
        for (int kx = 0; kx < Kernel; kx++) {
            int oy = yy - ky + Kernel / 2, ox = xx - kx + Kernel / 2;
            if (oy < 0 || ox < 0 || oy >= Side || ox >= Side)
                continue;
            float4 value = *reinterpret_cast<const float4 *>(
                col + (row / (Side * Side) * (Side * Side) + oy * Side + ox) * K +
                (ky * Kernel + kx) * Channels + channel);
            sum.x += value.x;
            sum.y += value.y;
            sum.z += value.z;
            sum.w += value.w;
        }
    float4 value = *reinterpret_cast<float4 *>(dx + row * Channels + channel);
    value.x += sum.x;
    value.y += sum.y;
    value.z += sum.z;
    value.w += sum.w;
    *reinterpret_cast<float4 *>(dx + row * Channels + channel) = value;
    (void)C;
    (void)side;
    (void)kernel;
}
extern "C" __global__ void gn_columns_fp16_back(unsigned short *out, const float *x, int R, int C,
                                                int side, int kernel, int precise) {
    if (side == 9 && C == 256 && kernel == 3)
        gn_columns_pack_back4_fp16_256(out, x, R);
    else if (side == 9 && C == 80 && kernel == 5)
        gn_columns_pack_back<80, 9, 5, true>(out, x, R, C, side, kernel, precise);
    else
        gn_columns_pack_back<0, 0, 0, true>(out, x, R, C, side, kernel, precise);
}
extern "C" __global__ void gn_columns_bf16_back(unsigned short *out, const float *x, int R, int C,
                                                int side, int kernel, int precise) {
    if (side == 9 && C == 256 && kernel == 3)
        gn_columns_pack_back<256, 9, 3>(out, x, R, C, side, kernel, precise);
    else if (side == 9 && C == 80 && kernel == 5)
        gn_columns_pack_back<80, 9, 5>(out, x, R, C, side, kernel, precise);
    else
        gn_columns_pack_back<>(out, x, R, C, side, kernel, precise);
}
/* Eight neighboring NHWC channels per CTA. Preserve double statistics and
 * FP32 normalization, while coalescing the formerly channel-strided loads. */
__device__ double gn_sum_channels(double v) {
    __shared__ double sums[256];
    int t = threadIdx.x;
    __syncthreads();
    sums[t] = v;
    __syncthreads();
    for (int d = 128; d >= 8; d /= 2) {
        if (t < d)
            sums[t] += sums[t + d];
        __syncthreads();
    }
    return sums[t % 8];
}
extern "C" __global__ void gn_bn_channels(float *y, float *aux, float *mean, float *var,
                                          const float *x, const float *w, const float *bias, int R,
                                          int C, int layer, int training) {
    int t = threadIdx.x, c = blockIdx.x * 8 + t % 8;
    double mu = 0, v = 0;
    if (training) {
        if (c < C)
            for (int r = t / 8; r < R; r += 32)
                mu += x[r * C + c];
        mu = gn_sum_channels(mu) / R;
        if (c < C)
            for (int r = t / 8; r < R; r += 32) {
                double d = x[r * C + c] - mu;
                v += d * d;
            }
        v = gn_sum_channels(v) / R;
        if (t < 8 && c < C) {
            mean[c] = .9f * mean[c] + .1f * (float)mu;
            var[c] = .9f * var[c] + .1f * (float)(R > 1 ? v * R / (R - 1) : v);
        }
    } else if (c < C) {
        mu = mean[c];
        v = var[c];
    }
    float inv = rsqrtf((float)v + 1e-5f);
    if (t < 8 && c < C) {
        aux[c] = (float)mu;
        aux[C + c] = inv;
    }
    if (c < C)
        for (int r = t / 8; r < R; r += 32)
            y[r * C + c] = (x[r * C + c] - (float)mu) * inv * w[c] + bias[c];
    (void)layer;
}
extern "C" __global__ void gn_bn_back_channels(float *dx, float *dw, float *db, const float *x,
                                               const float *dy, const float *w, const float *aux,
                                               int R, int C, int layer) {
    int t = threadIdx.x, c = blockIdx.x * 8 + t % 8;
    float mu = c < C ? aux[c] : 0, inv = c < C ? aux[C + c] : 0;
    double sum = 0, prod = 0;
    if (c < C)
        for (int r = t / 8; r < R; r += 32) {
            int j = r * C + c;
            sum += dy[j];
            prod += dy[j] * (x[j] - mu) * inv;
        }
    sum = gn_sum_channels(sum);
    prod = gn_sum_channels(prod);
    if (t < 8 && c < C) {
        dw[c] += (float)prod;
        db[c] += (float)sum;
    }
    if (c < C)
        for (int r = t / 8; r < R; r += 32) {
            int j = r * C + c;
            dx[j] +=
                inv * w[c] * (dy[j] - (float)(sum / R) - (x[j] - mu) * inv * (float)(prod / R));
        }
    (void)layer;
}
extern "C" __global__ void gn_bn_silu_channels(float *mid, float *y, float *aux, float *mean,
                                                float *var, float *x, const float *prebias,
                                                const float *w, const float *bias, int R, int C,
                                                int training) {
    int t = threadIdx.x, c = blockIdx.x * 8 + t % 8;
    double mu = 0, v = 0;
    if (training) {
        if (c < C)
            for (int r = t / 8; r < R; r += 32)
                mu += (x[r * C + c] = x[r * C + c] + prebias[c]);
        mu = gn_sum_channels(mu) / R;
        if (c < C)
            for (int r = t / 8; r < R; r += 32) {
                double d = x[r * C + c] - mu;
                v += d * d;
            }
        v = gn_sum_channels(v) / R;
        if (t < 8 && c < C) {
            mean[c] = .9f * mean[c] + .1f * (float)mu;
            var[c] = .9f * var[c] + .1f * (float)(R > 1 ? v * R / (R - 1) : v);
        }
    } else if (c < C) {
        mu = mean[c];
        v = var[c];
        for (int r = t / 8; r < R; r += 32)
            x[r * C + c] += prebias[c];
    }
    float inv = rsqrtf((float)v + 1e-5f);
    if (t < 8 && c < C) {
        aux[c] = (float)mu;
        aux[C + c] = inv;
    }
    if (c < C)
        for (int r = t / 8; r < R; r += 32) {
            int j = r * C + c;
            float z = (x[j] - (float)mu) * inv * w[c] + bias[c];
            mid[j] = z;
            y[j] = z / (1 + expf(-z));
        }
}
extern "C" __global__ void gn_bn_silu_back_channels(float *dx, float *dw, float *db,
                                                     const float *x, const float *mid,
                                                     const float *dy, const float *w,
                                                     const float *aux, float *conv_db, int R, int C) {
    int t = threadIdx.x, c = blockIdx.x * 8 + t % 8;
    float mu = c < C ? aux[c] : 0, inv = c < C ? aux[C + c] : 0;
    double sum = 0, prod = 0;
    if (c < C)
        for (int r = t / 8; r < R; r += 32) {
            int j = r * C + c;
            float s = 1 / (1 + expf(-mid[j]));
            float d = dy[j] * s * (1 + mid[j] * (1 - s));
            sum += d;
            prod += d * (x[j] - mu) * inv;
        }
    sum = gn_sum_channels(sum);
    prod = gn_sum_channels(prod);
    if (t < 8 && c < C) {
        dw[c] += (float)prod;
        db[c] += (float)sum;
    }
    double conv_sum = 0;
    if (c < C)
        for (int r = t / 8; r < R; r += 32) {
            int j = r * C + c;
            float s = 1 / (1 + expf(-mid[j]));
            float d = dy[j] * s * (1 + mid[j] * (1 - s));
            float v = inv * w[c] *
                      (d - (float)(sum / R) - (x[j] - mu) * inv * (float)(prod / R));
            dx[j] += v;
            conv_sum += v;
        }
    conv_sum = gn_sum_channels(conv_sum);
    if (t < 8 && c < C)
        conv_db[c] += (float)conv_sum;
}
extern "C" __global__ void gn_bias_back_parallel(float *db, const float *dy, int R, int C) {
    __shared__ double sums[256];
    int t = threadIdx.x, c = blockIdx.x * 8 + t % 8;
    double sum = 0;
    if (c < C)
        for (int r = t / 8; r < R; r += 32)
            sum += dy[r * C + c];
    sums[t] = sum;
    __syncthreads();
    for (int d = 128; d >= 8; d /= 2) {
        if (t < d)
            sums[t] += sums[t + d];
        __syncthreads();
    }
    if (t < 8 && c < C)
        db[c] += (float)sums[t];
}
extern "C" __global__ void gn_grad_norm_parallel(float *norm, const float *g, int count,
                                                 float inv) {
    float sum = 0;
    int invalid = 0;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < count; i += gridDim.x * 256) {
        float v = g[i] * inv;
        if (!isfinite(v))
            invalid = 1;
        sum += v * v;
    }
    sum = gn_block_sum(sum);
    invalid = gn_block_sum(invalid);
    if (!threadIdx.x) {
        atomicAdd(norm, sum);
        if (invalid)
            atomicExch(norm + 1, 1);
    }
}
struct gn_norm_desc {
    unsigned long long grad;
    unsigned offset, stride, count, pad;
};
extern "C" __global__ void gn_grad_norm_multi(float *norm, const gn_norm_desc *descriptors,
                                              unsigned descriptor_count, float inv) {
    if (blockIdx.x >= descriptor_count)
        return;
    gn_norm_desc d = descriptors[blockIdx.x];
    const float *g = reinterpret_cast<const float *>(d.grad);
    float sum = 0;
    int invalid = 0;
    for (unsigned i = d.offset + threadIdx.x; i < d.count; i += d.stride) {
        float v = g[i] * inv;
        invalid |= !isfinite(v);
        sum += v * v;
    }
    sum = gn_block_sum(sum);
    invalid = gn_block_sum(invalid);
    if (!threadIdx.x) {
        atomicAdd(norm, sum);
        if (invalid)
            atomicExch(norm + 1, 1);
    }
}
struct gn_adam_desc {
    unsigned long long x, mom, var, grad;
    unsigned offset, count;
};
extern "C" __global__ void gn_adam_multi(const gn_adam_desc *descriptors,
                                         unsigned descriptor_count, float scale, float lr,
                                         float decay, float b1, float b2) {
    if (blockIdx.x >= descriptor_count)
        return;
    gn_adam_desc d = descriptors[blockIdx.x];
    unsigned i = d.offset + threadIdx.x;
    if (i >= d.count)
        return;
    float *x = reinterpret_cast<float *>(d.x), *m = reinterpret_cast<float *>(d.mom),
          *v = reinterpret_cast<float *>(d.var), *g = reinterpret_cast<float *>(d.grad);
    float grad = g[i] * scale;
    float a = .9f * m[i] + .1f * grad, q = .999f * v[i] + .001f * grad * grad;
    m[i] = a;
    v[i] = q;
    x[i] -= lr * ((a / b1) / (sqrtf(q / b2) + 1e-8f) + decay * x[i]);
    g[i] = 0;
}
extern "C" __global__ void gn_lt_combine(float *y, const float *high, const float *low, int count,
                                         int add) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        y[i] = (high[i] + low[i]) + (add ? y[i] : 0);
}
extern "C" __global__ void gn_lt_combine_bias(float *y, const float *high, const float *low,
                                               const float *bias, int count, int N, int add) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        y[i] = (high[i] + low[i]) + bias[i % N] + (add ? y[i] : 0);
}
/* AccChunk=-1: FP32 accumulators; 0: native BF16 throughout the dot;
 * positive: native BF16 partial dots, widened once per AccChunk products.
 * This changes the training arithmetic and is always opt-in. */
template <bool Precise, int MR = 2, int NR = 2, int BK = 32, int AccChunk = -1, int Products = 6,
          bool Separate = true>
__device__ void gn_rdna4_body(float *y, const unsigned short *a, const unsigned short *b, int M,
                              int N, int K, int add) {
    static_assert(!Precise || AccChunk < 0, "BF16 accumulation changes the precision contract");
    constexpr int BM = 32 * MR, BN = 32 * NR, Planes = Precise ? (Products == 3 ? 2 : 3) : 1;
    __shared__ unsigned short sa[Planes][BM][BK + 8] __attribute__((aligned(16)));
    __shared__ unsigned short sb[Planes][BN][BK + 8] __attribute__((aligned(16)));
    int t = threadIdx.x, lane = t & 31, wave = t / 32, ix = lane & 15, half = lane / 16;
    int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;
    int wr = (wave / 2) * 16 * MR, wc = (wave % 2) * 16 * NR, stride = (K + 31) & ~31;
    float8 high[MR][NR] = {}, low[MR][NR] = {};
    ushort8 narrow[MR][NR] = {};
    for (int k = 0; k < stride; k += BK) {
#pragma unroll
        for (int p = 0; p < Planes; p++) {
            for (int q = t; q < BM * (BK / 8); q += 128) {
                int r = q / (BK / 8), c = (q % (BK / 8)) * 8;
                *reinterpret_cast<ushort8 *>(&sa[p][r][c]) =
                    r0 + r < M && k + c < stride
                        ? *reinterpret_cast<const ushort8 *>(a + (p * M + r0 + r) * stride + k + c)
                        : ushort8{};
            }
            for (int q = t; q < BN * (BK / 8); q += 128) {
                int r = q / (BK / 8), c = (q % (BK / 8)) * 8;
                *reinterpret_cast<ushort8 *>(&sb[p][r][c]) =
                    c0 + r < N && k + c < stride
                        ? *reinterpret_cast<const ushort8 *>(b + (p * N + c0 + r) * stride + k + c)
                        : ushort8{};
            }
        }
        __syncthreads();
#pragma unroll
        for (int sub = 0; sub < BK; sub += 16) {
            ushort8 av[Planes][MR], bv[Planes][NR];
#pragma unroll
            for (int p = 0; p < Planes; p++) {
#pragma unroll
                for (int i = 0; i < MR; i++)
                    av[p][i] =
                        *reinterpret_cast<ushort8 *>(&sa[p][wr + i * 16 + ix][sub + half * 8]);
#pragma unroll
                for (int j = 0; j < NR; j++)
                    bv[p][j] =
                        *reinterpret_cast<ushort8 *>(&sb[p][wc + j * 16 + ix][sub + half * 8]);
            }
#pragma unroll
            for (int i = 0; i < MR; i++) {
#pragma unroll
                for (int j = 0; j < NR; j++) {
                    if constexpr (Precise && Products == 3 && !Separate)
                        high[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                            av[0][i], bv[0][j], high[i][j]);
                    if constexpr (Precise) {
                        if constexpr (Separate) {
                            low[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                                av[1][i], bv[0][j], low[i][j]);
                            low[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                                av[0][i], bv[1][j], low[i][j]);
                        } else {
                            high[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                                av[1][i], bv[0][j], high[i][j]);
                            high[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                                av[0][i], bv[1][j], high[i][j]);
                        }
                        if constexpr (Products != 3) {
                            low[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                                av[1][i], bv[1][j], low[i][j]);
                            low[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                                av[2][i], bv[0][j], low[i][j]);
                            low[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                                av[0][i], bv[2][j], low[i][j]);
                        }
                    }
                    if constexpr (AccChunk < 0 && !(Precise && Products == 3 && !Separate)) {
                        high[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                            av[0][i], bv[0][j], high[i][j]);
                    } else {
                        narrow[i][j] = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(
                            av[0][i], bv[0][j], narrow[i][j]);
                        if constexpr (AccChunk > 0) {
                            if ((k + sub + 16) % AccChunk == 0) {
#pragma unroll
                                for (int q = 0; q < 8; q++)
                                    high[i][j][q] += unbf(narrow[i][j][q]);
                                narrow[i][j] = ushort8{};
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < MR; i++)
#pragma unroll
        for (int j = 0; j < NR; j++)
#pragma unroll
            for (int q = 0; q < 8; q++) {
                int r = r0 + wr + 16 * i + half * 8 + q, c = c0 + wc + 16 * j + ix;
                if (r < M && c < N) {
                    float v = AccChunk == 0  ? unbf(narrow[i][j][q])
                              : AccChunk > 0 ? high[i][j][q] + unbf(narrow[i][j][q])
                              : Precise && Products == 3 && !Separate
                                  ? high[i][j][q]
                                  : high[i][j][q] + low[i][j][q];
                    y[r * N + c] = v + (add ? y[r * N + c] : 0);
                }
            }
}
extern "C" __global__ __launch_bounds__(128) void gn_mm_tiled(float *y, const unsigned short *a,
                                                              const unsigned short *b, int M, int N,
                                                              int K, int add) {
    gn_rdna4_body<true, 1, 1>(y, a, b, M, N, K, add);
}
extern "C" __global__ __launch_bounds__(128) void gn_mm_tiled_fast(float *y,
                                                                   const unsigned short *a,
                                                                   const unsigned short *b, int M,
                                                                   int N, int K, int add) {
    gn_rdna4_body<false>(y, a, b, M, N, K, add);
}
/* Two BF16 components, three products. A single FP32 accumulator bank admits
 * fewer registers; qualification covers the altered but still-FP32 sum order. */
extern "C" __global__ __launch_bounds__(128) void gn_mm_bf16x3(float *y, const unsigned short *a,
                                                               const unsigned short *b, int M,
                                                               int N, int K, int add) {
    gn_rdna4_body<true, 1, 1, 32, -1, 3, false>(y, a, b, M, N, K, add);
}
extern "C" __global__ __launch_bounds__(128) void gn_mm_bf16_acc(float *y, const unsigned short *a,
                                                                 const unsigned short *b, int M,
                                                                 int N, int K, int add, int chunk) {
    if (chunk)
        gn_rdna4_body<false, 2, 2, 32, 128>(y, a, b, M, N, K, add);
    else
        gn_rdna4_body<false, 2, 2, 32, 0>(y, a, b, M, N, K, add);
}

/* Row quantization. INT16 is a signed high byte and UNSIGNED low byte:
 * q = 256*hi + lo. The four WMMA products below cover the complete INT16
 * range, including -32768 in the exact-integer diagnostic. */
extern "C" __global__ void gn_pack_integer(signed char *out, float *scales, const float *in, int R,
                                           int K, int trans, int bits) {
    int r = blockIdx.x, t = threadIdx.x, stride = (K + 31) & ~31;
    float peak = 0;
    for (int k = t; k < K; k += 256)
        peak = fmaxf(peak, fabsf(in[trans ? k * R + r : r * K + k]));
    peak = gn_block_max(peak);
    int bound = bits == 8 ? 127 : 32767;
    float scale = peak > 0 ? fmaxf(peak / bound, 1e-30f) : 1;
    if (!t)
        scales[r] = scale;
    for (int k = t; k < stride; k += 256) {
        float x = k < K ? in[trans ? k * R + r : r * K + k] : 0;
        int q = max(-bound, min(bound, __float2int_rn(x / scale)));
        out[r * stride + k] = (signed char)(bits == 8 ? q : q >> 8);
        if (bits == 16)
            out[(R + r) * stride + k] = (signed char)(q & 255);
    }
}
typedef int gn_int2 __attribute__((ext_vector_type(2)));
typedef int gn_int4 __attribute__((ext_vector_type(4)));
typedef int gn_int8 __attribute__((ext_vector_type(8)));
template <int Bits, bool Wide, int MR = 2, int NR = 2>
__device__ __forceinline__ void
gn_rdna4_integer(float *y, const signed char *a, const signed char *b, const float *as,
                 const float *bs, int M, int N, int K, int add, long long *exact) {
    constexpr int BM = 32 * MR, BN = 32 * NR, P = Bits / 8;
    __shared__ signed char sa[P][BM][48] __attribute__((aligned(16)));
    __shared__ signed char sb[P][BN][48] __attribute__((aligned(16)));
    int t = threadIdx.x, lane = t & 31, wave = t / 32, ix = lane & 15, half = lane / 16;
    int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN, stride = (K + 31) & ~31;
    int wr = (wave / 2) * 16 * MR, wc = (wave % 2) * 16 * NR;
    gn_int8 hh[MR][NR] = {}, hl[MR][NR] = {}, lh[MR][NR] = {}, ll[MR][NR] = {};
    long long wide[MR][NR][8] = {};
    for (int k = 0; k < stride; k += 32) {
#pragma unroll
        for (int p = 0; p < P; p++) {
            for (int q = t; q < BM * 2; q += 128) {
                int r = q / 2, c = (q % 2) * 16;
                *reinterpret_cast<gn_int4 *>(&sa[p][r][c]) =
                    r0 + r < M
                        ? *reinterpret_cast<const gn_int4 *>(a + (p * M + r0 + r) * stride + k + c)
                        : gn_int4{};
            }
            for (int q = t; q < BN * 2; q += 128) {
                int r = q / 2, c = (q % 2) * 16;
                *reinterpret_cast<gn_int4 *>(&sb[p][r][c]) =
                    c0 + r < N
                        ? *reinterpret_cast<const gn_int4 *>(b + (p * N + c0 + r) * stride + k + c)
                        : gn_int4{};
            }
        }
        __syncthreads();
        /* Two K=16 issues consume each 128-bit fragment. The integer dot
         * is exact, so permuting its K terms cannot change the result. */
        gn_int4 av[P][MR], bv[P][NR];
#pragma unroll
        for (int p = 0; p < P; p++) {
#pragma unroll
            for (int i = 0; i < MR; i++)
                av[p][i] = *reinterpret_cast<gn_int4 *>(&sa[p][wr + 16 * i + ix][half * 16]);
#pragma unroll
            for (int j = 0; j < NR; j++)
                bv[p][j] = *reinterpret_cast<gn_int4 *>(&sb[p][wc + 16 * j + ix][half * 16]);
        }
#pragma unroll
        for (int s = 0; s < 2; s++)
#pragma unroll
            for (int i = 0; i < MR; i++)
#pragma unroll
                for (int j = 0; j < NR; j++) {
                    gn_int2 ah = {av[0][i][2 * s], av[0][i][2 * s + 1]};
                    gn_int2 bh = {bv[0][j][2 * s], bv[0][j][2 * s + 1]};
                    hh[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, ah, true, bh,
                                                                                hh[i][j], false);
                    if constexpr (Bits == 16) {
                        gn_int2 al = {av[1][i][2 * s], av[1][i][2 * s + 1]};
                        gn_int2 bl = {bv[1][j][2 * s], bv[1][j][2 * s + 1]};
                        hl[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                            true, ah, false, bl, hl[i][j], false);
                        lh[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                            false, al, true, bh, lh[i][j], false);
                        ll[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                            false, al, false, bl, ll[i][j], false);
                    }
                }
        /* The largest partial is unsigned lo*lo: 16384*255^2 < INT32_MAX.
         * Widen BEFORE weighting or adding cross terms. No FP32 reduction,
         * no signed-overflow arithmetic, and no saturating/wrapping dots. */
        if constexpr (Wide || Bits == 16) {
            if ((k + 32) % 16384 == 0 || k + 32 == stride) {
#pragma unroll
                for (int i = 0; i < MR; i++)
#pragma unroll
                    for (int j = 0; j < NR; j++) {
#pragma unroll
                        for (int q = 0; q < 8; q++) {
                            if constexpr (Bits == 16)
                                wide[i][j][q] += (long long)hh[i][j][q] * 65536 +
                                                 ((long long)hl[i][j][q] + lh[i][j][q]) * 256 +
                                                 ll[i][j][q];
                            else
                                wide[i][j][q] += hh[i][j][q];
                        }
                        hh[i][j] = hl[i][j] = lh[i][j] = ll[i][j] = gn_int8{};
                    }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < MR; i++)
#pragma unroll
        for (int j = 0; j < NR; j++)
#pragma unroll
            for (int q = 0; q < 8; q++) {
                int r = r0 + wr + i * 16 + half * 8 + q, c = c0 + wc + j * 16 + ix;
                if (r < M && c < N) {
                    long long v = Wide || Bits == 16 ? wide[i][j][q] : (long long)hh[i][j][q];
                    if (exact)
                        exact[r * N + c] = v;
                    y[r * N + c] = (float)v * as[r] * bs[c] + (add ? y[r * N + c] : 0);
                }
            }
}
extern "C" __global__
__launch_bounds__(128) void gn_mm_integer(float *y, const signed char *a, const signed char *b,
                                          const float *as, const float *bs, int M, int N, int K,
                                          int add, int bits, int wide, long long *exact) {
    if (bits == 16)
        gn_rdna4_integer<16, true, 2, 1>(y, a, b, as, bs, M, N, K, add, exact);
    else if (wide || K > 131040) /* Account for Kpad32 in the signed INT8 bound. */
        gn_rdna4_integer<8, true>(y, a, b, as, bs, M, N, K, add, exact);
    else
        gn_rdna4_integer<8, false>(y, a, b, as, bs, M, N, K, add, exact);
}
#endif
