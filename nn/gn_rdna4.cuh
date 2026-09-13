/* SPDX-License-Identifier: MIT
 * Original gfx1201 wave32 WMMA kernels. Packed [component, row, Kpad32]
 * operands are shared across four waves, with independent high/correction
 * accumulators to preserve the six-product training precision contract.
 */
#if defined(GN_HIP)
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
extern "C" __global__ void gn_lt_combine(float *y, const float *high, const float *low, int count,
                                         int add) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        y[i] = (high[i] + low[i]) + (add ? y[i] : 0);
}
template <bool Precise, int MR = 2, int NR = 2, int BK = 32>
__device__ void gn_rdna4_body(float *y, const unsigned short *a, const unsigned short *b, int M,
                              int N, int K, int add) {
    constexpr int BM = 32 * MR, BN = 32 * NR, Planes = Precise ? 3 : 1;
    __shared__ unsigned short sa[Planes][BM][BK + 8] __attribute__((aligned(16)));
    __shared__ unsigned short sb[Planes][BN][BK + 8] __attribute__((aligned(16)));
    int t = threadIdx.x, lane = t & 31, wave = t / 32, ix = lane & 15, half = lane / 16;
    int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;
    int wr = (wave / 2) * 16 * MR, wc = (wave % 2) * 16 * NR, stride = (K + 31) & ~31;
    float8 high[MR][NR] = {}, low[MR][NR] = {};
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
                    if (Precise) {
                        low[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                            av[1][i], bv[0][j], low[i][j]);
                        low[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                            av[0][i], bv[1][j], low[i][j]);
                        low[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                            av[1][i], bv[1][j], low[i][j]);
                        low[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                            av[2][i], bv[0][j], low[i][j]);
                        low[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                            av[0][i], bv[2][j], low[i][j]);
                    }
                    high[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                        av[0][i], bv[0][j], high[i][j]);
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
                if (r < M && c < N)
                    y[r * N + c] = (high[i][j][q] + low[i][j][q]) + (add ? y[r * N + c] : 0);
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
#endif
