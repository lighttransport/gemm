#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

/* Per-row, per-32 activation quantization for SM120 native MXFP8. */
extern "C" __global__ void ds4f_cuda_quant_fp8_vec128(
        uint8_t *Q, uint8_t *scale, const float *X, int rows, int cols) {
    int r = blockIdx.y, b = blockIdx.x, lane = threadIdx.x;
    int k = b * 32 + lane;
    float v = r < rows && k < cols ? X[(size_t)r * cols + k] : 0.0f;
    float a = fabsf(v);
    for (int d = 16; d; d >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, d));
    a = __shfl_sync(0xffffffff, a, 0);
    int ep = a > 0.0f ? (int)ceilf(log2f(a / 448.0f)) : 0;
    ep = max(-127, min(127, ep));
    float s = ldexpf(1.0f, ep);
    if (lane == 0) {
        int nb=(cols+31)/32, nti=(nb+3)/4;
        size_t off=(size_t)((b>>2)+(r>>7)*nti)*512 +
                   (size_t)(r&31)*16 + (size_t)((r&127)>>5)*4 + (b&3);
        scale[off] = (uint8_t)(ep + 127);
    }
    if (r < rows && k < cols) {
        __nv_fp8_e4m3 q = __nv_fp8_e4m3(v / s);
        Q[(size_t)r * cols + k] = *(const uint8_t *)&q;
    }
}

__device__ __forceinline__ float ds4f_fp8_e4m3fn(uint8_t v) {
    int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    float x;
    if (e == 0) x = ldexpf((float)m, -9);
    else if (e == 15 && m == 7) x = 0.0f;
    else x = ldexpf(1.0f + (float)m * 0.125f, e - 7);
    return s ? -x : x;
}

__device__ __forceinline__ float ds4f_e8m0(uint8_t v) {
    return v == 0xff ? 1.0f : ldexpf(1.0f, (int)v - 127);
}

/* Compact DS4F FP8/E8M0 or BF16 weight GEMM. Each 256-thread block computes
 * a 64-token x 32-output tile using eight 16x16 tensor-core warps. Compact
 * weights are widened only into shared memory, never into the resident bank. */
extern "C" __global__ void ds4f_cuda_dense_gemm(
        float *Y, const void *Wv, const uint8_t *S, const float *X,
        int n_out, int n_in, int n_tok, int scale_cols, int kind) {
    __shared__ half As[64 * 16];
    __shared__ half Bs[32 * 16]; /* output-major: B[n][k] */
    int tid = threadIdx.x, warp = tid >> 5;
    int wm = warp >> 1, wn = warp & 1;
    int m0 = (int)blockIdx.y * 64, n0 = (int)blockIdx.x * 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < n_in; k0 += 16) {
        for (int e = tid; e < 64 * 16; e += 256) {
            int m = e >> 4, k = e & 15;
            int gm = m0 + m, gk = k0 + k;
            As[e] = __float2half_rn(gm < n_tok && gk < n_in
                ? X[(size_t)gm * n_in + gk] : 0.0f);
        }
        for (int e = tid; e < 32 * 16; e += 256) {
            int n = e >> 4, k = e & 15;
            int gn = n0 + n, gk = k0 + k;
            float v = 0.0f;
            if (gn < n_out && gk < n_in) {
                if (kind == 0) {
                    const uint8_t *W = (const uint8_t *)Wv;
                    v = ds4f_fp8_e4m3fn(W[(size_t)gn * n_in + gk]);
                    v *= ds4f_e8m0(S[(size_t)(gn >> 7) * scale_cols + (gk >> 7)]);
                } else {
                    const __nv_bfloat16 *W = (const __nv_bfloat16 *)Wv;
                    v = __bfloat162float(W[(size_t)gn * n_in + gk]);
                }
            }
            Bs[e] = __float2half_rn(v);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(a, As + wm * 16 * 16, 16);
        wmma::load_matrix_sync(b, Bs + wn * 16 * 16, 16);
        wmma::mma_sync(acc, a, b, acc);
        __syncthreads();
    }
    int gm = m0 + wm * 16, gn = n0 + wn * 16;
    if (gm + 15 < n_tok && gn + 15 < n_out)
        wmma::store_matrix_sync(Y + (size_t)gm * n_out + gn, acc, n_out,
                                wmma::mem_row_major);
    else {
        __shared__ float edge[8][16 * 16];
        wmma::store_matrix_sync(edge[warp], acc, 16, wmma::mem_row_major);
        for (int e = threadIdx.x & 31; e < 256; e += 32) {
            int m = e >> 4, n = e & 15;
            if (gm + m < n_tok && gn + n < n_out)
                Y[(size_t)(gm + m) * n_out + gn + n] = edge[warp][e];
        }
    }
}
