// SPDX-License-Identifier: MIT
// W4A8 experiment for the native Edit-2511 DiT: INT4-g64 weights x per-token INT8 activations on gfx12 INT8 WMMA,
// INT32 accumulation per 64-wide scale group folded into f32 with the group's weight scale. Compared against the
// runner's W4A16 kernel (qgemm_i4w_128x128w8, BF16 activations) and an exact f32 reference on the same weights.
// usage: bench_w4a8   (shapes: the 1024^2 + one-reference image stream, M = 8192 tokens)
#include <hip/hip_runtime.h>
#pragma clang diagnostic ignored "-Wunused-result"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include "qimg_gemm_wmma.hip"

typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x2 __attribute__((ext_vector_type(2)));

/* per-token symmetric INT8 quantization of f32 rows: one 256-thread block per row */
extern "C" __global__ void quant_rows_i8(signed char *Xq, float *xs, const float *X, int K) {
    __shared__ float red[8];
    const float *x = X + (long)blockIdx.x * K; signed char *q = Xq + (long)blockIdx.x * K;
    float m = 0.f;
    for (int k = threadIdx.x * 4; k < K; k += 1024) { float4 v = *(const float4 *)(x + k); m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)))); }
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m; __syncthreads();
    m = 0.f; for (int i = 0; i < 8; i++) m = fmaxf(m, red[i]);
    const float s = m > 0.f ? m / 127.f : 1.f, inv = 1.f / s;
    if (threadIdx.x == 0) xs[blockIdx.x] = s;
    for (int k = threadIdx.x * 4; k < K; k += 1024) {
        float4 v = *(const float4 *)(x + k);
        char4 c = make_char4((signed char)__float2int_rn(v.x * inv), (signed char)__float2int_rn(v.y * inv),
                             (signed char)__float2int_rn(v.z * inv), (signed char)__float2int_rn(v.w * inv));
        *(char4 *)(q + k) = c;
    }
}

#define W_BM 128
#define W_BN 128
#define W_BK 64
#define W_LD 80
/* Y[M,N] f32 (+)= (Xq*xs) . (Wq4*ws)^T + bias. Wq4 [N, K/2] (even k -> low nibble, signed), ws bf16 [N, K/64]. */
extern "C" __global__ __launch_bounds__(256) void gemm_w4a8_g64(float *Y, const unsigned char *Wq, const unsigned short *ws,
        const signed char *Xq, const float *xs, const float *bias, int N, int K, int M, int accum) {
    __shared__ __attribute__((aligned(16))) signed char As[2][W_BM * W_LD];
    __shared__ __attribute__((aligned(16))) signed char Bs[2][W_BN * W_LD];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, wM = wave & 1, wN = wave >> 1;
    const int half = lane >> 4, idx = lane & 15;
    /* grouped rasterization (GROUP_M = 8) over a 1-D grid, as in qgemm_body */
    const int num_m = (M + W_BM - 1) / W_BM, num_n = (N + W_BN - 1) / W_BN, per_group = 8 * num_n;
    const int pid = blockIdx.x, group = pid / per_group, first_m = group * 8;
    const int gsize = (num_m - first_m) < 8 ? (num_m - first_m) : 8;
    const int m0 = (first_m + (pid % per_group) % gsize) * W_BM, n0 = ((pid % per_group) / gsize) * W_BN;
    int4 ra[2]; uint4 rb;
    auto load = [&](int k) {
        for (int i = 0; i < 2; i++) {
            int c = tid + i * 256, r = c >> 2, col = (c & 3) * 16, row = m0 + r;
            ra[i] = row < M ? *(const int4 *)(Xq + (long)row * K + k + col) : make_int4(0, 0, 0, 0);
        }
        int r = tid >> 1, part = tid & 1, orow = n0 + r;   /* 32 packed bytes per W row per K tile: two 16-byte halves */
        rb = orow < N ? *(const uint4 *)(Wq + (long)orow * (K / 2) + k / 2 + part * 16) : make_uint4(0, 0, 0, 0);
    };
    auto store = [&](int buf) {
        for (int i = 0; i < 2; i++) { int c = tid + i * 256, r = c >> 2, col = (c & 3) * 16; *(int4 *)(&As[buf][r * W_LD + col]) = ra[i]; }
        int r = tid >> 1, part = tid & 1;
        unsigned wds[4] = {rb.x, rb.y, rb.z, rb.w}; unsigned out[8];
#pragma unroll
        for (int w = 0; w < 4; w++) {   /* 8 nibbles -> 8 sign-extended int8 */
            unsigned v = wds[w], lo = 0, hi = 0;
#pragma unroll
            for (int n = 0; n < 4; n++) lo |= (unsigned)(unsigned char)(signed char)((int)(v << (28 - 4 * n)) >> 28) << (8 * n);
#pragma unroll
            for (int n = 0; n < 4; n++) hi |= (unsigned)(unsigned char)(signed char)((int)(v << (12 - 4 * n)) >> 28) << (8 * n);
            out[2 * w] = lo; out[2 * w + 1] = hi;
        }
        *(uint4 *)(&Bs[buf][r * W_LD + part * 32]) = make_uint4(out[0], out[1], out[2], out[3]);
        *(uint4 *)(&Bs[buf][r * W_LD + part * 32 + 16]) = make_uint4(out[4], out[5], out[6], out[7]);
    };
    float acc[4][2][8];
#pragma unroll
    for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) for (int e = 0; e < 8; e++) acc[i][j][e] = 0.f;
    const int wn0 = n0 + wN * 32;
    const unsigned short *wsc[2] = {ws + (long)min(wn0 + idx, N - 1) * (K / 64), ws + (long)min(wn0 + 16 + idx, N - 1) * (K / 64)};
    load(0); store(0); __syncthreads();
    int buf = 0;
    for (int k = 0; k < K; k += W_BK) {
        const bool more = k + W_BK < K;
        if (more) load(k + W_BK);
        const signed char *A = As[buf], *B = Bs[buf];
        i32x8 g[4][2];
#pragma unroll
        for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) g[i][j] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < W_BK; ks += 16) {
            u32x2 a[4], b[2];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = *(const u32x2 *)(A + (wM * 64 + i * 16 + idx) * W_LD + ks + half * 8);
#pragma unroll
            for (int j = 0; j < 2; j++) b[j] = *(const u32x2 *)(B + (wN * 32 + j * 16 + idx) * W_LD + ks + half * 8);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 2; j++) g[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a[i], true, b[j], g[i][j], false);
        }
        const int grp = k / 64;
        float s[2] = {__uint_as_float((unsigned)wsc[0][grp] << 16), __uint_as_float((unsigned)wsc[1][grp] << 16)};
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int e = 0; e < 8; e++) acc[i][j][e] += (float)g[i][j][e] * s[j];
        if (more) { store(buf ^ 1); __syncthreads(); buf ^= 1; }
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++) {
            int col = wn0 + j * 16 + idx; if (col >= N) continue;
            float bv = bias ? bias[col] : 0.f;
            for (int e = 0; e < 8; e++) {
                int row = m0 + wM * 64 + i * 16 + half * 8 + e; if (row >= M) continue;
                float *y = Y + (long)row * N + col;
                float o = acc[i][j][e] * xs[row] + bv;
                if (accum & 1) o += *y;
                *y = o;
            }
        }
}

/* exact reference: Y = X . dequant(W)^T in f32 (naive, for error measurement) */
extern "C" __global__ void ref_gemm(float *Y, const float *X, const unsigned char *Wq, const unsigned short *ws, int N, int K, int M) {
    int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y; if (n >= N) return;
    const float *x = X + (long)m * K; const unsigned char *w = Wq + (long)n * (K / 2); float acc = 0.f;
    for (int k = 0; k < K; k += 2) {
        unsigned char b = w[k / 2]; float s = __uint_as_float((unsigned)ws[(long)n * (K / 64) + k / 64] << 16);
        int lo = (int)(b & 15), hi = (int)(b >> 4); lo = lo >= 8 ? lo - 16 : lo; hi = hi >= 8 ? hi - 16 : hi;
        acc += x[k] * lo * s + x[k + 1] * hi * s;
    }
    Y[(long)m * N + n] = acc;
}

int main() {
    int shapes[][2] = {{3072, 3072}, {12288, 3072}, {3072, 12288}};   /* N, K */
    const int M = 8192;
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    for (auto &sh : shapes) {
        int N = sh[0], K = sh[1];
        std::vector<float> hx((size_t)M * K); for (auto &v : hx) v = nd(rng) * (1.f + 3.f * (rng() % 64 == 0));   /* some outlier channels */
        std::vector<unsigned char> hw((size_t)N * K / 2); for (auto &v : hw) v = (unsigned char)rng();
        std::vector<unsigned short> hs((size_t)N * K / 64); for (auto &v : hs) { float f = 0.01f + 0.01f * (rng() % 100) / 100.f; unsigned u; memcpy(&u, &f, 4); v = (unsigned short)(u >> 16); }
        float *X, *Y0, *Y1, *Y2, *xs; signed char *Xq; unsigned short *Xb, *S; unsigned char *W;
        hipMalloc(&X, hx.size() * 4); hipMalloc(&Xq, hx.size()); hipMalloc(&Xb, hx.size() * 2); hipMalloc(&xs, M * 4);
        hipMalloc(&W, hw.size()); hipMalloc(&S, hs.size() * 2);
        hipMalloc(&Y0, (size_t)M * N * 4); hipMalloc(&Y1, (size_t)M * N * 4); hipMalloc(&Y2, (size_t)M * N * 4);
        hipMemcpy(X, hx.data(), hx.size() * 4, hipMemcpyHostToDevice); hipMemcpy(W, hw.data(), hw.size(), hipMemcpyHostToDevice);
        hipMemcpy(S, hs.data(), hs.size() * 2, hipMemcpyHostToDevice);
        long n = (long)M * K;
        cast_f32_bf16_smooth<<<(n / 8 + 255) / 256, 256>>>(Xb, X, nullptr, K, n);
        ref_gemm<<<dim3((N + 255) / 256, M), 256>>>(Y0, X, W, S, N, K, M);
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); float t16 = 0, t8 = 0, tq = 0;
        const int grid = ((M + 127) / 128) * ((N + 127) / 128);
        auto w4a16 = [&]() { qgemm_i4w_128x128w8<<<grid, 256>>>(Y1, W, S, Xb, nullptr, N, K, M, 0, nullptr, nullptr, 0); };
        auto quant = [&]() { quant_rows_i8<<<M, 256>>>(Xq, xs, X, K); };
        auto w4a8 = [&]() { gemm_w4a8_g64<<<grid, 256>>>(Y2, W, S, Xq, xs, nullptr, N, K, M, 0); };
        w4a16(); quant(); w4a8(); hipDeviceSynchronize();
        hipEventRecord(e0); for (int r = 0; r < 10; r++) w4a16(); hipEventRecord(e1); hipEventSynchronize(e1); hipEventElapsedTime(&t16, e0, e1);
        hipEventRecord(e0); for (int r = 0; r < 10; r++) quant(); hipEventRecord(e1); hipEventSynchronize(e1); hipEventElapsedTime(&tq, e0, e1);
        hipEventRecord(e0); for (int r = 0; r < 10; r++) w4a8(); hipEventRecord(e1); hipEventSynchronize(e1); hipEventElapsedTime(&t8, e0, e1);
        std::vector<float> a((size_t)M * N), b(a.size()), c(a.size());
        hipMemcpy(a.data(), Y0, a.size() * 4, hipMemcpyDeviceToHost); hipMemcpy(b.data(), Y1, b.size() * 4, hipMemcpyDeviceToHost);
        hipMemcpy(c.data(), Y2, c.size() * 4, hipMemcpyDeviceToHost);
        double d16 = 0, d8 = 0, nr = 0;
        for (size_t i = 0; i < a.size(); i++) { d16 += (b[i] - a[i]) * (double)(b[i] - a[i]); d8 += (c[i] - a[i]) * (double)(c[i] - a[i]); nr += (double)a[i] * a[i]; }
        double flop = 2.0 * M * N * K;
        printf("M=%d N=%5d K=%5d  W4A16 %6.2f ms (%5.1f TF) rel=%.5f | W4A8 %6.2f ms + quant %.2f ms (%5.1f TOPS) rel=%.5f | speedup %.2fx\n",
               M, N, K, t16 / 10, flop / (t16 / 10 * 1e-3) / 1e12, sqrt(d16 / nr), t8 / 10, tq / 10, flop / (t8 / 10 * 1e-3) / 1e12, sqrt(d8 / nr),
               t16 / (t8 + tq));
        hipFree(X); hipFree(Xq); hipFree(Xb); hipFree(xs); hipFree(W); hipFree(S); hipFree(Y0); hipFree(Y1); hipFree(Y2);
    }
    return 0;
}
