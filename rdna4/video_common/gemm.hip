// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
typedef float vf8 __attribute__((ext_vector_type(8)));
typedef unsigned short vbf8 __attribute__((ext_vector_type(8)));
typedef _Float16 vh8 __attribute__((ext_vector_type(8)));
typedef _Float16 vh16 __attribute__((ext_vector_type(16)));
typedef int vi8 __attribute__((ext_vector_type(8)));
typedef unsigned vu2 __attribute__((ext_vector_type(2)));
typedef unsigned char vb8 __attribute__((ext_vector_type(8)));
typedef unsigned char vb16 __attribute__((ext_vector_type(16)));
__device__ float video_bfloat(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7f800000u) != 0x7f800000u)
        u += 0x7fff + ((u >> 16) & 1);
    return __uint_as_float(u & 0xffff0000u);
}
// Transposed 8-byte LDS slots and a three-stage operand pipeline. Every wave
// retains a 64x64 output tile; vector global loads are guarded for M/N tails.
// K must be a multiple of 32 (the ConvRot checkpoints use multiples of 256).
__device__ __forceinline__ void video_signal() { asm volatile("s_barrier_signal -1" ::: "memory"); }
__device__ __forceinline__ void video_wait() { asm volatile("s_barrier_wait 0xffff" ::: "memory"); }
__device__ __forceinline__ vu2 video_words(vb8 bytes) {
    union {
        vb8 bytes;
        vu2 words;
    } u;
    u.bytes = bytes;
    return u.words;
}
__device__ __forceinline__ void video_store_i8(float *y, int *raw, vi8 acc, const float *xs,
                                               const float *ws, int row, int col, int m, int n,
                                               int rounded) {
    if (col >= n)
        return;
#pragma unroll
    for (int j = 0; j < 8; j++)
        if (row + j < m) {
            long at = (long)(row + j) * n + col;
            if (raw)
                raw[at] = acc[j];
            if (y) {
                float v = float(acc[j]) * (xs[row + j] * ws[col]);
                y[at] = rounded ? video_bfloat(v) : v;
            }
        }
}
extern "C" __global__ __launch_bounds__(128, 1) void video_gemm_i8_pipeline(
    float *y, const signed char *x, const signed char *w, const float *xs, const float *ws, int m,
    int n, int k, int rounded, int *raw) {
    int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, half = lane >> 4, col = lane & 15;
    int row0 = blockIdx.y * 128, col0 = blockIdx.x * 128, wr = (wave & 1) * 64,
        wc = (wave >> 1) * 64;
    __shared__ vb8 ax[3 * 4 * 128], bw[3 * 4 * 128];
#define VP_LOAD(a0, a1, b0, b1, base)                                                              \
    do {                                                                                           \
        a0 = (row0 + tid < m)                                                                      \
                 ? *reinterpret_cast<const vb16 *>(x + (long)(row0 + tid) * k + (base))            \
                 : vb16{};                                                                         \
        a1 = (row0 + tid < m)                                                                      \
                 ? *reinterpret_cast<const vb16 *>(x + (long)(row0 + tid) * k + (base) + 16)       \
                 : vb16{};                                                                         \
        b0 = (col0 + tid < n)                                                                      \
                 ? *reinterpret_cast<const vb16 *>(w + (long)(col0 + tid) * k + (base))            \
                 : vb16{};                                                                         \
        b1 = (col0 + tid < n)                                                                      \
                 ? *reinterpret_cast<const vb16 *>(w + (long)(col0 + tid) * k + (base) + 16)       \
                 : vb16{};                                                                         \
    } while (0)
#define VP_LO(v) __builtin_shufflevector(v, v, 0, 1, 2, 3, 4, 5, 6, 7)
#define VP_HI(v) __builtin_shufflevector(v, v, 8, 9, 10, 11, 12, 13, 14, 15)
#define VP_STORE(buffer, a0, a1, b0, b1)                                                           \
    do {                                                                                           \
        int at = (buffer) * 512 + tid;                                                             \
        ax[at] = VP_LO(a0);                                                                        \
        ax[at + 128] = VP_HI(a0);                                                                  \
        ax[at + 256] = VP_LO(a1);                                                                  \
        ax[at + 384] = VP_HI(a1);                                                                  \
        bw[at] = VP_LO(b0);                                                                        \
        bw[at + 128] = VP_HI(b0);                                                                  \
        bw[at + 256] = VP_LO(b1);                                                                  \
        bw[at + 384] = VP_HI(b1);                                                                  \
    } while (0)
    vb16 a0, a1, b0, b1;
    VP_LOAD(a0, a1, b0, b1, 0);
    VP_STORE(0, a0, a1, b0, b1);
    if (k > 32) {
        VP_LOAD(a0, a1, b0, b1, 32);
        VP_STORE(1, a0, a1, b0, b1);
    }
    video_signal();
    video_wait();
    vi8 z = {0, 0, 0, 0, 0, 0, 0, 0};
    vi8 c00 = z, c01 = z, c02 = z, c03 = z, c10 = z, c11 = z, c12 = z, c13 = z, c20 = z, c21 = z,
        c22 = z, c23 = z, c30 = z, c31 = z, c32 = z, c33 = z;
    int read = 0, write = 2;
    for (int base = 0; base < k; base += 32) {
        vb16 na0, na1, nb0, nb1;
        bool far = base + 64 < k;
        if (far)
            VP_LOAD(na0, na1, nb0, nb1, base + 64);
#pragma unroll
        for (int step = 0; step < 32; step += 16) {
            int slot = read * 512 + (step / 8 + half) * 128;
            vu2 aa0 = video_words(ax[slot + wr + col]), aa1 = video_words(ax[slot + wr + 16 + col]),
                aa2 = video_words(ax[slot + wr + 32 + col]),
                aa3 = video_words(ax[slot + wr + 48 + col]);
            vu2 bb0 = video_words(bw[slot + wc + col]), bb1 = video_words(bw[slot + wc + 16 + col]),
                bb2 = video_words(bw[slot + wc + 32 + col]),
                bb3 = video_words(bw[slot + wc + 48 + col]);
#define VP_MMA(a, b, c)                                                                            \
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false)
            VP_MMA(aa0, bb0, c00);
            VP_MMA(aa0, bb1, c01);
            VP_MMA(aa0, bb2, c02);
            VP_MMA(aa0, bb3, c03);
            VP_MMA(aa1, bb0, c10);
            VP_MMA(aa1, bb1, c11);
            VP_MMA(aa1, bb2, c12);
            VP_MMA(aa1, bb3, c13);
            VP_MMA(aa2, bb0, c20);
            VP_MMA(aa2, bb1, c21);
            VP_MMA(aa2, bb2, c22);
            VP_MMA(aa2, bb3, c23);
            VP_MMA(aa3, bb0, c30);
            VP_MMA(aa3, bb1, c31);
            VP_MMA(aa3, bb2, c32);
            VP_MMA(aa3, bb3, c33);
#undef VP_MMA
        }
        if (far) {
            VP_STORE(write, na0, na1, nb0, nb1);
            write = (write + 1) % 3;
        }
        if (base + 32 < k) {
            video_signal();
            read = (read + 1) % 3;
            video_wait();
        }
    }
#undef VP_LOAD
#undef VP_STORE
#undef VP_LO
#undef VP_HI
    int r = row0 + wr + half * 8, c = col0 + wc + col;
#define VP_OUTPUT(acc, rr, cc) video_store_i8(y, raw, acc, xs, ws, r + rr, c + cc, m, n, rounded)
    VP_OUTPUT(c00, 0, 0);
    VP_OUTPUT(c01, 0, 16);
    VP_OUTPUT(c02, 0, 32);
    VP_OUTPUT(c03, 0, 48);
    VP_OUTPUT(c10, 16, 0);
    VP_OUTPUT(c11, 16, 16);
    VP_OUTPUT(c12, 16, 32);
    VP_OUTPUT(c13, 16, 48);
    VP_OUTPUT(c20, 32, 0);
    VP_OUTPUT(c21, 32, 16);
    VP_OUTPUT(c22, 32, 32);
    VP_OUTPUT(c23, 32, 48);
    VP_OUTPUT(c30, 48, 0);
    VP_OUTPUT(c31, 48, 16);
    VP_OUTPUT(c32, 48, 32);
    VP_OUTPUT(c33, 48, 48);
#undef VP_OUTPUT
}
extern "C" __global__ void video_gemm_f16(float *y, const _Float16 *x, const _Float16 *w, int m,
                                          int n, int k) {
    int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, half = lane >> 4, col = lane & 15;
    int r0 = blockIdx.y * 32 + (wave & 1) * 16, c0 = blockIdx.x * 32 + (wave >> 1) * 16;
    x += (long)blockIdx.z * m * k;
    w += (long)blockIdx.z * n * k;
    y += (long)blockIdx.z * m * n;
    vf8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int a = 0; a < k; a += 16) {
        vh8 av, bv;
        for (int j = 0; j < 8; j++) {
            int p = a + half * 8 + j;
            av[j] = (r0 + col < m && p < k) ? x[(long)(r0 + col) * k + p] : (_Float16)0;
            bv[j] = (c0 + col < n && p < k) ? w[(long)(c0 + col) * k + p] : (_Float16)0;
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, acc);
    }
    for (int j = 0; j < 8; j++)
        if (r0 + half * 8 + j < m && c0 + col < n)
            y[(long)(r0 + half * 8 + j) * n + c0 + col] = acc[j];
}

// Two 16-K LDS stages reuse operands across a 128x128 output tile. The
// accumulator visits K in the same order as the small FP16 WMMA kernel.
// Callers pad K to 16; vector loads guard complete M/N rows.
extern "C" __global__ __launch_bounds__(128, 1) void video_gemm_f16_tiled(float *y,
                                                                          const _Float16 *x,
                                                                          const _Float16 *w, int m,
                                                                          int n, int k) {
    int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, half = lane >> 4, col = lane & 15;
    int row0 = blockIdx.y * 128, col0 = blockIdx.x * 128;
    int wr = (wave & 1) * 64, wc = (wave >> 1) * 64;
    __shared__ vh8 ax[2 * 2 * 128], bw[2 * 2 * 128];
#define VF_LOAD(a, b, base)                                                                        \
    do {                                                                                           \
        a = row0 + tid < m ? *reinterpret_cast<const vh16 *>(x + (long)(row0 + tid) * k + base)    \
                           : vh16{};                                                               \
        b = col0 + tid < n ? *reinterpret_cast<const vh16 *>(w + (long)(col0 + tid) * k + base)    \
                           : vh16{};                                                               \
    } while (0)
#define VF_STORE(buffer, a, b)                                                                     \
    do {                                                                                           \
        int at = (buffer) * 256 + tid;                                                             \
        ax[at] = __builtin_shufflevector(a, a, 0, 1, 2, 3, 4, 5, 6, 7);                            \
        ax[at + 128] = __builtin_shufflevector(a, a, 8, 9, 10, 11, 12, 13, 14, 15);                \
        bw[at] = __builtin_shufflevector(b, b, 0, 1, 2, 3, 4, 5, 6, 7);                            \
        bw[at + 128] = __builtin_shufflevector(b, b, 8, 9, 10, 11, 12, 13, 14, 15);                \
    } while (0)
    vh16 a, b;
    VF_LOAD(a, b, 0);
    VF_STORE(0, a, b);
    __syncthreads();
    vf8 zero = {};
    vf8 c00 = zero, c01 = zero, c02 = zero, c03 = zero;
    vf8 c10 = zero, c11 = zero, c12 = zero, c13 = zero;
    vf8 c20 = zero, c21 = zero, c22 = zero, c23 = zero;
    vf8 c30 = zero, c31 = zero, c32 = zero, c33 = zero;
    int read = 0;
    for (int base = 0; base < k; base += 16) {
        vh16 next_a, next_b;
        bool next = base + 16 < k;
        if (next)
            VF_LOAD(next_a, next_b, base + 16);
        int slot = read * 256 + half * 128;
        vh8 aa0 = ax[slot + wr + col], aa1 = ax[slot + wr + 16 + col];
        vh8 aa2 = ax[slot + wr + 32 + col], aa3 = ax[slot + wr + 48 + col];
        vh8 bb0 = bw[slot + wc + col], bb1 = bw[slot + wc + 16 + col];
        vh8 bb2 = bw[slot + wc + 32 + col], bb3 = bw[slot + wc + 48 + col];
#define VF_MMA(a, b, c) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c)
        VF_MMA(aa0, bb0, c00);
        VF_MMA(aa0, bb1, c01);
        VF_MMA(aa0, bb2, c02);
        VF_MMA(aa0, bb3, c03);
        VF_MMA(aa1, bb0, c10);
        VF_MMA(aa1, bb1, c11);
        VF_MMA(aa1, bb2, c12);
        VF_MMA(aa1, bb3, c13);
        VF_MMA(aa2, bb0, c20);
        VF_MMA(aa2, bb1, c21);
        VF_MMA(aa2, bb2, c22);
        VF_MMA(aa2, bb3, c23);
        VF_MMA(aa3, bb0, c30);
        VF_MMA(aa3, bb1, c31);
        VF_MMA(aa3, bb2, c32);
        VF_MMA(aa3, bb3, c33);
#undef VF_MMA
        if (next) {
            VF_STORE(read ^ 1, next_a, next_b);
            __syncthreads();
            read ^= 1;
        }
    }
#undef VF_LOAD
#undef VF_STORE
    int r = row0 + wr + half * 8, c = col0 + wc + col;
#define VF_OUTPUT(acc, rr, cc)                                                                     \
    do {                                                                                           \
        for (int j = 0; j < 8; j++)                                                                \
            if (r + rr + j < m && c + cc < n)                                                      \
                y[(long)(r + rr + j) * n + c + cc] = acc[j];                                       \
    } while (0)
    VF_OUTPUT(c00, 0, 0);
    VF_OUTPUT(c01, 0, 16);
    VF_OUTPUT(c02, 0, 32);
    VF_OUTPUT(c03, 0, 48);
    VF_OUTPUT(c10, 16, 0);
    VF_OUTPUT(c11, 16, 16);
    VF_OUTPUT(c12, 16, 32);
    VF_OUTPUT(c13, 16, 48);
    VF_OUTPUT(c20, 32, 0);
    VF_OUTPUT(c21, 32, 16);
    VF_OUTPUT(c22, 32, 32);
    VF_OUTPUT(c23, 32, 48);
    VF_OUTPUT(c30, 48, 0);
    VF_OUTPUT(c31, 48, 16);
    VF_OUTPUT(c32, 48, 32);
    VF_OUTPUT(c33, 48, 48);
#undef VF_OUTPUT
}
__device__ unsigned video_pack(const signed char *p, int row, int k, int base, int count) {
    unsigned u = 0;
    for (int j = 0; j < 4; j++)
        if (base + j < count)
            u |= (unsigned)(unsigned char)p[(long)row * k + base + j] << (j * 8);
    return u;
}
extern "C" __global__ void video_gemm_bf16(float *y, const float *x, const unsigned short *w, int m,
                                           int n, int k) {
    int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, half = lane >> 4, col = lane & 15;
    int r0 = blockIdx.y * 32 + (wave & 1) * 16, c0 = blockIdx.x * 32 + (wave >> 1) * 16;
    x += (long)blockIdx.z * m * k;
    w += (long)blockIdx.z * n * k;
    y += (long)blockIdx.z * m * n;
    vf8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int a = 0; a < k; a += 16) {
        vbf8 av, bv;
        for (int j = 0; j < 8; j++) {
            int p = a + half * 8 + j;
            float input = (r0 + col < m && p < k) ? x[(long)(r0 + col) * k + p] : 0.f;
            av[j] = __float_as_uint(video_bfloat(input)) >> 16;
            bv[j] = (c0 + col < n && p < k) ? w[(long)(c0 + col) * k + p] : 0;
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, bv, acc);
    }
    for (int j = 0; j < 8; j++)
        if (r0 + half * 8 + j < m && c0 + col < n)
            y[(long)(r0 + half * 8 + j) * n + c0 + col] = acc[j];
}
extern "C" __global__ void video_gemm_i8(float *y, const signed char *x, const signed char *w,
                                         const float *xs, const float *ws, int m, int n, int k,
                                         int rounded, int *raw) {
    int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, half = lane >> 4, col = lane & 15;
    int r0 = blockIdx.y * 32 + (wave & 1) * 16, c0 = blockIdx.x * 32 + (wave >> 1) * 16;
    vi8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int a = 0; a < k; a += 16) {
        vu2 av, bv;
        for (int j = 0; j < 2; j++) {
            int p = a + half * 8 + j * 4;
            av[j] = (r0 + col < m) ? video_pack(x, r0 + col, k, p, k) : 0;
            bv[j] = (c0 + col < n) ? video_pack(w, c0 + col, k, p, k) : 0;
        }
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av, true, bv, acc, false);
    }
    for (int j = 0; j < 8; j++)
        if (r0 + half * 8 + j < m && c0 + col < n) {
            int row = r0 + half * 8 + j;
            long at = (long)row * n + c0 + col;
            if (raw)
                raw[at] = acc[j];
            if (y) {
                float v = (float)acc[j] * (xs[row] * ws[c0 + col]);
                y[at] = rounded ? video_bfloat(v) : v;
            }
        }
}
// Eight waves compute a 128x128 output tile. Cooperative contiguous transfers
// reuse each input tile across output fragments; tails retain exact zero padding.
extern "C" __global__ void video_gemm_i8_tiled(float *y, const signed char *x, const signed char *w,
                                               const float *xs, const float *ws, int m, int n,
                                               int k, int rounded, int *raw) {
    int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, half = lane >> 4, col = lane & 15;
    int row0 = blockIdx.y * 128, col0 = blockIdx.x * 128, wr = (wave & 1) * 64,
        wc = (wave >> 1) * 32;
    // A 17-word pitch avoids the bank conflicts of a 64-byte row stride.
    __shared__ unsigned ax[128 * 17], bw[128 * 17];
    vi8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    vi8 c00 = zero, c01 = zero, c10 = zero, c11 = zero, c20 = zero, c21 = zero, c30 = zero,
        c31 = zero;
    for (int base = 0; base < k; base += 64) {
#pragma unroll
        for (int it = 0; it < 8; it++) {
            int i = tid + it * 256, row = i / 16, p = base + (i % 16) * 4;
            ax[row * 17 + i % 16] =
                (row0 + row < m)
                    ? ((k % 4 == 0 && p + 4 <= k)
                           ? reinterpret_cast<const unsigned *>(x + (long)(row0 + row) * k + p)[0]
                           : video_pack(x, row0 + row, k, p, k))
                    : 0;
            bw[row * 17 + i % 16] =
                (col0 + row < n)
                    ? ((k % 4 == 0 && p + 4 <= k)
                           ? reinterpret_cast<const unsigned *>(w + (long)(col0 + row) * k + p)[0]
                           : video_pack(w, col0 + row, k, p, k))
                    : 0;
        }
        __syncthreads();
#pragma unroll
        for (int a = 0; a < 64; a += 16) {
            vu2 a0, a1, a2, a3, b0, b1;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                int p = a / 4 + half * 2 + j;
                a0[j] = ax[(wr + col) * 17 + p];
                a1[j] = ax[(wr + 16 + col) * 17 + p];
                a2[j] = ax[(wr + 32 + col) * 17 + p];
                a3[j] = ax[(wr + 48 + col) * 17 + p];
                b0[j] = bw[(wc + col) * 17 + p];
                b1[j] = bw[(wc + 16 + col) * 17 + p];
            }
            c00 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, true, b0, c00, false);
            c01 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, true, b1, c01, false);
            c10 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a1, true, b0, c10, false);
            c11 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a1, true, b1, c11, false);
            c20 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a2, true, b0, c20, false);
            c21 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a2, true, b1, c21, false);
            c30 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a3, true, b0, c30, false);
            c31 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a3, true, b1, c31, false);
        }
        __syncthreads();
    }
    vi8 acc[8] = {c00, c01, c10, c11, c20, c21, c30, c31};
#pragma unroll
    for (int tile = 0; tile < 8; tile++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            int row = row0 + wr + (tile / 2) * 16 + half * 8 + j,
                outcol = col0 + wc + (tile % 2) * 16 + col;
            if (row < m && outcol < n) {
                long at = (long)row * n + outcol;
                if (raw)
                    raw[at] = acc[tile][j];
                if (y) {
                    float v = float(acc[tile][j]) * (xs[row] * ws[outcol]);
                    y[at] = rounded ? video_bfloat(v) : v;
                }
            }
        }
    }
}
