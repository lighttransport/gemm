/* Warp MMA/scale mapping and staging adapted from this repo's FP4 experiments.
 * PTX stays E2M1; patch_formats.py changes only designated OMMA format bits. */
#include <cuda_bf16.h>
#include <stdint.h>
#include "int4_numeric.h"

__device__ __forceinline__ void mma(float *d, const unsigned *a, const unsigned *b,
                                   unsigned sa, unsigned sb) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}

extern "C" __global__ void quantize_f32(const float *src, uint8_t *codes, uint8_t *scales,
                                       int rows, int k, int kp, int format,
                                       unsigned long long *stats) {
    size_t group = (size_t)blockIdx.x * 8 + (threadIdx.x >> 4);
    int lane = threadIdx.x & 31, j = lane & 15;
    size_t row = group / (kp / 16);
    int col = (int)(group % (kp / 16)) * 16 + j;
    float x = row < (size_t)rows && col < k ? src[row * k + col] : 0.0f;
    float amax = fabsf(x);
    #pragma unroll
    for (int offset = 8; offset; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    float limit = format ? 7.0f : 6.0f;
    unsigned sc = 0x38;
    if (j == 0 && amax != 0) sc = int4_scale_encode(int4_div(amax, limit));
    sc = __shfl_sync(0xffffffff, sc, lane & 16);
    float scale = int4_scale_decode(sc);
    unsigned code = int4_code_encode(x, scale, format);
    unsigned neighbor = __shfl_xor_sync(0xffffffff, code, 1);
    if (row < (size_t)rows) {
        if (j == 0) {
            scales[group] = (uint8_t)sc;
            float desired = int4_div(amax, limit);
            if (stats && amax && (desired < 1.0f / 512.0f || desired > 448)) atomicAdd(stats + 1, 1ULL);
        }
        if (!(j & 1)) codes[row * (kp / 2) + col / 2] = (uint8_t)(code | neighbor << 4);
        if (stats && int4_div(fabsf(x), scale) > limit) atomicAdd(stats, 1ULL);
    }
}

extern "C" __global__ void dequant_bf16(const uint8_t *codes, const uint8_t *scales,
                                       __nv_bfloat16 *dst, int rows, int kp, int format) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)rows * kp) {
        unsigned q = (codes[i / 2] >> (4 * (i & 1))) & 15;
        dst[i] = __float2bfloat16_rn(int4_code_decode(q, format) * int4_scale_decode(scales[i / 16]));
    }
}

extern "C" __global__ void unpack_int8(const uint8_t *codes, int8_t *dst, int rows, int kp) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)rows * kp) dst[i] = (int8_t)int4_code_decode((codes[i / 2] >> (4 * (i & 1))) & 15, 1);
}

#define GEMM_ARGS const unsigned *__restrict__ A, const unsigned *__restrict__ B, \
    const unsigned *__restrict__ SA, const unsigned *__restrict__ SB, \
    float *__restrict__ C, int M, int N, int K

extern "C" __global__ void gemm_ref(GEMM_ARGS) {
    int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    int mr = blockIdx.y * 16 + g, nr = blockIdx.x * 8 + g;
    int ku = K / 8, kg = K / 64;
    float d[4] = {};
    for (int kc = 0; kc < K / 64; kc++) {
        unsigned a[4] = {}, b[2] = {};
        if (mr < M) { a[0] = A[(size_t)mr * ku + kc * 8 + t]; a[2] = A[(size_t)mr * ku + kc * 8 + t + 4]; }
        if (mr + 8 < M) { a[1] = A[(size_t)(mr + 8) * ku + kc * 8 + t]; a[3] = A[(size_t)(mr + 8) * ku + kc * 8 + t + 4]; }
        if (nr < N) { b[0] = B[(size_t)nr * ku + kc * 8 + t]; b[1] = B[(size_t)nr * ku + kc * 8 + t + 4]; }
        unsigned sa = 0x38383838, sb = 0x38383838;
        int sr = t == 0 ? mr : t == 1 ? mr + 8 : M;
        if (sr < M) sa = SA[(size_t)sr * kg + kc];
        if (t == 0 && nr < N) sb = SB[(size_t)nr * kg + kc];
        mma(d, a, b, sa, sb);
    }
    int nc = blockIdx.x * 8 + t * 2;
    if (mr < M) {
        if (nc < N) C[(size_t)mr * N + nc] = d[0];
        if (nc + 1 < N) C[(size_t)mr * N + nc + 1] = d[1];
    }
    if (mr + 8 < M) {
        if (nc < N) C[(size_t)(mr + 8) * N + nc] = d[2];
        if (nc + 1 < N) C[(size_t)(mr + 8) * N + nc + 1] = d[3];
    }
}

template<int BYTES> __device__ __forceinline__ void copy_async(void *dst, const void *src, bool valid) {
    unsigned address = (unsigned)__cvta_generic_to_shared(dst);
    if constexpr (BYTES == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address), "l"(src), "r"(valid ? 16 : 0) : "memory");
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(address), "l"(src), "r"(valid ? 4 : 0) : "memory");
}

template<int BN, int BK, bool ASYNC>
__device__ __forceinline__ void stage(unsigned a[64][BK / 8], unsigned b[BN][BK / 8],
    unsigned sa[64][BK / 64], unsigned sb[BN][BK / 64],
    const unsigned *A, const unsigned *B, const unsigned *SA, const unsigned *SB,
    int M, int N, int K, int m0, int n0, int kt) {
    constexpr int THREADS = BN == 64 ? 128 : 256;
    int tid = threadIdx.x, ku = K / 8, kg = K / 64;
    #pragma unroll
    for (int i = tid; i < (64 + BN) * (BK / 32); i += THREADS) {
        int r = i / (BK / 32), q = (i % (BK / 32)) * 4;
        bool is_a = r < 64;
        int row = is_a ? m0 + r : n0 + r - 64;
        bool valid = row < (is_a ? M : N) && kt * (BK / 8) + q < ku;
        const unsigned *base = is_a ? A : B;
        const unsigned *src = valid ? base + (size_t)row * ku + kt * (BK / 8) + q : base;
        unsigned *dst = is_a ? &a[r][q] : &b[r - 64][q];
        if constexpr (ASYNC) copy_async<16>(dst, src, valid);
        else *reinterpret_cast<uint4 *>(dst) = valid ? *reinterpret_cast<const uint4 *>(src) : make_uint4(0, 0, 0, 0);
    }
    #pragma unroll
    for (int i = tid; i < (64 + BN) * (BK / 64); i += THREADS) {
        int r = i / (BK / 64), q = i % (BK / 64);
        bool is_a = r < 64;
        int row = is_a ? m0 + r : n0 + r - 64;
        bool valid = row < (is_a ? M : N) && kt * (BK / 64) + q < kg;
        const unsigned *base = is_a ? SA : SB;
        const unsigned *src = valid ? base + (size_t)row * kg + kt * (BK / 64) + q : base;
        unsigned *dst = is_a ? &sa[r][q] : &sb[r - 64][q];
        if constexpr (ASYNC) copy_async<4>(dst, src, valid);
        else *dst = valid ? *src : 0;
    }
    if constexpr (ASYNC) asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int BN, int BK, bool ASYNC> __device__ __forceinline__ void tiled(GEMM_ARGS) {
    constexpr int STAGES = ASYNC ? 2 : 1;
    __shared__ __align__(16) unsigned a[STAGES][64][BK / 8], b[STAGES][BN][BK / 8];
    __shared__ unsigned sa[STAGES][64][BK / 64], sb[STAGES][BN][BK / 64];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int wm = warp / (BN / 32), wn = warp % (BN / 32);
    int m0 = blockIdx.y * 64, n0 = blockIdx.x * BN, nkt = (K - 1) / BK + 1;
    float acc[2][4][4] = {};
    if constexpr (ASYNC) stage<BN, BK, true>(a[0], b[0], sa[0], sb[0], A, B, SA, SB, M, N, K, m0, n0, 0);
    for (int kt = 0; kt < nkt; kt++) {
        int cur = ASYNC ? kt & 1 : 0;
        if constexpr (ASYNC) {
            if (kt + 1 < nkt) {
                stage<BN, BK, true>(a[cur ^ 1], b[cur ^ 1], sa[cur ^ 1], sb[cur ^ 1], A, B, SA, SB, M, N, K, m0, n0, kt + 1);
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            } else asm volatile("cp.async.wait_group 0;" ::: "memory");
        } else stage<BN, BK, false>(a[0], b[0], sa[0], sb[0], A, B, SA, SB, M, N, K, m0, n0, kt);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK / 64; kk++) {
            unsigned af[2][4], bf[4][2], sf_a[2], sf_b[4];
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int r = wm * 32 + i * 16 + g, q = kk * 8 + t;
                af[i][0] = a[cur][r][q]; af[i][1] = a[cur][r + 8][q];
                af[i][2] = a[cur][r][q + 4]; af[i][3] = a[cur][r + 8][q + 4];
                sf_a[i] = t < 2 ? sa[cur][r + t * 8][kk] : 0x38383838;
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int r = wn * 32 + j * 8 + g, q = kk * 8 + t;
                bf[j][0] = b[cur][r][q]; bf[j][1] = b[cur][r][q + 4];
                sf_b[j] = t == 0 ? sb[cur][r][kk] : 0x38383838;
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) mma(acc[i][j], af[i], bf[j], sf_a[i], sf_b[j]);
            }
        }
        __syncthreads(); /* All readers finish before a staging buffer is reused. */
    }
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int r = m0 + wm * 32 + i * 16 + g, c = n0 + wn * 32 + j * 8 + t * 2;
            if (r < M) {
                if (c < N) C[(size_t)r * N + c] = acc[i][j][0];
                if (c + 1 < N) C[(size_t)r * N + c + 1] = acc[i][j][1];
            }
            if (r + 8 < M) {
                if (c < N) C[(size_t)(r + 8) * N + c] = acc[i][j][2];
                if (c + 1 < N) C[(size_t)(r + 8) * N + c + 1] = acc[i][j][3];
            }
        }
    }
}

#define TILE(NAME, BN, BK, ASYNC, THREADS) \
    extern "C" __global__ __launch_bounds__(THREADS) void NAME(GEMM_ARGS) { \
        tiled<BN, BK, ASYNC>(A, B, SA, SB, C, M, N, K); }
TILE(gemm_s64_64, 64, 64, false, 128)
TILE(gemm_a64_64, 64, 64, true, 128)
TILE(gemm_s64_128, 64, 128, false, 128)
TILE(gemm_a64_128, 64, 128, true, 128)
TILE(gemm_s128_64, 128, 64, false, 256)
TILE(gemm_a128_64, 128, 64, true, 256)
TILE(gemm_s128_128, 128, 128, false, 256)
TILE(gemm_a128_128, 128, 128, true, 256)

extern "C" __global__ void probe_fp4(float *out, int repeats, unsigned operand) {
    float d[8][4] = {};
    for (int r = 0; r < repeats; r++) {
        const unsigned a[4] = {operand, operand, operand, operand}, b[2] = {operand, operand};
        #pragma unroll
        for (int i = 0; i < 8; i++) mma(d[i], a, b, 0x38383838, 0x38383838);
        operand ^= 0x88888888; /* Both signs change; the product stays positive. */
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) for (int j = 0; j < 4; j++)
        out[((size_t)blockIdx.x * 32 + threadIdx.x) * 32 + i * 4 + j] = d[i][j];
}

extern "C" __global__ void probe_bf16(float *out, int repeats, unsigned v) {
    float d[8][4] = {};
    for (int r = 0; r < repeats; r++) {
        #pragma unroll
        for (int i = 0; i < 8; i++)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};"
                : "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3]) : "r"(v));
        v ^= 0x80008000;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) for (int j = 0; j < 4; j++)
        out[((size_t)blockIdx.x * 32 + threadIdx.x) * 32 + i * 4 + j] = d[i][j];
}

#define INTEGER_PROBE(NAME, KDEPTH, TYPE, MASK) \
extern "C" __global__ void NAME(int *out, int repeats, unsigned v) { \
    int d[8][4] = {}; \
    for (int r = 0; r < repeats; r++) { \
        _Pragma("unroll") \
        for (int i = 0; i < 8; i++) \
            asm volatile("mma.sync.aligned.m16n8k" #KDEPTH ".row.col.s32." #TYPE "." #TYPE ".s32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" \
                : "+r"(d[i][0]), "+r"(d[i][1]), "+r"(d[i][2]), "+r"(d[i][3]) : "r"(v)); \
        v ^= MASK; \
    } \
    _Pragma("unroll") \
    for (int i = 0; i < 8; i++) for (int j = 0; j < 4; j++) \
        out[((size_t)blockIdx.x * 32 + threadIdx.x) * 32 + i * 4 + j] = d[i][j]; \
}
INTEGER_PROBE(probe_int8, 32, s8, 0xfefefefe)
INTEGER_PROBE(probe_s4, 64, s4, 0xeeeeeeee)
