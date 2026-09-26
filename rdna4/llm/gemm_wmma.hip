/* Self-owned gfx12 (RDNA4) WMMA GEMM: Y[M,N] f32 = X[M,K] x W[N,K]^T.
 *
 * X is a row-major 16-bit (BF16 or F16) matrix, Y is row-major F32.  W is
 * either a row-major 16-bit matrix (the nn.Linear layout used by
 * mm_blaslt_run_bf16, so this is a drop-in replacement for the hipBLASLt
 * prefill projections) or a row-major GGML-quantized matrix that is decoded
 * to BF16 on the fly while staging each tile ("fused dequant").
 *
 * Generalized from rdna4/vlm gemm_mm0_bf16_pipe4w:
 *   - BMxBN CTA tile, K step 32, WAVES_M x WAVES_N wave32 grid;
 *   - the next K tile is prefetched into registers while WMMA consumes the
 *     current one from LDS (double-buffered, one barrier per K step); quantized
 *     weights prefetch only their raw bytes and are decoded after the WMMAs;
 *   - LDS holds 8-element vectors in kslot-major order ([4][rows]), so the 16
 *     lanes of a WMMA operand read consecutive 16-byte vectors;
 *   - grid.x walks M tiles fastest: the few M tiles that share a weight tile
 *     run concurrently, so each W tile is streamed from DRAM ~once;
 *   - rows past M/N are clamped on load and masked on store, so interior and
 *     edge tiles run the same code.
 * Requirements: K % 32 == 0 (K % 256 == 0 for quantized W), 16-byte aligned
 * X/W base pointers.  Deterministic: each output is reduced by exactly one
 * wave in fixed K order, or by a fixed-order reduction of split-K partials.
 *
 * The quantized decoders reproduce the dequant_*_to_bf16 kernels of
 * hip_llm_runner.c expression for expression, so a fused GEMM is intended to
 * be bit-identical to dequant -> BF16 GEMM (LLM_GEMM_FUSED_CHECK=1 in the
 * runner compares the two on the real weights). */

typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef unsigned short half_raw;
typedef unsigned short bf16_raw;

/* ---- scalar helpers: identical to hip_llm_runner.c ---- */
__device__ __forceinline__ float half_to_float(half_raw h) {
    __half hv;
    *((unsigned short*)&hv) = h;
    return __half2float(hv);
}
__device__ __forceinline__ bf16_raw f32_to_bf16(float f) {
    unsigned int u = __float_as_uint(f);
    /* Round-to-nearest-even */
    unsigned int rounded = u + ((u >> 16) & 1u) + 0x7FFFu;
    return (bf16_raw)(rounded >> 16);
}

/* ---- llama.cpp IQ tables (ggml authors, MIT; see reference_sampler.cpp) ---- */
__device__ static const unsigned long long gw_iq2xxs_grid[256] = {
#include "qwen35_iq2xxs_grid.inc"
};
__device__ static const unsigned long long gw_iq2xs_grid[512] = {
#include "qwen35_iq2xs_grid.inc"
};
__device__ static const unsigned long long gw_iq2s_grid[1024] = {
#include "../../common/iq2s_grid.inc"
};
__device__ static const unsigned int gw_iq3xxs_grid[256] = {
#include "../../common/iq3xxs_grid.inc"
};
__device__ static const unsigned int gw_iq3s_grid[512] = {
#include "../../common/iq3s_grid.inc"
};
__device__ static const unsigned long long gw_iq1s_grid[2048] = {
#include "iq1s_grid.inc"
};
__device__ static const unsigned char gw_ksigns_iq2xs[128] = {
#include "../../common/ksigns_iq2xs.inc"
};
__device__ static const signed char gw_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

__device__ __forceinline__ unsigned int ld16(const unsigned char *p) {
    return *(const unsigned short *)p;
}
__device__ __forceinline__ unsigned int ld32(const unsigned char *p) {
    return *(const unsigned int *)p;
}
__device__ __forceinline__ unsigned int ld32_a2(const unsigned char *p) {
    return ld16(p) | (ld16(p + 2) << 16);
}

/* ---- B-operand loaders -------------------------------------------------
 * row_bytes(K)   : bytes per weight row
 * load(row, k)   : fetch the raw data for the 8 weights at columns [k, k+8)
 *                  of this row (k % 8 == 0) into registers
 * decode(raw, k) : produce the 8 16-bit values
 * Quantized formats: superblock k/256, sub-block c = (k/32)%8, group of 8
 * g = (k/8)%4 inside the sub-block. */
struct Raw { unsigned int w[4]; };

struct LdDense {
    __device__ static size_t row_bytes(int K) { return (size_t)K * 2; }
    __device__ static Raw load(const unsigned char *row, int k) {
        return __builtin_bit_cast(Raw, *(const u16x8 *)(row + (size_t)k * 2));
    }
    __device__ static u16x8 decode(const Raw &r, int) {
        return __builtin_bit_cast(u16x8, r);
    }
};

#define QPOS const int c = (k >> 5) & 7, g = (k >> 3) & 3; (void)c; (void)g
#define QBP(BYTES) const unsigned char *bp = row + (size_t)(k >> 8) * (BYTES)

/* IQ2_XXS, 66 B: d | qs[32] u16.  Sub-block c: aux0 = qs[4c..4c+1] (grid
 * indices), aux1 = qs[4c+2..4c+3] (signs + scale). */
struct LdIQ2XXS {
    __device__ static size_t row_bytes(int K) { return (size_t)(K / 256) * 66; }
    __device__ static Raw load(const unsigned char *row, int k) {
        QPOS; QBP(66);
        const unsigned char *qs = bp + 2 + 8 * c;
        Raw r;
        r.w[0] = ld16(bp);
        r.w[1] = ld16(qs + 2 * (g >> 1));
        r.w[2] = ld32_a2(qs + 4);
        return r;
    }
    __device__ static u16x8 decode(const Raw &r, int k) {
        QPOS;
        float d = half_to_float((half_raw)r.w[0]);
        unsigned int aux1 = r.w[2];
        float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
        unsigned int idx = (g & 1) ? (r.w[1] >> 8) & 255 : r.w[1] & 255;
        unsigned long long grid = gw_iq2xxs_grid[idx];
        unsigned char signs = gw_ksigns_iq2xs[(aux1 >> (7 * g)) & 127];
        u16x8 o;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = db * (float)(unsigned char)(grid >> (8 * j)) * ((signs & (1 << j)) ? -1.0f : 1.0f);
            o[j] = f32_to_bf16(val);
        }
        return o;
    }
};

/* IQ2_XS, 74 B: d | qs[32] u16 (9-bit grid + 7-bit sign) | scales[8]. */
struct LdIQ2XS {
    __device__ static size_t row_bytes(int K) { return (size_t)(K / 256) * 74; }
    __device__ static Raw load(const unsigned char *row, int k) {
        QPOS; QBP(74);
        Raw r;
        r.w[0] = ld16(bp);
        r.w[1] = ld16(bp + 2 + 2 * (4 * c + g));
        r.w[2] = bp[66 + c];
        return r;
    }
    __device__ static u16x8 decode(const Raw &r, int k) {
        QPOS;
        float d = half_to_float((half_raw)r.w[0]);
        unsigned int sc = r.w[2];
        float dl = (g < 2)
            ? d * (0.5f + (float)(sc & 0xf)) * 0.25f
            : d * (0.5f + (float)(sc >>  4)) * 0.25f;
        unsigned int qval = r.w[1];
        unsigned long long grid = gw_iq2xs_grid[qval & 511];
        unsigned char signs = gw_ksigns_iq2xs[qval >> 9];
        u16x8 o;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = dl * (float)(unsigned char)(grid >> (8 * j)) * ((signs & (1 << j)) ? -1.0f : 1.0f);
            o[j] = f32_to_bf16(val);
        }
        return o;
    }
};

/* IQ2_S, 82 B: d | qs[32] | signs[32] | qh[8] | scales[8]. */
struct LdIQ2S {
    __device__ static size_t row_bytes(int K) { return (size_t)(K / 256) * 82; }
    __device__ static Raw load(const unsigned char *row, int k) {
        QPOS; QBP(82);
        Raw r;
        r.w[0] = ld16(bp);
        r.w[1] = bp[2 + 4 * c + g];
        r.w[2] = bp[34 + 4 * c + g];
        r.w[3] = (unsigned int)bp[66 + c] | ((unsigned int)bp[74 + c] << 8);
        return r;
    }
    __device__ static u16x8 decode(const Raw &r, int k) {
        QPOS;
        float d = half_to_float((half_raw)r.w[0]);
        unsigned int sc = r.w[3] >> 8;
        unsigned char qh = (unsigned char)(r.w[3] & 255);
        float dl = (g < 2)
            ? d * (0.5f + (float)(sc & 0xf)) * 0.25f
            : d * (0.5f + (float)(sc >>  4)) * 0.25f;
        int grid_idx = (int)r.w[1] | ((qh << (8 - 2 * g)) & 0x300);
        unsigned long long grid = gw_iq2s_grid[grid_idx];
        unsigned char s = (unsigned char)r.w[2];
        u16x8 o;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = dl * (float)(unsigned char)(grid >> (8 * j)) * ((s & (1 << j)) ? -1.0f : 1.0f);
            o[j] = f32_to_bf16(val);
        }
        return o;
    }
};

/* IQ3_XXS, 98 B: d | qs[64] | scales_and_signs[8] u32. */
struct LdIQ3XXS {
    __device__ static size_t row_bytes(int K) { return (size_t)(K / 256) * 98; }
    __device__ static Raw load(const unsigned char *row, int k) {
        QPOS; QBP(98);
        Raw r;
        r.w[0] = ld16(bp);
        r.w[1] = ld16(bp + 2 + 8 * c + 2 * g);
        r.w[2] = ld32_a2(bp + 66 + 4 * c);
        return r;
    }
    __device__ static u16x8 decode(const Raw &r, int k) {
        QPOS;
        float d = half_to_float((half_raw)r.w[0]);
        unsigned int aux32 = r.w[2];
        float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        unsigned char signs = gw_ksigns_iq2xs[(aux32 >> (7 * g)) & 127];
        unsigned int g1 = gw_iq3xxs_grid[r.w[1] & 255];
        unsigned int g2 = gw_iq3xxs_grid[(r.w[1] >> 8) & 255];
        u16x8 o;
#pragma unroll
        for (int t = 0; t < 8; t++) {
            unsigned int grid = t >= 4 ? g2 : g1;
            int j = t & 3;
            float val = db * (float)(unsigned char)(grid >> (8 * j)) * ((signs & (1 << t)) ? -1.0f : 1.0f);
            o[t] = f32_to_bf16(val);
        }
        return o;
    }
};

/* IQ3_S, 110 B: d | qs[64] | qh[8] | signs[32] | scales[4]. */
struct LdIQ3S {
    __device__ static size_t row_bytes(int K) { return (size_t)(K / 256) * 110; }
    __device__ static Raw load(const unsigned char *row, int k) {
        QPOS; QBP(110);
        Raw r;
        r.w[0] = ld16(bp);
        r.w[1] = ld16(bp + 2 + 8 * c + 2 * g);
        r.w[2] = (unsigned int)bp[66 + c] | ((unsigned int)bp[74 + 4 * c + g] << 8);
        r.w[3] = bp[106 + (c >> 1)];
        return r;
    }
    __device__ static u16x8 decode(const Raw &r, int k) {
        QPOS;
        float d = half_to_float((half_raw)r.w[0]);
        int scale_b = (int)r.w[3];
        float db = ((c & 1) == 0)
            ? d * (float)(1 + 2 * (scale_b & 0xf))
            : d * (float)(1 + 2 * (scale_b >> 4));
        int qh = (int)(r.w[2] & 255);
        unsigned char s = (unsigned char)(r.w[2] >> 8);
        int i1 = (int)(r.w[1] & 255) | ((qh << (8 - 2 * g)) & 256);
        int i2 = (int)((r.w[1] >> 8) & 255) | ((qh << (7 - 2 * g)) & 256);
        unsigned int g1 = gw_iq3s_grid[i1];
        unsigned int g2 = gw_iq3s_grid[i2];
        u16x8 o;
#pragma unroll
        for (int t = 0; t < 8; t++) {
            unsigned int grid = t >= 4 ? g2 : g1;
            int j4 = t & 3;
            float val = db * (float)(unsigned char)(grid >> (8 * j4)) * ((s & (1 << t)) ? -1.0f : 1.0f);
            o[t] = f32_to_bf16(val);
        }
        return o;
    }
};

/* IQ1_S, 50 B: d | qs[32] | qh[8] u16. */
struct LdIQ1S {
    __device__ static size_t row_bytes(int K) { return (size_t)(K / 256) * 50; }
    __device__ static Raw load(const unsigned char *row, int k) {
        QPOS; QBP(50);
        Raw r;
        r.w[0] = ld16(bp);
        r.w[1] = bp[2 + 4 * c + g];
        r.w[2] = ld16(bp + 34 + 2 * c);
        return r;
    }
    __device__ static u16x8 decode(const Raw &r, int k) {
        QPOS;
        float d = half_to_float((half_raw)r.w[0]);
        const float IQ1S_DELTA = 0.125f;
        unsigned int qh = r.w[2];
        float dl  = d * (float)(2 * ((qh >> 12) & 7) + 1);
        float delta = (qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
        int grid_idx = (int)r.w[1] | (((qh >> (3 * g)) & 7) << 8);
        unsigned long long grid = gw_iq1s_grid[grid_idx];
        u16x8 o;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = dl * ((float)(signed char)(grid >> (8 * j)) + delta);
            o[j] = f32_to_bf16(val);
        }
        return o;
    }
};

/* IQ1_M, 56 B: qs[32] | qh[16] | scales[4] u16 (d packed in the top nibbles). */
struct LdIQ1M {
    __device__ static size_t row_bytes(int K) { return (size_t)(K / 256) * 56; }
    __device__ static Raw load(const unsigned char *row, int k) {
        QPOS; QBP(56);
        Raw r;
        r.w[0] = ld32(bp + 48);
        r.w[1] = ld32(bp + 52);
        r.w[2] = (unsigned int)bp[4 * c + g] | ((unsigned int)bp[32 + 2 * c + (g >> 1)] << 8);
        return r;
    }
    __device__ static u16x8 decode(const Raw &r, int k) {
        QPOS;
        unsigned short sc0 = (unsigned short)(r.w[0] & 0xffff), sc1 = (unsigned short)(r.w[0] >> 16);
        unsigned short sc2 = (unsigned short)(r.w[1] & 0xffff), sc3 = (unsigned short)(r.w[1] >> 16);
        unsigned short su = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0u) |
                            ((sc2 >> 4) & 0x0f00u) | (sc3 & 0xf000u);
        float d = half_to_float(su);
        unsigned short sw = (c >> 1) == 0 ? sc0 : (c >> 1) == 1 ? sc1 : (c >> 1) == 2 ? sc2 : sc3;
        int shift = 6 * (c % 2) + (g >= 2 ? 3 : 0);
        float dl = d * (float)(2 * ((sw >> shift) & 7) + 1);
        unsigned char hv = (unsigned char)(r.w[2] >> 8);
        int qshift = (g & 1) ? 4 : 8;
        unsigned short idx = (unsigned short)(r.w[2] & 255) | ((unsigned short)(hv << qshift) & 0x700u);
        float delta = (hv & ((g & 1) ? 0x80 : 0x08)) ? -0.125f : 0.125f;
        unsigned long long grid = gw_iq1s_grid[idx];
        u16x8 o;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float val = dl * ((float)(signed char)(grid >> (8 * j)) + delta);
            o[j] = f32_to_bf16(val);
        }
        return o;
    }
};

/* IQ4_XS, 136 B: d | scales_h u16 | scales_l[4] | qs[128]. */
struct LdIQ4XS {
    __device__ static size_t row_bytes(int K) { return (size_t)(K / 256) * 136; }
    __device__ static Raw load(const unsigned char *row, int k) {
        QPOS; QBP(136);
        Raw r;
        r.w[0] = ld32(bp);                       /* d | scales_h */
        r.w[1] = bp[4 + (c >> 1)];
        const unsigned char *q = bp + 8 + 16 * c + 8 * (g & 1);
        r.w[2] = ld32(q);
        r.w[3] = ld32(q + 4);
        return r;
    }
    __device__ static u16x8 decode(const Raw &r, int k) {
        QPOS;
        float d = half_to_float((half_raw)(r.w[0] & 0xffff));
        unsigned short scales_h = (unsigned short)(r.w[0] >> 16);
        int ls = (((int)r.w[1] >> (4 * (c & 1))) & 0xf) |
                 (((scales_h >> (2 * c)) & 3) << 4);
        float dl = d * (float)(ls - 32);
        int hi = g >> 1;
        u16x8 o;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned int byte = ((j < 4 ? r.w[2] : r.w[3]) >> (8 * (j & 3))) & 255;
            int q = hi ? (int)(byte >> 4) : (int)(byte & 0xf);
            float val = dl * (float)gw_kvalues_iq4nl[q];
            o[j] = f32_to_bf16(val);
        }
        return o;
    }
};

/* Q2_K, 84 B: scales[16] | qs[64] | d | dmin.  Sub-block c = n0*4 + j. */
struct LdQ2K {
    __device__ static size_t row_bytes(int K) { return (size_t)(K / 256) * 84; }
    __device__ static Raw load(const unsigned char *row, int k) {
        QPOS; QBP(84);
        const int n0 = c >> 2, s16 = g >> 1;
        Raw r;
        r.w[0] = ld32(bp + 80);                  /* d | dmin */
        r.w[1] = bp[n0 * 8 + (c & 3) * 2 + s16];
        const unsigned char *q = bp + 16 + n0 * 32 + s16 * 16 + 8 * (g & 1);
        r.w[2] = ld32(q);
        r.w[3] = ld32(q + 4);
        return r;
    }
    __device__ static u16x8 decode(const Raw &r, int k) {
        QPOS;
        float d = half_to_float((half_raw)(r.w[0] & 0xffff));
        float dmin = half_to_float((half_raw)(r.w[0] >> 16));
        unsigned char sc = (unsigned char)r.w[1];
        float dl = d * (float)(sc & 0xF);
        float ml = dmin * (float)(sc >> 4);
        const int shift = (c & 3) * 2;
        u16x8 o;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned int qb = ((j < 4 ? r.w[2] : r.w[3]) >> (8 * (j & 3))) & 255;
            float val = dl * (float)((qb >> shift) & 3) - ml;
            o[j] = f32_to_bf16(val);
        }
        return o;
    }
};

/* ---- WMMA wrappers ---- */
struct TagBF16 {};
struct TagF16 {};

template <typename T>
__device__ __forceinline__ float8 wmma16(u16x8 a, u16x8 b, float8 c);

template <>
__device__ __forceinline__ float8 wmma16<TagBF16>(u16x8 a, u16x8 b, float8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
template <>
__device__ __forceinline__ float8 wmma16<TagF16>(u16x8 a, u16x8 b, float8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
        __builtin_bit_cast(f16x8, a), __builtin_bit_cast(f16x8, b), c);
}

/* gridDim.z > 1: split K; blockIdx.z selects a contiguous K range and the
 * partial tile goes to Y + z*M*N (reduced by gemm_wmma_splitk_reduce). */
template <typename T, typename LB, int BM, int BN, int WAVES_M, int WAVES_N>
__device__ __forceinline__ void gemm_wmma_body(
        float *__restrict__ Y, const unsigned char *__restrict__ W,
        const unsigned short *__restrict__ X, int N, int K, int M) {
    constexpr int NT = WAVES_M * WAVES_N * 32;
    constexpr int WTM = BM / WAVES_M;          /* per-wave tile rows */
    constexpr int WTN = BN / WAVES_N;          /* per-wave tile cols */
    constexpr int FM = WTM / 16;
    constexpr int FN = WTN / 16;
    constexpr int AV = BM * 4 / NT;            /* A vectors per thread */
    constexpr int BV = BN * 4 / NT;            /* B vectors per thread */
    static_assert(AV >= 1 && BV >= 1, "tile too small for thread count");
    static_assert(BM * 4 % NT == 0 && BN * 4 % NT == 0, "tile/thread mismatch");

    __shared__ u16x8 smA[2 * 4 * BM];
    __shared__ u16x8 smB[2 * 4 * BN];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int wm = wave % WAVES_M;
    const int wn = wave / WAVES_M;
    const int half = lane >> 4;
    const int idx = lane & 15;
    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;

    /* split-K range */
    const int ksplit = gridDim.z;
    const int kchunk = ((K / 32 + ksplit - 1) / ksplit) * 32;
    const int kbeg = blockIdx.z * kchunk;
    int kend = kbeg + kchunk;
    if (kend > K) kend = K;
    if (ksplit > 1) Y += (size_t)blockIdx.z * M * N;

    /* Per-thread load slots: vector e -> (row = e>>2, kslot = e&3). */
    const u16x8 *pa[AV];
    const unsigned char *pb[BV];
    int sa[AV], sb[BV], kb[BV];
    const size_t wrow = LB::row_bytes(K);
#pragma unroll
    for (int i = 0; i < AV; i++) {
        int e = tid + i * NT;
        int r = e >> 2, ks = e & 3;
        int gr = m0 + r; if (gr > M - 1) gr = M - 1;
        pa[i] = (const u16x8 *)(X + (size_t)gr * K + kbeg + ks * 8);
        sa[i] = ks * BM + r;
    }
#pragma unroll
    for (int i = 0; i < BV; i++) {
        int e = tid + i * NT;
        int r = e >> 2, ks = e & 3;
        int gr = n0 + r; if (gr > N - 1) gr = N - 1;
        pb[i] = W + (size_t)gr * wrow;
        kb[i] = kbeg + ks * 8;
        sb[i] = ks * BN + r;
    }

    float8 acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++)
            acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};

    const int nk = (kend - kbeg) / 32;
    if (nk > 0) {
        u16x8 ra[AV];
        Raw rb[BV];
#pragma unroll
        for (int i = 0; i < AV; i++) smA[sa[i]] = pa[i][0];
#pragma unroll
        for (int i = 0; i < BV; i++) smB[sb[i]] = LB::decode(LB::load(pb[i], kb[i]), kb[i]);
        __syncthreads();

        int buf = 0;
        for (int kt = 0; kt < nk; kt++) {
            const int has_next = kt + 1 < nk;
            const int kn = (kt + 1) * 32;
            if (has_next) {
#pragma unroll
                for (int i = 0; i < AV; i++) ra[i] = pa[i][(kt + 1) * 4];
#pragma unroll
                for (int i = 0; i < BV; i++) rb[i] = LB::load(pb[i], kb[i] + kn);
            }
            const u16x8 *cA = smA + buf * 4 * BM;
            const u16x8 *cB = smB + buf * 4 * BN;
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                const int ks = kk * 2 + half;
                u16x8 fa[FM], fb[FN];
#pragma unroll
                for (int i = 0; i < FM; i++) fa[i] = cA[ks * BM + wm * WTM + i * 16 + idx];
#pragma unroll
                for (int j = 0; j < FN; j++) fb[j] = cB[ks * BN + wn * WTN + j * 16 + idx];
#pragma unroll
                for (int i = 0; i < FM; i++)
#pragma unroll
                    for (int j = 0; j < FN; j++)
                        acc[i][j] = wmma16<T>(fa[i], fb[j], acc[i][j]);
            }
            if (has_next) {
                u16x8 *nA = smA + (buf ^ 1) * 4 * BM;
                u16x8 *nB = smB + (buf ^ 1) * 4 * BN;
#pragma unroll
                for (int i = 0; i < AV; i++) nA[sa[i]] = ra[i];
#pragma unroll
                for (int i = 0; i < BV; i++) nB[sb[i]] = LB::decode(rb[i], kb[i] + kn);
                buf ^= 1;
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < FN; j++) {
        const int col = n0 + wn * WTN + j * 16 + idx;
        if (col >= N) continue;
#pragma unroll
        for (int i = 0; i < FM; i++) {
            const int row0 = m0 + wm * WTM + i * 16 + half * 8;
#pragma unroll
            for (int v = 0; v < 8; v++) {
                if (row0 + v < M) Y[(size_t)(row0 + v) * N + col] = acc[i][j][v];
            }
        }
    }
}

#define GEMM_WMMA_INST(name, T, LB, BM, BN, WMs, WNs)                              \
extern "C" __global__ __launch_bounds__(WMs * WNs * 32) void name(                 \
        float *Y, const unsigned char *W, const unsigned short *X,                 \
        int N, int K, int M) {                                                     \
    gemm_wmma_body<T, LB, BM, BN, WMs, WNs>(Y, W, X, N, K, M);                     \
}

/* Dense 16-bit W.  name: gemm_wmma_<type>_<BM>x<BN>_w<WAVES_M>x<WAVES_N>.
 * The variant table in gemm_wmma_dispatch.h mirrors this list. */
GEMM_WMMA_INST(gemm_wmma_bf16_128x128_w2x2, TagBF16, LdDense, 128, 128, 2, 2)
GEMM_WMMA_INST(gemm_wmma_bf16_128x64_w2x2,  TagBF16, LdDense, 128,  64, 2, 2)
GEMM_WMMA_INST(gemm_wmma_bf16_64x128_w2x2,  TagBF16, LdDense,  64, 128, 2, 2)
GEMM_WMMA_INST(gemm_wmma_bf16_64x64_w2x2,   TagBF16, LdDense,  64,  64, 2, 2)
GEMM_WMMA_INST(gemm_wmma_bf16_128x128_w2x4, TagBF16, LdDense, 128, 128, 2, 4)
GEMM_WMMA_INST(gemm_wmma_bf16_128x256_w2x4, TagBF16, LdDense, 128, 256, 2, 4)
GEMM_WMMA_INST(gemm_wmma_bf16_256x128_w4x2, TagBF16, LdDense, 256, 128, 4, 2)
GEMM_WMMA_INST(gemm_wmma_f16_128x128_w2x2,  TagF16,  LdDense, 128, 128, 2, 2)
GEMM_WMMA_INST(gemm_wmma_f16_64x64_w2x2,    TagF16,  LdDense,  64,  64, 2, 2)
GEMM_WMMA_INST(gemm_wmma_bf16_32x128_w1x4,  TagBF16, LdDense,  32, 128, 1, 4)
GEMM_WMMA_INST(gemm_wmma_bf16_32x64_w1x4,   TagBF16, LdDense,  32,  64, 1, 4)
GEMM_WMMA_INST(gemm_wmma_bf16_64x128_w1x4,  TagBF16, LdDense,  64, 128, 1, 4)

/* Fused dequant: GGML-quantized W decoded to BF16 in the tile loader.
 * name: gemm_wmma_q_<fmt>_<BM>x<BN>. */
#define GEMM_WMMA_Q_INST(fmt, LB)                                                  \
    GEMM_WMMA_INST(gemm_wmma_q_##fmt##_128x128, TagBF16, LB, 128, 128, 2, 2)       \
    GEMM_WMMA_INST(gemm_wmma_q_##fmt##_64x64,   TagBF16, LB,  64,  64, 2, 2)       \
    GEMM_WMMA_INST(gemm_wmma_q_##fmt##_256x128, TagBF16, LB, 256, 128, 4, 2)       \
    GEMM_WMMA_INST(gemm_wmma_q_##fmt##_128x128w8, TagBF16, LB, 128, 128, 2, 4)     \
    GEMM_WMMA_INST(gemm_wmma_q_##fmt##_256x64,  TagBF16, LB, 256,  64, 4, 2)
GEMM_WMMA_Q_INST(iq2_xxs, LdIQ2XXS)
GEMM_WMMA_Q_INST(iq2_xs,  LdIQ2XS)
GEMM_WMMA_Q_INST(iq2_s,   LdIQ2S)
GEMM_WMMA_Q_INST(iq3_xxs, LdIQ3XXS)
GEMM_WMMA_Q_INST(iq3_s,   LdIQ3S)
GEMM_WMMA_Q_INST(iq1_s,   LdIQ1S)
GEMM_WMMA_Q_INST(iq1_m,   LdIQ1M)
GEMM_WMMA_Q_INST(iq4_xs,  LdIQ4XS)
GEMM_WMMA_Q_INST(q2_k,    LdQ2K)

/* Fixed-order split-K reduction: Y[i] = sum_z P[z][i] for z = 0..splits-1. */
extern "C" __global__ void gemm_wmma_splitk_reduce(
        float *__restrict__ Y, const float *__restrict__ P, int splits, long n) {
    long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 s = *(const float4 *)(P + i);
        for (int z = 1; z < splits; z++) {
            float4 t = *(const float4 *)(P + (long)z * n + i);
            s.x += t.x; s.y += t.y; s.z += t.z; s.w += t.w;
        }
        *(float4 *)(Y + i) = s;
    } else {
        for (; i < n; i++) {
            float s = P[i];
            for (int z = 1; z < splits; z++) s += P[(long)z * n + i];
            Y[i] = s;
        }
    }
}
