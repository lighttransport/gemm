/* Standalone dev harness for the IQ2_XXS MMQ port (cuda/llm MoE expert FFN).
 *
 * Goal: replace the per-expert "dequant IQ2_XXS -> F16 weights + tiny cuBLAS GEMM"
 * loop with a single quantized matmul that consumes the IQ2_XXS weights directly
 * (no F16 weight materialization) and int8 q8_1 activations, the way llama.cpp's
 * mmq.cuh / mul_mat_id does. This file de-risks the *data path* against an exact
 * CPU oracle before any tiling/mma or runner integration.
 *
 * Phases (this file = Phase 0 + Phase 1):
 *   P0  CPU oracle: exact IQ2_XXS dequant + F32 matmul (ground truth).
 *   P1  GPU MMQ (correctness-first): on-GPU q8_1 activation quant + on-GPU
 *       IQ2_XXS weight decode-to-int8 + dp4a dot, one thread per output. Validate
 *       vs oracle (int8-activation path => expect ~1e-2 rel_L2, like real MMQ).
 *   P2  (later) tile + mma.sync.s8.s8.s32 for sm_120 tensor cores.
 *   P3  (later) mul_mat_id expert compaction + integration into cuda_llm_runner.c.
 *
 * Build: make -C cuda/llm/mmq ; run: ./cuda/llm/mmq/mmq_iq2xxs_test
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "iq2xxs_tables.h"

#define QK_K 256          /* IQ2_XXS super-block elements */
#define IQ2XXS_BYTES 66   /* 2 (half d) + 32*2 (qs) */
#define CK 32             /* q8_1 / per-scale granularity */

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

/* ---------------- host half -> float ---------------- */
static float half_to_float_h(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, out;
    if (e == 0) {
        if (m == 0) out = s << 31;
        else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff;
               out = (s << 31) | (e << 23) | (m << 13); }
    } else if (e == 0x1f) {
        out = (s << 31) | (0xff << 23) | (m << 13);
    } else {
        out = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f; __builtin_memcpy(&f, &out, 4); return f;
}

/* host float -> half (round to nearest, finite inputs only) */
static uint16_t float_to_half_h(float f) {
    uint32_t x; __builtin_memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e <= 0) return (uint16_t)s;             /* underflow -> 0 (fine for test scales) */
    if (e >= 0x1f) return (uint16_t)(s | 0x7bff); /* clamp to max finite half */
    return (uint16_t)(s | (e << 10) | (m >> 13));
}

/* ---------------- IQ2_XXS block decode (host, exact) ----------------
 * Matches the validated in-tree dequant_iq2_xxs_triplet_to_f16 element order:
 * out[ib32*32 + l*8 + j], l in 0..3 (group), j in 0..7 (within grid u64). */
static void iq2xxs_decode_block_h(const uint8_t *bp, float *out) {
    float d = half_to_float_h(*(const uint16_t *)bp);
    const uint16_t *qs = (const uint16_t *)(bp + 2);
    for (int ib = 0; ib < 8; ib++) {
        uint32_t aux0 = (uint32_t)qs[4*ib] | ((uint32_t)qs[4*ib+1] << 16);
        uint32_t aux1 = (uint32_t)qs[4*ib+2] | ((uint32_t)qs[4*ib+3] << 16);
        float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            uint8_t idx = (uint8_t)((aux0 >> (8*l)) & 255);
            const uint8_t *g = (const uint8_t *)&iq2xxs_grid[idx];
            uint8_t s = ksigns_iq2xs[(aux1 >> (7*l)) & 127];
            for (int j = 0; j < 8; j++)
                out[ib*32 + l*8 + j] = db * (float)g[j] * ((s & (1 << j)) ? -1.0f : 1.0f);
        }
    }
}

/* Decode a full quantized row [K] (K/256 blocks) -> float[K]. */
static void iq2xxs_decode_row_h(const uint8_t *row, int K, float *out) {
    int nb = K / QK_K;
    for (int b = 0; b < nb; b++) iq2xxs_decode_block_h(row + b * IQ2XXS_BYTES, out + b * QK_K);
}

/* ---------------- device tables ---------------- */
__constant__ uint64_t c_grid[256];
__constant__ uint8_t  c_ksigns[128];

/* device half->float (used by kernel B) */
__device__ static float half_to_float_dev(uint16_t h) {
    return __half2float(*(const __half *)&h);
}

/* ---------------- P1 kernel A: quantize activations to q8_1 (per-32 scale) ----------------
 * X[M][K] f32 -> xq8[M][K] int8, xs[M][K/32] f32 scale. One block per (m, kb). */
__global__ void quantize_q8_1(const float *X, int8_t *xq8, float *xs, int M, int K) {
    int kb = blockIdx.x;            /* which 32-block */
    int m  = blockIdx.y;            /* which row */
    int t  = threadIdx.x;          /* 0..31 */
    int k  = kb * CK + t;
    float v = (m < M && k < K) ? X[(size_t)m * K + k] : 0.0f;
    float a = fabsf(v);
    /* warp max-abs */
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
    float scale = a / 127.0f;
    float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    int q = (int)rintf(v * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    xq8[(size_t)m * K + k] = (int8_t)q;
    if (t == 0) xs[(size_t)m * (K / CK) + kb] = scale;
}

/* ---------------- P1 kernel B: MMQ IQ2_XXS x q8_1, one thread per (m,n) ----------------
 * W: IQ2_XXS quantized [N][K] (row_bytes = K/256 * 66). dst[M][N] = sum_k W[n][k]*X[m][k].
 * Decodes each weight 32-group to int8 on the fly, dp4a against q8_1 activations. */
__global__ void mmq_iq2xxs_naive(float *dst, const uint8_t *W, const int8_t *xq8,
                                  const float *xs, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;   /* output col (weight row) */
    int m = blockIdx.y;                              /* output row (token) */
    if (n >= N || m >= M) return;
    int nb = K / QK_K;
    int row_bytes = nb * IQ2XXS_BYTES;
    const uint8_t *wrow = W + (size_t)n * row_bytes;
    const int8_t *xrow = xq8 + (size_t)m * K;
    const float *xsrow = xs + (size_t)m * (K / CK);
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *bp = wrow + b * IQ2XXS_BYTES;
        float d = half_to_float_dev(*(const uint16_t *)bp);
        const uint16_t *qs = (const uint16_t *)(bp + 2);
        for (int ib = 0; ib < 8; ib++) {           /* eight 32-sub-blocks */
            uint32_t aux0 = (uint32_t)qs[4*ib] | ((uint32_t)qs[4*ib+1] << 16);
            uint32_t aux1 = (uint32_t)qs[4*ib+2] | ((uint32_t)qs[4*ib+3] << 16);
            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            int kbase = b * QK_K + ib * CK;          /* global k of this 32-block */
            int sumi = 0;
            for (int l = 0; l < 4; l++) {            /* four 8-element grid groups */
                uint8_t idx = (uint8_t)((aux0 >> (8*l)) & 255);
                const uint8_t *g = (const uint8_t *)&c_grid[idx];
                uint8_t s = c_ksigns[(aux1 >> (7*l)) & 127];
                /* pack 4+4 signed int8 weights and dp4a against activations */
                int w0 = 0, w1 = 0, x0, x1;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int wv = (int)g[j] * ((s & (1 << j)) ? -1 : 1);
                    w0 |= (wv & 0xff) << (8 * j);
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int wv = (int)g[4 + j] * ((s & (1 << (4 + j))) ? -1 : 1);
                    w1 |= (wv & 0xff) << (8 * j);
                }
                const int8_t *xp = xrow + kbase + l * 8;
                x0 = *(const int *)(xp);
                x1 = *(const int *)(xp + 4);
                sumi = __dp4a(w0, x0, sumi);
                sumi = __dp4a(w1, x1, sumi);
            }
            acc += db * xsrow[(kbase) / CK] * (float)sumi;
        }
    }
    dst[(size_t)m * N + n] = acc;
}

/* ================= PHASE 2: tensor-core MMA kernel =================
 * One warp computes a 16(weight-row n) x 8(token m) output tile. Loops over K in
 * 32-element sub-blocks; each sub-block is exactly one mma.sync.m16n8k32.s8.s8.s32
 * k-step, so the IQ2_XXS per-32 scale (db) and the q8_1 per-32 activation scale
 * apply once per MMA. Canonical PTX m16n8k32 fragment layout:
 *   groupID = lane>>2 (0-7), tid = lane&3 (0-3)
 *   A rows: {groupID, groupID+8};  B/C col(token): tid*2 + {0,1}; C rows = A rows.
 * blockDim = 32 (1 warp). grid = (N/16, M_tiles of 8). Requires N % 16 == 0.
 */
static __device__ __forceinline__ int pack4(const int8_t *p) {
    return (p[0] & 0xff) | ((p[1] & 0xff) << 8) | ((p[2] & 0xff) << 16) | ((p[3] & 0xff) << 24);
}

__global__ void mmq_iq2xxs_mma(float *dst, const uint8_t *W, const int8_t *xq8,
                                const float *xs, int M, int N, int K) {
    int lane = threadIdx.x;
    int gid = lane >> 2;       /* 0..7 */
    int tid = lane & 3;        /* 0..3 */
    int n0 = blockIdx.x * 16;  /* first weight-row of this tile */
    int m0 = blockIdx.y * 8;   /* first token of this tile */
    int nb = K / QK_K, row_bytes = nb * IQ2XXS_BYTES;
    int nsb = K / CK;          /* number of 32-sub-blocks */

    __shared__ int8_t sW[16][CK];
    __shared__ int8_t sX[8][CK];
    __shared__ float  sWs[16];
    __shared__ float  sXs[8];

    float f0 = 0, f1 = 0, f2 = 0, f3 = 0;

    for (int sb = 0; sb < nsb; sb++) {
        /* stage activations: 8 tokens x 32 (zero-pad tokens >= M) */
        for (int i = lane; i < 8 * CK; i += 32) {
            int t = i / CK, kk = i % CK;
            int m = m0 + t;
            sX[t][kk] = (m < M) ? xq8[(size_t)m * K + sb * CK + kk] : 0;
        }
        if (lane < 8) { int m = m0 + lane; sXs[lane] = (m < M) ? xs[(size_t)m * nsb + sb] : 0.0f; }
        /* stage weights: decode IQ2_XXS sub-block sb for 16 rows (16 lanes active) */
        if (lane < 16) {
            int r = lane, n = n0 + r;
            const uint8_t *bp = W + (size_t)n * row_bytes + (sb / 8) * IQ2XXS_BYTES;
            float d = half_to_float_dev(*(const uint16_t *)bp);
            const uint16_t *qs = (const uint16_t *)(bp + 2);
            int ib = sb & 7;
            uint32_t aux0 = (uint32_t)qs[4*ib] | ((uint32_t)qs[4*ib+1] << 16);
            uint32_t aux1 = (uint32_t)qs[4*ib+2] | ((uint32_t)qs[4*ib+3] << 16);
            sWs[r] = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                uint8_t idx = (uint8_t)((aux0 >> (8*l)) & 255);
                const uint8_t *g = (const uint8_t *)&c_grid[idx];
                uint8_t s = c_ksigns[(aux1 >> (7*l)) & 127];
                for (int j = 0; j < 8; j++)
                    sW[r][l*8 + j] = (int8_t)((int)g[j] * ((s & (1 << j)) ? -1 : 1));
            }
        }
        __syncwarp();

        /* load fragments (canonical m16n8k32) */
        int a0 = pack4(&sW[gid][tid*4]);
        int a1 = pack4(&sW[gid+8][tid*4]);
        int a2 = pack4(&sW[gid][tid*4 + 16]);
        int a3 = pack4(&sW[gid+8][tid*4 + 16]);
        int b0 = pack4(&sX[gid][tid*4]);
        int b1 = pack4(&sX[gid][tid*4 + 16]);
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "r"(0), "r"(0), "r"(0), "r"(0));
        /* scale by per-sub-block weight scale (row) x activation scale (token) */
        float wr0 = sWs[gid], wr8 = sWs[gid + 8];
        float xc0 = sXs[tid*2], xc1 = sXs[tid*2 + 1];
        f0 += wr0 * xc0 * (float)c0;
        f1 += wr0 * xc1 * (float)c1;
        f2 += wr8 * xc0 * (float)c2;
        f3 += wr8 * xc1 * (float)c3;
        __syncwarp();
    }
    /* write: C row = weight-col n, C col = token m; dst is [M][N] row-major */
    int n_a = n0 + gid, n_b = n0 + gid + 8;
    int m_a = m0 + tid*2, m_b = m0 + tid*2 + 1;
    if (m_a < M) { dst[(size_t)m_a * N + n_a] = f0; dst[(size_t)m_a * N + n_b] = f2; }
    if (m_b < M) { dst[(size_t)m_b * N + n_a] = f1; dst[(size_t)m_b * N + n_b] = f3; }
}

/* ================= PHASE 2.5: occupancy-tuned MMA =================
 * Block = WN warps (128 threads). Each block computes (16*WN) weight-rows x 8
 * tokens. The 8-token activation sub-block is staged ONCE per block and reused by
 * all WN warps (the MoE shape is small-M / large-N, so weight rows dominate and
 * activation reuse + 128 thr/block lifts occupancy vs the 1-warp/tile v1).
 * grid = (N/(16*WN), ceil(M/8)). Requires N % (16*WN) == 0.
 */
#ifndef WN
#define WN 4   /* warps per block -> 64 weight-rows/block */
#endif

__global__ void mmq_iq2xxs_mma_v2(float *dst, const uint8_t *W, const int8_t *xq8,
                                   const float *xs, int M, int N, int K) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;          /* 0..WN-1 */
    int gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * (16 * WN) + warp * 16;  /* this warp's first weight-row */
    int m0 = blockIdx.y * 8;
    int nb = K / QK_K, row_bytes = nb * IQ2XXS_BYTES, nsb = K / CK;

    __shared__ int8_t sX[8][CK];
    __shared__ float  sXs[8];
    __shared__ int8_t sW[16 * WN][CK];
    __shared__ float  sWs[16 * WN];

    float f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    for (int sb = 0; sb < nsb; sb++) {
        /* activations: all 128 threads stage 8x32 once per block */
        for (int i = threadIdx.x; i < 8 * CK; i += blockDim.x) {
            int t = i / CK, kk = i % CK, m = m0 + t;
            sX[t][kk] = (m < M) ? xq8[(size_t)m * K + sb * CK + kk] : 0;
        }
        if (threadIdx.x < 8) { int m = m0 + threadIdx.x; sXs[threadIdx.x] = (m < M) ? xs[(size_t)m * nsb + sb] : 0.0f; }
        /* weights: each warp decodes its 16 rows (16 lanes active) */
        if (lane < 16) {
            int r = warp * 16 + lane, n = n0 + lane;
            const uint8_t *bp = W + (size_t)n * row_bytes + (sb / 8) * IQ2XXS_BYTES;
            float d = half_to_float_dev(*(const uint16_t *)bp);
            const uint16_t *qs = (const uint16_t *)(bp + 2);
            int ib = sb & 7;
            uint32_t aux0 = (uint32_t)qs[4*ib] | ((uint32_t)qs[4*ib+1] << 16);
            uint32_t aux1 = (uint32_t)qs[4*ib+2] | ((uint32_t)qs[4*ib+3] << 16);
            sWs[r] = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                uint8_t idx = (uint8_t)((aux0 >> (8*l)) & 255);
                const uint8_t *g = (const uint8_t *)&c_grid[idx];
                uint8_t s = c_ksigns[(aux1 >> (7*l)) & 127];
                for (int j = 0; j < 8; j++)
                    sW[r][l*8 + j] = (int8_t)((int)g[j] * ((s & (1 << j)) ? -1 : 1));
            }
        }
        __syncthreads();
        int wr = warp * 16;
        int a0 = pack4(&sW[wr+gid][tid*4]),   a1 = pack4(&sW[wr+gid+8][tid*4]);
        int a2 = pack4(&sW[wr+gid][tid*4+16]), a3 = pack4(&sW[wr+gid+8][tid*4+16]);
        int b0 = pack4(&sX[gid][tid*4]),       b1 = pack4(&sX[gid][tid*4+16]);
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "r"(0), "r"(0), "r"(0), "r"(0));
        float wr0 = sWs[wr+gid], wr8 = sWs[wr+gid+8];
        float xc0 = sXs[tid*2], xc1 = sXs[tid*2+1];
        f0 += wr0 * xc0 * (float)c0; f1 += wr0 * xc1 * (float)c1;
        f2 += wr8 * xc0 * (float)c2; f3 += wr8 * xc1 * (float)c3;
        __syncthreads();
    }
    int n_a = n0 + gid, n_b = n0 + gid + 8;
    int m_a = m0 + tid*2, m_b = m0 + tid*2 + 1;
    if (m_a < M) { dst[(size_t)m_a * N + n_a] = f0; dst[(size_t)m_a * N + n_b] = f2; }
    if (m_b < M) { dst[(size_t)m_b * N + n_a] = f1; dst[(size_t)m_b * N + n_b] = f3; }
}

/* ================= PHASE 2.6: decode-amortized MMA =================
 * The decode-bound finding: IQ2_XXS weight decode dominates, so amortize it over
 * MORE tokens per decode. Block = WN warps; decodes its (16*WN) weight-rows for a
 * sub-block ONCE, then reuses across up to TG=4 token-groups (32 tokens) staged
 * together. grid = (N/(16*WN), ceil(M/(8*TG))). 2 syncs per sub-block.
 */
#ifndef TG
#define TG 4   /* token-groups per block (8*TG = 32 tokens) */
#endif

__global__ void mmq_iq2xxs_mma_v3(float *dst, const uint8_t *W, const int8_t *xq8,
                                   const float *xs, int M, int N, int K) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * (16 * WN) + warp * 16;
    int m_base = blockIdx.y * (8 * TG);
    int ntg = (M - m_base + 7) / 8; if (ntg > TG) ntg = TG; if (ntg < 0) ntg = 0;
    int nb = K / QK_K, row_bytes = nb * IQ2XXS_BYTES, nsb = K / CK;

    __shared__ int8_t sX[8 * TG][CK];
    __shared__ float  sXs[8 * TG];
    __shared__ int8_t sW[16 * WN][CK];
    __shared__ float  sWs[16 * WN];

    float f[TG][4];
    for (int g = 0; g < TG; g++) { f[g][0]=f[g][1]=f[g][2]=f[g][3]=0; }

    for (int sb = 0; sb < nsb; sb++) {
        /* decode this block's weight rows ONCE for this sub-block */
        if (lane < 16) {
            int r = warp*16 + lane, n = n0 + lane;
            const uint8_t *bp = W + (size_t)n*row_bytes + (sb/8)*IQ2XXS_BYTES;
            float d = half_to_float_dev(*(const uint16_t*)bp);
            const uint16_t *qs = (const uint16_t*)(bp+2); int ib = sb&7;
            uint32_t a0=(uint32_t)qs[4*ib]|((uint32_t)qs[4*ib+1]<<16);
            uint32_t a1=(uint32_t)qs[4*ib+2]|((uint32_t)qs[4*ib+3]<<16);
            sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;
            for(int l=0;l<4;l++){ uint8_t idx=(a0>>(8*l))&255; const uint8_t*g=(const uint8_t*)&c_grid[idx];
                uint8_t s=c_ksigns[(a1>>(7*l))&127];
                for(int j=0;j<8;j++) sW[r][l*8+j]=(int8_t)((int)g[j]*((s&(1<<j))?-1:1)); } }
        /* stage all ntg token-groups' activations together */
        for (int i = threadIdx.x; i < ntg*8*CK; i += blockDim.x) {
            int t = i/CK, kk = i%CK, m = m_base + t;
            sX[t][kk] = (m < M) ? xq8[(size_t)m*K + sb*CK + kk] : 0;
        }
        for (int t = threadIdx.x; t < ntg*8; t += blockDim.x) {
            int m = m_base + t; sXs[t] = (m < M) ? xs[(size_t)m*nsb + sb] : 0.0f;
        }
        __syncthreads();
        int wr = warp*16;
        int a0 = pack4(&sW[wr+gid][tid*4]),   a1 = pack4(&sW[wr+gid+8][tid*4]);
        int a2 = pack4(&sW[wr+gid][tid*4+16]), a3 = pack4(&sW[wr+gid+8][tid*4+16]);
        float wr0 = sWs[wr+gid], wr8 = sWs[wr+gid+8];
        for (int g = 0; g < ntg; g++) {
            int b0 = pack4(&sX[g*8+gid][tid*4]), b1 = pack4(&sX[g*8+gid][tid*4+16]);
            int c0=0,c1=0,c2=0,c3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0),"r"(0),"r"(0),"r"(0));
            float xc0 = sXs[g*8+tid*2], xc1 = sXs[g*8+tid*2+1];
            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;
            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;
        }
        __syncthreads();
    }
    int n_a = n0+gid, n_b = n0+gid+8;
    for (int g = 0; g < ntg; g++) {
        int m_a = m_base + g*8 + tid*2, m_b = m_a + 1;
        if (m_a < M) { dst[(size_t)m_a*N+n_a]=f[g][0]; dst[(size_t)m_a*N+n_b]=f[g][2]; }
        if (m_b < M) { dst[(size_t)m_b*N+n_a]=f[g][1]; dst[(size_t)m_b*N+n_b]=f[g][3]; }
    }
}

/* mma_v4: like v3 (decode-amortized, full 32-lane decode) but SOFTWARE-PIPELINED:
 * decode sub-block sb+1's weights/acts into a 2nd shared buffer WHILE the mma for
 * sb runs from the 1st buffer. Overlaps decode-ALU with tensor cores and collapses
 * to ONE __syncthreads per sub-block (v3 has two). grid/launch identical to v3. */
#define V4_DECODE(SBV, KB, XB, WSB, XSB)                                              \
    do {                                                                              \
        int rl_=lane>>1, half_=lane&1, r_=warp*16+rl_, n_=n0+rl_;                      \
        const uint8_t *bp_ = W + (size_t)n_*row_bytes + ((SBV)/8)*IQ2XXS_BYTES;        \
        float d_ = half_to_float_dev(*(const uint16_t*)bp_);                           \
        const uint16_t *qs_ = (const uint16_t*)(bp_+2); int ib_=(SBV)&7;               \
        uint32_t a0_=(uint32_t)qs_[4*ib_]|((uint32_t)qs_[4*ib_+1]<<16);                \
        uint32_t a1_=(uint32_t)qs_[4*ib_+2]|((uint32_t)qs_[4*ib_+3]<<16);              \
        if(half_==0) (WSB)[r_]=d_*(0.5f+(float)(a1_>>28))*0.25f;                       \
        for(int l_=half_*2;l_<half_*2+2;l_++){ uint8_t idx_=(a0_>>(8*l_))&255;         \
            const uint8_t*g_=(const uint8_t*)&c_grid[idx_];                            \
            uint8_t s_=c_ksigns[(a1_>>(7*l_))&127];                                    \
            for(int j_=0;j_<8;j_++) (KB)[r_][l_*8+j_]=(int8_t)((int)g_[j_]*((s_&(1<<j_))?-1:1)); } \
        for(int i_=threadIdx.x;i_<ntg*8*CK;i_+=blockDim.x){ int t_=i_/CK,kk_=i_%CK,m_=m_base+t_; \
            (XB)[t_][kk_]=(m_<M)?xq8[(size_t)m_*K+(SBV)*CK+kk_]:0; }                    \
        for(int i_=threadIdx.x;i_<ntg*8;i_+=blockDim.x){ int m_=m_base+i_;             \
            (XSB)[i_]=(m_<M)?xs[(size_t)m_*nsb+(SBV)]:0.0f; }                           \
    } while(0)

__global__ void mmq_iq2xxs_mma_v4(float *dst, const uint8_t *W, const int8_t *xq8,
                                   const float *xs, int M, int N, int K) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * (16 * WN) + warp * 16;
    int m_base = blockIdx.y * (8 * TG);
    int ntg = (M - m_base + 7) / 8; if (ntg > TG) ntg = TG; if (ntg < 0) ntg = 0;
    int nb = K / QK_K, row_bytes = nb * IQ2XXS_BYTES, nsb = K / CK;

    __shared__ int8_t sXA[8 * TG][CK]; __shared__ float sXsXA[8 * TG];
    __shared__ int8_t sXB[8 * TG][CK]; __shared__ float sXsXB[8 * TG];
    __shared__ int8_t sWA[16 * WN][CK]; __shared__ float sWsA[16 * WN];
    __shared__ int8_t sWB[16 * WN][CK]; __shared__ float sWsB[16 * WN];
    int8_t (*sWbuf[2])[CK] = { sWA, sWB };
    int8_t (*sXbuf[2])[CK] = { sXA, sXB };
    float  *sWsbuf[2] = { sWsA, sWsB };
    float  *sXsbuf[2] = { sXsXA, sXsXB };

    float f[TG][4];
    for (int g = 0; g < TG; g++) { f[g][0]=f[g][1]=f[g][2]=f[g][3]=0; }

    if (nsb > 0) { V4_DECODE(0, sWA, sXA, sWsA, sXsXA); }
    __syncthreads();

    for (int sb = 0; sb < nsb; sb++) {
        int cur = sb & 1, nxt = cur ^ 1;
        /* (A) prefetch+decode sb+1 into the OTHER buffer (overlaps with the mma below) */
        if (sb + 1 < nsb) {
            int snx = sb + 1;
            if (nxt == 1) { V4_DECODE(snx, sWB, sXB, sWsB, sXsXB); }
            else          { V4_DECODE(snx, sWA, sXA, sWsA, sXsXA); }
        }
        /* (B) mma for sb from the current buffer */
        int wr = warp*16;
        int8_t (*sW)[CK] = sWbuf[cur]; int8_t (*sX)[CK] = sXbuf[cur];
        float *sWs = sWsbuf[cur], *sXs = sXsbuf[cur];
        int a0 = pack4(&sW[wr+gid][tid*4]),   a1 = pack4(&sW[wr+gid+8][tid*4]);
        int a2 = pack4(&sW[wr+gid][tid*4+16]), a3 = pack4(&sW[wr+gid+8][tid*4+16]);
        float wr0 = sWs[wr+gid], wr8 = sWs[wr+gid+8];
        for (int g = 0; g < ntg; g++) {
            int b0 = pack4(&sX[g*8+gid][tid*4]), b1 = pack4(&sX[g*8+gid][tid*4+16]);
            int c0=0,c1=0,c2=0,c3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0),"r"(0),"r"(0),"r"(0));
            float xc0 = sXs[g*8+tid*2], xc1 = sXs[g*8+tid*2+1];
            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;
            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;
        }
        __syncthreads();
    }
    int n_a = n0+gid, n_b = n0+gid+8;
    for (int g = 0; g < ntg; g++) {
        int m_a = m_base + g*8 + tid*2, m_b = m_a + 1;
        if (m_a < M) { dst[(size_t)m_a*N+n_a]=f[g][0]; dst[(size_t)m_a*N+n_b]=f[g][2]; }
        if (m_b < M) { dst[(size_t)m_b*N+n_a]=f[g][1]; dst[(size_t)m_b*N+n_b]=f[g][3]; }
    }
}

/* ---------------- CPU oracle ---------------- */
static void cpu_oracle(const uint8_t *W, const float *X, float *dst, int M, int N, int K) {
    float *wf = (float *)malloc((size_t)K * sizeof(float));
    for (int n = 0; n < N; n++) {
        iq2xxs_decode_row_h(W + (size_t)n * (K / QK_K) * IQ2XXS_BYTES, K, wf);
        for (int m = 0; m < M; m++) {
            const float *x = X + (size_t)m * K;
            double s = 0.0;
            for (int k = 0; k < K; k++) s += (double)wf[k] * (double)x[k];
            dst[(size_t)m * N + n] = (float)s;
        }
    }
    free(wf);
}

static double rel_l2(const float *a, const float *b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return den > 0 ? sqrt(num / den) : sqrt(num);
}

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 16;     /* tokens for one expert */
    int N = argc > 2 ? atoi(argv[2]) : 512;    /* expert_ff (gate/up out_dim) */
    int K = argc > 3 ? atoi(argv[3]) : 2048;   /* n_embd */
    printf("MMQ IQ2_XXS test: M=%d N=%d K=%d  (K/256=%d blocks/row)\n", M, N, K, K / QK_K);
    if (K % QK_K) { fprintf(stderr, "K must be multiple of 256\n"); return 1; }

    srand(1234);
    int nb = K / QK_K, row_bytes = nb * IQ2XXS_BYTES;
    size_t wbytes = (size_t)N * row_bytes;
    uint8_t *W = (uint8_t *)malloc(wbytes);
    for (size_t i = 0; i < wbytes; i++) W[i] = rand() & 0xff;      /* random qs (indices/signs/ls) */
    /* Overwrite each block's half scale d with a finite, realistic value (random bytes
     * could encode half-Inf/NaN; real GGUF d ~ small positive). */
    for (int n = 0; n < N; n++)
        for (int b = 0; b < nb; b++) {
            float d = 0.01f + (rand() / (float)RAND_MAX) * 0.05f;
            *(uint16_t *)(W + (size_t)n * row_bytes + b * IQ2XXS_BYTES) = float_to_half_h(d);
        }
    float *X = (float *)malloc((size_t)M * K * sizeof(float));
    for (int i = 0; i < M * K; i++) X[i] = ((rand() / (float)RAND_MAX) - 0.5f) * 2.0f;

    float *ref = (float *)malloc((size_t)M * N * sizeof(float));
    cpu_oracle(W, X, ref, M, N, K);

    /* device */
    CUDA_CHECK(cudaMemcpyToSymbol(c_grid, iq2xxs_grid, sizeof(iq2xxs_grid)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ksigns, ksigns_iq2xs, sizeof(ksigns_iq2xs)));
    uint8_t *dW; int8_t *dXq; float *dX, *dXs, *dDst;
    CUDA_CHECK(cudaMalloc(&dW, wbytes));
    CUDA_CHECK(cudaMalloc(&dX, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dXq, (size_t)M * K));
    CUDA_CHECK(cudaMalloc(&dXs, (size_t)M * (K / CK) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDst, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dW, W, wbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dX, X, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));

    dim3 qg(K / CK, M); quantize_q8_1<<<qg, 32>>>(dX, dXq, dXs, M, K);
    CUDA_CHECK(cudaGetLastError());
    dim3 bg((N + 127) / 128, M); mmq_iq2xxs_naive<<<bg, 128>>>(dDst, dW, dXq, dXs, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float *out = (float *)malloc((size_t)M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(out, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));

    double e = rel_l2(out, ref, M * N);
    printf("P1 dp4a   vs CPU-F32 oracle: rel_L2 = %.6f  (int8-activation path)\n", e);

    /* ---- Phase 2: tensor-core MMA ---- */
    if (N % 16 != 0) { fprintf(stderr, "P2 needs N %% 16 == 0\n"); return 1; }
    CUDA_CHECK(cudaMemset(dDst, 0, (size_t)M * N * sizeof(float)));
    dim3 mg(N / 16, (M + 7) / 8);
    mmq_iq2xxs_mma<<<mg, 32>>>(dDst, dW, dXq, dXs, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float *out2 = (float *)malloc((size_t)M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(out2, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    double e2 = rel_l2(out2, ref, M * N);
    double e2v1 = rel_l2(out2, out, M * N);   /* MMA vs dp4a (should be ~0: same int8 path) */
    printf("P2 mma    vs CPU-F32 oracle: rel_L2 = %.6f\n", e2);
    printf("P2 mma    vs P1 dp4a       : rel_L2 = %.6f  (same int8 path -> expect ~0)\n", e2v1);
    printf("  ref[0..3]=%.4f %.4f %.4f %.4f\n  mma[0..3]=%.4f %.4f %.4f %.4f\n",
           ref[0], ref[1], ref[2], ref[3], out2[0], out2[1], out2[2], out2[3]);
    /* ---- Phase 2.5: occupancy-tuned MMA (multi-warp block) ---- */
    double e3 = 1e9, e3v2 = 1e9;
    if (N % (16 * WN) == 0) {
        CUDA_CHECK(cudaMemset(dDst, 0, (size_t)M * N * sizeof(float)));
        dim3 vg(N / (16 * WN), (M + 7) / 8);
        mmq_iq2xxs_mma_v2<<<vg, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *out3 = (float *)malloc((size_t)M * N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(out3, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
        e3 = rel_l2(out3, ref, M * N); e3v2 = rel_l2(out3, out2, M * N);
        printf("P2.5 mma_v2 vs CPU-F32 oracle: rel_L2 = %.6f\n", e3);
        printf("P2.5 mma_v2 vs P2 mma         : rel_L2 = %.6f  (expect ~0)\n", e3v2);
        free(out3);
    } else printf("P2.5 skipped (N %% %d != 0)\n", 16 * WN);

    /* ---- Phase 2.6: decode-amortized MMA ---- */
    double e4 = 1e9, e4v = 1e9;
    if (N % (16 * WN) == 0) {
        CUDA_CHECK(cudaMemset(dDst, 0, (size_t)M * N * sizeof(float)));
        dim3 v3g(N / (16 * WN), (M + 8*TG - 1) / (8*TG));
        mmq_iq2xxs_mma_v3<<<v3g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *o4 = (float *)malloc((size_t)M * N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(o4, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
        e4 = rel_l2(o4, ref, M*N); e4v = rel_l2(o4, out2, M*N);
        printf("P2.6 mma_v3 vs CPU-F32 oracle: rel_L2 = %.6f\n", e4);
        printf("P2.6 mma_v3 vs P2 mma         : rel_L2 = %.6f  (expect ~0)\n", e4v);
        free(o4);
    } else printf("P2.6 skipped\n");

    /* ---- Phase 2.7: double-buffered decode (pipeline) ---- */
    double e5 = 1e9, e5v = 1e9;
    if (N % (16 * WN) == 0) {
        CUDA_CHECK(cudaMemset(dDst, 0, (size_t)M * N * sizeof(float)));
        dim3 v4g(N / (16 * WN), (M + 8*TG - 1) / (8*TG));
        mmq_iq2xxs_mma_v4<<<v4g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *o5 = (float *)malloc((size_t)M * N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(o5, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
        e5 = rel_l2(o5, ref, M*N); e5v = rel_l2(o5, out2, M*N);
        printf("P2.7 mma_v4 vs CPU-F32 oracle: rel_L2 = %.6f\n", e5);
        printf("P2.7 mma_v4 vs P2 mma         : rel_L2 = %.6f  (expect ~0)\n", e5v);
        free(o5);
    } else printf("P2.7 skipped\n");

    int ok = (e < 0.05) && (e2 < 0.05) && (e2v1 < 1e-4) && (e3 < 0.05) && (e3v2 < 1e-4) && (e4 < 0.05) && (e4v < 1e-4) && (e5 < 0.05) && (e5v < 1e-4);
    printf("%s\n", ok ? "PASS (P1..P2.6 data path correct)" : "FAIL");

    /* ---- rough throughput: dp4a vs mma (one GEMM, single-warp-per-tile) ---- */
    {
        int iters = 300;
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        dim3 bg((N + 127) / 128, M);
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) mmq_iq2xxs_naive<<<bg, 128>>>(dDst, dW, dXq, dXs, M, N, K);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_dp; cudaEventElapsedTime(&ms_dp, t0, t1); ms_dp /= iters;
        dim3 mg(N / 16, (M + 7) / 8);
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) mmq_iq2xxs_mma<<<mg, 32>>>(dDst, dW, dXq, dXs, M, N, K);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_mma; cudaEventElapsedTime(&ms_mma, t0, t1); ms_mma /= iters;
        float ms_v2 = 0;
        if (N % (16 * WN) == 0) {
            dim3 vg(N / (16 * WN), (M + 7) / 8);
            cudaEventRecord(t0);
            for (int i = 0; i < iters; i++) mmq_iq2xxs_mma_v2<<<vg, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms_v2, t0, t1); ms_v2 /= iters;
        }
        double flop = 2.0 * M * N * K;
        printf("\nperf (1 GEMM):\n");
        printf("  dp4a   : %.3f ms  %.1f GFLOP/s\n", ms_dp, flop / (ms_dp * 1e6));
        printf("  mma    : %.3f ms  %.1f GFLOP/s  (%.2fx vs dp4a)\n", ms_mma, flop / (ms_mma * 1e6), ms_dp / ms_mma);
        if (ms_v2 > 0) printf("  mma_v2 : %.3f ms  %.1f GFLOP/s  (%.2fx vs dp4a, %.2fx vs mma)\n",
                              ms_v2, flop / (ms_v2 * 1e6), ms_dp / ms_v2, ms_mma / ms_v2);
        float ms_v3 = 0;
        if (N % (16 * WN) == 0) {
            dim3 v3g(N / (16 * WN), (M + 8*TG - 1) / (8*TG));
            cudaEventRecord(t0);
            for (int i = 0; i < iters; i++) mmq_iq2xxs_mma_v3<<<v3g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms_v3, t0, t1); ms_v3 /= iters;
            printf("  mma_v3 : %.3f ms  %.1f GFLOP/s  (%.2fx vs dp4a, %.2fx vs mma)\n",
                   ms_v3, flop / (ms_v3 * 1e6), ms_dp / ms_v3, ms_mma / ms_v3);
        }
        float ms_v4 = 0;
        if (N % (16 * WN) == 0) {
            dim3 v4g(N / (16 * WN), (M + 8*TG - 1) / (8*TG));
            cudaEventRecord(t0);
            for (int i = 0; i < iters; i++) mmq_iq2xxs_mma_v4<<<v4g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms_v4, t0, t1); ms_v4 /= iters;
            printf("  mma_v4 : %.3f ms  %.1f GFLOP/s  (%.2fx vs dp4a, %.2fx vs v3)\n",
                   ms_v4, flop / (ms_v4 * 1e6), ms_dp / ms_v4, (ms_v3>0?ms_v3/ms_v4:0));
        }
    }
    return ok ? 0 : 1;
}
