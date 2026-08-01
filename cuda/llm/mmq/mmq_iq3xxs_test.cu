/* Standalone dev harness for the IQ3_XXS MMQ port (cuda/llm dense v_proj).
 *
 * Goal: replace the "dequant IQ3_XXS -> F16 weights + tiny cuBLAS GEMM"
 * per-layer v_proj (120ms prefill) with a single quantized matmul consuming
 * IQ3_XXS weights directly + int8 q8_1 activations.
 *
 * Build: make -C cuda/llm/mmq ; run: ./cuda/llm/mmq/mmq_iq3xxs_test
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "iq2xxs_tables.h"   /* for ksigns_iq2xs[128], reused by IQ3_XXS */

#define QK_K 256
#define IQ3XXS_BYTES 98    /* d(2) + qs(64) + scales_and_signs(32) */
#define CK 32

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

/* ---------------- host half helpers ---------------- */
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
static uint16_t float_to_half_h(float f) {
    uint32_t x; __builtin_memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e <= 0) return (uint16_t)s;
    if (e >= 0x1f) return (uint16_t)(s | 0x7bff);
    return (uint16_t)(s | (e << 10) | (m >> 13));
}

/* ---------------- IQ3_XXS grid table (uint32[256], 4 uint8 elements each) ---------------- */
static const uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

/* ---------------- host IQ3_XXS block decode ---------------- */
static void iq3xxs_decode_block_h(const uint8_t *bp, float *out) {
    float d = half_to_float_h(*(const uint16_t *)bp);
    const uint8_t *qs = bp + 2;
    const uint32_t *sas = (const uint32_t *)(bp + 66);
    for (int ib = 0; ib < 8; ib++) {
        float db = d * (0.5f + (float)(sas[ib] >> 28)) * 0.5f;
        for (int l = 0; l < 4; l++) {
            uint8_t g1_idx = qs[8*ib + 2*l + 0];
            uint8_t g2_idx = qs[8*ib + 2*l + 1];
            uint32_t g1 = iq3xxs_grid[g1_idx];
            uint32_t g2 = iq3xxs_grid[g2_idx];
            uint8_t sgn = ksigns_iq2xs[(sas[ib] >> (7*l)) & 127];
            for (int j = 0; j < 4; j++) {
                float w0 = db * (float)(uint8_t)(g1 >> (8*j)) * ((sgn & (1 << j)) ? -1.0f : 1.0f);
                float w1 = db * (float)(uint8_t)(g2 >> (8*j)) * ((sgn & (1 << (j+4))) ? -1.0f : 1.0f);
                out[ib*32 + l*8 + j + 0] = w0;
                out[ib*32 + l*8 + j + 4] = w1;
            }
        }
    }
}

static void iq3xxs_decode_row_h(const uint8_t *row, int K, float *out) {
    int nb = K / QK_K;
    for (int b = 0; b < nb; b++)
        iq3xxs_decode_block_h(row + b * IQ3XXS_BYTES, out + b * QK_K);
}

/* ---------------- device tables ---------------- */
__constant__ uint32_t c_grid[256];
__constant__ uint8_t  c_ksigns[128];

__device__ static float half_to_float_dev(uint16_t h) {
    return __half2float(*(const __half *)&h);
}

/* ---------------- kernel A: quantize activations to q8_1 ---------------- */
__global__ void quantize_q8_1(const float *X, int8_t *xq8, float *xs, int M, int K) {
    int kb = blockIdx.x, m = blockIdx.y, t = threadIdx.x, k = kb * CK + t;
    float v = (m < M && k < K) ? X[(size_t)m * K + k] : 0.0f;
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
    float scale = a / 127.0f;
    float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    int q = (int)rintf(v * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    xq8[(size_t)m * K + k] = (int8_t)q;
    if (t == 0) xs[(size_t)m * (K / CK) + kb] = scale;
}

/* ---------------- kernel B: dp4a MMQ (one thread per (m,n)) ---------------- */
__global__ void mmq_iq3xxs_naive(float *dst, const uint8_t *W, const int8_t *xq8,
                                  const float *xs, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;
    int nb = K / QK_K, row_bytes = nb * IQ3XXS_BYTES;
    const uint8_t *wrow = W + (size_t)n * row_bytes;
    const int8_t *xrow = xq8 + (size_t)m * K;
    const float *xsrow = xs + (size_t)m * (K / CK);
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *bp = wrow + b * IQ3XXS_BYTES;
        float d = half_to_float_dev(*(const uint16_t *)bp);
        const uint8_t *qs = bp + 2;
        const uint16_t *sas16 = (const uint16_t *)(bp + 66);
        for (int ib = 0; ib < 8; ib++) {
            uint32_t sas_word = (uint32_t)sas16[2*ib] | ((uint32_t)sas16[2*ib+1] << 16);
            float db = d * (0.5f + (float)(sas_word >> 28)) * 0.5f;
            int kbase = b * QK_K + ib * CK;
            int sumi = 0;
            for (int l = 0; l < 4; l++) {
                uint32_t gv1 = c_grid[qs[2*l+0]];
                uint32_t gv2 = c_grid[qs[2*l+1]];
                uint8_t sgn = c_ksigns[(sas_word >> (7*l)) & 127];
                int w0 = 0, w1 = 0, x0, x1;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int wv = (int)(uint8_t)(gv1 >> (8*j)) * ((sgn & (1 << j)) ? -1 : 1);
                    w0 |= (wv & 0xff) << (8 * j);
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int wv = (int)(uint8_t)(gv2 >> (8*j)) * ((sgn & (1 << (j+4))) ? -1 : 1);
                    w1 |= (wv & 0xff) << (8 * j);
                }
                const int8_t *xp = xrow + kbase + l * 8;
                x0 = *(const int *)xp;
                x1 = *(const int *)(xp + 4);
                sumi = __dp4a(w0, x0, sumi);
                sumi = __dp4a(w1, x1, sumi);
            }
            acc += db * xsrow[(kbase) / CK] * (float)sumi;
            qs += 8;
        }
    }
    dst[(size_t)m * N + n] = acc;
}

/* ---------------- PHASE 2: tensor-core MMA kernel ---------------- */
static __device__ __forceinline__ int pack4(const int8_t *p) {
    return (p[0] & 0xff) | ((p[1] & 0xff) << 8) | ((p[2] & 0xff) << 16) | ((p[3] & 0xff) << 24);
}

__global__ void mmq_iq3xxs_mma(float *dst, const uint8_t *W, const int8_t *xq8,
                                const float *xs, int M, int N, int K) {
    int lane = threadIdx.x, gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * 16, m0 = blockIdx.y * 8;
    int nb = K / QK_K, row_bytes = nb * IQ3XXS_BYTES, nsb = K / CK;

    __shared__ int8_t sW[16][CK];
    __shared__ int8_t sX[8][CK];
    __shared__ float  sWs[16];
    __shared__ float  sXs[8];

    float f0 = 0, f1 = 0, f2 = 0, f3 = 0;

    for (int sb = 0; sb < nsb; sb++) {
        for (int i = lane; i < 8 * CK; i += 32) {
            int t = i / CK, kk = i % CK, m = m0 + t;
            sX[t][kk] = (m < M) ? xq8[(size_t)m * K + sb * CK + kk] : 0;
        }
        if (lane < 8) { int m = m0 + lane; sXs[lane] = (m < M) ? xs[(size_t)m * nsb + sb] : 0.0f; }
        if (lane < 16) {
            int r = lane, n = n0 + r;
            const uint8_t *bp = W + (size_t)n * row_bytes + (sb / 8) * IQ3XXS_BYTES;
            float d = half_to_float_dev(*(const uint16_t *)bp);
            const uint8_t *qs = bp + 2;
            const uint16_t *sas16 = (const uint16_t *)(bp + 66);
            int ib = sb & 7;
            uint32_t sas_word = (uint32_t)sas16[2*ib] | ((uint32_t)sas16[2*ib+1] << 16);
            float db = d * (0.5f + (float)(sas_word >> 28)) * 0.5f;
            sWs[r] = db;
            for (int l = 0; l < 4; l++) {
                uint32_t gv1 = c_grid[qs[8*ib + 2*l + 0]];
                uint32_t gv2 = c_grid[qs[8*ib + 2*l + 1]];
                uint8_t sgn = c_ksigns[(sas_word >> (7*l)) & 127];
                for (int j = 0; j < 4; j++) {
                    sW[r][l*8 + j]     = (int8_t)((int)(uint8_t)(gv1 >> (8*j)) * ((sgn & (1 << j))     ? -1 : 1));
                    sW[r][l*8 + j + 4] = (int8_t)((int)(uint8_t)(gv2 >> (8*j)) * ((sgn & (1 << (j+4))) ? -1 : 1));
                }
            }
        }
        __syncwarp();

        int a0 = pack4(&sW[gid][tid*4]), a1 = pack4(&sW[gid+8][tid*4]);
        int a2 = pack4(&sW[gid][tid*4+16]), a3 = pack4(&sW[gid+8][tid*4+16]);
        int b0 = pack4(&sX[gid][tid*4]), b1 = pack4(&sX[gid][tid*4+16]);
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "r"(0), "r"(0), "r"(0), "r"(0));
        float wr0 = sWs[gid], wr8 = sWs[gid+8];
        float xc0 = sXs[tid*2], xc1 = sXs[tid*2+1];
        f0 += wr0 * xc0 * (float)c0; f1 += wr0 * xc1 * (float)c1;
        f2 += wr8 * xc0 * (float)c2; f3 += wr8 * xc1 * (float)c3;
        __syncwarp();
    }
    int n_a = n0 + gid, n_b = n0 + gid + 8;
    int m_a = m0 + tid*2, m_b = m0 + tid*2 + 1;
    if (m_a < M) { dst[(size_t)m_a * N + n_a] = f0; dst[(size_t)m_a * N + n_b] = f2; }
    if (m_b < M) { dst[(size_t)m_b * N + n_a] = f1; dst[(size_t)m_b * N + n_b] = f3; }
}

/* ---------------- PHASE 2.5: multi-warp MMA ---------------- */
#define WN 4
__global__ void mmq_iq3xxs_mma_v2(float *dst, const uint8_t *W, const int8_t *xq8,
                                    const float *xs, int M, int N, int K) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * (16 * WN) + warp * 16;
    int m0 = blockIdx.y * 8;
    int nb = K / QK_K, row_bytes = nb * IQ3XXS_BYTES, nsb = K / CK;

    __shared__ int8_t sX[8][CK];
    __shared__ float  sXs[8];
    __shared__ int8_t sW[16 * WN][CK];
    __shared__ float  sWs[16 * WN];

    float f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    for (int sb = 0; sb < nsb; sb++) {
        for (int i = threadIdx.x; i < 8 * CK; i += blockDim.x) {
            int t = i / CK, kk = i % CK, m = m0 + t;
            sX[t][kk] = (m < M) ? xq8[(size_t)m * K + sb * CK + kk] : 0;
        }
        if (threadIdx.x < 8) { int m = m0 + threadIdx.x; sXs[threadIdx.x] = (m < M) ? xs[(size_t)m * nsb + sb] : 0.0f; }
        if (lane < 16) {
            int r = warp * 16 + lane, n = n0 + lane;
            const uint8_t *bp = W + (size_t)n * row_bytes + (sb / 8) * IQ3XXS_BYTES;
            float d = half_to_float_dev(*(const uint16_t *)bp);
            const uint8_t *qs = bp + 2;
            const uint16_t *sas16 = (const uint16_t *)(bp + 66);
            int ib = sb & 7;
            uint32_t sas_word = (uint32_t)sas16[2*ib] | ((uint32_t)sas16[2*ib+1] << 16);
            sWs[r] = d * (0.5f + (float)(sas_word >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                uint32_t gv1 = c_grid[qs[8*ib + 2*l + 0]];
                uint32_t gv2 = c_grid[qs[8*ib + 2*l + 1]];
                uint8_t sgn = c_ksigns[(sas_word >> (7*l)) & 127];
                for (int j = 0; j < 4; j++) {
                    sW[r][l*8 + j]     = (int8_t)((int)(uint8_t)(gv1 >> (8*j)) * ((sgn & (1 << j))     ? -1 : 1));
                    sW[r][l*8 + j + 4] = (int8_t)((int)(uint8_t)(gv2 >> (8*j)) * ((sgn & (1 << (j+4))) ? -1 : 1));
                }
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

/* ---------------- PHASE 2.6: decode-amortized MMA ---------------- */
#define TG 4
__global__ void mmq_iq3xxs_mma_v3(float *dst, const uint8_t *W, const int8_t *xq8,
                                    const float *xs, int M, int N, int K) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * (16 * WN) + warp * 16;
    int m_base = blockIdx.y * (8 * TG);
    int ntg = (M - m_base + 7) / 8; if (ntg > TG) ntg = TG; if (ntg < 0) ntg = 0;
    int nb = K / QK_K, row_bytes = nb * IQ3XXS_BYTES, nsb = K / CK;

    __shared__ int8_t sX[8 * TG][CK];
    __shared__ float  sXs[8 * TG];
    __shared__ int8_t sW[16 * WN][CK];
    __shared__ float  sWs[16 * WN];

    float f[TG][4];
    for (int g = 0; g < TG; g++) { f[g][0]=f[g][1]=f[g][2]=f[g][3]=0; }

    for (int sb = 0; sb < nsb; sb++) {
        if (lane < 16) {
            int r = warp*16 + lane, n = n0 + lane;
            const uint8_t *bp = W + (size_t)n*row_bytes + (sb/8)*IQ3XXS_BYTES;
            float d = half_to_float_dev(*(const uint16_t*)bp);
            const uint8_t *qs = bp + 2;
            const uint16_t *sas16 = (const uint16_t *)(bp + 66);
            int ib = sb & 7;
            uint32_t sas_word = (uint32_t)sas16[2*ib] | ((uint32_t)sas16[2*ib+1] << 16);
            sWs[r] = d * (0.5f + (float)(sas_word >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                uint32_t gv1 = c_grid[qs[8*ib + 2*l + 0]];
                uint32_t gv2 = c_grid[qs[8*ib + 2*l + 1]];
                uint8_t sgn = c_ksigns[(sas_word >> (7*l)) & 127];
                for (int j = 0; j < 4; j++) {
                    sW[r][l*8 + j]     = (int8_t)((int)(uint8_t)(gv1 >> (8*j)) * ((sgn & (1 << j))     ? -1 : 1));
                    sW[r][l*8 + j + 4] = (int8_t)((int)(uint8_t)(gv2 >> (8*j)) * ((sgn & (1 << (j+4))) ? -1 : 1));
                }
            }
        }
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

/* ---------------- CPU oracle ---------------- */
static void cpu_oracle(const uint8_t *W, const float *X, float *dst, int M, int N, int K) {
    float *wf = (float *)malloc((size_t)K * sizeof(float));
    for (int n = 0; n < N; n++) {
        iq3xxs_decode_row_h(W + (size_t)n * (K / QK_K) * IQ3XXS_BYTES, K, wf);
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
    int M = argc > 1 ? atoi(argv[1]) : 8;     /* tokens */
    int N = argc > 2 ? atoi(argv[2]) : 512;    /* out_dim */
    int K = argc > 3 ? atoi(argv[3]) : 2048;   /* in_dim */
    printf("MMQ IQ3_XXS test: M=%d N=%d K=%d  (K/256=%d blocks/row)\n", M, N, K, K / QK_K);
    if (K % QK_K) { fprintf(stderr, "K must be multiple of 256\n"); return 1; }

    srand(1234);
    int nb = K / QK_K, row_bytes = nb * IQ3XXS_BYTES;
    size_t wbytes = (size_t)N * row_bytes;
    uint8_t *W = (uint8_t *)malloc(wbytes);
    for (size_t i = 0; i < wbytes; i++) W[i] = rand() & 0xff;
    /* Overwrite each block's half scale d with finite realistic values */
    for (int n = 0; n < N; n++)
        for (int b = 0; b < nb; b++) {
            float d = 0.01f + (rand() / (float)RAND_MAX) * 0.05f;
            *(uint16_t *)(W + (size_t)n * row_bytes + b * IQ3XXS_BYTES) = float_to_half_h(d);
        }
    float *X = (float *)malloc((size_t)M * K * sizeof(float));
    for (int i = 0; i < M * K; i++) X[i] = ((rand() / (float)RAND_MAX) - 0.5f) * 2.0f;

    float *ref = (float *)malloc((size_t)M * N * sizeof(float));
    cpu_oracle(W, X, ref, M, N, K);

    /* device */
    CUDA_CHECK(cudaMemcpyToSymbol(c_grid, iq3xxs_grid, sizeof(iq3xxs_grid)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ksigns, ksigns_iq2xs, sizeof(ksigns_iq2xs)));
    uint8_t *dW; int8_t *dXq; float *dX, *dXs, *dDst;
    CUDA_CHECK(cudaMalloc(&dW, wbytes));
    CUDA_CHECK(cudaMalloc(&dX, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dXq, (size_t)M * K));
    CUDA_CHECK(cudaMalloc(&dXs, (size_t)M * (K / CK) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDst, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dW, W, wbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dX, X, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));

    /* P1: dp4a */
    dim3 qg(K / CK, M); quantize_q8_1<<<qg, 32>>>(dX, dXq, dXs, M, K);
    CUDA_CHECK(cudaGetLastError());
    dim3 bg((N + 127) / 128, M); mmq_iq3xxs_naive<<<bg, 128>>>(dDst, dW, dXq, dXs, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float *out = (float *)malloc((size_t)M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(out, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    double e = rel_l2(out, ref, M * N);
    printf("P1 dp4a   vs CPU-F32 oracle: rel_L2 = %.6f\n", e);

    /* P2: tensor-core MMA */
    if (N % 16 != 0) { fprintf(stderr, "P2 needs N %% 16 == 0\n"); return 1; }
    CUDA_CHECK(cudaMemset(dDst, 0, (size_t)M * N * sizeof(float)));
    dim3 mg(N / 16, (M + 7) / 8); mmq_iq3xxs_mma<<<mg, 32>>>(dDst, dW, dXq, dXs, M, N, K);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    float *out2 = (float *)malloc((size_t)M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(out2, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    double e2 = rel_l2(out2, ref, M * N);
    double e2v1 = rel_l2(out2, out, M * N);
    printf("P2 mma    vs CPU-F32 oracle: rel_L2 = %.6f\n", e2);
    printf("P2 mma    vs P1 dp4a       : rel_L2 = %.6f\n", e2v1);
    printf("  ref[0..3]=%.4f %.4f %.4f %.4f\n  mma[0..3]=%.4f %.4f %.4f %.4f\n",
           ref[0], ref[1], ref[2], ref[3], out2[0], out2[1], out2[2], out2[3]);

    /* P2.5: multi-warp MMA */
    double e3 = 1e9, e3v2 = 1e9;
    if (N % (16 * WN) == 0) {
        CUDA_CHECK(cudaMemset(dDst, 0, (size_t)M * N * sizeof(float)));
        dim3 vg(N / (16 * WN), (M + 7) / 8);
        mmq_iq3xxs_mma_v2<<<vg, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *out3 = (float *)malloc((size_t)M * N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(out3, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
        e3 = rel_l2(out3, ref, M * N); e3v2 = rel_l2(out3, out2, M * N);
        printf("P2.5 mma_v2 vs CPU-F32 oracle: rel_L2 = %.6f\n", e3);
        printf("P2.5 mma_v2 vs P2 mma         : rel_L2 = %.6f\n", e3v2);
        free(out3);
    } else printf("P2.5 skipped (N %% %d != 0)\n", 16 * WN);

    /* P2.6: decode-amortized MMA */
    double e4 = 1e9, e4v = 1e9;
    if (N % (16 * WN) == 0) {
        CUDA_CHECK(cudaMemset(dDst, 0, (size_t)M * N * sizeof(float)));
        dim3 v3g(N / (16 * WN), (M + 8*TG - 1) / (8*TG));
        mmq_iq3xxs_mma_v3<<<v3g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *o4 = (float *)malloc((size_t)M * N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(o4, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
        e4 = rel_l2(o4, ref, M*N); e4v = rel_l2(o4, out2, M*N);
        printf("P2.6 mma_v3 vs CPU-F32 oracle: rel_L2 = %.6f\n", e4);
        printf("P2.6 mma_v3 vs P2 mma         : rel_L2 = %.6f\n", e4v);
        free(o4);
    } else printf("P2.6 skipped\n");

    int ok = (e < 0.05) && (e2 < 0.05) && (e2v1 < 1e-4) && (e3 < 0.05) && (e3v2 < 1e-4) && (e4 < 0.05) && (e4v < 1e-4);
    printf("%s\n", ok ? "PASS (P1..P2.6 data path correct)" : "FAIL");

    /* Rough throughput */
    {
        int iters = 300;
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        dim3 bg((N + 127) / 128, M);
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) mmq_iq3xxs_naive<<<bg, 128>>>(dDst, dW, dXq, dXs, M, N, K);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_dp; cudaEventElapsedTime(&ms_dp, t0, t1); ms_dp /= iters;
        dim3 mg(N / 16, (M + 7) / 8);
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) mmq_iq3xxs_mma<<<mg, 32>>>(dDst, dW, dXq, dXs, M, N, K);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_mma; cudaEventElapsedTime(&ms_mma, t0, t1); ms_mma /= iters;
        float ms_v2 = 0;
        if (N % (16 * WN) == 0) {
            dim3 vg(N / (16 * WN), (M + 7) / 8);
            cudaEventRecord(t0);
            for (int i = 0; i < iters; i++) mmq_iq3xxs_mma_v2<<<vg, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms_v2, t0, t1); ms_v2 /= iters;
        }
        double flop = 2.0 * M * N * K;
        printf("\nperf (1 GEMM):\n");
        printf("  dp4a   : %.3f ms  %.1f GFLOP/s\n", ms_dp, flop / (ms_dp * 1e6));
        printf("  mma    : %.3f ms  %.1f GFLOP/s  (%.2fx vs dp4a)\n", ms_mma, flop / (ms_mma * 1e6), ms_dp / ms_mma);
        if (ms_v2 > 0)
            printf("  mma_v2 : %.3f ms  %.1f GFLOP/s  (%.2fx vs dp4a, %.2fx vs mma)\n",
                   ms_v2, flop / (ms_v2 * 1e6), ms_dp / ms_v2, ms_mma / ms_v2);
        float ms_v3 = 0;
        if (N % (16 * WN) == 0) {
            dim3 v3g(N / (16 * WN), (M + 8*TG - 1) / (8*TG));
            cudaEventRecord(t0);
            for (int i = 0; i < iters; i++) mmq_iq3xxs_mma_v3<<<v3g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms_v3, t0, t1); ms_v3 /= iters;
            printf("  mma_v3 : %.3f ms  %.1f GFLOP/s  (%.2fx vs dp4a, %.2fx vs mma)\n",
                   ms_v3, flop / (ms_v3 * 1e6), ms_dp / ms_v3, ms_mma / ms_v3);
        }
    }
    return ok ? 0 : 1;
}
