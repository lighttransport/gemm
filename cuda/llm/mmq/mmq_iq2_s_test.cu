/* Standalone dev harness for the IQ2_S MMQ port (cuda/llm dense v_proj).
 *
 * Build: make -C cuda/llm/mmq mmq_iq2_s_test
 * Run:   ./cuda/llm/mmq/mmq_iq2_s_test
 *
 * IQ2_S block: 256 elements, 82 bytes
 *   d(2) + qs[32]@2 + signs[32]@34 + qh[8]@66 + scales[8]@74
 * Grid: uint64[1024] (8 x uint8 per entry, 10-bit index via qh bits 8-9)
 * Sub-blocks: 8 x 32 elements, split db0 (elems 0-15) / db1 (elems 16-31)
 *   -> 2 MMAs per sub-block (each k=16)
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define QK_K 256
#define IQ2S_BYTES 82
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

/* ---------------- IQ2_S grid table (uint64[1024], 8 uint8 elements each) ---------------- */
static const uint64_t iq2s_grid[1024] = {
#include "iq2s_grid.inc"
};

/* ---------------- host IQ2_S block decode ---------------- */
static void iq2s_decode_block_h(const uint8_t *bp, float *out) {
    float d = half_to_float_h(*(const uint16_t *)bp);
    const uint8_t *qs = bp + 2;
    const uint8_t *signs = bp + 34;
    const uint8_t *qh = bp + 66;
    const uint8_t *scales = bp + 74;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        float db0 = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;
        float db1 = d * (0.5f + (float)(scales[ib32] >>  4)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            float dl = (l < 2) ? db0 : db1;
            int grid_idx = qs[l] | ((qh[ib32] << (8-2*l)) & 0x300);
            uint64_t gv = iq2s_grid[grid_idx];
            uint8_t sgn = signs[l];
            for (int j = 0; j < 8; j++) {
                float w = dl * (float)(uint8_t)(gv >> (8*j)) * ((sgn & (1 << j)) ? -1.0f : 1.0f);
                out[ib32*32 + l*8 + j] = w;
            }
        }
        qs += 4;
        signs += 4;
    }
}

static void iq2s_decode_row_h(const uint8_t *row, int K, float *out) {
    int nb = K / QK_K;
    for (int b = 0; b < nb; b++)
        iq2s_decode_block_h(row + b * IQ2S_BYTES, out + b * QK_K);
}

/* ---------------- device tables ---------------- */
__constant__ uint64_t c_grid[1024];

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
__global__ void mmq_iq2s_naive(float *dst, const uint8_t *W, const int8_t *xq8,
                                  const float *xs, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;
    int nb = K / QK_K, row_bytes = nb * IQ2S_BYTES;
    const uint8_t *wrow = W + (size_t)n * row_bytes;
    const int8_t *xrow = xq8 + (size_t)m * K;
    const float *xsrow = xs + (size_t)m * (K / CK);
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *bp = wrow + b * IQ2S_BYTES;
        float d = half_to_float_dev(*(const uint16_t *)bp);
        const uint8_t *qs = bp + 2;
        const uint8_t *signs = bp + 34;
        const uint8_t *qh = bp + 66;
        const uint8_t *scales = bp + 74;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            float db0 = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;
            float db1 = d * (0.5f + (float)(scales[ib32] >>  4)) * 0.25f;
            int kbase = b * QK_K + ib32 * CK;
            int sumi0 = 0, sumi1 = 0;
            for (int l = 0; l < 4; l++) {
                int grid_idx = qs[l] | ((qh[ib32] << (8-2*l)) & 0x300);
                uint64_t gv = c_grid[grid_idx];
                uint8_t sgn = signs[l];
                int wlo = 0, whi = 0;
                for (int j = 0; j < 4; j++) {
                    int wv = (int)(uint8_t)(gv >> (8*j)) * ((sgn & (1 << j)) ? -1 : 1);
                    wlo |= (wv & 0xff) << (8 * j);
                }
                for (int j = 0; j < 4; j++) {
                    int wv = (int)(uint8_t)(gv >> (8*(j+4))) * ((sgn & (1 << (j+4))) ? -1 : 1);
                    whi |= (wv & 0xff) << (8 * j);
                }
                const int8_t *xp = xrow + kbase + l * 8;
                int xp0 = *(const int *)xp;
                int xp1 = *(const int *)(xp + 4);
                if (l < 2) {
                    sumi0 = __dp4a(wlo, xp0, sumi0);
                    sumi0 = __dp4a(whi, xp1, sumi0);
                } else {
                    sumi1 = __dp4a(wlo, xp0, sumi1);
                    sumi1 = __dp4a(whi, xp1, sumi1);
                }
            }
            float d8 = xsrow[(b * QK_K + ib32 * CK) / CK];
            acc += db0 * d8 * (float)sumi0 + db1 * d8 * (float)sumi1;
            qs += 4; signs += 4;
        }
    }
    dst[(size_t)m * N + n] = acc;
}

/* ---------------- PHASE 2: tensor-core MMA kernel (2 MMAs per sub-block, k=16 each) ---------------- */
static __device__ __forceinline__ int pack4(const int8_t *p) {
    return (p[0] & 0xff) | ((p[1] & 0xff) << 8) | ((p[2] & 0xff) << 16) | ((p[3] & 0xff) << 24);
}

/* Use m16n8k16 for each k=16 half. */
__global__ void mmq_iq2s_mma(float *dst, const uint8_t *W, const int8_t *xq8,
                                const float *xs, int M, int N, int K) {
    int lane = threadIdx.x, gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * 16, m0 = blockIdx.y * 8;
    int nb = K / QK_K, row_bytes = nb * IQ2S_BYTES, nsb = K / CK;

    __shared__ int8_t sW[16][CK];
    __shared__ int8_t sX[8][CK];
    __shared__ float  sWs_db0[16];
    __shared__ float  sWs_db1[16];
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
            const uint8_t *bp = W + (size_t)n * row_bytes + (sb / 8) * IQ2S_BYTES;
            float d = half_to_float_dev(*(const uint16_t *)bp);
            const uint8_t *qs = bp + 2;
            const uint8_t *signs = bp + 34;
            const uint8_t *qh = bp + 66;
            const uint8_t *scales = bp + 74;
            int ib32 = sb & 7;
            float db0 = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;
            float db1 = d * (0.5f + (float)(scales[ib32] >>  4)) * 0.25f;
            sWs_db0[r] = db0;
            sWs_db1[r] = db1;
            int qs_off = ib32 * 4, sign_off = ib32 * 4;
            uint8_t qh_byte = qh[ib32];
            /* Decode all 4 l-values into sW[r][0..31] */
            for (int l = 0; l < 4; l++) {
                int grid_idx = qs[qs_off + l] | ((qh_byte << (8-2*l)) & 0x300);
                uint64_t gv = c_grid[grid_idx];
                uint8_t sgn = signs[sign_off + l];
                for (int j = 0; j < 8; j++) {
                    int wv = (int)(uint8_t)(gv >> (8*j)) * ((sgn & (1 << j)) ? -1 : 1);
                    sW[r][l*8 + j] = (int8_t)wv;
                }
            }
        }
        __syncwarp();

        /* MMA_0: k=0..15 (half 0) */
        {
        int a0 = pack4(&sW[gid][tid*4]);
        int a1 = pack4(&sW[gid+8][tid*4]);
        int b0 = pack4(&sX[gid][tid*4]);
        int c0=0,c1=0,c2=0,c3=0;
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
            : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
            : "r"(a0),"r"(a1),"r"(b0),"r"(0),"r"(0),"r"(0),"r"(0));
        float db00 = sWs_db0[gid], db08 = sWs_db0[gid+8];
        float xc0 = sXs[tid*2], xc1 = sXs[tid*2+1];
        f0 += db00 * xc0 * (float)c0; f1 += db00 * xc1 * (float)c1;
        f2 += db08 * xc0 * (float)c2; f3 += db08 * xc1 * (float)c3;
        }

        /* MMA_1: k=16..31 (half 1) */
        {
        int a0 = pack4(&sW[gid][16+tid*4]);
        int a1 = pack4(&sW[gid+8][16+tid*4]);
        int b0 = pack4(&sX[gid][16+tid*4]);
        int c0=0,c1=0,c2=0,c3=0;
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
            : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
            : "r"(a0),"r"(a1),"r"(b0),"r"(0),"r"(0),"r"(0),"r"(0));
        float db10 = sWs_db1[gid], db18 = sWs_db1[gid+8];
        float xc0 = sXs[tid*2], xc1 = sXs[tid*2+1];
        f0 += db10 * xc0 * (float)c0; f1 += db10 * xc1 * (float)c1;
        f2 += db18 * xc0 * (float)c2; f3 += db18 * xc1 * (float)c3;
        }
        __syncwarp();
    }
    int n_a = n0 + gid, n_b = n0 + gid + 8;
    int m_a = m0 + tid*2, m_b = m0 + tid*2 + 1;
    if (m_a < M) { dst[(size_t)m_a * N + n_a] = f0; dst[(size_t)m_a * N + n_b] = f2; }
    if (m_b < M) { dst[(size_t)m_b * N + n_a] = f1; dst[(size_t)m_b * N + n_b] = f3; }
}

/* ---------------- PHASE 2.5: multi-warp MMA ---------------- */
#define WN 4
__global__ void mmq_iq2s_mma_v2(float *dst, const uint8_t *W, const int8_t *xq8,
                                   const float *xs, int M, int N, int K) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * (16 * WN) + warp * 16;
    int m0 = blockIdx.y * 8;
    int nb = K / QK_K, row_bytes = nb * IQ2S_BYTES, nsb = K / CK;

    __shared__ int8_t sX[8][CK];
    __shared__ float  sXs[8];
    __shared__ int8_t sW[16 * WN][CK];
    __shared__ float  sWs_db0[16 * WN];
    __shared__ float  sWs_db1[16 * WN];

    float f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    for (int sb = 0; sb < nsb; sb++) {
        for (int i = threadIdx.x; i < 8 * CK; i += blockDim.x) {
            int t = i / CK, kk = i % CK, m = m0 + t;
            sX[t][kk] = (m < M) ? xq8[(size_t)m * K + sb * CK + kk] : 0;
        }
        if (threadIdx.x < 8) { int m = m0 + threadIdx.x; sXs[threadIdx.x] = (m < M) ? xs[(size_t)m * nsb + sb] : 0.0f; }
        if (lane < 16) {
            int r = warp * 16 + lane, n = n0 + lane;
            const uint8_t *bp = W + (size_t)n * row_bytes + (sb / 8) * IQ2S_BYTES;
            float d = half_to_float_dev(*(const uint16_t *)bp);
            const uint8_t *qs = bp + 2;
            const uint8_t *signs = bp + 34;
            const uint8_t *qh = bp + 66;
            const uint8_t *scales = bp + 74;
            int ib32 = sb & 7;
            sWs_db0[r] = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;
            sWs_db1[r] = d * (0.5f + (float)(scales[ib32] >>  4)) * 0.25f;
            int qs_off = ib32 * 4, sign_off = ib32 * 4;
            uint8_t qh_byte = qh[ib32];
            for (int l = 0; l < 4; l++) {
                int grid_idx = qs[qs_off + l] | ((qh_byte << (8-2*l)) & 0x300);
                uint64_t gv = c_grid[grid_idx];
                uint8_t sgn = signs[sign_off + l];
                for (int j = 0; j < 8; j++) {
                    int wv = (int)(uint8_t)(gv >> (8*j)) * ((sgn & (1 << j)) ? -1 : 1);
                    sW[r][l*8 + j] = (int8_t)wv;
                }
            }
        }
        __syncthreads();
        int wr = warp * 16;

        /* MMA_0: k=0..15 */
        {
        int a0 = pack4(&sW[wr+gid][tid*4]);
        int a1 = pack4(&sW[wr+gid+8][tid*4]);
        int b0 = pack4(&sX[gid][tid*4]);
        int c0=0,c1=0,c2=0,c3=0;
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
            : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
            : "r"(a0),"r"(a1),"r"(b0),"r"(0),"r"(0),"r"(0),"r"(0));
        float db00 = sWs_db0[wr+gid], db08 = sWs_db0[wr+gid+8];
        float xc0 = sXs[tid*2], xc1 = sXs[tid*2+1];
        f0 += db00 * xc0 * (float)c0; f1 += db00 * xc1 * (float)c1;
        f2 += db08 * xc0 * (float)c2; f3 += db08 * xc1 * (float)c3;
        }
        /* MMA_1: k=16..31 */
        {
        int a0 = pack4(&sW[wr+gid][16+tid*4]);
        int a1 = pack4(&sW[wr+gid+8][16+tid*4]);
        int b0 = pack4(&sX[gid][16+tid*4]);
        int c0=0,c1=0,c2=0,c3=0;
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
            : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
            : "r"(a0),"r"(a1),"r"(b0),"r"(0),"r"(0),"r"(0),"r"(0));
        float db10 = sWs_db1[wr+gid], db18 = sWs_db1[wr+gid+8];
        float xc0 = sXs[tid*2], xc1 = sXs[tid*2+1];
        f0 += db10 * xc0 * (float)c0; f1 += db10 * xc1 * (float)c1;
        f2 += db18 * xc0 * (float)c2; f3 += db18 * xc1 * (float)c3;
        }
        __syncthreads();
    }
    int n_a = n0 + gid, n_b = n0 + gid + 8;
    int m_a = m0 + tid*2, m_b = m0 + tid*2 + 1;
    if (m_a < M) { dst[(size_t)m_a * N + n_a] = f0; dst[(size_t)m_a * N + n_b] = f2; }
    if (m_b < M) { dst[(size_t)m_b * N + n_a] = f1; dst[(size_t)m_b * N + n_b] = f3; }
}

/* ---------------- PHASE 2.6: decode-amortized MMA ---------------- */
#define TG 4
__global__ void mmq_iq2s_mma_v3(float *dst, const uint8_t *W, const int8_t *xq8,
                                   const float *xs, int M, int N, int K) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * (16 * WN) + warp * 16;
    int m_base = blockIdx.y * (8 * TG);
    int ntg = (M - m_base + 7) / 8; if (ntg > TG) ntg = TG; if (ntg < 0) ntg = 0;
    int nb = K / QK_K, row_bytes = nb * IQ2S_BYTES, nsb = K / CK;

    __shared__ int8_t sX[8 * TG][CK];
    __shared__ float  sXs[8 * TG];
    __shared__ int8_t sW[16 * WN][CK];
    __shared__ float  sWs_db0[16 * WN];
    __shared__ float  sWs_db1[16 * WN];

    float f[TG][4];
    for (int g = 0; g < TG; g++) { f[g][0]=f[g][1]=f[g][2]=f[g][3]=0; }

    for (int sb = 0; sb < nsb; sb++) {
        if (lane < 16) {
            int r = warp*16 + lane, n = n0 + lane;
            const uint8_t *bp = W + (size_t)n*row_bytes + (sb/8)*IQ2S_BYTES;
            float d = half_to_float_dev(*(const uint16_t*)bp);
            const uint8_t *qs = bp + 2;
            const uint8_t *signs = bp + 34;
            const uint8_t *qh = bp + 66;
            const uint8_t *scales = bp + 74;
            int ib32 = sb & 7;
            sWs_db0[r] = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;
            sWs_db1[r] = d * (0.5f + (float)(scales[ib32] >>  4)) * 0.25f;
            int qs_off = ib32 * 4, sign_off = ib32 * 4;
            uint8_t qh_byte = qh[ib32];
            for (int l = 0; l < 4; l++) {
                int grid_idx = qs[qs_off + l] | ((qh_byte << (8-2*l)) & 0x300);
                uint64_t gv = c_grid[grid_idx];
                uint8_t sgn = signs[sign_off + l];
                for (int j = 0; j < 8; j++) {
                    int wv = (int)(uint8_t)(gv >> (8*j)) * ((sgn & (1 << j)) ? -1 : 1);
                    sW[r][l*8 + j] = (int8_t)wv;
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

        /* MMA_0: k=0..15 */
        int a0_h0 = pack4(&sW[wr+gid][tid*4]);
        int a1_h0 = pack4(&sW[wr+gid+8][tid*4]);
        int b0_h0 = pack4(&sX[gid][tid*4]);
        float wr0_db0 = sWs_db0[wr+gid], wr8_db0 = sWs_db0[wr+gid+8];

        /* MMA_1: k=16..31 */
        int a0_h1 = pack4(&sW[wr+gid][16+tid*4]);
        int a1_h1 = pack4(&sW[wr+gid+8][16+tid*4]);
        int b0_h1 = pack4(&sX[gid][16+tid*4]);
        float wr0_db1 = sWs_db1[wr+gid], wr8_db1 = sWs_db1[wr+gid+8];

        for (int g = 0; g < ntg; g++) {
            int b0_x = pack4(&sX[g*8+gid][tid*4]);
            int b1_x = pack4(&sX[g*8+gid][tid*4+16]);

            /* MMA_0 */
            {
            int c0=0,c1=0,c2=0,c3=0;
            asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
                :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                :"r"(a0_h0),"r"(a1_h0),"r"(b0_h0),"r"(0),"r"(0),"r"(0),"r"(0));
            float xc0 = sXs[g*8+tid*2], xc1 = sXs[g*8+tid*2+1];
            f[g][0]+=wr0_db0*xc0*(float)c0; f[g][1]+=wr0_db0*xc1*(float)c1;
            f[g][2]+=wr8_db0*xc0*(float)c2; f[g][3]+=wr8_db0*xc1*(float)c3;
            }

            /* MMA_1 with k=16..31 requires different activation packing */
            {
            int c0=0,c1=0,c2=0,c3=0;
            asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
                :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                :"r"(a0_h1),"r"(a1_h1),"r"(b1_x),"r"(0),"r"(0),"r"(0),"r"(0));
            float xc0 = sXs[g*8+tid*2], xc1 = sXs[g*8+tid*2+1];
            f[g][0]+=wr0_db1*xc0*(float)c0; f[g][1]+=wr0_db1*xc1*(float)c1;
            f[g][2]+=wr8_db1*xc0*(float)c2; f[g][3]+=wr8_db1*xc1*(float)c3;
            }
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
        iq2s_decode_row_h(W + (size_t)n * (K / QK_K) * IQ2S_BYTES, K, wf);
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
    int M = argc > 1 ? atoi(argv[1]) : 8;
    int N = argc > 2 ? atoi(argv[2]) : 512;
    int K = argc > 3 ? atoi(argv[3]) : 2048;
    printf("MMQ IQ2_S test: M=%d N=%d K=%d  (K/256=%d blocks/row)\n", M, N, K, K / QK_K);
    if (K % QK_K) { fprintf(stderr, "K must be multiple of 256\n"); return 1; }

    srand(1234);
    int nb = K / QK_K, row_bytes = nb * IQ2S_BYTES;
    size_t wbytes = (size_t)N * row_bytes;
    uint8_t *W = (uint8_t *)malloc(wbytes);
    for (size_t i = 0; i < wbytes; i++) W[i] = rand() & 0xff;
    for (int n = 0; n < N; n++)
        for (int b = 0; b < nb; b++) {
            float d = 0.01f + (rand() / (float)RAND_MAX) * 0.05f;
            *(uint16_t *)(W + (size_t)n * row_bytes + b * IQ2S_BYTES) = float_to_half_h(d);
        }
    float *X = (float *)malloc((size_t)M * K * sizeof(float));
    for (int i = 0; i < M * K; i++) X[i] = ((rand() / (float)RAND_MAX) - 0.5f) * 2.0f;

    float *ref = (float *)malloc((size_t)M * N * sizeof(float));
    cpu_oracle(W, X, ref, M, N, K);

    uint8_t *dW; int8_t *dXq; float *dX, *dXs, *dDst;
    CUDA_CHECK(cudaMalloc(&dW, wbytes));
    CUDA_CHECK(cudaMalloc(&dX, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dXq, (size_t)M * K));
    CUDA_CHECK(cudaMalloc(&dXs, (size_t)M * (K / CK) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDst, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dW, W, wbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dX, X, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_grid, iq2s_grid, sizeof(iq2s_grid)));

    /* P1: dp4a */
    dim3 qg(K / CK, M); quantize_q8_1<<<qg, 32>>>(dX, dXq, dXs, M, K);
    CUDA_CHECK(cudaGetLastError());
    dim3 bg((N + 127) / 128, M); mmq_iq2s_naive<<<bg, 128>>>(dDst, dW, dXq, dXs, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float *out = (float *)malloc((size_t)M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(out, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    double e = rel_l2(out, ref, M * N);
    printf("P1 dp4a   vs CPU-F32 oracle: rel_L2 = %.6f\n", e);

    /* P2: tensor-core MMA */
    if (N % 16 != 0) { fprintf(stderr, "P2 needs N %% 16 == 0\n"); return 1; }
    CUDA_CHECK(cudaMemset(dDst, 0, (size_t)M * N * sizeof(float)));
    dim3 mg(N / 16, (M + 7) / 8); mmq_iq2s_mma<<<mg, 32>>>(dDst, dW, dXq, dXs, M, N, K);
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
        mmq_iq2s_mma_v2<<<vg, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
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
        mmq_iq2s_mma_v3<<<v3g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
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

    {
        int iters = 300;
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        dim3 bg((N + 127) / 128, M);
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) mmq_iq2s_naive<<<bg, 128>>>(dDst, dW, dXq, dXs, M, N, K);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_dp; cudaEventElapsedTime(&ms_dp, t0, t1); ms_dp /= iters;
        dim3 mg(N / 16, (M + 7) / 8);
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) mmq_iq2s_mma<<<mg, 32>>>(dDst, dW, dXq, dXs, M, N, K);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_mma; cudaEventElapsedTime(&ms_mma, t0, t1); ms_mma /= iters;
        float ms_v2 = 0;
        if (N % (16 * WN) == 0) {
            dim3 vg(N / (16 * WN), (M + 7) / 8);
            cudaEventRecord(t0);
            for (int i = 0; i < iters; i++) mmq_iq2s_mma_v2<<<vg, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
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
            for (int i = 0; i < iters; i++) mmq_iq2s_mma_v3<<<v3g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms_v3, t0, t1); ms_v3 /= iters;
            printf("  mma_v3 : %.3f ms  %.1f GFLOP/s  (%.2fx vs dp4a, %.2fx vs mma)\n",
                   ms_v3, flop / (ms_v3 * 1e6), ms_dp / ms_v3, ms_mma / ms_v3);
        }
    }
    return ok ? 0 : 1;
}
