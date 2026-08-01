/* Standalone dev harness for the IQ3_S MMQ port (cuda/llm dense v_proj).
 *
 * Build: make -C cuda/llm/mmq mmq_iq3_s_test
 * Run:   ./cuda/llm/mmq/mmq_iq3_s_test
 *
 * IQ3_S block: 256 elements, 110 bytes
 *   d(2) + qs(64) + qh(8) + signs(32) + scales(4)
 * Grid: uint32[512] (4 x uint8 per entry, 9-bit index via qh bit 8)
 * Sub-blocks: 8 x 32 elements, 1 MMA per sub-block (k=32)
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define QK_K 256
#define IQ3S_BYTES 110
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

/* ---------------- IQ3_S grid table (uint32[512], 4 uint8 elements each) ---------------- */
static const uint32_t iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

/* ---------------- host IQ3_S block decode ---------------- */
static void iq3s_decode_block_h(const uint8_t *bp, float *out) {
    float d = half_to_float_h(*(const uint16_t *)bp);
    const uint8_t *qs = bp + 2;
    const uint8_t *qh = bp + 66;
    const uint8_t *signs = bp + 74;
    const uint8_t *scales = bp + 106;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        float db1 = d * (1 + 2*(scales[ib32/2] & 0xf));
        float db2 = d * (1 + 2*(scales[ib32/2] >>  4));
        for (int l = 0; l < 4; l++) {
            int g1_idx = qs[2*l+0] | ((qh[0] << (8-2*l)) & 256);
            int g2_idx = qs[2*l+1] | ((qh[0] << (7-2*l)) & 256);
            uint32_t g1 = iq3s_grid[g1_idx];
            uint32_t g2 = iq3s_grid[g2_idx];
            uint8_t sgn = signs[l];
            for (int j = 0; j < 4; j++) {
                float w0 = db1 * (float)(uint8_t)(g1 >> (8*j)) * ((sgn & (1 << j))     ? -1.0f : 1.0f);
                float w1 = db1 * (float)(uint8_t)(g2 >> (8*j)) * ((sgn & (1 << (j+4))) ? -1.0f : 1.0f);
                out[ib32*32 + l*8 + j + 0] = w0;
                out[ib32*32 + l*8 + j + 4] = w1;
            }
        }
        qs += 8; signs += 4;
        for (int l = 0; l < 4; l++) {
            int g1_idx = qs[2*l+0] | ((qh[1] << (8-2*l)) & 256);
            int g2_idx = qs[2*l+1] | ((qh[1] << (7-2*l)) & 256);
            uint32_t g1 = iq3s_grid[g1_idx];
            uint32_t g2 = iq3s_grid[g2_idx];
            uint8_t sgn = signs[l];
            for (int j = 0; j < 4; j++) {
                float w0 = db2 * (float)(uint8_t)(g1 >> (8*j)) * ((sgn & (1 << j))     ? -1.0f : 1.0f);
                float w1 = db2 * (float)(uint8_t)(g2 >> (8*j)) * ((sgn & (1 << (j+4))) ? -1.0f : 1.0f);
                out[(ib32+1)*32 + l*8 + j + 0] = w0;
                out[(ib32+1)*32 + l*8 + j + 4] = w1;
            }
        }
        qh += 2; qs += 8; signs += 4;
    }
}

static void iq3s_decode_row_h(const uint8_t *row, int K, float *out) {
    int nb = K / QK_K;
    for (int b = 0; b < nb; b++)
        iq3s_decode_block_h(row + b * IQ3S_BYTES, out + b * QK_K);
}

/* ---------------- device tables ---------------- */
__constant__ uint32_t c_grid[512];

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
__global__ void mmq_iq3s_naive(float *dst, const uint8_t *W, const int8_t *xq8,
                                  const float *xs, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;
    int nb = K / QK_K, row_bytes = nb * IQ3S_BYTES;
    const uint8_t *wrow = W + (size_t)n * row_bytes;
    const int8_t *xrow = xq8 + (size_t)m * K;
    const float *xsrow = xs + (size_t)m * (K / CK);
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *bp = wrow + b * IQ3S_BYTES;
        float d = half_to_float_dev(*(const uint16_t *)bp);
        const uint8_t *qs = bp + 2;
        const uint8_t *qh = bp + 66;
        const uint8_t *signs = bp + 74;
        const uint8_t *scales = bp + 106;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            int pair = ib32 / 2, sub_in_pair = ib32 & 1;
            float db = d * (float)(1 + 2 * (sub_in_pair ? (scales[pair] >> 4) : (scales[pair] & 0xf)));
            int qs_off = ib32 * 8, sign_off = ib32 * 4;
            int kbase = b * QK_K + ib32 * CK;
            int sumi = 0;
            for (int l = 0; l < 4; l++) {
                int g1_idx = qs[qs_off + 2*l + 0] | ((qh[ib32] << (8-2*l)) & 256);
                int g2_idx = qs[qs_off + 2*l + 1] | ((qh[ib32] << (7-2*l)) & 256);
                uint32_t gv1 = c_grid[g1_idx];
                uint32_t gv2 = c_grid[g2_idx];
                uint8_t sgn = signs[sign_off + l];
                int w0 = 0, w1 = 0;
                for (int j = 0; j < 4; j++) {
                    int wv = (int)(uint8_t)(gv1 >> (8*j)) * ((sgn & (1 << j))     ? -1 : 1);
                    w0 |= (wv & 0xff) << (8 * j);
                }
                for (int j = 0; j < 4; j++) {
                    int wv = (int)(uint8_t)(gv2 >> (8*j)) * ((sgn & (1 << (j+4))) ? -1 : 1);
                    w1 |= (wv & 0xff) << (8 * j);
                }
                const int8_t *xp = xrow + kbase + l * 8;
                int x0 = *(const int *)xp;
                int x1 = *(const int *)(xp + 4);
                sumi = __dp4a(w0, x0, sumi);
                sumi = __dp4a(w1, x1, sumi);
            }
            acc += db * xsrow[(b * QK_K + ib32 * CK) / CK] * (float)sumi;
        }
    }
    dst[(size_t)m * N + n] = acc;
}

/* ---------------- PHASE 2: tensor-core MMA kernel ---------------- */
static __device__ __forceinline__ int pack4(const int8_t *p) {
    return (p[0] & 0xff) | ((p[1] & 0xff) << 8) | ((p[2] & 0xff) << 16) | ((p[3] & 0xff) << 24);
}

__global__ void mmq_iq3s_mma(float *dst, const uint8_t *W, const int8_t *xq8,
                                const float *xs, int M, int N, int K) {
    int lane = threadIdx.x, gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * 16, m0 = blockIdx.y * 8;
    int nb = K / QK_K, row_bytes = nb * IQ3S_BYTES, nsb = K / CK;

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
            const uint8_t *bp = W + (size_t)n * row_bytes + (sb / 8) * IQ3S_BYTES;
            float d = half_to_float_dev(*(const uint16_t *)bp);
            const uint8_t *qs = bp + 2;
            const uint8_t *qh = bp + 66;
            const uint8_t *signs_base = bp + 74;
            const uint8_t *scales = bp + 106;
            int ib32 = sb & 7;
            int pair = ib32 / 2, sub_in_pair = ib32 & 1;
            float db = d * (float)(1 + 2 * (sub_in_pair ? (scales[pair] >> 4) : (scales[pair] & 0xf)));
            sWs[r] = db;
            int qs_off = ib32 * 8, sign_off = ib32 * 4;
            uint8_t qh_byte = qh[ib32];
            for (int l = 0; l < 4; l++) {
                int g1_idx = qs[qs_off + 2*l + 0] | ((qh_byte << (8-2*l)) & 256);
                int g2_idx = qs[qs_off + 2*l + 1] | ((qh_byte << (7-2*l)) & 256);
                uint32_t gv1 = c_grid[g1_idx];
                uint32_t gv2 = c_grid[g2_idx];
                uint8_t sgn = signs_base[sign_off + l];
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
__global__ void mmq_iq3s_mma_v2(float *dst, const uint8_t *W, const int8_t *xq8,
                                   const float *xs, int M, int N, int K) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * (16 * WN) + warp * 16;
    int m0 = blockIdx.y * 8;
    int nb = K / QK_K, row_bytes = nb * IQ3S_BYTES, nsb = K / CK;

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
            const uint8_t *bp = W + (size_t)n * row_bytes + (sb / 8) * IQ3S_BYTES;
            float d = half_to_float_dev(*(const uint16_t *)bp);
            const uint8_t *qs = bp + 2;
            const uint8_t *qh = bp + 66;
            const uint8_t *signs_base = bp + 74;
            const uint8_t *scales = bp + 106;
            int ib32 = sb & 7;
            int pair = ib32 / 2, sub_in_pair = ib32 & 1;
            sWs[r] = d * (float)(1 + 2 * (sub_in_pair ? (scales[pair] >> 4) : (scales[pair] & 0xf)));
            int qs_off = ib32 * 8, sign_off = ib32 * 4;
            uint8_t qh_byte = qh[ib32];
            for (int l = 0; l < 4; l++) {
                int g1_idx = qs[qs_off + 2*l + 0] | ((qh_byte << (8-2*l)) & 256);
                int g2_idx = qs[qs_off + 2*l + 1] | ((qh_byte << (7-2*l)) & 256);
                uint32_t gv1 = c_grid[g1_idx];
                uint32_t gv2 = c_grid[g2_idx];
                uint8_t sgn = signs_base[sign_off + l];
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
__global__ void mmq_iq3s_mma_v3(float *dst, const uint8_t *W, const int8_t *xq8,
                                   const float *xs, int M, int N, int K) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * (16 * WN) + warp * 16;
    int m_base = blockIdx.y * (8 * TG);
    int ntg = (M - m_base + 7) / 8; if (ntg > TG) ntg = TG; if (ntg < 0) ntg = 0;
    int nb = K / QK_K, row_bytes = nb * IQ3S_BYTES, nsb = K / CK;

    __shared__ int8_t sX[8 * TG][CK];
    __shared__ float  sXs[8 * TG];
    __shared__ int8_t sW[16 * WN][CK];
    __shared__ float  sWs[16 * WN];

    float f[TG][4];
    for (int g = 0; g < TG; g++) { f[g][0]=f[g][1]=f[g][2]=f[g][3]=0; }

    for (int sb = 0; sb < nsb; sb++) {
        if (lane < 16) {
            int r = warp*16 + lane, n = n0 + lane;
            const uint8_t *bp = W + (size_t)n*row_bytes + (sb/8)*IQ3S_BYTES;
            float d = half_to_float_dev(*(const uint16_t*)bp);
            const uint8_t *qs = bp + 2;
            const uint8_t *qh = bp + 66;
            const uint8_t *signs_base = bp + 74;
            const uint8_t *scales = bp + 106;
            int ib32 = sb & 7;
            int pair = ib32 / 2, sub_in_pair = ib32 & 1;
            sWs[r] = d * (float)(1 + 2 * (sub_in_pair ? (scales[pair] >> 4) : (scales[pair] & 0xf)));
            int qs_off = ib32 * 8, sign_off = ib32 * 4;
            uint8_t qh_byte = qh[ib32];
            for (int l = 0; l < 4; l++) {
                int g1_idx = qs[qs_off + 2*l + 0] | ((qh_byte << (8-2*l)) & 256);
                int g2_idx = qs[qs_off + 2*l + 1] | ((qh_byte << (7-2*l)) & 256);
                uint32_t gv1 = c_grid[g1_idx];
                uint32_t gv2 = c_grid[g2_idx];
                uint8_t sgn = signs_base[sign_off + l];
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
        iq3s_decode_row_h(W + (size_t)n * (K / QK_K) * IQ3S_BYTES, K, wf);
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
    printf("MMQ IQ3_S test: M=%d N=%d K=%d  (K/256=%d blocks/row)\n", M, N, K, K / QK_K);
    if (K % QK_K) { fprintf(stderr, "K must be multiple of 256\n"); return 1; }

    srand(1234);
    int nb = K / QK_K, row_bytes = nb * IQ3S_BYTES;
    size_t wbytes = (size_t)N * row_bytes;
    uint8_t *W = (uint8_t *)malloc(wbytes);
    for (size_t i = 0; i < wbytes; i++) W[i] = rand() & 0xff;
    for (int n = 0; n < N; n++)
        for (int b = 0; b < nb; b++) {
            float d = 0.01f + (rand() / (float)RAND_MAX) * 0.05f;
            *(uint16_t *)(W + (size_t)n * row_bytes + b * IQ3S_BYTES) = float_to_half_h(d);
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
    CUDA_CHECK(cudaMemcpyToSymbol(c_grid, iq3s_grid, sizeof(iq3s_grid)));

    /* P1: dp4a */
    dim3 qg(K / CK, M); quantize_q8_1<<<qg, 32>>>(dX, dXq, dXs, M, K);
    CUDA_CHECK(cudaGetLastError());
    dim3 bg((N + 127) / 128, M); mmq_iq3s_naive<<<bg, 128>>>(dDst, dW, dXq, dXs, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float *out = (float *)malloc((size_t)M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(out, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    double e = rel_l2(out, ref, M * N);
    printf("P1 dp4a   vs CPU-F32 oracle: rel_L2 = %.6f\n", e);

    /* P2: tensor-core MMA */
    if (N % 16 != 0) { fprintf(stderr, "P2 needs N %% 16 == 0\n"); return 1; }
    CUDA_CHECK(cudaMemset(dDst, 0, (size_t)M * N * sizeof(float)));
    dim3 mg(N / 16, (M + 7) / 8); mmq_iq3s_mma<<<mg, 32>>>(dDst, dW, dXq, dXs, M, N, K);
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
        mmq_iq3s_mma_v2<<<vg, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
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
        mmq_iq3s_mma_v3<<<v3g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
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
        for (int i = 0; i < iters; i++) mmq_iq3s_naive<<<bg, 128>>>(dDst, dW, dXq, dXs, M, N, K);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_dp; cudaEventElapsedTime(&ms_dp, t0, t1); ms_dp /= iters;
        dim3 mg(N / 16, (M + 7) / 8);
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) mmq_iq3s_mma<<<mg, 32>>>(dDst, dW, dXq, dXs, M, N, K);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_mma; cudaEventElapsedTime(&ms_mma, t0, t1); ms_mma /= iters;
        float ms_v2 = 0;
        if (N % (16 * WN) == 0) {
            dim3 vg(N / (16 * WN), (M + 7) / 8);
            cudaEventRecord(t0);
            for (int i = 0; i < iters; i++) mmq_iq3s_mma_v2<<<vg, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
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
            for (int i = 0; i < iters; i++) mmq_iq3s_mma_v3<<<v3g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms_v3, t0, t1); ms_v3 /= iters;
            printf("  mma_v3 : %.3f ms  %.1f GFLOP/s  (%.2fx vs dp4a, %.2fx vs mma)\n",
                   ms_v3, flop / (ms_v3 * 1e6), ms_dp / ms_v3, ms_mma / ms_v3);
        }
    }
    return ok ? 0 : 1;
}
