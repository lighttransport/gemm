/* Standalone dev harness for the Q4_K MMQ port (cuda/llm dense FFN/QKV).
 *
 * Goal: replace the "dequant Q4_K -> F16 weights + cuBLAS GEMM" prefill path
 * (62% of Gemma4-12B prefill, materializes ~113 MB F16/matvec) with a single
 * quantized matmul consuming Q4_K weights directly + int8 q8_1 activations.
 *
 * Q4_K block layout (144 bytes, 256 elements, 8 sub-blocks of 32):
 *   d/half     offset 0 : super-block scale
 *   dmin/half  offset 2 : super-block minimum scale
 *   sc[12]     offset 4 : 6-bit sub-scales (sv) + 6-bit sub-mins (mv), packed
 *   qs[128]    offset 16: 256x4-bit values packed 2/byte
 *   sub-block ib (0..7): chunk = ib/2, nibble = (ib&1)?hi:lo of qs[(ib/2)*32 + k]
 *   6-bit scale sv[ib]/min mv[ib]:
 *     ib<4 : sv = sc[ib]&0x3F,  mv = sc[ib+4]&0x3F
 *     ib>=4: sv = (sc[8+ib-4]&0xF)|(((sc[ib-4]>>6)&3)<<4)
 *            mv = (sc[8+ib-4]>>4)|(((sc[ib]  >>6)&3)<<4)
 *   weight[k] = d*sv*nibble[k] - dmin*mv
 *
 * KEY DIFFERENCE vs Q2_K MMQ: sv (0..63) * nibble (0..15) = up to 945 does NOT
 * fit int8, so sv is NOT folded into the int8 weight (Q2_K folds dl<=15*raw<=3).
 * Instead sW_plus = raw nibble (0..15), and d*sv / dmin*mv are applied as floats
 * after the MMA (the m16n8k32 MMA spans exactly one 32-element sub-block, so the
 * per-sub-block sv/mv scale applies cleanly). sW_minus = 1 -> MMA gives sum(q).
 *
 * Build: make -C cuda/llm/mmq mmq_q4_K_test ; run: ./cuda/llm/mmq/mmq_q4_K_test
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define QK_K 256
#define Q4K_BYTES 144   /* d(2) + dmin(2) + sc[12] + qs[128] */
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

/* ---------------- 6-bit sub-scale extraction (matches matvec_q4_K_q8_1_dp4a) ---- */
static __host__ __device__ inline void q4k_sub_scale(const uint8_t *sc, int ib, int *sv, int *mv) {
    if (ib < 4) { *sv = sc[ib] & 0x3F; *mv = sc[ib + 4] & 0x3F; }
    else { int i4 = ib - 4;
           *sv = (sc[8 + i4] & 0x0F) | (((sc[i4]     >> 6) & 3) << 4);
           *mv = (sc[8 + i4] >> 4)   | (((sc[4 + i4] >> 6) & 3) << 4); }
}

/* ---------------- host Q4_K block decode (canonical order) ---------------- */
static void q4k_decode_block_h(const uint8_t *bp, float *out) {
    float d    = half_to_float_h(*(const uint16_t *)(bp + 0));
    float dmin = half_to_float_h(*(const uint16_t *)(bp + 2));
    const uint8_t *sc = bp + 4;
    const uint8_t *qs = bp + 16;
    for (int ib = 0; ib < 8; ib++) {
        int sv, mv; q4k_sub_scale(sc, ib, &sv, &mv);
        float dl = d * sv, ml = dmin * mv;
        const uint8_t *q = qs + (ib >> 1) * 32;
        int hi = ib & 1;
        for (int k = 0; k < 32; k++) {
            int nib = hi ? (q[k] >> 4) : (q[k] & 0xF);
            out[ib * 32 + k] = dl * nib - ml;
        }
    }
}

static void q4k_decode_row_h(const uint8_t *row, int K, float *out) {
    int nb = K / QK_K;
    for (int b = 0; b < nb; b++)
        q4k_decode_block_h(row + b * Q4K_BYTES, out + b * QK_K);
}

/* ---------------- device helpers ---------------- */
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
/* weight = d*sv*nibble - dmin*mv ; result = d8*(d*sv*sum(nibble*x) - dmin*mv*sum(x)) */
__global__ void mmq_q4_K_naive(float *dst, const uint8_t *W, const int8_t *xq8,
                                const float *xs, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;
    int nb = K / QK_K, row_bytes = nb * Q4K_BYTES;
    const uint8_t *wrow = W + (size_t)n * row_bytes;
    const int8_t *xrow = xq8 + (size_t)m * K;
    const float *xsrow = xs + (size_t)m * (K / CK);
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *bp = wrow + b * Q4K_BYTES;
        float d_val    = half_to_float_dev(*(const uint16_t *)(bp + 0));
        float dmin_val = half_to_float_dev(*(const uint16_t *)(bp + 2));
        const uint8_t *sc = bp + 4;
        const uint8_t *qs = bp + 16;
        for (int ib = 0; ib < 8; ib++) {
            int sv, mv; q4k_sub_scale(sc, ib, &sv, &mv);
            const uint8_t *q = qs + (ib >> 1) * 32;
            int hi = ib & 1;
            int kbase = b * QK_K + ib * CK;
            float d8 = xsrow[kbase / CK];
            int sum_d = 0, sum_m = 0;
            for (int k = 0; k < 32; k++) {
                int nib = hi ? (q[k] >> 4) : (q[k] & 0xF);
                int xv = xrow[kbase + k];
                sum_d += nib * xv;
                sum_m += xv;
            }
            acc += d_val * sv * d8 * (float)sum_d - dmin_val * mv * d8 * (float)sum_m;
        }
    }
    dst[(size_t)m * N + n] = acc;
}

/* ---------------- MMA helpers ---------------- */
static __device__ __forceinline__ int pack4(const int8_t *p) {
    return (p[0] & 0xff) | ((p[1] & 0xff) << 8) | ((p[2] & 0xff) << 16) | ((p[3] & 0xff) << 24);
}

/* ---------------- PHASE 2: single-warp tensor-core MMA ---------------- */
/* sW_plus[r][k] = nibble (0..15), sW_minus[r][k] = 1.
   sWs_dsv[r] = d*sv, sWs_dm[r] = dmin*mv (per sub-block).
   result += d8 * (dsv * cp - dm * cm)  (cp = sum(nibble*x), cm = sum(x)) */
__global__ void mmq_q4_K_mma(float *dst, const uint8_t *W, const int8_t *xq8,
                              const float *xs, int M, int N, int K) {
    int lane = threadIdx.x, gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * 16, m0 = blockIdx.y * 8;
    int nb = K / QK_K, row_bytes = nb * Q4K_BYTES, nsb = K / CK;

    __shared__ int8_t sW_plus[16][CK];
    __shared__ int8_t sW_minus[16][CK];
    __shared__ float  sWs_dsv[16];
    __shared__ float  sWs_dm[16];
    __shared__ int8_t sX[8][CK];
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
            const uint8_t *bp = W + (size_t)n * row_bytes + (sb / 8) * Q4K_BYTES;
            float d_val    = half_to_float_dev(*(const uint16_t *)(bp + 0));
            float dmin_val = half_to_float_dev(*(const uint16_t *)(bp + 2));
            const uint8_t *sc = bp + 4;
            const uint8_t *qs_full = bp + 16;
            int ib = sb & 7;
            int sv, mv; q4k_sub_scale(sc, ib, &sv, &mv);
            const uint8_t *qs = qs_full + (ib >> 1) * 32;
            int hi = ib & 1;
            sWs_dsv[r] = d_val * sv;
            sWs_dm[r]  = dmin_val * mv;
            for (int k = 0; k < 32; k++) {
                int nib = hi ? (qs[k] >> 4) : (qs[k] & 0xF);
                sW_plus[r][k]  = (int8_t)nib;
                sW_minus[r][k] = (int8_t)1;
            }
        }
        __syncwarp();

        int a0 = pack4(&sW_plus[gid][tid*4]),   a1 = pack4(&sW_plus[gid+8][tid*4]);
        int a2 = pack4(&sW_plus[gid][tid*4+16]), a3 = pack4(&sW_plus[gid+8][tid*4+16]);
        int a0m = pack4(&sW_minus[gid][tid*4]),   a1m = pack4(&sW_minus[gid+8][tid*4]);
        int a2m = pack4(&sW_minus[gid][tid*4+16]), a3m = pack4(&sW_minus[gid+8][tid*4+16]);
        int b0 = pack4(&sX[gid][tid*4]), b1 = pack4(&sX[gid][tid*4+16]);
        int cp0=0, cp1=0, cp2=0, cp3=0, cm0=0, cm1=0, cm2=0, cm3=0;

        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(cp0), "=r"(cp1), "=r"(cp2), "=r"(cp3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "r"(0), "r"(0), "r"(0), "r"(0));
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(cm0), "=r"(cm1), "=r"(cm2), "=r"(cm3)
            : "r"(a0m), "r"(a1m), "r"(a2m), "r"(a3m), "r"(b0), "r"(b1),
              "r"(0), "r"(0), "r"(0), "r"(0));

        float dsv0 = sWs_dsv[gid], dsv8 = sWs_dsv[gid+8];
        float dm0  = sWs_dm[gid],  dm8  = sWs_dm[gid+8];
        float xc0 = sXs[tid*2], xc1 = sXs[tid*2+1];
        f0 += dsv0 * xc0 * (float)cp0 - dm0 * xc0 * (float)cm0;
        f1 += dsv0 * xc1 * (float)cp1 - dm0 * xc1 * (float)cm1;
        f2 += dsv8 * xc0 * (float)cp2 - dm8 * xc0 * (float)cm2;
        f3 += dsv8 * xc1 * (float)cp3 - dm8 * xc1 * (float)cm3;
        __syncwarp();
    }
    int n_a = n0 + gid, n_b = n0 + gid + 8;
    int m_a = m0 + tid*2, m_b = m0 + tid*2 + 1;
    if (m_a < M) { dst[(size_t)m_a * N + n_a] = f0; dst[(size_t)m_a * N + n_b] = f2; }
    if (m_b < M) { dst[(size_t)m_b * N + n_a] = f1; dst[(size_t)m_b * N + n_b] = f3; }
}

/* ---------------- PHASE 2.6: decode-amortized multi-warp MMA (production shape) ---- */
#define WN 4
#define TG 4
__global__ void mmq_q4_K_mma_v3(float *dst, const uint8_t *W, const int8_t *xq8,
                                  const float *xs, int M, int N, int K) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * (16 * WN) + warp * 16;
    int m_base = blockIdx.y * (8 * TG);
    int ntg = (M - m_base + 7) / 8; if (ntg > TG) ntg = TG; if (ntg < 0) ntg = 0;
    int nb = K / QK_K, row_bytes = nb * Q4K_BYTES, nsb = K / CK;

    __shared__ int8_t sW_plus[16 * WN][CK];
    __shared__ int8_t sW_minus[16 * WN][CK];
    __shared__ float  sWs_dsv[16 * WN];
    __shared__ float  sWs_dm[16 * WN];
    __shared__ int8_t sX[8 * TG][CK];
    __shared__ float  sXs[8 * TG];

    float f[TG][4];
    for (int g = 0; g < TG; g++) { f[g][0]=f[g][1]=f[g][2]=f[g][3]=0; }

    for (int sb = 0; sb < nsb; sb++) {
        if (lane < 16) {
            int r = warp*16 + lane, n = n0 + lane;
            const uint8_t *bp = W + (size_t)n*row_bytes + (sb/8)*Q4K_BYTES;
            float d_val    = half_to_float_dev(*(const uint16_t*)(bp + 0));
            float dmin_val = half_to_float_dev(*(const uint16_t*)(bp + 2));
            const uint8_t *sc = bp + 4;
            const uint8_t *qs_full = bp + 16;
            int ib = sb & 7;
            int sv, mv; q4k_sub_scale(sc, ib, &sv, &mv);
            const uint8_t *qs = qs_full + (ib >> 1) * 32;
            int hi = ib & 1;
            sWs_dsv[r] = d_val * sv;
            sWs_dm[r]  = dmin_val * mv;
            for (int k = 0; k < 32; k++) {
                int nib = hi ? (qs[k] >> 4) : (qs[k] & 0xF);
                sW_plus[r][k]  = (int8_t)nib;
                sW_minus[r][k] = (int8_t)1;
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
        int a0  = pack4(&sW_plus[wr+gid][tid*4]),   a1  = pack4(&sW_plus[wr+gid+8][tid*4]);
        int a2  = pack4(&sW_plus[wr+gid][tid*4+16]), a3  = pack4(&sW_plus[wr+gid+8][tid*4+16]);
        int a0m = pack4(&sW_minus[wr+gid][tid*4]),   a1m = pack4(&sW_minus[wr+gid+8][tid*4]);
        int a2m = pack4(&sW_minus[wr+gid][tid*4+16]), a3m = pack4(&sW_minus[wr+gid+8][tid*4+16]);
        float dsv0 = sWs_dsv[wr+gid], dsv8 = sWs_dsv[wr+gid+8];
        float dm0  = sWs_dm[wr+gid],  dm8  = sWs_dm[wr+gid+8];
        for (int g = 0; g < ntg; g++) {
            int b0 = pack4(&sX[g*8+gid][tid*4]), b1 = pack4(&sX[g*8+gid][tid*4+16]);
            int cp0=0, cp1=0, cp2=0, cp3=0, cm0=0, cm1=0, cm2=0, cm3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=r"(cp0),"=r"(cp1),"=r"(cp2),"=r"(cp3)
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                 "r"(0),"r"(0),"r"(0),"r"(0));
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=r"(cm0),"=r"(cm1),"=r"(cm2),"=r"(cm3)
                :"r"(a0m),"r"(a1m),"r"(a2m),"r"(a3m),"r"(b0),"r"(b1),
                 "r"(0),"r"(0),"r"(0),"r"(0));
            float xc0 = sXs[g*8+tid*2], xc1 = sXs[g*8+tid*2+1];
            f[g][0] += dsv0*xc0*(float)cp0 - dm0*xc0*(float)cm0;
            f[g][1] += dsv0*xc1*(float)cp1 - dm0*xc1*(float)cm1;
            f[g][2] += dsv8*xc0*(float)cp2 - dm8*xc0*(float)cm2;
            f[g][3] += dsv8*xc1*(float)cp3 - dm8*xc1*(float)cm3;
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
        q4k_decode_row_h(W + (size_t)n * (K / QK_K) * Q4K_BYTES, K, wf);
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
    int M = argc > 1 ? atoi(argv[1]) : 8;      /* tokens */
    int N = argc > 2 ? atoi(argv[2]) : 512;     /* out_dim */
    int K = argc > 3 ? atoi(argv[3]) : 2048;    /* in_dim  */
    printf("MMQ Q4_K test: M=%d N=%d K=%d  (K/256=%d blocks/row)\n", M, N, K, K / QK_K);
    if (K % QK_K) { fprintf(stderr, "K must be multiple of 256\n"); return 1; }

    srand(1234);
    int nb = K / QK_K, row_bytes = nb * Q4K_BYTES;
    size_t wbytes = (size_t)N * row_bytes;
    uint8_t *W = (uint8_t *)malloc(wbytes);
    /* random data; then overwrite d/dmin with finite half values.
       sc[] (6-bit scales) and qs[] (4-bit nibbles) stay random — the extraction
       and nibble split work correctly regardless of raw byte values. */
    for (size_t i = 0; i < wbytes; i++) W[i] = rand() & 0xff;
    for (int n = 0; n < N; n++)
        for (int b = 0; b < nb; b++) {
            float d    = 0.01f + (rand() / (float)RAND_MAX) * 0.05f;
            float dmin = 0.005f + (rand() / (float)RAND_MAX) * 0.02f;
            *(uint16_t *)(W + (size_t)n * row_bytes + b * Q4K_BYTES + 0) = float_to_half_h(d);
            *(uint16_t *)(W + (size_t)n * row_bytes + b * Q4K_BYTES + 2) = float_to_half_h(dmin);
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

    /* P1: dp4a */
    dim3 qg(K / CK, M); quantize_q8_1<<<qg, 32>>>(dX, dXq, dXs, M, K);
    CUDA_CHECK(cudaGetLastError());
    dim3 bg((N + 127) / 128, M); mmq_q4_K_naive<<<bg, 128>>>(dDst, dW, dXq, dXs, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float *out = (float *)malloc((size_t)M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(out, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    double e = rel_l2(out, ref, M * N);
    printf("P1 dp4a   vs CPU-F32 oracle: rel_L2 = %.6f\n", e);

    /* P2: single-warp MMA */
    if (N % 16 != 0) { fprintf(stderr, "P2 needs N %% 16 == 0\n"); return 1; }
    CUDA_CHECK(cudaMemset(dDst, 0, (size_t)M * N * sizeof(float)));
    dim3 mg(N / 16, (M + 7) / 8); mmq_q4_K_mma<<<mg, 32>>>(dDst, dW, dXq, dXs, M, N, K);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    float *out2 = (float *)malloc((size_t)M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(out2, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    double e2 = rel_l2(out2, ref, M * N);
    double e2v1 = rel_l2(out2, out, M * N);
    printf("P2 mma    vs CPU-F32 oracle: rel_L2 = %.6f\n", e2);
    printf("P2 mma    vs P1 dp4a       : rel_L2 = %.6f\n", e2v1);
    printf("  ref[0..3]=%.4f %.4f %.4f %.4f\n  mma[0..3]=%.4f %.4f %.4f %.4f\n",
           ref[0], ref[1], ref[2], ref[3], out2[0], out2[1], out2[2], out2[3]);

    /* P2.6: decode-amortized MMA (production shape) */
    double e4 = 1e9, e4v = 1e9;
    if (N % (16 * WN) == 0) {
        CUDA_CHECK(cudaMemset(dDst, 0, (size_t)M * N * sizeof(float)));
        dim3 v3g(N / (16 * WN), (M + 8*TG - 1) / (8*TG));
        mmq_q4_K_mma_v3<<<v3g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *o4 = (float *)malloc((size_t)M * N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(o4, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
        e4 = rel_l2(o4, ref, M*N); e4v = rel_l2(o4, out2, M*N);
        printf("P2.6 mma_v3 vs CPU-F32 oracle: rel_L2 = %.6f\n", e4);
        printf("P2.6 mma_v3 vs P2 mma         : rel_L2 = %.6f\n", e4v);
        free(o4);
    } else printf("P2.6 skipped (N %% %d != 0)\n", 16 * WN);

    int ok = (e < 0.05) && (e2 < 0.05) && (e2v1 < 1e-4) && (e4 < 0.05) && (e4v < 1e-4);
    printf("%s\n", ok ? "PASS (P1..P2.6 data path correct)" : "FAIL");

    /* Rough throughput */
    {
        int iters = 300;
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        dim3 bgp((N + 127) / 128, M);
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) mmq_q4_K_naive<<<bgp, 128>>>(dDst, dW, dXq, dXs, M, N, K);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_dp; cudaEventElapsedTime(&ms_dp, t0, t1); ms_dp /= iters;
        float ms_v3 = 0;
        if (N % (16 * WN) == 0) {
            dim3 v3g(N / (16 * WN), (M + 8*TG - 1) / (8*TG));
            cudaEventRecord(t0);
            for (int i = 0; i < iters; i++) mmq_q4_K_mma_v3<<<v3g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms_v3, t0, t1); ms_v3 /= iters;
        }
        double flop = 2.0 * M * N * K;
        printf("\nperf (1 GEMM):\n");
        printf("  dp4a   : %.3f ms  %.1f GFLOP/s\n", ms_dp, flop / (ms_dp * 1e6));
        if (ms_v3 > 0)
            printf("  mma_v3 : %.3f ms  %.1f GFLOP/s  (%.2fx vs dp4a)\n",
                   ms_v3, flop / (ms_v3 * 1e6), ms_dp / ms_v3);
    }
    return ok ? 0 : 1;
}
