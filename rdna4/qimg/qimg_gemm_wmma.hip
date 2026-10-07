/* qimg tiled gfx12 (RDNA4) WMMA GEMM: Y[M,N] f32 (+)= X[M,K] bf16 x W[N,K]^T (+ bias).
 *
 * Trimmed port of rdna4/llm/gemm_wmma.hip (main branch, commit 2422e4d8):
 * BMxBN CTA tile, K step 32, register prefetch of the next K tile, double-
 * buffered kslot-major LDS (16 lanes read consecutive 16-byte vectors), M tiles
 * fastest in the grid, clamped loads / masked stores.  Added for the SVDQuant
 * W4A16 DiT:
 *   - LdInt4G64: logical int4 nibbles [N, K/2] (even k -> low nibble, signed
 *     -8..7) times a BF16 per-(row, 64-group) scale, decoded to BF16 in the
 *     tile loader (same values as gemm_int4w_bf16a_wmma_t);
 *   - epilogue bias add and optional accumulate into Y (fuses the LoRA add).
 * X is BF16; cast_f32_bf16_smooth produces it (dividing by the SmoothQuant
 * factor for the main GEMM).  Requires K % 32 == 0 (K % 64 for int4). */

typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ unsigned int bf16_rne_hi(float f) {
    unsigned int u = __float_as_uint(f);
    return u + ((u >> 16) & 1u) + 0x7FFFu;
}
__device__ __forceinline__ unsigned int pack2(unsigned int lo_hi, unsigned int hi_hi) {
    return __builtin_amdgcn_perm(hi_hi, lo_hi, 0x07060302u);
}

struct Raw { unsigned int w[4]; };

struct LdDense {
    __device__ static size_t row_bytes(int K) { return (size_t)K * 2; }
    __device__ static Raw load(const unsigned char *row, int k, const unsigned short *) {
        return __builtin_bit_cast(Raw, *(const u16x8 *)(row + (size_t)k * 2));
    }
    __device__ static u16x8 decode(const Raw &r, int, const unsigned short *) {
        return __builtin_bit_cast(u16x8, r);
    }
};

struct LdInt4G64 {
    __device__ static size_t row_bytes(int K) { return (size_t)K / 2; }
    /* 8 nibbles = 4 bytes; w[1] carries the group scale (bf16 in the high half). */
    __device__ static Raw load(const unsigned char *row, int k, const unsigned short *srow) {
        Raw r;
        r.w[0] = *(const unsigned int *)(row + (k >> 1));
        r.w[1] = (unsigned int)srow[k >> 6] << 16;
        r.w[2] = r.w[3] = 0;
        return r;
    }
    __device__ static u16x8 decode(const Raw &r, int, const unsigned short *) {
        const float s = __uint_as_float(r.w[1]);
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int nb = (r.w[0] >> (4 * i)) & 0xF;
            v[i] = (float)(nb >= 8 ? nb - 16 : nb) * s;
        }
        u32x4 w;
        w[0] = pack2(bf16_rne_hi(v[0]), bf16_rne_hi(v[1]));
        w[1] = pack2(bf16_rne_hi(v[2]), bf16_rne_hi(v[3]));
        w[2] = pack2(bf16_rne_hi(v[4]), bf16_rne_hi(v[5]));
        w[3] = pack2(bf16_rne_hi(v[6]), bf16_rne_hi(v[7]));
        return __builtin_bit_cast(u16x8, w);
    }
};

/* One K range of the tiled WMMA loop, accumulating into acc.  X rows have stride ldx elements; W rows
 * are LB-encoded with K = ldw columns (for row_bytes / scale groups). */
/* INT8 g64 weights (Q4_K_M-style 8-bit for sensitive linears): [N, K] int8, BF16 scale per (row, 64-group). */
struct LdInt8G64 {
    __device__ static size_t row_bytes(int K) { return (size_t)K; }
    __device__ static Raw load(const unsigned char *row, int k, const unsigned short *srow) {
        Raw r;
        const uint2 v = *(const uint2 *)(row + k);
        r.w[0] = v.x; r.w[1] = v.y; r.w[2] = (unsigned int)srow[k >> 6] << 16; r.w[3] = 0;
        return r;
    }
    __device__ static u16x8 decode(const Raw &r, int, const unsigned short *) {
        const float s = __uint_as_float(r.w[2]);
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = (float)(signed char)((r.w[i >> 2] >> (8 * (i & 3))) & 0xFF) * s;
        u32x4 w;
        w[0] = pack2(bf16_rne_hi(v[0]), bf16_rne_hi(v[1])); w[1] = pack2(bf16_rne_hi(v[2]), bf16_rne_hi(v[3]));
        w[2] = pack2(bf16_rne_hi(v[4]), bf16_rne_hi(v[5])); w[3] = pack2(bf16_rne_hi(v[6]), bf16_rne_hi(v[7]));
        return __builtin_bit_cast(u16x8, w);
    }
};

/* Wide int4 loader: one thread stages a whole K32 slice of one weight row (16 bytes = 32 nibbles) plus its
 * g64 scale, decoding all four kslot vectors; replaces 4 narrow 4-byte loads + 4 scale loads per row. */
struct Int4Wide { uint4 q; float s; };
__device__ __forceinline__ Int4Wide i4w_load(const unsigned char *row, int k, const unsigned short *srow) {
    Int4Wide w; w.q = *(const uint4 *)(row + (k >> 1)); w.s = __uint_as_float((unsigned)srow[k >> 6] << 16); return w;
}
__device__ __forceinline__ u16x8 i4w_decode(unsigned word, float s) {
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; i++) { int nb = (word >> (4 * i)) & 0xF; v[i] = (float)(nb >= 8 ? nb - 16 : nb) * s; }
    u32x4 o;
    o[0] = pack2(bf16_rne_hi(v[0]), bf16_rne_hi(v[1])); o[1] = pack2(bf16_rne_hi(v[2]), bf16_rne_hi(v[3]));
    o[2] = pack2(bf16_rne_hi(v[4]), bf16_rne_hi(v[5])); o[3] = pack2(bf16_rne_hi(v[6]), bf16_rne_hi(v[7]));
    return __builtin_bit_cast(u16x8, o);
}

/* K loop with int4 W staged by the wide loader (threads [0,BN) load B rows; all threads load A). */
template <int BM, int BN, int WAVES_M, int WAVES_N>
__device__ __forceinline__ void qgemm_kloop_i4w(
        float8 (&acc)[BM / WAVES_M / 16][BN / WAVES_N / 16], u16x8 *smA, u16x8 *smB,
        const unsigned short *__restrict__ X, int ldx, const unsigned char *__restrict__ W, int K,
        const unsigned short *__restrict__ S, int m0, int n0, int M, int N, int kbeg, int kend) {
    constexpr int NT = WAVES_M * WAVES_N * 32;
    constexpr int WTM = BM / WAVES_M, WTN = BN / WAVES_N;
    constexpr int FM = WTM / 16, FN = WTN / 16;
    constexpr int AV = BM * 4 / NT;
    static_assert(BN <= NT && AV >= 1, "wide int4 loader needs BN <= threads");
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M, half = lane >> 4, idx = lane & 15;
    const u16x8 *pa[AV]; int sa[AV];
#pragma unroll
    for (int i = 0; i < AV; i++) {
        int e = tid + i * NT, r = e >> 2, ks = e & 3;
        int gr = m0 + r; if (gr > M - 1) gr = M - 1;
        pa[i] = (const u16x8 *)(X + (size_t)gr * ldx + kbeg + ks * 8);
        sa[i] = ks * BM + r;
    }
    const bool bth = tid < BN;
    int br = n0 + tid; if (br > N - 1) br = N - 1;
    const unsigned char *pb = W + (size_t)br * (K / 2);
    const unsigned short *ps = S + (size_t)br * (K / 64);
    const int nk = (kend - kbeg) / 32;
    if (nk <= 0) return;
    u16x8 ra[AV]; Int4Wide rb;
#pragma unroll
    for (int i = 0; i < AV; i++) smA[sa[i]] = pa[i][0];
    if (bth) { Int4Wide w = i4w_load(pb, kbeg, ps);
#pragma unroll
        for (int ks = 0; ks < 4; ks++) smB[ks * BN + tid] = i4w_decode(((const unsigned *)&w.q)[ks], w.s); }
    __syncthreads();
    int buf = 0;
    for (int kt = 0; kt < nk; kt++) {
        const int has_next = kt + 1 < nk, kn = kbeg + (kt + 1) * 32;
        if (has_next) {
#pragma unroll
            for (int i = 0; i < AV; i++) ra[i] = pa[i][(kt + 1) * 4];
            if (bth) rb = i4w_load(pb, kn, ps);
        }
        const u16x8 *cA = smA + buf * 4 * BM, *cB = smB + buf * 4 * BN;
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
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(fa[i], fb[j], acc[i][j]);
        }
        if (has_next) {
            u16x8 *nA = smA + (buf ^ 1) * 4 * BM, *nB = smB + (buf ^ 1) * 4 * BN;
#pragma unroll
            for (int i = 0; i < AV; i++) nA[sa[i]] = ra[i];
            if (bth) {
#pragma unroll
                for (int ks = 0; ks < 4; ks++) nB[ks * BN + tid] = i4w_decode(((const unsigned *)&rb.q)[ks], rb.s);
            }
            buf ^= 1;
        }
        __syncthreads();
    }
}

template <typename LB, int BM, int BN, int WAVES_M, int WAVES_N>
__device__ __forceinline__ void qgemm_kloop(
        float8 (&acc)[BM / WAVES_M / 16][BN / WAVES_N / 16], u16x8 *smA, u16x8 *smB,
        const unsigned short *__restrict__ X, int ldx, const unsigned char *__restrict__ W, int ldw,
        const unsigned short *__restrict__ S, int m0, int n0, int M, int N, int kbeg, int kend) {
    constexpr int NT = WAVES_M * WAVES_N * 32;
    constexpr int WTM = BM / WAVES_M, WTN = BN / WAVES_N;
    constexpr int FM = WTM / 16, FN = WTN / 16;
    constexpr int AV = BM * 4 / NT, BV = BN * 4 / NT;
    static_assert(AV >= 1 && BV >= 1 && BM * 4 % NT == 0 && BN * 4 % NT == 0, "tile/thread mismatch");
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    const int half = lane >> 4, idx = lane & 15;
    const size_t wrow = LB::row_bytes(ldw);
    const int ng = ldw / 64;
    const u16x8 *pa[AV];
    const unsigned char *pb[BV];
    const unsigned short *ps[BV];
    int sa[AV], sb[BV], kb[BV];
#pragma unroll
    for (int i = 0; i < AV; i++) {
        int e = tid + i * NT, r = e >> 2, ks = e & 3;
        int gr = m0 + r; if (gr > M - 1) gr = M - 1;
        pa[i] = (const u16x8 *)(X + (size_t)gr * ldx + kbeg + ks * 8);
        sa[i] = ks * BM + r;
    }
#pragma unroll
    for (int i = 0; i < BV; i++) {
        int e = tid + i * NT, r = e >> 2, ks = e & 3;
        int gr = n0 + r; if (gr > N - 1) gr = N - 1;
        pb[i] = W + (size_t)gr * wrow;
        ps[i] = S ? S + (size_t)gr * ng : (const unsigned short *)0;
        kb[i] = kbeg + ks * 8;
        sb[i] = ks * BN + r;
    }
    const int nk = (kend - kbeg) / 32;
    if (nk <= 0) return;
    u16x8 ra[AV];
    Raw rb[BV];
#pragma unroll
    for (int i = 0; i < AV; i++) smA[sa[i]] = pa[i][0];
#pragma unroll
    for (int i = 0; i < BV; i++) smB[sb[i]] = LB::decode(LB::load(pb[i], kb[i], ps[i]), kb[i], ps[i]);
    __syncthreads();
    int buf = 0;
    for (int kt = 0; kt < nk; kt++) {
        const int has_next = kt + 1 < nk;
        const int kn = (kt + 1) * 32;
        if (has_next) {
#pragma unroll
            for (int i = 0; i < AV; i++) ra[i] = pa[i][(kt + 1) * 4];
#pragma unroll
            for (int i = 0; i < BV; i++) rb[i] = LB::load(pb[i], kb[i] + kn, ps[i]);
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
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(fa[i], fb[j], acc[i][j]);
        }
        if (has_next) {
            u16x8 *nA = smA + (buf ^ 1) * 4 * BM;
            u16x8 *nB = smB + (buf ^ 1) * 4 * BN;
#pragma unroll
            for (int i = 0; i < AV; i++) nA[sa[i]] = ra[i];
#pragma unroll
            for (int i = 0; i < BV; i++) nB[sb[i]] = LB::decode(rb[i], kb[i] + kn, ps[i]);
            buf ^= 1;
        }
        __syncthreads();
    }
}

/* Y[M,N] (+)= X[M,K] W^T (+ X2[M,K2] W2^T, dense bf16: the fused SVDQuant lora_up phase) + bias.
 * accum: bit0 = add into Y, bit1 = tanh-GELU epilogue.
 * gridDim.y > 1: split-K partials of the main phase to Y + z*M*N (no X2, no accum). */
template <typename LB, int BM, int BN, int WAVES_M, int WAVES_N, bool WIDE = false>
__device__ __forceinline__ void qgemm_body(
        float *__restrict__ Y, const unsigned char *__restrict__ W,
        const unsigned short *__restrict__ S, const unsigned short *__restrict__ X,
        const float *__restrict__ bias, int N, int K, int M, int accum,
        const unsigned short *__restrict__ X2, const unsigned char *__restrict__ W2, int K2) {
    constexpr int WTM = BM / WAVES_M, WTN = BN / WAVES_N;
    constexpr int FM = WTM / 16, FN = WTN / 16;
    __shared__ u16x8 smA[2 * 4 * BM];
    __shared__ u16x8 smB[2 * 4 * BN];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M, half = lane >> 4, idx = lane & 15;
    /* Grouped rasterization over a 1-D grid: GROUP_M M-tiles sweep all N tiles before the next
     * group, so the X slab (GROUP_M*BM rows) stays in the Infinity Cache while W streams. */
    constexpr int GROUP_M = 8;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN;
    const int pid = blockIdx.x, per_group = GROUP_M * num_n, group = pid / per_group;
    const int first_m = group * GROUP_M;
    const int gsize = (num_m - first_m) < GROUP_M ? (num_m - first_m) : GROUP_M;
    const int m0 = (first_m + (pid % per_group) % gsize) * BM;
    const int n0 = ((pid % per_group) / gsize) * BN;
    const int ksplit = gridDim.y;
    const int kchunk = ((K / 32 + ksplit - 1) / ksplit) * 32;
    const int kbeg = blockIdx.y * kchunk;
    const int kend = kbeg + kchunk < K ? kbeg + kchunk : K;
    if (ksplit > 1) Y += (size_t)blockIdx.y * M * N;

    float8 acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    if constexpr (WIDE) qgemm_kloop_i4w<BM, BN, WAVES_M, WAVES_N>(acc, smA, smB, X, K, W, K, S, m0, n0, M, N, kbeg, kend);
    else qgemm_kloop<LB, BM, BN, WAVES_M, WAVES_N>(acc, smA, smB, X, K, W, K, S, m0, n0, M, N, kbeg, kend);
    if (X2) qgemm_kloop<LdDense, BM, BN, WAVES_M, WAVES_N>(acc, smA, smB, X2, K2, W2, K2, 0, m0, n0, M, N, 0, K2);

#pragma unroll
    for (int j = 0; j < FN; j++) {
        const int col = n0 + wn * WTN + j * 16 + idx;
        if (col >= N) continue;
        const float bv = (bias && blockIdx.y == 0) ? bias[col] : 0.f;
#pragma unroll
        for (int i = 0; i < FM; i++) {
            const int row0 = m0 + wm * WTM + i * 16 + half * 8;
#pragma unroll
            for (int v = 0; v < 8; v++) {
                if (row0 + v < M) {
                    float *y = Y + (size_t)(row0 + v) * N + col;
                    float o = ((accum & 1) ? *y : 0.f) + acc[i][j][v] + bv;
                    if (accum & 2) {   /* fused tanh-GELU epilogue (fc1) */
                        float u = 0.7978845608f * (o + 0.044715f * o * o * o);
                        o = o / (1.0f + __expf(-2.0f * u));
                    }
                    *y = o;
                }
            }
        }
    }
}

#define QGEMM_INST(name, LB, BM, BN, WMs, WNs)                                     \
extern "C" __global__ __launch_bounds__(WMs * WNs * 32) void name(                 \
        float *Y, const unsigned char *W, const unsigned short *S,                 \
        const unsigned short *X, const float *bias, int N, int K, int M, int accum, \
        const unsigned short *X2, const unsigned char *W2, int K2) {               \
    qgemm_body<LB, BM, BN, WMs, WNs>(Y, W, S, X, bias, N, K, M, accum, X2, W2, K2); \
}

QGEMM_INST(qgemm_bf16_128x128, LdDense,   128, 128, 2, 2)
QGEMM_INST(qgemm_bf16_64x128,  LdDense,    64, 128, 1, 4)
QGEMM_INST(qgemm_bf16_128x64,  LdDense,   128,  64, 2, 2)
QGEMM_INST(qgemm_i4_128x128,   LdInt4G64, 128, 128, 2, 2)
QGEMM_INST(qgemm_i4_64x128,    LdInt4G64,  64, 128, 1, 4)
QGEMM_INST(qgemm_i8_128x128w8, LdInt8G64, 128, 128, 2, 4)
QGEMM_INST(qgemm_i8_64x128,   LdInt8G64,  64, 128, 1, 4)
QGEMM_INST(qgemm_bf16_64x64,   LdDense,    64,  64, 2, 2)
QGEMM_INST(qgemm_i4_128x128w8, LdInt4G64, 128, 128, 2, 4)
QGEMM_INST(qgemm_i4_128x256,   LdInt4G64, 128, 256, 2, 4)
QGEMM_INST(qgemm_bf16_128x128w8, LdDense, 128, 128, 2, 4)
extern "C" __global__ __launch_bounds__(256) void qgemm_i4w_128x128w8(
        float *Y, const unsigned char *W, const unsigned short *S,
        const unsigned short *X, const float *bias, int N, int K, int M, int accum,
        const unsigned short *X2, const unsigned char *W2, int K2) {
    qgemm_body<LdInt4G64, 128, 128, 2, 4, true>(Y, W, S, X, bias, N, K, M, accum, X2, W2, K2);
}

/* Xb[t,k] = bf16_rne(X[t,k] / smooth[k]) (smooth may be NULL). 8 elements per thread, K % 8 == 0. */
extern "C" __global__ void cast_f32_bf16_smooth(unsigned short *Xb, const float *X,
                                                const float *smooth, int K, long n) {
    long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= n) return;
    const float4 a = *(const float4 *)(X + i), b = *(const float4 *)(X + i + 4);
    float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    if (smooth) {
        const int k = (int)(i % K);
        const float4 sa = *(const float4 *)(smooth + k), sb = *(const float4 *)(smooth + k + 4);
        const float sv[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] /= sv[j];
    }
    u32x4 w;
    w[0] = pack2(bf16_rne_hi(v[0]), bf16_rne_hi(v[1]));
    w[1] = pack2(bf16_rne_hi(v[2]), bf16_rne_hi(v[3]));
    w[2] = pack2(bf16_rne_hi(v[4]), bf16_rne_hi(v[5]));
    w[3] = pack2(bf16_rne_hi(v[6]), bf16_rne_hi(v[7]));
    *(u32x4 *)(Xb + i) = w;
}

/* dtb[i] = bf16(sum_z P[z][i]) : fixed-order split-K reduce fused with the bf16 cast of the rank activations. */
extern "C" __global__ void splitk_reduce_bf16(unsigned short *dtb, const float *P, int splits, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = P[i];
    for (int z = 1; z < splits; z++) v += P[(long)z * n + i];
    dtb[i] = (unsigned short)(bf16_rne_hi(v) >> 16);
}

/* ---- vectorized elementwise / normalization kernels (replace the scalar qimg ones) ---- */
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor(v, m, 32);
    return v;
}

/* tanh-GELU, 4 per thread: 0.5v(1+tanh(u)) == v / (1 + exp(-2u)). n % 4 == 0. */
extern "C" __global__ void ew_gelu4(float *x, long n) {
    long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    float4 v = *(float4 *)(x + i);
    float *p = (float *)&v;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        float a = p[j], u = 0.7978845608f * (a + 0.044715f * a * a * a);
        p[j] = a / (1.0f + __expf(-2.0f * u));
    }
    *(float4 *)(x + i) = v;
}

/* x[t,c] += gate[c] * proj[t,c], 4 per thread (dim % 4 == 0). */
extern "C" __global__ void ew_gated_add4(float *x, const float *proj, const float *gate, int dim, long n) {
    long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    int c = (int)(i % dim);
    float4 a = *(float4 *)(x + i), p = *(const float4 *)(proj + i), g = *(const float4 *)(gate + c);
    a.x += g.x * p.x; a.y += g.y * p.y; a.z += g.z * p.z; a.w += g.w * p.w;
    *(float4 *)(x + i) = a;
}

/* BF16 round-to-nearest-even truncation in place, 4 per thread. */
extern "C" __global__ void ew_trunc4(float *x, long n) {
    long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    float4 v = *(float4 *)(x + i);
    unsigned *u = (unsigned *)&v;
#pragma unroll
    for (int j = 0; j < 4; j++) u[j] = (u[j] + 0x7FFFu + ((u[j] >> 16) & 1u)) & 0xFFFF0000u;
    *(float4 *)(x + i) = v;
}

/* Per-head RMSNorm (head_dim 128): one wave per (token, head), 4 elements per lane; 8 waves per block. */
extern "C" __global__ void ew_rmsnorm_ph(float *x, const float *w, int N, int n_heads) {
    int wave = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (wave >= N * n_heads) return;
    float *h = x + (long)wave * 128 + lane * 4;   /* (tok, head) rows are contiguous 128-float slices */
    float4 v = *(float4 *)h;
    float inv = rsqrtf(wave_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w) / 128.0f + 1e-6f);
    float4 g = *(const float4 *)(w + lane * 4);
    v.x *= inv * g.x; v.y *= inv * g.y; v.z *= inv * g.z; v.w *= inv * g.w;
    *(float4 *)h = v;
}

/* adaLN: out = LayerNorm(x) * (1 + scale) + shift, one 256-thread block per token, dim <= 4096 and
 * dim % 1024 == 0; values stay in registers (two-pass mean/var without a second global read). */
extern "C" __global__ void ew_adaln(float *out, const float *x, const float *shift, const float *scale, int dim) {
    __shared__ float red[8];
    const int t = blockIdx.x, tid = threadIdx.x, nv = dim / 1024;
    const float *xr = x + (long)t * dim;
    float4 v[4];
    float s = 0.f;
    for (int k = 0; k < nv; k++) { v[k] = *(const float4 *)(xr + (k * 256 + tid) * 4); s += v[k].x + v[k].y + v[k].z + v[k].w; }
    s = wave_sum(s); if ((tid & 31) == 0) red[tid >> 5] = s; __syncthreads();
    float mean = 0.f; for (int i = 0; i < 8; i++) mean += red[i]; mean /= dim; __syncthreads();
    float q = 0.f;
    for (int k = 0; k < nv; k++) {
        float a = v[k].x - mean, b = v[k].y - mean, c = v[k].z - mean, d = v[k].w - mean;
        q += a * a + b * b + c * c + d * d;
    }
    q = wave_sum(q); if ((tid & 31) == 0) red[tid >> 5] = q; __syncthreads();
    float var = 0.f; for (int i = 0; i < 8; i++) var += red[i];
    const float inv = rsqrtf(var / dim + 1e-6f);
    for (int k = 0; k < nv; k++) {
        int c = (k * 256 + tid) * 4;
        float4 sc = *(const float4 *)(scale + c), sh = *(const float4 *)(shift + c), o;
        o.x = (v[k].x - mean) * inv * (1.f + sc.x) + sh.x; o.y = (v[k].y - mean) * inv * (1.f + sc.y) + sh.y;
        o.z = (v[k].z - mean) * inv * (1.f + sc.z) + sh.z; o.w = (v[k].w - mean) * inv * (1.f + sc.w) + sh.w;
        *(float4 *)(out + (long)t * dim + c) = o;
    }
}
