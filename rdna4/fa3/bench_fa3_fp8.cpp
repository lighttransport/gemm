/*
 * bench_fa3_fp8.cpp - RDNA4/gfx1201 FP8 flash-attention design harness.
 *
 * Standalone HD=128 image-attention benchmark.  It keeps the existing repo's
 * BQ=64, 4-wave WMMA lane layout, then explores FA3-transferable pieces that
 * do not require Hopper async MMA: log2-domain online softmax, deferred sum
 * reduction, FP8 P rebias, and wider KV tiles.
 */

#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../rocew.h"

#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

static float fp8_e4m3_to_f32(uint8_t v) {
    if (v == 0x00) return 0.0f;
    if (v == 0x80) return -0.0f;
    uint32_t sign = (v >> 7) & 1u;
    uint32_t exp  = (v >> 3) & 0xFu;
    uint32_t mant = v & 0x7u;
    if (exp == 0xF && mant == 0x7) return NAN;
    if (exp == 0) {
        float f = ldexpf((float)mant / 8.0f, -6);
        return sign ? -f : f;
    }
    float f = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    return sign ? -f : f;
}

static double timer_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static double cosine_sim(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

static float max_abs_diff(const float *a, const float *b, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static void fill_fp8_random(uint8_t *dst, size_t n, float abs_max, unsigned *rng) {
    for (size_t i = 0; i < n; i++) {
        unsigned r = *rng = *rng * 1103515245u + 12345u;
        float u = ((int)(r & 0xFFFFFFu) - 0x800000) / (float)0x800000;
        float v = u * abs_max;
        if (v >  448.0f) v =  448.0f;
        if (v < -448.0f) v = -448.0f;
        dst[i] = hip_f32_to_fp8_e4m3(v);
    }
}

static void fill_ones(float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = 1.0f;
}

static void fa_ref_fp32(float *out, const uint8_t *Q, const uint8_t *K_t,
                        const uint8_t *V_t, const float *sQ, const float *sK,
                        const float *sV, int n_tok, int n_heads, int head_dim) {
    int dim = n_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    float *scores = (float *)malloc((size_t)n_tok * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        for (int q = 0; q < n_tok; q++) {
            float row_max = -1e30f;
            for (int kv = 0; kv < n_tok; kv++) {
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float qv = fp8_e4m3_to_f32(Q[(size_t)q * dim + h * head_dim + d]) * sQ[(size_t)h * n_tok + q];
                    float kvv = fp8_e4m3_to_f32(K_t[((size_t)h * n_tok + kv) * head_dim + d]) * sK[(size_t)h * n_tok + kv];
                    s += qv * kvv;
                }
                s *= scale;
                scores[kv] = s;
                if (s > row_max) row_max = s;
            }
            float l = 0.0f;
            for (int kv = 0; kv < n_tok; kv++) {
                scores[kv] = expf(scores[kv] - row_max);
                l += scores[kv];
            }
            float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int kv = 0; kv < n_tok; kv++) {
                    float p = scores[kv] * inv_l;
                    float vv = fp8_e4m3_to_f32(V_t[((size_t)h * n_tok + kv) * head_dim + d]) * sV[h];
                    acc += p * vv;
                }
                out[(size_t)q * dim + h * head_dim + d] = acc;
            }
        }
    }
    free(scores);
}

static const char *kernel_src = R"FA3SRC(
typedef unsigned char u8;
typedef unsigned int  u32;
typedef u32 u32x2 __attribute__((ext_vector_type(2)));
typedef u8  u8x8  __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

#define HD   128
#define K_NB (HD / 16)
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ u32x2 pack_u8x8(u8x8 v) {
    union { u8x8 b; u32x2 w; } u;
    u.b = v;
    return u.w;
}

__device__ __forceinline__ u8 f32_to_fp8_e4m3(float f) {
    if (!(f == f)) return 0x7f;
    if (f == 0.0f) return 0;
    union { float f; u32 i; } u;
    u.f = f;
    u32 bits = u.i;
    u32 sign = (bits >> 31) & 1u;
    int e = (int)((bits >> 23) & 0xffu) - 127 + 7;
    u32 mant = (bits >> 20) & 0x7u;
    if (e > 15 || (e == 15 && mant > 6u)) { e = 15; mant = 6u; }
    if (e <= 0) return (u8)(sign << 7);
    return (u8)((sign << 7) | ((u32)(e & 0xf) << 3) | (mant & 0x7u));
}

__device__ __forceinline__ u8 f32_to_fp8_e4m3_pos(float f) {
    if (f <= 0.0f) return 0;
    union { float f; u32 i; } u;
    u.f = f;
    int e = (int)((u.i >> 23) & 0xffu) - 120;
    u32 mant = (u.i >> 20) & 0x7u;
    if (e > 15 || (e == 15 && mant > 6u)) return 0x7e;
    if (e <= 0) return 0;
    return (u8)(((u32)(e & 0xf) << 3) | (mant & 0x7u));
}

__device__ __forceinline__ float exp2_fast_bits(float x) {
    if (x <= -24.0f) return 0.0f;
    if (x > 8.0f) x = 8.0f;
    int i = (int)((x + 127.0f) * 8388608.0f);
    union { int i; float f; } u;
    u.i = i;
    return u.f;
}

__device__ __forceinline__ float fa3_exp2(float x, int exp_mode) {
    return exp_mode ? exp2_fast_bits(x) : exp2f(x);
}

#define WMMA_FP8(A,B,C) C = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(A, B, C)

#define DEFINE_FA3_KERNEL(NAME, BKV, APPROX, EXPMODE, WAVE_CT, BQX, DB, QG) \
extern "C" __global__ __launch_bounds__((WAVE_CT) * 32, 1) \
void NAME(float *out, const u8 *Q, const u8 *K_t, const u8 *V_t, \
          const float *sQ, const float *sK, const float *sV, \
          int n_tok, int n_heads, float inv_sqrtd) { \
    enum { CH = (BKV) / 16 }; \
    enum { DBUF = (DB) ? 2 : 1 }; \
    enum { QGROUPS = (QG) }; \
    int h = blockIdx.x; \
    int qb = blockIdx.y; \
    int q0 = qb * (BQX); \
    int tid = threadIdx.x; \
    int wid = tid >> 5; \
    int lid = tid & 31; \
    int half = lid >> 4; \
    int idx = lid & 15; \
    int dim = n_heads * HD; \
    __shared__ u8 smK[DBUF * (BKV) * HD]; \
    __shared__ u8 smV[DBUF * HD * (BKV)]; \
    __shared__ u8 smP[(WAVE_CT) * CH * 16 * 16]; \
    __shared__ float smSq[(BQX)]; \
    __shared__ float smSk[DBUF * (BKV)]; \
    u8 *smP_w = smP + wid * CH * 16 * 16; \
    for (int i = tid; i < (BQX); i += (WAVE_CT) * 32) { \
        int row = q0 + i; \
        smSq[i] = (row < n_tok) ? sQ[(size_t)h * n_tok + row] : 1.0f; \
    } \
    u32x2 q_reg[QGROUPS][K_NB]; \
    for (int g = 0; g < QGROUPS; g++) { \
        int q_row = q0 + (g * (WAVE_CT) + wid) * 16 + idx; \
        u8x8 tmp; \
        for (int kb = 0; kb < K_NB; kb++) { \
            for (int i = 0; i < 8; i++) { \
                int d = kb * 16 + half * 8 + i; \
                u8 qv = 0; \
                if (q_row < n_tok) qv = Q[(size_t)q_row * dim + h * HD + d]; \
                tmp[i] = qv; \
            } \
            q_reg[g][kb] = pack_u8x8(tmp); \
        } \
    } \
    __syncthreads(); \
    float q_scale_lane[QGROUPS][8]; \
    for (int g = 0; g < QGROUPS; g++) { \
        for (int i = 0; i < 8; i++) { \
            int m_row = (g * (WAVE_CT) + wid) * 16 + half * 8 + i; \
            q_scale_lane[g][i] = smSq[m_row]; \
        } \
    } \
    float8 O_acc[QGROUPS][K_NB]; \
    for (int g = 0; g < QGROUPS; g++) for (int kb = 0; kb < K_NB; kb++) for (int i = 0; i < 8; i++) O_acc[g][kb][i] = 0.0f; \
    float m_i[QGROUPS][8], l_i[QGROUPS][8]; \
    for (int g = 0; g < QGROUPS; g++) for (int i = 0; i < 8; i++) { m_i[g][i] = -1e30f; l_i[g][i] = 0.0f; } \
    int n_kv_tiles = (n_tok + (BKV) - 1) / (BKV); \
    int total = (BKV) * HD; \
    if (DB) { \
        int kv0 = 0; \
        for (int e = tid; e < total; e += (WAVE_CT) * 32) { \
            int r = e / HD; \
            int d = e - r * HD; \
            int kv = kv0 + r; \
            size_t base = ((size_t)h * n_tok + kv) * HD; \
            u8 kk = 0, vv = 0; \
            if (kv < n_tok) { kk = K_t[base + d]; vv = V_t[base + d]; } \
            smK[r * HD + d] = kk; \
            smV[d * (BKV) + r] = vv; \
        } \
        for (int i = tid; i < (BKV); i += (WAVE_CT) * 32) { \
            int kv = kv0 + i; \
            smSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f; \
        } \
        __syncthreads(); \
    } \
    for (int t = 0; t < n_kv_tiles; t++) { \
        int kv0 = t * (BKV); \
        int cur = (DB) ? (t & 1) : 0; \
        int nxt = cur ^ 1; \
        u8 *curK = smK + cur * total; \
        u8 *curV = smV + cur * HD * (BKV); \
        float *curSk = smSk + cur * (BKV); \
        if (DB) { \
            int next_t = t + 1; \
            if (next_t < n_kv_tiles) { \
                int next_kv0 = next_t * (BKV); \
                u8 *nextK = smK + nxt * total; \
                u8 *nextV = smV + nxt * HD * (BKV); \
                float *nextSk = smSk + nxt * (BKV); \
                for (int e = tid; e < total; e += (WAVE_CT) * 32) { \
                    int r = e / HD; \
                    int d = e - r * HD; \
                    int kv = next_kv0 + r; \
                    size_t base = ((size_t)h * n_tok + kv) * HD; \
                    u8 kk = 0, vv = 0; \
                    if (kv < n_tok) { kk = K_t[base + d]; vv = V_t[base + d]; } \
                    nextK[r * HD + d] = kk; \
                    nextV[d * (BKV) + r] = vv; \
                } \
                for (int i = tid; i < (BKV); i += (WAVE_CT) * 32) { \
                    int kv = next_kv0 + i; \
                    nextSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f; \
                } \
            } \
        } else { \
            for (int e = tid; e < total; e += (WAVE_CT) * 32) { \
                int r = e / HD; \
                int d = e - r * HD; \
                int kv = kv0 + r; \
                size_t base = ((size_t)h * n_tok + kv) * HD; \
                u8 kk = 0, vv = 0; \
                if (kv < n_tok) { kk = K_t[base + d]; vv = V_t[base + d]; } \
                curK[r * HD + d] = kk; \
                curV[d * (BKV) + r] = vv; \
            } \
            for (int i = tid; i < (BKV); i += (WAVE_CT) * 32) { \
                int kv = kv0 + i; \
                curSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f; \
            } \
            __syncthreads(); \
        } \
        for (int g = 0; g < QGROUPS; g++) { \
            float8 S[CH]; \
            for (int c = 0; c < CH; c++) for (int i = 0; i < 8; i++) S[c][i] = 0.0f; \
            for (int kb = 0; kb < K_NB; kb++) { \
                for (int c = 0; c < CH; c++) { \
                    u8x8 b_K; \
                    for (int i = 0; i < 8; i++) { \
                        int dpos = kb * 16 + half * 8 + i; \
                        b_K[i] = curK[(c * 16 + idx) * HD + dpos]; \
                    } \
                    u32x2 b_K_p = pack_u8x8(b_K); \
                    WMMA_FP8(q_reg[g][kb], b_K_p, S[c]); \
                } \
            } \
            float row_max[8]; \
            for (int i = 0; i < 8; i++) { \
                float mx = -1e30f; \
                for (int c = 0; c < CH; c++) { \
                    int col = c * 16 + idx; \
                    int kv = kv0 + col; \
                    float s = S[c][i] * q_scale_lane[g][i] * curSk[col] * inv_sqrtd; \
                    if (APPROX) s *= LOG2E; \
                    if (kv >= n_tok) s = -1e30f; \
                    S[c][i] = s; \
                    mx = fmaxf(mx, s); \
                } \
                mx = fmaxf(mx, __shfl_xor(mx, 1, 32)); \
                mx = fmaxf(mx, __shfl_xor(mx, 2, 32)); \
                mx = fmaxf(mx, __shfl_xor(mx, 4, 32)); \
                mx = fmaxf(mx, __shfl_xor(mx, 8, 32)); \
                row_max[i] = mx; \
            } \
            float alpha[8]; \
            for (int i = 0; i < 8; i++) { \
                float new_max = fmaxf(m_i[g][i], row_max[i]); \
                alpha[i] = (APPROX) ? fa3_exp2(m_i[g][i] - new_max, (EXPMODE)) : __expf(m_i[g][i] - new_max); \
                float local = 0.0f; \
                for (int c = 0; c < CH; c++) { \
                    int col = c * 16 + idx; \
                    int kv = kv0 + col; \
                    float p = (APPROX) ? fa3_exp2(S[c][i] - new_max + 8.0f, (EXPMODE)) : __expf(S[c][i] - new_max); \
                    if (kv >= n_tok) p = 0.0f; \
                    S[c][i] = p; \
                    local += p; \
                } \
                if (APPROX) { \
                    l_i[g][i] = l_i[g][i] * alpha[i] + local; \
                } else { \
                    float sum = local; \
                    sum += __shfl_xor(sum, 1, 32); \
                    sum += __shfl_xor(sum, 2, 32); \
                    sum += __shfl_xor(sum, 4, 32); \
                    sum += __shfl_xor(sum, 8, 32); \
                    l_i[g][i] = l_i[g][i] * alpha[i] + sum; \
                } \
                m_i[g][i] = new_max; \
            } \
            for (int kb = 0; kb < K_NB; kb++) for (int i = 0; i < 8; i++) O_acc[g][kb][i] *= alpha[i]; \
            for (int c = 0; c < CH; c++) { \
                u8 *smPc = smP_w + c * 16 * 16; \
                for (int i = 0; i < 8; i++) { \
                    int m_row = half * 8 + i; \
                    smPc[m_row * 16 + idx] = f32_to_fp8_e4m3_pos(S[c][i]); \
                } \
            } \
            __builtin_amdgcn_s_waitcnt(0); \
            for (int c = 0; c < CH; c++) { \
                u8 *smPc = smP_w + c * 16 * 16; \
                u8x8 ap_b; \
                for (int i = 0; i < 8; i++) ap_b[i] = smPc[idx * 16 + half * 8 + i]; \
                u32x2 ap = pack_u8x8(ap_b); \
                for (int kb = 0; kb < K_NB; kb++) { \
                    u8x8 bv; \
                    for (int i = 0; i < 8; i++) { \
                        int d_col = kb * 16 + idx; \
                        int kv_k = c * 16 + half * 8 + i; \
                        bv[i] = curV[d_col * (BKV) + kv_k]; \
                    } \
                    u32x2 bv_p = pack_u8x8(bv); \
                    WMMA_FP8(ap, bv_p, O_acc[g][kb]); \
                } \
            } \
        } \
        __syncthreads(); \
    } \
    float vs = sV[h]; \
    for (int g = 0; g < QGROUPS; g++) { \
        int q_base = q0 + (g * (WAVE_CT) + wid) * 16; \
        float inv_l[8]; \
        for (int i = 0; i < 8; i++) { \
            float l = l_i[g][i]; \
            if (APPROX) { \
                l += __shfl_xor(l, 1, 32); \
                l += __shfl_xor(l, 2, 32); \
                l += __shfl_xor(l, 4, 32); \
                l += __shfl_xor(l, 8, 32); \
            } \
            inv_l[i] = (l > 0.0f) ? 1.0f / l : 0.0f; \
        } \
        for (int kb = 0; kb < K_NB; kb++) { \
            for (int i = 0; i < 8; i++) { \
                int row = q_base + half * 8 + i; \
                int d = kb * 16 + idx; \
                if (row < n_tok) out[(size_t)row * dim + h * HD + d] = O_acc[g][kb][i] * inv_l[i] * vs; \
            } \
        } \
    } \
}

DEFINE_FA3_KERNEL(fa3_exact_b16,            16, 0, 0, 4,  64, 0, 1)
DEFINE_FA3_KERNEL(fa3_approx_b32_fast,      32, 1, 1, 4,  64, 0, 1)
DEFINE_FA3_KERNEL(fa3_approx_b32_exp2,      32, 1, 0, 4,  64, 0, 1)
DEFINE_FA3_KERNEL(fa3_approx_b64_fast,      64, 1, 1, 4,  64, 0, 1)
DEFINE_FA3_KERNEL(fa3_approx_b32_fast_8w,   32, 1, 1, 8, 128, 0, 1)
DEFINE_FA3_KERNEL(fa3_approx_b32_exp2_8w,   32, 1, 0, 8, 128, 0, 1)
DEFINE_FA3_KERNEL(fa3_approx_b32_exp2_8w_db, 32, 1, 0, 8, 128, 1, 1)
DEFINE_FA3_KERNEL(fa3_approx_b64_fast_8w,   64, 1, 1, 8, 128, 0, 1)
DEFINE_FA3_KERNEL(fa3_approx_b32_fast_16w,  32, 1, 1, 16, 256, 0, 1)
DEFINE_FA3_KERNEL(fa3_approx_b32_fast_32w,  32, 1, 1, 32, 512, 0, 1)
DEFINE_FA3_KERNEL(fa3_approx_b32_fast_16w_qg2, 32, 1, 1, 16, 512, 0, 2)
DEFINE_FA3_KERNEL(fa3_approx_b64_fast_16w,  64, 1, 1, 16, 256, 0, 1)

/* Roofline: QK@K WMMA -> exp2 -> P@V WMMA. NO max-reduce, NO l_i, NO transpose.
 * Data is garbage; this measures the raw matmul+exp ceiling only. */
extern "C" __global__ __launch_bounds__(512, 1)
void fa3_roof_16w(float *out, const u8 *Q, const u8 *K_t, const u8 *V_t,
                  const float *sQ, const float *sK, const float *sV,
                  int n_tok, int n_heads, float inv_sqrtd) {
    enum { BKV = 32, CH = 2, BQX = 256, WAVE_CT = 16 };
    int h = blockIdx.x, qb = blockIdx.y, q0 = qb * BQX;
    int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31, half = lid >> 4, idx = lid & 15;
    int dim = n_heads * HD;
    __shared__ u8 smK[BKV * HD]; __shared__ u8 smV[HD * BKV];
    int q_row = q0 + wid * 16 + idx;
    u32x2 q_reg[K_NB];
    { u8x8 t; for (int kb = 0; kb < K_NB; kb++) { for (int i = 0; i < 8; i++) { int d = kb*16+half*8+i; t[i] = (q_row<n_tok)?Q[(size_t)q_row*dim+h*HD+d]:(u8)0; } q_reg[kb]=pack_u8x8(t);} }
    float8 O_acc[K_NB]; for (int kb=0; kb<K_NB; kb++) for (int i=0;i<8;i++) O_acc[kb][i]=0.0f;
    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    for (int t = 0; t < n_kv_tiles; t++) {
        int kv0 = t*BKV;
        for (int e=tid; e<BKV*HD; e+=WAVE_CT*32){int r=e/HD,d=e-r*HD,kv=kv0+r;size_t b=((size_t)h*n_tok+kv)*HD;u8 kk=0,vv=0;if(kv<n_tok){kk=K_t[b+d];vv=V_t[b+d];}smK[r*HD+d]=kk;smV[d*BKV+r]=vv;}
        __syncthreads();
        float8 S[CH]; for (int c=0;c<CH;c++) for(int i=0;i<8;i++) S[c][i]=0.0f;
        for (int kb=0; kb<K_NB; kb++) for (int c=0;c<CH;c++){u8x8 bK;for(int i=0;i<8;i++)bK[i]=smK[(c*16+idx)*HD+kb*16+half*8+i];WMMA_FP8(q_reg[kb],pack_u8x8(bK),S[c]);}
        for (int c=0;c<CH;c++) for (int i=0;i<8;i++) S[c][i]=exp2f(S[c][i]*inv_sqrtd);
        for (int c=0;c<CH;c++){u8x8 ap;for(int i=0;i<8;i++)ap[i]=f32_to_fp8_e4m3_pos(S[c][i]);u32x2 ap_p=pack_u8x8(ap);for(int kb=0;kb<K_NB;kb++){u8x8 bv;for(int i=0;i<8;i++)bv[i]=smV[(kb*16+idx)*BKV+c*16+half*8+i];WMMA_FP8(ap_p,pack_u8x8(bv),O_acc[kb]);}}
        __syncthreads();
    }
    for (int kb=0;kb<K_NB;kb++) for(int i=0;i<8;i++){int row=q0+wid*16+half*8+i;int d=kb*16+idx;if(row<n_tok)out[(size_t)row*dim+h*HD+d]=O_acc[kb][i];}
}

/* Roofline w/ 4-head software pipeline: 4 indep heads share a CTA so QK/exp/PV
 * of different heads fill dependency gaps. Garbage data, ceiling probe only. */
extern "C" __global__ __launch_bounds__(256, 1)
void fa3_roof_4h(float *out, const u8 *Q, const u8 *K_t, const u8 *V_t,
                 const float *sQ, const float *sK, const float *sV,
                 int n_tok, int n_heads, float inv_sqrtd) {
    enum { BKV = 32, CH = 2, BQX = 128, WAVE_CT = 8, NH = 2 };
    int h0 = blockIdx.x * NH, qb = blockIdx.y, q0 = qb * BQX;
    int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31, half = lid >> 4, idx = lid & 15;
    int dim = n_heads * HD;
    __shared__ u8 smK[NH][BKV*HD]; __shared__ u8 smV[NH][HD*BKV];
    int q_row = q0 + wid*16 + idx;
    u32x2 q_reg[NH][K_NB];
    for (int n=0;n<NH;n++){u8x8 t;for(int kb=0;kb<K_NB;kb++){for(int i=0;i<8;i++){int d=kb*16+half*8+i;t[i]=(q_row<n_tok)?Q[(size_t)q_row*dim+(h0+n)*HD+d]:(u8)0;}q_reg[n][kb]=pack_u8x8(t);}}
    float8 O[NH][K_NB]; for(int n=0;n<NH;n++)for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++)O[n][kb][i]=0.0f;
    int n_kv=(n_tok+BKV-1)/BKV;
    for(int t=0;t<n_kv;t++){int kv0=t*BKV;
        for(int n=0;n<NH;n++)for(int e=tid;e<BKV*HD;e+=WAVE_CT*32){int r=e/HD,d=e-r*HD,kv=kv0+r;size_t b=((size_t)(h0+n)*n_tok+kv)*HD;u8 kk=0,vv=0;if(kv<n_tok){kk=K_t[b+d];vv=V_t[b+d];}smK[n][r*HD+d]=kk;smV[n][d*BKV+r]=vv;}
        __syncthreads();
        float8 S[NH][CH]; for(int n=0;n<NH;n++)for(int c=0;c<CH;c++)for(int i=0;i<8;i++)S[n][c][i]=0.0f;
        for(int n=0;n<NH;n++)for(int kb=0;kb<K_NB;kb++)for(int c=0;c<CH;c++){u8x8 bK;for(int i=0;i<8;i++)bK[i]=smK[n][(c*16+idx)*HD+kb*16+half*8+i];WMMA_FP8(q_reg[n][kb],pack_u8x8(bK),S[n][c]);}
        for(int n=0;n<NH;n++)for(int c=0;c<CH;c++)for(int i=0;i<8;i++)S[n][c][i]=exp2f(S[n][c][i]*inv_sqrtd);
        for(int n=0;n<NH;n++)for(int c=0;c<CH;c++){u8x8 ap;for(int i=0;i<8;i++)ap[i]=f32_to_fp8_e4m3_pos(S[n][c][i]);u32x2 app=pack_u8x8(ap);for(int kb=0;kb<K_NB;kb++){u8x8 bv;for(int i=0;i<8;i++)bv[i]=smV[n][(kb*16+idx)*BKV+c*16+half*8+i];WMMA_FP8(app,pack_u8x8(bv),O[n][kb]);}}
        __syncthreads();
    }
    for(int n=0;n<NH;n++)for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++){int row=q0+wid*16+half*8+i;int d=kb*16+idx;if(row<n_tok)out[(size_t)row*dim+(h0+n)*HD+d]=O[n][kb][i];}
}

/* Shuffle P-transpose (modes shflp_8w/shflp_16w). Lane map IS correct now
 * (cos=0.99963 vs LDS), but ~13% slower: 25.2 vs 29.0 TF/s @ S=4096,16w.
 * Conclusion: LDS P-transpose is NOT the FP8-FA bottleneck — 64 shfls/tile cost
 * more than the LDS round-trip. Ceiling is small-tile occupancy, not transpose. */
extern "C" __global__ __launch_bounds__(512, 1)
void fa3_approx_b32_16w_shflp(float *out, const u8 *Q, const u8 *K_t, const u8 *V_t,
                              const float *sQ, const float *sK, const float *sV,
                              int n_tok, int n_heads, float inv_sqrtd) {
    enum { BKV = 32, CH = 2, BQX = 256, WAVE_CT = 16 };
    int h = blockIdx.x;
    int qb = blockIdx.y;
    int q0 = qb * BQX;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int half = lid >> 4;
    int idx = lid & 15;
    int dim = n_heads * HD;
    __shared__ u8 smK[BKV * HD];
    __shared__ u8 smV[HD * BKV];
    __shared__ float smSq[BQX];
    __shared__ float smSk[BKV];
    for (int i = tid; i < BQX; i += WAVE_CT * 32) {
        int row = q0 + i;
        smSq[i] = (row < n_tok) ? sQ[(size_t)h * n_tok + row] : 1.0f;
    }
    int q_row = q0 + wid * 16 + idx;
    u32x2 q_reg[K_NB];
    {
        u8x8 tmp;
        for (int kb = 0; kb < K_NB; kb++) {
            for (int i = 0; i < 8; i++) {
                int d = kb * 16 + half * 8 + i;
                tmp[i] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (u8)0;
            }
            q_reg[kb] = pack_u8x8(tmp);
        }
    }
    __syncthreads();
    float q_scale_lane[8];
    for (int i = 0; i < 8; i++) q_scale_lane[i] = smSq[wid * 16 + half * 8 + i];
    float8 O_acc[K_NB];
    float m_i[8], l_i[8];
    for (int kb = 0; kb < K_NB; kb++) for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;
    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }
    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    for (int t = 0; t < n_kv_tiles; t++) {
        int kv0 = t * BKV;
        for (int e = tid; e < BKV * HD; e += WAVE_CT * 32) {
            int r = e / HD; int d = e - r * HD; int kv = kv0 + r;
            size_t base = ((size_t)h * n_tok + kv) * HD;
            u8 kk = 0, vv = 0;
            if (kv < n_tok) { kk = K_t[base + d]; vv = V_t[base + d]; }
            smK[r * HD + d] = kk; smV[d * BKV + r] = vv;
        }
        for (int i = tid; i < BKV; i += WAVE_CT * 32) {
            int kv = kv0 + i;
            smSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f;
        }
        __syncthreads();
        float8 S[CH];
        for (int c = 0; c < CH; c++) for (int i = 0; i < 8; i++) S[c][i] = 0.0f;
        for (int kb = 0; kb < K_NB; kb++) for (int c = 0; c < CH; c++) {
            u8x8 b_K;
            for (int i = 0; i < 8; i++) b_K[i] = smK[(c * 16 + idx) * HD + kb * 16 + half * 8 + i];
            WMMA_FP8(q_reg[kb], pack_u8x8(b_K), S[c]);
        }
        float row_max[8];
        for (int i = 0; i < 8; i++) {
            float mx = -1e30f;
            for (int c = 0; c < CH; c++) {
                int col = c * 16 + idx; int kv = kv0 + col;
                float s = S[c][i] * q_scale_lane[i] * smSk[col] * inv_sqrtd * LOG2E;
                if (kv >= n_tok) s = -1e30f; S[c][i] = s; mx = fmaxf(mx, s);
            }
            mx = fmaxf(mx, __shfl_xor(mx, 1, 32)); mx = fmaxf(mx, __shfl_xor(mx, 2, 32));
            mx = fmaxf(mx, __shfl_xor(mx, 4, 32)); mx = fmaxf(mx, __shfl_xor(mx, 8, 32));
            row_max[i] = mx;
        }
        float alpha[8];
        for (int i = 0; i < 8; i++) {
            float new_max = fmaxf(m_i[i], row_max[i]);
            alpha[i] = exp2f(m_i[i] - new_max); float local = 0.0f;
            for (int c = 0; c < CH; c++) {
                int col = c * 16 + idx; int kv = kv0 + col;
                float p = exp2f(S[c][i] - new_max + 8.0f); if (kv >= n_tok) p = 0.0f;
                S[c][i] = p; local += p;
            }
            l_i[i] = l_i[i] * alpha[i] + local; m_i[i] = new_max;
        }
        for (int kb = 0; kb < K_NB; kb++) for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];
        int row_slot = idx & 7; int row_half = idx >> 3;
        for (int c = 0; c < CH; c++) {
            u8x8 ap_b;
            for (int i = 0; i < 8; i++) {
                int src_lane = row_half * 16 + half * 8 + i;
                float v[8]; for (int j = 0; j < 8; j++) v[j] = __shfl(S[c][j], src_lane, 32);
                ap_b[i] = f32_to_fp8_e4m3(v[row_slot]);
            }
            u32x2 ap = pack_u8x8(ap_b);
            for (int kb = 0; kb < K_NB; kb++) {
                u8x8 bv;
                for (int i = 0; i < 8; i++) bv[i] = smV[(kb * 16 + idx) * BKV + c * 16 + half * 8 + i];
                WMMA_FP8(ap, pack_u8x8(bv), O_acc[kb]);
            }
        }
        __syncthreads();
    }
    float inv_l[8];
    for (int i = 0; i < 8; i++) {
        float l = l_i[i];
        l += __shfl_xor(l, 1, 32); l += __shfl_xor(l, 2, 32);
        l += __shfl_xor(l, 4, 32); l += __shfl_xor(l, 8, 32);
        inv_l[i] = (l > 0.0f) ? 1.0f / l : 0.0f;
    }
    float vs = sV[h];
    for (int kb = 0; kb < K_NB; kb++) for (int i = 0; i < 8; i++) {
        int row = q0 + wid * 16 + half * 8 + i; int d = kb * 16 + idx;
        if (row < n_tok) out[(size_t)row * dim + h * HD + d] = O_acc[kb][i] * inv_l[i] * vs;
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void fa3_approx_b32_8w_shflp(float *out, const u8 *Q, const u8 *K_t, const u8 *V_t,
                              const float *sQ, const float *sK, const float *sV,
                              int n_tok, int n_heads, float inv_sqrtd) {
    enum { BKV = 32, CH = 2, BQX = 128, WAVE_CT = 8 };
    int h = blockIdx.x;
    int qb = blockIdx.y;
    int q0 = qb * BQX;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int half = lid >> 4;
    int idx = lid & 15;
    int dim = n_heads * HD;
    __shared__ u8 smK[BKV * HD];
    __shared__ u8 smV[HD * BKV];
    __shared__ float smSq[BQX];
    __shared__ float smSk[BKV];
    for (int i = tid; i < BQX; i += WAVE_CT * 32) {
        int row = q0 + i;
        smSq[i] = (row < n_tok) ? sQ[(size_t)h * n_tok + row] : 1.0f;
    }
    int q_row = q0 + wid * 16 + idx;
    u32x2 q_reg[K_NB];
    {
        u8x8 tmp;
        for (int kb = 0; kb < K_NB; kb++) {
            for (int i = 0; i < 8; i++) {
                int d = kb * 16 + half * 8 + i;
                tmp[i] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (u8)0;
            }
            q_reg[kb] = pack_u8x8(tmp);
        }
    }
    __syncthreads();
    float q_scale_lane[8];
    for (int i = 0; i < 8; i++) q_scale_lane[i] = smSq[wid * 16 + half * 8 + i];

    float8 O_acc[K_NB];
    float m_i[8], l_i[8];
    for (int kb = 0; kb < K_NB; kb++) for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;
    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }

    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    for (int t = 0; t < n_kv_tiles; t++) {
        int kv0 = t * BKV;
        for (int e = tid; e < BKV * HD; e += WAVE_CT * 32) {
            int r = e / HD;
            int d = e - r * HD;
            int kv = kv0 + r;
            size_t base = ((size_t)h * n_tok + kv) * HD;
            u8 kk = 0, vv = 0;
            if (kv < n_tok) { kk = K_t[base + d]; vv = V_t[base + d]; }
            smK[r * HD + d] = kk;
            smV[d * BKV + r] = vv;
        }
        for (int i = tid; i < BKV; i += WAVE_CT * 32) {
            int kv = kv0 + i;
            smSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f;
        }
        __syncthreads();

        float8 S[CH];
        for (int c = 0; c < CH; c++) for (int i = 0; i < 8; i++) S[c][i] = 0.0f;
        for (int kb = 0; kb < K_NB; kb++) {
            for (int c = 0; c < CH; c++) {
                u8x8 b_K;
                for (int i = 0; i < 8; i++) {
                    int dpos = kb * 16 + half * 8 + i;
                    b_K[i] = smK[(c * 16 + idx) * HD + dpos];
                }
                WMMA_FP8(q_reg[kb], pack_u8x8(b_K), S[c]);
            }
        }
        float row_max[8];
        for (int i = 0; i < 8; i++) {
            float mx = -1e30f;
            for (int c = 0; c < CH; c++) {
                int col = c * 16 + idx;
                int kv = kv0 + col;
                float s = S[c][i] * q_scale_lane[i] * smSk[col] * inv_sqrtd * LOG2E;
                if (kv >= n_tok) s = -1e30f;
                S[c][i] = s;
                mx = fmaxf(mx, s);
            }
            mx = fmaxf(mx, __shfl_xor(mx, 1, 32));
            mx = fmaxf(mx, __shfl_xor(mx, 2, 32));
            mx = fmaxf(mx, __shfl_xor(mx, 4, 32));
            mx = fmaxf(mx, __shfl_xor(mx, 8, 32));
            row_max[i] = mx;
        }
        float alpha[8];
        for (int i = 0; i < 8; i++) {
            float new_max = fmaxf(m_i[i], row_max[i]);
            alpha[i] = exp2f(m_i[i] - new_max);
            float local = 0.0f;
            for (int c = 0; c < CH; c++) {
                int col = c * 16 + idx;
                int kv = kv0 + col;
                float p = exp2f(S[c][i] - new_max + 8.0f);
                if (kv >= n_tok) p = 0.0f;
                S[c][i] = p;
                local += p;
            }
            l_i[i] = l_i[i] * alpha[i] + local;
            m_i[i] = new_max;
        }
        for (int kb = 0; kb < K_NB; kb++) for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];
        int row_slot = idx & 7;
        int row_half = idx >> 3;
        for (int c = 0; c < CH; c++) {
            u8x8 ap_b;
            for (int i = 0; i < 8; i++) {
                int src_lane = row_half * 16 + half * 8 + i;
                float pv0 = __shfl(S[c][0], src_lane, 32);
                float pv1 = __shfl(S[c][1], src_lane, 32);
                float pv2 = __shfl(S[c][2], src_lane, 32);
                float pv3 = __shfl(S[c][3], src_lane, 32);
                float pv4 = __shfl(S[c][4], src_lane, 32);
                float pv5 = __shfl(S[c][5], src_lane, 32);
                float pv6 = __shfl(S[c][6], src_lane, 32);
                float pv7 = __shfl(S[c][7], src_lane, 32);
                float pv = (row_slot == 0) ? pv0 : ((row_slot == 1) ? pv1 :
                           ((row_slot == 2) ? pv2 : ((row_slot == 3) ? pv3 :
                           ((row_slot == 4) ? pv4 : ((row_slot == 5) ? pv5 :
                           ((row_slot == 6) ? pv6 : pv7))))));
                ap_b[i] = f32_to_fp8_e4m3(pv);
            }
            u32x2 ap = pack_u8x8(ap_b);
            for (int kb = 0; kb < K_NB; kb++) {
                u8x8 bv;
                for (int i = 0; i < 8; i++) {
                    int d_col = kb * 16 + idx;
                    int kv_k = c * 16 + half * 8 + i;
                    bv[i] = smV[d_col * BKV + kv_k];
                }
                WMMA_FP8(ap, pack_u8x8(bv), O_acc[kb]);
            }
        }
        __syncthreads();
    }
    float inv_l[8];
    for (int i = 0; i < 8; i++) {
        float l = l_i[i];
        l += __shfl_xor(l, 1, 32);
        l += __shfl_xor(l, 2, 32);
        l += __shfl_xor(l, 4, 32);
        l += __shfl_xor(l, 8, 32);
        inv_l[i] = (l > 0.0f) ? 1.0f / l : 0.0f;
    }
    float vs = sV[h];
    for (int kb = 0; kb < K_NB; kb++) {
        for (int i = 0; i < 8; i++) {
            int row = q0 + wid * 16 + half * 8 + i;
            int d = kb * 16 + idx;
            if (row < n_tok) out[(size_t)row * dim + h * HD + d] = O_acc[kb][i] * inv_l[i] * vs;
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void fa3_two_pass_b32_8w_max(float *m_out, const u8 *Q, const u8 *K_t,
                             const float *sQ, const float *sK,
                             int n_tok, int n_heads, float inv_sqrtd) {
    enum { BKV = 32, CH = 2, BQX = 128, WAVE_CT = 8 };
    int h = blockIdx.x;
    int qb = blockIdx.y;
    int q0 = qb * BQX;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int half = lid >> 4;
    int idx = lid & 15;
    int dim = n_heads * HD;
    __shared__ u8 smK[BKV * HD];
    __shared__ float smSq[BQX];
    __shared__ float smSk[BKV];
    for (int i = tid; i < BQX; i += WAVE_CT * 32) {
        int row = q0 + i;
        smSq[i] = (row < n_tok) ? sQ[(size_t)h * n_tok + row] : 1.0f;
    }
    int q_row = q0 + wid * 16 + idx;
    u32x2 q_reg[K_NB];
    {
        u8x8 tmp;
        for (int kb = 0; kb < K_NB; kb++) {
            for (int i = 0; i < 8; i++) {
                int d = kb * 16 + half * 8 + i;
                tmp[i] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (u8)0;
            }
            q_reg[kb] = pack_u8x8(tmp);
        }
    }
    __syncthreads();
    float q_scale_lane[8];
    for (int i = 0; i < 8; i++) q_scale_lane[i] = smSq[wid * 16 + half * 8 + i];
    float m_i[8];
    for (int i = 0; i < 8; i++) m_i[i] = -1e30f;
    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    for (int t = 0; t < n_kv_tiles; t++) {
        int kv0 = t * BKV;
        for (int e = tid; e < BKV * HD; e += WAVE_CT * 32) {
            int r = e / HD;
            int d = e - r * HD;
            int kv = kv0 + r;
            smK[r * HD + d] = (kv < n_tok) ? K_t[((size_t)h * n_tok + kv) * HD + d] : (u8)0;
        }
        for (int i = tid; i < BKV; i += WAVE_CT * 32) {
            int kv = kv0 + i;
            smSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f;
        }
        __syncthreads();
        float8 S[CH];
        for (int c = 0; c < CH; c++) for (int i = 0; i < 8; i++) S[c][i] = 0.0f;
        for (int kb = 0; kb < K_NB; kb++) {
            for (int c = 0; c < CH; c++) {
                u8x8 b_K;
                for (int i = 0; i < 8; i++) {
                    int dpos = kb * 16 + half * 8 + i;
                    b_K[i] = smK[(c * 16 + idx) * HD + dpos];
                }
                WMMA_FP8(q_reg[kb], pack_u8x8(b_K), S[c]);
            }
        }
        for (int i = 0; i < 8; i++) {
            float mx = m_i[i];
            for (int c = 0; c < CH; c++) {
                int col = c * 16 + idx;
                int kv = kv0 + col;
                float s = S[c][i] * q_scale_lane[i] * smSk[col] * inv_sqrtd * LOG2E;
                if (kv >= n_tok) s = -1e30f;
                mx = fmaxf(mx, s);
            }
            mx = fmaxf(mx, __shfl_xor(mx, 1, 32));
            mx = fmaxf(mx, __shfl_xor(mx, 2, 32));
            mx = fmaxf(mx, __shfl_xor(mx, 4, 32));
            mx = fmaxf(mx, __shfl_xor(mx, 8, 32));
            m_i[i] = mx;
        }
        __syncthreads();
    }
    for (int i = 0; i < 8; i++) {
        int row = q0 + wid * 16 + half * 8 + i;
        if (row < n_tok) m_out[(size_t)h * n_tok + row] = m_i[i];
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void fa3_two_pass_b32_8w_out(float *out, const u8 *Q, const u8 *K_t, const u8 *V_t,
                             const float *sQ, const float *sK, const float *sV,
                             const float *m_in, int n_tok, int n_heads,
                             float inv_sqrtd) {
    enum { BKV = 32, CH = 2, BQX = 128, WAVE_CT = 8 };
    int h = blockIdx.x;
    int qb = blockIdx.y;
    int q0 = qb * BQX;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int half = lid >> 4;
    int idx = lid & 15;
    int dim = n_heads * HD;
    __shared__ u8 smK[BKV * HD];
    __shared__ u8 smV[HD * BKV];
    __shared__ u8 smP[WAVE_CT * CH * 16 * 16];
    __shared__ float smSq[BQX];
    __shared__ float smSk[BKV];
    u8 *smP_w = smP + wid * CH * 16 * 16;
    for (int i = tid; i < BQX; i += WAVE_CT * 32) {
        int row = q0 + i;
        smSq[i] = (row < n_tok) ? sQ[(size_t)h * n_tok + row] : 1.0f;
    }
    int q_row = q0 + wid * 16 + idx;
    u32x2 q_reg[K_NB];
    {
        u8x8 tmp;
        for (int kb = 0; kb < K_NB; kb++) {
            for (int i = 0; i < 8; i++) {
                int d = kb * 16 + half * 8 + i;
                tmp[i] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (u8)0;
            }
            q_reg[kb] = pack_u8x8(tmp);
        }
    }
    __syncthreads();
    float q_scale_lane[8], m_i[8], l_i[8];
    for (int i = 0; i < 8; i++) {
        int m_row = wid * 16 + half * 8 + i;
        int row = q0 + m_row;
        q_scale_lane[i] = smSq[m_row];
        m_i[i] = (row < n_tok) ? m_in[(size_t)h * n_tok + row] : -1e30f;
        l_i[i] = 0.0f;
    }
    float8 O_acc[K_NB];
    for (int kb = 0; kb < K_NB; kb++) for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;
    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    for (int t = 0; t < n_kv_tiles; t++) {
        int kv0 = t * BKV;
        for (int e = tid; e < BKV * HD; e += WAVE_CT * 32) {
            int r = e / HD;
            int d = e - r * HD;
            int kv = kv0 + r;
            size_t base = ((size_t)h * n_tok + kv) * HD;
            u8 kk = 0, vv = 0;
            if (kv < n_tok) { kk = K_t[base + d]; vv = V_t[base + d]; }
            smK[r * HD + d] = kk;
            smV[d * BKV + r] = vv;
        }
        for (int i = tid; i < BKV; i += WAVE_CT * 32) {
            int kv = kv0 + i;
            smSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f;
        }
        __syncthreads();
        float8 S[CH];
        for (int c = 0; c < CH; c++) for (int i = 0; i < 8; i++) S[c][i] = 0.0f;
        for (int kb = 0; kb < K_NB; kb++) {
            for (int c = 0; c < CH; c++) {
                u8x8 b_K;
                for (int i = 0; i < 8; i++) {
                    int dpos = kb * 16 + half * 8 + i;
                    b_K[i] = smK[(c * 16 + idx) * HD + dpos];
                }
                WMMA_FP8(q_reg[kb], pack_u8x8(b_K), S[c]);
            }
        }
        for (int i = 0; i < 8; i++) {
            float local = 0.0f;
            for (int c = 0; c < CH; c++) {
                int col = c * 16 + idx;
                int kv = kv0 + col;
                float s = S[c][i] * q_scale_lane[i] * smSk[col] * inv_sqrtd * LOG2E;
                float p = exp2f(s - m_i[i] + 8.0f);
                if (kv >= n_tok) p = 0.0f;
                S[c][i] = p;
                local += p;
            }
            l_i[i] += local;
        }
        for (int c = 0; c < CH; c++) {
            u8 *smPc = smP_w + c * 16 * 16;
            for (int i = 0; i < 8; i++) smPc[(half * 8 + i) * 16 + idx] = f32_to_fp8_e4m3(S[c][i]);
        }
        __builtin_amdgcn_s_waitcnt(0);
        for (int c = 0; c < CH; c++) {
            u8 *smPc = smP_w + c * 16 * 16;
            u8x8 ap_b;
            for (int i = 0; i < 8; i++) ap_b[i] = smPc[idx * 16 + half * 8 + i];
            u32x2 ap = pack_u8x8(ap_b);
            for (int kb = 0; kb < K_NB; kb++) {
                u8x8 bv;
                for (int i = 0; i < 8; i++) {
                    int d_col = kb * 16 + idx;
                    int kv_k = c * 16 + half * 8 + i;
                    bv[i] = smV[d_col * BKV + kv_k];
                }
                WMMA_FP8(ap, pack_u8x8(bv), O_acc[kb]);
            }
        }
        __syncthreads();
    }
    float inv_l[8];
    for (int i = 0; i < 8; i++) {
        float l = l_i[i];
        l += __shfl_xor(l, 1, 32);
        l += __shfl_xor(l, 2, 32);
        l += __shfl_xor(l, 4, 32);
        l += __shfl_xor(l, 8, 32);
        inv_l[i] = (l > 0.0f) ? 1.0f / l : 0.0f;
    }
    float vs = sV[h];
    for (int kb = 0; kb < K_NB; kb++) {
        for (int i = 0; i < 8; i++) {
            int row = q0 + wid * 16 + half * 8 + i;
            int d = kb * 16 + idx;
            if (row < n_tok) out[(size_t)row * dim + h * HD + d] = O_acc[kb][i] * inv_l[i] * vs;
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void fa3_pc_b32_8w(float *out, const u8 *Q, const u8 *K_t, const u8 *V_t,
                   const float *sQ, const float *sK, const float *sV,
                   int n_tok, int n_heads, float inv_sqrtd) {
    enum { BKV = 32, CH = 2, BQX = 64, PROD_WAVES = 4, WAVE_CT = 8 };
    int h = blockIdx.x;
    int qb = blockIdx.y;
    int q0 = qb * BQX;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int half = lid >> 4;
    int idx = lid & 15;
    int dim = n_heads * HD;
    int is_prod = (wid < PROD_WAVES);
    int role = is_prod ? wid : (wid - PROD_WAVES);

    __shared__ u8 smK[2 * BKV * HD];
    __shared__ u8 smV[2 * HD * BKV];
    __shared__ float smS[2 * PROD_WAVES * CH * 16 * 16];
    __shared__ u8 smP[PROD_WAVES * CH * 16 * 16];
    __shared__ float smSq[BQX];
    __shared__ float smSk[2 * BKV];

    for (int i = tid; i < BQX; i += WAVE_CT * 32) {
        int row = q0 + i;
        smSq[i] = (row < n_tok) ? sQ[(size_t)h * n_tok + row] : 1.0f;
    }

    u32x2 q_reg[K_NB];
    if (is_prod) {
        int q_row = q0 + role * 16 + idx;
        u8x8 tmp;
        for (int kb = 0; kb < K_NB; kb++) {
            for (int i = 0; i < 8; i++) {
                int d = kb * 16 + half * 8 + i;
                tmp[i] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (u8)0;
            }
            q_reg[kb] = pack_u8x8(tmp);
        }
    }
    __syncthreads();

    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    if (n_kv_tiles <= 0) return;

    /* Prologue: all waves load tile 0, producer waves compute tile 0 scores. */
    for (int e = tid; e < BKV * HD; e += WAVE_CT * 32) {
        int r = e / HD;
        int d = e - r * HD;
        int kv = r;
        size_t base = ((size_t)h * n_tok + kv) * HD;
        u8 kk = 0, vv = 0;
        if (kv < n_tok) { kk = K_t[base + d]; vv = V_t[base + d]; }
        smK[e] = kk;
        smV[d * BKV + r] = vv;
    }
    for (int i = tid; i < BKV; i += WAVE_CT * 32)
        smSk[i] = (i < n_tok) ? sK[(size_t)h * n_tok + i] : 1.0f;
    __syncthreads();

    if (is_prod) {
        float q_scale_lane[8];
        for (int i = 0; i < 8; i++) q_scale_lane[i] = smSq[role * 16 + half * 8 + i];
        float8 S[CH];
        for (int c = 0; c < CH; c++) for (int i = 0; i < 8; i++) S[c][i] = 0.0f;
        for (int kb = 0; kb < K_NB; kb++) {
            for (int c = 0; c < CH; c++) {
                u8x8 b_K;
                for (int i = 0; i < 8; i++) {
                    int dpos = kb * 16 + half * 8 + i;
                    b_K[i] = smK[(c * 16 + idx) * HD + dpos];
                }
                WMMA_FP8(q_reg[kb], pack_u8x8(b_K), S[c]);
            }
        }
        float *dst = smS + role * CH * 16 * 16;
        for (int c = 0; c < CH; c++) {
            for (int i = 0; i < 8; i++) {
                int col = c * 16 + idx;
                int kv = col;
                float s = S[c][i] * q_scale_lane[i] * smSk[col] * inv_sqrtd * LOG2E;
                if (kv >= n_tok) s = -1e30f;
                dst[c * 16 * 16 + (half * 8 + i) * 16 + idx] = s;
            }
        }
    }
    __syncthreads();

    float8 O_acc[K_NB];
    float m_i[8], l_i[8];
    if (!is_prod) {
        for (int kb = 0; kb < K_NB; kb++) for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;
        for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }
    }

    for (int t = 0; t < n_kv_tiles; t++) {
        int cur = t & 1;
        int nxt = cur ^ 1;
        int kv0_cur = t * BKV;
        int kv0_next = (t + 1) * BKV;

        if (is_prod && t + 1 < n_kv_tiles) {
            u8 *nK = smK + nxt * BKV * HD;
            u8 *nV = smV + nxt * HD * BKV;
            float *nSk = smSk + nxt * BKV;
            for (int e = lid; e < BKV * HD; e += 32) {
                int r = e / HD;
                int d = e - r * HD;
                int kv = kv0_next + r;
                size_t base = ((size_t)h * n_tok + kv) * HD;
                u8 kk = 0, vv = 0;
                if (kv < n_tok) { kk = K_t[base + d]; vv = V_t[base + d]; }
                nK[r * HD + d] = kk;
                nV[d * BKV + r] = vv;
            }
            for (int i = lid; i < BKV; i += 32) {
                int kv = kv0_next + i;
                nSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f;
            }
            __builtin_amdgcn_s_waitcnt(0);
            float q_scale_lane[8];
            for (int i = 0; i < 8; i++) q_scale_lane[i] = smSq[role * 16 + half * 8 + i];
            float8 Snext[CH];
            for (int c = 0; c < CH; c++) for (int i = 0; i < 8; i++) Snext[c][i] = 0.0f;
            for (int kb = 0; kb < K_NB; kb++) {
                for (int c = 0; c < CH; c++) {
                    u8x8 b_K;
                    for (int i = 0; i < 8; i++) {
                        int dpos = kb * 16 + half * 8 + i;
                        b_K[i] = nK[(c * 16 + idx) * HD + dpos];
                    }
                    WMMA_FP8(q_reg[kb], pack_u8x8(b_K), Snext[c]);
                }
            }
            float *dst = smS + nxt * PROD_WAVES * CH * 16 * 16 + role * CH * 16 * 16;
            for (int c = 0; c < CH; c++) {
                for (int i = 0; i < 8; i++) {
                    int col = c * 16 + idx;
                    int kv = kv0_next + col;
                    float s = Snext[c][i] * q_scale_lane[i] * nSk[col] * inv_sqrtd * LOG2E;
                    if (kv >= n_tok) s = -1e30f;
                    dst[c * 16 * 16 + (half * 8 + i) * 16 + idx] = s;
                }
            }
        }

        if (!is_prod) {
            float *src = smS + cur * PROD_WAVES * CH * 16 * 16 + role * CH * 16 * 16;
            u8 *curV = smV + cur * HD * BKV;
            float8 S[CH];
            for (int c = 0; c < CH; c++) {
                for (int i = 0; i < 8; i++)
                    S[c][i] = src[c * 16 * 16 + (half * 8 + i) * 16 + idx];
            }
            float row_max[8];
            for (int i = 0; i < 8; i++) {
                float mx = fmaxf(S[0][i], S[1][i]);
                mx = fmaxf(mx, __shfl_xor(mx, 1, 32));
                mx = fmaxf(mx, __shfl_xor(mx, 2, 32));
                mx = fmaxf(mx, __shfl_xor(mx, 4, 32));
                mx = fmaxf(mx, __shfl_xor(mx, 8, 32));
                row_max[i] = mx;
            }
            float alpha[8];
            for (int i = 0; i < 8; i++) {
                float new_max = fmaxf(m_i[i], row_max[i]);
                alpha[i] = exp2f(m_i[i] - new_max);
                float local = 0.0f;
                for (int c = 0; c < CH; c++) {
                    float p = exp2f(S[c][i] - new_max + 8.0f);
                    S[c][i] = p;
                    local += p;
                }
                l_i[i] = l_i[i] * alpha[i] + local;
                m_i[i] = new_max;
            }
            for (int kb = 0; kb < K_NB; kb++) for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];
            u8 *myP = smP + role * CH * 16 * 16;
            for (int c = 0; c < CH; c++) {
                u8 *pc = myP + c * 16 * 16;
                for (int i = 0; i < 8; i++) pc[(half * 8 + i) * 16 + idx] = f32_to_fp8_e4m3(S[c][i]);
            }
            __builtin_amdgcn_s_waitcnt(0);
            for (int c = 0; c < CH; c++) {
                u8 *pc = myP + c * 16 * 16;
                u8x8 ap_b;
                for (int i = 0; i < 8; i++) ap_b[i] = pc[idx * 16 + half * 8 + i];
                u32x2 ap = pack_u8x8(ap_b);
                for (int kb = 0; kb < K_NB; kb++) {
                    u8x8 bv;
                    for (int i = 0; i < 8; i++) {
                        int d_col = kb * 16 + idx;
                        int kv_k = c * 16 + half * 8 + i;
                        bv[i] = curV[d_col * BKV + kv_k];
                    }
                    WMMA_FP8(ap, pack_u8x8(bv), O_acc[kb]);
                }
            }
        }

        __syncthreads();
        (void)kv0_cur;
    }

    if (!is_prod) {
        float inv_l[8];
        for (int i = 0; i < 8; i++) {
            float l = l_i[i];
            l += __shfl_xor(l, 1, 32);
            l += __shfl_xor(l, 2, 32);
            l += __shfl_xor(l, 4, 32);
            l += __shfl_xor(l, 8, 32);
            inv_l[i] = (l > 0.0f) ? 1.0f / l : 0.0f;
        }
        float vs = sV[h];
        for (int kb = 0; kb < K_NB; kb++) {
            for (int i = 0; i < 8; i++) {
                int row = q0 + role * 16 + half * 8 + i;
                int d = kb * 16 + idx;
                if (row < n_tok) out[(size_t)row * dim + h * HD + d] = O_acc[kb][i] * inv_l[i] * vs;
            }
        }
    }
}
)FA3SRC";

typedef struct {
    hipFunction_t exact_b16;
    hipFunction_t approx_b32_fast;
    hipFunction_t approx_b32_exp2;
    hipFunction_t approx_b64_fast;
    hipFunction_t approx_b32_fast_8w;
    hipFunction_t approx_b32_exp2_8w;
    hipFunction_t approx_b32_exp2_8w_db;
    hipFunction_t approx_b64_fast_8w;
    hipFunction_t approx_b32_fast_16w;
    hipFunction_t approx_b32_fast_32w;
    hipFunction_t approx_b32_fast_16w_qg2;
    hipFunction_t approx_b64_fast_16w;
    hipFunction_t roof_16w;
    hipFunction_t roof_4h;
    hipFunction_t pc_b32_8w;
    hipFunction_t shflp_8w;
    hipFunction_t shflp_16w;
    hipFunction_t two_pass_b32_8w_max;
    hipFunction_t two_pass_b32_8w_out;
} fa3_kernels;

static int load_kernels(hipModule_t mod, fa3_kernels *k) {
    HIP_CHECK(hipModuleGetFunction(&k->exact_b16,       mod, "fa3_exact_b16"));
    HIP_CHECK(hipModuleGetFunction(&k->approx_b32_fast, mod, "fa3_approx_b32_fast"));
    HIP_CHECK(hipModuleGetFunction(&k->approx_b32_exp2, mod, "fa3_approx_b32_exp2"));
    HIP_CHECK(hipModuleGetFunction(&k->approx_b64_fast, mod, "fa3_approx_b64_fast"));
    HIP_CHECK(hipModuleGetFunction(&k->approx_b32_fast_8w, mod, "fa3_approx_b32_fast_8w"));
    HIP_CHECK(hipModuleGetFunction(&k->approx_b32_exp2_8w, mod, "fa3_approx_b32_exp2_8w"));
    HIP_CHECK(hipModuleGetFunction(&k->approx_b32_exp2_8w_db, mod, "fa3_approx_b32_exp2_8w_db"));
    HIP_CHECK(hipModuleGetFunction(&k->approx_b64_fast_8w, mod, "fa3_approx_b64_fast_8w"));
    HIP_CHECK(hipModuleGetFunction(&k->approx_b32_fast_16w, mod, "fa3_approx_b32_fast_16w"));
    HIP_CHECK(hipModuleGetFunction(&k->approx_b32_fast_32w, mod, "fa3_approx_b32_fast_32w"));
    HIP_CHECK(hipModuleGetFunction(&k->approx_b32_fast_16w_qg2, mod, "fa3_approx_b32_fast_16w_qg2"));
    HIP_CHECK(hipModuleGetFunction(&k->approx_b64_fast_16w, mod, "fa3_approx_b64_fast_16w"));
    HIP_CHECK(hipModuleGetFunction(&k->roof_16w, mod, "fa3_roof_16w"));
    HIP_CHECK(hipModuleGetFunction(&k->roof_4h, mod, "fa3_roof_4h"));
    HIP_CHECK(hipModuleGetFunction(&k->pc_b32_8w, mod, "fa3_pc_b32_8w"));
    HIP_CHECK(hipModuleGetFunction(&k->shflp_8w, mod, "fa3_approx_b32_8w_shflp"));
    HIP_CHECK(hipModuleGetFunction(&k->shflp_16w, mod, "fa3_approx_b32_16w_shflp"));
    HIP_CHECK(hipModuleGetFunction(&k->two_pass_b32_8w_max, mod, "fa3_two_pass_b32_8w_max"));
    HIP_CHECK(hipModuleGetFunction(&k->two_pass_b32_8w_out, mod, "fa3_two_pass_b32_8w_out"));
    return 0;
}

static hipFunction_t select_kernel(const fa3_kernels *k, const char *mode) {
    if (!strcmp(mode, "exact_b16")) return k->exact_b16;
    if (!strcmp(mode, "approx_b32")) return k->approx_b32_fast;
    if (!strcmp(mode, "approx_b32_exp2")) return k->approx_b32_exp2;
    if (!strcmp(mode, "approx_b64")) return k->approx_b64_fast;
    if (!strcmp(mode, "approx_b32_8w_fast")) return k->approx_b32_fast_8w;
    if (!strcmp(mode, "approx_b32_8w")) return k->approx_b32_exp2_8w;
    if (!strcmp(mode, "approx_b32_8w_db")) return k->approx_b32_exp2_8w_db;
    if (!strcmp(mode, "approx_b64_8w_fast")) return k->approx_b64_fast_8w;
    if (!strcmp(mode, "approx_b32_16w_fast")) return k->approx_b32_fast_16w;
    if (!strcmp(mode, "approx_b32_32w_fast")) return k->approx_b32_fast_32w;
    if (!strcmp(mode, "approx_b32_16w_qg2_fast")) return k->approx_b32_fast_16w_qg2;
    if (!strcmp(mode, "b64_16w")) return k->approx_b64_fast_16w;
    if (!strcmp(mode, "roof_16w")) return k->roof_16w;
    if (!strcmp(mode, "roof_4h")) return k->roof_4h;
    if (!strcmp(mode, "pc_b32_8w")) return k->pc_b32_8w;
    if (!strcmp(mode, "shflp_8w")) return k->shflp_8w;
    if (!strcmp(mode, "shflp_16w")) return k->shflp_16w;
    return NULL;
}

static int is_two_pass_mode(const char *mode) {
    return !strcmp(mode, "two_pass_b32_8w");
}

static int run_shape(const fa3_kernels *kernels, const char *mode, int n_tok,
                     int n_heads, int iters, int do_check, float abs_max,
                     double peak_tfs) {
    int two_pass = is_two_pass_mode(mode);
    hipFunction_t fn = two_pass ? kernels->two_pass_b32_8w_out : select_kernel(kernels, mode);
    if (!fn || (two_pass && !kernels->two_pass_b32_8w_max)) {
        fprintf(stderr, "unknown mode '%s'\n", mode);
        return -1;
    }
    if (n_tok % 64) {
        fprintf(stderr, "skip: n_tok must be %%64 (got %d)\n", n_tok);
        return 0;
    }
    const int head_dim = 128;
    int dim = n_heads * head_dim;
    size_t qkv_elems = (size_t)n_tok * dim;
    size_t qkv_bytes = qkv_elems;
    size_t out_elems = (size_t)n_tok * dim;
    size_t out_bytes = out_elems * sizeof(float);
    size_t row_scale_elems = (size_t)n_heads * n_tok;

    uint8_t *hQ  = (uint8_t *)malloc(qkv_bytes);
    uint8_t *hKt = (uint8_t *)malloc(qkv_bytes);
    uint8_t *hVt = (uint8_t *)malloc(qkv_bytes);
    float *hSq = (float *)malloc(row_scale_elems * sizeof(float));
    float *hSk = (float *)malloc(row_scale_elems * sizeof(float));
    float *hSv = (float *)malloc((size_t)n_heads * sizeof(float));
    if (!hQ || !hKt || !hVt || !hSq || !hSk || !hSv) {
        fprintf(stderr, "host allocation failed\n");
        return -1;
    }
    unsigned rng = 0x1234abcd;
    fill_fp8_random(hQ,  qkv_elems, abs_max, &rng);
    fill_fp8_random(hKt, qkv_elems, abs_max, &rng);
    fill_fp8_random(hVt, qkv_elems, abs_max, &rng);
    fill_ones(hSq, row_scale_elems);
    fill_ones(hSk, row_scale_elems);
    fill_ones(hSv, (size_t)n_heads);

    void *dQ = hip_upload_raw(hQ, qkv_bytes);
    void *dK = hip_upload_raw(hKt, qkv_bytes);
    void *dV = hip_upload_raw(hVt, qkv_bytes);
    void *dSq = hip_upload_raw(hSq, row_scale_elems * sizeof(float));
    void *dSk = hip_upload_raw(hSk, row_scale_elems * sizeof(float));
    void *dSv = hip_upload_raw(hSv, (size_t)n_heads * sizeof(float));
    void *dO = NULL;
    void *dM = NULL;
    HIP_CHECK(hipMalloc(&dO, out_bytes));
    HIP_CHECK(hipMemset(dO, 0, out_bytes));
    if (two_pass) HIP_CHECK(hipMalloc(&dM, row_scale_elems * sizeof(float)));

    float inv_sqrtd = 1.0f / sqrtf((float)head_dim);
    void *args[] = { &dO, &dQ, &dK, &dV, &dSq, &dSk, &dSv, &n_tok, &n_heads, &inv_sqrtd };
    void *args_max[] = { &dM, &dQ, &dK, &dSq, &dSk, &n_tok, &n_heads, &inv_sqrtd };
    void *args_out[] = { &dO, &dQ, &dK, &dV, &dSq, &dSk, &dSv, &dM, &n_tok, &n_heads, &inv_sqrtd };
    int use_32w = !strcmp(mode, "approx_b32_32w_fast");
    int use_16w_qg2 = !strcmp(mode, "approx_b32_16w_qg2_fast");
    int use_16w = !strcmp(mode, "approx_b32_16w_fast") || use_16w_qg2 || !strcmp(mode, "shflp_16w") || !strcmp(mode, "b64_16w") || !strcmp(mode, "roof_16w");
    int use_8w = !strcmp(mode, "approx_b32_8w_fast") || !strcmp(mode, "approx_b32_8w") || !strcmp(mode, "approx_b32_8w_db") || !strcmp(mode, "approx_b64_8w_fast") || !strcmp(mode, "pc_b32_8w") || !strcmp(mode, "shflp_8w") || !strcmp(mode, "roof_4h") || two_pass;
    int block_threads = use_32w ? 1024 : (use_16w ? 512 : (use_8w ? 256 : 128));
    int bq = (use_32w || use_16w_qg2) ? 512 : (use_16w ? 256 : ((!strcmp(mode, "approx_b32_8w_fast") || !strcmp(mode, "approx_b32_8w") || !strcmp(mode, "approx_b32_8w_db") || !strcmp(mode, "approx_b64_8w_fast") || !strcmp(mode, "shflp_8w") || !strcmp(mode, "roof_4h") || two_pass) ? 128 : 64));
    dim3 block((unsigned)block_threads, 1, 1);
    int gx = !strcmp(mode,"roof_4h") ? n_heads/2 : n_heads;
    dim3 grid((unsigned)gx, (unsigned)((n_tok + bq - 1) / bq), 1);

    if (two_pass) {
        HIP_CHECK(hipModuleLaunchKernel(kernels->two_pass_b32_8w_max, grid.x, grid.y, grid.z,
                                        block.x, block.y, block.z, 0, NULL, args_max, NULL));
        HIP_CHECK(hipModuleLaunchKernel(kernels->two_pass_b32_8w_out, grid.x, grid.y, grid.z,
                                        block.x, block.y, block.z, 0, NULL, args_out, NULL));
    } else {
        HIP_CHECK(hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z,
                                        block.x, block.y, block.z, 0, NULL, args, NULL));
    }
    HIP_CHECK(hipDeviceSynchronize());

    double cos = -2.0;
    float maxd = -1.0f;
    if (do_check) {
        float *hO = (float *)malloc(out_bytes);
        float *hRef = (float *)malloc(out_bytes);
        HIP_CHECK(hipMemcpy(hO, dO, out_bytes, hipMemcpyDeviceToHost));
        fa_ref_fp32(hRef, hQ, hKt, hVt, hSq, hSk, hSv, n_tok, n_heads, head_dim);
        cos = cosine_sim(hO, hRef, out_elems);
        maxd = max_abs_diff(hO, hRef, out_elems);
        free(hO);
        free(hRef);
    }

    double t0 = timer_ms();
    for (int i = 0; i < iters; i++) {
        if (two_pass) {
            HIP_CHECK(hipModuleLaunchKernel(kernels->two_pass_b32_8w_max, grid.x, grid.y, grid.z,
                                            block.x, block.y, block.z, 0, NULL, args_max, NULL));
            HIP_CHECK(hipModuleLaunchKernel(kernels->two_pass_b32_8w_out, grid.x, grid.y, grid.z,
                                            block.x, block.y, block.z, 0, NULL, args_out, NULL));
        } else {
            HIP_CHECK(hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z,
                                            block.x, block.y, block.z, 0, NULL, args, NULL));
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
    double t1 = timer_ms();
    double ms = (t1 - t0) / (double)iters;
    double flops = 4.0 * (double)n_heads * (double)n_tok * (double)n_tok * (double)head_dim;
    double tflops = flops / (ms * 1.0e-3) * 1.0e-12;
    double pct = peak_tfs > 0.0 ? 100.0 * tflops / peak_tfs : 0.0;

    printf("  [%-15s] S=%5d H=%2d HD=128  %8.4f ms  %7.2f TF/s  %5.1f%% peak",
           mode, n_tok, n_heads, ms, tflops, pct);
    if (do_check) printf("  cos=%.6f  maxd=%.5f", cos, maxd);
    printf("\n");

    hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dSq); hipFree(dSk); hipFree(dSv); hipFree(dO);
    if (dM) hipFree(dM);
    free(hQ); free(hKt); free(hVt); free(hSq); free(hSk); free(hSv);
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [--n-tok N] [--heads H] [--iters N] [--mode MODE]\n"
        "          [--check] [--abs-max V] [--peak-tfs V] [--verbose] [--dump-code]\n"
        "  --mode      exact_b16 | approx_b32 | approx_b32_exp2 | approx_b64 | approx_b32_8w_fast | approx_b32_8w | approx_b32_8w_db | approx_b64_8w_fast | approx_b32_16w_fast | approx_b32_32w_fast | approx_b32_16w_qg2_fast | pc_b32_8w | two_pass_b32_8w | all\n"
        "  --n-tok     token count, must be %%64 (default 1024)\n"
        "  --heads     attention heads (default 16)\n"
        "  --iters     timing iterations (default 100)\n"
        "  --check     compare against CPU FP32 reference (slow)\n"
        "  --abs-max   random input clamp before FP8 quantization (default 1.0)\n"
        "  --peak-tfs  peak denominator for percent reporting (default 342.9)\n",
        prog);
}

int main(int argc, char **argv) {
    int n_tok = 1024;
    int n_heads = 16;
    int iters = 100;
    int do_check = 0;
    int verbose = 1;
    float abs_max = 1.0f;
    double peak_tfs = 342.9;
    const char *mode = "all";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-tok") && i + 1 < argc) n_tok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads") && i + 1 < argc) n_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--abs-max") && i + 1 < argc) abs_max = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--peak-tfs") && i + 1 < argc) peak_tfs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--check")) do_check = 1;
        else if (!strcmp(argv[i], "--verbose")) verbose = 2;
        else if (!strcmp(argv[i], "--dump-code")) verbose = 3;
        else { usage(argv[0]); return 1; }
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "failed to load HIP runtime\n");
        return 1;
    }
    int dev = 0;
    HIP_CHECK(hipSetDevice(dev));

    hipModule_t mod;
    if (hip_compile_kernels(&mod, dev, kernel_src, "fa3_fp8", verbose, "fa3") < 0) return 1;
    fa3_kernels kernels;
    if (load_kernels(mod, &kernels) < 0) return 1;

    printf("rdna4/fa3 FP8 FA3-style attention bench (iters=%d, abs_max=%.2f, peak=%.1f TF/s%s)\n",
           iters, abs_max, peak_tfs, do_check ? ", check=on" : "");
    const char *all_modes[] = {"exact_b16", "approx_b32", "approx_b32_exp2", "approx_b64", "approx_b32_8w_fast", "approx_b32_8w", "approx_b32_8w_db", "approx_b64_8w_fast", "approx_b32_16w_fast", "approx_b32_32w_fast", "approx_b32_16w_qg2_fast", "pc_b32_8w", "two_pass_b32_8w"};
    if (!strcmp(mode, "all")) {
        for (size_t i = 0; i < sizeof(all_modes) / sizeof(all_modes[0]); i++) {
            if (run_shape(&kernels, all_modes[i], n_tok, n_heads, iters, do_check, abs_max, peak_tfs) < 0)
                return 1;
        }
    } else {
        if (run_shape(&kernels, mode, n_tok, n_heads, iters, do_check, abs_max, peak_tfs) < 0)
            return 1;
    }
    return 0;
}
