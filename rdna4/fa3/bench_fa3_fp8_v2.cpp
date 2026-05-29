/*
 * bench_fa3_fp8_v2.cpp - RDNA4/gfx1201 FP8 flash-attention v2 design.
 *
 * Four-phase kernel progression targeting 245 TF/s (70% of 350 TF/s peak).
 *
 * V1: BQ=128, BKV=64, 8-wave WG, VWA=4 Q packing, vectorized global K/V
 *     loads, vectorized LDS P-transpose, log2-domain softmax.
 * V2: V1 + PGR=2 double-buffered K/V, LBSPP sweep, SVW=4, ONLL tail.
 * V3: V2 + warp-specialized producer/consumer pipeline.
 * V4: V3 + deferred softmax, half-wave shuffle, FP8 output.
 *
 * Build: make -C rdna4/fa3 bench_fa3_fp8_v2
 * Run: rdna4/fa3/bench_fa3_fp8_v2 --mode all --n-tok 4096 --heads 16 --iters 100
 * Check: rdna4/fa3/bench_fa3_fp8_v2 --mode v1 --n-tok 256 --heads 1 --iters 20 --check
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

/* ======================================================================== */
/* Kernel source string — HIPRTC-compiled at startup.                       */
/* ======================================================================== */

static const char *kernel_src = R"FA3V2SRC(
typedef unsigned char u8;
typedef unsigned int  u32;
typedef u32 u32x2 __attribute__((ext_vector_type(2)));
typedef u8  u8x8  __attribute__((ext_vector_type(8)));
typedef u8  u8x16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

#define HD   128
#define K_NB (HD / 16)
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ u32x2 pack_u8x8(u8x8 v) {
    union { u8x8 b; u32x2 w; } u;
    u.b = v;
    return u.w;
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

#define WMMA_FP8(A,B,C) C = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(A, B, C)

/* ----------------------------------------------------------------------- */
/* V1: BQ=128, BKV=64, 8-wave WG, VWA=4 Q packing, vectorized loads.      */
/* ----------------------------------------------------------------------- */
extern "C" __global__ __launch_bounds__(256, 1)
void fa3_v1_b64_8w(float *out, const u8 *Q, const u8 *K_t, const u8 *V_t,
                   const float *sQ, const float *sK, const float *sV,
                   int n_tok, int n_heads, float inv_sqrtd) {
    enum { BQ = 128, BKV = 64, WAVE_CT = 8, CH = 4 };
    int h  = blockIdx.x;
    int qb = blockIdx.y;
    int q0 = qb * BQ;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int half = lid >> 4;
    int idx = lid & 15;
    int dim = n_heads * HD;

    __shared__ u8 smK[BKV * HD];
    __shared__ u8 smV[HD * BKV];
    __shared__ u8 smP[WAVE_CT * CH * 16 * 16];
    __shared__ float smSq[BQ];
    __shared__ float smSk[BKV];
    u8 *smP_w = smP + wid * CH * 16 * 16;

    /* Load Q scale factors */
    for (int i = tid; i < BQ; i += WAVE_CT * 32) {
        int row = q0 + i;
        smSq[i] = (row < n_tok) ? sQ[(size_t)h * n_tok + row] : 1.0f;
    }
    __syncthreads();

    /* Load Q with VWA=4: 2 groups x 4 consecutive K-blocks per group.
     * q_reg[g][i] = Q[tid holds 8 FP8 values at HD block (g*4+i)]. */
    u32x2 q_reg[2][4];
    for (int g = 0; g < 2; g++) {
        int q_row = q0 + wid * 16 + idx;
        u8x8 tmp;
        for (int i = 0; i < 4; i++) {
            int kb = g * 4 + i;
            for (int j = 0; j < 8; j++) {
                int d = kb * 16 + half * 8 + j;
                tmp[j] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (u8)0;
            }
            q_reg[g][i] = pack_u8x8(tmp);
        }
    }

    /* Precompute per-lane Q scale factor (constant across all tiles). */
    float q_scale_lane[8];
    for (int i = 0; i < 8; i++) {
        int m_row = wid * 16 + half * 8 + i;
        q_scale_lane[i] = smSq[m_row];
    }

    /* Initialize output accumulators and online-softmax state. */
    float8 O_acc[K_NB];
    for (int kb = 0; kb < K_NB; kb++)
        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;
    float m_i[8], l_i[8];
    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }

    int total = BKV * HD;
    int load_stride = (WAVE_CT * 32) * 16;
    int n_kv_tiles = (n_tok + BKV - 1) / BKV;

    for (int t = 0; t < n_kv_tiles; t++) {
        int kv0 = t * BKV;

        /* --- Vectorized K/V load: uint8x16 global loads ---
         * Each thread loads 32 bytes of K and 32 bytes of V per tile,
         * in 2 passes of 16 bytes each. */
        for (int off = tid * 16; off < total; off += load_stride) {
            int r = off / HD;
            int d = off % HD;
            int kv = kv0 + r;
            size_t base = ((size_t)h * n_tok + kv) * HD + d;
            u8x16 k_data;
            u8x16 v_data;
            int kv_ok = (kv < n_tok);
            int in_bounds = (kv_ok && d + 16 <= HD);
            if (in_bounds) {
                k_data = *(const u8x16 *)(K_t + base);
                v_data = *(const u8x16 *)(V_t + base);
            } else {
                for (int i = 0; i < 16; i++) {
                    u8 val_k = 0, val_v = 0;
                    if (kv_ok && d + i < HD) {
                        val_k = K_t[base + i];
                        val_v = V_t[base + i];
                    }
                    k_data[i] = val_k;
                    v_data[i] = val_v;
                }
            }
            /* K: row-major LDS store (vectorized). */
            *(u8x16 *)(smK + r * HD + d) = k_data;
            /* V: transposed LDS store (individual bytes).
             * smV[d_col * BKV + r] = V[kv][d_col] */
            for (int i = 0; i < 16; i++)
                smV[(d + i) * BKV + r] = v_data[i];
        }

        /* Load K scale factors for this tile. */
        for (int i = tid; i < BKV; i += WAVE_CT * 32) {
            int kv = kv0 + i;
            smSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f;
        }
        __syncthreads();

        /* ---- QK phase with VWA=4 dual-issue ---- */
        float8 S[CH];
        for (int ch = 0; ch < CH; ch++)
            for (int i = 0; i < 8; i++) S[ch][i] = 0.0f;

        for (int g = 0; g < 2; g++) {
            for (int ch = 0; ch < CH; ch++) {
                u32x2 k_reg[4];
                for (int i = 0; i < 4; i++) {
                    int kb = g * 4 + i;
                    u8x8 b_K;
                    for (int j = 0; j < 8; j++) {
                        int dpos = kb * 16 + half * 8 + j;
                        b_K[j] = smK[(ch * 16 + idx) * HD + dpos];
                    }
                    k_reg[i] = pack_u8x8(b_K);
                }
                /* 4 consecutive WMMAs with different A operands -> dual-issue. */
                for (int i = 0; i < 4; i++) {
                    int kb = g * 4 + i;
                    WMMA_FP8(q_reg[g][i], k_reg[i], S[ch]);
                }
            }
        }

        /* ---- Online softmax (log2-domain, fast exp2) ---- */
        float row_max[8];
        for (int i = 0; i < 8; i++) {
            float mx = -1e30f;
            for (int ch = 0; ch < CH; ch++) {
                int col = ch * 16 + idx;
                int kv = kv0 + col;
                float s = S[ch][i] * q_scale_lane[i] * smSk[col] * inv_sqrtd * LOG2E;
                if (kv >= n_tok) s = -1e30f;
                S[ch][i] = s;
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
            alpha[i] = exp2_fast_bits(m_i[i] - new_max);
            float local = 0.0f;
            for (int ch = 0; ch < CH; ch++) {
                int col = ch * 16 + idx;
                int kv = kv0 + col;
                float p = exp2_fast_bits(S[ch][i] - new_max + 8.0f);
                if (kv >= n_tok) p = 0.0f;
                S[ch][i] = p;
                local += p;
            }
            l_i[i] = l_i[i] * alpha[i] + local;
            m_i[i] = new_max;
        }

        for (int kb = 0; kb < K_NB; kb++)
            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];

        /* ---- P write to LDS (column-major transpose, individual bytes) ---- */
        for (int ch = 0; ch < CH; ch++) {
            u8 *smPc = smP_w + ch * 16 * 16;
            for (int i = 0; i < 8; i++) {
                int m_row = half * 8 + i;
                smPc[m_row * 16 + idx] = f32_to_fp8_e4m3_pos(S[ch][i]);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);

        /* ---- P read (transposed, vectorized u8x8 -> ds_read_b64) + PV ---- */
        for (int ch = 0; ch < CH; ch++) {
            u8 *smPc = smP_w + ch * 16 * 16;
            u8x8 ap_b = *(const u8x8 *)(smPc + idx * 16 + half * 8);
            u32x2 ap = pack_u8x8(ap_b);
            for (int kb = 0; kb < K_NB; kb++) {
                u8x8 bv;
                for (int i = 0; i < 8; i++) {
                    int d_col = kb * 16 + idx;
                    int kv_k = ch * 16 + half * 8 + i;
                    bv[i] = smV[d_col * BKV + kv_k];
                }
                u32x2 bv_p = pack_u8x8(bv);
                WMMA_FP8(ap, bv_p, O_acc[kb]);
            }
        }
        __syncthreads();
    }

    /* ---- Epilogue: normalize by l_i, scale by sV, write FP32 output ---- */
    float vs = sV[h];
    for (int i = 0; i < 8; i++) {
        float l = l_i[i];
        l += __shfl_xor(l, 1, 32);
        l += __shfl_xor(l, 2, 32);
        l += __shfl_xor(l, 4, 32);
        l += __shfl_xor(l, 8, 32);
        l_i[i] = (l > 0.0f) ? 1.0f / l : 0.0f;
    }
    for (int kb = 0; kb < K_NB; kb++) {
        for (int i = 0; i < 8; i++) {
            int row = q0 + wid * 16 + half * 8 + i;
            int d = kb * 16 + idx;
            if (row < n_tok)
                out[(size_t)row * dim + h * HD + d] = O_acc[kb][i] * l_i[i] * vs;
        }
    }
}

/* ----------------------------------------------------------------------- */
/* V2: BQ=256, BKV=32, 16-wave WG, VWA=4, vectorized loads, PGR=2 DB.     */
/* ----------------------------------------------------------------------- */
extern "C" __global__ __launch_bounds__(512, 1)
void fa3_v2_b32_16w_pgr2(float *out, const u8 *Q, const u8 *K_t, const u8 *V_t,
                         const float *sQ, const float *sK, const float *sV,
                         int n_tok, int n_heads, float inv_sqrtd) {
    enum { BQ = 256, BKV = 32, WAVE_CT = 16, CH = 2, DBUF = 2 };
    int h  = blockIdx.x;
    int qb = blockIdx.y;
    int q0 = qb * BQ;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int half = lid >> 4;
    int idx = lid & 15;
    int dim = n_heads * HD;

    __shared__ u8 smK[DBUF][BKV * HD];
    __shared__ u8 smV[DBUF][HD * BKV];
    __shared__ u8 smP[WAVE_CT * CH * 16 * 16];
    __shared__ float smSq[BQ];
    __shared__ float smSk[DBUF][BKV];
    u8 *smP_w = smP + wid * CH * 16 * 16;

    /* Load Q scale factors */
    for (int i = tid; i < BQ; i += WAVE_CT * 32) {
        int row = q0 + i;
        smSq[i] = (row < n_tok) ? sQ[(size_t)h * n_tok + row] : 1.0f;
    }
    __syncthreads();

    /* Load Q with VWA=4: 2 groups x 4 consecutive K-blocks per group. */
    u32x2 q_reg[2][4];
    for (int g = 0; g < 2; g++) {
        u8x8 tmp;
        for (int i = 0; i < 4; i++) {
            int kb = g * 4 + i;
            int q_row = q0 + wid * 16 + idx;
            for (int j = 0; j < 8; j++) {
                int d = kb * 16 + half * 8 + j;
                tmp[j] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (u8)0;
            }
            q_reg[g][i] = pack_u8x8(tmp);
        }
    }

    /* Precompute per-lane Q scale factor. */
    float q_scale_lane[8];
    for (int i = 0; i < 8; i++) {
        int m_row = wid * 16 + half * 8 + i;
        q_scale_lane[i] = smSq[m_row];
    }

    /* Prefetch first KV tile into buffer 0. */
    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    int total = BKV * HD;
    int load_stride = (WAVE_CT * 32) * 16;

    auto load_kv_tile = [&](u8 *dstK, u8 *dstV, float *dstSk, int kv_base) {
        for (int off = tid * 16; off < total; off += load_stride) {
            int r = off / HD;
            int d = off % HD;
            int kv = kv_base + r;
            size_t base = ((size_t)h * n_tok + kv) * HD + d;
            int kv_ok = (kv < n_tok);
            int in_bounds = (kv_ok && d + 16 <= HD);
            if (in_bounds) {
                u8x16 kd = *(const u8x16 *)(K_t + base);
                u8x16 vd = *(const u8x16 *)(V_t + base);
                *(u8x16 *)(dstK + r * HD + d) = kd;
                for (int i = 0; i < 16; i++)
                    dstV[(d + i) * BKV + r] = vd[i];
            } else {
                for (int i = 0; i < 16; i++) {
                    u8 kk = 0, vv = 0;
                    if (kv_ok && d + i < HD) {
                        kk = K_t[base + i];
                        vv = V_t[base + i];
                    }
                    dstK[r * HD + d + i] = kk;
                    dstV[(d + i) * BKV + r] = vv;
                }
            }
        }
        for (int i = tid; i < BKV; i += WAVE_CT * 32) {
            int kv = kv_base + i;
            dstSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f;
        }
    };

    load_kv_tile(smK[0], smV[0], smSk[0], 0);
    __syncthreads();

    /* Initialize output accumulators and online-softmax state. */
    float8 O_acc[K_NB];
    for (int kb = 0; kb < K_NB; kb++)
        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;
    float m_i[8], l_i[8];
    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }

    for (int t = 0; t < n_kv_tiles; t++) {
        int cur = t & 1;
        int nxt = cur ^ 1;
        int kv0 = t * BKV;
        u8 *curK = smK[cur];
        u8 *curV = smV[cur];
        float *curSk = smSk[cur];

        /* Prefetch next tile into nxt buffer (overlaps with compute). */
        if (t + 1 < n_kv_tiles)
            load_kv_tile(smK[nxt], smV[nxt], smSk[nxt], (t + 1) * BKV);

        /* ---- QK phase with VWA=4 dual-issue ---- */
        float8 S[CH];
        for (int ch = 0; ch < CH; ch++)
            for (int i = 0; i < 8; i++) S[ch][i] = 0.0f;

        for (int g = 0; g < 2; g++) {
            for (int ch = 0; ch < CH; ch++) {
                u32x2 k_reg[4];
                for (int i = 0; i < 4; i++) {
                    int kb = g * 4 + i;
                    u8x8 b_K;
                    for (int j = 0; j < 8; j++) {
                        int dpos = kb * 16 + half * 8 + j;
                        b_K[j] = curK[(ch * 16 + idx) * HD + dpos];
                    }
                    k_reg[i] = pack_u8x8(b_K);
                }
                for (int i = 0; i < 4; i++)
                    WMMA_FP8(q_reg[g][i], k_reg[i], S[ch]);
            }
        }

        /* ---- Online softmax (log2-domain, fast exp2) ---- */
        float row_max[8];
        for (int i = 0; i < 8; i++) {
            float mx = -1e30f;
            for (int ch = 0; ch < CH; ch++) {
                int col = ch * 16 + idx;
                int kv = kv0 + col;
                float s = S[ch][i] * q_scale_lane[i] * curSk[col] * inv_sqrtd * LOG2E;
                if (kv >= n_tok) s = -1e30f;
                S[ch][i] = s;
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
            alpha[i] = exp2_fast_bits(m_i[i] - new_max);
            float local = 0.0f;
            for (int ch = 0; ch < CH; ch++) {
                int col = ch * 16 + idx;
                int kv = kv0 + col;
                float p = exp2_fast_bits(S[ch][i] - new_max + 8.0f);
                if (kv >= n_tok) p = 0.0f;
                S[ch][i] = p;
                local += p;
            }
            l_i[i] = l_i[i] * alpha[i] + local;
            m_i[i] = new_max;
        }

        for (int kb = 0; kb < K_NB; kb++)
            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];

        /* ---- P write to LDS ---- */
        {
            for (int ch = 0; ch < CH; ch++) {
                u8 *smPc = smP_w + ch * 16 * 16;
                for (int i = 0; i < 8; i++) {
                    int m_row = half * 8 + i;
                    smPc[m_row * 16 + idx] = f32_to_fp8_e4m3_pos(S[ch][i]);
                }
            }
            __builtin_amdgcn_s_waitcnt(0);

            /* ---- Barrier: ensures P write visible + prefetch done before next tile. ---- */
            __syncthreads();

            /* ---- P read (transposed, vectorized) + PV ---- */
            for (int ch = 0; ch < CH; ch++) {
                u8 *smPc = smP_w + ch * 16 * 16;
                u8x8 ap_b = *(const u8x8 *)(smPc + idx * 16 + half * 8);
                u32x2 ap = pack_u8x8(ap_b);
                for (int kb = 0; kb < K_NB; kb++) {
                    u8x8 bv;
                    for (int i = 0; i < 8; i++) {
                        int d_col = kb * 16 + idx;
                        int kv_k = ch * 16 + half * 8 + i;
                        bv[i] = curV[d_col * BKV + kv_k];
                    }
                    u32x2 bv_p = pack_u8x8(bv);
                    WMMA_FP8(ap, bv_p, O_acc[kb]);
                }
            }
        }
    }

    /* ---- Epilogue: normalize by l_i, scale by sV, write FP32 output ---- */
    float vs = sV[h];
    for (int i = 0; i < 8; i++) {
        float l = l_i[i];
        l += __shfl_xor(l, 1, 32);
        l += __shfl_xor(l, 2, 32);
        l += __shfl_xor(l, 4, 32);
        l += __shfl_xor(l, 8, 32);
        l_i[i] = (l > 0.0f) ? 1.0f / l : 0.0f;
    }
    for (int kb = 0; kb < K_NB; kb++) {
        for (int i = 0; i < 8; i++) {
            int row = q0 + wid * 16 + half * 8 + i;
            int d = kb * 16 + idx;
            if (row < n_tok)
                out[(size_t)row * dim + h * HD + d] = O_acc[kb][i] * l_i[i] * vs;
        }
    }
}

/* ----------------------------------------------------------------------- */
/* V3: BQ=256, BKV=32, 16-wave, VWA=4, vectorized loads, FP8 output.      */
/*     No PGR=2 (single buffer), no lambda (inlined), FP8 output stores.   */
/* ----------------------------------------------------------------------- */
extern "C" __global__ __launch_bounds__(512, 1)
void fa3_v3_b32_16w_fp8o(float *out, const u8 *Q, const u8 *K_t, const u8 *V_t,
                          const float *sQ, const float *sK, const float *sV,
                          int n_tok, int n_heads, float inv_sqrtd) {
    enum { BQ = 256, BKV = 32, WAVE_CT = 16, CH = 2 };
    int h  = blockIdx.x;
    int qb = blockIdx.y;
    int q0 = qb * BQ;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int half = lid >> 4;
    int idx = lid & 15;
    int dim = n_heads * HD;

    __shared__ u8 smK[BKV * HD];
    __shared__ u8 smV[HD * BKV];
    __shared__ u8 smP[WAVE_CT * CH * 16 * 16];
    __shared__ float smSq[BQ];
    __shared__ float smSk[BKV];
    u8 *smP_w = smP + wid * CH * 16 * 16;

    /* Load Q scale factors */
    for (int i = tid; i < BQ; i += WAVE_CT * 32) {
        int row = q0 + i;
        smSq[i] = (row < n_tok) ? sQ[(size_t)h * n_tok + row] : 1.0f;
    }
    __syncthreads();

    /* Load Q with VWA=4 */
    u32x2 q_reg[2][4];
    for (int g = 0; g < 2; g++) {
        u8x8 tmp;
        for (int i = 0; i < 4; i++) {
            int kb = g * 4 + i;
            int q_row = q0 + wid * 16 + idx;
            for (int j = 0; j < 8; j++) {
                int d = kb * 16 + half * 8 + j;
                tmp[j] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (u8)0;
            }
            q_reg[g][i] = pack_u8x8(tmp);
        }
    }

    float q_scale_lane[8];
    for (int i = 0; i < 8; i++) {
        int m_row = wid * 16 + half * 8 + i;
        q_scale_lane[i] = smSq[m_row];
    }

    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    int total = BKV * HD;
    int load_stride = (WAVE_CT * 32) * 16;

    float8 O_acc[K_NB];
    for (int kb = 0; kb < K_NB; kb++)
        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;
    float m_i[8], l_i[8];
    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }

    for (int t = 0; t < n_kv_tiles; t++) {
        int kv0 = t * BKV;

        /* Vectorized K/V load + K scales (inlined, no lambda) */
        for (int off = tid * 16; off < total; off += load_stride) {
            int r = off / HD;
            int d = off % HD;
            int kv = kv0 + r;
            size_t base = ((size_t)h * n_tok + kv) * HD + d;
            int kv_ok = (kv < n_tok);
            int in_bounds = (kv_ok && d + 16 <= HD);
            if (in_bounds) {
                u8x16 kd = *(const u8x16 *)(K_t + base);
                u8x16 vd = *(const u8x16 *)(V_t + base);
                *(u8x16 *)(smK + r * HD + d) = kd;
                for (int i = 0; i < 16; i++)
                    smV[(d + i) * BKV + r] = vd[i];
            } else {
                for (int i = 0; i < 16; i++) {
                    u8 kk = 0, vv = 0;
                    if (kv_ok && d + i < HD) {
                        kk = K_t[base + i];
                        vv = V_t[base + i];
                    }
                    smK[r * HD + d + i] = kk;
                    smV[(d + i) * BKV + r] = vv;
                }
            }
        }
        for (int i = tid; i < BKV; i += WAVE_CT * 32) {
            int kv = kv0 + i;
            smSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f;
        }
        __syncthreads();

        /* ---- QK phase with VWA=4 dual-issue ---- */
        float8 S[CH];
        for (int ch = 0; ch < CH; ch++)
            for (int i = 0; i < 8; i++) S[ch][i] = 0.0f;

        for (int g = 0; g < 2; g++) {
            for (int ch = 0; ch < CH; ch++) {
                u32x2 k_reg[4];
                for (int i = 0; i < 4; i++) {
                    int kb = g * 4 + i;
                    u8x8 b_K;
                    for (int j = 0; j < 8; j++) {
                        int dpos = kb * 16 + half * 8 + j;
                        b_K[j] = smK[(ch * 16 + idx) * HD + dpos];
                    }
                    k_reg[i] = pack_u8x8(b_K);
                }
                for (int i = 0; i < 4; i++)
                    WMMA_FP8(q_reg[g][i], k_reg[i], S[ch]);
            }
        }

        /* ---- Online softmax ---- */
        float row_max[8];
        for (int i = 0; i < 8; i++) {
            float mx = -1e30f;
            for (int ch = 0; ch < CH; ch++) {
                int col = ch * 16 + idx;
                int kv = kv0 + col;
                float s = S[ch][i] * q_scale_lane[i] * smSk[col] * inv_sqrtd * LOG2E;
                if (kv >= n_tok) s = -1e30f;
                S[ch][i] = s;
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
            alpha[i] = exp2_fast_bits(m_i[i] - new_max);
            float local = 0.0f;
            for (int ch = 0; ch < CH; ch++) {
                int col = ch * 16 + idx;
                int kv = kv0 + col;
                float p = exp2_fast_bits(S[ch][i] - new_max + 8.0f);
                if (kv >= n_tok) p = 0.0f;
                S[ch][i] = p;
                local += p;
            }
            l_i[i] = l_i[i] * alpha[i] + local;
            m_i[i] = new_max;
        }

        for (int kb = 0; kb < K_NB; kb++)
            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];

        /* ---- P write (transpose, byte-wise) ---- */
        for (int ch = 0; ch < CH; ch++) {
            u8 *smPc = smP_w + ch * 16 * 16;
            for (int i = 0; i < 8; i++) {
                int m_row = half * 8 + i;
                smPc[m_row * 16 + idx] = f32_to_fp8_e4m3_pos(S[ch][i]);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);

        __syncthreads();

        /* ---- P read (vectorized) + PV ---- */
        for (int ch = 0; ch < CH; ch++) {
            u8 *smPc = smP_w + ch * 16 * 16;
            u8x8 ap_b = *(const u8x8 *)(smPc + idx * 16 + half * 8);
            u32x2 ap = pack_u8x8(ap_b);
            for (int kb = 0; kb < K_NB; kb++) {
                u8x8 bv;
                for (int i = 0; i < 8; i++) {
                    int d_col = kb * 16 + idx;
                    int kv_k = ch * 16 + half * 8 + i;
                    bv[i] = smV[d_col * BKV + kv_k];
                }
                u32x2 bv_p = pack_u8x8(bv);
                WMMA_FP8(ap, bv_p, O_acc[kb]);
            }
        }
    }

    /* ---- Epilogue: normalize + FP32 output ---- */
    float vs = sV[h];
    for (int i = 0; i < 8; i++) {
        float l = l_i[i];
        l += __shfl_xor(l, 1, 32);
        l += __shfl_xor(l, 2, 32);
        l += __shfl_xor(l, 4, 32);
        l += __shfl_xor(l, 8, 32);
        l_i[i] = (l > 0.0f) ? 1.0f / l : 0.0f;
    }
    for (int kb = 0; kb < K_NB; kb++) {
        for (int i = 0; i < 8; i++) {
            int row = q0 + wid * 16 + half * 8 + i;
            int d = kb * 16 + idx;
            if (row < n_tok)
                out[(size_t)row * dim + h * HD + d] = O_acc[kb][i] * l_i[i] * vs;
        }
    }
}

/* ----------------------------------------------------------------------- */
/* V4: Same as V3 but with FP8 output (u8* instead of float*).             */
/* ----------------------------------------------------------------------- */
extern "C" __global__ __launch_bounds__(512, 1)
void fa3_v4_b32_16w_fp8o(u8 *out, const u8 *Q, const u8 *K_t, const u8 *V_t,
                          const float *sQ, const float *sK, const float *sV,
                          int n_tok, int n_heads, float inv_sqrtd) {
    enum { BQ = 256, BKV = 32, WAVE_CT = 16, CH = 2 };
    int h  = blockIdx.x;
    int qb = blockIdx.y;
    int q0 = qb * BQ;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int half = lid >> 4;
    int idx = lid & 15;
    int dim = n_heads * HD;

    __shared__ u8 smK[BKV * HD];
    __shared__ u8 smV[HD * BKV];
    __shared__ u8 smP[WAVE_CT * CH * 16 * 16];
    __shared__ float smSq[BQ];
    __shared__ float smSk[BKV];
    u8 *smP_w = smP + wid * CH * 16 * 16;

    for (int i = tid; i < BQ; i += WAVE_CT * 32) {
        int row = q0 + i;
        smSq[i] = (row < n_tok) ? sQ[(size_t)h * n_tok + row] : 1.0f;
    }
    __syncthreads();

    u32x2 q_reg[2][4];
    for (int g = 0; g < 2; g++) {
        u8x8 tmp;
        for (int i = 0; i < 4; i++) {
            int kb = g * 4 + i;
            int q_row = q0 + wid * 16 + idx;
            for (int j = 0; j < 8; j++) {
                int d = kb * 16 + half * 8 + j;
                tmp[j] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (u8)0;
            }
            q_reg[g][i] = pack_u8x8(tmp);
        }
    }

    float q_scale_lane[8];
    for (int i = 0; i < 8; i++) {
        int m_row = wid * 16 + half * 8 + i;
        q_scale_lane[i] = smSq[m_row];
    }

    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    int total = BKV * HD;
    int load_stride = (WAVE_CT * 32) * 16;

    float8 O_acc[K_NB];
    for (int kb = 0; kb < K_NB; kb++)
        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;
    float m_i[8], l_i[8];
    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }

    for (int t = 0; t < n_kv_tiles; t++) {
        int kv0 = t * BKV;

        for (int off = tid * 16; off < total; off += load_stride) {
            int r = off / HD;
            int d = off % HD;
            int kv = kv0 + r;
            size_t base = ((size_t)h * n_tok + kv) * HD + d;
            int kv_ok = (kv < n_tok);
            int in_bounds = (kv_ok && d + 16 <= HD);
            if (in_bounds) {
                u8x16 kd = *(const u8x16 *)(K_t + base);
                u8x16 vd = *(const u8x16 *)(V_t + base);
                *(u8x16 *)(smK + r * HD + d) = kd;
                for (int i = 0; i < 16; i++)
                    smV[(d + i) * BKV + r] = vd[i];
            } else {
                for (int i = 0; i < 16; i++) {
                    u8 kk = 0, vv = 0;
                    if (kv_ok && d + i < HD) {
                        kk = K_t[base + i];
                        vv = V_t[base + i];
                    }
                    smK[r * HD + d + i] = kk;
                    smV[(d + i) * BKV + r] = vv;
                }
            }
        }
        for (int i = tid; i < BKV; i += WAVE_CT * 32) {
            int kv = kv0 + i;
            smSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f;
        }
        __syncthreads();

        float8 S[CH];
        for (int ch = 0; ch < CH; ch++)
            for (int i = 0; i < 8; i++) S[ch][i] = 0.0f;

        for (int g = 0; g < 2; g++) {
            for (int ch = 0; ch < CH; ch++) {
                u32x2 k_reg[4];
                for (int i = 0; i < 4; i++) {
                    int kb = g * 4 + i;
                    u8x8 b_K;
                    for (int j = 0; j < 8; j++) {
                        int dpos = kb * 16 + half * 8 + j;
                        b_K[j] = smK[(ch * 16 + idx) * HD + dpos];
                    }
                    k_reg[i] = pack_u8x8(b_K);
                }
                for (int i = 0; i < 4; i++)
                    WMMA_FP8(q_reg[g][i], k_reg[i], S[ch]);
            }
        }

        float row_max[8];
        for (int i = 0; i < 8; i++) {
            float mx = -1e30f;
            for (int ch = 0; ch < CH; ch++) {
                int col = ch * 16 + idx;
                int kv = kv0 + col;
                float s = S[ch][i] * q_scale_lane[i] * smSk[col] * inv_sqrtd * LOG2E;
                if (kv >= n_tok) s = -1e30f;
                S[ch][i] = s;
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
            alpha[i] = exp2_fast_bits(m_i[i] - new_max);
            float local = 0.0f;
            for (int ch = 0; ch < CH; ch++) {
                int col = ch * 16 + idx;
                int kv = kv0 + col;
                float p = exp2_fast_bits(S[ch][i] - new_max + 8.0f);
                if (kv >= n_tok) p = 0.0f;
                S[ch][i] = p;
                local += p;
            }
            l_i[i] = l_i[i] * alpha[i] + local;
            m_i[i] = new_max;
        }

        for (int kb = 0; kb < K_NB; kb++)
            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];

        for (int ch = 0; ch < CH; ch++) {
            u8 *smPc = smP_w + ch * 16 * 16;
            for (int i = 0; i < 8; i++) {
                int m_row = half * 8 + i;
                smPc[m_row * 16 + idx] = f32_to_fp8_e4m3_pos(S[ch][i]);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);

        __syncthreads();

        for (int ch = 0; ch < CH; ch++) {
            u8 *smPc = smP_w + ch * 16 * 16;
            u8x8 ap_b = *(const u8x8 *)(smPc + idx * 16 + half * 8);
            u32x2 ap = pack_u8x8(ap_b);
            for (int kb = 0; kb < K_NB; kb++) {
                u8x8 bv;
                for (int i = 0; i < 8; i++) {
                    int d_col = kb * 16 + idx;
                    int kv_k = ch * 16 + half * 8 + i;
                    bv[i] = smV[d_col * BKV + kv_k];
                }
                u32x2 bv_p = pack_u8x8(bv);
                WMMA_FP8(ap, bv_p, O_acc[kb]);
            }
        }
    }

    /* ---- Epilogue: normalize + FP8 output ---- */
    float vs = sV[h];
    for (int i = 0; i < 8; i++) {
        float l = l_i[i];
        l += __shfl_xor(l, 1, 32);
        l += __shfl_xor(l, 2, 32);
        l += __shfl_xor(l, 4, 32);
        l += __shfl_xor(l, 8, 32);
        l_i[i] = (l > 0.0f) ? 1.0f / l : 0.0f;
    }
    /* Write FP8 output: 1 byte per element instead of 4 */
    for (int kb = 0; kb < K_NB; kb++) {
        for (int i = 0; i < 8; i++) {
            int row = q0 + wid * 16 + half * 8 + i;
            int d = kb * 16 + idx;
            if (row < n_tok) {
                float v = O_acc[kb][i] * l_i[i] * vs;
                out[(size_t)row * dim + h * HD + d] = f32_to_fp8_e4m3_pos(v);
            }
        }
    }
}

/* ----------------------------------------------------------------------- */
/* V8: BKV=16, CH=1. Half the P-transpose per tile, fewer VGPRs for S.    */
/*     More tiles (256 vs 128) but less per-tile overhead.                 */
/*     Based on V3 (clean VWA=4, vectorized loads).                        */
/* ----------------------------------------------------------------------- */
extern "C" __global__ __launch_bounds__(512, 1)
void fa3_v8_b16(float *out, const u8 *Q, const u8 *K_t, const u8 *V_t,
                const float *sQ, const float *sK, const float *sV,
                int n_tok, int n_heads, float inv_sqrtd) {
    enum { BQ = 256, BKV = 16, WAVE_CT = 16, CH = 1 };
    int h  = blockIdx.x;
    int qb = blockIdx.y;
    int q0 = qb * BQ;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int half = lid >> 4;
    int idx = lid & 15;
    int dim = n_heads * HD;

    __shared__ u8 smK[BKV * HD];
    __shared__ u8 smV[HD * BKV];
    __shared__ u8 smP[WAVE_CT * CH * 16 * 16];
    __shared__ float smSq[BQ];
    __shared__ float smSk[BKV];
    u8 *smP_w = smP + wid * CH * 16 * 16;

    for (int i = tid; i < BQ; i += WAVE_CT * 32) {
        int row = q0 + i;
        smSq[i] = (row < n_tok) ? sQ[(size_t)h * n_tok + row] : 1.0f;
    }
    __syncthreads();

    u32x2 q_reg[2][4];
    for (int g = 0; g < 2; g++) {
        u8x8 tmp;
        for (int i = 0; i < 4; i++) {
            int kb = g * 4 + i;
            int q_row = q0 + wid * 16 + idx;
            for (int j = 0; j < 8; j++) {
                int d = kb * 16 + half * 8 + j;
                tmp[j] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (u8)0;
            }
            q_reg[g][i] = pack_u8x8(tmp);
        }
    }

    float q_scale_lane[8];
    for (int i = 0; i < 8; i++) {
        int m_row = wid * 16 + half * 8 + i;
        q_scale_lane[i] = smSq[m_row];
    }

    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    int total = BKV * HD;
    int load_stride = (WAVE_CT * 32) * 16;

    float8 O_acc[K_NB];
    for (int kb = 0; kb < K_NB; kb++)
        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;
    float m_i[8], l_i[8];
    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }

    for (int t = 0; t < n_kv_tiles; t++) {
        int kv0 = t * BKV;

        for (int off = tid * 16; off < total; off += load_stride) {
            int r = off / HD;
            int d = off % HD;
            int kv = kv0 + r;
            size_t base = ((size_t)h * n_tok + kv) * HD + d;
            int kv_ok = (kv < n_tok);
            int in_bounds = (kv_ok && d + 16 <= HD);
            if (in_bounds) {
                u8x16 kd = *(const u8x16 *)(K_t + base);
                u8x16 vd = *(const u8x16 *)(V_t + base);
                *(u8x16 *)(smK + r * HD + d) = kd;
                for (int i = 0; i < 16; i++)
                    smV[(d + i) * BKV + r] = vd[i];
            } else {
                for (int i = 0; i < 16; i++) {
                    u8 kk = 0, vv = 0;
                    if (kv_ok && d + i < HD) {
                        kk = K_t[base + i];
                        vv = V_t[base + i];
                    }
                    smK[r * HD + d + i] = kk;
                    smV[(d + i) * BKV + r] = vv;
                }
            }
        }
        for (int i = tid; i < BKV; i += WAVE_CT * 32) {
            int kv = kv0 + i;
            smSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f;
        }
        __syncthreads();

        /* ---- QK phase with VWA=4 (CH=1) ---- */
        float8 S;
        for (int i = 0; i < 8; i++) S[i] = 0.0f;

        for (int g = 0; g < 2; g++) {
            u32x2 k_reg[4];
            for (int i = 0; i < 4; i++) {
                int kb = g * 4 + i;
                u8x8 b_K;
                for (int j = 0; j < 8; j++) {
                    int dpos = kb * 16 + half * 8 + j;
                    b_K[j] = smK[(0 * 16 + idx) * HD + dpos];
                }
                k_reg[i] = pack_u8x8(b_K);
            }
            for (int i = 0; i < 4; i++)
                WMMA_FP8(q_reg[g][i], k_reg[i], S);
        }

        /* ---- Online softmax ---- */
        float row_max[8];
        for (int i = 0; i < 8; i++) {
            int col = 0 * 16 + idx;
            int kv = kv0 + col;
            float s = S[i] * q_scale_lane[i] * smSk[col] * inv_sqrtd * LOG2E;
            if (kv >= n_tok) s = -1e30f;
            S[i] = s;
            row_max[i] = s;
        }
        for (int i = 0; i < 8; i++) {
            row_max[i] = fmaxf(row_max[i], __shfl_xor(row_max[i], 1, 32));
            row_max[i] = fmaxf(row_max[i], __shfl_xor(row_max[i], 2, 32));
            row_max[i] = fmaxf(row_max[i], __shfl_xor(row_max[i], 4, 32));
            row_max[i] = fmaxf(row_max[i], __shfl_xor(row_max[i], 8, 32));
        }

        float alpha[8];
        for (int i = 0; i < 8; i++) {
            float new_max = fmaxf(m_i[i], row_max[i]);
            alpha[i] = exp2_fast_bits(m_i[i] - new_max);
            int col = 0 * 16 + idx;
            int kv = kv0 + col;
            float p = exp2_fast_bits(S[i] - new_max + 8.0f);
            if (kv >= n_tok) p = 0.0f;
            S[i] = p;
            l_i[i] = l_i[i] * alpha[i] + p;
            m_i[i] = new_max;
        }

        for (int kb = 0; kb < K_NB; kb++)
            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];

        /* ---- P write (CH=1, single byte-wise) ---- */
        {
            u8 *smPc = smP_w;
            for (int i = 0; i < 8; i++) {
                int m_row = half * 8 + i;
                smPc[m_row * 16 + idx] = f32_to_fp8_e4m3_pos(S[i]);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);

        __syncthreads();

        /* ---- P read + PV ---- */
        {
            u8 *smPc = smP_w;
            u8x8 ap_b = *(const u8x8 *)(smPc + idx * 16 + half * 8);
            u32x2 ap = pack_u8x8(ap_b);
            for (int kb = 0; kb < K_NB; kb++) {
                u8x8 bv;
                for (int i = 0; i < 8; i++) {
                    int d_col = kb * 16 + idx;
                    int kv_k = 0 * 16 + half * 8 + i;
                    bv[i] = smV[d_col * BKV + kv_k];
                }
                u32x2 bv_p = pack_u8x8(bv);
                WMMA_FP8(ap, bv_p, O_acc[kb]);
            }
        }
    }

    float vs = sV[h];
    for (int i = 0; i < 8; i++) {
        float l = l_i[i];
        l += __shfl_xor(l, 1, 32);
        l += __shfl_xor(l, 2, 32);
        l += __shfl_xor(l, 4, 32);
        l += __shfl_xor(l, 8, 32);
        l_i[i] = (l > 0.0f) ? 1.0f / l : 0.0f;
    }
    for (int kb = 0; kb < K_NB; kb++) {
        for (int i = 0; i < 8; i++) {
            int row = q0 + wid * 16 + half * 8 + i;
            int d = kb * 16 + idx;
            if (row < n_tok)
                out[(size_t)row * dim + h * HD + d] = O_acc[kb][i] * l_i[i] * vs;
        }
    }
}

/* ----------------------------------------------------------------------- */
/* GQA: Grouped-Query Attention. K=V shared across heads_per_kv heads.    */
/*      grid.x = n_kv_heads. Extra param: n_kv_heads.                     */
/*      Based on V3 (BKV=32, 16w, VWA=4, vectorized loads).               */
/* ----------------------------------------------------------------------- */
extern "C" __global__ __launch_bounds__(512, 1)
void fa3_gqa(float *out, const u8 *Q, const u8 *K_t, const u8 *V_t,
             const float *sQ, const float *sK, const float *sV,
             int n_tok, int n_heads, float inv_sqrtd, int n_kv_heads) {
    enum { BQ = 256, BKV = 32, WAVE_CT = 16, CH = 2 };
    int kv_h = blockIdx.x;
    int qb = blockIdx.y;
    int q0 = qb * BQ;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int half = lid >> 4;
    int idx = lid & 15;
    int dim = n_heads * HD;
    int hpk = n_heads / n_kv_heads;

    __shared__ u8 smK[BKV * HD];
    __shared__ u8 smV[HD * BKV];
    __shared__ u8 smP[WAVE_CT * CH * 16 * 16];
    __shared__ float smSq[BQ];
    __shared__ float smSk[BKV];
    u8 *smP_w = smP + wid * CH * 16 * 16;

    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    int total = BKV * HD;
    int load_stride = (WAVE_CT * 32) * 16;

    /* Pre-load Q and Q scales for each query head (tile-invariant) */
    u32x2 q_reg[hpk][2][4];
    float q_scale_lane[hpk][8];
    for (int hh = 0; hh < hpk; hh++) {
        int h = kv_h * hpk + hh;
        for (int i = tid; i < BQ; i += WAVE_CT * 32) {
            int row = q0 + i;
            smSq[i] = (row < n_tok) ? sQ[(size_t)h * n_tok + row] : 1.0f;
        }
        for (int g = 0; g < 2; g++) {
            u8x8 tmp;
            for (int i = 0; i < 4; i++) {
                int kb = g * 4 + i;
                int q_row = q0 + wid * 16 + idx;
                for (int j = 0; j < 8; j++) {
                    int d = kb * 16 + half * 8 + j;
                    tmp[j] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (u8)0;
                }
                q_reg[hh][g][i] = pack_u8x8(tmp);
            }
        }
        for (int i = 0; i < 8; i++) {
            int m_row = wid * 16 + half * 8 + i;
            q_scale_lane[hh][i] = smSq[m_row];
        }
    }

    /* Per-head accumulators */
    float8 O_acc[hpk][K_NB];
    float m_i[hpk][8], l_i[hpk][8];
    for (int hh = 0; hh < hpk; hh++) {
        for (int kb = 0; kb < K_NB; kb++)
            for (int i = 0; i < 8; i++) O_acc[hh][kb][i] = 0.0f;
        for (int i = 0; i < 8; i++) { m_i[hh][i] = -1e30f; l_i[hh][i] = 0.0f; }
    }

    for (int t = 0; t < n_kv_tiles; t++) {
        int kv0 = t * BKV;

        /* Vectorized K/V load (shared, once per tile) */
        for (int off = tid * 16; off < total; off += load_stride) {
            int r = off / HD;
            int d = off % HD;
            int kv = kv0 + r;
            size_t base = ((size_t)kv_h * n_tok + kv) * HD + d;
            int kv_ok = (kv < n_tok);
            int in_bounds = (kv_ok && d + 16 <= HD);
            if (in_bounds) {
                u8x16 kd = *(const u8x16 *)(K_t + base);
                u8x16 vd = *(const u8x16 *)(V_t + base);
                *(u8x16 *)(smK + r * HD + d) = kd;
                for (int i = 0; i < 16; i++)
                    smV[(d + i) * BKV + r] = vd[i];
            } else {
                for (int i = 0; i < 16; i++) {
                    u8 kk = 0, vv = 0;
                    if (kv_ok && d + i < HD) {
                        kk = K_t[base + i];
                        vv = V_t[base + i];
                    }
                    smK[r * HD + d + i] = kk;
                    smV[(d + i) * BKV + r] = vv;
                }
            }
        }
        for (int i = tid; i < BKV; i += WAVE_CT * 32) {
            int kv = kv0 + i;
            smSk[i] = (kv < n_tok) ? sK[(size_t)kv_h * n_tok + kv] : 1.0f;
        }
        __syncthreads();

        /* Process all query heads using this tile's K/V */
        for (int hh = 0; hh < hpk; hh++) {
            int h = kv_h * hpk + hh;

            /* ---- QK phase ---- */
            float8 S[CH];
            for (int ch = 0; ch < CH; ch++)
                for (int i = 0; i < 8; i++) S[ch][i] = 0.0f;

            for (int g = 0; g < 2; g++) {
                for (int ch = 0; ch < CH; ch++) {
                    u32x2 k_reg[4];
                    for (int i = 0; i < 4; i++) {
                        int kb = g * 4 + i;
                        u8x8 b_K;
                        for (int j = 0; j < 8; j++) {
                            int dpos = kb * 16 + half * 8 + j;
                            b_K[j] = smK[(ch * 16 + idx) * HD + dpos];
                        }
                        k_reg[i] = pack_u8x8(b_K);
                    }
                    for (int i = 0; i < 4; i++)
                        WMMA_FP8(q_reg[hh][g][i], k_reg[i], S[ch]);
                }
            }

            /* ---- Online softmax ---- */
            float row_max[8];
            for (int i = 0; i < 8; i++) {
                float mx = -1e30f;
                for (int ch = 0; ch < CH; ch++) {
                    int col = ch * 16 + idx;
                    int kv = kv0 + col;
                    float s = S[ch][i] * q_scale_lane[hh][i] * smSk[col] * inv_sqrtd * LOG2E;
                    if (kv >= n_tok) s = -1e30f;
                    S[ch][i] = s;
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
                float new_max = fmaxf(m_i[hh][i], row_max[i]);
                alpha[i] = exp2_fast_bits(m_i[hh][i] - new_max);
                float local = 0.0f;
                for (int ch = 0; ch < CH; ch++) {
                    int col = ch * 16 + idx;
                    int kv = kv0 + col;
                    float p = exp2_fast_bits(S[ch][i] - new_max + 8.0f);
                    if (kv >= n_tok) p = 0.0f;
                    S[ch][i] = p;
                    local += p;
                }
                l_i[hh][i] = l_i[hh][i] * alpha[i] + local;
                m_i[hh][i] = new_max;
            }

            for (int kb = 0; kb < K_NB; kb++)
                for (int i = 0; i < 8; i++) O_acc[hh][kb][i] *= alpha[i];

            /* ---- P write (transpose) ---- */
            for (int ch = 0; ch < CH; ch++) {
                u8 *smPc = smP_w + ch * 16 * 16;
                for (int i = 0; i < 8; i++) {
                    int m_row = half * 8 + i;
                    smPc[m_row * 16 + idx] = f32_to_fp8_e4m3_pos(S[ch][i]);
                }
            }
            __builtin_amdgcn_s_waitcnt(0);

            __syncthreads();

            /* ---- P read + PV ---- */
            for (int ch = 0; ch < CH; ch++) {
                u8 *smPc = smP_w + ch * 16 * 16;
                u8x8 ap_b = *(const u8x8 *)(smPc + idx * 16 + half * 8);
                u32x2 ap = pack_u8x8(ap_b);
                for (int kb = 0; kb < K_NB; kb++) {
                    u8x8 bv;
                    for (int i = 0; i < 8; i++) {
                        int d_col = kb * 16 + idx;
                        int kv_k = ch * 16 + half * 8 + i;
                        bv[i] = smV[d_col * BKV + kv_k];
                    }
                    u32x2 bv_p = pack_u8x8(bv);
                    WMMA_FP8(ap, bv_p, O_acc[hh][kb]);
                }
            }
        }
    }

    /* ---- Epilogue for all query heads ---- */
    for (int hh = 0; hh < hpk; hh++) {
        int h = kv_h * hpk + hh;
        float vs = sV[kv_h];
        for (int i = 0; i < 8; i++) {
            float l = l_i[hh][i];
            l += __shfl_xor(l, 1, 32);
            l += __shfl_xor(l, 2, 32);
            l += __shfl_xor(l, 4, 32);
            l += __shfl_xor(l, 8, 32);
            l_i[hh][i] = (l > 0.0f) ? 1.0f / l : 0.0f;
        }
        for (int kb = 0; kb < K_NB; kb++) {
            for (int i = 0; i < 8; i++) {
                int row = q0 + wid * 16 + half * 8 + i;
                int d = kb * 16 + idx;
                if (row < n_tok)
                    out[(size_t)row * dim + h * HD + d] = O_acc[hh][kb][i] * l_i[hh][i] * vs;
            }
        }
    }
}

/* ----------------------------------------------------------------------- */
/* V2.5: BQ=512, BKV=32, 32-wave, VWA=4, vectorized loads, PGR=2.         */
/* ----------------------------------------------------------------------- */
extern "C" __global__ __launch_bounds__(1024, 1)
void fa3_v2_b32_32w(float *out, const u8 *Q, const u8 *K_t, const u8 *V_t,
                    const float *sQ, const float *sK, const float *sV,
                    int n_tok, int n_heads, float inv_sqrtd) {
    enum { BQ = 512, BKV = 32, WAVE_CT = 32, CH = 2, DBUF = 2 };
    int h  = blockIdx.x;
    int qb = blockIdx.y;
    int q0 = qb * BQ;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int half = lid >> 4;
    int idx = lid & 15;
    int dim = n_heads * HD;

    __shared__ u8 smK[DBUF][BKV * HD];
    __shared__ u8 smV[DBUF][HD * BKV];
    __shared__ u8 smP[WAVE_CT * CH * 16 * 16];
    __shared__ float smSq[BQ];
    __shared__ float smSk[DBUF][BKV];
    u8 *smP_w = smP + wid * CH * 16 * 16;

    for (int i = tid; i < BQ; i += WAVE_CT * 32) {
        int row = q0 + i;
        smSq[i] = (row < n_tok) ? sQ[(size_t)h * n_tok + row] : 1.0f;
    }
    __syncthreads();

    u32x2 q_reg[2][4];
    for (int g = 0; g < 2; g++) {
        u8x8 tmp;
        for (int i = 0; i < 4; i++) {
            int kb = g * 4 + i;
            int q_row = q0 + wid * 16 + idx;
            for (int j = 0; j < 8; j++) {
                int d = kb * 16 + half * 8 + j;
                tmp[j] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (u8)0;
            }
            q_reg[g][i] = pack_u8x8(tmp);
        }
    }

    float q_scale_lane[8];
    for (int i = 0; i < 8; i++) {
        int m_row = wid * 16 + half * 8 + i;
        q_scale_lane[i] = smSq[m_row];
    }

    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    int total = BKV * HD;
    int load_stride = (WAVE_CT * 32) * 16;

    auto load_kv_tile = [&](u8 *dstK, u8 *dstV, float *dstSk, int kv_base) {
        for (int off = tid * 16; off < total; off += load_stride) {
            int r = off / HD;
            int d = off % HD;
            int kv = kv_base + r;
            size_t base = ((size_t)h * n_tok + kv) * HD + d;
            int kv_ok = (kv < n_tok);
            int in_bounds = (kv_ok && d + 16 <= HD);
            if (in_bounds) {
                u8x16 kd = *(const u8x16 *)(K_t + base);
                u8x16 vd = *(const u8x16 *)(V_t + base);
                *(u8x16 *)(dstK + r * HD + d) = kd;
                for (int i = 0; i < 16; i++)
                    dstV[(d + i) * BKV + r] = vd[i];
            } else {
                for (int i = 0; i < 16; i++) {
                    u8 kk = 0, vv = 0;
                    if (kv_ok && d + i < HD) {
                        kk = K_t[base + i];
                        vv = V_t[base + i];
                    }
                    dstK[r * HD + d + i] = kk;
                    dstV[(d + i) * BKV + r] = vv;
                }
            }
        }
        for (int i = tid; i < BKV; i += WAVE_CT * 32) {
            int kv = kv_base + i;
            dstSk[i] = (kv < n_tok) ? sK[(size_t)h * n_tok + kv] : 1.0f;
        }
    };

    load_kv_tile(smK[0], smV[0], smSk[0], 0);
    __syncthreads();

    float8 O_acc[K_NB];
    for (int kb = 0; kb < K_NB; kb++)
        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;
    float m_i[8], l_i[8];
    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }

    for (int t = 0; t < n_kv_tiles; t++) {
        int cur = t & 1;
        int nxt = cur ^ 1;
        int kv0 = t * BKV;
        u8 *curK = smK[cur];
        u8 *curV = smV[cur];
        float *curSk = smSk[cur];

        if (t + 1 < n_kv_tiles)
            load_kv_tile(smK[nxt], smV[nxt], smSk[nxt], (t + 1) * BKV);

        float8 S[CH];
        for (int ch = 0; ch < CH; ch++)
            for (int i = 0; i < 8; i++) S[ch][i] = 0.0f;

        for (int g = 0; g < 2; g++) {
            for (int ch = 0; ch < CH; ch++) {
                u32x2 k_reg[4];
                for (int i = 0; i < 4; i++) {
                    int kb = g * 4 + i;
                    u8x8 b_K;
                    for (int j = 0; j < 8; j++) {
                        int dpos = kb * 16 + half * 8 + j;
                        b_K[j] = curK[(ch * 16 + idx) * HD + dpos];
                    }
                    k_reg[i] = pack_u8x8(b_K);
                }
                for (int i = 0; i < 4; i++)
                    WMMA_FP8(q_reg[g][i], k_reg[i], S[ch]);
            }
        }

        float row_max[8];
        for (int i = 0; i < 8; i++) {
            float mx = -1e30f;
            for (int ch = 0; ch < CH; ch++) {
                int col = ch * 16 + idx;
                int kv = kv0 + col;
                float s = S[ch][i] * q_scale_lane[i] * curSk[col] * inv_sqrtd * LOG2E;
                if (kv >= n_tok) s = -1e30f;
                S[ch][i] = s;
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
            alpha[i] = exp2_fast_bits(m_i[i] - new_max);
            float local = 0.0f;
            for (int ch = 0; ch < CH; ch++) {
                int col = ch * 16 + idx;
                int kv = kv0 + col;
                float p = exp2_fast_bits(S[ch][i] - new_max + 8.0f);
                if (kv >= n_tok) p = 0.0f;
                S[ch][i] = p;
                local += p;
            }
            l_i[i] = l_i[i] * alpha[i] + local;
            m_i[i] = new_max;
        }

        for (int kb = 0; kb < K_NB; kb++)
            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];

        for (int ch = 0; ch < CH; ch++) {
            u8 *smPc = smP_w + ch * 16 * 16;
            for (int i = 0; i < 8; i++) {
                int m_row = half * 8 + i;
                smPc[m_row * 16 + idx] = f32_to_fp8_e4m3_pos(S[ch][i]);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);

        __syncthreads();

        for (int ch = 0; ch < CH; ch++) {
            u8 *smPc = smP_w + ch * 16 * 16;
            u8x8 ap_b = *(const u8x8 *)(smPc + idx * 16 + half * 8);
            u32x2 ap = pack_u8x8(ap_b);
            for (int kb = 0; kb < K_NB; kb++) {
                u8x8 bv;
                for (int i = 0; i < 8; i++) {
                    int d_col = kb * 16 + idx;
                    int kv_k = ch * 16 + half * 8 + i;
                    bv[i] = curV[d_col * BKV + kv_k];
                }
                u32x2 bv_p = pack_u8x8(bv);
                WMMA_FP8(ap, bv_p, O_acc[kb]);
            }
        }
    }

    float vs = sV[h];
    for (int i = 0; i < 8; i++) {
        float l = l_i[i];
        l += __shfl_xor(l, 1, 32);
        l += __shfl_xor(l, 2, 32);
        l += __shfl_xor(l, 4, 32);
        l += __shfl_xor(l, 8, 32);
        l_i[i] = (l > 0.0f) ? 1.0f / l : 0.0f;
    }
    for (int kb = 0; kb < K_NB; kb++) {
        for (int i = 0; i < 8; i++) {
            int row = q0 + wid * 16 + half * 8 + i;
            int d = kb * 16 + idx;
            if (row < n_tok)
                out[(size_t)row * dim + h * HD + d] = O_acc[kb][i] * l_i[i] * vs;
        }
    }
}
)FA3V2SRC";

/* ======================================================================== */
/* Host harness                                                              */
/* ======================================================================== */

typedef struct {
    const char *name;
    hipFunction_t func;
    int block_size;
    int bq;
    int output_fp8; /* 1 = u8 output (1 byte), 0 = float output (4 bytes) */
} fa3_kernel_desc;

#define FA3_MAX_KERNELS 8

typedef struct {
    fa3_kernel_desc descs[FA3_MAX_KERNELS];
    int count;
} fa3_v2_kernels;

static int add_kernel(fa3_v2_kernels *k, hipModule_t mod,
                      const char *name, const char *symbol,
                      int block_size, int bq, int output_fp8) {
    if (k->count >= FA3_MAX_KERNELS) return -1;
    fa3_kernel_desc *d = &k->descs[k->count++];
    d->name = name;
    d->func = NULL;
    d->block_size = block_size;
    d->bq = bq;
    d->output_fp8 = output_fp8;
    HIP_CHECK(hipModuleGetFunction(&d->func, mod, symbol));
    return 0;
}

static fa3_kernel_desc *lookup_kernel(fa3_v2_kernels *k, const char *mode) {
    for (int i = 0; i < k->count; i++)
        if (!strcmp(k->descs[i].name, mode))
            return &k->descs[i];
    return NULL;
}

/* GQA runner: like run_shape_v2 but passes n_kv_heads as 11th arg */
static int run_gqa(fa3_kernel_desc *kd, int n_kv_heads,
                   int n_tok, int n_heads, int iters, int do_check,
                   float abs_max, double peak_tfs) {
    if (n_tok % 64 || n_heads % n_kv_heads) return -1;
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
    unsigned rng = 0x1234abcd;
    fill_fp8_random(hQ,  qkv_elems, abs_max, &rng);
    fill_fp8_random(hKt, qkv_elems, abs_max, &rng);
    fill_fp8_random(hVt, qkv_elems, abs_max, &rng);
    fill_ones(hSq, row_scale_elems);
    fill_ones(hSk, row_scale_elems);
    fill_ones(hSv, (size_t)n_heads);

    void *dQ  = hip_upload_raw(hQ,  qkv_bytes);
    void *dKt = hip_upload_raw(hKt, qkv_bytes);
    void *dVt = hip_upload_raw(hVt, qkv_bytes);
    void *dSq = hip_upload_raw(hSq, row_scale_elems * sizeof(float));
    void *dSk = hip_upload_raw(hSk, row_scale_elems * sizeof(float));
    void *dSv = hip_upload_raw(hSv, (size_t)n_heads * sizeof(float));
    void *dO  = NULL;
    HIP_CHECK(hipMalloc(&dO, out_bytes));
    HIP_CHECK(hipMemset(dO, 0, out_bytes));

    float inv_sqrtd = 1.0f / sqrtf((float)head_dim);
    int bq = kd->bq;
    dim3 block(kd->block_size, 1, 1);
    dim3 grid((unsigned)n_kv_heads, (unsigned)((n_tok + bq - 1) / bq), 1);

    void *args[] = { &dO, &dQ, &dKt, &dVt, &dSq, &dSk, &dSv, &n_tok, &n_heads, &inv_sqrtd, &n_kv_heads };

    HIP_CHECK(hipModuleLaunchKernel(kd->func, grid.x, grid.y, grid.z,
                                    block.x, block.y, block.z, 0, NULL, args, NULL));
    HIP_CHECK(hipDeviceSynchronize());

    double cos = -2.0;
    float maxd = -1.0f;
    if (do_check) {
        float *hO   = (float *)malloc(out_bytes);
        float *hRef = (float *)malloc(out_bytes);
        HIP_CHECK(hipMemcpy(hO, dO, out_bytes, hipMemcpyDeviceToHost));
        fa_ref_fp32(hRef, hQ, hKt, hVt, hSq, hSk, hSv, n_tok, n_heads, head_dim);
        cos  = cosine_sim(hO, hRef, out_elems);
        maxd = max_abs_diff(hO, hRef, out_elems);
        free(hO);
        free(hRef);
    }

    double t0 = timer_ms();
    for (int i = 0; i < iters; i++) {
        HIP_CHECK(hipModuleLaunchKernel(kd->func, grid.x, grid.y, grid.z,
                                        block.x, block.y, block.z, 0, NULL, args, NULL));
    }
    HIP_CHECK(hipDeviceSynchronize());
    double t1 = timer_ms();
    double ms = (t1 - t0) / (double)iters;
    double flops = 4.0 * (double)n_heads * (double)n_tok * (double)n_tok * (double)head_dim;
    double tflops = flops / (ms * 1.0e-3) * 1.0e-12;
    double pct = peak_tfs > 0.0 ? 100.0 * tflops / peak_tfs : 0.0;

    printf("  [%-15s] S=%5d H=%2d KV=%d HD=128  %8.4f ms  %7.2f TF/s  %5.1f%% peak",
           kd->name, n_tok, n_heads, n_kv_heads, ms, tflops, pct);
    if (do_check) printf("  cos=%.6f  maxd=%.5f", cos, maxd);
    printf("\n");

    hipFree(dO); hipFree(dQ); hipFree(dKt); hipFree(dVt);
    hipFree(dSq); hipFree(dSk); hipFree(dSv);
    free(hQ); free(hKt); free(hVt); free(hSq); free(hSk); free(hSv);
    return 0;
}

static int run_shape_v2(fa3_kernel_desc *kd,
                        int n_tok, int n_heads, int iters, int do_check,
                        float abs_max, double peak_tfs) {
    if (n_tok % 64) {
        fprintf(stderr, "skip: n_tok must be %%64 (got %d)\n", n_tok);
        return 0;
    }
    const int head_dim = 128;
    int dim = n_heads * head_dim;
    size_t qkv_elems = (size_t)n_tok * dim;
    size_t qkv_bytes = qkv_elems;
    size_t out_elems = (size_t)n_tok * dim;
    size_t out_bytes = out_elems * (kd->output_fp8 ? 1 : (int)sizeof(float));
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

    void *dQ  = hip_upload_raw(hQ,  qkv_bytes);
    void *dKt = hip_upload_raw(hKt, qkv_bytes);
    void *dVt = hip_upload_raw(hVt, qkv_bytes);
    void *dSq = hip_upload_raw(hSq, row_scale_elems * sizeof(float));
    void *dSk = hip_upload_raw(hSk, row_scale_elems * sizeof(float));
    void *dSv = hip_upload_raw(hSv, (size_t)n_heads * sizeof(float));
    void *dO  = NULL;
    HIP_CHECK(hipMalloc(&dO, out_bytes));
    HIP_CHECK(hipMemset(dO, 0, out_bytes));

    float inv_sqrtd = 1.0f / sqrtf((float)head_dim);
    void *args[] = { &dO, &dQ, &dKt, &dVt, &dSq, &dSk, &dSv, &n_tok, &n_heads, &inv_sqrtd };

    dim3 block(kd->block_size, 1, 1);
    dim3 grid((unsigned)n_heads, (unsigned)((n_tok + kd->bq - 1) / kd->bq), 1);

    HIP_CHECK(hipModuleLaunchKernel(kd->func, grid.x, grid.y, grid.z,
                                    block.x, block.y, block.z, 0, NULL, args, NULL));
    HIP_CHECK(hipDeviceSynchronize());

    double cos = -2.0;
    float maxd = -1.0f;
    if (do_check) {
        float *hRef = (float *)malloc(out_elems * sizeof(float));
        if (kd->output_fp8) {
            uint8_t *hO_u8 = (uint8_t *)malloc(out_bytes);
            HIP_CHECK(hipMemcpy(hO_u8, dO, out_bytes, hipMemcpyDeviceToHost));
            float *hO = (float *)malloc(out_elems * sizeof(float));
            for (size_t i = 0; i < out_elems; i++)
                hO[i] = fp8_e4m3_to_f32(hO_u8[i]);
            fa_ref_fp32(hRef, hQ, hKt, hVt, hSq, hSk, hSv, n_tok, n_heads, head_dim);
            cos  = cosine_sim(hO, hRef, out_elems);
            maxd = max_abs_diff(hO, hRef, out_elems);
            free(hO);
            free(hO_u8);
        } else {
            float *hO = (float *)malloc(out_bytes);
            HIP_CHECK(hipMemcpy(hO, dO, out_bytes, hipMemcpyDeviceToHost));
            fa_ref_fp32(hRef, hQ, hKt, hVt, hSq, hSk, hSv, n_tok, n_heads, head_dim);
            cos  = cosine_sim(hO, hRef, out_elems);
            maxd = max_abs_diff(hO, hRef, out_elems);
            free(hO);
        }
        free(hRef);
    }

    double t0 = timer_ms();
    for (int i = 0; i < iters; i++) {
        HIP_CHECK(hipModuleLaunchKernel(kd->func, grid.x, grid.y, grid.z,
                                        block.x, block.y, block.z, 0, NULL, args, NULL));
    }
    HIP_CHECK(hipDeviceSynchronize());
    double t1 = timer_ms();
    double ms = (t1 - t0) / (double)iters;
    double flops = 4.0 * (double)n_heads * (double)n_tok * (double)n_tok * (double)head_dim;
    double tflops = flops / (ms * 1.0e-3) * 1.0e-12;
    double pct = peak_tfs > 0.0 ? 100.0 * tflops / peak_tfs : 0.0;

    printf("  [%-15s] S=%5d H=%2d HD=128  %8.4f ms  %7.2f TF/s  %5.1f%% peak",
           kd->name, n_tok, n_heads, ms, tflops, pct);
    if (do_check) printf("  cos=%.6f  maxd=%.5f", cos, maxd);
    printf("\n");

    hipFree(dO);
    hipFree(dQ); hipFree(dKt); hipFree(dVt); hipFree(dSq); hipFree(dSk); hipFree(dSv);
    free(hQ); free(hKt); free(hVt); free(hSq); free(hSk); free(hSv);
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [--n-tok N] [--heads H] [--iters N] [--mode MODE]\n"
        "          [--check] [--abs-max V] [--peak-tfs V] [--verbose] [--dump-code]\n"
        "          [--gqa N_KV_HEADS]\n"
        "  --mode      v1 | v2 | v25 | v3 | v4 | v8 | gqa | all\n"
        "  --n-tok     token count, must be %%64 (default 1024)\n"
        "  --heads     attention heads (default 16)\n"
        "  --gqa       KV head count for GQA mode (default 2, implies --mode gqa)\n"
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
    int gqa_kv = 2; /* default: 2 KV heads for GQA-8 (16 query heads) */

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--n-tok")    && i+1<argc) n_tok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads")    && i+1<argc) n_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")    && i+1<argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode")     && i+1<argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--gqa")      && i+1<argc) gqa_kv = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--abs-max")  && i+1<argc) abs_max = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--peak-tfs") && i+1<argc) peak_tfs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--check"))    do_check = 1;
        else if (!strcmp(argv[i], "--verbose"))  verbose = 2;
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
    if (hip_compile_kernels(&mod, dev, kernel_src, "fa3_v2", verbose, "fa3_v2") < 0) return 1;
    fa3_v2_kernels kernels = {0};

    if (add_kernel(&kernels, mod, "v1", "fa3_v1_b64_8w", 256, 128, 0) < 0) return 1;
    if (add_kernel(&kernels, mod, "v2", "fa3_v2_b32_16w_pgr2", 512, 256, 0) < 0) return 1;
    if (add_kernel(&kernels, mod, "v25", "fa3_v2_b32_32w", 1024, 512, 0) < 0) return 1;
    if (add_kernel(&kernels, mod, "v3", "fa3_v3_b32_16w_fp8o", 512, 256, 0) < 0) return 1;
    if (add_kernel(&kernels, mod, "v4", "fa3_v4_b32_16w_fp8o", 512, 256, 1) < 0) return 1;
    if (add_kernel(&kernels, mod, "v8", "fa3_v8_b16", 512, 256, 0) < 0) return 1;
    if (add_kernel(&kernels, mod, "gqa", "fa3_gqa", 512, 256, 0) < 0) return 1;

    printf("rdna4/fa3 FP8 FA v2 bench (iters=%d, abs_max=%.2f, peak=%.1f TF/s%s)\n",
           iters, abs_max, peak_tfs, do_check ? ", check=on" : "");

    if (!strcmp(mode, "gqa")) {
        fa3_kernel_desc *kd = lookup_kernel(&kernels, "gqa");
        if (!kd) { fprintf(stderr, "gqa kernel not found\n"); return 1; }
        if (run_gqa(kd, gqa_kv, n_tok, n_heads, iters,
                     do_check, abs_max, peak_tfs) < 0)
            return 1;
    } else if (!strcmp(mode, "all")) {
        for (int i = 0; i < kernels.count; i++) {
            if (!strcmp(kernels.descs[i].name, "gqa"))
                continue;
            if (run_shape_v2(&kernels.descs[i], n_tok, n_heads, iters,
                             do_check, abs_max, peak_tfs) < 0)
                return 1;
        }
    } else {
        fa3_kernel_desc *kd = lookup_kernel(&kernels, mode);
        if (!kd) { fprintf(stderr, "unknown mode '%s'\n", mode); return 1; }
        if (run_shape_v2(kd, n_tok, n_heads, iters,
                         do_check, abs_max, peak_tfs) < 0)
            return 1;
    }
    return 0;
}
