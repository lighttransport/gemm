/*
 * bench_fa3_fp16.cpp - RDNA4/gfx1201 FP16/BF16 flash-attention via WMMA.
 *
 * Single WG = (h, qb). BQ=256, BKV=32, 16 waves, VWA=4.
 *
 * Key architectural difference from FP8 V3:
 *   1. QK WMMA: A=Q native, B=K native → Q@K^T (no P-transpose needed)
 *   2. PV WMMA: A=P must be transposed (C-layout→A-layout via LDS P-transpose)
 *   3. No scale factors (sQ, sK, sV)
 *   4. K/V are 2× larger (fp16 vs fp8)
 *   5. Output is FP32 (float*)
 *
 * Build:
 *   make -C rdna4/fa3 bench_fa3_fp16
 * Run FP16:
 *   rdna4/fa3/bench_fa3_fp16 --mode f16 --n-tok 4096 --heads 16 --iters 100
 * Run BF16:
 *   rdna4/fa3/bench_fa3_fp16 --mode bf16 --n-tok 4096 --heads 16 --iters 100
 * Check:
 *   rdna4/fa3/bench_fa3_fp16 --mode f16 --n-tok 256 --heads 1 --iters 20 --check
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

/* ---- F16 → F32 ---- */

static float f16_to_f32(uint16_t v) {
    uint32_t sign = (v >> 15) & 1;
    int exp = (v >> 10) & 0x1F;
    uint32_t mant = v & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        return ldexpf((float)mant / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
    }
    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
    return ldexpf(1.0f + (float)mant / 1024.0f, exp - 15) * (sign ? -1.0f : 1.0f);
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

static void fill_f16_random(uint16_t *dst, size_t n, float abs_max, unsigned *rng) {
    for (size_t i = 0; i < n; i++) {
        unsigned r = *rng = *rng * 1103515245u + 12345u;
        float u = ((int)(r & 0xFFFFFFu) - 0x800000) / (float)0x800000;
        float v = u * abs_max;
        dst[i] = hip_f32_to_f16(v);
    }
}

/* ---- FP32 reference: float inputs (generic) ---- */

static void fa_ref_fp32_gen(float *out, const float *Q, const float *K_t,
                            const float *V_t, int n_tok, int n_heads, int head_dim) {
    int dim = n_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    float *scores = (float *)malloc((size_t)n_tok * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        for (int q = 0; q < n_tok; q++) {
            float row_max = -1e30f;
            for (int kv = 0; kv < n_tok; kv++) {
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    s += Q[(size_t)q * dim + h * head_dim + d] *
                         K_t[((size_t)h * n_tok + kv) * head_dim + d];
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
                    acc += scores[kv] * inv_l * V_t[((size_t)h * n_tok + kv) * head_dim + d];
                }
                out[(size_t)q * dim + h * head_dim + d] = acc;
            }
        }
    }
    free(scores);
}

/* ======================================================================== */
/* Kernel source string — HIPRTC-compiled at startup.                       */
/* Compiled TWICE: once as-is (FP16), once with -DUSE_BF16 (BF16).          */
/* ======================================================================== */

static const char *kernel_src = R"FA16SRC(
typedef unsigned int u32;
typedef float float8 __attribute__((ext_vector_type(8)));

#define HD   128
#define K_NB (HD / 16)
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ float exp2_fast_bits(float x) {
    if (x <= -24.0f) return 0.0f;
    if (x > 8.0f) x = 8.0f;
    int i = (int)((x + 127.0f) * 8388608.0f);
    union { int i; float f; } u;
    u.i = i;
    return u.f;
}

/* ---- Type selection ---- */
#ifdef USE_BF16
  typedef unsigned short wmma_t;
  typedef wmma_t wmma_vec8 __attribute__((ext_vector_type(8)));
  #define WMMA(A,B,C) C = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(A, B, C)
#else
  typedef _Float16 wmma_t;
  typedef wmma_t wmma_vec8 __attribute__((ext_vector_type(8)));
  #define WMMA(A,B,C) C = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(A, B, C)
#endif

/* ----------------------------------------------------------------------- */
/* FP16/BF16 FA3: BQ=256, BKV=32, CH=2, 16-wave, VWA=4                    */
/* ----------------------------------------------------------------------- */
extern "C" __global__ __launch_bounds__(512, 1)
void fa3_b32_16w(float *out, const wmma_t *Q, const wmma_t *K_t, const wmma_t *V_t,
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

    __shared__ wmma_t smK[BKV * HD];
    __shared__ wmma_t smV[HD * BKV];
    __shared__ wmma_t smP[WAVE_CT * CH * 16 * 16];
    wmma_t *smP_w = smP + wid * CH * 16 * 16;

    /* ---- Load Q (once, from global to registers) ---- */
    wmma_vec8 q_reg[2][4];
    for (int g = 0; g < 2; g++) {
        for (int i = 0; i < 4; i++) {
            int kb = g * 4 + i;
            int q_row = q0 + wid * 16 + idx;
            wmma_vec8 tmp;
            for (int j = 0; j < 8; j++) {
                int d = kb * 16 + half * 8 + j;
                tmp[j] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (wmma_t)0;
            }
            q_reg[g][i] = tmp;
        }
    }

    /* ---- Initialize output accumulators and online-softmax state ---- */
    float8 O_acc[K_NB];
    for (int kb = 0; kb < K_NB; kb++)
        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;
    float m_i[8], l_i[8];
    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }

    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    int total = BKV * HD;
    int load_stride = (WAVE_CT * 32) * 16;

    for (int t = 0; t < n_kv_tiles; t++) {
        int kv0 = t * BKV;

        /* ---- Vectorized K/V load (16 wmma_t elements per thread) ---- */
        for (int off = tid * 16; off < total; off += load_stride) {
            int r = off / HD;
            int d = off % HD;
            int kv = kv0 + r;
            size_t base = ((size_t)h * n_tok + kv) * HD + d;
            int kv_ok = (kv < n_tok);
            int in_bounds = (kv_ok && d + 16 <= HD);
            if (in_bounds) {
                wmma_vec8 kd0 = *(const wmma_vec8 *)(K_t + base);
                wmma_vec8 kd1 = *(const wmma_vec8 *)(K_t + base + 8);
                wmma_vec8 vd0 = *(const wmma_vec8 *)(V_t + base);
                wmma_vec8 vd1 = *(const wmma_vec8 *)(V_t + base + 8);
                *(wmma_vec8 *)(smK + r * HD + d) = kd0;
                *(wmma_vec8 *)(smK + r * HD + d + 8) = kd1;
                for (int i = 0; i < 8; i++)
                    smV[(d + i) * BKV + r] = vd0[i];
                for (int i = 0; i < 8; i++)
                    smV[(d + 8 + i) * BKV + r] = vd1[i];
            } else {
                for (int i = 0; i < 16; i++) {
                    wmma_t kk = 0, vv = 0;
                    if (kv_ok && d + i < HD) {
                        kk = K_t[base + i];
                        vv = V_t[base + i];
                    }
                    smK[r * HD + d + i] = kk;
                    smV[(d + i) * BKV + r] = vv;
                }
            }
        }
        __syncthreads();

        /* ---- QK phase (no K P-transpose needed) ---- */
        float8 S[CH];
        for (int ch = 0; ch < CH; ch++)
            for (int i = 0; i < 8; i++) S[ch][i] = 0.0f;

        for (int g = 0; g < 2; g++) {
            for (int ch = 0; ch < CH; ch++) {
                wmma_vec8 k_reg[4];
                for (int i = 0; i < 4; i++) {
                    int kb = g * 4 + i;
                    int base_off = (ch * 16 + idx) * HD + kb * 16 + half * 8;
                    k_reg[i] = *(const wmma_vec8 *)(smK + base_off);
                }
                for (int i = 0; i < 4; i++)
                    WMMA(q_reg[g][i], k_reg[i], S[ch]);
            }
        }

        /* ---- Online softmax (log2-domain, fast exp2) ---- */
        float row_max[8];
        for (int i = 0; i < 8; i++) {
            float mx = -1e30f;
            for (int ch = 0; ch < CH; ch++) {
                int col = ch * 16 + idx;
                int kv = kv0 + col;
                float s = S[ch][i] * inv_sqrtd * LOG2E;
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

        /* ---- P write to LDS (C-layout → smP, fp16/bf16 elements) ---- */
        for (int ch = 0; ch < CH; ch++) {
            wmma_t *smPc = smP_w + ch * 16 * 16;
            for (int i = 0; i < 8; i++) {
                int m_row = half * 8 + i;
                smPc[m_row * 16 + idx] = (wmma_t)S[ch][i];
            }
        }
        __builtin_amdgcn_s_waitcnt(0);

        __syncthreads();

        /* ---- P read (A-layout, transposed) + PV ---- */
        for (int ch = 0; ch < CH; ch++) {
            wmma_t *smPc = smP_w + ch * 16 * 16;
            wmma_vec8 ap = *(const wmma_vec8 *)(smPc + idx * 16 + half * 8);
            for (int kb = 0; kb < K_NB; kb++) {
                int d_col = kb * 16 + idx;
                int v_base_off = d_col * BKV + ch * 16 + half * 8;
                wmma_vec8 bv = *(const wmma_vec8 *)(smV + v_base_off);
                WMMA(ap, bv, O_acc[kb]);
            }
        }
    }

    /* ---- Epilogue: normalize by l_i, write FP32 output ---- */
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
                out[(size_t)row * dim + h * HD + d] = O_acc[kb][i] * l_i[i];
        }
    }
}

/* ----------------------------------------------------------------------- */
/* FP16/BF16 FA3: PGR=2 double-buffered K/V (BQ=256, BKV=32, CH=2, 16w).  */
/* Overlap K/V global load with compute. Same LDS layout but double.       */
/* smK[2][BKV*HD] = 16 KB, smV[2][HD*BKV] = 16 KB, smP = 16 KB → 48 KB.  */
/* ----------------------------------------------------------------------- */
extern "C" __global__ __launch_bounds__(512, 1)
void fa3_b32_16w_pgr2(float *out, const wmma_t *Q, const wmma_t *K_t, const wmma_t *V_t,
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

    __shared__ wmma_t smK[DBUF][BKV * HD];
    __shared__ wmma_t smV[DBUF][HD * BKV];
    __shared__ wmma_t smP[WAVE_CT * CH * 16 * 16];
    wmma_t *smP_w = smP + wid * CH * 16 * 16;

    wmma_vec8 q_reg[2][4];
    for (int g = 0; g < 2; g++) {
        for (int i = 0; i < 4; i++) {
            int kb = g * 4 + i;
            int q_row = q0 + wid * 16 + idx;
            wmma_vec8 tmp;
            for (int j = 0; j < 8; j++) {
                int d = kb * 16 + half * 8 + j;
                tmp[j] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (wmma_t)0;
            }
            q_reg[g][i] = tmp;
        }
    }

    float8 O_acc[K_NB];
    for (int kb = 0; kb < K_NB; kb++)
        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;
    float m_i[8], l_i[8];
    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }

    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    int total = BKV * HD;
    int load_stride = (WAVE_CT * 32) * 16;

    /* Lambda to load K/V into a specified buffer */
    auto load_kv = [&](wmma_t *dstK, wmma_t *dstV, int kv_base) {
        for (int off = tid * 16; off < total; off += load_stride) {
            int r = off / HD;
            int d = off % HD;
            int kv = kv_base + r;
            size_t base = ((size_t)h * n_tok + kv) * HD + d;
            int kv_ok = (kv < n_tok);
            int in_bounds = (kv_ok && d + 16 <= HD);
            if (in_bounds) {
                wmma_vec8 kd0 = *(const wmma_vec8 *)(K_t + base);
                wmma_vec8 kd1 = *(const wmma_vec8 *)(K_t + base + 8);
                wmma_vec8 vd0 = *(const wmma_vec8 *)(V_t + base);
                wmma_vec8 vd1 = *(const wmma_vec8 *)(V_t + base + 8);
                *(wmma_vec8 *)(dstK + r * HD + d) = kd0;
                *(wmma_vec8 *)(dstK + r * HD + d + 8) = kd1;
                for (int i = 0; i < 8; i++)
                    dstV[(d + i) * BKV + r] = vd0[i];
                for (int i = 0; i < 8; i++)
                    dstV[(d + 8 + i) * BKV + r] = vd1[i];
            } else {
                for (int i = 0; i < 16; i++) {
                    wmma_t kk = 0, vv = 0;
                    if (kv_ok && d + i < HD) {
                        kk = K_t[base + i];
                        vv = V_t[base + i];
                    }
                    dstK[r * HD + d + i] = kk;
                    dstV[(d + i) * BKV + r] = vv;
                }
            }
        }
    };

    /* Pre-fetch first tile */
    load_kv(smK[0], smV[0], 0);
    __syncthreads();

    for (int t = 0; t < n_kv_tiles; t++) {
        int cur = t & 1;
        int nxt = cur ^ 1;
        int kv0 = t * BKV;
        wmma_t *curK = smK[cur];
        wmma_t *curV = smV[cur];

        /* Prefetch next tile (overlaps with compute) */
        if (t + 1 < n_kv_tiles)
            load_kv(smK[nxt], smV[nxt], (t + 1) * BKV);

        float8 S[CH];
        for (int ch = 0; ch < CH; ch++)
            for (int i = 0; i < 8; i++) S[ch][i] = 0.0f;

        for (int g = 0; g < 2; g++) {
            for (int ch = 0; ch < CH; ch++) {
                wmma_vec8 k_reg[4];
                for (int i = 0; i < 4; i++) {
                    int kb = g * 4 + i;
                    int base_off = (ch * 16 + idx) * HD + kb * 16 + half * 8;
                    k_reg[i] = *(const wmma_vec8 *)(curK + base_off);
                }
                for (int i = 0; i < 4; i++)
                    WMMA(q_reg[g][i], k_reg[i], S[ch]);
            }
        }

        float row_max[8];
        for (int i = 0; i < 8; i++) {
            float mx = -1e30f;
            for (int ch = 0; ch < CH; ch++) {
                int col = ch * 16 + idx;
                int kv = kv0 + col;
                float s = S[ch][i] * inv_sqrtd * LOG2E;
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
            wmma_t *smPc = smP_w + ch * 16 * 16;
            for (int i = 0; i < 8; i++) {
                int m_row = half * 8 + i;
                smPc[m_row * 16 + idx] = (wmma_t)S[ch][i];
            }
        }
        __builtin_amdgcn_s_waitcnt(0);

        __syncthreads();

        for (int ch = 0; ch < CH; ch++) {
            wmma_t *smPc = smP_w + ch * 16 * 16;
            wmma_vec8 ap = *(const wmma_vec8 *)(smPc + idx * 16 + half * 8);
            for (int kb = 0; kb < K_NB; kb++) {
                int d_col = kb * 16 + idx;
                int v_base_off = d_col * BKV + ch * 16 + half * 8;
                wmma_vec8 bv = *(const wmma_vec8 *)(curV + v_base_off);
                WMMA(ap, bv, O_acc[kb]);
            }
        }
    }

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
                out[(size_t)row * dim + h * HD + d] = O_acc[kb][i] * l_i[i];
        }
    }
}

/* ----------------------------------------------------------------------- */
/* FP16/BF16 FA3: BQ=256, BKV=64, CH=4, 16-wave, VWA=4                    */
/* BKV=64 → half the tiles (64 vs 128), fewer sync points.                 */
/* smP: 16×4×256×2 = 32 KB → total LDS = 16+16+32 = 64 KB (exact limit).  */
/* ----------------------------------------------------------------------- */
extern "C" __global__ __launch_bounds__(512, 1)
void fa3_b64_16w(float *out, const wmma_t *Q, const wmma_t *K_t, const wmma_t *V_t,
                     int n_tok, int n_heads, float inv_sqrtd) {
    enum { BQ = 256, BKV = 64, WAVE_CT = 16, CH = 4 };
    int h  = blockIdx.x;
    int qb = blockIdx.y;
    int q0 = qb * BQ;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int half = lid >> 4;
    int idx = lid & 15;
    int dim = n_heads * HD;

    __shared__ wmma_t smK[BKV * HD];
    __shared__ wmma_t smV[HD * BKV];
    __shared__ wmma_t smP[WAVE_CT * CH * 16 * 16];
    wmma_t *smP_w = smP + wid * CH * 16 * 16;

    wmma_vec8 q_reg[2][4];
    for (int g = 0; g < 2; g++) {
        for (int i = 0; i < 4; i++) {
            int kb = g * 4 + i;
            int q_row = q0 + wid * 16 + idx;
            wmma_vec8 tmp;
            for (int j = 0; j < 8; j++) {
                int d = kb * 16 + half * 8 + j;
                tmp[j] = (q_row < n_tok) ? Q[(size_t)q_row * dim + h * HD + d] : (wmma_t)0;
            }
            q_reg[g][i] = tmp;
        }
    }

    float8 O_acc[K_NB];
    for (int kb = 0; kb < K_NB; kb++)
        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;
    float m_i[8], l_i[8];
    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }

    int n_kv_tiles = (n_tok + BKV - 1) / BKV;
    int total = BKV * HD;
    int load_stride = (WAVE_CT * 32) * 16;

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
                wmma_vec8 kd0 = *(const wmma_vec8 *)(K_t + base);
                wmma_vec8 kd1 = *(const wmma_vec8 *)(K_t + base + 8);
                wmma_vec8 vd0 = *(const wmma_vec8 *)(V_t + base);
                wmma_vec8 vd1 = *(const wmma_vec8 *)(V_t + base + 8);
                *(wmma_vec8 *)(smK + r * HD + d) = kd0;
                *(wmma_vec8 *)(smK + r * HD + d + 8) = kd1;
                for (int i = 0; i < 8; i++)
                    smV[(d + i) * BKV + r] = vd0[i];
                for (int i = 0; i < 8; i++)
                    smV[(d + 8 + i) * BKV + r] = vd1[i];
            } else {
                for (int i = 0; i < 16; i++) {
                    wmma_t kk = 0, vv = 0;
                    if (kv_ok && d + i < HD) {
                        kk = K_t[base + i];
                        vv = V_t[base + i];
                    }
                    smK[r * HD + d + i] = kk;
                    smV[(d + i) * BKV + r] = vv;
                }
            }
        }
        __syncthreads();

        float8 S[CH];
        for (int ch = 0; ch < CH; ch++)
            for (int i = 0; i < 8; i++) S[ch][i] = 0.0f;

        for (int g = 0; g < 2; g++) {
            for (int ch = 0; ch < CH; ch++) {
                wmma_vec8 k_reg[4];
                for (int i = 0; i < 4; i++) {
                    int kb = g * 4 + i;
                    int base_off = (ch * 16 + idx) * HD + kb * 16 + half * 8;
                    k_reg[i] = *(const wmma_vec8 *)(smK + base_off);
                }
                for (int i = 0; i < 4; i++)
                    WMMA(q_reg[g][i], k_reg[i], S[ch]);
            }
        }

        float row_max[8];
        for (int i = 0; i < 8; i++) {
            float mx = -1e30f;
            for (int ch = 0; ch < CH; ch++) {
                int col = ch * 16 + idx;
                int kv = kv0 + col;
                float s = S[ch][i] * inv_sqrtd * LOG2E;
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
            wmma_t *smPc = smP_w + ch * 16 * 16;
            for (int i = 0; i < 8; i++) {
                int m_row = half * 8 + i;
                smPc[m_row * 16 + idx] = (wmma_t)S[ch][i];
            }
        }
        __builtin_amdgcn_s_waitcnt(0);

        __syncthreads();

        for (int ch = 0; ch < CH; ch++) {
            wmma_t *smPc = smP_w + ch * 16 * 16;
            wmma_vec8 ap = *(const wmma_vec8 *)(smPc + idx * 16 + half * 8);
            for (int kb = 0; kb < K_NB; kb++) {
                int d_col = kb * 16 + idx;
                int v_base_off = d_col * BKV + ch * 16 + half * 8;
                wmma_vec8 bv = *(const wmma_vec8 *)(smV + v_base_off);
                WMMA(ap, bv, O_acc[kb]);
            }
        }
    }

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
                out[(size_t)row * dim + h * HD + d] = O_acc[kb][i] * l_i[i];
        }
    }
}
)FA16SRC";

/* ======================================================================== */
/* Host harness                                                              */
/* ======================================================================== */

typedef struct {
    const char *name;
    hipFunction_t func;
    int block_size;
    int bq;
} fa16_kernel_desc;

#define FA16_MAX_KERNELS 8

typedef struct {
    fa16_kernel_desc descs[FA16_MAX_KERNELS];
    int count;
} fa16_kernels;

static int add_kernel(fa16_kernels *k, hipModule_t mod,
                      const char *name, const char *symbol,
                      int block_size, int bq) {
    if (k->count >= FA16_MAX_KERNELS) return -1;
    fa16_kernel_desc *d = &k->descs[k->count++];
    d->name = name;
    d->func = NULL;
    d->block_size = block_size;
    d->bq = bq;
    HIP_CHECK(hipModuleGetFunction(&d->func, mod, symbol));
    return 0;
}

static fa16_kernel_desc *lookup_kernel(fa16_kernels *k, const char *mode) {
    for (int i = 0; i < k->count; i++)
        if (!strcmp(k->descs[i].name, mode))
            return &k->descs[i];
    return NULL;
}

/* ---- Modified compile: passes extra defines to HIPRTC ---- */

static int compile_fa16_module(hipModule_t *mod, int device_id,
                               const char *source, const char *prog_name,
                               int verbose, const char *prefix,
                               int use_bf16) {
    const char *arch = rocewGetRDNA4ArchString(device_id);
    if (!arch) {
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, device_id) != hipSuccess) {
            fprintf(stderr, "%s: cannot query device properties\n", prefix);
            return -1;
        }
        arch = props.gcnArchName;
    }

    if (verbose >= 1)
        fprintf(stderr, "%s: compiling kernels for %s (%s) ...\n",
                prefix, arch, use_bf16 ? "BF16" : "FP16");

    hiprtcProgram prog;
    hiprtcResult cres = hiprtcCreateProgram(&prog, source, prog_name, 0, NULL, NULL);
    if (cres != HIPRTC_SUCCESS) {
        fprintf(stderr, "%s: hiprtcCreateProgram failed: %d\n", prefix, (int)cres);
        return -1;
    }

    char arch_flag[64];
    snprintf(arch_flag, sizeof(arch_flag), "--gpu-architecture=%s", arch);
    const char *base_opts[] = { arch_flag, "-O3", "-ffast-math" };
    const char *bf16_opt = "-DUSE_BF16";
    int nopts = use_bf16 ? 4 : 3;
    const char *opts[4];
    opts[0] = base_opts[0];
    opts[1] = base_opts[1];
    opts[2] = base_opts[2];
    if (use_bf16) opts[3] = bf16_opt;

    hiprtcResult nres = hiprtcCompileProgram(prog, nopts, opts);

    if (nres != HIPRTC_SUCCESS) {
        fprintf(stderr, "%s: HIPRTC compile error %d\n", prefix, (int)nres);
        size_t log_sz;
        hiprtcGetProgramLogSize(prog, &log_sz);
        char *log = (char *)malloc(log_sz + 1);
        hiprtcGetProgramLog(prog, log);
        log[log_sz] = '\0';
        fprintf(stderr, "%s: HIPRTC log (%zu bytes):\n%s\n", prefix, log_sz, log);
        free(log);
        hiprtcDestroyProgram(&prog);
        return -1;
    }

    size_t code_sz = 0;
    hiprtcGetCodeSize(prog, &code_sz);
    char *code = (char *)malloc(code_sz);
    hiprtcGetCode(prog, code);
    hiprtcDestroyProgram(&prog);

    if (verbose >= 2)
        fprintf(stderr, "%s: loading code object (%zu bytes)\n", prefix, code_sz);

    if (verbose >= 3) {
        char path[256];
        snprintf(path, sizeof(path), "/tmp/%s.co", prog_name);
        FILE *fp = fopen(path, "wb");
        if (fp) { fwrite(code, 1, code_sz, fp); fclose(fp);
            fprintf(stderr, "%s: code object saved to %s\n", prefix, path); }
    }

    hipError_t err = hipModuleLoadData(mod, code);
    free(code);
    if (err != hipSuccess) {
        fprintf(stderr, "%s: hipModuleLoadData failed: %d\n", prefix, (int)err);
        return -1;
    }
    return 1;
}

/* ---- Run a kernel ---- */

static int run_shape_f16(fa16_kernel_desc *kd,
                         int n_tok, int n_heads, int iters, int do_check,
                         float abs_max, double peak_tfs) {
    if (n_tok % 64) {
        fprintf(stderr, "skip: n_tok must be %%64 (got %d)\n", n_tok);
        return 0;
    }
    const int head_dim = 128;
    int dim = n_heads * head_dim;
    size_t qkv_elems = (size_t)n_tok * dim;
    size_t qkv_bytes = qkv_elems * sizeof(uint16_t); /* fp16/bf16 = 2 bytes each */
    size_t out_elems = (size_t)n_tok * dim;
    size_t out_bytes = out_elems * sizeof(float);

    uint16_t *hQ  = (uint16_t *)malloc(qkv_bytes);
    uint16_t *hKt = (uint16_t *)malloc(qkv_bytes);
    uint16_t *hVt = (uint16_t *)malloc(qkv_bytes);
    if (!hQ || !hKt || !hVt) {
        fprintf(stderr, "host allocation failed\n");
        return -1;
    }
    unsigned rng = 0x1234abcd;
    fill_f16_random(hQ,  qkv_elems, abs_max, &rng);
    fill_f16_random(hKt, qkv_elems, abs_max, &rng);
    fill_f16_random(hVt, qkv_elems, abs_max, &rng);

    void *dQ  = hip_upload_raw(hQ,  qkv_bytes);
    void *dKt = hip_upload_raw(hKt, qkv_bytes);
    void *dVt = hip_upload_raw(hVt, qkv_bytes);
    void *dO  = NULL;
    HIP_CHECK(hipMalloc(&dO, out_bytes));
    HIP_CHECK(hipMemset(dO, 0, out_bytes));

    float inv_sqrtd = 1.0f / sqrtf((float)head_dim);
    int bq = kd->bq;
    dim3 block(kd->block_size, 1, 1);
    dim3 grid((unsigned)n_heads, (unsigned)((n_tok + bq - 1) / bq), 1);

    void *args[] = { &dO, &dQ, &dKt, &dVt, &n_tok, &n_heads, &inv_sqrtd };

    /* Warmup */
    HIP_CHECK(hipModuleLaunchKernel(kd->func, grid.x, grid.y, grid.z,
                                    block.x, block.y, block.z, 0, NULL, args, NULL));
    HIP_CHECK(hipDeviceSynchronize());

    double cos = -2.0;
    float maxd = -1.0f;
    if (do_check) {
        float *hO   = (float *)malloc(out_bytes);
        float *hRef = (float *)malloc(out_elems * sizeof(float));
        HIP_CHECK(hipMemcpy(hO, dO, out_bytes, hipMemcpyDeviceToHost));

        /* Convert fp16→fp32 for reference */
        float *fQ  = (float *)malloc(qkv_elems * sizeof(float));
        float *fKt = (float *)malloc(qkv_elems * sizeof(float));
        float *fVt = (float *)malloc(qkv_elems * sizeof(float));
        for (size_t i = 0; i < qkv_elems; i++) {
            fQ[i]  = f16_to_f32(hQ[i]);
            fKt[i] = f16_to_f32(hKt[i]);
            fVt[i] = f16_to_f32(hVt[i]);
        }
        fa_ref_fp32_gen(hRef, fQ, fKt, fVt, n_tok, n_heads, head_dim);
        cos  = cosine_sim(hO, hRef, out_elems);
        maxd = max_abs_diff(hO, hRef, out_elems);
        free(hO); free(hRef);
        free(fQ); free(fKt); free(fVt);
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
    hipFree(dQ); hipFree(dKt); hipFree(dVt);
    free(hQ); free(hKt); free(hVt);
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [--n-tok N] [--heads H] [--iters N] [--mode MODE]\n"
        "          [--check] [--abs-max V] [--peak-tfs V] [--verbose] [--dump-code]\n"
        "  --mode      f32 | fp2 | b32 | bp2 | all\n"
        "  --n-tok     token count, must be %%64 (default 1024)\n"
        "  --heads     attention heads (default 16)\n"
        "  --iters     timing iterations (default 100)\n"
        "  --check     compare against CPU FP32 reference (slow)\n"
        "  --abs-max   random input clamp (default 1.0)\n"
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
        if      (!strcmp(argv[i], "--n-tok")    && i+1<argc) n_tok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads")    && i+1<argc) n_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")    && i+1<argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode")     && i+1<argc) mode = argv[++i];
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

    hipModule_t mod_f16, mod_bf16;
    if (compile_fa16_module(&mod_f16, dev, kernel_src, "fa3_f16",
                            verbose, "fa3_f16", 0) < 0) return 1;
    if (compile_fa16_module(&mod_bf16, dev, kernel_src, "fa3_bf16",
                            verbose, "fa3_bf16", 1) < 0) return 1;

    fa16_kernels kernels = {0};
    if (add_kernel(&kernels, mod_f16, "f32", "fa3_b32_16w", 512, 256) < 0) return 1;
    if (add_kernel(&kernels, mod_f16, "fp2", "fa3_b32_16w_pgr2", 512, 256) < 0) return 1;
    if (add_kernel(&kernels, mod_f16, "f64", "fa3_b64_16w", 512, 256) < 0) return 1;
    if (add_kernel(&kernels, mod_bf16, "b32", "fa3_b32_16w", 512, 256) < 0) return 1;
    if (add_kernel(&kernels, mod_bf16, "bp2", "fa3_b32_16w_pgr2", 512, 256) < 0) return 1;
    if (add_kernel(&kernels, mod_bf16, "b64", "fa3_b64_16w", 512, 256) < 0) return 1;

    printf("rdna4/fa3 FP16/BF16 FA3 bench (iters=%d, abs_max=%.2f, peak=%.1f TF/s%s)\n",
           iters, abs_max, peak_tfs, do_check ? ", check=on" : "");

    if (!strcmp(mode, "all")) {
        for (int i = 0; i < kernels.count; i++) {
            /* Skip BKV=64 variants (LDS overflow at 64KB limit) */
            if (!strcmp(kernels.descs[i].name, "f64") ||
                !strcmp(kernels.descs[i].name, "b64"))
                continue;
            if (run_shape_f16(&kernels.descs[i], n_tok, n_heads, iters,
                              do_check, abs_max, peak_tfs) < 0)
                return 1;
        }
    } else {
        fa16_kernel_desc *kd = lookup_kernel(&kernels, mode);
        if (!kd) { fprintf(stderr, "unknown mode '%s' (try f32/fp2/b32/bp2/all)\n", mode); return 1; }
        if (run_shape_f16(kd, n_tok, n_heads, iters,
                          do_check, abs_max, peak_tfs) < 0)
            return 1;
    }
    return 0;
}
