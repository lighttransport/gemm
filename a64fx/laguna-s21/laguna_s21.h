/* Laguna S 2.1 INT4 shared scalar/SVE-safe primitives. */
#ifndef LAGUNA_S21_H
#define LAGUNA_S21_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

enum {
    LAGUNA_LAYERS = 48, LAGUNA_HIDDEN = 3072, LAGUNA_VOCAB = 100352,
    LAGUNA_KV_HEADS = 8, LAGUNA_HEAD_DIM = 128, LAGUNA_EXPERTS = 256,
    LAGUNA_ACTIVE = 10, LAGUNA_EXPERT_INTER = 1024, LAGUNA_GROUP = 32,
    LAGUNA_SLIDING_WINDOW = 512,
    /* Ring capacity for sliding layers.  Must exceed the window by at least the
     * prefill chunk size, so a whole chunk's K/V can be written BEFORE any of the
     * chunk's queries attend (which is what lets sliding attention be query-blocked
     * like the full layers).  With cap=768 and C=256 the ring retains positions
     * [p-512, p] while the oldest needed is p-511 -- valid with one slot to spare.
     * NB: capacity != window.  The attended range is always SLIDING_WINDOW. */
    LAGUNA_SLIDING_CAP = 768,
    LAGUNA_DENSE_INTER = 12288, LAGUNA_SHARED_INTER = 1024,
    LAGUNA_FULL_HEADS = 48, LAGUNA_SLIDING_HEADS = 72,
    LAGUNA_ROPE_FULL_DIM = 64, LAGUNA_ROPE_SLIDING_DIM = 128,
    LAGUNA_MAX_HEADS = 72,
    LAGUNA_MAX_BATCH = 16      /* concurrent sequences the serve path can step together */
};
#define LAGUNA_RMS_EPS      1e-6f
#define LAGUNA_ROUTED_SCALE 2.5f
/* YaRN (full_attention) rope parameters, from config.json rope_parameters.full_attention */
#define LAGUNA_YARN_THETA   500000.0
#define LAGUNA_YARN_FACTOR  32.0
#define LAGUNA_YARN_BETA_FAST 32.0
#define LAGUNA_YARN_BETA_SLOW 1.0
#define LAGUNA_YARN_ORIG_MAX  8192.0
#define LAGUNA_YARN_ATTN_FACTOR 1.3465735902799727
/* default rope (sliding_attention) */
#define LAGUNA_SWA_THETA    10000.0

static inline float laguna_bf16_to_f32(uint16_t x) {
    uint32_t u = (uint32_t)x << 16; float f; memcpy(&f, &u, sizeof f); return f;
}
static inline uint16_t laguna_f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, sizeof u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* compressed-tensors pack-quantized symmetric INT4: eight low-to-high nibbles
 * per little-endian I32.  Each output row has one BF16 scale per 32 input values. */
static inline int laguna_i4_at(const uint32_t *packed, int col) {
    uint32_t w = packed[(unsigned)col >> 3];
    return (int)((w >> (4 * (col & 7))) & 15u) - 8;
}
static inline float laguna_i4g32_dot(const uint32_t *packed, const uint16_t *scales,
                                     const float *x, int cols) {
    float sum = 0.0f;
    for (int g = 0; g < cols / LAGUNA_GROUP; ++g) {
        float s = laguna_bf16_to_f32(scales[g]);
        float part = 0.0f;
        for (int j = 0; j < LAGUNA_GROUP; ++j)
            part += (float)laguna_i4_at(packed, g * LAGUNA_GROUP + j) * x[g * LAGUNA_GROUP + j];
        sum += s * part;
    }
    return sum;
}

/* Normalized Walsh-Hadamard transform over a 128-wide block. The checkpoint's
 * transform config declares head_dim=128, so dimensions are transformed blockwise. */
static inline void laguna_fht128(float *v) {
    for (int step = 1; step < 128; step <<= 1)
        for (int i = 0; i < 128; i += 2 * step)
            for (int j = 0; j < step; ++j) {
                float a = v[i + j], b = v[i + j + step];
                v[i + j] = a + b; v[i + j + step] = a - b;
            }
    const float scale = 0.08838834764831843f; /* 1/sqrt(128) */
    for (int i = 0; i < 128; ++i) v[i] *= scale;
}
static inline void laguna_fht_blocks(float *v, int n) {
    for (int i = 0; i < n; i += 128) laguna_fht128(v + i);
}

typedef struct { float score; int id; } laguna_route;
static inline void laguna_top10(const float *logits, const float *bias, int out_id[LAGUNA_ACTIVE], float out_w[LAGUNA_ACTIVE]) {
    laguna_route best[LAGUNA_ACTIVE];
    for (int i = 0; i < LAGUNA_ACTIVE; ++i) best[i] = (laguna_route){-INFINITY, -1};
    for (int e = 0; e < LAGUNA_EXPERTS; ++e) {
        float p = 1.0f / (1.0f + expf(-logits[e]));
        float s = p + bias[e];
        for (int k = 0; k < LAGUNA_ACTIVE; ++k) if (s > best[k].score || (s == best[k].score && e < best[k].id)) {
            for (int q = LAGUNA_ACTIVE - 1; q > k; --q) best[q] = best[q - 1];
            best[k] = (laguna_route){s, e}; break;
        }
    }
    float z = 0.0f;
    for (int k = 0; k < LAGUNA_ACTIVE; ++k) { out_id[k] = best[k].id; out_w[k] = 1.0f / (1.0f + expf(-logits[best[k].id])); z += out_w[k]; }
    for (int k = 0; k < LAGUNA_ACTIVE; ++k) out_w[k] /= z;
}

static inline int laguna_expert_owner(int expert, int ep_size) { return expert % ep_size; }

/* ===================================================================== *
 *  Forward-pass primitives (correctness-first; f32 accumulation).       *
 * ===================================================================== */

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
/* Widen a contiguous bf16 vector to f32: (uint16 << 16) reinterpreted as float. */
static inline svfloat32_t laguna_ld_bf16(svbool_t pg, const uint16_t *p) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, p), 16));
}
/* 2^x via the A64FX FEXPA accelerator (adapted from glm5_impl.h). */
static inline svfloat32_t laguna_exp2_fexpa(svbool_t pg, svfloat32_t x) {
    const float shift_f = 204927.0f;               /* 0x48481fc0: FEXPA rounding shift */
    svfloat32_t shift = svdup_f32(shift_f);
    svfloat32_t z = svadd_f32_x(pg, x, shift);
    svfloat32_t n = svsub_f32_x(pg, z, shift);
    svfloat32_t r = svsub_f32_x(pg, x, n);
    svfloat32_t scale = svexpa_f32(svreinterpret_u32_f32(z));
    svfloat32_t corr = svmla_n_f32_x(pg, svdup_f32(1.0f), r, 0.6931471805599453f);
    return svmul_f32_x(pg, scale, corr);
}
/* s[i] = exp(s[i] - shift) for i in [0,n); returns sum.  Vectorized softmax expf
 * (the O(context) prefill bottleneck).  e^x = 2^(x*log2e). */
static inline float laguna_exp_shift_sum(float *restrict s, int n, float shift) {
    svfloat32_t acc = svdup_f32(0);
    svfloat32_t sh = svdup_f32(shift), l2e = svdup_f32(1.4426950408889634f);
    for (int i=0;i<n;i+=(int)svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1_f32(pg, s+i);
        v = laguna_exp2_fexpa(pg, svmul_f32_x(pg, svsub_f32_x(pg, v, sh), l2e));
        svst1_f32(pg, s+i, v);
        acc = svadd_f32_m(pg, acc, v);
    }
    return svaddv_f32(svptrue_b32(), acc);
}
/* Compute rows [r, r+nr) (nr<=4) of y = W*x, SVE-widened bf16, sharing each x load. */
static inline void laguna_bf16_rowblock(float *restrict y, const uint16_t *restrict W,
                                        const float *restrict x, int r, int nr, int cols) {
    if (nr == 4) {
        const uint16_t *w0=W+(size_t)r*cols, *w1=w0+cols, *w2=w1+cols, *w3=w2+cols;
        svfloat32_t a0=svdup_f32(0), a1=svdup_f32(0), a2=svdup_f32(0), a3=svdup_f32(0);
        for (int c = 0; c < cols; c += (int)svcntw()) {
            svbool_t pg = svwhilelt_b32(c, cols);
            svfloat32_t xf = svld1_f32(pg, x + c);
            a0 = svmla_f32_x(pg, a0, laguna_ld_bf16(pg, w0+c), xf);
            a1 = svmla_f32_x(pg, a1, laguna_ld_bf16(pg, w1+c), xf);
            a2 = svmla_f32_x(pg, a2, laguna_ld_bf16(pg, w2+c), xf);
            a3 = svmla_f32_x(pg, a3, laguna_ld_bf16(pg, w3+c), xf);
        }
        svbool_t pt = svptrue_b32();
        y[r]=svaddv_f32(pt,a0); y[r+1]=svaddv_f32(pt,a1);
        y[r+2]=svaddv_f32(pt,a2); y[r+3]=svaddv_f32(pt,a3);
    } else {
        for (int rr = r; rr < r+nr; ++rr) {
            const uint16_t *w = W + (size_t)rr * cols;
            svfloat32_t a = svdup_f32(0);
            for (int c = 0; c < cols; c += (int)svcntw()) {
                svbool_t pg = svwhilelt_b32(c, cols);
                a = svmla_f32_x(pg, a, laguna_ld_bf16(pg, w+c), svld1_f32(pg, x+c));
            }
            y[rr] = svaddv_f32(svptrue_b32(), a);
        }
    }
}
/* Batched matvec (chunked prefill): Y[C][rows] = X[C][cols] @ W[rows][cols]^T,
 * token-major (Y[c*rows+r], X[c*cols]).  8 tokens share each weight-row load, so
 * the weight bandwidth is amortized ~8x vs C separate matvecs. */
static inline void laguna_matmat_bf16(float *restrict Y, const uint16_t *restrict W,
                                      const float *restrict X, int rows, int cols, int C) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r=0;r<rows;++r) {
        const uint16_t *w=W+(size_t)r*cols; int ct=0;
        for (; ct+8<=C; ct+=8) {
            const float *x0=X+(size_t)(ct+0)*cols,*x1=X+(size_t)(ct+1)*cols,*x2=X+(size_t)(ct+2)*cols,*x3=X+(size_t)(ct+3)*cols;
            const float *x4=X+(size_t)(ct+4)*cols,*x5=X+(size_t)(ct+5)*cols,*x6=X+(size_t)(ct+6)*cols,*x7=X+(size_t)(ct+7)*cols;
            svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
            svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
            for (int c=0;c<cols;c+=(int)svcntw()) {
                svbool_t pg=svwhilelt_b32(c,cols); svfloat32_t wv=laguna_ld_bf16(pg,w+c);
                a0=svmla_f32_x(pg,a0,wv,svld1_f32(pg,x0+c)); a1=svmla_f32_x(pg,a1,wv,svld1_f32(pg,x1+c));
                a2=svmla_f32_x(pg,a2,wv,svld1_f32(pg,x2+c)); a3=svmla_f32_x(pg,a3,wv,svld1_f32(pg,x3+c));
                a4=svmla_f32_x(pg,a4,wv,svld1_f32(pg,x4+c)); a5=svmla_f32_x(pg,a5,wv,svld1_f32(pg,x5+c));
                a6=svmla_f32_x(pg,a6,wv,svld1_f32(pg,x6+c)); a7=svmla_f32_x(pg,a7,wv,svld1_f32(pg,x7+c));
            }
            svbool_t pt=svptrue_b32();
            Y[(size_t)(ct+0)*rows+r]=svaddv_f32(pt,a0); Y[(size_t)(ct+1)*rows+r]=svaddv_f32(pt,a1);
            Y[(size_t)(ct+2)*rows+r]=svaddv_f32(pt,a2); Y[(size_t)(ct+3)*rows+r]=svaddv_f32(pt,a3);
            Y[(size_t)(ct+4)*rows+r]=svaddv_f32(pt,a4); Y[(size_t)(ct+5)*rows+r]=svaddv_f32(pt,a5);
            Y[(size_t)(ct+6)*rows+r]=svaddv_f32(pt,a6); Y[(size_t)(ct+7)*rows+r]=svaddv_f32(pt,a7);
        }
        for (; ct<C; ++ct) {
            const float *x=X+(size_t)ct*cols; svfloat32_t a=svdup_f32(0);
            for (int c=0;c<cols;c+=(int)svcntw()){ svbool_t pg=svwhilelt_b32(c,cols);
                a=svmla_f32_x(pg,a,laguna_ld_bf16(pg,w+c),svld1_f32(pg,x+c)); }
            Y[(size_t)ct*rows+r]=svaddv_f32(svptrue_b32(),a);
        }
    }
}

/* Decode matvec: y[rows] = W[rows,cols](bf16) * x[cols](f32). */
static inline void laguna_matvec_bf16(float *restrict y, const uint16_t *restrict W,
                                      const float *restrict x, int rows, int cols) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r += 4)
        laguna_bf16_rowblock(y, W, x, r, rows-r<4?rows-r:4, cols);
}
/* Fused multi-matvec: several y[i]=W[i]*x sharing input x and column count, in ONE
 * parallel region (nowait between matrices) to amortize OpenMP fork/join. */
static inline void laguna_matvec_bf16_multi(float *const *ys, const uint16_t *const *Ws,
                                            const int *rows, int nmat, const float *x, int cols) {
#ifdef _OPENMP
    #pragma omp parallel
    {
        for (int m = 0; m < nmat; ++m) {
            #pragma omp for schedule(static) nowait
            for (int r = 0; r < rows[m]; r += 4)
                laguna_bf16_rowblock(ys[m], Ws[m], x, r, rows[m]-r<4?rows[m]-r:4, cols);
        }
    }
#else
    for (int m = 0; m < nmat; ++m)
        for (int r = 0; r < rows[m]; r += 4)
            laguna_bf16_rowblock(ys[m], Ws[m], x, r, rows[m]-r<4?rows[m]-r:4, cols);
#endif
}
#else
/* y[rows] = W[rows,cols] (bf16, row-major) * x[cols] (f32). */
static inline void laguna_matvec_bf16(float *restrict y, const uint16_t *restrict W,
                                      const float *restrict x, int rows, int cols) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        const uint16_t *w = W + (size_t)r * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; ++c) acc += laguna_bf16_to_f32(w[c]) * x[c];
        y[r] = acc;
    }
}
#endif

/* RMSNorm in-place style: out[n] = (x/rms(x)) * w (bf16 weight). */
static inline void laguna_rmsnorm(float *restrict out, const float *restrict x,
                                  const uint16_t *restrict w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; ++i) out[i] = (x[i] * inv) * laguna_bf16_to_f32(w[i]);
}

static inline float laguna_silu(float x) { return x / (1.0f + expf(-x)); }
static inline float laguna_softplus(float x) {
    /* numerically stable log(1+e^x) */
    return x > 20.0f ? x : log1pf(expf(x));
}

/* GPT-NeoX ("half") rotary applied to the first rotary_dim entries of a head
 * vector.  cos/sin hold rotary_dim/2 entries each (freqs, not duplicated). */
static inline void laguna_rope_half(float *restrict v, const float *restrict cosp,
                                    const float *restrict sinp, int rotary_dim) {
    int half = rotary_dim / 2;
    for (int i = 0; i < half; ++i) {
        float a = v[i], b = v[i + half];
        v[i]        = a * cosp[i] - b * sinp[i];
        v[i + half] = b * cosp[i] + a * sinp[i];
    }
}

/* Build YaRN inv_freq[dim/2] for the full-attention rope (matches transformers
 * _compute_yarn_parameters with truncate=True). */
static inline void laguna_yarn_inv_freq(float *inv_freq, int dim) {
    int half = dim / 2;
    double lo_d = (dim * log(LAGUNA_YARN_ORIG_MAX / (LAGUNA_YARN_BETA_FAST * 2.0 * M_PI)))
                  / (2.0 * log(LAGUNA_YARN_THETA));
    double hi_d = (dim * log(LAGUNA_YARN_ORIG_MAX / (LAGUNA_YARN_BETA_SLOW * 2.0 * M_PI)))
                  / (2.0 * log(LAGUNA_YARN_THETA));
    double low = floor(lo_d); if (low < 0) low = 0;
    double high = ceil(hi_d); if (high > dim - 1) high = dim - 1;
    if (low == high) high += 0.001;
    for (int i = 0; i < half; ++i) {
        double pos_freq = pow(LAGUNA_YARN_THETA, (double)(2 * i) / (double)dim);
        double extrap = 1.0 / pos_freq;                       /* extrapolation inv_freq */
        double interp = 1.0 / (LAGUNA_YARN_FACTOR * pos_freq);/* interpolation inv_freq */
        double ramp = ((double)i - low) / (high - low);       /* linear ramp over dim/2 */
        if (ramp < 0) ramp = 0; if (ramp > 1) ramp = 1;
        double extrap_factor = 1.0 - ramp;                    /* 1 - linear_ramp */
        inv_freq[i] = (float)(interp * (1.0 - extrap_factor) + extrap * extrap_factor);
    }
}

/* Default rope inv_freq[dim/2] = 1 / theta^(2i/dim). */
static inline void laguna_default_inv_freq(float *inv_freq, int dim, double theta) {
    int half = dim / 2;
    for (int i = 0; i < half; ++i)
        inv_freq[i] = (float)(1.0 / pow(theta, (double)(2 * i) / (double)dim));
}

/* ---- W8: per-row symmetric int8 weights (halves bf16 weight bandwidth) ---- */
typedef struct { int8_t *q; float *s; } laguna_w8;   /* q[rows*cols], s[rows] */

/* Quantize a bf16 [rows,cols] weight to per-row symmetric int8 + f32 scale.
 * Parallel over rows so the int8/scale pages first-touch on the reader's CMG. */
static inline void laguna_quant_w8(int8_t *q, float *scale, const uint16_t *w,
                                   int rows, int cols) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        const uint16_t *wr = w + (size_t)r * cols;
        float mx = 0.0f;
        for (int c = 0; c < cols; ++c) { float a = fabsf(laguna_bf16_to_f32(wr[c])); if (a>mx) mx=a; }
        float s = mx > 0.0f ? mx / 127.0f : 1.0f;
        scale[r] = s;
        float inv = 1.0f / s;
        int8_t *qr = q + (size_t)r * cols;
        for (int c = 0; c < cols; ++c) {
            int v = (int)lrintf(laguna_bf16_to_f32(wr[c]) * inv);
            qr[c] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
        }
    }
}

#if defined(__ARM_FEATURE_SVE)
/* Rows [r,r+nr) (nr<=8) of y = (Q*x) scaled per row, int8 weights widened to f32.
 * 8 independent accumulators give enough ILP to hide FMA/load latency (the matvec
 * is latency-bound, not bandwidth-bound, on A64FX). */
#define LAGUNA_I8_BLK 8
static inline void laguna_i8_rowblock(float *restrict y, const int8_t *restrict Q,
                                      const float *restrict sc, const float *restrict x,
                                      int r, int nr, int cols) {
    if (nr == LAGUNA_I8_BLK) {
        const int8_t *w0=Q+(size_t)r*cols,*w1=w0+cols,*w2=w1+cols,*w3=w2+cols;
        const int8_t *w4=w3+cols,*w5=w4+cols,*w6=w5+cols,*w7=w6+cols;
        svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
        svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
        for (int c = 0; c < cols; c += (int)svcntw()) {
            svbool_t pg = svwhilelt_b32(c, cols);
            svfloat32_t xf = svld1_f32(pg, x + c);
            a0=svmla_f32_x(pg,a0,svcvt_f32_s32_x(pg,svld1sb_s32(pg,w0+c)),xf);
            a1=svmla_f32_x(pg,a1,svcvt_f32_s32_x(pg,svld1sb_s32(pg,w1+c)),xf);
            a2=svmla_f32_x(pg,a2,svcvt_f32_s32_x(pg,svld1sb_s32(pg,w2+c)),xf);
            a3=svmla_f32_x(pg,a3,svcvt_f32_s32_x(pg,svld1sb_s32(pg,w3+c)),xf);
            a4=svmla_f32_x(pg,a4,svcvt_f32_s32_x(pg,svld1sb_s32(pg,w4+c)),xf);
            a5=svmla_f32_x(pg,a5,svcvt_f32_s32_x(pg,svld1sb_s32(pg,w5+c)),xf);
            a6=svmla_f32_x(pg,a6,svcvt_f32_s32_x(pg,svld1sb_s32(pg,w6+c)),xf);
            a7=svmla_f32_x(pg,a7,svcvt_f32_s32_x(pg,svld1sb_s32(pg,w7+c)),xf);
        }
        svbool_t pt = svptrue_b32();
        y[r]=svaddv_f32(pt,a0)*sc[r];     y[r+1]=svaddv_f32(pt,a1)*sc[r+1];
        y[r+2]=svaddv_f32(pt,a2)*sc[r+2]; y[r+3]=svaddv_f32(pt,a3)*sc[r+3];
        y[r+4]=svaddv_f32(pt,a4)*sc[r+4]; y[r+5]=svaddv_f32(pt,a5)*sc[r+5];
        y[r+6]=svaddv_f32(pt,a6)*sc[r+6]; y[r+7]=svaddv_f32(pt,a7)*sc[r+7];
    } else {
        for (int rr = r; rr < r+nr; ++rr) {
            const int8_t *ww = Q + (size_t)rr * cols;
            svfloat32_t a = svdup_f32(0);
            for (int c = 0; c < cols; c += (int)svcntw()) {
                svbool_t pg = svwhilelt_b32(c, cols);
                a = svmla_f32_x(pg, a, svcvt_f32_s32_x(pg, svld1sb_s32(pg, ww+c)), svld1_f32(pg, x+c));
            }
            y[rr] = svaddv_f32(svptrue_b32(), a) * sc[rr];
        }
    }
}
static inline void laguna_matvec_i8(float *restrict y, const laguna_w8 *w,
                                    const float *restrict x, int rows, int cols) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r += LAGUNA_I8_BLK)
        laguna_i8_rowblock(y, w->q, w->s, x, r, rows-r<LAGUNA_I8_BLK?rows-r:LAGUNA_I8_BLK, cols);
}

static inline void laguna_matvec_i8_multi(float *const *ys, const laguna_w8 *const *ws,
                                          const int *rows, int nmat, const float *x, int cols) {
#ifdef _OPENMP
    #pragma omp parallel
    {
        for (int m = 0; m < nmat; ++m) {
            #pragma omp for schedule(static) nowait
            for (int r = 0; r < rows[m]; r += LAGUNA_I8_BLK)
                laguna_i8_rowblock(ys[m], ws[m]->q, ws[m]->s, x, r, rows[m]-r<LAGUNA_I8_BLK?rows[m]-r:LAGUNA_I8_BLK, cols);
        }
    }
#else
    for (int m = 0; m < nmat; ++m)
        for (int r = 0; r < rows[m]; r += LAGUNA_I8_BLK)
            laguna_i8_rowblock(ys[m], ws[m]->q, ws[m]->s, x, r, rows[m]-r<LAGUNA_I8_BLK?rows[m]-r:LAGUNA_I8_BLK, cols);
#endif
}
/* Batched int8 matvec (chunked prefill): Y[C][rows] = X[C][cols] @ Q^T * scale,
 * token-major.  8 tokens share each weight-row widen, so both the weight
 * bandwidth and the int8->f32 conversion are amortized ~8x vs C matvecs.
 *
 * Tokens are processed in blocks of TB chosen so the live X slice (TB*cols*4 B)
 * stays inside the CMG's 8 MB L2 for the whole row sweep.  Without this, a wide
 * weight makes every row re-stream all C tokens of X from memory: at C=256,
 * o_proj (cols=9216, X=9.4 MB) ran at 66 GMAC/s vs q_proj's (cols=3072,
 * X=3.1 MB) 108, and blocking takes it to 141.  Narrow shapes get TB=C, i.e.
 * exactly the unblocked loop.  Bit-identical either way -- each output's
 * summation order is unchanged, only the order outputs are produced in. */
static inline void laguna_matmat_i8(float *restrict Y, const laguna_w8 *w,
                                    const float *restrict X, int rows, int cols, int C) {
    int TB = (int)(4194304u/((unsigned)cols*4u));   /* ~4 MB of X per block */
    if (TB > C) TB = C;
    TB &= ~7; if (TB < 8) TB = 8;
    for (int t0=0; t0<C; t0+=TB) {
        int TN = C-t0 < TB ? C-t0 : TB;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int r=0;r<rows;++r) {
            const int8_t *q=w->q+(size_t)r*cols; float s=w->s[r]; int ct=0;
            for (; ct+8<=TN; ct+=8) {
                const float *x0=X+(size_t)(t0+ct+0)*cols,*x1=X+(size_t)(t0+ct+1)*cols,
                            *x2=X+(size_t)(t0+ct+2)*cols,*x3=X+(size_t)(t0+ct+3)*cols,
                            *x4=X+(size_t)(t0+ct+4)*cols,*x5=X+(size_t)(t0+ct+5)*cols,
                            *x6=X+(size_t)(t0+ct+6)*cols,*x7=X+(size_t)(t0+ct+7)*cols;
                svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
                svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
                for (int c=0;c<cols;c+=(int)svcntw()) {
                    svbool_t pg=svwhilelt_b32(c,cols);
                    svfloat32_t wv=svcvt_f32_s32_x(pg,svld1sb_s32(pg,q+c));
                    a0=svmla_f32_x(pg,a0,wv,svld1_f32(pg,x0+c)); a1=svmla_f32_x(pg,a1,wv,svld1_f32(pg,x1+c));
                    a2=svmla_f32_x(pg,a2,wv,svld1_f32(pg,x2+c)); a3=svmla_f32_x(pg,a3,wv,svld1_f32(pg,x3+c));
                    a4=svmla_f32_x(pg,a4,wv,svld1_f32(pg,x4+c)); a5=svmla_f32_x(pg,a5,wv,svld1_f32(pg,x5+c));
                    a6=svmla_f32_x(pg,a6,wv,svld1_f32(pg,x6+c)); a7=svmla_f32_x(pg,a7,wv,svld1_f32(pg,x7+c));
                }
                svbool_t pt=svptrue_b32();
                Y[(size_t)(t0+ct+0)*rows+r]=svaddv_f32(pt,a0)*s; Y[(size_t)(t0+ct+1)*rows+r]=svaddv_f32(pt,a1)*s;
                Y[(size_t)(t0+ct+2)*rows+r]=svaddv_f32(pt,a2)*s; Y[(size_t)(t0+ct+3)*rows+r]=svaddv_f32(pt,a3)*s;
                Y[(size_t)(t0+ct+4)*rows+r]=svaddv_f32(pt,a4)*s; Y[(size_t)(t0+ct+5)*rows+r]=svaddv_f32(pt,a5)*s;
                Y[(size_t)(t0+ct+6)*rows+r]=svaddv_f32(pt,a6)*s; Y[(size_t)(t0+ct+7)*rows+r]=svaddv_f32(pt,a7)*s;
            }
            for (; ct<TN; ++ct) {
                const float *x=X+(size_t)(t0+ct)*cols; svfloat32_t a=svdup_f32(0);
                for (int c=0;c<cols;c+=(int)svcntw()){ svbool_t pg=svwhilelt_b32(c,cols);
                    a=svmla_f32_x(pg,a,svcvt_f32_s32_x(pg,svld1sb_s32(pg,q+c)),svld1_f32(pg,x+c)); }
                Y[(size_t)(t0+ct)*rows+r]=svaddv_f32(svptrue_b32(),a)*s;
            }
        }
    }
}
#else /* scalar fallback */
static inline void laguna_matvec_i8(float *restrict y, const laguna_w8 *w,
                                    const float *restrict x, int rows, int cols) {
    for (int r=0;r<rows;++r){ const int8_t*q=w->q+(size_t)r*cols; float a=0;
        for(int c=0;c<cols;++c)a+=(float)q[c]*x[c]; y[r]=a*w->s[r]; }
}
static inline void laguna_matvec_i8_multi(float *const *ys, const laguna_w8 *const *ws,
                                          const int *rows, int nmat, const float *x, int cols) {
    for (int m=0;m<nmat;++m) laguna_matvec_i8(ys[m], ws[m], x, rows[m], cols);
}
static inline void laguna_matmat_i8(float *restrict Y, const laguna_w8 *w,
                                    const float *restrict X, int rows, int cols, int C) {
    for (int c=0;c<C;++c) laguna_matvec_i8(Y+(size_t)c*rows, w, X+(size_t)c*cols, rows, cols);
}
#endif

#if defined(__ARM_FEATURE_SVE)
/* SVE group-32 INT4 dot. Requires a 512-bit VL (16 f32 lanes): 16 packed bytes =
 * 32 nibbles = exactly one group.  Byte b holds col 2b (low nibble) and 2b+1
 * (high nibble), so deinterleave x with svld2 into even/odd lanes. */
static inline float laguna_i4g32_dot_sve(const uint32_t *packed, const uint16_t *scales,
                                         const float *x, int cols) {
    const uint8_t *p = (const uint8_t *)packed;
    int ngrp = cols / LAGUNA_GROUP;
    svbool_t pg = svptrue_b32();
    float sum = 0.0f;
    for (int g = 0; g < ngrp; ++g) {
        svuint32_t bytes = svld1ub_u32(pg, p + (size_t)g * 16);
        svfloat32_t lo = svsub_n_f32_x(pg, svcvt_f32_u32_x(pg, svand_n_u32_x(pg, bytes, 0xfu)), 8.0f);
        svfloat32_t hi = svsub_n_f32_x(pg, svcvt_f32_u32_x(pg, svlsr_n_u32_x(pg, bytes, 4)), 8.0f);
        svfloat32x2_t xv = svld2_f32(pg, x + (size_t)g * LAGUNA_GROUP);
        svfloat32_t part = svmla_f32_x(pg, svmul_f32_x(pg, lo, svget2_f32(xv, 0)), hi, svget2_f32(xv, 1));
        sum += svaddv_f32(pg, part) * laguna_bf16_to_f32(scales[g]);
    }
    return sum;
}
#endif

/* INT4 group-32 symmetric matvec: y[rows] = sum_c (nibble(packed)-8)*scale * x[c].
 * packed is row-major uint32 (cols/8 per row); scales bf16 (cols/32 per row). */
static inline void laguna_matvec_i4g32(float *restrict y, const uint32_t *restrict packed,
                                       const uint16_t *restrict scales,
                                       const float *restrict x, int rows, int cols) {
    int ppr = cols / 8, spr = cols / LAGUNA_GROUP;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; ++r)
#if defined(__ARM_FEATURE_SVE)
        y[r] = (svcntw() == 16)
             ? laguna_i4g32_dot_sve(packed + (size_t)r * ppr, scales + (size_t)r * spr, x, cols)
             : laguna_i4g32_dot    (packed + (size_t)r * ppr, scales + (size_t)r * spr, x, cols);
#else
        y[r] = laguna_i4g32_dot(packed + (size_t)r * ppr, scales + (size_t)r * spr, x, cols);
#endif
}

/* ---- fp8 e4m3 (OCP e4m3fn) dequant, for the fp8-expert build ---- */
#if defined(LAGUNA_FP8)
enum { LAGUNA_FP8_BLK = 128 };               /* weight block-scale is 128x128 */
extern float laguna_fp8_lut[256];            /* byte -> f32, filled by laguna_fp8_init_lut() */
static inline void laguna_fp8_init_lut(void) {
    for (int b = 0; b < 256; ++b) {
        int s=(b>>7)&1, e=(b>>3)&0xf, m=b&0x7; float v;
        if (e==0)            v = (float)m * 0.001953125f;      /* subnormal: m*2^-9 */
        else if (e==15&&m==7)v = 0.0f;                          /* e4m3fn NaN -> 0  */
        else                 v = ldexpf(1.0f + (float)m*0.125f, e-7);
        laguna_fp8_lut[b] = s ? -v : v;
    }
}
/* Block-scaled fp8 matvec: y[r] = sum over 128-col blocks cb of
 * scale[r/128, cb] * sum_{c in cb} lut[W[r,c]] * x[c].  cols multiple of 128. */
#if defined(__ARM_FEATURE_SVE)
/* 8 rows share each x load; fp8 bytes -> f32 via an SVE gather from the LUT (exact,
 * incl. subnormals / e4m3fn NaN). 8 consecutive rows share one scale row because
 * 8 | 128, so an 8-row group never crosses a block-row boundary. */
static inline void laguna_matvec_fp8blk(float *restrict y, const uint8_t *restrict W,
                                        const uint16_t *restrict scales,
                                        const float *restrict x, int rows, int cols) {
    int cblk = cols / LAGUNA_FP8_BLK, VL = (int)svcntw();
    svbool_t pt = svptrue_b32();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r += 8) {
        int nr = rows - r < 8 ? rows - r : 8;
        const uint16_t *sr = scales + (size_t)(r / LAGUNA_FP8_BLK) * cblk;
        if (nr == 8) {
            const uint8_t *w0=W+(size_t)r*cols,*w1=w0+cols,*w2=w1+cols,*w3=w2+cols;
            const uint8_t *w4=w3+cols,*w5=w4+cols,*w6=w5+cols,*w7=w6+cols;
            float t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0,t7=0;
            for (int cb = 0; cb < cblk; ++cb) {
                int c0 = cb*LAGUNA_FP8_BLK;
                svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
                svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
                for (int c = c0; c < c0+LAGUNA_FP8_BLK; c += VL) {
                    svfloat32_t xf = svld1_f32(pt, x + c);
                    a0=svmla_f32_x(pt,a0,svld1_gather_u32index_f32(pt,laguna_fp8_lut,svld1ub_u32(pt,w0+c)),xf);
                    a1=svmla_f32_x(pt,a1,svld1_gather_u32index_f32(pt,laguna_fp8_lut,svld1ub_u32(pt,w1+c)),xf);
                    a2=svmla_f32_x(pt,a2,svld1_gather_u32index_f32(pt,laguna_fp8_lut,svld1ub_u32(pt,w2+c)),xf);
                    a3=svmla_f32_x(pt,a3,svld1_gather_u32index_f32(pt,laguna_fp8_lut,svld1ub_u32(pt,w3+c)),xf);
                    a4=svmla_f32_x(pt,a4,svld1_gather_u32index_f32(pt,laguna_fp8_lut,svld1ub_u32(pt,w4+c)),xf);
                    a5=svmla_f32_x(pt,a5,svld1_gather_u32index_f32(pt,laguna_fp8_lut,svld1ub_u32(pt,w5+c)),xf);
                    a6=svmla_f32_x(pt,a6,svld1_gather_u32index_f32(pt,laguna_fp8_lut,svld1ub_u32(pt,w6+c)),xf);
                    a7=svmla_f32_x(pt,a7,svld1_gather_u32index_f32(pt,laguna_fp8_lut,svld1ub_u32(pt,w7+c)),xf);
                }
                float s = laguna_bf16_to_f32(sr[cb]);
                t0+=s*svaddv_f32(pt,a0); t1+=s*svaddv_f32(pt,a1); t2+=s*svaddv_f32(pt,a2); t3+=s*svaddv_f32(pt,a3);
                t4+=s*svaddv_f32(pt,a4); t5+=s*svaddv_f32(pt,a5); t6+=s*svaddv_f32(pt,a6); t7+=s*svaddv_f32(pt,a7);
            }
            y[r]=t0;y[r+1]=t1;y[r+2]=t2;y[r+3]=t3;y[r+4]=t4;y[r+5]=t5;y[r+6]=t6;y[r+7]=t7;
        } else {
            for (int rr = r; rr < r+nr; ++rr) {
                const uint8_t *wr = W + (size_t)rr * cols; float acc = 0.0f;
                for (int cb = 0; cb < cblk; ++cb) {
                    int c0=cb*LAGUNA_FP8_BLK; svfloat32_t a=svdup_f32(0);
                    for (int c=c0;c<c0+LAGUNA_FP8_BLK;c+=VL)
                        a=svmla_f32_x(pt,a,svld1_gather_u32index_f32(pt,laguna_fp8_lut,svld1ub_u32(pt,wr+c)),svld1_f32(pt,x+c));
                    acc += laguna_bf16_to_f32(sr[cb]) * svaddv_f32(pt,a);
                }
                y[rr]=acc;
            }
        }
    }
}
#else
static inline void laguna_matvec_fp8blk(float *restrict y, const uint8_t *restrict W,
                                        const uint16_t *restrict scales,
                                        const float *restrict x, int rows, int cols) {
    int cblk = cols / LAGUNA_FP8_BLK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        const uint8_t *wr = W + (size_t)r * cols;
        const uint16_t *sr = scales + (size_t)(r / LAGUNA_FP8_BLK) * cblk;
        float acc = 0.0f;
        for (int cb = 0; cb < cblk; ++cb) {
            const uint8_t *wb = wr + cb*LAGUNA_FP8_BLK; const float *xb = x + cb*LAGUNA_FP8_BLK;
            float part = 0.0f;
            for (int j = 0; j < LAGUNA_FP8_BLK; ++j) part += laguna_fp8_lut[wb[j]] * xb[j];
            acc += laguna_bf16_to_f32(sr[cb]) * part;
        }
        y[r] = acc;
    }
}
#endif
#endif

#if defined(LAGUNA_FP8) && defined(__ARM_FEATURE_SVE)
/* Batched fp8 block matvec: Y[N][rows] = X[N][cols] @ fp8_W^T (token-major).  Each
 * weight row is dequanted ONCE (gather LUT + block scale) into an f32 scratch and
 * reused across the N tokens -> the gather (the fp8 bottleneck) is amortized ~N x,
 * and the N dots are plain f32 SVE.  cols multiple of 128, <= LAGUNA_HIDDEN. */
static inline void laguna_matmat_fp8blk(float *restrict Y, const uint8_t *restrict W,
                                        const uint16_t *restrict scales, const float *restrict X,
                                        int rows, int cols, int N) {
    int cblk = cols / LAGUNA_FP8_BLK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r=0;r<rows;++r) {
        const uint8_t *wr=W+(size_t)r*cols; const uint16_t *sr=scales+(size_t)(r/LAGUNA_FP8_BLK)*cblk;
        float wrow[LAGUNA_HIDDEN];
        for (int cb=0;cb<cblk;++cb){ float s=laguna_bf16_to_f32(sr[cb]); const uint8_t *wb=wr+cb*LAGUNA_FP8_BLK; float *o=wrow+cb*LAGUNA_FP8_BLK;
            for (int j=0;j<LAGUNA_FP8_BLK;++j) o[j]=laguna_fp8_lut[wb[j]]*s; }
        int ct=0;
        for (; ct+4<=N; ct+=4) {
            const float *x0=X+(size_t)(ct+0)*cols,*x1=X+(size_t)(ct+1)*cols,*x2=X+(size_t)(ct+2)*cols,*x3=X+(size_t)(ct+3)*cols;
            svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
            for (int c=0;c<cols;c+=(int)svcntw()){ svbool_t pg=svwhilelt_b32(c,cols); svfloat32_t wv=svld1_f32(pg,wrow+c);
                a0=svmla_f32_x(pg,a0,wv,svld1_f32(pg,x0+c)); a1=svmla_f32_x(pg,a1,wv,svld1_f32(pg,x1+c));
                a2=svmla_f32_x(pg,a2,wv,svld1_f32(pg,x2+c)); a3=svmla_f32_x(pg,a3,wv,svld1_f32(pg,x3+c)); }
            svbool_t pt=svptrue_b32();
            Y[(size_t)(ct+0)*rows+r]=svaddv_f32(pt,a0); Y[(size_t)(ct+1)*rows+r]=svaddv_f32(pt,a1);
            Y[(size_t)(ct+2)*rows+r]=svaddv_f32(pt,a2); Y[(size_t)(ct+3)*rows+r]=svaddv_f32(pt,a3);
        }
        for (; ct<N; ++ct){ const float *x=X+(size_t)ct*cols; svfloat32_t a=svdup_f32(0);
            for (int c=0;c<cols;c+=(int)svcntw()){ svbool_t pg=svwhilelt_b32(c,cols); a=svmla_f32_x(pg,a,svld1_f32(pg,wrow+c),svld1_f32(pg,x+c)); }
            Y[(size_t)ct*rows+r]=svaddv_f32(svptrue_b32(),a); }
    }
}
#endif

/* ---- fp8 experts re-quantized to int8 with a per-128-block scale ----
 * The e4m3 LUT gather is the fp8 decode bottleneck: it runs at ~1.2 MAC/cycle/core
 * (8 gathers per 128 MACs) vs ~2.9 for an int8 kernel, and e4m3 costs 2x the bytes
 * of int8.  Converting each 128-col block to int8 (scale = block_scale*max|w|/127)
 * lets decode reuse the plain ld1sb+cvt+fmla path.
 *
 * Accuracy: measured on Gaussian weights block-scaled to e4m3 and back, the dot
 * product error vs the ORIGINAL (pre-fp8) weights is 2.63e-2 for exact e4m3 and
 * 2.74e-2 after the int8 re-quantization -- i.e. the conversion adds ~4% on top of
 * what fp8 quantization already lost, because int8-per-128-block resolves a
 * near-Gaussian block more finely than e4m3's 3 mantissa bits.  See fp8_dq_bench.c.
 * Set LAGUNA_FP8_EXACT=1 to keep the exact e4m3 kernels instead (A/B). */
#if defined(LAGUNA_FP8)
typedef struct { int8_t *q; float *s; } laguna_w8b;  /* q[rows*cols], s[rows*cblk] */

static inline size_t laguna_w8b_bytes(int rows, int cols) {
    return (size_t)rows*cols + (size_t)rows*(cols/LAGUNA_FP8_BLK)*sizeof(float);
}
/* Convert an fp8 e4m3 block-scaled weight to int8 + per-(row,colblock) f32 scale.
 * Parallel over rows so q/s first-touch on the CMG that will read them. */
static inline void laguna_fp8_to_i8blk(laguna_w8b *o, const uint8_t *W,
                                       const uint16_t *bs, int rows, int cols) {
    int cblk = cols / LAGUNA_FP8_BLK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        const uint8_t *wr = W + (size_t)r*cols;
        const uint16_t *sr = bs + (size_t)(r/LAGUNA_FP8_BLK)*cblk;
        for (int cb = 0; cb < cblk; ++cb) {
            const uint8_t *wb = wr + cb*LAGUNA_FP8_BLK;
            float mx = 0.0f;
            for (int j = 0; j < LAGUNA_FP8_BLK; ++j) {
                float a = fabsf(laguna_fp8_lut[wb[j]]); if (a > mx) mx = a;
            }
            float blk = laguna_bf16_to_f32(sr[cb]);
            o->s[(size_t)r*cblk + cb] = mx > 0.0f ? blk*mx/127.0f : blk;
            float inv = mx > 0.0f ? 127.0f/mx : 0.0f;
            int8_t *qb = o->q + (size_t)r*cols + cb*LAGUNA_FP8_BLK;
            for (int j = 0; j < LAGUNA_FP8_BLK; ++j) {
                int v = (int)lrintf(laguna_fp8_lut[wb[j]]*inv);
                qb[j] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
            }
        }
    }
}

#if defined(__ARM_FEATURE_SVE)
/* Rows [r,r+nr) (nr<=8) of y = W8B * x.  8 rows share each x load; 8 | 128 so an
 * 8-row group never crosses a block-scale row boundary.
 *
 * The per-block scale is folded LANE-WISE:  sum_cb s_cb*addv(a_cb) is the same as
 * addv(sum_cb s_cb*a_cb), so each block contributes one vector*scalar FMLA into a
 * running vector accumulator and the cross-lane reduction happens ONCE per row.
 * Doing an svaddv per 128-col block instead (24 of them per 3072-col row) puts a
 * long-latency cross-lane reduction in the dependency chain and costs 1.67x. */
static inline void laguna_i8blk_rowblock(float *restrict y, const laguna_w8b *w,
                                         const float *restrict x, int r, int nr, int cols) {
    int cblk = cols/LAGUNA_FP8_BLK, VL = (int)svcntw();
    svbool_t pt = svptrue_b32();
    if (nr == 8) {
        const int8_t *w0=w->q+(size_t)r*cols,*w1=w0+cols,*w2=w1+cols,*w3=w2+cols;
        const int8_t *w4=w3+cols,*w5=w4+cols,*w6=w5+cols,*w7=w6+cols;
        const float *sr = w->s + (size_t)r*cblk;
        svfloat32_t T0=svdup_f32(0),T1=svdup_f32(0),T2=svdup_f32(0),T3=svdup_f32(0);
        svfloat32_t T4=svdup_f32(0),T5=svdup_f32(0),T6=svdup_f32(0),T7=svdup_f32(0);
        for (int cb = 0; cb < cblk; ++cb) {
            int c0 = cb*LAGUNA_FP8_BLK;
            svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
            svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
            for (int c = c0; c < c0+LAGUNA_FP8_BLK; c += VL) {
                svfloat32_t xf = svld1_f32(pt, x + c);
                a0=svmla_f32_x(pt,a0,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w0+c)),xf);
                a1=svmla_f32_x(pt,a1,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w1+c)),xf);
                a2=svmla_f32_x(pt,a2,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w2+c)),xf);
                a3=svmla_f32_x(pt,a3,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w3+c)),xf);
                a4=svmla_f32_x(pt,a4,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w4+c)),xf);
                a5=svmla_f32_x(pt,a5,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w5+c)),xf);
                a6=svmla_f32_x(pt,a6,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w6+c)),xf);
                a7=svmla_f32_x(pt,a7,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w7+c)),xf);
            }
            T0=svmla_n_f32_x(pt,T0,a0,sr[cb]);         T1=svmla_n_f32_x(pt,T1,a1,sr[cblk+cb]);
            T2=svmla_n_f32_x(pt,T2,a2,sr[2*cblk+cb]);  T3=svmla_n_f32_x(pt,T3,a3,sr[3*cblk+cb]);
            T4=svmla_n_f32_x(pt,T4,a4,sr[4*cblk+cb]);  T5=svmla_n_f32_x(pt,T5,a5,sr[5*cblk+cb]);
            T6=svmla_n_f32_x(pt,T6,a6,sr[6*cblk+cb]);  T7=svmla_n_f32_x(pt,T7,a7,sr[7*cblk+cb]);
        }
        y[r]=svaddv_f32(pt,T0);   y[r+1]=svaddv_f32(pt,T1);
        y[r+2]=svaddv_f32(pt,T2); y[r+3]=svaddv_f32(pt,T3);
        y[r+4]=svaddv_f32(pt,T4); y[r+5]=svaddv_f32(pt,T5);
        y[r+6]=svaddv_f32(pt,T6); y[r+7]=svaddv_f32(pt,T7);
    } else {
        for (int rr = r; rr < r+nr; ++rr) {
            const int8_t *wq = w->q + (size_t)rr*cols;
            const float *sr = w->s + (size_t)rr*cblk;
            svfloat32_t T = svdup_f32(0);
            for (int cb = 0; cb < cblk; ++cb) {
                int c0 = cb*LAGUNA_FP8_BLK; svfloat32_t a = svdup_f32(0);
                for (int c = c0; c < c0+LAGUNA_FP8_BLK; c += VL)
                    a=svmla_f32_x(pt,a,svcvt_f32_s32_x(pt,svld1sb_s32(pt,wq+c)),svld1_f32(pt,x+c));
                T = svmla_n_f32_x(pt,T,a,sr[cb]);
            }
            y[rr]=svaddv_f32(pt,T);
        }
    }
}
static inline void laguna_matvec_i8blk(float *restrict y, const laguna_w8b *w,
                                       const float *restrict x, int rows, int cols) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r += 8)
        laguna_i8blk_rowblock(y, w, x, r, rows-r<8?rows-r:8, cols);
}
/* Fused multi-matvec: several y[i] = W8B[i]*x sharing x, rows and cols, in ONE
 * parallel region (nowait between matrices).  An expert's gate and up projections
 * are exactly this, so it halves the expert's OpenMP fork/joins. */
static inline void laguna_matvec_i8blk_multi(float *const *ys, const laguna_w8b *const *ws,
                                             int nmat, const float *x, int rows, int cols) {
#ifdef _OPENMP
    #pragma omp parallel
    {
        for (int m = 0; m < nmat; ++m) {
            #pragma omp for schedule(static) nowait
            for (int r = 0; r < rows; r += 8)
                laguna_i8blk_rowblock(ys[m], ws[m], x, r, rows-r<8?rows-r:8, cols);
        }
    }
#else
    for (int m = 0; m < nmat; ++m)
        for (int r = 0; r < rows; r += 8)
            laguna_i8blk_rowblock(ys[m], ws[m], x, r, rows-r<8?rows-r:8, cols);
#endif
}
/* Batched (chunked prefill): Y[N][rows] = X[N][cols] @ W8B^T, token-major.
 * Each weight row is widened and scaled ONCE into an f32 scratch, then reused for
 * all N token dots (which are then plain f32 SVE).  Widening per 8-token tile
 * instead loses badly as N grows -- at N=128 it re-converts every row 16x and runs
 * ~30% slower than this form.  cols <= LAGUNA_HIDDEN (gate/up 3072, down 1024). */
static inline void laguna_matmat_i8blk(float *restrict Y, const laguna_w8b *w,
                                       const float *restrict X, int rows, int cols, int N) {
    int cblk = cols/LAGUNA_FP8_BLK, VL = (int)svcntw();
    svbool_t pt = svptrue_b32();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        const int8_t *wq = w->q + (size_t)r*cols;
        const float  *sr = w->s + (size_t)r*cblk;
        float wrow[LAGUNA_HIDDEN];
        for (int cb = 0; cb < cblk; ++cb) {
            svfloat32_t s = svdup_f32(sr[cb]);
            int c0 = cb*LAGUNA_FP8_BLK;
            for (int c = c0; c < c0+LAGUNA_FP8_BLK; c += VL)
                svst1_f32(pt, wrow+c, svmul_f32_x(pt, svcvt_f32_s32_x(pt, svld1sb_s32(pt,wq+c)), s));
        }
        int ct = 0;
        for (; ct+8 <= N; ct += 8) {
            const float *x0=X+(size_t)(ct+0)*cols,*x1=X+(size_t)(ct+1)*cols,
                        *x2=X+(size_t)(ct+2)*cols,*x3=X+(size_t)(ct+3)*cols,
                        *x4=X+(size_t)(ct+4)*cols,*x5=X+(size_t)(ct+5)*cols,
                        *x6=X+(size_t)(ct+6)*cols,*x7=X+(size_t)(ct+7)*cols;
            svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
            svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
            for (int c = 0; c < cols; c += VL) {
                svfloat32_t wv = svld1_f32(pt, wrow+c);
                a0=svmla_f32_x(pt,a0,wv,svld1_f32(pt,x0+c)); a1=svmla_f32_x(pt,a1,wv,svld1_f32(pt,x1+c));
                a2=svmla_f32_x(pt,a2,wv,svld1_f32(pt,x2+c)); a3=svmla_f32_x(pt,a3,wv,svld1_f32(pt,x3+c));
                a4=svmla_f32_x(pt,a4,wv,svld1_f32(pt,x4+c)); a5=svmla_f32_x(pt,a5,wv,svld1_f32(pt,x5+c));
                a6=svmla_f32_x(pt,a6,wv,svld1_f32(pt,x6+c)); a7=svmla_f32_x(pt,a7,wv,svld1_f32(pt,x7+c));
            }
            Y[(size_t)(ct+0)*rows+r]=svaddv_f32(pt,a0); Y[(size_t)(ct+1)*rows+r]=svaddv_f32(pt,a1);
            Y[(size_t)(ct+2)*rows+r]=svaddv_f32(pt,a2); Y[(size_t)(ct+3)*rows+r]=svaddv_f32(pt,a3);
            Y[(size_t)(ct+4)*rows+r]=svaddv_f32(pt,a4); Y[(size_t)(ct+5)*rows+r]=svaddv_f32(pt,a5);
            Y[(size_t)(ct+6)*rows+r]=svaddv_f32(pt,a6); Y[(size_t)(ct+7)*rows+r]=svaddv_f32(pt,a7);
        }
        for (; ct < N; ++ct) {
            const float *x = X+(size_t)ct*cols; svfloat32_t a = svdup_f32(0);
            for (int c = 0; c < cols; c += VL)
                a=svmla_f32_x(pt,a,svld1_f32(pt,wrow+c),svld1_f32(pt,x+c));
            Y[(size_t)ct*rows+r]=svaddv_f32(pt,a);
        }
    }
}
#else  /* scalar fallback */
static inline void laguna_matvec_i8blk(float *restrict y, const laguna_w8b *w,
                                       const float *restrict x, int rows, int cols) {
    int cblk = cols/LAGUNA_FP8_BLK;
    for (int r = 0; r < rows; ++r) {
        const int8_t *wq = w->q + (size_t)r*cols; const float *sr = w->s + (size_t)r*cblk;
        float acc = 0.0f;
        for (int cb = 0; cb < cblk; ++cb) { float p = 0.0f;
            for (int j = 0; j < LAGUNA_FP8_BLK; ++j)
                p += (float)wq[cb*LAGUNA_FP8_BLK+j]*x[cb*LAGUNA_FP8_BLK+j];
            acc += sr[cb]*p; }
        y[r]=acc;
    }
}
static inline void laguna_matvec_i8blk_multi(float *const *ys, const laguna_w8b *const *ws,
                                             int nmat, const float *x, int rows, int cols) {
    for (int m = 0; m < nmat; ++m) laguna_matvec_i8blk(ys[m], ws[m], x, rows, cols);
}
static inline void laguna_matmat_i8blk(float *restrict Y, const laguna_w8b *w,
                                       const float *restrict X, int rows, int cols, int N) {
    for (int c = 0; c < N; ++c)
        laguna_matvec_i8blk(Y+(size_t)c*rows, w, X+(size_t)c*cols, rows, cols);
}
#endif
#endif /* LAGUNA_FP8 */

/* ---- linear-weight abstraction: int8 W8 (production + fp8 build) or bf16.
 * Build variants with -DLAGUNA_BF16 or -DLAGUNA_FP8.  Only the pure-bf16
 * reference build keeps bf16 linears; call sites use laguna_lin_mv* and stay
 * identical.
 *
 * The fp8 build quantizes its non-expert linears (q/k/v/o/g, dense MLP, shared
 * expert, router, lm_head) to int8 at load, exactly as the int4 build does.
 * Those weights -- not the experts -- dominate decode: they are ~7.2 GB of bf16
 * read per token per rank (replicated on every rank) against ~0.4 GB of routed
 * expert weight, so halving them is the single largest decode win available. ---- */
#if defined(LAGUNA_BF16)
typedef const uint16_t *laguna_lin;          /* a plain bf16 weight pointer */
static inline void laguna_lin_mv(float *restrict y, const laguna_lin *w,
                                 const float *restrict x, int rows, int cols) {
    laguna_matvec_bf16(y, *w, x, rows, cols);
}
static inline void laguna_lin_mv_multi(float *const *ys, const laguna_lin *const *ws,
                                       const int *rows, int nmat, const float *x, int cols) {
    const uint16_t *Ws[8]; for (int m=0;m<nmat;++m) Ws[m]=*ws[m];
    laguna_matvec_bf16_multi(ys, Ws, rows, nmat, x, cols);
}
/* batched (chunked prefill): Y[C][rows] = X[C][cols] @ w^T */
static inline void laguna_lin_mm(float *Y, const laguna_lin *w, const float *X,
                                 int rows, int cols, int C) {
    laguna_matmat_bf16(Y, *w, X, rows, cols, C);
}
#else
typedef laguna_w8 laguna_lin;                /* int8 per-row weight (q + scale) */
static inline void laguna_lin_mv(float *restrict y, const laguna_lin *w,
                                 const float *restrict x, int rows, int cols) {
    laguna_matvec_i8(y, w, x, rows, cols);
}
static inline void laguna_lin_mv_multi(float *const *ys, const laguna_lin *const *ws,
                                       const int *rows, int nmat, const float *x, int cols) {
    laguna_matvec_i8_multi(ys, ws, rows, nmat, x, cols);
}
/* batched (chunked prefill): Y[C][rows] = X[C][cols] @ w^T */
static inline void laguna_lin_mm(float *Y, const laguna_lin *w, const float *X,
                                 int rows, int cols, int C) {
    laguna_matmat_i8(Y, w, X, rows, cols, C);
}
#endif

#if defined(LAGUNA_FP8)
typedef struct { const uint8_t *gate, *up, *down;   /* fp8 e4m3 [rows,cols]     */
                 const uint16_t *gs, *us, *ds;      /* bf16 block scales        */
                 laguna_w8b qg, qu, qd;             /* int8-per-block (default) */
                 int present; } laguna_expert;
#elif defined(LAGUNA_BF16)
typedef struct { const uint16_t *gate, *up, *down; int present; } laguna_expert;
#else
typedef struct {
    const uint32_t *gp, *up, *dp;   /* weight_packed (uint32) */
    const uint16_t *gs, *us, *ds;   /* weight_scale  (bf16)   */
    int present;                    /* owned by this rank      */
} laguna_expert;
#endif

/* ===================== per-rank model ===================== */
typedef struct {
    int num_heads;         /* per-layer query heads (48 full / 72 sliding) */
    int is_sliding;        /* 1 => sliding_attention, 0 => full_attention  */
    int is_moe;            /* 1 => MoE mlp, 0 => dense mlp (layer 0)        */
    laguna_lin q_proj, k_proj, v_proj, o_proj, g_proj;
    const uint16_t *q_norm, *k_norm, *in_ln, *post_ln;   /* norms stay bf16 */
    laguna_lin dense_gate, dense_up, dense_down;         /* dense mlp (layer 0) */
    laguna_lin shared_gate, shared_up, shared_down;      /* moe shared expert */
    laguna_lin router_w;                                 /* router [256,hidden] */
    const float    *router_bias;                         /* [256] (converted to f32) */
    laguna_expert   experts[LAGUNA_EXPERTS];
} laguna_layer;

typedef struct {
    int n_layers, max_pos, ep_rank, ep_size;
    const uint16_t *embed;      /* bf16 [vocab,hidden] (embedding lookup) */
    laguna_lin lm_head;         /* [vocab,hidden] */
    const uint16_t *final_norm; /* bf16 [hidden]       */
    laguna_layer layers[LAGUNA_LAYERS];
    /* rope tables: cos/sin[pos * half] for full (half=32) and sliding (half=64) */
    float *full_cos, *full_sin;       /* [max_pos * 32] */
    float *swa_cos, *swa_sin;         /* [max_pos * 64] */
    /* KV cache (bf16), replicated per rank.  Full-attention layers keep the whole
     * context; sliding layers use a SLIDING_WINDOW ring buffer (position p -> slot
     * p%cap), so long-context KV stays small (128k: ~6.5 GB not 25.8). */
    uint16_t *kcache, *vcache;
    size_t kv_off[LAGUNA_LAYERS];     /* element offset of each layer's KV block */
    int    kv_cap[LAGUNA_LAYERS];     /* capacity: max_pos (full) or window (sliding) */
    /* Batched serving keeps n_seq independent caches back to back, so sequence s
     * starts at s*kv_seq_stride.  n_seq==1 for --generate, and every existing
     * single-sequence call site passes seq=0 and is unaffected. */
    size_t kv_seq_stride;
    int    n_seq;
} laguna_model;

static inline void laguna_build_rope_tables(laguna_model *m) {
    float inv_full[LAGUNA_ROPE_FULL_DIM / 2];
    float inv_swa[LAGUNA_ROPE_SLIDING_DIM / 2];
    laguna_yarn_inv_freq(inv_full, LAGUNA_ROPE_FULL_DIM);
    laguna_default_inv_freq(inv_swa, LAGUNA_ROPE_SLIDING_DIM, LAGUNA_SWA_THETA);
    int hf = LAGUNA_ROPE_FULL_DIM / 2, hs = LAGUNA_ROPE_SLIDING_DIM / 2;
    for (int p = 0; p < m->max_pos; ++p) {
        for (int i = 0; i < hf; ++i) {
            double ang = (double)p * inv_full[i];
            m->full_cos[(size_t)p * hf + i] = (float)(cos(ang) * LAGUNA_YARN_ATTN_FACTOR);
            m->full_sin[(size_t)p * hf + i] = (float)(sin(ang) * LAGUNA_YARN_ATTN_FACTOR);
        }
        for (int i = 0; i < hs; ++i) {
            double ang = (double)p * inv_swa[i];
            m->swa_cos[(size_t)p * hs + i] = (float)cos(ang);
            m->swa_sin[(size_t)p * hs + i] = (float)sin(ang);
        }
    }
}
#endif
