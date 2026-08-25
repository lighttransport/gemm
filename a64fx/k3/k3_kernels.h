#ifndef K3_KERNELS_H
#define K3_KERNELS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

enum {
    K3_HIDDEN = 7168,
    K3_HEADS = 96,
    K3_HEAD_DIM = 128,
    K3_KDA_LAYERS = 69,
    K3_MLA_LAYERS = 24,
    K3_LAYERS = 93,
    K3_MOE_LAYERS = 92,
    K3_EXPERTS = 896,
    K3_TOP_K = 16,
    K3_LATENT = 3584,
    K3_EXPERT_INTER = 3072
};

static inline float k3_sigmoidf(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static inline float k3_dot_ref(const float *a, const float *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)a[i] * b[i];
    return (float)s;
}

static inline float k3_dot_sve(const float *a, const float *b, int n) {
#if defined(__ARM_FEATURE_SVE)
    int vl = (int)svcntw(), i = 0;
    svbool_t all = svptrue_b32();
    svfloat32_t s0 = svdup_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    for (; i + 4 * vl <= n; i += 4 * vl) {
        s0 = svmla_f32_x(all, s0, svld1(all, a + i), svld1(all, b + i));
        s1 = svmla_f32_x(all, s1, svld1(all, a + i + vl), svld1(all, b + i + vl));
        s2 = svmla_f32_x(all, s2, svld1(all, a + i + 2 * vl), svld1(all, b + i + 2 * vl));
        s3 = svmla_f32_x(all, s3, svld1(all, a + i + 3 * vl), svld1(all, b + i + 3 * vl));
    }
    for (; i < n; i += vl) {
        svbool_t pg = svwhilelt_b32(i, n);
        s0 = svmla_f32_m(pg, s0, svld1(pg, a + i), svld1(pg, b + i));
    }
    s0 = svadd_f32_x(all, svadd_f32_x(all, s0, s1), svadd_f32_x(all, s2, s3));
    return svaddv_f32(all, s0);
#else
    return k3_dot_ref(a, b, n);
#endif
}

/* QK tile with query lanes.  qt is packed as [qk_dim][16]; A64FX has 16
 * FP32 lanes, so one streamed K element updates a full 16-query tile. */
static inline void k3_dot8_tile_sve(float *scores, const float qt[][16],
                                    const float *k, int qn,
                                    int qk_dim) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svwhilelt_b32(0, qn);
    svfloat32_t sum = svdup_f32(0.0f);
    for (int d = 0; d < qk_dim; ++d)
        sum = svmla_n_f32_x(pg, sum, svld1(pg, qt[d]), k[d]);
    float tmp[16];
    svst1(pg, tmp, sum);
    for (int qi = 0; qi < qn; ++qi) scores[qi] = tmp[qi];
#else
    for (int qi = 0; qi < qn; ++qi) {
        scores[qi] = 0.0f;
        for (int d = 0; d < qk_dim; ++d) scores[qi] += qt[d][qi] * k[d];
    }
#endif
}

static inline void k3_l2_normalize_ref(float *x, int n, float eps) {
    float inv = 1.0f / sqrtf(k3_dot_ref(x, x, n) + eps);
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

static inline void k3_l2_normalize_sve(float *x, int n, float eps) {
    float inv = 1.0f / sqrtf(k3_dot_sve(x, x, n) + eps);
#if defined(__ARM_FEATURE_SVE)
    int vl = (int)svcntw();
    for (int i = 0; i < n; i += vl) {
        svbool_t pg = svwhilelt_b32(i, n);
        svst1(pg, x + i, svmul_n_f32_x(pg, svld1(pg, x + i), inv));
    }
#else
    for (int i = 0; i < n; ++i) x[i] *= inv;
#endif
}

static inline void k3_rmsnorm_ref(float *out, const float *x, const float *weight,
                                  int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    float inv = (float)(1.0 / sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * weight[i];
}

static inline void k3_rmsnorm_sve(float *out, const float *x, const float *weight,
                                  int n, float eps) {
    float inv = 1.0f / sqrtf(k3_dot_sve(x, x, n) / n + eps);
#if defined(__ARM_FEATURE_SVE)
    int vl = (int)svcntw();
    for (int i = 0; i < n; i += vl) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svmul_n_f32_x(pg, svld1(pg, x + i), inv);
        svst1(pg, out + i, svmul_f32_x(pg, v, svld1(pg, weight + i)));
    }
#else
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * weight[i];
#endif
}

static inline void k3_gated_rmsnorm_ref(float *out, const float *x, const float *gate,
                                        const float *weight, int n, float eps) {
    k3_rmsnorm_ref(out, x, weight, n, eps);
    for (int i = 0; i < n; ++i) out[i] *= k3_sigmoidf(gate[i]);
}

static inline void k3_gated_rmsnorm_sve(float *out, const float *x, const float *gate,
                                        const float *weight, int n, float eps) {
    k3_rmsnorm_sve(out, x, weight, n, eps);
    /* Keep sigmoid exact and scalar until an approximation has an explicit error contract. */
    for (int i = 0; i < n; ++i) out[i] *= k3_sigmoidf(gate[i]);
}

/* Kimi SiTU: beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta). */
static inline void k3_situ_ref(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; ++i)
        out[i] = 4.0f * tanhf(gate[i] * 0.25f) * k3_sigmoidf(gate[i])
               * 25.0f * tanhf(up[i] * 0.04f);
}

/* Transcendentals are scalar; SVE covers the bandwidth/FMA part without an approximation contract. */
static inline void k3_situ_sve(float *out, const float *gate, const float *up, int n) {
    k3_situ_ref(out, gate, up, n);
}

#if defined(__ARM_FEATURE_SVE)
static inline svfloat32_t k3_exp2_fexpa_sve(svbool_t pg, svfloat32_t x) {
    const float shift_f = 204927.0f;
    svfloat32_t shift = svdup_f32(shift_f);
    svfloat32_t z = svadd_f32_x(pg, x, shift);
    svfloat32_t rounded = svsub_f32_x(pg, z, shift);
    svfloat32_t residual = svsub_f32_x(pg, x, rounded);
    svfloat32_t scale = svexpa_f32(svreinterpret_u32_f32(z));
    svfloat32_t correction = svmla_n_f32_x(
        pg, svdup_f32(1.0f), residual, 0.6931471805599453f);
    return svmul_f32_x(pg, scale, correction);
}

static inline svfloat32_t k3_sigmoid_fast_sve(svbool_t pg, svfloat32_t x) {
    svfloat32_t t = svmul_n_f32_x(pg, x, -1.4426950408889634f);
    t = svmax_n_f32_x(pg, svmin_n_f32_x(pg, t, 80.0f), -80.0f);
    svfloat32_t den = svadd_n_f32_x(pg, k3_exp2_fexpa_sve(pg, t), 1.0f);
    svfloat32_t inv = svrecpe_f32(den);
    return svmul_f32_x(pg, inv, svrecps_f32(den, inv));
}

/* SiTU via tanh(x)=2*sigmoid(2*x)-1.  FEXPA plus residual correction has an
 * explicit real-input error gate in k3_kernel_test before this is enabled in
 * the expert path. */
static inline void k3_situ_fast_sve(float *out, const float *gate,
                                    const float *up, int n) {
    int vl = (int)svcntw();
    for (int i = 0; i < n; i += vl) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t g = svld1(pg, gate + i);
        svfloat32_t u = svld1(pg, up + i);
        svfloat32_t tg = svsub_n_f32_x(pg,
            svmul_n_f32_x(pg, k3_sigmoid_fast_sve(
                pg, svmul_n_f32_x(pg, g, 0.5f)), 2.0f), 1.0f);
        svfloat32_t su = svsub_n_f32_x(pg,
            svmul_n_f32_x(pg, k3_sigmoid_fast_sve(
                pg, svmul_n_f32_x(pg, u, 0.08f)), 2.0f), 1.0f);
        svfloat32_t y = svmul_f32_x(pg, tg, k3_sigmoid_fast_sve(pg, g));
        y = svmul_f32_x(pg, y, su);
        svst1(pg, out + i, svmul_n_f32_x(pg, y, 100.0f));
    }
}
#else
static inline void k3_situ_fast_sve(float *out, const float *gate,
                                    const float *up, int n) {
    k3_situ_ref(out, gate, up, n);
}
#endif

/* State is [channels][kernel-1], oldest to newest; weight is [channels][kernel]. */
static inline void k3_conv_step_ref(float *out, const float *x, float *state,
                                    const float *weight, const float *bias,
                                    int channels, int kernel) {
    for (int c = 0; c < channels; ++c) {
        float *s = state + (size_t)c * (kernel - 1);
        const float *w = weight + (size_t)c * kernel;
        float y = bias ? bias[c] : 0.0f;
        for (int j = 0; j < kernel - 1; ++j) y += w[j] * s[j];
        y += w[kernel - 1] * x[c];
        out[c] = y;
        for (int j = 0; j + 1 < kernel - 1; ++j) s[j] = s[j + 1];
        if (kernel > 1) s[kernel - 2] = x[c];
    }
}

static inline void k3_conv_step_sve(float *out, const float *x, float *state,
                                    const float *weight, const float *bias,
                                    int channels, int kernel) {
    k3_conv_step_ref(out, x, state, weight, bias, channels, kernel);
}

/* GGUF stores A=-exp(A_log), one scalar per head.  K3 uses the bounded
 * safe-gate form with lower_bound=-5. */
static inline void k3_kda_log_decay(float *out, const float *g_raw, const float *a,
                                    const float *dt_bias, int heads, int key_dim) {
    for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < key_dim; ++d) {
            int i = h * key_dim + d;
            float z = (-a[h]) * (g_raw[i] + dt_bias[i]);
            out[i] = -5.0f * k3_sigmoidf(z);
        }
    }
}

/* Official recurrent delta rule. State layout is [head][value][key]. */
static inline void k3_kda_step_ref(float *out, const float *q, const float *k,
                                   const float *v, const float *log_decay,
                                   const float *beta, float *state,
                                   int heads, int key_dim, int value_dim) {
    float scale = 1.0f / sqrtf((float)key_dim);
    for (int h = 0; h < heads; ++h) {
        const float *qh = q + (size_t)h * key_dim;
        const float *kh = k + (size_t)h * key_dim;
        const float *vh = v + (size_t)h * value_dim;
        float *sh = state + (size_t)h * value_dim * key_dim;
        for (int j = 0; j < value_dim; ++j) {
            float *row = sh + (size_t)j * key_dim;
            double pred = 0.0;
            for (int d = 0; d < key_dim; ++d) {
                row[d] *= expf(log_decay[h * key_dim + d]);
                pred += (double)kh[d] * row[d];
            }
            float delta = beta[h] * (vh[j] - (float)pred);
            for (int d = 0; d < key_dim; ++d) row[d] += kh[d] * delta;
            out[h * value_dim + j] = k3_dot_ref(qh, row, key_dim) * scale;
        }
    }
}

static inline void k3_kda_step_decay_sve(float *out, const float *q, const float *k,
                                   const float *v, const float *decay,
                                   const float *beta, float *state,
                                   int heads, int key_dim, int value_dim) {
    float scale = 1.0f / sqrtf((float)key_dim);
    for (int h = 0; h < heads; ++h) {
        const float *qh = q + (size_t)h * key_dim;
        const float *kh = k + (size_t)h * key_dim;
        const float *dh = decay + (size_t)h * key_dim;
        float *sh = state + (size_t)h * value_dim * key_dim;
        for (int j = 0; j < value_dim; ++j) {
            float *row = sh + (size_t)j * key_dim;
#if defined(__ARM_FEATURE_SVE)
            int vl = (int)svcntw();
            for (int d = 0; d < key_dim; d += vl) {
                svbool_t pg = svwhilelt_b32(d, key_dim);
                svfloat32_t r = svmul_f32_x(pg, svld1(pg, row + d), svld1(pg, dh + d));
                svst1(pg, row + d, r);
            }
#else
            for (int d = 0; d < key_dim; ++d) row[d] *= dh[d];
#endif
            /* Match k3_kda_step_ref: the recurrent prediction is accumulated
             * in FP64 before narrowing to float.  The SVE FP32 reduction is
             * fast but its small error compounds over 69 KDA layers and can
             * change greedy logits after a few tokens. */
            float delta = beta[h] * (v[h * value_dim + j] - k3_dot_ref(kh, row, key_dim));
#if defined(__ARM_FEATURE_SVE)
            int vl2 = (int)svcntw();
            for (int d = 0; d < key_dim; d += vl2) {
                svbool_t pg = svwhilelt_b32(d, key_dim);
                svfloat32_t r = svmla_n_f32_x(pg, svld1(pg, row + d), svld1(pg, kh + d), delta);
                svst1(pg, row + d, r);
            }
#else
            for (int d = 0; d < key_dim; ++d) row[d] += kh[d] * delta;
#endif
            out[h * value_dim + j] = k3_dot_ref(qh, row, key_dim) * scale;
        }
    }
}

static inline void k3_kda_step_decay_row_sve(float *out, const float *q,
        const float *k, const float *v, const float *decay,
        const float *beta, float *state, int heads, int key_dim,
        int value_dim, int task) {
    (void)heads;
    float scale=1.0f/sqrtf((float)key_dim);
    int h=task/value_dim,j=task%value_dim;
    const float *qh=q+(size_t)h*key_dim,*kh=k+(size_t)h*key_dim;
    const float *dh=decay+(size_t)h*key_dim;
    float *row=state+((size_t)h*value_dim+j)*key_dim;
#if defined(__ARM_FEATURE_SVE)
    int vl=(int)svcntw();
    for(int d=0;d<key_dim;d+=vl){svbool_t pg=svwhilelt_b32(d,key_dim);
        svst1(pg,row+d,svmul_f32_x(pg,svld1(pg,row+d),svld1(pg,dh+d)));}
#else
    for(int d=0;d<key_dim;++d)row[d]*=dh[d];
#endif
    float delta=beta[h]*(v[(size_t)h*value_dim+j]-k3_dot_ref(kh,row,key_dim));
#if defined(__ARM_FEATURE_SVE)
    int vl2=(int)svcntw();
    for(int d=0;d<key_dim;d+=vl2){svbool_t pg=svwhilelt_b32(d,key_dim);
        svst1(pg,row+d,svmla_n_f32_x(pg,svld1(pg,row+d),svld1(pg,kh+d),delta));}
#else
    for(int d=0;d<key_dim;++d)row[d]+=kh[d]*delta;
#endif
    out[(size_t)h*value_dim+j]=k3_dot_ref(qh,row,key_dim)*scale;
}

/* Orphaned workshare: every thread in an existing OpenMP team must call it. */
static inline void k3_kda_step_decay_team_sve(float *out, const float *q,
        const float *k, const float *v, const float *decay,
        const float *beta, float *state, int heads, int key_dim,
        int value_dim) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for(int task=0;task<heads*value_dim;++task)
        k3_kda_step_decay_row_sve(out,q,k,v,decay,beta,state,heads,key_dim,value_dim,task);
}

/* Flatten [head][value] into enough independent row tasks to occupy all A64FX
 * cores even when tensor parallelism leaves only one to eight heads per rank. */
static inline void k3_kda_step_decay_parallel_sve(float *out, const float *q,
        const float *k, const float *v, const float *decay,
        const float *beta, float *state, int heads, int key_dim,
        int value_dim, int threads) {
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel
    k3_kda_step_decay_team_sve(out,q,k,v,decay,beta,state,heads,key_dim,value_dim);
#else
    (void)threads;
    for(int task=0;task<heads*value_dim;++task)
        k3_kda_step_decay_row_sve(out,q,k,v,decay,beta,state,heads,key_dim,value_dim,task);
#endif
}

static inline void k3_kda_step_sve(float *out, const float *q, const float *k,
                                   const float *v, const float *log_decay,
                                   const float *beta, float *state,
                                   int heads, int key_dim, int value_dim) {
    size_t n = (size_t)heads * key_dim;
    float decay[n];
    for (size_t i = 0; i < n; ++i) decay[i] = expf(log_decay[i]);
    k3_kda_step_decay_sve(out,q,k,v,decay,beta,state,heads,key_dim,value_dim);
}

/* Stable online softmax for one query. keys=[tokens][qk_dim], values=[tokens][v_dim]. */
static inline void k3_attention_ref(float *out, const float *q, const float *keys,
                                    const float *values, int tokens, int qk_dim, int v_dim) {
    for (int j = 0; j < v_dim; ++j) out[j] = 0.0f;
    float m = -INFINITY, l = 0.0f, scale = 1.0f / sqrtf((float)qk_dim);
    for (int t = 0; t < tokens; ++t) {
        float score = k3_dot_ref(q, keys + (size_t)t * qk_dim, qk_dim) * scale;
        float nm,old,add;if(score<=m){nm=m;old=1.0f;add=expf(score-m);}
        else{nm=score;old=expf(m-score);add=1.0f;}
        for (int j = 0; j < v_dim; ++j)
            out[j] = out[j] * old + values[(size_t)t * v_dim + j] * add;
        l = l * old + add; m = nm;
    }
    if (l != 0.0f) for (int j = 0; j < v_dim; ++j) out[j] /= l;
}

static inline void k3_attention_sve(float *out, const float *q, const float *keys,
                                    const float *values, int tokens, int qk_dim, int v_dim) {
    for (int j = 0; j < v_dim; ++j) out[j] = 0.0f;
    float m = -INFINITY, l = 0.0f, scale = 1.0f / sqrtf((float)qk_dim);
    for (int t = 0; t < tokens; ++t) {
        float score = k3_dot_sve(q, keys + (size_t)t * qk_dim, qk_dim) * scale;
        float nm,old,add;if(score<=m){nm=m;old=1.0f;add=expf(score-m);}
        else{nm=score;old=expf(m-score);add=1.0f;}
#if defined(__ARM_FEATURE_SVE)
        int vl = (int)svcntw();
        for (int j = 0; j < v_dim; j += vl) {
            svbool_t pg = svwhilelt_b32(j, v_dim);
            svfloat32_t z = svmul_n_f32_x(pg, svld1(pg, out + j), old);
            z = svmla_n_f32_x(pg, z, svld1(pg, values + (size_t)t * v_dim + j), add);
            svst1(pg, out + j, z);
        }
#else
        for (int j = 0; j < v_dim; ++j)
            out[j] = out[j] * old + values[(size_t)t * v_dim + j] * add;
#endif
        l = l * old + add; m = nm;
    }
    if (l != 0.0f) {
#if defined(__ARM_FEATURE_SVE)
        int vl = (int)svcntw();
        for (int j = 0; j < v_dim; j += vl) {
            svbool_t pg = svwhilelt_b32(j, v_dim);
            svst1(pg, out + j, svdiv_n_f32_x(pg, svld1(pg, out + j), l));
        }
#else
        for (int j = 0; j < v_dim; ++j) out[j] /= l;
#endif
    }
}

static inline float k3_dot_f32_i8_sve(const float *a, const int8_t *b, int n) {
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    for (int i = 0; i < n; i += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svint32_t bi = svld1sb_s32(pg, b + i);
        sum = svmla_f32_m(pg, sum, svld1(pg, a + i), svcvt_f32_s32_x(pg, bi));
    }
    return svaddv_f32(svptrue_b32(), sum);
#else
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i] * (float)b[i];
    return sum;
#endif
}

/* Symmetric int8 quantization of one query vector.  The query is reused across
 * every key in the scan, so this pays for itself after the first token. */
static inline float k3_quantize_i8(const float *x, int n, int8_t *out) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
    if (amax <= 0.0f) {
        for (int i = 0; i < n; ++i) out[i] = 0;
        return 0.0f;
    }
    float inv = 127.0f / amax;
    for (int i = 0; i < n; ++i) {
        int v = (int)lrintf(x[i] * inv);
        out[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
    }
    return amax / 127.0f;
}

static inline int32_t k3_dot_i8_i8_sve(const int8_t *a, const int8_t *b, int n) {
#if defined(__ARM_FEATURE_SVE)
    svint32_t acc = svdup_s32(0);
    for (int i = 0; i < n; i += (int)svcntb()) {
        svbool_t pg = svwhilelt_b8(i, n);
        acc = svdot_s32(acc, svld1_s8(pg, a + i), svld1_s8(pg, b + i));
    }
    return svaddv_s32(svptrue_b32(), acc);
#else
    int32_t acc = 0;
    for (int i = 0; i < n; ++i) acc += (int32_t)a[i] * b[i];
    return acc;
#endif
}

/* Symmetric per-token INT8 MLA attention.  Scales are stored separately for
 * each key/value token vector; softmax and accumulation remain FP32.
 *
 * The QK score is a true int8 SDOT: quantizing the query once turns the scan's
 * inner product from one fp32 lane per element into four int8 lanes per lane,
 * where the old kernel widened every stored key byte back to fp32 before a
 * plain FMLA.  PV stays fp32 — it is the accuracy-sensitive half and it is not
 * the term that grows with context. */
static inline void k3_attention_i8_sve(float *out, const float *q,
        const int8_t *keys, const float *key_scales, const int8_t *values,
        const float *value_scales, int tokens, int qk_dim, int v_dim) {
    for (int j = 0; j < v_dim; ++j) out[j] = 0.0f;
    float m = -INFINITY, l = 0.0f, scale = 1.0f / sqrtf((float)qk_dim);
    int8_t q8[qk_dim];
    float qs = k3_quantize_i8(q, qk_dim, q8);
    for (int t = 0; t < tokens; ++t) {
        const int8_t *kt = keys + (size_t)t * qk_dim;
        float score = (float)k3_dot_i8_i8_sve(q8, kt, qk_dim) * qs *
                      key_scales[t] * scale;
        float nm, old, add;
        if (score <= m) { nm = m; old = 1.0f; add = expf(score - m); }
        else { nm = score; old = expf(m - score); add = 1.0f; }
        const int8_t *vt = values + (size_t)t * v_dim;
        float vs = value_scales[t];
#if defined(__ARM_FEATURE_SVE)
        int vl = (int)svcntw();
        for (int j = 0; j < v_dim; j += vl) {
            svbool_t pg = svwhilelt_b32(j, v_dim);
            svint32_t iz = svld1sb_s32(pg, vt + j);
            svfloat32_t z = svmul_n_f32_x(pg, svld1(pg, out + j), old);
            z = svmla_n_f32_x(pg, z, svcvt_f32_s32_x(pg, iz), add * vs);
            svst1(pg, out + j, z);
        }
#else
        for (int j = 0; j < v_dim; ++j) out[j] = out[j] * old + (float)vt[j] * vs * add;
#endif
        l = l * old + add; m = nm;
    }
    if (l != 0.0f) {
#if defined(__ARM_FEATURE_SVE)
        int vl = (int)svcntw();
        for (int j = 0; j < v_dim; j += vl) {
            svbool_t pg = svwhilelt_b32(j, v_dim);
            svst1(pg, out + j, svdiv_n_f32_x(pg, svld1(pg, out + j), l));
        }
#else
        for (int j = 0; j < v_dim; ++j) out[j] /= l;
#endif
    }
}

static inline uint16_t k3_f32_to_bf16_rne(float f){
    uint32_t x;memcpy(&x,&f,sizeof x);x+=UINT32_C(0x7fff)+((x>>16)&1);return(uint16_t)(x>>16);
}
static inline float k3_bf16_to_f32(uint16_t b){uint32_t x=(uint32_t)b<<16;float f;memcpy(&f,&x,sizeof f);return f;}

static inline float k3_dot_f32_bf16_sve(const float *a,const uint16_t *b,int n){
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t sum=svdup_f32(0.0f);for(int i=0;i<n;i+=(int)svcntw()){svbool_t pg=svwhilelt_b32(i,n);
        svuint32_t bits=svlsl_n_u32_x(pg,svld1uh_u32(pg,b+i),16);
        sum=svmla_f32_m(pg,sum,svld1(pg,a+i),svreinterpret_f32_u32(bits));}
    return svaddv_f32(svptrue_b32(),sum);
#else
    double sum=0;for(int i=0;i<n;++i)sum+=(double)a[i]*k3_bf16_to_f32(b[i]);return(float)sum;
#endif
}

static inline void k3_attention_bf16_strided_sve(float *out,const float *q,
        const uint16_t *keys,const uint16_t *values,int tokens,int qk_dim,int v_dim,
        int key_stride,int value_stride){
    for(int j=0;j<v_dim;++j)out[j]=0.0f;
    float m=-INFINITY,l=0.0f,scale=1.0f/sqrtf((float)qk_dim);
    for(int t=0;t<tokens;++t){float score=k3_dot_f32_bf16_sve(q,keys+(size_t)t*key_stride,qk_dim)*scale;
        float nm,old,add;if(score<=m){nm=m;old=1.0f;add=expf(score-m);}
        else{nm=score;old=expf(m-score);add=1.0f;}
#if defined(__ARM_FEATURE_SVE)
        int vl=(int)svcntw();for(int j=0;j<v_dim;j+=vl){svbool_t pg=svwhilelt_b32(j,v_dim);
            svuint32_t bits=svlsl_n_u32_x(pg,svld1uh_u32(pg,values+(size_t)t*value_stride+j),16);
            svfloat32_t z=svmul_n_f32_x(pg,svld1(pg,out+j),old);
            z=svmla_n_f32_x(pg,z,svreinterpret_f32_u32(bits),add);svst1(pg,out+j,z);}
#else
        for(int j=0;j<v_dim;++j)out[j]=out[j]*old+k3_bf16_to_f32(values[(size_t)t*value_stride+j])*add;
#endif
        l=l*old+add;m=nm;
    }
    if(l!=0.0f){
#if defined(__ARM_FEATURE_SVE)
        int vl=(int)svcntw();for(int j=0;j<v_dim;j+=vl){svbool_t pg=svwhilelt_b32(j,v_dim);
            svst1(pg,out+j,svdiv_n_f32_x(pg,svld1(pg,out+j),l));}
#else
        for(int j=0;j<v_dim;++j)out[j]/=l;
#endif
    }
}

static inline void k3_attention_bf16_sve(float *out,const float *q,const uint16_t *keys,
        const uint16_t *values,int tokens,int qk_dim,int v_dim){
    k3_attention_bf16_strided_sve(out,q,keys,values,tokens,qk_dim,v_dim,qk_dim,v_dim);
}

static inline void k3_attention_bf16_interleaved_sve(float *out,const float *q,
        const uint16_t *kv,int tokens,int qk_dim,int v_dim){
    int stride=qk_dim+v_dim;
    k3_attention_bf16_strided_sve(out,q,kv,kv+qk_dim,tokens,qk_dim,v_dim,stride,stride);
}

static inline void k3_attention_heads_parallel_bf16_strided_sve(float *out,const float *q,
        const uint16_t *keys,const uint16_t *values,int heads,int tokens,int qk_dim,int v_dim,
        size_t key_head_stride,size_t value_head_stride,int key_token_stride,int value_token_stride,
        int threads,float *scratch,float *stats){
    int parts=(threads+heads-1)/heads;if(parts>tokens)parts=tokens;
    float scale=1.0f/sqrtf((float)qk_dim);
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel
    {
#pragma omp for schedule(static)
#endif
        for(int task=0;task<heads*parts;++task){int h=task/parts,p=task%parts;
            int begin=(tokens*p)/parts,end=(tokens*(p+1))/parts;float *num=scratch+(size_t)task*v_dim;
            const float *qh=q+(size_t)h*qk_dim;const uint16_t *kh=keys+(size_t)h*key_head_stride;
            const uint16_t *vh=values+(size_t)h*value_head_stride;for(int j=0;j<v_dim;++j)num[j]=0.0f;
            float m=-INFINITY,l=0.0f;for(int t=begin;t<end;++t){
                float score=k3_dot_f32_bf16_sve(qh,kh+(size_t)t*key_token_stride,qk_dim)*scale;
                float nm,old,add;if(score<=m){nm=m;old=1.0f;add=expf(score-m);}
                else{nm=score;old=expf(m-score);add=1.0f;}
#if defined(__ARM_FEATURE_SVE)
                int vl=(int)svcntw();for(int j=0;j<v_dim;j+=vl){svbool_t pg=svwhilelt_b32(j,v_dim);
                    svuint32_t bits=svlsl_n_u32_x(pg,svld1uh_u32(pg,vh+(size_t)t*value_token_stride+j),16);
                    svfloat32_t z=svmul_n_f32_x(pg,svld1(pg,num+j),old);
                    z=svmla_n_f32_x(pg,z,svreinterpret_f32_u32(bits),add);svst1(pg,num+j,z);}
#else
                for(int j=0;j<v_dim;++j)num[j]=num[j]*old+k3_bf16_to_f32(vh[(size_t)t*value_token_stride+j])*add;
#endif
                l=l*old+add;m=nm;}
            stats[(size_t)task*2]=m;stats[(size_t)task*2+1]=l;}
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for(int h=0;h<heads;++h){float m=-INFINITY;double l=0.0;
            for(int p=0;p<parts;++p)m=fmaxf(m,stats[((size_t)h*parts+p)*2]);
            for(int p=0;p<parts;++p){size_t s=((size_t)h*parts+p)*2;float weight=expf(stats[s]-m);
                l+=(double)stats[s+1]*weight;stats[s]=weight;}size_t g=(size_t)heads*parts*2+(size_t)h*2;
            stats[g]=m;stats[g+1]=(float)l;}
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for(int task=0;task<heads*v_dim;++task){int h=task/v_dim,j=task%v_dim;double z=0.0;
            size_t g=(size_t)heads*parts*2+(size_t)h*2;for(int p=0;p<parts;++p){size_t s=((size_t)h*parts+p)*2;
                z+=(double)scratch[((size_t)h*parts+p)*v_dim+j]*stats[s];}out[task]=(float)(z/stats[g+1]);}
#if defined(_OPENMP)
    }
#endif
}

static inline void k3_attention_heads_parallel_bf16_sve(float *out,const float *q,
        const uint16_t *keys,const uint16_t *values,int heads,int tokens,int cache_tokens,
        int qk_dim,int v_dim,int threads,float *scratch,float *stats){
    k3_attention_heads_parallel_bf16_strided_sve(out,q,keys,values,heads,tokens,qk_dim,v_dim,
        (size_t)cache_tokens*qk_dim,(size_t)cache_tokens*v_dim,qk_dim,v_dim,threads,scratch,stats);
}

static inline void k3_attention_heads_parallel_bf16_interleaved_sve(float *out,const float *q,
        const uint16_t *kv,int heads,int tokens,int cache_tokens,int qk_dim,int v_dim,
        int threads,float *scratch,float *stats){
    int stride=qk_dim+v_dim;size_t head_stride=(size_t)cache_tokens*stride;
    k3_attention_heads_parallel_bf16_strided_sve(out,q,kv,kv+qk_dim,heads,tokens,qk_dim,v_dim,
        head_stride,head_stride,stride,stride,threads,scratch,stats);
}

/* Parallel exact online attention for head-TP decode. Token blocks retain
 * independent (max, denominator, numerator) triples and are combined with the
 * log-sum-exp identity. scratch holds heads*threads*v_dim floats and stats
 * holds heads*threads*2 + heads*2 floats. */
static inline void k3_attention_heads_parallel_sve(float *out, const float *q,
        const float *keys, const float *values, int heads, int tokens,
        int cache_tokens, int qk_dim, int v_dim, int threads,
        float *scratch, float *stats) {
    int parts=(threads+heads-1)/heads;if(parts>tokens)parts=tokens;
    float scale=1.0f/sqrtf((float)qk_dim);
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel
    {
#pragma omp for schedule(static)
#endif
        for(int task=0;task<heads*parts;++task){
            int h=task/parts,p=task%parts;
            int begin=(tokens*p)/parts,end=(tokens*(p+1))/parts;
            float *num=scratch+(size_t)task*v_dim;
            const float *qh=q+(size_t)h*qk_dim;
            const float *kh=keys+(size_t)h*cache_tokens*qk_dim;
            const float *vh=values+(size_t)h*cache_tokens*v_dim;
            for(int j=0;j<v_dim;++j)num[j]=0.0f;
            float m=-INFINITY,l=0.0f;
            for(int t=begin;t<end;++t){
                float score=k3_dot_sve(qh,kh+(size_t)t*qk_dim,qk_dim)*scale;
                float nm,old,add;if(score<=m){nm=m;old=1.0f;add=expf(score-m);}
                else{nm=score;old=expf(m-score);add=1.0f;}
#if defined(__ARM_FEATURE_SVE)
                int vl=(int)svcntw();
                for(int j=0;j<v_dim;j+=vl){svbool_t pg=svwhilelt_b32(j,v_dim);
                    svfloat32_t z=svmul_n_f32_x(pg,svld1(pg,num+j),old);
                    z=svmla_n_f32_x(pg,z,svld1(pg,vh+(size_t)t*v_dim+j),add);
                    svst1(pg,num+j,z);}
#else
                for(int j=0;j<v_dim;++j)num[j]=num[j]*old+vh[(size_t)t*v_dim+j]*add;
#endif
                l=l*old+add;m=nm;
            }
            stats[(size_t)task*2]=m;stats[(size_t)task*2+1]=l;
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for(int h=0;h<heads;++h){
            float m=-INFINITY;double l=0.0;
            for(int p=0;p<parts;++p)m=fmaxf(m,stats[((size_t)h*parts+p)*2]);
            for(int p=0;p<parts;++p){size_t s=((size_t)h*parts+p)*2;
                float weight=expf(stats[s]-m);l+=(double)stats[s+1]*weight;stats[s]=weight;}
            size_t g=(size_t)heads*parts*2+(size_t)h*2;
            stats[g]=m;stats[g+1]=(float)l;
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for(int task=0;task<heads*v_dim;++task){
            int h=task/v_dim,j=task%v_dim;double z=0.0;
            size_t g=(size_t)heads*parts*2+(size_t)h*2;
            for(int p=0;p<parts;++p){size_t s=((size_t)h*parts+p)*2;
                z+=(double)scratch[((size_t)h*parts+p)*v_dim+j]*stats[s];}
            out[task]=(float)(z/stats[g+1]);
        }
#if defined(_OPENMP)
    }
#endif
}

/* Chunked prefill attention.  The scalar API above launches one OpenMP team
 * per query.  For long prefill chunks that makes team startup dominate the
 * MLA scan.  Keep one team for all query/head/position partitions; each query
 * still uses its own causal prefix and exact online-softmax merge. */
static inline void k3_attention_heads_batched_sve(float *out, const float *q,
        const float *keys, const float *values, int batch, int start_pos,
        int heads, int cache_tokens, int qk_dim, int v_dim, int threads,
        float *scratch, float *stats) {
    int parts = (threads + heads - 1) / heads;
    if (parts < 1) parts = 1;
    size_t scratch_stride = (size_t)heads * parts * v_dim;
    size_t stats_stride = (size_t)heads * parts * 2 + (size_t)heads * 2;
    float scale = 1.0f / sqrtf((float)qk_dim);
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel
    {
#pragma omp for schedule(static)
#endif
        for (long task = 0; task < (long)batch * heads * parts; ++task) {
            int t = (int)(task / (heads * parts));
            int rem = (int)(task % (heads * parts));
            int h = rem / parts, p = rem % parts;
            int tokens = start_pos + t + 1;
            int begin = (tokens * p) / parts;
            int end = (tokens * (p + 1)) / parts;
            float *num = scratch + (size_t)t * scratch_stride +
                         (size_t)rem * v_dim;
            const float *qh = q + (size_t)t * heads * qk_dim +
                              (size_t)h * qk_dim;
            const float *kh = keys + (size_t)h * cache_tokens * qk_dim;
            const float *vh = values + (size_t)h * cache_tokens * v_dim;
            for (int j = 0; j < v_dim; ++j) num[j] = 0.0f;
            float mm = -INFINITY, ll = 0.0f;
            for (int pos = begin; pos < end; ++pos) {
                float score = k3_dot_sve(qh, kh + (size_t)pos * qk_dim,
                                         qk_dim) * scale;
                float nm, old, add;
                if (score <= mm) {
                    nm = mm; old = 1.0f; add = expf(score - mm);
                } else {
                    nm = score; old = expf(mm - score); add = 1.0f;
                }
#if defined(__ARM_FEATURE_SVE)
                int vl = (int)svcntw();
                for (int j = 0; j < v_dim; j += vl) {
                    svbool_t pg = svwhilelt_b32(j, v_dim);
                    svfloat32_t z = svmul_n_f32_x(pg, svld1(pg, num + j), old);
                    z = svmla_n_f32_x(pg, z,
                        svld1(pg, vh + (size_t)pos * v_dim + j), add);
                    svst1(pg, num + j, z);
                }
#else
                for (int j = 0; j < v_dim; ++j)
                    num[j] = num[j] * old + vh[(size_t)pos * v_dim + j] * add;
#endif
                ll = ll * old + add;
                mm = nm;
            }
            size_t s = (size_t)t * stats_stride + (size_t)rem * 2;
            stats[s] = mm; stats[s + 1] = ll;
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (long task = 0; task < (long)batch * heads; ++task) {
            int t = (int)(task / heads), h = (int)(task % heads);
            size_t base = (size_t)t * stats_stride;
            float mm = -INFINITY; double ll = 0.0;
            for (int p = 0; p < parts; ++p)
                mm = fmaxf(mm, stats[base + (size_t)(h * parts + p) * 2]);
            for (int p = 0; p < parts; ++p) {
                size_t s = base + (size_t)(h * parts + p) * 2;
                float weight = expf(stats[s] - mm);
                ll += (double)stats[s + 1] * weight;
                stats[s] = weight;
            }
            size_t g = base + (size_t)heads * parts * 2 + (size_t)h * 2;
            stats[g] = mm; stats[g + 1] = (float)ll;
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (long task = 0; task < (long)batch * heads * v_dim; ++task) {
            int t = (int)(task / (heads * v_dim));
            int rem = (int)(task % (heads * v_dim));
            int h = rem / v_dim, j = rem % v_dim;
            size_t base = (size_t)t * stats_stride;
            size_t g = base + (size_t)heads * parts * 2 + (size_t)h * 2;
            double z = 0.0;
            for (int p = 0; p < parts; ++p) {
                size_t s = base + (size_t)(h * parts + p) * 2;
                z += (double)scratch[(size_t)t * scratch_stride +
                                     (size_t)(h * parts + p) * v_dim + j] * stats[s];
            }
            out[(size_t)t * heads * v_dim + (size_t)h * v_dim + j] =
                (float)(z / stats[g + 1]);
        }
#if defined(_OPENMP)
    }
#endif
}

/* Query-tiled MLA prefill kernel.  Vector mode uses sixteen queries to fill
 * the A64FX SVE register; scalar mode retains eight queries.  Queries share
 * every KV load and the
 * online-softmax/value update is fused.  The ordinary batched kernel remains
 * available as a correctness fallback. */
static inline void k3_attention_heads_flash8_sve(float *out, const float *q,
        const float *keys, const float *values, int batch, int start_pos,
        int heads, int cache_tokens, int qk_dim, int v_dim, int threads,
        int vector_qk) {
    int block = vector_qk ? 16 : 8;
    int packed_qk = vector_qk == 1;
    int qblocks = (batch + block - 1) / block;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int h = 0; h < heads; ++h)
        for (int qb = 0; qb < qblocks; ++qb) {
            int q0 = qb * block, qn = batch - q0;
            if (qn > block) qn = block;
            float acc[16][v_dim];
            float mm[16], ll[16];
            for (int qi = 0; qi < qn; ++qi) {
                for (int j = 0; j < v_dim; ++j) acc[qi][j] = 0.0f;
                mm[qi] = -INFINITY; ll[qi] = 0.0f;
            }
            const float *kh = keys + (size_t)h * cache_tokens * qk_dim;
            const float *vh = values + (size_t)h * cache_tokens * v_dim;
            float scale = 1.0f / sqrtf((float)qk_dim);
            float qt[576][16];
            if (packed_qk) {
                for (int d = 0; d < qk_dim; ++d) {
                    for (int qi = 0; qi < qn; ++qi)
                        qt[d][qi] = q[(size_t)(q0 + qi) * heads * qk_dim +
                                      (size_t)h * qk_dim + d];
                    for (int qi = qn; qi < 16; ++qi) qt[d][qi] = 0.0f;
                }
            }
            for (int pos = 0; pos < start_pos + q0 + qn; ++pos) {
                float scores[16] = {0};
                int qi_start = pos - (start_pos + q0);
                if (qi_start < 0) qi_start = 0;
                if (qi_start >= qn) continue;
                if (packed_qk) {
                    k3_dot8_tile_sve(scores, qt,
                        kh + (size_t)pos * qk_dim, qn, qk_dim);
                    for (int qi = qi_start; qi < qn; ++qi) scores[qi] *= scale;
                } else {
                    for (int qi = qi_start; qi < qn; ++qi) {
                        const float *qh = q + (size_t)(q0 + qi) * heads * qk_dim +
                                          (size_t)h * qk_dim;
                        scores[qi] = k3_dot_sve(qh,
                            kh + (size_t)pos * qk_dim, qk_dim) * scale;
                    }
                }
                float old[16], add[16];
                for (int qi = qi_start; qi < qn; ++qi) {
                    float score = scores[qi];
                    float nm;
                    if (score <= mm[qi]) {
                        nm = mm[qi]; old[qi] = 1.0f;
                        add[qi] = expf(score - mm[qi]);
                    } else {
                        nm = score; old[qi] = expf(mm[qi] - score);
                        add[qi] = 1.0f;
                    }
                    ll[qi] = ll[qi] * old[qi] + add[qi];
                    mm[qi] = nm;
                }
#if defined(__ARM_FEATURE_SVE)
                /* The V row is common to every query in this block.  Keep it
                 * in an SVE register while updating all live queries, avoiding
                 * qn redundant cache-line loads. */
                int vl = (int)svcntw();
                for (int j = 0; j < v_dim; j += vl) {
                    svbool_t pg = svwhilelt_b32(j, v_dim);
                    svfloat32_t vv = svld1(pg, vh + (size_t)pos * v_dim + j);
                    for (int qi = qi_start; qi < qn; ++qi) {
                        svfloat32_t z = svmul_n_f32_x(pg, svld1(pg, acc[qi] + j), old[qi]);
                        z = svmla_n_f32_x(pg, z, vv, add[qi]);
                        svst1(pg, acc[qi] + j, z);
                    }
                }
#else
                for (int qi = qi_start; qi < qn; ++qi) {
                    for (int j = 0; j < v_dim; ++j)
                        acc[qi][j] = acc[qi][j] * old[qi] +
                            vh[(size_t)pos * v_dim + j] * add[qi];
                }
#endif
            }
            for (int qi = 0; qi < qn; ++qi) {
                float *dst = out + (size_t)(q0 + qi) * heads * v_dim +
                             (size_t)h * v_dim;
                for (int j = 0; j < v_dim; ++j) dst[j] = acc[qi][j] / ll[qi];
            }
        }
}

/* Attention residual mixture: scores are normalized with softmax over candidates. */
static inline void k3_attnres_ref(float *out, const float *candidates, const float *scores,
                                  int count, int hidden) {
    float m = scores[0], z = 0.0f;
    for (int c = 1; c < count; ++c) m = fmaxf(m, scores[c]);
    for (int c = 0; c < count; ++c) z += expf(scores[c] - m);
    for (int j = 0; j < hidden; ++j) {
        double v = 0.0;
        for (int c = 0; c < count; ++c)
            v += (double)candidates[(size_t)c * hidden + j] * (expf(scores[c] - m) / z);
        out[j] = (float)v;
    }
}

static inline void k3_attnres_sve(float *out, const float *candidates, const float *scores,
                                  int count, int hidden) {
    float m = scores[0], z = 0.0f;
    for (int c = 1; c < count; ++c) m = fmaxf(m, scores[c]);
    for (int c = 0; c < count; ++c) z += expf(scores[c] - m);
#if defined(__ARM_FEATURE_SVE)
    int vl = (int)svcntw();
    for (int j = 0; j < hidden; j += vl) {
        svbool_t pg = svwhilelt_b32(j, hidden);
        svfloat32_t v = svdup_f32(0.0f);
        for (int c = 0; c < count; ++c)
            v = svmla_n_f32_x(pg, v, svld1(pg, candidates + (size_t)c * hidden + j),
                              expf(scores[c] - m) / z);
        svst1(pg, out + j, v);
    }
#else
    k3_attnres_ref(out, candidates, scores, count, hidden);
#endif
}

/* Selection uses corrected sigmoid score; returned weights use uncorrected sigmoid and sum to 1. */
static inline void k3_router_topk(const float *logits, const float *bias, int experts, int topk,
                                  int *indices, float *weights) {
    float probability[experts], score[experts];
    /* sigmoid is strictly monotonic, so the common bias-free route can pick
     * from logits directly and evaluate sigmoid for only the selected experts.
     * Besides removing most transcendental work, this preserves the selected
     * probabilities bit-for-bit with the generic path. */
    if (!bias) {
        memcpy(score, logits, (size_t)experts * sizeof(*score));
        for (int k = 0; k < topk; ++k) {
            int best = -1; float best_score = -INFINITY;
            for (int e = 0; e < experts; ++e) {
                if (score[e] > best_score ||
                    (score[e] == best_score && e < best)) {
                    best = e; best_score = score[e];
                }
            }
            indices[k] = best;
            score[best] = -INFINITY;
            probability[k] = k3_sigmoidf(logits[best]);
        }
        float sum = 0.0f;
        for (int k = 0; k < topk; ++k) sum += probability[k];
        for (int k = 0; k < topk; ++k) weights[k] = probability[k] / sum;
        return;
    }
    for (int e = 0; e < experts; ++e) {
        probability[e] = k3_sigmoidf(logits[e]);
        score[e] = probability[e] + bias[e];
    }
    for (int k = 0; k < topk; ++k) {
        int best = -1; float best_score = -INFINITY;
        for (int e = 0; e < experts; ++e) {
            if (score[e] > best_score ||
                (score[e] == best_score && e < best)) {
                best = e; best_score = score[e];
            }
        }
        indices[k] = best;
        score[best] = -INFINITY;
    }
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k)
        sum += (weights[k] = probability[indices[k]]);
    for (int k = 0; k < topk; ++k) weights[k] /= sum;
}

#endif
