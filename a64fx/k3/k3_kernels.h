#ifndef K3_KERNELS_H
#define K3_KERNELS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

enum {
    K3_HIDDEN = 7168,
    K3_HEADS = 96,
    K3_HEAD_DIM = 128,
    K3_KDA_LAYERS = 69,
    K3_MLA_LAYERS = 24,
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

/* Kimi's safe lower-bounded gate. The release checkpoint stores A_log[key_dim]
 * (128), shared over heads, despite the bundled Python constructor declaring
 * A_log[num_heads]. The checkpoint shape is the runtime contract. */
static inline void k3_kda_log_decay(float *out, const float *g_raw, const float *a_log,
                                    const float *dt_bias, int heads, int key_dim) {
    for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < key_dim; ++d) {
            int i = h * key_dim + d;
            out[i] = -5.0f * k3_sigmoidf(expf(a_log[d]) * (g_raw[i] + dt_bias[i]));
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
            float delta = beta[h] * (v[h * value_dim + j] - k3_dot_sve(kh, row, key_dim));
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
            out[h * value_dim + j] = k3_dot_sve(qh, row, key_dim) * scale;
        }
    }
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
        float nm = fmaxf(m, score), old = expf(m - nm), add = expf(score - nm);
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
        float nm = fmaxf(m, score), old = expf(m - nm), add = expf(score - nm);
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
    for (int k = 0; k < topk; ++k) {
        int best = -1; float best_score = -INFINITY;
        for (int e = 0; e < experts; ++e) {
            int used = 0;
            for (int p = 0; p < k; ++p) used |= indices[p] == e;
            float score = k3_sigmoidf(logits[e]) + (bias ? bias[e] : 0.0f);
            if (!used && (score > best_score || (score == best_score && e < best))) {
                best = e; best_score = score;
            }
        }
        indices[k] = best;
    }
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) sum += (weights[k] = k3_sigmoidf(logits[indices[k]]));
    for (int k = 0; k < topk; ++k) weights[k] /= sum;
}

#endif
