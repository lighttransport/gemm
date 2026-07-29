#ifndef K3_KERNELS_H
#define K3_KERNELS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
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

/* Flatten [head][value] into enough independent row tasks to occupy all A64FX
 * cores even when tensor parallelism leaves only one to eight heads per rank. */
static inline void k3_kda_step_decay_parallel_sve(float *out, const float *q,
        const float *k, const float *v, const float *decay,
        const float *beta, float *state, int heads, int key_dim,
        int value_dim, int threads) {
    float scale=1.0f/sqrtf((float)key_dim);
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#else
    (void)threads;
#endif
    for(int task=0;task<heads*value_dim;++task){
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
        float delta=beta[h]*(v[(size_t)h*value_dim+j]-k3_dot_sve(kh,row,key_dim));
#if defined(__ARM_FEATURE_SVE)
        int vl2=(int)svcntw();
        for(int d=0;d<key_dim;d+=vl2){svbool_t pg=svwhilelt_b32(d,key_dim);
            svst1(pg,row+d,svmla_n_f32_x(pg,svld1(pg,row+d),svld1(pg,kh+d),delta));}
#else
        for(int d=0;d<key_dim;++d)row[d]+=kh[d]*delta;
#endif
        out[(size_t)h*value_dim+j]=k3_dot_sve(qh,row,key_dim)*scale;
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

/* Parallel exact online attention for head-TP decode. Token blocks retain
 * independent (max, denominator, numerator) triples and are combined with the
 * log-sum-exp identity. scratch holds heads*threads*v_dim floats and stats
 * holds heads*threads*2 + heads*2 floats. */
static inline void k3_attention_heads_parallel_sve(float *out, const float *q,
        const float *keys, const float *values, int heads, int tokens,
        int cache_tokens, int qk_dim, int v_dim, int threads,
        float *scratch, float *stats) {
    int parts=threads<tokens?threads:tokens;
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
                float nm=fmaxf(m,score),old=expf(m-nm),add=expf(score-nm);
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
            float m=-INFINITY,l=0.0f;
            for(int p=0;p<parts;++p)m=fmaxf(m,stats[((size_t)h*parts+p)*2]);
            for(int p=0;p<parts;++p){size_t s=((size_t)h*parts+p)*2;
                l+=stats[s+1]*expf(stats[s]-m);}
            size_t g=(size_t)heads*parts*2+(size_t)h*2;
            stats[g]=m;stats[g+1]=l;
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for(int task=0;task<heads*v_dim;++task){
            int h=task/v_dim,j=task%v_dim;float z=0.0f;
            size_t g=(size_t)heads*parts*2+(size_t)h*2;
            for(int p=0;p<parts;++p){size_t s=((size_t)h*parts+p)*2;
                z+=scratch[((size_t)h*parts+p)*v_dim+j]*expf(stats[s]-stats[g]);}
            out[task]=z/stats[g+1];
        }
#if defined(_OPENMP)
    }
#endif
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
