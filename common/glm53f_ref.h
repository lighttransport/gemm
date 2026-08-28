/* Scalar reference primitives for GLM-5.3F graph bring-up.
 * These are correctness oracles, not performance kernels. */
#ifndef GLM53F_REF_H
#define GLM53F_REF_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline float glm53f_bf16_to_f32(uint16_t x) {
    uint32_t u = (uint32_t)x << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline void glm53f_l2norm(float *x, int n, float eps) {
    double ss = 0.0;
    int i;
    for (i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    ss = 1.0 / sqrt(ss + eps);
    for (i = 0; i < n; ++i) x[i] = (float)(x[i] * ss);
}

/* One token/head of the HF recurrent_kimi_delta_attention fallback.
 * state is row-major [key_dim,value_dim]. q and k are normalized in-place by
 * the caller when use_qk_l2norm_in_kernel is enabled. */
static inline void glm53f_kda_step(float *state, const float *q, const float *k,
                                   const float *v, float log_decay, float beta,
                                   int key_dim, int value_dim, float *out) {
    float decay = expf(log_decay);
    float scale = 1.0f / sqrtf((float)key_dim);
    int d, j;
    for (j = 0; j < value_dim; ++j) {
        float mem = 0.0f;
        for (d = 0; d < key_dim; ++d) {
            size_t z = (size_t)d * value_dim + j;
            state[z] *= decay;
            mem += state[z] * k[d];
        }
        {
            float delta = (v[j] - mem) * beta;
            float y = 0.0f;
            for (d = 0; d < key_dim; ++d) {
                size_t z = (size_t)d * value_dim + j;
                state[z] += k[d] * delta;
                y += state[z] * (q[d] * scale);
            }
            out[j] = y;
        }
    }
}

/* Same recurrence with state-streaming loop order.  This is the A64FX decode
 * layout: two contiguous passes over [key_dim,value_dim], no strided loads. */
static inline void glm53f_kda_step_streamed(float *state, const float *q, const float *k,
                                            const float *v, float log_decay, float beta,
                                            int key_dim, int value_dim, float *out,
                                            float *work) {
    float decay = expf(log_decay), scale = 1.0f / sqrtf((float)key_dim);
    int d, j;
    memset(work, 0, (size_t)value_dim * sizeof(*work));
    for (d = 0; d < key_dim; ++d) {
        float *row = state + (size_t)d * value_dim;
        float kd = k[d];
        for (j = 0; j < value_dim; ++j) {
            row[j] *= decay;
            work[j] += row[j] * kd;
        }
    }
    for (j = 0; j < value_dim; ++j) { work[j] = (v[j] - work[j]) * beta; out[j] = 0.0f; }
    for (d = 0; d < key_dim; ++d) {
        float *row = state + (size_t)d * value_dim;
        float kd = k[d], qd = q[d] * scale;
        for (j = 0; j < value_dim; ++j) {
            row[j] += kd * work[j];
            out[j] += row[j] * qd;
        }
    }
}

static inline float glm53f_sigmoid(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}

/* Exact GLM-5.3F router rule for n_group=topk_group=1. */
static inline void glm53f_router_topk(const float *logits, const float *bias,
                                      int n, int k, float routed_scale,
                                      int *ids, float *weights) {
    int i, j;
    float sum = 0.0f;
    for (j = 0; j < k; ++j) { ids[j] = -1; weights[j] = -INFINITY; }
    for (i = 0; i < n; ++i) {
        float choice = glm53f_sigmoid(logits[i]) + (bias ? bias[i] : 0.0f);
        for (j = 0; j < k; ++j) if (choice > weights[j]) {
            int z;
            for (z = k - 1; z > j; --z) { weights[z] = weights[z - 1]; ids[z] = ids[z - 1]; }
            weights[j] = choice; ids[j] = i; break;
        }
    }
    for (j = 0; j < k; ++j) { weights[j] = glm53f_sigmoid(logits[ids[j]]); sum += weights[j]; }
    for (j = 0; j < k; ++j) weights[j] = weights[j] / (sum + 1e-20f) * routed_scale;
}

/* Sinkhorn projection used by mHC after comb softmax + eps. */
static inline void glm53f_mhc_sinkhorn(float *comb, int hc, int iters, float eps) {
    int i, j, z;
    for (i = 0; i < hc; ++i) {
        float mx = comb[(size_t)i * hc];
        float sum = 0.0f;
        for (j = 1; j < hc; ++j) if (comb[(size_t)i * hc + j] > mx) mx = comb[(size_t)i * hc + j];
        for (j = 0; j < hc; ++j) { size_t q = (size_t)i * hc + j; comb[q] = expf(comb[q] - mx) + eps; sum += comb[q]; }
        for (j = 0; j < hc; ++j) comb[(size_t)i * hc + j] /= sum;
    }
    for (z = 0; z < iters; ++z) {
        if (z > 0) for (i = 0; i < hc; ++i) {
            float sum = eps;
            for (j = 0; j < hc; ++j) sum += comb[(size_t)i * hc + j];
            for (j = 0; j < hc; ++j) comb[(size_t)i * hc + j] /= sum;
        }
        for (j = 0; j < hc; ++j) {
            float sum = eps;
            for (i = 0; i < hc; ++i) sum += comb[(size_t)i * hc + j];
            for (i = 0; i < hc; ++i) comb[(size_t)i * hc + j] /= sum;
        }
    }
}

static inline size_t glm53f_cp_slots(size_t ctx, int ranks) {
    return (ctx + (size_t)ranks - 1) / (size_t)ranks;
}
static inline int glm53f_cp_owner(size_t pos, int ranks) { return (int)(pos % (size_t)ranks); }
static inline size_t glm53f_cp_slot(size_t pos, int ranks) { return pos / (size_t)ranks; }

#endif
