/* Scalar reference primitives for GLM-5.3F graph bring-up.
 * These are correctness oracles, not performance kernels. */
#ifndef GLM53F_REF_H
#define GLM53F_REF_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline float glm53f_bf16_to_f32(uint16_t x) {
    uint32_t u = (uint32_t)x << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline void glm53f_rmsnorm_bf16(float *out, const float *x,
                                       const uint16_t *weight, int n, float eps) {
    double ss = 0.0;
    int i;
    for (i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    {
        float inv = 1.0f / sqrtf((float)(ss / n) + eps);
        for (i = 0; i < n; ++i)
            out[i] = x[i] * inv * glm53f_bf16_to_f32(weight[i]);
    }
}

static inline void glm53f_layernorm_bf16(float *out, const float *x,
        const uint16_t *weight, const uint16_t *bias, int n, float eps) {
    double sum = 0.0, ss = 0.0;
    int i;
    for (i = 0; i < n; ++i) sum += x[i];
    {
        float mean = (float)(sum / n);
        for (i = 0; i < n; ++i) {
            double d = (double)x[i] - mean;
            ss += d * d;
        }
        float inv = 1.0f / sqrtf((float)(ss / n) + eps);
        for (i = 0; i < n; ++i)
            out[i] = (x[i] - mean) * inv * glm53f_bf16_to_f32(weight[i]) +
                     glm53f_bf16_to_f32(bias[i]);
    }
}

static inline float glm53f_dot_bf16(const uint16_t *weight,
                                    const float *x, int n) {
    double sum = 0.0;
    int i;
    for (i = 0; i < n; ++i)
        sum += (double)glm53f_bf16_to_f32(weight[i]) * x[i];
    return (float)sum;
}

/* GLM-5.3F MTP fusion. The embedding of the already accepted token and the
 * target model hidden state are normalized independently, concatenated in
 * that order, then projected back to hidden width by eh_proj [H, 2H]. */
static inline void glm53f_mtp_fuse_bf16(float *out, float *scratch_2h,
        const float *embedding, const float *hidden,
        const uint16_t *enorm, const uint16_t *hnorm,
        const uint16_t *eh_proj, int width, float eps) {
    int r;
    glm53f_rmsnorm_bf16(scratch_2h, embedding, enorm, width, eps);
    glm53f_rmsnorm_bf16(scratch_2h + width, hidden, hnorm, width, eps);
    for (r = 0; r < width; ++r)
        out[r] = glm53f_dot_bf16(eh_proj + (size_t)r * 2 * width,
                                 scratch_2h, 2 * width);
}

/* Return the global vocabulary ID for this row shard. Ties retain the lower
 * global ID, allowing MPI reductions to reproduce a single-rank argmax. */
static inline int glm53f_vocab_argmax_bf16(const float *hidden,
        const uint16_t *norm, const uint16_t *head_rows, int width,
        int row_begin, int rows, float eps, float *normalized,
        float *best_logit) {
    int r, best = -1;
    float value = -INFINITY;
    glm53f_rmsnorm_bf16(normalized, hidden, norm, width, eps);
    for (r = 0; r < rows; ++r) {
        float z = glm53f_dot_bf16(head_rows + (size_t)r * width,
                                  normalized, width);
        int id = row_begin + r;
        if (best < 0 || z > value || (z == value && id < best)) {
            best = id;
            value = z;
        }
    }
    if (best_logit) *best_logit = value;
    return best;
}

/* Stable greedy top-k. Equal scores retain the lower source index, which makes
 * the CPU oracle deterministic across thread counts and MPI layouts. */
static inline void glm53f_topk_stable(const float *score, int n, int k,
                                      int *index) {
    int i, j;
    for (j = 0; j < k; ++j) index[j] = -1;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            int old = index[j];
            if (old < 0 || score[i] > score[old] ||
                (score[i] == score[old] && i < old)) {
                int z;
                for (z = k - 1; z > j; --z) index[z] = index[z - 1];
                index[j] = i;
                break;
            }
        }
    }
}

/* Decode-time k-pool indexer for an unpadded causal sequence. Complete pools
 * compete by their learned compressed key; the current incomplete tail is
 * appended as raw indices and never competes with complete pools. */
static inline int glm53f_index_select_decode(float *pool_keys, int *selected,
        const float *query, const float *head_weight, const float *key_cache,
        const float *gate_cache, const float *ape, int tokens, int kpool,
        int index_topk, int heads, int dim) {
    int pools = tokens / kpool;
    int choose = index_topk / kpool;
    int p, h, d, z, out = 0;
    float *score;
    int *picked;
    if (choose > pools) choose = pools;
    score = (float *)malloc((size_t)(pools ? pools : 1) * sizeof(float));
    picked = (int *)malloc((size_t)(choose ? choose : 1) * sizeof(int));
    if (!score || !picked) { free(score); free(picked); return -1; }
    for (p = 0; p < pools; ++p) {
        float *pk = pool_keys + (size_t)p * dim;
        for (d = 0; d < dim; ++d) {
            float mx = -INFINITY, den = 0.0f, val = 0.0f;
            for (z = 0; z < kpool; ++z) {
                float a = gate_cache[(size_t)(p * kpool + z) * dim + d] +
                          ape[(size_t)z * dim + d];
                if (a > mx) mx = a;
            }
            for (z = 0; z < kpool; ++z) {
                float a = expf(gate_cache[(size_t)(p * kpool + z) * dim + d] +
                                ape[(size_t)z * dim + d] - mx);
                den += a;
                val += a * key_cache[(size_t)(p * kpool + z) * dim + d];
            }
            pk[d] = val / den;
        }
        score[p] = 0.0f;
        for (h = 0; h < heads; ++h) {
            double dot = 0.0;
            for (d = 0; d < dim; ++d)
                dot += (double)query[(size_t)h * dim + d] * pk[d];
            if (dot > 0.0)
                score[p] += head_weight[h] * (float)(dot / sqrt((double)dim)) /
                            sqrtf((float)heads);
        }
    }
    glm53f_topk_stable(score, pools, choose, picked);
    for (z = 0; z < choose; ++z)
        for (d = 0; d < kpool; ++d) selected[out++] = picked[z] * kpool + d;
    for (z = pools * kpool; z < tokens; ++z) selected[out++] = z;
    free(picked); free(score);
    return out;
}

/* Exact decode reference for NoPE compressed-latent MLA after sparse indices
 * have been selected. kv_b is [heads*(key_dim+value_dim), latent_dim] BF16. */
static inline void glm53f_mla_selected_bf16(float *out, const float *query,
        const float *latent_cache, const uint16_t *kv_b, const int *selected,
        int n_selected, int heads, int key_dim, int value_dim, int latent_dim) {
    int h, j, p, d;
    float *logit = (float *)malloc((size_t)n_selected * sizeof(float));
    float *value = (float *)malloc((size_t)n_selected * value_dim * sizeof(float));
    if (!logit || !value) { free(logit); free(value); return; }
    for (h = 0; h < heads; ++h) {
        float mx = -INFINITY, sum = 0.0f;
        const uint16_t *wk = kv_b + (size_t)h * (key_dim + value_dim) * latent_dim;
        const uint16_t *wv = wk + (size_t)key_dim * latent_dim;
        for (p = 0; p < n_selected; ++p) {
            const float *z = latent_cache + (size_t)selected[p] * latent_dim;
            double qk = 0.0;
            for (j = 0; j < key_dim; ++j) {
                double kval = 0.0;
                for (d = 0; d < latent_dim; ++d)
                    kval += (double)glm53f_bf16_to_f32(wk[(size_t)j * latent_dim + d]) * z[d];
                qk += (double)query[(size_t)h * key_dim + j] * kval;
            }
            logit[p] = (float)(qk / sqrt((double)key_dim));
            if (logit[p] > mx) mx = logit[p];
            for (j = 0; j < value_dim; ++j) {
                double v = 0.0;
                for (d = 0; d < latent_dim; ++d)
                    v += (double)glm53f_bf16_to_f32(wv[(size_t)j * latent_dim + d]) * z[d];
                value[(size_t)p * value_dim + j] = (float)v;
            }
        }
        for (p = 0; p < n_selected; ++p) { logit[p] = expf(logit[p] - mx); sum += logit[p]; }
        for (j = 0; j < value_dim; ++j) {
            double y = 0.0;
            for (p = 0; p < n_selected; ++p)
                y += (double)(logit[p] / sum) * value[(size_t)p * value_dim + j];
            out[(size_t)h * value_dim + j] = (float)y;
        }
    }
    free(value); free(logit);
}

/* Algebraically absorbed form of the selected MLA reference. It avoids
 * expanding K/V for every cached token:
 *   q_latent = q @ Wk; logits = q_latent @ latent;
 *   v_latent = softmax(logits) @ latent; out = Wv @ v_latent.
 * Accumulation order differs from the direct oracle, so callers should compare
 * with a numerical tolerance rather than requiring bit identity. */
static inline int glm53f_mla_selected_absorbed_bf16(float *out,
        const float *query, const float *latent_cache, const uint16_t *kv_b,
        const int *selected, int n_selected, int heads, int key_dim,
        int value_dim, int latent_dim) {
    float *q_latent, *v_latent, *logit;
    int h, j, p, d;
    if (n_selected <= 0) return -1;
    q_latent = (float *)malloc((size_t)latent_dim * sizeof(float));
    v_latent = (float *)malloc((size_t)latent_dim * sizeof(float));
    logit = (float *)malloc((size_t)n_selected * sizeof(float));
    if (!q_latent || !v_latent || !logit) {
        free(logit); free(v_latent); free(q_latent); return -1;
    }
    for (h = 0; h < heads; ++h) {
        const uint16_t *wk = kv_b + (size_t)h * (key_dim + value_dim) * latent_dim;
        const uint16_t *wv = wk + (size_t)key_dim * latent_dim;
        float mx = -INFINITY, sum = 0.0f;
        for (d = 0; d < latent_dim; ++d) {
            double v = 0.0;
            for (j = 0; j < key_dim; ++j)
                v += (double)query[(size_t)h * key_dim + j] *
                     glm53f_bf16_to_f32(wk[(size_t)j * latent_dim + d]);
            q_latent[d] = (float)(v / sqrt((double)key_dim));
        }
        for (p = 0; p < n_selected; ++p) {
            const float *z = latent_cache + (size_t)selected[p] * latent_dim;
            double v = 0.0;
            for (d = 0; d < latent_dim; ++d) v += (double)q_latent[d] * z[d];
            logit[p] = (float)v;
            if (logit[p] > mx) mx = logit[p];
        }
        for (p = 0; p < n_selected; ++p) {
            logit[p] = expf(logit[p] - mx); sum += logit[p];
        }
        memset(v_latent, 0, (size_t)latent_dim * sizeof(float));
        for (p = 0; p < n_selected; ++p) {
            const float *z = latent_cache + (size_t)selected[p] * latent_dim;
            float a = logit[p] / sum;
            for (d = 0; d < latent_dim; ++d) v_latent[d] += a * z[d];
        }
        for (j = 0; j < value_dim; ++j) {
            double v = 0.0;
            for (d = 0; d < latent_dim; ++d)
                v += (double)glm53f_bf16_to_f32(wv[(size_t)j * latent_dim + d]) * v_latent[d];
            out[(size_t)h * value_dim + j] = (float)v;
        }
    }
    free(logit); free(v_latent); free(q_latent);
    return 0;
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

/* Scalar checkpoint mHC site oracle. fn is BF16 [(2+hc)*hc, hc*width],
 * while base and scale are F32. */
static inline void glm53f_mhc_pre(float *collapsed, float *post, float *comb,
        const float *streams, const uint16_t *fn, const float *base,
        const float *scale, int hc, int width, int iters, float norm_eps,
        float hc_eps) {
    int flat = hc * width, mix = (2 + hc) * hc, m, k, d;
    double ss = 0.0;
    float *logits = (float *)malloc((size_t)mix * sizeof(float));
    if (!logits) return;
    for (d = 0; d < flat; ++d) ss += (double)streams[d] * streams[d];
    {
        float inv = 1.0f / sqrtf((float)(ss / flat) + norm_eps);
        for (m = 0; m < mix; ++m) {
            double v = 0.0;
            for (d = 0; d < flat; ++d)
                v += (double)glm53f_bf16_to_f32(fn[(size_t)m * flat + d]) * streams[d];
            logits[m] = (float)v * inv;
        }
    }
    for (k = 0; k < hc; ++k) {
        logits[k] = glm53f_sigmoid(logits[k] * scale[0] + base[k]) + hc_eps;
        post[k] = 2.0f * glm53f_sigmoid(logits[hc + k] * scale[1] + base[hc + k]);
    }
    for (m = 0; m < hc * hc; ++m)
        comb[m] = logits[2 * hc + m] * scale[2] + base[2 * hc + m];
    glm53f_mhc_sinkhorn(comb, hc, iters, hc_eps);
    for (d = 0; d < width; ++d) {
        double v = 0.0;
        for (k = 0; k < hc; ++k) v += (double)logits[k] * streams[(size_t)k * width + d];
        collapsed[d] = (float)v;
    }
    free(logits);
}

static inline void glm53f_mhc_post(float *streams, const float *residual,
        const float *sublayer, const float *post, const float *comb,
        int hc, int width) {
    int k, j, d;
    for (k = 0; k < hc; ++k)
        for (d = 0; d < width; ++d) {
            double v = (double)post[k] * sublayer[d];
            for (j = 0; j < hc; ++j)
                v += (double)comb[(size_t)j * hc + k] *
                     residual[(size_t)j * width + d];
            streams[(size_t)k * width + d] = (float)v;
        }
}

static inline size_t glm53f_cp_slots(size_t ctx, int ranks) {
    return (ctx + (size_t)ranks - 1) / (size_t)ranks;
}
static inline int glm53f_cp_owner(size_t pos, int ranks) { return (int)(pos % (size_t)ranks); }
static inline size_t glm53f_cp_slot(size_t pos, int ranks) { return pos / (size_t)ranks; }

#endif
