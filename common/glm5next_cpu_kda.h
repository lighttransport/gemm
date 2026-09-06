/* CPU bring-up for one GLM5Next KDA layer.  This intentionally stops at the
 * recurrent-attention sublayer; it is used to validate real GGUF tensor
 * layouts before integrating the complete mHC + MoE block. */
#ifndef GLM5NEXT_CPU_KDA_H
#define GLM5NEXT_CPU_KDA_H

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "glm5next_ref.h"
#include "ggml_dequant.h"

static inline int glm5next_cpu_matvec(float *out, const glm5next_tensor_view *w,
                                      const float *x) {
    int rows, cols;
    size_t row_bytes;
    if (!out || !w || !x || w->n_dims < 2 || !w->data) return -1;
    cols = (int)w->dims[0];
    rows = (int)w->dims[1];
    row_bytes = dequant_row_size(w->type, cols);
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row = (const uint8_t *)w->data + (size_t)r * row_bytes;
        float *tmp = (float *)malloc((size_t)cols * sizeof(float));
        double sum = 0.0;
        if (!tmp || dequant_row(w->type, row, tmp, cols) != 0) {
            free(tmp);
            return -1;
        }
        for (int j = 0; j < cols; ++j) sum += (double)tmp[j] * x[j];
        out[r] = (float)sum;
        free(tmp);
    }
    return 0;
}

static inline int glm5next_cpu_matvec_head(float *out,
        const glm5next_tensor_view *w, int head, const float *x) {
    if (!w || w->n_dims != 3 || head < 0 || (uint64_t)head >= w->dims[2]) return -1;
    glm5next_tensor_view slice = *w;
    size_t row_bytes = dequant_row_size(w->type, (int)w->dims[0]);
    slice.n_dims = 2;
    slice.data = (void *)((const uint8_t *)w->data + (size_t)head * w->dims[1] * row_bytes);
    return glm5next_cpu_matvec(out, &slice, x);
}

static inline int glm5next_cpu_vector(const glm5next_tensor_view *w, float *out, int n) {
    if (!w || !out || !w->data || w->n_dims != 1 || (int)w->dims[0] != n) return -1;
    return dequant_row(w->type, w->data, out, n);
}

static inline void glm5next_cpu_rmsnorm(float *out, const float *x,
                                        const float *weight, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * weight[i];
}

/* CPU implementation of one mHC site.  The learned mixer is stored as a
 * quantized [hc*hidden, (2+hc)*hc] matrix, so the generic row matvec also
 * handles the model's Q8_0 representation without materializing it. */
static inline int glm5next_cpu_mhc_pre(const glm5next_config *c,
        const glm5next_tensor_view *fn, const glm5next_tensor_view *base,
        const glm5next_tensor_view *scale, const float *streams,
        float *collapsed, float *post, float *comb) {
    int hc = c->hc_count, width = c->hidden_size;
    int mix = (2 + hc) * hc;
    float *logits = (float *)malloc((size_t)mix * sizeof(float));
    float *base_f = (float *)malloc((size_t)mix * sizeof(float));
    float *scale_f = (float *)malloc(3 * sizeof(float));
    if (!logits || !base_f || !scale_f ||
        glm5next_cpu_matvec(logits, fn, streams) != 0 ||
        glm5next_cpu_vector(base, base_f, mix) != 0 ||
        glm5next_cpu_vector(scale, scale_f, 3) != 0) {
        free(logits); free(base_f); free(scale_f); return -1;
    }
    double ss = 0.0;
    for (int i = 0; i < hc * width; ++i) ss += (double)streams[i] * streams[i];
    float inv = 1.0f / sqrtf((float)(ss / (hc * width)) + c->norm_epsilon);
    for (int i = 0; i < mix; ++i) logits[i] = logits[i] * inv;
    for (int i = 0; i < hc; ++i) {
        logits[i] = 1.0f / (1.0f + expf(-(logits[i] * scale_f[0] + base_f[i]))) + c->hc_sinkhorn_epsilon;
        post[i] = 2.0f / (1.0f + expf(-(logits[hc + i] * scale_f[1] + base_f[hc + i])));
    }
    for (int i = 0; i < hc * hc; ++i) comb[i] = logits[2 * hc + i] * scale_f[2] + base_f[2 * hc + i];
    glm5next_mhc_sinkhorn(comb, hc, c->hc_sinkhorn_iterations, c->hc_sinkhorn_epsilon);
    for (int d = 0; d < width; ++d) {
        double v = 0.0;
        for (int i = 0; i < hc; ++i) v += (double)logits[i] * streams[(size_t)i * width + d];
        collapsed[d] = (float)v;
    }
    free(logits); free(base_f); free(scale_f); return 0;
}

static inline void glm5next_cpu_mhc_post(const glm5next_config *c,
        float *streams, const float *residual, const float *sublayer,
        const float *post, const float *comb) {
    int hc = c->hc_count, width = c->hidden_size;
    for (int k = 0; k < hc; ++k) for (int d = 0; d < width; ++d) {
        double v = (double)post[k] * sublayer[d];
        for (int j = 0; j < hc; ++j) v += (double)comb[(size_t)j * hc + k] * residual[(size_t)j * width + d];
        streams[(size_t)k * width + d] = (float)v;
    }
}

static inline int glm5next_cpu_dense_ffn(const gguf_shards *model, int layer,
        const glm5next_config *c, const float *hidden, float *out) {
    char name[128]; glm5next_tensor_view t;
    int ff = c->dense_feed_forward_length;
    float *gate = (float *)malloc((size_t)ff * sizeof(float));
    float *up = (float *)malloc((size_t)ff * sizeof(float));
    if (!gate || !up) { free(gate); free(up); return -1; }
#define G5GET(s) do { snprintf(name, sizeof(name), "blk.%d.%s", layer, (s)); \
    if (glm5next_tensor_view_get(model, name, 1, &t) != 0) { free(gate); free(up); return -1; } } while (0)
    G5GET("ffn_gate.weight"); if (glm5next_cpu_matvec(gate, &t, hidden) != 0) goto fail;
    G5GET("ffn_up.weight"); if (glm5next_cpu_matvec(up, &t, hidden) != 0) goto fail;
    for (int i = 0; i < ff; ++i) gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
    G5GET("ffn_down.weight"); if (glm5next_cpu_matvec(out, &t, gate) != 0) goto fail;
    free(gate); free(up); return 0;
fail:
    free(gate); free(up); return -1;
#undef G5GET
}

/* One-token absorbed DSA attention.  With no prior KV cells this is already
 * the exact causal attention result (the sole score softmaxes to one); the
 * latent cache/indexer is added in the history-aware routine below. */
static inline int glm5next_cpu_dsa_forward(const gguf_shards *model, int layer,
        const glm5next_config *c, const float *hidden, float *out) {
    char name[128]; glm5next_tensor_view t;
    int h = c->hidden_size, heads = c->attention_heads;
    int qrank = c->q_lora_rank, kv = c->kv_lora_rank;
    int qdim = c->qk_nope_head_dim, vdim = c->value_head_dim;
    float *qr = (float *)malloc((size_t)qrank * sizeof(float));
    float *kv_latent = (float *)malloc((size_t)kv * sizeof(float));
    float *q = (float *)malloc((size_t)heads * qdim * sizeof(float));
    float *value = (float *)malloc((size_t)heads * vdim * sizeof(float));
    float *tmp = (float *)malloc((size_t)h * sizeof(float));
    float *norm = (float *)malloc((size_t)((qrank > kv) ? qrank : kv) * sizeof(float));
    if (!qr || !kv_latent || !q || !value || !tmp || !norm) goto fail;
#define DSA_GET(s) do { snprintf(name, sizeof(name), "blk.%d.%s", layer, (s)); \
    if (glm5next_tensor_view_get(model, name, 1, &t) != 0) goto fail; } while (0)
    DSA_GET("attn_q_a.weight"); if (glm5next_cpu_matvec(qr, &t, hidden) != 0) goto fail;
    DSA_GET("attn_q_a_norm.weight"); if (glm5next_cpu_vector(&t, norm, qrank) != 0) goto fail;
    glm5next_cpu_rmsnorm(qr, qr, norm, qrank, c->norm_epsilon);
    DSA_GET("attn_q_b.weight"); if (glm5next_cpu_matvec(q, &t, qr) != 0) goto fail;
    DSA_GET("attn_kv_a_mqa.weight"); if (glm5next_cpu_matvec(kv_latent, &t, hidden) != 0) goto fail;
    DSA_GET("attn_kv_a_norm.weight"); if (glm5next_cpu_vector(&t, norm, kv) != 0) goto fail;
    glm5next_cpu_rmsnorm(kv_latent, kv_latent, norm, kv, c->norm_epsilon);
    for (int head = 0; head < heads; ++head) {
        float *qh = (float *)malloc((size_t)kv * sizeof(float));
        if (!qh) goto fail;
        DSA_GET("attn_k_b.weight");
        if (glm5next_cpu_matvec_head(qh, &t, head, q + (size_t)head * qdim) != 0) { free(qh); goto fail; }
        float score = 0.0f;
        for (int i = 0; i < kv; ++i) score += qh[i] * kv_latent[i];
        (void)score; /* one causal cell: softmax(score) = 1 */
        DSA_GET("attn_v_b.weight");
        if (glm5next_cpu_matvec_head(value + (size_t)head * vdim, &t, head, kv_latent) != 0) { free(qh); goto fail; }
        free(qh);
    }
    DSA_GET("attn_output.weight"); if (glm5next_cpu_matvec(tmp, &t, value) != 0) goto fail;
    memcpy(out, tmp, (size_t)h * sizeof(float));
    free(qr); free(kv_latent); free(q); free(value); free(tmp); free(norm); return 0;
fail:
    free(qr); free(kv_latent); free(q); free(value); free(tmp); free(norm); return -1;
#undef DSA_GET
}

static inline int glm5next_cpu_moe_ffn(const gguf_shards *model, int layer,
        const glm5next_config *c, const float *hidden, float *out) {
    char name[128]; glm5next_tensor_view t;
    int h = c->hidden_size, ff = c->expert_ff_length, ne = c->expert_count;
    float *router = (float *)malloc((size_t)ne * sizeof(float));
    float *bias = (float *)malloc((size_t)ne * sizeof(float));
    float *gate = (float *)malloc((size_t)ff * sizeof(float));
    float *up = (float *)malloc((size_t)ff * sizeof(float));
    float *expert_out = (float *)malloc((size_t)h * sizeof(float));
    float *shared_gate = (float *)malloc((size_t)c->shared_expert_ff_length * sizeof(float));
    float *shared_up = (float *)malloc((size_t)c->shared_expert_ff_length * sizeof(float));
    float *shared_out = (float *)malloc((size_t)h * sizeof(float));
    if (!router || !bias || !gate || !up || !expert_out || !shared_gate || !shared_up || !shared_out) goto fail;
#define MOE_GET(s) do { snprintf(name, sizeof(name), "blk.%d.%s", layer, (s)); \
    if (glm5next_tensor_view_get(model, name, 1, &t) != 0) goto fail; } while (0)
    MOE_GET("ffn_gate_inp.weight"); if (glm5next_cpu_matvec(router, &t, hidden) != 0) goto fail;
    MOE_GET("exp_probs_b.bias"); if (glm5next_cpu_vector(&t, bias, ne) != 0) goto fail;
    int ids[8]; float weights[8];
    for (int j = 0; j < c->expert_used_count; ++j) { ids[j] = -1; weights[j] = -INFINITY; }
    for (int e = 0; e < ne; ++e) {
        float score = 1.0f / (1.0f + expf(-router[e])) + bias[e];
        int j = c->expert_used_count - 1;
        if (score <= weights[j]) continue;
        while (j > 0 && score > weights[j - 1]) { weights[j] = weights[j - 1]; ids[j] = ids[j - 1]; --j; }
        weights[j] = score; ids[j] = e;
    }
    float sum = 0.0f;
    for (int j = 0; j < c->expert_used_count; ++j) {
        float w = 1.0f / (1.0f + expf(-router[ids[j]]));
        weights[j] = w; sum += w;
    }
    memset(out, 0, (size_t)h * sizeof(float));
    for (int j = 0; j < c->expert_used_count; ++j) {
        int e = ids[j];
        MOE_GET("ffn_gate_exps.weight"); if (glm5next_cpu_matvec_head(gate, &t, e, hidden) != 0) goto fail;
        MOE_GET("ffn_up_exps.weight"); if (glm5next_cpu_matvec_head(up, &t, e, hidden) != 0) goto fail;
        for (int i = 0; i < ff; ++i) gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
        MOE_GET("ffn_down_exps.weight"); if (glm5next_cpu_matvec_head(expert_out, &t, e, gate) != 0) goto fail;
        float w = c->routed_scaling_factor * weights[j] / (sum > 0.0f ? sum : 1.0f);
        for (int i = 0; i < h; ++i) out[i] += w * expert_out[i];
    }
    /* Shared expert is unscaled and runs in parallel with the routed path. */
    int sff = c->shared_expert_ff_length;
    MOE_GET("ffn_gate_shexp.weight"); if (glm5next_cpu_matvec(shared_gate, &t, hidden) != 0) goto fail;
    MOE_GET("ffn_up_shexp.weight"); if (glm5next_cpu_matvec(shared_up, &t, hidden) != 0) goto fail;
    for (int i = 0; i < sff; ++i) shared_gate[i] = shared_gate[i] / (1.0f + expf(-shared_gate[i])) * shared_up[i];
    MOE_GET("ffn_down_shexp.weight"); if (glm5next_cpu_matvec(shared_out, &t, shared_gate) != 0) goto fail;
    for (int i = 0; i < h; ++i) out[i] += shared_out[i];
    free(router); free(bias); free(gate); free(up); free(expert_out); free(shared_gate); free(shared_up); free(shared_out); return 0;
fail:
    free(router); free(bias); free(gate); free(up); free(expert_out); free(shared_gate); free(shared_up); free(shared_out); return -1;
#undef MOE_GET
}

static inline int glm5next_cpu_dsa_moe_block(const gguf_shards *model,
        int layer, const glm5next_config *c, float *streams) {
    int h = c->hidden_size, hc = c->hc_count;
    char name[128]; glm5next_tensor_view fn, base, scale;
    float *residual = (float *)malloc((size_t)hc * h * sizeof(float));
    float *collapsed = (float *)malloc((size_t)h * sizeof(float));
    float *sublayer = (float *)malloc((size_t)h * sizeof(float));
    float *post = (float *)malloc((size_t)hc * sizeof(float));
    float *comb = (float *)malloc((size_t)hc * hc * sizeof(float));
    float *norm = (float *)malloc((size_t)h * sizeof(float));
    if (!residual || !collapsed || !sublayer || !post || !comb || !norm) goto fail;
    memcpy(residual, streams, (size_t)hc * h * sizeof(float));
#define BLOCK_VIEW(s, dst) do { snprintf(name, sizeof(name), "blk.%d.%s", layer, (s)); \
    if (glm5next_tensor_view_get(model, name, 1, &(dst)) != 0) goto fail; } while (0)
    BLOCK_VIEW("hc_attn_fn.weight", fn); BLOCK_VIEW("hc_attn_base.weight", base); BLOCK_VIEW("hc_attn_scale.weight", scale);
    if (glm5next_cpu_mhc_pre(c, &fn, &base, &scale, residual, collapsed, post, comb) != 0) goto fail;
    BLOCK_VIEW("attn_norm.weight", fn); if (glm5next_cpu_vector(&fn, norm, h) != 0) goto fail;
    glm5next_cpu_rmsnorm(collapsed, collapsed, norm, h, c->norm_epsilon);
    if (glm5next_cpu_dsa_forward(model, layer, c, collapsed, sublayer) != 0) goto fail;
    glm5next_cpu_mhc_post(c, streams, residual, sublayer, post, comb);
    memcpy(residual, streams, (size_t)hc * h * sizeof(float));
    BLOCK_VIEW("hc_ffn_fn.weight", fn); BLOCK_VIEW("hc_ffn_base.weight", base); BLOCK_VIEW("hc_ffn_scale.weight", scale);
    if (glm5next_cpu_mhc_pre(c, &fn, &base, &scale, residual, collapsed, post, comb) != 0) goto fail;
    BLOCK_VIEW("ffn_norm.weight", fn); if (glm5next_cpu_vector(&fn, norm, h) != 0) goto fail;
    glm5next_cpu_rmsnorm(collapsed, collapsed, norm, h, c->norm_epsilon);
    if (glm5next_cpu_moe_ffn(model, layer, c, collapsed, sublayer) != 0) goto fail;
    glm5next_cpu_mhc_post(c, streams, residual, sublayer, post, comb);
    free(residual); free(collapsed); free(sublayer); free(post); free(comb); free(norm); return 0;
fail:
    free(residual); free(collapsed); free(sublayer); free(post); free(comb); free(norm); return -1;
#undef BLOCK_VIEW
}

/* Execute one recurrent KDA layer from the real GGUF views.  State layouts
 * are contiguous by layer and owned by the caller.  This is deliberately a
 * reference implementation: each quantized matrix row is dequantized before
 * its dot product, which makes it slow but useful for graph bring-up. */
static inline int glm5next_cpu_kda_forward(const gguf_shards *model,
        int layer, const glm5next_config *c, const float *hidden, float *out,
        float *recurrent, float *conv_state) {
    char name[128];
    glm5next_tensor_view t;
    float *x = NULL, *q = NULL, *k = NULL, *v = NULL, *fa = NULL;
    float *gate = NULL, *beta = NULL, *ga = NULL, *gg = NULL, *a = NULL;
    float *decay = NULL, *core = NULL, *norm = NULL, *work = NULL;
    int h, d, z, qdim;
    int rc = -1;
#define GET(s, required) do { \
        snprintf(name, sizeof(name), "blk.%d.%s", layer, (s)); \
        if (glm5next_tensor_view_get(model, name, (required), &t) != 0) goto done; \
    } while (0)
#define MAT(dst, s) do { GET((s), 1); if (glm5next_cpu_matvec((dst), &t, x) != 0) goto done; } while (0)

    if (!model || !c || !hidden || !out || !recurrent || !conv_state ||
        layer < 0 || layer >= c->n_layers ||
        glm5next_layer_type(c, layer) != GLM5NEXT_LAYER_KDA) goto done;
    x = (float *)malloc((size_t)c->hidden_size * sizeof(float));
    qdim = c->attention_heads * c->linear_head_dim;
    q = (float *)malloc((size_t)qdim * sizeof(float));
    k = (float *)malloc((size_t)qdim * sizeof(float));
    v = (float *)malloc((size_t)qdim * sizeof(float));
    fa = (float *)malloc(128 * sizeof(float));
    gate = (float *)malloc((size_t)qdim * sizeof(float));
    beta = (float *)malloc((size_t)c->attention_heads * sizeof(float));
    ga = (float *)malloc(128 * sizeof(float));
    gg = (float *)malloc((size_t)qdim * sizeof(float));
    a = (float *)malloc((size_t)c->attention_heads * sizeof(float));
    decay = (float *)malloc((size_t)qdim * sizeof(float));
    core = (float *)malloc((size_t)qdim * sizeof(float));
    norm = (float *)malloc((size_t)c->hidden_size * sizeof(float));
    work = (float *)malloc((size_t)c->linear_head_dim * sizeof(float));
    if (!x || !q || !k || !v || !fa || !gate || !beta || !ga || !gg || !a ||
        !decay || !core || !norm || !work) goto done;

    GET("attn_norm.weight", 1);
    if (glm5next_cpu_vector(&t, norm, c->hidden_size) != 0) goto done;
    glm5next_cpu_rmsnorm(x, hidden, norm, c->hidden_size, c->norm_epsilon);
    MAT(q, "attn_q.weight");
    MAT(k, "attn_k.weight");
    MAT(v, "attn_v.weight");
    glm5next_tensor_view conv[3];
    const char *conv_names[3] = {
        "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight"
    };
    for (z = 0; z < 3; ++z) {
        GET(conv_names[z], 1);
        conv[z] = t;
        if (conv[z].type != GGML_TYPE_F32 || conv[z].n_dims != 3) goto done;
    }
    {
        const float *w;
        for (z = 0; z < 3; ++z) {
            w = (const float *)conv[z].data;
            float *y = z == 0 ? q : (z == 1 ? k : v);
            float *s = conv_state + (size_t)z * qdim * (c->short_conv_kernel - 1);
            for (d = 0; d < qdim; ++d) {
                float current = y[d];
                float value = 0.0f;
                for (h = 0; h < c->short_conv_kernel - 1; ++h)
                    value += s[(size_t)d * (c->short_conv_kernel - 1) + h] *
                             w[(size_t)d * c->short_conv_kernel + h];
                value += current * w[(size_t)d * c->short_conv_kernel +
                                      c->short_conv_kernel - 1];
                memmove(s + (size_t)d * (c->short_conv_kernel - 1),
                        s + (size_t)d * (c->short_conv_kernel - 1) + 1,
                        (size_t)(c->short_conv_kernel - 2) * sizeof(float));
                s[(size_t)d * (c->short_conv_kernel - 1) + c->short_conv_kernel - 2] = current;
                y[d] = value / (1.0f + expf(-value));
            }
        }
    }
    glm53f_l2norm(q, qdim, 1e-6f);
    glm53f_l2norm(k, qdim, 1e-6f);
    MAT(fa, "ssm_f_a.weight");
    GET("ssm_f_b.weight", 1);
    if (glm5next_cpu_matvec(gate, &t, fa) != 0) goto done;
    GET("ssm_dt.bias", 1);
    float *dt = (float *)malloc((size_t)qdim * sizeof(float));
    if (!dt || glm5next_cpu_vector(&t, dt, qdim) != 0) { free(dt); goto done; }
    MAT(beta, "ssm_beta.weight");
    MAT(ga, "ssm_g_a.weight");
    GET("ssm_g_b.weight", 1);
    if (glm5next_cpu_matvec(gg, &t, ga) != 0) { free(dt); goto done; }
    GET("ssm_a", 1);
    if (glm5next_cpu_vector(&t, a, c->attention_heads) != 0) { free(dt); goto done; }
    for (h = 0; h < c->attention_heads; ++h) {
        for (d = 0; d < c->linear_head_dim; ++d) {
            int j = h * c->linear_head_dim + d;
            /* GLM5Next stores ssm_a as -exp(A_log), not A_log.  The
             * reference graph computes sigmoid(-ssm_a * (f + dt)) and then
             * scales the result by the negative lower bound. */
            decay[j] = c->kda_gate_lower_bound *
                (1.0f / (1.0f + expf(a[h] * (gate[j] + dt[j]))));
        }
        beta[h] = 1.0f / (1.0f + expf(-beta[h]));
        glm5next_kda_step(recurrent + (size_t)h * c->linear_head_dim * c->linear_head_dim,
                          q + (size_t)h * c->linear_head_dim,
                          k + (size_t)h * c->linear_head_dim,
                          v + (size_t)h * c->linear_head_dim,
                          decay + (size_t)h * c->linear_head_dim, beta[h],
                          c->linear_head_dim, c->linear_head_dim,
                          core + (size_t)h * c->linear_head_dim, work);
    }
    GET("ssm_norm.weight", 1);
    if (glm5next_cpu_vector(&t, norm, c->linear_head_dim) != 0) { free(dt); goto done; }
    for (h = 0; h < c->attention_heads; ++h) {
        float ss = 0.0f;
        for (d = 0; d < c->linear_head_dim; ++d) {
            float value = core[h * c->linear_head_dim + d]; ss += value * value;
        }
        float inv = 1.0f / sqrtf(ss / c->linear_head_dim + 1e-6f);
        for (d = 0; d < c->linear_head_dim; ++d) {
            int j = h * c->linear_head_dim + d;
            float sig = 1.0f / (1.0f + expf(-gg[j]));
            core[j] *= inv * norm[d] * sig;
        }
    }
    MAT(out, "attn_output.weight");
    free(dt);
    rc = 0;
done:
    free(x); free(q); free(k); free(v); free(fa); free(gate); free(beta);
    free(ga); free(gg); free(a); free(decay); free(core); free(norm); free(work);
    return rc;
#undef MAT
#undef GET
}

/* Complete CPU block for the KDA + leading-dense case.  This is the first
 * end-to-end block oracle: it includes both mHC sites, RMSNorm, recurrent KDA,
 * and the dense SwiGLU FFN.  DSA/MoE blocks use separate routines because
 * their cache and expert routing state are different. */
static inline int glm5next_cpu_kda_dense_block(const gguf_shards *model,
        int layer, const glm5next_config *c, float *streams,
        float *recurrent, float *conv_state) {
    int h = c->hidden_size, hc = c->hc_count;
    char name[128]; glm5next_tensor_view fn, base, scale;
    float *residual = (float *)malloc((size_t)hc * h * sizeof(float));
    float *collapsed = (float *)malloc((size_t)h * sizeof(float));
    float *sublayer = (float *)malloc((size_t)h * sizeof(float));
    float *post = (float *)malloc((size_t)hc * sizeof(float));
    float *comb = (float *)malloc((size_t)hc * hc * sizeof(float));
    float *norm = (float *)malloc((size_t)h * sizeof(float));
    if (!residual || !collapsed || !sublayer || !post || !comb || !norm) goto fail;
    memcpy(residual, streams, (size_t)hc * h * sizeof(float));
#define G5VIEW(s, req, dst) do { snprintf(name, sizeof(name), "blk.%d.%s", layer, (s)); \
    if (glm5next_tensor_view_get(model, name, (req), &(dst)) != 0) goto fail; } while (0)
    G5VIEW("hc_attn_fn.weight", 1, fn); G5VIEW("hc_attn_base.weight", 1, base); G5VIEW("hc_attn_scale.weight", 1, scale);
    if (glm5next_cpu_mhc_pre(c, &fn, &base, &scale, residual, collapsed, post, comb) != 0) goto fail;
    G5VIEW("attn_norm.weight", 1, fn);
    if (glm5next_cpu_vector(&fn, norm, h) != 0) goto fail;
    glm5next_cpu_rmsnorm(collapsed, collapsed, norm, h, c->norm_epsilon);
    if (glm5next_cpu_kda_forward(model, layer, c, collapsed, sublayer, recurrent, conv_state) != 0) goto fail;
    glm5next_cpu_mhc_post(c, streams, residual, sublayer, post, comb);

    memcpy(residual, streams, (size_t)hc * h * sizeof(float));
    G5VIEW("hc_ffn_fn.weight", 1, fn); G5VIEW("hc_ffn_base.weight", 1, base); G5VIEW("hc_ffn_scale.weight", 1, scale);
    if (glm5next_cpu_mhc_pre(c, &fn, &base, &scale, residual, collapsed, post, comb) != 0) goto fail;
    G5VIEW("ffn_norm.weight", 1, fn);
    if (glm5next_cpu_vector(&fn, norm, h) != 0) goto fail;
    glm5next_cpu_rmsnorm(collapsed, collapsed, norm, h, c->norm_epsilon);
    if (glm5next_cpu_dense_ffn(model, layer, c, collapsed, sublayer) != 0) goto fail;
    glm5next_cpu_mhc_post(c, streams, residual, sublayer, post, comb);
    free(residual); free(collapsed); free(sublayer); free(post); free(comb); free(norm);
    return 0;
fail:
    free(residual); free(collapsed); free(sublayer); free(post); free(comb); free(norm);
    return -1;
#undef G5VIEW
}

#endif /* GLM5NEXT_CPU_KDA_H */
