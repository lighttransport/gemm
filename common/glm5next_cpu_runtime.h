/* Reusable single-token GLM5Next CPU runtime.  DSA retains latent and pooled
 * indexer state, while KDA retains recurrent and convolution state across
 * calls.  This is also the correctness oracle for the HIP runner. */
#ifndef GLM5NEXT_CPU_RUNTIME_H
#define GLM5NEXT_CPU_RUNTIME_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "glm5next_cpu_kda.h"

typedef int (*glm5next_nextn_fusion_callback)(const gguf_shards *model,
        const glm5next_config *config, const float *embedding_norm,
        const float *hidden_norm, float *out, void *opaque);

typedef struct {
    const gguf_shards *model;
    glm5next_config config;
    float *streams, *recurrent, *conv, *latent_kv, *hidden, *target_hidden, *normed, *logits;
    float *nextn_latent_kv, *nextn_hidden, *nextn_fusion;
    int max_seq_len;
    int position;
    int target_position;
    glm5next_dsa_callback dsa_callback;
    void *dsa_callback_opaque;
    glm5next_kda_callback kda_callback;
    void *kda_callback_opaque;
    glm5next_moe_callback moe_callback;
    void *moe_callback_opaque;
    glm5next_mhc_callback mhc_callback;
    void *mhc_callback_opaque;
    glm5next_output_callback output_callback;
    void *output_callback_opaque;
    glm5next_output_callback nextn_output_callback;
    void *nextn_output_callback_opaque;
    glm5next_nextn_fusion_callback nextn_fusion_callback;
    void *nextn_fusion_callback_opaque;
    float *indexer_keys;
    float *indexer_gates;
} glm5next_cpu_runtime;

static inline void glm5next_cpu_runtime_set_dsa_callback(glm5next_cpu_runtime *r,
        glm5next_dsa_callback callback, void *opaque) {
    if (!r) return;
    r->dsa_callback = callback;
    r->dsa_callback_opaque = opaque;
}

static inline void glm5next_cpu_runtime_set_kda_callback(glm5next_cpu_runtime *r,
        glm5next_kda_callback callback, void *opaque) {
    if (!r) return;
    r->kda_callback = callback;
    r->kda_callback_opaque = opaque;
}

static inline void glm5next_cpu_runtime_set_moe_callback(glm5next_cpu_runtime *r,
        glm5next_moe_callback callback, void *opaque) {
    if (!r) return;
    r->moe_callback = callback;
    r->moe_callback_opaque = opaque;
}

static inline void glm5next_cpu_runtime_set_mhc_callback(glm5next_cpu_runtime *r,
        glm5next_mhc_callback callback, void *opaque) {
    if (!r) return;
    r->mhc_callback = callback;
    r->mhc_callback_opaque = opaque;
}

static inline void glm5next_cpu_runtime_set_output_callback(glm5next_cpu_runtime *r,
        glm5next_output_callback callback, void *opaque) {
    if (!r) return;
    r->output_callback = callback;
    r->output_callback_opaque = opaque;
}

static inline void glm5next_cpu_runtime_set_nextn_output_callback(glm5next_cpu_runtime *r,
        glm5next_output_callback callback, void *opaque) {
    if (!r) return;
    r->nextn_output_callback = callback;
    r->nextn_output_callback_opaque = opaque;
}

static inline void glm5next_cpu_runtime_set_nextn_fusion_callback(glm5next_cpu_runtime *r,
        glm5next_nextn_fusion_callback callback, void *opaque) {
    if (!r) return;
    r->nextn_fusion_callback = callback;
    r->nextn_fusion_callback_opaque = opaque;
}

static inline void glm5next_cpu_runtime_free(glm5next_cpu_runtime *r) {
    if (!r) return;
    free(r->streams); free(r->recurrent); free(r->conv); free(r->latent_kv);
    free(r->indexer_keys); free(r->indexer_gates);
    free(r->hidden); free(r->normed); free(r->logits);
    free(r->target_hidden); free(r->nextn_latent_kv); free(r->nextn_hidden); free(r->nextn_fusion);
    glm5next_config_free(&r->config); memset(r, 0, sizeof(*r));
}

static inline int glm5next_cpu_runtime_init(glm5next_cpu_runtime *r,
        const gguf_shards *model, int max_seq_len, char *error, size_t error_cap) {
    if (!r || !model) return -1;
    memset(r, 0, sizeof(*r));
    if (glm5next_config_load(model->metadata, &r->config, error, error_cap) != 0 ||
        glm5next_validate_tensors(model, &r->config, error, error_cap) != 0) {
        glm5next_cpu_runtime_free(r); return -1;
    }
    int h = r->config.hidden_size, hc = r->config.hc_count, d = r->config.linear_head_dim;
    if (max_seq_len < 1 || max_seq_len > r->config.context_length) max_seq_len = 256;
    size_t rn = (size_t)r->config.n_layers * r->config.attention_heads * d * d;
    size_t cn = (size_t)r->config.n_layers * 3 * r->config.attention_heads * d *
                (r->config.short_conv_kernel - 1);
    r->streams = (float *)malloc((size_t)hc * h * sizeof(float));
    r->recurrent = (float *)calloc(rn, sizeof(float));
    r->conv = (float *)calloc(cn, sizeof(float));
    r->latent_kv = (float *)calloc((size_t)r->config.n_layers * max_seq_len *
                                   r->config.kv_lora_rank, sizeof(float));
    r->indexer_keys = (float *)calloc((size_t)r->config.n_layers * max_seq_len *
                                      r->config.indexer_key_length, sizeof(float));
    r->indexer_gates = (float *)calloc((size_t)r->config.n_layers * max_seq_len *
                                       r->config.indexer_key_length, sizeof(float));
    r->hidden = (float *)malloc((size_t)h * sizeof(float));
    r->normed = (float *)malloc((size_t)h * sizeof(float));
    r->logits = (float *)malloc((size_t)r->config.vocab_size * sizeof(float));
    r->target_hidden = (float *)malloc((size_t)h * sizeof(float));
    r->nextn_latent_kv = r->config.n_nextn_layers > 0
        ? (float *)calloc((size_t)r->config.n_nextn_layers * max_seq_len *
                          r->config.kv_lora_rank, sizeof(float)) : NULL;
    r->nextn_hidden = r->config.n_nextn_layers > 0
        ? (float *)malloc((size_t)h * sizeof(float)) : NULL;
    r->nextn_fusion = r->config.n_nextn_layers > 0
        ? (float *)malloc((size_t)2 * h * sizeof(float)) : NULL;
    r->model = model; r->max_seq_len = max_seq_len; r->position = 0; r->target_position = -1;
    if (!r->streams || !r->recurrent || !r->conv || !r->latent_kv ||
        !r->indexer_keys || !r->indexer_gates || !r->hidden || !r->target_hidden ||
        !r->normed || !r->logits || (r->config.n_nextn_layers > 0 &&
        (!r->nextn_latent_kv || !r->nextn_hidden || !r->nextn_fusion))) {
        glm5next_cpu_runtime_free(r); return -1;
    }
    return 0;
}

static inline void glm5next_cpu_runtime_reset(glm5next_cpu_runtime *r) {
    if (!r) return;
    int h = r->config.hidden_size, hc = r->config.hc_count, d = r->config.linear_head_dim;
    size_t rn = (size_t)r->config.n_layers * r->config.attention_heads * d * d;
    size_t cn = (size_t)r->config.n_layers * 3 * r->config.attention_heads * d *
                (r->config.short_conv_kernel - 1);
    memset(r->streams, 0, (size_t)hc * h * sizeof(float));
    memset(r->recurrent, 0, rn * sizeof(float)); memset(r->conv, 0, cn * sizeof(float));
    memset(r->latent_kv, 0, (size_t)r->config.n_layers * r->max_seq_len *
           r->config.kv_lora_rank * sizeof(float));
    memset(r->indexer_keys, 0, (size_t)r->config.n_layers * r->max_seq_len *
           r->config.indexer_key_length * sizeof(float));
    memset(r->indexer_gates, 0, (size_t)r->config.n_layers * r->max_seq_len *
           r->config.indexer_key_length * sizeof(float));
    if (r->nextn_latent_kv)
        memset(r->nextn_latent_kv, 0, (size_t)r->config.n_nextn_layers * r->max_seq_len *
               r->config.kv_lora_rank * sizeof(float));
    r->position = 0;
    r->target_position = -1;
}

static inline int glm5next_cpu_runtime_step(glm5next_cpu_runtime *r, int token,
                                             int position) {
    if (!r || !r->model || token < 0 || token >= r->config.vocab_size ||
        position < 0 || position >= r->max_seq_len ||
        (position != 0 && position != r->position)) return -1;
    int h = r->config.hidden_size, hc = r->config.hc_count, d = r->config.linear_head_dim;
    const int profile = getenv("GLM5NEXT_PROFILE") && atoi(getenv("GLM5NEXT_PROFILE")) != 0;
    double profile_layers_ms = 0.0;
    struct timespec profile_t0, profile_t1;
    glm5next_tensor_view t;
    if (position == 0) glm5next_cpu_runtime_reset(r);
    if (glm5next_tensor_view_get(r->model, "token_embd.weight", 1, &t) != 0 ||
        t.n_dims != 2 || t.dims[0] != (uint64_t)h ||
        dequant_row(t.type, (const unsigned char *)t.data +
                    dequant_row_size(t.type, h) * (size_t)token, r->hidden, h) != 0)
    {
        fprintf(stderr, "glm5next: embedding lookup failed for token %d\n", token);
        return -1;
    }
    for (int s = 0; s < hc; ++s)
        memcpy(r->streams + (size_t)s * h, r->hidden, (size_t)h * sizeof(float));
    for (int l = 0; l < r->config.n_layers; ++l) {
        if (profile) clock_gettime(CLOCK_MONOTONIC, &profile_t0);
        float *rs = r->recurrent + (size_t)l * r->config.attention_heads * d * d;
        float *cs = r->conv + (size_t)l * 3 * r->config.attention_heads * d *
                    (r->config.short_conv_kernel - 1);
        int rc;
        if (glm5next_layer_type(&r->config, l) == GLM5NEXT_LAYER_KDA) {
            rc = l < r->config.first_k_dense_replace
                ? glm5next_cpu_kda_dense_block_cb(r->model, l, &r->config, r->streams, rs, cs,
                    r->kda_callback, r->moe_callback, r->kda_callback_opaque,
                    r->moe_callback_opaque, r->mhc_callback, r->mhc_callback_opaque)
                : glm5next_cpu_kda_moe_block_cb(r->model, l, &r->config, r->streams, rs, cs,
                    r->kda_callback, r->moe_callback, r->kda_callback_opaque,
                    r->moe_callback_opaque, r->mhc_callback, r->mhc_callback_opaque);
        } else {
            float *cache = r->latent_kv + (size_t)l * r->max_seq_len * r->config.kv_lora_rank;
            rc = glm5next_cpu_dsa_moe_block_cached_cb(r->model, l, &r->config, r->streams,
                cache, r->max_seq_len, position, r->dsa_callback,
                r->moe_callback, r->dsa_callback_opaque, r->moe_callback_opaque,
                r->indexer_keys + (size_t)l * r->max_seq_len * r->config.indexer_key_length,
                r->indexer_gates + (size_t)l * r->max_seq_len * r->config.indexer_key_length,
                r->mhc_callback, r->mhc_callback_opaque);
        }
        if (rc != 0) {
            fprintf(stderr, "glm5next: layer %d (%s) failed at position %d\n", l,
                    glm5next_layer_type(&r->config, l) == GLM5NEXT_LAYER_KDA ? "KDA" : "DSA",
                    position);
            return -1;
        }
        if (profile) {
            clock_gettime(CLOCK_MONOTONIC, &profile_t1);
            double ms = (double)(profile_t1.tv_sec - profile_t0.tv_sec) * 1000.0 +
                        (double)(profile_t1.tv_nsec - profile_t0.tv_nsec) / 1000000.0;
            profile_layers_ms += ms;
            fprintf(stderr, "glm5next profile: pos=%d layer=%d kind=%s %.3f ms\n", position, l,
                    glm5next_layer_type(&r->config, l) == GLM5NEXT_LAYER_KDA ? "KDA" : "DSA", ms);
        }
    }
    for (int i = 0; i < h; ++i) {
        double sum = 0.0; for (int s = 0; s < hc; ++s) sum += r->streams[(size_t)s * h + i];
        r->hidden[i] = (float)(sum / hc);
    }
    memcpy(r->target_hidden, r->hidden, (size_t)h * sizeof(float));
    r->target_position = position;
    if (r->output_callback) {
        if (profile) clock_gettime(CLOCK_MONOTONIC, &profile_t0);
        if (r->output_callback(r->model, &r->config, r->hidden, r->logits,
                               r->output_callback_opaque) != 0) return -1;
        if (profile) {
            clock_gettime(CLOCK_MONOTONIC, &profile_t1);
            double ms = (double)(profile_t1.tv_sec - profile_t0.tv_sec) * 1000.0 +
                        (double)(profile_t1.tv_nsec - profile_t0.tv_nsec) / 1000000.0;
            fprintf(stderr, "glm5next profile: pos=%d output %.3f ms layers %.3f ms total %.3f ms\n",
                    position, ms, profile_layers_ms, profile_layers_ms + ms);
        }
    } else {
        if (glm5next_tensor_view_get(r->model, "output_norm.weight", 1, &t) != 0 ||
            glm5next_cpu_vector(&t, r->normed, h) != 0) return -1;
        glm5next_cpu_rmsnorm(r->hidden, r->hidden, r->normed, h, r->config.norm_epsilon);
        if (glm5next_tensor_view_get(r->model, "output.weight", 1, &t) != 0 ||
            glm5next_cpu_matvec(r->logits, &t, r->hidden) != 0) return -1;
    }
    r->position = position + 1; return 0;
}

/* Execute one GLM5Next NextN/MTP block from the trunk's un-normalized final
 * hidden state.  NextN is a plain residual block: embedding/hidden fusion,
 * absorbed MLA attention (without the trunk indexer), MoE FFN, then the
 * shared draft head.  This mirrors the GLM-DSA MTP graph and keeps its KV
 * state separate from the trunk DSA caches. */
static inline float *glm5next_cpu_runtime_nextn_logits(glm5next_cpu_runtime *r,
                                                       int prev_token, int position) {
    if (!r || !r->model || r->config.n_nextn_layers <= 0 || !r->target_hidden ||
        !r->nextn_latent_kv || !r->nextn_hidden || !r->nextn_fusion ||
        prev_token < 0 || prev_token >= r->config.vocab_size || position < 0 ||
        position >= r->max_seq_len || r->target_position < 0 ||
        position != r->target_position + 1) return NULL;
    int h = r->config.hidden_size, layer = r->config.n_layers;
    float *embedding = (float *)malloc((size_t)h * sizeof(float));
    float *enorm = (float *)malloc((size_t)h * sizeof(float));
    float *hnorm = (float *)malloc((size_t)h * sizeof(float));
    float *x = (float *)malloc((size_t)h * sizeof(float));
    float *norm = (float *)malloc((size_t)h * sizeof(float));
    float *attn = (float *)malloc((size_t)h * sizeof(float));
    float *ffn_norm = (float *)malloc((size_t)h * sizeof(float));
    float *ffn = (float *)malloc((size_t)h * sizeof(float));
    float *head_norm = (float *)malloc((size_t)h * sizeof(float));
    glm5next_tensor_view t;
    char name[128];
    int rc = -1;
    if (!embedding || !enorm || !hnorm || !x || !norm || !attn || !ffn_norm ||
        !ffn || !head_norm) goto done;
    if (glm5next_tensor_view_get(r->model, "token_embd.weight", 1, &t) != 0 ||
        t.n_dims != 2 || t.dims[0] != (uint64_t)h ||
        dequant_row(t.type, (const unsigned char *)t.data +
                    dequant_row_size(t.type, h) * (size_t)prev_token, embedding, h) != 0)
        goto done;
#define NEXTN_GET(s) do { snprintf(name, sizeof(name), "blk.%d.%s", layer, (s)); \
    if (glm5next_tensor_view_get(r->model, name, 1, &t) != 0) goto done; } while (0)
    NEXTN_GET("nextn.enorm.weight");
    if (glm5next_cpu_vector(&t, norm, h) != 0) goto done;
    glm5next_cpu_rmsnorm(enorm, embedding, norm, h, r->config.norm_epsilon);
    NEXTN_GET("nextn.hnorm.weight");
    if (glm5next_cpu_vector(&t, norm, h) != 0) goto done;
    glm5next_cpu_rmsnorm(hnorm, r->target_hidden, norm, h, r->config.norm_epsilon);
    memcpy(r->nextn_fusion, enorm, (size_t)h * sizeof(float));
    memcpy(r->nextn_fusion + h, hnorm, (size_t)h * sizeof(float));
    if (r->nextn_fusion_callback) {
        if (r->nextn_fusion_callback(r->model, &r->config, enorm, hnorm, x,
                                     r->nextn_fusion_callback_opaque) != 0) goto done;
    } else {
        NEXTN_GET("nextn.eh_proj.weight");
        if (glm5next_cpu_matvec(x, &t, r->nextn_fusion) != 0) goto done;
    }
    NEXTN_GET("attn_norm.weight");
    if (glm5next_cpu_vector(&t, norm, h) != 0) goto done;
    glm5next_cpu_rmsnorm(ffn_norm, x, norm, h, r->config.norm_epsilon);
    if ((r->dsa_callback
            ? r->dsa_callback(r->model, layer, &r->config, ffn_norm, attn,
                              r->nextn_latent_kv, NULL, NULL, r->max_seq_len,
                              position, r->dsa_callback_opaque)
            : glm5next_cpu_dsa_forward_cached(r->model, layer, &r->config, ffn_norm, attn,
                              r->nextn_latent_kv, r->max_seq_len, position)) != 0) goto done;
    for (int i = 0; i < h; ++i) x[i] += attn[i];
    NEXTN_GET("ffn_norm.weight");
    if (glm5next_cpu_vector(&t, norm, h) != 0) goto done;
    glm5next_cpu_rmsnorm(ffn_norm, x, norm, h, r->config.norm_epsilon);
    if ((r->moe_callback
            ? r->moe_callback(r->model, layer, &r->config, ffn_norm, ffn,
                               r->moe_callback_opaque)
            : glm5next_cpu_moe_ffn(r->model, layer, &r->config, ffn_norm, ffn)) != 0) goto done;
    for (int i = 0; i < h; ++i) x[i] += ffn[i];
    memcpy(r->nextn_hidden, x, (size_t)h * sizeof(float));
    NEXTN_GET("nextn.shared_head_norm.weight");
    if (glm5next_cpu_vector(&t, norm, h) != 0) goto done;
    glm5next_cpu_rmsnorm(head_norm, x, norm, h, r->config.norm_epsilon);
    if (r->nextn_output_callback) {
        memcpy(r->nextn_hidden, x, (size_t)h * sizeof(float));
        if (r->nextn_output_callback(r->model, &r->config, r->nextn_hidden,
                                     r->logits, r->nextn_output_callback_opaque) != 0)
            goto done;
    } else if (glm5next_tensor_view_get(r->model, "output.weight", 1, &t) != 0 ||
               glm5next_cpu_matvec(r->logits, &t, head_norm) != 0) goto done;
    rc = 0;
done:
    free(embedding); free(enorm); free(hnorm); free(x); free(norm); free(attn);
    free(ffn_norm); free(ffn); free(head_norm);
    return rc == 0 ? r->logits : NULL;
#undef NEXTN_GET
}

#endif /* GLM5NEXT_CPU_RUNTIME_H */
