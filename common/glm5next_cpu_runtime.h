/* Reusable single-token GLM5Next CPU runtime.  DSA currently has a one-cell
 * attention implementation; KDA recurrent and convolution state is retained
 * across calls.  This is also the correctness oracle for the HIP runner. */
#ifndef GLM5NEXT_CPU_RUNTIME_H
#define GLM5NEXT_CPU_RUNTIME_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "glm5next_cpu_kda.h"

typedef struct {
    const gguf_shards *model;
    glm5next_config config;
    float *streams, *recurrent, *conv, *latent_kv, *hidden, *normed, *logits;
    int max_seq_len;
    int position;
} glm5next_cpu_runtime;

static inline void glm5next_cpu_runtime_free(glm5next_cpu_runtime *r) {
    if (!r) return;
    free(r->streams); free(r->recurrent); free(r->conv); free(r->latent_kv);
    free(r->hidden); free(r->normed); free(r->logits);
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
    r->hidden = (float *)malloc((size_t)h * sizeof(float));
    r->normed = (float *)malloc((size_t)h * sizeof(float));
    r->logits = (float *)malloc((size_t)r->config.vocab_size * sizeof(float));
    r->model = model; r->max_seq_len = max_seq_len; r->position = 0;
    if (!r->streams || !r->recurrent || !r->conv || !r->latent_kv || !r->hidden || !r->normed || !r->logits) {
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
    r->position = 0;
}

static inline int glm5next_cpu_runtime_step(glm5next_cpu_runtime *r, int token,
                                             int position) {
    if (!r || !r->model || token < 0 || token >= r->config.vocab_size) return -1;
    int h = r->config.hidden_size, hc = r->config.hc_count, d = r->config.linear_head_dim;
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
        float *rs = r->recurrent + (size_t)l * r->config.attention_heads * d * d;
        float *cs = r->conv + (size_t)l * 3 * r->config.attention_heads * d *
                    (r->config.short_conv_kernel - 1);
        int rc = glm5next_layer_type(&r->config, l) == GLM5NEXT_LAYER_KDA
            ? (l < r->config.first_k_dense_replace
                ? glm5next_cpu_kda_dense_block(r->model, l, &r->config, r->streams, rs, cs)
                : glm5next_cpu_kda_moe_block(r->model, l, &r->config, r->streams, rs, cs))
            : glm5next_cpu_dsa_moe_block_cached(r->model, l, &r->config, r->streams,
                r->latent_kv + (size_t)l * r->max_seq_len * r->config.kv_lora_rank,
                r->max_seq_len, position);
        if (rc != 0) {
            fprintf(stderr, "glm5next: layer %d (%s) failed at position %d\n", l,
                    glm5next_layer_type(&r->config, l) == GLM5NEXT_LAYER_KDA ? "KDA" : "DSA",
                    position);
            return -1;
        }
    }
    for (int i = 0; i < h; ++i) {
        double sum = 0.0; for (int s = 0; s < hc; ++s) sum += r->streams[(size_t)s * h + i];
        r->hidden[i] = (float)(sum / hc);
    }
    if (glm5next_tensor_view_get(r->model, "output_norm.weight", 1, &t) != 0 ||
        glm5next_cpu_vector(&t, r->normed, h) != 0) return -1;
    glm5next_cpu_rmsnorm(r->hidden, r->hidden, r->normed, h, r->config.norm_epsilon);
    if (glm5next_tensor_view_get(r->model, "output.weight", 1, &t) != 0 ||
        glm5next_cpu_matvec(r->logits, &t, r->hidden) != 0) return -1;
    r->position = position + 1; return 0;
}

#endif /* GLM5NEXT_CPU_RUNTIME_H */
