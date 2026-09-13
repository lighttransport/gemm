/* Scalar GLM5Next operator and decode-state reference helpers.
 * These are backend-neutral oracles; HIP kernels must match them within the
 * configured numerical tolerance before optimized dispatch is enabled. */
#ifndef GLM5NEXT_REF_H
#define GLM5NEXT_REF_H

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "glm5next.h"
#include "glm53f_ref.h"

typedef struct {
    float *kda_conv;             /* [layers, 3*head_dim, kernel-1] */
    float *kda_recurrent;        /* [layers, head_dim, head_dim] */
    float *latent_kv;            /* [layers, context, kv_lora_rank] */
    float *indexer_keys;         /* [layers, context, indexer_key_length] */
    float *indexer_gates;        /* same logical cache as indexer_keys */
    int *selected;               /* [layers, context] */
    int *selected_count;         /* [layers] */
    float *mhc_streams;          /* [hc_count, hidden] */
    int max_seq_len;
} glm5next_decode_state;

static inline void glm5next_decode_state_free(glm5next_decode_state *s) {
    if (!s) return;
    free(s->kda_conv); free(s->kda_recurrent); free(s->latent_kv);
    free(s->indexer_keys); free(s->indexer_gates); free(s->selected);
    free(s->selected_count); free(s->mhc_streams);
    memset(s, 0, sizeof(*s));
}

static inline int glm5next_decode_state_alloc(const glm5next_config *c, int max_seq_len,
                                              glm5next_decode_state *s) {
    size_t layers, ctx, linear, index;
    if (!c || !s || max_seq_len <= 0) return -1;
    memset(s, 0, sizeof(*s));
    layers = (size_t)c->n_layers;
    ctx = (size_t)max_seq_len;
    linear = (size_t)3 * (size_t)c->linear_head_dim *
             (size_t)(c->short_conv_kernel - 1);
    index = (size_t)c->indexer_key_length;
    s->kda_conv = (float *)calloc(layers * linear, sizeof(float));
    s->kda_recurrent = (float *)calloc(layers * (size_t)c->linear_head_dim *
                                       (size_t)c->linear_head_dim, sizeof(float));
    s->latent_kv = (float *)calloc(layers * ctx * (size_t)c->kv_lora_rank, sizeof(float));
    s->indexer_keys = (float *)calloc(layers * ctx * index, sizeof(float));
    s->indexer_gates = (float *)calloc(layers * ctx * index, sizeof(float));
    s->selected = (int *)calloc(layers * ctx, sizeof(int));
    s->selected_count = (int *)calloc(layers, sizeof(int));
    s->mhc_streams = (float *)calloc((size_t)c->hc_count * c->hidden_size, sizeof(float));
    s->max_seq_len = max_seq_len;
    if (!s->kda_conv || !s->kda_recurrent || !s->latent_kv ||
        !s->indexer_keys || !s->indexer_gates || !s->selected ||
        !s->selected_count || !s->mhc_streams) {
        glm5next_decode_state_free(s);
        return -1;
    }
    return 0;
}

static inline void glm5next_decode_state_reset(const glm5next_config *c,
                                               glm5next_decode_state *s) {
    size_t layers, ctx;
    if (!c || !s) return;
    layers = (size_t)c->n_layers;
    ctx = (size_t)s->max_seq_len;
    memset(s->kda_conv, 0, layers * (size_t)3 * c->linear_head_dim *
           (size_t)(c->short_conv_kernel - 1) * sizeof(float));
    memset(s->kda_recurrent, 0, layers * (size_t)c->linear_head_dim *
           c->linear_head_dim * sizeof(float));
    memset(s->latent_kv, 0, layers * ctx * c->kv_lora_rank * sizeof(float));
    memset(s->indexer_keys, 0, layers * ctx * c->indexer_key_length * sizeof(float));
    memset(s->indexer_gates, 0, layers * ctx * c->indexer_key_length * sizeof(float));
    memset(s->selected, 0, layers * ctx * sizeof(int));
    memset(s->selected_count, 0, layers * sizeof(int));
    memset(s->mhc_streams, 0, (size_t)c->hc_count * c->hidden_size * sizeof(float));
}

static inline size_t glm5next_decode_state_bytes(const glm5next_config *c, int max_seq_len) {
    glm5next_state_layout l;
    return glm5next_state_layout_compute(c, max_seq_len, &l) == 0
        ? l.conv_bytes + l.recurrent_bytes + l.latent_kv_bytes +
          2 * l.indexer_bytes + l.mhc_bytes +
          (size_t)c->n_layers * (max_seq_len * sizeof(int) + sizeof(int)) : 0;
}

static inline void glm5next_kda_step(float *state, const float *q, const float *k,
                                     const float *v, const float *log_decay, float beta,
                                     int key_dim, int value_dim, float *out, float *work) {
    glm53f_kda_step_vec_streamed(state, q, k, v, log_decay, beta,
                                 key_dim, value_dim, out, work);
}

static inline void glm5next_kda_log_decay(float *out, const float *gate,
                                           const float *dt_bias, float a_log,
                                           float lower_bound, int n) {
    glm53f_kda_safe_log_decay(out, gate, dt_bias, a_log, lower_bound, n);
}

static inline int glm5next_index_select(float *pool_keys, int *selected,
                                        const float *query, const float *head_weight,
                                        const float *key_cache, const float *gate_cache,
                                        const float *ape, int tokens, int kpool,
                                        int index_top_k, int heads, int dim) {
    return glm53f_index_select_decode(pool_keys, selected, query, head_weight,
                                      key_cache, gate_cache, ape, tokens, kpool,
                                      index_top_k, heads, dim);
}

static inline void glm5next_mhc_sinkhorn(float *combine, int count, int iterations,
                                         float epsilon) {
    glm53f_mhc_sinkhorn(combine, count, iterations, epsilon);
}

#endif /* GLM5NEXT_REF_H */
