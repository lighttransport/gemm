/*
 * glm5next.h - GLM-5-Next / GLM-5.3-Flash model contract.
 *
 * This header deliberately contains no backend code.  It is the shared
 * architecture contract used by the CPU reference and the HIP runner.  The
 * model is not a Qwen hybrid model: attention.head_count_kv is a per-layer
 * schedule where zero denotes KDA and a non-zero value denotes DSA/MLA.
 */
#ifndef GLM5NEXT_H
#define GLM5NEXT_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "gguf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GLM5NEXT_LAYER_KDA = 0,
    GLM5NEXT_LAYER_DSA = 1
} glm5next_layer_kind;

typedef struct {
    int n_layers;                 /* decoder layers, excluding optional NextN */
    int n_layers_all;             /* decoder + optional NextN layers */
    int n_nextn_layers;
    int hidden_size;
    int vocab_size;
    int dense_feed_forward_length;
    int context_length;
    int attention_heads;
    int kv_heads_default;
    int q_lora_rank;
    int kv_lora_rank;
    int qk_nope_head_dim;
    int qk_rope_head_dim;
    int value_head_dim;
    int linear_head_dim;
    int indexer_heads;
    int indexer_key_length;
    int indexer_top_k;
    int indexer_kpool;
    int expert_count;
    int expert_used_count;
    int expert_group_count;
    int expert_group_used_count;
    int expert_ff_length;
    int shared_expert_count;
    int shared_expert_ff_length;
    int expert_weights_norm;
    int expert_gating_func;
    int first_k_dense_replace;
    int hc_count;
    int hc_sinkhorn_iterations;
    int short_conv_kernel;
    float hc_sinkhorn_epsilon;
    float norm_epsilon;
    float kda_gate_lower_bound;
    float routed_scaling_factor;
    int32_t *layer_kv_heads;       /* [n_layers], owned by this structure */
    float *swiglu_clamp_exp;       /* [n_layers_all], optional */
    float *swiglu_clamp_shexp;     /* [n_layers_all], optional */
} glm5next_config;

typedef struct {
    size_t conv_bytes;
    size_t recurrent_bytes;
    size_t latent_kv_bytes;
    size_t indexer_bytes;
    size_t mhc_bytes;
} glm5next_state_layout;

typedef struct {
    const gguf_context *owner;
    int index;
    uint32_t type;
    uint32_t n_dims;
    uint64_t dims[4];
    void *data;
    size_t bytes;
} glm5next_tensor_view;

static inline void glm5next_config_free(glm5next_config *c) {
    if (!c) return;
    free(c->layer_kv_heads);
    free(c->swiglu_clamp_exp);
    free(c->swiglu_clamp_shexp);
    memset(c, 0, sizeof(*c));
}

static inline int glm5next_is_arch(const gguf_context *gguf) {
    int i;
    if (!gguf) return 0;
    i = gguf_find_key(gguf, "general.architecture");
    return i >= 0 && gguf->kv[i].type == GGUF_TYPE_STRING &&
           strcmp(gguf->kv[i].value.str.str, "glm5next") == 0;
}

static inline int glm5next_get_int(const gguf_context *gguf, const char *key, int def) {
    int i = gguf_find_key(gguf, key);
    if (i < 0) return def;
    switch (gguf->kv[i].type) {
    case GGUF_TYPE_UINT32: return (int)gguf->kv[i].value.u32;
    case GGUF_TYPE_INT32:  return gguf->kv[i].value.i32;
    case GGUF_TYPE_UINT64: return (int)gguf->kv[i].value.u64;
    case GGUF_TYPE_INT64:  return (int)gguf->kv[i].value.i64;
    case GGUF_TYPE_BOOL:   return gguf->kv[i].value.b ? 1 : 0;
    default: return def;
    }
}

static inline float glm5next_get_float(const gguf_context *gguf, const char *key, float def) {
    int i = gguf_find_key(gguf, key);
    if (i < 0) return def;
    if (gguf->kv[i].type == GGUF_TYPE_FLOAT32) return gguf->kv[i].value.f32;
    if (gguf->kv[i].type == GGUF_TYPE_FLOAT64) return (float)gguf->kv[i].value.f64;
    return def;
}

static inline int glm5next_get_layer_heads(const gguf_context *gguf,
                                           const char *key, int32_t *out, int n) {
    int i = gguf_find_key(gguf, key), j;
    if (i < 0 || gguf->kv[i].type != GGUF_TYPE_ARRAY ||
        (int)gguf->kv[i].value.arr.n < n) return -1;
    if (gguf->kv[i].value.arr.type == GGUF_TYPE_INT32) {
        const int32_t *v = (const int32_t *)gguf->kv[i].value.arr.data;
        for (j = 0; j < n; ++j) out[j] = v[j];
        return 0;
    }
    if (gguf->kv[i].value.arr.type == GGUF_TYPE_UINT32) {
        const uint32_t *v = (const uint32_t *)gguf->kv[i].value.arr.data;
        for (j = 0; j < n; ++j) out[j] = (int32_t)v[j];
        return 0;
    }
    return -1;
}

static inline int glm5next_get_layer_floats(const gguf_context *gguf,
                                            const char *key, float *out, int n) {
    int i = gguf_find_key(gguf, key);
    if (i < 0) return 0; /* optional */
    if (gguf->kv[i].type != GGUF_TYPE_ARRAY || (int)gguf->kv[i].value.arr.n < n)
        return -1;
    if (gguf->kv[i].value.arr.type == GGUF_TYPE_FLOAT32) {
        const float *v = (const float *)gguf->kv[i].value.arr.data;
        for (int j = 0; j < n; ++j) out[j] = v[j];
        return 0;
    }
    if (gguf->kv[i].value.arr.type == GGUF_TYPE_FLOAT64) {
        const double *v = (const double *)gguf->kv[i].value.arr.data;
        for (int j = 0; j < n; ++j) out[j] = (float)v[j];
        return 0;
    }
    return -1;
}

/* Load and validate the architecture contract.  A non-zero return means the
 * file is not GLM5Next or is missing a field needed by either backend. */
static inline int glm5next_config_load(const gguf_context *gguf,
                                       glm5next_config *c, char *err, size_t err_cap) {
    int i, kda = 0, dsa = 0;
    if (!c) return -1;
    memset(c, 0, sizeof(*c));
    if (err && err_cap) err[0] = '\0';
    if (!glm5next_is_arch(gguf)) goto bad_arch;

#define G5I(s, d) glm5next_get_int(gguf, "glm5next." s, (d))
#define G5F(s, d) glm5next_get_float(gguf, "glm5next." s, (d))
    c->n_layers_all       = G5I("block_count", 0);
    c->n_nextn_layers     = G5I("nextn_predict_layers", 0);
    c->n_layers           = c->n_layers_all - c->n_nextn_layers;
    c->hidden_size        = G5I("embedding_length", 0);
    c->vocab_size         = G5I("vocab_size", 0);
    c->dense_feed_forward_length = G5I("feed_forward_length", 0);
    c->context_length     = G5I("context_length", 0);
    c->attention_heads    = G5I("attention.head_count", 0);
    c->kv_heads_default   = G5I("attention.head_count_kv", 0);
    c->q_lora_rank        = G5I("attention.q_lora_rank", 0);
    c->kv_lora_rank       = G5I("attention.kv_lora_rank", 0);
    c->qk_nope_head_dim   = G5I("attention.key_length_mla", 0);
    c->qk_rope_head_dim   = G5I("rope.dimension_count", 0);
    c->value_head_dim     = G5I("attention.value_length_mla", 0);
    c->linear_head_dim    = G5I("kda.head_dim", 0);
    c->indexer_heads      = G5I("attention.indexer.head_count", 0);
    c->indexer_key_length = G5I("attention.indexer.key_length", 0);
    c->indexer_top_k      = G5I("attention.indexer.top_k", 0);
    c->indexer_kpool      = G5I("attention.indexer.kpool", 0);
    c->expert_count       = G5I("expert_count", 0);
    c->expert_used_count  = G5I("expert_used_count", 0);
    c->expert_group_count = G5I("expert_group_count", 1);
    c->expert_group_used_count = G5I("expert_group_used_count", 1);
    c->expert_ff_length   = G5I("expert_feed_forward_length", 0);
    c->shared_expert_count = G5I("expert_shared_count", 0);
    c->shared_expert_ff_length = G5I("expert_shared_feed_forward_length", 0);
    c->expert_weights_norm = G5I("expert_weights_norm", 1);
    c->expert_gating_func = G5I("expert_gating_func", 2);
    c->first_k_dense_replace = G5I("leading_dense_block_count", 0);
    c->hc_count            = G5I("hyper_connection.count", 0);
    c->hc_sinkhorn_iterations = G5I("hyper_connection.sinkhorn_iterations", 0);
    c->short_conv_kernel   = G5I("ssm.conv_kernel", 0);
    c->hc_sinkhorn_epsilon = G5F("hyper_connection.sinkhorn_epsilon", 0.0f);
    c->norm_epsilon        = G5F("attention.layer_norm_epsilon", 0.0f);
    c->kda_gate_lower_bound = G5F("kda.gate_lower_bound", 0.0f);
    c->routed_scaling_factor = G5F("expert_weights_scale", 1.0f);
#undef G5I
#undef G5F

    if (c->n_layers <= 0 || c->n_layers > c->n_layers_all || c->hidden_size <= 0 ||
        c->vocab_size <= 0 || c->dense_feed_forward_length <= 0 || c->attention_heads <= 0 || c->q_lora_rank <= 0 ||
        c->kv_lora_rank <= 0 || c->qk_nope_head_dim <= 0 || c->value_head_dim <= 0 ||
        c->linear_head_dim <= 0 || c->indexer_heads <= 0 || c->indexer_key_length <= 0 ||
        c->indexer_top_k <= 0 || c->indexer_kpool <= 0 ||
        c->indexer_top_k % c->indexer_kpool != 0 || c->expert_count <= 0 ||
        c->expert_used_count <= 0 || c->expert_used_count > c->expert_count ||
        c->expert_ff_length <= 0 || c->expert_weights_norm != 1 ||
        c->expert_gating_func != 2 || c->hc_count != 4 ||
        c->hc_sinkhorn_iterations <= 0 || c->short_conv_kernel <= 1 ||
        c->norm_epsilon <= 0.0f || c->kda_gate_lower_bound >= 0.0f)
        goto bad_values;

    c->layer_kv_heads = (int32_t *)calloc((size_t)c->n_layers, sizeof(int32_t));
    c->swiglu_clamp_exp = (float *)calloc((size_t)c->n_layers_all, sizeof(float));
    c->swiglu_clamp_shexp = (float *)calloc((size_t)c->n_layers_all, sizeof(float));
    if (!c->layer_kv_heads || !c->swiglu_clamp_exp || !c->swiglu_clamp_shexp) goto bad_values;
    if (glm5next_get_layer_heads(gguf, "glm5next.attention.head_count_kv",
                                 c->layer_kv_heads, c->n_layers) != 0)
        goto bad_values;
    if (glm5next_get_layer_floats(gguf, "glm5next.swiglu_clamp_exp",
                                  c->swiglu_clamp_exp, c->n_layers_all) != 0 ||
        glm5next_get_layer_floats(gguf, "glm5next.swiglu_clamp_shexp",
                                  c->swiglu_clamp_shexp, c->n_layers_all) != 0)
        goto bad_values;
    for (i = 0; i < c->n_layers; ++i) {
        if (c->layer_kv_heads[i] == 0) ++kda;
        else if (c->layer_kv_heads[i] > 0) ++dsa;
        else goto bad_values;
    }
    if (kda == 0 || dsa == 0) goto bad_values;
    return 0;

bad_arch:
    if (err && err_cap) snprintf(err, err_cap, "general.architecture is not glm5next");
    return -1;
bad_values:
    if (err && err_cap) snprintf(err, err_cap, "invalid or incomplete glm5next metadata");
    glm5next_config_free(c);
    return -1;
}

static inline glm5next_layer_kind glm5next_layer_type(const glm5next_config *c, int layer) {
    return c && layer >= 0 && layer < c->n_layers && c->layer_kv_heads[layer] == 0
        ? GLM5NEXT_LAYER_KDA : GLM5NEXT_LAYER_DSA;
}

static inline int glm5next_tensor_present(const gguf_shards *model, const char *name) {
    return model && name && gguf_shards_find_tensor(model, name, NULL, NULL) == 0;
}

/* Resolve a tensor without copying it.  In mmap mode data points into the
 * owning shard; in metadata-only mode it is NULL and only the shape/type are
 * available. */
static inline int glm5next_tensor_view_get(const gguf_shards *model, const char *name,
                                           int required, glm5next_tensor_view *out) {
    const gguf_context *owner = NULL;
    int index = -1;
    if (out) memset(out, 0, sizeof(*out));
    if (!model || !name || !out || gguf_shards_find_tensor(model, name, &owner, &index) != 0) {
        if (required) return -1;
        return 1;
    }
    out->owner = owner;
    out->index = index;
    out->type = owner->tensors[index].type;
    out->n_dims = owner->tensors[index].n_dims > 4 ? 4 : owner->tensors[index].n_dims;
    for (uint32_t i = 0; i < out->n_dims; ++i) out->dims[i] = owner->tensors[index].dims[i];
    out->data = gguf_tensor_data(owner, index);
    out->bytes = gguf_tensor_size(owner, index);
    return 0;
}

/* Validate the tensor-name contract without touching tensor payloads.  This is
 * intentionally separate from config_load so CPU/HIP loaders can fail before
 * allocating large activation or expert buffers. */
static inline int glm5next_validate_tensors(const gguf_shards *model,
                                            const glm5next_config *c,
                                            char *err, size_t err_cap) {
    char name[128];
    const char *common[] = { "token_embd.weight", "output_norm.weight", "output.weight" };
    int i, l;
    if (err && err_cap) err[0] = '\0';
    if (!model || !c) return -1;
    for (i = 0; i < (int)(sizeof(common) / sizeof(common[0])); ++i) {
        if (!glm5next_tensor_present(model, common[i])) {
            if (err && err_cap) snprintf(err, err_cap, "missing tensor %s", common[i]);
            return -1;
        }
    }
    for (l = 0; l < c->n_layers; ++l) {
        const char *attn[] = {
            "attn_norm.weight", "attn_output.weight", "ffn_norm.weight",
            "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
            "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight"
        };
        for (i = 0; i < (int)(sizeof(attn) / sizeof(attn[0])); ++i) {
            snprintf(name, sizeof(name), "blk.%d.%s", l, attn[i]);
            if (!glm5next_tensor_present(model, name)) {
                if (err && err_cap) snprintf(err, err_cap, "missing tensor %s", name);
                return -1;
            }
        }
        if (l < c->first_k_dense_replace) {
            const char *dense[] = { "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight" };
            for (i = 0; i < (int)(sizeof(dense) / sizeof(dense[0])); ++i) {
                snprintf(name, sizeof(name), "blk.%d.%s", l, dense[i]);
                if (!glm5next_tensor_present(model, name)) {
                    if (err && err_cap) snprintf(err, err_cap, "missing tensor %s", name);
                    return -1;
                }
            }
        } else {
            const char *moe[] = {
                "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                "ffn_down_shexp.weight", "exp_probs_b.bias"
            };
            for (i = 0; i < (int)(sizeof(moe) / sizeof(moe[0])); ++i) {
                snprintf(name, sizeof(name), "blk.%d.%s", l, moe[i]);
                if (!glm5next_tensor_present(model, name)) {
                    if (err && err_cap) snprintf(err, err_cap, "missing tensor %s", name);
                    return -1;
                }
            }
        }
        if (glm5next_layer_type(c, l) == GLM5NEXT_LAYER_KDA) {
            const char *kda[] = {
                "ssm_a", "ssm_beta.weight", "ssm_conv1d_q.weight",
                "ssm_conv1d_k.weight", "ssm_conv1d_v.weight", "ssm_dt.bias",
                "ssm_f_a.weight", "ssm_f_b.weight", "ssm_g_a.weight",
                "ssm_g_b.weight", "ssm_norm.weight"
            };
            for (i = 0; i < (int)(sizeof(kda) / sizeof(kda[0])); ++i) {
                snprintf(name, sizeof(name), "blk.%d.%s", l, kda[i]);
                if (!glm5next_tensor_present(model, name)) {
                    if (err && err_cap) snprintf(err, err_cap, "missing tensor %s", name);
                    return -1;
                }
            }
        } else {
            const char *dsa[] = {
                "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
                "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight",
                "attn_v_b.weight", "indexer.attn_k.weight", "indexer.attn_q_b.weight",
                "indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.proj.weight",
                "indexer_compressor_ape.weight", "indexer_compressor_gate.weight"
            };
            for (i = 0; i < (int)(sizeof(dsa) / sizeof(dsa[0])); ++i) {
                snprintf(name, sizeof(name), "blk.%d.%s", l, dsa[i]);
                if (!glm5next_tensor_present(model, name)) {
                    if (err && err_cap) snprintf(err, err_cap, "missing tensor %s", name);
                    return -1;
                }
            }
        }
    }
    /* NextN/MTP blocks are stored after the trunk.  The current single-token
     * runtime does not execute speculative NextN yet, but validating its
     * weights here prevents a split or damaged checkpoint from appearing
     * usable merely because trunk generation succeeds.  GLM5Next's NextN
     * block is a dense-attention MoE block (no KDA or mHC tensors). */
    for (l = c->n_layers; l < c->n_layers_all; ++l) {
        const char *base[] = { "attn_norm.weight", "attn_output.weight", "ffn_norm.weight",
                               "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                               "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                               "ffn_down_shexp.weight", "exp_probs_b.bias",
                               "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
                               "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight",
                               "attn_v_b.weight", "indexer.attn_k.weight", "indexer.attn_q_b.weight",
                               "indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.proj.weight",
                               "indexer_compressor_ape.weight", "indexer_compressor_gate.weight",
                               "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
                               "nextn.shared_head_norm.weight" };
        for (i = 0; i < (int)(sizeof(base) / sizeof(base[0])); ++i) {
            snprintf(name, sizeof(name), "blk.%d.%s", l, base[i]);
            if (!glm5next_tensor_present(model, name)) {
                if (err && err_cap) snprintf(err, err_cap, "missing NextN tensor %s", name);
                return -1;
            }
        }
    }
    return 0;
}

static inline int glm5next_state_layout_compute(const glm5next_config *c, int max_seq_len,
                                                glm5next_state_layout *out) {
    size_t n;
    if (!c || !out || max_seq_len <= 0) return -1;
    memset(out, 0, sizeof(*out));
    n = (size_t)c->n_layers;
    /* KDA q/k/v and recurrent state are laid out per attention head.  The
     * previous contract omitted this factor, under-reporting a 64-head model
     * by 64x even though the CPU runtime correctly allocated the full state. */
    out->conv_bytes = n * (size_t)(c->short_conv_kernel - 1) *
                      (size_t)(3 * c->attention_heads * c->linear_head_dim) * sizeof(float);
    out->recurrent_bytes = n * (size_t)c->attention_heads *
                           c->linear_head_dim * c->linear_head_dim * sizeof(float);
    out->latent_kv_bytes = n * (size_t)max_seq_len * (size_t)c->kv_lora_rank * sizeof(float);
    out->indexer_bytes = n * (size_t)max_seq_len * (size_t)c->indexer_key_length * sizeof(float);
    out->mhc_bytes = (size_t)c->hc_count * c->hidden_size * sizeof(float);
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* GLM5NEXT_H */
