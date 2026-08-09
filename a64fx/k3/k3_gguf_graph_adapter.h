/*
 * K3 GGUF -> native K3 graph schema.
 *
 * The IQ GGUF export has no architecture metadata and its names are not the
 * names used by the safetensor runner.  Keep this mapping in one place: the
 * staging/loader code and graph validation use the same contract.
 */
#ifndef K3_GGUF_GRAPH_ADAPTER_H
#define K3_GGUF_GRAPH_ADAPTER_H

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define K3_GGUF_LAYERS       93
#define K3_GGUF_HIDDEN       7168
#define K3_GGUF_HEADS        96
#define K3_GGUF_HEAD_DIM     128
#define K3_GGUF_EXPERTS      896
#define K3_GGUF_TOP_K       16
#define K3_GGUF_EXPERT_FF   3072
#define K3_GGUF_LATENT      3584
#define K3_GGUF_SHARED_FF   6144
#define K3_GGUF_DENSE_FF    33792

typedef enum {
    K3_GGUF_KDA = 0,
    K3_GGUF_MLA = 1
} k3_gguf_layer_kind;

static inline int k3_gguf_is_mla(int layer) {
    static const unsigned char mla[K3_GGUF_LAYERS] = {
        [3] = 1, [7] = 1, [11] = 1, [15] = 1, [19] = 1, [23] = 1,
        [27] = 1, [31] = 1, [35] = 1, [39] = 1, [43] = 1, [47] = 1,
        [51] = 1, [55] = 1, [59] = 1, [63] = 1, [67] = 1, [71] = 1,
        [75] = 1, [79] = 1, [83] = 1, [87] = 1, [91] = 1, [92] = 1
    };
    return layer >= 0 && layer < K3_GGUF_LAYERS && mla[layer];
}

static inline k3_gguf_layer_kind k3_gguf_kind(int layer) {
    return k3_gguf_is_mla(layer) ? K3_GGUF_MLA : K3_GGUF_KDA;
}

/* Return the GGUF tensor suffix for a native graph role. */
static inline const char *k3_gguf_role_suffix(int layer, const char *role) {
    if (!strcmp(role, "input_norm")) return "attn_norm.weight";
    if (!strcmp(role, "post_norm")) return "ffn_norm.weight";
    if (!strcmp(role, "attn_res_norm")) return "attn_res_score.weight";
    if (!strcmp(role, "mlp_res_norm")) return "ffn_res_score.weight";
    if (!strcmp(role, "router")) return "ffn_gate_inp.weight";
    if (!strcmp(role, "router_bias")) return "exp_probs_b.bias";
    if (!strcmp(role, "routed_down")) return "ffn_routed_down.weight";
    if (!strcmp(role, "routed_norm")) return "ffn_routed_norm.weight";
    if (!strcmp(role, "routed_up")) return "ffn_routed_up.weight";
    if (!strcmp(role, "shared_gate")) return "ffn_gate_shexp.weight";
    if (!strcmp(role, "shared_up")) return "ffn_up_shexp.weight";
    if (!strcmp(role, "shared_down")) return "ffn_down_shexp.weight";
    if (layer == 0) {
        if (!strcmp(role, "dense_gate")) return "ffn_gate.weight";
        if (!strcmp(role, "dense_up")) return "ffn_up.weight";
        if (!strcmp(role, "dense_down")) return "ffn_down.weight";
    }
    if (k3_gguf_is_mla(layer)) {
        if (!strcmp(role, "q_a_proj")) return "attn_q_a.weight";
        if (!strcmp(role, "q_a_norm")) return "attn_q_a_norm.weight";
        if (!strcmp(role, "q_b_proj")) return "attn_q_b.weight";
        if (!strcmp(role, "kv_a_proj")) return "attn_kv_a_mqa.weight";
        if (!strcmp(role, "kv_a_norm")) return "attn_kv_a_norm.weight";
        if (!strcmp(role, "kv_b_proj")) return "attn_k_b.weight";
        if (!strcmp(role, "mla_g_proj")) return "attn_gate.weight";
        if (!strcmp(role, "mla_o_proj")) return "attn_output.weight";
    } else {
        if (!strcmp(role, "q_proj")) return "attn_q.weight";
        if (!strcmp(role, "k_proj")) return "attn_k.weight";
        if (!strcmp(role, "v_proj")) return "attn_v.weight";
        if (!strcmp(role, "g_proj")) return "ssm_g.weight";
        if (!strcmp(role, "f_a_proj")) return "ssm_f_a.weight";
        if (!strcmp(role, "f_b_proj")) return "ssm_f_b.weight";
        if (!strcmp(role, "b_proj")) return "ssm_beta.weight";
        if (!strcmp(role, "q_conv")) return "ssm_conv1d_q.weight";
        if (!strcmp(role, "k_conv")) return "ssm_conv1d_k.weight";
        if (!strcmp(role, "v_conv")) return "ssm_conv1d_v.weight";
        if (!strcmp(role, "a_log")) return "ssm_a";
        if (!strcmp(role, "dt_bias")) return "ssm_dt.bias";
        if (!strcmp(role, "o_norm")) return "ssm_norm.weight";
        if (!strcmp(role, "o_proj")) return "attn_output.weight";
    }
    return NULL;
}

static inline int k3_gguf_role_name(char *dst, size_t cap, int layer,
                                    const char *role) {
    const char *suffix = k3_gguf_role_suffix(layer, role);
    if (!suffix || !dst || cap == 0) return -1;
    int n = snprintf(dst, cap, "blk.%d.%s", layer, suffix);
    return n < 0 || (size_t)n >= cap ? -1 : 0;
}

static inline int k3_gguf_expert_name(char *dst, size_t cap, int layer,
                                      const char *which) {
    if (!dst || cap == 0 || layer <= 0 || layer >= K3_GGUF_LAYERS) return -1;
    const char *suffix = NULL;
    if (!strcmp(which, "w1")) suffix = "ffn_up_exps.weight";
    else if (!strcmp(which, "w2")) suffix = "ffn_down_exps.weight";
    else if (!strcmp(which, "w3")) suffix = "ffn_gate_exps.weight";
    if (!suffix) return -1;
    int n = snprintf(dst, cap, "blk.%d.%s", layer, suffix);
    return n < 0 || (size_t)n >= cap ? -1 : 0;
}

/* A callback-based validator keeps this header independent of GGUF storage. */
typedef int (*k3_gguf_has_tensor_fn)(void *opaque, const char *name);

static inline int k3_gguf_validate_graph(void *opaque,
                                         k3_gguf_has_tensor_fn has,
                                         int first_layer, int last_layer,
                                         FILE *diag) {
    static const char *common[] = {
        "input_norm", "post_norm", "attn_res_norm", "mlp_res_norm",
        "router", "router_bias", "routed_down", "routed_norm", "routed_up",
        "shared_gate", "shared_up", "shared_down"
    };
    static const char *kda[] = {
        "q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "f_b_proj",
        "b_proj", "q_conv", "k_conv", "v_conv", "a_log", "dt_bias",
        "o_norm", "o_proj"
    };
    static const char *mla[] = {
        "q_a_proj", "q_a_norm", "q_b_proj", "kv_a_proj", "kv_a_norm",
        "kv_b_proj", "mla_g_proj", "mla_o_proj"
    };
    int missing = 0;
    if (!has || first_layer < 0 || last_layer > K3_GGUF_LAYERS ||
        first_layer >= last_layer) return -1;
    for (int l = first_layer; l < last_layer; ++l) {
        char name[128];
        for (size_t i = 0; i < sizeof common / sizeof common[0]; ++i) {
            if (l == 0 && i >= 4) continue;
            if (k3_gguf_role_name(name, sizeof name, l, common[i]) ||
                !has(opaque, name)) {
                ++missing;
                if (diag) fprintf(diag, "k3-gguf: missing %s layer=%d\n", common[i], l);
            }
        }
        if (l == 0) {
            const char *dense[] = {"dense_gate", "dense_up", "dense_down"};
            for (size_t i = 0; i < 3; ++i) {
                if (k3_gguf_role_name(name, sizeof name, l, dense[i]) ||
                    !has(opaque, name)) {
                    ++missing;
                    if (diag) fprintf(diag, "k3-gguf: missing %s layer=%d\n", dense[i], l);
                }
            }
        }
        const char *const *attn = k3_gguf_is_mla(l) ? mla : kda;
        size_t n_attn = k3_gguf_is_mla(l) ? sizeof mla / sizeof mla[0] : sizeof kda / sizeof kda[0];
        for (size_t i = 0; i < n_attn; ++i) {
            if (k3_gguf_role_name(name, sizeof name, l, attn[i]) ||
                !has(opaque, name)) {
                ++missing;
                if (diag) fprintf(diag, "k3-gguf: missing %s layer=%d\n", attn[i], l);
            }
        }
    }
    return missing;
}

#endif
