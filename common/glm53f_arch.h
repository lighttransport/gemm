/* GLM5.3 Flash architecture contract.
 * Keep this separate from glm5.h until the new forward graph is complete:
 * glm5.h describes the older GLM5.2 MLA/78-layer implementation.
 */
#ifndef GLM53F_ARCH_H
#define GLM53F_ARCH_H

#include <stddef.h>

typedef enum {
    GLM53F_LINEAR_ATTENTION = 0,
    GLM53F_SPARSE_ATTENTION = 1
} glm53f_layer_kind;

typedef struct {
    int n_layers;               /* 45 */
    int hidden_size;            /* 4096 */
    int vocab_size;             /* 154880 */
    int max_position_embeddings;/* 1048576 */
    int n_attention_heads;      /* 64 */
    int n_key_value_heads;      /* 64 */
    int linear_head_dim;        /* 128 */
    int index_head_dim;         /* 128 */
    int index_n_heads;          /* 32 */
    int index_topk;              /* 2048 */
    int q_lora_rank;             /* 1536 */
    int kv_lora_rank;            /* 512 */
    int qk_head_dim;             /* 256 */
    int v_head_dim;              /* 256 */
    int n_routed_experts;        /* 288 */
    int n_experts_per_tok;       /* 8 */
    int n_shared_experts;        /* 1 */
    int moe_intermediate_size;   /* 2048 */
    int first_k_dense_replace;   /* 3 */
    int mtp_layers;              /* 1 auxiliary layer at index 45 */
    int mhc_enabled;             /* 1: manifold-constrained hyper-connections */
    int hc_mult;                 /* 4 */
    int hc_sinkhorn_iters;       /* 20 */
    int short_conv_kernel_size;  /* 4 */
    float routed_scaling_factor; /* 2.5 */
} glm53f_arch;

static inline glm53f_arch glm53f_arch_default(void) {
    glm53f_arch a = {0};
    a.n_layers=45; a.hidden_size=4096; a.vocab_size=154880;
    a.max_position_embeddings=1048576; a.n_attention_heads=64;
    a.n_key_value_heads=64; a.linear_head_dim=128; a.index_head_dim=128;
    a.index_n_heads=32; a.index_topk=2048; a.q_lora_rank=1536;
    a.kv_lora_rank=512; a.qk_head_dim=256; a.v_head_dim=256;
    a.n_routed_experts=288; a.n_experts_per_tok=8; a.n_shared_experts=1;
    a.moe_intermediate_size=2048; a.first_k_dense_replace=3;
    a.mtp_layers=1; a.mhc_enabled=1; a.hc_mult=4; a.hc_sinkhorn_iters=20;
    a.short_conv_kernel_size=4;
    a.routed_scaling_factor=2.5f;
    return a;
}

static inline glm53f_layer_kind glm53f_layer_type(size_t layer) {
    /* text_config.layer_types: sparse attention every fourth layer starting at 3 */
    return layer < 45 && layer >= 3 && ((layer - 3) % 4) == 0
        ? GLM53F_SPARSE_ATTENTION : GLM53F_LINEAR_ATTENTION;
}

static inline int glm53f_is_moe(size_t layer) {
    /* The three dense decoder layers are followed by MoE layers, including
     * the auxiliary MTP layer at index 45. */
    return layer >= 3 && layer <= 45;
}

static inline int glm53f_is_mtp(size_t layer) { return layer == 45; }

/* Split each routed expert's intermediate dimension across `parts` ranks.
 * parts=4 on 12 A64FX nodes gives the best measured decode critical path.
 * The offsets keep every part on a distinct rank and balance 288 experts. */
static inline int glm53f_expert_part_owner(int expert, int part, int parts, int ranks) {
    if (expert < 0 || part < 0 || part >= parts || parts < 1 || ranks < parts || ranks % parts)
        return -1;
    return (expert % ranks + part * (ranks / parts)) % ranks;
}

static inline void glm53f_balanced_slice(int n, int part, int parts, int *begin, int *count) {
    int a = n * part / parts, b = n * (part + 1) / parts;
    if (begin) *begin = a;
    if (count) *count = b - a;
}

static inline int glm53f_block_aligned_slice(
        int n, int block, int part, int parts, int *begin, int *count) {
    if (n < 1 || block < 1 || n % block || parts < 1 || part < 0 || part >= parts)
        return -1;
    int blocks = n / block;
    int a = blocks * part / parts, b = blocks * (part + 1) / parts;
    if (a == b) return -1;
    if (begin) *begin = a * block;
    if (count) *count = (b - a) * block;
    return 0;
}

#endif
