/* Q38FN BF16 tensor-parallel ownership rules. */
#ifndef Q38FN_TP_LAYOUT_H
#define Q38FN_TP_LAYOUT_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "q38fn_arch.h"

#ifndef Q38FN_TP_RANKS
#define Q38FN_TP_RANKS 12
#endif
enum { Q38FN_TP_MAX_SEGMENTS = 4,
       Q38FN_TP_LAYOUT_VERSION = 5 };

typedef enum {
    Q38FN_TP_SKIP = 0,
    Q38FN_TP_FULL,
    Q38FN_TP_AXIS0,
    Q38FN_TP_AXIS1,
    Q38FN_TP_EXPERT_GATE_UP,
    Q38FN_TP_NGRAM_OWNER,
    Q38FN_TP_DELTA_QKV
} q38fn_tp_kind;

typedef struct {
    uint64_t start;
    uint64_t count;
} q38fn_tp_range;

typedef struct {
    q38fn_tp_kind kind;
    int axis;
    int n_ranges;
    q38fn_tp_range range[Q38FN_TP_MAX_SEGMENTS];
} q38fn_tp_plan;

static inline void q38fn_tp_split(uint64_t total, int rank, int ranks,
                                  uint64_t *start, uint64_t *count)
{
    uint64_t base = total / (uint64_t)ranks;
    uint64_t rem = total % (uint64_t)ranks;
    *count = base + ((uint64_t)rank < rem);
    *start = (uint64_t)rank * base + ((uint64_t)rank < rem ? (uint64_t)rank : rem);
}

/* Split whole quantization blocks.  Decode matrices use 64-column Q8 SDOT
 * blocks, so assigning fractional blocks would force a BF16 fallback. */
static inline int q38fn_tp_split_blocks(uint64_t total, uint64_t block,
        int rank, int ranks, uint64_t *start, uint64_t *count)
{
    uint64_t block_start, block_count;
    if (!block || total % block) return -1;
    q38fn_tp_split(total / block, rank, ranks, &block_start, &block_count);
    *start = block_start * block;
    *count = block_count * block;
    return 0;
}

static inline int q38fn_tp_split_decode(uint64_t total, uint64_t block,
        int rank, int ranks, uint64_t *start, uint64_t *count)
{
    if (ranks == 4 || ranks == 6)
        return q38fn_tp_split_blocks(total, block, rank, ranks, start, count);
    q38fn_tp_split(total, rank, ranks, start, count);
    return 0;
}

static inline int q38fn_tp_ends_with(const char *name, const char *suffix)
{
    size_t a = strlen(name), b = strlen(suffix);
    return a >= b && !memcmp(name + a - b, suffix, b);
}

static inline int q38fn_tp_ngram_shard(const char *name)
{
    const char *p = strstr(name, "ngram_embedding.shard_");
    int shard = -1;
    return p && sscanf(p, "ngram_embedding.shard_%d.weight", &shard) == 1 ? shard : -1;
}

/* Return a storage plan for one checkpoint tensor.  AXIS0 stores a contiguous
 * row range. AXIS1 stores the same column range from every row. The packed
 * expert gate/up tensor is special: its [expert,2*intermediate,hidden] middle
 * axis is represented by two ranges with the same local intermediate IDs. */
static inline int q38fn_tp_make_plan(const char *name, const uint64_t *shape,
                                     int ndims, int rank, int ranks,
                                     q38fn_tp_plan *plan)
{
    uint64_t start = 0, count = 0;
    int shard;
    if (!name || !shape || !plan || rank < 0 || rank >= ranks ||
        ranks != Q38FN_TP_RANKS || Q38FN_HEADS % ranks ||
        Q38FN_LINEAR_VALUE_HEADS % ranks)
        return -1;
    memset(plan, 0, sizeof(*plan));
    plan->kind = Q38FN_TP_FULL;
    plan->axis = -1;

    /* The checkpoint is multimodal, but q38fn_runner is a text decoder.  The
     * vision tower sorts after all model.language_model tensors and must not
     * consume the last 1--2 GiB of a 32 GiB A64FX HBM node. */
    if (!strncmp(name, "model.visual.", 13)) {
        plan->kind = Q38FN_TP_SKIP;
        return 0;
    }

    shard = q38fn_tp_ngram_shard(name);
    if (shard >= 0) {
        plan->kind = q38fn_ngram_owner_for_ranks(shard, ranks) == rank ?
                     Q38FN_TP_NGRAM_OWNER : Q38FN_TP_SKIP;
        return 0;
    }

    if (q38fn_tp_ends_with(name, ".mlp.experts.gate_up_proj")) {
        if (ndims != 3 || shape[0] != Q38FN_EXPERTS ||
            shape[1] != 2 * Q38FN_EXPERT_INTERMEDIATE || shape[2] != Q38FN_HIDDEN)
            return -1;
        if (q38fn_tp_split_decode(Q38FN_EXPERT_INTERMEDIATE, 64, rank,
                                 ranks, &start, &count)) return -1;
        plan->kind = Q38FN_TP_EXPERT_GATE_UP; plan->axis = 1; plan->n_ranges = 2;
        plan->range[0] = (q38fn_tp_range){start, count};
        plan->range[1] = (q38fn_tp_range){Q38FN_EXPERT_INTERMEDIATE + start, count};
        return 0;
    }
    if (q38fn_tp_ends_with(name, ".mlp.experts.down_proj")) {
        if (ndims != 3 || shape[0] != Q38FN_EXPERTS ||
            shape[1] != Q38FN_HIDDEN || shape[2] != Q38FN_EXPERT_INTERMEDIATE)
            return -1;
        if (q38fn_tp_split_decode(shape[2], 64, rank, ranks, &start, &count))
            return -1;
        plan->kind = Q38FN_TP_AXIS1; plan->axis = 2; plan->n_ranges = 1;
        plan->range[0] = (q38fn_tp_range){start, count}; return 0;
    }

    if (q38fn_tp_ends_with(name, ".mlp.shared_expert.gate_proj.weight") ||
        q38fn_tp_ends_with(name, ".mlp.shared_expert.up_proj.weight")) {
        if (ndims != 2) return -1;
        if (q38fn_tp_split_decode(shape[0], 64, rank, ranks, &start, &count))
            return -1;
        plan->kind = Q38FN_TP_AXIS0; plan->axis = 0; plan->n_ranges = 1;
        plan->range[0] = (q38fn_tp_range){start, count}; return 0;
    }
    if (q38fn_tp_ends_with(name, ".mlp.shared_expert.down_proj.weight")) {
        if (ndims != 2) return -1;
        if (q38fn_tp_split_decode(shape[1], 64, rank, ranks, &start, &count))
            return -1;
        plan->kind = Q38FN_TP_AXIS1; plan->axis = 1; plan->n_ranges = 1;
        plan->range[0] = (q38fn_tp_range){start, count}; return 0;
    }

    if (q38fn_tp_ends_with(name, ".linear_attn.in_proj_z.weight")) {
        if (ndims != 2) return -1;
        q38fn_tp_split(shape[0], rank, ranks, &start, &count);
        plan->kind = Q38FN_TP_AXIS0; plan->axis = 0; plan->n_ranges = 1;
        plan->range[0] = (q38fn_tp_range){start, count}; return 0;
    }
    if (q38fn_tp_ends_with(name, ".linear_attn.out_proj.weight")) {
        if (ndims != 2) return -1;
        q38fn_tp_split(shape[1], rank, ranks, &start, &count);
        plan->kind = Q38FN_TP_AXIS1; plan->axis = 1; plan->n_ranges = 1;
        plan->range[0] = (q38fn_tp_range){start, count}; return 0;
    }
    if (q38fn_tp_ends_with(name, ".linear_attn.in_proj_qkv.weight")) {
        uint64_t values_per_rank = Q38FN_LINEAR_VALUE_HEADS / (uint64_t)ranks;
        uint64_t first_value = (uint64_t)rank * values_per_rank;
        uint64_t first_key = first_value / 3;
        uint64_t last_value = first_value + values_per_rank - 1;
        uint64_t last_key = last_value / 3;
        uint64_t key_rows = (last_key - first_key + 1) * Q38FN_LINEAR_HEAD_DIM;
        if (ndims != 2 || shape[0] != Q38FN_LINEAR_CONV_DIM ||
            shape[1] != Q38FN_HIDDEN) return -1;
        plan->kind = Q38FN_TP_DELTA_QKV; plan->axis = 0; plan->n_ranges = 3;
        plan->range[0] = (q38fn_tp_range){first_key * Q38FN_LINEAR_HEAD_DIM, key_rows};
        plan->range[1] = (q38fn_tp_range){Q38FN_LINEAR_KEY_DIM +
                                          first_key * Q38FN_LINEAR_HEAD_DIM, key_rows};
        plan->range[2] = (q38fn_tp_range){2 * Q38FN_LINEAR_KEY_DIM +
                                          first_value * Q38FN_LINEAR_HEAD_DIM,
                                          values_per_rank * Q38FN_LINEAR_HEAD_DIM};
        return 0;
    }

    if (q38fn_tp_ends_with(name, ".self_attn.q_proj.weight")) {
        if (ndims != 2) return -1;
        q38fn_tp_split(Q38FN_HEADS, rank, ranks, &start, &count);
        plan->kind = Q38FN_TP_AXIS0; plan->axis = 0; plan->n_ranges = 1;
        plan->range[0] = (q38fn_tp_range){start * 2 * Q38FN_HEAD_DIM,
                                          count * 2 * Q38FN_HEAD_DIM}; return 0;
    }
    if (q38fn_tp_ends_with(name, ".self_attn.o_proj.weight")) {
        if (ndims != 2) return -1;
        q38fn_tp_split(Q38FN_HEADS, rank, ranks, &start, &count);
        plan->kind = Q38FN_TP_AXIS1; plan->axis = 1; plan->n_ranges = 1;
        plan->range[0] = (q38fn_tp_range){start * Q38FN_HEAD_DIM,
                                          count * Q38FN_HEAD_DIM}; return 0;
    }

    if (strstr(name, ".layers.") &&
        q38fn_tp_ends_with(name, ".input_mix_weight_down.weight")) {
        if (ndims != 2) return -1;
        q38fn_tp_split(shape[0], rank, ranks, &start, &count);
        plan->kind = Q38FN_TP_AXIS0; plan->axis = 0; plan->n_ranges = 1;
        plan->range[0] = (q38fn_tp_range){start, count}; return 0;
    }
    if (strstr(name, ".layers.") &&
        q38fn_tp_ends_with(name, ".input_mix_weight_up.weight")) {
        if (ndims != 2) return -1;
        q38fn_tp_split(shape[1], rank, ranks, &start, &count);
        plan->kind = Q38FN_TP_AXIS1; plan->axis = 1; plan->n_ranges = 1;
        plan->range[0] = (q38fn_tp_range){start, count}; return 0;
    }
    /* The final mixer has rank 320 rather than rank 32 and remains sharded. */
    if (q38fn_tp_ends_with(name, ".input_mix_weight_down.weight")) {
        if (ndims != 2) return -1;
        q38fn_tp_split(shape[0], rank, ranks, &start, &count);
        plan->kind = Q38FN_TP_AXIS0; plan->axis = 0; plan->n_ranges = 1;
        plan->range[0] = (q38fn_tp_range){start, count}; return 0;
    }
    if (q38fn_tp_ends_with(name, ".input_mix_weight_up.weight")) {
        if (ndims != 2) return -1;
        q38fn_tp_split(shape[1], rank, ranks, &start, &count);
        plan->kind = Q38FN_TP_AXIS1; plan->axis = 1; plan->n_ranges = 1;
        plan->range[0] = (q38fn_tp_range){start, count}; return 0;
    }

    if (!strcmp(name, "model.language_model.embed_tokens.weight") ||
        !strcmp(name, "lm_head.weight")) {
        if (ndims != 2) return -1;
        q38fn_tp_split(shape[0], rank, ranks, &start, &count);
        plan->kind = Q38FN_TP_AXIS0; plan->axis = 0; plan->n_ranges = 1;
        plan->range[0] = (q38fn_tp_range){start, count}; return 0;
    }
    return 0;
}

#endif
