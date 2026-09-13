/* Qwen3.8-Flash-Next (Qwen4-Exp) architecture constants used by the A64FX bring-up. */
#ifndef Q38FN_ARCH_H
#define Q38FN_ARCH_H

#include <stddef.h>
#include <stdint.h>

#define Q38FN_LAYERS 48
#define Q38FN_HIDDEN 2560
#define Q38FN_VOCAB 248320
#define Q38FN_CONTEXT 262144
#define Q38FN_HEADS 24
#define Q38FN_KV_HEADS 2
#define Q38FN_HEAD_DIM 256
#define Q38FN_EXPERTS 512
#define Q38FN_ACTIVE_EXPERTS 10
#define Q38FN_EXPERT_INTERMEDIATE 640
#define Q38FN_HC_COUNT 4
#define Q38FN_LINEAR_KEY_HEADS 16
#define Q38FN_LINEAR_VALUE_HEADS 48
#define Q38FN_LINEAR_HEAD_DIM 128
#define Q38FN_LINEAR_KEY_DIM (Q38FN_LINEAR_KEY_HEADS * Q38FN_LINEAR_HEAD_DIM)
#define Q38FN_LINEAR_VALUE_DIM (Q38FN_LINEAR_VALUE_HEADS * Q38FN_LINEAR_HEAD_DIM)
#define Q38FN_LINEAR_CONV_DIM (2 * Q38FN_LINEAR_KEY_DIM + Q38FN_LINEAR_VALUE_DIM)
#define Q38FN_LINEAR_CONV_KERNEL 4
#define Q38FN_PLE_LAYER 1       /* config is one-based; checkpoint layer index 1 */
#define Q38FN_NGRAM_SIZE 3
#define Q38FN_NGRAM_HEADS 16    /* 8 bigram + 8 trigram heads */
#define Q38FN_NGRAM_HEAD_DIM 160
#define Q38FN_NGRAM_ROWS 320001536
#define Q38FN_NGRAM_SHARDS 128
#define Q38FN_NGRAM_ROWS_PER_SHARD 2500012
#define Q38FN_NGRAM_ROW_BYTES (Q38FN_NGRAM_HEAD_DIM * 2)
#define Q38FN_EOS 248044

_Static_assert(Q38FN_NGRAM_SHARDS * Q38FN_NGRAM_ROWS_PER_SHARD ==
               Q38FN_NGRAM_ROWS, "n-gram shard geometry mismatch");
_Static_assert(Q38FN_NGRAM_ROW_BYTES % 64 == 0,
               "n-gram row must preserve cache-line transfer alignment");

static const int64_t q38fn_layer_multipliers[3] = {
    INT64_C(23703573157769), INT64_C(20109073645365), INT64_C(8052911324071)
};
static const uint64_t q38fn_head_vocab_sizes[Q38FN_NGRAM_HEADS] = {
    20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,
    20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171
};
static const uint64_t q38fn_head_offsets[Q38FN_NGRAM_HEADS] = {
    0,20000003,40000026,60000059,80000106,100000165,120000228,140000297,
    160000374,180000455,200000548,220000655,240000802,260000955,280001114,300001275
};

static inline int q38fn_layer_is_full_attention(size_t layer) {
    return layer < Q38FN_LAYERS && (layer % 4) == 3;
}
static inline int q38fn_layer_has_ple(size_t layer) { return layer == Q38FN_PLE_LAYER; }

/* Keep endpoint ranks below the HBM high-water mark: rank 0 also owns the
 * token embedding and rank 11 owns the final mixer and LM head. */
static inline int q38fn_ngram_owner_for_ranks(int shard, int ranks) {
    if (shard < 0 || shard >= Q38FN_NGRAM_SHARDS) return -1;
    if (ranks < 1 || ranks > Q38FN_NGRAM_SHARDS) return -1;
    if (ranks != 12)
        return (int)(((uint64_t)shard * (uint64_t)ranks) /
                     Q38FN_NGRAM_SHARDS);
    if (shard < 9) return 0;
    shard -= 9;
    if (shard < 110) return 1 + shard / 11;
    return 11;
}

static inline int q38fn_ngram_owner(int shard) {
    return q38fn_ngram_owner_for_ranks(shard, 12);
}

/* Hash the current token and its two preceding raw tokenizer IDs.  Unsigned
 * arithmetic intentionally provides the signed-int64 two's-complement wrap
 * used by the reference implementation.  An EOS in the look-back window
 * resets the segment, so callers should pass EOS for unavailable history. */
static inline void q38fn_ngram_rows(uint64_t current, uint64_t previous,
                                    uint64_t previous2, uint64_t rows[Q38FN_NGRAM_HEADS]) {
    uint64_t t[3] = {current, previous, previous2};
    uint64_t bigram = (t[0] * (uint64_t)q38fn_layer_multipliers[0]) ^
                      (t[1] * (uint64_t)q38fn_layer_multipliers[1]);
    uint64_t trigram = bigram ^ (t[2] * (uint64_t)q38fn_layer_multipliers[2]);
    for (int h = 0; h < 8; ++h) rows[h] = q38fn_head_offsets[h] + bigram % q38fn_head_vocab_sizes[h];
    for (int h = 8; h < 16; ++h) rows[h] = q38fn_head_offsets[h] + trigram % q38fn_head_vocab_sizes[h];
}

#endif
