#include <stdio.h>
#include <string.h>
#include "../common/q38fn_tp_layout.h"

static int check_cover(const char *name, const uint64_t *shape, int ndims,
                       q38fn_tp_kind kind, uint64_t total)
{
    uint64_t covered = 0, next = 0;
    for (int rank = 0; rank < Q38FN_TP_RANKS; ++rank) {
        q38fn_tp_plan p;
        if (q38fn_tp_make_plan(name, shape, ndims, rank, Q38FN_TP_RANKS, &p) || p.kind != kind)
            return -1;
        if (p.n_ranges != 1 || p.range[0].start != next) return -1;
        next += p.range[0].count; covered += p.range[0].count;
    }
    return covered == total ? 0 : -1;
}

int main(void)
{
    uint64_t gu[3] = {Q38FN_EXPERTS, 2 * Q38FN_EXPERT_INTERMEDIATE, Q38FN_HIDDEN};
    uint64_t down[3] = {Q38FN_EXPERTS, Q38FN_HIDDEN, Q38FN_EXPERT_INTERMEDIATE};
    uint64_t q[2] = {2 * Q38FN_HEADS * Q38FN_HEAD_DIM, Q38FN_HIDDEN};
    uint64_t o[2] = {Q38FN_HIDDEN, Q38FN_HEADS * Q38FN_HEAD_DIM};
    uint64_t vocab[2] = {Q38FN_VOCAB, Q38FN_HIDDEN};
    uint64_t delta_qkv[2] = {Q38FN_LINEAR_CONV_DIM, Q38FN_HIDDEN};
    uint64_t hc_down[2] = {32, Q38FN_HC_COUNT * Q38FN_HIDDEN};
    uint64_t hc_up[2] = {Q38FN_HC_COUNT * Q38FN_HIDDEN, 32};
    uint64_t sum = 0;
    for (int rank = 0; rank < Q38FN_TP_RANKS; ++rank) {
        q38fn_tp_plan p;
        if (q38fn_tp_make_plan("model.language_model.layers.0.mlp.experts.gate_up_proj",
                               gu, 3, rank, Q38FN_TP_RANKS, &p) ||
            p.kind != Q38FN_TP_EXPERT_GATE_UP || p.n_ranges != 2 ||
            p.range[0].count != p.range[1].count ||
            p.range[1].start != Q38FN_EXPERT_INTERMEDIATE + p.range[0].start)
            return 1;
        sum += p.range[0].count;
    }
    if (sum != Q38FN_EXPERT_INTERMEDIATE ||
        check_cover("model.language_model.layers.0.mlp.experts.down_proj", down, 3,
                    Q38FN_TP_AXIS1, Q38FN_EXPERT_INTERMEDIATE) ||
        check_cover("model.language_model.layers.3.self_attn.q_proj.weight", q, 2,
                    Q38FN_TP_AXIS0, q[0]) ||
        check_cover("model.language_model.layers.3.self_attn.o_proj.weight", o, 2,
                    Q38FN_TP_AXIS1, o[1]) ||
        check_cover("lm_head.weight", vocab, 2, Q38FN_TP_AXIS0, Q38FN_VOCAB))
        return 1;
    for (int rank = 0; rank < Q38FN_TP_RANKS; ++rank) {
        q38fn_tp_plan p;
        if (q38fn_tp_make_plan("model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                               delta_qkv, 2, rank, Q38FN_TP_RANKS, &p) ||
            p.kind != Q38FN_TP_DELTA_QKV || p.n_ranges != 3 ||
            p.range[0].count != p.range[1].count ||
            p.range[2].count != (uint64_t)(Q38FN_LINEAR_VALUE_HEADS /
                                           Q38FN_TP_RANKS) * Q38FN_LINEAR_HEAD_DIM)
            return 1;
        if (q38fn_tp_make_plan("model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
                               hc_down, 2, rank, Q38FN_TP_RANKS, &p) ||
            p.kind != Q38FN_TP_AXIS0 || p.n_ranges != 1 ||
            q38fn_tp_make_plan("model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
                               hc_up, 2, rank, Q38FN_TP_RANKS, &p) ||
            p.kind != Q38FN_TP_AXIS1 || p.n_ranges != 1) return 1;
        if (q38fn_tp_make_plan("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                               hc_down, 2, rank, Q38FN_TP_RANKS, &p) ||
            p.kind != Q38FN_TP_AXIS0 || p.n_ranges != 1 ||
            q38fn_tp_make_plan("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                               hc_up, 2, rank, Q38FN_TP_RANKS, &p) ||
            p.kind != Q38FN_TP_AXIS1 || p.n_ranges != 1) return 1;
    }
    {
        uint64_t visual[2]={1024,1024};q38fn_tp_plan p;
        if(q38fn_tp_make_plan("model.visual.blocks.0.attn.qkv.weight",visual,2,0,Q38FN_TP_RANKS,&p)||p.kind!=Q38FN_TP_SKIP)return 1;
        if(q38fn_tp_make_plan("mtp.fc.weight",visual,2,0,Q38FN_TP_RANKS,&p)||p.kind!=Q38FN_TP_SKIP)return 1;
    }
    for (int shard = 0; shard < Q38FN_NGRAM_SHARDS; ++shard) {
        char name[160]; int owners = 0;
        snprintf(name, sizeof(name),
                 "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%d.weight",
                 shard);
        uint64_t s[2] = {Q38FN_NGRAM_ROWS_PER_SHARD, Q38FN_NGRAM_HEAD_DIM};
        for (int rank = 0; rank < Q38FN_TP_RANKS; ++rank) {
            q38fn_tp_plan p;
            if (q38fn_tp_make_plan(name, s, 2, rank, Q38FN_TP_RANKS, &p)) return 1;
            owners += p.kind == Q38FN_TP_NGRAM_OWNER;
        }
        if (owners != 1) return 1;
    }
    puts("Q38FN_TP_LAYOUT_TEST ok");
    return 0;
}
