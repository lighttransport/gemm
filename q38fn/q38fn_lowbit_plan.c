#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#define Q38FN_TP_RANKS 4
#include "../common/glm53f_safetensors.h"
#include "../common/q38fn_lowbit.h"
#include "../common/q38fn_tp_layout.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t bf16_main, q5_main, mxfp4_main;
    uint64_t bf16_ngram, q5_ngram, mxfp4_ngram;
} rank_bytes;

static uint64_t elements_for_plan(const st_tensor_info *tensor,
                                  const q38fn_tp_plan *plan)
{
    uint64_t elements = 1;
    if (plan->kind == Q38FN_TP_SKIP) return 0;
    for (int d = 0; d < tensor->n_dims; ++d) elements *= tensor->shape[d];
    if (plan->kind == Q38FN_TP_FULL || plan->kind == Q38FN_TP_NGRAM_OWNER)
        return elements;
    uint64_t selected = 0;
    for (int range = 0; range < plan->n_ranges; ++range)
        selected += plan->range[range].count;
    return elements / tensor->shape[plan->axis] * selected;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    glm53f_st_context *context = glm53f_st_open(argv[1]);
    if (!context) return 1;
    rank_bytes rank[Q38FN_TP_RANKS] = {{0}};
    int error = 0;
    for (int entry = 0; entry < context->n_entries && !error; ++entry) {
        const glm53f_st_entry *item = &context->entries[entry];
        const st_tensor_info *tensor = &context->shards[item->shard].st->tensors[item->tensor];
        if (strcmp(tensor->dtype_str, "BF16")) continue;
        int ngram = q38fn_tp_ngram_shard(item->name) >= 0;
        for (int r = 0; r < Q38FN_TP_RANKS; ++r) {
            q38fn_tp_plan plan;
            if (q38fn_tp_make_plan(item->name, tensor->shape, tensor->n_dims,
                                   r, Q38FN_TP_RANKS, &plan)) {
                fprintf(stderr, "unsupported tensor layout: %s\n", item->name);
                error = 1;
                break;
            }
            uint64_t elements = elements_for_plan(tensor, &plan);
            if (!elements) continue;
            uint64_t bf16 = elements * 2;
            uint64_t local_columns = plan.kind == Q38FN_TP_AXIS1 ?
                                     plan.range[0].count :
                                     tensor->shape[tensor->n_dims - 1];
            int quantizable = tensor->n_dims >= 2 && local_columns % 32 == 0;
            uint64_t q5 = quantizable ? elements / 32 * 22 : bf16;
            uint64_t mxfp4 = quantizable ? elements / 32 * 17 : bf16;
            if (ngram) {
                rank[r].bf16_ngram += bf16;
                rank[r].q5_ngram += q5;
                rank[r].mxfp4_ngram += mxfp4;
            } else {
                rank[r].bf16_main += bf16;
                rank[r].q5_main += q5;
                rank[r].mxfp4_main += mxfp4;
            }
        }
    }
    for (int r = 0; r < Q38FN_TP_RANKS && !error; ++r) {
        const double gib = 1073741824.0;
        printf("rank=%d bf16_main=%.3f q5_main=%.3f mxfp4_main=%.3f "
               "bf16_ngram=%.3f q5_ngram=%.3f mxfp4_ngram=%.3f "
               "q5_plus_q5=%.3f q5_plus_mxfp4=%.3f\n", r,
               rank[r].bf16_main/gib, rank[r].q5_main/gib,
               rank[r].mxfp4_main/gib, rank[r].bf16_ngram/gib,
               rank[r].q5_ngram/gib, rank[r].mxfp4_ngram/gib,
               (rank[r].q5_main+rank[r].q5_ngram)/gib,
               (rank[r].q5_main+rank[r].mxfp4_ngram)/gib);
    }
    glm53f_st_close(context);
    return error;
}
