/* Print the exact tensor contract for a Q38FN checkpoint layer. */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../common/glm53f_safetensors.h"
#include "../common/q38fn_arch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int has_layer_prefix(const char *name, int layer)
{
    char prefix[96];
    int n = snprintf(prefix, sizeof(prefix),
                     "model.language_model.layers.%d.", layer);
    return n > 0 && (size_t)n < sizeof(prefix) &&
           strncmp(name, prefix, (size_t)n) == 0;
}

static void print_tensor(const glm53f_st_context *ctx, int entry)
{
    const glm53f_st_entry *e = &ctx->entries[entry];
    const st_context *owner = ctx->shards[e->shard].st;
    const st_tensor_info *t = &owner->tensors[e->tensor];

    printf("%s\tdtype=%s\tshape=", e->name, t->dtype_str);
    for (int d = 0; d < t->n_dims; ++d)
        printf("%s%llu", d ? "x" : "", (unsigned long long)t->shape[d]);
    printf("\tbytes=%zu\tshard=%s\n", t->nbytes, ctx->shards[e->shard].name);
}

int main(int argc, char **argv)
{
    glm53f_st_context *ctx;
    int layer;

    if (argc != 3 || (layer = atoi(argv[2])) < 0 || layer >= Q38FN_LAYERS) {
        fprintf(stderr, "usage: %s MODEL_DIR LAYER\n", argv[0]);
        return 2;
    }
    ctx = glm53f_st_open(argv[1]);
    if (!ctx) {
        fprintf(stderr, "q38fn_inspect: checkpoint open failed\n");
        return 1;
    }
    for (int i = 0; i < ctx->n_entries; ++i) {
        const char *name = ctx->entries[i].name;
        if (has_layer_prefix(name, layer) ||
            strcmp(name, "model.language_model.embed_tokens.weight") == 0 ||
            strcmp(name, "model.language_model.norm.weight") == 0 ||
            strcmp(name, "lm_head.weight") == 0)
            print_tensor(ctx, i);
    }
    glm53f_st_close(ctx);
    return 0;
}
