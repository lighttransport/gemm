/* Print selected GLM-5.3F checkpoint tensor contracts without reading payloads. */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *prefix;
    glm53f_st_context *ctx;
    int found = 0;
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_DIR PREFIX\n", argv[0]);
        return 2;
    }
    prefix = argv[2];
    ctx = glm53f_st_open(argv[1]);
    if (!ctx) return 1;
    for (int i = 0; i < ctx->n_entries; ++i) {
        const glm53f_st_entry *e = &ctx->entries[i];
        const st_tensor_info *t;
        if (strncmp(e->name, prefix, strlen(prefix))) continue;
        t = &ctx->shards[e->shard].st->tensors[e->tensor];
        printf("%s dtype=%s shape=[", e->name, t->dtype_str);
        for (int d = 0; d < t->n_dims; ++d)
            printf("%s%" PRIu64, d ? "," : "", t->shape[d]);
        printf("] bytes=%zu shard=%s\n", t->nbytes, ctx->shards[e->shard].name);
        found++;
    }
    printf("SENTINEL glm53f_tensor_info=%s found=%d\n", found ? "OK" : "EMPTY", found);
    glm53f_st_close(ctx);
    return found ? 0 : 1;
}
