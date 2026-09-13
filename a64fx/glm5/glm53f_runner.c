/* Resource-bounded GLM5.3F checkpoint runner validation.
 *
 * This is the bring-up executable for one A64FX node.  It validates the real
 * sharded checkpoint, binds the first 1..3 decoder layers, and reads one
 * embedding row through bounded pread.  It intentionally does not claim
 * inference: linear-attention/MHC/DSA/MoE kernels are the next graph stage.
 */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int layer_of(const char *name) {
    const char *p = strstr(name, ".layers.");
    char *end;
    long v;
    if (!p) return -1;
    v = strtol(p + 8, &end, 10);
    return end == p + 8 || v < 0 || v > 1000 ? -1 : (int)v;
}

int main(int argc, char **argv) {
    char default_dir[4096];
    const char *home = getenv("HOME");
    const char *dir;
    int layers = argc > 2 ? atoi(argv[2]) : 3;
    int token = argc > 3 ? atoi(argv[3]) : 0;
    glm53f_st_context *ctx;
    uint16_t row[4096];
    size_t logical = 0;
    int i, selected = 0, row_ok;
    double checksum = 0.0;
    if (argc > 1) dir = argv[1];
    else {
        if (!home || snprintf(default_dir, sizeof(default_dir), "%s/models/glm53f", home) >= (int)sizeof(default_dir)) {
            fprintf(stderr, "HOME is unset; pass MODEL_DIR explicitly\n");
            return 2;
        }
        dir = default_dir;
    }
    if (layers < 1 || layers > 3 || token < 0 || token >= 154880) {
        fprintf(stderr, "usage: %s MODEL_DIR [layers=1..3] [token=0..154879]\n", argv[0]);
        return 2;
    }
    ctx = glm53f_st_open(dir);
    if (!ctx || glm53f_st_validate_contract(ctx, 1) != 0) {
        fprintf(stderr, "glm53f_runner: checkpoint contract failed\n");
        glm53f_st_close(ctx);
        return 1;
    }
    for (i = 0; i < ctx->n_entries; ++i) {
        int layer = layer_of(ctx->entries[i].name);
        if (layer >= 0 && layer < layers) {
            const st_context *owner = ctx->shards[ctx->entries[i].shard].st;
            logical += owner->tensors[ctx->entries[i].tensor].nbytes;
            selected++;
        }
    }
    row_ok = glm53f_st_read(ctx, "model.language_model.embed_tokens.weight",
                            (size_t)token * 4096 * sizeof(uint16_t), row, sizeof(row)) == 0;
    if (row_ok) for (i = 0; i < 4096; ++i) checksum += (double)row[i];
    printf("GLM53F_RUNNER_VALIDATE layers=%d tensors=%d logical_bytes=%zu "
           "embedding_row=%s checksum=%.0f\n", layers, selected, logical,
           row_ok ? "ok" : "FAIL", checksum);
    glm53f_st_close(ctx);
    return row_ok ? 0 : 1;
}
