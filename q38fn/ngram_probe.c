/* q38fn n-gram lookup probe: metadata from MODEL_DIR, payload from STORAGE_DIR. */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../common/glm53f_safetensors.h"
#include "../common/q38fn_arch.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : NULL;
    const char *storage = argc > 2 ? argv[2] : model;
    int iters = argc > 3 ? atoi(argv[3]) : 1000;
    int fixed_part = argc > 4 ? atoi(argv[4]) : -1;
    glm53f_st_context *ctx;
    int fds[256];
    uint64_t rows[Q38FN_NGRAM_HEADS];
    uint16_t vec[Q38FN_NGRAM_HEAD_DIM];
    uint64_t checksum = 0;
    int opened = 0, reads = 0;
    double t0, elapsed;
    if (!model || !storage || iters < 1 || fixed_part < -1 || fixed_part >= Q38FN_NGRAM_SHARDS) {
        fprintf(stderr, "usage: %s MODEL_DIR [STORAGE_DIR] [iterations] [partition=-1..127]\n", argv[0]);
        return 2;
    }
    ctx = glm53f_st_open(model);
    if (!ctx || !glm53f_st_find(ctx, "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", NULL)) {
        fprintf(stderr, "q38fn_ngram_probe: checkpoint contract failed\n");
        glm53f_st_close(ctx);
        return 1;
    }
    for (int i = 0; i < 256; ++i) fds[i] = -1;
    t0 = seconds();
    for (int n = 0; n < iters; ++n) {
        /* Vary all three IDs so the benchmark does not repeatedly hit one row. */
        uint64_t cur = (uint64_t)(1000 + n * 17), prev = (uint64_t)(2000 + n * 31);
        uint64_t prev2 = (uint64_t)(3000 + n * 43);
        q38fn_ngram_rows(cur, prev, prev2, rows);
        for (int h = 0; h < Q38FN_NGRAM_HEADS; ++h) {
            char name[128];
            const st_context *owner = NULL;
            const st_tensor_info *ti;
            int sid = -1;
            /* shard_N splits the packed global-row axis; it is not one table
             * per hash head. */
            int part = fixed_part >= 0 ? fixed_part : (int)(rows[h] / Q38FN_NGRAM_ROWS_PER_SHARD);
            snprintf(name, sizeof(name), "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%d.weight", part);
            ti = glm53f_st_find(ctx, name, &owner);
            for (int s = 0; s < ctx->n_shards; ++s) if (ctx->shards[s].st == owner) { sid = s; break; }
            if (!ti || sid < 0) { fprintf(stderr, "missing %s\n", name); goto fail; }
            if (fds[sid] < 0) {
                char path[4096];
                if (snprintf(path, sizeof(path), "%s/%s", storage, ctx->shards[sid].name) >= (int)sizeof(path)) goto fail;
                fds[sid] = open(path, O_RDONLY); if (fds[sid] < 0) { perror(path); goto fail; }
                opened++;
            }
            /* The index's tensor is authoritative; rows are contiguous within
             * each split tensor and the split size is fixed by the checkpoint. */
            uint64_t table_row = rows[h] % Q38FN_NGRAM_ROWS_PER_SHARD;
            off_t off = (off_t)(owner->data_offset + ti->offset + table_row * Q38FN_NGRAM_ROW_BYTES);
            if (sid < 0 || pread(fds[sid], vec, sizeof(vec), off) != (ssize_t)sizeof(vec)) goto fail;
            checksum ^= ((uint64_t)vec[(n + h) % Q38FN_NGRAM_HEAD_DIM] << (h & 31));
            reads++;
        }
    }
    elapsed = seconds() - t0;
    printf("Q38FN_NGRAM_PROBE iterations=%d reads=%d bytes=%lld elapsed=%.6f "
           "lookups_s=%.2f payload_GB_s=%.3f checksum=%llu opened_shards=%d\n",
           iters, reads, (long long)reads * (long long)sizeof(vec), elapsed,
           (double)iters / elapsed, (double)reads * sizeof(vec) / elapsed / 1e9,
           (unsigned long long)checksum, opened);
    for (int i = 0; i < 256; ++i) if (fds[i] >= 0) close(fds[i]);
    glm53f_st_close(ctx);
    return 0;
fail:
    for (int i = 0; i < 256; ++i) if (fds[i] >= 0) close(fds[i]);
    glm53f_st_close(ctx);
    return 1;
}
