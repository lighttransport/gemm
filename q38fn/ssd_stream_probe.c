/* Sequential payload-bandwidth probe for one packed Qwen3.8 PLE split. */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../common/glm53f_safetensors.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_sec(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : NULL;
    const char *storage = argc > 2 ? argv[2] : NULL;
    size_t mib = argc > 3 ? (size_t)atoll(argv[3]) : 512;
    const char *name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight";
    glm53f_st_context *ctx;
    const st_context *owner = NULL;
    const st_tensor_info *t;
    char path[4096];
    uint8_t *buf;
    int fd = -1, sid = -1;
    size_t total = mib * 1024u * 1024u, done = 0;
    uint64_t checksum = 0;
    double t0, elapsed;
    if (!model || !storage || mib < 1 || mib > 2048) {
        fprintf(stderr, "usage: %s MODEL_DIR STORAGE_DIR [MiB=512]\n", argv[0]); return 2;
    }
    ctx = glm53f_st_open(model);
    t = ctx ? glm53f_st_find(ctx, name, &owner) : NULL;
    if (!t || !owner) { fprintf(stderr, "missing n-gram split 0\n"); glm53f_st_close(ctx); return 1; }
    for (int i = 0; i < ctx->n_shards; ++i) if (ctx->shards[i].st == owner) { sid = i; break; }
    if (sid < 0 || snprintf(path, sizeof(path), "%s/%s", storage, ctx->shards[sid].name) >= (int)sizeof(path)) goto fail;
    fd = open(path, O_RDONLY); if (fd < 0) { perror(path); goto fail; }
    buf = (uint8_t *)malloc(1024 * 1024); if (!buf) goto fail;
    if (total > t->nbytes) total = t->nbytes;
    t0 = now_sec();
    while (done < total) {
        size_t n = total - done; if (n > 1024 * 1024) n = 1024 * 1024;
        ssize_t got = pread(fd, buf, n, (off_t)(owner->data_offset + t->offset + done));
        if (got != (ssize_t)n) { free(buf); goto fail; }
        for (size_t i = 0; i < n; i += 4096) checksum = checksum * 131 + buf[i];
        done += n;
    }
    elapsed = now_sec() - t0;
    printf("Q38FN_SSD_STREAM bytes=%zu elapsed=%.6f bandwidth_GB_s=%.3f checksum=%llu\n",
           done, elapsed, (double)done / elapsed / 1e9, (unsigned long long)checksum);
    free(buf); close(fd); glm53f_st_close(ctx); return 0;
fail:
    if (fd >= 0) close(fd);
    glm53f_st_close(ctx);
    return 1;
}
