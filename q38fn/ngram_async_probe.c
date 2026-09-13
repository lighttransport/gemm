/* Batched/deduplicated n-gram row-read probe for A64FX node-local storage. */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../common/glm53f_safetensors.h"
#include "../common/q38fn_arch.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct { int fd; off_t off; uint16_t data[Q38FN_NGRAM_HEAD_DIM]; } request;
typedef struct { request *req; int n; atomic_int next; } work;

static void *worker(void *arg) {
    work *w = (work *)arg;
    for (;;) {
        int i = atomic_fetch_add_explicit(&w->next, 1, memory_order_relaxed);
        if (i >= w->n) break;
        if (pread(w->req[i].fd, w->req[i].data, sizeof(w->req[i].data), w->req[i].off) !=
            (ssize_t)sizeof(w->req[i].data)) return (void *)(uintptr_t)1;
    }
    return NULL;
}

static double now_sec(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : NULL;
    const char *storage = argc > 2 ? argv[2] : NULL;
    int iters = argc > 3 ? atoi(argv[3]) : 1000;
    int partition = argc > 4 ? atoi(argv[4]) : 0;
    int workers = argc > 5 ? atoi(argv[5]) : 4;
    int duplicate_period = argc > 6 ? atoi(argv[6]) : 0;
    glm53f_st_context *ctx = NULL;
    char tensor_name[160], path[4096];
    const st_context *owner = NULL;
    const st_tensor_info *tensor;
    int fd = -1, sid = -1, unique_total = 0, reads = 0;
    uint64_t checksum = 0;
    double elapsed;
    if (!model || !storage || iters < 1 || partition < 0 || partition >= Q38FN_NGRAM_SHARDS ||
        workers < 1 || workers > 16 || duplicate_period < 0 || duplicate_period > 16) {
        fprintf(stderr, "usage: %s MODEL_DIR STORAGE_DIR iterations partition workers duplicate_period\n", argv[0]);
        return 2;
    }
    snprintf(tensor_name, sizeof(tensor_name),
             "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%d.weight", partition);
    ctx = glm53f_st_open(model);
    tensor = ctx ? glm53f_st_find(ctx, tensor_name, &owner) : NULL;
    if (!tensor || !owner) { fprintf(stderr, "missing %s\n", tensor_name); goto fail; }
    for (int s = 0; s < ctx->n_shards; ++s) if (ctx->shards[s].st == owner) { sid = s; break; }
    if (sid < 0 || snprintf(path, sizeof(path), "%s/%s", storage, ctx->shards[sid].name) >= (int)sizeof(path)) goto fail;
    fd = open(path, O_RDONLY); if (fd < 0) { perror(path); goto fail; }

    elapsed = now_sec();
    for (int n = 0; n < iters; ++n) {
        request req[Q38FN_NGRAM_HEADS], unique[Q38FN_NGRAM_HEADS];
        int nu = 0;
        for (int h = 0; h < Q38FN_NGRAM_HEADS; ++h) {
            uint64_t local = duplicate_period ? (uint64_t)(h % duplicate_period) :
                (uint64_t)((n * 7919 + h * 104729) % Q38FN_NGRAM_ROWS_PER_SHARD);
            req[h].fd = fd;
            req[h].off = (off_t)(owner->data_offset + tensor->offset + local * Q38FN_NGRAM_ROW_BYTES);
            int found = 0;
            for (int j = 0; j < nu; ++j) if (unique[j].off == req[h].off) { found = 1; break; }
            if (!found) unique[nu++] = req[h];
        }
        if (workers == 1) {
            for (int j = 0; j < nu; ++j)
                if (pread(fd, unique[j].data, sizeof(unique[j].data), unique[j].off) != (ssize_t)sizeof(unique[j].data)) goto fail;
        } else {
            pthread_t th[16]; work w = { unique, nu, 0 };
            int nt = workers < nu ? workers : nu;
            for (int j = 0; j < nt; ++j) if (pthread_create(&th[j], NULL, worker, &w) != 0) goto fail;
            for (int j = 0; j < nt; ++j) { void *rc; pthread_join(th[j], &rc); if (rc) goto fail; }
        }
        for (int j = 0; j < nu; ++j) checksum ^= (uint64_t)unique[j].data[(n + j) % Q38FN_NGRAM_HEAD_DIM];
        unique_total += nu; reads += Q38FN_NGRAM_HEADS;
    }
    elapsed = now_sec() - elapsed;
    printf("Q38FN_NGRAM_ASYNC iterations=%d requested=%d unique=%d workers=%d duplicate_period=%d "
           "elapsed=%.6f logical_lookups_s=%.2f unique_reads_s=%.2f checksum=%llu\n",
           iters, reads, unique_total, workers, duplicate_period, elapsed,
           (double)iters / elapsed, (double)unique_total / elapsed, (unsigned long long)checksum);
    close(fd); glm53f_st_close(ctx); return 0;
fail:
    if (fd >= 0) close(fd);
    glm53f_st_close(ctx);
    return 1;
}
