#define _GNU_SOURCE
#include "ds41f_engram.h"

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef DS41F_USE_UTOFU
#include "../q38fn/ngram_utofu_source.h"
#endif

typedef struct {
    ds41f_engram *e;
    uint32_t rank, ranks, worker, iters;
    int remote;
#ifdef DS41F_USE_UTOFU
    q38fn_ngram_source *sources;
    q38fn_utofu_source *transport;
#endif
    uint64_t ok, errors, checksum, local, remote_rows;
    uint64_t *lat_ns;
} worker_arg;

static uint64_t now_ns(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}

static uint64_t rng_next(uint64_t *s)
{
    *s ^= *s << 7; *s ^= *s >> 9; *s ^= *s << 8; return *s;
}

static uint64_t table_rows(int layer)
{
    return layer == 0 ? UINT64_C(384006168) : UINT64_C(384016682);
}

static uint64_t owner_first(uint64_t rows, uint32_t owner, uint32_t ranks)
{
    return ((rows + ranks - 1) / ranks) * owner;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

#ifdef DS41F_USE_UTOFU
static int read_local_cb(void *opaque, uint32_t shard, uint64_t first,
                         uint32_t rows, void *dst)
{
    worker_arg *w = (worker_arg *)opaque;
    uint32_t owner = shard % w->ranks;
    int layer = (int)(shard / w->ranks);
    if (layer < 0 || layer >= DS41F_ENGRAM_LAYERS || owner != w->rank || rows != 1)
        return EINVAL;
    return ds41f_engram_read_local(w->e, layer,
                                   owner_first(table_rows(layer), owner, w->ranks) + first,
                                   (uint16_t *)dst);
}
#endif

static void *worker_main(void *opaque)
{
    worker_arg *w = (worker_arg *)opaque;
    uint64_t seed = UINT64_C(0x9e3779b97f4a7c15) ^
                    ((uint64_t)w->rank << 32) ^ w->worker;
    uint16_t row[DS41F_ENGRAM_DIM];
    for (uint32_t i = 0; i < w->iters; ++i) {
        int layer = (int)(rng_next(&seed) & 1u);
        uint32_t owner = (uint32_t)(rng_next(&seed) % w->ranks);
        if (!w->remote) owner = w->rank;
        uint64_t first = owner_first(table_rows(layer), owner, w->ranks);
        uint64_t owned = (table_rows(layer) - first + w->ranks - 1) / w->ranks;
        uint64_t local_row = rng_next(&seed) % owned;
        if (local_row >= table_rows(layer) - first) local_row = table_rows(layer) - first - 1;
        uint64_t start = now_ns();
        int rc;
#ifdef DS41F_USE_UTOFU
        if (owner != w->rank) {
            uint32_t shard = (uint32_t)layer * w->ranks + owner;
            rc = w->sources[shard].read_span(w->sources[shard].opaque, local_row, 1, row);
            ++w->remote_rows;
        } else {
            rc = ds41f_engram_read_local(w->e, layer, first + local_row, row);
            ++w->local;
        }
#else
        (void)owner;
        rc = ds41f_engram_read_local(w->e, layer, first + local_row, row);
        ++w->local;
#endif
        uint64_t elapsed = now_ns() - start;
        w->lat_ns[i] = elapsed;
        if (rc) { ++w->errors; continue; }
        ++w->ok;
        for (int j = 0; j < DS41F_ENGRAM_DIM; j += 32) w->checksum += row[j];
    }
    return NULL;
}

static void usage(const char *p)
{
    fprintf(stderr, "usage: %s --stage-dir DIR [--rank R --ranks N] [--iters N] [--workers N] [--remote]\n", p);
}

int main(int argc, char **argv)
{
    const char *stage = NULL, *topo = "tofu_topo.txt";
    uint32_t rank = UINT32_MAX, ranks = 1, iters = 1000, workers = 1;
    int remote = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--stage-dir") && i + 1 < argc) stage = argv[++i];
        else if (!strcmp(argv[i], "--rank") && i + 1 < argc) rank = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ranks") && i + 1 < argc) ranks = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--workers") && i + 1 < argc) workers = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--topology") && i + 1 < argc) topo = argv[++i];
        else if (!strcmp(argv[i], "--remote")) remote = 1;
        else { usage(argv[0]); return 2; }
    }
    if (rank == UINT32_MAX) {
        const char *v = getenv("OMPI_COMM_WORLD_RANK");
        if (!v) v = getenv("PMI_RANK");
        rank = v ? (uint32_t)strtoul(v, NULL, 10) : 0;
    }
    if (!stage || !ranks || rank >= ranks || ranks > DS41F_ENGRAM_MAX_RANKS ||
        !iters || !workers || workers > 64) { usage(argv[0]); return 2; }
    ds41f_engram e;
    int rc = ds41f_engram_open(&e, stage, rank, ranks);
    if (rc) { fprintf(stderr, "open stage failed: %s\n", strerror(rc)); return 1; }
    worker_arg base = { .e = &e, .rank = rank, .ranks = ranks,
                        .iters = iters, .remote = remote };
#ifdef DS41F_USE_UTOFU
    q38fn_ngram_source sources[128];
    q38fn_utofu_source *transport = NULL;
    (void)topo;
    if (remote) {
        base.sources = sources;
        rc = q38fn_utofu_source_init(&transport, topo, rank, ranks, read_local_cb,
                                     &base, sources, 128);
        if (rc) { fprintf(stderr, "uTofu init failed: %s\n", strerror(rc)); ds41f_engram_close(&e); return 1; }
    }
#else
    (void)topo;
    if (remote) { fprintf(stderr, "remote mode requires DS41F_USE_UTOFU build\n"); ds41f_engram_close(&e); return 2; }
#endif
    pthread_t *threads = calloc(workers, sizeof *threads);
    worker_arg *args = calloc(workers, sizeof *args);
    if (!threads || !args) return 1;
    uint64_t start = now_ns();
    for (uint32_t i = 0; i < workers; ++i) {
        args[i] = base; args[i].worker = i; args[i].lat_ns = calloc(iters, sizeof(uint64_t));
#ifdef DS41F_USE_UTOFU
        args[i].sources = sources;
#endif
        if (pthread_create(&threads[i], NULL, worker_main, &args[i])) return 1;
    }
    uint64_t ok = 0, errors = 0, checksum = 0, local = 0, remote_rows = 0;
    uint64_t min = UINT64_MAX, max = 0, total = 0, samples = (uint64_t)workers * iters;
    uint64_t *all_lat = calloc(samples ? samples : 1, sizeof *all_lat);
    uint64_t all_n = 0;
    for (uint32_t i = 0; i < workers; ++i) {
        pthread_join(threads[i], NULL); ok += args[i].ok; errors += args[i].errors;
        checksum += args[i].checksum; local += args[i].local; remote_rows += args[i].remote_rows;
        for (uint32_t j = 0; j < iters; ++j) if (args[i].lat_ns[j]) {
            if (args[i].lat_ns[j] < min) min = args[i].lat_ns[j];
            if (args[i].lat_ns[j] > max) max = args[i].lat_ns[j];
            total += args[i].lat_ns[j];
            if (all_n < samples) all_lat[all_n++] = args[i].lat_ns[j];
        }
    }
    qsort(all_lat, all_n, sizeof *all_lat, cmp_u64);
    uint64_t p50 = all_n ? all_lat[(all_n - 1) * 50 / 100] : 0;
    uint64_t p95 = all_n ? all_lat[(all_n - 1) * 95 / 100] : 0;
    uint64_t p99 = all_n ? all_lat[(all_n - 1) * 99 / 100] : 0;
    double sec = (double)(now_ns() - start) / 1e9;
    printf("DS41F_BENCH rank=%u ranks=%u workers=%u iters=%u remote=%d ok=%" PRIu64
           " errors=%" PRIu64 " lookups_s=%.1f local=%" PRIu64 " remote=%" PRIu64
           " latency_avg_us=%.3f latency_p50_us=%.3f latency_p95_us=%.3f"
           " latency_p99_us=%.3f latency_min_us=%.3f latency_max_us=%.3f checksum=%" PRIu64 "\n",
           rank, ranks, workers, iters, remote, ok, errors, ok / sec, local,
           remote_rows, ok ? (double)total / ok / 1000.0 : 0.0,
           p50 / 1000.0, p95 / 1000.0, p99 / 1000.0,
           min == UINT64_MAX ? 0.0 : min / 1000.0, max / 1000.0, checksum);
    for (uint32_t i = 0; i < workers; ++i) free(args[i].lat_ns);
    free(threads); free(args);
    free(all_lat);
#ifdef DS41F_USE_UTOFU
    if (remote) q38fn_utofu_source_destroy(transport);
#endif
    ds41f_engram_close(&e);
    return errors ? 1 : 0;
}
