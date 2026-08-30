/* Synthetic correctness/bandwidth probe for q38fn_ngram_utofu_source. */
#define _GNU_SOURCE
#include "ngram_utofu_source.h"
#include "../utofu-tests/tofu_demo.h"
#include <utofu.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#if defined(Q38FN_USE_MPI)
#include <mpi.h>
#endif

#if defined(Q38FN_USE_MPI)
static int q38fn_mpi_active;
static void q38fn_mpi_finalize(void)
{
    if (q38fn_mpi_active) MPI_Finalize();
}
#endif

static double now_sec(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

typedef struct { uint32_t rank, nranks; } synthetic_ctx;

typedef struct {
    q38fn_ngram_source source;
    uint32_t rank, shard, rows, iters, worker;
    int rc;
} probe_worker;

static int synthetic_read(void *opaque, uint32_t shard, uint64_t first,
                          uint32_t rows, void *dst)
{
    const synthetic_ctx *ctx = (const synthetic_ctx *)opaque;
    if (!ctx || shard % ctx->nranks != ctx->rank) return EINVAL;
    uint16_t *p = (uint16_t *)dst;
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t lane = 0; lane < Q38FN_NGRAM_HEAD_DIM; ++lane)
            p[(size_t)r * Q38FN_NGRAM_HEAD_DIM + lane] =
                (uint16_t)((shard * 257u + (first + r) * 17u + lane) & 0xffffu);
    return 0;
}

static uint32_t env_rank(const char *topology, uint32_t nranks)
{
    const char *v = getenv("OMPI_COMM_WORLD_RANK");
    if (!v) v = getenv("PMI_RANK");
    if (!v) v = getenv("PMIX_RANK");
    if (v) return (uint32_t)strtoul(v, NULL, 10);
#if defined(Q38FN_USE_MPI)
    int mpi_rank = 0;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) == MPI_SUCCESS && mpi_rank >= 0)
        return (uint32_t)mpi_rank;
#endif
    utofu_tni_id_t *tnis = NULL; size_t ntni = 0;
    if (utofu_get_onesided_tnis(&tnis, &ntni) != UTOFU_SUCCESS || !ntni) {
        free(tnis); return UINT32_MAX;
    }
    uint8_t mine[TOFU_NCOORDS];
    int rc = utofu_query_my_coords(mine); free(tnis);
    if (rc != UTOFU_SUCCESS) return UINT32_MAX;
    FILE *f = fopen(topology, "r"); if (!f) return UINT32_MAX;
    char line[256]; uint32_t found = UINT32_MAX;
    while (fgets(line, sizeof line, f)) {
        unsigned rank, c[TOFU_NCOORDS];
        if (sscanf(line, "%u %u %u %u %u %u %u", &rank, &c[0], &c[1],
                   &c[2], &c[3], &c[4], &c[5]) == 7 && rank < nranks) {
            int match = 1;
            for (unsigned k = 0; k < TOFU_NCOORDS; ++k)
                if (mine[k] != (uint8_t)c[k]) match = 0;
            if (match) { found = rank; break; }
        }
    }
    fclose(f); return found;
}

static void *run_worker(void *opaque)
{
    probe_worker *w = (probe_worker *)opaque;
    uint16_t *buf = NULL;
    if (posix_memalign((void **)&buf, 64,
                       (size_t)w->rows * Q38FN_NGRAM_ROW_BYTES) != 0) {
        w->rc = ENOMEM; return NULL;
    }
    uint64_t first = 1234 + (uint64_t)w->rank * 31 + (uint64_t)w->worker * 997;
    for (uint32_t it = 0; it < w->iters && !w->rc; ++it) {
        w->rc = w->source.read_span(w->source.opaque, first, w->rows, buf);
        for (uint32_t lane = 0; !w->rc && lane < Q38FN_NGRAM_HEAD_DIM; ++lane) {
            uint16_t expect = (uint16_t)((w->shard * 257u + first * 17u + lane) & 0xffffu);
            if (buf[lane] != expect) {
                if (!w->rc)
                    fprintf(stderr, "rank %u worker %u mismatch shard=%u first=%llu lane=%u got=%u expect=%u\n",
                            w->rank, w->worker, w->shard,
                            (unsigned long long)first, lane, buf[lane], expect);
                w->rc = EPROTO;
            }
        }
    }
    free(buf);
    return NULL;
}

int main(int argc, char **argv)
{
#if defined(Q38FN_USE_MPI)
    MPI_Init(&argc, &argv);
    q38fn_mpi_active = 1;
    atexit(q38fn_mpi_finalize);
#endif
    const char *topo = argc > 1 ? argv[1] : "tofu_topo.txt";
    uint32_t nranks = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 4;
    uint32_t rows = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 256;
    uint32_t iters = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 32;
    uint32_t workers = argc > 5 ? (uint32_t)strtoul(argv[5], NULL, 10) : 1;
    uint32_t rank = env_rank(topo, nranks);
    if (!nranks || rank >= nranks || !rows || rows > Q38FN_UTOFU_MAX_SPAN || !iters ||
        !workers || workers > 64) {
        fprintf(stderr, "usage: %s [tofu_topo.txt] [ranks=4] [rows=256] [iters=32] [workers=1..64]\n", argv[0]);
        return 2;
    }
    q38fn_ngram_source sources[Q38FN_NGRAM_SHARDS];
    q38fn_utofu_source *transport = NULL;
    synthetic_ctx ctx = { rank, nranks };
    int rc = q38fn_utofu_source_init(&transport, topo, rank, nranks,
                                     synthetic_read, &ctx, sources,
                                     Q38FN_NGRAM_SHARDS);
    if (rc) { fprintf(stderr, "rank %u: init failed: %d\n", rank, rc); return 1; }
    uint32_t owner = (rank + 1) % nranks;
    uint32_t shard = owner;
    pthread_t *threads = calloc(workers, sizeof(*threads));
    probe_worker *jobs = calloc(workers, sizeof(*jobs));
    if (!threads || !jobs) {
        free(threads); free(jobs); q38fn_utofu_source_destroy(transport); return 1;
    }
    double t0 = now_sec();
    uint32_t started = 0;
    for (uint32_t w = 0; w < workers; ++w) {
        jobs[w] = (probe_worker){ sources[shard], rank, shard, rows, iters, w, 0 };
        if (pthread_create(&threads[w], NULL, run_worker, &jobs[w]) != 0) {
            rc = EAGAIN; break;
        }
        ++started;
    }
    for (uint32_t w = 0; w < started; ++w) pthread_join(threads[w], NULL);
    for (uint32_t w = 0; w < started && !rc; ++w) rc = jobs[w].rc;
    double dt = now_sec() - t0;
    double requests = (double)started * iters;
    double bytes = requests * rows * Q38FN_NGRAM_ROW_BYTES;
    printf("Q38FN_UTOFU_PROBE rank=%u peer=%u rows=%u iters=%u workers=%u rc=%d "
           "seconds=%.6f payload_GB_s=%.3f requests_s=%.1f\n",
           rank, owner, rows, iters, started, rc, dt, rc ? 0.0 : bytes / dt / 1e9,
           rc ? 0.0 : requests / dt);
    const char *prefix = getenv("Q38FN_RESULT_PREFIX");
    if (prefix && *prefix) {
        char path[4096];
        if (snprintf(path, sizeof(path), "%s.%u", prefix, rank) <
            (int)sizeof(path)) {
            FILE *result = fopen(path, "w");
            if (result) {
                fprintf(result,
                        "rank=%u peer=%u rows=%u iters=%u workers=%u rc=%d "
                        "seconds=%.6f payload_GB_s=%.3f requests_s=%.1f\n",
                        rank, owner, rows, iters, started, rc, dt,
                        rc ? 0.0 : bytes / dt / 1e9,
                        rc ? 0.0 : requests / dt);
                fclose(result);
            }
        }
    }
    free(threads); free(jobs); q38fn_utofu_source_destroy(transport);
    return rc ? 1 : 0;
}
