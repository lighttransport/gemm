/* End-to-end uTofu probe using the real Qwen3.8 n-gram safetensors. */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "ngram_utofu_source.h"
#include "ngram_owner_hbm.h"
#include "../../common/glm53f_safetensors.h"
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    glm53f_st_context *ctx;
    q38fn_ngram_owner_hbm *hbm;
} real_reader;

typedef struct {
    q38fn_ngram_source source;
    real_reader *reader;
    uint32_t shard, rows, iters, worker;
    int rc;
} real_worker;

static int read_file(void *opaque, uint32_t shard, uint64_t first,
                     uint32_t rows, void *dst)
{
    real_reader *r = (real_reader *)opaque;
    char name[192];
    int n = snprintf(name, sizeof name,
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%u.weight",
        shard);
    if (n < 0 || (size_t)n >= sizeof name) return -1;
    return glm53f_st_read(r->ctx, name, (size_t)first * Q38FN_NGRAM_ROW_BYTES,
                          dst, (size_t)rows * Q38FN_NGRAM_ROW_BYTES);
}

static int read_real(void *opaque, uint32_t shard, uint64_t first,
                     uint32_t rows, void *dst)
{
    real_reader *r = (real_reader *)opaque;
    return r->hbm ? q38fn_ngram_owner_hbm_read(r->hbm, shard, first, rows, dst) :
                    read_file(opaque, shard, first, rows, dst);
}

static uint32_t rank_from_env(void)
{
    const char *v = getenv("OMPI_COMM_WORLD_RANK");
    if (!v) v = getenv("PMI_RANK");
    if (!v) v = getenv("PMIX_RANK");
    return v ? (uint32_t)strtoul(v, NULL, 10) : 0;
}

static double now_sec(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void *run_real_worker(void *opaque)
{
    real_worker *w = (real_worker *)opaque;
    size_t bytes = (size_t)w->rows * Q38FN_NGRAM_ROW_BYTES;
    uint16_t *got = malloc(bytes), *expect = malloc(bytes);
    if (!got || !expect) { free(got); free(expect); w->rc = ENOMEM; return NULL; }
    uint64_t first = 1000 + (uint64_t)w->worker * 97;
    for (uint32_t it = 0; it < w->iters && !w->rc; ++it) {
        w->rc = w->source.read_span(w->source.opaque, first, w->rows, got);
        if (!w->rc && it == 0) {
            w->rc = read_file(w->reader, w->shard, first, w->rows, expect);
            if (!w->rc && memcmp(got, expect, bytes)) w->rc = EPROTO;
        }
    }
    free(got); free(expect);
    return NULL;
}

int main(int argc, char **argv)
{
    const char *model = argc > 1 ? argv[1] : NULL;
    const char *topo = argc > 2 ? argv[2] : "tofu_topo.txt";
    uint32_t nranks = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 4;
    uint32_t rank = rank_from_env();
    uint32_t rows = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 16;
    uint32_t iters = argc > 5 ? (uint32_t)strtoul(argv[5], NULL, 10) : 100;
    if (!model || !nranks || rank >= nranks || !rows || rows > Q38FN_UTOFU_MAX_SPAN || !iters)
        return fprintf(stderr, "usage: %s MODEL_DIR [topo] [ranks=4] [rows=16] [iters=100] [resident] [workers=1..64]\n", argv[0]), 2;
    int rc = 0;
    int resident = 0;
    uint32_t workers = 1;
    int arg = 6;
    if (argc > arg && !strcmp(argv[arg], "resident")) { resident = 1; ++arg; }
    if (argc > arg) workers = (uint32_t)strtoul(argv[arg], NULL, 10);
    if (!workers || workers > 64) return fprintf(stderr, "workers must be in [1,64]\n"), 2;
    real_reader reader = { glm53f_st_open(model), NULL };
    if (!reader.ctx) { fprintf(stderr, "rank %u: cannot open model metadata\n", rank); return 1; }
    q38fn_ngram_source sources[Q38FN_NGRAM_SHARDS];
    q38fn_utofu_source *transport = NULL;
    /* Establish the Tofu registration before reserving the large resident
     * HBM image.  Some Fugaku runtime configurations reject a new Tofu
     * registration once the process has consumed most of its memory; the
     * callback observes reader.hbm becoming non-NULL after this step. */
    rc = q38fn_utofu_source_init(&transport, topo, rank, nranks,
                                     read_real, &reader, sources, Q38FN_NGRAM_SHARDS);
    if (rc) { fprintf(stderr, "rank %u: uTofu init failed: %d\n", rank, rc);
        glm53f_st_close(reader.ctx); return 1; }
    if (resident) {
        rc = q38fn_ngram_owner_hbm_open(&reader.hbm, model, rank, nranks);
        if (rc) { fprintf(stderr, "rank %u: HBM owner load failed: %d\n", rank, rc);
            q38fn_utofu_source_destroy(transport); glm53f_st_close(reader.ctx); return 1; }
    }
    uint32_t shard = (rank + 1) % nranks;
    pthread_t *threads = calloc(workers, sizeof(*threads));
    real_worker *jobs = calloc(workers, sizeof(*jobs));
    if (!threads || !jobs) rc = ENOMEM;
    double t0 = now_sec();
    uint32_t started = 0;
    for (uint32_t w = 0; !rc && w < workers; ++w) {
        jobs[w] = (real_worker){ sources[shard], &reader, shard, rows, iters, w, 0 };
        if (pthread_create(&threads[w], NULL, run_real_worker, &jobs[w]) != 0) {
            rc = EAGAIN; break;
        }
        ++started;
    }
    for (uint32_t w = 0; w < started; ++w) pthread_join(threads[w], NULL);
    for (uint32_t w = 0; w < started && !rc; ++w) rc = jobs[w].rc;
    double dt = now_sec() - t0;
    double requests = (double)started * iters;
    double bytes = requests * rows * Q38FN_NGRAM_ROW_BYTES;
    printf("Q38FN_UTOFU_REAL rank=%u peer=%u rows=%u iters=%u workers=%u rc=%d seconds=%.6f "
           "payload_GB_s=%.3f requests_s=%.1f\n", rank, shard % nranks, rows,
           iters, started, rc, dt, rc ? 0.0 : bytes / dt / 1e9,
           rc ? 0.0 : requests / dt);
    free(threads); free(jobs); q38fn_utofu_source_destroy(transport);
    q38fn_ngram_owner_hbm_close(reader.hbm); glm53f_st_close(reader.ctx);
    return rc ? 1 : 0;
}
