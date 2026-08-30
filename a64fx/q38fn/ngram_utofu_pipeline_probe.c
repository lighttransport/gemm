/* End-to-end Q38FN token-window pipeline probe over uTofu. */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "ngram_utofu_source.h"
#include "ngram_owner_hbm.h"
#include "../../common/glm53f_safetensors.h"
#include "../../common/q38fn_ngram_pipeline.h"
#include "../utofu-tests/tofu_demo.h"
#include <utofu.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

typedef struct {
    glm53f_st_context *ctx;
    q38fn_ngram_owner_hbm *hbm;
    /* The safetensors reader owns mutable scratch/seek state.  uTofu has one
     * service thread per peer, so protect the shared file path.  The resident
     * HBM path bypasses this lock and remains concurrent. */
    pthread_mutex_t file_mu;
} real_reader;

static int read_file(void *opaque, uint32_t shard, uint64_t first,
                     uint32_t rows, void *dst)
{
    real_reader *r = (real_reader *)opaque;
    char name[192];
    int n = snprintf(name, sizeof name,
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%u.weight",
        shard);
    if (n < 0 || (size_t)n >= sizeof name) return EINVAL;
    pthread_mutex_lock(&r->file_mu);
    int rc = glm53f_st_read(r->ctx, name,
                            (size_t)first * Q38FN_NGRAM_ROW_BYTES,
                            dst, (size_t)rows * Q38FN_NGRAM_ROW_BYTES);
    pthread_mutex_unlock(&r->file_mu);
    return rc;
}

static int read_real(void *opaque, uint32_t shard, uint64_t first,
                     uint32_t rows, void *dst)
{
    real_reader *r = (real_reader *)opaque;
    return r->hbm ? q38fn_ngram_owner_hbm_read(r->hbm, shard, first, rows, dst) :
                    read_file(opaque, shard, first, rows, dst);
}

static uint32_t rank_from_env(const char *topology, uint32_t nranks)
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

    /* Fujitsu's in-allocation mpiexec may export no rank variable.  uTofu
     * coordinates are process-local and stable, so use the allocation's
     * rank-ordered topology as the fallback identity. */
    utofu_tni_id_t *tnis = NULL;
    size_t ntni = 0;
    if (utofu_get_onesided_tnis(&tnis, &ntni) != UTOFU_SUCCESS || !ntni) {
        free(tnis); return UINT32_MAX;
    }
    uint8_t mine[TOFU_NCOORDS];
    int rc = utofu_query_my_coords(mine);
    free(tnis);
    if (rc != UTOFU_SUCCESS) return UINT32_MAX;
    FILE *f = fopen(topology, "r");
    if (!f) return UINT32_MAX;
    char line[256];
    uint32_t found = UINT32_MAX;
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
    fclose(f);
    return found;
}

static double now_sec(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void make_contexts(uint64_t base, uint64_t *cur, uint64_t *prev,
                          uint64_t *prev2, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        cur[i] = base + i;
        prev[i] = base > 1 ? base - 1 + i : Q38FN_EOS;
        prev2[i] = base > 2 ? base - 2 + i : Q38FN_EOS;
    }
}

int main(int argc, char **argv)
{
#if defined(Q38FN_USE_MPI)
    MPI_Init(&argc, &argv);
    q38fn_mpi_active = 1;
    atexit(q38fn_mpi_finalize);
#endif
    const char *model = argc > 1 ? argv[1] : NULL;
    const char *topo = argc > 2 ? argv[2] : "tofu_topo.txt";
    uint32_t nranks = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 4;
    uint32_t rank = rank_from_env(topo, nranks);
    uint32_t iters = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 1000;
    uint32_t workers = argc > 5 ? (uint32_t)strtoul(argv[5], NULL, 10) : 8;
    uint32_t window = argc > 6 ? (uint32_t)strtoul(argv[6], NULL, 10) : 8;
    uint32_t max_span = 16;
    const char *span_env = getenv("Q38FN_PIPELINE_MAX_SPAN");
    if (span_env && *span_env) {
        char *end = NULL;
        unsigned long value = strtoul(span_env, &end, 10);
        if (end != span_env && *end == '\0' && value >= 1 && value <= 32)
            max_span = (uint32_t)value;
    }
    int resident = argc > 7 && !strcmp(argv[7], "resident");
    if (!model || !nranks || rank >= nranks || !iters || !workers || workers > 64 ||
        !window || window > Q38FN_NGRAM_MAX_TOKEN_WINDOW)
        return fprintf(stderr, "usage: %s MODEL_DIR [topo] [ranks=4] [iters=1000] "
                              "[workers=8] [window=8] [resident]\n", argv[0]), 2;

    real_reader reader = { glm53f_st_open(model), NULL, PTHREAD_MUTEX_INITIALIZER };
    if (!reader.ctx) return fprintf(stderr, "rank %u: checkpoint open failed\n", rank), 1;
    q38fn_ngram_source sources[Q38FN_NGRAM_SHARDS];
    q38fn_utofu_source *transport = NULL;
    int rc = q38fn_utofu_source_init(&transport, topo, rank, nranks,
                                     read_real, &reader, sources,
                                     Q38FN_NGRAM_SHARDS);
    if (rc) {
        fprintf(stderr, "rank %u: uTofu init failed: %d\n", rank, rc);
        glm53f_st_close(reader.ctx); return 1;
    }
    if (resident) {
        rc = q38fn_ngram_owner_hbm_open(&reader.hbm, model, rank, nranks);
        if (rc) {
            fprintf(stderr, "rank %u: HBM owner load failed: %d\n", rank, rc);
            q38fn_utofu_source_destroy(transport); glm53f_st_close(reader.ctx); return 1;
        }
        uint64_t hbm_checksum = 0;
        uint32_t hbm_samples = 0;
        rc = q38fn_ngram_owner_hbm_validate(reader.hbm, &hbm_checksum,
                                             &hbm_samples);
        if (rc) {
            fprintf(stderr, "rank %u: HBM owner validation failed: %d\n", rank, rc);
            q38fn_utofu_source_destroy(transport); q38fn_ngram_owner_hbm_close(reader.hbm);
            glm53f_st_close(reader.ctx); return 1;
        }
        fprintf(stderr, "q38fn_hbm rank=%u validated samples=%u checksum=%llu\n",
                rank, hbm_samples, (unsigned long long)hbm_checksum);
    }

    uint32_t depth = workers * 4;
    if (depth < 8) depth = 8;
    q38fn_ngram_pipeline *pipeline = NULL;
    rc = q38fn_ngram_pipeline_init_ex(&pipeline, sources, Q38FN_NGRAM_SHARDS,
                                      depth, workers, max_span, 2, 2048);
    if (rc) {
        fprintf(stderr, "rank %u: pipeline init failed: %d\n", rank, rc);
        q38fn_utofu_source_destroy(transport); q38fn_ngram_owner_hbm_close(reader.hbm);
        glm53f_st_close(reader.ctx); return 1;
    }
    q38fn_ngram_set_cache_remote_only(pipeline, rank);
    q38fn_ngram_ticket *tickets = calloc(depth, sizeof(*tickets));
    size_t batch_rows = (size_t)window * Q38FN_NGRAM_HEADS;
    uint16_t *out = malloc((size_t)depth * batch_rows * Q38FN_NGRAM_ROW_BYTES);
    uint64_t *cur = malloc((size_t)window * sizeof(*cur));
    uint64_t *prev = malloc((size_t)window * sizeof(*prev));
    uint64_t *prev2 = malloc((size_t)window * sizeof(*prev2));
    if (!tickets || !out || !cur || !prev || !prev2) rc = ENOMEM;

    uint32_t submitted = 0, completed = 0, outstanding = 0;
    uint64_t checksum = 0;
    double t0 = now_sec();
    while (!rc && completed < iters) {
        while (submitted < iters && outstanding < depth) {
            uint32_t slot = submitted % depth;
            uint64_t base = (uint64_t)submitted * window + 1000;
            make_contexts(base, cur, prev, prev2, window);
            rc = q38fn_ngram_submit_token_window(pipeline, cur, prev, prev2,
                                                  window, &tickets[slot]);
            if (rc) break;
            ++submitted; ++outstanding;
        }
        if (rc) break;
        uint32_t slot = completed % depth;
        uint16_t *batch = out + (size_t)slot * batch_rows * Q38FN_NGRAM_HEAD_DIM;
        rc = q38fn_ngram_wait(pipeline, tickets[slot], batch,
                              batch_rows * Q38FN_NGRAM_ROW_BYTES);
        if (rc) break;
        if (completed == 0) {
            uint64_t cur0[Q38FN_NGRAM_MAX_TOKEN_WINDOW];
            uint64_t prev0[Q38FN_NGRAM_MAX_TOKEN_WINDOW];
            uint64_t prev20[Q38FN_NGRAM_MAX_TOKEN_WINDOW];
            uint64_t rows0[Q38FN_NGRAM_HEADS];
            uint16_t expected[Q38FN_NGRAM_HEAD_DIM];
            make_contexts(1000, cur0, prev0, prev20, window);
            q38fn_ngram_rows(cur0[0], prev0[0], prev20[0], rows0);
            uint32_t expected_shard = (uint32_t)(rows0[0] / Q38FN_NGRAM_ROWS_PER_SHARD);
            uint64_t expected_first = rows0[0] % Q38FN_NGRAM_ROWS_PER_SHARD;
            rc = read_file(&reader, expected_shard, expected_first, 1, expected);
            if (!rc && memcmp(batch, expected, Q38FN_NGRAM_ROW_BYTES)) rc = EPROTO;
            if (rc) break;
        }
        for (size_t i = 0; i < batch_rows * Q38FN_NGRAM_HEAD_DIM; ++i)
            checksum = (checksum << 7) ^ (checksum >> 3) ^ batch[i];
        --outstanding; ++completed;
    }
    double elapsed = now_sec() - t0;
    q38fn_ngram_stats stats = {0};
    q38fn_ngram_get_stats(pipeline, &stats);
    printf("Q38FN_UTOFU_PIPELINE rank=%u iters=%u completed=%u workers=%u window=%u "
           "depth=%u rc=%d seconds=%.6f logical_s=%.2f unique_s=%.2f "
           "useful_GB_s=%.3f physical_GB_s=%.3f spans=%llu direct=%llu reorder=%llu "
           "cache_hits=%llu cache_misses=%llu checksum=%llu\n", rank, iters,
           completed, workers, window, depth, rc, elapsed,
           elapsed > 0 ? stats.logical_rows / elapsed : 0.0,
           elapsed > 0 ? stats.unique_rows / elapsed : 0.0,
           elapsed > 0 ? stats.useful_bytes / elapsed / 1e9 : 0.0,
           elapsed > 0 ? stats.physical_bytes / elapsed / 1e9 : 0.0,
           (unsigned long long)stats.spans, (unsigned long long)stats.direct_spans,
           (unsigned long long)stats.reorder_spans,
           (unsigned long long)stats.cache_hits, (unsigned long long)stats.cache_misses,
           (unsigned long long)checksum);
    const char *result_prefix = getenv("Q38FN_RESULT_PREFIX");
    if (result_prefix && *result_prefix) {
        char result_path[4096];
        if (snprintf(result_path, sizeof result_path, "%s.%u",
                     result_prefix, rank) < (int)sizeof result_path) {
            FILE *result = fopen(result_path, "w");
            if (result) {
                fprintf(result, "Q38FN_UTOFU_PIPELINE rank=%u iters=%u completed=%u "
                        "workers=%u window=%u depth=%u rc=%d seconds=%.6f "
                        "logical_s=%.2f unique_s=%.2f useful_GB_s=%.3f "
                        "physical_GB_s=%.3f spans=%llu direct=%llu reorder=%llu "
                        "cache_hits=%llu cache_misses=%llu checksum=%llu\n",
                        rank, iters, completed, workers, window, depth, rc, elapsed,
                        elapsed > 0 ? stats.logical_rows / elapsed : 0.0,
                        elapsed > 0 ? stats.unique_rows / elapsed : 0.0,
                        elapsed > 0 ? stats.useful_bytes / elapsed / 1e9 : 0.0,
                        elapsed > 0 ? stats.physical_bytes / elapsed / 1e9 : 0.0,
                        (unsigned long long)stats.spans,
                        (unsigned long long)stats.direct_spans,
                        (unsigned long long)stats.reorder_spans,
                        (unsigned long long)stats.cache_hits,
                        (unsigned long long)stats.cache_misses,
                        (unsigned long long)checksum);
                fclose(result);
            }
        }
    }
    free(tickets); free(out); free(cur); free(prev); free(prev2);
    q38fn_ngram_pipeline_destroy(pipeline);
    /* Service threads retain read_opaque until transport destruction.  Stop
     * them before unmapping the rank-owned HBM image. */
    q38fn_utofu_source_destroy(transport);
    q38fn_ngram_owner_hbm_close(reader.hbm);
    glm53f_st_close(reader.ctx);
    return rc ? 1 : 0;
}
