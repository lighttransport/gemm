#include "glm53f_pipeline.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL rank=%d line=%d: %s\n", rank, __LINE__, #c); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
typedef struct { int produced, executed, consumed, fail; } fixture;
static float input(int token, int component) { return (float)(token * 16 + component); }
static int produce(void *v, const glm53f_dist *d, float *x, int offset, int n, int flat) {
    fixture *f = v;
    if (d->map.stage || f->produced != offset) return -1;
    for (int t = 0; t < n; ++t) for (int j = 0; j < flat; ++j)
        x[(size_t)t * flat + j] = input(offset + t, j);
    f->produced += n; return 0;
}
static int execute(void *v, const glm53f_dist *d, float *x, int offset, int n, int flat) {
    fixture *f = v;
    if (f->executed != offset || (f->fail && d->map.world_rank == 5)) return -1;
    float value = (float)(d->map.tp_rank + 1), sum;
    if (glm53f_dist_sum(d, &value, &sum, 1)) return -1;
    float expected = (float)(d->map.tp_size * (d->map.tp_size + 1) / 2);
    if (sum != expected) return -1;
    for (int t = 0; t < n; ++t) for (int j = 0; j < flat; ++j) {
        size_t k = (size_t)t * flat + j;
        if (x[k] != input(offset + t, j) + d->map.stage * expected) return -1;
        x[k] += sum;
    }
    f->executed += n; return 0;
}
static int consume(void *v, const glm53f_dist *d, float *x, int offset, int n, int flat) {
    fixture *f = v;
    if (d->map.stage != d->map.stages - 1 || f->consumed != offset) return -1;
    float addition = (float)(d->map.stages * d->map.tp_size * (d->map.tp_size + 1) / 2);
    for (int t = 0; t < n; ++t) for (int j = 0; j < flat; ++j)
        if (x[(size_t)t * flat + j] != input(offset + t, j) + addition) return -1;
    f->consumed += n; return 0;
}
int main(int argc, char **argv) {
    int provided, rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    CHECK(provided >= MPI_THREAD_SERIALIZED && size == 12);
    glm53f_parallel_config c = glm53f_parallel_default();
    glm53f_dist d;
    c.layout = GLM53F_PP3_TP4;
    c.cuts[0] = rank ? 15 : 14;
    CHECK(glm53f_dist_init(&d, MPI_COMM_WORLD, &c) == -1);
    c.cuts[0] = 15;
    c.microbatch = rank ? 512 : 17;
    CHECK(glm53f_dist_init(&d, MPI_COMM_WORLD, &c) == -1);
    const int batches[] = {512, 1024, 2048};
    for (int layout = argc > 1 ? 1 : 0; layout < 2; ++layout) for (int b = 0; b < 3; ++b) {
        c.layout = (glm53f_parallel_layout)layout; c.microbatch = batches[b];
        CHECK(!glm53f_dist_init(&d, MPI_COMM_WORLD, &c));
        int local_rank, local_size;
        MPI_Comm_rank(d.tp, &local_rank); MPI_Comm_size(d.tp, &local_size);
        CHECK(local_rank == d.map.tp_rank && local_size == d.map.tp_size);
        MPI_Comm_rank(d.pipeline, &local_rank); MPI_Comm_size(d.pipeline, &local_size);
        CHECK(local_rank == d.map.stage && local_size == d.map.stages);
        fixture f = {0};
        CHECK(glm53f_pipeline_run(&d, rank ? 1 : 2, 16, GLM53F_PIPELINE_SERIAL,
            produce, execute, consume, &f, NULL) == -1);
        CHECK(glm53f_pipeline_run(&d, 1, 15, GLM53F_PIPELINE_SERIAL,
            produce, execute, consume, &f, NULL) == -1);
        const int lengths[] = {1, batches[b] - 1, batches[b], batches[b] + 1,
            4 * batches[b], 4 * batches[b] + 3, 8049};
        for (int s = 0; s < 2; ++s) for (int i = 0; i < 7; ++i) for (int repeat = 0; repeat < 2; ++repeat) {
            memset(&f, 0, sizeof(f));
            f.fail = argc > 1 && !strcmp(argv[1], "--fail-callback");
            glm53f_pipeline_profile p;
            CHECK(!glm53f_pipeline_run(&d, lengths[i], 16, (glm53f_pipeline_schedule)s,
                produce, execute, consume, &f, &p));
            CHECK(f.executed == lengths[i]);
            CHECK(f.produced == (d.map.stage == 0 ? lengths[i] : 0));
            CHECK(f.consumed == (d.map.stage == d.map.stages - 1 ? lengths[i] : 0));
            CHECK(p.positions == lengths[i] && p.microbatches == 1 + (lengths[i] - 1) / batches[b]);
            CHECK(p.compute_seconds >= 0 && p.receive_seconds >= 0 && p.send_wait_seconds >= 0);
        }
        if (layout == GLM53F_PP3_TP4) {
            memset(&f, 0, sizeof(f));
            CHECK(!glm53f_pipeline_run(&d, 4 * batches[b] + 3, 4 * 4096,
                GLM53F_PIPELINE_OVERLAP, produce, execute, consume, &f, NULL));
        }
        glm53f_dist_free(&d);
        CHECK(!d.initialized);
    }
    if (!rank) puts("GLM53F_PIPELINE_PASS layouts=2 batches=3 schedules=2 tails=7 repeats=2");
    MPI_Finalize(); return 0;
}
