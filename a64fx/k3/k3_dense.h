#ifndef K3_DENSE_H
#define K3_DENSE_H

#include <stddef.h>
#include <stdint.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "ggml_dequant.h"

typedef struct {
    const uint16_t *weight;
    int rows;
    int cols;
} k3_bf16_matrix;

/* A64FX loses bandwidth at 48 workers for these two matrices.  Reserving the
 * highest cpuset core for uTofu progress also makes 47 the runner default. */
#define K3_DENSE_THREADS 47

static inline void k3_dense_pair_bf16(float *router_out, float *latent_out,
                                      const k3_bf16_matrix *router,
                                      const k3_bf16_matrix *latent_down,
                                      const float *hidden, int threads) {
    int router_groups = router->rows / 8;
    int down_groups = latent_down->rows / 8;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#else
    (void)threads;
#endif
    for (int task = 0; task < router_groups + down_groups; ++task) {
        const k3_bf16_matrix *m = task < router_groups ? router : latent_down;
        float *out = task < router_groups ? router_out : latent_out;
        int group = task < router_groups ? task : task - router_groups;
        int row = group * 8;
        const uint16_t *w = m->weight + (size_t)row * m->cols;
        matvec_bf16_8row(out + row, w, w + m->cols, w + 2 * m->cols,
                         w + 3 * m->cols, w + 4 * m->cols, w + 5 * m->cols,
                         w + 6 * m->cols, w + 7 * m->cols, hidden, m->cols);
    }
}

#endif
