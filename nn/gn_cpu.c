/* SPDX-License-Identifier: MIT
 * Small NHWC projections with runtime AVX2/FMA dispatch and a persistent pool.
 * Portable scalar paths also cover AArch64; no global -march=native flag. */
#include "gn_cpu.h"
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#include <immintrin.h>
__attribute__((target("avx2,fma"))) static float dot_avx2(const float *a, const float *b,
                                                          size_t n) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum);
    float lanes[8];
    _mm256_storeu_ps(lanes, sum);
    float value = 0;
    for (int j = 0; j < 8; j++)
        value += lanes[j];
    for (; i < n; i++)
        value += a[i] * b[i];
    return value;
}
__attribute__((target("avx2,fma"))) static void
accumulate_avx2(float *dx, float *dw, const float *x, const float *w, float dy, size_t n) {
    __m256 g = _mm256_set1_ps(dy);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dx + i,
                         _mm256_fmadd_ps(g, _mm256_loadu_ps(w + i), _mm256_loadu_ps(dx + i)));
        _mm256_storeu_ps(dw + i,
                         _mm256_fmadd_ps(g, _mm256_loadu_ps(x + i), _mm256_loadu_ps(dw + i)));
    }
    for (; i < n; i++) {
        dx[i] += dy * w[i];
        dw[i] += dy * x[i];
    }
}
static int vectorized(void) {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif
static float dot(const float *a, const float *b, size_t n) {
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
    if (n >= 32 && vectorized())
        return dot_avx2(a, b, n);
#endif
    float sum = 0;
    for (size_t i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}
void gn_cpu_accumulate(float *dx, float *dw, const float *x, const float *w, float dy, size_t n) {
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
    if (n >= 32 && vectorized()) {
        accumulate_avx2(dx, dw, x, w, dy, n);
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) {
        dx[i] += dy * w[i];
        dw[i] += dy * x[i];
    }
}
typedef struct Pool Pool;
typedef struct {
    Pool *pool;
    unsigned id;
} Lane;
struct Pool {
    pthread_t threads[7];
    Lane lanes[7];
    pthread_mutex_t mutex;
    pthread_cond_t start, done;
    unsigned workers, finished, generation;
    int stop;
    gn_model *model;
    Node *node;
};
static void rows(Pool *p, unsigned id) {
    gn_model *m = p->model;
    Node *n = p->node, *a = &m->n[n->a];
    size_t side = m->cfg.side, S = side * side, C = n->c;
    for (size_t r = id; r < n->r; r += p->workers + 1)
        for (size_t c = 0; c < C; c++) {
            float value = n->bias->x[c];
            if (n->kind == LINEAR)
                value += dot(a->x + r * n->k, n->w->x + c * n->k, (size_t)n->k);
            else {
                int yy = (int)((r % S) / side), xx = (int)(r % side), K = n->k;
                for (int dy = 0; dy < K; dy++)
                    for (int dx = 0; dx < K; dx++) {
                        int sy = yy + dy - K / 2, sx = xx + dx - K / 2;
                        if (sy < 0 || sx < 0 || sy >= (int)side || sx >= (int)side)
                            continue;
                        size_t base = ((r / S) * S + (size_t)sy * side + sx) * a->c,
                               wb = (c * K * K + dy * K + dx) * a->c;
                        value += dot(a->x + base, n->w->x + wb, a->c);
                    }
            }
            n->x[r * C + c] = value;
        }
}
static void *worker(void *opaque) {
    Lane *lane = opaque;
    Pool *p = lane->pool;
    unsigned seen = 0;
    pthread_mutex_lock(&p->mutex);
    for (;;) {
        while (!p->stop && p->generation == seen)
            pthread_cond_wait(&p->start, &p->mutex);
        if (p->stop)
            break;
        seen = p->generation;
        pthread_mutex_unlock(&p->mutex);
        rows(p, lane->id);
        pthread_mutex_lock(&p->mutex);
        p->finished++;
        if (p->finished == p->workers)
            pthread_cond_signal(&p->done);
    }
    pthread_mutex_unlock(&p->mutex);
    return NULL;
}
void gn_cpu_close(void *opaque) {
    Pool *p = opaque;
    if (!p)
        return;
    pthread_mutex_lock(&p->mutex);
    p->stop = 1;
    pthread_cond_broadcast(&p->start);
    pthread_mutex_unlock(&p->mutex);
    for (unsigned i = 0; i < p->workers; i++)
        pthread_join(p->threads[i], NULL);
    pthread_cond_destroy(&p->done);
    pthread_cond_destroy(&p->start);
    pthread_mutex_destroy(&p->mutex);
    free(p);
}
static Pool *create(void) {
    Pool *p = calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    if (pthread_mutex_init(&p->mutex, NULL)) {
        free(p);
        return NULL;
    }
    if (pthread_cond_init(&p->start, NULL)) {
        pthread_mutex_destroy(&p->mutex);
        free(p);
        return NULL;
    }
    if (pthread_cond_init(&p->done, NULL)) {
        pthread_cond_destroy(&p->start);
        pthread_mutex_destroy(&p->mutex);
        free(p);
        return NULL;
    }
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    if (count > 8)
        count = 8;
    if (count < 1)
        count = 1;
    for (unsigned i = 0; i < (unsigned)count - 1; i++) {
        p->lanes[i] = (Lane){p, i + 1};
        if (pthread_create(&p->threads[i], NULL, worker, &p->lanes[i]))
            break;
        p->workers++;
    }
    return p;
}
int gn_cpu_projection(gn_model *m, Node *n) {
    /* Small correctness networks use a scalar scheduling path, retaining the
     * same operators and enabling cheap finite-difference checks. */
    if (m->cfg.channels < 32) {
        Pool local = {0};
        local.model = m;
        local.node = n;
        rows(&local, 0);
        return 0;
    }
    if (!m->cpu)
        m->cpu = create();
    Pool *p = m->cpu;
    if (!p)
        return gn_fail("CPU pool creation failed");
    pthread_mutex_lock(&p->mutex);
    p->model = m;
    p->node = n;
    p->finished = 0;
    p->generation++;
    pthread_cond_broadcast(&p->start);
    pthread_mutex_unlock(&p->mutex);
    rows(p, 0);
    pthread_mutex_lock(&p->mutex);
    while (p->finished < p->workers)
        pthread_cond_wait(&p->done, &p->mutex);
    pthread_mutex_unlock(&p->mutex);
    return 0;
}
