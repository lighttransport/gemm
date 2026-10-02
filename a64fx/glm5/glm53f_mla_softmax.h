#ifndef GLM53F_MLA_SOFTMAX_H
#define GLM53F_MLA_SOFTMAX_H
#include <math.h>
#include <stddef.h>

/* Orphaned workshares: every member of the existing MLA team must call this.
 * Scalar expf is retained. Let the compiler use the same reduction policy as
 * the legacy exp-and-sum loop; forcing sequential sum changes fast-math bits.
 * Caller supplies one maximum per local head and disjoint logit/sum storage. */
static inline void glm53f_mla_softmax_parallel(float *logits, float *sums,
        float *maxima, int heads, int tokens, size_t stride) {
#pragma omp for schedule(static)
    for (int h = 0; h < heads; ++h) {
        const float *l = logits + (size_t)h * stride;
        float mx = -INFINITY;
        for (int t = 0; t < tokens; ++t) if (l[t] > mx) mx = l[t];
        maxima[h] = mx;
    }
    const int groups = (tokens + 63) / 64;
#pragma omp for schedule(static)
    for (int task = 0; task < heads * groups; ++task) {
        const int h = task / groups, begin = task % groups * 64;
        const int end = tokens - begin < 64 ? tokens : begin + 64;
        float *l = logits + (size_t)h * stride;
        for (int t = begin; t < end; ++t) l[t] = expf(l[t] - maxima[h]);
    }
#pragma omp for schedule(static)
    for (int h = 0; h < heads; ++h) {
        const float *l = logits + (size_t)h * stride;
        float sum = 0;
        for (int t = 0; t < tokens; ++t) sum += l[t];
        sums[h] = sum;
    }
}
#endif
