#ifndef GLM53F_POOL_SELECT_H
#define GLM53F_POOL_SELECT_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct { float score; int id; } glm53f_pool_score;
static inline int glm53f_pool_cmp(const void *a, const void *b) {
    const glm53f_pool_score *x = a, *y = b;
    if (x->score > y->score) return -1;
    if (x->score < y->score) return 1;
    return x->id < y->id ? -1 : x->id > y->id;
}
static inline void glm53f_pool_heap_down(glm53f_pool_score *h, int n, int p) {
    for (;;) {
        int w = p, l = 2 * p + 1, r = l + 1;
        if (l < n && glm53f_pool_cmp(h + l, h + w) > 0) w = l;
        if (r < n && glm53f_pool_cmp(h + r, h + w) > 0) w = r;
        if (w == p) return;
        glm53f_pool_score t = h[p]; h[p] = h[w]; h[w] = t; p = w;
    }
}
static inline void glm53f_pool_top_heap(glm53f_pool_score *h,
        const float *score, int np, int nc) {
    for (int p = 0; p < nc; ++p) h[p] = (glm53f_pool_score){score[p], p};
    for (int p = nc / 2; p-- > 0;) glm53f_pool_heap_down(h, nc, p);
    for (int p = nc; p < np; ++p) {
        glm53f_pool_score x = {score[p], p};
        if (glm53f_pool_cmp(&x, h) < 0) { h[0] = x; glm53f_pool_heap_down(h, nc, 0); }
    }
    qsort(h, nc, sizeof(*h), glm53f_pool_cmp);
}
static inline int glm53f_pool_id_cmp(const float *score, int a, int b) {
    if (score[a] > score[b]) return -1;
    if (score[a] < score[b]) return 1;
    return a < b ? -1 : a > b;
}
/* IDs occupy np ints in a separate scratch buffer; the output needs only nc
 * entries. No score arithmetic changes. NaNs retain the legacy heap path.
 * Stop after bounded partition work and fall back to the heap on bad pivots. */
static inline void glm53f_pool_top_partition(glm53f_pool_score *h,
        const float *score, int np, int nc, int *ids) {
    if (np <= nc) { glm53f_pool_top_heap(h, score, np, nc); return; }
    for (int p = 0; p < np; ++p) {
        uint32_t bits;
        memcpy(&bits, score + p, sizeof(bits));
        if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) {
            glm53f_pool_top_heap(h, score, np, nc); return;
        }
        ids[p] = p;
    }
    int lo = 0, hi = np - 1, depth = 0;
    for (int n = np; n > 1; n >>= 1) depth += 2;
    while (lo < hi) {
        if (!depth--) { glm53f_pool_top_heap(h, score, np, nc); return; }
        int x = ids[lo], y = ids[lo + (hi - lo) / 2], z = ids[hi];
        if (glm53f_pool_id_cmp(score, x, y) > 0) { int t = x; x = y; y = t; }
        if (glm53f_pool_id_cmp(score, y, z) > 0) { int t = y; y = z; z = t; }
        if (glm53f_pool_id_cmp(score, x, y) > 0) y = x;
        int i = lo, j = hi;
        while (i <= j) {
            while (glm53f_pool_id_cmp(score, ids[i], y) < 0) ++i;
            while (glm53f_pool_id_cmp(score, ids[j], y) > 0) --j;
            if (i <= j) { int t = ids[i]; ids[i++] = ids[j]; ids[j--] = t; }
        }
        if (nc - 1 <= j) hi = j;
        else if (nc - 1 >= i) lo = i;
        else break;
    }
    for (int p = 0; p < nc; ++p) h[p] = (glm53f_pool_score){score[ids[p]], ids[p]};
    qsort(h, nc, sizeof(*h), glm53f_pool_cmp);
}
#endif
