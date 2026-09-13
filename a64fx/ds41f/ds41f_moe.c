#include "ds41f_moe.h"

#include <math.h>

int ds41f_route_topk(const float *logits, int n, int k, ds41f_route *out)
{
    if (!logits || !out || n <= 0 || k <= 0 || k > n) return -1;
    for (int i = 0; i < k; ++i) out[i] = (ds41f_route){ -1, -INFINITY, i };
    for (int e = 0; e < n; ++e) {
        float score = logits[e];
        int pos = k;
        for (int i = 0; i < k; ++i) {
            if (score > out[i].score ||
                (score == out[i].score && e < out[i].expert)) { pos = i; break; }
        }
        if (pos == k) continue;
        for (int i = k - 1; i > pos; --i) out[i] = out[i - 1];
        out[pos] = (ds41f_route){ e, score, pos };
    }
    return 0;
}

int ds41f_route_owner(const ds41f_route *routes, int n, int rank, int ranks,
                      ds41f_route *out)
{
    if (!routes || !out || n < 0 || rank < 0 || ranks <= 0 || rank >= ranks) return -1;
    int m = 0;
    for (int i = 0; i < n; ++i)
        if (ds41f_expert_owner(routes[i].expert, rank, ranks) >= 0) out[m++] = routes[i];
    return m;
}

void ds41f_moe_weighted_add(float *dst, const float *expert_out,
                            const ds41f_route *routes, int n, size_t hidden)
{
    for (int i = 0; i < n; ++i)
        for (size_t j = 0; j < hidden; ++j)
            dst[j] += routes[i].score * expert_out[(size_t)i * hidden + j];
}
