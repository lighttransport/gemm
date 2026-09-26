#ifndef Q38P_PARTITION_H
#define Q38P_PARTITION_H
#include <math.h>
#define Q38P_MAX_STAGES 128
/* Positive unit costs; every stage owns a nonempty contiguous range.
 * Returns zero for invalid inputs. Caller supplies n+1 cut entries. */
static int q38p_partition(const double *weights, int units, int n, int *cut) {
    if (units < 1 || units > Q38P_MAX_STAGES || n < 1 || n > units) return 0;
    double prefix[Q38P_MAX_STAGES + 1] = {0};
    for (int u = 0; u < units; u++) {
        if (!(weights[u] > 0) || !isfinite(weights[u])) return 0;
        prefix[u + 1] = prefix[u] + weights[u];
        if (!isfinite(prefix[u + 1])) return 0;
    }
    /* Minimize the slowest contiguous stage, rather than rounding cumulative
     * targets independently (which can bunch two costly attention units). */
    double best[Q38P_MAX_STAGES + 1][Q38P_MAX_STAGES + 1];
    int split[Q38P_MAX_STAGES + 1][Q38P_MAX_STAGES + 1];
    for (int k = 0; k <= n; k++)
        for (int u = 0; u <= units; u++) best[k][u] = HUGE_VAL;
    best[0][0] = 0;
    for (int k = 1; k <= n; k++)
        for (int u = k; u <= units; u++)
            for (int v = k - 1; v < u; v++) {
                double cost = fmax(best[k - 1][v], prefix[u] - prefix[v]);
                if (cost < best[k][u]) { best[k][u] = cost; split[k][u] = v; }
            }
    cut[n] = units;
    for (int k = n; k > 0; k--) cut[k - 1] = split[k][cut[k]];
    return 1;
}
#endif
