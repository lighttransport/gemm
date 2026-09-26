#include "q38p_partition.h"
#include <assert.h>
#include <stdio.h>
/* Exhaustively enumerate all partitions for small inputs. */
static double brute(const double *w, int units, int n) {
    if (n == 1) {
        double sum = 0;
        for (int i = 0; i < units; i++) sum += w[i];
        return sum;
    }
    double sum = 0, best = HUGE_VAL;
    for (int i = 1; i <= units - n + 1; i++) {
        sum += w[i - 1];
        double cost = fmax(sum, brute(w + i, units - i, n - 1));
        if (cost < best) best = cost;
    }
    return best;
}
static double check(const double *w, int units, int n) {
    int cuts[Q38P_MAX_STAGES + 1];
    assert(q38p_partition(w, units, n, cuts));
    assert(cuts[0] == 0 && cuts[n] == units);
    double worst = 0;
    for (int r = 0; r < n; r++) {
        assert(cuts[r] < cuts[r + 1]);
        double cost = 0;
        for (int u = cuts[r]; u < cuts[r + 1]; u++) cost += w[u];
        worst = fmax(worst, cost);
    }
    return worst;
}
int main(void) {
    double w[128];
    for (int variant = 0; variant < 8; variant++) {
        for (int i = 0; i < 12; i++) w[i] = 1 + ((i * 17 + variant * 13) % 23);
        for (int units = 1; units <= 12; units++)
            for (int n = 1; n <= units; n++)
                assert(check(w, units, n) == brute(w, units, n));
    }
    for (int i = 0; i < 128; i++) w[i] = i % 2 ? 566.1 : i % 8 == 6 ? 3500 : 294.8;
    for (int n = 1; n <= 128; n++) check(w, 128, n);
    int cuts[129];
    assert(!q38p_partition(w, 128, 129, cuts));
    assert(!q38p_partition(w, 0, 1, cuts));
    w[0] = NAN; assert(!q38p_partition(w, 128, 12, cuts));
    w[0] = INFINITY; assert(!q38p_partition(w, 128, 12, cuts));
    puts("PASS: minimax partition versus exhaustive reference; 1..128 ranks cover every unit once");
    return 0;
}
