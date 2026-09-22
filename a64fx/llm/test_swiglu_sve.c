#include "swiglu_sve.h"
#include <math.h>
#include <stdio.h>
int main(void) {
    float g[257], u[257], y[257];
    double max_abs = 0, max_rel = 0;
    for (int trial = 0; trial < 1000; trial++) {
        int n = 1 + trial % 257;
        for (int i = 0; i < n; i++) {
            g[i] = ((trial * 257 + i) % 20001 - 10000) * 0.01f;
            u[i] = ((trial * 17 + i) % 1001 - 500) * 0.004f;
        }
        tf_swiglu_approx_sve(y, g, u, n);
        for (int i = 0; i < n; i++) {
            float ref = g[i] / (1.0f + expf(-g[i])) * u[i];
            double e = fabs(y[i] - ref);
            if (e > max_abs) max_abs = e;
            if (fabs(ref) > 1e-6 && e / fabs(ref) > max_rel) max_rel = e / fabs(ref);
            if (!isfinite(y[i]) || e > 1e-6 + fabs(ref) * 1e-6) {
                printf("FAIL g=%g u=%g got=%g ref=%g\n", g[i], u[i], y[i], ref);return 1;
            }
        }
    }
    for (int trial = 0; trial < 1000; trial++) {
        int n = 1 + trial % 257;
        for (int i = 0; i < n; i++) {
            g[i] = ((trial * 257 + i) % 20001 - 10000) * 0.01f;
            y[i] = g[i];
        }
        tf_swiglu_approx_sve(y, y, NULL, n);
        for (int i = 0; i < n; i++) {
            float ref = g[i] / (1.0f + expf(-g[i]));
            if (!isfinite(y[i]) || fabs(y[i] - ref) > 1e-6 + fabs(ref) * 1e-6) {
                printf("SiLU FAIL g=%g got=%g ref=%g\n", g[i], y[i], ref);
                return 1;
            }
        }
    }
    puts("SiLU in-place approximation PASS, n=1..257");
    printf("SwiGLU approximation PASS max_abs=%.9g max_rel=%.9g, n=1..257\n", max_abs, max_rel);
    return 0;
}
