#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../common/glm53f_ref.h"
#include "glm53f_kda_prefill.h"

static int check(const char *label, const float *a, const float *b, size_t n) {
    double error = 0, norm = 0, maximum = 0, reference = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - b[i];
        if (!isfinite(a[i]) || !isfinite(b[i])) return 1;
        error += d * d; norm += (double)b[i] * b[i];
        if (fabs(d) > maximum) maximum = fabs(d);
        if (fabs(b[i]) > reference) reference = fabs(b[i]);
    }
    double relative = sqrt(error / fmax(norm, 1e-30));
    int fail = relative > 1e-4 || maximum > 1e-6 + 5e-4 * reference;
    printf("KDA_COLUMN %s rel_l2=%.9g max_abs=%.9g %s\n", label, relative, maximum, fail ? "FAIL" : "PASS");
    return fail;
}

int main(void) {
    enum {D = 128, T = 33, H = 6, Q = H * D};
    float *q = malloc(T * Q * sizeof(float)), *k = malloc(T * Q * sizeof(float));
    float *v = malloc(T * Q * sizeof(float)), *logd = malloc(T * Q * sizeof(float));
    float *decay = malloc(T * Q * sizeof(float)), *beta = malloc(T * H * sizeof(float));
    float *a = malloc(H * D * D * sizeof(float)), *b = malloc(H * D * D * sizeof(float));
    float *out = malloc(T * Q * sizeof(float)), *ref = malloc(T * Q * sizeof(float));
    if (!q || !k || !v || !logd || !decay || !beta || !a || !b || !out || !ref) return 2;
    int failed = 0;
    const int lengths[] = {1, 4, 5, 8, 16, 31, 32, 33};
    for (int zero = 0; zero < 2; ++zero) {
        for (int i = 0; i < T * Q; ++i) {
            q[i] = zero ? 0 : sinf(i * .31f) * .12f;
            k[i] = zero ? 0 : cosf(i * .23f) * .12f;
            v[i] = zero ? 0 : sinf(i * .17f);
            logd[i] = -(i % 53) * .07f; decay[i] = expf(logd[i]);
        }
        for (int i = 0; i < T * H; ++i) beta[i] = (i % 17) / 16.0f;
        for (size_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) {
            int tokens = lengths[l];
            for (int i = 0; i < H * D * D; ++i) a[i] = b[i] = zero ? 0 : sinf(i * .19f) * .03f;
            for (int t = 0; t < tokens; ++t)
                for (int h = 0; h < H; ++h) {
                    float work[D]; size_t off = (size_t)t * Q + h * D;
                    glm53f_kda_step_vec_streamed(b + h * D * D, q + off, k + off,
                        v + off, logd + off, beta[t * H + h], D, D, ref + off, work);
                }
#pragma omp parallel for collapse(2) schedule(static)
            for (int h = 0; h < H; ++h)
                for (int block = 0; block < 8; ++block) {
                    float packed[D * 16];
                    for (int d = 0; d < D; ++d)
                        memcpy(packed + d * 16, a + (h * D + d) * D + block * 16, 16 * sizeof(float));
                    glm53f_kda_column_tile(packed, out + h * D + block * 16,
                        q + h * D, k + h * D, v + h * D + block * 16,
                        decay + h * D, beta + h, tokens, Q, H);
                    for (int d = 0; d < D; ++d)
                        memcpy(a + (h * D + d) * D + block * 16, packed + d * 16, 16 * sizeof(float));
                }
            failed |= check("state", a, b, H * D * D);
            failed |= check("output", out, ref, tokens * Q);
        }
    }
    free(q); free(k); free(v); free(logd); free(decay); free(beta);
    free(a); free(b); free(out); free(ref);
    return failed;
}
