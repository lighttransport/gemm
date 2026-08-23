#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void kernel_q8v2_6x2(const int8_t *, const float *, const int8_t *,
                     const float *, long, float *, long);

static double sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int nb = argc > 1 ? atoi(argv[1]) : 160;
    int reps = argc > 2 ? atoi(argv[2]) : 1000;
    const int mr = 6, nr = 32, blk = 32;
    size_t na = (size_t)nb * mr * blk;
    size_t nbq = (size_t)nb * 8 * 4 * 64;
    int8_t *aq = aligned_alloc(256, na), *bq = aligned_alloc(256, nbq);
    float *ad = aligned_alloc(256, (size_t)nb * mr * sizeof(float));
    float *bd = aligned_alloc(256, (size_t)nb * 64 * sizeof(float));
    float *c = aligned_alloc(256, (size_t)mr * nr * sizeof(float));
    if (!aq || !bq || !ad || !bd || !c) return 2;
    for (size_t i = 0; i < na; i++) aq[i] = (int8_t)((i * 17 + 5) % 255 - 127);
    for (size_t i = 0; i < nbq; i++) bq[i] = (int8_t)((i * 29 + 3) % 255 - 127);
    for (int b = 0; b < nb; b++) {
        for (int r = 0; r < mr; r++) ad[b * mr + r] = 0.001f * (1 + (b + r) % 13);
        for (int n = 0; n < 64; n++) bd[b * 64 + n] = 0.002f * (1 + (b + n) % 11);
    }
    kernel_q8v2_6x2(aq, ad, bq, bd, nb, c, nr * 4L);
    float maxrel = 0.0f;
    for (int r = 0; r < mr; r++) for (int n = 0; n < nr; n++) {
        float ref = 0.0f;
        for (int b = 0; b < nb; b++) {
            int dot = 0;
            for (int k = 0; k < blk; k++) {
                int g = k / 4, kk = k % 4, vec = n / 16, col = n % 16;
                size_t bo = ((size_t)b * 8 * 4 + (size_t)g * 4 + vec) * 64 + col * 4 + kk;
                dot += aq[((size_t)b * mr + r) * blk + k] * bq[bo];
            }
            ref += dot * ad[b * mr + r] * bd[b * 64 + n];
        }
        float rel = fabsf(c[r * nr + n] - ref) / (fabsf(ref) + 1e-6f);
        if (rel > maxrel) maxrel = rel;
    }
    double t0 = sec();
    for (int i = 0; i < reps; i++) kernel_q8v2_6x2(aq, ad, bq, bd, nb, c, nr * 4L);
    double dt = sec() - t0;
    double ops = 2.0 * mr * nr * nb * blk * reps;
    printf("q8v2_6x2 K=%d %.1f GIOPS maxrel=%.3g\n", nb * blk, ops / dt / 1e9, maxrel);
    free(aq); free(bq); free(ad); free(bd); free(c);
    return maxrel < 1e-4f ? 0 : 1;
}
