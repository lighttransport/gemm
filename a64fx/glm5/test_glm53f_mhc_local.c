/* --mhc-kernel local versus the legacy mHC team on chained synthetic sites:
 * stream updates must be bit-identical for identical inputs; coefficients,
 * normalized outputs and the chained state must agree within float
 * reduction-order tolerance. Also times both paths.
 * run: OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores ./test_glm53f_mhc_local */
#define _GNU_SOURCE
#include "glm53f_mhc_sve.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { SITES = 8, CALLS = 24 };
static unsigned rng = 7;
static float rnd(void) { rng = rng * 1664525u + 1013904223u; return ((int)(rng >> 9) - (1 << 22)) / (float)(1 << 22); }
static uint16_t bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static double rel(const float *a, const float *b, int n) {
    double e = 0, r = 0;
    for (int i = 0; i < n; ++i) { double d = (double)a[i] - b[i]; e += d * d; r += (double)b[i] * b[i]; }
    return sqrt(e / (r + 1e-300));
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

typedef struct { float streams[GLM53F_MHC_FLAT]; glm53f_mhc_scratch scratch; float logits[GLM53F_MHC_MIX]; } state;

static void run(state *s, const glm53f_mhc_site *site, const uint16_t *norm, const float *sublayer, int do_post, int mode) {
    _Atomic int pending;
    atomic_init(&pending, 0);
#pragma omp parallel
    glm53f_mhc_fast_team(s->streams, sublayer, &s->scratch, site, norm, do_post, s->logits, NULL, NULL, mode, &pending);
}

int main(void) {
    static uint16_t fn[SITES][GLM53F_MHC_MIX * GLM53F_MHC_FLAT], norm[GLM53F_MHC_WIDTH];
    static float base[SITES][GLM53F_MHC_MIX], scale[SITES][3], sub[CALLS][GLM53F_MHC_WIDTH];
    glm53f_mhc_site site[SITES];
    for (int s = 0; s < SITES; ++s) {
        for (size_t i = 0; i < (size_t)GLM53F_MHC_MIX * GLM53F_MHC_FLAT; ++i) fn[s][i] = bf16(rnd() * 0.02f);
        for (int i = 0; i < GLM53F_MHC_MIX; ++i) base[s][i] = rnd();
        for (int i = 0; i < 3; ++i) scale[s][i] = 0.5f + 0.5f * rnd();
        site[s] = (glm53f_mhc_site){fn[s], base[s], scale[s]};
    }
    for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) norm[i] = bf16(1.0f + 0.1f * rnd());
    for (int c = 0; c < CALLS; ++c) for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) sub[c][i] = rnd();
    static state ref, loc, probe;
    for (int i = 0; i < GLM53F_MHC_FLAT; ++i) ref.streams[i] = rnd();
    run(&ref, &site[0], norm, sub[0], 0, 0);   /* prime residual/post/combine */
    loc = ref;
    double worst_norm = 0, worst_coef = 0, worst_stream = 0;
    int stream_exact = 1;
    for (int c = 1; c < CALLS; ++c) {
        /* identical-input probe: from the legacy state, both kernels must produce the same streams */
        probe = ref;
        run(&probe, &site[c % SITES], norm, sub[c], 1, 2);
        state legacy_next = ref;
        run(&legacy_next, &site[c % SITES], norm, sub[c], 1, 0);
        stream_exact &= !memcmp(probe.streams, legacy_next.streams, sizeof(probe.streams));
        if (c == 1) {
            int dl = 0, dn = 0;
            for (int m = 0; m < GLM53F_MHC_MIX; ++m) dl += memcmp(&probe.logits[m], &legacy_next.logits[m], 4) != 0;
            for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) dn += memcmp(&probe.scratch.normalized[i], &legacy_next.scratch.normalized[i], 4) != 0;
            double pabs = 0; for (int i = 0; i < 128 * 32; ++i) pabs += fabs(glm53f_mhc_partl[i]);
            printf("MHC_LOCAL_PATH partials_abs_sum=%g (nonzero => local path ran)\n", pabs);
            printf("MHC_LOCAL_DIFF logits_bits_differ=%d/%d normalized_bits_differ=%d/%d logit0=%a/%a\n", dl, GLM53F_MHC_MIX, dn,
                   GLM53F_MHC_WIDTH, probe.logits[0], legacy_next.logits[0]);
        }
        const double en = rel(probe.scratch.normalized, legacy_next.scratch.normalized, GLM53F_MHC_WIDTH);
        const double ec = rel(probe.scratch.combine, legacy_next.scratch.combine, 16);
        if (en > worst_norm) worst_norm = en;
        if (ec > worst_coef) worst_coef = ec;
        /* chained: each kernel on its own state */
        ref = legacy_next;
        run(&loc, &site[c % SITES], norm, sub[c], 1, 2);
        const double es = rel(loc.streams, ref.streams, GLM53F_MHC_FLAT);
        if (es > worst_stream) worst_stream = es;
    }
    const int ok = stream_exact && worst_norm < 1e-5 && worst_coef < 1e-5 && worst_stream < 1e-4;
    printf("MHC_LOCAL_CHECK threads=%d streams_bit_exact=%d worst_normalized_rel=%.2e worst_combine_rel=%.2e "
           "chained_stream_rel=%.2e %s\n", omp_get_max_threads(), stream_exact, worst_norm, worst_coef, worst_stream,
           ok ? "PASS" : "FAIL");
    for (int mode = 0; mode <= 2; mode += 2) {
        state t = ref;
        double best = 1e30, sum = 0;
        const int iters = 400;
        for (int it = -20; it < iters; ++it) {
            const double t0 = now();
            run(&t, &site[(it + 20) % SITES], norm, sub[(it + 20) % CALLS], 1, mode);
            const double dt = now() - t0;
            if (it >= 0) { sum += dt; if (dt < best) best = dt; }
        }
        printf("MHC_LOCAL_TIME mode=%s mean_us=%.2f best_us=%.2f\n", mode ? "local" : "legacy", sum / iters * 1e6, best * 1e6);
    }
    return ok ? 0 : 1;
}
