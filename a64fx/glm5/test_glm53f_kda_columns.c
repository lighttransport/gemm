#define _GNU_SOURCE
#include "../../common/glm53f_ref.h"
#include "glm53f_team.h"
#include "glm53f_kda_columns_sve.h"
#include "glm53f_kda_prefill.h"
#include <arm_sve.h>
#include <omp.h>
#include <stdio.h>
#include <stdint.h>

enum { D = 128, MAX_H = 6, MAX_T = 129, GUARD = 64 };
typedef struct {
    int heads, tokens, variant;
    float *state, *packed, *out, *q, *k, *v, *decay, *beta;
    float *work, *factors;
} call;

static void worker(void *context) {
    call *a = context;
    const int heads = a->heads, tokens = a->tokens, qstride = heads * D;
    if (!a->variant) {
#pragma omp for schedule(static)
        for (int h = 0; h < heads; ++h)
            for (int t = 0; t < tokens; ++t) {
                const size_t off = (size_t)t * qstride + h * D;
                glm53f_kda_step_vec_streamed(a->state + (size_t)h * D * D,
                    a->q + off, a->k + off, a->v + off, a->decay + off,
                    a->beta[(size_t)t * heads + h], D, D, a->out + off,
                    a->work + h * D);
            }
    } else if (a->variant == 1 || a->variant == 4) {
#pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < heads; ++h)
            for (int t = 0; t < tokens; ++t)
                for (int d = 0; d < D; ++d) {
                    const size_t i = ((size_t)t * heads + h) * D + d;
                    a->factors[i] = glm53f_kda_scalar_factor(a->decay[i]);
                }
        if (a->variant == 4) {   /* decode 16-column blocks */
#pragma omp for collapse(2) schedule(static)
            for (int h = 0; h < heads; ++h)
                for (int b = 0; b < 8; ++b)
                    for (int t = 0; t < tokens; ++t) {
                        const size_t off = (size_t)t * qstride + h * D;
                        glm53f_kda_columns16_sve(a->state + (size_t)h * D * D + b * 16,
                            a->out + off + b * 16, a->q + off, a->k + off,
                            a->v + off + b * 16, a->factors + off,
                            a->beta[(size_t)t * heads + h]);
                    }
        } else
#pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < heads; ++h)
            for (int b = 0; b < 2; ++b)
                for (int t = 0; t < tokens; ++t) {
                    const size_t off = (size_t)t * qstride + h * D;
                    glm53f_kda_columns64_sve(a->state + (size_t)h * D * D + b * 64,
                        a->out + off + b * 64, a->q + off, a->k + off,
                        a->v + off + b * 64, a->factors + off,
                        a->beta[(size_t)t * heads + h]);
                }
    } else if (a->variant >= 2) {
        /* The existing prefill decay preparation is intentionally retained. */
#pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < heads; ++h)
            for (int t = 0; t < tokens; ++t)
                for (int d = 0; d < D; ++d) {
                    const size_t i = ((size_t)t * heads + h) * D + d;
                    a->factors[i] = expf(a->decay[i]);
                }
        if (a->variant == 2) {
#pragma omp for collapse(2) schedule(static)
            for (int h = 0; h < heads; ++h)
                for (int b = 0; b < 8; ++b) {
                    float *state = a->packed + ((size_t)h * 8 + b) * D * 16;
                    for (int d = 0; d < D; ++d)
                        memcpy(state + d * 16,
                            a->state + ((size_t)h * D + d) * D + b * 16, 16 * sizeof(float));
                    glm53f_kda_column_tile(state, a->out + h * D + b * 16,
                        a->q + h * D, a->k + h * D, a->v + h * D + b * 16,
                        a->factors + h * D, a->beta + h, tokens, qstride, heads);
                }
#pragma omp for collapse(2) schedule(static)
            for (int h = 0; h < heads; ++h)
                for (int d = 0; d < D; ++d)
                    for (int b = 0; b < 8; ++b)
                        memcpy(a->state + ((size_t)h * D + d) * D + b * 16,
                            a->packed + ((size_t)h * 8 + b) * D * 16 + d * 16, 16 * sizeof(float));
        } else {
#pragma omp for collapse(2) schedule(static)
            for (int h = 0; h < heads; ++h)
                for (int b = 0; b < 2; ++b)
                    for (int t = 0; t < tokens; ++t) {
                        const size_t off = (size_t)t * qstride + h * D;
                        glm53f_kda_columns64_sve(a->state + (size_t)h * D * D + b * 64,
                            a->out + off + b * 64, a->q + off, a->k + off,
                            a->v + off + b * 64, a->factors + off,
                            a->beta[(size_t)t * heads + h]);
                    }
        }
    }
}

static void run(call *a) {
    if (glm53f_team_available()) glm53f_team_dispatch(worker, a);
    else {
#pragma omp parallel
        worker(a);
    }
}
static void *alloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 256, n)) return NULL;
    memset(p, 0, n);
    return p;
}
static uint32_t rng = 1;
static float value(int mode, int role) {
    rng = rng * 1664525u + 1013904223u;
    float x = (float)((int)(rng >> 8) - 8388608) / 8388608.0f;
    if (mode == 0) x = 0;
    if (mode == 2) x *= 0x1p-20f;
    if (mode == 3) x = x < 0 ? -1.0f : 1.0f;
    if (mode == 4) x = (rng & 7) ? 0 : x * 2;
    if (mode == 5) x = (rng & 1) ? -0.0f : 0.0f;
    if (mode == 6) x = nextafterf(x, 0.0f);
    if (mode == 7) x = (float)((int)(rng >> 27) - 16) * .0625f;
    if (mode == 8) x *= 0x1p-40f;
    if (mode == 9) x *= 0x1p-100f;
    if (mode == 10) x = x < 0 ? -0x1.fffffep-1f : 0x1.fffffep-1f;
    if (mode == 11) x = (float)((int)(rng & 4095) - 2048) * 0x1p-11f;
    if (role == 1) return -fabsf(x) * 5;
    if (role == 2) return fminf(1.0f, fabsf(x));
    if (role == 3) return x * .03f;
    return x * .125f;
}
static void fill(call *a, int mode) {
    rng = 1;
    size_t n = (size_t)a->tokens * a->heads * D;
    for (size_t i = 0; i < n; ++i) {
        a->q[i] = value(mode, 0); a->k[i] = value(mode, 0);
        a->v[i] = value(mode, 0); a->decay[i] = value(mode, 1);
    }
    for (int i = 0; i < a->tokens * a->heads; ++i) a->beta[i] = value(mode, 2);
    for (int i = 0; i < a->heads * D * D; ++i) a->state[i] = value(mode, 3);
}
static int exact(const char *label, const float *a, const float *b, size_t n,
        int heads, int tokens, int mode, int variant) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t af, bf; memcpy(&af, a + i, 4); memcpy(&bf, b + i, 4);
        if ((af & 0x7f800000u) == 0x7f800000u || (bf & 0x7f800000u) == 0x7f800000u ||
            memcmp(a + i, b + i, sizeof(float))) {
            uint32_t av, bv; memcpy(&av, a + i, 4); memcpy(&bv, b + i, 4);
            printf("GLM53F_KDA_COLUMNS_FAIL label=%s heads=%d tokens=%d mode=%d variant=%d index=%zu actual=%08x expected=%08x\n",
                label, heads, tokens, mode, variant, i, av, bv);
            return 1;
        }
    }
    return 0;
}
static uint64_t hash(const float *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char *q = (const unsigned char *)p;
    for (size_t i = 0; i < n * sizeof(float); ++i) h = (h ^ q[i]) * 1099511628211ull;
    return h;
}
typedef struct { call *a; int rounds; double seconds; } timing;
static void time_controller(void *context) {
    timing *x = context;
    const double begin = omp_get_wtime();
    for (int r = 0; r < x->rounds; ++r) run(x->a);
    x->seconds = omp_get_wtime() - begin;
}
int main(int argc, char **argv) {
    const size_t sn = MAX_H * D * D, qn = (size_t)MAX_T * MAX_H * D;
    float *state = alloc((sn + 2 * GUARD) * 4), *packed = alloc((sn + 2 * GUARD) * 4);
    float *out = alloc((qn + 2 * GUARD) * 4), *ref = alloc(qn * 4), *saved = alloc(sn * 4);
    call a = {MAX_H, MAX_T, 0, state + GUARD, packed + GUARD, out + GUARD,
        alloc(qn * 4), alloc(qn * 4), alloc(qn * 4), alloc(qn * 4),
        alloc(MAX_T * MAX_H * 4), alloc(MAX_H * D * 4), alloc(qn * 4)};
    if (!state || !packed || !out || !ref || !saved || !a.q || !a.k || !a.v ||
        !a.decay || !a.beta || !a.work || !a.factors) return 2;
    int cases = 0, failed = 0;
    const int lengths[] = {1, 4, 5, 8, 16, 31, 32, 47, 63, 64, 129};
    for (int heads = 5; heads <= 6; ++heads)
        for (int mode = 0; mode < 12; ++mode)
            for (size_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) {
                a.heads = heads; a.tokens = lengths[l]; a.variant = 0;
                fill(&a, mode); run(&a);
                memcpy(saved, a.state, (size_t)heads * D * D * 4);
                memcpy(ref, a.out, (size_t)a.tokens * heads * D * 4);
                for (int variant = 1; variant <= 4; variant += 3) {
                    a.variant = variant; fill(&a, mode);
                    for (int g = 0; g < GUARD; ++g) {
                        state[g] = packed[g] = out[g] = -71.25f;
                        a.state[(size_t)heads * D * D + g] = -72.25f;
                        a.packed[(size_t)heads * D * D + g] = -73.25f;
                        a.out[(size_t)a.tokens * heads * D + g] = -74.25f;
                    }
                    run(&a);
                    failed |= exact("state", a.state, saved, (size_t)heads * D * D,
                        heads, a.tokens, mode, variant);
                    failed |= exact("output", a.out, ref, (size_t)a.tokens * heads * D,
                        heads, a.tokens, mode, variant);
                    for (int g = 0; g < GUARD; ++g)
                        if (state[g] != -71.25f || packed[g] != -71.25f || out[g] != -71.25f ||
                            a.state[(size_t)heads * D * D + g] != -72.25f ||
                            a.packed[(size_t)heads * D * D + g] != -73.25f ||
                            a.out[(size_t)a.tokens * heads * D + g] != -74.25f) failed = 1;
                    ++cases;
                }
                a.variant = 2; fill(&a, mode); run(&a);
                memcpy(saved, a.state, (size_t)heads * D * D * 4);
                memcpy(ref, a.out, (size_t)a.tokens * heads * D * 4);
                a.variant = 3; fill(&a, mode);
                for (int g = 0; g < GUARD; ++g) {
                    state[g] = packed[g] = out[g] = -71.25f;
                    a.state[(size_t)heads * D * D + g] = -72.25f;
                    a.packed[(size_t)heads * D * D + g] = -73.25f;
                    a.out[(size_t)a.tokens * heads * D + g] = -74.25f;
                }
                run(&a);
                failed |= exact("prefill-state", a.state, saved, (size_t)heads * D * D,
                    heads, a.tokens, mode, 3);
                failed |= exact("prefill-output", a.out, ref, (size_t)a.tokens * heads * D,
                    heads, a.tokens, mode, 3);
                for (int g = 0; g < GUARD; ++g)
                    if (state[g] != -71.25f || packed[g] != -71.25f || out[g] != -71.25f ||
                        a.state[(size_t)heads * D * D + g] != -72.25f ||
                        a.packed[(size_t)heads * D * D + g] != -73.25f ||
                        a.out[(size_t)a.tokens * heads * D + g] != -74.25f) failed = 1;
                ++cases;
            }
    printf("GLM53F_KDA_COLUMNS_UNIT threads=%d cases=%d state_and_output=BIT_EXACT guards=exact %s\n",
        omp_get_max_threads(), cases, failed ? "FAIL" : "PASS");
    if (failed || argc < 2 || strcmp(argv[1], "--bench")) return failed;
    const int bench_lengths[] = {1, 4, 32, 47, 64};
    for (int heads = 5; heads <= 6; ++heads)
        for (size_t l = 0; l < sizeof(bench_lengths) / sizeof(bench_lengths[0]); ++l) {
            a.heads = heads; a.tokens = bench_lengths[l];
            for (int trial = -1; trial < 7; ++trial)
                for (int order = 0; order < 4; ++order) {
                    a.variant = (order + (trial + 1)) % 4;
                    fill(&a, 1);
                    timing x = {&a, a.tokens < 5 ? 90 : 1, 0};
                    if (a.tokens < 5) glm53f_team_run(time_controller, &x);
                    else time_controller(&x);
                    if (trial >= 0)
                        printf("GLM53F_KDA_COLUMNS_BENCH threads=%d heads=%d tokens=%d variant=%d trial=%d us_token=%.6f state_hash=%016llx output_hash=%016llx\n",
                            omp_get_max_threads(), heads, a.tokens, a.variant, trial,
                            x.seconds * 1e6 / (x.rounds * a.tokens),
                            (unsigned long long)hash(a.state, (size_t)heads * D * D),
                            (unsigned long long)hash(a.out, (size_t)a.tokens * heads * D));
                }
        }
    free(state); free(packed); free(out); free(saved); free(ref);
    free(a.q); free(a.k); free(a.v); free(a.decay); free(a.beta); free(a.work); free(a.factors);
    return 0;
}
