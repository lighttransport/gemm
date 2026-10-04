#ifndef GLM53F_MHC_SVE_H
#define GLM53F_MHC_SVE_H

#include <arm_sve.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdatomic.h>
/* Optional decode profile (GLM53F_MHC_DETAIL=1): seconds per stage, printed at exit. */
static double glm53f_mhc_acc[8]; static long glm53f_mhc_calls[2]; static int glm53f_mhc_detail = -1;
static inline double glm53f_mhc_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void glm53f_mhc_report(void) {
    if (glm53f_mhc_calls[0])
        fprintf(stderr, "GLM53F_MHC_DETAIL us per pre-call: sumsq=%.1f logits=%.1f sigmoid+sinkhorn=%.1f collapse=%.1f copy+rmsnorm=%.1f | post=%.1f (pre calls=%ld post calls=%ld)\n",
                glm53f_mhc_acc[0] * 1e6 / glm53f_mhc_calls[0], glm53f_mhc_acc[1] * 1e6 / glm53f_mhc_calls[0], glm53f_mhc_acc[2] * 1e6 / glm53f_mhc_calls[0],
                glm53f_mhc_acc[3] * 1e6 / glm53f_mhc_calls[0], glm53f_mhc_acc[4] * 1e6 / glm53f_mhc_calls[0],
                glm53f_mhc_calls[1] ? glm53f_mhc_acc[5] * 1e6 / glm53f_mhc_calls[1] : 0.0, glm53f_mhc_calls[0], glm53f_mhc_calls[1]);
}
static inline int glm53f_mhc_detail_on(void) {
    if (glm53f_mhc_detail < 0) { glm53f_mhc_detail = getenv("GLM53F_MHC_DETAIL") != NULL; if (glm53f_mhc_detail) atexit(glm53f_mhc_report); }
    return glm53f_mhc_detail;
}
#include "glm53f_prefill.h"
#include "glm53f_team.h"
#include "glm53f_pf_plan.h"
#include "../../common/glm53f_ref.h"

/* The default remains the validated implementation.  The fused variant keeps
 * the mHC norm reduction and 24-row projection in one OpenMP team, avoiding a
 * fork/join on every mHC invocation during scalar decode. */
#ifndef GLM53F_MHC_FUSED
#define GLM53F_MHC_FUSED 0
#endif
#ifndef GLM53F_MHC_POST_FLOAT
#define GLM53F_MHC_POST_FLOAT 0
#endif

enum {
    GLM53F_MHC_STREAMS = 4,
    GLM53F_MHC_WIDTH = 4096,
    GLM53F_MHC_FLAT = GLM53F_MHC_STREAMS * GLM53F_MHC_WIDTH,
    GLM53F_MHC_MIX = (2 + GLM53F_MHC_STREAMS) * GLM53F_MHC_STREAMS
};

typedef struct {
    const uint16_t *fn;
    const float *base;
    const float *scale;
} glm53f_mhc_site;

typedef struct {
    float collapsed[GLM53F_MHC_WIDTH];
    float normalized[GLM53F_MHC_WIDTH];
    float residual[GLM53F_MHC_FLAT];
    float post[GLM53F_MHC_STREAMS];
    float combine[GLM53F_MHC_STREAMS * GLM53F_MHC_STREAMS];
} glm53f_mhc_scratch;

static inline float glm53f_mhc_dot_bf16_sve(
        const uint16_t *weight, const float *x, int n) {
    svfloat32_t acc = svdup_f32(0);
    int vl = (int)svcntw();
    for (int i = 0; i < n; i += vl) {
        svbool_t pg = svwhilelt_b32(i, n);
        svuint32_t bits = svlsl_n_u32_x(pg, svld1uh_u32(pg, weight + i), 16);
        acc = svmla_x(pg, acc, svreinterpret_f32_u32(bits), svld1(pg, x + i));
    }
    return svaddv_f32(svptrue_b32(), acc);
}

/* mHC has only 24 projection rows. Keep all rows parallel while reusing each
 * BF16 weight vector across the four verification positions. */
static inline void glm53f_mhc_row_batch4(
        float *out, const uint16_t *weight, const float *input, int tokens,
        int row) {
        svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
        svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
        const uint16_t *w = weight + (size_t)row * GLM53F_MHC_FLAT;
        int vl = (int)svcntw();
        for (int i = 0; i < GLM53F_MHC_FLAT; i += vl) {
            svbool_t pg = svwhilelt_b32(i, GLM53F_MHC_FLAT);
            svuint32_t bits = svlsl_n_u32_x(pg, svld1uh_u32(pg, w + i), 16);
            svfloat32_t wf = svreinterpret_f32_u32(bits);
            a0 = svmla_x(pg, a0, wf, svld1(pg, input + i));
            if (tokens > 1) a1 = svmla_x(pg, a1, wf,
                svld1(pg, input + GLM53F_MHC_FLAT + i));
            if (tokens > 2) a2 = svmla_x(pg, a2, wf,
                svld1(pg, input + 2 * GLM53F_MHC_FLAT + i));
            if (tokens > 3) a3 = svmla_x(pg, a3, wf,
                svld1(pg, input + 3 * GLM53F_MHC_FLAT + i));
        }
        svbool_t all = svptrue_b32();
        out[row] = svaddv_f32(all, a0);
        if (tokens > 1) out[GLM53F_MHC_MIX + row] = svaddv_f32(all, a1);
        if (tokens > 2) out[2 * GLM53F_MHC_MIX + row] = svaddv_f32(all, a2);
        if (tokens > 3) out[3 * GLM53F_MHC_MIX + row] = svaddv_f32(all, a3);
}
static inline void glm53f_mhc_mv_batch4(
        float *out, const uint16_t *weight, const float *input, int tokens) {
#pragma omp parallel for schedule(static)
    for (int row = 0; row < GLM53F_MHC_MIX; ++row)
        glm53f_mhc_row_batch4(out, weight, input, tokens, row);
}


/* ---- fast decode mHC: ONE team, ~5 barriers, no serial 4K-element passes ----------------------------------
 * [post of the previous site] -> per-thread sum of squares of the new streams -> 24 logits -> sigmoid/Sinkhorn
 * (one thread) -> collapse + residual copy + per-thread sum of squares of the collapsed vector -> RMS-normalize.
 * Same formulas as glm53f_mhc_pre_sve / glm53f_mhc_post_pre_sve; only the summation partition of the two
 * double-precision norm reductions differs (thread partials in thread order). */
static double glm53f_mhc_part1[128 * 8], glm53f_mhc_part2[128 * 8];
static inline void glm53f_mhc_sinkhorn_fast(float *comb, int hc, int iters, float eps);
static int glm53f_mhc_fast_mode = -1;
static inline int glm53f_mhc_fused_sync_on(void) {
    /* Read on the controller before publication, also allowing diagnostic
     * reference/candidate switches with one restored resident model. */
    const char *e = getenv("GLM53F_MHC_FUSED_SYNC");
    return e && *e && atoi(e);
}
static inline int glm53f_mhc_fast_on(void) {
    if (glm53f_mhc_fast_mode < 0) glm53f_mhc_fast_mode = getenv("GLM53F_MHC_FAST") ? atoi(getenv("GLM53F_MHC_FAST")) : 1;
    return glm53f_mhc_fast_mode;
}
static inline void glm53f_mhc_coefficients(float *logits,
        glm53f_mhc_scratch *scratch, const glm53f_mhc_site *site) {
    for (int k = 0; k < GLM53F_MHC_STREAMS; ++k) {
        logits[k] = glm53f_sigmoid(logits[k] * site->scale[0] + site->base[k]) + 1e-6f;
        scratch->post[k] = 2.0f * glm53f_sigmoid(logits[GLM53F_MHC_STREAMS + k] * site->scale[1] +
                                                 site->base[GLM53F_MHC_STREAMS + k]);
    }
    for (int m = 0; m < GLM53F_MHC_STREAMS * GLM53F_MHC_STREAMS; ++m)
        scratch->combine[m] = logits[2 * GLM53F_MHC_STREAMS + m] * site->scale[2] + site->base[2 * GLM53F_MHC_STREAMS + m];
    glm53f_mhc_sinkhorn_fast(scratch->combine, GLM53F_MHC_STREAMS, 20, 1e-6f);
}
/* Opt-in locality variant (GLM53F_MHC_FUSED_SYNC=2, --mhc-kernel local).
 * Each thread owns whole 16-column vectors of all four streams for the
 * whole call: post update, sum of squares and partial dots of the 24 mixing
 * rows run on its own slice; after one barrier every thread reduces the
 * partials in thread order and computes sigmoid/Sinkhorn itself (no single
 * section), then collapses and normalizes the same slice. Stream values are
 * bit-identical to the legacy path; the RMS and mixing-dot reductions are
 * reordered, so downstream values may differ in the last bits. */
static double glm53f_mhc_partl[128 * 32];
static inline void glm53f_mhc_local_team(float *streams, const float *sublayer, glm53f_mhc_scratch *scratch,
        const glm53f_mhc_site *site, const uint16_t *norm, int do_post, float *logits,
        void (*after_normalize)(void *, const float *), void *context) {
    enum { S = GLM53F_MHC_STREAMS, W = GLM53F_MHC_WIDTH, VL = 16 };
    const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
    if (nt > 128 || svcntw() != VL) abort();
    glm53f_pf_run(tid);
    const int lo = (int)((long)(W / VL) * tid / nt) * VL, hi = (int)((long)(W / VL) * (tid + 1) / nt) * VL;
    const svbool_t pg = svptrue_b32();
    double ss = 0.0;
    if (do_post) {
        for (int k = 0; k < S; ++k)
            for (int d = lo; d < hi; ++d) {
                double v = (double)scratch->post[k] * sublayer[d];
                for (int j = 0; j < S; ++j)
                    v += (double)scratch->combine[(size_t)j * S + k] * scratch->residual[(size_t)j * W + d];
                streams[(size_t)k * W + d] = (float)v;
                ss += (double)streams[(size_t)k * W + d] * streams[(size_t)k * W + d];
            }
    } else {
        for (int k = 0; k < S; ++k)
            for (int d = lo; d < hi; ++d) ss += (double)streams[(size_t)k * W + d] * streams[(size_t)k * W + d];
    }
    double *part = glm53f_mhc_partl + (size_t)tid * 32;
    for (int m = 0; m < GLM53F_MHC_MIX; ++m) {
        const uint16_t *fn = site->fn + (size_t)m * GLM53F_MHC_FLAT;
        svfloat32_t acc = svdup_f32(0.0f);
        for (int k = 0; k < S; ++k)
            for (int d = lo; d < hi; d += VL) {
                const svfloat32_t w = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, fn + (size_t)k * W + d), 16));
                acc = svmla_f32_x(pg, acc, w, svld1_f32(pg, streams + (size_t)k * W + d));
            }
        part[m] = svaddv_f32(pg, acc);
    }
    part[GLM53F_MHC_MIX] = ss;
#pragma omp barrier
    float lg[GLM53F_MHC_MIX], post[S], comb[S * S];
    double total = 0.0;
    for (int t = 0; t < nt; ++t) total += glm53f_mhc_partl[(size_t)t * 32 + GLM53F_MHC_MIX];
    const float inv = 1.0f / sqrtf((float)(total / GLM53F_MHC_FLAT) + 1e-5f);
    for (int m = 0; m < GLM53F_MHC_MIX; ++m) {
        double dot = 0.0;
        for (int t = 0; t < nt; ++t) dot += glm53f_mhc_partl[(size_t)t * 32 + m];
        lg[m] = (float)dot * inv;
    }
    for (int k = 0; k < S; ++k) {
        lg[k] = glm53f_sigmoid(lg[k] * site->scale[0] + site->base[k]) + 1e-6f;
        post[k] = 2.0f * glm53f_sigmoid(lg[S + k] * site->scale[1] + site->base[S + k]);
    }
    for (int m = 0; m < S * S; ++m) comb[m] = lg[2 * S + m] * site->scale[2] + site->base[2 * S + m];
    glm53f_mhc_sinkhorn_fast(comb, S, 20, 1e-6f);
    double ss2 = 0.0;
    for (int d = lo; d < hi; ++d) {
        float value = 0.0f;
        for (int k = 0; k < S; ++k) {
            const float sv = streams[(size_t)k * W + d];
            value += lg[k] * sv;
            scratch->residual[(size_t)k * W + d] = sv;
        }
        scratch->collapsed[d] = value;
        ss2 += (double)value * value;
    }
    glm53f_mhc_part2[tid * 8] = ss2;
    if (!tid) {   /* every thread finished reading post/combine before the barrier above */
        memcpy(scratch->post, post, sizeof(post));
        memcpy(scratch->combine, comb, sizeof(comb));
        memcpy(logits, lg, sizeof(lg));
    }
#pragma omp barrier
    double total2 = 0.0;
    for (int t = 0; t < nt; ++t) total2 += glm53f_mhc_part2[t * 8];
    const float inv2 = 1.0f / sqrtf((float)(total2 / W) + 1e-5f);
    for (int d = lo; d < hi; ++d)
        scratch->normalized[d] = scratch->collapsed[d] * inv2 * glm53f_bf16_to_f32(norm[d]);
    if (after_normalize) {
#pragma omp barrier
        after_normalize(context, scratch->normalized);
    }
}
static inline void glm53f_mhc_fast_team(float *streams, const float *sublayer, glm53f_mhc_scratch *scratch,
        const glm53f_mhc_site *site, const uint16_t *norm, int do_post, float *logits,
        void (*after_normalize)(void *, const float *), void *context,
        int fused_sync, _Atomic int *mix_pending) {
    if (fused_sync == 2) {
        glm53f_mhc_local_team(streams, sublayer, scratch, site, norm, do_post, logits, after_normalize, context);
        return;
    }
    {
        const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        if (nt > 128) abort();
        if (fused_sync && !tid)
            atomic_store_explicit(mix_pending, nt < GLM53F_MHC_MIX ? nt : GLM53F_MHC_MIX,
                                  memory_order_relaxed);
        glm53f_pf_run(tid); /* weights of an upcoming stage, prefetched while the mHC math runs (glm53f_pf_plan.h) */
        /* 1. new streams (post) + partial sum of squares over this thread's contiguous slice */
        {
            const int lo = (int)((long)GLM53F_MHC_FLAT * tid / nt), hi = (int)((long)GLM53F_MHC_FLAT * (tid + 1) / nt);
            double ss = 0.0;
            if (do_post) {
                for (int i = lo; i < hi; ++i) {
                    const int k = i / GLM53F_MHC_WIDTH, d = i - k * GLM53F_MHC_WIDTH;
                    double v = (double)scratch->post[k] * sublayer[d];
                    for (int j = 0; j < GLM53F_MHC_STREAMS; ++j)
                        v += (double)scratch->combine[(size_t)j * GLM53F_MHC_STREAMS + k] *
                             scratch->residual[(size_t)j * GLM53F_MHC_WIDTH + d];
                    streams[i] = (float)v;
                    ss += (double)streams[i] * streams[i];
                }
            } else {
                for (int i = lo; i < hi; ++i) ss += (double)streams[i] * streams[i];
            }
            glm53f_mhc_part1[tid * 8] = ss;
        }
#pragma omp barrier
        double total = 0.0;
        for (int t = 0; t < nt; ++t) total += glm53f_mhc_part1[t * 8];
        const float inv = 1.0f / sqrtf((float)(total / GLM53F_MHC_FLAT) + 1e-5f);
        /* 2. 24 mixing logits. The fused scheduler keeps OpenMP's contiguous
         * static row ownership and each row's original accumulator chain.
         * The final owner acquires all previous owners' logit writes through
         * the counter's RMW sequence, then computes the coefficients while
         * the other workers wait at the sole publication barrier. */
        if (fused_sync) {
            const int chunk = GLM53F_MHC_MIX / nt, rem = GLM53F_MHC_MIX % nt;
            const int lo = tid * chunk + (tid < rem ? tid : rem);
            const int hi = lo + chunk + (tid < rem);
            if (hi > lo) {
                for (int m = lo; m < hi; ++m)
                    logits[m] = glm53f_mhc_dot_bf16_sve(site->fn + (size_t)m * GLM53F_MHC_FLAT,
                                                      streams, GLM53F_MHC_FLAT) * inv;
                if (atomic_fetch_sub_explicit(mix_pending, 1, memory_order_acq_rel) == 1)
                    glm53f_mhc_coefficients(logits, scratch, site);
            }
#pragma omp barrier
        } else {
#pragma omp for schedule(static)
            for (int m = 0; m < GLM53F_MHC_MIX; ++m)
                logits[m] = glm53f_mhc_dot_bf16_sve(site->fn + (size_t)m * GLM53F_MHC_FLAT, streams, GLM53F_MHC_FLAT) * inv;
            /* 3. sigmoids + Sinkhorn on one thread (implicit barrier) */
#pragma omp single
            { glm53f_mhc_coefficients(logits, scratch, site); }
        }
        /* 4. collapse + residual copy + partial sum of squares of the collapsed vector */
        {
            const int lo = (int)((long)GLM53F_MHC_WIDTH * tid / nt), hi = (int)((long)GLM53F_MHC_WIDTH * (tid + 1) / nt);
            double ss = 0.0;
            for (int d = lo; d < hi; ++d) {
                float value = 0.0f;
                for (int k = 0; k < GLM53F_MHC_STREAMS; ++k) {
                    const float sv = streams[(size_t)k * GLM53F_MHC_WIDTH + d];
                    value += logits[k] * sv;
                    scratch->residual[(size_t)k * GLM53F_MHC_WIDTH + d] = sv;
                }
                scratch->collapsed[d] = value;
                ss += (double)value * value;
            }
            glm53f_mhc_part2[tid * 8] = ss;
        }
#pragma omp barrier
        /* 5. RMS-normalize this thread's slice of the collapsed vector */
        {
            const int lo = (int)((long)GLM53F_MHC_WIDTH * tid / nt), hi = (int)((long)GLM53F_MHC_WIDTH * (tid + 1) / nt);
            double total2 = 0.0;
            for (int t = 0; t < nt; ++t) total2 += glm53f_mhc_part2[t * 8];
            const float inv2 = 1.0f / sqrtf((float)(total2 / GLM53F_MHC_WIDTH) + 1e-5f);
            for (int d = lo; d < hi; ++d)
                scratch->normalized[d] = scratch->collapsed[d] * inv2 * glm53f_bf16_to_f32(norm[d]);
        }
        if (after_normalize) {
#pragma omp barrier
            after_normalize(context, scratch->normalized);
        }
    }
}

typedef struct {
    float *streams;
    const float *sublayer;
    glm53f_mhc_scratch *scratch;
    const glm53f_mhc_site *site;
    const uint16_t *norm;
    int do_post;
    float logits[GLM53F_MHC_MIX];
    void (*after_normalize)(void *, const float *);
    void *context;
    int fused_sync;
    _Alignas(256) _Atomic int mix_pending;
} glm53f_mhc_call;
static void glm53f_mhc_worker(void *context) {
    glm53f_mhc_call *a = context;
    glm53f_mhc_fast_team(a->streams, a->sublayer, a->scratch, a->site, a->norm,
        a->do_post, a->logits, a->after_normalize, a->context,
        a->fused_sync, &a->mix_pending);
}
static inline void glm53f_mhc_fast_route(float *streams, const float *sublayer,
        glm53f_mhc_scratch *scratch, const glm53f_mhc_site *site,
        const uint16_t *norm, int do_post,
        void (*after_normalize)(void *, const float *), void *context) {
    glm53f_mhc_call call = {.streams = streams, .sublayer = sublayer,
        .scratch = scratch, .site = site, .norm = norm, .do_post = do_post,
        .after_normalize = after_normalize, .context = context,
        .fused_sync = glm53f_mhc_fused_sync_on()};
    atomic_init(&call.mix_pending, 0);
    if (glm53f_team_available()) glm53f_team_dispatch(glm53f_mhc_worker, &call);
    else {
#pragma omp parallel
        { glm53f_mhc_worker(&call); }
    }
}
static inline void glm53f_mhc_fast(float *streams, const float *sublayer,
        glm53f_mhc_scratch *scratch, const glm53f_mhc_site *site,
        const uint16_t *norm, int do_post) {
    glm53f_mhc_fast_route(streams, sublayer, scratch, site, norm, do_post, NULL, NULL);
}

/* Variant with distributed mixing dots (GLM53F_MHC_FAST=2): every thread computes the 24 partial dot products over the
 * slice of the streams it just produced, so the logits need one barrier (shared with the sum of squares) instead of a
 * 24-way work split whose dot chains dominate; Sinkhorn runs redundantly on every thread (thread 0 publishes post /
 * combine).  Two barriers in total.  Summation order of the logits differs from mode 1 (fp32 rounding only). */
/* Bit-identical SVE version of glm53f_mhc_sinkhorn for hc == 4: the whole 4x4 matrix lives in one 16-lane vector
 * (lane = 4 * row + col).  Every lane recomputes its row / column sum with the same ordered additions
 * ((eps + c0) + c1 + c2 + c3) as the scalar loops and divides by it, so the result equals the scalar code exactly. */
static inline void glm53f_mhc_sinkhorn4_sve(float *comb, int iters, float eps) {
    const svbool_t pg = svptrue_b32();
    uint32_t rowidx[4][16], colidx[4][16];
    for (int l = 0; l < 16; ++l)
        for (int j = 0; j < 4; ++j) { rowidx[j][l] = (uint32_t)((l >> 2) * 4 + j); colidx[j][l] = (uint32_t)(j * 4 + (l & 3)); }
    const svuint32_t r0 = svld1_u32(pg, rowidx[0]), r1 = svld1_u32(pg, rowidx[1]), r2 = svld1_u32(pg, rowidx[2]),
                     r3 = svld1_u32(pg, rowidx[3]), c0 = svld1_u32(pg, colidx[0]), c1 = svld1_u32(pg, colidx[1]),
                     c2 = svld1_u32(pg, colidx[2]), c3 = svld1_u32(pg, colidx[3]);
    /* initial row softmax (expf per element: keep the scalar code, it matches the reference bit for bit) */
    for (int i = 0; i < 4; ++i) {
        float mx = comb[i * 4], sum = 0.0f;
        for (int j = 1; j < 4; ++j) if (comb[i * 4 + j] > mx) mx = comb[i * 4 + j];
        for (int j = 0; j < 4; ++j) { comb[i * 4 + j] = expf(comb[i * 4 + j] - mx) + eps; sum += comb[i * 4 + j]; }
        for (int j = 0; j < 4; ++j) comb[i * 4 + j] /= sum;
    }
    svfloat32_t m = svld1_f32(pg, comb);
    for (int z = 0; z < iters; ++z) {
        if (z > 0) {
            svfloat32_t s = svdup_n_f32(eps);
            s = svadd_f32_x(pg, s, svtbl_f32(m, r0)); s = svadd_f32_x(pg, s, svtbl_f32(m, r1));
            s = svadd_f32_x(pg, s, svtbl_f32(m, r2)); s = svadd_f32_x(pg, s, svtbl_f32(m, r3));
            m = svdiv_f32_x(pg, m, s);
        }
        svfloat32_t s = svdup_n_f32(eps);
        s = svadd_f32_x(pg, s, svtbl_f32(m, c0)); s = svadd_f32_x(pg, s, svtbl_f32(m, c1));
        s = svadd_f32_x(pg, s, svtbl_f32(m, c2)); s = svadd_f32_x(pg, s, svtbl_f32(m, c3));
        m = svdiv_f32_x(pg, m, s);
    }
    svst1_f32(pg, comb, m);
}
static inline void glm53f_mhc_sinkhorn_fast(float *comb, int hc, int iters, float eps) {
    if (hc == 4) glm53f_mhc_sinkhorn4_sve(comb, iters, eps); else glm53f_mhc_sinkhorn(comb, hc, iters, eps);
}
static float glm53f_mhc_pdot[128 * 32];
static double glm53f_mhc_stamp[8]; static int glm53f_mhc_stamp_on;
static inline void glm53f_mhc_fast2(float *streams, const float *sublayer, glm53f_mhc_scratch *scratch,
        const glm53f_mhc_site *site, const uint16_t *norm, int do_post) {
#pragma omp parallel
    {
        const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        if (nt > 128) abort();
        double tq0 = 0, tq1 = 0, tq2 = 0, tq3 = 0, tq4 = 0;
        if (tid == 0) tq0 = glm53f_mhc_now();
        const int lo = (int)((long)GLM53F_MHC_FLAT * tid / nt), hi = (int)((long)GLM53F_MHC_FLAT * (tid + 1) / nt);
        double ss = 0.0;
        if (do_post) {
            for (int i = lo; i < hi; ++i) {
                const int k = i / GLM53F_MHC_WIDTH, d = i - k * GLM53F_MHC_WIDTH;
                double v = (double)scratch->post[k] * sublayer[d];
                for (int j = 0; j < GLM53F_MHC_STREAMS; ++j)
                    v += (double)scratch->combine[(size_t)j * GLM53F_MHC_STREAMS + k] *
                         scratch->residual[(size_t)j * GLM53F_MHC_WIDTH + d];
                streams[i] = (float)v;
                ss += (double)streams[i] * streams[i];
            }
        } else {
            for (int i = lo; i < hi; ++i) ss += (double)streams[i] * streams[i];
        }
        glm53f_mhc_part1[tid * 8] = ss;
        if (tid == 0) tq1 = glm53f_mhc_now();
        for (int m = 0; m < GLM53F_MHC_MIX; ++m)
            glm53f_mhc_pdot[tid * 32 + m] = glm53f_mhc_dot_bf16_sve(site->fn + (size_t)m * GLM53F_MHC_FLAT + lo, streams + lo, hi - lo);
        if (tid == 0) tq2 = glm53f_mhc_now();
#pragma omp barrier
        if (tid == 0) tq3 = glm53f_mhc_now();
        double total = 0.0;
        for (int t = 0; t < nt; ++t) total += glm53f_mhc_part1[t * 8];
        const float inv = 1.0f / sqrtf((float)(total / GLM53F_MHC_FLAT) + 1e-5f);
        float logits[GLM53F_MHC_MIX], post[GLM53F_MHC_STREAMS], comb[GLM53F_MHC_STREAMS * GLM53F_MHC_STREAMS];
        for (int m = 0; m < GLM53F_MHC_MIX; ++m) {
            float v = 0.f;
            for (int t = 0; t < nt; ++t) v += glm53f_mhc_pdot[t * 32 + m];
            logits[m] = v * inv;
        }
        for (int k = 0; k < GLM53F_MHC_STREAMS; ++k) {
            logits[k] = glm53f_sigmoid(logits[k] * site->scale[0] + site->base[k]) + 1e-6f;
            post[k] = 2.0f * glm53f_sigmoid(logits[GLM53F_MHC_STREAMS + k] * site->scale[1] + site->base[GLM53F_MHC_STREAMS + k]);
        }
        for (int m = 0; m < GLM53F_MHC_STREAMS * GLM53F_MHC_STREAMS; ++m)
            comb[m] = logits[2 * GLM53F_MHC_STREAMS + m] * site->scale[2] + site->base[2 * GLM53F_MHC_STREAMS + m];
        glm53f_mhc_sinkhorn(comb, GLM53F_MHC_STREAMS, 20, 1e-6f);
        if (tid == 0) tq4 = glm53f_mhc_now();
        if (tid == 0) {
            for (int k = 0; k < GLM53F_MHC_STREAMS; ++k) scratch->post[k] = post[k];
            for (int m = 0; m < GLM53F_MHC_STREAMS * GLM53F_MHC_STREAMS; ++m) scratch->combine[m] = comb[m];
        }
        {
            const int lo2 = (int)((long)GLM53F_MHC_WIDTH * tid / nt), hi2 = (int)((long)GLM53F_MHC_WIDTH * (tid + 1) / nt);
            double ss2 = 0.0;
            for (int d = lo2; d < hi2; ++d) {
                float value = 0.0f;
                for (int k = 0; k < GLM53F_MHC_STREAMS; ++k) {
                    const float sv = streams[(size_t)k * GLM53F_MHC_WIDTH + d];
                    value += logits[k] * sv;
                    scratch->residual[(size_t)k * GLM53F_MHC_WIDTH + d] = sv;
                }
                scratch->collapsed[d] = value;
                ss2 += (double)value * value;
            }
            glm53f_mhc_part2[tid * 8] = ss2;
        }
        const double tq5 = tid == 0 ? glm53f_mhc_now() : 0.0;
#pragma omp barrier
        const double tq6 = tid == 0 ? glm53f_mhc_now() : 0.0;
        {
            const int lo2 = (int)((long)GLM53F_MHC_WIDTH * tid / nt), hi2 = (int)((long)GLM53F_MHC_WIDTH * (tid + 1) / nt);
            double total2 = 0.0;
            for (int t = 0; t < nt; ++t) total2 += glm53f_mhc_part2[t * 8];
            const float inv2 = 1.0f / sqrtf((float)(total2 / GLM53F_MHC_WIDTH) + 1e-5f);
            for (int d = lo2; d < hi2; ++d)
                scratch->normalized[d] = scratch->collapsed[d] * inv2 * glm53f_bf16_to_f32(norm[d]);
        }
        if (tid == 0 && glm53f_mhc_stamp_on) {
            const double te = glm53f_mhc_now();
            glm53f_mhc_stamp[0] += tq1 - tq0; glm53f_mhc_stamp[1] += tq2 - tq1; glm53f_mhc_stamp[2] += tq3 - tq2; glm53f_mhc_stamp[3] += tq4 - tq3;
            glm53f_mhc_stamp[4] += tq5 - tq4; glm53f_mhc_stamp[5] += tq6 - tq5; glm53f_mhc_stamp[6] += te - tq6; glm53f_mhc_stamp[7] += 1;
        }
    }
}

static inline void glm53f_mhc_pre_sve(
        glm53f_mhc_scratch *scratch, const float *streams,
        const glm53f_mhc_site *site, const uint16_t *norm) {
    if (glm53f_mhc_fast_on()) { if (glm53f_mhc_fast_on() == 2) glm53f_mhc_fast2((float *)streams, NULL, scratch, site, norm, 0); else glm53f_mhc_fast((float *)streams, NULL, scratch, site, norm, 0); return; }
    double sumsq = 0.0;
    float logits[GLM53F_MHC_MIX];
#if GLM53F_MHC_FUSED
#pragma omp parallel shared(sumsq,logits)
    {
#pragma omp for reduction(+:sumsq) schedule(static)
        for (int i = 0; i < GLM53F_MHC_FLAT; ++i)
            sumsq += (double)streams[i] * streams[i];
#pragma omp barrier
#pragma omp single
        { sumsq = (double)(1.0f / sqrtf((float)(sumsq / GLM53F_MHC_FLAT) + 1e-5f)); }
#pragma omp barrier
#pragma omp for schedule(static)
        for (int m = 0; m < GLM53F_MHC_MIX; ++m)
            logits[m] = glm53f_mhc_dot_bf16_sve(
                site->fn + (size_t)m * GLM53F_MHC_FLAT, streams,
                GLM53F_MHC_FLAT) * (float)sumsq;
    }
#else
    const int dtl = glm53f_mhc_detail_on(); double t0 = dtl ? glm53f_mhc_now() : 0.0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;
#pragma omp parallel for reduction(+:sumsq)
    for (int i = 0; i < GLM53F_MHC_FLAT; ++i)
        sumsq += (double)streams[i] * streams[i];
    float inv = 1.0f / sqrtf((float)(sumsq / GLM53F_MHC_FLAT) + 1e-5f);
    if (dtl) t1 = glm53f_mhc_now();
#pragma omp parallel for schedule(static)
    for (int m = 0; m < GLM53F_MHC_MIX; ++m)
        logits[m] = glm53f_mhc_dot_bf16_sve(
            site->fn + (size_t)m * GLM53F_MHC_FLAT, streams,
            GLM53F_MHC_FLAT) * inv;
    if (dtl) t2 = glm53f_mhc_now();
#endif
    for (int k = 0; k < GLM53F_MHC_STREAMS; ++k) {
        logits[k] = glm53f_sigmoid(logits[k] * site->scale[0] + site->base[k]) + 1e-6f;
        scratch->post[k] = 2.0f * glm53f_sigmoid(
            logits[GLM53F_MHC_STREAMS + k] * site->scale[1] +
            site->base[GLM53F_MHC_STREAMS + k]);
    }
    for (int m = 0; m < GLM53F_MHC_STREAMS * GLM53F_MHC_STREAMS; ++m)
        scratch->combine[m] = logits[2 * GLM53F_MHC_STREAMS + m] *
                              site->scale[2] + site->base[2 * GLM53F_MHC_STREAMS + m];
    glm53f_mhc_sinkhorn_fast(scratch->combine, GLM53F_MHC_STREAMS, 20, 1e-6f);
#if !GLM53F_MHC_FUSED
    if (dtl) t3 = glm53f_mhc_now();
#endif
#pragma omp parallel for schedule(static)
    for (int d = 0; d < GLM53F_MHC_WIDTH; ++d) {
        float value = 0.0f;
        for (int k = 0; k < GLM53F_MHC_STREAMS; ++k)
            value += logits[k] * streams[(size_t)k * GLM53F_MHC_WIDTH + d];
        scratch->collapsed[d] = value;
    }
#if !GLM53F_MHC_FUSED
    if (dtl) t4 = glm53f_mhc_now();
#endif
    memcpy(scratch->residual, streams, sizeof(scratch->residual));
    glm53f_rmsnorm_bf16(scratch->normalized, scratch->collapsed, norm,
                        GLM53F_MHC_WIDTH, 1e-5f);
#if !GLM53F_MHC_FUSED
    if (dtl) {
        double t5 = glm53f_mhc_now();
        glm53f_mhc_acc[0] += t1 - t0; glm53f_mhc_acc[1] += t2 - t1; glm53f_mhc_acc[2] += t3 - t2;
        glm53f_mhc_acc[3] += t4 - t3; glm53f_mhc_acc[4] += t5 - t4; glm53f_mhc_calls[0]++;
    }
#endif
}

static inline void glm53f_mhc_pre_prefill_sve(
        glm53f_mhc_scratch *scratch, const float *streams,
        const glm53f_mhc_site *site, const uint16_t *norm, int tokens,
        size_t scratch_stride, float *normalized) {
    float inv[GLM53F_PREFILL_MAX_TOKENS];
    float logits[GLM53F_PREFILL_MAX_TOKENS * GLM53F_MHC_MIX];
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int t = 0; t < tokens; ++t) {
            const float *s = streams + (size_t)t * GLM53F_MHC_FLAT;
            double sumsq = 0;
            for (int i = 0; i < GLM53F_MHC_FLAT; ++i) sumsq += (double)s[i] * s[i];
            inv[t] = 1.0f / sqrtf((float)(sumsq / GLM53F_MHC_FLAT) + 1e-5f);
        }
#pragma omp for collapse(2) schedule(static)
        for (int base = 0; base < tokens; base += 4)
            for (int row = 0; row < GLM53F_MHC_MIX; ++row) {
                int n = tokens - base;
                if (n > 4) n = 4;
                glm53f_mhc_row_batch4(logits + (size_t)base * GLM53F_MHC_MIX,
                    site->fn, streams + (size_t)base * GLM53F_MHC_FLAT, n, row);
            }
#pragma omp for schedule(static)
        for (int t = 0; t < tokens; ++t) {
            glm53f_mhc_scratch *q = (glm53f_mhc_scratch *)
                ((unsigned char *)scratch + (size_t)t * scratch_stride);
            const float *s = streams + (size_t)t * GLM53F_MHC_FLAT;
            float *z = logits + (size_t)t * GLM53F_MHC_MIX;
            for (int m = 0; m < GLM53F_MHC_MIX; ++m) z[m] *= inv[t];
            for (int k = 0; k < GLM53F_MHC_STREAMS; ++k) {
                z[k] = glm53f_sigmoid(z[k] * site->scale[0] + site->base[k]) + 1e-6f;
                q->post[k] = 2.0f * glm53f_sigmoid(z[GLM53F_MHC_STREAMS + k] *
                    site->scale[1] + site->base[GLM53F_MHC_STREAMS + k]);
            }
            for (int m = 0; m < GLM53F_MHC_STREAMS * GLM53F_MHC_STREAMS; ++m)
                q->combine[m] = z[2 * GLM53F_MHC_STREAMS + m] * site->scale[2] +
                    site->base[2 * GLM53F_MHC_STREAMS + m];
            glm53f_mhc_sinkhorn_fast(q->combine, GLM53F_MHC_STREAMS, 20, 1e-6f);
            for (int d = 0; d < GLM53F_MHC_WIDTH; ++d) {
                float v = 0;
                for (int k = 0; k < GLM53F_MHC_STREAMS; ++k)
                    v += z[k] * s[(size_t)k * GLM53F_MHC_WIDTH + d];
                q->collapsed[d] = v;
            }
            memcpy(q->residual, s, sizeof(q->residual));
            glm53f_rmsnorm_bf16(q->normalized, q->collapsed, norm, GLM53F_MHC_WIDTH, 1e-5f);
            memcpy(normalized + (size_t)t * GLM53F_MHC_WIDTH, q->normalized,
                   GLM53F_MHC_WIDTH * sizeof(float));
        }
    }
}

/* Small verification batches retain the legacy per-token FP64 norm
 * partitions and four-position BF16 dot chains. Coefficients are independent
 * across tokens; one collapse workshare replaces per-token team creation.
 * Residual copies and serial RMS normalization keep their original order. */
static inline void glm53f_mhc_pre_batch_team_sve(glm53f_mhc_scratch *scratch, const float *streams,
        const glm53f_mhc_site *site, const uint16_t *norm, int tokens,
        size_t scratch_stride, float *normalized) {
    double sumsq = 0.0;
    float inv[5], logits[5 * GLM53F_MHC_MIX];
    if (tokens < 1 || tokens > 5) abort();
#pragma omp parallel shared(sumsq, inv, logits)
    {
        for (int t = 0; t < tokens; ++t) {
            const float *s = streams + (size_t)t * GLM53F_MHC_FLAT;
#pragma omp for reduction(+:sumsq)
            for (int i = 0; i < GLM53F_MHC_FLAT; ++i)
                sumsq += (double)s[i] * s[i];
#pragma omp single
            {
                inv[t] = 1.0f / sqrtf((float)(sumsq / GLM53F_MHC_FLAT) + 1e-5f);
                sumsq = 0.0;
            }
        }
        for (int base = 0; base < tokens; base += 4) {
            int n = tokens - base;
            if (n > 4) n = 4;
#pragma omp for schedule(static)
            for (int row = 0; row < GLM53F_MHC_MIX; ++row)
                glm53f_mhc_row_batch4(logits + (size_t)base * GLM53F_MHC_MIX,
                    site->fn, streams + (size_t)base * GLM53F_MHC_FLAT, n, row);
        }
#pragma omp for schedule(static)
        for (int t = 0; t < tokens; ++t) {
            glm53f_mhc_scratch *q = (glm53f_mhc_scratch *)
                ((unsigned char *)scratch + (size_t)t * scratch_stride);
            float *z = logits + (size_t)t * GLM53F_MHC_MIX;
            for (int m = 0; m < GLM53F_MHC_MIX; ++m) z[m] *= inv[t];
            for (int k = 0; k < GLM53F_MHC_STREAMS; ++k) {
                z[k] = glm53f_sigmoid(z[k] * site->scale[0] + site->base[k]) + 1e-6f;
                q->post[k] = 2.0f * glm53f_sigmoid(z[GLM53F_MHC_STREAMS + k] *
                    site->scale[1] + site->base[GLM53F_MHC_STREAMS + k]);
            }
            for (int m = 0; m < GLM53F_MHC_STREAMS * GLM53F_MHC_STREAMS; ++m)
                q->combine[m] = z[2 * GLM53F_MHC_STREAMS + m] * site->scale[2] +
                    site->base[2 * GLM53F_MHC_STREAMS + m];
            glm53f_mhc_sinkhorn_fast(q->combine, GLM53F_MHC_STREAMS, 20, 1e-6f);
        }
#pragma omp for collapse(2) schedule(static)
        for (int t = 0; t < tokens; ++t) {
            for (int d = 0; d < GLM53F_MHC_WIDTH; ++d) {
                glm53f_mhc_scratch *q = (glm53f_mhc_scratch *)
                    ((unsigned char *)scratch + (size_t)t * scratch_stride);
                const float *s = streams + (size_t)t * GLM53F_MHC_FLAT;
                float *z = logits + (size_t)t * GLM53F_MHC_MIX;
                float v = 0.0f;
                for (int k = 0; k < GLM53F_MHC_STREAMS; ++k)
                    v += z[k] * s[(size_t)k * GLM53F_MHC_WIDTH + d];
                q->collapsed[d] = v;
            }
        }
    }
    /* Retain the reference's serial residual copy and FP64 RMS chain. */
    for (int t = 0; t < tokens; ++t) {
        glm53f_mhc_scratch *q = (glm53f_mhc_scratch *)
            ((unsigned char *)scratch + (size_t)t * scratch_stride);
        memcpy(q->residual, streams + (size_t)t * GLM53F_MHC_FLAT, sizeof(q->residual));
        glm53f_rmsnorm_bf16(q->normalized, q->collapsed, norm, GLM53F_MHC_WIDTH, 1e-5f);
        memcpy(normalized + (size_t)t * GLM53F_MHC_WIDTH, q->normalized,
               GLM53F_MHC_WIDTH * sizeof(float));
    }
}

static inline void glm53f_mhc_pre_batch_sve(
        glm53f_mhc_scratch *scratch, const float *streams,
        const glm53f_mhc_site *site, const uint16_t *norm, int tokens,
        size_t scratch_stride, float *normalized) {
    if (tokens > 5 && getenv("GLM53F_MHC_PREFILL") &&
        atoi(getenv("GLM53F_MHC_PREFILL"))) {
        glm53f_mhc_pre_prefill_sve(scratch, streams, site, norm, tokens,
                                  scratch_stride, normalized);
        return;
    }
    const char *team = tokens > 1 && tokens <= 5 ? getenv("GLM53F_MHC_BATCH_TEAM") : NULL;
    if (team && atoi(team)) {
        glm53f_mhc_pre_batch_team_sve(scratch, streams, site, norm, tokens,
                                    scratch_stride, normalized);
        return;
    }
    enum { GLM53F_MHC_PREFILL_BATCH = GLM53F_PREFILL_MAX_TOKENS };
    float inv[GLM53F_MHC_PREFILL_BATCH];
    float logits[GLM53F_MHC_PREFILL_BATCH * GLM53F_MHC_MIX];
    for(int t=0;t<tokens;t++){double sumsq=0;const float*s=streams+(size_t)t*GLM53F_MHC_FLAT;
#pragma omp parallel for reduction(+:sumsq)
        for(int i=0;i<GLM53F_MHC_FLAT;i++)sumsq+=(double)s[i]*s[i];inv[t]=1.0f/sqrtf((float)(sumsq/GLM53F_MHC_FLAT)+1e-5f);}
    for (int base = 0; base < tokens; base += 4) {
        int n = tokens - base;
        if (n > 4) n = 4;
        glm53f_mhc_mv_batch4(logits + (size_t)base * GLM53F_MHC_MIX,
            site->fn, streams + (size_t)base * GLM53F_MHC_FLAT, n);
    }
    for(int t=0;t<tokens;t++){glm53f_mhc_scratch*q=(glm53f_mhc_scratch*)((unsigned char*)scratch+(size_t)t*scratch_stride);const float*s=streams+(size_t)t*GLM53F_MHC_FLAT;float*z=logits+(size_t)t*GLM53F_MHC_MIX;for(int m=0;m<GLM53F_MHC_MIX;m++)z[m]*=inv[t];for(int k=0;k<GLM53F_MHC_STREAMS;k++){z[k]=glm53f_sigmoid(z[k]*site->scale[0]+site->base[k])+1e-6f;q->post[k]=2.0f*glm53f_sigmoid(z[GLM53F_MHC_STREAMS+k]*site->scale[1]+site->base[GLM53F_MHC_STREAMS+k]);}for(int m=0;m<GLM53F_MHC_STREAMS*GLM53F_MHC_STREAMS;m++)q->combine[m]=z[2*GLM53F_MHC_STREAMS+m]*site->scale[2]+site->base[2*GLM53F_MHC_STREAMS+m];glm53f_mhc_sinkhorn_fast(q->combine,GLM53F_MHC_STREAMS,20,1e-6f);
#pragma omp parallel for schedule(static)
        for(int d=0;d<GLM53F_MHC_WIDTH;d++){float v=0;for(int k=0;k<GLM53F_MHC_STREAMS;k++)v+=z[k]*s[(size_t)k*GLM53F_MHC_WIDTH+d];q->collapsed[d]=v;}memcpy(q->residual,s,sizeof(q->residual));glm53f_rmsnorm_bf16(q->normalized,q->collapsed,norm,GLM53F_MHC_WIDTH,1e-5f);memcpy(normalized+(size_t)t*GLM53F_MHC_WIDTH,q->normalized,GLM53F_MHC_WIDTH*4);}
}

typedef struct { float *streams; const float *sublayer; const glm53f_mhc_scratch *scratch; } glm53f_mhc_post_call;
static void glm53f_mhc_post_worker(void *context) {
    glm53f_mhc_post_call *a = context;
    float *streams = a->streams;
    const float *sublayer = a->sublayer;
    const glm53f_mhc_scratch *scratch = a->scratch;
#pragma omp for collapse(2) schedule(static)
        for (int k = 0; k < GLM53F_MHC_STREAMS; ++k)
        for (int d = 0; d < GLM53F_MHC_WIDTH; ++d) {
#if GLM53F_MHC_POST_FLOAT
            float v = scratch->post[k] * sublayer[d];
            for (int j = 0; j < GLM53F_MHC_STREAMS; ++j)
                v += scratch->combine[(size_t)j * GLM53F_MHC_STREAMS + k] *
                     scratch->residual[(size_t)j * GLM53F_MHC_WIDTH + d];
#else
            double v = (double)scratch->post[k] * sublayer[d];
            for (int j = 0; j < GLM53F_MHC_STREAMS; ++j)
                v += (double)scratch->combine[(size_t)j * GLM53F_MHC_STREAMS + k] *
                     scratch->residual[(size_t)j * GLM53F_MHC_WIDTH + d];
#endif
            streams[(size_t)k * GLM53F_MHC_WIDTH + d] = (float)v;
        }
 }
static inline void glm53f_mhc_post_sve(
        float *streams, const float *sublayer, const glm53f_mhc_scratch *scratch) {
    const double dt0 = glm53f_mhc_detail_on() ? glm53f_mhc_now() : 0.0;
    glm53f_mhc_post_call call = {streams, sublayer, scratch};
    if (glm53f_team_available()) glm53f_team_dispatch(glm53f_mhc_post_worker, &call);
    else {
#pragma omp parallel
        { glm53f_mhc_post_worker(&call); }
    }
    if (glm53f_mhc_detail_on()) { glm53f_mhc_acc[5] += glm53f_mhc_now() - dt0; glm53f_mhc_calls[1]++; }
}

/* Finish one site and prepare the next in one team. This is deliberately an
 * arithmetic-preserving scalar-decode optimization: loop schedules, FP64
 * post accumulation, Sinkhorn, and the serial RMS normalization match the
 * separate post/pre calls above. */
static inline void glm53f_mhc_post_pre_sve(
        float *streams, const float *sublayer, glm53f_mhc_scratch *scratch,
        const glm53f_mhc_site *next_site, const uint16_t *next_norm) {
    if (glm53f_mhc_fast_on()) { if (glm53f_mhc_fast_on() == 2) glm53f_mhc_fast2(streams, sublayer, scratch, next_site, next_norm, 1); else glm53f_mhc_fast(streams, sublayer, scratch, next_site, next_norm, 1); return; }
    double sumsq = 0.0;
    float inv = 0.0f, logits[GLM53F_MHC_MIX];
#pragma omp parallel shared(sumsq,inv,logits)
    {
#pragma omp for collapse(2) schedule(static)
        for (int k = 0; k < GLM53F_MHC_STREAMS; ++k)
            for (int d = 0; d < GLM53F_MHC_WIDTH; ++d) {
                double v = (double)scratch->post[k] * sublayer[d];
                for (int j = 0; j < GLM53F_MHC_STREAMS; ++j)
                    v += (double)scratch->combine[(size_t)j * GLM53F_MHC_STREAMS + k] *
                         scratch->residual[(size_t)j * GLM53F_MHC_WIDTH + d];
                streams[(size_t)k * GLM53F_MHC_WIDTH + d] = (float)v;
            }
#pragma omp for reduction(+:sumsq)
        for (int i = 0; i < GLM53F_MHC_FLAT; ++i)
            sumsq += (double)streams[i] * streams[i];
#pragma omp single
        { inv = 1.0f / sqrtf((float)(sumsq / GLM53F_MHC_FLAT) + 1e-5f); }
#pragma omp for schedule(static)
        for (int m = 0; m < GLM53F_MHC_MIX; ++m)
            logits[m] = glm53f_mhc_dot_bf16_sve(
                next_site->fn + (size_t)m * GLM53F_MHC_FLAT, streams,
                GLM53F_MHC_FLAT) * inv;
#pragma omp single
        {
            for (int k = 0; k < GLM53F_MHC_STREAMS; ++k) {
                logits[k] = glm53f_sigmoid(logits[k] * next_site->scale[0] +
                                            next_site->base[k]) + 1e-6f;
                scratch->post[k] = 2.0f * glm53f_sigmoid(
                    logits[GLM53F_MHC_STREAMS + k] * next_site->scale[1] +
                    next_site->base[GLM53F_MHC_STREAMS + k]);
            }
            for (int m = 0; m < GLM53F_MHC_STREAMS * GLM53F_MHC_STREAMS; ++m)
                scratch->combine[m] = logits[2 * GLM53F_MHC_STREAMS + m] *
                                      next_site->scale[2] + next_site->base[2 * GLM53F_MHC_STREAMS + m];
            glm53f_mhc_sinkhorn_fast(scratch->combine, GLM53F_MHC_STREAMS, 20, 1e-6f);
        }
#pragma omp for schedule(static)
        for (int d = 0; d < GLM53F_MHC_WIDTH; ++d) {
            float value = 0.0f;
            for (int k = 0; k < GLM53F_MHC_STREAMS; ++k)
                value += logits[k] * streams[(size_t)k * GLM53F_MHC_WIDTH + d];
            scratch->collapsed[d] = value;
        }
#pragma omp single
        {
            memcpy(scratch->residual, streams, sizeof(scratch->residual));
            glm53f_rmsnorm_bf16(scratch->normalized, scratch->collapsed,
                                next_norm, GLM53F_MHC_WIDTH, 1e-5f);
        }
    }
}

/* Batch-only post mix.  Verification positions are independent at this
 * point, so distribute the (otherwise scalar) mHC post over token/head
 * pairs.  The inner accumulation order is identical to glm53f_mhc_post,
 * preserving target-token decisions while removing four serial calls. */
static inline void glm53f_mhc_post_batch_sve(
        float *streams, const float *sublayer,
        const glm53f_mhc_scratch *scratch, int tokens, size_t stride) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < GLM53F_MHC_STREAMS; ++k) {
            float *dst = streams + (size_t)t * GLM53F_MHC_FLAT +
                         (size_t)k * GLM53F_MHC_WIDTH;
            const float *res = (const float *)((const unsigned char *)scratch +
                              (size_t)t * stride + offsetof(glm53f_mhc_scratch, residual));
            const glm53f_mhc_scratch *s = (const glm53f_mhc_scratch *)
                              ((const unsigned char *)scratch + (size_t)t * stride);
            for (int d = 0; d < GLM53F_MHC_WIDTH; ++d) {
#if GLM53F_MHC_POST_FLOAT
                float v = s->post[k] * sublayer[(size_t)t * GLM53F_MHC_WIDTH + d];
                for (int j = 0; j < GLM53F_MHC_STREAMS; ++j)
                    v += s->combine[(size_t)j * GLM53F_MHC_STREAMS + k] *
                         res[(size_t)j * GLM53F_MHC_WIDTH + d];
#else
                double v = (double)s->post[k] * sublayer[(size_t)t * GLM53F_MHC_WIDTH + d];
                for (int j = 0; j < GLM53F_MHC_STREAMS; ++j)
                    v += (double)s->combine[(size_t)j * GLM53F_MHC_STREAMS + k] *
                         res[(size_t)j * GLM53F_MHC_WIDTH + d];
#endif
                dst[d] = (float)v;
            }
        }
}

#endif
