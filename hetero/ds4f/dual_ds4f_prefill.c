#define _GNU_SOURCE
#include "../../common/ds4f.h"
#include "dual_ds4f_prefill.h"
#include <pthread.h>
#include <stdlib.h>

typedef struct {
    dual_ds4f_prefill *ctx;
    float *const *dst;
    const ds4f_tensor *const *t;
    const float *const *x;
    const int *M, *Ys, *Xs;
    const int *idx;
    int n;
    int rc;
} dual_job;

struct dual_ds4f_prefill {
    hip_ds4f_dense *hip;
    int owns_hip;
    cuda_ds4f_mxfp4 *cuda;
    const ds4f_tensor *cuda_tensor;
    int cuda_rows, cuda_cols;
    pthread_mutex_t cuda_lock;
    int verbose;
    int cuda_mxfp4;
    int max_batch;
    int small_buckets;
    /* Captured HIP layer-residency entry points.  gpu_dense_ctx becomes the
     * dual wrapper, so the MXFP4 expert streaming callbacks must be forwarded
     * to the wrapped HIP runner or they would dereference the dual struct as a
     * hip_ds4f_dense (crash). */
    ds4f_gpu_dense_layer_fn hip_layer_begin;
    ds4f_gpu_dense_layer_prefetch_fn hip_layer_prefetch;
};

static int cuda_eligible(const dual_ds4f_prefill *c, const ds4f_tensor *t,
                         int M, int Ys, int Xs) {
    /* Keep the exact CPU fallback for small routed buckets by default.  CUDA
     * is used only once a full x128 batch is available; this avoids partial-tile
     * instability and limits long-context accumulation drift.  c->small_buckets
     * opts into the padded small-bucket path (the MMQ pads internally).
     * A tensor with a positive gpu_id is owned by the HIP MXFP4 resident
     * upload (matrix id >= the dense bank), so it must NOT be re-routed to the
     * SM120 -- the dual split keeps the head layers on the ROCm and the tail
     * on the RTX card. */
    int mmin = c->small_buckets ? 1 : 128;
    return c && c->cuda_mxfp4 && t && t->type == DS4F_MXFP4 && t->w && t->scale &&
           t->gpu_id == 0 && M >= mmin &&
           Ys == t->rows && Xs == t->cols && (t->rows % 128) == 0 &&
           (t->cols % 32) == 0;
}

static int dual_cuda_one(dual_ds4f_prefill *c, float *dst,
                         const ds4f_tensor *t, const float *x, int M,
                         int Ys, int Xs) {
    if (!cuda_eligible(c, t, M, Ys, Xs)) return -1;
    pthread_mutex_lock(&c->cuda_lock);
    int rc = 0;
    if (c->cuda_tensor != t || c->cuda_rows != t->rows ||
        c->cuda_cols != t->cols) {
        rc = cuda_ds4f_mxfp4_load(c->cuda, (const uint8_t *)t->w, t->scale,
                                  t->rows, t->cols);
        if (!rc) {
            c->cuda_tensor = t;
            c->cuda_rows = t->rows;
            c->cuda_cols = t->cols;
        }
    }
    if (!rc) {
        rc = cuda_ds4f_mxfp4_gemm(c->cuda, dst, x, M, t->rows, t->cols);
    }
    if (rc && c->verbose)
        fprintf(stderr, "dual_ds4f_prefill: CUDA GEMM failed for %dx%d M=%d\n",
                t->rows, t->cols, M);
    pthread_mutex_unlock(&c->cuda_lock);
    return rc;
}

static void *dual_cuda_worker(void *opaque) {
    dual_job *j = (dual_job *)opaque;
    j->rc = 0;
    if (j->n > 0) {
        float *dst[32]; const ds4f_tensor *t[32]; const float *x[32];
        int M[32], Ys[32], Xs[32];
        const uint8_t *w[32], *s[32];
        for (int k = 0; k < j->n; ++k) {
            int i = j->idx[k];
            dst[k] = j->dst[i]; t[k] = j->t[i]; x[k] = j->x[i];
            M[k] = j->M[i]; Ys[k] = j->Ys[i]; Xs[k] = j->Xs[i];
            w[k] = (const uint8_t *)t[k]->w; s[k] = t[k]->scale;
        }
        pthread_mutex_lock(&j->ctx->cuda_lock);
        /* Async batch: all weights must already be resident (preloaded), so
         * the GEMMs overlap on the stream and one sync serves the whole
         * dispatch.  Falls back to per-call loads for a cold weight. */
        static int no_batch = -1;
        if (no_batch < 0) { const char *nb = getenv("DS4F_NO_BATCH"); no_batch = nb ? atoi(nb) : 0; }
        int rc = no_batch ? -1
                          : cuda_ds4f_mxfp4_gemm_batch(j->ctx->cuda, j->n, dst, w, s, x,
                                                       M, Ys, Xs);
        pthread_mutex_unlock(&j->ctx->cuda_lock);
        if (rc == 0) return NULL;
        if (getenv("DS4F_DBG_BATCH"))
            fprintf(stderr, "BATCH FALLBACK n=%d\n", j->n);
        for (int k = 0; k < j->n; ++k) {
            int i = j->idx[k];
            if (dual_cuda_one(j->ctx, j->dst[i], j->t[i], j->x[i], j->M[i],
                              j->Ys[i], j->Xs[i]) != 0)
                j->rc = -1;
        }
    }
    return NULL;
}

static void *dual_hip_worker(void *opaque) {
    dual_job *j = (dual_job *)opaque;
    j->rc = 0;
    if (j->n > 0) {
        float *dst[32]; const ds4f_tensor *t[32]; const float *x[32];
        int M[32], Ys[32], Xs[32];
        for (int k = 0; k < j->n; ++k) {
            int i = j->idx[k];
            dst[k] = j->dst[i]; t[k] = j->t[i]; x[k] = j->x[i];
            M[k] = j->M[i]; Ys[k] = j->Ys[i]; Xs[k] = j->Xs[i];
        }
        if (hip_ds4f_dense_gemm_tensors(j->ctx->hip, dst, t, x, M, Ys, Xs,
                                        j->n) != 0)
            j->rc = -1;
    }
    return NULL;
}

dual_ds4f_prefill *dual_ds4f_prefill_create(int hip_device, int cuda_device,
                                             int verbose) {
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->verbose = verbose;
    c->cuda_mxfp4 = 1;
    pthread_mutex_init(&c->cuda_lock, NULL);
    c->hip = hip_ds4f_dense_create(hip_device, verbose);
    c->owns_hip = 1;
    c->cuda = cuda_ds4f_mxfp4_create(cuda_device, verbose);
    if (!c->hip || !c->cuda) {
        if (verbose)
            fprintf(stderr, "dual_ds4f_prefill: create failed hip=%p cuda=%p\n",
                    (void *)c->hip, (void *)c->cuda);
        dual_ds4f_prefill_destroy(c);
        return NULL;
    }
    return c;
}

dual_ds4f_prefill *dual_ds4f_prefill_wrap_hip_budget(hip_ds4f_dense *hip,
                                                     int cuda_device, int verbose,
                                                     int cuda_cache_mb) {
    if (!hip) return NULL;
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->hip = hip;
    c->verbose = verbose;
    c->cuda_mxfp4 = 1;
    pthread_mutex_init(&c->cuda_lock, NULL);
    c->cuda = cuda_ds4f_mxfp4_create_ex(cuda_device, verbose, cuda_cache_mb);
    if (!c->cuda) {
        dual_ds4f_prefill_destroy(c);
        return NULL;
    }
    return c;
}

dual_ds4f_prefill *dual_ds4f_prefill_wrap_hip(hip_ds4f_dense *hip,
                                              int cuda_device, int verbose) {
    return dual_ds4f_prefill_wrap_hip_budget(hip, cuda_device, verbose, 12000);
}

void dual_ds4f_prefill_destroy(dual_ds4f_prefill *c) {
    if (!c) return;
    cuda_ds4f_mxfp4_destroy(c->cuda);
    if (c->owns_hip) hip_ds4f_dense_destroy(c->hip);
    pthread_mutex_destroy(&c->cuda_lock);
    free(c);
}

void dual_ds4f_prefill_set_cuda_mxfp4(dual_ds4f_prefill *c, int enabled) {
    if (c) c->cuda_mxfp4 = enabled != 0;
}

void dual_ds4f_prefill_set_cuda_terms(dual_ds4f_prefill *c, int terms) {
    if (c) cuda_ds4f_mxfp4_set_terms(c->cuda, terms);
}

void dual_ds4f_prefill_set_cuda_no_evict(dual_ds4f_prefill *c, int enabled) {
    if (c) cuda_ds4f_mxfp4_set_no_evict(c->cuda, enabled);
}

void dual_ds4f_prefill_set_max_batch(dual_ds4f_prefill *c, int max_batch) {
    if (c) c->max_batch = max_batch;
}

void dual_ds4f_prefill_set_cuda_small_buckets(dual_ds4f_prefill *c, int on) {
    if (c) c->small_buckets = on ? 1 : 0;
}

/* Warm an expert weight into the CUDA cache so the async batch path can run.
 * The repack + upload happen here (once per tensor), not inside the measured
 * prefill.  Returns 0 on success, -1 if the tensor is not CUDA-eligible. */
int dual_ds4f_prefill_warm(dual_ds4f_prefill *c, const ds4f_tensor *t) {
    if (!c || !c->cuda || !t || t->type != DS4F_MXFP4 || !t->w || !t->scale)
        return -1;
    pthread_mutex_lock(&c->cuda_lock);
    int rc = cuda_ds4f_mxfp4_warm(c->cuda, (const uint8_t *)t->w, t->scale,
                                  t->rows, t->cols);
    pthread_mutex_unlock(&c->cuda_lock);
    return rc;
}

int dual_ds4f_prefill_bind_tensor(dual_ds4f_prefill *c, ds4f_tensor *t) {
    if (!c || !t) return -1;
    /* CUDA owns raw MXFP4 in dual mode and uploads only the current matrix;
     * avoid duplicating the complete expert bank in HIP VRAM.  A nonnegative
     * sentinel keeps the common dispatcher eligible for batched prefill. */
    /* Only claim MXFP4 for CUDA when this run can actually reach the SM120
     * MMQ path's M >= 128, or when small-bucket routing is opted in (the MMQ
     * pads internally).  Otherwise the sentinel makes the common
     * dispatcher treat the tensor as device-owned, and every routed-expert
     * group then fails over one task at a time instead of using the fused CPU
     * multi-GEMM dispatch. */
    if (t->type == DS4F_MXFP4 && c->cuda_mxfp4 &&
        (c->max_batch <= 0 || c->max_batch >= 128 || c->small_buckets)) {
        if (!t->w || !t->scale || t->rows <= 0 || t->cols <= 0) return -1;
        t->gpu_id = 0;
        return 0;
    }
    if (t->type == DS4F_MXFP4 && c->cuda_mxfp4) {
        /* Deliberately CPU-owned, which is a successful bind, not a failure:
         * callers treat a negative return as a hard integration error. */
        t->gpu_id = -1;
        return 0;
    }
    if (t->type == DS4F_MXFP4) {
        /* The owned expert bank is ~17.9 GiB and does not fit alongside the
         * dense bank, so a HIP MXFP4 bind is not a usable full-model path.
         * Leave the experts CPU-owned instead of failing the whole run. */
        t->gpu_id = -1;
        return 0;
    }
    if (t->type == DS4F_BF16)
        return hip_ds4f_dense_bind_bf16_tensor(c->hip, t);
    return hip_ds4f_dense_bind_tensor(c->hip, t);
}

/* gpu_dense_ctx is a single pointer shared by every callback, so a dual
 * context cannot be mixed with HIP-bound entry points.  Forward the non-GEMM
 * operations to the wrapped HIP runner; dual only reimplements the batched
 * GEMMs.  Without these the M=1 residual matvecs silently fall back to the
 * CPU, which is a different reduction order from the single-GPU path. */
static int dual_matvec(void *opaque, float *dst, const ds4f_tensor *t,
                       const float *x) {
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)opaque;
    return hip_ds4f_dense_matvec_tensor(c->hip, dst, t, x);
}

static int dual_async_multi(void *opaque, float *const *dst,
                            const ds4f_tensor *const *t,
                            const float *const *x, int n) {
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)opaque;
    return hip_ds4f_dense_matvec_tensors_async(c->hip, dst, t, x, n);
}

static int dual_wait(void *opaque) {
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)opaque;
    return hip_ds4f_dense_wait_tensors(c->hip);
}

static int dual_blockdiag(void *opaque, float *dst, const ds4f_tensor *t,
                          const float *xbase, int gin, int glora, int goff) {
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)opaque;
    return hip_ds4f_dense_matvec_blockdiag(c->hip, dst, t, xbase,
                                           gin, glora, goff);
}

static int dual_prefill_attn(void *opaque, float *dst, const float *q,
                             const uint16_t *kv, const float *sink,
                             const float *rcos, const float *rsin,
                             int rope_offset, int rope_pairs, int M,
                             int pos0, int n_heads, int head_dim, int kv_dim,
                             int kv_slots, int window, float scale) {
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)opaque;
    return hip_ds4f_dense_prefill_attention(c->hip, dst, q, kv, sink, rcos,
                                            rsin, rope_offset, rope_pairs, M,
                                            pos0, n_heads, head_dim, kv_dim,
                                            kv_slots, window, scale);
}

static int dual_shared_ffn(void *opaque, float *dst,
                           const ds4f_tensor *w1, const ds4f_tensor *w3,
                           const ds4f_tensor *w2, const float *x,
                           int M, int inter, int C, float lim) {
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)opaque;
    return hip_ds4f_dense_shared_ffn(c->hip, dst, w1, w3, w2, x,
                                     M, inter, C, lim);
}

/* MXFP4 expert layer residency: forward to the wrapped HIP runner.  The
 * gpu_dense_ctx swap to the dual wrapper otherwise makes these HIP entry
 * points read the dual struct as a hip_ds4f_dense. */
static int dual_layer_begin(void *opaque, const ds4f_layer *layer) {
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)opaque;
    return c->hip_layer_begin ? c->hip_layer_begin(c->hip, layer) : 0;
}

static int dual_layer_prefetch(void *opaque, const ds4f_layer *layer) {
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)opaque;
    return c->hip_layer_prefetch ? c->hip_layer_prefetch(c->hip, layer) : 0;
}

void dual_ds4f_prefill_attach_model(ds4f_model *m, dual_ds4f_prefill *c) {
    if (!m) return;
    m->gpu_dense_ctx = c;
    m->gpu_dense_gemm = c ? dual_ds4f_prefill_gemm : NULL;
    m->gpu_dense_gemm_multi = c ? dual_ds4f_prefill_gemm_multi : NULL;
    m->gpu_dense_matvec = c ? dual_matvec : NULL;
    m->gpu_dense_async_multi = (c && m->gpu_dense_async_multi) ? dual_async_multi : NULL;
    m->gpu_dense_wait = c ? dual_wait : NULL;
    m->gpu_dense_blockdiag = c ? dual_blockdiag : NULL;
    m->gpu_prefill_attn = (c && m->gpu_prefill_attn) ? dual_prefill_attn : NULL;
    m->gpu_shared_ffn = c ? dual_shared_ffn : NULL;
    /* Capture the HIP MXFP4 layer callbacks before gpu_dense_ctx is swapped,
     * then forward them (dual layer residency on the wrapped HIP runner). */
    if (c) {
        c->hip_layer_begin = m->gpu_dense_layer_begin;
        c->hip_layer_prefetch = m->gpu_dense_layer_prefetch;
        m->gpu_dense_layer_begin = dual_layer_begin;
        m->gpu_dense_layer_prefetch = dual_layer_prefetch;
    } else {
        m->gpu_dense_layer_begin = NULL;
        m->gpu_dense_layer_prefetch = NULL;
    }
    m->gpu_dense_mixed = c ? 1 : 0;
}

int dual_ds4f_prefill_routed_ffn(void *opaque, float *dst, const float *x,
    const ds4f_tensor *const *w1, const ds4f_tensor *const *w3,
    const ds4f_tensor *const *w2, const int *counts, const int *offsets,
    int n_experts, int total, int C, int inter, float lim) {
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)opaque;
    if (!c || !c->cuda || !dst || !x || !w1 || !w3 || !w2 || !counts ||
        !offsets || n_experts < 1 || n_experts > 8 || total < 1 ||
        offsets[0] != 0 || offsets[n_experts] != total)
        return -1;
    /* gpu_id==-2 is the CUDA ownership sentinel established during preload.
     * Require an entire selected bundle on CUDA: mixing a partial route would
     * change scheduling/accumulation and is slower than the exact CPU group. */
    for (int e = 0; e < n_experts; ++e)
        if (counts[e] != 1 || w1[e]->gpu_id != -2 || w3[e]->gpu_id != -2 ||
            w2[e]->gpu_id != -2)
            return -1;
    float *gate = (float *)malloc((size_t)total * (size_t)inter * sizeof(float));
    float *up = (float *)malloc((size_t)total * (size_t)inter * sizeof(float));
    if (!gate || !up) { free(gate); free(up); return -1; }
    float *gd[8], *ud[8], *yd[8];
    const float *xe[8], *gx[8];
    const uint8_t *gw[8], *gs[8], *uw[8], *us[8], *dw[8], *ds[8];
    int one[8], grow[8], gcol[8], drow[8], dcol[8];
    for (int e = 0; e < n_experts; ++e) {
        int row = offsets[e];
        xe[e] = x + (size_t)row * C;
        gd[e] = gate + (size_t)row * inter;
        ud[e] = up + (size_t)row * inter;
        yd[e] = dst + (size_t)row * C;
        gw[e] = (const uint8_t *)w1[e]->w; gs[e] = w1[e]->scale;
        uw[e] = (const uint8_t *)w3[e]->w; us[e] = w3[e]->scale;
        dw[e] = (const uint8_t *)w2[e]->w; ds[e] = w2[e]->scale;
        one[e] = 1; grow[e] = inter; gcol[e] = C;
        drow[e] = C; dcol[e] = inter;
    }
    /* Dispatch all selected experts in one CUDA stream batch per projection.
     * M=1 launch/synchronization overhead dominated the original sequential
     * adapter, and hid any benefit from keeping the full expert layers hot. */
    int rc = 0;
    if (cuda_ds4f_mxfp4_gemm_batch(c->cuda, n_experts, gd, gw, gs, xe,
                                   one, grow, gcol) != 0 ||
        cuda_ds4f_mxfp4_gemm_batch(c->cuda, n_experts, ud, uw, us, xe,
                                   one, grow, gcol) != 0) rc = -1;
    for (int e = 0; e < n_experts && !rc; ++e) {
        float *ge = gd[e], *ue = ud[e];
        for (int j = 0; j < inter; ++j) {
            float v = ge[j];
            if (v > lim) v = lim;
            if (v < -lim) v = -lim;
            ge[j] = v / (1.0f + expf(-v)) * ue[j];
        }
        gx[e] = ge;
    }
    if (!rc && cuda_ds4f_mxfp4_gemm_batch(c->cuda, n_experts, yd, dw, ds, gx,
                                           one, drow, dcol) != 0) rc = -1;
    free(up); free(gate);
    return rc;
}

int dual_ds4f_prefill_gemm(void *opaque, float *dst, const ds4f_tensor *t,
                           const float *x, int M, int Ys, int Xs) {
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)opaque;
    if (!c || !dst || !t || !x) return -1;
    if (t->type == DS4F_MXFP4 && !c->small_buckets && M < 128) return -1;
    if (cuda_eligible(c, t, M, Ys, Xs) &&
        dual_cuda_one(c, dst, t, x, M, Ys, Xs) == 0)
        return 0;
    return hip_ds4f_dense_gemm_tensor(c->hip, dst, t, x, M, Ys, Xs);
}

int dual_ds4f_prefill_gemm_multi(
    void *opaque, float *const *dst, const ds4f_tensor *const *t,
    const float *const *x, const int *M, const int *Ys, const int *Xs,
    int n) {
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)opaque;
    if (!c || !dst || !t || !x || !M || !Ys || !Xs || n < 1 || n > 32)
        return -1;
    if (!c->small_buckets)
        for (int i = 0; i < n; ++i)
            if (t[i] && t[i]->type == DS4F_MXFP4 && M[i] < 128)
                return -1; /* force ds4f_gemm()'s exact CPU fallback */
    int ci[32], hi[32], nc = 0, nh = 0;
    for (int i = 0; i < n; ++i) {
        if (cuda_eligible(c, t[i], M[i], Ys[i], Xs[i])) ci[nc++] = i;
        else hi[nh++] = i;
        if (c->verbose)
            fprintf(stderr, "dual task %d type=%d M=%d rows=%d cols=%d cuda=%d\n",
                    i, t[i] ? t[i]->type : -1, M[i], t[i] ? t[i]->rows : 0,
                    t[i] ? t[i]->cols : 0,
                    cuda_eligible(c, t[i], M[i], Ys[i], Xs[i]));
    }
    if (!nc) return hip_ds4f_dense_gemm_tensors(c->hip, dst, t, x, M, Ys, Xs, n);
    if (!nh) {
        dual_job j = { c, dst, t, x, M, Ys, Xs, ci, nc, 0 };
        dual_cuda_worker(&j);
        return j.rc;
    }
    dual_job cj = { c, dst, t, x, M, Ys, Xs, ci, nc, 0 };
    dual_job hj = { c, dst, t, x, M, Ys, Xs, hi, nh, 0 };
    pthread_t ct, ht;
    if (pthread_create(&ct, NULL, dual_cuda_worker, &cj) != 0) return -1;
    if (pthread_create(&ht, NULL, dual_hip_worker, &hj) != 0) {
        pthread_join(ct, NULL);
        return -1;
    }
    pthread_join(ct, NULL);
    pthread_join(ht, NULL);
    return cj.rc || hj.rc ? -1 : 0;
}
