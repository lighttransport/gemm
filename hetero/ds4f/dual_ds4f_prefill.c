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
};

static int cuda_eligible(const dual_ds4f_prefill *c, const ds4f_tensor *t,
                         int M, int Ys, int Xs) {
    return c && c->cuda_mxfp4 && t && t->type == DS4F_MXFP4 && t->w && t->scale && M >= 1 &&
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
    for (int k = 0; k < j->n; ++k) {
        int i = j->idx[k];
        if (dual_cuda_one(j->ctx, j->dst[i], j->t[i], j->x[i], j->M[i],
                          j->Ys[i], j->Xs[i]) != 0)
            j->rc = -1;
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

dual_ds4f_prefill *dual_ds4f_prefill_wrap_hip(hip_ds4f_dense *hip,
                                              int cuda_device, int verbose) {
    if (!hip) return NULL;
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->hip = hip;
    c->verbose = verbose;
    c->cuda_mxfp4 = 1;
    pthread_mutex_init(&c->cuda_lock, NULL);
    c->cuda = cuda_ds4f_mxfp4_create(cuda_device, verbose);
    if (!c->cuda) {
        dual_ds4f_prefill_destroy(c);
        return NULL;
    }
    return c;
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

int dual_ds4f_prefill_bind_tensor(dual_ds4f_prefill *c, ds4f_tensor *t) {
    if (!c || !t) return -1;
    /* CUDA owns raw MXFP4 in dual mode and uploads only the current matrix;
     * avoid duplicating the complete expert bank in HIP VRAM.  A nonnegative
     * sentinel keeps the common dispatcher eligible for batched prefill. */
    if (t->type == DS4F_MXFP4 && c->cuda_mxfp4) {
        if (!t->w || !t->scale || t->rows <= 0 || t->cols <= 0) return -1;
        t->gpu_id = 0;
        return 0;
    }
    if (t->type == DS4F_MXFP4)
        return hip_ds4f_dense_bind_mxfp4_tensor(c->hip, t);
    if (t->type == DS4F_BF16)
        return hip_ds4f_dense_bind_bf16_tensor(c->hip, t);
    return hip_ds4f_dense_bind_tensor(c->hip, t);
}

void dual_ds4f_prefill_attach_model(ds4f_model *m, dual_ds4f_prefill *c) {
    if (!m) return;
    m->gpu_dense_ctx = c;
    m->gpu_dense_gemm = c ? dual_ds4f_prefill_gemm : NULL;
    m->gpu_dense_gemm_multi = c ? dual_ds4f_prefill_gemm_multi : NULL;
    m->gpu_dense_mixed = c ? 1 : 0;
}

int dual_ds4f_prefill_gemm(void *opaque, float *dst, const ds4f_tensor *t,
                           const float *x, int M, int Ys, int Xs) {
    dual_ds4f_prefill *c = (dual_ds4f_prefill *)opaque;
    if (!c || !dst || !t || !x) return -1;
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
