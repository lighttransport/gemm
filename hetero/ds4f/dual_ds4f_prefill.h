#ifndef DUAL_DS4F_PREFILL_H
#define DUAL_DS4F_PREFILL_H

#include "hip_ds4f_dense.h"
#include "cuda_ds4f_mxfp4.h"
#include "../../common/ds4f.h"

typedef struct dual_ds4f_prefill dual_ds4f_prefill;

dual_ds4f_prefill *dual_ds4f_prefill_create(int hip_device, int cuda_device,
                                             int verbose);
dual_ds4f_prefill *dual_ds4f_prefill_wrap_hip(hip_ds4f_dense *hip,
                                              int cuda_device, int verbose);
dual_ds4f_prefill *dual_ds4f_prefill_wrap_hip_budget(hip_ds4f_dense *hip,
                                                     int cuda_device, int verbose,
                                                     int cuda_cache_mb);
void dual_ds4f_prefill_destroy(dual_ds4f_prefill *ctx);
void dual_ds4f_prefill_set_cuda_mxfp4(dual_ds4f_prefill *ctx, int enabled);
void dual_ds4f_prefill_set_cuda_terms(dual_ds4f_prefill *ctx, int terms);
void dual_ds4f_prefill_set_cuda_no_evict(dual_ds4f_prefill *ctx, int enabled);
/* Opt into routing small MXFP4 expert buckets (M < 128) through the padded
 * SM120 MMQ instead of the exact CPU fallback.  Approximate (the MMQ quantizes
 * activations) and PCIe-bound on single-prefill runs; mainly useful for a
 * resident-weight server. */
void dual_ds4f_prefill_set_cuda_small_buckets(dual_ds4f_prefill *ctx, int on);
/* Largest M this run can present to a routed-expert GEMM.  The SM120 MMQ path
 * needs M >= 128, so below that CUDA can never serve MXFP4 and those tensors
 * must stay CPU-owned rather than carrying a device sentinel. */
void dual_ds4f_prefill_set_max_batch(dual_ds4f_prefill *ctx, int max_batch);
int dual_ds4f_prefill_bind_tensor(dual_ds4f_prefill *ctx, ds4f_tensor *t);
/* Preload an expert weight into the CUDA cache so the async batch path runs
 * (its GEMMs overlap on the stream; cold weights fall back to per-call). */
int dual_ds4f_prefill_warm(dual_ds4f_prefill *ctx, const ds4f_tensor *t);
/* M=1 routed-expert adapter.  It claims a layer only when every selected
 * expert is CUDA-resident; returning nonzero preserves the common CPU path. */
int dual_ds4f_prefill_routed_ffn(void *ctx, float *dst, const float *x,
    const ds4f_tensor *const *w1, const ds4f_tensor *const *w3,
    const ds4f_tensor *const *w2, const int *counts, const int *offsets,
    int n_experts, int total, int hidden, int inter, float limit);
void dual_ds4f_prefill_attach_model(ds4f_model *model,
                                    dual_ds4f_prefill *ctx);

int dual_ds4f_prefill_gemm(void *ctx, float *dst, const ds4f_tensor *t,
                           const float *x, int M, int Ystride, int Xstride);
int dual_ds4f_prefill_gemm_multi(
    void *ctx, float *const *dst, const ds4f_tensor *const *t,
    const float *const *x, const int *M, const int *Ystride,
    const int *Xstride, int n);

#endif
