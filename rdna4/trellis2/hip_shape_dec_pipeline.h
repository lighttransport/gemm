/*
 * hip_shape_dec_pipeline.h - HIP shape_dec forward pipeline (RDNA4)
 *
 * Factored out of test_hip_tex_dec.c (Milestone C). The runner and the
 * standalone test both call into this single entry point.
 *
 * The ctx owns:
 *   - hipModule_t       (caller-provided; not freed by ctx)
 *   - kernel function lookup table (K)
 *   - hipBLASLt + Triton bridges (lazily, on _create)
 *   - persistent device scratch (BF16 act, F32/F16/BF16 weight caches)
 *   - C2S transient scratch (grow-on-demand, shared across forward calls)
 *
 * Stream is caller-provided. Caller is responsible for hipDeviceSynchronize /
 * hipStreamSynchronize before reading the device output buffers.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef HIP_SHAPE_DEC_PIPELINE_H
#define HIP_SHAPE_DEC_PIPELINE_H

#include <stdint.h>
#include "../rocew.h"
#include "../../common/trellis2_shape_decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hip_shape_dec_ctx hip_shape_dec_ctx;

/* Optional pre-computed C2S guide arrays (one entry per stage, NULL == unguided
 * synth via to_subdiv head). Used by the test harness with --cache; the runner
 * passes NULL for full unguided forward. */
typedef struct {
    int     n_stages;
    int64_t *gi[8];      /* [Nf] */
    int64_t *gs[8];      /* [Nf] (widened from int32 on load) */
    int32_t *gxc[8];     /* [Nf, 4] */
    int      gN[8];
    /* Optional flex_gemm neighbor maps (per stage). */
    uint32_t *nmap_cn[8]; int nmap_cn_N[8];
    uint32_t *nmap_pc[8]; int nmap_pc_N[8];
} hip_shape_dec_cache;

/* Create a forward-only context. module must be a HIP module containing the
 * tex_dec kernels (compiled from hip_tex_dec_kernels_src). */
hip_shape_dec_ctx *hip_shape_dec_ctx_create(hipModule_t module,
                                            hipStream_t stream,
                                            const t2_shape_dec *dec,
                                            int verbose);

void hip_shape_dec_ctx_free(hip_shape_dec_ctx *ctx);

/* Run shape_dec forward end-to-end (input_layer + 4× {convnext + c2s} +
 * output_layer). cache may be NULL for unguided synth.
 *
 * On success, *out_d_feats holds [Nf, 7] f32 device buffer and *out_d_coords
 * holds [Nf, 4] i32 device buffer (caller frees with hipFree). */
int hip_shape_dec_forward_ex(hip_shape_dec_ctx *ctx,
                             const float *slat_feats,
                             const int32_t *coords,
                             int N, int slat_C,
                             const hip_shape_dec_cache *cache,
                             float **out_d_feats,
                             int32_t **out_d_coords,
                             int *out_Nf);

/* Convenience wrapper: unguided forward (cache == NULL). */
int hip_shape_dec_forward(hip_shape_dec_ctx *ctx,
                          const float *slat_feats,
                          const int32_t *coords,
                          int N, int slat_C,
                          float **out_d_feats,
                          int32_t **out_d_coords,
                          int *out_Nf);

#ifdef __cplusplus
}
#endif

#endif
