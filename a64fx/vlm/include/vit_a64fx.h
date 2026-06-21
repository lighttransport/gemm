/*
 * vit_a64fx.h - public API for the a64fx Qwen3-VL vision encoder runner.
 *
 * Reuses vision_load / vision_free from common/vision_encoder.h (loads the
 * mmproj GGUF into a `vision_model *`). This API replaces vision_encode()
 * with a SVE/threaded path and adds an optional per-stage tensor dump.
 */
#ifndef VIT_A64FX_H
#define VIT_A64FX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward decls so callers don't need to drag in common headers */
struct vision_model;
struct vlm_pool;
struct vlmd_writer;
struct vit_a64fx_cache;

enum vit_dtype {
    VIT_DTYPE_FP32 = 0,
    VIT_DTYPE_BF16 = 1,   /* M3 */
    VIT_DTYPE_FP16 = 2,   /* M4 */
};

typedef struct vit_a64fx_opts {
    int                          dtype;       /* enum vit_dtype */
    struct vlm_pool             *pool;        /* required */
    struct vlmd_writer          *dump;        /* may be NULL */
    int                          enable_prof; /* 1 = call prof_begin/end */
    struct vit_a64fx_cache      *cache;       /* optional pre-dequant weight cache */
} vit_a64fx_opts;

/* Pre-dequantize + transpose every weight in vm once. The returned cache lives
 * as long as vm; pass it via vit_a64fx_opts.cache to skip per-encode dequant
 * and transpose. NULL on alloc failure. Thread-safe to share across encodes.
 *
 * dtype = VIT_DTYPE_FP32 stores FP32 BT weights.
 * dtype = VIT_DTYPE_BF16 stores BF16 mirrors of all GEMM weights (half memory).
 *                       The encode path then calls gemm_bf16_BT. */
struct vit_a64fx_cache *vit_a64fx_cache_build(struct vision_model *vm, int dtype);
void vit_a64fx_cache_free(struct vit_a64fx_cache *cache);

/* Replicate every packed-B weight buffer across n_cmgs CMG-local pools
 * (one copy per CMG via cmg_alloc + mbind). After this call, GEMM dispatch
 * picks the local replica based on the running thread's CMG. The legacy
 * single-pointer fields are retargeted to replica 0 so non-CMG-aware paths
 * keep working. Returns 0 on success, -1 on error.
 *
 * Only valid for BF16/FP16 dtypes (FP32 cache has no packed-B form). */
int vit_a64fx_cache_replicate(struct vit_a64fx_cache *cache, int n_cmgs);

/* Encode an image with the a64fx-optimized pipeline.
 *
 *   vm        : loaded by vision_load() from common/vision_encoder.h
 *   rgb_norm  : [height*width*3] mean/std-normalized FP32
 *   width/height: image dims (must be multiples of patch_size)
 *   out_n_merged / out_embd : optional output shape — final buffer has
 *                             out_n_merged * out_embd FP32 elements
 *
 * Returns a malloc'd float buffer the caller must free, or NULL on error.
 * Output layout matches the scalar reference: [n_merged, proj_dim*(1+n_ds)].
 */
float *vit_a64fx_encode(struct vision_model *vm,
                        const float *rgb_norm,
                        int width, int height,
                        const vit_a64fx_opts *opts,
                        int *out_n_merged, int *out_embd);

#ifdef __cplusplus
}
#endif

#endif /* VIT_A64FX_H */
