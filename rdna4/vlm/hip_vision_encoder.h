/*
 * hip_vision_encoder.h - HIP/ROCm vision encoder for Qwen3-VL mmproj
 *
 * Uses HIPRTC to compile HIP C kernels at runtime (no hipcc needed).
 * Supports F32 (verification), F16, and BF16 weight modes.
 * Targets RDNA4 (gfx1200/gfx1201).
 *
 * Usage:
 *   hip_vision_runner *r = hip_vision_init(0, 1, 0);
 *   hip_vision_load_weights(r, mmproj_gguf);
 *   float *embd = hip_vision_encode(r, rgb_norm, width, height);
 *   hip_vision_free(r);
 */
#ifndef HIP_VISION_ENCODER_H
#define HIP_VISION_ENCODER_H

#include <stdint.h>
#include "../../common/gguf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hip_vision_runner hip_vision_runner;

/* Initialize HIP vision encoder.
 * device_id: HIP device index (0 = first GPU)
 * verbose: 0=quiet, 1=info, 2=debug, 3=dump code object
 * use_f16: 0=F32 weights, 1=F16 weights, 2=BF16 weights. Activations and accumulators stay F32. */
hip_vision_runner *hip_vision_init(int device_id, int verbose, int use_f16);

/* Infer preferred weight precision from the mmproj GGUF tensor storage.
 * Returns 0=F32, 1=F16, 2=BF16. */
int hip_vision_infer_precision(const gguf_context *mmproj_gguf);

/* Set maximum pixel budget for dynamic resolution. Must be called BEFORE load_weights.
 * max_pixels=0 means use model default (image_size^2). */
void hip_vision_set_max_pixels(hip_vision_runner *r, int max_pixels);

/* Load model weights from GGUF. Returns 0 on success, -1 on error. */
int hip_vision_load_weights(hip_vision_runner *r, gguf_context *mmproj_gguf);

/* Encode an image. rgb_norm is [height * width * 3] normalized float RGB (interleaved).
 * Returns malloc'd float array of [n_merged * total_embd].
 * Caller must free the result. */
float *hip_vision_encode(hip_vision_runner *r, const float *rgb_norm, int width, int height);

/* Cleanup */
void hip_vision_free(hip_vision_runner *r);

/* Accessors */
int hip_vision_n_merged(const hip_vision_runner *r);
int hip_vision_proj_dim(const hip_vision_runner *r);
int hip_vision_total_embd(const hip_vision_runner *r);

#ifdef __cplusplus
}
#endif

#endif /* HIP_VISION_ENCODER_H */
