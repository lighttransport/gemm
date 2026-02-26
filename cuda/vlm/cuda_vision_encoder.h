/*
 * cuda_vision_encoder.h - CUDA vision encoder for Qwen3-VL mmproj
 *
 * Uses NVRTC to compile CUDA C kernels at runtime (no nvcc needed).
 * Supports F32 (verification) and F16 (performance) weight modes.
 * Works on sm_70+ (V100 through Blackwell).
 *
 * Usage:
 *   cuda_vision_runner *r = cuda_vision_init(0, 1, 0);
 *   cuda_vision_load_weights(r, mmproj_gguf);
 *   float *embd = cuda_vision_encode(r, rgb_norm, width, height);
 *   cuda_vision_free(r);
 */
#ifndef CUDA_VISION_ENCODER_H
#define CUDA_VISION_ENCODER_H

#include <stdint.h>
#include "../../common/gguf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_vision_runner cuda_vision_runner;

/* Initialize CUDA vision encoder.
 * device_id: CUDA device index (0 = first GPU)
 * verbose: 0=quiet, 1=info, 2=debug, 3=dump PTX
 * use_f16: 0=F32 weights (exact match with CPU), 1=F16 weights (faster, ~1e-2 error) */
cuda_vision_runner *cuda_vision_init(int device_id, int verbose, int use_f16);

/* Load model weights from GGUF. Returns 0 on success, -1 on error. */
int cuda_vision_load_weights(cuda_vision_runner *r, gguf_context *mmproj_gguf);

/* Encode an image. rgb_norm is [height * width * 3] normalized float RGB (interleaved).
 * Returns malloc'd float array of [n_merged * total_embd].
 * Caller must free the result. */
float *cuda_vision_encode(cuda_vision_runner *r, const float *rgb_norm, int width, int height);

/* Cleanup */
void cuda_vision_free(cuda_vision_runner *r);

/* Accessors */
int cuda_vision_n_merged(const cuda_vision_runner *r);
int cuda_vision_proj_dim(const cuda_vision_runner *r);
int cuda_vision_total_embd(const cuda_vision_runner *r);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_VISION_ENCODER_H */
