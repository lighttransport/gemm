// SPDX-License-Identifier: MIT
#ifndef PIXAL3D_MINIMAX_H3_H
#define PIXAL3D_MINIMAX_H3_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct h3_context h3_context;
typedef struct {
    const char *model_dir;
    int device, vram_budget_mib;
    int bf16_hipblas;            /* 1=reference-order dense GEMM, 0=native BF16 WMMA */
    int convrot_hipblas;         /* 1=dense BF16 Hadamard, 0=factorized native rotation */
    const char *aotriton_bridge; /* Optional libvideo_aotriton.so for long BF16 attention */
    int vae_hipblas;             /* Optional FP16 decoder GEMM; default native WMMA */
} h3_config;
typedef struct {
    const char *prompt, *noise_file, *audio_noise_file, *dump_dir;
    int width, height, frames, steps;
    int64_t seed;
} h3_request;
typedef struct {
    void (*progress)(int step, int total, void *user);
    int (*frame)(int index, int width, int height, const unsigned char *rgb, void *user);
    int (*cancelled)(void *user);
    void *user;
} h3_callbacks;
void h3_config_defaults(h3_config *config);
void h3_request_defaults(h3_request *request);
int h3_validate(const h3_request *request, char *error, size_t capacity);
h3_context *h3_load(const h3_config *config, char *error, size_t capacity);
/* Idle-context selection; default 1. Configuration struct layout is unchanged. */
int h3_set_fp32_hipblas(h3_context *context, int enabled, char *error, size_t capacity);
/* Idle-context image conditioning. variant="ref2va" or "fl2va"; directory is a
 * caller-verified h3.image_conditioning.v1 bundle, or NULL for text-only generation.
 * Existing configuration/request ABI layouts remain unchanged. */
int h3_set_conditioning(h3_context *context, const char *variant, const char *directory,
                        char *error, size_t capacity);
/* CUDA only: opt-in cuDNN SDPA for DiT attention on an idle context. mode is NULL/"off"
 * (default, private FlashAttention-2), "auto" (libh3_cudnn.so next to this library or the
 * executable; falls back to FlashAttention-2 with a warning if unavailable), or an explicit
 * bridge path (errors if unusable). cudnn_library is a libcudnn.so.9 path, or NULL to use
 * the default loader search and fixed system locations. */
int h3_set_cudnn_attention(h3_context *context, const char *mode, const char *cudnn_library,
                           char *error, size_t capacity);
int h3_generate(h3_context *context, const h3_request *request, const h3_callbacks *callbacks,
                char *error, size_t capacity);
const char *h3_metrics(const h3_context *context);
void h3_cancel(h3_context *context);
void h3_free(h3_context *context);
#ifdef __cplusplus
}
#endif
#endif
