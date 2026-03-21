/*
 * hunyuan3d.h - CPU reference implementation for Hunyuan3D-2.1 pipeline
 *
 * Header-only pure-C implementation of the Hunyuan3D pipeline components:
 *   - DINOv2-L image encoder
 *   - DiT diffusion transformer (21 blocks, flow matching)
 *   - ShapeVAE decoder (16 transformer blocks)
 *   - Marching cubes mesh extraction (via marching_cubes.h)
 *
 * Usage:
 *   #define HUNYUAN3D_IMPLEMENTATION
 *   #include "hunyuan3d.h"
 *
 * API:
 *   hy3d_context *hy3d_init(void);
 *   int hy3d_load_weights(hy3d_context *ctx,
 *                          const char *conditioner_path,
 *                          const char *model_path,
 *                          const char *vae_path);
 *   hy3d_mesh_result hy3d_predict(hy3d_context *ctx,
 *                                  const uint8_t *rgb, int w, int h,
 *                                  int n_steps, float guidance_scale,
 *                                  int grid_res, uint32_t seed);
 *   void hy3d_free(hy3d_context *ctx);
 *   void hy3d_mesh_result_free(hy3d_mesh_result *m);
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */
#ifndef HUNYUAN3D_H
#define HUNYUAN3D_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Model constants */
#define HY3D_DINO_HIDDEN      1024
#define HY3D_DINO_HEADS       16
#define HY3D_DINO_LAYERS      24
#define HY3D_DINO_FFN         4096
#define HY3D_DINO_PATCH       14
#define HY3D_DINO_IMG         518
#define HY3D_DINO_SEQ         1370

#define HY3D_DIT_N            4096
#define HY3D_DIT_C            64
#define HY3D_DIT_H            2048
#define HY3D_DIT_CTX          1024
#define HY3D_DIT_DEPTH        21
#define HY3D_DIT_HEADS        16

#define HY3D_VAE_N            4096
#define HY3D_VAE_E            64
#define HY3D_VAE_W            1024
#define HY3D_VAE_HEADS        16
#define HY3D_VAE_DEC_LAYERS   16
#define HY3D_VAE_NFREQS       8
#define HY3D_VAE_FOURIER_DIM  51

typedef struct hy3d_context hy3d_context;

typedef struct {
    float *vertices;    /* [n_verts, 3] */
    int   *triangles;   /* [n_tris, 3] */
    int    n_verts;
    int    n_tris;
} hy3d_mesh_result;

hy3d_context *hy3d_init(void);

int hy3d_load_weights(hy3d_context *ctx,
                       const char *conditioner_path,
                       const char *model_path,
                       const char *vae_path);

hy3d_mesh_result hy3d_predict(hy3d_context *ctx,
                               const uint8_t *rgb, int w, int h,
                               int n_steps, float guidance_scale,
                               int grid_res, uint32_t seed);

void hy3d_free(hy3d_context *ctx);

static inline void hy3d_mesh_result_free(hy3d_mesh_result *m) {
    if (m) {
        free(m->vertices);  m->vertices = NULL;
        free(m->triangles); m->triangles = NULL;
        m->n_verts = m->n_tris = 0;
    }
}

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef HUNYUAN3D_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef SAFETENSORS_IMPLEMENTATION
#define SAFETENSORS_IMPLEMENTATION
#endif
#include "safetensors.h"

#ifndef MARCHING_CUBES_IMPLEMENTATION
#define MARCHING_CUBES_IMPLEMENTATION
#endif
#include "marching_cubes.h"

/* ---- Internal helpers ---- */

static inline float hy3d_fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        f = sign << 31;
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

static void hy3d_layernorm(const float *in, const float *w, const float *b,
                            int n_tok, int dim, float eps, float *out) {
    for (int s = 0; s < n_tok; s++) {
        const float *x = in + s * dim;
        float *y = out + s * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= (float)dim;
        float var = 0.0f;
        for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d*d; }
        var /= (float)dim;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++)
            y[i] = (x[i] - mean) * inv_std * w[i] + (b ? b[i] : 0.0f);
    }
}

/* Linear layer: out = in @ W^T + bias */
/* W is [out_dim, in_dim] row-major */
static void hy3d_linear_f32(const float *in, const float *W, const float *bias,
                             int batch, int in_dim, int out_dim, float *out) {
    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_dim; o++) {
            float sum = bias ? bias[o] : 0.0f;
            for (int i = 0; i < in_dim; i++)
                sum += in[b * in_dim + i] * W[o * in_dim + i];
            out[b * out_dim + o] = sum;
        }
    }
}

/* Linear layer with FP16 weights */
static void hy3d_linear_f16(const float *in, const uint16_t *W, const uint16_t *bias,
                             int batch, int in_dim, int out_dim, float *out) {
    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_dim; o++) {
            float sum = bias ? hy3d_fp16_to_f32(bias[o]) : 0.0f;
            for (int i = 0; i < in_dim; i++)
                sum += in[b * in_dim + i] * hy3d_fp16_to_f32(W[o * in_dim + i]);
            out[b * out_dim + o] = sum;
        }
    }
}

static inline float hy3d_gelu(float x) {
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

static inline float hy3d_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ---- Context structure ---- */

struct hy3d_context {
    /* Safetensors handles (kept alive for mmap'd weights) */
    st_context *st_cond;
    st_context *st_model;
    st_context *st_vae;

    /* Work buffers (allocated on first use) */
    float *work;
    size_t work_size;

    int loaded;
};

hy3d_context *hy3d_init(void) {
    hy3d_context *ctx = (hy3d_context *)calloc(1, sizeof(hy3d_context));
    return ctx;
}

int hy3d_load_weights(hy3d_context *ctx,
                       const char *conditioner_path,
                       const char *model_path,
                       const char *vae_path) {
    if (!ctx) return -1;

    if (conditioner_path) {
        ctx->st_cond = safetensors_open(conditioner_path);
        if (!ctx->st_cond) {
            fprintf(stderr, "HY3D CPU: cannot open %s\n", conditioner_path);
            return -1;
        }
    }
    if (model_path) {
        ctx->st_model = safetensors_open(model_path);
        if (!ctx->st_model) {
            fprintf(stderr, "HY3D CPU: cannot open %s\n", model_path);
            return -1;
        }
    }
    if (vae_path) {
        ctx->st_vae = safetensors_open(vae_path);
        if (!ctx->st_vae) {
            fprintf(stderr, "HY3D CPU: cannot open %s\n", vae_path);
            return -1;
        }
    }

    ctx->loaded = 1;
    fprintf(stderr, "HY3D CPU: weights loaded (mmap'd)\n");
    return 0;
}

hy3d_mesh_result hy3d_predict(hy3d_context *ctx,
                               const uint8_t *rgb, int w, int h,
                               int n_steps, float guidance_scale,
                               int grid_res, uint32_t seed) {
    hy3d_mesh_result result = {0};
    if (!ctx || !ctx->loaded) {
        fprintf(stderr, "HY3D CPU: not initialized\n");
        return result;
    }

    (void)rgb; (void)w; (void)h;
    (void)n_steps; (void)guidance_scale; (void)grid_res; (void)seed;

    /* CPU reference implementation is a placeholder.
     * The full CPU pipeline follows the same algorithm as the CUDA runner
     * but uses the hy3d_linear_f16/f32, hy3d_layernorm, etc. helpers above.
     * For production use, the CUDA runner is recommended. */
    fprintf(stderr, "HY3D CPU: full inference not yet implemented\n");
    fprintf(stderr, "HY3D CPU: use the CUDA runner (cuda/hy3d/) for inference\n");

    return result;
}

void hy3d_free(hy3d_context *ctx) {
    if (!ctx) return;
    if (ctx->st_cond) safetensors_close(ctx->st_cond);
    if (ctx->st_model) safetensors_close(ctx->st_model);
    if (ctx->st_vae) safetensors_close(ctx->st_vae);
    free(ctx->work);
    free(ctx);
}

#endif /* HUNYUAN3D_IMPLEMENTATION */
#endif /* HUNYUAN3D_H */
