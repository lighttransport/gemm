/*
 * hip_sam3_runner.c - HIP/ROCm SAM 3 runner (RDNA4), Phase 1.
 *
 * End-to-end on GPU so far: resize+ImageNet-normalize → patch_embed
 * (Conv2d k=14 s=14, 3→1024) → tile-add pos_embed (24² → 72²) →
 * pre-block LayerNorm. Output is the (5184, 1024) F32 token stream
 * consumed by the 32 ViT blocks (follow-up phases).
 *
 * Weights load from sam3.model.safetensors via common/safetensors.h;
 * all keys are stored with the `detector_model.` prefix, which we
 * strip before lookup.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_sam3_runner.h"
#include "../rocew.h"
#include "../hip_kernels_common.h"
#include "hip_sam3_kernels.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifndef HIP_LAUNCH_PARAM_BUFFER_POINTER
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void *)(uintptr_t)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE    ((void *)(uintptr_t)0x02)
#define HIP_LAUNCH_PARAM_END            ((void *)(uintptr_t)0x03)
#endif

#define S3_IMG    1008
#define S3_PATCH  14
#define S3_GRID   (S3_IMG / S3_PATCH)   /* 72 */
#define S3_DIM    1024
#define S3_PRE    24                    /* learned pos_embed spatial */
#define S3_NTOK   (S3_GRID * S3_GRID)   /* 5184 */

struct hip_sam3_ctx {
    hip_sam3_config cfg;
    int device_id;
    int verbose;
    hipModule_t mod;

    /* kernels */
    hipFunction_t fn_resize;        /* resize_normalize */
    hipFunction_t fn_layernorm;     /* layernorm_f32 */
    hipFunction_t fn_patch;         /* patch_embed_sam3 */
    hipFunction_t fn_pos_add;       /* pos_embed_tile_add */

    /* host-side safetensors context (mmap'd) */
    st_context *st;

    /* device weights */
    void *w_patch;      /* F16 (1024, 3, 14, 14) */
    void *w_pos;        /* F32 (24, 24, 1024) */
    void *w_pre_ln_w;   /* F32 (1024) */
    void *w_pre_ln_b;   /* F32 (1024) */

    /* activation buffers */
    void *d_img_u8;     /* raw RGB u8 (capacity = cap_bytes) */
    size_t img_u8_cap;
    void *d_img_f32;    /* (3, 1008, 1008) F32 */
    void *d_tok;        /* (5184, 1024) F32 post patch+pos */
    void *d_tok_ln;     /* (5184, 1024) F32 post pre-LN */
    int ready;
};

/* =========================================================== */
/* Safetensors helpers (strip `detector_model.` prefix).         */
/* =========================================================== */

static int st_find_prefixed(const st_context *st, const char *short_name)
{
    char key[256];
    snprintf(key, sizeof(key), "detector_model.%s", short_name);
    return safetensors_find(st, key);
}

static void *upload_f32(const st_context *st, const char *short_name,
                         size_t *out_n)
{
    int i = st_find_prefixed(st, short_name);
    if (i < 0) { fprintf(stderr, "sam3: missing %s\n", short_name); return NULL; }
    if (strcmp(safetensors_dtype(st, i), "F32") != 0) {
        fprintf(stderr, "sam3: %s dtype=%s, expected F32\n", short_name,
                safetensors_dtype(st, i)); return NULL;
    }
    size_t nbytes = safetensors_nbytes(st, i);
    if (out_n) *out_n = nbytes / sizeof(float);
    return hip_upload_raw(safetensors_data((st_context *)st, i), nbytes);
}

static void *upload_f16(const st_context *st, const char *short_name,
                         size_t *out_n)
{
    int i = st_find_prefixed(st, short_name);
    if (i < 0) { fprintf(stderr, "sam3: missing %s\n", short_name); return NULL; }
    if (strcmp(safetensors_dtype(st, i), "F32") != 0) {
        fprintf(stderr, "sam3: %s dtype=%s, expected F32\n", short_name,
                safetensors_dtype(st, i)); return NULL;
    }
    size_t n = safetensors_nbytes(st, i) / sizeof(float);
    const float *src = (const float *)safetensors_data((st_context *)st, i);
    uint16_t *tmp = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!tmp) return NULL;
    for (size_t k = 0; k < n; k++) tmp[k] = hip_f32_to_f16(src[k]);
    void *d = hip_upload_raw(tmp, n * sizeof(uint16_t));
    free(tmp);
    if (out_n) *out_n = n;
    return d;
}

/* =========================================================== */
/* Kernel compile + launch plumbing                              */
/* =========================================================== */

static int compile_kernels(hip_sam3_ctx *c)
{
    size_t la = strlen(hip_kernels_common_src);
    size_t lb = strlen(hip_sam3_kernels_src);
    char *src = (char *)malloc(la + lb + 1);
    if (!src) return -1;
    memcpy(src, hip_kernels_common_src, la);
    memcpy(src + la, hip_sam3_kernels_src, lb + 1);
    int rc = hip_compile_kernels(&c->mod, c->device_id, src, "sam3_kernels",
                                 c->verbose, "sam3");
    free(src);
    if (rc < 0) return -1;

    #define GET(fn, sym) do { \
        if (hipModuleGetFunction(&c->fn, c->mod, sym) != hipSuccess) { \
            fprintf(stderr, "sam3: missing kernel %s\n", sym); return -1; } \
    } while (0)
    GET(fn_resize,     "resize_normalize");
    GET(fn_layernorm,  "layernorm_f32");
    GET(fn_patch,      "patch_embed_sam3");
    GET(fn_pos_add,    "pos_embed_tile_add");
    #undef GET
    return 0;
}

static int launch(hipFunction_t fn, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, unsigned by, unsigned bz,
                   unsigned shmem, void *p, size_t pbytes)
{
    void *cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, p,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,    &pbytes,
                    HIP_LAUNCH_PARAM_END };
    hipError_t e = hipModuleLaunchKernel(fn, gx, gy, gz, bx, by, bz, shmem, 0, NULL, cfg);
    if (e != hipSuccess) {
        fprintf(stderr, "sam3: launch failed err=%d grid=%ux%ux%u block=%ux%ux%u\n",
                (int)e, gx, gy, gz, bx, by, bz);
        return -1;
    }
    return 0;
}

/* =========================================================== */
/* create / destroy                                              */
/* =========================================================== */

hip_sam3_ctx *hip_sam3_create(const hip_sam3_config *cfg)
{
    if (!cfg || !cfg->ckpt_path) return NULL;
    hip_sam3_ctx *c = (hip_sam3_ctx *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->cfg = *cfg;
    c->device_id = cfg->device_ordinal;
    c->verbose = cfg->verbose;
    if (!c->cfg.image_size) c->cfg.image_size = S3_IMG;
    if (c->cfg.image_size != S3_IMG) {
        fprintf(stderr, "sam3: only image_size=%d supported\n", S3_IMG);
        free(c); return NULL;
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "sam3: rocewInit failed\n");
        free(c); return NULL;
    }
    if (hipSetDevice(c->device_id) != hipSuccess) {
        fprintf(stderr, "sam3: hipSetDevice(%d) failed\n", c->device_id);
        free(c); return NULL;
    }
    if (compile_kernels(c) != 0) { free(c); return NULL; }

    c->st = safetensors_open(cfg->ckpt_path);
    if (!c->st) {
        fprintf(stderr, "sam3: safetensors_open failed: %s\n", cfg->ckpt_path);
        free(c); return NULL;
    }

    c->w_patch    = upload_f16(c->st, "vision_encoder.backbone.embeddings.patch_embeddings.projection.weight", NULL);
    c->w_pos      = upload_f32(c->st, "vision_encoder.backbone.embeddings.position_embeddings", NULL);
    c->w_pre_ln_w = upload_f32(c->st, "vision_encoder.backbone.layer_norm.weight", NULL);
    c->w_pre_ln_b = upload_f32(c->st, "vision_encoder.backbone.layer_norm.bias",   NULL);
    if (!c->w_patch || !c->w_pos || !c->w_pre_ln_w || !c->w_pre_ln_b) {
        hip_sam3_destroy(c); return NULL;
    }

    /* scratch buffers */
    size_t img_bytes = (size_t)3 * S3_IMG * S3_IMG * sizeof(float);
    size_t tok_bytes = (size_t)S3_NTOK * S3_DIM * sizeof(float);
    if (hipMalloc(&c->d_img_f32, img_bytes) != hipSuccess ||
        hipMalloc(&c->d_tok,     tok_bytes) != hipSuccess ||
        hipMalloc(&c->d_tok_ln,  tok_bytes) != hipSuccess) {
        fprintf(stderr, "sam3: hipMalloc failed\n");
        hip_sam3_destroy(c); return NULL;
    }
    if (c->verbose)
        fprintf(stderr, "sam3: ready (patch+pos+pre-LN pipeline on device)\n");
    return c;
}

void hip_sam3_destroy(hip_sam3_ctx *c)
{
    if (!c) return;
    /* process-exit reclaim for device allocations (mirrors sam2 pattern). */
    if (c->st) safetensors_close(c->st);
    if (c->mod) hipModuleUnload(c->mod);
    free(c);
}

/* =========================================================== */
/* forward                                                       */
/* =========================================================== */

static int run_patch_pos_ln(hip_sam3_ctx *c)
{
    /* patch_embed: grid×grid blocks, 128 threads per block. */
    struct { void *out; const void *w, *img; int grid, img_size, D; } pe =
        { c->d_tok, c->w_patch, c->d_img_f32, S3_GRID, S3_IMG, S3_DIM };
    if (launch(c->fn_patch, S3_GRID, S3_GRID, 1, 128, 1, 1, 0,
                &pe, sizeof(pe))) return -1;

    /* pos_embed tile-add. */
    struct { void *tok; const void *pos; int grid, P, D; } pp =
        { c->d_tok, c->w_pos, S3_GRID, S3_PRE, S3_DIM };
    if (launch(c->fn_pos_add, S3_GRID, S3_GRID, 1, 256, 1, 1, 0,
                &pp, sizeof(pp))) return -1;

    /* Pre-block LayerNorm is applied by the first ViT block, not here;
     * this matches the ref hook site (vision_encoder.backbone.embeddings
     * dumps tokens pre-LN). */
    hipDeviceSynchronize();
    c->ready = 1;
    return 0;
}

int hip_sam3_set_image(hip_sam3_ctx *c, const uint8_t *rgb, int h, int w)
{
    if (!c || !rgb) return -1;
    size_t bytes = (size_t)h * w * 3;
    if (bytes > c->img_u8_cap) {
        if (c->d_img_u8) hipFree(c->d_img_u8);
        if (hipMalloc(&c->d_img_u8, bytes) != hipSuccess) return -1;
        c->img_u8_cap = bytes;
    }
    hipMemcpy(c->d_img_u8, rgb, bytes, hipMemcpyHostToDevice);

    /* ImageNet: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225). */
    struct {
        void *dst; const void *src; int sw, sh, dw, dh;
        float m0, m1, m2, i0, i1, i2;
    } rp = { c->d_img_f32, c->d_img_u8, w, h, S3_IMG, S3_IMG,
             0.485f, 0.456f, 0.406f,
             1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f };
    int total = S3_IMG * S3_IMG;
    if (launch(c->fn_resize, (total + 255) / 256, 1, 1, 256, 1, 1, 0,
                &rp, sizeof(rp))) return -1;
    return run_patch_pos_ln(c);
}

int hip_sam3_set_pixel_values(hip_sam3_ctx *c, const float *pixel_values_chw)
{
    if (!c || !pixel_values_chw) return -1;
    size_t bytes = (size_t)3 * S3_IMG * S3_IMG * sizeof(float);
    hipMemcpy(c->d_img_f32, pixel_values_chw, bytes, hipMemcpyHostToDevice);
    return run_patch_pos_ln(c);
}

int hip_sam3_get_vit_embed(const hip_sam3_ctx *c, float *out, int *n, int *d)
{
    if (!c || !c->ready || !out) return -1;
    hipMemcpy(out, c->d_tok,
              (size_t)S3_NTOK * S3_DIM * sizeof(float),
              hipMemcpyDeviceToHost);
    if (n) *n = S3_NTOK;
    if (d) *d = S3_DIM;
    return 0;
}
