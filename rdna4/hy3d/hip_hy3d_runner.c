/*
 * hip_hy3d_runner.c - HIP/ROCm Hunyuan3D-2.1 via HIPRTC-compiled kernels
 *
 * Pipeline: DINOv2 encoder -> DiT diffusion (flow matching) -> ShapeVAE -> MC mesh
 * Compiles with plain gcc (no hipcc). Uses rocew for dynamic HIP/HIPRTC loading.
 *
 * Port of cuda_hy3d_runner.c for AMD ROCm/HIP (RDNA4, gfx1200/gfx1201).
 * Key differences from CUDA version:
 *   - No MMA/tensor core kernels (RDNA4 has no MMA)
 *   - gemm_tiled_f16_f32 is the PRIMARY GEMM kernel
 *   - void * instead of CUdeviceptr (pointer arithmetic via char* casts)
 *   - hipMalloc/hipFree/hipMemcpy instead of cuMemAlloc/cuMemFree/cuMemcpyHtoD
 *
 * Architecture fixes applied:
 *   1. DiT U-Net skip connections (blocks 11-20)
 *   2. DiT MoE in last 6 blocks (15-20): 8 experts + shared expert, top-2 gating
 *   3. DiT prepends timestep token (seq_len 4096 -> 4097, stripped at final layer)
 *   4. TimestepEmbedder uses GELU (not SiLU)
 *   5. VAE include_pi=false (freqs = 2^i, not pi*2^i)
 *   6. VAE tensor name mapping (transformer.resblocks, c_qkv, c_proj, etc.)
 *   7. VAE geo_decoder tensor names (cross_attn_decoder prefix, c_q/c_kv, etc.)
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define MARCHING_CUBES_IMPLEMENTATION
#include "../../common/marching_cubes.h"
#include "hip_hy3d_runner.h"
#include "../rocew.h"
#include "../hip_kernels_common.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

/* Modular ops: kernel source strings + launch wrappers */
#include "hip_hy3d_kernels.h"
#include "hip_hy3d_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ======================================================================== */
/* Model constants                                                          */
/* ======================================================================== */

/* DINOv2-L config */
#define DINO_HIDDEN     1024
#define DINO_HEADS      16
#define DINO_HEAD_DIM   64
#define DINO_LAYERS     24
#define DINO_FFN        4096
#define DINO_PATCH      14
#define DINO_IMG_SIZE   518
#define DINO_NUM_PATCHES ((DINO_IMG_SIZE/DINO_PATCH)*(DINO_IMG_SIZE/DINO_PATCH)) /* 1369 */
#define DINO_SEQ_LEN    (DINO_NUM_PATCHES + 1)  /* 1370 */

/* DiT config */
#define DIT_INPUT_SIZE  4096
#define DIT_IN_CHANNELS 64
#define DIT_HIDDEN      2048
#define DIT_CONTEXT_DIM 1024
#define DIT_DEPTH       21
#define DIT_HEADS       16
#define DIT_HEAD_DIM    128
#define DIT_FFN         (DIT_HIDDEN * 4)    /* 8192 */
#define DIT_HALF_DEPTH  (DIT_DEPTH / 2)     /* 10 */
#define DIT_MOE_START   15
#define DIT_N_EXPERTS   8
#define DIT_MOE_TOP_K   2

/* ShapeVAE config */
#define VAE_NUM_LATENTS  4096
#define VAE_EMBED_DIM    64
#define VAE_WIDTH        1024
#define VAE_HEADS        16
#define VAE_HEAD_DIM     64
#define VAE_DEC_LAYERS   16
#define VAE_NUM_FREQS    8
#define VAE_FOURIER_DIM  51   /* 3*(2*8+1) */

/* ======================================================================== */
/* Internal structures                                                      */
/* ======================================================================== */

/* DINOv2 per-layer weights (on GPU, F16) */
typedef struct {
    void *ln1_w, *ln1_b;      /* LayerNorm 1 */
    void *q_w, *q_b;          /* Query projection */
    void *k_w, *k_b;          /* Key projection */
    void *v_w, *v_b;          /* Value projection */
    void *out_w, *out_b;      /* Output projection */
    void *ls1;                 /* LayerScale 1 */
    void *ln2_w, *ln2_b;      /* LayerNorm 2 */
    void *fc1_w, *fc1_b;      /* MLP FC1 */
    void *fc2_w, *fc2_b;      /* MLP FC2 */
    void *ls2;                 /* LayerScale 2 */
} dino_layer_gpu;

/* DiT per-block weights (on GPU, F16) */
typedef struct {
    /* Self-attention — fused QKV weight [3*dim, dim] for correct head interleaving */
    void *norm1_w, *norm1_b;
    void *sa_qkv_w;              /* [3*2048, 2048] = concat(to_q.w, to_k.w, to_v.w) */
    void *sa_out_w, *sa_out_b;
    void *sa_q_norm_w, *sa_k_norm_w;  /* RMSNorm weight only */
    /* Cross-attention — Q separate (from hidden), K/V fused (from context) */
    void *norm2_w, *norm2_b;
    void *ca_q_w;                /* [2048, 2048] */
    void *ca_kv_w;              /* [2*2048, 1024] = concat(to_k.w, to_v.w) */
    void *ca_out_w, *ca_out_b;
    void *ca_q_norm_w, *ca_k_norm_w;
    /* Norm3 + MLP/MoE */
    void *norm3_w, *norm3_b;
    /* Regular MLP (blocks 0-14) */
    void *mlp_fc1_w, *mlp_fc1_b;
    void *mlp_fc2_w, *mlp_fc2_b;
    /* MoE (blocks 15-20) */
    int use_moe;
    void *moe_gate_w;                                   /* [8, 2048] */
    void *moe_expert_fc1_w[DIT_N_EXPERTS];              /* [8192, 2048] each */
    void *moe_expert_fc1_b[DIT_N_EXPERTS];              /* [8192] each */
    void *moe_expert_fc2_w[DIT_N_EXPERTS];              /* [2048, 8192] each */
    void *moe_expert_fc2_b[DIT_N_EXPERTS];              /* [2048] each */
    void *moe_shared_fc1_w, *moe_shared_fc1_b;
    void *moe_shared_fc2_w, *moe_shared_fc2_b;
    /* Skip connection (blocks 11-20) */
    int use_skip;
    void *skip_linear_w, *skip_linear_b; /* [2048, 4096] */
    void *skip_norm_w, *skip_norm_b;
} dit_block_gpu;

/* ShapeVAE transformer block weights (on GPU, F32) */
typedef struct {
    void *ln1_w, *ln1_b;
    void *qkv_w;             /* Fused QKV [3*W, W] */
    void *proj_w, *proj_b;
    void *q_norm_w, *q_norm_b;
    void *k_norm_w, *k_norm_b;
    int use_qk_norm;
    void *ln2_w, *ln2_b;
    void *mlp_fc_w, *mlp_fc_b;
    void *mlp_proj_w, *mlp_proj_b;
} vae_block_gpu;

/* ShapeVAE geometry decoder weights */
typedef struct {
    void *query_proj_w, *query_proj_b;
    void *ln1_w, *ln1_b;     /* Query LN */
    void *ln2_w, *ln2_b;     /* Key/Value LN */
    void *c_q_w;            /* Cross-attn Q proj */
    void *c_kv_w;           /* Cross-attn KV proj [2*W, W] */
    void *c_proj_w, *c_proj_b;
    void *q_norm_w, *q_norm_b;
    void *k_norm_w, *k_norm_b;
    int use_qk_norm;
    void *ln3_w, *ln3_b;
    void *mlp_fc_w, *mlp_fc_b;
    void *mlp_proj_w, *mlp_proj_b;
    void *ln_post_w, *ln_post_b;
    void *output_w, *output_b;
} vae_geo_decoder_gpu;

struct hip_hy3d_runner {
    /* HIP context */
    hipDevice_t device;
    hipCtx_t ctx;
    hipStream_t stream;
    hipModule_t module;
    int sm_version;
    int verbose;

    /* Modular ops context (all compiled kernel functions) */
    hy3d_ops ops;

    /* DINOv2 weights */
    void *dino_patch_w, *dino_patch_b;
    void *dino_pos_emb;
    void *dino_cls_token;
    void *dino_final_ln_w, *dino_final_ln_b;
    dino_layer_gpu dino_layers[DINO_LAYERS];

    /* DiT weights */
    void *dit_x_emb_w, *dit_x_emb_b;
    void *dit_t_mlp0_w, *dit_t_mlp0_b;
    void *dit_t_mlp2_w, *dit_t_mlp2_b;
    void *dit_final_ln_w, *dit_final_ln_b;
    void *dit_final_linear_w, *dit_final_linear_b;
    dit_block_gpu dit_blocks[DIT_DEPTH];

    /* ShapeVAE weights */
    void *vae_post_kl_w, *vae_post_kl_b;
    vae_block_gpu vae_blocks[VAE_DEC_LAYERS];
    vae_geo_decoder_gpu vae_geo;
    void *vae_fourier_freqs;  /* [num_freqs] precomputed */

    /* Scratch buffers (GPU) */
    void *scratch[8];        /* General purpose scratch buffers */
    size_t scratch_size[8];

    /* Pre-computed cross-attention K,V for DiT (constant across diffusion steps) */
    void *dit_ca_K[DIT_DEPTH];   /* [DINO_SEQ_LEN, DIT_HIDDEN] */
    void *dit_ca_V[DIT_DEPTH];   /* [DINO_SEQ_LEN, DIT_HIDDEN] */
    int ca_kv_precomputed;

    /* Load status */
    int dino_loaded, dit_loaded, vae_loaded;
};

/* ======================================================================== */
/* Kernel compilation                                                       */
/* ======================================================================== */

static int hy3d_compile_kernels(hip_hy3d_runner *r) {
    /* Concatenate common + HY3D-specific kernel source */
    size_t len1 = strlen(hip_kernels_common_src);
    size_t len2 = strlen(hip_hy3d_specific_kernels);
    char *full_src = (char *)malloc(len1 + len2 + 1);
    memcpy(full_src, hip_kernels_common_src, len1);
    memcpy(full_src + len1, hip_hy3d_specific_kernels, len2);
    full_src[len1 + len2] = '\0';

    r->sm_version = hip_compile_kernels(&r->module, r->device, full_src,
                                        "hy3d_kernels", r->verbose, "HY3D");
    free(full_src);
    if (r->sm_version < 0) return -1;

    /* Load all kernel functions via the modular ops API */
    if (hy3d_ops_load(&r->ops, r->module, r->sm_version) != 0) {
        fprintf(stderr, "HY3D: ops_load failed\n");
        return -1;
    }

    return 0;
}

/* ======================================================================== */
/* Upload helpers                                                           */
/* ======================================================================== */

/* Upload safetensors tensor to GPU as F16 (converting F32->F16 if needed) */
static void *st_upload_f16(st_context *st, const char *name, int verbose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose) fprintf(stderr, "  [WARN] tensor '%s' not found\n", name);
        return NULL;
    }
    void *data = safetensors_data(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    if (strcmp(dtype, "F16") == 0 || strcmp(dtype, "BF16") == 0) {
        return hip_upload_raw(data, nbytes);
    } else if (strcmp(dtype, "F32") == 0) {
        /* Convert F32 to F16 on CPU then upload */
        size_t n = nbytes / sizeof(float);
        uint16_t *f16 = (uint16_t *)malloc(n * sizeof(uint16_t));
        const float *f32 = (const float *)data;
        for (size_t i = 0; i < n; i++) {
            f16[i] = hip_f32_to_f16(f32[i]);
        }
        void *d = hip_upload_raw(f16, n * sizeof(uint16_t));
        free(f16);
        return d;
    }
    if (verbose) fprintf(stderr, "  [WARN] tensor '%s' has unsupported dtype '%s'\n", name, dtype);
    return NULL;
}

/* Upload safetensors tensor to GPU as F32 (converting F16->F32 if needed) */
static void *st_upload_f32(st_context *st, const char *name, int verbose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose) fprintf(stderr, "  [WARN] tensor '%s' not found\n", name);
        return NULL;
    }
    void *data = safetensors_data(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    if (strcmp(dtype, "F32") == 0) {
        return hip_upload_raw(data, nbytes);
    } else if (strcmp(dtype, "F16") == 0) {
        size_t n = nbytes / sizeof(uint16_t);
        float *f32 = (float *)malloc(n * sizeof(float));
        const uint16_t *f16 = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            /* Simple F16 to F32 conversion */
            uint32_t sign = (f16[i] >> 15) & 0x1;
            uint32_t exp = (f16[i] >> 10) & 0x1f;
            uint32_t mant = f16[i] & 0x3ff;
            uint32_t f;
            if (exp == 0) {
                f = sign << 31;
            } else if (exp == 31) {
                f = (sign << 31) | 0x7f800000 | (mant << 13);
            } else {
                f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
            }
            memcpy(&f32[i], &f, sizeof(float));
        }
        void *d = hip_upload_raw(f32, n * sizeof(float));
        free(f32);
        return d;
    }
    return NULL;
}

/* Fuse 3 F16 weight tensors [dim, in_dim] into one [3*dim, in_dim] on GPU */
static void *st_fuse_3_f16(st_context *st,
                            const char *name_a, const char *name_b, const char *name_c,
                            int verbose) {
    int ia = safetensors_find(st, name_a);
    int ib = safetensors_find(st, name_b);
    int ic = safetensors_find(st, name_c);
    if (ia < 0 || ib < 0 || ic < 0) {
        if (verbose) fprintf(stderr, "  [WARN] fuse: missing tensor(s)\n");
        return NULL;
    }
    size_t na = safetensors_nbytes(st, ia);
    size_t nb = safetensors_nbytes(st, ib);
    size_t nc = safetensors_nbytes(st, ic);
    const char *da = safetensors_dtype(st, ia);

    /* All must be same dtype and size */
    if (na != nb || nb != nc) {
        if (verbose) fprintf(stderr, "  [WARN] fuse: size mismatch\n");
        return NULL;
    }

    /* Convert to F16 if needed, then concatenate */
    size_t total = na + nb + nc;
    if (strcmp(da, "F16") == 0) {
        uint8_t *buf = (uint8_t *)malloc(total);
        memcpy(buf, safetensors_data(st, ia), na);
        memcpy(buf + na, safetensors_data(st, ib), nb);
        memcpy(buf + na + nb, safetensors_data(st, ic), nc);
        void *d = hip_upload_raw(buf, total);
        free(buf);
        return d;
    } else if (strcmp(da, "F32") == 0) {
        /* Convert each F32 tensor to F16, then concatenate */
        size_t n_each = na / sizeof(float);
        size_t n_total = n_each * 3;
        uint16_t *f16 = (uint16_t *)malloc(n_total * sizeof(uint16_t));
        const float *fa = (const float *)safetensors_data(st, ia);
        const float *fb = (const float *)safetensors_data(st, ib);
        const float *fc = (const float *)safetensors_data(st, ic);
        for (size_t i = 0; i < n_each; i++) f16[i] = hip_f32_to_f16(fa[i]);
        for (size_t i = 0; i < n_each; i++) f16[n_each + i] = hip_f32_to_f16(fb[i]);
        for (size_t i = 0; i < n_each; i++) f16[2 * n_each + i] = hip_f32_to_f16(fc[i]);
        void *d = hip_upload_raw(f16, n_total * sizeof(uint16_t));
        free(f16);
        return d;
    }
    return NULL;
}

/* Allocate GPU buffer */
static void *gpu_alloc(size_t bytes) {
    void *d = NULL;
    if (hipMalloc(&d, bytes) != hipSuccess) {
        fprintf(stderr, "HY3D: GPU alloc failed for %zu bytes\n", bytes);
        return NULL;
    }
    return d;
}

/* Upload GEMM weight: F16 or F32 based on runner mode */
static void *st_upload_gemm_w(st_context *st, const char *name,
                               int use_f32, int verbose) {
    return use_f32 ? st_upload_f32(st, name, verbose)
                   : st_upload_f16(st, name, verbose);
}

/* Fuse 3 weight tensors for GEMM: F16 or F32 based on runner mode */
static void *st_fuse_3_gemm_w(st_context *st,
                                const char *a, const char *b, const char *c,
                                int use_f32, int verbose) {
    if (use_f32) {
        /* Concatenate as F32 */
        int ia = safetensors_find(st, a);
        int ib = safetensors_find(st, b);
        int ic = safetensors_find(st, c);
        if (ia < 0 || ib < 0 || ic < 0) return NULL;
        /* Get each as F32 on GPU */
        void *da = st_upload_f32(st, a, 0);
        void *db = st_upload_f32(st, b, 0);
        void *dc = st_upload_f32(st, c, 0);
        size_t na = safetensors_nbytes(st, ia);
        const char *dt = safetensors_dtype(st, ia);
        size_t n_elems = (strcmp(dt, "F16") == 0) ? na / 2 : na / 4;
        size_t bytes_each = n_elems * sizeof(float);
        /* Allocate combined GPU buffer and copy */
        void *d = gpu_alloc(bytes_each * 3);
        hipMemcpyAsync(d, da, bytes_each, hipMemcpyDeviceToDevice, 0);
        hipMemcpyAsync((char *)d + bytes_each, db, bytes_each, hipMemcpyDeviceToDevice, 0);
        hipMemcpyAsync((char *)d + 2 * bytes_each, dc, bytes_each, hipMemcpyDeviceToDevice, 0);
        hipFree(da); hipFree(db); hipFree(dc);
        return d;
    }
    return st_fuse_3_f16(st, a, b, c, verbose);
}

/* Fuse 2 weight tensors for GEMM: F16 or F32 */
static void *st_fuse_2_gemm_w(st_context *st,
                                const char *a, const char *b,
                                int use_f32, int verbose) {
    if (use_f32) {
        int ia = safetensors_find(st, a);
        int ib = safetensors_find(st, b);
        if (ia < 0 || ib < 0) return NULL;
        void *da = st_upload_f32(st, a, 0);
        void *db = st_upload_f32(st, b, 0);
        size_t na = safetensors_nbytes(st, ia);
        const char *dt = safetensors_dtype(st, ia);
        size_t n_elems = (strcmp(dt, "F16") == 0) ? na / 2 : na / 4;
        size_t bytes_each = n_elems * sizeof(float);
        void *d = gpu_alloc(bytes_each * 2);
        hipMemcpyAsync(d, da, bytes_each, hipMemcpyDeviceToDevice, 0);
        hipMemcpyAsync((char *)d + bytes_each, db, bytes_each, hipMemcpyDeviceToDevice, 0);
        hipFree(da); hipFree(db);
        return d;
    }
    /* F16 variant: concatenate raw F16 data */
    int ia = safetensors_find(st, a);
    int ib = safetensors_find(st, b);
    if (ia < 0 || ib < 0) return NULL;
    size_t na = safetensors_nbytes(st, ia);
    size_t nb = safetensors_nbytes(st, ib);
    uint8_t *buf = (uint8_t *)malloc(na + nb);
    memcpy(buf, safetensors_data(st, ia), na);
    memcpy(buf + na, safetensors_data(st, ib), nb);
    void *d = hip_upload_raw(buf, na + nb);
    free(buf);
    return d;
}

/* ======================================================================== */
/* Scratch buffer management                                                */
/* ======================================================================== */

static void ensure_scratch(hip_hy3d_runner *r, int idx, size_t bytes) {
    if (r->scratch_size[idx] < bytes) {
        if (r->scratch[idx]) hipFree(r->scratch[idx]);
        r->scratch[idx] = gpu_alloc(bytes);
        r->scratch_size[idx] = bytes;
    }
}

/* ======================================================================== */
/* Weight loading                                                           */
/* ======================================================================== */

static int load_dino_weights(hip_hy3d_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) {
        fprintf(stderr, "HY3D: cannot open DINOv2 weights: %s\n", path);
        return -1;
    }
    if (r->verbose) fprintf(stderr, "HY3D: loading DINOv2 from %s (%d tensors)\n",
                             path, st->n_tensors);

    /* Embeddings */
    r->dino_patch_w = st_upload_f32(st, "main_image_encoder.model.embeddings.patch_embeddings.projection.weight", r->verbose);
    r->dino_patch_b = st_upload_f32(st, "main_image_encoder.model.embeddings.patch_embeddings.projection.bias", r->verbose);
    r->dino_pos_emb = st_upload_f32(st, "main_image_encoder.model.embeddings.position_embeddings", r->verbose);
    r->dino_cls_token = st_upload_f32(st, "main_image_encoder.model.embeddings.cls_token", r->verbose);

    /* Encoder layers
     * GEMM weights -> F16 (gemm_tiled_f16_f32 expects half_raw)
     * LayerNorm weights, biases, LayerScale -> F32 (layernorm_f32 expects float) */
    for (int i = 0; i < DINO_LAYERS; i++) {
        char name[256];
        dino_layer_gpu *l = &r->dino_layers[i];

        /* F32 uploads: LayerNorm, LayerScale, biases used by LN or standalone */
        #define DINO_F32(field, suffix) \
            snprintf(name, sizeof(name), "main_image_encoder.model.encoder.layer.%d.%s", i, suffix); \
            l->field = st_upload_f32(st, name, r->verbose);
        /* GEMM weight uploads: F16 or F32 depending on mode */
        int f32g = r->ops.use_f32_gemm;
        #define DINO_GW(field, suffix) \
            snprintf(name, sizeof(name), "main_image_encoder.model.encoder.layer.%d.%s", i, suffix); \
            l->field = st_upload_gemm_w(st, name, f32g, r->verbose);

        DINO_F32(ln1_w, "norm1.weight");
        DINO_F32(ln1_b, "norm1.bias");
        DINO_GW(q_w,   "attention.attention.query.weight");
        DINO_F32(q_b,   "attention.attention.query.bias");
        DINO_GW(k_w,   "attention.attention.key.weight");
        DINO_F32(k_b,   "attention.attention.key.bias");
        DINO_GW(v_w,   "attention.attention.value.weight");
        DINO_F32(v_b,   "attention.attention.value.bias");
        DINO_GW(out_w, "attention.output.dense.weight");
        DINO_F32(out_b, "attention.output.dense.bias");
        DINO_F32(ls1,   "layer_scale1.lambda1");
        DINO_F32(ln2_w, "norm2.weight");
        DINO_F32(ln2_b, "norm2.bias");
        DINO_GW(fc1_w, "mlp.fc1.weight");
        DINO_F32(fc1_b, "mlp.fc1.bias");
        DINO_GW(fc2_w, "mlp.fc2.weight");
        DINO_F32(fc2_b, "mlp.fc2.bias");
        DINO_F32(ls2,   "layer_scale2.lambda1");
        #undef DINO_F32
        #undef DINO_GW
    }

    /* Final LN (F32 -- consumed by layernorm_f32) */
    r->dino_final_ln_w = st_upload_f32(st, "main_image_encoder.model.layernorm.weight", r->verbose);
    r->dino_final_ln_b = st_upload_f32(st, "main_image_encoder.model.layernorm.bias", r->verbose);
    if (!r->dino_final_ln_w) {
        r->dino_final_ln_w = st_upload_f32(st, "main_image_encoder.model.norm.weight", r->verbose);
        r->dino_final_ln_b = st_upload_f32(st, "main_image_encoder.model.norm.bias", r->verbose);
    }

    safetensors_close(st);
    r->dino_loaded = 1;
    if (r->verbose) fprintf(stderr, "HY3D: DINOv2 weights loaded\n");
    return 0;
}

static int load_dit_weights(hip_hy3d_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) {
        fprintf(stderr, "HY3D: cannot open DiT weights: %s\n", path);
        return -1;
    }
    if (r->verbose) fprintf(stderr, "HY3D: loading DiT from %s (%d tensors)\n",
                             path, st->n_tensors);

    int f32g = r->ops.use_f32_gemm;

    /* x_embedder */
    r->dit_x_emb_w = st_upload_gemm_w(st, "x_embedder.weight", f32g, r->verbose);
    r->dit_x_emb_b = st_upload_f32(st, "x_embedder.bias", r->verbose);

    /* t_embedder */
    r->dit_t_mlp0_w = st_upload_gemm_w(st, "t_embedder.mlp.0.weight", f32g, r->verbose);
    r->dit_t_mlp0_b = st_upload_f32(st, "t_embedder.mlp.0.bias", r->verbose);
    r->dit_t_mlp2_w = st_upload_gemm_w(st, "t_embedder.mlp.2.weight", f32g, r->verbose);
    r->dit_t_mlp2_b = st_upload_f32(st, "t_embedder.mlp.2.bias", r->verbose);

    /* Blocks: GEMM weights F16 or F32 (switchable), LN/RMSNorm always F32 */
    for (int i = 0; i < DIT_DEPTH; i++) {
        char name[256];
        dit_block_gpu *b = &r->dit_blocks[i];
        #define DIT_GW(field, suffix) \
            snprintf(name, sizeof(name), "blocks.%d.%s", i, suffix); \
            b->field = st_upload_gemm_w(st, name, f32g, r->verbose);
        #define DIT_F32(field, suffix) \
            snprintf(name, sizeof(name), "blocks.%d.%s", i, suffix); \
            b->field = st_upload_f32(st, name, r->verbose);

        /* LayerNorm -> F32 */
        DIT_F32(norm1_w,     "norm1.weight");
        DIT_F32(norm1_b,     "norm1.bias");
        /* Self-attn: fuse Q/K/V weights into [3*dim, dim] for correct head interleaving */
        {
            char nq[256], nk[256], nv[256];
            snprintf(nq, sizeof(nq), "blocks.%d.attn1.to_q.weight", i);
            snprintf(nk, sizeof(nk), "blocks.%d.attn1.to_k.weight", i);
            snprintf(nv, sizeof(nv), "blocks.%d.attn1.to_v.weight", i);
            b->sa_qkv_w = st_fuse_3_gemm_w(st, nq, nk, nv, f32g, r->verbose);
        }
        DIT_GW(sa_out_w,    "attn1.out_proj.weight");
        DIT_F32(sa_out_b,    "attn1.out_proj.bias");
        /* RMSNorm weights -> F32 */
        DIT_F32(sa_q_norm_w, "attn1.q_norm.weight");
        DIT_F32(sa_k_norm_w, "attn1.k_norm.weight");

        DIT_F32(norm2_w,     "norm2.weight");
        DIT_F32(norm2_b,     "norm2.bias");
        DIT_GW(ca_q_w,      "attn2.to_q.weight");
        /* Fuse K/V weights for correct head interleaving */
        {
            char nk[256], nv[256];
            snprintf(nk, sizeof(nk), "blocks.%d.attn2.to_k.weight", i);
            snprintf(nv, sizeof(nv), "blocks.%d.attn2.to_v.weight", i);
            b->ca_kv_w = st_fuse_2_gemm_w(st, nk, nv, f32g, r->verbose);
        }
        DIT_GW(ca_out_w,    "attn2.out_proj.weight");
        DIT_F32(ca_out_b,    "attn2.out_proj.bias");
        DIT_F32(ca_q_norm_w, "attn2.q_norm.weight");
        DIT_F32(ca_k_norm_w, "attn2.k_norm.weight");

        DIT_F32(norm3_w,     "norm3.weight");
        DIT_F32(norm3_b,     "norm3.bias");

        /* Skip connection (blocks 11-20) */
        b->use_skip = (i > DIT_HALF_DEPTH) ? 1 : 0;
        if (b->use_skip) {
            DIT_GW(skip_linear_w, "skip_linear.weight");
            DIT_F32(skip_linear_b, "skip_linear.bias");
            DIT_F32(skip_norm_w,   "skip_norm.weight");
            DIT_F32(skip_norm_b,   "skip_norm.bias");
        }

        /* MoE vs regular MLP */
        b->use_moe = (i >= DIT_MOE_START) ? 1 : 0;
        if (b->use_moe) {
            DIT_GW(moe_gate_w, "moe.gate.weight");
            for (int e = 0; e < DIT_N_EXPERTS; e++) {
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.net.0.proj.weight", i, e);
                b->moe_expert_fc1_w[e] = st_upload_gemm_w(st, name, f32g, r->verbose);
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.net.0.proj.bias", i, e);
                b->moe_expert_fc1_b[e] = st_upload_f32(st, name, r->verbose);
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.net.2.weight", i, e);
                b->moe_expert_fc2_w[e] = st_upload_gemm_w(st, name, f32g, r->verbose);
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.net.2.bias", i, e);
                b->moe_expert_fc2_b[e] = st_upload_f32(st, name, r->verbose);
            }
            DIT_GW(moe_shared_fc1_w, "moe.shared_experts.net.0.proj.weight");
            DIT_F32(moe_shared_fc1_b, "moe.shared_experts.net.0.proj.bias");
            DIT_GW(moe_shared_fc2_w, "moe.shared_experts.net.2.weight");
            DIT_F32(moe_shared_fc2_b, "moe.shared_experts.net.2.bias");
        } else {
            DIT_GW(mlp_fc1_w,   "mlp.fc1.weight");
            DIT_F32(mlp_fc1_b,   "mlp.fc1.bias");
            DIT_GW(mlp_fc2_w,   "mlp.fc2.weight");
            DIT_F32(mlp_fc2_b,   "mlp.fc2.bias");
        }
        #undef DIT_GW
        #undef DIT_F32
    }

    /* Final layer: LN -> F32, linear -> F16 */
    r->dit_final_ln_w = st_upload_f32(st, "final_layer.norm_final.weight", r->verbose);
    r->dit_final_ln_b = st_upload_f32(st, "final_layer.norm_final.bias", r->verbose);
    r->dit_final_linear_w = st_upload_gemm_w(st, "final_layer.linear.weight", f32g, r->verbose);
    r->dit_final_linear_b = st_upload_f32(st, "final_layer.linear.bias", r->verbose);

    safetensors_close(st);
    r->dit_loaded = 1;
    if (r->verbose) fprintf(stderr, "HY3D: DiT weights loaded\n");
    return 0;
}

/* Fix 6 & 7: VAE tensor name mapping */
static int load_vae_weights(hip_hy3d_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) {
        fprintf(stderr, "HY3D: cannot open VAE weights: %s\n", path);
        return -1;
    }
    if (r->verbose) fprintf(stderr, "HY3D: loading ShapeVAE from %s (%d tensors)\n",
                             path, st->n_tensors);

    /* Post-KL projection (GEMM: weight F16, bias F32) */
    r->vae_post_kl_w = st_upload_gemm_w(st, "post_kl.weight", r->ops.use_f32_gemm, r->verbose);
    r->vae_post_kl_b = st_upload_f32(st, "post_kl.bias", r->verbose);

    /* Fix 6: Transformer decoder blocks use transformer.resblocks.N prefix
     * GEMM weights -> F16/F32 (switchable), biases/LN/QKnorm -> F32 */
    int f32g = r->ops.use_f32_gemm;
    for (int i = 0; i < VAE_DEC_LAYERS; i++) {
        char name[256];
        vae_block_gpu *b = &r->vae_blocks[i];
        #define VAE_F16(field, suffix) \
            snprintf(name, sizeof(name), "transformer.resblocks.%d.%s", i, suffix); \
            b->field = st_upload_gemm_w(st, name, f32g, r->verbose);
        #define VAE_F32(field, suffix) \
            snprintf(name, sizeof(name), "transformer.resblocks.%d.%s", i, suffix); \
            b->field = st_upload_f32(st, name, r->verbose);

        VAE_F32(ln1_w,        "ln_1.weight");
        VAE_F32(ln1_b,        "ln_1.bias");
        VAE_F16(qkv_w,        "attn.c_qkv.weight");
        VAE_F16(proj_w,       "attn.c_proj.weight");
        VAE_F32(proj_b,       "attn.c_proj.bias");
        VAE_F32(q_norm_w,     "attn.attention.q_norm.weight");
        VAE_F32(q_norm_b,     "attn.attention.q_norm.bias");
        VAE_F32(k_norm_w,     "attn.attention.k_norm.weight");
        VAE_F32(k_norm_b,     "attn.attention.k_norm.bias");
        VAE_F32(ln2_w,        "ln_2.weight");
        VAE_F32(ln2_b,        "ln_2.bias");
        VAE_F16(mlp_fc_w,     "mlp.c_fc.weight");
        VAE_F32(mlp_fc_b,     "mlp.c_fc.bias");
        VAE_F16(mlp_proj_w,   "mlp.c_proj.weight");
        VAE_F32(mlp_proj_b,   "mlp.c_proj.bias");
        #undef VAE_F16
        #undef VAE_F32

        b->use_qk_norm = (b->q_norm_w != NULL);
    }

    /* Fix 7: Geometry decoder
     * GEMM weights -> F16, biases -> F32; LN/QKnorm -> F32 */
    vae_geo_decoder_gpu *g = &r->vae_geo;
    #define GEO_F16(field, suffix) g->field = st_upload_gemm_w(st, suffix, f32g, r->verbose);
    #define GEO_F32(field, suffix) g->field = st_upload_f32(st, suffix, r->verbose);

    GEO_F16(query_proj_w, "geo_decoder.query_proj.weight");
    GEO_F32(query_proj_b, "geo_decoder.query_proj.bias");
    GEO_F32(ln1_w,        "geo_decoder.cross_attn_decoder.ln_1.weight");
    GEO_F32(ln1_b,        "geo_decoder.cross_attn_decoder.ln_1.bias");
    GEO_F32(ln2_w,        "geo_decoder.cross_attn_decoder.ln_2.weight");
    GEO_F32(ln2_b,        "geo_decoder.cross_attn_decoder.ln_2.bias");
    GEO_F16(c_q_w,        "geo_decoder.cross_attn_decoder.attn.c_q.weight");
    GEO_F16(c_kv_w,       "geo_decoder.cross_attn_decoder.attn.c_kv.weight");
    GEO_F16(c_proj_w,     "geo_decoder.cross_attn_decoder.attn.c_proj.weight");
    GEO_F32(c_proj_b,     "geo_decoder.cross_attn_decoder.attn.c_proj.bias");
    GEO_F32(q_norm_w,     "geo_decoder.cross_attn_decoder.attn.attention.q_norm.weight");
    GEO_F32(q_norm_b,     "geo_decoder.cross_attn_decoder.attn.attention.q_norm.bias");
    GEO_F32(k_norm_w,     "geo_decoder.cross_attn_decoder.attn.attention.k_norm.weight");
    GEO_F32(k_norm_b,     "geo_decoder.cross_attn_decoder.attn.attention.k_norm.bias");
    GEO_F32(ln3_w,        "geo_decoder.cross_attn_decoder.ln_3.weight");
    GEO_F32(ln3_b,        "geo_decoder.cross_attn_decoder.ln_3.bias");
    GEO_F16(mlp_fc_w,     "geo_decoder.cross_attn_decoder.mlp.c_fc.weight");
    GEO_F32(mlp_fc_b,     "geo_decoder.cross_attn_decoder.mlp.c_fc.bias");
    GEO_F16(mlp_proj_w,   "geo_decoder.cross_attn_decoder.mlp.c_proj.weight");
    GEO_F32(mlp_proj_b,   "geo_decoder.cross_attn_decoder.mlp.c_proj.bias");
    GEO_F32(ln_post_w,    "geo_decoder.ln_post.weight");
    GEO_F32(ln_post_b,    "geo_decoder.ln_post.bias");
    GEO_F16(output_w,     "geo_decoder.output_proj.weight");
    GEO_F32(output_b,     "geo_decoder.output_proj.bias");
    #undef GEO_F16
    #undef GEO_F32

    g->use_qk_norm = (g->q_norm_w != NULL);

    /* Fix 5: Pre-compute Fourier frequencies WITHOUT pi multiplier */
    float freqs[VAE_NUM_FREQS];
    for (int i = 0; i < VAE_NUM_FREQS; i++) {
        freqs[i] = powf(2.0f, (float)i);  /* 2^i, no pi factor */
    }
    r->vae_fourier_freqs = hip_upload_raw(freqs, sizeof(freqs));

    safetensors_close(st);
    r->vae_loaded = 1;
    if (r->verbose) fprintf(stderr, "HY3D: ShapeVAE weights loaded\n");
    return 0;
}

/* ======================================================================== */
/* Pipeline stages                                                          */
/* ======================================================================== */

/* Stage 1: DINOv2 forward pass
 * Input:  d_image [3, 518, 518] F32 (normalized)
 * Output: d_out [1370, 1024] F32 */
static void run_dinov2(hip_hy3d_runner *r, void *d_image, void *d_out) {
    hy3d_ops *ops = &r->ops;
    hipStream_t stream = r->stream;
    const int seq = DINO_SEQ_LEN;
    const int dim = DINO_HIDDEN;
    const int heads = DINO_HEADS;
    const int hd = DINO_HEAD_DIM;
    const int ffn = DINO_FFN;
    const int ps = DINO_PATCH;
    const int gw = DINO_IMG_SIZE / ps;  /* 37 */

    /* Scratch: 0=hidden[seq*dim], 1=qkv[3*seq*dim], 2=attn_out[seq*dim],
     *          3=mlp[seq*ffn], 4=normed[seq*dim] */
    ensure_scratch(r, 0, (size_t)seq * dim * sizeof(float));
    ensure_scratch(r, 1, (size_t)3 * seq * dim * sizeof(float));
    ensure_scratch(r, 2, (size_t)seq * dim * sizeof(float));
    ensure_scratch(r, 3, (size_t)seq * ffn * sizeof(float));
    ensure_scratch(r, 4, (size_t)seq * dim * sizeof(float));

    void *d_hidden = r->scratch[0];
    void *d_qkv    = r->scratch[1];
    void *d_attn   = r->scratch[2];
    void *d_mlp    = r->scratch[3];
    void *d_normed = r->scratch[4];

    /* 1. Patch embedding: conv2d -> [num_patches, dim] */
    {
        int gw2 = gw;
        int dim2 = dim, ps2 = ps, img_w = DINO_IMG_SIZE;
        void *pw = r->dino_patch_w, *pb = r->dino_patch_b;
        void *args[] = {&d_hidden, &d_image, &pw, &pb, &gw2, &dim2, &ps2, &img_w};
        hipModuleLaunchKernel(ops->patch_embed,
                       (unsigned)(gw * gw), 1, 1,
                       256, 1, 1, 0, stream, args, NULL);
    }

    /* 2. CLS token + position embeddings */
    {
        int n_tok = seq, d2 = dim;
        void *cls = r->dino_cls_token, *pos = r->dino_pos_emb;
        void *args[] = {&d_hidden, &cls, &pos, &n_tok, &d2};
        hipModuleLaunchKernel(ops->cls_pos_embed,
                       (unsigned)((seq*dim+255)/256), 1, 1,
                       256, 1, 1, 0, stream, args, NULL);
    }

    /* 3. Encoder layers */
    for (int li = 0; li < DINO_LAYERS; li++) {
        dino_layer_gpu *l = &r->dino_layers[li];

        /* LN1 -> Q,K,V -> Attention -> LayerScale + residual */
        op_layernorm(ops, stream, d_normed, d_hidden, l->ln1_w, l->ln1_b, seq, dim);

        /* Q, K, V projections */
        void *d_Q = d_qkv;
        void *d_K = (char *)d_qkv + (size_t)seq * dim * sizeof(float);
        void *d_V = (char *)d_qkv + (size_t)2 * seq * dim * sizeof(float);
        op_gemm(ops, stream, d_Q, l->q_w, d_normed, l->q_b, dim, dim, seq);
        op_gemm(ops, stream, d_K, l->k_w, d_normed, l->k_b, dim, dim, seq);
        op_gemm(ops, stream, d_V, l->v_w, d_normed, l->v_b, dim, dim, seq);

        /* Self-attention */
        op_self_attn(ops, stream, d_attn, d_Q, d_K, d_V, seq, dim, heads, hd);

        /* Output projection */
        op_gemm(ops, stream, d_normed, l->out_w, d_attn, l->out_b, dim, dim, seq);

        /* LayerScale 1 + residual */
        if (l->ls1)
            op_layerscale_add(ops, stream, d_hidden, d_normed, l->ls1, seq * dim, dim);
        else
            op_add(ops, stream, d_hidden, d_normed, seq * dim);

        /* LN2 -> MLP -> LayerScale + residual */
        op_layernorm(ops, stream, d_normed, d_hidden, l->ln2_w, l->ln2_b, seq, dim);
        op_gemm(ops, stream, d_mlp, l->fc1_w, d_normed, l->fc1_b, ffn, dim, seq);
        op_gelu(ops, stream, d_mlp, seq * ffn);
        op_gemm(ops, stream, d_normed, l->fc2_w, d_mlp, l->fc2_b, dim, ffn, seq);

        if (l->ls2)
            op_layerscale_add(ops, stream, d_hidden, d_normed, l->ls2, seq * dim, dim);
        else
            op_add(ops, stream, d_hidden, d_normed, seq * dim);
    }

    /* 4. Final LN */
    if (r->dino_final_ln_w) {
        op_layernorm(ops, stream, d_out, d_hidden, r->dino_final_ln_w, r->dino_final_ln_b, seq, dim);
    } else {
        hipMemcpyAsync(d_out, d_hidden, (size_t)seq * dim * sizeof(float), hipMemcpyDeviceToDevice, stream);
    }
}

/* precompute_dit_ca_kv removed -- K/V computed per-block inside run_dit_forward */

/*
 * MoE forward: simplified "run all experts on all tokens" approach.
 */
static void run_dit_moe(hip_hy3d_runner *r, dit_block_gpu *blk,
                         void *d_input, void *d_output,
                         int N_tok, void *d_moe_scratch) {
    hy3d_ops *ops = &r->ops;
    hipStream_t stream = r->stream;
    const int H_dim = DIT_HIDDEN;
    const int ffn = DIT_FFN;

    /* Scratch layout within d_moe_scratch */
    size_t off = 0;
    void *d_gate   = (char *)d_moe_scratch + off; off += (size_t)N_tok * DIT_N_EXPERTS * sizeof(float);
    void *d_exp_h  = (char *)d_moe_scratch + off; off += (size_t)N_tok * ffn * sizeof(float);
    void *d_exp_o  = (char *)d_moe_scratch + off; /* [N_tok * H_dim] reused */

    /* Step 1: Compute gate logits: [N_tok, H_dim] @ [8, H_dim]^T -> [N_tok, 8] */
    op_gemm(ops, stream, d_gate, blk->moe_gate_w, d_input, NULL,
            DIT_N_EXPERTS, H_dim, N_tok);

    /* Step 2: Download gate logits, compute softmax + top-2 on CPU */
    float *gate_cpu = (float *)malloc((size_t)N_tok * DIT_N_EXPERTS * sizeof(float));
    hipStreamSynchronize(stream);
    hipMemcpy(gate_cpu, d_gate, (size_t)N_tok * DIT_N_EXPERTS * sizeof(float), hipMemcpyDeviceToHost);

    /* Softmax over experts per token, then top-2 masking */
    float *gate_weights = (float *)calloc((size_t)N_tok * DIT_N_EXPERTS, sizeof(float));
    for (int t = 0; t < N_tok; t++) {
        float *row = gate_cpu + t * DIT_N_EXPERTS;
        /* Softmax */
        float mx = row[0];
        for (int e = 1; e < DIT_N_EXPERTS; e++)
            if (row[e] > mx) mx = row[e];
        float sum = 0.0f;
        float softmax_vals[DIT_N_EXPERTS];
        for (int e = 0; e < DIT_N_EXPERTS; e++) {
            softmax_vals[e] = expf(row[e] - mx);
            sum += softmax_vals[e];
        }
        float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
        for (int e = 0; e < DIT_N_EXPERTS; e++)
            softmax_vals[e] *= inv;

        /* Top-2 selection */
        int top_idx[DIT_MOE_TOP_K];
        float top_val[DIT_MOE_TOP_K];
        for (int k = 0; k < DIT_MOE_TOP_K; k++) {
            int best = -1;
            float best_v = -1e30f;
            for (int e = 0; e < DIT_N_EXPERTS; e++) {
                int used = 0;
                for (int kk = 0; kk < k; kk++)
                    if (top_idx[kk] == e) { used = 1; break; }
                if (!used && softmax_vals[e] > best_v) {
                    best_v = softmax_vals[e];
                    best = e;
                }
            }
            top_idx[k] = best;
            top_val[k] = best_v;
        }

        /* Renormalize top-k weights so they sum to 1 */
        float top_sum = 0.0f;
        for (int k = 0; k < DIT_MOE_TOP_K; k++) top_sum += top_val[k];
        float top_inv = (top_sum > 0.0f) ? 1.0f / top_sum : 0.0f;
        for (int k = 0; k < DIT_MOE_TOP_K; k++) {
            if (top_idx[k] >= 0)
                gate_weights[t * DIT_N_EXPERTS + top_idx[k]] = top_val[k] * top_inv;
        }
    }
    free(gate_cpu);

    /* Step 3: Zero out accumulator (d_output) */
    hipMemsetAsync(d_output, 0, (size_t)N_tok * H_dim * sizeof(float), stream);

    /* Step 4: For each expert, compute output and weighted-add */
    float *expert_scale_cpu = (float *)malloc((size_t)N_tok * sizeof(float));
    for (int e = 0; e < DIT_N_EXPERTS; e++) {
        /* Check if any token uses this expert */
        int any_nonzero = 0;
        for (int t = 0; t < N_tok; t++) {
            expert_scale_cpu[t] = gate_weights[t * DIT_N_EXPERTS + e];
            if (expert_scale_cpu[t] > 0.0f) any_nonzero = 1;
        }
        if (!any_nonzero) continue;

        /* expert_out = GELU(x @ fc1_w_e.T + fc1_b_e) @ fc2_w_e.T + fc2_b_e */
        op_gemm(ops, stream, d_exp_h, blk->moe_expert_fc1_w[e], d_input,
                blk->moe_expert_fc1_b[e], ffn, H_dim, N_tok);
        op_gelu(ops, stream, d_exp_h, N_tok * ffn);
        op_gemm(ops, stream, d_exp_o, blk->moe_expert_fc2_w[e], d_exp_h,
                blk->moe_expert_fc2_b[e], H_dim, ffn, N_tok);

        /* Scale-add: d_output[t,:] += scale[t] * d_exp_o[t,:]  (on CPU) */
        hipStreamSynchronize(stream);
        float *exp_out_cpu = (float *)malloc((size_t)N_tok * H_dim * sizeof(float));
        float *accum_cpu = (float *)malloc((size_t)N_tok * H_dim * sizeof(float));
        hipMemcpy(exp_out_cpu, d_exp_o, (size_t)N_tok * H_dim * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(accum_cpu, d_output, (size_t)N_tok * H_dim * sizeof(float), hipMemcpyDeviceToHost);

        for (int t = 0; t < N_tok; t++) {
            float w = expert_scale_cpu[t];
            if (w == 0.0f) continue;
            for (int j = 0; j < H_dim; j++)
                accum_cpu[t * H_dim + j] += w * exp_out_cpu[t * H_dim + j];
        }

        hipMemcpyAsync(d_output, accum_cpu,
                          (size_t)N_tok * H_dim * sizeof(float), hipMemcpyHostToDevice, stream);
        free(exp_out_cpu);
        free(accum_cpu);
    }
    free(expert_scale_cpu);
    free(gate_weights);

    /* Step 5: Add shared expert output */
    op_gemm(ops, stream, d_exp_h, blk->moe_shared_fc1_w, d_input,
            blk->moe_shared_fc1_b, ffn, H_dim, N_tok);
    op_gelu(ops, stream, d_exp_h, N_tok * ffn);
    op_gemm(ops, stream, d_exp_o, blk->moe_shared_fc2_w, d_exp_h,
            blk->moe_shared_fc2_b, H_dim, ffn, N_tok);
    op_add(ops, stream, d_output, d_exp_o, N_tok * H_dim);
}

/* Stage 2: Single DiT forward pass */
static void run_dit_forward(hip_hy3d_runner *r, void *d_latents,
                             float timestep, void *d_context,
                             void *d_output) {
    hy3d_ops *ops = &r->ops;
    hipStream_t stream = r->stream;
    const int N = DIT_INPUT_SIZE;       /* 4096 */
    const int C = DIT_IN_CHANNELS;      /* 64 */
    const int H_dim = DIT_HIDDEN;       /* 2048 */
    const int heads = DIT_HEADS;        /* 16 */
    const int hd = DIT_HEAD_DIM;        /* 128 */
    const int ffn = DIT_FFN;            /* 8192 */
    const int ctx_len = DINO_SEQ_LEN;   /* 1370 */
    const int N1 = N + 1;              /* 4097 (with prepended timestep token) */

    size_t mlp_sz = (size_t)N1 * ffn * sizeof(float);
    size_t moe_scratch_sz = (size_t)N1 * (DIT_N_EXPERTS + ffn + H_dim) * sizeof(float);
    size_t mlp_moe_sz = mlp_sz > moe_scratch_sz ? mlp_sz : moe_scratch_sz;
    size_t cat_buf_sz = (size_t)N1 * 2 * H_dim * sizeof(float);
    size_t ca_kv_sz = (size_t)ctx_len * H_dim * sizeof(float);

    size_t qkv_sz = (size_t)3 * N1 * H_dim * sizeof(float);
    size_t shared1_sz = qkv_sz > mlp_moe_sz ? qkv_sz : mlp_moe_sz;
    size_t buf3_sz = (size_t)(H_dim + ffn) * sizeof(float)
                   + (size_t)N1 * H_dim * sizeof(float)
                   + cat_buf_sz + 2 * ca_kv_sz
                   + (size_t)N1 * H_dim * sizeof(float);
    ensure_scratch(r, 0, (size_t)N1 * H_dim * sizeof(float));
    ensure_scratch(r, 1, shared1_sz);
    ensure_scratch(r, 2, (size_t)2 * N1 * H_dim * sizeof(float));
    ensure_scratch(r, 3, buf3_sz);

    void *d_hidden  = r->scratch[0];
    void *d_qkv     = r->scratch[1];
    void *d_mlp     = r->scratch[1];  /* shares with d_qkv */
    void *d_attn    = r->scratch[2];
    void *d_normed  = (char *)r->scratch[2] + (size_t)N1 * H_dim * sizeof(float);
    void *d_temb    = r->scratch[3];
    void *d_tmlp    = (char *)d_temb + (size_t)H_dim * sizeof(float);
    void *d_cross_Q = (char *)d_tmlp + (size_t)ffn * sizeof(float);
    void *d_cat_buf = (char *)d_cross_Q + (size_t)N1 * H_dim * sizeof(float);
    void *d_ca_K    = (char *)d_cat_buf + cat_buf_sz;
    void *d_ca_V    = (char *)d_ca_K + ca_kv_sz;
    void *d_moe_scratch = r->scratch[1]; /* shared with d_qkv/d_mlp */

    /* Skip stack: stored in CPU RAM */
    size_t skip_entry_sz = (size_t)N1 * H_dim * sizeof(float);
    float *skip_stack_cpu = (float *)malloc((size_t)(DIT_HALF_DEPTH + 1) * skip_entry_sz);
    void *d_skip_tmp = d_attn;

    /* 1. Embed latents: [N, C] -> [N, H_dim] */
    op_gemm(ops, stream, d_normed, r->dit_x_emb_w, d_latents, r->dit_x_emb_b,
            H_dim, C, N);

    /* 2. Fix 4: Timestep embedding with GELU (not SiLU) */
    op_timestep_embed(ops, stream, d_temb, timestep, H_dim);
    op_gemm(ops, stream, d_tmlp, r->dit_t_mlp0_w, d_temb, r->dit_t_mlp0_b,
            ffn, H_dim, 1);
    op_gelu(ops, stream, d_tmlp, ffn);
    op_gemm(ops, stream, d_temb, r->dit_t_mlp2_w, d_tmlp, r->dit_t_mlp2_b,
            H_dim, ffn, 1);

    /* 3. Fix 3: Prepend timestep token to sequence */
    op_concat_first(ops, stream, d_hidden, d_temb, d_normed, N, H_dim);

    if (r->verbose > 1) {
        hipStreamSynchronize(stream);
        float *hcpu = (float *)malloc((size_t)N1 * H_dim * sizeof(float));
        hipMemcpy(hcpu, d_hidden, (size_t)N1 * H_dim * sizeof(float), hipMemcpyDeviceToHost);
        float mn = hcpu[0], mx = hcpu[0], sm = 0;
        for (int j = 0; j < N1 * H_dim; j++) {
            if (hcpu[j] < mn) mn = hcpu[j];
            if (hcpu[j] > mx) mx = hcpu[j];
            sm += hcpu[j];
        }
        float mean = sm / (float)(N1 * H_dim);
        float var = 0;
        for (int j = 0; j < N1 * H_dim; j++) { float d = hcpu[j] - mean; var += d*d; }
        fprintf(stderr, "  after_embed: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
                mean, sqrtf(var / (float)(N1 * H_dim)), mn, mx);
        free(hcpu);
    }

    /* Skip value stack pointer */
    int skip_sp = 0;

    /* 4. Transformer blocks */
    for (int bi = 0; bi < DIT_DEPTH; bi++) {
        dit_block_gpu *blk = &r->dit_blocks[bi];

        /* Fix 1: Skip connection (blocks 11-20) */
        if (blk->use_skip && skip_sp > 0) {
            skip_sp--;
            float *skip_cpu = skip_stack_cpu + (size_t)skip_sp * N1 * H_dim;

            hipMemcpyAsync(d_skip_tmp, skip_cpu, skip_entry_sz, hipMemcpyHostToDevice, stream);
            hipStreamSynchronize(stream);

            op_concat_last_dim(ops, stream, d_cat_buf, d_skip_tmp, d_hidden, N1, H_dim);

            op_gemm(ops, stream, d_hidden, blk->skip_linear_w, d_cat_buf,
                    blk->skip_linear_b, H_dim, 2 * H_dim, N1);

            op_layernorm(ops, stream, d_hidden, d_hidden,
                         blk->skip_norm_w, blk->skip_norm_b, N1, H_dim);
        }

        /* Save hidden state for skip connection (blocks 0..DIT_HALF_DEPTH) */
        if (bi <= DIT_HALF_DEPTH) {
            float *skip_cpu = skip_stack_cpu + (size_t)skip_sp * N1 * H_dim;
            hipStreamSynchronize(stream);
            hipMemcpy(skip_cpu, d_hidden, skip_entry_sz, hipMemcpyDeviceToHost);
            skip_sp++;
        }

        /* === Self-attention === */
        op_layernorm(ops, stream, d_normed, d_hidden, blk->norm1_w, blk->norm1_b, N1, H_dim);

        /* Fused QKV GEMM */
        op_gemm(ops, stream, d_qkv, blk->sa_qkv_w, d_normed, NULL, 3 * H_dim, H_dim, N1);

        /* Split interleaved QKV */
        void *d_Q = d_attn;    /* scratch[2] */
        void *d_K = d_normed;  /* scratch[2] offset */
        void *d_V = (char *)d_ca_V + ca_kv_sz; /* after ca_V in scratch[3] */
        op_split_qkv(ops, stream, d_Q, d_K, d_V, d_qkv, N1, heads, hd);

        /* QK RMSNorm */
        if (blk->sa_q_norm_w)
            op_rms_norm(ops, stream, d_Q, blk->sa_q_norm_w, N1, heads, hd, H_dim);
        if (blk->sa_k_norm_w)
            op_rms_norm(ops, stream, d_K, blk->sa_k_norm_w, N1, heads, hd, H_dim);

        op_self_attn(ops, stream, d_attn, d_Q, d_K, d_V, N1, H_dim, heads, hd);

        op_gemm(ops, stream, d_normed, blk->sa_out_w, d_attn, blk->sa_out_b, H_dim, H_dim, N1);
        op_add(ops, stream, d_hidden, d_normed, N1 * H_dim);

        /* === Cross-attention === */
        op_layernorm(ops, stream, d_normed, d_hidden, blk->norm2_w, blk->norm2_b, N1, H_dim);

        /* Q from hidden */
        op_gemm(ops, stream, d_cross_Q, blk->ca_q_w, d_normed, NULL, H_dim, H_dim, N1);
        if (blk->ca_q_norm_w)
            op_rms_norm(ops, stream, d_cross_Q, blk->ca_q_norm_w, N1, heads, hd, H_dim);

        /* Fused KV from context */
        {
            op_gemm(ops, stream, d_qkv, blk->ca_kv_w, d_context, NULL,
                    2 * H_dim, DIT_CONTEXT_DIM, ctx_len);
            op_split_kv(ops, stream, d_ca_K, d_ca_V, d_qkv, ctx_len, heads, hd);
        }

        if (blk->ca_k_norm_w)
            op_rms_norm(ops, stream, d_ca_K, blk->ca_k_norm_w,
                        ctx_len, heads, hd, H_dim);

        op_cross_attn(ops, stream, d_attn, d_cross_Q, d_ca_K, d_ca_V,
                      N1, ctx_len, H_dim, heads, hd);

        op_gemm(ops, stream, d_normed, blk->ca_out_w, d_attn, blk->ca_out_b, H_dim, H_dim, N1);
        op_add(ops, stream, d_hidden, d_normed, N1 * H_dim);

        /* === MLP or MoE === */
        if (blk->norm3_w) {
            op_layernorm(ops, stream, d_normed, d_hidden, blk->norm3_w, blk->norm3_b, N1, H_dim);
        } else {
            hipMemcpyAsync(d_normed, d_hidden, (size_t)N1 * H_dim * sizeof(float), hipMemcpyDeviceToDevice, stream);
        }

        if (blk->use_moe) {
            void *d_moe_out = d_mlp;
            run_dit_moe(r, blk, d_normed, d_moe_out, N1, d_moe_scratch);
            op_add(ops, stream, d_hidden, d_moe_out, N1 * H_dim);
        } else {
            op_gemm(ops, stream, d_mlp, blk->mlp_fc1_w, d_normed, blk->mlp_fc1_b, ffn, H_dim, N1);
            op_gelu(ops, stream, d_mlp, N1 * ffn);
            op_gemm(ops, stream, d_normed, blk->mlp_fc2_w, d_mlp, blk->mlp_fc2_b, H_dim, ffn, N1);
            op_add(ops, stream, d_hidden, d_normed, N1 * H_dim);
        }

        /* Per-block debug stats */
        if (r->verbose > 1) {
            hipStreamSynchronize(stream);
            float *hcpu = (float *)malloc((size_t)N1 * H_dim * sizeof(float));
            hipMemcpy(hcpu, d_hidden, (size_t)N1 * H_dim * sizeof(float), hipMemcpyDeviceToHost);
            float mn = hcpu[0], mx = hcpu[0], sm = 0;
            for (int j = 0; j < N1 * H_dim; j++) {
                if (hcpu[j] < mn) mn = hcpu[j];
                if (hcpu[j] > mx) mx = hcpu[j];
                sm += hcpu[j];
            }
            float mean = sm / (float)(N1 * H_dim);
            float var = 0;
            for (int j = 0; j < N1 * H_dim; j++) {
                float d = hcpu[j] - mean; var += d * d;
            }
            float std = sqrtf(var / (float)(N1 * H_dim));
            fprintf(stderr, "  block %2d: mean=%.6f std=%.6f min=%.6f max=%.6f%s%s\n",
                    bi, mean, std, mn, mx,
                    blk->use_moe ? " [MoE]" : "", blk->use_skip ? " [skip]" : "");
            free(hcpu);
        }
    }

    /* 5. Final layer: strip timestep token, LN, Linear -> output [N, C] */
    op_strip_first(ops, stream, d_normed, d_hidden, N1, H_dim);

    void *d_ln_out = d_attn;
    op_layernorm(ops, stream, d_ln_out, d_normed,
                 r->dit_final_ln_w, r->dit_final_ln_b, N, H_dim);

    op_gemm(ops, stream, d_output, r->dit_final_linear_w, d_ln_out,
            r->dit_final_linear_b, C, H_dim, N);

    free(skip_stack_cpu);
}

/* Stage 3: ShapeVAE single transformer block */
static void run_vae_block(hip_hy3d_runner *r, vae_block_gpu *b,
                           void *d_in, void *d_out,
                           void *d_scratch) {
    hy3d_ops *ops = &r->ops;
    hipStream_t stream = r->stream;
    const int N = VAE_NUM_LATENTS;
    const int W = VAE_WIDTH;
    const int H = VAE_HEADS;
    const int HD = VAE_HEAD_DIM;
    const int MLP = 4 * W;

    size_t off = 0;
    void *d_ln1   = (char *)d_scratch + off; off += (size_t)N * W * sizeof(float);
    void *d_qkv   = (char *)d_scratch + off; off += (size_t)N * 3 * W * sizeof(float);
    void *d_Q     = (char *)d_scratch + off; off += (size_t)N * W * sizeof(float);
    void *d_K     = (char *)d_scratch + off; off += (size_t)N * W * sizeof(float);
    void *d_V     = (char *)d_scratch + off; off += (size_t)N * W * sizeof(float);
    void *d_aout  = (char *)d_scratch + off; off += (size_t)N * W * sizeof(float);
    void *d_proj  = (char *)d_scratch + off; off += (size_t)N * W * sizeof(float);
    void *d_res1  = (char *)d_scratch + off; off += (size_t)N * W * sizeof(float);
    void *d_ln2   = (char *)d_scratch + off; off += (size_t)N * W * sizeof(float);
    void *d_mlph  = (char *)d_scratch + off; off += (size_t)N * MLP * sizeof(float);
    void *d_mlpo  = (char *)d_scratch + off;

    /* LN1 */
    op_layernorm(ops, stream, d_ln1, d_in, b->ln1_w, b->ln1_b, N, W);

    /* Fused QKV projection */
    op_gemm(ops, stream, d_qkv, b->qkv_w, d_ln1, NULL, 3 * W, W, N);

    /* Split interleaved QKV */
    op_split_qkv(ops, stream, d_Q, d_K, d_V, d_qkv, N, H, HD);

    /* QK normalization */
    if (b->use_qk_norm) {
        op_qk_layernorm(ops, stream, d_Q, b->q_norm_w, b->q_norm_b, N, H, HD, W);
        op_qk_layernorm(ops, stream, d_K, b->k_norm_w, b->k_norm_b, N, H, HD, W);
    }

    /* Self-attention */
    op_self_attn(ops, stream, d_aout, d_Q, d_K, d_V, N, W, H, HD);

    /* Output projection */
    op_gemm(ops, stream, d_proj, b->proj_w, d_aout, b->proj_b, W, W, N);

    /* Residual 1: res1 = input + proj */
    hipMemcpyAsync(d_res1, d_in, (size_t)N * W * sizeof(float), hipMemcpyDeviceToDevice, stream);
    op_add(ops, stream, d_res1, d_proj, N * W);

    /* LN2 -> MLP -> Residual 2 */
    op_layernorm(ops, stream, d_ln2, d_res1, b->ln2_w, b->ln2_b, N, W);
    op_gemm(ops, stream, d_mlph, b->mlp_fc_w, d_ln2, b->mlp_fc_b, MLP, W, N);
    op_gelu(ops, stream, d_mlph, N * MLP);
    op_gemm(ops, stream, d_mlpo, b->mlp_proj_w, d_mlph, b->mlp_proj_b, W, MLP, N);

    /* Output = res1 + mlp_out */
    hipMemcpyAsync(d_out, d_res1, (size_t)N * W * sizeof(float), hipMemcpyDeviceToDevice, stream);
    op_add(ops, stream, d_out, d_mlpo, N * W);
}

/* Stage 3: ShapeVAE decode + SDF query */
static void run_shapevae(hip_hy3d_runner *r, void *d_latents,
                          int grid_res, float *sdf_out) {
    hy3d_ops *ops = &r->ops;
    hipStream_t stream = r->stream;
    const int N = VAE_NUM_LATENTS;
    const int E = VAE_EMBED_DIM;
    const int W = VAE_WIDTH;

    /* Allocate decoder buffers */
    void *d_dec_a = gpu_alloc((size_t)N * W * sizeof(float));
    void *d_dec_b = gpu_alloc((size_t)N * W * sizeof(float));

    /* VAE block scratch size */
    size_t block_scratch = (size_t)N * (W * 12 + 4 * W * 4) * sizeof(float);
    void *d_block_scratch = gpu_alloc(block_scratch);

    /* Post-KL projection: [N, E] -> [N, W] */
    op_gemm(ops, stream, d_dec_a, r->vae_post_kl_w, d_latents, r->vae_post_kl_b, W, E, N);

    /* Run transformer blocks */
    void *d_cur = d_dec_a;
    void *d_next = d_dec_b;
    for (int i = 0; i < VAE_DEC_LAYERS; i++) {
        run_vae_block(r, &r->vae_blocks[i], d_cur, d_next, d_block_scratch);
        void *tmp = d_cur; d_cur = d_next; d_next = tmp;
    }

    /* d_cur now contains decoded latents [N, W] */
    int total_points = grid_res * grid_res * grid_res;
    int batch_size = 8192;
    float bounds[6] = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

    void *d_coords = gpu_alloc((size_t)batch_size * 3 * sizeof(float));
    void *d_fourier = gpu_alloc((size_t)batch_size * VAE_FOURIER_DIM * sizeof(float));
    void *d_query_proj = gpu_alloc((size_t)batch_size * W * sizeof(float));
    void *d_sdf_out = gpu_alloc((size_t)batch_size * sizeof(float));

    size_t geo_scratch_sz = (size_t)batch_size * (W * 10 + 4 * W * 4) * sizeof(float);
    void *d_geo_scratch = gpu_alloc(geo_scratch_sz);

    float dx = (bounds[3] - bounds[0]) / (float)(grid_res - 1);
    float dy = (bounds[4] - bounds[1]) / (float)(grid_res - 1);
    float dz = (bounds[5] - bounds[2]) / (float)(grid_res - 1);

    for (int start = 0; start < total_points; start += batch_size) {
        int count = (start + batch_size <= total_points) ? batch_size : (total_points - start);

        float *coords = (float *)malloc((size_t)count * 3 * sizeof(float));
        for (int i = 0; i < count; i++) {
            int idx = start + i;
            int iz = idx % grid_res;
            int iy = (idx / grid_res) % grid_res;
            int ix = idx / (grid_res * grid_res);
            coords[i*3+0] = bounds[0] + ix * dx;
            coords[i*3+1] = bounds[1] + iy * dy;
            coords[i*3+2] = bounds[2] + iz * dz;
        }
        hipMemcpyAsync(d_coords, coords, (size_t)count * 3 * sizeof(float), hipMemcpyHostToDevice, stream);
        free(coords);

        op_fourier_embed(ops, stream, d_fourier, d_coords, r->vae_fourier_freqs,
                         count, VAE_NUM_FREQS, VAE_FOURIER_DIM);

        op_gemm(ops, stream, d_query_proj, r->vae_geo.query_proj_w, d_fourier,
                r->vae_geo.query_proj_b, W, VAE_FOURIER_DIM, count);

        /* Cross-attention with decoded latents */
        vae_geo_decoder_gpu *g = &r->vae_geo;

        size_t geo_off = 0;
        void *d_g_ln1  = (char *)d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_ln2  = (char *)d_geo_scratch + geo_off; geo_off += (size_t)N * W * sizeof(float);
        void *d_g_Q    = (char *)d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_KV   = (char *)d_geo_scratch + geo_off; geo_off += (size_t)N * 2 * W * sizeof(float);
        void *d_g_K    = (char *)d_geo_scratch + geo_off; geo_off += (size_t)N * W * sizeof(float);
        void *d_g_V    = (char *)d_geo_scratch + geo_off; geo_off += (size_t)N * W * sizeof(float);
        void *d_g_aout = (char *)d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_proj = (char *)d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_res  = (char *)d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_ln3  = (char *)d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_mlph = (char *)d_geo_scratch + geo_off; geo_off += (size_t)count * 4 * W * sizeof(float);
        void *d_g_mlpo = (char *)d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_post = (char *)d_geo_scratch + geo_off;

        op_layernorm(ops, stream, d_g_ln1, d_query_proj, g->ln1_w, g->ln1_b, count, W);
        op_layernorm(ops, stream, d_g_ln2, d_cur, g->ln2_w, g->ln2_b, N, W);

        op_gemm(ops, stream, d_g_Q, g->c_q_w, d_g_ln1, NULL, W, W, count);
        op_gemm(ops, stream, d_g_KV, g->c_kv_w, d_g_ln2, NULL, 2 * W, W, N);

        op_split_kv(ops, stream, d_g_K, d_g_V, d_g_KV, N, VAE_HEADS, VAE_HEAD_DIM);

        if (g->use_qk_norm) {
            op_qk_layernorm(ops, stream, d_g_Q, g->q_norm_w, g->q_norm_b,
                            count, VAE_HEADS, VAE_HEAD_DIM, W);
            op_qk_layernorm(ops, stream, d_g_K, g->k_norm_w, g->k_norm_b,
                            N, VAE_HEADS, VAE_HEAD_DIM, W);
        }

        op_cross_attn(ops, stream, d_g_aout, d_g_Q, d_g_K, d_g_V,
                      count, N, W, VAE_HEADS, VAE_HEAD_DIM);

        op_gemm(ops, stream, d_g_proj, g->c_proj_w, d_g_aout, g->c_proj_b, W, W, count);
        hipMemcpyAsync(d_g_res, d_query_proj, (size_t)count * W * sizeof(float), hipMemcpyDeviceToDevice, stream);
        op_add(ops, stream, d_g_res, d_g_proj, count * W);

        op_layernorm(ops, stream, d_g_ln3, d_g_res, g->ln3_w, g->ln3_b, count, W);
        op_gemm(ops, stream, d_g_mlph, g->mlp_fc_w, d_g_ln3, g->mlp_fc_b, 4 * W, W, count);
        op_gelu(ops, stream, d_g_mlph, count * 4 * W);
        op_gemm(ops, stream, d_g_mlpo, g->mlp_proj_w, d_g_mlph, g->mlp_proj_b, W, 4 * W, count);
        hipMemcpyAsync(d_g_post, d_g_res, (size_t)count * W * sizeof(float), hipMemcpyDeviceToDevice, stream);
        op_add(ops, stream, d_g_post, d_g_mlpo, count * W);

        if (g->ln_post_w) {
            op_layernorm(ops, stream, d_g_ln1, d_g_post, g->ln_post_w, g->ln_post_b, count, W);
        } else {
            hipMemcpyAsync(d_g_ln1, d_g_post, (size_t)count * W * sizeof(float), hipMemcpyDeviceToDevice, stream);
        }

        op_gemm(ops, stream, d_sdf_out, g->output_w, d_g_ln1, g->output_b, 1, W, count);

        hipMemcpyAsync(sdf_out + start, d_sdf_out,
                          (size_t)count * sizeof(float), hipMemcpyDeviceToHost, stream);
    }

    hipStreamSynchronize(stream);

    /* Cleanup */
    hipFree(d_dec_a);
    hipFree(d_dec_b);
    hipFree(d_block_scratch);
    hipFree(d_coords);
    hipFree(d_fourier);
    hipFree(d_query_proj);
    hipFree(d_sdf_out);
    hipFree(d_geo_scratch);
}

/* ======================================================================== */
/* Random noise generation (CPU, Box-Muller)                                */
/* ======================================================================== */

static void generate_randn(float *buf, int n, uint32_t seed) {
    uint64_t s0 = seed ? seed : (uint64_t)time(NULL);
    uint64_t s1 = s0 ^ 0x6c62272e07bb0142ULL;

    for (int i = 0; i < n; i += 2) {
        uint64_t x = s0;
        uint64_t y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        uint64_t r1 = s1 + y;

        x = s0; y = s1; s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        uint64_t r2 = s1 + y;

        double u1 = ((r1 >> 11) + 0.5) / 2097152.0;
        double u2 = ((r2 >> 11) + 0.5) / 2097152.0;
        double rr = sqrt(-2.0 * log(u1));
        double theta = 2.0 * 3.141592653589793 * u2;
        buf[i] = (float)(rr * cos(theta));
        if (i + 1 < n) buf[i+1] = (float)(rr * sin(theta));
    }
}

/* ======================================================================== */
/* Public API                                                               */
/* ======================================================================== */

hip_hy3d_runner *hip_hy3d_init(int device_id, int verbose) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "HY3D: rocew init failed\n");
        return NULL;
    }
    if (hipInit(0) != hipSuccess) {
        fprintf(stderr, "HY3D: hipInit failed\n");
        return NULL;
    }

    hip_hy3d_runner *r = (hip_hy3d_runner *)calloc(1, sizeof(hip_hy3d_runner));
    r->verbose = verbose;
    r->device = device_id;

    HIP_CHECK_NULL(hipSetDevice(device_id));
    HIP_CHECK_NULL(hipCtxCreate(&r->ctx, 0, r->device));
    HIP_CHECK_NULL(hipStreamCreateWithFlags(&r->stream, hipStreamNonBlocking));

    if (verbose) {
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, device_id) == hipSuccess) {
            fprintf(stderr, "HY3D: using GPU %d: %s (%s)\n", device_id, props.name, props.gcnArchName);
        }
    }

    if (hy3d_compile_kernels(r) != 0) {
        fprintf(stderr, "HY3D: kernel compilation failed\n");
        free(r);
        return NULL;
    }

    return r;
}

void hip_hy3d_set_f32_gemm(hip_hy3d_runner *r, int enable) {
    if (!r) return;
    r->ops.use_f32_gemm = enable;
    if (r->verbose)
        fprintf(stderr, "HY3D: GEMM mode set to %s\n",
                enable ? "F32 (PyTorch-compatible)" : "F16 (default)");
}

int hip_hy3d_load_weights(hip_hy3d_runner *r,
                           const char *conditioner_path,
                           const char *model_path,
                           const char *vae_path) {
    if (!r) return -1;

    if (conditioner_path && load_dino_weights(r, conditioner_path) != 0)
        return -1;
    if (model_path && load_dit_weights(r, model_path) != 0)
        return -1;
    if (vae_path && load_vae_weights(r, vae_path) != 0)
        return -1;

    return 0;
}

hy3d_mesh hip_hy3d_predict(hip_hy3d_runner *r,
                            const uint8_t *rgb, int w, int h,
                            int n_steps, float guidance_scale,
                            int grid_res, uint32_t seed) {
    hy3d_ops *ops = &r->ops;
    hipStream_t stream = r->stream;
    hy3d_mesh result = {0};
    if (!r || !r->dino_loaded || !r->dit_loaded || !r->vae_loaded) {
        fprintf(stderr, "HY3D: runner not fully initialized\n");
        return result;
    }

    if (n_steps <= 0) n_steps = 30;
    if (guidance_scale <= 0.0f) guidance_scale = 7.5f;
    if (grid_res <= 0) grid_res = 256;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* ---- Stage 1: DINOv2 image encoding ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 1 - DINOv2 encoding...\n");

    void *d_rgb = gpu_alloc((size_t)w * h * 3);
    hipMemcpyAsync(d_rgb, rgb, (size_t)w * h * 3, hipMemcpyHostToDevice, stream);

    void *d_image = gpu_alloc((size_t)3 * DINO_IMG_SIZE * DINO_IMG_SIZE * sizeof(float));
    {
        int dw = DINO_IMG_SIZE, dh = DINO_IMG_SIZE;
        float mean0 = 0.485f, mean1 = 0.456f, mean2 = 0.406f;
        float istd0 = 1.0f/0.229f, istd1 = 1.0f/0.224f, istd2 = 1.0f/0.225f;
        void *args[] = {&d_image, &d_rgb, &w, &h, &dw, &dh,
                        &mean0, &mean1, &mean2, &istd0, &istd1, &istd2};
        hipModuleLaunchKernel(ops->resize_normalize,
                       (unsigned)((dw*dh+255)/256), 1, 1,
                       256, 1, 1, 0, stream, args, NULL);
    }

    void *d_dino_out = gpu_alloc((size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float));
    run_dinov2(r, d_image, d_dino_out);
    hipFree(d_rgb);
    hipFree(d_image);

    /* ---- Stage 2: DiT diffusion with flow matching ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 2 - DiT diffusion (%d steps)...\n", n_steps);

    int latent_size = DIT_INPUT_SIZE * DIT_IN_CHANNELS;
    float *noise_cpu = (float *)malloc((size_t)latent_size * sizeof(float));
    generate_randn(noise_cpu, latent_size, seed);

    void *d_latents = gpu_alloc((size_t)latent_size * sizeof(float));
    hipMemcpyAsync(d_latents, noise_cpu, (size_t)latent_size * sizeof(float), hipMemcpyHostToDevice, stream);
    free(noise_cpu);

    void *d_pred_cond = gpu_alloc((size_t)latent_size * sizeof(float));
    void *d_pred_uncond = gpu_alloc((size_t)latent_size * sizeof(float));
    void *d_pred_combined = gpu_alloc((size_t)latent_size * sizeof(float));

    void *d_uncond_ctx = gpu_alloc((size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float));
    hipMemsetAsync(d_uncond_ctx, 0, (size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float), stream);

    for (int step = 0; step < n_steps; step++) {
        float t_current = 1.0f - (float)step / (float)n_steps;
        float t_next = 1.0f - (float)(step + 1) / (float)n_steps;
        float dt = t_current - t_next;

        if (r->verbose && (step % 5 == 0 || step == n_steps - 1))
            fprintf(stderr, "  step %d/%d (t=%.3f)\n", step+1, n_steps, t_current);

        run_dit_forward(r, d_latents, t_current, d_dino_out, d_pred_cond);
        run_dit_forward(r, d_latents, t_current, d_uncond_ctx, d_pred_uncond);

        op_cfg_combine(ops, stream, d_pred_combined, d_pred_cond, d_pred_uncond,
                       guidance_scale, latent_size);

        op_euler_step(ops, stream, d_latents, d_pred_combined, dt, latent_size);
    }

    hipFree(d_pred_cond);
    hipFree(d_pred_uncond);
    hipFree(d_pred_combined);
    hipFree(d_uncond_ctx);
    hipFree(d_dino_out);

    /* ---- Stage 3: ShapeVAE decode + SDF query ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 3 - ShapeVAE decode (grid %d^3)...\n", grid_res);

    int total_pts = grid_res * grid_res * grid_res;
    float *sdf_grid = (float *)malloc((size_t)total_pts * sizeof(float));
    run_shapevae(r, d_latents, grid_res, sdf_grid);
    hipFree(d_latents);

    /* SDF grid stats */
    if (r->verbose) {
        float smin = sdf_grid[0], smax = sdf_grid[0], ssum = 0;
        int n_neg = 0, n_pos = 0;
        for (int i = 0; i < total_pts; i++) {
            if (sdf_grid[i] < smin) smin = sdf_grid[i];
            if (sdf_grid[i] > smax) smax = sdf_grid[i];
            ssum += sdf_grid[i];
            if (sdf_grid[i] < 0) n_neg++;
            else n_pos++;
        }
        fprintf(stderr, "HY3D: SDF stats: min=%.4f max=%.4f mean=%.4f neg=%d pos=%d\n",
                smin, smax, ssum / total_pts, n_neg, n_pos);
    }

    /* ---- Stage 4: Marching cubes ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 4 - Marching cubes...\n");

    mc_mesh mc = mc_marching_cubes(sdf_grid, grid_res, grid_res, grid_res, 0.0f, NULL);
    free(sdf_grid);

    result.vertices = mc.vertices;
    result.triangles = mc.triangles;
    result.n_verts = mc.n_verts;
    result.n_tris = mc.n_tris;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (r->verbose)
        fprintf(stderr, "HY3D: done in %.2fs (%d verts, %d tris)\n",
                elapsed, result.n_verts, result.n_tris);

    return result;
}

/* ======================================================================== */
/* Per-stage verification API                                               */
/* ======================================================================== */

int hip_hy3d_run_dinov2(hip_hy3d_runner *r,
                          const float *image_f32,
                          float *output) {
    if (!r || !r->dino_loaded) return -1;

    size_t img_bytes = (size_t)3 * DINO_IMG_SIZE * DINO_IMG_SIZE * sizeof(float);
    size_t out_bytes = (size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float);

    void *d_image = gpu_alloc(img_bytes);
    void *d_out = gpu_alloc(out_bytes);

    hipMemcpyAsync(d_image, image_f32, img_bytes, hipMemcpyHostToDevice, r->stream);
    run_dinov2(r, d_image, d_out);
    hipMemcpyAsync(output, d_out, out_bytes, hipMemcpyDeviceToHost, r->stream);
    hipStreamSynchronize(r->stream);

    hipFree(d_image);
    hipFree(d_out);
    return 0;
}

int hip_hy3d_run_vae(hip_hy3d_runner *r,
                       const float *latents,
                       int grid_res,
                       float *sdf_out) {
    if (!r || !r->vae_loaded) return -1;

    size_t lat_bytes = (size_t)VAE_NUM_LATENTS * VAE_EMBED_DIM * sizeof(float);
    void *d_latents = gpu_alloc(lat_bytes);
    hipMemcpyAsync(d_latents, latents, lat_bytes, hipMemcpyHostToDevice, r->stream);

    run_shapevae(r, d_latents, grid_res, sdf_out);

    hipFree(d_latents);
    return 0;
}

int hip_hy3d_run_dit(hip_hy3d_runner *r,
                       const float *latents,
                       float timestep,
                       const float *context,
                       float *output) {
    if (!r || !r->dit_loaded) return -1;

    size_t lat_bytes = (size_t)DIT_INPUT_SIZE * DIT_IN_CHANNELS * sizeof(float);
    size_t ctx_bytes = (size_t)DINO_SEQ_LEN * DIT_CONTEXT_DIM * sizeof(float);
    size_t out_bytes = lat_bytes;

    void *d_latents = gpu_alloc(lat_bytes);
    void *d_context = gpu_alloc(ctx_bytes);
    void *d_output  = gpu_alloc(out_bytes);

    hipMemcpyAsync(d_latents, latents, lat_bytes, hipMemcpyHostToDevice, r->stream);
    hipMemcpyAsync(d_context, context, ctx_bytes, hipMemcpyHostToDevice, r->stream);

    run_dit_forward(r, d_latents, timestep, d_context, d_output);

    hipMemcpyAsync(output, d_output, out_bytes, hipMemcpyDeviceToHost, r->stream);
    hipStreamSynchronize(r->stream);

    hipFree(d_latents);
    hipFree(d_context);
    hipFree(d_output);
    return 0;
}

void hip_hy3d_free(hip_hy3d_runner *r) {
    if (!r) return;

    /* Free scratch buffers */
    for (int i = 0; i < 8; i++) {
        if (r->scratch[i]) hipFree(r->scratch[i]);
    }

    /* Free pre-computed K,V */
    for (int i = 0; i < DIT_DEPTH; i++) {
        if (r->dit_ca_K[i]) hipFree(r->dit_ca_K[i]);
        if (r->dit_ca_V[i]) hipFree(r->dit_ca_V[i]);
    }

    /* Free Fourier frequencies */
    if (r->vae_fourier_freqs) hipFree(r->vae_fourier_freqs);

    /* Note: individual weight buffers are GPU allocations that should also be freed.
     * For a full cleanup, iterate all void* fields. For now, destroying
     * the HIP context reclaims all GPU memory. */

    if (r->module) hipModuleUnload(r->module);
    if (r->stream) hipStreamDestroy(r->stream);
    if (r->ctx) hipCtxDestroy(r->ctx);

    free(r);
}
