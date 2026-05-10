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
#include <stdint.h>

/* ======================================================================== */
/* .npy dump helpers (HY3D_DUMP_DIR env + per-step latent/velocity dumps)   */
/* ======================================================================== */

static void hy3d_write_npy_f32_2d(const char *path, const float *buf, int rows, int cols) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    static const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
    char hdr[256];
    int hlen = snprintf(hdr, sizeof(hdr),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", rows, cols);
    int total = 10 + hlen + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(hdr, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(buf, sizeof(float), (size_t)rows * (size_t)cols, f);
    fclose(f);
}

static void hy3d_dbg_dump_npy(hipStream_t stream, void *d_buf,
                              int rows, int cols, const char *fname) {
    const char *dir = getenv("HY3D_DUMP_DIR");
    if (!dir || !*dir) return;
    size_t n = (size_t)rows * (size_t)cols;
    float *h = (float *)malloc(n * sizeof(float));
    if (!h) return;
    hipStreamSynchronize(stream);
    hipMemcpy(h, d_buf, n * sizeof(float), hipMemcpyDeviceToHost);
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, fname);
    hy3d_write_npy_f32_2d(path, h, rows, cols);
    free(h);
}

static void hy3d_dump_latent(hipStream_t stream, void *d_latents,
                             int rows, int cols, const char *prefix, int step1) {
    if (!prefix || !*prefix) return;
    size_t n = (size_t)rows * (size_t)cols;
    float *h = (float *)malloc(n * sizeof(float));
    if (!h) return;
    hipStreamSynchronize(stream);
    hipMemcpy(h, d_latents, n * sizeof(float), hipMemcpyDeviceToHost);
    char path[1024];
    snprintf(path, sizeof(path), "%s_%03d.npy", prefix, step1);
    hy3d_write_npy_f32_2d(path, h, rows, cols);
    free(h);
}

#define HY3D_MAX_LATENT_DUMP_STEPS 64

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

    /* Per-step latent/velocity .npy dumps for trajectory checks */
    int latent_dump_count;
    int latent_dump_steps[HY3D_MAX_LATENT_DUMP_STEPS];
    char latent_dump_prefix[512];
    int velocity_dump_count;
    int velocity_dump_steps[HY3D_MAX_LATENT_DUMP_STEPS];
    char velocity_dump_prefix[512];

    /* Deterministic trajectory overrides */
    float *init_latents;      /* host-side [4096,64] override */
    int init_latents_n;
    float *init_ctx_cond;     /* host-side [1370,1024] override */
    float *init_ctx_uncond;   /* host-side [1370,1024] override (optional) */
    int init_ctx_n;

    /* Load status */
    int dino_loaded, dit_loaded, vae_loaded;

    /* Per-stage GPU timing */
    int timing_enabled;
    int timing_valid;
    hy3d_stage_times timings;
};

static int hy3d_should_dump_latent(const struct hip_hy3d_runner *r, int step1) {
    for (int i = 0; i < r->latent_dump_count; i++) {
        if (r->latent_dump_steps[i] == step1) return 1;
    }
    return 0;
}

static int hy3d_should_dump_velocity(const struct hip_hy3d_runner *r, int step1) {
    for (int i = 0; i < r->velocity_dump_count; i++) {
        if (r->velocity_dump_steps[i] == step1) return 1;
    }
    return 0;
}

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
    hipError_t e = hipMalloc(&d, bytes);
    if (e != hipSuccess) {
        size_t mfree = 0, mtotal = 0;
        if (hipMemGetInfo) hipMemGetInfo(&mfree, &mtotal);
        fprintf(stderr, "HY3D: GPU alloc failed for %zu bytes (err=%d; free=%zu total=%zu)\n",
                bytes, (int)e, mfree, mtotal);
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
        if (r->scratch[idx]) {
            hipFree(r->scratch[idx]);
            r->scratch[idx] = NULL;
            r->scratch_size[idx] = 0;
            hipDeviceSynchronize();
        }
        void *d = NULL;
        if (hipMalloc(&d, bytes) != hipSuccess) {
            fprintf(stderr, "HY3D: scratch[%d] GPU alloc failed for %zu bytes\n",
                    idx, bytes);
            r->scratch[idx] = NULL;
            return;
        }
        r->scratch[idx] = d;
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
        int dim2 = dim, ps2 = ps, img_w = DINO_IMG_SIZE, img_h = DINO_IMG_SIZE;
        void *pw = r->dino_patch_w, *pb = r->dino_patch_b;
        void *args[] = {&d_hidden, &d_image, &pw, &pb, &gw2, &dim2, &ps2, &img_w, &img_h};
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
    hy3d_dbg_dump_npy(stream, d_hidden, seq, dim, "hip_dinov2_hidden_0.npy");

    /* 3. Encoder layers */
    for (int li = 0; li < DINO_LAYERS; li++) {
        dino_layer_gpu *l = &r->dino_layers[li];

        if (getenv("HY3D_TRACE_DINO")) {
            hipStreamSynchronize(stream);
            fprintf(stderr, "  DINO block %d start\n", li);
        }

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
        op_gelu_exact(ops, stream, d_mlp, seq * ffn);
        op_gemm(ops, stream, d_normed, l->fc2_w, d_mlp, l->fc2_b, dim, ffn, seq);

        if (l->ls2)
            op_layerscale_add(ops, stream, d_hidden, d_normed, l->ls2, seq * dim, dim);
        else
            op_add(ops, stream, d_hidden, d_normed, seq * dim);

        if (getenv("HY3D_TRACE_DINO")) {
            hipStreamSynchronize(stream);
            fprintf(stderr, "  DINO block %d done\n", li);
        }

        /* HF hidden_states index = block_index + 1 */
        if (li == 11)
            hy3d_dbg_dump_npy(stream, d_hidden, seq, dim, "hip_dinov2_hidden_12.npy");
        else if (li == 22)
            hy3d_dbg_dump_npy(stream, d_hidden, seq, dim, "hip_dinov2_hidden_23.npy");
        else if (li == 23)
            hy3d_dbg_dump_npy(stream, d_hidden, seq, dim, "hip_dinov2_hidden_24.npy");
    }

    /* 4. Final LN */
    if (r->dino_final_ln_w) {
        op_layernorm(ops, stream, d_out, d_hidden, r->dino_final_ln_w, r->dino_final_ln_b, seq, dim);
    } else {
        hipMemcpyAsync(d_out, d_hidden, (size_t)seq * dim * sizeof(float), hipMemcpyDeviceToDevice, stream);
    }
}

/* fp16 DINOv2 forward — same structure as run_dinov2 but residual stream
 * carries f16 through the 24 encoder layers using hipBLASLt fp16 GEMMs and
 * WMMA fp16 self-attention.  patch_embed + cls_pos_embed stay f32 (run once);
 * we cast to f16 once before the loop and back to f32 once at the end so the
 * caller interface (f32 d_out) is unchanged.  Weights must already be f16. */
static void run_dinov2_fp16(hip_hy3d_runner *r, void *d_image, void *d_out) {
    hy3d_ops *ops = &r->ops;
    hipStream_t stream = r->stream;
    const int seq = DINO_SEQ_LEN;
    const int dim = DINO_HIDDEN;
    const int heads = DINO_HEADS;
    const int hd = DINO_HEAD_DIM;
    const int ffn = DINO_FFN;
    const int ps = DINO_PATCH;
    const int gw = DINO_IMG_SIZE / ps;

    const size_t f16b = sizeof(uint16_t);
    /* f32 buffers (one-shot patch+pos): scratch[0]=hidden_f32, scratch[2]=final_f32 */
    ensure_scratch(r, 0, (size_t)seq * dim * sizeof(float));
    ensure_scratch(r, 2, (size_t)seq * dim * sizeof(float));
    /* f16 buffers: scratch[1] is multi-purpose, we partition it.  Need:
     *   hidden_f16   [seq*dim]
     *   qkv_f16      [3*seq*dim]
     *   attn_f16     [seq*dim]
     *   mlp_f16      [seq*ffn]
     *   normed_f16   [seq*dim] */
    size_t need = (size_t)seq * dim * f16b           /* hidden */
                + (size_t)3 * seq * dim * f16b       /* qkv    */
                + (size_t)seq * dim * f16b           /* attn   */
                + (size_t)seq * ffn * f16b           /* mlp    */
                + (size_t)seq * dim * f16b;          /* normed */
    ensure_scratch(r, 1, need);

    void *d_hidden_f32 = r->scratch[0];
    char *p = (char *)r->scratch[1];
    void *d_hidden = p;       p += (size_t)seq * dim * f16b;
    void *d_qkv    = p;       p += (size_t)3 * seq * dim * f16b;
    void *d_attn   = p;       p += (size_t)seq * dim * f16b;
    void *d_mlp    = p;       p += (size_t)seq * ffn * f16b;
    void *d_normed = p;
    void *d_final_f32 = r->scratch[2];

    /* 1. patch embed (f32) */
    {
        int gw2 = gw, dim2 = dim, ps2 = ps, img_w = DINO_IMG_SIZE, img_h = DINO_IMG_SIZE;
        void *pw = r->dino_patch_w, *pb = r->dino_patch_b;
        void *args[] = {&d_hidden_f32, &d_image, &pw, &pb, &gw2, &dim2, &ps2, &img_w, &img_h};
        hipModuleLaunchKernel(ops->patch_embed,
                       (unsigned)(gw * gw), 1, 1,
                       256, 1, 1, 0, stream, args, NULL);
    }

    /* 2. cls + pos (f32) */
    {
        int n_tok = seq, d2 = dim;
        void *cls = r->dino_cls_token, *pos = r->dino_pos_emb;
        void *args[] = {&d_hidden_f32, &cls, &pos, &n_tok, &d2};
        hipModuleLaunchKernel(ops->cls_pos_embed,
                       (unsigned)((seq*dim+255)/256), 1, 1,
                       256, 1, 1, 0, stream, args, NULL);
    }

    /* cast f32 -> f16 once */
    op_cast_f32_to_f16(ops, stream, d_hidden, d_hidden_f32, seq * dim);

    /* 3. Encoder layers (24x) in f16 */
    for (int li = 0; li < DINO_LAYERS; li++) {
        dino_layer_gpu *l = &r->dino_layers[li];

        op_layernorm_f16(ops, stream, d_normed, d_hidden, l->ln1_w, l->ln1_b, seq, dim);

        void *d_Q = d_qkv;
        void *d_K = (char *)d_qkv + (size_t)seq * dim * f16b;
        void *d_V = (char *)d_qkv + (size_t)2 * seq * dim * f16b;
        op_gemm_f16_bias_f16d(ops, stream, d_Q, l->q_w, d_normed, l->q_b, dim, dim, seq);
        op_gemm_f16_bias_f16d(ops, stream, d_K, l->k_w, d_normed, l->k_b, dim, dim, seq);
        op_gemm_f16_bias_f16d(ops, stream, d_V, l->v_w, d_normed, l->v_b, dim, dim, seq);

        if (op_self_attn_f16(ops, stream, d_attn, d_Q, d_K, d_V, seq, dim, heads, hd) != 0) {
            fprintf(stderr, "HY3D: DINOv2 fp16 self-attn dispatch failed (head_dim=%d)\n", hd);
            return;
        }

        op_gemm_f16_bias_f16d(ops, stream, d_normed, l->out_w, d_attn, l->out_b, dim, dim, seq);

        if (l->ls1 && ops->layerscale_add_f16)
            op_layerscale_add_f16(ops, stream, d_hidden, d_normed, l->ls1, seq * dim, dim);
        else
            op_add_f16(ops, stream, d_hidden, d_normed, seq * dim);

        op_layernorm_f16(ops, stream, d_normed, d_hidden, l->ln2_w, l->ln2_b, seq, dim);
        op_gemm_f16_bias_gelu_f16d(ops, stream, d_mlp, l->fc1_w, d_normed, l->fc1_b, ffn, dim, seq);
        op_gemm_f16_bias_f16d(ops, stream, d_normed, l->fc2_w, d_mlp, l->fc2_b, dim, ffn, seq);

        if (l->ls2 && ops->layerscale_add_f16)
            op_layerscale_add_f16(ops, stream, d_hidden, d_normed, l->ls2, seq * dim, dim);
        else
            op_add_f16(ops, stream, d_hidden, d_normed, seq * dim);
    }

    /* 4. Final LN (f16 in, then cast to f32 out) */
    if (r->dino_final_ln_w) {
        /* Cast final hidden to f32, run f32 layernorm into d_out.  Avoids
         * needing an f16 layernorm with f32 D output. */
        op_cast_f16_to_f32(ops, stream, d_final_f32, d_hidden, seq * dim);
        op_layernorm(ops, stream, d_out, d_final_f32,
                     r->dino_final_ln_w, r->dino_final_ln_b, seq, dim);
    } else {
        op_cast_f16_to_f32(ops, stream, d_out, d_hidden, seq * dim);
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

    /* Scratch layout within d_moe_scratch.  d_weights replaces the host
     * gate_weights buffer; sized [N_tok, DIT_N_EXPERTS]. */
    size_t off = 0;
    void *d_gate    = (char *)d_moe_scratch + off; off += (size_t)N_tok * DIT_N_EXPERTS * sizeof(float);
    void *d_weights = (char *)d_moe_scratch + off; off += (size_t)N_tok * DIT_N_EXPERTS * sizeof(float);
    void *d_exp_h   = (char *)d_moe_scratch + off; off += (size_t)N_tok * ffn * sizeof(float);
    void *d_exp_o   = (char *)d_moe_scratch + off; /* [N_tok * H_dim] */

    /* Step 1: Gate logits: [N_tok, H_dim] @ [E, H_dim]^T -> [N_tok, E] */
    op_gemm(ops, stream, d_gate, blk->moe_gate_w, d_input, NULL,
            DIT_N_EXPERTS, H_dim, N_tok);

    /* Step 2: GPU softmax + top-K masking → d_weights (sparse, top-K only). */
    op_moe_gate_softmax_topk(ops, stream, d_gate, d_weights,
                             N_tok, DIT_N_EXPERTS, DIT_MOE_TOP_K);

    /* Step 3: Zero accumulator. */
    hipMemsetAsync(d_output, 0, (size_t)N_tok * H_dim * sizeof(float), stream);

    /* Step 4: All-experts × weighted-add on the GPU.  No host round-trip;
     * the per-token zero-weight short-circuit happens inside the
     * moe_weighted_add_f32 kernel. */
    for (int e = 0; e < DIT_N_EXPERTS; e++) {
        op_gemm(ops, stream, d_exp_h, blk->moe_expert_fc1_w[e], d_input,
                blk->moe_expert_fc1_b[e], ffn, H_dim, N_tok);
        op_gelu_exact(ops, stream, d_exp_h, N_tok * ffn);
        op_gemm(ops, stream, d_exp_o, blk->moe_expert_fc2_w[e], d_exp_h,
                blk->moe_expert_fc2_b[e], H_dim, ffn, N_tok);
        op_moe_weighted_add(ops, stream, d_output, d_exp_o, d_weights,
                            e, DIT_N_EXPERTS, N_tok, H_dim);
    }

    /* Step 5: Shared expert (always full weight). */
    op_gemm(ops, stream, d_exp_h, blk->moe_shared_fc1_w, d_input,
            blk->moe_shared_fc1_b, ffn, H_dim, N_tok);
    op_gelu_exact(ops, stream, d_exp_h, N_tok * ffn);
    op_gemm(ops, stream, d_exp_o, blk->moe_shared_fc2_w, d_exp_h,
            blk->moe_shared_fc2_b, H_dim, ffn, N_tok);
    op_add(ops, stream, d_output, d_exp_o, N_tok * H_dim);
}

#ifdef HY3D_HIPBLASLT_ENABLED
/* MoE forward (FP16 path): input f16, output f32, weights f16, biases f32.
 * Replaces the f32 op_gemm path with hipBLASLt fp16 GEMMs (with fused GELU
 * on fc1 and bias on fc2). Cuts MoE GEMM time roughly in line with the
 * non-MoE blocks (~3-4x in our DiT). */
static void run_dit_moe_fp16(hip_hy3d_runner *r, dit_block_gpu *blk,
                              void *d_input_f16, void *d_output_f32,
                              int N_tok, void *d_moe_scratch) {
    hy3d_ops *ops = &r->ops;
    hipStream_t stream = r->stream;
    const int H_dim = DIT_HIDDEN;
    const int ffn = DIT_FFN;

    /* Reuse the same scratch carve-up as run_dit_moe; sizes are upper bounds.
     * d_exp_h is reinterpreted as f16 (smaller), so it fits within the
     * f32-sized slot. */
    size_t off = 0;
    void *d_gate    = (char *)d_moe_scratch + off; off += (size_t)N_tok * DIT_N_EXPERTS * sizeof(float);
    void *d_weights = (char *)d_moe_scratch + off; off += (size_t)N_tok * DIT_N_EXPERTS * sizeof(float);
    void *d_exp_h_f16 = (char *)d_moe_scratch + off; off += (size_t)N_tok * ffn * sizeof(float); /* slot is f32-sized; we use f16 */
    void *d_exp_o_f32 = (char *)d_moe_scratch + off; /* [N_tok * H_dim] f32 */

    /* Step 1: Gate logits, fp16 input × fp16 weight → f32 output. */
    op_gemm_f16(ops, stream, d_gate, blk->moe_gate_w, d_input_f16,
                DIT_N_EXPERTS, H_dim, N_tok);

    /* Step 2: GPU softmax + top-K → d_weights (sparse, top-K only). */
    op_moe_gate_softmax_topk(ops, stream, d_gate, d_weights,
                             N_tok, DIT_N_EXPERTS, DIT_MOE_TOP_K);

    /* Step 3: Zero accumulator. */
    hipMemsetAsync(d_output_f32, 0, (size_t)N_tok * H_dim * sizeof(float), stream);

    /* Step 4: All-experts × weighted-add on the GPU. */
    for (int e = 0; e < DIT_N_EXPERTS; e++) {
        op_gemm_f16_bias_gelu_f16d(ops, stream, d_exp_h_f16,
                                   blk->moe_expert_fc1_w[e], d_input_f16,
                                   blk->moe_expert_fc1_b[e], ffn, H_dim, N_tok);
        op_gemm_f16_bias(ops, stream, d_exp_o_f32,
                         blk->moe_expert_fc2_w[e], d_exp_h_f16,
                         blk->moe_expert_fc2_b[e], H_dim, ffn, N_tok);
        op_moe_weighted_add(ops, stream, d_output_f32, d_exp_o_f32, d_weights,
                            e, DIT_N_EXPERTS, N_tok, H_dim);
    }

    /* Step 5: Shared expert. */
    op_gemm_f16_bias_gelu_f16d(ops, stream, d_exp_h_f16,
                               blk->moe_shared_fc1_w, d_input_f16,
                               blk->moe_shared_fc1_b, ffn, H_dim, N_tok);
    op_gemm_f16_bias(ops, stream, d_exp_o_f32,
                     blk->moe_shared_fc2_w, d_exp_h_f16,
                     blk->moe_shared_fc2_b, H_dim, ffn, N_tok);
    op_add(ops, stream, d_output_f32, d_exp_o_f32, N_tok * H_dim);
}

/* MoE forward (FP16 + top-K dispatch): only runs each expert on the tokens
 * that picked it (top-K=2 of 8 ⇒ ~4× less expert FC work vs run_dit_moe_fp16).
 *
 * Layout in d_moe_scratch (164MB available — same slot as the dense version):
 *   d_gate     [N_tok, E]      f32   (gate logits → softmax weights)
 *   d_weights  [N_tok, E]      f32   (top-K mask × softmax probs)
 *   d_counts   [E]             int   (per-expert dispatch counts)
 *   d_perm     [E, N_tok]      int   (per-expert token indices)
 *   d_packed_in[N_tok, H_dim]  f16   (gathered input rows)
 *   d_packed_h [N_tok, ffn]    f16   (fc1 output, GELU fused)
 *   d_packed_o [N_tok, H_dim]  f16   (fc2 output)
 *   d_exp_o_f32[N_tok, H_dim]  f32   (shared-expert output)
 * Requires one host sync per call to read counts. */
static void run_dit_moe_fp16_topk(hip_hy3d_runner *r, dit_block_gpu *blk,
                                   void *d_input_f16, void *d_output_f32,
                                   int N_tok, void *d_moe_scratch) {
    hy3d_ops *ops = &r->ops;
    hipStream_t stream = r->stream;
    const int H_dim = DIT_HIDDEN;
    const int ffn = DIT_FFN;
    const int E = DIT_N_EXPERTS;
    const size_t f16b = sizeof(uint16_t);
    const size_t f32b = sizeof(float);

    size_t off = 0;
    void *d_gate     = (char *)d_moe_scratch + off; off += (size_t)N_tok * E * f32b;
    void *d_weights  = (char *)d_moe_scratch + off; off += (size_t)N_tok * E * f32b;
    void *d_counts   = (char *)d_moe_scratch + off; off += (size_t)E * sizeof(int);
    void *d_perm     = (char *)d_moe_scratch + off; off += (size_t)E * N_tok * sizeof(int);
    void *d_packed_in= (char *)d_moe_scratch + off; off += (size_t)N_tok * H_dim * f16b;
    void *d_packed_h = (char *)d_moe_scratch + off; off += (size_t)N_tok * ffn   * f16b;
    void *d_packed_o = (char *)d_moe_scratch + off; off += (size_t)N_tok * H_dim * f16b;
    void *d_exp_o_f32= (char *)d_moe_scratch + off; /* [N_tok * H_dim] f32 */

    /* Step 1: Gate logits, fp16 input × fp16 weight → f32 output. */
    op_gemm_f16(ops, stream, d_gate, blk->moe_gate_w, d_input_f16,
                E, H_dim, N_tok);

    /* Step 2: GPU softmax + top-K → d_weights (only top-K entries non-zero). */
    op_moe_gate_softmax_topk(ops, stream, d_gate, d_weights,
                             N_tok, E, DIT_MOE_TOP_K);

    /* Step 3: Build per-expert dispatch (counts + permutations). */
    hipMemsetAsync(d_counts, 0, (size_t)E * sizeof(int), stream);
    op_moe_dispatch_build(ops, stream, d_weights, d_counts, d_perm, N_tok, E);

    /* Step 4: Sync to read counts. */
    int counts[16]; /* DIT_N_EXPERTS ≤ 16 */
    hipMemcpyAsync(counts, d_counts, (size_t)E * sizeof(int),
                   hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);

    /* Step 5: Zero accumulator. */
    hipMemsetAsync(d_output_f32, 0, (size_t)N_tok * H_dim * f32b, stream);

    /* Step 6: Per-expert gather → fc1+GELU → fc2 → scatter+weighted-add. */
    for (int e = 0; e < E; e++) {
        int cnt = counts[e];
        if (cnt == 0) continue;

        void *e_perm = (char *)d_perm + (size_t)e * N_tok * sizeof(int);

        op_moe_gather_f16(ops, stream, d_packed_in, d_input_f16, e_perm,
                          cnt, H_dim);

        op_gemm_f16_bias_gelu_f16d(ops, stream, d_packed_h,
                                    blk->moe_expert_fc1_w[e], d_packed_in,
                                    blk->moe_expert_fc1_b[e], ffn, H_dim, cnt);
        op_gemm_f16_bias_f16d(ops, stream, d_packed_o,
                               blk->moe_expert_fc2_w[e], d_packed_h,
                               blk->moe_expert_fc2_b[e], H_dim, ffn, cnt);

        op_moe_scatter_add_f16_to_f32(ops, stream, d_output_f32, d_packed_o,
                                       e_perm, d_weights, e, E, cnt, H_dim);
    }

    /* Step 7: Shared expert (always all tokens). */
    op_gemm_f16_bias_gelu_f16d(ops, stream, d_packed_h,
                               blk->moe_shared_fc1_w, d_input_f16,
                               blk->moe_shared_fc1_b, ffn, H_dim, N_tok);
    op_gemm_f16_bias(ops, stream, d_exp_o_f32,
                     blk->moe_shared_fc2_w, d_packed_h,
                     blk->moe_shared_fc2_b, H_dim, ffn, N_tok);
    op_add(ops, stream, d_output_f32, d_exp_o_f32, N_tok * H_dim);
}
#endif  /* HY3D_HIPBLASLT_ENABLED */

/* Stage 2 (FP16 path): residual stream stays in fp16 end-to-end.
 *
 * Mirrors run_dit_forward block-for-block but uses:
 *   - hipBLASLt fp16 GEMM with f32 accumulator (matches PyTorch fp16 Linear)
 *   - rms_norm_f16 / layernorm_f16 (f32 internal reduction)
 *   - flash_attn_sa_f16_wmma / cross_attn_f16_wmma (f32 softmax)
 *   - GELU exact in f32 (via op_gemm_f16_bias_gelu_f16d epilogue)
 *
 * MoE blocks (15..20) cast the residual f16->f32 at the boundary, run the
 * existing f32 MoE, then cast the result back. This is only ~30% of the
 * blocks and the cast cost is dwarfed by the 8-expert FCs themselves.
 *
 * Boundary casts: d_latents (f32) -> f16 at entry; d_output stays f32 since
 * the Euler step downstream consumes f32. */
#ifdef HY3D_HIPBLASLT_ENABLED
static void run_dit_forward_fp16(hip_hy3d_runner *r, void *d_latents,
                                  float timestep, void *d_context,
                                  void *d_output) {
    hy3d_ops *ops = &r->ops;
    hipStream_t stream = r->stream;
    static int trace = -1;
    if (trace < 0) {
        const char *e = getenv("HY3D_FP16_TRACE");
        trace = (e && *e && e[0] != '0') ? 1 : 0;
    }
    if (trace) { fprintf(stderr, "[fp16-trace] enter\n"); fflush(stderr); }
    struct timespec _t_blk_prev;

    const int N = DIT_INPUT_SIZE;
    const int C = DIT_IN_CHANNELS;
    const int H_dim = DIT_HIDDEN;
    const int heads = DIT_HEADS;
    const int hd = DIT_HEAD_DIM;
    const int ffn = DIT_FFN;
    const int ctx_len = DINO_SEQ_LEN;
    const int N1 = N + 1;
    const size_t f16b = sizeof(uint16_t);

    /* --- Scratch sizes (most buffers half the size of the f32 path) --- */
    size_t hidden_f16 = (size_t)N1 * H_dim * f16b;
    size_t qkv_f16    = (size_t)3 * N1 * H_dim * f16b;
    size_t mlp_f16    = (size_t)N1 * ffn * f16b;
    size_t cat_f16    = (size_t)N1 * 2 * H_dim * f16b;
    size_t kv_f16     = (size_t)ctx_len * H_dim * f16b;
    size_t ctx_f16    = (size_t)ctx_len * DIT_CONTEXT_DIM * f16b;
    size_t latents_f16 = (size_t)N * C * f16b;
    /* MoE scratch (still f32): d_gate + d_weights + d_exp_h + d_exp_o */
    size_t moe_scratch_sz = (size_t)N1 * (2 * DIT_N_EXPERTS + ffn + H_dim) * sizeof(float);
    /* slot 1 hosts qkv/mlp(f16) OR moe_scratch(f32), whichever is bigger. */
    size_t shared1 = qkv_f16;
    if (mlp_f16 > shared1) shared1 = mlp_f16;
    if (moe_scratch_sz > shared1) shared1 = moe_scratch_sz;
    /* slot 3: temb + tmlp + cross_Q + cat_buf + ca_K + ca_V (all f16) */
    size_t buf3 = (size_t)(H_dim + ffn) * f16b
                + hidden_f16
                + cat_f16
                + 2 * kv_f16;
    /* slot 4: one-shot fp16 cast scratch (latents_f16 + context_f16). */
    size_t buf4 = latents_f16 + ctx_f16;
    /* slot 5/6: MoE f32 in/out (used for the 6 MoE blocks). */
    size_t moe_iof32 = (size_t)N1 * H_dim * sizeof(float);

    ensure_scratch(r, 0, hidden_f16);
    ensure_scratch(r, 1, shared1);
    ensure_scratch(r, 2, 2 * hidden_f16);
    ensure_scratch(r, 3, buf3);
    ensure_scratch(r, 4, buf4);
    ensure_scratch(r, 5, moe_iof32);
    ensure_scratch(r, 6, moe_iof32);
    /* GPU-resident skip stack (replaces CPU buffer + sync H2D/D2H per layer). */
    size_t skip_stack_sz = (size_t)(DIT_HALF_DEPTH + 1) * hidden_f16;
    ensure_scratch(r, 7, skip_stack_sz);

    void *d_hidden  = r->scratch[0];
    void *d_qkv     = r->scratch[1];          /* aliases d_mlp & moe scratch */
    void *d_mlp     = r->scratch[1];
    void *d_moe_scratch = r->scratch[1];
    void *d_attn    = r->scratch[2];
    void *d_normed  = (char *)r->scratch[2] + hidden_f16;
    void *d_temb    = r->scratch[3];
    void *d_tmlp    = (char *)d_temb    + (size_t)H_dim * f16b;
    void *d_cross_Q = (char *)d_tmlp    + (size_t)ffn   * f16b;
    void *d_cat_buf = (char *)d_cross_Q + hidden_f16;
    void *d_ca_K    = (char *)d_cat_buf + cat_f16;
    void *d_ca_V    = (char *)d_ca_K    + kv_f16;
    void *d_latent_f16  = r->scratch[4];
    void *d_ctx_f16     = (char *)d_latent_f16 + latents_f16;
    void *d_moe_in_f32  = r->scratch[5];
    void *d_moe_out_f32 = r->scratch[6];

    /* GPU-resident skip stack (no host round-trips). */
    size_t skip_entry_sz = hidden_f16;
    void *d_skip_stack = r->scratch[7];

    if (trace) { hipStreamSynchronize(stream); hipError_t e=hipGetLastError(); fprintf(stderr, "[fp16-trace] scratch ok err=%d\n",(int)e); fflush(stderr); }

    /* 0a. Cast input boundaries to fp16 once. */
    op_cast_f32_to_f16(ops, stream, d_latent_f16, d_latents, N * C);
    if (trace) { hipStreamSynchronize(stream); hipError_t e=hipGetLastError(); fprintf(stderr, "[fp16-trace] cast latents err=%d d_latent_f16=%p d_latents=%p\n",(int)e, d_latent_f16, d_latents); fflush(stderr); }
    op_cast_f32_to_f16(ops, stream, d_ctx_f16, d_context, ctx_len * DIT_CONTEXT_DIM);
    if (trace) { hipStreamSynchronize(stream); hipError_t e=hipGetLastError(); fprintf(stderr, "[fp16-trace] cast ctx err=%d d_ctx_f16=%p d_context=%p\n",(int)e, d_ctx_f16, d_context); fflush(stderr); }

    /* Zero-bias workspace: hipBLASLt _f16d epilogues require a bias pointer.
     * Allocate once on first use, sized to the largest N we'll see (3*H_dim). */
    static void *d_zero_bias = NULL;
    if (!d_zero_bias) {
        size_t zb_sz = (size_t)(3 * H_dim) * sizeof(float);
        hipMalloc(&d_zero_bias, zb_sz);
        hipMemsetAsync(d_zero_bias, 0, zb_sz, stream);
        hipStreamSynchronize(stream);
    }

    /* 1. x_embedder: [N,C=64] -> [N, H_dim].  K=64 is too small for hipBLASLt
     *    F16-D path on RDNA4 (illegal memory access); use the existing
     *    tiled GEMM (F16 weight, F32 X/Y) on the F32 source latents, then
     *    cast to F16. */
    {
        float *xemb_f32 = (float *)d_qkv;  /* repurpose qkv slot */
        op_gemm(ops, stream, xemb_f32, r->dit_x_emb_w, d_latents,
                r->dit_x_emb_b, H_dim, C, N);
        op_cast_f32_to_f16(ops, stream, d_normed, xemb_f32, N * H_dim);
    }

    if (trace) { hipStreamSynchronize(stream); hipError_t e=hipGetLastError(); fprintf(stderr, "[fp16-trace] x_embed ok err=%d\n",(int)e); fflush(stderr); }

    /* 2. Timestep embedding: small, do it in f32 then cast to f16.  We
     *    repurpose d_qkv as a temporary f32 staging area. */
    {
        float *t_f32  = (float *)d_qkv;
        float *t_mlp  = t_f32 + H_dim;
        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] pre-tembed err=%d t_f32=%p\n", (int)e, t_f32); fflush(stderr); }
        op_timestep_embed(ops, stream, t_f32, timestep, H_dim);
        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] tembed err=%d\n", (int)e); fflush(stderr); }
        op_gemm(ops, stream, t_mlp, r->dit_t_mlp0_w, t_f32, r->dit_t_mlp0_b,
                ffn, H_dim, 1);
        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] tmlp0 err=%d\n", (int)e); fflush(stderr); }
        op_gelu_exact(ops, stream, t_mlp, ffn);
        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] tgelu err=%d\n", (int)e); fflush(stderr); }
        /* Reuse t_f32 region for the second linear's f32 output */
        op_gemm(ops, stream, t_f32, r->dit_t_mlp2_w, t_mlp, r->dit_t_mlp2_b,
                H_dim, ffn, 1);
        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] tmlp2 err=%d\n", (int)e); fflush(stderr); }
        op_cast_f32_to_f16(ops, stream, d_temb, t_f32, H_dim);
        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] temb_cast err=%d\n", (int)e); fflush(stderr); }
    }

    /* 3. Prepend timestep token -> d_hidden [N1, H_dim] f16 */
    op_concat_first_f16(ops, stream, d_hidden, d_temb, d_normed, N, H_dim);
    if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] concat_first err=%d\n", (int)e); fflush(stderr); }

    int skip_sp = 0;

    if (trace) {
        hipStreamSynchronize(stream);
        hipError_t err = hipGetLastError();
        clock_gettime(CLOCK_MONOTONIC, &_t_blk_prev);
        fprintf(stderr, "[fp16-trace] preamble done, last err=%d entering block loop\n", (int)err);
        fflush(stderr);
    }

    for (int bi = 0; bi < DIT_DEPTH; bi++) {
        dit_block_gpu *blk = &r->dit_blocks[bi];

        /* Skip connection (blocks 11..20): pop, concat with d_hidden, project, LN */
        if (blk->use_skip && skip_sp > 0) {
            skip_sp--;
            void *d_skip = (char *)d_skip_stack + (size_t)skip_sp * skip_entry_sz;
            op_concat_last_dim_f16(ops, stream, d_cat_buf, d_skip, d_hidden, N1, H_dim);
            op_gemm_f16_bias_f16d(ops, stream, d_hidden, blk->skip_linear_w, d_cat_buf,
                                  blk->skip_linear_b, H_dim, 2 * H_dim, N1);
            op_layernorm_f16(ops, stream, d_hidden, d_hidden,
                             blk->skip_norm_w, blk->skip_norm_b, N1, H_dim);
        }

        /* Push hidden onto skip stack (blocks 0..DIT_HALF_DEPTH) */
        if (bi <= DIT_HALF_DEPTH) {
            void *d_skip = (char *)d_skip_stack + (size_t)skip_sp * skip_entry_sz;
            hipMemcpyAsync(d_skip, d_hidden, skip_entry_sz, hipMemcpyDeviceToDevice, stream);
            skip_sp++;
        }

        /* === Self-attention === */
        op_layernorm_f16(ops, stream, d_normed, d_hidden, blk->norm1_w, blk->norm1_b, N1, H_dim);
        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] b%d ln1 err=%d\n", bi, (int)e); fflush(stderr); }

        /* Fused QKV GEMM (no bias in HunYuanDiT for SA QKV). */
        op_gemm_f16_bias_f16d(ops, stream, d_qkv, blk->sa_qkv_w, d_normed,
                              d_zero_bias, 3 * H_dim, H_dim, N1);
        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] b%d sa_qkv err=%d\n", bi, (int)e); fflush(stderr); }

        void *d_Q = d_attn;
        void *d_K = d_normed;
        /* d_V borrows scratch[5] (moe_in_f32, 32MB) — SA finishes before MoE
         * input is written, and non-MoE blocks don't touch scratch[5]. */
        void *d_V = r->scratch[5];
        op_split_qkv_f16(ops, stream, d_Q, d_K, d_V, d_qkv, N1, heads, hd);
        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] b%d split_qkv err=%d\n", bi, (int)e); fflush(stderr); }

        if (blk->sa_q_norm_w)
            op_rms_norm_f16(ops, stream, d_Q, blk->sa_q_norm_w, N1, heads, hd, H_dim);
        if (blk->sa_k_norm_w)
            op_rms_norm_f16(ops, stream, d_K, blk->sa_k_norm_w, N1, heads, hd, H_dim);
        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] b%d sa_qknorm err=%d\n", bi, (int)e); fflush(stderr); }

        /* WMMA flash-attention (f16 in/out, f32 softmax). */
        if (op_self_attn_f16(ops, stream, d_attn, d_Q, d_K, d_V,
                             N1, H_dim, heads, hd) != 0) {
            fprintf(stderr, "HY3D fp16: WMMA self-attn unavailable\n");
            return;
        }

        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] b%d sa_attn err=%d\n", bi, (int)e); fflush(stderr); }
        op_gemm_f16_bias_f16d(ops, stream, d_normed, blk->sa_out_w, d_attn,
                              blk->sa_out_b, H_dim, H_dim, N1);
        op_add_f16(ops, stream, d_hidden, d_normed, N1 * H_dim);
        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] b%d sa_done err=%d\n", bi, (int)e); fflush(stderr); }

        /* === Cross-attention === */
        op_layernorm_f16(ops, stream, d_normed, d_hidden, blk->norm2_w, blk->norm2_b, N1, H_dim);

        op_gemm_f16_bias_f16d(ops, stream, d_cross_Q, blk->ca_q_w, d_normed,
                              d_zero_bias, H_dim, H_dim, N1);
        if (blk->ca_q_norm_w)
            op_rms_norm_f16(ops, stream, d_cross_Q, blk->ca_q_norm_w, N1, heads, hd, H_dim);

        /* Fused KV from context. */
        op_gemm_f16_bias_f16d(ops, stream, d_qkv, blk->ca_kv_w, d_ctx_f16,
                              d_zero_bias, 2 * H_dim, DIT_CONTEXT_DIM, ctx_len);
        op_split_kv_f16(ops, stream, d_ca_K, d_ca_V, d_qkv, ctx_len, heads, hd);

        if (blk->ca_k_norm_w)
            op_rms_norm_f16(ops, stream, d_ca_K, blk->ca_k_norm_w,
                            ctx_len, heads, hd, H_dim);
        void *use_ca_K = d_ca_K;
        void *use_ca_V = d_ca_V;

        if (op_cross_attn_f16(ops, stream, d_attn, d_cross_Q, use_ca_K, use_ca_V,
                              N1, ctx_len, H_dim, heads, hd) != 0) {
            fprintf(stderr, "HY3D fp16: WMMA cross-attn unavailable\n");
            return;
        }

        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] b%d ca_attn err=%d\n", bi, (int)e); fflush(stderr); }
        op_gemm_f16_bias_f16d(ops, stream, d_normed, blk->ca_out_w, d_attn,
                              blk->ca_out_b, H_dim, H_dim, N1);
        op_add_f16(ops, stream, d_hidden, d_normed, N1 * H_dim);
        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] b%d ca_done err=%d\n", bi, (int)e); fflush(stderr); }

        /* === MLP / MoE === */
        if (blk->norm3_w) {
            op_layernorm_f16(ops, stream, d_normed, d_hidden, blk->norm3_w, blk->norm3_b, N1, H_dim);
        } else {
            hipMemcpyAsync(d_normed, d_hidden, hidden_f16, hipMemcpyDeviceToDevice, stream);
        }

        if (blk->use_moe) {
            /* FP16 MoE: fp16 input direct, f32 output, then cast back.
             * HY3D_MOE_TOPK=1 enables true top-K dispatch (4× less FC work). */
            static int topk_mode = -1;
            if (topk_mode < 0) {
                const char *e = getenv("HY3D_MOE_TOPK");
                /* default on; HY3D_MOE_TOPK=0 disables. */
                topk_mode = (e && *e && e[0] == '0') ? 0 : 1;
            }
            if (topk_mode) {
                run_dit_moe_fp16_topk(r, blk, d_normed, d_moe_out_f32, N1, d_moe_scratch);
            } else {
                run_dit_moe_fp16(r, blk, d_normed, d_moe_out_f32, N1, d_moe_scratch);
            }
            op_cast_f32_to_f16(ops, stream, d_normed, d_moe_out_f32, N1 * H_dim);
            op_add_f16(ops, stream, d_hidden, d_normed, N1 * H_dim);
        } else {
            /* fc1 with fused exact-GELU + f16d output, then fc2 + bias. */
            op_gemm_f16_bias_gelu_f16d(ops, stream, d_mlp, blk->mlp_fc1_w, d_normed,
                                       blk->mlp_fc1_b, ffn, H_dim, N1);
            if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] b%d mlp_fc1 err=%d\n", bi, (int)e); fflush(stderr); }
            op_gemm_f16_bias_f16d(ops, stream, d_normed, blk->mlp_fc2_w, d_mlp,
                                  blk->mlp_fc2_b, H_dim, ffn, N1);
            if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] b%d mlp_fc2 err=%d\n", bi, (int)e); fflush(stderr); }
            op_add_f16(ops, stream, d_hidden, d_normed, N1 * H_dim);
            if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] b%d mlp_done err=%d\n", bi, (int)e); fflush(stderr); }
        }

        if (trace) {
            hipStreamSynchronize(stream);
            struct timespec t_now;
            clock_gettime(CLOCK_MONOTONIC, &t_now);
            double ms = (t_now.tv_sec - _t_blk_prev.tv_sec) * 1000.0
                      + (t_now.tv_nsec - _t_blk_prev.tv_nsec) / 1e6;
            fprintf(stderr, "[fp16-trace] block %2d %s%s ms=%.1f\n",
                    bi, blk->use_skip ? "skip+" : "     ",
                    blk->use_moe ? "MoE" : "MLP", ms);
            _t_blk_prev = t_now;
        }

        /* Per-block hidden dump kept for cross-comparison; emit f16 → f32 view. */
        if (getenv("HY3D_DUMP_DIR") && (bi == 0 || bi == 5 || bi == 10 || bi == 11 ||
            bi == 14 || bi == 15 || bi == 20)) {
            char fname[64];
            snprintf(fname, sizeof(fname), "hip_dit_block_fp16_%d.npy", bi);
            /* convert to f32 into d_moe_out_f32 (free at this point), then dump */
            op_cast_f16_to_f32(ops, stream, d_moe_out_f32, d_hidden, N1 * H_dim);
            hy3d_dbg_dump_npy(stream, d_moe_out_f32, N1, H_dim, fname);
        }
        if (trace) { hipStreamSynchronize(stream); hipError_t e = hipGetLastError(); fprintf(stderr, "[fp16-trace] b%d end err=%d\n", bi, (int)e); fflush(stderr); }
    }

    /* 5. Final layer: strip timestep, LN, Linear -> f32 [N, C] */
    op_strip_first_f16(ops, stream, d_normed, d_hidden, N1, H_dim);

    void *d_ln_out = d_attn;
    op_layernorm_f16(ops, stream, d_ln_out, d_normed,
                     r->dit_final_ln_w, r->dit_final_ln_b, N, H_dim);

    /* Final linear writes f32 directly (consumed by Euler step). */
    op_gemm_f16_bias(ops, stream, d_output, r->dit_final_linear_w, d_ln_out,
                     r->dit_final_linear_b, C, H_dim, N);
}
#endif  /* HY3D_HIPBLASLT_ENABLED */

/* Stage 2: Single DiT forward pass */
static void run_dit_forward(hip_hy3d_runner *r, void *d_latents,
                             float timestep, void *d_context,
                             void *d_output) {
#ifdef HY3D_HIPBLASLT_ENABLED
    {
        static int fp16_mode = -1;
        if (fp16_mode < 0) {
            const char *e = getenv("HY3D_DIT_FP16");
            /* default on; HY3D_DIT_FP16=0 disables. */
            fp16_mode = (e && *e && e[0] == '0') ? 0 : 1;
        }
        if (fp16_mode) {
            run_dit_forward_fp16(r, d_latents, timestep, d_context, d_output);
            return;
        }
    }
#endif
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
    /* d_gate + d_weights (both [N1, DIT_N_EXPERTS]) + d_exp_h [N1, ffn] + d_exp_o [N1, H_dim]. */
    size_t moe_scratch_sz = (size_t)N1 * (2 * DIT_N_EXPERTS + ffn + H_dim) * sizeof(float);
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
    if (r->verbose > 1) {
        fprintf(stderr, "  x_emb_w=%p x_emb_b=%p d_latents=%p d_normed=%p\n",
                r->dit_x_emb_w, r->dit_x_emb_b, d_latents, d_normed);
        hipDeviceSynchronize();
        float chk[4];
        hipMemcpy(chk, d_latents, sizeof(chk), hipMemcpyDeviceToHost);
        fprintf(stderr, "  latents_in[0:4]: %.6f %.6f %.6f %.6f\n", chk[0], chk[1], chk[2], chk[3]);
        hipMemcpy(chk, r->dit_x_emb_w, sizeof(chk), hipMemcpyDeviceToHost);
        fprintf(stderr, "  x_emb_w[0:4]: %.6f %.6f %.6f %.6f\n", chk[0], chk[1], chk[2], chk[3]);
    }
    op_gemm(ops, stream, d_normed, r->dit_x_emb_w, d_latents, r->dit_x_emb_b,
            H_dim, C, N);

    /* 2. Fix 4: Timestep embedding with GELU (not SiLU) */
    op_timestep_embed(ops, stream, d_temb, timestep, H_dim);
    op_gemm(ops, stream, d_tmlp, r->dit_t_mlp0_w, d_temb, r->dit_t_mlp0_b,
            ffn, H_dim, 1);
    op_gelu_exact(ops, stream, d_tmlp, ffn);
    op_gemm(ops, stream, d_temb, r->dit_t_mlp2_w, d_tmlp, r->dit_t_mlp2_b,
            H_dim, ffn, 1);

    /* Debug: check timestep embedding */
    if (r->verbose > 1) {
        hipDeviceSynchronize();
        float chk[8];
        hipMemcpy(chk, d_temb, sizeof(chk), hipMemcpyDeviceToHost);
        int nc = 0; for (int i = 0; i < 8; i++) if (chk[i] != chk[i]) nc++;
        fprintf(stderr, "  temb[0:8]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f nan=%d\n",
                chk[0], chk[1], chk[2], chk[3], chk[4], chk[5], chk[6], chk[7], nc);
        /* Check latent embedding too */
        hipMemcpy(chk, d_normed, sizeof(chk), hipMemcpyDeviceToHost);
        nc = 0; for (int i = 0; i < 8; i++) if (chk[i] != chk[i]) nc++;
        fprintf(stderr, "  x_emb[0:8]: %.4f %.4f %.4f %.4f nan=%d\n", chk[0], chk[1], chk[2], chk[3], nc);
    }

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

        /* Debug: check after LN for block 0 */
        if (bi == 0 && r->verbose > 1) {
            hipDeviceSynchronize(); float chk[4];
            hipMemcpy(chk, d_normed, sizeof(chk), hipMemcpyDeviceToHost);
            fprintf(stderr, "    b0 after_LN1: %.6f %.6f %.6f %.6f\n", chk[0], chk[1], chk[2], chk[3]);
        }

        /* Fused QKV GEMM */
        op_gemm(ops, stream, d_qkv, blk->sa_qkv_w, d_normed, NULL, 3 * H_dim, H_dim, N1);

        if (bi == 0 && r->verbose > 1) {
            hipDeviceSynchronize(); float chk[4];
            hipMemcpy(chk, d_qkv, sizeof(chk), hipMemcpyDeviceToHost);
            fprintf(stderr, "    b0 after_QKV: %.6f %.6f %.6f %.6f\n", chk[0], chk[1], chk[2], chk[3]);
        }

        /* Split interleaved QKV */
        void *d_Q = d_attn;    /* scratch[2] */
        void *d_K = d_normed;  /* scratch[2] offset */
        void *d_V = (char *)d_ca_V + ca_kv_sz; /* after ca_V in scratch[3] */
        op_split_qkv(ops, stream, d_Q, d_K, d_V, d_qkv, N1, heads, hd);

        if (bi == 0 && r->verbose > 1) {
            hipDeviceSynchronize(); float chk[4];
            hipMemcpy(chk, d_Q, sizeof(chk), hipMemcpyDeviceToHost);
            fprintf(stderr, "    b0 Q[0:4]:  %.6f %.6f %.6f %.6f\n", chk[0], chk[1], chk[2], chk[3]);
            hipMemcpy(chk, d_K, sizeof(chk), hipMemcpyDeviceToHost);
            fprintf(stderr, "    b0 K[0:4]:  %.6f %.6f %.6f %.6f\n", chk[0], chk[1], chk[2], chk[3]);
        }

        /* QK RMSNorm */
        if (blk->sa_q_norm_w)
            op_rms_norm(ops, stream, d_Q, blk->sa_q_norm_w, N1, heads, hd, H_dim);
        if (blk->sa_k_norm_w)
            op_rms_norm(ops, stream, d_K, blk->sa_k_norm_w, N1, heads, hd, H_dim);

        if (bi == 0 && r->verbose > 1) {
            hipDeviceSynchronize(); float chk[4];
            hipMemcpy(chk, d_Q, sizeof(chk), hipMemcpyDeviceToHost);
            fprintf(stderr, "    b0 Q_norm:  %.6f %.6f %.6f %.6f\n", chk[0], chk[1], chk[2], chk[3]);
        }

        op_self_attn(ops, stream, d_attn, d_Q, d_K, d_V, N1, H_dim, heads, hd);

        if (bi == 0 && r->verbose > 1) {
            hipDeviceSynchronize(); float chk[4];
            hipMemcpy(chk, d_attn, sizeof(chk), hipMemcpyDeviceToHost);
            fprintf(stderr, "    b0 attn_out: %.6f %.6f %.6f %.6f\n", chk[0], chk[1], chk[2], chk[3]);
        }

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
            /* d_mlp aliases d_moe_scratch (both = scratch[1]); using it as
             * output would corrupt MoE working buffers. Use d_attn (scratch[2]
             * offset 0) instead — cross-attn output was already consumed by
             * op_add above, so d_attn is free and does not overlap d_normed
             * (at scratch[2] + N1*H_dim). */
            void *d_moe_out = d_attn;
            run_dit_moe(r, blk, d_normed, d_moe_out, N1, d_moe_scratch);
            op_add(ops, stream, d_hidden, d_moe_out, N1 * H_dim);
        } else {
            op_gemm(ops, stream, d_mlp, blk->mlp_fc1_w, d_normed, blk->mlp_fc1_b, ffn, H_dim, N1);
            op_gelu_exact(ops, stream, d_mlp, N1 * ffn);
            op_gemm(ops, stream, d_normed, blk->mlp_fc2_w, d_mlp, blk->mlp_fc2_b, H_dim, ffn, N1);
            op_add(ops, stream, d_hidden, d_normed, N1 * H_dim);
        }

        /* Per-block hidden dump (for error localization vs PyTorch) */
        if (bi == 0 || bi == 5 || bi == 10 || bi == 11 ||
            bi == 14 || bi == 15 || bi == 20) {
            char fname[64];
            snprintf(fname, sizeof(fname), "hip_dit_block_%d.npy", bi);
            hy3d_dbg_dump_npy(stream, d_hidden, N1, H_dim, fname);
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
    op_gelu_exact(ops, stream, d_mlph, N * MLP);
    op_gemm(ops, stream, d_mlpo, b->mlp_proj_w, d_mlph, b->mlp_proj_b, W, MLP, N);

    /* Output = res1 + mlp_out */
    hipMemcpyAsync(d_out, d_res1, (size_t)N * W * sizeof(float), hipMemcpyDeviceToDevice, stream);
    op_add(ops, stream, d_out, d_mlpo, N * W);
}

/* fp16 variant of run_vae_block: input/output f16, hipBLASLt fp16 GEMMs,
 * WMMA self-attn, f32 reductions inside layernorm/qk_norm.  Weights must
 * already be f16 (use_f32_gemm=0).  Bias buffers stay f32. */
static void run_vae_block_fp16(hip_hy3d_runner *r, vae_block_gpu *b,
                               void *d_in_f16, void *d_out_f16,
                               void *d_scratch /* f16 scratch */,
                               void *d_qk_f32   /* f32 buffer for qk_norm cast */,
                               void *d_zero_bias_f32 /* f32 [3*W] zeros */,
                               void *d_attn_f32 /* 3*N*W f32 (Q|K|V) for fallback attn */) {
    hy3d_ops *ops = &r->ops;
    hipStream_t stream = r->stream;
    const int N = VAE_NUM_LATENTS;
    const int W = VAE_WIDTH;
    const int H = VAE_HEADS;
    const int HD = VAE_HEAD_DIM;
    const int MLP = 4 * W;

    size_t off = 0;
    void *d_ln1   = (char *)d_scratch + off; off += (size_t)N * W * sizeof(uint16_t);
    void *d_qkv   = (char *)d_scratch + off; off += (size_t)N * 3 * W * sizeof(uint16_t);
    void *d_Q     = (char *)d_scratch + off; off += (size_t)N * W * sizeof(uint16_t);
    void *d_K     = (char *)d_scratch + off; off += (size_t)N * W * sizeof(uint16_t);
    void *d_V     = (char *)d_scratch + off; off += (size_t)N * W * sizeof(uint16_t);
    void *d_aout  = (char *)d_scratch + off; off += (size_t)N * W * sizeof(uint16_t);
    void *d_proj  = (char *)d_scratch + off; off += (size_t)N * W * sizeof(uint16_t);
    void *d_res1  = (char *)d_scratch + off; off += (size_t)N * W * sizeof(uint16_t);
    void *d_ln2   = (char *)d_scratch + off; off += (size_t)N * W * sizeof(uint16_t);
    void *d_mlph  = (char *)d_scratch + off; off += (size_t)N * MLP * sizeof(uint16_t);
    void *d_mlpo  = (char *)d_scratch + off;

    op_layernorm_f16(ops, stream, d_ln1, d_in_f16, b->ln1_w, b->ln1_b, N, W);

    /* Fused QKV via hipBLASLt fp16 (bias_f16d epilogue rejects NULL — pass
     * a static zero-bias). */
    op_gemm_f16_bias_f16d(ops, stream, d_qkv, b->qkv_w, d_ln1,
                          d_zero_bias_f32, 3 * W, W, N);
    op_split_qkv_f16(ops, stream, d_Q, d_K, d_V, d_qkv, N, H, HD);

    if (b->use_qk_norm) {
        /* Prefer f16 in-place (halves BW); fall back to cast-around f32. */
        if (op_qk_layernorm_f16(ops, stream, d_Q, b->q_norm_w, b->q_norm_b,
                                N, H, HD, W) != 0) {
            op_cast_f16_to_f32(ops, stream, d_qk_f32, d_Q, N * W);
            op_qk_layernorm(ops, stream, d_qk_f32, b->q_norm_w, b->q_norm_b, N, H, HD, W);
            op_cast_f32_to_f16(ops, stream, d_Q, d_qk_f32, N * W);
        }
        if (op_qk_layernorm_f16(ops, stream, d_K, b->k_norm_w, b->k_norm_b,
                                N, H, HD, W) != 0) {
            op_cast_f16_to_f32(ops, stream, d_qk_f32, d_K, N * W);
            op_qk_layernorm(ops, stream, d_qk_f32, b->k_norm_w, b->k_norm_b, N, H, HD, W);
            op_cast_f32_to_f16(ops, stream, d_K, d_qk_f32, N * W);
        }
    }

    /* Self-attn: try WMMA fp16 (head_dim=128 only).  ShapeVAE has hd=64
     * so this always falls back; keep the conditional for portability. */
    int rc = op_self_attn_f16(ops, stream, d_aout, d_Q, d_K, d_V, N, W, H, HD);
    if (rc != 0) {
        /* Fallback: cast Q/K/V f16->f32, run f32 scalar self-attn, cast
         * aout f32->f16.  d_attn_f32 is 4*N*W f32: [Q|K|V|out]. */
        size_t slot = (size_t)N * W * sizeof(float);
        void *d_Qf = d_attn_f32;
        void *d_Kf = (char *)d_attn_f32 + slot;
        void *d_Vf = (char *)d_attn_f32 + 2 * slot;
        void *d_Of = (char *)d_attn_f32 + 3 * slot;
        op_cast_f16_to_f32(ops, stream, d_Qf, d_Q, N * W);
        op_cast_f16_to_f32(ops, stream, d_Kf, d_K, N * W);
        op_cast_f16_to_f32(ops, stream, d_Vf, d_V, N * W);
        op_self_attn(ops, stream, d_Of, d_Qf, d_Kf, d_Vf, N, W, H, HD);
        op_cast_f32_to_f16(ops, stream, d_aout, d_Of, N * W);
    }

    /* Output projection: f16 D for chaining (has bias). */
    op_gemm_f16_bias_f16d(ops, stream, d_proj, b->proj_w, d_aout, b->proj_b, W, W, N);

    /* res1 = in + proj */
    hipMemcpyAsync(d_res1, d_in_f16, (size_t)N * W * sizeof(uint16_t),
                   hipMemcpyDeviceToDevice, stream);
    op_add_f16(ops, stream, d_res1, d_proj, N * W);

    op_layernorm_f16(ops, stream, d_ln2, d_res1, b->ln2_w, b->ln2_b, N, W);

    /* MLP fc + GELU fused into f16 D. */
    op_gemm_f16_bias_gelu_f16d(ops, stream, d_mlph, b->mlp_fc_w, d_ln2,
                               b->mlp_fc_b, MLP, W, N);

    /* MLP proj. */
    op_gemm_f16_bias_f16d(ops, stream, d_mlpo, b->mlp_proj_w, d_mlph,
                          b->mlp_proj_b, W, MLP, N);

    /* out = res1 + mlp_out */
    hipMemcpyAsync(d_out_f16, d_res1, (size_t)N * W * sizeof(uint16_t),
                   hipMemcpyDeviceToDevice, stream);
    op_add_f16(ops, stream, d_out_f16, d_mlpo, N * W);
}

/* ------------------------------------------------------------------------ */
/* Hierarchical volume decoding helpers (CPU-side mask + dilate)             */
/* ------------------------------------------------------------------------ */

/* Mark voxels where (a) a sign flip occurs vs any of the 6 axis neighbors,
 * or (b) |val| < 0.95.  Matches PyTorch extract_near_surface_volume_fn +
 * the |val|<0.95 clause from HierarchicalVolumeDecoding. */
static void hy3d_active_mask_3d(uint8_t *mask, const float *val, int R) {
    int RR = R * R;
#define IDX(x,y,z) ((x)*RR + (y)*R + (z))
    for (int x = 0; x < R; x++) {
        int xm = x > 0 ? x - 1 : 0;
        int xp = x < R - 1 ? x + 1 : R - 1;
        for (int y = 0; y < R; y++) {
            int ym = y > 0 ? y - 1 : 0;
            int yp = y < R - 1 ? y + 1 : R - 1;
            for (int z = 0; z < R; z++) {
                int zm = z > 0 ? z - 1 : 0;
                int zp = z < R - 1 ? z + 1 : R - 1;
                float v = val[IDX(x,y,z)];
                int sv = (v > 0.0f) - (v < 0.0f);
                int hit = 0;
                if (fabsf(v) < 0.95f) hit = 1;
                if (!hit) {
                    float nb[6] = {
                        val[IDX(xm,y,z)], val[IDX(xp,y,z)],
                        val[IDX(x,ym,z)], val[IDX(x,yp,z)],
                        val[IDX(x,y,zm)], val[IDX(x,y,zp)],
                    };
                    for (int k = 0; k < 6 && !hit; k++) {
                        int sn = (nb[k] > 0.0f) - (nb[k] < 0.0f);
                        if (sn != sv) hit = 1;
                    }
                }
                mask[IDX(x,y,z)] = (uint8_t)hit;
            }
        }
    }
#undef IDX
}

/* In-place 3x3x3 dilate (uint8 mask). */
static void hy3d_dilate_3x3x3(uint8_t *mask, int R) {
    int RR = R * R;
    size_t total = (size_t)R * R * R;
    uint8_t *tmp = (uint8_t *)malloc(total);
    memcpy(tmp, mask, total);
#define IDX(x,y,z) ((x)*RR + (y)*R + (z))
    for (int x = 0; x < R; x++) {
        int x0 = x > 0 ? x - 1 : 0, x1 = x < R - 1 ? x + 1 : R - 1;
        for (int y = 0; y < R; y++) {
            int y0 = y > 0 ? y - 1 : 0, y1 = y < R - 1 ? y + 1 : R - 1;
            for (int z = 0; z < R; z++) {
                int z0 = z > 0 ? z - 1 : 0, z1 = z < R - 1 ? z + 1 : R - 1;
                int hit = 0;
                for (int dx = x0; dx <= x1 && !hit; dx++)
                    for (int dy = y0; dy <= y1 && !hit; dy++)
                        for (int dz = z0; dz <= z1 && !hit; dz++)
                            if (tmp[IDX(dx,dy,dz)]) hit = 1;
                mask[IDX(x,y,z)] = (uint8_t)hit;
            }
        }
    }
#undef IDX
    free(tmp);
}

/* Geo-query context: bundles GPU scratch + model state for one shapevae run. */
typedef struct {
    hy3d_ops *ops;
    hipStream_t stream;
    void *d_coords, *d_fourier, *d_query_proj, *d_sdf_out, *d_geo_scratch;
    void *d_g_K, *d_g_V;
    /* Optional fp16 KV (NULL when fp16-attn path disabled). */
    void *d_g_K_f16, *d_g_V_f16;
    /* Optional fp16 batch scratch (NULL when fp16-attn path disabled). */
    void *d_Q_f16, *d_aout_f16;
    /* Optional fp16 batch scratch for VAE decoder fp16 path
     * (d_g_in_f16: ln1 src; d_g_qkv_f16: Q after proj; d_g_proj_f16: c_proj out;
     *  d_g_res_f16: residual; d_g_mlph_f16: MLP hidden 4*W; d_g_mlpo_f16: MLP out;
     *  d_g_post_f16: post-residual; d_g_lnpost_f16: post-LN). */
    void *d_g_ln1_f16, *d_g_proj_f16, *d_g_res_f16,
         *d_g_ln3_f16, *d_g_mlph_f16, *d_g_mlpo_f16,
         *d_g_post_f16, *d_g_lnpost_f16, *d_g_qknorm_f32;
    void *d_g_zero_bias_f32;   /* [W] f32 zeros for c_q (no-bias GEMM) */
    int use_dec_f16;
    int batch_size;
    int use_attn_f16;
    vae_geo_decoder_gpu *g;
    void *fourier_freqs;
    int N, W;
} hy3d_geo_ctx;

/* Run geo-query for n_pts host coords; write n_pts SDF values to sdf_out. */
static void hy3d_geo_query_run(hy3d_geo_ctx *c,
                               const float *coords, int n_pts, float *sdf_out) {
    if (n_pts <= 0) return;
    const int N = c->N;
    const int W = c->W;
    vae_geo_decoder_gpu *g = c->g;
    hy3d_ops *ops = c->ops;
    hipStream_t stream = c->stream;

    for (int start = 0; start < n_pts; start += c->batch_size) {
        int count = (start + c->batch_size <= n_pts) ? c->batch_size : (n_pts - start);

        hipMemcpyAsync(c->d_coords, coords + (size_t)start * 3,
                       (size_t)count * 3 * sizeof(float),
                       hipMemcpyHostToDevice, stream);

        op_fourier_embed(ops, stream, c->d_fourier, c->d_coords, c->fourier_freqs,
                         count, VAE_NUM_FREQS, VAE_FOURIER_DIM);

        op_gemm(ops, stream, c->d_query_proj, g->query_proj_w, c->d_fourier,
                g->query_proj_b, W, VAE_FOURIER_DIM, count);

        if (c->use_dec_f16) {
            /* fp16 path: residual stream from query_proj output to ln_post in
             * f16, hipBLASLt for c_q/c_proj/mlp_fc/mlp_proj.  Fourier+query_proj
             * stay f32 (small K=51) and final output stays f32 (N=1).  Q/K/V
             * cross-attn uses the existing op_cross_attn_f16 kernel. */
            void *d_in_f16   = c->d_g_ln1_f16;     /* count*W f16 — query_proj_f16 */
            void *d_ln1_f16  = c->d_Q_f16;          /* reuse Q scratch */
            void *d_Q_f16    = c->d_Q_f16;          /* same buffer (LN -> Q proj overwrites) */
            void *d_aout_f16 = c->d_aout_f16;
            void *d_proj_f16 = c->d_g_proj_f16;
            void *d_res_f16  = c->d_g_res_f16;
            void *d_ln3_f16  = c->d_g_ln3_f16;
            void *d_mlph_f16 = c->d_g_mlph_f16;
            void *d_mlpo_f16 = c->d_g_mlpo_f16;
            void *d_post_f16 = c->d_g_post_f16;
            void *d_lnpost_f16 = c->d_g_lnpost_f16;
            void *d_qk_f32   = c->d_g_qknorm_f32;

            op_cast_f32_to_f16(ops, stream, d_in_f16, c->d_query_proj, count * W);

            /* LN1 (f16 in, f16 out) — uses ln1_w/b (f32) */
            op_layernorm_f16(ops, stream, d_ln1_f16, d_in_f16,
                             g->ln1_w, g->ln1_b, count, W);
            /* Q proj (no bias).  bias_f16d epilogue requires a bias pointer —
             * pass the pre-zeroed buffer. */
            op_gemm_f16_bias_f16d(ops, stream, d_Q_f16, g->c_q_w, d_ln1_f16,
                                  c->d_g_zero_bias_f32, W, W, count);
            (void)d_qk_f32;
            if (g->use_qk_norm) {
                /* Prefer f16 in-place; fall back to cast-around f32. */
                if (op_qk_layernorm_f16(ops, stream, d_Q_f16,
                                        g->q_norm_w, g->q_norm_b,
                                        count, VAE_HEADS, VAE_HEAD_DIM, W) != 0) {
                    op_cast_f16_to_f32(ops, stream, c->d_g_qknorm_f32, d_Q_f16,
                                       count * W);
                    op_qk_layernorm(ops, stream, c->d_g_qknorm_f32,
                                    g->q_norm_w, g->q_norm_b,
                                    count, VAE_HEADS, VAE_HEAD_DIM, W);
                    op_cast_f32_to_f16(ops, stream, d_Q_f16, c->d_g_qknorm_f32,
                                       count * W);
                }
            }

            /* Cross-attn: Q[count, W], K/V[N, W] all f16. */
            int rc = op_cross_attn_f16(ops, stream, d_aout_f16,
                                       d_Q_f16, c->d_g_K_f16, c->d_g_V_f16,
                                       count, c->N, W, VAE_HEADS, VAE_HEAD_DIM);
            if (rc != 0) {
                fprintf(stderr, "HY3D: VAE decoder fp16 cross-attn failed\n");
                return;
            }

            /* c_proj + bias, f16 D. */
            op_gemm_f16_bias_f16d(ops, stream, d_proj_f16, g->c_proj_w,
                                  d_aout_f16, g->c_proj_b, W, W, count);
            /* res = query_proj_f16 + proj */
            hipMemcpyAsync(d_res_f16, d_in_f16,
                           (size_t)count * W * sizeof(uint16_t),
                           hipMemcpyDeviceToDevice, stream);
            op_add_f16(ops, stream, d_res_f16, d_proj_f16, count * W);

            /* LN3 + MLP */
            op_layernorm_f16(ops, stream, d_ln3_f16, d_res_f16,
                             g->ln3_w, g->ln3_b, count, W);
            op_gemm_f16_bias_gelu_f16d(ops, stream, d_mlph_f16, g->mlp_fc_w,
                                       d_ln3_f16, g->mlp_fc_b, 4 * W, W, count);
            op_gemm_f16_bias_f16d(ops, stream, d_mlpo_f16, g->mlp_proj_w,
                                  d_mlph_f16, g->mlp_proj_b, W, 4 * W, count);
            /* post = res + mlp_out */
            hipMemcpyAsync(d_post_f16, d_res_f16,
                           (size_t)count * W * sizeof(uint16_t),
                           hipMemcpyDeviceToDevice, stream);
            op_add_f16(ops, stream, d_post_f16, d_mlpo_f16, count * W);

            /* Optional post-LN (f16). */
            void *d_for_output_f32;
            if (g->ln_post_w) {
                op_layernorm_f16(ops, stream, d_lnpost_f16, d_post_f16,
                                 g->ln_post_w, g->ln_post_b, count, W);
                /* Cast to f32 for the final small-N output GEMM. */
                op_cast_f16_to_f32(ops, stream, c->d_g_qknorm_f32,
                                   d_lnpost_f16, count * W);
                d_for_output_f32 = c->d_g_qknorm_f32;
            } else {
                op_cast_f16_to_f32(ops, stream, c->d_g_qknorm_f32,
                                   d_post_f16, count * W);
                d_for_output_f32 = c->d_g_qknorm_f32;
            }

            /* Final output GEMM: N=1, keep scalar f32 path. */
            op_gemm(ops, stream, c->d_sdf_out, g->output_w, d_for_output_f32,
                    g->output_b, 1, W, count);

            hipMemcpyAsync(sdf_out + start, c->d_sdf_out,
                           (size_t)count * sizeof(float),
                           hipMemcpyDeviceToHost, stream);
            continue;
        }

        size_t geo_off = 0;
        void *d_g_ln1  = (char *)c->d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_Q    = (char *)c->d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_aout = (char *)c->d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_proj = (char *)c->d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_res  = (char *)c->d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_ln3  = (char *)c->d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_mlph = (char *)c->d_geo_scratch + geo_off; geo_off += (size_t)count * 4 * W * sizeof(float);
        void *d_g_mlpo = (char *)c->d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        void *d_g_post = (char *)c->d_geo_scratch + geo_off;

        op_layernorm(ops, stream, d_g_ln1, c->d_query_proj, g->ln1_w, g->ln1_b, count, W);
        op_gemm(ops, stream, d_g_Q, g->c_q_w, d_g_ln1, NULL, W, W, count);

        if (g->use_qk_norm) {
            op_qk_layernorm(ops, stream, d_g_Q, g->q_norm_w, g->q_norm_b,
                            count, VAE_HEADS, VAE_HEAD_DIM, W);
        }

        if (c->use_attn_f16) {
            /* Cast Q f32->f16 (count*W elems), run WMMA cross-attn,
             * cast output f16->f32 to keep downstream f32 path. */
            op_cast_f32_to_f16(ops, stream, c->d_Q_f16, d_g_Q, count * W);
            int rc = op_cross_attn_f16(ops, stream, c->d_aout_f16,
                                       c->d_Q_f16, c->d_g_K_f16, c->d_g_V_f16,
                                       count, N, W, VAE_HEADS, VAE_HEAD_DIM);
            if (rc == 0) {
                op_cast_f16_to_f32(ops, stream, d_g_aout, c->d_aout_f16, count * W);
            } else {
                /* WMMA path unavailable — fall back to scalar f32 attn. */
                op_cross_attn(ops, stream, d_g_aout, d_g_Q, c->d_g_K, c->d_g_V,
                              count, N, W, VAE_HEADS, VAE_HEAD_DIM);
            }
        } else {
            op_cross_attn(ops, stream, d_g_aout, d_g_Q, c->d_g_K, c->d_g_V,
                          count, N, W, VAE_HEADS, VAE_HEAD_DIM);
        }

        op_gemm(ops, stream, d_g_proj, g->c_proj_w, d_g_aout, g->c_proj_b, W, W, count);
        hipMemcpyAsync(d_g_res, c->d_query_proj, (size_t)count * W * sizeof(float),
                       hipMemcpyDeviceToDevice, stream);
        op_add(ops, stream, d_g_res, d_g_proj, count * W);

        op_layernorm(ops, stream, d_g_ln3, d_g_res, g->ln3_w, g->ln3_b, count, W);
        op_gemm(ops, stream, d_g_mlph, g->mlp_fc_w, d_g_ln3, g->mlp_fc_b, 4 * W, W, count);
        op_gelu_exact(ops, stream, d_g_mlph, count * 4 * W);
        op_gemm(ops, stream, d_g_mlpo, g->mlp_proj_w, d_g_mlph, g->mlp_proj_b, W, 4 * W, count);
        hipMemcpyAsync(d_g_post, d_g_res, (size_t)count * W * sizeof(float),
                       hipMemcpyDeviceToDevice, stream);
        op_add(ops, stream, d_g_post, d_g_mlpo, count * W);

        if (g->ln_post_w) {
            op_layernorm(ops, stream, d_g_ln1, d_g_post, g->ln_post_w, g->ln_post_b, count, W);
        } else {
            hipMemcpyAsync(d_g_ln1, d_g_post, (size_t)count * W * sizeof(float),
                           hipMemcpyDeviceToDevice, stream);
        }

        op_gemm(ops, stream, c->d_sdf_out, g->output_w, d_g_ln1, g->output_b, 1, W, count);

        hipMemcpyAsync(sdf_out + start, c->d_sdf_out,
                       (size_t)count * sizeof(float), hipMemcpyDeviceToHost, stream);
    }
    hipStreamSynchronize(stream);
}

/* Stage 3: ShapeVAE decode + SDF query */
static void run_shapevae(hip_hy3d_runner *r, void *d_latents,
                          int grid_res, float *sdf_out) {
    hy3d_ops *ops = &r->ops;
    hipStream_t stream = r->stream;
    const int N = VAE_NUM_LATENTS;
    const int E = VAE_EMBED_DIM;
    const int W = VAE_WIDTH;

    /* Decide decoder dtype: fp16 (hipBLASLt + WMMA-FA) when weights are f16
     * and the FP16 kernels compiled in.  Default ON; HY3D_VAE_DEC_FP16=0
     * forces the f32 fallback. */
    int dec_fp16 = !ops->use_f32_gemm
                   && ops->layernorm_f16 && ops->cast_f32_to_f16
                   && ops->flash_attn_sa_f16_wmma;
    {
        const char *e = getenv("HY3D_VAE_DEC_FP16");
        if (e && *e && e[0] == '0') dec_fp16 = 0;
    }

    void *d_cur = NULL, *d_next = NULL;
    void *d_dec_a = NULL, *d_dec_b = NULL;
    void *d_block_scratch = NULL;
    void *d_qk_f32 = NULL;
    void *d_zero_bias_f32 = NULL;
    void *d_attn_f32 = NULL;

    if (dec_fp16) {
        d_dec_a = gpu_alloc((size_t)N * W * sizeof(uint16_t));
        d_dec_b = gpu_alloc((size_t)N * W * sizeof(uint16_t));
        /* fp16 block scratch: layout in run_vae_block_fp16 is
         *   ln1[N,W] + qkv[N,3W] + Q[N,W] + K[N,W] + V[N,W]
         *   + aout[N,W] + proj[N,W] + res1[N,W] + ln2[N,W]
         *   + mlph[N,4W] + mlpo[N,W]  =  N * 16W f16. */
        size_t block_scratch = (size_t)N * (W * 16) * sizeof(uint16_t);
        d_block_scratch = gpu_alloc(block_scratch);
        d_qk_f32 = gpu_alloc((size_t)N * W * sizeof(float));
        /* Zero bias for QKV (fused, no bias).  3*W floats. */
        d_zero_bias_f32 = gpu_alloc((size_t)3 * W * sizeof(float));
        hipMemsetAsync(d_zero_bias_f32, 0, (size_t)3 * W * sizeof(float), stream);
        /* Self-attn fallback scratch (Q|K|V|Out) when WMMA hd!=128. */
        d_attn_f32 = gpu_alloc((size_t)4 * N * W * sizeof(float));

        /* Post-KL: f32 latent in, f16 dec_a out via fp16-D epilogue.
         * Cast latent f32 -> f16 first, then run f16 GEMM. */
        void *d_lat_f16 = gpu_alloc((size_t)N * E * sizeof(uint16_t));
        op_cast_f32_to_f16(ops, stream, d_lat_f16, d_latents, N * E);
        op_gemm_f16_bias_f16d(ops, stream, d_dec_a, r->vae_post_kl_w, d_lat_f16,
                              r->vae_post_kl_b, W, E, N);
        hipFree(d_lat_f16);

        d_cur = d_dec_a;
        d_next = d_dec_b;
        for (int i = 0; i < VAE_DEC_LAYERS; i++) {
            run_vae_block_fp16(r, &r->vae_blocks[i], d_cur, d_next,
                               d_block_scratch, d_qk_f32,
                               d_zero_bias_f32, d_attn_f32);
            void *tmp = d_cur; d_cur = d_next; d_next = tmp;
        }
        /* d_cur (f16) now holds decoded latents.  Cast back to f32 in-place
         * by allocating a fresh f32 buffer (geo-query path expects f32). */
        void *d_dec_f32 = gpu_alloc((size_t)N * W * sizeof(float));
        op_cast_f16_to_f32(ops, stream, d_dec_f32, d_cur, N * W);
        hipFree(d_dec_a);
        hipFree(d_dec_b);
        hipFree(d_block_scratch);
        hipFree(d_qk_f32);
        hipFree(d_zero_bias_f32);
        hipFree(d_attn_f32);
        d_cur = d_dec_f32;  /* freed at end as d_dec_a */
        d_dec_a = d_dec_f32;
        d_dec_b = NULL;
        d_block_scratch = NULL;
        d_qk_f32 = NULL;
        d_zero_bias_f32 = NULL;
        d_attn_f32 = NULL;
    } else {
        d_dec_a = gpu_alloc((size_t)N * W * sizeof(float));
        d_dec_b = gpu_alloc((size_t)N * W * sizeof(float));
        size_t block_scratch = (size_t)N * (W * 12 + 4 * W * 4) * sizeof(float);
        d_block_scratch = gpu_alloc(block_scratch);
        op_gemm(ops, stream, d_dec_a, r->vae_post_kl_w, d_latents,
                r->vae_post_kl_b, W, E, N);
        d_cur = d_dec_a;
        d_next = d_dec_b;
        for (int i = 0; i < VAE_DEC_LAYERS; i++) {
            run_vae_block(r, &r->vae_blocks[i], d_cur, d_next, d_block_scratch);
            void *tmp = d_cur; d_cur = d_next; d_next = tmp;
        }
    }

    /* d_cur now contains decoded latents [N, W] */
    int total_points = grid_res * grid_res * grid_res;
    int batch_size = 8192;
    float bounds[6] = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

    void *d_coords = gpu_alloc((size_t)batch_size * 3 * sizeof(float));
    void *d_fourier = gpu_alloc((size_t)batch_size * VAE_FOURIER_DIM * sizeof(float));
    void *d_query_proj = gpu_alloc((size_t)batch_size * W * sizeof(float));
    void *d_sdf_out = gpu_alloc((size_t)batch_size * sizeof(float));

    /* Per-batch scratch (Q-side only — K/V are hoisted out of the loop). */
    size_t geo_scratch_sz = (size_t)batch_size * (W * 8 + 4 * W * 4) * sizeof(float);
    void *d_geo_scratch = gpu_alloc(geo_scratch_sz);

    /* KV-side scratch: depends only on d_cur (decoded latents), constant across
     * all batches.  Hoisted out of the per-batch loop. */
    void *d_g_ln2 = gpu_alloc((size_t)N * W * sizeof(float));
    void *d_g_KV  = gpu_alloc((size_t)N * 2 * W * sizeof(float));
    void *d_g_K   = gpu_alloc((size_t)N * W * sizeof(float));
    void *d_g_V   = gpu_alloc((size_t)N * W * sizeof(float));

    {
        vae_geo_decoder_gpu *g = &r->vae_geo;
        op_layernorm(ops, stream, d_g_ln2, d_cur, g->ln2_w, g->ln2_b, N, W);
        op_gemm(ops, stream, d_g_KV, g->c_kv_w, d_g_ln2, NULL, 2 * W, W, N);
        op_split_kv(ops, stream, d_g_K, d_g_V, d_g_KV, N, VAE_HEADS, VAE_HEAD_DIM);
        if (g->use_qk_norm) {
            op_qk_layernorm(ops, stream, d_g_K, g->k_norm_w, g->k_norm_b,
                            N, VAE_HEADS, VAE_HEAD_DIM, W);
        }
    }

    /* Optional fp16 attention path: cast KV f32->f16 once, allocate Q/aout
     * scratch in fp16, dispatch via op_cross_attn_f16 (WMMA).  Default ON. */
    int use_attn_f16 = 1;
    {
        const char *e = getenv("HY3D_VAE_ATTN_F16");
        if (e && *e && e[0] == '0') use_attn_f16 = 0;
    }
    /* WMMA cross-attn requires head_dim==128 (which we have) and f16 kernel
     * compiled in. */
    if (use_attn_f16 && !ops->cross_attn_f16_wmma) use_attn_f16 = 0;

    /* Optional fp16 decoder per-batch query path: residual stream stays in
     * f16 between query_proj and ln_post, hipBLASLt for c_q/c_proj/mlp.
     * Default ON; HY3D_VAE_DEC_FP16=0 forces the f32 path. */
    int use_dec_f16 = use_attn_f16
                       && !ops->use_f32_gemm
                       && ops->layernorm_f16 && ops->add_f16
                       && ops->cast_f32_to_f16 && ops->cast_f16_to_f32;
    {
        const char *e = getenv("HY3D_VAE_DEC_FP16");
        if (e && *e && e[0] == '0') use_dec_f16 = 0;
    }

    void *d_g_K_f16 = NULL, *d_g_V_f16 = NULL;
    void *d_Q_f16 = NULL, *d_aout_f16 = NULL;
    if (use_attn_f16 || use_dec_f16) {
        d_g_K_f16  = gpu_alloc((size_t)N * W * sizeof(uint16_t));
        d_g_V_f16  = gpu_alloc((size_t)N * W * sizeof(uint16_t));
        d_Q_f16    = gpu_alloc((size_t)batch_size * W * sizeof(uint16_t));
        d_aout_f16 = gpu_alloc((size_t)batch_size * W * sizeof(uint16_t));
        op_cast_f32_to_f16(ops, stream, d_g_K_f16, d_g_K, N * W);
        op_cast_f32_to_f16(ops, stream, d_g_V_f16, d_g_V, N * W);
    }

    /* Per-batch fp16 scratch + zero-bias for the decoder fp16 path. */
    void *d_g_ln1_f16 = NULL, *d_g_proj_f16 = NULL, *d_g_res_f16 = NULL;
    void *d_g_ln3_f16 = NULL, *d_g_mlph_f16 = NULL, *d_g_mlpo_f16 = NULL;
    void *d_g_post_f16 = NULL, *d_g_lnpost_f16 = NULL;
    void *d_g_qknorm_f32 = NULL, *d_g_zero_bias_f32 = NULL;
    if (use_dec_f16) {
        size_t f16_w = (size_t)batch_size * W * sizeof(uint16_t);
        d_g_ln1_f16    = gpu_alloc(f16_w);
        d_g_proj_f16   = gpu_alloc(f16_w);
        d_g_res_f16    = gpu_alloc(f16_w);
        d_g_ln3_f16    = gpu_alloc(f16_w);
        d_g_mlph_f16   = gpu_alloc((size_t)batch_size * 4 * W * sizeof(uint16_t));
        d_g_mlpo_f16   = gpu_alloc(f16_w);
        d_g_post_f16   = gpu_alloc(f16_w);
        d_g_lnpost_f16 = gpu_alloc(f16_w);
        d_g_qknorm_f32 = gpu_alloc((size_t)batch_size * W * sizeof(float));
        d_g_zero_bias_f32 = gpu_alloc((size_t)W * sizeof(float));
        hipMemsetAsync(d_g_zero_bias_f32, 0, (size_t)W * sizeof(float), stream);
    }

    /* Build geo-query context (shared by dense and hierarchical paths). */
    hy3d_geo_ctx ctx = {
        .ops = ops, .stream = stream,
        .d_coords = d_coords, .d_fourier = d_fourier,
        .d_query_proj = d_query_proj, .d_sdf_out = d_sdf_out,
        .d_geo_scratch = d_geo_scratch,
        .d_g_K = d_g_K, .d_g_V = d_g_V,
        .d_g_K_f16 = d_g_K_f16, .d_g_V_f16 = d_g_V_f16,
        .d_Q_f16 = d_Q_f16, .d_aout_f16 = d_aout_f16,
        .d_g_ln1_f16 = d_g_ln1_f16, .d_g_proj_f16 = d_g_proj_f16,
        .d_g_res_f16 = d_g_res_f16, .d_g_ln3_f16 = d_g_ln3_f16,
        .d_g_mlph_f16 = d_g_mlph_f16, .d_g_mlpo_f16 = d_g_mlpo_f16,
        .d_g_post_f16 = d_g_post_f16, .d_g_lnpost_f16 = d_g_lnpost_f16,
        .d_g_qknorm_f32 = d_g_qknorm_f32,
        .d_g_zero_bias_f32 = d_g_zero_bias_f32,
        .use_dec_f16 = use_dec_f16,
        .batch_size = batch_size,
        .use_attn_f16 = use_attn_f16,
        .g = &r->vae_geo,
        .fourier_freqs = r->vae_fourier_freqs,
        .N = N, .W = W,
    };

    /* Decide path: hierarchical (default) vs dense (HY3D_VAE_HIER=0). */
    int hier = 1;
    {
        const char *e = getenv("HY3D_VAE_HIER");
        if (e && *e && e[0] == '0') hier = 0;
        if (grid_res < 32) hier = 0;  /* too small to bother */
    }

    if (!hier) {
        /* ---- Dense path: query every voxel. ---- */
        float dx = (bounds[3] - bounds[0]) / (float)(grid_res - 1);
        float dy = (bounds[4] - bounds[1]) / (float)(grid_res - 1);
        float dz = (bounds[5] - bounds[2]) / (float)(grid_res - 1);
        float *coords = (float *)malloc((size_t)total_points * 3 * sizeof(float));
        for (int ix = 0; ix < grid_res; ix++)
            for (int iy = 0; iy < grid_res; iy++)
                for (int iz = 0; iz < grid_res; iz++) {
                    int idx = (ix * grid_res + iy) * grid_res + iz;
                    coords[idx * 3 + 0] = bounds[0] + ix * dx;
                    coords[idx * 3 + 1] = bounds[1] + iy * dy;
                    coords[idx * 3 + 2] = bounds[2] + iz * dz;
                }
        hy3d_geo_query_run(&ctx, coords, total_points, sdf_out);
        free(coords);
    } else {
        /* ---- Hierarchical path: coarse dense + sparse near-surface fine. ---- */
        int fine_res   = grid_res;
        int coarse_res = grid_res / 2;
        if (coarse_res < 16) coarse_res = 16;

        int coarse_total = coarse_res * coarse_res * coarse_res;
        float *coarse_coords = (float *)malloc((size_t)coarse_total * 3 * sizeof(float));
        float *coarse_sdf    = (float *)malloc((size_t)coarse_total * sizeof(float));
        float dxc = (bounds[3] - bounds[0]) / (float)(coarse_res - 1);
        float dyc = (bounds[4] - bounds[1]) / (float)(coarse_res - 1);
        float dzc = (bounds[5] - bounds[2]) / (float)(coarse_res - 1);
        for (int ix = 0; ix < coarse_res; ix++)
            for (int iy = 0; iy < coarse_res; iy++)
                for (int iz = 0; iz < coarse_res; iz++) {
                    int idx = (ix * coarse_res + iy) * coarse_res + iz;
                    coarse_coords[idx * 3 + 0] = bounds[0] + ix * dxc;
                    coarse_coords[idx * 3 + 1] = bounds[1] + iy * dyc;
                    coarse_coords[idx * 3 + 2] = bounds[2] + iz * dzc;
                }
        if (r->verbose)
            fprintf(stderr, "HY3D: VAE hier coarse pass at %d^3 (%d points)\n",
                    coarse_res, coarse_total);
        hy3d_geo_query_run(&ctx, coarse_coords, coarse_total, coarse_sdf);
        free(coarse_coords);

        /* Build active mask at coarse, project to fine, dilate twice. */
        size_t fine_total_sz = (size_t)fine_res * fine_res * fine_res;
        uint8_t *cmask = (uint8_t *)malloc((size_t)coarse_total);
        uint8_t *fmask = (uint8_t *)calloc(fine_total_sz, 1);
        hy3d_active_mask_3d(cmask, coarse_sdf, coarse_res);

        int FRR = fine_res * fine_res, CRR = coarse_res * coarse_res;
        for (int cx = 0; cx < coarse_res; cx++) {
            int fx = cx * 2;
            if (fx >= fine_res) continue;
            for (int cy = 0; cy < coarse_res; cy++) {
                int fy = cy * 2;
                if (fy >= fine_res) continue;
                for (int cz = 0; cz < coarse_res; cz++) {
                    int fz = cz * 2;
                    if (fz >= fine_res) continue;
                    if (cmask[cx * CRR + cy * coarse_res + cz])
                        fmask[fx * FRR + fy * fine_res + fz] = 1;
                }
            }
        }
        /* PyTorch dilates fine mask 2 times (since this is the last/only refine
         * step → expand_num=0 → 2-0=2 dilations). */
        hy3d_dilate_3x3x3(fmask, fine_res);
        hy3d_dilate_3x3x3(fmask, fine_res);

        /* Compact active fine voxels. */
        int n_active = 0;
        for (size_t i = 0; i < fine_total_sz; i++)
            if (fmask[i]) n_active++;
        if (r->verbose)
            fprintf(stderr, "HY3D: VAE hier fine pass: %d active / %zu total (%.1f%%)\n",
                    n_active, fine_total_sz, 100.0 * n_active / (double)fine_total_sz);

        float dxf = (bounds[3] - bounds[0]) / (float)(fine_res - 1);
        float dyf = (bounds[4] - bounds[1]) / (float)(fine_res - 1);
        float dzf = (bounds[5] - bounds[2]) / (float)(fine_res - 1);
        float *active_coords = (float *)malloc((size_t)n_active * 3 * sizeof(float));
        int   *active_idx    = (int *)  malloc((size_t)n_active * sizeof(int));
        {
            int j = 0;
            for (int ix = 0; ix < fine_res; ix++)
                for (int iy = 0; iy < fine_res; iy++)
                    for (int iz = 0; iz < fine_res; iz++) {
                        int idx = ix * FRR + iy * fine_res + iz;
                        if (!fmask[idx]) continue;
                        active_coords[j * 3 + 0] = bounds[0] + ix * dxf;
                        active_coords[j * 3 + 1] = bounds[1] + iy * dyf;
                        active_coords[j * 3 + 2] = bounds[2] + iz * dzf;
                        active_idx[j] = idx;
                        j++;
                    }
        }
        float *active_sdf = (float *)malloc((size_t)n_active * sizeof(float));
        hy3d_geo_query_run(&ctx, active_coords, n_active, active_sdf);

        /* Fill fine grid: default = nearest-upsampled coarse value (carries
         * correct sign for inactive voxels, far from zero), overwrite at active. */
        for (int fx = 0; fx < fine_res; fx++) {
            int cx = fx / 2; if (cx >= coarse_res) cx = coarse_res - 1;
            for (int fy = 0; fy < fine_res; fy++) {
                int cy = fy / 2; if (cy >= coarse_res) cy = coarse_res - 1;
                for (int fz = 0; fz < fine_res; fz++) {
                    int cz = fz / 2; if (cz >= coarse_res) cz = coarse_res - 1;
                    sdf_out[fx * FRR + fy * fine_res + fz] =
                        coarse_sdf[cx * CRR + cy * coarse_res + cz];
                }
            }
        }
        for (int j = 0; j < n_active; j++)
            sdf_out[active_idx[j]] = active_sdf[j];

        free(coarse_sdf);
        free(cmask); free(fmask);
        free(active_coords); free(active_idx); free(active_sdf);
    }
    (void)total_points;

    /* Cleanup */
    if (d_dec_a) hipFree(d_dec_a);
    if (d_dec_b) hipFree(d_dec_b);
    if (d_block_scratch) hipFree(d_block_scratch);
    if (d_qk_f32) hipFree(d_qk_f32);
    hipFree(d_coords);
    hipFree(d_fourier);
    hipFree(d_query_proj);
    hipFree(d_sdf_out);
    hipFree(d_geo_scratch);
    hipFree(d_g_ln2);
    hipFree(d_g_KV);
    hipFree(d_g_K);
    hipFree(d_g_V);
    if (d_g_K_f16) hipFree(d_g_K_f16);
    if (d_g_V_f16) hipFree(d_g_V_f16);
    if (d_Q_f16)   hipFree(d_Q_f16);
    if (d_aout_f16) hipFree(d_aout_f16);
    if (d_g_ln1_f16)    hipFree(d_g_ln1_f16);
    if (d_g_proj_f16)   hipFree(d_g_proj_f16);
    if (d_g_res_f16)    hipFree(d_g_res_f16);
    if (d_g_ln3_f16)    hipFree(d_g_ln3_f16);
    if (d_g_mlph_f16)   hipFree(d_g_mlph_f16);
    if (d_g_mlpo_f16)   hipFree(d_g_mlpo_f16);
    if (d_g_post_f16)   hipFree(d_g_post_f16);
    if (d_g_lnpost_f16) hipFree(d_g_lnpost_f16);
    if (d_g_qknorm_f32) hipFree(d_g_qknorm_f32);
    if (d_g_zero_bias_f32) hipFree(d_g_zero_bias_f32);
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

        double u1 = ((r1 >> 11) + 0.5) / 9007199254740992.0; /* 2^53 */
        double u2 = ((r2 >> 11) + 0.5) / 9007199254740992.0;
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

static void hy3d_parse_csv_steps(const char *csv,
                                 int *steps, int *count_out, int max_count) {
    *count_out = 0;
    if (!csv || !*csv) return;
    const char *p = csv;
    while (*p && *count_out < max_count) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (v > 0 && v <= 1000000) {
            int step = (int)v;
            int seen = 0;
            for (int i = 0; i < *count_out; i++) {
                if (steps[i] == step) { seen = 1; break; }
            }
            if (!seen) steps[(*count_out)++] = step;
        }
        p = end;
        while (*p == ' ' || *p == ',') p++;
    }
}

void hip_hy3d_set_latent_dump(hip_hy3d_runner *r, const char *steps_csv, const char *prefix) {
    if (!r) return;
    r->latent_dump_count = 0;
    r->latent_dump_prefix[0] = '\0';
    if (prefix && *prefix) {
        size_t n = strlen(prefix);
        if (n >= sizeof(r->latent_dump_prefix)) n = sizeof(r->latent_dump_prefix) - 1;
        memcpy(r->latent_dump_prefix, prefix, n);
        r->latent_dump_prefix[n] = '\0';
    }
    hy3d_parse_csv_steps(steps_csv, r->latent_dump_steps,
                         &r->latent_dump_count, HY3D_MAX_LATENT_DUMP_STEPS);
}

void hip_hy3d_set_velocity_dump(hip_hy3d_runner *r, const char *steps_csv, const char *prefix) {
    if (!r) return;
    r->velocity_dump_count = 0;
    r->velocity_dump_prefix[0] = '\0';
    if (prefix && *prefix) {
        size_t n = strlen(prefix);
        if (n >= sizeof(r->velocity_dump_prefix)) n = sizeof(r->velocity_dump_prefix) - 1;
        memcpy(r->velocity_dump_prefix, prefix, n);
        r->velocity_dump_prefix[n] = '\0';
    }
    hy3d_parse_csv_steps(steps_csv, r->velocity_dump_steps,
                         &r->velocity_dump_count, HY3D_MAX_LATENT_DUMP_STEPS);
}

int hip_hy3d_set_init_latents(hip_hy3d_runner *r, const float *latents, int n) {
    if (!r || !latents) return -1;
    const int expected = DIT_INPUT_SIZE * DIT_IN_CHANNELS;
    if (n != expected) return -1;
    float *tmp = (float *)malloc((size_t)n * sizeof(float));
    if (!tmp) return -1;
    memcpy(tmp, latents, (size_t)n * sizeof(float));
    free(r->init_latents);
    r->init_latents = tmp;
    r->init_latents_n = n;
    return 0;
}

int hip_hy3d_set_init_contexts(hip_hy3d_runner *r,
                               const float *cond,
                               const float *uncond,
                               int n) {
    if (!r || !cond) return -1;
    const int expected = DINO_SEQ_LEN * DIT_CONTEXT_DIM;
    if (n != expected) return -1;
    float *c = (float *)malloc((size_t)n * sizeof(float));
    if (!c) return -1;
    memcpy(c, cond, (size_t)n * sizeof(float));

    float *u = NULL;
    if (uncond) {
        u = (float *)malloc((size_t)n * sizeof(float));
        if (!u) { free(c); return -1; }
        memcpy(u, uncond, (size_t)n * sizeof(float));
    }

    free(r->init_ctx_cond);
    free(r->init_ctx_uncond);
    r->init_ctx_cond = c;
    r->init_ctx_uncond = u;
    r->init_ctx_n = n;
    return 0;
}

#ifdef HY3D_HIPBLASLT_ENABLED
/* Lever A: hipBLASLt plan pre-warm.
 *
 * One-shot CLI pays first-call plan-build (~120 ms × 5-7 unique HHS shapes
 * across DiT + DINOv2) on every invocation.  Fire one dummy GEMM at every
 * (M,N,K,dtype,epilogue) shape we'll hit during inference; the bridge plan
 * cache hits warm thereafter.  Plan key is shape+dtype+epilogue — weight
 * identity doesn't matter — so we recycle a single oversized scratch.
 *
 * Default ON; HY3D_PLAN_PREWARM=0 disables.  Fully transparent: no
 * inference state changes. */
static void prewarm_blaslt_plans(hip_hy3d_runner *r) {
    static int prewarm_mode = -1;
    if (prewarm_mode < 0) {
        const char *e = getenv("HY3D_PLAN_PREWARM");
        prewarm_mode = (e && *e && e[0] == '0') ? 0 : 1;
    }
    if (!prewarm_mode) return;

    hy3d_ops *ops = &r->ops;
    hipStream_t stream = r->stream;
    if (r->verbose) fprintf(stderr, "HY3D: pre-warming hipBLASLt plans...\n");

    /* Sized for the largest shape we touch:
     *   max M = max(N1=4097, N=4096) = 4097
     *   max K = max(2048, 4096, 8192) = 8192
     *   max N_out = max(6144, 8192, 4096) = 8192
     * Pick conservative caps so a single buffer covers Y (HHS f16 out and
     * HSS f32 out), X, W, bias. */
    const size_t max_M = 4097;
    const size_t max_NK = 8192;
    const size_t max_act_f16 = max_M * max_NK * sizeof(uint16_t);
    const size_t max_W_f16   = max_NK * max_NK * sizeof(uint16_t);
    const size_t max_act_f32 = max_M * max_NK * sizeof(float);
    const size_t max_bias    = max_NK * sizeof(float);

    void *d_X = NULL, *d_W = NULL, *d_Yh = NULL, *d_Yf = NULL, *d_B = NULL;
    if (hipMalloc(&d_X, max_act_f16) != hipSuccess) goto done;
    if (hipMalloc(&d_W, max_W_f16) != hipSuccess) goto done;
    if (hipMalloc(&d_Yh, max_act_f16) != hipSuccess) goto done;
    if (hipMalloc(&d_Yf, max_act_f32) != hipSuccess) goto done;
    if (hipMalloc(&d_B, max_bias) != hipSuccess) goto done;
    /* Zero bias (f32) so plan's epilogue reads valid memory. */
    hipMemsetAsync(d_B, 0, max_bias, stream);
    /* Zero inputs to keep numerics tame (not strictly needed). */
    hipMemsetAsync(d_X, 0, max_act_f16, stream);
    hipMemsetAsync(d_W, 0, max_W_f16, stream);
    hipStreamSynchronize(stream);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* DiT HHS shapes (op_gemm_f16_bias_f16d). */
    const int N1   = DIT_INPUT_SIZE + 1;       /* 4097 */
    const int N0   = DIT_INPUT_SIZE;           /* 4096 */
    const int H    = DIT_HIDDEN;               /* 2048 */
    const int FFN  = DIT_FFN;                  /* 8192 */
    const int CTX  = DINO_SEQ_LEN;             /* 1370 */
    const int CDIM = DIT_CONTEXT_DIM;          /* 1024 */

    /* sa_qkv */
    op_gemm_f16_bias_f16d(ops, stream, d_Yh, d_W, d_X, d_B, 3 * H, H, N1);
    /* sa_out / ca_q (same shape) */
    op_gemm_f16_bias_f16d(ops, stream, d_Yh, d_W, d_X, d_B, H, H, N1);
    /* skip_linear */
    op_gemm_f16_bias_f16d(ops, stream, d_Yh, d_W, d_X, d_B, H, 2 * H, N1);
    /* mlp_fc1 with fused gelu */
    op_gemm_f16_bias_gelu_f16d(ops, stream, d_Yh, d_W, d_X, d_B, FFN, H, N1);
    /* mlp_fc2 */
    op_gemm_f16_bias_f16d(ops, stream, d_Yh, d_W, d_X, d_B, H, FFN, N1);
    /* ca_kv on context */
    op_gemm_f16_bias_f16d(ops, stream, d_Yh, d_W, d_X, d_B, 2 * H, CDIM, CTX);
    /* DiT final linear: HSS (f16 in, f32 out, with bias). */
    op_gemm_f16_bias(ops, stream, d_Yf, d_W, d_X, d_B, DIT_IN_CHANNELS, H, N0);

    /* DINOv2 HHS shapes (per run_dinov2_fp16). */
    const int DH    = DINO_HIDDEN;             /* 1024 */
    const int DSEQ  = DINO_SEQ_LEN;            /* 1370 */
    const int DFFN  = DINO_FFN;                /* 4096 */
    /* q/k/v/out: (DSEQ, DH, DH) */
    op_gemm_f16_bias_f16d(ops, stream, d_Yh, d_W, d_X, d_B, DH, DH, DSEQ);
    /* fc1 with fused gelu: (DSEQ, DFFN, DH) */
    op_gemm_f16_bias_gelu_f16d(ops, stream, d_Yh, d_W, d_X, d_B, DFFN, DH, DSEQ);
    /* fc2: (DSEQ, DH, DFFN) */
    op_gemm_f16_bias_f16d(ops, stream, d_Yh, d_W, d_X, d_B, DH, DFFN, DSEQ);

    hipStreamSynchronize(stream);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (r->verbose) {
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                  + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        fprintf(stderr, "HY3D: plan pre-warm done in %.1f ms (10 shapes)\n", ms);
    }

done:
    if (d_X)  hipFree(d_X);
    if (d_W)  hipFree(d_W);
    if (d_Yh) hipFree(d_Yh);
    if (d_Yf) hipFree(d_Yf);
    if (d_B)  hipFree(d_B);
}
#endif  /* HY3D_HIPBLASLT_ENABLED */

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

#ifdef HY3D_HIPBLASLT_ENABLED
    /* Lever A: pre-warm hipBLASLt plans for all DiT + DINOv2 hot shapes
     * once after weights are loaded.  Saves ~600 ms on first DiT step
     * and ~430 ms on the DINOv2 stage for one-shot CLI invocations. */
    prewarm_blaslt_plans(r);
#endif

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

    /* Per-stage GPU timing events */
    hipEvent_t ev_e2e_start = NULL, ev_e2e_end = NULL;
    hipEvent_t ev_dino_start = NULL, ev_dino_end = NULL;
    hipEvent_t ev_dit_start = NULL, ev_dit_end = NULL;
    hipEvent_t ev_vae_start = NULL, ev_vae_end = NULL;
    hipEvent_t ev_step_start[HY3D_MAX_DIT_STEPS] = {0};
    hipEvent_t ev_step_end[HY3D_MAX_DIT_STEPS] = {0};
    int n_steps_for_timing = (n_steps < HY3D_MAX_DIT_STEPS) ? n_steps : HY3D_MAX_DIT_STEPS;
    if (r->timing_enabled) {
        hipEventCreate(&ev_e2e_start);  hipEventCreate(&ev_e2e_end);
        hipEventCreate(&ev_dino_start); hipEventCreate(&ev_dino_end);
        hipEventCreate(&ev_dit_start);  hipEventCreate(&ev_dit_end);
        hipEventCreate(&ev_vae_start);  hipEventCreate(&ev_vae_end);
        for (int i = 0; i < n_steps_for_timing; i++) {
            hipEventCreate(&ev_step_start[i]);
            hipEventCreate(&ev_step_end[i]);
        }
        hipEventRecord(ev_e2e_start, r->stream);
        hipEventRecord(ev_dino_start, r->stream);
    }

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
    {
        static int dino_fp16 = -1;
        if (dino_fp16 < 0) {
            const char *e = getenv("HY3D_DINO_FP16");
            /* default on; HY3D_DINO_FP16=0 disables. */
            dino_fp16 = (e && *e && e[0] == '0') ? 0 : 1;
        }
        if (dino_fp16 && !r->ops.use_f32_gemm)
            run_dinov2_fp16(r, d_image, d_dino_out);
        else
            run_dinov2(r, d_image, d_dino_out);
    }
    if (r->init_ctx_cond && r->init_ctx_n == DINO_SEQ_LEN * DIT_CONTEXT_DIM) {
        hipMemcpyAsync(d_dino_out, r->init_ctx_cond,
                       (size_t)r->init_ctx_n * sizeof(float),
                       hipMemcpyHostToDevice, stream);
        if (r->verbose) fprintf(stderr, "HY3D: using user-provided conditional context\n");
    }
    if (r->verbose > 1) {
        hy3d_dbg_dump_npy(stream, d_dino_out, DINO_SEQ_LEN, DIT_CONTEXT_DIM,
                          "hip_dino_context_after_override.npy");
    }
    hipFree(d_rgb);
    hipFree(d_image);

    /* NaN check after DINOv2 */
    if (r->verbose) {
        hipDeviceSynchronize();
        float dino_check[8];
        hipMemcpy(dino_check, d_dino_out, sizeof(dino_check), hipMemcpyDeviceToHost);
        int has_nan = 0;
        for (int i = 0; i < 8; i++) if (dino_check[i] != dino_check[i]) has_nan = 1;
        float *dino_full = (float *)malloc((size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float));
        hipMemcpy(dino_full, d_dino_out, (size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float), hipMemcpyDeviceToHost);
        float dmin = dino_full[0], dmax = dino_full[0], dsum = 0; int nan_cnt = 0;
        for (int i = 0; i < DINO_SEQ_LEN * DINO_HIDDEN; i++) {
            if (dino_full[i] != dino_full[i]) { nan_cnt++; continue; }
            if (dino_full[i] < dmin) dmin = dino_full[i];
            if (dino_full[i] > dmax) dmax = dino_full[i];
            dsum += dino_full[i];
        }
        fprintf(stderr, "  DINOv2 output: min=%.4f max=%.4f mean=%.6f nan=%d%s\n",
                dmin, dmax, dsum / (DINO_SEQ_LEN * DINO_HIDDEN), nan_cnt,
                has_nan ? " *** NaN DETECTED ***" : "");
        free(dino_full);
    }

    if (r->timing_enabled) {
        hipEventRecord(ev_dino_end, r->stream);
        hipEventRecord(ev_dit_start, r->stream);
    }

    /* ---- Stage 2: DiT diffusion with flow matching ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 2 - DiT diffusion (%d steps)...\n", n_steps);

    int latent_size = DIT_INPUT_SIZE * DIT_IN_CHANNELS;
    float *noise_cpu = (float *)malloc((size_t)latent_size * sizeof(float));
    if (r->init_latents && r->init_latents_n == latent_size) {
        memcpy(noise_cpu, r->init_latents, (size_t)latent_size * sizeof(float));
        if (r->verbose) fprintf(stderr, "HY3D: using user-provided initial latents\n");
    } else {
        generate_randn(noise_cpu, latent_size, seed);
    }

    void *d_latents = gpu_alloc((size_t)latent_size * sizeof(float));
    hipMemcpyAsync(d_latents, noise_cpu, (size_t)latent_size * sizeof(float), hipMemcpyHostToDevice, stream);
    /* HtoDAsync reads from noise_cpu on the stream — sync before freeing
     * or the GPU copy races against libc recycling the host buffer. */
    hipStreamSynchronize(stream);
    free(noise_cpu);

    void *d_pred_cond = gpu_alloc((size_t)latent_size * sizeof(float));
    void *d_pred_uncond = gpu_alloc((size_t)latent_size * sizeof(float));
    void *d_pred_combined = gpu_alloc((size_t)latent_size * sizeof(float));

    void *d_uncond_ctx = gpu_alloc((size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float));
    if (r->init_ctx_uncond && r->init_ctx_n == DINO_SEQ_LEN * DIT_CONTEXT_DIM) {
        hipMemcpyAsync(d_uncond_ctx, r->init_ctx_uncond,
                       (size_t)r->init_ctx_n * sizeof(float),
                       hipMemcpyHostToDevice, stream);
        if (r->verbose) fprintf(stderr, "HY3D: using user-provided unconditional context\n");
    } else {
        hipMemsetAsync(d_uncond_ctx, 0, (size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float), stream);
    }

    /* Flow matching schedule — matches Hunyuan3DDiTFlowMatchingPipeline.__call__
     * (hy3dshape/pipelines.py line 736-764):
     *
     *   sigmas = np.linspace(0, 1, n_steps)          # ASCENDING from 0
     *   scheduler.set_timesteps(sigmas=sigmas)       # timesteps = sigmas*1000
     *   for t in timesteps:
     *       t_model = t / num_train_timesteps        # rescales back to [0, 1]
     *       v = model(latents, t_model, ctx)
     *       latents = latents + (sigmas[i+1] - sigmas[i]) * v
     *
     *   - The DiT is fed t in [0, 1] (no internal /1000).
     *   - Sigmas ASCEND 0 -> 1 with sentinel 1.0 at position n_steps (final
     *     step's dsigma is 0, no-op).
     *   - Update is prev = sample + (sn - s) * v. op_euler_step does
     *     x -= dt*v, so dt = sigma - sigma_next (negative except last). */
    for (int step = 0; step < n_steps; step++) {
        float sigma, sigma_next;
        if (n_steps == 1) {
            sigma = 0.0f;
            sigma_next = 1.0f;
        } else {
            sigma = (float)step / (float)(n_steps - 1);
            sigma_next = (step == n_steps - 1)
                              ? 1.0f  /* scheduler sentinel */
                              : (float)(step + 1) / (float)(n_steps - 1);
        }
        float dt = sigma - sigma_next; /* negative except last step */
        float model_t = sigma;

        if (r->verbose && (step % 5 == 0 || step == n_steps - 1))
            fprintf(stderr, "  step %d/%d (t=%.4f)\n", step + 1, n_steps, model_t);

        if (r->timing_enabled && step < n_steps_for_timing)
            hipEventRecord(ev_step_start[step], r->stream);

        if (r->latent_dump_count > 0 && hy3d_should_dump_latent(r, step + 1)) {
            hy3d_dump_latent(stream, d_latents, DIT_INPUT_SIZE, DIT_IN_CHANNELS,
                             r->latent_dump_prefix, step + 1);
            if (r->verbose)
                fprintf(stderr, "    dumped latent step %d -> %s_%03d.npy\n",
                        step + 1, r->latent_dump_prefix, step + 1);
        }

        run_dit_forward(r, d_latents, model_t, d_dino_out, d_pred_cond);
        run_dit_forward(r, d_latents, model_t, d_uncond_ctx, d_pred_uncond);
        op_cfg_combine(ops, stream, d_pred_combined, d_pred_cond, d_pred_uncond,
                       guidance_scale, latent_size);

        if (r->velocity_dump_count > 0 && hy3d_should_dump_velocity(r, step + 1)) {
            hy3d_dump_latent(stream, d_pred_combined, DIT_INPUT_SIZE, DIT_IN_CHANNELS,
                             r->velocity_dump_prefix, step + 1);
            if (r->verbose)
                fprintf(stderr, "    dumped velocity step %d -> %s_%03d.npy\n",
                        step + 1, r->velocity_dump_prefix, step + 1);
        }

        op_euler_step(ops, stream, d_latents, d_pred_combined, dt, latent_size);

        if (r->timing_enabled && step < n_steps_for_timing)
            hipEventRecord(ev_step_end[step], r->stream);

        /* NaN check after each step */
        if (r->verbose && step == 0) {
            hipDeviceSynchronize();
            float chk[8];
            hipMemcpy(chk, d_pred_cond, sizeof(chk), hipMemcpyDeviceToHost);
            int nan_c = 0; for (int i = 0; i < 8; i++) if (chk[i] != chk[i]) nan_c++;
            fprintf(stderr, "  DiT cond[0:8]: %.4f %.4f %.4f %.4f nan=%d\n",
                    chk[0], chk[1], chk[2], chk[3], nan_c);
            hipMemcpy(chk, d_latents, sizeof(chk), hipMemcpyDeviceToHost);
            nan_c = 0; for (int i = 0; i < 8; i++) if (chk[i] != chk[i]) nan_c++;
            fprintf(stderr, "  latents[0:8]:  %.4f %.4f %.4f %.4f nan=%d\n",
                    chk[0], chk[1], chk[2], chk[3], nan_c);
        }
    }

    hipFree(d_pred_cond);
    hipFree(d_pred_uncond);
    hipFree(d_pred_combined);
    hipFree(d_uncond_ctx);
    hipFree(d_dino_out);

    /* Final latent stats for debugging */
    if (r->verbose) {
        hipStreamSynchronize(stream);
        float *lcpu = (float *)malloc((size_t)latent_size * sizeof(float));
        hipMemcpy(lcpu, d_latents, (size_t)latent_size * sizeof(float), hipMemcpyDeviceToHost);
        float lmn = lcpu[0], lmx = lcpu[0], lsm = 0;
        for (int i = 0; i < latent_size; i++) {
            if (lcpu[i] < lmn) lmn = lcpu[i];
            if (lcpu[i] > lmx) lmx = lcpu[i];
            lsm += lcpu[i];
        }
        float lmean = lsm / (float)latent_size;
        float lvar = 0;
        for (int i = 0; i < latent_size; i++) { float d = lcpu[i]-lmean; lvar += d*d; }
        fprintf(stderr, "HY3D: final latent: min=%.4f max=%.4f mean=%.4f std=%.4f\n",
                lmn, lmx, lmean, sqrtf(lvar/(float)latent_size));
        free(lcpu);
    }

    if (r->timing_enabled) {
        hipEventRecord(ev_dit_end, r->stream);
        hipEventRecord(ev_vae_start, r->stream);
    }

    /* ---- Stage 3: ShapeVAE decode + SDF query ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 3 - ShapeVAE decode (grid %d^3)...\n", grid_res);

    int total_pts = grid_res * grid_res * grid_res;
    float *sdf_grid = (float *)malloc((size_t)total_pts * sizeof(float));
    run_shapevae(r, d_latents, grid_res, sdf_grid);
    if (r->timing_enabled) {
        hipEventRecord(ev_vae_end, r->stream);
        hipEventRecord(ev_e2e_end, r->stream);
    }
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

    if (r->timing_enabled) {
        hipEventSynchronize(ev_e2e_end);
        memset(&r->timings, 0, sizeof(r->timings));
        hipEventElapsedTime(&r->timings.dino_ms,      ev_dino_start, ev_dino_end);
        hipEventElapsedTime(&r->timings.dit_total_ms, ev_dit_start,  ev_dit_end);
        hipEventElapsedTime(&r->timings.vae_ms,       ev_vae_start,  ev_vae_end);
        hipEventElapsedTime(&r->timings.e2e_ms,       ev_e2e_start,  ev_e2e_end);
        r->timings.dit_steps = n_steps_for_timing;
        for (int i = 0; i < n_steps_for_timing; i++) {
            hipEventElapsedTime(&r->timings.dit_step_ms[i],
                                ev_step_start[i], ev_step_end[i]);
        }
        r->timing_valid = 1;

        hipEventDestroy(ev_e2e_start);  hipEventDestroy(ev_e2e_end);
        hipEventDestroy(ev_dino_start); hipEventDestroy(ev_dino_end);
        hipEventDestroy(ev_dit_start);  hipEventDestroy(ev_dit_end);
        hipEventDestroy(ev_vae_start);  hipEventDestroy(ev_vae_end);
        for (int i = 0; i < n_steps_for_timing; i++) {
            hipEventDestroy(ev_step_start[i]);
            hipEventDestroy(ev_step_end[i]);
        }
    }

    return result;
}

void hip_hy3d_set_per_stage_timing(hip_hy3d_runner *r, int enable) {
    if (!r) return;
    r->timing_enabled = enable ? 1 : 0;
    if (!enable) r->timing_valid = 0;
}

int hip_hy3d_get_stage_times(const hip_hy3d_runner *r, hy3d_stage_times *out) {
    if (!r || !out) return -1;
    if (!r->timing_valid) return -1;
    *out = r->timings;
    return 0;
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

    /* Free host-side trajectory overrides */
    free(r->init_latents);
    r->init_latents = NULL;
    r->init_latents_n = 0;
    free(r->init_ctx_cond);
    free(r->init_ctx_uncond);
    r->init_ctx_cond = NULL;
    r->init_ctx_uncond = NULL;
    r->init_ctx_n = 0;

    /* Note: individual weight buffers are GPU allocations that should also be freed.
     * For a full cleanup, iterate all void* fields. For now, destroying
     * the HIP context reclaims all GPU memory. */

    if (r->module) hipModuleUnload(r->module);
    if (r->stream) hipStreamDestroy(r->stream);
    if (r->ctx) hipCtxDestroy(r->ctx);

    free(r);
}
