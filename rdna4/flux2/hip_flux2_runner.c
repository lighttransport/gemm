/*
 * hip_flux2_runner.c - HIP/ROCm Flux.2 Klein text-to-image runner (RDNA4)
 *
 * GPU-accelerated DiT (5 double-stream + 20 single-stream blocks).
 * VAE decode falls back to CPU.
 *
 * Compiles with plain gcc (no hipcc). Uses rocew for dynamic HIP/HIPRTC loading.
 * F32 weights on GPU, F32 compute. Single-stream sequential kernel launches.
 *
 * Port of cuda_flux2_runner.h for AMD ROCm/HIP.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#define FLUX2_DIT_IMPLEMENTATION
#include "../../common/flux2_klein_dit.h"
#define FLUX2_VAE_IMPLEMENTATION
#include "../../common/flux2_klein_vae.h"

#include "hip_flux2_runner.h"
#include "../rocew.h"
#include "../hip_kernels_common.h"
#include "hip_flux2_kernels.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- GPU weight structures ---- */

typedef struct {
    void *qkv_w;     /* [3H, H] F32 */
    void *proj_w;    /* [H, H] F32 */
    void *mlp_up_w;  /* [2*n_ff, H] F32 */
    void *mlp_dn_w;  /* [H, n_ff] F32 */
    void *q_norm;    /* [head_dim] F32 */
    void *k_norm;    /* [head_dim] F32 */
} flux2_hip_stream_t;

typedef struct {
    flux2_hip_stream_t img, txt;
} flux2_hip_dblk_t;

typedef struct {
    void *linear1_w;   /* [3H+2*n_ff, H] F32 */
    void *l2_attn_w;   /* [H, H] F32 */
    void *l2_mlp_w;    /* [H, n_ff] F32 */
    void *q_norm;      /* [head_dim] F32 */
    void *k_norm;      /* [head_dim] F32 */
} flux2_hip_sblk_t;

/* ---- Runner struct ---- */

struct hip_flux2_runner {
    int device_id;
    int verbose;

    hipModule_t mod;
    hipFunction_t fn_gemm, fn_layernorm, fn_silu, fn_adaln, fn_gated_add;
    hipFunction_t fn_rmsnorm_ph, fn_swiglu, fn_flash_attn;
    hipFunction_t fn_rope_img, fn_rope_txt, fn_bf16_trunc, fn_add;
    /* VAE kernels */
    hipFunction_t fn_vae_im2col, fn_vae_add_bias, fn_vae_transpose, fn_vae_conv1, fn_vae_gnsilu;
    hipFunction_t fn_vae_up2x, fn_vae_attn, fn_vae_bn;

    /* CPU model (for arch params + VAE fallback) */
    flux2_dit_model *dit;
    flux2_vae_model *vae;
    int vae_gpu_loaded;

    int H, nH, hd, n_ff, pin, txt_dim, n_dbl, n_sgl;

    /* Global GPU weights */
    void *d_img_in_w, *d_img_in_b, *d_txt_in_w, *d_txt_in_b;
    void *d_t_fc1_w, *d_t_fc1_b, *d_t_fc2_w, *d_t_fc2_b;
    void *d_mod_img_w, *d_mod_txt_w, *d_mod_sgl_w;
    void *d_out_mod_w, *d_out_proj_w;

    flux2_hip_dblk_t *gpu_dblk;
    flux2_hip_sblk_t *gpu_sblk;

    /* Activation buffers */
    void *d_img, *d_txt, *d_joint;
    void *d_temb, *d_temb_silu;
    void *d_mod_img_v, *d_mod_txt_v, *d_mod_sgl_v;
    void *d_q, *d_k, *d_v, *d_attn_out;
    void *d_scratch1, *d_scratch2, *d_scratch3;
    void *d_img_in_buf, *d_txt_in_buf;
    int max_tok;
    int gpu_loaded;
};

/* ---- Upload helpers ---- */

static void *gpu_upload_f32(const float *data, int n) {
    return hip_upload_raw(data, (size_t)n * sizeof(float));
}

static void *gpu_upload_f32_or0(const float *data, int n) {
    return data ? gpu_upload_f32(data, n) : NULL;
}

static void *gpu_upload_mat(const flux2_mat *m) {
    return gpu_upload_f32(m->w, m->rows * m->cols);
}

/* ---- Op functions ---- */

static void op_gemm(hip_flux2_runner *r, void *Y, void *W,
                    void *X, void *bias, int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    hipModuleLaunchKernel(r->fn_gemm, gx, gy, 1, 16, 16, 1, 0, NULL, args, NULL);
}

static void op_adaln(hip_flux2_runner *r, void *out, void *x,
                     void *shift, void *scale, int N, int dim) {
    void *args[] = {&out, &x, &shift, &scale, &N, &dim};
    hipModuleLaunchKernel(r->fn_adaln, (unsigned)N, 1, 1, 256, 1, 1,
                          256 * sizeof(float), NULL, args, NULL);
}

static void op_silu(hip_flux2_runner *r, void *x, int n) {
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(r->fn_silu, (unsigned)((n+255)/256), 1, 1, 256, 1, 1,
                          0, NULL, args, NULL);
}

static void op_bf16_trunc(hip_flux2_runner *r, void *x, int n) {
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(r->fn_bf16_trunc, (unsigned)((n+255)/256), 1, 1, 256, 1, 1,
                          0, NULL, args, NULL);
}

static void op_rmsnorm_ph(hip_flux2_runner *r, void *x, void *w,
                          int N, int n_heads, int head_dim) {
    void *args[] = {&x, &w, &N, &n_heads, &head_dim};
    hipModuleLaunchKernel(r->fn_rmsnorm_ph, (unsigned)N, (unsigned)n_heads, 1,
                          32, 1, 1, 0, NULL, args, NULL);
}

static void op_gated_add(hip_flux2_runner *r, void *x, void *proj,
                         void *gate, int N, int dim) {
    int total = N * dim;
    void *args[] = {&x, &proj, &gate, &N, &dim};
    hipModuleLaunchKernel(r->fn_gated_add, (unsigned)((total+255)/256), 1, 1, 256, 1, 1,
                          0, NULL, args, NULL);
}

static void op_attn(hip_flux2_runner *r, void *out, void *q,
                    void *k, void *v, int n_tok, int n_heads, int head_dim) {
    int fa2_warps = 4, fa2_bkv = 16;
    unsigned gy = (unsigned)((n_tok + fa2_warps - 1) / fa2_warps);
    /* RDNA4 (gfx1201) uses wave32, same as NVIDIA */
    unsigned nt = (unsigned)(32 * fa2_warps);
    size_t smem = (size_t)2 * fa2_bkv * 128 * sizeof(float);
    void *args[] = {&out, &q, &k, &v, &n_tok, &n_heads, &head_dim};
    hipModuleLaunchKernel(r->fn_flash_attn, (unsigned)n_heads, gy, 1,
                          nt, 1, 1, smem, NULL, args, NULL);
}

static void op_swiglu(hip_flux2_runner *r, void *out, void *in,
                      int n_tok, int mlp_h) {
    unsigned gx = (unsigned)((mlp_h + 255) / 256);
    void *args[] = {&out, &in, &n_tok, &mlp_h};
    hipModuleLaunchKernel(r->fn_swiglu, (unsigned)n_tok, gx, 1, 256, 1, 1,
                          0, NULL, args, NULL);
}

static void op_rope_img(hip_flux2_runner *r, void *x, int n_tok,
                        int n_heads, int hd, int lat_w, float theta) {
    unsigned gx = (unsigned)((hd/2 + 31) / 32);
    void *args[] = {&x, &n_tok, &n_heads, &hd, &lat_w, &theta};
    hipModuleLaunchKernel(r->fn_rope_img, gx, (unsigned)n_heads, (unsigned)n_tok,
                          32, 1, 1, 0, NULL, args, NULL);
}

static void op_rope_txt(hip_flux2_runner *r, void *x, int n_tok,
                        int n_heads, int hd, float theta) {
    unsigned gx = (unsigned)((hd/2 + 31) / 32);
    void *args[] = {&x, &n_tok, &n_heads, &hd, &theta};
    hipModuleLaunchKernel(r->fn_rope_txt, gx, (unsigned)n_heads, (unsigned)n_tok,
                          32, 1, 1, 0, NULL, args, NULL);
}

static void op_layernorm(hip_flux2_runner *r, void *dst, void *src,
                         int n_tok, int dim) {
    void *null_ptr = NULL;
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &null_ptr, &null_ptr, &dim, &eps};
    hipModuleLaunchKernel(r->fn_layernorm, (unsigned)n_tok, 1, 1, 256, 1, 1,
                          256 * sizeof(float), NULL, args, NULL);
}

static void op_add(hip_flux2_runner *r, void *dst, void *src, int n) {
    void *args[] = {&dst, &src, &n};
    hipModuleLaunchKernel(r->fn_add, (unsigned)((n+255)/256), 1, 1, 256, 1, 1,
                          0, NULL, args, NULL);
}

/* Sinusoidal timestep embedding (CPU, small) */
static void flux2_ts_embed(float *out, float t, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)i * logf(10000.0f) / (float)half);
        out[i] = cosf(t * freq);
        out[half + i] = sinf(t * freq);
    }
}

/* ---- Init ---- */

hip_flux2_runner *hip_flux2_init(int device_id, int verbose) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "hip_flux2: rocewInit failed (HIP/HIPRTC libraries not found)\n");
        return NULL;
    }

    HIP_CHECK_NULL(hipSetDevice(device_id));

    hipDeviceProp_t props;
    HIP_CHECK_NULL(hipGetDeviceProperties(&props, device_id));
    if (verbose >= 1)
        fprintf(stderr, "hip_flux2: device %d = %s\n", device_id, props.name);

    /* Compile kernels */
    size_t len1 = strlen(hip_kernels_common_src);
    size_t len2 = strlen(hip_flux2_specific_kernels);
    char *full = (char *)malloc(len1 + len2 + 16);
    memcpy(full, hip_kernels_common_src, len1);
    memcpy(full + len1, hip_flux2_specific_kernels, len2 + 1);

    hipModule_t mod;
    if (hip_compile_kernels(&mod, device_id, full, "flux2.hip", verbose, "hip_flux2") < 0) {
        free(full);
        fprintf(stderr, "hip_flux2: kernel compilation failed\n");
        return NULL;
    }
    free(full);

    hip_flux2_runner *r = (hip_flux2_runner *)calloc(1, sizeof(hip_flux2_runner));
    r->device_id = device_id;
    r->verbose = verbose;
    r->mod = mod;

    /* Get function handles */
    hipModuleGetFunction(&r->fn_gemm,        mod, "gemm_f32_f32");
    hipModuleGetFunction(&r->fn_layernorm,   mod, "layernorm_f32");
    hipModuleGetFunction(&r->fn_silu,        mod, "silu_f32");
    hipModuleGetFunction(&r->fn_adaln,       mod, "adaln_modulate_f32");
    hipModuleGetFunction(&r->fn_gated_add,   mod, "gated_add_f32");
    hipModuleGetFunction(&r->fn_rmsnorm_ph,  mod, "rmsnorm_per_head_f32");
    hipModuleGetFunction(&r->fn_swiglu,      mod, "flux2_swiglu");
    hipModuleGetFunction(&r->fn_flash_attn,  mod, "flash_attn_f32");
    hipModuleGetFunction(&r->fn_rope_img,    mod, "flux2_rope_img_f32");
    hipModuleGetFunction(&r->fn_rope_txt,    mod, "flux2_rope_txt_f32");
    hipModuleGetFunction(&r->fn_bf16_trunc,  mod, "truncate_bf16_f32");
    hipModuleGetFunction(&r->fn_add,         mod, "add_f32");
    /* VAE kernels */
    hipModuleGetFunction(&r->fn_vae_im2col, mod, "vae_im2col_3x3");
    hipModuleGetFunction(&r->fn_vae_add_bias, mod, "vae_add_bias");
    hipModuleGetFunction(&r->fn_vae_transpose, mod, "vae_transpose_2d");
    hipModuleGetFunction(&r->fn_vae_conv1,  mod, "vae_conv2d_1x1");
    hipModuleGetFunction(&r->fn_vae_gnsilu, mod, "vae_groupnorm_silu");
    hipModuleGetFunction(&r->fn_vae_up2x,   mod, "vae_upsample2x");
    hipModuleGetFunction(&r->fn_vae_attn,   mod, "vae_self_attn");
    hipModuleGetFunction(&r->fn_vae_bn,     mod, "vae_bn_denorm");

    if (verbose >= 1) fprintf(stderr, "hip_flux2: init OK\n");
    return r;
}

/* Upload stream block weights to GPU */
static void upload_stream(flux2_hip_stream_t *gs, const flux2_stream_block *sb, int hd) {
    gs->qkv_w    = gpu_upload_mat(&sb->qkv);
    gs->proj_w   = gpu_upload_mat(&sb->proj);
    gs->mlp_up_w = gpu_upload_mat(&sb->mlp_up);
    gs->mlp_dn_w = gpu_upload_mat(&sb->mlp_down);
    gs->q_norm   = gpu_upload_f32(sb->q_norm, hd);
    gs->k_norm   = gpu_upload_f32(sb->k_norm, hd);
}

static void free_stream(flux2_hip_stream_t *gs) {
    if (gs->qkv_w)    hipFree(gs->qkv_w);
    if (gs->proj_w)   hipFree(gs->proj_w);
    if (gs->mlp_up_w) hipFree(gs->mlp_up_w);
    if (gs->mlp_dn_w) hipFree(gs->mlp_dn_w);
    if (gs->q_norm)   hipFree(gs->q_norm);
    if (gs->k_norm)   hipFree(gs->k_norm);
}

int hip_flux2_load_dit(hip_flux2_runner *r, const char *path) {
    if (r->verbose >= 1) fprintf(stderr, "hip_flux2: loading DiT %s\n", path);

    r->dit = flux2_dit_load_safetensors(path);
    if (!r->dit) return -1;

    flux2_dit_model *m = r->dit;
    r->H = m->hidden_dim;  r->nH = m->n_heads;  r->hd = m->head_dim;
    r->n_ff = m->n_ff;     r->pin = m->patch_in_channels;
    r->txt_dim = m->txt_dim;
    r->n_dbl = m->n_double_blocks;
    r->n_sgl = m->n_single_blocks;

    if (r->verbose >= 1)
        fprintf(stderr, "hip_flux2: uploading weights to GPU (H=%d, %d+%d blocks)...\n",
                r->H, r->n_dbl, r->n_sgl);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Global weights */
    r->d_img_in_w  = gpu_upload_mat(&m->img_in);
    r->d_img_in_b  = gpu_upload_f32_or0(m->img_in_b, r->H);
    r->d_txt_in_w  = gpu_upload_mat(&m->txt_in);
    r->d_txt_in_b  = gpu_upload_f32_or0(m->txt_in_b, r->H);
    r->d_t_fc1_w   = gpu_upload_mat(&m->time_in_lin1);
    r->d_t_fc1_b   = gpu_upload_f32_or0(m->time_in_lin1_b, r->H);
    r->d_t_fc2_w   = gpu_upload_mat(&m->time_in_lin2);
    r->d_t_fc2_b   = gpu_upload_f32_or0(m->time_in_lin2_b, r->H);
    r->d_mod_img_w = gpu_upload_mat(&m->mod_img);
    r->d_mod_txt_w = gpu_upload_mat(&m->mod_txt);
    r->d_mod_sgl_w = gpu_upload_mat(&m->mod_sgl);
    r->d_out_mod_w = gpu_upload_mat(&m->out_mod);
    r->d_out_proj_w= gpu_upload_mat(&m->out_proj);

    /* Double-stream block weights */
    r->gpu_dblk = (flux2_hip_dblk_t *)calloc((size_t)r->n_dbl, sizeof(flux2_hip_dblk_t));
    for (int i = 0; i < r->n_dbl; i++) {
        upload_stream(&r->gpu_dblk[i].img, &m->dblk[i].img, r->hd);
        upload_stream(&r->gpu_dblk[i].txt, &m->dblk[i].txt, r->hd);
        if (r->verbose >= 2)
            fprintf(stderr, "\r  double block %d/%d", i+1, r->n_dbl);
    }

    /* Single-stream block weights */
    r->gpu_sblk = (flux2_hip_sblk_t *)calloc((size_t)r->n_sgl, sizeof(flux2_hip_sblk_t));
    for (int i = 0; i < r->n_sgl; i++) {
        r->gpu_sblk[i].linear1_w = gpu_upload_mat(&m->sblk[i].linear1);
        /* linear2 is [H, H+n_ff]. Split columns: first H -> attn, last n_ff -> mlp */
        {
            const flux2_mat *l2 = &m->sblk[i].linear2;
            int Hd = r->H, nf = r->n_ff;
            int l2_in = l2->cols;  /* H + n_ff */
            float *attn = (float *)malloc((size_t)Hd * Hd * sizeof(float));
            float *mlp_ = (float *)malloc((size_t)Hd * nf * sizeof(float));
            for (int r2 = 0; r2 < Hd; r2++) {
                memcpy(attn + (size_t)r2 * Hd, l2->w + (size_t)r2 * l2_in, (size_t)Hd * sizeof(float));
                memcpy(mlp_ + (size_t)r2 * nf, l2->w + (size_t)r2 * l2_in + Hd, (size_t)nf * sizeof(float));
            }
            r->gpu_sblk[i].l2_attn_w = gpu_upload_f32(attn, Hd * Hd);
            r->gpu_sblk[i].l2_mlp_w  = gpu_upload_f32(mlp_, Hd * nf);
            free(attn); free(mlp_);
        }
        r->gpu_sblk[i].q_norm = gpu_upload_f32(m->sblk[i].q_norm, r->hd);
        r->gpu_sblk[i].k_norm = gpu_upload_f32(m->sblk[i].k_norm, r->hd);
        if (r->verbose >= 2)
            fprintf(stderr, "\r  single block %d/%d", i+1, r->n_sgl);
    }

    hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (r->verbose >= 1)
        fprintf(stderr, "\nhip_flux2: weights uploaded in %.1f s\n", dt);

    r->gpu_loaded = 1;
    return 0;
}

int hip_flux2_load_vae(hip_flux2_runner *r, const char *path) {
    if (r->verbose >= 1) fprintf(stderr, "hip_flux2: loading VAE %s\n", path);
    r->vae = flux2_vae_load(path);
    return r->vae ? 0 : -1;
}

/* ---- Activation buffer allocation ---- */

static void flux2_alloc_bufs(hip_flux2_runner *r, int n_img, int n_txt) {
    int n_tot = n_img + n_txt;
    int H = r->H, n_ff = r->n_ff;

    if (r->max_tok >= n_tot) return;

    /* Free old buffers */
    if (r->d_img)       hipFree(r->d_img);
    if (r->d_txt)       hipFree(r->d_txt);
    if (r->d_joint)     hipFree(r->d_joint);
    if (r->d_temb)      hipFree(r->d_temb);
    if (r->d_temb_silu) hipFree(r->d_temb_silu);
    if (r->d_mod_img_v) hipFree(r->d_mod_img_v);
    if (r->d_mod_txt_v) hipFree(r->d_mod_txt_v);
    if (r->d_mod_sgl_v) hipFree(r->d_mod_sgl_v);
    if (r->d_q)         hipFree(r->d_q);
    if (r->d_k)         hipFree(r->d_k);
    if (r->d_v)         hipFree(r->d_v);
    if (r->d_attn_out)  hipFree(r->d_attn_out);
    if (r->d_scratch1)  hipFree(r->d_scratch1);
    if (r->d_scratch2)  hipFree(r->d_scratch2);
    if (r->d_scratch3)  hipFree(r->d_scratch3);
    if (r->d_img_in_buf) hipFree(r->d_img_in_buf);
    if (r->d_txt_in_buf) hipFree(r->d_txt_in_buf);

    size_t F = sizeof(float);
    hipMalloc(&r->d_img,        (size_t)n_img * H * F);
    hipMalloc(&r->d_txt,        (size_t)n_txt * H * F);
    hipMalloc(&r->d_joint,      (size_t)n_tot * H * F);
    hipMalloc(&r->d_temb,       (size_t)H * F);
    hipMalloc(&r->d_temb_silu,  (size_t)H * F);
    hipMalloc(&r->d_mod_img_v,  (size_t)6 * H * F);
    hipMalloc(&r->d_mod_txt_v,  (size_t)6 * H * F);
    hipMalloc(&r->d_mod_sgl_v,  (size_t)3 * H * F);
    hipMalloc(&r->d_q,          (size_t)n_tot * H * F);
    hipMalloc(&r->d_k,          (size_t)n_tot * H * F);
    hipMalloc(&r->d_v,          (size_t)n_tot * H * F);
    hipMalloc(&r->d_attn_out,   (size_t)n_tot * H * F);
    /* scratch1: for QKV/lin1 output [n_tot, max(3H, 3H+2*n_ff)] */
    int lin1_dim = 3 * H + 2 * n_ff;
    hipMalloc(&r->d_scratch1,   (size_t)n_tot * lin1_dim * F);
    /* scratch2: for MLP gate_up [n_tot, 2*n_ff] */
    hipMalloc(&r->d_scratch2,   (size_t)n_tot * 2 * n_ff * F);
    /* scratch3: for MLP out / proj [n_tot, max(n_ff, H)] */
    int s3 = n_ff > H ? n_ff : H;
    hipMalloc(&r->d_scratch3,   (size_t)n_tot * s3 * F);
    hipMalloc(&r->d_img_in_buf, (size_t)n_img * r->pin * F);
    hipMalloc(&r->d_txt_in_buf, (size_t)n_txt * r->txt_dim * F);

    r->max_tok = n_tot;
    if (r->verbose >= 1)
        fprintf(stderr, "hip_flux2: allocated activation buffers for %d tokens\n", n_tot);
}

/* ---- Pointer arithmetic helper ---- */
static inline void *ptr_offset(void *base, size_t bytes) {
    return (char *)base + bytes;
}

/* ---- GPU DiT forward pass ---- */

int hip_flux2_dit_step(hip_flux2_runner *r,
                       const float *img_tokens, int n_img,
                       const float *txt_tokens, int n_txt,
                       float timestep, float guidance, float *out) {
    (void)guidance;
    if (!r->dit || !r->gpu_loaded) {
        fprintf(stderr, "hip_flux2: DiT not loaded\n");
        return -1;
    }

    int H = r->H, nH = r->nH, hd = r->hd, n_ff = r->n_ff;
    int n_tot = n_img + n_txt;
    size_t F = sizeof(float);

    flux2_alloc_bufs(r, n_img, n_txt);

    /* Upload inputs */
    hipMemcpy(r->d_img_in_buf, img_tokens, (size_t)n_img * r->pin * F, hipMemcpyHostToDevice);
    hipMemcpy(r->d_txt_in_buf, txt_tokens, (size_t)n_txt * r->txt_dim * F, hipMemcpyHostToDevice);

    /* 1. Timestep embedding (computed on CPU, uploaded) */
    float t_raw[256];
    flux2_ts_embed(t_raw, timestep * 1000.0f, 256);
    void *d_traw = gpu_upload_f32(t_raw, 256);

    /* temb = SiLU(time_in_lin1 @ t_raw + b1) */
    op_gemm(r, r->d_temb, r->d_t_fc1_w, d_traw, r->d_t_fc1_b, H, 256, 1);
    op_silu(r, r->d_temb, H);
    /* temb = time_in_lin2 @ temb + b2 */
    op_gemm(r, r->d_temb_silu, r->d_t_fc2_w, r->d_temb, r->d_t_fc2_b, H, H, 1);
    void *d_temb_final = r->d_temb_silu;

    hipFree(d_traw);

    /* 2. Project img and txt tokens */
    op_gemm(r, r->d_img, r->d_img_in_w, r->d_img_in_buf, r->d_img_in_b,
            H, r->pin, n_img);
    op_gemm(r, r->d_txt, r->d_txt_in_w, r->d_txt_in_buf, r->d_txt_in_b,
            H, r->txt_dim, n_txt);

    /* 3. Compute global modulations: SiLU(temb) -> mod vectors */
    hipMemcpy(r->d_temb, d_temb_final, (size_t)H * F, hipMemcpyDeviceToDevice);
    op_silu(r, r->d_temb, H);
    void *d0 = NULL;
    op_gemm(r, r->d_mod_img_v, r->d_mod_img_w, r->d_temb, d0, 6*H, H, 1);
    op_gemm(r, r->d_mod_txt_v, r->d_mod_txt_w, r->d_temb, d0, 6*H, H, 1);
    op_gemm(r, r->d_mod_sgl_v, r->d_mod_sgl_w, r->d_temb, d0, 3*H, H, 1);

    /* Modulation layout: [shift_attn(H), scale_attn(H), gate_attn(H),
     *                     shift_ffn(H), scale_ffn(H), gate_ffn(H)] */
    void *mi_shift_a = r->d_mod_img_v;
    void *mi_scale_a = ptr_offset(r->d_mod_img_v, (size_t)H * F);
    void *mi_gate_a  = ptr_offset(r->d_mod_img_v, (size_t)2*H * F);
    void *mi_shift_f = ptr_offset(r->d_mod_img_v, (size_t)3*H * F);
    void *mi_scale_f = ptr_offset(r->d_mod_img_v, (size_t)4*H * F);
    void *mi_gate_f  = ptr_offset(r->d_mod_img_v, (size_t)5*H * F);

    void *mt_shift_a = r->d_mod_txt_v;
    void *mt_scale_a = ptr_offset(r->d_mod_txt_v, (size_t)H * F);
    void *mt_gate_a  = ptr_offset(r->d_mod_txt_v, (size_t)2*H * F);
    void *mt_shift_f = ptr_offset(r->d_mod_txt_v, (size_t)3*H * F);
    void *mt_scale_f = ptr_offset(r->d_mod_txt_v, (size_t)4*H * F);
    void *mt_gate_f  = ptr_offset(r->d_mod_txt_v, (size_t)5*H * F);

    void *ms_shift = r->d_mod_sgl_v;
    void *ms_scale = ptr_offset(r->d_mod_sgl_v, (size_t)H * F);
    void *ms_gate  = ptr_offset(r->d_mod_sgl_v, (size_t)2*H * F);

    /* Compute lat_w for RoPE */
    int lat_w_p = (int)sqrtf((float)n_img);
    if (lat_w_p * lat_w_p != n_img) lat_w_p = n_img;
    float theta = FLUX2_ROPE_THETA;

    /* ---- Double-stream blocks ---- */
    for (int bi = 0; bi < r->n_dbl; bi++) {
        flux2_hip_dblk_t *b = &r->gpu_dblk[bi];

        /* IMG stream: adaLN -> QKV */
        op_adaln(r, r->d_scratch1, r->d_img, mi_shift_a, mi_scale_a, n_img, H);
        void *img_q_w = b->img.qkv_w;
        void *img_k_w = ptr_offset(b->img.qkv_w, (size_t)H * H * 4);
        void *img_v_w = ptr_offset(b->img.qkv_w, (size_t)2 * H * H * 4);
        op_gemm(r, r->d_q, img_q_w, r->d_scratch1, d0, H, H, n_img);
        op_gemm(r, r->d_k, img_k_w, r->d_scratch1, d0, H, H, n_img);
        op_gemm(r, r->d_v, img_v_w, r->d_scratch1, d0, H, H, n_img);

        /* TXT stream: adaLN -> QKV */
        op_adaln(r, r->d_scratch1, r->d_txt, mt_shift_a, mt_scale_a, n_txt, H);
        void *txt_q_w = b->txt.qkv_w;
        void *txt_k_w = ptr_offset(b->txt.qkv_w, (size_t)H * H * 4);
        void *txt_v_w = ptr_offset(b->txt.qkv_w, (size_t)2 * H * H * 4);
        void *tq = ptr_offset(r->d_q, (size_t)n_img * H * F);
        void *tk = ptr_offset(r->d_k, (size_t)n_img * H * F);
        void *tv = ptr_offset(r->d_v, (size_t)n_img * H * F);
        op_gemm(r, tq, txt_q_w, r->d_scratch1, d0, H, H, n_txt);
        op_gemm(r, tk, txt_k_w, r->d_scratch1, d0, H, H, n_txt);
        op_gemm(r, tv, txt_v_w, r->d_scratch1, d0, H, H, n_txt);

        /* Per-head RMSNorm on Q, K */
        op_rmsnorm_ph(r, r->d_q, b->img.q_norm, n_img, nH, hd);
        op_rmsnorm_ph(r, r->d_k, b->img.k_norm, n_img, nH, hd);
        op_rmsnorm_ph(r, tq, b->txt.q_norm, n_txt, nH, hd);
        op_rmsnorm_ph(r, tk, b->txt.k_norm, n_txt, nH, hd);

        /* RoPE: img=2D spatial, txt=1D sequence */
        op_rope_img(r, r->d_q, n_img, nH, hd, lat_w_p, theta);
        op_rope_img(r, r->d_k, n_img, nH, hd, lat_w_p, theta);
        op_rope_txt(r, tq, n_txt, nH, hd, theta);
        op_rope_txt(r, tk, n_txt, nH, hd, theta);

        /* Joint attention: Q=[img,txt], K=[img,txt], V=[img,txt] */
        op_attn(r, r->d_attn_out, r->d_q, r->d_k, r->d_v, n_tot, nH, hd);

        /* Output projections */
        op_gemm(r, r->d_scratch1, b->img.proj_w, r->d_attn_out, d0, H, H, n_img);
        void *txt_attn = ptr_offset(r->d_attn_out, (size_t)n_img * H * F);
        op_gemm(r, r->d_scratch2, b->txt.proj_w, txt_attn, d0, H, H, n_txt);

        /* Gated residual */
        op_gated_add(r, r->d_img, r->d_scratch1, mi_gate_a, n_img, H);
        op_gated_add(r, r->d_txt, r->d_scratch2, mt_gate_a, n_txt, H);

        /* FFN img */
        op_adaln(r, r->d_scratch1, r->d_img, mi_shift_f, mi_scale_f, n_img, H);
        op_gemm(r, r->d_scratch2, b->img.mlp_up_w, r->d_scratch1, d0, 2*n_ff, H, n_img);
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_img, n_ff);
        op_gemm(r, r->d_scratch1, b->img.mlp_dn_w, r->d_scratch3, d0, H, n_ff, n_img);
        op_gated_add(r, r->d_img, r->d_scratch1, mi_gate_f, n_img, H);

        /* FFN txt */
        op_adaln(r, r->d_scratch1, r->d_txt, mt_shift_f, mt_scale_f, n_txt, H);
        op_gemm(r, r->d_scratch2, b->txt.mlp_up_w, r->d_scratch1, d0, 2*n_ff, H, n_txt);
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_txt, n_ff);
        op_gemm(r, r->d_scratch1, b->txt.mlp_dn_w, r->d_scratch3, d0, H, n_ff, n_txt);
        op_gated_add(r, r->d_txt, r->d_scratch1, mt_gate_f, n_txt, H);

        if (r->verbose >= 2)
            fprintf(stderr, "\r  dit double block %d/%d", bi+1, r->n_dbl);
    }

    /* ---- Single-stream blocks ---- */
    /* Concatenate: joint = [txt, img] (txt first, then img) */
    hipMemcpy(r->d_joint, r->d_txt, (size_t)n_txt * H * F, hipMemcpyDeviceToDevice);
    hipMemcpy(ptr_offset(r->d_joint, (size_t)n_txt * H * F), r->d_img,
              (size_t)n_img * H * F, hipMemcpyDeviceToDevice);

    for (int bi = 0; bi < r->n_sgl; bi++) {
        flux2_hip_sblk_t *b = &r->gpu_sblk[bi];

        /* adaLN */
        op_adaln(r, r->d_scratch1, r->d_joint, ms_shift, ms_scale, n_tot, H);

        /* linear1: Q, K, V, gate_up via weight slices */
        void *q_w  = b->linear1_w;
        void *k_w  = ptr_offset(b->linear1_w, (size_t)H * H * 4);
        void *v_w  = ptr_offset(b->linear1_w, (size_t)2 * H * H * 4);
        void *gu_w = ptr_offset(b->linear1_w, (size_t)3 * H * H * 4);

        op_gemm(r, r->d_q, q_w, r->d_scratch1, d0, H, H, n_tot);
        op_gemm(r, r->d_k, k_w, r->d_scratch1, d0, H, H, n_tot);
        op_gemm(r, r->d_v, v_w, r->d_scratch1, d0, H, H, n_tot);
        op_gemm(r, r->d_scratch2, gu_w, r->d_scratch1, d0, 2*n_ff, H, n_tot);

        /* Per-head Q/K norm + RoPE */
        op_rmsnorm_ph(r, r->d_q, b->q_norm, n_tot, nH, hd);
        op_rmsnorm_ph(r, r->d_k, b->k_norm, n_tot, nH, hd);
        /* txt tokens at front [0..n_txt), img after */
        op_rope_txt(r, r->d_q, n_txt, nH, hd, theta);
        op_rope_txt(r, r->d_k, n_txt, nH, hd, theta);
        void *q_img = ptr_offset(r->d_q, (size_t)n_txt * H * F);
        void *k_img = ptr_offset(r->d_k, (size_t)n_txt * H * F);
        op_rope_img(r, q_img, n_img, nH, hd, lat_w_p, theta);
        op_rope_img(r, k_img, n_img, nH, hd, lat_w_p, theta);

        /* Self-attention */
        op_attn(r, r->d_attn_out, r->d_q, r->d_k, r->d_v, n_tot, nH, hd);

        /* Parallel MLP: SwiGLU on gate_up */
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_tot, n_ff);

        /* linear2 split: attn + mlp */
        op_gemm(r, r->d_scratch1, b->l2_attn_w, r->d_attn_out, d0, H, H, n_tot);
        op_gemm(r, r->d_scratch2, b->l2_mlp_w, r->d_scratch3, d0, H, n_ff, n_tot);

        /* Add the two halves */
        op_add(r, r->d_scratch1, r->d_scratch2, n_tot * H);

        /* Gated residual */
        op_gated_add(r, r->d_joint, r->d_scratch1, ms_gate, n_tot, H);

        if (r->verbose >= 2)
            fprintf(stderr, "\r  dit single block %d/%d", bi+1, r->n_sgl);
    }

    /* ---- Output ---- */
    void *d_img_out = ptr_offset(r->d_joint, (size_t)n_txt * H * F);

    /* Final adaLN */
    op_gemm(r, r->d_scratch1, r->d_out_mod_w, r->d_temb, d0, 2*H, H, 1);
    void *out_shift = r->d_scratch1;
    void *out_scale = ptr_offset(r->d_scratch1, (size_t)H * F);
    op_adaln(r, r->d_scratch2, d_img_out, out_shift, out_scale, n_img, H);

    /* Final linear */
    op_gemm(r, r->d_attn_out, r->d_out_proj_w, r->d_scratch2, d0, r->pin, H, n_img);

    /* Download result */
    hipMemcpy(out, r->d_attn_out, (size_t)n_img * r->pin * F, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    if (r->verbose >= 2)
        fprintf(stderr, "\nhip_flux2: dit step done\n");

    return 0;
}

/* ---- Free ---- */

void hip_flux2_free(hip_flux2_runner *r) {
    if (!r) return;

    if (r->gpu_dblk) {
        for (int i = 0; i < r->n_dbl; i++) {
            free_stream(&r->gpu_dblk[i].img);
            free_stream(&r->gpu_dblk[i].txt);
        }
        free(r->gpu_dblk);
    }
    if (r->gpu_sblk) {
        for (int i = 0; i < r->n_sgl; i++) {
            if (r->gpu_sblk[i].linear1_w) hipFree(r->gpu_sblk[i].linear1_w);
            if (r->gpu_sblk[i].l2_attn_w) hipFree(r->gpu_sblk[i].l2_attn_w);
            if (r->gpu_sblk[i].l2_mlp_w)  hipFree(r->gpu_sblk[i].l2_mlp_w);
            if (r->gpu_sblk[i].q_norm)    hipFree(r->gpu_sblk[i].q_norm);
            if (r->gpu_sblk[i].k_norm)    hipFree(r->gpu_sblk[i].k_norm);
        }
        free(r->gpu_sblk);
    }

    if (r->d_img_in_w)  hipFree(r->d_img_in_w);
    if (r->d_img_in_b)  hipFree(r->d_img_in_b);
    if (r->d_txt_in_w)  hipFree(r->d_txt_in_w);
    if (r->d_txt_in_b)  hipFree(r->d_txt_in_b);
    if (r->d_t_fc1_w)   hipFree(r->d_t_fc1_w);
    if (r->d_t_fc1_b)   hipFree(r->d_t_fc1_b);
    if (r->d_t_fc2_w)   hipFree(r->d_t_fc2_w);
    if (r->d_t_fc2_b)   hipFree(r->d_t_fc2_b);
    if (r->d_mod_img_w) hipFree(r->d_mod_img_w);
    if (r->d_mod_txt_w) hipFree(r->d_mod_txt_w);
    if (r->d_mod_sgl_w) hipFree(r->d_mod_sgl_w);
    if (r->d_out_mod_w) hipFree(r->d_out_mod_w);
    if (r->d_out_proj_w)hipFree(r->d_out_proj_w);

    if (r->d_img)       hipFree(r->d_img);
    if (r->d_txt)       hipFree(r->d_txt);
    if (r->d_joint)     hipFree(r->d_joint);
    if (r->d_temb)      hipFree(r->d_temb);
    if (r->d_temb_silu) hipFree(r->d_temb_silu);
    if (r->d_mod_img_v) hipFree(r->d_mod_img_v);
    if (r->d_mod_txt_v) hipFree(r->d_mod_txt_v);
    if (r->d_mod_sgl_v) hipFree(r->d_mod_sgl_v);
    if (r->d_q)         hipFree(r->d_q);
    if (r->d_k)         hipFree(r->d_k);
    if (r->d_v)         hipFree(r->d_v);
    if (r->d_attn_out)  hipFree(r->d_attn_out);
    if (r->d_scratch1)  hipFree(r->d_scratch1);
    if (r->d_scratch2)  hipFree(r->d_scratch2);
    if (r->d_scratch3)  hipFree(r->d_scratch3);
    if (r->d_img_in_buf)hipFree(r->d_img_in_buf);
    if (r->d_txt_in_buf)hipFree(r->d_txt_in_buf);

    if (r->dit) flux2_dit_free(r->dit);
    if (r->vae) flux2_vae_free(r->vae);
    if (r->mod) hipModuleUnload(r->mod);
    free(r);
}

/* ---- GPU VAE helpers ---- */

static void *vae_alloc(size_t n) {
    void *d = NULL;
    hipMalloc(&d, n);
    return d;
}

/* Conv2d 3x3 via im2col + GEMM:
 *   im2col: [ci,H,W] -> col[H*W, ci*9]
 *   GEMM:   out_t[H*W, co] = col[H*W, ci*9] @ w[co, ci*9]^T
 *   Then transpose out_t[H*W, co] -> out[co, H*W] and add bias.
 * Since our GEMM does Y[n_tok, n_out] = X[n_tok, n_in] @ W[n_out, n_in]^T,
 * we set n_tok=H*W, n_in=ci*9, n_out=co.
 * Output is [H*W, co] which we need as [co, H*W]. For same-size output this
 * is a transpose. We use a simple transpose kernel or work in col-major.
 *
 * Actually our GEMM writes Y[tok, out] = row-major, which IS [H*W, co].
 * But the conv expects NCHW output [co, H*W]. So we need a transpose after.
 * Alternative: reshape the weight to match. W_conv is [co, ci, 3, 3] =
 * [co, ci*9] row-major. Our GEMM does Y = X @ W^T where W is [n_out, n_in].
 * X = col[H*W, ci*9], W = w_conv[co, ci*9].
 * Y = col @ W^T = [H*W, co]. Need to transpose to [co, H*W].
 */
static void vae_conv3(hip_flux2_runner *r, void *out, void *in,
                      void *w, void *bias, int ci, int H, int W, int co) {
    int spatial = H * W;
    int col_w = ci * 9;
    size_t F = sizeof(float);

    /* 1. im2col */
    void *d_col = vae_alloc((size_t)spatial * col_w * F);
    {
        int total = spatial * col_w;
        void *args[] = {&d_col, &in, &ci, &H, &W};
        hipModuleLaunchKernel(r->fn_vae_im2col, (unsigned)((total+255)/256), 1, 1,
                              256, 1, 1, 0, NULL, args, NULL);
    }

    /* 2. GEMM: Y[H*W, co] = col[H*W, ci*9] @ w[co, ci*9]^T */
    void *d_yt = vae_alloc((size_t)spatial * co * F);
    {
        void *null_ptr = NULL;
        op_gemm(r, d_yt, w, d_col, null_ptr, co, col_w, spatial);
    }
    hipFree(d_col);

    /* 3. Transpose [H*W, co] -> [co, H*W] = out (GPU) */
    {
        int total = spatial * co;
        void *args[] = {&out, &d_yt, &spatial, &co};
        hipModuleLaunchKernel(r->fn_vae_transpose, (unsigned)((total+255)/256), 1, 1,
                              256, 1, 1, 0, NULL, args, NULL);
        hipFree(d_yt);
    }

    /* 4. Add bias */
    if (bias) {
        int total = co * spatial;
        void *args[] = {&out, &bias, &co, &spatial};
        hipModuleLaunchKernel(r->fn_vae_add_bias, (unsigned)((total+255)/256), 1, 1,
                              256, 1, 1, 0, NULL, args, NULL);
    }
}

static void vae_conv1(hip_flux2_runner *r, void *out, void *in,
                      void *w, void *bias, int ci, int spatial, int co) {
    int total = co * spatial;
    void *args[] = {&out, &in, &w, &bias, &ci, &spatial, &co};
    hipModuleLaunchKernel(r->fn_vae_conv1, (unsigned)((total+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, args, NULL);
}

static void vae_groupnorm_silu(hip_flux2_runner *r, void *out, void *in,
                               void *gamma, void *beta,
                               int C, int spatial, int num_groups, int do_silu) {
    void *args[] = {&out, &in, &gamma, &beta, &C, &spatial, &num_groups, &do_silu};
    hipModuleLaunchKernel(r->fn_vae_gnsilu, (unsigned)num_groups, 1, 1,
                          256, 1, 1, 256 * sizeof(float), NULL, args, NULL);
}

static void vae_upsample2x(hip_flux2_runner *r, void *out, void *in,
                            int C, int H, int W) {
    int total = C * H * W;
    void *args[] = {&out, &in, &C, &H, &W};
    hipModuleLaunchKernel(r->fn_vae_up2x, (unsigned)((total+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, args, NULL);
}

/* VAE ResBlock forward on GPU */
static void vae_resblock_gpu(hip_flux2_runner *r,
                             void *out, void *x,
                             void *n1w, void *n1b, void *c1w, void *c1b,
                             void *n2w, void *n2b, void *c2w, void *c2b,
                             void *skipw, void *skipb,
                             int ci, int co, int H, int W, int ng) {
    int spatial = H * W;
    /* tmp1 = GroupNorm+SiLU(x) */
    void *tmp1 = vae_alloc((size_t)ci * spatial * sizeof(float));
    vae_groupnorm_silu(r, tmp1, x, n1w, n1b, ci, spatial, ng, 1);
    /* tmp2 = Conv3x3(tmp1) [ci -> co] */
    void *tmp2 = vae_alloc((size_t)co * spatial * sizeof(float));
    vae_conv3(r, tmp2, tmp1, c1w, c1b, ci, H, W, co);
    hipFree(tmp1);
    /* tmp1 = GroupNorm+SiLU(tmp2) */
    tmp1 = vae_alloc((size_t)co * spatial * sizeof(float));
    vae_groupnorm_silu(r, tmp1, tmp2, n2w, n2b, co, spatial, ng, 1);
    /* tmp2 = Conv3x3(tmp1) [co -> co] */
    hipFree(tmp2);
    tmp2 = vae_alloc((size_t)co * spatial * sizeof(float));
    vae_conv3(r, tmp2, tmp1, c2w, c2b, co, H, W, co);
    hipFree(tmp1);
    /* skip connection */
    if (skipw) {
        vae_conv1(r, out, x, skipw, skipb, ci, spatial, co);
    } else {
        hipMemcpy(out, x, (size_t)co * spatial * sizeof(float), hipMemcpyDeviceToDevice);
    }
    /* out += tmp2 */
    op_add(r, out, tmp2, co * spatial);
    hipFree(tmp2);
}

/* VAE mid-block attention on GPU */
static void vae_mid_attn_gpu(hip_flux2_runner *r,
                             void *out, void *x,
                             void *nw, void *nb,
                             void *qw, void *qb, void *kw, void *kb,
                             void *vw, void *vb, void *ow, void *ob,
                             int c, int H, int W, int ng) {
    int spatial = H * W;
    size_t sz = (size_t)c * spatial * sizeof(float);
    /* GroupNorm (no SiLU) */
    void *normed = vae_alloc(sz);
    vae_groupnorm_silu(r, normed, x, nw, nb, c, spatial, ng, 0);

    /* Q/K/V projections: treat as [spatial, c] @ [c, c]^T
     * But data is [c, spatial] (channels-first). For the 1x1 conv approach:
     * Q = conv1x1(normed, qw, qb) → [c, spatial] */
    void *dQ = vae_alloc(sz);
    void *dK = vae_alloc(sz);
    void *dV = vae_alloc(sz);
    vae_conv1(r, dQ, normed, qw, qb, c, spatial, c);
    vae_conv1(r, dK, normed, kw, kb, c, spatial, c);
    vae_conv1(r, dV, normed, vw, vb, c, spatial, c);
    hipFree(normed);

    /* Transpose [c, spatial] -> [spatial, c] for attention */
    /* Do on CPU for simplicity (mid-block is small: spatial=H*W at lowest res) */
    float *hQ = (float *)malloc(sz);
    float *hK = (float *)malloc(sz);
    float *hV = (float *)malloc(sz);
    float *hO = (float *)malloc(sz);
    hipMemcpy(hQ, dQ, sz, hipMemcpyDeviceToHost);
    hipMemcpy(hK, dK, sz, hipMemcpyDeviceToHost);
    hipMemcpy(hV, dV, sz, hipMemcpyDeviceToHost);
    hipFree(dQ); hipFree(dK); hipFree(dV);

    /* Transpose CHW -> HWC */
    float *tQ = (float *)malloc(sz);
    float *tK = (float *)malloc(sz);
    float *tV = (float *)malloc(sz);
    for (int ch = 0; ch < c; ch++)
        for (int s = 0; s < spatial; s++) {
            tQ[s * c + ch] = hQ[ch * spatial + s];
            tK[s * c + ch] = hK[ch * spatial + s];
            tV[s * c + ch] = hV[ch * spatial + s];
        }
    free(hQ); free(hK); free(hV);

    /* Upload transposed Q/K/V */
    dQ = gpu_upload_f32(tQ, spatial * c);
    dK = gpu_upload_f32(tK, spatial * c);
    dV = gpu_upload_f32(tV, spatial * c);
    free(tQ); free(tK); free(tV);

    /* Self-attention kernel */
    void *dO = vae_alloc(sz);
    float scale = 1.0f / sqrtf((float)c);
    size_t smem = ((size_t)spatial + 256) * sizeof(float);
    void *args[] = {&dO, &dQ, &dK, &dV, &spatial, &c, &scale};
    hipModuleLaunchKernel(r->fn_vae_attn, (unsigned)spatial, 1, 1,
                          256, 1, 1, smem, NULL, args, NULL);
    hipFree(dQ); hipFree(dK); hipFree(dV);

    /* Transpose back HWC -> CHW */
    hipMemcpy(hO, dO, sz, hipMemcpyDeviceToHost);
    hipFree(dO);
    float *tO = (float *)malloc(sz);
    for (int s = 0; s < spatial; s++)
        for (int ch = 0; ch < c; ch++)
            tO[ch * spatial + s] = hO[s * c + ch];
    free(hO);

    /* Output projection: conv1x1 */
    void *dTO = gpu_upload_f32(tO, c * spatial);
    free(tO);
    void *proj_out = vae_alloc(sz);
    vae_conv1(r, proj_out, dTO, ow, ob, c, spatial, c);
    hipFree(dTO);

    /* Residual: out = x + proj_out */
    hipMemcpy(out, x, sz, hipMemcpyDeviceToDevice);
    op_add(r, out, proj_out, c * spatial);
    hipFree(proj_out);
}

/* Upload VAE resblock weights */
typedef struct {
    void *n1w, *n1b, *c1w, *c1b;
    void *n2w, *n2b, *c2w, *c2b;
    void *sw, *sb;
    int ci, co;
} vae_gpu_resblock;

static vae_gpu_resblock upload_resblock(const flux2_vae_resblock *rb) {
    vae_gpu_resblock g = {0};
    g.ci = rb->c_in; g.co = rb->c_out;
    g.n1w = gpu_upload_f32(rb->norm1_w, rb->c_in);
    g.n1b = gpu_upload_f32(rb->norm1_b, rb->c_in);
    g.c1w = gpu_upload_f32(rb->conv1_w, rb->c_out * rb->c_in * 9);
    g.c1b = gpu_upload_f32(rb->conv1_b, rb->c_out);
    g.n2w = gpu_upload_f32(rb->norm2_w, rb->c_out);
    g.n2b = gpu_upload_f32(rb->norm2_b, rb->c_out);
    g.c2w = gpu_upload_f32(rb->conv2_w, rb->c_out * rb->c_out * 9);
    g.c2b = gpu_upload_f32(rb->conv2_b, rb->c_out);
    if (rb->skip_w) {
        g.sw = gpu_upload_f32(rb->skip_w, rb->c_out * rb->c_in);
        g.sb = gpu_upload_f32(rb->skip_b, rb->c_out);
    }
    return g;
}

typedef struct {
    void *nw, *nb, *qw, *qb, *kw, *kb, *vw, *vb, *ow, *ob;
    int c;
} vae_gpu_attn;

static vae_gpu_attn upload_attn(const flux2_vae_attn *a) {
    vae_gpu_attn g = {0};
    g.c = a->c;
    g.nw = gpu_upload_f32(a->norm_w, a->c);
    g.nb = gpu_upload_f32(a->norm_b, a->c);
    g.qw = gpu_upload_f32(a->q_w, a->c * a->c);
    g.qb = gpu_upload_f32(a->q_b, a->c);
    g.kw = gpu_upload_f32(a->k_w, a->c * a->c);
    g.kb = gpu_upload_f32(a->k_b, a->c);
    g.vw = gpu_upload_f32(a->v_w, a->c * a->c);
    g.vb = gpu_upload_f32(a->v_b, a->c);
    g.ow = gpu_upload_f32(a->out_w, a->c * a->c);
    g.ob = gpu_upload_f32(a->out_b, a->c);
    return g;
}

/* ---- GPU VAE decode ---- */
int hip_flux2_vae_decode(hip_flux2_runner *r,
                         const float *latent, int lat_h, int lat_w,
                         float *out_rgb) {
    if (!r->vae) {
        fprintf(stderr, "hip_flux2: VAE not loaded\n");
        return -1;
    }
    flux2_vae_model *m = r->vae;
    int lc = m->latent_channels;
    int ng = m->num_groups;
    int h = lat_h, w = lat_w;
    size_t F = sizeof(float);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Upload VAE weights to GPU (lazy, once) */
    if (!r->vae_gpu_loaded) {
        if (r->verbose >= 1) fprintf(stderr, "hip_flux2: uploading VAE weights to GPU...\n");
        r->vae_gpu_loaded = 1;
    }

    /* Upload latent */
    void *d_lat = gpu_upload_f32(latent, lc * lat_h * lat_w);

    /* Stage 0: Batch-norm denormalize */
    void *d_x;
    if (m->bn_mean && m->bn_var) {
        void *d_bn_mean = gpu_upload_f32(m->bn_mean, m->bn_n_ch);
        void *d_bn_var  = gpu_upload_f32(m->bn_var, m->bn_n_ch);
        d_x = vae_alloc((size_t)lc * h * w * F);
        int total = lc * h * w;
        int ps = 2;
        float eps = m->bn_eps;
        void *args[] = {&d_x, &d_lat, &d_bn_mean, &d_bn_var, &eps, &lc, &h, &w, &ps};
        hipModuleLaunchKernel(r->fn_vae_bn, (unsigned)((total+255)/256), 1, 1,
                              256, 1, 1, 0, NULL, args, NULL);
        hipFree(d_lat); hipFree(d_bn_mean); hipFree(d_bn_var);
    } else {
        d_x = d_lat;
    }

    /* Stage 1: post_quant_conv (1x1) */
    if (m->pqc_w) {
        void *d_pqcw = gpu_upload_f32(m->pqc_w, lc * lc);
        void *d_pqcb = gpu_upload_f32(m->pqc_b, lc);
        void *d_tmp = vae_alloc((size_t)lc * h * w * F);
        vae_conv1(r, d_tmp, d_x, d_pqcw, d_pqcb, lc, h * w, lc);
        hipFree(d_x); hipFree(d_pqcw); hipFree(d_pqcb);
        d_x = d_tmp;
    }

    /* Stage 2: conv_in (3x3, lc -> c) */
    int c = m->conv_in_out_ch;
    {
        void *d_ciw = gpu_upload_f32(m->conv_in_w, c * lc * 9);
        void *d_cib = gpu_upload_f32(m->conv_in_b, c);
        void *d_tmp = vae_alloc((size_t)c * h * w * F);
        vae_conv3(r, d_tmp, d_x, d_ciw, d_cib, lc, h, w, c);
        hipFree(d_x); hipFree(d_ciw); hipFree(d_cib);
        d_x = d_tmp;
    }

    if (r->verbose >= 1) fprintf(stderr, "hip_flux2: VAE conv_in done [%d, %d, %d]\n", c, h, w);

    /* Stage 3: mid_block */
    {
        vae_gpu_resblock gr0 = upload_resblock(&m->mid_res0);
        void *d_tmp = vae_alloc((size_t)c * h * w * F);
        vae_resblock_gpu(r, d_tmp, d_x, gr0.n1w, gr0.n1b, gr0.c1w, gr0.c1b,
                         gr0.n2w, gr0.n2b, gr0.c2w, gr0.c2b, gr0.sw, gr0.sb,
                         gr0.ci, gr0.co, h, w, ng);
        hipFree(d_x); d_x = d_tmp;
        /* Free resblock weights */
        hipFree(gr0.n1w); hipFree(gr0.n1b); hipFree(gr0.c1w); hipFree(gr0.c1b);
        hipFree(gr0.n2w); hipFree(gr0.n2b); hipFree(gr0.c2w); hipFree(gr0.c2b);
        if (gr0.sw) { hipFree(gr0.sw); hipFree(gr0.sb); }
    }
    {
        vae_gpu_attn ga = upload_attn(&m->mid_attn);
        void *d_tmp = vae_alloc((size_t)c * h * w * F);
        vae_mid_attn_gpu(r, d_tmp, d_x, ga.nw, ga.nb,
                         ga.qw, ga.qb, ga.kw, ga.kb, ga.vw, ga.vb, ga.ow, ga.ob,
                         c, h, w, ng);
        hipFree(d_x); d_x = d_tmp;
        hipFree(ga.nw); hipFree(ga.nb); hipFree(ga.qw); hipFree(ga.qb);
        hipFree(ga.kw); hipFree(ga.kb); hipFree(ga.vw); hipFree(ga.vb);
        hipFree(ga.ow); hipFree(ga.ob);
    }
    {
        vae_gpu_resblock gr1 = upload_resblock(&m->mid_res1);
        void *d_tmp = vae_alloc((size_t)c * h * w * F);
        vae_resblock_gpu(r, d_tmp, d_x, gr1.n1w, gr1.n1b, gr1.c1w, gr1.c1b,
                         gr1.n2w, gr1.n2b, gr1.c2w, gr1.c2b, gr1.sw, gr1.sb,
                         gr1.ci, gr1.co, h, w, ng);
        hipFree(d_x); d_x = d_tmp;
        hipFree(gr1.n1w); hipFree(gr1.n1b); hipFree(gr1.c1w); hipFree(gr1.c1b);
        hipFree(gr1.n2w); hipFree(gr1.n2b); hipFree(gr1.c2w); hipFree(gr1.c2b);
        if (gr1.sw) { hipFree(gr1.sw); hipFree(gr1.sb); }
    }

    if (r->verbose >= 1) fprintf(stderr, "hip_flux2: VAE mid_block done\n");

    /* Stage 4: up_blocks */
    for (int bi = 0; bi < 4; bi++) {
        /* resnets.0 */
        {
            vae_gpu_resblock gr = upload_resblock(&m->up_res[bi][0]);
            int new_c = gr.co;
            void *d_tmp = vae_alloc((size_t)new_c * h * w * F);
            vae_resblock_gpu(r, d_tmp, d_x, gr.n1w, gr.n1b, gr.c1w, gr.c1b,
                             gr.n2w, gr.n2b, gr.c2w, gr.c2b, gr.sw, gr.sb,
                             gr.ci, gr.co, h, w, ng);
            hipFree(d_x); d_x = d_tmp; c = new_c;
            hipFree(gr.n1w); hipFree(gr.n1b); hipFree(gr.c1w); hipFree(gr.c1b);
            hipFree(gr.n2w); hipFree(gr.n2b); hipFree(gr.c2w); hipFree(gr.c2b);
            if (gr.sw) { hipFree(gr.sw); hipFree(gr.sb); }
        }
        /* resnets.1 */
        {
            vae_gpu_resblock gr = upload_resblock(&m->up_res[bi][1]);
            int new_c = gr.co;
            void *d_tmp = vae_alloc((size_t)new_c * h * w * F);
            vae_resblock_gpu(r, d_tmp, d_x, gr.n1w, gr.n1b, gr.c1w, gr.c1b,
                             gr.n2w, gr.n2b, gr.c2w, gr.c2b, gr.sw, gr.sb,
                             gr.ci, gr.co, h, w, ng);
            hipFree(d_x); d_x = d_tmp; c = new_c;
            hipFree(gr.n1w); hipFree(gr.n1b); hipFree(gr.c1w); hipFree(gr.c1b);
            hipFree(gr.n2w); hipFree(gr.n2b); hipFree(gr.c2w); hipFree(gr.c2b);
            if (gr.sw) { hipFree(gr.sw); hipFree(gr.sb); }
        }
        /* Upsample */
        if (m->up_has_sample[bi]) {
            void *d_up = vae_alloc((size_t)c * 2*h * 2*w * F);
            vae_upsample2x(r, d_up, d_x, c, h, w);
            h *= 2; w *= 2;
            void *d_usw = gpu_upload_f32(m->up_sample[bi].conv_w, c * c * 9);
            void *d_usb = gpu_upload_f32(m->up_sample[bi].conv_b, c);
            void *d_tmp = vae_alloc((size_t)c * h * w * F);
            vae_conv3(r, d_tmp, d_up, d_usw, d_usb, c, h, w, c);
            hipFree(d_up); hipFree(d_x); hipFree(d_usw); hipFree(d_usb);
            d_x = d_tmp;
        }
        if (r->verbose >= 1)
            fprintf(stderr, "hip_flux2: VAE up_block[%d] done [%d, %d, %d]\n", bi, c, h, w);
    }

    /* Stage 5: norm_out -> SiLU -> conv_out */
    {
        void *d_now = gpu_upload_f32(m->norm_out_w, c);
        void *d_nob = gpu_upload_f32(m->norm_out_b, c);
        void *d_normed = vae_alloc((size_t)c * h * w * F);
        vae_groupnorm_silu(r, d_normed, d_x, d_now, d_nob, c, h*w, ng, 1);
        hipFree(d_x); hipFree(d_now); hipFree(d_nob);

        void *d_cow = gpu_upload_f32(m->conv_out_w, 3 * c * 9);
        void *d_cob = gpu_upload_f32(m->conv_out_b, 3);
        void *d_rgb = vae_alloc((size_t)3 * h * w * F);
        vae_conv3(r, d_rgb, d_normed, d_cow, d_cob, c, h, w, 3);
        hipFree(d_normed); hipFree(d_cow); hipFree(d_cob);

        /* Download */
        hipMemcpy(out_rgb, d_rgb, (size_t)3 * h * w * F, hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
        hipFree(d_rgb);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (r->verbose >= 1)
        fprintf(stderr, "hip_flux2: GPU VAE decode done (%.1f s)\n", dt);

    return 0;
}
