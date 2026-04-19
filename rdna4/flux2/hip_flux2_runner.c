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

/* Weight tensor: device pointer + per-tensor FP8 scale.
 * scale < 0 means F32 (no dequant); scale >= 0 means raw FP8 bytes. */
typedef struct {
    void *w;
    float scale;   /* -1.0 = F32, otherwise FP8 per-tensor scale */
} flux2_hip_wt;

typedef struct {
    flux2_hip_wt qkv;     /* [3H, H]        */
    flux2_hip_wt proj;    /* [H, H]         */
    flux2_hip_wt mlp_up;  /* [2*n_ff, H]    */
    flux2_hip_wt mlp_dn;  /* [H, n_ff]      */
    void *q_norm;         /* [head_dim] F32 */
    void *k_norm;         /* [head_dim] F32 */
} flux2_hip_stream_t;

typedef struct {
    flux2_hip_stream_t img, txt;
} flux2_hip_dblk_t;

typedef struct {
    flux2_hip_wt linear1;   /* [3H+2*n_ff, H] */
    flux2_hip_wt l2_attn;   /* [H, H]         */
    flux2_hip_wt l2_mlp;    /* [H, n_ff]      */
    void *q_norm;           /* [head_dim] F32 */
    void *k_norm;           /* [head_dim] F32 */
} flux2_hip_sblk_t;

/* ---- Runner struct ---- */

struct hip_flux2_runner {
    int device_id;
    int verbose;

    hipModule_t mod;
    hipFunction_t fn_gemm, fn_gemm_fp8, fn_gemm_fp8_opt;
    hipFunction_t fn_gemm_fp8_wmma;      /* FP8 in × FP8 wt */
    hipFunction_t fn_gemm_fp8_bf16_wmma; /* BF16 act × FP8 wt (BF16 WMMA) */
    hipFunction_t fn_layernorm, fn_silu, fn_adaln, fn_gated_add;
    hipFunction_t fn_rmsnorm_ph, fn_swiglu, fn_flash_attn;
    hipFunction_t fn_rope_img, fn_rope_txt, fn_bf16_trunc, fn_add;
    /* VAE kernels */
    hipFunction_t fn_vae_im2col, fn_vae_add_bias, fn_vae_transpose, fn_vae_conv1, fn_vae_gnsilu;
    hipFunction_t fn_vae_up2x, fn_vae_attn, fn_vae_bn;

    int use_fp8;   /* 1 = raw FP8 weights + LUT GEMM (4x less VRAM) */
    int use_fp8_opt; /* 1 = 128x128 tiled FP8 LUT GEMM (fast path) */
    int use_wmma;  /* 1 = use gfx12 FP8 WMMA matrix cores for GEMM */

    /* CPU model (for arch params + VAE fallback) */
    flux2_dit_model *dit;
    flux2_vae_model *vae;
    int vae_gpu_loaded;

    int H, nH, hd, n_ff, pin, txt_dim, n_dbl, n_sgl;

    /* Global GPU weights (weights use flux2_hip_wt for FP8/F32 dispatch) */
    flux2_hip_wt d_img_in_w, d_txt_in_w;
    flux2_hip_wt d_t_fc1_w, d_t_fc2_w;
    flux2_hip_wt d_mod_img_w, d_mod_txt_w, d_mod_sgl_w;
    flux2_hip_wt d_out_mod_w, d_out_proj_w;
    /* Biases always F32 */
    void *d_img_in_b, *d_txt_in_b;
    void *d_t_fc1_b, *d_t_fc2_b;

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

/* FP8 E4M3 → F32 (host, for LUT init) */
static float hip_flux2_fp8_e4m3_to_f32(uint8_t b) {
    uint32_t sign = (b >> 7) & 1;
    uint32_t exp  = (b >> 3) & 0xF;
    uint32_t mant = b & 0x7;
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    float f;
    if (exp == 0)                       f = ldexpf((float)mant / 8.0f, -6);
    else if (exp == 15 && mant == 7)    return 0.0f;
    else                                f = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    return sign ? -f : f;
}

static float g_flux2_fp8_lut[256];
static int   g_flux2_fp8_lut_init = 0;
static void flux2_fp8_lut_host_init(void) {
    if (g_flux2_fp8_lut_init) return;
    for (int i = 0; i < 256; i++)
        g_flux2_fp8_lut[i] = hip_flux2_fp8_e4m3_to_f32((uint8_t)i);
    g_flux2_fp8_lut_init = 1;
}

/* Upload a weight tensor from the mmap'd safetensors file.
 * - F8_E4M3: upload raw bytes (1 byte/element) + read per-tensor weight_scale.
 * - BF16/F32: dequant to F32, mark scale as -1.0f (F32 dispatch).
 *
 * If use_fp8 == 0, always dequants to F32 regardless of storage dtype. */
static flux2_hip_wt gpu_upload_st_wt(hip_flux2_runner *r, st_context *st,
                                      const char *wname, const char *sname) {
    flux2_hip_wt out = { NULL, -1.0f };
    int widx = safetensors_find(st, wname);
    if (widx < 0) return out;

    const char *dtype = safetensors_dtype(st, widx);
    const uint64_t *sh = safetensors_shape(st, widx);
    int nd = safetensors_ndims(st, widx);
    size_t n = (nd >= 2) ? (size_t)sh[0] * sh[1] : (size_t)sh[0];

    if (r->use_fp8 && strcmp(dtype, "F8_E4M3") == 0) {
        /* Raw FP8 upload */
        const void *data = safetensors_data(st, widx);
        out.w = hip_upload_raw(data, n);  /* 1 byte/element */
        out.scale = 1.0f;
        if (sname) {
            int sidx = safetensors_find(st, sname);
            if (sidx >= 0) out.scale = *(const float *)safetensors_data(st, sidx);
        }
        return out;
    }

    /* F32 dispatch: dequant to float */
    float *f32 = (float *)malloc(n * sizeof(float));
    if (strcmp(dtype, "F8_E4M3") == 0) {
        flux2_fp8_lut_host_init();
        float scale = 1.0f;
        if (sname) {
            int sidx = safetensors_find(st, sname);
            if (sidx >= 0) scale = *(const float *)safetensors_data(st, sidx);
        }
        const uint8_t *src = (const uint8_t *)safetensors_data(st, widx);
        for (size_t i = 0; i < n; i++) f32[i] = g_flux2_fp8_lut[src[i]] * scale;
    } else if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *src = (const uint16_t *)safetensors_data(st, widx);
        for (size_t i = 0; i < n; i++) {
            uint32_t b = (uint32_t)src[i] << 16;
            memcpy(&f32[i], &b, 4);
        }
    } else if (strcmp(dtype, "F32") == 0) {
        memcpy(f32, safetensors_data(st, widx), n * sizeof(float));
    } else {
        fprintf(stderr, "hip_flux2: unexpected dtype '%s' for %s\n", dtype, wname);
        free(f32);
        return out;
    }
    out.w = hip_upload_raw(f32, n * sizeof(float));
    free(f32);
    out.scale = -1.0f;
    return out;
}

/* Upload an F32 tensor from safetensors (biases, norms — dequants if not F32). */
static void *gpu_upload_st_f32(st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return NULL;
    const char *dtype = safetensors_dtype(st, idx);
    const uint64_t *sh = safetensors_shape(st, idx);
    int nd = safetensors_ndims(st, idx);
    size_t n = 1;
    for (int i = 0; i < nd; i++) n *= sh[i];
    const void *data = safetensors_data(st, idx);
    if (strcmp(dtype, "F32") == 0)
        return hip_upload_raw(data, n * sizeof(float));
    float *f32 = (float *)malloc(n * sizeof(float));
    if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            uint32_t b = (uint32_t)src[i] << 16;
            memcpy(&f32[i], &b, 4);
        }
    } else {
        fprintf(stderr, "hip_flux2: unsupported bias dtype '%s' for %s\n", dtype, name);
        free(f32);
        return NULL;
    }
    void *d = hip_upload_raw(f32, n * sizeof(float));
    free(f32);
    return d;
}

/* ---- Op functions ---- */

static void op_gemm(hip_flux2_runner *r, void *Y, void *W,
                    void *X, void *bias, int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    hipModuleLaunchKernel(r->fn_gemm, gx, gy, 1, 16, 16, 1, 0, NULL, args, NULL);
}

/* FP8 LUT GEMM with per-tensor weight scale (scalar path). */
static void op_gemm_fp8(hip_flux2_runner *r, void *Y, void *W_fp8, float w_scale,
                        void *X, void *bias, int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W_fp8, &X, &bias, &n_out, &n_in, &n_tok, &w_scale};
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    hipModuleLaunchKernel(r->fn_gemm_fp8, gx, gy, 1, 16, 16, 1, 0, NULL, args, NULL);
}

/* Tiled 128x128 FP8 LUT GEMM with per-tensor weight scale. 256 threads/CTA. */
static void op_gemm_fp8_opt(hip_flux2_runner *r, void *Y, void *W_fp8, float w_scale,
                             void *X, void *bias, int n_out, int n_in, int n_tok) {
    /* Kernel signature: (Y, W, X, bias, N, K, M, w_scale)
     *   N = n_out, K = n_in, M = n_tok */
    void *args[] = {&Y, &W_fp8, &X, &bias, &n_out, &n_in, &n_tok, &w_scale};
    unsigned gx = (unsigned)((n_out + 127) / 128);
    unsigned gy = (unsigned)((n_tok + 127) / 128);
    hipModuleLaunchKernel(r->fn_gemm_fp8_opt, gx, gy, 1, 256, 1, 1, 0, NULL, args, NULL);
}

/* FP8 WMMA matrix-core GEMM (gfx12 RDNA4). Tile 16x16x16 per wave.
 * act=FP8 (on-the-fly quant) — aggressive, not accuracy-safe for large K. */
static void op_gemm_fp8_wmma(hip_flux2_runner *r, void *Y, void *W_fp8, float w_scale,
                              void *X, void *bias, int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W_fp8, &X, &bias, &n_out, &n_in, &n_tok, &w_scale};
    unsigned gx = (unsigned)((n_out + 15) / 16);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    hipModuleLaunchKernel(r->fn_gemm_fp8_wmma, gx, gy, 1, 32, 1, 1, 0, NULL, args, NULL);
}

/* BF16 WMMA with FP8 weights: 256 threads, CTA tile 128 × 128. */
static void op_gemm_fp8_bf16_wmma(hip_flux2_runner *r, void *Y, void *W_fp8, float w_scale,
                                   void *X, void *bias, int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W_fp8, &X, &bias, &n_out, &n_in, &n_tok, &w_scale};
    unsigned gx = (unsigned)((n_out + 127) / 128);
    unsigned gy = (unsigned)((n_tok + 127) / 128);
    hipModuleLaunchKernel(r->fn_gemm_fp8_bf16_wmma, gx, gy, 1, 256, 1, 1, 0, NULL, args, NULL);
}

/* Dispatch: FP8 (scale >= 0) vs F32 (scale == -1). */
static void op_gemm_wt(hip_flux2_runner *r, void *Y, const flux2_hip_wt *W,
                       void *X, void *bias, int n_out, int n_in, int n_tok) {
    if (W->scale >= 0.0f) {
        if (r->use_wmma == 2 && r->fn_gemm_fp8_wmma)
            op_gemm_fp8_wmma(r, Y, W->w, W->scale, X, bias, n_out, n_in, n_tok);
        else if (r->use_wmma == 1 && r->fn_gemm_fp8_bf16_wmma)
            op_gemm_fp8_bf16_wmma(r, Y, W->w, W->scale, X, bias, n_out, n_in, n_tok);
        else if (r->use_fp8_opt && r->fn_gemm_fp8_opt)
            op_gemm_fp8_opt(r, Y, W->w, W->scale, X, bias, n_out, n_in, n_tok);
        else
            op_gemm_fp8(r, Y, W->w, W->scale, X, bias, n_out, n_in, n_tok);
    } else {
        op_gemm(r, Y, W->w, X, bias, n_out, n_in, n_tok);
    }
}

/* Slice a weight tensor by element offset. FP8 storage is 1 byte/element,
 * F32 storage is 4 bytes/element; scale is preserved. */
static flux2_hip_wt wt_slice(const flux2_hip_wt *w, size_t elem_offset) {
    size_t esz = (w->scale >= 0.0f) ? 1 : 4;
    flux2_hip_wt out;
    out.w = (char *)w->w + elem_offset * esz;
    out.scale = w->scale;
    return out;
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
    int compile_verbose = verbose;
    { const char *v = getenv("FLUX2_COMPILE_VERBOSE");
      if (v) compile_verbose = atoi(v); }
    if (hip_compile_kernels(&mod, device_id, full, "flux2.hip", compile_verbose, "hip_flux2") < 0) {
        free(full);
        fprintf(stderr, "hip_flux2: kernel compilation failed\n");
        return NULL;
    }
    free(full);

    hip_flux2_runner *r = (hip_flux2_runner *)calloc(1, sizeof(hip_flux2_runner));
    r->device_id = device_id;
    r->verbose = verbose;
    r->mod = mod;

    /* FP8 mode: env-gated. Default ON — dramatically smaller VRAM footprint. */
    {
        const char *v = getenv("FLUX2_FP8_GEMM");
        if (v && (strcmp(v, "0") == 0 || strcmp(v, "false") == 0))
            r->use_fp8 = 0;
        else
            r->use_fp8 = 1;
    }
    /* Tiled 128×128 FP8 LUT: default ON (big speedup, no correctness change). */
    {
        const char *v = getenv("FLUX2_FP8_OPT");
        if (v && (strcmp(v, "0") == 0 || strcmp(v, "false") == 0))
            r->use_fp8_opt = 0;
        else
            r->use_fp8_opt = 1;
    }
    /* WMMA (gfx12 matrix cores): opt-in via FLUX2_FP8_WMMA.
     *   =1 (or set) : BF16 activation × FP8 weight via BF16 WMMA (accuracy-safe)
     *   =2 (aka FP8 act): FP8×FP8 WMMA (fastest but lossy, debug only) */
    {
        const char *v = getenv("FLUX2_FP8_WMMA");
        if (!v || strcmp(v, "0") == 0 || strcmp(v, "false") == 0) r->use_wmma = 0;
        else if (strcmp(v, "2") == 0 || strcmp(v, "fp8") == 0)    r->use_wmma = 2;
        else                                                       r->use_wmma = 1;
    }

    /* Get function handles */
    hipModuleGetFunction(&r->fn_gemm,        mod, "gemm_f32_f32");
    hipModuleGetFunction(&r->fn_gemm_fp8,    mod, "gemm_fp8_scaled_f32");
    if (hipModuleGetFunction(&r->fn_gemm_fp8_opt, mod, "gemm_fp8_scaled_opt") != hipSuccess)
        r->fn_gemm_fp8_opt = NULL;
    if (hipModuleGetFunction(&r->fn_gemm_fp8_wmma, mod, "gemm_fp8_wmma") != hipSuccess)
        r->fn_gemm_fp8_wmma = NULL;
    if (hipModuleGetFunction(&r->fn_gemm_fp8_bf16_wmma, mod, "gemm_fp8w_bf16a_wmma_t") != hipSuccess)
        r->fn_gemm_fp8_bf16_wmma = NULL;
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

    if (r->use_fp8_opt) {
        if (!r->fn_gemm_fp8_opt) {
            fprintf(stderr, "hip_flux2: gemm_fp8_scaled_opt kernel missing — falling back to scalar FP8\n");
            r->use_fp8_opt = 0;
        } else if (verbose >= 1) {
            fprintf(stderr, "hip_flux2: FP8 128x128 tiled GEMM enabled\n");
        }
    }
    if (r->use_wmma) {
        hipFunction_t want = (r->use_wmma == 2) ? r->fn_gemm_fp8_wmma : r->fn_gemm_fp8_bf16_wmma;
        const char *label = (r->use_wmma == 2) ? "FP8xFP8" : "BF16xFP8";
        if (!want) {
            fprintf(stderr, "hip_flux2: WMMA kernel (%s) not found (not gfx12?) — falling back\n", label);
            r->use_wmma = 0;
        } else if (!r->use_fp8) {
            fprintf(stderr, "hip_flux2: FLUX2_FP8_WMMA requires FP8 mode — ignoring\n");
            r->use_wmma = 0;
        } else if (verbose >= 1) {
            fprintf(stderr, "hip_flux2: %s WMMA (gfx12) enabled\n", label);
        }
    }

    /* Upload FP8 LUT to constant memory if FP8 mode is requested and kernel loaded. */
    if (r->use_fp8) {
        if (!r->fn_gemm_fp8) {
            fprintf(stderr, "hip_flux2: FP8 kernel missing — disabling FP8 mode\n");
            r->use_fp8 = 0;
        } else {
            flux2_fp8_lut_host_init();
            hipDeviceptr_t d_lut;
            size_t lut_size = 0;
            hipError_t e = hipModuleGetGlobal(&d_lut, &lut_size, mod, "d_fp8_to_f32_lut");
            if (e != hipSuccess || lut_size != 256 * sizeof(float)) {
                fprintf(stderr, "hip_flux2: FP8 LUT symbol missing — disabling FP8 mode\n");
                r->use_fp8 = 0;
            } else {
                hipMemcpyHtoD(d_lut, g_flux2_fp8_lut, 256 * sizeof(float));
                if (verbose >= 1)
                    fprintf(stderr, "hip_flux2: FP8 LUT GEMM enabled (raw FP8 weights)\n");
            }
        }
    }

    if (verbose >= 1) fprintf(stderr, "hip_flux2: init OK\n");
    return r;
}

/* Upload one double-stream sub-block (img or txt) from safetensors.
 * prefix is e.g. "double_blocks.3.img" or "double_blocks.3.txt". */
static void upload_stream_st(hip_flux2_runner *r, flux2_hip_stream_t *gs,
                              st_context *st, const char *prefix,
                              const float *cpu_q_norm, const float *cpu_k_norm, int hd) {
    char wn[256], sn[256];
    #define UPL(field, suffix) do { \
        snprintf(wn, sizeof(wn), "%s_" suffix ".weight", prefix); \
        snprintf(sn, sizeof(sn), "%s_" suffix ".weight_scale", prefix); \
        gs->field = gpu_upload_st_wt(r, st, wn, sn); \
    } while(0)
    UPL(qkv,    "attn.qkv");
    UPL(proj,   "attn.proj");
    UPL(mlp_up, "mlp.0");
    UPL(mlp_dn, "mlp.2");
    #undef UPL
    gs->q_norm = gpu_upload_f32_or0(cpu_q_norm, hd);
    gs->k_norm = gpu_upload_f32_or0(cpu_k_norm, hd);
}

static void free_wt(flux2_hip_wt *wt) {
    if (wt && wt->w) { hipFree(wt->w); wt->w = NULL; }
}
static void free_stream(flux2_hip_stream_t *gs) {
    free_wt(&gs->qkv);
    free_wt(&gs->proj);
    free_wt(&gs->mlp_up);
    free_wt(&gs->mlp_dn);
    if (gs->q_norm)   hipFree(gs->q_norm);
    if (gs->k_norm)   hipFree(gs->k_norm);
}

/* Load single_blocks.%d.linear2 as a column-split pair: [H,H+n_ff] → l2_attn[H,H] + l2_mlp[H,n_ff]
 *
 * Each output row (H+n_ff elements) is split into leading H columns (attn) and
 * trailing n_ff columns (mlp). In FP8 mode we split the raw byte stream; otherwise
 * we dequant to F32 and split. Both parts share the same per-tensor weight_scale. */
static int load_linear2_split(hip_flux2_runner *r, st_context *st, int bi,
                               int H, int n_ff,
                               flux2_hip_wt *out_attn, flux2_hip_wt *out_mlp) {
    char wn[256], sn[256];
    snprintf(wn, sizeof(wn), "single_blocks.%d.linear2.weight", bi);
    snprintf(sn, sizeof(sn), "single_blocks.%d.linear2.weight_scale", bi);
    int widx = safetensors_find(st, wn);
    if (widx < 0) {
        fprintf(stderr, "hip_flux2: missing %s\n", wn);
        return -1;
    }
    const char *dtype = safetensors_dtype(st, widx);
    int l2_in = H + n_ff;  /* column count */

    float scale = 1.0f;
    int sidx = safetensors_find(st, sn);
    if (sidx >= 0) scale = *(const float *)safetensors_data(st, sidx);

    if (r->use_fp8 && strcmp(dtype, "F8_E4M3") == 0) {
        const uint8_t *src = (const uint8_t *)safetensors_data(st, widx);
        uint8_t *attn = (uint8_t *)malloc((size_t)H * H);
        uint8_t *mlp_ = (uint8_t *)malloc((size_t)H * n_ff);
        for (int row = 0; row < H; row++) {
            memcpy(attn + (size_t)row * H,      src + (size_t)row * l2_in,          (size_t)H);
            memcpy(mlp_ + (size_t)row * n_ff,   src + (size_t)row * l2_in + H,      (size_t)n_ff);
        }
        out_attn->w = hip_upload_raw(attn, (size_t)H * H);
        out_attn->scale = scale;
        out_mlp->w  = hip_upload_raw(mlp_, (size_t)H * n_ff);
        out_mlp->scale  = scale;
        free(attn); free(mlp_);
        return 0;
    }

    /* F32 dispatch: dequant whole tensor, split, upload */
    size_t n = (size_t)H * l2_in;
    float *f32 = (float *)malloc(n * sizeof(float));
    if (strcmp(dtype, "F8_E4M3") == 0) {
        flux2_fp8_lut_host_init();
        const uint8_t *src = (const uint8_t *)safetensors_data(st, widx);
        for (size_t i = 0; i < n; i++) f32[i] = g_flux2_fp8_lut[src[i]] * scale;
    } else if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *src = (const uint16_t *)safetensors_data(st, widx);
        for (size_t i = 0; i < n; i++) {
            uint32_t b = (uint32_t)src[i] << 16;
            memcpy(&f32[i], &b, 4);
        }
    } else if (strcmp(dtype, "F32") == 0) {
        memcpy(f32, safetensors_data(st, widx), n * sizeof(float));
    } else {
        fprintf(stderr, "hip_flux2: unexpected linear2 dtype '%s'\n", dtype);
        free(f32);
        return -1;
    }
    float *attn = (float *)malloc((size_t)H * H * sizeof(float));
    float *mlp_ = (float *)malloc((size_t)H * n_ff * sizeof(float));
    for (int row = 0; row < H; row++) {
        memcpy(attn + (size_t)row * H,    f32 + (size_t)row * l2_in,     (size_t)H * sizeof(float));
        memcpy(mlp_ + (size_t)row * n_ff, f32 + (size_t)row * l2_in + H, (size_t)n_ff * sizeof(float));
    }
    free(f32);
    out_attn->w     = hip_upload_raw(attn, (size_t)H * H * sizeof(float));
    out_attn->scale = -1.0f;
    out_mlp->w      = hip_upload_raw(mlp_, (size_t)H * n_ff * sizeof(float));
    out_mlp->scale  = -1.0f;
    free(attn); free(mlp_);
    return 0;
}

int hip_flux2_load_dit(hip_flux2_runner *r, const char *path) {
    if (r->verbose >= 1) fprintf(stderr, "hip_flux2: loading DiT %s\n", path);

    /* CPU load gives us arch params + float q/k norm arrays. Large weight
     * buffers are dequanted to F32 on CPU but will be freed below if we're
     * in FP8 mode so they don't permanently balloon host RAM. */
    r->dit = flux2_dit_load_safetensors(path);
    if (!r->dit) return -1;

    flux2_dit_model *m = r->dit;
    r->H = m->hidden_dim;  r->nH = m->n_heads;  r->hd = m->head_dim;
    r->n_ff = m->n_ff;     r->pin = m->patch_in_channels;
    r->txt_dim = m->txt_dim;
    r->n_dbl = m->n_double_blocks;
    r->n_sgl = m->n_single_blocks;

    /* Reopen safetensors for raw FP8 / per-tensor scale access. */
    st_context *st = safetensors_open(path);
    if (!st) {
        fprintf(stderr, "hip_flux2: reopen %s failed\n", path);
        flux2_dit_free(r->dit);
        r->dit = NULL;
        return -1;
    }

    if (r->verbose >= 1)
        fprintf(stderr, "hip_flux2: uploading weights to GPU (H=%d, %d+%d blocks, %s)...\n",
                r->H, r->n_dbl, r->n_sgl, r->use_fp8 ? "FP8 raw" : "F32 dequanted");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Global weights (FP8 in fp8-file, BF16 in base-file). Biases always F32. */
    r->d_img_in_w  = gpu_upload_st_wt(r, st, "img_in.weight",  "img_in.weight_scale");
    r->d_img_in_b  = gpu_upload_f32_or0(m->img_in_b, r->H);
    r->d_txt_in_w  = gpu_upload_st_wt(r, st, "txt_in.weight",  "txt_in.weight_scale");
    r->d_txt_in_b  = gpu_upload_f32_or0(m->txt_in_b, r->H);
    r->d_t_fc1_w   = gpu_upload_st_wt(r, st, "time_in.in_layer.weight",  "time_in.in_layer.weight_scale");
    r->d_t_fc1_b   = gpu_upload_f32_or0(m->time_in_lin1_b, r->H);
    r->d_t_fc2_w   = gpu_upload_st_wt(r, st, "time_in.out_layer.weight", "time_in.out_layer.weight_scale");
    r->d_t_fc2_b   = gpu_upload_f32_or0(m->time_in_lin2_b, r->H);
    r->d_mod_img_w = gpu_upload_st_wt(r, st, "double_stream_modulation_img.lin.weight",
                                             "double_stream_modulation_img.lin.weight_scale");
    r->d_mod_txt_w = gpu_upload_st_wt(r, st, "double_stream_modulation_txt.lin.weight",
                                             "double_stream_modulation_txt.lin.weight_scale");
    r->d_mod_sgl_w = gpu_upload_st_wt(r, st, "single_stream_modulation.lin.weight",
                                             "single_stream_modulation.lin.weight_scale");
    r->d_out_mod_w = gpu_upload_st_wt(r, st, "final_layer.adaLN_modulation.1.weight",
                                             "final_layer.adaLN_modulation.1.weight_scale");
    r->d_out_proj_w= gpu_upload_st_wt(r, st, "final_layer.linear.weight",
                                             "final_layer.linear.weight_scale");

    /* Double-stream block weights */
    r->gpu_dblk = (flux2_hip_dblk_t *)calloc((size_t)r->n_dbl, sizeof(flux2_hip_dblk_t));
    for (int i = 0; i < r->n_dbl; i++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "double_blocks.%d.img", i);
        upload_stream_st(r, &r->gpu_dblk[i].img, st, prefix,
                         m->dblk[i].img.q_norm, m->dblk[i].img.k_norm, r->hd);
        snprintf(prefix, sizeof(prefix), "double_blocks.%d.txt", i);
        upload_stream_st(r, &r->gpu_dblk[i].txt, st, prefix,
                         m->dblk[i].txt.q_norm, m->dblk[i].txt.k_norm, r->hd);
        if (r->verbose >= 2)
            fprintf(stderr, "\r  double block %d/%d", i+1, r->n_dbl);
    }

    /* Single-stream block weights */
    r->gpu_sblk = (flux2_hip_sblk_t *)calloc((size_t)r->n_sgl, sizeof(flux2_hip_sblk_t));
    for (int i = 0; i < r->n_sgl; i++) {
        char wn[256], sn[256];
        snprintf(wn, sizeof(wn), "single_blocks.%d.linear1.weight", i);
        snprintf(sn, sizeof(sn), "single_blocks.%d.linear1.weight_scale", i);
        r->gpu_sblk[i].linear1 = gpu_upload_st_wt(r, st, wn, sn);

        if (load_linear2_split(r, st, i, r->H, r->n_ff,
                               &r->gpu_sblk[i].l2_attn, &r->gpu_sblk[i].l2_mlp) != 0) {
            safetensors_close(st);
            return -1;
        }

        r->gpu_sblk[i].q_norm = gpu_upload_f32(m->sblk[i].q_norm, r->hd);
        r->gpu_sblk[i].k_norm = gpu_upload_f32(m->sblk[i].k_norm, r->hd);
        if (r->verbose >= 2)
            fprintf(stderr, "\r  single block %d/%d", i+1, r->n_sgl);
    }

    safetensors_close(st);
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
    op_gemm_wt(r, r->d_temb, &r->d_t_fc1_w, d_traw, r->d_t_fc1_b, H, 256, 1);
    op_silu(r, r->d_temb, H);
    /* temb = time_in_lin2 @ temb + b2 */
    op_gemm_wt(r, r->d_temb_silu, &r->d_t_fc2_w, r->d_temb, r->d_t_fc2_b, H, H, 1);
    void *d_temb_final = r->d_temb_silu;

    hipFree(d_traw);

    /* 2. Project img and txt tokens */
    op_gemm_wt(r, r->d_img, &r->d_img_in_w, r->d_img_in_buf, r->d_img_in_b,
               H, r->pin, n_img);
    op_gemm_wt(r, r->d_txt, &r->d_txt_in_w, r->d_txt_in_buf, r->d_txt_in_b,
               H, r->txt_dim, n_txt);

    /* 3. Compute global modulations: SiLU(temb) -> mod vectors */
    hipMemcpy(r->d_temb, d_temb_final, (size_t)H * F, hipMemcpyDeviceToDevice);
    op_silu(r, r->d_temb, H);
    void *d0 = NULL;
    op_gemm_wt(r, r->d_mod_img_v, &r->d_mod_img_w, r->d_temb, d0, 6*H, H, 1);
    op_gemm_wt(r, r->d_mod_txt_v, &r->d_mod_txt_w, r->d_temb, d0, 6*H, H, 1);
    op_gemm_wt(r, r->d_mod_sgl_v, &r->d_mod_sgl_w, r->d_temb, d0, 3*H, H, 1);

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
        flux2_hip_wt img_q_w = b->img.qkv;
        flux2_hip_wt img_k_w = wt_slice(&b->img.qkv, (size_t)H * H);
        flux2_hip_wt img_v_w = wt_slice(&b->img.qkv, (size_t)2 * H * H);
        op_gemm_wt(r, r->d_q, &img_q_w, r->d_scratch1, d0, H, H, n_img);
        op_gemm_wt(r, r->d_k, &img_k_w, r->d_scratch1, d0, H, H, n_img);
        op_gemm_wt(r, r->d_v, &img_v_w, r->d_scratch1, d0, H, H, n_img);

        /* TXT stream: adaLN -> QKV */
        op_adaln(r, r->d_scratch1, r->d_txt, mt_shift_a, mt_scale_a, n_txt, H);
        flux2_hip_wt txt_q_w = b->txt.qkv;
        flux2_hip_wt txt_k_w = wt_slice(&b->txt.qkv, (size_t)H * H);
        flux2_hip_wt txt_v_w = wt_slice(&b->txt.qkv, (size_t)2 * H * H);
        void *tq = ptr_offset(r->d_q, (size_t)n_img * H * F);
        void *tk = ptr_offset(r->d_k, (size_t)n_img * H * F);
        void *tv = ptr_offset(r->d_v, (size_t)n_img * H * F);
        op_gemm_wt(r, tq, &txt_q_w, r->d_scratch1, d0, H, H, n_txt);
        op_gemm_wt(r, tk, &txt_k_w, r->d_scratch1, d0, H, H, n_txt);
        op_gemm_wt(r, tv, &txt_v_w, r->d_scratch1, d0, H, H, n_txt);

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
        op_gemm_wt(r, r->d_scratch1, &b->img.proj, r->d_attn_out, d0, H, H, n_img);
        void *txt_attn = ptr_offset(r->d_attn_out, (size_t)n_img * H * F);
        op_gemm_wt(r, r->d_scratch2, &b->txt.proj, txt_attn, d0, H, H, n_txt);

        /* Gated residual */
        op_gated_add(r, r->d_img, r->d_scratch1, mi_gate_a, n_img, H);
        op_gated_add(r, r->d_txt, r->d_scratch2, mt_gate_a, n_txt, H);

        /* FFN img */
        op_adaln(r, r->d_scratch1, r->d_img, mi_shift_f, mi_scale_f, n_img, H);
        op_gemm_wt(r, r->d_scratch2, &b->img.mlp_up, r->d_scratch1, d0, 2*n_ff, H, n_img);
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_img, n_ff);
        op_gemm_wt(r, r->d_scratch1, &b->img.mlp_dn, r->d_scratch3, d0, H, n_ff, n_img);
        op_gated_add(r, r->d_img, r->d_scratch1, mi_gate_f, n_img, H);

        /* FFN txt */
        op_adaln(r, r->d_scratch1, r->d_txt, mt_shift_f, mt_scale_f, n_txt, H);
        op_gemm_wt(r, r->d_scratch2, &b->txt.mlp_up, r->d_scratch1, d0, 2*n_ff, H, n_txt);
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_txt, n_ff);
        op_gemm_wt(r, r->d_scratch1, &b->txt.mlp_dn, r->d_scratch3, d0, H, n_ff, n_txt);
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
        flux2_hip_wt q_w  = b->linear1;
        flux2_hip_wt k_w  = wt_slice(&b->linear1, (size_t)H * H);
        flux2_hip_wt v_w  = wt_slice(&b->linear1, (size_t)2 * H * H);
        flux2_hip_wt gu_w = wt_slice(&b->linear1, (size_t)3 * H * H);

        op_gemm_wt(r, r->d_q, &q_w, r->d_scratch1, d0, H, H, n_tot);
        op_gemm_wt(r, r->d_k, &k_w, r->d_scratch1, d0, H, H, n_tot);
        op_gemm_wt(r, r->d_v, &v_w, r->d_scratch1, d0, H, H, n_tot);
        op_gemm_wt(r, r->d_scratch2, &gu_w, r->d_scratch1, d0, 2*n_ff, H, n_tot);

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
        op_gemm_wt(r, r->d_scratch1, &b->l2_attn, r->d_attn_out, d0, H, H, n_tot);
        op_gemm_wt(r, r->d_scratch2, &b->l2_mlp,  r->d_scratch3, d0, H, n_ff, n_tot);

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
    op_gemm_wt(r, r->d_scratch1, &r->d_out_mod_w, r->d_temb, d0, 2*H, H, 1);
    void *out_shift = r->d_scratch1;
    void *out_scale = ptr_offset(r->d_scratch1, (size_t)H * F);
    op_adaln(r, r->d_scratch2, d_img_out, out_shift, out_scale, n_img, H);

    /* Final linear */
    op_gemm_wt(r, r->d_attn_out, &r->d_out_proj_w, r->d_scratch2, d0, r->pin, H, n_img);

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
            free_wt(&r->gpu_sblk[i].linear1);
            free_wt(&r->gpu_sblk[i].l2_attn);
            free_wt(&r->gpu_sblk[i].l2_mlp);
            if (r->gpu_sblk[i].q_norm) hipFree(r->gpu_sblk[i].q_norm);
            if (r->gpu_sblk[i].k_norm) hipFree(r->gpu_sblk[i].k_norm);
        }
        free(r->gpu_sblk);
    }

    free_wt(&r->d_img_in_w);
    free_wt(&r->d_txt_in_w);
    free_wt(&r->d_t_fc1_w);
    free_wt(&r->d_t_fc2_w);
    free_wt(&r->d_mod_img_w);
    free_wt(&r->d_mod_txt_w);
    free_wt(&r->d_mod_sgl_w);
    free_wt(&r->d_out_mod_w);
    free_wt(&r->d_out_proj_w);
    if (r->d_img_in_b)  hipFree(r->d_img_in_b);
    if (r->d_txt_in_b)  hipFree(r->d_txt_in_b);
    if (r->d_t_fc1_b)   hipFree(r->d_t_fc1_b);
    if (r->d_t_fc2_b)   hipFree(r->d_t_fc2_b);

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

/* Dump a GPU buffer to disk as raw F32 when FLUX2_VAE_DUMP_DIR is set. */
static void vae_dump_stage(const char *name, void *d_buf, size_t n_elems) {
    const char *dir = getenv("FLUX2_VAE_DUMP_DIR");
    if (!dir || !dir[0]) return;
    float *h = (float *)malloc(n_elems * sizeof(float));
    hipMemcpy(h, d_buf, n_elems * sizeof(float), hipMemcpyDeviceToHost);
    char path[512];
    snprintf(path, sizeof(path), "%s/hip_vae_trace_%s.bin", dir, name);
    FILE *fp = fopen(path, "wb");
    if (fp) {
        fwrite(h, sizeof(float), n_elems, fp);
        fclose(fp);
        double mn = 0, mx = -1e30, mi = 1e30, sq = 0;
        for (size_t i = 0; i < n_elems; i++) {
            double v = h[i];
            mn += v; sq += v * v;
            if (v > mx) mx = v; if (v < mi) mi = v;
        }
        double mean = mn / n_elems;
        double var  = sq / n_elems - mean * mean;
        fprintf(stderr, "  hip vae %-10s n=%zu mean %.4f std %.4f min %.4f max %.4f\n",
                name, n_elems, mean, sqrt(var > 0 ? var : 0), mi, mx);
    }
    free(h);
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

    /* Stage 0: Batch-norm denormalize (skippable via FLUX2_SKIP_VAE_BN=1) */
    void *d_x;
    int skip_bn = 0;
    { const char *e = getenv("FLUX2_SKIP_VAE_BN");
      if (e && !(strcmp(e,"0")==0 || strcmp(e,"false")==0)) skip_bn = 1; }
    if (!skip_bn && m->bn_mean && m->bn_var) {
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

    vae_dump_stage("in", d_x, (size_t)lc * h * w);

    /* Stage 1: post_quant_conv (1x1) */
    if (m->pqc_w) {
        void *d_pqcw = gpu_upload_f32(m->pqc_w, lc * lc);
        void *d_pqcb = gpu_upload_f32(m->pqc_b, lc);
        void *d_tmp = vae_alloc((size_t)lc * h * w * F);
        vae_conv1(r, d_tmp, d_x, d_pqcw, d_pqcb, lc, h * w, lc);
        hipFree(d_x); hipFree(d_pqcw); hipFree(d_pqcb);
        d_x = d_tmp;
    }
    vae_dump_stage("post_q", d_x, (size_t)lc * h * w);

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
    vae_dump_stage("conv_in", d_x, (size_t)c * h * w);

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
    vae_dump_stage("mid", d_x, (size_t)c * h * w);

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
        /* resnets.2 — diffusers Decoder uses layers_per_block+1 = 3 resnets per up_block */
        {
            vae_gpu_resblock gr = upload_resblock(&m->up_res[bi][2]);
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
        char name[16]; snprintf(name, sizeof(name), "up%d", bi);
        vae_dump_stage(name, d_x, (size_t)c * h * w);
    }

    /* Stage 5: norm_out -> SiLU -> conv_out */
    {
        void *d_now = gpu_upload_f32(m->norm_out_w, c);
        void *d_nob = gpu_upload_f32(m->norm_out_b, c);
        void *d_normed = vae_alloc((size_t)c * h * w * F);
        vae_groupnorm_silu(r, d_normed, d_x, d_now, d_nob, c, h*w, ng, 1);
        hipFree(d_x); hipFree(d_now); hipFree(d_nob);
        vae_dump_stage("act", d_normed, (size_t)c * h * w);

        void *d_cow = gpu_upload_f32(m->conv_out_w, 3 * c * 9);
        void *d_cob = gpu_upload_f32(m->conv_out_b, 3);
        void *d_rgb = vae_alloc((size_t)3 * h * w * F);
        vae_conv3(r, d_rgb, d_normed, d_cow, d_cob, c, h, w, 3);
        hipFree(d_normed); hipFree(d_cow); hipFree(d_cob);
        vae_dump_stage("conv_out", d_rgb, (size_t)3 * h * w);

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
