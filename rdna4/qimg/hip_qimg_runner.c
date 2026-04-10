/*
 * hip_qimg_runner.c - HIP/ROCm Qwen-Image text-to-image runner (RDNA4)
 *
 * GPU-accelerated DiT (60 dual-stream blocks) with block-by-block streaming.
 * VAE decode on GPU.
 *
 * Compiles with plain gcc (no hipcc). Uses rocew for dynamic HIP/HIPRTC loading.
 * F32 weights on GPU, F32 compute. Single-stream sequential kernel launches.
 *
 * Port of cuda_qimg_runner.h for AMD ROCm/HIP.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"

#include "hip_qimg_runner.h"
#include "../rocew.h"
#include "../hip_kernels_common.h"
#include "hip_qimg_kernels.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- FP8 E4M3 → F32 CPU conversion ---- */

static float fp8_e4m3_to_f32(uint8_t b) {
    uint32_t sign = (b >> 7) & 1;
    uint32_t exp  = (b >> 3) & 0xF;
    uint32_t mant = b & 0x7;
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    float f;
    if (exp == 0) {
        f = ldexpf((float)mant / 8.0f, -6);
    } else if (exp == 15 && mant == 7) {
        return 0.0f;  /* NaN → 0 */
    } else {
        f = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -f : f;
}

static float hip_fp8_to_f32_lut[256];
static int hip_fp8_to_f32_lut_init = 0;

static void init_fp8_to_f32_lut(void) {
    if (hip_fp8_to_f32_lut_init) return;
    for (int i = 0; i < 256; i++)
        hip_fp8_to_f32_lut[i] = fp8_e4m3_to_f32((uint8_t)i);
    hip_fp8_to_f32_lut_init = 1;
}

/* ---- Per-block GPU weight struct ---- */

typedef struct {
    void *attn_q_w, *attn_q_b, *attn_k_w, *attn_k_b, *attn_v_w, *attn_v_b;
    void *attn_out_w, *attn_out_b;
    void *attn_add_q_w, *attn_add_q_b, *attn_add_k_w, *attn_add_k_b;
    void *attn_add_v_w, *attn_add_v_b, *attn_add_out_w, *attn_add_out_b;
    void *norm_q_w, *norm_k_w, *norm_added_q_w, *norm_added_k_w;
    void *img_mod_w, *img_mod_b;
    void *img_mlp_fc1_w, *img_mlp_fc1_b, *img_mlp_fc2_w, *img_mlp_fc2_b;
    void *txt_mod_w, *txt_mod_b;
    void *txt_mlp_fc1_w, *txt_mlp_fc1_b, *txt_mlp_fc2_w, *txt_mlp_fc2_b;
} qimg_block_gpu;

/* ---- Runner struct ---- */

struct hip_qimg_runner {
    int device_id;
    int verbose;

    hipModule_t mod;
    hipFunction_t fn_gemm, fn_layernorm, fn_silu, fn_gelu, fn_adaln, fn_gated_add;
    hipFunction_t fn_rmsnorm_ph, fn_flash_attn;
    hipFunction_t fn_rope_2d, fn_rope_1d, fn_bf16_trunc, fn_add;
    hipFunction_t fn_patchify, fn_unpatchify, fn_euler_step, fn_cfg_combine;
    hipFunction_t fn_rmsnorm_weighted, fn_fp8_roundtrip;
    /* VAE kernels */
    hipFunction_t fn_vae_conv2d, fn_vae_rmsnorm, fn_vae_silu, fn_vae_up2x;

    /* DiT config */
    int dim, n_heads, head_dim, n_blocks;
    int in_ch, txt_dim, mlp_h;

    /* Safetensors context (mmap'd) */
    void *dit_st;
    void *vae_st;

    /* Preloaded blocks on GPU */
    qimg_block_gpu *gpu_blocks;
    int n_preloaded;

    /* Global GPU weights */
    void *d_img_in_w, *d_img_in_b;
    void *d_txt_in_w, *d_txt_in_b;
    void *d_txt_norm_w;
    void *d_t_fc1_w, *d_t_fc1_b;
    void *d_t_fc2_w, *d_t_fc2_b;
    void *d_norm_out_w, *d_norm_out_b;
    void *d_proj_out_w, *d_proj_out_b;
};


/* ---- Upload helpers ---- */

/* Upload safetensor as F32 to GPU (handles FP8 E4M3, BF16, F16, F32 inputs) */
static void *qimg_st_upload_f32(st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return NULL;
    const uint64_t *shape = safetensors_shape(st, idx);
    int ndims = safetensors_ndims(st, idx);
    size_t n = 1;
    for (int d = 0; d < ndims; d++) n *= shape[d];
    const uint8_t *src = (const uint8_t *)safetensors_data(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    float *f32 = (float *)malloc(n * sizeof(float));
    if (!f32) return NULL;

    if (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0) {
        init_fp8_to_f32_lut();
        for (size_t i = 0; i < n; i++)
            f32[i] = hip_fp8_to_f32_lut[src[i]];
    } else if (strcmp(dtype, "F32") == 0) {
        memcpy(f32, src, n * 4);
    } else if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *bf = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = (uint32_t)bf[i] << 16;
            memcpy(&f32[i], &bits, 4);
        }
    } else if (strcmp(dtype, "F16") == 0) {
        const uint16_t *f16 = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = f16[i];
            uint32_t s = (bits >> 15) & 1;
            uint32_t e = (bits >> 10) & 0x1F;
            uint32_t m = bits & 0x3FF;
            uint32_t f;
            if (e == 0) f = s << 31;
            else if (e == 31) f = (s << 31) | (0xFF << 23) | (m << 13);
            else f = (s << 31) | ((e + 112) << 23) | (m << 13);
            memcpy(&f32[i], &f, 4);
        }
    } else {
        fprintf(stderr, "hip_qimg: unsupported dtype '%s' for %s\n", dtype, name);
        free(f32);
        return NULL;
    }

    void *d = NULL;
    if (hipMalloc(&d, n * sizeof(float)) != hipSuccess) {
        fprintf(stderr, "hip_qimg: hipMalloc(%.1f MB) FAILED for %s\n",
                (float)(n * 4) / (1 << 20), name);
        free(f32);
        return NULL;
    }
    hipMemcpy(d, f32, n * sizeof(float), hipMemcpyHostToDevice);
    free(f32);
    return d;
}

/* Upload 3D conv weight → 2D by taking last temporal slice */
static void *qimg_upload_conv3d(st_context *st, const char *name,
                                 int *out_co, int *out_ci) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return NULL;
    const uint64_t *shape = safetensors_shape(st, idx);
    int co = (int)shape[0], ci = (int)shape[1], kd = (int)shape[2];
    int kh = (int)shape[3], kw = (int)shape[4];
    if (out_co) *out_co = co;
    if (out_ci) *out_ci = ci;
    size_t n2d = (size_t)co * ci * kh * kw;
    const uint16_t *bf = (const uint16_t *)safetensors_data(st, idx);
    int d_last = kd - 1;
    float *w2d = (float *)malloc(n2d * sizeof(float));
    for (int o = 0; o < co; o++)
        for (int i = 0; i < ci; i++)
            for (int h = 0; h < kh; h++)
                for (int w = 0; w < kw; w++) {
                    size_t idx3 = ((((size_t)o*ci+i)*kd+d_last)*kh+h)*kw+w;
                    uint32_t bits = (uint32_t)bf[idx3] << 16;
                    float f; memcpy(&f, &bits, 4);
                    w2d[(((size_t)o*ci+i)*kh+h)*kw+w] = f;
                }
    void *dp = NULL;
    hipMalloc(&dp, n2d * sizeof(float));
    hipMemcpy(dp, w2d, n2d * sizeof(float), hipMemcpyHostToDevice);
    free(w2d);
    return dp;
}

/* ---- Load/free one DiT block ---- */

static void qimg_free_block(qimg_block_gpu *b) {
    void **ptrs = (void **)b;
    int n = sizeof(qimg_block_gpu) / sizeof(void *);
    for (int i = 0; i < n; i++) {
        if (ptrs[i]) { hipFree(ptrs[i]); ptrs[i] = NULL; }
    }
}

static int qimg_load_block(hip_qimg_runner *r, int block_idx, qimg_block_gpu *b) {
    st_context *st = (st_context *)r->dit_st;
    char name[256];
    int ok = 1;

    #define BLK_W(field, suffix) do { if (ok) { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, block_idx); \
        b->field = qimg_st_upload_f32(st, name); \
        if (!b->field) ok = 0; \
    } } while(0)

    BLK_W(attn_q_w, "attn.to_q.weight"); BLK_W(attn_q_b, "attn.to_q.bias");
    BLK_W(attn_k_w, "attn.to_k.weight"); BLK_W(attn_k_b, "attn.to_k.bias");
    BLK_W(attn_v_w, "attn.to_v.weight"); BLK_W(attn_v_b, "attn.to_v.bias");
    BLK_W(attn_out_w, "attn.to_out.0.weight"); BLK_W(attn_out_b, "attn.to_out.0.bias");

    BLK_W(attn_add_q_w, "attn.add_q_proj.weight"); BLK_W(attn_add_q_b, "attn.add_q_proj.bias");
    BLK_W(attn_add_k_w, "attn.add_k_proj.weight"); BLK_W(attn_add_k_b, "attn.add_k_proj.bias");
    BLK_W(attn_add_v_w, "attn.add_v_proj.weight"); BLK_W(attn_add_v_b, "attn.add_v_proj.bias");
    BLK_W(attn_add_out_w, "attn.to_add_out.weight"); BLK_W(attn_add_out_b, "attn.to_add_out.bias");

    BLK_W(norm_q_w, "attn.norm_q.weight");
    BLK_W(norm_k_w, "attn.norm_k.weight");
    BLK_W(norm_added_q_w, "attn.norm_added_q.weight");
    BLK_W(norm_added_k_w, "attn.norm_added_k.weight");

    BLK_W(img_mod_w, "img_mod.1.weight"); BLK_W(img_mod_b, "img_mod.1.bias");
    BLK_W(img_mlp_fc1_w, "img_mlp.net.0.proj.weight"); BLK_W(img_mlp_fc1_b, "img_mlp.net.0.proj.bias");
    BLK_W(img_mlp_fc2_w, "img_mlp.net.2.weight"); BLK_W(img_mlp_fc2_b, "img_mlp.net.2.bias");

    BLK_W(txt_mod_w, "txt_mod.1.weight"); BLK_W(txt_mod_b, "txt_mod.1.bias");
    BLK_W(txt_mlp_fc1_w, "txt_mlp.net.0.proj.weight"); BLK_W(txt_mlp_fc1_b, "txt_mlp.net.0.proj.bias");
    BLK_W(txt_mlp_fc2_w, "txt_mlp.net.2.weight"); BLK_W(txt_mlp_fc2_b, "txt_mlp.net.2.bias");

    #undef BLK_W

    if (!ok) {
        qimg_free_block(b);
        return -1;
    }
    return 0;
}


/* ---- Kernel launch helpers ---- */

static void op_gemm(hip_qimg_runner *r, void *Y, void *W, void *X, void *bias,
                    int n_out, int n_in, int n_tok) {
    struct { void *Y; void *W; void *X; void *bias; int n_out; int n_in; int n_tok; } args =
        { Y, W, X, bias, n_out, n_in, n_tok };
    size_t sz = sizeof(args);
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END };
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    hipModuleLaunchKernel(r->fn_gemm, gx, gy, 1, 16, 16, 1,
                          0, NULL, NULL, config);
}

static void op_bf16_trunc(hip_qimg_runner *r, void *x, int n) {
    struct { void *x; int n; } args = { x, n };
    size_t sz = sizeof(args);
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END };
    hipModuleLaunchKernel(r->fn_bf16_trunc, (unsigned)((n+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, NULL, config);
}

static void op_gemm_bf16(hip_qimg_runner *r, void *Y, void *W, void *X, void *bias,
                         int n_out, int n_in, int n_tok) {
    op_gemm(r, Y, W, X, bias, n_out, n_in, n_tok);
    op_bf16_trunc(r, Y, n_out * n_tok);
}

static void op_silu(hip_qimg_runner *r, void *x, int n) {
    struct { void *x; int n; } args = { x, n };
    size_t sz = sizeof(args);
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END };
    hipModuleLaunchKernel(r->fn_silu, (unsigned)((n+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, NULL, config);
}

static void op_gelu(hip_qimg_runner *r, void *x, int n) {
    struct { void *x; int n; } args = { x, n };
    size_t sz = sizeof(args);
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END };
    hipModuleLaunchKernel(r->fn_gelu, (unsigned)((n+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, NULL, config);
}

static void op_adaln(hip_qimg_runner *r, void *out, void *x,
                     void *shift, void *scale, int N, int dim) {
    struct { void *out; void *x; void *shift; void *scale; int N; int dim; } args =
        { out, x, shift, scale, N, dim };
    size_t sz = sizeof(args);
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END };
    hipModuleLaunchKernel(r->fn_adaln, (unsigned)N, 1, 1, 256, 1, 1,
                          256 * sizeof(float), NULL, NULL, config);
}

static void op_rmsnorm_ph(hip_qimg_runner *r, void *x, void *w,
                          int N, int n_heads, int head_dim) {
    struct { void *x; void *w; int N; int n_heads; int head_dim; } args =
        { x, w, N, n_heads, head_dim };
    size_t sz = sizeof(args);
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END };
    hipModuleLaunchKernel(r->fn_rmsnorm_ph, (unsigned)N, (unsigned)n_heads, 1,
                          32, 1, 1, 0, NULL, NULL, config);
}

static void op_gated_add(hip_qimg_runner *r, void *x, void *proj,
                         void *gate, int N, int dim) {
    int total = N * dim;
    struct { void *x; void *proj; void *gate; int N; int dim; } args =
        { x, proj, gate, N, dim };
    size_t sz = sizeof(args);
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END };
    hipModuleLaunchKernel(r->fn_gated_add, (unsigned)((total+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, NULL, config);
}

static void op_attn(hip_qimg_runner *r, void *d_out, void *d_q,
                    void *d_k, void *d_v,
                    int n_tok, int n_heads, int head_dim) {
    int fa2_warps = 4, fa2_bkv = 16;
    unsigned gy = (unsigned)((n_tok + fa2_warps - 1) / fa2_warps);
    unsigned n_threads = (unsigned)(32 * fa2_warps);
    size_t smem = (size_t)2 * fa2_bkv * 128 * sizeof(float);
    struct { void *out; void *Q; void *K; void *V; int N; int nh; int hd; } args =
        { d_out, d_q, d_k, d_v, n_tok, n_heads, head_dim };
    size_t sz = sizeof(args);
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END };
    hipModuleLaunchKernel(r->fn_flash_attn,
                          (unsigned)n_heads, gy, 1,
                          n_threads, 1, 1,
                          smem, NULL, NULL, config);
}

static void op_rmsnorm_weighted(hip_qimg_runner *r, void *x, void *w, int N, int dim) {
    struct { void *x; void *w; int N; int dim; } args = { x, w, N, dim };
    size_t sz = sizeof(args);
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END };
    hipModuleLaunchKernel(r->fn_rmsnorm_weighted, (unsigned)N, 1, 1,
                          256, 1, 1, 256 * sizeof(float), NULL, NULL, config);
}

/* ---- VAE kernel helpers ---- */

static void vae_op_conv2d(hip_qimg_runner *r, void *out, void *inp,
                          void *w, void *b,
                          int ci, int h, int w_s, int co, int kh, int kw, int rep_pad) {
    int total = co * h * w_s;
    struct { void *out; void *inp; void *w; void *b;
             int ci; int h; int ws; int co; int kh; int kw; int rp; } args =
        { out, inp, w, b, ci, h, w_s, co, kh, kw, rep_pad };
    size_t sz = sizeof(args);
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END };
    hipModuleLaunchKernel(r->fn_vae_conv2d, (unsigned)((total+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, NULL, config);
}

static void vae_op_gn(hip_qimg_runner *r, void *out, void *inp,
                      void *gamma, int C, int spatial) {
    struct { void *out; void *inp; void *gamma; int C; int sp; } args =
        { out, inp, gamma, C, spatial };
    size_t sz = sizeof(args);
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END };
    hipModuleLaunchKernel(r->fn_vae_rmsnorm, (unsigned)((spatial+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, NULL, config);
}

static void vae_op_silu(hip_qimg_runner *r, void *x, int n) {
    struct { void *x; int n; } args = { x, n };
    size_t sz = sizeof(args);
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END };
    hipModuleLaunchKernel(r->fn_vae_silu, (unsigned)((n+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, NULL, config);
}

static void *vae_op_upsample(hip_qimg_runner *r, void *inp, int c, int h, int w) {
    int oh = h*2, ow = w*2;
    void *out = NULL;
    hipMalloc(&out, (size_t)c*oh*ow*sizeof(float));
    int total = c*oh*ow;
    struct { void *out; void *inp; int C; int H; int W; } args =
        { out, inp, c, h, w };
    size_t sz = sizeof(args);
    void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                       HIP_LAUNCH_PARAM_END };
    hipModuleLaunchKernel(r->fn_vae_up2x, (unsigned)((total+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, NULL, config);
    return out;
}

/* GPU VAE ResBlock */
static void *vae_resblock_gpu(hip_qimg_runner *r, void *x,
                               void *n1_g, void *c1_w, void *c1_b,
                               void *n2_g, void *c2_w, void *c2_b,
                               void *sc_w, void *sc_b,
                               int ci, int co, int h, int w) {
    int sp = h * w;
    void *tmp = NULL; hipMalloc(&tmp, (size_t)ci*sp*sizeof(float));
    vae_op_gn(r, tmp, x, n1_g, ci, sp);
    vae_op_silu(r, tmp, ci*sp);
    void *c1_out = NULL; hipMalloc(&c1_out, (size_t)co*sp*sizeof(float));
    vae_op_conv2d(r, c1_out, tmp, c1_w, c1_b, ci, h, w, co, 3, 3, 0);
    hipFree(tmp);

    tmp = NULL; hipMalloc(&tmp, (size_t)co*sp*sizeof(float));
    vae_op_gn(r, tmp, c1_out, n2_g, co, sp);
    vae_op_silu(r, tmp, co*sp);
    void *c2_out = NULL; hipMalloc(&c2_out, (size_t)co*sp*sizeof(float));
    vae_op_conv2d(r, c2_out, tmp, c2_w, c2_b, co, h, w, co, 3, 3, 0);
    hipFree(tmp); hipFree(c1_out);

    void *out = NULL; hipMalloc(&out, (size_t)co*sp*sizeof(float));
    if (sc_w) {
        vae_op_conv2d(r, out, x, sc_w, sc_b, ci, h, w, co, 1, 1, 0);
        /* out += c2_out (euler_step with dt=1) */
        int n = co * sp;
        struct { void *x; void *v; float dt; int n; } ea = { out, c2_out, 1.0f, n };
        size_t esz = sizeof(ea);
        void *ecfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &ea,
                         HIP_LAUNCH_PARAM_BUFFER_SIZE, &esz,
                         HIP_LAUNCH_PARAM_END };
        hipModuleLaunchKernel(r->fn_euler_step, (unsigned)((n+255)/256), 1, 1,
                              256, 1, 1, 0, NULL, NULL, ecfg);
    } else {
        int n = co * sp;
        hipMemcpy(out, x, (size_t)n * sizeof(float), hipMemcpyDeviceToDevice);
        struct { void *x; void *v; float dt; int n; } ea = { out, c2_out, 1.0f, n };
        size_t esz = sizeof(ea);
        void *ecfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &ea,
                         HIP_LAUNCH_PARAM_BUFFER_SIZE, &esz,
                         HIP_LAUNCH_PARAM_END };
        hipModuleLaunchKernel(r->fn_euler_step, (unsigned)((n+255)/256), 1, 1,
                              256, 1, 1, 0, NULL, NULL, ecfg);
    }
    hipFree(c2_out);
    return out;
}


/* ---- Init ---- */

hip_qimg_runner *hip_qimg_init(int device_id, int verbose) {
    if (rocewInit() != 0) {
        fprintf(stderr, "hip_qimg: rocewInit failed (no HIP runtime?)\n");
        return NULL;
    }
    HIP_CHECK_NULL(hipSetDevice(device_id));

    if (verbose) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, device_id);
        fprintf(stderr, "hip_qimg: %s (%.1f GB)\n",
                props.name, (float)props.totalGlobalMem / (1<<30));
    }

    /* Compile kernels */
    size_t len1 = strlen(hip_kernels_common_src);
    size_t len2 = strlen(hip_qimg_specific_kernels);
    char *full_src = (char *)malloc(len1 + len2 + 1);
    memcpy(full_src, hip_kernels_common_src, len1);
    memcpy(full_src + len1, hip_qimg_specific_kernels, len2);
    full_src[len1 + len2] = '\0';

    hipModule_t mod;
    int rc = hip_compile_kernels(&mod, device_id, full_src, "qimg.hip", verbose, "hip_qimg");
    free(full_src);
    if (rc < 0) return NULL;

    hip_qimg_runner *r = (hip_qimg_runner *)calloc(1, sizeof(*r));
    r->device_id = device_id;
    r->verbose = verbose;
    r->mod = mod;

    #define GET(field, name) hipModuleGetFunction(&r->field, mod, name)
    GET(fn_gemm, "gemm_f32_f32");
    GET(fn_layernorm, "layernorm_f32");
    GET(fn_silu, "silu_f32");
    GET(fn_gelu, "gelu_f32");
    GET(fn_adaln, "adaln_modulate_f32");
    GET(fn_gated_add, "gated_add_f32");
    GET(fn_rmsnorm_ph, "rmsnorm_per_head_f32");
    GET(fn_flash_attn, "flash_attn_f32");
    GET(fn_rope_2d, "rope_2d_f32");
    GET(fn_rope_1d, "rope_1d_f32");
    GET(fn_bf16_trunc, "truncate_bf16_f32");
    GET(fn_patchify, "patchify_f32");
    GET(fn_unpatchify, "unpatchify_f32");
    GET(fn_euler_step, "euler_step_f32");
    GET(fn_cfg_combine, "cfg_combine_f32");
    GET(fn_rmsnorm_weighted, "rmsnorm_weighted_f32");
    GET(fn_fp8_roundtrip, "quantize_fp8_roundtrip_f32");
    GET(fn_vae_conv2d, "vae_conv2d_f32");
    GET(fn_vae_rmsnorm, "vae_rmsnorm_f32");
    GET(fn_vae_silu, "vae_silu_f32");
    GET(fn_vae_up2x, "nn_upsample2x_f32");
    #undef GET

    if (verbose) fprintf(stderr, "hip_qimg: kernels compiled OK\n");
    return r;
}

/* ---- Load DiT ---- */

int hip_qimg_load_dit(hip_qimg_runner *r, const char *path) {
    fprintf(stderr, "hip_qimg: loading DiT %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    r->dit_st = st;

    r->dim = 3072; r->n_heads = 24; r->head_dim = 128;
    r->in_ch = 64; r->txt_dim = 3584; r->mlp_h = 12288;

    /* Count blocks */
    r->n_blocks = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const char *bp = strstr(nm, "transformer_blocks.");
        if (bp) {
            int blk = atoi(bp + 19);
            if (blk + 1 > r->n_blocks) r->n_blocks = blk + 1;
        }
    }

    /* Upload global weights (F32) */
    r->d_img_in_w = qimg_st_upload_f32(st, "img_in.weight");
    r->d_img_in_b = qimg_st_upload_f32(st, "img_in.bias");
    r->d_txt_in_w = qimg_st_upload_f32(st, "txt_in.weight");
    r->d_txt_in_b = qimg_st_upload_f32(st, "txt_in.bias");
    r->d_txt_norm_w = qimg_st_upload_f32(st, "txt_norm.weight");
    r->d_t_fc1_w = qimg_st_upload_f32(st, "time_text_embed.timestep_embedder.linear_1.weight");
    r->d_t_fc1_b = qimg_st_upload_f32(st, "time_text_embed.timestep_embedder.linear_1.bias");
    r->d_t_fc2_w = qimg_st_upload_f32(st, "time_text_embed.timestep_embedder.linear_2.weight");
    r->d_t_fc2_b = qimg_st_upload_f32(st, "time_text_embed.timestep_embedder.linear_2.bias");
    r->d_norm_out_w = qimg_st_upload_f32(st, "norm_out.linear.weight");
    r->d_norm_out_b = qimg_st_upload_f32(st, "norm_out.linear.bias");
    r->d_proj_out_w = qimg_st_upload_f32(st, "proj_out.weight");
    r->d_proj_out_b = qimg_st_upload_f32(st, "proj_out.bias");

    /* Preload as many blocks as fit in VRAM */
    {
        size_t free_mem = 0, total_mem = 0;
        hipMemGetInfo(&free_mem, &total_mem);
        /* ~324M params/block × 4 bytes = ~1.3 GB/block in F32 */
        size_t block_bytes = 1296ULL * 1024 * 1024;
        size_t workspace = 2ULL * 1024 * 1024 * 1024;
        int max_preload = (free_mem > workspace)
            ? (int)((free_mem - workspace) / block_bytes) : 0;
        if (max_preload > r->n_blocks) max_preload = r->n_blocks;

        r->gpu_blocks = (qimg_block_gpu *)calloc((size_t)r->n_blocks,
                                                  sizeof(qimg_block_gpu));
        r->n_preloaded = max_preload;
        fprintf(stderr, "hip_qimg: preloading %d/%d blocks to GPU "
                "(%.1f GB free, %.0f MB/block)\n",
                max_preload, r->n_blocks,
                (float)free_mem / (1<<30), (float)block_bytes / (1<<20));

        for (int i = 0; i < max_preload; i++) {
            if (qimg_load_block(r, i, &r->gpu_blocks[i]) != 0) {
                fprintf(stderr, "hip_qimg: stopped preloading at block %d (OOM)\n", i);
                r->n_preloaded = i;
                break;
            }
        }
        hipDeviceSynchronize();

        hipMemGetInfo(&free_mem, &total_mem);
        fprintf(stderr, "hip_qimg: after preload: %.1f GB free\n",
                (float)free_mem / (1<<30));
    }

    fprintf(stderr, "hip_qimg: loaded %d blocks, dim=%d\n", r->n_blocks, r->dim);
    return 0;
}

/* ---- Load VAE ---- */

int hip_qimg_load_vae(hip_qimg_runner *r, const char *path) {
    fprintf(stderr, "hip_qimg: loading VAE %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    r->vae_st = st;
    fprintf(stderr, "hip_qimg: VAE loaded (%d tensors)\n", st->n_tensors);
    return 0;
}

/* ---- Free ---- */

void hip_qimg_free(hip_qimg_runner *r) {
    if (!r) return;
    if (r->gpu_blocks) {
        for (int i = 0; i < r->n_preloaded; i++)
            qimg_free_block(&r->gpu_blocks[i]);
        free(r->gpu_blocks);
    }
    /* Free global weights */
    void **globals[] = {
        &r->d_img_in_w, &r->d_img_in_b, &r->d_txt_in_w, &r->d_txt_in_b,
        &r->d_txt_norm_w, &r->d_t_fc1_w, &r->d_t_fc1_b, &r->d_t_fc2_w,
        &r->d_t_fc2_b, &r->d_norm_out_w, &r->d_norm_out_b,
        &r->d_proj_out_w, &r->d_proj_out_b
    };
    for (int i = 0; i < 13; i++) {
        if (*globals[i]) { hipFree(*globals[i]); *globals[i] = NULL; }
    }
    if (r->dit_st) safetensors_close((st_context *)r->dit_st);
    if (r->vae_st) safetensors_close((st_context *)r->vae_st);
    if (r->mod) hipModuleUnload(r->mod);
    free(r);
}


/* ---- DiT single step ---- */

int hip_qimg_dit_step(hip_qimg_runner *r,
                      const float *img_tokens, int n_img,
                      const float *txt_tokens, int n_txt,
                      float timestep, float *out) {
    int dim = r->dim;
    int nh = r->n_heads, hd = r->head_dim;
    int in_ch = r->in_ch, txt_dim = r->txt_dim, mlp_h = r->mlp_h;
    int n_total = n_img + n_txt;

    /* Allocate GPU activation buffers */
    void *d_img = NULL, *d_txt = NULL, *d_t_emb = NULL;
    hipMalloc(&d_img, (size_t)n_img * dim * sizeof(float));
    hipMalloc(&d_txt, (size_t)n_txt * dim * sizeof(float));
    hipMalloc(&d_t_emb, (size_t)dim * sizeof(float));

    /* Upload inputs */
    void *d_img_in = NULL, *d_txt_in = NULL;
    hipMalloc(&d_img_in, (size_t)n_img * in_ch * sizeof(float));
    hipMemcpy(d_img_in, img_tokens, (size_t)n_img * in_ch * sizeof(float), hipMemcpyHostToDevice);
    hipMalloc(&d_txt_in, (size_t)n_txt * txt_dim * sizeof(float));
    hipMemcpy(d_txt_in, txt_tokens, (size_t)n_txt * txt_dim * sizeof(float), hipMemcpyHostToDevice);

    /* BF16 truncate inputs (match ComfyUI) */
    op_bf16_trunc(r, d_img_in, n_img * in_ch);
    op_bf16_trunc(r, d_txt_in, n_txt * txt_dim);

    /* 1. Timestep embedding: sinusoidal(256) → SiLU(GEMM) → GEMM */
    float t_sin[256];
    int half = 128;
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)i / (float)half * logf(10000.0f));
        float angle = timestep * freq;
        t_sin[i]        = cosf(angle);
        t_sin[half + i] = sinf(angle);
    }
    void *d_t_sin = NULL;
    hipMalloc(&d_t_sin, 256 * sizeof(float));
    hipMemcpy(d_t_sin, t_sin, 256 * sizeof(float), hipMemcpyHostToDevice);

    op_gemm_bf16(r, d_t_emb, r->d_t_fc1_w, d_t_sin, r->d_t_fc1_b, dim, 256, 1);
    op_silu(r, d_t_emb, dim);
    void *d_t_emb2 = NULL;
    hipMalloc(&d_t_emb2, (size_t)dim * sizeof(float));
    op_gemm_bf16(r, d_t_emb2, r->d_t_fc2_w, d_t_emb, r->d_t_fc2_b, dim, dim, 1);
    hipFree(d_t_emb); d_t_emb = d_t_emb2;
    hipFree(d_t_sin);

    /* 2. Text input: RMSNorm → Linear */
    if (r->d_txt_norm_w) {
        op_rmsnorm_weighted(r, d_txt_in, r->d_txt_norm_w, n_txt, txt_dim);
    }
    op_gemm_bf16(r, d_txt, r->d_txt_in_w, d_txt_in, r->d_txt_in_b, dim, txt_dim, n_txt);
    hipFree(d_txt_in);

    /* 3. Image input: GEMM(64→3072) */
    op_gemm_bf16(r, d_img, r->d_img_in_w, d_img_in, r->d_img_in_b, dim, in_ch, n_img);
    hipFree(d_img_in);

    /* BF16 truncation after projection */
    op_bf16_trunc(r, d_img, n_img * dim);
    op_bf16_trunc(r, d_txt, n_txt * dim);
    op_bf16_trunc(r, d_t_emb, dim);

    /* Scratch buffers */
    void *d_scratch1 = NULL, *d_scratch2 = NULL, *d_scratch3 = NULL;
    size_t max_scratch = (size_t)n_total * dim * sizeof(float);
    hipMalloc(&d_scratch1, max_scratch);
    hipMalloc(&d_scratch2, max_scratch);
    size_t ffn_scratch = (size_t)(n_img > n_txt ? n_img : n_txt) * mlp_h * sizeof(float);
    hipMalloc(&d_scratch3, ffn_scratch);

    /* Joint Q/K/V buffers */
    void *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_attn_out = NULL;
    hipMalloc(&d_q, (size_t)n_total * dim * sizeof(float));
    hipMalloc(&d_k, (size_t)n_total * dim * sizeof(float));
    hipMalloc(&d_v, (size_t)n_total * dim * sizeof(float));
    hipMalloc(&d_attn_out, (size_t)n_total * dim * sizeof(float));

    /* RoPE params */
    int hp_rope = (int)sqrtf((float)n_img);
    int wp_rope = n_img / hp_rope;
    float rope_theta = 10000.0f;
    int t_dim_rope = 16, h_dim_rope = 56, w_dim_rope = 56;

    /* 4. Process all blocks */
    for (int L = 0; L < r->n_blocks; L++) {
        if (r->verbose && (L % 10 == 0 || L == r->n_blocks - 1))
            fprintf(stderr, "\r  hip_qimg: block %d/%d", L + 1, r->n_blocks);

        /* Use preloaded block if available, otherwise load on-demand */
        qimg_block_gpu blk;
        int need_free = 0;
        if (L < r->n_preloaded && r->gpu_blocks[L].attn_q_w) {
            blk = r->gpu_blocks[L];
        } else {
            memset(&blk, 0, sizeof(blk));
            qimg_load_block(r, L, &blk);
            need_free = 1;
        }

        /* Image modulation: SiLU(t_emb) → Linear → 6×dim */
        void *d_t_silu = NULL;
        hipMalloc(&d_t_silu, (size_t)dim * sizeof(float));
        hipMemcpy(d_t_silu, d_t_emb, (size_t)dim * sizeof(float), hipMemcpyDeviceToDevice);
        op_silu(r, d_t_silu, dim);

        void *d_img_mod = NULL;
        hipMalloc(&d_img_mod, (size_t)6 * dim * sizeof(float));
        op_gemm_bf16(r, d_img_mod, blk.img_mod_w, d_t_silu, blk.img_mod_b, 6 * dim, dim, 1);

        void *d_txt_mod = NULL;
        hipMalloc(&d_txt_mod, (size_t)6 * dim * sizeof(float));
        op_gemm_bf16(r, d_txt_mod, blk.txt_mod_w, d_t_silu, blk.txt_mod_b, 6 * dim, dim, 1);
        hipFree(d_t_silu);

        /* Modulation offsets */
        #define MOD_OFF(base, idx) ((void *)((char *)(base) + (size_t)(idx) * dim * sizeof(float)))
        void *img_sh1 = MOD_OFF(d_img_mod, 0);
        void *img_sc1 = MOD_OFF(d_img_mod, 1);
        void *img_g1  = MOD_OFF(d_img_mod, 2);
        void *img_sh2 = MOD_OFF(d_img_mod, 3);
        void *img_sc2 = MOD_OFF(d_img_mod, 4);
        void *img_g2  = MOD_OFF(d_img_mod, 5);

        void *txt_sh1 = MOD_OFF(d_txt_mod, 0);
        void *txt_sc1 = MOD_OFF(d_txt_mod, 1);
        void *txt_g1  = MOD_OFF(d_txt_mod, 2);
        void *txt_sh2 = MOD_OFF(d_txt_mod, 3);
        void *txt_sc2 = MOD_OFF(d_txt_mod, 4);
        void *txt_g2  = MOD_OFF(d_txt_mod, 5);
        #undef MOD_OFF

        /* adaLN image → d_scratch1 */
        op_adaln(r, d_scratch1, d_img, img_sh1, img_sc1, n_img, dim);
        /* adaLN text → d_scratch2 */
        op_adaln(r, d_scratch2, d_txt, txt_sh1, txt_sc1, n_txt, dim);

        /* Image QKV → offset into joint buffers at [n_txt:] */
        void *d_img_q = (char *)d_q + (size_t)n_txt * dim * sizeof(float);
        void *d_img_k = (char *)d_k + (size_t)n_txt * dim * sizeof(float);
        void *d_img_v = (char *)d_v + (size_t)n_txt * dim * sizeof(float);
        op_gemm_bf16(r, d_img_q, blk.attn_q_w, d_scratch1, blk.attn_q_b, dim, dim, n_img);
        op_gemm_bf16(r, d_img_k, blk.attn_k_w, d_scratch1, blk.attn_k_b, dim, dim, n_img);
        op_gemm_bf16(r, d_img_v, blk.attn_v_w, d_scratch1, blk.attn_v_b, dim, dim, n_img);

        /* Text QKV → offset at [0:n_txt] */
        void *d_txt_q = d_q;
        void *d_txt_k = d_k;
        void *d_txt_v = d_v;
        op_gemm_bf16(r, d_txt_q, blk.attn_add_q_w, d_scratch2, blk.attn_add_q_b, dim, dim, n_txt);
        op_gemm_bf16(r, d_txt_k, blk.attn_add_k_w, d_scratch2, blk.attn_add_k_b, dim, dim, n_txt);
        op_gemm_bf16(r, d_txt_v, blk.attn_add_v_w, d_scratch2, blk.attn_add_v_b, dim, dim, n_txt);

        /* QK RMSNorm */
        op_rmsnorm_ph(r, d_img_q, blk.norm_q_w, n_img, nh, hd);
        op_rmsnorm_ph(r, d_img_k, blk.norm_k_w, n_img, nh, hd);
        op_rmsnorm_ph(r, d_txt_q, blk.norm_added_q_w, n_txt, nh, hd);
        op_rmsnorm_ph(r, d_txt_k, blk.norm_added_k_w, n_txt, nh, hd);

        /* RoPE */
        {
            struct { void *q; void *k; int n_tok; int nh; int hd;
                     int hp; int wp; int td; int hd2; int wd; float theta; } rope_args =
                { d_img_q, d_img_k, n_img, nh, hd, hp_rope, wp_rope,
                  t_dim_rope, h_dim_rope, w_dim_rope, rope_theta };
            size_t rsz = sizeof(rope_args);
            void *rcfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &rope_args,
                             HIP_LAUNCH_PARAM_BUFFER_SIZE, &rsz,
                             HIP_LAUNCH_PARAM_END };
            hipModuleLaunchKernel(r->fn_rope_2d, (unsigned)n_img, 1, 1,
                                  (unsigned)nh, 1, 1, 0, NULL, NULL, rcfg);

            int txt_start = hp_rope > wp_rope ? hp_rope / 2 : wp_rope / 2;
            struct { void *q; void *k; int n_tok; int nh; int hd; int ts;
                     int td; int hd2; int wd; float theta; } trope_args =
                { d_txt_q, d_txt_k, n_txt, nh, hd, txt_start,
                  t_dim_rope, h_dim_rope, w_dim_rope, rope_theta };
            size_t trsz = sizeof(trope_args);
            void *trcfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &trope_args,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &trsz,
                              HIP_LAUNCH_PARAM_END };
            hipModuleLaunchKernel(r->fn_rope_1d, (unsigned)n_txt, 1, 1,
                                  (unsigned)nh, 1, 1, 0, NULL, NULL, trcfg);
        }

        /* Joint attention */
        op_attn(r, d_attn_out, d_q, d_k, d_v, n_total, nh, hd);

        /* Output projections */
        void *d_img_attn = (char *)d_attn_out + (size_t)n_txt * dim * sizeof(float);
        void *d_txt_attn = d_attn_out;
        op_gemm_bf16(r, d_scratch1, blk.attn_out_w, d_img_attn, blk.attn_out_b, dim, dim, n_img);
        op_gemm_bf16(r, d_scratch2, blk.attn_add_out_w, d_txt_attn, blk.attn_add_out_b, dim, dim, n_txt);

        /* Gated residual */
        op_gated_add(r, d_img, d_scratch1, img_g1, n_img, dim);
        op_gated_add(r, d_txt, d_scratch2, txt_g1, n_txt, dim);

        /* MLP: Image (GELU) */
        op_adaln(r, d_scratch1, d_img, img_sh2, img_sc2, n_img, dim);
        op_gemm_bf16(r, d_scratch3, blk.img_mlp_fc1_w, d_scratch1, blk.img_mlp_fc1_b,
                     mlp_h, dim, n_img);
        op_gelu(r, d_scratch3, n_img * mlp_h);
        op_gemm_bf16(r, d_scratch1, blk.img_mlp_fc2_w, d_scratch3, blk.img_mlp_fc2_b,
                     dim, mlp_h, n_img);
        op_gated_add(r, d_img, d_scratch1, img_g2, n_img, dim);

        /* MLP: Text (GELU) */
        op_adaln(r, d_scratch2, d_txt, txt_sh2, txt_sc2, n_txt, dim);
        op_gemm_bf16(r, d_scratch3, blk.txt_mlp_fc1_w, d_scratch2, blk.txt_mlp_fc1_b,
                     mlp_h, dim, n_txt);
        op_gelu(r, d_scratch3, n_txt * mlp_h);
        op_gemm_bf16(r, d_scratch2, blk.txt_mlp_fc2_w, d_scratch3, blk.txt_mlp_fc2_b,
                     dim, mlp_h, n_txt);
        op_gated_add(r, d_txt, d_scratch2, txt_g2, n_txt, dim);

        /* BF16 truncation */
        op_bf16_trunc(r, d_img, n_img * dim);
        op_bf16_trunc(r, d_txt, n_txt * dim);

        /* Free modulation + on-demand block weights */
        hipFree(d_img_mod); hipFree(d_txt_mod);
        if (need_free) {
            hipDeviceSynchronize();
            void **ptrs = (void **)&blk;
            int np = sizeof(qimg_block_gpu) / sizeof(void *);
            for (int pi = 0; pi < np; pi++)
                if (ptrs[pi]) hipFree(ptrs[pi]);
        }
    }
    if (r->verbose) fprintf(stderr, "\n");

    /* 5. Final output: adaLN → proj_out */
    {
        void *d_t_silu = NULL;
        hipMalloc(&d_t_silu, (size_t)dim * sizeof(float));
        hipMemcpy(d_t_silu, d_t_emb, (size_t)dim * sizeof(float), hipMemcpyDeviceToDevice);
        op_silu(r, d_t_silu, dim);
        void *d_final_mod = NULL;
        hipMalloc(&d_final_mod, (size_t)2 * dim * sizeof(float));
        op_gemm_bf16(r, d_final_mod, r->d_norm_out_w, d_t_silu, r->d_norm_out_b,
                     2 * dim, dim, 1);
        hipFree(d_t_silu);

        void *f_scale = d_final_mod;
        void *f_shift = (char *)d_final_mod + (size_t)dim * sizeof(float);
        op_adaln(r, d_scratch1, d_img, f_shift, f_scale, n_img, dim);
        hipFree(d_final_mod);

        void *d_out = NULL;
        hipMalloc(&d_out, (size_t)n_img * in_ch * sizeof(float));
        op_gemm_bf16(r, d_out, r->d_proj_out_w, d_scratch1, r->d_proj_out_b,
                     in_ch, dim, n_img);

        hipDeviceSynchronize();
        hipMemcpy(out, d_out, (size_t)n_img * in_ch * sizeof(float), hipMemcpyDeviceToHost);
        hipFree(d_out);
    }

    /* Cleanup */
    hipFree(d_img); hipFree(d_txt); hipFree(d_t_emb);
    hipFree(d_scratch1); hipFree(d_scratch2); hipFree(d_scratch3);
    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_attn_out);

    return 0;
}


/* ---- VAE decode on GPU ---- */

int hip_qimg_vae_decode(hip_qimg_runner *r,
                        const float *latent, int lat_h, int lat_w,
                        float *out_rgb) {
    st_context *st = (st_context *)r->vae_st;
    if (!st) { fprintf(stderr, "hip_qimg: VAE not loaded\n"); return -1; }

    /* Free preloaded DiT blocks to make room */
    if (r->gpu_blocks) {
        for (int i = 0; i < r->n_preloaded; i++)
            qimg_free_block(&r->gpu_blocks[i]);
        r->n_preloaded = 0;
    }
    hipDeviceSynchronize();

    int h = lat_h, w = lat_w, c = 16;
    fprintf(stderr, "hip_qimg_vae: decoding [%d, %d, %d] on GPU\n", c, h, w);

    /* Upload latent */
    void *d_x = NULL;
    hipMalloc(&d_x, (size_t)c * h * w * sizeof(float));
    hipMemcpy(d_x, latent, (size_t)c * h * w * sizeof(float), hipMemcpyHostToDevice);

    /* post_quant_conv */
    void *d_pqc_w = qimg_st_upload_f32(st, "conv2.weight");
    void *d_pqc_b = qimg_st_upload_f32(st, "conv2.bias");
    if (d_pqc_w) {
        void *d_tmp = NULL; hipMalloc(&d_tmp, (size_t)c*h*w*sizeof(float));
        vae_op_conv2d(r, d_tmp, d_x, d_pqc_w, d_pqc_b, c, h, w, c, 1, 1, 0);
        hipFree(d_x); d_x = d_tmp;
        hipFree(d_pqc_w); hipFree(d_pqc_b);
    }

    /* decoder.conv1: 16→384, 3×3 */
    int co_c1, ci_c1;
    void *d_c1_w = qimg_upload_conv3d(st, "decoder.conv1.weight", &co_c1, &ci_c1);
    void *d_c1_b = qimg_st_upload_f32(st, "decoder.conv1.bias");
    c = co_c1;
    {
        void *d_tmp = NULL; hipMalloc(&d_tmp, (size_t)c*h*w*sizeof(float));
        vae_op_conv2d(r, d_tmp, d_x, d_c1_w, d_c1_b, ci_c1, h, w, c, 3, 3, 0);
        hipFree(d_x); d_x = d_tmp;
        hipFree(d_c1_w); hipFree(d_c1_b);
    }
    fprintf(stderr, "  after conv1: [%d, %d, %d]\n", c, h, w);

    /* Load resblock weights helper macro */
    #define LOAD_RB_NAMED(pfx_str, n1, c1w, c1b, n2, c2w, c2b, scw, scb) \
        void *n1, *c1w, *c1b, *n2, *c2w, *c2b, *scw = NULL, *scb = NULL; \
        { char _nm[256]; \
          snprintf(_nm, sizeof(_nm), "%s.residual.0.gamma", pfx_str); n1 = qimg_st_upload_f32(st, _nm); \
          snprintf(_nm, sizeof(_nm), "%s.residual.2.weight", pfx_str); { int _co, _ci; c1w = qimg_upload_conv3d(st, _nm, &_co, &_ci); } \
          snprintf(_nm, sizeof(_nm), "%s.residual.2.bias", pfx_str); c1b = qimg_st_upload_f32(st, _nm); \
          snprintf(_nm, sizeof(_nm), "%s.residual.3.gamma", pfx_str); n2 = qimg_st_upload_f32(st, _nm); \
          snprintf(_nm, sizeof(_nm), "%s.residual.6.weight", pfx_str); { int _co2, _ci2; c2w = qimg_upload_conv3d(st, _nm, &_co2, &_ci2); } \
          snprintf(_nm, sizeof(_nm), "%s.residual.6.bias", pfx_str); c2b = qimg_st_upload_f32(st, _nm); \
          snprintf(_nm, sizeof(_nm), "%s.shortcut.weight", pfx_str); \
          if (safetensors_find(st, _nm) >= 0) { scw = qimg_st_upload_f32(st, _nm); \
            snprintf(_nm, sizeof(_nm), "%s.shortcut.bias", pfx_str); scb = qimg_st_upload_f32(st, _nm); } }

    /* mid.0 */
    { LOAD_RB_NAMED("decoder.middle.0", n1, c1w, c1b, n2, c2w, c2b, scw, scb);
      void *d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b, n2, c2w, c2b, scw, scb, c, c, h, w);
      hipFree(d_x); d_x = d_tmp;
      hipFree(n1); hipFree(c1w); hipFree(c1b); hipFree(n2); hipFree(c2w); hipFree(c2b);
      if (scw) { hipFree(scw); } if (scb) { hipFree(scb); } }

    /* Middle attention: CPU fallback for spatial self-attention */
    {
        int spatial = h * w;
        void *d_gn_g = qimg_st_upload_f32(st, "decoder.middle.1.norm.gamma");
        void *d_qkv_w = qimg_st_upload_f32(st, "decoder.middle.1.to_qkv.weight");
        void *d_qkv_b = qimg_st_upload_f32(st, "decoder.middle.1.to_qkv.bias");
        void *d_proj_w = qimg_st_upload_f32(st, "decoder.middle.1.proj.weight");
        void *d_proj_b = qimg_st_upload_f32(st, "decoder.middle.1.proj.bias");

        void *d_normed = NULL; hipMalloc(&d_normed, (size_t)c*spatial*sizeof(float));
        vae_op_gn(r, d_normed, d_x, d_gn_g, c, spatial);
        hipFree(d_gn_g);

        void *d_qkv = NULL; hipMalloc(&d_qkv, (size_t)3*c*spatial*sizeof(float));
        vae_op_conv2d(r, d_qkv, d_normed, d_qkv_w, d_qkv_b, c, h, w, 3*c, 1, 1, 0);
        hipFree(d_normed); hipFree(d_qkv_w); hipFree(d_qkv_b);

        /* CPU attention */
        float *h_qkv = (float *)malloc((size_t)3 * c * spatial * sizeof(float));
        hipMemcpy(h_qkv, d_qkv, (size_t)3 * c * spatial * sizeof(float), hipMemcpyDeviceToHost);
        hipFree(d_qkv);

        float *h_qkv_t = (float *)malloc((size_t)3 * c * spatial * sizeof(float));
        for (int s_pos = 0; s_pos < spatial; s_pos++)
            for (int ch = 0; ch < 3 * c; ch++)
                h_qkv_t[s_pos * 3 * c + ch] = h_qkv[ch * spatial + s_pos];
        free(h_qkv);

        float *h_q = (float *)malloc((size_t)spatial * c * sizeof(float));
        float *h_k = (float *)malloc((size_t)spatial * c * sizeof(float));
        float *h_v = (float *)malloc((size_t)spatial * c * sizeof(float));
        for (int s_pos = 0; s_pos < spatial; s_pos++) {
            memcpy(h_q + (size_t)s_pos * c, h_qkv_t + (size_t)s_pos * 3 * c,       (size_t)c * sizeof(float));
            memcpy(h_k + (size_t)s_pos * c, h_qkv_t + (size_t)s_pos * 3 * c + c,   (size_t)c * sizeof(float));
            memcpy(h_v + (size_t)s_pos * c, h_qkv_t + (size_t)s_pos * 3 * c + 2*c, (size_t)c * sizeof(float));
        }

        float *h_attn = (float *)malloc((size_t)spatial * c * sizeof(float));
        float scale_at = 1.0f / sqrtf((float)c);
        for (int i = 0; i < spatial; i++) {
            float mx = -1e30f;
            for (int j = 0; j < spatial; j++) {
                float dot = 0;
                for (int d = 0; d < c; d++) dot += h_q[i*c+d] * h_k[j*c+d];
                dot *= scale_at;
                if (dot > mx) mx = dot;
            }
            float esum = 0;
            memset(h_attn + i*c, 0, (size_t)c * sizeof(float));
            for (int j = 0; j < spatial; j++) {
                float dot = 0;
                for (int d = 0; d < c; d++) dot += h_q[i*c+d] * h_k[j*c+d];
                float w_at = expf(dot * scale_at - mx);
                esum += w_at;
                for (int d = 0; d < c; d++) h_attn[i*c+d] += w_at * h_v[j*c+d];
            }
            float inv = 1.0f / esum;
            for (int d = 0; d < c; d++) h_attn[i*c+d] *= inv;
        }
        free(h_qkv_t); free(h_q); free(h_k); free(h_v);

        float *h_attn_chw = (float *)malloc((size_t)c * spatial * sizeof(float));
        for (int s_pos = 0; s_pos < spatial; s_pos++)
            for (int ch = 0; ch < c; ch++)
                h_attn_chw[ch * spatial + s_pos] = h_attn[s_pos * c + ch];
        free(h_attn);

        void *d_attn_chw = NULL; hipMalloc(&d_attn_chw, (size_t)c*spatial*sizeof(float));
        hipMemcpy(d_attn_chw, h_attn_chw, (size_t)c*spatial*sizeof(float), hipMemcpyHostToDevice);
        free(h_attn_chw);

        void *d_proj_out_v = NULL; hipMalloc(&d_proj_out_v, (size_t)c*spatial*sizeof(float));
        vae_op_conv2d(r, d_proj_out_v, d_attn_chw, d_proj_w, d_proj_b, c, h, w, c, 1, 1, 0);
        hipFree(d_attn_chw); hipFree(d_proj_w); hipFree(d_proj_b);

        /* Residual: d_x += d_proj_out_v */
        {
            int n = c * spatial;
            struct { void *x; void *v; float dt; int n; } ea = { d_x, d_proj_out_v, 1.0f, n };
            size_t esz = sizeof(ea);
            void *ecfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &ea,
                             HIP_LAUNCH_PARAM_BUFFER_SIZE, &esz,
                             HIP_LAUNCH_PARAM_END };
            hipModuleLaunchKernel(r->fn_euler_step, (unsigned)((n+255)/256), 1, 1,
                                  256, 1, 1, 0, NULL, NULL, ecfg);
        }
        hipFree(d_proj_out_v);
        hipDeviceSynchronize();
    }

    /* mid.2 */
    { LOAD_RB_NAMED("decoder.middle.2", n1, c1w, c1b, n2, c2w, c2b, scw, scb);
      void *d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b, n2, c2w, c2b, scw, scb, c, c, h, w);
      hipFree(d_x); d_x = d_tmp;
      hipFree(n1); hipFree(c1w); hipFree(c1b); hipFree(n2); hipFree(c2w); hipFree(c2b);
      if (scw) { hipFree(scw); } if (scb) { hipFree(scb); } }
    fprintf(stderr, "  after middle: [%d, %d, %d]\n", c, h, w);

    /* Upsample blocks 0-14 */
    for (int i = 0; i < 15; i++) {
        char pfx[128];
        snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.residual.2.weight", i);
        if (safetensors_find(st, pfx) >= 0) {
            char rb_pfx[128];
            snprintf(rb_pfx, sizeof(rb_pfx), "decoder.upsamples.%d", i);
            LOAD_RB_NAMED(rb_pfx, n1, c1w, c1b, n2, c2w, c2b, scw, scb);
            snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.residual.2.weight", i);
            int _idx = safetensors_find(st, pfx);
            int new_co = (int)safetensors_shape(st, _idx)[0];
            int old_ci = c;
            void *d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b, n2, c2w, c2b,
                                           scw, scb, old_ci, new_co, h, w);
            hipFree(d_x); d_x = d_tmp;
            c = new_co;
            hipFree(n1); hipFree(c1w); hipFree(c1b);
            hipFree(n2); hipFree(c2w); hipFree(c2b);
            if (scw) { hipFree(scw); } if (scb) { hipFree(scb); }
        }

        snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.resample.1.weight", i);
        if (safetensors_find(st, pfx) >= 0) {
            void *rs_w = qimg_st_upload_f32(st, pfx);
            snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.resample.1.bias", i);
            void *rs_b = qimg_st_upload_f32(st, pfx);
            void *d_up = vae_op_upsample(r, d_x, c, h, w);
            h *= 2; w *= 2;
            snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.resample.1.weight", i);
            int rs_idx = safetensors_find(st, pfx);
            int new_c = (int)safetensors_shape(st, rs_idx)[0];
            void *d_tmp = NULL; hipMalloc(&d_tmp, (size_t)new_c*h*w*sizeof(float));
            vae_op_conv2d(r, d_tmp, d_up, rs_w, rs_b, c, h, w, new_c, 3, 3, 0);
            hipFree(d_up); hipFree(d_x);
            hipFree(rs_w); hipFree(rs_b);
            d_x = d_tmp; c = new_c;
            fprintf(stderr, "  upsample %d: [%d, %d, %d]\n", i, c, h, w);
        }
    }
    #undef LOAD_RB_NAMED

    /* Head: GroupNorm → SiLU → Conv(96→3) */
    {
        void *d_gn = qimg_st_upload_f32(st, "decoder.head.0.gamma");
        int spatial = h * w;
        void *d_tmp = NULL; hipMalloc(&d_tmp, (size_t)c*spatial*sizeof(float));
        vae_op_gn(r, d_tmp, d_x, d_gn, c, spatial);
        vae_op_silu(r, d_tmp, c * spatial);
        hipFree(d_gn);

        int head_co, head_ci;
        void *d_hw = qimg_upload_conv3d(st, "decoder.head.2.weight", &head_co, &head_ci);
        void *d_hb = qimg_st_upload_f32(st, "decoder.head.2.bias");
        void *d_rgb = NULL; hipMalloc(&d_rgb, (size_t)3*spatial*sizeof(float));
        vae_op_conv2d(r, d_rgb, d_tmp, d_hw, d_hb, c, h, w, 3, 3, 3, 0);
        hipFree(d_tmp); hipFree(d_x); hipFree(d_hw); hipFree(d_hb);
        d_x = d_rgb;
        c = 3;
    }

    hipDeviceSynchronize();
    hipMemcpy(out_rgb, d_x, (size_t)3 * h * w * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_x);

    fprintf(stderr, "hip_qimg_vae: decode complete [%d, %d, %d]\n", c, h, w);
    return 0;
}
