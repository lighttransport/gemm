/*
 * cuda_paint_vae_runner.h - Header-only CUDA port of the SD-2.1 paint VAE
 * (encoder + decoder), extracted from test_paint_vae.c (Phase 4.12 step 2).
 *
 * Mirrors cuda_paint_unet_runner.h: the heavyweight IMPLEMENTATION macros for
 * cuda_runner_common.h and safetensors.h are gated behind
 * CUDA_PAINT_VAE_RUNNER_IMPLEMENTATION so a second TU (the upcoming top-level
 * paint pipeline runner) can include this header without duplicate-symbol
 * conflicts. Exactly one TU per binary defines the macro.
 *
 * The per-stage helpers remain `static` (file-local) — adequate while each
 * consumer is its own binary; revisit when we link multiple consumers into
 * one TU.
 */

#ifndef CUDA_PAINT_VAE_RUNNER_H_
#define CUDA_PAINT_VAE_RUNNER_H_

#include "../cuew.h"
/* cuda_runner_common helpers are file-local `static`, so every including TU
 * needs the impls (no link conflict). Define unconditionally. */
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "cuda_paint_vae_kernels.h"

#ifdef CUDA_PAINT_VAE_RUNNER_IMPLEMENTATION
#define SAFETENSORS_IMPLEMENTATION
#endif
#include "safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ===== .npy I/O =========================================================== */

/* Read a flat float32 .npy file into a malloc'd buffer; returns dims/shape.
 * Only handles the simple case: '<f4', fortran_order=False, 1-4 dims. */
static float *read_npy_f32(const char *path, int *out_ndim,
                            uint64_t *out_shape, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return NULL; }
    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6)) {
        fprintf(stderr, "ERROR: not a .npy file: %s\n", path); fclose(f); return NULL;
    }
    uint8_t ver[2]; fread(ver, 1, 2, f);
    uint16_t hlen; fread(&hlen, 2, 1, f);
    char hdr[1024];
    if (hlen >= sizeof(hdr)) { fclose(f); return NULL; }
    fread(hdr, 1, hlen, f); hdr[hlen] = 0;
    if (!strstr(hdr, "'descr': '<f4'")) {
        fprintf(stderr, "ERROR: expected <f4 dtype, got %s\n", hdr);
        fclose(f); return NULL;
    }
    /* parse shape (D0, D1, ...) */
    const char *p = strstr(hdr, "'shape': (");
    if (!p) { fclose(f); return NULL; }
    p += strlen("'shape': (");
    int nd = 0; uint64_t shape[8]; size_t total = 1;
    while (*p && *p != ')') {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ')') break;
        char *end;
        uint64_t v = strtoull(p, &end, 10);
        shape[nd++] = v; total *= v;
        p = end;
    }
    *out_ndim = nd;
    for (int i = 0; i < nd; i++) out_shape[i] = shape[i];
    *out_n = total;
    float *buf = (float *)malloc(total * sizeof(float));
    if (fread(buf, sizeof(float), total, f) != total) {
        fprintf(stderr, "ERROR: short read on %s\n", path);
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    return buf;
}

static void write_npy_f32(const char *path, const float *data,
                            const int *shape, int ndim) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite("\x93NUMPY", 1, 6, f);
    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
    char hdr[256], shape_s[128] = "";
    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        char tmp[32]; snprintf(tmp, sizeof(tmp), "%d, ", shape[i]);
        strcat(shape_s, tmp); total *= (size_t)shape[i];
    }
    int hlen = snprintf(hdr, sizeof(hdr),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", shape_s);
    int tot = 10 + hlen + 1;
    int pad = ((tot + 63) / 64) * 64 - tot;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(hdr, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), total, f);
    fclose(f);
}

/* ===== Weight upload ====================================================== */

static CUdeviceptr upload_st(const st_context *st, const char *name,
                              size_t *out_n) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        fprintf(stderr, "ERROR: tensor not found: %s\n", name);
        return 0;
    }
    const char *dt = safetensors_dtype(st, idx);
    if (strcmp(dt, "F32")) {
        fprintf(stderr, "ERROR: %s dtype %s, expected F32\n", name, dt);
        return 0;
    }
    size_t bytes = safetensors_nbytes(st, idx);
    CUdeviceptr d;
    cuMemAlloc(&d, bytes);
    cuMemcpyHtoD(d, safetensors_data(st, idx), bytes);
    if (out_n) *out_n = bytes / sizeof(float);
    return d;
}

/* ===== Kernel handles ===================================================== */

typedef struct {
    CUmodule mod;
    CUfunction f_gn;       /* vae_groupnorm_f32 */
    CUfunction f_conv;     /* vae_conv2d_f32 */
    CUfunction f_conv_down;/* vae_conv2d_down_f32 */
    CUfunction f_up2x;     /* vae_upsample2x_f32 */
    CUfunction f_add;      /* vae_add_f32 */
    CUfunction f_attn;     /* vae_attn_f32 */
    CUfunction f_chw_nc;   /* vae_chw_to_nc_f32 */
    CUfunction f_nc_chw;   /* vae_nc_to_chw_f32 */
    /* TC dispatch (Phase 4.13): may be NULL if compile/lookup failed. */
    CUfunction f_im2col_3x3_p1;        /* pvae_im2col_3x3_p1_f32 */
    CUfunction f_im2col_3x3_p1_s2;     /* pvae_im2col_3x3_p1_s2_f32 */
    CUfunction f_im2col_3x3_p1_tiled;  /* pvae_im2col_3x3_p1_tiled_f32 */
    CUfunction f_t_hwc_chw;            /* pvae_t_hwc_to_chw_f32 */
    CUfunction f_t_hwc_chw_tiled;      /* pvae_t_hwc_to_chw_tiled_f32 */
    CUfunction f_gemm_fp8;          /* gemm_bf16_pipe_scaled_f32 */
    CUfunction f_gemm_fp8_mt4;      /* gemm_bf16_pipe_mt4_scaled_f32 */
    CUfunction f_gemm_fp8_v7_fused; /* gemm_fp8_v7_fused */
    CUfunction f_gemm_fp8_v7_fused_p2; /* gemm_fp8_v7_fused_p2 (2x2 panel) */
    CUfunction f_gemm_bf16_v7;      /* gemm_bf16_v7 */
    CUfunction f_quant_bf16;        /* quant_bf16 */
    CUfunction f_add_bias_f32;      /* add_bias_inplace_f32 */
    CUfunction f_reduce_max_abs;    /* reduce_max_abs_f32 */
    CUfunction f_quantize_fp8;      /* quantize_to_fp8_e4m3 */
} pvae_kernels;

/* ===== Kernel launchers =================================================== */

static void k_groupnorm(const pvae_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                         CUdeviceptr gamma, CUdeviceptr beta,
                         int C, int spatial, int num_groups, int do_silu) {
    void *args[] = { &out, &in, &gamma, &beta, &C, &spatial,
                     &num_groups, &do_silu };
    /* PVAE_GN_THREADS=128, smem = threads*4 bytes */
    cuLaunchKernel(kk->f_gn, (unsigned)num_groups, 1, 1, 128, 1, 1,
                    128 * sizeof(float), 0, args, NULL);
}

/* ===== TC dispatch state (Phase 4.13) =====================================
 *
 * Mirror of cuda_paint_unet_runner.h's FP8/BF16 dispatch infra, prefixed
 * `pvae_` and `g_pvae_` so VAE TU and UNet TU don't collide on file-local
 * statics. Call pvae_init_runtime(kk) once after kernel resolution to read
 * env vars and decide which paths are eligible. Each k_conv/k_conv_down call
 * then dispatches through the same gemm_fp8_v7_fused / gemm_bf16_v7 /
 * gemm_*_pipe_*_scaled_f32 kernels the UNet uses.
 */
typedef struct {
    CUdeviceptr w_f32;
    CUdeviceptr w_fp8;
    CUdeviceptr w_scale;
    CUdeviceptr w_bf16;
    int n_out, n_in;
    float w_scale_host;
} pvae_fp8_entry;

#define PVAE_FP8_REG_MAX 1024
static int g_pvae_use_fp8_gemm     = 0;
static int g_pvae_use_fp8_gemm_mt4 = 0;
static int g_pvae_use_bf16_gemm    = 0;
static int g_pvae_use_fp8_v7       = 0;
static int g_pvae_use_fp8_v7_p2    = 0;
static int g_pvae_use_fp8_conv     = 0;
static pvae_fp8_entry g_pvae_fp8_reg[PVAE_FP8_REG_MAX];
static int    g_pvae_n_fp8_reg = 0;
/* Conv scratch (xcol / yt). Lazy-grown. */
static CUdeviceptr g_pvae_d_xcol = 0; static size_t g_pvae_xcol_max_bytes = 0;
static CUdeviceptr g_pvae_d_yt   = 0; static size_t g_pvae_yt_max_bytes   = 0;
/* BF16-quant activation scratch (gemm_bf16_v7 path). */
static CUdeviceptr g_pvae_d_xbf = 0; static size_t g_pvae_xbf_max_elem = 0;
/* FP8 v7 activation scratches: X_fp8 + per-tensor x_scale + x_max. */
static CUdeviceptr g_pvae_d_xfp8 = 0; static size_t g_pvae_xfp8_max_elem = 0;
static CUdeviceptr g_pvae_d_xscale = 0;
static CUdeviceptr g_pvae_d_xmax   = 0;
static CUdeviceptr g_pvae_v7_last_x = 0;
static int         g_pvae_v7_last_n = 0;

static int pvae_xfp8_ensure(size_t n_elem) {
    size_t need = n_elem; /* uint8 */
    if (need <= g_pvae_xfp8_max_elem && g_pvae_d_xfp8) return 0;
    if (g_pvae_d_xfp8) { cuMemFree(g_pvae_d_xfp8); g_pvae_d_xfp8 = 0; }
    if (cuMemAlloc(&g_pvae_d_xfp8, need) != CUDA_SUCCESS) {
        g_pvae_d_xfp8 = 0; g_pvae_xfp8_max_elem = 0;
        g_pvae_v7_last_x = 0; g_pvae_v7_last_n = 0;
        return -1;
    }
    g_pvae_xfp8_max_elem = n_elem;
    g_pvae_v7_last_x = 0; g_pvae_v7_last_n = 0;
    if (!g_pvae_d_xscale) cuMemAlloc(&g_pvae_d_xscale, sizeof(float));
    if (!g_pvae_d_xmax)   cuMemAlloc(&g_pvae_d_xmax,   sizeof(float));
    return 0;
}

static int pvae_xbf_ensure(size_t n_elem) {
    size_t need = n_elem * sizeof(unsigned short);
    size_t cur  = g_pvae_xbf_max_elem * sizeof(unsigned short);
    if (need <= cur) return 0;
    if (g_pvae_d_xbf) { cuMemFree(g_pvae_d_xbf); g_pvae_d_xbf = 0; }
    if (cuMemAlloc(&g_pvae_d_xbf, need) != CUDA_SUCCESS) {
        g_pvae_d_xbf = 0; g_pvae_xbf_max_elem = 0; return -1;
    }
    g_pvae_xbf_max_elem = n_elem;
    return 0;
}

static int pvae_conv_scratch_ensure(size_t xcol_bytes, size_t yt_bytes) {
    if (xcol_bytes > g_pvae_xcol_max_bytes) {
        if (g_pvae_d_xcol) cuMemFree(g_pvae_d_xcol);
        g_pvae_d_xcol = 0;
        if (cuMemAlloc(&g_pvae_d_xcol, xcol_bytes) != CUDA_SUCCESS) {
            g_pvae_xcol_max_bytes = 0; g_pvae_d_xcol = 0; return -1;
        }
        g_pvae_xcol_max_bytes = xcol_bytes;
    }
    if (yt_bytes > g_pvae_yt_max_bytes) {
        if (g_pvae_d_yt) cuMemFree(g_pvae_d_yt);
        g_pvae_d_yt = 0;
        if (cuMemAlloc(&g_pvae_d_yt, yt_bytes) != CUDA_SUCCESS) {
            g_pvae_yt_max_bytes = 0; g_pvae_d_yt = 0; return -1;
        }
        g_pvae_yt_max_bytes = yt_bytes;
    }
    return 0;
}

/* Release transient TC scratch (xcol/yt/xbf). Safe to call between encode and
 * decode phases — buffers are lazily reallocated via pvae_conv_scratch_ensure
 * on the next k_conv* dispatch. Use to give VRAM back before downstream stages
 * (e.g., UNet weights load) on memory-tight GPUs. */
static void pvae_free_scratch(void) {
    if (g_pvae_d_xcol) { cuMemFree(g_pvae_d_xcol); g_pvae_d_xcol = 0; }
    g_pvae_xcol_max_bytes = 0;
    if (g_pvae_d_yt)   { cuMemFree(g_pvae_d_yt);   g_pvae_d_yt   = 0; }
    g_pvae_yt_max_bytes = 0;
    if (g_pvae_d_xbf)  { cuMemFree(g_pvae_d_xbf);  g_pvae_d_xbf  = 0; }
    g_pvae_xbf_max_elem = 0;
}

static const pvae_fp8_entry *pvae_fp8_lookup(CUdeviceptr w_f32) {
    for (int i = 0; i < g_pvae_n_fp8_reg; i++)
        if (g_pvae_fp8_reg[i].w_f32 == w_f32) return &g_pvae_fp8_reg[i];
    return NULL;
}

static int pvae_quantize_w_fp8(const pvae_kernels *kk, CUdeviceptr d_w_f32,
                                int n_out, int n_in,
                                CUdeviceptr *out_fp8, CUdeviceptr *out_scale) {
    *out_fp8 = 0; *out_scale = 0;
    if (!kk->f_reduce_max_abs || !kk->f_quantize_fp8 || !d_w_f32 ||
        n_out <= 0 || n_in <= 0) return -1;
    size_t n_elem     = (size_t)n_out * (size_t)n_in;
    int    n_out_pad  = ((n_out + 255) / 256) * 256;
    size_t n_elem_pad = (size_t)n_out_pad * (size_t)n_in;
    CUdeviceptr d_max = 0, d_fp8 = 0, d_scale = 0;
    if (cuMemAlloc(&d_max,   sizeof(float)) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_fp8,   n_elem_pad)    != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_scale, sizeof(float)) != CUDA_SUCCESS) goto fail;
    cuMemsetD8Async(d_fp8, 0, n_elem_pad, 0);
    cuMemsetD8Async(d_max, 0, sizeof(float), 0);
    int n = (int)n_elem;
    {
        void *args[] = {&d_max, &d_w_f32, &n};
        cuLaunchKernel(kk->f_reduce_max_abs, (unsigned)((n + 255) / 256), 1, 1,
                        256, 1, 1, 0, 0, args, NULL);
    }
    {
        void *args[] = {&d_fp8, &d_scale, &d_w_f32, &d_max, &n};
        cuLaunchKernel(kk->f_quantize_fp8, (unsigned)((n + 255) / 256), 1, 1,
                        256, 1, 1, 0, 0, args, NULL);
    }
    cuCtxSynchronize();
    cuMemFree(d_max);
    *out_fp8 = d_fp8; *out_scale = d_scale;
    return 0;
fail:
    if (d_max)   cuMemFree(d_max);
    if (d_fp8)   cuMemFree(d_fp8);
    if (d_scale) cuMemFree(d_scale);
    return -1;
}

static void pvae_fp8_register(const pvae_kernels *kk, CUdeviceptr w_f32,
                                int n_out, int n_in) {
    if (!w_f32 || g_pvae_n_fp8_reg >= PVAE_FP8_REG_MAX) return;
    if (n_in <= 0 || n_out <= 0 || (n_in % 32) != 0) return;
    if (pvae_fp8_lookup(w_f32)) return;
    CUdeviceptr fp8 = 0, sc = 0;
    if (pvae_quantize_w_fp8(kk, w_f32, n_out, n_in, &fp8, &sc) != 0) return;
    pvae_fp8_entry *e = &g_pvae_fp8_reg[g_pvae_n_fp8_reg++];
    e->w_f32 = w_f32; e->w_fp8 = fp8; e->w_scale = sc; e->w_bf16 = 0;
    e->n_out = n_out; e->n_in = n_in;
    e->w_scale_host = 1.0f;
    cuCtxSynchronize();
    cuMemcpyDtoH(&e->w_scale_host, sc, sizeof(float));
    /* Note: we keep w_f32 alive (unlike UNet which frees it). VAE weights are
     * loaded by load_decoder/load_encoder via upload_st as raw F32; freeing
     * here would invalidate caller-held pointers in pvae_decoder/encoder. The
     * extra ~80MB is acceptable (encoder+decoder is small). */
}

static int pvae_try_fp8_v7(const pvae_kernels *kk,
                            CUdeviceptr y, CUdeviceptr x,
                            const pvae_fp8_entry *e, CUdeviceptr b,
                            int M, int K, int N) {
    if (!g_pvae_use_fp8_v7) return 0;
    if (!kk->f_gemm_fp8_v7_fused || !kk->f_quantize_fp8 || !kk->f_reduce_max_abs)
        return 0;
    if (!e || !e->w_fp8 || !e->w_scale) return 0;
    if ((K % 64) != 0 || M < 1) return 0;
    size_t n_elem = (size_t)M * (size_t)K;
    if (pvae_xfp8_ensure(n_elem) != 0) return 0;
    int n = (int)n_elem;
    if (!(g_pvae_v7_last_x == x && g_pvae_v7_last_n == n)) {
        cuMemsetD32Async(g_pvae_d_xmax, 0, 1, 0);
        {
            void *args[] = {&g_pvae_d_xmax, &x, &n};
            cuLaunchKernel(kk->f_reduce_max_abs,
                            (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1,
                            0, 0, args, NULL);
        }
        {
            void *args[] = {&g_pvae_d_xfp8, &g_pvae_d_xscale, &x,
                             &g_pvae_d_xmax, &n};
            cuLaunchKernel(kk->f_quantize_fp8,
                            (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1,
                            0, 0, args, NULL);
        }
        g_pvae_v7_last_x = x; g_pvae_v7_last_n = n;
    }
    CUdeviceptr w_fp8 = e->w_fp8;
    float w_scale_val = e->w_scale_host;
    int has_bias = (b != 0) ? 1 : 0;
    int accumulate = 0;
    int Mv = M, Nv = N, Kv = K;
    void *gargs[] = {&y, &g_pvae_d_xfp8, &w_fp8, &b,
                     &g_pvae_d_xscale, &w_scale_val, &has_bias,
                     &accumulate, &Mv, &Nv, &Kv};
    unsigned npx = (unsigned)((N + 127) / 128);
    unsigned npy = (unsigned)((M + 63) / 64);
    int use_p2 = (g_pvae_use_fp8_v7_p2 && kk->f_gemm_fp8_v7_fused_p2);
    unsigned pad = use_p2 ? 1u : 3u;
    unsigned gx  = (npx + pad) & ~pad;
    unsigned gy  = (npy + pad) & ~pad;
    unsigned smem_bytes = 2u * (64u * 32u + 128u * 32u) * 2u;
    CUfunction f = use_p2 ? kk->f_gemm_fp8_v7_fused_p2 : kk->f_gemm_fp8_v7_fused;
    cuLaunchKernel(f, gx, gy, 1, 256, 1, 1,
                    smem_bytes, 0, gargs, NULL);
    return 1;
}

/* Read env gates and decide which TC paths are live. Defaults ON when the
 * required kernels are present (mirrors UNet). Disable individually with
 *   PAINT_VAE_FP8_GEMM=0  PAINT_VAE_BF16_GEMM=0
 *   PAINT_VAE_FP8_V7=0    PAINT_VAE_FP8_CONV=0
 */
static void pvae_init_runtime(const pvae_kernels *kk) {
    int env_fp8   = 1, env_bf16 = 1, env_v7 = 1, env_conv = 1, env_mt4 = 1;
    int env_v7_p2 = 1;
    const char *e;
    if ((e = getenv("PAINT_VAE_FP8_GEMM")) && atoi(e) == 0) env_fp8 = 0;
    if ((e = getenv("PAINT_VAE_BF16_GEMM"))&& atoi(e) == 0) env_bf16 = 0;
    if ((e = getenv("PAINT_VAE_FP8_V7"))   && atoi(e) == 0) env_v7  = 0;
    if ((e = getenv("PAINT_VAE_FP8_V7_P2"))&& atoi(e) == 0) env_v7_p2 = 0;
    if ((e = getenv("PAINT_VAE_FP8_CONV")) && atoi(e) == 0) env_conv = 0;
    if ((e = getenv("PAINT_VAE_FP8_MT4"))  && atoi(e) == 0) env_mt4  = 0;
    g_pvae_use_fp8_gemm     = env_fp8  && (kk->f_gemm_fp8 || kk->f_gemm_fp8_mt4) &&
                              kk->f_reduce_max_abs && kk->f_quantize_fp8;
    g_pvae_use_fp8_gemm_mt4 = env_mt4 && g_pvae_use_fp8_gemm && kk->f_gemm_fp8_mt4;
    g_pvae_use_bf16_gemm    = env_bf16 && kk->f_gemm_bf16_v7 && kk->f_quant_bf16 &&
                              kk->f_add_bias_f32 && kk->f_reduce_max_abs &&
                              kk->f_quantize_fp8;
    g_pvae_use_fp8_v7       = env_v7  && kk->f_gemm_fp8_v7_fused &&
                              kk->f_reduce_max_abs && kk->f_quantize_fp8;
    g_pvae_use_fp8_v7_p2    = env_v7_p2 && g_pvae_use_fp8_v7 && kk->f_gemm_fp8_v7_fused_p2;
    g_pvae_use_fp8_conv     = env_conv && kk->f_im2col_3x3_p1 &&
                              kk->f_im2col_3x3_p1_s2 && kk->f_t_hwc_chw &&
                              kk->f_chw_nc &&
                              (g_pvae_use_fp8_gemm || g_pvae_use_bf16_gemm ||
                               g_pvae_use_fp8_v7);
    fprintf(stderr, "[pvae] TC dispatch: fp8=%d bf16=%d v7=%d v7_p2=%d conv=%d mt4=%d\n",
            g_pvae_use_fp8_gemm, g_pvae_use_bf16_gemm,
            g_pvae_use_fp8_v7, g_pvae_use_fp8_v7_p2,
            g_pvae_use_fp8_conv, g_pvae_use_fp8_gemm_mt4);
    (void)pvae_conv_scratch_ensure;
}

/* TC GEMM helper used by k_conv / k_conv_down: dispatches the GEMM half of
 *   y[hw, co] = xcol[hw, K] @ Wt[co, K] (+ bias)
 * into FP8 v7 / BF16 v7 / FP8-pipe-MT4 / FP8-pipe in priority order.
 * Returns 1 on dispatch (caller still does the t_hwc_chw transpose).
 * `xcol_in_pvae_d_xcol`: input (M=hw rows, K cols) lives in g_pvae_d_xcol.
 * `out_in_pvae_d_yt`: output (M rows, N=co cols) goes to g_pvae_d_yt. */
__attribute__((unused))
static int pvae_dispatch_conv_gemm(const pvae_kernels *kk, CUdeviceptr w_f32,
                                    CUdeviceptr b, int M, int K, int N) {
    if (!w_f32) return 0;
    const pvae_fp8_entry *e = pvae_fp8_lookup(w_f32);
    if (!e) { pvae_fp8_register(kk, w_f32, N, K); e = pvae_fp8_lookup(w_f32); }
    if (!e || e->n_out != N || e->n_in != K) return 0;
    /* FP8 v7 fused (folds bias + descale into writeback). */
    if (g_pvae_use_fp8_v7 &&
        pvae_try_fp8_v7(kk, g_pvae_d_yt, g_pvae_d_xcol, e, b, M, K, N)) {
        return 1;
    }
    /* BF16 v7: quant_bf16(xcol) -> gemm_bf16_v7 -> add_bias_inplace_f32. */
    if (g_pvae_use_bf16_gemm && e->w_bf16 == 0) {
        /* register didn't allocate BF16 mirror because UNet's path isn't
         * using cast_bf16; skip here -- pvae_fp8_register already populates
         * w_bf16 only when use_bf16 is set, but VAE has no f_cast_bf16
         * (the bf16-prelude doesn't add it). Fall through to FP8 pipe. */
    }
    if (g_pvae_use_bf16_gemm && e->w_bf16 && kk->f_gemm_bf16_v7 &&
        kk->f_quant_bf16 && kk->f_add_bias_f32) {
        size_t n_elem = (size_t)M * (size_t)K;
        if (pvae_xbf_ensure(n_elem) == 0) {
            int n = (int)n_elem;
            void *qargs[] = {&g_pvae_d_xbf, &g_pvae_d_xcol, &n};
            cuLaunchKernel(kk->f_quant_bf16,
                            (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1,
                            0, 0, qargs, NULL);
            int Mv = M, Nv = N, Kv = K;
            CUdeviceptr w_bf = e->w_bf16;
            unsigned npx = (unsigned)((Nv + 127) / 128);
            unsigned npy = (unsigned)((Mv + 63) / 64);
            unsigned gxv = (npx + 3u) & ~3u;
            unsigned gyv = (npy + 3u) & ~3u;
            unsigned smemv = 2u * (64u * 32u + 128u * 32u) * 2u;
            void *gargs[] = {&g_pvae_d_yt, &g_pvae_d_xbf, &w_bf, &Mv, &Nv, &Kv};
            cuLaunchKernel(kk->f_gemm_bf16_v7, gxv, gyv, 1, 256, 1, 1,
                            smemv, 0, gargs, NULL);
            int has_bias = (b != 0) ? 1 : 0;
            int rows = M, cols = N;
            void *bargs[] = {&g_pvae_d_yt, &b, &rows, &cols, &has_bias};
            unsigned bx = 256;
            unsigned gxd = (unsigned)((cols + bx - 1) / bx);
            unsigned gyd = (unsigned)rows;
            cuLaunchKernel(kk->f_add_bias_f32, gxd, gyd, 1, bx, 1, 1,
                            0, 0, bargs, NULL);
            return 1;
        }
    }
    /* FP8-pipe scaled (BF16 weight x F32 act, scaled). */
    if (g_pvae_use_fp8_gemm) {
        int n_out = N, n_in = K, n_tok = M;
        CUdeviceptr w_fp8 = e->w_fp8, w_scale = e->w_scale;
        void *gargs[] = { &g_pvae_d_yt, &w_fp8, &g_pvae_d_xcol, &b,
                          &n_out, &n_in, &n_tok, &w_scale };
        unsigned gx = (unsigned)((N + 255) / 256); gx = (gx + 3u) & ~3u;
        if (g_pvae_use_fp8_gemm_mt4 && kk->f_gemm_fp8_mt4 && M >= 64) {
            unsigned gy = (unsigned)((M + 63) / 64); gy = (gy + 3u) & ~3u;
            size_t smem = 4096 + 8192 * 2;
            cuLaunchKernel(kk->f_gemm_fp8_mt4, gx, gy, 1, 128, 1, 1,
                            smem, 0, gargs, NULL);
            return 1;
        }
        if (kk->f_gemm_fp8) {
            unsigned gy = (unsigned)((M + 31) / 32); gy = (gy + 3u) & ~3u;
            size_t smem = 2048 + 8192 * 2;
            cuLaunchKernel(kk->f_gemm_fp8, gx, gy, 1, 128, 1, 1,
                            smem, 0, gargs, NULL);
            return 1;
        }
    }
    return 0;
}

static void k_conv(const pvae_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                    CUdeviceptr w, CUdeviceptr b,
                    int ci, int h, int wd, int co, int kh, int kw, int pad) {
    /* Budget cap: skip TC if im2col scratch would exceed 512MB. Large convs
     * (e.g., decoder up_res ci=256 at 512x512 needs xcol=2.4GB) blow VRAM
     * and cause downstream cuMemAlloc OOM in UNet stage. */
    const size_t XCOL_BUDGET = (size_t)1024 * 1024 * 1024;
    if (g_pvae_use_fp8_conv && kh == 1 && kw == 1 && pad == 0 &&
        (ci % 32) == 0 && co >= 32 && w) {
        int hw = h * wd;
        size_t xcol_bytes = (size_t)hw * ci * sizeof(float);
        size_t yt_bytes   = (size_t)hw * co * sizeof(float);
        if (xcol_bytes <= XCOL_BUDGET &&
            pvae_conv_scratch_ensure(xcol_bytes, yt_bytes) == 0) {
            int C_arg = ci, N_arg = hw;
            void *cargs[] = { &g_pvae_d_xcol, &in, &C_arg, &N_arg };
            cuLaunchKernel(kk->f_chw_nc, (unsigned)((hw + 255) / 256),
                            (unsigned)ci, 1, 256, 1, 1, 0, 0, cargs, NULL);
            if (pvae_dispatch_conv_gemm(kk, w, b, hw, ci, co)) {
                int co_arg = co, hw_arg = hw;
                void *targs[] = { &out, &g_pvae_d_yt, &co_arg, &hw_arg };
                cuLaunchKernel(kk->f_t_hwc_chw,
                                (unsigned)((co + 31) / 32), (unsigned)((hw + 7) / 8),
                                1, 32, 8, 1, 0, 0, targs, NULL);
                return;
            }
        }
    }
    if (g_pvae_use_fp8_conv && kh == 3 && kw == 3 && pad == 1 &&
        (ci % 32) == 0 && co >= 32 && w) {
        int K = ci * 9;
        int hw = h * wd;
        size_t xcol_bytes = (size_t)hw * K * sizeof(float);
        size_t yt_bytes   = (size_t)hw * co * sizeof(float);
        if (xcol_bytes <= XCOL_BUDGET &&
            pvae_conv_scratch_ensure(xcol_bytes, yt_bytes) == 0) {
            int ci_arg = ci, h_arg = h, w_arg = wd;
            void *im_args[] = { &g_pvae_d_xcol, &in, &ci_arg, &h_arg, &w_arg };
            cuLaunchKernel(kk->f_im2col_3x3_p1,
                            (unsigned)((K + 31) / 32), (unsigned)((hw + 7) / 8),
                            1, 32, 8, 1, 0, 0, im_args, NULL);
            if (pvae_dispatch_conv_gemm(kk, w, b, hw, K, co)) {
                int co_arg = co, hw_arg = hw;
                void *targs[] = { &out, &g_pvae_d_yt, &co_arg, &hw_arg };
                cuLaunchKernel(kk->f_t_hwc_chw,
                                (unsigned)((co + 31) / 32), (unsigned)((hw + 7) / 8),
                                1, 32, 8, 1, 0, 0, targs, NULL);
                return;
            }
        }
        /* Tiled fallback: xcol exceeds budget, chunk along output rows.
         * Requires tiled im2col + tiled transpose kernels. */
        if (xcol_bytes > XCOL_BUDGET &&
            kk->f_im2col_3x3_p1_tiled && kk->f_t_hwc_chw_tiled) {
            size_t row_bytes = (size_t)K * sizeof(float) +
                               (size_t)co * sizeof(float);
            int chunk_rows = (int)(XCOL_BUDGET / row_bytes);
            if (chunk_rows > hw) chunk_rows = hw;
            if (chunk_rows >= 8) {
                size_t cx_bytes = (size_t)chunk_rows * K * sizeof(float);
                size_t cy_bytes = (size_t)chunk_rows * co * sizeof(float);
                if (pvae_conv_scratch_ensure(cx_bytes, cy_bytes) == 0) {
                    int dispatched_all = 1;
                    for (int row_off = 0; row_off < hw; row_off += chunk_rows) {
                        int rows = chunk_rows;
                        if (row_off + rows > hw) rows = hw - row_off;
                        int ci_arg = ci, h_arg = h, w_arg = wd;
                        int roff = row_off, nrows = rows;
                        void *im_args[] = { &g_pvae_d_xcol, &in, &ci_arg,
                                            &h_arg, &w_arg, &roff, &nrows };
                        cuLaunchKernel(kk->f_im2col_3x3_p1_tiled,
                                        (unsigned)((K + 31) / 32),
                                        (unsigned)((rows + 7) / 8), 1,
                                        32, 8, 1, 0, 0, im_args, NULL);
                        if (!pvae_dispatch_conv_gemm(kk, w, b, rows, K, co)) {
                            dispatched_all = 0; break;
                        }
                        int co_arg = co, hw_arg2 = h * wd;
                        (void)hw_arg2;
                        void *targs[] = { &out, &g_pvae_d_yt, &co_arg,
                                          &hw_arg2, &roff, &nrows };
                        cuLaunchKernel(kk->f_t_hwc_chw_tiled,
                                        (unsigned)((co + 31) / 32),
                                        (unsigned)((rows + 7) / 8), 1,
                                        32, 8, 1, 0, 0, targs, NULL);
                    }
                    if (dispatched_all) return;
                }
            }
        }
    }
    void *args[] = { &out, &in, &w, &b, &ci, &h, &wd, &co, &kh, &kw, &pad };
    int total = co * h * wd;
    unsigned grid = (unsigned)((total + 255) / 256);
    cuLaunchKernel(kk->f_conv, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
}

static void k_conv_down(const pvae_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                          CUdeviceptr w, CUdeviceptr b,
                          int ci, int h, int wd, int co) {
    if (g_pvae_use_fp8_conv && (ci % 32) == 0 && co >= 32 && w) {
        int K = ci * 9;
        int oh = h >> 1, ow = wd >> 1;
        int ohw = oh * ow;
        size_t xcol_bytes = (size_t)ohw * K * sizeof(float);
        size_t yt_bytes   = (size_t)ohw * co * sizeof(float);
        const size_t XCOL_BUDGET = (size_t)1024 * 1024 * 1024;
        if (xcol_bytes <= XCOL_BUDGET &&
            pvae_conv_scratch_ensure(xcol_bytes, yt_bytes) == 0) {
            int ci_arg = ci, h_arg = h, w_arg = wd;
            void *im_args[] = { &g_pvae_d_xcol, &in, &ci_arg, &h_arg, &w_arg };
            cuLaunchKernel(kk->f_im2col_3x3_p1_s2,
                            (unsigned)((K + 31) / 32), (unsigned)((ohw + 7) / 8),
                            1, 32, 8, 1, 0, 0, im_args, NULL);
            if (pvae_dispatch_conv_gemm(kk, w, b, ohw, K, co)) {
                int co_arg = co, ohw_arg = ohw;
                void *targs[] = { &out, &g_pvae_d_yt, &co_arg, &ohw_arg };
                cuLaunchKernel(kk->f_t_hwc_chw,
                                (unsigned)((co + 31) / 32), (unsigned)((ohw + 7) / 8),
                                1, 32, 8, 1, 0, 0, targs, NULL);
                return;
            }
        }
    }
    void *args[] = { &out, &in, &w, &b, &ci, &h, &wd, &co };
    int oh = h >> 1, ow = wd >> 1;
    int total = co * oh * ow;
    unsigned grid = (unsigned)((total + 255) / 256);
    cuLaunchKernel(kk->f_conv_down, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
}

static void k_up2x(const pvae_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                    int C, int H, int W) {
    void *args[] = { &out, &in, &C, &H, &W };
    int total = C * (H*2) * (W*2);
    unsigned grid = (unsigned)((total + 255) / 256);
    cuLaunchKernel(kk->f_up2x, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
}

static void k_add(const pvae_kernels *kk, CUdeviceptr out, CUdeviceptr a,
                   CUdeviceptr b, int n) {
    void *args[] = { &out, &a, &b, &n };
    unsigned grid = (unsigned)((n + 255) / 256);
    cuLaunchKernel(kk->f_add, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
}

static void k_chw_to_nc(const pvae_kernels *kk, CUdeviceptr out,
                          CUdeviceptr in, int C, int N) {
    void *args[] = { &out, &in, &C, &N };
    unsigned gx = (unsigned)((N + 255) / 256);
    cuLaunchKernel(kk->f_chw_nc, gx, (unsigned)C, 1, 256, 1, 1, 0, 0, args, NULL);
}

static void k_nc_to_chw(const pvae_kernels *kk, CUdeviceptr out,
                          CUdeviceptr in, int C, int N) {
    void *args[] = { &out, &in, &C, &N };
    unsigned gx = (unsigned)((N + 255) / 256);
    cuLaunchKernel(kk->f_nc_chw, gx, (unsigned)C, 1, 256, 1, 1, 0, 0, args, NULL);
}

static void k_attn(const pvae_kernels *kk, CUdeviceptr out, CUdeviceptr Q,
                    CUdeviceptr K, CUdeviceptr V, int N, int dim, float scale) {
    void *args[] = { &out, &Q, &K, &V, &N, &dim, &scale };
    /* WARPS=4 -> 4 query rows per CTA, 128 threads. smem = 2*BKV(8)*dim*4. */
    unsigned grid = (unsigned)((N + 4 - 1) / 4);
    size_t smem = 2 * 8 * dim * sizeof(float);
    cuLaunchKernel(kk->f_attn, grid, 1, 1, 128, 1, 1, smem, 0, args, NULL);
}

/* ===== ResBlock ============================================================
 * h = norm1(x) ; silu ; conv1(h)
 * h = norm2(h) ; silu ; conv2(h)
 * out = h + (skip(x) if c_in!=c_out else x)
 *
 * Buffers:
 *   d_x       in [c_in, H, W]
 *   d_out     out [c_out, H, W]
 *   d_t1, d_t2 — workspaces sized max(c_in,c_out) * H * W
 * Weights laid out per-resblock by caller. */
typedef struct {
    CUdeviceptr norm1_g, norm1_b;
    CUdeviceptr conv1_w, conv1_b;
    CUdeviceptr norm2_g, norm2_b;
    CUdeviceptr conv2_w, conv2_b;
    CUdeviceptr skip_w, skip_b;   /* may be 0 if c_in==c_out */
    int c_in, c_out;
} pvae_resblock;

static void run_resblock(const pvae_kernels *kk, const pvae_resblock *r,
                          CUdeviceptr d_x, CUdeviceptr d_out,
                          CUdeviceptr d_t1, CUdeviceptr d_t2,
                          int H, int W, int num_groups) {
    int sp = H * W;
    int n_out = r->c_out * sp;
    /* h = silu(norm1(x))  [c_in, H, W] -> d_t1 */
    k_groupnorm(kk, d_t1, d_x, r->norm1_g, r->norm1_b,
                 r->c_in, sp, num_groups, 1);
    /* h = conv1(h)  -> d_t2 [c_out, H, W] */
    k_conv(kk, d_t2, d_t1, r->conv1_w, r->conv1_b,
            r->c_in, H, W, r->c_out, 3, 3, 1);
    /* h = silu(norm2(h)) in-place -> d_t1 */
    k_groupnorm(kk, d_t1, d_t2, r->norm2_g, r->norm2_b,
                 r->c_out, sp, num_groups, 1);
    /* h = conv2(h) -> d_t2 [c_out, H, W] */
    k_conv(kk, d_t2, d_t1, r->conv2_w, r->conv2_b,
            r->c_out, H, W, r->c_out, 3, 3, 1);
    /* skip path */
    if (r->skip_w) {
        /* d_t1 = skip(x) [c_out, H, W] via 1x1 conv */
        k_conv(kk, d_t1, d_x, r->skip_w, r->skip_b,
                r->c_in, H, W, r->c_out, 1, 1, 0);
        k_add(kk, d_out, d_t2, d_t1, n_out);
    } else {
        k_add(kk, d_out, d_t2, d_x, n_out);
    }
}

/* ===== mid-block self-attention ============================================
 * h = norm(x)
 * q = Q_w @ h_chw + Q_b ; same for k,v   (1x1 convs)
 * h_chw -> [N, C] transpose
 * y_nc = attn(q,k,v) (single head, scale=1/sqrt(C))
 * y_chw = transpose back
 * y_chw = proj(y_chw) (1x1 conv)
 * out = x + y_chw */
typedef struct {
    CUdeviceptr norm_g, norm_b;
    CUdeviceptr q_w, q_b;
    CUdeviceptr k_w, k_b;
    CUdeviceptr v_w, v_b;
    CUdeviceptr p_w, p_b;
    int dim;
} pvae_attn_layer;

/* Buffer requirements:
 *   d_h     [C, H, W]   norm output / proj output scratch
 *   d_chw   [C, H, W]   reused for Q, K, V chw before each transpose
 *   d_qnc, d_knc, d_vnc, d_ync   [N, C] each
 * (d_x and d_out can be the same allocation as d_h or d_chw; caller decides). */
static void run_attn(const pvae_kernels *kk, const pvae_attn_layer *a,
                      CUdeviceptr d_x, CUdeviceptr d_out,
                      CUdeviceptr d_h, CUdeviceptr d_chw,
                      CUdeviceptr d_qnc, CUdeviceptr d_knc,
                      CUdeviceptr d_vnc, CUdeviceptr d_ync,
                      int H, int W, int num_groups) {
    int N = H * W, C = a->dim;
    int n = C * N;
    k_groupnorm(kk, d_h, d_x, a->norm_g, a->norm_b, C, N, num_groups, 0);
    /* Q -> chw -> nc */
    k_conv(kk, d_chw, d_h, a->q_w, a->q_b, C, H, W, C, 1, 1, 0);
    k_chw_to_nc(kk, d_qnc, d_chw, C, N);
    /* K -> chw -> nc */
    k_conv(kk, d_chw, d_h, a->k_w, a->k_b, C, H, W, C, 1, 1, 0);
    k_chw_to_nc(kk, d_knc, d_chw, C, N);
    /* V -> chw -> nc */
    k_conv(kk, d_chw, d_h, a->v_w, a->v_b, C, H, W, C, 1, 1, 0);
    k_chw_to_nc(kk, d_vnc, d_chw, C, N);
    /* attention */
    float scale = 1.0f / sqrtf((float)C);
    k_attn(kk, d_ync, d_qnc, d_knc, d_vnc, N, C, scale);
    /* NC -> CHW */
    k_nc_to_chw(kk, d_h, d_ync, C, N);
    k_conv(kk, d_chw, d_h, a->p_w, a->p_b, C, H, W, C, 1, 1, 0);
    k_add(kk, d_out, d_x, d_chw, n);
}

/* ===== Decoder weight container =========================================== */

typedef struct {
    CUdeviceptr pqc_w, pqc_b;
    CUdeviceptr conv_in_w, conv_in_b;
    pvae_resblock mid_r0, mid_r1;
    pvae_attn_layer mid_attn;
    /* up_blocks[0..3], each up to 3 resnets + optional upsampler */
    pvae_resblock up_res[4][3];
    CUdeviceptr   up_conv_w[4], up_conv_b[4];   /* upsampler conv (3x3); 0 if absent */
    int           up_has_sampler[4];
    CUdeviceptr conv_norm_out_g, conv_norm_out_b;
    CUdeviceptr conv_out_w, conv_out_b;
} pvae_decoder;

/* up_blocks channel topology: see file header */
static const int UP_CH_IN[4]  = { 512, 512, 512, 256 };
static const int UP_CH_OUT[4] = { 512, 512, 256, 128 };

static void load_resblock(st_context *st, pvae_resblock *r, const char *prefix,
                            int c_in, int c_out) {
    char buf[256];
    r->c_in = c_in; r->c_out = c_out;
    snprintf(buf, sizeof(buf), "%s.norm1.weight", prefix); r->norm1_g = upload_st(st, buf, NULL);
    snprintf(buf, sizeof(buf), "%s.norm1.bias",   prefix); r->norm1_b = upload_st(st, buf, NULL);
    snprintf(buf, sizeof(buf), "%s.conv1.weight", prefix); r->conv1_w = upload_st(st, buf, NULL);
    snprintf(buf, sizeof(buf), "%s.conv1.bias",   prefix); r->conv1_b = upload_st(st, buf, NULL);
    snprintf(buf, sizeof(buf), "%s.norm2.weight", prefix); r->norm2_g = upload_st(st, buf, NULL);
    snprintf(buf, sizeof(buf), "%s.norm2.bias",   prefix); r->norm2_b = upload_st(st, buf, NULL);
    snprintf(buf, sizeof(buf), "%s.conv2.weight", prefix); r->conv2_w = upload_st(st, buf, NULL);
    snprintf(buf, sizeof(buf), "%s.conv2.bias",   prefix); r->conv2_b = upload_st(st, buf, NULL);
    if (c_in != c_out) {
        snprintf(buf, sizeof(buf), "%s.conv_shortcut.weight", prefix);
        r->skip_w = upload_st(st, buf, NULL);
        snprintf(buf, sizeof(buf), "%s.conv_shortcut.bias", prefix);
        r->skip_b = upload_st(st, buf, NULL);
    } else { r->skip_w = 0; r->skip_b = 0; }
}

static void load_decoder(st_context *st, pvae_decoder *d) {
    d->pqc_w = upload_st(st, "post_quant_conv.weight", NULL);
    d->pqc_b = upload_st(st, "post_quant_conv.bias", NULL);
    d->conv_in_w = upload_st(st, "decoder.conv_in.weight", NULL);
    d->conv_in_b = upload_st(st, "decoder.conv_in.bias", NULL);

    load_resblock(st, &d->mid_r0, "decoder.mid_block.resnets.0", 512, 512);
    load_resblock(st, &d->mid_r1, "decoder.mid_block.resnets.1", 512, 512);

    d->mid_attn.dim = 512;
    d->mid_attn.norm_g = upload_st(st, "decoder.mid_block.attentions.0.group_norm.weight", NULL);
    d->mid_attn.norm_b = upload_st(st, "decoder.mid_block.attentions.0.group_norm.bias", NULL);
    d->mid_attn.q_w = upload_st(st, "decoder.mid_block.attentions.0.query.weight", NULL);
    d->mid_attn.q_b = upload_st(st, "decoder.mid_block.attentions.0.query.bias", NULL);
    d->mid_attn.k_w = upload_st(st, "decoder.mid_block.attentions.0.key.weight", NULL);
    d->mid_attn.k_b = upload_st(st, "decoder.mid_block.attentions.0.key.bias", NULL);
    d->mid_attn.v_w = upload_st(st, "decoder.mid_block.attentions.0.value.weight", NULL);
    d->mid_attn.v_b = upload_st(st, "decoder.mid_block.attentions.0.value.bias", NULL);
    d->mid_attn.p_w = upload_st(st, "decoder.mid_block.attentions.0.proj_attn.weight", NULL);
    d->mid_attn.p_b = upload_st(st, "decoder.mid_block.attentions.0.proj_attn.bias", NULL);

    for (int b = 0; b < 4; b++) {
        for (int i = 0; i < 3; i++) {
            char prefix[128];
            snprintf(prefix, sizeof(prefix), "decoder.up_blocks.%d.resnets.%d", b, i);
            int c_in  = (i == 0) ? UP_CH_IN[b] : UP_CH_OUT[b];
            int c_out = UP_CH_OUT[b];
            load_resblock(st, &d->up_res[b][i], prefix, c_in, c_out);
        }
        char buf[160];
        snprintf(buf, sizeof(buf), "decoder.up_blocks.%d.upsamplers.0.conv.weight", b);
        if (safetensors_find(st, buf) >= 0) {
            d->up_has_sampler[b] = 1;
            d->up_conv_w[b] = upload_st(st, buf, NULL);
            snprintf(buf, sizeof(buf), "decoder.up_blocks.%d.upsamplers.0.conv.bias", b);
            d->up_conv_b[b] = upload_st(st, buf, NULL);
        } else {
            d->up_has_sampler[b] = 0;
            d->up_conv_w[b] = 0; d->up_conv_b[b] = 0;
        }
    }
    d->conv_norm_out_g = upload_st(st, "decoder.conv_norm_out.weight", NULL);
    d->conv_norm_out_b = upload_st(st, "decoder.conv_norm_out.bias", NULL);
    d->conv_out_w = upload_st(st, "decoder.conv_out.weight", NULL);
    d->conv_out_b = upload_st(st, "decoder.conv_out.bias", NULL);
}

/* ===== Encoder weight container =========================================== */

typedef struct {
    CUdeviceptr conv_in_w, conv_in_b;
    pvae_resblock down_res[4][2];
    CUdeviceptr   down_conv_w[4], down_conv_b[4];
    int           down_has_sampler[4];
    pvae_resblock mid_r0, mid_r1;
    pvae_attn_layer mid_attn;
    CUdeviceptr conv_norm_out_g, conv_norm_out_b;
    CUdeviceptr conv_out_w, conv_out_b;
    CUdeviceptr qc_w, qc_b;   /* quant_conv 8->8, 1x1 */
} pvae_encoder;

static const int DOWN_CH_IN[4]  = { 128, 128, 256, 512 };
static const int DOWN_CH_OUT[4] = { 128, 256, 512, 512 };

static void load_encoder(st_context *st, pvae_encoder *e) {
    e->conv_in_w = upload_st(st, "encoder.conv_in.weight", NULL);
    e->conv_in_b = upload_st(st, "encoder.conv_in.bias", NULL);

    for (int b = 0; b < 4; b++) {
        for (int i = 0; i < 2; i++) {
            char prefix[128];
            snprintf(prefix, sizeof(prefix), "encoder.down_blocks.%d.resnets.%d", b, i);
            int c_in  = (i == 0) ? DOWN_CH_IN[b] : DOWN_CH_OUT[b];
            int c_out = DOWN_CH_OUT[b];
            load_resblock(st, &e->down_res[b][i], prefix, c_in, c_out);
        }
        char buf[160];
        snprintf(buf, sizeof(buf), "encoder.down_blocks.%d.downsamplers.0.conv.weight", b);
        if (safetensors_find(st, buf) >= 0) {
            e->down_has_sampler[b] = 1;
            e->down_conv_w[b] = upload_st(st, buf, NULL);
            snprintf(buf, sizeof(buf), "encoder.down_blocks.%d.downsamplers.0.conv.bias", b);
            e->down_conv_b[b] = upload_st(st, buf, NULL);
        } else {
            e->down_has_sampler[b] = 0;
            e->down_conv_w[b] = 0; e->down_conv_b[b] = 0;
        }
    }

    load_resblock(st, &e->mid_r0, "encoder.mid_block.resnets.0", 512, 512);
    load_resblock(st, &e->mid_r1, "encoder.mid_block.resnets.1", 512, 512);

    e->mid_attn.dim = 512;
    e->mid_attn.norm_g = upload_st(st, "encoder.mid_block.attentions.0.group_norm.weight", NULL);
    e->mid_attn.norm_b = upload_st(st, "encoder.mid_block.attentions.0.group_norm.bias", NULL);
    e->mid_attn.q_w = upload_st(st, "encoder.mid_block.attentions.0.query.weight", NULL);
    e->mid_attn.q_b = upload_st(st, "encoder.mid_block.attentions.0.query.bias", NULL);
    e->mid_attn.k_w = upload_st(st, "encoder.mid_block.attentions.0.key.weight", NULL);
    e->mid_attn.k_b = upload_st(st, "encoder.mid_block.attentions.0.key.bias", NULL);
    e->mid_attn.v_w = upload_st(st, "encoder.mid_block.attentions.0.value.weight", NULL);
    e->mid_attn.v_b = upload_st(st, "encoder.mid_block.attentions.0.value.bias", NULL);
    e->mid_attn.p_w = upload_st(st, "encoder.mid_block.attentions.0.proj_attn.weight", NULL);
    e->mid_attn.p_b = upload_st(st, "encoder.mid_block.attentions.0.proj_attn.bias", NULL);

    e->conv_norm_out_g = upload_st(st, "encoder.conv_norm_out.weight", NULL);
    e->conv_norm_out_b = upload_st(st, "encoder.conv_norm_out.bias", NULL);
    e->conv_out_w      = upload_st(st, "encoder.conv_out.weight", NULL);
    e->conv_out_b      = upload_st(st, "encoder.conv_out.bias", NULL);
    e->qc_w            = upload_st(st, "quant_conv.weight", NULL);
    e->qc_b            = upload_st(st, "quant_conv.bias", NULL);
}

/* ===== Decode pipeline ==================================================== */

static void decode(const pvae_kernels *kk, const pvae_decoder *D,
                    CUdeviceptr d_lat, int lat_h, int lat_w,
                    CUdeviceptr d_rgb,
                    CUdeviceptr d_a, CUdeviceptr d_b,
                    CUdeviceptr d_t1, CUdeviceptr d_t2,
                    CUdeviceptr d_qnc, CUdeviceptr d_knc,
                    CUdeviceptr d_vnc, CUdeviceptr d_ync) {
    int H = lat_h, W = lat_w;
    int NG = 32;
    /* post_quant_conv [4->4, 1x1] */
    k_conv(kk, d_a, d_lat, D->pqc_w, D->pqc_b, 4, H, W, 4, 1, 1, 0);
    /* conv_in [4->512, 3x3, pad=1] */
    k_conv(kk, d_b, d_a, D->conv_in_w, D->conv_in_b, 4, H, W, 512, 3, 3, 1);
    /* d_b holds [512,H,W] after conv_in. Workspace alternates a<->b. */

    /* mid: ResBlock(512) → Attn → ResBlock(512).
     * Stage in/out buffers:
     *   d_b -- conv_in output (input to mid_r0)
     *   d_a := mid_r0(d_b)   using d_t1/d_t2 scratch
     *   d_b := attn(d_a)     using d_t1 (norm-out=h scratch),
     *                              d_t2 (Q chw, also reused as proj-output scratch),
     *                              d_qnc (K chw), d_vnc (V chw),
     *                              d_qnc/knc/vnc/ync (NC layout)
     *   d_a := mid_r1(d_b)   using d_t1/d_t2 scratch */
    run_resblock(kk, &D->mid_r0, d_b, d_a, d_t1, d_t2, H, W, NG);
    run_attn(kk, &D->mid_attn,
              /*in*/ d_a, /*out*/ d_b,
              /*h*/  d_t1, /*chw*/ d_t2,
              d_qnc, d_knc, d_vnc, d_ync,
              H, W, NG);
    run_resblock(kk, &D->mid_r1, d_b, d_a, d_t1, d_t2, H, W, NG);

    /* up_blocks */
    /* d_a holds [512,H,W] going into up_blocks[0] */
    int cur_C = 512;
    int cur_H = H, cur_W = W;
    /* Ping-pong with d_a/d_b for resblock outputs. */
    CUdeviceptr d_in = d_a, d_outbuf = d_b;
    for (int blk = 0; blk < 4; blk++) {
        for (int i = 0; i < 3; i++) {
            run_resblock(kk, &D->up_res[blk][i], d_in, d_outbuf,
                          d_t1, d_t2, cur_H, cur_W, NG);
            cur_C = UP_CH_OUT[blk];
            CUdeviceptr tmp = d_in; d_in = d_outbuf; d_outbuf = tmp;
        }
        if (D->up_has_sampler[blk]) {
            /* nearest 2x upsample then 3x3 conv same-channels */
            k_up2x(kk, d_outbuf, d_in, cur_C, cur_H, cur_W);
            cur_H *= 2; cur_W *= 2;
            k_conv(kk, d_in, d_outbuf, D->up_conv_w[blk], D->up_conv_b[blk],
                    cur_C, cur_H, cur_W, cur_C, 3, 3, 1);
            /* d_in still has the upsample-conv output. */
        }
    }

    /* final norm + silu + conv_out */
    k_groupnorm(kk, d_outbuf, d_in, D->conv_norm_out_g, D->conv_norm_out_b,
                 cur_C, cur_H * cur_W, NG, 1);
    k_conv(kk, d_rgb, d_outbuf, D->conv_out_w, D->conv_out_b,
            cur_C, cur_H, cur_W, 3, 3, 3, 1);
}

/* ===== Encode pipeline ====================================================
 * Input  d_img : [3, H, W] in [-1, 1]
 * Output d_lat : [4, H/8, W/8] (mean of posterior)
 *
 * Workspace buffers d_a/d_b/d_t1/d_t2 each sized for the worst-case stage. */
static void encode(const pvae_kernels *kk, const pvae_encoder *E,
                    CUdeviceptr d_img, int H, int W,
                    CUdeviceptr d_lat,
                    CUdeviceptr d_a, CUdeviceptr d_b,
                    CUdeviceptr d_t1, CUdeviceptr d_t2,
                    CUdeviceptr d_qnc, CUdeviceptr d_knc,
                    CUdeviceptr d_vnc, CUdeviceptr d_ync) {
    int NG = 32;
    int cur_H = H, cur_W = W;
    /* conv_in [3 -> 128, 3x3] */
    k_conv(kk, d_a, d_img, E->conv_in_w, E->conv_in_b,
            3, cur_H, cur_W, 128, 3, 3, 1);
    int cur_C = 128;
    /* down_blocks: ping-pong d_a (input) -> d_b (output). After loop d_a holds last output. */
    CUdeviceptr d_in = d_a, d_outbuf = d_b;
    for (int blk = 0; blk < 4; blk++) {
        for (int i = 0; i < 2; i++) {
            run_resblock(kk, &E->down_res[blk][i], d_in, d_outbuf,
                          d_t1, d_t2, cur_H, cur_W, NG);
            cur_C = DOWN_CH_OUT[blk];
            CUdeviceptr tmp = d_in; d_in = d_outbuf; d_outbuf = tmp;
        }
        if (E->down_has_sampler[blk]) {
            /* asymmetric pad+stride2 3x3 conv: cur_C ch, H/2, W/2, same out_C. */
            k_conv_down(kk, d_outbuf, d_in, E->down_conv_w[blk],
                         E->down_conv_b[blk], cur_C, cur_H, cur_W, cur_C);
            cur_H >>= 1; cur_W >>= 1;
            CUdeviceptr tmp = d_in; d_in = d_outbuf; d_outbuf = tmp;
        }
    }

    /* mid: ResBlock(512) → Attn → ResBlock(512). d_in holds [512, H/8, W/8]. */
    run_resblock(kk, &E->mid_r0, d_in, d_outbuf, d_t1, d_t2, cur_H, cur_W, NG);
    run_attn(kk, &E->mid_attn,
              /*in*/ d_outbuf, /*out*/ d_in,
              /*h*/  d_t1, /*chw*/ d_t2,
              d_qnc, d_knc, d_vnc, d_ync,
              cur_H, cur_W, NG);
    run_resblock(kk, &E->mid_r1, d_in, d_outbuf, d_t1, d_t2, cur_H, cur_W, NG);
    /* d_outbuf holds [512, H/8, W/8]. */

    /* conv_norm_out + silu, conv_out [512->8, 3x3], quant_conv [8->8, 1x1] */
    k_groupnorm(kk, d_in, d_outbuf, E->conv_norm_out_g, E->conv_norm_out_b,
                 512, cur_H * cur_W, NG, 1);
    k_conv(kk, d_outbuf, d_in, E->conv_out_w, E->conv_out_b,
            512, cur_H, cur_W, 8, 3, 3, 1);
    k_conv(kk, d_in, d_outbuf, E->qc_w, E->qc_b,
            8, cur_H, cur_W, 8, 1, 1, 0);
    /* mean = first 4 channels. Copy [4, H/8, W/8] block out of d_in. */
    cuMemcpyDtoD(d_lat, d_in, 4 * (size_t)cur_H * cur_W * sizeof(float));
}

#endif /* CUDA_PAINT_VAE_RUNNER_H_ */
