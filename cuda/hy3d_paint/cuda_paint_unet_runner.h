/*
 * cuda_paint_unet_runner.h - Header-only CUDA port of the Hunyuan3D-2.1
 * paint UNet (UNet2p5DConditionModel: SD-2.1 UNet wrapped with the four
 * Basic2p5DTransformerBlock attention paths DINO/MA/MDA/RA + dual-stream).
 *
 * Hosts the kernel-launch wrappers, weight loaders, runner data structures,
 * and per-block helpers that were previously inline in test_paint_unet.c.
 * Phase 4.8 extraction so that downstream pipeline code (UniPC scheduler
 * loop, end-to-end texgen glue) can call into the same runner without
 * duplicating the per-block forward.
 *
 * The contents are file-local (`static`) so each translation unit that
 * includes this header gets its own copy of the helpers — fine while only
 * one .c uses it; revisit when a second consumer arrives.
 */

#ifndef CUDA_PAINT_UNET_RUNNER_H_
#define CUDA_PAINT_UNET_RUNNER_H_

#include "../cuew.h"
/* cuda_runner_common helpers are file-local `static`, so every including TU
 * needs the impls (no link conflict). Define unconditionally. */
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "cuda_paint_unet_kernels.h"

#ifdef CUDA_PAINT_UNET_RUNNER_IMPLEMENTATION
#define SAFETENSORS_IMPLEMENTATION
#endif
#include "safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ===== .npy I/O (float32 + int64) ========================================= */

static void *read_npy(const char *path, int *out_ndim, uint64_t *out_shape,
                       size_t *out_n, char *out_dtype) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return NULL; }
    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6)) {
        fprintf(stderr, "ERROR: not a .npy file: %s\n", path); fclose(f); return NULL;
    }
    uint8_t ver[2]; if (fread(ver, 1, 2, f) != 2) { fclose(f); return NULL; }
    uint16_t hlen; if (fread(&hlen, 2, 1, f) != 1) { fclose(f); return NULL; }
    char hdr[1024];
    if (hlen >= sizeof(hdr)) { fclose(f); return NULL; }
    if (fread(hdr, 1, hlen, f) != hlen) { fclose(f); return NULL; }
    hdr[hlen] = 0;
    /* dtype */
    const char *dt;
    int elt;
    if ((dt = strstr(hdr, "'descr': '<f4'"))) { strcpy(out_dtype, "f4"); elt = 4; }
    else if ((dt = strstr(hdr, "'descr': '<i8'"))) { strcpy(out_dtype, "i8"); elt = 8; }
    else { fprintf(stderr, "ERROR: unsupported dtype in %s\n", path); fclose(f); return NULL; }
    /* shape */
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
    void *buf = malloc(total * (size_t)elt);
    if (fread(buf, (size_t)elt, total, f) != total) {
        fprintf(stderr, "ERROR: short read on %s\n", path);
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    return buf;
}

/* ===== Weight upload ====================================================== */

static CUdeviceptr upload_st(const st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        fprintf(stderr, "ERROR: tensor not found: %s\n", name);
        return 0;
    }
    if (strcmp(safetensors_dtype(st, idx), "F32")) {
        fprintf(stderr, "ERROR: %s dtype %s, expected F32\n", name,
                safetensors_dtype(st, idx));
        return 0;
    }
    size_t bytes = safetensors_nbytes(st, idx);
    CUdeviceptr d;
    cuMemAlloc(&d, bytes);
    cuMemcpyHtoD(d, safetensors_data(st, idx), bytes);
    return d;
}

/* ===== Diff helper ======================================================== */

static int diff_against(const float *cu, const char *ref_path, size_t expect_n,
                          float warn_mae) {
    int nd; uint64_t shape[8]; size_t n; char dt[8];
    float *ref = (float *)read_npy(ref_path, &nd, shape, &n, dt);
    if (!ref) return -1;
    if (n != expect_n) {
        fprintf(stderr, "ERROR: ref %s has %zu elements, expected %zu\n",
                ref_path, n, expect_n);
        free(ref); return -1;
    }
    double sae = 0, smax = 0, sum_r = 0, sum_c = 0, sum_rc = 0, sum_rr = 0, sum_cc = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)cu[i] - (double)ref[i];
        if (d < 0) d = -d;
        sae += d; if (d > smax) smax = d;
        double r = ref[i], c = cu[i];
        sum_r += r; sum_c += c; sum_rc += r*c; sum_rr += r*r; sum_cc += c*c;
    }
    double mae = sae / n;
    double mr = sum_r/n, mc = sum_c/n;
    double cov = sum_rc/n - mr*mc;
    double vr  = sum_rr/n - mr*mr;
    double vc  = sum_cc/n - mc*mc;
    double corr = cov / sqrt(vr * vc + 1e-30);
    int ok = mae <= warn_mae;
    fprintf(stderr, "  vs %s : mae=%.4e max=%.4e corr=%.6f  %s\n",
            ref_path, mae, smax, corr, ok ? "OK" : "WARN");
    free(ref);
    return ok ? 0 : 1;
}

/* ===== Kernels ============================================================ */

typedef struct {
    CUmodule mod;
    CUfunction f_tse;     /* unet_timestep_embed_f32 */
    CUfunction f_lin;     /* unet_linear_f32 */
    CUfunction f_silu;    /* unet_silu_f32 */
    CUfunction f_conv;    /* unet_conv2d_f32 */
    CUfunction f_gn;      /* unet_groupnorm_f32 */
    CUfunction f_addc;    /* unet_add_chan_f32 */
    CUfunction f_add;     /* unet_add_f32 */
    CUfunction f_ln;      /* unet_layernorm_f32 */
    CUfunction f_chw_nc;  /* unet_chw_to_nc_f32 */
    CUfunction f_nc_chw;  /* unet_nc_to_chw_f32 */
    CUfunction f_mha;     /* unet_mha_f32 */
    CUfunction f_geglu;   /* unet_geglu_f32 */
    CUfunction f_conv_s2; /* unet_conv2d_stride2_f32 */
    CUfunction f_up2x;    /* unet_upsample2x_f32 */
    CUfunction f_concat;  /* unet_concat_chan_f32 */
    CUfunction f_rope;    /* unet_rope_apply_f32 */
    CUfunction f_ra_split_v; /* unet_ra_split_v_f32 */
    /* BF16 TC dispatch (Phase 4.9): NULL = unavailable, fall back to f_mha. */
    CUfunction f_cast_bf16;        /* cast_f32_to_bf16 */
    CUfunction f_attn_bf16_hd64;   /* flash_attn_bf16_hd64 */
    CUfunction f_attn_bf16_hd64_xq;/* flash_attn_bf16_hd64_xq */
    /* FP8 GEMM dispatch (Phase 4.9.2). */
    CUfunction f_gemm_fp8_mt4;     /* gemm_bf16_pipe_mt4_scaled_f32 */
    CUfunction f_gemm_fp8;         /* gemm_bf16_pipe_scaled_f32 (mt1) */
    CUfunction f_reduce_max_abs;   /* reduce_max_abs_f32 */
    CUfunction f_quantize_fp8;     /* quantize_to_fp8_e4m3 */
    /* Conv2d FP8 path (Phase 4.9.3): im2col + GEMM + transpose. */
    CUfunction f_im2col_3x3_p1;    /* unet_im2col_3x3_p1_f32 */
    CUfunction f_im2col_3x3_p1_s2; /* unet_im2col_3x3_p1_s2_f32 */
    CUfunction f_t_hwc_chw;        /* unet_t_hwc_to_chw_f32 */
    /* Pure BF16 path (Phase 4.9.4): F32->BF16 quant + v7 BF16 GEMM + bias. */
    CUfunction f_quant_bf16;       /* quant_bf16 */
    CUfunction f_add_bias_f32;     /* add_bias_inplace_f32 */
    CUfunction f_gemm_bf16_v7;     /* gemm_bf16_v7 */
} pu_kernels;

/* BF16-attn dispatch state. paint_stage_unet_create() flips on use_bf16_attn
 * after looking up the kernels and reading PAINT_BF16_ATTN. The buffers are
 * shared scratch sized for the largest Q/K/V tensor seen across all blocks. */
static int g_paint_use_bf16_attn = 0;
static CUdeviceptr g_paint_d_qbf = 0; /* [max_elem] u16 */
static CUdeviceptr g_paint_d_kbf = 0;
static CUdeviceptr g_paint_d_vbf = 0;
static size_t g_paint_qkv_bf16_max_elem = 0;

/* FP8 GEMM state (Phase 4.9.2).
 *   g_paint_use_fp8_gemm: 1 if all kernels resolved, env didn't disable, and
 *                         prequant succeeded for at least one weight.
 *   g_paint_d_xbf16     : BF16 scratch for activation X cast (one per call).
 *   g_paint_xbf16_max   : capacity in elements (uint16).
 *   g_paint_fp8_reg     : [F32-W ptr] -> (FP8, scale, n_out, n_in) mapping.
 * The registry is populated by paint_unet_prequant_weights(); k_linear()
 * does a linear scan on the W ptr and dispatches the FP8 path when a hit
 * matches the call's (N,K) and the kernel constraints allow it
 * (n_in % 32 == 0, n_out % 256 == 0, n_tok >= 16). */
typedef struct {
    CUdeviceptr w_f32;     /* key: original F32 weight ptr */
    CUdeviceptr w_fp8;     /* uint8 [n_out * n_in] */
    CUdeviceptr w_scale;   /* float [1] (per-tensor) */
    CUdeviceptr w_bf16;    /* uint16 [n_out * n_in] (Phase 4.9.4 BF16 path) */
    int n_out, n_in;
} pu_fp8_entry;

#define PAINT_FP8_REG_MAX 4096
static int g_paint_use_fp8_gemm = 0;
static int g_paint_use_fp8_gemm_mt4 = 0;
static CUdeviceptr g_paint_d_xbf16 = 0;
static size_t g_paint_xbf16_max_elem = 0;
static pu_fp8_entry g_paint_fp8_reg[PAINT_FP8_REG_MAX];
static int g_paint_n_fp8_reg = 0;
static size_t g_paint_fp8_total_bytes = 0;
/* Conv FP8 path scratch (lazy-grown). */
static int g_paint_use_fp8_conv = 0;
static CUdeviceptr g_paint_d_xcol = 0; /* [hw * ci*9] f32 */
static size_t g_paint_xcol_max_bytes = 0;
static CUdeviceptr g_paint_d_yt = 0;   /* [hw * co] f32 */
static size_t g_paint_yt_max_bytes = 0;
/* Pure BF16 path scratch (lazy-grown). */
static int g_paint_use_bf16_gemm = 0;
static CUdeviceptr g_paint_d_xbf = 0;  /* uint16 [n_tok * n_in] */
static size_t g_paint_xbf_max_elem = 0;

static int paint_xbf_ensure(size_t n_elem) {
    size_t need = n_elem * sizeof(unsigned short);
    size_t cur  = g_paint_xbf_max_elem * sizeof(unsigned short);
    if (need <= cur) return 0;
    if (g_paint_d_xbf) { cuMemFree(g_paint_d_xbf); g_paint_d_xbf = 0; }
    if (cuMemAlloc(&g_paint_d_xbf, need) != CUDA_SUCCESS) {
        g_paint_d_xbf = 0; g_paint_xbf_max_elem = 0; return -1;
    }
    g_paint_xbf_max_elem = n_elem;
    return 0;
}

static const pu_fp8_entry *paint_fp8_lookup(CUdeviceptr w_f32) {
    /* Hot path: linear scan, ~1500 entries max. Tested fast enough vs
     * the GEMM cost it gates. */
    for (int i = 0; i < g_paint_n_fp8_reg; i++) {
        if (g_paint_fp8_reg[i].w_f32 == w_f32) return &g_paint_fp8_reg[i];
    }
    return NULL;
}

/* F32 weight -> FP8 + per-tensor scale. Mirrors hy3d_quantize_w_fp8 but
 * skips the f16->f32 step (paint UNet weights are loaded as F32).
 * Pads the FP8 row count up to a 256-multiple so the BF16-pipe MT1 kernel
 * can be dispatched for any n_out (its W tile loader reads 256 contiguous
 * rows per CTA without bounds-checking; writes are already guarded by
 * `yc < n_out`). Padded rows are zero, contributing nothing to the dot
 * product on out-of-range output channels (which are also write-guarded). */
static int paint_quantize_w_fp8(const pu_kernels *kk, CUstream stream,
                                  CUdeviceptr d_w_f32, int n_out, int n_in,
                                  CUdeviceptr *out_fp8, CUdeviceptr *out_scale) {
    *out_fp8 = 0; *out_scale = 0;
    if (!kk->f_reduce_max_abs || !kk->f_quantize_fp8 || !d_w_f32 ||
        n_out <= 0 || n_in <= 0)
        return -1;
    size_t n_elem     = (size_t)n_out * (size_t)n_in;
    int    n_out_pad  = ((n_out + 255) / 256) * 256;
    size_t n_elem_pad = (size_t)n_out_pad * (size_t)n_in;
    CUdeviceptr d_max = 0, d_fp8 = 0, d_scale = 0;
    if (cuMemAlloc(&d_max, sizeof(float)) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_fp8, n_elem_pad)    != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_scale, sizeof(float)) != CUDA_SUCCESS) goto fail;
    cuMemsetD8Async(d_fp8, 0, n_elem_pad, stream);
    int n = (int)n_elem;
    cuMemsetD8Async(d_max, 0, sizeof(float), stream);
    {
        unsigned grid = (unsigned)((n + 255) / 256);
        void *args[] = {&d_max, &d_w_f32, &n};
        cuLaunchKernel(kk->f_reduce_max_abs, grid, 1, 1, 256, 1, 1, 0,
                        stream, args, NULL);
    }
    {
        unsigned grid = (unsigned)((n + 255) / 256);
        void *args[] = {&d_fp8, &d_scale, &d_w_f32, &d_max, &n};
        cuLaunchKernel(kk->f_quantize_fp8, grid, 1, 1, 256, 1, 1, 0,
                        stream, args, NULL);
    }
    cuStreamSynchronize(stream);
    cuMemFree(d_max);
    *out_fp8 = d_fp8; *out_scale = d_scale;
    return 0;
fail:
    if (d_max)   cuMemFree(d_max);
    if (d_fp8)   cuMemFree(d_fp8);
    if (d_scale) cuMemFree(d_scale);
    return -1;
}

static void paint_fp8_register(const pu_kernels *kk, CUstream stream,
                                 CUdeviceptr w_f32, int n_out, int n_in) {
    if (!w_f32) return;
    if (g_paint_n_fp8_reg >= PAINT_FP8_REG_MAX) return;
    /* The BF16-pipe MT1 kernel handles any n_out via FP8-row padding
     * (see paint_quantize_w_fp8); it still requires n_in % 32 == 0. */
    if (n_in <= 0 || n_out <= 0 || (n_in % 32) != 0) return;
    if (paint_fp8_lookup(w_f32)) return;
    CUdeviceptr fp8 = 0, sc = 0;
    size_t n_elem = (size_t)n_out * (size_t)n_in;
    if (paint_quantize_w_fp8(kk, stream, w_f32, n_out, n_in, &fp8, &sc) != 0) return;
    pu_fp8_entry *e = &g_paint_fp8_reg[g_paint_n_fp8_reg++];
    e->w_f32 = w_f32; e->w_fp8 = fp8; e->w_scale = sc; e->w_bf16 = 0;
    e->n_out = n_out; e->n_in = n_in;
    g_paint_fp8_total_bytes += n_elem;
    /* Optional BF16 mirror (Phase 4.9.4). Allocated only when the BF16 path
     * is enabled at create time; cast_f32_to_bf16 must be available. */
    if (g_paint_use_bf16_gemm && kk->f_cast_bf16) {
        size_t bytes = n_elem * sizeof(unsigned short);
        CUdeviceptr w_bf = 0;
        if (cuMemAlloc(&w_bf, bytes) == CUDA_SUCCESS) {
            int n = (int)n_elem;
            void *args[] = {&w_f32, &w_bf, &n};
            cuLaunchKernel(kk->f_cast_bf16, (unsigned)((n + 255) / 256),
                            1, 1, 256, 1, 1, 0, stream, args, NULL);
            cuStreamSynchronize(stream);
            e->w_bf16 = w_bf;
        }
    }
    /* Reclaim the F32 weight: the FP8 kernel handles any M (small-tok
     * tiles short-circuit on the `tok_base >= n_tok` guard), so dispatch
     * is unconditional once registered + shape-matches. */
    cuMemFree(w_f32);
}

/* ===== Per-launch debug wrapper (PAINT_FP8_DEBUG=1) =====================
 * Wraps every cuLaunchKernel below with a post-launch cuCtxSynchronize and
 * prints the first launch that flips status to non-success. Use to bisect
 * the FP8 OOB at idx ~535. Default off; zero overhead when off. */
static unsigned long g_paint_launch_idx = 0;
static int g_paint_fp8_debug = 0;
static int g_paint_profile = 0;
static const char *g_paint_last_kernel = "(none)";

#define PAINT_PROF_MAX 64
static const char *g_paint_prof_name[PAINT_PROF_MAX];
static double      g_paint_prof_ms  [PAINT_PROF_MAX];
static unsigned long g_paint_prof_n [PAINT_PROF_MAX];
static int g_paint_prof_count = 0;

static int paint_prof_slot(const char *name) {
    for (int i = 0; i < g_paint_prof_count; i++)
        if (g_paint_prof_name[i] == name) return i;
    if (g_paint_prof_count >= PAINT_PROF_MAX) return -1;
    int s = g_paint_prof_count++;
    g_paint_prof_name[s] = name;
    g_paint_prof_ms[s]   = 0.0;
    g_paint_prof_n[s]    = 0;
    return s;
}

static void paint_prof_dump(void) {
    if (g_paint_prof_count == 0) return;
    /* selection-sort by ms desc, top all */
    int idx[PAINT_PROF_MAX];
    for (int i = 0; i < g_paint_prof_count; i++) idx[i] = i;
    for (int i = 0; i < g_paint_prof_count; i++)
        for (int j = i+1; j < g_paint_prof_count; j++)
            if (g_paint_prof_ms[idx[j]] > g_paint_prof_ms[idx[i]]) {
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
    double total = 0;
    for (int i = 0; i < g_paint_prof_count; i++) total += g_paint_prof_ms[i];
    fprintf(stderr, "\n[paint_prof] ===== per-kernel time (PAINT_PROFILE=1) =====\n");
    fprintf(stderr, "[paint_prof] %-40s %10s %10s %10s\n",
            "kernel", "total_ms", "calls", "avg_us");
    for (int i = 0; i < g_paint_prof_count; i++) {
        int k = idx[i];
        fprintf(stderr, "[paint_prof] %-40s %10.2f %10lu %10.2f\n",
                g_paint_prof_name[k], g_paint_prof_ms[k], g_paint_prof_n[k],
                g_paint_prof_n[k] ? (g_paint_prof_ms[k] * 1000.0 / g_paint_prof_n[k]) : 0.0);
    }
    fprintf(stderr, "[paint_prof] %-40s %10.2f %10lu\n", "TOTAL", total, g_paint_launch_idx);
    fflush(stderr);
}

static inline CUresult paint_dbg_launch(const char *name, CUfunction f,
        unsigned gx, unsigned gy, unsigned gz,
        unsigned bx, unsigned by, unsigned bz,
        unsigned smem, CUstream stream, void **args, void **extra) {
    g_paint_launch_idx++;
    unsigned long idx = g_paint_launch_idx;
    if (g_paint_profile) {
        /* sync before to flush prior async work, then time the launch+sync */
        cuCtxSynchronize();
        CUevent e0, e1;
        cuEventCreate(&e0, 0); cuEventCreate(&e1, 0);
        cuEventRecord(e0, stream);
        CUresult r = cuLaunchKernel(f, gx, gy, gz, bx, by, bz, smem, stream, args, extra);
        cuEventRecord(e1, stream);
        cuEventSynchronize(e1);
        float ms = 0.f;
        cuEventElapsedTime(&ms, e0, e1);
        cuEventDestroy(e0); cuEventDestroy(e1);
        int s = paint_prof_slot(name);
        if (s >= 0) { g_paint_prof_ms[s] += ms; g_paint_prof_n[s]++; }
        return r;
    }
    CUresult r = cuLaunchKernel(f, gx, gy, gz, bx, by, bz, smem, stream, args, extra);
    if (g_paint_fp8_debug) {
        if (r != CUDA_SUCCESS) {
            const char *es = "?"; cuGetErrorString(r, &es);
            fprintf(stderr, "[paint_dbg] LAUNCH-FAIL idx=%lu name=%s err=%d (%s) prev=%s\n",
                    idx, name, r, es, g_paint_last_kernel);
            fflush(stderr); abort();
        }
        CUresult sr = cuCtxSynchronize();
        if (sr != CUDA_SUCCESS) {
            const char *es = "?"; cuGetErrorString(sr, &es);
            fprintf(stderr, "[paint_dbg] SYNC-FAIL idx=%lu name=%s err=%d (%s) prev=%s\n",
                    idx, name, sr, es, g_paint_last_kernel);
            fflush(stderr); abort();
        }
        g_paint_last_kernel = name;
    }
    return r;
}

#define cuLaunchKernel(f, gx, gy, gz, bx, by, bz, smem, stream, args, extra) \
    paint_dbg_launch(#f, (f), (gx), (gy), (gz), (bx), (by), (bz), (smem), (stream), (args), (extra))

static void k_timestep_embed(const pu_kernels *kk, CUdeviceptr out,
                              CUdeviceptr ts, int B, int dim) {
    void *args[] = { &out, &ts, &B, &dim };
    int tx = 64;
    cuLaunchKernel(kk->f_tse, (unsigned)B, (unsigned)((dim + tx - 1) / tx), 1,
                    tx, 1, 1, 0, 0, args, NULL);
}

static void k_linear(const pu_kernels *kk, CUdeviceptr y, CUdeviceptr x,
                      CUdeviceptr W, CUdeviceptr b, int M, int K, int N) {
    /* Pure BF16 path (Phase 4.9.4, PAINT_BF16_GEMM=1): F32 X -> BF16 quant,
     * gemm_bf16_v7 (BF16xBF16 m16n8k16 mma, F32 accum), bias post-pass.
     * Uses the same FP8 weight registry to also store a BF16 mirror; falls
     * through to the FP8 path when the BF16 mirror isn't available. */
    if (g_paint_use_bf16_gemm && kk->f_gemm_bf16_v7 && kk->f_quant_bf16 &&
        kk->f_add_bias_f32 && W && M >= 1 && (K % 32) == 0) {
        const pu_fp8_entry *e = paint_fp8_lookup(W);
        if (!e) {
            paint_fp8_register(kk, 0, W, N, K);
            e = paint_fp8_lookup(W);
        }
        if (e && e->n_out == N && e->n_in == K && e->w_bf16) {
            size_t n_elem = (size_t)M * (size_t)K;
            if (paint_xbf_ensure(n_elem) == 0) {
                int n = (int)n_elem;
                void *qargs[] = {&g_paint_d_xbf, &x, &n};
                cuLaunchKernel(kk->f_quant_bf16,
                                (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1,
                                0, 0, qargs, NULL);
                int Mv = M, Nv = N, Kv = K;
                CUdeviceptr w_bf = e->w_bf16;
                unsigned npx = (unsigned)((Nv + 127) / 128);
                unsigned npy = (unsigned)((Mv + 63) / 64);
                unsigned gx  = (npx + 3u) & ~3u;
                unsigned gy  = (npy + 3u) & ~3u;
                unsigned smem_bytes = 2u * (64u * 32u + 128u * 32u) * 2u;
                void *gargs[] = {&y, &g_paint_d_xbf, &w_bf, &Mv, &Nv, &Kv};
                cuLaunchKernel(kk->f_gemm_bf16_v7, gx, gy, 1, 256, 1, 1,
                                smem_bytes, 0, gargs, NULL);
                int has_bias = (b != 0) ? 1 : 0;
                int rows = M, cols = N;
                void *bargs[] = {&y, &b, &rows, &cols, &has_bias};
                unsigned bx = 256;
                unsigned gxd = (unsigned)((cols + bx - 1) / bx);
                unsigned gyd = (unsigned)rows;
                cuLaunchKernel(kk->f_add_bias_f32, gxd, gyd, 1, bx, 1, 1,
                                0, 0, bargs, NULL);
                return;
            }
        }
    }
    /* FP8 GEMM dispatch (Phase 4.9.2): per-tensor BF16-pipe MT4 scaled.
     * Constraints: n_in%32==0, n_out%256==0, n_tok>=16. Registry lookup
     * confirms the weight was prequantized at load with matching dims. */
    if (g_paint_use_fp8_gemm && (kk->f_gemm_fp8_mt4 || kk->f_gemm_fp8) && W &&
        M >= 1 && (K % 32) == 0) {
        const pu_fp8_entry *e = paint_fp8_lookup(W);
        if (!e) {
            paint_fp8_register(kk, 0, W, N, K);
            e = paint_fp8_lookup(W);
        }
        if (e && e->n_out == N && e->n_in == K) {
            int n_out = N, n_in = K, n_tok = M;
            CUdeviceptr w_fp8 = e->w_fp8, w_scale = e->w_scale;
            void *gargs[] = { &y, &w_fp8, &x, &b,
                              &n_out, &n_in, &n_tok, &w_scale };
            /* MT4 (64-row tiles) when n_tok large; falls back to MT1 (32-row).
             * The earlier MT4 OOB was a downstream OOM caused by holding both
             * FP8 + F32 weights — fixed by freeing F32 in paint_fp8_register. */
            unsigned gx = (unsigned)((N + 255) / 256);
            gx = (gx + 3u) & ~3u;
            if (g_paint_use_fp8_gemm_mt4 && kk->f_gemm_fp8_mt4 && M >= 64) {
                unsigned gy = (unsigned)((M + 63) / 64);
                gy = (gy + 3u) & ~3u;
                size_t smem = 4096 + 8192 * 2;
                cuLaunchKernel(kk->f_gemm_fp8_mt4, gx, gy, 1, 128, 1, 1,
                                smem, 0, gargs, NULL);
                return;
            }
            if (kk->f_gemm_fp8) {
                unsigned gy = (unsigned)((M + 31) / 32);
                gy = (gy + 3u) & ~3u;
                size_t smem = 2048 + 8192 * 2;
                cuLaunchKernel(kk->f_gemm_fp8, gx, gy, 1, 128, 1, 1,
                                smem, 0, gargs, NULL);
                return;
            }
        }
    }
    /* Scalar F32 fallback (the original 16x16 tile kernel). */
    void *args[] = { &y, &x, &W, &b, &M, &K, &N };
    unsigned gx = (unsigned)((N + 15) / 16), gy = (unsigned)((M + 15) / 16);
    cuLaunchKernel(kk->f_lin, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL);
}

static void k_silu(const pu_kernels *kk, CUdeviceptr x, int n) {
    void *args[] = { &x, &n };
    cuLaunchKernel(kk->f_silu, (unsigned)((n + 255) / 256), 1, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

static void k_groupnorm(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                          CUdeviceptr g, CUdeviceptr b, int C, int spatial,
                          int num_groups, int do_silu) {
    float eps = 1e-5f;
    void *args[] = { &out, &in, &g, &b, &C, &spatial, &num_groups, &do_silu, &eps };
    cuLaunchKernel(kk->f_gn, (unsigned)num_groups, 1, 1, 128, 1, 1,
                    128 * sizeof(float), 0, args, NULL);
}

static void k_add_chan(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr a,
                         CUdeviceptr temb, int C, int spatial) {
    void *args[] = { &out, &a, &temb, &C, &spatial };
    int total = C * spatial;
    cuLaunchKernel(kk->f_addc, (unsigned)((total + 255) / 256), 1, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

static void k_add(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr a,
                   CUdeviceptr b, int n) {
    void *args[] = { &out, &a, &b, &n };
    cuLaunchKernel(kk->f_add, (unsigned)((n + 255) / 256), 1, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

/* Lazy-grow conv scratch buffers. */
static int paint_conv_scratch_ensure(size_t xcol_bytes, size_t yt_bytes) {
    if (xcol_bytes > g_paint_xcol_max_bytes) {
        if (g_paint_d_xcol) cuMemFree(g_paint_d_xcol);
        g_paint_d_xcol = 0;
        if (cuMemAlloc(&g_paint_d_xcol, xcol_bytes) != CUDA_SUCCESS) {
            g_paint_xcol_max_bytes = 0; g_paint_d_xcol = 0; return -1;
        }
        g_paint_xcol_max_bytes = xcol_bytes;
    }
    if (yt_bytes > g_paint_yt_max_bytes) {
        if (g_paint_d_yt) cuMemFree(g_paint_d_yt);
        g_paint_d_yt = 0;
        if (cuMemAlloc(&g_paint_d_yt, yt_bytes) != CUDA_SUCCESS) {
            g_paint_yt_max_bytes = 0; g_paint_d_yt = 0; return -1;
        }
        g_paint_yt_max_bytes = yt_bytes;
    }
    return 0;
}

static void k_conv(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                    CUdeviceptr W, CUdeviceptr b,
                    int ci, int h, int w, int co, int kh, int kw, int pad) {
    /* 1x1 pad=0 FP8 path: pure GEMM, no im2col.
     *   y[hw, co] = x[hw, ci] @ Wt[co, ci]
     * Input is [ci,h,w] CHW = [ci, hw] row-major; the GEMM wants x[hw, ci],
     * so we need a CHW->HWC transpose first. Use t_hwc_chw inverse via the
     * existing chw_to_nc kernel (treats spatial as N). */
    if (g_paint_use_fp8_conv && g_paint_use_fp8_gemm && kk->f_gemm_fp8 &&
        kk->f_t_hwc_chw && kk->f_chw_nc &&
        kh == 1 && kw == 1 && pad == 0 && (ci % 32) == 0 && W) {
        int hw = h * w;
        size_t xcol_bytes = (size_t)hw * ci * sizeof(float);
        size_t yt_bytes   = (size_t)hw * co * sizeof(float);
        if (paint_conv_scratch_ensure(xcol_bytes, yt_bytes) == 0) {
            const pu_fp8_entry *e = paint_fp8_lookup(W);
            if (!e) { paint_fp8_register(kk, 0, W, co, ci); e = paint_fp8_lookup(W); }
            if (e && e->n_out == co && e->n_in == ci) {
                /* CHW [ci,hw] -> HWC [hw,ci] using chw_to_nc(C=ci, N=hw).
                 * Inline launch (k_chw_to_nc is defined later in the file). */
                {
                    int C_arg = ci, N_arg = hw;
                    void *cargs[] = { &g_paint_d_xcol, &in, &C_arg, &N_arg };
                    cuLaunchKernel(kk->f_chw_nc,
                                    (unsigned)((hw + 255) / 256),
                                    (unsigned)ci, 1, 256, 1, 1, 0, 0, cargs, NULL);
                }
                int n_out = co, n_in = ci, n_tok = hw;
                CUdeviceptr w_fp8 = e->w_fp8, w_scale = e->w_scale;
                void *gargs[] = { &g_paint_d_yt, &w_fp8, &g_paint_d_xcol, &b,
                                  &n_out, &n_in, &n_tok, &w_scale };
                unsigned gx = (unsigned)((co + 255) / 256);  gx = (gx + 3u) & ~3u;
                if (g_paint_use_fp8_gemm_mt4 && kk->f_gemm_fp8_mt4 && hw >= 64) {
                    unsigned gy = (unsigned)((hw + 63) / 64); gy = (gy + 3u) & ~3u;
                    size_t smem = 4096 + 8192 * 2;
                    cuLaunchKernel(kk->f_gemm_fp8_mt4, gx, gy, 1, 128, 1, 1,
                                    smem, 0, gargs, NULL);
                } else {
                    unsigned gy = (unsigned)((hw + 31) / 32); gy = (gy + 3u) & ~3u;
                    size_t smem = 2048 + 8192 * 2;
                    cuLaunchKernel(kk->f_gemm_fp8, gx, gy, 1, 128, 1, 1,
                                    smem, 0, gargs, NULL);
                }
                int co_arg = co, hw_arg = hw;
                void *targs[] = { &out, &g_paint_d_yt, &co_arg, &hw_arg };
                unsigned tx = (unsigned)((co + 31) / 32);
                unsigned ty = (unsigned)((hw + 7)  / 8);
                cuLaunchKernel(kk->f_t_hwc_chw, tx, ty, 1, 32, 8, 1,
                                0, 0, targs, NULL);
                return;
            }
        }
    }
    /* FP8 path: 3x3, pad=1, stride=1, ci%32==0. Reshape conv as
     *   y[hw, co] = im2col(in)[hw, ci*9] @ Wt[co, ci*9]
     * then transpose to CHW. The conv weight tensor is laid out
     * [co, ci, kh, kw] = [co, ci*9] in memory, exactly the W[n_out, n_in]
     * the GEMM wants. */
    if (g_paint_use_fp8_conv && g_paint_use_fp8_gemm && kk->f_gemm_fp8 &&
        kk->f_im2col_3x3_p1 && kk->f_t_hwc_chw &&
        kh == 3 && kw == 3 && pad == 1 && (ci % 32) == 0 && W) {
        int K = ci * 9;
        int hw = h * w;
        size_t xcol_bytes = (size_t)hw * K * sizeof(float);
        size_t yt_bytes   = (size_t)hw * co * sizeof(float);
        if (paint_conv_scratch_ensure(xcol_bytes, yt_bytes) == 0) {
            const pu_fp8_entry *e = paint_fp8_lookup(W);
            if (!e) { paint_fp8_register(kk, 0, W, co, K); e = paint_fp8_lookup(W); }
            if (e && e->n_out == co && e->n_in == K) {
                /* im2col */
                int ci_arg = ci, h_arg = h, w_arg = w;
                void *im_args[] = { &g_paint_d_xcol, &in, &ci_arg, &h_arg, &w_arg };
                unsigned imx = (unsigned)((K  + 31) / 32);
                unsigned imy = (unsigned)((hw + 7)  / 8);
                cuLaunchKernel(kk->f_im2col_3x3_p1, imx, imy, 1, 32, 8, 1,
                                0, 0, im_args, NULL);
                /* GEMM: y[hw, co] = xcol[hw, K] @ W_fp8[co, K] + b */
                int n_out = co, n_in = K, n_tok = hw;
                CUdeviceptr w_fp8 = e->w_fp8, w_scale = e->w_scale;
                void *gargs[] = { &g_paint_d_yt, &w_fp8, &g_paint_d_xcol, &b,
                                  &n_out, &n_in, &n_tok, &w_scale };
                unsigned gx = (unsigned)((co + 255) / 256);  gx = (gx + 3u) & ~3u;
                if (g_paint_use_fp8_gemm_mt4 && kk->f_gemm_fp8_mt4 && hw >= 64) {
                    unsigned gy = (unsigned)((hw + 63) / 64); gy = (gy + 3u) & ~3u;
                    size_t smem = 4096 + 8192 * 2;
                    cuLaunchKernel(kk->f_gemm_fp8_mt4, gx, gy, 1, 128, 1, 1,
                                    smem, 0, gargs, NULL);
                } else {
                    unsigned gy = (unsigned)((hw + 31) / 32); gy = (gy + 3u) & ~3u;
                    size_t smem = 2048 + 8192 * 2;
                    cuLaunchKernel(kk->f_gemm_fp8, gx, gy, 1, 128, 1, 1,
                                    smem, 0, gargs, NULL);
                }
                /* transpose [hw, co] -> [co, h, w] */
                int co_arg = co, hw_arg = hw;
                void *targs[] = { &out, &g_paint_d_yt, &co_arg, &hw_arg };
                unsigned tx = (unsigned)((co + 31) / 32);
                unsigned ty = (unsigned)((hw + 7)  / 8);
                cuLaunchKernel(kk->f_t_hwc_chw, tx, ty, 1, 32, 8, 1,
                                0, 0, targs, NULL);
                return;
            }
        }
    }
    void *args[] = { &out, &in, &W, &b, &ci, &h, &w, &co, &kh, &kw, &pad };
    int total = co * h * w;
    cuLaunchKernel(kk->f_conv, (unsigned)((total + 255) / 256), 1, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

static void k_layernorm(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                          CUdeviceptr g, CUdeviceptr b, int N, int C) {
    float eps = 1e-5f;
    void *args[] = { &out, &in, &g, &b, &N, &C, &eps };
    int tx = 128;
    cuLaunchKernel(kk->f_ln, (unsigned)N, 1, 1, tx, 1, 1,
                    tx * sizeof(float), 0, args, NULL);
}

static void k_chw_to_nc(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                          int C, int N) {
    void *args[] = { &out, &in, &C, &N };
    cuLaunchKernel(kk->f_chw_nc, (unsigned)((N + 255) / 256), (unsigned)C, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

static void k_nc_to_chw(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                          int C, int N) {
    void *args[] = { &out, &in, &C, &N };
    cuLaunchKernel(kk->f_nc_chw, (unsigned)((N + 255) / 256), (unsigned)C, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

static void k_cast_f32_to_bf16(const pu_kernels *kk, CUdeviceptr dst,
                                  CUdeviceptr src, int n) {
    void *args[] = { &src, &dst, &n };
    cuLaunchKernel(kk->f_cast_bf16, (unsigned)((n + 255) / 256), 1, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

static void k_mha(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr Q,
                   CUdeviceptr K, CUdeviceptr V,
                   int B, int N, int M, int heads, int head_dim) {
    /* BF16 TC path: flash_attn_bf16_hd64_xq for head_dim=64.
     * Layout: Q [B,N,dim], K/V [B,M,dim], dim=heads*head_dim. The BF16 kernel
     * works on a flat [N_q, dim] / [N_kv, dim] view, so we cast all of Q/K/V
     * once and launch B kernels with per-batch byte offsets. */
    if (g_paint_use_bf16_attn && head_dim == 64 &&
        kk->f_cast_bf16 && kk->f_attn_bf16_hd64_xq && g_paint_d_qbf) {
        int dim = heads * head_dim;
        size_t qn = (size_t)B * (size_t)N * (size_t)dim;
        size_t kn = (size_t)B * (size_t)M * (size_t)dim;
        if (qn <= g_paint_qkv_bf16_max_elem && kn <= g_paint_qkv_bf16_max_elem) {
            k_cast_f32_to_bf16(kk, g_paint_d_qbf, Q, (int)qn);
            k_cast_f32_to_bf16(kk, g_paint_d_kbf, K, (int)kn);
            k_cast_f32_to_bf16(kk, g_paint_d_vbf, V, (int)kn);
            size_t per_q  = (size_t)N * dim * sizeof(unsigned short);
            size_t per_kv = (size_t)M * dim * sizeof(unsigned short);
            size_t per_o  = (size_t)N * dim * sizeof(float);
            unsigned gy = (unsigned)((N + 63) / 64);
            size_t smem = (size_t)(4 * 32 * 72 * 2);
            for (int b = 0; b < B; b++) {
                CUdeviceptr qb = g_paint_d_qbf + (CUdeviceptr)((size_t)b * per_q);
                CUdeviceptr kb = g_paint_d_kbf + (CUdeviceptr)((size_t)b * per_kv);
                CUdeviceptr vb = g_paint_d_vbf + (CUdeviceptr)((size_t)b * per_kv);
                CUdeviceptr ob = out          + (CUdeviceptr)((size_t)b * per_o);
                int n_q = N, n_kv = M, nh = heads, hd = head_dim;
                void *args[] = { &ob, &qb, &kb, &vb, &n_q, &n_kv, &nh, &hd };
                cuLaunchKernel(kk->f_attn_bf16_hd64_xq, (unsigned)heads, gy, 1,
                                128, 1, 1, smem, 0, args, NULL);
            }
            return;
        }
    }
    float scale = 1.f / sqrtf((float)head_dim);
    void *args[] = { &out, &Q, &K, &V, &B, &N, &M, &heads, &head_dim, &scale };
    int tx = 32;
    cuLaunchKernel(kk->f_mha, (unsigned)(B * heads),
                    (unsigned)((N + tx - 1) / tx), 1,
                    tx, 1, 1, 0, 0, args, NULL);
}

static void k_geglu(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr gh,
                     int N, int H) {
    void *args[] = { &out, &gh, &N, &H };
    int total = N * H;
    cuLaunchKernel(kk->f_geglu, (unsigned)((total + 255) / 256), 1, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

static void k_conv_stride2(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                            CUdeviceptr W, CUdeviceptr b,
                            int ci, int h, int w, int co) {
    /* FP8 path: 3x3 stride=2 pad=1, ci%32==0. Same as the stride=1 path with
     * a stride-2 im2col producing [oh*ow, ci*9]. */
    if (g_paint_use_fp8_conv && g_paint_use_fp8_gemm && kk->f_gemm_fp8 &&
        kk->f_im2col_3x3_p1_s2 && kk->f_t_hwc_chw && (ci % 32) == 0 && W) {
        int K = ci * 9;
        int oh = h / 2, ow = w / 2;
        int ohw = oh * ow;
        size_t xcol_bytes = (size_t)ohw * K * sizeof(float);
        size_t yt_bytes   = (size_t)ohw * co * sizeof(float);
        if (paint_conv_scratch_ensure(xcol_bytes, yt_bytes) == 0) {
            const pu_fp8_entry *e = paint_fp8_lookup(W);
            if (!e) { paint_fp8_register(kk, 0, W, co, K); e = paint_fp8_lookup(W); }
            if (e && e->n_out == co && e->n_in == K) {
                int ci_arg = ci, h_arg = h, w_arg = w;
                void *im_args[] = { &g_paint_d_xcol, &in, &ci_arg, &h_arg, &w_arg };
                unsigned imx = (unsigned)((K   + 31) / 32);
                unsigned imy = (unsigned)((ohw + 7)  / 8);
                cuLaunchKernel(kk->f_im2col_3x3_p1_s2, imx, imy, 1, 32, 8, 1,
                                0, 0, im_args, NULL);
                int n_out = co, n_in = K, n_tok = ohw;
                CUdeviceptr w_fp8 = e->w_fp8, w_scale = e->w_scale;
                void *gargs[] = { &g_paint_d_yt, &w_fp8, &g_paint_d_xcol, &b,
                                  &n_out, &n_in, &n_tok, &w_scale };
                unsigned gx = (unsigned)((co + 255) / 256);  gx = (gx + 3u) & ~3u;
                if (g_paint_use_fp8_gemm_mt4 && kk->f_gemm_fp8_mt4 && ohw >= 64) {
                    unsigned gy = (unsigned)((ohw + 63) / 64); gy = (gy + 3u) & ~3u;
                    size_t smem = 4096 + 8192 * 2;
                    cuLaunchKernel(kk->f_gemm_fp8_mt4, gx, gy, 1, 128, 1, 1,
                                    smem, 0, gargs, NULL);
                } else {
                    unsigned gy = (unsigned)((ohw + 31) / 32); gy = (gy + 3u) & ~3u;
                    size_t smem = 2048 + 8192 * 2;
                    cuLaunchKernel(kk->f_gemm_fp8, gx, gy, 1, 128, 1, 1,
                                    smem, 0, gargs, NULL);
                }
                int co_arg = co, hw_arg = ohw;
                void *targs[] = { &out, &g_paint_d_yt, &co_arg, &hw_arg };
                unsigned tx = (unsigned)((co  + 31) / 32);
                unsigned ty = (unsigned)((ohw + 7)  / 8);
                cuLaunchKernel(kk->f_t_hwc_chw, tx, ty, 1, 32, 8, 1,
                                0, 0, targs, NULL);
                return;
            }
        }
    }
    void *args[] = { &out, &in, &W, &b, &ci, &h, &w, &co };
    int total = co * (h / 2) * (w / 2);
    cuLaunchKernel(kk->f_conv_s2, (unsigned)((total + 255) / 256), 1, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

static void k_upsample2x(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                          int C, int H, int W) {
    void *args[] = { &out, &in, &C, &H, &W };
    int total = C * (H * 2) * (W * 2);
    cuLaunchKernel(kk->f_up2x, (unsigned)((total + 255) / 256), 1, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

static void k_concat_chan(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr a,
                           CUdeviceptr b, int Ca, int Cb, int spatial) {
    void *args[] = { &out, &a, &b, &Ca, &Cb, &spatial };
    int total = (Ca + Cb) * spatial;
    cuLaunchKernel(kk->f_concat, (unsigned)((total + 255) / 256), 1, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

/* In-place safe RoPE: cos/sin layout [B*N, head_dim] (broadcast across heads).
 * x layout [B, N, heads, head_dim]. */
static void k_rope(const pu_kernels *kk, CUdeviceptr x,
                    CUdeviceptr cos_, CUdeviceptr sin_,
                    int B, int N, int heads, int head_dim) {
    void *args[] = { &x, &x, &cos_, &sin_, &B, &N, &heads, &head_dim };
    int half = head_dim >> 1;
    int total = B * N * heads * half;
    cuLaunchKernel(kk->f_rope, (unsigned)((total + 255) / 256), 1, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

static void k_ra_split_v(const pu_kernels *kk, CUdeviceptr V_lower,
                          CUdeviceptr V_upper, CUdeviceptr V_a, CUdeviceptr V_m,
                          int BM, int H, int D) {
    int C = H * D, total = BM * C;
    void *args[] = { &V_lower, &V_upper, &V_a, &V_m, &BM, &H, &D };
    cuLaunchKernel(kk->f_ra_split_v, (unsigned)((total + 255) / 256), 1, 1,
                    256, 1, 1, 0, 0, args, NULL);
}

/* ===== ResBlock ============================================================
 * Diffusers ResnetBlock2D ("default" mode) forward:
 *   h = norm1(x); silu; conv1
 *   t = silu(temb); time_emb_proj(t)         [B, c_out]
 *   h = h + t[:, :, None, None]
 *   h = norm2(h); silu; (dropout=identity); conv2
 *   skip = conv_shortcut(x) if c_in != c_out else x
 *   out = h + skip
 *
 * Buffers: d_x in [c_in, H, W], d_out [c_out, H, W], d_t1 / d_t2 each
 * sized max(c_in, c_out)*H*W, d_temb_proj scratch [c_out].
 */
typedef struct {
    CUdeviceptr norm1_g, norm1_b;
    CUdeviceptr conv1_w, conv1_b;
    CUdeviceptr temb_w,  temb_b;     /* time_emb_proj: 1280 -> c_out */
    CUdeviceptr norm2_g, norm2_b;
    CUdeviceptr conv2_w, conv2_b;
    CUdeviceptr skip_w,  skip_b;     /* may be 0 if c_in == c_out */
    int c_in, c_out;
} pu_resblock;

static void load_resblock(st_context *st, pu_resblock *r, const char *prefix,
                            int c_in, int c_out) {
    char buf[256];
    r->c_in = c_in; r->c_out = c_out;
    snprintf(buf, sizeof(buf), "%s.norm1.weight", prefix); r->norm1_g = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.norm1.bias",   prefix); r->norm1_b = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.conv1.weight", prefix); r->conv1_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.conv1.bias",   prefix); r->conv1_b = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.time_emb_proj.weight", prefix); r->temb_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.time_emb_proj.bias",   prefix); r->temb_b = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.norm2.weight", prefix); r->norm2_g = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.norm2.bias",   prefix); r->norm2_b = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.conv2.weight", prefix); r->conv2_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.conv2.bias",   prefix); r->conv2_b = upload_st(st, buf);
    if (c_in != c_out) {
        snprintf(buf, sizeof(buf), "%s.conv_shortcut.weight", prefix);
        r->skip_w = upload_st(st, buf);
        snprintf(buf, sizeof(buf), "%s.conv_shortcut.bias", prefix);
        r->skip_b = upload_st(st, buf);
    } else { r->skip_w = 0; r->skip_b = 0; }
}

/* d_temb : [B, 1280] (the upstream time MLP output, NOT yet silu'd).
 * Resblock applies silu(d_temb) then linear -> [B, c_out] internally.
 * d_temb_act, d_temb_proj are scratch [B*1280] and [B*c_out]. */
static void run_resblock(const pu_kernels *kk, const pu_resblock *r,
                          CUdeviceptr d_x, CUdeviceptr d_out,
                          CUdeviceptr d_t1, CUdeviceptr d_t2,
                          CUdeviceptr d_temb, CUdeviceptr d_temb_act,
                          CUdeviceptr d_temb_proj,
                          int B, int H, int W, int num_groups) {
    int sp = H * W;
    size_t in_stride  = (size_t)r->c_in  * sp * sizeof(float);
    size_t out_stride = (size_t)r->c_out * sp * sizeof(float);
    /* d_t1 / d_t2 carry max(c_in, c_out)*sp per sample (norm1 writes c_in,
     * conv1/norm2/conv2 write c_out; skip conv writes c_out). */
    int c_max = r->c_in > r->c_out ? r->c_in : r->c_out;
    size_t scratch_stride = (size_t)c_max * sp * sizeof(float);
    /* time embedding: silu(d_temb) -> linear(1280, c_out). Once for all B. */
    cuMemcpyDtoD(d_temb_act, d_temb, (size_t)B * 1280 * sizeof(float));
    k_silu(kk, d_temb_act, B * 1280);
    k_linear(kk, d_temb_proj, d_temb_act, r->temb_w, r->temb_b, B, 1280, r->c_out);
    /* Per-batch CHW path (groupnorm/conv kernels are single-sample). */
    for (int b = 0; b < B; b++) {
        CUdeviceptr xb  = d_x  + (CUdeviceptr)b * in_stride;
        CUdeviceptr ob  = d_out + (CUdeviceptr)b * out_stride;
        CUdeviceptr t1b = d_t1 + (CUdeviceptr)b * scratch_stride;
        CUdeviceptr t2b = d_t2 + (CUdeviceptr)b * scratch_stride;
        CUdeviceptr tb  = d_temb_proj + (CUdeviceptr)b * r->c_out * sizeof(float);
        /* h = silu(norm1(x))  [c_in, H, W] -> t1b */
        k_groupnorm(kk, t1b, xb, r->norm1_g, r->norm1_b,
                     r->c_in, sp, num_groups, 1);
        /* h = conv1(h)  -> t2b [c_out, H, W] */
        k_conv(kk, t2b, t1b, r->conv1_w, r->conv1_b,
                r->c_in, H, W, r->c_out, 3, 3, 1);
        /* h += temb broadcast */
        k_add_chan(kk, t2b, t2b, tb, r->c_out, sp);
        /* h = silu(norm2(h)) -> t1b */
        k_groupnorm(kk, t1b, t2b, r->norm2_g, r->norm2_b,
                     r->c_out, sp, num_groups, 1);
        /* h = conv2(h) -> t2b [c_out, H, W] */
        k_conv(kk, t2b, t1b, r->conv2_w, r->conv2_b,
                r->c_out, H, W, r->c_out, 3, 3, 1);
        /* skip path */
        if (r->skip_w) {
            k_conv(kk, t1b, xb, r->skip_w, r->skip_b,
                    r->c_in, H, W, r->c_out, 1, 1, 0);
            k_add(kk, ob, t2b, t1b, r->c_out * sp);
        } else {
            k_add(kk, ob, t2b, xb, r->c_out * sp);
        }
    }
}

/* ===== Transformer2DModel block ============================================
 * Diffusers Transformer2DModel(use_linear_projection=True) forward:
 *   residual = x                     [B, C, H, W]
 *   h = norm(x)                       (GroupNorm 32 grp)
 *   h = chw_to_nc(h)                  -> [B, N=H*W, C]
 *   h = proj_in(h)                    Linear C -> C
 *   for each BasicTransformerBlock:
 *     a1 = self_attn(layernorm(h))            ; h += a1
 *     a2 = cross_attn(layernorm(h), text)     ; h += a2
 *     a3 = ff(layernorm(h))                   ; h += a3
 *   h = proj_out(h)                   Linear C -> C
 *   h = nc_to_chw(h)                  -> [B, C, H, W]
 *   out = h + residual
 *
 * BasicTransformerBlock attention: Attention class
 *   to_q [C, C] (no bias)             Q = x @ to_q_w + 0
 *   to_k [Ckv, C] (no bias)
 *   to_v [Ckv, C] (no bias)
 *   to_out.0 [C, C] (with bias)
 *   For self-attn: Ckv = C, K/V = x. For cross-attn: Ckv = cross_dim, K/V = text.
 *   heads = num_attention_heads, head_dim = C / heads.
 *
 * FF (GEGLU + Linear):
 *   net.0.proj [2*4C, C] (with bias)  -> [N, 8C]
 *   GEGLU: out = first_half * GELU(second_half) -> [N, 4C]
 *   net.2 [C, 4C] (with bias)         -> [N, C]
 */
typedef struct {
    /* attn1 (self) and attn2 (cross) share layout */
    CUdeviceptr to_q_w, to_k_w, to_v_w;     /* no bias */
    CUdeviceptr to_out_w, to_out_b;
    /* RefAttn extras (only populated for attn_refview when has_ra=1).
     * RefAttnProcessor2_0 uses shared Q/K but per-material V/out:
     *   attn_refview.processor.to_v_mr.weight
     *   attn_refview.processor.to_out_mr.0.{weight,bias} */
    CUdeviceptr to_v_mr_w;
    CUdeviceptr to_out_mr_w, to_out_mr_b;
} pu_attention;

typedef struct {
    CUdeviceptr norm1_g, norm1_b;
    pu_attention attn1;                     /* self-attn */
    pu_attention attn_multiview;            /* MA cross-view (zero if !has_ma) */
    pu_attention attn1_mr;                  /* MDA per-material (mr) self-attn (zero if !has_mda) */
    pu_attention attn_refview;              /* RA reference cross-attn (zero if !has_ra) */
    CUdeviceptr norm2_g, norm2_b;
    pu_attention attn2;                     /* cross-attn (text) */
    pu_attention attn_dino;                 /* DINO cross-attn (zero if !has_dino) */
    CUdeviceptr norm3_g, norm3_b;
    CUdeviceptr ff0_w, ff0_b;               /* GEGLU.proj */
    CUdeviceptr ff2_w, ff2_b;               /* output linear */
} pu_basic_block;

typedef struct {
    CUdeviceptr norm_g, norm_b;             /* group_norm */
    CUdeviceptr proj_in_w, proj_in_b;       /* Linear */
    CUdeviceptr proj_out_w, proj_out_b;     /* Linear */
    pu_basic_block *blocks;                 /* num_blocks */
    int num_blocks;
    int channels;        /* C */
    int num_heads;
    int head_dim;
    int cross_dim;       /* 1024 for text */
    int ff_inner;        /* GEGLU inner dim = 4*C in SD */
    int has_dino;        /* 1 if attn_dino weights loaded */
    int has_ma;          /* 1 if attn_multiview weights loaded */
    int has_mda;         /* 1 if attn1_mr (per-material self-attn) weights loaded */
    int has_ra;          /* 1 if attn_refview weights loaded */
    int n_pbr;           /* MA/MDA/RA: number of PBR materials (e.g. 2). 0 if none */
    int n_gen;           /* MA/MDA/RA: number of generation views (e.g. 2). 0 if none */
} pu_transformer;

/* ------- RA condition_embed_dict shared cache --------------------------------
 * Each transformer block owns a slot. The dual-stream pass walks all blocks in
 * deterministic order (down/mid/up) with ra_mode='w' and writes its
 * norm_hidden_states (rearranged (b n) l c -> b (n l) c) into the slot. The
 * main-stream pass walks the SAME ordering with ra_mode='r' and reads it.
 * Both UNet topologies are identical, so a flat counter index suffices. */
typedef struct {
    CUdeviceptr d;
    int B;       /* B (== Beff_dual) */
    int NL;      /* N_ref * L (token count) */
    int C;       /* channels */
} pu_ra_slot;

typedef struct {
    pu_ra_slot *slots;
    int n_slots;
    int idx;     /* walk-counter; reset between passes */
} pu_ra_cache;

/* RA mode global, set per pass:
 *   0 = RA off (default; no caching, no RA branch)
 *   1 = 'w' = dual-stream pass: cache norm_hidden_states per block
 *   2 = 'r' = main pass: read cache, apply attn_refview branch */
static int g_ra_mode = 0;
static pu_ra_cache g_ra_cache = {0};
static int g_ra_n_ref = 1;          /* N_ref for the write pass */

/* ----- PoseRoPE per-resolution cos/sin lookup (global, shared by all xfm
 * blocks of a forward pass). Populated by build_rope_levels() before forward;
 * looked up in run_transformer's MA branch via rope_for_N(N). ---------- */
typedef struct {
    int N;               /* H*W of this latent level */
    int Np;              /* n_gen * N (== voxel_indices key) */
    CUdeviceptr d_cos;   /* [Bp*Np, head_dim] f32 */
    CUdeviceptr d_sin;
} pu_rope_level;

static pu_rope_level g_rope_levels[8];
static int g_rope_n_levels = 0;
static int g_rope_head_dim = 0;
static int g_rope_Bp = 0;

static const pu_rope_level *rope_for_N(int N) {
    for (int i = 0; i < g_rope_n_levels; i++)
        if (g_rope_levels[i].N == N) return &g_rope_levels[i];
    return NULL;
}

static void load_attention(st_context *st, pu_attention *a, const char *prefix) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s.to_q.weight", prefix); a->to_q_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.to_k.weight", prefix); a->to_k_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.to_v.weight", prefix); a->to_v_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.to_out.0.weight", prefix); a->to_out_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.to_out.0.bias",   prefix); a->to_out_b = upload_st(st, buf);
}

/* Load attn_refview: stock to_{q,k,v}/to_out plus per-material extras
 *   processor.to_v_mr.weight
 *   processor.to_out_mr.0.{weight,bias} */
static void load_attention_refview(st_context *st, pu_attention *a, const char *prefix) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s.to_q.weight",      prefix); a->to_q_w   = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.to_k.weight",      prefix); a->to_k_w   = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.to_v.weight",      prefix); a->to_v_w   = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.to_out.0.weight",  prefix); a->to_out_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.to_out.0.bias",    prefix); a->to_out_b = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.processor.to_v_mr.weight",     prefix); a->to_v_mr_w   = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.processor.to_out_mr.0.weight", prefix); a->to_out_mr_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.processor.to_out_mr.0.bias",   prefix); a->to_out_mr_b = upload_st(st, buf);
}

/* Load MDA `_mr` per-material self-attn weights. Naming differs from the stock
 * Attention module: q/k/v live directly under `attn1.processor.to_{q,k,v}_mr.weight`
 * (no bias) and the out lives at `attn1.processor.to_out_mr.0.{weight,bias}`. */
static void load_attention_mr(st_context *st, pu_attention *a, const char *prefix) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s.processor.to_q_mr.weight", prefix); a->to_q_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.processor.to_k_mr.weight", prefix); a->to_k_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.processor.to_v_mr.weight", prefix); a->to_v_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.processor.to_out_mr.0.weight", prefix); a->to_out_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.processor.to_out_mr.0.bias",   prefix); a->to_out_b = upload_st(st, buf);
}

static void load_transformer(st_context *st, pu_transformer *T,
                               const char *prefix, int channels,
                               int num_heads, int cross_dim, int num_blocks,
                               int has_dino, int has_ma, int has_mda, int has_ra,
                               int n_pbr, int n_gen) {
    char buf[256];
    T->channels  = channels;
    T->num_heads = num_heads;
    T->head_dim  = channels / num_heads;
    T->cross_dim = cross_dim;
    T->ff_inner  = channels * 4;
    T->num_blocks = num_blocks;
    T->has_dino   = has_dino;
    T->has_ma     = has_ma;
    T->has_mda    = has_mda;
    T->has_ra     = has_ra;
    T->n_pbr      = (has_ma || has_mda || has_ra) ? n_pbr : 0;
    T->n_gen      = (has_ma || has_mda || has_ra) ? n_gen : 0;

    snprintf(buf, sizeof(buf), "%s.norm.weight", prefix); T->norm_g = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.norm.bias",   prefix); T->norm_b = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.proj_in.weight",  prefix); T->proj_in_w  = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.proj_in.bias",    prefix); T->proj_in_b  = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.proj_out.weight", prefix); T->proj_out_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.proj_out.bias",   prefix); T->proj_out_b = upload_st(st, buf);

    T->blocks = (pu_basic_block *)calloc((size_t)num_blocks, sizeof(pu_basic_block));
    for (int i = 0; i < num_blocks; i++) {
        char bp[200], sub[256];
        snprintf(bp, sizeof(bp), "%s.transformer_blocks.%d", prefix, i);
        pu_basic_block *bb = &T->blocks[i];
        snprintf(sub, sizeof(sub), "%s.norm1.weight", bp); bb->norm1_g = upload_st(st, sub);
        snprintf(sub, sizeof(sub), "%s.norm1.bias",   bp); bb->norm1_b = upload_st(st, sub);
        snprintf(sub, sizeof(sub), "%s.attn1", bp); load_attention(st, &bb->attn1, sub);
        snprintf(sub, sizeof(sub), "%s.norm2.weight", bp); bb->norm2_g = upload_st(st, sub);
        snprintf(sub, sizeof(sub), "%s.norm2.bias",   bp); bb->norm2_b = upload_st(st, sub);
        snprintf(sub, sizeof(sub), "%s.attn2", bp); load_attention(st, &bb->attn2, sub);
        snprintf(sub, sizeof(sub), "%s.norm3.weight", bp); bb->norm3_g = upload_st(st, sub);
        snprintf(sub, sizeof(sub), "%s.norm3.bias",   bp); bb->norm3_b = upload_st(st, sub);
        snprintf(sub, sizeof(sub), "%s.ff.net.0.proj.weight", bp); bb->ff0_w = upload_st(st, sub);
        snprintf(sub, sizeof(sub), "%s.ff.net.0.proj.bias",   bp); bb->ff0_b = upload_st(st, sub);
        snprintf(sub, sizeof(sub), "%s.ff.net.2.weight",      bp); bb->ff2_w = upload_st(st, sub);
        snprintf(sub, sizeof(sub), "%s.ff.net.2.bias",        bp); bb->ff2_b = upload_st(st, sub);
        if (has_dino) {
            snprintf(sub, sizeof(sub), "%s.attn_dino", bp);
            load_attention(st, &bb->attn_dino, sub);
        } else {
            bb->attn_dino.to_q_w = bb->attn_dino.to_k_w = bb->attn_dino.to_v_w = 0;
            bb->attn_dino.to_out_w = bb->attn_dino.to_out_b = 0;
        }
        if (has_ma) {
            snprintf(sub, sizeof(sub), "%s.attn_multiview", bp);
            load_attention(st, &bb->attn_multiview, sub);
        } else {
            bb->attn_multiview.to_q_w = bb->attn_multiview.to_k_w = bb->attn_multiview.to_v_w = 0;
            bb->attn_multiview.to_out_w = bb->attn_multiview.to_out_b = 0;
        }
        if (has_mda) {
            snprintf(sub, sizeof(sub), "%s.attn1", bp);
            load_attention_mr(st, &bb->attn1_mr, sub);
        } else {
            bb->attn1_mr.to_q_w = bb->attn1_mr.to_k_w = bb->attn1_mr.to_v_w = 0;
            bb->attn1_mr.to_out_w = bb->attn1_mr.to_out_b = 0;
        }
        if (has_ra) {
            snprintf(sub, sizeof(sub), "%s.attn_refview", bp);
            load_attention_refview(st, &bb->attn_refview, sub);
        } else {
            bb->attn_refview.to_q_w = bb->attn_refview.to_k_w = bb->attn_refview.to_v_w = 0;
            bb->attn_refview.to_out_w = bb->attn_refview.to_out_b = 0;
            bb->attn_refview.to_v_mr_w = bb->attn_refview.to_out_mr_w = bb->attn_refview.to_out_mr_b = 0;
        }
    }
}

/* run_attention: x[B,N,C] -> out[B,N,C], with K/V from kvsrc[B,M,Ckv].
 * For self-attn pass kvsrc=x, M=N, Ckv=C. For cross-attn pass text, M=77,
 * Ckv=cross_dim. Note diffusers' to_k/to_v have shape [C, Ckv] (out_features
 * = C, in_features = Ckv) so output of those matmul is [B*M, C] = inner dim.
 *
 * Scratch needed per call: d_q, d_k, d_v each [B, max(N,M), C], d_attn [B,N,C].
 */
static void run_attention(const pu_kernels *kk, const pu_attention *a,
                            CUdeviceptr d_in, CUdeviceptr d_kvsrc,
                            CUdeviceptr d_out,
                            CUdeviceptr d_q, CUdeviceptr d_k, CUdeviceptr d_v,
                            CUdeviceptr d_attn,
                            int B, int N, int M, int C, int Ckv,
                            int heads, int head_dim) {
    /* Q = in @ to_q.weight^T  (in:[B*N, C], to_q_w:[C,C]) -> [B*N, C] */
    k_linear(kk, d_q, d_in, a->to_q_w, 0, B * N, C,   C);
    /* K = kvsrc @ to_k.weight^T (kvsrc:[B*M,Ckv], to_k_w:[C,Ckv]) -> [B*M, C] */
    k_linear(kk, d_k, d_kvsrc, a->to_k_w, 0, B * M, Ckv, C);
    k_linear(kk, d_v, d_kvsrc, a->to_v_w, 0, B * M, Ckv, C);
    /* MHA */
    k_mha(kk, d_attn, d_q, d_k, d_v, B, N, M, heads, head_dim);
    /* out = attn @ to_out.weight^T + to_out.bias */
    k_linear(kk, d_out, d_attn, a->to_out_w, a->to_out_b, B * N, C, C);
}

/* run_transformer:
 *  d_x : [B, C, H, W] in/out (overwritten)
 *  d_text : [B, M=77, cross_dim]
 *  Scratch: a fistful of [B, N, C] / [B, M, C] / [B, N, C] buffers laid out
 *  by the caller (run_transformer doesn't allocate). */
typedef struct {
    CUdeviceptr d_resid;     /* [B,C,H,W] copy of input */
    CUdeviceptr d_nc;        /* [B,N,C]   primary token buffer */
    CUdeviceptr d_nc_b;      /* [B,N,C]   secondary token buffer */
    CUdeviceptr d_norm;      /* [B,N,C]   layernorm output */
    CUdeviceptr d_q;         /* [B,N,C]   */
    CUdeviceptr d_k;         /* [B,M,C]   K projection (M=N for self-attn) */
    CUdeviceptr d_v;         /* [B,M,C]   */
    CUdeviceptr d_attn;      /* [B,N,C]   pre-out_proj attn output */
    CUdeviceptr d_ff_gh;     /* [B,N,2*ff_inner] GEGLU pre-act */
    CUdeviceptr d_ff_h;      /* [B,N,ff_inner]   GEGLU post */
} pu_xfm_scratch;

/* d_dino : [B, M_dino, cross_dim] (already tiled per batch); pass 0 to skip.
 * Used only if T->has_dino and d_dino != 0. The DINO branch reuses the
 * post-norm2 norm_hidden_states (cached in S->d_norm) so we apply attn_dino
 * with the same Q-side tokens as text-cross-attn. */
static void run_transformer(const pu_kernels *kk, const pu_transformer *T,
                              CUdeviceptr d_x, CUdeviceptr d_text,
                              CUdeviceptr d_dino, int M_dino,
                              int B, int H, int W, int M_text,
                              const pu_xfm_scratch *S) {
    int C = T->channels;
    int N = H * W;
    int sp = N;
    /* residual = x */
    cuMemcpyDtoD(S->d_resid, d_x, (size_t)B * C * sp * sizeof(float));
    /* GroupNorm in CHW (no fused silu) */
    /* The kernel expects single-batch CHW; loop over B */
    for (int b = 0; b < B; b++) {
        CUdeviceptr xb = d_x      + (CUdeviceptr)b * C * sp * sizeof(float);
        CUdeviceptr nb = S->d_nc  + (CUdeviceptr)b * C * sp * sizeof(float);
        k_groupnorm(kk, nb, xb, T->norm_g, T->norm_b, C, sp, 32, 0);
    }
    /* CHW -> NC (per batch) */
    for (int b = 0; b < B; b++) {
        CUdeviceptr nb = S->d_nc + (CUdeviceptr)b * C * sp * sizeof(float);
        CUdeviceptr ob = S->d_nc_b + (CUdeviceptr)b * C * sp * sizeof(float);
        k_chw_to_nc(kk, ob, nb, C, N);
    }
    /* d_nc_b now [B, N, C]. Do proj_in -> d_nc */
    k_linear(kk, S->d_nc, S->d_nc_b, T->proj_in_w, T->proj_in_b, B * N, C, C);
    /* h := S->d_nc; for each block: */
    for (int bi = 0; bi < T->num_blocks; bi++) {
        const pu_basic_block *bb = &T->blocks[bi];
        /* --- self-attn ---
         * Stock (use_mda=False): one attn1 over [B,N,C].
         * MDA path (use_mda=True, pbr_setting=["albedo","mr"]):
         *   group B as B_in*n_pbr*n_gen (outer-most b, then pbr, then gen),
         *   each material gets its own per-(B_in,n_gen)-slice run_attention
         *   with weights attn1 (albedo) or attn1_mr (mr). */
        k_layernorm(kk, S->d_norm, S->d_nc, bb->norm1_g, bb->norm1_b, B * N, C);
        /* --- RA write: cache norm_hidden_states for this layer.
         * Layout (b n) l c -> b (n l) c is a no-op memcpy since both are
         * row-major over the same B*N*C floats. */
        if (g_ra_mode == 1) {
            int ci = g_ra_cache.idx++;
            if (ci >= g_ra_cache.n_slots) {
                fprintf(stderr, "RA cache overflow: idx=%d slots=%d\n", ci, g_ra_cache.n_slots);
                exit(1);
            }
            pu_ra_slot *sl = &g_ra_cache.slots[ci];
            size_t bytes = (size_t)B * N * C * sizeof(float);
            if (sl->d == 0) {
                cuMemAlloc(&sl->d, bytes);
                sl->B = B; sl->NL = N; sl->C = C;
            }
            cuMemcpyDtoD(sl->d, S->d_norm, bytes);
        }
        if (T->has_mda && T->n_pbr > 0 && T->n_gen > 0) {
            int n_pbr = T->n_pbr, n_gen = T->n_gen;
            int B_in = B / (n_pbr * n_gen);
            size_t row_bytes = (size_t)N * C * sizeof(float);
            for (int b = 0; b < B_in; b++) {
                for (int p = 0; p < n_pbr; p++) {
                    int off = (b * n_pbr + p) * n_gen;
                    const pu_attention *a = (p == 0) ? &bb->attn1 : &bb->attn1_mr;
                    CUdeviceptr dn  = S->d_norm  + (CUdeviceptr)off * row_bytes;
                    CUdeviceptr dnb = S->d_nc_b  + (CUdeviceptr)off * row_bytes;
                    CUdeviceptr dq  = S->d_q     + (CUdeviceptr)off * row_bytes;
                    CUdeviceptr dk  = S->d_k     + (CUdeviceptr)off * row_bytes;
                    CUdeviceptr dv  = S->d_v     + (CUdeviceptr)off * row_bytes;
                    CUdeviceptr da  = S->d_attn  + (CUdeviceptr)off * row_bytes;
                    run_attention(kk, a, dn, dn, dnb, dq, dk, dv, da,
                                   n_gen, N, N, C, C, T->num_heads, T->head_dim);
                }
            }
        } else {
            run_attention(kk, &bb->attn1,
                           S->d_norm, S->d_norm, S->d_nc_b,
                           S->d_q, S->d_k, S->d_v, S->d_attn,
                           B, N, N, C, C, T->num_heads, T->head_dim);
        }
        k_add(kk, S->d_nc, S->d_nc, S->d_nc_b, B * N * C);
        /* --- RA read: reference cross-attn against cached norm_hidden_states.
         * Layout: norm_hidden_states is [(b n_pbr n_gen), L, C].
         * Albedo-only Q: take rows [b*n_pbr*n_gen .. b*n_pbr*n_gen + n_gen)
         * for each b -> [B_in, n_gen*L, C].
         * K from cache [B_in, N_ref*L, C] (B_in == cached B).
         * V two ways: V_albedo using to_v, V_mr using to_v_mr — same Q,K so
         * we run two separate run_attention calls (mathematically equivalent
         * to concat-V split-out_proj).
         * Result: out_albedo [B_in, n_gen*L, C] -> rows [b*n_pbr*n_gen + 0*n_gen)
         *          out_mr     [B_in, n_gen*L, C] -> rows [b*n_pbr*n_gen + 1*n_gen)
         * Add into d_nc with ref_scale=1.0.
         * Only fires on read pass (g_ra_mode==2) and only if has_ra. */
        if (g_ra_mode == 2 && T->has_ra && T->n_pbr > 0 && T->n_gen > 0) {
            int n_pbr = T->n_pbr, n_gen = T->n_gen;
            int B_in = B / (n_pbr * n_gen);
            int ci = g_ra_cache.idx++;
            if (ci >= g_ra_cache.n_slots) {
                fprintf(stderr, "RA cache underflow read: idx=%d slots=%d\n", ci, g_ra_cache.n_slots);
                exit(1);
            }
            const pu_ra_slot *sl = &g_ra_cache.slots[ci];
            const pu_attention *a = &bb->attn_refview;
            int M_ref = sl->NL;     /* N_ref * L */
            size_t row_bytes = (size_t)N * C * sizeof(float);
            int Nq = n_gen * N;
            /* Zero d_nc_b — RA writes only the rows it covers; remaining rows
             * (e.g. n_pbr=1 padding) must contribute 0 to the k_add below. */
            cuMemsetD8(S->d_nc_b, 0, (size_t)B * N * C * sizeof(float));
            for (int b = 0; b < B_in; b++) {
                /* Q source: albedo-only slice of d_norm */
                CUdeviceptr q_src   = S->d_norm + (CUdeviceptr)(b * n_pbr * n_gen) * row_bytes;
                CUdeviceptr kv_src  = sl->d     + (CUdeviceptr) b * M_ref * C * sizeof(float);
                CUdeviceptr out_alb = S->d_nc_b + (CUdeviceptr)(b * n_pbr * n_gen) * row_bytes;
                CUdeviceptr out_mr  = S->d_nc_b + (CUdeviceptr)(b * n_pbr * n_gen + n_gen) * row_bytes;
                /* RefAttnProcessor2_0: shared Q/K, V = cat(to_v(eh), to_v_mr(eh))
                 * along last dim (=2C). Reshape V into [M, H, 2D] per-head. SDPA
                 * with per-head V dim 2D. Split output along last dim into 2
                 * chunks of D each: chunk0 -> to_out (albedo), chunk1 -> to_out_mr.
                 *
                 * Equivalently: build V_lower[m, h*D+d] = V_total[m, 2hD+d] and
                 * V_upper[m, h*D+d] = V_total[m, 2hD+D+d]. Each is [M, C]. Run
                 * the standard MHA twice (shared QK pass cost duplicated, but
                 * simpler than reshaping the kernel). */
                /* Q proj (shared) */
                k_linear(kk, S->d_q, q_src, a->to_q_w, 0, Nq, C, C);
                /* K proj (shared) */
                k_linear(kk, S->d_k, kv_src, a->to_k_w, 0, M_ref, C, C);
                /* V_a, V_m (use d_v slot for V_a; need a temp for V_m). Use
                 * d_attn temp slot for V_m since it's only valid post-MHA. */
                CUdeviceptr d_va = S->d_v;
                CUdeviceptr d_vm = S->d_attn;
                k_linear(kk, d_va, kv_src, a->to_v_w,    0, M_ref, C, C);
                k_linear(kk, d_vm, kv_src, a->to_v_mr_w, 0, M_ref, C, C);
                /* V_lower and V_upper into separate FF scratch slots
                 * (no overlap with d_va/d_vm inputs). */
                CUdeviceptr d_vlow = S->d_ff_h;
                CUdeviceptr d_vup  = S->d_ff_gh;
                k_ra_split_v(kk, d_vlow, d_vup, d_va, d_vm,
                              M_ref, T->num_heads, T->head_dim);
                /* MHA chunk0 (albedo) -> reuse d_va */
                k_mha(kk, d_va, S->d_q, S->d_k, d_vlow,
                       1, Nq, M_ref, T->num_heads, T->head_dim);
                k_linear(kk, out_alb, d_va, a->to_out_w, a->to_out_b, Nq, C, C);
                /* MHA chunk1 (mr) -> reuse d_vm */
                k_mha(kk, d_vm, S->d_q, S->d_k, d_vup,
                       1, Nq, M_ref, T->num_heads, T->head_dim);
                k_linear(kk, out_mr, d_vm, a->to_out_mr_w, a->to_out_mr_b, Nq, C, C);
            }
            /* Add the per-material RA output into d_nc. We wrote albedo rows
             * and mr rows into d_nc_b's same locations, so a single B*N*C
             * add covers both. (rows that aren't written stay valid because
             * only the n_pbr*n_gen rows per b matter, and we wrote all of them.) */
            k_add(kk, S->d_nc, S->d_nc, S->d_nc_b, B * N * C);
        }
        /* --- multiview cross-view attn (MA) ---
         * Reshape (b n_pbr n) l c -> (b n_pbr) (n l) c — pure stride math
         * since data is laid out [b, n_pbr, n, L, C] contiguously.
         * Self-attn at B'=B/n_gen, N'=n_gen*N (=n_gen*L). RoPE OFF (Phase 4.3a).
         * Output added to d_nc with mva_scale=1.0.
         * Operates on the SAME norm1 output (S->d_norm), not on post-self-attn
         * d_nc — matches modules.py's "norm_hidden_states" reuse. */
        if (T->has_ma && T->n_gen > 1) {
            int Bp = B / T->n_gen;     /* (b * n_pbr) */
            int Np = T->n_gen * N;
            const pu_attention *a = &bb->attn_multiview;
            const pu_rope_level *rl = rope_for_N(N);
            /* QKV projections */
            k_linear(kk, S->d_q, S->d_norm, a->to_q_w, 0, Bp * Np, C, C);
            k_linear(kk, S->d_k, S->d_norm, a->to_k_w, 0, Bp * Np, C, C);
            k_linear(kk, S->d_v, S->d_norm, a->to_v_w, 0, Bp * Np, C, C);
            if (rl) {
                k_rope(kk, S->d_q, rl->d_cos, rl->d_sin,
                        Bp, Np, T->num_heads, T->head_dim);
                k_rope(kk, S->d_k, rl->d_cos, rl->d_sin,
                        Bp, Np, T->num_heads, T->head_dim);
            }
            k_mha(kk, S->d_attn, S->d_q, S->d_k, S->d_v,
                   Bp, Np, Np, T->num_heads, T->head_dim);
            k_linear(kk, S->d_nc_b, S->d_attn, a->to_out_w, a->to_out_b,
                      Bp * Np, C, C);
            k_add(kk, S->d_nc, S->d_nc, S->d_nc_b, B * N * C);
        }
        /* --- cross-attn (text) --- */
        k_layernorm(kk, S->d_norm, S->d_nc, bb->norm2_g, bb->norm2_b, B * N, C);
        run_attention(kk, &bb->attn2,
                       S->d_norm, d_text, S->d_nc_b,
                       S->d_q, S->d_k, S->d_v, S->d_attn,
                       B, N, M_text, C, T->cross_dim,
                       T->num_heads, T->head_dim);
        k_add(kk, S->d_nc, S->d_nc, S->d_nc_b, B * N * C);
        /* --- cross-attn (DINO) --- shares S->d_norm with text-cross.
         * Output residual added to d_nc (same as text path). */
        if (T->has_dino && d_dino) {
            run_attention(kk, &bb->attn_dino,
                           S->d_norm, d_dino, S->d_nc_b,
                           S->d_q, S->d_k, S->d_v, S->d_attn,
                           B, N, M_dino, C, T->cross_dim,
                           T->num_heads, T->head_dim);
            k_add(kk, S->d_nc, S->d_nc, S->d_nc_b, B * N * C);
        }
        /* --- FF (GEGLU) --- */
        k_layernorm(kk, S->d_norm, S->d_nc, bb->norm3_g, bb->norm3_b, B * N, C);
        k_linear(kk, S->d_ff_gh, S->d_norm, bb->ff0_w, bb->ff0_b,
                  B * N, C, 2 * T->ff_inner);
        k_geglu(kk, S->d_ff_h, S->d_ff_gh, B * N, T->ff_inner);
        k_linear(kk, S->d_nc_b, S->d_ff_h, bb->ff2_w, bb->ff2_b,
                  B * N, T->ff_inner, C);
        k_add(kk, S->d_nc, S->d_nc, S->d_nc_b, B * N * C);
    }
    /* proj_out -> d_nc_b */
    k_linear(kk, S->d_nc_b, S->d_nc, T->proj_out_w, T->proj_out_b, B * N, C, C);
    /* NC -> CHW back into d_x (pre-residual) */
    for (int b = 0; b < B; b++) {
        CUdeviceptr nb = S->d_nc_b + (CUdeviceptr)b * C * sp * sizeof(float);
        CUdeviceptr xb = d_x       + (CUdeviceptr)b * C * sp * sizeof(float);
        k_nc_to_chw(kk, xb, nb, C, N);
    }
    /* out = x + residual */
    k_add(kk, d_x, d_x, S->d_resid, B * C * sp);
}

/* ===== Down/Mid/Up blocks =================================================
 *
 * SD-2.1 paint UNet topology (block_out=[320,640,1280,1280], layers/block=2):
 *   conv_in 12->320                                            push 320@64
 *   db0 CrossAttnDownBlock2D (320, heads=5):
 *      res 320->320 + xfm                                      push 320@64
 *      res 320->320 + xfm                                      push 320@64
 *      down 320 stride2                                        push 320@32
 *   db1 CrossAttnDownBlock2D (640, heads=10):
 *      res 320->640 + xfm                                      push 640@32
 *      res 640->640 + xfm                                      push 640@32
 *      down 640 stride2                                        push 640@16
 *   db2 CrossAttnDownBlock2D (1280, heads=20):
 *      res 640->1280 + xfm                                     push 1280@16
 *      res 1280->1280 + xfm                                    push 1280@16
 *      down 1280 stride2                                       push 1280@8
 *   db3 DownBlock2D (1280, no attn, no down):
 *      res 1280->1280                                          push 1280@8
 *      res 1280->1280                                          push 1280@8
 *   mid: res 1280 + xfm 1280(20h) + res 1280
 *   up0 UpBlock2D (out=1280, in=1280, prev=1280, +up):
 *      pop[1280@8] cat -> res(2560->1280) (3x)  -> upsample 16
 *   up1 CrossAttnUpBlock2D (out=1280, in=640, prev=1280, +up):
 *      pop[1280@16] cat -> res(2560->1280) + xfm   x2
 *      pop[640@16]  cat -> res(1920->1280) + xfm
 *      upsample 32
 *   up2 CrossAttnUpBlock2D (out=640, in=320, prev=1280, +up):
 *      pop[640@32] cat -> res(1920->640) + xfm
 *      pop[640@32] cat -> res(1280->640) + xfm
 *      pop[320@32] cat -> res(960->640)  + xfm
 *      upsample 64
 *   up3 CrossAttnUpBlock2D (out=320, in=320, prev=640, no up):
 *      pop[320@64] cat -> res(960->320) + xfm
 *      pop[320@64] cat -> res(640->320) + xfm
 *      pop[320@64] cat -> res(640->320) + xfm
 *   conv_norm_out (32 grp, +silu) -> conv_out 320->4 (3x3, pad 1)
 */

typedef struct {
    pu_resblock res[2];
    pu_transformer xfm[2];
    int has_xfm;
    CUdeviceptr down_w, down_b;   /* 0 if no downsampler */
    int c_in, c_out;
    int num_heads;                /* unused if !has_xfm */
} pu_down_block;

typedef struct {
    pu_resblock res[3];
    pu_transformer xfm[3];
    int has_xfm;
    CUdeviceptr up_w, up_b;       /* 0 if no upsampler */
    int c_out;
    int num_heads;
    /* Per-resnet input channel counts vary, computed at load time. */
    int res_in[3];
    /* Per-resnet skip channel counts (popped from down stack). */
    int skip_ch[3];
} pu_up_block;

/* Global weight-prefix used by all load_*_block helpers — set to ""
 * for the main UNet, "unet_dual." when loading the dual-stream copy. */
static const char *g_load_wp = "";

static void load_down_block(st_context *st, pu_down_block *db, int idx,
                              int c_in, int c_out, int num_heads,
                              int has_xfm, int has_down, int cross_dim,
                              int has_dino, int has_ma, int has_mda, int has_ra,
                              int n_pbr, int n_gen) {
    char buf[256];
    db->c_in = c_in; db->c_out = c_out; db->has_xfm = has_xfm;
    db->num_heads = num_heads;
    snprintf(buf, sizeof(buf), "%sdown_blocks.%d.resnets.0", g_load_wp, idx);
    load_resblock(st, &db->res[0], buf, c_in, c_out);
    snprintf(buf, sizeof(buf), "%sdown_blocks.%d.resnets.1", g_load_wp, idx);
    load_resblock(st, &db->res[1], buf, c_out, c_out);
    if (has_xfm) {
        snprintf(buf, sizeof(buf), "%sdown_blocks.%d.attentions.0", g_load_wp, idx);
        load_transformer(st, &db->xfm[0], buf, c_out, num_heads, cross_dim, 1, has_dino, has_ma, has_mda, has_ra, n_pbr, n_gen);
        snprintf(buf, sizeof(buf), "%sdown_blocks.%d.attentions.1", g_load_wp, idx);
        load_transformer(st, &db->xfm[1], buf, c_out, num_heads, cross_dim, 1, has_dino, has_ma, has_mda, has_ra, n_pbr, n_gen);
    }
    if (has_down) {
        snprintf(buf, sizeof(buf), "%sdown_blocks.%d.downsamplers.0.conv.weight", g_load_wp, idx);
        db->down_w = upload_st(st, buf);
        snprintf(buf, sizeof(buf), "%sdown_blocks.%d.downsamplers.0.conv.bias", g_load_wp, idx);
        db->down_b = upload_st(st, buf);
    } else { db->down_w = 0; db->down_b = 0; }
}

typedef struct {
    pu_resblock mid_res0, mid_res1;
    pu_transformer mid_xfm;
} pu_mid_block;

static void load_mid_block(st_context *st, pu_mid_block *m, int cross_dim,
                            int has_dino, int has_ma, int has_mda, int has_ra,
                            int n_pbr, int n_gen) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%smid_block.resnets.0", g_load_wp);
    load_resblock(st, &m->mid_res0, buf, 1280, 1280);
    snprintf(buf, sizeof(buf), "%smid_block.attentions.0", g_load_wp);
    load_transformer(st, &m->mid_xfm, buf,
                      1280, /*heads*/ 20, cross_dim, 1, has_dino, has_ma, has_mda, has_ra, n_pbr, n_gen);
    snprintf(buf, sizeof(buf), "%smid_block.resnets.1", g_load_wp);
    load_resblock(st, &m->mid_res1, buf, 1280, 1280);
}

static void load_up_block(st_context *st, pu_up_block *ub, int idx,
                           int c_in, int c_out, int prev_out,
                           int num_heads, int has_xfm, int has_up,
                           int cross_dim, int has_dino,
                           int has_ma, int has_mda, int has_ra, int n_pbr, int n_gen) {
    char buf[256];
    ub->c_out = c_out; ub->has_xfm = has_xfm;
    ub->num_heads = num_heads;
    /* res_in_channels and skip_channels per diffusers CrossAttnUpBlock2D ctor */
    for (int i = 0; i < 3; i++) {
        ub->skip_ch[i] = (i == 2) ? c_in : c_out;
        ub->res_in[i]  = (i == 0) ? prev_out : c_out;
    }
    for (int i = 0; i < 3; i++) {
        int rin = ub->res_in[i] + ub->skip_ch[i];
        snprintf(buf, sizeof(buf), "%sup_blocks.%d.resnets.%d", g_load_wp, idx, i);
        load_resblock(st, &ub->res[i], buf, rin, c_out);
    }
    if (has_xfm) {
        for (int i = 0; i < 3; i++) {
            snprintf(buf, sizeof(buf), "%sup_blocks.%d.attentions.%d", g_load_wp, idx, i);
            load_transformer(st, &ub->xfm[i], buf, c_out, num_heads, cross_dim, 1, has_dino, has_ma, has_mda, has_ra, n_pbr, n_gen);
        }
    }
    if (has_up) {
        snprintf(buf, sizeof(buf), "%sup_blocks.%d.upsamplers.0.conv.weight", g_load_wp, idx);
        ub->up_w = upload_st(st, buf);
        snprintf(buf, sizeof(buf), "%sup_blocks.%d.upsamplers.0.conv.bias", g_load_wp, idx);
        ub->up_b = upload_st(st, buf);
    } else { ub->up_w = 0; ub->up_b = 0; }
}

/* ===== Skip stack ========================================================= */
typedef struct {
    CUdeviceptr buf;
    int channels;
    int H, W;
} pu_skip;

typedef struct {
    pu_skip s[16];
    int top;
    int B;   /* batch shared across all entries (set per stack) */
} pu_skip_stack;

static void skip_push_copy(pu_skip_stack *ss, CUdeviceptr src, int C, int H, int W) {
    size_t sz = (size_t)ss->B * C * H * W * sizeof(float);
    CUdeviceptr d; cuMemAlloc(&d, sz);
    cuMemcpyDtoD(d, src, sz);
    ss->s[ss->top++] = (pu_skip){ d, C, H, W };
}
static pu_skip skip_pop(pu_skip_stack *ss) {
    return ss->s[--ss->top];
}

/* ===== Run helpers (B=1 only) ============================================= */

typedef struct {
    /* Activation ping-pong */
    CUdeviceptr d_a, d_b;            /* big enough for max C*H*W in flow */
    /* ResBlock scratch */
    CUdeviceptr d_t1, d_t2;
    CUdeviceptr d_temb_act;          /* [B*1280] */
    CUdeviceptr d_temb_proj;         /* [B*1280] (oversized) */
    /* Transformer scratch (sized for largest level: 320 ch @ 64x64). */
    pu_xfm_scratch X;
} pu_workspace;

static void run_xfm_inplace(const pu_kernels *kk, const pu_transformer *T,
                              CUdeviceptr d_x, CUdeviceptr d_text,
                              CUdeviceptr d_dino, int M_dino,
                              int B, int H, int W, int M_text,
                              const pu_xfm_scratch *S) {
    run_transformer(kk, T, d_x, d_text, d_dino, M_dino, B, H, W, M_text, S);
}

/* d_in -> d_out via the down_block. push 2-3 skips. Returns new H/W in *pH/*pW. */
static void run_down_block(const pu_kernels *kk, const pu_down_block *db,
                             CUdeviceptr d_in, CUdeviceptr d_out,
                             CUdeviceptr d_temb, CUdeviceptr d_text,
                             CUdeviceptr d_dino, int M_dino,
                             int B, int *pH, int *pW, int M_text,
                             pu_workspace *ws, pu_skip_stack *ss) {
    int H = *pH, W = *pW;
    int Cout = db->c_out;
    /* res[0]: c_in -> c_out */
    run_resblock(kk, &db->res[0], d_in, d_out,
                  ws->d_t1, ws->d_t2, d_temb, ws->d_temb_act, ws->d_temb_proj,
                  B, H, W, 32);
    if (db->has_xfm) run_xfm_inplace(kk, &db->xfm[0], d_out, d_text, d_dino, M_dino, B, H, W, M_text, &ws->X);
    skip_push_copy(ss, d_out, Cout, H, W);

    /* res[1]: c_out -> c_out, in=d_out -> out=d_in (reuse) */
    run_resblock(kk, &db->res[1], d_out, d_in,
                  ws->d_t1, ws->d_t2, d_temb, ws->d_temb_act, ws->d_temb_proj,
                  B, H, W, 32);
    if (db->has_xfm) run_xfm_inplace(kk, &db->xfm[1], d_in, d_text, d_dino, M_dino, B, H, W, M_text, &ws->X);
    skip_push_copy(ss, d_in, Cout, H, W);

    /* downsampler if any: writes into d_out at H/2, W/2 */
    if (db->down_w) {
        size_t in_str  = (size_t)Cout * H * W * sizeof(float);
        size_t out_str = (size_t)Cout * (H/2) * (W/2) * sizeof(float);
        for (int b = 0; b < B; b++) {
            CUdeviceptr ib = d_in  + (CUdeviceptr)b * in_str;
            CUdeviceptr ob = d_out + (CUdeviceptr)b * out_str;
            k_conv_stride2(kk, ob, ib, db->down_w, db->down_b, Cout, H, W, Cout);
        }
        H >>= 1; W >>= 1;
        skip_push_copy(ss, d_out, Cout, H, W);
        /* Make d_in the "current" buffer for next block */
        cuMemcpyDtoD(d_in, d_out, (size_t)B * Cout * H * W * sizeof(float));
    }
    /* After this fn, the "current" tensor lives in d_in. */
    *pH = H; *pW = W;
}

static void run_mid_block(const pu_kernels *kk, const pu_mid_block *m,
                            CUdeviceptr d_x, CUdeviceptr d_tmp,
                            CUdeviceptr d_temb, CUdeviceptr d_text,
                            CUdeviceptr d_dino, int M_dino,
                            int B, int H, int W, int M_text,
                            pu_workspace *ws) {
    /* res0 in d_x -> d_tmp */
    run_resblock(kk, &m->mid_res0, d_x, d_tmp,
                  ws->d_t1, ws->d_t2, d_temb, ws->d_temb_act, ws->d_temb_proj,
                  B, H, W, 32);
    /* xfm in d_tmp (in-place) */
    run_xfm_inplace(kk, &m->mid_xfm, d_tmp, d_text, d_dino, M_dino, B, H, W, M_text, &ws->X);
    /* res1 d_tmp -> d_x */
    run_resblock(kk, &m->mid_res1, d_tmp, d_x,
                  ws->d_t1, ws->d_t2, d_temb, ws->d_temb_act, ws->d_temb_proj,
                  B, H, W, 32);
}

/* d_in/d_out alternate. Activation enters in d_in (Cin@H,W), exits in d_in
 * after upsample (or d_in after last resnet if no upsample). */
static void run_up_block(const pu_kernels *kk, const pu_up_block *ub,
                           CUdeviceptr d_in, CUdeviceptr d_out,
                           CUdeviceptr d_concat,
                           CUdeviceptr d_temb, CUdeviceptr d_text,
                           CUdeviceptr d_dino, int M_dino,
                           int B, int *pH, int *pW, int M_text,
                           pu_workspace *ws, pu_skip_stack *ss) {
    int H = *pH, W = *pW;
    int Cout = ub->c_out;
    int sp = H * W;
    for (int i = 0; i < 3; i++) {
        /* current tensor in d_in [B, res_in, H, W]; pop skip [B, skip_ch, H, W] */
        pu_skip sk = skip_pop(ss);
        /* per-batch concat: [res_in + skip_ch, H, W] -> d_concat */
        size_t in_str   = (size_t)ub->res_in[i]  * sp * sizeof(float);
        size_t skip_str = (size_t)ub->skip_ch[i] * sp * sizeof(float);
        size_t cat_str  = in_str + skip_str;
        size_t out_str  = (size_t)Cout * sp * sizeof(float);
        for (int b = 0; b < B; b++) {
            CUdeviceptr ib = d_in    + (CUdeviceptr)b * in_str;
            CUdeviceptr sb = sk.buf  + (CUdeviceptr)b * skip_str;
            CUdeviceptr cb = d_concat+ (CUdeviceptr)b * cat_str;
            k_concat_chan(kk, cb, ib, sb,
                           ub->res_in[i], ub->skip_ch[i], sp);
        }
        cuMemFree(sk.buf);
        /* resnet (res_in+skip_ch -> Cout): d_concat -> d_out */
        run_resblock(kk, &ub->res[i], d_concat, d_out,
                      ws->d_t1, ws->d_t2, d_temb, ws->d_temb_act, ws->d_temb_proj,
                      B, H, W, 32);
        /* xfm in d_out (in-place) */
        if (ub->has_xfm)
            run_xfm_inplace(kk, &ub->xfm[i], d_out, d_text, d_dino, M_dino, B, H, W, M_text, &ws->X);
        /* swap: next iter's "current" is in d_in */
        cuMemcpyDtoD(d_in, d_out, (size_t)B * out_str);
    }
    /* upsample if any: nearest 2x then 3x3 conv. d_in -> d_out -> d_in */
    if (ub->up_w) {
        size_t in_str  = (size_t)Cout * H * W * sizeof(float);
        size_t up_str  = (size_t)Cout * (H*2) * (W*2) * sizeof(float);
        for (int b = 0; b < B; b++) {
            CUdeviceptr ib = d_in  + (CUdeviceptr)b * in_str;
            CUdeviceptr ob = d_out + (CUdeviceptr)b * up_str;
            k_upsample2x(kk, ob, ib, Cout, H, W);
        }
        H <<= 1; W <<= 1; sp = H * W;
        size_t up_str2 = (size_t)Cout * sp * sizeof(float);
        for (int b = 0; b < B; b++) {
            CUdeviceptr ob = d_out + (CUdeviceptr)b * up_str2;
            CUdeviceptr ib = d_in  + (CUdeviceptr)b * up_str2;
            k_conv(kk, ib, ob, ub->up_w, ub->up_b,
                    Cout, H, W, Cout, 3, 3, 1);
        }
    }
    *pH = H; *pW = W;
}

/* ===== PoseRoPE host-side cos/sin builder ================================= */
/* Mirrors RotaryEmbedding.get_3d_rotary_pos_embed in attn_processor.py.
 *  - dim_xy = head_dim/8*3, dim_z = head_dim/8*2 (head_dim must be /8).
 *  - For each axis: freqs[k] = 1/theta^(2k/dim); cos/sin tables [voxel_res, dim]
 *    where cos[p,2k]==cos[p,2k+1]==cos(p*freqs[k]) (repeat_interleave).
 *  - Per-token cos = cat(xy_cos[x_idx], xy_cos[y_idx], z_cos[z_idx]) -> [head_dim].
 *
 * Reads voxel_indices_<Np>.npy ([1, Np, 3] int64), tiles over Bp (= n_pbr) so
 * cos/sin shape is [Bp*Np, head_dim]. Pushes one entry per spatial level into
 * g_rope_levels keyed by N = Np / n_gen. */
static int build_rope_level_from_voxels(const int64_t *vox, int Np, int n_pbr,
                                          int n_gen, int head_dim, int voxel_res) {
    if (g_rope_n_levels >= (int)(sizeof(g_rope_levels)/sizeof(g_rope_levels[0]))) return -1;
    int Bp = n_pbr;          /* B_input=1 in our test */
    int N  = Np / n_gen;
    int dim_xy = head_dim / 8 * 3;
    int dim_z  = head_dim / 8 * 2;
    int half_xy = dim_xy / 2;
    int half_z  = dim_z  / 2;
    float theta = 10000.f;

    /* Per-axis freqs and (per-pos) cos/sin tables on host. */
    float *xy_cos = (float *)malloc((size_t)voxel_res * dim_xy * sizeof(float));
    float *xy_sin = (float *)malloc((size_t)voxel_res * dim_xy * sizeof(float));
    float *z_cos  = (float *)malloc((size_t)voxel_res * dim_z  * sizeof(float));
    float *z_sin  = (float *)malloc((size_t)voxel_res * dim_z  * sizeof(float));
    for (int p = 0; p < voxel_res; p++) {
        for (int k = 0; k < half_xy; k++) {
            float fk = 1.f / powf(theta, (float)(2 * k) / (float)dim_xy);
            float ang = (float)p * fk;
            float c = cosf(ang), s = sinf(ang);
            xy_cos[p * dim_xy + 2*k] = xy_cos[p * dim_xy + 2*k + 1] = c;
            xy_sin[p * dim_xy + 2*k] = xy_sin[p * dim_xy + 2*k + 1] = s;
        }
        for (int k = 0; k < half_z; k++) {
            float fk = 1.f / powf(theta, (float)(2 * k) / (float)dim_z);
            float ang = (float)p * fk;
            float c = cosf(ang), s = sinf(ang);
            z_cos[p * dim_z + 2*k] = z_cos[p * dim_z + 2*k + 1] = c;
            z_sin[p * dim_z + 2*k] = z_sin[p * dim_z + 2*k + 1] = s;
        }
    }

    /* Per-token cos/sin [Bp*Np, head_dim] (Bp tiles over n_pbr; the source
     * voxel_indices has only one batch in our wrapper test). */
    size_t total = (size_t)Bp * Np * head_dim;
    float *cos_h = (float *)malloc(total * sizeof(float));
    float *sin_h = (float *)malloc(total * sizeof(float));
    for (int b = 0; b < Bp; b++) {
        for (int t = 0; t < Np; t++) {
            int64_t ix = vox[(size_t)t * 3 + 0];
            int64_t iy = vox[(size_t)t * 3 + 1];
            int64_t iz = vox[(size_t)t * 3 + 2];
            if (ix < 0 || ix >= voxel_res || iy < 0 || iy >= voxel_res ||
                iz < 0 || iz >= voxel_res) {
                fprintf(stderr, "ERROR: voxel idx out of range: (%lld,%lld,%lld) res=%d\n",
                        (long long)ix, (long long)iy, (long long)iz, voxel_res);
                return -1;
            }
            float *cdst = cos_h + ((size_t)b * Np + t) * head_dim;
            float *sdst = sin_h + ((size_t)b * Np + t) * head_dim;
            memcpy(cdst,                  xy_cos + ix * dim_xy, dim_xy * sizeof(float));
            memcpy(cdst + dim_xy,         xy_cos + iy * dim_xy, dim_xy * sizeof(float));
            memcpy(cdst + dim_xy*2,       z_cos  + iz * dim_z,  dim_z  * sizeof(float));
            memcpy(sdst,                  xy_sin + ix * dim_xy, dim_xy * sizeof(float));
            memcpy(sdst + dim_xy,         xy_sin + iy * dim_xy, dim_xy * sizeof(float));
            memcpy(sdst + dim_xy*2,       z_sin  + iz * dim_z,  dim_z  * sizeof(float));
        }
    }

    pu_rope_level *L = &g_rope_levels[g_rope_n_levels++];
    L->N = N;
    L->Np = Np;
    cuMemAlloc(&L->d_cos, total * sizeof(float));
    cuMemAlloc(&L->d_sin, total * sizeof(float));
    cuMemcpyHtoD(L->d_cos, cos_h, total * sizeof(float));
    cuMemcpyHtoD(L->d_sin, sin_h, total * sizeof(float));

    free(xy_cos); free(xy_sin); free(z_cos); free(z_sin);
    free(cos_h); free(sin_h);
    g_rope_head_dim = head_dim;
    g_rope_Bp = Bp;
    return 0;
}

/* ===== main =============================================================== */


#endif  /* CUDA_PAINT_UNET_RUNNER_H_ */
