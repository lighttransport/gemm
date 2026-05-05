/*
 * test_paint_unet.c - Native CUDA SD-2.1 paint UNet (Phase 3 skeleton).
 *
 * Loads stock paint UNet weights (paint_unet_stock.safetensors produced by
 * ref/hy3d/export_paint_unet_safetensors.py), runs forward pieces, and
 * diffs them against the diffusers reference dump from
 * ref/hy3d/dump_paint_unet.py.
 *
 * Phase 3 incremental: at each iteration we add another stage and validate
 * one intermediate. Current stages live behind --stage <name>:
 *   time_emb : timestep_embedding + time MLP -> [B, 1280]
 *   conv_in  : conv_in 12->320 -> [B, 320, 64, 64]
 *
 * Usage:
 *   ./test_paint_unet --stage conv_in \\
 *       /mnt/disk01/.../unet/paint_unet_stock.safetensors \\
 *       /tmp/hy3d_paint_unet_ref/
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "cuda_paint_unet_kernels.h"

#define SAFETENSORS_IMPLEMENTATION
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
} pu_kernels;

static void k_timestep_embed(const pu_kernels *kk, CUdeviceptr out,
                              CUdeviceptr ts, int B, int dim) {
    void *args[] = { &out, &ts, &B, &dim };
    int tx = 64;
    cuLaunchKernel(kk->f_tse, (unsigned)B, (unsigned)((dim + tx - 1) / tx), 1,
                    tx, 1, 1, 0, 0, args, NULL);
}

static void k_linear(const pu_kernels *kk, CUdeviceptr y, CUdeviceptr x,
                      CUdeviceptr W, CUdeviceptr b, int M, int K, int N) {
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

static void k_conv(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                    CUdeviceptr W, CUdeviceptr b,
                    int ci, int h, int w, int co, int kh, int kw, int pad) {
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

static void k_mha(const pu_kernels *kk, CUdeviceptr out, CUdeviceptr Q,
                   CUdeviceptr K, CUdeviceptr V,
                   int B, int N, int M, int heads, int head_dim) {
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
    /* h = silu(norm1(x))  [c_in, H, W] -> d_t1 */
    k_groupnorm(kk, d_t1, d_x, r->norm1_g, r->norm1_b,
                 r->c_in, sp, num_groups, 1);
    /* h = conv1(h)  -> d_t2 [c_out, H, W] */
    k_conv(kk, d_t2, d_t1, r->conv1_w, r->conv1_b,
            r->c_in, H, W, r->c_out, 3, 3, 1);
    /* time embedding: silu(d_temb) -> linear(1280, c_out) */
    cuMemcpyDtoD(d_temb_act, d_temb, B * 1280 * sizeof(float));
    k_silu(kk, d_temb_act, B * 1280);
    k_linear(kk, d_temb_proj, d_temb_act, r->temb_w, r->temb_b, B, 1280, r->c_out);
    /* h = h + t.broadcast over spatial */
    /* Single-batch shortcut: kernel handles one [c_out, H, W] map at a time. */
    for (int b = 0; b < B; b++) {
        CUdeviceptr h_b = d_t2        + (CUdeviceptr)b * r->c_out * sp * sizeof(float);
        CUdeviceptr t_b = d_temb_proj + (CUdeviceptr)b * r->c_out      * sizeof(float);
        k_add_chan(kk, h_b, h_b, t_b, r->c_out, sp);
    }
    /* h = silu(norm2(h)) -> d_t1 */
    k_groupnorm(kk, d_t1, d_t2, r->norm2_g, r->norm2_b,
                 r->c_out, sp, num_groups, 1);
    /* h = conv2(h) -> d_t2 [c_out, H, W] */
    k_conv(kk, d_t2, d_t1, r->conv2_w, r->conv2_b,
            r->c_out, H, W, r->c_out, 3, 3, 1);
    /* skip path */
    if (r->skip_w) {
        k_conv(kk, d_t1, d_x, r->skip_w, r->skip_b,
                r->c_in, H, W, r->c_out, 1, 1, 0);
        k_add(kk, d_out, d_t2, d_t1, r->c_out * sp * B);
    } else {
        k_add(kk, d_out, d_t2, d_x, r->c_out * sp * B);
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
} pu_attention;

typedef struct {
    CUdeviceptr norm1_g, norm1_b;
    pu_attention attn1;                     /* self-attn */
    CUdeviceptr norm2_g, norm2_b;
    pu_attention attn2;                     /* cross-attn */
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
} pu_transformer;

static void load_attention(st_context *st, pu_attention *a, const char *prefix) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s.to_q.weight", prefix); a->to_q_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.to_k.weight", prefix); a->to_k_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.to_v.weight", prefix); a->to_v_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.to_out.0.weight", prefix); a->to_out_w = upload_st(st, buf);
    snprintf(buf, sizeof(buf), "%s.to_out.0.bias",   prefix); a->to_out_b = upload_st(st, buf);
}

static void load_transformer(st_context *st, pu_transformer *T,
                               const char *prefix, int channels,
                               int num_heads, int cross_dim, int num_blocks) {
    char buf[256];
    T->channels  = channels;
    T->num_heads = num_heads;
    T->head_dim  = channels / num_heads;
    T->cross_dim = cross_dim;
    T->ff_inner  = channels * 4;
    T->num_blocks = num_blocks;

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

static void run_transformer(const pu_kernels *kk, const pu_transformer *T,
                              CUdeviceptr d_x, CUdeviceptr d_text,
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
        /* --- self-attn --- */
        k_layernorm(kk, S->d_norm, S->d_nc, bb->norm1_g, bb->norm1_b, B * N, C);
        run_attention(kk, &bb->attn1,
                       S->d_norm, S->d_norm, S->d_nc_b,
                       S->d_q, S->d_k, S->d_v, S->d_attn,
                       B, N, N, C, C, T->num_heads, T->head_dim);
        k_add(kk, S->d_nc, S->d_nc, S->d_nc_b, B * N * C);
        /* --- cross-attn --- */
        k_layernorm(kk, S->d_norm, S->d_nc, bb->norm2_g, bb->norm2_b, B * N, C);
        run_attention(kk, &bb->attn2,
                       S->d_norm, d_text, S->d_nc_b,
                       S->d_q, S->d_k, S->d_v, S->d_attn,
                       B, N, M_text, C, T->cross_dim,
                       T->num_heads, T->head_dim);
        k_add(kk, S->d_nc, S->d_nc, S->d_nc_b, B * N * C);
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

static void load_down_block(st_context *st, pu_down_block *db, int idx,
                              int c_in, int c_out, int num_heads,
                              int has_xfm, int has_down, int cross_dim) {
    char buf[256];
    db->c_in = c_in; db->c_out = c_out; db->has_xfm = has_xfm;
    db->num_heads = num_heads;
    snprintf(buf, sizeof(buf), "down_blocks.%d.resnets.0", idx);
    load_resblock(st, &db->res[0], buf, c_in, c_out);
    snprintf(buf, sizeof(buf), "down_blocks.%d.resnets.1", idx);
    load_resblock(st, &db->res[1], buf, c_out, c_out);
    if (has_xfm) {
        snprintf(buf, sizeof(buf), "down_blocks.%d.attentions.0", idx);
        load_transformer(st, &db->xfm[0], buf, c_out, num_heads, cross_dim, 1);
        snprintf(buf, sizeof(buf), "down_blocks.%d.attentions.1", idx);
        load_transformer(st, &db->xfm[1], buf, c_out, num_heads, cross_dim, 1);
    }
    if (has_down) {
        snprintf(buf, sizeof(buf), "down_blocks.%d.downsamplers.0.conv.weight", idx);
        db->down_w = upload_st(st, buf);
        snprintf(buf, sizeof(buf), "down_blocks.%d.downsamplers.0.conv.bias", idx);
        db->down_b = upload_st(st, buf);
    } else { db->down_w = 0; db->down_b = 0; }
}

typedef struct {
    pu_resblock mid_res0, mid_res1;
    pu_transformer mid_xfm;
} pu_mid_block;

static void load_mid_block(st_context *st, pu_mid_block *m, int cross_dim) {
    load_resblock(st, &m->mid_res0, "mid_block.resnets.0", 1280, 1280);
    load_transformer(st, &m->mid_xfm, "mid_block.attentions.0",
                      1280, /*heads*/ 20, cross_dim, 1);
    load_resblock(st, &m->mid_res1, "mid_block.resnets.1", 1280, 1280);
}

static void load_up_block(st_context *st, pu_up_block *ub, int idx,
                           int c_in, int c_out, int prev_out,
                           int num_heads, int has_xfm, int has_up,
                           int cross_dim) {
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
        snprintf(buf, sizeof(buf), "up_blocks.%d.resnets.%d", idx, i);
        load_resblock(st, &ub->res[i], buf, rin, c_out);
    }
    if (has_xfm) {
        for (int i = 0; i < 3; i++) {
            snprintf(buf, sizeof(buf), "up_blocks.%d.attentions.%d", idx, i);
            load_transformer(st, &ub->xfm[i], buf, c_out, num_heads, cross_dim, 1);
        }
    }
    if (has_up) {
        snprintf(buf, sizeof(buf), "up_blocks.%d.upsamplers.0.conv.weight", idx);
        ub->up_w = upload_st(st, buf);
        snprintf(buf, sizeof(buf), "up_blocks.%d.upsamplers.0.conv.bias", idx);
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
} pu_skip_stack;

static void skip_push_copy(pu_skip_stack *ss, CUdeviceptr src, int C, int H, int W) {
    size_t sz = (size_t)C * H * W * sizeof(float);
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
                              int B, int H, int W, int M_text,
                              const pu_xfm_scratch *S) {
    run_transformer(kk, T, d_x, d_text, B, H, W, M_text, S);
}

/* d_in -> d_out via the down_block. push 2-3 skips. Returns new H/W in *pH/*pW. */
static void run_down_block(const pu_kernels *kk, const pu_down_block *db,
                             CUdeviceptr d_in, CUdeviceptr d_out,
                             CUdeviceptr d_temb, CUdeviceptr d_text,
                             int B, int *pH, int *pW, int M_text,
                             pu_workspace *ws, pu_skip_stack *ss) {
    int H = *pH, W = *pW;
    int Cout = db->c_out;
    /* res[0]: c_in -> c_out */
    run_resblock(kk, &db->res[0], d_in, d_out,
                  ws->d_t1, ws->d_t2, d_temb, ws->d_temb_act, ws->d_temb_proj,
                  B, H, W, 32);
    if (db->has_xfm) run_xfm_inplace(kk, &db->xfm[0], d_out, d_text, B, H, W, M_text, &ws->X);
    skip_push_copy(ss, d_out, Cout, H, W);

    /* res[1]: c_out -> c_out, in=d_out -> out=d_in (reuse) */
    run_resblock(kk, &db->res[1], d_out, d_in,
                  ws->d_t1, ws->d_t2, d_temb, ws->d_temb_act, ws->d_temb_proj,
                  B, H, W, 32);
    if (db->has_xfm) run_xfm_inplace(kk, &db->xfm[1], d_in, d_text, B, H, W, M_text, &ws->X);
    skip_push_copy(ss, d_in, Cout, H, W);

    /* downsampler if any: writes into d_out at H/2, W/2 */
    if (db->down_w) {
        k_conv_stride2(kk, d_out, d_in, db->down_w, db->down_b, Cout, H, W, Cout);
        H >>= 1; W >>= 1;
        skip_push_copy(ss, d_out, Cout, H, W);
        /* Make d_in the "current" buffer for next block */
        cuMemcpyDtoD(d_in, d_out, (size_t)Cout * H * W * sizeof(float));
    }
    /* After this fn, the "current" tensor lives in d_in. */
    *pH = H; *pW = W;
}

static void run_mid_block(const pu_kernels *kk, const pu_mid_block *m,
                            CUdeviceptr d_x, CUdeviceptr d_tmp,
                            CUdeviceptr d_temb, CUdeviceptr d_text,
                            int B, int H, int W, int M_text,
                            pu_workspace *ws) {
    /* res0 in d_x -> d_tmp */
    run_resblock(kk, &m->mid_res0, d_x, d_tmp,
                  ws->d_t1, ws->d_t2, d_temb, ws->d_temb_act, ws->d_temb_proj,
                  B, H, W, 32);
    /* xfm in d_tmp (in-place) */
    run_xfm_inplace(kk, &m->mid_xfm, d_tmp, d_text, B, H, W, M_text, &ws->X);
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
                           int B, int *pH, int *pW, int M_text,
                           pu_workspace *ws, pu_skip_stack *ss) {
    int H = *pH, W = *pW;
    int Cout = ub->c_out;
    int sp = H * W;
    for (int i = 0; i < 3; i++) {
        /* current tensor in d_in [res_in, H, W]; pop skip [skip_ch, H, W] */
        pu_skip sk = skip_pop(ss);
        /* concat: [res_in + skip_ch, H, W] -> d_concat */
        k_concat_chan(kk, d_concat, d_in, sk.buf,
                       ub->res_in[i], ub->skip_ch[i], sp);
        cuMemFree(sk.buf);
        /* resnet (res_in+skip_ch -> Cout): d_concat -> d_out */
        run_resblock(kk, &ub->res[i], d_concat, d_out,
                      ws->d_t1, ws->d_t2, d_temb, ws->d_temb_act, ws->d_temb_proj,
                      B, H, W, 32);
        /* xfm in d_out (in-place) */
        if (ub->has_xfm)
            run_xfm_inplace(kk, &ub->xfm[i], d_out, d_text, B, H, W, M_text, &ws->X);
        /* swap: next iter's "current" is in d_in */
        cuMemcpyDtoD(d_in, d_out, (size_t)Cout * sp * sizeof(float));
    }
    /* upsample if any: nearest 2x then 3x3 conv. d_in -> d_out -> d_in */
    if (ub->up_w) {
        k_upsample2x(kk, d_out, d_in, Cout, H, W);
        H <<= 1; W <<= 1; sp = H * W;
        k_conv(kk, d_in, d_out, ub->up_w, ub->up_b,
                Cout, H, W, Cout, 3, 3, 1);
    }
    *pH = H; *pW = W;
}

/* ===== main =============================================================== */

int main(int argc, char **argv) {
    const char *stage = "conv_in";
    int argi = 1;
    if (argi < argc && !strcmp(argv[argi], "--stage")) {
        stage = argv[argi+1]; argi += 2;
    }
    if (argc - argi < 2) {
        fprintf(stderr,
            "Usage: %s [--stage time_emb|conv_in|db0_res0|db0_attn0|out] <unet.safetensors> <ref_dir>\n",
            argv[0]);
        return 1;
    }
    const char *st_path = argv[argi];
    const char *ref_dir = argv[argi+1];

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1;
    }
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    pu_kernels kk = {0};
    if (cu_compile_kernels(&kk.mod, dev, cuda_paint_unet_kernels_src,
                             "hy3d_paint_unet", 1, "HY3D-PAINT-UNET") < 0)
        return 1;
    cuModuleGetFunction(&kk.f_tse,  kk.mod, "unet_timestep_embed_f32");
    cuModuleGetFunction(&kk.f_lin,  kk.mod, "unet_linear_f32");
    cuModuleGetFunction(&kk.f_silu, kk.mod, "unet_silu_f32");
    cuModuleGetFunction(&kk.f_conv, kk.mod, "unet_conv2d_f32");
    cuModuleGetFunction(&kk.f_gn,   kk.mod, "unet_groupnorm_f32");
    cuModuleGetFunction(&kk.f_addc, kk.mod, "unet_add_chan_f32");
    cuModuleGetFunction(&kk.f_add,  kk.mod, "unet_add_f32");
    cuModuleGetFunction(&kk.f_ln,     kk.mod, "unet_layernorm_f32");
    cuModuleGetFunction(&kk.f_chw_nc, kk.mod, "unet_chw_to_nc_f32");
    cuModuleGetFunction(&kk.f_nc_chw, kk.mod, "unet_nc_to_chw_f32");
    cuModuleGetFunction(&kk.f_mha,    kk.mod, "unet_mha_f32");
    cuModuleGetFunction(&kk.f_geglu,  kk.mod, "unet_geglu_f32");
    cuModuleGetFunction(&kk.f_conv_s2, kk.mod, "unet_conv2d_stride2_f32");
    cuModuleGetFunction(&kk.f_up2x,    kk.mod, "unet_upsample2x_f32");
    cuModuleGetFunction(&kk.f_concat,  kk.mod, "unet_concat_chan_f32");

    st_context *st = safetensors_open(st_path);
    if (!st) { fprintf(stderr, "ERROR: cannot open %s\n", st_path); return 1; }
    fprintf(stderr, "loaded safetensors %s\n", st_path);

    /* Load reference inputs from the dump dir */
    char path[512];
    int nd; uint64_t shape[8]; size_t n; char dt[8];
    snprintf(path, sizeof(path), "%s/ref_timestep.npy", ref_dir);
    int64_t *ts = (int64_t *)read_npy(path, &nd, shape, &n, dt);
    if (!ts) return 1;
    int B = (int)shape[0];
    fprintf(stderr, "B=%d, timestep[0]=%lld\n", B, (long long)ts[0]);

    if (!strcmp(stage, "time_emb")) {
        /* timestep -> sinusoidal[320] -> linear(320,1280) silu linear(1280,1280)
         * Output [B,1280] vs ref_time_emb.npy */
        CUdeviceptr d_ts; cuMemAlloc(&d_ts, B * sizeof(int64_t));
        cuMemcpyHtoD(d_ts, ts, B * sizeof(int64_t));
        CUdeviceptr d_emb;  cuMemAlloc(&d_emb, B * 320 * sizeof(float));
        CUdeviceptr d_h1;   cuMemAlloc(&d_h1,  B * 1280 * sizeof(float));
        CUdeviceptr d_h2;   cuMemAlloc(&d_h2,  B * 1280 * sizeof(float));
        CUdeviceptr l1_w = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b = upload_st(st, "time_embedding.linear_2.bias");

        k_timestep_embed(&kk, d_emb, d_ts, B, 320);
        k_linear(&kk, d_h1, d_emb, l1_w, l1_b, B, 320, 1280);
        k_silu(&kk, d_h1, B * 1280);
        k_linear(&kk, d_h2, d_h1, l2_w, l2_b, B, 1280, 1280);
        cuCtxSynchronize();

        float *cu = (float *)malloc(B * 1280 * sizeof(float));
        cuMemcpyDtoH(cu, d_h2, B * 1280 * sizeof(float));
        snprintf(path, sizeof(path), "%s/ref_time_emb.npy", ref_dir);
        diff_against(cu, path, (size_t)B * 1280, 1e-3f);
        free(cu);
    } else if (!strcmp(stage, "conv_in")) {
        /* Read sample, run conv_in 12->320, compare to ref_conv_in.npy */
        snprintf(path, sizeof(path), "%s/ref_sample.npy", ref_dir);
        float *sample = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!sample) return 1;
        int IC = (int)shape[1], H = (int)shape[2], W = (int)shape[3];
        if (IC != 12) {
            fprintf(stderr, "ERROR: expected sample channels=12, got %d\n", IC);
            return 1;
        }
        fprintf(stderr, "sample [%d, %d, %d, %d]\n", B, IC, H, W);
        size_t in_n  = (size_t)B * IC * H * W;
        size_t out_n = (size_t)B * 320 * H * W;
        CUdeviceptr d_in;  cuMemAlloc(&d_in,  in_n  * sizeof(float));
        CUdeviceptr d_out; cuMemAlloc(&d_out, out_n * sizeof(float));
        cuMemcpyHtoD(d_in, sample, in_n * sizeof(float));
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");
        /* Batch loop: kernel handles one sample (CHW) at a time. */
        for (int b = 0; b < B; b++) {
            CUdeviceptr in_b  = d_in  + (CUdeviceptr)b * IC  * H * W * sizeof(float);
            CUdeviceptr out_b = d_out + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            k_conv(&kk, out_b, in_b, cw, cb, IC, H, W, 320, 3, 3, 1);
        }
        cuCtxSynchronize();

        float *cu = (float *)malloc(out_n * sizeof(float));
        cuMemcpyDtoH(cu, d_out, out_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/ref_conv_in.npy", ref_dir);
        diff_against(cu, path, out_n, 1e-3f);
        free(cu); free(sample);
    } else if (!strcmp(stage, "db0_res0")) {
        /* Pipeline up to and including down_blocks[0].resnets[0]:
         *   time_emb      [B, 1280]
         *   conv_in(x)    [B, 320, 64, 64]
         *   resblock(.,t) [B, 320, 64, 64]   vs ref_db0_res0.npy
         */
        snprintf(path, sizeof(path), "%s/ref_sample.npy", ref_dir);
        float *sample = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!sample) return 1;
        int IC = (int)shape[1], H = (int)shape[2], W = (int)shape[3];
        if (IC != 12) { fprintf(stderr, "ERROR: IC!=12\n"); return 1; }

        /* --- time embedding --- */
        CUdeviceptr d_ts;   cuMemAlloc(&d_ts, B * sizeof(int64_t));
        cuMemcpyHtoD(d_ts, ts, B * sizeof(int64_t));
        CUdeviceptr d_temb_in; cuMemAlloc(&d_temb_in, B * 320 * sizeof(float));
        CUdeviceptr d_temb_h1; cuMemAlloc(&d_temb_h1, B * 1280 * sizeof(float));
        CUdeviceptr d_temb;    cuMemAlloc(&d_temb,    B * 1280 * sizeof(float));
        CUdeviceptr l1_w = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b = upload_st(st, "time_embedding.linear_2.bias");
        k_timestep_embed(&kk, d_temb_in, d_ts, B, 320);
        k_linear(&kk, d_temb_h1, d_temb_in, l1_w, l1_b, B, 320, 1280);
        k_silu(&kk, d_temb_h1, B * 1280);
        k_linear(&kk, d_temb, d_temb_h1, l2_w, l2_b, B, 1280, 1280);

        /* --- conv_in --- */
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");
        size_t in_n  = (size_t)B * IC * H * W;
        size_t hw_n  = (size_t)B * 320 * H * W;
        CUdeviceptr d_in;  cuMemAlloc(&d_in,  in_n * sizeof(float));
        CUdeviceptr d_x;   cuMemAlloc(&d_x,   hw_n * sizeof(float));
        cuMemcpyHtoD(d_in, sample, in_n * sizeof(float));
        for (int b = 0; b < B; b++) {
            CUdeviceptr in_b = d_in + (CUdeviceptr)b * IC  * H * W * sizeof(float);
            CUdeviceptr x_b  = d_x  + (CUdeviceptr)b * 320 * H * W * sizeof(float);
            k_conv(&kk, x_b, in_b, cw, cb, IC, H, W, 320, 3, 3, 1);
        }

        /* --- resblock --- */
        pu_resblock r;
        load_resblock(st, &r, "down_blocks.0.resnets.0", 320, 320);
        CUdeviceptr d_out, d_t1, d_t2, d_temb_act, d_temb_proj;
        cuMemAlloc(&d_out, hw_n * sizeof(float));
        cuMemAlloc(&d_t1,  hw_n * sizeof(float));
        cuMemAlloc(&d_t2,  hw_n * sizeof(float));
        cuMemAlloc(&d_temb_act,  B * 1280 * sizeof(float));
        cuMemAlloc(&d_temb_proj, B * 320  * sizeof(float));
        run_resblock(&kk, &r, d_x, d_out, d_t1, d_t2,
                      d_temb, d_temb_act, d_temb_proj, B, H, W, 32);
        cuCtxSynchronize();

        float *cu = (float *)malloc(hw_n * sizeof(float));
        cuMemcpyDtoH(cu, d_out, hw_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/ref_db0_res0.npy", ref_dir);
        diff_against(cu, path, hw_n, 1e-3f);
        free(cu); free(sample);
    } else if (!strcmp(stage, "db0_attn0")) {
        /* Pipeline: time_emb + conv_in + db0.res0 + db0.attn0 (Transformer2D)
         * Validate against ref_db0_attn0.npy [B, 320, 64, 64]. */
        snprintf(path, sizeof(path), "%s/ref_sample.npy", ref_dir);
        float *sample = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!sample) return 1;
        int IC = (int)shape[1], H = (int)shape[2], W = (int)shape[3];
        if (IC != 12) { fprintf(stderr, "ERROR: IC!=12\n"); return 1; }
        snprintf(path, sizeof(path), "%s/ref_encoder_hidden.npy", ref_dir);
        float *text = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!text) return 1;
        int M_text = (int)shape[1], cross_dim = (int)shape[2];
        fprintf(stderr, "text [%d, %d, %d]\n", B, M_text, cross_dim);

        int C = 320, N = H * W;

        /* --- time embedding --- */
        CUdeviceptr d_ts;   cuMemAlloc(&d_ts, B * sizeof(int64_t));
        cuMemcpyHtoD(d_ts, ts, B * sizeof(int64_t));
        CUdeviceptr d_temb_in; cuMemAlloc(&d_temb_in, B * 320 * sizeof(float));
        CUdeviceptr d_temb_h1; cuMemAlloc(&d_temb_h1, B * 1280 * sizeof(float));
        CUdeviceptr d_temb;    cuMemAlloc(&d_temb,    B * 1280 * sizeof(float));
        CUdeviceptr l1_w = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b = upload_st(st, "time_embedding.linear_2.bias");
        k_timestep_embed(&kk, d_temb_in, d_ts, B, 320);
        k_linear(&kk, d_temb_h1, d_temb_in, l1_w, l1_b, B, 320, 1280);
        k_silu(&kk, d_temb_h1, B * 1280);
        k_linear(&kk, d_temb, d_temb_h1, l2_w, l2_b, B, 1280, 1280);

        /* --- conv_in --- */
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");
        size_t in_n = (size_t)B * IC * H * W;
        size_t hw_n = (size_t)B * C * H * W;
        CUdeviceptr d_in;  cuMemAlloc(&d_in,  in_n * sizeof(float));
        CUdeviceptr d_x;   cuMemAlloc(&d_x,   hw_n * sizeof(float));
        cuMemcpyHtoD(d_in, sample, in_n * sizeof(float));
        for (int b = 0; b < B; b++) {
            CUdeviceptr in_b = d_in + (CUdeviceptr)b * IC * H * W * sizeof(float);
            CUdeviceptr x_b  = d_x  + (CUdeviceptr)b * C  * H * W * sizeof(float);
            k_conv(&kk, x_b, in_b, cw, cb, IC, H, W, C, 3, 3, 1);
        }

        /* --- res0 --- */
        pu_resblock r;
        load_resblock(st, &r, "down_blocks.0.resnets.0", C, C);
        CUdeviceptr d_res, d_t1, d_t2, d_temb_act, d_temb_proj;
        cuMemAlloc(&d_res, hw_n * sizeof(float));
        cuMemAlloc(&d_t1,  hw_n * sizeof(float));
        cuMemAlloc(&d_t2,  hw_n * sizeof(float));
        cuMemAlloc(&d_temb_act,  B * 1280 * sizeof(float));
        cuMemAlloc(&d_temb_proj, B * 1280 * sizeof(float)); /* big enough */
        run_resblock(&kk, &r, d_x, d_res, d_t1, d_t2,
                      d_temb, d_temb_act, d_temb_proj, B, H, W, 32);

        /* --- attn0 (Transformer2DModel, 1 BasicTransformerBlock at this level) --- */
        pu_transformer T;
        load_transformer(st, &T, "down_blocks.0.attentions.0", C,
                          /*num_heads*/ 5, cross_dim, /*num_blocks*/ 1);
        /* text upload */
        CUdeviceptr d_text; cuMemAlloc(&d_text, (size_t)B * M_text * cross_dim * sizeof(float));
        cuMemcpyHtoD(d_text, text, (size_t)B * M_text * cross_dim * sizeof(float));

        /* Scratch: B*N*C floats per buffer (and the 2x ff_inner one). */
        size_t bnc = (size_t)B * N * C;
        size_t bmc = (size_t)B * (M_text > N ? M_text : N) * C;
        size_t bn2ff = (size_t)B * N * 2 * T.ff_inner;
        size_t bnff  = (size_t)B * N *     T.ff_inner;
        pu_xfm_scratch S;
        cuMemAlloc(&S.d_resid, bnc * sizeof(float));
        cuMemAlloc(&S.d_nc,    bnc * sizeof(float));
        cuMemAlloc(&S.d_nc_b,  bnc * sizeof(float));
        cuMemAlloc(&S.d_norm,  bnc * sizeof(float));
        cuMemAlloc(&S.d_q,     bnc * sizeof(float));
        cuMemAlloc(&S.d_k,     bmc * sizeof(float));
        cuMemAlloc(&S.d_v,     bmc * sizeof(float));
        cuMemAlloc(&S.d_attn,  bnc * sizeof(float));
        cuMemAlloc(&S.d_ff_gh, bn2ff * sizeof(float));
        cuMemAlloc(&S.d_ff_h,  bnff  * sizeof(float));

        run_transformer(&kk, &T, d_res, d_text, B, H, W, M_text, &S);
        cuCtxSynchronize();

        float *cu = (float *)malloc(hw_n * sizeof(float));
        cuMemcpyDtoH(cu, d_res, hw_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/ref_db0_attn0.npy", ref_dir);
        diff_against(cu, path, hw_n, 1e-3f);
        free(cu); free(sample); free(text);
    } else if (!strcmp(stage, "out")) {
        /* Full UNet forward -> [B, 4, 64, 64] vs ref_out.npy */
        snprintf(path, sizeof(path), "%s/ref_sample.npy", ref_dir);
        float *sample = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!sample) return 1;
        int IC = (int)shape[1], H0 = (int)shape[2], W0 = (int)shape[3];
        if (IC != 12) { fprintf(stderr, "ERROR: IC!=12\n"); return 1; }
        snprintf(path, sizeof(path), "%s/ref_encoder_hidden.npy", ref_dir);
        float *text = (float *)read_npy(path, &nd, shape, &n, dt);
        if (!text) return 1;
        int M_text = (int)shape[1], cross_dim = (int)shape[2];
        fprintf(stderr, "input [%d, %d, %d, %d]  text [%d, %d, %d]\n",
                B, IC, H0, W0, B, M_text, cross_dim);
        if (B != 1) { fprintf(stderr, "ERROR: out stage assumes B=1\n"); return 1; }

        /* time embedding */
        CUdeviceptr d_ts; cuMemAlloc(&d_ts, B * sizeof(int64_t));
        cuMemcpyHtoD(d_ts, ts, B * sizeof(int64_t));
        CUdeviceptr d_temb_in, d_temb_h1, d_temb;
        cuMemAlloc(&d_temb_in, B * 320 * sizeof(float));
        cuMemAlloc(&d_temb_h1, B * 1280 * sizeof(float));
        cuMemAlloc(&d_temb,    B * 1280 * sizeof(float));
        CUdeviceptr l1_w = upload_st(st, "time_embedding.linear_1.weight");
        CUdeviceptr l1_b = upload_st(st, "time_embedding.linear_1.bias");
        CUdeviceptr l2_w = upload_st(st, "time_embedding.linear_2.weight");
        CUdeviceptr l2_b = upload_st(st, "time_embedding.linear_2.bias");
        k_timestep_embed(&kk, d_temb_in, d_ts, B, 320);
        k_linear(&kk, d_temb_h1, d_temb_in, l1_w, l1_b, B, 320, 1280);
        k_silu(&kk, d_temb_h1, B * 1280);
        k_linear(&kk, d_temb, d_temb_h1, l2_w, l2_b, B, 1280, 1280);

        /* Activation buffers - pick max sizes for any tensor in the flow.
         * Largest single tensor = 320@64 = 1.31M; concat output max = 640@64
         * = 2.62M; res_in_concat max = 2560@8 = 164K; ff_gh max = 4096*2560
         * = 10.5M. Round generously. */
        size_t MAX_ACT  = (size_t)1280 * 64 * 64;          /* 5.24M */
        /* MAX_CCAT covers worst-case concat C_in (up3 first iter: 960@64 = 3.93M)
         * AND worst-case resnet scratch max(c_in,c_out)*HW (also up3.res0: 960@64). */
        size_t MAX_CCAT = (size_t)960 * 64 * 64;           /* 3.93M */
        size_t MAX_FF_GH = (size_t)320 * 64 * 64 * 2 * 4;  /* 10.5M */
        size_t MAX_FF_H  = (size_t)320 * 64 * 64 * 4;      /* 5.24M */
        size_t MAX_BNC   = (size_t)320 * 64 * 64;          /* 1.31M (token buffers, max channel*N) */
        if ((size_t)1280 * 16 * 16 > MAX_BNC) MAX_BNC = (size_t)1280 * 16 * 16;
        if ((size_t)640 * 32 * 32 > MAX_BNC)  MAX_BNC = (size_t)640 * 32 * 32;
        size_t MAX_BMC = MAX_BNC;
        if ((size_t)1280 * M_text > MAX_BMC) MAX_BMC = (size_t)1280 * M_text;

        pu_workspace ws;
        cuMemAlloc(&ws.d_a, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_b, MAX_ACT * sizeof(float));
        cuMemAlloc(&ws.d_t1, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_t2, MAX_CCAT * sizeof(float));
        cuMemAlloc(&ws.d_temb_act,  B * 1280 * sizeof(float));
        cuMemAlloc(&ws.d_temb_proj, B * 1280 * sizeof(float));
        cuMemAlloc(&ws.X.d_resid, MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc,    MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_nc_b,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_norm,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_q,     MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_k,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_v,     MAX_BMC * sizeof(float));
        cuMemAlloc(&ws.X.d_attn,  MAX_BNC * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_gh, MAX_FF_GH * sizeof(float));
        cuMemAlloc(&ws.X.d_ff_h,  MAX_FF_H * sizeof(float));

        /* d_concat is sized for the largest concat (up3 first iter: 640@64). */
        CUdeviceptr d_concat;
        cuMemAlloc(&d_concat, MAX_CCAT * sizeof(float));

        /* conv_in 12->320 into d_a */
        CUdeviceptr cw = upload_st(st, "conv_in.weight");
        CUdeviceptr cb = upload_st(st, "conv_in.bias");
        size_t in_n = (size_t)B * IC * H0 * W0;
        CUdeviceptr d_in_raw; cuMemAlloc(&d_in_raw, in_n * sizeof(float));
        cuMemcpyHtoD(d_in_raw, sample, in_n * sizeof(float));
        k_conv(&kk, ws.d_a, d_in_raw, cw, cb, IC, H0, W0, 320, 3, 3, 1);
        cuMemFree(d_in_raw);

        /* text */
        CUdeviceptr d_text; cuMemAlloc(&d_text, (size_t)B * M_text * cross_dim * sizeof(float));
        cuMemcpyHtoD(d_text, text, (size_t)B * M_text * cross_dim * sizeof(float));

        /* Skip stack: push initial (post conv_in) */
        pu_skip_stack ss = {.top = 0};
        skip_push_copy(&ss, ws.d_a, 320, H0, W0);

        /* Load all blocks */
        pu_down_block db[4];
        load_down_block(st, &db[0], 0,  320,  320,  5, 1, 1, cross_dim);
        load_down_block(st, &db[1], 1,  320,  640, 10, 1, 1, cross_dim);
        load_down_block(st, &db[2], 2,  640, 1280, 20, 1, 1, cross_dim);
        load_down_block(st, &db[3], 3, 1280, 1280, 20, 0, 0, cross_dim);
        pu_mid_block mid;
        load_mid_block(st, &mid, cross_dim);
        pu_up_block ub[4];
        /* up_blocks per diffusers: prev_out tracks the previous up's c_out (init=last block_out=1280). */
        load_up_block(st, &ub[0], 0, /*in*/1280, /*out*/1280, /*prev*/1280,  0, 0, 1, cross_dim); /* UpBlock2D */
        load_up_block(st, &ub[1], 1, /*in*/ 640, /*out*/1280, /*prev*/1280, 20, 1, 1, cross_dim);
        load_up_block(st, &ub[2], 2, /*in*/ 320, /*out*/ 640, /*prev*/1280, 10, 1, 1, cross_dim);
        load_up_block(st, &ub[3], 3, /*in*/ 320, /*out*/ 320, /*prev*/ 640,  5, 1, 0, cross_dim);

        /* Down path. Current activation lives in ws.d_a. */
        int H = H0, W = W0;
        run_down_block(&kk, &db[0], ws.d_a, ws.d_b, d_temb, d_text, B, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[1], ws.d_a, ws.d_b, d_temb, d_text, B, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[2], ws.d_a, ws.d_b, d_temb, d_text, B, &H, &W, M_text, &ws, &ss);
        run_down_block(&kk, &db[3], ws.d_a, ws.d_b, d_temb, d_text, B, &H, &W, M_text, &ws, &ss);
        /* Sanity: H=W=8 here, top of stack = db3.r1[1280@8], 12 entries total */
        if (ss.top != 12) {
            fprintf(stderr, "ERROR: skip stack top=%d, expected 12\n", ss.top);
            return 1;
        }

        /* Mid: in d_a, out also d_a (using d_b as scratch) */
        run_mid_block(&kk, &mid, ws.d_a, ws.d_b, d_temb, d_text, B, H, W, M_text, &ws);

        /* Up path */
        run_up_block(&kk, &ub[0], ws.d_a, ws.d_b, d_concat, d_temb, d_text, B, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[1], ws.d_a, ws.d_b, d_concat, d_temb, d_text, B, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[2], ws.d_a, ws.d_b, d_concat, d_temb, d_text, B, &H, &W, M_text, &ws, &ss);
        run_up_block(&kk, &ub[3], ws.d_a, ws.d_b, d_concat, d_temb, d_text, B, &H, &W, M_text, &ws, &ss);

        if (ss.top != 0) {
            fprintf(stderr, "WARN: skip stack not drained, top=%d\n", ss.top);
        }

        /* conv_norm_out (32 grp, +silu) -> d_b ; conv_out 320->4 -> d_a */
        CUdeviceptr ng = upload_st(st, "conv_norm_out.weight");
        CUdeviceptr nb = upload_st(st, "conv_norm_out.bias");
        CUdeviceptr ow = upload_st(st, "conv_out.weight");
        CUdeviceptr ob = upload_st(st, "conv_out.bias");
        k_groupnorm(&kk, ws.d_b, ws.d_a, ng, nb, 320, H * W, 32, 1);
        k_conv(&kk, ws.d_a, ws.d_b, ow, ob, 320, H, W, 4, 3, 3, 1);
        cuCtxSynchronize();

        size_t out_n = (size_t)B * 4 * H * W;
        float *cu = (float *)malloc(out_n * sizeof(float));
        cuMemcpyDtoH(cu, ws.d_a, out_n * sizeof(float));
        snprintf(path, sizeof(path), "%s/ref_out.npy", ref_dir);
        diff_against(cu, path, out_n, 1e-2f);
        free(cu); free(sample); free(text);
    } else {
        fprintf(stderr, "unknown stage: %s\n", stage); return 1;
    }

    free(ts);
    safetensors_close(st);
    cuModuleUnload(kk.mod);
    cuCtxDestroy(ctx);
    return 0;
}
