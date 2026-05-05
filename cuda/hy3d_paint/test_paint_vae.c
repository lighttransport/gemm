/*
 * test_paint_vae.c - Native CUDA SD-2.1 paint VAE decoder.
 *
 * Phase 2 of the Hunyuan3D-2.1 texgen port. Loads weights from
 * paint_vae.safetensors (run ref/hy3d/export_vae_safetensors.py to convert
 * the upstream .bin), decodes a latent .npy, and writes the resulting RGB
 * back as a .npy for diffing against the diffusers reference produced by
 * ref/hy3d/dump_paint_vae.py.
 *
 * Usage:
 *   ./test_paint_vae <vae.safetensors> <latent.npy> <out_recon.npy>
 *
 * Reference flow:
 *   uv run --with torch --with diffusers --with safetensors --with pillow \
 *       ref/hy3d/dump_paint_vae.py --image fujisan.jpg
 *   ./test_paint_vae \
 *       /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/vae/paint_vae.safetensors \
 *       /tmp/hy3d_paint_vae_ref/ref_latent.npy \
 *       /tmp/hy3d_paint_vae_ref/cuda_recon.npy
 *
 * Architecture (stock diffusers AutoencoderKL):
 *   post_quant_conv 1x1 (4->4)
 *   conv_in 3x3 (4->512)
 *   mid: ResBlock(512) → Attn(512) → ResBlock(512)
 *   up_blocks[0]: 3 ResBlocks(512->512), upsample(512)
 *   up_blocks[1]: 3 ResBlocks(512->512), upsample(512)
 *   up_blocks[2]: 3 ResBlocks(512->256), upsample(256)
 *   up_blocks[3]: 3 ResBlocks(256->128), no upsample
 *   conv_norm_out 32grp -> SiLU -> conv_out 3x3 (128->3)
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "cuda_paint_vae_kernels.h"

#define SAFETENSORS_IMPLEMENTATION
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
    CUfunction f_up2x;     /* vae_upsample2x_f32 */
    CUfunction f_add;      /* vae_add_f32 */
    CUfunction f_attn;     /* vae_attn_f32 */
    CUfunction f_chw_nc;   /* vae_chw_to_nc_f32 */
    CUfunction f_nc_chw;   /* vae_nc_to_chw_f32 */
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

static void k_conv(const pvae_kernels *kk, CUdeviceptr out, CUdeviceptr in,
                    CUdeviceptr w, CUdeviceptr b,
                    int ci, int h, int wd, int co, int kh, int kw, int pad) {
    void *args[] = { &out, &in, &w, &b, &ci, &h, &wd, &co, &kh, &kw, &pad };
    int total = co * h * wd;
    unsigned grid = (unsigned)((total + 255) / 256);
    cuLaunchKernel(kk->f_conv, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
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

/* ===== main =============================================================== */

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <vae.safetensors> <latent.npy> <out_recon.npy>\n",
            argv[0]);
        return 1;
    }
    const char *st_path  = argv[1];
    const char *lat_path = argv[2];
    const char *out_path = argv[3];

    int nd; uint64_t shape[8]; size_t total;
    float *lat = read_npy_f32(lat_path, &nd, shape, &total);
    if (!lat) return 1;
    if (nd != 3 || shape[0] != 4) {
        fprintf(stderr, "ERROR: expected latent shape [4,H,W], got nd=%d\n", nd);
        return 1;
    }
    int LC = (int)shape[0], LH = (int)shape[1], LW = (int)shape[2];
    int OH = LH * 8, OW = LW * 8;
    fprintf(stderr, "latent  [%d, %d, %d]   recon [3, %d, %d]\n",
            LC, LH, LW, OH, OW);

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1;
    }
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    pvae_kernels kk = {0};
    int sm = cu_compile_kernels(&kk.mod, dev,
                                cuda_paint_vae_kernels_src,
                                "hy3d_paint_vae", 1, "HY3D-PAINT-VAE");
    if (sm < 0) return 1;
    cuModuleGetFunction(&kk.f_gn,     kk.mod, "vae_groupnorm_f32");
    cuModuleGetFunction(&kk.f_conv,   kk.mod, "vae_conv2d_f32");
    cuModuleGetFunction(&kk.f_up2x,   kk.mod, "vae_upsample2x_f32");
    cuModuleGetFunction(&kk.f_add,    kk.mod, "vae_add_f32");
    cuModuleGetFunction(&kk.f_attn,   kk.mod, "vae_attn_f32");
    cuModuleGetFunction(&kk.f_chw_nc, kk.mod, "vae_chw_to_nc_f32");
    cuModuleGetFunction(&kk.f_nc_chw, kk.mod, "vae_nc_to_chw_f32");

    st_context *st = safetensors_open(st_path);
    if (!st) { fprintf(stderr, "ERROR: cannot open %s\n", st_path); return 1; }
    pvae_decoder D = {0};
    load_decoder(st, &D);
    fprintf(stderr, "loaded decoder weights from %s\n", st_path);

    /* Allocate working buffers. Worst case [512, OH/2, OW/2] (after up_blocks[0]
     * upsample). Actually after each up_block channels drop and spatial doubles;
     * worst-case bytes = max over stages of c*spatial. Compute and use that. */
    size_t max_n = 0;
    /* mid + up_blocks[0] before upsample: 512 * LH*LW = 512 * 64*64 = 2.0M */
    max_n = 512 * (size_t)LH * LW;
    /* up_blocks[0] after upsample: 512 * (2H)*(2W) */
    if (512 * (size_t)(LH*2) * (LW*2) > max_n) max_n = 512 * (size_t)(LH*2) * (LW*2);
    /* up_blocks[1] after upsample: 512 * 4H*4W */
    if (512 * (size_t)(LH*4) * (LW*4) > max_n) max_n = 512 * (size_t)(LH*4) * (LW*4);
    /* up_blocks[2] after upsample: 256 * 8H*8W */
    if (256 * (size_t)(LH*8) * (LW*8) > max_n) max_n = 256 * (size_t)(LH*8) * (LW*8);
    /* up_blocks[3] (no upsample): 128 * 8H*8W */
    if (128 * (size_t)(LH*8) * (LW*8) > max_n) max_n = 128 * (size_t)(LH*8) * (LW*8);
    fprintf(stderr, "workspace = %.1f MB / buffer\n",
            max_n * 4 / 1024.0 / 1024.0);

    /* For attention scratch in NC layout: 512 * 4096 = 2M floats per Q/K/V/Y. */
    size_t attn_n = 512 * (size_t)LH * LW;

    CUdeviceptr d_lat, d_rgb, d_a, d_b, d_t1, d_t2;
    CUdeviceptr d_qnc, d_knc, d_vnc, d_ync;
    cuMemAlloc(&d_lat, LC * (size_t)LH * LW * sizeof(float));
    cuMemAlloc(&d_rgb, 3  * (size_t)OH * OW * sizeof(float));
    cuMemAlloc(&d_a,   max_n * sizeof(float));
    cuMemAlloc(&d_b,   max_n * sizeof(float));
    cuMemAlloc(&d_t1,  max_n * sizeof(float));
    cuMemAlloc(&d_t2,  max_n * sizeof(float));
    cuMemAlloc(&d_qnc, attn_n * sizeof(float));
    cuMemAlloc(&d_knc, attn_n * sizeof(float));
    cuMemAlloc(&d_vnc, attn_n * sizeof(float));
    cuMemAlloc(&d_ync, attn_n * sizeof(float));
    cuMemcpyHtoD(d_lat, lat, LC * (size_t)LH * LW * sizeof(float));

    decode(&kk, &D, d_lat, LH, LW, d_rgb,
            d_a, d_b, d_t1, d_t2, d_qnc, d_knc, d_vnc, d_ync);
    cuCtxSynchronize();

    float *rgb = (float *)malloc(3 * (size_t)OH * OW * sizeof(float));
    cuMemcpyDtoH(rgb, d_rgb, 3 * (size_t)OH * OW * sizeof(float));

    int sh3[3] = { 3, OH, OW };
    write_npy_f32(out_path, rgb, sh3, 3);
    fprintf(stderr, "wrote %s  range=[%.3f, %.3f]\n",
            out_path,
            (float)({float mn=rgb[0]; for (int i=1;i<3*OH*OW;i++) if(rgb[i]<mn)mn=rgb[i]; mn;}),
            (float)({float mx=rgb[0]; for (int i=1;i<3*OH*OW;i++) if(rgb[i]>mx)mx=rgb[i]; mx;}));

    free(rgb); free(lat);
    safetensors_close(st);
    cuModuleUnload(kk.mod);
    cuCtxDestroy(ctx);
    return 0;
}
