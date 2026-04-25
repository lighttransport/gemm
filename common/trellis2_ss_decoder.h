/*
 * trellis2_ss_decoder.h - TRELLIS.2 Sparse Structure Decoder
 *
 * Usage:
 *   #define T2_SS_DEC_IMPLEMENTATION
 *   #include "trellis2_ss_decoder.h"
 *
 * Dependencies: ggml_dequant.h, safetensors.h
 *
 * Decodes DiT latent [8, 16, 16, 16] to occupancy [1, 64, 64, 64].
 *
 * Architecture:
 *   Conv3d(8->512, k=3) at 16^3
 *   2x ResBlock3d(512) -> pixel_shuffle_3d(2) -> 32^3
 *   Conv3d(64->128, k=3) at 32^3
 *   2x ResBlock3d(128) -> pixel_shuffle_3d(2) -> 64^3
 *   Conv3d(16->32, k=3) at 64^3
 *   2x ResBlock3d(32)
 *   GroupNorm(32) -> SiLU -> Conv3d(32->1, k=3)
 *   Output: [1, 64, 64, 64] occupancy logits
 *
 * All convolutions use NCDHW layout, kernel_size=3, padding=1.
 *
 * API:
 *   t2_ss_dec  *t2_ss_dec_load(const char *st_path);
 *   void        t2_ss_dec_free(t2_ss_dec *d);
 *   float      *t2_ss_dec_forward(t2_ss_dec *d, const float *latent, int n_threads);
 */
#ifndef T2_SS_DEC_H
#define T2_SS_DEC_H

#include <stdint.h>
#include <stddef.h>
#include "ggml_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "qtensor_utils.h"

/* ResBlock3d: GN -> SiLU -> Conv3d -> GN -> SiLU -> Conv3d [+ skip conv] */
typedef struct {
    qtensor gn1_w, gn1_b;
    qtensor conv1_w, conv1_b;   /* [C, C, 3, 3, 3] */
    qtensor gn2_w, gn2_b;
    qtensor conv2_w, conv2_b;   /* [C, C, 3, 3, 3] */
    /* Optional skip projection (if in_channels != out_channels) */
    qtensor skip_w, skip_b;     /* [C_out, C_in, 1, 1, 1] */
    int has_skip;
} t2_ss_dec_resblock;

typedef struct {
    /* input_layer: Conv3d(8->512, k=3) at 16^3 */
    qtensor conv_in_w, conv_in_b;

    /* middle_block: 2x ResBlock3d(512) at 16^3 */
    t2_ss_dec_resblock middle[2];

    /* blocks.0-1: 2x ResBlock3d(512) at 16^3 */
    t2_ss_dec_resblock res_16[2];

    /* blocks.2.conv: Conv3d(512->1024, k=3) at 16^3, then pixel_shuffle -> [128, 32^3] */
    qtensor up1_conv_w, up1_conv_b;

    /* blocks.3-4: 2x ResBlock3d(128) at 32^3 */
    t2_ss_dec_resblock res_32[2];

    /* blocks.5.conv: Conv3d(128->256, k=3) at 32^3, then pixel_shuffle -> [32, 64^3] */
    qtensor up2_conv_w, up2_conv_b;

    /* blocks.6-7: 2x ResBlock3d(32) at 64^3 */
    t2_ss_dec_resblock res_64[2];

    /* out_layer: GN(32) -> SiLU -> Conv3d(32->1, k=3) */
    qtensor out_gn_w, out_gn_b;
    qtensor out_conv_w, out_conv_b;

    int gn_groups;   /* 32 */

    void *st_ctx;
} t2_ss_dec;

t2_ss_dec *t2_ss_dec_load(const char *st_path);
void       t2_ss_dec_free(t2_ss_dec *d);

/* Returns [64*64*64] float logits. Caller must free(). */
float *t2_ss_dec_forward(t2_ss_dec *d, const float *latent, int n_threads);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef T2_SS_DEC_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

static double t2dec_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- Tensor helpers ---- */

static int t2dec_numel(const qtensor *t) {
    if (!t->data) return 0;
    /* Use n_rows * n_cols for correct total even with >4D tensors */
    return t->n_rows * t->n_cols;
}

static float *t2dec_dequant(const qtensor *t) {
    if (!t->data) return NULL;
    int n = t2dec_numel(t);
    if (n <= 0) return NULL;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (t->type == GGML_TYPE_F32) {
        memcpy(buf, t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        memset(buf, 0, (size_t)n * sizeof(float));
    }
    return buf;
}

/* ---- ChannelLayerNorm: LayerNorm across C per spatial position ---- */
/* Input/Output: NCDHW layout [1, C, D, H, W] (batch=1)                */
/* G arg retained for ABI compat but ignored.                          */

static void t2dec_groupnorm(float *dst, const float *src,
                              const float *w, const float *b,
                              int C, int D, int H, int W, int G) {
    (void)G;
    int spatial = D * H * W;
    float eps = 1e-5f;
    for (int s = 0; s < spatial; s++) {
        float mean = 0.0f, m2 = 0.0f;
        for (int c = 0; c < C; c++) {
            float v = src[(size_t)c * spatial + s];
            mean += v; m2 += v * v;
        }
        mean /= (float)C;
        float var = m2 / (float)C - mean * mean;
        float inv = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < C; c++) {
            float wc = w ? w[c] : 1.0f;
            float bc = b ? b[c] : 0.0f;
            dst[(size_t)c * spatial + s] =
                (src[(size_t)c * spatial + s] - mean) * inv * wc + bc;
        }
    }
}

/* ---- Dense Conv3d: kernel_size=3, padding=1, stride=1 ---- */
/* Weight: [Co, Ci, 3, 3, 3], Input/Output: NCDHW [1, C, D, H, W] */

static void t2dec_conv3d(float *dst, const float *src, const float *weight,
                           const float *bias, int Ci, int Co,
                           int D, int H, int Wi) {
    int spatial = D * H * Wi;

    /* Parallel over output channels: each `co` writes an independent slab.
     * Kernel layout [Co, Ci, kD, kH, kW]. */
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int co = 0; co < Co; co++) {
        float *dst_co = dst + (size_t)co * spatial;
        float bv = bias ? bias[co] : 0.0f;
        for (int s = 0; s < spatial; s++) dst_co[s] = bv;

        for (int ci = 0; ci < Ci; ci++) {
            const float *kern = weight + ((size_t)co * Ci + ci) * 27;
            const float *src_ci = src + (size_t)ci * spatial;
            for (int d = 0; d < D; d++) {
                for (int h = 0; h < H; h++) {
                    for (int w_pos = 0; w_pos < Wi; w_pos++) {
                        float sum = 0.0f;
                        for (int kd = 0; kd < 3; kd++) {
                            int dd = d + kd - 1;
                            if (dd < 0 || dd >= D) continue;
                            for (int kh = 0; kh < 3; kh++) {
                                int hh = h + kh - 1;
                                if (hh < 0 || hh >= H) continue;
                                for (int kw = 0; kw < 3; kw++) {
                                    int ww = w_pos + kw - 1;
                                    if (ww < 0 || ww >= Wi) continue;
                                    sum += kern[kd * 9 + kh * 3 + kw]
                                         * src_ci[dd * H * Wi + hh * Wi + ww];
                                }
                            }
                        }
                        dst_co[d * H * Wi + h * Wi + w_pos] += sum;
                    }
                }
            }
        }
    }
}

/* ---- pixel_shuffle_3d: rearrange [C*8, D, H, W] -> [C, 2D, 2H, 2W] ---- */

static void t2dec_pixel_shuffle_3d(float *dst, const float *src,
                                     int C, int D, int H, int W) {
    /* src: [C*8, D, H, W], dst: [C, 2D, 2H, 2W] */
    int spatial_in = D * H * W;
    int D2 = 2 * D, H2 = 2 * H, W2 = 2 * W;
    int spatial_out = D2 * H2 * W2;

    for (int c = 0; c < C; c++) {
        for (int d = 0; d < D; d++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    for (int sd = 0; sd < 2; sd++) {
                        for (int sh = 0; sh < 2; sh++) {
                            for (int sw = 0; sw < 2; sw++) {
                                int sub_ch = (sd * 2 + sh) * 2 + sw;
                                int src_ch = c * 8 + sub_ch;
                                int od = 2 * d + sd;
                                int oh = 2 * h + sh;
                                int ow = 2 * w + sw;
                                dst[(size_t)c * spatial_out + od * H2 * W2 + oh * W2 + ow] =
                                    src[(size_t)src_ch * spatial_in + d * H * W + h * W + w];
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ---- SiLU activation (in-place) ---- */

static void t2dec_silu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* ---- ResBlock3d forward ---- */
/* GN -> SiLU -> Conv3d -> GN -> SiLU -> Conv3d + skip */

static void t2dec_resblock(float *dst, const float *src,
                             const t2_ss_dec_resblock *blk,
                             int Ci, int Co, int D, int H, int W, int G) {
    int spatial = D * H * W;

    /* Dequant weights */
    float *gn1_w = t2dec_dequant(&blk->gn1_w);
    float *gn1_b = t2dec_dequant(&blk->gn1_b);
    float *conv1_w = t2dec_dequant(&blk->conv1_w);
    float *conv1_b = t2dec_dequant(&blk->conv1_b);
    float *gn2_w = t2dec_dequant(&blk->gn2_w);
    float *gn2_b = t2dec_dequant(&blk->gn2_b);
    float *conv2_w = t2dec_dequant(&blk->conv2_w);
    float *conv2_b = t2dec_dequant(&blk->conv2_b);

    /* h = GN1(src) */
    float *h = (float *)malloc((size_t)Ci * spatial * sizeof(float));
    t2dec_groupnorm(h, src, gn1_w, gn1_b, Ci, D, H, W, G < Ci ? G : Ci);
    t2dec_silu(h, Ci * spatial);

    /* h = Conv1(h) */
    float *h2 = (float *)malloc((size_t)Co * spatial * sizeof(float));
    t2dec_conv3d(h2, h, conv1_w, conv1_b, Ci, Co, D, H, W);
    free(h);

    /* h2 = GN2(h2) */
    float *h3 = (float *)malloc((size_t)Co * spatial * sizeof(float));
    t2dec_groupnorm(h3, h2, gn2_w, gn2_b, Co, D, H, W, G < Co ? G : Co);
    t2dec_silu(h3, Co * spatial);

    /* h3 = Conv2(h3) */
    t2dec_conv3d(h2, h3, conv2_w, conv2_b, Co, Co, D, H, W);
    free(h3);

    /* Skip connection */
    if (blk->has_skip && blk->skip_w.data) {
        /* 1x1x1 conv: dst[co, s] = sum_ci skip_w[co, ci] * src[ci, s] + skip_b[co] */
        float *skip_w = t2dec_dequant(&blk->skip_w);
        float *skip_b = t2dec_dequant(&blk->skip_b);
        for (int co = 0; co < Co; co++) {
            float bv = skip_b ? skip_b[co] : 0.0f;
            for (int s = 0; s < spatial; s++) {
                float sum = bv;
                for (int ci = 0; ci < Ci; ci++)
                    sum += skip_w[(size_t)co * Ci + ci] * src[(size_t)ci * spatial + s];
                dst[(size_t)co * spatial + s] = h2[(size_t)co * spatial + s] + sum;
            }
        }
        free(skip_w); free(skip_b);
    } else {
        /* Identity skip: dst = h2 + src */
        for (int i = 0; i < Co * spatial; i++)
            dst[i] = h2[i] + src[i];
    }

    free(h2);
    free(gn1_w); free(gn1_b); free(conv1_w); free(conv1_b);
    free(gn2_w); free(gn2_b); free(conv2_w); free(conv2_b);
}

/* ---- Full decoder forward ---- */

float *t2_ss_dec_forward(t2_ss_dec *d, const float *latent, int n_threads) {
    (void)n_threads;
    double t0 = t2dec_time_ms();
    int G = d->gn_groups;

    /* Input: latent [8, 16, 16, 16] NCDHW */
    /* input_layer: Conv3d(8->512, k=3) at 16^3 */
    float *conv_in_w = t2dec_dequant(&d->conv_in_w);
    float *conv_in_b = t2dec_dequant(&d->conv_in_b);
    float *h = (float *)malloc((size_t)512 * 16 * 16 * 16 * sizeof(float));
    t2dec_conv3d(h, latent, conv_in_w, conv_in_b, 8, 512, 16, 16, 16);
    free(conv_in_w); free(conv_in_b);
    fprintf(stderr, "ss_dec: conv_in done (512, 16^3)\n");

    /* middle_block: 2x ResBlock3d(512) at 16^3 */
    float *h2 = (float *)malloc((size_t)512 * 16 * 16 * 16 * sizeof(float));
    t2dec_resblock(h2, h, &d->middle[0], 512, 512, 16, 16, 16, G);
    t2dec_resblock(h, h2, &d->middle[1], 512, 512, 16, 16, 16, G);
    fprintf(stderr, "ss_dec: middle done (512, 16^3)\n");

    /* blocks.0-1: 2x ResBlock3d(512) at 16^3 */
    t2dec_resblock(h2, h, &d->res_16[0], 512, 512, 16, 16, 16, G);
    t2dec_resblock(h, h2, &d->res_16[1], 512, 512, 16, 16, 16, G);
    free(h2);
    fprintf(stderr, "ss_dec: res_16 done (512, 16^3)\n");

    /* blocks.2.conv: Conv3d(512->1024, k=3) at 16^3 */
    float *up1_w = t2dec_dequant(&d->up1_conv_w);
    float *up1_b = t2dec_dequant(&d->up1_conv_b);
    float *h_up = (float *)malloc((size_t)1024 * 16 * 16 * 16 * sizeof(float));
    t2dec_conv3d(h_up, h, up1_w, up1_b, 512, 1024, 16, 16, 16);
    free(h); free(up1_w); free(up1_b);

    /* pixel_shuffle_3d(2): [1024, 16, 16, 16] -> [128, 32, 32, 32] */
    float *ps1 = (float *)malloc((size_t)128 * 32 * 32 * 32 * sizeof(float));
    t2dec_pixel_shuffle_3d(ps1, h_up, 128, 16, 16, 16);
    free(h_up);
    fprintf(stderr, "ss_dec: upsample -> (128, 32^3)\n");

    /* blocks.3-4: 2x ResBlock3d(128) at 32^3 */
    h = (float *)malloc((size_t)128 * 32 * 32 * 32 * sizeof(float));
    t2dec_resblock(h, ps1, &d->res_32[0], 128, 128, 32, 32, 32, G);
    t2dec_resblock(ps1, h, &d->res_32[1], 128, 128, 32, 32, 32, G);
    free(h);
    fprintf(stderr, "ss_dec: res_32 done (128, 32^3)\n");

    /* blocks.5.conv: Conv3d(128->256, k=3) at 32^3 */
    float *up2_w = t2dec_dequant(&d->up2_conv_w);
    float *up2_b = t2dec_dequant(&d->up2_conv_b);
    h_up = (float *)malloc((size_t)256 * 32 * 32 * 32 * sizeof(float));
    t2dec_conv3d(h_up, ps1, up2_w, up2_b, 128, 256, 32, 32, 32);
    free(ps1); free(up2_w); free(up2_b);

    /* pixel_shuffle_3d(2): [256, 32, 32, 32] -> [32, 64, 64, 64] */
    float *ps2 = (float *)malloc((size_t)32 * 64 * 64 * 64 * sizeof(float));
    t2dec_pixel_shuffle_3d(ps2, h_up, 32, 32, 32, 32);
    free(h_up);
    fprintf(stderr, "ss_dec: upsample -> (32, 64^3)\n");

    /* blocks.6-7: 2x ResBlock3d(32) at 64^3 */
    h = (float *)malloc((size_t)32 * 64 * 64 * 64 * sizeof(float));
    t2dec_resblock(h, ps2, &d->res_64[0], 32, 32, 64, 64, 64, G);
    t2dec_resblock(ps2, h, &d->res_64[1], 32, 32, 64, 64, 64, G);
    free(h);
    fprintf(stderr, "ss_dec: res_64 done (32, 64^3)\n");

    /* out_layer: GN(32) -> SiLU -> Conv3d(32->1, k=3) */
    float *gn_w = t2dec_dequant(&d->out_gn_w);
    float *gn_b = t2dec_dequant(&d->out_gn_b);
    h = (float *)malloc((size_t)32 * 64 * 64 * 64 * sizeof(float));
    t2dec_groupnorm(h, ps2, gn_w, gn_b, 32, 64, 64, 64, G);
    t2dec_silu(h, 32 * 64 * 64 * 64);
    free(gn_w); free(gn_b); free(ps2);

    float *out_conv_w = t2dec_dequant(&d->out_conv_w);
    float *out_conv_b = t2dec_dequant(&d->out_conv_b);
    float *out = (float *)malloc((size_t)64 * 64 * 64 * sizeof(float));
    t2dec_conv3d(out, h, out_conv_w, out_conv_b, 32, 1, 64, 64, 64);
    free(h); free(out_conv_w); free(out_conv_b);

    double t1 = t2dec_time_ms();
    fprintf(stderr, "ss_dec: forward done in %.1f ms, output (1, 64^3)\n", t1 - t0);
    return out;
}

/* ==================================================================== */
/* SafeTensors loading                                                   */
/* ==================================================================== */

#ifdef SAFETENSORS_H

static void t2dec_load_resblock(st_context *st, t2_ss_dec_resblock *blk,
                                  const char *prefix, int Ci, int Co) {
    char name[512];
    int idx;

    #define DEC_LOAD(field, suffix) do { \
        snprintf(name, sizeof(name), "%s%s", prefix, suffix); \
        idx = safetensors_find(st, name); \
        if (idx >= 0) blk->field = qt_make_tensor(st, idx); \
    } while(0)

    DEC_LOAD(gn1_w,   "norm1.weight");
    DEC_LOAD(gn1_b,   "norm1.bias");
    DEC_LOAD(conv1_w, "conv1.weight");
    DEC_LOAD(conv1_b, "conv1.bias");
    DEC_LOAD(gn2_w,   "norm2.weight");
    DEC_LOAD(gn2_b,   "norm2.bias");
    DEC_LOAD(conv2_w, "conv2.weight");
    DEC_LOAD(conv2_b, "conv2.bias");

    blk->has_skip = (Ci != Co) ? 1 : 0;
    if (blk->has_skip) {
        DEC_LOAD(skip_w, "conv_shortcut.weight");
        DEC_LOAD(skip_b, "conv_shortcut.bias");
        if (!blk->skip_w.data) {
            DEC_LOAD(skip_w, "skip.weight");
            DEC_LOAD(skip_b, "skip.bias");
        }
    }

    #undef DEC_LOAD
}

t2_ss_dec *t2_ss_dec_load(const char *st_path) {
    st_context *st = safetensors_open(st_path);
    if (!st) return NULL;

    fprintf(stderr, "ss_dec: opened safetensors, %d tensors\n", st->n_tensors);

    /* Print tensor names */
    int show = st->n_tensors < 15 ? st->n_tensors : 15;
    for (int i = 0; i < show; i++) {
        const char *nm = safetensors_name(st, i);
        const char *dt = safetensors_dtype(st, i);
        int nd = safetensors_ndims(st, i);
        const uint64_t *sh = safetensors_shape(st, i);
        fprintf(stderr, "  [%d] %s: %s [", i, nm, dt);
        for (int d = 0; d < nd; d++) fprintf(stderr, "%s%lu", d ? "," : "", (unsigned long)sh[d]);
        fprintf(stderr, "]\n");
    }
    if (st->n_tensors > 15)
        fprintf(stderr, "  ... (%d more)\n", st->n_tensors - 15);

    /* Detect prefix */
    char prefix[256] = "";
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const char *p = strstr(nm, "conv_in");
        if (!p) p = strstr(nm, "blocks.");
        if (!p) p = strstr(nm, "input_layer");
        if (p) {
            size_t pl = (size_t)(p - nm);
            if (pl < sizeof(prefix)) {
                memcpy(prefix, nm, pl);
                prefix[pl] = '\0';
            }
            break;
        }
    }
    fprintf(stderr, "ss_dec: detected prefix: '%s'\n", prefix);

    #define DEC_FIND(suffix) ({ \
        char _buf[512]; \
        snprintf(_buf, sizeof(_buf), "%s%s", prefix, suffix); \
        int _idx = safetensors_find(st, _buf); \
        (_idx >= 0) ? qt_make_tensor(st, _idx) : (qtensor){0}; \
    })

    t2_ss_dec *d = (t2_ss_dec *)calloc(1, sizeof(t2_ss_dec));
    d->gn_groups = 32;
    d->st_ctx = st;

    /* input_layer: Conv3d(8->512, k=3) */
    d->conv_in_w = DEC_FIND("input_layer.weight");
    d->conv_in_b = DEC_FIND("input_layer.bias");

    /* middle_block: 2x ResBlock3d(512) */
    char blk_prefix[512];
    for (int i = 0; i < 2; i++) {
        snprintf(blk_prefix, sizeof(blk_prefix), "%smiddle_block.%d.", prefix, i);
        t2dec_load_resblock(st, &d->middle[i], blk_prefix, 512, 512);
    }

    /* blocks.0-1: 2x ResBlock3d(512) at 16^3 */
    for (int i = 0; i < 2; i++) {
        snprintf(blk_prefix, sizeof(blk_prefix), "%sblocks.%d.", prefix, i);
        t2dec_load_resblock(st, &d->res_16[i], blk_prefix, 512, 512);
    }

    /* blocks.2.conv: Conv3d(512->1024, k=3) — upsample conv */
    d->up1_conv_w = DEC_FIND("blocks.2.conv.weight");
    d->up1_conv_b = DEC_FIND("blocks.2.conv.bias");

    /* blocks.3-4: 2x ResBlock3d(128) at 32^3 */
    for (int i = 0; i < 2; i++) {
        snprintf(blk_prefix, sizeof(blk_prefix), "%sblocks.%d.", prefix, 3 + i);
        t2dec_load_resblock(st, &d->res_32[i], blk_prefix, 128, 128);
    }

    /* blocks.5.conv: Conv3d(128->256, k=3) — upsample conv */
    d->up2_conv_w = DEC_FIND("blocks.5.conv.weight");
    d->up2_conv_b = DEC_FIND("blocks.5.conv.bias");

    /* blocks.6-7: 2x ResBlock3d(32) at 64^3 */
    for (int i = 0; i < 2; i++) {
        snprintf(blk_prefix, sizeof(blk_prefix), "%sblocks.%d.", prefix, 6 + i);
        t2dec_load_resblock(st, &d->res_64[i], blk_prefix, 32, 32);
    }

    /* out_layer: GN(32) -> SiLU -> Conv3d(32->1, k=3) */
    d->out_gn_w = DEC_FIND("out_layer.0.weight");
    d->out_gn_b = DEC_FIND("out_layer.0.bias");
    d->out_conv_w = DEC_FIND("out_layer.2.weight");
    d->out_conv_b = DEC_FIND("out_layer.2.bias");

    #undef DEC_FIND

    fprintf(stderr, "ss_dec: input_layer: %s\n", d->conv_in_w.data ? "loaded" : "MISSING");
    fprintf(stderr, "ss_dec: middle[0]: %s\n", d->middle[0].conv1_w.data ? "loaded" : "MISSING");
    fprintf(stderr, "ss_dec: res_16[0]: %s\n", d->res_16[0].conv1_w.data ? "loaded" : "MISSING");
    fprintf(stderr, "ss_dec: up1_conv: %s\n", d->up1_conv_w.data ? "loaded" : "MISSING");
    fprintf(stderr, "ss_dec: res_32[0]: %s\n", d->res_32[0].conv1_w.data ? "loaded" : "MISSING");
    fprintf(stderr, "ss_dec: up2_conv: %s\n", d->up2_conv_w.data ? "loaded" : "MISSING");
    fprintf(stderr, "ss_dec: res_64[0]: %s\n", d->res_64[0].conv1_w.data ? "loaded" : "MISSING");
    fprintf(stderr, "ss_dec: out_layer: %s\n", d->out_conv_w.data ? "loaded" : "MISSING");

    return d;
}

#endif /* SAFETENSORS_H */

void t2_ss_dec_free(t2_ss_dec *d) {
    if (!d) return;
#ifdef SAFETENSORS_H
    if (d->st_ctx) safetensors_close((st_context *)d->st_ctx);
#endif
    free(d);
}

#endif /* T2_SS_DEC_IMPLEMENTATION */
#endif /* T2_SS_DEC_H */
