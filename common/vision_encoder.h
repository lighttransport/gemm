/*
 * vision_encoder.h - Qwen3-VL CLIP vision encoder (mmproj)
 *
 * Usage:
 *   #define VISION_ENCODER_IMPLEMENTATION
 *   #include "vision_encoder.h"
 *
 * Dependencies: gguf_loader.h, ggml_dequant.h
 *
 * API:
 *   vision_model *vision_load(gguf_context *mmproj_gguf);
 *   void vision_free(vision_model *vm);
 *   float *vision_encode(vision_model *vm, const float *rgb_norm, int width, int height);
 */
#ifndef VISION_ENCODER_H
#define VISION_ENCODER_H

#include <stdint.h>
#include <stddef.h>
#include "gguf_loader.h"
#include "ggml_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Fused QKV: [dim, 3*dim] */
    qtensor attn_qkv_w, attn_qkv_b;
    /* Attention output: [dim, dim] */
    qtensor attn_out_w, attn_out_b;
    /* FFN up: [dim, ffn_dim], down: [ffn_dim, dim] */
    qtensor ffn_up_w, ffn_up_b;
    qtensor ffn_down_w, ffn_down_b;
    /* LayerNorms */
    qtensor ln1_w, ln1_b;
    qtensor ln2_w, ln2_b;
} vision_block;

typedef struct {
    qtensor fc1_w, fc1_b;
    qtensor fc2_w, fc2_b;
    qtensor norm_w, norm_b;
} vision_deepstack;

typedef struct {
    /* Hyperparameters */
    int n_blocks;
    int dim;          /* embedding_length (1024 for 2B) */
    int n_heads;
    int head_dim;
    int ffn_dim;
    int patch_size;
    int image_size;
    int n_patches;    /* (image_size/patch_size)^2 */
    int proj_dim;     /* LLM projection dim */
    int spatial_merge; /* 2 */
    int n_merged;     /* n_patches / (spatial_merge^2) */
    float ln_eps;
    float image_mean[3];
    float image_std[3];

    /* Patch embedding (dual conv2d: two kernels added together) */
    qtensor patch_embd_w;   /* [16, 16, 3, dim] - patch_embeddings_0 */
    qtensor patch_embd_w1;  /* [16, 16, 3, dim] - patch_embeddings_1 */
    qtensor patch_embd_b;   /* [dim] */

    /* Position embedding */
    qtensor position_embd; /* [dim, n_patches] */

    /* Blocks */
    vision_block *blocks;

    /* DeepStack */
    int n_deepstack;
    int *deepstack_indices;
    vision_deepstack *deepstack;

    /* Post LN */
    qtensor post_ln_w, post_ln_b;

    /* MM projection */
    qtensor mm0_w, mm0_b;  /* [merged_dim, merged_dim] */
    qtensor mm2_w, mm2_b;  /* [merged_dim, proj_dim] */
} vision_model;

vision_model *vision_load(gguf_context *mmproj_gguf);
void vision_free(vision_model *vm);

/* Encode an image. rgb_norm is [height * width * 3] normalized float RGB.
 * Returns malloc'd float array of [n_merged * proj_dim].
 * Caller must free the result. */
float *vision_encode(vision_model *vm, const float *rgb_norm, int width, int height);

/* Helper: normalize raw uint8 RGB image to float with model's mean/std */
float *vision_normalize_image(const vision_model *vm, const uint8_t *rgb, int width, int height);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef VISION_ENCODER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Profiling macros: active only if profiler.h was included before this file */
#ifdef PROFILER_H
#define VIT_PROF_BEGIN(name, layer, op, prec) prof_begin(name, "vision", layer, op, prec)
#define VIT_PROF_END(name, flops, iops) prof_end(name, flops, iops)
#else
#define VIT_PROF_BEGIN(name, layer, op, prec) ((void)0)
#define VIT_PROF_END(name, flops, iops) ((void)0)
#endif

/* ---- Tensor loading ---- */

static int vit_find_tensor(const gguf_context *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name.str, name) == 0) return (int)i;
    }
    return -1;
}

static qtensor vit_load(const gguf_context *g, const char *name, int req) {
    qtensor t = {0};
    int idx = vit_find_tensor(g, name);
    if (idx < 0) {
        if (req) fprintf(stderr, "vision: missing tensor '%s'\n", name);
        return t;
    }
    t.data = gguf_tensor_data(g, idx);
    t.type = g->tensors[idx].type;
    t.n_cols = (int)g->tensors[idx].dims[0];
    t.n_rows = (g->tensors[idx].n_dims >= 2) ? (int)g->tensors[idx].dims[1] : 1;
    return t;
}

static int vit_get_int(const gguf_context *g, const char *key, int def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_UINT32) return (int)g->kv[idx].value.u32;
    if (g->kv[idx].type == GGUF_TYPE_INT32) return g->kv[idx].value.i32;
    return def;
}

static float vit_get_float(const gguf_context *g, const char *key, float def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_FLOAT32) return g->kv[idx].value.f32;
    return def;
}

/* ---- Compute ops ---- */

static void vit_dequant_row(const qtensor *t, int row, float *dst) {
    int n_cols = t->n_cols;
    int block_size, type_size;
    switch (t->type) {
        case GGML_TYPE_Q8_0: block_size = 32;  type_size = 34;  break;
        case GGML_TYPE_Q4_K: block_size = 256; type_size = 144; break;
        case GGML_TYPE_Q6_K: block_size = 256; type_size = 210; break;
        case GGML_TYPE_F32:  block_size = 1;   type_size = 4;   break;
        case GGML_TYPE_F16:  block_size = 1;   type_size = 2;   break;
        default:
            memset(dst, 0, n_cols * sizeof(float));
            return;
    }
    size_t row_bytes = (size_t)((n_cols + block_size - 1) / block_size) * type_size;
    const void *row_data = (const uint8_t *)t->data + row * row_bytes;
    dequant_row(t->type, row_data, dst, n_cols);
}

/* Standard LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b */
static void vit_layernorm(float *dst, const float *x, const qtensor *w, const qtensor *b,
                          int n, float eps, float *w_buf, float *b_buf) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;

    float var = 0.0f;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;

    float inv_std = 1.0f / sqrtf(var + eps);

    vit_dequant_row(w, 0, w_buf);
    vit_dequant_row(b, 0, b_buf);

    for (int i = 0; i < n; i++)
        dst[i] = (x[i] - mean) * inv_std * w_buf[i] + b_buf[i];
}

/* GELU activation */
static void vit_gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        /* GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

/* Matrix-vector multiply with bias: dst[i] = sum_j(M[i][j] * x[j]) + bias[i] */
static void vit_matvec_bias(float *dst, const qtensor *mat, const qtensor *bias,
                            const float *x, int n_rows, float *tmp) {
    int n_cols = mat->n_cols;
    float *bias_buf = NULL;
    if (bias && bias->data) {
        bias_buf = (float *)malloc(n_rows * sizeof(float));
        vit_dequant_row(bias, 0, bias_buf);
    }
    /* F16 fast path: fused dequant + dot product */
    if (mat->type == GGML_TYPE_F16) {
        const uint8_t *base = (const uint8_t *)mat->data;
        size_t row_bytes = (size_t)n_cols * 2;
        for (int i = 0; i < n_rows; i++) {
            const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
            dst[i] = vec_dot_f16_f32(row, x, n_cols) + (bias_buf ? bias_buf[i] : 0.0f);
        }
        free(bias_buf);
        return;
    }
    for (int i = 0; i < n_rows; i++) {
        vit_dequant_row(mat, i, tmp);
        float sum = 0.0f;
        for (int j = 0; j < n_cols; j++) sum += tmp[j] * x[j];
        dst[i] = sum + (bias_buf ? bias_buf[i] : 0.0f);
    }
    free(bias_buf);
}

/* Batched matrix multiply for attention: process one patch vector at a time */
/* Matvec for batch of vectors: dst[b][i] = sum_j(M[i][j] * src[b][j]) + bias[i] */
static void vit_batch_matvec(float *dst, const qtensor *mat, const qtensor *bias,
                             const float *src, int n_tokens, int n_out, int n_in, float *tmp) {
    float *bias_buf = NULL;
    if (bias && bias->data) {
        bias_buf = (float *)malloc(n_out * sizeof(float));
        vit_dequant_row(bias, 0, bias_buf);
    }
    /* F16 fast path: fused dequant + dot product */
    if (mat->type == GGML_TYPE_F16) {
        const uint8_t *base = (const uint8_t *)mat->data;
        size_t row_bytes = (size_t)n_in * 2;
        for (int b = 0; b < n_tokens; b++) {
            const float *x = src + b * n_in;
            float *y = dst + b * n_out;
            for (int i = 0; i < n_out; i++) {
                const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
                y[i] = vec_dot_f16_f32(row, x, n_in) + (bias_buf ? bias_buf[i] : 0.0f);
            }
        }
        free(bias_buf);
        return;
    }
    for (int b = 0; b < n_tokens; b++) {
        const float *x = src + b * n_in;
        float *y = dst + b * n_out;
        for (int i = 0; i < n_out; i++) {
            vit_dequant_row(mat, i, tmp);
            float sum = 0.0f;
            for (int j = 0; j < n_in; j++) sum += tmp[j] * x[j];
            y[i] = sum + (bias_buf ? bias_buf[i] : 0.0f);
        }
    }
    free(bias_buf);
}

/* Softmax */
static void vit_softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* ---- Load ---- */

vision_model *vision_load(gguf_context *g) {
    if (!g) return NULL;

    vision_model *vm = (vision_model *)calloc(1, sizeof(vision_model));

    vm->n_blocks    = vit_get_int(g, "clip.vision.block_count", 24);
    vm->dim         = vit_get_int(g, "clip.vision.embedding_length", 1024);
    vm->n_heads     = vit_get_int(g, "clip.vision.attention.head_count", 16);
    vm->ffn_dim     = vit_get_int(g, "clip.vision.feed_forward_length", 4096);
    vm->patch_size  = vit_get_int(g, "clip.vision.patch_size", 16);
    vm->image_size  = vit_get_int(g, "clip.vision.image_size", 768);
    vm->proj_dim    = vit_get_int(g, "clip.vision.projection_dim", 2048);
    vm->spatial_merge = vit_get_int(g, "clip.vision.spatial_merge_size", 2);
    vm->ln_eps      = vit_get_float(g, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    vm->head_dim    = vm->dim / vm->n_heads;

    int ps = vm->patch_size;
    int gs = vm->image_size / ps;  /* grid size */
    vm->n_patches   = gs * gs;
    vm->n_merged    = vm->n_patches / (vm->spatial_merge * vm->spatial_merge);

    /* Image mean/std */
    int idx = gguf_find_key(g, "clip.vision.image_mean");
    if (idx >= 0) {
        float *d = (float *)g->kv[idx].value.arr.data;
        vm->image_mean[0] = d[0]; vm->image_mean[1] = d[1]; vm->image_mean[2] = d[2];
    }
    idx = gguf_find_key(g, "clip.vision.image_std");
    if (idx >= 0) {
        float *d = (float *)g->kv[idx].value.arr.data;
        vm->image_std[0] = d[0]; vm->image_std[1] = d[1]; vm->image_std[2] = d[2];
    }

    fprintf(stderr, "vision: dim=%d heads=%d blocks=%d ffn=%d patch=%d image=%d patches=%d merged=%d proj=%d\n",
            vm->dim, vm->n_heads, vm->n_blocks, vm->ffn_dim,
            vm->patch_size, vm->image_size, vm->n_patches, vm->n_merged, vm->proj_dim);

    /* Patch + position embeddings */
    vm->patch_embd_w = vit_load(g, "v.patch_embd.weight", 1);
    vm->patch_embd_w1 = vit_load(g, "v.patch_embd.weight.1", 0);  /* second conv kernel */
    vm->patch_embd_b = vit_load(g, "v.patch_embd.bias", 0);
    vm->position_embd = vit_load(g, "v.position_embd.weight", 1);
    if (vm->patch_embd_w1.data)
        fprintf(stderr, "vision: loaded dual conv2d patch embeddings\n");

    /* Blocks */
    vm->blocks = (vision_block *)calloc(vm->n_blocks, sizeof(vision_block));
    for (int l = 0; l < vm->n_blocks; l++) {
        char name[128];
        #define VL(field, suffix, req) \
            snprintf(name, sizeof(name), "v.blk.%d." suffix, l); \
            vm->blocks[l].field = vit_load(g, name, req);
        VL(attn_qkv_w, "attn_qkv.weight", 1)
        VL(attn_qkv_b, "attn_qkv.bias", 1)
        VL(attn_out_w, "attn_out.weight", 1)
        VL(attn_out_b, "attn_out.bias", 1)
        VL(ffn_up_w, "ffn_up.weight", 1)
        VL(ffn_up_b, "ffn_up.bias", 1)
        VL(ffn_down_w, "ffn_down.weight", 1)
        VL(ffn_down_b, "ffn_down.bias", 1)
        VL(ln1_w, "ln1.weight", 1)
        VL(ln1_b, "ln1.bias", 1)
        VL(ln2_w, "ln2.weight", 1)
        VL(ln2_b, "ln2.bias", 1)
        #undef VL
    }

    /* DeepStack */
    idx = gguf_find_key(g, "clip.vision.is_deepstack_layers");
    if (idx >= 0) {
        uint8_t *flags = (uint8_t *)g->kv[idx].value.arr.data;
        int n = (int)g->kv[idx].value.arr.n;
        /* Count */
        int ns = 0;
        for (int i = 0; i < n; i++) if (flags[i]) ns++;
        vm->n_deepstack = ns;
        vm->deepstack_indices = (int *)malloc(ns * sizeof(int));
        vm->deepstack = (vision_deepstack *)calloc(ns, sizeof(vision_deepstack));
        int si = 0;
        for (int i = 0; i < n; i++) {
            if (!flags[i]) continue;
            vm->deepstack_indices[si] = i;
            char name[128];
            #define DL(field, suffix, req) \
                snprintf(name, sizeof(name), "v.deepstack.%d." suffix, i); \
                vm->deepstack[si].field = vit_load(g, name, req);
            DL(fc1_w, "fc1.weight", 1)
            DL(fc1_b, "fc1.bias", 1)
            DL(fc2_w, "fc2.weight", 1)
            DL(fc2_b, "fc2.bias", 1)
            DL(norm_w, "norm.weight", 1)
            DL(norm_b, "norm.bias", 1)
            #undef DL
            si++;
        }
        fprintf(stderr, "vision: %d deepstack layers at:", ns);
        for (int i = 0; i < ns; i++) fprintf(stderr, " %d", vm->deepstack_indices[i]);
        fprintf(stderr, "\n");
    }

    /* Post LN */
    vm->post_ln_w = vit_load(g, "v.post_ln.weight", 1);
    vm->post_ln_b = vit_load(g, "v.post_ln.bias", 1);

    /* MM projection */
    vm->mm0_w = vit_load(g, "mm.0.weight", 1);
    vm->mm0_b = vit_load(g, "mm.0.bias", 1);
    vm->mm2_w = vit_load(g, "mm.2.weight", 1);
    vm->mm2_b = vit_load(g, "mm.2.bias", 1);

    return vm;
}

void vision_free(vision_model *vm) {
    if (!vm) return;
    free(vm->blocks);
    free(vm->deepstack_indices);
    free(vm->deepstack);
    free(vm);
}

float *vision_normalize_image(const vision_model *vm, const uint8_t *rgb, int width, int height) {
    float *out = (float *)malloc(width * height * 3 * sizeof(float));
    for (int i = 0; i < width * height; i++) {
        out[i * 3 + 0] = ((float)rgb[i * 3 + 0] / 255.0f - vm->image_mean[0]) / vm->image_std[0];
        out[i * 3 + 1] = ((float)rgb[i * 3 + 1] / 255.0f - vm->image_mean[1]) / vm->image_std[1];
        out[i * 3 + 2] = ((float)rgb[i * 3 + 2] / 255.0f - vm->image_mean[2]) / vm->image_std[2];
    }
    return out;
}

/* ---- Encode ---- */

float *vision_encode(vision_model *vm, const float *rgb_norm, int width, int height) {
    int ps = vm->patch_size;
    int dim = vm->dim;
    int n_heads = vm->n_heads;
    int head_dim = vm->head_dim;
    int ffn_dim = vm->ffn_dim;
    int gw = width / ps;
    int gh = height / ps;
    int n_patches = gw * gh;
    int sm = vm->spatial_merge;
    int merged_dim = dim * sm * sm;  /* 1024 * 4 = 4096 */
    int n_merged = n_patches / (sm * sm);

    if (n_patches > vm->n_patches) {
        fprintf(stderr, "vision: too many patches %d (max %d)\n", n_patches, vm->n_patches);
        return NULL;
    }

    fprintf(stderr, "vision: encoding %dx%d image (%d patches, %d merged tokens)\n",
            width, height, n_patches, n_merged);

    /* Allocate buffers */
    float *hidden = (float *)calloc(n_patches * dim, sizeof(float));   /* [n_patches, dim] */
    float *hidden2 = (float *)calloc(n_patches * dim, sizeof(float));
    float *qkv = (float *)calloc(n_patches * 3 * dim, sizeof(float));
    float *attn_out = (float *)calloc(n_patches * dim, sizeof(float));
    float *ffn_buf = (float *)calloc(n_patches * ffn_dim, sizeof(float));
    float *ln_buf = (float *)calloc(n_patches * dim, sizeof(float));
    float *att = (float *)calloc(n_patches * n_patches, sizeof(float)); /* one head at a time */
    float *tmp = (float *)calloc(ffn_dim > 3 * dim ? ffn_dim : 3 * dim, sizeof(float));
    float *w_buf = (float *)calloc(dim > ffn_dim ? dim : ffn_dim, sizeof(float));
    float *b_buf = (float *)calloc(dim > ffn_dim ? dim : ffn_dim, sizeof(float));
    int n_ds = vm->n_deepstack;
    int total_embd = vm->proj_dim * (1 + n_ds);  /* main + deepstack features */
    float *deepstack_feats = (float *)calloc(n_merged * n_ds * vm->proj_dim, sizeof(float));
    int ds_count = 0;  /* how many deepstack features computed so far */
    float *merge_buf = (float *)calloc(n_merged * merged_dim, sizeof(float));

    /* 1. Patch embedding: Dual Conv2D with stride=patch_size
     * Qwen3-VL uses two conv kernels (temporal_patch_size=2): both applied to the
     * same image and outputs added together, then bias added. */
    fprintf(stderr, "  patch embedding (dual conv2d)...\n");
    VIT_PROF_BEGIN("patch_embed", -1, "conv2d", "FP32");
    {
        int kernel_size = ps * ps * 3;
        float *kernel0 = (float *)malloc(dim * kernel_size * sizeof(float));
        dequant_row(vm->patch_embd_w.type, vm->patch_embd_w.data, kernel0,
                    ps * ps * 3 * dim);

        /* Second conv kernel (if available) */
        float *kernel1 = NULL;
        if (vm->patch_embd_w1.data) {
            kernel1 = (float *)malloc(dim * kernel_size * sizeof(float));
            dequant_row(vm->patch_embd_w1.type, vm->patch_embd_w1.data, kernel1,
                        ps * ps * 3 * dim);
        }

        float *bias_buf2 = NULL;
        if (vm->patch_embd_b.data) {
            bias_buf2 = (float *)malloc(dim * sizeof(float));
            vit_dequant_row(&vm->patch_embd_b, 0, bias_buf2);
        }

        for (int py = 0; py < gh; py++) {
            for (int px = 0; px < gw; px++) {
                int patch_idx = py * gw + px;
                float *out = hidden + patch_idx * dim;

                /* Extract patch pixels in CHW order */
                float patch[768]; /* 3*16*16 */
                for (int c = 0; c < 3; c++) {
                    for (int dy = 0; dy < ps; dy++) {
                        for (int dx = 0; dx < ps; dx++) {
                            int img_y = py * ps + dy;
                            int img_x = px * ps + dx;
                            int pix_idx = (img_y * width + img_x) * 3;
                            patch[c * ps * ps + dy * ps + dx] = rgb_norm[pix_idx + c];
                        }
                    }
                }

                /* Dot with each output channel: conv0 + conv1 + bias */
                for (int d = 0; d < dim; d++) {
                    float sum = 0.0f;
                    const float *kd0 = kernel0 + d * kernel_size;
                    for (int j = 0; j < kernel_size; j++) sum += kd0[j] * patch[j];
                    if (kernel1) {
                        const float *kd1 = kernel1 + d * kernel_size;
                        for (int j = 0; j < kernel_size; j++) sum += kd1[j] * patch[j];
                    }
                    out[d] = sum + (bias_buf2 ? bias_buf2[d] : 0.0f);
                }
            }
        }
        free(kernel0);
        free(kernel1);
        free(bias_buf2);
    }
    /* FLOPs: each patch does dim dot products of kernel_size, times 2 kernels */
    VIT_PROF_END("patch_embed", 2.0 * n_patches * dim * ps * ps * 3 * (vm->patch_embd_w1.data ? 2 : 1), 0);

    /* 2. Add position embeddings: position_embd is [dim, orig_n_patches] */
    fprintf(stderr, "  position embeddings...\n");
    {
        int orig_n = vm->n_patches;
        int orig_gw = vm->image_size / ps;
        float *pos_emb = (float *)malloc(dim * orig_n * sizeof(float));
        dequant_row(vm->position_embd.type, vm->position_embd.data, pos_emb,
                    dim * orig_n);
        /* pos_emb GGUF dims=[dim, n_patches], with dims[0]=dim contiguous.
         * So layout is [n_patches, dim] in row-major: pos_emb[p * dim + d].
         * For smaller images, select positions from the top-left subgrid. */
        for (int py = 0; py < gh; py++) {
            for (int px = 0; px < gw; px++) {
                int p = py * gw + px;
                int orig_p = py * orig_gw + px;  /* position in original grid */
                for (int d = 0; d < dim; d++) {
                    hidden[p * dim + d] += pos_emb[orig_p * dim + d];
                }
            }
        }
        free(pos_emb);
    }

    /* 2b. Compute M-RoPE position IDs for each patch (raster order).
     * Layout: pos_ids[4][n_patches] — (t, h, w, e) per patch.
     * Our patches are in raster order (py*gw+px), so position is simply (py, px).
     * llama.cpp uses pixel-unshuffle reordering but we skip that, so assign
     * positions directly from the raster grid coordinates. */
    int *pos_ids = (int *)calloc(4 * n_patches, sizeof(int));
    for (int py = 0; py < gh; py++) {
        for (int px = 0; px < gw; px++) {
            int p = py * gw + px;
            pos_ids[0 * n_patches + p] = py;  /* p_t */
            pos_ids[1 * n_patches + p] = px;  /* p_h */
            pos_ids[2 * n_patches + p] = py;  /* p_w */
            pos_ids[3 * n_patches + p] = px;  /* p_e */
        }
    }

    /* 2c. Precompute M-RoPE cos/sin cache for all patches.
     * Vision RoPE: theta_base=10000, independent sections, pairs [i, i+half].
     * Sections [s0, s1, s2, s3] = [hd/4, hd/4, hd/4, hd/4].
     * For d_head=64: 32 pairs, section 0 (pairs 0-15) uses p_t,
     * section 1 (pairs 16-31) uses p_h. */
    int half = head_dim / 2;  /* n_dims = d_head/2 */
    int sect_size = head_dim / 4;
    float freq_base = 10000.0f;
    /* theta_scale = base^(-2/n_dims) where n_dims = d_head/2 (RoPE dimension count) */
    float theta_scale = powf(freq_base, -2.0f / (float)half);
    float *rope_cos = (float *)malloc(n_patches * head_dim * sizeof(float));
    float *rope_sin = (float *)malloc(n_patches * head_dim * sizeof(float));
    for (int p = 0; p < n_patches; p++) {
        float p_t = (float)pos_ids[0 * n_patches + p];
        float p_h = (float)pos_ids[1 * n_patches + p];
        float p_w = (float)pos_ids[2 * n_patches + p];
        float p_e = (float)pos_ids[3 * n_patches + p];
        /* Four running thetas, one per section. All advance each iteration
         * (matching llama.cpp), but reset at section boundaries (indep_sects). */
        float cur_t = p_t, cur_h = p_h, cur_w = p_w, cur_e = p_e;
        for (int i0 = 0; i0 < head_dim; i0 += 2) {
            int sector = (i0 / 2);  /* 0..half-1 for d_head elements */
            /* Independent sections: reset theta at section boundaries */
            if (sector == 0) cur_t = p_t;
            if (sector == sect_size) cur_h = p_h;
            if (sector == 2 * sect_size) cur_w = p_w;
            if (sector == 3 * sect_size) cur_e = p_e;

            float theta;
            if (sector < sect_size) theta = cur_t;
            else if (sector < 2 * sect_size) theta = cur_h;
            else if (sector < 3 * sect_size) theta = cur_w;
            else theta = cur_e;

            rope_cos[p * head_dim + i0] = cosf(theta);
            rope_sin[p * head_dim + i0] = sinf(theta);
            rope_cos[p * head_dim + i0 + 1] = cosf(theta);
            rope_sin[p * head_dim + i0 + 1] = sinf(theta);

            cur_t *= theta_scale;
            cur_h *= theta_scale;
            cur_w *= theta_scale;
            cur_e *= theta_scale;
        }
    }

    /* 3. ViT blocks */
    for (int l = 0; l < vm->n_blocks; l++) {
        if (l == 0 || l == vm->n_blocks - 1 || (l + 1) % 6 == 0)
            fprintf(stderr, "  vit block %d/%d\n", l, vm->n_blocks);

        vision_block *blk = &vm->blocks[l];

        /* --- Self-attention --- */
        /* LayerNorm1 */
        VIT_PROF_BEGIN("ln1", l, "layernorm", "FP32");
        for (int p = 0; p < n_patches; p++) {
            vit_layernorm(ln_buf + p * dim, hidden + p * dim,
                          &blk->ln1_w, &blk->ln1_b, dim, vm->ln_eps, w_buf, b_buf);
        }
        VIT_PROF_END("ln1", 5.0 * n_patches * dim, 0);

        /* QKV projection: [n_patches, dim] -> [n_patches, 3*dim] */
        VIT_PROF_BEGIN("qkv_matmul", l, "matmul", "FP32");
        vit_batch_matvec(qkv, &blk->attn_qkv_w, &blk->attn_qkv_b,
                         ln_buf, n_patches, 3 * dim, dim, tmp);
        VIT_PROF_END("qkv_matmul", 2.0 * n_patches * 3 * dim * dim, 0);

        /* Apply M-RoPE to Q and K (not V).
         * Vision rotation: pairs [i, i+half] for i in 0..half-1.
         * cos/sin indexed by pair i0/2 from precomputed cache. */
        VIT_PROF_BEGIN("mrope", l, "rope", "FP32");
        for (int p = 0; p < n_patches; p++) {
            float *q = qkv + p * 3 * dim;      /* Q starts at offset 0 */
            float *k = qkv + p * 3 * dim + dim; /* K starts at offset dim */
            for (int h = 0; h < n_heads; h++) {
                float *qh = q + h * head_dim;
                float *kh = k + h * head_dim;
                for (int i = 0; i < half; i++) {
                    float cos_t = rope_cos[p * head_dim + 2 * i];
                    float sin_t = rope_sin[p * head_dim + 2 * i];
                    /* Rotate Q */
                    float q0 = qh[i], q1 = qh[i + half];
                    qh[i]        = q0 * cos_t - q1 * sin_t;
                    qh[i + half] = q0 * sin_t + q1 * cos_t;
                    /* Rotate K */
                    float k0 = kh[i], k1 = kh[i + half];
                    kh[i]        = k0 * cos_t - k1 * sin_t;
                    kh[i + half] = k0 * sin_t + k1 * cos_t;
                }
            }
        }

        VIT_PROF_END("mrope", (double)n_patches * n_heads * half * 8.0 * 2, 0);

        /* Multi-head attention (head by head to save memory) */
        VIT_PROF_BEGIN("attention", l, "attention", "FP32");
        memset(attn_out, 0, n_patches * dim * sizeof(float));
        float scale = 1.0f / sqrtf((float)head_dim);

        for (int h = 0; h < n_heads; h++) {
            /* Extract Q, K for this head and compute attention scores */
            for (int qi = 0; qi < n_patches; qi++) {
                const float *q_h = qkv + qi * 3 * dim + h * head_dim;  /* Q */
                for (int ki = 0; ki < n_patches; ki++) {
                    const float *k_h = qkv + ki * 3 * dim + dim + h * head_dim;  /* K */
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) score += q_h[d] * k_h[d];
                    att[qi * n_patches + ki] = score * scale;
                }
            }

            /* Softmax per query */
            for (int qi = 0; qi < n_patches; qi++) {
                vit_softmax(att + qi * n_patches, n_patches);
            }

            /* Weighted sum of V */
            for (int qi = 0; qi < n_patches; qi++) {
                float *out_h = attn_out + qi * dim + h * head_dim;
                for (int vi = 0; vi < n_patches; vi++) {
                    const float *v_h = qkv + vi * 3 * dim + 2 * dim + h * head_dim;
                    float a = att[qi * n_patches + vi];
                    for (int d = 0; d < head_dim; d++) out_h[d] += a * v_h[d];
                }
            }
        }

        /* QK: heads*seq^2*hd, AV: same */
        VIT_PROF_END("attention", 2.0 * n_heads * (double)n_patches * n_patches * head_dim * 2, 0);

        /* Attention output projection */
        VIT_PROF_BEGIN("attn_out", l, "matmul", "FP32");
        vit_batch_matvec(hidden2, &blk->attn_out_w, &blk->attn_out_b,
                         attn_out, n_patches, dim, dim, tmp);
        VIT_PROF_END("attn_out", 2.0 * n_patches * dim * dim, 0);

        /* Residual */
        for (int i = 0; i < n_patches * dim; i++) hidden[i] += hidden2[i];

        /* --- FFN --- */
        /* LayerNorm2 */
        VIT_PROF_BEGIN("ln2", l, "layernorm", "FP32");
        for (int p = 0; p < n_patches; p++) {
            vit_layernorm(ln_buf + p * dim, hidden + p * dim,
                          &blk->ln2_w, &blk->ln2_b, dim, vm->ln_eps, w_buf, b_buf);
        }
        VIT_PROF_END("ln2", 5.0 * n_patches * dim, 0);

        /* FFN: up -> GELU -> down */
        VIT_PROF_BEGIN("ffn_up", l, "matmul", "FP32");
        vit_batch_matvec(ffn_buf, &blk->ffn_up_w, &blk->ffn_up_b,
                         ln_buf, n_patches, ffn_dim, dim, tmp);
        VIT_PROF_END("ffn_up", 2.0 * n_patches * ffn_dim * dim, 0);

        VIT_PROF_BEGIN("gelu", l, "gelu", "FP32");
        for (int i = 0; i < n_patches * ffn_dim; i++) {
            float v = ffn_buf[i];
            ffn_buf[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        }
        VIT_PROF_END("gelu", 8.0 * n_patches * ffn_dim, 0);

        VIT_PROF_BEGIN("ffn_down", l, "matmul", "FP32");
        vit_batch_matvec(hidden2, &blk->ffn_down_w, &blk->ffn_down_b,
                         ffn_buf, n_patches, dim, ffn_dim, tmp);
        VIT_PROF_END("ffn_down", 2.0 * n_patches * dim * ffn_dim, 0);

        /* Residual */
        for (int i = 0; i < n_patches * dim; i++) hidden[i] += hidden2[i];

        /* --- DeepStack --- */
        for (int ds = 0; ds < vm->n_deepstack; ds++) {
            if (vm->deepstack_indices[ds] != l) continue;

            fprintf(stderr, "  deepstack at layer %d\n", l);
            vision_deepstack *dsl = &vm->deepstack[ds];

            /* Spatial merge current hidden [gh, gw, dim] -> [gh/sm, gw/sm, dim*sm*sm] */
            int mgw = gw / sm, mgh = gh / sm;
            for (int my = 0; my < mgh; my++) {
                for (int mx = 0; mx < mgw; mx++) {
                    float *dst = merge_buf + (my * mgw + mx) * merged_dim;
                    int di = 0;
                    for (int sy = 0; sy < sm; sy++) {
                        for (int sx = 0; sx < sm; sx++) {
                            int py2 = my * sm + sy;
                            int px2 = mx * sm + sx;
                            const float *src = hidden + (py2 * gw + px2) * dim;
                            memcpy(dst + di, src, dim * sizeof(float));
                            di += dim;
                        }
                    }
                }
            }

            /* LayerNorm on merge_buf (before fc1) */
            float *ds_w = (float *)malloc(merged_dim * sizeof(float));
            float *ds_b = (float *)malloc(merged_dim * sizeof(float));
            for (int p = 0; p < n_merged; p++) {
                vit_layernorm(merge_buf + p * merged_dim, merge_buf + p * merged_dim,
                              &dsl->norm_w, &dsl->norm_b, merged_dim, vm->ln_eps, ds_w, ds_b);
            }

            /* fc1: [merged_dim -> merged_dim] with bias */
            float *ds_tmp = (float *)malloc(merged_dim * sizeof(float));
            float *ds_buf = (float *)malloc(n_merged * merged_dim * sizeof(float));
            VIT_PROF_BEGIN("ds_fc1", l, "matmul", "FP32");
            vit_batch_matvec(ds_buf, &dsl->fc1_w, &dsl->fc1_b,
                             merge_buf, n_merged, merged_dim, merged_dim, ds_tmp);
            VIT_PROF_END("ds_fc1", 2.0 * n_merged * merged_dim * merged_dim, 0);

            /* GELU */
            VIT_PROF_BEGIN("ds_gelu", l, "gelu", "FP32");
            vit_gelu(ds_buf, n_merged * merged_dim);
            VIT_PROF_END("ds_gelu", 8.0 * n_merged * merged_dim, 0);

            /* fc2: [merged_dim -> proj_dim] */
            float *ds_out = (float *)malloc(n_merged * vm->proj_dim * sizeof(float));
            VIT_PROF_BEGIN("ds_fc2", l, "matmul", "FP32");
            vit_batch_matvec(ds_out, &dsl->fc2_w, &dsl->fc2_b,
                             ds_buf, n_merged, vm->proj_dim, merged_dim, ds_tmp);
            VIT_PROF_END("ds_fc2", 2.0 * n_merged * vm->proj_dim * merged_dim, 0);

            /* Store as separate feature slice (for concat) */
            memcpy(deepstack_feats + ds_count * n_merged * vm->proj_dim,
                   ds_out, n_merged * vm->proj_dim * sizeof(float));
            ds_count++;

            free(ds_tmp);
            free(ds_buf);
            free(ds_w);
            free(ds_b);
            free(ds_out);
        }
    }

    /* 4. Post LayerNorm */
    fprintf(stderr, "  post layernorm...\n");
    VIT_PROF_BEGIN("post_ln", -1, "layernorm", "FP32");
    for (int p = 0; p < n_patches; p++) {
        vit_layernorm(hidden + p * dim, hidden + p * dim,
                      &vm->post_ln_w, &vm->post_ln_b, dim, vm->ln_eps, w_buf, b_buf);
    }
    VIT_PROF_END("post_ln", 5.0 * n_patches * dim, 0);

    /* 5. Spatial merge: [gh, gw, dim] -> [gh/sm, gw/sm, dim*sm*sm] */
    fprintf(stderr, "  spatial merge...\n");
    {
        int mgw = gw / sm, mgh = gh / sm;
        for (int my = 0; my < mgh; my++) {
            for (int mx = 0; mx < mgw; mx++) {
                float *dst = merge_buf + (my * mgw + mx) * merged_dim;
                int di = 0;
                for (int sy = 0; sy < sm; sy++) {
                    for (int sx = 0; sx < sm; sx++) {
                        int py2 = my * sm + sy;
                        int px2 = mx * sm + sx;
                        const float *src = hidden + (py2 * gw + px2) * dim;
                        memcpy(dst + di, src, dim * sizeof(float));
                        di += dim;
                    }
                }
            }
        }
    }

    /* 6. MM projection: mm.0 -> GELU -> mm.2 */
    fprintf(stderr, "  mm projection...\n");
    float *mm_tmp = (float *)malloc(merged_dim * sizeof(float));
    float *mm_buf = (float *)malloc(n_merged * merged_dim * sizeof(float));
    /* Result: [n_merged, proj_dim * (1 + n_deepstack)] - main concat deepstack features */
    float *result = (float *)calloc(n_merged * total_embd, sizeof(float));
    float *mm_out = (float *)malloc(n_merged * vm->proj_dim * sizeof(float));

    VIT_PROF_BEGIN("mm0", -1, "matmul", "FP32");
    vit_batch_matvec(mm_buf, &vm->mm0_w, &vm->mm0_b,
                     merge_buf, n_merged, merged_dim, merged_dim, mm_tmp);
    VIT_PROF_END("mm0", 2.0 * n_merged * merged_dim * merged_dim, 0);

    VIT_PROF_BEGIN("mm_gelu", -1, "gelu", "FP32");
    vit_gelu(mm_buf, n_merged * merged_dim);
    VIT_PROF_END("mm_gelu", 8.0 * n_merged * merged_dim, 0);

    VIT_PROF_BEGIN("mm2", -1, "matmul", "FP32");
    vit_batch_matvec(mm_out, &vm->mm2_w, &vm->mm2_b,
                     mm_buf, n_merged, vm->proj_dim, merged_dim, mm_tmp);
    VIT_PROF_END("mm2", 2.0 * n_merged * vm->proj_dim * merged_dim, 0);

    /* 7. Concat main embeddings + deepstack features per token
     * Layout: [main_proj_dim, ds0_proj_dim, ds1_proj_dim, ...] for each token
     * This matches llama.cpp Qwen3-VL: LLM layers 0..n_ds-1 each add the
     * corresponding deepstack slice to their output. */
    for (int t = 0; t < n_merged; t++) {
        float *dst = result + t * total_embd;
        memcpy(dst, mm_out + t * vm->proj_dim, vm->proj_dim * sizeof(float));
        for (int d = 0; d < ds_count; d++) {
            memcpy(dst + (1 + d) * vm->proj_dim,
                   deepstack_feats + d * n_merged * vm->proj_dim + t * vm->proj_dim,
                   vm->proj_dim * sizeof(float));
        }
    }

    fprintf(stderr, "  vision encoding done: %d tokens of dim %d (main %d + %d deepstack)\n",
            n_merged, total_embd, vm->proj_dim, ds_count);

    /* Cleanup */
    free(hidden); free(hidden2); free(qkv); free(attn_out);
    free(ffn_buf); free(ln_buf); free(att); free(tmp);
    free(w_buf); free(b_buf); free(deepstack_feats); free(merge_buf);
    free(pos_ids); free(rope_cos); free(rope_sin);
    free(mm_tmp); free(mm_buf); free(mm_out);

    return result;
}

#endif /* VISION_ENCODER_IMPLEMENTATION */
#endif /* VISION_ENCODER_H */
