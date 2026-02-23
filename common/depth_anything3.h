/*
 * depth_anything3.h - Depth-Anything-3 monocular depth estimation (all sizes)
 *
 * Usage:
 *   #define DEPTH_ANYTHING3_IMPLEMENTATION
 *   #include "depth_anything3.h"
 *
 * Dependencies: gguf_loader.h, ggml_dequant.h, safetensors.h (optional)
 *
 * API:
 *   da3_model      *da3_load(gguf_context *gguf);
 *   da3_model      *da3_load_safetensors(const char *st_path, const char *config_path);
 *   void            da3_free(da3_model *m);
 *   da3_result      da3_predict(da3_model *m, const uint8_t *rgb, int w, int h, int n_threads);
 *   da3_full_result da3_predict_full(da3_model *m, const uint8_t *rgb, int w, int h,
 *                                    int n_threads, int output_flags);
 *   void            da3_result_free(da3_result *r);
 *   void            da3_full_result_free(da3_full_result *r);
 */
#ifndef DEPTH_ANYTHING3_H
#define DEPTH_ANYTHING3_H

#include <stdint.h>
#include <stddef.h>
#include "gguf_loader.h"
#include "ggml_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Reuse qtensor from transformer.h if available, else define it */
#ifndef TRANSFORMER_H
typedef struct {
    void    *data;
    uint32_t type;
    int      n_rows;
    int      n_cols;
    int      n_dims;
    uint64_t dims[4];
} qtensor;
#endif

typedef struct {
    qtensor ln1_w, ln1_b;
    qtensor attn_qkv_w, attn_qkv_b;
    qtensor attn_q_norm_w, attn_q_norm_b;
    qtensor attn_k_norm_w, attn_k_norm_b;
    qtensor attn_out_w, attn_out_b;
    qtensor ls1;
    qtensor ln2_w, ln2_b;
    qtensor ffn_gate_up_w, ffn_gate_up_b;  /* SwiGLU: [2*hidden, dim] */
    qtensor ffn_up_w, ffn_up_b;            /* GELU:   [ffn_dim, dim]  */
    qtensor ffn_down_w, ffn_down_b;
    qtensor ls2;
} da3_block;

typedef struct {
    qtensor norm_w, norm_b;
    qtensor proj_w[4], proj_b[4];
    qtensor upsample_0_w, upsample_0_b;   /* ConvTranspose2d 4x4 s4 */
    qtensor upsample_1_w, upsample_1_b;   /* ConvTranspose2d 2x2 s2 */
    qtensor downsample_w, downsample_b;    /* Conv2d 3x3 s2 p1 */
    qtensor adapter_w[4];                  /* Conv2d 3x3 p1, no bias */
    /* RefineNet fusion blocks */
    qtensor fuse_out_w[4], fuse_out_b[4];
    qtensor fuse_rcu1_c1_w[4], fuse_rcu1_c1_b[4];
    qtensor fuse_rcu1_c2_w[4], fuse_rcu1_c2_b[4];
    qtensor fuse_rcu2_c1_w[4], fuse_rcu2_c1_b[4];
    qtensor fuse_rcu2_c2_w[4], fuse_rcu2_c2_b[4];
    /* Output convolutions */
    qtensor neck_w, neck_b;
    qtensor out_0_w, out_0_b;
    qtensor out_2_w, out_2_b;
} da3_dpt_head;

/* CameraDec: backbone_norm(CLS) → MLP → 3 linear heads → pose[9] */
typedef struct {
    qtensor mlp0_w, mlp0_b;       /* [mlp_dim, dim] */
    qtensor mlp2_w, mlp2_b;       /* [mlp_dim, mlp_dim] */
    qtensor fc_t_w, fc_t_b;       /* [3, mlp_dim] */
    qtensor fc_qvec_w, fc_qvec_b; /* [4, mlp_dim] */
    qtensor fc_fov_w, fc_fov_b;   /* [2, mlp_dim] */
    int mlp_dim;
} da3_cam_dec;

/* Aux DPT: rays[6] + ray_confidence[1] */
typedef struct {
    /* RefineNet fusion (same structure as main DPT) */
    qtensor fuse_out_w[4], fuse_out_b[4];
    qtensor fuse_rcu1_c1_w[4], fuse_rcu1_c1_b[4];
    qtensor fuse_rcu1_c2_w[4], fuse_rcu1_c2_b[4];
    qtensor fuse_rcu2_c1_w[4], fuse_rcu2_c1_b[4];
    qtensor fuse_rcu2_c2_w[4], fuse_rcu2_c2_b[4];
    int has_rcu1[4], has_rcu2[4];
    /* output_conv1_aux: up to 5 Conv3x3 layers per level (F32, no activations) */
    qtensor oc1_w[4][5], oc1_b[4][5];
    int oc1_ci[4][5], oc1_co[4][5];
    int oc1_count[4];
    /* output_conv2_aux: conv + channel_layernorm + relu + out_conv */
    qtensor oc2_conv_w[4], oc2_conv_b[4];
    qtensor oc2_gn_w[4], oc2_gn_b[4];
    qtensor oc2_out_w[4], oc2_out_b[4];
} da3_aux_head;

/* GSDPT: 38ch gaussians */
typedef struct {
    da3_dpt_head head;              /* reuse identical DPT structure */
    qtensor merger_w[3], merger_b[3];  /* 3 stride-2 Conv2d layers */
    int merger_ci[3], merger_co[3];
    int gs_out_channels;            /* 38 */
} da3_gsdpt;

/* Output flags for da3_predict_full() */
#define DA3_OUTPUT_DEPTH      0x01
#define DA3_OUTPUT_POSE       0x02
#define DA3_OUTPUT_RAYS       0x04
#define DA3_OUTPUT_GAUSSIANS  0x08
#define DA3_OUTPUT_ALL        0x0F

typedef struct {
    int n_blocks, dim, n_heads, head_dim, ffn_hidden;
    int patch_size, image_size, grid_h, grid_w, n_patches, n_tokens;
    float ln_eps;
    float image_mean[3], image_std[3];
    int rope_start_layer, qk_norm_start_layer;
    int feature_layers[4];
    int use_swiglu;
    int head_features;
    int head_out_channels[4];

    qtensor patch_embed_w, patch_embed_b;
    qtensor cls_token, pos_embed;

    da3_block *blocks;
    da3_dpt_head head;

    /* Full output modules (loaded from safetensors) */
    qtensor backbone_norm_w, backbone_norm_b;
    da3_cam_dec cam_dec;
    da3_aux_head aux_head;
    da3_gsdpt gsdpt;
    int has_cam_dec, has_aux, has_gsdpt;

    /* Safetensors context (kept alive for mmap'd weights) */
    void *st_ctx;
} da3_model;

typedef struct {
    float *depth;
    float *confidence;
    int width, height;
} da3_result;

typedef struct {
    float *depth, *confidence;     /* main DPT output */
    float *rays;                   /* [6, H, W] ray directions */
    float *ray_confidence;         /* [H, W] */
    float pose[9];                 /* [t(3), qvec(4), fov(2)] */
    float *gaussians;              /* [38, H, W] */
    int width, height;
    int has_pose, has_rays, has_gaussians;
} da3_full_result;

da3_model      *da3_load(gguf_context *gguf);
da3_model      *da3_load_safetensors(const char *st_path, const char *config_path);
void            da3_free(da3_model *m);
da3_result      da3_predict(da3_model *m, const uint8_t *rgb, int w, int h, int n_threads);
da3_full_result da3_predict_full(da3_model *m, const uint8_t *rgb, int w, int h,
                                 int n_threads, int output_flags);
void            da3_result_free(da3_result *r);
void            da3_full_result_free(da3_full_result *r);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef DEPTH_ANYTHING3_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

static double da3_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- Tensor loading helpers ---- */

static int da3_find_tensor(const gguf_context *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name.str, name) == 0) return (int)i;
    return -1;
}

static qtensor da3_load_tensor(const gguf_context *g, const char *name, int req) {
    qtensor t = {0};
    int idx = da3_find_tensor(g, name);
    if (idx < 0) {
        if (req) fprintf(stderr, "da3: missing tensor '%s'\n", name);
        return t;
    }
    t.data = gguf_tensor_data(g, idx);
    t.type = g->tensors[idx].type;
    t.n_dims = (int)g->tensors[idx].n_dims;
    for (int d = 0; d < t.n_dims; d++) t.dims[d] = g->tensors[idx].dims[d];
    /* Set n_rows/n_cols for GEMM: n_rows=outer dim (Co), n_cols=inner (Ci or Ci*kH*kW).
     * GGML 4D dims are [kW, kH, Ci, Co] (reversed from PyTorch [Co, Ci, kH, kW]). */
    if (t.n_dims == 4) {
        t.n_rows = (int)g->tensors[idx].dims[3];  /* Co */
        t.n_cols = (int)(g->tensors[idx].dims[0] * g->tensors[idx].dims[1]
                       * g->tensors[idx].dims[2]); /* kW*kH*Ci */
    } else if (t.n_dims >= 2) {
        t.n_cols = (int)g->tensors[idx].dims[0];
        t.n_rows = (int)g->tensors[idx].dims[1];
    } else {
        t.n_cols = (int)g->tensors[idx].dims[0];
        t.n_rows = 1;
    }
    return t;
}

static int da3_get_int(const gguf_context *g, const char *key, int def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_UINT32) return (int)g->kv[idx].value.u32;
    if (g->kv[idx].type == GGUF_TYPE_INT32) return g->kv[idx].value.i32;
    return def;
}

static float da3_get_float(const gguf_context *g, const char *key, float def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_FLOAT32) return g->kv[idx].value.f32;
    return def;
}

/* ---- Dequantization helpers ---- */

static int da3_tensor_numel(const qtensor *t) {
    if (!t->data) return 0;
    int n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= (int)t->dims[d];
    return n;
}

/* Dequantize entire tensor to F32. Caller must free. Returns NULL if no data. */
static float *da3_dequant(const qtensor *t) {
    if (!t->data) return NULL;
    int n = da3_tensor_numel(t);
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else if (t->type == GGML_TYPE_F32) {
        memcpy(buf, t->data, (size_t)n * sizeof(float));
    } else {
        /* Row-based dequant for quantized types */
        int bs, ts;
        switch (t->type) {
            case GGML_TYPE_Q8_0: bs = 32; ts = 34; break;
            case GGML_TYPE_Q4_K: bs = 256; ts = 144; break;
            case GGML_TYPE_Q6_K: bs = 256; ts = 210; break;
            default: memset(buf, 0, (size_t)n * sizeof(float)); return buf;
        }
        size_t row_bytes = (size_t)((t->n_cols + bs - 1) / bs) * ts;
        for (int r = 0; r < t->n_rows; r++) {
            const void *row = (const uint8_t *)t->data + r * row_bytes;
            dequant_row(t->type, row, buf + r * t->n_cols, t->n_cols);
        }
    }
    return buf;
}

/* Dequant single row (1D tensor or specific row of 2D) */
static void da3_dequant_row(const qtensor *t, int row, float *dst) {
    int n = t->n_cols;
    if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data + (size_t)row * n;
        for (int i = 0; i < n; i++) dst[i] = ggml_fp16_to_fp32(src[i]);
    } else if (t->type == GGML_TYPE_F32) {
        memcpy(dst, (const float *)t->data + (size_t)row * n, (size_t)n * sizeof(float));
    } else {
        int bs, ts;
        switch (t->type) {
            case GGML_TYPE_Q8_0: bs = 32; ts = 34; break;
            case GGML_TYPE_Q4_K: bs = 256; ts = 144; break;
            case GGML_TYPE_Q6_K: bs = 256; ts = 210; break;
            default: memset(dst, 0, (size_t)n * sizeof(float)); return;
        }
        size_t row_bytes = (size_t)((n + bs - 1) / bs) * ts;
        dequant_row(t->type, (const uint8_t *)t->data + row * row_bytes, dst, n);
    }
}

/* ---- Batch GEMM: dst[tok][out] = W[out][in] * src[tok][in] + bias[out] ---- */

/* Thread worker for row-parallel F16 GEMM */
typedef struct {
    float *dst;
    const uint16_t *W;
    const float *src;
    int n_out, n_in, n_tok;
    int r_start, r_end;
} da3_gemm_task;

static void *da3_gemm_worker(void *arg) {
    da3_gemm_task *t = (da3_gemm_task *)arg;
    int count = t->r_end - t->r_start;
    if (count <= 0) return NULL;
    /* Row-parallel: each thread computes a subset of output rows.
     * dst is offset by r_start so that gemm writes to the correct positions
     * in the token-major layout: dst[tok * n_out + r_start + local_r] */
    gemm_f16_f32_tokmajor(t->dst + t->r_start,
                           t->W + (size_t)t->r_start * t->n_in,
                           t->src,
                           count, t->n_in, t->n_tok,
                           t->n_out, t->n_in);
    return NULL;
}

static void da3_batch_gemm(float *dst, const qtensor *W, const qtensor *bias,
                           const float *src, int n_tok, int n_out, int n_in,
                           int n_threads) {
    if (W->type == GGML_TYPE_F16) {
        if (n_threads > 1 && n_out >= n_threads * 3) {
            /* Row-parallel multi-threaded GEMM */
            da3_gemm_task *tasks = (da3_gemm_task *)calloc((size_t)n_threads, sizeof(da3_gemm_task));
            pthread_t *threads = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
            /* Round rows per thread to multiple of 3 for the 3-row tiling kernel */
            int rows_per = ((n_out / n_threads) / 3) * 3;
            if (rows_per < 3) rows_per = 3;
            int r = 0;
            int actual_threads = 0;
            for (int i = 0; i < n_threads && r < n_out; i++) {
                int end = (i == n_threads - 1) ? n_out : r + rows_per;
                if (end > n_out) end = n_out;
                tasks[i] = (da3_gemm_task){dst, (const uint16_t *)W->data, src,
                                            n_out, n_in, n_tok, r, end};
                pthread_create(&threads[i], NULL, da3_gemm_worker, &tasks[i]);
                r = end;
                actual_threads = i + 1;
            }
            for (int i = 0; i < actual_threads; i++) pthread_join(threads[i], NULL);
            free(tasks); free(threads);
        } else {
            gemm_f16_f32_tokmajor(dst, (const uint16_t *)W->data, src,
                                  n_out, n_in, n_tok, n_out, n_in);
        }
    } else {
        float *tmp = (float *)malloc((size_t)n_in * sizeof(float));
        for (int t = 0; t < n_tok; t++) {
            for (int r = 0; r < n_out; r++) {
                da3_dequant_row(W, r, tmp);
                float s = 0.0f;
                for (int j = 0; j < n_in; j++) s += tmp[j] * src[t * n_in + j];
                dst[t * n_out + r] = s;
            }
        }
        free(tmp);
    }
    if (bias && bias->data) {
        float *b = (float *)malloc((size_t)n_out * sizeof(float));
        da3_dequant_row(bias, 0, b);
        for (int t = 0; t < n_tok; t++)
            for (int i = 0; i < n_out; i++)
                dst[t * n_out + i] += b[i];
        free(b);
    }
}

/* ---- LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b ---- */

static void da3_layernorm_batch(float *dst, const float *src, const qtensor *w,
                                const qtensor *b, int n_tok, int dim, float eps) {
    float *wf = (float *)malloc((size_t)dim * sizeof(float));
    float *bf = (float *)malloc((size_t)dim * sizeof(float));
    da3_dequant_row(w, 0, wf);
    da3_dequant_row(b, 0, bf);
    for (int t = 0; t < n_tok; t++) {
        const float *x = src + t * dim;
        float *y = dst + t * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= dim;
        float var = 0.0f;
        for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
        var /= dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++)
            y[i] = (x[i] - mean) * inv * wf[i] + bf[i];
    }
    free(wf); free(bf);
}

/* ---- QK Normalization: per-head LayerNorm on Q/K ---- */

static void da3_qk_norm_batch(float *vec, int n_tok, int n_heads, int head_dim,
                               const qtensor *nw, const qtensor *nb, float eps) {
    if (!nw->data) return;
    float *w = (float *)malloc((size_t)head_dim * sizeof(float));
    float *b = (float *)malloc((size_t)head_dim * sizeof(float));
    da3_dequant_row(nw, 0, w);
    da3_dequant_row(nb, 0, b);
    int dim = n_heads * head_dim;
    for (int t = 0; t < n_tok; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *v = vec + t * dim + h * head_dim;
            float mean = 0.0f;
            for (int i = 0; i < head_dim; i++) mean += v[i];
            mean /= head_dim;
            float var = 0.0f;
            for (int i = 0; i < head_dim; i++) { float d = v[i] - mean; var += d * d; }
            var /= head_dim;
            float s = 1.0f / sqrtf(var + eps);
            for (int i = 0; i < head_dim; i++)
                v[i] = (v[i] - mean) * s * w[i] + b[i];
        }
    }
    free(w); free(b);
}

/* ---- RoPE 2D: split head_dim in half for y/x positions ---- */

static void da3_rope_2d_batch(float *vec, int n_tok, int n_heads, int head_dim,
                               const int *pos_y, const int *pos_x, float freq_base) {
    int half = head_dim / 2;
    int quarter = half / 2;
    int dim = n_heads * head_dim;
    for (int t = 0; t < n_tok; t++) {
        float py = (float)pos_y[t];
        float px = (float)pos_x[t];
        for (int h = 0; h < n_heads; h++) {
            float *v = vec + t * dim + h * head_dim;
            /* Y rotation: first half */
            for (int j = 0; j < quarter; j++) {
                float freq = 1.0f / powf(freq_base, (float)(2 * j) / (float)half);
                float theta = py * freq;
                float c = cosf(theta), s = sinf(theta);
                float v0 = v[j], v1 = v[j + quarter];
                v[j]           = v0 * c - v1 * s;
                v[j + quarter] = v0 * s + v1 * c;
            }
            /* X rotation: second half */
            for (int j = 0; j < quarter; j++) {
                float freq = 1.0f / powf(freq_base, (float)(2 * j) / (float)half);
                float theta = px * freq;
                float c = cosf(theta), s = sinf(theta);
                float v0 = v[half + j], v1 = v[half + j + quarter];
                v[half + j]           = v0 * c - v1 * s;
                v[half + j + quarter] = v0 * s + v1 * c;
            }
        }
    }
}

/* ---- LayerScale: out[i] = x[i] * gamma[i] ---- */

static void da3_layerscale(float *x, const qtensor *gamma, int n_tok, int dim) {
    if (!gamma->data) return;
    float *g = (float *)malloc((size_t)dim * sizeof(float));
    da3_dequant_row(gamma, 0, g);
    for (int t = 0; t < n_tok; t++) {
        float *v = x + t * dim;
        for (int i = 0; i < dim; i++) v[i] *= g[i];
    }
    free(g);
}

/* ---- Softmax ---- */

static void da3_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ---- Multi-head attention (full sequence) ---- */

typedef struct {
    const float *qkv;
    float *attn_out;
    int n_tok, dim, head_dim, n_heads;
    int h_start, h_end;
    float scale;
} da3_attn_task;

static void *da3_attn_worker(void *arg) {
    da3_attn_task *t = (da3_attn_task *)arg;
    int N = t->n_tok, hd = t->head_dim, dim3 = 3 * t->dim;
    float *att = (float *)malloc((size_t)N * sizeof(float));

    for (int h = t->h_start; h < t->h_end; h++) {
        for (int qi = 0; qi < N; qi++) {
            const float *q_h = t->qkv + qi * dim3 + h * hd;
            /* Compute QK^T scores */
            for (int ki = 0; ki < N; ki++) {
                const float *k_h = t->qkv + ki * dim3 + t->dim + h * hd;
                float dot = 0.0f;
                for (int d = 0; d < hd; d++) dot += q_h[d] * k_h[d];
                att[ki] = dot * t->scale;
            }
            da3_softmax(att, N);
            /* Weighted sum of V */
            float *out_h = t->attn_out + qi * t->dim + h * hd;
            memset(out_h, 0, (size_t)hd * sizeof(float));
            for (int vi = 0; vi < N; vi++) {
                const float *v_h = t->qkv + vi * dim3 + 2 * t->dim + h * hd;
                float w = att[vi];
                for (int d = 0; d < hd; d++) out_h[d] += w * v_h[d];
            }
        }
    }
    free(att);
    return NULL;
}

static void da3_attention(float *out, const float *qkv, int n_tok, int dim,
                          int n_heads, int head_dim, int n_threads) {
    float scale = 1.0f / sqrtf((float)head_dim);

    if (n_threads <= 1 || n_heads < n_threads) {
        da3_attn_task task = {qkv, out, n_tok, dim, head_dim, n_heads,
                              0, n_heads, scale};
        da3_attn_worker(&task);
        return;
    }

    da3_attn_task *tasks = (da3_attn_task *)calloc((size_t)n_threads, sizeof(da3_attn_task));
    pthread_t *threads = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
    int heads_per = n_heads / n_threads;
    int extra = n_heads % n_threads;
    int h = 0;
    for (int i = 0; i < n_threads; i++) {
        int count = heads_per + (i < extra ? 1 : 0);
        tasks[i] = (da3_attn_task){qkv, out, n_tok, dim, head_dim, n_heads,
                                    h, h + count, scale};
        h += count;
        pthread_create(&threads[i], NULL, da3_attn_worker, &tasks[i]);
    }
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
    free(tasks); free(threads);
}

/* ---- Conv2d helpers (CHW layout) ---- */

/* General Conv2d: weight[C_out, C_in, kH, kW], bias[C_out] */
static void da3_conv2d(float *dst, const float *src, const float *weight, const float *bias,
                       int H, int W, int Ci, int Co, int kH, int kW, int stride, int pad) {
    int Ho = (H + 2 * pad - kH) / stride + 1;
    int Wo = (W + 2 * pad - kW) / stride + 1;
    for (int co = 0; co < Co; co++) {
        float b = bias ? bias[co] : 0.0f;
        for (int oh = 0; oh < Ho; oh++) {
            for (int ow = 0; ow < Wo; ow++) {
                float sum = b;
                for (int ci = 0; ci < Ci; ci++) {
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = oh * stride - pad + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = ow * stride - pad + kw;
                            if (iw < 0 || iw >= W) continue;
                            sum += weight[((co * Ci + ci) * kH + kh) * kW + kw]
                                 * src[ci * H * W + ih * W + iw];
                        }
                    }
                }
                dst[co * Ho * Wo + oh * Wo + ow] = sum;
            }
        }
    }
}

/* ConvTranspose2d: weight[C_in, C_out, kH, kW] (PyTorch convention) */
static void da3_conv_transpose2d(float *dst, const float *src, const float *weight,
                                  const float *bias, int Hi, int Wi, int Ci, int Co,
                                  int kH, int kW, int stride) {
    int Ho = (Hi - 1) * stride + kH;
    int Wo = (Wi - 1) * stride + kW;
    /* Initialize with bias */
    for (int co = 0; co < Co; co++) {
        float b = bias ? bias[co] : 0.0f;
        for (int i = 0; i < Ho * Wo; i++) dst[co * Ho * Wo + i] = b;
    }
    for (int ci = 0; ci < Ci; ci++) {
        for (int ih = 0; ih < Hi; ih++) {
            for (int iw = 0; iw < Wi; iw++) {
                float val = src[ci * Hi * Wi + ih * Wi + iw];
                for (int co = 0; co < Co; co++) {
                    for (int kh = 0; kh < kH; kh++) {
                        int oh = ih * stride + kh;
                        for (int kw = 0; kw < kW; kw++) {
                            int ow = iw * stride + kw;
                            dst[co * Ho * Wo + oh * Wo + ow] +=
                                val * weight[((ci * Co + co) * kH + kh) * kW + kw];
                        }
                    }
                }
            }
        }
    }
}

/* Conv2d using qtensor weights (dequant + conv) */
static float *da3_conv2d_qt(const float *src, const qtensor *wt, const qtensor *bt,
                             int H, int W, int Ci, int Co, int kH, int kW,
                             int stride, int pad, int *Ho, int *Wo) {
    *Ho = (H + 2 * pad - kH) / stride + 1;
    *Wo = (W + 2 * pad - kW) / stride + 1;
    float *weight = da3_dequant(wt);
    float *bias = bt ? da3_dequant(bt) : NULL;
    float *dst = (float *)calloc((size_t)Co * (*Ho) * (*Wo), sizeof(float));
    da3_conv2d(dst, src, weight, bias, H, W, Ci, Co, kH, kW, stride, pad);
    free(weight); free(bias);
    return dst;
}

static float *da3_conv_transpose2d_qt(const float *src, const qtensor *wt, const qtensor *bt,
                                       int Hi, int Wi, int Ci, int Co,
                                       int kH, int kW, int stride, int *Ho, int *Wo) {
    *Ho = (Hi - 1) * stride + kH;
    *Wo = (Wi - 1) * stride + kW;
    float *weight = da3_dequant(wt);
    float *bias = bt ? da3_dequant(bt) : NULL;
    float *dst = (float *)calloc((size_t)Co * (*Ho) * (*Wo), sizeof(float));
    da3_conv_transpose2d(dst, src, weight, bias, Hi, Wi, Ci, Co, kH, kW, stride);
    free(weight); free(bias);
    return dst;
}

/* ---- Bilinear interpolation (align_corners=True, CHW layout) ---- */

static void da3_bilinear(float *dst, const float *src, int C,
                         int Hi, int Wi, int Ho, int Wo) {
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < Ho; oh++) {
            float fy = (Ho > 1) ? (float)oh * (Hi - 1) / (Ho - 1) : 0.0f;
            int y0 = (int)fy;
            int y1 = (y0 + 1 < Hi) ? y0 + 1 : y0;
            float dy = fy - y0;
            for (int ow = 0; ow < Wo; ow++) {
                float fx = (Wo > 1) ? (float)ow * (Wi - 1) / (Wo - 1) : 0.0f;
                int x0 = (int)fx;
                int x1 = (x0 + 1 < Wi) ? x0 + 1 : x0;
                float dx = fx - x0;
                const float *s = src + c * Hi * Wi;
                dst[c * Ho * Wo + oh * Wo + ow] =
                    s[y0 * Wi + x0] * (1 - dy) * (1 - dx) +
                    s[y0 * Wi + x1] * (1 - dy) * dx +
                    s[y1 * Wi + x0] * dy * (1 - dx) +
                    s[y1 * Wi + x1] * dy * dx;
            }
        }
    }
}

/* ---- ResidualConvUnit ---- */

static void da3_rcu(float *out, const float *x, const float *c1w, const float *c1b,
                    const float *c2w, const float *c2b, int C, int H, int W) {
    int sz = C * H * W;
    float *tmp = (float *)calloc((size_t)sz, sizeof(float));
    /* ReLU */
    for (int i = 0; i < sz; i++) tmp[i] = x[i] > 0 ? x[i] : 0.0f;
    /* Conv1 3x3 pad1 */
    float *mid = (float *)calloc((size_t)sz, sizeof(float));
    da3_conv2d(mid, tmp, c1w, c1b, H, W, C, C, 3, 3, 1, 1);
    /* ReLU */
    for (int i = 0; i < sz; i++) mid[i] = mid[i] > 0 ? mid[i] : 0.0f;
    /* Conv2 3x3 pad1 */
    da3_conv2d(tmp, mid, c2w, c2b, H, W, C, C, 3, 3, 1, 1);
    /* Residual */
    for (int i = 0; i < sz; i++) out[i] = tmp[i] + x[i];
    free(tmp); free(mid);
}

/* ---- RefineNet fusion block ---- */

static float *da3_refinenet(const da3_dpt_head *head, int stage,
                             const float *feat, int fH, int fW,
                             const float *deeper, int dH, int dW,
                             int features) {
    int sz = features * fH * fW;
    float *output = (float *)malloc((size_t)sz * sizeof(float));
    memcpy(output, feat, (size_t)sz * sizeof(float));

    /* If deeper input exists, upsample and add (with optional RCU1) */
    if (deeper) {
        float *up = (float *)malloc((size_t)sz * sizeof(float));
        da3_bilinear(up, deeper, features, dH, dW, fH, fW);
        if (head->fuse_rcu1_c1_w[stage].data) {
            float *c1w = da3_dequant(&head->fuse_rcu1_c1_w[stage]);
            float *c1b = da3_dequant(&head->fuse_rcu1_c1_b[stage]);
            float *c2w = da3_dequant(&head->fuse_rcu1_c2_w[stage]);
            float *c2b = da3_dequant(&head->fuse_rcu1_c2_b[stage]);
            float *rcu_out = (float *)malloc((size_t)sz * sizeof(float));
            da3_rcu(rcu_out, up, c1w, c1b, c2w, c2b, features, fH, fW);
            for (int i = 0; i < sz; i++) output[i] += rcu_out[i];
            free(rcu_out); free(c1w); free(c1b); free(c2w); free(c2b);
        } else {
            for (int i = 0; i < sz; i++) output[i] += up[i];
        }
        free(up);
    }

    /* RCU2 (if weights exist) */
    if (head->fuse_rcu2_c1_w[stage].data) {
        float *c1w = da3_dequant(&head->fuse_rcu2_c1_w[stage]);
        float *c1b = da3_dequant(&head->fuse_rcu2_c1_b[stage]);
        float *c2w = da3_dequant(&head->fuse_rcu2_c2_w[stage]);
        float *c2b = da3_dequant(&head->fuse_rcu2_c2_b[stage]);
        float *rcu_out = (float *)malloc((size_t)sz * sizeof(float));
        da3_rcu(rcu_out, output, c1w, c1b, c2w, c2b, features, fH, fW);
        memcpy(output, rcu_out, (size_t)sz * sizeof(float));
        free(rcu_out); free(c1w); free(c1b); free(c2w); free(c2b);
    }

    /* out_conv: 1x1 */
    float *ow = da3_dequant(&head->fuse_out_w[stage]);
    float *ob = da3_dequant(&head->fuse_out_b[stage]);
    float *conv_out = (float *)calloc((size_t)sz, sizeof(float));
    da3_conv2d(conv_out, output, ow, ob, fH, fW, features, features, 1, 1, 1, 0);
    free(output); free(ow); free(ob);
    return conv_out;
}

/* ==================================================================== */
/* API: da3_load                                                        */
/* ==================================================================== */

da3_model *da3_load(gguf_context *gguf) {
    da3_model *m = (da3_model *)calloc(1, sizeof(da3_model));

    m->dim        = da3_get_int(gguf, "da3.embed_dim", 384);
    m->n_heads    = da3_get_int(gguf, "da3.n_heads", 6);
    m->head_dim   = da3_get_int(gguf, "da3.head_dim", 64);
    m->n_blocks   = da3_get_int(gguf, "da3.n_blocks", 12);
    m->ffn_hidden = da3_get_int(gguf, "da3.ffn_hidden", 1024);
    m->patch_size = da3_get_int(gguf, "da3.patch_size", 14);
    m->image_size = da3_get_int(gguf, "da3.image_size", 518);
    m->ln_eps     = da3_get_float(gguf, "da3.ln_eps", 1e-6f);
    m->rope_start_layer    = da3_get_int(gguf, "da3.rope_start_layer", 4);
    m->qk_norm_start_layer = da3_get_int(gguf, "da3.qk_norm_start_layer", 4);
    m->head_features = da3_get_int(gguf, "da3.head.features", 64);

    m->grid_h = m->image_size / m->patch_size;
    m->grid_w = m->grid_h;
    m->n_patches = m->grid_h * m->grid_w;
    m->n_tokens = m->n_patches + 1;

    /* Feature layers */
    int fl_idx = gguf_find_key(gguf, "da3.feature_layers");
    if (fl_idx >= 0 && gguf->kv[fl_idx].type == GGUF_TYPE_ARRAY) {
        int32_t *arr = (int32_t *)gguf->kv[fl_idx].value.arr.data;
        for (int i = 0; i < 4; i++) m->feature_layers[i] = arr[i];
    } else {
        m->feature_layers[0] = 5; m->feature_layers[1] = 7;
        m->feature_layers[2] = 9; m->feature_layers[3] = 11;
    }

    /* Head out_channels */
    int oc_idx = gguf_find_key(gguf, "da3.head.out_channels");
    if (oc_idx >= 0 && gguf->kv[oc_idx].type == GGUF_TYPE_ARRAY) {
        int32_t *arr = (int32_t *)gguf->kv[oc_idx].value.arr.data;
        for (int i = 0; i < 4; i++) m->head_out_channels[i] = arr[i];
    } else {
        m->head_out_channels[0] = 48; m->head_out_channels[1] = 96;
        m->head_out_channels[2] = 192; m->head_out_channels[3] = 384;
    }

    /* Image normalization */
    int mi = gguf_find_key(gguf, "da3.image_mean");
    if (mi >= 0 && gguf->kv[mi].type == GGUF_TYPE_ARRAY) {
        float *arr = (float *)gguf->kv[mi].value.arr.data;
        for (int i = 0; i < 3; i++) m->image_mean[i] = arr[i];
    } else {
        m->image_mean[0] = 0.485f; m->image_mean[1] = 0.456f; m->image_mean[2] = 0.406f;
    }
    int si = gguf_find_key(gguf, "da3.image_std");
    if (si >= 0 && gguf->kv[si].type == GGUF_TYPE_ARRAY) {
        float *arr = (float *)gguf->kv[si].value.arr.data;
        for (int i = 0; i < 3; i++) m->image_std[i] = arr[i];
    } else {
        m->image_std[0] = 0.229f; m->image_std[1] = 0.224f; m->image_std[2] = 0.225f;
    }

    /* FFN type */
    int ft_idx = gguf_find_key(gguf, "da3.ffn_type");
    m->use_swiglu = 0;
    if (ft_idx >= 0 && gguf->kv[ft_idx].type == GGUF_TYPE_STRING) {
        if (strcmp(gguf->kv[ft_idx].value.str.str, "swiglu") == 0)
            m->use_swiglu = 1;
    }

    /* Embeddings */
    m->cls_token     = da3_load_tensor(gguf, "da3.cls_token", 1);
    m->pos_embed     = da3_load_tensor(gguf, "da3.pos_embed", 1);
    m->patch_embed_w = da3_load_tensor(gguf, "da3.patch_embed.weight", 1);
    m->patch_embed_b = da3_load_tensor(gguf, "da3.patch_embed.bias", 1);

    /* Blocks */
    m->blocks = (da3_block *)calloc((size_t)m->n_blocks, sizeof(da3_block));
    for (int L = 0; L < m->n_blocks; L++) {
        char name[128];
        da3_block *blk = &m->blocks[L];
#define DA3_BLK(field, fmt, ...) \
        snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
        blk->field = da3_load_tensor(gguf, name, 0);

        DA3_BLK(ln1_w,          "da3.blk.%d.ln1.weight", L)
        DA3_BLK(ln1_b,          "da3.blk.%d.ln1.bias", L)
        DA3_BLK(attn_qkv_w,     "da3.blk.%d.attn_qkv.weight", L)
        DA3_BLK(attn_qkv_b,     "da3.blk.%d.attn_qkv.bias", L)
        DA3_BLK(attn_q_norm_w,  "da3.blk.%d.attn_q_norm.weight", L)
        DA3_BLK(attn_q_norm_b,  "da3.blk.%d.attn_q_norm.bias", L)
        DA3_BLK(attn_k_norm_w,  "da3.blk.%d.attn_k_norm.weight", L)
        DA3_BLK(attn_k_norm_b,  "da3.blk.%d.attn_k_norm.bias", L)
        DA3_BLK(attn_out_w,     "da3.blk.%d.attn_out.weight", L)
        DA3_BLK(attn_out_b,     "da3.blk.%d.attn_out.bias", L)
        DA3_BLK(ls1,            "da3.blk.%d.ls1", L)
        DA3_BLK(ln2_w,          "da3.blk.%d.ln2.weight", L)
        DA3_BLK(ln2_b,          "da3.blk.%d.ln2.bias", L)
        DA3_BLK(ffn_gate_up_w,  "da3.blk.%d.ffn_gate_up.weight", L)
        DA3_BLK(ffn_gate_up_b,  "da3.blk.%d.ffn_gate_up.bias", L)
        DA3_BLK(ffn_up_w,       "da3.blk.%d.ffn_up.weight", L)
        DA3_BLK(ffn_up_b,       "da3.blk.%d.ffn_up.bias", L)
        DA3_BLK(ffn_down_w,     "da3.blk.%d.ffn_down.weight", L)
        DA3_BLK(ffn_down_b,     "da3.blk.%d.ffn_down.bias", L)
        DA3_BLK(ls2,            "da3.blk.%d.ls2", L)
#undef DA3_BLK

        /* Detect SwiGLU from actual weights */
        if (blk->ffn_gate_up_w.data) m->use_swiglu = 1;
    }

    /* DPT Head */
    da3_dpt_head *h = &m->head;
    h->norm_w = da3_load_tensor(gguf, "da3.head.norm.weight", 0);
    h->norm_b = da3_load_tensor(gguf, "da3.head.norm.bias", 0);

    for (int i = 0; i < 4; i++) {
        char name[128];
        snprintf(name, sizeof(name), "da3.head.proj.%d.weight", i);
        h->proj_w[i] = da3_load_tensor(gguf, name, 0);
        snprintf(name, sizeof(name), "da3.head.proj.%d.bias", i);
        h->proj_b[i] = da3_load_tensor(gguf, name, 0);
    }

    h->upsample_0_w = da3_load_tensor(gguf, "da3.head.upsample_0.weight", 0);
    h->upsample_0_b = da3_load_tensor(gguf, "da3.head.upsample_0.bias", 0);
    h->upsample_1_w = da3_load_tensor(gguf, "da3.head.upsample_1.weight", 0);
    h->upsample_1_b = da3_load_tensor(gguf, "da3.head.upsample_1.bias", 0);
    h->downsample_w = da3_load_tensor(gguf, "da3.head.downsample.weight", 0);
    h->downsample_b = da3_load_tensor(gguf, "da3.head.downsample.bias", 0);

    for (int i = 0; i < 4; i++) {
        char name[128];
        snprintf(name, sizeof(name), "da3.head.adapter.%d.weight", i);
        h->adapter_w[i] = da3_load_tensor(gguf, name, 0);

        snprintf(name, sizeof(name), "da3.head.fuse.%d.out.weight", i);
        h->fuse_out_w[i] = da3_load_tensor(gguf, name, 0);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.out.bias", i);
        h->fuse_out_b[i] = da3_load_tensor(gguf, name, 0);

        /* RCU weights (optional) */
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.weight", i);
        h->fuse_rcu1_c1_w[i] = da3_load_tensor(gguf, name, 0);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.bias", i);
        h->fuse_rcu1_c1_b[i] = da3_load_tensor(gguf, name, 0);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.weight", i);
        h->fuse_rcu1_c2_w[i] = da3_load_tensor(gguf, name, 0);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.bias", i);
        h->fuse_rcu1_c2_b[i] = da3_load_tensor(gguf, name, 0);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.weight", i);
        h->fuse_rcu2_c1_w[i] = da3_load_tensor(gguf, name, 0);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.bias", i);
        h->fuse_rcu2_c1_b[i] = da3_load_tensor(gguf, name, 0);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.weight", i);
        h->fuse_rcu2_c2_w[i] = da3_load_tensor(gguf, name, 0);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.bias", i);
        h->fuse_rcu2_c2_b[i] = da3_load_tensor(gguf, name, 0);
    }

    h->neck_w  = da3_load_tensor(gguf, "da3.head.neck.weight", 0);
    h->neck_b  = da3_load_tensor(gguf, "da3.head.neck.bias", 0);
    h->out_0_w = da3_load_tensor(gguf, "da3.head.out_0.weight", 0);
    h->out_0_b = da3_load_tensor(gguf, "da3.head.out_0.bias", 0);
    h->out_2_w = da3_load_tensor(gguf, "da3.head.out_2.weight", 0);
    h->out_2_b = da3_load_tensor(gguf, "da3.head.out_2.bias", 0);

    fprintf(stderr, "da3: loaded %d blocks, dim=%d, heads=%d, patches=%dx%d=%d, swiglu=%d\n",
            m->n_blocks, m->dim, m->n_heads, m->grid_h, m->grid_w, m->n_patches, m->use_swiglu);
    return m;
}

/* ==================================================================== */
/* API: da3_load_safetensors                                            */
/* ==================================================================== */

#ifdef SAFETENSORS_H

/* Helper: create qtensor from safetensors by internal name */
typedef struct { char st_name[256]; char gg_name[256]; } da3_name_map;

static qtensor da3s_make_tensor(st_context *st, int idx) {
    qtensor t = {0};
    if (idx < 0) return t;
    t.data = safetensors_data(st, idx);
    const char *dt = safetensors_dtype(st, idx);
    if (strcmp(dt, "F32") == 0)       t.type = GGML_TYPE_F32;
    else if (strcmp(dt, "F16") == 0)  t.type = GGML_TYPE_F16;
    else if (strcmp(dt, "BF16") == 0) t.type = GGML_TYPE_F32; /* BF16 needs convert */
    else return (qtensor){0}; /* unsupported type */
    t.n_dims = safetensors_ndims(st, idx);
    const uint64_t *shape = safetensors_shape(st, idx);
    for (int d = 0; d < t.n_dims; d++) t.dims[d] = shape[d];
    /* Set n_rows/n_cols for GEMM compatibility:
     * - 1D: n_rows=1, n_cols=N
     * - 2D [out, in]: n_rows=out, n_cols=in
     * - 4D [out, in, kH, kW]: n_rows=out, n_cols=in*kH*kW
     * This matches how da3_dequant_row indexes the data. */
    if (t.n_dims >= 2) {
        t.n_rows = (int)shape[0];
        t.n_cols = 1;
        for (int d = 1; d < t.n_dims; d++) t.n_cols *= (int)shape[d];
    } else {
        t.n_cols = (int)shape[0];
        t.n_rows = 1;
    }
    /* Handle BF16 → F32 conversion (in-place not possible with mmap, allocate) */
    if (strcmp(dt, "BF16") == 0) {
        int numel = t.n_cols * t.n_rows;
        float *buf = (float *)malloc((size_t)numel * sizeof(float));
        const uint16_t *src = (const uint16_t *)t.data;
        for (int i = 0; i < numel; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            memcpy(&buf[i], &bits, 4);
        }
        t.data = buf;
        t.type = GGML_TYPE_F32;
    }
    return t;
}

/* Map PyTorch safetensors name to internal GGUF name.
 * Returns 1 if mapped, 0 if not recognized. */
static int da3s_map_name(const char *st_name, char *gg_name, int gg_size) {
    gg_name[0] = '\0';

    /* Detect and strip backbone prefix */
    const char *s = st_name;
    static const char *bb_pfx[] = {
        "model.backbone.pretrained.", "backbone.pretrained.",
        "backbone.", "pretrained.", "encoder.", NULL
    };
    for (int i = 0; bb_pfx[i]; i++) {
        size_t pl = strlen(bb_pfx[i]);
        if (strncmp(s, bb_pfx[i], pl) == 0) { s += pl; break; }
    }

    /* Head prefix */
    static const char *hd_pfx[] = { "model.head.", "head.", "depth_head.", NULL };
    int is_head = 0;
    const char *orig_s = s;
    for (int i = 0; hd_pfx[i]; i++) {
        size_t pl = strlen(hd_pfx[i]);
        if (strncmp(s, hd_pfx[i], pl) == 0) { s += pl; is_head = 1; break; }
    }

    /* CameraDec prefix */
    static const char *cam_pfx[] = { "model.cam_dec.", "cam_dec.", NULL };
    int is_cam = 0;
    for (int i = 0; cam_pfx[i]; i++) {
        size_t pl = strlen(cam_pfx[i]);
        if (strncmp(orig_s, cam_pfx[i], pl) == 0) {
            s = orig_s + pl;
            is_cam = 1; break;
        }
    }

    /* GSDPT prefix */
    static const char *gs_pfx[] = { "model.gs_head.", "gs_head.", NULL };
    int is_gs = 0;
    for (int i = 0; gs_pfx[i]; i++) {
        size_t pl = strlen(gs_pfx[i]);
        if (strncmp(orig_s, gs_pfx[i], pl) == 0) {
            s = orig_s + pl;
            is_gs = 1; break;
        }
    }

    if (is_cam) {
        /* CameraDec mapping */
        static const struct { const char *st; const char *gg; } cam_map[] = {
            {"backbone.0.weight",  "cam_dec.mlp.0.weight"},
            {"backbone.0.bias",    "cam_dec.mlp.0.bias"},
            {"backbone.2.weight",  "cam_dec.mlp.2.weight"},
            {"backbone.2.bias",    "cam_dec.mlp.2.bias"},
            {"fc_t.weight",        "cam_dec.fc_t.weight"},
            {"fc_t.bias",          "cam_dec.fc_t.bias"},
            {"fc_qvec.weight",     "cam_dec.fc_qvec.weight"},
            {"fc_qvec.bias",       "cam_dec.fc_qvec.bias"},
            {"fc_fov.0.weight",    "cam_dec.fc_fov.weight"},
            {"fc_fov.0.bias",      "cam_dec.fc_fov.bias"},
            {NULL, NULL}
        };
        for (int i = 0; cam_map[i].st; i++) {
            if (strcmp(s, cam_map[i].st) == 0) {
                snprintf(gg_name, gg_size, "da3.%s", cam_map[i].gg);
                return 1;
            }
        }
        return 0;
    }

    if (is_gs) {
        /* GSDPT mapping */
        if (strncmp(s, "images_merger.", 14) == 0) {
            const char *ms = s + 14;
            int idx = -1;
            if (ms[0] >= '0' && ms[0] <= '9') {
                int raw = ms[0] - '0';
                if (raw == 0) idx = 0;
                else if (raw == 2) idx = 1;
                else if (raw == 4) idx = 2;
            }
            if (idx >= 0 && ms[1] == '.') {
                snprintf(gg_name, gg_size, "da3.gsdpt.merger.%d.%s", idx, ms + 2);
                return 1;
            }
            return 0;
        }
        /* GSDPT DPT head mapping (under scratch) */
        const char *ss = NULL;
        if (strncmp(s, "scratch.", 8) == 0) ss = s + 8;
        if (ss) {
            /* output convs */
            static const struct { const char *st; const char *gg; } gsout_map[] = {
                {"output_conv1.weight",   "da3.gsdpt.head.neck.weight"},
                {"output_conv1.bias",     "da3.gsdpt.head.neck.bias"},
                {"output_conv2.0.weight", "da3.gsdpt.head.out_0.weight"},
                {"output_conv2.0.bias",   "da3.gsdpt.head.out_0.bias"},
                {"output_conv2.2.weight", "da3.gsdpt.head.out_2.weight"},
                {"output_conv2.2.bias",   "da3.gsdpt.head.out_2.bias"},
                {NULL, NULL}
            };
            for (int i = 0; gsout_map[i].st; i++) {
                if (strcmp(ss, gsout_map[i].st) == 0) {
                    snprintf(gg_name, gg_size, "%s", gsout_map[i].gg);
                    return 1;
                }
            }
            /* Layer_rn adapters */
            for (int li = 1; li <= 4; li++) {
                char pfx[32];
                snprintf(pfx, sizeof(pfx), "layer%d_rn.weight", li);
                if (strcmp(ss, pfx) == 0) {
                    snprintf(gg_name, gg_size, "da3.gsdpt.head.adapter.%d.weight", li - 1);
                    return 1;
                }
            }
            /* RefineNet */
            static const struct { const char *st; const char *gg; } rn_map[] = {
                {"out_conv.weight",          "out.weight"},
                {"out_conv.bias",            "out.bias"},
                {"resConfUnit1.conv1.weight", "rcu1.conv1.weight"},
                {"resConfUnit1.conv1.bias",   "rcu1.conv1.bias"},
                {"resConfUnit1.conv2.weight", "rcu1.conv2.weight"},
                {"resConfUnit1.conv2.bias",   "rcu1.conv2.bias"},
                {"resConfUnit2.conv1.weight", "rcu2.conv1.weight"},
                {"resConfUnit2.conv1.bias",   "rcu2.conv1.bias"},
                {"resConfUnit2.conv2.weight", "rcu2.conv2.weight"},
                {"resConfUnit2.conv2.bias",   "rcu2.conv2.bias"},
                {NULL, NULL}
            };
            for (int ri = 1; ri <= 4; ri++) {
                char pfx[32];
                snprintf(pfx, sizeof(pfx), "refinenet%d.", ri);
                size_t plen = strlen(pfx);
                if (strncmp(ss, pfx, plen) == 0) {
                    const char *rn = ss + plen;
                    for (int j = 0; rn_map[j].st; j++) {
                        if (strcmp(rn, rn_map[j].st) == 0) {
                            snprintf(gg_name, gg_size, "da3.gsdpt.head.fuse.%d.%s",
                                     ri - 1, rn_map[j].gg);
                            return 1;
                        }
                    }
                }
            }
            return 0;
        }
        /* GSDPT projects, resize_layers, norm */
        if (strncmp(s, "projects.", 9) == 0) {
            int idx = s[9] - '0';
            snprintf(gg_name, gg_size, "da3.gsdpt.head.proj.%d.%s", idx, s + 11);
            return 1;
        }
        if (strncmp(s, "resize_layers.", 14) == 0) {
            int idx = s[14] - '0';
            const char *wb = s + 16;
            if (idx == 0) snprintf(gg_name, gg_size, "da3.gsdpt.head.upsample_0.%s", wb);
            else if (idx == 1) snprintf(gg_name, gg_size, "da3.gsdpt.head.upsample_1.%s", wb);
            else if (idx == 3) snprintf(gg_name, gg_size, "da3.gsdpt.head.downsample.%s", wb);
            return gg_name[0] != '\0';
        }
        if (strcmp(s, "readout_projects.0.0.weight") == 0 ||
            strcmp(s, "readout_projects.0.0.bias") == 0) {
            snprintf(gg_name, gg_size, "da3.gsdpt.head.norm.%s",
                     strstr(s, "weight") ? "weight" : "bias");
            return 1;
        }
        return 0;
    }

    if (is_head) {
        /* Main DPT head under scratch */
        const char *ss = NULL;
        if (strncmp(s, "scratch.", 8) == 0) ss = s + 8;
        if (ss) {
            /* Main output convs */
            static const struct { const char *st; const char *gg; } out_map[] = {
                {"output_conv1.weight",   "da3.head.neck.weight"},
                {"output_conv1.bias",     "da3.head.neck.bias"},
                {"output_conv2.0.weight", "da3.head.out_0.weight"},
                {"output_conv2.0.bias",   "da3.head.out_0.bias"},
                {"output_conv2.2.weight", "da3.head.out_2.weight"},
                {"output_conv2.2.bias",   "da3.head.out_2.bias"},
                {NULL, NULL}
            };
            for (int i = 0; out_map[i].st; i++) {
                if (strcmp(ss, out_map[i].st) == 0) {
                    snprintf(gg_name, gg_size, "%s", out_map[i].gg);
                    return 1;
                }
            }
            /* Layer_rn adapters */
            for (int li = 1; li <= 4; li++) {
                char pfx[32];
                snprintf(pfx, sizeof(pfx), "layer%d_rn.weight", li);
                if (strcmp(ss, pfx) == 0) {
                    snprintf(gg_name, gg_size, "da3.head.adapter.%d.weight", li - 1);
                    return 1;
                }
            }
            /* RefineNet main */
            static const struct { const char *st; const char *gg; } rn_map[] = {
                {"out_conv.weight",          "out.weight"},
                {"out_conv.bias",            "out.bias"},
                {"resConfUnit1.conv1.weight", "rcu1.conv1.weight"},
                {"resConfUnit1.conv1.bias",   "rcu1.conv1.bias"},
                {"resConfUnit1.conv2.weight", "rcu1.conv2.weight"},
                {"resConfUnit1.conv2.bias",   "rcu1.conv2.bias"},
                {"resConfUnit2.conv1.weight", "rcu2.conv1.weight"},
                {"resConfUnit2.conv1.bias",   "rcu2.conv1.bias"},
                {"resConfUnit2.conv2.weight", "rcu2.conv2.weight"},
                {"resConfUnit2.conv2.bias",   "rcu2.conv2.bias"},
                {NULL, NULL}
            };
            for (int ri = 1; ri <= 4; ri++) {
                char pfx[32];
                snprintf(pfx, sizeof(pfx), "refinenet%d.", ri);
                size_t plen = strlen(pfx);
                if (strncmp(ss, pfx, plen) == 0) {
                    const char *rn = ss + plen;
                    for (int j = 0; rn_map[j].st; j++) {
                        if (strcmp(rn, rn_map[j].st) == 0) {
                            snprintf(gg_name, gg_size, "da3.head.fuse.%d.%s",
                                     ri - 1, rn_map[j].gg);
                            return 1;
                        }
                    }
                }
            }
            /* Aux RefineNet */
            for (int ri = 1; ri <= 4; ri++) {
                char pfx[32];
                snprintf(pfx, sizeof(pfx), "refinenet%d_aux.", ri);
                size_t plen = strlen(pfx);
                if (strncmp(ss, pfx, plen) == 0) {
                    const char *rn = ss + plen;
                    for (int j = 0; rn_map[j].st; j++) {
                        if (strcmp(rn, rn_map[j].st) == 0) {
                            snprintf(gg_name, gg_size, "da3.head.aux_fuse.%d.%s",
                                     ri - 1, rn_map[j].gg);
                            return 1;
                        }
                    }
                }
            }
            /* output_conv1_aux: .{level}.{layer}.weight/bias */
            if (strncmp(ss, "output_conv1_aux.", 17) == 0) {
                int level = ss[17] - '0';
                if (level >= 0 && level < 4 && ss[18] == '.') {
                    int ci = ss[19] - '0';
                    if (ci >= 0 && ci <= 4 && ss[20] == '.') {
                        snprintf(gg_name, gg_size, "da3.head.aux_oc1.%d.%d.%s",
                                 level, ci, ss + 21);
                        return 1;
                    }
                }
                return 0;
            }
            /* output_conv2_aux: .{level}.{si}.weight/bias */
            if (strncmp(ss, "output_conv2_aux.", 17) == 0) {
                int level = ss[17] - '0';
                if (level >= 0 && level < 4 && ss[18] == '.') {
                    int si = ss[19] - '0';
                    if (ss[20] == '.') {
                        const char *wb = ss + 21;
                        if (si == 0) snprintf(gg_name, gg_size, "da3.head.aux_oc2.%d.conv.%s", level, wb);
                        else if (si == 2) snprintf(gg_name, gg_size, "da3.head.aux_oc2.%d.gn.%s", level, wb);
                        else if (si == 5) snprintf(gg_name, gg_size, "da3.head.aux_oc2.%d.out.%s", level, wb);
                        return gg_name[0] != '\0';
                    }
                }
                return 0;
            }
            return 0;
        }
        /* Head projects, resize_layers, norm */
        if (strncmp(s, "projects.", 9) == 0) {
            int idx = s[9] - '0';
            snprintf(gg_name, gg_size, "da3.head.proj.%d.%s", idx, s + 11);
            return 1;
        }
        if (strncmp(s, "resize_layers.", 14) == 0) {
            int idx = s[14] - '0';
            const char *wb = s + 16;
            if (idx == 0) snprintf(gg_name, gg_size, "da3.head.upsample_0.%s", wb);
            else if (idx == 1) snprintf(gg_name, gg_size, "da3.head.upsample_1.%s", wb);
            else if (idx == 3) snprintf(gg_name, gg_size, "da3.head.downsample.%s", wb);
            return gg_name[0] != '\0';
        }
        if (strcmp(s, "readout_projects.0.0.weight") == 0 ||
            strcmp(s, "readout_projects.0.0.bias") == 0) {
            snprintf(gg_name, gg_size, "da3.head.norm.%s",
                     strstr(s, "weight") ? "weight" : "bias");
            return 1;
        }
        if (strcmp(s, "norm.weight") == 0 || strcmp(s, "norm.bias") == 0) {
            snprintf(gg_name, gg_size, "da3.head.norm.%s",
                     strstr(s, "weight") ? "weight" : "bias");
            return 1;
        }
        return 0;
    }

    /* Backbone tensors */
    if (strcmp(s, "cls_token") == 0) { snprintf(gg_name, gg_size, "da3.cls_token"); return 1; }
    if (strcmp(s, "pos_embed") == 0) { snprintf(gg_name, gg_size, "da3.pos_embed"); return 1; }
    if (strcmp(s, "patch_embed.proj.weight") == 0) { snprintf(gg_name, gg_size, "da3.patch_embed.weight"); return 1; }
    if (strcmp(s, "patch_embed.proj.bias") == 0)   { snprintf(gg_name, gg_size, "da3.patch_embed.bias"); return 1; }
    if (strcmp(s, "norm.weight") == 0) { snprintf(gg_name, gg_size, "da3.backbone_norm.weight"); return 1; }
    if (strcmp(s, "norm.bias") == 0)   { snprintf(gg_name, gg_size, "da3.backbone_norm.bias"); return 1; }

    /* Transformer blocks */
    if (strncmp(s, "blocks.", 7) == 0) {
        int L = 0;
        const char *p = s + 7;
        while (*p >= '0' && *p <= '9') { L = L * 10 + (*p - '0'); p++; }
        if (*p == '.') p++;
        static const struct { const char *st; const char *gg; } blk_map[] = {
            {"norm1.weight",        "ln1.weight"},
            {"norm1.bias",          "ln1.bias"},
            {"attn.norm1.weight",   "ln1.weight"},
            {"attn.norm1.bias",     "ln1.bias"},
            {"attn.qkv.weight",     "attn_qkv.weight"},
            {"attn.qkv.bias",       "attn_qkv.bias"},
            {"attn.q_norm.weight",  "attn_q_norm.weight"},
            {"attn.q_norm.bias",    "attn_q_norm.bias"},
            {"attn.k_norm.weight",  "attn_k_norm.weight"},
            {"attn.k_norm.bias",    "attn_k_norm.bias"},
            {"attn.proj.weight",    "attn_out.weight"},
            {"attn.proj.bias",      "attn_out.bias"},
            {"ls1.gamma",           "ls1"},
            {"attn.ls1",            "ls1"},
            {"norm2.weight",        "ln2.weight"},
            {"norm2.bias",          "ln2.bias"},
            {"mlp.w12.weight",      "ffn_gate_up.weight"},
            {"mlp.w12.bias",        "ffn_gate_up.bias"},
            {"mlp.w3.weight",       "ffn_down.weight"},
            {"mlp.w3.bias",         "ffn_down.bias"},
            {"mlp.fc1.weight",      "ffn_up.weight"},
            {"mlp.fc1.bias",        "ffn_up.bias"},
            {"mlp.fc2.weight",      "ffn_down.weight"},
            {"mlp.fc2.bias",        "ffn_down.bias"},
            {"ls2.gamma",           "ls2"},
            {"ls2",                 "ls2"},
            {NULL, NULL}
        };
        for (int i = 0; blk_map[i].st; i++) {
            if (strcmp(p, blk_map[i].st) == 0) {
                snprintf(gg_name, gg_size, "da3.blk.%d.%s", L, blk_map[i].gg);
                return 1;
            }
        }
    }

    return 0;
}

/* Find mapped tensor by GGUF name in the name map */
static qtensor da3s_find(st_context *st, const da3_name_map *map, int map_count,
                          const char *gg_name) {
    qtensor t = {0};
    for (int i = 0; i < map_count; i++) {
        if (strcmp(map[i].gg_name, gg_name) == 0) {
            int idx = safetensors_find(st, map[i].st_name);
            if (idx >= 0) return da3s_make_tensor(st, idx);
            break;
        }
    }
    return t;
}

da3_model *da3_load_safetensors(const char *st_path, const char *config_path) {
    st_context *st = safetensors_open(st_path);
    if (!st) return NULL;

    fprintf(stderr, "da3: safetensors opened, %d tensors\n", st->n_tensors);

    /* Build name mapping */
    int map_cap = st->n_tensors;
    da3_name_map *map = (da3_name_map *)calloc(map_cap, sizeof(da3_name_map));
    int map_count = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        char gg[256];
        if (da3s_map_name(nm, gg, sizeof(gg))) {
            if (map_count < map_cap) {
                snprintf(map[map_count].st_name, 256, "%s", nm);
                snprintf(map[map_count].gg_name, 256, "%s", gg);
                map_count++;
            }
        }
    }
    fprintf(stderr, "da3: mapped %d/%d tensors\n", map_count, st->n_tensors);

    /* Auto-detect model parameters from tensor shapes */
    int embed_dim = 384, n_heads = 6, head_dim = 64;
    int n_blocks = 0, ffn_hidden = 1536, has_swiglu = 0;
    int patch_size = 14, image_size = 518;

    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        if (strstr(nm, "patch_embed.proj.weight")) {
            embed_dim = (int)safetensors_shape(st, i)[0];
            patch_size = (int)safetensors_shape(st, i)[2];
        }
        if (strstr(nm, "attn.q_norm.weight") && !strstr(nm, "_aux")) {
            head_dim = (int)safetensors_shape(st, i)[0];
        }
        if (strstr(nm, "blocks.0.mlp.w12.weight")) {
            ffn_hidden = (int)safetensors_shape(st, i)[0] / 2;
            has_swiglu = 1;
        }
        if (strstr(nm, "blocks.0.mlp.fc1.weight") && !has_swiglu) {
            ffn_hidden = (int)safetensors_shape(st, i)[0];
        }
        /* Count blocks */
        const char *p = strstr(nm, "blocks.");
        if (p && !strstr(nm, "_aux")) {
            p += 7;
            int blk = 0;
            while (*p >= '0' && *p <= '9') { blk = blk * 10 + (*p - '0'); p++; }
            if (blk + 1 > n_blocks) n_blocks = blk + 1;
        }
    }

    /* Derive n_heads from embed_dim / head_dim (DinoV2 default head_dim=64) */
    n_heads = embed_dim / head_dim;

    /* Config defaults */
    int feature_layers[4] = {5, 7, 9, 11};
    int head_features = 64;
    int head_oc[4] = {48, 96, 192, 384};
    int rope_start = 4, qknorm_start = 4;

    /* Size-based defaults */
    if (embed_dim >= 1536) {
        /* Giant */
        feature_layers[0]=19; feature_layers[1]=27; feature_layers[2]=33; feature_layers[3]=39;
        head_features = 256; rope_start = 13; qknorm_start = 13;
        head_oc[0]=256; head_oc[1]=512; head_oc[2]=1024; head_oc[3]=1024;
    } else if (embed_dim >= 1024) {
        /* Large */
        feature_layers[0]=11; feature_layers[1]=15; feature_layers[2]=19; feature_layers[3]=23;
        head_features = 256; rope_start = 8; qknorm_start = 8;
        head_oc[0]=256; head_oc[1]=512; head_oc[2]=1024; head_oc[3]=1024;
    } else if (embed_dim >= 768) {
        /* Base */
        feature_layers[0]=5; feature_layers[1]=7; feature_layers[2]=9; feature_layers[3]=11;
        head_features = 128; rope_start = 4; qknorm_start = 4;
        head_oc[0]=96; head_oc[1]=192; head_oc[2]=384; head_oc[3]=768;
    } else {
        /* Small */
        head_features = 64;
        head_oc[0]=48; head_oc[1]=96; head_oc[2]=192; head_oc[3]=384;
    }

    /* Parse config.json if provided */
    if (config_path) {
        FILE *f = fopen(config_path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *buf = (char *)malloc(sz + 1);
            size_t nr = fread(buf, 1, sz, f);
            buf[nr] = '\0';
            fclose(f);
            json_val *root = json_parse(buf, (int)nr);
            if (root) {
                json_val *cfg = json_obj_get(root, "config");
                if (!cfg) cfg = root;
                json_val *hcfg = json_obj_get(cfg, "head");
                if (hcfg) {
                    json_val *v = json_obj_get(hcfg, "features");
                    if (v && v->type == JSON_NUMBER) head_features = (int)v->num;
                    json_val *oc = json_obj_get(hcfg, "out_channels");
                    if (oc && oc->type == JSON_ARRAY)
                        for (int i = 0; i < 4 && i < oc->arr.count; i++)
                            head_oc[i] = (int)oc->arr.items[i].num;
                }
                json_val *net = json_obj_get(cfg, "net");
                if (net) {
                    json_val *ol = json_obj_get(net, "out_layers");
                    if (ol && ol->type == JSON_ARRAY)
                        for (int i = 0; i < 4 && i < ol->arr.count; i++)
                            feature_layers[i] = (int)ol->arr.items[i].num;
                    json_val *rs = json_obj_get(net, "rope_start");
                    if (rs && rs->type == JSON_NUMBER) rope_start = (int)rs->num;
                    json_val *qs = json_obj_get(net, "qknorm_start");
                    if (qs && qs->type == JSON_NUMBER) qknorm_start = (int)qs->num;
                }
                json_free(root);
            }
            free(buf);
        }
    }

    /* Allocate model */
    da3_model *m = (da3_model *)calloc(1, sizeof(da3_model));
    m->dim = embed_dim;
    m->n_heads = n_heads;
    m->head_dim = head_dim;
    m->n_blocks = n_blocks;
    m->ffn_hidden = ffn_hidden;
    m->patch_size = patch_size;
    m->image_size = image_size;
    m->ln_eps = 1e-6f;
    m->use_swiglu = has_swiglu;
    m->rope_start_layer = rope_start;
    m->qk_norm_start_layer = qknorm_start;
    m->head_features = head_features;
    for (int i = 0; i < 4; i++) {
        m->feature_layers[i] = feature_layers[i];
        m->head_out_channels[i] = head_oc[i];
    }
    m->image_mean[0] = 0.485f; m->image_mean[1] = 0.456f; m->image_mean[2] = 0.406f;
    m->image_std[0]  = 0.229f; m->image_std[1]  = 0.224f; m->image_std[2]  = 0.225f;
    m->grid_h = m->image_size / m->patch_size;
    m->grid_w = m->grid_h;
    m->n_patches = m->grid_h * m->grid_w;
    m->n_tokens = m->n_patches + 1;
    m->st_ctx = st;

    /* Load embeddings */
    m->cls_token     = da3s_find(st, map, map_count, "da3.cls_token");
    m->pos_embed     = da3s_find(st, map, map_count, "da3.pos_embed");
    m->patch_embed_w = da3s_find(st, map, map_count, "da3.patch_embed.weight");
    m->patch_embed_b = da3s_find(st, map, map_count, "da3.patch_embed.bias");

    /* Load backbone norm (for CameraDec) */
    m->backbone_norm_w = da3s_find(st, map, map_count, "da3.backbone_norm.weight");
    m->backbone_norm_b = da3s_find(st, map, map_count, "da3.backbone_norm.bias");

    /* Load blocks */
    m->blocks = (da3_block *)calloc((size_t)m->n_blocks, sizeof(da3_block));
    for (int L = 0; L < m->n_blocks; L++) {
        da3_block *blk = &m->blocks[L];
        char name[128];
#define DA3S_BLK(field, fmt, ...) \
        snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
        blk->field = da3s_find(st, map, map_count, name);

        DA3S_BLK(ln1_w,          "da3.blk.%d.ln1.weight", L)
        DA3S_BLK(ln1_b,          "da3.blk.%d.ln1.bias", L)
        DA3S_BLK(attn_qkv_w,     "da3.blk.%d.attn_qkv.weight", L)
        DA3S_BLK(attn_qkv_b,     "da3.blk.%d.attn_qkv.bias", L)
        DA3S_BLK(attn_q_norm_w,  "da3.blk.%d.attn_q_norm.weight", L)
        DA3S_BLK(attn_q_norm_b,  "da3.blk.%d.attn_q_norm.bias", L)
        DA3S_BLK(attn_k_norm_w,  "da3.blk.%d.attn_k_norm.weight", L)
        DA3S_BLK(attn_k_norm_b,  "da3.blk.%d.attn_k_norm.bias", L)
        DA3S_BLK(attn_out_w,     "da3.blk.%d.attn_out.weight", L)
        DA3S_BLK(attn_out_b,     "da3.blk.%d.attn_out.bias", L)
        DA3S_BLK(ls1,            "da3.blk.%d.ls1", L)
        DA3S_BLK(ln2_w,          "da3.blk.%d.ln2.weight", L)
        DA3S_BLK(ln2_b,          "da3.blk.%d.ln2.bias", L)
        DA3S_BLK(ffn_gate_up_w,  "da3.blk.%d.ffn_gate_up.weight", L)
        DA3S_BLK(ffn_gate_up_b,  "da3.blk.%d.ffn_gate_up.bias", L)
        DA3S_BLK(ffn_up_w,       "da3.blk.%d.ffn_up.weight", L)
        DA3S_BLK(ffn_up_b,       "da3.blk.%d.ffn_up.bias", L)
        DA3S_BLK(ffn_down_w,     "da3.blk.%d.ffn_down.weight", L)
        DA3S_BLK(ffn_down_b,     "da3.blk.%d.ffn_down.bias", L)
        DA3S_BLK(ls2,            "da3.blk.%d.ls2", L)
#undef DA3S_BLK
        if (blk->ffn_gate_up_w.data) m->use_swiglu = 1;
    }

    /* Load DPT head */
    da3_dpt_head *h = &m->head;
    h->norm_w = da3s_find(st, map, map_count, "da3.head.norm.weight");
    h->norm_b = da3s_find(st, map, map_count, "da3.head.norm.bias");
    for (int i = 0; i < 4; i++) {
        char name[128];
        snprintf(name, sizeof(name), "da3.head.proj.%d.weight", i);
        h->proj_w[i] = da3s_find(st, map, map_count, name);
        snprintf(name, sizeof(name), "da3.head.proj.%d.bias", i);
        h->proj_b[i] = da3s_find(st, map, map_count, name);
    }
    h->upsample_0_w = da3s_find(st, map, map_count, "da3.head.upsample_0.weight");
    h->upsample_0_b = da3s_find(st, map, map_count, "da3.head.upsample_0.bias");
    h->upsample_1_w = da3s_find(st, map, map_count, "da3.head.upsample_1.weight");
    h->upsample_1_b = da3s_find(st, map, map_count, "da3.head.upsample_1.bias");
    h->downsample_w = da3s_find(st, map, map_count, "da3.head.downsample.weight");
    h->downsample_b = da3s_find(st, map, map_count, "da3.head.downsample.bias");
    for (int i = 0; i < 4; i++) {
        char name[128];
        snprintf(name, sizeof(name), "da3.head.adapter.%d.weight", i);
        h->adapter_w[i] = da3s_find(st, map, map_count, name);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.out.weight", i);
        h->fuse_out_w[i] = da3s_find(st, map, map_count, name);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.out.bias", i);
        h->fuse_out_b[i] = da3s_find(st, map, map_count, name);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.weight", i);
        h->fuse_rcu1_c1_w[i] = da3s_find(st, map, map_count, name);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.bias", i);
        h->fuse_rcu1_c1_b[i] = da3s_find(st, map, map_count, name);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.weight", i);
        h->fuse_rcu1_c2_w[i] = da3s_find(st, map, map_count, name);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.bias", i);
        h->fuse_rcu1_c2_b[i] = da3s_find(st, map, map_count, name);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.weight", i);
        h->fuse_rcu2_c1_w[i] = da3s_find(st, map, map_count, name);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.bias", i);
        h->fuse_rcu2_c1_b[i] = da3s_find(st, map, map_count, name);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.weight", i);
        h->fuse_rcu2_c2_w[i] = da3s_find(st, map, map_count, name);
        snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.bias", i);
        h->fuse_rcu2_c2_b[i] = da3s_find(st, map, map_count, name);
    }
    h->neck_w  = da3s_find(st, map, map_count, "da3.head.neck.weight");
    h->neck_b  = da3s_find(st, map, map_count, "da3.head.neck.bias");
    h->out_0_w = da3s_find(st, map, map_count, "da3.head.out_0.weight");
    h->out_0_b = da3s_find(st, map, map_count, "da3.head.out_0.bias");
    h->out_2_w = da3s_find(st, map, map_count, "da3.head.out_2.weight");
    h->out_2_b = da3s_find(st, map, map_count, "da3.head.out_2.bias");

    /* Load CameraDec */
    {
        qtensor t = da3s_find(st, map, map_count, "da3.cam_dec.mlp.0.weight");
        if (t.data) {
            da3_cam_dec *cd = &m->cam_dec;
            cd->mlp0_w = t;
            cd->mlp_dim = t.n_rows;
            cd->mlp0_b = da3s_find(st, map, map_count, "da3.cam_dec.mlp.0.bias");
            cd->mlp2_w = da3s_find(st, map, map_count, "da3.cam_dec.mlp.2.weight");
            cd->mlp2_b = da3s_find(st, map, map_count, "da3.cam_dec.mlp.2.bias");
            cd->fc_t_w = da3s_find(st, map, map_count, "da3.cam_dec.fc_t.weight");
            cd->fc_t_b = da3s_find(st, map, map_count, "da3.cam_dec.fc_t.bias");
            cd->fc_qvec_w = da3s_find(st, map, map_count, "da3.cam_dec.fc_qvec.weight");
            cd->fc_qvec_b = da3s_find(st, map, map_count, "da3.cam_dec.fc_qvec.bias");
            cd->fc_fov_w = da3s_find(st, map, map_count, "da3.cam_dec.fc_fov.weight");
            cd->fc_fov_b = da3s_find(st, map, map_count, "da3.cam_dec.fc_fov.bias");
            m->has_cam_dec = 1;
            fprintf(stderr, "da3: CameraDec loaded (mlp_dim=%d)\n", cd->mlp_dim);
        }
    }

    /* Load Aux DPT */
    {
        qtensor t = da3s_find(st, map, map_count, "da3.head.aux_fuse.0.out.weight");
        if (t.data) {
            da3_aux_head *ah = &m->aux_head;
            for (int i = 0; i < 4; i++) {
                char name[128];
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.out.weight", i);
                ah->fuse_out_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.out.bias", i);
                ah->fuse_out_b[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv1.weight", i);
                ah->fuse_rcu1_c1_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv1.bias", i);
                ah->fuse_rcu1_c1_b[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv2.weight", i);
                ah->fuse_rcu1_c2_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv2.bias", i);
                ah->fuse_rcu1_c2_b[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv1.weight", i);
                ah->fuse_rcu2_c1_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv1.bias", i);
                ah->fuse_rcu2_c1_b[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv2.weight", i);
                ah->fuse_rcu2_c2_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv2.bias", i);
                ah->fuse_rcu2_c2_b[i] = da3s_find(st, map, map_count, name);
                ah->has_rcu1[i] = (ah->fuse_rcu1_c1_w[i].data != NULL);
                ah->has_rcu2[i] = (ah->fuse_rcu2_c1_w[i].data != NULL);

                /* output_conv1_aux: up to 5 Conv3x3 layers per level */
                ah->oc1_count[i] = 0;
                for (int j = 0; j < 5; j++) {
                    snprintf(name, sizeof(name), "da3.head.aux_oc1.%d.%d.weight", i, j);
                    ah->oc1_w[i][j] = da3s_find(st, map, map_count, name);
                    if (!ah->oc1_w[i][j].data) break;
                    snprintf(name, sizeof(name), "da3.head.aux_oc1.%d.%d.bias", i, j);
                    ah->oc1_b[i][j] = da3s_find(st, map, map_count, name);
                    ah->oc1_ci[i][j] = ah->oc1_w[i][j].n_cols / 9; /* weight [Co, Ci, 3, 3] → n_cols = Ci*9 in flat */
                    ah->oc1_co[i][j] = ah->oc1_w[i][j].n_rows;
                    /* For safetensors 4D: dims[0]=Co, dims[1]=Ci, dims[2]=3, dims[3]=3 */
                    if (ah->oc1_w[i][j].n_dims >= 4) {
                        ah->oc1_co[i][j] = (int)ah->oc1_w[i][j].dims[0];
                        ah->oc1_ci[i][j] = (int)ah->oc1_w[i][j].dims[1];
                    }
                    ah->oc1_count[i]++;
                }

                /* output_conv2_aux */
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.conv.weight", i);
                ah->oc2_conv_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.conv.bias", i);
                ah->oc2_conv_b[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.gn.weight", i);
                ah->oc2_gn_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.gn.bias", i);
                ah->oc2_gn_b[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.out.weight", i);
                ah->oc2_out_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.out.bias", i);
                ah->oc2_out_b[i] = da3s_find(st, map, map_count, name);
            }
            m->has_aux = 1;
            fprintf(stderr, "da3: Aux DPT loaded (oc1_count=[%d,%d,%d,%d])\n",
                    ah->oc1_count[0], ah->oc1_count[1], ah->oc1_count[2], ah->oc1_count[3]);
        }
    }

    /* Load GSDPT */
    {
        qtensor t = da3s_find(st, map, map_count, "da3.gsdpt.head.proj.0.weight");
        if (t.data) {
            da3_gsdpt *gs = &m->gsdpt;
            da3_dpt_head *gh = &gs->head;

            /* Norm */
            gh->norm_w = da3s_find(st, map, map_count, "da3.gsdpt.head.norm.weight");
            gh->norm_b = da3s_find(st, map, map_count, "da3.gsdpt.head.norm.bias");

            /* Projects */
            for (int i = 0; i < 4; i++) {
                char name[128];
                snprintf(name, sizeof(name), "da3.gsdpt.head.proj.%d.weight", i);
                gh->proj_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.gsdpt.head.proj.%d.bias", i);
                gh->proj_b[i] = da3s_find(st, map, map_count, name);
            }

            /* Resize layers */
            gh->upsample_0_w = da3s_find(st, map, map_count, "da3.gsdpt.head.upsample_0.weight");
            gh->upsample_0_b = da3s_find(st, map, map_count, "da3.gsdpt.head.upsample_0.bias");
            gh->upsample_1_w = da3s_find(st, map, map_count, "da3.gsdpt.head.upsample_1.weight");
            gh->upsample_1_b = da3s_find(st, map, map_count, "da3.gsdpt.head.upsample_1.bias");
            gh->downsample_w = da3s_find(st, map, map_count, "da3.gsdpt.head.downsample.weight");
            gh->downsample_b = da3s_find(st, map, map_count, "da3.gsdpt.head.downsample.bias");

            /* Adapters */
            for (int i = 0; i < 4; i++) {
                char name[128];
                snprintf(name, sizeof(name), "da3.gsdpt.head.adapter.%d.weight", i);
                gh->adapter_w[i] = da3s_find(st, map, map_count, name);
            }

            /* RefineNet fusion */
            for (int i = 0; i < 4; i++) {
                char name[128];
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.out.weight", i);
                gh->fuse_out_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.out.bias", i);
                gh->fuse_out_b[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv1.weight", i);
                gh->fuse_rcu1_c1_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv1.bias", i);
                gh->fuse_rcu1_c1_b[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv2.weight", i);
                gh->fuse_rcu1_c2_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv2.bias", i);
                gh->fuse_rcu1_c2_b[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv1.weight", i);
                gh->fuse_rcu2_c1_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv1.bias", i);
                gh->fuse_rcu2_c1_b[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv2.weight", i);
                gh->fuse_rcu2_c2_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv2.bias", i);
                gh->fuse_rcu2_c2_b[i] = da3s_find(st, map, map_count, name);
            }

            /* Output convs */
            gh->neck_w  = da3s_find(st, map, map_count, "da3.gsdpt.head.neck.weight");
            gh->neck_b  = da3s_find(st, map, map_count, "da3.gsdpt.head.neck.bias");
            gh->out_0_w = da3s_find(st, map, map_count, "da3.gsdpt.head.out_0.weight");
            gh->out_0_b = da3s_find(st, map, map_count, "da3.gsdpt.head.out_0.bias");
            gh->out_2_w = da3s_find(st, map, map_count, "da3.gsdpt.head.out_2.weight");
            gh->out_2_b = da3s_find(st, map, map_count, "da3.gsdpt.head.out_2.bias");

            /* Detect gs_out_channels from out_2 weight shape */
            gs->gs_out_channels = 38;
            if (gh->out_2_w.data && gh->out_2_w.n_dims >= 4)
                gs->gs_out_channels = (int)gh->out_2_w.dims[0];

            /* Images merger */
            for (int i = 0; i < 3; i++) {
                char name[128];
                snprintf(name, sizeof(name), "da3.gsdpt.merger.%d.weight", i);
                gs->merger_w[i] = da3s_find(st, map, map_count, name);
                snprintf(name, sizeof(name), "da3.gsdpt.merger.%d.bias", i);
                gs->merger_b[i] = da3s_find(st, map, map_count, name);
                if (gs->merger_w[i].data && gs->merger_w[i].n_dims >= 4) {
                    gs->merger_co[i] = (int)gs->merger_w[i].dims[0];
                    gs->merger_ci[i] = (int)gs->merger_w[i].dims[1];
                }
            }

            m->has_gsdpt = 1;
            fprintf(stderr, "da3: GSDPT loaded (out_ch=%d, merger=%d→%d→%d→%d)\n",
                    gs->gs_out_channels, 3,
                    gs->merger_co[0], gs->merger_co[1], gs->merger_co[2]);
        }
    }

    free(map);

    fprintf(stderr, "da3: safetensors loaded %d blocks, dim=%d, heads=%d, patches=%dx%d, swiglu=%d\n",
            m->n_blocks, m->dim, m->n_heads, m->grid_h, m->grid_w, m->use_swiglu);
    fprintf(stderr, "  features=%d, head_oc=[%d,%d,%d,%d], feat_layers=[%d,%d,%d,%d]\n",
            m->head_features, m->head_out_channels[0], m->head_out_channels[1],
            m->head_out_channels[2], m->head_out_channels[3],
            m->feature_layers[0], m->feature_layers[1],
            m->feature_layers[2], m->feature_layers[3]);
    fprintf(stderr, "  has_cam_dec=%d, has_aux=%d, has_gsdpt=%d\n",
            m->has_cam_dec, m->has_aux, m->has_gsdpt);
    return m;
}

#endif /* SAFETENSORS_H */

/* ==================================================================== */
/* API: da3_free                                                        */
/* ==================================================================== */

void da3_free(da3_model *m) {
    if (!m) return;
    free(m->blocks);
#ifdef SAFETENSORS_H
    if (m->st_ctx) safetensors_close((st_context *)m->st_ctx);
#endif
    free(m);
}

/* ==================================================================== */
/* API: da3_predict                                                     */
/* ==================================================================== */

da3_result da3_predict(da3_model *m, const uint8_t *rgb, int img_w, int img_h, int n_threads) {
    da3_result result = {0};
    int ps = m->patch_size;
    int gh = m->grid_h, gw = m->grid_w;
    int np = m->n_patches;
    int nt = m->n_tokens;
    int dim = m->dim;
    int target_h = gh * ps;
    int target_w = gw * ps;

    if (n_threads < 1) n_threads = 1;
    double t_start = da3_time_ms();

    /* ─── Step 1: Preprocess ─── */
    /* Resize to target_h x target_w, normalize */
    float *img_norm = (float *)malloc((size_t)3 * target_h * target_w * sizeof(float));
    for (int c = 0; c < 3; c++) {
        for (int oh = 0; oh < target_h; oh++) {
            float fy = (target_h > 1) ? (float)oh * (img_h - 1) / (target_h - 1) : 0.0f;
            int y0 = (int)fy; int y1 = y0 + 1 < img_h ? y0 + 1 : y0;
            float dy = fy - y0;
            for (int ow = 0; ow < target_w; ow++) {
                float fx = (target_w > 1) ? (float)ow * (img_w - 1) / (target_w - 1) : 0.0f;
                int x0 = (int)fx; int x1 = x0 + 1 < img_w ? x0 + 1 : x0;
                float dx = fx - x0;
                float v = (float)rgb[(y0 * img_w + x0) * 3 + c] * (1-dy)*(1-dx)
                        + (float)rgb[(y0 * img_w + x1) * 3 + c] * (1-dy)*dx
                        + (float)rgb[(y1 * img_w + x0) * 3 + c] * dy*(1-dx)
                        + (float)rgb[(y1 * img_w + x1) * 3 + c] * dy*dx;
                img_norm[c * target_h * target_w + oh * target_w + ow] =
                    (v / 255.0f - m->image_mean[c]) / m->image_std[c];
            }
        }
    }

    /* ─── Step 2: Patch embedding ─── */
    float *hidden = (float *)calloc((size_t)nt * dim, sizeof(float));
    {
        float *pw = da3_dequant(&m->patch_embed_w);
        float *pb = da3_dequant(&m->patch_embed_b);
        /* patch_embed_w: [Co=384, Ci=3, kH=14, kW=14] in PyTorch order */
        int Co = dim, Ci = 3;
        for (int py = 0; py < gh; py++) {
            for (int px = 0; px < gw; px++) {
                int tok = 1 + py * gw + px;  /* skip CLS at 0 */
                float *out = hidden + tok * dim;
                if (pb) memcpy(out, pb, (size_t)dim * sizeof(float));
                for (int co = 0; co < Co; co++) {
                    float sum = 0.0f;
                    for (int ci = 0; ci < Ci; ci++) {
                        for (int kh = 0; kh < ps; kh++) {
                            for (int kw = 0; kw < ps; kw++) {
                                int ih = py * ps + kh;
                                int iw = px * ps + kw;
                                sum += pw[((co * Ci + ci) * ps + kh) * ps + kw]
                                     * img_norm[ci * target_h * target_w + ih * target_w + iw];
                            }
                        }
                    }
                    out[co] += sum;
                }
            }
        }
        free(pw); free(pb);
    }
    free(img_norm);

    /* ─── Step 3: CLS token + positional embedding ─── */
    {
        float *cls = da3_dequant(&m->cls_token);
        memcpy(hidden, cls, (size_t)dim * sizeof(float));
        free(cls);

        float *pe = da3_dequant(&m->pos_embed);
        /* pos_embed: [n_tokens, dim], add to all tokens */
        for (int t = 0; t < nt; t++)
            for (int i = 0; i < dim; i++)
                hidden[t * dim + i] += pe[t * dim + i];
        free(pe);
    }

    /* Build position arrays for RoPE 2D (CLS=0, patches use grid coords) */
    int *pos_y = (int *)calloc((size_t)nt, sizeof(int));
    int *pos_x = (int *)calloc((size_t)nt, sizeof(int));
    for (int p = 0; p < np; p++) {
        pos_y[1 + p] = p / gw;
        pos_x[1 + p] = p % gw;
    }

    /* ─── Step 4: Transformer blocks ─── */
    double t_backbone = da3_time_ms();
    float *ln_buf   = (float *)malloc((size_t)nt * dim * sizeof(float));
    float *qkv      = (float *)malloc((size_t)nt * 3 * dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)nt * dim * sizeof(float));
    int ffn_out_dim = m->use_swiglu ? 2 * m->ffn_hidden : (m->blocks[0].ffn_up_w.data ? m->blocks[0].ffn_up_w.n_rows : 4 * dim);
    float *ffn_buf  = (float *)malloc((size_t)nt * ffn_out_dim * sizeof(float));
    float *ffn_mid  = (float *)malloc((size_t)nt * m->ffn_hidden * sizeof(float));

    /* Saved features for DPT head (4 layers) */
    float *features[4] = {NULL, NULL, NULL, NULL};

    for (int L = 0; L < m->n_blocks; L++) {
        da3_block *blk = &m->blocks[L];

        /* LayerNorm 1 */
        da3_layernorm_batch(ln_buf, hidden, &blk->ln1_w, &blk->ln1_b,
                           nt, dim, m->ln_eps);

        /* QKV projection */
        da3_batch_gemm(qkv, &blk->attn_qkv_w, &blk->attn_qkv_b,
                       ln_buf, nt, 3 * dim, dim, n_threads);

        /* QK normalization (layers >= qk_norm_start_layer) */
        if (L >= m->qk_norm_start_layer && blk->attn_q_norm_w.data) {
            /* Q is at qkv[t][0..dim-1], K at qkv[t][dim..2*dim-1] */
            /* Extract Q and K as contiguous for norm, then put back */
            float *q_flat = (float *)malloc((size_t)nt * dim * sizeof(float));
            float *k_flat = (float *)malloc((size_t)nt * dim * sizeof(float));
            for (int t = 0; t < nt; t++) {
                memcpy(q_flat + t * dim, qkv + t * 3 * dim, (size_t)dim * sizeof(float));
                memcpy(k_flat + t * dim, qkv + t * 3 * dim + dim, (size_t)dim * sizeof(float));
            }
            da3_qk_norm_batch(q_flat, nt, m->n_heads, m->head_dim,
                              &blk->attn_q_norm_w, &blk->attn_q_norm_b, m->ln_eps);
            da3_qk_norm_batch(k_flat, nt, m->n_heads, m->head_dim,
                              &blk->attn_k_norm_w, &blk->attn_k_norm_b, m->ln_eps);
            for (int t = 0; t < nt; t++) {
                memcpy(qkv + t * 3 * dim, q_flat + t * dim, (size_t)dim * sizeof(float));
                memcpy(qkv + t * 3 * dim + dim, k_flat + t * dim, (size_t)dim * sizeof(float));
            }
            free(q_flat); free(k_flat);
        }

        /* RoPE 2D (layers >= rope_start_layer, applied to Q and K in qkv) */
        if (L >= m->rope_start_layer) {
            /* Apply RoPE to Q (patch tokens only, skip CLS) */
            float *q_flat = (float *)malloc((size_t)nt * dim * sizeof(float));
            float *k_flat = (float *)malloc((size_t)nt * dim * sizeof(float));
            for (int t = 0; t < nt; t++) {
                memcpy(q_flat + t * dim, qkv + t * 3 * dim, (size_t)dim * sizeof(float));
                memcpy(k_flat + t * dim, qkv + t * 3 * dim + dim, (size_t)dim * sizeof(float));
            }
            /* Only apply to patch tokens (1..nt-1), leave CLS unchanged */
            da3_rope_2d_batch(q_flat + dim, np, m->n_heads, m->head_dim,
                              pos_y + 1, pos_x + 1, 10000.0f);
            da3_rope_2d_batch(k_flat + dim, np, m->n_heads, m->head_dim,
                              pos_y + 1, pos_x + 1, 10000.0f);
            for (int t = 0; t < nt; t++) {
                memcpy(qkv + t * 3 * dim, q_flat + t * dim, (size_t)dim * sizeof(float));
                memcpy(qkv + t * 3 * dim + dim, k_flat + t * dim, (size_t)dim * sizeof(float));
            }
            free(q_flat); free(k_flat);
        }

        /* Multi-head attention */
        da3_attention(attn_out, qkv, nt, dim, m->n_heads, m->head_dim, n_threads);

        /* Attention output projection */
        float *proj_out = (float *)malloc((size_t)nt * dim * sizeof(float));
        da3_batch_gemm(proj_out, &blk->attn_out_w, &blk->attn_out_b,
                       attn_out, nt, dim, dim, n_threads);

        /* LayerScale 1 + residual */
        da3_layerscale(proj_out, &blk->ls1, nt, dim);
        for (int i = 0; i < nt * dim; i++) hidden[i] += proj_out[i];
        free(proj_out);

        /* LayerNorm 2 */
        da3_layernorm_batch(ln_buf, hidden, &blk->ln2_w, &blk->ln2_b,
                           nt, dim, m->ln_eps);

        /* FFN */
        float *ffn_out = (float *)malloc((size_t)nt * dim * sizeof(float));
        if (m->use_swiglu && blk->ffn_gate_up_w.data) {
            /* SwiGLU: gate_up = ln_buf @ w12^T + b12 */
            int gu_dim = blk->ffn_gate_up_w.n_rows; /* 2*hidden */
            int hidden_dim = gu_dim / 2;
            da3_batch_gemm(ffn_buf, &blk->ffn_gate_up_w, &blk->ffn_gate_up_b,
                           ln_buf, nt, gu_dim, dim, n_threads);
            /* SiLU(gate) * up */
            for (int t = 0; t < nt; t++) {
                float *gu = ffn_buf + t * gu_dim;
                float *mid = ffn_mid + t * hidden_dim;
                for (int i = 0; i < hidden_dim; i++) {
                    float gate = gu[i];
                    gate = gate / (1.0f + expf(-gate));
                    mid[i] = gate * gu[i + hidden_dim];
                }
            }
            /* Down: ffn_out = mid @ w3^T + b3 */
            da3_batch_gemm(ffn_out, &blk->ffn_down_w, &blk->ffn_down_b,
                           ffn_mid, nt, dim, hidden_dim, n_threads);
        } else if (blk->ffn_up_w.data) {
            /* GELU MLP: fc1 -> GELU -> fc2 */
            int fd = blk->ffn_up_w.n_rows;
            da3_batch_gemm(ffn_buf, &blk->ffn_up_w, &blk->ffn_up_b,
                           ln_buf, nt, fd, dim, n_threads);
            /* GELU */
            for (int i = 0; i < nt * fd; i++) {
                float v = ffn_buf[i];
                ffn_buf[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
            }
            da3_batch_gemm(ffn_out, &blk->ffn_down_w, &blk->ffn_down_b,
                           ffn_buf, nt, dim, fd, n_threads);
        }

        /* LayerScale 2 + residual */
        da3_layerscale(ffn_out, &blk->ls2, nt, dim);
        for (int i = 0; i < nt * dim; i++) hidden[i] += ffn_out[i];
        free(ffn_out);

        /* Save features if this is a feature layer */
        for (int fi = 0; fi < 4; fi++) {
            if (L == m->feature_layers[fi]) {
                features[fi] = (float *)malloc((size_t)nt * dim * sizeof(float));
                memcpy(features[fi], hidden, (size_t)nt * dim * sizeof(float));
            }
        }
    }

    free(hidden); free(ln_buf); free(qkv); free(attn_out);
    free(ffn_buf); free(ffn_mid); free(pos_y); free(pos_x);
    double t_dpt = da3_time_ms();
    fprintf(stderr, "da3: preprocess+embed %.1f ms, backbone %.1f ms (%d threads)\n",
            t_backbone - t_start, t_dpt - t_backbone, n_threads);

    /* ─── Step 5: DPT Head ─── */
    if (!m->head.proj_w[0].data) {
        fprintf(stderr, "da3: no DPT head weights, returning empty result\n");
        for (int i = 0; i < 4; i++) free(features[i]);
        return result;
    }

    int feat = m->head_features;  /* 64 */
    int head_dim_in = dim * 2;    /* CLS concat: 768 */

    /* 5a: Token processing + projection for each feature level */
    float *spatial[4] = {NULL};
    int sp_h[4], sp_w[4];
    for (int fi = 0; fi < 4; fi++) {
        if (!features[fi]) continue;
        int oc = m->head_out_channels[fi];

        /* Extract patch tokens + concat CLS */
        float *cat = (float *)malloc((size_t)np * head_dim_in * sizeof(float));
        float *cls_vec = features[fi]; /* CLS at index 0 */
        for (int p = 0; p < np; p++) {
            float *dst = cat + p * head_dim_in;
            memcpy(dst, features[fi] + (1 + p) * dim, (size_t)dim * sizeof(float));
            memcpy(dst + dim, cls_vec, (size_t)dim * sizeof(float));
        }

        /* Apply head norm */
        if (m->head.norm_w.data) {
            float *normed = (float *)malloc((size_t)np * head_dim_in * sizeof(float));
            da3_layernorm_batch(normed, cat, &m->head.norm_w, &m->head.norm_b,
                               np, head_dim_in, m->ln_eps);
            free(cat);
            cat = normed;
        }

        /* Project: 1x1 conv = GEMM [np, head_dim_in] -> [np, oc] */
        float *proj = (float *)malloc((size_t)np * oc * sizeof(float));
        da3_batch_gemm(proj, &m->head.proj_w[fi], &m->head.proj_b[fi],
                       cat, np, oc, head_dim_in, n_threads);
        free(cat);

        /* Reshape to spatial CHW: [oc, gh, gw] */
        /* proj is [np, oc] = [gh*gw, oc] in token-major. Transpose to CHW. */
        float *chw = (float *)malloc((size_t)oc * gh * gw * sizeof(float));
        for (int p = 0; p < np; p++) {
            int ph = p / gw, pw_idx = p % gw;
            for (int c = 0; c < oc; c++)
                chw[c * gh * gw + ph * gw + pw_idx] = proj[p * oc + c];
        }
        free(proj);
        free(features[fi]);
        features[fi] = NULL;

        /* Spatial alignment via resize */
        if (fi == 0) {
            /* ConvTranspose2d 4x4 stride 4 */
            spatial[0] = da3_conv_transpose2d_qt(chw, &m->head.upsample_0_w,
                            &m->head.upsample_0_b, gh, gw, oc, oc, 4, 4, 4,
                            &sp_h[0], &sp_w[0]);
        } else if (fi == 1) {
            /* ConvTranspose2d 2x2 stride 2 */
            spatial[1] = da3_conv_transpose2d_qt(chw, &m->head.upsample_1_w,
                            &m->head.upsample_1_b, gh, gw, oc, oc, 2, 2, 2,
                            &sp_h[1], &sp_w[1]);
        } else if (fi == 2) {
            /* Identity */
            spatial[2] = chw; chw = NULL;
            sp_h[2] = gh; sp_w[2] = gw;
        } else {
            /* Conv2d 3x3 stride 2 pad 1 */
            spatial[3] = da3_conv2d_qt(chw, &m->head.downsample_w,
                            &m->head.downsample_b, gh, gw, oc, oc, 3, 3, 2, 1,
                            &sp_h[3], &sp_w[3]);
        }
        free(chw);
    }

    /* 5b: Adapter layers (3x3 conv -> features channels) */
    float *adapted[4] = {NULL};
    int ad_h[4], ad_w[4];
    for (int fi = 0; fi < 4; fi++) {
        if (!spatial[fi] || !m->head.adapter_w[fi].data) continue;
        int oc = m->head_out_channels[fi];
        adapted[fi] = da3_conv2d_qt(spatial[fi], &m->head.adapter_w[fi], NULL,
                                     sp_h[fi], sp_w[fi], oc, feat, 3, 3, 1, 1,
                                     &ad_h[fi], &ad_w[fi]);
        free(spatial[fi]); spatial[fi] = NULL;
    }

    /* 5c: Bottom-up fusion */
    /* Level 3 (deepest) */
    float *fused = da3_refinenet(&m->head, 3, adapted[3], ad_h[3], ad_w[3],
                                  NULL, 0, 0, feat);
    free(adapted[3]);
    int fh = ad_h[3], fw = ad_w[3];

    /* Level 2 */
    float *fused2 = da3_refinenet(&m->head, 2, adapted[2], ad_h[2], ad_w[2],
                                   fused, fh, fw, feat);
    free(adapted[2]); free(fused);
    fh = ad_h[2]; fw = ad_w[2]; fused = fused2;

    /* Level 1 */
    fused2 = da3_refinenet(&m->head, 1, adapted[1], ad_h[1], ad_w[1],
                            fused, fh, fw, feat);
    free(adapted[1]); free(fused);
    fh = ad_h[1]; fw = ad_w[1]; fused = fused2;

    /* Level 0 */
    fused2 = da3_refinenet(&m->head, 0, adapted[0], ad_h[0], ad_w[0],
                            fused, fh, fw, feat);
    free(adapted[0]); free(fused);
    fh = ad_h[0]; fw = ad_w[0]; fused = fused2;

    /* 5d: Output convolutions — derive Co from weight shapes */
    int neck_Co = m->head.neck_w.n_rows;   /* e.g. 32 (small) or 64 (base) */
    int nh, nw;
    float *neck_out = da3_conv2d_qt(fused, &m->head.neck_w, &m->head.neck_b,
                                     fh, fw, feat, neck_Co, 3, 3, 1, 1, &nh, &nw);
    free(fused);
    for (int i = 0; i < neck_Co * nh * nw; i++)
        neck_out[i] = neck_out[i] > 0 ? neck_out[i] : 0.0f;

    int out0_Co = m->head.out_0_w.n_rows;  /* e.g. 32 (both small and base) */
    int o0h, o0w;
    float *out0 = da3_conv2d_qt(neck_out, &m->head.out_0_w, &m->head.out_0_b,
                                 nh, nw, neck_Co, out0_Co, 3, 3, 1, 1, &o0h, &o0w);
    free(neck_out);

    /* ReLU */
    for (int i = 0; i < out0_Co * o0h * o0w; i++)
        out0[i] = out0[i] > 0 ? out0[i] : 0.0f;

    int out_dim = m->head.out_2_w.n_rows;  /* 2 */
    int oh, ow;
    float *out2 = da3_conv2d_qt(out0, &m->head.out_2_w, &m->head.out_2_b,
                                 o0h, o0w, out0_Co, out_dim, 1, 1, 1, 0, &oh, &ow);
    free(out0);

    /* 5e: Activation + upsample to original resolution */
    result.width = img_w;
    result.height = img_h;
    result.depth = (float *)malloc((size_t)img_w * img_h * sizeof(float));
    result.confidence = (float *)malloc((size_t)img_w * img_h * sizeof(float));

    /* Separate depth (channel 0) and confidence (channel 1) */
    float *depth_small = out2;                       /* [oh, ow] */
    float *conf_small  = out2 + oh * ow;             /* [oh, ow] */

    /* Apply activations: exp(depth), expp1(confidence) */
    for (int i = 0; i < oh * ow; i++) {
        depth_small[i] = expf(depth_small[i]);
        conf_small[i] = expf(conf_small[i]) + 1.0f;
    }

    /* Bilinear upsample to original resolution */
    da3_bilinear(result.depth, depth_small, 1, oh, ow, img_h, img_w);
    da3_bilinear(result.confidence, conf_small, 1, oh, ow, img_h, img_w);

    free(out2);
    double t_end = da3_time_ms();
    fprintf(stderr, "da3: DPT head %.1f ms, total %.1f ms\n",
            t_end - t_dpt, t_end - t_start);
    return result;
}

void da3_result_free(da3_result *r) {
    free(r->depth);
    free(r->confidence);
    r->depth = NULL;
    r->confidence = NULL;
}

/* ==================================================================== */
/* Channel LayerNorm: normalize across C channels at each (h,w) pos     */
/* ==================================================================== */

static void da3_channel_layernorm(float *dst, const float *src, const qtensor *w,
                                   const qtensor *b, int C, int HW, float eps) {
    float *wf = NULL, *bf = NULL;
    if (w && w->data) { wf = da3_dequant(w); }
    if (b && b->data) { bf = da3_dequant(b); }
    for (int hw = 0; hw < HW; hw++) {
        float mean = 0.0f;
        for (int c = 0; c < C; c++) mean += src[c * HW + hw];
        mean /= (float)C;
        float var = 0.0f;
        for (int c = 0; c < C; c++) {
            float d = src[c * HW + hw] - mean;
            var += d * d;
        }
        var /= (float)C;
        float inv = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < C; c++) {
            float val = src[c * HW + hw];
            float sc = wf ? wf[c] : 1.0f;
            float bi = bf ? bf[c] : 0.0f;
            dst[c * HW + hw] = (val - mean) * inv * sc + bi;
        }
    }
    free(wf); free(bf);
}

/* ==================================================================== */
/* RefineNet with explicit weight pointers (for aux/gsdpt branches)     */
/* ==================================================================== */

static float *da3_refinenet_w(const float *feat, int fH, int fW,
                                const float *deeper, int dH, int dW,
                                int features,
                                const qtensor *out_w, const qtensor *out_b,
                                const qtensor *rcu1_c1_w, const qtensor *rcu1_c1_b,
                                const qtensor *rcu1_c2_w, const qtensor *rcu1_c2_b,
                                const qtensor *rcu2_c1_w, const qtensor *rcu2_c1_b,
                                const qtensor *rcu2_c2_w, const qtensor *rcu2_c2_b) {
    int sz = features * fH * fW;
    float *output = (float *)malloc((size_t)sz * sizeof(float));
    memcpy(output, feat, (size_t)sz * sizeof(float));

    if (deeper) {
        float *up = (float *)malloc((size_t)sz * sizeof(float));
        da3_bilinear(up, deeper, features, dH, dW, fH, fW);
        if (rcu1_c1_w && rcu1_c1_w->data) {
            float *c1w = da3_dequant(rcu1_c1_w);
            float *c1b = da3_dequant(rcu1_c1_b);
            float *c2w = da3_dequant(rcu1_c2_w);
            float *c2b = da3_dequant(rcu1_c2_b);
            float *rcu_out = (float *)malloc((size_t)sz * sizeof(float));
            da3_rcu(rcu_out, up, c1w, c1b, c2w, c2b, features, fH, fW);
            for (int i = 0; i < sz; i++) output[i] += rcu_out[i];
            free(rcu_out); free(c1w); free(c1b); free(c2w); free(c2b);
        } else {
            for (int i = 0; i < sz; i++) output[i] += up[i];
        }
        free(up);
    }

    if (rcu2_c1_w && rcu2_c1_w->data) {
        float *c1w = da3_dequant(rcu2_c1_w);
        float *c1b = da3_dequant(rcu2_c1_b);
        float *c2w = da3_dequant(rcu2_c2_w);
        float *c2b = da3_dequant(rcu2_c2_b);
        float *rcu_out = (float *)malloc((size_t)sz * sizeof(float));
        da3_rcu(rcu_out, output, c1w, c1b, c2w, c2b, features, fH, fW);
        memcpy(output, rcu_out, (size_t)sz * sizeof(float));
        free(rcu_out); free(c1w); free(c1b); free(c2w); free(c2b);
    }

    float *ow_f = da3_dequant(out_w);
    float *ob_f = da3_dequant(out_b);
    float *conv_out = (float *)calloc((size_t)sz, sizeof(float));
    da3_conv2d(conv_out, output, ow_f, ob_f, fH, fW, features, features, 1, 1, 1, 0);
    free(output); free(ow_f); free(ob_f);
    return conv_out;
}

/* ==================================================================== */
/* CameraDec: backbone_norm(CLS) → MLP(GELU) → 3 heads → pose[9]       */
/* ==================================================================== */

/* noinline for code organization: keep da3_predict_full manageable */
__attribute__((noinline))
static void da3_run_camera_dec(da3_model *m, const float *hidden, float *pose) {
    int dim = m->dim;
    da3_cam_dec *cd = &m->cam_dec;
    int dim_in = cd->mlp0_w.n_cols; /* input dim (typically 2*dim=768 for CLS concat) */

    /* LayerNorm on CLS token (token 0) */
    float *cls_normed = (float *)malloc((size_t)dim * sizeof(float));
    da3_layernorm_batch(cls_normed, hidden, &m->backbone_norm_w, &m->backbone_norm_b,
                        1, dim, m->ln_eps);

    /* If dim_in > dim, concatenate CLS with itself (same as DPT readout) */
    float *cam_input;
    if (dim_in > dim) {
        cam_input = (float *)malloc((size_t)dim_in * sizeof(float));
        memcpy(cam_input, cls_normed, (size_t)dim * sizeof(float));
        memcpy(cam_input + dim, cls_normed, (size_t)dim * sizeof(float));
    } else {
        cam_input = cls_normed;
    }

    /* MLP layer 0: [mlp_dim, dim_in] × [dim_in] + bias → GELU */
    float *mlp_h = (float *)calloc(cd->mlp_dim, sizeof(float));
    da3_batch_gemm(mlp_h, &cd->mlp0_w, &cd->mlp0_b, cam_input, 1, cd->mlp_dim, dim_in, 1);
    for (int i = 0; i < cd->mlp_dim; i++) {
        float v = mlp_h[i];
        mlp_h[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }

    /* MLP layer 2: [mlp_dim, mlp_dim] × [mlp_dim] + bias → GELU */
    float *mlp_h2 = (float *)calloc(cd->mlp_dim, sizeof(float));
    da3_batch_gemm(mlp_h2, &cd->mlp2_w, &cd->mlp2_b, mlp_h, 1, cd->mlp_dim, cd->mlp_dim, 1);
    for (int i = 0; i < cd->mlp_dim; i++) {
        float v = mlp_h2[i];
        mlp_h2[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }

    /* 3 output heads (small matmuls on CPU) */
    float *fc_t_w = da3_dequant(&cd->fc_t_w);
    float *fc_t_b = da3_dequant(&cd->fc_t_b);
    for (int o = 0; o < 3; o++) {
        float s = fc_t_b ? fc_t_b[o] : 0.0f;
        for (int k = 0; k < cd->mlp_dim; k++) s += fc_t_w[o * cd->mlp_dim + k] * mlp_h2[k];
        pose[o] = s;
    }
    float *fc_q_w = da3_dequant(&cd->fc_qvec_w);
    float *fc_q_b = da3_dequant(&cd->fc_qvec_b);
    for (int o = 0; o < 4; o++) {
        float s = fc_q_b ? fc_q_b[o] : 0.0f;
        for (int k = 0; k < cd->mlp_dim; k++) s += fc_q_w[o * cd->mlp_dim + k] * mlp_h2[k];
        pose[3 + o] = s;
    }
    float *fc_f_w = da3_dequant(&cd->fc_fov_w);
    float *fc_f_b = da3_dequant(&cd->fc_fov_b);
    for (int o = 0; o < 2; o++) {
        float s = fc_f_b ? fc_f_b[o] : 0.0f;
        for (int k = 0; k < cd->mlp_dim; k++) s += fc_f_w[o * cd->mlp_dim + k] * mlp_h2[k];
        pose[7 + o] = s;
    }

    if (cam_input != cls_normed) free(cam_input);
    free(cls_normed); free(mlp_h); free(mlp_h2);
    free(fc_t_w); free(fc_t_b); free(fc_q_w); free(fc_q_b); free(fc_f_w); free(fc_f_b);
}

/* ==================================================================== */
/* Aux DPT: run aux refinenet + output conv chains → rays + confidence  */
/* ==================================================================== */

__attribute__((noinline))
static void da3_run_aux_dpt(da3_model *m, float **adapted, int *ad_h, int *ad_w,
                              int feat, float **rays_out, float **ray_conf_out,
                              int *out_h, int *out_w) {
    da3_aux_head *ah = &m->aux_head;

    /* Bottom-up aux RefineNet fusion */
    float *fused = da3_refinenet_w(adapted[3], ad_h[3], ad_w[3], NULL, 0, 0, feat,
                                     &ah->fuse_out_w[3], &ah->fuse_out_b[3],
                                     &ah->fuse_rcu1_c1_w[3], &ah->fuse_rcu1_c1_b[3],
                                     &ah->fuse_rcu1_c2_w[3], &ah->fuse_rcu1_c2_b[3],
                                     &ah->fuse_rcu2_c1_w[3], &ah->fuse_rcu2_c1_b[3],
                                     &ah->fuse_rcu2_c2_w[3], &ah->fuse_rcu2_c2_b[3]);
    int fh = ad_h[3], fw = ad_w[3];

    for (int lv = 2; lv >= 0; lv--) {
        float *fused2 = da3_refinenet_w(adapted[lv], ad_h[lv], ad_w[lv], fused, fh, fw, feat,
                                          &ah->fuse_out_w[lv], &ah->fuse_out_b[lv],
                                          &ah->fuse_rcu1_c1_w[lv], &ah->fuse_rcu1_c1_b[lv],
                                          &ah->fuse_rcu1_c2_w[lv], &ah->fuse_rcu1_c2_b[lv],
                                          &ah->fuse_rcu2_c1_w[lv], &ah->fuse_rcu2_c1_b[lv],
                                          &ah->fuse_rcu2_c2_w[lv], &ah->fuse_rcu2_c2_b[lv]);
        free(fused);
        fh = ad_h[lv]; fw = ad_w[lv];
        fused = fused2;
    }

    /* Only level 3 (finest = level 0 in the fused pyramid after bottom-up) output is used.
     * After bottom-up fusion, fused is at level 0 spatial size. */
    int oh = fh, ow = fw;
    int lv = 3; /* use level 3 weights for output conv chains */

    /* output_conv1_aux: sequential Conv3x3 (F32, NO activations) */
    float *cur = (float *)malloc((size_t)feat * oh * ow * sizeof(float));
    memcpy(cur, fused, (size_t)feat * oh * ow * sizeof(float));
    free(fused);
    int ci = feat;
    for (int j = 0; j < ah->oc1_count[lv]; j++) {
        int co = ah->oc1_co[lv][j];
        int ci_j = ah->oc1_ci[lv][j];
        (void)ci_j; /* ci should match */
        float *w = da3_dequant(&ah->oc1_w[lv][j]);
        float *b = da3_dequant(&ah->oc1_b[lv][j]);
        float *next = (float *)calloc((size_t)co * oh * ow, sizeof(float));
        da3_conv2d(next, cur, w, b, oh, ow, ci, co, 3, 3, 1, 1);
        free(w); free(b); free(cur);
        cur = next;
        ci = co;
    }

    /* output_conv2_aux[lv]: Conv2d(ci,32,3) + channel_layernorm(32) + ReLU + Conv2d(32,7,1) */
    int mid_ch = 32;
    if (ah->oc2_conv_w[lv].data && ah->oc2_conv_w[lv].n_dims >= 4)
        mid_ch = (int)ah->oc2_conv_w[lv].dims[0];
    {
        float *cw = da3_dequant(&ah->oc2_conv_w[lv]);
        float *cb = da3_dequant(&ah->oc2_conv_b[lv]);
        float *mid = (float *)calloc((size_t)mid_ch * oh * ow, sizeof(float));
        da3_conv2d(mid, cur, cw, cb, oh, ow, ci, mid_ch, 3, 3, 1, 1);
        free(cw); free(cb); free(cur);

        /* Channel LayerNorm */
        float *ln_out = (float *)malloc((size_t)mid_ch * oh * ow * sizeof(float));
        da3_channel_layernorm(ln_out, mid, &ah->oc2_gn_w[lv], &ah->oc2_gn_b[lv],
                               mid_ch, oh * ow, m->ln_eps);
        free(mid);

        /* ReLU */
        for (int i = 0; i < mid_ch * oh * ow; i++)
            ln_out[i] = ln_out[i] > 0 ? ln_out[i] : 0.0f;

        /* Final Conv2d(32, 7, 1) */
        float *ow_f = da3_dequant(&ah->oc2_out_w[lv]);
        float *ob_f = da3_dequant(&ah->oc2_out_b[lv]);
        int out_ch = 7;
        if (ah->oc2_out_w[lv].n_dims >= 4)
            out_ch = (int)ah->oc2_out_w[lv].dims[0];
        float *out = (float *)calloc((size_t)out_ch * oh * ow, sizeof(float));
        da3_conv2d(out, ln_out, ow_f, ob_f, oh, ow, mid_ch, out_ch, 1, 1, 1, 0);
        free(ln_out); free(ow_f); free(ob_f);

        /* Split: rays[0:6] (linear), ray_confidence[6] (expp1) */
        int npix = oh * ow;
        *rays_out = (float *)malloc((size_t)6 * npix * sizeof(float));
        memcpy(*rays_out, out, (size_t)6 * npix * sizeof(float));
        *ray_conf_out = (float *)malloc((size_t)npix * sizeof(float));
        for (int i = 0; i < npix; i++)
            (*ray_conf_out)[i] = expf(out[6 * npix + i]) + 1.0f;
        free(out);
    }

    *out_h = oh;
    *out_w = ow;
}

/* ==================================================================== */
/* GSDPT: images_merger + DPT pipeline → 38ch gaussians                 */
/* ==================================================================== */

__attribute__((noinline))
static void da3_run_gsdpt(da3_model *m, const float *img_norm, int target_h, int target_w,
                            float **features, int np, int dim, int n_threads,
                            float **gaussians_out, int *gs_out_h, int *gs_out_w) {
    da3_gsdpt *gs = &m->gsdpt;
    da3_dpt_head *gh = &gs->head;
    int feat = m->head_features;
    int gh_grid = m->grid_h, gw_grid = m->grid_w;
    int head_dim_in = dim * 2;
    int gs_oc = gs->gs_out_channels;
    if (gs_oc < 2) gs_oc = 38;

    /* 1. Images merger: 3 stride-2 Conv2d on img_norm → [128, mg_h, mg_w] */
    float *merger_out = NULL;
    int mh = target_h, mw = target_w;
    {
        float *cur = (float *)malloc((size_t)3 * mh * mw * sizeof(float));
        memcpy(cur, img_norm, (size_t)3 * mh * mw * sizeof(float));
        for (int mi = 0; mi < 3; mi++) {
            if (!gs->merger_w[mi].data) break;
            int mci = gs->merger_ci[mi];
            int mco = gs->merger_co[mi];
            int moh = (mh + 2 * 1 - 3) / 2 + 1;
            int mow = (mw + 2 * 1 - 3) / 2 + 1;
            float *w = da3_dequant(&gs->merger_w[mi]);
            float *b = da3_dequant(&gs->merger_b[mi]);
            float *next = (float *)calloc((size_t)mco * moh * mow, sizeof(float));
            da3_conv2d(next, cur, w, b, mh, mw, mci, mco, 3, 3, 2, 1);
            free(w); free(b); free(cur);
            /* SiLU after layers 0, 1 but NOT after layer 2 */
            if (mi < 2) {
                for (int i = 0; i < mco * moh * mow; i++)
                    next[i] = next[i] / (1.0f + expf(-next[i]));
            }
            cur = next;
            mh = moh; mw = mow;
        }
        merger_out = cur;
    }
    int mg_h = mh, mg_w = mw;

    /* 2. GSDPT DPT pipeline (same as main DPT but with GSDPT weights) */
    float *spatial[4] = {NULL};
    int sp_h[4], sp_w[4];
    for (int fi = 0; fi < 4; fi++) {
        if (!features[fi] || !gh->proj_w[fi].data) continue;
        int oc = m->head_out_channels[fi];

        float *cat = (float *)malloc((size_t)np * head_dim_in * sizeof(float));
        float *cls_vec = features[fi];
        for (int p = 0; p < np; p++) {
            float *dst = cat + p * head_dim_in;
            memcpy(dst, features[fi] + (1 + p) * dim, (size_t)dim * sizeof(float));
            memcpy(dst + dim, cls_vec, (size_t)dim * sizeof(float));
        }

        if (gh->norm_w.data) {
            float *normed = (float *)malloc((size_t)np * head_dim_in * sizeof(float));
            da3_layernorm_batch(normed, cat, &gh->norm_w, &gh->norm_b,
                                np, head_dim_in, m->ln_eps);
            free(cat);
            cat = normed;
        }

        float *proj = (float *)malloc((size_t)np * oc * sizeof(float));
        da3_batch_gemm(proj, &gh->proj_w[fi], &gh->proj_b[fi],
                        cat, np, oc, head_dim_in, n_threads);
        free(cat);

        float *chw = (float *)malloc((size_t)oc * gh_grid * gw_grid * sizeof(float));
        for (int p = 0; p < np; p++) {
            int ph = p / gw_grid, pw_idx = p % gw_grid;
            for (int c = 0; c < oc; c++)
                chw[c * gh_grid * gw_grid + ph * gw_grid + pw_idx] = proj[p * oc + c];
        }
        free(proj);

        if (fi == 0) {
            spatial[0] = da3_conv_transpose2d_qt(chw, &gh->upsample_0_w, &gh->upsample_0_b,
                            gh_grid, gw_grid, oc, oc, 4, 4, 4, &sp_h[0], &sp_w[0]);
        } else if (fi == 1) {
            spatial[1] = da3_conv_transpose2d_qt(chw, &gh->upsample_1_w, &gh->upsample_1_b,
                            gh_grid, gw_grid, oc, oc, 2, 2, 2, &sp_h[1], &sp_w[1]);
        } else if (fi == 2) {
            spatial[2] = chw; chw = NULL;
            sp_h[2] = gh_grid; sp_w[2] = gw_grid;
        } else {
            spatial[3] = da3_conv2d_qt(chw, &gh->downsample_w, &gh->downsample_b,
                            gh_grid, gw_grid, oc, oc, 3, 3, 2, 1, &sp_h[3], &sp_w[3]);
        }
        free(chw);
    }

    /* Adapters */
    float *adapted[4] = {NULL};
    int ad_h[4], ad_w[4];
    for (int fi = 0; fi < 4; fi++) {
        if (!spatial[fi] || !gh->adapter_w[fi].data) continue;
        int oc = m->head_out_channels[fi];
        adapted[fi] = da3_conv2d_qt(spatial[fi], &gh->adapter_w[fi], NULL,
                                      sp_h[fi], sp_w[fi], oc, feat, 3, 3, 1, 1,
                                      &ad_h[fi], &ad_w[fi]);
        free(spatial[fi]); spatial[fi] = NULL;
    }

    /* Bottom-up RefineNet fusion */
    float *fused = da3_refinenet(&gs->head, 3, adapted[3], ad_h[3], ad_w[3],
                                   NULL, 0, 0, feat);
    free(adapted[3]);
    int fh = ad_h[3], fw = ad_w[3];
    for (int lv = 2; lv >= 0; lv--) {
        float *fused2 = da3_refinenet(&gs->head, lv, adapted[lv], ad_h[lv], ad_w[lv],
                                        fused, fh, fw, feat);
        free(adapted[lv]); free(fused);
        fh = ad_h[lv]; fw = ad_w[lv]; fused = fused2;
    }
    int gs_fh = fh, gs_fw = fw;

    /* Output convolutions: neck (no ReLU for GSDPT) + merger inject + out */
    int gs_neck_Co = gh->neck_w.n_rows;
    int nk_h, nk_w;
    float *neck_out = da3_conv2d_qt(fused, &gh->neck_w, &gh->neck_b,
                                      gs_fh, gs_fw, feat, gs_neck_Co, 3, 3, 1, 1,
                                      &nk_h, &nk_w);
    free(fused);
    /* No ReLU for GSDPT neck (unlike main DPT) */

    /* Inject merger features: bilinear upsample + element-wise add */
    if (merger_out) {
        float *up_merger = (float *)malloc((size_t)gs_neck_Co * nk_h * nk_w * sizeof(float));
        da3_bilinear(up_merger, merger_out, gs_neck_Co, mg_h, mg_w, nk_h, nk_w);
        for (int i = 0; i < gs_neck_Co * nk_h * nk_w; i++)
            neck_out[i] += up_merger[i];
        free(up_merger);
    }
    free(merger_out);

    /* out_0: Conv2d(gs_neck_Co, out0_Co, 3) + ReLU */
    int gs_out0_Co = gh->out_0_w.n_rows;
    int o0h, o0w;
    float *out0 = da3_conv2d_qt(neck_out, &gh->out_0_w, &gh->out_0_b,
                                  nk_h, nk_w, gs_neck_Co, gs_out0_Co, 3, 3, 1, 1,
                                  &o0h, &o0w);
    free(neck_out);
    for (int i = 0; i < gs_out0_Co * o0h * o0w; i++)
        out0[i] = out0[i] > 0 ? out0[i] : 0.0f;

    /* out_2: Conv2d(gs_out0_Co, gs_oc, 1) */
    int o2h, o2w;
    float *out2 = da3_conv2d_qt(out0, &gh->out_2_w, &gh->out_2_b,
                                  o0h, o0w, gs_out0_Co, gs_oc, 1, 1, 1, 0,
                                  &o2h, &o2w);
    free(out0);

    *gaussians_out = out2;
    *gs_out_h = o2h;
    *gs_out_w = o2w;
}

/* Compute adapted features from backbone features (readout + proj + spatial + adapter).
 * Extracted for code organization: readout + projection + spatial + adapter. */
__attribute__((noinline))
static void da3_compute_adapted(da3_model *m, float **features, int np, int dim,
                                  int gh, int gw, int feat, int n_threads,
                                  float **adapted_out, int *ad_h_out, int *ad_w_out) {
    int head_dim_in = dim * 2;
    for (int fi = 0; fi < 4; fi++) {
        if (!features[fi] || !m->head.proj_w[fi].data) continue;
        int oc = m->head_out_channels[fi];
        float *cat = (float *)malloc((size_t)np * head_dim_in * sizeof(float));
        for (int p = 0; p < np; p++) {
            float *dst = cat + p * head_dim_in;
            memcpy(dst, features[fi] + (1 + p) * dim, (size_t)dim * sizeof(float));
            memcpy(dst + dim, features[fi], (size_t)dim * sizeof(float));
        }
        if (m->head.norm_w.data) {
            float *normed = (float *)malloc((size_t)np * head_dim_in * sizeof(float));
            da3_layernorm_batch(normed, cat, &m->head.norm_w, &m->head.norm_b, np, head_dim_in, m->ln_eps);
            free(cat); cat = normed;
        }
        float *proj = (float *)malloc((size_t)np * oc * sizeof(float));
        da3_batch_gemm(proj, &m->head.proj_w[fi], &m->head.proj_b[fi], cat, np, oc, head_dim_in, n_threads);
        free(cat);
        float *chw = (float *)malloc((size_t)oc * gh * gw * sizeof(float));
        for (int p = 0; p < np; p++) {
            int ph = p / gw, pw_idx = p % gw;
            for (int c = 0; c < oc; c++)
                chw[c * gh * gw + ph * gw + pw_idx] = proj[p * oc + c];
        }
        free(proj);
        int sph, spw;
        float *sp = NULL;
        if (fi == 0)      sp = da3_conv_transpose2d_qt(chw, &m->head.upsample_0_w, &m->head.upsample_0_b,
                                gh, gw, oc, oc, 4, 4, 4, &sph, &spw);
        else if (fi == 1) sp = da3_conv_transpose2d_qt(chw, &m->head.upsample_1_w, &m->head.upsample_1_b,
                                gh, gw, oc, oc, 2, 2, 2, &sph, &spw);
        else if (fi == 2) { sp = chw; chw = NULL; sph = gh; spw = gw; }
        else              sp = da3_conv2d_qt(chw, &m->head.downsample_w, &m->head.downsample_b,
                                gh, gw, oc, oc, 3, 3, 2, 1, &sph, &spw);
        free(chw);
        adapted_out[fi] = da3_conv2d_qt(sp, &m->head.adapter_w[fi], NULL,
                                          sph, spw, oc, feat, 3, 3, 1, 1,
                                          &ad_h_out[fi], &ad_w_out[fi]);
        free(sp);
    }
}

/* Run DPT fusion + output convs → depth/confidence.
 * Extracted for code organization: refinenet fusion + output convs + activation. */
__attribute__((noinline))
static void da3_run_dpt_output(da3_model *m, float **adapted, int *ad_h, int *ad_w,
                                 int feat, int img_w, int img_h,
                                 float **depth_out, float **conf_out) {
    /* Bottom-up fusion */
    float *fused = da3_refinenet(&m->head, 3, adapted[3], ad_h[3], ad_w[3], NULL, 0, 0, feat);
    free(adapted[3]); adapted[3] = NULL;
    int fh = ad_h[3], fw = ad_w[3];
    for (int lv = 2; lv >= 0; lv--) {
        float *fused2 = da3_refinenet(&m->head, lv, adapted[lv], ad_h[lv], ad_w[lv], fused, fh, fw, feat);
        free(adapted[lv]); adapted[lv] = NULL;
        free(fused);
        fh = ad_h[lv]; fw = ad_w[lv]; fused = fused2;
    }

    /* Output convs — derive Co from weight shapes */
    int neck_Co = m->head.neck_w.n_rows;
    int nh, nw;
    float *neck_out = da3_conv2d_qt(fused, &m->head.neck_w, &m->head.neck_b,
                                      fh, fw, feat, neck_Co, 3, 3, 1, 1, &nh, &nw);
    free(fused);
    for (int i = 0; i < neck_Co * nh * nw; i++)
        neck_out[i] = neck_out[i] > 0 ? neck_out[i] : 0.0f;

    int out0_Co = m->head.out_0_w.n_rows;
    int o0h, o0w;
    float *out0 = da3_conv2d_qt(neck_out, &m->head.out_0_w, &m->head.out_0_b,
                                  nh, nw, neck_Co, out0_Co, 3, 3, 1, 1, &o0h, &o0w);
    free(neck_out);
    for (int i = 0; i < out0_Co * o0h * o0w; i++)
        out0[i] = out0[i] > 0 ? out0[i] : 0.0f;

    int out_dim = m->head.out_2_w.n_rows;
    int doh, dow;
    float *out2 = da3_conv2d_qt(out0, &m->head.out_2_w, &m->head.out_2_b,
                                  o0h, o0w, out0_Co, out_dim, 1, 1, 1, 0, &doh, &dow);
    free(out0);

    /* Activations + upsample */
    *depth_out = (float *)malloc((size_t)img_w * img_h * sizeof(float));
    *conf_out = (float *)malloc((size_t)img_w * img_h * sizeof(float));
    float *depth_small = out2;
    float *conf_small = out2 + doh * dow;
    for (int i = 0; i < doh * dow; i++) {
        depth_small[i] = expf(depth_small[i]);
        conf_small[i] = expf(conf_small[i]) + 1.0f;
    }
    da3_bilinear(*depth_out, depth_small, 1, doh, dow, img_h, img_w);
    da3_bilinear(*conf_out, conf_small, 1, doh, dow, img_h, img_w);
    free(out2);
}

/* ==================================================================== */
/* API: da3_predict_full                                                */
/* ==================================================================== */

/* Backbone: preprocess + patch embed + CLS + transformer blocks.
 * Returns hidden (nt*dim CLS+patch tokens) and 4 feature maps.
 * Caller must free hidden and features[0..3]. */
__attribute__((noinline))
static void da3_run_backbone(da3_model *m, const uint8_t *rgb, int img_w, int img_h,
                               float *img_norm, int n_threads,
                               float **hidden_out, float **features_out) {
    int ps = m->patch_size;
    int gh = m->grid_h, gw = m->grid_w;
    int np = m->n_patches;
    int nt = m->n_tokens;
    int dim = m->dim;
    int target_h = gh * ps;
    int target_w = gw * ps;

    /* Step 1: Preprocess */
    for (int c = 0; c < 3; c++) {
        for (int oh = 0; oh < target_h; oh++) {
            float fy = (target_h > 1) ? (float)oh * (img_h - 1) / (target_h - 1) : 0.0f;
            int y0 = (int)fy; int y1 = y0 + 1 < img_h ? y0 + 1 : y0;
            float dy = fy - y0;
            for (int ow = 0; ow < target_w; ow++) {
                float fx = (target_w > 1) ? (float)ow * (img_w - 1) / (target_w - 1) : 0.0f;
                int x0 = (int)fx; int x1 = x0 + 1 < img_w ? x0 + 1 : x0;
                float dx = fx - x0;
                float v = (float)rgb[(y0 * img_w + x0) * 3 + c] * (1-dy)*(1-dx)
                        + (float)rgb[(y0 * img_w + x1) * 3 + c] * (1-dy)*dx
                        + (float)rgb[(y1 * img_w + x0) * 3 + c] * dy*(1-dx)
                        + (float)rgb[(y1 * img_w + x1) * 3 + c] * dy*dx;
                img_norm[c * target_h * target_w + oh * target_w + ow] =
                    (v / 255.0f - m->image_mean[c]) / m->image_std[c];
            }
        }
    }

    /* Step 2: Patch embedding */
    float *hidden = (float *)calloc((size_t)nt * dim, sizeof(float));
    {
        float *pw = da3_dequant(&m->patch_embed_w);
        float *pb = da3_dequant(&m->patch_embed_b);
        int Co = dim, Ci = 3;
        for (int py = 0; py < gh; py++) {
            for (int px = 0; px < gw; px++) {
                int tok = 1 + py * gw + px;
                float *out = hidden + tok * dim;
                if (pb) memcpy(out, pb, (size_t)dim * sizeof(float));
                for (int co = 0; co < Co; co++) {
                    float sum = 0.0f;
                    for (int ci = 0; ci < Ci; ci++)
                        for (int kh = 0; kh < ps; kh++)
                            for (int kw = 0; kw < ps; kw++)
                                sum += pw[((co * Ci + ci) * ps + kh) * ps + kw]
                                     * img_norm[ci * target_h * target_w + (py*ps+kh) * target_w + (px*ps+kw)];
                    out[co] += sum;
                }
            }
        }
        free(pw); free(pb);
    }

    /* Step 3: CLS + pos embed */
    {
        float *cls = da3_dequant(&m->cls_token);
        memcpy(hidden, cls, (size_t)dim * sizeof(float));
        free(cls);
        float *pe = da3_dequant(&m->pos_embed);
        for (int t = 0; t < nt; t++)
            for (int i = 0; i < dim; i++)
                hidden[t * dim + i] += pe[t * dim + i];
        free(pe);
    }

    int *pos_y = (int *)calloc((size_t)nt, sizeof(int));
    int *pos_x = (int *)calloc((size_t)nt, sizeof(int));
    for (int p = 0; p < np; p++) {
        pos_y[1 + p] = p / gw;
        pos_x[1 + p] = p % gw;
    }

    /* Step 4: Transformer blocks */
    float *ln_buf   = (float *)malloc((size_t)nt * dim * sizeof(float));
    float *qkv      = (float *)malloc((size_t)nt * 3 * dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)nt * dim * sizeof(float));
    int ffn_out_dim = m->use_swiglu ? 2 * m->ffn_hidden : (m->blocks[0].ffn_up_w.data ? m->blocks[0].ffn_up_w.n_rows : 4 * dim);
    float *ffn_buf  = (float *)malloc((size_t)nt * ffn_out_dim * sizeof(float));
    float *ffn_mid  = (float *)malloc((size_t)nt * m->ffn_hidden * sizeof(float));

    for (int fi = 0; fi < 4; fi++) features_out[fi] = NULL;

    for (int L = 0; L < m->n_blocks; L++) {
        da3_block *blk = &m->blocks[L];

        da3_layernorm_batch(ln_buf, hidden, &blk->ln1_w, &blk->ln1_b, nt, dim, m->ln_eps);
        da3_batch_gemm(qkv, &blk->attn_qkv_w, &blk->attn_qkv_b, ln_buf, nt, 3*dim, dim, n_threads);

        if (L >= m->qk_norm_start_layer && blk->attn_q_norm_w.data) {
            float *q_flat = (float *)malloc((size_t)nt * dim * sizeof(float));
            float *k_flat = (float *)malloc((size_t)nt * dim * sizeof(float));
            for (int t = 0; t < nt; t++) {
                memcpy(q_flat + t*dim, qkv + t*3*dim, (size_t)dim * sizeof(float));
                memcpy(k_flat + t*dim, qkv + t*3*dim + dim, (size_t)dim * sizeof(float));
            }
            da3_qk_norm_batch(q_flat, nt, m->n_heads, m->head_dim, &blk->attn_q_norm_w, &blk->attn_q_norm_b, m->ln_eps);
            da3_qk_norm_batch(k_flat, nt, m->n_heads, m->head_dim, &blk->attn_k_norm_w, &blk->attn_k_norm_b, m->ln_eps);
            for (int t = 0; t < nt; t++) {
                memcpy(qkv + t*3*dim, q_flat + t*dim, (size_t)dim * sizeof(float));
                memcpy(qkv + t*3*dim + dim, k_flat + t*dim, (size_t)dim * sizeof(float));
            }
            free(q_flat); free(k_flat);
        }

        if (L >= m->rope_start_layer) {
            float *q_flat = (float *)malloc((size_t)nt * dim * sizeof(float));
            float *k_flat = (float *)malloc((size_t)nt * dim * sizeof(float));
            for (int t = 0; t < nt; t++) {
                memcpy(q_flat + t*dim, qkv + t*3*dim, (size_t)dim * sizeof(float));
                memcpy(k_flat + t*dim, qkv + t*3*dim + dim, (size_t)dim * sizeof(float));
            }
            da3_rope_2d_batch(q_flat + dim, np, m->n_heads, m->head_dim, pos_y+1, pos_x+1, 10000.0f);
            da3_rope_2d_batch(k_flat + dim, np, m->n_heads, m->head_dim, pos_y+1, pos_x+1, 10000.0f);
            for (int t = 0; t < nt; t++) {
                memcpy(qkv + t*3*dim, q_flat + t*dim, (size_t)dim * sizeof(float));
                memcpy(qkv + t*3*dim + dim, k_flat + t*dim, (size_t)dim * sizeof(float));
            }
            free(q_flat); free(k_flat);
        }

        da3_attention(attn_out, qkv, nt, dim, m->n_heads, m->head_dim, n_threads);

        float *proj_out = (float *)malloc((size_t)nt * dim * sizeof(float));
        da3_batch_gemm(proj_out, &blk->attn_out_w, &blk->attn_out_b, attn_out, nt, dim, dim, n_threads);
        da3_layerscale(proj_out, &blk->ls1, nt, dim);
        for (int i = 0; i < nt * dim; i++) hidden[i] += proj_out[i];
        free(proj_out);

        da3_layernorm_batch(ln_buf, hidden, &blk->ln2_w, &blk->ln2_b, nt, dim, m->ln_eps);

        float *ffn_out = (float *)malloc((size_t)nt * dim * sizeof(float));
        if (m->use_swiglu && blk->ffn_gate_up_w.data) {
            int gu_dim = blk->ffn_gate_up_w.n_rows;
            int hidden_dim = gu_dim / 2;
            da3_batch_gemm(ffn_buf, &blk->ffn_gate_up_w, &blk->ffn_gate_up_b, ln_buf, nt, gu_dim, dim, n_threads);
            for (int t = 0; t < nt; t++) {
                float *gu = ffn_buf + t * gu_dim;
                float *mid = ffn_mid + t * hidden_dim;
                for (int i = 0; i < hidden_dim; i++) {
                    float gate = gu[i];
                    gate = gate / (1.0f + expf(-gate));
                    mid[i] = gate * gu[i + hidden_dim];
                }
            }
            da3_batch_gemm(ffn_out, &blk->ffn_down_w, &blk->ffn_down_b, ffn_mid, nt, dim, hidden_dim, n_threads);
        } else if (blk->ffn_up_w.data) {
            int fd = blk->ffn_up_w.n_rows;
            da3_batch_gemm(ffn_buf, &blk->ffn_up_w, &blk->ffn_up_b, ln_buf, nt, fd, dim, n_threads);
            for (int i = 0; i < nt * fd; i++) {
                float v = ffn_buf[i];
                ffn_buf[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
            }
            da3_batch_gemm(ffn_out, &blk->ffn_down_w, &blk->ffn_down_b, ffn_buf, nt, dim, fd, n_threads);
        }
        da3_layerscale(ffn_out, &blk->ls2, nt, dim);
        for (int i = 0; i < nt * dim; i++) hidden[i] += ffn_out[i];
        free(ffn_out);

        for (int fi = 0; fi < 4; fi++) {
            if (L == m->feature_layers[fi]) {
                features_out[fi] = (float *)malloc((size_t)nt * dim * sizeof(float));
                memcpy(features_out[fi], hidden, (size_t)nt * dim * sizeof(float));
            }
        }
    }

    free(ln_buf); free(qkv); free(attn_out); free(ffn_buf); free(ffn_mid);
    free(pos_y); free(pos_x);
    *hidden_out = hidden;
}

da3_full_result da3_predict_full(da3_model *m, const uint8_t *rgb, int img_w, int img_h,
                                  int n_threads, int output_flags) {
    da3_full_result result = {0};
    int gh = m->grid_h, gw = m->grid_w;
    int np = m->n_patches;
    int dim = m->dim;
    int target_h = gh * m->patch_size;
    int target_w = gw * m->patch_size;
    int feat = m->head_features;

    if (n_threads < 1) n_threads = 1;
    double t_start = da3_time_ms();

    /* ─── Backbone ─── */
    float *img_norm = (float *)malloc((size_t)3 * target_h * target_w * sizeof(float));
    float *hidden = NULL;
    float *features[4];
    double t_backbone = da3_time_ms();
    da3_run_backbone(m, rgb, img_w, img_h, img_norm, n_threads, &hidden, features);
    double t_dpt = da3_time_ms();
    fprintf(stderr, "da3_full: backbone %.1f ms (%d threads)\n", t_dpt - t_backbone, n_threads);

    /* ─── CameraDec (pose) ─── */
    if (m->has_cam_dec && (output_flags & DA3_OUTPUT_POSE)) {
        double t0 = da3_time_ms();
        da3_run_camera_dec(m, hidden, result.pose);
        result.has_pose = 1;
        fprintf(stderr, "da3_full: CameraDec %.1f ms\n", da3_time_ms() - t0);
    }
    free(hidden);

    /* ─── Main DPT head (depth + confidence) ─── */
    {
        float *adapted[4] = {NULL};
        int ad_h[4], ad_w[4];
        da3_compute_adapted(m, features, np, dim, gh, gw, feat, n_threads,
                              adapted, ad_h, ad_w);
        da3_run_dpt_output(m, adapted, ad_h, ad_w, feat, img_w, img_h,
                             &result.depth, &result.confidence);
        result.width = img_w;
        result.height = img_h;
    }
    double t_dpt_end = da3_time_ms();
    fprintf(stderr, "da3_full: DPT head %.1f ms\n", t_dpt_end - t_dpt);

    /* ─── Aux DPT (rays) ─── */
    if (m->has_aux && (output_flags & DA3_OUTPUT_RAYS)) {
        float *aux_adapted[4] = {NULL};
        int aux_ad_h[4], aux_ad_w[4];
        da3_compute_adapted(m, features, np, dim, gh, gw, feat, n_threads,
                              aux_adapted, aux_ad_h, aux_ad_w);
        double t_aux0 = da3_time_ms();
        float *rays_raw = NULL, *ray_conf_raw = NULL;
        int aux_oh, aux_ow;
        da3_run_aux_dpt(m, aux_adapted, aux_ad_h, aux_ad_w, feat,
                          &rays_raw, &ray_conf_raw, &aux_oh, &aux_ow);
        for (int fi = 0; fi < 4; fi++) free(aux_adapted[fi]);

        /* Upsample to original resolution */
        int npix = img_w * img_h;
        result.rays = (float *)malloc((size_t)6 * npix * sizeof(float));
        da3_bilinear(result.rays, rays_raw, 6, aux_oh, aux_ow, img_h, img_w);
        result.ray_confidence = (float *)malloc((size_t)npix * sizeof(float));
        da3_bilinear(result.ray_confidence, ray_conf_raw, 1, aux_oh, aux_ow, img_h, img_w);
        free(rays_raw); free(ray_conf_raw);
        result.has_rays = 1;
        fprintf(stderr, "da3_full: Aux DPT %.1f ms\n", da3_time_ms() - t_aux0);
    }

    /* ─── GSDPT (gaussians) ─── */
    if (m->has_gsdpt && (output_flags & DA3_OUTPUT_GAUSSIANS)) {
        double t_gs0 = da3_time_ms();
        float *gs_raw = NULL;
        int gs_h, gs_w;
        da3_run_gsdpt(m, img_norm, target_h, target_w, features, np, dim, n_threads,
                        &gs_raw, &gs_h, &gs_w);
        int npix = img_w * img_h;
        int gs_oc = m->gsdpt.gs_out_channels;
        if (gs_oc < 2) gs_oc = 38;
        result.gaussians = (float *)malloc((size_t)gs_oc * npix * sizeof(float));
        da3_bilinear(result.gaussians, gs_raw, gs_oc, gs_h, gs_w, img_h, img_w);
        free(gs_raw);
        result.has_gaussians = 1;
        fprintf(stderr, "da3_full: GSDPT (%d channels) %.1f ms\n", gs_oc, da3_time_ms() - t_gs0);
    }

    /* Cleanup */
    for (int fi = 0; fi < 4; fi++) free(features[fi]);
    free(img_norm);

    double t_end = da3_time_ms();
    fprintf(stderr, "da3_full: total %.1f ms\n", t_end - t_start);
    return result;
}

void da3_full_result_free(da3_full_result *r) {
    free(r->depth); free(r->confidence);
    free(r->rays); free(r->ray_confidence);
    free(r->gaussians);
    memset(r, 0, sizeof(*r));
}

#endif /* DEPTH_ANYTHING3_IMPLEMENTATION */
#endif /* DEPTH_ANYTHING3_H */
