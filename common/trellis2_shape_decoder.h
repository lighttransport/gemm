/*
 * trellis2_shape_decoder.h - TRELLIS.2 Stage 2 Shape Decoder (SC-VAE)
 *
 * Usage:
 *   #define T2_SHAPE_DEC_IMPLEMENTATION
 *   #include "trellis2_shape_decoder.h"
 *
 * Dependencies: sparse3d.h, safetensors.h, ggml_dequant.h
 *
 * Decodes shape structured latent [N, 32] to per-voxel predictions [N', 7]
 * using sparse ConvNeXt blocks with channel-to-spatial upsampling.
 *
 * Architecture:
 *   from_latent: Linear(32 → 1024)
 *   blocks.0: 4× ConvNeXtBlock(1024) + C2S(1024→512)
 *   blocks.1: 16× ConvNeXtBlock(512) + C2S(512→256)
 *   blocks.2: 8× ConvNeXtBlock(256) + C2S(256→128)
 *   blocks.3: 4× ConvNeXtBlock(128) + C2S(128→64)
 *   output_layer: Linear(64 → 7)
 *
 * API:
 *   t2_shape_dec *t2_shape_dec_load(const char *st_path);
 *   void          t2_shape_dec_free(t2_shape_dec *d);
 *   t2_shape_dec_result t2_shape_dec_forward(t2_shape_dec *d,
 *       const sp3d_tensor *slat, int n_threads);
 */
#ifndef T2_SHAPE_DEC_H
#define T2_SHAPE_DEC_H

#include <stdint.h>
#include "sparse3d.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *feats;       /* [N, 7]: vertex_xyz(3), intersected(3), quad_lerp(1) */
    int32_t *coords;    /* [N, 4]: batch, z, y, x */
    int N;
} t2_shape_dec_result;

typedef struct t2_shape_dec t2_shape_dec;

t2_shape_dec *t2_shape_dec_load(const char *st_path);
void          t2_shape_dec_free(t2_shape_dec *d);
t2_shape_dec_result t2_shape_dec_forward(t2_shape_dec *d,
    const sp3d_tensor *slat, int n_threads);
void t2_shape_dec_result_free(t2_shape_dec_result *r);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef T2_SHAPE_DEC_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- ConvNeXtBlock weights ---- */
typedef struct {
    float *conv_w, *conv_b;   /* [C, 27, C], [C] */
    float *norm_w, *norm_b;   /* [C] */
    float *mlp0_w, *mlp0_b;  /* [4C, C] */
    float *mlp2_w, *mlp2_b;  /* [C, 4C] */
    int C;
} t2sd_convnext;

/* ---- Channel-to-Spatial block weights ---- */
typedef struct {
    float *norm1_w, *norm1_b;       /* [C_in] */
    float *conv1_w, *conv1_b;       /* [C_out*8, 27, C_in] */
    float *conv2_w, *conv2_b;       /* [C_out, 27, C_out] */
    float *to_subdiv_w, *to_subdiv_b; /* [8, C_in] */
    int C_in, C_out;
} t2sd_c2s;

/* ---- Decoder model ---- */
#define T2SD_MAX_BLOCKS 20
#define T2SD_MAX_STAGES 5

struct t2_shape_dec {
    float *from_latent_w, *from_latent_b;  /* [1024, 32] */
    float *output_w, *output_b;             /* [7, 64] */

    /* Stages: each has N convnext blocks + 1 C2S block */
    int n_stages;
    int n_convnext[T2SD_MAX_STAGES];
    int channels[T2SD_MAX_STAGES];  /* [1024, 512, 256, 128, 64] */
    t2sd_convnext convnext[T2SD_MAX_STAGES][T2SD_MAX_BLOCKS];
    t2sd_c2s c2s[T2SD_MAX_STAGES];

    void *st_ctx;
};

static double t2sd_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- Simple F32 LayerNorm ---- */
static void t2sd_layernorm(float *dst, const float *src, const float *w, const float *b,
                             int N, int C, float eps) {
    for (int i = 0; i < N; i++) {
        const float *xi = src + (size_t)i * C;
        float *yi = dst + (size_t)i * C;
        float mean = 0;
        for (int j = 0; j < C; j++) mean += xi[j];
        mean /= C;
        float var = 0;
        for (int j = 0; j < C; j++) { float d = xi[j] - mean; var += d * d; }
        var /= C;
        float inv = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < C; j++)
            yi[j] = (xi[j] - mean) * inv * (w ? w[j] : 1.0f) + (b ? b[j] : 0.0f);
    }
}

/* ---- Simple F32 GELU ---- */
static void t2sd_gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

/* ---- Simple F32 linear: dst[N, out_C] = src[N, in_C] @ W^T[in_C, out_C] + bias ---- */
static void t2sd_linear(float *dst, const float *src, int N,
                          const float *W, const float *bias,
                          int out_C, int in_C) {
    for (int i = 0; i < N; i++) {
        for (int o = 0; o < out_C; o++) {
            float sum = bias ? bias[o] : 0.0f;
            const float *wr = W + (size_t)o * in_C;
            const float *xi = src + (size_t)i * in_C;
            for (int j = 0; j < in_C; j++) sum += wr[j] * xi[j];
            dst[(size_t)i * out_C + o] = sum;
        }
    }
}

/* ---- Simple sparse conv3d: uses sp3d hash for neighbor lookup ---- */
static void t2sd_sparse_conv(float *dst, const sp3d_tensor *t,
                               const float *weight, const float *bias,
                               int in_C, int out_C, int n_threads) {
    /* Weight layout: [out_C, 3, 3, 3, in_C] = [out_C, 27, in_C] */
    int N = t->N;
    sp3d_ensure_hash((sp3d_tensor *)t);

    /* Initialize with bias */
    for (int i = 0; i < N; i++)
        for (int o = 0; o < out_C; o++)
            dst[(size_t)i * out_C + o] = bias ? bias[o] : 0.0f;

    /* Accumulate 3x3x3 neighbor contributions */
    for (int i = 0; i < N; i++) {
        int32_t bz = t->coords[i * 4];
        int32_t z = t->coords[i * 4 + 1];
        int32_t y = t->coords[i * 4 + 2];
        int32_t x = t->coords[i * 4 + 3];
        for (int kd = 0; kd < 3; kd++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int nz = z + kd - 1, ny = y + kh - 1, nx = x + kw - 1;
                    int ni = sp3d_hash_lookup(t->hash, bz, nz, ny, nx);
                    if (ni < 0) continue;
                    int k_idx = kd * 9 + kh * 3 + kw;
                    const float *feat_n = t->feats + (size_t)ni * in_C;
                    for (int o = 0; o < out_C; o++) {
                        const float *kern = weight + ((size_t)o * 27 + k_idx) * in_C;
                        float sum = 0;
                        for (int j = 0; j < in_C; j++) sum += kern[j] * feat_n[j];
                        dst[(size_t)i * out_C + o] += sum;
                    }
                }
            }
        }
    }
}

/* ---- Forward helpers ---- */

static void t2sd_convnext_forward(float *feats, int N, const t2sd_convnext *blk,
                                    sp3d_tensor *t, int n_threads) {
    int C = blk->C;
    float *tmp = (float *)malloc((size_t)N * C * sizeof(float));
    float *mlp_buf = (float *)malloc((size_t)N * 4 * C * sizeof(float));

    /* conv(feats) -> tmp */
    t2sd_sparse_conv(tmp, t, blk->conv_w, blk->conv_b, C, C, n_threads);

    /* layernorm(tmp) */
    t2sd_layernorm(tmp, tmp, blk->norm_w, blk->norm_b, N, C, 1e-6f);

    /* mlp: Linear(C, 4C) -> GELU -> Linear(4C, C) */
    t2sd_linear(mlp_buf, tmp, N, blk->mlp0_w, blk->mlp0_b, 4 * C, C);
    t2sd_gelu(mlp_buf, N * 4 * C);
    t2sd_linear(tmp, mlp_buf, N, blk->mlp2_w, blk->mlp2_b, C, 4 * C);

    /* residual: feats += tmp */
    for (int i = 0; i < N * C; i++) feats[i] += tmp[i];

    free(tmp);
    free(mlp_buf);
}

static sp3d_tensor *t2sd_c2s_forward(sp3d_tensor *t, const t2sd_c2s *blk, int n_threads) {
    int N = t->N;
    int C_in = blk->C_in, C_out = blk->C_out;

    /* 1. Predict subdivision: which sub-voxels to activate */
    /* to_subdiv: [8, C_in] @ feats[N, C_in] -> [N, 8] */
    float *sub_logits = (float *)malloc((size_t)N * 8 * sizeof(float));
    t2sd_linear(sub_logits, t->feats, N, blk->to_subdiv_w, blk->to_subdiv_b, 8, C_in);

    /* Threshold: activate sub-voxels where logit > 0 */
    int total_sub = 0;
    for (int i = 0; i < N * 8; i++)
        if (sub_logits[i] > 0) total_sub++;

    fprintf(stderr, "    C2S %d->%d: %d voxels -> %d sub-voxels (%.1f avg)\n",
            C_in, C_out, N, total_sub, (float)total_sub / N);

    /* 2. LayerNorm input */
    float *normed = (float *)malloc((size_t)N * C_in * sizeof(float));
    t2sd_layernorm(normed, t->feats, blk->norm1_w, blk->norm1_b, N, C_in, 1e-6f);

    /* 3. conv1: [C_out*8, 27, C_in] -> [N, C_out*8] */
    float *expanded = (float *)malloc((size_t)N * C_out * 8 * sizeof(float));
    /* Need a temp tensor with normed features for conv */
    sp3d_tensor *t_normed = sp3d_replace_feats(t, normed, C_in);
    t2sd_sparse_conv(expanded, t_normed, blk->conv1_w, blk->conv1_b,
                         C_in, C_out * 8, n_threads);

    /* 4. Create sub-voxel coordinates and features */
    int32_t *sub_coords = (int32_t *)malloc((size_t)total_sub * 4 * sizeof(int32_t));
    float *sub_feats = (float *)malloc((size_t)total_sub * C_out * sizeof(float));
    int si = 0;
    for (int i = 0; i < N; i++) {
        int32_t bz = t->coords[i * 4 + 0];
        int32_t z  = t->coords[i * 4 + 1];
        int32_t y  = t->coords[i * 4 + 2];
        int32_t x  = t->coords[i * 4 + 3];
        for (int s = 0; s < 8; s++) {
            if (sub_logits[i * 8 + s] <= 0) continue;
            /* Sub-voxel offset: s = (dz*4 + dy*2 + dx) */
            int dz = (s >> 2) & 1, dy = (s >> 1) & 1, dx = s & 1;
            sub_coords[si * 4 + 0] = bz;
            sub_coords[si * 4 + 1] = z * 2 + dz;
            sub_coords[si * 4 + 2] = y * 2 + dy;
            sub_coords[si * 4 + 3] = x * 2 + dx;
            /* Feature: extract sub-channel from expanded[i, s*C_out : (s+1)*C_out] */
            memcpy(sub_feats + (size_t)si * C_out,
                   expanded + (size_t)i * C_out * 8 + (size_t)s * C_out,
                   (size_t)C_out * sizeof(float));
            si++;
        }
    }

    free(sub_logits); free(normed); free(expanded);
    sp3d_free(t_normed);

    /* 5. Create new sparse tensor at higher resolution */
    sp3d_tensor *t_sub = sp3d_create(sub_coords, sub_feats, total_sub, C_out, 1);
    free(sub_coords); free(sub_feats);

    /* 6. conv2: [C_out, 27, C_out] at new resolution */
    float *conv2_out = (float *)malloc((size_t)total_sub * C_out * sizeof(float));
    t2sd_sparse_conv(conv2_out, t_sub, blk->conv2_w, blk->conv2_b,
                         C_out, C_out, n_threads);
    /* Replace features */
    memcpy(t_sub->feats, conv2_out, (size_t)total_sub * C_out * sizeof(float));
    free(conv2_out);

    return t_sub;
}

/* ---- Full forward ---- */

t2_shape_dec_result t2_shape_dec_forward(t2_shape_dec *d,
    const sp3d_tensor *slat, int n_threads) {
    t2_shape_dec_result result = {0};
    double t0 = t2sd_time_ms();
    int N = slat->N;
    int C = d->channels[0]; /* 1024 */

    fprintf(stderr, "shape_dec: input N=%d, C=%d\n", N, slat->C);

    /* from_latent: Linear(32 -> 1024) */
    float *feats = (float *)malloc((size_t)N * C * sizeof(float));
    t2sd_linear(feats, slat->feats, N, d->from_latent_w, d->from_latent_b, C, slat->C);
    fprintf(stderr, "shape_dec: from_latent -> [%d, %d]\n", N, C);

    /* Create working sparse tensor */
    sp3d_tensor *t = sp3d_create(slat->coords, feats, N, C, 1);
    free(feats);

    /* Process stages */
    for (int stage = 0; stage < d->n_stages; stage++) {
        int nc = d->n_convnext[stage];
        int ch = d->channels[stage];

        fprintf(stderr, "shape_dec: stage %d: %d ConvNeXt(%d), N=%d\n",
                stage, nc, ch, t->N);
        double ts0 = t2sd_time_ms();

        /* ConvNeXt blocks */
        for (int b = 0; b < nc; b++) {
            t2sd_convnext_forward(t->feats, t->N, &d->convnext[stage][b], t, n_threads);
        }

        double ts1 = t2sd_time_ms();
        fprintf(stderr, "  convnext: %.1f s\n", (ts1 - ts0) / 1000.0);

        /* C2S upsample (if not last stage, or if c2s weights exist) */
        if (d->c2s[stage].conv1_w) {
            sp3d_tensor *t_new = t2sd_c2s_forward(t, &d->c2s[stage], n_threads);
            sp3d_free(t);
            t = t_new;
            fprintf(stderr, "  c2s: -> N=%d, C=%d (%.1f s)\n",
                    t->N, t->C, (t2sd_time_ms() - ts1) / 1000.0);
        }
    }

    /* output_layer: Linear(64 -> 7) */
    float *out_feats = (float *)malloc((size_t)t->N * 7 * sizeof(float));
    t2sd_linear(out_feats, t->feats, t->N, d->output_w, d->output_b, 7, t->C);

    result.feats = out_feats;
    result.coords = (int32_t *)malloc((size_t)t->N * 4 * sizeof(int32_t));
    memcpy(result.coords, t->coords, (size_t)t->N * 4 * sizeof(int32_t));
    result.N = t->N;

    sp3d_free(t);

    double t1 = t2sd_time_ms();
    fprintf(stderr, "shape_dec: done in %.1f s, output N=%d\n",
            (t1 - t0) / 1000.0, result.N);
    return result;
}

void t2_shape_dec_result_free(t2_shape_dec_result *r) {
    free(r->feats); free(r->coords);
    r->feats = NULL; r->coords = NULL; r->N = 0;
}

/* ---- Weight loading ---- */

#ifdef SAFETENSORS_H

static float *t2sd_load_f32(st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return NULL;
    const char *dt = safetensors_dtype(st, idx);
    size_t nb = safetensors_nbytes(st, idx);
    void *data = safetensors_data(st, idx);

    size_t n_elem = (strcmp(dt, "F32") == 0) ? nb / 4 :
                    (strcmp(dt, "F16") == 0 || strcmp(dt, "BF16") == 0) ? nb / 2 : 0;
    float *buf = (float *)malloc(n_elem * sizeof(float));

    if (strcmp(dt, "F32") == 0) {
        memcpy(buf, data, nb);
    } else if (strcmp(dt, "F16") == 0) {
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < n_elem; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else if (strcmp(dt, "BF16") == 0) {
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < n_elem; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            memcpy(&buf[i], &bits, 4);
        }
    }
    return buf;
}

t2_shape_dec *t2_shape_dec_load(const char *st_path) {
    st_context *st = safetensors_open(st_path);
    if (!st) return NULL;
    fprintf(stderr, "shape_dec: loading from %s (%d tensors)\n", st_path, st->n_tensors);

    t2_shape_dec *d = (t2_shape_dec *)calloc(1, sizeof(t2_shape_dec));
    d->st_ctx = st;

    d->from_latent_w = t2sd_load_f32(st, "from_latent.weight");
    d->from_latent_b = t2sd_load_f32(st, "from_latent.bias");
    d->output_w = t2sd_load_f32(st, "output_layer.weight");
    d->output_b = t2sd_load_f32(st, "output_layer.bias");

    /* Detect stages and blocks from weight names */
    int channels[] = {1024, 512, 256, 128, 64};
    int n_convnext[] = {4, 16, 8, 4, 0};
    d->n_stages = 4;  /* stages with C2S blocks */
    memcpy(d->channels, channels, sizeof(channels));
    memcpy(d->n_convnext, n_convnext, sizeof(n_convnext));

    for (int s = 0; s < d->n_stages; s++) {
        int C = channels[s];
        char name[256];

        /* ConvNeXt blocks */
        for (int b = 0; b < n_convnext[s]; b++) {
            t2sd_convnext *blk = &d->convnext[s][b];
            blk->C = C;
            #define LOAD(field, suffix) do { \
                snprintf(name, sizeof(name), "blocks.%d.%d.%s", s, b, suffix); \
                blk->field = t2sd_load_f32(st, name); \
            } while(0)
            LOAD(conv_w, "conv.weight");
            LOAD(conv_b, "conv.bias");
            LOAD(norm_w, "norm.weight");
            LOAD(norm_b, "norm.bias");
            LOAD(mlp0_w, "mlp.0.weight");
            LOAD(mlp0_b, "mlp.0.bias");
            LOAD(mlp2_w, "mlp.2.weight");
            LOAD(mlp2_b, "mlp.2.bias");
            #undef LOAD
        }

        /* C2S block (last block in the stage) */
        int c2s_idx = n_convnext[s];
        t2sd_c2s *c = &d->c2s[s];
        c->C_in = C;
        c->C_out = (s + 1 < 5) ? channels[s + 1] : C;
        #define LOAD_C2S(field, suffix) do { \
            snprintf(name, sizeof(name), "blocks.%d.%d.%s", s, c2s_idx, suffix); \
            c->field = t2sd_load_f32(st, name); \
        } while(0)
        LOAD_C2S(norm1_w, "norm1.weight");
        LOAD_C2S(norm1_b, "norm1.bias");
        LOAD_C2S(conv1_w, "conv1.weight");
        LOAD_C2S(conv1_b, "conv1.bias");
        LOAD_C2S(conv2_w, "conv2.weight");
        LOAD_C2S(conv2_b, "conv2.bias");
        LOAD_C2S(to_subdiv_w, "to_subdiv.weight");
        LOAD_C2S(to_subdiv_b, "to_subdiv.bias");
        #undef LOAD_C2S

        fprintf(stderr, "  stage %d: %d ConvNeXt(%d) + C2S(%d->%d) %s\n",
                s, n_convnext[s], C, C, c->C_out,
                c->conv1_w ? "loaded" : "MISSING");
    }

    fprintf(stderr, "shape_dec: loaded\n");
    return d;
}

#endif /* SAFETENSORS_H */

void t2_shape_dec_free(t2_shape_dec *d) {
    if (!d) return;
    /* Free all weight buffers */
    free(d->from_latent_w); free(d->from_latent_b);
    free(d->output_w); free(d->output_b);
    for (int s = 0; s < d->n_stages; s++) {
        for (int b = 0; b < d->n_convnext[s]; b++) {
            t2sd_convnext *blk = &d->convnext[s][b];
            free(blk->conv_w); free(blk->conv_b);
            free(blk->norm_w); free(blk->norm_b);
            free(blk->mlp0_w); free(blk->mlp0_b);
            free(blk->mlp2_w); free(blk->mlp2_b);
        }
        t2sd_c2s *c = &d->c2s[s];
        free(c->norm1_w); free(c->norm1_b);
        free(c->conv1_w); free(c->conv1_b);
        free(c->conv2_w); free(c->conv2_b);
        free(c->to_subdiv_w); free(c->to_subdiv_b);
    }
#ifdef SAFETENSORS_H
    if (d->st_ctx) safetensors_close((st_context *)d->st_ctx);
#endif
    free(d);
}

#endif /* T2_SHAPE_DEC_IMPLEMENTATION */
#endif /* T2_SHAPE_DEC_H */
