/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * qtts_codec.h - Qwen3-TTS-Tokenizer-12Hz decoder (codes -> 24 kHz waveform), CPU.
 *
 * Follows the Apache-2.0 reference `qwen_tts/core/tokenizer_12hz/
 * modeling_qwen3_tts_tokenizer_v2.py` (Qwen3TTSTokenizerV2Decoder):
 *   split RVQ dequant (1 semantic + 15 acoustic codebooks, 256-d, 1x1 out proj to 512)
 *   -> causal conv k3 (512->1024)
 *   -> 8-layer transformer (in 1024->512, RoPE, sliding window 72, LayerScale, SwiGLU, out 512->1024)
 *   -> 2x [causal transposed conv x2 + ConvNeXt]
 *   -> causal conv k7 (1024->1536) -> 4 decoder blocks [SnakeBeta, convT x{8,5,4,3}, 3 residual units]
 *   -> SnakeBeta -> causal conv k7 (96->1) -> clamp [-1, 1]
 * Chunked decoding (300 frames, 25 frames left context) matches chunked_decode().
 *
 * Requires safetensors.h and qtts_ops.h. Define QTTS_CODEC_IMPLEMENTATION in one TU.
 */
#ifndef QTTS_CODEC_H
#define QTTS_CODEC_H

#include <stdint.h>
#include "qtts_ops.h"

typedef struct qtts_codec qtts_codec;

/* dir: the speech_tokenizer directory (config.json + model.safetensors). */
qtts_codec *qtts_codec_load(const char *dir);
void        qtts_codec_free(qtts_codec *c);
int         qtts_codec_num_quantizers(const qtts_codec *c);
int         qtts_codec_upsample(const qtts_codec *c);     /* samples per frame (1920) */
int         qtts_codec_sample_rate(const qtts_codec *c);  /* 24000 */
/* RVQ dequantization: codes [T][nq] -> x [T][codebook_dim] */
void        qtts_codec_rvq(const qtts_codec *c, const int32_t *codes, int T, float *x);

/* codes: [T][nq] int32. Returns malloc'd wav of T*upsample samples in *n_out.
 * If dump_dir is non-NULL, intermediate stages of the first chunk are saved as .npy
 * using the reference names (codec_rvq, codec_preconv, codec_pretf, codec_up*, codec_dec*). */
float      *qtts_codec_decode(qtts_codec *c, const int32_t *codes, int T, int *n_out,
                              const char *dump_dir);

#endif /* QTTS_CODEC_H */

/* ======================================================================== */
#if defined(QTTS_CODEC_IMPLEMENTATION) && !defined(QTTS_CODEC_IMPL_DONE)
#define QTTS_CODEC_IMPL_DONE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "safetensors.h"

typedef struct {           /* conv1d (causal) or transposed conv1d */
    int cin, cout, k, dil, stride, groups, transposed;
    qt_packed *taps;       /* dense: k packed [cout x cin] (conv) or 1 packed [k*cout x cin] (convT) */
    float *dw;             /* depthwise weights [c][k] */
    float *bias;           /* [cout] */
} qc_conv;

typedef struct { float *alpha, *beta; int c; } qc_snake; /* stored as exp(alpha), 1/(exp(beta)+1e-9) */

typedef struct {
    float *ln1, *ln2, *ls_attn, *ls_mlp;
    qt_packed q, k, v, o, gate, up, down;
} qc_tf_layer;

typedef struct {
    qc_conv convt;
    qc_conv dw;
    float *ln_w, *ln_b, *gamma;
    qt_packed pw1, pw2;
    float *pw1_b, *pw2_b;
} qc_upblock;

typedef struct {
    qc_snake act1, act2;
    qc_conv conv1, conv2;
} qc_resunit;

typedef struct {
    qc_snake snake;
    qc_conv convt;
    qc_resunit res[3];
} qc_decblock;

struct qtts_codec {
    int nq, cb_size, cb_dim, latent, dec_dim, hidden, inter, n_layers, n_heads, n_kv, head_dim;
    int window, n_up_ratio, n_rates;
    int up_ratios[4], rates[8];
    float rms_eps, rope_theta;
    int upsample;
    /* RVQ: dequantized codebooks [nq][cb_size][256], output projections */
    float *codebooks;
    float *out_first, *out_rest;    /* [512][256] */
    qc_conv pre_conv;
    qt_packed in_proj, out_proj;
    float *in_proj_b, *out_proj_b, *norm_w;
    qc_tf_layer *layers;
    qc_upblock up[4];
    qc_conv dec_in;
    qc_decblock blocks[8];
    qc_snake final_snake;
    qc_conv final_conv;
    st_context *st;
};

/* ---- loading helpers ---- */

static const float *qc__f32(qtts_codec *c, const char *name, int expect_n) {
    int i = safetensors_find(c->st, name);
    if (i < 0) { fprintf(stderr, "qtts_codec: missing tensor %s\n", name); exit(1); }
    if (strcmp(safetensors_dtype(c->st, i), "F32")) {
        fprintf(stderr, "qtts_codec: %s is %s, expected F32\n", name, safetensors_dtype(c->st, i)); exit(1);
    }
    if (expect_n >= 0 && (int)(safetensors_nbytes(c->st, i) / 4) != expect_n) {
        fprintf(stderr, "qtts_codec: %s has %zu elems, expected %d\n", name,
                safetensors_nbytes(c->st, i) / 4, expect_n); exit(1);
    }
    return (const float *)safetensors_data(c->st, i);
}

static float *qc__copy(qtts_codec *c, const char *name, int n) {
    float *d = (float *)malloc(sizeof(float) * (size_t)n);
    memcpy(d, qc__f32(c, name, n), sizeof(float) * (size_t)n);
    return d;
}

typedef struct { const float *w; int cin, cout, k, j; } qc__tapctx;
/* conv weight [cout][cin][k] -> tap j as W[n=cout][k=cin] */
static float qc__get_conv_tap(const void *w, int n, int kk, const void *ctx) {
    const qc__tapctx *t = (const qc__tapctx *)ctx; (void)w;
    return t->w[((size_t)n * t->cin + kk) * t->k + t->j];
}
/* convT weight [cin][cout][k] -> W[n = j*cout + co][k = cin] */
static float qc__get_convt(const void *w, int n, int kk, const void *ctx) {
    const qc__tapctx *t = (const qc__tapctx *)ctx; (void)w;
    int j = n / t->cout, co = n % t->cout;
    return t->w[((size_t)kk * t->cout + co) * t->k + j];
}

static void qc__load_conv(qtts_codec *c, qc_conv *cv, const char *prefix, int cin, int cout, int k,
                          int dil, int groups, int transposed, int stride) {
    char nm[600];
    memset(cv, 0, sizeof(*cv));
    cv->cin = cin; cv->cout = cout; cv->k = k; cv->dil = dil; cv->groups = groups;
    cv->transposed = transposed; cv->stride = stride;
    snprintf(nm, sizeof(nm), "%s.weight", prefix);
    if (groups == cin && groups > 1) {
        cv->dw = qc__copy(c, nm, cin * k);
    } else if (transposed) {
        const float *w = qc__f32(c, nm, cin * cout * k);
        qc__tapctx t = { w, cin, cout, k, 0 };
        cv->taps = (qt_packed *)calloc(1, sizeof(qt_packed));
        qt_pack_b(cv->taps, k * cout, cin, qc__get_convt, w, &t);
    } else {
        const float *w = qc__f32(c, nm, cout * cin * k);
        cv->taps = (qt_packed *)calloc((size_t)k, sizeof(qt_packed));
        for (int j = 0; j < k; j++) {
            qc__tapctx t = { w, cin, cout, k, j };
            qt_pack_b(&cv->taps[j], cout, cin, qc__get_conv_tap, w, &t);
        }
    }
    snprintf(nm, sizeof(nm), "%s.bias", prefix);
    if (safetensors_find(c->st, nm) >= 0) cv->bias = qc__copy(c, nm, cout);
}

static void qc__load_snake(qtts_codec *c, qc_snake *s, const char *prefix, int ch) {
    char nm[600];
    s->c = ch;
    snprintf(nm, sizeof(nm), "%s.alpha", prefix);
    s->alpha = qc__copy(c, nm, ch);
    snprintf(nm, sizeof(nm), "%s.beta", prefix);
    s->beta = qc__copy(c, nm, ch);
    for (int i = 0; i < ch; i++) {
        s->alpha[i] = expf(s->alpha[i]);
        s->beta[i] = 1.0f / (expf(s->beta[i]) + 1e-9f);
    }
}

static void qc__pack_linear(qtts_codec *c, qt_packed *p, const char *name, int n, int k) {
    qt_pack_b_f32_nk(p, qc__f32(c, name, n * k), n, k);
}

static int qc__json_int(const json_val *o, const char *k, int def) {
    const json_val *v = json_obj_get(o, k);
    return v && v->type == JSON_NUMBER ? (int)v->num : def;
}
static double qc__json_num(const json_val *o, const char *k, double def) {
    const json_val *v = json_obj_get(o, k);
    return v && v->type == JSON_NUMBER ? v->num : def;
}
static int qc__json_int_array(const json_val *o, const char *k, int *out, int cap) {
    const json_val *v = json_obj_get(o, k);
    if (!v || v->type != JSON_ARRAY) return 0;
    int n = v->arr.count < cap ? v->arr.count : cap;
    for (int i = 0; i < n; i++) out[i] = (int)v->arr.items[i].num;
    return n;
}

static char *qc__read_file(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    b[n] = 0;
    fclose(f);
    if (len) *len = n;
    return b;
}

qtts_codec *qtts_codec_load(const char *dir) {
    char path[1024];
    long len = 0;
    snprintf(path, sizeof(path), "%s/config.json", dir);
    char *js = qc__read_file(path, &len);
    if (!js) { fprintf(stderr, "qtts_codec: cannot read %s\n", path); return NULL; }
    json_val *root = json_parse(js, (int)len);
    free(js);
    const json_val *dc = json_obj_get(root, "decoder_config");
    if (!dc) dc = root;

    qtts_codec *c = (qtts_codec *)calloc(1, sizeof(*c));
    c->nq = qc__json_int(dc, "num_quantizers", 16);
    c->cb_size = qc__json_int(dc, "codebook_size", 2048);
    c->cb_dim = qc__json_int(dc, "codebook_dim", 512);
    c->latent = qc__json_int(dc, "latent_dim", 1024);
    c->dec_dim = qc__json_int(dc, "decoder_dim", 1536);
    c->hidden = qc__json_int(dc, "hidden_size", 512);
    c->inter = qc__json_int(dc, "intermediate_size", 1024);
    c->n_layers = qc__json_int(dc, "num_hidden_layers", 8);
    c->n_heads = qc__json_int(dc, "num_attention_heads", 16);
    c->n_kv = qc__json_int(dc, "num_key_value_heads", 16);
    c->head_dim = qc__json_int(dc, "head_dim", 64);
    c->window = qc__json_int(dc, "sliding_window", 72);
    c->rms_eps = (float)qc__json_num(dc, "rms_norm_eps", 1e-5);
    c->rope_theta = (float)qc__json_num(dc, "rope_theta", 10000.0);
    c->n_up_ratio = qc__json_int_array(dc, "upsampling_ratios", c->up_ratios, 4);
    c->n_rates = qc__json_int_array(dc, "upsample_rates", c->rates, 8);
    json_free(root);
    c->upsample = 1;
    for (int i = 0; i < c->n_up_ratio; i++) c->upsample *= c->up_ratios[i];
    for (int i = 0; i < c->n_rates; i++) c->upsample *= c->rates[i];

    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    c->st = safetensors_open(path);
    if (!c->st) { fprintf(stderr, "qtts_codec: cannot open %s\n", path); free(c); return NULL; }

    char nm[512];
    int vqd = c->cb_dim / 2;
    /* RVQ codebooks: embedding_sum / clamp(cluster_usage, 1e-5) */
    c->codebooks = (float *)malloc(sizeof(float) * (size_t)c->nq * c->cb_size * vqd);
    for (int q = 0; q < c->nq; q++) {
        const char *grp = q == 0 ? "rvq_first" : "rvq_rest";
        int li = q == 0 ? 0 : q - 1;
        snprintf(nm, sizeof(nm), "decoder.quantizer.%s.vq.layers.%d._codebook.embedding_sum", grp, li);
        const float *es = qc__f32(c, nm, c->cb_size * vqd);
        snprintf(nm, sizeof(nm), "decoder.quantizer.%s.vq.layers.%d._codebook.cluster_usage", grp, li);
        const float *cu = qc__f32(c, nm, c->cb_size);
        float *dst = c->codebooks + (size_t)q * c->cb_size * vqd;
        for (int e = 0; e < c->cb_size; e++) {
            float u = cu[e] < 1e-5f ? 1e-5f : cu[e];
            for (int d = 0; d < vqd; d++) dst[(size_t)e * vqd + d] = es[(size_t)e * vqd + d] / u;
        }
    }
    c->out_first = qc__copy(c, "decoder.quantizer.rvq_first.output_proj.weight", c->cb_dim * vqd);
    c->out_rest = qc__copy(c, "decoder.quantizer.rvq_rest.output_proj.weight", c->cb_dim * vqd);

    qc__load_conv(c, &c->pre_conv, "decoder.pre_conv.conv", c->cb_dim, c->latent, 3, 1, 1, 0, 1);

    qc__pack_linear(c, &c->in_proj, "decoder.pre_transformer.input_proj.weight", c->hidden, c->latent);
    c->in_proj_b = qc__copy(c, "decoder.pre_transformer.input_proj.bias", c->hidden);
    qc__pack_linear(c, &c->out_proj, "decoder.pre_transformer.output_proj.weight", c->latent, c->hidden);
    c->out_proj_b = qc__copy(c, "decoder.pre_transformer.output_proj.bias", c->latent);
    c->norm_w = qc__copy(c, "decoder.pre_transformer.norm.weight", c->hidden);
    int qd = c->n_heads * c->head_dim, kd = c->n_kv * c->head_dim;
    c->layers = (qc_tf_layer *)calloc((size_t)c->n_layers, sizeof(qc_tf_layer));
    for (int l = 0; l < c->n_layers; l++) {
        qc_tf_layer *L = &c->layers[l];
#define QC_P(s) (snprintf(nm, sizeof(nm), "decoder.pre_transformer.layers.%d.%s", l, s), nm)
        L->ln1 = qc__copy(c, QC_P("input_layernorm.weight"), c->hidden);
        L->ln2 = qc__copy(c, QC_P("post_attention_layernorm.weight"), c->hidden);
        L->ls_attn = qc__copy(c, QC_P("self_attn_layer_scale.scale"), c->hidden);
        L->ls_mlp = qc__copy(c, QC_P("mlp_layer_scale.scale"), c->hidden);
        qc__pack_linear(c, &L->q, QC_P("self_attn.q_proj.weight"), qd, c->hidden);
        qc__pack_linear(c, &L->k, QC_P("self_attn.k_proj.weight"), kd, c->hidden);
        qc__pack_linear(c, &L->v, QC_P("self_attn.v_proj.weight"), kd, c->hidden);
        qc__pack_linear(c, &L->o, QC_P("self_attn.o_proj.weight"), c->hidden, qd);
        qc__pack_linear(c, &L->gate, QC_P("mlp.gate_proj.weight"), c->inter, c->hidden);
        qc__pack_linear(c, &L->up, QC_P("mlp.up_proj.weight"), c->inter, c->hidden);
        qc__pack_linear(c, &L->down, QC_P("mlp.down_proj.weight"), c->hidden, c->inter);
#undef QC_P
    }
    for (int u = 0; u < c->n_up_ratio; u++) {
        qc_upblock *U = &c->up[u];
        int f = c->up_ratios[u], d = c->latent;
        snprintf(nm, sizeof(nm), "decoder.upsample.%d.0.conv", u);
        qc__load_conv(c, &U->convt, nm, d, d, f, 1, 1, 1, f);
        snprintf(nm, sizeof(nm), "decoder.upsample.%d.1.dwconv.conv", u);
        qc__load_conv(c, &U->dw, nm, d, d, 7, 1, d, 0, 1);
#define QC_U(s) (snprintf(nm, sizeof(nm), "decoder.upsample.%d.1.%s", u, s), nm)
        U->ln_w = qc__copy(c, QC_U("norm.weight"), d);
        U->ln_b = qc__copy(c, QC_U("norm.bias"), d);
        U->gamma = qc__copy(c, QC_U("gamma"), d);
        qc__pack_linear(c, &U->pw1, QC_U("pwconv1.weight"), 4 * d, d);
        U->pw1_b = qc__copy(c, QC_U("pwconv1.bias"), 4 * d);
        qc__pack_linear(c, &U->pw2, QC_U("pwconv2.weight"), d, 4 * d);
        U->pw2_b = qc__copy(c, QC_U("pwconv2.bias"), d);
#undef QC_U
    }
    qc__load_conv(c, &c->dec_in, "decoder.decoder.0.conv", c->latent, c->dec_dim, 7, 1, 1, 0, 1);
    for (int b = 0; b < c->n_rates; b++) {
        qc_decblock *B = &c->blocks[b];
        int ind = c->dec_dim >> b, outd = c->dec_dim >> (b + 1), r = c->rates[b];
        snprintf(nm, sizeof(nm), "decoder.decoder.%d.block.0", b + 1);
        qc__load_snake(c, &B->snake, nm, ind);
        snprintf(nm, sizeof(nm), "decoder.decoder.%d.block.1.conv", b + 1);
        qc__load_conv(c, &B->convt, nm, ind, outd, 2 * r, 1, 1, 1, r);
        static const int dils[3] = { 1, 3, 9 };
        for (int ru = 0; ru < 3; ru++) {
            qc_resunit *R = &B->res[ru];
            snprintf(nm, sizeof(nm), "decoder.decoder.%d.block.%d.act1", b + 1, ru + 2);
            qc__load_snake(c, &R->act1, nm, outd);
            snprintf(nm, sizeof(nm), "decoder.decoder.%d.block.%d.act2", b + 1, ru + 2);
            qc__load_snake(c, &R->act2, nm, outd);
            snprintf(nm, sizeof(nm), "decoder.decoder.%d.block.%d.conv1.conv", b + 1, ru + 2);
            qc__load_conv(c, &R->conv1, nm, outd, outd, 7, dils[ru], 1, 0, 1);
            snprintf(nm, sizeof(nm), "decoder.decoder.%d.block.%d.conv2.conv", b + 1, ru + 2);
            qc__load_conv(c, &R->conv2, nm, outd, outd, 1, 1, 1, 0, 1);
        }
    }
    int outd = c->dec_dim >> c->n_rates;
    snprintf(nm, sizeof(nm), "decoder.decoder.%d", c->n_rates + 1);
    qc__load_snake(c, &c->final_snake, nm, outd);
    snprintf(nm, sizeof(nm), "decoder.decoder.%d.conv", c->n_rates + 2);
    qc__load_conv(c, &c->final_conv, nm, outd, 1, 7, 1, 1, 0, 1);
    return c;
}

static void qc__free_conv(qc_conv *cv) {
    if (cv->taps) {
        int n = cv->transposed ? 1 : cv->k;
        for (int j = 0; j < n; j++) qt_packed_free(&cv->taps[j]);
        free(cv->taps);
    }
    free(cv->dw); free(cv->bias);
}
static void qc__free_snake(qc_snake *s) { free(s->alpha); free(s->beta); }

void qtts_codec_free(qtts_codec *c) {
    if (!c) return;
    free(c->codebooks); free(c->out_first); free(c->out_rest);
    qc__free_conv(&c->pre_conv);
    qt_packed_free(&c->in_proj); qt_packed_free(&c->out_proj);
    free(c->in_proj_b); free(c->out_proj_b); free(c->norm_w);
    for (int l = 0; l < c->n_layers; l++) {
        qc_tf_layer *L = &c->layers[l];
        free(L->ln1); free(L->ln2); free(L->ls_attn); free(L->ls_mlp);
        qt_packed_free(&L->q); qt_packed_free(&L->k); qt_packed_free(&L->v); qt_packed_free(&L->o);
        qt_packed_free(&L->gate); qt_packed_free(&L->up); qt_packed_free(&L->down);
    }
    free(c->layers);
    for (int u = 0; u < c->n_up_ratio; u++) {
        qc_upblock *U = &c->up[u];
        qc__free_conv(&U->convt); qc__free_conv(&U->dw);
        free(U->ln_w); free(U->ln_b); free(U->gamma); free(U->pw1_b); free(U->pw2_b);
        qt_packed_free(&U->pw1); qt_packed_free(&U->pw2);
    }
    qc__free_conv(&c->dec_in);
    for (int b = 0; b < c->n_rates; b++) {
        qc_decblock *B = &c->blocks[b];
        qc__free_snake(&B->snake); qc__free_conv(&B->convt);
        for (int r = 0; r < 3; r++) {
            qc__free_snake(&B->res[r].act1); qc__free_snake(&B->res[r].act2);
            qc__free_conv(&B->res[r].conv1); qc__free_conv(&B->res[r].conv2);
        }
    }
    qc__free_snake(&c->final_snake);
    qc__free_conv(&c->final_conv);
    safetensors_close(c->st);
    free(c);
}

int qtts_codec_num_quantizers(const qtts_codec *c) { return c->nq; }
int qtts_codec_upsample(const qtts_codec *c) { return c->upsample; }
int qtts_codec_sample_rate(const qtts_codec *c) { (void)c; return 24000; }

/* ---- forward pieces (activations [T][C]) ---- */

/* Causal conv (stride 1): y[t] = b + sum_j W_j x[t + j*dil - (k-1)*dil]. Returns new buffer. */
static float *qc__conv(const qc_conv *cv, const float *x, int T) {
    int cin = cv->cin, cout = cv->cout, k = cv->k, d = cv->dil;
    int pad = (k - 1) * d;
    float *y = (float *)malloc(sizeof(float) * (size_t)T * cout);
    if (cv->dw) {
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++)
            for (int ch = 0; ch < cin; ch++) {
                float s = cv->bias ? cv->bias[ch] : 0.0f;
                for (int j = 0; j < k; j++) {
                    int ti = t + j * d - pad;
                    if (ti >= 0) s += cv->dw[(size_t)ch * k + j] * x[(size_t)ti * cin + ch];
                }
                y[(size_t)t * cout + ch] = s;
            }
        return y;
    }
    /* zero-padded copy so every tap is a plain GEMM on a shifted row window */
    float *xp = (float *)calloc((size_t)(T + pad), (size_t)cin * sizeof(float));
    memcpy(xp + (size_t)pad * cin, x, sizeof(float) * (size_t)T * cin);
    for (int j = 0; j < k; j++)
        qt_sgemm(T, xp + (size_t)j * d * cin, cin, &cv->taps[j], y, cout, j > 0);
    free(xp);
    if (cv->bias) {
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++)
            for (int ch = 0; ch < cout; ch++) y[(size_t)t * cout + ch] += cv->bias[ch];
    }
    return y;
}

/* Causal transposed conv: full convT then crop (k - stride) samples on the right. Output T*stride. */
static float *qc__convt(const qc_conv *cv, const float *x, int T) {
    int cout = cv->cout, k = cv->k, s = cv->stride;
    int To = T * s;
    float *z = (float *)malloc(sizeof(float) * (size_t)T * k * cout);
    qt_sgemm(T, x, cv->cin, cv->taps, z, k * cout, 0);
    float *y = (float *)malloc(sizeof(float) * (size_t)To * cout);
    #pragma omp parallel for schedule(static)
    for (int to = 0; to < To; to++) {
        float *yr = y + (size_t)to * cout;
        for (int ch = 0; ch < cout; ch++) yr[ch] = cv->bias ? cv->bias[ch] : 0.0f;
        /* contributions: input t, tap j with t*s + j == to */
        for (int j = to % s; j < k; j += s) {
            int t = (to - j) / s;
            if (t < 0 || t >= T) continue;
            const float *zr = z + ((size_t)t * k + j) * cout;
            for (int ch = 0; ch < cout; ch++) yr[ch] += zr[ch];
        }
    }
    free(z);
    return y;
}

static void qc__snake(const qc_snake *sn, float *x, int T) {
    int C = sn->c;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        float *r = x + (size_t)t * C;
        for (int ch = 0; ch < C; ch++) {
            float v = sinf(r[ch] * sn->alpha[ch]);
            r[ch] += sn->beta[ch] * v * v;
        }
    }
}

static void qc__add_bias(float *y, const float *b, int T, int C) {
    for (int t = 0; t < T; t++)
        for (int ch = 0; ch < C; ch++) y[(size_t)t * C + ch] += b[ch];
}

static void qc__dump(const char *dir, const char *name, const float *x, int T, int C) {
    if (!dir) return;
    char p[1024];
    snprintf(p, sizeof(p), "%s/%s.npy", dir, name);
    /* reference stages are [C, T] (channels-first) except pre-transformer output [T, C] */
    float *tr = (float *)malloc(sizeof(float) * (size_t)T * C);
    for (int t = 0; t < T; t++)
        for (int ch = 0; ch < C; ch++) tr[(size_t)ch * T + t] = x[(size_t)t * C + ch];
    int dims[2] = { C, T };
    qt_npy_save_f32(p, tr, 2, dims);
    free(tr);
}

static float *qc__transformer(const qtts_codec *c, const float *x, int T) {
    int H = c->hidden, qd = c->n_heads * c->head_dim, kd = c->n_kv * c->head_dim, I = c->inter;
    float *h = (float *)malloc(sizeof(float) * (size_t)T * H);
    qt_sgemm(T, x, c->latent, &c->in_proj, h, H, 0);
    qc__add_bias(h, c->in_proj_b, T, H);
    float *xn = (float *)malloc(sizeof(float) * (size_t)T * H);
    float *q = (float *)malloc(sizeof(float) * (size_t)T * qd);
    float *k = (float *)malloc(sizeof(float) * (size_t)T * kd);
    float *v = (float *)malloc(sizeof(float) * (size_t)T * kd);
    float *att = (float *)malloc(sizeof(float) * (size_t)T * qd);
    float *o = (float *)malloc(sizeof(float) * (size_t)T * H);
    float *g = (float *)malloc(sizeof(float) * (size_t)T * I);
    float *u = (float *)malloc(sizeof(float) * (size_t)T * I);
    for (int l = 0; l < c->n_layers; l++) {
        const qc_tf_layer *L = &c->layers[l];
        for (int t = 0; t < T; t++) qt_rmsnorm(xn + (size_t)t * H, h + (size_t)t * H, L->ln1, H, c->rms_eps);
        qt_sgemm(T, xn, H, &L->q, q, qd, 0);
        qt_sgemm(T, xn, H, &L->k, k, kd, 0);
        qt_sgemm(T, xn, H, &L->v, v, kd, 0);
        for (int t = 0; t < T; t++) {
            for (int hh = 0; hh < c->n_heads; hh++)
                qt_rope_neox(q + (size_t)t * qd + hh * c->head_dim, c->head_dim, t, c->rope_theta);
            for (int hh = 0; hh < c->n_kv; hh++)
                qt_rope_neox(k + (size_t)t * kd + hh * c->head_dim, c->head_dim, t, c->rope_theta);
        }
        qt_attention(att, q, k, v, T, 0, c->n_heads, c->n_kv, c->head_dim, c->window);
        qt_sgemm(T, att, qd, &L->o, o, H, 0);
        for (size_t i = 0; i < (size_t)T * H; i++) h[i] += L->ls_attn[i % H] * o[i];
        for (int t = 0; t < T; t++) qt_rmsnorm(xn + (size_t)t * H, h + (size_t)t * H, L->ln2, H, c->rms_eps);
        qt_sgemm(T, xn, H, &L->gate, g, I, 0);
        qt_sgemm(T, xn, H, &L->up, u, I, 0);
        for (size_t i = 0; i < (size_t)T * I; i++) g[i] = qt_silu(g[i]) * u[i];
        qt_sgemm(T, g, I, &L->down, o, H, 0);
        for (size_t i = 0; i < (size_t)T * H; i++) h[i] += L->ls_mlp[i % H] * o[i];
    }
    for (int t = 0; t < T; t++) qt_rmsnorm(h + (size_t)t * H, h + (size_t)t * H, c->norm_w, H, c->rms_eps);
    float *y = (float *)malloc(sizeof(float) * (size_t)T * c->latent);
    qt_sgemm(T, h, H, &c->out_proj, y, c->latent, 0);
    qc__add_bias(y, c->out_proj_b, T, c->latent);
    free(h); free(xn); free(q); free(k); free(v); free(att); free(o); free(g); free(u);
    return y;
}

static float *qc__convnext(const qc_upblock *U, float *x, int T, int d) {
    float *y = qc__conv(&U->dw, x, T);
    float *n = (float *)malloc(sizeof(float) * (size_t)T * d);
    for (int t = 0; t < T; t++) qt_layernorm(n + (size_t)t * d, y + (size_t)t * d, U->ln_w, U->ln_b, d, 1e-6f);
    float *hbuf = (float *)malloc(sizeof(float) * (size_t)T * 4 * d);
    qt_sgemm(T, n, d, &U->pw1, hbuf, 4 * d, 0);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int i = 0; i < 4 * d; i++) {
            float *p = hbuf + (size_t)t * 4 * d + i;
            *p = qt_gelu_erf(*p + U->pw1_b[i]);
        }
    qt_sgemm(T, hbuf, 4 * d, &U->pw2, y, d, 0);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++) {
            size_t idx = (size_t)t * d + i;
            x[idx] += U->gamma[i] * (y[idx] + U->pw2_b[i]);
        }
    free(y); free(n); free(hbuf);
    return x;
}

void qtts_codec_rvq(const qtts_codec *c, const int32_t *codes, int T, float *x) {
    int vqd = c->cb_dim / 2, D = c->cb_dim;
    /* sum within each group (semantic / acoustic) in 256-d, then the 1x1 output projections */
    float *sf = (float *)calloc((size_t)T * vqd, sizeof(float));
    float *sr = (float *)calloc((size_t)T * vqd, sizeof(float));
    for (int t = 0; t < T; t++)
        for (int q = 0; q < c->nq; q++) {
            int code = codes[(size_t)t * c->nq + q];
            if (code < 0) code = 0;
            const float *e = c->codebooks + ((size_t)q * c->cb_size + code) * vqd;
            float *dst = (q == 0 ? sf : sr) + (size_t)t * vqd;
            for (int i = 0; i < vqd; i++) dst[i] += e[i];
        }
    for (int t = 0; t < T; t++)
        for (int o = 0; o < D; o++) {
            float s = 0.0f;
            for (int i = 0; i < vqd; i++)
                s += c->out_first[(size_t)o * vqd + i] * sf[(size_t)t * vqd + i]
                   + c->out_rest[(size_t)o * vqd + i] * sr[(size_t)t * vqd + i];
            x[(size_t)t * D + o] = s;
        }
    free(sf); free(sr);
}

/* Decode one chunk (no chunking logic). Returns wav of T*upsample samples. */
static float *qc__decode_chunk(const qtts_codec *c, const int32_t *codes, int T, const char *dump) {
    int D = c->cb_dim;
    float *x = (float *)malloc(sizeof(float) * (size_t)T * D);
    qtts_codec_rvq(c, codes, T, x);
    qc__dump(dump, "codec_rvq", x, T, D);

    float *h = qc__conv(&c->pre_conv, x, T);
    free(x);
    qc__dump(dump, "codec_preconv", h, T, c->latent);
    float *tf = qc__transformer(c, h, T);
    free(h);
    if (dump) { /* pre_transformer output is [T, latent] in the reference */
        char p[1024]; int dims[2] = { T, c->latent };
        snprintf(p, sizeof(p), "%s/codec_pretf.npy", dump);
        qt_npy_save_f32(p, tf, 2, dims);
    }
    int Tc = T;
    h = tf;
    for (int u = 0; u < c->n_up_ratio; u++) {
        float *y = qc__convt(&c->up[u].convt, h, Tc);
        free(h);
        Tc *= c->up_ratios[u];
        h = qc__convnext(&c->up[u], y, Tc, c->latent);
        char nm[32]; snprintf(nm, sizeof(nm), "codec_up%d", u);
        qc__dump(dump, nm, h, Tc, c->latent);
    }
    float *y = qc__conv(&c->dec_in, h, Tc);
    free(h);
    h = y;
    qc__dump(dump, "codec_dec0", h, Tc, c->dec_dim);
    int C = c->dec_dim;
    for (int b = 0; b < c->n_rates; b++) {
        const qc_decblock *B = &c->blocks[b];
        qc__snake(&B->snake, h, Tc);
        y = qc__convt(&B->convt, h, Tc);
        free(h);
        h = y;
        Tc *= c->rates[b];
        C = B->convt.cout;
        for (int r = 0; r < 3; r++) {
            const qc_resunit *R = &B->res[r];
            size_t n = (size_t)Tc * C;
            float *a = (float *)malloc(sizeof(float) * n);
            memcpy(a, h, sizeof(float) * n);
            qc__snake(&R->act1, a, Tc);
            float *t1 = qc__conv(&R->conv1, a, Tc);
            free(a);
            qc__snake(&R->act2, t1, Tc);
            float *t2 = qc__conv(&R->conv2, t1, Tc);
            free(t1);
            for (size_t i = 0; i < n; i++) h[i] += t2[i];
            free(t2);
        }
        char nm[32]; snprintf(nm, sizeof(nm), "codec_dec%d", b + 1);
        qc__dump(dump, nm, h, Tc, C);
    }
    qc__snake(&c->final_snake, h, Tc);
    { char nm[32]; snprintf(nm, sizeof(nm), "codec_dec%d", c->n_rates + 1); qc__dump(dump, nm, h, Tc, C); }
    y = qc__conv(&c->final_conv, h, Tc);
    free(h);
    { char nm[32]; snprintf(nm, sizeof(nm), "codec_dec%d", c->n_rates + 2); qc__dump(dump, nm, y, Tc, 1); }
    for (int i = 0; i < Tc; i++) y[i] = y[i] < -1.0f ? -1.0f : (y[i] > 1.0f ? 1.0f : y[i]);
    return y;
}

float *qtts_codec_decode(qtts_codec *c, const int32_t *codes, int T, int *n_out, const char *dump_dir) {
    const int chunk = 300, left_ctx = 25;
    float *wav = (float *)malloc(sizeof(float) * (size_t)T * c->upsample + 1);
    int out = 0;
    for (int start = 0; start < T; ) {
        int end = start + chunk < T ? start + chunk : T;
        int ctx = start - left_ctx > 0 ? left_ctx : start;
        float *w = qc__decode_chunk(c, codes + (size_t)(start - ctx) * c->nq, end - start + ctx,
                                    start == 0 ? dump_dir : NULL);
        int keep = (end - start) * c->upsample;
        memcpy(wav + out, w + (size_t)ctx * c->upsample, sizeof(float) * (size_t)keep);
        out += keep;
        free(w);
        start = end;
    }
    *n_out = out;
    return wav;
}

#endif /* QTTS_CODEC_IMPLEMENTATION */
