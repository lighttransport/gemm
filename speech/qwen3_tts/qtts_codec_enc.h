/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * qtts_codec_enc.h - Qwen3-TTS-Tokenizer-12Hz encoder (24 kHz audio -> [T][16] codes), CPU.
 *
 * Follows the Apache-2.0 HF transformers MimiModel.encode path used by qwen_tts
 * (Qwen3TTSTokenizerV2Encoder):
 *   SEANet encoder: causal conv k7 (1->64), then per ratio (4,5,6,8): residual unit
 *     [ELU, conv k3 (C->C/2), ELU, conv k1 (C/2->C)] + skip, ELU, strided conv k=2r (C->2C);
 *     ELU, conv k3 (1024->512).  Causal "Mimi" padding: left = (k-1)*d + 1 - stride zeros,
 *     right = extra zeros so the last window is complete.
 *   8-layer causal transformer (LayerNorm, RoPE theta 1e4, 8x64 heads, GELU MLP, LayerScale)
 *   downsample: conv k4 stride 2 (replicate padding) -> 12.5 Hz
 *   split RVQ encode: semantic (1) and acoustic (first 15) codebooks, each group with its own
 *     1x1 input projection, nearest-centroid (euclidean) residual quantization.
 * Codes are trimmed to ceil(n / 1920) frames like the reference wrapper.
 *
 * Requires qtts_ops.h, safetensors.h. Define QTTS_CODEC_ENC_IMPLEMENTATION once.
 */
#ifndef QTTS_CODEC_ENC_H
#define QTTS_CODEC_ENC_H

#include <stdint.h>
#include "qtts_ops.h"

typedef struct qtts_cenc qtts_cenc;

qtts_cenc *qtts_cenc_load(const char *speech_tokenizer_dir);
void       qtts_cenc_free(qtts_cenc *e);
/* wav 24 kHz mono -> malloc'd codes [T][16] (T returned in *T). dump_dir: enc_seanet/enc_tf/enc_down */
int32_t   *qtts_cenc_encode(qtts_cenc *e, const float *wav, int n, int *T, const char *dump_dir);

#endif /* QTTS_CODEC_ENC_H */

#if defined(QTTS_CODEC_ENC_IMPLEMENTATION) && !defined(QTTS_CODEC_ENC_IMPL_DONE)
#define QTTS_CODEC_ENC_IMPL_DONE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "safetensors.h"

typedef struct { qt_packed w; float *b; int cin, cout, k, s, replicate; } qe_conv;

typedef struct {
    float *ln1w, *ln1b, *ln2w, *ln2b, *ls1, *ls2;
    qt_packed q, k, v, o, fc1, fc2;
} qe_layer;

struct qtts_cenc {
    st_context *st;
    int n_ratio, ratios[8], nf, hidden, n_layers, nh, hd, inter, nq_valid, n_acoustic, cb_size, cb_dim;
    float theta, eps;
    qe_conv c0, res1[8], res2[8], down[8], last, ds;
    qe_layer *L;
    float *sem_in, *ac_in;       /* input projections [256][512] */
    float *sem_cb, *ac_cb;       /* codebooks [n][2048][256] (embed_sum / usage) */
    float *sem_n2, *ac_n2;       /* squared norms of centroids */
};

typedef struct { const float *w; int cin, k; } qe__ctx;
static float qe__get(const void *w, int n, int kk, const void *ctx) {
    const qe__ctx *c = (const qe__ctx *)ctx; (void)w;
    int j = kk / c->cin, ci = kk % c->cin;
    return c->w[((size_t)n * c->cin + ci) * c->k + j];
}

static const float *qe__f(qtts_cenc *e, const char *name, size_t n) {
    int i = safetensors_find(e->st, name);
    if (i < 0) { fprintf(stderr, "qtts_cenc: missing %s\n", name); exit(1); }
    if (strcmp(safetensors_dtype(e->st, i), "F32") || safetensors_nbytes(e->st, i) != n * 4) {
        fprintf(stderr, "qtts_cenc: %s has unexpected dtype/size\n", name); exit(1);
    }
    return (const float *)safetensors_data(e->st, i);
}
static float *qe__copy(qtts_cenc *e, const char *name, size_t n) {
    float *d = (float *)malloc(sizeof(float) * n);
    memcpy(d, qe__f(e, name, n), sizeof(float) * n);
    return d;
}

static void qe__conv_load(qtts_cenc *e, qe_conv *c, const char *prefix, int cin, int cout, int k, int s, int bias) {
    char nm[256];
    c->cin = cin; c->cout = cout; c->k = k; c->s = s; c->replicate = 0;
    snprintf(nm, sizeof(nm), "%s.weight", prefix);
    qe__ctx ctx = { qe__f(e, nm, (size_t)cout * cin * k), cin, k };
    qt_pack_b(&c->w, cout, k * cin, qe__get, ctx.w, &ctx);
    c->b = NULL;
    if (bias) { snprintf(nm, sizeof(nm), "%s.bias", prefix); c->b = qe__copy(e, nm, (size_t)cout); }
}

/* Mimi causal conv on x[T][cin] -> *To rows of [cout]; strided windows are contiguous rows */
static float *qe__conv(const qe_conv *c, const float *x, int T, int *To) {
    int pt = c->k - c->s;                                   /* dilation 1 everywhere here */
    double nfr = (double)(T - c->k + pt) / c->s + 1.0;
    int n_frames = (int)ceil(nfr) - 1;
    int ideal = n_frames * c->s + c->k - pt;
    int extra = ideal - T;
    int Tp = T + pt + extra;
    float *xp = (float *)malloc(sizeof(float) * (size_t)Tp * c->cin);
    for (int t = 0; t < Tp; t++) {
        int src = t - pt;
        float *dst = xp + (size_t)t * c->cin;
        if (src >= 0 && src < T) memcpy(dst, x + (size_t)src * c->cin, sizeof(float) * c->cin);
        else if (c->replicate) memcpy(dst, x + (size_t)(src < 0 ? 0 : T - 1) * c->cin, sizeof(float) * c->cin);
        else memset(dst, 0, sizeof(float) * c->cin);
    }
    int n = (Tp - c->k) / c->s + 1;
    float *y = (float *)malloc(sizeof(float) * (size_t)n * c->cout);
    qt_sgemm(n, xp, c->s * c->cin, &c->w, y, c->cout, 0);
    if (c->b)
        for (int t = 0; t < n; t++) for (int o = 0; o < c->cout; o++) y[(size_t)t * c->cout + o] += c->b[o];
    free(xp);
    *To = n;
    return y;
}

static void qe__elu(float *x, size_t n) { for (size_t i = 0; i < n; i++) if (x[i] < 0) x[i] = expm1f(x[i]); }

static int qe__ji(const json_val *o, const char *k, int d) { const json_val *v = json_obj_get(o, k); return v && v->type == JSON_NUMBER ? (int)v->num : d; }

qtts_cenc *qtts_cenc_load(const char *dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *js = (char *)malloc((size_t)len + 1);
    if (fread(js, 1, (size_t)len, f) != (size_t)len) { fclose(f); free(js); return NULL; }
    js[len] = 0;
    fclose(f);
    json_val *root = json_parse(js, (int)len);
    free(js);
    const json_val *ec = json_obj_get(root, "encoder_config");
    qtts_cenc *e = (qtts_cenc *)calloc(1, sizeof(*e));
    e->nf = qe__ji(ec, "num_filters", 64);
    e->hidden = qe__ji(ec, "hidden_size", 512);
    e->n_layers = qe__ji(ec, "num_hidden_layers", 8);
    e->nh = qe__ji(ec, "num_attention_heads", 8);
    e->hd = qe__ji(ec, "head_dim", 64);
    e->inter = qe__ji(ec, "intermediate_size", 2048);
    e->cb_size = qe__ji(ec, "codebook_size", 2048);
    e->cb_dim = qe__ji(ec, "vector_quantization_hidden_dimension", 256);
    e->nq_valid = qe__ji(root, "encoder_valid_num_quantizers", 16);
    e->n_acoustic = e->nq_valid - 1;
    e->theta = 10000.0f;
    e->eps = 1e-5f;
    const json_val *ur = json_obj_get(ec, "upsampling_ratios");
    e->n_ratio = ur ? ur->arr.count : 0;
    for (int i = 0; i < e->n_ratio; i++) e->ratios[i] = (int)ur->arr.items[e->n_ratio - 1 - i].num;  /* reversed */
    const json_val *rt = json_obj_get(ec, "rope_theta");
    if (rt && rt->type == JSON_NUMBER) e->theta = (float)rt->num;
    json_free(root);

    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    e->st = safetensors_open(path);
    if (!e->st) { free(e); return NULL; }
    char nm[256];
    qe__conv_load(e, &e->c0, "encoder.encoder.layers.0.conv", 1, e->nf, 7, 1, 1);
    int C = e->nf, li = 1;
    for (int r = 0; r < e->n_ratio; r++) {
        snprintf(nm, sizeof(nm), "encoder.encoder.layers.%d.block.1.conv", li);
        qe__conv_load(e, &e->res1[r], nm, C, C / 2, 3, 1, 1);
        snprintf(nm, sizeof(nm), "encoder.encoder.layers.%d.block.3.conv", li);
        qe__conv_load(e, &e->res2[r], nm, C / 2, C, 1, 1, 1);
        snprintf(nm, sizeof(nm), "encoder.encoder.layers.%d.conv", li + 2);
        qe__conv_load(e, &e->down[r], nm, C, 2 * C, 2 * e->ratios[r], e->ratios[r], 1);
        C *= 2;
        li += 3;
    }
    snprintf(nm, sizeof(nm), "encoder.encoder.layers.%d.conv", li + 1);
    qe__conv_load(e, &e->last, nm, C, e->hidden, 3, 1, 1);
    int H = e->hidden, qd = e->nh * e->hd, I = e->inter;
    e->L = (qe_layer *)calloc((size_t)e->n_layers, sizeof(qe_layer));
    for (int l = 0; l < e->n_layers; l++) {
        qe_layer *L = &e->L[l];
#define QEP(s) (snprintf(nm, sizeof(nm), "encoder.encoder_transformer.layers.%d.%s", l, s), nm)
        L->ln1w = qe__copy(e, QEP("input_layernorm.weight"), H); L->ln1b = qe__copy(e, QEP("input_layernorm.bias"), H);
        L->ln2w = qe__copy(e, QEP("post_attention_layernorm.weight"), H); L->ln2b = qe__copy(e, QEP("post_attention_layernorm.bias"), H);
        L->ls1 = qe__copy(e, QEP("self_attn_layer_scale.scale"), H); L->ls2 = qe__copy(e, QEP("mlp_layer_scale.scale"), H);
        qt_pack_b_f32_nk(&L->q, qe__f(e, QEP("self_attn.q_proj.weight"), (size_t)qd * H), qd, H);
        qt_pack_b_f32_nk(&L->k, qe__f(e, QEP("self_attn.k_proj.weight"), (size_t)qd * H), qd, H);
        qt_pack_b_f32_nk(&L->v, qe__f(e, QEP("self_attn.v_proj.weight"), (size_t)qd * H), qd, H);
        qt_pack_b_f32_nk(&L->o, qe__f(e, QEP("self_attn.o_proj.weight"), (size_t)H * qd), H, qd);
        qt_pack_b_f32_nk(&L->fc1, qe__f(e, QEP("mlp.fc1.weight"), (size_t)I * H), I, H);
        qt_pack_b_f32_nk(&L->fc2, qe__f(e, QEP("mlp.fc2.weight"), (size_t)H * I), H, I);
#undef QEP
    }
    qe__conv_load(e, &e->ds, "encoder.downsample.conv", H, H, 4, 2, 0);
    e->ds.replicate = 1;
    int D = e->cb_dim;
    e->sem_in = qe__copy(e, "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight", (size_t)D * H);
    e->ac_in = qe__copy(e, "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight", (size_t)D * H);
    e->sem_cb = (float *)malloc(sizeof(float) * (size_t)e->cb_size * D);
    e->ac_cb = (float *)malloc(sizeof(float) * (size_t)e->n_acoustic * e->cb_size * D);
    e->sem_n2 = (float *)malloc(sizeof(float) * (size_t)e->cb_size);
    e->ac_n2 = (float *)malloc(sizeof(float) * (size_t)e->n_acoustic * e->cb_size);
    for (int q = 0; q < 1 + e->n_acoustic; q++) {
        const char *grp = q == 0 ? "semantic" : "acoustic";
        int idx = q == 0 ? 0 : q - 1;
        snprintf(nm, sizeof(nm), "encoder.quantizer.%s_residual_vector_quantizer.layers.%d.codebook.embed_sum", grp, idx);
        const float *es = qe__f(e, nm, (size_t)e->cb_size * D);
        snprintf(nm, sizeof(nm), "encoder.quantizer.%s_residual_vector_quantizer.layers.%d.codebook.cluster_usage", grp, idx);
        const float *cu = qe__f(e, nm, (size_t)e->cb_size);
        float *cb = q == 0 ? e->sem_cb : e->ac_cb + (size_t)idx * e->cb_size * D;
        float *n2 = q == 0 ? e->sem_n2 : e->ac_n2 + (size_t)idx * e->cb_size;
        for (int c = 0; c < e->cb_size; c++) {
            float u = cu[c] < 1e-5f ? 1e-5f : cu[c];
            double s2 = 0.0;
            for (int d = 0; d < D; d++) { float v = es[(size_t)c * D + d] / u; cb[(size_t)c * D + d] = v; s2 += (double)v * v; }
            n2[c] = (float)s2;
        }
    }
    return e;
}

static void qe__conv_free(qe_conv *c) { qt_packed_free(&c->w); free(c->b); }

void qtts_cenc_free(qtts_cenc *e) {
    if (!e) return;
    qe__conv_free(&e->c0); qe__conv_free(&e->last); qe__conv_free(&e->ds);
    for (int r = 0; r < e->n_ratio; r++) { qe__conv_free(&e->res1[r]); qe__conv_free(&e->res2[r]); qe__conv_free(&e->down[r]); }
    for (int l = 0; l < e->n_layers; l++) {
        qe_layer *L = &e->L[l];
        free(L->ln1w); free(L->ln1b); free(L->ln2w); free(L->ln2b); free(L->ls1); free(L->ls2);
        qt_packed_free(&L->q); qt_packed_free(&L->k); qt_packed_free(&L->v); qt_packed_free(&L->o);
        qt_packed_free(&L->fc1); qt_packed_free(&L->fc2);
    }
    free(e->L); free(e->sem_in); free(e->ac_in); free(e->sem_cb); free(e->ac_cb); free(e->sem_n2); free(e->ac_n2);
    safetensors_close(e->st);
    free(e);
}

static void qe__dump(const char *dir, const char *name, const float *x, int T, int C) {
    if (!dir) return;
    char p[1024];
    snprintf(p, sizeof(p), "%s/%s.npy", dir, name);
    int dims[2] = { T, C };
    qt_npy_save_f32(p, x, 2, dims);
}

/* nearest centroid by squared euclidean distance ||c||^2 - 2 x.c (same argmin as cdist) */
static int qe__nearest(const float *x, const float *cb, const float *n2, int N, int D) {
    int best = 0;
    double bd = 1e300;
    for (int c = 0; c < N; c++) {
        const float *r = cb + (size_t)c * D;
        double dot = 0.0;
        for (int d = 0; d < D; d++) dot += (double)x[d] * r[d];
        double dist = n2[c] - 2.0 * dot;
        if (dist < bd) { bd = dist; best = c; }
    }
    return best;
}

int32_t *qtts_cenc_encode(qtts_cenc *e, const float *wav, int n, int *Tout, const char *dump) {
    int T = n, To;
    float *h = qe__conv(&e->c0, wav, T, &To);
    T = To;
    int C = e->nf;
    for (int r = 0; r < e->n_ratio; r++) {
        size_t sz = (size_t)T * C;
        float *a = (float *)malloc(sizeof(float) * sz);
        memcpy(a, h, sizeof(float) * sz);
        qe__elu(a, sz);
        float *b = qe__conv(&e->res1[r], a, T, &To);
        free(a);
        qe__elu(b, (size_t)T * (C / 2));
        float *c2 = qe__conv(&e->res2[r], b, T, &To);
        free(b);
        for (size_t i = 0; i < sz; i++) h[i] += c2[i];
        free(c2);
        qe__elu(h, sz);
        float *d = qe__conv(&e->down[r], h, T, &To);
        free(h);
        h = d;
        T = To;
        C *= 2;
    }
    qe__elu(h, (size_t)T * C);
    float *x = qe__conv(&e->last, h, T, &To);
    free(h);
    int H = e->hidden, qd = e->nh * e->hd, I = e->inter;
    qe__dump(dump, "enc_seanet", x, T, H);
    /* causal transformer */
    float *xn = (float *)malloc(sizeof(float) * (size_t)T * H), *q = (float *)malloc(sizeof(float) * (size_t)T * qd);
    float *k = (float *)malloc(sizeof(float) * (size_t)T * qd), *v = (float *)malloc(sizeof(float) * (size_t)T * qd);
    float *att = (float *)malloc(sizeof(float) * (size_t)T * qd), *o = (float *)malloc(sizeof(float) * (size_t)T * H);
    float *ff = (float *)malloc(sizeof(float) * (size_t)T * I);
    for (int l = 0; l < e->n_layers; l++) {
        const qe_layer *L = &e->L[l];
        for (int t = 0; t < T; t++) qt_layernorm(xn + (size_t)t * H, x + (size_t)t * H, L->ln1w, L->ln1b, H, e->eps);
        qt_sgemm(T, xn, H, &L->q, q, qd, 0);
        qt_sgemm(T, xn, H, &L->k, k, qd, 0);
        qt_sgemm(T, xn, H, &L->v, v, qd, 0);
        for (int t = 0; t < T; t++)
            for (int hh = 0; hh < e->nh; hh++) {
                qt_rope_neox(q + (size_t)t * qd + hh * e->hd, e->hd, t, e->theta);
                qt_rope_neox(k + (size_t)t * qd + hh * e->hd, e->hd, t, e->theta);
            }
        qt_attention(att, q, k, v, T, 0, e->nh, e->nh, e->hd, 0);
        qt_sgemm(T, att, qd, &L->o, o, H, 0);
        for (size_t i = 0; i < (size_t)T * H; i++) x[i] += L->ls1[i % H] * o[i];
        for (int t = 0; t < T; t++) qt_layernorm(xn + (size_t)t * H, x + (size_t)t * H, L->ln2w, L->ln2b, H, e->eps);
        qt_sgemm(T, xn, H, &L->fc1, ff, I, 0);
        for (size_t i = 0; i < (size_t)T * I; i++) ff[i] = qt_gelu_erf(ff[i]);
        qt_sgemm(T, ff, I, &L->fc2, o, H, 0);
        for (size_t i = 0; i < (size_t)T * H; i++) x[i] += L->ls2[i % H] * o[i];
    }
    free(xn); free(q); free(k); free(v); free(att); free(o); free(ff);
    qe__dump(dump, "enc_tf", x, T, H);
    float *y = qe__conv(&e->ds, x, T, &To);
    free(x);
    T = To;
    qe__dump(dump, "enc_down", y, T, H);
    /* split RVQ encode */
    int D = e->cb_dim, NQ = 1 + e->n_acoustic;
    int keep = (n + 1919) / 1920;
    if (keep > T) keep = T;
    int32_t *codes = (int32_t *)malloc(sizeof(int32_t) * (size_t)keep * NQ);
    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < keep; t++) {
        float rs[512], ra[512];
        const float *yt = y + (size_t)t * H;
        for (int d = 0; d < D; d++) {
            double a = 0.0, b = 0.0;
            for (int c = 0; c < H; c++) { a += (double)e->sem_in[(size_t)d * H + c] * yt[c]; b += (double)e->ac_in[(size_t)d * H + c] * yt[c]; }
            rs[d] = (float)a; ra[d] = (float)b;
        }
        codes[(size_t)t * NQ] = qe__nearest(rs, e->sem_cb, e->sem_n2, e->cb_size, D);
        for (int qi = 0; qi < e->n_acoustic; qi++) {
            const float *cb = e->ac_cb + (size_t)qi * e->cb_size * D;
            int c = qe__nearest(ra, cb, e->ac_n2 + (size_t)qi * e->cb_size, e->cb_size, D);
            codes[(size_t)t * NQ + 1 + qi] = c;
            for (int d = 0; d < D; d++) ra[d] -= cb[(size_t)c * D + d];
        }
    }
    free(y);
    *Tout = keep;
    return codes;
}

#endif /* QTTS_CODEC_ENC_IMPLEMENTATION */
