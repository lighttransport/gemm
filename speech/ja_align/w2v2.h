/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * w2v2.h - wav2vec2 (stable-layer-norm / "large" variant) encoder with dual
 * CTC heads, as used by sakasegawa/japanese-wav2vec2-large-hiragana-ctc.
 *
 * Follows the Apache-2.0 HF transformers Wav2Vec2 modeling code
 * (Wav2Vec2FeatureEncoder, Wav2Vec2FeatureProjection, Wav2Vec2PositionalConvEmbedding,
 * Wav2Vec2EncoderStableLayerNorm) and hiragana-asr's DualCTCModel:
 *   input: 16 kHz mono, normalized to zero mean / unit variance (eps 1e-7)
 *   CNN: conv(1->512,k10,s5) + per-channel GroupNorm + GELU, then 4x conv(k3,s2) and
 *        2x conv(k2,s2) + GELU (no bias)       -> 50 frames/s, 512-d
 *   projection: LayerNorm(512) -> Linear(512->1024)
 *   positional conv: Conv1d(1024,1024,k128,pad64,groups16), drop last frame, GELU, add
 *   24 pre-LN transformer layers (16 heads, FFN 4096, GELU), final LayerNorm
 *   phoneme head on layer `inter` output (un-normed), kana head on the final output.
 */
#ifndef JA_W2V2_H
#define JA_W2V2_H

#include "ja_base.h"

typedef struct w2v2_model w2v2_model;

typedef struct {
    int T;              /* frames (20 ms each) */
    int n_phon, n_kana; /* classes incl. blank (index 0) */
    float *phon_logp;   /* [T][n_phon] log posteriors */
    float *kana_logp;   /* [T][n_kana] */
} w2v2_output;

w2v2_model *w2v2_load(const char *safetensors_path);
void        w2v2_free(w2v2_model *m);
/* wav: 16 kHz mono float. dump_dir (optional) receives w2v_feat/w2v_h0/w2v_h<inter>/w2v_final .npy */
int         w2v2_run(w2v2_model *m, const float *wav, int n, w2v2_output *out, const char *dump_dir);
void        w2v2_output_free(w2v2_output *o);

#endif /* JA_W2V2_H */

#if defined(JA_W2V2_IMPLEMENTATION) && !defined(JA_W2V2_IMPL_DONE)
#define JA_W2V2_IMPL_DONE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *ln1_w, *ln1_b, *ln2_w, *ln2_b;
    float *bq, *bk, *bv, *bo, *b1, *b2;
    ja_packed q, k, v, o, fc1, fc2;
} w2v2_layer;

struct w2v2_model {
    int n_conv, conv_k[8], conv_s[8], conv_c;
    float *conv0_w;          /* [512][10] */
    float *gn_w, *gn_b;
    ja_packed conv[8];       /* layers 1..: W[n=cout][k = j*cin + ci] */
    float *fp_ln_w, *fp_ln_b, *fp_b;
    ja_packed fp;
    int hidden, n_heads, inter_dim, n_layers, inter, pos_k, pos_groups;
    ja_packed pos[64];       /* per group: W[n=64][k = j*64 + i] */
    float *pos_b;
    w2v2_layer *layers;
    float *fln_w, *fln_b;
    int n_phon, n_kana;
    ja_packed phon, kana;
    float *phon_b, *kana_b;
};

static float *w2v2__need(const ja_st *st, const char *name, size_t expect) {
    size_t n = 0;
    float *d = ja_st_f32(st, name, &n);
    if (!d) { fprintf(stderr, "w2v2: missing tensor %s\n", name); exit(1); }
    if (expect && n != expect) { fprintf(stderr, "w2v2: %s has %zu elems, expected %zu\n", name, n, expect); exit(1); }
    return d;
}

static void w2v2__linear(const ja_st *st, ja_packed *p, float **bias, const char *prefix, int n, int k) {
    char nm[256];
    snprintf(nm, sizeof(nm), "%s.weight", prefix);
    float *w = w2v2__need(st, nm, (size_t)n * k);
    ja_pack_nk(p, w, n, k);
    free(w);
    if (bias) { snprintf(nm, sizeof(nm), "%s.bias", prefix); *bias = w2v2__need(st, nm, (size_t)n); }
}

w2v2_model *w2v2_load(const char *path) {
    ja_st st;
    if (ja_st_open(&st, path)) { fprintf(stderr, "w2v2: cannot open %s\n", path); return NULL; }
    w2v2_model *m = (w2v2_model *)calloc(1, sizeof(*m));
    char nm[256];
    /* conv stack: shapes from the tensors */
    m->conv_c = 512;
    for (m->n_conv = 0; m->n_conv < 8; m->n_conv++) {
        snprintf(nm, sizeof(nm), "feature_extractor.conv_layers.%d.conv.weight", m->n_conv);
        const ja_st_tensor *t = ja_st_find(&st, nm);
        if (!t) break;
        m->conv_k[m->n_conv] = (int)t->shape[2];
        m->conv_c = (int)t->shape[0];
    }
    /* strides of the wav2vec2 feature encoder: 5 then 2 */
    for (int i = 0; i < m->n_conv; i++) m->conv_s[i] = i == 0 ? 5 : 2;
    int C = m->conv_c;
    m->conv0_w = w2v2__need(&st, "feature_extractor.conv_layers.0.conv.weight", (size_t)C * m->conv_k[0]);
    m->gn_w = w2v2__need(&st, "feature_extractor.conv_layers.0.layer_norm.weight", (size_t)C);
    m->gn_b = w2v2__need(&st, "feature_extractor.conv_layers.0.layer_norm.bias", (size_t)C);
    for (int i = 1; i < m->n_conv; i++) {
        int k = m->conv_k[i];
        snprintf(nm, sizeof(nm), "feature_extractor.conv_layers.%d.conv.weight", i);
        float *w = w2v2__need(&st, nm, (size_t)C * C * k);
        float *r = (float *)malloc(sizeof(float) * (size_t)C * C * k);
        for (int o = 0; o < C; o++)
            for (int ci = 0; ci < C; ci++)
                for (int j = 0; j < k; j++) r[(size_t)o * C * k + (size_t)j * C + ci] = w[((size_t)o * C + ci) * k + j];
        ja_pack_nk(&m->conv[i], r, C, C * k);
        free(w); free(r);
    }
    m->fp_ln_w = w2v2__need(&st, "feature_projection.layer_norm.weight", (size_t)C);
    m->fp_ln_b = w2v2__need(&st, "feature_projection.layer_norm.bias", (size_t)C);
    const ja_st_tensor *tp = ja_st_find(&st, "feature_projection.projection.weight");
    if (!tp) { fprintf(stderr, "w2v2: missing feature projection\n"); exit(1); }
    m->hidden = (int)tp->shape[0];
    int H = m->hidden;
    w2v2__linear(&st, &m->fp, &m->fp_b, "feature_projection.projection", H, C);

    const ja_st_tensor *tpc = ja_st_find(&st, "encoder.pos_conv_embed.conv.weight");
    if (!tpc) { fprintf(stderr, "w2v2: missing folded pos conv (run convert_ckpt.py)\n"); exit(1); }
    int gi = (int)tpc->shape[1];
    m->pos_k = (int)tpc->shape[2];
    m->pos_groups = H / gi;
    {
        float *w = w2v2__need(&st, "encoder.pos_conv_embed.conv.weight", (size_t)H * gi * m->pos_k);
        float *r = (float *)malloc(sizeof(float) * (size_t)gi * gi * m->pos_k);
        for (int g = 0; g < m->pos_groups; g++) {
            for (int o = 0; o < gi; o++)
                for (int i = 0; i < gi; i++)
                    for (int j = 0; j < m->pos_k; j++)
                        r[(size_t)o * gi * m->pos_k + (size_t)j * gi + i] = w[(((size_t)(g * gi + o)) * gi + i) * m->pos_k + j];
            ja_pack_nk(&m->pos[g], r, gi, gi * m->pos_k);
        }
        free(w); free(r);
    }
    m->pos_b = w2v2__need(&st, "encoder.pos_conv_embed.conv.bias", (size_t)H);

    for (m->n_layers = 0;; m->n_layers++) {
        snprintf(nm, sizeof(nm), "encoder.layers.%d.attention.q_proj.weight", m->n_layers);
        if (!ja_st_find(&st, nm)) break;
    }
    snprintf(nm, sizeof(nm), "encoder.layers.0.feed_forward.intermediate_dense.weight");
    m->inter_dim = (int)ja_st_find(&st, nm)->shape[0];
    m->n_heads = 16;
    m->layers = (w2v2_layer *)calloc((size_t)m->n_layers, sizeof(w2v2_layer));
    for (int l = 0; l < m->n_layers; l++) {
        w2v2_layer *L = &m->layers[l];
        char p[128];
#define W2P(s) (snprintf(p, sizeof(p), "encoder.layers.%d.%s", l, s), p)
        snprintf(nm, sizeof(nm), "%s.weight", W2P("layer_norm")); L->ln1_w = w2v2__need(&st, nm, (size_t)H);
        snprintf(nm, sizeof(nm), "%s.bias", W2P("layer_norm")); L->ln1_b = w2v2__need(&st, nm, (size_t)H);
        snprintf(nm, sizeof(nm), "%s.weight", W2P("final_layer_norm")); L->ln2_w = w2v2__need(&st, nm, (size_t)H);
        snprintf(nm, sizeof(nm), "%s.bias", W2P("final_layer_norm")); L->ln2_b = w2v2__need(&st, nm, (size_t)H);
        w2v2__linear(&st, &L->q, &L->bq, W2P("attention.q_proj"), H, H);
        w2v2__linear(&st, &L->k, &L->bk, W2P("attention.k_proj"), H, H);
        w2v2__linear(&st, &L->v, &L->bv, W2P("attention.v_proj"), H, H);
        w2v2__linear(&st, &L->o, &L->bo, W2P("attention.out_proj"), H, H);
        w2v2__linear(&st, &L->fc1, &L->b1, W2P("feed_forward.intermediate_dense"), m->inter_dim, H);
        w2v2__linear(&st, &L->fc2, &L->b2, W2P("feed_forward.output_dense"), H, m->inter_dim);
#undef W2P
    }
    m->fln_w = w2v2__need(&st, "encoder.layer_norm.weight", (size_t)H);
    m->fln_b = w2v2__need(&st, "encoder.layer_norm.bias", (size_t)H);
    m->n_phon = (int)ja_st_find(&st, "phoneme_head.weight")->shape[0];
    m->n_kana = (int)ja_st_find(&st, "kana_head.weight")->shape[0];
    w2v2__linear(&st, &m->phon, &m->phon_b, "phoneme_head", m->n_phon, H);
    w2v2__linear(&st, &m->kana, &m->kana_b, "kana_head", m->n_kana, H);
    m->inter = st.meta_inter[0] ? atoi(st.meta_inter) : m->n_layers / 2;
    ja_st_close(&st);
    return m;
}

void w2v2_free(w2v2_model *m) {
    if (!m) return;
    free(m->conv0_w); free(m->gn_w); free(m->gn_b);
    for (int i = 1; i < m->n_conv; i++) ja_packed_free(&m->conv[i]);
    free(m->fp_ln_w); free(m->fp_ln_b); free(m->fp_b); ja_packed_free(&m->fp);
    for (int g = 0; g < m->pos_groups; g++) ja_packed_free(&m->pos[g]);
    free(m->pos_b);
    for (int l = 0; l < m->n_layers; l++) {
        w2v2_layer *L = &m->layers[l];
        free(L->ln1_w); free(L->ln1_b); free(L->ln2_w); free(L->ln2_b);
        free(L->bq); free(L->bk); free(L->bv); free(L->bo); free(L->b1); free(L->b2);
        ja_packed_free(&L->q); ja_packed_free(&L->k); ja_packed_free(&L->v); ja_packed_free(&L->o);
        ja_packed_free(&L->fc1); ja_packed_free(&L->fc2);
    }
    free(m->layers); free(m->fln_w); free(m->fln_b);
    ja_packed_free(&m->phon); ja_packed_free(&m->kana); free(m->phon_b); free(m->kana_b);
    free(m);
}

static void w2v2__bias(float *y, const float *b, int T, int n) {
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) for (int i = 0; i < n; i++) y[(size_t)t * n + i] += b[i];
}

static void w2v2__dump(const char *dir, const char *name, const float *x, int T, int C) {
    if (!dir) return;
    char p[1024];
    snprintf(p, sizeof(p), "%s/%s.npy", dir, name);
    int dims[2] = { T, C };
    ja_npy_save_f32(p, x, 2, dims);
}

/* Full bidirectional multi-head attention: out[T][H] from q,k,v [T][H]. */
static void w2v2__attention(float *out, const float *q, const float *k, const float *v, int T, int nh, int hd) {
    int H = nh * hd;
    float scale = 1.0f / sqrtf((float)hd);
    #pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int h = 0; h < nh; h++)
        for (int i = 0; i < T; i++) {
            float *s = (float *)malloc(sizeof(float) * (size_t)T);
            const float *qi = q + (size_t)i * H + h * hd;
            float mx = -INFINITY;
            for (int j = 0; j < T; j++) {
                const float *kj = k + (size_t)j * H + h * hd;
                float d = 0.0f;
                for (int t = 0; t < hd; t++) d += qi[t] * kj[t];
                s[j] = d * scale;
                if (s[j] > mx) mx = s[j];
            }
            double den = 0.0;
            for (int j = 0; j < T; j++) { s[j] = expf(s[j] - mx); den += s[j]; }
            float inv = (float)(1.0 / den);
            float *o = out + (size_t)i * H + h * hd;
            for (int t = 0; t < hd; t++) o[t] = 0.0f;
            for (int j = 0; j < T; j++) {
                const float *vj = v + (size_t)j * H + h * hd;
                float p = s[j] * inv;
                for (int t = 0; t < hd; t++) o[t] += p * vj[t];
            }
            free(s);
        }
}

int w2v2_run(w2v2_model *m, const float *wav, int n, w2v2_output *out, const char *dump) {
    memset(out, 0, sizeof(*out));
    int C = m->conv_c, H = m->hidden;
    /* normalize */
    double mu = 0.0, var = 0.0;
    for (int i = 0; i < n; i++) mu += wav[i];
    mu /= n;
    for (int i = 0; i < n; i++) { double d = wav[i] - mu; var += d * d; }
    var /= n;
    float inv = (float)(1.0 / sqrt(var + 1e-7));
    float *x = (float *)malloc(sizeof(float) * (size_t)n);
    for (int i = 0; i < n; i++) x[i] = (float)(wav[i] - mu) * inv;

    /* conv0 (1 -> C) + GroupNorm(C groups) + GELU */
    int k0 = m->conv_k[0], s0 = m->conv_s[0];
    int T = (n - k0) / s0 + 1;
    if (T < 1) { free(x); return -1; }
    float *h = (float *)malloc(sizeof(float) * (size_t)T * C);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++) {
            float s = 0.0f;
            for (int j = 0; j < k0; j++) s += m->conv0_w[(size_t)c * k0 + j] * x[(size_t)t * s0 + j];
            h[(size_t)t * C + c] = s;
        }
    free(x);
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; c++) {
        double a = 0.0, b = 0.0;
        for (int t = 0; t < T; t++) a += h[(size_t)t * C + c];
        a /= T;
        for (int t = 0; t < T; t++) { double d = h[(size_t)t * C + c] - a; b += d * d; }
        b /= T;
        float r = (float)(1.0 / sqrt(b + 1e-5));
        for (int t = 0; t < T; t++) {
            float *p = &h[(size_t)t * C + c];
            *p = ja_gelu((float)(*p - a) * r * m->gn_w[c] + m->gn_b[c]);
        }
    }
    /* strided convs: the k*C window of rows t*s..t*s+k-1 is contiguous (channels-last) */
    for (int i = 1; i < m->n_conv; i++) {
        int k = m->conv_k[i], s = m->conv_s[i];
        int To = (T - k) / s + 1;
        float *y = (float *)malloc(sizeof(float) * (size_t)To * C);
        ja_sgemm(To, h, s * C, &m->conv[i], y, C, 0);
        #pragma omp parallel for schedule(static)
        for (size_t e = 0; e < (size_t)To * C; e++) y[e] = ja_gelu(y[e]);
        free(h);
        h = y;
        T = To;
    }
    w2v2__dump(dump, "w2v_feat", h, T, C);

    /* feature projection */
    float *hn = (float *)malloc(sizeof(float) * (size_t)T * C);
    for (int t = 0; t < T; t++) ja_layernorm(hn + (size_t)t * C, h + (size_t)t * C, m->fp_ln_w, m->fp_ln_b, C, 1e-5f);
    free(h);
    float *hs = (float *)malloc(sizeof(float) * (size_t)T * H);
    ja_sgemm(T, hn, C, &m->fp, hs, H, 0);
    w2v2__bias(hs, m->fp_b, T, H);
    free(hn);

    /* positional conv: pad k/2 both sides, output T+1 frames, keep the first T */
    {
        int K = m->pos_k, pad = K / 2, gi = H / m->pos_groups;
        int Tp = T + 2 * pad;
        float *col = (float *)malloc(sizeof(float) * (size_t)T * K * gi);
        float *pc = (float *)malloc(sizeof(float) * (size_t)T * H);
        for (int g = 0; g < m->pos_groups; g++) {
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < T; t++)
                for (int j = 0; j < K; j++) {
                    int ti = t + j - pad;
                    float *dst = col + ((size_t)t * K + j) * gi;
                    if (ti < 0 || ti >= T) memset(dst, 0, sizeof(float) * gi);
                    else memcpy(dst, hs + (size_t)ti * H + g * gi, sizeof(float) * gi);
                }
            float *tmp = (float *)malloc(sizeof(float) * (size_t)T * gi);
            ja_sgemm(T, col, K * gi, &m->pos[g], tmp, gi, 0);
            for (int t = 0; t < T; t++) memcpy(pc + (size_t)t * H + g * gi, tmp + (size_t)t * gi, sizeof(float) * gi);
            free(tmp);
        }
        (void)Tp;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++)
            for (int c = 0; c < H; c++) {
                size_t e = (size_t)t * H + c;
                hs[e] += ja_gelu(pc[e] + m->pos_b[c]);
            }
        free(col); free(pc);
    }
    w2v2__dump(dump, "w2v_h0", hs, T, H);

    int hd = H / m->n_heads, I = m->inter_dim;
    float *xn = (float *)malloc(sizeof(float) * (size_t)T * H);
    float *q = (float *)malloc(sizeof(float) * (size_t)T * H);
    float *kk = (float *)malloc(sizeof(float) * (size_t)T * H);
    float *vv = (float *)malloc(sizeof(float) * (size_t)T * H);
    float *att = (float *)malloc(sizeof(float) * (size_t)T * H);
    float *ff = (float *)malloc(sizeof(float) * (size_t)T * I);
    out->T = T;
    out->n_phon = m->n_phon;
    out->n_kana = m->n_kana;
    out->phon_logp = (float *)malloc(sizeof(float) * (size_t)T * m->n_phon);
    out->kana_logp = (float *)malloc(sizeof(float) * (size_t)T * m->n_kana);
    for (int l = 0; l < m->n_layers; l++) {
        const w2v2_layer *L = &m->layers[l];
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) ja_layernorm(xn + (size_t)t * H, hs + (size_t)t * H, L->ln1_w, L->ln1_b, H, 1e-5f);
        ja_sgemm(T, xn, H, &L->q, q, H, 0); w2v2__bias(q, L->bq, T, H);
        ja_sgemm(T, xn, H, &L->k, kk, H, 0); w2v2__bias(kk, L->bk, T, H);
        ja_sgemm(T, xn, H, &L->v, vv, H, 0); w2v2__bias(vv, L->bv, T, H);
        w2v2__attention(att, q, kk, vv, T, m->n_heads, hd);
        ja_sgemm(T, att, H, &L->o, q, H, 0); w2v2__bias(q, L->bo, T, H);
        for (size_t e = 0; e < (size_t)T * H; e++) hs[e] += q[e];
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) ja_layernorm(xn + (size_t)t * H, hs + (size_t)t * H, L->ln2_w, L->ln2_b, H, 1e-5f);
        ja_sgemm(T, xn, H, &L->fc1, ff, I, 0);
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++)
            for (int i = 0; i < I; i++) { float *p = &ff[(size_t)t * I + i]; *p = ja_gelu(*p + L->b1[i]); }
        ja_sgemm(T, ff, I, &L->fc2, q, H, 0); w2v2__bias(q, L->b2, T, H);
        for (size_t e = 0; e < (size_t)T * H; e++) hs[e] += q[e];
        if (l + 1 == m->inter) {
            char nm[32]; snprintf(nm, sizeof(nm), "w2v_h%d", m->inter);
            w2v2__dump(dump, nm, hs, T, H);
            ja_sgemm(T, hs, H, &m->phon, out->phon_logp, m->n_phon, 0);
            w2v2__bias(out->phon_logp, m->phon_b, T, m->n_phon);
        }
    }
    for (int t = 0; t < T; t++) ja_layernorm(hs + (size_t)t * H, hs + (size_t)t * H, m->fln_w, m->fln_b, H, 1e-5f);
    w2v2__dump(dump, "w2v_final", hs, T, H);
    ja_sgemm(T, hs, H, &m->kana, out->kana_logp, m->n_kana, 0);
    w2v2__bias(out->kana_logp, m->kana_b, T, m->n_kana);
    for (int t = 0; t < T; t++) {
        ja_log_softmax(out->phon_logp + (size_t)t * m->n_phon, m->n_phon);
        ja_log_softmax(out->kana_logp + (size_t)t * m->n_kana, m->n_kana);
    }
    w2v2__dump(dump, "phoneme_logp", out->phon_logp, T, m->n_phon);
    w2v2__dump(dump, "kana_logp", out->kana_logp, T, m->n_kana);
    free(hs); free(xn); free(q); free(kk); free(vv); free(att); free(ff);
    return 0;
}

void w2v2_output_free(w2v2_output *o) {
    free(o->phon_logp); free(o->kana_logp);
    memset(o, 0, sizeof(*o));
}

#endif /* JA_W2V2_IMPLEMENTATION */
