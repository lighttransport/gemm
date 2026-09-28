/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * qtts_spk.h - Qwen3-TTS (Base) speaker encoder: 24 kHz audio -> x-vector [enc_dim].
 *
 * Log-mel front end, written from the standard definitions (the reference computes the
 * same quantities with torch.stft + librosa, which serve only as numeric oracles here):
 *   reflect-pad (n_fft - hop) / 2, periodic Hann window, radix-2 FFT, |X| = sqrt(re^2 + im^2 + 1e-9),
 *   Slaney-scale mel filterbank with Slaney area normalization, log(max(mel, 1e-5)).
 *   n_fft 1024, hop 256, 128 mels, 0..12 kHz.
 * ECAPA-TDNN (Desplanques et al. 2020) following the Apache-2.0 qwen_tts
 * Qwen3TTSSpeakerEncoder: TDNN(k5) -> 3 SE-Res2Net blocks (k3, dil 2/3/4, scale 8) ->
 * multi-layer aggregation -> attentive statistics pooling -> 1x1 conv to enc_dim.
 * All convolutions use "same" reflect padding.
 *
 * Requires qtts_ops.h and safetensors.h. Define QTTS_SPK_IMPLEMENTATION once.
 */
#ifndef QTTS_SPK_H
#define QTTS_SPK_H

#include "qtts_ops.h"

typedef struct qtts_spk qtts_spk;

/* model_dir: Base model directory (config.json + model.safetensors with speaker_encoder.*) */
qtts_spk *qtts_spk_load(const char *model_dir);
void      qtts_spk_free(qtts_spk *s);
int       qtts_spk_dim(const qtts_spk *s);
/* 24 kHz mono -> out[enc_dim]. mel_out (optional, malloc'd [T][128]) receives the log-mel. */
int       qtts_spk_embed(qtts_spk *s, const float *wav, int n, float *out, float **mel_out, int *mel_T);
/* stand-alone log-mel (exposed for tests): returns malloc'd [T][n_mels] */
float    *qtts_logmel(const float *wav, int n, int n_fft, int hop, int n_mels, int sr, float fmin, float fmax, int *T);

#endif /* QTTS_SPK_H */

#if defined(QTTS_SPK_IMPLEMENTATION) && !defined(QTTS_SPK_IMPL_DONE)
#define QTTS_SPK_IMPL_DONE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "safetensors.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- FFT (iterative radix-2, in place) ---- */
static void qs__fft(double *re, double *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { double t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / len;
        for (int i = 0; i < n; i += len)
            for (int k = 0; k < len / 2; k++) {
                double wr = cos(ang * k), wi = sin(ang * k);
                double ur = re[i + k], ui = im[i + k];
                double vr = re[i + k + len / 2] * wr - im[i + k + len / 2] * wi;
                double vi = re[i + k + len / 2] * wi + im[i + k + len / 2] * wr;
                re[i + k] = ur + vr; im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
            }
    }
}

/* Slaney mel scale: linear below 1 kHz (200/3 Hz per mel), logarithmic above */
static double qs__hz2mel(double f) {
    const double fsp = 200.0 / 3.0, minhz = 1000.0, minmel = minhz / fsp, step = log(6.4) / 27.0;
    return f < minhz ? f / fsp : minmel + log(f / minhz) / step;
}
static double qs__mel2hz(double m) {
    const double fsp = 200.0 / 3.0, minhz = 1000.0, minmel = minhz / fsp, step = log(6.4) / 27.0;
    return m < minmel ? m * fsp : minhz * exp(step * (m - minmel));
}

float *qtts_logmel(const float *wav, int n, int n_fft, int hop, int n_mels, int sr, float fmin, float fmax, int *Tout) {
    int nb = n_fft / 2 + 1, pad = (n_fft - hop) / 2;
    int Lp = n + 2 * pad;
    if (Lp < n_fft) { *Tout = 0; return NULL; }
    int T = 1 + (Lp - n_fft) / hop;
    float *xp = (float *)malloc(sizeof(float) * (size_t)Lp);
    for (int i = 0; i < Lp; i++) {
        int j = i - pad;
        if (j < 0) j = -j;                  /* reflect (edge sample not repeated) */
        if (j >= n) j = 2 * (n - 1) - j;
        xp[i] = wav[j];
    }
    /* mel filterbank [n_mels][nb], float32 like the reference */
    float *fb = (float *)calloc((size_t)n_mels * nb, sizeof(float));
    double *hz = (double *)malloc(sizeof(double) * (size_t)(n_mels + 2));
    double m0 = qs__hz2mel(fmin), m1 = qs__hz2mel(fmax);
    for (int i = 0; i < n_mels + 2; i++) hz[i] = qs__mel2hz(m0 + (m1 - m0) * i / (n_mels + 1));
    for (int m = 0; m < n_mels; m++) {
        double enorm = 2.0 / (hz[m + 2] - hz[m]);
        for (int k = 0; k < nb; k++) {
            double f = (double)k * sr / n_fft;
            double lo = (f - hz[m]) / (hz[m + 1] - hz[m]), up = (hz[m + 2] - f) / (hz[m + 2] - hz[m + 1]);
            double w = lo < up ? lo : up;
            fb[(size_t)m * nb + k] = (float)((w > 0 ? w : 0.0) * enorm);
        }
    }
    float *out = (float *)malloc(sizeof(float) * (size_t)T * n_mels);
    #pragma omp parallel
    {
        double *re = (double *)malloc(sizeof(double) * n_fft), *im = (double *)malloc(sizeof(double) * n_fft);
        float *mag = (float *)malloc(sizeof(float) * nb);
        #pragma omp for schedule(static)
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < n_fft; i++) {
                double w = 0.5 - 0.5 * cos(2.0 * M_PI * i / n_fft);   /* periodic Hann */
                re[i] = xp[(size_t)t * hop + i] * w;
                im[i] = 0.0;
            }
            qs__fft(re, im, n_fft);
            for (int k = 0; k < nb; k++) mag[k] = (float)sqrt(re[k] * re[k] + im[k] * im[k] + 1e-9);
            for (int m = 0; m < n_mels; m++) {
                float s = 0.0f;
                for (int k = 0; k < nb; k++) s += fb[(size_t)m * nb + k] * mag[k];
                out[(size_t)t * n_mels + m] = logf(s > 1e-5f ? s : 1e-5f);
            }
        }
        free(re); free(im); free(mag);
    }
    free(xp); free(fb); free(hz);
    *Tout = T;
    return out;
}

/* ---- ECAPA-TDNN ---- */

typedef struct { qt_packed w; float *b; int cin, cout, k, dil; } qs_conv;

typedef struct {
    qs_conv tdnn1, tdnn2, res[8], se1, se2;
    int scale;
} qs_block;

struct qtts_spk {
    int mel_dim, enc_dim, C, Cl, att_ch, scale, se_ch;
    qs_conv b0, mfa, asp_tdnn, asp_conv, fc;
    qs_block blk[3];
    int n_blk;
    st_context *st;
};

typedef struct { const float *w; int cin, k; } qs__wctx;
static float qs__get(const void *w, int n, int kk, const void *ctx) {
    const qs__wctx *c = (const qs__wctx *)ctx; (void)w;
    int j = kk / c->cin, ci = kk % c->cin;
    return c->w[((size_t)n * c->cin + ci) * c->k + j];
}

static const float *qs__f32(qtts_spk *s, const char *name, size_t n) {
    int i = safetensors_find(s->st, name);
    if (i < 0) { fprintf(stderr, "qtts_spk: missing %s\n", name); exit(1); }
    const char *dt = safetensors_dtype(s->st, i);
    if (strcmp(dt, "F32")) { fprintf(stderr, "qtts_spk: %s is %s (F32 expected)\n", name, dt); exit(1); }
    if (safetensors_nbytes(s->st, i) != n * 4) { fprintf(stderr, "qtts_spk: %s size mismatch\n", name); exit(1); }
    return (const float *)safetensors_data(s->st, i);
}

/* weights may be stored as BF16 in the Base checkpoint: convert to a temporary F32 copy */
static float *qs__load(qtts_spk *s, const char *name, size_t n) {
    int i = safetensors_find(s->st, name);
    if (i < 0) { fprintf(stderr, "qtts_spk: missing %s\n", name); exit(1); }
    float *d = (float *)malloc(sizeof(float) * n);
    const char *dt = safetensors_dtype(s->st, i);
    if (!strcmp(dt, "BF16")) {
        if (safetensors_nbytes(s->st, i) != n * 2) { fprintf(stderr, "qtts_spk: %s size mismatch\n", name); exit(1); }
        const uint16_t *p = (const uint16_t *)safetensors_data(s->st, i);
        for (size_t k = 0; k < n; k++) d[k] = qt_bf16_to_f32(p[k]);
    } else {
        memcpy(d, qs__f32(s, name, n), n * 4);
    }
    return d;
}

static void qs__conv_load(qtts_spk *s, qs_conv *c, const char *prefix, int cin, int cout, int k, int dil) {
    char nm[256];
    c->cin = cin; c->cout = cout; c->k = k; c->dil = dil;
    snprintf(nm, sizeof(nm), "%s.weight", prefix);
    float *w = qs__load(s, nm, (size_t)cout * cin * k);
    qs__wctx ctx = { w, cin, k };
    qt_pack_b(&c->w, cout, k * cin, qs__get, w, &ctx);
    free(w);
    snprintf(nm, sizeof(nm), "%s.bias", prefix);
    c->b = qs__load(s, nm, (size_t)cout);
}

/* "same" conv with reflect padding on x[T][cin] (row stride ldx) -> y[T][cout] (row stride ldy) */
static void qs__conv(const qs_conv *c, const float *x, int ldx, int T, float *y, int ldy, int relu) {
    int tot = c->dil * (c->k - 1), padl = tot / 2;
    int K = c->k * c->cin;
    float *col = (float *)malloc(sizeof(float) * (size_t)T * K);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int j = 0; j < c->k; j++) {
            int src = t + j * c->dil - padl;
            if (src < 0) src = -src;
            if (src >= T) src = 2 * (T - 1) - src;
            if (src < 0) src = 0;
            memcpy(col + (size_t)t * K + (size_t)j * c->cin, x + (size_t)src * ldx, sizeof(float) * c->cin);
        }
    float *tmp = (float *)malloc(sizeof(float) * (size_t)T * c->cout);
    qt_sgemm(T, col, K, &c->w, tmp, c->cout, 0);
    for (int t = 0; t < T; t++)
        for (int o = 0; o < c->cout; o++) {
            float v = tmp[(size_t)t * c->cout + o] + c->b[o];
            y[(size_t)t * ldy + o] = relu && v < 0 ? 0.0f : v;
        }
    free(col); free(tmp);
}

qtts_spk *qtts_spk_load(const char *dir) {
    char path[1024];
    qtts_spk *s = (qtts_spk *)calloc(1, sizeof(*s));
    s->mel_dim = 128; s->C = 512; s->Cl = 1536; s->att_ch = 128; s->scale = 8; s->se_ch = 128; s->n_blk = 3;
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    s->st = safetensors_open(path);
    if (!s->st) { free(s); return NULL; }
    int fi = safetensors_find(s->st, "speaker_encoder.fc.weight");
    if (fi < 0) { fprintf(stderr, "qtts_spk: no speaker encoder in %s (Base model required)\n", path); safetensors_close(s->st); free(s); return NULL; }
    s->enc_dim = (int)safetensors_shape(s->st, fi)[0];
    static const int dil[3] = { 2, 3, 4 };
    char nm[256];
    qs__conv_load(s, &s->b0, "speaker_encoder.blocks.0.conv", s->mel_dim, s->C, 5, 1);
    for (int b = 0; b < s->n_blk; b++) {
        qs_block *B = &s->blk[b];
        B->scale = s->scale;
        snprintf(nm, sizeof(nm), "speaker_encoder.blocks.%d.tdnn1.conv", b + 1);
        qs__conv_load(s, &B->tdnn1, nm, s->C, s->C, 1, 1);
        for (int r = 0; r < s->scale - 1; r++) {
            snprintf(nm, sizeof(nm), "speaker_encoder.blocks.%d.res2net_block.blocks.%d.conv", b + 1, r);
            qs__conv_load(s, &B->res[r], nm, s->C / s->scale, s->C / s->scale, 3, dil[b]);
        }
        snprintf(nm, sizeof(nm), "speaker_encoder.blocks.%d.tdnn2.conv", b + 1);
        qs__conv_load(s, &B->tdnn2, nm, s->C, s->C, 1, 1);
        snprintf(nm, sizeof(nm), "speaker_encoder.blocks.%d.se_block.conv1", b + 1);
        qs__conv_load(s, &B->se1, nm, s->C, s->se_ch, 1, 1);
        snprintf(nm, sizeof(nm), "speaker_encoder.blocks.%d.se_block.conv2", b + 1);
        qs__conv_load(s, &B->se2, nm, s->se_ch, s->C, 1, 1);
    }
    qs__conv_load(s, &s->mfa, "speaker_encoder.mfa.conv", s->C * 3, s->Cl, 1, 1);
    qs__conv_load(s, &s->asp_tdnn, "speaker_encoder.asp.tdnn.conv", s->Cl * 3, s->att_ch, 1, 1);
    qs__conv_load(s, &s->asp_conv, "speaker_encoder.asp.conv", s->att_ch, s->Cl, 1, 1);
    qs__conv_load(s, &s->fc, "speaker_encoder.fc", s->Cl * 2, s->enc_dim, 1, 1);
    return s;
}

static void qs__conv_free(qs_conv *c) { qt_packed_free(&c->w); free(c->b); }

void qtts_spk_free(qtts_spk *s) {
    if (!s) return;
    qs__conv_free(&s->b0); qs__conv_free(&s->mfa); qs__conv_free(&s->asp_tdnn); qs__conv_free(&s->asp_conv);
    qs__conv_free(&s->fc);
    for (int b = 0; b < s->n_blk; b++) {
        qs_block *B = &s->blk[b];
        qs__conv_free(&B->tdnn1); qs__conv_free(&B->tdnn2); qs__conv_free(&B->se1); qs__conv_free(&B->se2);
        for (int r = 0; r < B->scale - 1; r++) qs__conv_free(&B->res[r]);
    }
    safetensors_close(s->st);
    free(s);
}

int qtts_spk_dim(const qtts_spk *s) { return s->enc_dim; }

int qtts_spk_embed(qtts_spk *s, const float *wav, int n, float *out, float **mel_out, int *mel_T) {
    int T = 0;
    float *mel = qtts_logmel(wav, n, 1024, 256, s->mel_dim, 24000, 0.0f, 12000.0f, &T);
    if (!mel || T < 2) { free(mel); return -1; }
    int C = s->C, Cl = s->Cl, sc = s->scale, w = C / sc;
    float *h0 = (float *)malloc(sizeof(float) * (size_t)T * C);
    float *cat = (float *)malloc(sizeof(float) * (size_t)T * 3 * C);   /* outputs of the 3 SE-Res2Net blocks */
    float *t1 = (float *)malloc(sizeof(float) * (size_t)T * C);
    float *t2 = (float *)malloc(sizeof(float) * (size_t)T * C);
    qs__conv(&s->b0, mel, s->mel_dim, T, h0, C, 1);
    const float *in = h0;
    int ldin = C;
    for (int b = 0; b < s->n_blk; b++) {
        const qs_block *B = &s->blk[b];
        qs__conv(&B->tdnn1, in, ldin, T, t1, C, 1);
        /* Res2Net: chunk 0 passes through; chunk i = tdnn_i(x_i + out_{i-1}) (i >= 2) */
        for (int t = 0; t < T; t++) memcpy(t2 + (size_t)t * C, t1 + (size_t)t * C, sizeof(float) * w);
        float *acc = (float *)malloc(sizeof(float) * (size_t)T * w);
        for (int i = 1; i < sc; i++) {
            for (int t = 0; t < T; t++)
                for (int c = 0; c < w; c++) {
                    float v = t1[(size_t)t * C + i * w + c];
                    acc[(size_t)t * w + c] = i == 1 ? v : v + t2[(size_t)t * C + (i - 1) * w + c];
                }
            qs__conv(&B->res[i - 1], acc, w, T, t2 + (size_t)i * w, C, 1);
        }
        free(acc);
        float *o = cat + (size_t)b * C;   /* row stride 3C */
        qs__conv(&B->tdnn2, t2, C, T, t1, C, 1);
        /* squeeze-excitation */
        float *m = (float *)calloc((size_t)C, sizeof(float)), *se = (float *)malloc(sizeof(float) * s->se_ch);
        float *g = (float *)malloc(sizeof(float) * C);
        for (int t = 0; t < T; t++) for (int c = 0; c < C; c++) m[c] += t1[(size_t)t * C + c];
        for (int c = 0; c < C; c++) m[c] /= T;
        qs__conv(&B->se1, m, C, 1, se, s->se_ch, 1);
        qs__conv(&B->se2, se, s->se_ch, 1, g, C, 0);
        for (int c = 0; c < C; c++) g[c] = 1.0f / (1.0f + expf(-g[c]));
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++)
                o[(size_t)t * 3 * C + c] = t1[(size_t)t * C + c] * g[c] + in[(size_t)t * ldin + c];
        free(m); free(se); free(g);
        in = o;
        ldin = 3 * C;
    }
    float *hm = (float *)malloc(sizeof(float) * (size_t)T * Cl);
    qs__conv(&s->mfa, cat, 3 * C, T, hm, Cl, 1);
    /* attentive statistics pooling */
    double *mean = (double *)calloc((size_t)Cl, sizeof(double)), *sd = (double *)calloc((size_t)Cl, sizeof(double));
    for (int t = 0; t < T; t++) for (int c = 0; c < Cl; c++) mean[c] += hm[(size_t)t * Cl + c];
    for (int c = 0; c < Cl; c++) mean[c] /= T;
    for (int t = 0; t < T; t++) for (int c = 0; c < Cl; c++) { double d = hm[(size_t)t * Cl + c] - mean[c]; sd[c] += d * d; }
    for (int c = 0; c < Cl; c++) { double v = sd[c] / T; sd[c] = sqrt(v > 1e-12 ? v : 1e-12); }
    float *att_in = (float *)malloc(sizeof(float) * (size_t)T * 3 * Cl);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < Cl; c++) {
            att_in[(size_t)t * 3 * Cl + c] = hm[(size_t)t * Cl + c];
            att_in[(size_t)t * 3 * Cl + Cl + c] = (float)mean[c];
            att_in[(size_t)t * 3 * Cl + 2 * Cl + c] = (float)sd[c];
        }
    float *a1 = (float *)malloc(sizeof(float) * (size_t)T * s->att_ch);
    float *a2 = (float *)malloc(sizeof(float) * (size_t)T * Cl);
    qs__conv(&s->asp_tdnn, att_in, 3 * Cl, T, a1, s->att_ch, 1);
    for (size_t i = 0; i < (size_t)T * s->att_ch; i++) a1[i] = tanhf(a1[i]);
    qs__conv(&s->asp_conv, a1, s->att_ch, T, a2, Cl, 0);
    float *pooled = (float *)malloc(sizeof(float) * 2 * Cl);
    for (int c = 0; c < Cl; c++) {
        double mx = -1e30, z = 0.0, mu = 0.0, var = 0.0;
        for (int t = 0; t < T; t++) if (a2[(size_t)t * Cl + c] > mx) mx = a2[(size_t)t * Cl + c];
        for (int t = 0; t < T; t++) z += exp(a2[(size_t)t * Cl + c] - mx);
        for (int t = 0; t < T; t++) mu += exp(a2[(size_t)t * Cl + c] - mx) / z * hm[(size_t)t * Cl + c];
        for (int t = 0; t < T; t++) { double d = hm[(size_t)t * Cl + c] - mu; var += exp(a2[(size_t)t * Cl + c] - mx) / z * d * d; }
        pooled[c] = (float)mu;
        pooled[Cl + c] = (float)sqrt(var > 1e-12 ? var : 1e-12);
    }
    qs__conv(&s->fc, pooled, 2 * Cl, 1, out, s->enc_dim, 0);
    free(h0); free(cat); free(t1); free(t2); free(hm); free(mean); free(sd); free(att_in); free(a1); free(a2); free(pooled);
    if (mel_out) { *mel_out = mel; *mel_T = T; } else free(mel);
    return 0;
}

#endif /* QTTS_SPK_IMPLEMENTATION */
