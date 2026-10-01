/* SPDX-License-Identifier: MIT
 * Stateful CPU codec reference. Include after QTTS_CODEC_IMPLEMENTATION.
 * Absolute RoPE, bounded per-layer KV caches and causal convolution histories.
 * Decoder temporaries allocate on the model worker, never on the audio callback.
 * Unlike prefix recomputation, transformer work and memory are duration bounded.
 */
#ifndef QTTS_CODEC_STREAM_H
#define QTTS_CODEC_STREAM_H
#include <limits.h>

typedef struct {
    const qc_conv *conv;
    float *history;
    int rows;
} qcs_conv;
typedef struct {
    const qtts_codec *codec;
    int position;
    float *keys, *values;
    qcs_conv conv[64];
} qtts_codec_stream;

static void qtts_codec_stream_free(qtts_codec_stream *s) {
    if (!s) return;
    free(s->keys); free(s->values);
    for (int i = 0; i < 64; ++i) free(s->conv[i].history);
    free(s);
}
static qtts_codec_stream *qtts_codec_stream_create(const qtts_codec *c) {
    if (!c || c->window <= 0 || c->upsample != 1920 || c->n_up_ratio < 0 || c->n_up_ratio > 4 ||
        c->n_rates < 0 || c->n_rates > 8 || 3 + 2 * c->n_up_ratio + 7 * c->n_rates > 64) return NULL;
    qtts_codec_stream *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->codec = c;
    size_t count = (size_t)c->n_layers * c->window * c->n_kv * c->head_dim;
    s->keys = calloc(count, sizeof(float)); s->values = calloc(count, sizeof(float));
    if (!s->keys || !s->values) { qtts_codec_stream_free(s); return NULL; }
    return s;
}

static float *qcs_conv_forward(qcs_conv *state, const qc_conv *cv, const float *x, int T) {
    int rows = cv->transposed ? (cv->k - 1) / cv->stride : (cv->k - 1) * cv->dil;
    if (!state->conv) {
        state->conv = cv; state->rows = rows;
        state->history = calloc((size_t)(rows ? rows : 1) * cv->cin, sizeof(float));
    }
    if (state->conv != cv || !state->history) return NULL;
    size_t history = (size_t)rows * cv->cin;
    float *joined = malloc((size_t)(rows + T) * cv->cin * sizeof(float));
    if (!joined) return NULL;
    memcpy(joined, state->history, history * sizeof(float));
    memcpy(joined + history, x, (size_t)T * cv->cin * sizeof(float));
    int ratio = cv->transposed ? cv->stride : 1;
    float *out = malloc((size_t)T * ratio * cv->cout * sizeof(float));
    if (!out) { free(joined); return NULL; }
    if (cv->transposed) {
        float *full = qc__convt(cv, joined, rows + T);
        memcpy(out, full + (size_t)rows * ratio * cv->cout, (size_t)T * ratio * cv->cout * sizeof(float));
        free(full);
    } else if (cv->dw) {
        for (int t = 0; t < T; ++t) for (int ch = 0; ch < cv->cin; ++ch) {
            float sum = cv->bias ? cv->bias[ch] : 0;
            for (int j = 0; j < cv->k; ++j) sum += cv->dw[(size_t)ch * cv->k + j] * joined[((size_t)t + j * cv->dil) * cv->cin + ch];
            out[(size_t)t * cv->cout + ch] = sum;
        }
    } else {
        /* Only current outputs are evaluated. Cached history is not convolved again. */
        for (int j = 0; j < cv->k; ++j)
            qt_sgemm(T, joined + (size_t)j * cv->dil * cv->cin, cv->cin, &cv->taps[j], out, cv->cout, j > 0);
        if (cv->bias) for (int t = 0; t < T; ++t) for (int ch = 0; ch < cv->cout; ++ch)
            out[(size_t)t * cv->cout + ch] += cv->bias[ch];
    }
    memcpy(state->history, joined + (size_t)T * cv->cin, history * sizeof(float));
    free(joined);
    return out;
}

static float *qcs_transformer(qtts_codec_stream *s, const float *x) {
    const qtts_codec *c = s->codec;
    int H = c->hidden, Q = c->n_heads * c->head_dim, K = c->n_kv * c->head_dim, I = c->inter;
    float *h = malloc((size_t)H * sizeof(float)), *n = malloc((size_t)H * sizeof(float));
    float *q = malloc((size_t)Q * sizeof(float)), *att = malloc((size_t)Q * sizeof(float));
    float *o = malloc((size_t)H * sizeof(float)), *g = malloc((size_t)I * sizeof(float)), *u = malloc((size_t)I * sizeof(float));
    qt_sgemm(1, x, c->latent, &c->in_proj, h, H, 0); qc__add_bias(h, c->in_proj_b, 1, H);
    int at = s->position < c->window ? s->position : c->window - 1;
    for (int l = 0; l < c->n_layers; ++l) {
        const qc_tf_layer *L = &c->layers[l];
        float *kc = s->keys + (size_t)l * c->window * K, *vc = s->values + (size_t)l * c->window * K;
        if (s->position >= c->window) {
            memmove(kc, kc + K, (size_t)(c->window - 1) * K * sizeof(float));
            memmove(vc, vc + K, (size_t)(c->window - 1) * K * sizeof(float));
        }
        qt_rmsnorm(n, h, L->ln1, H, c->rms_eps);
        qt_sgemm(1, n, H, &L->q, q, Q, 0);
        qt_sgemm(1, n, H, &L->k, kc + (size_t)at * K, K, 0);
        qt_sgemm(1, n, H, &L->v, vc + (size_t)at * K, K, 0);
        for (int head = 0; head < c->n_heads; ++head) qt_rope_neox(q + head * c->head_dim, c->head_dim, s->position, c->rope_theta);
        for (int head = 0; head < c->n_kv; ++head) qt_rope_neox(kc + (size_t)at * K + head * c->head_dim, c->head_dim, s->position, c->rope_theta);
        qt_attention(att, q, kc, vc, 1, at, c->n_heads, c->n_kv, c->head_dim, c->window);
        qt_sgemm(1, att, Q, &L->o, o, H, 0);
        for (int i = 0; i < H; ++i) h[i] += L->ls_attn[i] * o[i];
        qt_rmsnorm(n, h, L->ln2, H, c->rms_eps);
        qt_sgemm(1, n, H, &L->gate, g, I, 0); qt_sgemm(1, n, H, &L->up, u, I, 0);
        for (int i = 0; i < I; ++i) g[i] = qt_silu(g[i]) * u[i];
        qt_sgemm(1, g, I, &L->down, o, H, 0);
        for (int i = 0; i < H; ++i) h[i] += L->ls_mlp[i] * o[i];
    }
    qt_rmsnorm(h, h, c->norm_w, H, c->rms_eps);
    float *out = malloc((size_t)c->latent * sizeof(float));
    qt_sgemm(1, h, H, &c->out_proj, out, c->latent, 0); qc__add_bias(out, c->out_proj_b, 1, c->latent);
    free(h); free(n); free(q); free(att); free(o); free(g); free(u);
    return out;
}

/* Returns exactly 1920 samples; caller frees. Model must outlive the state. */
static float *qtts_codec_stream_push(qtts_codec_stream *s, const int32_t *codes) {
    if (!s || !codes || s->position == INT_MAX) return NULL;
    const qtts_codec *c = s->codec;
    for (int i = 0; i < c->nq; ++i) if (codes[i] < 0 || codes[i] >= c->cb_size) return NULL;
    int index = 0, T = 1;
    float *x = malloc((size_t)c->cb_dim * sizeof(float));
    qtts_codec_rvq(c, codes, 1, x);
    float *h = qcs_conv_forward(&s->conv[index++], &c->pre_conv, x, T); free(x);
    x = qcs_transformer(s, h); free(h); h = x;
    for (int u = 0; u < c->n_up_ratio; ++u) {
        const qc_upblock *U = &c->up[u];
        x = qcs_conv_forward(&s->conv[index++], &U->convt, h, T); free(h); h = x;
        T *= c->up_ratios[u]; int D = c->latent;
        float *y = qcs_conv_forward(&s->conv[index++], &U->dw, h, T);
        float *n = malloc((size_t)T * D * sizeof(float)), *buf = malloc((size_t)T * 4 * D * sizeof(float));
        for (int t = 0; t < T; ++t) qt_layernorm(n + (size_t)t * D, y + (size_t)t * D, U->ln_w, U->ln_b, D, 1e-6f);
        qt_sgemm(T, n, D, &U->pw1, buf, 4 * D, 0);
        for (int t = 0; t < T; ++t) for (int i = 0; i < 4 * D; ++i) {
            size_t p = (size_t)t * 4 * D + i; buf[p] = qt_gelu_erf(buf[p] + U->pw1_b[i]);
        }
        qt_sgemm(T, buf, 4 * D, &U->pw2, y, D, 0);
        for (int t = 0; t < T; ++t) for (int i = 0; i < D; ++i) {
            size_t p = (size_t)t * D + i; h[p] += U->gamma[i] * (y[p] + U->pw2_b[i]);
        }
        free(y); free(n); free(buf);
    }
    x = qcs_conv_forward(&s->conv[index++], &c->dec_in, h, T); free(h); h = x;
    for (int b = 0; b < c->n_rates; ++b) {
        const qc_decblock *B = &c->blocks[b];
        qc__snake(&B->snake, h, T);
        x = qcs_conv_forward(&s->conv[index++], &B->convt, h, T); free(h); h = x;
        T *= c->rates[b]; int C = B->convt.cout;
        for (int r = 0; r < 3; ++r) {
            const qc_resunit *R = &B->res[r];
            size_t count = (size_t)T * C;
            x = malloc(count * sizeof(float)); memcpy(x, h, count * sizeof(float));
            qc__snake(&R->act1, x, T);
            float *y = qcs_conv_forward(&s->conv[index++], &R->conv1, x, T); free(x);
            qc__snake(&R->act2, y, T);
            x = qcs_conv_forward(&s->conv[index++], &R->conv2, y, T); free(y);
            for (size_t i = 0; i < count; ++i) h[i] += x[i];
            free(x);
        }
    }
    qc__snake(&c->final_snake, h, T);
    x = qcs_conv_forward(&s->conv[index++], &c->final_conv, h, T); free(h);
    for (int i = 0; i < T; ++i) x[i] = fmaxf(-1, fminf(1, x[i]));
    s->position++;
    return x;
}
#endif
