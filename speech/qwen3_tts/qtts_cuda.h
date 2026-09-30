/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * qtts_cuda.h - CUDA backend for the Qwen3-TTS runner (driver API + NVRTC via cuew).
 *
 *   qtts_cuda_create   upload talker / code predictor (BF16, fused QKV and gate|up) and
 *                      the codec decoder (F32, conv weights packed for implicit GEMM)
 *   qtts_cuda_backend  qtts_backend hooks for qtts_generate (sampling stays on the host,
 *                      sharing the CPU sampler, so CPU and GPU runs consume the same RNG)
 *   qtts_cuda_decode   codec decode on the GPU with the reference 300/25 chunking
 *
 * Single-TU header: include after qtts_talker.h / qtts_codec.h implementations and define
 * QTTS_CUDA_IMPLEMENTATION (needs their private structs). Link cuda/cuew.c, -ldl.
 */
#ifndef QTTS_CUDA_H
#define QTTS_CUDA_H

#include "qtts_talker.h"
#include "qtts_codec.h"

typedef struct qtts_cuda qtts_cuda;

qtts_cuda   *qtts_cuda_create(qtts_model *m, qtts_codec *c, int device, int verbose);
void         qtts_cuda_free(qtts_cuda *g);
qtts_backend qtts_cuda_backend(qtts_cuda *g);
float       *qtts_cuda_decode(qtts_cuda *g, const int32_t *codes, int T, int *n_out, const char *dump_dir);
const char  *qtts_cuda_device_name(const qtts_cuda *g);

#endif /* QTTS_CUDA_H */

#ifdef QTTS_CUDA_IMPLEMENTATION

#ifdef QTTS_WITH_HIP
#include "../../rdna4/cuda_driver_compat.h"
#else
#include "cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "cuda_runner_common.h"
#endif
#include "qtts_cuda_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    CUdeviceptr qkv, o, gu, down, ln1, ln2, qn, kn;   /* BF16 */
} qg_layer;

typedef struct {
    int n_layers, H, nh, nkv, hd, I, max_ctx, qkv_n;
    float eps, theta;
    qg_layer *L;
    CUdeviceptr norm, kc, vc;
} qg_lm;

typedef struct { CUdeviceptr w, b; int cin, cout, k, dil, st, transposed, dw; } qg_conv;
typedef struct { CUdeviceptr a, ib; } qg_snake;

struct qtts_cuda {
    CUcontext ctx;
    CUmodule mod;
    CUstream st;
    CUfunction f_gemv, f_gemm32, f_gemm16, f_rms, f_ln, f_qkr, f_kvs, f_att, f_silu, f_add, f_emb,
               f_gelu, f_snake, f_snake_to, f_dw, f_ola, f_clamp;
    char name[128];
    qtts_model *m;
    qtts_codec *c;
    qg_lm talker, cp;
    CUdeviceptr codec_head, codec_emb, cp_emb[16], cp_head[16], mtp_w, mtp_b;
    /* LM scratch, sized for cap_M rows */
    int cap_M;
    CUdeviceptr x, h, xn, qkv, att, gu, g, out, logits, in2, proj;
    float *h_last;
    /* codec */
    CUdeviceptr c_inproj, c_inproj_b, c_outproj, c_outproj_b, c_norm;
    CUdeviceptr *c_qkv, *c_o, *c_gu, *c_down, *c_ln1, *c_ln2, *c_ls1, *c_ls2;
    qg_conv c_pre, c_upt[4], c_updw[4], c_dec_in, c_final, c_bt[8], c_r1[8][3], c_r2[8][3];
    CUdeviceptr c_up_lnw[4], c_up_lnb[4], c_up_gamma[4], c_pw1[4], c_pw1b[4], c_pw2[4], c_pw2b[4];
    qg_snake c_bsn[8], c_ra1[8][3], c_ra2[8][3], c_fsn;
};

/* ---- helpers ---- */

static CUdeviceptr qg__up(const void *p, size_t bytes) {
    CUdeviceptr d = 0;
    if (cuMemAlloc(&d, bytes ? bytes : 4) != CUDA_SUCCESS) { fprintf(stderr, "qtts_cuda: alloc %zu failed\n", bytes); exit(1); }
    if (bytes) cuMemcpyHtoD(d, p, bytes);
    return d;
}
static CUdeviceptr qg__alloc(size_t bytes) {
    CUdeviceptr d = 0;
    if (cuMemAlloc(&d, bytes ? bytes : 4) != CUDA_SUCCESS) { fprintf(stderr, "qtts_cuda: alloc %zu failed\n", bytes); exit(1); }
    return d;
}

#define QG_LAUNCH(fn, gx, gy, bx, ...) do { \
    void *args_[] = { __VA_ARGS__ }; \
    CUresult r_ = cuLaunchKernel((fn), (gx), (gy), 1, (bx), 1, 1, 0, g->st, args_, NULL); \
    if (r_ != CUDA_SUCCESS) { fprintf(stderr, "qtts_cuda: launch failed (%d) at %s:%d\n", (int)r_, __FILE__, __LINE__); return -1; } \
} while (0)

static unsigned qg__blocks(size_t n, unsigned b) { return (unsigned)((n + b - 1) / b); }

/* Y[M][N] (+)= X[M][K] W[N][K]^T + bias: GEMV for small M, tiled GEMM otherwise (BF16 W) */
static int qg__linear16(qtts_cuda *g, CUdeviceptr W, CUdeviceptr bias, CUdeviceptr X, CUdeviceptr Y,
                        int M, int N, int K, int accum) {
    if (M <= 4) {
        QG_LAUNCH(g->f_gemv, qg__blocks((size_t)N * 32, 256), 1, 256, &W, &X, &Y, &bias, &N, &K, &M, &accum);
    } else {
        int ldc = N;
        void *args[] = { &X, &M, &K, &W, &N, &bias, &Y, &ldc, &accum };
        if (cuLaunchKernel(g->f_gemm16, (unsigned)((N + 63) / 64), (unsigned)((M + 63) / 64), 1, 256, 1, 1, 0,
                           g->st, args, NULL) != CUDA_SUCCESS) return -1;
    }
    return 0;
}

/* F32-weight implicit-conv GEMM: C[T][N] (+)= conv(X[T][Cin]) */
static int qg__gemm32(qtts_cuda *g, CUdeviceptr X, int T, int Cin, int ks, int dil, int pad,
                      CUdeviceptr W, int N, CUdeviceptr bias, CUdeviceptr C, int ldc, int accum) {
    void *args[] = { &X, &T, &Cin, &ks, &dil, &pad, &W, &N, &bias, &C, &ldc, &accum };
    CUresult r = cuLaunchKernel(g->f_gemm32, (unsigned)((N + 63) / 64), (unsigned)((T + 63) / 64), 1, 256, 1, 1, 0,
                                g->st, args, NULL);
    if (r != CUDA_SUCCESS) { fprintf(stderr, "qtts_cuda: gemm32 launch failed %d\n", (int)r); return -1; }
    return 0;
}

/* ---- LM upload / forward ---- */

static CUdeviceptr qg__cat_bf16(const uint16_t *a, size_t na, const uint16_t *b, size_t nb,
                                const uint16_t *c, size_t nc) {
    CUdeviceptr d = qg__alloc((na + nb + nc) * 2);
    cuMemcpyHtoD(d, a, na * 2);
    cuMemcpyHtoD(d + na * 2, b, nb * 2);
    if (c) cuMemcpyHtoD(d + (na + nb) * 2, c, nc * 2);
    return d;
}

static void qg__upload_lm(qg_lm *d, const qtl_lm *s) {
    d->n_layers = s->n_layers; d->H = s->hidden; d->nh = s->n_heads; d->nkv = s->n_kv;
    d->hd = s->head_dim; d->I = s->inter; d->max_ctx = s->max_ctx; d->eps = s->eps; d->theta = s->theta;
    int H = d->H, qd = d->nh * d->hd, kd = d->nkv * d->hd, I = d->I;
    d->qkv_n = qd + 2 * kd;
    d->L = (qg_layer *)calloc((size_t)d->n_layers, sizeof(qg_layer));
    for (int l = 0; l < d->n_layers; l++) {
        const qtl_layer *L = &s->layers[l];
        qg_layer *G = &d->L[l];
        G->qkv = qg__cat_bf16(L->q, (size_t)qd * H, L->k, (size_t)kd * H, L->v, (size_t)kd * H);
        G->o = qg__up(L->o, (size_t)H * qd * 2);
        G->gu = qg__cat_bf16(L->gate, (size_t)I * H, L->up, (size_t)I * H, NULL, 0);
        G->down = qg__up(L->down, (size_t)H * I * 2);
        G->ln1 = qg__up(L->ln1, (size_t)H * 2);
        G->ln2 = qg__up(L->ln2, (size_t)H * 2);
        G->qn = qg__up(L->qn, (size_t)d->hd * 2);
        G->kn = qg__up(L->kn, (size_t)d->hd * 2);
    }
    d->norm = qg__up(s->norm, (size_t)H * 2);
    size_t kv = (size_t)d->n_layers * d->max_ctx * kd * 4;
    d->kc = qg__alloc(kv);
    d->vc = qg__alloc(kv);
}

static int qg__ensure(qtts_cuda *g, int M) {
    if (M <= g->cap_M) return 0;
    CU_FREE(g->x); CU_FREE(g->h); CU_FREE(g->xn); CU_FREE(g->qkv); CU_FREE(g->att);
    CU_FREE(g->gu); CU_FREE(g->g); CU_FREE(g->out);
    int cap = M < 64 ? 64 : M;
    const qg_lm *t = &g->talker;
    size_t H = (size_t)(t->H > g->cp.H ? t->H : g->cp.H);
    size_t QKV = (size_t)(t->qkv_n > g->cp.qkv_n ? t->qkv_n : g->cp.qkv_n);
    size_t I = (size_t)(t->I > g->cp.I ? t->I : g->cp.I);
    size_t QD = (size_t)t->nh * t->hd > (size_t)g->cp.nh * g->cp.hd ? (size_t)t->nh * t->hd : (size_t)g->cp.nh * g->cp.hd;
    g->x = qg__alloc(cap * H * 4);
    g->h = qg__alloc(cap * H * 4);
    g->xn = qg__alloc(cap * H * 4);
    g->qkv = qg__alloc(cap * QKV * 4);
    g->att = qg__alloc(cap * QD * 4);
    g->gu = qg__alloc(cap * 2 * I * 4);
    g->g = qg__alloc(cap * I * 4);
    g->out = qg__alloc(cap * H * 4);
    g->cap_M = cap;
    return 0;
}

/* x (device, [M][H]) -> out (device, [M][H], final-normed). Updates the KV cache at pos0. */
static int qg__forward(qtts_cuda *g, qg_lm *lm, CUdeviceptr x, int M, int pos0, CUdeviceptr out) {
    int H = lm->H, nh = lm->nh, nkv = lm->nkv, hd = lm->hd, I = lm->I, N = lm->qkv_n;
    int qd = nh * hd, kd = nkv * hd, vo = qd + kd, zero = 0, one = 1, win = 0;
    float scale = 1.0f / sqrtf((float)hd), eps = lm->eps, theta = lm->theta;
    CUdeviceptr nul = 0;
    if (pos0 + M > lm->max_ctx) { fprintf(stderr, "qtts_cuda: context overflow\n"); return -1; }
    cuMemcpyDtoDAsync(g->h, x, (size_t)M * H * 4, g->st);
    for (int l = 0; l < lm->n_layers; l++) {
        qg_layer *L = &lm->L[l];
        CUdeviceptr kc = lm->kc + (size_t)l * lm->max_ctx * kd * 4, vc = lm->vc + (size_t)l * lm->max_ctx * kd * 4;
        QG_LAUNCH(g->f_rms, (unsigned)M, 1, 256, &g->h, &g->xn, &L->ln1, &nul, &H, &eps);
        if (qg__linear16(g, L->qkv, 0, g->xn, g->qkv, M, N, H, 0)) return -1;
        QG_LAUNCH(g->f_qkr, (unsigned)M, (unsigned)(nh + nkv), 32, &g->qkv, &N, &qd, &nh, &nkv, &hd, &L->qn, &L->kn,
                  &eps, &pos0, &theta);
        QG_LAUNCH(g->f_kvs, qg__blocks((size_t)M * kd, 256), 1, 256, &g->qkv, &N, &qd, &vo, &kd, &kc, &vc, &pos0, &M);
        QG_LAUNCH(g->f_att, (unsigned)M, (unsigned)nh, 128, &g->qkv, &N, &kc, &vc, &g->att, &pos0, &nh, &nkv, &hd, &win, &scale);
        if (qg__linear16(g, L->o, 0, g->att, g->h, M, H, qd, 1)) return -1;
        QG_LAUNCH(g->f_rms, (unsigned)M, 1, 256, &g->h, &g->xn, &L->ln2, &nul, &H, &eps);
        if (qg__linear16(g, L->gu, 0, g->xn, g->gu, M, 2 * I, H, 0)) return -1;
        QG_LAUNCH(g->f_silu, qg__blocks((size_t)M * I, 256), 1, 256, &g->gu, &g->g, &I, &M);
        if (qg__linear16(g, L->down, 0, g->g, g->h, M, H, I, 1)) return -1;
    }
    QG_LAUNCH(g->f_rms, (unsigned)M, 1, 256, &g->h, &out, &lm->norm, &nul, &H, &eps);
    (void)zero; (void)one;
    return 0;
}

/* ---- backend hooks ---- */

static int qg__talker(void *ctx, const float *x, int M, int pos0, float *last) {
    qtts_cuda *g = (qtts_cuda *)ctx;
    cuCtxSetCurrent(g->ctx);
    if (qg__ensure(g, M)) return -1;
    int H = g->talker.H;
    cuMemcpyHtoDAsync(g->x, x, (size_t)M * H * 4, g->st);
    if (qg__forward(g, &g->talker, g->x, M, pos0, g->out)) return -1;
    /* keep the last row as the current hidden */
    cuMemcpyDtoDAsync(g->in2, g->out + (size_t)(M - 1) * H * 4, (size_t)H * 4, g->st);
    cuMemcpyDtoHAsync(g->h_last, g->in2, (size_t)H * 4, g->st);
    if (cuStreamSynchronize(g->st) != CUDA_SUCCESS) return -1;
    if (last) memcpy(last, g->h_last, sizeof(float) * H);
    return 0;
}

static int qg__head(void *ctx, float *logits) {
    qtts_cuda *g = (qtts_cuda *)ctx;
    int V = g->m->codec_vocab, H = g->talker.H;
    if (qg__linear16(g, g->codec_head, 0, g->in2, g->logits, 1, V, H, 0)) return -1;
    cuMemcpyDtoHAsync(logits, g->logits, (size_t)V * 4, g->st);
    return cuStreamSynchronize(g->st) == CUDA_SUCCESS ? 0 : -1;
}

static int qg__predict(void *ctx, int32_t *codes, qtts_sample_fn sample, void *ud) {
    qtts_cuda *g = (qtts_cuda *)ctx;
    qtts_model *m = g->m;
    int H = g->talker.H, CH = g->cp.H, V = m->cp_vocab, zero = 0;
    float *lg = (float *)malloc(sizeof(float) * (size_t)V);
    CUdeviceptr row1 = g->in2 + (size_t)H * 4;   /* in2 = [talker hidden | embedding] */
    for (int gi = 0; gi < m->num_groups - 1; gi++) {
        int M = gi == 0 ? 2 : 1, pos0 = gi == 0 ? 0 : gi + 1;
        CUdeviceptr table = gi == 0 ? g->codec_emb : g->cp_emb[gi - 1];
        int row = codes[gi];
        QG_LAUNCH(g->f_emb, qg__blocks((size_t)H, 256), 1, 256, &table, &row, &row1, &H, &zero);
        CUdeviceptr src = gi == 0 ? g->in2 : row1;
        if (g->mtp_w) {
            if (qg__linear16(g, g->mtp_w, g->mtp_b, src, g->proj, M, CH, H, 0)) { free(lg); return -1; }
        } else {
            cuMemcpyDtoDAsync(g->proj, src, (size_t)M * CH * 4, g->st);
        }
        if (qg__forward(g, &g->cp, g->proj, M, pos0, g->out)) { free(lg); return -1; }
        CUdeviceptr lastrow = g->out + (size_t)(M - 1) * CH * 4;
        if (qg__linear16(g, g->cp_head[gi], 0, lastrow, g->logits, 1, V, CH, 0)) { free(lg); return -1; }
        cuMemcpyDtoHAsync(lg, g->logits, (size_t)V * 4, g->st);
        if (cuStreamSynchronize(g->st) != CUDA_SUCCESS) { free(lg); return -1; }
        codes[gi + 1] = sample(ud, lg, V);
    }
    free(lg);
    return 0;
}

qtts_backend qtts_cuda_backend(qtts_cuda *g) {
    qtts_backend b = { g, qg__talker, qg__head, qg__predict };
    return b;
}

/* ---- codec upload ---- */

static float *qg__f32(qtts_codec *c, const char *name, size_t n) {
    return (float *)qc__f32(c, name, (int)n);
}

static void qg__conv(qtts_codec *c, qg_conv *d, const char *prefix, int cin, int cout, int k, int dil,
                     int dw, int transposed, int st) {
    char nm[600];
    memset(d, 0, sizeof(*d));
    d->cin = cin; d->cout = cout; d->k = k; d->dil = dil; d->dw = dw; d->transposed = transposed; d->st = st;
    snprintf(nm, sizeof(nm), "%s.weight", prefix);
    if (dw) {
        d->w = qg__up(qg__f32(c, nm, (size_t)cin * k), (size_t)cin * k * 4);
    } else if (transposed) {
        const float *w = qg__f32(c, nm, (size_t)cin * cout * k);  /* [cin][cout][k] -> [k*cout][cin] */
        float *r = (float *)malloc(sizeof(float) * (size_t)cin * cout * k);
        for (int ci = 0; ci < cin; ci++)
            for (int co = 0; co < cout; co++)
                for (int j = 0; j < k; j++) r[((size_t)j * cout + co) * cin + ci] = w[((size_t)ci * cout + co) * k + j];
        d->w = qg__up(r, (size_t)cin * cout * k * 4);
        free(r);
    } else {
        const float *w = qg__f32(c, nm, (size_t)cout * cin * k);  /* [cout][cin][k] -> [cout][k*cin] */
        float *r = (float *)malloc(sizeof(float) * (size_t)cin * cout * k);
        for (int co = 0; co < cout; co++)
            for (int ci = 0; ci < cin; ci++)
                for (int j = 0; j < k; j++) r[(size_t)co * k * cin + (size_t)j * cin + ci] = w[((size_t)co * cin + ci) * k + j];
        d->w = qg__up(r, (size_t)cin * cout * k * 4);
        free(r);
    }
    snprintf(nm, sizeof(nm), "%s.bias", prefix);
    if (safetensors_find(c->st, nm) >= 0) d->b = qg__up(qg__f32(c, nm, (size_t)cout), (size_t)cout * 4);
}

static void qg__snake_up(qg_snake *d, const qc_snake *s) {
    d->a = qg__up(s->alpha, (size_t)s->c * 4);  /* already exp(alpha) */
    d->ib = qg__up(s->beta, (size_t)s->c * 4);  /* already 1/(exp(beta)+eps) */
}

static CUdeviceptr qg__vec(qtts_codec *c, const char *name, size_t n) {
    return qg__up(qg__f32(c, name, n), n * 4);
}

static void qg__upload_codec(qtts_cuda *g, qtts_codec *c) {
    char nm[512];
    int H = c->hidden, qd = c->n_heads * c->head_dim, kd = c->n_kv * c->head_dim, I = c->inter;
    g->c_inproj = qg__vec(c, "decoder.pre_transformer.input_proj.weight", (size_t)H * c->latent);
    g->c_inproj_b = qg__vec(c, "decoder.pre_transformer.input_proj.bias", (size_t)H);
    g->c_outproj = qg__vec(c, "decoder.pre_transformer.output_proj.weight", (size_t)c->latent * H);
    g->c_outproj_b = qg__vec(c, "decoder.pre_transformer.output_proj.bias", (size_t)c->latent);
    g->c_norm = qg__vec(c, "decoder.pre_transformer.norm.weight", (size_t)H);
    int nl = c->n_layers;
    g->c_qkv = calloc((size_t)nl, sizeof(CUdeviceptr)); g->c_o = calloc((size_t)nl, sizeof(CUdeviceptr));
    g->c_gu = calloc((size_t)nl, sizeof(CUdeviceptr)); g->c_down = calloc((size_t)nl, sizeof(CUdeviceptr));
    g->c_ln1 = calloc((size_t)nl, sizeof(CUdeviceptr)); g->c_ln2 = calloc((size_t)nl, sizeof(CUdeviceptr));
    g->c_ls1 = calloc((size_t)nl, sizeof(CUdeviceptr)); g->c_ls2 = calloc((size_t)nl, sizeof(CUdeviceptr));
    for (int l = 0; l < nl; l++) {
#define QGP(s) (snprintf(nm, sizeof(nm), "decoder.pre_transformer.layers.%d.%s", l, s), nm)
        size_t nq = (size_t)qd * H, nk = (size_t)kd * H;
        float *cat = (float *)malloc(sizeof(float) * (nq + 2 * nk));
        memcpy(cat, qg__f32(c, QGP("self_attn.q_proj.weight"), nq), nq * 4);
        memcpy(cat + nq, qg__f32(c, QGP("self_attn.k_proj.weight"), nk), nk * 4);
        memcpy(cat + nq + nk, qg__f32(c, QGP("self_attn.v_proj.weight"), nk), nk * 4);
        g->c_qkv[l] = qg__up(cat, (nq + 2 * nk) * 4);
        free(cat);
        cat = (float *)malloc(sizeof(float) * 2 * (size_t)I * H);
        memcpy(cat, qg__f32(c, QGP("mlp.gate_proj.weight"), (size_t)I * H), (size_t)I * H * 4);
        memcpy(cat + (size_t)I * H, qg__f32(c, QGP("mlp.up_proj.weight"), (size_t)I * H), (size_t)I * H * 4);
        g->c_gu[l] = qg__up(cat, 2 * (size_t)I * H * 4);
        free(cat);
        g->c_o[l] = qg__vec(c, QGP("self_attn.o_proj.weight"), (size_t)H * qd);
        g->c_down[l] = qg__vec(c, QGP("mlp.down_proj.weight"), (size_t)H * I);
        g->c_ln1[l] = qg__vec(c, QGP("input_layernorm.weight"), (size_t)H);
        g->c_ln2[l] = qg__vec(c, QGP("post_attention_layernorm.weight"), (size_t)H);
        g->c_ls1[l] = qg__vec(c, QGP("self_attn_layer_scale.scale"), (size_t)H);
        g->c_ls2[l] = qg__vec(c, QGP("mlp_layer_scale.scale"), (size_t)H);
#undef QGP
    }
    qg__conv(c, &g->c_pre, "decoder.pre_conv.conv", c->cb_dim, c->latent, 3, 1, 0, 0, 1);
    for (int u = 0; u < c->n_up_ratio; u++) {
        int f = c->up_ratios[u], d = c->latent;
        snprintf(nm, sizeof(nm), "decoder.upsample.%d.0.conv", u);
        qg__conv(c, &g->c_upt[u], nm, d, d, f, 1, 0, 1, f);
        snprintf(nm, sizeof(nm), "decoder.upsample.%d.1.dwconv.conv", u);
        qg__conv(c, &g->c_updw[u], nm, d, d, 7, 1, 1, 0, 1);
#define QGU(s) (snprintf(nm, sizeof(nm), "decoder.upsample.%d.1.%s", u, s), nm)
        g->c_up_lnw[u] = qg__vec(c, QGU("norm.weight"), (size_t)d);
        g->c_up_lnb[u] = qg__vec(c, QGU("norm.bias"), (size_t)d);
        g->c_up_gamma[u] = qg__vec(c, QGU("gamma"), (size_t)d);
        g->c_pw1[u] = qg__vec(c, QGU("pwconv1.weight"), (size_t)4 * d * d);
        g->c_pw1b[u] = qg__vec(c, QGU("pwconv1.bias"), (size_t)4 * d);
        g->c_pw2[u] = qg__vec(c, QGU("pwconv2.weight"), (size_t)4 * d * d);
        g->c_pw2b[u] = qg__vec(c, QGU("pwconv2.bias"), (size_t)d);
#undef QGU
    }
    qg__conv(c, &g->c_dec_in, "decoder.decoder.0.conv", c->latent, c->dec_dim, 7, 1, 0, 0, 1);
    static const int dils[3] = { 1, 3, 9 };
    for (int b = 0; b < c->n_rates; b++) {
        int ind = c->dec_dim >> b, outd = c->dec_dim >> (b + 1), r = c->rates[b];
        qg__snake_up(&g->c_bsn[b], &c->blocks[b].snake);
        snprintf(nm, sizeof(nm), "decoder.decoder.%d.block.1.conv", b + 1);
        qg__conv(c, &g->c_bt[b], nm, ind, outd, 2 * r, 1, 0, 1, r);
        for (int ru = 0; ru < 3; ru++) {
            qg__snake_up(&g->c_ra1[b][ru], &c->blocks[b].res[ru].act1);
            qg__snake_up(&g->c_ra2[b][ru], &c->blocks[b].res[ru].act2);
            snprintf(nm, sizeof(nm), "decoder.decoder.%d.block.%d.conv1.conv", b + 1, ru + 2);
            qg__conv(c, &g->c_r1[b][ru], nm, outd, outd, 7, dils[ru], 0, 0, 1);
            snprintf(nm, sizeof(nm), "decoder.decoder.%d.block.%d.conv2.conv", b + 1, ru + 2);
            qg__conv(c, &g->c_r2[b][ru], nm, outd, outd, 1, 1, 0, 0, 1);
        }
    }
    qg__snake_up(&g->c_fsn, &c->final_snake);
    snprintf(nm, sizeof(nm), "decoder.decoder.%d.conv", c->n_rates + 2);
    qg__conv(c, &g->c_final, nm, c->dec_dim >> c->n_rates, 1, 7, 1, 0, 0, 1);
}

/* ---- create / free ---- */

qtts_cuda *qtts_cuda_create(qtts_model *m, qtts_codec *c, int device, int verbose) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) { fprintf(stderr, "qtts_cuda: cuew init failed\n"); return NULL; }
    if (cuInit(0) != CUDA_SUCCESS) return NULL;
    qtts_cuda *g = (qtts_cuda *)calloc(1, sizeof(*g));
    CUdevice dev;
    if (cuDeviceGet(&dev, device) != CUDA_SUCCESS) { free(g); return NULL; }
    cuDeviceGetName(g->name, sizeof(g->name), dev);
    if (cuCtxCreate(&g->ctx, 0, dev) != CUDA_SUCCESS) { free(g); return NULL; }
    if (cu_compile_kernels_ex(&g->mod, dev, qtts_cuda_kernel_src, "qtts_cuda", verbose, "qtts_cuda", 0) < 0) {
        free(g); return NULL;
    }
#define QG_FN(f, n) if (cuModuleGetFunction(&g->f, g->mod, n) != CUDA_SUCCESS) { fprintf(stderr, "qtts_cuda: no kernel %s\n", n); return NULL; }
    QG_FN(f_gemv, "gemv_bf16"); QG_FN(f_gemm32, "gemm_f32w"); QG_FN(f_gemm16, "gemm_bf16w");
    QG_FN(f_rms, "rmsnorm"); QG_FN(f_ln, "layernorm"); QG_FN(f_qkr, "qk_norm_rope"); QG_FN(f_kvs, "kv_store");
    QG_FN(f_att, "attention"); QG_FN(f_silu, "silu_mul"); QG_FN(f_add, "add_scaled"); QG_FN(f_emb, "embed_add");
    QG_FN(f_gelu, "gelu_bias"); QG_FN(f_snake, "snake"); QG_FN(f_snake_to, "snake_to"); QG_FN(f_dw, "dwconv");
    QG_FN(f_ola, "convt_ola"); QG_FN(f_clamp, "clamp1");
#undef QG_FN
    cuStreamCreate(&g->st, CU_STREAM_NON_BLOCKING);
    g->m = m;
    g->c = c;
    if (m) {
        qg__upload_lm(&g->talker, &m->talker);
        qg__upload_lm(&g->cp, &m->cp);
        int H = m->talker.hidden, CH = m->cp.hidden;
        g->codec_head = qg__up(m->codec_head, (size_t)m->codec_vocab * H * 2);
        g->codec_emb = qg__up(m->codec_emb, (size_t)m->codec_vocab * H * 2);
        for (int i = 0; i < m->num_groups - 1; i++) {
            g->cp_emb[i] = qg__up(m->cp_emb[i], (size_t)m->cp_vocab * H * 2);
            g->cp_head[i] = qg__up(m->cp_head[i], (size_t)m->cp_vocab * CH * 2);
        }
        if (m->mtp_w) {
            g->mtp_w = qg__up(m->mtp_w, (size_t)CH * H * 2);
            g->mtp_b = qg__up(m->mtp_b, (size_t)CH * 2);
        }
        g->logits = qg__alloc((size_t)(m->codec_vocab > m->cp_vocab ? m->codec_vocab : m->cp_vocab) * 4);
        g->in2 = qg__alloc((size_t)2 * H * 4);
        g->proj = qg__alloc((size_t)2 * (H > CH ? H : CH) * 4);
        cuMemAllocHost((void **)&g->h_last, (size_t)H * 4);
        qg__ensure(g, 64);
    }
    if (c) qg__upload_codec(g, c);
    if (verbose) {
        size_t fr = 0, tot = 0;
        cuMemGetInfo(&fr, &tot);
        fprintf(stderr, "qtts_cuda: %s, %.2f GB used by the process context, %.2f GB free\n", g->name,
                (tot - fr) / 1e9, fr / 1e9);
    }
    return g;
}

const char *qtts_cuda_device_name(const qtts_cuda *g) { (void)cu_compile_kernels; return g->name; }

void qtts_cuda_free(qtts_cuda *g) {
    if (!g) return;
    /* destroying the context releases every device allocation */
    if (g->h_last) cuMemFreeHost(g->h_last);
    free(g->talker.L); free(g->cp.L);
    free(g->c_qkv); free(g->c_o); free(g->c_gu); free(g->c_down); free(g->c_ln1); free(g->c_ln2);
    free(g->c_ls1); free(g->c_ls2);
    if (g->st) cuStreamDestroy(g->st);
    if (g->mod) cuModuleUnload(g->mod);
    if (g->ctx) cuCtxDestroy(g->ctx);
    free(g);
}

/* ---- codec decode ---- */

static void qg__dump(qtts_cuda *g, const char *dir, const char *name, CUdeviceptr x, int T, int C, int chan_first) {
    if (!dir) return;
    size_t n = (size_t)T * C;
    float *h = (float *)malloc(sizeof(float) * n);
    cuStreamSynchronize(g->st);
    cuMemcpyDtoH(h, x, n * 4);
    char p[1024];
    snprintf(p, sizeof(p), "%s/%s.npy", dir, name);
    if (chan_first) {
        float *tr = (float *)malloc(sizeof(float) * n);
        for (int t = 0; t < T; t++) for (int c = 0; c < C; c++) tr[(size_t)c * T + t] = h[(size_t)t * C + c];
        int dims[2] = { C, T };
        qt_npy_save_f32(p, tr, 2, dims);
        free(tr);
    } else {
        int dims[2] = { T, C };
        qt_npy_save_f32(p, h, 2, dims);
    }
    free(h);
}

/* conv (dense, causal) with optional accumulate */
static int qg__conv_run(qtts_cuda *g, const qg_conv *cv, CUdeviceptr x, int T, CUdeviceptr y, int accum) {
    return qg__gemm32(g, x, T, cv->cin, cv->k, cv->dil, (cv->k - 1) * cv->dil, cv->w, cv->cout, cv->b, y, cv->cout, accum);
}

static int qg__decode_chunk(qtts_cuda *g, const int32_t *codes, int T, float *wav, const char *dump) {
    qtts_codec *c = g->c;
    int D = c->cb_dim, Lt = c->latent, H = c->hidden, I = c->inter;
    int qd = c->n_heads * c->head_dim, kd = c->n_kv * c->head_dim, N = qd + 2 * kd, vo = qd + kd;
    /* scratch sizes: largest [rows x channels] activation and transposed-conv product */
    size_t big = (size_t)T * c->latent, zbig = 0, rows = (size_t)T;
    for (int u = 0; u < c->n_up_ratio; u++) {
        zbig = zbig > rows * (size_t)c->up_ratios[u] * Lt ? zbig : rows * (size_t)c->up_ratios[u] * Lt;
        rows *= (size_t)c->up_ratios[u];
        big = big > rows * Lt ? big : rows * Lt;
    }
    big = big > rows * (size_t)c->dec_dim ? big : rows * (size_t)c->dec_dim;
    for (int b = 0; b < c->n_rates; b++) {
        size_t co = (size_t)(c->dec_dim >> (b + 1));
        zbig = zbig > rows * 2 * (size_t)c->rates[b] * co ? zbig : rows * 2 * (size_t)c->rates[b] * co;
        rows *= (size_t)c->rates[b];
        big = big > rows * co ? big : rows * co;
    }
    if (zbig < big) zbig = big;
    size_t qn = (size_t)T * N;
    qn = qn > (size_t)T * 2 * I ? qn : (size_t)T * 2 * I;
    rows = (size_t)T;
    for (int u = 0; u < c->n_up_ratio; u++) rows *= (size_t)c->up_ratios[u];
    qn = qn > rows * 4 * Lt ? qn : rows * 4 * Lt;
    CUdeviceptr A = qg__alloc(big * 4), B = qg__alloc(big * 4), Z = qg__alloc(zbig * 4);
    CUdeviceptr q = qg__alloc(qn * 4);
    CUdeviceptr kc = qg__alloc((size_t)T * kd * 4), vc = qg__alloc((size_t)T * kd * 4);
    int rc = -1;
    float *x = (float *)malloc(sizeof(float) * (size_t)T * D);
    qtts_codec_rvq(c, codes, T, x);
    cuMemcpyHtoDAsync(A, x, (size_t)T * D * 4, g->st);
    free(x);
    qg__dump(g, dump, "codec_rvq", A, T, D, 1);
    CUdeviceptr nul = 0, hbuf = B;
    float eps = c->rms_eps, eps6 = 1e-6f, theta = c->rope_theta, scale = 1.0f / sqrtf((float)c->head_dim);
    int zero = 0, hd = c->head_dim, nh = c->n_heads, nkv = c->n_kv, win = c->window;
    size_t n;
#define CK(e) do { if (e) goto done; } while (0)
    CK(qg__conv_run(g, &g->c_pre, A, T, Z, 0));                       /* Z = preconv [T][Lt] */
    qg__dump(g, dump, "codec_preconv", Z, T, Lt, 1);
    /* transformer (hidden 512) */
    CK(qg__gemm32(g, Z, T, Lt, 1, 1, 0, g->c_inproj, H, g->c_inproj_b, hbuf, H, 0));
    for (int l = 0; l < c->n_layers; l++) {
        QG_LAUNCH(g->f_rms, (unsigned)T, 1, 256, &hbuf, &A, &nul, &g->c_ln1[l], &H, &eps);
        CK(qg__gemm32(g, A, T, H, 1, 1, 0, g->c_qkv[l], N, 0, q, N, 0));
        QG_LAUNCH(g->f_qkr, (unsigned)T, (unsigned)(nh + nkv), 32, &q, &N, &qd, &nh, &nkv, &hd, &nul, &nul, &eps, &zero, &theta);
        QG_LAUNCH(g->f_kvs, qg__blocks((size_t)T * kd, 256), 1, 256, &q, &N, &qd, &vo, &kd, &kc, &vc, &zero, &T);
        QG_LAUNCH(g->f_att, (unsigned)T, (unsigned)nh, 128, &q, &N, &kc, &vc, &A, &zero, &nh, &nkv, &hd, &win, &scale);
        CK(qg__gemm32(g, A, T, qd, 1, 1, 0, g->c_o[l], H, 0, Z, H, 0));
        n = (size_t)T * H;
        QG_LAUNCH(g->f_add, qg__blocks(n, 256), 1, 256, &hbuf, &Z, &g->c_ls1[l], &H, &n);
        QG_LAUNCH(g->f_rms, (unsigned)T, 1, 256, &hbuf, &A, &nul, &g->c_ln2[l], &H, &eps);
        CK(qg__gemm32(g, A, T, H, 1, 1, 0, g->c_gu[l], 2 * I, 0, q, 2 * I, 0));
        QG_LAUNCH(g->f_silu, qg__blocks((size_t)T * I, 256), 1, 256, &q, &A, &I, &T);
        CK(qg__gemm32(g, A, T, I, 1, 1, 0, g->c_down[l], H, 0, Z, H, 0));
        QG_LAUNCH(g->f_add, qg__blocks(n, 256), 1, 256, &hbuf, &Z, &g->c_ls2[l], &H, &n);
    }
    QG_LAUNCH(g->f_rms, (unsigned)T, 1, 256, &hbuf, &A, &nul, &g->c_norm, &H, &eps);
    CK(qg__gemm32(g, A, T, H, 1, 1, 0, g->c_outproj, Lt, g->c_outproj_b, Z, Lt, 0));   /* Z = [T][Lt] */
    qg__dump(g, dump, "codec_pretf", Z, T, Lt, 0);
    int Tc = T;
    CUdeviceptr cur = Z, oth = A;
    for (int u = 0; u < c->n_up_ratio; u++) {
        qg_conv *ct = &g->c_upt[u];
        /* convT: q = cur * W' ([Tc][k*Lt]) then overlap-add into oth */
        CK(qg__gemm32(g, cur, Tc, Lt, 1, 1, 0, ct->w, ct->k * Lt, 0, q, ct->k * Lt, 0));
        QG_LAUNCH(g->f_ola, qg__blocks((size_t)Tc * ct->st * Lt, 256), 1, 256, &q, &oth, &ct->b, &Tc, &Lt, &ct->k, &ct->st);
        Tc *= ct->st;
        /* ConvNeXt: cur = dwconv(oth); LN; pw1+gelu; pw2; oth += gamma * (pw2 + b) */
        int k7 = 7;
        QG_LAUNCH(g->f_dw, qg__blocks((size_t)Tc * Lt, 256), 1, 256, &oth, &cur, &g->c_updw[u].w, &g->c_updw[u].b, &Tc, &Lt, &k7);
        QG_LAUNCH(g->f_ln, (unsigned)Tc, 1, 256, &cur, &cur, &g->c_up_lnw[u], &g->c_up_lnb[u], &Lt, &eps6);
        CK(qg__gemm32(g, cur, Tc, Lt, 1, 1, 0, g->c_pw1[u], 4 * Lt, 0, q, 4 * Lt, 0));
        n = (size_t)Tc * 4 * Lt;
        int L4 = 4 * Lt;
        QG_LAUNCH(g->f_gelu, qg__blocks(n, 256), 1, 256, &q, &g->c_pw1b[u], &L4, &n);
        CK(qg__gemm32(g, q, Tc, 4 * Lt, 1, 1, 0, g->c_pw2[u], Lt, g->c_pw2b[u], cur, Lt, 0));
        n = (size_t)Tc * Lt;
        QG_LAUNCH(g->f_add, qg__blocks(n, 256), 1, 256, &oth, &cur, &g->c_up_gamma[u], &Lt, &n);
        CUdeviceptr t2 = cur; cur = oth; oth = t2;
        char nm[32]; snprintf(nm, sizeof(nm), "codec_up%d", u);
        qg__dump(g, dump, nm, cur, Tc, Lt, 1);
    }
    /* decoder: cur [Tc][Lt] -> oth [Tc][dec_dim] */
    if (cur != Z) { cuMemcpyDtoDAsync(Z, cur, (size_t)Tc * Lt * 4, g->st); }
    CK(qg__conv_run(g, &g->c_dec_in, Z, Tc, A, 0));
    int C = c->dec_dim;
    qg__dump(g, dump, "codec_dec0", A, Tc, C, 1);
    CUdeviceptr h = A, t1 = B;
    for (int b = 0; b < c->n_rates; b++) {
        n = (size_t)Tc * C;
        QG_LAUNCH(g->f_snake, qg__blocks(n, 256), 1, 256, &h, &g->c_bsn[b].a, &g->c_bsn[b].ib, &C, &n);
        qg_conv *ct = &g->c_bt[b];
        int Co = ct->cout;
        CK(qg__gemm32(g, h, Tc, C, 1, 1, 0, ct->w, ct->k * Co, 0, Z, ct->k * Co, 0));
        QG_LAUNCH(g->f_ola, qg__blocks((size_t)Tc * ct->st * Co, 256), 1, 256, &Z, &t1, &ct->b, &Tc, &Co, &ct->k, &ct->st);
        Tc *= ct->st;
        C = Co;
        CUdeviceptr tmp = h; h = t1; t1 = tmp;      /* h = block output [Tc][C] */
        for (int ru = 0; ru < 3; ru++) {
            n = (size_t)Tc * C;
            QG_LAUNCH(g->f_snake_to, qg__blocks(n, 256), 1, 256, &h, &t1, &g->c_ra1[b][ru].a, &g->c_ra1[b][ru].ib, &C, &n);
            CK(qg__conv_run(g, &g->c_r1[b][ru], t1, Tc, Z, 0));
            QG_LAUNCH(g->f_snake, qg__blocks(n, 256), 1, 256, &Z, &g->c_ra2[b][ru].a, &g->c_ra2[b][ru].ib, &C, &n);
            CK(qg__conv_run(g, &g->c_r2[b][ru], Z, Tc, h, 1));   /* residual accumulate */
        }
        char nm[32]; snprintf(nm, sizeof(nm), "codec_dec%d", b + 1);
        qg__dump(g, dump, nm, h, Tc, C, 1);
    }
    n = (size_t)Tc * C;
    QG_LAUNCH(g->f_snake, qg__blocks(n, 256), 1, 256, &h, &g->c_fsn.a, &g->c_fsn.ib, &C, &n);
    { char nm[32]; snprintf(nm, sizeof(nm), "codec_dec%d", c->n_rates + 1); qg__dump(g, dump, nm, h, Tc, C, 1); }
    CK(qg__conv_run(g, &g->c_final, h, Tc, Z, 0));
    { char nm[32]; snprintf(nm, sizeof(nm), "codec_dec%d", c->n_rates + 2); qg__dump(g, dump, nm, Z, Tc, 1, 1); }
    n = (size_t)Tc;
    QG_LAUNCH(g->f_clamp, qg__blocks(n, 256), 1, 256, &Z, &n);
    cuMemcpyDtoHAsync(wav, Z, (size_t)Tc * 4, g->st);
    rc = cuStreamSynchronize(g->st) == CUDA_SUCCESS ? 0 : -1;
#undef CK
done:
    cuMemFree(A); cuMemFree(B); cuMemFree(Z); cuMemFree(q); cuMemFree(kc); cuMemFree(vc);
    return rc;
}

float *qtts_cuda_decode(qtts_cuda *g, const int32_t *codes, int T, int *n_out, const char *dump_dir) {
    const int chunk = 300, left_ctx = 25;
    qtts_codec *c = g->c;
    cuCtxSetCurrent(g->ctx);
    float *wav = (float *)malloc(sizeof(float) * ((size_t)T * c->upsample + 1));
    float *tmp = (float *)malloc(sizeof(float) * ((size_t)(chunk + left_ctx) * c->upsample));
    int out = 0;
    for (int start = 0; start < T; ) {
        int end = start + chunk < T ? start + chunk : T;
        int ctx = start - left_ctx > 0 ? left_ctx : start;
        if (qg__decode_chunk(g, codes + (size_t)(start - ctx) * c->nq, end - start + ctx, tmp,
                             start == 0 ? dump_dir : NULL)) {
            free(wav); free(tmp); *n_out = 0; return NULL;
        }
        int keep = (end - start) * c->upsample;
        memcpy(wav + out, tmp + (size_t)ctx * c->upsample, sizeof(float) * (size_t)keep);
        out += keep;
        start = end;
    }
    free(tmp);
    *n_out = out;
    return wav;
}

#endif /* QTTS_CUDA_IMPLEMENTATION */
