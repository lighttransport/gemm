/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * qtts_talker.h - Qwen3-TTS talker (Qwen3 LM over codec codebook 0) and
 * multi-token code predictor (codebooks 1..15), CPU reference path.
 *
 * Follows the Apache-2.0 reference `qwen_tts/core/models/modeling_qwen3_tts.py`:
 *   - talker: Qwen3 decoder (q/k RMSNorm, GQA, SwiGLU). Its interleaved MRoPE
 *     reduces to 1-D RoPE because text/audio positions are equal on all 3 axes.
 *   - input = text_projection(text_embedding(tok)) [+ codec_embedding(code)]
 *   - code predictor: 5-layer Qwen3 fed [talker hidden, codec_emb(code0)], then
 *     cp.codec_embedding[g-1](code_g), all through small_to_mtp_projection;
 *     lm_head[g] predicts codebook g+1.
 *   - prompt construction follows Qwen3TTSForConditionalGeneration.generate()
 *     for the CustomVoice path (speaker / language / optional instruct,
 *     streaming or non-streaming text feed).
 * Weights are BF16 safetensors read in place; activations are F32.
 *
 * Requires safetensors.h, qtts_ops.h, philox_rng.h. Define QTTS_TALKER_IMPLEMENTATION once.
 */
#ifndef QTTS_TALKER_H
#define QTTS_TALKER_H

#include <stdint.h>
#include "qtts_ops.h"

typedef struct qtts_model qtts_model;

/* Voice cloning (Base model). spk_emb alone = x-vector mode; with ref_ids + ref_codes = ICL
 * (in-context) mode: the reference transcript and its codec frames are placed in the prompt and
 * generation continues the reference speech. */
typedef struct {
    const float *spk_emb;          /* [hidden] speaker embedding (qtts_spk_embed), required */
    const int32_t *ref_ids;        /* tokenized "<|im_start|>assistant\n{ref}<|im_end|>\n" (ICL) */
    int n_ref_ids;
    const int32_t *ref_codes;      /* [n_ref][16] reference codes (qtts_cenc_encode) (ICL) */
    int n_ref;
} qtts_voice_clone;

typedef struct {
    int greedy;               /* 1: argmax (validation), 0: sample */
    float temperature, top_p; /* talker */
    int top_k;
    float rep_penalty;
    float sub_temperature, sub_top_p; /* code predictor */
    int sub_top_k;
    int max_frames;
    uint64_t seed;
    int streaming;            /* 0: non-streaming text feed (CustomVoice default) */
    const qtts_voice_clone *clone; /* NULL unless cloning a voice with the Base model */
} qtts_gen_params;

typedef struct {
    /* codes [n_frames][16] */
    int32_t *codes;
    int n_frames;
    /* optional validation captures (NULL unless requested) */
    float *prefill_embeds; int prefill_len;
    float *step_logits;    /* [n_steps][vocab] */
    float *step_hidden;    /* [n_steps][hidden] */
    int n_steps;
} qtts_gen_result;

/* Compute backend for the generation loop. Prompt construction, sampling and the
 * frame loop are shared; a backend only runs the networks.
 *   talker : run the talker on x[M][H] at positions pos0.., write the final (normed)
 *            hidden of the last row to last[H] and keep it for head/predict
 *   head   : codebook-0 logits [codec_vocab] from the kept hidden
 *   predict: given codes[0], fill codes[1..G-1]; sample(ud, logits, n) picks each code */
typedef int (*qtts_sample_fn)(void *ud, float *logits, int n);
typedef struct {
    void *ctx;
    int (*talker)(void *ctx, const float *x, int M, int pos0, float *last);
    int (*head)(void *ctx, float *logits);
    int (*predict)(void *ctx, int32_t *codes, qtts_sample_fn sample, void *ud);
} qtts_backend;

qtts_model *qtts_model_load(const char *model_dir, int max_ctx);
void        qtts_model_free(qtts_model *m);
void        qtts_gen_params_default(qtts_gen_params *p);
int         qtts_model_hidden(const qtts_model *m);
int         qtts_model_codec_vocab(const qtts_model *m);
/* speaker/language are case-insensitive names from config.json (e.g. "ono_anna", "japanese").
 * text_ids: tokenized "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n".
 * instruct_ids: tokenized "<|im_start|>user\n{instruct}<|im_end|>\n" or NULL.
 * capture: store prefill/step fixtures in the result. Returns 0 on success. */
int  qtts_generate(qtts_model *m, const qtts_backend *be, const int32_t *text_ids, int n_text,
                   const int32_t *instruct_ids, int n_instruct, const char *speaker, const char *language,
                   const qtts_gen_params *gp, int capture, qtts_gen_result *out,
                   void (*on_frame)(void *user, const int32_t *codes16, int frame), void *user);
/* Borrowed features are valid only during this callback; copy before returning.
 * hidden is the normed talker state used to predict this codec frame, not the
 * state for the next frame. Return nonzero to cancel after committing the frame.
 * sample_start uses the exact 24 kHz / 1920-sample codec clock (12.5 Hz). */
typedef int (*qtts_feature_fn)(void *user, const int32_t *codes16,
                              const float *hidden, int hidden_size, int64_t sample_start);
int qtts_generate_ex(qtts_model *m, const qtts_backend *be, const int32_t *text_ids, int n_text,
                     const int32_t *instruct_ids, int n_instruct, const char *speaker, const char *language,
                     const qtts_gen_params *gp, int capture, qtts_gen_result *out,
                     void (*on_frame)(void *user, const int32_t *codes16, int frame), void *user,
                     qtts_feature_fn on_feature, void *feature_user);
/* host-side helpers shared with other backends */
void qtts_codec_embed_sum(const qtts_model *m, const int32_t *codes, float *x);  /* x[H] = sum of 16 rows */
void qtts_gen_result_free(qtts_gen_result *r);

#endif /* QTTS_TALKER_H */

/* ======================================================================== */
#if defined(QTTS_TALKER_IMPLEMENTATION) && !defined(QTTS_TALKER_IMPL_DONE)
#define QTTS_TALKER_IMPL_DONE

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "safetensors.h"
#include "philox_rng.h"

typedef struct {
    const uint16_t *ln1, *ln2, *qn, *kn, *q, *k, *v, *o, *gate, *up, *down;
} qtl_layer;

typedef struct {           /* generic Qwen3 decoder stack with KV cache */
    int n_layers, hidden, n_heads, n_kv, head_dim, inter, max_ctx;
    float eps, theta;
    qtl_layer *layers;
    const uint16_t *norm;
    float *kc, *vc;        /* [layer][max_ctx][n_kv*head_dim] */
} qtl_lm;

#define QTTS_MAX_NAMED 32
typedef struct { char name[64]; int id; } qtl_named;

struct qtts_model {
    st_context *st;
    qtl_lm talker, cp;
    int text_hidden, codec_vocab, text_vocab, num_groups, cp_vocab;
    const uint16_t *text_emb, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
    const uint16_t *codec_emb, *codec_head;
    const uint16_t *cp_emb[16], *cp_head[16], *mtp_w, *mtp_b;
    int codec_pad, codec_bos, codec_eos, think, nothink, think_bos, think_eos;
    int tts_bos, tts_eos, tts_pad;
    qtl_named spk[QTTS_MAX_NAMED], lang[QTTS_MAX_NAMED], dialect[QTTS_MAX_NAMED];
    int n_spk, n_lang, n_dialect;
};

/* ---- loading ---- */

static const uint16_t *qtl__w(qtts_model *m, const char *name, size_t expect) {
    int i = safetensors_find(m->st, name);
    if (i < 0) { fprintf(stderr, "qtts: missing tensor %s\n", name); exit(1); }
    if (strcmp(safetensors_dtype(m->st, i), "BF16")) {
        fprintf(stderr, "qtts: %s is %s (BF16 expected)\n", name, safetensors_dtype(m->st, i)); exit(1);
    }
    if (expect && safetensors_nbytes(m->st, i) != expect * 2) {
        fprintf(stderr, "qtts: %s size %zu != %zu\n", name, safetensors_nbytes(m->st, i) / 2, expect); exit(1);
    }
    return (const uint16_t *)safetensors_data(m->st, i);
}

static void qtl__load_lm(qtts_model *m, qtl_lm *lm, const char *prefix) {
    char nm[256];
    int H = lm->hidden, qd = lm->n_heads * lm->head_dim, kd = lm->n_kv * lm->head_dim, I = lm->inter;
    lm->layers = (qtl_layer *)calloc((size_t)lm->n_layers, sizeof(qtl_layer));
    for (int l = 0; l < lm->n_layers; l++) {
        qtl_layer *L = &lm->layers[l];
#define QTL_W(field, s, n) (snprintf(nm, sizeof(nm), "%s.layers.%d.%s", prefix, l, s), L->field = qtl__w(m, nm, n))
        QTL_W(ln1, "input_layernorm.weight", (size_t)H);
        QTL_W(ln2, "post_attention_layernorm.weight", (size_t)H);
        QTL_W(qn, "self_attn.q_norm.weight", (size_t)lm->head_dim);
        QTL_W(kn, "self_attn.k_norm.weight", (size_t)lm->head_dim);
        QTL_W(q, "self_attn.q_proj.weight", (size_t)qd * H);
        QTL_W(k, "self_attn.k_proj.weight", (size_t)kd * H);
        QTL_W(v, "self_attn.v_proj.weight", (size_t)kd * H);
        QTL_W(o, "self_attn.o_proj.weight", (size_t)H * qd);
        QTL_W(gate, "mlp.gate_proj.weight", (size_t)I * H);
        QTL_W(up, "mlp.up_proj.weight", (size_t)I * H);
        QTL_W(down, "mlp.down_proj.weight", (size_t)H * I);
#undef QTL_W
    }
    snprintf(nm, sizeof(nm), "%s.norm.weight", prefix);
    lm->norm = qtl__w(m, nm, (size_t)H);
    size_t kv = (size_t)lm->n_layers * lm->max_ctx * kd;
    lm->kc = (float *)calloc(kv, sizeof(float));
    lm->vc = (float *)calloc(kv, sizeof(float));
}

static char *qtl__read(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    b[n] = 0; fclose(f);
    if (len) *len = n;
    return b;
}

static int qtl__ji(const json_val *o, const char *k, int def) {
    const json_val *v = o ? json_obj_get(o, k) : NULL;
    return v && v->type == JSON_NUMBER ? (int)v->num : def;
}
static double qtl__jd(const json_val *o, const char *k, double def) {
    const json_val *v = o ? json_obj_get(o, k) : NULL;
    return v && v->type == JSON_NUMBER ? v->num : def;
}
static int qtl__named(const json_val *o, qtl_named *dst, int cap, int strings) {
    int n = 0;
    if (!o || o->type != JSON_OBJECT) return 0;
    for (int i = 0; i < o->obj.count && n < cap; i++) {
        const json_val *v = &o->obj.vals[i];
        if (strings) {
            if (v->type != JSON_STRING) continue;
            snprintf(dst[n].name, sizeof(dst[n].name), "%s", o->obj.keys[i]);
            /* dialect: key -> language-name (resolved later) stored in id = -1, name "spk\0lang" */
            size_t kl = strlen(dst[n].name);
            if (kl + 1 + (size_t)v->str.len < sizeof(dst[n].name)) {
                memcpy(dst[n].name + kl + 1, v->str.ptr, (size_t)v->str.len);
                dst[n].name[kl + 1 + v->str.len] = 0;
            }
            dst[n].id = -1;
        } else {
            if (v->type != JSON_NUMBER) continue;
            snprintf(dst[n].name, sizeof(dst[n].name), "%s", o->obj.keys[i]);
            dst[n].id = (int)v->num;
        }
        n++;
    }
    return n;
}

static void qtl__init_lm_cfg(qtl_lm *lm, const json_val *c, int max_ctx) {
    lm->n_layers = qtl__ji(c, "num_hidden_layers", 28);
    lm->hidden = qtl__ji(c, "hidden_size", 1024);
    lm->n_heads = qtl__ji(c, "num_attention_heads", 16);
    lm->n_kv = qtl__ji(c, "num_key_value_heads", 8);
    lm->head_dim = qtl__ji(c, "head_dim", 128);
    lm->inter = qtl__ji(c, "intermediate_size", 3072);
    lm->eps = (float)qtl__jd(c, "rms_norm_eps", 1e-6);
    lm->theta = (float)qtl__jd(c, "rope_theta", 1e6);
    lm->max_ctx = max_ctx;
}

qtts_model *qtts_model_load(const char *dir, int max_ctx) {
    char path[1024];
    long len = 0;
    snprintf(path, sizeof(path), "%s/config.json", dir);
    char *js = qtl__read(path, &len);
    if (!js) { fprintf(stderr, "qtts: cannot read %s\n", path); return NULL; }
    json_val *root = json_parse(js, (int)len);
    free(js);
    const json_val *tc = json_obj_get(root, "talker_config");
    const json_val *cc = tc ? json_obj_get(tc, "code_predictor_config") : NULL;
    if (!tc || !cc) { fprintf(stderr, "qtts: config.json lacks talker/code predictor config\n"); return NULL; }

    qtts_model *m = (qtts_model *)calloc(1, sizeof(*m));
    qtl__init_lm_cfg(&m->talker, tc, max_ctx);
    qtl__init_lm_cfg(&m->cp, cc, 32);
    m->text_hidden = qtl__ji(tc, "text_hidden_size", 2048);
    m->text_vocab = qtl__ji(tc, "text_vocab_size", 151936);
    m->codec_vocab = qtl__ji(tc, "vocab_size", 3072);
    m->num_groups = qtl__ji(tc, "num_code_groups", 16);
    m->cp_vocab = qtl__ji(cc, "vocab_size", 2048);
    m->codec_pad = qtl__ji(tc, "codec_pad_id", 2148);
    m->codec_bos = qtl__ji(tc, "codec_bos_id", 2149);
    m->codec_eos = qtl__ji(tc, "codec_eos_token_id", 2150);
    m->think = qtl__ji(tc, "codec_think_id", 2154);
    m->nothink = qtl__ji(tc, "codec_nothink_id", 2155);
    m->think_bos = qtl__ji(tc, "codec_think_bos_id", 2156);
    m->think_eos = qtl__ji(tc, "codec_think_eos_id", 2157);
    m->tts_bos = qtl__ji(root, "tts_bos_token_id", 151672);
    m->tts_eos = qtl__ji(root, "tts_eos_token_id", 151673);
    m->tts_pad = qtl__ji(root, "tts_pad_token_id", 151671);
    m->n_spk = qtl__named(json_obj_get(tc, "spk_id"), m->spk, QTTS_MAX_NAMED, 0);
    m->n_lang = qtl__named(json_obj_get(tc, "codec_language_id"), m->lang, QTTS_MAX_NAMED, 0);
    m->n_dialect = qtl__named(json_obj_get(tc, "spk_is_dialect"), m->dialect, QTTS_MAX_NAMED, 1);
    json_free(root);

    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    m->st = safetensors_open(path);
    if (!m->st) { fprintf(stderr, "qtts: cannot open %s\n", path); free(m); return NULL; }
    int H = m->talker.hidden, TH = m->text_hidden, CH = m->cp.hidden;
    m->text_emb = qtl__w(m, "talker.model.text_embedding.weight", (size_t)m->text_vocab * TH);
    m->fc1_w = qtl__w(m, "talker.text_projection.linear_fc1.weight", (size_t)TH * TH);
    m->fc1_b = qtl__w(m, "talker.text_projection.linear_fc1.bias", (size_t)TH);
    m->fc2_w = qtl__w(m, "talker.text_projection.linear_fc2.weight", (size_t)H * TH);
    m->fc2_b = qtl__w(m, "talker.text_projection.linear_fc2.bias", (size_t)H);
    m->codec_emb = qtl__w(m, "talker.model.codec_embedding.weight", (size_t)m->codec_vocab * H);
    m->codec_head = qtl__w(m, "talker.codec_head.weight", (size_t)m->codec_vocab * H);
    qtl__load_lm(m, &m->talker, "talker.model");
    qtl__load_lm(m, &m->cp, "talker.code_predictor.model");
    char nm[256];
    for (int g = 0; g < m->num_groups - 1; g++) {
        snprintf(nm, sizeof(nm), "talker.code_predictor.model.codec_embedding.%d.weight", g);
        m->cp_emb[g] = qtl__w(m, nm, (size_t)m->cp_vocab * H);
        snprintf(nm, sizeof(nm), "talker.code_predictor.lm_head.%d.weight", g);
        m->cp_head[g] = qtl__w(m, nm, (size_t)m->cp_vocab * CH);
    }
    if (safetensors_find(m->st, "talker.code_predictor.small_to_mtp_projection.weight") >= 0) {
        m->mtp_w = qtl__w(m, "talker.code_predictor.small_to_mtp_projection.weight", (size_t)CH * H);
        m->mtp_b = qtl__w(m, "talker.code_predictor.small_to_mtp_projection.bias", (size_t)CH);
    } else if (CH != H) {
        fprintf(stderr, "qtts: missing small_to_mtp_projection\n"); exit(1);
    }
    return m;
}

void qtts_model_free(qtts_model *m) {
    if (!m) return;
    free(m->talker.layers); free(m->talker.kc); free(m->talker.vc);
    free(m->cp.layers); free(m->cp.kc); free(m->cp.vc);
    safetensors_close(m->st);
    free(m);
}

int qtts_model_hidden(const qtts_model *m) { return m->talker.hidden; }
int qtts_model_codec_vocab(const qtts_model *m) { return m->codec_vocab; }

void qtts_gen_params_default(qtts_gen_params *p) {
    memset(p, 0, sizeof(*p));
    p->temperature = 0.9f; p->top_k = 50; p->top_p = 1.0f; p->rep_penalty = 1.05f;
    p->sub_temperature = 0.9f; p->sub_top_k = 50; p->sub_top_p = 1.0f;
    p->max_frames = 2048;
    p->seed = 1234;
}

/* ---- forward ---- */

static void qtl__bias_add_bf16(float *y, const uint16_t *b, int n) {
    for (int i = 0; i < n; i++) y[i] += qt_bf16_to_f32(b[i]);
}

/* x: [M][H] inputs at positions pos0..pos0+M-1 (updates KV cache). out: [M][H] final-normed hidden. */
static void qtl__forward(qtl_lm *lm, const float *x, int M, int pos0, float *out) {
    int H = lm->hidden, hd = lm->head_dim, nh = lm->n_heads, nkv = lm->n_kv;
    int qd = nh * hd, kd = nkv * hd, I = lm->inter;
    if (pos0 + M > lm->max_ctx) { fprintf(stderr, "qtts: context overflow (%d > %d)\n", pos0 + M, lm->max_ctx); exit(1); }
    float *h = (float *)malloc(sizeof(float) * (size_t)M * H);
    float *xn = (float *)malloc(sizeof(float) * (size_t)M * H);
    float *q = (float *)malloc(sizeof(float) * (size_t)M * qd);
    float *kv = (float *)malloc(sizeof(float) * (size_t)M * kd * 2);
    float *att = (float *)malloc(sizeof(float) * (size_t)M * qd);
    float *o = (float *)malloc(sizeof(float) * (size_t)M * H);
    float *g = (float *)malloc(sizeof(float) * (size_t)M * I);
    float *u = (float *)malloc(sizeof(float) * (size_t)M * I);
    memcpy(h, x, sizeof(float) * (size_t)M * H);
    for (int l = 0; l < lm->n_layers; l++) {
        const qtl_layer *L = &lm->layers[l];
        float *kc = lm->kc + (size_t)l * lm->max_ctx * kd;
        float *vc = lm->vc + (size_t)l * lm->max_ctx * kd;
        for (int i = 0; i < M; i++) qt_rmsnorm_bf16w(xn + (size_t)i * H, h + (size_t)i * H, L->ln1, H, lm->eps);
        qt_gemv_bf16(M, xn, L->q, qd, H, q);
        qt_gemv_bf16(M, xn, L->k, kd, H, kv);
        qt_gemv_bf16(M, xn, L->v, kd, H, kv + (size_t)M * kd);
        for (int i = 0; i < M; i++) {
            int pos = pos0 + i;
            for (int hh = 0; hh < nh; hh++) {
                float *qh = q + (size_t)i * qd + hh * hd;
                qt_rmsnorm_bf16w(qh, qh, L->qn, hd, lm->eps);
                qt_rope_neox(qh, hd, pos, lm->theta);
            }
            for (int hh = 0; hh < nkv; hh++) {
                float *kh = kv + (size_t)i * kd + hh * hd;
                qt_rmsnorm_bf16w(kh, kh, L->kn, hd, lm->eps);
                qt_rope_neox(kh, hd, pos, lm->theta);
            }
            memcpy(kc + (size_t)pos * kd, kv + (size_t)i * kd, sizeof(float) * kd);
            memcpy(vc + (size_t)pos * kd, kv + (size_t)(M + i) * kd, sizeof(float) * kd);
        }
        qt_attention(att, q, kc, vc, M, pos0, nh, nkv, hd, 0);
        qt_gemv_bf16(M, att, L->o, H, qd, o);
        for (size_t i = 0; i < (size_t)M * H; i++) h[i] += o[i];
        for (int i = 0; i < M; i++) qt_rmsnorm_bf16w(xn + (size_t)i * H, h + (size_t)i * H, L->ln2, H, lm->eps);
        qt_gemv_bf16(M, xn, L->gate, I, H, g);
        qt_gemv_bf16(M, xn, L->up, I, H, u);
        for (size_t i = 0; i < (size_t)M * I; i++) g[i] = qt_silu(g[i]) * u[i];
        qt_gemv_bf16(M, g, L->down, H, I, o);
        for (size_t i = 0; i < (size_t)M * H; i++) h[i] += o[i];
    }
    for (int i = 0; i < M; i++) qt_rmsnorm_bf16w(out + (size_t)i * H, h + (size_t)i * H, lm->norm, H, lm->eps);
    free(h); free(xn); free(q); free(kv); free(att); free(o); free(g); free(u);
}

/* text_projection(text_embedding(ids)) -> out [n][H] */
static void qtl__text_embed(const qtts_model *m, const int32_t *ids, int n, float *out) {
    int TH = m->text_hidden, H = m->talker.hidden;
    float *e = (float *)malloc(sizeof(float) * (size_t)n * TH);
    float *t = (float *)malloc(sizeof(float) * (size_t)n * TH);
    for (int i = 0; i < n; i++) {
        const uint16_t *row = m->text_emb + (size_t)ids[i] * TH;
        for (int j = 0; j < TH; j++) e[(size_t)i * TH + j] = qt_bf16_to_f32(row[j]);
    }
    qt_gemv_bf16(n, e, m->fc1_w, TH, TH, t);
    for (int i = 0; i < n; i++) {
        float *r = t + (size_t)i * TH;
        qtl__bias_add_bf16(r, m->fc1_b, TH);
        for (int j = 0; j < TH; j++) r[j] = qt_silu(r[j]);
    }
    qt_gemv_bf16(n, t, m->fc2_w, H, TH, out);
    for (int i = 0; i < n; i++) qtl__bias_add_bf16(out + (size_t)i * H, m->fc2_b, H);
    free(e); free(t);
}

static void qtl__row_add(float *dst, const uint16_t *table, int id, int n) {
    const uint16_t *r = table + (size_t)id * n;
    for (int j = 0; j < n; j++) dst[j] += qt_bf16_to_f32(r[j]);
}

static int qtl__lookup(const qtl_named *t, int n, const char *name) {
    char lower[64];
    size_t i = 0;
    for (; name[i] && i + 1 < sizeof(lower); i++) lower[i] = (char)tolower((unsigned char)name[i]);
    lower[i] = 0;
    for (int k = 0; k < n; k++) if (!strcmp(t[k].name, lower)) return k;
    return -1;
}

/* ---- sampling (HF order: repetition penalty, suppress, temperature, top-k, top-p) ---- */

typedef struct { float v; int i; } qtl_vi;
static int qtl__cmp_desc(const void *a, const void *b) {
    float x = ((const qtl_vi *)a)->v, y = ((const qtl_vi *)b)->v;
    return x < y ? 1 : x > y ? -1 : (((const qtl_vi *)a)->i - ((const qtl_vi *)b)->i);
}

static int qtl__sample(float *logits, int n, int greedy, float temp, int top_k, float top_p,
                       philox_rng_state *rng) {
    int best = 0;
    for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    if (greedy) return best;
    qtl_vi *a = (qtl_vi *)malloc(sizeof(qtl_vi) * (size_t)n);
    int na = 0;
    for (int i = 0; i < n; i++) if (logits[i] > -INFINITY) { a[na].v = logits[i] / temp; a[na].i = i; na++; }
    qsort(a, (size_t)na, sizeof(qtl_vi), qtl__cmp_desc);
    if (top_k > 0 && top_k < na) {
        /* keep ties with the k-th value, as torch.topk-threshold filtering does */
        float thr = a[top_k - 1].v;
        int k = top_k;
        while (k < na && a[k].v >= thr) k++;
        na = k;
    }
    double mx = a[0].v, sum = 0.0;
    double *p = (double *)malloc(sizeof(double) * (size_t)na);
    for (int i = 0; i < na; i++) { p[i] = exp(a[i].v - mx); sum += p[i]; }
    if (top_p < 1.0f) {
        double c = 0.0; int keep = na;
        for (int i = 0; i < na; i++) { c += p[i] / sum; if (c >= top_p) { keep = i + 1; break; } }
        na = keep; sum = 0.0;
        for (int i = 0; i < na; i++) sum += p[i];
    }
    uint32_t r4[4];
    philox_next4_u32(rng, r4);
    double r = philox_u32_to_uniform_f32(r4[0]) * sum, c = 0.0;
    int pick = a[na - 1].i;
    for (int i = 0; i < na; i++) { c += p[i]; if (r < c) { pick = a[i].i; break; } }
    free(a); free(p);
    return pick;
}

/* ---- CPU backend ---- */

typedef struct { qtts_model *m; float *last; } qtl_cpu_ctx;

static int qtl__cpu_talker(void *ctx, const float *x, int M, int pos0, float *last) {
    qtl_cpu_ctx *c = (qtl_cpu_ctx *)ctx;
    int H = c->m->talker.hidden;
    float *hid = (float *)malloc(sizeof(float) * (size_t)M * H);
    qtl__forward(&c->m->talker, x, M, pos0, hid);
    memcpy(c->last, hid + (size_t)(M - 1) * H, sizeof(float) * H);
    if (last) memcpy(last, c->last, sizeof(float) * H);
    free(hid);
    return 0;
}

static int qtl__cpu_head(void *ctx, float *logits) {
    qtl_cpu_ctx *c = (qtl_cpu_ctx *)ctx;
    qt_gemv_bf16(1, c->last, c->m->codec_head, c->m->codec_vocab, c->m->talker.hidden, logits);
    return 0;
}

/* code predictor for one frame: fills codes[1..G-1] given talker hidden and code0 */
static int qtl__cpu_predict(void *ctx, int32_t *codes, qtts_sample_fn sample, void *ud) {
    qtl_cpu_ctx *c = (qtl_cpu_ctx *)ctx;
    qtts_model *m = c->m;
    int H = m->talker.hidden, CH = m->cp.hidden, V = m->cp_vocab;
    float in2[2 * 4096], proj[2 * 4096], hid[2 * 4096];
    float *logits = (float *)malloc(sizeof(float) * (size_t)V);
    memcpy(in2, c->last, sizeof(float) * H);
    memset(in2 + H, 0, sizeof(float) * H);
    qtl__row_add(in2 + H, m->codec_emb, codes[0], H);
    for (int g = 0; g < m->num_groups - 1; g++) {
        int M = g == 0 ? 2 : 1, pos0 = g == 0 ? 0 : g + 1;
        const float *src = g == 0 ? in2 : in2 + H;
        if (g > 0) {
            memset(in2 + H, 0, sizeof(float) * H);
            qtl__row_add(in2 + H, m->cp_emb[g - 1], codes[g], H);
        }
        if (m->mtp_w) {
            qt_gemv_bf16(M, src, m->mtp_w, CH, H, proj);
            for (int i = 0; i < M; i++) qtl__bias_add_bf16(proj + (size_t)i * CH, m->mtp_b, CH);
        } else {
            memcpy(proj, src, sizeof(float) * (size_t)M * CH);
        }
        qtl__forward(&m->cp, proj, M, pos0, hid);
        qt_gemv_bf16(1, hid + (size_t)(M - 1) * CH, m->cp_head[g], V, CH, logits);
        codes[g + 1] = sample(ud, logits, V);
    }
    free(logits);
    return 0;
}

void qtts_codec_embed_sum(const qtts_model *m, const int32_t *codes, float *x) {
    int H = m->talker.hidden;
    memset(x, 0, sizeof(float) * H);
    qtl__row_add(x, m->codec_emb, codes[0], H);
    for (int g = 1; g < m->num_groups; g++) qtl__row_add(x, m->cp_emb[g - 1], codes[g], H);
}

typedef struct { const qtts_gen_params *gp; philox_rng_state *rng; } qtl_sub_sampler;
static int qtl__sub_sample(void *ud, float *logits, int n) {
    qtl_sub_sampler *s = (qtl_sub_sampler *)ud;
    return qtl__sample(logits, n, s->gp->greedy, s->gp->sub_temperature, s->gp->sub_top_k, s->gp->sub_top_p, s->rng);
}

int qtts_generate_ex(qtts_model *m, const qtts_backend *be_in, const int32_t *ids, int n_ids,
                  const int32_t *inst, int n_inst, const char *speaker, const char *language,
                  const qtts_gen_params *gp, int capture, qtts_gen_result *out,
                  void (*on_frame)(void *user, const int32_t *codes16, int frame), void *user,
                  qtts_feature_fn on_feature, void *feature_user) {
    int H = m->talker.hidden, V = m->codec_vocab, G = m->num_groups;
    memset(out, 0, sizeof(*out));
    if (n_ids < 9) { fprintf(stderr, "qtts: text ids too short\n"); return -1; }

    /* codec prefix: [think|nothink, think_bos, (lang), think_eos, (speaker), pad, bos] */
    int spk_id = -1, lang_id = -1;
    if (speaker && *speaker) {
        int k = qtl__lookup(m->spk, m->n_spk, speaker);
        if (k < 0) { fprintf(stderr, "qtts: unknown speaker %s\n", speaker); return -1; }
        spk_id = m->spk[k].id;
    }
    if (language && *language && strcmp(language, "auto") && strcmp(language, "Auto")) {
        int k = qtl__lookup(m->lang, m->n_lang, language);
        if (k < 0) { fprintf(stderr, "qtts: unknown language %s\n", language); return -1; }
        lang_id = m->lang[k].id;
    }
    /* dialect speakers override Chinese/auto */
    if (spk_id >= 0 && (lang_id < 0 || !strcmp(language, "chinese") || !strcmp(language, "Chinese"))) {
        int k = qtl__lookup(m->dialect, m->n_dialect, speaker);
        if (k >= 0) {
            const char *dl = m->dialect[k].name + strlen(m->dialect[k].name) + 1;
            int li = qtl__lookup(m->lang, m->n_lang, dl);
            if (li >= 0) lang_id = m->lang[li].id;
        }
    }
    int cpre[8], nc = 0;
    if (lang_id < 0) { cpre[nc++] = m->nothink; cpre[nc++] = m->think_bos; cpre[nc++] = m->think_eos; }
    else { cpre[nc++] = m->think; cpre[nc++] = m->think_bos; cpre[nc++] = lang_id; cpre[nc++] = m->think_eos; }
    const qtts_voice_clone *vc = gp->clone;
    int spk_row = -1;                   /* position of the speaker row in the codec prefix */
    if (vc && vc->spk_emb) spk_row = nc++;
    else if (spk_id >= 0) cpre[nc++] = spk_id;
    cpre[nc++] = m->codec_pad;
    cpre[nc++] = m->codec_bos;
    int icl = vc && vc->ref_codes && vc->n_ref > 0 && vc->ref_ids && vc->n_ref_ids > 5;

    int n_text = n_ids - 3 - 5;         /* ids[3:-5] */
    int n_rtext = icl ? vc->n_ref_ids - 3 - 2 : 0;   /* ref_ids[3:-2] */
    int cap = n_inst + 3 + nc + n_text + 4 + (icl ? n_rtext + vc->n_ref + 4 : 0);
    float *pre = (float *)calloc((size_t)cap * H, sizeof(float));
    int L = 0;
    if (inst && n_inst > 0) { qtl__text_embed(m, inst, n_inst, pre); L += n_inst; }
    qtl__text_embed(m, ids, 3, pre + (size_t)L * H);   /* <|im_start|>assistant\n */
    L += 3;
    int32_t special[3] = { m->tts_bos, m->tts_eos, m->tts_pad };
    float *sp = (float *)malloc(sizeof(float) * 3 * H);
    qtl__text_embed(m, special, 3, sp);
    const float *e_bos = sp, *e_eos = sp + H, *e_pad = sp + 2 * H;
    /* tts_pad x (nc-2) + tts_bos, each + codec prefix row 0..nc-2 (speaker row = x-vector) */
    for (int i = 0; i < nc - 1; i++) {
        float *r = pre + (size_t)L * H;
        memcpy(r, i < nc - 2 ? e_pad : e_bos, sizeof(float) * H);
        if (i == spk_row) for (int j = 0; j < H; j++) r[j] += vc->spk_emb[j];
        else qtl__row_add(r, m->codec_emb, cpre[i], H);
        L++;
    }
    float *trailing = NULL;
    int n_trailing = 0;
    if (icl) {
        /* text track: (ref text + target text) + tts_eos; codec track: codec_bos + sum of ref codes */
        int Lt = n_rtext + n_text + 1, Lc = 1 + vc->n_ref;
        float *te = (float *)malloc(sizeof(float) * (size_t)Lt * H);
        int32_t *cat_ids = (int32_t *)malloc(sizeof(int32_t) * (size_t)(Lt > 1 ? Lt - 1 : 1));
        memcpy(cat_ids, vc->ref_ids + 3, sizeof(int32_t) * (size_t)n_rtext);
        memcpy(cat_ids + n_rtext, ids + 3, sizeof(int32_t) * (size_t)n_text);
        qtl__text_embed(m, cat_ids, Lt - 1, te);
        memcpy(te + (size_t)(Lt - 1) * H, e_eos, sizeof(float) * H);
        free(cat_ids);
        float *ce = (float *)calloc((size_t)Lc * H, sizeof(float));
        qtl__row_add(ce, m->codec_emb, m->codec_bos, H);
        for (int f = 0; f < vc->n_ref; f++) qtts_codec_embed_sum(m, vc->ref_codes + (size_t)f * m->num_groups, ce + (size_t)(1 + f) * H);
        if (!gp->streaming) {
            for (int i = 0; i < Lt; i++) {
                float *r = pre + (size_t)L++ * H;
                memcpy(r, te + (size_t)i * H, sizeof(float) * H);
                qtl__row_add(r, m->codec_emb, m->codec_pad, H);
            }
            for (int i = 0; i < Lc; i++) {
                float *r = pre + (size_t)L++ * H;
                for (int j = 0; j < H; j++) r[j] = ce[(size_t)i * H + j] + e_pad[j];
            }
            n_trailing = 1;
            trailing = (float *)malloc(sizeof(float) * H);
            memcpy(trailing, e_pad, sizeof(float) * H);
        } else {
            for (int i = 0; i < Lc; i++) {
                const float *t = i < Lt ? te + (size_t)i * H : e_pad;
                float *r = pre + (size_t)L++ * H;
                for (int j = 0; j < H; j++) r[j] = t[j] + ce[(size_t)i * H + j];
            }
            n_trailing = Lt > Lc ? Lt - Lc : 1;
            trailing = (float *)malloc(sizeof(float) * (size_t)n_trailing * H);
            if (Lt > Lc) memcpy(trailing, te + (size_t)Lc * H, sizeof(float) * (size_t)n_trailing * H);
            else memcpy(trailing, e_pad, sizeof(float) * H);
        }
        free(te); free(ce);
    } else if (!gp->streaming) {
        /* text tokens + tts_eos, each + codec_pad; then tts_pad + codec_bos */
        qtl__text_embed(m, ids + 3, n_text, pre + (size_t)L * H);
        memcpy(pre + (size_t)(L + n_text) * H, e_eos, sizeof(float) * H);
        for (int i = 0; i <= n_text; i++) qtl__row_add(pre + (size_t)(L + i) * H, m->codec_emb, m->codec_pad, H);
        L += n_text + 1;
        memcpy(pre + (size_t)L * H, e_pad, sizeof(float) * H);
        qtl__row_add(pre + (size_t)L * H, m->codec_emb, m->codec_bos, H);
        L++;
        trailing = (float *)malloc(sizeof(float) * H);
        memcpy(trailing, e_pad, sizeof(float) * H);
        n_trailing = 1;
    } else {
        /* first text token + codec_bos; remaining text + tts_eos trail one per frame */
        qtl__text_embed(m, ids + 3, 1, pre + (size_t)L * H);
        qtl__row_add(pre + (size_t)L * H, m->codec_emb, m->codec_bos, H);
        L++;
        n_trailing = n_text - 1 + 1;
        trailing = (float *)malloc(sizeof(float) * (size_t)n_trailing * H);
        if (n_text > 1) qtl__text_embed(m, ids + 4, n_text - 1, trailing);
        memcpy(trailing + (size_t)(n_trailing - 1) * H, e_eos, sizeof(float) * H);
    }

    int max_steps = gp->max_frames + 1;
    if (L + max_steps > m->talker.max_ctx) max_steps = m->talker.max_ctx - L;
    out->codes = (int32_t *)malloc(sizeof(int32_t) * (size_t)max_steps * G);
    if (capture) {
        out->prefill_embeds = pre; out->prefill_len = L;
        out->step_logits = (float *)malloc(sizeof(float) * (size_t)max_steps * V);
        out->step_hidden = (float *)malloc(sizeof(float) * (size_t)max_steps * H);
    }
    philox_rng_state rng;
    philox_rng_init(&rng, gp->seed, 0);

    qtl_cpu_ctx cpu = { m, (float *)malloc(sizeof(float) * H) };
    qtts_backend cpu_be = { &cpu, qtl__cpu_talker, qtl__cpu_head, qtl__cpu_predict };
    const qtts_backend *be = be_in ? be_in : &cpu_be;
    qtl_sub_sampler sub = { gp, &rng };
    float *logits = (float *)malloc(sizeof(float) * (size_t)V);
    float *x = (float *)malloc(sizeof(float) * H);
    float *last = (float *)malloc(sizeof(float) * H);
    unsigned char *seen = (unsigned char *)calloc((size_t)V, 1);
    int rc = be->talker(be->ctx, pre, L, 0, last);
    int pos = L, step = 0, frames = 0;
    while (rc == 0) {
        if ((rc = be->head(be->ctx, logits))) break;
        if (capture) {
            memcpy(out->step_logits + (size_t)step * V, logits, sizeof(float) * V);
            memcpy(out->step_hidden + (size_t)step * H, last, sizeof(float) * H);
        }
        step++;
        /* repetition penalty over previously generated code0 tokens */
        if (gp->rep_penalty != 1.0f)
            for (int i = 0; i < V; i++)
                if (seen[i]) logits[i] = logits[i] > 0 ? logits[i] / gp->rep_penalty : logits[i] * gp->rep_penalty;
        /* suppress [V-1024, V) except EOS; min_new_tokens = 2 suppresses EOS for the first 2 steps */
        for (int i = V - 1024; i < V; i++) if (i != m->codec_eos) logits[i] = -INFINITY;
        if (frames < 2) logits[m->codec_eos] = -INFINITY;
        int c0 = qtl__sample(logits, V, gp->greedy, gp->temperature, gp->top_k, gp->top_p, &rng);
        /* HF semantics: max_frames == max_new_tokens; the last sampled code0 has no
         * following forward, so at most max_frames - 1 complete frames are produced. */
        if (c0 == m->codec_eos || step >= gp->max_frames || step >= max_steps) break;
        seen[c0] = 1;
        int32_t *codes = out->codes + (size_t)frames * G;
        codes[0] = c0;
        if ((rc = be->predict(be->ctx, codes, qtl__sub_sample, &sub))) break;
        if (on_frame) on_frame(user, codes, frames);
        if (on_feature && on_feature(feature_user, codes, last, H, (int64_t)frames * 1920)) {
            frames++;
            rc = 1; /* cancellation: result includes the already emitted frame */
            break;
        }
        /* next talker input: sum of all codebook embeddings + trailing text */
        qtts_codec_embed_sum(m, codes, x);
        const float *tt = frames < n_trailing ? trailing + (size_t)frames * H : e_pad;
        for (int j = 0; j < H; j++) x[j] += tt[j];
        frames++;
        rc = be->talker(be->ctx, x, 1, pos, last);
        pos++;
    }
    out->n_frames = frames;
    out->n_steps = step;
    if (!capture) free(pre);
    free(sp); free(trailing); free(logits); free(x); free(seen); free(last); free(cpu.last);
    return rc;
}

int qtts_generate(qtts_model *m, const qtts_backend *be, const int32_t *ids, int n_ids,
                  const int32_t *inst, int n_inst, const char *speaker, const char *language,
                  const qtts_gen_params *gp, int capture, qtts_gen_result *out,
                  void (*on_frame)(void *user, const int32_t *codes16, int frame), void *user) {
    return qtts_generate_ex(m, be, ids, n_ids, inst, n_inst, speaker, language,
                            gp, capture, out, on_frame, user, NULL, NULL);
}

void qtts_gen_result_free(qtts_gen_result *r) {
    free(r->codes); free(r->prefill_embeds); free(r->step_logits); free(r->step_hidden);
    memset(r, 0, sizeof(*r));
}

#endif /* QTTS_TALKER_IMPLEMENTATION */
