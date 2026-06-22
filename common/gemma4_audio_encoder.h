/* gemma4_audio_encoder.h — CPU Conformer/USM audio encoder for gemma-4
 * "qat-mobile" checkpoints, loaded DIRECTLY from safetensors (gemma wNa8o8 QAT:
 * 2-bit U8 linears, 4-bit lconv1d.linear_start, F32 norms/convs/proj).
 *
 * Mirrors HF transformers Gemma4AudioModel:
 *   subsample_conv_projection (2x conv2d stride2 + LayerNorm + ReLU + input_proj)
 *   -> 12 Conformer layers (ff1 half-step / rel-pos chunked attn / lconv1d / ff2
 *      half-step / norm_out) -> output_proj -> embed_audio (RMSNorm -> projection).
 *
 *   g4a_model *m = g4a_load_safetensors(model_dir);
 *   float *soft = g4a_encode(m, mel, n_frames, &n_tokens, &dim);  // mel: [n_frames*128]
 *   ... splice soft [n_tokens*dim] into the LLM ...  free(soft); g4a_free(m);
 *
 * Self-contained (host F32 weights, no qtensor dependency). Requires safetensors.h
 * (with SAFETENSORS_IMPLEMENTATION) to be included by the TU before this header's
 * implementation. Validated bit-exact (cos 1.0, rel_L2 2e-6) vs the HF transformers
 * Gemma4AudioModel torch oracle on a 15 s clip (375 soft tokens).
 */
#ifndef GEMMA4_AUDIO_ENCODER_H
#define GEMMA4_AUDIO_ENCODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct g4a_model g4a_model;

/* model_path: directory with config.json + model.safetensors (or a .safetensors file). */
g4a_model *g4a_load_safetensors(const char *model_path);
void       g4a_free(g4a_model *m);

/* mel: row-major [n_frames, 128] log-mel features. Returns malloc'd soft tokens
 * [n_tokens * out_dim] (out_dim = LLM embed dim 1536). Caller frees. */
float *g4a_encode(g4a_model *m, const float *mel, int n_frames, int *out_tokens, int *out_dim);

#ifdef __cplusplus
}
#endif

#ifdef GEMMA4_AUDIO_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

/* ----- config (gemma4 E2B audio_config) ----- */
#define G4A_HID    1024
#define G4A_HEADS  8
#define G4A_HDIM   128      /* 1024/8 */
#define G4A_LAYERS 12
#define G4A_FFN    4096     /* hidden*4 */
#define G4A_KCONV  5        /* conv_kernel_size */
#define G4A_CHUNK  12
#define G4A_PAST   12       /* attention_context_left - 1 */
#define G4A_FUT    0
#define G4A_CTX    24       /* chunk + past + future */
#define G4A_MEL    128
#define G4A_SC0    128      /* subsampling_conv_channels[0] */
#define G4A_SC1    32       /* subsampling_conv_channels[1] */
#define G4A_OUT    1536     /* output_proj_dims */
#define G4A_EPS    1e-6f
#define G4A_SOFTCAP 50.0f
#define G4A_RESID  0.5f

typedef struct {
    /* ff1, ff2 */
    float *ff1_w1, *ff1_w2, *ff1_pre, *ff1_post;   /* w1 [4096,1024] w2 [1024,4096] norms [1024] */
    float *ff2_w1, *ff2_w2, *ff2_pre, *ff2_post;
    /* attention */
    float *q_w, *k_w, *v_w, *post_w;   /* [1024,1024] */
    float *rel_k_w;                    /* [1024,1024] */
    float *per_dim_scale;              /* [128] */
    /* lconv1d */
    float *lc_start, *lc_end;          /* start [2048,1024], end [1024,1024] */
    float *lc_dw;                      /* depthwise [1024,5] */
    float *lc_pre, *lc_conv_norm;      /* [1024] */
    /* layer norms */
    float *norm_pre_attn, *norm_post_attn, *norm_out;  /* [1024] */
} g4a_layer;

struct g4a_model {
    /* subsample */
    float *conv0_w;   /* [128,1,3,3] */
    float *conv0_norm;/* [128] */
    float *conv1_w;   /* [32,128,3,3] */
    float *conv1_norm;/* [32] */
    float *in_proj;   /* [1024,1024] */
    /* blocks */
    g4a_layer layers[G4A_LAYERS];
    /* output */
    float *out_proj_w; /* [1536,1024] */
    float *out_proj_b; /* [1536] */
    float *embed_proj; /* [1536,1536] embed_audio.embedding_projection */
};

/* ---------- safetensors helpers (uses safetensors.h API) ---------- */
static void *g4a_st_find(st_context **sh, int ns, const char *name, const char **dtype, const uint64_t **shape, int *nd) {
    for (int s = 0; s < ns; s++) {
        int i = safetensors_find(sh[s], name);
        if (i >= 0) { if (dtype) *dtype = safetensors_dtype(sh[s], i);
                      if (shape) *shape = safetensors_shape(sh[s], i);
                      if (nd) *nd = safetensors_ndims(sh[s], i);
                      return safetensors_data(sh[s], i); }
    }
    return NULL;
}
static float g4a_bf16(uint16_t u) { union { uint32_t i; float f; } v; v.i = (uint32_t)u << 16; return v.f; }

/* Load an F32 or BF16 tensor -> malloc'd float[]. */
static float *g4a_load_raw(st_context **sh, int ns, const char *name, int n) {
    const char *dt; void *d = g4a_st_find(sh, ns, name, &dt, NULL, NULL);
    if (!d) { fprintf(stderr, "g4a: missing %s\n", name); return NULL; }
    float *out = (float *)malloc((size_t)n * sizeof(float));
    if (strcmp(dt, "F32") == 0) memcpy(out, d, (size_t)n * sizeof(float));
    else if (strcmp(dt, "BF16") == 0) { const uint16_t *s = (const uint16_t *)d; for (int i = 0; i < n; i++) out[i] = g4a_bf16(s[i]); }
    else { fprintf(stderr, "g4a: %s unexpected dtype %s\n", name, dt); free(out); return NULL; }
    return out;
}

/* Load a gemma-QAT linear "<base>.weight" (U8 bits-packed interleaved) + ".weight_scale"
 * -> malloc'd float[out*logical_in]. */
static float *g4a_load_qat(st_context **sh, int ns, const char *base, int bits) {
    char wn[300], scn[316];
    snprintf(wn, sizeof(wn), "%s.weight", base);
    snprintf(scn, sizeof(scn), "%s.weight_scale", base);
    const char *dt; const uint64_t *shp; int nd;
    const uint8_t *w = (const uint8_t *)g4a_st_find(sh, ns, wn, &dt, &shp, &nd);
    if (!w) { fprintf(stderr, "g4a: missing %s\n", wn); return NULL; }
    const float *sc = (const float *)g4a_st_find(sh, ns, scn, NULL, NULL, NULL);
    if (!sc) { fprintf(stderr, "g4a: missing %s\n", scn); return NULL; }
    int out = (int)shp[0], packed = (int)shp[1];
    int per, zp, mask, lin;
    if (strcmp(dt, "I8") == 0) { per = 1; zp = 0; mask = 0xFF; lin = packed; }
    else { per = 8 / bits; zp = 1 << (bits - 1); mask = (1 << bits) - 1; lin = packed * per; }
    float *o = (float *)malloc((size_t)out * lin * sizeof(float));
    for (int r = 0; r < out; r++) {
        float s = sc[r]; const uint8_t *wr = w + (size_t)r * packed; float *orow = o + (size_t)r * lin;
        if (per == 1) { for (int j = 0; j < packed; j++) orow[j] = (float)((int8_t)wr[j]) * s; }
        else for (int j = 0; j < packed; j++) { int b = wr[j];
            for (int k = 0; k < per; k++) orow[per * j + k] = (float)(((b >> (k * bits)) & mask) - zp) * s; }
    }
    return o;
}

g4a_model *g4a_load_safetensors(const char *model_path) {
    st_context *sh[16] = {0};
    int ns = 0;
    struct stat sb;
    if (stat(model_path, &sb) == 0 && (sb.st_mode & S_IFDIR)) {
        char p[512]; snprintf(p, sizeof(p), "%s/model.safetensors", model_path);
        st_context *s = safetensors_open(p); if (s) sh[ns++] = s;
        for (int i = 1; i <= 64 && !ns; i++) for (int t = i; t <= 64; t++) {
            snprintf(p, sizeof(p), "%s/model-%05d-of-%05d.safetensors", model_path, i, t);
            s = safetensors_open(p); if (s) { sh[ns++] = s; break; }
        }
    } else { st_context *s = safetensors_open(model_path); if (s) sh[ns++] = s; }
    if (!ns) { fprintf(stderr, "g4a: cannot open %s\n", model_path); return NULL; }

    g4a_model *m = (g4a_model *)calloc(1, sizeof(g4a_model));
    const char *A = "model.audio_tower.";
    char nm[300], base[300];
    #define RAW(dst, name, n) do { m->dst = g4a_load_raw(sh, ns, name, n); if (!m->dst) goto fail; } while (0)

    snprintf(nm, sizeof(nm), "%ssubsample_conv_projection.layer0.conv.weight", A); RAW(conv0_w, nm, 128*1*3*3);
    snprintf(nm, sizeof(nm), "%ssubsample_conv_projection.layer0.norm.weight", A); RAW(conv0_norm, nm, 128);
    snprintf(nm, sizeof(nm), "%ssubsample_conv_projection.layer1.conv.weight", A); RAW(conv1_w, nm, 32*128*3*3);
    snprintf(nm, sizeof(nm), "%ssubsample_conv_projection.layer1.norm.weight", A); RAW(conv1_norm, nm, 32);
    snprintf(nm, sizeof(nm), "%ssubsample_conv_projection.input_proj_linear.weight", A); RAW(in_proj, nm, 1024*1024);
    snprintf(nm, sizeof(nm), "%soutput_proj.weight", A); RAW(out_proj_w, nm, 1536*1024);
    snprintf(nm, sizeof(nm), "%soutput_proj.bias", A); RAW(out_proj_b, nm, 1536);
    RAW(embed_proj, "model.embed_audio.embedding_projection.weight", 1536*1536);

    for (int L = 0; L < G4A_LAYERS; L++) {
        g4a_layer *cl = &m->layers[L];
        #define QAT(dst, suffix, bits) do { snprintf(base, sizeof(base), "%slayers.%d.%s", A, L, suffix); \
            cl->dst = g4a_load_qat(sh, ns, base, bits); if (!cl->dst) goto fail; } while (0)
        #define LRAW(dst, suffix, n) do { snprintf(nm, sizeof(nm), "%slayers.%d.%s", A, L, suffix); \
            cl->dst = g4a_load_raw(sh, ns, nm, n); if (!cl->dst) goto fail; } while (0)
        QAT(ff1_w1, "feed_forward1.ffw_layer_1.linear", 2);
        QAT(ff1_w2, "feed_forward1.ffw_layer_2.linear", 2);
        LRAW(ff1_pre, "feed_forward1.pre_layer_norm.weight", 1024);
        LRAW(ff1_post, "feed_forward1.post_layer_norm.weight", 1024);
        QAT(ff2_w1, "feed_forward2.ffw_layer_1.linear", 2);
        QAT(ff2_w2, "feed_forward2.ffw_layer_2.linear", 2);
        LRAW(ff2_pre, "feed_forward2.pre_layer_norm.weight", 1024);
        LRAW(ff2_post, "feed_forward2.post_layer_norm.weight", 1024);
        QAT(q_w, "self_attn.q_proj.linear", 2);
        QAT(k_w, "self_attn.k_proj.linear", 2);
        QAT(v_w, "self_attn.v_proj.linear", 2);
        QAT(post_w, "self_attn.post.linear", 2);
        LRAW(rel_k_w, "self_attn.relative_k_proj.weight", 1024*1024);
        LRAW(per_dim_scale, "self_attn.per_dim_scale", 128);
        QAT(lc_start, "lconv1d.linear_start.linear", 4);
        QAT(lc_end, "lconv1d.linear_end.linear", 2);
        LRAW(lc_dw, "lconv1d.depthwise_conv1d.weight", 1024*5);
        LRAW(lc_pre, "lconv1d.pre_layer_norm.weight", 1024);
        LRAW(lc_conv_norm, "lconv1d.conv_norm.weight", 1024);
        LRAW(norm_pre_attn, "norm_pre_attn.weight", 1024);
        LRAW(norm_post_attn, "norm_post_attn.weight", 1024);
        LRAW(norm_out, "norm_out.weight", 1024);
        #undef QAT
        #undef LRAW
    }
    #undef RAW
    for (int s = 0; s < ns; s++) safetensors_close(sh[s]);
    return m;
fail:
    for (int s = 0; s < ns; s++) safetensors_close(sh[s]);
    g4a_free(m);
    return NULL;
}

/* ---------- math helpers ---------- */
static void g4a_rmsnorm(float *x, const float *w, int n) {  /* in-place, with weight */
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = (float)(1.0 / sqrt(ss / n + G4A_EPS));
    for (int i = 0; i < n; i++) x[i] = x[i] * inv * w[i];
}
/* y[out] = W[out,in] . x[in] */
static void g4a_matmul(const float *W, const float *x, float *y, int out, int in) {
    for (int o = 0; o < out; o++) { const float *wr = W + (size_t)o * in; double s = 0;
        for (int i = 0; i < in; i++) s += (double)wr[i] * x[i]; y[o] = (float)s; }
}
static inline float g4a_silu(float x) { return x / (1.0f + expf(-x)); }
static inline float g4a_softplus(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }

/* half-step FFN (Gemma4AudioFeedForward), in-place on x[T,1024] row t */
static void g4a_ffn(const float *w1, const float *w2, const float *pre, const float *post,
                    float *x, float *tmp_h, float *tmp_n) {
    memcpy(tmp_n, x, G4A_HID * sizeof(float));
    g4a_rmsnorm(tmp_n, pre, G4A_HID);
    g4a_matmul(w1, tmp_n, tmp_h, G4A_FFN, G4A_HID);
    for (int i = 0; i < G4A_FFN; i++) tmp_h[i] = g4a_silu(tmp_h[i]);
    g4a_matmul(w2, tmp_h, tmp_n, G4A_HID, G4A_FFN);
    g4a_rmsnorm(tmp_n, post, G4A_HID);
    for (int i = 0; i < G4A_HID; i++) x[i] += G4A_RESID * tmp_n[i];
}

/* ---------- subsample conv front-end: mel[T,128] -> hs[T4,1024], returns T4 ---------- */
static int g4a_subsample(g4a_model *m, const float *mel, int T, float **out_hs) {
    int H0 = (T - 1) / 2 + 1, W0 = (G4A_MEL - 1) / 2 + 1;       /* layer0 out spatial (W0=64) */
    int H1 = (H0 - 1) / 2 + 1, W1 = (W0 - 1) / 2 + 1;           /* layer1 out spatial (W1=32) */
    /* layer0: in [1,T,128] -> [128,H0,W0] */
    float *l0 = (float *)calloc((size_t)G4A_SC0 * H0 * W0, sizeof(float));
    for (int oc = 0; oc < G4A_SC0; oc++) {
        const float *wf = m->conv0_w + (size_t)oc * 9;  /* [1,3,3] */
        for (int oh = 0; oh < H0; oh++) for (int ow = 0; ow < W0; ow++) {
            float acc = 0;
            for (int kh = 0; kh < 3; kh++) for (int kw = 0; kw < 3; kw++) {
                int ih = oh * 2 - 1 + kh, iw = ow * 2 - 1 + kw;
                if (ih >= 0 && ih < T && iw >= 0 && iw < G4A_MEL) acc += mel[(size_t)ih * G4A_MEL + iw] * wf[kh * 3 + kw];
            }
            l0[((size_t)oc * H0 + oh) * W0 + ow] = acc;
        }
    }
    /* LayerNorm over channel(128) per (oh,ow) + ReLU */
    for (int oh = 0; oh < H0; oh++) for (int ow = 0; ow < W0; ow++) {
        double mean = 0; for (int c = 0; c < G4A_SC0; c++) mean += l0[((size_t)c * H0 + oh) * W0 + ow];
        mean /= G4A_SC0; double var = 0;
        for (int c = 0; c < G4A_SC0; c++) { double d = l0[((size_t)c * H0 + oh) * W0 + ow] - mean; var += d * d; }
        var /= G4A_SC0; float inv = (float)(1.0 / sqrt(var + G4A_EPS));
        for (int c = 0; c < G4A_SC0; c++) { size_t idx = ((size_t)c * H0 + oh) * W0 + ow;
            float v = ((float)(l0[idx] - mean)) * inv * m->conv0_norm[c]; l0[idx] = v > 0 ? v : 0; }
    }
    /* layer1: in [128,H0,W0] -> [32,H1,W1] */
    float *l1 = (float *)calloc((size_t)G4A_SC1 * H1 * W1, sizeof(float));
    for (int oc = 0; oc < G4A_SC1; oc++) {
        for (int oh = 0; oh < H1; oh++) for (int ow = 0; ow < W1; ow++) {
            float acc = 0;
            for (int ic = 0; ic < G4A_SC0; ic++) { const float *wf = m->conv1_w + (((size_t)oc * G4A_SC0 + ic) * 9);
                for (int kh = 0; kh < 3; kh++) for (int kw = 0; kw < 3; kw++) {
                    int ih = oh * 2 - 1 + kh, iw = ow * 2 - 1 + kw;
                    if (ih >= 0 && ih < H0 && iw >= 0 && iw < W0) acc += l0[((size_t)ic * H0 + ih) * W0 + iw] * wf[kh * 3 + kw];
                }
            }
            l1[((size_t)oc * H1 + oh) * W1 + ow] = acc;
        }
    }
    free(l0);
    for (int oh = 0; oh < H1; oh++) for (int ow = 0; ow < W1; ow++) {
        double mean = 0; for (int c = 0; c < G4A_SC1; c++) mean += l1[((size_t)c * H1 + oh) * W1 + ow];
        mean /= G4A_SC1; double var = 0;
        for (int c = 0; c < G4A_SC1; c++) { double d = l1[((size_t)c * H1 + oh) * W1 + ow] - mean; var += d * d; }
        var /= G4A_SC1; float inv = (float)(1.0 / sqrt(var + G4A_EPS));
        for (int c = 0; c < G4A_SC1; c++) { size_t idx = ((size_t)c * H1 + oh) * W1 + ow;
            float v = ((float)(l1[idx] - mean)) * inv * m->conv1_norm[c]; l1[idx] = v > 0 ? v : 0; }
    }
    /* reshape permute(0,2,3,1): hs_in[t][w*32 + c], w in [0,W1) freq, c in [0,32) chan ; then input_proj */
    int T4 = H1, feat = W1 * G4A_SC1;   /* 32*32 = 1024 */
    float *hs = (float *)malloc((size_t)T4 * G4A_HID * sizeof(float));
    float *row = (float *)malloc(feat * sizeof(float));
    for (int t = 0; t < T4; t++) {
        for (int w = 0; w < W1; w++) for (int c = 0; c < G4A_SC1; c++)
            row[w * G4A_SC1 + c] = l1[((size_t)c * H1 + t) * W1 + w];
        g4a_matmul(m->in_proj, row, hs + (size_t)t * G4A_HID, G4A_HID, feat);
    }
    free(row); free(l1);
    *out_hs = hs;
    return T4;
}

/* ---------- relative position encoding: pos_embed[13,1024] ---------- */
static void g4a_rel_pos(float *pe) {  /* pe[13*1024] */
    int nts = G4A_HID / 2;  /* 512 */
    double log_inc = log(10000.0) / (nts - 1);
    for (int p = 0; p < 13; p++) {
        float pos = (float)(12 - p);   /* arange(12,-1,-1) */
        for (int j = 0; j < nts; j++) {
            double its = exp(-(double)j * log_inc);
            float st = pos * (float)its;
            pe[p * G4A_HID + j] = sinf(st);
            pe[p * G4A_HID + nts + j] = cosf(st);
        }
    }
}

/* ---------- conformer attention for one layer; hs[T,1024] in-place residual handled by caller ----------
 * Writes attn output (post proj) into out[T,1024]. */
static void g4a_attention(g4a_layer *cl, const float *x, float *out, int T, const float *pe) {
    int nb = (T + G4A_CHUNK - 1) / G4A_CHUNK;
    float *q = (float *)malloc((size_t)T * G4A_HID * sizeof(float));
    float *k = (float *)malloc((size_t)T * G4A_HID * sizeof(float));
    float *v = (float *)malloc((size_t)T * G4A_HID * sizeof(float));
    for (int t = 0; t < T; t++) {
        g4a_matmul(cl->q_w, x + (size_t)t * G4A_HID, q + (size_t)t * G4A_HID, G4A_HID, G4A_HID);
        g4a_matmul(cl->k_w, x + (size_t)t * G4A_HID, k + (size_t)t * G4A_HID, G4A_HID, G4A_HID);
        g4a_matmul(cl->v_w, x + (size_t)t * G4A_HID, v + (size_t)t * G4A_HID, G4A_HID, G4A_HID);
    }
    float q_scale = (float)(pow(G4A_HDIM, -0.5) / log(2.0));
    float k_scale = (float)(log(1.0 + exp(1.0)) / log(2.0));
    /* scale q by q_scale*softplus(per_dim_scale[d]); k by k_scale */
    for (int t = 0; t < T; t++) for (int h = 0; h < G4A_HEADS; h++) for (int d = 0; d < G4A_HDIM; d++) {
        q[(size_t)t * G4A_HID + h * G4A_HDIM + d] *= q_scale * g4a_softplus(cl->per_dim_scale[d]);
        k[(size_t)t * G4A_HID + h * G4A_HDIM + d] *= k_scale;
    }
    /* relative key: rel_k_proj(pe) -> [13, H, hd] */
    float *relk = (float *)malloc(13 * G4A_HID * sizeof(float));
    for (int p = 0; p < 13; p++) g4a_matmul(cl->rel_k_w, pe + (size_t)p * G4A_HID, relk + (size_t)p * G4A_HID, G4A_HID, G4A_HID);

    memset(out, 0, (size_t)T * G4A_HID * sizeof(float));
    float logits[G4A_CTX], bd_raw[G4A_CHUNK * 13];
    for (int h = 0; h < G4A_HEADS; h++) {
        for (int b = 0; b < nb; b++) {
            /* matrix_bd raw for this block: [chunk,13] = q . relk per head */
            for (int i = 0; i < G4A_CHUNK; i++) {
                int g = b * G4A_CHUNK + i;
                for (int p = 0; p < 13; p++) {
                    double s = 0;
                    if (g < T) { const float *qd = q + (size_t)g * G4A_HID + h * G4A_HDIM;
                                 const float *rd = relk + (size_t)p * G4A_HID + h * G4A_HDIM;
                                 for (int d = 0; d < G4A_HDIM; d++) s += (double)qd[d] * rd[d]; }
                    bd_raw[i * 13 + p] = (float)s;
                }
            }
            for (int i = 0; i < G4A_CHUNK; i++) {
                int g = b * G4A_CHUNK + i;
                if (g >= T) continue;
                const float *qd = q + (size_t)g * G4A_HID + h * G4A_HDIM;
                /* logits over context c in [0,24) */
                int any = 0;
                for (int c = 0; c < G4A_CTX; c++) {
                    int gk = b * G4A_CHUNK + c - G4A_PAST;   /* global key */
                    /* sliding window: valid iff 0 <= q-kv < 12  ->  i < c <= i+12 */
                    int valid = (gk >= 0 && gk < T && c > i && c <= i + G4A_PAST);
                    if (!valid) { logits[c] = -1e30f; continue; }
                    const float *kd = k + (size_t)gk * G4A_HID + h * G4A_HDIM;
                    double ac = 0; for (int d = 0; d < G4A_HDIM; d++) ac += (double)qd[d] * kd[d];
                    /* matrix_bd via rel_shift: idx=i*24+c; row=idx/25; col=idx%25; bd_raw[row][col] if col<13 */
                    int idx = i * G4A_CTX + c, rr = idx / 25, cc = idx % 25;
                    float bd = (cc < 13 && rr < G4A_CHUNK) ? bd_raw[rr * 13 + cc] : 0.0f;
                    float lg = (float)ac + bd;
                    lg = tanhf(lg / G4A_SOFTCAP) * G4A_SOFTCAP;
                    logits[c] = lg; any = 1;
                }
                if (!any) continue;
                float mx = -1e30f; for (int c = 0; c < G4A_CTX; c++) if (logits[c] > mx) mx = logits[c];
                float sum = 0; for (int c = 0; c < G4A_CTX; c++) { logits[c] = expf(logits[c] - mx); sum += logits[c]; }
                float inv = 1.0f / sum;
                float *od = out + (size_t)g * G4A_HID + h * G4A_HDIM;
                for (int c = 0; c < G4A_CTX; c++) {
                    int gk = b * G4A_CHUNK + c - G4A_PAST;
                    if (gk < 0 || gk >= T || c <= i || c > i + G4A_PAST) continue;
                    float a = logits[c] * inv;
                    const float *vd = v + (size_t)gk * G4A_HID + h * G4A_HDIM;
                    for (int d = 0; d < G4A_HDIM; d++) od[d] += a * vd[d];
                }
            }
        }
    }
    free(q); free(k); free(v); free(relk);
    /* post projection in-place via temp */
    float *tmp = (float *)malloc(G4A_HID * sizeof(float));
    for (int t = 0; t < T; t++) {
        g4a_matmul(cl->post_w, out + (size_t)t * G4A_HID, tmp, G4A_HID, G4A_HID);
        memcpy(out + (size_t)t * G4A_HID, tmp, G4A_HID * sizeof(float));
    }
    free(tmp);
}

/* ---------- lconv1d module, in-place on hs[T,1024] ---------- */
static void g4a_lconv(g4a_layer *cl, float *hs, int T) {
    float *glu = (float *)malloc((size_t)T * G4A_HID * sizeof(float));   /* after GLU [T,1024] */
    float *st = (float *)malloc(2 * G4A_HID * sizeof(float));
    float *nrm = (float *)malloc(G4A_HID * sizeof(float));
    for (int t = 0; t < T; t++) {
        memcpy(nrm, hs + (size_t)t * G4A_HID, G4A_HID * sizeof(float));
        g4a_rmsnorm(nrm, cl->lc_pre, G4A_HID);
        g4a_matmul(cl->lc_start, nrm, st, 2 * G4A_HID, G4A_HID);  /* [2048] */
        for (int d = 0; d < G4A_HID; d++) glu[(size_t)t * G4A_HID + d] = st[d] * (1.0f / (1.0f + expf(-st[G4A_HID + d])));
    }
    /* depthwise causal conv1d (kernel 5, left pad 4) over time, per channel */
    float *cv = (float *)malloc((size_t)T * G4A_HID * sizeof(float));
    for (int t = 0; t < T; t++) for (int d = 0; d < G4A_HID; d++) {
        float acc = 0; const float *wf = cl->lc_dw + (size_t)d * G4A_KCONV;
        for (int kk = 0; kk < G4A_KCONV; kk++) { int tt = t - (G4A_KCONV - 1) + kk;
            if (tt >= 0) acc += glu[(size_t)tt * G4A_HID + d] * wf[kk]; }
        cv[(size_t)t * G4A_HID + d] = acc;
    }
    free(glu);
    for (int t = 0; t < T; t++) {
        float *c = cv + (size_t)t * G4A_HID;
        g4a_rmsnorm(c, cl->lc_conv_norm, G4A_HID);
        for (int d = 0; d < G4A_HID; d++) c[d] = g4a_silu(c[d]);
        g4a_matmul(cl->lc_end, c, nrm, G4A_HID, G4A_HID);
        for (int d = 0; d < G4A_HID; d++) hs[(size_t)t * G4A_HID + d] += nrm[d];
    }
    free(cv); free(st); free(nrm);
}

float *g4a_encode(g4a_model *m, const float *mel, int n_frames, int *out_tokens, int *out_dim) {
    float *hs = NULL;
    int T = g4a_subsample(m, mel, n_frames, &hs);   /* [T,1024] */
    float pe[13 * G4A_HID];
    g4a_rel_pos(pe);

    float *attn = (float *)malloc((size_t)T * G4A_HID * sizeof(float));
    float *tmp_h = (float *)malloc(G4A_FFN * sizeof(float));
    float *tmp_n = (float *)malloc(G4A_HID * sizeof(float));
    for (int L = 0; L < G4A_LAYERS; L++) {
        g4a_layer *cl = &m->layers[L];
        for (int t = 0; t < T; t++) g4a_ffn(cl->ff1_w1, cl->ff1_w2, cl->ff1_pre, cl->ff1_post, hs + (size_t)t * G4A_HID, tmp_h, tmp_n);
        /* attention block: residual = hs; hs = norm_pre_attn(hs); attn; norm_post_attn(attn); hs = residual + attn */
        float *resid = (float *)malloc((size_t)T * G4A_HID * sizeof(float));
        memcpy(resid, hs, (size_t)T * G4A_HID * sizeof(float));
        for (int t = 0; t < T; t++) g4a_rmsnorm(hs + (size_t)t * G4A_HID, cl->norm_pre_attn, G4A_HID);
        g4a_attention(cl, hs, attn, T, pe);
        for (int t = 0; t < T; t++) { g4a_rmsnorm(attn + (size_t)t * G4A_HID, cl->norm_post_attn, G4A_HID);
            for (int d = 0; d < G4A_HID; d++) hs[(size_t)t * G4A_HID + d] = resid[(size_t)t * G4A_HID + d] + attn[(size_t)t * G4A_HID + d]; }
        free(resid);
        g4a_lconv(cl, hs, T);
        for (int t = 0; t < T; t++) g4a_ffn(cl->ff2_w1, cl->ff2_w2, cl->ff2_pre, cl->ff2_post, hs + (size_t)t * G4A_HID, tmp_h, tmp_n);
        for (int t = 0; t < T; t++) g4a_rmsnorm(hs + (size_t)t * G4A_HID, cl->norm_out, G4A_HID);
    }
    free(attn); free(tmp_h); free(tmp_n);

    /* output_proj [1536,1024] + bias, then embed_audio: RMSNorm(no scale) -> embedding_projection [1536,1536] */
    float *soft = (float *)malloc((size_t)T * G4A_OUT * sizeof(float));
    float *tower = (float *)malloc((size_t)T * G4A_OUT * sizeof(float));
    float *tw = (float *)malloc(G4A_OUT * sizeof(float));
    for (int t = 0; t < T; t++) {
        g4a_matmul(m->out_proj_w, hs + (size_t)t * G4A_HID, tw, G4A_OUT, G4A_HID);
        for (int i = 0; i < G4A_OUT; i++) tw[i] += m->out_proj_b[i];
        memcpy(tower + (size_t)t * G4A_OUT, tw, G4A_OUT * sizeof(float));
        /* RMSNorm (no scale) over 1536 */
        double ss = 0; for (int i = 0; i < G4A_OUT; i++) ss += (double)tw[i] * tw[i];
        float inv = (float)(1.0 / sqrt(ss / G4A_OUT + G4A_EPS));
        for (int i = 0; i < G4A_OUT; i++) tw[i] *= inv;
        g4a_matmul(m->embed_proj, tw, soft + (size_t)t * G4A_OUT, G4A_OUT, G4A_OUT);
    }
    { const char *dt = getenv("G4A_DUMP_TOWER");
      if (dt) { FILE *fp = fopen(dt, "wb"); if (fp) { fwrite(tower, sizeof(float), (size_t)T * G4A_OUT, fp); fclose(fp); } } }
    free(tower); free(tw); free(hs);
    *out_tokens = T; *out_dim = G4A_OUT;
    return soft;
}

void g4a_free(g4a_model *m) {
    if (!m) return;
    free(m->conv0_w); free(m->conv0_norm); free(m->conv1_w); free(m->conv1_norm); free(m->in_proj);
    free(m->out_proj_w); free(m->out_proj_b); free(m->embed_proj);
    for (int L = 0; L < G4A_LAYERS; L++) { g4a_layer *c = &m->layers[L];
        free(c->ff1_w1); free(c->ff1_w2); free(c->ff1_pre); free(c->ff1_post);
        free(c->ff2_w1); free(c->ff2_w2); free(c->ff2_pre); free(c->ff2_post);
        free(c->q_w); free(c->k_w); free(c->v_w); free(c->post_w); free(c->rel_k_w); free(c->per_dim_scale);
        free(c->lc_start); free(c->lc_end); free(c->lc_dw); free(c->lc_pre); free(c->lc_conv_norm);
        free(c->norm_pre_attn); free(c->norm_post_attn); free(c->norm_out); }
    free(m);
}

#endif /* GEMMA4_AUDIO_IMPLEMENTATION */
#endif /* GEMMA4_AUDIO_ENCODER_H */
