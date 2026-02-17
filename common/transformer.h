/*
 * transformer.h - Reference C transformer inference for Qwen2-style models
 *
 * Usage:
 *   #define TRANSFORMER_IMPLEMENTATION
 *   #include "transformer.h"
 *
 * Dependencies: gguf_loader.h, ggml_dequant.h (include with IMPLEMENTATION defined separately)
 *
 * API:
 *   transformer_model *transformer_load(gguf_context *gguf, int max_seq_len);
 *   void transformer_free(transformer_model *model);
 *   float *transformer_forward(transformer_model *model, int32_t token_id, int position);
 */
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdint.h>
#include <stddef.h>
#include "gguf_loader.h"
#include "ggml_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Quantized tensor reference (points into GGUF mmap) */
typedef struct {
    void    *data;
    uint32_t type;       /* ggml_dtype */
    int      n_rows;
    int      n_cols;     /* number of elements per row */
} qtensor;

typedef struct {
    qtensor attn_norm;   /* [n_embd] */
    qtensor attn_q;      /* [n_embd, n_embd] */
    qtensor attn_k;      /* [n_kv_dim, n_embd] */
    qtensor attn_v;      /* [n_kv_dim, n_embd] */
    qtensor attn_q_norm; /* [head_dim] */
    qtensor attn_k_norm; /* [head_dim] */
    qtensor attn_output; /* [n_embd, n_embd] */
    qtensor ffn_norm;    /* [n_embd] */
    qtensor ffn_gate;    /* [n_ff, n_embd] */
    qtensor ffn_up;      /* [n_ff, n_embd] */
    qtensor ffn_down;    /* [n_embd, n_ff] */
} transformer_layer;

typedef struct {
    /* Hyperparameters */
    int n_layers;
    int n_embd;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int n_ff;
    int n_vocab;
    int max_seq_len;
    float rope_freq_base;
    float rms_norm_eps;

    int has_lm_head;     /* 1 if output.weight exists (generative model) */

    /* Global tensors */
    qtensor token_embd;  /* [n_vocab, n_embd] */
    qtensor output_norm; /* [n_embd] */
    qtensor output;      /* [n_vocab, n_embd] LM head (may be absent for embedding models) */

    /* Per-layer weights */
    transformer_layer *layers;

    /* KV cache: [n_layers][max_seq_len * n_kv_heads * head_dim] */
    float **key_cache;
    float **value_cache;

    /* Scratch buffers */
    float *x;        /* [n_embd] current hidden state */
    float *xb;       /* [n_embd] scratch after norm */
    float *xb2;      /* [n_embd] scratch for residual */
    float *q;        /* [n_embd] query */
    float *k;        /* [n_kv_dim] key */
    float *v;        /* [n_kv_dim] value */
    float *att;      /* [n_heads * max_seq_len] attention scores */
    float *ffn_buf1; /* [n_ff] */
    float *ffn_buf2; /* [n_ff] */
    float *ffn_buf3; /* [n_ff] */
    float *logits;     /* [n_vocab] output logits (only if has_lm_head) */
    float *matvec_tmp; /* max(n_embd, n_ff) for row dequant */
} transformer_model;

transformer_model *transformer_load(gguf_context *gguf, int max_seq_len);
void transformer_free(transformer_model *model);

/* Run one token through the transformer. Returns pointer to hidden state [n_embd].
 * For embedding models (no output projection), this is the final hidden state. */
float *transformer_forward(transformer_model *model, int32_t token_id, int position);

/* Run forward pass and compute logits [n_vocab]. Returns NULL if no LM head.
 * The returned pointer is valid until the next call. */
float *transformer_forward_logits(transformer_model *model, int32_t token_id, int position);

/* Sample next token from logits using temperature and top-k. */
int32_t transformer_sample_topk(const float *logits, int n_vocab, float temperature, int top_k);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef TRANSFORMER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Tensor lookup helpers ---- */

static int tf_find_tensor(const gguf_context *gguf, const char *name) {
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        if (strcmp(gguf->tensors[i].name.str, name) == 0) return (int)i;
    }
    return -1;
}

static qtensor tf_load_tensor(const gguf_context *gguf, const char *name, int required) {
    qtensor t = {0};
    int idx = tf_find_tensor(gguf, name);
    if (idx < 0) {
        if (required) fprintf(stderr, "transformer: missing tensor '%s'\n", name);
        return t;
    }
    t.data = gguf_tensor_data(gguf, idx);
    t.type = gguf->tensors[idx].type;
    /* dims[0] = inner dimension (cols), dims[1] = outer (rows) for 2D */
    t.n_cols = (int)gguf->tensors[idx].dims[0];
    t.n_rows = (gguf->tensors[idx].n_dims >= 2) ? (int)gguf->tensors[idx].dims[1] : 1;
    return t;
}

static int tf_get_int(const gguf_context *gguf, const char *key, int default_val) {
    int idx = gguf_find_key(gguf, key);
    if (idx < 0) return default_val;
    if (gguf->kv[idx].type == GGUF_TYPE_UINT32) return (int)gguf->kv[idx].value.u32;
    if (gguf->kv[idx].type == GGUF_TYPE_INT32)  return gguf->kv[idx].value.i32;
    return default_val;
}

static float tf_get_float(const gguf_context *gguf, const char *key, float default_val) {
    int idx = gguf_find_key(gguf, key);
    if (idx < 0) return default_val;
    if (gguf->kv[idx].type == GGUF_TYPE_FLOAT32) return gguf->kv[idx].value.f32;
    return default_val;
}

/* ---- Compute helpers ---- */

/* Dequantize one row of a quantized matrix.
 * For a matrix stored as [n_rows, n_cols], row i starts at offset computed
 * from the quantization block size. */
static void tf_dequant_row(const qtensor *t, int row, float *dst) {
    /* Compute byte offset for this row */
    int n_cols = t->n_cols;
    int block_size, type_size;

    /* Get block/type size from ggml_type_info (defined in gguf_loader.h impl) */
    switch (t->type) {
        case GGML_TYPE_Q8_0: block_size = 32;  type_size = 34;  break;
        case GGML_TYPE_Q4_K: block_size = 256; type_size = 144; break;
        case GGML_TYPE_Q6_K: block_size = 256; type_size = 210; break;
        case GGML_TYPE_F32:  block_size = 1;   type_size = 4;   break;
        case GGML_TYPE_F16:  block_size = 1;   type_size = 2;   break;
        default:
            fprintf(stderr, "tf_dequant_row: unsupported type %u\n", t->type);
            memset(dst, 0, n_cols * sizeof(float));
            return;
    }

    size_t row_bytes = (size_t)((n_cols + block_size - 1) / block_size) * type_size;
    const void *row_data = (const uint8_t *)t->data + row * row_bytes;
    dequant_row(t->type, row_data, dst, n_cols);
}

/* RMSNorm: y[i] = x[i] * w[i] / sqrt(mean(x^2) + eps) */
static void tf_rmsnorm(float *dst, const float *x, const qtensor *w, int n, float eps, float *w_buf) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);

    /* Dequant weight */
    tf_dequant_row(w, 0, w_buf);
    for (int i = 0; i < n; i++) dst[i] = x[i] * ss * w_buf[i];
}

/* Quantized matrix-vector multiply: dst[i] = sum_j(M[i][j] * x[j]) for i in [0, n_rows) */
static void tf_qmatvec(float *dst, const qtensor *mat, const float *x, int n_rows, float *tmp) {
    int n_cols = mat->n_cols;
    for (int i = 0; i < n_rows; i++) {
        tf_dequant_row(mat, i, tmp);
        float sum = 0.0f;
        for (int j = 0; j < n_cols; j++) sum += tmp[j] * x[j];
        dst[i] = sum;
    }
}

/* RoPE: apply rotary position encoding to a vector of shape [n_heads, head_dim] */
static void tf_rope(float *vec, int n_heads, int head_dim, int pos, float freq_base) {
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(freq_base, (float)i / head_dim);
            float theta = pos * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            float v0 = v[i];
            float v1 = v[i + 1];
            v[i]     = v0 * cos_t - v1 * sin_t;
            v[i + 1] = v0 * sin_t + v1 * cos_t;
        }
    }
}

/* Per-head RMSNorm (QK-norm): normalize each head_dim-sized chunk */
static void tf_qk_norm(float *vec, int n_heads, int head_dim, const qtensor *norm_w, float eps, float *w_buf) {
    tf_dequant_row(norm_w, 0, w_buf);
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
        float ss = 0.0f;
        for (int i = 0; i < head_dim; i++) ss += v[i] * v[i];
        ss = 1.0f / sqrtf(ss / head_dim + eps);
        for (int i = 0; i < head_dim; i++) v[i] = v[i] * ss * w_buf[i];
    }
}

/* Softmax in-place over n elements */
static void tf_softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* ---- Load ---- */

transformer_model *transformer_load(gguf_context *gguf, int max_seq_len) {
    if (!gguf) return NULL;

    transformer_model *m = (transformer_model *)calloc(1, sizeof(transformer_model));
    if (!m) return NULL;

    /* Detect architecture prefix: try qwen3 first, fall back to qwen2 */
    const char *arch = "qwen2";
    if (gguf_find_key(gguf, "qwen3.block_count") >= 0) arch = "qwen3";

    char kbuf[128];
    #define ARCH_KEY(suffix) (snprintf(kbuf, sizeof(kbuf), "%s." suffix, arch), kbuf)

    /* Read hyperparameters from GGUF metadata */
    m->n_embd      = tf_get_int(gguf, ARCH_KEY("embedding_length"), 4096);
    m->n_heads     = tf_get_int(gguf, ARCH_KEY("attention.head_count"), 32);
    m->n_kv_heads  = tf_get_int(gguf, ARCH_KEY("attention.head_count_kv"), 8);
    m->n_layers    = tf_get_int(gguf, ARCH_KEY("block_count"), 36);
    m->n_ff        = tf_get_int(gguf, ARCH_KEY("feed_forward_length"), 12288);
    m->n_vocab     = tf_get_int(gguf, ARCH_KEY("vocab_size"), 0);
    m->rms_norm_eps = tf_get_float(gguf, ARCH_KEY("attention.layer_norm_rms_epsilon"), 1e-6f);
    m->rope_freq_base = tf_get_float(gguf, ARCH_KEY("rope.freq_base"), 5000000.0f);
    #undef ARCH_KEY
    fprintf(stderr, "transformer: architecture=%s\n", arch);
    m->head_dim    = m->n_embd / m->n_heads;
    m->max_seq_len = max_seq_len;

    fprintf(stderr, "transformer: n_embd=%d n_heads=%d n_kv_heads=%d n_layers=%d n_ff=%d head_dim=%d\n",
            m->n_embd, m->n_heads, m->n_kv_heads, m->n_layers, m->n_ff, m->head_dim);
    fprintf(stderr, "transformer: rope_freq_base=%.0f rms_norm_eps=%.1e max_seq_len=%d\n",
            m->rope_freq_base, m->rms_norm_eps, max_seq_len);

    /* Global tensors */
    m->token_embd = tf_load_tensor(gguf, "token_embd.weight", 1);
    m->output_norm = tf_load_tensor(gguf, "output_norm.weight", 1);
    m->output = tf_load_tensor(gguf, "output.weight", 0);
    m->has_lm_head = (m->output.data != NULL) ? 1 : 0;

    if (m->n_vocab == 0 && m->token_embd.data) {
        m->n_vocab = m->token_embd.n_rows;
    }
    fprintf(stderr, "transformer: n_vocab=%d has_lm_head=%d\n", m->n_vocab, m->has_lm_head);

    /* Per-layer tensors */
    m->layers = (transformer_layer *)calloc(m->n_layers, sizeof(transformer_layer));
    for (int l = 0; l < m->n_layers; l++) {
        char name[128];
        #define LOAD(field, suffix, req) \
            snprintf(name, sizeof(name), "blk.%d." suffix ".weight", l); \
            m->layers[l].field = tf_load_tensor(gguf, name, req);

        LOAD(attn_norm,    "attn_norm",   1)
        LOAD(attn_q,       "attn_q",      1)
        LOAD(attn_k,       "attn_k",      1)
        LOAD(attn_v,       "attn_v",      1)
        LOAD(attn_q_norm,  "attn_q_norm", 0)
        LOAD(attn_k_norm,  "attn_k_norm", 0)
        LOAD(attn_output,  "attn_output", 1)
        LOAD(ffn_norm,     "ffn_norm",    1)
        LOAD(ffn_gate,     "ffn_gate",    1)
        LOAD(ffn_up,       "ffn_up",      1)
        LOAD(ffn_down,     "ffn_down",    1)
        #undef LOAD
    }

    /* Allocate KV cache */
    int kv_dim = m->n_kv_heads * m->head_dim;
    m->key_cache   = (float **)calloc(m->n_layers, sizeof(float *));
    m->value_cache = (float **)calloc(m->n_layers, sizeof(float *));
    for (int l = 0; l < m->n_layers; l++) {
        m->key_cache[l]   = (float *)calloc(max_seq_len * kv_dim, sizeof(float));
        m->value_cache[l] = (float *)calloc(max_seq_len * kv_dim, sizeof(float));
    }

    /* Allocate scratch buffers */
    int max_dim = m->n_embd > m->n_ff ? m->n_embd : m->n_ff;
    m->x         = (float *)calloc(m->n_embd, sizeof(float));
    m->xb        = (float *)calloc(m->n_embd, sizeof(float));
    m->xb2       = (float *)calloc(m->n_embd, sizeof(float));
    m->q         = (float *)calloc(m->n_embd, sizeof(float));
    m->k         = (float *)calloc(kv_dim, sizeof(float));
    m->v         = (float *)calloc(kv_dim, sizeof(float));
    m->att       = (float *)calloc(m->n_heads * max_seq_len, sizeof(float));
    m->ffn_buf1  = (float *)calloc(m->n_ff, sizeof(float));
    m->ffn_buf2  = (float *)calloc(m->n_ff, sizeof(float));
    m->ffn_buf3  = (float *)calloc(m->n_ff, sizeof(float));
    m->logits     = m->has_lm_head ? (float *)calloc(m->n_vocab, sizeof(float)) : NULL;
    m->matvec_tmp = (float *)calloc(max_dim, sizeof(float));

    return m;
}

void transformer_free(transformer_model *model) {
    if (!model) return;
    free(model->layers);
    if (model->key_cache) {
        for (int l = 0; l < model->n_layers; l++) {
            free(model->key_cache[l]);
            free(model->value_cache[l]);
        }
        free(model->key_cache);
        free(model->value_cache);
    }
    free(model->x);
    free(model->xb);
    free(model->xb2);
    free(model->q);
    free(model->k);
    free(model->v);
    free(model->att);
    free(model->ffn_buf1);
    free(model->ffn_buf2);
    free(model->ffn_buf3);
    free(model->logits);
    free(model->matvec_tmp);
    free(model);
}

/* ---- Forward pass ---- */

float *transformer_forward(transformer_model *model, int32_t token_id, int position) {
    transformer_model *m = model;
    int n_embd = m->n_embd;
    int n_heads = m->n_heads;
    int n_kv_heads = m->n_kv_heads;
    int head_dim = m->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int gqa_ratio = n_heads / n_kv_heads;

    /* 1. Embedding lookup: dequantize one row of token_embd */
    tf_dequant_row(&m->token_embd, token_id, m->x);

    /* 2. Transformer blocks */
    for (int l = 0; l < m->n_layers; l++) {
        transformer_layer *layer = &m->layers[l];

        /* --- Attention --- */
        /* RMSNorm */
        tf_rmsnorm(m->xb, m->x, &layer->attn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);

        /* Q/K/V projections */
        tf_qmatvec(m->q, &layer->attn_q, m->xb, n_embd, m->matvec_tmp);
        tf_qmatvec(m->k, &layer->attn_k, m->xb, kv_dim, m->matvec_tmp);
        tf_qmatvec(m->v, &layer->attn_v, m->xb, kv_dim, m->matvec_tmp);

        /* QK-Norm (if present) */
        if (layer->attn_q_norm.data) {
            tf_qk_norm(m->q, n_heads, head_dim, &layer->attn_q_norm, m->rms_norm_eps, m->matvec_tmp);
        }
        if (layer->attn_k_norm.data) {
            tf_qk_norm(m->k, n_kv_heads, head_dim, &layer->attn_k_norm, m->rms_norm_eps, m->matvec_tmp);
        }

        /* RoPE on Q and K */
        tf_rope(m->q, n_heads, head_dim, position, m->rope_freq_base);
        tf_rope(m->k, n_kv_heads, head_dim, position, m->rope_freq_base);

        /* Store K/V into cache at position */
        float *kc = m->key_cache[l]   + position * kv_dim;
        float *vc = m->value_cache[l] + position * kv_dim;
        memcpy(kc, m->k, kv_dim * sizeof(float));
        memcpy(vc, m->v, kv_dim * sizeof(float));

        /* Multi-head attention with GQA */
        int seq_len = position + 1;
        memset(m->xb2, 0, n_embd * sizeof(float));

        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / gqa_ratio;  /* which KV head this query head uses */
            float *q_h = m->q + h * head_dim;
            float *att_h = m->att + h * m->max_seq_len;

            /* Compute attention scores: Q . K^T / sqrt(head_dim) */
            float scale = 1.0f / sqrtf((float)head_dim);
            for (int t = 0; t < seq_len; t++) {
                float *k_t = m->key_cache[l] + t * kv_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) score += q_h[d] * k_t[d];
                att_h[t] = score * scale;
            }

            /* Causal mask not needed for embedding model (process all tokens),
             * but include for correctness in autoregressive mode.
             * Since we're processing position `position`, all t <= position are visible. */

            /* Softmax */
            tf_softmax(att_h, seq_len);

            /* Weighted sum of values */
            float *out_h = m->xb2 + h * head_dim;
            for (int t = 0; t < seq_len; t++) {
                float *v_t = m->value_cache[l] + t * kv_dim + kv_h * head_dim;
                float a = att_h[t];
                for (int d = 0; d < head_dim; d++) out_h[d] += a * v_t[d];
            }
        }

        /* Output projection */
        tf_qmatvec(m->xb, &layer->attn_output, m->xb2, n_embd, m->matvec_tmp);

        /* Residual */
        for (int i = 0; i < n_embd; i++) m->x[i] += m->xb[i];

        /* --- FFN --- */
        /* RMSNorm */
        tf_rmsnorm(m->xb, m->x, &layer->ffn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);

        /* SwiGLU: down @ (silu(gate @ x) * (up @ x)) */
        tf_qmatvec(m->ffn_buf1, &layer->ffn_gate, m->xb, m->n_ff, m->matvec_tmp);
        tf_qmatvec(m->ffn_buf2, &layer->ffn_up,   m->xb, m->n_ff, m->matvec_tmp);

        /* SiLU(x) = x * sigmoid(x) */
        for (int i = 0; i < m->n_ff; i++) {
            float gate = m->ffn_buf1[i];
            gate = gate / (1.0f + expf(-gate));  /* SiLU */
            m->ffn_buf3[i] = gate * m->ffn_buf2[i];
        }

        tf_qmatvec(m->xb, &layer->ffn_down, m->ffn_buf3, n_embd, m->matvec_tmp);

        /* Residual */
        for (int i = 0; i < n_embd; i++) m->x[i] += m->xb[i];

        if (l == 0 || l == m->n_layers - 1 || (l + 1) % 10 == 0) {
            float norm = 0.0f;
            for (int i = 0; i < n_embd; i++) norm += m->x[i] * m->x[i];
            fprintf(stderr, "  layer %2d: hidden norm = %.4f\n", l, sqrtf(norm));
        }
    }

    /* Final RMSNorm */
    tf_rmsnorm(m->x, m->x, &m->output_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);

    return m->x;
}

float *transformer_forward_logits(transformer_model *model, int32_t token_id, int position) {
    float *hidden = transformer_forward(model, token_id, position);
    if (!model->has_lm_head || !hidden) return NULL;

    /* Project hidden state to vocab: logits[i] = output_weight[i] . hidden */
    tf_qmatvec(model->logits, &model->output, hidden, model->n_vocab, model->matvec_tmp);
    return model->logits;
}

/* Top-k sampling with temperature */
int32_t transformer_sample_topk(const float *logits, int n_vocab, float temperature, int top_k) {
    if (top_k <= 0 || top_k > n_vocab) top_k = n_vocab;

    /* Find top-k indices by partial sort */
    int32_t *indices = (int32_t *)malloc(top_k * sizeof(int32_t));
    float *vals = (float *)malloc(top_k * sizeof(float));
    int k = 0;

    for (int i = 0; i < n_vocab; i++) {
        float v = logits[i] / temperature;
        if (k < top_k) {
            indices[k] = i;
            vals[k] = v;
            k++;
            /* Bubble up to maintain min-heap property at vals[0] */
            for (int j = k - 1; j > 0; j--) {
                int p = (j - 1) / 2;
                if (vals[j] < vals[p]) {
                    float tv = vals[j]; vals[j] = vals[p]; vals[p] = tv;
                    int32_t ti = indices[j]; indices[j] = indices[p]; indices[p] = ti;
                }
            }
        } else if (v > vals[0]) {
            vals[0] = v;
            indices[0] = i;
            /* Sift down */
            int j = 0;
            for (;;) {
                int s = j, l = 2*j+1, r = 2*j+2;
                if (l < top_k && vals[l] < vals[s]) s = l;
                if (r < top_k && vals[r] < vals[s]) s = r;
                if (s == j) break;
                float tv = vals[j]; vals[j] = vals[s]; vals[s] = tv;
                int32_t ti = indices[j]; indices[j] = indices[s]; indices[s] = ti;
                j = s;
            }
        }
    }

    /* Softmax over top-k */
    float max_v = vals[0];
    for (int i = 1; i < top_k; i++) if (vals[i] > max_v) max_v = vals[i];
    float sum = 0.0f;
    for (int i = 0; i < top_k; i++) { vals[i] = expf(vals[i] - max_v); sum += vals[i]; }
    for (int i = 0; i < top_k; i++) vals[i] /= sum;

    /* Sample */
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    int32_t result = (top_k > 0) ? indices[0] : 0;
    for (int i = 0; i < top_k; i++) {
        cum += vals[i];
        if (r <= cum) { result = indices[i]; break; }
    }

    free(indices);
    free(vals);
    return result;
}

#endif /* TRANSFORMER_IMPLEMENTATION */
#endif /* TRANSFORMER_H */
