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
    int mrope_sections[4]; /* M-RoPE dimension sections [temporal, height, width, pad] */
    int use_mrope;         /* 1 if M-RoPE is enabled (Qwen3-VL) */
    int n_deepstack;       /* number of deepstack layers (Qwen3-VL) */
    const float *ds_embd;  /* pointer to current full embedding (incl deepstack slices) */
    int ds_embd_stride;    /* total embedding dim = proj_dim * (1 + n_deepstack) */

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
    float *matvec_tmp; /* max(n_embd, n_ff) for row dequant (thread 0) */

    /* Multi-threading */
    int n_threads;           /* number of threads (default: 1) */
    float **thread_tmp;      /* per-thread dequant scratch [n_threads][max_dim] */
} transformer_model;

transformer_model *transformer_load(gguf_context *gguf, int max_seq_len);
void transformer_free(transformer_model *model);

/* Set number of threads for parallel matmul/attention (default: 1) */
void transformer_set_threads(transformer_model *model, int n_threads);

/* Run one token through the transformer. Returns pointer to hidden state [n_embd].
 * For embedding models (no output projection), this is the final hidden state. */
float *transformer_forward(transformer_model *model, int32_t token_id, int position);

/* Run forward pass and compute logits [n_vocab]. Returns NULL if no LM head.
 * The returned pointer is valid until the next call. */
float *transformer_forward_logits(transformer_model *model, int32_t token_id, int position);

/* Run forward pass with a pre-computed embedding vector instead of token lookup.
 * Used to inject vision embeddings into the sequence. */
float *transformer_forward_embd(transformer_model *model, const float *embd, int position);
float *transformer_forward_embd_logits(transformer_model *model, const float *embd, int position);

/* M-RoPE variants: cache_pos = KV cache slot, pos_t/h/w = RoPE temporal/height/width.
 * For text tokens, use cache_pos = pos_t = pos_h = pos_w = position. */
float *transformer_forward_pos(transformer_model *model, int32_t token_id, int cache_pos, int pos_t, int pos_h, int pos_w);
float *transformer_forward_logits_pos(transformer_model *model, int32_t token_id, int cache_pos, int pos_t, int pos_h, int pos_w);
float *transformer_forward_embd_pos(transformer_model *model, const float *embd, int cache_pos, int pos_t, int pos_h, int pos_w);
float *transformer_forward_embd_logits_pos(transformer_model *model, const float *embd, int cache_pos, int pos_t, int pos_h, int pos_w);

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
#include <pthread.h>

/* Profiling macros: active only if profiler.h was included before this file */
#ifdef PROFILER_H
#define TF_PROF_BEGIN(name, layer, op, prec) prof_begin(name, "llm", layer, op, prec)
#define TF_PROF_END(name, flops, iops) prof_end(name, flops, iops)
#else
#define TF_PROF_BEGIN(name, layer, op, prec) ((void)0)
#define TF_PROF_END(name, flops, iops) ((void)0)
#endif

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
typedef struct {
    float *dst;
    const qtensor *mat;
    const float *x;
    int row_start, row_end;
    float *tmp; /* per-thread scratch */
} tf_matvec_task;

static void *tf_qmatvec_worker(void *arg) {
    tf_matvec_task *t = (tf_matvec_task *)arg;
    int n_cols = t->mat->n_cols;
    /* F16 fast path: use multi-row when chunk is large enough for cache amortization */
    if (t->mat->type == GGML_TYPE_F16) {
        const uint8_t *base = (const uint8_t *)t->mat->data;
        size_t row_bytes = (size_t)n_cols * 2;
        int i = t->row_start;
        int chunk = t->row_end - t->row_start;
        if (chunk >= 3072) {
            for (; i + 5 < t->row_end; i += 6) {
                matvec_f16_6row(t->dst + i,
                    (const uint16_t *)(base + (size_t)i * row_bytes),
                    (const uint16_t *)(base + (size_t)(i+1) * row_bytes),
                    (const uint16_t *)(base + (size_t)(i+2) * row_bytes),
                    (const uint16_t *)(base + (size_t)(i+3) * row_bytes),
                    (const uint16_t *)(base + (size_t)(i+4) * row_bytes),
                    (const uint16_t *)(base + (size_t)(i+5) * row_bytes),
                    t->x, n_cols);
            }
        }
        for (; i < t->row_end; i++) {
            const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
            t->dst[i] = vec_dot_f16_f32(row, t->x, n_cols);
        }
        return NULL;
    }
    for (int i = t->row_start; i < t->row_end; i++) {
        tf_dequant_row(t->mat, i, t->tmp);
        float sum = 0.0f;
        for (int j = 0; j < n_cols; j++) sum += t->tmp[j] * t->x[j];
        t->dst[i] = sum;
    }
    return NULL;
}

static void tf_qmatvec(float *dst, const qtensor *mat, const float *x, int n_rows, float *tmp) {
    int n_cols = mat->n_cols;
    /* F16 fast path: use multi-row when row count is large enough */
    if (mat->type == GGML_TYPE_F16) {
        const uint8_t *base = (const uint8_t *)mat->data;
        size_t row_bytes = (size_t)n_cols * 2;
        int i = 0;
        if (n_rows >= 3072) {
            for (; i + 5 < n_rows; i += 6) {
                matvec_f16_6row(dst + i,
                    (const uint16_t *)(base + (size_t)i * row_bytes),
                    (const uint16_t *)(base + (size_t)(i+1) * row_bytes),
                    (const uint16_t *)(base + (size_t)(i+2) * row_bytes),
                    (const uint16_t *)(base + (size_t)(i+3) * row_bytes),
                    (const uint16_t *)(base + (size_t)(i+4) * row_bytes),
                    (const uint16_t *)(base + (size_t)(i+5) * row_bytes),
                    x, n_cols);
            }
        }
        for (; i < n_rows; i++) {
            const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
            dst[i] = vec_dot_f16_f32(row, x, n_cols);
        }
        return;
    }
    for (int i = 0; i < n_rows; i++) {
        tf_dequant_row(mat, i, tmp);
        float sum = 0.0f;
        for (int j = 0; j < n_cols; j++) sum += tmp[j] * x[j];
        dst[i] = sum;
    }
}

/* Multi-threaded version */
static void tf_qmatvec_mt(float *dst, const qtensor *mat, const float *x, int n_rows,
                           int n_threads, float **thread_tmp) {
    if (n_threads <= 1 || n_rows < n_threads * 4) {
        tf_qmatvec(dst, mat, x, n_rows, thread_tmp[0]);
        return;
    }
    pthread_t *threads = (pthread_t *)alloca(n_threads * sizeof(pthread_t));
    tf_matvec_task *tasks = (tf_matvec_task *)alloca(n_threads * sizeof(tf_matvec_task));
    int rows_per = n_rows / n_threads;
    int extra = n_rows % n_threads;
    int offset = 0;
    for (int t = 0; t < n_threads; t++) {
        int count = rows_per + (t < extra ? 1 : 0);
        tasks[t] = (tf_matvec_task){dst, mat, x, offset, offset + count, thread_tmp[t]};
        offset += count;
        pthread_create(&threads[t], NULL, tf_qmatvec_worker, &tasks[t]);
    }
    for (int t = 0; t < n_threads; t++) pthread_join(threads[t], NULL);
}

/* RoPE: apply rotary position encoding to a vector of shape [n_heads, head_dim] */
/* NeoX-style RoPE: pairs (v[j], v[j + half_dim]) instead of consecutive (v[2j], v[2j+1]) */
static void tf_rope(float *vec, int n_heads, int head_dim, int pos, float freq_base) {
    int half_dim = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
        for (int j = 0; j < half_dim; j++) {
            float freq = 1.0f / powf(freq_base, (float)(2 * j) / head_dim);
            float theta = pos * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            float v0 = v[j];
            float v1 = v[j + half_dim];
            v[j]            = v0 * cos_t - v1 * sin_t;
            v[j + half_dim] = v0 * sin_t + v1 * cos_t;
        }
    }
}

/* M-RoPE (IMROPE): apply rotary position encoding with interleaved sections.
 * sections[4] = [temporal, height, width, pad], positions = [pos_t, pos_h, pos_w].
 * Dimension pairs are assigned to sections via interleaved pattern (sector % 3). */
/* M-RoPE (IMROPE) with NeoX-style rotation: pairs (v[j], v[j + half_dim]).
 * Cache index i0 = 2*j determines the sector and frequency.
 * sector = (i0/2) % sect_dims = j % sect_dims. */
static void tf_rope_mrope(float *vec, int n_heads, int head_dim, int pos_t, int pos_h, int pos_w,
                           float freq_base, const int sections[4]) {
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    if (sect_dims <= 0) sect_dims = head_dim / 2; /* fallback */
    int half_dim = head_dim / 2;

    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
        for (int j = 0; j < half_dim; j++) {
            int sector = j % sect_dims;
            int pos;
            /* IMROPE interleaved pattern (matches llama.cpp) */
            if (sector % 3 == 1 && sector < 3 * sections[1]) {
                pos = pos_h;
            } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
                pos = pos_w;
            } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
                pos = pos_t;
            } else {
                pos = pos_t; /* extra/padding: use temporal */
            }
            float freq = 1.0f / powf(freq_base, (float)(2 * j) / head_dim);
            float theta = pos * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            float v0 = v[j];
            float v1 = v[j + half_dim];
            v[j]            = v0 * cos_t - v1 * sin_t;
            v[j + half_dim] = v0 * sin_t + v1 * cos_t;
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

/* Multi-head attention worker for threading */
typedef struct {
    const float *q;          /* full Q buffer */
    float *att;              /* full attention scores buffer */
    float *xb2;              /* full output buffer (each head writes its own slice) */
    const float *key_cache;  /* layer key cache */
    const float *value_cache;/* layer value cache */
    int head_start, head_end;
    int head_dim, kv_dim, gqa_ratio, seq_len, max_seq_len;
    float scale;
} tf_attn_task;

static void *tf_attn_worker(void *arg) {
    tf_attn_task *t = (tf_attn_task *)arg;
    for (int h = t->head_start; h < t->head_end; h++) {
        int kv_h = h / t->gqa_ratio;
        const float *q_h = t->q + h * t->head_dim;
        float *att_h = t->att + h * t->max_seq_len;

        for (int p = 0; p < t->seq_len; p++) {
            const float *k_p = t->key_cache + p * t->kv_dim + kv_h * t->head_dim;
            float score = 0.0f;
            for (int d = 0; d < t->head_dim; d++) score += q_h[d] * k_p[d];
            att_h[p] = score * t->scale;
        }

        /* Softmax */
        float max_val = att_h[0];
        for (int i = 1; i < t->seq_len; i++) if (att_h[i] > max_val) max_val = att_h[i];
        float sum = 0.0f;
        for (int i = 0; i < t->seq_len; i++) { att_h[i] = expf(att_h[i] - max_val); sum += att_h[i]; }
        for (int i = 0; i < t->seq_len; i++) att_h[i] /= sum;

        /* Weighted sum of values */
        float *out_h = t->xb2 + h * t->head_dim;
        memset(out_h, 0, t->head_dim * sizeof(float));
        for (int p = 0; p < t->seq_len; p++) {
            const float *v_p = t->value_cache + p * t->kv_dim + kv_h * t->head_dim;
            float a = att_h[p];
            for (int d = 0; d < t->head_dim; d++) out_h[d] += a * v_p[d];
        }
    }
    return NULL;
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

    /* Detect architecture prefix: try qwen3vl, qwen3, fall back to qwen2 */
    const char *arch = "qwen2";
    if (gguf_find_key(gguf, "qwen3vl.block_count") >= 0) arch = "qwen3vl";
    else if (gguf_find_key(gguf, "qwen3.block_count") >= 0) arch = "qwen3";

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

    /* M-RoPE sections (Qwen3-VL) */
    m->use_mrope = 0;
    memset(m->mrope_sections, 0, sizeof(m->mrope_sections));
    {
        int idx = gguf_find_key(gguf, ARCH_KEY("rope.dimension_sections"));
        if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_ARRAY &&
            gguf->kv[idx].value.arr.type == GGUF_TYPE_INT32) {
            int n = (int)gguf->kv[idx].value.arr.n;
            if (n > 4) n = 4;
            int32_t *data = (int32_t *)gguf->kv[idx].value.arr.data;
            for (int i = 0; i < n; i++) m->mrope_sections[i] = data[i];
            int sect_sum = m->mrope_sections[0] + m->mrope_sections[1] + m->mrope_sections[2] + m->mrope_sections[3];
            if (sect_sum > 0) {
                m->use_mrope = 1;
                fprintf(stderr, "transformer: M-RoPE sections=[%d, %d, %d, %d]\n",
                        m->mrope_sections[0], m->mrope_sections[1], m->mrope_sections[2], m->mrope_sections[3]);
            }
        }
    }

    /* DeepStack layers (Qwen3-VL) */
    m->n_deepstack = tf_get_int(gguf, ARCH_KEY("n_deepstack_layers"), 0);
    m->ds_embd = NULL;
    m->ds_embd_stride = 0;

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
    if (!m->output.data && m->token_embd.data) {
        /* Weight tying: use token_embd as output projection */
        m->output = m->token_embd;
        fprintf(stderr, "transformer: using weight-tied output (token_embd)\n");
    }
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

    /* Default: single-threaded */
    m->n_threads = 1;
    m->thread_tmp = (float **)calloc(1, sizeof(float *));
    m->thread_tmp[0] = m->matvec_tmp;

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
    /* Free extra per-thread scratch (thread_tmp[0] == matvec_tmp, already freed) */
    if (model->thread_tmp) {
        for (int t = 1; t < model->n_threads; t++) free(model->thread_tmp[t]);
        free(model->thread_tmp);
    }
    free(model);
}

void transformer_set_threads(transformer_model *model, int n_threads) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads == model->n_threads) return;

    /* Free old extra per-thread buffers */
    for (int t = 1; t < model->n_threads; t++) free(model->thread_tmp[t]);
    free(model->thread_tmp);

    int max_dim = model->n_embd > model->n_ff ? model->n_embd : model->n_ff;
    if (model->n_vocab > max_dim) max_dim = model->n_vocab;
    model->n_threads = n_threads;
    model->thread_tmp = (float **)calloc(n_threads, sizeof(float *));
    model->thread_tmp[0] = model->matvec_tmp;
    for (int t = 1; t < n_threads; t++) {
        model->thread_tmp[t] = (float *)calloc(max_dim, sizeof(float));
    }
    fprintf(stderr, "transformer: using %d threads\n", n_threads);
}

/* Helper: apply standard or M-RoPE depending on model config */
static void tf_apply_rope(transformer_model *m, float *q, float *k,
                           int n_heads, int n_kv_heads, int head_dim,
                           int pos_t, int pos_h, int pos_w) {
    if (m->use_mrope) {
        tf_rope_mrope(q, n_heads,    head_dim, pos_t, pos_h, pos_w, m->rope_freq_base, m->mrope_sections);
        tf_rope_mrope(k, n_kv_heads, head_dim, pos_t, pos_h, pos_w, m->rope_freq_base, m->mrope_sections);
    } else {
        tf_rope(q, n_heads,    head_dim, pos_t, m->rope_freq_base);
        tf_rope(k, n_kv_heads, head_dim, pos_t, m->rope_freq_base);
    }
}

/* ---- Forward pass ---- */

/* Internal: forward pass block loop with separate cache position and 3D RoPE positions */
static float *tf_forward_blocks(transformer_model *m, int cache_pos, int pos_t, int pos_h, int pos_w);

/* Forward with M-RoPE: cache_pos = KV cache slot, pos_t/h/w = RoPE temporal/height/width */
float *transformer_forward_pos(transformer_model *model, int32_t token_id, int cache_pos, int pos_t, int pos_h, int pos_w) {
    tf_dequant_row(&model->token_embd, token_id, model->x);
    return tf_forward_blocks(model, cache_pos, pos_t, pos_h, pos_w);
}

float *transformer_forward(transformer_model *model, int32_t token_id, int position) {
    return transformer_forward_pos(model, token_id, position, position, position, position);
}

float *transformer_forward_logits_pos(transformer_model *model, int32_t token_id, int cache_pos, int pos_t, int pos_h, int pos_w) {
    float *hidden = transformer_forward_pos(model, token_id, cache_pos, pos_t, pos_h, pos_w);
    if (!model->has_lm_head || !hidden) return NULL;
    TF_PROF_BEGIN("lm_head", -1, "matvec", "FP32");
    tf_qmatvec_mt(model->logits, &model->output, hidden, model->n_vocab, model->n_threads, model->thread_tmp);
    TF_PROF_END("lm_head", 2.0 * model->n_vocab * model->n_embd, 0);
    return model->logits;
}

float *transformer_forward_logits(transformer_model *model, int32_t token_id, int position) {
    return transformer_forward_logits_pos(model, token_id, position, position, position, position);
}

static float *tf_forward_blocks(transformer_model *m, int position, int pos_t, int pos_h, int pos_w) {
    int n_embd = m->n_embd;
    int n_heads = m->n_heads;
    int n_kv_heads = m->n_kv_heads;
    int head_dim = m->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int gqa_ratio = n_heads / n_kv_heads;

    /* 2. Transformer blocks */
    for (int l = 0; l < m->n_layers; l++) {
        transformer_layer *layer = &m->layers[l];

        /* --- Attention --- */
        /* RMSNorm */
        TF_PROF_BEGIN("attn_norm", l, "rmsnorm", "FP32");
        tf_rmsnorm(m->xb, m->x, &layer->attn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        TF_PROF_END("attn_norm", 5.0 * n_embd, 0);

        /* Q/K/V projections */
        TF_PROF_BEGIN("q_proj", l, "matvec", "FP32");
        tf_qmatvec_mt(m->q, &layer->attn_q, m->xb, n_embd, m->n_threads, m->thread_tmp);
        TF_PROF_END("q_proj", 2.0 * n_embd * n_embd, 0);

        TF_PROF_BEGIN("k_proj", l, "matvec", "FP32");
        tf_qmatvec_mt(m->k, &layer->attn_k, m->xb, kv_dim, m->n_threads, m->thread_tmp);
        TF_PROF_END("k_proj", 2.0 * kv_dim * n_embd, 0);

        TF_PROF_BEGIN("v_proj", l, "matvec", "FP32");
        tf_qmatvec_mt(m->v, &layer->attn_v, m->xb, kv_dim, m->n_threads, m->thread_tmp);
        TF_PROF_END("v_proj", 2.0 * kv_dim * n_embd, 0);

        /* QK-Norm (if present) */
        TF_PROF_BEGIN("qk_norm", l, "rmsnorm", "FP32");
        if (layer->attn_q_norm.data) {
            tf_qk_norm(m->q, n_heads, head_dim, &layer->attn_q_norm, m->rms_norm_eps, m->matvec_tmp);
        }
        if (layer->attn_k_norm.data) {
            tf_qk_norm(m->k, n_kv_heads, head_dim, &layer->attn_k_norm, m->rms_norm_eps, m->matvec_tmp);
        }
        TF_PROF_END("qk_norm", 5.0 * (n_heads + n_kv_heads) * head_dim, 0);

        /* RoPE on Q and K (standard or M-RoPE) */
        TF_PROF_BEGIN("rope", l, "rope", "FP32");
        tf_apply_rope(m, m->q, m->k, n_heads, n_kv_heads, head_dim, pos_t, pos_h, pos_w);
        TF_PROF_END("rope", 8.0 * (n_heads + n_kv_heads) * head_dim / 2, 0);

        /* Store K/V into cache at position */
        float *kc = m->key_cache[l]   + position * kv_dim;
        float *vc = m->value_cache[l] + position * kv_dim;
        memcpy(kc, m->k, kv_dim * sizeof(float));
        memcpy(vc, m->v, kv_dim * sizeof(float));

        /* Multi-head attention with GQA */
        TF_PROF_BEGIN("attention", l, "attention", "FP32");
        int seq_len = position + 1;
        float scale = 1.0f / sqrtf((float)head_dim);

        if (m->n_threads > 1 && n_heads >= m->n_threads) {
            /* Threaded attention */
            int nt = m->n_threads;
            pthread_t *athreads = (pthread_t *)alloca(nt * sizeof(pthread_t));
            tf_attn_task *atasks = (tf_attn_task *)alloca(nt * sizeof(tf_attn_task));
            int heads_per = n_heads / nt;
            int heads_extra = n_heads % nt;
            int hoff = 0;
            for (int t = 0; t < nt; t++) {
                int hcount = heads_per + (t < heads_extra ? 1 : 0);
                atasks[t] = (tf_attn_task){m->q, m->att, m->xb2, m->key_cache[l], m->value_cache[l],
                                           hoff, hoff + hcount, head_dim, kv_dim, gqa_ratio, seq_len,
                                           m->max_seq_len, scale};
                hoff += hcount;
                pthread_create(&athreads[t], NULL, tf_attn_worker, &atasks[t]);
            }
            for (int t = 0; t < nt; t++) pthread_join(athreads[t], NULL);
        } else {
            memset(m->xb2, 0, n_embd * sizeof(float));
            for (int h = 0; h < n_heads; h++) {
                int kv_h = h / gqa_ratio;
                float *q_h = m->q + h * head_dim;
                float *att_h = m->att + h * m->max_seq_len;
                for (int t = 0; t < seq_len; t++) {
                    float *k_t = m->key_cache[l] + t * kv_dim + kv_h * head_dim;
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) score += q_h[d] * k_t[d];
                    att_h[t] = score * scale;
                }
                tf_softmax(att_h, seq_len);
                float *out_h = m->xb2 + h * head_dim;
                for (int t = 0; t < seq_len; t++) {
                    float *v_t = m->value_cache[l] + t * kv_dim + kv_h * head_dim;
                    float a = att_h[t];
                    for (int d = 0; d < head_dim; d++) out_h[d] += a * v_t[d];
                }
            }
        }

        /* QK: heads*seq*hd, AV: heads*seq*hd */
        TF_PROF_END("attention", 2.0 * n_heads * seq_len * head_dim * 2, 0);

        /* Output projection */
        TF_PROF_BEGIN("out_proj", l, "matvec", "FP32");
        tf_qmatvec_mt(m->xb, &layer->attn_output, m->xb2, n_embd, m->n_threads, m->thread_tmp);
        TF_PROF_END("out_proj", 2.0 * n_embd * n_embd, 0);

        /* Residual */
        for (int i = 0; i < n_embd; i++) m->x[i] += m->xb[i];

        /* --- FFN --- */
        /* RMSNorm */
        TF_PROF_BEGIN("ffn_norm", l, "rmsnorm", "FP32");
        tf_rmsnorm(m->xb, m->x, &layer->ffn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        TF_PROF_END("ffn_norm", 5.0 * n_embd, 0);

        /* SwiGLU: down @ (silu(gate @ x) * (up @ x)) */
        TF_PROF_BEGIN("ffn_gate", l, "matvec", "FP32");
        tf_qmatvec_mt(m->ffn_buf1, &layer->ffn_gate, m->xb, m->n_ff, m->n_threads, m->thread_tmp);
        TF_PROF_END("ffn_gate", 2.0 * m->n_ff * n_embd, 0);

        TF_PROF_BEGIN("ffn_up", l, "matvec", "FP32");
        tf_qmatvec_mt(m->ffn_buf2, &layer->ffn_up,   m->xb, m->n_ff, m->n_threads, m->thread_tmp);
        TF_PROF_END("ffn_up", 2.0 * m->n_ff * n_embd, 0);

        /* SiLU(x) = x * sigmoid(x) */
        TF_PROF_BEGIN("silu_mul", l, "silu_mul", "FP32");
        for (int i = 0; i < m->n_ff; i++) {
            float gate = m->ffn_buf1[i];
            gate = gate / (1.0f + expf(-gate));  /* SiLU */
            m->ffn_buf3[i] = gate * m->ffn_buf2[i];
        }
        TF_PROF_END("silu_mul", 5.0 * m->n_ff, 0);

        TF_PROF_BEGIN("ffn_down", l, "matvec", "FP32");
        tf_qmatvec_mt(m->xb, &layer->ffn_down, m->ffn_buf3, n_embd, m->n_threads, m->thread_tmp);
        TF_PROF_END("ffn_down", 2.0 * n_embd * m->n_ff, 0);

        /* Residual */
        for (int i = 0; i < n_embd; i++) m->x[i] += m->xb[i];

        /* DeepStack injection: add deepstack slice after each early layer */
        if (m->ds_embd && l < m->n_deepstack && m->ds_embd_stride > n_embd) {
            const float *ds_slice = m->ds_embd + (1 + l) * n_embd;
            for (int i = 0; i < n_embd; i++) m->x[i] += ds_slice[i];
        }

        if (l == 0 || l == m->n_layers - 1 || (l + 1) % 10 == 0) {
            float norm = 0.0f;
            for (int i = 0; i < n_embd; i++) norm += m->x[i] * m->x[i];
            fprintf(stderr, "  layer %2d: hidden norm = %.4f\n", l, sqrtf(norm));
        }
    }

    /* Final RMSNorm */
    TF_PROF_BEGIN("final_norm", -1, "rmsnorm", "FP32");
    tf_rmsnorm(m->x, m->x, &m->output_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
    TF_PROF_END("final_norm", 5.0 * n_embd, 0);

    return m->x;
}

float *transformer_forward_embd_pos(transformer_model *model, const float *embd, int cache_pos, int pos_t, int pos_h, int pos_w) {
    memcpy(model->x, embd, model->n_embd * sizeof(float));
    /* Store full embedding pointer for deepstack injection in tf_forward_blocks */
    model->ds_embd = embd;
    float *result = tf_forward_blocks(model, cache_pos, pos_t, pos_h, pos_w);
    model->ds_embd = NULL;
    return result;
}

float *transformer_forward_embd(transformer_model *model, const float *embd, int position) {
    return transformer_forward_embd_pos(model, embd, position, position, position, position);
}

float *transformer_forward_embd_logits_pos(transformer_model *model, const float *embd, int cache_pos, int pos_t, int pos_h, int pos_w) {
    float *hidden = transformer_forward_embd_pos(model, embd, cache_pos, pos_t, pos_h, pos_w);
    if (!model->has_lm_head || !hidden) return NULL;
    TF_PROF_BEGIN("lm_head", -1, "matvec", "FP32");
    tf_qmatvec_mt(model->logits, &model->output, hidden, model->n_vocab, model->n_threads, model->thread_tmp);
    TF_PROF_END("lm_head", 2.0 * model->n_vocab * model->n_embd, 0);
    return model->logits;
}

float *transformer_forward_embd_logits(transformer_model *model, const float *embd, int position) {
    return transformer_forward_embd_logits_pos(model, embd, position, position, position, position);
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
