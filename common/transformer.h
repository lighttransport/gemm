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
    int      n_dims;     /* up to 4 for GGUF tensors */
    uint64_t dims[4];
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
    /* MoE tensors (Qwen3MoE / Qwen3VLMoE): [n_expert variants] */
    qtensor ffn_gate_inp;  /* [n_embd, n_expert] */
    qtensor ffn_up_exps;   /* [n_embd, n_ff_expert, n_expert] */
    qtensor ffn_gate_exps; /* [n_embd, n_ff_expert, n_expert] */
    qtensor ffn_down_exps; /* [n_ff_expert, n_embd, n_expert] */
} transformer_layer;

typedef struct {
    /* Hyperparameters */
    int n_layers;
    int n_embd;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int n_ff;
    int n_ff_expert;
    int n_expert;
    int n_expert_used;
    int use_moe;
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

    /* Thread pool (persistent workers, no per-call pthread_create) */
    pthread_t *pool_threads;   /* [n_threads] worker threads */
    void *(*pool_fn)(void *);  /* current work function */
    void *pool_args;           /* array of per-thread task structs */
    size_t pool_arg_stride;    /* sizeof(task struct) */
    volatile int pool_phase;   /* incremented to signal work */
    volatile int pool_done;    /* number of workers done */
    int pool_alive;            /* 1 if pool is running */
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

/* --- Batched prefill API --- */

/* Batch descriptor for prefill: process N tokens through all layers at once. */
typedef struct {
    const float *embds;        /* [N, embd_stride] pre-expanded token embeddings */
    int N;                     /* number of tokens */
    int embd_stride;           /* stride between tokens in embds (>= n_embd) */
    const int *cache_pos;      /* [N] KV cache slot for each token */
    const int *pos_t;          /* [N] M-RoPE temporal positions */
    const int *pos_h;          /* [N] M-RoPE height positions */
    const int *pos_w;          /* [N] M-RoPE width positions */
    const float *const *ds_embds;  /* [N] per-token deepstack embedding pointers (NULL = no deepstack) */
    int ds_embd_stride;        /* total embedding stride for deepstack tokens */
} transformer_batch;

/* Run batched prefill and return logits for the last token. Returns NULL if no LM head.
 * All N tokens are processed through GEMM instead of individual matvecs. */
float *transformer_forward_batch_logits(transformer_model *m, const transformer_batch *b);

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
#include <alloca.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>

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
    t.n_dims = (int)gguf->tensors[idx].n_dims;
    if (t.n_dims > 4) t.n_dims = 4;
    for (int d = 0; d < t.n_dims; d++) t.dims[d] = gguf->tensors[idx].dims[d];
    /* dims[0] = inner dimension (cols), rows = product(dims[1:]) */
    t.n_cols = (int)gguf->tensors[idx].dims[0];
    uint64_t n_rows = 1;
    for (uint32_t d = 1; d < gguf->tensors[idx].n_dims; d++) n_rows *= gguf->tensors[idx].dims[d];
    t.n_rows = (int)n_rows;
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
        case GGML_TYPE_Q2_K: block_size = 256; type_size = 84;  break;
        case GGML_TYPE_Q3_K: block_size = 256; type_size = 110; break;
        case GGML_TYPE_Q8_0: block_size = 32;  type_size = 34;  break;
        case GGML_TYPE_Q4_K: block_size = 256; type_size = 144; break;
        case GGML_TYPE_Q5_K: block_size = 256; type_size = 176; break;
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

static int tf_is_supported_weight_type(uint32_t type) {
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return 1;
        default:
            return 0;
    }
}

static size_t tf_row_bytes(uint32_t type, int n_cols) {
    int block_size = 1, type_size = 4;
    switch (type) {
        case GGML_TYPE_Q2_K: block_size = 256; type_size = 84;  break;
        case GGML_TYPE_Q3_K: block_size = 256; type_size = 110; break;
        case GGML_TYPE_Q8_0: block_size = 32;  type_size = 34;  break;
        case GGML_TYPE_Q4_K: block_size = 256; type_size = 144; break;
        case GGML_TYPE_Q5_K: block_size = 256; type_size = 176; break;
        case GGML_TYPE_Q6_K: block_size = 256; type_size = 210; break;
        case GGML_TYPE_F32:  block_size = 1;   type_size = 4;   break;
        case GGML_TYPE_F16:  block_size = 1;   type_size = 2;   break;
        default: return 0;
    }
    return (size_t)((n_cols + block_size - 1) / block_size) * type_size;
}

/* Matvec for a single expert slice from 3D tensor [cols, rows_per_expert, n_expert]. */
static void tf_qmatvec_expert(float *dst, const qtensor *mat, int expert, const float *x,
                              int rows_per_expert, float *tmp) {
    size_t row_bytes = tf_row_bytes(mat->type, mat->n_cols);
    if (row_bytes == 0) {
        memset(dst, 0, rows_per_expert * sizeof(float));
        return;
    }

    const uint8_t *base = (const uint8_t *)mat->data + (size_t)expert * rows_per_expert * row_bytes;
    if (mat->type == GGML_TYPE_F16) {
        for (int i = 0; i < rows_per_expert; i++) {
            const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
            dst[i] = vec_dot_f16_f32(row, x, mat->n_cols);
        }
        return;
    }

    for (int i = 0; i < rows_per_expert; i++) {
        const void *row_data = base + (size_t)i * row_bytes;
        dequant_row(mat->type, row_data, tmp, mat->n_cols);
        float sum = 0.0f;
        for (int j = 0; j < mat->n_cols; j++) sum += tmp[j] * x[j];
        dst[i] = sum;
    }
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
    if (t->mat->type == GGML_TYPE_F16) {
        const uint8_t *base = (const uint8_t *)t->mat->data;
        size_t row_bytes = (size_t)n_cols * 2;
        int i = t->row_start;
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

/* Forward declarations for fused matvec and thread pool */
static void tf_qmatvec(float *dst, const qtensor *mat, const float *x, int n_rows, float *tmp);
static void tf_pool_dispatch(transformer_model *model, void *(*fn)(void *),
                              void *args, size_t arg_stride);
static void tf_pool_shutdown(transformer_model *model);

/* Fused dual-matrix matvec: computes dst1 = mat1 × x AND dst2 = mat2 × x in one dispatch.
 * Both matrices must have the same n_cols and same n_rows. */
typedef struct {
    float *dst1, *dst2;
    const qtensor *mat1, *mat2;
    const float *x;
    int row_start, row_end;
} tf_matvec_fused2_task;

static void tf_matvec_f16_rows(float *dst, const uint8_t *base, size_t row_bytes,
                                 const float *x, int n_cols, int row_start, int row_end) {
    int i = row_start;
    for (; i + 5 < row_end; i += 6) {
        matvec_f16_6row(dst + i,
            (const uint16_t *)(base + (size_t)i * row_bytes),
            (const uint16_t *)(base + (size_t)(i+1) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+2) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+3) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+4) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+5) * row_bytes),
            x, n_cols);
    }
    for (; i < row_end; i++) {
        const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
        dst[i] = vec_dot_f16_f32(row, x, n_cols);
    }
}

static void *tf_qmatvec_fused2_worker(void *arg) {
    tf_matvec_fused2_task *t = (tf_matvec_fused2_task *)arg;
    int n_cols = t->mat1->n_cols;
    size_t row_bytes = (size_t)n_cols * 2;
    tf_matvec_f16_rows(t->dst1, (const uint8_t *)t->mat1->data, row_bytes,
                        t->x, n_cols, t->row_start, t->row_end);
    tf_matvec_f16_rows(t->dst2, (const uint8_t *)t->mat2->data, row_bytes,
                        t->x, n_cols, t->row_start, t->row_end);
    return NULL;
}

static void tf_qmatvec_fused2_pool(transformer_model *m, float *dst1, const qtensor *mat1,
                                    float *dst2, const qtensor *mat2,
                                    const float *x, int n_rows) {
    int nt = m->n_threads;
    if (nt <= 1 || !m->pool_alive) {
        tf_qmatvec(dst1, mat1, x, n_rows, m->thread_tmp[0]);
        tf_qmatvec(dst2, mat2, x, n_rows, m->thread_tmp[0]);
        return;
    }
    tf_matvec_fused2_task *tasks = (tf_matvec_fused2_task *)alloca(nt * sizeof(tf_matvec_fused2_task));
    int rows_per = n_rows / nt, extra = n_rows % nt, offset = 0;
    for (int t = 0; t < nt; t++) {
        int count = rows_per + (t < extra ? 1 : 0);
        tasks[t] = (tf_matvec_fused2_task){dst1, dst2, mat1, mat2, x, offset, offset + count};
        offset += count;
    }
    tf_pool_dispatch(m, tf_qmatvec_fused2_worker, tasks, sizeof(tf_matvec_fused2_task));
}

/* Fused triple-matrix matvec: Q/K/V in one dispatch.
 * mat1 has n_rows1 rows, mat2 and mat3 have n_rows2 rows each. All same n_cols. */
typedef struct {
    float *dst1, *dst2, *dst3;
    const qtensor *mat1, *mat2, *mat3;
    const float *x;
    int row_start1, row_end1; /* range for mat1 */
    int row_start2, row_end2; /* range for mat2 and mat3 */
} tf_matvec_fused3_task;

static void *tf_qmatvec_fused3_worker(void *arg) {
    tf_matvec_fused3_task *t = (tf_matvec_fused3_task *)arg;
    int n_cols = t->mat1->n_cols;
    size_t row_bytes = (size_t)n_cols * 2;
    if (t->row_end1 > t->row_start1)
        tf_matvec_f16_rows(t->dst1, (const uint8_t *)t->mat1->data, row_bytes,
                            t->x, n_cols, t->row_start1, t->row_end1);
    if (t->row_end2 > t->row_start2) {
        tf_matvec_f16_rows(t->dst2, (const uint8_t *)t->mat2->data, row_bytes,
                            t->x, n_cols, t->row_start2, t->row_end2);
        tf_matvec_f16_rows(t->dst3, (const uint8_t *)t->mat3->data, row_bytes,
                            t->x, n_cols, t->row_start2, t->row_end2);
    }
    return NULL;
}

static void tf_qmatvec_fused_qkv_pool(transformer_model *m,
                                        float *q, const qtensor *mat_q, int n_q,
                                        float *k, const qtensor *mat_k,
                                        float *v, const qtensor *mat_v, int n_kv) {
    int nt = m->n_threads;
    if (nt <= 1 || !m->pool_alive) {
        tf_qmatvec(q, mat_q, m->xb, n_q, m->thread_tmp[0]);
        tf_qmatvec(k, mat_k, m->xb, n_kv, m->thread_tmp[0]);
        tf_qmatvec(v, mat_v, m->xb, n_kv, m->thread_tmp[0]);
        return;
    }
    /* Distribute total_rows = n_q + 2*n_kv across threads.
     * Each thread gets a proportional slice of Q rows and KV rows. */
    int total = n_q + 2 * n_kv;
    tf_matvec_fused3_task *tasks = (tf_matvec_fused3_task *)alloca(nt * sizeof(tf_matvec_fused3_task));
    int q_per = n_q / nt, q_extra = n_q % nt, q_off = 0;
    int kv_per = n_kv / nt, kv_extra = n_kv % nt, kv_off = 0;
    for (int t = 0; t < nt; t++) {
        int qc = q_per + (t < q_extra ? 1 : 0);
        int kvc = kv_per + (t < kv_extra ? 1 : 0);
        tasks[t] = (tf_matvec_fused3_task){
            q, k, v, mat_q, mat_k, mat_v, m->xb,
            q_off, q_off + qc, kv_off, kv_off + kvc
        };
        q_off += qc;
        kv_off += kvc;
    }
    tf_pool_dispatch(m, tf_qmatvec_fused3_worker, tasks, sizeof(tf_matvec_fused3_task));
}

static void tf_qmatvec(float *dst, const qtensor *mat, const float *x, int n_rows, float *tmp) {
    int n_cols = mat->n_cols;
    if (mat->type == GGML_TYPE_F16) {
        const uint8_t *base = (const uint8_t *)mat->data;
        size_t row_bytes = (size_t)n_cols * 2;
        int i = 0;
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

/* Pool-based multi-threaded matvec (avoids pthread_create per call) */
static void tf_qmatvec_pool(transformer_model *m, float *dst, const qtensor *mat, const float *x, int n_rows) {
    int n_threads = m->n_threads;
    if (n_threads <= 1 || n_rows < n_threads * 4 || !m->pool_alive) {
        tf_qmatvec(dst, mat, x, n_rows, m->thread_tmp[0]);
        return;
    }
    tf_matvec_task *tasks = (tf_matvec_task *)alloca(n_threads * sizeof(tf_matvec_task));
    int rows_per = n_rows / n_threads;
    int extra = n_rows % n_threads;
    int offset = 0;
    for (int t = 0; t < n_threads; t++) {
        int count = rows_per + (t < extra ? 1 : 0);
        tasks[t] = (tf_matvec_task){dst, mat, x, offset, offset + count, m->thread_tmp[t]};
        offset += count;
    }
    tf_pool_dispatch(m, tf_qmatvec_worker, tasks, sizeof(tf_matvec_task));
}

/* Legacy multi-threaded version (pthread_create per call) */
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

/* Forward declaration — defined after fast_exp_avx2 */
static void tf_softmax(float *x, int n);

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
    int hd = t->head_dim;
    for (int h = t->head_start; h < t->head_end; h++) {
        int kv_h = h / t->gqa_ratio;
        const float *q_h = t->q + h * hd;
        float *att_h = t->att + h * t->max_seq_len;
        int seq_len = t->seq_len;

#if defined(__AVX2__) && defined(__FMA__)
        if (hd == 64) {
            __m256 q0=_mm256_loadu_ps(q_h),    q1=_mm256_loadu_ps(q_h+8);
            __m256 q2=_mm256_loadu_ps(q_h+16), q3=_mm256_loadu_ps(q_h+24);
            __m256 q4=_mm256_loadu_ps(q_h+32), q5=_mm256_loadu_ps(q_h+40);
            __m256 q6=_mm256_loadu_ps(q_h+48), q7=_mm256_loadu_ps(q_h+56);

            /* QK scores: 4 positions at a time */
            int p = 0;
            for (; p + 3 < seq_len; p += 4) {
                const float *k0 = t->key_cache + (size_t)(p+0)*t->kv_dim + kv_h*hd;
                const float *k1 = t->key_cache + (size_t)(p+1)*t->kv_dim + kv_h*hd;
                const float *k2 = t->key_cache + (size_t)(p+2)*t->kv_dim + kv_h*hd;
                const float *k3 = t->key_cache + (size_t)(p+3)*t->kv_dim + kv_h*hd;
                __m256 s0=_mm256_mul_ps(q0,_mm256_loadu_ps(k0));
                __m256 s1=_mm256_mul_ps(q0,_mm256_loadu_ps(k1));
                __m256 s2=_mm256_mul_ps(q0,_mm256_loadu_ps(k2));
                __m256 s3=_mm256_mul_ps(q0,_mm256_loadu_ps(k3));
                s0=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k0+8),s0);
                s1=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k1+8),s1);
                s2=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k2+8),s2);
                s3=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k3+8),s3);
                s0=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k0+16),s0);
                s1=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k1+16),s1);
                s2=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k2+16),s2);
                s3=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k3+16),s3);
                s0=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k0+24),s0);
                s1=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k1+24),s1);
                s2=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k2+24),s2);
                s3=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k3+24),s3);
                s0=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k0+32),s0);
                s1=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k1+32),s1);
                s2=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k2+32),s2);
                s3=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k3+32),s3);
                s0=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k0+40),s0);
                s1=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k1+40),s1);
                s2=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k2+40),s2);
                s3=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k3+40),s3);
                s0=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k0+48),s0);
                s1=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k1+48),s1);
                s2=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k2+48),s2);
                s3=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k3+48),s3);
                s0=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k0+56),s0);
                s1=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k1+56),s1);
                s2=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k2+56),s2);
                s3=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k3+56),s3);
                __m256 h01=_mm256_hadd_ps(s0,s1);
                __m256 h23=_mm256_hadd_ps(s2,s3);
                __m256 h0123=_mm256_hadd_ps(h01,h23);
                __m128 lo=_mm256_castps256_ps128(h0123);
                __m128 hi=_mm256_extractf128_ps(h0123,1);
                _mm_storeu_ps(att_h+p, _mm_mul_ps(_mm_add_ps(lo,hi),_mm_set1_ps(t->scale)));
            }
            for (; p < seq_len; p++) {
                const float *kp = t->key_cache + (size_t)p*t->kv_dim + kv_h*hd;
                __m256 s=_mm256_mul_ps(q0,_mm256_loadu_ps(kp));
                s=_mm256_fmadd_ps(q1,_mm256_loadu_ps(kp+8),s);
                s=_mm256_fmadd_ps(q2,_mm256_loadu_ps(kp+16),s);
                s=_mm256_fmadd_ps(q3,_mm256_loadu_ps(kp+24),s);
                s=_mm256_fmadd_ps(q4,_mm256_loadu_ps(kp+32),s);
                s=_mm256_fmadd_ps(q5,_mm256_loadu_ps(kp+40),s);
                s=_mm256_fmadd_ps(q6,_mm256_loadu_ps(kp+48),s);
                s=_mm256_fmadd_ps(q7,_mm256_loadu_ps(kp+56),s);
                __m128 _hi=_mm256_extractf128_ps(s,1);
                __m128 _lo=_mm256_castps256_ps128(s);
                __m128 _s=_mm_add_ps(_lo,_hi);
                _s=_mm_add_ps(_s,_mm_movehl_ps(_s,_s));
                _s=_mm_add_ss(_s,_mm_movehdup_ps(_s));
                att_h[p] = _mm_cvtss_f32(_s) * t->scale;
            }

            /* Softmax */
            tf_softmax(att_h, seq_len);

            /* V accumulation: 8 AVX accumulators for head_dim=64 */
            float *out_h = t->xb2 + h * hd;
            __m256 o0=_mm256_setzero_ps(), o1=_mm256_setzero_ps();
            __m256 o2=_mm256_setzero_ps(), o3=_mm256_setzero_ps();
            __m256 o4=_mm256_setzero_ps(), o5=_mm256_setzero_ps();
            __m256 o6=_mm256_setzero_ps(), o7=_mm256_setzero_ps();
            for (p = 0; p < seq_len; p++) {
                const float *vp = t->value_cache + (size_t)p*t->kv_dim + kv_h*hd;
                __m256 a = _mm256_set1_ps(att_h[p]);
                o0=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp),   o0);
                o1=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+8), o1);
                o2=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+16),o2);
                o3=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+24),o3);
                o4=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+32),o4);
                o5=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+40),o5);
                o6=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+48),o6);
                o7=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+56),o7);
            }
            _mm256_storeu_ps(out_h,   o0); _mm256_storeu_ps(out_h+8, o1);
            _mm256_storeu_ps(out_h+16,o2); _mm256_storeu_ps(out_h+24,o3);
            _mm256_storeu_ps(out_h+32,o4); _mm256_storeu_ps(out_h+40,o5);
            _mm256_storeu_ps(out_h+48,o6); _mm256_storeu_ps(out_h+56,o7);
        } else
#endif
        {
            for (int p = 0; p < seq_len; p++) {
                const float *k_p = t->key_cache + (size_t)p * t->kv_dim + kv_h * hd;
                float score = 0.0f;
                for (int d = 0; d < hd; d++) score += q_h[d] * k_p[d];
                att_h[p] = score * t->scale;
            }
            tf_softmax(att_h, seq_len);
            float *out_h = t->xb2 + h * hd;
            memset(out_h, 0, hd * sizeof(float));
            for (int p = 0; p < seq_len; p++) {
                const float *v_p = t->value_cache + (size_t)p * t->kv_dim + kv_h * hd;
                float a = att_h[p];
                for (int d = 0; d < hd; d++) out_h[d] += a * v_p[d];
            }
        }
    }
    return NULL;
}

#if defined(__AVX2__) && defined(__FMA__)
/* Fast AVX2 exp approximation — defined early so tf_softmax can use it. */
static inline __m256 fast_exp_avx2(__m256 x);
#endif

/* Softmax in-place over n elements */
static void tf_softmax(float *x, int n) {
#if defined(__AVX2__) && defined(__FMA__)
    /* AVX2 max */
    int i = 0;
    __m256 vmax = _mm256_set1_ps(-1e30f);
    for (; i + 7 < n; i += 8)
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
    /* Reduce vmax to scalar */
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m4 = _mm_max_ps(lo, hi);
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
    float max_val = _mm_cvtss_f32(m4);
    for (; i < n; i++) if (x[i] > max_val) max_val = x[i];

    /* exp(x - max) and sum, using fast_exp_avx2 */
    __m256 vmax_b = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmax_b);
        v = fast_exp_avx2(v);
        _mm256_storeu_ps(x + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }
    /* Reduce vsum */
    hi = _mm256_extractf128_ps(vsum, 1);
    lo = _mm256_castps256_ps128(vsum);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    float sum = _mm_cvtss_f32(s4);
    for (; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }

    /* Normalize */
    __m256 vinv = _mm256_set1_ps(1.0f / sum);
    i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
    float inv_sum = 1.0f / sum;
    for (; i < n; i++) x[i] *= inv_sum;
#else
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
#endif
}

/* ---- Load ---- */

transformer_model *transformer_load(gguf_context *gguf, int max_seq_len) {
    if (!gguf) return NULL;

    transformer_model *m = (transformer_model *)calloc(1, sizeof(transformer_model));
    if (!m) return NULL;

    /* Detect architecture prefix. */
    const char *arch = "qwen2";
    if (gguf_find_key(gguf, "qwen3vlmoe.block_count") >= 0) {
        arch = "qwen3vlmoe";
    } else if (gguf_find_key(gguf, "qwen3moe.block_count") >= 0) {
        arch = "qwen3moe";
    } else if (gguf_find_key(gguf, "qwen3vl.block_count") >= 0) {
        arch = "qwen3vl";
    } else if (gguf_find_key(gguf, "qwen3.block_count") >= 0) {
        arch = "qwen3";
    }

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
    m->n_expert = tf_get_int(gguf, ARCH_KEY("expert_count"), 0);
    m->n_expert_used = tf_get_int(gguf, ARCH_KEY("expert_used_count"), 0);
    m->n_ff_expert = tf_get_int(gguf, ARCH_KEY("expert_feed_forward_length"), 0);
    m->use_moe = (m->n_expert > 0);
    if (m->use_moe) {
        if (m->n_expert_used <= 0 || m->n_expert_used > m->n_expert) {
            fprintf(stderr, "transformer: invalid expert_used_count=%d (expert_count=%d)\n",
                    m->n_expert_used, m->n_expert);
            free(m);
            return NULL;
        }
        if (m->n_ff_expert <= 0) {
            fprintf(stderr, "transformer: invalid expert_feed_forward_length=%d\n", m->n_ff_expert);
            free(m);
            return NULL;
        }
    }
    int ctx_len = tf_get_int(gguf, ARCH_KEY("context_length"), 0);
    if (max_seq_len <= 0) {
        max_seq_len = (ctx_len > 0) ? ctx_len : 1024;
    } else if (ctx_len > 0 && max_seq_len > ctx_len) {
        fprintf(stderr, "transformer: requested max_seq_len=%d exceeds model context=%d, clamping\n",
                max_seq_len, ctx_len);
        max_seq_len = ctx_len;
    }
    if (max_seq_len <= 0) {
        fprintf(stderr, "transformer: invalid max_seq_len=%d\n", max_seq_len);
        free(m);
        return NULL;
    }

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

    /* head_dim: prefer attention.key_length from GGUF (e.g. Qwen3-VL-4B has head_dim=128
     * despite n_embd/n_heads=80), fall back to n_embd/n_heads */
    m->head_dim = tf_get_int(gguf, ARCH_KEY("attention.key_length"), m->n_embd / m->n_heads);

    #undef ARCH_KEY
    fprintf(stderr, "transformer: architecture=%s\n", arch);
    m->max_seq_len = max_seq_len;

    fprintf(stderr, "transformer: n_embd=%d n_heads=%d n_kv_heads=%d n_layers=%d n_ff=%d head_dim=%d\n",
            m->n_embd, m->n_heads, m->n_kv_heads, m->n_layers, m->n_ff, m->head_dim);
    fprintf(stderr, "transformer: rope_freq_base=%.0f rms_norm_eps=%.1e max_seq_len=%d\n",
            m->rope_freq_base, m->rms_norm_eps, max_seq_len);
    if (m->use_moe) {
        fprintf(stderr, "transformer: MoE enabled (n_expert=%d n_expert_used=%d n_ff_expert=%d)\n",
                m->n_expert, m->n_expert_used, m->n_ff_expert);
    }

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
    if (!m->token_embd.data || !m->output_norm.data) {
        fprintf(stderr, "transformer: missing required global tensor(s)\n");
        transformer_free(m);
        return NULL;
    }
    if (m->n_vocab <= 0) {
        fprintf(stderr, "transformer: invalid vocabulary size %d\n", m->n_vocab);
        transformer_free(m);
        return NULL;
    }
    fprintf(stderr, "transformer: n_vocab=%d has_lm_head=%d\n", m->n_vocab, m->has_lm_head);

    /* Per-layer tensors */
    m->layers = (transformer_layer *)calloc(m->n_layers, sizeof(transformer_layer));
    if (!m->layers) {
        fprintf(stderr, "transformer: failed to allocate layer metadata\n");
        transformer_free(m);
        return NULL;
    }
    int missing_required = 0;
    int unsupported_reported = 0;
    for (int l = 0; l < m->n_layers; l++) {
        char name[128];
        #define LOAD(field, suffix, req) \
            snprintf(name, sizeof(name), "blk.%d." suffix ".weight", l); \
            m->layers[l].field = tf_load_tensor(gguf, name, req); \
            if ((req) && !m->layers[l].field.data) missing_required = 1;
        #define REQUIRE_SUPPORTED(field, suffix) \
            do { \
                if (m->layers[l].field.data && !tf_is_supported_weight_type(m->layers[l].field.type)) { \
                    if (!unsupported_reported) { \
                        fprintf(stderr, "transformer: unsupported tensor type for blk.%d." suffix ".weight (type=%u)\n", \
                                l, m->layers[l].field.type); \
                        unsupported_reported = 1; \
                    } \
                    missing_required = 1; \
                } \
            } while (0)

        LOAD(attn_norm,    "attn_norm",   1)
        LOAD(attn_q,       "attn_q",      1)
        LOAD(attn_k,       "attn_k",      1)
        LOAD(attn_v,       "attn_v",      1)
        LOAD(attn_q_norm,  "attn_q_norm", 0)
        LOAD(attn_k_norm,  "attn_k_norm", 0)
        LOAD(attn_output,  "attn_output", 1)
        LOAD(ffn_norm,     "ffn_norm",    1)
        if (m->use_moe) {
            LOAD(ffn_gate_inp,  "ffn_gate_inp",  1)
            LOAD(ffn_up_exps,   "ffn_up_exps",   1)
            LOAD(ffn_gate_exps, "ffn_gate_exps", 1)
            LOAD(ffn_down_exps, "ffn_down_exps", 1)
            REQUIRE_SUPPORTED(ffn_gate_inp,  "ffn_gate_inp");
            REQUIRE_SUPPORTED(ffn_up_exps,   "ffn_up_exps");
            REQUIRE_SUPPORTED(ffn_gate_exps, "ffn_gate_exps");
            REQUIRE_SUPPORTED(ffn_down_exps, "ffn_down_exps");
            /* Validate expert tensor shape convention [cols, rows_per_expert, n_expert]. */
            if (m->layers[l].ffn_up_exps.data) {
                if (m->layers[l].ffn_up_exps.n_dims < 3 ||
                    (int)m->layers[l].ffn_up_exps.dims[1] != m->n_ff_expert ||
                    (int)m->layers[l].ffn_up_exps.dims[2] != m->n_expert) {
                    fprintf(stderr, "transformer: unexpected shape for blk.%d.ffn_up_exps.weight\n", l);
                    missing_required = 1;
                }
            }
            if (m->layers[l].ffn_gate_exps.data) {
                if (m->layers[l].ffn_gate_exps.n_dims < 3 ||
                    (int)m->layers[l].ffn_gate_exps.dims[1] != m->n_ff_expert ||
                    (int)m->layers[l].ffn_gate_exps.dims[2] != m->n_expert) {
                    fprintf(stderr, "transformer: unexpected shape for blk.%d.ffn_gate_exps.weight\n", l);
                    missing_required = 1;
                }
            }
            if (m->layers[l].ffn_down_exps.data) {
                if (m->layers[l].ffn_down_exps.n_dims < 3 ||
                    (int)m->layers[l].ffn_down_exps.dims[1] != m->n_embd ||
                    (int)m->layers[l].ffn_down_exps.dims[2] != m->n_expert) {
                    fprintf(stderr, "transformer: unexpected shape for blk.%d.ffn_down_exps.weight\n", l);
                    missing_required = 1;
                }
            }
        } else {
            LOAD(ffn_gate,     "ffn_gate",    1)
            LOAD(ffn_up,       "ffn_up",      1)
            LOAD(ffn_down,     "ffn_down",    1)
            REQUIRE_SUPPORTED(ffn_gate, "ffn_gate");
            REQUIRE_SUPPORTED(ffn_up,   "ffn_up");
            REQUIRE_SUPPORTED(ffn_down, "ffn_down");
        }
        REQUIRE_SUPPORTED(attn_norm,   "attn_norm");
        REQUIRE_SUPPORTED(attn_q,      "attn_q");
        REQUIRE_SUPPORTED(attn_k,      "attn_k");
        REQUIRE_SUPPORTED(attn_v,      "attn_v");
        REQUIRE_SUPPORTED(attn_output, "attn_output");
        REQUIRE_SUPPORTED(ffn_norm,    "ffn_norm");
        #undef LOAD
        #undef REQUIRE_SUPPORTED
    }
    if (missing_required) {
        fprintf(stderr, "transformer: aborting due to missing required per-layer tensors\n");
        transformer_free(m);
        return NULL;
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
    int max_ff = m->n_ff;
    if (m->use_moe && m->n_ff_expert > max_ff) max_ff = m->n_ff_expert;
    if (m->use_moe && m->n_expert > max_ff) max_ff = m->n_expert;
    int max_dim = m->n_embd > max_ff ? m->n_embd : max_ff;
    int q_dim = m->n_heads * m->head_dim;  /* may differ from n_embd (e.g. 4B: 4096 vs 2560) */
    m->x         = (float *)calloc(m->n_embd, sizeof(float));
    m->xb        = (float *)calloc(m->n_embd, sizeof(float));
    m->xb2       = (float *)calloc(q_dim, sizeof(float));
    m->q         = (float *)calloc(q_dim, sizeof(float));
    m->k         = (float *)calloc(kv_dim, sizeof(float));
    m->v         = (float *)calloc(kv_dim, sizeof(float));
    m->att       = (float *)calloc(m->n_heads * max_seq_len, sizeof(float));
    m->ffn_buf1  = (float *)calloc(max_ff, sizeof(float));
    m->ffn_buf2  = (float *)calloc(max_ff, sizeof(float));
    m->ffn_buf3  = (float *)calloc(max_ff, sizeof(float));
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
    tf_pool_shutdown(model);
    if (model->thread_tmp) {
        for (int t = 1; t < model->n_threads; t++) free(model->thread_tmp[t]);
        free(model->thread_tmp);
    }
    free(model);
}

/* ---- Thread pool ---- */
typedef struct {
    transformer_model *model;
    int tid;
} tf_pool_worker_ctx;

static void *tf_pool_worker_main(void *arg) {
    tf_pool_worker_ctx *ctx = (tf_pool_worker_ctx *)arg;
    transformer_model *m = ctx->model;
    int tid = ctx->tid;
    free(ctx);

    int last_phase = 0;
    while (1) {
        /* Spin-wait for work */
        while (m->pool_phase == last_phase) {
            if (!m->pool_alive) return NULL;
            __builtin_ia32_pause();
        }
        last_phase = m->pool_phase;
        if (!m->pool_alive) return NULL;

        /* Execute work */
        void *task = (char *)m->pool_args + (size_t)tid * m->pool_arg_stride;
        m->pool_fn(task);

        /* Signal done */
        __sync_add_and_fetch(&m->pool_done, 1);
    }
    return NULL;
}

static void tf_pool_shutdown(transformer_model *model) {
    if (!model->pool_alive) return;
    model->pool_alive = 0;
    __sync_synchronize();
    model->pool_phase++;
    for (int t = 0; t < model->n_threads; t++)
        pthread_join(model->pool_threads[t], NULL);
    free(model->pool_threads);
    model->pool_threads = NULL;
}

static void tf_pool_start(transformer_model *model) {
    int nt = model->n_threads;
    model->pool_threads = (pthread_t *)calloc(nt, sizeof(pthread_t));
    model->pool_phase = 0;
    model->pool_done = 0;
    model->pool_alive = 1;
    __sync_synchronize();
    for (int t = 0; t < nt; t++) {
        tf_pool_worker_ctx *ctx = (tf_pool_worker_ctx *)malloc(sizeof(*ctx));
        ctx->model = model;
        ctx->tid = t;
        pthread_create(&model->pool_threads[t], NULL, tf_pool_worker_main, ctx);
    }
}

/* Dispatch work to the thread pool: calls fn(args + tid * arg_stride) for each thread */
static void tf_pool_dispatch(transformer_model *model, void *(*fn)(void *),
                              void *args, size_t arg_stride) {
    int nt = model->n_threads;
    model->pool_fn = fn;
    model->pool_args = args;
    model->pool_arg_stride = arg_stride;

    model->pool_done = 0;
    __sync_synchronize();
    model->pool_phase++;

    /* Spin-wait for all workers */
    while (model->pool_done < nt)
        __builtin_ia32_pause();
}

void transformer_set_threads(transformer_model *model, int n_threads) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads == model->n_threads) return;

    /* Shutdown old pool */
    tf_pool_shutdown(model);

    /* Free old extra per-thread buffers */
    for (int t = 1; t < model->n_threads; t++) free(model->thread_tmp[t]);
    free(model->thread_tmp);

    int max_ff = model->n_ff;
    if (model->use_moe && model->n_ff_expert > max_ff) max_ff = model->n_ff_expert;
    if (model->use_moe && model->n_expert > max_ff) max_ff = model->n_expert;
    int max_dim = model->n_embd > max_ff ? model->n_embd : max_ff;
    model->n_threads = n_threads;
    model->thread_tmp = (float **)calloc(n_threads, sizeof(float *));
    model->thread_tmp[0] = model->matvec_tmp;
    for (int t = 1; t < n_threads; t++) {
        model->thread_tmp[t] = (float *)calloc(max_dim, sizeof(float));
    }

    /* Start new pool */
    if (n_threads > 1) tf_pool_start(model);
    fprintf(stderr, "transformer: using %d threads (thread pool)\n", n_threads);
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
    if (!model || !model->token_embd.data) return NULL;
    if (token_id < 0 || token_id >= model->n_vocab) {
        fprintf(stderr, "transformer_forward_pos: token_id=%d out of range [0, %d)\n", token_id, model->n_vocab);
        return NULL;
    }
    if (cache_pos < 0 || cache_pos >= model->max_seq_len) {
        fprintf(stderr, "transformer_forward_pos: cache_pos=%d out of range [0, %d)\n", cache_pos, model->max_seq_len);
        return NULL;
    }
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
    tf_qmatvec_pool(model, model->logits, &model->output, hidden, model->n_vocab);
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
    int q_dim = n_heads * head_dim;  /* may differ from n_embd */
    int gqa_ratio = n_heads / n_kv_heads;

    /* 2. Transformer blocks */
    for (int l = 0; l < m->n_layers; l++) {
        transformer_layer *layer = &m->layers[l];

        /* --- Attention --- */
        /* RMSNorm */
        TF_PROF_BEGIN("attn_norm", l, "rmsnorm", "FP32");
        tf_rmsnorm(m->xb, m->x, &layer->attn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        TF_PROF_END("attn_norm", 5.0 * n_embd, 0);

        /* Q/K/V projections (fused: one pool dispatch for all three) */
        TF_PROF_BEGIN("qkv_proj", l, "matvec", "FP32");
        tf_qmatvec_fused_qkv_pool(m, m->q, &layer->attn_q, q_dim,
                                   m->k, &layer->attn_k, m->v, &layer->attn_v, kv_dim);
        TF_PROF_END("qkv_proj", 2.0 * (q_dim + 2.0 * kv_dim) * n_embd, 0);

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

        if (m->n_threads > 1 && n_heads >= m->n_threads && m->pool_alive) {
            /* Pool-based threaded attention */
            int nt = m->n_threads;
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
            }
            tf_pool_dispatch(m, tf_attn_worker, atasks, sizeof(tf_attn_task));
        } else {
            memset(m->xb2, 0, q_dim * sizeof(float));
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
        tf_qmatvec_pool(m, m->xb, &layer->attn_output, m->xb2, n_embd);
        TF_PROF_END("out_proj", 2.0 * n_embd * n_embd, 0);

        /* Residual */
        for (int i = 0; i < n_embd; i++) m->x[i] += m->xb[i];

        /* --- FFN --- */
        /* RMSNorm */
        TF_PROF_BEGIN("ffn_norm", l, "rmsnorm", "FP32");
        tf_rmsnorm(m->xb, m->x, &layer->ffn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        TF_PROF_END("ffn_norm", 5.0 * n_embd, 0);

        if (m->use_moe && layer->ffn_gate_inp.data) {
            const int n_expert = m->n_expert;
            const int n_top = m->n_expert_used;
            const int n_ff_exp = m->n_ff_expert;

            /* Router logits -> probs over experts */
            TF_PROF_BEGIN("ffn_gate_inp", l, "matvec", "FP32");
            tf_qmatvec_pool(m, m->ffn_buf1, &layer->ffn_gate_inp, m->xb, n_expert);
            TF_PROF_END("ffn_gate_inp", 2.0 * n_expert * n_embd, 0);
            tf_softmax(m->ffn_buf1, n_expert);

            /* Select top-k experts */
            int *top_idx = (int *)alloca(n_top * sizeof(int));
            float *top_w = (float *)alloca(n_top * sizeof(float));
            float wsum = 0.0f;
            for (int i = 0; i < n_top; i++) {
                int best = -1;
                float best_w = -1.0f;
                for (int e = 0; e < n_expert; e++) {
                    float w = m->ffn_buf1[e];
                    if (w > best_w) {
                        best_w = w;
                        best = e;
                    }
                }
                top_idx[i] = best;
                top_w[i] = best_w;
                wsum += best_w;
                m->ffn_buf1[best] = -1.0f; /* mark as used */
            }
            if (wsum > 0.0f) {
                for (int i = 0; i < n_top; i++) top_w[i] /= wsum;
            }

            /* Aggregate selected experts */
            memset(m->xb2, 0, n_embd * sizeof(float));
            for (int ei = 0; ei < n_top; ei++) {
                int e = top_idx[ei];
                float ew = top_w[ei];

                TF_PROF_BEGIN("ffn_up_exp", l, "matvec", "FP32");
                tf_qmatvec_expert(m->ffn_buf2, &layer->ffn_up_exps, e, m->xb, n_ff_exp, m->thread_tmp[0]);
                TF_PROF_END("ffn_up_exp", 2.0 * n_ff_exp * n_embd, 0);

                TF_PROF_BEGIN("ffn_gate_exp", l, "matvec", "FP32");
                tf_qmatvec_expert(m->ffn_buf3, &layer->ffn_gate_exps, e, m->xb, n_ff_exp, m->thread_tmp[0]);
                TF_PROF_END("ffn_gate_exp", 2.0 * n_ff_exp * n_embd, 0);

                TF_PROF_BEGIN("silu_mul", l, "silu_mul", "FP32");
                for (int i = 0; i < n_ff_exp; i++) {
                    float gate = m->ffn_buf3[i];
                    gate = gate / (1.0f + expf(-gate));
                    m->ffn_buf3[i] = gate * m->ffn_buf2[i];
                }
                TF_PROF_END("silu_mul", 5.0 * n_ff_exp, 0);

                TF_PROF_BEGIN("ffn_down_exp", l, "matvec", "FP32");
                tf_qmatvec_expert(m->q, &layer->ffn_down_exps, e, m->ffn_buf3, n_embd, m->thread_tmp[0]);
                TF_PROF_END("ffn_down_exp", 2.0 * n_embd * n_ff_exp, 0);

                for (int i = 0; i < n_embd; i++) m->xb2[i] += ew * m->q[i];
            }

            for (int i = 0; i < n_embd; i++) m->x[i] += m->xb2[i];
        } else {
            /* Dense SwiGLU: down @ (silu(gate @ x) * (up @ x)) */
            TF_PROF_BEGIN("ffn_gate_up", l, "matvec", "FP32");
            tf_qmatvec_fused2_pool(m, m->ffn_buf1, &layer->ffn_gate,
                                    m->ffn_buf2, &layer->ffn_up, m->xb, m->n_ff);
            TF_PROF_END("ffn_gate_up", 2.0 * 2.0 * m->n_ff * n_embd, 0);

            TF_PROF_BEGIN("silu_mul", l, "silu_mul", "FP32");
            for (int i = 0; i < m->n_ff; i++) {
                float gate = m->ffn_buf1[i];
                gate = gate / (1.0f + expf(-gate));  /* SiLU */
                m->ffn_buf3[i] = gate * m->ffn_buf2[i];
            }
            TF_PROF_END("silu_mul", 5.0 * m->n_ff, 0);

            TF_PROF_BEGIN("ffn_down", l, "matvec", "FP32");
            tf_qmatvec_pool(m, m->xb, &layer->ffn_down, m->ffn_buf3, n_embd);
            TF_PROF_END("ffn_down", 2.0 * n_embd * m->n_ff, 0);

            for (int i = 0; i < n_embd; i++) m->x[i] += m->xb[i];
        }

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
    if (!model || !embd) return NULL;
    if (cache_pos < 0 || cache_pos >= model->max_seq_len) {
        fprintf(stderr, "transformer_forward_embd_pos: cache_pos=%d out of range [0, %d)\n", cache_pos, model->max_seq_len);
        return NULL;
    }
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
    tf_qmatvec_pool(model, model->logits, &model->output, hidden, model->n_vocab);
    TF_PROF_END("lm_head", 2.0 * model->n_vocab * model->n_embd, 0);
    return model->logits;
}

float *transformer_forward_embd_logits(transformer_model *model, const float *embd, int position) {
    return transformer_forward_embd_logits_pos(model, embd, position, position, position, position);
}

/* ---- Batched prefill ---- */

/* Multi-threaded GEMM worker for F16 weights */
typedef struct {
    float *Y;
    const uint16_t *W;
    const float *X;
    int row_start, row_end;
    int K, N, Y_stride, X_stride;
} tf_gemm_task;

static void *tf_gemm_worker(void *arg) {
    tf_gemm_task *t = (tf_gemm_task *)arg;
    int nrows = t->row_end - t->row_start;
    gemm_f16_f32(t->Y + t->row_start * t->Y_stride,
                  t->W + (size_t)t->row_start * t->K,
                  t->X, nrows, t->K, t->N, t->Y_stride, t->X_stride);
    return NULL;
}

/* Multi-threaded GEMM for F16 weight matrices: Y[n_rows, N] = W[n_rows, K] × X[N, K]^T
 * Output Y is row-major: Y[row][tok], Y_stride >= N. */
static void tf_gemm_f16_mt(float *Y, const qtensor *mat, const float *X,
                            int n_rows, int N, int Y_stride, int X_stride,
                            int n_threads) {
    if (mat->type != GGML_TYPE_F16) {
        /* Fallback: per-token matvec for non-F16 weights */
        for (int t = 0; t < N; t++) {
            const float *xt = X + (size_t)t * X_stride;
            float *yt_col = Y + t; /* column t, but Y is row-major [n_rows, N] */
            /* We need a temp buffer; use stack for moderate sizes */
            float *tmp = (float *)alloca(n_rows * sizeof(float));
            /* This is inefficient but handles quantized weights correctly */
            for (int r = 0; r < n_rows; r++) {
                float row_buf[4096]; /* stack scratch for dequant */
                int n_cols = mat->n_cols;
                int block_size = 1, type_size = 4;
                switch (mat->type) {
                    case GGML_TYPE_Q2_K: block_size = 256; type_size = 84;  break;
                    case GGML_TYPE_Q3_K: block_size = 256; type_size = 110; break;
                    case GGML_TYPE_Q8_0: block_size = 32;  type_size = 34;  break;
                    case GGML_TYPE_Q4_K: block_size = 256; type_size = 144; break;
                    case GGML_TYPE_Q5_K: block_size = 256; type_size = 176; break;
                    case GGML_TYPE_Q6_K: block_size = 256; type_size = 210; break;
                    case GGML_TYPE_F32:  block_size = 1;   type_size = 4;   break;
                    default: break;
                }
                size_t row_bytes = (size_t)((n_cols + block_size - 1) / block_size) * type_size;
                const void *row_data = (const uint8_t *)mat->data + (size_t)r * row_bytes;
                dequant_row(mat->type, row_data, row_buf, n_cols);
                float sum = 0.0f;
                for (int j = 0; j < n_cols; j++) sum += row_buf[j] * xt[j];
                Y[r * Y_stride + t] = sum;
            }
        }
        return;
    }

    const uint16_t *W = (const uint16_t *)mat->data;
    int K = mat->n_cols;

    if (n_threads <= 1 || n_rows < n_threads * 6) {
        gemm_f16_f32(Y, W, X, n_rows, K, N, Y_stride, X_stride);
        return;
    }

    pthread_t *threads = (pthread_t *)alloca(n_threads * sizeof(pthread_t));
    tf_gemm_task *tasks = (tf_gemm_task *)alloca(n_threads * sizeof(tf_gemm_task));
    int rows_per = n_rows / n_threads;
    int extra = n_rows % n_threads;
    int offset = 0;
    for (int i = 0; i < n_threads; i++) {
        int count = rows_per + (i < extra ? 1 : 0);
        tasks[i] = (tf_gemm_task){Y, W, X, offset, offset + count, K, N, Y_stride, X_stride};
        offset += count;
        pthread_create(&threads[i], NULL, tf_gemm_worker, &tasks[i]);
    }
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
}

/* Token-major threaded GEMM worker */
typedef struct {
    float *Y;
    const uint16_t *W;
    const float *X;
    int row_start, row_end;
    int K, N, Y_stride, X_stride;
} tf_gemm_tm_task;

static void *tf_gemm_tm_worker(void *arg) {
    tf_gemm_tm_task *t = (tf_gemm_tm_task *)arg;
    int nrows = t->row_end - t->row_start;
    /* Write to Y[tok * Y_stride + row], offset row by row_start */
    /* We need a shifted Y pointer that accounts for row_start offset in the token-major layout */
    gemm_f16_f32_tokmajor(t->Y + t->row_start,
                           t->W + (size_t)t->row_start * t->K,
                           t->X, nrows, t->K, t->N, t->Y_stride, t->X_stride);
    return NULL;
}

/* Token-major GEMM: Y[tok * out_stride + row] = dot(W[row,:], X[tok,:])
 * Direct output without transpose. */
static void tf_gemm_f16_mt_tokenmajor(float *Y_out, const qtensor *mat, const float *X,
                                       int n_rows, int N, int out_stride, int X_stride,
                                       int n_threads) {
    if (mat->type != GGML_TYPE_F16) {
        /* Fallback for non-F16: per-token matvec */
        for (int t = 0; t < N; t++) {
            const float *xt = X + (size_t)t * X_stride;
            for (int r = 0; r < n_rows; r++) {
                size_t row_bytes = tf_row_bytes(mat->type, mat->n_cols);
                const void *row_data = (const uint8_t *)mat->data + (size_t)r * row_bytes;
                float row_buf[8192];
                dequant_row(mat->type, row_data, row_buf, mat->n_cols);
                float sum = 0.0f;
                for (int j = 0; j < mat->n_cols; j++) sum += row_buf[j] * xt[j];
                Y_out[(size_t)t * out_stride + r] = sum;
            }
        }
        return;
    }

    const uint16_t *W = (const uint16_t *)mat->data;
    int K = mat->n_cols;

    if (n_threads <= 1 || n_rows < n_threads * 6) {
        gemm_f16_f32_tokmajor(Y_out, W, X, n_rows, K, N, out_stride, X_stride);
        return;
    }

    pthread_t *threads = (pthread_t *)alloca(n_threads * sizeof(pthread_t));
    tf_gemm_tm_task *tasks = (tf_gemm_tm_task *)alloca(n_threads * sizeof(tf_gemm_tm_task));
    int rows_per = n_rows / n_threads;
    int extra = n_rows % n_threads;
    int offset = 0;
    for (int i = 0; i < n_threads; i++) {
        int count = rows_per + (i < extra ? 1 : 0);
        tasks[i] = (tf_gemm_tm_task){Y_out, W, X, offset, offset + count, K, N, out_stride, X_stride};
        offset += count;
        pthread_create(&threads[i], NULL, tf_gemm_tm_worker, &tasks[i]);
    }
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
}

/* Fused dual-matrix threaded GEMM: compute gate and up simultaneously, reading X once */
typedef struct {
    float *Y1, *Y2;
    const uint16_t *W1, *W2;
    const float *X;
    int row_start, row_end;
    int K, N, Y1_stride, Y2_stride, X_stride;
} tf_gemm_fused2_task;

static void *tf_gemm_fused2_worker(void *arg) {
    tf_gemm_fused2_task *t = (tf_gemm_fused2_task *)arg;
    int nrows = t->row_end - t->row_start;
    gemm_f16_f32_tokmajor_fused2(
        t->Y1 + t->row_start, t->W1 + (size_t)t->row_start * t->K,
        t->Y2 + t->row_start, t->W2 + (size_t)t->row_start * t->K,
        t->X, nrows, t->K, t->N,
        t->Y1_stride, t->Y2_stride, t->X_stride);
    return NULL;
}

static void tf_gemm_f16_mt_fused2(float *Y1, const qtensor *mat1,
                                    float *Y2, const qtensor *mat2,
                                    const float *X, int n_rows, int N,
                                    int out_stride, int X_stride, int n_threads) {
    const uint16_t *W1 = (const uint16_t *)mat1->data;
    const uint16_t *W2 = (const uint16_t *)mat2->data;
    int K = mat1->n_cols;

    if (n_threads <= 1 || n_rows < n_threads * 4) {
        gemm_f16_f32_tokmajor_fused2(Y1, W1, Y2, W2, X, n_rows, K, N,
                                      out_stride, out_stride, X_stride);
        return;
    }

    pthread_t *threads = (pthread_t *)alloca(n_threads * sizeof(pthread_t));
    tf_gemm_fused2_task *tasks = (tf_gemm_fused2_task *)alloca(n_threads * sizeof(tf_gemm_fused2_task));
    int rows_per = n_rows / n_threads;
    int extra = n_rows % n_threads;
    int offset = 0;
    for (int i = 0; i < n_threads; i++) {
        int count = rows_per + (i < extra ? 1 : 0);
        tasks[i] = (tf_gemm_fused2_task){Y1, Y2, W1, W2, X, offset, offset + count,
                                          K, N, out_stride, out_stride, X_stride};
        offset += count;
        pthread_create(&threads[i], NULL, tf_gemm_fused2_worker, &tasks[i]);
    }
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
}

/* ---- Vectorized batch helpers ---- */

#if defined(__AVX2__) && defined(__FMA__)
/* Fast AVX2 exp approximation (Schraudolph-style, ~1e-4 relative error).
 * Based on: exp(x) ≈ 2^(x/ln2) using float bit manipulation. */
static inline __m256 fast_exp_avx2(__m256 x) {
    const __m256 log2e  = _mm256_set1_ps(1.442695040f);
    const __m256 c0     = _mm256_set1_ps(12582912.0f); /* 1.5 * 2^23 (magic bias) */
    const __m256 c1     = _mm256_set1_ps(1065353216.0f); /* 127 * 2^23 */
    const __m256 c2     = _mm256_set1_ps(8388608.0f);  /* 2^23 */
    /* Polynomial coefficients for fractional part */
    const __m256 p0     = _mm256_set1_ps(0.9999999f);
    const __m256 p1     = _mm256_set1_ps(0.6931472f);
    const __m256 p2     = _mm256_set1_ps(0.2402265f);
    const __m256 p3     = _mm256_set1_ps(0.0554953f);
    const __m256 p4     = _mm256_set1_ps(0.0096813f);
    const __m256 clamp_hi = _mm256_set1_ps(88.0f);
    const __m256 clamp_lo = _mm256_set1_ps(-88.0f);

    x = _mm256_max_ps(_mm256_min_ps(x, clamp_hi), clamp_lo);
    __m256 z = _mm256_fmadd_ps(x, log2e, c0);
    __m256 n_f = _mm256_sub_ps(z, c0);  /* floor(x * log2e) */
    __m256 f = _mm256_fmsub_ps(x, log2e, n_f); /* fractional part */
    /* 2^n via integer shift */
    __m256i n_i = _mm256_cvtps_epi32(n_f);
    __m256i pow2n = _mm256_slli_epi32(_mm256_add_epi32(n_i, _mm256_set1_epi32(127)), 23);
    /* Polynomial: p0 + f*(p1 + f*(p2 + f*(p3 + f*p4))) */
    __m256 poly = _mm256_fmadd_ps(f, p4, p3);
    poly = _mm256_fmadd_ps(f, poly, p2);
    poly = _mm256_fmadd_ps(f, poly, p1);
    poly = _mm256_fmadd_ps(f, poly, p0);
    return _mm256_mul_ps(poly, _mm256_castsi256_ps(pow2n));
}

/* Vectorized SiLU(gate) × up for a contiguous array: out[i] = silu(gate[i]) * up[i] */
static void tf_silu_mul_avx2(float *out, const float *gate, const float *up, int n) {
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        /* silu(g) = g / (1 + exp(-g)) = g * sigmoid(g) */
        __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, fast_exp_avx2(neg_g)));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(g, sig), u));
    }
    for (; i < n; i++) {
        float g = gate[i];
        out[i] = g / (1.0f + expf(-g)) * up[i];
    }
}
#else
static void tf_silu_mul_avx2(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        out[i] = g / (1.0f + expf(-g)) * up[i];
    }
}
#endif

/* Batched M-RoPE with precomputed frequency table.
 * Computes freq[j] and sector_pos_map[j] once, then applies to all N tokens.
 * freq[j] = 1/pow(base, 2j/head_dim), pos_idx[j] = which of {t,h,w} to use. */
static void tf_rope_mrope_batch(float *bq, float *bk, int N, int n_heads, int n_kv_heads,
                                 int head_dim, int q_dim, int kv_dim,
                                 const int *pos_t, const int *pos_h, const int *pos_w,
                                 float freq_base, const int sections[4]) {
    int half_dim = head_dim / 2;
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    if (sect_dims <= 0) sect_dims = half_dim;

    /* Precompute per-dimension: freq[j] and which position axis to use (0=t,1=h,2=w) */
    float *freq = (float *)alloca(half_dim * sizeof(float));
    int *pos_axis = (int *)alloca(half_dim * sizeof(int));

    for (int j = 0; j < half_dim; j++) {
        freq[j] = 1.0f / powf(freq_base, (float)(2 * j) / head_dim);
        int sector = j % sect_dims;
        if (sector % 3 == 1 && sector < 3 * sections[1]) {
            pos_axis[j] = 1; /* height */
        } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
            pos_axis[j] = 2; /* width */
        } else {
            pos_axis[j] = 0; /* temporal (default) */
        }
    }

    /* Precompute cos/sin tables per token: [N][half_dim] */
    float *cos_tab = (float *)malloc((size_t)N * half_dim * sizeof(float));
    float *sin_tab = (float *)malloc((size_t)N * half_dim * sizeof(float));

    for (int t = 0; t < N; t++) {
        float *ct = cos_tab + (size_t)t * half_dim;
        float *st = sin_tab + (size_t)t * half_dim;
        for (int j = 0; j < half_dim; j++) {
            int pos;
            switch (pos_axis[j]) {
                case 1: pos = pos_h[t]; break;
                case 2: pos = pos_w[t]; break;
                default: pos = pos_t[t]; break;
            }
            float theta = pos * freq[j];
            ct[j] = cosf(theta);
            st[j] = sinf(theta);
        }
    }

    /* Apply rotation to Q: [N, n_heads, head_dim] */
    for (int t = 0; t < N; t++) {
        const float *ct = cos_tab + (size_t)t * half_dim;
        const float *st = sin_tab + (size_t)t * half_dim;
        for (int h = 0; h < n_heads; h++) {
            float *v = bq + (size_t)t * q_dim + h * head_dim;
            int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
            for (; j + 7 < half_dim; j += 8) {
                __m256 vc = _mm256_loadu_ps(ct + j);
                __m256 vs = _mm256_loadu_ps(st + j);
                __m256 v0 = _mm256_loadu_ps(v + j);
                __m256 v1 = _mm256_loadu_ps(v + j + half_dim);
                _mm256_storeu_ps(v + j,            _mm256_fmsub_ps(v0, vc, _mm256_mul_ps(v1, vs)));
                _mm256_storeu_ps(v + j + half_dim, _mm256_fmadd_ps(v0, vs, _mm256_mul_ps(v1, vc)));
            }
#endif
            for (; j < half_dim; j++) {
                float v0 = v[j], v1 = v[j + half_dim];
                v[j]            = v0 * ct[j] - v1 * st[j];
                v[j + half_dim] = v0 * st[j] + v1 * ct[j];
            }
        }
    }

    /* Apply rotation to K: [N, n_kv_heads, head_dim] */
    for (int t = 0; t < N; t++) {
        const float *ct = cos_tab + (size_t)t * half_dim;
        const float *st = sin_tab + (size_t)t * half_dim;
        for (int h = 0; h < n_kv_heads; h++) {
            float *v = bk + (size_t)t * kv_dim + h * head_dim;
            int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
            for (; j + 7 < half_dim; j += 8) {
                __m256 vc = _mm256_loadu_ps(ct + j);
                __m256 vs = _mm256_loadu_ps(st + j);
                __m256 v0 = _mm256_loadu_ps(v + j);
                __m256 v1 = _mm256_loadu_ps(v + j + half_dim);
                _mm256_storeu_ps(v + j,            _mm256_fmsub_ps(v0, vc, _mm256_mul_ps(v1, vs)));
                _mm256_storeu_ps(v + j + half_dim, _mm256_fmadd_ps(v0, vs, _mm256_mul_ps(v1, vc)));
            }
#endif
            for (; j < half_dim; j++) {
                float v0 = v[j], v1 = v[j + half_dim];
                v[j]            = v0 * ct[j] - v1 * st[j];
                v[j + half_dim] = v0 * st[j] + v1 * ct[j];
            }
        }
    }

    free(cos_tab);
    free(sin_tab);
}

/* Batched RMSNorm: apply to N tokens. bx[i] = rmsnorm(bx_in[i], w) */
static void tf_rmsnorm_batch(float *dst, const float *src, const qtensor *w,
                              int n_embd, int N, float eps, float *w_buf) {
    tf_dequant_row(w, 0, w_buf);
    for (int t = 0; t < N; t++) {
        const float *xi = src + (size_t)t * n_embd;
        float *yi = dst + (size_t)t * n_embd;
        float ss = 0.0f;
        for (int i = 0; i < n_embd; i++) ss += xi[i] * xi[i];
        ss = 1.0f / sqrtf(ss / n_embd + eps);
        for (int i = 0; i < n_embd; i++) yi[i] = xi[i] * ss * w_buf[i];
    }
}

/* Batched QK-norm: per-head RMSNorm for N tokens */
static void tf_qk_norm_batch(float *vec, int n_heads, int head_dim, int N,
                               const qtensor *norm_w, float eps, float *w_buf) {
    tf_dequant_row(norm_w, 0, w_buf);
    int vec_dim = n_heads * head_dim;
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *v = vec + (size_t)t * vec_dim + h * head_dim;
            float ss = 0.0f;
            for (int i = 0; i < head_dim; i++) ss += v[i] * v[i];
            ss = 1.0f / sqrtf(ss / head_dim + eps);
            for (int i = 0; i < head_dim; i++) v[i] = v[i] * ss * w_buf[i];
        }
    }
}

/* Batched causal attention worker: processes a range of (token, head) work items.
 * Work item i maps to token = i / n_heads, head = i % n_heads. */
typedef struct {
    const float *bq;           /* [N, q_dim] queries */
    float *bxb2;               /* [N, q_dim] output (each head writes its own slice) */
    const float *key_cache;    /* layer key cache */
    const float *value_cache;  /* layer value cache */
    const int *cache_pos;      /* [N] cache positions */
    float *att_scratch;        /* per-thread scratch [max_seq_len] */
    int work_start, work_end;  /* range of flattened (tok*n_heads + head) indices */
    int N, n_heads, n_kv_heads;
    int head_dim, kv_dim, q_dim, gqa_ratio, max_seq_len;
} tf_batch_attn_task;

static void *tf_batch_attn_worker(void *arg) {
    tf_batch_attn_task *t = (tf_batch_attn_task *)arg;
    float scale = 1.0f / sqrtf((float)t->head_dim);
    int hd = t->head_dim;

    for (int wi = t->work_start; wi < t->work_end; wi++) {
        int tok = wi / t->n_heads;
        int h   = wi % t->n_heads;
        int kv_h = h / t->gqa_ratio;
        int seq_len = t->cache_pos[tok] + 1;

        const float *q_h = t->bq + (size_t)tok * t->q_dim + h * hd;
        float *att_h = t->att_scratch;

        /* QK scores: att[p] = dot(q, k[p]) * scale */
#if defined(__AVX2__) && defined(__FMA__)
        if (hd == 64) {
            /* Fully unrolled for head_dim=64: 8 AVX vectors */
            __m256 q0 = _mm256_loadu_ps(q_h);
            __m256 q1 = _mm256_loadu_ps(q_h+8);
            __m256 q2 = _mm256_loadu_ps(q_h+16);
            __m256 q3 = _mm256_loadu_ps(q_h+24);
            __m256 q4 = _mm256_loadu_ps(q_h+32);
            __m256 q5 = _mm256_loadu_ps(q_h+40);
            __m256 q6 = _mm256_loadu_ps(q_h+48);
            __m256 q7 = _mm256_loadu_ps(q_h+56);
            __m256 vscale = _mm256_set1_ps(scale);

            int p = 0;
            for (; p + 3 < seq_len; p += 4) {
                /* Process 4 positions at once to amortize q loads */
                const float *k0 = t->key_cache + (size_t)(p+0) * t->kv_dim + kv_h * hd;
                const float *k1 = t->key_cache + (size_t)(p+1) * t->kv_dim + kv_h * hd;
                const float *k2 = t->key_cache + (size_t)(p+2) * t->kv_dim + kv_h * hd;
                const float *k3 = t->key_cache + (size_t)(p+3) * t->kv_dim + kv_h * hd;

                __m256 s0=_mm256_mul_ps(q0, _mm256_loadu_ps(k0));
                __m256 s1=_mm256_mul_ps(q0, _mm256_loadu_ps(k1));
                __m256 s2=_mm256_mul_ps(q0, _mm256_loadu_ps(k2));
                __m256 s3=_mm256_mul_ps(q0, _mm256_loadu_ps(k3));
                s0=_mm256_fmadd_ps(q1, _mm256_loadu_ps(k0+8), s0);
                s1=_mm256_fmadd_ps(q1, _mm256_loadu_ps(k1+8), s1);
                s2=_mm256_fmadd_ps(q1, _mm256_loadu_ps(k2+8), s2);
                s3=_mm256_fmadd_ps(q1, _mm256_loadu_ps(k3+8), s3);
                s0=_mm256_fmadd_ps(q2, _mm256_loadu_ps(k0+16), s0);
                s1=_mm256_fmadd_ps(q2, _mm256_loadu_ps(k1+16), s1);
                s2=_mm256_fmadd_ps(q2, _mm256_loadu_ps(k2+16), s2);
                s3=_mm256_fmadd_ps(q2, _mm256_loadu_ps(k3+16), s3);
                s0=_mm256_fmadd_ps(q3, _mm256_loadu_ps(k0+24), s0);
                s1=_mm256_fmadd_ps(q3, _mm256_loadu_ps(k1+24), s1);
                s2=_mm256_fmadd_ps(q3, _mm256_loadu_ps(k2+24), s2);
                s3=_mm256_fmadd_ps(q3, _mm256_loadu_ps(k3+24), s3);
                s0=_mm256_fmadd_ps(q4, _mm256_loadu_ps(k0+32), s0);
                s1=_mm256_fmadd_ps(q4, _mm256_loadu_ps(k1+32), s1);
                s2=_mm256_fmadd_ps(q4, _mm256_loadu_ps(k2+32), s2);
                s3=_mm256_fmadd_ps(q4, _mm256_loadu_ps(k3+32), s3);
                s0=_mm256_fmadd_ps(q5, _mm256_loadu_ps(k0+40), s0);
                s1=_mm256_fmadd_ps(q5, _mm256_loadu_ps(k1+40), s1);
                s2=_mm256_fmadd_ps(q5, _mm256_loadu_ps(k2+40), s2);
                s3=_mm256_fmadd_ps(q5, _mm256_loadu_ps(k3+40), s3);
                s0=_mm256_fmadd_ps(q6, _mm256_loadu_ps(k0+48), s0);
                s1=_mm256_fmadd_ps(q6, _mm256_loadu_ps(k1+48), s1);
                s2=_mm256_fmadd_ps(q6, _mm256_loadu_ps(k2+48), s2);
                s3=_mm256_fmadd_ps(q6, _mm256_loadu_ps(k3+48), s3);
                s0=_mm256_fmadd_ps(q7, _mm256_loadu_ps(k0+56), s0);
                s1=_mm256_fmadd_ps(q7, _mm256_loadu_ps(k1+56), s1);
                s2=_mm256_fmadd_ps(q7, _mm256_loadu_ps(k2+56), s2);
                s3=_mm256_fmadd_ps(q7, _mm256_loadu_ps(k3+56), s3);

                /* Horizontal sum each into scalar, multiply by scale */
                /* Use hadd pairs to reduce 4 accumulators efficiently */
                __m256 h01 = _mm256_hadd_ps(s0, s1);  /* [s0a+s0b, s0c+s0d, s1a+s1b, s1c+s1d, ...] */
                __m256 h23 = _mm256_hadd_ps(s2, s3);
                __m256 h0123 = _mm256_hadd_ps(h01, h23); /* each lane has partial sums */
                /* h0123 = [A0, A1, A2, A3 | B0, B1, B2, B3] where A=low128, B=hi128 */
                __m128 lo = _mm256_castps256_ps128(h0123);
                __m128 hi = _mm256_extractf128_ps(h0123, 1);
                __m128 sums = _mm_add_ps(lo, hi);
                sums = _mm_mul_ps(sums, _mm256_castps256_ps128(vscale));
                _mm_storeu_ps(att_h + p, sums);
            }
            for (; p < seq_len; p++) {
                const float *k_p = t->key_cache + (size_t)p * t->kv_dim + kv_h * hd;
                __m256 s = _mm256_mul_ps(q0, _mm256_loadu_ps(k_p));
                s = _mm256_fmadd_ps(q1, _mm256_loadu_ps(k_p+8), s);
                s = _mm256_fmadd_ps(q2, _mm256_loadu_ps(k_p+16), s);
                s = _mm256_fmadd_ps(q3, _mm256_loadu_ps(k_p+24), s);
                s = _mm256_fmadd_ps(q4, _mm256_loadu_ps(k_p+32), s);
                s = _mm256_fmadd_ps(q5, _mm256_loadu_ps(k_p+40), s);
                s = _mm256_fmadd_ps(q6, _mm256_loadu_ps(k_p+48), s);
                s = _mm256_fmadd_ps(q7, _mm256_loadu_ps(k_p+56), s);
                __m128 _hi = _mm256_extractf128_ps(s, 1);
                __m128 _lo = _mm256_castps256_ps128(s);
                __m128 _s = _mm_add_ps(_lo, _hi);
                _s = _mm_add_ps(_s, _mm_movehl_ps(_s, _s));
                _s = _mm_add_ss(_s, _mm_movehdup_ps(_s));
                att_h[p] = _mm_cvtss_f32(_s) * scale;
            }
        } else
#endif
        {
            for (int p = 0; p < seq_len; p++) {
                const float *k_p = t->key_cache + (size_t)p * t->kv_dim + kv_h * hd;
                float score = 0.0f;
                for (int d = 0; d < hd; d++) score += q_h[d] * k_p[d];
                att_h[p] = score * scale;
            }
        }

        tf_softmax(att_h, seq_len);

        /* Weighted V accumulation: out[d] = sum_p att[p] * V[p][d] */
        float *out_h = t->bxb2 + (size_t)tok * t->q_dim + h * hd;
#if defined(__AVX2__) && defined(__FMA__)
        if (hd == 64) {
            __m256 o0=_mm256_setzero_ps(), o1=_mm256_setzero_ps();
            __m256 o2=_mm256_setzero_ps(), o3=_mm256_setzero_ps();
            __m256 o4=_mm256_setzero_ps(), o5=_mm256_setzero_ps();
            __m256 o6=_mm256_setzero_ps(), o7=_mm256_setzero_ps();

            for (int p = 0; p < seq_len; p++) {
                const float *v_p = t->value_cache + (size_t)p * t->kv_dim + kv_h * hd;
                __m256 a = _mm256_set1_ps(att_h[p]);
                o0=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p),    o0);
                o1=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+8),  o1);
                o2=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+16), o2);
                o3=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+24), o3);
                o4=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+32), o4);
                o5=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+40), o5);
                o6=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+48), o6);
                o7=_mm256_fmadd_ps(a, _mm256_loadu_ps(v_p+56), o7);
            }

            _mm256_storeu_ps(out_h,    o0); _mm256_storeu_ps(out_h+8,  o1);
            _mm256_storeu_ps(out_h+16, o2); _mm256_storeu_ps(out_h+24, o3);
            _mm256_storeu_ps(out_h+32, o4); _mm256_storeu_ps(out_h+40, o5);
            _mm256_storeu_ps(out_h+48, o6); _mm256_storeu_ps(out_h+56, o7);
        } else
#endif
        {
            memset(out_h, 0, hd * sizeof(float));
            for (int p = 0; p < seq_len; p++) {
                const float *v_p = t->value_cache + (size_t)p * t->kv_dim + kv_h * hd;
                float a = att_h[p];
                for (int d = 0; d < hd; d++) out_h[d] += a * v_p[d];
            }
        }
    }
    return NULL;
}

/* GEMM-based attention worker: processes a range of Q heads */
typedef struct {
    const float *bq; float *bxb2;
    const float *key_cache; const float *value_cache;
    const int *cache_pos;
    float *att_buf;
    int h_start, h_end;
    int N, S, head_dim, kv_dim, q_dim, gqa_ratio;
    float scale;
} tf_attn_gemm_task;

static void *tf_attn_gemm_worker(void *arg) {
    tf_attn_gemm_task *t = (tf_attn_gemm_task *)arg;
    int hd = t->head_dim;
    int N = t->N;

    for (int h = t->h_start; h < t->h_end; h++) {
        int kv_h = h / t->gqa_ratio;

        for (int i = 0; i < N; i++) {
            const float *q_i = t->bq + (size_t)i * t->q_dim + h * hd;
            int seq_len = t->cache_pos[i] + 1;
            float *att_row = t->att_buf;

#if defined(__AVX2__) && defined(__FMA__)
            if (hd == 64) {
                __m256 q0=_mm256_loadu_ps(q_i), q1=_mm256_loadu_ps(q_i+8);
                __m256 q2=_mm256_loadu_ps(q_i+16), q3=_mm256_loadu_ps(q_i+24);
                __m256 q4=_mm256_loadu_ps(q_i+32), q5=_mm256_loadu_ps(q_i+40);
                __m256 q6=_mm256_loadu_ps(q_i+48), q7=_mm256_loadu_ps(q_i+56);

                int p = 0;
                for (; p + 3 < seq_len; p += 4) {
                    const float *k0 = t->key_cache + (size_t)(p+0)*t->kv_dim + kv_h*hd;
                    const float *k1 = t->key_cache + (size_t)(p+1)*t->kv_dim + kv_h*hd;
                    const float *k2 = t->key_cache + (size_t)(p+2)*t->kv_dim + kv_h*hd;
                    const float *k3 = t->key_cache + (size_t)(p+3)*t->kv_dim + kv_h*hd;
                    __m256 s0=_mm256_mul_ps(q0,_mm256_loadu_ps(k0));
                    __m256 s1=_mm256_mul_ps(q0,_mm256_loadu_ps(k1));
                    __m256 s2=_mm256_mul_ps(q0,_mm256_loadu_ps(k2));
                    __m256 s3=_mm256_mul_ps(q0,_mm256_loadu_ps(k3));
                    s0=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k0+8),s0);
                    s1=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k1+8),s1);
                    s2=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k2+8),s2);
                    s3=_mm256_fmadd_ps(q1,_mm256_loadu_ps(k3+8),s3);
                    s0=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k0+16),s0);
                    s1=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k1+16),s1);
                    s2=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k2+16),s2);
                    s3=_mm256_fmadd_ps(q2,_mm256_loadu_ps(k3+16),s3);
                    s0=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k0+24),s0);
                    s1=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k1+24),s1);
                    s2=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k2+24),s2);
                    s3=_mm256_fmadd_ps(q3,_mm256_loadu_ps(k3+24),s3);
                    s0=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k0+32),s0);
                    s1=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k1+32),s1);
                    s2=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k2+32),s2);
                    s3=_mm256_fmadd_ps(q4,_mm256_loadu_ps(k3+32),s3);
                    s0=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k0+40),s0);
                    s1=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k1+40),s1);
                    s2=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k2+40),s2);
                    s3=_mm256_fmadd_ps(q5,_mm256_loadu_ps(k3+40),s3);
                    s0=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k0+48),s0);
                    s1=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k1+48),s1);
                    s2=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k2+48),s2);
                    s3=_mm256_fmadd_ps(q6,_mm256_loadu_ps(k3+48),s3);
                    s0=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k0+56),s0);
                    s1=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k1+56),s1);
                    s2=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k2+56),s2);
                    s3=_mm256_fmadd_ps(q7,_mm256_loadu_ps(k3+56),s3);
                    __m256 h01=_mm256_hadd_ps(s0,s1);
                    __m256 h23=_mm256_hadd_ps(s2,s3);
                    __m256 h0123=_mm256_hadd_ps(h01,h23);
                    __m128 lo=_mm256_castps256_ps128(h0123);
                    __m128 hi=_mm256_extractf128_ps(h0123,1);
                    _mm_storeu_ps(att_row+p, _mm_mul_ps(_mm_add_ps(lo,hi),_mm_set1_ps(t->scale)));
                }
                for (; p < seq_len; p++) {
                    const float *kp = t->key_cache + (size_t)p*t->kv_dim + kv_h*hd;
                    __m256 s=_mm256_mul_ps(q0,_mm256_loadu_ps(kp));
                    s=_mm256_fmadd_ps(q1,_mm256_loadu_ps(kp+8),s);
                    s=_mm256_fmadd_ps(q2,_mm256_loadu_ps(kp+16),s);
                    s=_mm256_fmadd_ps(q3,_mm256_loadu_ps(kp+24),s);
                    s=_mm256_fmadd_ps(q4,_mm256_loadu_ps(kp+32),s);
                    s=_mm256_fmadd_ps(q5,_mm256_loadu_ps(kp+40),s);
                    s=_mm256_fmadd_ps(q6,_mm256_loadu_ps(kp+48),s);
                    s=_mm256_fmadd_ps(q7,_mm256_loadu_ps(kp+56),s);
                    __m128 _hi=_mm256_extractf128_ps(s,1);
                    __m128 _lo=_mm256_castps256_ps128(s);
                    __m128 _s=_mm_add_ps(_lo,_hi);
                    _s=_mm_add_ps(_s,_mm_movehl_ps(_s,_s));
                    _s=_mm_add_ss(_s,_mm_movehdup_ps(_s));
                    att_row[p] = _mm_cvtss_f32(_s) * t->scale;
                }
            } else
#endif
            {
                for (int p = 0; p < seq_len; p++) {
                    const float *kp = t->key_cache + (size_t)p*t->kv_dim + kv_h*hd;
                    float score = 0;
                    for (int d = 0; d < hd; d++) score += q_i[d] * kp[d];
                    att_row[p] = score * t->scale;
                }
            }

            tf_softmax(att_row, seq_len);

            float *out_h = t->bxb2 + (size_t)i * t->q_dim + h * hd;
#if defined(__AVX2__) && defined(__FMA__)
            if (hd == 64) {
                __m256 o0=_mm256_setzero_ps(), o1=_mm256_setzero_ps();
                __m256 o2=_mm256_setzero_ps(), o3=_mm256_setzero_ps();
                __m256 o4=_mm256_setzero_ps(), o5=_mm256_setzero_ps();
                __m256 o6=_mm256_setzero_ps(), o7=_mm256_setzero_ps();
                for (int p = 0; p < seq_len; p++) {
                    const float *vp = t->value_cache + (size_t)p*t->kv_dim + kv_h*hd;
                    __m256 a = _mm256_set1_ps(att_row[p]);
                    o0=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp),   o0);
                    o1=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+8), o1);
                    o2=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+16),o2);
                    o3=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+24),o3);
                    o4=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+32),o4);
                    o5=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+40),o5);
                    o6=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+48),o6);
                    o7=_mm256_fmadd_ps(a,_mm256_loadu_ps(vp+56),o7);
                }
                _mm256_storeu_ps(out_h,   o0); _mm256_storeu_ps(out_h+8, o1);
                _mm256_storeu_ps(out_h+16,o2); _mm256_storeu_ps(out_h+24,o3);
                _mm256_storeu_ps(out_h+32,o4); _mm256_storeu_ps(out_h+40,o5);
                _mm256_storeu_ps(out_h+48,o6); _mm256_storeu_ps(out_h+56,o7);
            } else
#endif
            {
                memset(out_h, 0, hd * sizeof(float));
                for (int p = 0; p < seq_len; p++) {
                    const float *vp = t->value_cache + (size_t)p*t->kv_dim + kv_h*hd;
                    float a = att_row[p];
                    for (int d = 0; d < hd; d++) out_h[d] += a * vp[d];
                }
            }
        }
    }
    return NULL;
}

/* Timing helper */
static inline double tf_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

float *transformer_forward_batch_logits(transformer_model *m, const transformer_batch *b) {
    if (!m || !b || b->N <= 0) return NULL;
    if (!m->has_lm_head) return NULL;

    /* Pause thread pool during batch forward (uses its own pthread-based GEMM) */
    int pool_was_alive = m->pool_alive;
    if (pool_was_alive) tf_pool_shutdown(m);

    int N = b->N;
    int n_embd = m->n_embd;
    int n_heads = m->n_heads;
    int n_kv_heads = m->n_kv_heads;
    int head_dim = m->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int q_dim = n_heads * head_dim;
    int gqa_ratio = n_heads / n_kv_heads;
    int n_ff = m->n_ff;

    /* Profiling accumulators (ms) */
    double t_attn_norm = 0, t_qkv_gemm = 0, t_qk_norm = 0, t_rope = 0;
    double t_kv_store = 0, t_attention = 0, t_out_proj = 0, t_residual = 0;
    double t_ffn_norm = 0, t_ffn_gemm_up = 0, t_silu = 0, t_ffn_gemm_down = 0;
    double t_deepstack = 0;
    double t0p;

    /* Allocate batch scratch buffers */
    float *bx     = (float *)malloc((size_t)N * n_embd * sizeof(float));
    float *bxb    = (float *)malloc((size_t)N * n_embd * sizeof(float));
    float *bq     = (float *)malloc((size_t)N * q_dim * sizeof(float));
    float *bk     = (float *)malloc((size_t)N * kv_dim * sizeof(float));
    float *bv     = (float *)malloc((size_t)N * kv_dim * sizeof(float));
    float *bxb2   = (float *)malloc((size_t)N * q_dim * sizeof(float));
    float *bffn1  = (float *)malloc((size_t)N * n_ff * sizeof(float));
    float *bffn2  = (float *)malloc((size_t)N * n_ff * sizeof(float));
    float *bffn3  = (float *)malloc((size_t)N * n_ff * sizeof(float));

    /* Per-thread attention score scratch: each thread needs [max_seq_len] */
    float *batch_att_scratch = (float *)malloc((size_t)m->n_threads * m->max_seq_len * sizeof(float));

    /* Copy input embeddings to bx (extract n_embd from embd_stride) */
    for (int t = 0; t < N; t++) {
        memcpy(bx + (size_t)t * n_embd,
               b->embds + (size_t)t * b->embd_stride,
               n_embd * sizeof(float));
    }

    /* Layer loop */
    for (int l = 0; l < m->n_layers; l++) {
        transformer_layer *layer = &m->layers[l];

        /* 1. Attention RMSNorm */
        t0p = tf_time_ms();
        tf_rmsnorm_batch(bxb, bx, &layer->attn_norm, n_embd, N, m->rms_norm_eps, m->matvec_tmp);
        t_attn_norm += tf_time_ms() - t0p;

        /* 2. Q/K/V GEMM */
        t0p = tf_time_ms();
        tf_gemm_f16_mt_tokenmajor(bq, &layer->attn_q, bxb, q_dim, N, q_dim, n_embd, m->n_threads);
        tf_gemm_f16_mt_tokenmajor(bk, &layer->attn_k, bxb, kv_dim, N, kv_dim, n_embd, m->n_threads);
        tf_gemm_f16_mt_tokenmajor(bv, &layer->attn_v, bxb, kv_dim, N, kv_dim, n_embd, m->n_threads);
        t_qkv_gemm += tf_time_ms() - t0p;

        /* 3. QK-Norm */
        t0p = tf_time_ms();
        if (layer->attn_q_norm.data) {
            tf_qk_norm_batch(bq, n_heads, head_dim, N, &layer->attn_q_norm, m->rms_norm_eps, m->matvec_tmp);
        }
        if (layer->attn_k_norm.data) {
            tf_qk_norm_batch(bk, n_kv_heads, head_dim, N, &layer->attn_k_norm, m->rms_norm_eps, m->matvec_tmp);
        }
        t_qk_norm += tf_time_ms() - t0p;

        /* 4. M-RoPE (batched with precomputed cos/sin tables + AVX2) */
        t0p = tf_time_ms();
        tf_rope_mrope_batch(bq, bk, N, n_heads, n_kv_heads, head_dim,
                            q_dim, kv_dim, b->pos_t, b->pos_h, b->pos_w,
                            m->rope_freq_base, m->mrope_sections);
        t_rope += tf_time_ms() - t0p;

        /* 5. Store all K/V into cache first (before attention) */
        t0p = tf_time_ms();
        for (int t = 0; t < N; t++) {
            int cp = b->cache_pos[t];
            memcpy(m->key_cache[l]   + (size_t)cp * kv_dim, bk + (size_t)t * kv_dim, kv_dim * sizeof(float));
            memcpy(m->value_cache[l] + (size_t)cp * kv_dim, bv + (size_t)t * kv_dim, kv_dim * sizeof(float));
        }
        t_kv_store += tf_time_ms() - t0p;

        /* 6. GEMM-based causal attention.
         * For each Q head h: scores[N, max_seq] = Q_h[N, hd] × K_kv[max_seq, hd]^T
         * Then causal mask, softmax, then out[N, hd] = scores[N, max_seq] × V[max_seq, hd].
         * Thread across Q heads (16 heads → 16 threads). */
        t0p = tf_time_ms();
        {
            float scale = 1.0f / sqrtf((float)head_dim);
            int max_pos = b->cache_pos[N-1]; /* last token has max cache_pos */
            int S = max_pos + 1; /* sequence length in cache */

            int nt = m->n_threads;
            if (nt > n_heads) nt = n_heads;
            pthread_t *athreads = (pthread_t *)alloca(nt * sizeof(pthread_t));
            tf_attn_gemm_task *atasks = (tf_attn_gemm_task *)alloca(nt * sizeof(tf_attn_gemm_task));
            int heads_per = n_heads / nt;
            int heads_extra = n_heads % nt;
            int hoff = 0;
            for (int ti = 0; ti < nt; ti++) {
                int hcount = heads_per + (ti < heads_extra ? 1 : 0);
                atasks[ti] = (tf_attn_gemm_task){
                    bq, bxb2, m->key_cache[l], m->value_cache[l],
                    b->cache_pos, batch_att_scratch + (size_t)ti * m->max_seq_len,
                    hoff, hoff + hcount, N, S, head_dim, kv_dim, q_dim, gqa_ratio, scale
                };
                hoff += hcount;
                pthread_create(&athreads[ti], NULL, tf_attn_gemm_worker, &atasks[ti]);
            }
            for (int ti = 0; ti < nt; ti++) pthread_join(athreads[ti], NULL);
        }
        t_attention += tf_time_ms() - t0p;

        /* 7. Output projection GEMM: bxb[N, n_embd] = attn_output[n_embd, q_dim] × bxb2[N, q_dim]^T */
        t0p = tf_time_ms();
        tf_gemm_f16_mt_tokenmajor(bxb, &layer->attn_output, bxb2, n_embd, N, n_embd, q_dim, m->n_threads);
        t_out_proj += tf_time_ms() - t0p;

        /* 8. Residual add */
        t0p = tf_time_ms();
        for (int i = 0; i < N * n_embd; i++) bx[i] += bxb[i];
        t_residual += tf_time_ms() - t0p;

        /* 9. FFN */
        t0p = tf_time_ms();
        tf_rmsnorm_batch(bxb, bx, &layer->ffn_norm, n_embd, N, m->rms_norm_eps, m->matvec_tmp);
        t_ffn_norm += tf_time_ms() - t0p;

        /* Dense SwiGLU FFN */
        t0p = tf_time_ms();
        tf_gemm_f16_mt_tokenmajor(bffn1, &layer->ffn_gate, bxb, n_ff, N, n_ff, n_embd, m->n_threads);
        tf_gemm_f16_mt_tokenmajor(bffn2, &layer->ffn_up,   bxb, n_ff, N, n_ff, n_embd, m->n_threads);
        t_ffn_gemm_up += tf_time_ms() - t0p;

        /* SiLU × mul for all N tokens */
        t0p = tf_time_ms();
        tf_silu_mul_avx2(bffn3, bffn1, bffn2, N * n_ff);
        t_silu += tf_time_ms() - t0p;

        t0p = tf_time_ms();
        tf_gemm_f16_mt_tokenmajor(bxb, &layer->ffn_down, bffn3, n_embd, N, n_embd, n_ff, m->n_threads);
        t_ffn_gemm_down += tf_time_ms() - t0p;

        t0p = tf_time_ms();
        for (int i = 0; i < N * n_embd; i++) bx[i] += bxb[i];
        t_residual += tf_time_ms() - t0p;

        /* 10. DeepStack injection */
        t0p = tf_time_ms();
        if (b->ds_embds && l < m->n_deepstack) {
            for (int t = 0; t < N; t++) {
                if (b->ds_embds[t] && b->ds_embd_stride > n_embd) {
                    const float *ds_slice = b->ds_embds[t] + (1 + l) * n_embd;
                    float *xt = bx + (size_t)t * n_embd;
                    for (int i = 0; i < n_embd; i++) xt[i] += ds_slice[i];
                }
            }
        }

        t_deepstack += tf_time_ms() - t0p;

        if (l == 0 || l == m->n_layers - 1 || (l + 1) % 10 == 0) {
            float *last = bx + (size_t)(N-1) * n_embd;
            float norm = 0.0f;
            for (int i = 0; i < n_embd; i++) norm += last[i] * last[i];
            fprintf(stderr, "  [batch] layer %2d: last token hidden norm = %.4f\n", l, sqrtf(norm));
        }
    }

    /* Print batch profiling summary */
    double t_total = t_attn_norm + t_qkv_gemm + t_qk_norm + t_rope + t_kv_store +
                     t_attention + t_out_proj + t_residual + t_ffn_norm +
                     t_ffn_gemm_up + t_silu + t_ffn_gemm_down + t_deepstack;
    fprintf(stderr, "\n  === Batch Prefill Profile (%d tokens × %d layers) ===\n", N, m->n_layers);
    fprintf(stderr, "  attn_norm:     %7.1f ms (%4.1f%%)\n", t_attn_norm, 100*t_attn_norm/t_total);
    fprintf(stderr, "  QKV GEMM:      %7.1f ms (%4.1f%%)\n", t_qkv_gemm, 100*t_qkv_gemm/t_total);
    fprintf(stderr, "  QK norm:       %7.1f ms (%4.1f%%)\n", t_qk_norm, 100*t_qk_norm/t_total);
    fprintf(stderr, "  RoPE:          %7.1f ms (%4.1f%%)\n", t_rope, 100*t_rope/t_total);
    fprintf(stderr, "  KV store:      %7.1f ms (%4.1f%%)\n", t_kv_store, 100*t_kv_store/t_total);
    fprintf(stderr, "  attention:     %7.1f ms (%4.1f%%)\n", t_attention, 100*t_attention/t_total);
    fprintf(stderr, "  out_proj GEMM: %7.1f ms (%4.1f%%)\n", t_out_proj, 100*t_out_proj/t_total);
    fprintf(stderr, "  residual:      %7.1f ms (%4.1f%%)\n", t_residual, 100*t_residual/t_total);
    fprintf(stderr, "  ffn_norm:      %7.1f ms (%4.1f%%)\n", t_ffn_norm, 100*t_ffn_norm/t_total);
    fprintf(stderr, "  FFN up+gate:   %7.1f ms (%4.1f%%)\n", t_ffn_gemm_up, 100*t_ffn_gemm_up/t_total);
    fprintf(stderr, "  SiLU×mul:      %7.1f ms (%4.1f%%)\n", t_silu, 100*t_silu/t_total);
    fprintf(stderr, "  FFN down:      %7.1f ms (%4.1f%%)\n", t_ffn_gemm_down, 100*t_ffn_gemm_down/t_total);
    fprintf(stderr, "  deepstack:     %7.1f ms (%4.1f%%)\n", t_deepstack, 100*t_deepstack/t_total);
    fprintf(stderr, "  TOTAL:         %7.1f ms\n", t_total);
    fprintf(stderr, "  GEMM total:    %7.1f ms (%4.1f%%)\n",
            t_qkv_gemm + t_out_proj + t_ffn_gemm_up + t_ffn_gemm_down,
            100*(t_qkv_gemm + t_out_proj + t_ffn_gemm_up + t_ffn_gemm_down)/t_total);
    fprintf(stderr, "\n");

    /* Final RMSNorm on last token only */
    float *last_hidden = bx + (size_t)(N-1) * n_embd;
    tf_rmsnorm(m->x, last_hidden, &m->output_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);

    /* LM head matvec on last token */
    tf_qmatvec_pool(m, m->logits, &m->output, m->x, m->n_vocab);

    /* Copy last hidden state to m->x for subsequent single-token decode */
    /* m->x already has the normed state, but we need the pre-norm state for KV continuity */

    free(bx); free(bxb); free(bq); free(bk); free(bv); free(bxb2);
    free(bffn1); free(bffn2); free(bffn3); free(batch_att_scratch);

    /* Restart thread pool for decode phase */
    if (pool_was_alive) tf_pool_start(m);

    return m->logits;
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
