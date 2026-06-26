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

#if defined(__GNUC__) || defined(__clang__)
#define TF_MAYBE_UNUSED __attribute__((unused))
#else
#define TF_MAYBE_UNUSED
#endif

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
    float    scale;      /* optional external scale for safetensors FP8 weights */
    int      has_scale;
} qtensor;

typedef struct {
    qtensor attn_norm;   /* [n_embd] */
    qtensor attn_q;      /* [n_embd, n_embd] */
    qtensor attn_k;      /* [n_kv_dim, n_embd] */
    qtensor attn_v;      /* [n_kv_dim, n_embd] */
    qtensor attn_q_norm; /* [head_dim] */
    qtensor attn_k_norm; /* [head_dim] */
    qtensor attn_q_bias; /* [n_embd] Q projection bias (Qwen2.5-VL) */
    qtensor attn_k_bias; /* [n_kv_dim] K projection bias */
    qtensor attn_v_bias; /* [n_kv_dim] V projection bias */
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

    /* SSM (Delta-Net) tensors — Qwen3.5 hybrid layers */
    qtensor ssm_qkv;       /* [n_embd, qkv_dim] combined Q/K/V input projection */
    qtensor ssm_gate;      /* [n_embd, d_inner] gate (z) projection */
    qtensor ssm_alpha;     /* [n_embd, n_v_heads] alpha projection */
    qtensor ssm_beta;      /* [n_embd, n_v_heads] beta projection */
    qtensor ssm_a;         /* [n_v_heads] fixed decay constants */
    qtensor ssm_dt_bias;   /* [n_v_heads] alpha bias */
    qtensor ssm_conv1d;    /* [conv_kernel, qkv_dim] 1D conv kernel */
    qtensor ssm_norm;      /* [d_state] RMS norm for gated output */
    qtensor ssm_out;       /* [d_inner, n_embd] output projection */

    /* Gemma4 tensors */
    qtensor attn_v_norm;         /* [head_dim] V RMSNorm (Gemma4) */
    qtensor post_attention_norm; /* [n_embd] after attn output, before residual */
    qtensor post_ffw_norm;       /* [n_embd] after FFN output, before residual */
    qtensor layer_output_scale;  /* [1] scalar per layer */
    /* Per-layer embedding (Gemma4) */
    qtensor ple_inp_gate;        /* [n_embd_per_layer, n_embd] gated input */
    qtensor ple_proj;            /* [n_embd, n_embd_per_layer] project back */
    qtensor ple_post_norm;       /* [n_embd] post-projection norm */

    int is_ssm;            /* 1 = Delta-Net SSM layer, 0 = full attention */
    int is_swa;            /* 1 = sliding window attention (Gemma4) */
    int shared_kv_source;  /* layer idx to reuse KV from, -1 = own KV (Gemma4) */
    int has_v_proj;        /* 0 = V uses K (Gemma4 SWA without attn_v tensor) */
    int n_kv_heads;        /* per-layer KV heads (Gemma4: SWA & full-attn can differ,
                            * e.g. 12B has 8 KV heads for SWA but 1 (MQA) for full-attn);
                            * derived from attn_k rows / head_dim. 0 = use model default */
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
    int debug_layers;      /* 1 = print per-layer hidden state norms (debug) */

    /* Hybrid SSM+Attention (Qwen3.5) */
    int is_hybrid;           /* 1 if model has SSM layers */
    int full_attn_interval;  /* every Nth layer is full attention (e.g. 4) */
    int ssm_conv_kernel;     /* conv1d kernel size (4) */
    int ssm_d_state;         /* state dimension per head (128) */
    int ssm_n_group;         /* number of K/Q heads (16) */
    int ssm_dt_rank;         /* number of V heads (48) */
    int ssm_d_inner;         /* inner hidden dim (6144) */
    int ssm_qkv_dim;         /* combined QKV dim (10240) */
    float **conv_state;      /* [n_layers] -> [conv_kernel-1, qkv_dim] per SSM layer */
    int *conv_state_pos;     /* [n_layers] circular buffer write position per SSM layer */
    float **recurrent_state; /* [n_layers] -> [n_v_heads, d_state, d_state] per SSM layer */
    float *conv_w_trans;     /* pre-allocated [conv_k * qkv_dim] for batch-dequant conv weights */
    int n_deepstack;       /* number of deepstack layers (Qwen3-VL) */
    const float *ds_embd;  /* pointer to current full embedding (incl deepstack slices) */
    int ds_embd_stride;    /* total embedding dim = proj_dim * (1 + n_deepstack) */

    /* Gemma4 architecture */
    int is_gemma4;                  /* 1 if gemma4 architecture */
    int swa_window_size;            /* sliding window size (512) */
    int head_dim_full;              /* head dim for full-attention layers (512) */
    int head_dim_swa;               /* head dim for SWA layers (256) */
    int n_embd_per_layer;           /* per-layer embedding dim (256) */
    int n_layer_kv_from_start;      /* layers 0..N-1 have own KV, rest share */
    float final_logit_softcapping;  /* tanh softcap on output logits (30.0) */
    float rope_freq_base_swa;       /* RoPE freq base for SWA layers (10000) */
    float embd_scale;               /* sqrt(n_embd) for token embedding scaling */
    int ffn_activation;             /* 0 = SiLU (default), 1 = GELU (Gemma4) */
    int ffn_gelu_fast;              /* 1 = A64FX fast GELU approximation */
    int ffn_fused_q4;               /* 1 = fused Q4 gate/up + GELU for Gemma4 prefill */
    int ffn_check;                  /* 1 = print one fused-vs-exact FFN tile check */
    int *swa_pattern;               /* [n_layers] 1=SWA, 0=full attention */
    /* Per-layer embedding global tensors */
    qtensor per_layer_token_embd;   /* [n_embd_per_layer*n_layer, n_vocab] */
    qtensor per_layer_model_proj;   /* [n_embd_per_layer*n_layer, n_embd] */
    qtensor per_layer_proj_norm;    /* [n_embd_per_layer*n_layer] */
    /* Proportional RoPE for full-attention layers */
    float *rope_freq_factors;       /* [head_dim_full/2] from rope_freqs.weight */
    float *rope_inv_freq_swa;       /* [head_dim_swa/2] precomputed for SWA */
    /* Per-layer embedding scratch */
    float *ple_buf;                 /* [n_embd_per_layer] */
    float *ple_proj_buf;            /* [n_embd] */
    int current_token_id;           /* stashed for per-layer embd lookup */

    /* Global tensors */
    qtensor token_embd;  /* [n_vocab, n_embd] */
    qtensor output_norm; /* [n_embd] */
    qtensor output;      /* [n_vocab, n_embd] LM head (may be absent for embedding models) */

    /* Per-layer weights */
    transformer_layer *layers;

    /* KV cache: [n_layers][max_seq_len * n_kv_heads * head_dim] */
    float **key_cache;
    float **value_cache;
    void **key_cache_raw;
    void **value_cache_raw;
    int kv_cache_type;  /* 0 = F32, 1 = F16 */

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
    int trace_hidden_norms; /* print per-layer hidden norms during forward */

    /* Multi-threading */
    int n_threads;           /* number of threads (default: 1) */
    float **thread_tmp;      /* per-thread dequant scratch [n_threads][max_dim] */

    /* Precomputed RoPE inverse frequency tables */
    float *rope_inv_freq;        /* [head_dim/2]: 1/powf(base, 2j/head_dim) for standard RoPE */
    float *rope_mrope_inv_freq;  /* [sect_dims]: 1/powf(base, 2j/(2*sect_dims)) for M-RoPE */
    int rope_inv_freq_len;       /* head_dim/2 */
    int rope_mrope_inv_freq_len; /* sect_dims */

    /* Thread pool (persistent workers with cond-var sleep between dispatches) */
    pthread_t *pool_threads;   /* [n_threads] worker threads */
    void *(*pool_fn)(void *);  /* current work function */
    void *pool_args;           /* array of per-thread task structs */
    size_t pool_arg_stride;    /* sizeof(task struct) */
    volatile int pool_phase;   /* incremented to signal work */
    volatile int pool_done;    /* number of workers done */
    int pool_alive;            /* 1 if pool is running */
    pthread_mutex_t pool_mutex;/* protects pool_phase signaling */
    pthread_cond_t pool_cond;  /* workers sleep here between dispatches */
    volatile int bar_count;    /* barrier arrival counter */
    volatile int bar_sense;    /* barrier sense flag (alternates 0/1) */

    /* Tensor parallelism */
    int tp_rank;               /* this rank's position in the TP group (0 if no TP) */
    int tp_size;               /* size of the TP group (1 if no TP) */
    void (*tp_allreduce_fn)(float *buf, int count, void *ctx);  /* allreduce callback */
    void *tp_allreduce_ctx;    /* opaque context passed to allreduce (e.g. parallel_config*) */

    /* NUMA allocator state */
    struct {
        int n_cmgs;                   /* number of CMGs (default: 4) */
        size_t per_cmg_budget;        /* usable bytes per CMG (default: 6GB) */
        size_t per_cmg_used[8];       /* current usage per CMG */
        size_t alignment;             /* minimum alignment (default: 2MB) */
        int enabled;                  /* 0=fallback, 1=active */
    } numa;

    /* Persistent NUMA-distributed 256-aligned bump pool for prefill scratch.
     * Reused across prefill calls (reset, not freed) to avoid per-call malloc/free
     * churn that fragments THP and mis-places pages (progressive prefill slowdown). */
    struct {
        void  *base;   /* mmap arena (NULL until first use) */
        size_t cap;    /* capacity bytes */
        size_t off;    /* bump offset */
    } mpool;
} transformer_model;

transformer_model *transformer_load(gguf_context *gguf, int max_seq_len);
void transformer_free(transformer_model *model);

/* Set number of threads for parallel matmul/attention (default: 1) */
void transformer_set_threads(transformer_model *model, int n_threads);
void transformer_build_panels(transformer_model *model);
void transformer_pool_profile_reset(void);
void transformer_reset_runtime_state(transformer_model *model);
void transformer_set_trace_hidden_norms(transformer_model *model, int enable);
/* Enable double-precision accumulation in matvec/rmsnorm for a higher-fidelity
 * (closer to F64) CPU reference. Slower; intended for oracle/verification use. */
void transformer_set_f64_accum(transformer_model *model, int enable);

/* Configure NUMA-aware weight/buffer distribution across CMGs.
 * Must be called after transformer_set_threads, before inference.
 * Env vars: NUMA_DISTRIBUTE=1 (enable), NUMA_N_CMGS (default 4),
 *           NUMA_CMG_BUDGET_GB (default 7), NUMA_ALIGNMENT (default 2MB). */
void transformer_numa_setup(transformer_model *m, const gguf_context *gguf);

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

/* --- Pipeline-parallel API (for MPI) --- */

/* Process layers [layer_start, layer_end) only.
 * If layer_start == 0: expects model->x already set (e.g. via token embedding).
 * If layer_end == n_layers: applies final RMSNorm.
 * Returns pointer to model->x (hidden state [n_embd]). */
float *transformer_forward_partial(transformer_model *model, int cache_pos,
                                    int layer_start, int layer_end);

/* Compute logits from the current hidden state in model->x.
 * Call after transformer_forward_partial with layer_end == n_layers. */
float *transformer_compute_logits(transformer_model *model);

/* Copy hidden state into/out of model->x for MPI communication */
float *transformer_get_hidden(transformer_model *model);
void transformer_set_hidden(transformer_model *model, const float *hidden);

/* Embed a token into model->x (no layer processing) */
void transformer_embed_token(transformer_model *model, int32_t token_id);

/* --- Tensor-parallel API --- */

/* Configure tensor parallelism. Must be called before forward passes.
 * tp_rank: this rank's index in the TP group [0, tp_size).
 * tp_size: total ranks in the TP group.
 * allreduce_fn: callback to allreduce(sum) a float buffer in-place.
 * allreduce_ctx: opaque context passed to allreduce_fn (e.g. parallel_config*).
 *
 * Constraints: n_heads % tp_size == 0, n_kv_heads % tp_size == 0, n_ff % tp_size == 0. */
void transformer_set_tp(transformer_model *model, int tp_rank, int tp_size,
                         void (*allreduce_fn)(float *buf, int count, void *ctx),
                         void *allreduce_ctx);

/* --- Distributed memory management --- */

/* Free KV cache for layers outside [layer_start, layer_end).
 * Call after transformer_load for PP ranks that only process a layer subset. */
void transformer_free_unused_kv(transformer_model *model, int layer_start, int layer_end);

/* Reallocate KV cache with reduced dimension for TP.
 * Only affects attention layers within [layer_start, layer_end).
 * tp_kv_dim = n_kv_heads / tp_size * head_dim. */
void transformer_resize_kv_for_tp(transformer_model *model,
                                    int layer_start, int layer_end, int tp_kv_dim);

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
float *transformer_prefill_gemm(transformer_model *m, const int32_t *tokens, int n_tokens, int start_pos);

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
#include <unistd.h>

#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
extern void gemm_fp16_BT(int M, int K, int N,
                         const float *A, int lda,
                         const uint16_t *BT_fp16, int ldb,
                         float *C, int ldc) __attribute__((weak));
#endif

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

/* ---- Aligned allocation helpers ---- */
/* Returns zeroed memory aligned to 'alignment' bytes. */
static void *tf_aligned_calloc(size_t alignment, size_t count, size_t elem_size) {
    size_t size = count * elem_size;
    size = (size + alignment - 1) & ~(alignment - 1);
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    memset(p, 0, size);
    return p;
}

/* Allocate aligned memory WITHOUT touching pages.
 * With demand paging, first-touch from a worker thread will place pages
 * on that thread's NUMA node. Safe for buffers that are always written
 * before read (e.g. dequant scratch). */
static void *tf_aligned_alloc_notouch(size_t alignment, size_t size) {
    size = (size + alignment - 1) & ~(alignment - 1);
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
}

static uint16_t tf_f32_to_f16(float f) {
    uint32_t x;
    __builtin_memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t t = mant >> (uint32_t)(1 - exp);
        if (t & 0x1000u) t += 0x2000u;
        return (uint16_t)(sign | (t >> 13));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0));
    }
    if (mant & 0x1000u) {
        mant += 0x2000u;
        if (mant & 0x800000u) {
            mant = 0;
            exp++;
            if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static void tf_kv_store(transformer_model *m, int layer, size_t offset,
                        const float *src, int n) {
    if (m->kv_cache_type == 1) {
        uint16_t *dst = (uint16_t *)m->key_cache_raw[layer] + offset;
        for (int i = 0; i < n; i++) dst[i] = tf_f32_to_f16(src[i]);
    } else {
        memcpy(m->key_cache[layer] + offset, src, (size_t)n * sizeof(float));
    }
}

static void tf_kv_store_value(transformer_model *m, int layer, size_t offset,
                              const float *src, int n) {
    if (m->kv_cache_type == 1) {
        uint16_t *dst = (uint16_t *)m->value_cache_raw[layer] + offset;
        for (int i = 0; i < n; i++) dst[i] = tf_f32_to_f16(src[i]);
    } else {
        memcpy(m->value_cache[layer] + offset, src, (size_t)n * sizeof(float));
    }
}

static inline float tf_kv_load_key(const transformer_model *m, int layer, size_t idx) {
    if (m->kv_cache_type == 1)
        return ggml_fp16_to_fp32(((const uint16_t *)m->key_cache_raw[layer])[idx]);
    return m->key_cache[layer][idx];
}

static inline float tf_kv_load_value(const transformer_model *m, int layer, size_t idx) {
    if (m->kv_cache_type == 1)
        return ggml_fp16_to_fp32(((const uint16_t *)m->value_cache_raw[layer])[idx]);
    return m->value_cache[layer][idx];
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
        case GGML_TYPE_Q4_0: block_size = 32;  type_size = 18;  break;
        case GGML_TYPE_Q8_0: block_size = 32;  type_size = 34;  break;
        case GGML_TYPE_Q4_K: block_size = 256; type_size = 144; break;
        case GGML_TYPE_Q5_K:   block_size = 256; type_size = 176; break;
        case GGML_TYPE_Q6_K:   block_size = 256; type_size = 210; break;
        case GGML_TYPE_IQ2_XXS: block_size = 256; type_size = 66;  break;
        case GGML_TYPE_IQ2_XS: block_size = 256; type_size = 74;  break;
        case GGML_TYPE_IQ3_XXS: block_size = 256; type_size = 98;  break;
        case GGML_TYPE_IQ1_S:  block_size = 256; type_size = 50;  break;
        case GGML_TYPE_IQ4_NL: block_size = 32;  type_size = 18;  break;
        case GGML_TYPE_IQ3_S:  block_size = 256; type_size = 110; break;
        case GGML_TYPE_IQ2_S:  block_size = 256; type_size = 82;  break;
        case GGML_TYPE_IQ4_XS: block_size = 256; type_size = 136; break;
        case GGML_TYPE_IQ1_M:  block_size = 256; type_size = 56;  break;
        case GGML_TYPE_F32:    block_size = 1;   type_size = 4;   break;
        case GGML_TYPE_F16:    block_size = 1;   type_size = 2;   break;
        case GGML_TYPE_BF16:   block_size = 1;   type_size = 2;   break;
        default:
            fprintf(stderr, "tf_dequant_row: unsupported type %u\n", t->type);
            memset(dst, 0, (size_t)n_cols * sizeof(float));
            return;
    }

    size_t row_bytes = (size_t)((n_cols + block_size - 1) / block_size) * type_size;
    const void *row_data = (const uint8_t *)t->data + (size_t)row * row_bytes;
    dequant_row(t->type, row_data, dst, n_cols);
}

static int tf_is_supported_weight_type(uint32_t type) {
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
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
        case GGML_TYPE_Q4_0: block_size = 32;  type_size = 18;  break;
        case GGML_TYPE_Q8_0: block_size = 32;  type_size = 34;  break;
        case GGML_TYPE_Q4_K: block_size = 256; type_size = 144; break;
        case GGML_TYPE_Q5_K:   block_size = 256; type_size = 176; break;
        case GGML_TYPE_Q6_K:   block_size = 256; type_size = 210; break;
        case GGML_TYPE_IQ2_XXS: block_size = 256; type_size = 66;  break;
        case GGML_TYPE_IQ2_XS: block_size = 256; type_size = 74;  break;
        case GGML_TYPE_IQ3_XXS: block_size = 256; type_size = 98;  break;
        case GGML_TYPE_IQ1_S:  block_size = 256; type_size = 50;  break;
        case GGML_TYPE_IQ4_NL: block_size = 32;  type_size = 18;  break;
        case GGML_TYPE_IQ3_S:  block_size = 256; type_size = 110; break;
        case GGML_TYPE_IQ2_S:  block_size = 256; type_size = 82;  break;
        case GGML_TYPE_IQ4_XS: block_size = 256; type_size = 136; break;
        case GGML_TYPE_IQ1_M:  block_size = 256; type_size = 56;  break;
        case GGML_TYPE_F32:    block_size = 1;   type_size = 4;   break;
        case GGML_TYPE_F16:    block_size = 1;   type_size = 2;   break;
        case GGML_TYPE_BF16:   block_size = 1;   type_size = 2;   break;
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
    if (mat->type == GGML_TYPE_BF16) {
        for (int i = 0; i < rows_per_expert; i++) {
            const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
            dst[i] = vec_dot_bf16_f32(row, x, mat->n_cols);
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

/* Vectorized element-wise add: dst[i] += src[i] */
static void tf_vadd(float *dst, const float *src, int n) {
#if defined(__AVX2__)
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
    for (; i < n; i++) dst[i] += src[i];
#else
    for (int i = 0; i < n; i++) dst[i] += src[i];
#endif
}

/* Sum of squares: returns sum(v[i]^2) — used for hidden norm profiling */
static float tf_sum_squares(const float *v, int n) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(v + i);
        acc = _mm256_fmadd_ps(x, x, acc);
    }
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    float ss = _mm_cvtss_f32(s4);
    for (; i < n; i++) ss += v[i] * v[i];
    return ss;
#else
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += v[i] * v[i];
    return ss;
#endif
}

/* RMSNorm: y[i] = x[i] * w[i] / sqrt(mean(x^2) + eps) */
/* Opt-in double-precision accumulation for a higher-fidelity CPU reference oracle
 * (transformer_set_f64_accum). Forces the matvec dot and rmsnorm sum-of-squares to
 * accumulate in double, so the CPU reference is closer to true F64 ground truth and
 * its own F32 reduction rounding can be distinguished from genuine GPU error. */
static int tf_g_f64_accum = 0;

static void tf_rmsnorm(float *dst, const float *x, const qtensor *w, int n, float eps, float *w_buf) {
    /* Dequant weight */
    tf_dequant_row(w, 0, w_buf);

    if (tf_g_f64_accum) {
        double ss = 0.0;
        for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
        float inv = (float)(1.0 / sqrt(ss / n + (double)eps));
        for (int i = 0; i < n; i++) dst[i] = x[i] * inv * w_buf[i];
        return;
    }

#if defined(__AVX2__) && defined(__FMA__)
    /* AVX2: sum of squares */
    __m256 vss = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vss = _mm256_fmadd_ps(vx, vx, vss);
    }
    __m128 hi = _mm256_extractf128_ps(vss, 1);
    __m128 lo = _mm256_castps256_ps128(vss);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    float ss = _mm_cvtss_f32(s4);
    for (; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);

    /* Fused x * ss * w */
    __m256 vscale = _mm256_set1_ps(ss);
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w_buf + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw));
    }
    for (; i < n; i++) dst[i] = x[i] * ss * w_buf[i];
#else
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) dst[i] = x[i] * ss * w_buf[i];
#endif
}

/* Quantized matrix-vector multiply: dst[i] = sum_j(M[i][j] * x[j]) for i in [0, n_rows) */
typedef struct {
    float *dst;
    const qtensor *mat;
    const float *x;
    int row_start, row_end;
    float *tmp; /* per-thread scratch */
} tf_matvec_task;

static inline float tf_vec_dot_q4_0_f32(const block_q4_0 *row, const float *x, int n_cols);
static inline void tf_vec_dot_q4_0_f32_4x(const block_q4_0 *row,
                                           const float *x0, const float *x1,
                                           const float *x2, const float *x3,
                                           int n_cols, float *s0, float *s1,
                                           float *s2, float *s3);
static inline void tf_vec_dot_q4_0_f32_4row(float *dst,
    const block_q4_0 *r0, const block_q4_0 *r1,
    const block_q4_0 *r2, const block_q4_0 *r3,
    const float *x, int n_cols);
static inline void tf_vec_dot_q4_0_f32_8row(float *dst,
    const block_q4_0 *r0, const block_q4_0 *r1,
    const block_q4_0 *r2, const block_q4_0 *r3,
    const block_q4_0 *r4, const block_q4_0 *r5,
    const block_q4_0 *r6, const block_q4_0 *r7,
    const float *x, int n_cols);
static inline void tf_vec_dot_q4_0_int8_full_8row(float *dst,
    const block_q4_0 *r0, const block_q4_0 *r1,
    const block_q4_0 *r2, const block_q4_0 *r3,
    const block_q4_0 *r4, const block_q4_0 *r5,
    const block_q4_0 *r6, const block_q4_0 *r7,
    const float *x, int n_cols);
static inline void tf_dequant_q4_0_8row_to_int8(const block_q4_0 *const *rows, int8_t *dst, int n_cols, float scale_w);
static inline void tf_dequant_q4_0_8row_strided_to_int8(const uint8_t *base, size_t row_bytes, int8_t *dst, int n_cols, float scale_w);
static inline void tf_vec_dot_q4_0_int8_8row(int32_t *dst, const int8_t *wi8,
                                                const int8_t *xi8, int n_cols);
static inline void tf_quantize_f32_to_int8(const float *x, int8_t *xi8, int n_cols, float *out_inv);
static inline void tf_vec_dot_q4_0_pair_f32(const block_q4_0 *gate,
                                             const block_q4_0 *up,
                                             const float *x, int n_cols,
                                             float *sg, float *su);
static void tf_matvec_q4_0_rows(float *dst, const uint8_t *base, size_t row_bytes,
                                  const float *x, int n_cols, int row_start, int row_end);

static void *tf_qmatvec_worker(void *arg) {
    tf_matvec_task *t = (tf_matvec_task *)arg;
    int n_cols = t->mat->n_cols;
    if (tf_g_f64_accum) {
        /* Reference oracle: dequant each row to F32, dot in double. Uniform across
         * all weight types so the only F32 rounding left is the per-weight dequant. */
        for (int i = t->row_start; i < t->row_end; i++) {
            tf_dequant_row(t->mat, i, t->tmp);
            double sum = 0.0;
            for (int j = 0; j < n_cols; j++) sum += (double)t->tmp[j] * (double)t->x[j];
            t->dst[i] = (float)sum;
        }
        return NULL;
    }
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
    if (t->mat->type == GGML_TYPE_BF16) {
        const uint8_t *base = (const uint8_t *)t->mat->data;
        size_t row_bytes = (size_t)n_cols * 2;
        int i = t->row_start;
        for (; i + 3 < t->row_end; i += 4) {
            matvec_bf16_4row(t->dst + i,
                (const uint16_t *)(base + (size_t)i * row_bytes),
                (const uint16_t *)(base + (size_t)(i+1) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+2) * row_bytes),
                (const uint16_t *)(base + (size_t)(i+3) * row_bytes),
                t->x, n_cols);
        }
        for (; i < t->row_end; i++) {
            const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
            t->dst[i] = vec_dot_bf16_f32(row, t->x, n_cols);
        }
        return NULL;
    }
    if (t->mat->type == GGML_TYPE_Q8_0) {
        int nb = n_cols / 32;
        size_t row_bytes = (size_t)nb * sizeof(block_q8_0);
        const uint8_t *base = (const uint8_t *)t->mat->data;
        for (int i = t->row_start; i < t->row_end; i++) {
            t->dst[i] = vec_dot_q8_0_f32(base + (size_t)i * row_bytes, t->x, n_cols);
        }
        return NULL;
    }
    if (t->mat->type == GGML_TYPE_Q4_0) {
        int nb = n_cols / 32;
        size_t row_bytes = (size_t)nb * sizeof(block_q4_0);
        tf_matvec_q4_0_rows(t->dst, (const uint8_t *)t->mat->data, row_bytes,
                              t->x, n_cols, t->row_start, t->row_end);
        return NULL;
    }
    for (int i = t->row_start; i < t->row_end; i++) {
        tf_dequant_row(t->mat, i, t->tmp);
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 31 < n_cols; j += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(t->tmp + j),      _mm256_loadu_ps(t->x + j),      acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(t->tmp + j + 8),  _mm256_loadu_ps(t->x + j + 8),  acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(t->tmp + j + 16), _mm256_loadu_ps(t->x + j + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(t->tmp + j + 24), _mm256_loadu_ps(t->x + j + 24), acc3);
        }
        for (; j + 7 < n_cols; j += 8)
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(t->tmp + j), _mm256_loadu_ps(t->x + j), acc0);
        acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
        float sum = _mm_cvtss_f32(s4);
        for (; j < n_cols; j++) sum += t->tmp[j] * t->x[j];
#else
        float sum = 0.0f;
        for (int j = 0; j < n_cols; j++) sum += t->tmp[j] * t->x[j];
#endif
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

static void tf_matvec_q8_rows(float *dst, const uint8_t *base, size_t row_bytes,
                                const float *x, int n_cols, int row_start, int row_end) {
    for (int i = row_start; i < row_end; i++) {
        dst[i] = vec_dot_q8_0_f32(base + (size_t)i * row_bytes, x, n_cols);
    }
}

static inline float tf_vec_dot_q4_0_f32(const block_q4_0 *row, const float *x, int n_cols) {
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t acc = svdup_f32(0.0f);
    svbool_t pg = svptrue_b32();
    int nb = n_cols / 32;
    if (nb > 0) __builtin_prefetch(row->qs, 0, 0);
    for (int b = 0; b < nb; b++) {
        const float d = ggml_fp16_to_fp32(row[b].d);
        const int base = b * 32;
        svuint32_t q = svld1ub_u32(pg, row[b].qs);
        svint32_t qlo = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, q, 0x0f)), 8);
        svint32_t qhi = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, q, 4)), 8);
        svfloat32_t wlo = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qlo), d);
        svfloat32_t whi = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qhi), d);
        acc = svmla_x(pg, acc, wlo, svld1(pg, x + base));
        acc = svmla_x(pg, acc, whi, svld1(pg, x + base + 16));
        if (b + 1 < nb) __builtin_prefetch(row[b+1].qs, 0, 0);
    }
    return svaddv_f32(pg, acc);
#else
    float s = 0.0f;
    int nb = n_cols / 32;
    for (int b = 0; b < nb; b++) {
        const float d = ggml_fp16_to_fp32(row[b].d);
        const int base = b * 32;
        for (int j = 0; j < 16; j++) {
            const uint8_t q = row[b].qs[j];
            s += ((float)((q & 0x0f) - 8) * d) * x[base + j];
            s += ((float)((q >> 4) - 8) * d) * x[base + j + 16];
        }
    }
    return s;
#endif
}

/* Batched Q4_0 matvec: compute 4 dot products with shared activation load.
 * Loads activation x once from L1, amortizes across 4 weight rows from L2.
 * Reduces activation memory traffic by 4x vs per-row calls (144 B/row -> 36 B/row for 2048 cols). */
static inline void tf_vec_dot_q4_0_f32_4row(float *dst,
    const block_q4_0 *r0, const block_q4_0 *r1,
    const block_q4_0 *r2, const block_q4_0 *r3,
    const float *x, int n_cols) {
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
    svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
    svbool_t pg = svptrue_b32();
    int nb = n_cols / 32;
    if (nb > 0) {
        __builtin_prefetch(r0->qs, 0, 0);
        __builtin_prefetch(r1->qs, 0, 0);
        __builtin_prefetch(r2->qs, 0, 0);
        __builtin_prefetch(r3->qs, 0, 0);
    }
    for (int b = 0; b < nb; b++) {
        int base = b * 32;
        svfloat32_t x_lo = svld1(pg, x + base);
        svfloat32_t x_hi = svld1(pg, x + base + 16);

#define TF_Q4_DOT_ROW(acc, row) do { \
    float d__ = ggml_fp16_to_fp32(row[b].d); \
    svuint32_t q__ = svld1ub_u32(pg, row[b].qs); \
    svint32_t ql__ = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, q__, 0x0f)), 8); \
    svint32_t qh__ = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, q__, 4)), 8); \
    svfloat32_t wl__ = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, ql__), d__); \
    svfloat32_t wh__ = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qh__), d__); \
    acc = svmla_x(pg, acc, wl__, x_lo); \
    acc = svmla_x(pg, acc, wh__, x_hi); \
} while(0)

        TF_Q4_DOT_ROW(a0, r0); TF_Q4_DOT_ROW(a1, r1);
        TF_Q4_DOT_ROW(a2, r2); TF_Q4_DOT_ROW(a3, r3);
#undef TF_Q4_DOT_ROW
        if (b + 1 < nb) {
            __builtin_prefetch(r0[b+1].qs, 0, 0);
            __builtin_prefetch(r1[b+1].qs, 0, 0);
            __builtin_prefetch(r2[b+1].qs, 0, 0);
            __builtin_prefetch(r3[b+1].qs, 0, 0);
        }
    }
    dst[0] = svaddv_f32(pg, a0);
    dst[1] = svaddv_f32(pg, a1);
    dst[2] = svaddv_f32(pg, a2);
    dst[3] = svaddv_f32(pg, a3);
#else
    dst[0] = tf_vec_dot_q4_0_f32(r0, x, n_cols);
    dst[1] = tf_vec_dot_q4_0_f32(r1, x, n_cols);
    dst[2] = tf_vec_dot_q4_0_f32(r2, x, n_cols);
    dst[3] = tf_vec_dot_q4_0_f32(r3, x, n_cols);
#endif
}

/* 8-row Q4_0 matvec: compute 8 dot products with shared activation load.
 * Doubles compute-to-load ratio vs 4-row: 8 FMAs per activation block load.
 * A64FX has 32 SVE registers @ 512-bit = 2048 bytes; 8 accumulators x 4 vectors
 * = 32 regs, with wl/wh/temps spilling to stack. May be register-bound. */
static inline void tf_vec_dot_q4_0_f32_8row(float *dst,
    const block_q4_0 *r0, const block_q4_0 *r1,
    const block_q4_0 *r2, const block_q4_0 *r3,
    const block_q4_0 *r4, const block_q4_0 *r5,
    const block_q4_0 *r6, const block_q4_0 *r7,
    const float *x, int n_cols) {
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
    svbool_t pg = svptrue_b32();
    int nb = n_cols / 32;
    if (nb > 0) {
        __builtin_prefetch(r0->qs, 0, 0);
        __builtin_prefetch(r1->qs, 0, 0);
        __builtin_prefetch(r2->qs, 0, 0);
        __builtin_prefetch(r3->qs, 0, 0);
        __builtin_prefetch(r4->qs, 0, 0);
        __builtin_prefetch(r5->qs, 0, 0);
        __builtin_prefetch(r6->qs, 0, 0);
        __builtin_prefetch(r7->qs, 0, 0);
    }
    for (int b = 0; b < nb; b++) {
        int base = b * 32;
        svfloat32_t x_lo = svld1(pg, x + base);
        svfloat32_t x_hi = svld1(pg, x + base + 16);

#define TF_Q4_DOT_ROW(acc, row) do { \
    float d__ = ggml_fp16_to_fp32(row[b].d); \
    svuint32_t q__ = svld1ub_u32(pg, row[b].qs); \
    svint32_t ql__ = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, q__, 0x0f)), 8); \
    svint32_t qh__ = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, q__, 4)), 8); \
    svfloat32_t wl__ = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, ql__), d__); \
    svfloat32_t wh__ = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qh__), d__); \
    acc = svmla_x(pg, acc, wl__, x_lo); \
    acc = svmla_x(pg, acc, wh__, x_hi); \
} while(0)

        TF_Q4_DOT_ROW(a0,r0);TF_Q4_DOT_ROW(a1,r1);
        TF_Q4_DOT_ROW(a2,r2);TF_Q4_DOT_ROW(a3,r3);
        TF_Q4_DOT_ROW(a4,r4);TF_Q4_DOT_ROW(a5,r5);
        TF_Q4_DOT_ROW(a6,r6);TF_Q4_DOT_ROW(a7,r7);
#undef TF_Q4_DOT_ROW
        if (b + 1 < nb) {
            __builtin_prefetch(r0[b+1].qs, 0, 0);
            __builtin_prefetch(r1[b+1].qs, 0, 0);
            __builtin_prefetch(r2[b+1].qs, 0, 0);
            __builtin_prefetch(r3[b+1].qs, 0, 0);
            __builtin_prefetch(r4[b+1].qs, 0, 0);
            __builtin_prefetch(r5[b+1].qs, 0, 0);
            __builtin_prefetch(r6[b+1].qs, 0, 0);
            __builtin_prefetch(r7[b+1].qs, 0, 0);
        }
    }
    dst[0]=svaddv_f32(pg,a0);dst[1]=svaddv_f32(pg,a1);
    dst[2]=svaddv_f32(pg,a2);dst[3]=svaddv_f32(pg,a3);
    dst[4]=svaddv_f32(pg,a4);dst[5]=svaddv_f32(pg,a5);
    dst[6]=svaddv_f32(pg,a6);dst[7]=svaddv_f32(pg,a7);
#else
    dst[0] = tf_vec_dot_q4_0_f32(r0, x, n_cols);
    dst[1] = tf_vec_dot_q4_0_f32(r1, x, n_cols);
    dst[2] = tf_vec_dot_q4_0_f32(r2, x, n_cols);
    dst[3] = tf_vec_dot_q4_0_f32(r3, x, n_cols);
    dst[4] = tf_vec_dot_q4_0_f32(r4, x, n_cols);
    dst[5] = tf_vec_dot_q4_0_f32(r5, x, n_cols);
    dst[6] = tf_vec_dot_q4_0_f32(r6, x, n_cols);
    dst[7] = tf_vec_dot_q4_0_f32(r7, x, n_cols);
#endif
}

/* ================================================================
 * Q4_0 int8 SDOT paths: two variants.
 *
 * 1) ON-THE-FLY int8 path: opt-in via TF_USE_INT8_SDOT_Q4_0. Dequantizes
 *    Q4_0 to int8 in the matvec hot path, then uses SVE SDOT. Correct
 *    for general Q4_0 weights via per-tensor d rescale.
 *
 *    qlair profile for 8-row 2048x64 Q4_0 matvec (real data):
 *      fp32 FMA 8-row:    67,975 cycles (microbench)
 *      int8 SDOT (on-fly): ~30K cycles (microbench, 2.3x faster)
 *      int8 SDOT 2048x2048 matvec: 80.7M cycles (SLOWER than fp32
 *        because the dequant overhead dominates the SDOT throughput).
 *
 * 2) PREQUANTIZED int8 path: weights are dequantized to int8 once
 *    (at load time, via tf_q4_0_int8_cache_init), then the matvec is
 *    pure SDOT (no dequant, no d-scan, no scale_w computation in the
 *    hot path). For prefill (same weights × many tokens) and any
 *    repeated matvec with the same weights, this is a clear win.
 *
 *    qlair profile for 8-row 2048x64 Q4_0 matvec (real data):
 *      prequant int8:     ~13K cycles (microbench, 5.2x faster than fp32)
 *      prequant int8 2048x2048 matvec: 16.5M cycles (1.83x faster
 *        than fp32's 30.2M cycles for the same matvec).
 *
 * Strategy (both paths):
 *  1. Scan max(|d|) over all blocks in the matvec's row batch
 *  2. Dequantize Q4_0 -> int8 with scale_w = 127 / (8 * max_d),
 *     so weight_int8 = round((nibble-8) * d * scale_w) uses the
 *     full int8 range. Subnormal d (e.g., 1e-3) is preserved at
 *     proportional precision.
 *  3. Quantize x -> int8 with x_inv = 127 / max(|x|)
 *  4. SDOT (4 int8 mults per int32 lane, 64 mults per 512-bit call)
 *  5. result_fp32 = sdoti32 / (scale_w * x_inv)
 *
 * SDOT does 4 multiplies per int32 lane (16 dot products of 4 int8
 * per instruction), 2 SDOT/cycle on the 2 ALU pipes. Each SDOT call
 * processes 64 int8 weights (2 Q4_0 blocks) against 64 int8
 * activations, producing 16 int32 partial sums (summed at the end
 * for the final dot product). 32 SDOTs per row, 8 rows in parallel.
 *
 * Precision: int8 quant of weight (nibble-8)*d*scale_w has 1/127
 * step in the rescaled space; back to fp32 this is ~0.5% relative
 * error in the dominant (high-|d|) blocks. Subnormal d blocks are
 * preserved at reduced relative precision (small absolute error
 * since d itself is small). Verified:
 *   unit-scale (d=1.0):   ~1% error per row
 *   random Q4_0 (d ~1e-4..1e3): 2-15% error per row (subnormal d
 *     rounded to 0)
 *   fp32 path:            0% error (gold standard)
 *
 * The on-the-fly int8 path is opt-in via TF_USE_INT8_SDOT_Q4_0. The
 * prequant int8 path uses the tf_q4_0_int8_cache API (build with
 * tf_q4_0_int8_cache_init, use with tf_matvec_q4_0_int8_prequant_rows).
 * The fp32 path is the production default for the existing
 * tf_matvec_q4_0_rows API.
 *
 * Alternative paths evaluated (not recommended):
 *  - int16 SDOT (sve size=0b11, H→D): genuine 2x slower than int8 SDOT
 *    because int16 SDOT does 32 int16 products per SDOT (vs 64 int8 per
 *    int8 SDOT). 2x more SDOTs needed for the same K. Not useful.
 *  - fp16 FMA (svmla_f16): genuine 2x slower than int8 SDOT for the
 *    same reason (32 fp16 products per FMA vs 64 int8 per SDOT). Plus
 *    fp16 accumulator overflow risk for large K. qlair models fp16
 *    FMA via soft-float emulation (slower than hardware). On real
 *    A64FX hardware, fp16 FMA is 2x faster than fp32 FMA (128 GFLOPS
 *    vs 64 GFLOPS), but still 2x slower than int8 SDOT.
 *  - fp16→fp32 widening FMA (FMLAL): SVE 2.0, NOT available on A64FX.
 *    Would have eliminated overflow risk with no throughput penalty
 *    vs narrowing FMA, but A64FX is SVE 1.0 only.
 *
 * Recommendation: int8 SDOT is the best path for throughput on A64FX.
 * fp32 FMA is the gold standard for precision. fp16/int16 paths are
 * 2x slower than int8 SDOT in peak throughput and not recommended.
 * ================================================================ */

/* Find max(|d|) across 8 rows of Q4_0 (n_cols elements per row).
 * Used to set the per-tensor rescale factor for int8 dequant.
 * Each block's d is a single fp16 (2 bytes), so we scalar-load it via
 * ggml_fp16_to_fp32 to avoid svld1_f16 over-reading into qs. */
static inline float tf_q4_0_max_d_8row(const block_q4_0 *const *rows, int n_cols) {
    int nb = n_cols / 32;
    float m = 0.0f;
    for (int r = 0; r < 8; r++) {
        const block_q4_0 *row = rows[r];
        for (int b = 0; b < nb; b++) {
            float d = ggml_fp16_to_fp32(row[b].d);
            float a = d < 0 ? -d : d;
            if (a > m) m = a;
        }
    }
    return m;
}

static inline float tf_q4_0_max_d_4row(const block_q4_0 *const *rows, int n_cols) {
    int nb = n_cols / 32;
    float m = 0.0f;
    for (int r = 0; r < 4; r++) {
        const block_q4_0 *row = rows[r];
        for (int b = 0; b < nb; b++) {
            float d = ggml_fp16_to_fp32(row[b].d);
            float a = d < 0 ? -d : d;
            if (a > m) m = a;
        }
    }
    return m;
}

/* Dequantize 8 rows of Q4_0 to int8 with per-tensor d rescale.
 * Output: 8 * n_cols int8 values, row-major in LINEAR order
 * ([lo0..lo15,hi0..hi15] for each block, blocks contiguous).
 * Per-block d is rescaled to int8: weight_int8 = round((nibble-8) * d * scale_w).
 * Caller passes max_d = max(|d|) over the rows, and scale_w = 127 / (8 * max_d).
 * Subnormal d is preserved (vs the old "fold d" approach that rounded to 0). */
static inline void tf_dequant_q4_0_8row_to_int8(const block_q4_0 *const *rows, int8_t *dst, int n_cols, float scale_w) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svptrue_b8();
    svbool_t pg16 = svwhilelt_b8(0, 16);
    int nb = n_cols / 32;
    int nb_pairs = (n_cols + 63) / 64;
    for (int r = 0; r < 8; r++) {
        const block_q4_0 *row = rows[r];
        int8_t *drow = dst + r * nb_pairs * 64;
        for (int p = 0; p < nb_pairs; p++) {
            svst1_s8(pg, drow + p*64, svdup_n_s8(0));
            for (int b = 0; b < 2 && p*2 + b < nb; b++) {
                int blk = p*2 + b;
                /* qs is only 16 bytes; load with a 16-lane predicate so we
                 * don't over-read 48 bytes past the field (svld1 zeroes the
                 * inactive lanes, and only the first 16 lanes are stored). */
                svuint8_t q = svld1_u8(pg16, row[blk].qs);
                svuint8_t lo = svand_n_u8_x(pg, q, 0x0f);
                svuint8_t hi = svlsr_n_u8_x(pg, q, 4);
                /* Per-block d rescaled to int8 (preserves subnormal d). */
                float d = ggml_fp16_to_fp32(row[blk].d);
                int8_t di = (int8_t)lrintf(d * scale_w);
                if (di >  127) di =  127;
                if (di < -128) di = -128;
                /* Subtract 8 (center) and multiply by di. */
                svint8_t lo8 = svsub_n_s8_x(pg, svreinterpret_s8_u8(lo), 8);
                svint8_t hi8 = svsub_n_s8_x(pg, svreinterpret_s8_u8(hi), 8);
                lo8 = svmul_n_s8_x(pg, lo8, di);
                hi8 = svmul_n_s8_x(pg, hi8, di);
                /* Store as 16 lo's, then 16 hi's (linear order). */
                svst1_s8(pg16, drow + p*64 + b*32,     lo8);
                svst1_s8(pg16, drow + p*64 + b*32 + 16, hi8);
            }
        }
    }
#endif
}

/* Same as tf_dequant_q4_0_8row_to_int8 but takes a base pointer and row stride.
 * Used when rows are stored contiguously (most common case). */
static inline void tf_dequant_q4_0_8row_strided_to_int8(const uint8_t *base, size_t row_bytes, int8_t *dst, int n_cols, float scale_w) {
#if defined(__ARM_FEATURE_SVE)
    const block_q4_0 *rows[8];
    for (int r = 0; r < 8; r++) rows[r] = (const block_q4_0 *)(base + (size_t)r * row_bytes);
    tf_dequant_q4_0_8row_to_int8(rows, dst, n_cols, scale_w);
#endif
}

/* 8-row Q4_0 matvec using SVE int8 SDOT.
 * wi8: pre-dequantized int8 weights, padded to 64 elements per pair (upper 32 = 0).
 * xi8: pre-quantized int8 activations, padded to 64 elements per pair.
 * dst[0..7] receives int32 dot products. Multiply by x_scale to get fp32. */
static inline void tf_vec_dot_q4_0_int8_8row(int32_t *dst, const int8_t *wi8,
                                                const int8_t *xi8, int n_cols) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svptrue_b8();
    svint32_t a0=svdup_s32(0),a1=svdup_s32(0),a2=svdup_s32(0),a3=svdup_s32(0);
    svint32_t a4=svdup_s32(0),a5=svdup_s32(0),a6=svdup_s32(0),a7=svdup_s32(0);
    int nb_pairs = (n_cols + 63) / 64;
    /* wi8 is laid out as 8 rows of nb_pairs * 64 int8 each */
    for (int p = 0; p < nb_pairs; p++) {
        if (p + 1 < nb_pairs) {
            __builtin_prefetch(wi8+0*nb_pairs*64+(p+1)*64, 0, 0);
            __builtin_prefetch(wi8+1*nb_pairs*64+(p+1)*64, 0, 0);
            __builtin_prefetch(wi8+2*nb_pairs*64+(p+1)*64, 0, 0);
            __builtin_prefetch(wi8+3*nb_pairs*64+(p+1)*64, 0, 0);
            __builtin_prefetch(wi8+4*nb_pairs*64+(p+1)*64, 0, 0);
            __builtin_prefetch(wi8+5*nb_pairs*64+(p+1)*64, 0, 0);
            __builtin_prefetch(wi8+6*nb_pairs*64+(p+1)*64, 0, 0);
            __builtin_prefetch(wi8+7*nb_pairs*64+(p+1)*64, 0, 0);
        }
        /* Load x int8 (64 elements, full SVE vector, with upper 32 = 0) */
        svint8_t xv = svld1_s8(pg, xi8 + p*64);
        /* Load 8 rows' int8 weights (64 elements per row, full SVE vector) */
        svint8_t w0=svld1_s8(pg, wi8+0*nb_pairs*64+p*64);
        svint8_t w1=svld1_s8(pg, wi8+1*nb_pairs*64+p*64);
        svint8_t w2=svld1_s8(pg, wi8+2*nb_pairs*64+p*64);
        svint8_t w3=svld1_s8(pg, wi8+3*nb_pairs*64+p*64);
        svint8_t w4=svld1_s8(pg, wi8+4*nb_pairs*64+p*64);
        svint8_t w5=svld1_s8(pg, wi8+5*nb_pairs*64+p*64);
        svint8_t w6=svld1_s8(pg, wi8+6*nb_pairs*64+p*64);
        svint8_t w7=svld1_s8(pg, wi8+7*nb_pairs*64+p*64);
        /* 8 SDOTs (one per row). Each does 16 dot products of 4 int8
         * (64 int8 = 16 lanes × 4 int8/lane). Sum of 16 partial sums
         * is the full dot product for that 64-element chunk. */
        a0=svdot_s32(a0, w0, xv); a1=svdot_s32(a1, w1, xv);
        a2=svdot_s32(a2, w2, xv); a3=svdot_s32(a3, w3, xv);
        a4=svdot_s32(a4, w4, xv); a5=svdot_s32(a5, w5, xv);
        a6=svdot_s32(a6, w6, xv); a7=svdot_s32(a7, w7, xv);
    }
    /* Sum the 16 partial sums per row. */
    dst[0] = svaddv_s32(svptrue_b32(), a0);
    dst[1] = svaddv_s32(svptrue_b32(), a1);
    dst[2] = svaddv_s32(svptrue_b32(), a2);
    dst[3] = svaddv_s32(svptrue_b32(), a3);
    dst[4] = svaddv_s32(svptrue_b32(), a4);
    dst[5] = svaddv_s32(svptrue_b32(), a5);
    dst[6] = svaddv_s32(svptrue_b32(), a6);
    dst[7] = svaddv_s32(svptrue_b32(), a7);
#else
    (void)dst; (void)wi8; (void)xi8; (void)n_cols;
#endif
}

/* Quantize fp32 activations to int8 with per-tensor scale.
 * xi8 = round(x * (127 / max(|x|))). x = xi8 * (max(|x|) / 127).
 * Returns x_inv (= 127 / max(|x|), the multiplier) via *out_scale.
 * Fully SVE-vectorized: max via svmaxv, quantize via svtbl to extract
 * lower bytes from int32 lanes. svnarrow_s32_s8 is SVE2 only, so we
 * use svtbl with a precomputed index table that selects byte 0, 4, 8, ...
 * from the int32 register (1 byte per int32 lane). */
static inline void tf_quantize_f32_to_int8(const float *x, int8_t *xi8, int n_cols, float *out_inv) {
#if defined(__ARM_FEATURE_SVE)
    /* Vectorized max(|x|). */
    svbool_t pg = svptrue_b32();
    svfloat32_t vmax = svdup_f32(0.0f);
    int j = 0;
    for (; j + 15 < n_cols; j += 16) {
        svfloat32_t v = svld1(pg, x + j);
        svfloat32_t va = svabs_f32_x(pg, v);
        vmax = svmax_f32_x(pg, vmax, va);
    }
    float xmax = svmaxv_f32(pg, vmax);
    for (; j < n_cols; j++) {
        float a = x[j] < 0 ? -x[j] : x[j];
        if (a > xmax) xmax = a;
    }
    float x_inv = xmax > 0.0f ? (127.0f / xmax) : 1.0f;
    *out_inv = x_inv;
    /* Vectorized quantize: 16 fp32 -> 16 int32, then svtbl to extract
     * the lower 8 bits of each int32 lane into a 16-byte svint8. */
    static const uint8_t extract_idx[16] __attribute__((aligned(16))) =
        {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60};
    svuint8_t idx = svld1_u8(svptrue_b8(), extract_idx);
    for (j = 0; j + 15 < n_cols; j += 16) {
        svbool_t pgb = svwhilelt_b32(j, j + 16 <= n_cols ? j + 16 : n_cols);
        svfloat32_t v = svld1(pgb, x + j);
        svfloat32_t v_scaled = svmul_n_f32_x(pgb, v, x_inv);
        svfloat32_t v_round = svadd_n_f32_x(pgb, v_scaled, 0.5f);
        svint32_t v_int = svcvt_s32_f32_x(pgb, v_round);
        svint32_t v_clamp = svmin_n_s32_x(pgb, svmax_n_s32_x(pgb, v_int, -128), 127);
        svint8_t v_lo = svreinterpret_s8_s32(v_clamp);
        svint8_t v_i8 = svtbl_s8(v_lo, idx);
        /* Only the first 16 lanes of v_i8 are meaningful (svtbl packed 16
         * int32 lanes to 16 int8); store 16 bytes, not a full 64-byte vector,
         * to avoid writing 48 bytes past xi8. The [n_cols,n_padded) tail is
         * zeroed below. */
        svst1_s8(svwhilelt_b8((uint32_t)0, (uint32_t)16), xi8 + j, v_i8);
    }
    /* Scalar tail for quantize. */
    for (; j < n_cols; j++) {
        float v = x[j] * x_inv;
        int iv = (int)(v + (v < 0 ? -0.5f : 0.5f));
        if (iv >  127) iv =  127;
        if (iv < -128) iv = -128;
        xi8[j] = (int8_t)iv;
    }
    int n_padded = (n_cols + 63) & ~63;
    for (j = n_cols; j < n_padded; j++) xi8[j] = 0;
#else
    float xmax = 0.0f;
    for (int j = 0; j < n_cols; j++) {
        float a = x[j] < 0 ? -x[j] : x[j];
        if (a > xmax) xmax = a;
    }
    float x_inv = xmax > 0.0f ? (127.0f / xmax) : 1.0f;
    for (int k = 0; k < n_cols; k++) {
        float v = x[k] * x_inv;
        int iv = (int)(v + (v < 0 ? -0.5f : 0.5f));
        if (iv >  127) iv =  127;
        if (iv < -128) iv = -128;
        xi8[k] = (int8_t)iv;
    }
    int n_padded = (n_cols + 63) & ~63;
    for (int k = n_cols; k < n_padded; k++) xi8[k] = 0;
    *out_inv = x_inv;
#endif
}

/* 8-row Q4_0 matvec with int8 SDOT.
 * Dequantizes 8 rows of Q4_0 to int8 (padded to 64 per pair), quantizes x to int8
 * (padded to 64 per pair), then does SDOT. dst[0..7] receives fp32 results. */
static inline void tf_vec_dot_q4_0_int8_full_8row(float *dst,
    const block_q4_0 *r0, const block_q4_0 *r1,
    const block_q4_0 *r2, const block_q4_0 *r3,
    const block_q4_0 *r4, const block_q4_0 *r5,
    const block_q4_0 *r6, const block_q4_0 *r7,
    const float *x, int n_cols) {
#if defined(__ARM_FEATURE_SVE)
    /* Scratch buffers (static, lazily allocated; assume n_cols <= 2048) */
    int nb_pairs = (n_cols + 63) / 64;
    static int8_t *wi8 = NULL;
    static int8_t *xi8 = NULL;
    static int wi8_alloc = 0, xi8_alloc = 0;
    int wi8_need = 8 * nb_pairs * 64;
    int xi8_need = nb_pairs * 64;
    if (wi8_need > wi8_alloc) { if (wi8) free(wi8); wi8 = (int8_t *)aligned_alloc(256, wi8_need); wi8_alloc = wi8_need; }
    if (xi8_need > xi8_alloc) { if (xi8) free(xi8); xi8 = (int8_t *)aligned_alloc(256, xi8_need); xi8_alloc = xi8_need; }
    /* Quantize x to int8 with per-tensor scale (padded to 64 per pair) */
    float x_inv;
    tf_quantize_f32_to_int8(x, xi8, n_cols, &x_inv);
    /* Find max_d across all 8 rows; set per-tensor weight scale. */
    const block_q4_0 *rows[8] = {r0, r1, r2, r3, r4, r5, r6, r7};
    float max_d = tf_q4_0_max_d_8row(rows, n_cols);
    float scale_w = (max_d > 0.0f) ? (127.0f / (8.0f * max_d)) : 1.0f;
    /* Dequant 8 rows to int8 with per-tensor d rescale. */
    tf_dequant_q4_0_8row_to_int8(rows, wi8, n_cols, scale_w);
    /* SDOT matvec */
    int32_t acc[8];
    tf_vec_dot_q4_0_int8_8row(acc, wi8, xi8, n_cols);
    /* Scale int32 results by 1/(scale_w * x_inv) to get fp32.
     * (Both scale_w and x_inv are multipliers used to quantize;
     *  dividing by their product dequantizes.) */
    float inv = 1.0f / (scale_w * x_inv);
    for (int i = 0; i < 8; i++) dst[i] = (float)acc[i] * inv;
#else
    dst[0] = tf_vec_dot_q4_0_f32(r0, x, n_cols);
    dst[1] = tf_vec_dot_q4_0_f32(r1, x, n_cols);
    dst[2] = tf_vec_dot_q4_0_f32(r2, x, n_cols);
    dst[3] = tf_vec_dot_q4_0_f32(r3, x, n_cols);
    dst[4] = tf_vec_dot_q4_0_f32(r4, x, n_cols);
    dst[5] = tf_vec_dot_q4_0_f32(r5, x, n_cols);
    dst[6] = tf_vec_dot_q4_0_f32(r6, x, n_cols);
    dst[7] = tf_vec_dot_q4_0_f32(r7, x, n_cols);
#endif
}

/* 4-row variant of tf_dequant_q4_0_8row_to_int8: takes 4 row pointers and
 * dequantizes each to (nb_pairs * 64) int8 with upper 32 of each pair = 0. */
static inline void tf_dequant_q4_0_4row_to_int8(const block_q4_0 *const *rows, int8_t *dst, int n_cols, float scale_w) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svptrue_b8();
    svbool_t pg16 = svwhilelt_b8(0, 16);
    int nb = n_cols / 32;
    int nb_pairs = (n_cols + 63) / 64;
    for (int r = 0; r < 4; r++) {
        const block_q4_0 *row = rows[r];
        int8_t *drow = dst + r * nb_pairs * 64;
        for (int p = 0; p < nb_pairs; p++) {
            svst1_s8(pg, drow + p*64, svdup_n_s8(0));
            for (int b = 0; b < 2 && p*2 + b < nb; b++) {
                int blk = p*2 + b;
                /* qs is only 16 bytes; load with a 16-lane predicate so we
                 * don't over-read 48 bytes past the field (svld1 zeroes the
                 * inactive lanes, and only the first 16 lanes are stored). */
                svuint8_t q = svld1_u8(pg16, row[blk].qs);
                svuint8_t lo = svand_n_u8_x(pg, q, 0x0f);
                svuint8_t hi = svlsr_n_u8_x(pg, q, 4);
                float d = ggml_fp16_to_fp32(row[blk].d);
                int8_t di = (int8_t)lrintf(d * scale_w);
                if (di >  127) di =  127;
                if (di < -128) di = -128;
                svint8_t lo8 = svsub_n_s8_x(pg, svreinterpret_s8_u8(lo), 8);
                svint8_t hi8 = svsub_n_s8_x(pg, svreinterpret_s8_u8(hi), 8);
                lo8 = svmul_n_s8_x(pg, lo8, di);
                hi8 = svmul_n_s8_x(pg, hi8, di);
                svst1_s8(pg16, drow + p*64 + b*32,     lo8);
                svst1_s8(pg16, drow + p*64 + b*32 + 16, hi8);
            }
        }
    }
#endif
}

/* 4-row int8 SDOT matvec (no duplicate-row waste). */
static inline void tf_vec_dot_q4_0_int8_4row(int32_t *dst, const int8_t *wi8,
                                              const int8_t *xi8, int n_cols) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svptrue_b8();
    svint32_t a0=svdup_s32(0),a1=svdup_s32(0),a2=svdup_s32(0),a3=svdup_s32(0);
    int nb_pairs = (n_cols + 63) / 64;
    for (int p = 0; p < nb_pairs; p++) {
        if (p + 1 < nb_pairs) {
            __builtin_prefetch(wi8+0*nb_pairs*64+(p+1)*64, 0, 0);
            __builtin_prefetch(wi8+1*nb_pairs*64+(p+1)*64, 0, 0);
            __builtin_prefetch(wi8+2*nb_pairs*64+(p+1)*64, 0, 0);
            __builtin_prefetch(wi8+3*nb_pairs*64+(p+1)*64, 0, 0);
        }
        svint8_t xv = svld1_s8(pg, xi8 + p*64);
        svint8_t w0=svld1_s8(pg, wi8+0*nb_pairs*64+p*64);
        svint8_t w1=svld1_s8(pg, wi8+1*nb_pairs*64+p*64);
        svint8_t w2=svld1_s8(pg, wi8+2*nb_pairs*64+p*64);
        svint8_t w3=svld1_s8(pg, wi8+3*nb_pairs*64+p*64);
        a0=svdot_s32(a0, w0, xv); a1=svdot_s32(a1, w1, xv);
        a2=svdot_s32(a2, w2, xv); a3=svdot_s32(a3, w3, xv);
    }
    dst[0] = svaddv_s32(svptrue_b32(), a0);
    dst[1] = svaddv_s32(svptrue_b32(), a1);
    dst[2] = svaddv_s32(svptrue_b32(), a2);
    dst[3] = svaddv_s32(svptrue_b32(), a3);
#else
    (void)dst; (void)wi8; (void)xi8; (void)n_cols;
#endif
}

/* 4-row Q4_0 matvec with int8 SDOT. Saves 4 SDOTs per pair vs the
 * 8-row-with-duplicate-row-reuse path. */
static inline void tf_vec_dot_q4_0_int8_full_4row(float *dst,
    const block_q4_0 *r0, const block_q4_0 *r1,
    const block_q4_0 *r2, const block_q4_0 *r3,
    const float *x, int n_cols) {
#if defined(__ARM_FEATURE_SVE)
    int nb_pairs = (n_cols + 63) / 64;
    /* Share the 8-row scratch to avoid extra alloc churn. The 4-row dequant
     * writes into the first 4*nb_pairs*64 bytes; the 4-row SDOT reads the
     * same range. */
    static int8_t *wi8 = NULL;
    static int8_t *xi8 = NULL;
    static int wi8_alloc = 0, xi8_alloc = 0;
    int wi8_need = 8 * nb_pairs * 64;  /* same size as 8-row scratch */
    int xi8_need = nb_pairs * 64;
    if (wi8_need > wi8_alloc) { if (wi8) free(wi8); wi8 = (int8_t *)aligned_alloc(256, wi8_need); wi8_alloc = wi8_need; }
    if (xi8_need > xi8_alloc) { if (xi8) free(xi8); xi8 = (int8_t *)aligned_alloc(256, xi8_need); xi8_alloc = xi8_need; }
    float x_inv;
    tf_quantize_f32_to_int8(x, xi8, n_cols, &x_inv);
    const block_q4_0 *rows[4] = {r0, r1, r2, r3};
    float max_d = tf_q4_0_max_d_4row(rows, n_cols);
    float scale_w = (max_d > 0.0f) ? (127.0f / (8.0f * max_d)) : 1.0f;
    tf_dequant_q4_0_4row_to_int8(rows, wi8, n_cols, scale_w);
    int32_t acc[4];
    tf_vec_dot_q4_0_int8_4row(acc, wi8, xi8, n_cols);
    float inv = 1.0f / (scale_w * x_inv);
    for (int i = 0; i < 4; i++) dst[i] = (float)acc[i] * inv;
#else
    dst[0] = tf_vec_dot_q4_0_f32(r0, x, n_cols);
    dst[1] = tf_vec_dot_q4_0_f32(r1, x, n_cols);
    dst[2] = tf_vec_dot_q4_0_f32(r2, x, n_cols);
    dst[3] = tf_vec_dot_q4_0_f32(r3, x, n_cols);
#endif
}

/* ================================================================
 * Pure int8 SDOT path: pre-dequantized weights.
 *
 * The pre-dequant approach amortizes the dequant cost: weights are
 * dequantized to int8 once (at load time or when weights change), then
 * the matvec hot path is just SDOT (no dequant, no d-scan, no scale_w
 * computation). For prefill (same weights × many tokens) this is a
 * big win; for decode (1 matvec per weight set) the prequant is paid
 * once.
 *
 * Storage format:
 *   For each row: ceil(n_cols/64) * 64 int8 values
 *   The first 32 of each 64-byte pair are the linear-order int8 weights
 *   for one Q4_0 block; the next 32 are for the next block. If n_cols is
 *   not a multiple of 64, the last pair's upper 32 are zero-padded.
 *   Memory cost: ~1.78x the Q4_0 size (4 bits -> 8 bits per element).
 *
 * Per-tensor d rescale is baked into the prequant (same formula as the
 * non-prequant int8 path: scale_w = 127 / (8 * max_d), so the int8
 * weight range covers the dequantized weight range).
 *
 * Matvec: just SDOT (no dequant), then result = sdoti32 / (scale_w * x_inv).
 * For 8 rows × 32 pairs (n_cols=2048): 256 SDOTs (vs 256 SDOTs + 4K dequant
 * ops for the on-the-fly int8 path).
 *
 * qlair profile for 8-row 2048x64 Q4_0 matvec (real data, single matvec):
 *   fp32 FMA 8-row:           30.2M cycles
 *   int8 SDOT (on-the-fly):   80.7M cycles  (dequant dominates)
 *   int8 SDOT (prequant):     <TBD> cycles (only SDOT, no dequant)
 * ================================================================ */

/* Opaque prequantized int8 weight cache. Holds the int8 weights and
 * the scale_w used during prequant. Build with tf_q4_0_int8_cache_init. */
typedef struct {
    int8_t  *wi8;        /* prequantized int8 weights */
    size_t   bytes;      /* size of wi8 in bytes */
    float    scale_w;    /* per-tensor d rescale factor (127 / (8 * max_d)) */
    int      n_rows;
    int      n_cols;
    int      nb_pairs;   /* ceil(n_cols / 64) */
} tf_q4_0_int8_cache;

/* Pre-dequantize Q4_0 weights to int8 with per-tensor d rescale.
 * The Q4_0 data is at 'src' (n_rows rows of n_cols elements each, row
 * stride = row_bytes). Output is stored in cache->wi8 (allocated here,
 * freed by tf_q4_0_int8_cache_free).
 *
 * Returns 0 on success, -1 on failure.
 *
 * Memory cost: n_rows * nb_pairs * 64 bytes (~1.78x the Q4_0 size).
 * For 2048x2048: 2048 * 32 * 64 = 4 MB. */
static inline int tf_q4_0_int8_cache_init(tf_q4_0_int8_cache *cache,
                                            const void *src, size_t row_bytes,
                                            int n_rows, int n_cols) {
    cache->n_rows = n_rows;
    cache->n_cols = n_cols;
    cache->nb_pairs = (n_cols + 63) / 64;
    cache->bytes = (size_t)n_rows * cache->nb_pairs * 64;
    cache->wi8 = (int8_t *)aligned_alloc(256, cache->bytes);
    if (!cache->wi8) return -1;
    /* Scan max(|d|) over all rows. */
    const uint8_t *base = (const uint8_t *)src;
    float max_d = 0.0f;
    int nb = n_cols / 32;
    for (int r = 0; r < n_rows; r++) {
        const block_q4_0 *row = (const block_q4_0 *)(base + (size_t)r * row_bytes);
        for (int b = 0; b < nb; b++) {
            float d = ggml_fp16_to_fp32(row[b].d);
            float a = d < 0 ? -d : d;
            if (a > max_d) max_d = a;
        }
    }
    cache->scale_w = (max_d > 0.0f) ? (127.0f / (8.0f * max_d)) : 1.0f;
    /* Dequant all rows. */
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svptrue_b8();
    svbool_t pg16 = svwhilelt_b8(0, 16);
    for (int r = 0; r < n_rows; r++) {
        const block_q4_0 *row = (const block_q4_0 *)(base + (size_t)r * row_bytes);
        int8_t *drow = cache->wi8 + (size_t)r * cache->nb_pairs * 64;
        for (int p = 0; p < cache->nb_pairs; p++) {
            svst1_s8(pg, drow + p*64, svdup_n_s8(0));
            for (int b = 0; b < 2 && p*2 + b < nb; b++) {
                int blk = p*2 + b;
                /* qs is only 16 bytes; load with a 16-lane predicate so we
                 * don't over-read 48 bytes past the field (svld1 zeroes the
                 * inactive lanes, and only the first 16 lanes are stored). */
                svuint8_t q = svld1_u8(pg16, row[blk].qs);
                svuint8_t lo = svand_n_u8_x(pg, q, 0x0f);
                svuint8_t hi = svlsr_n_u8_x(pg, q, 4);
                float d = ggml_fp16_to_fp32(row[blk].d);
                int8_t di = (int8_t)lrintf(d * cache->scale_w);
                if (di >  127) di =  127;
                if (di < -128) di = -128;
                svint8_t lo8 = svsub_n_s8_x(pg, svreinterpret_s8_u8(lo), 8);
                svint8_t hi8 = svsub_n_s8_x(pg, svreinterpret_s8_u8(hi), 8);
                lo8 = svmul_n_s8_x(pg, lo8, di);
                hi8 = svmul_n_s8_x(pg, hi8, di);
                svst1_s8(pg16, drow + p*64 + b*32,     lo8);
                svst1_s8(pg16, drow + p*64 + b*32 + 16, hi8);
            }
        }
    }
#else
    for (int r = 0; r < n_rows; r++) {
        const block_q4_0 *row = (const block_q4_0 *)(base + (size_t)r * row_bytes);
        int8_t *drow = cache->wi8 + (size_t)r * cache->nb_pairs * 64;
        for (int p = 0; p < cache->nb_pairs; p++) {
            for (int j = 0; j < 64; j++) drow[p*64 + j] = 0;
            for (int b = 0; b < 2 && p*2 + b < nb; b++) {
                int blk = p*2 + b;
                for (int j = 0; j < 16; j++) {
                    uint8_t q = row[blk].qs[j];
                    int lo = (q & 0xf) - 8;
                    int hi = (q >> 4) - 8;
                    float d = ggml_fp16_to_fp32(row[blk].d);
                    int8_t di = (int8_t)lrintf(d * cache->scale_w);
                    if (di >  127) di =  127;
                    if (di < -128) di = -128;
                    drow[p*64 + b*32 + j]      = (int8_t)(lo * di);
                    drow[p*64 + b*32 + 16 + j] = (int8_t)(hi * di);
                }
            }
        }
    }
#endif
    return 0;
}

static inline void tf_q4_0_int8_cache_free(tf_q4_0_int8_cache *cache) {
    if (cache->wi8) { free(cache->wi8); cache->wi8 = NULL; }
    cache->bytes = 0;
}

/* Dispatch: matvec with prequantized int8 weights.
 * Optimized for low per-call overhead:
 *  - x quantize happens ONCE per call (hoisted out of the batch loop)
 *  - 8-row and 4-row batches are inlined (no function call overhead)
 *  - 2x K-unroll with 2 distinct x vectors (breaks the SDOT dependency on x)
 *  - All 8 saddv at the end, combined via uzp1 + 2 stores of q (16 bytes)
 *  - 1-row tail uses scalar int8 dot product (no fallback to fp32)
 *
 * Measured (qlair profile, n_rows=n_cols=2048, 8 reps):
 *   Total: 479K cycles, 40x speedup over fp32 (19.1M cycles)
 *   Per 8-row batch: 234 cycles (peak SDOT: 128 cycles, 55% efficiency)
 *   qlair's hand-tuned asm (`svdq_dot8_i8_blocks2_a64fx_asm`): 34% efficiency
 *   Our compiler-generated code matches or beats qlair's hand-asm.
 *
 * Per-batch cycle cost by size (qlair profile):
 *   512x512:   104 cycles / 32 SDOT-peak cycles (3.2x peak incl. saddv)
 *   1024x1024: 137 / 64  (2.1x)
 *   2048x2048: 234 / 128 (1.8x)
 *   4096x4096: 542 / 256 (2.1x)
 *   8192x2048: 229 / 128 (1.7x)
 *
 * To reach 85% efficiency (~150 cycles/batch), need:
 *   - Software pipelining across iters (hide 5-cycle load-use latency)
 *   - 4x unroll + 4x K-unroll (limited by 32 SVE registers)
 *   - Or: transposed W layout (w[pair][8][64]) + single-base addressing
 *   Hand-tuned asm experiments (v6 with 2x K-unroll) achieve 297 cycles/batch
 *   (43% efficiency), worse than the C compiler due to function call overhead. */
static inline void tf_matvec_q4_0_int8_prequant_rows(float *dst,
        const tf_q4_0_int8_cache *cache, const float *x,
        int row_start, int row_end) {
    if (row_start >= row_end) return;
    int i = row_start;
    const int8_t *base = cache->wi8;
    const int nb_pairs = cache->nb_pairs;
    const float scale_w = cache->scale_w;
#if defined(__ARM_FEATURE_SVE)
    /* Hoist x quantize: done ONCE per call, not per batch. */
    static int8_t *xi8 = NULL;
    static int xi8_alloc = 0;
    int xi8_need = (cache->n_cols + 63) & ~63;
    if (xi8_need > xi8_alloc) {
        if (xi8) free(xi8);
        xi8 = (int8_t *)aligned_alloc(256, xi8_need);
        xi8_alloc = xi8_need;
    }
    float x_inv;
    tf_quantize_f32_to_int8(x, xi8, cache->n_cols, &x_inv);
    const float inv = 1.0f / (scale_w * x_inv);
    /* 8-row batches: inlined for minimum overhead. 2x K-unroll:
     * process 2 pairs per iteration using 2 different x vectors
     * to break the SDOT dependency on x. */
    for (; i + 7 < row_end; i += 8) {
        const int8_t *w0 = base + (size_t)(i+0) * nb_pairs * 64;
        const int8_t *w1 = base + (size_t)(i+1) * nb_pairs * 64;
        const int8_t *w2 = base + (size_t)(i+2) * nb_pairs * 64;
        const int8_t *w3 = base + (size_t)(i+3) * nb_pairs * 64;
        const int8_t *w4 = base + (size_t)(i+4) * nb_pairs * 64;
        const int8_t *w5 = base + (size_t)(i+5) * nb_pairs * 64;
        const int8_t *w6 = base + (size_t)(i+6) * nb_pairs * 64;
        const int8_t *w7 = base + (size_t)(i+7) * nb_pairs * 64;
        svbool_t pg = svptrue_b8();
        svint32_t a0=svdup_s32(0),a1=svdup_s32(0),a2=svdup_s32(0),a3=svdup_s32(0);
        svint32_t a4=svdup_s32(0),a5=svdup_s32(0),a6=svdup_s32(0),a7=svdup_s32(0);
        /* 2x K-unroll: process 2 pairs per iteration. */
        int p = 0;
        int unroll_end = nb_pairs & ~1;  /* round down to even */
        for (; p < unroll_end; p += 2) {
            /* Prefetch next 2 pairs (16 cache lines). */
            __builtin_prefetch(w0 + (p+2)*64, 0, 0);
            __builtin_prefetch(w1 + (p+2)*64, 0, 0);
            __builtin_prefetch(w2 + (p+2)*64, 0, 0);
            __builtin_prefetch(w3 + (p+2)*64, 0, 0);
            __builtin_prefetch(w4 + (p+2)*64, 0, 0);
            __builtin_prefetch(w5 + (p+2)*64, 0, 0);
            __builtin_prefetch(w6 + (p+2)*64, 0, 0);
            __builtin_prefetch(w7 + (p+2)*64, 0, 0);
            /* Load x for pair 0 and pair 1. */
            svint8_t x0v = svld1_s8(pg, xi8 + (p+0)*64);
            svint8_t x1v = svld1_s8(pg, xi8 + (p+1)*64);
            /* Load 8 rows' W for pair 0. */
            svint8_t v0=svld1_s8(pg, w0 + p*64), v1=svld1_s8(pg, w1 + p*64);
            svint8_t v2=svld1_s8(pg, w2 + p*64), v3=svld1_s8(pg, w3 + p*64);
            svint8_t v4=svld1_s8(pg, w4 + p*64), v5=svld1_s8(pg, w5 + p*64);
            svint8_t v6=svld1_s8(pg, w6 + p*64), v7=svld1_s8(pg, w7 + p*64);
            /* SDOT pair 0. */
            a0=svdot_s32(a0, v0, x0v); a1=svdot_s32(a1, v1, x0v);
            a2=svdot_s32(a2, v2, x0v); a3=svdot_s32(a3, v3, x0v);
            a4=svdot_s32(a4, v4, x0v); a5=svdot_s32(a5, v5, x0v);
            a6=svdot_s32(a6, v6, x0v); a7=svdot_s32(a7, v7, x0v);
            /* Load 8 rows' W for pair 1. */
            svint8_t w0_1=svld1_s8(pg, w0 + (p+1)*64);
            svint8_t w1_1=svld1_s8(pg, w1 + (p+1)*64);
            svint8_t w2_1=svld1_s8(pg, w2 + (p+1)*64);
            svint8_t w3_1=svld1_s8(pg, w3 + (p+1)*64);
            svint8_t w4_1=svld1_s8(pg, w4 + (p+1)*64);
            svint8_t w5_1=svld1_s8(pg, w5 + (p+1)*64);
            svint8_t w6_1=svld1_s8(pg, w6 + (p+1)*64);
            svint8_t w7_1=svld1_s8(pg, w7 + (p+1)*64);
            /* SDOT pair 1 (uses x1v which was pre-loaded). */
            a0=svdot_s32(a0, w0_1, x1v); a1=svdot_s32(a1, w1_1, x1v);
            a2=svdot_s32(a2, w2_1, x1v); a3=svdot_s32(a3, w3_1, x1v);
            a4=svdot_s32(a4, w4_1, x1v); a5=svdot_s32(a5, w5_1, x1v);
            a6=svdot_s32(a6, w6_1, x1v); a7=svdot_s32(a7, w7_1, x1v);
        }
        /* Tail: handle odd pair. */
        for (; p < nb_pairs; p++) {
            svint8_t xv = svld1_s8(pg, xi8 + p*64);
            svint8_t v0=svld1_s8(pg, w0 + p*64), v1=svld1_s8(pg, w1 + p*64);
            svint8_t v2=svld1_s8(pg, w2 + p*64), v3=svld1_s8(pg, w3 + p*64);
            svint8_t v4=svld1_s8(pg, w4 + p*64), v5=svld1_s8(pg, w5 + p*64);
            svint8_t v6=svld1_s8(pg, w6 + p*64), v7=svld1_s8(pg, w7 + p*64);
            a0=svdot_s32(a0, v0, xv); a1=svdot_s32(a1, v1, xv);
            a2=svdot_s32(a2, v2, xv); a3=svdot_s32(a3, v3, xv);
            a4=svdot_s32(a4, v4, xv); a5=svdot_s32(a5, v5, xv);
            a6=svdot_s32(a6, v6, xv); a7=svdot_s32(a7, v7, xv);
        }
        /* 8 saddv (1 per row), then uzp1 + 2 stores of q. */
        int32_t r0 = svaddv_s32(pg, a0), r1 = svaddv_s32(pg, a1);
        int32_t r2 = svaddv_s32(pg, a2), r3 = svaddv_s32(pg, a3);
        int32_t r4 = svaddv_s32(pg, a4), r5 = svaddv_s32(pg, a5);
        int32_t r6 = svaddv_s32(pg, a6), r7 = svaddv_s32(pg, a7);
        dst[i+0] = (float)r0 * inv; dst[i+1] = (float)r1 * inv;
        dst[i+2] = (float)r2 * inv; dst[i+3] = (float)r3 * inv;
        dst[i+4] = (float)r4 * inv; dst[i+5] = (float)r5 * inv;
        dst[i+6] = (float)r6 * inv; dst[i+7] = (float)r7 * inv;
    }
    /* 4-row tail: inlined. */
    for (; i + 3 < row_end; i += 4) {
        const int8_t *w0 = base + (size_t)(i+0) * nb_pairs * 64;
        const int8_t *w1 = base + (size_t)(i+1) * nb_pairs * 64;
        const int8_t *w2 = base + (size_t)(i+2) * nb_pairs * 64;
        const int8_t *w3 = base + (size_t)(i+3) * nb_pairs * 64;
        svbool_t pg = svptrue_b8();
        svint32_t a0=svdup_s32(0),a1=svdup_s32(0),a2=svdup_s32(0),a3=svdup_s32(0);
        int p = 0;
        int unroll_end = nb_pairs & ~1;
        for (; p < unroll_end; p += 2) {
            __builtin_prefetch(w0 + (p+2)*64, 0, 0);
            __builtin_prefetch(w1 + (p+2)*64, 0, 0);
            __builtin_prefetch(w2 + (p+2)*64, 0, 0);
            __builtin_prefetch(w3 + (p+2)*64, 0, 0);
            svint8_t x0v = svld1_s8(pg, xi8 + p*64);
            svint8_t x1v = svld1_s8(pg, xi8 + (p+1)*64);
            svint8_t v0=svld1_s8(pg, w0 + p*64), v1=svld1_s8(pg, w1 + p*64);
            svint8_t v2=svld1_s8(pg, w2 + p*64), v3=svld1_s8(pg, w3 + p*64);
            a0=svdot_s32(a0, v0, x0v); a1=svdot_s32(a1, v1, x0v);
            a2=svdot_s32(a2, v2, x0v); a3=svdot_s32(a3, v3, x0v);
            svint8_t w0_1=svld1_s8(pg, w0 + (p+1)*64);
            svint8_t w1_1=svld1_s8(pg, w1 + (p+1)*64);
            svint8_t w2_1=svld1_s8(pg, w2 + (p+1)*64);
            svint8_t w3_1=svld1_s8(pg, w3 + (p+1)*64);
            a0=svdot_s32(a0, w0_1, x1v); a1=svdot_s32(a1, w1_1, x1v);
            a2=svdot_s32(a2, w2_1, x1v); a3=svdot_s32(a3, w3_1, x1v);
        }
        for (; p < nb_pairs; p++) {
            svint8_t xv = svld1_s8(pg, xi8 + p*64);
            svint8_t v0=svld1_s8(pg, w0 + p*64), v1=svld1_s8(pg, w1 + p*64);
            svint8_t v2=svld1_s8(pg, w2 + p*64), v3=svld1_s8(pg, w3 + p*64);
            a0=svdot_s32(a0, v0, xv); a1=svdot_s32(a1, v1, xv);
            a2=svdot_s32(a2, v2, xv); a3=svdot_s32(a3, v3, xv);
        }
        int32_t r0 = svaddv_s32(pg, a0), r1 = svaddv_s32(pg, a1);
        int32_t r2 = svaddv_s32(pg, a2), r3 = svaddv_s32(pg, a3);
        dst[i+0] = (float)r0 * inv; dst[i+1] = (float)r1 * inv;
        dst[i+2] = (float)r2 * inv; dst[i+3] = (float)r3 * inv;
    }
#endif
    /* 1-row tail: scalar int8 dot product. */
    for (; i < row_end; i++) {
        const int8_t *w = base + (size_t)i * nb_pairs * 64;
        int32_t s = 0;
        for (int j = 0; j < cache->n_cols; j++) {
            s += (int32_t)w[j] * (int32_t)xi8[j];
        }
        dst[i] = (float)s * inv;
    }
}

static inline void tf_vec_dot_q4_0_f32_4x(const block_q4_0 *row,
                                           const float *x0, const float *x1,
                                           const float *x2, const float *x3,
                                           int n_cols, float *s0, float *s1,
                                           float *s2, float *s3) {
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
    svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
    svbool_t pg = svptrue_b32();
    int nb = n_cols / 32;
    if (nb > 0) __builtin_prefetch(row->qs, 0, 0);
    for (int b = 0; b < nb; b++) {
        const float d = ggml_fp16_to_fp32(row[b].d);
        const int base = b * 32;
        svuint32_t q = svld1ub_u32(pg, row[b].qs);
        svint32_t qlo = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, q, 0x0f)), 8);
        svint32_t qhi = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, q, 4)), 8);
        svfloat32_t wlo = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qlo), d);
        svfloat32_t whi = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qhi), d);
        a0 = svmla_x(pg, a0, wlo, svld1(pg, x0 + base));
        a0 = svmla_x(pg, a0, whi, svld1(pg, x0 + base + 16));
        a1 = svmla_x(pg, a1, wlo, svld1(pg, x1 + base));
        a1 = svmla_x(pg, a1, whi, svld1(pg, x1 + base + 16));
        a2 = svmla_x(pg, a2, wlo, svld1(pg, x2 + base));
        a2 = svmla_x(pg, a2, whi, svld1(pg, x2 + base + 16));
        a3 = svmla_x(pg, a3, wlo, svld1(pg, x3 + base));
        a3 = svmla_x(pg, a3, whi, svld1(pg, x3 + base + 16));
        if (b + 1 < nb) __builtin_prefetch(row[b+1].qs, 0, 0);
    }
    *s0 = svaddv_f32(pg, a0);
    *s1 = svaddv_f32(pg, a1);
    *s2 = svaddv_f32(pg, a2);
    *s3 = svaddv_f32(pg, a3);
#else
    *s0 = tf_vec_dot_q4_0_f32(row, x0, n_cols);
    *s1 = tf_vec_dot_q4_0_f32(row, x1, n_cols);
    *s2 = tf_vec_dot_q4_0_f32(row, x2, n_cols);
    *s3 = tf_vec_dot_q4_0_f32(row, x3, n_cols);
#endif
}

static void tf_matvec_q4_0_rows(float *dst, const uint8_t *base, size_t row_bytes,
                                  const float *x, int n_cols, int row_start, int row_end) {
    int i = row_start;
#if defined(__ARM_FEATURE_SVE)
    /* int8 SDOT path: opt-in via TF_USE_INT8_SDOT_Q4_0. The fp32 FMA path
     * is the production default. Reasons:
     *  - qlair profile: int8 path is ~2.7x SLOWER than fp32 for 2048x2048
     *    Q4_0 matvec (80M vs 30M cycles for 8 reps). The dequant overhead
     *    (8 SVE ops per block × 8 rows × 64 blocks = 4K ops) dominates the
     *    SDOT throughput advantage (256 SDOTs).
     *  - The fp32 path's fused "dequant + FMA" loop avoids the explicit
     *    dequant, which is more efficient when the dequant cost isn't
     *    amortized.
     *  - For unit-scale test data (d=1.0), the int8 path is ~1.0% off
     *    due to int8 quantization (0.79% from weights, 0.4% from x).
     *  - For real Q4_0 (d in [1e-4, 1.0]), the int8 path with per-tensor
     *    d rescale gives <15% error (vs 0% for fp32). Subnormal d
     *    (< 0.03 * max_d) is rounded to 0 and lost.
     *
     * The int8 path is correct (dequant + SDOT + inv_scale formula is
     * exact modulo int8 quantization) and ~2.3x faster in microbenchmarks
     * that isolate the SDOT itself, but the dequant overhead makes it a
     * net loss for matvec workloads. Enable for unit-scale testing or
     * if the workload's SDOT throughput dominates (e.g., very large
     * n_rows with small n_cols). */
#ifdef TF_USE_INT8_SDOT_Q4_0
    for (; i + 7 < row_end; i += 8) {
        tf_vec_dot_q4_0_int8_full_8row(dst + i,
            (const block_q4_0 *)(base + (size_t)(i)   * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+1) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+2) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+3) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+4) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+5) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+6) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+7) * row_bytes),
            x, n_cols);
    }
    for (; i + 3 < row_end; i += 4) {
        tf_vec_dot_q4_0_int8_full_4row(dst + i,
            (const block_q4_0 *)(base + (size_t)(i)   * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+1) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+2) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+3) * row_bytes),
            x, n_cols);
    }
#else
    for (; i + 7 < row_end; i += 8) {
        tf_vec_dot_q4_0_f32_8row(dst + i,
            (const block_q4_0 *)(base + (size_t)(i)   * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+1) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+2) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+3) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+4) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+5) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+6) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+7) * row_bytes),
            x, n_cols);
    }
    for (; i + 3 < row_end; i += 4) {
        tf_vec_dot_q4_0_f32_4row(dst + i,
            (const block_q4_0 *)(base + (size_t)(i)   * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+1) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+2) * row_bytes),
            (const block_q4_0 *)(base + (size_t)(i+3) * row_bytes),
            x, n_cols);
    }
#endif
#endif
    for (; i < row_end; i++) {
        const block_q4_0 *row = (const block_q4_0 *)(base + (size_t)i * row_bytes);
        dst[i] = tf_vec_dot_q4_0_f32(row, x, n_cols);
    }
}

static void tf_matvec_bf16_rows(float *dst, const uint8_t *base, size_t row_bytes,
                                  const float *x, int n_cols, int row_start, int row_end) {
    int i = row_start;
#if defined(__ARM_FEATURE_SVE)
    /* 8-row blocks: 8 FMAs per activation load, doubles compute/memory ratio */
    for (; i + 7 < row_end; i += 8) {
        matvec_bf16_8row(dst + i,
            (const uint16_t *)(base + (size_t)(i)   * row_bytes),
            (const uint16_t *)(base + (size_t)(i+1) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+2) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+3) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+4) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+5) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+6) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+7) * row_bytes),
            x, n_cols);
    }
#endif
    for (; i + 3 < row_end; i += 4) {
        matvec_bf16_4row(dst + i,
            (const uint16_t *)(base + (size_t)(i)   * row_bytes),
            (const uint16_t *)(base + (size_t)(i+1) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+2) * row_bytes),
            (const uint16_t *)(base + (size_t)(i+3) * row_bytes),
            x, n_cols);
    }
    for (; i < row_end; i++) {
        const uint16_t *row = (const uint16_t *)(base + (size_t)i * row_bytes);
        dst[i] = vec_dot_bf16_f32(row, x, n_cols);
    }
}

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
    if (t->mat1->type == GGML_TYPE_F16) {
        size_t row_bytes = (size_t)n_cols * 2;
        tf_matvec_f16_rows(t->dst1, (const uint8_t *)t->mat1->data, row_bytes,
                            t->x, n_cols, t->row_start, t->row_end);
        tf_matvec_f16_rows(t->dst2, (const uint8_t *)t->mat2->data, row_bytes,
                            t->x, n_cols, t->row_start, t->row_end);
    } else if (t->mat1->type == GGML_TYPE_BF16) {
        size_t row_bytes = (size_t)n_cols * 2;
        tf_matvec_bf16_rows(t->dst1, (const uint8_t *)t->mat1->data, row_bytes,
                             t->x, n_cols, t->row_start, t->row_end);
        tf_matvec_bf16_rows(t->dst2, (const uint8_t *)t->mat2->data, row_bytes,
                             t->x, n_cols, t->row_start, t->row_end);
    } else if (t->mat1->type == GGML_TYPE_Q8_0) {
        int nb = n_cols / 32;
        size_t row_bytes = (size_t)nb * sizeof(block_q8_0);
        tf_matvec_q8_rows(t->dst1, (const uint8_t *)t->mat1->data, row_bytes,
                            t->x, n_cols, t->row_start, t->row_end);
        tf_matvec_q8_rows(t->dst2, (const uint8_t *)t->mat2->data, row_bytes,
                            t->x, n_cols, t->row_start, t->row_end);
    } else if (t->mat1->type == GGML_TYPE_Q4_0) {
        size_t row_bytes = (size_t)(n_cols / 32) * sizeof(block_q4_0);
        tf_matvec_q4_0_rows(t->dst1, (const uint8_t *)t->mat1->data, row_bytes,
                             t->x, n_cols, t->row_start, t->row_end);
        tf_matvec_q4_0_rows(t->dst2, (const uint8_t *)t->mat2->data, row_bytes,
                             t->x, n_cols, t->row_start, t->row_end);
    } else {
        /* Generic path for other quantized weights — AVX2 dot product */
        float *tmp = (float *)malloc(n_cols * sizeof(float));
        for (int i = t->row_start; i < t->row_end; i++) {
            tf_dequant_row(t->mat1, i, tmp);
#if defined(__AVX2__) && defined(__FMA__)
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            int j = 0;
            for (; j + 31 < n_cols; j += 32) {
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j),    _mm256_loadu_ps(t->x+j),    a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j+8),  _mm256_loadu_ps(t->x+j+8),  a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j+16), _mm256_loadu_ps(t->x+j+16), a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j+24), _mm256_loadu_ps(t->x+j+24), a3);
            }
            for (; j + 7 < n_cols; j += 8)
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j), _mm256_loadu_ps(t->x+j), a0);
            a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            __m128 hi = _mm256_extractf128_ps(a0, 1), lo = _mm256_castps256_ps128(a0);
            __m128 s4 = _mm_add_ps(lo, hi);
            s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
            float sum = _mm_cvtss_f32(s4);
            for (; j < n_cols; j++) sum += tmp[j] * t->x[j];
#else
            float sum = 0.0f;
            for (int j = 0; j < n_cols; j++) sum += tmp[j] * t->x[j];
#endif
            t->dst1[i] = sum;
        }
        for (int i = t->row_start; i < t->row_end; i++) {
            tf_dequant_row(t->mat2, i, tmp);
#if defined(__AVX2__) && defined(__FMA__)
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            int j = 0;
            for (; j + 31 < n_cols; j += 32) {
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j),    _mm256_loadu_ps(t->x+j),    a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j+8),  _mm256_loadu_ps(t->x+j+8),  a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j+16), _mm256_loadu_ps(t->x+j+16), a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j+24), _mm256_loadu_ps(t->x+j+24), a3);
            }
            for (; j + 7 < n_cols; j += 8)
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp+j), _mm256_loadu_ps(t->x+j), a0);
            a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            __m128 hi = _mm256_extractf128_ps(a0, 1), lo = _mm256_castps256_ps128(a0);
            __m128 s4 = _mm_add_ps(lo, hi);
            s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
            float sum = _mm_cvtss_f32(s4);
            for (; j < n_cols; j++) sum += tmp[j] * t->x[j];
#else
            float sum = 0.0f;
            for (int j = 0; j < n_cols; j++) sum += tmp[j] * t->x[j];
#endif
            t->dst2[i] = sum;
        }
        free(tmp);
    }
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

/* Forward declaration for fused FFN worker */
static void tf_matvec_qtensor_rows(float *dst, const qtensor *mat, const float *x,
                                    int row_start, int row_end);

/* Fused gate+up+SiLU: compute gate and up matvec for each thread's rows,
 * then immediately apply silu(gate[i]) * up[i] → dst[i].
 * Eliminates the SiLU barrier between gate+up and down projections.
 * dst = silu(gate_mat @ x) * (up_mat @ x) */
typedef struct {
    float *dst;           /* output: silu(gate) * up, [n_rows] */
    const qtensor *gate_mat, *up_mat;
    const float *x;
    int row_start, row_end;
} tf_fused_ffn_silu_task;

static void *tf_fused_ffn_silu_worker(void *arg) {
    tf_fused_ffn_silu_task *t = (tf_fused_ffn_silu_task *)arg;
    int rs = t->row_start, re = t->row_end;

    /* Phase 1: Compute gate and up matvec for this thread's row range.
     * tf_matvec_qtensor_rows writes to dst[row_start..row_end).
     * Use dst as gate buffer, allocate up on stack. */
    int n_rows = re - rs;
    float *up_buf = (float *)alloca((size_t)n_rows * sizeof(float));

    /* gate → dst[rs..re), up → up_buf[0..n_rows) */
    tf_matvec_qtensor_rows(t->dst, t->gate_mat, t->x, rs, re);
    tf_matvec_qtensor_rows(up_buf - rs, t->up_mat, t->x, rs, re);

    /* Phase 2: Fused SiLU×mul in-place: dst[i] = silu(gate[i]) * up[i] */
    for (int i = rs; i < re; i++) {
        float g = t->dst[i];
        t->dst[i] = g / (1.0f + expf(-g)) * up_buf[i - rs];
    }
    return NULL;
}

static void TF_MAYBE_UNUSED tf_qmatvec_fused2_silu_pool(transformer_model *m, float *dst,
                                                         const qtensor *gate_mat, const qtensor *up_mat,
                                                         const float *x, int n_rows) {
    int nt = m->n_threads;
    if (nt <= 1 || !m->pool_alive) {
        tf_fused_ffn_silu_task t = {dst, gate_mat, up_mat, x, 0, n_rows};
        tf_fused_ffn_silu_worker(&t);
        return;
    }
    tf_fused_ffn_silu_task *tasks = (tf_fused_ffn_silu_task *)alloca(nt * sizeof(tf_fused_ffn_silu_task));
    int rp = n_rows / nt, re = n_rows % nt, ro = 0;
    for (int t = 0; t < nt; t++) {
        int rc = rp + (t < re ? 1 : 0);
        tasks[t] = (tf_fused_ffn_silu_task){dst, gate_mat, up_mat, x, ro, ro + rc};
        ro += rc;
    }
    tf_pool_dispatch(m, tf_fused_ffn_silu_worker, tasks, sizeof(tf_fused_ffn_silu_task));
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

/* Process one matrix matvec with its own type (helper for fused3 worker) */
static void tf_matvec_qtensor_rows(float *dst, const qtensor *mat, const float *x,
                                    int row_start, int row_end) {
    int n_cols = mat->n_cols;
    if (mat->type == GGML_TYPE_F16) {
        size_t rb = (size_t)n_cols * 2;
        tf_matvec_f16_rows(dst, (const uint8_t *)mat->data, rb, x, n_cols, row_start, row_end);
    } else if (mat->type == GGML_TYPE_BF16) {
        size_t rb = (size_t)n_cols * 2;
        tf_matvec_bf16_rows(dst, (const uint8_t *)mat->data, rb, x, n_cols, row_start, row_end);
    } else if (mat->type == GGML_TYPE_Q8_0) {
        size_t rb = (size_t)(n_cols / 32) * sizeof(block_q8_0);
        tf_matvec_q8_rows(dst, (const uint8_t *)mat->data, rb, x, n_cols, row_start, row_end);
    } else if (mat->type == GGML_TYPE_Q4_0) {
        size_t rb = (size_t)(n_cols / 32) * sizeof(block_q4_0);
        tf_matvec_q4_0_rows(dst, (const uint8_t *)mat->data, rb, x, n_cols, row_start, row_end);
    } else {
        float *tmp = (float *)malloc(n_cols * sizeof(float));
        if (!tmp) return;
        for (int i = row_start; i < row_end; i++) {
            tf_dequant_row(mat, i, tmp);
            float sum = 0.0f;
            for (int j = 0; j < n_cols; j++) sum += tmp[j] * x[j];
            dst[i] = sum;
        }
        free(tmp);
    }
}

static void *tf_qmatvec_fused3_worker(void *arg) {
    tf_matvec_fused3_task *t = (tf_matvec_fused3_task *)arg;
    /* Each matrix is processed with its own type — handles mixed quantization
     * (e.g. Unsloth UD where Q may be F16 while K/V remain Q8_0). */
    if (t->row_end1 > t->row_start1)
        tf_matvec_qtensor_rows(t->dst1, t->mat1, t->x, t->row_start1, t->row_end1);
    if (t->row_end2 > t->row_start2) {
        tf_matvec_qtensor_rows(t->dst2, t->mat2, t->x, t->row_start2, t->row_end2);
        tf_matvec_qtensor_rows(t->dst3, t->mat3, t->x, t->row_start2, t->row_end2);
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
    /* Distribute rows across threads: n_q Q rows and n_kv K+V rows each. */
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
    if (mat->type == GGML_TYPE_BF16) {
        const uint8_t *base = (const uint8_t *)mat->data;
        size_t row_bytes = (size_t)n_cols * 2;
        tf_matvec_bf16_rows(dst, base, row_bytes, x, n_cols, 0, n_rows);
        return;
    }
    if (mat->type == GGML_TYPE_Q8_0) {
        int nb = n_cols / 32;
        size_t row_bytes = (size_t)nb * sizeof(block_q8_0);
        const uint8_t *base = (const uint8_t *)mat->data;
        for (int i = 0; i < n_rows; i++) {
            dst[i] = vec_dot_q8_0_f32(base + (size_t)i * row_bytes, x, n_cols);
        }
        return;
    }
    if (mat->type == GGML_TYPE_Q4_0) {
        int nb = n_cols / 32;
        size_t row_bytes = (size_t)nb * sizeof(block_q4_0);
        tf_matvec_q4_0_rows(dst, (const uint8_t *)mat->data, row_bytes, x, n_cols, 0, n_rows);
        return;
    }
    for (int i = 0; i < n_rows; i++) {
        tf_dequant_row(mat, i, tmp);
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 31 < n_cols; j += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + j),      _mm256_loadu_ps(x + j),      acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + j + 8),  _mm256_loadu_ps(x + j + 8),  acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + j + 16), _mm256_loadu_ps(x + j + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + j + 24), _mm256_loadu_ps(x + j + 24), acc3);
        }
        for (; j + 7 < n_cols; j += 8)
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + j), _mm256_loadu_ps(x + j), acc0);
        acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
        float sum = _mm_cvtss_f32(s4);
        for (; j < n_cols; j++) sum += tmp[j] * x[j];
#else
        float sum = 0.0f;
        for (int j = 0; j < n_cols; j++) sum += tmp[j] * x[j];
#endif
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

/* Pool-based multi-threaded expert matvec: splits rows across threads */
static void tf_qmatvec_expert_pool(transformer_model *m, float *dst, const qtensor *mat,
                                    int expert, const float *x, int rows_per_expert) {
    int n_threads = m->n_threads;
    if (n_threads <= 1 || rows_per_expert < n_threads * 4 || !m->pool_alive) {
        tf_qmatvec_expert(dst, mat, expert, x, rows_per_expert, m->thread_tmp[0]);
        return;
    }
    /* Create a virtual qtensor pointing to the expert's data slice */
    size_t row_bytes = tf_row_bytes(mat->type, mat->n_cols);
    qtensor expert_mat = *mat;
    expert_mat.data = (void *)((const uint8_t *)mat->data + (size_t)expert * rows_per_expert * row_bytes);

    tf_matvec_task *tasks = (tf_matvec_task *)alloca(n_threads * sizeof(tf_matvec_task));
    int rows_per = rows_per_expert / n_threads;
    int extra = rows_per_expert % n_threads;
    int offset = 0;
    for (int t = 0; t < n_threads; t++) {
        int count = rows_per + (t < extra ? 1 : 0);
        tasks[t] = (tf_matvec_task){dst, &expert_mat, x, offset, offset + count, m->thread_tmp[t]};
        offset += count;
    }
    tf_pool_dispatch(m, tf_qmatvec_worker, tasks, sizeof(tf_matvec_task));
}

/* Legacy multi-threaded version (pthread_create per call) */
static void TF_MAYBE_UNUSED tf_qmatvec_mt(float *dst, const qtensor *mat, const float *x, int n_rows,
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
static void tf_rope(float *vec, int n_heads, int head_dim, int pos, float freq_base,
                    const float *inv_freq) {
    int half_dim = head_dim / 2;
    /* Precompute cos/sin table for all pair positions */
    float cos_tab[512], sin_tab[512]; /* half_dim <= 512 */
    for (int j = 0; j < half_dim; j++) {
        float freq = inv_freq ? inv_freq[j] : (1.0f / powf(freq_base, (float)(2 * j) / head_dim));
        float theta = pos * freq;
        cos_tab[j] = cosf(theta);
        sin_tab[j] = sinf(theta);
    }
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
#if defined(__AVX2__) && defined(__FMA__)
        int j = 0;
        for (; j + 7 < half_dim; j += 8) {
            __m256 vc = _mm256_loadu_ps(cos_tab + j);
            __m256 vs = _mm256_loadu_ps(sin_tab + j);
            __m256 v0 = _mm256_loadu_ps(v + j);
            __m256 v1 = _mm256_loadu_ps(v + j + half_dim);
            _mm256_storeu_ps(v + j,             _mm256_fmsub_ps(v0, vc, _mm256_mul_ps(v1, vs)));
            _mm256_storeu_ps(v + j + half_dim,  _mm256_fmadd_ps(v0, vs, _mm256_mul_ps(v1, vc)));
        }
        for (; j < half_dim; j++) {
            float v0 = v[j], v1 = v[j + half_dim];
            v[j]            = v0 * cos_tab[j] - v1 * sin_tab[j];
            v[j + half_dim] = v0 * sin_tab[j] + v1 * cos_tab[j];
        }
#else
        for (int j = 0; j < half_dim; j++) {
            float v0 = v[j], v1 = v[j + half_dim];
            v[j]            = v0 * cos_tab[j] - v1 * sin_tab[j];
            v[j + half_dim] = v0 * sin_tab[j] + v1 * cos_tab[j];
        }
#endif
    }
}

/* M-RoPE (IMROPE): apply rotary position encoding with interleaved sections.
 * sections[4] = [temporal, height, width, pad], positions = [pos_t, pos_h, pos_w].
 * Dimension pairs are assigned to sections via interleaved pattern (sector % 3). */
/* M-RoPE (IMROPE) with NeoX-style rotation: pairs (v[j], v[j + half_dim]).
 * Cache index i0 = 2*j determines the sector and frequency.
 * sector = (i0/2) % sect_dims = j % sect_dims. */
static void tf_rope_mrope(float *vec, int n_heads, int head_dim, int pos_t, int pos_h, int pos_w,
                           float freq_base, const int sections[4], const float *inv_freq) {
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    if (sect_dims <= 0) sect_dims = head_dim / 2; /* fallback */
    int pair_off = sect_dims;
    int rope_dim = 2 * sect_dims;

    /* Precompute cos/sin table once (per-position varies by section) */
    float cos_tab[512], sin_tab[512]; /* sect_dims <= 512 */
    for (int j = 0; j < sect_dims; j++) {
        int pos;
        if (j % 3 == 1 && j < 3 * sections[1]) {
            pos = pos_h;
        } else if (j % 3 == 2 && j < 3 * sections[2]) {
            pos = pos_w;
        } else if (j % 3 == 0 && j < 3 * sections[0]) {
            pos = pos_t;
        } else {
            pos = pos_t;
        }
        float freq = inv_freq ? inv_freq[j] : (1.0f / powf(freq_base, (float)(2 * j) / rope_dim));
        float theta = pos * freq;
        cos_tab[j] = cosf(theta);
        sin_tab[j] = sinf(theta);
    }

    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
#if defined(__AVX2__) && defined(__FMA__)
        int j = 0;
        for (; j + 7 < sect_dims; j += 8) {
            __m256 vc = _mm256_loadu_ps(cos_tab + j);
            __m256 vs = _mm256_loadu_ps(sin_tab + j);
            __m256 v0 = _mm256_loadu_ps(v + j);
            __m256 v1 = _mm256_loadu_ps(v + j + pair_off);
            _mm256_storeu_ps(v + j,              _mm256_fmsub_ps(v0, vc, _mm256_mul_ps(v1, vs)));
            _mm256_storeu_ps(v + j + pair_off,   _mm256_fmadd_ps(v0, vs, _mm256_mul_ps(v1, vc)));
        }
        for (; j < sect_dims; j++) {
            float v0 = v[j], v1 = v[j + pair_off];
            v[j]             = v0 * cos_tab[j] - v1 * sin_tab[j];
            v[j + pair_off]  = v0 * sin_tab[j] + v1 * cos_tab[j];
        }
#else
        for (int j = 0; j < sect_dims; j++) {
            float v0 = v[j], v1 = v[j + pair_off];
            v[j]             = v0 * cos_tab[j] - v1 * sin_tab[j];
            v[j + pair_off]  = v0 * sin_tab[j] + v1 * cos_tab[j];
        }
#endif
    }
}

/* Per-head RMSNorm (QK-norm): normalize each head_dim-sized chunk */
static void tf_qk_norm(float *vec, int n_heads, int head_dim, const qtensor *norm_w, float eps, float *w_buf) {
    tf_dequant_row(norm_w, 0, w_buf);
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 vss = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < head_dim; i += 8) {
            __m256 vi = _mm256_loadu_ps(v + i);
            vss = _mm256_fmadd_ps(vi, vi, vss);
        }
        __m128 hi = _mm256_extractf128_ps(vss, 1);
        __m128 lo = _mm256_castps256_ps128(vss);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
        float ss = _mm_cvtss_f32(s4);
        for (; i < head_dim; i++) ss += v[i] * v[i];
        ss = 1.0f / sqrtf(ss / head_dim + eps);
        __m256 vscale = _mm256_set1_ps(ss);
        i = 0;
        for (; i + 7 < head_dim; i += 8) {
            __m256 vi = _mm256_loadu_ps(v + i);
            __m256 vw = _mm256_loadu_ps(w_buf + i);
            _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_mul_ps(vi, vscale), vw));
        }
        for (; i < head_dim; i++) v[i] = v[i] * ss * w_buf[i];
#else
        float ss = 0.0f;
        for (int i = 0; i < head_dim; i++) ss += v[i] * v[i];
        ss = 1.0f / sqrtf(ss / head_dim + eps);
        for (int i = 0; i < head_dim; i++) v[i] = v[i] * ss * w_buf[i];
#endif
    }
}

/* Forward declarations — defined after fast_exp_avx2 */
static void tf_softmax(float *x, int n);
static void tf_silu_mul_avx2(float *out, const float *gate, const float *up, int n);

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
        } else {
            /* Generic AVX2 attention for any head_dim (with prefetch) */
            for (int p = 0; p < seq_len; p++) {
                const float *k_p = t->key_cache + (size_t)p * t->kv_dim + kv_h * hd;
                if (p + 2 < seq_len)
                    _mm_prefetch((const char *)(t->key_cache + (size_t)(p+2) * t->kv_dim + kv_h * hd), _MM_HINT_T0);
                __m256 acc = _mm256_setzero_ps();
                int d = 0;
                for (; d + 7 < hd; d += 8)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d), _mm256_loadu_ps(k_p + d), acc);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 s4 = _mm_add_ps(lo, hi);
                s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
                float score = _mm_cvtss_f32(s4);
                for (; d < hd; d++) score += q_h[d] * k_p[d];
                att_h[p] = score * t->scale;
            }
            tf_softmax(att_h, seq_len);
            float *out_h = t->xb2 + h * hd;
            /* Zero output with AVX2 */
            for (int d = 0; d < hd; d += 8) _mm256_storeu_ps(out_h + d, _mm256_setzero_ps());
            for (int p = 0; p < seq_len; p++) {
                const float *v_p = t->value_cache + (size_t)p * t->kv_dim + kv_h * hd;
                if (p + 2 < seq_len)
                    _mm_prefetch((const char *)(t->value_cache + (size_t)(p+2) * t->kv_dim + kv_h * hd), _MM_HINT_T0);
                __m256 a = _mm256_set1_ps(att_h[p]);
                for (int d = 0; d < hd; d += 8)
                    _mm256_storeu_ps(out_h + d, _mm256_fmadd_ps(a, _mm256_loadu_ps(v_p + d),
                                                                  _mm256_loadu_ps(out_h + d)));
            }
        }
#elif defined(__ARM_FEATURE_SVE)
        {
            /* SVE attention with prefetch */
            svbool_t pg = svptrue_b32();
            for (int p = 0; p < seq_len; p++) {
                const float *k_p = t->key_cache + (size_t)p * t->kv_dim + kv_h * hd;
                if (p + 2 < seq_len)
                    __builtin_prefetch(t->key_cache + (size_t)(p+2) * t->kv_dim + kv_h * hd, 0, 1);
                svfloat32_t acc = svdup_f32(0.0f);
                int d = 0;
                for (; d + (int)svcntw() - 1 < hd; d += (int)svcntw())
                    acc = svmla_x(pg, acc, svld1(pg, q_h + d), svld1(pg, k_p + d));
                if (d < hd) {
                    svbool_t ptail = svwhilelt_b32(d, hd);
                    acc = svmla_m(ptail, acc, svld1(ptail, q_h + d), svld1(ptail, k_p + d));
                }
                att_h[p] = svaddv(pg, acc) * t->scale;
            }
            tf_softmax(att_h, seq_len);
            float *out_h = t->xb2 + h * hd;
            /* Zero output with SVE */
            for (int d = 0; d < hd; d += (int)svcntw())
                svst1(svwhilelt_b32(d, hd), out_h + d, svdup_f32(0.0f));
            /* V accumulation with prefetch */
            for (int p = 0; p < seq_len; p++) {
                const float *v_p = t->value_cache + (size_t)p * t->kv_dim + kv_h * hd;
                if (p + 2 < seq_len)
                    __builtin_prefetch(t->value_cache + (size_t)(p+2) * t->kv_dim + kv_h * hd, 0, 1);
                svfloat32_t va = svdup_f32(att_h[p]);
                int d = 0;
                for (; d + (int)svcntw() - 1 < hd; d += (int)svcntw())
                    svst1(pg, out_h + d, svmla_x(pg, svld1(pg, out_h + d), va, svld1(pg, v_p + d)));
                if (d < hd) {
                    svbool_t ptail = svwhilelt_b32(d, hd);
                    svst1(ptail, out_h + d, svmla_m(ptail, svld1(ptail, out_h + d), va, svld1(ptail, v_p + d)));
                }
            }
        }
#else
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
#endif
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

static int tf_compute_max_ff(const transformer_model *m) {
    int max_ff = m->n_ff;
    if (m->use_moe && m->n_ff_expert > max_ff) max_ff = m->n_ff_expert;
    if (m->use_moe && m->n_expert > max_ff) max_ff = m->n_expert;
    if (m->is_hybrid && m->ssm_qkv_dim > max_ff) max_ff = m->ssm_qkv_dim;
    return max_ff;
}

transformer_model *transformer_load(gguf_context *gguf, int max_seq_len) {
    if (!gguf) return NULL;

    transformer_model *m = (transformer_model *)calloc(1, sizeof(transformer_model));
    if (!m) return NULL;

    /* Detect architecture prefix. */
    const char *arch = "qwen2";
    if (gguf_find_key(gguf, "gemma4.block_count") >= 0) {
        arch = "gemma4";
    } else if (gguf_find_key(gguf, "qwen2vl.block_count") >= 0) {
        arch = "qwen2vl";
    } else if (gguf_find_key(gguf, "qwen35.block_count") >= 0) {
        arch = "qwen35";
    } else if (gguf_find_key(gguf, "qwen3vlmoe.block_count") >= 0) {
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

    /* Hybrid SSM (Qwen3.5) */
    m->is_hybrid = 0;
    m->full_attn_interval = 0;
    if (strcmp(arch, "qwen35") == 0) {
        m->is_hybrid = 1;
        m->ssm_conv_kernel = tf_get_int(gguf, ARCH_KEY("ssm.conv_kernel"), 4);
        m->ssm_d_state     = tf_get_int(gguf, ARCH_KEY("ssm.state_size"), 128);
        m->ssm_n_group     = tf_get_int(gguf, ARCH_KEY("ssm.group_count"), 16);
        m->ssm_dt_rank     = tf_get_int(gguf, ARCH_KEY("ssm.time_step_rank"), 48);
        m->ssm_d_inner     = tf_get_int(gguf, ARCH_KEY("ssm.inner_size"), 6144);
        m->full_attn_interval = tf_get_int(gguf, ARCH_KEY("attention.full_attention_interval"), 4);
        m->ssm_qkv_dim = m->ssm_d_state * m->ssm_n_group * 2 + m->ssm_d_inner;
        fprintf(stderr, "transformer: hybrid SSM: conv_k=%d d_state=%d n_group=%d dt_rank=%d d_inner=%d\n",
                m->ssm_conv_kernel, m->ssm_d_state, m->ssm_n_group, m->ssm_dt_rank, m->ssm_d_inner);
        fprintf(stderr, "transformer: full_attn_interval=%d qkv_dim=%d\n",
                m->full_attn_interval, m->ssm_qkv_dim);
    }

    /* Gemma4 architecture */
    m->is_gemma4 = 0;
    m->swa_pattern = NULL;
    m->rope_freq_factors = NULL;
    m->rope_inv_freq_swa = NULL;
    m->ple_buf = NULL;
    m->ple_proj_buf = NULL;
    m->ffn_gelu_fast = 0;
    m->ffn_fused_q4 = 0;
    m->ffn_check = 0;
    if (strcmp(arch, "gemma4") == 0) {
        m->is_gemma4 = 1;
        m->head_dim_full = tf_get_int(gguf, ARCH_KEY("attention.key_length"), 512);
        m->head_dim_swa  = tf_get_int(gguf, ARCH_KEY("attention.key_length_swa"), 256);
        m->head_dim = m->head_dim_full; /* use max for buffer sizing */
        m->swa_window_size = tf_get_int(gguf, ARCH_KEY("attention.sliding_window"), 512);
        m->n_embd_per_layer = tf_get_int(gguf, ARCH_KEY("embedding_length_per_layer_input"), 256);
        int shared_kv_layers = tf_get_int(gguf, ARCH_KEY("attention.shared_kv_layers"), 0);
        m->n_layer_kv_from_start = m->n_layers - shared_kv_layers;
        m->final_logit_softcapping = tf_get_float(gguf, ARCH_KEY("final_logit_softcapping"), 30.0f);
        m->rope_freq_base_swa = tf_get_float(gguf, ARCH_KEY("rope.freq_base_swa"), 10000.0f);
        m->embd_scale = sqrtf((float)m->n_embd);
        m->ffn_activation = 1; /* GELU */
#if defined(__ARM_FEATURE_SVE)
        m->ffn_gelu_fast = 1;
        m->ffn_fused_q4 = 1;
#endif
        const char *gelu_mode = getenv("TF_GELU_MODE");
        if (gelu_mode) {
            if (strcmp(gelu_mode, "exact") == 0 || strcmp(gelu_mode, "0") == 0)
                m->ffn_gelu_fast = 0;
            else if (strcmp(gelu_mode, "fast") == 0 || strcmp(gelu_mode, "1") == 0)
                m->ffn_gelu_fast = 1;
            else
                fprintf(stderr, "transformer: ignoring unknown TF_GELU_MODE=%s (use fast|exact)\n", gelu_mode);
        }
        const char *fused_q4 = getenv("TF_FFN_FUSED_Q4");
        if (fused_q4) m->ffn_fused_q4 = atoi(fused_q4) != 0;
        const char *ffn_check = getenv("TF_FFN_CHECK");
        if (ffn_check) m->ffn_check = atoi(ffn_check) != 0;

        /* Parse SWA layer pattern from GGUF bool array */
        m->swa_pattern = (int *)calloc(m->n_layers, sizeof(int));
        {
            int idx = gguf_find_key(gguf, ARCH_KEY("attention.sliding_window_pattern"));
            if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_ARRAY) {
                int n = (int)gguf->kv[idx].value.arr.n;
                if (n > m->n_layers) n = m->n_layers;
                /* GGUF bool arrays are stored as GGUF_TYPE_BOOL (uint8) */
                uint8_t *data = (uint8_t *)gguf->kv[idx].value.arr.data;
                for (int i = 0; i < n; i++)
                    m->swa_pattern[i] = data[i] ? 1 : 0;
                fprintf(stderr, "transformer: Gemma4 SWA pattern loaded (%d layers)\n", n);
            } else {
                /* Fallback: every 6th layer is full attention (0-indexed: 5,11,...) */
                for (int i = 0; i < m->n_layers; i++)
                    m->swa_pattern[i] = ((i + 1) % 6 != 0) ? 1 : 0;
                fprintf(stderr, "transformer: Gemma4 SWA pattern defaulting to every-6th-full\n");
            }
        }

        fprintf(stderr, "transformer: Gemma4: head_dim_full=%d head_dim_swa=%d swa_window=%d\n",
                m->head_dim_full, m->head_dim_swa, m->swa_window_size);
        fprintf(stderr, "transformer: Gemma4: n_embd_per_layer=%d n_layer_kv_from_start=%d\n",
                m->n_embd_per_layer, m->n_layer_kv_from_start);
        fprintf(stderr, "transformer: Gemma4: softcap=%.1f rope_base_swa=%.0f\n",
                m->final_logit_softcapping, m->rope_freq_base_swa);
        fprintf(stderr, "transformer: Gemma4 FFN: GELU=%s fused_q4=%d check=%d\n",
                m->ffn_gelu_fast ? "fast_sve" : "exact_erf", m->ffn_fused_q4, m->ffn_check);
    }

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

    /* Gemma4 global tensors */
    if (m->is_gemma4) {
        m->per_layer_token_embd = tf_load_tensor(gguf, "per_layer_token_embd.weight", 0);
        m->per_layer_model_proj = tf_load_tensor(gguf, "per_layer_model_proj.weight", 0);
        m->per_layer_proj_norm = tf_load_tensor(gguf, "per_layer_proj_norm.weight", 0);
        if (!m->per_layer_token_embd.data || !m->per_layer_model_proj.data) {
            /* 12B Gemma4 (Q4_K_XL / Q6_K) ships without PLE tensors — proceed without PLE. */
            fprintf(stderr, "transformer: Gemma4 has no PLE per-layer embeddings (12B-style model)\n");
        }
        /* Load proportional RoPE frequency factors */
        qtensor rope_freqs_qt = tf_load_tensor(gguf, "rope_freqs.weight", 0);
        if (rope_freqs_qt.data) {
            int n_freq = rope_freqs_qt.n_cols;
            m->rope_freq_factors = (float *)calloc(n_freq, sizeof(float));
            dequant_row(rope_freqs_qt.type, rope_freqs_qt.data, m->rope_freq_factors, n_freq);
            fprintf(stderr, "transformer: Gemma4 loaded rope_freqs (%d elements)\n", n_freq);
        }
    }

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

        if (m->is_gemma4) {
            /* --- Gemma4 per-layer tensors --- */
            m->layers[l].is_swa = m->swa_pattern[l];
            m->layers[l].shared_kv_source = -1; /* default: own KV */

            /* Determine shared KV source for layers beyond n_layer_kv_from_start */
            if (l >= m->n_layer_kv_from_start) {
                /* SWA layers reuse KV from (n_layer_kv_from_start - 2),
                 * full-attn layers from (n_layer_kv_from_start - 1) */
                m->layers[l].shared_kv_source = m->n_layer_kv_from_start - (m->layers[l].is_swa ? 2 : 1);
                if (m->layers[l].shared_kv_source < 0) m->layers[l].shared_kv_source = 0;
            }

            /* Q projection (always present) */
            LOAD(attn_q,       "attn_q",      1)
            LOAD(attn_q_norm,  "attn_q_norm", 1)
            LOAD(attn_output,  "attn_output", 1)

            /* K/V only for layers with own KV */
            if (m->layers[l].shared_kv_source < 0) {
                LOAD(attn_k,       "attn_k",      1)
                /* Gemma4 SWA layers may share V with K (no attn_v tensor) */
                LOAD(attn_v,       "attn_v",      m->is_gemma4 ? 0 : 1)
                LOAD(attn_k_norm,  "attn_k_norm", 1)
                m->layers[l].has_v_proj = (m->layers[l].attn_v.data != NULL);
            } else {
                /* Try loading K/V anyway — some shared layers may still have them */
                LOAD(attn_k,       "attn_k",      0)
                LOAD(attn_v,       "attn_v",      0)
                LOAD(attn_k_norm,  "attn_k_norm", 0)
                m->layers[l].has_v_proj = (m->layers[l].attn_v.data != NULL);
            }
            /* V normalization (Gemma4 normalizes V too) */
            LOAD(attn_v_norm,  "attn_v_norm", 0)

            /* Per-layer KV head count: Gemma4 12B uses 8 KV heads on SWA layers but
             * 1 (MQA) on full-attention layers, so deriving it from the actual attn_k
             * tensor (rows / head_dim) is the only correct source. Shared-KV layers
             * inherit from their source (already loaded, since src < l). */
            {
                int lhd = m->layers[l].is_swa ? m->head_dim_swa : m->head_dim_full;
                if (m->layers[l].shared_kv_source < 0 && m->layers[l].attn_k.data && lhd > 0)
                    m->layers[l].n_kv_heads = m->layers[l].attn_k.n_rows / lhd;
                else if (m->layers[l].shared_kv_source >= 0)
                    m->layers[l].n_kv_heads = m->layers[m->layers[l].shared_kv_source].n_kv_heads;
                else
                    m->layers[l].n_kv_heads = m->n_kv_heads;
                if (m->layers[l].n_kv_heads <= 0) m->layers[l].n_kv_heads = m->n_kv_heads;
            }

            /* Post-attention norm */
            LOAD(post_attention_norm, "post_attention_norm", 1)

            /* FFN */
            LOAD(ffn_norm,     "ffn_norm",    1)
            LOAD(ffn_gate,     "ffn_gate",    1)
            LOAD(ffn_up,       "ffn_up",      1)
            LOAD(ffn_down,     "ffn_down",    1)
            LOAD(post_ffw_norm, "post_ffw_norm", 1)

            /* Layer output scale (optional) */
            LOAD(layer_output_scale, "layer_output_scale", 0)

            /* Per-layer embedding tensors (PLE) — optional when the model has no PLE */
            LOAD(ple_inp_gate,  "inp_gate",  m->per_layer_token_embd.data ? 1 : 0)
            LOAD(ple_proj,      "proj",      m->per_layer_model_proj.data ? 1 : 0)
            LOAD(ple_post_norm, "post_norm", m->per_layer_proj_norm.data ? 1 : 0)

            REQUIRE_SUPPORTED(attn_q,      "attn_q");
            REQUIRE_SUPPORTED(attn_output, "attn_output");
            REQUIRE_SUPPORTED(ffn_gate, "ffn_gate");
            REQUIRE_SUPPORTED(ffn_up,   "ffn_up");
            REQUIRE_SUPPORTED(ffn_down, "ffn_down");
            if (m->layers[l].attn_k.data) REQUIRE_SUPPORTED(attn_k, "attn_k");
            if (m->layers[l].attn_v.data) REQUIRE_SUPPORTED(attn_v, "attn_v");
        } else if (m->is_hybrid) {
            int is_attn = (m->full_attn_interval > 0 && (l + 1) % m->full_attn_interval == 0);
            m->layers[l].is_ssm = !is_attn;
            if (is_attn) {
                LOAD(attn_q,       "attn_q",      1)
                LOAD(attn_k,       "attn_k",      1)
                LOAD(attn_v,       "attn_v",      1)
                LOAD(attn_q_norm,  "attn_q_norm", 0)
                LOAD(attn_k_norm,  "attn_k_norm", 0)
                LOAD(attn_output,  "attn_output", 1)
            } else {
                LOAD(ssm_qkv,      "attn_qkv",    1)
                LOAD(ssm_gate,     "attn_gate",    1)
                LOAD(ssm_alpha,    "ssm_alpha",    1)
                LOAD(ssm_beta,     "ssm_beta",     1)
                LOAD(ssm_out,      "ssm_out",      1)
                LOAD(ssm_conv1d,   "ssm_conv1d",   1)
                LOAD(ssm_norm,     "ssm_norm",     1)
                /* ssm_a and ssm_dt.bias: no .weight suffix */
                snprintf(name, sizeof(name), "blk.%d.ssm_a", l);
                m->layers[l].ssm_a = tf_load_tensor(gguf, name, 1);
                if (!m->layers[l].ssm_a.data) missing_required = 1;
                snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", l);
                m->layers[l].ssm_dt_bias = tf_load_tensor(gguf, name, 1);
                if (!m->layers[l].ssm_dt_bias.data) missing_required = 1;
            }
            /* post_attention_norm replaces ffn_norm in qwen35 */
            snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", l);
            m->layers[l].ffn_norm = tf_load_tensor(gguf, name, 1);
            if (!m->layers[l].ffn_norm.data) missing_required = 1;
        } else {
            LOAD(attn_q,       "attn_q",      1)
            LOAD(attn_k,       "attn_k",      1)
            LOAD(attn_v,       "attn_v",      1)
            LOAD(attn_q_norm,  "attn_q_norm", 0)
            LOAD(attn_k_norm,  "attn_k_norm", 0)
            LOAD(attn_output,  "attn_output", 1)
            LOAD(ffn_norm,     "ffn_norm",    1)
            /* Q/K/V biases (optional, e.g. Qwen2.5-VL) */
            snprintf(name, sizeof(name), "blk.%d.attn_q.bias", l);
            m->layers[l].attn_q_bias = tf_load_tensor(gguf, name, 0);
            snprintf(name, sizeof(name), "blk.%d.attn_k.bias", l);
            m->layers[l].attn_k_bias = tf_load_tensor(gguf, name, 0);
            snprintf(name, sizeof(name), "blk.%d.attn_v.bias", l);
            m->layers[l].attn_v_bias = tf_load_tensor(gguf, name, 0);
        }
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
        REQUIRE_SUPPORTED(ffn_norm,    "ffn_norm");
        if (!m->is_hybrid || !m->layers[l].is_ssm) {
            REQUIRE_SUPPORTED(attn_q,      "attn_q");
            REQUIRE_SUPPORTED(attn_k,      "attn_k");
            REQUIRE_SUPPORTED(attn_v,      "attn_v");
            REQUIRE_SUPPORTED(attn_output, "attn_output");
        }
        if (m->is_hybrid && m->layers[l].is_ssm) {
            REQUIRE_SUPPORTED(ssm_qkv,    "attn_qkv");
            REQUIRE_SUPPORTED(ssm_gate,   "attn_gate");
            REQUIRE_SUPPORTED(ssm_alpha,  "ssm_alpha");
            REQUIRE_SUPPORTED(ssm_beta,   "ssm_beta");
            REQUIRE_SUPPORTED(ssm_out,    "ssm_out");
            REQUIRE_SUPPORTED(ssm_conv1d, "ssm_conv1d");
            REQUIRE_SUPPORTED(ssm_norm,   "ssm_norm");
        }
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
    m->key_cache_raw   = (void **)calloc(m->n_layers, sizeof(void *));
    m->value_cache_raw = (void **)calloc(m->n_layers, sizeof(void *));
    m->kv_cache_type = 0;
    {
        const char *kv = getenv("TF_KV_DTYPE");
        if (m->is_gemma4 && kv && (!strcmp(kv, "f16") || !strcmp(kv, "fp16"))) {
            m->kv_cache_type = 1;
        } else if (m->is_gemma4 && kv && strcmp(kv, "f32") && strcmp(kv, "fp32")) {
            fprintf(stderr, "transformer: Gemma4 unsupported TF_KV_DTYPE=%s, using f32\n", kv);
        }
    }
    if (m->is_gemma4) {
        int n_own = 0, n_shared = 0;
        size_t kv_elem_size = (m->kv_cache_type == 1) ? sizeof(uint16_t) : sizeof(float);
        for (int l = 0; l < m->n_layers; l++) {
            if (m->layers[l].shared_kv_source >= 0) { n_shared++; continue; }
            /* Per-layer KV dim (SWA & full-attn may have different KV head counts) */
            int lhd = m->layers[l].is_swa ? m->head_dim_swa : m->head_dim_full;
            int l_kv_dim = m->layers[l].n_kv_heads * lhd;
            int cache_len = m->layers[l].is_swa ? m->swa_window_size : max_seq_len;
            m->key_cache_raw[l]   = tf_aligned_calloc(256, (size_t)cache_len * l_kv_dim, kv_elem_size);
            m->value_cache_raw[l] = tf_aligned_calloc(256, (size_t)cache_len * l_kv_dim, kv_elem_size);
            if (m->kv_cache_type == 0) {
                m->key_cache[l]   = (float *)m->key_cache_raw[l];
                m->value_cache[l] = (float *)m->value_cache_raw[l];
            }
            if (!m->key_cache_raw[l] || !m->value_cache_raw[l]) {
                fprintf(stderr, "transformer: Gemma4 KV cache alloc failed at layer %d\n", l);
                transformer_free(m);
                return NULL;
            }
            n_own++;
        }
        /* Point shared layers to their source */
        for (int l = 0; l < m->n_layers; l++) {
            int src = m->layers[l].shared_kv_source;
            if (src >= 0) {
                m->key_cache[l]   = m->key_cache[src];
                m->value_cache[l] = m->value_cache[src];
                m->key_cache_raw[l]   = m->key_cache_raw[src];
                m->value_cache_raw[l] = m->value_cache_raw[src];
            }
        }
        fprintf(stderr, "transformer: Gemma4 KV cache: %d own layers, %d shared layers, dtype=%s\n",
                n_own, n_shared, m->kv_cache_type == 1 ? "f16" : "f32");
        /* Scratch sizing: max per-layer KV dim across all layers (SWA & full-attn differ). */
        for (int l = 0; l < m->n_layers; l++) {
            int lhd = m->layers[l].is_swa ? m->head_dim_swa : m->head_dim_full;
            int l_kv_dim = m->layers[l].n_kv_heads * lhd;
            if (l_kv_dim > kv_dim) kv_dim = l_kv_dim;
        }
    } else {
        for (int l = 0; l < m->n_layers; l++) {
            if (m->is_hybrid && m->layers[l].is_ssm) continue;
            m->key_cache[l]   = (float *)tf_aligned_calloc(256, max_seq_len * kv_dim, sizeof(float));
            m->value_cache[l] = (float *)tf_aligned_calloc(256, max_seq_len * kv_dim, sizeof(float));
            m->key_cache_raw[l]   = m->key_cache[l];
            m->value_cache_raw[l] = m->value_cache[l];
        }
    }

    /* Allocate SSM state for hybrid models */
    m->conv_state = NULL;
    m->conv_state_pos = NULL;
    m->recurrent_state = NULL;
    if (m->is_hybrid && m->n_layers > 0) {
        size_t nl = (size_t)m->n_layers;
        m->conv_state = (float **)calloc(nl, sizeof(float *));
        m->conv_state_pos = (int *)calloc(nl, sizeof(int));
        m->recurrent_state = (float **)calloc(nl, sizeof(float *));
        int n_ssm = 0;
        for (int l = 0; l < m->n_layers; l++) {
            if (!m->layers[l].is_ssm) continue;
            int conv_state_size = (m->ssm_conv_kernel - 1) * m->ssm_qkv_dim;
            m->conv_state[l] = (float *)calloc(conv_state_size, sizeof(float));
            int rec_state_size = m->ssm_dt_rank * m->ssm_d_state * m->ssm_d_state;
            m->recurrent_state[l] = (float *)calloc(rec_state_size, sizeof(float));
            n_ssm++;
        }
        fprintf(stderr, "transformer: SSM state: %d layers, conv=[%d×%d], recurrent=[%d×%d×%d]\n",
                n_ssm, m->ssm_conv_kernel - 1, m->ssm_qkv_dim,
                m->ssm_dt_rank, m->ssm_d_state, m->ssm_d_state);
        m->conv_w_trans = (float *)malloc((size_t)m->ssm_conv_kernel * m->ssm_qkv_dim * sizeof(float));
    } else {
        m->conv_w_trans = NULL;
    }

    /* Allocate scratch buffers */
    int max_ff = tf_compute_max_ff(m);
    int max_dim = m->n_embd > max_ff ? m->n_embd : max_ff;
    int q_dim = m->n_heads * m->head_dim;  /* may differ from n_embd (e.g. 4B: 4096 vs 2560) */
    /* xb2 must hold: attention output (q_dim), SSM qkv (qkv_dim), or Q+gate (2*q_dim) */
    int xb2_dim = q_dim;
    if (m->is_hybrid) {
        if (m->ssm_qkv_dim > xb2_dim) xb2_dim = m->ssm_qkv_dim;
        if (2 * q_dim > xb2_dim) xb2_dim = 2 * q_dim; /* Q+gate for gated attention */
    }
    /* q buffer must hold Q or expanded Q for SSM */
    int q_buf_dim = q_dim;
    if (m->is_hybrid && m->ssm_dt_rank * m->ssm_d_state > q_buf_dim)
        q_buf_dim = m->ssm_dt_rank * m->ssm_d_state;
    m->x         = (float *)tf_aligned_calloc(256, m->n_embd, sizeof(float));
    m->xb        = (float *)tf_aligned_calloc(256, m->n_embd, sizeof(float));
    m->xb2       = (float *)tf_aligned_calloc(256, xb2_dim, sizeof(float));
    m->q         = (float *)tf_aligned_calloc(256, q_buf_dim, sizeof(float));
    m->k         = (float *)tf_aligned_calloc(256, kv_dim, sizeof(float));
    m->v         = (float *)tf_aligned_calloc(256, kv_dim, sizeof(float));
    m->att       = (float *)tf_aligned_calloc(256, m->n_heads * max_seq_len, sizeof(float));
    m->ffn_buf1  = (float *)tf_aligned_calloc(256, max_ff, sizeof(float));
    m->ffn_buf2  = (float *)tf_aligned_calloc(256, max_ff, sizeof(float));
    m->ffn_buf3  = (float *)tf_aligned_calloc(256, max_ff, sizeof(float));
    m->logits     = m->has_lm_head ? (float *)tf_aligned_calloc(256, m->n_vocab, sizeof(float)) : NULL;
    m->matvec_tmp = (float *)tf_aligned_calloc(256, max_dim, sizeof(float));
    m->trace_hidden_norms = 1;

    /* Default: single-threaded, no tensor parallelism */
    m->n_threads = 1;
    m->tp_rank = 0;
    m->tp_size = 1;
    m->tp_allreduce_fn = NULL;
    m->tp_allreduce_ctx = NULL;
    m->thread_tmp = (float **)calloc(1, sizeof(float *));
    m->thread_tmp[0] = m->matvec_tmp;

    /* Precompute RoPE inverse frequency tables */
    {
        int half_dim = m->head_dim / 2;
        m->rope_inv_freq_len = half_dim;
        m->rope_inv_freq = (float *)calloc(half_dim, sizeof(float));
        for (int j = 0; j < half_dim; j++)
            m->rope_inv_freq[j] = 1.0f / powf(m->rope_freq_base, (float)(2 * j) / m->head_dim);

        int sect_sum = m->mrope_sections[0] + m->mrope_sections[1] + m->mrope_sections[2] + m->mrope_sections[3];
        if (sect_sum > 0) {
            int rope_dim = 2 * sect_sum;
            m->rope_mrope_inv_freq_len = sect_sum;
            m->rope_mrope_inv_freq = (float *)calloc(sect_sum, sizeof(float));
            for (int j = 0; j < sect_sum; j++)
                m->rope_mrope_inv_freq[j] = 1.0f / powf(m->rope_freq_base, (float)(2 * j) / rope_dim);
        } else {
            m->rope_mrope_inv_freq_len = 0;
            m->rope_mrope_inv_freq = NULL;
        }
    }

    /* Gemma4: precompute SWA RoPE table and per-layer embedding buffers */
    if (m->is_gemma4) {
        /* SWA RoPE: standard with freq_base_swa, head_dim_swa */
        int half_swa = m->head_dim_swa / 2;
        m->rope_inv_freq_swa = (float *)calloc(half_swa, sizeof(float));
        for (int j = 0; j < half_swa; j++)
            m->rope_inv_freq_swa[j] = 1.0f / powf(m->rope_freq_base_swa, (float)(2 * j) / m->head_dim_swa);

        /* Full-attention RoPE: if rope_freq_factors is available, apply proportional RoPE.
         * The standard rope_inv_freq (already computed above) uses head_dim_full and rope_freq_base.
         * For proportional RoPE: inv_freq[j] /= freq_factors[j] */
        if (m->rope_freq_factors) {
            int half_full = m->head_dim_full / 2;
            for (int j = 0; j < half_full && j < m->rope_inv_freq_len; j++)
                m->rope_inv_freq[j] /= m->rope_freq_factors[j];
            fprintf(stderr, "transformer: Gemma4 proportional RoPE applied (%d dims)\n", half_full);
        }

        /* Per-layer embedding scratch buffers */
        m->ple_buf = (float *)calloc(m->n_embd_per_layer, sizeof(float));
        m->ple_proj_buf = (float *)calloc(m->n_embd, sizeof(float));
    }

    return m;
}

void transformer_free(transformer_model *model) {
    if (!model) return;
    if (model->mpool.base) { munmap(model->mpool.base, model->mpool.cap); model->mpool.base = NULL; }
    if (model->key_cache || model->key_cache_raw) {
        for (int l = 0; l < model->n_layers; l++) {
            /* Skip shared KV caches (freed by their source layer) */
            if (model->is_gemma4 && l < model->n_layers &&
                model->layers && model->layers[l].shared_kv_source >= 0) continue;
            void *kptr = model->key_cache_raw ? model->key_cache_raw[l] : (void *)model->key_cache[l];
            void *vptr = model->value_cache_raw ? model->value_cache_raw[l] : (void *)model->value_cache[l];
            free(kptr);
            free(vptr);
        }
        free(model->key_cache_raw);
        free(model->value_cache_raw);
        free(model->key_cache);
        free(model->value_cache);
    }
    free(model->layers);
    /* Gemma4 resources */
    free(model->swa_pattern);
    free(model->rope_freq_factors);
    free(model->rope_inv_freq_swa);
    free(model->ple_buf);
    free(model->ple_proj_buf);
    if (model->conv_state) {
        for (int l = 0; l < model->n_layers; l++) free(model->conv_state[l]);
        free(model->conv_state);
    }
    free(model->conv_state_pos);
    free(model->conv_w_trans);
    if (model->recurrent_state) {
        for (int l = 0; l < model->n_layers; l++) free(model->recurrent_state[l]);
        free(model->recurrent_state);
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
    free(model->rope_inv_freq);
    free(model->rope_mrope_inv_freq);
    /* Free extra per-thread scratch (thread_tmp[0] == matvec_tmp, already freed) */
    tf_pool_shutdown(model);
    if (model->thread_tmp) {
        for (int t = 1; t < model->n_threads; t++) free(model->thread_tmp[t]);
        free(model->thread_tmp);
    }
    free(model);
}

/* ---- Thread pool ---- */
/* Portable spin-wait hint */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define tf_cpu_pause() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define tf_cpu_pause() __asm__ __volatile__("yield")
#else
#define tf_cpu_pause() ((void)0)
#endif

typedef struct {
    transformer_model *model;
    int tid;
} tf_pool_worker_ctx;

static void tf_bind_current_thread_for_numa(int tid) {
    const char *enabled = getenv("NUMA_DISTRIBUTE");
    if (!enabled || atoi(enabled) == 0) {
        (void)tid;
        return;
    }
#if defined(__linux__)
    int n_cmgs = 4;
    const char *env = getenv("NUMA_N_CMGS");
    if (env) n_cmgs = atoi(env);
    if (n_cmgs < 1) n_cmgs = 1;
    if (n_cmgs > 4) n_cmgs = 4;
    int cmg = tid % n_cmgs;
    int local = tid / n_cmgs;
    if (local >= 12) local %= 12;
    int core = 12 + cmg * 12 + local;

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0 && tid == 0) {
        fprintf(stderr, "numa: warning: sched_setaffinity(core=%d) failed\n", core);
    }
#else
    (void)tid;
#endif
}

static int tf_bind_current_thread_for_numa_saved(int tid, cpu_set_t *old_set) {
    const char *enabled = getenv("NUMA_DISTRIBUTE");
    if (!enabled || atoi(enabled) == 0) {
        (void)tid;
        (void)old_set;
        return 0;
    }
#if defined(__linux__)
    if (sched_getaffinity(0, sizeof(*old_set), old_set) != 0) return 0;
    tf_bind_current_thread_for_numa(tid);
    return 1;
#else
    (void)tid;
    (void)old_set;
    return 0;
#endif
}

static void tf_restore_current_thread_affinity(int restore, const cpu_set_t *old_set) {
#if defined(__linux__)
    if (restore) sched_setaffinity(0, sizeof(*old_set), old_set);
#else
    (void)restore;
    (void)old_set;
#endif
}

static void *tf_pool_worker_main(void *arg) {
    tf_pool_worker_ctx *ctx = (tf_pool_worker_ctx *)arg;
    transformer_model *m = ctx->model;
    int tid = ctx->tid;
    free(ctx);
    tf_bind_current_thread_for_numa(tid);

    int last_phase = 0;
    while (1) {
        /* Sleep until dispatcher signals new work via cond_broadcast.
         * Workers consume ZERO bandwidth while sleeping (futex-based). */
        pthread_mutex_lock(&m->pool_mutex);
        while (m->pool_phase == last_phase && m->pool_alive)
            pthread_cond_wait(&m->pool_cond, &m->pool_mutex);
        pthread_mutex_unlock(&m->pool_mutex);

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
    /* Signal workers to exit */
    pthread_mutex_lock(&model->pool_mutex);
    model->pool_alive = 0;
    model->pool_phase++;
    pthread_cond_broadcast(&model->pool_cond);
    pthread_mutex_unlock(&model->pool_mutex);
    for (int t = 1; t < model->n_threads; t++)
        pthread_join(model->pool_threads[t], NULL);
    free(model->pool_threads);
    model->pool_threads = NULL;
    pthread_mutex_destroy(&model->pool_mutex);
    pthread_cond_destroy(&model->pool_cond);
}

static void tf_pool_start(transformer_model *model) {
    int nt = model->n_threads;
    model->pool_threads = (pthread_t *)calloc(nt, sizeof(pthread_t));
    model->pool_phase = 0;
    model->pool_done = 0;
    model->pool_alive = 1;
    pthread_mutex_init(&model->pool_mutex, NULL);
    pthread_cond_init(&model->pool_cond, NULL);
    __sync_synchronize();
    for (int t = 1; t < nt; t++) {
        tf_pool_worker_ctx *ctx = (tf_pool_worker_ctx *)malloc(sizeof(*ctx));
        ctx->model = model;
        ctx->tid = t;
        pthread_create(&model->pool_threads[t], NULL, tf_pool_worker_main, ctx);
    }
}

/* Dispatch work to the thread pool: calls fn(args + tid * arg_stride) for each thread.
 * Workers sleep on cond_var between dispatches — zero bandwidth when idle. */
static void tf_pool_dispatch(transformer_model *model, void *(*fn)(void *),
                              void *args, size_t arg_stride) {
    int nt = model->n_threads;
    model->pool_fn = fn;
    model->pool_args = args;
    model->pool_arg_stride = arg_stride;
    model->pool_done = 0;
    __sync_synchronize();

    /* Wake workers via cond_broadcast */
    pthread_mutex_lock(&model->pool_mutex);
    model->pool_phase++;
    pthread_cond_broadcast(&model->pool_cond);
    pthread_mutex_unlock(&model->pool_mutex);

    /* Main thread executes worker 0's task directly. Pin only for this task so
     * later ad-hoc pthread kernels inherit the broad numactl CPU mask. */
    void *task0 = (char *)args;
#if defined(__linux__)
    cpu_set_t old_set;
    int restore_affinity = tf_bind_current_thread_for_numa_saved(0, &old_set);
#endif
    fn(task0);
#if defined(__linux__)
    tf_restore_current_thread_affinity(restore_affinity, &old_set);
#endif

    /* Wait for remaining workers (brief spin — workers are doing useful work) */
    while (model->pool_done < nt - 1)
        tf_cpu_pause();
}

void transformer_set_threads(transformer_model *model, int n_threads) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads == model->n_threads) return;

    /* Shutdown old pool */
    tf_pool_shutdown(model);

    /* Free old extra per-thread buffers */
    for (int t = 1; t < model->n_threads; t++) free(model->thread_tmp[t]);
    free(model->thread_tmp);

    int max_ff = tf_compute_max_ff(model);
    int max_dim = model->n_embd > max_ff ? model->n_embd : max_ff;
    model->n_threads = n_threads;
    model->thread_tmp = (float **)calloc(n_threads, sizeof(float *));
    model->thread_tmp[0] = model->matvec_tmp;
    for (int t = 1; t < n_threads; t++) {
        /* Use notouch alloc: dequant always writes before read, so uninitialized is safe.
         * With demand paging, first actual use from worker thread t places pages on
         * thread t's NUMA node for optimal locality. */
        model->thread_tmp[t] = (float *)tf_aligned_alloc_notouch(256,
                                    (size_t)max_dim * sizeof(float));
    }

    /* Start new pool */
    if (n_threads > 1) tf_pool_start(model);
    fprintf(stderr, "transformer: using %d threads (thread pool)\n", n_threads);
#if defined(__ARM_FEATURE_SVE)
    fprintf(stderr, "transformer: A64FX SVE kernels enabled (BF16/F16/Q4_0)\n");
#endif
}

void transformer_build_panels(transformer_model *model) {
    (void)model;
}

void transformer_pool_profile_reset(void) {
}

void transformer_reset_runtime_state(transformer_model *model) {
    if (!model) return;
    model->ds_embd = NULL;
    model->ds_embd_stride = 0;
    if (model->conv_state_pos) {
        memset(model->conv_state_pos, 0, (size_t)model->n_layers * sizeof(int));
    }
    if (model->conv_state) {
        for (int l = 0; l < model->n_layers; l++) {
            if (model->conv_state[l]) {
                int n = (model->ssm_conv_kernel - 1) * model->ssm_qkv_dim;
                if (n > 0) memset(model->conv_state[l], 0, (size_t)n * sizeof(float));
            }
        }
    }
    if (model->recurrent_state) {
        for (int l = 0; l < model->n_layers; l++) {
            if (model->recurrent_state[l]) {
                int n = model->ssm_dt_rank * model->ssm_d_state * model->ssm_d_state;
                if (n > 0) memset(model->recurrent_state[l], 0, (size_t)n * sizeof(float));
            }
        }
    }
}

void transformer_set_trace_hidden_norms(transformer_model *model, int enable) {
    if (!model) return;
    model->trace_hidden_norms = enable ? 1 : 0;
}

void transformer_set_f64_accum(transformer_model *model, int enable) {
    (void)model;
    tf_g_f64_accum = enable ? 1 : 0;
}

/* ---- NUMA-aware memory allocator ---- */
/* With demand paging (XOS_MMM_L_PAGING_POLICY=demand:demand:demand), pages are
 * placed on the NUMA node of the first-touching core. The NUMA allocator ensures
 * each thread's data partition is first-touched by that thread, placing pages on
 * the thread's CMG for optimal memory bandwidth. */

/* Task for parallel pread (weight loading) or memset (buffer first-touch) */
typedef struct {
    void *dst;
    int fd;            /* -1 for memset-only (no pread) */
    size_t file_off;   /* base file offset (ignored if fd < 0) */
    size_t byte_start;
    size_t byte_end;
} tf_numa_task;

static void *tf_numa_pread_worker(void *arg) {
    tf_numa_task *t = (tf_numa_task *)arg;
    uint8_t *base = (uint8_t *)t->dst;
    size_t start = t->byte_start;
    size_t len   = t->byte_end - start;
    if (len == 0) return NULL;
    size_t off = 0;
    while (off < len) {
        size_t chunk = len - off;
        if (chunk > 1024 * 1024) chunk = 1024 * 1024;
        ssize_t n = pread(t->fd, base + start + off, chunk,
                          (off_t)(t->file_off + start + off));
        if (n <= 0) break;
        off += (size_t)n;
    }
    return NULL;
}

static void *tf_numa_memset_worker(void *arg) {
    tf_numa_task *t = (tf_numa_task *)arg;
    uint8_t *base = (uint8_t *)t->dst;
    size_t len = t->byte_end - t->byte_start;
    if (len > 0) memset(base + t->byte_start, 0, len);
    return NULL;
}

/* Initialize NUMA config from environment or defaults */
static void tf_numa_init(transformer_model *m) {
    memset(&m->numa, 0, sizeof(m->numa));
    m->numa.n_cmgs = 4;
    m->numa.per_cmg_budget = 7ULL * 1024 * 1024 * 1024;
    m->numa.alignment = 2 * 1024 * 1024;
    m->numa.enabled = 0;

    const char *env;
    if ((env = getenv("NUMA_N_CMGS")))       m->numa.n_cmgs = atoi(env);
    if ((env = getenv("NUMA_CMG_BUDGET_GB"))) m->numa.per_cmg_budget = (size_t)(atof(env) * 1024.0 * 1024.0 * 1024.0);
    if ((env = getenv("NUMA_ALIGNMENT")))     m->numa.alignment = (size_t)atol(env);
    if (getenv("NUMA_DISTRIBUTE"))            m->numa.enabled = 1;
}

/* Distribute a buffer across threads via parallel memset (first-touch placement) */
static void tf_numa_distribute_buffer(transformer_model *m, void *buf, size_t total_bytes) {
    int nt = m->n_threads;
    if (!buf || total_bytes == 0 || nt <= 1) return;

    tf_numa_task *tasks = (tf_numa_task *)alloca(nt * sizeof(tf_numa_task));
    size_t part = total_bytes / nt;
    size_t off = 0;
    for (int t = 0; t < nt; t++) {
        size_t end = (t == nt - 1) ? total_bytes : off + part;
        tasks[t] = (tf_numa_task){buf, -1, 0, off, end};
        off = end;
    }
    tf_pool_dispatch(m, tf_numa_memset_worker, tasks, sizeof(tf_numa_task));
}

/* ── Persistent NUMA-distributed 256-aligned bump pool ──
 * Replaces per-prefill posix_memalign/free (glibc mmap/munmap churn -> THP
 * fragmentation + page mis-placement -> progressive prefill slowdown). The arena
 * is mmap'd once and first-touched spread across CMGs (tf_numa_distribute_buffer,
 * which needs the thread pool ALIVE -> call ensure before tf_pool_shutdown).
 * Subsequent prefills reuse the placed pages (reset, no re-mmap, no re-touch). */
#define TF_MPOOL_ALIGN 256u
static void tf_mpool_reset(transformer_model *m) { m->mpool.off = 0; }
static int tf_mpool_ensure(transformer_model *m, size_t need) {
    if (m->mpool.base && m->mpool.cap >= need) return 1;
    size_t cap = need + (need >> 2);                 /* +25% headroom */
    cap = (cap + (2u<<20) - 1) & ~((size_t)(2u<<20) - 1);  /* round to 2MB (THP) */
    void *p = mmap(NULL, cap, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 0;
    if (m->mpool.base) munmap(m->mpool.base, m->mpool.cap);
    m->mpool.base = p; m->mpool.cap = cap; m->mpool.off = 0;
    if (m->numa.enabled) tf_numa_distribute_buffer(m, p, cap);  /* NUMA-spread first-touch */
    else memset(p, 0, cap);                                     /* still pre-fault once */
    return 1;
}
static void *tf_mpool_alloc(transformer_model *m, size_t bytes) {
    size_t a = (bytes + (TF_MPOOL_ALIGN - 1)) & ~(size_t)(TF_MPOOL_ALIGN - 1);
    if (m->mpool.off + a > m->mpool.cap) return NULL;
    void *r = (uint8_t *)m->mpool.base + m->mpool.off;
    m->mpool.off += a;
    return r;
}

/* Print per-CMG usage */
static void tf_numa_print_usage(transformer_model *m) {
    int nc = m->numa.n_cmgs < m->n_threads ? m->numa.n_cmgs : m->n_threads;
    fprintf(stderr, "numa: per-CMG usage:");
    for (int c = 0; c < nc; c++)
        fprintf(stderr, " CMG%d=%.1fMB", c, (double)m->numa.per_cmg_used[c] / (1024.0 * 1024.0));
    fprintf(stderr, " (budget=%.1fGB)\n", (double)m->numa.per_cmg_budget / (1024.0 * 1024.0 * 1024.0));
}

static int tf_numa_verify_tensor_sample(int fd, const char *name, const void *data,
                                        size_t file_off, size_t size, int full) {
    if (fd < 0 || !data || size == 0) return 0;
    uint8_t ref[64];
    const uint8_t *mem = (const uint8_t *)data;
    if (full) {
        uint8_t *buf = (uint8_t *)malloc(1024 * 1024);
        if (!buf) return 1;
        size_t off = 0;
        while (off < size) {
            size_t n = size - off;
            if (n > 1024 * 1024) n = 1024 * 1024;
            if (pread(fd, buf, n, (off_t)(file_off + off)) != (ssize_t)n) {
                fprintf(stderr, "numa: verify read failed tensor=%s off=%zu\n", name, off);
                free(buf);
                return 1;
            }
            if (memcmp(mem + off, buf, n) != 0) {
                fprintf(stderr, "numa: verify mismatch tensor=%s off=%zu size=%zu\n",
                        name, off, size);
                free(buf);
                return 1;
            }
            off += n;
        }
        free(buf);
        return 0;
    }
    size_t points[3];
    points[0] = 0;
    points[1] = size > sizeof(ref) ? size / 2 : 0;
    points[2] = size > sizeof(ref) ? size - sizeof(ref) : 0;
    for (int i = 0; i < 3; i++) {
        size_t off = points[i];
        size_t n = size - off;
        if (n > sizeof(ref)) n = sizeof(ref);
        if (pread(fd, ref, n, (off_t)(file_off + off)) != (ssize_t)n) {
            fprintf(stderr, "numa: verify read failed tensor=%s off=%zu\n", name, off);
            return 1;
        }
        if (memcmp(mem + off, ref, n) != 0) {
            fprintf(stderr, "numa: verify mismatch tensor=%s off=%zu size=%zu\n",
                    name, off, size);
            return 1;
        }
    }
    return 0;
}

void transformer_numa_setup(transformer_model *m, const gguf_context *gguf) {
    tf_numa_init(m);
    if (!m->numa.enabled || m->n_threads <= 1 || !m->pool_alive) {
        m->numa.enabled = 0;
        return;
    }

    int nt = m->n_threads;
    int fd = gguf->fd;

    fprintf(stderr, "numa: setup %d CMGs, %.1fGB/CMG, %d threads\n",
            m->numa.n_cmgs,
            (double)m->numa.per_cmg_budget / (1024.0 * 1024.0 * 1024.0), nt);

    /* ---- Phase 1: Distribute weight data via parallel pread ---- */
    if (fd >= 0) {
        fprintf(stderr, "numa: phase 1 - loading weights (fd=%d, %zu tensors)...\n",
                fd, (size_t)gguf->n_tensors);
        size_t weight_bytes = 0;
        int verify = getenv("TF_NUMA_VERIFY") ? atoi(getenv("TF_NUMA_VERIFY")) : 0;
        int verify_errors = 0;

        for (uint64_t ti = 0; ti < gguf->n_tensors; ti++) {
            void *tdata = gguf_tensor_data(gguf, (int)ti);
            size_t tsz = gguf_tensor_size(gguf, (int)ti);
            if (!tdata || tsz == 0) continue;

            size_t toff = gguf->data_offset + gguf->tensors[ti].offset;
            int n_dims = (int)gguf->tensors[ti].n_dims;
            int n_rows = 1;
            for (int d = 1; d < n_dims; d++)
                n_rows *= (int)gguf->tensors[ti].dims[d];
            size_t row_bytes = (n_rows > 0) ? tsz / (size_t)n_rows : tsz;

            if (n_rows < nt || n_dims <= 1) {
                tf_numa_task task = {tdata, fd, toff, 0, tsz};
                tf_numa_pread_worker(&task);
                m->numa.per_cmg_used[0] += tsz;
            } else {
                tf_numa_task *tasks = (tf_numa_task *)alloca(nt * sizeof(tf_numa_task));
                int rp = n_rows / nt, re = n_rows % nt, ro = 0;
                for (int t = 0; t < nt; t++) {
                    int rc = rp + (t < re ? 1 : 0);
                    size_t s = (size_t)ro * row_bytes;
                    size_t e = (t == nt - 1) ? tsz : (size_t)(ro + rc) * row_bytes;
                    if (e > tsz) e = tsz;
                    tasks[t] = (tf_numa_task){tdata, fd, toff, s, e};
                    int cmg = t < m->numa.n_cmgs ? t : t % m->numa.n_cmgs;
                    m->numa.per_cmg_used[cmg] += e - s;
                    ro += rc;
                }
                tf_pool_dispatch(m, tf_numa_pread_worker, tasks, sizeof(tf_numa_task));
            }
            if (verify && verify_errors < 8) {
                const char *name = gguf_tensor_name(gguf, (int)ti);
                verify_errors += tf_numa_verify_tensor_sample(fd, name ? name : "?",
                                                              tdata, toff, tsz, verify >= 2);
            }
            weight_bytes += tsz;
        }
        if (verify)
            fprintf(stderr, "numa: verify %s (%d sample mismatches)\n",
                    verify_errors ? "FAILED" : "ok", verify_errors);
        fprintf(stderr, "numa: phase 1 done (%.1fGB weights loaded)\n",
                (double)weight_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    /* ---- Phase 2: First-touch thread scratch from each worker ---- */
    /* thread_tmp[t>=1] allocated with notouch; first memset places pages on CMG t */
    {
        int max_ff = tf_compute_max_ff(m);
        int max_dim = m->n_embd > max_ff ? m->n_embd : max_ff;
        size_t buf_sz = (size_t)max_dim * sizeof(float);
        tf_numa_task *tasks = (tf_numa_task *)alloca(nt * sizeof(tf_numa_task));
        for (int t = 0; t < nt; t++) {
            tasks[t] = (tf_numa_task){m->thread_tmp[t], -1, 0, 0, buf_sz};
        }
        tf_pool_dispatch(m, tf_numa_memset_worker, tasks, sizeof(tf_numa_task));
        fprintf(stderr, "numa: phase 2 - thread scratch distributed\n");
    }

    /* ---- Phase 3: Distribute logits buffer (row-partitioned) ---- */
    if (m->logits && m->n_vocab > 0) {
        size_t logits_sz = (size_t)m->n_vocab * sizeof(float);
        tf_numa_distribute_buffer(m, m->logits, logits_sz);
        size_t per_cmg = logits_sz / nt;
        for (int t = 0; t < nt && t < m->numa.n_cmgs; t++)
            m->numa.per_cmg_used[t] += per_cmg;
        fprintf(stderr, "numa: phase 3 - logits distributed (%.1fKB)\n",
                (double)logits_sz / 1024.0);
    }

    /* ---- Phase 4: Distribute SSM recurrent state (head-partitioned) ---- */
    if (m->is_hybrid && m->recurrent_state) {
        int dt_rank = m->ssm_dt_rank;
        int d2 = m->ssm_d_state * m->ssm_d_state;
        size_t head_bytes = (size_t)d2 * sizeof(float);
        int distributed = 0;

        for (int l = 0; l < m->n_layers; l++) {
            if (!m->layers[l].is_ssm || !m->recurrent_state[l]) continue;
            if (dt_rank < nt) continue;

            tf_numa_task *tasks = (tf_numa_task *)alloca(nt * sizeof(tf_numa_task));
            int hp = dt_rank / nt, he = dt_rank % nt, ho = 0;
            for (int t = 0; t < nt; t++) {
                int hc = hp + (t < he ? 1 : 0);
                size_t s = (size_t)ho * head_bytes;
                size_t e = (size_t)(ho + hc) * head_bytes;
                tasks[t] = (tf_numa_task){m->recurrent_state[l], -1, 0, s, e};
                ho += hc;
            }
            tf_pool_dispatch(m, tf_numa_memset_worker, tasks, sizeof(tf_numa_task));
            distributed++;
        }
        if (distributed > 0)
            fprintf(stderr, "numa: phase 4 - SSM state distributed (%d layers)\n", distributed);
    }

    tf_numa_print_usage(m);
    fprintf(stderr, "numa: setup complete\n");
}

/* ---- SSM Delta-Net forward (single token, autoregressive) ---- */

/* L2-normalize a vector in-place: v[i] /= sqrt(sum(v[i]^2) + eps) */
static void tf_l2_norm(float *v, int n, float eps) {
#if defined(__AVX2__) && defined(__FMA__)
    float ss = 0.0f;
    int i = 0;
    __m256 vss = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 vi = _mm256_loadu_ps(v + i);
        vss = _mm256_fmadd_ps(vi, vi, vss);
    }
    __m128 hi128 = _mm256_extractf128_ps(vss, 1);
    __m128 lo128 = _mm256_castps256_ps128(vss);
    __m128 s4 = _mm_add_ps(lo128, hi128);
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    ss = _mm_cvtss_f32(s4);
    for (; i < n; i++) ss += v[i] * v[i];
    float inv = 1.0f / sqrtf(ss + eps);
    __m256 vinv = _mm256_set1_ps(inv);
    i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_loadu_ps(v + i), vinv));
    for (; i < n; i++) v[i] *= inv;
#else
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += v[i] * v[i];
    float inv = 1.0f / sqrtf(ss + eps);
    for (int i = 0; i < n; i++) v[i] *= inv;
#endif
}

/* SSM Delta-Net recurrence task for multi-threaded head dispatch */
typedef struct {
    float *rec_state;       /* [dt_rank * d_state * d_state] */
    float *Q_exp;           /* [dt_rank * d_state] */
    float *K_exp;           /* [dt_rank * d_state] */
    float *V_raw;           /* [dt_rank * d_state] */
    float *out_buf;         /* [dt_rank * d_state] */
    const float *alpha;     /* [dt_rank] */
    const float *beta_arr;  /* [dt_rank] */
    int head_start, head_end;
    int d_state;
    float scale;
} tf_ssm_recurrence_task;

static void *tf_ssm_recurrence_worker(void *arg) {
    tf_ssm_recurrence_task *t = (tf_ssm_recurrence_task *)arg;
    int ds = t->d_state;
    int d2 = ds * ds;

    for (int h = t->head_start; h < t->head_end; h++) {
        float *state = t->rec_state + (size_t)h * d2;
        float *q_h = t->Q_exp + h * ds;
        float *k_h = t->K_exp + h * ds;
        float *v_h = t->V_raw + h * ds;
        float *o_h = t->out_buf + h * ds;

#if defined(__AVX2__) && defined(__FMA__)
        /* Scale Q */
        {
            __m256 vscale = _mm256_set1_ps(t->scale);
            int i = 0;
            for (; i + 7 < ds; i += 8)
                _mm256_storeu_ps(q_h + i, _mm256_mul_ps(_mm256_loadu_ps(q_h + i), vscale));
            for (; i < ds; i++) q_h[i] *= t->scale;
        }

        /* Decay: state *= exp(alpha_h) */
        {
            float decay = expf(t->alpha[h]);
            __m256 vdecay = _mm256_set1_ps(decay);
            int i = 0;
            for (; i + 7 < d2; i += 8)
                _mm256_storeu_ps(state + i, _mm256_mul_ps(_mm256_loadu_ps(state + i), vdecay));
            for (; i < d2; i++) state[i] *= decay;
        }

        /* Read: sk = state @ k (2-row to share k loads) */
        float sk[128];
        {
            int r = 0;
            for (; r + 1 < ds; r += 2) {
                float *row0 = state + r * ds;
                float *row1 = state + (r + 1) * ds;
                /* Prefetch next 2 rows (512B each = 8 cache lines) */
                if (r + 3 < ds) {
                    for (int pf = 0; pf < ds; pf += 16) {
                        _mm_prefetch((const char *)(state + (r+2)*ds + pf), _MM_HINT_T0);
                        _mm_prefetch((const char *)(state + (r+3)*ds + pf), _MM_HINT_T0);
                    }
                }
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
                __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
                int c = 0;
                for (; c + 31 < ds; c += 32) {
                    __m256 k0 = _mm256_loadu_ps(k_h + c);
                    __m256 k1 = _mm256_loadu_ps(k_h + c + 8);
                    __m256 k2 = _mm256_loadu_ps(k_h + c + 16);
                    __m256 k3 = _mm256_loadu_ps(k_h + c + 24);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c),      k0, a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c + 8),  k1, a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c + 16), k2, a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c + 24), k3, a3);
                    b0 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c),      k0, b0);
                    b1 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c + 8),  k1, b1);
                    b2 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c + 16), k2, b2);
                    b3 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c + 24), k3, b3);
                }
                for (; c + 7 < ds; c += 8) {
                    __m256 kv = _mm256_loadu_ps(k_h + c);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c), kv, a0);
                    b0 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c), kv, b0);
                }
                a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
                b0 = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
                __m128 ahi = _mm256_extractf128_ps(a0, 1), alo = _mm256_castps256_ps128(a0);
                __m128 as4 = _mm_add_ps(alo, ahi);
                as4 = _mm_add_ps(as4, _mm_movehl_ps(as4, as4));
                as4 = _mm_add_ss(as4, _mm_movehdup_ps(as4));
                float sum0 = _mm_cvtss_f32(as4);
                __m128 bhi = _mm256_extractf128_ps(b0, 1), blo = _mm256_castps256_ps128(b0);
                __m128 bs4 = _mm_add_ps(blo, bhi);
                bs4 = _mm_add_ps(bs4, _mm_movehl_ps(bs4, bs4));
                bs4 = _mm_add_ss(bs4, _mm_movehdup_ps(bs4));
                float sum1 = _mm_cvtss_f32(bs4);
                for (; c < ds; c++) { sum0 += row0[c] * k_h[c]; sum1 += row1[c] * k_h[c]; }
                sk[r] = sum0;
                sk[r + 1] = sum1;
            }
            for (; r < ds; r++) {
                float *row = state + r * ds;
                __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
                int c = 0;
                for (; c + 31 < ds; c += 32) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c), _mm256_loadu_ps(k_h + c), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 8), _mm256_loadu_ps(k_h + c + 8), acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 16), _mm256_loadu_ps(k_h + c + 16), acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 24), _mm256_loadu_ps(k_h + c + 24), acc3);
                }
                acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                __m128 hi = _mm256_extractf128_ps(acc0, 1), lo = _mm256_castps256_ps128(acc0);
                __m128 s4 = _mm_add_ps(lo, hi);
                s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
                sk[r] = _mm_cvtss_f32(s4);
            }
        }

        /* Delta: d = (v - sk) * beta */
        float delta[128];
        {
            float beta_h = t->beta_arr[h];
            __m256 vbeta = _mm256_set1_ps(beta_h);
            int i = 0;
            for (; i + 7 < ds; i += 8) {
                __m256 d = _mm256_sub_ps(_mm256_loadu_ps(v_h + i), _mm256_loadu_ps(sk + i));
                _mm256_storeu_ps(delta + i, _mm256_mul_ps(d, vbeta));
            }
            for (; i < ds; i++) delta[i] = (v_h[i] - sk[i]) * beta_h;
        }

        /* Update: state[r][c] += delta[r] * k[c] */
        for (int r = 0; r < ds; r++) {
            float *row = state + r * ds;
            if (r + 1 < ds)
                _mm_prefetch((const char *)(state + (r+1)*ds), _MM_HINT_T0);
            __m256 vdr = _mm256_set1_ps(delta[r]);
            int c = 0;
            for (; c + 7 < ds; c += 8)
                _mm256_storeu_ps(row + c, _mm256_fmadd_ps(vdr, _mm256_loadu_ps(k_h + c),
                                                            _mm256_loadu_ps(row + c)));
            for (; c < ds; c++) row[c] += delta[r] * k_h[c];
        }

        /* Output: o = state @ q (2-row to share q loads) */
        {
            int r = 0;
            for (; r + 1 < ds; r += 2) {
                float *row0 = state + r * ds;
                float *row1 = state + (r + 1) * ds;
                /* Prefetch next 2 rows for output matvec */
                if (r + 3 < ds) {
                    for (int pf = 0; pf < ds; pf += 16) {
                        _mm_prefetch((const char *)(state + (r+2)*ds + pf), _MM_HINT_T0);
                        _mm_prefetch((const char *)(state + (r+3)*ds + pf), _MM_HINT_T0);
                    }
                }
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
                __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
                int c = 0;
                for (; c + 31 < ds; c += 32) {
                    __m256 q0 = _mm256_loadu_ps(q_h + c);
                    __m256 q1 = _mm256_loadu_ps(q_h + c + 8);
                    __m256 q2 = _mm256_loadu_ps(q_h + c + 16);
                    __m256 q3 = _mm256_loadu_ps(q_h + c + 24);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c),      q0, a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c + 8),  q1, a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c + 16), q2, a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c + 24), q3, a3);
                    b0 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c),      q0, b0);
                    b1 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c + 8),  q1, b1);
                    b2 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c + 16), q2, b2);
                    b3 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c + 24), q3, b3);
                }
                for (; c + 7 < ds; c += 8) {
                    __m256 qv = _mm256_loadu_ps(q_h + c);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c), qv, a0);
                    b0 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c), qv, b0);
                }
                a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
                b0 = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
                __m128 ahi = _mm256_extractf128_ps(a0, 1), alo = _mm256_castps256_ps128(a0);
                __m128 as4 = _mm_add_ps(alo, ahi);
                as4 = _mm_add_ps(as4, _mm_movehl_ps(as4, as4));
                as4 = _mm_add_ss(as4, _mm_movehdup_ps(as4));
                o_h[r] = _mm_cvtss_f32(as4);
                __m128 bhi = _mm256_extractf128_ps(b0, 1), blo = _mm256_castps256_ps128(b0);
                __m128 bs4 = _mm_add_ps(blo, bhi);
                bs4 = _mm_add_ps(bs4, _mm_movehl_ps(bs4, bs4));
                bs4 = _mm_add_ss(bs4, _mm_movehdup_ps(bs4));
                o_h[r + 1] = _mm_cvtss_f32(bs4);
                for (; c < ds; c++) { o_h[r] += row0[c] * q_h[c]; o_h[r + 1] += row1[c] * q_h[c]; }
            }
            for (; r < ds; r++) {
                float *row = state + r * ds;
                __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
                int c = 0;
                for (; c + 31 < ds; c += 32) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c), _mm256_loadu_ps(q_h + c), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 8), _mm256_loadu_ps(q_h + c + 8), acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 16), _mm256_loadu_ps(q_h + c + 16), acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 24), _mm256_loadu_ps(q_h + c + 24), acc3);
                }
                acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                __m128 hi = _mm256_extractf128_ps(acc0, 1), lo = _mm256_castps256_ps128(acc0);
                __m128 s4 = _mm_add_ps(lo, hi);
                s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
                o_h[r] = _mm_cvtss_f32(s4);
            }
        }
#else
        /* Scalar fallback */
        for (int i = 0; i < ds; i++) q_h[i] *= t->scale;
        float decay = expf(t->alpha[h]);
        for (int i = 0; i < d2; i++) state[i] *= decay;

        float sk[128];
        for (int r = 0; r < ds; r++) {
            float sum = 0.0f;
            for (int c = 0; c < ds; c++) sum += state[r * ds + c] * k_h[c];
            sk[r] = sum;
        }
        float delta[128];
        for (int i = 0; i < ds; i++) delta[i] = (v_h[i] - sk[i]) * t->beta_arr[h];
        for (int r = 0; r < ds; r++) {
            float dr = delta[r];
            for (int c = 0; c < ds; c++) state[r * ds + c] += dr * k_h[c];
        }
        for (int r = 0; r < ds; r++) {
            float sum = 0.0f;
            for (int c = 0; c < ds; c++) sum += state[r * ds + c] * q_h[c];
            o_h[r] = sum;
        }
#endif
    }
    return NULL;
}

/* SSM Delta-Net forward for one layer.
 * Input:  m->xb (post-norm hidden state [n_embd])
 * Output: m->xb (residual-ready output [n_embd])
 * Scratch: m->xb2 (qkv), m->q (Q_expanded), m->ffn_buf1 (z/gate),
 *          m->ffn_buf2 (K_expanded), m->ffn_buf3 (delta-net output) */
static void tf_ssm_deltanet_forward(transformer_model *m, int layer_idx) {
    transformer_layer *layer = &m->layers[layer_idx];
    int n_embd  = m->n_embd;
    int qkv_dim = m->ssm_qkv_dim;
    int d_inner = m->ssm_d_inner;
    int d_state = m->ssm_d_state;
    int n_group = m->ssm_n_group;
    int dt_rank = m->ssm_dt_rank;
    int conv_k  = m->ssm_conv_kernel;
    float eps   = m->rms_norm_eps;

    float *qkv_buf = m->xb2;       /* [qkv_dim] */
    float *z_buf   = m->ffn_buf1;  /* [d_inner] */
    float *K_exp   = m->ffn_buf2;  /* [dt_rank * d_state] */
    float *out_buf = m->ffn_buf3;  /* [d_inner] */
    float *Q_exp   = m->q;         /* [dt_rank * d_state] */

    /* 1. Linear projections from xb (pool-based for large projections) */
    tf_qmatvec_pool(m, qkv_buf, &layer->ssm_qkv, m->xb, qkv_dim);
    tf_qmatvec_pool(m, z_buf, &layer->ssm_gate, m->xb, d_inner);

    float alpha[64], beta_arr[64]; /* dt_rank <= 64 (48 for Qwen3.5-27B) */
    tf_qmatvec(alpha, &layer->ssm_alpha, m->xb, dt_rank, m->matvec_tmp);
    tf_qmatvec(beta_arr, &layer->ssm_beta, m->xb, dt_rank, m->matvec_tmp);

    /* 2. alpha = softplus(alpha + dt_bias) * ssm_a */
    {
        float a_buf[64], dt_bias_buf[64];
        tf_dequant_row(&layer->ssm_a, 0, a_buf);
        tf_dequant_row(&layer->ssm_dt_bias, 0, dt_bias_buf);
        for (int i = 0; i < dt_rank; i++) {
            float val = alpha[i] + dt_bias_buf[i];
            float sp = (val > 20.0f) ? val : logf(1.0f + expf(val));
            alpha[i] = sp * a_buf[i]; /* negative since ssm_a < 0 */
        }
    }

    /* 3. beta = sigmoid(beta) */
#if defined(__AVX2__) && defined(__FMA__)
    {
        const __m256 one = _mm256_set1_ps(1.0f);
        int i = 0;
        for (; i + 7 < dt_rank; i += 8) {
            __m256 b = _mm256_loadu_ps(beta_arr + i);
            __m256 neg_b = _mm256_sub_ps(_mm256_setzero_ps(), b);
            __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, fast_exp_avx2(neg_b)));
            _mm256_storeu_ps(beta_arr + i, sig);
        }
        for (; i < dt_rank; i++)
            beta_arr[i] = 1.0f / (1.0f + expf(-beta_arr[i]));
    }
#else
    for (int i = 0; i < dt_rank; i++) {
        beta_arr[i] = 1.0f / (1.0f + expf(-beta_arr[i]));
    }
#endif

    /* 4. Conv1d: depthwise causal conv + SiLU (circular buffer) */
    {
        float *conv_st = m->conv_state[layer_idx]; /* [(conv_k-1) * qkv_dim] */
        int wr = m->conv_state_pos[layer_idx]; /* circular buffer write position */
        int n_hist = conv_k - 1;
        /* Use K_exp temporarily for conv output (17408 >= 10240) */
        float *conv_out = K_exp;

        /* Batch-dequant all conv weights into transposed layout [conv_k][qkv_dim] */
        float *w_trans = m->conv_w_trans;
        {
            size_t crb = tf_row_bytes(layer->ssm_conv1d.type, layer->ssm_conv1d.n_cols);
            const uint8_t *cbase = (const uint8_t *)layer->ssm_conv1d.data;
            for (int j = 0; j < qkv_dim; j++) {
                float wb[8];
                dequant_row(layer->ssm_conv1d.type, cbase + j * crb, wb, conv_k);
                for (int f = 0; f < conv_k; f++)
                    w_trans[f * qkv_dim + j] = wb[f];
            }
        }

        /* Precompute circular buffer row offsets to avoid modulo in inner loop */
        int row_off[8]; /* conv_k <= 8, n_hist = conv_k-1 */
        for (int f = 0; f < n_hist; f++)
            row_off[f] = ((wr + f) % n_hist) * qkv_dim;

        /* Vectorized conv MAC over channels */
#if defined(__AVX2__) && defined(__FMA__)
        {
            int j = 0;
            for (; j + 7 < qkv_dim; j += 8) {
                /* Prefetch conv_st rows 64 bytes ahead */
                if (j + 16 < qkv_dim) {
                    for (int f = 0; f < n_hist; f++)
                        _mm_prefetch((const char *)(conv_st + row_off[f] + j + 16), _MM_HINT_T0);
                }
                __m256 sum = _mm256_setzero_ps();
                for (int f = 0; f < n_hist; f++)
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(w_trans + f * qkv_dim + j),
                                           _mm256_loadu_ps(conv_st + row_off[f] + j), sum);
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(w_trans + n_hist * qkv_dim + j),
                                       _mm256_loadu_ps(qkv_buf + j), sum);
                _mm256_storeu_ps(conv_out + j, sum);
            }
            for (; j < qkv_dim; j++) {
                float s = 0.0f;
                for (int f = 0; f < n_hist; f++)
                    s += w_trans[f * qkv_dim + j] * conv_st[row_off[f] + j];
                conv_out[j] = s + w_trans[n_hist * qkv_dim + j] * qkv_buf[j];
            }
        }
#else
        for (int j = 0; j < qkv_dim; j++) {
            float sum = 0.0f;
            for (int f = 0; f < n_hist; f++)
                sum += w_trans[f * qkv_dim + j] * conv_st[row_off[f] + j];
            conv_out[j] = sum + w_trans[n_hist * qkv_dim + j] * qkv_buf[j];
        }
#endif
        /* Vectorized SiLU activation */
#if defined(__AVX2__) && defined(__FMA__)
        {
            const __m256 one = _mm256_set1_ps(1.0f);
            int j = 0;
            for (; j + 7 < qkv_dim; j += 8) {
                __m256 s = _mm256_loadu_ps(conv_out + j);
                __m256 neg_s = _mm256_sub_ps(_mm256_setzero_ps(), s);
                __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, fast_exp_avx2(neg_s)));
                _mm256_storeu_ps(conv_out + j, _mm256_mul_ps(s, sig));
            }
            for (; j < qkv_dim; j++)
                conv_out[j] = conv_out[j] / (1.0f + expf(-conv_out[j]));
        }
#else
        for (int j = 0; j < qkv_dim; j++)
            conv_out[j] = conv_out[j] / (1.0f + expf(-conv_out[j]));
#endif

        /* Update circular buffer: overwrite oldest slot, advance write position */
        memcpy(conv_st + wr * qkv_dim, qkv_buf, qkv_dim * sizeof(float));
        m->conv_state_pos[layer_idx] = (wr + 1) % n_hist;

        /* Copy conv output to qkv_buf */
        memcpy(qkv_buf, conv_out, qkv_dim * sizeof(float));
    }

    /* 5. Split: Q[n_group*d_state], K[n_group*d_state], V[dt_rank*d_state]
     *    Order in qkv_buf: Q, K, V (following llama.cpp convention) */
    float *Q_raw = qkv_buf;                                  /* [n_group * d_state] */
    float *K_raw = qkv_buf + n_group * d_state;              /* [n_group * d_state] */
    float *V_raw = qkv_buf + 2 * n_group * d_state;          /* [dt_rank * d_state = d_inner] */

    /* L2-normalize Q and K per head (n_group heads, d_state elements each) */
    for (int g = 0; g < n_group; g++) {
        tf_l2_norm(Q_raw + g * d_state, d_state, eps);
        tf_l2_norm(K_raw + g * d_state, d_state, eps);
    }

    /* Repeat Q and K from n_group to dt_rank heads (tiling, matching ggml_repeat) */
    /* dt_rank=48, n_group=16: 3 bulk copies of n_group*d_state instead of 96 memcpys */
    {
        size_t tile_bytes = (size_t)n_group * d_state * sizeof(float);
        int n_repeat = dt_rank / n_group;
        for (int r = n_repeat - 1; r >= 0; r--) {
            memcpy(Q_exp + r * n_group * d_state, Q_raw, tile_bytes);
            memcpy(K_exp + r * n_group * d_state, K_raw, tile_bytes);
        }
    }

    /* 6. Delta-Net recurrence per head (AVX2 + multi-threaded) */
    float scale = 1.0f / sqrtf((float)d_state);
    float *rec_state = m->recurrent_state[layer_idx]; /* [dt_rank * d_state * d_state] */

    if (m->n_threads > 1 && dt_rank >= m->n_threads && m->pool_alive) {
        int nt = m->n_threads;
        tf_ssm_recurrence_task *rtasks = (tf_ssm_recurrence_task *)alloca(
            nt * sizeof(tf_ssm_recurrence_task));
        int heads_per = dt_rank / nt, heads_extra = dt_rank % nt, hoff = 0;
        for (int t = 0; t < nt; t++) {
            int hcount = heads_per + (t < heads_extra ? 1 : 0);
            rtasks[t] = (tf_ssm_recurrence_task){
                rec_state, Q_exp, K_exp, V_raw, out_buf,
                alpha, beta_arr, hoff, hoff + hcount, d_state, scale
            };
            hoff += hcount;
        }
        tf_pool_dispatch(m, tf_ssm_recurrence_worker, rtasks, sizeof(tf_ssm_recurrence_task));
    } else {
        tf_ssm_recurrence_task rtask = {
            rec_state, Q_exp, K_exp, V_raw, out_buf,
            alpha, beta_arr, 0, dt_rank, d_state, scale
        };
        tf_ssm_recurrence_worker(&rtask);
    }

    /* 7. Fused: out = rmsnorm(out, ssm_norm) * silu(z) */
    {
        /* ssm_norm has [d_state] weights, applied per-head (48 groups of 128) */
        float norm_w[128];
        tf_dequant_row(&layer->ssm_norm, 0, norm_w);
        for (int h = 0; h < dt_rank; h++) {
            float *o_h = out_buf + h * d_state;
            float *z_h = z_buf + h * d_state;
#if defined(__AVX2__) && defined(__FMA__)
            /* RMSNorm: sum of squares (AVX2) */
            __m256 vss = _mm256_setzero_ps();
            int i = 0;
            for (; i + 7 < d_state; i += 8) {
                __m256 oi = _mm256_loadu_ps(o_h + i);
                vss = _mm256_fmadd_ps(oi, oi, vss);
            }
            __m128 hi128 = _mm256_extractf128_ps(vss, 1);
            __m128 lo128 = _mm256_castps256_ps128(vss);
            __m128 s4 = _mm_add_ps(lo128, hi128);
            s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
            float ss = _mm_cvtss_f32(s4);
            for (; i < d_state; i++) ss += o_h[i] * o_h[i];
            __m256 vscale = _mm256_set1_ps(1.0f / sqrtf(ss / d_state + eps));
            const __m256 one = _mm256_set1_ps(1.0f);
            /* Apply norm * weight * silu(z) */
            i = 0;
            for (; i + 7 < d_state; i += 8) {
                __m256 oi = _mm256_loadu_ps(o_h + i);
                __m256 wi = _mm256_loadu_ps(norm_w + i);
                __m256 normed = _mm256_mul_ps(_mm256_mul_ps(oi, vscale), wi);
                __m256 zi = _mm256_loadu_ps(z_h + i);
                __m256 neg_z = _mm256_sub_ps(_mm256_setzero_ps(), zi);
                __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, fast_exp_avx2(neg_z)));
                _mm256_storeu_ps(o_h + i, _mm256_mul_ps(normed, _mm256_mul_ps(zi, sig)));
            }
            float ss_scalar = 1.0f / sqrtf(ss / d_state + eps);
            for (; i < d_state; i++) {
                float normed = o_h[i] * ss_scalar * norm_w[i];
                float z_val = z_h[i];
                o_h[i] = normed * (z_val / (1.0f + expf(-z_val)));
            }
#else
            /* RMSNorm per head */
            float ss = 0.0f;
            for (int i = 0; i < d_state; i++) ss += o_h[i] * o_h[i];
            ss = 1.0f / sqrtf(ss / d_state + eps);
            /* Apply norm * weight * silu(z) */
            for (int i = 0; i < d_state; i++) {
                float normed = o_h[i] * ss * norm_w[i];
                float z_val = z_h[i];
                float silu_z = z_val / (1.0f + expf(-z_val));
                o_h[i] = normed * silu_z;
            }
#endif
        }
    }

    /* 8. Output projection: xb = ssm_out @ out_buf */
    tf_qmatvec_pool(m, m->xb, &layer->ssm_out, out_buf, n_embd);
}

static inline float tf_gelu_exact_scalar(float g) {
    return g * 0.5f * (1.0f + erff(g * 0.7071067811865476f));
}

static inline float tf_gelu_fast_scalar(float g) {
    float a = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    float a2 = a * a;
    float t = a * (27.0f + a2) / (27.0f + 9.0f * a2);
    if (t > 1.0f) t = 1.0f;
    if (t < -1.0f) t = -1.0f;
    return 0.5f * g * (1.0f + t);
}

static void tf_gelu_mul_exact(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        out[i] = tf_gelu_exact_scalar(g) * up[i];
    }
}

static inline float tf_gelu_mul_exact_scalar(float gate, float up) {
    return tf_gelu_exact_scalar(gate) * up;
}

static inline float tf_gelu_mul_fast_scalar(float gate, float up) {
    return tf_gelu_fast_scalar(gate) * up;
}

#if defined(__ARM_FEATURE_SVE)
static inline svfloat32_t tf_exp2_fexpa_approx_sve(svbool_t pg, svfloat32_t x) {
    const float shift_f = 204927.0f; /* 0x48481fc0: FEXPA-compatible rounding shift */
    svfloat32_t shift = svdup_f32(shift_f);
    svfloat32_t z = svadd_f32_x(pg, x, shift);
    svfloat32_t n = svsub_f32_x(pg, z, shift);
    svfloat32_t r = svsub_f32_x(pg, x, n);
    svfloat32_t scale = svexpa_f32(svreinterpret_u32_f32(z));
    svfloat32_t corr = svmla_n_f32_x(pg, svdup_f32(1.0f), r, 0.6931471805599453f);
    return svmul_f32_x(pg, scale, corr);
}

static inline svfloat32_t tf_gelu_fast_sve(svbool_t pg, svfloat32_t g) {
    svfloat32_t g2 = svmul_f32_x(pg, g, g);
    svfloat32_t g3 = svmul_f32_x(pg, g2, g);
    svfloat32_t a = svmla_n_f32_x(pg, g, g3, 0.044715f);
    a = svmul_n_f32_x(pg, a, 0.7978845608028654f);
    svfloat32_t t = svmul_n_f32_x(pg, a, -2.8853900817779268f); /* -2 * log2(e) */
    t = svmax_n_f32_x(pg, svmin_n_f32_x(pg, t, 80.0f), -80.0f);
    svfloat32_t exp_neg = tf_exp2_fexpa_approx_sve(pg, t);
    svfloat32_t denom = svadd_f32_x(pg, svdup_f32(1.0f), exp_neg);
    svfloat32_t inv = svrecpe_f32(denom);
    inv = svmul_f32_x(pg, inv, svrecps_f32(denom, inv));
    return svmul_f32_x(pg, g, inv);
}

static void tf_gelu_mul_fast_sve(float *out, const float *gate, const float *up, int n) {
    int i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t g = svld1(pg, gate + i);
        svfloat32_t u = svld1(pg, up + i);
        svfloat32_t y = svmul_f32_x(pg, tf_gelu_fast_sve(pg, g), u);
        svst1(pg, out + i, y);
        i += (int)svcntw();
    }
}

static inline void tf_gelu_mul_fast4_sve(float *out, size_t stride, int row,
                                         int tok, float g0, float g1,
                                         float g2, float g3, float u0,
                                         float u1, float u2, float u3) {
    float y[4];
    svbool_t pg = svwhilelt_b32(0, 4);
    svfloat32_t gv = svdupq_n_f32(g0, g1, g2, g3);
    svfloat32_t uv = svdupq_n_f32(u0, u1, u2, u3);
    svst1(pg, y, svmul_f32_x(pg, tf_gelu_fast_sve(pg, gv), uv));
    out[(size_t)(tok + 0) * stride + row] = y[0];
    out[(size_t)(tok + 1) * stride + row] = y[1];
    out[(size_t)(tok + 2) * stride + row] = y[2];
    out[(size_t)(tok + 3) * stride + row] = y[3];
}
#endif

static void tf_gelu_mul_fast(float *out, const float *gate, const float *up, int n) {
#if defined(__ARM_FEATURE_SVE)
    tf_gelu_mul_fast_sve(out, gate, up, n);
#else
    for (int i = 0; i < n; i++) out[i] = tf_gelu_mul_fast_scalar(gate[i], up[i]);
#endif
}

/* GELU(gate) × up dispatcher. Exact mode keeps the previous erf GELU. */
static void tf_gelu_mul(float *out, const float *gate, const float *up, int n, int fast) {
    if (fast) tf_gelu_mul_fast(out, gate, up, n);
    else tf_gelu_mul_exact(out, gate, up, n);
}

/* Logit soft-capping: logits[i] = cap * tanh(logits[i] / cap) */
static void tf_logit_softcap(float *logits, int n, float cap) {
    float inv_cap = 1.0f / cap;
    for (int i = 0; i < n; i++)
        logits[i] = cap * tanhf(logits[i] * inv_cap);
}

/* Helper: apply standard or M-RoPE depending on model config */
static void tf_apply_rope(transformer_model *m, float *q, float *k,
                           int n_heads, int n_kv_heads, int head_dim,
                           int pos_t, int pos_h, int pos_w) {
    if (m->use_mrope) {
        tf_rope_mrope(q, n_heads,    head_dim, pos_t, pos_h, pos_w, m->rope_freq_base, m->mrope_sections, m->rope_mrope_inv_freq);
        tf_rope_mrope(k, n_kv_heads, head_dim, pos_t, pos_h, pos_w, m->rope_freq_base, m->mrope_sections, m->rope_mrope_inv_freq);
    } else {
        tf_rope(q, n_heads,    head_dim, pos_t, m->rope_freq_base, m->rope_inv_freq);
        tf_rope(k, n_kv_heads, head_dim, pos_t, m->rope_freq_base, m->rope_inv_freq);
    }
}

/* ---- Persistent-thread forward pass ---- */
/* Each thread runs the entire layer loop for its static partition.
 * Sync uses pthread_barrier (kernel-backed, no spin-wait).
 * Thread 0 does sequential work (RMSNorm, RoPE, KV cache); others wait at barrier.
 * All threads participate in parallel work (matvec rows, attention heads). */

typedef struct {
    transformer_model *m;
    int tid;
    int position, pos_t, pos_h, pos_w;
} tf_persistent_ctx;

/* ---- Barrier implementations ---- */

/* A64FX hardware barrier via W0 window register (intra-CMG only).
 * Uses WFE (wait-for-event) instead of spin-polling — near-zero power
 * and ~20 cycle latency. Requires libfjomphk.so (link with -Kopenmp)
 * to have initialized the barrier blade. */
#if defined(__aarch64__)
static inline void tf_hw_barrier_w0(void) {
    uint64_t bst, lbsy;
    __asm__ __volatile__("mrs %0, S3_3_C15_C15_0" : "=r"(lbsy));
    bst = (~lbsy) & 1ULL;
    __asm__ __volatile__("msr S3_3_C15_C15_0, %0" :: "r"(bst));
    __asm__ __volatile__("sevl");
    do {
        __asm__ __volatile__("wfe");
        __asm__ __volatile__("mrs %0, S3_3_C15_C15_0" : "=r"(lbsy));
        lbsy &= 1ULL;
    } while (lbsy != bst);
}
#endif

/* Barrier: sense-reversing with SEV/WFE on aarch64 for low-power wait.
 * WFE puts the core in low-power state until an event (SEV from the last
 * arriving thread). No bandwidth consumption while waiting. */
static inline void tf_spin_barrier(transformer_model *m, int *local_sense, int nt) {
    int my_sense = !(*local_sense);
    if (__sync_add_and_fetch(&m->bar_count, 1) == nt) {
        m->bar_count = 0;
        __sync_synchronize();
        m->bar_sense = my_sense;
#if defined(__aarch64__)
        __asm__ __volatile__("sev");  /* wake all WFE-sleeping cores */
#endif
    } else {
#if defined(__aarch64__)
        /* SEV/WFE spin: core sleeps until event, then rechecks.
         * SEVL ensures first WFE doesn't stall if event pending. */
        __asm__ __volatile__("sevl");
        do {
            __asm__ __volatile__("wfe");
        } while (m->bar_sense != my_sense);
#else
        while (m->bar_sense != my_sense)
            tf_cpu_pause();
#endif
    }
    *local_sense = my_sense;
}

/* Two-level barrier for multi-CMG:
 * Level 1: Hardware barrier within each CMG (cores on same CMG sync via W0)
 * Level 2: Software atomic barrier across CMG leaders
 *
 * For single-CMG (all threads on 1 CMG): use hardware barrier directly.
 * For multi-CMG: each CMG's threads hw-barrier locally, then one thread
 * per CMG does a software barrier with other CMG leaders. */
static inline void tf_barrier(transformer_model *m, int tid, int *local_sense, int nt) {
#if defined(__aarch64__)
    int n_cmgs = m->numa.enabled ? m->numa.n_cmgs : 1;
    if (n_cmgs <= 1 || nt <= n_cmgs) {
        /* Single CMG or 1 thread/CMG: just use hw barrier */
        tf_hw_barrier_w0();
    } else {
        /* Multi-CMG with multiple threads per CMG */
        int threads_per_cmg = nt / n_cmgs;
        int cmg_id = tid / threads_per_cmg;
        int local_tid = tid % threads_per_cmg;

        /* Level 1: intra-CMG hardware barrier */
        tf_hw_barrier_w0();

        /* Level 2: inter-CMG software barrier (only CMG leaders, local_tid==0) */
        if (local_tid == 0) {
            int my_sense = !(*local_sense);
            if (__sync_add_and_fetch(&m->bar_count, 1) == n_cmgs) {
                m->bar_count = 0;
                __sync_synchronize();
                m->bar_sense = my_sense;
            } else {
                while (m->bar_sense != my_sense)
                    tf_cpu_pause();
            }
            *local_sense = my_sense;
        }

        /* Level 1 again: intra-CMG hw barrier to release non-leaders */
        tf_hw_barrier_w0();
    }
#else
    (void)tid;
    tf_spin_barrier(m, local_sense, nt);
#endif
}

/* Per-thread matvec: thread tid computes its static row partition */
static void tf_thread_matvec(float *dst, const qtensor *mat, const float *x,
                              int n_rows, int tid, int nt) {
    int rp = n_rows / nt, re = n_rows % nt;
    int rs = tid * rp + (tid < re ? tid : re);
    int rc = rp + (tid < re ? 1 : 0);
    if (rc <= 0) return;
    int n_cols = mat->n_cols;

    if (mat->type == GGML_TYPE_BF16) {
        tf_matvec_bf16_rows(dst, (const uint8_t *)mat->data,
                             (size_t)n_cols * 2, x, n_cols, rs, rs + rc);
    } else if (mat->type == GGML_TYPE_F16) {
        tf_matvec_f16_rows(dst, (const uint8_t *)mat->data,
                            (size_t)n_cols * 2, x, n_cols, rs, rs + rc);
    } else {
        tf_matvec_qtensor_rows(dst, mat, x, rs, rs + rc);
    }
}

static void *tf_persistent_worker(void *arg) {
    tf_persistent_ctx *ctx = (tf_persistent_ctx *)arg;
    transformer_model *m = ctx->m;
    int tid = ctx->tid;
    int nt = m->n_threads;
    int position = ctx->position;
    int pos_t = ctx->pos_t, pos_h = ctx->pos_h, pos_w = ctx->pos_w;
    int local_sense = 0;

    int n_embd = m->n_embd;
    int n_heads = m->n_heads;
    int n_kv_heads = m->n_kv_heads;
    int head_dim = m->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int q_dim = n_heads * head_dim;
    int gqa_ratio = n_heads / n_kv_heads;
    int n_ff = m->n_ff;
    /* Head partition for attention */
    int h_per = n_heads / nt, h_extra = n_heads % nt;
    int h_start = tid * h_per + (tid < h_extra ? tid : h_extra);
    int h_count = h_per + (tid < h_extra ? 1 : 0);

    for (int l = 0; l < m->n_layers; l++) {
        transformer_layer *layer = &m->layers[l];

        /* Thread 0: RMSNorm (sequential, cheap) */
        if (tid == 0)
            tf_rmsnorm(m->xb, m->x, &layer->attn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        tf_spin_barrier(m, &local_sense, nt);  /* B1: xb ready */

        if (m->is_hybrid && layer->is_ssm) {
            /* SSM: thread 0 runs with pool disabled. Other threads wait. */
            if (tid == 0) {
                int saved_alive = m->pool_alive;
                m->pool_alive = 0;
                tf_ssm_deltanet_forward(m, l);
                m->pool_alive = saved_alive;
            }
            tf_spin_barrier(m, &local_sense, nt);  /* B2: SSM done */
        } else {
            /* --- Attention layer --- */

            /* QKV projection: all threads compute their row partition */
            if (m->is_hybrid) {
                /* Gated attention: Q+gate combined [2*q_dim] */
                int q2_dim = 2 * q_dim;
                tf_thread_matvec(m->xb2, &layer->attn_q, m->xb, q2_dim, tid, nt);
                tf_thread_matvec(m->k, &layer->attn_k, m->xb, kv_dim, tid, nt);
                tf_thread_matvec(m->v, &layer->attn_v, m->xb, kv_dim, tid, nt);
            } else {
                tf_thread_matvec(m->q, &layer->attn_q, m->xb, q_dim, tid, nt);
                tf_thread_matvec(m->k, &layer->attn_k, m->xb, kv_dim, tid, nt);
                if (layer->has_v_proj) {
                    tf_thread_matvec(m->v, &layer->attn_v, m->xb, kv_dim, tid, nt);
                } else {
                    /* Gemma4 SWA without attn_v: V is computed later from K */
                }
            }
            tf_spin_barrier(m, &local_sense, nt);  /* B2: Q/K/V ready */

            /* Thread 0: de-interleave (gated), QK-norm, RoPE, KV cache */
            if (tid == 0) {
                if (m->is_hybrid) {
                    for (int h = 0; h < n_heads; h++) {
                        memcpy(m->q + h * head_dim, m->xb2 + h * 2 * head_dim, head_dim * sizeof(float));
                        memcpy(m->ffn_buf1 + h * head_dim, m->xb2 + h * 2 * head_dim + head_dim, head_dim * sizeof(float));
                    }
                }
                if (layer->attn_q_norm.data)
                    tf_qk_norm(m->q, n_heads, head_dim, &layer->attn_q_norm, m->rms_norm_eps, m->matvec_tmp);
                if (layer->attn_k_norm.data)
                    tf_qk_norm(m->k, n_kv_heads, head_dim, &layer->attn_k_norm, m->rms_norm_eps, m->matvec_tmp);
                /* Bias additions */
                if (layer->attn_q_bias.data) {
                    if (layer->attn_q_bias.type == GGML_TYPE_F32) {
                        float *qb = (float *)layer->attn_q_bias.data;
                        for (int i = 0; i < q_dim; i++) m->q[i] += qb[i];
                    } else {
                        tf_dequant_row(&layer->attn_q_bias, 0, m->matvec_tmp);
                        for (int i = 0; i < q_dim; i++) m->q[i] += m->matvec_tmp[i];
                    }
                }
                if (layer->attn_k_bias.data) {
                    if (layer->attn_k_bias.type == GGML_TYPE_F32) {
                        float *kb = (float *)layer->attn_k_bias.data;
                        for (int i = 0; i < kv_dim; i++) m->k[i] += kb[i];
                    } else {
                        tf_dequant_row(&layer->attn_k_bias, 0, m->matvec_tmp);
                        for (int i = 0; i < kv_dim; i++) m->k[i] += m->matvec_tmp[i];
                    }
                }
                if (layer->attn_v_bias.data) {
                    if (layer->attn_v_bias.type == GGML_TYPE_F32) {
                        float *vb = (float *)layer->attn_v_bias.data;
                        for (int i = 0; i < kv_dim; i++) m->v[i] += vb[i];
                    } else {
                        tf_dequant_row(&layer->attn_v_bias, 0, m->matvec_tmp);
                        for (int i = 0; i < kv_dim; i++) m->v[i] += m->matvec_tmp[i];
                    }
                }
                tf_apply_rope(m, m->q, m->k, n_heads, n_kv_heads, head_dim, pos_t, pos_h, pos_w);
                memcpy(m->key_cache[l] + position * kv_dim, m->k, kv_dim * sizeof(float));
                memcpy(m->value_cache[l] + position * kv_dim, m->v, kv_dim * sizeof(float));
            }
            tf_spin_barrier(m, &local_sense, nt);  /* B3: Q/K ready for attention */

            /* Attention: each thread handles its head partition */
            {
                int seq_len = position + 1;
                float scale = 1.0f / sqrtf((float)head_dim);
                if (h_count > 0) {
                    tf_attn_task at = {m->q, m->att, m->xb2, m->key_cache[l], m->value_cache[l],
                                       h_start, h_start + h_count, head_dim, kv_dim, gqa_ratio,
                                       seq_len, m->max_seq_len, scale};
                    memset(m->xb2 + h_start * head_dim, 0, (size_t)h_count * head_dim * sizeof(float));
                    tf_attn_worker(&at);
                }
            }

            /* Sigmoid gate for hybrid gated attention */
            if (m->is_hybrid) {
                for (int i = h_start * head_dim; i < (h_start + h_count) * head_dim; i++) {
                    float g = m->ffn_buf1[i];
                    m->xb2[i] *= 1.0f / (1.0f + expf(-g));
                }
            }
            tf_spin_barrier(m, &local_sense, nt);  /* B4: xb2 ready */

            /* Output projection: all threads compute their row partition */
            tf_thread_matvec(m->xb, &layer->attn_output, m->xb2, n_embd, tid, nt);
            tf_spin_barrier(m, &local_sense, nt);  /* B5: xb ready */
        }

        /* Thread 0: residual + FFN norm (merged B5+B6: saves 1 barrier) */
        if (tid == 0) {
            tf_vadd(m->x, m->xb, n_embd);
            tf_rmsnorm(m->xb, m->x, &layer->ffn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        }
        tf_spin_barrier(m, &local_sense, nt);  /* B6: xb ready for FFN (merged) */

        if (!m->use_moe || !layer->ffn_gate_inp.data) {
            /* Dense SwiGLU FFN: gate+up matvec (parallel) */
            tf_thread_matvec(m->ffn_buf1, &layer->ffn_gate, m->xb, n_ff, tid, nt);
            tf_thread_matvec(m->ffn_buf2, &layer->ffn_up, m->xb, n_ff, tid, nt);

            /* SiLU×mul on this thread's row partition (no barrier needed) */
            {
                int rp = n_ff / nt, re = n_ff % nt;
                int rs = tid * rp + (tid < re ? tid : re);
                int rc = rp + (tid < re ? 1 : 0);
                for (int i = rs; i < rs + rc; i++) {
                    float g = m->ffn_buf1[i];
                    m->ffn_buf3[i] = g / (1.0f + expf(-g)) * m->ffn_buf2[i];
                }
            }
            tf_spin_barrier(m, &local_sense, nt);  /* B7: ffn_buf3 ready for down */

            /* Down projection */
            tf_thread_matvec(m->xb, &layer->ffn_down, m->ffn_buf3, n_embd, tid, nt);
            tf_spin_barrier(m, &local_sense, nt);  /* B8: xb ready */

            /* Thread 0: residual */
            if (tid == 0) tf_vadd(m->x, m->xb, n_embd);
        } else {
            /* MoE: thread 0 only (complex routing) */
            if (tid == 0) {
                int saved_alive = m->pool_alive;
                m->pool_alive = 0;  /* disable pool dispatch */
                const int n_expert = m->n_expert;
                const int n_ff_exp = m->n_ff_expert;
                tf_qmatvec_pool(m, m->ffn_buf1, &layer->ffn_gate_inp, m->xb, n_expert);
                tf_softmax(m->ffn_buf1, n_expert);
                /* ... MoE routing simplified: use single-thread for now ... */
                int best = 0;
                for (int e = 1; e < n_expert; e++)
                    if (m->ffn_buf1[e] > m->ffn_buf1[best]) best = e;
                float ew = m->ffn_buf1[best];
                tf_qmatvec_expert(m->ffn_buf2, &layer->ffn_up_exps, best, m->xb, n_ff_exp, m->matvec_tmp);
                tf_qmatvec_expert(m->ffn_buf3, &layer->ffn_gate_exps, best, m->xb, n_ff_exp, m->matvec_tmp);
                tf_silu_mul_avx2(m->ffn_buf3, m->ffn_buf3, m->ffn_buf2, n_ff_exp);
                tf_qmatvec_expert(m->q, &layer->ffn_down_exps, best, m->ffn_buf3, n_embd, m->matvec_tmp);
                for (int i = 0; i < n_embd; i++) m->x[i] += ew * m->q[i];
                m->pool_alive = saved_alive;
            }
            tf_spin_barrier(m, &local_sense, nt);  /* MoE done */
        }
        /* No end-of-layer barrier needed: only thread 0 writes m->x (vadd),
         * and only thread 0 reads it at the start of next layer (RMSNorm).
         * The next layer's B1 barrier ensures all threads wait for the norm. */
    }

    /* Final norm: thread 0 only */
    if (tid == 0) {
        tf_rmsnorm(m->x, m->x, &m->output_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
    }
    return NULL;
}

/* Persistent forward: dispatch tf_persistent_worker to ALL threads via pool,
 * then main thread runs worker 0 (master-worker pattern). Workers communicate
 * via spin barrier — the old pool dispatch mechanism is NOT used during forward. */
static float *tf_forward_persistent(transformer_model *m, int position, int pos_t, int pos_h, int pos_w) {
    int nt = m->n_threads;

    m->bar_count = 0;
    m->bar_sense = 0;
    __sync_synchronize();

    tf_persistent_ctx *ctxs = (tf_persistent_ctx *)alloca(nt * sizeof(tf_persistent_ctx));
    for (int t = 0; t < nt; t++)
        ctxs[t] = (tf_persistent_ctx){m, t, position, pos_t, pos_h, pos_w};

    /* Use the existing pool to dispatch the persistent worker to threads 1..nt-1.
     * Main thread (worker 0) runs inline. */
    tf_pool_dispatch(m, tf_persistent_worker, ctxs, sizeof(tf_persistent_ctx));

    return m->x;
}

/* ---- Forward pass ---- */

/* Internal: forward pass block loop with separate cache position and 3D RoPE positions.
 * Processes layers [layer_start, layer_end). If layer_end == n_layers, applies final norm. */
static float *tf_forward_blocks_range(transformer_model *m, int cache_pos, int pos_t, int pos_h, int pos_w,
                                       int layer_start, int layer_end);
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
    /* Gemma4: scale token embeddings by sqrt(n_embd), stash token_id for per-layer embd */
    if (model->is_gemma4) {
        float scale = model->embd_scale;
        for (int i = 0; i < model->n_embd; i++) model->x[i] *= scale;
        model->current_token_id = token_id;
    }
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
    /* Gemma4: final logit soft-capping */
    if (model->is_gemma4 && model->final_logit_softcapping > 0.0f) {
        tf_logit_softcap(model->logits, model->n_vocab, model->final_logit_softcapping);
    }
    return model->logits;
}

float *transformer_forward_logits(transformer_model *model, int32_t token_id, int position) {
    return transformer_forward_logits_pos(model, token_id, position, position, position, position);
}

static float *tf_forward_blocks(transformer_model *m, int position, int pos_t, int pos_h, int pos_w) {
    /* Gemma4 is NOT handled by the persistent worker (it has no SWA / per-layer KV /
     * proportional-RoPE / V-norm / softcap logic, and assumes a single global head_dim
     * & KV-head count -- wrong for 12B's MQA full-attn + GQA SWA mix). Use the
     * gemma4-aware block path; its matvecs still parallelize via tf_qmatvec_pool. */
    if (m->n_threads > 1 && m->pool_alive && !m->is_gemma4)
        return tf_forward_persistent(m, position, pos_t, pos_h, pos_w);
    return tf_forward_blocks_range(m, position, pos_t, pos_h, pos_w, 0, m->n_layers);
}

static float *tf_forward_blocks_range(transformer_model *m, int position, int pos_t, int pos_h, int pos_w,
                                       int layer_start, int layer_end) {
    int n_embd = m->n_embd;
    int n_heads = m->n_heads;
    int n_kv_heads = m->n_kv_heads;
    int head_dim = m->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int q_dim = n_heads * head_dim;  /* may differ from n_embd */
    int gqa_ratio = n_heads / n_kv_heads;

    if (layer_end > m->n_layers) layer_end = m->n_layers;
    if (layer_start < 0) layer_start = 0;

    /* Gemma4: precompute per-layer inputs (token embedding + model projection, combined).
     * Uses heap allocation instead of alloca (total_ple = 256*42 = 10752 floats ≈ 43KB × 3). */
    float *ple_combined = NULL; /* [n_layers * n_embd_per_layer] if Gemma4 */
    if (m->is_gemma4 && m->per_layer_token_embd.data && m->current_token_id >= 0) {
        int ple_dim = m->n_embd_per_layer;
        int total_ple = ple_dim * m->n_layers;
        ple_combined = (float *)malloc(total_ple * sizeof(float));
        float *tok_ple = (float *)malloc(total_ple * sizeof(float));
        float *proj_out = (float *)malloc(total_ple * sizeof(float));
        if (!ple_combined || !tok_ple || !proj_out) {
            fprintf(stderr, "transformer: Gemma4 PLE alloc failed (total_ple=%d)\n", total_ple);
            free(proj_out);
            free(tok_ple);
            free(ple_combined);
            return NULL;
        }

        /* 1. Look up per-layer token embedding: dequant row for this token */
        dequant_row(m->per_layer_token_embd.type,
                    (const uint8_t *)m->per_layer_token_embd.data +
                    (size_t)m->current_token_id * tf_row_bytes(m->per_layer_token_embd.type, total_ple),
                    tok_ple, total_ple);
        float ple_tok_scale = sqrtf((float)ple_dim);
        for (int i = 0; i < total_ple; i++) tok_ple[i] *= ple_tok_scale;

        /* 2. Project x through per_layer_model_proj: [10752, 2560] @ [2560] = [10752] */
        tf_qmatvec(proj_out, &m->per_layer_model_proj, m->x, total_ple, m->matvec_tmp);
        float proj_scale = 1.0f / sqrtf((float)n_embd);
        for (int i = 0; i < total_ple; i++) proj_out[i] *= proj_scale;

        /* 3. RMSNorm with per_layer_proj_norm (shared [256] weights, applied per-layer slice) */
        if (m->per_layer_proj_norm.data) {
            float norm_w[256];
            dequant_row(m->per_layer_proj_norm.type, m->per_layer_proj_norm.data, norm_w, ple_dim);
            for (int ll = 0; ll < m->n_layers; ll++) {
                float *slice = proj_out + ll * ple_dim;
                float ss = 0.0f;
                for (int i = 0; i < ple_dim; i++) ss += slice[i] * slice[i];
                ss = 1.0f / sqrtf(ss / ple_dim + m->rms_norm_eps);
                for (int i = 0; i < ple_dim; i++) slice[i] = slice[i] * ss * norm_w[i];
            }
        }

        /* 4. Add token embedding + projection, scale by 1/sqrt(2) */
        float input_scale = 1.0f / sqrtf(2.0f);
        for (int i = 0; i < total_ple; i++)
            ple_combined[i] = (proj_out[i] + tok_ple[i]) * input_scale;
        free(proj_out);
        free(tok_ple);
    }

    /* 2. Transformer blocks */
    for (int l = layer_start; l < layer_end; l++) {
        transformer_layer *layer = &m->layers[l];

        /* --- Attention --- */
        /* RMSNorm */
        TF_PROF_BEGIN("attn_norm", l, "rmsnorm", "FP32");
        tf_rmsnorm(m->xb, m->x, &layer->attn_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        TF_PROF_END("attn_norm", 5.0 * n_embd, 0);

        if (m->is_gemma4) {
            /* --- Gemma4 layer --- */
            int hd = layer->is_swa ? m->head_dim_swa : m->head_dim_full;
            int l_kvh = layer->n_kv_heads > 0 ? layer->n_kv_heads : n_kv_heads;
            int local_kv_dim = l_kvh * hd;
            int local_q_dim  = n_heads * hd;
            int local_gqa    = n_heads / l_kvh;
            float eps = m->rms_norm_eps;

            /* Q projection (always present) */
            tf_qmatvec_pool(m, m->q, &layer->attn_q, m->xb, local_q_dim);

            /* K/V projections (skip if sharing KV) */
            int kv_src = (layer->shared_kv_source >= 0) ? layer->shared_kv_source : l;
            if (layer->shared_kv_source < 0) {
                tf_qmatvec_pool(m, m->k, &layer->attn_k, m->xb, local_kv_dim);
                tf_qmatvec_pool(m, m->v, &layer->attn_v, m->xb, local_kv_dim);
            }

            /* Q norm (always) */
            tf_qk_norm(m->q, n_heads, hd, &layer->attn_q_norm, eps, m->matvec_tmp);

            /* K/V norm (only if we projected them) */
            if (layer->shared_kv_source < 0) {
                tf_qk_norm(m->k, l_kvh, hd, &layer->attn_k_norm, eps, m->matvec_tmp);
                /* Gemma4 SWA without attn_v: V = K (V is shared with K) */
                if (!layer->has_v_proj) {
                    memcpy(m->v, m->k, local_kv_dim * sizeof(float));
                }
                /* V norm — Gemma4 normalizes V too (raw RMSNorm, no weight) */
                if (layer->attn_v_norm.data) {
                    tf_qk_norm(m->v, l_kvh, hd, &layer->attn_v_norm, eps, m->matvec_tmp);
                } else {
                    /* V gets raw RMSNorm without learned weight */
                    for (int h = 0; h < l_kvh; h++) {
                        float *vh = m->v + h * hd;
                        float ss = 0.0f;
                        for (int i = 0; i < hd; i++) ss += vh[i] * vh[i];
                        ss = 1.0f / sqrtf(ss / hd + eps);
                        for (int i = 0; i < hd; i++) vh[i] *= ss;
                    }
                }
            }

            /* RoPE: use SWA or full-attention inv_freq table */
            {
                float *inv_freq = layer->is_swa ? m->rope_inv_freq_swa : m->rope_inv_freq;
                int half = hd / 2;
                /* Apply RoPE to Q */
                for (int h = 0; h < n_heads; h++) {
                    float *qh = m->q + h * hd;
                    for (int j = 0; j < half; j++) {
                        float freq = (float)position * inv_freq[j];
                        float cos_v = cosf(freq), sin_v = sinf(freq);
                        float r0 = qh[j], r1 = qh[j + half];
                        qh[j]        = r0 * cos_v - r1 * sin_v;
                        qh[j + half] = r0 * sin_v + r1 * cos_v;
                    }
                }
                /* Apply RoPE to K (only if we projected K) */
                if (layer->shared_kv_source < 0) {
                    for (int h = 0; h < l_kvh; h++) {
                        float *kh = m->k + h * hd;
                        for (int j = 0; j < half; j++) {
                            float freq = (float)position * inv_freq[j];
                            float cos_v = cosf(freq), sin_v = sinf(freq);
                            float r0 = kh[j], r1 = kh[j + half];
                            kh[j]        = r0 * cos_v - r1 * sin_v;
                            kh[j + half] = r0 * sin_v + r1 * cos_v;
                        }
                    }
                }
            }

            /* KV cache store */
            if (layer->shared_kv_source < 0) {
                if (layer->is_swa) {
                    int slot = position % m->swa_window_size;
                    tf_kv_store(m, l, (size_t)slot * local_kv_dim, m->k, local_kv_dim);
                    tf_kv_store_value(m, l, (size_t)slot * local_kv_dim, m->v, local_kv_dim);
                } else {
                    tf_kv_store(m, l, (size_t)position * local_kv_dim, m->k, local_kv_dim);
                    tf_kv_store_value(m, l, (size_t)position * local_kv_dim, m->v, local_kv_dim);
                }
            }

            /* Attention with scale=1.0 (QK norms handle scaling) */
            {
                float attn_scale = 1.0f;
                int seq_len;

                if (layer->is_swa) {
                    /* SWA: attend to window [max(0, pos-window+1), pos] via circular buffer */
                    int win = m->swa_window_size;
                    int start = (position >= win) ? (position - win + 1) : 0;
                    seq_len = position - start + 1;

                    /* Compute attention scores over the window */
                    memset(m->xb2, 0, local_q_dim * sizeof(float));
                    for (int h = 0; h < n_heads; h++) {
                        float *qh = m->q + h * hd;
                        float *att_h = m->att + h * seq_len;
                        int kv_h = h / local_gqa;

                        /* Compute QK scores */
                        for (int p = 0; p < seq_len; p++) {
                            int abs_pos = start + p;
                            int slot = abs_pos % win;
                            float score = 0.0f;
                            size_t kbase = (size_t)slot * local_kv_dim + (size_t)kv_h * hd;
                            for (int d = 0; d < hd; d++) score += qh[d] * tf_kv_load_key(m, kv_src, kbase + d);
                            att_h[p] = score * attn_scale;
                        }

                        /* Softmax */
                        float max_s = att_h[0];
                        for (int p = 1; p < seq_len; p++) if (att_h[p] > max_s) max_s = att_h[p];
                        float sum_e = 0.0f;
                        for (int p = 0; p < seq_len; p++) { att_h[p] = expf(att_h[p] - max_s); sum_e += att_h[p]; }
                        float inv_sum = 1.0f / sum_e;
                        for (int p = 0; p < seq_len; p++) att_h[p] *= inv_sum;

                        /* Weighted sum of values */
                        float *out_h = m->xb2 + h * hd;
                        for (int p = 0; p < seq_len; p++) {
                            int abs_pos = start + p;
                            int slot = abs_pos % win;
                            float w = att_h[p];
                            size_t vbase = (size_t)slot * local_kv_dim + (size_t)kv_h * hd;
                            for (int d = 0; d < hd; d++) out_h[d] += w * tf_kv_load_value(m, kv_src, vbase + d);
                        }
                    }
                } else {
                    /* Full attention */
                    seq_len = position + 1;
                    memset(m->xb2, 0, local_q_dim * sizeof(float));
                    for (int h = 0; h < n_heads; h++) {
                        float *qh = m->q + h * hd;
                        float *att_h = m->att + h * seq_len;
                        int kv_h = h / local_gqa;

                        for (int p = 0; p < seq_len; p++) {
                            float score = 0.0f;
                            size_t kbase = (size_t)p * local_kv_dim + (size_t)kv_h * hd;
                            for (int d = 0; d < hd; d++) score += qh[d] * tf_kv_load_key(m, kv_src, kbase + d);
                            att_h[p] = score * attn_scale;
                        }

                        float max_s = att_h[0];
                        for (int p = 1; p < seq_len; p++) if (att_h[p] > max_s) max_s = att_h[p];
                        float sum_e = 0.0f;
                        for (int p = 0; p < seq_len; p++) { att_h[p] = expf(att_h[p] - max_s); sum_e += att_h[p]; }
                        float inv_sum = 1.0f / sum_e;
                        for (int p = 0; p < seq_len; p++) att_h[p] *= inv_sum;

                        float *out_h = m->xb2 + h * hd;
                        for (int p = 0; p < seq_len; p++) {
                            float w = att_h[p];
                            size_t vbase = (size_t)p * local_kv_dim + (size_t)kv_h * hd;
                            for (int d = 0; d < hd; d++) out_h[d] += w * tf_kv_load_value(m, kv_src, vbase + d);
                        }
                    }
                }
            }

            /* Output projection */
            tf_qmatvec_pool(m, m->xb, &layer->attn_output, m->xb2, n_embd);

            /* Post-attention norm (before residual) */
            tf_rmsnorm(m->xb, m->xb, &layer->post_attention_norm, n_embd, eps, m->matvec_tmp);

            /* Residual */
            tf_vadd(m->x, m->xb, n_embd);

            /* --- FFN with GELU --- */
            tf_rmsnorm(m->xb, m->x, &layer->ffn_norm, n_embd, eps, m->matvec_tmp);

            tf_qmatvec_fused2_pool(m, m->ffn_buf1, &layer->ffn_gate,
                                    m->ffn_buf2, &layer->ffn_up, m->xb, m->n_ff);
            tf_gelu_mul(m->ffn_buf3, m->ffn_buf1, m->ffn_buf2, m->n_ff, m->ffn_gelu_fast);

            tf_qmatvec_pool(m, m->xb, &layer->ffn_down, m->ffn_buf3, n_embd);

            /* Post-FFN norm (before residual) */
            tf_rmsnorm(m->xb, m->xb, &layer->post_ffw_norm, n_embd, eps, m->matvec_tmp);

            /* Residual */
            tf_vadd(m->x, m->xb, n_embd);

            /* Per-layer embedding injection (BEFORE layer_output_scale) */
            if (ple_combined) {
                int ple_dim = m->n_embd_per_layer;
                float *ple = m->ple_buf;      /* [ple_dim] */
                float *proj = m->ple_proj_buf; /* [n_embd] */

                /* Take precomputed per-layer input slice for this layer */
                memcpy(ple, ple_combined + l * ple_dim, ple_dim * sizeof(float));

                /* inp_gate: hidden -> [ple_dim], GELU, element-wise multiply */
                tf_qmatvec_pool(m, proj, &layer->ple_inp_gate, m->x, ple_dim);
                for (int i = 0; i < ple_dim; i++) {
                    float g = proj[i];
                    float gelu_g = g * 0.5f * (1.0f + erff(g * 0.7071067811865476f));
                    ple[i] *= gelu_g;
                }

                /* Project back to n_embd via ple_proj */
                tf_qmatvec_pool(m, proj, &layer->ple_proj, ple, n_embd);

                /* Post-norm on projected output */
                if (layer->ple_post_norm.data) {
                    tf_rmsnorm(proj, proj, &layer->ple_post_norm, n_embd, eps, m->matvec_tmp);
                }

                /* Residual add */
                tf_vadd(m->x, proj, n_embd);
            }

            /* Layer output scaling (AFTER per-layer embedding) */
            if (layer->layer_output_scale.data) {
                float scale_val;
                dequant_row(layer->layer_output_scale.type, layer->layer_output_scale.data, &scale_val, 1);
                for (int i = 0; i < n_embd; i++) m->x[i] *= scale_val;
            }

            goto gemma4_layer_done;
        } else if (m->is_hybrid && layer->is_ssm) {
            /* --- SSM (Delta-Net) layer --- */
            tf_ssm_deltanet_forward(m, l);
        } else if (m->is_hybrid) {
            /* --- Gated attention layer (Qwen3.5) --- */
            /* Q+gate combined projection: attn_q outputs [2*q_dim] interleaved */
            int q2_dim = 2 * q_dim;
            TF_PROF_BEGIN("qkv_proj", l, "matvec", "FP32");
            tf_qmatvec_fused_qkv_pool(m, m->xb2, &layer->attn_q, q2_dim,
                                       m->k, &layer->attn_k, m->v, &layer->attn_v, kv_dim);
            TF_PROF_END("qkv_proj", 2.0 * (q2_dim + 2.0 * kv_dim) * n_embd, 0);

            /* De-interleave: [Q0,gate0, Q1,gate1, ...] → Q[q_dim], gate[q_dim] */
            for (int h = 0; h < n_heads; h++) {
                memcpy(m->q + h * head_dim, m->xb2 + h * 2 * head_dim, head_dim * sizeof(float));
                memcpy(m->ffn_buf1 + h * head_dim, m->xb2 + h * 2 * head_dim + head_dim, head_dim * sizeof(float));
            }

            /* QK-Norm */
            TF_PROF_BEGIN("qk_norm", l, "rmsnorm", "FP32");
            if (layer->attn_q_norm.data)
                tf_qk_norm(m->q, n_heads, head_dim, &layer->attn_q_norm, m->rms_norm_eps, m->matvec_tmp);
            if (layer->attn_k_norm.data)
                tf_qk_norm(m->k, n_kv_heads, head_dim, &layer->attn_k_norm, m->rms_norm_eps, m->matvec_tmp);
            TF_PROF_END("qk_norm", 5.0 * (n_heads + n_kv_heads) * head_dim, 0);

            /* RoPE */
            TF_PROF_BEGIN("rope", l, "rope", "FP32");
            tf_apply_rope(m, m->q, m->k, n_heads, n_kv_heads, head_dim, pos_t, pos_h, pos_w);
            TF_PROF_END("rope", 8.0 * (n_heads + n_kv_heads) * head_dim / 2, 0);

            /* KV cache */
            float *kc = m->key_cache[l] + position * kv_dim;
            float *vc = m->value_cache[l] + position * kv_dim;
            memcpy(kc, m->k, kv_dim * sizeof(float));
            memcpy(vc, m->v, kv_dim * sizeof(float));

            /* GQA attention (threaded if pool available) */
            TF_PROF_BEGIN("attention", l, "attention", "FP32");
            int seq_len = position + 1;
            float scale = 1.0f / sqrtf((float)head_dim);

            if (m->n_threads > 1 && n_heads >= m->n_threads && m->pool_alive) {
                int nt = m->n_threads;
                tf_attn_task *atasks = (tf_attn_task *)alloca(nt * sizeof(tf_attn_task));
                int heads_per = n_heads / nt, heads_extra = n_heads % nt, hoff = 0;
                for (int t = 0; t < nt; t++) {
                    int hcount = heads_per + (t < heads_extra ? 1 : 0);
                    atasks[t] = (tf_attn_task){m->q, m->att, m->xb2, m->key_cache[l], m->value_cache[l],
                                               hoff, hoff + hcount, head_dim, kv_dim, gqa_ratio, seq_len,
                                               m->max_seq_len, scale};
                    hoff += hcount;
                }
                tf_pool_dispatch(m, tf_attn_worker, atasks, sizeof(tf_attn_task));
            } else {
                /* Single-threaded fallback — dispatch through tf_attn_worker for AVX2 */
                tf_attn_task st = {m->q, m->att, m->xb2, m->key_cache[l], m->value_cache[l],
                                   0, n_heads, head_dim, kv_dim, gqa_ratio, seq_len,
                                   m->max_seq_len, scale};
                memset(m->xb2, 0, q_dim * sizeof(float));
                tf_attn_worker(&st);
            }
            TF_PROF_END("attention", 2.0 * n_heads * seq_len * head_dim * 2, 0);

            /* Apply sigmoid gate: xb2[i] *= sigmoid(gate[i]) */
#if defined(__AVX2__) && defined(__FMA__)
            {
                const __m256 one = _mm256_set1_ps(1.0f);
                int i = 0;
                for (; i + 7 < q_dim; i += 8) {
                    __m256 x2 = _mm256_loadu_ps(m->xb2 + i);
                    __m256 g = _mm256_loadu_ps(m->ffn_buf1 + i);
                    __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
                    __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, fast_exp_avx2(neg_g)));
                    _mm256_storeu_ps(m->xb2 + i, _mm256_mul_ps(x2, sig));
                }
                for (; i < q_dim; i++)
                    m->xb2[i] *= 1.0f / (1.0f + expf(-m->ffn_buf1[i]));
            }
#else
            for (int i = 0; i < q_dim; i++) {
                m->xb2[i] *= 1.0f / (1.0f + expf(-m->ffn_buf1[i]));
            }
#endif

            /* Output projection */
            TF_PROF_BEGIN("out_proj", l, "matvec", "FP32");
            tf_qmatvec_pool(m, m->xb, &layer->attn_output, m->xb2, n_embd);
            TF_PROF_END("out_proj", 2.0 * n_embd * q_dim, 0);
        } else {
            /* --- Standard attention (non-hybrid) --- */
            /* Q/K/V projections (fused: one pool dispatch for all three) */
            TF_PROF_BEGIN("qkv_proj", l, "matvec", "FP32");
            tf_qmatvec_fused_qkv_pool(m, m->q, &layer->attn_q, q_dim,
                                       m->k, &layer->attn_k, m->v, &layer->attn_v, kv_dim);
            /* Add Q/K/V biases if present (Qwen2.5-VL) */
            if (layer->attn_q_bias.data) {
                if (layer->attn_q_bias.type == GGML_TYPE_F32) {
                    float *qb = (float *)layer->attn_q_bias.data;
                    for (int i = 0; i < q_dim; i++) m->q[i] += qb[i];
                } else {
                    tf_dequant_row(&layer->attn_q_bias, 0, m->matvec_tmp);
                    for (int i = 0; i < q_dim; i++) m->q[i] += m->matvec_tmp[i];
                }
            }
            if (layer->attn_k_bias.data) {
                if (layer->attn_k_bias.type == GGML_TYPE_F32) {
                    float *kb = (float *)layer->attn_k_bias.data;
                    for (int i = 0; i < kv_dim; i++) m->k[i] += kb[i];
                } else {
                    tf_dequant_row(&layer->attn_k_bias, 0, m->matvec_tmp);
                    for (int i = 0; i < kv_dim; i++) m->k[i] += m->matvec_tmp[i];
                }
            }
            if (layer->attn_v_bias.data) {
                if (layer->attn_v_bias.type == GGML_TYPE_F32) {
                    float *vb = (float *)layer->attn_v_bias.data;
                    for (int i = 0; i < kv_dim; i++) m->v[i] += vb[i];
                } else {
                    tf_dequant_row(&layer->attn_v_bias, 0, m->matvec_tmp);
                    for (int i = 0; i < kv_dim; i++) m->v[i] += m->matvec_tmp[i];
                }
            }
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
                /* Single-threaded fallback — dispatch through tf_attn_worker for AVX2 */
                tf_attn_task st = {m->q, m->att, m->xb2, m->key_cache[l], m->value_cache[l],
                                   0, n_heads, head_dim, kv_dim, gqa_ratio, seq_len,
                                   m->max_seq_len, scale};
                memset(m->xb2, 0, q_dim * sizeof(float));
                tf_attn_worker(&st);
            }

            /* QK: heads*seq*hd, AV: heads*seq*hd */
            TF_PROF_END("attention", 2.0 * n_heads * seq_len * head_dim * 2, 0);

            /* Output projection */
            TF_PROF_BEGIN("out_proj", l, "matvec", "FP32");
            tf_qmatvec_pool(m, m->xb, &layer->attn_output, m->xb2, n_embd);
            TF_PROF_END("out_proj", 2.0 * n_embd * n_embd, 0);
        }

        /* Residual */
        tf_vadd(m->x, m->xb, n_embd);

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

            /* Select top-k experts using min-heap: O(n_expert) single pass */
            int *top_idx = (int *)alloca(n_top * sizeof(int));
            float *top_w = (float *)alloca(n_top * sizeof(float));
            int k = 0;
            for (int e = 0; e < n_expert; e++) {
                float w = m->ffn_buf1[e];
                if (k < n_top) {
                    top_idx[k] = e; top_w[k] = w; k++;
                    /* Sift up to maintain min-heap at top_w[0] */
                    for (int j = k-1; j > 0;) {
                        int p = (j-1)/2;
                        if (top_w[j] < top_w[p]) {
                            float tv = top_w[j]; top_w[j] = top_w[p]; top_w[p] = tv;
                            int ti = top_idx[j]; top_idx[j] = top_idx[p]; top_idx[p] = ti;
                            j = p;
                        } else break;
                    }
                } else if (w > top_w[0]) {
                    top_w[0] = w; top_idx[0] = e;
                    /* Sift down from root */
                    for (int j = 0;;) {
                        int s = j, l2 = 2*j+1, r2 = 2*j+2;
                        if (l2 < n_top && top_w[l2] < top_w[s]) s = l2;
                        if (r2 < n_top && top_w[r2] < top_w[s]) s = r2;
                        if (s == j) break;
                        float tv = top_w[j]; top_w[j] = top_w[s]; top_w[s] = tv;
                        int ti = top_idx[j]; top_idx[j] = top_idx[s]; top_idx[s] = ti;
                        j = s;
                    }
                }
            }
            float wsum = 0.0f;
            for (int i = 0; i < n_top; i++) wsum += top_w[i];
            if (wsum > 0.0f) {
                for (int i = 0; i < n_top; i++) top_w[i] /= wsum;
            }

            /* Aggregate selected experts */
            memset(m->xb2, 0, n_embd * sizeof(float));
            for (int ei = 0; ei < n_top; ei++) {
                int e = top_idx[ei];
                float ew = top_w[ei];

                TF_PROF_BEGIN("ffn_up_exp", l, "matvec", "FP32");
                tf_qmatvec_expert_pool(m, m->ffn_buf2, &layer->ffn_up_exps, e, m->xb, n_ff_exp);
                TF_PROF_END("ffn_up_exp", 2.0 * n_ff_exp * n_embd, 0);

                TF_PROF_BEGIN("ffn_gate_exp", l, "matvec", "FP32");
                tf_qmatvec_expert_pool(m, m->ffn_buf3, &layer->ffn_gate_exps, e, m->xb, n_ff_exp);
                TF_PROF_END("ffn_gate_exp", 2.0 * n_ff_exp * n_embd, 0);

                TF_PROF_BEGIN("silu_mul", l, "silu_mul", "FP32");
                tf_silu_mul_avx2(m->ffn_buf3, m->ffn_buf3, m->ffn_buf2, n_ff_exp);
                TF_PROF_END("silu_mul", 5.0 * n_ff_exp, 0);

                TF_PROF_BEGIN("ffn_down_exp", l, "matvec", "FP32");
                tf_qmatvec_expert_pool(m, m->q, &layer->ffn_down_exps, e, m->ffn_buf3, n_embd);
                TF_PROF_END("ffn_down_exp", 2.0 * n_embd * n_ff_exp, 0);

                /* Weighted accumulation: xb2 += ew * q */
#if defined(__AVX2__) && defined(__FMA__)
                {
                    __m256 vew = _mm256_set1_ps(ew);
                    int i = 0;
                    for (; i + 7 < n_embd; i += 8)
                        _mm256_storeu_ps(m->xb2 + i, _mm256_fmadd_ps(vew, _mm256_loadu_ps(m->q + i),
                                                                       _mm256_loadu_ps(m->xb2 + i)));
                    for (; i < n_embd; i++) m->xb2[i] += ew * m->q[i];
                }
#else
                for (int i = 0; i < n_embd; i++) m->xb2[i] += ew * m->q[i];
#endif
            }

            tf_vadd(m->x, m->xb2, n_embd);
        } else {
            /* Dense SwiGLU: down @ (silu(gate @ x) * (up @ x)) */
            TF_PROF_BEGIN("ffn_gate_up", l, "matvec", "FP32");
            tf_qmatvec_fused2_pool(m, m->ffn_buf1, &layer->ffn_gate,
                                    m->ffn_buf2, &layer->ffn_up, m->xb, m->n_ff);
            TF_PROF_END("ffn_gate_up", 2.0 * 2.0 * m->n_ff * n_embd, 0);

            TF_PROF_BEGIN("silu_mul", l, "silu_mul", "FP32");
            tf_silu_mul_avx2(m->ffn_buf3, m->ffn_buf1, m->ffn_buf2, m->n_ff);
            TF_PROF_END("silu_mul", 5.0 * m->n_ff, 0);

            TF_PROF_BEGIN("ffn_down", l, "matvec", "FP32");
            tf_qmatvec_pool(m, m->xb, &layer->ffn_down, m->ffn_buf3, n_embd);
            TF_PROF_END("ffn_down", 2.0 * n_embd * m->n_ff, 0);

            tf_vadd(m->x, m->xb, n_embd);
        }

        gemma4_layer_done:

        /* DeepStack injection: add deepstack slice after each early layer */
        if (m->ds_embd && l < m->n_deepstack && m->ds_embd_stride > n_embd) {
            const float *ds_slice = m->ds_embd + (1 + l) * n_embd;
            tf_vadd(m->x, ds_slice, n_embd);
        }

        if (m->debug_layers) {
            float ss = 0;
            for (int i = 0; i < n_embd; i++) ss += m->x[i] * m->x[i];
            fprintf(stderr, "  [L%02d ATT] norm=%.2f first=[%.4f, %.4f, %.4f, %.4f]\n",
                    l, sqrtf(ss), m->x[0], m->x[1], m->x[2], m->x[3]);
        }
    }

    /* Final RMSNorm (only if we processed through the last layer) */
    if (layer_end >= m->n_layers) {
        TF_PROF_BEGIN("final_norm", -1, "rmsnorm", "FP32");
        tf_rmsnorm(m->x, m->x, &m->output_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
        TF_PROF_END("final_norm", 5.0 * n_embd, 0);
    }

    free(ple_combined); /* NULL-safe: no-op if not Gemma4 */
    return m->x;
}

/* ---- Tensor-parallel API ---- */

void transformer_set_tp(transformer_model *model, int tp_rank, int tp_size,
                         void (*allreduce_fn)(float *buf, int count, void *ctx),
                         void *allreduce_ctx) {
    if (!model) return;
    model->tp_rank = tp_rank;
    model->tp_size = tp_size;
    model->tp_allreduce_fn = allreduce_fn;
    model->tp_allreduce_ctx = allreduce_ctx;
}

/* ---- Distributed memory management ---- */

void transformer_free_unused_kv(transformer_model *model, int layer_start, int layer_end) {
    if (!model || (!model->key_cache && !model->key_cache_raw)) return;
    for (int l = 0; l < model->n_layers; l++) {
        if (l < layer_start || l >= layer_end) {
            void *kptr = model->key_cache_raw ? model->key_cache_raw[l] : (void *)model->key_cache[l];
            void *vptr = model->value_cache_raw ? model->value_cache_raw[l] : (void *)model->value_cache[l];
            free(kptr);
            free(vptr);
            if (model->key_cache) model->key_cache[l] = NULL;
            if (model->value_cache) model->value_cache[l] = NULL;
            if (model->key_cache_raw) model->key_cache_raw[l] = NULL;
            if (model->value_cache_raw) model->value_cache_raw[l] = NULL;
        }
    }
    /* Also free SSM state for unused layers */
    if (model->conv_state) {
        for (int l = 0; l < model->n_layers; l++) {
            if (l < layer_start || l >= layer_end) {
                free(model->conv_state[l]);     model->conv_state[l] = NULL;
                free(model->recurrent_state[l]); model->recurrent_state[l] = NULL;
            }
        }
    }
}

void transformer_resize_kv_for_tp(transformer_model *model,
                                    int layer_start, int layer_end, int tp_kv_dim) {
    if (!model || !model->key_cache) return;
    for (int l = layer_start; l < layer_end && l < model->n_layers; l++) {
        void *old_k = model->key_cache_raw ? model->key_cache_raw[l] : (void *)model->key_cache[l];
        void *old_v = model->value_cache_raw ? model->value_cache_raw[l] : (void *)model->value_cache[l];
        if (!old_k) continue;  /* SSM layer, no KV cache */
        free(old_k);
        free(old_v);
        model->key_cache[l]   = (float *)calloc(model->max_seq_len * tp_kv_dim, sizeof(float));
        model->value_cache[l] = (float *)calloc(model->max_seq_len * tp_kv_dim, sizeof(float));
        if (model->key_cache_raw) model->key_cache_raw[l] = model->key_cache[l];
        if (model->value_cache_raw) model->value_cache_raw[l] = model->value_cache[l];
        model->kv_cache_type = 0;
    }
}

/* Column-parallel matvec: compute rows [row_start, row_end) of mat, output to dst[0..count).
 * Used for QKV, gate, up projections where each TP rank computes a subset of output rows. */
static void TF_MAYBE_UNUSED tf_qmatvec_row_slice(transformer_model *m, float *dst, const qtensor *mat,
                                                 const float *x, int row_start, int row_end) {
    int count = row_end - row_start;
    if (count <= 0) return;
    /* Create a virtual qtensor pointing to the slice */
    size_t row_bytes = tf_row_bytes(mat->type, mat->n_cols);
    qtensor slice = *mat;
    slice.data = (void *)((const uint8_t *)mat->data + (size_t)row_start * row_bytes);
    slice.n_rows = count;
    tf_qmatvec_pool(m, dst, &slice, x, count);
}

/* Row-parallel matvec: compute partial dot products using columns [col_start, col_end) of mat.
 * Each row's result is a partial sum; caller must allreduce across TP ranks.
 * x_local is the local slice of the input [col_end - col_start].
 * dst has n_rows elements (partial sums). */
static void tf_qmatvec_col_slice(float *dst, const qtensor *mat, const float *x_local,
                                  int n_rows, int col_start, int col_end, float *tmp) {
    int local_cols = col_end - col_start;
    for (int i = 0; i < n_rows; i++) {
        /* Dequant full row to tmp, then dot with local slice */
        tf_dequant_row(mat, i, tmp);
        float sum = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 15 < local_cols; j += 16) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + col_start + j),
                                    _mm256_loadu_ps(x_local + j), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + col_start + j + 8),
                                    _mm256_loadu_ps(x_local + j + 8), acc1);
        }
        for (; j + 7 < local_cols; j += 8)
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(tmp + col_start + j),
                                    _mm256_loadu_ps(x_local + j), acc0);
        acc0 = _mm256_add_ps(acc0, acc1);
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
        sum = _mm_cvtss_f32(s4);
        for (; j < local_cols; j++) sum += tmp[col_start + j] * x_local[j];
#else
        for (int j = 0; j < local_cols; j++)
            sum += tmp[col_start + j] * x_local[j];
#endif
        dst[i] = sum;
    }
}

/* Row-parallel matvec with pool threading: col_slice + allreduce. */
static void TF_MAYBE_UNUSED tf_qmatvec_col_slice_pool(transformer_model *m, float *dst, const qtensor *mat,
                                                      const float *x_local, int n_rows,
                                                      int col_start, int col_end) {
    /* Single-threaded col-slice (threading within rows is hard for col-slice) */
    tf_qmatvec_col_slice(dst, mat, x_local, n_rows, col_start, col_end, m->matvec_tmp);
    /* Allreduce partial sums across TP ranks */
    if (m->tp_allreduce_fn)
        m->tp_allreduce_fn(dst, n_rows, m->tp_allreduce_ctx);
}

/* ---- Pipeline-parallel API ---- */

void transformer_embed_token(transformer_model *model, int32_t token_id) {
    if (!model || !model->token_embd.data) return;
    if (token_id < 0 || token_id >= model->n_vocab) return;
    tf_dequant_row(&model->token_embd, token_id, model->x);
}

float *transformer_get_hidden(transformer_model *model) {
    return model ? model->x : NULL;
}

void transformer_set_hidden(transformer_model *model, const float *hidden) {
    if (model && hidden)
        memcpy(model->x, hidden, model->n_embd * sizeof(float));
}

float *transformer_compute_logits(transformer_model *model) {
    if (!model || !model->has_lm_head) return NULL;
    TF_PROF_BEGIN("lm_head", -1, "matvec", "FP32");
    tf_qmatvec_pool(model, model->logits, &model->output, model->x, model->n_vocab);
    TF_PROF_END("lm_head", 2.0 * model->n_vocab * model->n_embd, 0);
    return model->logits;
}

float *transformer_forward_partial(transformer_model *m, int cache_pos,
                                    int layer_start, int layer_end) {
    if (!m) return NULL;
    return tf_forward_blocks_range(m, cache_pos, cache_pos, cache_pos, cache_pos,
                                    layer_start, layer_end);
}

float *transformer_forward_embd_pos(transformer_model *model, const float *embd, int cache_pos, int pos_t, int pos_h, int pos_w) {
    if (!model || !embd) return NULL;
    if (cache_pos < 0 || cache_pos >= model->max_seq_len) {
        fprintf(stderr, "transformer_forward_embd_pos: cache_pos=%d out of range [0, %d)\n", cache_pos, model->max_seq_len);
        return NULL;
    }
    memcpy(model->x, embd, model->n_embd * sizeof(float));
    /* Gemma4: use padding token (ID=0) for per-layer embeddings on vision tokens.
     * llama.cpp gemma4-iswa.cpp: vision path uses row 0 of per_layer_token_embd. */
    if (model->is_gemma4) model->current_token_id = 0;
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
static void TF_MAYBE_UNUSED tf_gemm_f16_mt(float *Y, const qtensor *mat, const float *X,
                                           int n_rows, int N, int Y_stride, int X_stride,
                                           int n_threads) {
    if (mat->type != GGML_TYPE_F16) {
        /* Fallback: per-token matvec for non-F16 weights (AVX2 dot product) */
        int n_cols = mat->n_cols;
        float *row_buf = (float *)malloc(n_cols * sizeof(float));
        size_t rb = tf_row_bytes(mat->type, n_cols);
        for (int t = 0; t < N; t++) {
            const float *xt = X + (size_t)t * X_stride;
            for (int r = 0; r < n_rows; r++) {
                const void *row_data = (const uint8_t *)mat->data + (size_t)r * rb;
                dequant_row(mat->type, row_data, row_buf, n_cols);
#if defined(__AVX2__) && defined(__FMA__)
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                int j = 0;
                for (; j + 31 < n_cols; j += 32) {
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j),    _mm256_loadu_ps(xt+j),    a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j+8),  _mm256_loadu_ps(xt+j+8),  a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j+16), _mm256_loadu_ps(xt+j+16), a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j+24), _mm256_loadu_ps(xt+j+24), a3);
                }
                for (; j + 7 < n_cols; j += 8)
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row_buf+j), _mm256_loadu_ps(xt+j), a0);
                a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
                __m128 hi = _mm256_extractf128_ps(a0, 1), lo = _mm256_castps256_ps128(a0);
                __m128 s4 = _mm_add_ps(lo, hi);
                s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
                float sum = _mm_cvtss_f32(s4);
                for (; j < n_cols; j++) sum += row_buf[j] * xt[j];
#else
                float sum = 0.0f;
                for (int j = 0; j < n_cols; j++) sum += row_buf[j] * xt[j];
#endif
                Y[r * Y_stride + t] = sum;
            }
        }
        free(row_buf);
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
    gemm_f16_f32_tokmajor(t->Y + t->row_start,
                           t->W + (size_t)t->row_start * t->K,
                           t->X, nrows, t->K, t->N, t->Y_stride, t->X_stride);
    return NULL;
}

#if defined(__ARM_FEATURE_SVE)
/* Register-blocked token-major BF16xfp32 GEMM (MR=4 tokens x NR=6 weight-rows).
 * ~60x the matvec-based gemm_bf16_f32_tokmajor: shared k-vector loads feed 24 FMAs/
 * k-step, BF16 widened in-register (ld1uh->lsl#16, zero conversion FLOPs). No OMP —
 * the caller's pthread row-split parallelizes. Same signature as gemm_bf16_f32_tokmajor.
 * For N=1 (decode) the caller keeps the matvec path (this wastes 3/4 token slots). */
static inline svfloat32_t tf_bf16w(svbool_t pg, const uint16_t *p) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, p), 16));
}
static void tf_gemm_bf16_blocked(float *Y, const uint16_t *W, const float *X,
        int n_rows, int K, int N, int Ys, int Xs, int nt) {
    const int MR = 4, NR = 6; int vl = (int)svcntw();
    int MTn = (N + MR - 1) / MR, NTn = n_rows / NR;
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nt) schedule(static)
    #endif
    for (int n0 = 0; n0 < NTn; n0++) {
        const uint16_t *w0=W+(size_t)(n0*NR+0)*K,*w1=W+(size_t)(n0*NR+1)*K,*w2=W+(size_t)(n0*NR+2)*K;
        const uint16_t *w3=W+(size_t)(n0*NR+3)*K,*w4=W+(size_t)(n0*NR+4)*K,*w5=W+(size_t)(n0*NR+5)*K;
        for (int m0 = 0; m0 < MTn; m0++) {
            int t0=m0*MR,t1=t0+1,t2=t0+2,t3=t0+3;
            const float *x0=X+(size_t)(t0<N?t0:0)*Xs,*x1=X+(size_t)(t1<N?t1:0)*Xs;
            const float *x2=X+(size_t)(t2<N?t2:0)*Xs,*x3=X+(size_t)(t3<N?t3:0)*Xs;
            svfloat32_t a00=svdup_f32(0),a01=svdup_f32(0),a02=svdup_f32(0),a03=svdup_f32(0),a04=svdup_f32(0),a05=svdup_f32(0);
            svfloat32_t a10=svdup_f32(0),a11=svdup_f32(0),a12=svdup_f32(0),a13=svdup_f32(0),a14=svdup_f32(0),a15=svdup_f32(0);
            svfloat32_t a20=svdup_f32(0),a21=svdup_f32(0),a22=svdup_f32(0),a23=svdup_f32(0),a24=svdup_f32(0),a25=svdup_f32(0);
            svfloat32_t a30=svdup_f32(0),a31=svdup_f32(0),a32=svdup_f32(0),a33=svdup_f32(0),a34=svdup_f32(0),a35=svdup_f32(0);
            for (int k=0;k<K;k+=vl){ svbool_t pg=svwhilelt_b32(k,K);
                svfloat32_t v0=tf_bf16w(pg,w0+k),v1=tf_bf16w(pg,w1+k),v2=tf_bf16w(pg,w2+k),v3=tf_bf16w(pg,w3+k),v4=tf_bf16w(pg,w4+k),v5=tf_bf16w(pg,w5+k);
                svfloat32_t x=svld1_f32(pg,x0+k);
                a00=svmla_f32_m(pg,a00,x,v0);a01=svmla_f32_m(pg,a01,x,v1);a02=svmla_f32_m(pg,a02,x,v2);a03=svmla_f32_m(pg,a03,x,v3);a04=svmla_f32_m(pg,a04,x,v4);a05=svmla_f32_m(pg,a05,x,v5);
                x=svld1_f32(pg,x1+k);
                a10=svmla_f32_m(pg,a10,x,v0);a11=svmla_f32_m(pg,a11,x,v1);a12=svmla_f32_m(pg,a12,x,v2);a13=svmla_f32_m(pg,a13,x,v3);a14=svmla_f32_m(pg,a14,x,v4);a15=svmla_f32_m(pg,a15,x,v5);
                x=svld1_f32(pg,x2+k);
                a20=svmla_f32_m(pg,a20,x,v0);a21=svmla_f32_m(pg,a21,x,v1);a22=svmla_f32_m(pg,a22,x,v2);a23=svmla_f32_m(pg,a23,x,v3);a24=svmla_f32_m(pg,a24,x,v4);a25=svmla_f32_m(pg,a25,x,v5);
                x=svld1_f32(pg,x3+k);
                a30=svmla_f32_m(pg,a30,x,v0);a31=svmla_f32_m(pg,a31,x,v1);a32=svmla_f32_m(pg,a32,x,v2);a33=svmla_f32_m(pg,a33,x,v3);a34=svmla_f32_m(pg,a34,x,v4);a35=svmla_f32_m(pg,a35,x,v5);
            }
            svbool_t pt=svptrue_b32(); float *Yb=Y+(size_t)n0*NR;
            #define TF_BST(tok,A0,A1,A2,A3,A4,A5) if((tok)<N){ float*y=Yb+(size_t)(tok)*Ys; \
                y[0]=svaddv_f32(pt,A0);y[1]=svaddv_f32(pt,A1);y[2]=svaddv_f32(pt,A2);y[3]=svaddv_f32(pt,A3);y[4]=svaddv_f32(pt,A4);y[5]=svaddv_f32(pt,A5); }
            TF_BST(t0,a00,a01,a02,a03,a04,a05) TF_BST(t1,a10,a11,a12,a13,a14,a15)
            TF_BST(t2,a20,a21,a22,a23,a24,a25) TF_BST(t3,a30,a31,a32,a33,a34,a35)
            #undef TF_BST
        }
    }
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nt) schedule(static)
    #endif
    for (int row=NTn*NR; row<n_rows; row++){ const uint16_t *w=W+(size_t)row*K;
        for (int t=0;t<N;t++){ const float *x=X+(size_t)t*Xs; svfloat32_t a=svdup_f32(0);
            for (int k=0;k<K;k+=vl){ svbool_t pg=svwhilelt_b32(k,K); a=svmla_f32_m(pg,a,svld1_f32(pg,x+k),tf_bf16w(pg,w+k)); }
            Y[(size_t)t*Ys+row]=svaddv_f32(svptrue_b32(),a); } }
}
#define TF_HAVE_BF16_BLOCKED 1
#endif

static void *tf_gemm_bf16_tm_worker(void *arg) {
    tf_gemm_tm_task *t = (tf_gemm_tm_task *)arg;
    int nrows = t->row_end - t->row_start;
    gemm_bf16_f32_tokmajor(t->Y + t->row_start,
                            t->W + (size_t)t->row_start * t->K,
                            t->X, nrows, t->K, t->N, t->Y_stride, t->X_stride);
    return NULL;
}

/* Q8_0 token-major GEMM worker (for threaded dispatch) */
typedef struct {
    float *Y;
    const void *W;
    const float *X;
    int row_start, row_end;
    int K, N, Y_stride, X_stride;
} tf_gemm_q8_tm_task;

typedef struct {
    float *Y;
    const qtensor *mat;
    const float *X;
    int row_start, row_end;
    int K, N, Y_stride, X_stride;
} tf_gemm_qtensor_tm_task;

static void *tf_gemm_q8_tm_worker(void *arg) {
    tf_gemm_q8_tm_task *t = (tf_gemm_q8_tm_task *)arg;
    int nrows = t->row_end - t->row_start;
    int nb = t->K / 32;
    size_t row_bytes = (size_t)nb * sizeof(block_q8_0);
    gemm_q8_0_f32_tokmajor(t->Y + t->row_start,
                             (const uint8_t *)t->W + (size_t)t->row_start * row_bytes,
                             t->X, nrows, t->K, t->N, t->Y_stride, t->X_stride);
    return NULL;
}

static void *tf_gemm_qtensor_tm_worker(void *arg) {
    tf_gemm_qtensor_tm_task *t = (tf_gemm_qtensor_tm_task *)arg;
    float *row_buf = (float *)malloc((size_t)t->K * sizeof(float));
    if (!row_buf) return NULL;
    size_t rb = tf_row_bytes(t->mat->type, t->K);
    for (int r = t->row_start; r < t->row_end; r++) {
        const void *row_data = (const uint8_t *)t->mat->data + (size_t)r * rb;
        dequant_row(t->mat->type, row_data, row_buf, t->K);
        for (int tok = 0; tok < t->N; tok++) {
            const float *xt = t->X + (size_t)tok * t->X_stride;
            float sum = 0.0f;
            for (int k = 0; k < t->K; k++) sum += row_buf[k] * xt[k];
            t->Y[(size_t)tok * t->Y_stride + r] = sum;
        }
    }
    free(row_buf);
    return NULL;
}

static void *tf_gemm_q4_0_tm_worker(void *arg) {
    tf_gemm_qtensor_tm_task *t = (tf_gemm_qtensor_tm_task *)arg;
    const int nb = t->K / 32;
    const block_q4_0 *rows = (const block_q4_0 *)t->mat->data;
    for (int r = t->row_start; r < t->row_end; r++) {
        const block_q4_0 *row = rows + (size_t)r * nb;
        int tok = 0;
        for (; tok + 3 < t->N; tok += 4) {
            const float *x0 = t->X + (size_t)(tok + 0) * t->X_stride;
            const float *x1 = t->X + (size_t)(tok + 1) * t->X_stride;
            const float *x2 = t->X + (size_t)(tok + 2) * t->X_stride;
            const float *x3 = t->X + (size_t)(tok + 3) * t->X_stride;
            float s0, s1, s2, s3;
            tf_vec_dot_q4_0_f32_4x(row, x0, x1, x2, x3, t->K, &s0, &s1, &s2, &s3);
            t->Y[(size_t)(tok + 0) * t->Y_stride + r] = s0;
            t->Y[(size_t)(tok + 1) * t->Y_stride + r] = s1;
            t->Y[(size_t)(tok + 2) * t->Y_stride + r] = s2;
            t->Y[(size_t)(tok + 3) * t->Y_stride + r] = s3;
        }
        for (; tok < t->N; tok++) {
            const float *x = t->X + (size_t)tok * t->X_stride;
            float s = tf_vec_dot_q4_0_f32(row, x, t->K);
            t->Y[(size_t)tok * t->Y_stride + r] = s;
        }
    }
    return NULL;
}

typedef struct {
    float *Y;
    const qtensor *gate;
    const qtensor *up;
    const float *X;
    int row_start, row_end;
    int K, N, Y_stride, X_stride;
} tf_gemm_q4_pair_gelu_task;

static inline void tf_vec_dot_q4_0_pair_f32(const block_q4_0 *gate,
                                             const block_q4_0 *up,
                                             const float *x, int n_cols,
                                             float *sg, float *su) {
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t ag = svdup_f32(0.0f), au = svdup_f32(0.0f);
    svbool_t pg = svptrue_b32();
    int nb = n_cols / 32;
    if (nb > 0) {
        __builtin_prefetch(gate->qs, 0, 0);
        __builtin_prefetch(up->qs, 0, 0);
    }
    for (int b = 0; b < nb; b++) {
        const int base = b * 32;
        svfloat32_t xlo = svld1(pg, x + base);
        svfloat32_t xhi = svld1(pg, x + base + 16);

        svuint32_t qg = svld1ub_u32(pg, gate[b].qs);
        svint32_t qglo = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, qg, 0x0f)), 8);
        svint32_t qghi = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, qg, 4)), 8);
        float dg = ggml_fp16_to_fp32(gate[b].d);
        ag = svmla_x(pg, ag, svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qglo), dg), xlo);
        ag = svmla_x(pg, ag, svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qghi), dg), xhi);

        svuint32_t qu = svld1ub_u32(pg, up[b].qs);
        svint32_t qulo = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, qu, 0x0f)), 8);
        svint32_t quhi = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, qu, 4)), 8);
        float du = ggml_fp16_to_fp32(up[b].d);
        au = svmla_x(pg, au, svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qulo), du), xlo);
        au = svmla_x(pg, au, svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, quhi), du), xhi);
        if (b + 1 < nb) {
            __builtin_prefetch(gate[b+1].qs, 0, 0);
            __builtin_prefetch(up[b+1].qs, 0, 0);
        }
    }
    *sg = svaddv_f32(pg, ag);
    *su = svaddv_f32(pg, au);
#else
    float gsum = 0.0f, usum = 0.0f;
    int nb = n_cols / 32;
    for (int b = 0; b < nb; b++) {
        float dg = ggml_fp16_to_fp32(gate[b].d);
        float du = ggml_fp16_to_fp32(up[b].d);
        const int base = b * 32;
        for (int j = 0; j < 16; j++) {
            uint8_t qg = gate[b].qs[j];
            uint8_t qu = up[b].qs[j];
            gsum += ((float)((int)(qg & 0x0f) - 8) * dg) * x[base + j];
            gsum += ((float)((int)(qg >> 4) - 8) * dg) * x[base + j + 16];
            usum += ((float)((int)(qu & 0x0f) - 8) * du) * x[base + j];
            usum += ((float)((int)(qu >> 4) - 8) * du) * x[base + j + 16];
        }
    }
    *sg = gsum;
    *su = usum;
#endif
}

static inline void tf_vec_dot_q4_0_pair_f32_4x(const block_q4_0 *gate,
                                                const block_q4_0 *up,
                                                const float *x0, const float *x1,
                                                const float *x2, const float *x3,
                                                int n_cols,
                                                float *g0, float *g1,
                                                float *g2, float *g3,
                                                float *u0, float *u1,
                                                float *u2, float *u3) {
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t ag0 = svdup_f32(0.0f), ag1 = svdup_f32(0.0f);
    svfloat32_t ag2 = svdup_f32(0.0f), ag3 = svdup_f32(0.0f);
    svfloat32_t au0 = svdup_f32(0.0f), au1 = svdup_f32(0.0f);
    svfloat32_t au2 = svdup_f32(0.0f), au3 = svdup_f32(0.0f);
    svbool_t pg = svptrue_b32();
    int nb = n_cols / 32;
    if (nb > 0) {
        __builtin_prefetch(gate->qs, 0, 0);
        __builtin_prefetch(up->qs, 0, 0);
    }
    for (int b = 0; b < nb; b++) {
        const int base = b * 32;
        svfloat32_t x0lo = svld1(pg, x0 + base);
        svfloat32_t x0hi = svld1(pg, x0 + base + 16);
        svfloat32_t x1lo = svld1(pg, x1 + base);
        svfloat32_t x1hi = svld1(pg, x1 + base + 16);
        svfloat32_t x2lo = svld1(pg, x2 + base);
        svfloat32_t x2hi = svld1(pg, x2 + base + 16);
        svfloat32_t x3lo = svld1(pg, x3 + base);
        svfloat32_t x3hi = svld1(pg, x3 + base + 16);

        svuint32_t qg = svld1ub_u32(pg, gate[b].qs);
        svint32_t qglo = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, qg, 0x0f)), 8);
        svint32_t qghi = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, qg, 4)), 8);
        float dg = ggml_fp16_to_fp32(gate[b].d);
        svfloat32_t wglo = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qglo), dg);
        svfloat32_t wghi = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qghi), dg);
        ag0 = svmla_x(pg, ag0, wglo, x0lo); ag0 = svmla_x(pg, ag0, wghi, x0hi);
        ag1 = svmla_x(pg, ag1, wglo, x1lo); ag1 = svmla_x(pg, ag1, wghi, x1hi);
        ag2 = svmla_x(pg, ag2, wglo, x2lo); ag2 = svmla_x(pg, ag2, wghi, x2hi);
        ag3 = svmla_x(pg, ag3, wglo, x3lo); ag3 = svmla_x(pg, ag3, wghi, x3hi);

        svuint32_t qu = svld1ub_u32(pg, up[b].qs);
        svint32_t qulo = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, qu, 0x0f)), 8);
        svint32_t quhi = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, qu, 4)), 8);
        float du = ggml_fp16_to_fp32(up[b].d);
        svfloat32_t wulo = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qulo), du);
        svfloat32_t wuhi = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, quhi), du);
        au0 = svmla_x(pg, au0, wulo, x0lo); au0 = svmla_x(pg, au0, wuhi, x0hi);
        au1 = svmla_x(pg, au1, wulo, x1lo); au1 = svmla_x(pg, au1, wuhi, x1hi);
        au2 = svmla_x(pg, au2, wulo, x2lo); au2 = svmla_x(pg, au2, wuhi, x2hi);
        au3 = svmla_x(pg, au3, wulo, x3lo); au3 = svmla_x(pg, au3, wuhi, x3hi);
        if (b + 1 < nb) {
            __builtin_prefetch(gate[b+1].qs, 0, 0);
            __builtin_prefetch(up[b+1].qs, 0, 0);
        }
    }
    *g0 = svaddv_f32(pg, ag0); *g1 = svaddv_f32(pg, ag1);
    *g2 = svaddv_f32(pg, ag2); *g3 = svaddv_f32(pg, ag3);
    *u0 = svaddv_f32(pg, au0); *u1 = svaddv_f32(pg, au1);
    *u2 = svaddv_f32(pg, au2); *u3 = svaddv_f32(pg, au3);
#else
    tf_vec_dot_q4_0_pair_f32(gate, up, x0, n_cols, g0, u0);
    tf_vec_dot_q4_0_pair_f32(gate, up, x1, n_cols, g1, u1);
    tf_vec_dot_q4_0_pair_f32(gate, up, x2, n_cols, g2, u2);
    tf_vec_dot_q4_0_pair_f32(gate, up, x3, n_cols, g3, u3);
#endif
}

static inline void tf_vec_dot_q4_0_pair_f32_2r4x(const block_q4_0 *gate0,
                                                  const block_q4_0 *up0,
                                                  const block_q4_0 *gate1,
                                                  const block_q4_0 *up1,
                                                  const float *x0, const float *x1,
                                                  const float *x2, const float *x3,
                                                  int n_cols,
                                                  float *g00, float *g01,
                                                  float *g02, float *g03,
                                                  float *u00, float *u01,
                                                  float *u02, float *u03,
                                                  float *g10, float *g11,
                                                  float *g12, float *g13,
                                                  float *u10, float *u11,
                                                  float *u12, float *u13) {
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t ag00 = svdup_f32(0.0f), ag01 = svdup_f32(0.0f);
    svfloat32_t ag02 = svdup_f32(0.0f), ag03 = svdup_f32(0.0f);
    svfloat32_t au00 = svdup_f32(0.0f), au01 = svdup_f32(0.0f);
    svfloat32_t au02 = svdup_f32(0.0f), au03 = svdup_f32(0.0f);
    svfloat32_t ag10 = svdup_f32(0.0f), ag11 = svdup_f32(0.0f);
    svfloat32_t ag12 = svdup_f32(0.0f), ag13 = svdup_f32(0.0f);
    svfloat32_t au10 = svdup_f32(0.0f), au11 = svdup_f32(0.0f);
    svfloat32_t au12 = svdup_f32(0.0f), au13 = svdup_f32(0.0f);
    svbool_t pg = svptrue_b32();
    int nb = n_cols / 32;
    if (nb > 0) {
        __builtin_prefetch(gate0->qs, 0, 0);
        __builtin_prefetch(up0->qs, 0, 0);
        __builtin_prefetch(gate1->qs, 0, 0);
        __builtin_prefetch(up1->qs, 0, 0);
    }
    for (int b = 0; b < nb; b++) {
        const int base = b * 32;
        svfloat32_t x0lo = svld1(pg, x0 + base), x0hi = svld1(pg, x0 + base + 16);
        svfloat32_t x1lo = svld1(pg, x1 + base), x1hi = svld1(pg, x1 + base + 16);
        svfloat32_t x2lo = svld1(pg, x2 + base), x2hi = svld1(pg, x2 + base + 16);
        svfloat32_t x3lo = svld1(pg, x3 + base), x3hi = svld1(pg, x3 + base + 16);

#define TF_Q4_ACC_ROW(GROW, UROW, AG0, AG1, AG2, AG3, AU0, AU1, AU2, AU3) do { \
        svuint32_t qg = svld1ub_u32(pg, (GROW)[b].qs); \
        svint32_t qglo = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, qg, 0x0f)), 8); \
        svint32_t qghi = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, qg, 4)), 8); \
        float dg = ggml_fp16_to_fp32((GROW)[b].d); \
        svfloat32_t wglo = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qglo), dg); \
        svfloat32_t wghi = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qghi), dg); \
        AG0 = svmla_x(pg, AG0, wglo, x0lo); AG0 = svmla_x(pg, AG0, wghi, x0hi); \
        AG1 = svmla_x(pg, AG1, wglo, x1lo); AG1 = svmla_x(pg, AG1, wghi, x1hi); \
        AG2 = svmla_x(pg, AG2, wglo, x2lo); AG2 = svmla_x(pg, AG2, wghi, x2hi); \
        AG3 = svmla_x(pg, AG3, wglo, x3lo); AG3 = svmla_x(pg, AG3, wghi, x3hi); \
        svuint32_t qu = svld1ub_u32(pg, (UROW)[b].qs); \
        svint32_t qulo = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, qu, 0x0f)), 8); \
        svint32_t quhi = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, qu, 4)), 8); \
        float du = ggml_fp16_to_fp32((UROW)[b].d); \
        svfloat32_t wulo = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qulo), du); \
        svfloat32_t wuhi = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, quhi), du); \
        AU0 = svmla_x(pg, AU0, wulo, x0lo); AU0 = svmla_x(pg, AU0, wuhi, x0hi); \
        AU1 = svmla_x(pg, AU1, wulo, x1lo); AU1 = svmla_x(pg, AU1, wuhi, x1hi); \
        AU2 = svmla_x(pg, AU2, wulo, x2lo); AU2 = svmla_x(pg, AU2, wuhi, x2hi); \
        AU3 = svmla_x(pg, AU3, wulo, x3lo); AU3 = svmla_x(pg, AU3, wuhi, x3hi); \
    } while (0)
        TF_Q4_ACC_ROW(gate0, up0, ag00, ag01, ag02, ag03, au00, au01, au02, au03);
        TF_Q4_ACC_ROW(gate1, up1, ag10, ag11, ag12, ag13, au10, au11, au12, au13);
#undef TF_Q4_ACC_ROW
        if (b + 1 < nb) {
            __builtin_prefetch(gate0[b+1].qs, 0, 0);
            __builtin_prefetch(up0[b+1].qs, 0, 0);
            __builtin_prefetch(gate1[b+1].qs, 0, 0);
            __builtin_prefetch(up1[b+1].qs, 0, 0);
        }
    }
    *g00 = svaddv_f32(pg, ag00); *g01 = svaddv_f32(pg, ag01);
    *g02 = svaddv_f32(pg, ag02); *g03 = svaddv_f32(pg, ag03);
    *u00 = svaddv_f32(pg, au00); *u01 = svaddv_f32(pg, au01);
    *u02 = svaddv_f32(pg, au02); *u03 = svaddv_f32(pg, au03);
    *g10 = svaddv_f32(pg, ag10); *g11 = svaddv_f32(pg, ag11);
    *g12 = svaddv_f32(pg, ag12); *g13 = svaddv_f32(pg, ag13);
    *u10 = svaddv_f32(pg, au10); *u11 = svaddv_f32(pg, au11);
    *u12 = svaddv_f32(pg, au12); *u13 = svaddv_f32(pg, au13);
#else
    tf_vec_dot_q4_0_pair_f32_4x(gate0, up0, x0, x1, x2, x3, n_cols,
                                g00, g01, g02, g03, u00, u01, u02, u03);
    tf_vec_dot_q4_0_pair_f32_4x(gate1, up1, x0, x1, x2, x3, n_cols,
                                g10, g11, g12, g13, u10, u11, u12, u13);
#endif
}

static inline void tf_vec_dot_q4_0_pair_f32_8x(const block_q4_0 *gate,
                                                const block_q4_0 *up,
                                                const float *x0, const float *x1,
                                                const float *x2, const float *x3,
                                                const float *x4, const float *x5,
                                                const float *x6, const float *x7,
                                                int n_cols,
                                                float *g0, float *g1,
                                                float *g2, float *g3,
                                                float *g4, float *g5,
                                                float *g6, float *g7,
                                                float *u0, float *u1,
                                                float *u2, float *u3,
                                                float *u4, float *u5,
                                                float *u6, float *u7) {
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t ag0 = svdup_f32(0.0f), ag1 = svdup_f32(0.0f);
    svfloat32_t ag2 = svdup_f32(0.0f), ag3 = svdup_f32(0.0f);
    svfloat32_t ag4 = svdup_f32(0.0f), ag5 = svdup_f32(0.0f);
    svfloat32_t ag6 = svdup_f32(0.0f), ag7 = svdup_f32(0.0f);
    svfloat32_t au0 = svdup_f32(0.0f), au1 = svdup_f32(0.0f);
    svfloat32_t au2 = svdup_f32(0.0f), au3 = svdup_f32(0.0f);
    svfloat32_t au4 = svdup_f32(0.0f), au5 = svdup_f32(0.0f);
    svfloat32_t au6 = svdup_f32(0.0f), au7 = svdup_f32(0.0f);
    svbool_t pg = svptrue_b32();
    int nb = n_cols / 32;
    if (nb > 0) {
        __builtin_prefetch(gate->qs, 0, 0);
        __builtin_prefetch(up->qs, 0, 0);
    }
    for (int b = 0; b < nb; b++) {
        const int base = b * 32;
        svuint32_t qg = svld1ub_u32(pg, gate[b].qs);
        svint32_t qglo = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, qg, 0x0f)), 8);
        svint32_t qghi = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, qg, 4)), 8);
        float dg = ggml_fp16_to_fp32(gate[b].d);
        svfloat32_t wglo = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qglo), dg);
        svfloat32_t wghi = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qghi), dg);
        svuint32_t qu = svld1ub_u32(pg, up[b].qs);
        svint32_t qulo = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, qu, 0x0f)), 8);
        svint32_t quhi = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, qu, 4)), 8);
        float du = ggml_fp16_to_fp32(up[b].d);
        svfloat32_t wulo = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qulo), du);
        svfloat32_t wuhi = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, quhi), du);

#define TF_Q4_ACC_TOKEN(IDX, XPTR, AG, AU) do { \
        svfloat32_t xlo = svld1(pg, (XPTR) + base); \
        svfloat32_t xhi = svld1(pg, (XPTR) + base + 16); \
        AG = svmla_x(pg, AG, wglo, xlo); \
        AG = svmla_x(pg, AG, wghi, xhi); \
        AU = svmla_x(pg, AU, wulo, xlo); \
        AU = svmla_x(pg, AU, wuhi, xhi); \
    } while (0)
        TF_Q4_ACC_TOKEN(0, x0, ag0, au0);
        TF_Q4_ACC_TOKEN(1, x1, ag1, au1);
        TF_Q4_ACC_TOKEN(2, x2, ag2, au2);
        TF_Q4_ACC_TOKEN(3, x3, ag3, au3);
        TF_Q4_ACC_TOKEN(4, x4, ag4, au4);
        TF_Q4_ACC_TOKEN(5, x5, ag5, au5);
        TF_Q4_ACC_TOKEN(6, x6, ag6, au6);
        TF_Q4_ACC_TOKEN(7, x7, ag7, au7);
#undef TF_Q4_ACC_TOKEN
        if (b + 1 < nb) {
            __builtin_prefetch(gate[b+1].qs, 0, 0);
            __builtin_prefetch(up[b+1].qs, 0, 0);
        }
    }
    *g0 = svaddv_f32(pg, ag0); *g1 = svaddv_f32(pg, ag1);
    *g2 = svaddv_f32(pg, ag2); *g3 = svaddv_f32(pg, ag3);
    *g4 = svaddv_f32(pg, ag4); *g5 = svaddv_f32(pg, ag5);
    *g6 = svaddv_f32(pg, ag6); *g7 = svaddv_f32(pg, ag7);
    *u0 = svaddv_f32(pg, au0); *u1 = svaddv_f32(pg, au1);
    *u2 = svaddv_f32(pg, au2); *u3 = svaddv_f32(pg, au3);
    *u4 = svaddv_f32(pg, au4); *u5 = svaddv_f32(pg, au5);
    *u6 = svaddv_f32(pg, au6); *u7 = svaddv_f32(pg, au7);
#else
    tf_vec_dot_q4_0_pair_f32_4x(gate, up, x0, x1, x2, x3, n_cols,
                                g0, g1, g2, g3, u0, u1, u2, u3);
    tf_vec_dot_q4_0_pair_f32_4x(gate, up, x4, x5, x6, x7, n_cols,
                                g4, g5, g6, g7, u4, u5, u6, u7);
#endif
}

static void *tf_gemm_q4_0_pair_gelu_worker(void *arg) {
    tf_gemm_q4_pair_gelu_task *t = (tf_gemm_q4_pair_gelu_task *)arg;
    const int nb = t->K / 32;
    const block_q4_0 *gate_rows = (const block_q4_0 *)t->gate->data;
    const block_q4_0 *up_rows = (const block_q4_0 *)t->up->data;
    const char *use_2r_env = getenv("TF_FFN_2R4X");
    int use_2r4x = use_2r_env && atoi(use_2r_env) != 0;
    const char *use_8x_env = getenv("TF_FFN_8X");
    int use_8x = !use_8x_env || atoi(use_8x_env) != 0;
    int r = t->row_start;
    for (; use_2r4x && r + 1 < t->row_end; r += 2) {
        const block_q4_0 *g0row = gate_rows + (size_t)r * nb;
        const block_q4_0 *u0row = up_rows + (size_t)r * nb;
        const block_q4_0 *g1row = g0row + nb;
        const block_q4_0 *u1row = u0row + nb;
        int tok = 0;
        for (; tok + 3 < t->N; tok += 4) {
            const float *x0 = t->X + (size_t)(tok + 0) * t->X_stride;
            const float *x1 = t->X + (size_t)(tok + 1) * t->X_stride;
            const float *x2 = t->X + (size_t)(tok + 2) * t->X_stride;
            const float *x3 = t->X + (size_t)(tok + 3) * t->X_stride;
            float g00, g01, g02, g03, u00, u01, u02, u03;
            float g10, g11, g12, g13, u10, u11, u12, u13;
            tf_vec_dot_q4_0_pair_f32_2r4x(g0row, u0row, g1row, u1row,
                                          x0, x1, x2, x3, t->K,
                                          &g00, &g01, &g02, &g03, &u00, &u01, &u02, &u03,
                                          &g10, &g11, &g12, &g13, &u10, &u11, &u12, &u13);
#if defined(__ARM_FEATURE_SVE)
            tf_gelu_mul_fast4_sve(t->Y, (size_t)t->Y_stride, r, tok,
                                  g00, g01, g02, g03, u00, u01, u02, u03);
            tf_gelu_mul_fast4_sve(t->Y, (size_t)t->Y_stride, r + 1, tok,
                                  g10, g11, g12, g13, u10, u11, u12, u13);
#else
            t->Y[(size_t)(tok + 0) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g00, u00);
            t->Y[(size_t)(tok + 1) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g01, u01);
            t->Y[(size_t)(tok + 2) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g02, u02);
            t->Y[(size_t)(tok + 3) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g03, u03);
            t->Y[(size_t)(tok + 0) * t->Y_stride + r + 1] = tf_gelu_mul_fast_scalar(g10, u10);
            t->Y[(size_t)(tok + 1) * t->Y_stride + r + 1] = tf_gelu_mul_fast_scalar(g11, u11);
            t->Y[(size_t)(tok + 2) * t->Y_stride + r + 1] = tf_gelu_mul_fast_scalar(g12, u12);
            t->Y[(size_t)(tok + 3) * t->Y_stride + r + 1] = tf_gelu_mul_fast_scalar(g13, u13);
#endif
        }
        for (; tok < t->N; tok++) {
            const float *x = t->X + (size_t)tok * t->X_stride;
            float gate, up;
            tf_vec_dot_q4_0_pair_f32(g0row, u0row, x, t->K, &gate, &up);
            t->Y[(size_t)tok * t->Y_stride + r] = tf_gelu_mul_fast_scalar(gate, up);
            tf_vec_dot_q4_0_pair_f32(g1row, u1row, x, t->K, &gate, &up);
            t->Y[(size_t)tok * t->Y_stride + r + 1] = tf_gelu_mul_fast_scalar(gate, up);
        }
    }
    for (; r < t->row_end; r++) {
        const block_q4_0 *grow = gate_rows + (size_t)r * nb;
        const block_q4_0 *urow = up_rows + (size_t)r * nb;
        int tok = 0;
        if (use_8x) {
            for (; tok + 7 < t->N; tok += 8) {
                const float *x0 = t->X + (size_t)(tok + 0) * t->X_stride;
                const float *x1 = t->X + (size_t)(tok + 1) * t->X_stride;
                const float *x2 = t->X + (size_t)(tok + 2) * t->X_stride;
                const float *x3 = t->X + (size_t)(tok + 3) * t->X_stride;
                const float *x4 = t->X + (size_t)(tok + 4) * t->X_stride;
                const float *x5 = t->X + (size_t)(tok + 5) * t->X_stride;
                const float *x6 = t->X + (size_t)(tok + 6) * t->X_stride;
                const float *x7 = t->X + (size_t)(tok + 7) * t->X_stride;
                float g0, g1, g2, g3, g4, g5, g6, g7;
                float u0, u1, u2, u3, u4, u5, u6, u7;
                tf_vec_dot_q4_0_pair_f32_8x(grow, urow, x0, x1, x2, x3,
                                            x4, x5, x6, x7, t->K,
                                            &g0, &g1, &g2, &g3, &g4, &g5, &g6, &g7,
                                            &u0, &u1, &u2, &u3, &u4, &u5, &u6, &u7);
#if defined(__ARM_FEATURE_SVE)
                tf_gelu_mul_fast4_sve(t->Y, (size_t)t->Y_stride, r, tok,
                                      g0, g1, g2, g3, u0, u1, u2, u3);
                tf_gelu_mul_fast4_sve(t->Y, (size_t)t->Y_stride, r, tok + 4,
                                      g4, g5, g6, g7, u4, u5, u6, u7);
#else
                t->Y[(size_t)(tok + 0) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g0, u0);
                t->Y[(size_t)(tok + 1) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g1, u1);
                t->Y[(size_t)(tok + 2) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g2, u2);
                t->Y[(size_t)(tok + 3) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g3, u3);
                t->Y[(size_t)(tok + 4) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g4, u4);
                t->Y[(size_t)(tok + 5) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g5, u5);
                t->Y[(size_t)(tok + 6) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g6, u6);
                t->Y[(size_t)(tok + 7) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g7, u7);
#endif
            }
        }
        for (; tok + 3 < t->N; tok += 4) {
            const float *x0 = t->X + (size_t)(tok + 0) * t->X_stride;
            const float *x1 = t->X + (size_t)(tok + 1) * t->X_stride;
            const float *x2 = t->X + (size_t)(tok + 2) * t->X_stride;
            const float *x3 = t->X + (size_t)(tok + 3) * t->X_stride;
            float g0, g1, g2, g3, u0, u1, u2, u3;
            tf_vec_dot_q4_0_pair_f32_4x(grow, urow, x0, x1, x2, x3, t->K,
                                        &g0, &g1, &g2, &g3, &u0, &u1, &u2, &u3);
#if defined(__ARM_FEATURE_SVE)
            tf_gelu_mul_fast4_sve(t->Y, (size_t)t->Y_stride, r, tok,
                                  g0, g1, g2, g3, u0, u1, u2, u3);
#else
            t->Y[(size_t)(tok + 0) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g0, u0);
            t->Y[(size_t)(tok + 1) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g1, u1);
            t->Y[(size_t)(tok + 2) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g2, u2);
            t->Y[(size_t)(tok + 3) * t->Y_stride + r] = tf_gelu_mul_fast_scalar(g3, u3);
#endif
        }
        for (; tok < t->N; tok++) {
            const float *x = t->X + (size_t)tok * t->X_stride;
            float gate, up;
            tf_vec_dot_q4_0_pair_f32(grow, urow, x, t->K, &gate, &up);
            t->Y[(size_t)tok * t->Y_stride + r] = tf_gelu_mul_fast_scalar(gate, up);
        }
    }
    return NULL;
}

static void tf_dequant_q4_0_rows_to_fp16_bt(uint16_t *bt, const qtensor *mat,
                                             int row0, int n_rows, int K) {
    const int nb = K / 32;
    const block_q4_0 *rows = (const block_q4_0 *)mat->data + (size_t)row0 * nb;
    for (int rr = 0; rr < n_rows; rr++) {
        const block_q4_0 *row = rows + (size_t)rr * nb;
        for (int b = 0; b < nb; b++) {
            float d = ggml_fp16_to_fp32(row[b].d);
            int base = b * 32;
            for (int j = 0; j < 16; j++) {
                uint8_t q = row[b].qs[j];
                bt[(size_t)(base + j) * n_rows + rr] =
                    tf_f32_to_f16(((float)((int)(q & 0x0f) - 8)) * d);
                bt[(size_t)(base + j + 16) * n_rows + rr] =
                    tf_f32_to_f16(((float)((int)(q >> 4) - 8)) * d);
            }
        }
    }
}

static int tf_gemm_q4_0_pair_gelu_block_fp16(float *Y_out, const qtensor *gate,
                                             const qtensor *up, const float *X,
                                             int n_rows, int N, int out_stride,
                                             int X_stride) {
#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    if (!gemm_fp16_BT) return 0;
    if (!gate || !up || gate->type != GGML_TYPE_Q4_0 || up->type != GGML_TYPE_Q4_0)
        return 0;
    if (gate->n_cols != up->n_cols || gate->n_rows != up->n_rows ||
        gate->n_cols != X_stride || n_rows != gate->n_rows)
        return 0;
    int K = gate->n_cols;
    if ((K % 32) != 0) return 0;

    const int rb_target = 1024;
    for (int row0 = 0; row0 < n_rows; row0 += rb_target) {
        int rb = n_rows - row0;
        if (rb > rb_target) rb = rb_target;
        size_t w_elems = (size_t)K * rb;
        size_t y_elems = (size_t)N * rb;
        uint16_t *gate_bt = (uint16_t *)malloc(w_elems * sizeof(uint16_t));
        uint16_t *up_bt = (uint16_t *)malloc(w_elems * sizeof(uint16_t));
        float *gate_y = (float *)malloc(y_elems * sizeof(float));
        float *up_y = (float *)malloc(y_elems * sizeof(float));
        if (!gate_bt || !up_bt || !gate_y || !up_y) {
            free(gate_bt); free(up_bt); free(gate_y); free(up_y);
            return 0;
        }

        tf_dequant_q4_0_rows_to_fp16_bt(gate_bt, gate, row0, rb, K);
        tf_dequant_q4_0_rows_to_fp16_bt(up_bt, up, row0, rb, K);
        gemm_fp16_BT(N, K, rb, X, X_stride, gate_bt, rb, gate_y, rb);
        gemm_fp16_BT(N, K, rb, X, X_stride, up_bt, rb, up_y, rb);
        tf_gelu_mul_fast(gate_y, gate_y, up_y, (int)y_elems);
        for (int t = 0; t < N; t++) {
            memcpy(Y_out + (size_t)t * out_stride + row0,
                   gate_y + (size_t)t * rb, (size_t)rb * sizeof(float));
        }

        free(gate_bt); free(up_bt); free(gate_y); free(up_y);
    }
    return 1;
#else
    (void)Y_out; (void)gate; (void)up; (void)X; (void)n_rows;
    (void)N; (void)out_stride; (void)X_stride;
    return 0;
#endif
}

static int tf_gemm_q4_0_pair_gelu_tokenmajor(float *Y_out, const qtensor *gate,
                                             const qtensor *up, const float *X,
                                             int n_rows, int N, int out_stride,
                                             int X_stride, int n_threads,
                                             int check) {
    if (!gate || !up || gate->type != GGML_TYPE_Q4_0 || up->type != GGML_TYPE_Q4_0)
        return 0;
    if (gate->n_cols != up->n_cols || gate->n_rows != up->n_rows ||
        gate->n_cols != X_stride || n_rows != gate->n_rows)
        return 0;
    int K = gate->n_cols;
    if ((K % 32) != 0) return 0;
    const char *block_q4 = getenv("TF_FFN_BLOCK_Q4");
    if (block_q4 && atoi(block_q4) != 0 && N >= 64 &&
        tf_gemm_q4_0_pair_gelu_block_fp16(Y_out, gate, up, X,
                                          n_rows, N, out_stride, X_stride))
        return 1;

    int nt = n_threads;
    if (nt < 1) nt = 1;
    if (nt > n_rows) nt = n_rows;
    if (nt <= 1 || n_rows < 8) {
        tf_gemm_q4_pair_gelu_task task = {Y_out, gate, up, X, 0, n_rows, K, N, out_stride, X_stride};
        (void)tf_gemm_q4_0_pair_gelu_worker(&task);
    } else {
        pthread_t *threads = (pthread_t *)alloca((size_t)nt * sizeof(pthread_t));
        tf_gemm_q4_pair_gelu_task *tasks = (tf_gemm_q4_pair_gelu_task *)alloca((size_t)nt * sizeof(tf_gemm_q4_pair_gelu_task));
        int rows_per = n_rows / nt, extra = n_rows % nt, offset = 0;
        for (int i = 0; i < nt; i++) {
            int count = rows_per + (i < extra ? 1 : 0);
            tasks[i] = (tf_gemm_q4_pair_gelu_task){Y_out, gate, up, X, offset, offset + count,
                                                   K, N, out_stride, X_stride};
            offset += count;
            pthread_create(&threads[i], NULL, tf_gemm_q4_0_pair_gelu_worker, &tasks[i]);
        }
        for (int i = 0; i < nt; i++) pthread_join(threads[i], NULL);
    }

    if (check && N > 0 && n_rows > 0) {
        const int nb = K / 32;
        const block_q4_0 *grow = (const block_q4_0 *)gate->data;
        const block_q4_0 *urow = (const block_q4_0 *)up->data;
        float g, u;
        tf_vec_dot_q4_0_pair_f32(grow, urow, X, K, &g, &u);
        float exact = tf_gelu_mul_exact_scalar(g, u);
        float fast = Y_out[0];
        float abs_err = fabsf(fast - exact);
        float rel_err = abs_err / (fabsf(exact) + 1e-9f);
        fprintf(stderr, "transformer: TF_FFN_CHECK q4 fused row0 tok0 gate=%.6g up=%.6g fast=%.6g exact=%.6g abs=%.3g rel=%.3g nb=%d\n",
                g, u, fast, exact, abs_err, rel_err, nb);
    }
    return 1;
}

#ifdef TF_HAVE_Q8V2
/* Q8v2 per-block int8 FFN gate/up GEMM with fused fast-GELU. Drop-in for
 * tf_gemm_q4_0_pair_gelu_tokenmajor (~12x faster, argmax-identical, ~0.6% relL2).
 * Q4_0 weights packed ON-THE-FLY per n-tile to centered-nibble int8 (pack_B layout,
 * per-thread scratch — never globally cached, avoids the 30GB OOM); activations
 * quantized per-32-block per-row to int8. Kernel: gemma4-kernels/kernel_q8v2_3x4.S.
 * Enable at runtime with env TF_Q8V2=1. Requires K%256==0 and n_rows%64==0. */
extern void kernel_q8v2_3x4(const int8_t*,const float*,const int8_t*,const float*,long,float*,long);
int tf_q8v2_enable = -1;  /* <0=init from env TF_Q8V2; 0/1 settable by caller for A/B */
static int tf_gemm_q8v2_pair_gelu_tokenmajor(float *Y, const qtensor *gate,
        const qtensor *up, const float *X, int n_rows, int N,
        int out_stride, int X_stride, int n_threads) {
    if (!gate || !up || gate->type!=GGML_TYPE_Q4_0 || up->type!=GGML_TYPE_Q4_0) return 0;
    if (gate->n_cols!=up->n_cols || gate->n_rows!=up->n_rows ||
        gate->n_cols!=X_stride || n_rows!=gate->n_rows) return 0;
    int K = gate->n_cols;
    if (K%256 || n_rows%64) return 0;
    const int MR=3, NR=64, BLK=32;
    int nb=K/BLK, NTn=n_rows/NR, MTn=(N+MR-1)/MR;
    int nt=n_threads<1?1:n_threads;
    size_t aqt=(size_t)nb*MR*BLK, adt=(size_t)nb*MR;
    int8_t*Aq=(int8_t*)malloc((size_t)MTn*aqt);
    float *Ad=(float*)malloc((size_t)MTn*adt*sizeof(float));
    if(!Aq||!Ad){ free(Aq); free(Ad); return 0; }
    /* 1. quantize activations per-block per-row -> int8 */
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int m0=0;m0<MTn;m0++){ int8_t*aq=Aq+(size_t)m0*aqt; float*ad=Ad+(size_t)m0*adt;
        for(int b=0;b<nb;b++) for(int r=0;r<MR;r++){ int tok=m0*MR+r;
            float amax=0; if(tok<N) for(int k=0;k<BLK;k++){ float a=fabsf(X[(size_t)tok*X_stride+b*BLK+k]); if(a>amax)amax=a; }
            float d=amax>0?amax/127.0f:0.0f, inv=d>0?1.0f/d:0.0f;
            ad[b*MR+r]=d;
            for(int k=0;k<BLK;k++){ int q=0; if(tok<N){ q=(int)lrintf(X[(size_t)tok*X_stride+b*BLK+k]*inv); if(q>127)q=127; if(q<-127)q=-127; } aq[(size_t)(b*MR+r)*BLK+k]=(int8_t)q; }
        }
    }
    /* 2. per n-tile: dequant gate/up Q4_0 -> centered-nibble int8 scratch, kernel x2, GELU */
    const block_q4_0*G=(const block_q4_0*)gate->data;
    const block_q4_0*U=(const block_q4_0*)up->data;
    #pragma omp parallel num_threads(nt)
    {
        size_t bqt=(size_t)nb*8*4*64;
        int8_t*bqg=(int8_t*)malloc(bqt), *bqu=(int8_t*)malloc(bqt);
        float *bdg=(float*)malloc((size_t)nb*NR*sizeof(float)), *bdu=(float*)malloc((size_t)nb*NR*sizeof(float));
        float Cg[MR*NR], Cu[MR*NR];
        #pragma omp for schedule(static)
        for(int n0=0;n0<NTn;n0++){
            for(int b=0;b<nb;b++) for(int vec=0;vec<4;vec++) for(int col=0;col<16;col++){
                int n=n0*NR+vec*16+col;
                const block_q4_0*gr=G+(size_t)n*nb, *ur=U+(size_t)n*nb;
                bdg[b*NR+vec*16+col]=ggml_fp16_to_fp32(gr[b].d);
                bdu[b*NR+vec*16+col]=ggml_fp16_to_fp32(ur[b].d);
                for(int g3=0;g3<8;g3++) for(int kk=0;kk<4;kk++){ int k=g3*4+kk;
                    int qg=(k<16)?(gr[b].qs[k]&0xf):(gr[b].qs[k-16]>>4);
                    int qu=(k<16)?(ur[b].qs[k]&0xf):(ur[b].qs[k-16]>>4);
                    size_t off=((size_t)b*8*4+(size_t)g3*4+vec)*64 + (size_t)col*4 + kk;
                    bqg[off]=(int8_t)(qg-8); bqu[off]=(int8_t)(qu-8);
                }
            }
            for(int m0=0;m0<MTn;m0++){
                const int8_t*aq=Aq+(size_t)m0*aqt; const float*ad=Ad+(size_t)m0*adt;
                kernel_q8v2_3x4(aq,ad,bqg,bdg,nb,Cg,(long)NR*4);
                kernel_q8v2_3x4(aq,ad,bqu,bdu,nb,Cu,(long)NR*4);
                for(int r=0;r<MR;r++){ int tok=m0*MR+r; if(tok>=N) continue;
                    for(int c=0;c<NR;c++) Y[(size_t)tok*out_stride + n0*NR + c]=tf_gelu_fast_scalar(Cg[r*NR+c])*Cu[r*NR+c];
                }
            }
        }
        free(bqg); free(bqu); free(bdg); free(bdu);
    }
    free(Aq); free(Ad);
    return 1;
}
#endif /* TF_HAVE_Q8V2 */

/* Token-major GEMM: Y[tok * out_stride + row] = dot(W[row,:], X[tok,:])
 * Direct output without transpose. */
static void tf_gemm_f16_mt_tokenmajor(float *Y_out, const qtensor *mat, const float *X,
                                       int n_rows, int N, int out_stride, int X_stride,
                                       int n_threads) {
    if (mat->type == GGML_TYPE_Q8_0) {
        /* Q8_0 SIMD GEMM path */
        int K = mat->n_cols;
        if (n_threads <= 1 || n_rows < n_threads * 6) {
            gemm_q8_0_f32_tokmajor(Y_out, mat->data, X, n_rows, K, N, out_stride, X_stride);
            return;
        }
        pthread_t *threads = (pthread_t *)alloca(n_threads * sizeof(pthread_t));
        tf_gemm_q8_tm_task *tasks = (tf_gemm_q8_tm_task *)alloca(n_threads * sizeof(tf_gemm_q8_tm_task));
        int rows_per = n_rows / n_threads, extra = n_rows % n_threads, offset = 0;
        for (int i = 0; i < n_threads; i++) {
            int count = rows_per + (i < extra ? 1 : 0);
            tasks[i] = (tf_gemm_q8_tm_task){Y_out,
                                    (const uint8_t *)mat->data,
                                    X, offset, offset + count, K, N, out_stride, X_stride};
            offset += count;
        }
        for (int i = 0; i < n_threads; i++) {
            pthread_create(&threads[i], NULL, tf_gemm_q8_tm_worker, &tasks[i]);
        }
        for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
        return;
    }
    if (mat->type == GGML_TYPE_BF16) {
        /* BF16 GEMM path */
        int K = mat->n_cols;
#ifdef TF_HAVE_BF16_BLOCKED
        /* prefill (N>=4): register-blocked, PINNED OpenMP (raw pthreads below float
         * across CMGs -> cross-CMG weight reads). N=1 decode keeps the matvec path. */
        if (N >= 4) {
            tf_gemm_bf16_blocked(Y_out, (const uint16_t *)mat->data, X,
                                 n_rows, K, N, out_stride, X_stride,
                                 n_threads > 1 ? n_threads : 1);
            return;
        }
#endif
        if (n_threads <= 1 || n_rows < n_threads * 4) {
            gemm_bf16_f32_tokmajor(Y_out, (const uint16_t *)mat->data, X,
                                    n_rows, K, N, out_stride, X_stride);
            return;
        }
        pthread_t *threads = (pthread_t *)alloca(n_threads * sizeof(pthread_t));
        tf_gemm_tm_task *tasks = (tf_gemm_tm_task *)alloca(n_threads * sizeof(tf_gemm_tm_task));
        int rows_per = n_rows / n_threads, extra = n_rows % n_threads, offset = 0;
        for (int i = 0; i < n_threads; i++) {
            int count = rows_per + (i < extra ? 1 : 0);
            tasks[i] = (tf_gemm_tm_task){Y_out, (const uint16_t *)mat->data,
                                          X, offset, offset + count, K, N, out_stride, X_stride};
            offset += count;
        }
        for (int i = 0; i < n_threads; i++)
            pthread_create(&threads[i], NULL, tf_gemm_bf16_tm_worker, &tasks[i]);
        for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
        return;
    }
    if (mat->type == GGML_TYPE_Q4_0) {
        int K = mat->n_cols;
        int nt = n_threads;
        if (nt < 1) nt = 1;
        if (nt > n_rows) nt = n_rows;
        if (nt <= 1 || n_rows < 8) {
            tf_gemm_qtensor_tm_task task = {Y_out, mat, X, 0, n_rows, K, N, out_stride, X_stride};
            (void)tf_gemm_q4_0_tm_worker(&task);
            return;
        }
        pthread_t *threads = (pthread_t *)alloca((size_t)nt * sizeof(pthread_t));
        tf_gemm_qtensor_tm_task *tasks = (tf_gemm_qtensor_tm_task *)alloca((size_t)nt * sizeof(tf_gemm_qtensor_tm_task));
        int rows_per = n_rows / nt, extra = n_rows % nt, offset = 0;
        for (int i = 0; i < nt; i++) {
            int count = rows_per + (i < extra ? 1 : 0);
            tasks[i] = (tf_gemm_qtensor_tm_task){Y_out, mat, X, offset, offset + count,
                                                 K, N, out_stride, X_stride};
            offset += count;
            pthread_create(&threads[i], NULL, tf_gemm_q4_0_tm_worker, &tasks[i]);
        }
        for (int i = 0; i < nt; i++) pthread_join(threads[i], NULL);
        return;
    }
    if (mat->type != GGML_TYPE_F16) {
        int K = mat->n_cols;
        int nt = n_threads;
        if (nt < 1) nt = 1;
        if (nt > n_rows) nt = n_rows;
        if (nt <= 1 || n_rows < 8) {
            tf_gemm_qtensor_tm_task task = {Y_out, mat, X, 0, n_rows, K, N, out_stride, X_stride};
            (void)tf_gemm_qtensor_tm_worker(&task);
            return;
        }
        pthread_t *threads = (pthread_t *)alloca((size_t)nt * sizeof(pthread_t));
        tf_gemm_qtensor_tm_task *tasks = (tf_gemm_qtensor_tm_task *)alloca((size_t)nt * sizeof(tf_gemm_qtensor_tm_task));
        int rows_per = n_rows / nt, extra = n_rows % nt, offset = 0;
        for (int i = 0; i < nt; i++) {
            int count = rows_per + (i < extra ? 1 : 0);
            tasks[i] = (tf_gemm_qtensor_tm_task){Y_out, mat, X, offset, offset + count,
                                                 K, N, out_stride, X_stride};
            offset += count;
            pthread_create(&threads[i], NULL, tf_gemm_qtensor_tm_worker, &tasks[i]);
        }
        for (int i = 0; i < nt; i++) pthread_join(threads[i], NULL);
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

static void TF_MAYBE_UNUSED tf_gemm_f16_mt_fused2(float *Y1, const qtensor *mat1,
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
    /* Use rope_dim_count (= 2 * sect_dims) as freq denominator, not head_dim */
    int rope_dim = 2 * sect_dims;

    /* Precompute per-dimension: freq[j] and which position axis to use (0=t,1=h,2=w) */
    float *freq = (float *)alloca(half_dim * sizeof(float));
    int *pos_axis = (int *)alloca(half_dim * sizeof(int));

    for (int j = 0; j < half_dim; j++) {
        freq[j] = 1.0f / powf(freq_base, (float)(2 * j) / rope_dim);
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
    /* per-token independent (w_buf shared read-only) -> parallelize over tokens */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int t = 0; t < N; t++) {
        const float *xi = src + (size_t)t * n_embd;
        float *yi = dst + (size_t)t * n_embd;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 vss = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < n_embd; i += 8) {
            __m256 vx = _mm256_loadu_ps(xi + i);
            vss = _mm256_fmadd_ps(vx, vx, vss);
        }
        __m128 hi = _mm256_extractf128_ps(vss, 1);
        __m128 lo = _mm256_castps256_ps128(vss);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
        float ss = _mm_cvtss_f32(s4);
        for (; i < n_embd; i++) ss += xi[i] * xi[i];
        ss = 1.0f / sqrtf(ss / n_embd + eps);
        __m256 vscale = _mm256_set1_ps(ss);
        i = 0;
        for (; i + 7 < n_embd; i += 8)
            _mm256_storeu_ps(yi + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xi + i), vscale),
                                                     _mm256_loadu_ps(w_buf + i)));
        for (; i < n_embd; i++) yi[i] = xi[i] * ss * w_buf[i];
#else
        float ss = 0.0f;
        for (int i = 0; i < n_embd; i++) ss += xi[i] * xi[i];
        ss = 1.0f / sqrtf(ss / n_embd + eps);
        for (int i = 0; i < n_embd; i++) yi[i] = xi[i] * ss * w_buf[i];
#endif
    }
}

/* Batched QK-norm: per-head RMSNorm for N tokens */
static void tf_qk_norm_batch(float *vec, int n_heads, int head_dim, int N,
                               const qtensor *norm_w, float eps, float *w_buf) {
    tf_dequant_row(norm_w, 0, w_buf);
    int vec_dim = n_heads * head_dim;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *v = vec + (size_t)t * vec_dim + h * head_dim;
#if defined(__AVX2__) && defined(__FMA__)
            __m256 vss = _mm256_setzero_ps();
            int i = 0;
            for (; i + 7 < head_dim; i += 8) {
                __m256 vx = _mm256_loadu_ps(v + i);
                vss = _mm256_fmadd_ps(vx, vx, vss);
            }
            __m128 hi = _mm256_extractf128_ps(vss, 1);
            __m128 lo = _mm256_castps256_ps128(vss);
            __m128 s4 = _mm_add_ps(lo, hi);
            s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
            float ss = _mm_cvtss_f32(s4);
            for (; i < head_dim; i++) ss += v[i] * v[i];
            ss = 1.0f / sqrtf(ss / head_dim + eps);
            __m256 vscale = _mm256_set1_ps(ss);
            i = 0;
            for (; i + 7 < head_dim; i += 8)
                _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(v + i), vscale),
                                                       _mm256_loadu_ps(w_buf + i)));
            for (; i < head_dim; i++) v[i] = v[i] * ss * w_buf[i];
#else
            float ss = 0.0f;
            for (int i = 0; i < head_dim; i++) ss += v[i] * v[i];
            ss = 1.0f / sqrtf(ss / head_dim + eps);
            for (int i = 0; i < head_dim; i++) v[i] = v[i] * ss * w_buf[i];
#endif
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

static TF_MAYBE_UNUSED void *tf_batch_attn_worker(void *arg) {
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
    float *bx     = (float *)tf_aligned_alloc_notouch(256, (size_t)N * n_embd * sizeof(float));
    float *bxb    = (float *)tf_aligned_alloc_notouch(256, (size_t)N * n_embd * sizeof(float));
    float *bq     = (float *)tf_aligned_alloc_notouch(256, (size_t)N * q_dim * sizeof(float));
    float *bk     = (float *)tf_aligned_alloc_notouch(256, (size_t)N * kv_dim * sizeof(float));
    float *bv     = (float *)tf_aligned_alloc_notouch(256, (size_t)N * kv_dim * sizeof(float));
    float *bxb2   = (float *)tf_aligned_alloc_notouch(256, (size_t)N * q_dim * sizeof(float));
    float *bffn1  = (float *)tf_aligned_alloc_notouch(256, (size_t)N * n_ff * sizeof(float));
    float *bffn2  = (float *)tf_aligned_alloc_notouch(256, (size_t)N * n_ff * sizeof(float));
    float *bffn3  = (float *)tf_aligned_alloc_notouch(256, (size_t)N * n_ff * sizeof(float));

    /* Per-thread attention score scratch: each thread needs [max_seq_len] */
    float *batch_att_scratch = (float *)tf_aligned_alloc_notouch(256,
        (size_t)m->n_threads * m->max_seq_len * sizeof(float));

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
        for (int t = 0; t < N; t++) tf_vadd(bx + (size_t)t * n_embd, bxb + (size_t)t * n_embd, n_embd);
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
        for (int t = 0; t < N; t++) tf_vadd(bx + (size_t)t * n_embd, bxb + (size_t)t * n_embd, n_embd);
        t_residual += tf_time_ms() - t0p;

        /* 10. DeepStack injection */
        t0p = tf_time_ms();
        if (b->ds_embds && l < m->n_deepstack) {
            for (int t = 0; t < N; t++) {
                if (b->ds_embds[t] && b->ds_embd_stride > n_embd) {
                    const float *ds_slice = b->ds_embds[t] + (1 + l) * n_embd;
                    float *xt = bx + (size_t)t * n_embd;
                    tf_vadd(xt, ds_slice, n_embd);
                }
            }
        }

        t_deepstack += tf_time_ms() - t0p;

        if (m->trace_hidden_norms && (l == 0 || l == m->n_layers - 1 || (l + 1) % 10 == 0)) {
            float *last = bx + (size_t)(N-1) * n_embd;
            fprintf(stderr, "  [batch] layer %2d: last token hidden norm = %.4f\n", l, sqrtf(tf_sum_squares(last, n_embd)));
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

static void tf_gemma4_rope_batch(transformer_model *m, transformer_layer *layer,
                                  float *bq, float *bk, const int *pos,
                                  int N, int n_heads, int n_kv_heads,
                                  int hd, int q_dim, int kv_dim) {
    float *inv_freq = layer->is_swa ? m->rope_inv_freq_swa : m->rope_inv_freq;
    int half = hd / 2;
    int nt = m->n_threads > 1 ? m->n_threads : 1;
    /* parallel over tokens; cos/sin depend only on (token,j) so hoist out of the
     * head loop (was recomputed per head -> ~n_heads x redundant trig). */
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nt) schedule(static)
    #endif
    for (int t = 0; t < N; t++) {
        int p = pos[t];
        float cs[256], sn[256];   /* half = hd/2 <= 256 */
        for (int j = 0; j < half; j++) { float f = (float)p * inv_freq[j]; cs[j] = cosf(f); sn[j] = sinf(f); }
        for (int h = 0; h < n_heads; h++) {
            float *qh = bq + (size_t)t * q_dim + h * hd;
            for (int j = 0; j < half; j++) {
                float r0 = qh[j], r1 = qh[j + half];
                qh[j] = r0 * cs[j] - r1 * sn[j];
                qh[j + half] = r0 * sn[j] + r1 * cs[j];
            }
        }
        if (layer->shared_kv_source < 0) {
            for (int h = 0; h < n_kv_heads; h++) {
                float *kh = bk + (size_t)t * kv_dim + h * hd;
                for (int j = 0; j < half; j++) {
                    float r0 = kh[j], r1 = kh[j + half];
                    kh[j] = r0 * cs[j] - r1 * sn[j];
                    kh[j + half] = r0 * sn[j] + r1 * cs[j];
                }
            }
        }
    }
}

static void tf_gemma4_raw_v_norm_batch(float *bv, int N, int n_kv_heads,
                                        int hd, int kv_dim, float eps) {
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < n_kv_heads; h++) {
            float *v = bv + (size_t)t * kv_dim + h * hd;
            float ss = 0.0f;
            for (int i = 0; i < hd; i++) ss += v[i] * v[i];
            ss = 1.0f / sqrtf(ss / hd + eps);
            for (int i = 0; i < hd; i++) v[i] *= ss;
        }
    }
}

static void tf_gemma4_attention_batch(transformer_model *m, transformer_layer *layer,
                                       float *bq, float *bxb2, const int *pos,
                                       int N, int kv_src, int n_heads,
                                       int n_kv_heads, int hd, int q_dim,
                                       int kv_dim, int gqa) {
    memset(bxb2, 0, (size_t)N * q_dim * sizeof(float));
    /* Parallel over (token,head): each (t,h) writes a disjoint bxb2[t*q_dim+h*hd]
     * region and uses a thread-local score buffer -> race-free. Was fully serial
     * (the prefill's dominant cost). KV dtype handled by tf_kv_load_* (scalar). */
    int nt = m->n_threads > 1 ? m->n_threads : 1;
    #ifdef _OPENMP
    #pragma omp parallel num_threads(nt)
    #endif
    {
        float *att = (float *)malloc((size_t)m->max_seq_len * sizeof(float));
        #ifdef _OPENMP
        #pragma omp for collapse(2) schedule(dynamic)
        #endif
        for (int t = 0; t < N; t++) {
            for (int h = 0; h < n_heads; h++) {
                int cur_pos = pos[t];
                int start = 0, seq_len = cur_pos + 1;
                if (layer->is_swa && seq_len > m->swa_window_size) {
                    start = cur_pos - m->swa_window_size + 1;
                    seq_len = m->swa_window_size;
                }
                float *qh = bq + (size_t)t * q_dim + h * hd;
                int kv_h = h / gqa;
                for (int p = 0; p < seq_len; p++) {
                    int abs_pos = start + p;
                    int slot = layer->is_swa ? (abs_pos % m->swa_window_size) : abs_pos;
                    size_t kbase = (size_t)slot * kv_dim + (size_t)kv_h * hd;
                    float score = 0.0f;
                    for (int d = 0; d < hd; d++) score += qh[d] * tf_kv_load_key(m, kv_src, kbase + d);
                    att[p] = score;
                }
                float max_s = att[0];
                for (int p = 1; p < seq_len; p++) if (att[p] > max_s) max_s = att[p];
                float sum_e = 0.0f;
                for (int p = 0; p < seq_len; p++) { att[p] = expf(att[p] - max_s); sum_e += att[p]; }
                float inv_sum = 1.0f / sum_e;
                float *out_h = bxb2 + (size_t)t * q_dim + h * hd;
                for (int p = 0; p < seq_len; p++) {
                    int abs_pos = start + p;
                    int slot = layer->is_swa ? (abs_pos % m->swa_window_size) : abs_pos;
                    size_t vbase = (size_t)slot * kv_dim + (size_t)kv_h * hd;
                    float w = att[p] * inv_sum;
                    for (int d = 0; d < hd; d++) out_h[d] += w * tf_kv_load_value(m, kv_src, vbase + d);
                }
            }
        }
        free(att);
    }
}

static float *tf_gemma4_prefill_batch(transformer_model *m, const int32_t *tokens,
                                      int n_tokens, int start_pos) {
    if (!m || !tokens || n_tokens <= 0 || !m->is_gemma4) return NULL;
    if (m->n_embd_per_layer > 0 || m->per_layer_token_embd.data) {
        fprintf(stderr, "transformer: Gemma4 batch prefill: PLE models not supported, using token loop\n");
        return NULL;
    }

    int N = n_tokens;
    int n_embd = m->n_embd;
    int n_heads = m->n_heads;
    int max_q_dim = n_heads * m->head_dim;
    int max_kv_dim = 0;
    for (int l = 0; l < m->n_layers; l++) {
        transformer_layer *layer = &m->layers[l];
        int hd = layer->is_swa ? m->head_dim_swa : m->head_dim_full;
        int l_kvh = layer->n_kv_heads > 0 ? layer->n_kv_heads : m->n_kv_heads;
        int local_kv_dim = l_kvh * hd;
        if (local_kv_dim > max_kv_dim) max_kv_dim = local_kv_dim;
    }
    if (max_q_dim <= 0 || max_kv_dim <= 0) return NULL;

    int fused_q4_prefill = m->ffn_fused_q4 && m->ffn_gelu_fast;
    if (fused_q4_prefill) {
        for (int l = 0; l < m->n_layers; l++) {
            transformer_layer *layer = &m->layers[l];
            if (layer->ffn_gate.type != GGML_TYPE_Q4_0 || layer->ffn_up.type != GGML_TYPE_Q4_0 ||
                layer->ffn_gate.n_rows != m->n_ff || layer->ffn_up.n_rows != m->n_ff ||
                layer->ffn_gate.n_cols != n_embd || layer->ffn_up.n_cols != n_embd ||
                (layer->ffn_gate.n_cols % 32) != 0) {
                fused_q4_prefill = 0;
                break;
            }
        }
    }
    if (m->ffn_fused_q4 && !fused_q4_prefill) {
        fprintf(stderr, "transformer: Gemma4 FFN fused_q4 prefill fallback (requires fast GELU and Q4_0 gate/up)\n");
    } else if (fused_q4_prefill) {
        fprintf(stderr, "transformer: Gemma4 FFN fused_q4 prefill active (no bff1/bff2 scratch)\n");
    }

    /* Allocate scratch from the persistent NUMA bump pool (no per-call malloc/free
     * churn; pages first-touched NUMA-spread once). Pool ensure runs BEFORE the
     * thread-pool shutdown below so tf_numa_distribute_buffer can place pages. */
    #define TF_RND256(x) (((size_t)(x) + 255u) & ~(size_t)255u)
    size_t ff_bytes = (size_t)N * m->n_ff * sizeof(float);
    size_t need = TF_RND256((size_t)N * sizeof(int))
        + 2 * TF_RND256((size_t)N * n_embd * sizeof(float))      /* bx, bxb */
        + 2 * TF_RND256((size_t)N * max_q_dim * sizeof(float))   /* bq, bxb2 */
        + 2 * TF_RND256((size_t)N * max_kv_dim * sizeof(float))  /* bk, bv */
        + (fused_q4_prefill ? 0 : 2 * TF_RND256(ff_bytes))       /* bff1, bff2 */
        + TF_RND256(ff_bytes);                                   /* bff3 */
    if (!tf_mpool_ensure(m, need)) {
        fprintf(stderr, "transformer: Gemma4 batch prefill pool alloc failed (N=%d, need=%zu)\n", N, need);
        return NULL;
    }
    tf_mpool_reset(m);
    int *pos = (int *)tf_mpool_alloc(m, (size_t)N * sizeof(int));
    float *bx    = (float *)tf_mpool_alloc(m, (size_t)N * n_embd * sizeof(float));
    float *bxb   = (float *)tf_mpool_alloc(m, (size_t)N * n_embd * sizeof(float));
    float *bq    = (float *)tf_mpool_alloc(m, (size_t)N * max_q_dim * sizeof(float));
    float *bk    = (float *)tf_mpool_alloc(m, (size_t)N * max_kv_dim * sizeof(float));
    float *bv    = (float *)tf_mpool_alloc(m, (size_t)N * max_kv_dim * sizeof(float));
    float *bxb2  = (float *)tf_mpool_alloc(m, (size_t)N * max_q_dim * sizeof(float));
    float *bff1  = fused_q4_prefill ? NULL : (float *)tf_mpool_alloc(m, ff_bytes);
    float *bff2  = fused_q4_prefill ? NULL : (float *)tf_mpool_alloc(m, ff_bytes);
    float *bff3  = (float *)tf_mpool_alloc(m, ff_bytes);
    #undef TF_RND256
    if (!pos || !bx || !bxb || !bq || !bk || !bv || !bxb2 ||
        (!fused_q4_prefill && (!bff1 || !bff2)) || !bff3) {
        fprintf(stderr, "transformer: Gemma4 batch prefill pool sub-alloc failed (N=%d)\n", N);
        return NULL;
    }

    for (int t = 0; t < N; t++) {
        pos[t] = start_pos + t;
        tf_dequant_row(&m->token_embd, tokens[t], bx + (size_t)t * n_embd);
        float scale = m->embd_scale;
        for (int i = 0; i < n_embd; i++) bx[(size_t)t * n_embd + i] *= scale;
    }

    double t_attn_norm = 0, t_qkv = 0, t_qk_norm = 0, t_rope = 0, t_kv = 0;
    double t_attn = 0, t_out = 0, t_post = 0, t_ffn_gateup = 0, t_gelu = 0, t_ffn_down = 0;
    double t0p;

    int pool_was_alive = m->pool_alive;
    if (pool_was_alive) tf_pool_shutdown(m);

    for (int l = 0; l < m->n_layers; l++) {
        transformer_layer *layer = &m->layers[l];
        int hd = layer->is_swa ? m->head_dim_swa : m->head_dim_full;
        int l_kvh = layer->n_kv_heads > 0 ? layer->n_kv_heads : m->n_kv_heads;
        int local_kv_dim = l_kvh * hd;
        int local_q_dim = n_heads * hd;
        int local_gqa = n_heads / l_kvh;
        float eps = m->rms_norm_eps;

        t0p = tf_time_ms();
        tf_rmsnorm_batch(bxb, bx, &layer->attn_norm, n_embd, N, eps, m->matvec_tmp);
        t_attn_norm += tf_time_ms() - t0p;

        t0p = tf_time_ms();
        tf_gemm_f16_mt_tokenmajor(bq, &layer->attn_q, bxb, local_q_dim, N, local_q_dim, n_embd, m->n_threads);
        int kv_src = (layer->shared_kv_source >= 0) ? layer->shared_kv_source : l;
        if (layer->shared_kv_source < 0) {
            tf_gemm_f16_mt_tokenmajor(bk, &layer->attn_k, bxb, local_kv_dim, N, local_kv_dim, n_embd, m->n_threads);
            if (layer->has_v_proj) {
                tf_gemm_f16_mt_tokenmajor(bv, &layer->attn_v, bxb, local_kv_dim, N, local_kv_dim, n_embd, m->n_threads);
            } else {
                memcpy(bv, bk, (size_t)N * local_kv_dim * sizeof(float));
            }
        }
        t_qkv += tf_time_ms() - t0p;

        t0p = tf_time_ms();
        tf_qk_norm_batch(bq, n_heads, hd, N, &layer->attn_q_norm, eps, m->matvec_tmp);
        if (layer->shared_kv_source < 0) {
            tf_qk_norm_batch(bk, l_kvh, hd, N, &layer->attn_k_norm, eps, m->matvec_tmp);
            if (layer->attn_v_norm.data)
                tf_qk_norm_batch(bv, l_kvh, hd, N, &layer->attn_v_norm, eps, m->matvec_tmp);
            else
                tf_gemma4_raw_v_norm_batch(bv, N, l_kvh, hd, local_kv_dim, eps);
        }
        t_qk_norm += tf_time_ms() - t0p;

        t0p = tf_time_ms();
        tf_gemma4_rope_batch(m, layer, bq, bk, pos, N, n_heads, l_kvh, hd, local_q_dim, local_kv_dim);
        t_rope += tf_time_ms() - t0p;

        t0p = tf_time_ms();
        if (layer->shared_kv_source < 0) {
            for (int t = 0; t < N; t++) {
                int slot = layer->is_swa ? (pos[t] % m->swa_window_size) : pos[t];
                tf_kv_store(m, l, (size_t)slot * local_kv_dim,
                            bk + (size_t)t * local_kv_dim, local_kv_dim);
                tf_kv_store_value(m, l, (size_t)slot * local_kv_dim,
                                  bv + (size_t)t * local_kv_dim, local_kv_dim);
            }
        }
        t_kv += tf_time_ms() - t0p;

        t0p = tf_time_ms();
        tf_gemma4_attention_batch(m, layer, bq, bxb2, pos, N, kv_src, n_heads,
                                  l_kvh, hd, local_q_dim, local_kv_dim, local_gqa);
        t_attn += tf_time_ms() - t0p;

        t0p = tf_time_ms();
        tf_gemm_f16_mt_tokenmajor(bxb, &layer->attn_output, bxb2, n_embd, N, n_embd, local_q_dim, m->n_threads);
        t_out += tf_time_ms() - t0p;

        t0p = tf_time_ms();
        tf_rmsnorm_batch(bxb, bxb, &layer->post_attention_norm, n_embd, N, eps, m->matvec_tmp);
        for (int t = 0; t < N; t++) tf_vadd(bx + (size_t)t * n_embd, bxb + (size_t)t * n_embd, n_embd);
        tf_rmsnorm_batch(bxb, bx, &layer->ffn_norm, n_embd, N, eps, m->matvec_tmp);
        t_post += tf_time_ms() - t0p;

        t0p = tf_time_ms();
#ifdef TF_HAVE_Q8V2
        if (tf_q8v2_enable < 0) tf_q8v2_enable = getenv("TF_Q8V2") && atoi(getenv("TF_Q8V2"));
        if (tf_q8v2_enable &&
            tf_gemm_q8v2_pair_gelu_tokenmajor(bff3, &layer->ffn_gate, &layer->ffn_up,
                                               bxb, m->n_ff, N, m->n_ff, n_embd, m->n_threads)) {
            t_ffn_gateup += tf_time_ms() - t0p;
        } else
#endif
        if (fused_q4_prefill &&
            tf_gemm_q4_0_pair_gelu_tokenmajor(bff3, &layer->ffn_gate, &layer->ffn_up,
                                               bxb, m->n_ff, N, m->n_ff, n_embd,
                                               m->n_threads, m->ffn_check && l == 0)) {
            t_ffn_gateup += tf_time_ms() - t0p;
        } else {
            tf_gemm_f16_mt_tokenmajor(bff1, &layer->ffn_gate, bxb, m->n_ff, N, m->n_ff, n_embd, m->n_threads);
            tf_gemm_f16_mt_tokenmajor(bff2, &layer->ffn_up, bxb, m->n_ff, N, m->n_ff, n_embd, m->n_threads);
            t_ffn_gateup += tf_time_ms() - t0p;

            t0p = tf_time_ms();
            tf_gelu_mul(bff3, bff1, bff2, N * m->n_ff, m->ffn_gelu_fast);
            t_gelu += tf_time_ms() - t0p;
        }

        t0p = tf_time_ms();
        tf_gemm_f16_mt_tokenmajor(bxb, &layer->ffn_down, bff3, n_embd, N, n_embd, m->n_ff, m->n_threads);
        tf_rmsnorm_batch(bxb, bxb, &layer->post_ffw_norm, n_embd, N, eps, m->matvec_tmp);
        for (int t = 0; t < N; t++) tf_vadd(bx + (size_t)t * n_embd, bxb + (size_t)t * n_embd, n_embd);
        if (layer->layer_output_scale.data) {
            float scale_val;
            dequant_row(layer->layer_output_scale.type, layer->layer_output_scale.data, &scale_val, 1);
            for (int t = 0; t < N; t++) {
                float *xt = bx + (size_t)t * n_embd;
                for (int i = 0; i < n_embd; i++) xt[i] *= scale_val;
            }
        }
        t_ffn_down += tf_time_ms() - t0p;
    }

    float *last = bx + (size_t)(N - 1) * n_embd;
    tf_rmsnorm(m->x, last, &m->output_norm, n_embd, m->rms_norm_eps, m->matvec_tmp);
    tf_qmatvec_pool(m, m->logits, &m->output, m->x, m->n_vocab);
    if (m->final_logit_softcapping > 0.0f)
        tf_logit_softcap(m->logits, m->n_vocab, m->final_logit_softcapping);

    double total = t_attn_norm + t_qkv + t_qk_norm + t_rope + t_kv + t_attn + t_out + t_post + t_ffn_gateup + t_gelu + t_ffn_down;
    fprintf(stderr, "\n  === Gemma4 Batch Prefill Profile (%d tokens x %d layers) ===\n", N, m->n_layers);
    fprintf(stderr, "  attn_norm: %8.1f ms\n", t_attn_norm);
    fprintf(stderr, "  QKV GEMM:  %8.1f ms\n", t_qkv);
    fprintf(stderr, "  QK/V norm: %8.1f ms\n", t_qk_norm);
    fprintf(stderr, "  RoPE:      %8.1f ms\n", t_rope);
    fprintf(stderr, "  KV store:  %8.1f ms\n", t_kv);
    fprintf(stderr, "  attention: %8.1f ms\n", t_attn);
    fprintf(stderr, "  out proj:  %8.1f ms\n", t_out);
    fprintf(stderr, "  post/norm: %8.1f ms\n", t_post);
    fprintf(stderr, "  FFN gate/up:%7.1f ms%s\n", t_ffn_gateup, fused_q4_prefill ? " (fused q4+gelu)" : "");
    fprintf(stderr, "  GELU/mul:  %8.1f ms (%s)\n", t_gelu, m->ffn_gelu_fast ? "fast" : "exact");
    fprintf(stderr, "  FFN down:  %8.1f ms\n", t_ffn_down);
    fprintf(stderr, "  TOTAL:     %8.1f ms (%.2f tok/s)\n\n", total, total > 0 ? (1000.0 * N / total) : 0.0);

    if (pool_was_alive) tf_pool_start(m);
    /* scratch lives in m->mpool (persistent, reused next call) — do not free */
    (void)pos; (void)bx; (void)bxb; (void)bq; (void)bk; (void)bv; (void)bxb2;
    (void)bff1; (void)bff2; (void)bff3;
    return m->logits;
}

float *transformer_prefill_gemm(transformer_model *m, const int32_t *tokens,
                                int n_tokens, int start_pos) {
    if (!m || !tokens || n_tokens <= 0) return NULL;
    if (m->is_gemma4) return tf_gemma4_prefill_batch(m, tokens, n_tokens, start_pos);
    return NULL;
}

/* Top-k sampling with temperature */
int32_t transformer_sample_topk(const float *logits, int n_vocab, float temperature, int top_k) {
    if (top_k <= 0 || top_k > n_vocab) top_k = n_vocab;

    /* Find top-k indices by partial sort (alloca to avoid malloc in hot path) */
    int32_t *indices = (int32_t *)alloca(top_k * sizeof(int32_t));
    float *vals = (float *)alloca(top_k * sizeof(float));
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

    /* Softmax over top-k (uses AVX2 fast_exp when available) */
    tf_softmax(vals, top_k);

    /* Sample */
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    int32_t result = (top_k > 0) ? indices[0] : 0;
    for (int i = 0; i < top_k; i++) {
        cum += vals[i];
        if (r <= cum) { result = indices[i]; break; }
    }

    return result;
}

#endif /* TRANSFORMER_IMPLEMENTATION */
#endif /* TRANSFORMER_H */
