/*
 * m3.h - MiniMax-M3 (text LM) inference harness for A64FX/Fugaku.
 *
 * MiniMax-M3 is a 428B-param (~23B activated) sparse-MoE model, 869 GB bf16,
 * 60 layers, 1M context. This header is the M3 analogue of ds4f.h: it carries
 * the verified architecture config + the per-rank tensor/layer/model structs for
 * a custom expert-parallel (EP) + tensor-parallel (TP) runner on A64FX, reusing
 * the DS4P stack's infrastructure (ggml_dequant SVE matvec kernels, the uTofu EP
 * all-reduce, the parallel weight stager, the pjsub job patterns).
 *
 * M3 differs from DeepSeek-V4 in the FORWARD GRAPH (so it is a new graph, not a
 * ds4f config switch):
 *   - attention      = GQA, 64 q heads / 4 kv heads, head_dim 128 (NOT MLA latent)
 *   - sparse attn    = MiniMax Sparse Attention (MSA): a small index projection
 *                      scores 128-token blocks, selects top-16, full attention runs
 *                      only over selected + init + local blocks. Kernels live in
 *                      ~/work/clair/main/tests/a64fx/msa (block 128, head_dim 128).
 *   - MoE            = sigmoid+bias router, top-4 of 128, 1 shared expert,
 *                      routed_scaling 2.0, SwiGLU-OAI (alpha 1.702, limit 7.0)
 *   - RoPE           = partial rotary (rotary_dim 64 of 128), theta 5e6
 *   - norm           = Gemma RMSNorm (x * (1 + w)) + per-head QK-norm
 *   - layers 0..2    = DENSE (mlp intermediate 12288, full attention, no MSA/MoE)
 *   - layers 3..59   = MoE + MSA sparse attention
 *   - dtype          = bf16 throughout (router gate + e_score bias are f32)
 *
 * Verified against ~/models/m3/config.json + safetensors headers (2026-06-15):
 *   embed_tokens / lm_head        BF16 [200064, 6144]
 *   layers.L.self_attn.q_proj     BF16 [8192, 6144]    (64 * 128)
 *   layers.L.self_attn.k_proj     BF16 [512, 6144]     (4  * 128)
 *   layers.L.self_attn.v_proj     BF16 [512, 6144]
 *   layers.L.self_attn.o_proj     BF16 [6144, 8192]
 *   layers.L.self_attn.q_norm     BF16 [128]           (per-head RMSNorm)
 *   layers.L.self_attn.k_norm     BF16 [128]
 *   layers.L.self_attn.index_q_proj   BF16 [512, 6144] (4 index q heads * 128)
 *   layers.L.self_attn.index_k_proj   BF16 [128, 6144] (1 index k head  * 128, MQA)
 *   layers.L.self_attn.index_q_norm   BF16 [128]
 *   layers.L.self_attn.index_k_norm   BF16 [128]
 *   layers.L.block_sparse_moe.gate.weight             F32  [128, 6144] (router)
 *   layers.L.block_sparse_moe.e_score_correction_bias F32  [128]       (routing bias)
 *   layers.L.block_sparse_moe.experts.N.w1/w3         BF16 [3072, 6144]
 *   layers.L.block_sparse_moe.experts.N.w2            BF16 [6144, 3072]
 *   layers.L.block_sparse_moe.shared_experts.gate/up_proj BF16 [3072, 6144]
 *   layers.L.block_sparse_moe.shared_experts.down_proj    BF16 [6144, 3072]
 *   layers.0..2.mlp.gate/up_proj  BF16 [12288, 6144]   (dense layers)
 *   layers.0..2.mlp.down_proj     BF16 [6144, 12288]
 *
 * SCOPE (bring-up, see a64fx/m3/m3.md): text-only (no vision tower / multimodal
 * projector / MTP), bf16 experts (reuse matvec_bf16_8row_pv), full KV cache,
 * correctness + modest context first. int8/MXFP4 experts, MSA-cache CP sharding,
 * and 1M context are deferred perf phases.
 */
#ifndef M3_H
#define M3_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <arm_sve.h>

#include "ggml_dequant.h"   /* matvec_bf16_8row_pv + e8m0/fp8 helpers (shared with ds4f) */

/* ===================== config ===================== */
typedef struct {
    int n_layers;        /* 60 */
    int n_dense_layers;  /* 3   (layers 0..2 dense FFN + full attention; 3..59 MoE + MSA) */
    int hidden;          /* 6144 */
    int vocab;           /* 200064 */
    float norm_eps;      /* 1e-6  (RMSNorm) */
    int use_gemma_norm;  /* 1 => RMSNorm weight applied as x*(1+w) */
    /* GQA attention */
    int n_heads;         /* 64  query heads */
    int n_kv_heads;      /* 4   key/value heads (GQA group = 16) */
    int head_dim;        /* 128 */
    int use_qk_norm;     /* 1 => per-head RMSNorm on q and k (q_norm/k_norm [head_dim]) */
    /* partial RoPE: rotary_dim of head_dim dims rotated, the rest pass through */
    int   rotary_dim;    /* 64  (== head_dim * partial_rotary_factor 0.5) */
    float rope_theta;    /* 5e6 */
    /* MiniMax Sparse Attention (MSA), layers >= n_dense_layers */
    int   msa_index_dim;     /* 128  (index_q/k head dim) */
    int   msa_n_index_heads; /* 4    (index_q_proj rows / index_dim; index_k is MQA, 1 head) */
    int   msa_block_size;    /* 128 */
    int   msa_topk_blocks;   /* 16 */
    int   msa_init_block;    /* 0  (always-attend first N blocks; 0 here => none forced) */
    int   msa_local_block;   /* 1  (always-attend the most recent N blocks) */
    /* MoE */
    int   n_experts;         /* 128 */
    int   n_active;          /* 4   experts per token */
    int   n_shared;          /* 1 */
    int   moe_inter;         /* 3072  routed + shared expert intermediate */
    int   dense_inter;       /* 12288 dense-layer (0..2) FFN intermediate */
    float routed_scale;      /* 2.0 */
    /* SwiGLU-OAI activation: out = (x_up clamped to [-lim,lim]) * silu_oai(x_gate)
     *   silu_oai(g) = g_c * sigmoid(alpha * g_c),  g_c = min(g, lim)   (no lower clamp on gate) */
    float swiglu_alpha;      /* 1.702 */
    float swiglu_limit;      /* 7.0 */
    /* runtime */
    int max_pos;             /* KV cache capacity */
} m3_config;

static inline m3_config m3_default_config(void) {
    m3_config c = {0};
    c.n_layers = 60; c.n_dense_layers = 3; c.hidden = 6144; c.vocab = 200064;
    c.norm_eps = 1e-6f; c.use_gemma_norm = 1;
    c.n_heads = 64; c.n_kv_heads = 4; c.head_dim = 128; c.use_qk_norm = 1;
    c.rotary_dim = 64; c.rope_theta = 5000000.0f;
    c.msa_index_dim = 128; c.msa_n_index_heads = 4; c.msa_block_size = 128;
    c.msa_topk_blocks = 16; c.msa_init_block = 0; c.msa_local_block = 1;
    c.n_experts = 128; c.n_active = 4; c.n_shared = 1;
    c.moe_inter = 3072; c.dense_inter = 12288; c.routed_scale = 2.0f;
    c.swiglu_alpha = 1.702f; c.swiglu_limit = 7.0f;
    c.max_pos = 8192;
    return c;
}

/* M3_LAYERS / M3_MAXPOS env overrides applied by the runner, not here. */
static inline m3_config m3_config_from_env(void) { return m3_default_config(); }

/* layer kind */
static inline int m3_is_moe(const m3_config *c, int L)    { return L >= c->n_dense_layers; }
static inline int m3_is_sparse(const m3_config *c, int L) { return L >= c->n_dense_layers; }

/* derived dims */
static inline int m3_q_dim(const m3_config *c)  { return c->n_heads    * c->head_dim; }  /* 8192 */
static inline int m3_kv_dim(const m3_config *c) { return c->n_kv_heads * c->head_dim; }  /* 512  */
static inline int m3_idx_q_dim(const m3_config *c) { return c->msa_n_index_heads * c->msa_index_dim; } /* 512 */

/* ===================== tensor ===================== */
/* Bring-up uses bf16 weights only. M3_BF16_PV is the pair-interleaved layout
 * consumed by matvec_bf16_8row_pv (+22..28%, byte-identical) — used for the
 * dominant matvecs (q/k/v/o, experts, shared, dense FFN, head). M3_BF16 is the
 * flat layout for norms/embed/index-norm read directly. M3_F32 for the router
 * gate + e_score bias (argmax-critical, kept high precision). MXFP4/Q8 reserved
 * for the later perf phase (same enum values as ds4f for kernel sharing). */
typedef enum { M3_BF16 = 0, M3_FP8 = 1, M3_MXFP4 = 2, M3_F32 = 3, M3_BF16_PV = 4, M3_Q8_PV = 5 } m3_qtype;

typedef struct {
    void    *w;       /* weight bytes */
    uint8_t *scale;   /* E8M0 scale bytes (NULL for BF16/F32) */
    m3_qtype type;
    int rows, cols;   /* logical [rows, cols] */
} m3_tensor;

static inline size_t m3_wbytes(m3_qtype t, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    switch (t) {
        case M3_BF16:    return n * 2;
        case M3_BF16_PV: return n * 2;
        case M3_FP8:     return n;
        case M3_MXFP4:   return n / 2;
        case M3_F32:     return n * 4;
        case M3_Q8_PV:   return (size_t)(rows / 8) * (cols / 64) * 528;
    }
    return 0;
}
static inline size_t m3_sbytes(m3_qtype t, int rows, int cols) {
    switch (t) {
        case M3_FP8:   return (size_t)((rows + 127) / 128) * ((cols + 127) / 128);
        case M3_MXFP4: return (size_t)rows * (cols / 32);
        default:       return 0;
    }
}

/* ===================== tensor parallelism (mandatory at 96 nodes) =====================
 * Replicated dense alone (embed+head ~5 GB, attn q/o ~12 GB, shared expert ~6.4 GB) is
 * ~25 GB/rank; with ~13 GB of owned experts that overflows the 32 GB node. So the same
 * TP set as DS4P is mandatory: shard the row dimension of the big projections evenly
 * across the EP ranks (each rank also being a TP rank). Even split with remainder going
 * to the low ranks — bit-exact reconstruct via the EP all-reduce (sum for column-shard
 * outputs, gather for row-shard). See a64fx/m3/m3.md for which matvec each shard cuts. */
static inline void m3_shard(int total, int rank, int size, int *r0, int *rows) {
    int base = total / size, rem = total % size;
    int lo = rank < rem ? rank * (base + 1) : rem * (base + 1) + (rank - rem) * base;
    int n  = base + (rank < rem ? 1 : 0);
    *r0 = lo; *rows = n;
}
/* head/index heads must shard whole (head_dim contiguous): shard the head COUNT. */
static inline void m3_shard_heads(int n_heads, int rank, int size, int *h0, int *h1) {
    int r0, rows; m3_shard(n_heads, rank, size, &r0, &rows); *h0 = r0; *h1 = r0 + rows;
}

/* experts: e owned by rank (e % size). 128 experts / 96 ranks => 32 ranks own 2, 64 own 1. */
static inline int m3_n_owned(int n_experts, int ep_rank, int ep_size) {
    int n = 0; for (int e = 0; e < n_experts; e++) if (e % ep_size == ep_rank) n++; return n;
}

/* ===================== layer / model ===================== */
typedef struct {
    /* norms (BF16 [hidden], Gemma: applied as x*(1+w)) */
    uint16_t *input_norm;     /* input_layernorm */
    uint16_t *post_norm;      /* post_attention_layernorm */
    /* GQA attention (BF16_PV projections; q_norm/k_norm flat BF16 [head_dim]) */
    m3_tensor wq, wk, wv, wo; /* [q_dim,H] [kv_dim,H] [kv_dim,H] [H,q_dim] */
    uint16_t *q_norm, *k_norm;/* [head_dim] per-head RMSNorm */
    /* TP_ATTN: this rank's owned query heads [qh0, qh1); wq holds only their rows,
     * wo holds only their input columns. kv heads + index are REPLICATED (small). */
    int qh0, qh1;
    /* MSA indexer (sparse layers only; NULL on dense layers) */
    m3_tensor idx_wq, idx_wk; /* [idx_q_dim,H]=[512,H]  [index_dim,H]=[128,H] (MQA) */
    uint16_t *idx_q_norm, *idx_k_norm; /* [index_dim] */
    /* MoE (sparse layers) — router f32, experts/shared bf16_pv */
    m3_tensor gate;           /* F32 [n_experts, hidden] */
    float    *gate_bias;      /* F32 [n_experts] e_score_correction_bias */
    m3_tensor sh_w1, sh_w3, sh_w2;    /* shared expert (gate_proj/up_proj/down_proj) */
    m3_tensor *ex_w1, *ex_w3, *ex_w2; /* owned routed experts, 0..n_owned-1 */
    int      *owned_eid;      /* global expert id of each owned slot */
    int       n_owned;
    /* TP_SHARED: shared-expert intermediate shard [sh_r0, sh_r0+sh_rows) for w1/w3
     * (w2 input-sharded correspondingly, output EP-summed). */
    int sh_r0, sh_rows;
    /* Dense FFN (layers 0..2; NULL on MoE layers) — intermediate 12288, SwiGLU-OAI.
     * TP-sharded on the intermediate dim [ff_r0, ff_r0+ff_rows). */
    m3_tensor ff_gate, ff_up, ff_down;
    int ff_r0, ff_rows;
    /* per-layer KV cache: GQA keeps 4 kv heads x head_dim per position, bf16.
     * Full attention (dense layers) + MSA both read this; MSA additionally selects
     * blocks via the indexer. cmp/index caches for long-ctx CP are a later phase. */
    uint16_t *k_cache;        /* [max_pos, kv_dim] bf16 */
    uint16_t *v_cache;        /* [max_pos, kv_dim] bf16 */
    /* MSA index-key cache: per-position index key [index_dim] bf16 (1 MQA head),
     * post index_k_norm + RoPE, used to score blocks. NULL on dense layers. */
    uint16_t *idx_k_cache;    /* [max_pos, index_dim] bf16 */
} m3_layer;

typedef struct m3_pool m3_pool;

typedef struct {
    m3_config cfg;
    int ep_rank, ep_size;
    m3_layer *layers;
    /* embeddings / head (BF16; TP vocab-sharded) */
    uint16_t *embed;          /* [emb_rows, hidden] this rank's vocab shard */
    int emb_r0, emb_rows;
    m3_tensor head;           /* BF16_PV [head_rows, hidden] this rank's vocab shard */
    int head_r0;              /* global vocab offset of head shard (head.rows = shard rows) */
    uint16_t *out_norm;       /* final norm [hidden] */
    /* RoPE tables (partial rotary, single theta): cos/sin[pos*half + k], half=rotary_dim/2 */
    float *rope_cos, *rope_sin;
    /* dense-weight type used for the bf16 matvec path (M3_BF16 or M3_BF16_PV) */
    m3_qtype bf16_mv_qt;
    /* arena */
    uint8_t *arena; size_t arena_sz, arena_used;
    /* pool */
    m3_pool *pool;
    int n_threads, n_cmgs;
    /* scratch (per-forward, single token) */
    float *s_norm, *s_q, *s_k, *s_v, *s_attn, *s_o;
    float *s_idx_q, *s_idx_k;          /* MSA index projections */
    float *s_blk_score; int *s_blk_sel; int s_blk_nsel;  /* per-block scores + selected block ids */
    float *s_router, *s_shg, *s_shu, *s_sh, *s_exg, *s_exu, *s_moe;
    float *s_route;                    /* routed-expert partial (owned-only); EP-summed via ar_cb */
    float *s_ff_g, *s_ff_u, *s_ff;     /* dense FFN scratch */
    float *s_logits;
    /* EP combine hook: sum the routed-expert partial [hidden] across the EP group
     * (== tp_allreduce_sum). Also used to gather TP column-shard partials. NULL => single node. */
    void  (*ar_cb)(float *buf, int count, void *ctx);
    void   *ar_ctx;
    void  (*ar_argmax_cb)(float *val, int32_t *idx, void *ctx);  /* TP_HEAD argmax merge */
    void   *ar_argmax_ctx;
    /* perf accounting */
    size_t bytes_read;
    double prof[16];
} m3_model;

/* phase ids for m3_model.prof[] */
enum { M3_P_QKV=0, M3_P_QKNORM=1, M3_P_ROPE=2, M3_P_MSA_INDEX=3, M3_P_ATTN=4,
       M3_P_OPROJ=5, M3_P_ROUTER=6, M3_P_EXPERTS=7, M3_P_SHARED=8, M3_P_DENSE_FFN=9,
       M3_P_HEAD=10, M3_P_OTHER=11 };
#define M3_NPHASE 16
static const char *m3_prof_names[M3_NPHASE] = {
    "qkv_proj","qk_norm","rope","msa_index","attn","o_proj","router","experts",
    "shared","dense_ffn","head","other","-","-","-","-" };

/* ---- implementation (forward graph, arena, synth/load_real) relocated to m3_impl.h ----
 * NOT YET IMPLEMENTED — Phase 2/3 of a64fx/m3/m3.md. The graph mirrors ds4f_impl.h's
 * structure (arena bump allocator, m3_alloc_synth/m3_load_real, m3_forward_token) but
 * with M3's GQA + MSA + sigmoid-MoE math and the MSA SVE kernels from
 * ~/work/clair/main/tests/a64fx/msa. The thin runner m3_ep_runner.c is a near-copy of
 * ds4f_ep_runner.c (uTofu bootstrap / topo / barriers / EP all-reduce / decode loop). */
#if defined(M3_IMPL)
#include "m3_impl.h"
#endif

#endif /* M3_H */
