/*
 * glm5.h - GLM-5.2 (text LM) inference harness for A64FX/Fugaku.
 *
 * GLM-5.2 is a 744B-param / ~40B active sparse-MoE model, 1.5 TB bf16,
 * 78 main decoder layers plus one next-N layer. This header carries
 * the verified architecture config + the per-rank tensor/layer/model structs for
 * a custom expert-parallel (EP) + tensor-parallel (TP) runner on A64FX, reusing
 * the DS4P stack's infrastructure (ggml_dequant SVE matvec kernels, the uTofu EP
 * all-reduce, the parallel weight stager, the pjsub job patterns).
 *
 * GLM5 differs from DeepSeek-V4 in the FORWARD GRAPH (so it is a new graph, not a
 * ds4f config switch):
 *   - attention      = MLA: q_lora 2048, kv_lora 512, qk_nope 192, rope 64, v 256
 *   - sparse attn    = DSA indexer top-2048 over index dim 128. Bring-up uses full
 *                      causal attention for short contexts where pos <= index_topk.
 *   - MoE            = sigmoid+bias router, top-8 of 256, 1 shared expert,
 *                      routed_scaling 2.5, SiLU-SwiGLU
 *   - RoPE           = interleaved rotary on the 64 rope dims, theta 1e6
 *   - norm           = standard RMSNorm x*w
 *   - layers 0..2    = DENSE (mlp intermediate 12288, full attention, no MSA/MoE)
 *   - layers 3..77   = MoE + DSA
 *   - dtype          = bf16 throughout except router bias f32
 *
 * Verified against ~/models/glm5.2/config.json + safetensors headers (2026-06-18):
 *   embed_tokens / lm_head        BF16 [154880, 6144]
 *   q_a/q_b                      BF16 [2048,6144] / [16384,2048]
 *   kv_a_with_mqa / kv_b          BF16 [576,6144] / [28672,512]
 *   o_proj                        BF16 [6144,16384]
 *   dense FFN                     BF16 [12288,6144] x2 / [6144,12288]
 *   MoE experts                   BF16 [2048,6144] x2 / [6144,2048], 256 experts
 *
 * SCOPE (bring-up, see a64fx/glm5/glm5.md): text-only (no vision tower / multimodal
 * projector / MTP), bf16 experts (reuse matvec_bf16_8row_pv), full KV cache,
 * correctness + modest context first. int8/MXFP4 experts, MSA-cache CP sharding,
 * and 1M context are deferred perf phases.
 */
#ifndef GLM5_H
#define GLM5_H

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
#include "glm5_mem.h"         /* glm5_amalloc/glm5_acalloc/glm5_afree: 256-aligned NUMA-interleaved */

/* ===================== config ===================== */
typedef struct {
    int n_layers;        /* 78 main decoder layers; skip layer 78 next-N in bring-up */
    int n_dense_layers;  /* 3   (layers 0..2 dense FFN; 3..77 MoE) */
    int hidden;          /* 6144 */
    int vocab;           /* 154880 */
    float norm_eps;      /* 1e-5  (RMSNorm) */
    int use_gemma_norm;  /* 0 => standard RMSNorm x*w */
    /* MLA attention */
    int n_heads;         /* 64  query heads */
    int n_kv_heads;      /* compatibility: GLM MLA uses 64 logical kv heads */
    int head_dim;        /* compatibility: use qk_head_dim for old helpers */
    int use_qk_norm;     /* compatibility */
    int qk_head_dim;      /* 256 = qk_nope + qk_rope */
    int qk_nope_dim;      /* 192 */
    int qk_rope_dim;      /* 64 */
    int v_head_dim;       /* 256 */
    int q_lora;           /* 2048 */
    int kv_lora;          /* 512 */
    float rope_theta;     /* 8e6 */
    /* DSA indexer */
    int   index_dim;      /* 128 */
    int   index_n_heads;  /* 32 */
    int   index_topk;     /* 2048 */
    int   index_topk_freq; /* 4: full indexer at 0..2, then every fourth layer */
    int   rotary_dim;     /* compatibility alias for qk_rope_dim */
    int   msa_index_dim, msa_n_index_heads, msa_block_size, msa_topk_blocks, msa_init_block, msa_local_block;
    /* MoE */
    int   n_experts;         /* 256 */
    int   n_active;          /* 8   experts per token */
    int   n_shared;          /* 1 */
    int   moe_inter;         /* 2048  routed + shared expert intermediate */
    int   dense_inter;       /* 12288 dense-layer (0..2) FFN intermediate */
    float routed_scale;      /* 2.5 */
    float swiglu_alpha, swiglu_limit; /* compatibility; GLM uses plain SiLU-SwiGLU */
    /* runtime */
    int max_pos;             /* KV cache capacity */
} glm5_config;

static inline glm5_config glm5_default_config(void) {
    glm5_config c = {0};
    c.n_layers = 78; c.n_dense_layers = 3; c.hidden = 6144; c.vocab = 154880;
    c.norm_eps = 1e-5f; c.use_gemma_norm = 0;
    c.n_heads = 64; c.n_kv_heads = 64; c.head_dim = 256; c.use_qk_norm = 0;
    c.qk_head_dim = 256; c.qk_nope_dim = 192; c.qk_rope_dim = 64;
    c.v_head_dim = 256; c.q_lora = 2048; c.kv_lora = 512; c.rope_theta = 8000000.0f;
    c.index_dim = 128; c.index_n_heads = 32; c.index_topk = 2048; c.index_topk_freq = 4;
    c.rotary_dim = c.qk_rope_dim;
    c.msa_index_dim = c.index_dim; c.msa_n_index_heads = c.index_n_heads;
    c.msa_block_size = 128; c.msa_topk_blocks = 16; c.msa_init_block = 0; c.msa_local_block = 1;
    c.n_experts = 256; c.n_active = 8; c.n_shared = 1;
    c.moe_inter = 2048; c.dense_inter = 12288; c.routed_scale = 2.5f;
    c.swiglu_alpha = 1.0f; c.swiglu_limit = 1.0e30f;
    c.max_pos = 2048;
    return c;
}

/* GLM5_LAYERS / GLM5_MAXPOS env overrides applied by the runner, not here. */
static inline glm5_config glm5_config_from_env(void) { return glm5_default_config(); }

/* layer kind */
static inline int glm5_is_moe(const glm5_config *c, int L)    { return L >= c->n_dense_layers; }
static inline int glm5_is_sparse(const glm5_config *c, int L) { return L >= c->n_dense_layers; }
/* GLM-5.2 stores full DSA indexer tensors only for layers 0..2 and then every
 * fourth layer starting at 6; intervening sparse layers share the previous full
 * layer's top-k selection in the reference implementation. The current Fugaku
 * bring-up keeps GLM5_MSA=0, so shared-indexer reuse is intentionally deferred. */
static inline int glm5_has_full_indexer(const glm5_config *c, int L) {
    if (L < c->n_dense_layers) return 1;
    int first_sparse_full = c->n_dense_layers + c->index_topk_freq - 1;
    return L >= first_sparse_full &&
           ((L - first_sparse_full) % c->index_topk_freq) == 0;
}

/* derived dims */
static inline int glm5_q_dim(const glm5_config *c)  { return c->n_heads * c->qk_head_dim; } /* 16384 */
static inline int glm5_attn_dim(const glm5_config *c) { return c->n_heads * c->v_head_dim; } /* 16384 */
static inline int glm5_kv_cache_dim(const glm5_config *c) { return c->kv_lora + c->qk_rope_dim; } /* 576 */
static inline int glm5_kv_dim(const glm5_config *c) { return glm5_kv_cache_dim(c); }
static inline int glm5_idx_q_dim(const glm5_config *c) { return c->index_n_heads * c->index_dim; } /* 4096 */

/* ===================== tensor ===================== */
/* Bring-up uses bf16 weights only. GLM5_BF16_PV is the pair-interleaved layout
 * consumed by matvec_bf16_8row_pv (+22..28%, byte-identical) — used for the
 * dominant matvecs (q/k/v/o, experts, shared, dense FFN, head). GLM5_BF16 is the
 * flat layout for norms/embed/index-norm read directly. GLM5_F32 for the router
 * gate + e_score bias (argmax-critical, kept high precision). MXFP4/Q8 reserved
 * for the later perf phase (same enum values as ds4f for kernel sharing). */
typedef enum { GLM5_BF16 = 0, GLM5_FP8 = 1, GLM5_MXFP4 = 2, GLM5_F32 = 3, GLM5_BF16_PV = 4, GLM5_Q8_PV = 5, GLM5_MXFP8 = 6 } glm5_qtype;

typedef struct {
    void    *w;       /* weight bytes */
    uint8_t *scale;   /* FP8 scale bytes (GLM5.2: F32 scale_inv blocks; NULL for BF16/F32) */
    glm5_qtype type;
    int rows, cols;   /* logical [rows, cols] */
} glm5_tensor;

static inline size_t glm5_wbytes(glm5_qtype t, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    switch (t) {
        case GLM5_BF16:    return n * 2;
        case GLM5_BF16_PV: return n * 2;
        case GLM5_FP8:     return n;
        case GLM5_MXFP8:   return n;          /* 1 byte/elem (FP8 E4GLM5) */
        case GLM5_MXFP4:   return n / 2;
        case GLM5_F32:     return n * 4;
        case GLM5_Q8_PV:   return (size_t)(rows / 8) * (cols / 64) * 528;
    }
    return 0;
}
static inline size_t glm5_sbytes(glm5_qtype t, int rows, int cols) {
    switch (t) {
        case GLM5_FP8:   return (size_t)((rows + 127) / 128) * ((cols + 127) / 128) * 4;
        case GLM5_MXFP8: return (size_t)((rows + 127) / 128) * ((cols + 127) / 128) * 4;
        case GLM5_MXFP4: return (size_t)rows * (cols / 32);
        default:       return 0;
    }
}

/* ===================== tensor parallelism (mandatory at 96 nodes) =====================
 * Replicated dense alone (embed+head ~5 GB, attn q/o ~12 GB, shared expert ~6.4 GB) is
 * ~25 GB/rank; with ~13 GB of owned experts that overflows the 32 GB node. So the same
 * TP set as DS4P is mandatory: shard the row dimension of the big projections evenly
 * across the EP ranks (each rank also being a TP rank). Even split with remainder going
 * to the low ranks — bit-exact reconstruct via the EP all-reduce (sum for column-shard
 * outputs, gather for row-shard). See a64fx/glm5/glm5.md for which matvec each shard cuts. */
static inline void glm5_shard(int total, int rank, int size, int *r0, int *rows) {
    int base = total / size, rem = total % size;
    int lo = rank < rem ? rank * (base + 1) : rem * (base + 1) + (rank - rem) * base;
    int n  = base + (rank < rem ? 1 : 0);
    *r0 = lo; *rows = n;
}
static inline void glm5_shard_blocks(int total, int block, int rank, int size, int *r0, int *rows) {
    int nb = (total + block - 1) / block;
    int b0, bn; glm5_shard(nb, rank, size, &b0, &bn);
    int lo = b0 * block, hi = (b0 + bn) * block;
    if (lo > total) lo = total;
    if (hi > total) hi = total;
    *r0 = lo; *rows = hi - lo;
}
/* head/index heads must shard whole (head_dim contiguous): shard the head COUNT. */
static inline void glm5_shard_heads(int n_heads, int rank, int size, int *h0, int *h1) {
    int r0, rows; glm5_shard(n_heads, rank, size, &r0, &rows); *h0 = r0; *h1 = r0 + rows;
}

/* experts: e owned by rank (e % size). 256 experts / 192 ranks => 64 ranks own 2, 128 own 1. */
static inline int glm5_n_owned(int n_experts, int ep_rank, int ep_size) {
    int n = 0; for (int e = 0; e < n_experts; e++) if (e % ep_size == ep_rank) n++; return n;
}

/* ===================== layer / model ===================== */
typedef struct {
    /* norms (BF16 [hidden], Gemma: applied as x*(1+w)) */
    uint16_t *input_norm;     /* input_layernorm */
    uint16_t *post_norm;      /* post_attention_layernorm */
    /* MLA attention */
    glm5_tensor wq_a, wq_b, wkv_a, wkv_b, wo;
    uint16_t *q_a_norm, *kv_a_norm;
    /* compatibility aliases for copied optional test/batch helpers */
    glm5_tensor wq, wk, wv;
    uint16_t *q_norm, *k_norm;
    /* TP_ATTN: this rank's owned query heads [qh0, qh1); wq holds only their rows,
     * wo holds only their input columns. kv heads + index are REPLICATED (small). */
    int qh0, qh1;
    /* DSA indexer. Bring-up computes/stores these but full-attends for short context. */
    glm5_tensor idx_wq_b, idx_wk, idx_wproj;
    uint16_t *idx_k_norm, *idx_k_bias;
    glm5_tensor idx_wq;
    uint16_t *idx_q_norm;
    /* MoE (sparse layers) — router f32, experts/shared bf16_pv */
    glm5_tensor gate;           /* F32 [n_experts, hidden] */
    float    *gate_bias;      /* F32 [n_experts] e_score_correction_bias */
    glm5_tensor sh_w1, sh_w3, sh_w2;    /* shared expert (gate_proj/up_proj/down_proj) */
    glm5_tensor *ex_w1, *ex_w3, *ex_w2; /* owned routed experts, 0..n_owned-1 */
    int      *owned_eid;      /* global expert id of each owned slot */
    int       n_owned;
    /* TP_SHARED: shared-expert intermediate shard [sh_r0, sh_r0+sh_rows) for w1/w3
     * (w2 input-sharded correspondingly, output EP-summed). */
    int sh_r0, sh_rows;
    /* Dense FFN (layers 0..2; NULL on MoE layers) — intermediate 12288, SwiGLU-OAI.
     * TP-sharded on the intermediate dim [ff_r0, ff_r0+ff_rows). */
    glm5_tensor ff_gate, ff_up, ff_down;
    int ff_r0, ff_rows;
    /* per-layer KV cache: GQA keeps 4 kv heads x head_dim per position, bf16.
     * Full attention (dense layers) + MSA both read this; MSA additionally selects
     * blocks via the indexer. cmp/index caches for long-ctx CP are a later phase. */
    uint16_t *kv_cache;       /* [max_pos, kv_lora + qk_rope_dim] bf16 */
    uint16_t *k_cache, *v_cache; /* compatibility aliases; unused by GLM path */
    /* MSA index-key cache: per-position index key [index_dim] bf16 (1 MQA head),
     * post index_k_norm + RoPE, used to score blocks. NULL on dense layers. */
    uint16_t *idx_k_cache;    /* [max_pos, index_dim] bf16 */
    /* int4-KV (GLM5_INT4_KV): replaces the bf16 caches above, ~3.9x smaller, for 1M ctx.
     * Per (pos, head) group of head_dim: int4 nibbles + bf16 absmax/7 scale.
     * CP (GLM5_CP): only positions owned by this rank (block b -> rank b%ep_size) are stored;
     * the stride is cp_nslot (<= max_pos) and pos maps to a local slot via glm5_cp_slot(). */
    uint8_t  *k_q4, *v_q4;    /* GLM MLA uses k_q4 as [cp_nslot, latent_kv/2]; v_q4 is legacy */
    uint16_t *k_qs, *v_qs;    /* GLM MLA uses k_qs as [cp_nslot] scale; v_qs is legacy */
    uint8_t  *idx_q4;         /* [cp_nslot, index_dim/2] int4 (sparse layers) */
    uint16_t *idx_qs;         /* [cp_nslot] bf16 scale */
} glm5_layer;

typedef struct glm5_pool glm5_pool;

typedef struct {
    glm5_config cfg;
    int ep_rank, ep_size;
    glm5_layer *layers;
    /* embeddings / head (BF16; TP vocab-sharded) */
    uint16_t *embed;          /* [emb_rows, hidden] this rank's vocab shard */
    int emb_r0, emb_rows;
    glm5_tensor head;           /* BF16_PV [head_rows, hidden] this rank's vocab shard */
    int head_r0;              /* global vocab offset of head shard (head.rows = shard rows) */
    uint16_t *out_norm;       /* final norm [hidden] */
    /* RoPE tables (partial rotary, single theta): cos/sin[pos*half + k], half=rotary_dim/2 */
    float *rope_cos, *rope_sin;
    /* dense-weight type used for the bf16 matvec path (GLM5_BF16 or GLM5_BF16_PV) */
    glm5_qtype bf16_mv_qt;
    uint32_t fp8_lut[256];   /* FP8 E4GLM5 -> f32 bits (built once; used by the MXFP8 matvec) */
    /* arena */
    uint8_t *arena; size_t arena_sz, arena_used;
    int owns_weights;           /* 1 for loaded/synth root model; 0 for runtime KV/scratch clones */
    /* pool */
    glm5_pool *pool;
    int n_threads, n_cmgs;
    /* multi-stream batched decode state (glm5_mstream*, defined in glm5_impl.h); NULL unless allocated */
    void *ms;
    /* int4-KV + context-parallel KV (1M context). int4_kv: caches stored int4 (~3.9x).
     * cp_on: positions sharded across the EP ranks (block b -> rank b%ep_size); each rank
     * stores only cp_nslot owned slots. Decode does a flash-style cross-rank combine of the
     * per-rank partial attention ( kv_combine_cb) + a cross-rank top-k block merge for MSA. */
    int int4_kv, cp_on, cp_nslot, cp_block;
    int kv_fp16;   /* uint16 KV path stores IEEE fp16 (GLM5_KV_FP16) instead of bf16 (quality ref) */
    /* flash-combine of [n_heads*head_dim] partial out + per-head (max,sumexp) across EP ranks.
     * The runner provides a uTofu all-reduce specialized for the online-softmax merge. */
    void  (*kv_combine_cb)(float *out, float *mx, float *sumexp, int n_heads, int head_dim, void *ctx);
    void   *kv_combine_ctx;
    /* all-reduce-MAX of the per-block index scores across EP ranks (each block owned by one
     * rank; non-owned entries are -inf) so every rank derives the same global top-k selection. */
    void  (*blk_reduce_cb)(float *scores, int nblk, void *ctx);
    void   *blk_reduce_ctx;
    /* scratch (per-forward, single token) */
    float *s_norm, *s_q, *s_k, *s_v, *s_kvb, *s_qabs, *s_ctx, *s_attn, *s_o;
    float *s_idx_q, *s_idx_k;          /* MSA index projections */
    float *s_blk_score; int *s_blk_sel; int s_blk_nsel;  /* per-block scores + selected block ids */
    float *s_attn_score;               /* [local_heads, max_pos] decode attention scores */
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
    /* comm-overlap (optional): ar_async_start issues the all-reduce on a comm-driver thread
     * (returns immediately); ar_wait blocks for it. NULL => no overlap (use ar_cb). The
     * batched MoE overlaps the routed-expert reduce with the (replicated) shared-expert compute. */
    void  (*ar_async_start)(float *buf, int count, void *ctx);
    void  (*ar_wait)(void *ctx);
    void   *ar_async_ctx;
    /* perf accounting */
    size_t bytes_read;
    double prof[16];
} glm5_model;

/* phase ids for glm5_model.prof[] */
enum { GLM5_P_QKV=0, GLM5_P_QKNORM=1, GLM5_P_ROPE=2, GLM5_P_MSA_INDEX=3, GLM5_P_ATTN=4,
       GLM5_P_OPROJ=5, GLM5_P_ROUTER=6, GLM5_P_EXPERTS=7, GLM5_P_SHARED=8, GLM5_P_DENSE_FFN=9,
       GLM5_P_HEAD=10, GLM5_P_OTHER=11 };
#define GLM5_NPHASE 16
static const char *glm5_prof_names[GLM5_NPHASE] = {
    "qkv_proj","qk_norm","rope","msa_index","attn","o_proj","router","experts",
    "shared","dense_ffn","head","other","-","-","-","-" };

/* CP slot mapping: block b=pos/cp_block owned by rank b%ep_size; owner stores it at a local
 * slot packed over its owned blocks. When cp_on==0 every rank stores all positions (slot==pos). */
static inline int  glm5_cp_owner(const glm5_model*m,int pos){ return m->cp_on ? (pos/m->cp_block)%m->ep_size : m->ep_rank; }
static inline long glm5_cp_slot (const glm5_model*m,int pos){ if(!m->cp_on) return pos;
    int b=pos/m->cp_block; return (long)(b/m->ep_size)*m->cp_block + (pos%m->cp_block); }
static inline int  glm5_cp_mine (const glm5_model*m,int pos){ return !m->cp_on || (pos/m->cp_block)%m->ep_size==m->ep_rank; }
/* owned-slot capacity for max_pos positions block-cyclic over ep_size ranks */
static inline int  glm5_cp_nslot(int max_pos,int cp_block,int ep_size,int cp_on){
    if(!cp_on) return max_pos; int nblk=(max_pos+cp_block-1)/cp_block;
    return ((nblk+ep_size-1)/ep_size)*cp_block; }

/* ---- implementation (forward graph, arena, synth/load_real) relocated to glm5_impl.h ----
 * NOT YET IMPLEMENTED — Phase 2/3 of a64fx/glm5/glm5.md. The graph mirrors ds4f_impl.h's
 * structure (arena bump allocator, glm5_alloc_synth/glm5_load_real, glm5_forward_token) but
 * with GLM5's GQA + MSA + sigmoid-MoE math and the MSA SVE kernels from
 * ~/work/clair/main/tests/a64fx/msa. The thin runner glm5_ep_runner.c is a near-copy of
 * ds4f_ep_runner.c (uTofu bootstrap / topo / barriers / EP all-reduce / decode loop). */
#if defined(GLM5_IMPL)
#include "glm5_impl.h"
#endif

#endif /* GLM5_H */
