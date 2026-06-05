/*
 * ds4f.h - DeepSeek-V4-Flash synthetic inference harness for A64FX/Fugaku.
 *
 * Perf/plumbing harness: weights are SYNTHETIC (deterministic junk filled in
 * HBM, no disk load), logits are MEANINGLESS. Validates the parallel forward
 * pass shape/FLOPs/throughput/lockstep/memory-fit, NOT output quality.
 *
 * Weights stay quantized in HBM and are dequantized on demand per token:
 *   - dense (MLA attn + shared expert) = FP8 E4M3, 128x128 block E8M0 scale
 *   - routed experts                   = split MXFP4 (e2m1), per-32 E8M0 scale
 *   - embed / head / router / norms    = BF16 ;  attn_sink / mHC = F32
 * Kernels (matvec_fp8e4m3_8row / matvec_mxfp4_8row / matvec_bf16_8row) and the
 * e8m0/fp8 helpers live in ggml_dequant.h (validated by ds4f_kernels_bench.c).
 *
 * Parallelism = expert-parallel (EP): experts e with e%ep_size==ep_rank live
 * on this node (~23/node at ep_size=11); dense is replicated and computed
 * redundantly. Single-node Stage 1 uses ep_size=11/ep_rank=0 (no comm) so it
 * fits ~21 GB; the runner adds the uTofu MoE combine for the 11-node stages.
 *
 * Tensor layouts (from ~/models/ds4f safetensors headers, layer 0):
 *   attn.wq_a   F8_E4M3 [1024,4096]   scale [8,32]
 *   attn.q_norm BF16    [1024]
 *   attn.wq_b   F8_E4M3 [32768,1024]  scale [256,8]      (64 heads x 512)
 *   attn.wkv    F8_E4M3 [512,4096]    scale [4,32]       (kv latent, 1 kv head)
 *   attn.kv_norm BF16   [512]
 *   attn.attn_sink F32  [64]
 *   attn.wo_a   F8_E4M3 [8192,4096]   scale [64,32]
 *   attn.wo_b   F8_E4M3 [4096,8192]   scale [32,64]
 *   ffn.gate.weight BF16 [256,4096]   (router)
 *   ffn.shared_experts.w1/w3 F8 [2048,4096] scale[16,32]; w2 F8 [4096,2048] scale[32,16]
 *   ffn.experts.N.w1/w3 I8 [2048,2048] scale[2048,128];  w2 I8 [4096,1024] scale[4096,64]
 *   hc_attn_* / hc_ffn_* F32 (small, FLOP stand-in)
 */
#ifndef DS4F_H
#define DS4F_H

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

#include "ggml_dequant.h"

/* ===================== config ===================== */
typedef struct {
    int n_layers;       /* 43 */
    int hidden;         /* 4096 */
    int vocab;          /* 129280 */
    /* MLA */
    int n_heads;        /* 64 */
    int q_head_dim;     /* 512  (qk_nope 448 + qk_rope 64) */
    int qk_rope_dim;    /* 64 */
    int q_lora;         /* 1024 */
    int kv_lora;        /* 512  (compressed latent, 1 kv head) */
    int o_inter;        /* 8192 (wo_a out / wo_b in) */
    int o_groups;       /* 8    (32768 -> 4096 reduction before wo_a) */
    /* MoE */
    int n_experts;      /* 256 */
    int n_active;       /* 6 */
    int moe_inter;      /* 2048 */
    int shared_inter;   /* 2048 */
    float routed_scale; /* 1.5 */
    /* mHC (Manifold-Constrained Hyper-Connections): the hidden state is hc_mult
     * residual streams [hc_mult, hidden], not a single vector. Each block collapses
     * 4->1 (hc_pre) before attn/ffn and expands 1->4 (hc_post) after; a final
     * hc_head collapses 4->1 before lm_head. mix_hc = (2+hc_mult)*hc_mult = 24. */
    int hc_mult;        /* 4 */
    int hc_iters;       /* 20 sinkhorn iters */
    float hc_eps;       /* 1e-6 */
    float norm_eps;     /* 1e-6 (RMSNorm + mHC pre-norm) */
    /* sparse lightning indexer (Stage 4): per-layer compress_ratios pick CSA(4)/
     * HCA(128); ratio 0 = dense full attention (layers 0,1,last). The indexer
     * scores nP/ratio compressed blocks over index_head_dim dims, selects the
     * top index_topk original positions, and the expensive weighted-V runs only
     * over them -> O(topk) instead of O(nP) at long context. */
    int index_topk;     /* 512 */
    int index_head_dim; /* 128 */
    int index_n_heads;  /* 64  (Tier-B2 indexer scoring heads) */
    int compress_ratios[64];  /* per-layer; [0..n_layers), 0 = dense */
    /* exact forward math (DS4F_EXACT; the synthetic stand-ins ignore these) */
    int o_lora;             /* 1024 (o_lora_rank; o_inter = o_groups*o_lora) */
    int window_size;        /* 128 sliding-window attention */
    int n_hash_layers;      /* 3 (layers 0..2 route by token id, no bias) */
    float swiglu_limit;     /* 10.0 (clamp up to [-lim,lim], gate to [...,lim]) */
    /* RoPE/YaRN (precompute_freqs_cis): dense layers use rope_theta + no YaRN;
     * sparse layers use compress_rope_theta + YaRN(original_seq_len,factor,beta_*) */
    float rope_theta;            /* 10000 */
    float compress_rope_theta;   /* 160000 */
    int   rope_factor;           /* 16 */
    int   beta_fast, beta_slow;  /* 32, 1 */
    int   original_seq_len;      /* 65536 (YaRN orig ctx; 0 => YaRN off) */
    /* runtime */
    int max_pos;        /* KV cache capacity */
} ds4f_config;

static inline ds4f_config ds4f_default_config(void) {
    ds4f_config c = {0};
    c.n_layers = 43; c.hidden = 4096; c.vocab = 129280;
    c.n_heads = 64; c.q_head_dim = 512; c.qk_rope_dim = 64;
    c.q_lora = 1024; c.kv_lora = 512; c.o_inter = 8192; c.o_groups = 8;
    c.n_experts = 256; c.n_active = 6; c.moe_inter = 2048; c.shared_inter = 2048;
    c.routed_scale = 1.5f; c.max_pos = 4096;
    c.hc_mult = 4; c.hc_iters = 20; c.hc_eps = 1e-6f; c.norm_eps = 1e-6f;
    /* lightning indexer (matches DeepSeek-V4-Flash config.json) */
    c.index_topk = 512; c.index_head_dim = 128; c.index_n_heads = 64;
    /* exact-forward hyperparameters (config.json) */
    c.o_lora = c.o_inter / c.o_groups;   /* 1024 */
    c.window_size = 128; c.n_hash_layers = 3; c.swiglu_limit = 10.0f;
    c.rope_theta = 10000.0f; c.compress_rope_theta = 160000.0f;
    c.rope_factor = 16; c.beta_fast = 32; c.beta_slow = 1; c.original_seq_len = 65536;
    /* compress_ratios: EXACT config.json array (layers 0,1 dense; then CSA(4)/HCA(128)
     * alternating even/odd; index 43 dense=MTP). Picks per-layer RoPE config in exact
     * mode (ratio!=0 => YaRN + compress_rope_theta). Only consulted when sparse/exact. */
    static const int RATIOS[44] = {
        0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,
        4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0 };
    for (int L = 0; L < 64; L++) c.compress_ratios[L] = (L < 44) ? RATIOS[L] : 0;
    return c;
}

/* ===================== tensor ===================== */
/* DS4F_BF16_PV: BF16 weights in the pair-interleaved layout consumed by
 * matvec_bf16_8row_pv (p_odd predicated load, +22..28% over plain bf16,
 * BYTE-IDENTICAL result). Layout is tied to the TYPE so fill and read can
 * never disagree — use it ONLY for tensors read via the pv matvec (head,
 * router, predequant dense), never for flat-read norms/embed. Same bytes as
 * DS4F_BF16, no scale. */
typedef enum { DS4F_BF16 = 0, DS4F_FP8 = 1, DS4F_MXFP4 = 2, DS4F_F32 = 3, DS4F_BF16_PV = 4 } ds4f_qtype;

typedef struct {
    void    *w;       /* weight bytes */
    uint8_t *scale;   /* E8M0 scale bytes (NULL for BF16/F32) */
    ds4f_qtype type;
    int rows, cols;   /* logical [rows, cols] */
} ds4f_tensor;

/* bytes of the weight body for a logical [rows,cols] of the given type */
static inline size_t ds4f_wbytes(ds4f_qtype t, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    switch (t) {
        case DS4F_BF16:    return n * 2;
        case DS4F_BF16_PV: return n * 2;   /* same bytes, pair-interleaved layout */
        case DS4F_FP8:     return n;       /* 1 B/elem */
        case DS4F_MXFP4:   return n / 2;   /* 2 nibbles/byte */
        case DS4F_F32:     return n * 4;
    }
    return 0;
}
static inline size_t ds4f_sbytes(ds4f_qtype t, int rows, int cols) {
    switch (t) {
        case DS4F_FP8:   return (size_t)((rows + 127) / 128) * ((cols + 127) / 128);
        case DS4F_MXFP4: return (size_t)rows * (cols / 32);
        default:         return 0;
    }
}

/* ===================== layer / model ===================== */
typedef struct {
    /* norms (BF16) */
    uint16_t *attn_norm, *ffn_norm, *q_norm, *kv_norm;
    /* MLA (FP8) */
    ds4f_tensor wq_a, wq_b, wkv, wo_a, wo_b;
    float *attn_sink;                 /* [n_heads] F32 */
    /* MoE */
    ds4f_tensor gate;                 /* BF16 [n_experts, hidden] router */
    float *gate_bias;                 /* [n_experts] F32 selection bias (exact, layers>=n_hash); NULL=hash/synth */
    ds4f_tensor sh_w1, sh_w2, sh_w3;  /* shared expert (FP8) */
    ds4f_tensor *ex_w1, *ex_w2, *ex_w3; /* owned experts (MXFP4), indexed 0..n_owned-1 */
    int *owned_eid;                   /* global expert id of each owned slot */
    int  n_owned;
    /* mHC (F32): fn = [mix_hc=24, hc_mult*hidden=16384] Linear; base = [24] bias;
     * scale = [3] (pre/post/comb gates). Exact math in ds4f_hc_pre/post. */
    float *hc_attn_fn, *hc_ffn_fn;    /* [24,16384] */
    float *hc_attn_base, *hc_ffn_base;  /* [24] */
    float *hc_attn_scale, *hc_ffn_scale; /* [3] */
    /* per-layer KV cache: [max_pos, kv_lora] F32 (latent, 1 kv head) */
    float *kv_cache;
    /* ---- Tier-B2 (DS4F_TIERB2; off-arena, only on ratio!=0 layers) ----
     * The stateful compressor/indexer long-range compressed-KV term that Tier-B1
     * omits. The matvec weights are stored bf16 (sources are bf16/FP8-e4m3, both
     * lossless into bf16) and widened in-lane by the pooled SVE kernels — half the
     * bytes, bit-identical to the prior f32-widened arena. ape stays f32, norm bf16.
     * All NULL unless m->tierb2 && compress_ratio!=0. */
    uint16_t *cmp_wkv, *cmp_wgate;   /* layer compressor (rotate=0): [coff*kv_lora, hidden] bf16 */
    float    *cmp_ape;               /* [compress_ratio, coff*kv_lora] */
    uint16_t *cmp_norm;              /* [kv_lora] bf16 */
    float    *cmp_kv_state, *cmp_score_state;  /* [coff*ratio, coff*kv_lora] ring state */
    float    *cmp_kv;                /* [max_pos/ratio, kv_lora] compressed latents */
    /* indexer (CSA layers, compress_ratio==4 only) */
    uint16_t *idx_wq_b;              /* [index_n_heads*index_head_dim, q_lora] bf16 (FP8 src, lossless) */
    uint16_t *idx_wproj;             /* [index_n_heads, hidden] bf16 */
    uint16_t *idx_cmp_wkv, *idx_cmp_wgate;  /* indexer compressor (rotate=1): [coff*index_head_dim, hidden] bf16 */
    float    *idx_cmp_ape;           /* [4, coff*index_head_dim] */
    uint16_t *idx_cmp_norm;          /* [index_head_dim] bf16 */
    float    *idx_cmp_kv_state, *idx_cmp_score_state;  /* [coff*4, coff*index_head_dim] */
    float    *idx_kv;                /* [max_pos/ratio, index_head_dim] indexer compressed */
} ds4f_layer;

typedef struct ds4f_pool ds4f_pool;

typedef struct {
    ds4f_config cfg;
    int ep_rank, ep_size;
    ds4f_layer *layers;
    uint16_t *embed;        /* BF16 [vocab, hidden] (unused in synth decode) */
    ds4f_tensor head;       /* BF16 [vocab, hidden] */
    uint16_t *out_norm;     /* BF16 [hidden] */
    /* global mHC head (collapses the 4 streams 1x before lm_head; NO sinkhorn) */
    float *hc_head_fn;      /* [hc_mult=4, hc_mult*hidden=16384] */
    float *hc_head_base;    /* [hc_mult=4] */
    float *hc_head_scale;   /* [1] */
    uint32_t fp8_lut[256];
    /* dense (MLA + shared-expert) quant type: DS4F_FP8 (on-demand dequant, ~20GB)
     * or DS4F_BF16 (predequant, +5.7GB -> ~26GB, routes the dominant matvecs
     * through the BW-bound bf16 kernel ~400 GB/s instead of the gather-bound
     * fp8 kernel ~70 GB/s). Set via DS4F_FP8_BF16=1. Experts stay MXFP4. */
    ds4f_qtype dense_qt;
    /* FP8 dense decode kernel: 0 = gather (LUT, bit-exact), 1 = magic-multiply
     * (FTZ, ~6 ops/lane, no gather; +2..18% in the HBM-stream decode regime,
     * subnormals flush to 0 -> values ~5e-5 off, fine for the harness). The
     * magic path also enables FTZ on every pool worker. Set via DS4F_FP8_MAGIC=1. */
    int fp8_magic;
    /* BF16 matvec layout: 0 = row-major (matvec_bf16_8row), 1 = pair-interleaved
     * (matvec_bf16_8row_pv, p_odd predicated-load, +22..28% over plain bf16 and
     * BYTE-IDENTICAL — same column order + 8-row svaddv reduction). Applies to
     * ALL DS4F_BF16 matvecs (predequant dense + router + lm-head). Fill writes
     * the interleaved layout to match. Set via DS4F_BF16_PV=1; pairs naturally
     * with DS4F_FP8_BF16=1 to route the dominant dense matvecs through pv. */
    int bf16_pv;
    ds4f_qtype bf16_mv_qt;  /* DS4F_BF16 or DS4F_BF16_PV; for router gate + lm-head */
    /* sparse lightning-indexer attention (Stage 4): 0 = dense full attention all
     * layers (default); 1 = on sparse layers (compress_ratios[L]!=0) with nP>topk,
     * cheap compressed index selects topk positions, weighted-V over them only.
     * Synthetic stand-in (logits meaningless); the point is the O(topk) long-ctx
     * perf model. Set via DS4F_SPARSE=1. */
    int sparse;
    /* exact mHC (Manifold-Constrained Hyper-Connections): 0 = plain residual
     * stand-in (default, byte-identical to the Stage 1-4a path); 1 = carry the
     * hidden state as hc_mult=4 streams and run the EXACT hc_pre/hc_post/hc_head
     * + sinkhorn math (so a real-weight loader yields meaningful logits). The
     * mixes Linear (hc_*_fn) is the only added matvec; collapse/expand are cheap
     * scalar folds. Set via DS4F_MHC=1. */
    int mhc;
    /* exact DeepSeek-V4-Flash forward math (DS4F_EXACT): 0 = the synthetic
     * stand-ins (silu o-proj, no RoPE/q-norm, bias-free gate; default, byte-
     * identical to Stage 1-4b); 1 = the real math (per-head q-norm + RoPE/YaRN on
     * q/kv, sliding-window(128)+sink attention, de-rotate + grouped low-rank
     * o-proj, gate bias-select/unbiased-weight, swiglu clamp). With real weights
     * this yields meaningful logits. Tier-B1 (this) is bit-faithful for
     * dense+HCA layers and the sliding-window term of CSA layers. KNOWN
     * omissions, all deferred to Tier-B2 (need extra plumbing, not just math):
     *   1. stateful compressor/indexer long-range compressed-KV term on sparse
     *      layers -> omits CSA compressed tokens at pos>=4 (needs Hadamard +
     *      FP4 act-quant + raw-hidden history);
     *   2. hash-layer routing (layers < n_hash_layers route by tid2eid[token_id])
     *      -> falls back to score-based topk here (the harness threads a hidden
     *      state, not a token id, per position);
     *   3. act_quant FP8-sim of the kv non-rope dims (a QAT precision detail).
     * RoPE/YaRN, per-head q-norm, sliding-window+sink attention, grouped low-rank
     * o-proj, sqrtsoftplus gate (bias-select/unbiased-weight), and swiglu clamp
     * are exact (validated vs pure-Python ref to 5e-8: ds4f_exact_test.c). */
    int exact;
    /* Tier-B2 stateful sparse attention (DS4F_TIERB2): 0 = Tier-B1 (default, the
     * sliding-window-only term; byte-identical). 1 = ALSO run the per-layer
     * compressor/indexer and fold the long-range compressed-KV term into the same
     * softmax (the omission #1 above). Implies exact (reuses q-norm/RoPE/window).
     * Off OR on a dense layer (ratio==0) == exact path, byte-identical. The decode
     * kernels are bit-validated (ds4f_tierb2_test.c); here they are wired stateful
     * token-at-a-time. Set via DS4F_TIERB2=1. */
    int tierb2;
    /* RoPE freqs tables (only built when exact): cos/sin[pos*half + k],
     * half = qk_rope_dim/2. Two configs: dense (theta, no YaRN) + comp (YaRN). */
    float *rope_dense_cos, *rope_dense_sin;
    float *rope_comp_cos,  *rope_comp_sin;
    /* arena */
    uint8_t *arena; size_t arena_sz, arena_used;
    /* pool */
    ds4f_pool *pool;
    int n_threads, n_cmgs;
    /* scratch (per-forward, single token) */
    float *s_hn, *s_q, *s_qlat, *s_kvlat, *s_attn, *s_oin, *s_o1, *s_o;
    float *s_h2, *s_router, *s_shg, *s_shu, *s_exg, *s_exu, *s_moe, *s_logits;
    float *s_route;         /* routed-expert partial (owned-only); EP-summed via ar_cb */
    /* mHC 4-stream state (only used when m->mhc): x4/resid = [hc_mult*hidden]
     * stream buffers, xc = [hidden] collapsed hc_pre/hc_head output. */
    float *s_x4, *s_resid, *s_xc;
    /* sparse-indexer per-thread scratch: block scores [n_threads * idx_blk_stride]
     * and selected positions [n_threads * index_topk]. Only used when m->sparse. */
    float *s_idx_scores; int *s_idx_sel; int idx_blk_stride;
    /* Tier-B2 per-token scratch (off-arena, only when m->tierb2): compressor output
     * [kv_lora], indexer q [index_n_heads*index_head_dim] + score [max_pos/4],
     * selected LOCAL compressed indices [max_pos/4] + their count for this layer. */
    float *s_cmp_out, *s_idx_q, *s_idx_score; int *s_tb2_sel; int s_tb2_nsel;
    /* EP combine hook: if set, called once per layer on the routed-expert partial
     * [hidden] to sum it across the expert-parallel group (Stage 2 = tp_allreduce_sum).
     * Shared expert stays replicated (added locally). NULL => single-node (all owned). */
    void  (*ar_cb)(float *buf, int count, void *ctx);
    void   *ar_ctx;
    /* perf accounting (weight HBM bytes touched, reset per token by the runner) */
    size_t bytes_read;
    /* per-phase wall-time profiler (seconds, accumulated; printed by runner) */
    double prof[9];
} ds4f_model;

/* phase ids for ds4f_model.prof[] */
#define DS4F_NPHASE 9
enum { DS4F_P_QKV=0, DS4F_P_ATTN=1, DS4F_P_OPROJ=2, DS4F_P_SHARED=3,
       DS4F_P_ROUTER=4, DS4F_P_EXPERTS=5, DS4F_P_HEAD=6, DS4F_P_OTHER=7,
       DS4F_P_TB2PREP=8 };
static const char *ds4f_prof_names[9] = {
    "qkv_proj","attn","o_proj","shared","router","experts","head","other","tb2prep" };

/* ===================== thread pool (pinned, spin) ===================== */
typedef void (*ds4f_fn)(void *arg, int tid, int nthr);

struct ds4f_pool {
    int nthr, n_cmgs;
    pthread_t *threads;
    _Atomic int seq;
    _Atomic int done;
    _Atomic int stop;
    ds4f_fn fn;
    void *arg;
};

#if defined(__aarch64__) && defined(__linux__)
static int ds4f_pin(int tid, int nthr, int n_cmgs) {
    if (nthr < 1) nthr = 1; if (n_cmgs < 1) n_cmgs = 1; if (n_cmgs > 4) n_cmgs = 4;
    int cmg = (int)((long)tid * n_cmgs / nthr);
    int cmg_first = (int)(((long)cmg * nthr + n_cmgs - 1) / n_cmgs);
    int local = tid - cmg_first; if (local < 0) local = 0; if (local > 11) local = 11;
    int core = 12 + cmg * 12 + local;   /* A64FX compute cores 12..59, 12/CMG */
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}
#else
static int ds4f_pin(int t, int n, int c){ (void)t;(void)n;(void)c; return -1; }
#endif

static inline void ds4f_relax(void){ __asm__ __volatile__("yield" ::: "memory"); }

typedef struct { ds4f_pool *p; int tid; } ds4f_wctx;

static void *ds4f_worker(void *v) {
    ds4f_wctx *w = (ds4f_wctx *)v;
    ds4f_pool *p = w->p; int tid = w->tid;
    ds4f_pin(tid, p->nthr, p->n_cmgs);
    int last = 0;
    for (;;) {
        while (atomic_load_explicit(&p->seq, memory_order_acquire) == last &&
               !atomic_load_explicit(&p->stop, memory_order_acquire))
            ds4f_relax();
        if (atomic_load_explicit(&p->stop, memory_order_acquire)) break;
        last = atomic_load_explicit(&p->seq, memory_order_acquire);
        if (p->fn) p->fn(p->arg, tid, p->nthr);
        atomic_fetch_add_explicit(&p->done, 1, memory_order_release);
    }
    free(w);
    return NULL;
}

static ds4f_pool *ds4f_pool_start(int nthr, int n_cmgs) {
    ds4f_pool *p = (ds4f_pool *)calloc(1, sizeof(*p));
    p->nthr = nthr; p->n_cmgs = n_cmgs;
    atomic_store(&p->seq, 0); atomic_store(&p->done, 0); atomic_store(&p->stop, 0);
    p->threads = (pthread_t *)calloc(nthr, sizeof(pthread_t));
    for (int t = 1; t < nthr; t++) {   /* main thread acts as tid 0 */
        ds4f_wctx *w = (ds4f_wctx *)malloc(sizeof(*w));
        w->p = p; w->tid = t;
        pthread_create(&p->threads[t], NULL, ds4f_worker, w);
    }
    ds4f_pin(0, nthr, n_cmgs);          /* pin main to core 12 */
    return p;
}

static void ds4f_pool_run(ds4f_pool *p, ds4f_fn fn, void *arg) {
    p->fn = fn; p->arg = arg;
    atomic_store_explicit(&p->done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->seq, 1, memory_order_release);
    fn(arg, 0, p->nthr);                /* main = tid 0 */
    while (atomic_load_explicit(&p->done, memory_order_acquire) < p->nthr - 1)
        ds4f_relax();
}

static void ds4f_pool_stop(ds4f_pool *p) {
    atomic_store_explicit(&p->stop, 1, memory_order_release);
    for (int t = 1; t < p->nthr; t++) pthread_join(p->threads[t], NULL);
    free(p->threads); free(p);
}

/* row split into 8-aligned blocks for worker tid of nthr */
static inline void ds4f_rowsplit8(int rows, int nthr, int tid, int *r0, int *r1) {
    int blk = (rows + 7) / 8;                 /* number of 8-row groups */
    int per = blk / nthr, extra = blk % nthr;
    int g0 = per * tid + (tid < extra ? tid : extra);
    int g1 = g0 + per + (tid < extra ? 1 : 0);
    *r0 = g0 * 8; *r1 = g1 * 8; if (*r1 > rows) *r1 = rows;
}

/* ===================== helpers ===================== */
static inline float ds4f_bf16(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
static inline uint16_t ds4f_f32_bf16(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }

/* ===================== matvec dispatch ===================== */
typedef struct {
    ds4f_model *m; float *dst; const ds4f_tensor *t; const float *x;
} ds4f_mv_task;

static void ds4f_mv_worker(void *arg, int tid, int nthr) {
    ds4f_mv_task *T = (ds4f_mv_task *)arg;
    const ds4f_tensor *t = T->t; const float *x = T->x; float *dst = T->dst;
    int K = t->cols;
    int r0, r1; ds4f_rowsplit8(t->rows, nthr, tid, &r0, &r1);
    if (t->type == DS4F_BF16) {
        const uint16_t *base = (const uint16_t *)t->w;
        for (int i = r0; i + 7 < r1; i += 8) {
            const uint16_t *w = base + (size_t)i * K;
            matvec_bf16_8row(dst + i, w, w+K, w+2*K, w+3*K, w+4*K, w+5*K, w+6*K, w+7*K, x, K);
        }
    } else if (t->type == DS4F_BF16_PV) {
        /* pair-interleaved: group g (8 rows) stored as 4 pair-bufs of 2K hw
         * each: [pAB(2K) | pCD | pEF | pGH], group stride 8K hw. */
        const uint16_t *base = (const uint16_t *)t->w;
        for (int i = r0; i + 7 < r1; i += 8) {
            const uint16_t *g = base + (size_t)(i / 8) * 8 * K;
            matvec_bf16_8row_pv(dst + i, g, g + 2*K, g + 4*K, g + 6*K, x, K);
        }
    } else if (t->type == DS4F_FP8) {
        const uint8_t *base = (const uint8_t *)t->w;
        int sb_cols = (K + 127) / 128;
        if (T->m->fp8_magic) {
            ds4f_set_ftz();   /* per-worker, idempotent; required by the magic decode */
            for (int i = r0; i + 7 < r1; i += 8) {
                const uint8_t *w = base + (size_t)i * K;
                const uint8_t *es = t->scale + (size_t)(i / 128) * sb_cols;
                matvec_fp8e4m3_8row_magic(dst + i, w, w+K, w+2*K, w+3*K, w+4*K, w+5*K, w+6*K, w+7*K,
                                          es, x, K);
            }
        } else {
            for (int i = r0; i + 7 < r1; i += 8) {
                const uint8_t *w = base + (size_t)i * K;
                const uint8_t *es = t->scale + (size_t)(i / 128) * sb_cols;
                matvec_fp8e4m3_8row(dst + i, w, w+K, w+2*K, w+3*K, w+4*K, w+5*K, w+6*K, w+7*K,
                                    es, T->m->fp8_lut, x, K);
            }
        }
    } else if (t->type == DS4F_MXFP4) { /* split */
        const uint8_t *base = (const uint8_t *)t->w; size_t rb = K / 2;
        const uint8_t *sbase = t->scale; size_t sb = K / 32;
        for (int i = r0; i + 7 < r1; i += 8) {
            const uint8_t *w = base + (size_t)i * rb;
            const uint8_t *s = sbase + (size_t)i * sb;
            matvec_mxfp4_8row(dst + i,
                w, w+rb, w+2*rb, w+3*rb, w+4*rb, w+5*rb, w+6*rb, w+7*rb,
                s, s+sb, s+2*sb, s+3*sb, s+4*sb, s+5*sb, s+6*sb, s+7*sb, x, K);
        }
    } else { /* DS4F_F32: small mHC mixes Linear ([24 or 4] x [hc*hidden]).
                plain rowsplit (rows may be 4, not %8); fcc SVE-reduces the inner. */
        const float *base = (const float *)t->w;
        int rows = t->rows, per = rows / nthr, extra = rows % nthr;
        int f0 = per * tid + (tid < extra ? tid : extra);
        int f1 = f0 + per + (tid < extra ? 1 : 0);
        for (int i = f0; i < f1; i++) {
            const float *w = base + (size_t)i * K;
            float acc = 0.f;
            for (int j = 0; j < K; j++) acc += w[j] * x[j];
            dst[i] = acc;
        }
    }
}

static void ds4f_matvec(ds4f_model *m, float *dst, const ds4f_tensor *t, const float *x) {
    ds4f_mv_task T = { m, dst, t, x };
    m->bytes_read += ds4f_wbytes(t->type, t->rows, t->cols) + ds4f_sbytes(t->type, t->rows, t->cols);
    ds4f_pool_run(m->pool, ds4f_mv_worker, &T);
}

/* a [nrows, cols] row-slice VIEW of tensor t starting at logical row row0 (no copy).
 * row0 must be 8-aligned (kernel 8-row blocking) and, for FP8, 128-aligned (its
 * scale is addressed in 128-row blocks). Used to drive the grouped low-rank
 * o-projection (wo_a is block-diagonal: 8 groups of o_lora rows) as 8 matvecs. */
static inline ds4f_tensor ds4f_row_slice(const ds4f_tensor *t, int row0, int nrows) {
    ds4f_tensor v = *t; v.rows = nrows;
    size_t wbpr;                              /* weight bytes per logical row */
    switch (t->type) {
        case DS4F_FP8:   wbpr = (size_t)t->cols;     break;
        case DS4F_MXFP4: wbpr = (size_t)t->cols / 2; break;
        case DS4F_F32:   wbpr = (size_t)t->cols * 4; break;
        default:         wbpr = (size_t)t->cols * 2; break;   /* BF16 / BF16_PV */
    }
    v.w = (uint8_t *)t->w + (size_t)row0 * wbpr;
    if (t->scale) {
        if (t->type == DS4F_FP8)
            v.scale = t->scale + (size_t)(row0 / 128) * ((t->cols + 127) / 128);
        else if (t->type == DS4F_MXFP4)
            v.scale = t->scale + (size_t)row0 * (t->cols / 32);
    }
    return v;
}

static inline float ds4f_clampf(float x, float lo, float hi){ return x < lo ? lo : (x > hi ? hi : x); }

/* rmsnorm with BF16 weight, in/out f32 [n] */
static void ds4f_rmsnorm(float *out, const float *x, const uint16_t *w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * ds4f_bf16(w[i]);
}

static inline float ds4f_silu(float x){ return x / (1.0f + expf(-x)); }

/* ===================== RoPE / YaRN (exact) =====================
 * Mirrors model.py precompute_freqs_cis + apply_rotary_emb. Builds cos/sin tables
 * [pos*half + k], half = dim/2, freq[k] = base^(-2k/dim). When original_seq_len>0
 * applies the YaRN smooth ramp (low freqs interpolated by 1/factor, high freqs
 * untouched) between the beta_fast/beta_slow correction dims. */
static void ds4f_rope_table(float *cosb, float *sinb, int dim, int max_pos,
                            double base, int factor, int beta_fast, int beta_slow,
                            int original_seq_len) {
    int half = dim / 2;
    double freq[64];                          /* half <= 32 */
    for (int k = 0; k < half; k++) freq[k] = 1.0 / pow(base, (2.0 * k) / dim);
    if (original_seq_len > 0) {
        double lo_d = dim * log((double)original_seq_len / ((double)beta_fast * 2.0 * M_PI)) / (2.0 * log(base));
        double hi_d = dim * log((double)original_seq_len / ((double)beta_slow * 2.0 * M_PI)) / (2.0 * log(base));
        double low = floor(lo_d); if (low < 0) low = 0;
        double high = ceil(hi_d); if (high > dim - 1) high = dim - 1;
        if (low == high) high += 0.001;
        for (int k = 0; k < half; k++) {
            double lin = ((double)k - low) / (high - low);
            double ramp = lin < 0.0 ? 0.0 : (lin > 1.0 ? 1.0 : lin);
            double smooth = 1.0 - ramp;       /* model.py: smooth = 1 - linear_ramp */
            freq[k] = freq[k] / factor * (1.0 - smooth) + freq[k] * smooth;
        }
    }
    for (int p = 0; p < max_pos; p++)
        for (int k = 0; k < half; k++) {
            double ang = (double)p * freq[k];
            cosb[(size_t)p * half + k] = (float)cos(ang);
            sinb[(size_t)p * half + k] = (float)sin(ang);
        }
}

/* apply_rotary_emb on one rope segment v[dim] (consecutive pairs (v[2k],v[2k+1])
 * as a complex), at position pos. inverse uses the conjugate (de-rotation). */
static inline void ds4f_rope_apply(float *v, const float *cosb, const float *sinb,
                                   int pos, int half, int inverse) {
    const float *cs = cosb + (size_t)pos * half, *sn = sinb + (size_t)pos * half;
    for (int k = 0; k < half; k++) {
        float a = v[2 * k], b = v[2 * k + 1], c = cs[k], s = sn[k];
        if (!inverse) { v[2 * k] = a * c - b * s; v[2 * k + 1] = a * s + b * c; }
        else          { v[2 * k] = a * c + b * s; v[2 * k + 1] = -a * s + b * c; }
    }
}

/* build the two RoPE tables (dense + compressed) when exact is on. */
static void ds4f_build_freqs(ds4f_model *m) {
    if (!m->exact) return;
    ds4f_config *c = &m->cfg;
    int dim = c->qk_rope_dim, half = dim / 2, P = c->max_pos;
    size_t n = (size_t)P * half;
    m->rope_dense_cos = (float *)aligned_alloc(64, n * 4);
    m->rope_dense_sin = (float *)aligned_alloc(64, n * 4);
    m->rope_comp_cos  = (float *)aligned_alloc(64, n * 4);
    m->rope_comp_sin  = (float *)aligned_alloc(64, n * 4);
    ds4f_rope_table(m->rope_dense_cos, m->rope_dense_sin, dim, P,
                    c->rope_theta, c->rope_factor, c->beta_fast, c->beta_slow, 0);
    ds4f_rope_table(m->rope_comp_cos, m->rope_comp_sin, dim, P,
                    c->compress_rope_theta, c->rope_factor, c->beta_fast, c->beta_slow,
                    c->original_seq_len);
}

/* ===================== Tier-B2 activation-quant / rotate kernels =====================
 * QAT activation quantizers used by the (Tier-B2) compressor/indexer; mirror kernel.py.
 * In-place, fused quant->dequant, with the model's power-of-2 ("ue8m0") block scale
 *   s = 2^ceil(log2(amax/qmax))   (kernel.py fast_round_scale, exact bit trick).
 * Validated standalone against tools/ds4f_q2_ref.py (a64fx/llm/ds4f_q2_test.c).
 *
 * NOTE on FP8 kv-quant (model.py:506 act_quant(kv[..,:-rd],64,..,inplace=True)): the
 * reference's INPLACE fp8 path casts the snapped value through out_dtype=in_dtype=BF16
 * (kernel.py:86-91) and s is a power of 2, so on already-bf16 kv it is an EXACT no-op —
 * confirmed by model.py:527 ("kv could also use fp8 format, though current implementation
 * uses bf16"). The exact attention path therefore deliberately omits it (Tier-B1 #3): it
 * would not change logits. The FP4 path below is genuinely lossy (1-bit mantissa) and IS
 * applied (indexer q at model.py:414, rotate=True compressor kv at 369-370).
 *
 * These are not yet wired (the Tier-B2 compressor/indexer is pending); kept here as the
 * canonical validated implementations the compressor/indexer will call. */

/* s = 2^ceil(log2(amax * max_inv)) via the IEEE-754 bit trick (kernel.py fast_round_scale).
 * Bit-exact: ceil(log2(t)) = (exp(t)-127) + (mantissa(t)!=0). */
static inline float ds4f_round_scale_pow2(float amax, float max_inv) {
    float t = amax * max_inv;
    uint32_t b; memcpy(&b, &t, 4);
    int e = (int)((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) ? 1 : 0);
    uint32_t sb = (uint32_t)((e + 127) & 0xFF) << 23;            /* 2^e */
    float s; memcpy(&s, &sb, 4); return s;
}

/* RNE round of f to bf16 (8-bit significand). */
static inline float ds4f_bf16_round(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7FFFFFFFu) >= 0x7F800000u) return f;             /* nan/inf passthrough */
    uint32_t r = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;/* round-to-nearest-even */
    memcpy(&f, &r, 4); return f;
}

/* round v (|v|<=6) to nearest float4_e2m1 value, RNE. grid {0,.5,1,1.5,2,3,4,6}. */
static inline float ds4f_fp4_e2m1_snap(float v) {
    static const float g[8]  = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
    static const int   ev[8] = {1,0,1,0,1,0,1,0};               /* mantissa-even flag */
    float sign = v < 0.0f ? -1.0f : 1.0f, a = sign * v, best = g[0];
    float bd = a < 0 ? -a : a; int bi = 0;
    for (int i = 1; i < 8; i++) {
        float d = a - g[i]; if (d < 0) d = -d;
        if (d < bd - 1e-12f || (d <= bd + 1e-12f && ev[i] && !ev[bi])) { bd = d; bi = i; best = g[i]; }
    }
    return sign * best;
}

/* FP4 E2M1 block quant, fused quant->dequant, bf16 output (kernel.py fp4_quant inplace).
 * block divides n. amax floored at 6*2^-126 (kernel.py). */
static inline void ds4f_fp4_act_quant_inplace(float *x, int n, int block) {
    for (int b0 = 0; b0 < n; b0 += block) {
        int bn = (b0 + block <= n) ? block : n - b0;
        float amax = 6.0f * 1.1754944e-38f;                     /* 6 * 2^-126 floor */
        for (int j = 0; j < bn; j++) { float a = x[b0+j] < 0 ? -x[b0+j] : x[b0+j]; if (a > amax) amax = a; }
        float s = ds4f_round_scale_pow2(amax, 1.0f/6.0f), inv = 1.0f/s;
        for (int j = 0; j < bn; j++)
            x[b0+j] = ds4f_bf16_round(ds4f_fp4_e2m1_snap(ds4f_clampf(x[b0+j]*inv, -6.0f, 6.0f)) * s);
    }
}

/* randomized-Hadamard rotate (model.py rotate_activation = hadamard_transform * dim^-0.5).
 * The call applies no random sign -> plain scaled Sylvester FWHT. n must be a power of 2. */
static inline void ds4f_rotate_activation(float *x, int n) {
    for (int h = 1; h < n; h <<= 1)
        for (int i = 0; i < n; i += (h << 1))
            for (int j = i; j < i + h; j++) { float a = x[j], b = x[j+h]; x[j] = a + b; x[j+h] = a - b; }
    float sc = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) x[i] *= sc;
}

/* ===================== Tier-B2 sparse-attention primitives =====================
 * Causal index helpers + the gather/online-softmax sparse attention kernel that
 * the compressor/indexer feed. Standalone (plain pointers) so they validate against
 * tools/ds4f_tierb2_ref.py (a64fx/llm/ds4f_tierb2_test.c) and the forward path wraps
 * them. Mirror model.py get_window/get_compress_topk_idxs + kernel.py sparse_attn. */

/* get_window_topk_idxs (prefill, start_pos==0): valid sliding-window positions for
 * query s. Fills row[0..wq) (wq=min(seqlen,window)); -1 = masked slot. Returns wq. */
static inline int ds4f_window_idx_prefill(int window, int seqlen, int s, int *row) {
    int wq = seqlen < window ? seqlen : window;
    int b = s - window + 1; if (b < 0) b = 0;
    for (int c = 0; c < wq; c++) { int v = b + c; row[c] = (v > s) ? -1 : v; }
    return wq;
}

/* get_compress_topk_idxs (prefill, start_pos==0): compressed positions for query s.
 * col t covers raw [t*ratio,(t+1)*ratio); query s may attend t < (s+1)/ratio. The
 * (s+1)/ratio threshold = arange(1,seqlen+1)//ratio. +offset shifts into the combined
 * kv (window positions occupy [0,offset)). Fills row[0..ncol); -1 = masked. */
static inline int ds4f_compress_idx_prefill(int ratio, int seqlen, int offset, int s, int *row) {
    int ncol = seqlen / ratio, thr = (s + 1) / ratio;
    for (int t = 0; t < ncol; t++) row[t] = (t >= thr) ? -1 : (t + offset);
    return ncol;
}

/* sparse_attn (kernel.py sparse_attn_kernel): for each (query s, head hd) gather the
 * topk kv positions named by topk_idxs[s] (-1 = skip), score q.kv*scale, online-softmax
 * with attn_sink[hd] added to the denominator only, weighted-V. q[m*h*d], kv[n*d],
 * sink[h], topk_idxs[m*topk], out o[m*h*d]. d = head_dim (kv is both K and V latent).
 * Math identical to the Tier-B1 window worker, generalized to gathered indices. */
static void ds4f_sparse_attn(const float *q, const float *kv, const float *sink,
                             const int *topk_idxs, int m, int h, int d, int topk,
                             float scale, float *o) {
    float *sc = (float *)alloca((size_t)topk * sizeof(float));
    int   *vi = (int   *)alloca((size_t)topk * sizeof(int));
    for (int s = 0; s < m; s++) {
        const int *idxr = topk_idxs + (size_t)s * topk;
        for (int hd = 0; hd < h; hd++) {
            const float *qd = q + ((size_t)s * h + hd) * d;
            int nv = 0; float mx = -1e30f;
            for (int t = 0; t < topk; t++) {
                int idx = idxr[t]; if (idx < 0) continue;
                const float *kvr = kv + (size_t)idx * d;
                float acc = 0.f;
                for (int dd = 0; dd < d; dd++) acc += qd[dd] * kvr[dd];
                float sv = acc * scale; sc[nv] = sv; vi[nv] = idx; nv++;
                if (sv > mx) mx = sv;
            }
            float denom = expf(sink[hd] - mx);
            for (int k = 0; k < nv; k++) { float e = expf(sc[k] - mx); sc[k] = e; denom += e; }
            float inv = 1.0f / denom;
            float *od = o + ((size_t)s * h + hd) * d;
            for (int dd = 0; dd < d; dd++) od[dd] = 0.f;
            for (int k = 0; k < nv; k++) {
                float w = sc[k] * inv; const float *kvr = kv + (size_t)vi[k] * d;
                for (int dd = 0; dd < d; dd++) od[dd] += w * kvr[dd];
            }
        }
    }
}

/* ===================== Tier-B2 compressor (prefill) =====================
 * model.py Compressor.forward, start_pos==0: gated-pool x over `ratio` consecutive
 * tokens into one compressed kv latent per window. The gate is PER-DIMENSION — for
 * each output dim e the `P` sub-position values are combined with a softmax computed
 * from the P score values at that same dim e (score.softmax(dim=2), kv*that, sum).
 *
 *   overlap (ratio==4, coff=2): P=2*ratio sub-positions. wkv/wgate emit 2*d dims;
 *     dims [0,d)="overlap" half, [d,2d)="normal" half + per-slot bias ape[r][.].
 *     compressed window w pools: sub p in [0,ratio) <- PREV window's overlap half
 *     (raw pos (w-1)*ratio+p, dims [0,d)); sub p in [ratio,2ratio) <- CUR window's
 *     normal half (raw pos w*ratio+(p-ratio), dims [d,2d)). w==0's prev half is
 *     score=-inf => weight 0 (overlap_transform fill).
 *   non-overlap (ratio!=4, coff=1): P=ratio, wkv/wgate emit d dims, window w pools
 *     raw pos w*ratio+p over all d.
 * Then RMSNorm(d), RoPE last rd dims at raw pos w*ratio. rotate=1 (indexer compressor):
 * rotate_activation+fp4 over the d-vector. rotate=0 (layer compressor): the fp8-on-nope
 * act_quant is a bf16 no-op (see header note) -> omitted. Remainder tokens feed decode
 * state only (not the prefill compressed output of nwin=seqlen/ratio latents).
 *
 * x[seqlen*dim], wkv/wgate[(coff*d)*dim] row-major(out,in), ape[ratio*(coff*d)],
 * norm_w bf16[d], rcos/rsin rope tables[pos*(rd/2)+k], out[nwin*d]. */
static void ds4f_compress_prefill(
    const float *x, int seqlen, int dim, int d, int rd, int ratio,
    const float *wkv, const float *wgate, const float *ape,
    const uint16_t *norm_w, const float *rcos, const float *rsin,
    float eps, int rotate, float *out)
{
    int overlap = (ratio == 4), coff = overlap ? 2 : 1, W = coff * d;
    int cutoff = seqlen - seqlen % ratio, nwin = cutoff / ratio;
    if (nwin <= 0) return;
    float *kvl = (float *)malloc((size_t)cutoff * W * sizeof(float));
    float *scl = (float *)malloc((size_t)cutoff * W * sizeof(float));
    for (int pos = 0; pos < cutoff; pos++) {                  /* wkv/wgate linear */
        const float *xp = x + (size_t)pos * dim;
        for (int o = 0; o < W; o++) {
            const float *wk = wkv + (size_t)o * dim, *wg = wgate + (size_t)o * dim;
            float a = 0.f, b = 0.f;
            for (int i = 0; i < dim; i++) { a += wk[i] * xp[i]; b += wg[i] * xp[i]; }
            kvl[(size_t)pos * W + o] = a; scl[(size_t)pos * W + o] = b;
        }
    }
    int P = overlap ? 2 * ratio : ratio;
    float *ksub = (float *)alloca((size_t)P * sizeof(float));
    float *ssub = (float *)alloca((size_t)P * sizeof(float));
    for (int w = 0; w < nwin; w++) {
        float *ow = out + (size_t)w * d;
        for (int e = 0; e < d; e++) {
            if (overlap) {
                for (int p = 0; p < ratio; p++) {            /* prev-window overlap half */
                    if (w == 0) { ksub[p] = 0.f; ssub[p] = -1e30f; }
                    else {
                        int pos = (w - 1) * ratio + p;
                        ksub[p] = kvl[(size_t)pos * W + e];
                        ssub[p] = scl[(size_t)pos * W + e] + ape[(size_t)p * W + e];
                    }
                }
                for (int r = 0; r < ratio; r++) {            /* cur-window normal half */
                    int pos = w * ratio + r, p = ratio + r;
                    ksub[p] = kvl[(size_t)pos * W + d + e];
                    ssub[p] = scl[(size_t)pos * W + d + e] + ape[(size_t)r * W + d + e];
                }
            } else {
                for (int p = 0; p < ratio; p++) {
                    int pos = w * ratio + p;
                    ksub[p] = kvl[(size_t)pos * W + e];
                    ssub[p] = scl[(size_t)pos * W + e] + ape[(size_t)p * W + e];
                }
            }
            float mx = -1e30f; for (int p = 0; p < P; p++) if (ssub[p] > mx) mx = ssub[p];
            float den = 0.f; for (int p = 0; p < P; p++) { float ex = expf(ssub[p] - mx); ssub[p] = ex; den += ex; }
            float acc = 0.f; for (int p = 0; p < P; p++) acc += ksub[p] * (ssub[p] / den);
            ow[e] = acc;
        }
        ds4f_rmsnorm(ow, ow, norm_w, d, eps);                /* norm over d */
        ds4f_rope_apply(ow + (d - rd), rcos, rsin, w * ratio, rd / 2, 0);  /* rope @ w*ratio */
        if (rotate) { ds4f_rotate_activation(ow, d); ds4f_fp4_act_quant_inplace(ow, d, 32); }
    }
    free(kvl); free(scl);
}

/* ===================== Tier-B2 pooled+SVE matvec workers =====================
 * The compressor/indexer linear projections are the Tier-B2 decode bottleneck
 * (single-threaded scalar f32 = 94-97% of per-token time at ctx>=256). These
 * pooled SVE workers replace the inner triple-loops when ds4f_tb2_prepare hands
 * a live pool down (decode forward); the standalone correctness test passes
 * pool==NULL and keeps the original serial scalar order bit-exact. SVE tree
 * reduction reorders the sum vs scalar => results differ at the ~1e-6 ULP level,
 * which is why the validated path stays serial. cols are multiples of 16 (svcntw
 * on A64FX), but the whilelt tail keeps these correct for any width. */
typedef struct { float *out; const float *w, *x; int rows, cols; } ds4f_f32mv_task;
static void ds4f_f32mv_worker(void *arg, int tid, int nthr) {
    ds4f_f32mv_task *T = (ds4f_f32mv_task *)arg;
    int rows = T->rows, cols = T->cols, vl = (int)svcntw();
    int per = rows / nthr, extra = rows % nthr;
    int o0 = per * tid + (tid < extra ? tid : extra);
    int o1 = o0 + per + (tid < extra ? 1 : 0);
    const float *x = T->x;
    for (int o = o0; o < o1; o++) {
        const float *w = T->w + (size_t)o * cols;
        svfloat32_t acc = svdup_f32(0.f);
        for (int i = 0; i < cols; i += vl) {
            svbool_t pg = svwhilelt_b32(i, cols);
            acc = svmla_f32_x(pg, acc, svld1(pg, w + i), svld1(pg, x + i));
        }
        T->out[o] = svaddv_f32(svptrue_b32(), acc);
    }
}

/* two outputs sharing one x load: kv[o]=wkv[o].x, score[o]=wgate[o].x (compressor) */
typedef struct { float *kv, *score; const float *wkv, *wgate, *x; int W, dim; } ds4f_cmpmv_task;
static void ds4f_cmpmv_worker(void *arg, int tid, int nthr) {
    ds4f_cmpmv_task *T = (ds4f_cmpmv_task *)arg;
    int W = T->W, dim = T->dim, vl = (int)svcntw();
    int per = W / nthr, extra = W % nthr;
    int o0 = per * tid + (tid < extra ? tid : extra);
    int o1 = o0 + per + (tid < extra ? 1 : 0);
    const float *x = T->x;
    for (int o = o0; o < o1; o++) {
        const float *wk = T->wkv + (size_t)o * dim, *wg = T->wgate + (size_t)o * dim;
        svfloat32_t a = svdup_f32(0.f), b = svdup_f32(0.f);
        for (int i = 0; i < dim; i += vl) {
            svbool_t pg = svwhilelt_b32(i, dim);
            svfloat32_t xv = svld1(pg, x + i);
            a = svmla_f32_x(pg, a, svld1(pg, wk + i), xv);
            b = svmla_f32_x(pg, b, svld1(pg, wg + i), xv);
        }
        svbool_t pt = svptrue_b32();
        T->kv[o] = svaddv_f32(pt, a); T->score[o] = svaddv_f32(pt, b);
    }
}

/* ---- bf16-weight variants (same accumulation shape, weight widened in-lane) ----
 * The compressor/indexer weights are stored bf16 (their sources are bf16/FP8-e4m3,
 * both of which fit bf16 losslessly), so widen(stored_bf16) == the f32-widened value
 * BIT-EXACTLY: svld1uh zero-extends the halfword, <<16 reconstructs the f32 the f32
 * path would have loaded => the svmla inputs (and thus svaddv) are bit-identical to
 * ds4f_f32mv_worker / ds4f_cmpmv_worker. Half the weight bytes for identical output. */
typedef struct { float *out; const uint16_t *w; const float *x; int rows, cols; } ds4f_bf16mv_task;
static void ds4f_bf16mv_worker(void *arg, int tid, int nthr) {
    ds4f_bf16mv_task *T = (ds4f_bf16mv_task *)arg;
    int rows = T->rows, cols = T->cols, vl = (int)svcntw();
    int per = rows / nthr, extra = rows % nthr;
    int o0 = per * tid + (tid < extra ? tid : extra);
    int o1 = o0 + per + (tid < extra ? 1 : 0);
    const float *x = T->x;
    for (int o = o0; o < o1; o++) {
        const uint16_t *w = T->w + (size_t)o * cols;
        svfloat32_t acc = svdup_f32(0.f);
        for (int i = 0; i < cols; i += vl) {
            svbool_t pg = svwhilelt_b32(i, cols);
            svfloat32_t wf = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, w + i), 16));
            acc = svmla_f32_x(pg, acc, wf, svld1(pg, x + i));
        }
        T->out[o] = svaddv_f32(svptrue_b32(), acc);
    }
}

typedef struct { float *kv, *score; const uint16_t *wkv, *wgate; const float *x; int W, dim; } ds4f_cmpmv_bf16_task;
static void ds4f_cmpmv_bf16_worker(void *arg, int tid, int nthr) {
    ds4f_cmpmv_bf16_task *T = (ds4f_cmpmv_bf16_task *)arg;
    int W = T->W, dim = T->dim, vl = (int)svcntw();
    int per = W / nthr, extra = W % nthr;
    int o0 = per * tid + (tid < extra ? tid : extra);
    int o1 = o0 + per + (tid < extra ? 1 : 0);
    const float *x = T->x;
    for (int o = o0; o < o1; o++) {
        const uint16_t *wk = T->wkv + (size_t)o * dim, *wg = T->wgate + (size_t)o * dim;
        svfloat32_t a = svdup_f32(0.f), b = svdup_f32(0.f);
        for (int i = 0; i < dim; i += vl) {
            svbool_t pg = svwhilelt_b32(i, dim);
            svfloat32_t xv = svld1(pg, x + i);
            svfloat32_t wkf = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wk + i), 16));
            svfloat32_t wgf = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wg + i), 16));
            a = svmla_f32_x(pg, a, wkf, xv);
            b = svmla_f32_x(pg, b, wgf, xv);
        }
        svbool_t pt = svptrue_b32();
        T->kv[o] = svaddv_f32(pt, a); T->score[o] = svaddv_f32(pt, b);
    }
}

/* index_score parallelized over compressed positions t: score[t] = sum_h
 * relu(q[h].kvc[t]) * weights[h]. q[H*hd], kvc[T*hd], weights[H]. */
typedef struct { const float *q, *kvc, *weights; int H, hd, T; float *score; } ds4f_idxsc_task;
static void ds4f_idxsc_worker(void *arg, int tid, int nthr) {
    ds4f_idxsc_task *Tk = (ds4f_idxsc_task *)arg;
    int Tn = Tk->T, H = Tk->H, hd = Tk->hd, vl = (int)svcntw();
    int per = Tn / nthr, extra = Tn % nthr;
    int t0 = per * tid + (tid < extra ? tid : extra);
    int t1 = t0 + per + (tid < extra ? 1 : 0);
    for (int t = t0; t < t1; t++) {
        const float *kt = Tk->kvc + (size_t)t * hd;
        float acc = 0.f;
        for (int h = 0; h < H; h++) {
            const float *qh = Tk->q + (size_t)h * hd;
            svfloat32_t d = svdup_f32(0.f);
            for (int x = 0; x < hd; x += vl) {
                svbool_t pg = svwhilelt_b32(x, hd);
                d = svmla_f32_x(pg, d, svld1(pg, qh + x), svld1(pg, kt + x));
            }
            float dot = svaddv_f32(svptrue_b32(), d);
            if (dot < 0.f) dot = 0.f;
            acc += dot * Tk->weights[h];
        }
        Tk->score[t] = acc;
    }
}

/* ===================== Tier-B2 indexer scoring + top-k =====================
 * model.py Indexer.forward scoring (after q is wq_b-projected, RoPE'd, rotate+fp4'd
 * and the compressor has filled kvc): index_score[t] = sum_h relu(q[h].kvc[t]) *
 * weights[h]. q[H*hd] (one query's heads, contiguous), kvc[T*hd], weights[H]. */
static void ds4f_index_score(const float *q, const float *kvc, const float *weights,
                             int H, int hd, int T, float *score, ds4f_pool *pool) {
    if (pool && T >= 64) {
        ds4f_idxsc_task tk = { q, kvc, weights, H, hd, T, score };
        ds4f_pool_run(pool, ds4f_idxsc_worker, &tk);
        return;
    }
    for (int t = 0; t < T; t++) {
        const float *kt = kvc + (size_t)t * hd;
        float acc = 0.f;
        for (int h = 0; h < H; h++) {
            const float *qh = q + (size_t)h * hd;
            float dot = 0.f;
            for (int d = 0; d < hd; d++) dot += qh[d] * kt[d];
            if (dot < 0.f) dot = 0.f;                       /* relu */
            acc += dot * weights[h];
        }
        score[t] = acc;
    }
}

/* masked top-k compressed-position selection (model.py Indexer topk + prefill causal
 * mask): only t < thr are valid (thr = (query+1)/ratio); pick min(k,thr) by score
 * (descending), write (t+offset) into sel[0..k) SORTED ASCENDING, pad with -1. The
 * gather (sparse_attn) is order-invariant, so sorted output validates the SELECTED SET
 * without depending on topk's unspecified tie order. */
static void ds4f_index_topk(const float *score, int T, int thr, int k, int offset, int *sel) {
    if (thr > T) thr = T;
    int npick = k < thr ? k : thr;
    int  *chosen = (int  *)alloca((size_t)(npick > 0 ? npick : 1) * sizeof(int));
    char *used   = (char *)alloca((size_t)(thr > 0 ? thr : 1));
    for (int t = 0; t < thr; t++) used[t] = 0;
    for (int n = 0; n < npick; n++) {
        int best = -1; float bv = -1e30f;
        for (int t = 0; t < thr; t++) if (!used[t] && score[t] > bv) { bv = score[t]; best = t; }
        used[best] = 1; chosen[n] = best;
    }
    for (int a = 0; a < npick; a++)                          /* sort ascending */
        for (int b = a + 1; b < npick; b++)
            if (chosen[b] < chosen[a]) { int tmp = chosen[a]; chosen[a] = chosen[b]; chosen[b] = tmp; }
    for (int n = 0; n < k; n++) sel[n] = (n < npick) ? (chosen[n] + offset) : -1;
}

/* ===================== Tier-B2 stateful decode =====================
 * The token-at-a-time forward (ds4f_forward_token) runs the INCREMENTAL compressor:
 * pos==0 is the seqlen==1 special case of model.py Compressor's start_pos==0 branch
 * (seeds the ring state, no compressed token), pos>=1 is the start_pos>0 decode branch.
 * The kv_state/score_state ring buffers persist across calls (one set per compressor).
 * Validated bit-exact vs the incremental ref in tools/ds4f_tierb2_ref.py — the same
 * compressed tokens the batched ds4f_compress_prefill would emit, produced one at a time.
 *
 * State shapes (model.py Compressor.__init__): kv_state[coff*ratio, W], score_state same,
 * W = coff*head_dim, coff = 1+overlap, overlap = (ratio==4). overlap: rows [0,ratio) hold
 * the previous (overlapping) window, [ratio,2ratio) the current; non-overlap: rows [0,ratio).
 * reset = kv_state 0, score_state -1e30 (= model.py's -inf for the softmax: exp(-1e30-mx)->0). */
static inline void ds4f_compress_state_reset(float *kv_state, float *score_state, int ratio, int d) {
    int overlap = (ratio == 4), coff = overlap ? 2 : 1, W = coff * d, rows = coff * ratio;
    for (size_t i = 0; i < (size_t)rows * W; i++) { kv_state[i] = 0.f; score_state[i] = -1e30f; }
}

/* One incremental compressor step for the current token x[dim] at absolute `start_pos`.
 * Updates kv_state/score_state in place; on a compress boundary fills out[d] with the new
 * compressed latent (RMSNorm + RoPE @ first-token-of-block + optional rotate/fp4) and
 * returns 1; otherwise returns 0 (out untouched). Mirrors model.py Compressor.forward
 * seqlen==1: start_pos==0 seeds, start_pos>0 decodes. rotate=1 => indexer compressor. */
static int ds4f_compress_step(
    const float *x, int dim, int d, int rd, int ratio, int start_pos,
    const void *wkv, const void *wgate, int w_bf16, const float *ape, const uint16_t *norm_w,
    const float *rcos, const float *rsin, float eps, int rotate,
    float *kv_state, float *score_state, float *out, ds4f_pool *pool)
{
    int overlap = (ratio == 4), coff = overlap ? 2 : 1, W = coff * d;
    float *kv = (float *)alloca((size_t)W * 4), *score = (float *)alloca((size_t)W * 4);
    if (pool && w_bf16) {                                   /* pooled SVE, bf16 weights */
        ds4f_cmpmv_bf16_task ct = { kv, score, (const uint16_t *)wkv, (const uint16_t *)wgate, x, W, dim };
        ds4f_pool_run(pool, ds4f_cmpmv_bf16_worker, &ct);
    } else if (pool) {                                      /* pooled SVE, f32 weights */
        ds4f_cmpmv_task ct = { kv, score, (const float *)wkv, (const float *)wgate, x, W, dim };
        ds4f_pool_run(pool, ds4f_cmpmv_worker, &ct);
    } else                                                  /* serial f32 (validation/test) */
    for (int o = 0; o < W; o++) {                            /* wkv/wgate linear */
        const float *wk = (const float *)wkv + (size_t)o * dim, *wg = (const float *)wgate + (size_t)o * dim;
        float a = 0.f, b = 0.f;
        for (int i = 0; i < dim; i++) { a += wk[i] * x[i]; b += wg[i] * x[i]; }
        kv[o] = a; score[o] = b;
    }
    if (start_pos == 0) {                                    /* seqlen==1 seed (no compress) */
        int offset = overlap ? ratio : 0;                   /* remainder=1 slot */
        for (int o = 0; o < W; o++) {
            kv_state[(size_t)offset * W + o] = kv[o];
            score_state[(size_t)offset * W + o] = score[o] + ape[o];   /* + ape[0] */
        }
        return 0;
    }
    int should = ((start_pos + 1) % ratio) == 0;
    int apr = start_pos % ratio;
    for (int o = 0; o < W; o++) score[o] += ape[(size_t)apr * W + o];  /* score += ape[start_pos%ratio] */
    int P = overlap ? 2 * ratio : ratio;
    float *ksub = (float *)alloca((size_t)P * 4), *ssub = (float *)alloca((size_t)P * 4);
    if (overlap) {
        int slot = ratio + apr;
        for (int o = 0; o < W; o++) { kv_state[(size_t)slot*W+o] = kv[o]; score_state[(size_t)slot*W+o] = score[o]; }
        if (should) {
            for (int e = 0; e < d; e++) {                   /* cat([:ratio,:d],[ratio:,d:]) over rows, per col e */
                for (int p = 0; p < ratio; p++) { ksub[p] = kv_state[(size_t)p*W+e];       ssub[p] = score_state[(size_t)p*W+e]; }
                for (int p = 0; p < ratio; p++) { ksub[ratio+p] = kv_state[(size_t)(ratio+p)*W+d+e]; ssub[ratio+p] = score_state[(size_t)(ratio+p)*W+d+e]; }
                float mx = -1e30f; for (int p = 0; p < P; p++) if (ssub[p] > mx) mx = ssub[p];
                float den = 0.f; for (int p = 0; p < P; p++) { float ex = expf(ssub[p]-mx); ssub[p] = ex; den += ex; }
                float acc = 0.f; for (int p = 0; p < P; p++) acc += ksub[p] * (ssub[p]/den);
                out[e] = acc;
            }
            for (int p = 0; p < ratio; p++)                 /* shift current window -> overlap window */
                for (int o = 0; o < W; o++) {
                    kv_state[(size_t)p*W+o] = kv_state[(size_t)(ratio+p)*W+o];
                    score_state[(size_t)p*W+o] = score_state[(size_t)(ratio+p)*W+o];
                }
        }
    } else {
        int slot = apr;
        for (int o = 0; o < W; o++) { kv_state[(size_t)slot*W+o] = kv[o]; score_state[(size_t)slot*W+o] = score[o]; }
        if (should) {
            for (int e = 0; e < d; e++) {                   /* W==d; softmax over ratio rows */
                for (int p = 0; p < ratio; p++) { ksub[p] = kv_state[(size_t)p*W+e]; ssub[p] = score_state[(size_t)p*W+e]; }
                float mx = -1e30f; for (int p = 0; p < P; p++) if (ssub[p] > mx) mx = ssub[p];
                float den = 0.f; for (int p = 0; p < P; p++) { float ex = expf(ssub[p]-mx); ssub[p] = ex; den += ex; }
                float acc = 0.f; for (int p = 0; p < P; p++) acc += ksub[p] * (ssub[p]/den);
                out[e] = acc;
            }
        }
    }
    if (!should) return 0;
    ds4f_rmsnorm(out, out, norm_w, d, eps);
    ds4f_rope_apply(out + (d - rd), rcos, rsin, start_pos + 1 - ratio, rd / 2, 0);  /* first token of block */
    if (rotate) { ds4f_rotate_activation(out, d); ds4f_fp4_act_quant_inplace(out, d, 32); }
    return 1;
}

/* get_window_topk_idxs (decode, seqlen==1, start_pos>0): window-ring SLOT indices, newest
 * window in chronological order, -1 for not-yet-filled. Always fills `window` columns. */
static inline int ds4f_window_idx_decode(int window, int start_pos, int *row) {
    if (start_pos >= window - 1) {
        int sp = start_pos % window, c = 0;
        for (int v = sp + 1; v < window; v++) row[c++] = v;
        for (int v = 0; v <= sp; v++) row[c++] = v;
    } else {
        int c = 0;
        for (int v = 0; v <= start_pos; v++) row[c++] = v;
        for (; c < window; c++) row[c] = -1;
    }
    return window;
}

/* get_compress_topk_idxs (decode, seqlen==1, start_pos>0): arange(0,(start_pos+1)//ratio)+offset.
 * Used by HCA(128) layers (no indexer). Returns the count n; fills row[0..n). */
static inline int ds4f_compress_idx_decode(int ratio, int start_pos, int offset, int *row) {
    int n = (start_pos + 1) / ratio;
    for (int t = 0; t < n; t++) row[t] = t + offset;
    return n;
}

/* Indexer decode step (model.py Indexer.forward, seqlen==1, start_pos>0): project q via wq_b,
 * per-head RoPE(last rd)+rotate+fp4; step the OWN (rotate) compressor (fills idx_kv_cache);
 * weights = weights_proj(x) * (hd^-0.5 * H^-0.5); index_score over idx_kv_cache[:end//ratio];
 * select top-min(k,T) compressed positions (+offset). q_scr[H*hd], score_scr[>=T], sel[k]. */
static int ds4f_index_step(
    const float *x, int dim, const float *qr, int qlora,
    int H, int hd, int rd, int ratio, int start_pos, int offset, int k,
    const void *wq_b, const void *weights_proj, int w_bf16,
    const void *cwkv, const void *cwgate, const float *cape, const uint16_t *cnorm,
    const float *rcos, const float *rsin, float eps,
    float *comp_kv_state, float *comp_score_state, float *idx_kv_cache,
    float *q_scr, float *score_scr, int *sel, ds4f_pool *pool)
{
    int end_pos = start_pos + 1, half = rd / 2;
    if (pool && w_bf16) {                                    /* q = wq_b(qr), pooled bf16 */
        ds4f_bf16mv_task qt = { q_scr, (const uint16_t *)wq_b, qr, H * hd, qlora };
        ds4f_pool_run(pool, ds4f_bf16mv_worker, &qt);
    } else if (pool) {                                       /* q = wq_b(qr), pooled f32 */
        ds4f_f32mv_task qt = { q_scr, (const float *)wq_b, qr, H * hd, qlora };
        ds4f_pool_run(pool, ds4f_f32mv_worker, &qt);
    } else                                                   /* serial f32 (validation/test) */
    for (int o = 0; o < H * hd; o++) {                       /* q = wq_b(qr) */
        const float *w = (const float *)wq_b + (size_t)o * qlora; float a = 0.f;
        for (int i = 0; i < qlora; i++) a += w[i] * qr[i];
        q_scr[o] = a;
    }
    for (int h = 0; h < H; h++) {                            /* RoPE + rotate + fp4 per head */
        float *qh = q_scr + (size_t)h * hd;
        ds4f_rope_apply(qh + (hd - rd), rcos, rsin, start_pos, half, 0);
        ds4f_rotate_activation(qh, hd);
        ds4f_fp4_act_quant_inplace(qh, hd, 32);
    }
    float *comp_out = (float *)alloca((size_t)hd * 4);       /* own compressor (rotate=1) */
    if (ds4f_compress_step(x, dim, hd, rd, ratio, start_pos, cwkv, cwgate, w_bf16, cape, cnorm,
                           rcos, rsin, eps, 1, comp_kv_state, comp_score_state, comp_out, pool))
        memcpy(idx_kv_cache + (size_t)(start_pos / ratio) * hd, comp_out, (size_t)hd * 4);
    float sm = (float)(1.0 / sqrt((double)hd)), wscale = sm * (float)(1.0 / sqrt((double)H));
    float *weights = (float *)alloca((size_t)H * 4);
    if (pool && w_bf16) {                                    /* weights = weights_proj(x), pooled bf16 */
        ds4f_bf16mv_task wt = { weights, (const uint16_t *)weights_proj, x, H, dim };
        ds4f_pool_run(pool, ds4f_bf16mv_worker, &wt);
        for (int h = 0; h < H; h++) weights[h] *= wscale;
    } else if (pool) {                                       /* weights = weights_proj(x), pooled f32 */
        ds4f_f32mv_task wt = { weights, (const float *)weights_proj, x, H, dim };
        ds4f_pool_run(pool, ds4f_f32mv_worker, &wt);
        for (int h = 0; h < H; h++) weights[h] *= wscale;
    } else                                                   /* serial f32 (validation/test) */
    for (int h = 0; h < H; h++) {
        const float *w = (const float *)weights_proj + (size_t)h * dim; float a = 0.f;
        for (int i = 0; i < dim; i++) a += w[i] * x[i];
        weights[h] = a * wscale;
    }
    int T = end_pos / ratio;
    ds4f_index_score(q_scr, idx_kv_cache, weights, H, hd, T, score_scr, pool);
    ds4f_index_topk(score_scr, T, T, k, offset, sel);        /* decode: no causal mask (thr=T) */
    return T;
}

/* ===================== synthetic allocator ===================== */
static inline int ds4f_n_owned(int n_experts, int ep_rank, int ep_size) {
    int c = 0; for (int e = 0; e < n_experts; e++) if (e % ep_size == ep_rank) c++; return c;
}

static size_t ds4f_arena_size(const ds4f_config *c, int ep_rank, int ep_size, int dense_bf16) {
    size_t pad = 256; /* per-tensor alignment slack */
    ds4f_qtype dq = dense_bf16 ? DS4F_BF16 : DS4F_FP8;
    int no = ds4f_n_owned(c->n_experts, ep_rank, ep_size);
    size_t per_layer = 0;
    per_layer += (size_t)(c->hidden*2 + c->hidden*2 + c->q_lora*2 + c->kv_lora*2) + 4*pad;
    /* MLA dense (FP8 on-demand, or BF16 predequant) */
    per_layer += ds4f_wbytes(dq, c->q_lora, c->hidden) + ds4f_sbytes(dq, c->q_lora, c->hidden) + 2*pad;
    per_layer += ds4f_wbytes(dq, c->n_heads*c->q_head_dim, c->q_lora) + ds4f_sbytes(dq, c->n_heads*c->q_head_dim, c->q_lora) + 2*pad;
    per_layer += ds4f_wbytes(dq, c->kv_lora, c->hidden) + ds4f_sbytes(dq, c->kv_lora, c->hidden) + 2*pad;
    per_layer += ds4f_wbytes(dq, c->o_inter, c->hidden) + ds4f_sbytes(dq, c->o_inter, c->hidden) + 2*pad;
    per_layer += ds4f_wbytes(dq, c->hidden, c->o_inter) + ds4f_sbytes(dq, c->hidden, c->o_inter) + 2*pad;
    per_layer += (size_t)c->n_heads*4 + pad;
    /* MoE */
    per_layer += ds4f_wbytes(DS4F_BF16, c->n_experts, c->hidden) + pad;            /* router */
    per_layer += 2*(ds4f_wbytes(dq, c->shared_inter, c->hidden) + ds4f_sbytes(dq, c->shared_inter, c->hidden)) + 4*pad;
    per_layer += ds4f_wbytes(dq, c->hidden, c->shared_inter) + ds4f_sbytes(dq, c->hidden, c->shared_inter) + 2*pad;
    size_t per_ex = ds4f_wbytes(DS4F_MXFP4, c->moe_inter, c->hidden) + ds4f_sbytes(DS4F_MXFP4, c->moe_inter, c->hidden)
                  + ds4f_wbytes(DS4F_MXFP4, c->hidden, c->moe_inter) + ds4f_sbytes(DS4F_MXFP4, c->hidden, c->moe_inter)
                  + ds4f_wbytes(DS4F_MXFP4, c->moe_inter, c->hidden) + ds4f_sbytes(DS4F_MXFP4, c->moe_inter, c->hidden) + 6*pad;
    per_layer += (size_t)no * per_ex;
    {   int hc = c->hc_mult, mix = (2+hc)*hc, hd = hc*c->hidden;
        per_layer += 2*((size_t)mix*hd*4 + (size_t)mix*4 + 3*4) + 6*pad;            /* hc_attn/ffn fn+base+scale */
    }
    per_layer += (size_t)c->max_pos * c->kv_lora * 4 + pad;                         /* kv cache */

    size_t total = per_layer * c->n_layers;
    total += ds4f_wbytes(DS4F_BF16, c->vocab, c->hidden) + pad;                     /* embed */
    total += ds4f_wbytes(DS4F_BF16, c->vocab, c->hidden) + pad;                     /* head */
    total += (size_t)c->hidden*2 + pad;                                            /* out_norm */
    {   int hc = c->hc_mult, hd = hc*c->hidden;
        total += (size_t)hc*hd*4 + (size_t)hc*4 + 4 + 3*pad;                        /* hc_head fn+base+scale */
    }
    total += 64u*1024*1024;                                                        /* slack */
    return total;
}

static void *ds4f_bump(ds4f_model *m, size_t bytes, size_t align) {
    size_t off = (m->arena_used + align - 1) & ~(align - 1);
    if (off + bytes > m->arena_sz) {
        fprintf(stderr, "ds4f arena overflow: need %zu have %zu\n", off + bytes, m->arena_sz);
        abort();
    }
    void *p = m->arena + off; m->arena_used = off + bytes; return p;
}

/* type-safe deterministic fill LUTs (avoid FP8-NaN / E8M0-NaN, keep magnitudes small) */
static const uint16_t ds4f_bf16_fill[16] = { /* small ~±0.5..±0.06 bf16 */
    0x3F00,0xBE80,0x3E00,0xBD80,0x3D00,0xBC80,0x3C00,0xBC00,
    0x3B80,0xBB00,0x3B00,0xBA80,0x3A80,0xBA00,0x3A00,0xB980 };
static const uint8_t ds4f_fp8_fill[16] = { /* E4M3 exp 4..6, signed (zero-mean), never exp==15 */
    0x20,0xA8,0x30,0xB8,0x21,0xA9,0x31,0xB9,0x22,0xAA,0x32,0xBA,0x23,0xAB,0x33,0xBB };
#define DS4F_E8M0_ONE 127   /* 2^0 = 1.0 */

typedef struct { ds4f_tensor t; } ds4f_fill_task;

static void ds4f_fill_worker(void *arg, int tid, int nthr) {
    ds4f_tensor *t = &((ds4f_fill_task *)arg)->t;
    int r0, r1; ds4f_rowsplit8(t->rows, nthr, tid, &r0, &r1);
    int K = t->cols;
    if (t->type == DS4F_BF16) {
        uint16_t *w = (uint16_t *)t->w;
        for (int i = r0; i < r1; i++) for (int j = 0; j < K; j++) w[(size_t)i*K+j] = ds4f_bf16_fill[(i+j)&15];
    } else if (t->type == DS4F_BF16_PV) {
        /* pair-interleaved layout (see matvec_bf16_8row_pv): group g of 8 rows
         * stored as [pAB | pCD | pEF | pGH], each pair-buf 2K hw holding two
         * rows interleaved (pair[2c]=rowA[c], pair[2c+1]=rowB[c]). r0/r1 are
         * 8-aligned, so each row's pair-buf slot is well-defined. Logical W[i][j]
         * is filled with the SAME value as DS4F_BF16, just at the pv address. */
        uint16_t *w = (uint16_t *)t->w;
        for (int i = r0; i < r1; i++) {
            size_t gbase = (size_t)(i / 8) * 8 * K;     /* group base in hw */
            int local = i & 7, pair = local >> 1, slot = local & 1;
            uint16_t *pb = w + gbase + (size_t)pair * 2 * K;
            for (int j = 0; j < K; j++) pb[2*j + slot] = ds4f_bf16_fill[(i+j)&15];
        }
    } else if (t->type == DS4F_FP8) {
        uint8_t *w = (uint8_t *)t->w; int sbc = (K+127)/128;
        for (int i = r0; i < r1; i++) for (int j = 0; j < K; j++) w[(size_t)i*K+j] = ds4f_fp8_fill[(i+j)&15];
        for (int i = r0/128; i <= (r1-1)/128 && i < (t->rows+127)/128; i++)
            for (int j = 0; j < sbc; j++) t->scale[(size_t)i*sbc+j] = DS4F_E8M0_ONE;
    } else if (t->type == DS4F_MXFP4) {
        uint8_t *w = (uint8_t *)t->w; size_t rb = K/2, sb = K/32;
        for (int i = r0; i < r1; i++) {
            for (size_t j = 0; j < rb; j++) w[(size_t)i*rb+j] = (uint8_t)((i*131+j*17) & 0xff);
            for (size_t j = 0; j < sb; j++) t->scale[(size_t)i*sb+j] = DS4F_E8M0_ONE;
        }
    } else if (t->type == DS4F_F32) {
        /* small deterministic fill for the mHC mixes Linear; identical integer
         * hash in the pure-Python reference (ds4f_mhc_ref.py). Plain rowsplit so
         * hc_head_fn's 4 rows are covered (rowsplit8 would zero them). */
        float *w = (float *)t->w;
        int per = t->rows / nthr, extra = t->rows % nthr;
        int f0 = per*tid + (tid<extra?tid:extra), f1 = f0 + per + (tid<extra?1:0);
        for (int i = f0; i < f1; i++)
            for (int j = 0; j < K; j++)
                w[(size_t)i*K+j] = (float)((((i*131 + j*17) % 97) - 48)) * (0.02f/48.0f);
    }
}

static void ds4f_fill(ds4f_model *m, ds4f_tensor t) {
    ds4f_fill_task ft; ft.t = t;
    ds4f_pool_run(m->pool, ds4f_fill_worker, &ft);
}

/* deterministic small fill for mHC base[nbase] / scale[nscale] (the tiny F32
 * bias/gate params). Same integer hash as ds4f_mhc_ref.py so the C model and the
 * pure-Python reference share weights. base ~ [-0.1,0.1]; scale ~ 0.5,0.6,0.7. */
static void ds4f_hc_fill_meta(float *base, int nbase, float *scale, int nscale, int seed) {
    for (int j = 0; j < nbase; j++)
        base[j] = (float)((((j + seed)*13) % 17) - 8) * (0.1f/8.0f);
    for (int s = 0; s < nscale; s++) scale[s] = 0.5f + 0.1f*s;
}

/* allocate one quantized tensor in the arena and fill it (parallel first-touch) */
static ds4f_tensor ds4f_new_tensor(ds4f_model *m, ds4f_qtype type, int rows, int cols) {
    ds4f_tensor t; t.type = type; t.rows = rows; t.cols = cols;
    t.w = ds4f_bump(m, ds4f_wbytes(type, rows, cols), 256);
    size_t sb = ds4f_sbytes(type, rows, cols);
    t.scale = sb ? (uint8_t *)ds4f_bump(m, sb, 64) : NULL;
    return t;
}

/* Tier-B2 off-arena allocation (+ optional synth fill) of the per-layer compressor/
 * indexer float weights, ring state, and compressed-KV caches. The decode kernels
 * take plain float weights, so these live OUTSIDE the quantized arena (calloc) — and
 * therefore do not perturb the arena bump offsets, so enabling Tier-B2 leaves every
 * synthetic dense weight/scratch byte-identical (the off==on proof at ratio==0).
 * fill!=0 => same deterministic F32/BF16 junk fill as the synth path (bounded by the
 * in-kernel RMSNorm); fill==0 => zeroed, for ds4f_load_real to overwrite by name.
 * Only ratio!=0 layers allocate; the indexer is CSA(ratio==4)-only. Gated by caller. */
static void ds4f_alloc_tb2(ds4f_model *m, int fill) {
    ds4f_config *c = &m->cfg;
    int C = c->hidden, KV = c->kv_lora, ihd = c->index_head_dim, iH = c->index_n_heads;
    int qlora = c->q_lora, np = c->max_pos;
    /* model-level per-token scratch (allocated once) */
    m->s_cmp_out   = (float *)aligned_alloc(256, (size_t)KV*4);
    m->s_idx_q     = (float *)aligned_alloc(256, (size_t)iH*ihd*4);
    m->s_idx_score = (float *)aligned_alloc(256, (size_t)np*4);
    m->s_tb2_sel   = (int   *)aligned_alloc(256, (size_t)np*4);
    for (int L = 0; L < c->n_layers; L++) {
        int ratio = c->compress_ratios[L];
        if (ratio == 0) continue;
        ds4f_layer *ly = &m->layers[L];
        int overlap = (ratio == 4), coff = overlap ? 2 : 1, W = coff*KV;
        int nslot = np / ratio;
        ly->cmp_wkv   = (uint16_t *)aligned_alloc(256, (size_t)W*C*2);
        ly->cmp_wgate = (uint16_t *)aligned_alloc(256, (size_t)W*C*2);
        ly->cmp_ape   = (float *)aligned_alloc(256, (size_t)ratio*W*4);
        ly->cmp_norm  = (uint16_t *)aligned_alloc(64, ((size_t)KV*2 + 63) & ~63u);
        ly->cmp_kv_state    = (float *)aligned_alloc(256, (size_t)coff*ratio*W*4);
        ly->cmp_score_state = (float *)aligned_alloc(256, (size_t)coff*ratio*W*4);
        ly->cmp_kv = (float *)aligned_alloc(256, (size_t)nslot*KV*4);
        ds4f_compress_state_reset(ly->cmp_kv_state, ly->cmp_score_state, ratio, KV);
        if (fill) {
            ds4f_tensor t1 = { ly->cmp_wkv,   NULL, DS4F_BF16, W, C };      ds4f_fill(m, t1);
            ds4f_tensor t2 = { ly->cmp_wgate, NULL, DS4F_BF16, W, C };      ds4f_fill(m, t2);
            ds4f_tensor t3 = { ly->cmp_ape,   NULL, DS4F_F32, ratio, W };   ds4f_fill(m, t3);
            ds4f_tensor t4 = { ly->cmp_norm,  NULL, DS4F_BF16, 1, KV };     ds4f_fill(m, t4);
        }
        if (ratio == 4) {                                       /* indexer (CSA only) */
            int icoff = 2, iW = icoff*ihd;                      /* index ratio==4 => overlap */
            ly->idx_wq_b  = (uint16_t *)aligned_alloc(256, (size_t)iH*ihd*qlora*2);
            ly->idx_wproj = (uint16_t *)aligned_alloc(256, (size_t)iH*C*2);
            ly->idx_cmp_wkv   = (uint16_t *)aligned_alloc(256, (size_t)iW*C*2);
            ly->idx_cmp_wgate = (uint16_t *)aligned_alloc(256, (size_t)iW*C*2);
            ly->idx_cmp_ape   = (float *)aligned_alloc(256, (size_t)ratio*iW*4);
            ly->idx_cmp_norm  = (uint16_t *)aligned_alloc(64, ((size_t)ihd*2 + 63) & ~63u);
            ly->idx_cmp_kv_state    = (float *)aligned_alloc(256, (size_t)icoff*ratio*iW*4);
            ly->idx_cmp_score_state = (float *)aligned_alloc(256, (size_t)icoff*ratio*iW*4);
            ly->idx_kv = (float *)aligned_alloc(256, (size_t)nslot*ihd*4);
            ds4f_compress_state_reset(ly->idx_cmp_kv_state, ly->idx_cmp_score_state, ratio, ihd);
            if (fill) {
                ds4f_tensor u1 = { ly->idx_wq_b,      NULL, DS4F_BF16, iH*ihd, qlora }; ds4f_fill(m, u1);
                ds4f_tensor u2 = { ly->idx_wproj,     NULL, DS4F_BF16, iH, C };         ds4f_fill(m, u2);
                ds4f_tensor u3 = { ly->idx_cmp_wkv,   NULL, DS4F_BF16, iW, C };         ds4f_fill(m, u3);
                ds4f_tensor u4 = { ly->idx_cmp_wgate, NULL, DS4F_BF16, iW, C };         ds4f_fill(m, u4);
                ds4f_tensor u5 = { ly->idx_cmp_ape,   NULL, DS4F_F32, ratio, iW };     ds4f_fill(m, u5);
                ds4f_tensor u6 = { ly->idx_cmp_norm,  NULL, DS4F_BF16, 1, ihd };       ds4f_fill(m, u6);
            }
        }
    }
}

static ds4f_model *ds4f_alloc_synth(ds4f_config cfg, int ep_rank, int ep_size,
                                    int n_threads, int n_cmgs) {
    ds4f_model *m = (ds4f_model *)calloc(1, sizeof(*m));
    m->cfg = cfg; m->ep_rank = ep_rank; m->ep_size = ep_size;
    m->n_threads = n_threads; m->n_cmgs = n_cmgs;
    ds4f_init_fp8_e4m3_lut(m->fp8_lut);
    /* Dense default is FP8 on-demand (lean ~21.6 GB/node, safe to 128K ctx).
     * DS4F_FP8_BF16=1 predequants the replicated dense to BF16 (+6 GB, faster,
     * intended ≤8K ctx). The pv pair-interleaved layout is byte-identical and
     * strictly faster at zero memory cost, so it AUTO-ENABLES whenever predequant
     * is on; DS4F_BF16_PV=0 is the explicit escape, =1 forces pv even in FP8 mode
     * (only speeds the always-bf16 head+router then). Empty env string == unset. */
    {   const char *e = getenv("DS4F_FP8_BF16");
        int pre = (e && *e && atoi(e)) ? 1 : 0;
        const char *p = getenv("DS4F_BF16_PV");
        m->bf16_pv = (p && *p) ? (atoi(p) ? 1 : 0) : pre;   /* default: track predequant */
        m->dense_qt = pre ? (m->bf16_pv ? DS4F_BF16_PV : DS4F_BF16) : DS4F_FP8;
        m->bf16_mv_qt = m->bf16_pv ? DS4F_BF16_PV : DS4F_BF16;
        /* DS4F_DENSE_MXFP4=1: route the replicated dense (MLA + shared) through the
         * MXFP4 split kernel (0.53 B/elem vs FP8 1 / BF16 2). Synthetic harness =>
         * accuracy-free; tests whether the 2-4x byte cut beats the nibble-unpack
         * cost at M=1. Overrides FP8/BF16; head+router stay bf16 (bf16_mv_qt). All
         * dense K are %32 (4096/1024/512/8192/2048) so the split layout is valid. */
        const char *mx = getenv("DS4F_DENSE_MXFP4");
        if (mx && *mx && atoi(mx)) m->dense_qt = DS4F_MXFP4; }
    {   const char *e = getenv("DS4F_FP8_MAGIC");
        m->fp8_magic = (e && *e && atoi(e)) ? 1 : 0; }
    {   const char *e = getenv("DS4F_SPARSE");
        m->sparse = (e && *e && atoi(e)) ? 1 : 0; }
    {   const char *e = getenv("DS4F_MHC");
        m->mhc = (e && *e && atoi(e)) ? 1 : 0; }
    {   const char *e = getenv("DS4F_EXACT");
        m->exact = (e && *e && atoi(e)) ? 1 : 0; }
    {   const char *e = getenv("DS4F_TIERB2");
        m->tierb2 = (e && *e && atoi(e)) ? 1 : 0; }
    if (m->tierb2) m->exact = 1;   /* Tier-B2 reuses the exact q-norm/RoPE/window path */
    m->pool = ds4f_pool_start(n_threads, n_cmgs);

    m->arena_sz = ds4f_arena_size(&cfg, ep_rank, ep_size,
                                  m->dense_qt == DS4F_BF16 || m->dense_qt == DS4F_BF16_PV);
    m->arena = (uint8_t *)mmap(NULL, m->arena_sz, PROT_READ|PROT_WRITE,
                               MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (m->arena == MAP_FAILED) { fprintf(stderr, "mmap %zu failed\n", m->arena_sz); abort(); }
    /* CRITICAL for A64FX/Fugaku NUMA: disable transparent huge pages so that
     * the per-thread first-touch below (ds4f_fill, compute-thread == touch-thread)
     * actually places each thread's row block on its own CMG node. With THP on,
     * Fugaku pre-faults whole huge pages onto the allocating node and the matvec
     * BW collapses to ~1 CMG (see reference_a64fx_full_node_bw). */
#ifdef MADV_NOHUGEPAGE
    madvise(m->arena, m->arena_sz, MADV_NOHUGEPAGE);
#endif
    m->arena_used = 0;

    int C = cfg.hidden;
    m->out_norm = (uint16_t *)ds4f_bump(m, (size_t)C*2, 64);
    ds4f_tensor embed = ds4f_new_tensor(m, DS4F_BF16, cfg.vocab, C); m->embed = (uint16_t *)embed.w; /* flat gather */
    m->head = ds4f_new_tensor(m, m->bf16_mv_qt, cfg.vocab, C);       /* matvec'd -> pv when enabled */
    {   int hc = cfg.hc_mult, hd = hc*C;
        m->hc_head_fn    = (float *)ds4f_bump(m, (size_t)hc*hd*4, 256);
        m->hc_head_base  = (float *)ds4f_bump(m, (size_t)hc*4, 64);
        m->hc_head_scale = (float *)ds4f_bump(m, (size_t)4, 64); }

    m->layers = (ds4f_layer *)calloc(cfg.n_layers, sizeof(ds4f_layer));
    int no = ds4f_n_owned(cfg.n_experts, ep_rank, ep_size);
    for (int L = 0; L < cfg.n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        ly->attn_norm = (uint16_t *)ds4f_bump(m, (size_t)C*2, 64);
        ly->ffn_norm  = (uint16_t *)ds4f_bump(m, (size_t)C*2, 64);
        ly->q_norm    = (uint16_t *)ds4f_bump(m, (size_t)cfg.q_lora*2, 64);
        ly->kv_norm   = (uint16_t *)ds4f_bump(m, (size_t)cfg.kv_lora*2, 64);
        ds4f_qtype dq = m->dense_qt;
        ly->wq_a = ds4f_new_tensor(m, dq, cfg.q_lora, C);
        ly->wq_b = ds4f_new_tensor(m, dq, cfg.n_heads*cfg.q_head_dim, cfg.q_lora);
        ly->wkv  = ds4f_new_tensor(m, dq, cfg.kv_lora, C);
        ly->wo_a = ds4f_new_tensor(m, dq, cfg.o_inter, C);
        ly->wo_b = ds4f_new_tensor(m, dq, C, cfg.o_inter);
        ly->attn_sink = (float *)ds4f_bump(m, (size_t)cfg.n_heads*4, 64);
        ly->gate = ds4f_new_tensor(m, m->bf16_mv_qt, cfg.n_experts, C); /* router matvec -> pv when enabled */
        ly->sh_w1 = ds4f_new_tensor(m, dq, cfg.shared_inter, C);
        ly->sh_w3 = ds4f_new_tensor(m, dq, cfg.shared_inter, C);
        ly->sh_w2 = ds4f_new_tensor(m, dq, C, cfg.shared_inter);
        ly->ex_w1 = (ds4f_tensor *)calloc(no, sizeof(ds4f_tensor));
        ly->ex_w2 = (ds4f_tensor *)calloc(no, sizeof(ds4f_tensor));
        ly->ex_w3 = (ds4f_tensor *)calloc(no, sizeof(ds4f_tensor));
        ly->owned_eid = (int *)calloc(no, sizeof(int));
        ly->n_owned = no;
        int slot = 0;
        for (int e = 0; e < cfg.n_experts; e++) if (e % ep_size == ep_rank) {
            ly->ex_w1[slot] = ds4f_new_tensor(m, DS4F_MXFP4, cfg.moe_inter, C);
            ly->ex_w3[slot] = ds4f_new_tensor(m, DS4F_MXFP4, cfg.moe_inter, C);
            ly->ex_w2[slot] = ds4f_new_tensor(m, DS4F_MXFP4, C, cfg.moe_inter);
            ly->owned_eid[slot] = e; slot++;
        }
        {   int hc = cfg.hc_mult, mix = (2+hc)*hc, hd = hc*C;
            ly->hc_attn_fn    = (float *)ds4f_bump(m, (size_t)mix*hd*4, 256);
            ly->hc_attn_base  = (float *)ds4f_bump(m, (size_t)mix*4, 64);
            ly->hc_attn_scale = (float *)ds4f_bump(m, (size_t)3*4, 64);
            ly->hc_ffn_fn     = (float *)ds4f_bump(m, (size_t)mix*hd*4, 256);
            ly->hc_ffn_base   = (float *)ds4f_bump(m, (size_t)mix*4, 64);
            ly->hc_ffn_scale  = (float *)ds4f_bump(m, (size_t)3*4, 64); }
        ly->kv_cache   = (float *)ds4f_bump(m, (size_t)cfg.max_pos*cfg.kv_lora*4, 256);
    }

    /* parallel first-touch fill of all quantized tensors (compute-thread ==
     * touch-thread, so each row block lands on the CMG that later reads it) */
    ds4f_tensor onrm = { m->out_norm, NULL, DS4F_BF16, 1, C }; ds4f_fill(m, onrm);
    ds4f_fill(m, embed); ds4f_fill(m, m->head);
    {   int hc = cfg.hc_mult, hd = hc*C;
        ds4f_tensor hf = { m->hc_head_fn, NULL, DS4F_F32, hc, hd }; ds4f_fill(m, hf);
        ds4f_hc_fill_meta(m->hc_head_base, hc, m->hc_head_scale, 1, 4096); }
    for (int L = 0; L < cfg.n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        ds4f_tensor an = { ly->attn_norm, NULL, DS4F_BF16, 1, C }; ds4f_fill(m, an);
        ds4f_tensor fn = { ly->ffn_norm,  NULL, DS4F_BF16, 1, C }; ds4f_fill(m, fn);
        ds4f_tensor qn = { ly->q_norm,    NULL, DS4F_BF16, 1, cfg.q_lora }; ds4f_fill(m, qn);
        ds4f_tensor kn = { ly->kv_norm,   NULL, DS4F_BF16, 1, cfg.kv_lora }; ds4f_fill(m, kn);
        ds4f_fill(m, ly->wq_a); ds4f_fill(m, ly->wq_b); ds4f_fill(m, ly->wkv);
        ds4f_fill(m, ly->wo_a); ds4f_fill(m, ly->wo_b);
        ds4f_fill(m, ly->gate); ds4f_fill(m, ly->sh_w1); ds4f_fill(m, ly->sh_w2); ds4f_fill(m, ly->sh_w3);
        for (int s = 0; s < no; s++) { ds4f_fill(m, ly->ex_w1[s]); ds4f_fill(m, ly->ex_w2[s]); ds4f_fill(m, ly->ex_w3[s]); }
        for (int h = 0; h < cfg.n_heads; h++) ly->attn_sink[h] = -2.0f;          /* mild sink */
        {   int hc = cfg.hc_mult, mix = (2+hc)*hc, hd = hc*C;
            ds4f_tensor af = { ly->hc_attn_fn, NULL, DS4F_F32, mix, hd }; ds4f_fill(m, af);
            ds4f_tensor ff = { ly->hc_ffn_fn,  NULL, DS4F_F32, mix, hd }; ds4f_fill(m, ff);
            ds4f_hc_fill_meta(ly->hc_attn_base, mix, ly->hc_attn_scale, 3, L*2);
            ds4f_hc_fill_meta(ly->hc_ffn_base,  mix, ly->hc_ffn_scale,  3, L*2+1); }
    }

    /* scratch */
    int H = cfg.n_heads*cfg.q_head_dim;
    m->s_hn    = (float *)aligned_alloc(256, (size_t)C*4);
    m->s_qlat  = (float *)aligned_alloc(256, (size_t)cfg.q_lora*4);
    m->s_q     = (float *)aligned_alloc(256, (size_t)H*4);
    m->s_kvlat = (float *)aligned_alloc(256, (size_t)cfg.kv_lora*4);
    m->s_attn  = (float *)aligned_alloc(256, (size_t)H*4);
    m->s_oin   = (float *)aligned_alloc(256, (size_t)C*4);
    m->s_o1    = (float *)aligned_alloc(256, (size_t)cfg.o_inter*4);
    m->s_o     = (float *)aligned_alloc(256, (size_t)C*4);
    m->s_h2    = (float *)aligned_alloc(256, (size_t)C*4);
    m->s_router= (float *)aligned_alloc(256, (size_t)cfg.n_experts*4);
    m->s_shg   = (float *)aligned_alloc(256, (size_t)cfg.shared_inter*4);
    m->s_shu   = (float *)aligned_alloc(256, (size_t)cfg.shared_inter*4);
    m->s_exg   = (float *)aligned_alloc(256, (size_t)cfg.moe_inter*4);
    m->s_exu   = (float *)aligned_alloc(256, (size_t)cfg.moe_inter*4);
    m->s_moe   = (float *)aligned_alloc(256, (size_t)C*4);
    m->s_route = (float *)aligned_alloc(256, (size_t)C*4);
    m->s_logits= (float *)aligned_alloc(256, (size_t)cfg.vocab*4);
    /* sparse-indexer scratch: block scores reuse as the selected-score buffer,
     * so the per-thread stride must cover both the worst-case block count
     * (ceil(nP/R), R>=1 => up to max_pos) and index_topk selected positions. */
    m->idx_blk_stride = cfg.max_pos > cfg.index_topk ? cfg.max_pos : cfg.index_topk;
    m->s_idx_scores = (float *)aligned_alloc(256, (size_t)n_threads * m->idx_blk_stride * 4);
    m->s_idx_sel    = (int   *)aligned_alloc(256, (size_t)n_threads * cfg.index_topk * 4);
    /* mHC 4-stream scratch (only used when m->mhc) */
    m->s_x4    = (float *)aligned_alloc(256, (size_t)cfg.hc_mult*C*4);
    m->s_resid = (float *)aligned_alloc(256, (size_t)cfg.hc_mult*C*4);
    m->s_xc    = (float *)aligned_alloc(256, (size_t)C*4);
    ds4f_build_freqs(m);   /* RoPE/YaRN tables (only when exact) */
    if (m->tierb2) ds4f_alloc_tb2(m, 1);   /* off-arena compressor/indexer (synth fill) */
    return m;
}

static void ds4f_free(ds4f_model *m) {
    if (!m) return;
    ds4f_pool_stop(m->pool);
    if (m->arena && m->arena != MAP_FAILED) munmap(m->arena, m->arena_sz);
    free(m->layers);
    free(m);
}

/* ===================== real-weight loader (Stage 4b) =====================
 *
 * ds4f_load_real() builds a ds4f_model from the REAL DeepSeek-V4-Flash weights
 * that ds4f_stage.c packed into THIS rank's node-local blob, replacing the
 * synthetic fill of ds4f_alloc_synth. Same arena / scratch / forward wiring;
 * only the bytes differ. The blob+manifest live at
 *   <blob_dir>/rank<rr>.blob       packed weights (256B aligned/tensor)
 *   <blob_dir>/rank<rr>.manifest   "<off> <nbytes> <dtype> <ndims> <shape..> <name>"
 *
 * Real dtypes (verified against the staged manifest):
 *   dense MLA (wq_a/wq_b/wkv/wo_a/wo_b) + shared expert (w1/w2/w3)
 *                                  = F8_E4M3 weight + F8_E8M0 128x128 block scale
 *   routed experts (w1/w2/w3)      = I8 (2 fp4 nibbles/byte, float4_e2m1fn_x2)
 *                                    + F8_E8M0 block-32 scale
 *   router ffn.gate.weight, embed, lm_head, all norms = BF16 (row-major)
 *   attn.attn_sink, hc_attn/ffn/head_*                = F32
 *   ffn.gate.tid2eid (I64 routing table)              = ignored (not used here)
 *
 * Two upstream conventions are reconciled ON COPY so the EXISTING kernels
 * dequant the right VALUES (not just the right bytes):
 *   1. DeepSeek FP8 = float8_e4m3fn (exp==15 is FINITE, max 448; only
 *      S.1111.111 = NaN). The real path builds the LUT with e4m3fn semantics
 *      (ds4f_init_fp8_e4m3fn_lut), unlike the synth LUT (ds4f_init_fp8_e4m3_lut)
 *      that maps every exp==15 to NaN. The E8M0 block-scale layout
 *      [rows/128, cols/128] already matches matvec_fp8e4m3_8row's escale index.
 *   2. DeepSeek packs experts as float4_e2m1fn_x2 SEQUENTIALLY (byte j ->
 *      element 2j low / 2j+1 high) over the standard e2m1 table (max 6).
 *      matvec_mxfp4_8row expects (byte j -> element j / j+16) interleave over a
 *      2x table {0,1,2,3,4,6,8,12}. The copy REPACKS each 16-byte block into the
 *      kernel layout AND DECREMENTS each E8M0 expert-scale byte by 1 (an exact
 *      /2, pure power of two) to cancel the 2x value table. Result: the kernel
 *      reproduces DeepSeek's e2m1 weights bit-for-bit (modulo the e==0 block,
 *      which is ~0 anyway).
 *
 * The mHC/sparse forward math is still partly a stand-in (e.g. o-proj silu), so
 * end-to-end logits are not yet bit-exact; this loader gets the WEIGHT bytes,
 * footprint, and dequant VALUES right -- the prerequisite for a later exact
 * forward -- and is the real-weight counterpart of the synthetic throughput
 * harness. Any missing / wrong-dtype / wrong-size tensor aborts with a named
 * error (the completeness + integrity check).
 */

/* float8_e4m3fn -> f32 bits. "fn" (finite): exp==15 is a normal value (max 448),
 * only the single code S.1111.111 is NaN. Distinct from the synth-path
 * ds4f_fp8_e4m3_to_fp32_bits() which maps all exp==15 to NaN. */
static inline uint32_t ds4f_fp8_e4m3fn_to_fp32_bits(uint8_t x) {
    uint8_t sign = (x >> 7) & 1, exp = (x >> 3) & 0xF, mant = x & 0x7;
    if (exp == 0) {                              /* zero / subnormal */
        if (mant == 0) return (uint32_t)sign << 31;
        int sh = 0; while ((mant & 0x4) == 0) { mant <<= 1; sh++; }
        mant &= 0x3;
        uint32_t e = (uint32_t)(127 - 7 - sh);
        return ((uint32_t)sign << 31) | (e << 23) | ((uint32_t)mant << 20);
    }
    if (exp == 15 && mant == 7)                  /* the one NaN code */
        return ((uint32_t)sign << 31) | (0xFFu << 23) | (1u << 22);
    uint32_t e = (uint32_t)exp + (127 - 7);      /* normal; exp==15 stays finite */
    return ((uint32_t)sign << 31) | (e << 23) | ((uint32_t)mant << 20);
}
static inline void ds4f_init_fp8_e4m3fn_lut(uint32_t *lut) {
    for (int i = 0; i < 256; i++) lut[i] = ds4f_fp8_e4m3fn_to_fp32_bits((uint8_t)i);
}

static inline double ds4f_wall(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- manifest + blob ---- */
typedef struct {
    char      name[192];
    uint64_t  off, nbytes;
    char      dtype[16];
    int       ndims;
    long long shape[8];
} ds4f_mani_ent;

typedef struct {
    ds4f_mani_ent *e; int n, cap;
    int       rank, ep_size;
    uint64_t  total_bytes;
    uint8_t  *blob; size_t blob_sz; int blob_fd;
} ds4f_blob;

static const ds4f_mani_ent *ds4f_mani_find(const ds4f_blob *B, const char *name) {
    for (int i = 0; i < B->n; i++)
        if (strcmp(B->e[i].name, name) == 0) return &B->e[i];
    return NULL;
}

/* parse <dir>/rank<rr>.manifest and mmap <dir>/rank<rr>.blob (read-only). The
 * manifest is parsed single-threaded (strtok is fine), so no _GNU_SOURCE dep. */
static int ds4f_blob_open(ds4f_blob *B, const char *dir, int rank) {
    memset(B, 0, sizeof *B);
    char mp[1200], bp[1200];
    snprintf(mp, sizeof mp, "%s/rank%02d.manifest", dir, rank);
    snprintf(bp, sizeof bp, "%s/rank%02d.blob", dir, rank);
    FILE *mf = fopen(mp, "r");
    if (!mf) { fprintf(stderr, "ds4f_load: cannot open %s: %s\n", mp, strerror(errno)); return -1; }
    B->cap = 8192; B->e = (ds4f_mani_ent *)malloc((size_t)B->cap * sizeof(*B->e)); B->n = 0;
    char line[1024];
    while (fgets(line, sizeof line, mf)) {
        if (line[0] == '#') {                    /* header carries rank / ep_size */
            char *p;
            if ((p = strstr(line, "rank=")))    B->rank    = atoi(p + 5);
            if ((p = strstr(line, "ep_size="))) B->ep_size = atoi(p + 8);
            continue;
        }
        if (B->n == B->cap) { B->cap *= 2; B->e = (ds4f_mani_ent *)realloc(B->e, (size_t)B->cap * sizeof(*B->e)); }
        ds4f_mani_ent *t = &B->e[B->n];
        char *tok = strtok(line, " \t\n");           if (!tok) continue; t->off    = strtoull(tok, NULL, 10);
        tok = strtok(NULL, " \t\n");                 if (!tok) continue; t->nbytes = strtoull(tok, NULL, 10);
        tok = strtok(NULL, " \t\n");                 if (!tok) continue; snprintf(t->dtype, sizeof t->dtype, "%s", tok);
        tok = strtok(NULL, " \t\n");                 if (!tok) continue; t->ndims  = atoi(tok);
        if (t->ndims < 0 || t->ndims > 8) t->ndims = 0;
        for (int d = 0; d < t->ndims; d++) { tok = strtok(NULL, " \t\n"); t->shape[d] = tok ? strtoll(tok, NULL, 10) : 0; }
        tok = strtok(NULL, " \t\n");                 if (!tok) continue; snprintf(t->name, sizeof t->name, "%s", tok);
        B->total_bytes += t->nbytes;
        B->n++;
    }
    fclose(mf);
    B->blob_fd = open(bp, O_RDONLY);
    if (B->blob_fd < 0) { fprintf(stderr, "ds4f_load: cannot open %s: %s\n", bp, strerror(errno)); free(B->e); B->e = NULL; return -1; }
    struct stat sb;
    if (fstat(B->blob_fd, &sb) != 0) { fprintf(stderr, "ds4f_load: fstat %s failed\n", bp); close(B->blob_fd); free(B->e); B->e = NULL; return -1; }
    B->blob_sz = (size_t)sb.st_size;
    B->blob = (uint8_t *)mmap(NULL, B->blob_sz, PROT_READ, MAP_PRIVATE, B->blob_fd, 0);
    if (B->blob == MAP_FAILED) { fprintf(stderr, "ds4f_load: mmap %s (%zu) failed\n", bp, B->blob_sz); close(B->blob_fd); free(B->e); B->e = NULL; return -1; }
    return 0;
}

static void ds4f_blob_close(ds4f_blob *B) {
    if (B->blob && B->blob != MAP_FAILED) munmap(B->blob, B->blob_sz);
    if (B->blob_fd > 0) close(B->blob_fd);
    free(B->e); B->e = NULL; B->n = 0;
}

/* After a tensor's bytes are copied blob->arena, its source pages in the blob
 * mmap are dead weight in HBM. Drop the fully-contained INTERIOR pages (clean,
 * read-only -> correctness-neutral: a re-fault would re-read identical file
 * bytes) so the ~22 GB blob page cache does not pile up on top of the ~22 GB
 * arena during load and overflow HBM. Aligned inward so a partial page shared
 * with an adjacent (not-yet-copied) tensor is never dropped. Gated by
 * DS4F_LOAD_DROP_BLOB (default on); set =0 to keep the old mmap-cached behavior. */
static int ds4f_drop_blob = 1;
static void ds4f_blob_drop(const ds4f_blob *B, uint64_t off, size_t nbytes) {
    if (!ds4f_drop_blob || nbytes == 0) return;
    long pg = sysconf(_SC_PAGESIZE); if (pg <= 0) pg = 4096;
    uint64_t start = (off + (uint64_t)(pg - 1)) & ~(uint64_t)(pg - 1);  /* round up */
    uint64_t end   = (off + nbytes) & ~(uint64_t)(pg - 1);             /* round down */
    if (end > start) madvise(B->blob + start, (size_t)(end - start), MADV_DONTNEED);
}

/* ---- parallel copy/transform: blob src -> arena dst (NUMA first-touch via the
 * SAME rowsplit8 the matvec later uses, so each row block lands on its CMG) ---- */
typedef struct { ds4f_tensor t; const uint8_t *src_w, *src_s; } ds4f_copy_task;

static void ds4f_copy_worker(void *arg, int tid, int nthr) {
    ds4f_copy_task *T = (ds4f_copy_task *)arg; ds4f_tensor *t = &T->t;
    int rows = t->rows, K = t->cols;
    if (t->type == DS4F_F32) {                   /* plain element split (rows may be <8 or 1-D) */
        size_t n = (size_t)rows * K, per = n / nthr; int ex = (int)(n % nthr);
        size_t i0 = per * tid + (tid < ex ? (size_t)tid : (size_t)ex);
        size_t i1 = i0 + per + (tid < ex ? 1 : 0);
        if (i1 > i0) memcpy((float *)t->w + i0, (const float *)T->src_w + i0, (i1 - i0) * 4);
        return;
    }
    int r0, r1; ds4f_rowsplit8(rows, nthr, tid, &r0, &r1);
    if (t->type == DS4F_BF16) {                  /* real BF16 = row-major -> direct */
        if (r1 > r0) memcpy((uint16_t *)t->w + (size_t)r0 * K,
                            (const uint16_t *)T->src_w + (size_t)r0 * K, (size_t)(r1 - r0) * K * 2);
    } else if (t->type == DS4F_FP8) {            /* e4m3fn bytes row-major -> direct */
        if (r1 > r0) memcpy((uint8_t *)t->w + (size_t)r0 * K, T->src_w + (size_t)r0 * K, (size_t)(r1 - r0) * K);
        if (tid == 0)                            /* tiny 128x128 block scale; whole on tid0 (no 128-split race) */
            memcpy(t->scale, T->src_s, ds4f_sbytes(DS4F_FP8, rows, K));
    } else if (t->type == DS4F_MXFP4) {          /* fp4: repack sequential -> (j,j+16); scale e-=1 (x0.5) */
        size_t rb = K / 2, sb = K / 32, nb = rb / 16;   /* nb = K/32 sixteen-byte blocks per row */
        for (int i = r0; i < r1; i++) {
            const uint8_t *sw = T->src_w + (size_t)i * rb; uint8_t *dw = (uint8_t *)t->w + (size_t)i * rb;
            for (size_t b = 0; b < nb; b++) {
                const uint8_t *s = sw + b * 16; uint8_t *d = dw + b * 16;
                for (int j = 0; j < 16; j++) {   /* dst byte j: low = elem j, high = elem j+16 */
                    uint8_t lo = (j & 1) ? (uint8_t)(s[j >> 1] >> 4)        : (uint8_t)(s[j >> 1] & 0xf);
                    uint8_t hi = (j & 1) ? (uint8_t)(s[(j >> 1) + 8] >> 4)  : (uint8_t)(s[(j >> 1) + 8] & 0xf);
                    d[j] = (uint8_t)((hi << 4) | lo);
                }
            }
            const uint8_t *ss = T->src_s + (size_t)i * sb; uint8_t *ds = t->scale + (size_t)i * sb;
            for (size_t j = 0; j < sb; j++) { uint8_t e = ss[j]; ds[j] = e ? (uint8_t)(e - 1) : 0; }
        }
    }
}

static void ds4f_copy_run(ds4f_model *m, ds4f_tensor dst, const uint8_t *sw, const uint8_t *ss) {
    ds4f_copy_task T; T.t = dst; T.src_w = sw; T.src_s = ss;
    ds4f_pool_run(m->pool, ds4f_copy_worker, &T);
}

static const char *ds4f_qtype_dtstr(ds4f_qtype q) {
    switch (q) { case DS4F_FP8: return "F8_E4M3"; case DS4F_MXFP4: return "I8";
                 case DS4F_F32: return "F32"; default: return "BF16"; }
}

/* find a manifest entry and assert its dtype + byte size; abort otherwise */
static const ds4f_mani_ent *ds4f_need(const ds4f_blob *B, const char *name,
                                      const char *dtype, size_t nbytes) {
    const ds4f_mani_ent *e = ds4f_mani_find(B, name);
    if (!e) { fprintf(stderr, "ds4f_load: MISSING tensor '%s'\n", name); abort(); }
    if (strcmp(e->dtype, dtype) != 0) {
        fprintf(stderr, "ds4f_load: '%s' dtype %s != expected %s\n", name, e->dtype, dtype); abort(); }
    if (e->nbytes != nbytes) {
        fprintf(stderr, "ds4f_load: '%s' nbytes %llu != expected %zu\n",
                name, (unsigned long long)e->nbytes, nbytes); abort(); }
    return e;
}

/* load a quantized matvec tensor (FP8/MXFP4/BF16) by BASE name (+.weight/.scale) */
static void ds4f_load_q(ds4f_model *m, const ds4f_blob *B, ds4f_tensor *dst, const char *base) {
    char wn[256], sn[256];
    snprintf(wn, sizeof wn, "%s.weight", base);
    size_t wb = ds4f_wbytes(dst->type, dst->rows, dst->cols);
    const ds4f_mani_ent *we = ds4f_need(B, wn, ds4f_qtype_dtstr(dst->type), wb);
    const uint8_t *sw = B->blob + we->off, *ss = NULL;
    size_t sb = ds4f_sbytes(dst->type, dst->rows, dst->cols);
    if (sb) {
        snprintf(sn, sizeof sn, "%s.scale", base);
        const ds4f_mani_ent *se = ds4f_need(B, sn, "F8_E8M0", sb);
        ss = B->blob + se->off;
    }
    ds4f_copy_run(m, *dst, sw, ss);
    ds4f_blob_drop(B, we->off, wb);                       /* release copied blob pages */
    if (ss) ds4f_blob_drop(B, (uint64_t)(ss - B->blob), sb);
    m->bytes_read += wb + sb;
}

/* load a raw BF16/F32 buffer (un-scaled) into a plain arena pointer */
static void ds4f_load_raw(ds4f_model *m, const ds4f_blob *B, void *dst,
                          const char *name, ds4f_qtype type, int rows, int cols) {
    size_t wb = ds4f_wbytes(type, rows, cols);
    const ds4f_mani_ent *e = ds4f_need(B, name, ds4f_qtype_dtstr(type), wb);
    ds4f_tensor t = { dst, NULL, type, rows, cols };
    ds4f_copy_run(m, t, B->blob + e->off, NULL);
    ds4f_blob_drop(B, e->off, wb);                        /* release copied blob pages */
    m->bytes_read += wb;
}

/* ---- Tier-B2 weight conversion: real bytes -> plain f32 (the compressor/indexer
 * kernels consume float* weights + bf16 norms, so FP8/BF16 sources are widened at
 * load time). Off-arena destinations; same blob-drop discipline as the dense path. */
typedef struct { float *dst; const uint16_t *src; size_t n; } ds4f_bf16f32_task;
static void ds4f_bf16f32_worker(void *arg, int tid, int nthr) {
    ds4f_bf16f32_task *T = (ds4f_bf16f32_task *)arg;
    size_t n = T->n, per = n / nthr; int ex = (int)(n % nthr);
    size_t i0 = per * tid + (size_t)(tid < ex ? tid : ex);
    size_t i1 = i0 + per + (tid < ex ? 1 : 0);
    for (size_t i = i0; i < i1; i++) {
        uint32_t b = (uint32_t)T->src[i] << 16;   /* bf16 -> f32: zero-extend mantissa */
        memcpy(&T->dst[i], &b, 4);
    }
}
static void ds4f_load_bf16_to_f32(ds4f_model *m, const ds4f_blob *B, float *dst,
                                  const char *name, int rows, int cols) {
    size_t wb = ds4f_wbytes(DS4F_BF16, rows, cols);
    const ds4f_mani_ent *e = ds4f_need(B, name, "BF16", wb);
    ds4f_bf16f32_task T = { dst, (const uint16_t *)(B->blob + e->off), (size_t)rows * cols };
    ds4f_pool_run(m->pool, ds4f_bf16f32_worker, &T);
    ds4f_blob_drop(B, e->off, wb);
    m->bytes_read += wb;
}

/* dequant a real FP8 e4m3fn [rows,cols] tensor (+E8M0 128x128 block scale) -> f32.
 * Mirrors matvec_fp8e4m3_8row EXACTLY: value = reinterpret(lut[byte]) * 2^(e-127),
 * scale block = escale[(row/128)*sb_cols + col/128]. Plain row split (no 8-align). */
typedef struct { float *dst; const uint8_t *w, *es; const uint32_t *lut;
                 int rows, cols, sbc; } ds4f_fp8f32_task;
static void ds4f_fp8f32_worker(void *arg, int tid, int nthr) {
    ds4f_fp8f32_task *T = (ds4f_fp8f32_task *)arg;
    int rows = T->rows, K = T->cols, sbc = T->sbc;
    int per = rows / nthr, ex = rows % nthr;
    int r0 = per * tid + (tid < ex ? tid : ex);
    int r1 = r0 + per + (tid < ex ? 1 : 0);
    for (int r = r0; r < r1; r++) {
        const uint8_t *wr  = T->w  + (size_t)r * K;
        const uint8_t *esr = T->es + (size_t)(r >> 7) * sbc;
        float *dr = T->dst + (size_t)r * K;
        for (int c = 0; c < K; c++) {
            uint32_t bits = T->lut[wr[c]]; float v; memcpy(&v, &bits, 4);
            dr[c] = v * ggml_e8m0_to_fp32(esr[c >> 7]);
        }
    }
}
static void ds4f_load_fp8_to_f32(ds4f_model *m, const ds4f_blob *B, float *dst,
                                 const char *base, int rows, int cols) {
    char wn[256], sn[256];
    snprintf(wn, sizeof wn, "%s.weight", base);
    snprintf(sn, sizeof sn, "%s.scale",  base);
    size_t wb = ds4f_wbytes(DS4F_FP8, rows, cols), sb = ds4f_sbytes(DS4F_FP8, rows, cols);
    const ds4f_mani_ent *we = ds4f_need(B, wn, "F8_E4M3", wb);
    const ds4f_mani_ent *se = ds4f_need(B, sn, "F8_E8M0", sb);
    ds4f_fp8f32_task T = { dst, B->blob + we->off, B->blob + se->off, m->fp8_lut,
                           rows, cols, (cols + 127) / 128 };
    ds4f_pool_run(m->pool, ds4f_fp8f32_worker, &T);
    ds4f_blob_drop(B, we->off, wb);
    ds4f_blob_drop(B, se->off, sb);
    m->bytes_read += wb + sb;
}

/* FP8 e4m3fn (+E8M0 block scale) -> bf16. value = lut[byte]*2^(e-127); e4m3's 3-bit
 * mantissa * a power-of-2 scale has only its top 3 f32 mantissa bits set (low 16 are
 * zero), so the f32->bf16 truncation drops only zero bits => EXACT, and widen(bf16)
 * reproduces the same f32 ds4f_load_fp8_to_f32 would have stored. Half the bytes. */
typedef struct { uint16_t *dst; const uint8_t *w, *es; const uint32_t *lut;
                 int rows, cols, sbc; } ds4f_fp8bf16_task;
static void ds4f_fp8bf16_worker(void *arg, int tid, int nthr) {
    ds4f_fp8bf16_task *T = (ds4f_fp8bf16_task *)arg;
    int rows = T->rows, K = T->cols, sbc = T->sbc;
    int per = rows / nthr, ex = rows % nthr;
    int r0 = per * tid + (tid < ex ? tid : ex);
    int r1 = r0 + per + (tid < ex ? 1 : 0);
    for (int r = r0; r < r1; r++) {
        const uint8_t *wr  = T->w  + (size_t)r * K;
        const uint8_t *esr = T->es + (size_t)(r >> 7) * sbc;
        uint16_t *dr = T->dst + (size_t)r * K;
        for (int c = 0; c < K; c++) {
            uint32_t bits = T->lut[wr[c]]; float v; memcpy(&v, &bits, 4);
            v *= ggml_e8m0_to_fp32(esr[c >> 7]);
            uint32_t fb; memcpy(&fb, &v, 4);
            dr[c] = (uint16_t)(fb >> 16);                 /* f32 -> bf16 (dropped bits are zero) */
        }
    }
}
static void ds4f_load_fp8_to_bf16(ds4f_model *m, const ds4f_blob *B, uint16_t *dst,
                                  const char *base, int rows, int cols) {
    char wn[256], sn[256];
    snprintf(wn, sizeof wn, "%s.weight", base);
    snprintf(sn, sizeof sn, "%s.scale",  base);
    size_t wb = ds4f_wbytes(DS4F_FP8, rows, cols), sb = ds4f_sbytes(DS4F_FP8, rows, cols);
    const ds4f_mani_ent *we = ds4f_need(B, wn, "F8_E4M3", wb);
    const ds4f_mani_ent *se = ds4f_need(B, sn, "F8_E8M0", sb);
    ds4f_fp8bf16_task T = { dst, B->blob + we->off, B->blob + se->off, m->fp8_lut,
                            rows, cols, (cols + 127) / 128 };
    ds4f_pool_run(m->pool, ds4f_fp8bf16_worker, &T);
    ds4f_blob_drop(B, we->off, wb);
    ds4f_blob_drop(B, se->off, sb);
    m->bytes_read += wb + sb;
}

/* ---- dense load-time PROMOTE: staged source (FP8 e4m3fn+E8M0, or BF16) -> arena
 * dest m->dense_qt / m->bf16_mv_qt (FP8 | BF16 | BF16_PV). FP8->BF16 is EXACT (e4m3's
 * 3-bit mantissa * 2^k block scale fits bf16's 7-bit mantissa with no rounding), so
 * the pv promote changes ONLY speed, not output. Same-dtype falls back to the direct
 * ds4f_load_q copy => default (no promote knob) is byte-identical to the FP8 path. */
typedef struct { ds4f_tensor t; const uint8_t *src_w, *src_s; const uint32_t *lut; int src_fp8; } ds4f_promote_task;
static void ds4f_promote_worker(void *arg, int tid, int nthr) {
    ds4f_promote_task *T = (ds4f_promote_task *)arg; ds4f_tensor *t = &T->t;
    int rows = t->rows, K = t->cols, sbc = (K + 127) / 128;
    int r0, r1; ds4f_rowsplit8(rows, nthr, tid, &r0, &r1);   /* 8-aligned -> pv groups intact */
    for (int i = r0; i < r1; i++) {
        const uint8_t  *fw = T->src_fp8 ? T->src_w + (size_t)i * K : NULL;
        const uint16_t *bw = T->src_fp8 ? NULL : (const uint16_t *)T->src_w + (size_t)i * K;
        const uint8_t  *es = T->src_fp8 ? T->src_s + (size_t)(i >> 7) * sbc : NULL;
        uint16_t *d; int slot = 0;
        if (t->type == DS4F_BF16_PV) {                       /* pair-interleaved address */
            int local = i & 7, pair = local >> 1; slot = local & 1;
            d = (uint16_t *)t->w + (size_t)(i / 8) * 8 * K + (size_t)pair * 2 * K;
        } else {                                             /* plain BF16 row */
            d = (uint16_t *)t->w + (size_t)i * K; slot = 0;
        }
        int step = (t->type == DS4F_BF16_PV) ? 2 : 1;
        for (int j = 0; j < K; j++) {
            uint16_t hv;
            if (T->src_fp8) {
                uint32_t bits = T->lut[fw[j]]; float v; memcpy(&v, &bits, 4);
                hv = ds4f_f32_bf16(v * ggml_e8m0_to_fp32(es[j >> 7]));
            } else hv = bw[j];                               /* bf16 -> bf16 (exact relayout) */
            d[step * j + slot] = hv;
        }
    }
}
static void ds4f_load_dense(ds4f_model *m, const ds4f_blob *B, ds4f_tensor *dst, const char *base) {
    char wn[256];
    snprintf(wn, sizeof wn, "%s.weight", base);
    const ds4f_mani_ent *we = ds4f_mani_find(B, wn);
    if (!we) { fprintf(stderr, "ds4f_load: MISSING tensor '%s'\n", wn); abort(); }
    int rows = dst->rows, K = dst->cols;
    int src_fp8 = (strcmp(we->dtype, "F8_E4M3") == 0), src_bf16 = (strcmp(we->dtype, "BF16") == 0);
    if (!src_fp8 && !src_bf16) { fprintf(stderr, "ds4f_load_dense: '%s' src dtype %s unsupported\n", wn, we->dtype); abort(); }
    if ((src_fp8 && dst->type == DS4F_FP8) || (src_bf16 && dst->type == DS4F_BF16)) {
        ds4f_load_q(m, B, dst, base); return;                /* no promote: direct copy */
    }
    if (dst->type != DS4F_BF16 && dst->type != DS4F_BF16_PV) {
        fprintf(stderr, "ds4f_load_dense: '%s' dest dtype %d unsupported "
                        "(real MXFP4 dense promote is lossy/NYI; use DS4F_FP8_BF16=1)\n", wn, dst->type); abort(); }
    size_t wb = src_fp8 ? ds4f_wbytes(DS4F_FP8, rows, K) : ds4f_wbytes(DS4F_BF16, rows, K);
    if (we->nbytes != wb) { fprintf(stderr, "ds4f_load_dense: '%s' nbytes %llu != %zu\n", wn, (unsigned long long)we->nbytes, wb); abort(); }
    const uint8_t *sw = B->blob + we->off, *ss = NULL; size_t sb = 0;
    if (src_fp8) {
        char sn[256]; snprintf(sn, sizeof sn, "%s.scale", base);
        sb = ds4f_sbytes(DS4F_FP8, rows, K);
        const ds4f_mani_ent *se = ds4f_need(B, sn, "F8_E8M0", sb);
        ss = B->blob + se->off;
    }
    ds4f_promote_task T = { *dst, sw, ss, m->fp8_lut, src_fp8 };
    ds4f_pool_run(m->pool, ds4f_promote_worker, &T);
    ds4f_blob_drop(B, we->off, wb);
    if (ss) ds4f_blob_drop(B, (uint64_t)(ss - B->blob), sb);
    m->bytes_read += wb + sb;
}

static ds4f_model *ds4f_load_real(ds4f_config cfg, int ep_rank, int ep_size,
                                  const char *blob_dir, int n_threads, int n_cmgs) {
    double t0 = ds4f_wall();
    { const char *e = getenv("DS4F_LOAD_DROP_BLOB"); if (e && *e) ds4f_drop_blob = atoi(e); }
    if (!blob_dir || !*blob_dir) {
        const char *e = getenv("DS4F_STAGE_DIR");
        blob_dir = (e && *e) ? e : "/local/ds4f";
    }
    ds4f_blob B;
    if (ds4f_blob_open(&B, blob_dir, ep_rank) != 0) {
        fprintf(stderr, "ds4f_load_real: no staged blob for rank %d in %s (run ds4f_stage first)\n",
                ep_rank, blob_dir);
        return NULL;
    }
    if (B.ep_size && B.ep_size != ep_size) {
        fprintf(stderr, "ds4f_load_real: staged ep_size %d != requested %d -- re-stage with DS4F_EP_SIZE=%d\n",
                B.ep_size, ep_size, ep_size);
        ds4f_blob_close(&B); return NULL;
    }

    ds4f_model *m = (ds4f_model *)calloc(1, sizeof(*m));
    m->cfg = cfg; m->ep_rank = ep_rank; m->ep_size = ep_size;
    m->n_threads = n_threads; m->n_cmgs = n_cmgs;
    /* real dtypes: staged dense = FP8(e4m3fn), experts = MXFP4, router/head/embed/
     * norm = BF16 row-major. DS4F_FP8_BF16=1 PROMOTES the replicated dense FP8->bf16
     * at load time (EXACT: e4m3 fits bf16) and auto-enables the pv pair-interleaved
     * layout -> ~1.7x faster matvecs at +~6 GB; DS4F_BF16_PV=0 forces plain bf16, =1
     * forces pv. Mirrors ds4f_alloc_synth's knobs. (DENSE_MXFP4 dense is lossy from
     * real FP8 => NYI; the loader aborts if requested.) */
    ds4f_init_fp8_e4m3fn_lut(m->fp8_lut);
    {   const char *e = getenv("DS4F_FP8_BF16");
        int pre = (e && *e && atoi(e)) ? 1 : 0;
        const char *p = getenv("DS4F_BF16_PV");
        m->bf16_pv = (p && *p) ? (atoi(p) ? 1 : 0) : pre;
        m->dense_qt = pre ? (m->bf16_pv ? DS4F_BF16_PV : DS4F_BF16) : DS4F_FP8;
        m->bf16_mv_qt = m->bf16_pv ? DS4F_BF16_PV : DS4F_BF16; }
    { const char *e = getenv("DS4F_FP8_MAGIC"); m->fp8_magic = (e && *e && atoi(e)) ? 1 : 0; }
    { const char *e = getenv("DS4F_SPARSE");    m->sparse    = (e && *e && atoi(e)) ? 1 : 0; }
    { const char *e = getenv("DS4F_MHC");       m->mhc       = (e && *e && atoi(e)) ? 1 : 0; }
    { const char *e = getenv("DS4F_EXACT");     m->exact     = (e && *e && atoi(e)) ? 1 : 0; }
    { const char *e = getenv("DS4F_TIERB2");    m->tierb2    = (e && *e && atoi(e)) ? 1 : 0; }
    if (m->tierb2) m->exact = 1;   /* Tier-B2 reuses the exact q-norm/RoPE/window path */
    m->pool = ds4f_pool_start(n_threads, n_cmgs);

    m->arena_sz = ds4f_arena_size(&cfg, ep_rank, ep_size,
                                  m->dense_qt == DS4F_BF16 || m->dense_qt == DS4F_BF16_PV);
    m->arena = (uint8_t *)mmap(NULL, m->arena_sz, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m->arena == MAP_FAILED) { fprintf(stderr, "ds4f_load_real: arena mmap %zu failed\n", m->arena_sz); abort(); }
#ifdef MADV_NOHUGEPAGE
    madvise(m->arena, m->arena_sz, MADV_NOHUGEPAGE);   /* per-thread first-touch NUMA placement */
#endif
    m->arena_used = 0;

    int C = cfg.hidden;
    /* ---- allocate (bump order MIRRORS ds4f_alloc_synth exactly) ---- */
    m->out_norm = (uint16_t *)ds4f_bump(m, (size_t)C * 2, 64);
    ds4f_tensor embed = ds4f_new_tensor(m, DS4F_BF16, cfg.vocab, C); m->embed = (uint16_t *)embed.w;
    m->head = ds4f_new_tensor(m, m->bf16_mv_qt, cfg.vocab, C);   /* matvec'd -> pv when promoted */
    {   int hc = cfg.hc_mult, hd = hc * C;
        m->hc_head_fn    = (float *)ds4f_bump(m, (size_t)hc * hd * 4, 256);
        m->hc_head_base  = (float *)ds4f_bump(m, (size_t)hc * 4, 64);
        m->hc_head_scale = (float *)ds4f_bump(m, (size_t)4, 64); }

    m->layers = (ds4f_layer *)calloc(cfg.n_layers, sizeof(ds4f_layer));
    int no = ds4f_n_owned(cfg.n_experts, ep_rank, ep_size);
    for (int L = 0; L < cfg.n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        ly->attn_norm = (uint16_t *)ds4f_bump(m, (size_t)C * 2, 64);
        ly->ffn_norm  = (uint16_t *)ds4f_bump(m, (size_t)C * 2, 64);
        ly->q_norm    = (uint16_t *)ds4f_bump(m, (size_t)cfg.q_lora * 2, 64);
        ly->kv_norm   = (uint16_t *)ds4f_bump(m, (size_t)cfg.kv_lora * 2, 64);
        ds4f_qtype dq = m->dense_qt;                            /* FP8 | BF16 | BF16_PV */
        ly->wq_a = ds4f_new_tensor(m, dq, cfg.q_lora, C);
        ly->wq_b = ds4f_new_tensor(m, dq, cfg.n_heads * cfg.q_head_dim, cfg.q_lora);
        ly->wkv  = ds4f_new_tensor(m, dq, cfg.kv_lora, C);
        ly->wo_a = ds4f_new_tensor(m, dq, cfg.o_inter, C);
        ly->wo_b = ds4f_new_tensor(m, dq, C, cfg.o_inter);
        ly->attn_sink = (float *)ds4f_bump(m, (size_t)cfg.n_heads * 4, 64);
        ly->gate = ds4f_new_tensor(m, m->bf16_mv_qt, cfg.n_experts, C); /* router matvec -> pv */
        /* router selection bias (F32[n_experts]); only non-hash layers have it.
         * Off-arena (tiny, read single-threaded in the exact gate, not a matvec). */
        if (L >= cfg.n_hash_layers) ly->gate_bias = (float *)aligned_alloc(64, (size_t)cfg.n_experts * 4);
        ly->sh_w1 = ds4f_new_tensor(m, dq, cfg.shared_inter, C);
        ly->sh_w3 = ds4f_new_tensor(m, dq, cfg.shared_inter, C);
        ly->sh_w2 = ds4f_new_tensor(m, dq, C, cfg.shared_inter);
        ly->ex_w1 = (ds4f_tensor *)calloc(no, sizeof(ds4f_tensor));
        ly->ex_w2 = (ds4f_tensor *)calloc(no, sizeof(ds4f_tensor));
        ly->ex_w3 = (ds4f_tensor *)calloc(no, sizeof(ds4f_tensor));
        ly->owned_eid = (int *)calloc(no, sizeof(int));
        ly->n_owned = no;
        int slot = 0;
        for (int e = 0; e < cfg.n_experts; e++) if (e % ep_size == ep_rank) {
            ly->ex_w1[slot] = ds4f_new_tensor(m, DS4F_MXFP4, cfg.moe_inter, C);
            ly->ex_w3[slot] = ds4f_new_tensor(m, DS4F_MXFP4, cfg.moe_inter, C);
            ly->ex_w2[slot] = ds4f_new_tensor(m, DS4F_MXFP4, C, cfg.moe_inter);
            ly->owned_eid[slot] = e; slot++;
        }
        {   int hc = cfg.hc_mult, mix = (2 + hc) * hc, hd = hc * C;
            ly->hc_attn_fn    = (float *)ds4f_bump(m, (size_t)mix * hd * 4, 256);
            ly->hc_attn_base  = (float *)ds4f_bump(m, (size_t)mix * 4, 64);
            ly->hc_attn_scale = (float *)ds4f_bump(m, (size_t)3 * 4, 64);
            ly->hc_ffn_fn     = (float *)ds4f_bump(m, (size_t)mix * hd * 4, 256);
            ly->hc_ffn_base   = (float *)ds4f_bump(m, (size_t)mix * 4, 64);
            ly->hc_ffn_scale  = (float *)ds4f_bump(m, (size_t)3 * 4, 64); }
        ly->kv_cache = (float *)ds4f_bump(m, (size_t)cfg.max_pos * cfg.kv_lora * 4, 256);
    }
    if (m->tierb2) ds4f_alloc_tb2(m, 0);   /* off-arena compressor/indexer (load by name below) */

    /* ---- copy REAL bytes by name (verify dtype/size; abort on any mismatch) ---- */
    int mix = (2 + cfg.hc_mult) * cfg.hc_mult;
    ds4f_load_raw(m, &B, m->out_norm,      "norm.weight",   DS4F_BF16, 1, C);
    ds4f_load_raw(m, &B, m->embed,         "embed.weight",  DS4F_BF16, cfg.vocab, C);
    ds4f_load_dense(m, &B, &m->head,       "head");
    ds4f_load_raw(m, &B, m->hc_head_fn,    "hc_head_fn",    DS4F_F32, cfg.hc_mult, cfg.hc_mult * C);
    ds4f_load_raw(m, &B, m->hc_head_base,  "hc_head_base",  DS4F_F32, 1, cfg.hc_mult);
    ds4f_load_raw(m, &B, m->hc_head_scale, "hc_head_scale", DS4F_F32, 1, 1);

    char nm[256];
    for (int L = 0; L < cfg.n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        #define DS4F_LN(field) (snprintf(nm, sizeof nm, "layers.%d.%s", L, field), nm)
        ds4f_load_raw(m, &B, ly->attn_norm, DS4F_LN("attn_norm.weight"),     DS4F_BF16, 1, C);
        ds4f_load_raw(m, &B, ly->ffn_norm,  DS4F_LN("ffn_norm.weight"),      DS4F_BF16, 1, C);
        ds4f_load_raw(m, &B, ly->q_norm,    DS4F_LN("attn.q_norm.weight"),   DS4F_BF16, 1, cfg.q_lora);
        ds4f_load_raw(m, &B, ly->kv_norm,   DS4F_LN("attn.kv_norm.weight"),  DS4F_BF16, 1, cfg.kv_lora);
        ds4f_load_dense(m, &B, &ly->wq_a,   DS4F_LN("attn.wq_a"));
        ds4f_load_dense(m, &B, &ly->wq_b,   DS4F_LN("attn.wq_b"));
        ds4f_load_dense(m, &B, &ly->wkv,    DS4F_LN("attn.wkv"));
        ds4f_load_dense(m, &B, &ly->wo_a,   DS4F_LN("attn.wo_a"));
        ds4f_load_dense(m, &B, &ly->wo_b,   DS4F_LN("attn.wo_b"));
        ds4f_load_raw(m, &B, ly->attn_sink, DS4F_LN("attn.attn_sink"),       DS4F_F32, 1, cfg.n_heads);
        ds4f_load_dense(m, &B, &ly->gate,   DS4F_LN("ffn.gate"));
        if (ly->gate_bias)   /* noaux_tc selection bias (F32[n_experts]); non-hash layers only */
            ds4f_load_raw(m, &B, ly->gate_bias, DS4F_LN("ffn.gate.bias"), DS4F_F32, 1, cfg.n_experts);
        ds4f_load_dense(m, &B, &ly->sh_w1,  DS4F_LN("ffn.shared_experts.w1"));
        ds4f_load_dense(m, &B, &ly->sh_w3,  DS4F_LN("ffn.shared_experts.w3"));
        ds4f_load_dense(m, &B, &ly->sh_w2,  DS4F_LN("ffn.shared_experts.w2"));
        for (int s = 0; s < no; s++) {
            int e = ly->owned_eid[s];
            snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w1", L, e); ds4f_load_q(m, &B, &ly->ex_w1[s], nm);
            snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w3", L, e); ds4f_load_q(m, &B, &ly->ex_w3[s], nm);
            snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w2", L, e); ds4f_load_q(m, &B, &ly->ex_w2[s], nm);
        }
        ds4f_load_raw(m, &B, ly->hc_attn_fn,    DS4F_LN("hc_attn_fn"),    DS4F_F32, mix, cfg.hc_mult * C);
        ds4f_load_raw(m, &B, ly->hc_attn_base,  DS4F_LN("hc_attn_base"),  DS4F_F32, 1, mix);
        ds4f_load_raw(m, &B, ly->hc_attn_scale, DS4F_LN("hc_attn_scale"), DS4F_F32, 1, 3);
        ds4f_load_raw(m, &B, ly->hc_ffn_fn,     DS4F_LN("hc_ffn_fn"),     DS4F_F32, mix, cfg.hc_mult * C);
        ds4f_load_raw(m, &B, ly->hc_ffn_base,   DS4F_LN("hc_ffn_base"),   DS4F_F32, 1, mix);
        ds4f_load_raw(m, &B, ly->hc_ffn_scale,  DS4F_LN("hc_ffn_scale"),  DS4F_F32, 1, 3);
        if (m->tierb2 && cfg.compress_ratios[L]) {
            /* Tier-B2 layer compressor + (CSA-only) lightning indexer. Matvec weights
             * stored bf16 (BF16 src = plain copy; FP8 wq_b -> bf16, lossless), F32 ape
             * direct, BF16 norm direct -- shapes mirror ds4f_alloc_tb2. The indexer's
             * internal compressor is always coff=2. */
            int ratio = cfg.compress_ratios[L];
            int coff = (ratio == 4) ? 2 : 1, W = coff * cfg.kv_lora;
            ds4f_load_raw        (m, &B, ly->cmp_wkv,   DS4F_LN("attn.compressor.wkv.weight"),   DS4F_BF16, W, C);
            ds4f_load_raw        (m, &B, ly->cmp_wgate, DS4F_LN("attn.compressor.wgate.weight"), DS4F_BF16, W, C);
            ds4f_load_raw        (m, &B, ly->cmp_ape,   DS4F_LN("attn.compressor.ape"),  DS4F_F32, ratio, W);
            ds4f_load_raw        (m, &B, ly->cmp_norm,  DS4F_LN("attn.compressor.norm.weight"), DS4F_BF16, 1, cfg.kv_lora);
            if (ratio == 4) {                          /* CSA layer => indexer present */
                int iW = 2 * cfg.index_head_dim;       /* indexer compressor coff=2 */
                ds4f_load_fp8_to_bf16(m, &B, ly->idx_wq_b,  DS4F_LN("attn.indexer.wq_b"),
                                      cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora);
                ds4f_load_raw        (m, &B, ly->idx_wproj, DS4F_LN("attn.indexer.weights_proj.weight"),
                                      DS4F_BF16, cfg.index_n_heads, C);
                ds4f_load_raw        (m, &B, ly->idx_cmp_wkv,   DS4F_LN("attn.indexer.compressor.wkv.weight"),   DS4F_BF16, iW, C);
                ds4f_load_raw        (m, &B, ly->idx_cmp_wgate, DS4F_LN("attn.indexer.compressor.wgate.weight"), DS4F_BF16, iW, C);
                ds4f_load_raw        (m, &B, ly->idx_cmp_ape,   DS4F_LN("attn.indexer.compressor.ape"),  DS4F_F32, ratio, iW);
                ds4f_load_raw        (m, &B, ly->idx_cmp_norm,  DS4F_LN("attn.indexer.compressor.norm.weight"), DS4F_BF16, 1, cfg.index_head_dim);
            }
        }
        #undef DS4F_LN
    }

    double loaded_gb = (double)m->bytes_read / 1e9;
    int n_tensors = B.n; uint64_t staged = B.total_bytes;
    m->bytes_read = 0;                 /* runner resets per token; start clean */
    ds4f_blob_close(&B);               /* arena holds the copies; release the blob mmap */

    /* ---- scratch (identical to ds4f_alloc_synth) ---- */
    int H = cfg.n_heads * cfg.q_head_dim;
    m->s_hn    = (float *)aligned_alloc(256, (size_t)C * 4);
    m->s_qlat  = (float *)aligned_alloc(256, (size_t)cfg.q_lora * 4);
    m->s_q     = (float *)aligned_alloc(256, (size_t)H * 4);
    m->s_kvlat = (float *)aligned_alloc(256, (size_t)cfg.kv_lora * 4);
    m->s_attn  = (float *)aligned_alloc(256, (size_t)H * 4);
    m->s_oin   = (float *)aligned_alloc(256, (size_t)C * 4);
    m->s_o1    = (float *)aligned_alloc(256, (size_t)cfg.o_inter * 4);
    m->s_o     = (float *)aligned_alloc(256, (size_t)C * 4);
    m->s_h2    = (float *)aligned_alloc(256, (size_t)C * 4);
    m->s_router= (float *)aligned_alloc(256, (size_t)cfg.n_experts * 4);
    m->s_shg   = (float *)aligned_alloc(256, (size_t)cfg.shared_inter * 4);
    m->s_shu   = (float *)aligned_alloc(256, (size_t)cfg.shared_inter * 4);
    m->s_exg   = (float *)aligned_alloc(256, (size_t)cfg.moe_inter * 4);
    m->s_exu   = (float *)aligned_alloc(256, (size_t)cfg.moe_inter * 4);
    m->s_moe   = (float *)aligned_alloc(256, (size_t)C * 4);
    m->s_route = (float *)aligned_alloc(256, (size_t)C * 4);
    m->s_logits= (float *)aligned_alloc(256, (size_t)cfg.vocab * 4);
    m->idx_blk_stride = cfg.max_pos > cfg.index_topk ? cfg.max_pos : cfg.index_topk;
    m->s_idx_scores = (float *)aligned_alloc(256, (size_t)n_threads * m->idx_blk_stride * 4);
    m->s_idx_sel    = (int   *)aligned_alloc(256, (size_t)n_threads * cfg.index_topk * 4);
    m->s_x4    = (float *)aligned_alloc(256, (size_t)cfg.hc_mult * C * 4);
    m->s_resid = (float *)aligned_alloc(256, (size_t)cfg.hc_mult * C * 4);
    m->s_xc    = (float *)aligned_alloc(256, (size_t)C * 4);

    double el = ds4f_wall() - t0;
    fprintf(stderr,
        "ds4f_load_real rank %d/%d: %d staged tensors, loaded %.2f GB "
        "(FP8 e4m3fn dense + MXFP4 experts repacked, %d owned), arena %.2f GB, %.1f s, %.2f GB/s\n",
        ep_rank, ep_size, n_tensors, loaded_gb, no, (double)m->arena_used / 1e9,
        el, el > 0 ? loaded_gb / el : 0.0);
    (void)staged;
    ds4f_build_freqs(m);   /* RoPE/YaRN tables (only when exact) */
    return m;
}

/* ===================== attention (pooled over heads) ===================== */
typedef struct { ds4f_model *m; ds4f_layer *ly; int pos; float scale; int ratio; } ds4f_attn_task;

static void ds4f_attn_worker(void *arg, int tid, int nthr) {
    ds4f_attn_task *T = (ds4f_attn_task *)arg;
    ds4f_model *m = T->m; ds4f_layer *ly = T->ly;
    int HD = m->cfg.q_head_dim, KV = m->cfg.kv_lora, nP = T->pos + 1;
    /* split heads across threads */
    int nh = m->cfg.n_heads;
    int per = nh / nthr, extra = nh % nthr;
    int h0 = per*tid + (tid < extra ? tid : extra);
    int h1 = h0 + per + (tid < extra ? 1 : 0);
    /* ---- sparse lightning-indexer gate (Stage 4) ----
     * On a sparse layer (ratio R>0) with more positions than the index budget,
     * a cheap compressed index over ceil(nP/R) blocks selects the top-mBlk
     * blocks (=> up to index_topk positions); the full-KV softmax+weighted-V
     * then runs over ONLY that subset -> O(topk) instead of O(nP) at long ctx.
     * When off, this is byte-identical to the dense path below. */
    int R = T->ratio, topk = m->cfg.index_topk, idim = m->cfg.index_head_dim;
    int do_sparse = m->sparse && R > 0 && nP > topk;
    if (idim > KV) idim = KV;
    if (do_sparse) {
        int nBlk = (nP + R - 1) / R;
        int mBlk = (topk + R - 1) / R; if (mBlk > nBlk) mBlk = nBlk;
        /* per-thread scratch: block scores + selected positions */
        float *bs  = m->s_idx_scores + (size_t)tid * m->idx_blk_stride;
        int   *sel = m->s_idx_sel    + (size_t)tid * topk;
        for (int h = h0; h < h1; h++) {
            const float *q = m->s_q + (size_t)h*HD;
            /* (1) compressed index: representative key = block's first position
             * latent, dot over the first idim index dims (O(nBlk*idim)). */
            for (int b = 0; b < nBlk; b++) {
                const float *kc = ly->kv_cache + (size_t)(b*R)*KV;
                float s = 0.f;
                for (int d = 0; d < idim; d++) s += q[d] * kc[d];
                bs[b] = s;
            }
            /* (2) select top-mBlk blocks by index score (partial selection;
             * mBlk is small relative to nBlk so O(nBlk*mBlk) is acceptable,
             * and mBlk<=ceil(topk/R) caps it). */
            /* gather selected positions into sel[] (<= topk) */
            int nsel = 0;
            {
                /* simple selection: repeatedly take the current max block,
                 * mark it consumed; cheaper than a heap at these sizes. */
                /* use a tiny consumed bitmap in the high bit of bs via a
                 * parallel pass: copy scores, then mBlk linear-max scans. */
                for (int t = 0; t < mBlk; t++) {
                    int best = -1; float bv = -1e30f;
                    for (int b = 0; b < nBlk; b++)
                        if (bs[b] > bv) { bv = bs[b]; best = b; }
                    if (best < 0) break;
                    bs[best] = -1e30f;                 /* consume */
                    int p0 = best*R, p1 = p0 + R; if (p1 > nP) p1 = nP;
                    for (int p = p0; p < p1 && nsel < topk; p++) sel[nsel++] = p;
                }
            }
            /* (3) full-KV softmax(sink) + weighted-V over the selected subset */
            float mx = -1e30f;
            /* reuse bs as the score buffer for selected positions (nsel<=topk
             * <= nBlk*R; bs has idx_blk_stride >= nBlk slots, and nsel<=topk
             * which is <= stride, so this is in-bounds). */
            float *sc = bs;
            for (int j = 0; j < nsel; j++) {
                const float *kc = ly->kv_cache + (size_t)sel[j]*KV;
                float s = 0.f;
                for (int d = 0; d < KV; d++) s += q[d] * kc[d];
                s *= T->scale; sc[j] = s; if (s > mx) mx = s;
            }
            float snk = ly->attn_sink[h];
            float denom = expf(snk - mx);
            for (int j = 0; j < nsel; j++) { sc[j] = expf(sc[j] - mx); denom += sc[j]; }
            float inv = 1.0f / denom;
            float *out = m->s_attn + (size_t)h*HD;
            for (int d = 0; d < HD; d++) out[d] = 0.f;
            for (int j = 0; j < nsel; j++) {
                float w = sc[j]*inv;
                const float *kc = ly->kv_cache + (size_t)sel[j]*KV;
                for (int d = 0; d < KV; d++) out[d] += w * kc[d];   /* V = latent */
            }
        }
        return;
    }
    /* ---- dense path (all positions) ---- */
    float *sc = (float *)alloca((size_t)nP * 4);
    for (int h = h0; h < h1; h++) {
        const float *q = m->s_q + (size_t)h*HD;
        float mx = -1e30f;
        for (int p = 0; p < nP; p++) {
            const float *kc = ly->kv_cache + (size_t)p*KV;
            float s = 0.f;
            /* score over the kv_lora latent dims (q_head_dim >= kv_lora; use first KV dims) */
            for (int d = 0; d < KV; d++) s += q[d] * kc[d];
            s *= T->scale; sc[p] = s; if (s > mx) mx = s;
        }
        float snk = ly->attn_sink[h];
        float denom = expf(snk - mx);
        for (int p = 0; p < nP; p++) { sc[p] = expf(sc[p] - mx); denom += sc[p]; }
        float inv = 1.0f / denom;
        float *out = m->s_attn + (size_t)h*HD;
        for (int d = 0; d < HD; d++) out[d] = 0.f;
        for (int p = 0; p < nP; p++) {
            float w = sc[p]*inv;
            const float *kc = ly->kv_cache + (size_t)p*KV;
            for (int d = 0; d < KV; d++) out[d] += w * kc[d];   /* V = latent */
        }
    }
}

/* ===================== exact attention (sliding window + sink + de-rotate) =====
 * model.py Attention.forward, window term: q (already per-head-normed + RoPE'd)
 * scores the kv latent (RoPE'd, stored in cache) over the last `win` positions;
 * sink contributes exp(sink-max) to the denominator only; weighted-V over the
 * same latent; the output's rope dims are de-rotated by the QUERY position. The
 * long-range compressor/indexer term (sparse layers) is the Tier-B2 follow-up. */
typedef struct {
    ds4f_model *m; ds4f_layer *ly; int pos; float scale;
    int win, half; const float *rcos, *rsin;
} ds4f_attn_ex_task;

static void ds4f_attn_exact_worker(void *arg, int tid, int nthr) {
    ds4f_attn_ex_task *T = (ds4f_attn_ex_task *)arg;
    ds4f_model *m = T->m; ds4f_layer *ly = T->ly;
    int HD = m->cfg.q_head_dim, KV = m->cfg.kv_lora;
    int rd = m->cfg.qk_rope_dim, nope = HD - rd, half = T->half;
    int pos = T->pos, p_lo = pos - T->win + 1; if (p_lo < 0) p_lo = 0;
    int nP = pos - p_lo + 1;
    int nh = m->cfg.n_heads, per = nh / nthr, extra = nh % nthr;
    int h0 = per*tid + (tid < extra ? tid : extra);
    int h1 = h0 + per + (tid < extra ? 1 : 0);
    float *sc = (float *)alloca((size_t)nP * 4);
    for (int h = h0; h < h1; h++) {
        const float *q = m->s_q + (size_t)h*HD;
        float mx = -1e30f;
        for (int j = 0; j < nP; j++) {
            const float *kc = ly->kv_cache + (size_t)(p_lo + j)*KV;
            float s = 0.f;
            for (int d = 0; d < KV; d++) s += q[d]*kc[d];      /* HD==KV (512) */
            s *= T->scale; sc[j] = s; if (s > mx) mx = s;
        }
        float snk = ly->attn_sink[h];
        float denom = expf(snk - mx);
        for (int j = 0; j < nP; j++) { sc[j] = expf(sc[j] - mx); denom += sc[j]; }
        float inv = 1.0f/denom;
        float *out = m->s_attn + (size_t)h*HD;
        for (int d = 0; d < HD; d++) out[d] = 0.f;
        for (int j = 0; j < nP; j++) {
            float w = sc[j]*inv;
            const float *kc = ly->kv_cache + (size_t)(p_lo + j)*KV;
            for (int d = 0; d < HD; d++) out[d] += w*kc[d];
        }
        ds4f_rope_apply(out + nope, T->rcos, T->rsin, pos, half, 1);  /* de-rotate */
    }
}

/* exact q: per-head RMS-normalize (no weight) over q_head_dim, then RoPE last rd. */
static void ds4f_q_norm_rope(ds4f_model *m, float *q, int pos,
                             const float *rcos, const float *rsin) {
    ds4f_config *c = &m->cfg;
    int HD = c->q_head_dim, rd = c->qk_rope_dim, half = rd/2, nope = HD - rd;
    for (int h = 0; h < c->n_heads; h++) {
        float *qh = q + (size_t)h*HD;
        double ss = 0.0; for (int d = 0; d < HD; d++) ss += (double)qh[d]*qh[d];
        float inv = 1.0f/sqrtf((float)(ss/HD) + c->norm_eps);
        for (int d = 0; d < HD; d++) qh[d] *= inv;
        ds4f_rope_apply(qh + nope, rcos, rsin, pos, half, 0);
    }
}

/* ===================== Tier-B2 attention worker + prepare =====================
 * tb2 worker = the exact sliding-window worker PLUS the compressed-KV term folded
 * into the SAME online softmax. The window latents (ly->kv_cache, RoPE'd @ abs pos)
 * and the compressed latents (ly->cmp_kv, RoPE'd @ their block's first token) both
 * score against q (RoPE'd @ query pos); sink contributes exp(sink-max) to the
 * denominator once; weighted-V over the union; the output rope dims are de-rotated
 * by the QUERY position (model.py de-rotates o once, regardless of kv origin).
 * m->s_tb2_sel[0..nsel) holds LOCAL indices into ly->cmp_kv (already offset-stripped
 * by ds4f_tb2_prepare). nsel==0 => identical to the window-only exact worker. */
static void ds4f_attn_tb2_worker(void *arg, int tid, int nthr) {
    ds4f_attn_ex_task *T = (ds4f_attn_ex_task *)arg;
    ds4f_model *m = T->m; ds4f_layer *ly = T->ly;
    int HD = m->cfg.q_head_dim, KV = m->cfg.kv_lora;
    int rd = m->cfg.qk_rope_dim, nope = HD - rd;
    int pos = T->pos, p_lo = pos - T->win + 1; if (p_lo < 0) p_lo = 0;
    int nP = pos - p_lo + 1;
    int nsel = m->s_tb2_nsel, total = nP + nsel;
    const int *sel = m->s_tb2_sel; const float *cmp = ly->cmp_kv;
    int nh = m->cfg.n_heads, per = nh / nthr, extra = nh % nthr;
    int h0 = per*tid + (tid < extra ? tid : extra);
    int h1 = h0 + per + (tid < extra ? 1 : 0);
    float *sc = (float *)alloca((size_t)(total > 0 ? total : 1) * 4);
    for (int h = h0; h < h1; h++) {
        const float *q = m->s_q + (size_t)h*HD;
        float mx = -1e30f;
        for (int j = 0; j < nP; j++) {                          /* window term */
            const float *kc = ly->kv_cache + (size_t)(p_lo + j)*KV;
            float s = 0.f; for (int d = 0; d < KV; d++) s += q[d]*kc[d];
            s *= T->scale; sc[j] = s; if (s > mx) mx = s;
        }
        for (int j = 0; j < nsel; j++) {                        /* compressed term */
            const float *kc = cmp + (size_t)sel[j]*KV;
            float s = 0.f; for (int d = 0; d < KV; d++) s += q[d]*kc[d];
            s *= T->scale; sc[nP + j] = s; if (s > mx) mx = s;
        }
        float denom = expf(ly->attn_sink[h] - mx);
        for (int j = 0; j < total; j++) { sc[j] = expf(sc[j] - mx); denom += sc[j]; }
        float inv = 1.0f/denom;
        float *out = m->s_attn + (size_t)h*HD;
        for (int d = 0; d < HD; d++) out[d] = 0.f;
        for (int j = 0; j < nP; j++) {
            float w = sc[j]*inv; const float *kc = ly->kv_cache + (size_t)(p_lo + j)*KV;
            for (int d = 0; d < HD; d++) out[d] += w*kc[d];
        }
        for (int j = 0; j < nsel; j++) {
            float w = sc[nP + j]*inv; const float *kc = cmp + (size_t)sel[j]*KV;
            for (int d = 0; d < HD; d++) out[d] += w*kc[d];
        }
        ds4f_rope_apply(out + nope, T->rcos, T->rsin, pos, T->half, 1);  /* de-rotate @ query pos */
    }
}

/* Step the per-layer compressor (and, on CSA layers, the indexer) for the current
 * token at absolute position `pos`, then fill m->s_tb2_sel/s_tb2_nsel with the LOCAL
 * compressed indices this query attends. Token-at-a-time: pos==0 seeds the ring state
 * (no compressed token yet), pos>=1 decodes; on a compress boundary a new compressed
 * latent lands in ly->cmp_kv[pos/ratio]. Input is m->s_hn (attn_norm output); the
 * indexer's q-lora input is m->s_qlat (q_norm(wq_a(s_hn))). For HCA(128) layers every
 * available compressed token is attended (arange); for CSA(4) the indexer scores them
 * and selects top-min(index_topk, T). ratio==0 (dense) => no-op, nsel=0. */
static void ds4f_tb2_prepare(ds4f_model *m, ds4f_layer *ly, int ratio, int pos,
                             const float *rcos, const float *rsin) {
    m->s_tb2_nsel = 0;
    if (ratio == 0) return;
    ds4f_config *c = &m->cfg;
    int KV = c->kv_lora, ihd = c->index_head_dim, rd = c->qk_rope_dim; float eps = c->norm_eps;
    int offset = c->window_size;                                /* decode combined-buffer offset */
    /* layer compressor (rotate=0): input s_hn -> cmp_kv[pos/ratio] on a boundary */
    if (ds4f_compress_step(m->s_hn, c->hidden, KV, rd, ratio, pos,
                           ly->cmp_wkv, ly->cmp_wgate, 1, ly->cmp_ape, ly->cmp_norm,
                           rcos, rsin, eps, 0,
                           ly->cmp_kv_state, ly->cmp_score_state, m->s_cmp_out, m->pool))
        memcpy(ly->cmp_kv + (size_t)(pos/ratio)*KV, m->s_cmp_out, (size_t)KV*4);
    int T = (pos + 1) / ratio;
    if (ratio == 4) {                                           /* CSA: indexer-selected */
        if (pos == 0) {                                         /* seed indexer compressor ring */
            float *seed = (float *)alloca((size_t)ihd * 4);     /* index_step drives it for pos>=1 */
            ds4f_compress_step(m->s_hn, c->hidden, ihd, rd, ratio, 0,
                               ly->idx_cmp_wkv, ly->idx_cmp_wgate, 1, ly->idx_cmp_ape, ly->idx_cmp_norm,
                               rcos, rsin, eps, 1,
                               ly->idx_cmp_kv_state, ly->idx_cmp_score_state, seed, m->pool);
            return;                                             /* T==0, nothing compressed yet */
        }
        int k = c->index_topk;
        ds4f_index_step(m->s_hn, c->hidden, m->s_qlat, c->q_lora,
                        c->index_n_heads, ihd, rd, ratio, pos, offset, k,
                        ly->idx_wq_b, ly->idx_wproj, 1,
                        ly->idx_cmp_wkv, ly->idx_cmp_wgate, ly->idx_cmp_ape, ly->idx_cmp_norm,
                        rcos, rsin, eps,
                        ly->idx_cmp_kv_state, ly->idx_cmp_score_state, ly->idx_kv,
                        m->s_idx_q, m->s_idx_score, m->s_tb2_sel, m->pool);
        int nsel = 0;                                           /* compact + strip offset -> local idx */
        for (int i = 0; i < k; i++) { int v = m->s_tb2_sel[i]; if (v < 0) break; m->s_tb2_sel[nsel++] = v - offset; }
        m->s_tb2_nsel = nsel;
    } else {                                                    /* HCA: all compressed tokens */
        for (int t = 0; t < T; t++) m->s_tb2_sel[t] = t;
        m->s_tb2_nsel = T;
    }
}

/* ===================== MoE routing ===================== */
static void ds4f_topk(const float *logits, int n, int k, int *idx, float *wt, float routed_scale) {
    /* sqrtsoftplus score, top-k by score, norm_topk_prob, routed_scale */
    for (int i = 0; i < k; i++) { idx[i] = -1; wt[i] = -1e30f; }
    for (int e = 0; e < n; e++) {
        float z = logits[e];                     /* numerically-stable softplus */
        float sp = (z > 0.f) ? z + log1pf(expf(-z)) : log1pf(expf(z));
        float sc = sqrtf(sp < 0 ? 0 : sp);       /* sqrtsoftplus */
        int lo = 0; for (int j = 1; j < k; j++) if (wt[j] < wt[lo]) lo = j;
        if (sc > wt[lo]) { wt[lo] = sc; idx[lo] = e; }
    }
    float sum = 0.f; for (int i = 0; i < k; i++) if (idx[i] >= 0) sum += wt[i];
    if (sum <= 0) sum = 1.f;
    for (int i = 0; i < k; i++) wt[i] = (wt[i]/sum) * routed_scale;
}

/* Exact DeepSeek-V4-Flash gate (model.py Gate.forward, score_func=sqrtsoftplus):
 *   score[e] = sqrt(softplus(logit[e]))          (UNBIASED)
 *   select top-k by (score[e] + bias[e])         (bias for SELECTION ONLY)
 *   weight[k] = score[idx[k]] (unbiased) / sum(selected scores) * route_scale
 * With bias==NULL (synthetic, or hash layers where token-id routing is deferred)
 * this is byte-identical to ds4f_topk. */
static void ds4f_topk_exact(const float *logits, const float *bias, int n, int k,
                            int *idx, float *wt, float routed_scale) {
    float selkey[8], selsc[8];                   /* k <= n_active <= 8 */
    for (int i = 0; i < k; i++) { idx[i] = -1; selkey[i] = -1e30f; selsc[i] = 0.f; }
    for (int e = 0; e < n; e++) {
        float z = logits[e];
        float sp = (z > 0.f) ? z + log1pf(expf(-z)) : log1pf(expf(z));
        float sc = sqrtf(sp < 0 ? 0 : sp);       /* unbiased sqrtsoftplus score */
        float key = sc + (bias ? bias[e] : 0.f); /* biased selection key */
        int lo = 0; for (int j = 1; j < k; j++) if (selkey[j] < selkey[lo]) lo = j;
        if (key > selkey[lo]) { selkey[lo] = key; selsc[lo] = sc; idx[lo] = e; }
    }
    float sum = 0.f; for (int i = 0; i < k; i++) if (idx[i] >= 0) sum += selsc[i];
    if (sum <= 0) sum = 1.f;
    for (int i = 0; i < k; i++) wt[i] = (selsc[i] / sum) * routed_scale;
}

/* ===================== synthetic KV warm (ctx benchmark) =====================
 * Fill every layer's KV cache positions [0,npos) with bounded synthetic latents
 * so decode-attn cost at a large context can be measured WITHOUT running npos
 * real prefill tokens. After this, decode at pos>=npos. Synthetic-only: the
 * latent values are deterministic junk; the point is the position COUNT that
 * the attention loop (dense O(nP) vs sparse O(topk)) iterates over. */
static void ds4f_warm_kv(ds4f_model *m, int npos) {
    int KV = m->cfg.kv_lora;
    if (npos > m->cfg.max_pos) npos = m->cfg.max_pos;
    for (int L = 0; L < m->cfg.n_layers; L++) {
        float *kc = m->layers[L].kv_cache;
        uint64_t s = 0x4B7700D5F0ull ^ ((uint64_t)L << 40);
        for (int p = 0; p < npos; p++) {
            float *row = kc + (size_t)p*KV;
            for (int d = 0; d < KV; d++) {
                s += 0x9E3779B97F4A7C15ull;
                uint64_t z = s; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= (z >> 31);
                row[d] = (float)((double)(z >> 11) / (double)(1ull << 53)) * 2.0f - 1.0f;
            }
        }
    }
}

/* ===================== Tier-B2 compressed-state warm (ctx benchmark) =====================
 * Fill every sparse layer's compressed caches (cmp_kv, and idx_kv on CSA layers) with
 * bounded synthetic latents so a decode at pos=npos exercises the REAL long-context
 * indexer scan (index_score over T=npos/ratio compressed positions) + the capped
 * top-k compressed-attention term, WITHOUT running npos real prefill tokens. Pairs with
 * ds4f_warm_kv (which fills the window kv_cache). Synthetic-only: values are deterministic
 * junk; the point is the POSITION COUNT the O(ctx) index scan and O(topk) gather iterate. */
static void ds4f_warm_tb2(ds4f_model *m, int npos) {
    if (!m->tierb2) return;
    ds4f_config *c = &m->cfg;
    int KV = c->kv_lora, ihd = c->index_head_dim;
    if (npos > c->max_pos) npos = c->max_pos;
    for (int L = 0; L < c->n_layers; L++) {
        int ratio = c->compress_ratios[L];
        if (ratio == 0) continue;
        ds4f_layer *ly = &m->layers[L];
        int nslot = c->max_pos / ratio;
        int T = (npos + 1) / ratio; if (T > nslot) T = nslot;
        uint64_t s = 0xC0FFEE5F0ull ^ ((uint64_t)L << 40);
        for (int t = 0; t < T; t++) {
            float *kr = ly->cmp_kv + (size_t)t*KV;
            for (int d = 0; d < KV; d++) {
                s += 0x9E3779B97F4A7C15ull; uint64_t z = s;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= (z >> 31);
                kr[d] = (float)((double)(z >> 11) / (double)(1ull << 53)) * 2.0f - 1.0f;
            }
            if (ratio == 4) {                                  /* indexer compressed key */
                float *ir = ly->idx_kv + (size_t)t*ihd;
                for (int d = 0; d < ihd; d++) {
                    s += 0x9E3779B97F4A7C15ull; uint64_t z = s;
                    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= (z >> 31);
                    ir[d] = (float)((double)(z >> 11) / (double)(1ull << 53)) * 2.0f - 1.0f;
                }
            }
        }
    }
}

/* ===================== single-token forward ===================== */
static int ds4f_prof_on = -1;
static inline double ds4f_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
/* TIC/TOC accumulate into m->prof[id] when DS4F_PROF=1 */
#define DS4F_TIC() double _t0 = ds4f_prof_on ? ds4f_now() : 0.0
#define DS4F_TOC(id) do { if (ds4f_prof_on) m->prof[id] += ds4f_now() - _t0; } while (0)

static int ds4f_dbg = -1;
static void ds4f_chk(const char *tag, int L, const float *v, int n) {
    if (ds4f_dbg < 0) { const char *e = getenv("DS4F_DEBUG"); ds4f_dbg = e ? atoi(e) : 0; }
    if (!ds4f_dbg) return;
    double ss = 0.0; int nan = 0, inf = 0; float mx = 0;
    for (int i = 0; i < n; i++) {
        float a = v[i];
        if (!(a == a)) nan++;
        else if (a > 3.0e38f || a < -3.0e38f) inf++;
        else { ss += (double)a*a; float aa = a < 0 ? -a : a; if (aa > mx) mx = aa; }
    }
    fprintf(stderr, "  L%-2d %-10s ||.||=%.3e max=%.3e nan=%d inf=%d\n",
            L, tag, sqrt(ss), mx, nan, inf);
}

/* ===================== mHC (Hyper-Connections) — exact ===================== */
static inline float ds4f_sigmoidf(float x){ return 1.0f/(1.0f+expf(-x)); }

/* hc_split_sinkhorn (one token): mixes[(2+hc)*hc] -> pre[hc], post[hc],
 * comb[hc*hc] (row-major comb[j*hc+k]). Mirrors kernel.py hc_split_sinkhorn_kernel:
 *   pre[j]  = sigmoid(mixes[j]    *scale[0]+base[j])     + eps
 *   post[j] = 2*sigmoid(mixes[j+hc]*scale[1]+base[j+hc])
 *   comb[j,k]= mixes[j*hc+k+2hc] *scale[2]+base[...]
 * then row-softmax(+eps), col-normalize(/+eps), and (iters-1) {row,col}-normalize. */
static void ds4f_hc_sinkhorn(const float *mixes, const float *scale, const float *base,
                             int hc, int iters, float eps,
                             float *pre, float *post, float *comb) {
    for (int j = 0; j < hc; j++)
        pre[j]  = ds4f_sigmoidf(mixes[j]*scale[0] + base[j]) + eps;
    for (int j = 0; j < hc; j++)
        post[j] = 2.0f*ds4f_sigmoidf(mixes[j+hc]*scale[1] + base[j+hc]);
    for (int j = 0; j < hc; j++)
        for (int k = 0; k < hc; k++)
            comb[j*hc+k] = mixes[j*hc + k + 2*hc]*scale[2] + base[j*hc + k + 2*hc];
    /* comb = comb.softmax(-1) + eps  (per row j) */
    for (int j = 0; j < hc; j++) {
        float mx = comb[j*hc];
        for (int k = 1; k < hc; k++) if (comb[j*hc+k] > mx) mx = comb[j*hc+k];
        float s = 0.f;
        for (int k = 0; k < hc; k++) { float e = expf(comb[j*hc+k]-mx); comb[j*hc+k] = e; s += e; }
        for (int k = 0; k < hc; k++) comb[j*hc+k] = comb[j*hc+k]/s + eps;
    }
    /* comb = comb / (comb.sum(-2) + eps)  (per col k) */
    for (int k = 0; k < hc; k++) {
        float cs = 0.f; for (int j = 0; j < hc; j++) cs += comb[j*hc+k];
        cs += eps; for (int j = 0; j < hc; j++) comb[j*hc+k] /= cs;
    }
    for (int it = 0; it < iters-1; it++) {
        for (int j = 0; j < hc; j++) {                 /* row-normalize */
            float rs = 0.f; for (int k = 0; k < hc; k++) rs += comb[j*hc+k];
            rs += eps; for (int k = 0; k < hc; k++) comb[j*hc+k] /= rs;
        }
        for (int k = 0; k < hc; k++) {                 /* col-normalize */
            float cs = 0.f; for (int j = 0; j < hc; j++) cs += comb[j*hc+k];
            cs += eps; for (int j = 0; j < hc; j++) comb[j*hc+k] /= cs;
        }
    }
}

/* hc_pre: x4[hc*C] (4 streams) -> collapsed y[C]; also yields post[hc], comb[hc*hc].
 * mixes = (fn @ flatten(x4)) * rsqrt(mean(x4^2)+norm_eps); sinkhorn; y[d]=Σ_k pre[k]·x4[k,d]. */
static void ds4f_hc_pre(ds4f_model *m, const float *x4, const float *fn,
                        const float *scale, const float *base,
                        float *y, float *post, float *comb) {
    ds4f_config *c = &m->cfg;
    int hc = c->hc_mult, C = c->hidden, hd = hc*C, mix_hc = (2+hc)*hc;
    double ss = 0.0; for (int i = 0; i < hd; i++) { float v = x4[i]; ss += (double)v*v; }
    float rsq = 1.0f/sqrtf((float)(ss/hd) + c->norm_eps);
    float mixes[64];                 /* mix_hc <= 24 for hc<=4 */
    ds4f_tensor fnt = { (void*)fn, NULL, DS4F_F32, mix_hc, hd };
    ds4f_matvec(m, mixes, &fnt, x4); /* threaded F32 Linear */
    for (int mm = 0; mm < mix_hc; mm++) mixes[mm] *= rsq;
    float pre[16];
    ds4f_hc_sinkhorn(mixes, scale, base, hc, c->hc_iters, c->hc_eps, pre, post, comb);
    for (int d = 0; d < C; d++) {
        float a = 0.f; for (int k = 0; k < hc; k++) a += pre[k]*x4[(size_t)k*C+d];
        y[d] = a;
    }
}

/* hc_post: expand block output f[C] back to hc streams x4[hc*C], folding the
 * pre-block residual: x4[k,d] = post[k]·f[d] + Σ_j comb[j,k]·resid[j,d]. */
static void ds4f_hc_post(ds4f_model *m, float *x4, const float *resid, const float *f,
                         const float *post, const float *comb) {
    int hc = m->cfg.hc_mult, C = m->cfg.hidden;
    for (int k = 0; k < hc; k++) {
        float pk = post[k]; float *ok = x4 + (size_t)k*C;
        for (int d = 0; d < C; d++) ok[d] = pk*f[d];
        for (int j = 0; j < hc; j++) {
            float cjk = comb[j*hc+k]; const float *rj = resid + (size_t)j*C;
            for (int d = 0; d < C; d++) ok[d] += cjk*rj[d];
        }
    }
}

/* hc_head: final collapse hc streams x4[hc*C] -> y[C] via per-stream sigmoid gate
 * (sigmoid(mixes*scale+base)+eps), NO sinkhorn. Mirrors ParallelHead.hc_head. */
static void ds4f_hc_head(ds4f_model *m, const float *x4, float *y) {
    ds4f_config *c = &m->cfg;
    int hc = c->hc_mult, C = c->hidden, hd = hc*C;
    double ss = 0.0; for (int i = 0; i < hd; i++) { float v = x4[i]; ss += (double)v*v; }
    float rsq = 1.0f/sqrtf((float)(ss/hd) + c->norm_eps);
    float mixes[16];
    ds4f_tensor fnt = { (void*)m->hc_head_fn, NULL, DS4F_F32, hc, hd };
    ds4f_matvec(m, mixes, &fnt, x4);
    float pre[16];
    for (int k = 0; k < hc; k++)
        pre[k] = ds4f_sigmoidf(mixes[k]*rsq*m->hc_head_scale[0] + m->hc_head_base[k]) + c->hc_eps;
    for (int d = 0; d < C; d++) {
        float a = 0.f; for (int k = 0; k < hc; k++) a += pre[k]*x4[(size_t)k*C+d];
        y[d] = a;
    }
}

/* Runs MLA + MoE for one token at position `pos`, hidden state in/out `x`[hidden].
 * Returns next-token argmax of synthetic logits (meaningless). */
static int ds4f_forward_token(ds4f_model *m, float *x, int pos) {
    ds4f_config *c = &m->cfg;
    int C = c->hidden, HD = c->q_head_dim, KV = c->kv_lora, H = c->n_heads*HD;
    float eps = 1e-6f;
    if (ds4f_prof_on < 0) { const char *e = getenv("DS4F_PROF"); ds4f_prof_on = e ? atoi(e) : 0; }
    /* exact mHC: carry x as hc_mult residual streams (s_x4). Expand the single
     * embedding x[C] into 4 identical streams; collapse back at the head. When
     * m->mhc==0 the plain-residual stand-in below runs unchanged (byte-identical). */
    int hc = c->hc_mult; size_t hcC = (size_t)hc*C;
    float post_a[16], comb_a[64], post_f[16], comb_f[64];   /* per-block sinkhorn weights */
    if (m->mhc) for (int k = 0; k < hc; k++) memcpy(m->s_x4 + (size_t)k*C, x, (size_t)C*4);
    for (int L = 0; L < c->n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        /* exact RoPE table for this layer: sparse layers (compress_ratio!=0) use
         * the YaRN/compress_rope_theta table, dense layers the plain rope_theta. */
        int ratio = c->compress_ratios[L];
        const float *rcos = m->exact ? (ratio ? m->rope_comp_cos : m->rope_dense_cos) : NULL;
        const float *rsin = m->exact ? (ratio ? m->rope_comp_sin : m->rope_dense_sin) : NULL;
        /* ---- mHC pre (attn): collapse 4 streams -> attn input; save residual ---- */
        float *asrc = x;
        if (m->mhc) { DS4F_TIC();
            ds4f_hc_pre(m, m->s_x4, ly->hc_attn_fn, ly->hc_attn_scale, ly->hc_attn_base,
                        m->s_xc, post_a, comb_a);
            memcpy(m->s_resid, m->s_x4, hcC*4);
            asrc = m->s_xc;
            ds4f_chk("hc_pre_a", L, asrc, C);
            DS4F_TOC(DS4F_P_OTHER); }
        /* ---- MLA: q/kv projections ---- */
        { DS4F_TIC();
        ds4f_rmsnorm(m->s_hn, asrc, ly->attn_norm, C, eps);
        ds4f_chk("attn_norm", L, m->s_hn, C);
        ds4f_matvec(m, m->s_qlat, &ly->wq_a, m->s_hn);
        ds4f_rmsnorm(m->s_qlat, m->s_qlat, ly->q_norm, c->q_lora, eps);
        ds4f_matvec(m, m->s_q, &ly->wq_b, m->s_qlat);            /* [n_heads*q_head_dim] */
        if (m->exact) ds4f_q_norm_rope(m, m->s_q, pos, rcos, rsin); /* per-head norm + RoPE */
        ds4f_chk("q", L, m->s_q, H);
        ds4f_matvec(m, m->s_kvlat, &ly->wkv, m->s_hn);          /* [kv_lora] */
        ds4f_rmsnorm(m->s_kvlat, m->s_kvlat, ly->kv_norm, KV, eps);
        if (m->exact)                                            /* RoPE the kv rope dims */
            ds4f_rope_apply(m->s_kvlat + (KV - c->qk_rope_dim), rcos, rsin, pos, c->qk_rope_dim/2, 0);
        ds4f_chk("kvlat", L, m->s_kvlat, KV);
        memcpy(ly->kv_cache + (size_t)pos*KV, m->s_kvlat, (size_t)KV*4);  /* append latent */
        DS4F_TOC(DS4F_P_QKV); }
        /* ---- Tier-B2 compressor/indexer step (timed apart from the attn worker) ---- */
        if (m->tierb2 && ratio) { DS4F_TIC();
            ds4f_tb2_prepare(m, ly, ratio, pos, rcos, rsin);  /* fills s_tb2_sel/nsel */
            DS4F_TOC(DS4F_P_TB2PREP); }
        /* ---- attention ---- */
        { DS4F_TIC();
        if (m->tierb2 && ratio) {
            /* window + indexer-selected compressed term (prepare ran above). */
            ds4f_attn_ex_task at = { m, ly, pos, 1.0f/sqrtf((float)HD),
                                     c->window_size, c->qk_rope_dim/2, rcos, rsin };
            ds4f_pool_run(m->pool, ds4f_attn_tb2_worker, &at);
        } else if (m->exact) {
            ds4f_attn_ex_task at = { m, ly, pos, 1.0f/sqrtf((float)HD),
                                     c->window_size, c->qk_rope_dim/2, rcos, rsin };
            ds4f_pool_run(m->pool, ds4f_attn_exact_worker, &at);
        } else {
            ds4f_attn_task at = { m, ly, pos, 1.0f/sqrtf((float)KV), c->compress_ratios[L] };
            ds4f_pool_run(m->pool, ds4f_attn_worker, &at);      /* fills s_attn[H] */
        }
        ds4f_chk("attn", L, m->s_attn, H);
        DS4F_TOC(DS4F_P_ATTN); }
        /* ---- o projection ---- */
        { DS4F_TIC();
        int og = c->o_groups; (void)H;
        if (m->exact) {
            /* grouped low-rank o-proj (block-diagonal wo_a, NO nonlinearity):
             * s_attn is [og groups, n_heads/og*q_head_dim = C] already de-rotated;
             * group g: o1[g*o_lora ..] = wo_a_rows[g*o_lora ..] @ s_attn[g*C ..].
             * wo_a is FP8 -> o_lora(1024) rows are 128-aligned per group. */
            int gin = H / og;                          /* 32768/8 = 4096 == C */
            for (int g = 0; g < og; g++) {
                ds4f_tensor vg = ds4f_row_slice(&ly->wo_a, g * c->o_lora, c->o_lora);
                ds4f_matvec(m, m->s_o1 + (size_t)g * c->o_lora, &vg, m->s_attn + (size_t)g * gin);
            }
            ds4f_matvec(m, m->s_o, &ly->wo_b, m->s_o1);         /* [hidden], no silu */
        } else {
            for (int i = 0; i < C; i++) m->s_oin[i] = 0.f;
            for (int g = 0; g < og; g++) for (int i = 0; i < C; i++) m->s_oin[i] += m->s_attn[g*C + i];
            ds4f_matvec(m, m->s_o1, &ly->wo_a, m->s_oin);       /* [o_inter] */
            for (int i = 0; i < c->o_inter; i++) m->s_o1[i] = ds4f_silu(m->s_o1[i]);  /* stand-in nonlin */
            ds4f_matvec(m, m->s_o, &ly->wo_b, m->s_o1);         /* [hidden] */
        }
        ds4f_chk("o", L, m->s_o, C);
        if (m->mhc) ds4f_hc_post(m, m->s_x4, m->s_resid, m->s_o, post_a, comb_a); /* expand 1->4 */
        else for (int i = 0; i < C; i++) x[i] += m->s_o[i];     /* plain-residual stand-in */
        ds4f_chk("x+attn", L, m->mhc ? m->s_x4 : x, C);
        DS4F_TOC(DS4F_P_OPROJ); }

        /* ---- mHC pre (ffn): collapse 4 streams -> ffn input; save residual ---- */
        float *fsrc = x;
        if (m->mhc) { DS4F_TIC();
            ds4f_hc_pre(m, m->s_x4, ly->hc_ffn_fn, ly->hc_ffn_scale, ly->hc_ffn_base,
                        m->s_xc, post_f, comb_f);
            memcpy(m->s_resid, m->s_x4, hcC*4);
            fsrc = m->s_xc;
            DS4F_TOC(DS4F_P_OTHER); }
        /* ---- MoE: shared expert ---- */
        ds4f_rmsnorm(m->s_h2, fsrc, ly->ffn_norm, C, eps);
        ds4f_chk("ffn_norm", L, m->s_h2, C);
        for (int i = 0; i < C; i++) { m->s_moe[i] = 0.f; m->s_route[i] = 0.f; }
        { DS4F_TIC();
        ds4f_matvec(m, m->s_shg, &ly->sh_w1, m->s_h2);
        ds4f_matvec(m, m->s_shu, &ly->sh_w3, m->s_h2);
        if (m->exact) { float lim = c->swiglu_limit;            /* clamp up both sides, gate max */
            for (int i = 0; i < c->shared_inter; i++)
                m->s_shg[i] = ds4f_silu(m->s_shg[i] > lim ? lim : m->s_shg[i]) * ds4f_clampf(m->s_shu[i], -lim, lim);
        } else
            for (int i = 0; i < c->shared_inter; i++) m->s_shg[i] = ds4f_silu(m->s_shg[i]) * m->s_shu[i];
        ds4f_chk("sh_gu", L, m->s_shg, c->shared_inter);
        ds4f_matvec(m, m->s_o, &ly->sh_w2, m->s_shg);            /* reuse s_o as tmp [hidden] */
        ds4f_chk("sh_out", L, m->s_o, C);
        for (int i = 0; i < C; i++) m->s_moe[i] += m->s_o[i];
        DS4F_TOC(DS4F_P_SHARED); }
        /* ---- MoE: router + top-6 ---- */
        { DS4F_TIC();
        ds4f_matvec(m, m->s_router, &ly->gate, m->s_h2);        /* [n_experts] */
        ds4f_chk("router", L, m->s_router, c->n_experts);
        DS4F_TOC(DS4F_P_ROUTER); }
        int idx[8]; float wt[8];
        if (m->exact)
            ds4f_topk_exact(m->s_router, ly->gate_bias, c->n_experts, c->n_active, idx, wt, c->routed_scale);
        else
            ds4f_topk(m->s_router, c->n_experts, c->n_active, idx, wt, c->routed_scale);
        if (ds4f_dbg) { fprintf(stderr, "  L%-2d topk wt=", L);
            for (int k=0;k<c->n_active;k++) fprintf(stderr,"%.3f(e%d) ", wt[k], idx[k]);
            fprintf(stderr,"\n"); }
        /* ---- MoE: routed experts (owned-only) ---- */
        { DS4F_TIC();
        for (int k = 0; k < c->n_active; k++) {
            int e = idx[k]; if (e < 0) continue;
            if (e % m->ep_size != m->ep_rank) continue;          /* owned-only (Stage 1/local) */
            int slot = e / m->ep_size;                           /* dense owned index */
            ds4f_matvec(m, m->s_exg, &ly->ex_w1[slot], m->s_h2);
            ds4f_matvec(m, m->s_exu, &ly->ex_w3[slot], m->s_h2);
            if (m->exact) { float lim = c->swiglu_limit;
                for (int i = 0; i < c->moe_inter; i++)
                    m->s_exg[i] = ds4f_silu(m->s_exg[i] > lim ? lim : m->s_exg[i]) * ds4f_clampf(m->s_exu[i], -lim, lim);
            } else
                for (int i = 0; i < c->moe_inter; i++) m->s_exg[i] = ds4f_silu(m->s_exg[i]) * m->s_exu[i];
            ds4f_matvec(m, m->s_o, &ly->ex_w2[slot], m->s_exg);
            for (int i = 0; i < C; i++) m->s_route[i] += wt[k] * m->s_o[i];
        }
        DS4F_TOC(DS4F_P_EXPERTS); }
        /* EP combine: sum the routed-expert partial across the expert-parallel
         * group (each rank owns a disjoint expert subset). Shared expert is
         * replicated, so it is NOT reduced — added locally below. */
        if (m->ar_cb) { DS4F_TIC(); m->ar_cb(m->s_route, C, m->ar_ctx); DS4F_TOC(DS4F_P_OTHER); }
        ds4f_chk("moe", L, m->s_route, C);
        if (m->mhc) {
            for (int i = 0; i < C; i++) m->s_o[i] = m->s_moe[i] + m->s_route[i]; /* ffn output f(x) */
            ds4f_hc_post(m, m->s_x4, m->s_resid, m->s_o, post_f, comb_f);         /* expand 1->4 */
        } else for (int i = 0; i < C; i++) x[i] += m->s_moe[i] + m->s_route[i];   /* shared(local)+routed(reduced) */
        ds4f_chk("x+moe", L, m->mhc ? m->s_x4 : x, C);
    }
    /* head: mHC collapse 4 streams -> 1 (no sinkhorn), then out_norm + lm_head */
    float *hsrc = x;
    if (m->mhc) { ds4f_hc_head(m, m->s_x4, m->s_xc); hsrc = m->s_xc; ds4f_chk("hc_head", -1, hsrc, C); }
    { DS4F_TIC();
    ds4f_rmsnorm(m->s_hn, hsrc, m->out_norm, C, eps);
    ds4f_matvec(m, m->s_logits, &m->head, m->s_hn);
    int best = 0; float bv = m->s_logits[0];
    for (int v = 1; v < c->vocab; v++) if (m->s_logits[v] > bv) { bv = m->s_logits[v]; best = v; }
    DS4F_TOC(DS4F_P_HEAD);
    return best; }
}

#endif /* DS4F_H */
