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
    int compress_ratios[64];  /* per-layer; [0..n_layers), 0 = dense */
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
    c.index_topk = 512; c.index_head_dim = 128;
    /* compress_ratios: layers 0,1 and last = 0 (dense); else alternate CSA(4)/HCA(128) */
    for (int L = 0; L < 64; L++) c.compress_ratios[L] = 0;
    for (int L = 2; L < c.n_layers - 1; L++) c.compress_ratios[L] = (L & 1) ? 4 : 128;
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
    /* EP combine hook: if set, called once per layer on the routed-expert partial
     * [hidden] to sum it across the expert-parallel group (Stage 2 = tp_allreduce_sum).
     * Shared expert stays replicated (added locally). NULL => single-node (all owned). */
    void  (*ar_cb)(float *buf, int count, void *ctx);
    void   *ar_ctx;
    /* perf accounting (weight HBM bytes touched, reset per token by the runner) */
    size_t bytes_read;
    /* per-phase wall-time profiler (seconds, accumulated; printed by runner) */
    double prof[8];
} ds4f_model;

/* phase ids for ds4f_model.prof[] */
enum { DS4F_P_QKV=0, DS4F_P_ATTN=1, DS4F_P_OPROJ=2, DS4F_P_SHARED=3,
       DS4F_P_ROUTER=4, DS4F_P_EXPERTS=5, DS4F_P_HEAD=6, DS4F_P_OTHER=7 };
static const char *ds4f_prof_names[8] = {
    "qkv_proj","attn","o_proj","shared","router","experts","head","other" };

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

/* rmsnorm with BF16 weight, in/out f32 [n] */
static void ds4f_rmsnorm(float *out, const float *x, const uint16_t *w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * ds4f_bf16(w[i]);
}

static inline float ds4f_silu(float x){ return x / (1.0f + expf(-x)); }

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
    return m;
}

static void ds4f_free(ds4f_model *m) {
    if (!m) return;
    ds4f_pool_stop(m->pool);
    if (m->arena && m->arena != MAP_FAILED) munmap(m->arena, m->arena_sz);
    free(m->layers);
    free(m);
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
        ds4f_chk("q", L, m->s_q, H);
        ds4f_matvec(m, m->s_kvlat, &ly->wkv, m->s_hn);          /* [kv_lora] */
        ds4f_rmsnorm(m->s_kvlat, m->s_kvlat, ly->kv_norm, KV, eps);
        ds4f_chk("kvlat", L, m->s_kvlat, KV);
        memcpy(ly->kv_cache + (size_t)pos*KV, m->s_kvlat, (size_t)KV*4);  /* append latent */
        DS4F_TOC(DS4F_P_QKV); }
        /* ---- attention ---- */
        { DS4F_TIC();
        ds4f_attn_task at = { m, ly, pos, 1.0f/sqrtf((float)KV), c->compress_ratios[L] };
        ds4f_pool_run(m->pool, ds4f_attn_worker, &at);          /* fills s_attn[H] */
        ds4f_chk("attn", L, m->s_attn, H);
        DS4F_TOC(DS4F_P_ATTN); }
        /* ---- o projection ---- */
        { DS4F_TIC();
        int og = c->o_groups; (void)H;
        for (int i = 0; i < C; i++) m->s_oin[i] = 0.f;
        for (int g = 0; g < og; g++) for (int i = 0; i < C; i++) m->s_oin[i] += m->s_attn[g*C + i];
        ds4f_matvec(m, m->s_o1, &ly->wo_a, m->s_oin);           /* [o_inter] */
        for (int i = 0; i < c->o_inter; i++) m->s_o1[i] = ds4f_silu(m->s_o1[i]);  /* stand-in nonlin */
        ds4f_matvec(m, m->s_o, &ly->wo_b, m->s_o1);             /* [hidden] */
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
