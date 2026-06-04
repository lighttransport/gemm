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
    /* mHC stand-in (F32) */
    float *hc_attn_fn, *hc_ffn_fn;    /* [24,16384] */
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
    /* arena */
    uint8_t *arena; size_t arena_sz, arena_used;
    /* pool */
    ds4f_pool *pool;
    int n_threads, n_cmgs;
    /* scratch (per-forward, single token) */
    float *s_hn, *s_q, *s_qlat, *s_kvlat, *s_attn, *s_oin, *s_o1, *s_o;
    float *s_h2, *s_router, *s_shg, *s_shu, *s_exg, *s_exu, *s_moe, *s_logits;
    float *s_route;         /* routed-expert partial (owned-only); EP-summed via ar_cb */
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
    } else { /* DS4F_MXFP4 split */
        const uint8_t *base = (const uint8_t *)t->w; size_t rb = K / 2;
        const uint8_t *sbase = t->scale; size_t sb = K / 32;
        for (int i = r0; i + 7 < r1; i += 8) {
            const uint8_t *w = base + (size_t)i * rb;
            const uint8_t *s = sbase + (size_t)i * sb;
            matvec_mxfp4_8row(dst + i,
                w, w+rb, w+2*rb, w+3*rb, w+4*rb, w+5*rb, w+6*rb, w+7*rb,
                s, s+sb, s+2*sb, s+3*sb, s+4*sb, s+5*sb, s+6*sb, s+7*sb, x, K);
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
    per_layer += 2*(size_t)24*16384*4 + 2*pad;                                     /* hc_attn_fn, hc_ffn_fn */
    per_layer += (size_t)c->max_pos * c->kv_lora * 4 + pad;                         /* kv cache */

    size_t total = per_layer * c->n_layers;
    total += ds4f_wbytes(DS4F_BF16, c->vocab, c->hidden) + pad;                     /* embed */
    total += ds4f_wbytes(DS4F_BF16, c->vocab, c->hidden) + pad;                     /* head */
    total += (size_t)c->hidden*2 + pad;                                            /* out_norm */
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
    }
}

static void ds4f_fill(ds4f_model *m, ds4f_tensor t) {
    ds4f_fill_task ft; ft.t = t;
    ds4f_pool_run(m->pool, ds4f_fill_worker, &ft);
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
        m->bf16_mv_qt = m->bf16_pv ? DS4F_BF16_PV : DS4F_BF16; }
    {   const char *e = getenv("DS4F_FP8_MAGIC");
        m->fp8_magic = (e && *e && atoi(e)) ? 1 : 0; }
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
        ly->hc_attn_fn = (float *)ds4f_bump(m, (size_t)24*16384*4, 256);
        ly->hc_ffn_fn  = (float *)ds4f_bump(m, (size_t)24*16384*4, 256);
        ly->kv_cache   = (float *)ds4f_bump(m, (size_t)cfg.max_pos*cfg.kv_lora*4, 256);
    }

    /* parallel first-touch fill of all quantized tensors (compute-thread ==
     * touch-thread, so each row block lands on the CMG that later reads it) */
    ds4f_tensor onrm = { m->out_norm, NULL, DS4F_BF16, 1, C }; ds4f_fill(m, onrm);
    ds4f_fill(m, embed); ds4f_fill(m, m->head);
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
typedef struct { ds4f_model *m; ds4f_layer *ly; int pos; float scale; } ds4f_attn_task;

static void ds4f_attn_worker(void *arg, int tid, int nthr) {
    ds4f_attn_task *T = (ds4f_attn_task *)arg;
    ds4f_model *m = T->m; ds4f_layer *ly = T->ly;
    int HD = m->cfg.q_head_dim, KV = m->cfg.kv_lora, nP = T->pos + 1;
    /* split heads across threads */
    int nh = m->cfg.n_heads;
    int per = nh / nthr, extra = nh % nthr;
    int h0 = per*tid + (tid < extra ? tid : extra);
    int h1 = h0 + per + (tid < extra ? 1 : 0);
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

/* Runs MLA + MoE for one token at position `pos`, hidden state in/out `x`[hidden].
 * Returns next-token argmax of synthetic logits (meaningless). */
static int ds4f_forward_token(ds4f_model *m, float *x, int pos) {
    ds4f_config *c = &m->cfg;
    int C = c->hidden, HD = c->q_head_dim, KV = c->kv_lora, H = c->n_heads*HD;
    float eps = 1e-6f;
    if (ds4f_prof_on < 0) { const char *e = getenv("DS4F_PROF"); ds4f_prof_on = e ? atoi(e) : 0; }
    for (int L = 0; L < c->n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        /* ---- MLA: q/kv projections ---- */
        { DS4F_TIC();
        ds4f_rmsnorm(m->s_hn, x, ly->attn_norm, C, eps);
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
        ds4f_attn_task at = { m, ly, pos, 1.0f/sqrtf((float)KV) };
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
        for (int i = 0; i < C; i++) x[i] += m->s_o[i];          /* mHC stand-in: plain residual */
        ds4f_chk("x+attn", L, x, C);
        DS4F_TOC(DS4F_P_OPROJ); }

        /* ---- MoE: shared expert ---- */
        ds4f_rmsnorm(m->s_h2, x, ly->ffn_norm, C, eps);
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
        for (int i = 0; i < C; i++) x[i] += m->s_moe[i] + m->s_route[i];  /* shared(local)+routed(reduced) */
        ds4f_chk("x+moe", L, x, C);
    }
    /* head */
    { DS4F_TIC();
    ds4f_rmsnorm(m->s_hn, x, m->out_norm, C, eps);
    ds4f_matvec(m, m->s_logits, &m->head, m->s_hn);
    int best = 0; float bv = m->s_logits[0];
    for (int v = 1; v < c->vocab; v++) if (m->s_logits[v] > bv) { bv = m->s_logits[v]; best = v; }
    DS4F_TOC(DS4F_P_HEAD);
    return best; }
}

#endif /* DS4F_H */
