/*
 * ds4f_serve_lib.c -- single-node DS4F model serving library.
 *
 * Thin ctypes-friendly wrapper around the exact/tierb2/mHC forward path so a
 * Python persistent runner (ds4f_serve_runner.py) can drive the model without
 * reimplementing the MLA/MoE/compressor math in Python.
 *
 *   open(stage_dir, use_hip, hip_device, threads, cmgs, max_pos)
 *   prefill(ids, n)          -- prompt tokens -> KV + last logits
 *   decode(token, pos)       -- one generation step
 *   sample(sampling)         -- pick the next token from the current logits
 *   logits()                 -- full-vocab logits of the last step
 *   kv_save(path) / kv_restore(path)   -- prefix-cache KV + compressor state
 *   reset()                  -- fresh context at position 0
 *
 * The model math (ds4f_forward_token / the compressor) stays in C; the Python
 * runner owns the request protocol, sampling penalties, and the chat loop.
 *
 * Build:  see the Makefile target `libds4f_serve.so`.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "common/ds4f.h"
#if defined(DS4F_SERVE_HIP)
#include "hetero/ds4f/hip_ds4f_dense.h"
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double temperature, top_p;
    int top_k;
    double presence_penalty, repeat_penalty;
    long long seed;
} ds4f_serve_sampling;

typedef struct ds4f_serve {
    ds4f_model *m;
    ds4f_mem_pool *pool;
    float *x;          /* embedding scratch [hidden] */
    float *logits;     /* last-step full-vocab logits */
    size_t logits_cap;
    int vocab, hidden;
    int pos;           /* next position to decode into */
    uint64_t rng;
    int eos;
    int *hist;         /* recent generated tokens (for the penalties) */
    int n_hist, hist_cap;
    int prefill_tile;  /* logical request tile, clamped to kernel capacity */
    int speculative_tokens; /* greedy-only DSpark proposals; 0 disables */
    float *prefill_x;  /* [prefill_tile, hidden] embedding slab */
    int *prefill_tok;  /* [prefill_tile] argmax scratch */
    int slot_cap;
    unsigned char *slot_used;
    int *slot_pos, *slot_n_hist, *slot_hist_cap;
    uint64_t *slot_rng;
    int **slot_hist;
    float *slot_logits;
#if defined(DS4F_SERVE_HIP)
    hip_ds4f_dense *hip;
    /* Adaptive hot-expert cache: route_hits is cumulative, so a snapshot
     * taken at the last refresh lets us derive a recent-window delta
     * (current - snapshot) without touching the per-token routing hot path.
     * Refresh only ever runs right after ds4f_serve_decode returns, when the
     * token's GPU work is already complete -- eviction frees device memory
     * immediately, so it must never race a kernel still reading it. */
    uint64_t *ac_snapshot;   /* [n_layers*n_experts], NULL until first refresh */
    uint64_t *ac_delta;      /* scratch, same size, reused every refresh */
    int ac_decode_calls;     /* since last refresh */
    int ac_period;           /* DS4F_ADAPTIVE_CACHE_PERIOD tokens; 0 = disabled */
    int ac_cache_mb, ac_reserve_mb;
#endif
} ds4f_serve;

void ds4f_serve_close(ds4f_serve *s);
size_t ds4f_serve_context_bytes(ds4f_serve *s);
int ds4f_serve_context_export(ds4f_serve *s, void *dst, size_t cap);
int ds4f_serve_context_import(ds4f_serve *s, const void *src, size_t len);
int ds4f_serve_sample(ds4f_serve *s, const ds4f_serve_sampling *sp);

int ds4f_serve_enable_route_telemetry(ds4f_serve *s, int enabled) {
    if (!s || !s->m) return -1;
    s->m->route_telemetry = enabled != 0;
    return 0;
}

int ds4f_serve_cache_hot_experts(ds4f_serve *s, int cache_mb,
                                 int reserve_mb, int stats) {
#if defined(DS4F_SERVE_HIP)
    if (!s || !s->m || !s->hip) return -1;
    return hip_ds4f_dense_cache_hot_experts(s->hip, s->m, cache_mb,
                                             reserve_mb, stats);
#else
    (void)s; (void)cache_mb; (void)reserve_mb; (void)stats;
    return -1;
#endif
}

/* Explicit serving knob for the decode-window expert cache.  Keep the
 * environment variables as diagnostics/backward compatibility, but expose
 * production tuning through the runner's program arguments. */
int ds4f_serve_set_adaptive_cache(ds4f_serve *s, int period,
                                   int cache_mb, int reserve_mb) {
#if defined(DS4F_SERVE_HIP)
    if (!s) return -1;
    s->ac_period = period > 0 ? period : 0;
    s->ac_cache_mb = cache_mb;
    s->ac_reserve_mb = reserve_mb > 0 ? reserve_mb : 1536;
    s->ac_decode_calls = 0;
    /* The setter is normally called after prefill.  Snapshot now so the
     * first refresh ranks experts from decode traffic only, rather than
     * accidentally treating the entire prompt as a decode window. */
    if (s->ac_period > 0 && s->m && s->m->route_hits) {
        size_t n = (size_t)s->m->cfg.n_layers * (size_t)s->m->cfg.n_experts;
        if (!s->ac_snapshot) s->ac_snapshot = (uint64_t *)malloc(n * sizeof(uint64_t));
        if (!s->ac_delta) s->ac_delta = (uint64_t *)malloc(n * sizeof(uint64_t));
        if (!s->ac_snapshot || !s->ac_delta) return -1;
        memcpy(s->ac_snapshot, s->m->route_hits, n * sizeof(uint64_t));
    }
    return 0;
#else
    (void)s; (void)period; (void)cache_mb; (void)reserve_mb;
    return -1;
#endif
}

int ds4f_serve_set_attn_hybrid(int enabled) {
#if defined(DS4F_SERVE_HIP)
    ds4f_attn_hybrid_gpu = enabled != 0;
    return 0;
#else
    (void)enabled;
    return -1;
#endif
}

int ds4f_serve_set_attn_cmp_fast(int enabled) {
#if defined(DS4F_SERVE_HIP)
    ds4f_attn_cmp_fast = enabled != 0;
    return 0;
#else
    (void)enabled;
    return -1;
#endif
}

int ds4f_serve_set_logical_ep_lanes(ds4f_serve *s, int lanes) {
    if (!s || !s->m) return -1;
    if (lanes < 0) lanes = 0;
    if (lanes > 32) lanes = 32;
    s->m->logical_ep_lanes = lanes > 1 ? lanes : 0;
    if (s->m->logical_ep_lanes && !s->m->s_lane_route) {
        s->m->s_lane_route = (float *)ds4f_mem_alloc(
            s->m->mem, (size_t)s->m->logical_ep_lanes *
            (size_t)s->m->cfg.hidden * sizeof(float), 256, 1);
        if (!s->m->s_lane_route) return -1;
    }
    return 0;
}

static int env_i(const char *k, int d) { const char *e = getenv(k); return e && *e ? atoi(e) : d; }

/* Acceptance telemetry: without it there is no way to tell a speculation win
 * from a drafting loss. */
static long ds4f_spec_blocks = 0, ds4f_spec_drafted = 0, ds4f_spec_committed = 0;

#if defined(DS4F_SERVE_HIP)
/* Speculative verification calls the exact token kernel directly, so it must
 * share the ordinary decode path's adaptive-cache bookkeeping explicitly. */
static void ds4f_serve_adaptive_cache_tick(ds4f_serve *s, int steps) {
    if (!s || !s->hip || s->ac_period <= 0 || !s->m->route_hits || steps < 1)
        return;
    size_t ncand = (size_t)s->m->cfg.n_layers * (size_t)s->m->cfg.n_experts;
    if (!s->ac_snapshot) {
        s->ac_snapshot = (uint64_t *)calloc(ncand, sizeof(uint64_t));
        s->ac_delta = (uint64_t *)malloc(ncand * sizeof(uint64_t));
    }
    if (!s->ac_snapshot || !s->ac_delta) return;
    s->ac_decode_calls += steps;
    if (s->ac_decode_calls < s->ac_period) return;
    for (size_t i = 0; i < ncand; ++i)
        s->ac_delta[i] = s->m->route_hits[i] - s->ac_snapshot[i];
    hip_ds4f_dense_refresh_hot_experts(s->hip, s->m, s->ac_delta,
        s->ac_cache_mb, s->ac_reserve_mb,
        env_i("DS4F_ADAPTIVE_CACHE_STATS", 0));
    memcpy(s->ac_snapshot, s->m->route_hits, ncand * sizeof(uint64_t));
    s->ac_decode_calls = 0;
}
#endif

typedef struct { float l; int id; } ds4f_pair;
static int cmp_pair(const void *a, const void *b) {
    const ds4f_pair *pa = (const ds4f_pair *)a, *pb = (const ds4f_pair *)b;
    if (pa->l < pb->l) return -1;
    if (pa->l > pb->l) return 1;
    return pa->id - pb->id;
}

static int embed_lookup(const ds4f_model *m, int tok, float *x) {
    if (tok < 0 || tok >= m->cfg.vocab || m->emb_rows != m->cfg.vocab || !m->embed)
        return -1;
    const uint16_t *row = m->embed + (size_t)tok * (size_t)m->cfg.hidden;
    for (int i = 0; i < m->cfg.hidden; i++) {
        uint32_t u = (uint32_t)row[i] << 16;
        memcpy(x + i, &u, sizeof(u));
    }
    return 0;
}

#if defined(DS4F_SERVE_HIP)
/* Bind the dense bank (FP8 ordered, matching the benchmark's decode config) and
 * attach the decode hooks so ds4f_forward_token uses the ROCm for the dense
 * projections + the streamed MXFP4 experts; the routed experts stay CPU. */
static int serve_attach_hip(ds4f_serve *s, int hip_device, int verbose,
                            int ordered_layers) {
    ds4f_model *m = s->m;
    s->hip = hip_ds4f_dense_create_ex(hip_device, verbose, 1);
    if (!s->hip) return -1;
    /* GPU-resident decode KV is deliberately left off in serving until the
     * multi-token CPU/GPU greedy gate passes. It is a fast benchmark option,
     * not a quality-safe default. */
    hip_ds4f_dense_set_decode_features(s->hip, 0);
    for (int L = 0; L < m->cfg.n_layers; ++L) {
        ds4f_layer *z = &m->layers[L];
        z->cmp_wkv_gpu_id = z->cmp_wgate_gpu_id = z->idx_wq_b_gpu_id = -1;
        ds4f_tensor *ts[9] = { &z->wq_a, &z->wq_b, &z->wkv, &z->wo_a, &z->wo_b,
                               &z->sh_w1, &z->sh_w3, &z->sh_w2, &z->gate };
        for (int j = 0; j < 9; ++j) {
            int ordered = L < ordered_layers && ts[j]->type == DS4F_FP8;
            int id;
            if (ts[j]->type == DS4F_BF16)
                id = hip_ds4f_dense_bind_bf16_tensor(s->hip, ts[j]);
            else if (ts[j]->type == DS4F_MXFP4)
                id = hip_ds4f_dense_bind_mxfp4_tensor(s->hip, ts[j]);
            else if (ts[j]->type == DS4F_FP8)
                id = ordered
                    ? hip_ds4f_dense_bind_fp8_ordered_tensor(s->hip, ts[j])
                    : hip_ds4f_dense_bind_tensor(s->hip, ts[j]);
            else {
                /* F32/PV tensors are intentionally left on the exact CPU path. */
                ts[j]->gpu_id = -1;
                continue;
            }
            if (id < 0)
                fprintf(stderr, "ds4f_serve: HIP bind failed layer=%d tensor=%d "
                                "type=%d rows=%d cols=%d w=%p scale=%p\n",
                        L, j, ts[j]->type, ts[j]->rows, ts[j]->cols,
                        ts[j]->w, ts[j]->scale);
            if (id < 0) return -1;
        }
        if (m->tierb2 && m->cfg.compress_ratios[L] && z->cmp_wkv && z->cmp_wgate) {
            int cw = (m->cfg.compress_ratios[L] == 4 ? 2 : 1) * m->cfg.kv_lora;
            ds4f_tensor ck = { z->cmp_wkv, NULL, DS4F_BF16, cw, m->cfg.hidden, -1 };
            ds4f_tensor cg = { z->cmp_wgate, NULL, DS4F_BF16, cw, m->cfg.hidden, -1 };
            if (hip_ds4f_dense_bind_bf16_tensor(s->hip, &ck) >= 0 &&
                hip_ds4f_dense_bind_bf16_tensor(s->hip, &cg) >= 0) {
                z->cmp_wkv_gpu_id = ck.gpu_id;
                z->cmp_wgate_gpu_id = cg.gpu_id;
            }
        }
        /* Indexer q-projection: [index_n_heads*index_head_dim, q_lora] bf16 on
         * the CSA layers only.  Unlike the compressor this was never bound, so
         * both decode and the batched prefill GEMM read it from host DRAM. */
        if (m->tierb2 && m->cfg.compress_ratios[L] == 4 && z->idx_wq_b) {
            ds4f_tensor iq = { z->idx_wq_b, NULL, DS4F_BF16,
                               m->cfg.index_n_heads * m->cfg.index_head_dim,
                               m->cfg.q_lora, -1 };
            if (hip_ds4f_dense_bind_bf16_tensor(s->hip, &iq) >= 0)
                z->idx_wq_b_gpu_id = iq.gpu_id;
        }
    }
    /* DSpark residency is independent of whether speculation is enabled in
     * the scheduler.  Keep an explicit opt-out for reduced-VRAM diagnostics;
     * every unbound tensor has the same exact CPU fallback. */
    if (m->has_mtp && env_i("DS4F_HIP_MTP_DENSE", 1)) {
        for (int st=0;st<m->dspark_n_stages;st++) {
            ds4f_layer *z=&m->dspark[st].layer;
            ds4f_tensor *ts[9]={&z->wq_a,&z->wq_b,&z->wkv,&z->wo_a,&z->wo_b,
                                 &z->sh_w1,&z->sh_w3,&z->sh_w2,&z->gate};
            for (int j=0;j<9;j++) {
                int id=ts[j]->type==DS4F_BF16 ? hip_ds4f_dense_bind_bf16_tensor(s->hip,ts[j]) :
                       ts[j]->type==DS4F_MXFP4 ? hip_ds4f_dense_bind_mxfp4_tensor(s->hip,ts[j]) :
                       hip_ds4f_dense_bind_tensor(s->hip,ts[j]);
                if (id<0) return -1;
            }
        }
        if (hip_ds4f_dense_bind_tensor(s->hip,&m->dspark[0].main_proj)<0) return -1;
        if (hip_ds4f_dense_bind_bf16_tensor(s->hip,&m->dspark[2].markov_w2)<0) return -1;
    }
    if (m->head.type == DS4F_BF16)
        hip_ds4f_dense_bind_bf16_tensor(s->hip, &m->head);
    m->gpu_dense_ctx = s->hip;
    m->gpu_dense_matvec = hip_ds4f_dense_matvec_tensor;
    m->gpu_dense_async_multi = hip_ds4f_dense_matvec_tensors_async;
    m->gpu_dense_wait = hip_ds4f_dense_wait_tensors;
    m->gpu_dense_blockdiag = hip_ds4f_dense_matvec_blockdiag;
    m->gpu_dense_gemm = hip_ds4f_dense_gemm_tensor;
    m->gpu_dense_gemm_multi = hip_ds4f_dense_gemm_tensors;
    m->gpu_dense_mixed = 1;
    /* Expert streaming is selected later by the runner's explicit
     * --hip-expert-stream program argument. */
    return 0;
}

/* Install the same full-batch callbacks used by test_hip_ds4f_real.  Keep
 * this separate from open() so the Python runner can select the production
 * profile with explicit program arguments rather than tuning environment
 * variables. */
int ds4f_serve_configure_hip_prefill(ds4f_serve *s, int enabled,
    int fused_shared_ffn, int prefill_attn, int qkv_fuse,
    int qkv_device_chain, int attn_device_chain, int attn_no_d2h,
    int routed_ffn, int fp8_wmma, int bf16_wmma, int attn_wmma,
    int oproj_group_wmma, int mxfp4_wmma, int expert_stream,
    int block_threads, int expert_pinned_staging, int tb2_batch,
    int decode_routed_ffn, int decode_qkv_fuse, int decode_attn_oproj,
    int decode_kv_resident) {
    if (!s || !s->m || !s->hip) return -1;
    ds4f_model *m = s->m;
    if (!enabled) {
        m->gpu_shared_ffn = NULL;
        m->gpu_shared_ffn_begin = NULL;
        m->gpu_shared_ffn_wait = NULL;
        m->gpu_routed_ffn = NULL;
        m->gpu_oproj = NULL;
        m->gpu_prefill_attn = NULL;
        m->gpu_prefill_attn_oproj = NULL;
        m->gpu_tb2_batch_enabled = 0;
        m->gpu_prefill_qkv = NULL;
        m->gpu_prefill_qkv_enabled = 0;
        m->gpu_qkv_device_chain = 0;
        m->gpu_attn_device_chain = 0;
        m->gpu_attn_no_d2h = 0;
        m->gpu_routed_ffn_enabled = 0;
        m->gpu_dense_layer_prefetch = NULL;
        m->gpu_dense_layer_begin = NULL;
        m->gpu_dense_stream_prefill_only = 0;
        return 0;
    }
    /* mHC/Tier-B2 prefill performs compressor/indexer work and sparse
     * attention on the host. It can use fused QKV, but Q and KV must return to
     * host memory; the device-only QKV->attention chain is for the plain
     * batched prefill path. */
    if (m->mhc || m->tierb2) {
        qkv_device_chain = 0;
        attn_device_chain = 0;
        attn_no_d2h = 0;
    }
    m->gpu_shared_ffn = fused_shared_ffn ? hip_ds4f_dense_shared_ffn : NULL;
    m->gpu_shared_ffn_begin = fused_shared_ffn ? hip_ds4f_dense_shared_ffn_begin : NULL;
    m->gpu_shared_ffn_wait = fused_shared_ffn ? hip_ds4f_dense_shared_ffn_wait : NULL;
    m->gpu_routed_ffn = (routed_ffn || decode_routed_ffn)
        ? hip_ds4f_dense_routed_ffn : NULL;
    m->gpu_oproj = hip_ds4f_dense_oproj;
    m->gpu_prefill_attn = prefill_attn ? hip_ds4f_dense_prefill_attention : NULL;
    m->gpu_prefill_attn_partial = prefill_attn ? hip_ds4f_dense_prefill_attention_partial : NULL;
    m->gpu_tb2_batch_enabled = prefill_attn && tb2_batch;
#if defined(DS4F_SERVE_HIP)
    /* Batched Tier-B2 prefill opts into the exact GPU-window/CPU-compressed
     * split; the implementation itself gates it to large tiles (K>=1024). */
    if (m->gpu_tb2_batch_enabled) ds4f_attn_hybrid_gpu = 1;
#endif
    m->gpu_prefill_qkv = qkv_fuse ? hip_ds4f_dense_prefill_qkv : NULL;
    m->gpu_prefill_qkv_enabled = qkv_fuse != 0;
    m->gpu_qkv_device_chain = qkv_device_chain != 0;
    m->gpu_attn_device_chain = attn_device_chain != 0;
    m->gpu_attn_no_d2h = attn_no_d2h != 0;
    m->gpu_routed_ffn_enabled = routed_ffn != 0;
    /* The same routed callback is valid for the single-token path.  Keeping
     * this disabled forced every decode token through host-side MXFP4 expert
     * execution, even after prompt-hot experts had been admitted to VRAM. */
    m->gpu_decode_routed_ffn_enabled = decode_routed_ffn != 0;
    /* The fused single-upload QKV stage and the grouped o-projection are valid
     * at M=1 too, and both already fall back to the host matvec when a tensor
     * is not device-resident (ds4f_impl.h decode_gpu_qkv / decode_gpu_oproj).
     * Nothing set these before, so the decode path always paid a separate
     * round trip per projection. */
    m->gpu_decode_qkv_enabled = decode_qkv_fuse != 0;
    m->gpu_decode_attn_oproj_enabled = decode_attn_oproj != 0;
    hip_ds4f_dense_set_decode_features(s->hip, decode_kv_resident != 0);
    /* Exact mHC/Tier-B2 routed execution stages only the selected experts in
     * hip_ds4f_dense_routed_ffn.  Whole-layer streaming would upload roughly
     * 3 GB per layer before the router has even selected six experts. */
    m->gpu_dense_layer_prefetch = expert_stream && !m->mhc && !m->tierb2
        ? hip_ds4f_dense_prefetch_layer_raw : NULL;
    m->gpu_dense_layer_begin = expert_stream && !m->mhc && !m->tierb2
        ? hip_ds4f_dense_begin_layer : NULL;
    m->gpu_dense_stream_prefill_only = expert_stream != 0;
    hip_ds4f_dense_set_prefill_features(s->hip, qkv_fuse,
        qkv_device_chain, attn_device_chain, attn_no_d2h, fp8_wmma,
        bf16_wmma, attn_wmma, oproj_group_wmma, mxfp4_wmma, block_threads);
    hip_ds4f_dense_set_expert_pinned_staging(s->hip, expert_pinned_staging);
    m->gpu_prefill_attn_oproj = prefill_attn && !qkv_device_chain
        ? hip_ds4f_dense_prefill_attn_oproj : NULL;
    return 0;
}
#endif

static ds4f_serve *ds4f_serve_open_ep(const char *stage_dir, int use_hip, int hip_device,
                            int threads, int cmgs, long long max_pos,
                            int speculative_tokens,
                            int ep_rank, int ep_size,
                            char *err, size_t err_cap) {
    if (!stage_dir || !*stage_dir) {
        if (err && err_cap) snprintf(err, err_cap, "ds4f_serve_open: no stage dir");
        return NULL;
    }
    ds4f_runtime_options o = ds4f_runtime_options_debug_env(
        ds4f_default_config(), stage_dir, 0, 1, threads > 0 ? threads : 16,
        cmgs > 0 ? cmgs : 1);
    o.ep_rank = ep_rank >= 0 ? ep_rank : 0;
    o.ep_size = ep_size > 0 ? ep_size : 1;
    o.mhc = env_i("DS4F_SERVE_MHC", 1);
    o.tierb2 = env_i("DS4F_SERVE_TIERB2", 1);
    o.exact = 1;                                  /* serve bundle */
    o.mtp = speculative_tokens > 0;
    o.use_hip = use_hip ? 1 : 0; o.hip_device = hip_device;
    if (max_pos > 0) o.cfg.max_pos = (int)max_pos;
    ds4f_model *m = ds4f_load_real_opts(&o);
    if (!m) {
        if (err && err_cap) snprintf(err, err_cap, "ds4f_serve_open: ds4f_load_real_opts failed");
        return NULL;
    }
    /* The checkpoint's hash-routing table always retains its native row
     * stride. The opt-in fast profile may consume only the first N routes;
     * all buffers were allocated at the native width, so shrinking the live
     * count after load is safe and leaves the exact default untouched. */
    m->route_n_active = m->cfg.n_active;
    int expert_active = env_i("DS4F_EXPERT_ACTIVE", m->cfg.n_active);
    if (expert_active < 2) expert_active = 2;
    if (expert_active > m->route_n_active) expert_active = m->route_n_active;
    m->cfg.n_active = expert_active;
    if (m->emb_rows != m->cfg.vocab || m->head.rows != m->cfg.vocab) {
        if (err && err_cap) snprintf(err, err_cap,
            "ds4f_serve_open: single-node serve needs replicated embed/head");
        ds4f_free(m);
        return NULL;
    }
    ds4f_serve *s = (ds4f_serve *)calloc(1, sizeof(*s));
    if (!s) { ds4f_free(m); return NULL; }
    s->m = m;
    s->speculative_tokens = m->has_mtp ? speculative_tokens : 0;
    s->pool = m->mem;
    s->vocab = m->cfg.vocab;
    s->hidden = m->cfg.hidden;
    s->x = (float *)ds4f_mem_alloc(s->pool, (size_t)s->hidden * 4, 256, 0);
    if (!s->x) { ds4f_serve_close(s); return NULL; }
    {
        int requested = env_i("DS4F_SERVE_PREFILL_TILE", 4096);
        if (requested < 1) requested = 1;
        s->prefill_tile = requested < DS4F_MAX_MTILE ? requested : DS4F_MAX_MTILE;
        ds4f_alloc_prefill_batch(m, s->prefill_tile);
        s->prefill_x = (float *)malloc((size_t)s->prefill_tile * (size_t)s->hidden * sizeof(float));
        s->prefill_tok = (int *)malloc((size_t)s->prefill_tile * sizeof(int));
        if (!s->prefill_x || !s->prefill_tok) {
            if (err && err_cap) snprintf(err, err_cap, "ds4f_serve_open: prefill scratch allocation failed");
            ds4f_serve_close(s);
            return NULL;
        }
    }
#if defined(DS4F_SERVE_HIP)
    if (use_hip) {
        if (serve_attach_hip(s, hip_device, env_i("DS4F_HIP_VERBOSE", 0),
                             env_i("DS4F_SERVE_ORDERED_LAYERS", m->cfg.n_layers)) != 0) {
            if (err && err_cap) snprintf(err, err_cap, "ds4f_serve_open: HIP attach failed");
            ds4f_serve_close(s);
            return NULL;
        }
    }
    /* Adaptive hot-expert cache refresh: off by default (0 tokens = disabled)
     * until enabled via program args / DS4F_ADAPTIVE_CACHE_PERIOD, since it
     * only helps once a static prompt-hot admission has already run. */
    s->ac_period = env_i("DS4F_ADAPTIVE_CACHE_PERIOD", 0);
    s->ac_cache_mb = env_i("DS4F_ADAPTIVE_CACHE_MB", -1);
    s->ac_reserve_mb = env_i("DS4F_ADAPTIVE_CACHE_RESERVE_MB", 1536);
#endif
    m->want_full_logits = 1;
    s->eos = 1;
    s->rng = 0x9e3779b97f4a7c15ULL;
    /* Echo the forward-path configuration that actually took effect.  The
     * decode cost depends on these far more than on any tuning flag, and
     * several of them are silently downgraded during load (missing tensors,
     * unsupported dtype combinations), so reporting the requested value is
     * not enough. */
    fprintf(stderr, "[serve] forward: exact=%d mhc=%d tierb2=%d sparse=%d mtp=%d "
            "mxfp4_raw=%d w4a8=%d group_split=%d active=%d/%d int8_kv=%d int8_cmp=%d "
            "max_pos=%d threads=%d spec=%d\n",
            m->exact, m->mhc, m->tierb2, m->sparse, m->has_mtp,
            m->mxfp4_raw, m->mxfp4_w4a8, ds4f_mv_group_split_on(),
            m->cfg.n_active, m->route_n_active,
            m->int8_kv, m->int8_cmp, m->cfg.max_pos,
            threads > 0 ? threads : 16, s->speculative_tokens);
    if (err && err_cap) snprintf(err, err_cap, "ok");
    return s;
}

ds4f_serve *ds4f_serve_open_ex_ep(const char *stage_dir, int use_hip, int hip_device,
                            int threads, int cmgs, long long max_pos,
                            int speculative_tokens, int ep_rank, int ep_size,
                            char *err, size_t err_cap) {
    return ds4f_serve_open_ep(stage_dir, use_hip, hip_device, threads, cmgs,
                              max_pos, speculative_tokens, ep_rank, ep_size,
                              err, err_cap);
}

ds4f_serve *ds4f_serve_open_ex(const char *stage_dir, int use_hip, int hip_device,
                            int threads, int cmgs, long long max_pos,
                            int speculative_tokens,
                            char *err, size_t err_cap) {
    return ds4f_serve_open_ep(stage_dir, use_hip, hip_device, threads, cmgs,
                              max_pos, speculative_tokens, 0, 1, err, err_cap);
}

ds4f_serve *ds4f_serve_open(const char *stage_dir, int use_hip, int hip_device,
                            int threads, int cmgs, long long max_pos,
                            char *err, size_t err_cap) {
    return ds4f_serve_open_ep(stage_dir,use_hip,hip_device,threads,cmgs,max_pos,
                              0, 0, 1, err, err_cap);
}

void ds4f_serve_close(ds4f_serve *s) {
    if (!s) return;
    if (ds4f_spec_blocks)
        fprintf(stderr, "  speculate blocks=%ld drafted=%ld committed=%ld "
                "accepted=%.2f tokens/block (%.1f%% of drafted)\n",
                ds4f_spec_blocks, ds4f_spec_drafted, ds4f_spec_committed,
                (double)ds4f_spec_committed / ds4f_spec_blocks,
                100.0 * ds4f_spec_committed / (double)ds4f_spec_drafted);
    if (ds4f_expbw_on() && ds4f_expbw_s > 0.0)
        fprintf(stderr, "  expert_bw %.3f GB in %.3f s = %.1f GB/s "
                "(%llu CPU experts, %.2f ms per expert)\n",
                ds4f_expbw_bytes / 1e9, ds4f_expbw_s,
                ds4f_expbw_bytes / 1e9 / ds4f_expbw_s,
                (unsigned long long)ds4f_expbw_experts,
                ds4f_expbw_experts ? 1e3 * ds4f_expbw_s / ds4f_expbw_experts : 0.0);
    if (s->m && getenv("DS4F_PROF") && atoi(getenv("DS4F_PROF")) != 0) {
        double acc = 0.0;
        for (int i = 0; i < DS4F_NPHASE; ++i) acc += s->m->prof[i];
        for (int i = 0; i < DS4F_NPHASE; ++i) {
            if (s->m->prof[i] > 1e-3)
                fprintf(stderr, "  %-9s %8.3f s %5.1f%%\n", ds4f_prof_names[i],
                        s->m->prof[i], acc > 0 ? 100.0 * s->m->prof[i] / acc : 0.0);
        }
        fprintf(stderr, "  %-9s %8.3f s (profiled)\n", "TOTAL", acc);
        fprintf(stderr, "  attn_gemm hit=%ld miss=%ld (fast 8-head-blocked path usage)\n",
                ds4f_attn_gemm_hit, ds4f_attn_gemm_miss);
        if (ds4f_attn_gemm_hit) fprintf(stderr,
            "  attn_gemm avg_nP=%.1f avg_nsel=%.1f nh=%ld hd=%ld window=%ld topk=%ld\n",
            (double)ds4f_attn_total_np / ds4f_attn_gemm_hit,
            (double)ds4f_attn_total_nsel / ds4f_attn_gemm_hit,
            ds4f_attn_nh_hd_ws_topk[0], ds4f_attn_nh_hd_ws_topk[1],
            ds4f_attn_nh_hd_ws_topk[2], ds4f_attn_nh_hd_ws_topk[3]);
        extern long ds4f_route_upload_calls, ds4f_route_upload_bytes;
        if (ds4f_route_upload_calls) fprintf(stderr,
            "  route_upload calls=%ld bytes=%.3f GB avg=%.1f KB/call\n",
            ds4f_route_upload_calls, (double)ds4f_route_upload_bytes / 1e9,
            (double)ds4f_route_upload_bytes / 1024.0 / ds4f_route_upload_calls);
        if (ds4f_attn_hybrid_hit || ds4f_attn_hybrid_miss) fprintf(stderr,
            "  attn_hybrid hit=%ld miss=%ld nsel_hit=%ld (GPU-window+CPU-compressed merge usage)\n",
            ds4f_attn_hybrid_hit, ds4f_attn_hybrid_miss, ds4f_attn_hybrid_nsel_hit);
    }
    if (s->m) ds4f_route_report(s->m, stderr);
#if defined(DS4F_SERVE_HIP)
    if (s->hip) hip_ds4f_dense_destroy(s->hip);
    free(s->ac_snapshot); free(s->ac_delta);
#endif
    if (s->slot_hist) for (int i = 0; i < s->slot_cap; ++i) free(s->slot_hist[i]);
    free(s->slot_used); free(s->slot_pos); free(s->slot_n_hist);
    free(s->slot_hist_cap); free(s->slot_rng); free(s->slot_hist);
    free(s->slot_logits);
    if (s->m) ds4f_free_decode_batch(s->m);
    if (s->m) ds4f_free(s->m);
    free(s->hist);
    free(s->logits);
    free(s->prefill_x);
    free(s->prefill_tok);
    free(s);
}

int ds4f_serve_vocab(ds4f_serve *s) { return s ? s->vocab : 0; }
int ds4f_serve_maxpos(ds4f_serve *s) { return s && s->m ? s->m->cfg.max_pos : 0; }
int ds4f_serve_pos(ds4f_serve *s) { return s ? s->pos : 0; }
int ds4f_serve_eos(ds4f_serve *s) { return s ? s->eos : 1; }

/* Prefill `n` prompt tokens starting at absolute position pos0.  The tokens
 * must already lie within max_pos (the caller enforces truncation).  On
 * return, logits() holds the logits for the token AFTER the last prompt
 * token. */
int ds4f_serve_prefill(ds4f_serve *s, const int *ids, int n, int pos0) {
    if (!s || !ids || n < 1) return -1;
    int p = pos0, off = 0, last_tile = 0;
    while (off < n) {
        int tile = n - off;
        if (tile > s->prefill_tile) tile = s->prefill_tile;
        last_tile = tile;
        for (int i = 0; i < tile; ++i) {
            int id = ids[off + i];
            if (id < 0 || id >= s->vocab) return -1;
            if (embed_lookup(s->m, id, s->prefill_x + (size_t)i * s->hidden) != 0) return -1;
        }
        ds4f_set_forward_token_ids(s->m, ids + off, tile);
        ds4f_forward_verify(s->m, s->prefill_x, tile, p,
                            s->prefill_tok, NULL, NULL);
        p += tile;
        off += tile;
    }
    s->pos = p;
    s->n_hist = 0;
    if (s->m->cfg.vocab > (int)s->logits_cap) {
        s->logits = (float *)realloc(s->logits, (size_t)s->m->cfg.vocab * sizeof(float));
        s->logits_cap = (size_t)s->m->cfg.vocab;
    }
    memcpy(s->logits,
           s->m->p_logits + (size_t)(last_tile - 1) * s->vocab,
           (size_t)s->vocab * sizeof(float));
    return 0;
}

/* Decode one token (its id is `token`) at position `pos`.  Returns the argmax
 * of the resulting logits; logits() holds the full distribution. */
static double dc_wall(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int ds4f_serve_decode(ds4f_serve *s, int token, int pos) {
    double tw0 = dc_wall();
    if (!s || token < 0 || token >= s->vocab) return -1;
    /* One-shot: report the attention regime the decode path actually enters.
     * Prefill and decode read the same m->tierb2, but only prefill was
     * observed spending time in the Tier-B2 phases, so record which branch
     * this path takes rather than inferring it from the load-time flag. */
    { static int once = 0;
      if (!once) { once = 1;
          fprintf(stderr, "[serve] decode attn: tierb2=%d sparse=%d ratio[2]=%d "
                  "kv_slots[2]=%d window=%d gpu_attn=%d\n",
                  s->m->tierb2, s->m->sparse, s->m->cfg.compress_ratios[2],
                  s->m->layers[2].kv_slots, s->m->cfg.window_size,
                  s->m->gpu_decode_attn_enabled); } }
    if (embed_lookup(s->m, token, s->x) != 0) return -1;
    ds4f_set_forward_token_ids(s->m, &token, 1);
    int ar = ds4f_forward_token(s->m, s->x, pos);
    double tw1 = dc_wall();
    if (getenv("DS4F_SERVE_TIME") && (pos % 4) == 0)
        fprintf(stderr, "serve decode pos=%d forward=%.1f ms\n", pos, (tw1 - tw0) * 1e3);
    if (ar < 0) return -1;
    s->pos = pos + 1;
    if (s->n_hist >= s->hist_cap) {
        int cap = s->hist_cap ? s->hist_cap * 2 : 64;
        int *p = (int *)realloc(s->hist, (size_t)cap * sizeof(int));
        if (!p) return ar;
        s->hist = p; s->hist_cap = cap;
    }
    s->hist[s->n_hist++] = token;
    if (s->m->cfg.vocab > (int)s->logits_cap) {
        s->logits = (float *)realloc(s->logits, (size_t)s->m->cfg.vocab * sizeof(float));
        s->logits_cap = (size_t)s->m->cfg.vocab;
    }
    memcpy(s->logits, s->m->s_logits, (size_t)s->vocab * sizeof(float));
#if defined(DS4F_SERVE_HIP)
    /* Safe here: the exact token kernel has returned before cache eviction. */
    ds4f_serve_adaptive_cache_tick(s, 1);
#endif
    return ar;
}

/* Greedy DSpark step.  `anchor` is the token selected from the current main
 * logits.  The function returns the number of committed output tokens in
 * out_tokens (anchor plus the accepted draft prefix), advances the exact
 * verifier state by the same amount, and leaves s->logits at the next
 * verifier distribution.  Sampling modes bypass this API in the runner. */
int ds4f_serve_speculate(ds4f_serve *s, int anchor, int pos, int max_tokens,
                         int *out_tokens, float *confidence) {
    if (!s || !s->m || !s->m->has_mtp || s->speculative_tokens<1 ||
        anchor<0 || anchor>=s->vocab || !out_tokens) return -1;
    int K=max_tokens;
    if (K>s->speculative_tokens) K=s->speculative_tokens;
    if (K>s->m->dspark_block_size) K=s->m->dspark_block_size;
    if (K<1 || pos+K>s->m->cfg.max_pos) return -1;
    if (s->n_hist+K>s->hist_cap) {
        int cap=s->hist_cap ? s->hist_cap:64;
        while (cap<s->n_hist+K) cap*=2;
        int *q=(int *)realloc(s->hist,(size_t)cap*sizeof(int));
        if (!q) return -1;
        s->hist=q; s->hist_cap=cap;
    }
    int *draft=(int *)malloc((size_t)K*sizeof(int));
    float *draft_conf=confidence ? (float *)malloc((size_t)K*sizeof(float)):NULL;
    if (!draft || (confidence && !draft_conf)) {
        free(draft); free(draft_conf); return -1;
    }
    if (ds4f_dspark_predict_block(s->m,anchor,pos,K,draft,draft_conf)!=K) {
        free(draft); free(draft_conf); return -1;
    }
    int matched=0,committed=0;
    /* Use the ordinary token kernel as the verifier.  Unlike the batched
     * prefill GEMM it has exactly the same accumulation order as baseline
     * greedy decode. Stop immediately at the first rejected proposal, so no
     * main-model KV or Tier-B2 rollback is required. */
    for (int k=0;k<K;k++) {
        int id=k ? draft[k-1]:anchor;
        if (embed_lookup(s->m,id,s->x)!=0) { free(draft); free(draft_conf); return -1; }
        ds4f_set_forward_token_ids(s->m,&id,1);
        int next=ds4f_forward_token(s->m,s->x,pos+k);
        if (next<0) { free(draft); free(draft_conf); return -1; }
        committed++;
        if (k<K-1 && next==draft[k]) { matched++; continue; }
        break;
    }
    out_tokens[0]=anchor;
    for (int k=0;k<matched;k++) out_tokens[k+1]=draft[k];
    if (confidence) {
        confidence[0]=-1.f;
        for (int k=0;k<matched;k++) confidence[k+1]=draft_conf[k];
    }
    s->pos=pos+committed;
    if (s->m->cfg.vocab>(int)s->logits_cap) {
        s->logits=(float *)realloc(s->logits,(size_t)s->vocab*sizeof(float));
        s->logits_cap=(size_t)s->vocab;
    }
    memcpy(s->logits,s->m->s_logits,(size_t)s->vocab*sizeof(float));
    for (int k=0;k<committed;k++) s->hist[s->n_hist++]=out_tokens[k];
    ds4f_spec_blocks++; ds4f_spec_drafted += K; ds4f_spec_committed += committed;
#if defined(DS4F_SERVE_HIP)
    ds4f_serve_adaptive_cache_tick(s, committed);
#endif
    free(draft); free(draft_conf);
    return committed;
}

/* Decode one token for N independent serialized contexts. The stateful
 * attention rows use separate ds4f_lseq cache sets while all dense/MoE/head
 * projections execute as one M=N forward. Context images are returned in
 * caller-owned buffers and can be scheduled or stashed normally. */
int ds4f_serve_decode_batch(ds4f_serve *s, const int *tokens, int n,
                            const void *const *inputs, const size_t *input_bytes,
                            void *const *outputs, const size_t *output_caps,
                            size_t *output_bytes) {
    if (!s || !s->m || !tokens || !inputs || !input_bytes || !outputs ||
        !output_caps || !output_bytes || n < 2 || n > s->prefill_tile) return -1;
    ds4f_model *m = s->m;
    if (ds4f_alloc_decode_batch(m, n) != 0) return -2;
    int *pos = (int *)malloc((size_t)n * sizeof(int));
    uint64_t *rng = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    int *nh = (int *)malloc((size_t)n * sizeof(int));
    int **hist = (int **)calloc((size_t)n, sizeof(int *));
    int *argmax = (int *)malloc((size_t)n * sizeof(int));
    if (!pos || !rng || !nh || !hist || !argmax) { free(pos); free(rng); free(nh); free(hist); free(argmax); return -1; }
    int rc = -1;
    for (int k = 0; k < n; ++k) {
        for (int L = 0; L < m->cfg.n_layers; ++L)
            ds4f_lseq_apply(&m->layers[L], &m->dec_batch_seq[(size_t)k*m->cfg.n_layers + L]);
        if (ds4f_serve_context_import(s, inputs[k], input_bytes[k]) != 0) goto done;
        pos[k] = s->pos; rng[k] = s->rng; nh[k] = s->n_hist;
        if (nh[k]) {
            hist[k] = (int *)malloc((size_t)nh[k] * sizeof(int));
            if (!hist[k]) goto done;
            memcpy(hist[k], s->hist, (size_t)nh[k] * sizeof(int));
        }
        if (tokens[k] < 0 || tokens[k] >= s->vocab ||
            embed_lookup(m, tokens[k], s->prefill_x + (size_t)k*s->hidden) != 0) goto done;
        m->dec_batch_pos[k] = pos[k];
    }
    for (int L = 0; L < m->cfg.n_layers; ++L)
        ds4f_lseq_apply(&m->layers[L], &m->dec_batch_seq[L]);
    ds4f_set_forward_token_ids(m, tokens, n);
    ds4f_forward_verify(m, s->prefill_x, n, 0, argmax, NULL, NULL);
    for (int k = 0; k < n; ++k) {
        for (int L = 0; L < m->cfg.n_layers; ++L)
            ds4f_lseq_apply(&m->layers[L], &m->dec_batch_seq[(size_t)k*m->cfg.n_layers + L]);
        s->pos = pos[k] + 1; s->rng = rng[k]; s->n_hist = nh[k];
        if (s->hist_cap < nh[k] + 1) {
            int cap = nh[k] + 16;
            int *q = (int *)realloc(s->hist, (size_t)cap * sizeof(int));
            if (!q) goto done;
            s->hist = q; s->hist_cap = cap;
        }
        if (nh[k]) memcpy(s->hist, hist[k], (size_t)nh[k] * sizeof(int));
        s->hist[s->n_hist++] = tokens[k];
        memcpy(s->logits, m->p_logits + (size_t)k*s->vocab,
               (size_t)s->vocab * sizeof(float));
        size_t need = ds4f_serve_context_bytes(s);
        output_bytes[k] = need;
        if (output_caps[k] < need || ds4f_serve_context_export(s, outputs[k], output_caps[k]) != 0)
            goto done;
    }
    rc = 0;
done:
    for (int L = 0; L < m->cfg.n_layers; ++L)
        ds4f_lseq_apply(&m->layers[L], &m->dec_batch_seq[L]);
#if defined(DS4F_SERVE_HIP)
    if (s->hip) hip_ds4f_dense_invalidate_decode_kv(s->hip);
#endif
    for (int k = 0; k < n; ++k) free(hist[k]);
    free(pos); free(rng); free(nh); free(hist); free(argmax);
    return rc;
}

int ds4f_serve_decode_batch_reserve(ds4f_serve *s, int capacity) {
    if (!s || !s->m || capacity < 2 || capacity > s->prefill_tile) return -1;
    /* Cache set zero remains the cooperative runner's prefill/import working
     * context. Persistent decode slots use sets [1, capacity], so admitting a
     * new prompt cannot overwrite an already-active slot. */
    int rc = ds4f_alloc_decode_batch(s->m, capacity + 1);
    if (rc == 0 && !s->slot_cap) {
        s->slot_used = (unsigned char *)calloc((size_t)capacity, 1);
        s->slot_pos = (int *)calloc((size_t)capacity, sizeof(int));
        s->slot_n_hist = (int *)calloc((size_t)capacity, sizeof(int));
        s->slot_hist_cap = (int *)calloc((size_t)capacity, sizeof(int));
        s->slot_rng = (uint64_t *)calloc((size_t)capacity, sizeof(uint64_t));
        s->slot_hist = (int **)calloc((size_t)capacity, sizeof(int *));
        s->slot_logits = (float *)malloc((size_t)capacity * (size_t)s->vocab * sizeof(float));
        if (!s->slot_used || !s->slot_pos || !s->slot_n_hist || !s->slot_hist_cap ||
            !s->slot_rng || !s->slot_hist || !s->slot_logits) rc = -1;
        else s->slot_cap = capacity;
    }
    if (rc == 0) s->m->dec_nseq = 0;
    return rc;
}

static int serve_slot_meta_store(ds4f_serve *s, int slot) {
    if (slot < 0 || slot >= s->slot_cap) return -1;
    if (s->slot_hist_cap[slot] < s->n_hist) {
        int *p = (int *)realloc(s->slot_hist[slot], (size_t)s->n_hist * sizeof(int));
        if (!p && s->n_hist) return -1;
        s->slot_hist[slot] = p; s->slot_hist_cap[slot] = s->n_hist;
    }
    s->slot_pos[slot] = s->pos; s->slot_rng[slot] = s->rng;
    s->slot_n_hist[slot] = s->n_hist;
    if (s->n_hist) memcpy(s->slot_hist[slot], s->hist, (size_t)s->n_hist * sizeof(int));
    memcpy(s->slot_logits + (size_t)slot*s->vocab, s->logits,
           (size_t)s->vocab * sizeof(float));
    s->slot_used[slot] = 1;
    return 0;
}

static int serve_slot_meta_load(ds4f_serve *s, int slot) {
    if (slot < 0 || slot >= s->slot_cap || !s->slot_used[slot]) return -1;
    int n = s->slot_n_hist[slot];
    if (s->hist_cap < n) {
        int *p = (int *)realloc(s->hist, (size_t)n * sizeof(int));
        if (!p && n) return -1;
        s->hist = p; s->hist_cap = n;
    }
    s->pos = s->slot_pos[slot]; s->rng = s->slot_rng[slot]; s->n_hist = n;
    if (n) memcpy(s->hist, s->slot_hist[slot], (size_t)n * sizeof(int));
    memcpy(s->logits, s->slot_logits + (size_t)slot*s->vocab,
           (size_t)s->vocab * sizeof(float));
    return 0;
}

static void serve_slot_apply(ds4f_serve *s, int slot) {
    for (int L = 0; L < s->m->cfg.n_layers; ++L)
        ds4f_lseq_apply(&s->m->layers[L],
            &s->m->dec_batch_seq[(size_t)(slot + 1)*s->m->cfg.n_layers + L]);
}

int ds4f_serve_slot_import(ds4f_serve *s, int slot, const void *src, size_t len) {
    if (!s || slot < 0 || slot >= s->slot_cap) return -1;
    serve_slot_apply(s, slot);
    int rc = ds4f_serve_context_import(s, src, len);
    if (rc == 0) rc = serve_slot_meta_store(s, slot);
    for (int L = 0; L < s->m->cfg.n_layers; ++L)
        ds4f_lseq_apply(&s->m->layers[L], &s->m->dec_batch_seq[L]);
    s->m->dec_nseq = 0;
    return rc;
}

size_t ds4f_serve_slot_context_bytes(ds4f_serve *s, int slot) {
    if (serve_slot_meta_load(s, slot) != 0) return 0;
    return ds4f_serve_context_bytes(s);
}

int ds4f_serve_slot_export(ds4f_serve *s, int slot, void *dst, size_t cap) {
    if (!s || serve_slot_meta_load(s, slot) != 0) return -1;
    serve_slot_apply(s, slot);
    int rc = ds4f_serve_context_export(s, dst, cap);
    for (int L = 0; L < s->m->cfg.n_layers; ++L)
        ds4f_lseq_apply(&s->m->layers[L], &s->m->dec_batch_seq[L]);
    s->m->dec_nseq = 0;
    return rc;
}

int ds4f_serve_slot_sample(ds4f_serve *s, int slot, const ds4f_serve_sampling *sp) {
    if (serve_slot_meta_load(s, slot) != 0) return -1;
    int tok = ds4f_serve_sample(s, sp);
    s->slot_rng[slot] = s->rng;
    return tok;
}

int ds4f_serve_slot_release(ds4f_serve *s, int slot) {
    if (!s || slot < 0 || slot >= s->slot_cap) return -1;
    s->slot_used[slot] = 0; s->slot_n_hist[slot] = 0;
    ds4f_discard_decode_slot(s->m, slot + 1);
    return 0;
}

int ds4f_serve_decode_slots(ds4f_serve *s, const int *slots,
                            const int *tokens, int n) {
    if (!s || !slots || !tokens || n < 1 || n > s->slot_cap) return -1;
    ds4f_model *m = s->m; int NL = m->cfg.n_layers;
    int cache_cap = s->slot_cap + 1;
    ds4f_lseq *saved = (ds4f_lseq *)malloc((size_t)cache_cap*NL*sizeof(*saved));
    if (!saved) return -1;
    memcpy(saved, m->dec_batch_seq, (size_t)cache_cap*NL*sizeof(*saved));
    for (int k = 0; k < n; ++k) {
        int q = slots[k];
        if (q < 0 || q >= s->slot_cap || !s->slot_used[q] ||
            tokens[k] < 0 || tokens[k] >= s->vocab) {
            memcpy(m->dec_batch_seq, saved, (size_t)cache_cap*NL*sizeof(*saved));
            free(saved); return -1;
        }
        memcpy(m->dec_batch_seq + (size_t)k*NL,
               saved + (size_t)(q + 1)*NL, (size_t)NL*sizeof(*saved));
        m->dec_batch_pos[k] = s->slot_pos[q];
        if (embed_lookup(m, tokens[k], s->prefill_x + (size_t)k*s->hidden) != 0) {
            memcpy(m->dec_batch_seq, saved, (size_t)cache_cap*NL*sizeof(*saved)); free(saved); return -1;
        }
    }
    m->dec_nseq = n;
    int *argmax = (int *)malloc((size_t)n*sizeof(int));
    if (!argmax) { memcpy(m->dec_batch_seq, saved, (size_t)cache_cap*NL*sizeof(*saved)); free(saved); return -1; }
    ds4f_set_forward_token_ids(m, tokens, n);
    ds4f_forward_verify(m, s->prefill_x, n, 0, argmax, NULL, NULL);
    for (int k = 0; k < n; ++k) {
        int q = slots[k], nh = s->slot_n_hist[q];
        if (s->slot_hist_cap[q] < nh + 1) {
            int cap = nh + 16; int *p = (int *)realloc(s->slot_hist[q], (size_t)cap*sizeof(int));
            if (!p) { free(argmax); memcpy(m->dec_batch_seq, saved, (size_t)cache_cap*NL*sizeof(*saved)); free(saved); return -1; }
            s->slot_hist[q] = p; s->slot_hist_cap[q] = cap;
        }
        s->slot_hist[q][nh] = tokens[k]; s->slot_n_hist[q] = nh + 1;
        s->slot_pos[q]++;
        memcpy(s->slot_logits + (size_t)q*s->vocab,
               m->p_logits + (size_t)k*s->vocab, (size_t)s->vocab*sizeof(float));
    }
    free(argmax);
    memcpy(m->dec_batch_seq, saved, (size_t)cache_cap*NL*sizeof(*saved)); free(saved);
    for (int L = 0; L < NL; ++L) ds4f_lseq_apply(&m->layers[L], &m->dec_batch_seq[L]);
    m->dec_nseq = 0;
#if defined(DS4F_SERVE_HIP)
    if (s->hip) hip_ds4f_dense_invalidate_decode_kv(s->hip);
#endif
    return 0;
}

const float *ds4f_serve_logits(ds4f_serve *s, int *n) {
    if (!s) { if (n) *n = 0; return NULL; }
    if (n) *n = s->vocab;
    return s->logits;
}

/* Greedy or temperature/top-p/top-k sample over logits(). */
int ds4f_serve_sample(ds4f_serve *s, const ds4f_serve_sampling *sp) {
    double tw0 = dc_wall();
    if (!s || !s->logits || !sp) return -1;
    const float *lg = s->logits;
    int n = s->vocab;
    /* greedy */
    int greedy = 0; float bv = lg[0];
    for (int i = 1; i < n; ++i) if (lg[i] > bv) { bv = lg[i]; greedy = i; }
    double temp = sp->temperature > 0.001 ? sp->temperature : 0.0;
    if (temp <= 0.0) return greedy;
    double top_p = (sp->top_p > 0.0 && sp->top_p <= 1.0) ? sp->top_p : 1.0;
    int top_k = sp->top_k > 0 ? sp->top_k : n;
    double inv = 1.0 / temp;

    /* candidate list (id, logit) over the finite logits.  The penalties
     * (presence/repeat) adjust a copy of the const logits first. */
    float *adj = NULL;
    if (sp->presence_penalty > 0.0 || sp->repeat_penalty > 1.0) {
        adj = (float *)malloc((size_t)n * sizeof(float));
        if (!adj) return greedy;
        memcpy(adj, lg, (size_t)n * sizeof(float));
        for (int h = 0; h < s->n_hist; ++h) {
            int t = s->hist[h];
            if (t < 0 || t >= n) continue;
            if (sp->presence_penalty > 0.0) adj[t] -= (float)sp->presence_penalty;
            if (sp->repeat_penalty > 1.0 && adj[t] > 0.0) adj[t] /= (float)sp->repeat_penalty;
            else if (sp->repeat_penalty > 1.0) adj[t] *= (float)sp->repeat_penalty;
        }
        lg = adj;
    }
    int nc = 0;
    for (int i = 0; i < n; ++i) if (isfinite(lg[i])) nc++;
    if (nc < 1) { free(adj); return greedy; }
    float *sc = (float *)malloc((size_t)nc * 2 * sizeof(float));
    int *ids = (int *)malloc((size_t)nc * sizeof(int));
    if (!sc || !ids) { free(sc); free(ids); free(adj); return greedy; }
    nc = 0;
    for (int i = 0; i < n; ++i) if (isfinite(lg[i])) { sc[nc] = (float)(lg[i] * inv); ids[nc] = i; nc++; }
    if (adj) { free(adj); }

    /* sort by descending logit (qsort, O(n log n) -- the vocab is 129280, so an
     * insertion sort here was ~30 s/step) */
    for (int i = 0; i < nc; ++i) { float t = -sc[i]; sc[i] = t; }
    /* sort (logit,id) pairs ascending by negated logit == descending by logit */
    { ds4f_pair *pr = (ds4f_pair *)malloc((size_t)nc * sizeof(*pr));
      if (!pr) { free(sc); free(ids); return greedy; }
      for (int i = 0; i < nc; ++i) { pr[i].l = sc[i]; pr[i].id = ids[i]; }
      qsort(pr, (size_t)nc, sizeof(*pr), cmp_pair);
      for (int i = 0; i < nc; ++i) { sc[i] = pr[i].l; ids[i] = pr[i].id; }
      free(pr); }
    int keep = nc < top_k ? nc : top_k;
    float mx = sc[0];
    float total = 0.0f;
    for (int i = 0; i < keep; ++i) total += expf(sc[i] - mx);
    if (top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < keep; ++i) {
            cum += expf(sc[i] - mx);
            if (i > 0 && cum / (total + 1e-30f) >= top_p) { keep = i + 1; break; }
        }
        total = 0.0f;
        for (int i = 0; i < keep; ++i) total += expf(sc[i] - mx);
    }
    s->rng = s->rng * 6364136223846793005ULL + 1442695040888963407ULL;
    double draw = ((double)(s->rng >> 11) / (double)(1ULL << 53)) * total;
    int chosen = ids[keep - 1];
    for (int i = 0; i < keep; ++i) {
        draw -= expf(sc[i] - mx);
        if (draw <= 0.0) { chosen = ids[i]; break; }
    }
    free(sc); free(ids);
    if (getenv("DS4F_SERVE_TIME")) {
        double tw1 = dc_wall();
        fprintf(stderr, "serve sample %.1f ms\n", (tw1 - tw0) * 1e3);
    }
    return chosen;
}

int ds4f_serve_reset(ds4f_serve *s) {
    if (!s) return -1;
    s->pos = 0;
    s->n_hist = 0;
#if defined(DS4F_SERVE_HIP)
    if (s->hip) hip_ds4f_dense_invalidate_decode_kv(s->hip);
#endif
    return 0;
}

typedef struct {
    uint64_t magic;
    uint32_t version, header_bytes;
    uint32_t max_pos, n_layers, kv_lora, vocab;
    uint32_t pos, n_hist;
    uint64_t rng, kv_bytes, snap_bytes, logits_bytes;
} ds4f_context_header;

#define DS4F_CONTEXT_MAGIC UINT64_C(0x4453344643545831) /* DS4FCTX1 */

static size_t context_kv_bytes(const ds4f_model *m) {
    size_t n = 0;
    for (int L = 0; L < m->cfg.n_layers; ++L)
        n += (size_t)m->layers[L].kv_slots * (size_t)m->cfg.kv_lora * 2;
    return n;
}

static size_t context_kv_bytes_at(const ds4f_model *m, size_t pos) {
    size_t n = 0;
    for (int L = 0; L < m->cfg.n_layers; ++L) {
        size_t rows = pos < (size_t)m->layers[L].kv_slots
                    ? pos : (size_t)m->layers[L].kv_slots;
        n += rows * (size_t)m->cfg.kv_lora * 2;
    }
    if (m->has_mtp) for (int st=0;st<m->dspark_n_stages;st++) {
        const ds4f_layer *ly=&m->dspark[st].layer;
        size_t rows=pos<(size_t)ly->kv_slots ? pos:(size_t)ly->kv_slots;
        n+=rows*(size_t)m->cfg.kv_lora*2;
    }
    return n;
}

/* Complete, process-local context image used by the cooperative server.  In
 * contrast to the legacy prefix file this also carries logits and sampler
 * state, so a decode can be paused between tokens and resumed bit-exactly. */
size_t ds4f_serve_context_bytes(ds4f_serve *s) {
    if (!s || !s->m) return 0;
    size_t snap = s->m->tierb2 ? ds4f_tb2_snap_bytes(s->m) : 0;
    return sizeof(ds4f_context_header) + context_kv_bytes_at(s->m, (size_t)s->pos) + snap +
           (size_t)s->vocab * sizeof(float) + (size_t)s->n_hist * sizeof(int);
}

int ds4f_serve_context_export(ds4f_serve *s, void *dst, size_t cap) {
    size_t need = ds4f_serve_context_bytes(s);
    if (!s || !s->m || !dst || cap < need) return -1;
    ds4f_model *m = s->m;
    size_t kvb = context_kv_bytes_at(m, (size_t)s->pos);
    size_t snap = m->tierb2 ? ds4f_tb2_snap_bytes(m) : 0;
    ds4f_context_header h = {
        DS4F_CONTEXT_MAGIC, 3, (uint32_t)sizeof(ds4f_context_header),
        (uint32_t)m->cfg.max_pos, (uint32_t)m->cfg.n_layers,
        (uint32_t)m->cfg.kv_lora, (uint32_t)s->vocab,
        (uint32_t)s->pos, (uint32_t)s->n_hist, s->rng, kvb, snap,
        (uint64_t)s->vocab * sizeof(float)
    };
    unsigned char *p = (unsigned char *)dst;
    memcpy(p, &h, sizeof(h)); p += sizeof(h);
    for (int L = 0; L < m->cfg.n_layers; ++L) {
        size_t rows = (size_t)s->pos < (size_t)m->layers[L].kv_slots
                    ? (size_t)s->pos : (size_t)m->layers[L].kv_slots;
        size_t n = rows * (size_t)m->cfg.kv_lora * 2;
        memcpy(p, m->layers[L].kv_cache, n); p += n;
    }
    if (m->has_mtp) for (int st=0;st<m->dspark_n_stages;st++) {
        ds4f_layer *ly=&m->dspark[st].layer;
        size_t rows=(size_t)s->pos<(size_t)ly->kv_slots ? (size_t)s->pos:(size_t)ly->kv_slots;
        size_t n=rows*(size_t)m->cfg.kv_lora*2;
        memcpy(p,ly->kv_cache,n); p+=n;
    }
    if (snap) { ds4f_tb2_snap(m, p, 0); p += snap; }
    memcpy(p, s->logits, (size_t)s->vocab * sizeof(float));
    p += (size_t)s->vocab * sizeof(float);
    if (s->n_hist) memcpy(p, s->hist, (size_t)s->n_hist * sizeof(int));
    return 0;
}

int ds4f_serve_context_import(ds4f_serve *s, const void *src, size_t len) {
    if (!s || !s->m || !src || len < sizeof(ds4f_context_header)) return -1;
    ds4f_context_header h;
    memcpy(&h, src, sizeof(h));
    ds4f_model *m = s->m;
    size_t kvb = h.version == 1 ? context_kv_bytes(m)
                                : (h.version==2 && m->has_mtp ? 0 : context_kv_bytes_at(m,(size_t)h.pos));
    size_t snap = m->tierb2 ? ds4f_tb2_snap_bytes(m) : 0;
    size_t need = sizeof(h) + kvb + snap + (size_t)s->vocab * sizeof(float) +
                  (size_t)h.n_hist * sizeof(int);
    if (h.magic != DS4F_CONTEXT_MAGIC || (h.version != 1 && h.version != 2 && h.version != 3) ||
        (m->has_mtp && h.version<3) ||
        h.header_bytes != sizeof(h) || h.max_pos != (uint32_t)m->cfg.max_pos ||
        h.n_layers != (uint32_t)m->cfg.n_layers ||
        h.kv_lora != (uint32_t)m->cfg.kv_lora || h.vocab != (uint32_t)s->vocab ||
        h.pos > (uint32_t)m->cfg.max_pos || h.kv_bytes != kvb ||
        h.snap_bytes != snap || h.logits_bytes != (uint64_t)s->vocab * sizeof(float) ||
        need != len) return -1;
    if ((int)h.n_hist > s->hist_cap) {
        int *q = (int *)realloc(s->hist, (size_t)h.n_hist * sizeof(int));
        if (!q && h.n_hist) return -1;
        s->hist = q; s->hist_cap = (int)h.n_hist;
    }
    const unsigned char *p = (const unsigned char *)src + sizeof(h);
    for (int L = 0; L < m->cfg.n_layers; ++L) {
        size_t rows = h.version == 1 ? (size_t)m->layers[L].kv_slots
                    : ((size_t)h.pos < (size_t)m->layers[L].kv_slots
                       ? (size_t)h.pos : (size_t)m->layers[L].kv_slots);
        size_t n = rows * (size_t)m->cfg.kv_lora * 2;
        memcpy(m->layers[L].kv_cache, p, n); p += n;
    }
    if (h.version>=3 && m->has_mtp) for (int st=0;st<m->dspark_n_stages;st++) {
        ds4f_layer *ly=&m->dspark[st].layer;
        size_t rows=(size_t)h.pos<(size_t)ly->kv_slots ? (size_t)h.pos:(size_t)ly->kv_slots;
        size_t n=rows*(size_t)m->cfg.kv_lora*2;
        memcpy(ly->kv_cache,p,n); p+=n;
    }
    if (snap) { ds4f_tb2_snap(m, (void *)p, 1); p += snap; }
    memcpy(s->logits, p, (size_t)s->vocab * sizeof(float));
    p += (size_t)s->vocab * sizeof(float);
    if (h.n_hist) memcpy(s->hist, p, (size_t)h.n_hist * sizeof(int));
    s->pos = (int)h.pos; s->n_hist = (int)h.n_hist; s->rng = h.rng;
#if defined(DS4F_SERVE_HIP)
    if (s->hip) hip_ds4f_dense_invalidate_decode_kv(s->hip);
#endif
    return 0;
}

/* Prefix-cache snapshot: copy every layer's kv_cache rows [0,pos) plus the
 * compressor ring state to `path`.  restore() puts them back so a request that
 * extends the same prefix skips re-prefilling.  The file format is the raw
 * kv_cache bytes followed by the tierb2 compressor snapshot. */
int ds4f_serve_kv_save(ds4f_serve *s, const char *path) {
    if (!s || !s->m || !path) return -1;
    ds4f_model *m = s->m;
    size_t pos = (size_t)s->pos;
    size_t snap = 0;
    if (m->tierb2) snap = ds4f_tb2_snap_bytes(m);
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t hdr[3] = { pos, snap, m->has_mtp ? DS4F_DSPARK_STAGES : 0 };
    if (fwrite(hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }
    for (int L = 0; L < m->cfg.n_layers; ++L) {
        ds4f_layer *ly = &m->layers[L];
        size_t rows = (size_t)ly->kv_slots;
        size_t n = rows * (size_t)m->cfg.kv_lora;
        if (fwrite(ly->kv_cache, 2, n, f) != n) { fclose(f); return -1; }
    }
    if (m->has_mtp) for (int st=0;st<m->dspark_n_stages;st++) {
        ds4f_layer *ly=&m->dspark[st].layer;
        size_t n=(size_t)ly->kv_slots*(size_t)m->cfg.kv_lora;
        if (fwrite(ly->kv_cache,2,n,f)!=n) { fclose(f); return -1; }
    }
    if (snap > 0) {
        char *buf = (char *)malloc(snap);
        if (buf) {
            ds4f_tb2_snap(m, buf, 0);
            if (fwrite(buf, 1, snap, f) != snap) { free(buf); fclose(f); return -1; }
        }
        free(buf);
    }
    fclose(f);
    return 0;
}

int ds4f_serve_kv_restore(ds4f_serve *s, const char *path) {
    if (!s || !s->m || !path) return -1;
    ds4f_model *m = s->m;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t hdr[3];
    if (fread(hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }
    size_t pos = hdr[0], snap = hdr[1];
    if (hdr[2]!=(size_t)(m->has_mtp ? DS4F_DSPARK_STAGES:0)) { fclose(f); return -1; }
    for (int L = 0; L < m->cfg.n_layers; ++L) {
        ds4f_layer *ly = &m->layers[L];
        size_t rows = (size_t)ly->kv_slots;
        size_t n = rows * (size_t)m->cfg.kv_lora;
        if (fread(ly->kv_cache, 2, n, f) != n) { fclose(f); return -1; }
    }
    if (m->has_mtp) for (int st=0;st<m->dspark_n_stages;st++) {
        ds4f_layer *ly=&m->dspark[st].layer;
        size_t n=(size_t)ly->kv_slots*(size_t)m->cfg.kv_lora;
        if (fread(ly->kv_cache,2,n,f)!=n) { fclose(f); return -1; }
    }
    if (snap > 0) {
        char *buf = (char *)malloc(snap);
        if (buf && fread(buf, 1, snap, f) == snap) ds4f_tb2_snap(m, buf, 1);
        free(buf);
    }
    fclose(f);
    s->pos = (int)pos;
    return 0;
}
