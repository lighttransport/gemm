/* ds4f_impl.h — implementation fragment of ds4f.h (NOT standalone).
 * Relocated verbatim from ds4f.h to keep the public header small; it is
 * #included at the end of ds4f.h after all public types are defined.
 * Do not #include this directly — include "ds4f.h". */

/* ===================== mapped ownership allocators =====================
 *
 * Model memory is long-lived and has very different lifetimes from request
 * scratch.  A bump pool keeps metadata and scratch together, avoids allocator
 * locks in the hot path, and makes teardown a single set of munmaps.  Large
 * allocations get their own mapping so a temporary prefill tile does not pin
 * unrelated pool pages.  There is deliberately no malloc/free in this path. */
typedef struct ds4f_mem_chunk ds4f_mem_chunk;
struct ds4f_mem_chunk {
    ds4f_mem_chunk *next;
    size_t map_sz, used, cap;
};
struct ds4f_mem_pool {
    ds4f_mem_chunk *head;
    size_t chunk_sz, bytes_reserved, bytes_used, alloc_count;
    pthread_mutex_t lock;
};

static size_t ds4f_page_round(size_t n) {
    long p = sysconf(_SC_PAGESIZE); if (p <= 0) p = 4096;
    size_t page = (size_t)p;
    return (n + page - 1) & ~(page - 1);
}
static size_t ds4f_align_up(size_t n, size_t a) {
    if (a < sizeof(void *)) a = sizeof(void *);
    if ((a & (a - 1)) != 0) { size_t p = 1; while (p < a) p <<= 1; a = p; }
    return (n + a - 1) & ~(a - 1);
}
static void *ds4f_map_alloc(size_t size, size_t align, int zero) {
    if (!size) return NULL;
    size_t h = ds4f_align_up(sizeof(size_t) * 2, align);
    size_t map_sz = ds4f_page_round(h + size + align);
    uint8_t *base = (uint8_t *)mmap(NULL, map_sz, PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) return NULL;
    size_t *meta = (size_t *)base; meta[0] = map_sz; meta[1] = size;
    void *p = base + h;
    if (zero) memset(p, 0, size);
    return p;
}
static void ds4f_map_free(void *ptr) {
    if (!ptr) return;
    size_t h = ds4f_align_up(sizeof(size_t) * 2, 64);
    uint8_t *base = (uint8_t *)ptr - h;
    size_t map_sz = ((size_t *)base)[0];
    if (map_sz) munmap(base, map_sz);
}
static void *ds4f_map_realloc(void *old, size_t old_size, size_t size, size_t align) {
    if (!size) { ds4f_map_free(old); return NULL; }
    void *p = ds4f_map_alloc(size, align, 0);
    if (!p) return NULL;
    if (old && old_size) memcpy(p, old, old_size < size ? old_size : size);
    ds4f_map_free(old);
    return p;
}

static ds4f_mem_pool *ds4f_mem_pool_create(void) {
    ds4f_mem_pool *p = (ds4f_mem_pool *)ds4f_map_alloc(sizeof(*p), 64, 1);
    if (!p) return NULL;
    p->chunk_sz = (size_t)64 * 1024 * 1024;
    pthread_mutex_init(&p->lock, NULL);
    return p;
}
static void *ds4f_mem_alloc(ds4f_mem_pool *p, size_t size, size_t align, int zero) {
    if (!p || !size) return NULL;
    if (align < 64) align = 64;
    pthread_mutex_lock(&p->lock);
    ds4f_mem_chunk *c = p->head;
    size_t off = c ? ds4f_align_up(sizeof(*c) + c->used, align) : 0;
    if (!c || off + size > sizeof(*c) + c->cap) {
        size_t cap = size > p->chunk_sz / 2 ? size + align : p->chunk_sz;
        size_t map_sz = ds4f_page_round(sizeof(*c) + cap + align);
        c = (ds4f_mem_chunk *)mmap(NULL, map_sz, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (c == MAP_FAILED) { pthread_mutex_unlock(&p->lock); return NULL; }
        c->next = p->head; c->map_sz = map_sz;
        c->cap = map_sz - sizeof(*c); c->used = 0; p->head = c;
        p->bytes_reserved += map_sz;
        off = ds4f_align_up(sizeof(*c), align);
    }
    void *out = (uint8_t *)c + off;
    c->used = off + size - sizeof(*c);
    p->bytes_used += size; p->alloc_count++;
    if (zero) memset(out, 0, size);
    pthread_mutex_unlock(&p->lock);
    return out;
}
static void *ds4f_mem_calloc(ds4f_mem_pool *p, size_t n, size_t size, size_t align) {
    if (n && size > SIZE_MAX / n) return NULL;
    return ds4f_mem_alloc(p, n * size, align, 1);
}
static void *ds4f_mem_realloc(ds4f_mem_pool *p, void *old, size_t old_size,
                              size_t size, size_t align) {
    void *out = ds4f_mem_alloc(p, size, align, 0);
    if (out && old && old_size) memcpy(out, old, old_size < size ? old_size : size);
    return out;
}
static void ds4f_mem_pool_destroy(ds4f_mem_pool *p) {
    if (!p) return;
    ds4f_mem_chunk *c = p->head;
    while (c) { ds4f_mem_chunk *next = c->next; munmap(c, c->map_sz); c = next; }
    pthread_mutex_destroy(&p->lock);
    ds4f_map_free(p);
}
static void ds4f_mem_pool_report(const ds4f_mem_pool *p, const char *tag) {
    if (!p || !tag) return;
    fprintf(stderr, "[ds4f/mem] %s reserved=%.3f GiB used=%.3f GiB allocs=%zu\n",
            tag, (double)p->bytes_reserved / (1024.0 * 1024.0 * 1024.0),
            (double)p->bytes_used / (1024.0 * 1024.0 * 1024.0), p->alloc_count);
}

/* DS4F_FLAGBAR: per-worker completion flag on its own cache line (kills the 47-way
 * fetch_add contention of the centralized `done` counter; main spins on distinct lines). */
typedef struct { _Atomic int v; char pad[64 - sizeof(_Atomic int)]; } ds4f_cacheline;
struct ds4f_pool {
    int nthr, n_cmgs;
    pthread_t *threads;
    _Atomic int seq;
    _Atomic int done;            /* counter barrier (flagbar=0) */
    _Atomic int stop;
    ds4f_fn fn;
    void *arg;
    ds4f_cacheline *donef;       /* per-worker flags (flagbar=1): donef[tid].v = last seq run */
    int flagbar;                 /* DS4F_FLAGBAR, resolved once at pool_start */
    ds4f_mem_pool *mem;           /* non-NULL for model-owned pools */
    int own_mem;                  /* compatibility pools created by ds4f_pool_start */
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
#elif (defined(__x86_64__) || defined(__i386__)) && defined(__linux__)
/* One thread per PHYSICAL core while the pool fits: on Zen1 the SMT siblings
 * share the load/store unit, so they add no gather bandwidth (measured: 32
 * threads 42.7 GB/s vs 47.6 at 16 -- hetero/ds4f/README.md). CPU ids alternate
 * between the two SMT threads of a core, hence the *2. */
static int ds4f_pin(int tid, int nthr, int n_cmgs) {
    (void)n_cmgs;
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu <= 0) return -1;
    int cpu = (nthr <= ncpu / 2) ? (tid * 2) : tid;
    if (cpu >= ncpu) cpu = tid % ncpu;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}
#else
static int ds4f_pin(int t, int n, int c){ (void)t;(void)n;(void)c; return -1; }
#endif

/* Spin-wait relax hint for the pool's busy-wait barriers. */
#if defined(__aarch64__) || defined(__arm__)
static inline void ds4f_relax(void){ __asm__ __volatile__("yield" ::: "memory"); }
#elif defined(__x86_64__) || defined(__i386__)
static inline void ds4f_relax(void){ __asm__ __volatile__("pause" ::: "memory"); }
#else
static inline void ds4f_relax(void){ __asm__ __volatile__("" ::: "memory"); }
#endif

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
        if (p->flagbar)
            atomic_store_explicit(&p->donef[tid].v, last, memory_order_release);  /* own cache line */
        else
            atomic_fetch_add_explicit(&p->done, 1, memory_order_release);
    }
    return NULL;
}

static ds4f_pool *ds4f_pool_start_ex(ds4f_mem_pool *mem, int nthr, int n_cmgs) {
    if (!mem) return NULL;
    ds4f_pool *p = (ds4f_pool *)ds4f_mem_calloc(mem, 1, sizeof(*p), 64);
    if (!p) return NULL;
    p->nthr = nthr; p->n_cmgs = n_cmgs;
    p->mem = mem;
    atomic_store(&p->seq, 0); atomic_store(&p->done, 0); atomic_store(&p->stop, 0);
    { const char *e = getenv("DS4F_FLAGBAR"); p->flagbar = e ? atoi(e) : 0; }
    p->donef = (ds4f_cacheline *)ds4f_mem_alloc(mem,
                                                 (size_t)nthr * sizeof(ds4f_cacheline), 64, 1);
    for (int t = 0; t < nthr; t++) atomic_store_explicit(&p->donef[t].v, 0, memory_order_relaxed);
    p->threads = (pthread_t *)ds4f_mem_calloc(mem, (size_t)nthr, sizeof(pthread_t), 64);
    for (int t = 1; t < nthr; t++) {   /* main thread acts as tid 0 */
        ds4f_wctx *w = (ds4f_wctx *)ds4f_mem_alloc(mem, sizeof(*w), 64, 0);
        w->p = p; w->tid = t;
        pthread_create(&p->threads[t], NULL, ds4f_worker, w);
    }
    ds4f_pin(0, nthr, n_cmgs);          /* pin main to core 12 */
    return p;
}
static ds4f_pool *ds4f_pool_start(int nthr, int n_cmgs) {
    ds4f_mem_pool *mem = ds4f_mem_pool_create();
    ds4f_pool *p = ds4f_pool_start_ex(mem, nthr, n_cmgs);
    if (p) p->own_mem = 1;
    else ds4f_mem_pool_destroy(mem);
    return p;
}

static void ds4f_pool_run(ds4f_pool *p, ds4f_fn fn, void *arg) {
    p->fn = fn; p->arg = arg;
    if (p->flagbar) {
        /* seq is single-producer (main only) -> load+1+store is race-free. Workers write
         * donef[tid]=this seq when done; main waits each distinct line -> no shared-counter
         * contention. Bit-identical to the counter path (same fn, same per-worker work). */
        int s = atomic_load_explicit(&p->seq, memory_order_relaxed) + 1;
        atomic_store_explicit(&p->seq, s, memory_order_release);
        fn(arg, 0, p->nthr);            /* main = tid 0 */
        for (int t = 1; t < p->nthr; t++)
            while (atomic_load_explicit(&p->donef[t].v, memory_order_acquire) != s)
                ds4f_relax();
    } else {
        atomic_store_explicit(&p->done, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&p->seq, 1, memory_order_release);
        fn(arg, 0, p->nthr);            /* main = tid 0 */
        while (atomic_load_explicit(&p->done, memory_order_acquire) < p->nthr - 1)
            ds4f_relax();
    }
}

static void ds4f_pool_stop(ds4f_pool *p) {
    if (!p) return;
    ds4f_mem_pool *owned = p->own_mem ? p->mem : NULL;
    atomic_store_explicit(&p->stop, 1, memory_order_release);
    for (int t = 1; t < p->nthr; t++) pthread_join(p->threads[t], NULL);
    if (owned) ds4f_mem_pool_destroy(owned);
}

/* row split into 8-aligned blocks for worker tid of nthr */
static inline void ds4f_rowsplit8(int rows, int nthr, int tid, int *r0, int *r1) {
    int blk = (rows + 7) / 8;                 /* number of 8-row groups */
    int per = blk / nthr, extra = blk % nthr;
    int g0 = per * tid + (tid < extra ? tid : extra);
    int g1 = g0 + per + (tid < extra ? 1 : 0);
    *r0 = g0 * 8; *r1 = g1 * 8; if (*r1 > rows) *r1 = rows;
}

/* ---- tensor-parallel (TP): shard a dense tensor's ROWS across the EP nodes ----
 * Like ds4f_rowsplit8 but for `n` nodes with a configurable alignment (8 for bf16/bf16-pv,
 * 128 for FP8 whose scale is 128-row-blocked). Node r owns aligned rows [*r0,*r1). */
static inline void ds4f_tp_rowshard(int rows, int n, int r, int align, int *r0, int *r1) {
    int blk = (rows + align - 1) / align;
    int per = blk / n, extra = blk % n;
    int g0 = per * r + (r < extra ? r : extra);
    int g1 = g0 + per + (r < extra ? 1 : 0);
    *r0 = g0 * align; *r1 = g1 * align; if (*r1 > rows) *r1 = rows;
}
/* DS4F_TP_HEAD: vocab-shard the lm_head across the EP group (8-aligned for bf16/bf16-pv).
 * Sets r0/rows to this rank's shard; returns 1 when sharded, 0 (replicated, full vocab). */
static inline int ds4f_tp_head_shard(int vocab, int ep_rank, int ep_size, int *r0, int *rows, int enabled) {
    if (!enabled || ep_size <= 1) { *r0 = 0; *rows = vocab; return 0; }
    int a0, a1; ds4f_tp_rowshard(vocab, ep_size, ep_rank, 8, &a0, &a1);
    *r0 = a0; *rows = a1 - a0; return 1;
}
/* DS4F_TP_SHARED: col-shard the shared-expert up/gate (sh_w1/sh_w3) over shared_inter
 * (align = 128 for FP8 dst, else 8). The default keeps sh_w2 replicated. The optional
 * DS4F_TP_SHARED_FULL profile also row-shards sh_w2 over hidden; its output is a
 * disjoint partial and is folded into the existing routed-expert ar_cb reduce. */
static inline int ds4f_tp_shared_shard(int shared_inter, int ep_rank, int ep_size, int align, int *r0, int *rows, int enabled) {
    if (!enabled || ep_size <= 1) { *r0 = 0; *rows = shared_inter; return 0; }
    int a0, a1; ds4f_tp_rowshard(shared_inter, ep_size, ep_rank, align, &a0, &a1);
    *r0 = a0; *rows = a1 - a0; return 1;
}
static inline int ds4f_tp_shared_out_shard(int hidden, int ep_rank, int ep_size, int align, int *r0, int *rows, int enabled) {
    if (!enabled || ep_size <= 1) { *r0 = 0; *rows = hidden; return 0; }
    ds4f_tp_rowshard(hidden, ep_size, ep_rank, align, r0, rows);
    return 1;
}
/* DS4F_TP_ATTN: shard the n_heads across the EP group (heads are independent -> align 1).
 * wq_b gets the owned heads' rows; q-norm/attn workers process [h0,h1); the o-proj output is a
 * per-node partial reduced via ar_cb. Returns 1 + [h0,h1), else 0 + [0,n_heads). */
static inline int ds4f_tp_attn_shard(int n_heads, int ep_rank, int ep_size, int *h0, int *h1, int enabled) {
    if (!enabled || ep_size <= 1) { *h0 = 0; *h1 = n_heads; return 0; }
    ds4f_tp_rowshard(n_heads, ep_size, ep_rank, 1, h0, h1); return 1;
}
/* DS4F_TP_OPROJ: row-shard wo_a by o_inter (align 128 for FP8). The block-diagonal kernel
 * picks each row's group via (goff+i)/o_lora, so the shard need NOT align to groups. wo_b stays
 * replicated (it contracts the full o_inter, reconstructed by summing the partial s_o). The
 * o_inter rows need the FULL s_attn -> reduce s_attn first when TP_ATTN. Returns 1 + [r0,rows). */
static inline int ds4f_tp_oproj_shard(int o_inter, int ep_rank, int ep_size, int align, int *r0, int *rows, int enabled) {
    if (!enabled || ep_size <= 1) { *r0 = 0; *rows = o_inter; return 0; }
    int a0, a1; ds4f_tp_rowshard(o_inter, ep_size, ep_rank, align, &a0, &a1);
    *r0 = a0; *rows = a1 - a0; return 1;
}
/* DS4F_TP_EMBED: vocab-shard the input embedding (bf16, 8-aligned). embed_lookup fills only the
 * owner's row + zeros, then ar_cb-SUMs -> the full embedding BIT-EXACT. Returns 1 + [r0,rows). */
static inline int ds4f_tp_embed_shard(int vocab, int ep_rank, int ep_size, int *r0, int *rows, int enabled) {
    if (!enabled || ep_size <= 1) { *r0 = 0; *rows = vocab; return 0; }
    int a0, a1; ds4f_tp_rowshard(vocab, ep_size, ep_rank, 8, &a0, &a1);
    *r0 = a0; *rows = a1 - a0; return 1;
}
/* DS4F_CP (Phase 2 context parallelism): shard the compressed-cache slots [0,S) by slot range across
 * the EP nodes -- node r owns [s0,s1); S = max_pos/ratio (per layer). The per-node compressed caches
 * (cmp_q4/idx_kv8_4) shrink to the owned slots (ceiling -> node-count x ctx); the indexer scans local
 * slots -> local top-k, merged cross-node to the global top-k, and the selected latents are gathered
 * (Stage C). 8-aligned (i4 nibble pairs / svdot blocks). Off => [0,S). Composes with the int4 caches. */
static inline int ds4f_cp_slot_shard(int S, int ep_rank, int ep_size, int *s0, int *s1, int enabled) {
    if (!enabled || ep_size <= 1) { *s0 = 0; *s1 = S; return 0; }
    int a0, a1; ds4f_tp_rowshard(S, ep_size, ep_rank, 8, &a0, &a1);
    /* ds4f_tp_rowshard clamps only r1 to `rows`, so a high rank whose 8-aligned block START exceeds S
     * (small tail, S < n*align) comes back with a0 > S >= a1 -- an INVALID a1<a0 range. The dense shards
     * never hit this (rows >> n*align); the small CP tail does. Clamp to a valid (possibly empty) range,
     * else nslot_q = CAL+(a1-a0) < CAL and the replicated [0,CAL) freeze write overruns the alloc. */
    if (a0 > S) a0 = S;
    if (a1 < a0) a1 = a0;
    *s0 = a0; *s1 = a1; return 1;
}
/* split this node's owned heads [m->attn_h0, m->attn_h1) across the thread pool (worker tid/nthr).
 * When TP_ATTN is off the range is [0, n_heads) so this matches the old per=nh/nthr split. */
static inline void ds4f_head_split(const ds4f_model *m, int nthr, int tid, int *h0, int *h1) {
    int lo = m->attn_h0, n = m->attn_h1 - m->attn_h0;
    int per = n / nthr, extra = n % nthr;
    *h0 = lo + per * tid + (tid < extra ? tid : extra);
    *h1 = *h0 + per + (tid < extra ? 1 : 0);
}

/* ===================== helpers ===================== */
static inline float ds4f_bf16(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
static inline uint16_t ds4f_f32_bf16(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }

/* Keep the expert activation mode per model so an exact-f32 and W4A8 model
 * can coexist in one correctness process. */
static inline int ds4f_mxfp4_w4a8_env(void) {
    const char *e = getenv("DS4F_MXFP4_W4A8");
    return e ? atoi(e) : 1;
}
static inline int ds4f_mxfp4_w4a8_on(const ds4f_model *m) {
    return m ? m->mxfp4_w4a8 : ds4f_mxfp4_w4a8_env();
}

/* ---- W8A8 (DS4F_Q8_PV) activation quantization ----
 * Quantize K activations into int8 + per-64-block fp16 scale, the input form
 * matvec_sdot_8row consumes. Deterministic (absmax + round-to-nearest via svcvt,
 * same on every rank => lockstep preserved). K must be a multiple of 64. Mirrors
 * transformer.h's tf_quant_x_sdot_blocks (the proven sibling implementation). */
#if defined(__ARM_FEATURE_SVE)
static inline void ds4f_quant_x_sdot_into(const float *x, int K, int8_t *xq, float *xs) {
    svbool_t pg = svptrue_b32();
    svint32_t qlo = svdup_s32(-127), qhi = svdup_s32(127);
    int nb = K / 64;
    for (int b = 0; b < nb; b++) {
        const float *xb = x + (size_t)b * 64;
        svfloat32_t v0 = svld1_f32(pg, xb + 0),  v1 = svld1_f32(pg, xb + 16);
        svfloat32_t v2 = svld1_f32(pg, xb + 32), v3 = svld1_f32(pg, xb + 48);
        svfloat32_t mx = svmax_x(pg, svmax_x(pg, svabs_x(pg, v0), svabs_x(pg, v1)),
                                     svmax_x(pg, svabs_x(pg, v2), svabs_x(pg, v3)));
        float amax = svmaxv_f32(pg, mx);
        float scale = amax / 127.0f, inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        xs[b] = scale;   /* WS6: fp32 scale. fp16 overflowed to +Inf for amax > 127*65504 (~8.3e6)
                          * -> dequant sc=ws*Inf=Inf -> 0*Inf=NaN in zero-dot lanes. int8 xq unchanged
                          * (inv is fp32) so this is a pure scale-precision/overflow fix. */
        svfloat32_t vinv = svdup_f32(inv);
        int8_t *q = xq + (size_t)b * 64;
        #define DS4F_QX(V, OFF) do {                                          \
            svint32_t qi = svcvt_s32_f32_x(pg, svmul_x(pg, (V), vinv));        \
            qi = svmax_s32_x(pg, svmin_s32_x(pg, qi, qhi), qlo);              \
            svst1b_s32(pg, q + (OFF), qi);                                    \
        } while (0)
        DS4F_QX(v0, 0); DS4F_QX(v1, 16); DS4F_QX(v2, 32); DS4F_QX(v3, 48);
        #undef DS4F_QX
    }
}
#endif /* __ARM_FEATURE_SVE */
/* thread-local scratch for M-token activation quant (grow-on-demand) */
static __thread int8_t   *ds4f_xq_buf = NULL;
static __thread float    *ds4f_xs_buf = NULL;   /* WS6: fp32 activation scale (was fp16 -> overflow NaN) */
static __thread size_t    ds4f_xq_cap = 0;   /* in (M*K) int8 elems */
static inline void ds4f_q8_xscratch(int M, int K, int8_t **xq, float **xs) {
    size_t need = (size_t)M * K;
    if (need > ds4f_xq_cap) {
        ds4f_map_free(ds4f_xq_buf); ds4f_map_free(ds4f_xs_buf);
        ds4f_xq_buf = (int8_t *)ds4f_map_alloc(need, 64, 0);
        ds4f_xs_buf = (float *)ds4f_map_alloc((size_t)M * (K / 64) * sizeof(float), 64, 0);
        ds4f_xq_cap = need;
    }
    *xq = ds4f_xq_buf; *xs = ds4f_xs_buf;
}

/* Per-thread [M,8] accumulation scratch is shared by the SVE and AVX2 BF16
 * GEMM workers.  Keep it outside either ISA guard. */
static __thread float *ds4f_acc_buf = NULL;
static __thread size_t ds4f_acc_cap = 0;
static inline float (*ds4f_acc_scratch(size_t M))[8] {
    size_t need = M * 8;
    if (need > ds4f_acc_cap) {
        ds4f_map_free(ds4f_acc_buf);
        ds4f_acc_buf = (float *)ds4f_map_alloc(need * sizeof(float), 64, 0);
        ds4f_acc_cap = ds4f_acc_buf ? need : 0;
    }
    return (float (*)[8])ds4f_acc_buf;
}

#if !defined(__ARM_FEATURE_SVE) && defined(__AVX2__) && defined(__FMA__)
/* Activation scratch for the AVX2 MXFP4 W4A8 expert path: int8 values plus the
 * per-32-block scale and debias term. Quantized ONCE per matvec (in
 * ds4f_mv_worker) and shared by every row, exactly as the Q8_PV path does. */
static __thread int8_t *ds4f_mxq_buf = NULL;
static __thread float  *ds4f_mxs_buf = NULL;
static __thread float  *ds4f_mxc_buf = NULL;
static __thread size_t  ds4f_mxq_cap = 0;
static inline void ds4f_mxfp4_xscratch(int K, int8_t **xq, float **xs, float **xc) {
    if ((size_t)K > ds4f_mxq_cap) {
        ds4f_map_free(ds4f_mxq_buf); ds4f_map_free(ds4f_mxs_buf); ds4f_map_free(ds4f_mxc_buf);
        ds4f_mxq_buf = (int8_t *)ds4f_map_alloc((size_t)K, 64, 0);
        ds4f_mxs_buf = (float *)ds4f_map_alloc((size_t)(K / 32) * sizeof(float), 64, 0);
        ds4f_mxc_buf = (float *)ds4f_map_alloc((size_t)(K / 32) * sizeof(float), 64, 0);
        ds4f_mxq_cap = (size_t)K;
    }
    *xq = ds4f_mxq_buf; *xs = ds4f_mxs_buf; *xc = ds4f_mxc_buf;
}

/* Same scratch for a prefill tile: n tokens quantized up front so the row
 * loop can hoist outside the token loop and reuse each weight decode.
 * xq is [n, K], xs and xc are [n, K/32]. *//* Per-thread [M,8] float accumulation scratch for the BF16-PV GEMM worker.
 * Sized by the actual batch M instead of DS4F_MAX_MTILE: at MTILE=8192 the
 * old stack array was 256 KB and spilled, changing the reduction order (and
 * the argmax) at small batches. */
static __thread int8_t *ds4f_mxqn_buf = NULL;
static __thread float  *ds4f_mxsn_buf = NULL;
static __thread float  *ds4f_mxcn_buf = NULL;
static __thread size_t  ds4f_mxqn_cap = 0;
static inline int ds4f_mxfp4_xscratch_n(int K, int n, int8_t **xq,
                                        float **xs, float **xc) {
    size_t need = (size_t)K * (size_t)n;
    if (need > ds4f_mxqn_cap) {
        ds4f_map_free(ds4f_mxqn_buf); ds4f_map_free(ds4f_mxsn_buf);
        ds4f_map_free(ds4f_mxcn_buf);
        ds4f_mxqn_buf = (int8_t *)ds4f_map_alloc(need, 64, 0);
        ds4f_mxsn_buf = (float *)ds4f_map_alloc((need / 32) * sizeof(float), 64, 0);
        ds4f_mxcn_buf = (float *)ds4f_map_alloc((need / 32) * sizeof(float), 64, 0);
        if (!ds4f_mxqn_buf || !ds4f_mxsn_buf || !ds4f_mxcn_buf) {
            ds4f_mxqn_cap = 0;
            return -1;
        }
        ds4f_mxqn_cap = need;
    }
    *xq = ds4f_mxqn_buf; *xs = ds4f_mxsn_buf; *xc = ds4f_mxcn_buf;
    return 0;
}

/* Permuted f32 activation scratch, for the exact (non-W4A8) raw-layout kernel. */
static __thread float *ds4f_mxp_buf = NULL;
static __thread size_t ds4f_mxp_cap = 0;
static inline float *ds4f_mxfp4_xperm(int K) {
    if ((size_t)K > ds4f_mxp_cap) {
        ds4f_map_free(ds4f_mxp_buf);
        ds4f_mxp_buf = (float *)ds4f_map_alloc((size_t)K * sizeof(float), 64, 0);
        ds4f_mxp_cap = (size_t)K;
    }
    return ds4f_mxp_buf;
}

/* Permuted f32 activation scratch for a whole prefill tile: [n, K]. */
static __thread float *ds4f_mxpn_buf = NULL;
static __thread size_t ds4f_mxpn_cap = 0;
static inline float *ds4f_mxfp4_xperm_n(int K, int n) {
    size_t need = (size_t)K * (size_t)n;
    if (need > ds4f_mxpn_cap) {
        ds4f_map_free(ds4f_mxpn_buf);
        ds4f_mxpn_buf = (float *)ds4f_map_alloc(need * sizeof(float), 64, 0);
        ds4f_mxpn_cap = ds4f_mxpn_buf ? need : 0;
    }
    return ds4f_mxpn_buf;
}

#endif

/* ---- repack a DS4F_BF16_PV dense tensor in place to DS4F_Q8_PV ----
 * Reads the pair-interleaved bf16 source (element (row,j) at
 * base[(row/8)*8*K + (row&7>>1)*2*K + 2*j + (row&1)]), writes the q8_pv group
 * layout (8 fp16 per-(row,64-block) scales + 8x64 int8). Per-64-block absmax
 * keeps quantization fine-grained (better argmax fidelity than per-row). */
typedef struct { const uint16_t *src; uint8_t *q8; int N, K; } ds4f_q8repack_task;
static void ds4f_q8repack_worker(void *arg, int tid, int nthr) {
    ds4f_q8repack_task *T = (ds4f_q8repack_task *)arg;
    int N = T->N, K = T->K, nb = K / 64, groups = N / 8;
    int per = groups / nthr, extra = groups % nthr;
    int g0 = per * tid + (tid < extra ? tid : extra);
    int g1 = g0 + per + (tid < extra ? 1 : 0);
    const uint16_t *base = T->src;
    for (int g = g0; g < g1; g++) {
        const uint16_t *gsrc = base + (size_t)g * 8 * K;       /* group's pair-bufs */
        uint8_t *gbuf = T->q8 + (size_t)g * nb * 528;
        for (int b = 0; b < nb; b++) {
            uint8_t *blk = gbuf + (size_t)b * 528;
            uint16_t *scl = (uint16_t *)blk;
            int8_t   *qs  = (int8_t *)(blk + 16);
            for (int r = 0; r < 8; r++) {
                int pair = r >> 1, slot = r & 1;
                const uint16_t *src = gsrc + (size_t)pair * 2 * K + (size_t)2 * (b * 64) + slot;
                float amax = 0.0f;
                for (int j = 0; j < 64; j++) {
                    float f = ds4f_bf16(src[2 * j]); float a = f < 0 ? -f : f;
                    if (a > amax) amax = a;
                }
                float invs = amax > 0 ? 127.0f / amax : 0.0f;
                scl[r] = ggml_fp32_to_fp16(amax / 127.0f);
                for (int j = 0; j < 64; j++) {
                    int q = (int)lrintf(ds4f_bf16(src[2 * j]) * invs);
                    if (q < -127) q = -127; else if (q > 127) q = 127;
                    qs[r * 64 + j] = (int8_t)q;
                }
            }
        }
    }
}

/* Repack a DS4F_BF16_PV dense tensor in place -> DS4F_Q8_PV (int8 W8A8). No-op
 * (returns 0) unless bf16-pv with rows%8==0 and cols%64==0. Allocates a fresh
 * q8 group buffer (mmap), fills it in parallel (SAME group split as the matvec
 * rowsplit8, so each q8 group is first-touched by the CMG that later reads it),
 * then swaps t->w/t->type. The model's bf16 source is arena-owned (MAP_PRIVATE|
 * ANON); once q8 exists it is dead, so we MADV_DONTNEED its page-aligned interior
 * to reclaim the ~2 B/elem (q8 is ~1.03 B/elem -> net dense memory SHRINKS, no
 * OOM risk on the 11-node budget). Aligned inward so an adjacent arena tensor's
 * pages are never dropped. m==NULL or reclaim==0 keeps the source (test path). */
static int ds4f_repack_bf16pv_to_q8pv_ex(ds4f_model *m, ds4f_tensor *t, int reclaim) {
    if (t->type != DS4F_BF16_PV) return 0;
    int N = t->rows, K = t->cols;
    if ((N & 7) || (K & 63)) return 0;
    size_t bytes = ds4f_wbytes(DS4F_Q8_PV, N, K);
    void *q8 = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (q8 == MAP_FAILED) return 0;
#ifdef MADV_NOHUGEPAGE
    madvise(q8, bytes, MADV_NOHUGEPAGE);
#endif
    void *old = t->w; size_t oldb = ds4f_wbytes(DS4F_BF16_PV, N, K);
    ds4f_q8repack_task T = { (const uint16_t *)t->w, (uint8_t *)q8, N, K };
    ds4f_pool_run(m->pool, ds4f_q8repack_worker, &T);
    t->w = q8;
    t->type = DS4F_Q8_PV;
    if (reclaim) {
        long pg = sysconf(_SC_PAGESIZE); if (pg <= 0) pg = 4096;
        uintptr_t a = ((uintptr_t)old + (uintptr_t)(pg - 1)) & ~(uintptr_t)(pg - 1);
        uintptr_t b = ((uintptr_t)old + oldb) & ~(uintptr_t)(pg - 1);
        if (b > a) madvise((void *)a, (size_t)(b - a), MADV_DONTNEED);
    }
    return 1;
}
/* default driver: no reclaim (test/standalone — source not arena-owned). */
static int ds4f_repack_bf16pv_to_q8pv(ds4f_model *m, ds4f_tensor *t) {
    return ds4f_repack_bf16pv_to_q8pv_ex(m, t, 0);
}

/* DS4F_Q8_DENSE hook: after the dense weights hold their final bf16-pv values
 * (synth fill OR real load), repack the dominant dense tensors to int8 W8A8 so
 * the prefill GEMM uses svdot. Reclaims each bf16 source (net memory shrinks).
 * Router gate + lm-head are intentionally LEFT bf16-pv (argmax protection).
 * No-op unless m->q8_dense && the source is DS4F_BF16_PV. Lockstep-safe: the
 * int8 quantization is deterministic (absmax + round-to-nearest) so every rank
 * produces bit-identical q8 from bit-identical bf16. */
static void ds4f_q8_promote_dense(ds4f_model *m) {
    if (!m->q8_dense) return;
    if (m->dense_qt != DS4F_BF16_PV) {
        fprintf(stderr, "[ds4f] DS4F_Q8_DENSE=1 ignored: dense is not bf16-pv "
                "(need DS4F_FP8_BF16=1; current dense_qt=%d)\n", m->dense_qt);
        m->q8_dense = 0; return;
    }
    int nrep = 0;
    for (int L = 0; L < m->cfg.n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        ds4f_tensor *dt[8] = { &ly->wq_a, &ly->wq_b, &ly->wkv, &ly->wo_a,
                               &ly->wo_b, &ly->sh_w1, &ly->sh_w3, &ly->sh_w2 };
        for (int i = 0; i < 8; i++) nrep += ds4f_repack_bf16pv_to_q8pv_ex(m, dt[i], 1);
    }
    if (m->ep_rank == 0)
        fprintf(stderr, "[ds4f] DS4F_Q8_DENSE: repacked %d dense tensors -> int8 W8A8 "
                "(gate+head stay bf16-pv)\n", nrep);
}

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
    } else if (t->type == DS4F_Q8_PV) {
        /* W8A8 int8 svdot. Quantize x once (thread-local), then one sdot group
         * per 8-row block (weight group L1-resident; ~1.03 B/elem from HBM). */
        const uint8_t *base = (const uint8_t *)t->w;
        size_t gb = (size_t)(K / 64) * 528;
        int8_t *xq; float *xs; ds4f_q8_xscratch(1, K, &xq, &xs);
        ds4f_quant_x_sdot_into(x, K, xq, xs);
        for (int i = r0; i + 7 < r1; i += 8)
            matvec_sdot_8row(dst + i, base + (size_t)(i / 8) * gb, xq, xs, K);
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
#if !defined(__ARM_FEATURE_SVE) && defined(__AVX2__) && defined(__FMA__)
        /* x86: walk rows sequentially rather than eight at a time. The 8-row
         * grouping exists to amortize an f32 activation on A64FX; on Zen1 it
         * instead gives each thread eight concurrent memory streams and costs
         * ~25% of achievable bandwidth (hetero/ds4f/README.md). */
        int raw = T->m->mxfp4_raw;   /* weights are the ON-DISK bytes, not repacked */
        if (ds4f_mxfp4_w4a8_on(T->m)) {
            int8_t *xq; float *xs, *xc;
            ds4f_mxfp4_xscratch(K, &xq, &xs, &xc);
            if (raw) {
                ds4f_mxfp4_quant_act_raw(x, K, xq, xs, xc);  /* once per matvec */
                for (int i = r0; i < r1; i++)
                    matvec_mxfp4_1row_i8_raw(dst + i, base + (size_t)i * rb,
                                             sbase + (size_t)i * sb, xq, xs, xc, K);
            } else {
                ds4f_mxfp4_quant_act(x, K, xq, xs, xc);
                for (int i = r0; i < r1; i++)
                    matvec_mxfp4_1row_i8(dst + i, base + (size_t)i * rb,
                                         sbase + (size_t)i * sb, xq, xs, xc, K);
            }
        } else if (raw) {
            float *xp = ds4f_mxfp4_xperm(K);
            ds4f_mxfp4_perm_act_f32(x, K, xp);
            for (int i = r0; i < r1; i++)
                matvec_mxfp4_1row_f32_raw(dst + i, base + (size_t)i * rb,
                                          sbase + (size_t)i * sb, xp, K);
        } else {
            for (int i = r0; i < r1; i++)
                matvec_mxfp4_1row(dst + i, base + (size_t)i * rb,
                                  sbase + (size_t)i * sb, x, K);
        }
#else
        for (int i = r0; i + 7 < r1; i += 8) {
            const uint8_t *w = base + (size_t)i * rb;
            const uint8_t *s = sbase + (size_t)i * sb;
            matvec_mxfp4_8row(dst + i,
                w, w+rb, w+2*rb, w+3*rb, w+4*rb, w+5*rb, w+6*rb, w+7*rb,
                s, s+sb, s+2*sb, s+3*sb, s+4*sb, s+5*sb, s+6*sb, s+7*sb, x, K);
        }
#endif
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
    if (m->gpu_dense_matvec && t->gpu_id >= 0) {
        m->bytes_read += ds4f_wbytes(t->type, t->rows, t->cols)
                       + ds4f_sbytes(t->type, t->rows, t->cols);
        if (m->gpu_dense_matvec(m->gpu_dense_ctx, dst, t, x) != 0) {
            fprintf(stderr, "ds4f: GPU dense matvec failed (id=%d, %dx%d)\n",
                    t->gpu_id, t->rows, t->cols);
            abort();
        }
        return;
    }
    ds4f_mv_task T = { m, dst, t, x };
    m->bytes_read += ds4f_wbytes(t->type, t->rows, t->cols) + ds4f_sbytes(t->type, t->rows, t->cols);
    ds4f_pool_run(m->pool, ds4f_mv_worker, &T);
}

/* DS4F_MV_FUSE (WS2): run several INDEPENDENT matvecs in ONE pool_run so the thread
 * wake + barrier is paid once instead of N times (decode does ~10 pool_run/layer; in-loop
 * matvec BW is ~247 GB/s vs ~610 GB/s for the same matvec in isolation -> the gap is
 * dispatch / serial-gap overhead, not the kernel). Each sub-matvec reuses ds4f_mv_worker
 * with the IDENTICAL rowsplit8 + kernel + per-row dot order as a standalone ds4f_matvec
 * => BIT-EXACT; only the barrier is shared. The CALLER must guarantee the list entries are
 * mutually independent (no entry's x aliases another entry's dst). */
static int ds4f_mv_fuse = -1;
static inline int ds4f_mv_fuse_on(void) {
    if (ds4f_mv_fuse < 0) { const char *e = getenv("DS4F_MV_FUSE"); ds4f_mv_fuse = e ? atoi(e) : 0; }
    return ds4f_mv_fuse;
}

/* DS4F_EXPERT_BATCH: evaluate all locally-owned active experts in two pool
 * dispatches per layer, matching bench_expert_bw's gateup/down stages.  The
 * default is on for the real routed path; setting it to 0 retains the old
 * per-expert sequence for an A/B correctness/performance comparison. */
static int ds4f_expert_batch = -1;
static inline int ds4f_expert_batch_on(void) {
    if (ds4f_expert_batch < 0) {
        const char *e = getenv("DS4F_EXPERT_BATCH");
        ds4f_expert_batch = e ? atoi(e) : 1;
    }
    return ds4f_expert_batch;
}
typedef struct { float *dst; const ds4f_tensor *t; const float *x; } ds4f_mv1;
typedef struct { ds4f_model *m; const ds4f_mv1 *list; int n; } ds4f_mv_multi_task;
static void ds4f_mv_multi_worker(void *arg, int tid, int nthr) {
    ds4f_mv_multi_task *T = (ds4f_mv_multi_task *)arg;
    for (int s = 0; s < T->n; s++) {
        ds4f_mv_task sub = { T->m, T->list[s].dst, (const ds4f_tensor *)T->list[s].t, T->list[s].x };
        ds4f_mv_worker(&sub, tid, nthr);
    }
}
/* Must not exceed the dense adapter's async slot count. */
#define DS4F_MV_ASYNC_MAX 8
/* DS4F_MV_ASYNC: put a fused GPU matvec group in flight on separate streams
 * and download once, instead of one launch + event sync + blocking download
 * per tensor.  Each tensor is still computed by the same kernel independently,
 * so results are bit-identical; set to 0 to A/B the dispatch change alone. */
static int ds4f_mv_async = -1;
static inline int ds4f_mv_async_on(void) {
    if (ds4f_mv_async < 0) { const char *e = getenv("DS4F_MV_ASYNC"); ds4f_mv_async = e ? atoi(e) : 1; }
    return ds4f_mv_async;
}
static void ds4f_matvec_multi(ds4f_model *m, const ds4f_mv1 *list, int n) {
    if (m->gpu_dense_matvec) {
        int all_gpu = n > 0;
        for (int s = 0; s < n; s++) {
            const ds4f_tensor *t = list[s].t;
            if ((t->type != DS4F_FP8 && t->type != DS4F_MXFP4) ||
                t->gpu_id < 0) { all_gpu = 0; break; }
        }
        if (all_gpu) {
            for (int s = 0; s < n; s++) {
                const ds4f_tensor *t = list[s].t;
                m->bytes_read += ds4f_wbytes(t->type, t->rows, t->cols)
                               + ds4f_sbytes(t->type, t->rows, t->cols);
            }
            /* Put the whole group in flight on separate streams and download
             * once, instead of one launch + event sync + blocking download per
             * tensor.  The adapter refuses (and we fall back) if it is already
             * holding an async batch, e.g. the overlapped shared expert. */
            if (n > 1 && n <= DS4F_MV_ASYNC_MAX && ds4f_mv_async_on() &&
                m->gpu_dense_async_multi && m->gpu_dense_wait) {
                float *dsts[DS4F_MV_ASYNC_MAX];
                const ds4f_tensor *ts[DS4F_MV_ASYNC_MAX];
                const float *xs[DS4F_MV_ASYNC_MAX];
                for (int s = 0; s < n; s++) {
                    dsts[s] = list[s].dst; ts[s] = list[s].t; xs[s] = list[s].x;
                }
                if (m->gpu_dense_async_multi(m->gpu_dense_ctx, dsts, ts, xs, n) == 0) {
                    if (m->gpu_dense_wait(m->gpu_dense_ctx) != 0) {
                        fprintf(stderr, "ds4f: GPU dense async group wait failed\n");
                        abort();
                    }
                    return;
                }
            }
            for (int s = 0; s < n; s++) {
                const ds4f_tensor *t = list[s].t;
                if (m->gpu_dense_matvec(m->gpu_dense_ctx, list[s].dst, t, list[s].x) != 0) {
                    fprintf(stderr, "ds4f: GPU dense fused matvec failed (id=%d, %dx%d)\n",
                            t->gpu_id, t->rows, t->cols);
                    abort();
                }
            }
            return;
        }
    }
    for (int s = 0; s < n; s++) {
        const ds4f_tensor *t = list[s].t;
        m->bytes_read += ds4f_wbytes(t->type, t->rows, t->cols) + ds4f_sbytes(t->type, t->rows, t->cols);
    }
    ds4f_mv_multi_task T = { m, list, n };
    ds4f_pool_run(m->pool, ds4f_mv_multi_worker, &T);
}

/* ---- block-diagonal matvec: the grouped o-proj wo_a [o_groups*glora rows, cols].
 * Output row-block i (8-aligned; glora%8==0 so a block never straddles a group) uses
 * input xbase + (i/glora)*gin. Fuses the o_groups separate ds4f_matvec dispatches into
 * ONE pool_run -> kills (o_groups-1) cross-CMG barriers AND the load imbalance (glora=1024
 * rows split across 48 threads = ~2.6 8-row groups/thread, poorly balanced; fused = all
 * o_inter=8192 rows split evenly). BIT-EXACT to the per-group ds4f_matvec loop: same kernel,
 * same per-row dot, same accumulation order (each row fully owned by one thread); using the
 * FULL tensor + GLOBAL row index i reproduces ds4f_row_slice's byte/scale offsets exactly
 * (PV: (i/8)*8*K; FP8/MXFP4 scale: i-indexed). Only the row->thread mapping changes. */
typedef struct {
    ds4f_model *m; float *dst; const ds4f_tensor *t; const float *xbase;
    int gin, glora, goff;   /* goff = global o_inter row offset of this (TP-sharded) wo_a -> group=(goff+i)/glora */
} ds4f_mv_bd_task;
static void ds4f_mv_bd_worker(void *arg, int tid, int nthr) {
    ds4f_mv_bd_task *T = (ds4f_mv_bd_task *)arg;
    const ds4f_tensor *t = T->t; float *dst = T->dst;
    int K = t->cols, gin = T->gin, glora = T->glora, goff = T->goff;
    int r0, r1; ds4f_rowsplit8(t->rows, nthr, tid, &r0, &r1);
    if (t->type == DS4F_BF16_PV) {
        const uint16_t *base = (const uint16_t *)t->w;
        for (int i = r0; i + 7 < r1; i += 8) {
            const float *x = T->xbase + (size_t)((goff + i) / glora) * gin;
            const uint16_t *g = base + (size_t)(i / 8) * 8 * K;
            matvec_bf16_8row_pv(dst + i, g, g + 2*K, g + 4*K, g + 6*K, x, K);
        }
    } else if (t->type == DS4F_BF16) {
        const uint16_t *base = (const uint16_t *)t->w;
        for (int i = r0; i + 7 < r1; i += 8) {
            const float *x = T->xbase + (size_t)((goff + i) / glora) * gin;
            const uint16_t *w = base + (size_t)i * K;
            matvec_bf16_8row(dst + i, w, w+K, w+2*K, w+3*K, w+4*K, w+5*K, w+6*K, w+7*K, x, K);
        }
    } else if (t->type == DS4F_Q8_PV) {
        /* int8 W8A8 (DS4F_Q8_DENSE). Like the regular Q8_PV matvec, but the block-
         * diagonal x differs per group => re-quantize x only when the group changes
         * (groups are glora-aligned and glora%8==0, so a quantize lands on a block
         * boundary). Bit-exact to the per-group ds4f_matvec Q8 path (same xq/xs, same
         * (i/8)*gb weight offset). */
        const uint8_t *base = (const uint8_t *)t->w;
        size_t gb = (size_t)(K / 64) * 528;
        int8_t *xq; float *xs; ds4f_q8_xscratch(1, K, &xq, &xs);
        int cur_g = -1;
        for (int i = r0; i + 7 < r1; i += 8) {
            int g = (goff + i) / glora;
            if (g != cur_g) { ds4f_quant_x_sdot_into(T->xbase + (size_t)g * gin, K, xq, xs); cur_g = g; }
            matvec_sdot_8row(dst + i, base + (size_t)(i / 8) * gb, xq, xs, K);
        }
    } else if (t->type == DS4F_FP8) {
        const uint8_t *base = (const uint8_t *)t->w;
        int sb_cols = (K + 127) / 128;
        if (T->m->fp8_magic) {
            ds4f_set_ftz();
            for (int i = r0; i + 7 < r1; i += 8) {
                const float *x = T->xbase + (size_t)((goff + i) / glora) * gin;
                const uint8_t *w = base + (size_t)i * K;
                const uint8_t *es = t->scale + (size_t)(i / 128) * sb_cols;
                matvec_fp8e4m3_8row_magic(dst + i, w, w+K, w+2*K, w+3*K, w+4*K, w+5*K, w+6*K, w+7*K,
                                          es, x, K);
            }
        } else {
            for (int i = r0; i + 7 < r1; i += 8) {
                const float *x = T->xbase + (size_t)((goff + i) / glora) * gin;
                const uint8_t *w = base + (size_t)i * K;
                const uint8_t *es = t->scale + (size_t)(i / 128) * sb_cols;
                matvec_fp8e4m3_8row(dst + i, w, w+K, w+2*K, w+3*K, w+4*K, w+5*K, w+6*K, w+7*K,
                                    es, T->m->fp8_lut, x, K);
            }
        }
    } else if (t->type == DS4F_MXFP4) {
        const uint8_t *base = (const uint8_t *)t->w; size_t rb = K / 2;
        const uint8_t *sbase = t->scale; size_t sb = K / 32;
        for (int i = r0; i + 7 < r1; i += 8) {
            const float *x = T->xbase + (size_t)((goff + i) / glora) * gin;
            const uint8_t *w = base + (size_t)i * rb;
            const uint8_t *s = sbase + (size_t)i * sb;
            matvec_mxfp4_8row(dst + i,
                w, w+rb, w+2*rb, w+3*rb, w+4*rb, w+5*rb, w+6*rb, w+7*rb,
                s, s+sb, s+2*sb, s+3*sb, s+4*sb, s+5*sb, s+6*sb, s+7*sb, x, K);
        }
    }
}
static int ds4f_oproj_fuse = -1;     /* DS4F_OPROJ_FUSE: 1=fused wo_a (default), 0=per-group ref */
static void ds4f_matvec_blockdiag(ds4f_model *m, float *dst, const ds4f_tensor *t,
                                  const float *xbase, int gin, int glora, int goff) {
    if (m->gpu_dense_blockdiag && t->type == DS4F_FP8 && t->gpu_id >= 0) {
        m->bytes_read += ds4f_wbytes(t->type, t->rows, t->cols)
                       + ds4f_sbytes(t->type, t->rows, t->cols);
        if (m->gpu_dense_blockdiag(m->gpu_dense_ctx, dst, t, xbase,
                                   gin, glora, goff) != 0) {
            fprintf(stderr, "ds4f: GPU block-diagonal dense matvec failed (id=%d, %dx%d)\n",
                    t->gpu_id, t->rows, t->cols);
            abort();
        }
        return;
    }
    ds4f_mv_bd_task T = { m, dst, t, xbase, gin, glora, goff };
    m->bytes_read += ds4f_wbytes(t->type, t->rows, t->cols) + ds4f_sbytes(t->type, t->rows, t->cols);
    ds4f_pool_run(m->pool, ds4f_mv_bd_worker, &T);
}

/* ===================== batched (M>1) GEMM for prefill =====================
 * Y[t*Ystride + r] = sum_k W[r,k] * X[t*Xstride + k], for t in [0,M), r in [0,rows).
 * Token-major output. The bf16 path keeps each weight tile L1-resident across all
 * M tokens, so weights are read from HBM ~ONCE per GEMM instead of once per token
 * -> the M=1 BW-bound matvec becomes a compute-bound GEMM (the prefill speedup).
 * bf16 dense (DS4F_FP8_BF16=1) takes the true GEMM directly; FP8 dense (the
 * default, memory-lean) takes a FUSED FP8->bf16 tile-dequant GEMM: each 8-row
 * group's TILE_K FP8 sub-tile is dequanted into an 8 KB L1 PV pair-buffer (never
 * written to HBM) and consumed across all M tokens by the same peak bf16 8x3 pv
 * kernel -> compute-bound prefill at ~88% of the +6 GB resident-bf16 speed WITHOUT
 * the +6 GB, bit-identical dequant to DS4F_FP8_BF16=1 (same LUT + 128-block scale +
 * bf16 truncation). MXFP4 experts have their own group GEMM. Only DS4F_F32 (tiny
 * mHC mixes) falls back to a per-token matvec loop. K-tile reassociation makes the
 * result bit-SIMILAR (~1e-4) to the single-token matvec. */
#ifndef DS4F_MAX_MTILE
#define DS4F_MAX_MTILE 4096         /* large routed-expert buckets amortize weight decode; activation scratch grows on demand */
#endif
typedef struct { ds4f_model *m; float *Y; const ds4f_tensor *t;
                 const float *X; int M, Ystride, Xstride;
                 const int8_t *mx_xq; const float *mx_xs, *mx_xc, *mx_xp;
               } ds4f_gemm_task;
/* thread-local L1 scratch for the FUSED FP8->bf16 prefill GEMM (grow-on-demand).
 * Holds 4 PV pair-buffers (4 x 2*TILE_K halfwords = 8 KB for one 8-row group's
 * current K-tile); the FP8 weight is dequanted into it tile-by-tile and consumed
 * by the bf16 8x3 pv kernel without ever writing bf16 to HBM. EXACT vs the
 * DS4F_FP8_BF16=1 resident path (identical LUT + 128-block E8M0 scale + bf16
 * truncation, whose dropped f32 bits are zero) -- but no resident bf16 copy, no
 * +6 GB. */
static __thread uint16_t *ds4f_fp8t_buf = NULL;
static __thread size_t    ds4f_fp8t_cap = 0;          /* in uint16_t elems */
static inline uint16_t *ds4f_fp8bf16_tile(size_t need) {
    if (need > ds4f_fp8t_cap) {
        ds4f_map_free(ds4f_fp8t_buf);
        ds4f_fp8t_buf = (uint16_t *)ds4f_map_alloc(need * sizeof(uint16_t), 64, 0);
        ds4f_fp8t_cap = need;
    }
    return ds4f_fp8t_buf;
}
/* NOTE: a k-panel cache-blocking variant (k-tile OUTER, row-block INNER over a
 * panel of 8-row blocks) was tried to keep the activation tile L1/L2-resident
 * across the panel. It was NEUTRAL when TILE_K stayed 512 (bit-identical) and
 * CATASTROPHIC when TILE_K shrank to fit X in L1 (M=64 66->40, M=128 56->22
 * tok/s) because the per-tile accumulator read-modify-write traffic scales with
 * K/TILE_K. Diagnosis: this dense GEMM is weight-streaming/compute-bound (reads
 * N*K weight halfwords + FMAs), not activation-locality-bound -- the activation
 * re-reads were already L2-cheap. No loop transform over the same weights helps;
 * sub-f32 compute (e.g. int8 svdot) is the only remaining lever. Path reverted. */
#if defined(__ARM_FEATURE_SVE)
static void ds4f_gemm_worker(void *arg, int tid, int nthr) {
    ds4f_gemm_task *T = (ds4f_gemm_task *)arg;
    const ds4f_tensor *t = T->t; const float *X = T->X; float *Y = T->Y;
    int K = t->cols, M = T->M, Ys = T->Ystride, Xs = T->Xstride;
    int r0, r1; ds4f_rowsplit8(t->rows, nthr, tid, &r0, &r1);
    if (t->type == DS4F_Q8_PV) {
        /* W8A8 int8 svdot prefill. Quantize all M tokens once (thread-local),
         * then row-block OUTER, token INNER: each 8-row weight group (nb*528 B)
         * is read from HBM once and reused L1-resident across the M tokens
         * (matches the bf16 GEMM's once-per-group HBM traffic). 8x3 register-
         * blocked sdot keeps the FLA pipes fed; 1-2 token remainder single. */
        const uint8_t *base = (const uint8_t *)t->w;
        size_t gb = (size_t)(K / 64) * 528;
        int8_t *xq; float *xs; ds4f_q8_xscratch(M, K, &xq, &xs);   /* per-thread TLS scratch */
        int nbq = K / 64;
        for (int mm = 0; mm < M; mm++)
            ds4f_quant_x_sdot_into(X + (size_t)mm * Xs, K,
                                   xq + (size_t)mm * K, xs + (size_t)mm * nbq);
        for (int i = r0; i + 7 < r1; i += 8) {
            const uint8_t *g = base + (size_t)(i / 8) * gb;
            int mm = 0;
            for (; mm + 2 < M; mm += 3)
                matvec_sdot_8row_3x(Y + (size_t)mm*Ys + i, Y + (size_t)(mm+1)*Ys + i,
                                    Y + (size_t)(mm+2)*Ys + i, g,
                                    xq + (size_t)mm*K,     xs + (size_t)mm*nbq,
                                    xq + (size_t)(mm+1)*K, xs + (size_t)(mm+1)*nbq,
                                    xq + (size_t)(mm+2)*K, xs + (size_t)(mm+2)*nbq, K);
            for (; mm < M; mm++)
                matvec_sdot_8row(Y + (size_t)mm*Ys + i, g,
                                 xq + (size_t)mm*K, xs + (size_t)mm*nbq, K);
        }
    } else if (t->type == DS4F_BF16_PV) {
        const uint16_t *base = (const uint16_t *)t->w;
        /* K-tile sized so the per-token X block (not the weights) drives L1 reuse: the old 512 reloaded
         * X once PER TILE, throttling large M+K (the dominant cost). 4096 (min-of-20 bench: +20-40%
         * across all dense shapes/M; K=8192 wo_b -> 2 tiles, larger regresses). DS4F_GEMM_TILE_K overrides. */
        static int TILE_K = -1;
        if (TILE_K < 0) { const char *e = getenv("DS4F_GEMM_TILE_K"); TILE_K = (e && atoi(e) > 0) ? (atoi(e) & ~15) : 4096; }
        for (int i = r0; i + 7 < r1; i += 8) {
            const uint16_t *g = base + (size_t)(i / 8) * 8 * K;
            float (*acc)[8] = ds4f_acc_scratch(M);
            if (!acc) return;
            for (int mm = 0; mm < M; mm++)
                for (int r = 0; r < 8; r++) acc[mm][r] = 0.f;
            for (int k0 = 0; k0 < K; k0 += TILE_K) {       /* tile outer, token inner: */
                int klen = K - k0 < TILE_K ? K - k0 : TILE_K;
                const uint16_t *pA = g + 2*k0,       *pC = g + 2*K + 2*k0;
                const uint16_t *pE = g + 4*K + 2*k0, *pG = g + 6*K + 2*k0;
                int mm = 0;
                for (; mm + 2 < M; mm += 3)                /* 8x3 register-blocked: weight loaded once / 3 tokens */
                    matvec_bf16_8x3_pv_acc(acc[mm], acc[mm+1], acc[mm+2], pA, pC, pE, pG,
                                           X + (size_t)mm    *Xs + k0,
                                           X + (size_t)(mm+1)*Xs + k0,
                                           X + (size_t)(mm+2)*Xs + k0, klen);
                for (; mm < M; mm++)                       /* 1-2 token remainder */
                    matvec_bf16_8row_pv_acc(acc[mm], pA, pC, pE, pG,
                                            X + (size_t)mm*Xs + k0, klen);
            }
            for (int mm = 0; mm < M; mm++) {
                float *y = Y + (size_t)mm*Ys + i;
                for (int r = 0; r < 8; r++) y[r] = acc[mm][r];
            }
        }
    } else if (t->type == DS4F_BF16) {
        const uint16_t *base = (const uint16_t *)t->w;
        int nr = r1 - r0; if (nr <= 0) return;
        gemm_bf16_f32_tokmajor(Y + r0, base + (size_t)r0*K, X, nr, K, M, Ys, Xs);
    } else if (t->type == DS4F_MXFP4) {
        /* Expert GEMM. 8-row group's nibbles (K/2*8 B <= 16KB for K<=4096) fit L1,
         * so sweeping it once per token-pair keeps it HBM-read-once across all M
         * tokens (group-outer, token-inner). 2-token register-blocked + 1 rem. */
        const uint8_t *wbase = (const uint8_t *)t->w; size_t rb = (size_t)K / 2;
        const uint8_t *sbase = t->scale;              size_t sb = (size_t)K / 32;
        int mxtile = T->m->mxfp4_gemm_tile;
        if (mxtile && M >= mxtile) {
            /* TILE-DEQUANT (DS4F_MXFP4_GEMM_TILE, M>=thr): dequant each 8-row group's
             * nibbles ONCE (svtbl->scale->bf16, lossless: fp4*pow2 fits bf16's 7 mant
             * bits) into a tiny L1 bf16 PV tile, then reuse it across all M tokens via
             * the peak 8x3 bf16-pv microkernel -- so the svtbl dequant amortizes over M
             * instead of being re-run per token-pair. The plain svtbl 2x path (else
             * branch) is FLAT ~84 Gmac/s (dequant-bound); this scales toward the bf16
             * compute peak ~140 at M>=32 (measured ds4f_gemm_test, Step 2o). argmax-
             * exact + relL2~1e-6 vs per-token matvec (bf16 trunc lossless; K-tile
             * reassoc only). Assumes vl==16 (A64FX SVE-512): each 16-elem chunk is
             * exactly the lo (cc%32==0) or hi (cc%32==16) nibbles of one 32-block. */
            const int TK = 512;                       /* K-tile (mult of 32); 8x512 bf16 = 8 KB L1 */
            uint16_t *pv = ds4f_fp8bf16_tile((size_t)4 * 2 * TK);
            svbool_t pg = svptrue_b32(); svbool_t ph = svptrue_b16();
            svfloat32_t kv = svld1(pg, ds4f_kvalues_mxfp4_f32);
            int vl = (int)svcntw();
            for (int i = r0; i + 7 < r1; i += 8) {
                float (*acc)[8] = ds4f_acc_scratch(M);
                if (!acc) return;
                for (int mm = 0; mm < M; mm++) for (int r = 0; r < 8; r++) acc[mm][r] = 0.f;
                for (int k0 = 0; k0 < K; k0 += TK) {
                    int klen = K - k0 < TK ? K - k0 : TK;
                    for (int pr = 0; pr < 4; pr++) {
                        uint16_t *pb = pv + (size_t)pr * 2 * TK;
                        const uint8_t *wa = wbase + (size_t)(i + 2*pr)     * rb, *sa = sbase + (size_t)(i + 2*pr)     * sb;
                        const uint8_t *wb = wbase + (size_t)(i + 2*pr + 1) * rb, *sbp= sbase + (size_t)(i + 2*pr + 1) * sb;
                        for (int c = 0; c < klen; c += vl) {
                            int cc = k0 + c, blk = cc >> 5, half = (cc & 16) >> 4;   /* lo/hi nibble of 32-block */
                            svuint32_t ra = svld1ub_u32(pg, wa + (size_t)blk * 16);
                            svuint32_t rbw= svld1ub_u32(pg, wb + (size_t)blk * 16);
                            svuint32_t na = half ? svand_n_u32_x(pg, svlsr_n_u32_x(pg, ra,  4), 0xf) : svand_n_u32_x(pg, ra,  0xf);
                            svuint32_t nb = half ? svand_n_u32_x(pg, svlsr_n_u32_x(pg, rbw, 4), 0xf) : svand_n_u32_x(pg, rbw, 0xf);
                            svfloat32_t fa = svmul_x(pg, svtbl_f32(kv, na), svdup_f32(ggml_e8m0_to_fp32(sa[blk])));
                            svfloat32_t fb = svmul_x(pg, svtbl_f32(kv, nb), svdup_f32(ggml_e8m0_to_fp32(sbp[blk])));
                            svuint16_t a16 = svreinterpret_u16_u32(svlsr_n_u32_x(pg, svreinterpret_u32_f32(fa), 16));
                            svuint16_t b16 = svreinterpret_u16_u32(svlsr_n_u32_x(pg, svreinterpret_u32_f32(fb), 16));
                            svuint16_t ca = svuzp1_u16(a16, a16);
                            svuint16_t cb = svuzp1_u16(b16, b16);
                            svst1_u16(ph, pb + 2*c, svzip1_u16(ca, cb));
                        }
                    }
                    const uint16_t *pA = pv, *pC = pv + 2*TK, *pE = pv + 4*TK, *pG = pv + 6*TK;
                    int mm = 0;
                    for (; mm + 2 < M; mm += 3)
                        matvec_bf16_8x3_pv_acc(acc[mm], acc[mm+1], acc[mm+2], pA, pC, pE, pG,
                                               X + (size_t)mm    *Xs + k0,
                                               X + (size_t)(mm+1)*Xs + k0,
                                               X + (size_t)(mm+2)*Xs + k0, klen);
                    for (; mm < M; mm++)
                        matvec_bf16_8row_pv_acc(acc[mm], pA, pC, pE, pG, X + (size_t)mm*Xs + k0, klen);
                }
                for (int mm = 0; mm < M; mm++) {
                    float *y = Y + (size_t)mm*Ys + i;
                    for (int r = 0; r < 8; r++) y[r] = acc[mm][r];
                }
            }
        } else
        for (int i = r0; i + 7 < r1; i += 8) {
            const uint8_t *w = wbase + (size_t)i * rb, *s = sbase + (size_t)i * sb;
            const uint8_t *w0=w,*w1=w+rb,*w2=w+2*rb,*w3=w+3*rb,*w4=w+4*rb,*w5=w+5*rb,*w6=w+6*rb,*w7=w+7*rb;
            const uint8_t *s0=s,*s1=s+sb,*s2=s+2*sb,*s3=s+3*sb,*s4=s+4*sb,*s5=s+5*sb,*s6=s+6*sb,*s7=s+7*sb;
            int mm = 0;
            for (; mm + 1 < M; mm += 2)
                matvec_mxfp4_8row_2x(Y+(size_t)mm*Ys+i, Y+(size_t)(mm+1)*Ys+i,
                    w0,w1,w2,w3,w4,w5,w6,w7, s0,s1,s2,s3,s4,s5,s6,s7,
                    X+(size_t)mm*Xs, X+(size_t)(mm+1)*Xs, K);
            for (; mm < M; mm++)
                matvec_mxfp4_8row(Y+(size_t)mm*Ys+i, w0,w1,w2,w3,w4,w5,w6,w7,
                    s0,s1,s2,s3,s4,s5,s6,s7, X+(size_t)mm*Xs, K);
        }
    } else if (t->type == DS4F_FP8) {
        /* FP8 -> bf16 FUSED tile-dequant GEMM. For each 8-row group, dequant one
         * TILE_K-wide FP8 sub-tile into a tiny L1 PV pair-buffer (8 rows x 512 bf16
         * = 8 KB, never written to HBM) and immediately consume it across all M
         * tokens with the peak 8x3 pv microkernel, accumulating over K tiles. This
         * keeps the dequant L1-fused (no bf16 HBM round-trip) and reuses the exact
         * matvec_bf16_8x3_pv_acc kernel the DS4F_FP8_BF16=1 path uses -> compute-
         * bound prefill WITHOUT the +6 GB resident bf16 copy. argmax-exact vs the
         * per-token FP8 matvec (K-tile reassoc ~1e-4; FP8->bf16 itself lossless). */
        const uint8_t *wbase = (const uint8_t *)t->w;
        const uint8_t *sbase = t->scale;
        const uint32_t *lut = T->m->fp8_lut;
        int sbc = (K + 127) / 128;
        const int TK = 512;                       /* K-tile (mult of 128); 8x512 bf16 = 8 KB L1 */
        uint16_t *pv = ds4f_fp8bf16_tile((size_t)4 * 2 * TK);  /* 4 pair-bufs x 2*TK hw */
        svbool_t pg = svptrue_b32(); svbool_t ph = svptrue_b16();
        int vl = (int)svcntw();
        /* Tile-dequant uses the L1-LUT GATHER (not the register magic decode the M=1
         * decode matvec offers). Two reasons, both measured (ds4f_gemm_test FP8 sweep,
         * Step 2o): (1) this GEMM is bf16-FMA COMPUTE-bound, not dequant-bound -- the
         * dequant amortizes over M, so magic vs gather is NEUTRAL (1.00-1.06x, ~1% at
         * M>=32); (2) the magic decode needs FTZ, which flushes E4M3 SUBNORMALS to 0 ->
         * it is LOSSY vs the LUT, so it would break this path's bit-parity with the
         * DS4F_FP8_BF16=1 resident-bf16 prefill. So the gather (lossless) stays. The
         * on-register magic dequant's real win is the HBM-bound M=1 decode matvec. */
        for (int i = r0; i + 7 < r1; i += 8) {
            const uint8_t *es = sbase + (size_t)(i >> 7) * sbc;     /* i is 8-aligned -> 128-block via >>7 */
            float (*acc)[8] = ds4f_acc_scratch(M);
            if (!acc) return;
            for (int mm = 0; mm < M; mm++) for (int r = 0; r < 8; r++) acc[mm][r] = 0.f;
            for (int k0 = 0; k0 < K; k0 += TK) {
                int klen = K - k0 < TK ? K - k0 : TK;
                /* dequant 8 rows x klen -> 4 PV pair-bufs: pb[2c+2j]=bf16(rowA[j]),
                 * pb[2c+2j+1]=bf16(rowB[j]) (matches matvec_bf16_8row_pv's p_odd layout). */
                for (int pr = 0; pr < 4; pr++) {
                    uint16_t *pb = pv + (size_t)pr * 2 * TK;
                    const uint8_t *wa = wbase + (size_t)(i + 2*pr)     * K;
                    const uint8_t *wb = wbase + (size_t)(i + 2*pr + 1) * K;
                    for (int c = 0; c < klen; c += vl) {
                        int cc = k0 + c;       /* 16-chunk stays in one 128-scale block */
                        svfloat32_t vs = svdup_f32(ggml_e8m0_to_fp32(es[cc >> 7]));
                        svuint32_t ia = svld1ub_u32(pg, wa + cc);
                        svuint32_t ib = svld1ub_u32(pg, wb + cc);
                        svfloat32_t fa = svmul_x(pg, svreinterpret_f32_u32(svld1_gather_u32index_u32(pg, lut, ia)), vs);
                        svfloat32_t fb = svmul_x(pg, svreinterpret_f32_u32(svld1_gather_u32index_u32(pg, lut, ib)), vs);
                        svuint16_t a16 = svreinterpret_u16_u32(svlsr_n_u32_x(pg, svreinterpret_u32_f32(fa), 16));
                        svuint16_t b16 = svreinterpret_u16_u32(svlsr_n_u32_x(pg, svreinterpret_u32_f32(fb), 16));
                        svuint16_t ca = svuzp1_u16(a16, a16);   /* lanes 0..15 = bf16(rowA) */
                        svuint16_t cb = svuzp1_u16(b16, b16);
                        svst1_u16(ph, pb + 2*c, svzip1_u16(ca, cb));   /* [a0,b0,a1,b1,...] */
                    }
                }
                const uint16_t *pA = pv, *pC = pv + 2*TK, *pE = pv + 4*TK, *pG = pv + 6*TK;
                int mm = 0;
                for (; mm + 2 < M; mm += 3)
                    matvec_bf16_8x3_pv_acc(acc[mm], acc[mm+1], acc[mm+2], pA, pC, pE, pG,
                                           X + (size_t)mm    *Xs + k0,
                                           X + (size_t)(mm+1)*Xs + k0,
                                           X + (size_t)(mm+2)*Xs + k0, klen);
                for (; mm < M; mm++)
                    matvec_bf16_8row_pv_acc(acc[mm], pA, pC, pE, pG, X + (size_t)mm*Xs + k0, klen);
            }
            for (int mm = 0; mm < M; mm++) {
                float *y = Y + (size_t)mm*Ys + i;
                for (int r = 0; r < 8; r++) y[r] = acc[mm][r];
            }
        }
    }
    /* DS4F_F32 (tiny mHC mixes) never reaches the worker (handled in ds4f_gemm). */
}
#endif /* __ARM_FEATURE_SVE */

#if !defined(__ARM_FEATURE_SVE)
/* AMD prefill worker.  The A64FX worker above uses SVE GEMM microkernels; on
 * AVX2, dispatching the whole M-token tile through one worker is still a large
 * win over calling ds4f_matvec (and its barrier) M times for every projection.
 * The AVX2 8-row kernels keep the hot decode math unchanged while this worker
 * amortizes pool synchronization and row scheduling. */
static void ds4f_gemm_worker_x86(void *arg, int tid, int nthr) {
    ds4f_gemm_task *T = (ds4f_gemm_task *)arg;
    const ds4f_tensor *t = T->t;
    int K = t->cols, M = T->M, Ys = T->Ystride, Xs = T->Xstride;
    int r0, r1;
    ds4f_rowsplit8(t->rows, nthr, tid, &r0, &r1);
    if (t->type == DS4F_MXFP4 && T->m->mxfp4_raw) {
        /* Prepare the on-disk activation once per token, not once per
         * 8-row group. This matters for a prefill tile because a 4k vector
         * would otherwise be permuted/quantized hundreds of times. */
        size_t rb = (size_t)K / 2, sb = (size_t)K / 32;
        /* Prefill tile: quantize the whole tile once, then run rows outside
         * the token loop so each weight block's nibble decode is shared by
         * four tokens instead of repeated M times.  Per-token results are
         * bit-identical to the token-at-a-time path below. */
        if (M >= 4 && !ds4f_mxfp4_w4a8_on(T->m)) {
            float *xp = (float *)T->mx_xp;
            if (!xp) xp = ds4f_mxfp4_xperm_n(K, M);
            if (xp) {
                if (!T->mx_xp) for (int mm = 0; mm < M; mm++)
                        ds4f_mxfp4_perm_act_f32(T->X + (size_t)mm * Xs, K,
                                                xp + (size_t)mm * K);
                for (int i = r0; i + 7 < r1; i += 8) {
                    const uint8_t *w = (const uint8_t *)t->w + (size_t)i * rb;
                    const uint8_t *s = t->scale + (size_t)i * sb;
                    for (int r = 0; r < 8; r++) {
                        const uint8_t *wr = w + (size_t)r * rb;
                        const uint8_t *sr = s + (size_t)r * sb;
                        int mm = 0;
                        for (; mm + 3 < M; mm += 4)
                            matvec_mxfp4_1row_f32_raw_4x(
                                T->Y + (size_t)mm * Ys + i + r, Ys,
                                wr, sr, xp + (size_t)mm * K, K);
                        for (; mm < M; mm++)
                            matvec_mxfp4_1row_f32_raw(
                                T->Y + (size_t)mm * Ys + i + r, wr, sr,
                                xp + (size_t)mm * K, K);
                    }
                }
                return;
            }
        }
        if (M >= 4 && ds4f_mxfp4_w4a8_on(T->m)) {
            int8_t *xq = (int8_t *)T->mx_xq;
            float *xs = (float *)T->mx_xs, *xc = (float *)T->mx_xc;
            int ready = xq && xs && xc;
            if (ready || ds4f_mxfp4_xscratch_n(K, M, &xq, &xs, &xc) == 0) {
                if (!ready) for (int mm = 0; mm < M; mm++)
                    ds4f_mxfp4_quant_act_raw(T->X + (size_t)mm * Xs, K,
                                             xq + (size_t)mm * K,
                                             xs + (size_t)mm * (K / 32),
                                             xc + (size_t)mm * (K / 32));
                for (int i = r0; i + 7 < r1; i += 8) {
                    const uint8_t *w = (const uint8_t *)t->w + (size_t)i * rb;
                    const uint8_t *s = t->scale + (size_t)i * sb;
                    for (int r = 0; r < 8; r++) {
                        const uint8_t *wr = w + (size_t)r * rb;
                        const uint8_t *sr = s + (size_t)r * sb;
                        int mm = 0;
                        /* Token blocks innermost: this row's 2 KB of weights
                         * stays in L1 across the whole tile. */
                        for (; mm + 3 < M; mm += 4)
                            matvec_mxfp4_1row_i8_raw_4x(
                                T->Y + (size_t)mm * Ys + i + r, Ys,
                                wr, sr, xq + (size_t)mm * K,
                                xs + (size_t)mm * (K / 32),
                                xc + (size_t)mm * (K / 32), K);
                        for (; mm < M; mm++)
                            matvec_mxfp4_1row_i8_raw(
                                T->Y + (size_t)mm * Ys + i + r, wr, sr,
                                xq + (size_t)mm * K,
                                xs + (size_t)mm * (K / 32),
                                xc + (size_t)mm * (K / 32), K);
                    }
                }
                return;
            }
        }
        for (int mm = 0; mm < M; mm++) {
            float *y = T->Y + (size_t)mm * Ys;
            const float *x = T->X + (size_t)mm * Xs;
            int8_t *xq = NULL; float *xs = NULL, *xc = NULL, *xp = NULL;
            if (ds4f_mxfp4_w4a8_on(T->m)) {
                ds4f_mxfp4_xscratch(K, &xq, &xs, &xc);
                ds4f_mxfp4_quant_act_raw(x, K, xq, xs, xc);
            } else {
                xp = ds4f_mxfp4_xperm(K);
                ds4f_mxfp4_perm_act_f32(x, K, xp);
            }
            for (int i = r0; i + 7 < r1; i += 8) {
                const uint8_t *w = (const uint8_t *)t->w + (size_t)i * rb;
                const uint8_t *s = t->scale + (size_t)i * sb;
                for (int r = 0; r < 8; r++) {
                    if (ds4f_mxfp4_w4a8_on(T->m))
                        matvec_mxfp4_1row_i8_raw(y + i + r,
                            w + (size_t)r * rb, s + (size_t)r * sb,
                            xq, xs, xc, K);
                    else
                        matvec_mxfp4_1row_f32_raw(y + i + r,
                            w + (size_t)r * rb, s + (size_t)r * sb,
                            xp, K);
                }
            }
        }
        return;
    }
    /* Group rows outside the token loop. Each 8-row weight group is small
     * enough to stay in the shared cache while the M activations are consumed,
     * reducing repeated weight-stream traffic during prefill. */
    for (int i = r0; i + 7 < r1; i += 8) {
        int sbc = (K + 127) / 128;
        size_t rb = (size_t)K / 2, sb = (size_t)K / 32;
        const uint16_t *g16 = (const uint16_t *)t->w + (size_t)(i / 8) * 8 * K;
        const uint16_t *w16 = (const uint16_t *)t->w + (size_t)i * K;
        const uint8_t *w8 = (const uint8_t *)t->w + (size_t)i * K;
        const uint8_t *wm = (const uint8_t *)t->w + (size_t)i * rb;
        const uint8_t *sm = t->scale ? t->scale + (size_t)i * sb : NULL;
        const uint8_t *es = t->scale ? t->scale + (size_t)(i / 128) * sbc : NULL;
        int mm = 0;
        if (t->type == DS4F_BF16) {
            for (; mm + 1 < M; mm += 2) {
                float *y0 = T->Y + (size_t)mm * Ys + i;
                float *y1 = T->Y + (size_t)(mm + 1) * Ys + i;
                matvec_bf16_8row_2x(y0, y1, w16, w16 + K, w16 + 2*K, w16 + 3*K,
                                    w16 + 4*K, w16 + 5*K, w16 + 6*K, w16 + 7*K,
                                    T->X + (size_t)mm * Xs,
                                    T->X + (size_t)(mm + 1) * Xs, K);
            }
            if (mm < M) {
                float *y = T->Y + (size_t)mm * Ys + i;
                matvec_bf16_8row(y, w16, w16 + K, w16 + 2*K, w16 + 3*K,
                                 w16 + 4*K, w16 + 5*K, w16 + 6*K, w16 + 7*K,
                                 T->X + (size_t)mm * Xs, K);
            }
        } else if (t->type == DS4F_FP8) {
            for (; mm + 1 < M; mm += 2) {
                float *y0 = T->Y + (size_t)mm * Ys + i;
                float *y1 = T->Y + (size_t)(mm + 1) * Ys + i;
                matvec_fp8e4m3_8row_2x(
                    y0, y1, w8, w8 + K, w8 + 2*K, w8 + 3*K,
                    w8 + 4*K, w8 + 5*K, w8 + 6*K, w8 + 7*K,
                    es, T->m->fp8_lut,
                    T->X + (size_t)mm * Xs,
                    T->X + (size_t)(mm + 1) * Xs, K);
            }
            if (mm < M) {
                float *y = T->Y + (size_t)mm * Ys + i;
                matvec_fp8e4m3_8row(y, w8, w8 + K, w8 + 2*K, w8 + 3*K,
                                    w8 + 4*K, w8 + 5*K, w8 + 6*K, w8 + 7*K,
                                    es, T->m->fp8_lut,
                                    T->X + (size_t)mm * Xs, K);
            }
        } else {
            for (; mm < M; mm++) {
                float *y = T->Y + (size_t)mm * Ys + i;
                const float *x = T->X + (size_t)mm * Xs;
                if (t->type == DS4F_BF16_PV)
                    matvec_bf16_8row_pv(y, g16, g16 + 2*K, g16 + 4*K, g16 + 6*K, x, K);
                else if (t->type == DS4F_MXFP4)
                    matvec_mxfp4_8row(y, wm, wm + rb, wm + 2*rb, wm + 3*rb,
                                      wm + 4*rb, wm + 5*rb, wm + 6*rb, wm + 7*rb,
                                      sm, sm + sb, sm + 2*sb, sm + 3*sb,
                                      sm + 4*sb, sm + 5*sb, sm + 6*sb, sm + 7*sb,
                                      x, K);
            }
        }
    }
}
#endif /* !__ARM_FEATURE_SVE */

static int ds4f_gemm_warned = 0;
static void ds4f_gemm(ds4f_model *m, float *Y, const ds4f_tensor *t,
                      const float *X, int M, int Ystride, int Xstride) {
    if (M > DS4F_MAX_MTILE) { fprintf(stderr, "ds4f_gemm: M=%d > DS4F_MAX_MTILE=%d\n", M, DS4F_MAX_MTILE); abort(); }
    if (M > 1 && !m->gpu_exact_prefill && m->gpu_dense_gemm && t->gpu_id >= 0 &&
        (t->type == DS4F_FP8 || t->type == DS4F_BF16 ||
            t->type == DS4F_MXFP4)) {
        if (m->gpu_dense_gemm(m->gpu_dense_ctx, Y, t, X, M, Ystride, Xstride) == 0) {
            m->bytes_read += ds4f_wbytes(t->type, t->rows, t->cols)
                           + ds4f_sbytes(t->type, t->rows, t->cols);
            return;
        }
        /* The GPU adapter deliberately returns nonzero for an unavailable
         * shape/allocation so the existing CPU GEMM remains a safe fallback. */
    }
#if defined(__ARM_FEATURE_SVE)
    if (t->type == DS4F_BF16_PV || t->type == DS4F_BF16 || t->type == DS4F_MXFP4 ||
        t->type == DS4F_Q8_PV   || t->type == DS4F_FP8) {
        m->bytes_read += ds4f_wbytes(t->type, t->rows, t->cols)    /* read once across all M */
                       + ds4f_sbytes(t->type, t->rows, t->cols);   /* MXFP4/FP8 block scales */
        ds4f_gemm_task T = { m, Y, t, X, M, Ystride, Xstride, NULL, NULL, NULL, NULL };
        ds4f_pool_run(m->pool, ds4f_gemm_worker, &T);
    } else
#elif defined(__AVX2__) && defined(__FMA__)
    if (t->type == DS4F_BF16_PV || t->type == DS4F_BF16 || t->type == DS4F_MXFP4 ||
        t->type == DS4F_FP8) {
        m->bytes_read += ds4f_wbytes(t->type, t->rows, t->cols)
                       + ds4f_sbytes(t->type, t->rows, t->cols);
        ds4f_gemm_task T = { m, Y, t, X, M, Ystride, Xstride, NULL, NULL, NULL, NULL };
        ds4f_pool_run(m->pool, ds4f_gemm_worker_x86, &T);
    } else
#endif
    {   /* ds4f_gemm_worker's tiled/PV/tile-dequant kernels are SVE-only, so every
         * dtype takes this path off A64FX: correct, but one matvec per token
         * instead of a batched GEMM. Prefill throughput only; decode is M=1. */
        if (!ds4f_gemm_warned) {
            fprintf(stderr, "WARN: ds4f_gemm per-token matvec fallback (dtype %d, expected F32 mHC) -> "
                            "prefill not batched for this tensor\n", t->type);
            ds4f_gemm_warned = 1;
        }
        for (int mm = 0; mm < M; mm++)         /* ds4f_matvec accounts bytes_read per call */
            ds4f_matvec(m, Y + (size_t)mm*Ystride, t, X + (size_t)mm*Xstride);
    }
}

/* Run independent batched GEMMs under one pool barrier.  Prefill has several
 * projection groups with the same activation tile (wq_a/wkv, the eight
 * grouped o_proj contractions, shared gate/up, and expert gate/up); dispatching
 * them together removes wake/barrier gaps without changing any GEMM's row or K
 * reduction order. */
typedef struct { ds4f_gemm_task *tasks; int n; } ds4f_gemm_multi_task;

/* Quantize/permute each distinct routed activation bucket once on the caller
 * instead of once per pool worker. Gate/up tasks share the same prepared
 * buffer. The old worker-local preparation repeated M*K work 16 times. */
static int ds4f_gemm_prepare_mxfp4(ds4f_gemm_task *q, ds4f_gemm_task *prev,
                                   int nprev, void **owned, int *nowned) {
    if (!q || q->M < 4 || q->t->type != DS4F_MXFP4 || !q->m->mxfp4_raw) return 0;
    int K = q->t->cols;
    for (int i = 0; i < nprev; ++i) {
        ds4f_gemm_task *p = &prev[i];
        if (p->X == q->X && p->M == q->M && p->Xstride == q->Xstride &&
            p->t->cols == K) {
            q->mx_xq = p->mx_xq; q->mx_xs = p->mx_xs;
            q->mx_xc = p->mx_xc; q->mx_xp = p->mx_xp;
            if (q->mx_xq || q->mx_xp) return 0;
        }
    }
    size_t ne = (size_t)q->M * (size_t)K;
    if (ds4f_mxfp4_w4a8_on(q->m)) {
        size_t nb = ne / 32;
        uint8_t *mem = (uint8_t *)ds4f_map_alloc(ne + 2 * nb * sizeof(float), 64, 0);
        if (!mem) return -1;
        q->mx_xq = (int8_t *)mem; q->mx_xs = (float *)(mem + ne);
        q->mx_xc = q->mx_xs + nb;
        for (int mm = 0; mm < q->M; ++mm) ds4f_mxfp4_quant_act_raw(
            q->X + (size_t)mm*q->Xstride, K,
            (int8_t *)q->mx_xq + (size_t)mm*K,
            (float *)q->mx_xs + (size_t)mm*(K/32),
            (float *)q->mx_xc + (size_t)mm*(K/32));
        owned[(*nowned)++] = mem;
    } else {
        float *xp = (float *)ds4f_map_alloc(ne * sizeof(float), 64, 0);
        if (!xp) return -1;
        q->mx_xp = xp;
        for (int mm = 0; mm < q->M; ++mm) ds4f_mxfp4_perm_act_f32(
            q->X + (size_t)mm*q->Xstride, K, xp + (size_t)mm*K);
        owned[(*nowned)++] = xp;
    }
    return 0;
}

static void ds4f_gemm_multi_worker(void *arg, int tid, int nthr) {
    ds4f_gemm_multi_task *T = (ds4f_gemm_multi_task *)arg;
#if defined(__ARM_FEATURE_SVE)
    for (int i = 0; i < T->n; i++) ds4f_gemm_worker(&T->tasks[i], tid, nthr);
#elif defined(__AVX2__) && defined(__FMA__)
    for (int i = 0; i < T->n; i++) ds4f_gemm_worker_x86(&T->tasks[i], tid, nthr);
#else
    (void)T; (void)tid; (void)nthr;
#endif
}

static int ds4f_gemm_gpu_eligible(const ds4f_model *m, const ds4f_gemm_task *q) {
    return m && !m->gpu_exact_prefill && m->gpu_dense_gemm && q && q->M > 1 && q->t &&
           q->t->gpu_id >= 0 &&
           (q->t->type == DS4F_FP8 || q->t->type == DS4F_BF16 ||
            q->t->type == DS4F_MXFP4);
}

/* Dispatch just the GPU-owned members of a mixed independent group.  The
 * dense adapter is synchronous at this boundary, so CPU-owned siblings can
 * safely run immediately afterward without a new dependency or stream API. */
static void ds4f_gemm_gpu_subset(ds4f_model *m, ds4f_gemm_task *tasks,
                                 const int *idx, int n) {
    if (n > 1 && m->gpu_dense_gemm_multi) {
        float **Y = (float **)alloca((size_t)n * sizeof(*Y));
        const ds4f_tensor **T = (const ds4f_tensor **)alloca((size_t)n * sizeof(*T));
        const float **X = (const float **)alloca((size_t)n * sizeof(*X));
        int *M = (int *)alloca((size_t)n * sizeof(*M));
        int *Ys = (int *)alloca((size_t)n * sizeof(*Ys));
        int *Xs = (int *)alloca((size_t)n * sizeof(*Xs));
        for (int k = 0; k < n; ++k) {
            ds4f_gemm_task *q = &tasks[idx[k]];
            Y[k] = q->Y; T[k] = q->t; X[k] = q->X;
            M[k] = q->M; Ys[k] = q->Ystride; Xs[k] = q->Xstride;
        }
        if (m->gpu_dense_gemm_multi(m->gpu_dense_ctx, Y, T, X,
                                    M, Ys, Xs, n) == 0) {
            for (int k = 0; k < n; ++k)
                m->bytes_read += ds4f_wbytes(tasks[idx[k]].t->type,
                                             tasks[idx[k]].t->rows,
                                             tasks[idx[k]].t->cols)
                               + ds4f_sbytes(tasks[idx[k]].t->type,
                                             tasks[idx[k]].t->rows,
                                             tasks[idx[k]].t->cols);
            return;
        }
    }
    for (int k = 0; k < n; ++k) {
        ds4f_gemm_task *q = &tasks[idx[k]];
        ds4f_gemm(m, q->Y, q->t, q->X, q->M, q->Ystride, q->Xstride);
    }
}

static inline int ds4f_gemm_supported(const ds4f_tensor *t) {
    return t && (t->type == DS4F_BF16_PV || t->type == DS4F_BF16 ||
                 t->type == DS4F_MXFP4 || t->type == DS4F_FP8);
}

static void ds4f_gemm_multi(ds4f_model *m, ds4f_gemm_task *tasks, int n) {
    if (n > 0 && m->gpu_dense_gemm) {
        int gpu_idx[n];
        int ng = 0;
        for (int i = 0; i < n; i++)
            if (ds4f_gemm_gpu_eligible(m, &tasks[i])) gpu_idx[ng++] = i;
        if (ng > 0 && (ng == n || m->gpu_dense_mixed)) {
            ds4f_gemm_gpu_subset(m, tasks, gpu_idx, ng);
            for (int i = 0; i < n; i++) {
                int is_gpu = 0;
                for (int k = 0; k < ng; ++k)
                    if (gpu_idx[k] == i) { is_gpu = 1; break; }
                if (!is_gpu)
                    ds4f_gemm(m, tasks[i].Y, tasks[i].t, tasks[i].X,
                              tasks[i].M, tasks[i].Ystride, tasks[i].Xstride);
            }
            return;
        }
    }
#if defined(__ARM_FEATURE_SVE) || (defined(__AVX2__) && defined(__FMA__))
    int supported = n > 0;
    for (int i = 0; i < n; i++) {
        if (!ds4f_gemm_supported(tasks[i].t) || tasks[i].M > DS4F_MAX_MTILE) {
            supported = 0;
            break;
        }
    }
    if (supported) {
        for (int i = 0; i < n; i++)
            m->bytes_read += ds4f_wbytes(tasks[i].t->type, tasks[i].t->rows, tasks[i].t->cols)
                           + ds4f_sbytes(tasks[i].t->type, tasks[i].t->rows, tasks[i].t->cols);
        void **owned = (void **)alloca((size_t)n * sizeof(void *));
        int nowned = 0;
        for (int i = 0; i < n; i++)
            (void)ds4f_gemm_prepare_mxfp4(&tasks[i], tasks, i, owned, &nowned);
        ds4f_gemm_multi_task T = { tasks, n };
        ds4f_pool_run(m->pool, ds4f_gemm_multi_worker, &T);
        for (int i = 0; i < n; i++) {
            tasks[i].mx_xq = NULL; tasks[i].mx_xs = NULL;
            tasks[i].mx_xc = NULL; tasks[i].mx_xp = NULL;
        }
        for (int i = 0; i < nowned; i++) ds4f_map_free(owned[i]);
        return;
    }
#endif
    for (int i = 0; i < n; i++)
        ds4f_gemm(m, tasks[i].Y, tasks[i].t, tasks[i].X,
                  tasks[i].M, tasks[i].Ystride, tasks[i].Xstride);
}

static void ds4f_gemm_pair(ds4f_model *m,
                           float *Y0, const ds4f_tensor *t0,
                           float *Y1, const ds4f_tensor *t1,
                           const float *X, int M,
                           int Ystride0, int Ystride1, int Xstride) {
    ds4f_gemm_task tasks[2] = {
        { m, Y0, t0, X, M, Ystride0, Xstride, NULL, NULL, NULL, NULL },
        { m, Y1, t1, X, M, Ystride1, Xstride, NULL, NULL, NULL, NULL }
    };
    ds4f_gemm_multi(m, tasks, 2);
}

/* a [nrows, cols] row-slice VIEW of tensor t starting at logical row row0 (no copy).
 * row0 must be 8-aligned (kernel 8-row blocking) and, for FP8, 128-aligned (its
 * scale is addressed in 128-row blocks). Used to drive the grouped low-rank
 * o-projection (wo_a is block-diagonal: 8 groups of o_lora rows) as 8 matvecs. */
static inline ds4f_tensor ds4f_row_slice(const ds4f_tensor *t, int row0, int nrows) {
    /* Preserve gpu_id: the HIP bank resolves this host pointer back to the
     * already-uploaded full matrix and applies row0 as a device offset. CPU
     * backends ignore the optional id. */
    ds4f_tensor v = *t; v.rows = nrows;
    size_t wbpr;                              /* weight bytes per logical row */
    switch (t->type) {
        case DS4F_FP8:   wbpr = (size_t)t->cols;     break;
        case DS4F_MXFP4: wbpr = (size_t)t->cols / 2; break;
        case DS4F_F32:   wbpr = (size_t)t->cols * 4; break;
        case DS4F_Q8_PV: wbpr = ((size_t)(t->cols / 64) * 528) / 8; break;
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
    m->rope_dense_cos = (float *)ds4f_mem_alloc(m->mem, n * 4, 64, 1);
    m->rope_dense_sin = (float *)ds4f_mem_alloc(m->mem, n * 4, 64, 1);
    m->rope_comp_cos  = (float *)ds4f_mem_alloc(m->mem, n * 4, 64, 1);
    m->rope_comp_sin  = (float *)ds4f_mem_alloc(m->mem, n * 4, 64, 1);
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
    float *kvl = (float *)ds4f_map_alloc((size_t)cutoff * W * sizeof(float), 64, 0);
    float *scl = (float *)ds4f_map_alloc((size_t)cutoff * W * sizeof(float), 64, 0);
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
    ds4f_map_free(kvl); ds4f_map_free(scl);
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
#if defined(__ARM_FEATURE_SVE)
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
#else
static void ds4f_f32mv_worker(void *arg, int tid, int nthr) {
    ds4f_f32mv_task *T = (ds4f_f32mv_task *)arg;
    int rows = T->rows, cols = T->cols;
    int per = rows / nthr, extra = rows % nthr;
    int o0 = per * tid + (tid < extra ? tid : extra);
    int o1 = o0 + per + (tid < extra ? 1 : 0);
    const float *x = T->x;
    for (int o = o0; o < o1; o++) {
        const float *w = T->w + (size_t)o * cols;
        float acc = 0.f;
        for (int i = 0; i < cols; i++) acc += w[i] * x[i];
        T->out[o] = acc;
    }
}
#endif /* __ARM_FEATURE_SVE */

/* two outputs sharing one x load: kv[o]=wkv[o].x, score[o]=wgate[o].x (compressor) */
typedef struct { float *kv, *score; const float *wkv, *wgate, *x; int W, dim; } ds4f_cmpmv_task;
#if defined(__ARM_FEATURE_SVE)
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
#else
static void ds4f_cmpmv_worker(void *arg, int tid, int nthr) {
    ds4f_cmpmv_task *T = (ds4f_cmpmv_task *)arg;
    int W = T->W, dim = T->dim;
    int per = W / nthr, extra = W % nthr;
    int o0 = per * tid + (tid < extra ? tid : extra);
    int o1 = o0 + per + (tid < extra ? 1 : 0);
    const float *x = T->x;
    for (int o = o0; o < o1; o++) {
        const float *wk = T->wkv + (size_t)o * dim, *wg = T->wgate + (size_t)o * dim;
        float a = 0.f, b = 0.f;
        for (int i = 0; i < dim; i++) { float xv = x[i]; a += wk[i] * xv; b += wg[i] * xv; }
        T->kv[o] = a; T->score[o] = b;
    }
}
#endif /* __ARM_FEATURE_SVE */

/* ---- bf16-weight variants (same accumulation shape, weight widened in-lane) ----
 * The compressor/indexer weights are stored bf16 (their sources are bf16/FP8-e4m3,
 * both of which fit bf16 losslessly), so widen(stored_bf16) == the f32-widened value
 * BIT-EXACTLY: svld1uh zero-extends the halfword, <<16 reconstructs the f32 the f32
 * path would have loaded => the svmla inputs (and thus svaddv) are bit-identical to
 * ds4f_f32mv_worker / ds4f_cmpmv_worker. Half the weight bytes for identical output. */
typedef struct { float *out; const uint16_t *w; const float *x; int rows, cols; } ds4f_bf16mv_task;
#if defined(__ARM_FEATURE_SVE)
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
#else
static void ds4f_bf16mv_worker(void *arg, int tid, int nthr) {
    ds4f_bf16mv_task *T = (ds4f_bf16mv_task *)arg;
    int rows = T->rows, cols = T->cols;
    int per = rows / nthr, extra = rows % nthr;
    int o0 = per * tid + (tid < extra ? tid : extra);
    int o1 = o0 + per + (tid < extra ? 1 : 0);
    const float *x = T->x;
    for (int o = o0; o < o1; o++) {
        const uint16_t *w = T->w + (size_t)o * cols;
        float acc = 0.f;
        for (int i = 0; i < cols; i++) acc += ds4f_bf16_to_f32(w[i]) * x[i];
        T->out[o] = acc;
    }
}
#endif /* __ARM_FEATURE_SVE */

typedef struct { float *kv, *score; const uint16_t *wkv, *wgate; const float *x; int W, dim; } ds4f_cmpmv_bf16_task;
#if defined(__ARM_FEATURE_SVE)
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
#else
static void ds4f_cmpmv_bf16_worker(void *arg, int tid, int nthr) {
    ds4f_cmpmv_bf16_task *T = (ds4f_cmpmv_bf16_task *)arg;
    int W = T->W, dim = T->dim;
    int per = W / nthr, extra = W % nthr;
    int o0 = per * tid + (tid < extra ? tid : extra);
    int o1 = o0 + per + (tid < extra ? 1 : 0);
    const float *x = T->x;
    for (int o = o0; o < o1; o++) {
        const uint16_t *wk = T->wkv + (size_t)o * dim, *wg = T->wgate + (size_t)o * dim;
        float a = 0.f, b = 0.f;
        for (int i = 0; i < dim; i++) {
            float xv = x[i];
            a += ds4f_bf16_to_f32(wk[i]) * xv;
            b += ds4f_bf16_to_f32(wg[i]) * xv;
        }
        T->kv[o] = a; T->score[o] = b;
    }
}
#endif /* __ARM_FEATURE_SVE */

/* index_score parallelized over compressed positions t: score[t] = sum_h
 * relu(q[h].kvc[t]) * weights[h]. q[H*hd], kvc[T*hd], weights[H]. */
typedef struct { const float *q, *kvc, *weights; int H, hd, T; float *score; } ds4f_idxsc_task;
#if defined(__ARM_FEATURE_SVE)
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
#else
static void ds4f_idxsc_worker(void *arg, int tid, int nthr) {
    ds4f_idxsc_task *Tk = (ds4f_idxsc_task *)arg;
    int Tn = Tk->T, H = Tk->H, hd = Tk->hd;
    int per = Tn / nthr, extra = Tn % nthr;
    int t0 = per * tid + (tid < extra ? tid : extra);
    int t1 = t0 + per + (tid < extra ? 1 : 0);
    for (int t = t0; t < t1; t++) {
        const float *kt = Tk->kvc + (size_t)t * hd;
        float acc = 0.f;
        for (int h = 0; h < H; h++) {
            const float *qh = Tk->q + (size_t)h * hd;
            float dot = 0.f;
            for (int x = 0; x < hd; x++) dot += qh[x] * kt[x];
            if (dot < 0.f) dot = 0.f;
            acc += dot * Tk->weights[h];
        }
        Tk->score[t] = acc;
    }
}
#endif /* __ARM_FEATURE_SVE */

/* --- RESIDENT int8/SVE indexer scan (gated DS4F_IDX_INT8, default off) -----------------
 * idx_kv is Hadamard-rotated + fp4-act-quantized at write (rotation kills outlier/sink
 * channels), so PER-POSITION int8 is safe and slightly beats per-channel-static
 * (tools/idxscore_probe.c: top-512 SELECTED-SET 510/512 vs 503/512). The cache is quantized
 * ONCE at write (ds4f_idx_quant_pos, below) into idx_kv8 + a per-position scale; the scan
 * reads int8 DIRECTLY => 4x fewer bytes moved AND svdot_s32 (4 int8 MACs/lane). The offline
 * multi-thread BW probe (tools/idxscan_bw_probe.c) shows a constant ~1.87x over the f32
 * svmla scan that SCALES to 48 threads -- the scan is compute-bound, NOT bandwidth-bound, so
 * (unlike an on-the-fly-quant scan, which re-reads f32 and ties f32) the win survives at
 * production thread counts. dot = sq[h]*pscale[t]*svdot(q8[h],k8[t]). Requires hd%svcntb()==0
 * (hd=128). The dead on-the-fly variant was removed after it measured PARITY end-to-end. */
static inline void ds4f_idx_quant_pos(const float *v, int hd, int8_t *out, float *pscale) {
    float mx = 0.f;
    for (int d = 0; d < hd; d++) { float a = v[d] < 0 ? -v[d] : v[d]; if (a > mx) mx = a; }
    *pscale = mx * (1.0f / 127.0f);
    float inv = mx > 0 ? 127.0f / mx : 0.0f;
    for (int d = 0; d < hd; d++) { int q = (int)lrintf(v[d] * inv); q = q > 127 ? 127 : (q < -127 ? -127 : q); out[d] = (int8_t)q; }
}
typedef struct { const int8_t *q8, *k8; const float *sq, *pscale, *weights;
                 int H, hd, T; float *score; } ds4f_idxsc8r_task;
#if defined(__ARM_FEATURE_SVE)
static void ds4f_idxsc8r_worker(void *arg, int tid, int nthr) {
    ds4f_idxsc8r_task *Tk = (ds4f_idxsc8r_task *)arg;
    int Tn = Tk->T, H = Tk->H, hd = Tk->hd;
    int per = Tn / nthr, extra = Tn % nthr;
    int t0 = per * tid + (tid < extra ? tid : extra), t1 = t0 + per + (tid < extra ? 1 : 0);
    svbool_t pb = svptrue_b8(); int bf = (int)svcntb();
    for (int t = t0; t < t1; t++) {
        const int8_t *kt = Tk->k8 + (size_t)t * hd;
        float ps = Tk->pscale[t], acc = 0.f;
        for (int h = 0; h < H; h++) {
            const int8_t *q8 = Tk->q8 + (size_t)h * hd;
            svint32_t a = svdup_s32(0);
            for (int d = 0; d < hd; d += bf) a = svdot_s32(a, svld1_s8(pb, q8 + d), svld1_s8(pb, kt + d));
            float dot = Tk->sq[h] * ps * (float)svaddv_s32(svptrue_b32(), a);
            if (dot < 0.f) dot = 0.f;                /* relu */
            acc += dot * Tk->weights[h];
        }
        Tk->score[t] = acc;
    }
}
#else
static void ds4f_idxsc8r_worker(void *arg, int tid, int nthr) {
    ds4f_idxsc8r_task *Tk = (ds4f_idxsc8r_task *)arg;
    int Tn = Tk->T, H = Tk->H, hd = Tk->hd;
    int per = Tn / nthr, extra = Tn % nthr;
    int t0 = per * tid + (tid < extra ? tid : extra), t1 = t0 + per + (tid < extra ? 1 : 0);
    for (int t = t0; t < t1; t++) {
        const int8_t *kt = Tk->k8 + (size_t)t * hd;
        float ps = Tk->pscale[t], acc = 0.f;
        for (int h = 0; h < H; h++) {
            const int8_t *q8 = Tk->q8 + (size_t)h * hd;
            int32_t a = 0;
            for (int d = 0; d < hd; d++) a += (int32_t)q8[d] * (int32_t)kt[d];
            float dot = Tk->sq[h] * ps * (float)a;
            if (dot < 0.f) dot = 0.f;
            acc += dot * Tk->weights[h];
        }
        Tk->score[t] = acc;
    }
}
#endif /* __ARM_FEATURE_SVE */
/* int4 indexer cache (DS4F_IDX_INT4): same per-position scheme as ds4f_idx_quant_pos but +/-7,
 * 2 nibbles/byte -> idx_kv8_4 = nslot*hd/2 bytes (672->336 B/pos). The scan unpacks each
 * position's int4 key to an int8 temp (hd ops, amortized over H heads ~6%) then reuses svdot. */
static inline void ds4f_idx_quant_pos_i4(const float *v, int hd, uint8_t *out, float *pscale) {
    float mx = 0.f;
    for (int d = 0; d < hd; d++) { float a = v[d] < 0 ? -v[d] : v[d]; if (a > mx) mx = a; }
    *pscale = mx * (1.0f / 7.0f);
    float inv = mx > 0 ? 7.0f / mx : 0.0f;
    for (int d = 0; d < hd; d += 2) {
        int q0 = (int)lrintf(v[d]   * inv); q0 = q0 > 7 ? 7 : (q0 < -7 ? -7 : q0);
        int q1 = (int)lrintf(v[d+1] * inv); q1 = q1 > 7 ? 7 : (q1 < -7 ? -7 : q1);
        out[d>>1] = (uint8_t)(((q1 & 0xF) << 4) | (q0 & 0xF));
    }
}
typedef struct { const int8_t *q8; const uint8_t *k4; const float *sq, *pscale, *weights;
                 int H, hd, T, s0; float *score; } ds4f_idxsc8r4_task;
#if defined(__ARM_FEATURE_SVE)
static void ds4f_idxsc8r4_worker(void *arg, int tid, int nthr) {
    ds4f_idxsc8r4_task *Tk = (ds4f_idxsc8r4_task *)arg;
    int Tn = Tk->T, H = Tk->H, hd = Tk->hd, s0 = Tk->s0;   /* CP: k4/pscale are LOCAL [0,Tn); score is GLOBAL (s0+t) */
    int per = Tn / nthr, extra = Tn % nthr;
    int t0 = per * tid + (tid < extra ? tid : extra), t1 = t0 + per + (tid < extra ? 1 : 0);
    svbool_t pb = svptrue_b8(); int bf = (int)svcntb();
    int8_t kt8[256];   /* unpacked int8 key (hd <= 256; indexer hd=128) */
    for (int t = t0; t < t1; t++) {
        const uint8_t *k4 = Tk->k4 + (size_t)t * (hd/2);
        for (int b = 0; b < hd/2; b++) { int by = k4[b]; kt8[2*b] = (int8_t)(((by & 0xF) ^ 8) - 8); kt8[2*b+1] = (int8_t)(((by >> 4) ^ 8) - 8); }
        float ps = Tk->pscale[t], acc = 0.f;
        for (int h = 0; h < H; h++) {
            const int8_t *q8 = Tk->q8 + (size_t)h * hd;
            svint32_t a = svdup_s32(0);
            for (int d = 0; d < hd; d += bf) a = svdot_s32(a, svld1_s8(pb, q8 + d), svld1_s8(pb, kt8 + d));
            float dot = Tk->sq[h] * ps * (float)svaddv_s32(svptrue_b32(), a);
            if (dot < 0.f) dot = 0.f;                /* relu */
            acc += dot * Tk->weights[h];
        }
        Tk->score[s0 + t] = acc;
    }
}
#else
static void ds4f_idxsc8r4_worker(void *arg, int tid, int nthr) {
    ds4f_idxsc8r4_task *Tk = (ds4f_idxsc8r4_task *)arg;
    int Tn = Tk->T, H = Tk->H, hd = Tk->hd, s0 = Tk->s0;
    int per = Tn / nthr, extra = Tn % nthr;
    int t0 = per * tid + (tid < extra ? tid : extra), t1 = t0 + per + (tid < extra ? 1 : 0);
    int8_t kt8[256];
    for (int t = t0; t < t1; t++) {
        const uint8_t *k4 = Tk->k4 + (size_t)t * (hd/2);
        for (int b = 0; b < hd/2; b++) { int by = k4[b]; kt8[2*b] = (int8_t)(((by & 0xF) ^ 8) - 8); kt8[2*b+1] = (int8_t)(((by >> 4) ^ 8) - 8); }
        float ps = Tk->pscale[t], acc = 0.f;
        for (int h = 0; h < H; h++) {
            const int8_t *q8 = Tk->q8 + (size_t)h * hd;
            int32_t a = 0;
            for (int d = 0; d < hd; d++) a += (int32_t)q8[d] * (int32_t)kt8[d];
            float dot = Tk->sq[h] * ps * (float)a;
            if (dot < 0.f) dot = 0.f;
            acc += dot * Tk->weights[h];
        }
        Tk->score[s0 + t] = acc;
    }
}
#endif /* __ARM_FEATURE_SVE */

static inline double ds4f_now(void);              /* fwd decl (defined w/ the profiler below) */
static double ds4f_g_tb2scan = 0.0;               /* transfer accumulators: index_step component-only ns */
static double ds4f_g_tb2qproj = 0.0;              /* idx wq_b q-projection */
static double ds4f_g_tb2rope = 0.0;               /* per-head RoPE + rotate + fp4 */
static double ds4f_g_tb2icmp = 0.0;               /* idx compressor (compress_step inside index_step) */
static double ds4f_g_tb2wproj = 0.0;              /* weights_proj */
static double ds4f_g_tb2lcmp = 0.0;               /* layer compressor (top of tb2_prepare) */
static double ds4f_g_tb2topk = 0.0;               /* index_topk top-k selection */

/* ===================== Tier-B2 indexer scoring + top-k =====================
 * model.py Indexer.forward scoring (after q is wq_b-projected, RoPE'd, rotate+fp4'd
 * and the compressor has filled kvc): index_score[t] = sum_h relu(q[h].kvc[t]) *
 * weights[h]. q[H*hd] (one query's heads, contiguous), kvc[T*hd], weights[H]. */
/* DS4F_IDX_INT8 *replacement* mode: the f32 idx_kv scan (ds4f_index_score) runs ONLY when
 * T < DS4F_IDX_F32_SLOTS; for T >= it the resident int8 idx_kv8 svdot scan is used (the
 * `T >= DS4F_IDX_F32_SLOTS` branch in ds4f_index_step). So under idx_int8 the f32 idx_kv needs
 * only this many slots, not full nslot -- cutting it ~2,688 B/pos -> 0 (the idx memory lever:
 * slope 5.93 -> ~3.9 KB/pos, ctx ceiling ~1.12M -> ~1.7M). Keep in sync with that branch. */
#define DS4F_IDX_F32_SLOTS 1
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
/* Select the npick=min(k,thr) highest-scoring positions in [0,thr), tie-broken by lowest
 * index, written to sel[0..npick) in ASCENDING index order (+offset), sel[npick..k)=-1.
 * Default fast path is O(thr*log k + thr); set DS4F_TOPK_NAIVE=1 for the original O(k*thr)
 * reference (the two produce an identical selected set & order -- the validation gate). */
static void ds4f_index_topk(const float *score, int T, int thr, int k, int offset, int *sel) {
    if (thr > T) thr = T;
    int npick = k < thr ? k : thr;
    static int s_naive = -1;
    if (s_naive < 0) { const char *e = getenv("DS4F_TOPK_NAIVE"); s_naive = (e && *e && atoi(e)) ? 1 : 0; }
    if (s_naive) {                                           /* original O(k*thr) selection + O(k^2) sort */
        int  *chosen = (int  *)alloca((size_t)(npick > 0 ? npick : 1) * sizeof(int));
        char *used   = (char *)alloca((size_t)(thr > 0 ? thr : 1));
        for (int t = 0; t < thr; t++) used[t] = 0;
        for (int n = 0; n < npick; n++) {
            int best = -1; float bv = -1e30f;
            for (int t = 0; t < thr; t++) if (!used[t] && score[t] > bv) { bv = score[t]; best = t; }
            used[best] = 1; chosen[n] = best;
        }
        for (int a = 0; a < npick; a++)                      /* sort ascending */
            for (int b = a + 1; b < npick; b++)
                if (chosen[b] < chosen[a]) { int tmp = chosen[a]; chosen[a] = chosen[b]; chosen[b] = tmp; }
        for (int n = 0; n < k; n++) sel[n] = (n < npick) ? (chosen[n] + offset) : -1;
        return;
    }
    /* --- fast path --- */
    if (npick <= 0) { for (int n = 0; n < k; n++) sel[n] = -1; return; }
    if (npick >= thr) {                                      /* select everything in [0,thr) */
        for (int n = 0; n < k; n++) sel[n] = (n < thr) ? (n + offset) : -1;
        return;
    }
    /* min-heap of the npick largest scores -> heap[0] = npick-th largest = threshold tau */
    float *heap = (float *)alloca((size_t)npick * sizeof(float));
    int hn = 0;
    for (int t = 0; t < thr; t++) {
        float v = score[t];
        if (hn < npick) {                                    /* push + sift-up */
            int i = hn++; heap[i] = v;
            while (i > 0) { int p = (i - 1) >> 1; if (heap[p] <= heap[i]) break;
                            float tmp = heap[p]; heap[p] = heap[i]; heap[i] = tmp; i = p; }
        } else if (v > heap[0]) {                            /* replace-min + sift-down */
            heap[0] = v; int i = 0;
            for (;;) { int l = 2*i+1, r = 2*i+2, s = i;
                if (l < npick && heap[l] < heap[s]) s = l;
                if (r < npick && heap[r] < heap[s]) s = r;
                if (s == i) break;
                float tmp = heap[s]; heap[s] = heap[i]; heap[i] = tmp; i = s;
            }
        }
    }
    float tau = heap[0];                                     /* npick-th largest score */
    /* selected set = {t: score>tau} (all) + lowest-index (npick-G) of {t: score==tau}.
     * Collect each group ascending, then merge -> ascending output (no O(k^2) sort). */
    int *gt = (int *)alloca((size_t)npick * sizeof(int));
    int *eq = (int *)alloca((size_t)npick * sizeof(int));
    int ng = 0, ne = 0;
    for (int t = 0; t < thr; t++) {
        float v = score[t];
        if (v > tau) { if (ng < npick) gt[ng++] = t; }
        else if (v == tau) { if (ne < npick) eq[ne++] = t; }
    }
    int need_eq = npick - ng; if (need_eq > ne) need_eq = ne;   /* defensive (FP: should == npick-ng) */
    int BIG = thr + 1, ia = 0, ib = 0, n = 0;
    while (n < npick) {                                       /* merge two ascending lists */
        int va = (ia < ng)      ? gt[ia] : BIG;
        int vb = (ib < need_eq) ? eq[ib] : BIG;
        if (va == BIG && vb == BIG) break;                   /* ran out (NaN/edge) -> pad below */
        if (va <= vb) { sel[n++] = va + offset; ia++; }
        else          { sel[n++] = vb + offset; ib++; }
    }
    for (; n < k; n++) sel[n] = -1;
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

/* Parallel per-head indexer RoPE+rotate+fp4 (the tb2rope sub-phase). Each index head writes a
 * disjoint q_scr[h*hd..] slice and the three ops (rope_apply / rotate_activation /
 * fp4_act_quant_inplace) touch only that slice + read-only rcos/rsin -> splitting heads across
 * the pool is BIT-EXACT to the serial loop (same 2j pattern as decode q-norm). Was 4.68 ms/tok =
 * 5.0% of decode @ctx10240, running fully serial on tid0 between the qproj and weights pool runs. */
typedef struct { float *q_scr; int H, hd, rd, half, start_pos; const float *rcos, *rsin; } ds4f_tb2rope_task;
static void ds4f_tb2rope_worker(void *arg, int tid, int nthr) {
    ds4f_tb2rope_task *T = (ds4f_tb2rope_task *)arg;
    int H = T->H, hd = T->hd, rd = T->rd, half = T->half;
    int h0 = (int)((long)H * tid / nthr), h1 = (int)((long)H * (tid+1) / nthr);
    for (int h = h0; h < h1; h++) {
        float *qh = T->q_scr + (size_t)h * hd;
        ds4f_rope_apply(qh + (hd - rd), T->rcos, T->rsin, T->start_pos, half, 0);
        ds4f_rotate_activation(qh, hd);
        ds4f_fp4_act_quant_inplace(qh, hd, 32);
    }
}

/* Indexer decode step (model.py Indexer.forward, seqlen==1, start_pos>0): project q via wq_b,
 * per-head RoPE(last rd)+rotate+fp4; step the OWN (rotate) compressor (fills idx_kv_cache);
 * weights = weights_proj(x) * (hd^-0.5 * H^-0.5); index_score over idx_kv_cache[:end//ratio];
 * select top-min(k,T) compressed positions (+offset). q_scr[H*hd], score_scr[>=T], sel[k]. */
static void ds4f_cp_merge_topk(const float *, const float *, int, int, int, int *);  /* fwd (defined below) */
static int ds4f_index_step(
    const float *x, int dim, const float *qr, const float *q_pre, int qlora,
    int H, int hd, int rd, int ratio, int start_pos, int offset, int k,
    const void *wq_b, const void *weights_proj, int w_bf16,
    const void *cwkv, const void *cwgate, const float *cape, const uint16_t *cnorm,
    const float *rcos, const float *rsin, float eps,
    float *comp_kv_state, float *comp_score_state, float *idx_kv_cache,
    int8_t *idx_kv8, uint8_t *idx_kv8_4, float *idx_pscale,
    float *q_scr, float *score_scr, int *sel, ds4f_pool *pool,
    int idx_cp, int idx_s0, int idx_s1, void (*ar_cb)(float *, int, void *), void *ar_ctx,
    int ep_rank, int ep_size, float *cp_cand_slot, float *cp_cand_score)
{
    static int s_cp_merge = -1;   /* DS4F_CP_MERGE (default on): candidate-merge gather vs Option-A full-T */
    if (s_cp_merge < 0) { const char *e = getenv("DS4F_CP_MERGE"); s_cp_merge = (e ? atoi(e) : 1); }
    int end_pos = start_pos + 1, half = rd / 2;
    double _tqp0 = ds4f_now();
    if (q_pre) {                                             /* verify prefill: q projection was batched */
        memcpy(q_scr, q_pre, (size_t)H * hd * 4);
    } else if (pool && w_bf16) {                             /* q = wq_b(qr), pooled bf16 */
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
    ds4f_g_tb2qproj += ds4f_now() - _tqp0;
    double _trp0 = ds4f_now();
    static int s_rope_par = -1;
    if (s_rope_par < 0) { const char *e = getenv("DS4F_TB2ROPE_PAR"); s_rope_par = (e && *e) ? atoi(e) : 1; }
    if (pool && s_rope_par) {                                /* per-head RoPE+rotate+fp4, pooled (bit-exact) */
        ds4f_tb2rope_task rt = { q_scr, H, hd, rd, half, start_pos, rcos, rsin };
        ds4f_pool_run(pool, ds4f_tb2rope_worker, &rt);
    } else
    for (int h = 0; h < H; h++) {                            /* RoPE + rotate + fp4 per head (serial ref) */
        float *qh = q_scr + (size_t)h * hd;
        ds4f_rope_apply(qh + (hd - rd), rcos, rsin, start_pos, half, 0);
        ds4f_rotate_activation(qh, hd);
        ds4f_fp4_act_quant_inplace(qh, hd, 32);
    }
    ds4f_g_tb2rope += ds4f_now() - _trp0;
    double _tic0 = ds4f_now();
    float *comp_out = (float *)alloca((size_t)hd * 4);       /* own compressor (rotate=1) */
    if (ds4f_compress_step(x, dim, hd, rd, ratio, start_pos, cwkv, cwgate, w_bf16, cape, cnorm,
                           rcos, rsin, eps, 1, comp_kv_state, comp_score_state, comp_out, pool)) {
        int slot = start_pos / ratio;
        /* idx_int8 replacement: the f32 idx_kv buffer is only DS4F_IDX_F32_SLOTS slots (the f32
         * scan reads it only for T<that); skip the f32 write past it (the int8 idx_kv8 scan
         * serves T>=that). When idx_kv8==NULL (f32 mode) idx_kv is full nslot -> always write. */
        int int_mode = (idx_kv8 || idx_kv8_4);   /* int8 or int4 replacement active */
        if (!int_mode || slot < DS4F_IDX_F32_SLOTS)
            memcpy(idx_kv_cache + (size_t)slot * hd, comp_out, (size_t)hd * 4);
        int iloc = slot;   /* CP: idx_kv8_4/idx_pscale slot-sharded -> owner writes local (slot-idx_s0), else skip */
        if (idx_cp) iloc = (slot >= idx_s0 && slot < idx_s1) ? slot - idx_s0 : -1;
        if (idx_kv8_4)     { if (iloc >= 0) ds4f_idx_quant_pos_i4(comp_out, hd, idx_kv8_4 + (size_t)iloc * (hd/2), &idx_pscale[iloc]); }
        else if (idx_kv8)  ds4f_idx_quant_pos(comp_out, hd, idx_kv8 + (size_t)slot * hd, &idx_pscale[slot]);
    }
    ds4f_g_tb2icmp += ds4f_now() - _tic0;
    float sm = (float)(1.0 / sqrt((double)hd)), wscale = sm * (float)(1.0 / sqrt((double)H));
    float *weights = (float *)alloca((size_t)H * 4);
    double _twp0 = ds4f_now();
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
    ds4f_g_tb2wproj += ds4f_now() - _twp0;
    int T = end_pos / ratio;
    double _scan_t0 = ds4f_now();
    static int s_i8 = -1, s_i4 = -1;
    if (s_i8 < 0) { const char *e = getenv("DS4F_IDX_INT8"); s_i8 = (e && *e && atoi(e)) ? 1 : 0; }
    if (s_i4 < 0) { const char *e = getenv("DS4F_IDX_INT4"); s_i4 = (e && *e && atoi(e)) ? 1 : 0; }
    int do_i4 = (s_i4 && idx_kv8_4 && pool && T >= DS4F_IDX_F32_SLOTS && (hd % 64) == 0);  /* resident int4 scan */
    int do_i8 = (s_i8 && idx_kv8   && pool && T >= DS4F_IDX_F32_SLOTS && (hd % 64) == 0);  /* resident int8 scan */
    int sel_done = 0;   /* CP merge path fills sel directly -> skip the O(T) final topk */
    if (do_i4 || do_i8) {
        int8_t *q8 = (int8_t *)alloca((size_t)H * hd);       /* query stays int8 in both modes */
        float  *sq = (float *)alloca((size_t)H * 4);
        for (int h = 0; h < H; h++) {                        /* per-head q quant (round-to-nearest absmax) */
            const float *qh = q_scr + (size_t)h * hd; float mx = 0.f;
            for (int d = 0; d < hd; d++) { float a = qh[d] < 0 ? -qh[d] : qh[d]; if (a > mx) mx = a; }
            float inv = mx > 0 ? 127.0f / mx : 0.0f; sq[h] = mx * (1.0f / 127.0f);
            for (int d = 0; d < hd; d++) { int v = (int)lrintf(qh[d] * inv); v = v > 127 ? 127 : (v < -127 ? -127 : v); q8[(size_t)h * hd + d] = (int8_t)v; }
        }
        if (do_i4) {
            int merge = idx_cp && s_cp_merge && ar_cb && cp_cand_slot && cp_cand_score && ep_size > 1;
            int hi = (idx_cp && idx_s1 < T) ? idx_s1 : T;
            if (merge) {
                /* MERGE (decode win): scan owned slots -> LOCAL scores [0,nloc); local top-k -> candidates
                 * {global slot, score}; gather N*k candidates (ar_cb-SUM, ~45 KB vs ~12 MB full-T); merge ->
                 * global top-k. Exact to the replicated topk (tools/cpmerge_test). Skips the O(T) final topk. */
                int nloc = hi - idx_s0; if (nloc < 0) nloc = 0;
                ds4f_idxsc8r4_task tk = { q8, idx_kv8_4, sq, idx_pscale, weights, H, hd, nloc, 0, score_scr };
                ds4f_pool_run(pool, ds4f_idxsc8r4_worker, &tk);
                int *lsel = (int *)alloca((size_t)k * sizeof(int));
                ds4f_index_topk(score_scr, nloc, nloc, k, idx_s0, lsel);   /* local idx -> GLOBAL cmp slot */
                int base = ep_rank * k, ncand = ep_size * k;
                for (int i = 0; i < ncand; i++) { cp_cand_slot[i] = 0.f; cp_cand_score[i] = 0.f; }
                for (int j = 0; j < k; j++) { int g = lsel[j];
                    cp_cand_slot[base + j]  = (g >= 0) ? (float)g : -1.f;
                    cp_cand_score[base + j] = (g >= 0) ? score_scr[g - idx_s0] : 0.f; }
                ar_cb(cp_cand_slot, ncand, ar_ctx); ar_cb(cp_cand_score, ncand, ar_ctx);
                ds4f_cp_merge_topk(cp_cand_slot, cp_cand_score, ncand, k, offset, sel);
                sel_done = 1;
            } else {
                /* Option A: scan owned -> GLOBAL score positions, zero rest + ar_cb-SUM full [0,T). */
                int s0 = idx_cp ? idx_s0 : 0, nloc = hi - s0; if (nloc < 0) nloc = 0;
                if (idx_cp) { for (int i = 0; i < T; i++) score_scr[i] = 0.f; }
                ds4f_idxsc8r4_task tk = { q8, idx_kv8_4, sq, idx_pscale, weights, H, hd, nloc, s0, score_scr };
                ds4f_pool_run(pool, ds4f_idxsc8r4_worker, &tk);
                if (idx_cp && ar_cb) ar_cb(score_scr, T, ar_ctx);
            }
        }
        else       { ds4f_idxsc8r_task  tk = { q8, idx_kv8,   sq, idx_pscale, weights, H, hd, T, score_scr };
                     ds4f_pool_run(pool, ds4f_idxsc8r_worker, &tk); }
    } else {
        ds4f_index_score(q_scr, idx_kv_cache, weights, H, hd, T, score_scr, pool);
    }
    ds4f_g_tb2scan += ds4f_now() - _scan_t0;                  /* scan-only sub-timer (folded by tb2_prepare) */
    double _topk_t0 = ds4f_now();
    if (!sel_done) ds4f_index_topk(score_scr, T, T, k, offset, sel);  /* decode: no causal mask (thr=T) */
    ds4f_g_tb2topk += ds4f_now() - _topk_t0;
    return T;
}

/* CP idx-shard merge (decode win): combine N nodes' local top-k candidate lists {slot,score} into the
 * global top-k -- replaces Option-A's full-[0,T) score gather + O(T) topk with an N*k-candidate gather
 * (~45 KB vs ~12 MB @12M) + O(N*k log) sort. EXACT to the replicated topk (same rule: score desc, ties
 * -> lowest slot, ascending output) -- proven 200/200 in tools/cpmerge_test.c. Slots carried as float
 * (exact for slot < 2^24; T<=4M @16M ctx). cslot[i]<0 = padding (shard had <k owned slots). */
typedef struct { float s; int idx; } ds4f_cpcand;
static int ds4f_cpcand_cmp(const void *a, const void *b) {       /* score desc, ties -> lower slot */
    const ds4f_cpcand *x = a, *y = b;
    if (x->s > y->s) return -1;
    if (x->s < y->s) return 1;
    return x->idx - y->idx;
}
static int ds4f_cpcand_cmp_idx(const void *a, const void *b) {   /* slot ascending */
    return ((const ds4f_cpcand *)a)->idx - ((const ds4f_cpcand *)b)->idx;
}
static void ds4f_cp_merge_topk(const float *cslot, const float *cscore, int ncand, int k, int offset, int *sel) {
    ds4f_cpcand *c = (ds4f_cpcand *)alloca((size_t)(ncand > 0 ? ncand : 1) * sizeof(ds4f_cpcand));
    int nc = 0;
    for (int i = 0; i < ncand; i++) if (cslot[i] >= 0.f) { c[nc].s = cscore[i]; c[nc].idx = (int)cslot[i]; nc++; }
    qsort(c, nc, sizeof(ds4f_cpcand), ds4f_cpcand_cmp);          /* top-k by score (ties lower slot) */
    int npick = k < nc ? k : nc;
    qsort(c, npick, sizeof(ds4f_cpcand), ds4f_cpcand_cmp_idx);   /* emit ascending by slot (== index_topk) */
    for (int n = 0; n < k; n++) sel[n] = (n < npick) ? (c[n].idx + offset) : -1;
}

/* ===================== synthetic allocator ===================== */
static inline int ds4f_n_owned(int n_experts, int ep_rank, int ep_size) {
    int c = 0; for (int e = 0; e < n_experts; e++) if (e % ep_size == ep_rank) c++; return c;
}

/* ring = (tierb2 && !int8_kv): sparse layers ring-buffer kv_cache at window_size, so
 * size the per-layer kv term accordingly (else max_pos for all layers). MUST match the
 * kv_slots condition in ds4f_alloc_synth/ds4f_load_real or the arena over/under-shoots. */
/* Set before ds4f_arena_size when the routed experts will be referenced in place
 * rather than repacked into the arena (see ds4f_model.mxfp4_raw). A file-scope
 * flag rather than a parameter so the a64fx runners' calls keep compiling. */
static int ds4f_arena_no_experts = 0;

static size_t ds4f_arena_size(const ds4f_config *c, int ep_rank, int ep_size, int dense_bf16, int ring,
                              const ds4f_runtime_options *opt) {
    size_t pad = 256; /* per-tensor alignment slack */
    ds4f_qtype dq = dense_bf16 ? DS4F_BF16 : DS4F_FP8;
    int no = ds4f_n_owned(c->n_experts, ep_rank, ep_size);
    size_t per_layer = 0;
    per_layer += (size_t)(c->hidden*2 + c->hidden*2 + c->q_lora*2 + c->kv_lora*2) + 4*pad;
    /* MLA dense (FP8 on-demand, or BF16 predequant) */
    per_layer += ds4f_wbytes(dq, c->q_lora, c->hidden) + ds4f_sbytes(dq, c->q_lora, c->hidden) + 2*pad;
    {   int ah0, ah1; ds4f_tp_attn_shard(c->n_heads, ep_rank, ep_size, &ah0, &ah1, opt->tp_attn);  /* wq_b (TP: owned heads) */
        int qbr = (ah1 - ah0) * c->q_head_dim;
        per_layer += ds4f_wbytes(dq, qbr, c->q_lora) + ds4f_sbytes(dq, qbr, c->q_lora) + 2*pad; }
    per_layer += ds4f_wbytes(dq, c->kv_lora, c->hidden) + ds4f_sbytes(dq, c->kv_lora, c->hidden) + 2*pad;
    {   int oir0, oirows; ds4f_tp_oproj_shard(c->o_inter, ep_rank, ep_size, dense_bf16 ? 8 : 128, &oir0, &oirows, opt->tp_oproj);  /* wo_a (TP o_inter shard) */
        per_layer += ds4f_wbytes(dq, oirows, c->hidden) + ds4f_sbytes(dq, oirows, c->hidden) + 2*pad; }
    {   int oir0, oir; ds4f_tp_oproj_shard(c->o_inter, ep_rank, ep_size, dense_bf16 ? 8 : 128, &oir0, &oir, opt->tp_oproj);
        int wob_c = c->o_inter;                               /* wo_b (DS4F_TP_WOB: FP8 o_inter col-shard) */
        if (oir < c->o_inter && !dense_bf16 && opt->tp_wob) wob_c = oir;
        per_layer += ds4f_wbytes(dq, c->hidden, wob_c) + ds4f_sbytes(dq, c->hidden, wob_c) + 2*pad; }
    per_layer += (size_t)c->n_heads*4 + pad;
    /* MoE */
    per_layer += ds4f_wbytes(DS4F_BF16, c->n_experts, c->hidden) + pad;            /* router */
    {   int shr0, shrows; ds4f_tp_shared_shard(c->shared_inter, ep_rank, ep_size, dense_bf16 ? 8 : 128, &shr0, &shrows, opt->tp_shared);  /* sh_w1+sh_w3 (TP col-shard) */
        per_layer += 2*(ds4f_wbytes(dq, shrows, c->hidden) + ds4f_sbytes(dq, shrows, c->hidden)) + 4*pad; }
    {   int sh20, sh2rows; ds4f_tp_shared_out_shard(c->hidden, ep_rank, ep_size, dense_bf16 ? 8 : 128, &sh20, &sh2rows, opt->tp_shared && opt->tp_shared_full);
        per_layer += ds4f_wbytes(dq, sh2rows, c->shared_inter) + ds4f_sbytes(dq, sh2rows, c->shared_inter) + 2*pad;  /* sh_w2 */ }
    size_t per_ex = ds4f_wbytes(DS4F_MXFP4, c->moe_inter, c->hidden) + ds4f_sbytes(DS4F_MXFP4, c->moe_inter, c->hidden)
                  + ds4f_wbytes(DS4F_MXFP4, c->hidden, c->moe_inter) + ds4f_sbytes(DS4F_MXFP4, c->hidden, c->moe_inter)
                  + ds4f_wbytes(DS4F_MXFP4, c->moe_inter, c->hidden) + ds4f_sbytes(DS4F_MXFP4, c->moe_inter, c->hidden) + 6*pad;
    if (!ds4f_arena_no_experts) per_layer += (size_t)no * per_ex;
    {   int hc = c->hc_mult, mix = (2+hc)*hc, hd = hc*c->hidden;
        per_layer += 2*((size_t)mix*hd*4 + (size_t)mix*4 + 3*4) + 6*pad;            /* hc_attn/ffn fn+base+scale */
    }
    /* kv cache (bf16) is summed per-layer below (ring buffer => sparse layers shrink to
     * window_size; not part of the uniform per_layer term). */

    size_t total = per_layer * c->n_layers;
    for (int L = 0; L < c->n_layers; L++) {                                         /* kv cache (bf16), per-layer */
        size_t slots = ring ? (size_t)c->window_size : (size_t)c->max_pos;          /* all layers window under tierb2 */
        total += slots * c->kv_lora * sizeof(uint16_t) + pad;
    }
    {   int er0, erows; ds4f_tp_embed_shard(c->vocab, ep_rank, ep_size, &er0, &erows, opt->tp_embed);   /* embed (TP: vocab-shard) */
        total += ds4f_wbytes(DS4F_BF16, erows, c->hidden) + pad; }
    {   int hr0, hrows; ds4f_tp_head_shard(c->vocab, ep_rank, ep_size, &hr0, &hrows, opt->tp_head); /* head (TP: vocab-shard) */
        total += ds4f_wbytes(DS4F_BF16, hrows, c->hidden) + pad; }
    total += (size_t)c->hidden*2 + pad;                                            /* out_norm */
    {   int hc = c->hc_mult, hd = hc*c->hidden;
        total += (size_t)hc*hd*4 + (size_t)hc*4 + 4 + 3*pad;                        /* hc_head fn+base+scale */
    }
    if (opt->mtp) {   /* MTP block: a FULL (no-TP) dense layer + experts + fusion */
        int no = ds4f_n_owned(c->n_experts, ep_rank, ep_size), hc = c->hc_mult, mix = (2 + hc) * hc;
        size_t mtp = 0;
        mtp += ds4f_wbytes(dq,c->q_lora,c->hidden) + ds4f_sbytes(dq,c->q_lora,c->hidden);
        mtp += ds4f_wbytes(dq,c->n_heads*c->q_head_dim,c->q_lora) + ds4f_sbytes(dq,c->n_heads*c->q_head_dim,c->q_lora);
        mtp += ds4f_wbytes(dq,c->kv_lora,c->hidden) + ds4f_sbytes(dq,c->kv_lora,c->hidden);
        mtp += ds4f_wbytes(dq,c->o_inter,c->hidden) + ds4f_sbytes(dq,c->o_inter,c->hidden);
        mtp += ds4f_wbytes(dq,c->hidden,c->o_inter) + ds4f_sbytes(dq,c->hidden,c->o_inter);
        mtp += ds4f_wbytes(DS4F_BF16,c->n_experts,c->hidden);                                  /* gate */
        mtp += (size_t)no * (2*(ds4f_wbytes(DS4F_MXFP4,c->moe_inter,c->hidden)+ds4f_sbytes(DS4F_MXFP4,c->moe_inter,c->hidden))
                            + ds4f_wbytes(DS4F_MXFP4,c->hidden,c->moe_inter)+ds4f_sbytes(DS4F_MXFP4,c->hidden,c->moe_inter));
        mtp += 2*(ds4f_wbytes(dq,c->hidden,c->hidden)+ds4f_sbytes(dq,c->hidden,c->hidden));    /* e_proj + h_proj */
        mtp += (size_t)2*mix*hc*c->hidden*4 + (size_t)hc*hc*c->hidden*4;                       /* hc_attn/ffn/head fn */
        total += mtp + (size_t)64*1024*1024 + 64*pad;                                          /* norms/base/scale + slack */
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

static int ds4f_int8kv_cal = -1;   /* DS4F_INT8KV_CAL: calibration window (default 256) */
static int ds4f_int8cmp_cal = -1;  /* DS4F_INT8CMP_CAL: calib window in SLOTS (default 64) */
/* DS4F_INT8_KV: allocate the int8 KV store (kv_q, half the bf16 footprint) + the bf16
 * calibration staging buffer + per-channel scale arrays, in-arena. The arena is sized for
 * the bf16 cache (max_pos*KV*2); int8 (max_pos*KV + CAL*KV*2 + 3*KV*4) fits under that with
 * slack, and only touched pages count toward RSS -> the KV footprint halves at long ctx. */
static void ds4f_alloc_int8kv(ds4f_model *m, ds4f_layer *ly, const ds4f_config *cfg) {
    int KV = cfg->kv_lora, CAL = ds4f_int8kv_cal > 0 ? ds4f_int8kv_cal : 256;
    ly->kv_cache  = NULL;
    ly->kv_q      = (int8_t   *)ds4f_bump(m, (size_t)cfg->max_pos * KV, 256);
    ly->kv_calbuf = (uint16_t *)ds4f_bump(m, (size_t)CAL * KV * sizeof(uint16_t), 256);
    ly->kv_scale  = (float    *)ds4f_bump(m, (size_t)KV * 4, 64);
    ly->kv_iscale = (float    *)ds4f_bump(m, (size_t)KV * 4, 64);
    ly->kv_absmax = (float    *)ds4f_bump(m, (size_t)KV * 4, 64);
    ly->kv_caln = 0; ly->kv_frozen = 0;
    for (int d = 0; d < KV; d++) ly->kv_absmax[d] = 0.f;
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
    ds4f_tensor t; t.type = type; t.rows = rows; t.cols = cols; t.gpu_id = -1;
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
#define DS4F_TB2_ALLOC(bytes, align) ds4f_mem_alloc(m->mem, (bytes), (align), 1)
    ds4f_config *c = &m->cfg;
    int C = c->hidden, KV = c->kv_lora, ihd = c->index_head_dim, iH = c->index_n_heads;
    int qlora = c->q_lora, np = c->max_pos;
    /* ds4f_index_topk ALWAYS writes sel[0..index_topk) (real picks + -1 pad), and
     * ds4f_index_score writes score[0..T) with T<=np/ratio. When max_pos < index_topk
     * (e.g. short gen runs, np=216 < 512), an np-sized sel buffer overflows by
     * (index_topk-np) ints of 0xFFFFFFFF (=NaN bit pattern) into adjacent heap, which
     * (running after tb2_prepare's compressor write) corrupts the compressor ring
     * state -> all-NaN cmp_kv from the 3rd compressed block on. Size to hold both. */
    int nsel_cap = np > c->index_topk ? np : c->index_topk;
    /* model-level per-token scratch (allocated once) */
    m->s_cmp_out   = (float *)DS4F_TB2_ALLOC((size_t)KV*4, 256);
    m->s_idx_q     = (float *)DS4F_TB2_ALLOC((size_t)iH*ihd*4, 256);
    m->s_idx_score = (float *)DS4F_TB2_ALLOC((size_t)nsel_cap*4, 256);
    m->s_tb2_sel   = (int   *)DS4F_TB2_ALLOC((size_t)nsel_cap*4, 256);
    m->s_cmp_gather = (float *)DS4F_TB2_ALLOC((size_t)c->index_topk*KV*4, 256);  /* CP: gathered selected cmp latents
        (only ns<=index_topk slots ever written -- must NOT scale with nsel_cap=max(max_pos,topk): 16 GB @ 8M ctx) */
    { int eps = m->ep_size > 0 ? m->ep_size : 1;               /* CP idx-merge candidate gather buffers */
      m->s_cp_cand_slot  = (float *)DS4F_TB2_ALLOC(((size_t)eps*c->index_topk*4 + 255) & ~255ull, 256);
      m->s_cp_cand_score = (float *)DS4F_TB2_ALLOC(((size_t)eps*c->index_topk*4 + 255) & ~255ull, 256); }
    for (int L = 0; L < c->n_layers; L++) {
        int ratio = c->compress_ratios[L];
        if (ratio == 0) continue;
        ds4f_layer *ly = &m->layers[L];
        int overlap = (ratio == 4), coff = overlap ? 2 : 1, W = coff*KV;
        int nslot = np / ratio;
        ly->cmp_wkv   = (uint16_t *)DS4F_TB2_ALLOC((size_t)W*C*2, 256);
        ly->cmp_wgate = (uint16_t *)DS4F_TB2_ALLOC((size_t)W*C*2, 256);
        ly->cmp_ape   = (float *)DS4F_TB2_ALLOC((size_t)ratio*W*4, 256);
        ly->cmp_norm  = (uint16_t *)DS4F_TB2_ALLOC(((size_t)KV*2 + 63) & ~63u, 64);
        ly->cmp_kv_state    = (float *)DS4F_TB2_ALLOC((size_t)coff*ratio*W*4, 256);
        ly->cmp_score_state = (float *)DS4F_TB2_ALLOC((size_t)coff*ratio*W*4, 256);
        if (m->int8_cmp) {       /* int8 compressed-latent store (1/4 the f32 physical) + S5 calbuf/scales */
            int CAL = ds4f_int8cmp_cal > 0 ? ds4f_int8cmp_cal : 64;
            if (CAL > nslot) CAL = nslot > 0 ? nslot : 1;
            ly->cmp_kv     = NULL;
            if (m->int4_cmp) {  /* int4: half the bytes (2 nibbles/byte) -- the dominant ctx-cache halved */
                int nslot_q = nslot;
                /* DS4F_CP (default) = the validated gather (cmp replicated). DS4F_CP_SHARD adds the
                 * slot-sharding (the memory win): cmp_q4 -> [0,CAL) replicated + [cp_t0,cp_t1) tail.
                 * BYTE-IDENTICAL to CP-off (calibration replicated => same quant; gather reassembles). */
                if (m->cp && ratio == 4 && getenv("DS4F_CP_SHARD") && atoi(getenv("DS4F_CP_SHARD"))) {
                    int tail = nslot - CAL > 0 ? nslot - CAL : 0, t0, t1;
                    ds4f_cp_slot_shard(tail, m->ep_rank, m->ep_size, &t0, &t1, m->cp);
                    ly->cp_on = 1; ly->cp_t0 = CAL + t0; ly->cp_t1 = CAL + t1; nslot_q = CAL + (t1 - t0);
                }
                ly->cmp_q4 = (uint8_t *)DS4F_TB2_ALLOC(((size_t)nslot_q*(KV/2) + 255) & ~255ull, 256);
                ly->cp_nslot = nslot_q;   /* cmp_q4 slot capacity (DEBUG bounds guards) */
            } else
                ly->cmp_q  = (int8_t *)DS4F_TB2_ALLOC(((size_t)nslot*KV + 255) & ~255ull, 256);
            ly->cmp_calbuf = (uint16_t *)DS4F_TB2_ALLOC((size_t)CAL*KV*2, 256);
            ly->cmp_scale  = (float *)DS4F_TB2_ALLOC((size_t)KV*4, 64);
            ly->cmp_iscale = (float *)DS4F_TB2_ALLOC((size_t)KV*4, 64);
            ly->cmp_absmax = (float *)DS4F_TB2_ALLOC((size_t)KV*4, 64);
            ly->cmp_caln = 0; ly->cmp_frozen = 0;
            for (int d = 0; d < KV; d++) ly->cmp_absmax[d] = 0.f;
        } else {
            ly->cmp_kv = (float *)DS4F_TB2_ALLOC((size_t)nslot*KV*4, 256);
        }
        ds4f_compress_state_reset(ly->cmp_kv_state, ly->cmp_score_state, ratio, KV);
        if (fill) {
            ds4f_tensor t1 = { ly->cmp_wkv,   NULL, DS4F_BF16, W, C, -1 };      ds4f_fill(m, t1);
            ds4f_tensor t2 = { ly->cmp_wgate, NULL, DS4F_BF16, W, C, -1 };      ds4f_fill(m, t2);
            ds4f_tensor t3 = { ly->cmp_ape,   NULL, DS4F_F32, ratio, W, -1 };   ds4f_fill(m, t3);
            ds4f_tensor t4 = { ly->cmp_norm,  NULL, DS4F_BF16, 1, KV, -1 };     ds4f_fill(m, t4);
        }
        if (ratio == 4) {                                       /* indexer (CSA only) */
            int icoff = 2, iW = icoff*ihd;                      /* index ratio==4 => overlap */
            ly->idx_wq_b  = (uint16_t *)DS4F_TB2_ALLOC((size_t)iH*ihd*qlora*2, 256);
            ly->idx_wproj = (uint16_t *)DS4F_TB2_ALLOC((size_t)iH*C*2, 256);
            ly->idx_cmp_wkv   = (uint16_t *)DS4F_TB2_ALLOC((size_t)iW*C*2, 256);
            ly->idx_cmp_wgate = (uint16_t *)DS4F_TB2_ALLOC((size_t)iW*C*2, 256);
            ly->idx_cmp_ape   = (float *)DS4F_TB2_ALLOC((size_t)ratio*iW*4, 256);
            ly->idx_cmp_norm  = (uint16_t *)DS4F_TB2_ALLOC(((size_t)ihd*2 + 63) & ~63u, 64);
            ly->idx_cmp_kv_state    = (float *)DS4F_TB2_ALLOC((size_t)icoff*ratio*iW*4, 256);
            ly->idx_cmp_score_state = (float *)DS4F_TB2_ALLOC((size_t)icoff*ratio*iW*4, 256);
            { static int s_i8a = -1, s_i4a = -1;   /* DS4F_IDX_INT8/INT4: int store REPLACES f32 idx_kv */
              if (s_i8a < 0) { const char *e = getenv("DS4F_IDX_INT8"); s_i8a = (e && *e && atoi(e)) ? 1 : 0; }
              if (s_i4a < 0) { const char *e = getenv("DS4F_IDX_INT4"); s_i4a = (e && *e && atoi(e)) ? 1 : 0; }
              int use_i4 = s_i4a && (m->pool != NULL);   /* int4 idx_kv8_4: half of int8 (672->336 B/pos) */
              int use_i8 = (s_i8a || s_i4a) && (m->pool != NULL);   /* either => int-replacement infra */
              /* f32 idx_kv: full nslot in f32 mode; only DS4F_IDX_F32_SLOTS slots under int replacement
               * (the f32 scan reads it only for T<that; T>=that uses idx_kv8/idx_kv8_4) -> 2,688 B/pos -> 0. */
              int idxf32 = use_i8 ? (nslot < DS4F_IDX_F32_SLOTS ? nslot : DS4F_IDX_F32_SLOTS) : nslot;
              ly->idx_kv = (float *)DS4F_TB2_ALLOC((size_t)idxf32*ihd*4, 256);
              if (use_i8) {
                int insl = nslot;   /* DS4F_CP_IDX: slot-shard idx_kv8_4 (per-slot scale, clean [0,nslot) shard, no CAL split) */
                if (use_i4 && m->cp && ratio == 4 && getenv("DS4F_CP_IDX") && atoi(getenv("DS4F_CP_IDX"))) {
                    int a0, a1; ds4f_cp_slot_shard(nslot, m->ep_rank, m->ep_size, &a0, &a1, m->cp);
                    ly->idx_cp_on = 1; ly->idx_cp_s0 = a0; ly->idx_cp_s1 = a1; insl = a1 - a0;
                }
                ly->idx_cp_nslot = insl;
                if (use_i4) ly->idx_kv8_4 = (uint8_t *)DS4F_TB2_ALLOC(((size_t)insl*(ihd/2) + 255) & ~255ull, 256);
                else        ly->idx_kv8   = (int8_t  *)DS4F_TB2_ALLOC(((size_t)nslot*ihd     + 255) & ~255ull, 256);
                ly->idx_pscale = (float  *)DS4F_TB2_ALLOC(((size_t)(use_i4?insl:nslot)*4 + 255) & ~255ull, 256);
              } else { ly->idx_kv8 = NULL; ly->idx_kv8_4 = NULL; ly->idx_pscale = NULL; } }
            ds4f_compress_state_reset(ly->idx_cmp_kv_state, ly->idx_cmp_score_state, ratio, ihd);
            if (fill) {
                ds4f_tensor u1 = { ly->idx_wq_b,      NULL, DS4F_BF16, iH*ihd, qlora, -1 }; ds4f_fill(m, u1);
                ds4f_tensor u2 = { ly->idx_wproj,     NULL, DS4F_BF16, iH, C, -1 };         ds4f_fill(m, u2);
                ds4f_tensor u3 = { ly->idx_cmp_wkv,   NULL, DS4F_BF16, iW, C, -1 };         ds4f_fill(m, u3);
                ds4f_tensor u4 = { ly->idx_cmp_wgate, NULL, DS4F_BF16, iW, C, -1 };         ds4f_fill(m, u4);
                ds4f_tensor u5 = { ly->idx_cmp_ape,   NULL, DS4F_F32, ratio, iW, -1 };     ds4f_fill(m, u5);
                ds4f_tensor u6 = { ly->idx_cmp_norm,  NULL, DS4F_BF16, 1, ihd, -1 };       ds4f_fill(m, u6);
            }
        }
    }
#undef DS4F_TB2_ALLOC
}

static ds4f_model *ds4f_alloc_synth_opts(const ds4f_runtime_options *opt) {
    ds4f_config cfg = opt->cfg;
    int ep_rank = opt->ep_rank, ep_size = opt->ep_size;
    int n_threads = opt->n_threads, n_cmgs = opt->n_cmgs;
    ds4f_mem_pool *mem = ds4f_mem_pool_create();
    ds4f_model *m = mem ? (ds4f_model *)ds4f_mem_calloc(mem, 1, sizeof(*m), 64) : NULL;
    if (!m) { ds4f_mem_pool_destroy(mem); return NULL; }
    m->mem = mem;
    m->cfg = cfg; m->ep_rank = ep_rank; m->ep_size = ep_size;
    m->n_threads = n_threads; m->n_cmgs = n_cmgs;
    m->gpu_exact_prefill = opt->hip_exact_prefill ? 1 : 0;
    m->mxfp4_w4a8 = opt->mxfp4_w4a8;
    { const char *e = getenv("DS4F_ROUTE_TELEMETRY");
      m->route_telemetry = opt->hip_expert_cache_mb != 0 || opt->hip_expert_cache_stats ||
                           (e && *e && atoi(e) != 0); }
    ds4f_init_fp8_e4m3_lut(m->fp8_lut);
    /* Dense default is FP8 on-demand (lean ~21.6 GB/node, safe to 128K ctx).
     * DS4F_FP8_BF16=1 predequants the replicated dense to BF16 (+6 GB, faster,
     * intended ≤8K ctx). The pv pair-interleaved layout is byte-identical and
     * strictly faster at zero memory cost, so it AUTO-ENABLES whenever predequant
     * is on; DS4F_BF16_PV=0 is the explicit escape, =1 forces pv even in FP8 mode
     * (only speeds the always-bf16 head+router then). Empty env string == unset. */
    {   int pre = opt->dense_bf16 ? 1 : 0;
        m->bf16_pv = opt->bf16_pv >= 0 ? (opt->bf16_pv ? 1 : 0) : pre;
        m->dense_qt = pre ? (m->bf16_pv ? DS4F_BF16_PV : DS4F_BF16) : DS4F_FP8;
        m->bf16_mv_qt = m->bf16_pv ? DS4F_BF16_PV : DS4F_BF16;
        /* DS4F_DENSE_MXFP4=1: route the replicated dense (MLA + shared) through the
         * MXFP4 split kernel (0.53 B/elem vs FP8 1 / BF16 2). Synthetic harness =>
         * accuracy-free; tests whether the 2-4x byte cut beats the nibble-unpack
         * cost at M=1. Overrides FP8/BF16; head+router stay bf16 (bf16_mv_qt). All
         * dense K are %32 (4096/1024/512/8192/2048) so the split layout is valid. */
        if (opt->dense_mxfp4) m->dense_qt = DS4F_MXFP4;
        /* DS4F_Q8_DENSE=1: repack the dominant dense to int8 W8A8 after fill/load
         * (svdot prefill). Only meaningful with a bf16-pv source; ignored (warned
         * at repack time) otherwise. Head+router stay bf16 (bf16_mv_qt). */
        m->q8_dense = opt->q8_dense ? 1 : 0; }
    {   m->fp8_magic = opt->fp8_magic ? 1 : 0; }
#if !defined(__ARM_FEATURE_SVE)
    m->fp8_magic = 0;      /* the register magic decode is an SVE-only path */
#endif
    {   m->mxfp4_gemm_tile = opt->mxfp4_gemm_tile; }
    {   m->sparse = opt->sparse ? 1 : 0; }
    {   m->mhc = opt->mhc ? 1 : 0; }
    {   m->exact = opt->exact ? 1 : 0; }
    {   m->tierb2 = opt->tierb2 ? 1 : 0; }
    if (m->tierb2) m->exact = 1;   /* Tier-B2 reuses the exact q-norm/RoPE/window path */
    {   m->int8_kv = opt->int8_kv ? 1 : 0; }
    {   ds4f_int8kv_cal = opt->int8kv_cal > 0 ? opt->int8kv_cal : 256;
        if (ds4f_int8kv_cal < 1) ds4f_int8kv_cal = 1; }
    if (m->int8_kv) m->exact = 1;   /* int8 KV uses the exact streaming decode path */
    {   m->int8_cmp = opt->int8_cmp ? 1 : 0; }
    {   m->int4_cmp = opt->int4_cmp ? 1 : 0;
        if (m->int4_cmp) m->int8_cmp = 1; }  /* int4 cmp implies the int8_cmp infra (cmp_q4 sub-mode) */
    {   ds4f_int8cmp_cal = opt->int8cmp_cal > 0 ? opt->int8cmp_cal : 64;
        if (ds4f_int8cmp_cal < 1) ds4f_int8cmp_cal = 1; }
    if (m->int8_cmp) m->exact = 1;  /* int8 cmp uses the exact streaming tierb2 path */
    m->cp = opt->cp ? 1 : 0;
    ds4f_arena_no_experts = 0;   /* synth always materializes experts in the arena */

    m->pool = ds4f_pool_start_ex(mem, n_threads, n_cmgs);

    m->arena_sz = ds4f_arena_size(&cfg, ep_rank, ep_size,
                                  m->dense_qt == DS4F_BF16 || m->dense_qt == DS4F_BF16_PV,
                                  m->tierb2 && !m->int8_kv, opt);
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
    int er0, erows; ds4f_tp_embed_shard(cfg.vocab, ep_rank, ep_size, &er0, &erows, opt->tp_embed); m->emb_r0 = er0; m->emb_rows = erows;  /* TP: vocab-shard */
    ds4f_tensor embed = ds4f_new_tensor(m, DS4F_BF16, erows, C); m->embed = (uint16_t *)embed.w; /* flat gather */
    {   int hr0, hrows; ds4f_tp_head_shard(cfg.vocab, ep_rank, ep_size, &hr0, &hrows, opt->tp_head);  /* TP: vocab-shard */
        m->head = ds4f_new_tensor(m, m->bf16_mv_qt, hrows, C); m->head_r0 = hr0; }     /* matvec'd -> pv when enabled */
    ds4f_tp_shared_shard(cfg.shared_inter, ep_rank, ep_size, m->dense_qt == DS4F_FP8 ? 128 : 8, &m->sh_r0, &m->sh_rows, opt->tp_shared);
    ds4f_tp_shared_out_shard(C, ep_rank, ep_size, m->dense_qt == DS4F_FP8 ? 128 : 8, &m->sh2_r0, &m->sh2_rows, opt->tp_shared && opt->tp_shared_full);
    ds4f_tp_attn_shard(cfg.n_heads, ep_rank, ep_size, &m->attn_h0, &m->attn_h1, opt->tp_attn);  /* DS4F_TP_ATTN: head range */
    ds4f_tp_oproj_shard(cfg.o_inter, ep_rank, ep_size, m->dense_qt == DS4F_FP8 ? 128 : 8, &m->oi0, &m->oi_rows, opt->tp_oproj);  /* DS4F_TP_OPROJ: wo_a o_inter shard */
    {   int hc = cfg.hc_mult, hd = hc*C;
        m->hc_head_fn    = (float *)ds4f_bump(m, (size_t)hc*hd*4, 256);
        m->hc_head_base  = (float *)ds4f_bump(m, (size_t)hc*4, 64);
        m->hc_head_scale = (float *)ds4f_bump(m, (size_t)4, 64); }

    m->layers = (ds4f_layer *)ds4f_mem_calloc(m->mem, (size_t)cfg.n_layers, sizeof(ds4f_layer), 64);
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
        ly->wo_a = ds4f_new_tensor(m, dq, m->oi_rows, C);       /* TP: o_inter row-shard */
        {   /* wo_b: FP8 o_inter col-shard (pairs w/ TP_OPROJ) */
            int wob_s = (m->oi_rows < cfg.o_inter) && dq == DS4F_FP8 && opt->tp_wob;
            ly->wo_b = ds4f_new_tensor(m, dq, C, wob_s ? m->oi_rows : cfg.o_inter); }
        ly->attn_sink = (float *)ds4f_bump(m, (size_t)cfg.n_heads*4, 64);
        ly->gate = ds4f_new_tensor(m, m->bf16_mv_qt, cfg.n_experts, C); /* router matvec -> pv when enabled */
        ly->sh_w1 = ds4f_new_tensor(m, dq, m->sh_rows, C);       /* TP: col-shard shared_inter */
        ly->sh_w3 = ds4f_new_tensor(m, dq, m->sh_rows, C);
        ly->sh_w2 = ds4f_new_tensor(m, dq, m->sh2_rows, cfg.shared_inter); /* TP_SHARED_FULL: hidden-row shard */
        ly->ex_w1 = (ds4f_tensor *)ds4f_mem_calloc(m->mem, (size_t)no, sizeof(ds4f_tensor), 64);
        ly->ex_w2 = (ds4f_tensor *)ds4f_mem_calloc(m->mem, (size_t)no, sizeof(ds4f_tensor), 64);
        ly->ex_w3 = (ds4f_tensor *)ds4f_mem_calloc(m->mem, (size_t)no, sizeof(ds4f_tensor), 64);
        ly->owned_eid = (int *)ds4f_mem_calloc(m->mem, (size_t)no, sizeof(int), 64);
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
        /* kv_cache window ring buffer (long-ctx memory lever; the dominant ctx-scaling cost).
         * Under tierb2 EVERY read of kv_cache is window-only: sparse (ratio!=0) layers use
         * ds4f_attn_tb2_worker (window term + cmp_kv for long range); dense (ratio==0) layers
         * use ds4f_attn_exact_worker (pure sliding window + sink). Neither reads kv_cache past
         * the last window_size positions, and prefill is token-at-a-time (batched prefill is
         * forced off under tierb2/MHC), so ALL layers need only window_size slots -- older
         * history lives in cmp_kv. Caps kv_cache to a constant ~window_size*KV*43 regardless
         * of ctx. Gated on m->tierb2: the non-tierb2 synthetic path (ds4f_attn_worker full
         * read, or batched-prefill ds4f_attn_pf_task needing all M positions) needs full
         * max_pos; int8_kv keeps max_pos (kv_q ring deferred). All kv_cache indexing is
         * (idx % kv_slots), a no-op when kv_slots==max_pos (bit-exact). */
        ly->kv_slots = (!m->int8_kv && m->tierb2) ? cfg.window_size : cfg.max_pos;
        if (m->int8_kv) ds4f_alloc_int8kv(m, ly, &cfg);            /* int8 KV store (half) + calbuf + scales */
        else ly->kv_cache = (uint16_t *)ds4f_bump(m, (size_t)ly->kv_slots*cfg.kv_lora*sizeof(uint16_t), 256);
    }

    /* parallel first-touch fill of all quantized tensors (compute-thread ==
     * touch-thread, so each row block lands on the CMG that later reads it) */
    ds4f_tensor onrm = { m->out_norm, NULL, DS4F_BF16, 1, C, -1 }; ds4f_fill(m, onrm);
    ds4f_fill(m, embed); ds4f_fill(m, m->head);
    {   int hc = cfg.hc_mult, hd = hc*C;
        ds4f_tensor hf = { m->hc_head_fn, NULL, DS4F_F32, hc, hd, -1 }; ds4f_fill(m, hf);
        ds4f_hc_fill_meta(m->hc_head_base, hc, m->hc_head_scale, 1, 4096); }
    for (int L = 0; L < cfg.n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        ds4f_tensor an = { ly->attn_norm, NULL, DS4F_BF16, 1, C, -1 }; ds4f_fill(m, an);
        ds4f_tensor fn = { ly->ffn_norm,  NULL, DS4F_BF16, 1, C, -1 }; ds4f_fill(m, fn);
        ds4f_tensor qn = { ly->q_norm,    NULL, DS4F_BF16, 1, cfg.q_lora, -1 }; ds4f_fill(m, qn);
        ds4f_tensor kn = { ly->kv_norm,   NULL, DS4F_BF16, 1, cfg.kv_lora, -1 }; ds4f_fill(m, kn);
        ds4f_fill(m, ly->wq_a); ds4f_fill(m, ly->wq_b); ds4f_fill(m, ly->wkv);
        ds4f_fill(m, ly->wo_a); ds4f_fill(m, ly->wo_b);
        ds4f_fill(m, ly->gate); ds4f_fill(m, ly->sh_w1); ds4f_fill(m, ly->sh_w2); ds4f_fill(m, ly->sh_w3);
        for (int s = 0; s < no; s++) { ds4f_fill(m, ly->ex_w1[s]); ds4f_fill(m, ly->ex_w2[s]); ds4f_fill(m, ly->ex_w3[s]); }
        for (int h = 0; h < cfg.n_heads; h++) ly->attn_sink[h] = -2.0f;          /* mild sink */
        {   int hc = cfg.hc_mult, mix = (2+hc)*hc, hd = hc*C;
            ds4f_tensor af = { ly->hc_attn_fn, NULL, DS4F_F32, mix, hd, -1 }; ds4f_fill(m, af);
            ds4f_tensor ff = { ly->hc_ffn_fn,  NULL, DS4F_F32, mix, hd, -1 }; ds4f_fill(m, ff);
            ds4f_hc_fill_meta(ly->hc_attn_base, mix, ly->hc_attn_scale, 3, L*2);
            ds4f_hc_fill_meta(ly->hc_ffn_base,  mix, ly->hc_ffn_scale,  3, L*2+1); }
    }

    /* scratch */
    int H = cfg.n_heads*cfg.q_head_dim;
    m->s_hn    = (float *)ds4f_mem_alloc(m->mem, (size_t)C*4, 256, 1);
    m->s_qlat  = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.q_lora*4, 256, 1);
    m->s_q     = (float *)ds4f_mem_alloc(m->mem, (size_t)H*4, 256, 1);
    m->s_kvlat = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.kv_lora*4, 256, 1);
    m->s_attn  = (float *)ds4f_mem_alloc(m->mem, (size_t)H*4, 256, 1);
    m->s_oin   = (float *)ds4f_mem_alloc(m->mem, (size_t)C*4, 256, 1);
    m->s_o1    = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.o_inter*4, 256, 1);
    m->s_o     = (float *)ds4f_mem_alloc(m->mem, (size_t)C*4, 256, 1);
    m->s_h2    = (float *)ds4f_mem_alloc(m->mem, (size_t)C*4, 256, 1);
    m->s_router= (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.n_experts*4, 256, 1);
    m->s_shg   = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.shared_inter*4, 256, 1);
    m->s_shu   = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.shared_inter*4, 256, 1);
    m->s_exg   = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.moe_inter*4, 256, 1);
    m->s_exu   = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.moe_inter*4, 256, 1);
    m->s_exb_g = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.n_active * cfg.moe_inter * 4, 256, 1);
    m->s_exb_u = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.n_active * cfg.moe_inter * 4, 256, 1);
    m->s_exb_o = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.n_active * C * 4, 256, 1);
    m->s_moe   = (float *)ds4f_mem_alloc(m->mem, (size_t)C*4, 256, 1);
    m->s_route = (float *)ds4f_mem_alloc(m->mem, (size_t)C*4, 256, 1);
    m->s_logits= (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.vocab*4, 256, 1);
    /* sparse-indexer scratch: block scores reuse as the selected-score buffer,
     * so the per-thread stride must cover both the worst-case block count
     * (ceil(nP/R), R>=1 => up to max_pos) and index_topk selected positions. */
    m->idx_blk_stride = cfg.max_pos > cfg.index_topk ? cfg.max_pos : cfg.index_topk;
    m->s_idx_scores = (float *)ds4f_mem_alloc(m->mem, (size_t)n_threads * m->idx_blk_stride * 4, 256, 1);
    m->s_idx_sel    = (int   *)ds4f_mem_alloc(m->mem, (size_t)n_threads * cfg.index_topk * 4, 256, 1);
    /* mHC 4-stream scratch (only used when m->mhc) */
    m->s_x4    = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.hc_mult*C*4, 256, 1);
    m->s_resid = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.hc_mult*C*4, 256, 1);
    m->s_xc    = (float *)ds4f_mem_alloc(m->mem, (size_t)C*4, 256, 1);
    ds4f_build_freqs(m);   /* RoPE/YaRN tables (only when exact) */
    if (m->tierb2) ds4f_alloc_tb2(m, 1);   /* off-arena compressor/indexer (synth fill) */
    ds4f_q8_promote_dense(m);   /* DS4F_Q8_DENSE: dense bf16-pv -> int8 W8A8 (no-op if off) */
    return m;
}

static ds4f_runtime_options ds4f_runtime_options_debug_env(ds4f_config cfg,
                                                            const char *stage_dir,
                                                            int ep_rank, int ep_size,
                                                            int n_threads, int n_cmgs) {
    ds4f_runtime_options o;
    memset(&o, 0, sizeof(o));
    o.cfg = cfg; o.ep_rank = ep_rank; o.ep_size = ep_size;
    o.n_threads = n_threads; o.n_cmgs = n_cmgs; o.bf16_pv = -1;
    o.int8kv_cal = 256; o.int8cmp_cal = 64; o.debug_env = 1;
    if (stage_dir) snprintf(o.stage_dir, sizeof(o.stage_dir), "%s", stage_dir);
    { const char *e = getenv("DS4F_TOKENIZER"); if (e && *e) snprintf(o.tokenizer, sizeof(o.tokenizer), "%s", e); }
    { const char *e = getenv("DS4F_TOKENIZER_PY"); if (e && *e) snprintf(o.tokenizer_py, sizeof(o.tokenizer_py), "%s", e); }
    /* Legacy DS4F_MODEL selects the config in --debug-env mode.  Only apply it
     * when explicitly set: the caller's cfg may carry non-default dimensions
     * (e.g. ds4f_alloc_synth() with n_layers=1) that ds4f_config_from_env()
     * would clobber with the full 43-layer default. */
    { const char *m = getenv("DS4F_MODEL");
      if (m && *m) o.cfg = ds4f_config_from_env();
      else { const char *x = getenv("DS4F_EXPERTS");
             if (x && *x && strcmp(x, "q8pv") == 0) o.cfg.expert_qt = DS4F_Q8_PV; } }
    { const char *e = getenv("DS4F_FP8_BF16"); o.dense_bf16 = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_BF16_PV"); o.bf16_pv = e && *e ? atoi(e) : -1; }
    { const char *e = getenv("DS4F_DENSE_MXFP4"); o.dense_mxfp4 = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_MXFP4_W4A8"); o.mxfp4_w4a8 = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_Q8_DENSE"); o.q8_dense = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_FP8_MAGIC"); o.fp8_magic = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_MXFP4_GEMM_TILE"); o.mxfp4_gemm_tile = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_SHARED_BF16"); o.hip_shared_bf16 = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_SHARED_BF16_LAYERS"); o.hip_shared_bf16_layers = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_SHARED_FP16"); o.hip_shared_fp16 = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_SHARED_FP16_LAYERS"); o.hip_shared_fp16_layers = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_ORDERED_WKV_LAYERS"); o.hip_ordered_wkv_layers = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_ORDERED_FP8_LAYERS"); o.hip_ordered_fp8_layers = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_MXFP4_WIDEN_LAYERS"); o.hip_mxfp4_widen_layers = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_MXFP4_RESIDENT_LAYERS"); o.hip_mxfp4_resident_layers = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_MXFP4_RESIDENT_AUTO"); o.hip_mxfp4_resident_auto = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_VRAM_RESERVE_MB"); o.hip_vram_reserve_mb = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_MXFP4_STREAM_RAW"); o.hip_mxfp4_stream_raw = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_EXPERT_CACHE_MB");
      o.hip_expert_cache_mb = e && *e ? (strcmp(e, "auto") == 0 ? -1 : atoi(e)) : 0; }
    { const char *e = getenv("DS4F_HIP_EXPERT_CACHE_STATS"); o.hip_expert_cache_stats = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_PREFILL_ATTN"); o.hip_prefill_attn = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_HIP_EXACT_PREFILL"); o.hip_exact_prefill = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_SPARSE"); o.sparse = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_MHC"); o.mhc = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_EXACT"); o.exact = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_TIERB2"); o.tierb2 = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_INT8_KV"); o.int8_kv = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_INT8_CMP"); o.int8_cmp = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_INT4_CMP"); o.int4_cmp = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_TP_HEAD"); o.tp_head = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_TP_SHARED"); o.tp_shared = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_TP_SHARED_FULL"); o.tp_shared_full = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_TP_ATTN"); o.tp_attn = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_TP_OPROJ"); o.tp_oproj = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_TP_EMBED"); o.tp_embed = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_TP_WOB"); o.tp_wob = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_MTP"); o.mtp = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_EXPERT_RESIDENT"); o.expert_resident = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_INT8KV_CAL"); if (e && *e) o.int8kv_cal = atoi(e); }
    { const char *e = getenv("DS4F_INT8CMP_CAL"); if (e && *e) o.int8cmp_cal = atoi(e); }
    { const char *e = getenv("DS4F_CP"); o.cp = e && *e ? atoi(e) : 0; }
    { const char *e = getenv("DS4F_ZEROCOPY_EXPERTS"); o.zero_copy_experts = e ? atoi(e) : 1; }
    { const char *e = getenv("DS4F_LOAD_DROP_BLOB"); o.load_drop_blob = e && *e ? atoi(e) : 0; }
    return o;
}

/* One thread per PHYSICAL core (no SMT siblings) for the spin-wait pool.  SMT
 * siblings contend on the same execution units and add run-to-run variance, so
 * the default is the physical core count rather than nproc.  Counts unique
 * "core id" values from /proc/cpuinfo; falls back to online logical CPUs. */
static int ds4f_physical_cores(void) {
    static int cached = -1;
    if (cached > 0) return cached;
    int bits[2048] = {0}, n = 0;
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f))
            if (strncmp(line, "core id", 7) == 0) {
                const char *v = strchr(line, ':');
                if (v) {
                    int cid = atoi(v + 1);
                    if (cid >= 0 && cid < 2048 * 32 &&
                        !(bits[cid >> 5] & (1u << (cid & 31)))) {
                        bits[cid >> 5] |= 1u << (cid & 31);
                        n++;
                    }
                }
            }
        fclose(f);
        if (n > 0) { cached = n; return n; }
    }
    long l = sysconf(_SC_NPROCESSORS_ONLN);
    cached = l > 0 ? (int)l : 16;
    return cached;
}

static void ds4f_runtime_options_init(ds4f_runtime_options *o) {
    memset(o, 0, sizeof(*o));
    o->cfg = ds4f_default_config();
    o->ep_size = 1; o->n_threads = ds4f_physical_cores(); o->n_cmgs = 1;
    o->bf16_pv = -1; o->int8kv_cal = 256; o->int8cmp_cal = 64;
    o->exact = 1; o->zero_copy_experts = 1; o->load_drop_blob = 1;
    o->hip_async = 1;
}

static const char *ds4f_json_key(const char *json, const char *key) {
    char needle[96]; snprintf(needle, sizeof(needle), "\"%s\"", key);
    return strstr(json, needle);
}
static int ds4f_json_int(const char *json, const char *key, int def) {
    const char *p = ds4f_json_key(json, key);
    if (!p) return def;
    p = strchr(p, ':'); if (!p) return def; p++;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return (int)strtol(p, NULL, 10);
}
static int ds4f_json_string(const char *json, const char *key, char *out, size_t cap) {
    const char *p = ds4f_json_key(json, key);
    if (!p) return 0;
    p = strchr(p, ':'); if (!p) return 0; p++;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != '"') return 0;
    p++;
    size_t n = 0;
    while (p[n] && p[n] != '"' && n + 1 < cap) {
        if (p[n] == '\\' && p[n + 1]) n++; /* paths here only need basic escaping */
        out[n] = p[n]; n++;
    }
    out[n] = 0;
    return p[n] == '"';
}
static int ds4f_runtime_options_load_json(ds4f_runtime_options *o, const char *path) {
    if (!o || !path || !*path) return -1;
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    char *json = (char *)alloca(64 * 1024); size_t n = fread(json, 1, 64 * 1024 - 1, f);
    fclose(f); json[n] = 0;
    if (n == 64 * 1024 - 1) return -1;
    char model[32];
    if (ds4f_json_string(json, "model", model, sizeof(model))) {
        if (strcmp(model, "ds4p") == 0) o->cfg = ds4f_pro_config();
        else if (strcmp(model, "ds4fbase") == 0) o->cfg = ds4f_base_config();
        else o->cfg = ds4f_default_config();
        if (strcmp(model, "q8pv") == 0) o->cfg.expert_qt = DS4F_Q8_PV;
    }
    (void)ds4f_json_string(json, "stage_dir", o->stage_dir, sizeof(o->stage_dir));
    (void)ds4f_json_string(json, "tokenizer", o->tokenizer, sizeof(o->tokenizer));
    (void)ds4f_json_string(json, "tokenizer_py", o->tokenizer_py, sizeof(o->tokenizer_py));
    o->ep_rank = ds4f_json_int(json, "ep_rank", o->ep_rank);
    o->ep_size = ds4f_json_int(json, "ep_size", o->ep_size);
    o->n_threads = ds4f_json_int(json, "threads", o->n_threads);
    o->n_cmgs = ds4f_json_int(json, "cmgs", o->n_cmgs);
    o->cfg.max_pos = ds4f_json_int(json, "max_pos", o->cfg.max_pos);
    { const char *mp = getenv("DS4F_MAX_POS");
      if (mp && *mp) { long v = atol(mp); if (v > 0) o->cfg.max_pos = (int)v; } }
    o->dense_bf16 = ds4f_json_int(json, "dense_bf16", o->dense_bf16);
    o->bf16_pv = ds4f_json_int(json, "bf16_pv", o->bf16_pv);
    o->dense_mxfp4 = ds4f_json_int(json, "dense_mxfp4", o->dense_mxfp4);
    o->mxfp4_w4a8 = ds4f_json_int(json, "mxfp4_w4a8", o->mxfp4_w4a8);
    o->q8_dense = ds4f_json_int(json, "q8_dense", o->q8_dense);
    o->fp8_magic = ds4f_json_int(json, "fp8_magic", o->fp8_magic);
    o->mxfp4_gemm_tile = ds4f_json_int(json, "mxfp4_gemm_tile", o->mxfp4_gemm_tile);
    o->hip_shared_bf16 = ds4f_json_int(json, "hip_shared_bf16", o->hip_shared_bf16);
    o->hip_shared_bf16_layers = ds4f_json_int(json, "hip_shared_bf16_layers", o->hip_shared_bf16_layers);
    o->hip_shared_fp16 = ds4f_json_int(json, "hip_shared_fp16", o->hip_shared_fp16);
    o->hip_shared_fp16_layers = ds4f_json_int(json, "hip_shared_fp16_layers", o->hip_shared_fp16_layers);
    o->hip_ordered_wkv_layers = ds4f_json_int(json, "hip_ordered_wkv_layers", o->hip_ordered_wkv_layers);
    o->hip_ordered_fp8_layers = ds4f_json_int(json, "hip_ordered_fp8_layers", o->hip_ordered_fp8_layers);
    o->hip_mxfp4_widen_layers = ds4f_json_int(json, "hip_mxfp4_widen_layers", o->hip_mxfp4_widen_layers);
    o->hip_mxfp4_resident_layers = ds4f_json_int(json, "hip_mxfp4_resident_layers", o->hip_mxfp4_resident_layers);
    o->hip_mxfp4_resident_auto = ds4f_json_int(json, "hip_mxfp4_resident_auto", o->hip_mxfp4_resident_auto);
    o->hip_vram_reserve_mb = ds4f_json_int(json, "hip_vram_reserve_mb", o->hip_vram_reserve_mb);
    o->hip_mxfp4_stream_raw = ds4f_json_int(json, "hip_mxfp4_stream_raw", o->hip_mxfp4_stream_raw);
    o->hip_expert_cache_mb = ds4f_json_int(json, "hip_expert_cache_mb", o->hip_expert_cache_mb);
    o->hip_expert_cache_stats = ds4f_json_int(json, "hip_expert_cache_stats", o->hip_expert_cache_stats);
    o->hip_prefill_attn = ds4f_json_int(json, "hip_prefill_attn", o->hip_prefill_attn);
    o->hip_exact_prefill = ds4f_json_int(json, "hip_exact_prefill", o->hip_exact_prefill);
    o->sparse = ds4f_json_int(json, "sparse", o->sparse);
    o->mhc = ds4f_json_int(json, "mhc", o->mhc);
    o->exact = ds4f_json_int(json, "exact", o->exact);
    o->tierb2 = ds4f_json_int(json, "tierb2", o->tierb2);
    o->int8_kv = ds4f_json_int(json, "int8_kv", o->int8_kv);
    o->int8_cmp = ds4f_json_int(json, "int8_cmp", o->int8_cmp);
    o->int4_cmp = ds4f_json_int(json, "int4_cmp", o->int4_cmp);
    o->tp_head = ds4f_json_int(json, "tp_head", o->tp_head);
    o->tp_shared = ds4f_json_int(json, "tp_shared", o->tp_shared);
    o->tp_shared_full = ds4f_json_int(json, "tp_shared_full", o->tp_shared_full);
    o->tp_attn = ds4f_json_int(json, "tp_attn", o->tp_attn);
    o->tp_oproj = ds4f_json_int(json, "tp_oproj", o->tp_oproj);
    o->tp_embed = ds4f_json_int(json, "tp_embed", o->tp_embed);
    o->tp_wob = ds4f_json_int(json, "tp_wob", o->tp_wob);
    o->mtp = ds4f_json_int(json, "mtp", o->mtp);
    o->expert_resident = ds4f_json_int(json, "expert_resident", o->expert_resident);
    o->cp = ds4f_json_int(json, "cp", o->cp);
    o->zero_copy_experts = ds4f_json_int(json, "zero_copy_experts", o->zero_copy_experts);
    o->use_hip = ds4f_json_int(json, "hip", o->use_hip);
    o->hip_device = ds4f_json_int(json, "hip_device", o->hip_device);
    o->hip_async = ds4f_json_int(json, "hip_async", o->hip_async);
    o->hip_verbose = ds4f_json_int(json, "hip_verbose", o->hip_verbose);
    return 0;
}

static ds4f_model *ds4f_alloc_synth(ds4f_config cfg, int ep_rank, int ep_size,
                                    int n_threads, int n_cmgs) {
    ds4f_runtime_options o = ds4f_runtime_options_debug_env(cfg, NULL, ep_rank, ep_size,
                                                             n_threads, n_cmgs);
    return ds4f_alloc_synth_opts(&o);
}

static void ds4f_free(ds4f_model *m) {
    if (!m) return;
    ds4f_pool_stop(m->pool);
    if (m->arena && m->arena != MAP_FAILED) munmap(m->arena, m->arena_sz);
    if (m->blob_map) munmap(m->blob_map, m->blob_map_sz);   /* zero-copy expert weights */
    ds4f_mem_pool *mem = m->mem;
    if (mem && getenv("DS4F_MEM_STATS")) ds4f_mem_pool_report(mem, "destroy");
    ds4f_mem_pool_destroy(mem);
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

/* One original safetensors shard placed in the loader's virtual concatenation
 * (DS4F_STAGE_NOCOPY manifests only). */
typedef struct { size_t vbase, size; char path[1024]; } ds4f_shard_ref;

typedef struct {
    ds4f_mani_ent *e; int n, cap;
    int       rank, ep_size;
    uint64_t  total_bytes;
    uint8_t  *blob; size_t blob_sz; int blob_fd;
    int       nocopy;      /* blob is a set of mapped shards, not a staged file */
    ds4f_mem_pool *meta_mem;
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
    B->meta_mem = ds4f_mem_pool_create();
    if (!B->meta_mem) { fclose(mf); return -1; }
    B->cap = 8192; B->e = (ds4f_mani_ent *)ds4f_mem_calloc(B->meta_mem, (size_t)B->cap, sizeof(*B->e), 64); B->n = 0;
    ds4f_shard_ref *sh = NULL; int nsh = 0, shcap = 0;
    char line[2048];
    while (fgets(line, sizeof line, mf)) {
        if (strncmp(line, "#shard ", 7) == 0) {
            /* DS4F_STAGE_NOCOPY manifest: "#shard <idx> <vbase> <size> <path>".
             * Offsets point into a virtual concatenation of the original
             * safetensors shards instead of a copied blob. */
            if (nsh == shcap) {
                int oldcap = shcap;
                shcap = shcap ? shcap * 2 : 64;
                ds4f_shard_ref *new_sh = (ds4f_shard_ref *)ds4f_mem_realloc(
                    B->meta_mem, sh, (size_t)oldcap * sizeof(*sh),
                    (size_t)shcap * sizeof(*sh), 64);
                if (!new_sh) {
                    fclose(mf); ds4f_mem_pool_destroy(B->meta_mem);
                    B->meta_mem = NULL; B->e = NULL; return -1;
                }
                sh = new_sh;
            }
            unsigned long long idx, vb, sz;
            char path[1024];
            if (sscanf(line, "#shard %llu %llu %llu %1023s", &idx, &vb, &sz, path) != 4) {
                fprintf(stderr, "ds4f_load: malformed #shard line in %s\n", mp);
                fclose(mf); ds4f_mem_pool_destroy(B->meta_mem); B->meta_mem = NULL; B->e = NULL; return -1;
            }
            sh[nsh].vbase = (size_t)vb; sh[nsh].size = (size_t)sz;
            snprintf(sh[nsh].path, sizeof sh[nsh].path, "%s", path);
            nsh++;
            continue;
        }
        if (line[0] == '#') {                    /* header carries rank / ep_size */
            char *p;
            if ((p = strstr(line, "rank=")))    B->rank    = atoi(p + 5);
            if ((p = strstr(line, "ep_size="))) B->ep_size = atoi(p + 8);
            continue;
        }
        if (B->n == B->cap) {
            int oldcap = B->cap; B->cap *= 2;
            ds4f_mani_ent *new_e = (ds4f_mani_ent *)ds4f_mem_realloc(B->meta_mem, B->e,
                    (size_t)oldcap * sizeof(*B->e), (size_t)B->cap * sizeof(*B->e), 64);
            if (!new_e) {
                fclose(mf); ds4f_mem_pool_destroy(B->meta_mem);
                B->meta_mem = NULL; B->e = NULL; return -1;
            }
            B->e = new_e;
        }
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

    if (nsh > 0) {
        /* Reserve one address range covering every shard, then MAP_FIXED each
         * file into its slot. Every consumer keeps doing (B->blob + off), so the
         * rest of the load path is identical to the copied-blob case -- but no
         * bytes were ever duplicated on disk. */
        size_t total = 0;
        for (int i = 0; i < nsh; i++) {
            size_t end = sh[i].vbase + sh[i].size;
            if (end > total) total = end;
        }
        uint8_t *base = (uint8_t *)mmap(NULL, total, PROT_NONE,
                                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (base == MAP_FAILED) {
            fprintf(stderr, "ds4f_load: reserve %zu bytes for %d shards failed: %s\n",
                    total, nsh, strerror(errno));
            ds4f_mem_pool_destroy(B->meta_mem); B->meta_mem = NULL; B->e = NULL; return -1;
        }
        const char *huge_e = getenv("DS4F_BLOB_HUGEPAGE");
        const char *pop_e   = getenv("DS4F_BLOB_POPULATE");
        int huge = huge_e && *huge_e && atoi(huge_e);
        int populate = pop_e && *pop_e && atoi(pop_e);
        if (huge || populate)
            fprintf(stderr, "ds4f_load: shard map policy: hugepage=%d populate=%d\n",
                    huge, populate);
        for (int i = 0; i < nsh; i++) {
            int fd = open(sh[i].path, O_RDONLY);
            if (fd < 0) {
                fprintf(stderr, "ds4f_load: cannot open shard %s: %s\n", sh[i].path, strerror(errno));
                munmap(base, total); ds4f_mem_pool_destroy(B->meta_mem); B->meta_mem = NULL; B->e = NULL; return -1;
            }
            int map_flags = MAP_PRIVATE | MAP_FIXED;
#ifdef MAP_POPULATE
            if (populate) map_flags |= MAP_POPULATE;
#else
            if (populate)
                fprintf(stderr, "ds4f_load: MAP_POPULATE unavailable on this target\n");
#endif
            void *got = mmap(base + sh[i].vbase, sh[i].size, PROT_READ,
                             map_flags, fd, 0);
            close(fd);                       /* the mapping keeps its own reference */
            if (got == MAP_FAILED) {
                fprintf(stderr, "ds4f_load: map shard %s (%zu B at +%zu) failed: %s\n",
                        sh[i].path, sh[i].size, sh[i].vbase, strerror(errno));
                munmap(base, total); ds4f_mem_pool_destroy(B->meta_mem); B->meta_mem = NULL; B->e = NULL; return -1;
            }
#ifdef MADV_HUGEPAGE
            if (huge) madvise(got, sh[i].size, MADV_HUGEPAGE);
#else
            if (huge)
                fprintf(stderr, "ds4f_load: MADV_HUGEPAGE unavailable on this target\n");
#endif
        }
        (void)sh;
        B->blob = base; B->blob_sz = total; B->blob_fd = -1; B->nocopy = 1;
        return 0;
    }
    (void)sh;

    B->blob_fd = open(bp, O_RDONLY);
    if (B->blob_fd < 0) { fprintf(stderr, "ds4f_load: cannot open %s: %s\n", bp, strerror(errno)); ds4f_mem_pool_destroy(B->meta_mem); B->meta_mem = NULL; B->e = NULL; return -1; }
    struct stat sb;
    if (fstat(B->blob_fd, &sb) != 0) { fprintf(stderr, "ds4f_load: fstat %s failed\n", bp); close(B->blob_fd); ds4f_mem_pool_destroy(B->meta_mem); B->meta_mem = NULL; B->e = NULL; return -1; }
    B->blob_sz = (size_t)sb.st_size;
    B->blob = (uint8_t *)mmap(NULL, B->blob_sz, PROT_READ, MAP_PRIVATE, B->blob_fd, 0);
    if (B->blob == MAP_FAILED) { fprintf(stderr, "ds4f_load: mmap %s (%zu) failed\n", bp, B->blob_sz); close(B->blob_fd); ds4f_mem_pool_destroy(B->meta_mem); B->meta_mem = NULL; B->e = NULL; return -1; }
    return 0;
}

static void ds4f_blob_close(ds4f_blob *B) {
    if (B->blob && B->blob != MAP_FAILED) munmap(B->blob, B->blob_sz);
    if (B->blob_fd > 0) close(B->blob_fd);   /* -1 under nocopy: shard fds already closed */
    ds4f_mem_pool_destroy(B->meta_mem); B->meta_mem = NULL; B->e = NULL; B->n = 0;
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
    char wn[320], sn[320];
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

/* Point a tensor AT the blob instead of copying into the arena. Only valid for
 * DS4F_MXFP4 experts under m->mxfp4_raw, where the kernels read the on-disk
 * nibble order directly. The blob pages are deliberately NOT dropped -- they are
 * the live weights now. */
static void ds4f_load_q_ref(ds4f_model *m, const ds4f_blob *B, ds4f_tensor *dst, const char *base) {
    char wn[320], sn[320];
    snprintf(wn, sizeof wn, "%s.weight", base);
    size_t wb = ds4f_wbytes(dst->type, dst->rows, dst->cols);
    const ds4f_mani_ent *we = ds4f_need(B, wn, ds4f_qtype_dtstr(dst->type), wb);
    dst->w = (void *)(B->blob + we->off);
    size_t sb = ds4f_sbytes(dst->type, dst->rows, dst->cols);
    if (sb) {
        snprintf(sn, sizeof sn, "%s.scale", base);
        const ds4f_mani_ent *se = ds4f_need(B, sn, "F8_E8M0", sb);
        dst->scale = (uint8_t *)(B->blob + se->off);
    } else dst->scale = NULL;
    m->bytes_read += wb + sb;
}

/* The EP no-copy path deliberately leaves routed MXFP4 experts file-backed.
 * That keeps the model within HBM, but under host memory pressure the kernel
 * may evict expert pages between tokens and decode becomes an I/O benchmark.
 * DS4F_EXPERT_RESIDENT=1 issues MADV_WILLNEED for the owned shard; =2 also
 * walks one byte per page, synchronously faulting the shard in before decode.
 * The latter is bounded by MemAvailable so it cannot turn the residency aid
 * into an OOM trigger on the 32-GB EP8 node. */
static double ds4f_mem_avail_gb(void);
static size_t ds4f_madvise_span(void *ptr, size_t bytes, int advice) {
    if (!ptr || !bytes) return 0;
    long pg = sysconf(_SC_PAGESIZE); if (pg <= 0) pg = 4096;
    uintptr_t lo = (uintptr_t)ptr & ~((uintptr_t)pg - 1);
    uintptr_t hi = ((uintptr_t)ptr + bytes + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1);
    if (hi <= lo) return 0;
    (void)madvise((void *)lo, (size_t)(hi - lo), advice);
    return (size_t)(hi - lo);
}
static size_t ds4f_expert_resident(ds4f_model *m, int mode) {
    if (!m->mxfp4_raw) return 0;
    if (mode <= 0) return 0;
    const char *e;
    size_t total = 0; long pg = sysconf(_SC_PAGESIZE); if (pg <= 0) pg = 4096;
    for (int L = 0; L < m->cfg.n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        for (int s = 0; s < ly->n_owned; s++) {
            ds4f_tensor *ts[3] = { &ly->ex_w1[s], &ly->ex_w3[s], &ly->ex_w2[s] };
            for (int j = 0; j < 3; j++) {
                size_t wb = ds4f_wbytes(ts[j]->type, ts[j]->rows, ts[j]->cols);
                size_t sb = ds4f_sbytes(ts[j]->type, ts[j]->rows, ts[j]->cols);
                total += ds4f_madvise_span(ts[j]->w, wb, MADV_WILLNEED);
                total += ds4f_madvise_span(ts[j]->scale, sb, MADV_WILLNEED);
            }
        }
    }
    if (mode < 2) {
        fprintf(stderr, "ds4f_load_real: expert residency hint queued for %.2f GB\n",
                (double)total / 1e9);
        return total;
    }
    double floor_gb = 2.0;
    e = getenv("DS4F_EXPERT_RESIDENT_MIN_AVAIL_GB");
    if (e && *e) floor_gb = atof(e);
    double avail = ds4f_mem_avail_gb();
    if (avail >= 0.0 && avail < floor_gb) {
        fprintf(stderr, "ds4f_load_real: expert residency skipped (MemAvailable %.2f GB < %.2f GB)\n",
                avail, floor_gb);
        return total;
    }
    /* Touching the whole EP8 expert shard (about 18 GB here) defeats the
     * purpose of the no-copy mapping on a 32-GB HBM node.  Keep the default
     * synchronous warm-up small; raise this explicitly only when the host has
     * enough headroom for the complete owned shard. */
    double touch_gb = 2.0;
    e = getenv("DS4F_EXPERT_RESIDENT_GB");
    if (e && *e) touch_gb = atof(e);
    if (touch_gb <= 0.0) touch_gb = 2.0;
    size_t touch_cap = (size_t)(touch_gb * 1e9);
    if (touch_cap > total) touch_cap = total;
    volatile uint8_t sink = 0;
    size_t touched = 0;
    for (int L = 0; L < m->cfg.n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        for (int s = 0; s < ly->n_owned; s++) {
            ds4f_tensor *ts[3] = { &ly->ex_w1[s], &ly->ex_w3[s], &ly->ex_w2[s] };
            for (int j = 0; j < 3; j++) {
                size_t wb = ds4f_wbytes(ts[j]->type, ts[j]->rows, ts[j]->cols);
                size_t sb = ds4f_sbytes(ts[j]->type, ts[j]->rows, ts[j]->cols);
                const uint8_t *p[2] = { (const uint8_t *)ts[j]->w, ts[j]->scale };
                size_t n[2] = { wb, sb };
                for (int q = 0; q < 2; q++) for (size_t off = 0; off < n[q]; off += (size_t)pg) {
                    if (touched >= touch_cap) goto resident_done;
                    sink ^= p[q][off]; touched += (size_t)pg;
                    if ((touched & ((size_t)256*1024*1024 - 1)) == 0) {
                        avail = ds4f_mem_avail_gb();
                        if (avail >= 0.0 && avail < floor_gb) {
                            fprintf(stderr, "ds4f_load_real: expert residency stopped at %.2f/%.2f GB (MemAvailable %.2f GB)\n",
                                    (double)touched/1e9, (double)total/1e9, avail);
                            return touched;
                        }
                    }
                }
            }
        }
    }
resident_done:
    (void)sink;
    fprintf(stderr, "ds4f_load_real: expert residency established for %.2f/%.2f GB (MemAvailable %.2f GB)\n",
            (double)touched / 1e9, (double)total / 1e9, ds4f_mem_avail_gb());
    return touched;
}

/* load a raw BF16/F32 buffer (un-scaled) into a plain arena pointer */
static void ds4f_load_raw(ds4f_model *m, const ds4f_blob *B, void *dst,
                          const char *name, ds4f_qtype type, int rows, int cols) {
    size_t wb = ds4f_wbytes(type, rows, cols);
    const ds4f_mani_ent *e = ds4f_need(B, name, ds4f_qtype_dtstr(type), wb);
    ds4f_tensor t = { dst, NULL, type, rows, cols, -1 };
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
    char wn[320], sn[320];
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
/* TP column-shard (FP8 only): copy columns [c0, c0+dst->cols) of the full [rows, Kfull] FP8 dense
 * into dst -- strided per row (cols not contiguous) + the 128-blocked E8M0 scale columns. c0 and
 * dst->cols 128-aligned. Used for wo_b under TP_OPROJ: bit-exact (s_o1 is already zero outside the
 * owned o_inter slice, so the dropped columns multiplied zeros). */
typedef struct { uint8_t *dw, *dscale; const uint8_t *sw, *ss; int rows, cols, Kfull, c0; } ds4f_cshard_task;
static void ds4f_cshard_worker(void *arg, int tid, int nthr) {
    ds4f_cshard_task *T = (ds4f_cshard_task *)arg;
    int rows = T->rows, cols = T->cols, Kfull = T->Kfull, c0 = T->c0;
    int r0, r1; ds4f_rowsplit8(rows, nthr, tid, &r0, &r1);
    for (int i = r0; i < r1; i++)                                  /* FP8 weight: cols [c0,c0+cols) of row i */
        memcpy(T->dw + (size_t)i*cols, T->sw + (size_t)i*Kfull + c0, (size_t)cols);
    if (tid == 0) {                                                /* E8M0 scale (tiny): block-cols [c0/128,..) */
        int srows = (rows+127)/128, sbcf = (Kfull+127)/128, sbc = (cols+127)/128, c0b = c0/128;
        for (int si = 0; si < srows; si++) memcpy(T->dscale + (size_t)si*sbc, T->ss + (size_t)si*sbcf + c0b, (size_t)sbc);
    }
}
static void ds4f_load_dense_cshard(ds4f_model *m, const ds4f_blob *B, ds4f_tensor *dst,
                                   const char *base, int c0, int Kfull) {
    char wn[320]; snprintf(wn, sizeof wn, "%s.weight", base);
    const ds4f_mani_ent *we = ds4f_mani_find(B, wn);
    if (!we) { fprintf(stderr, "cshard: MISSING '%s'\n", wn); abort(); }
    if (strcmp(we->dtype, "F8_E4M3") != 0 || dst->type != DS4F_FP8) { fprintf(stderr, "cshard: FP8 src+dst only\n"); abort(); }
    int rows = dst->rows, cols = dst->cols, sbcf = (Kfull+127)/128;
    char sn[320]; snprintf(sn, sizeof sn, "%s.scale", base);
    const ds4f_mani_ent *se = ds4f_need(B, sn, "F8_E8M0", (size_t)((rows+127)/128)*sbcf);
    ds4f_cshard_task T = { (uint8_t *)dst->w, (uint8_t *)dst->scale, B->blob + we->off, B->blob + se->off, rows, cols, Kfull, c0 };
    ds4f_pool_run(m->pool, ds4f_cshard_worker, &T);
    m->bytes_read += (size_t)rows*cols;
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

/* TP row-shard load: copy only rows [r0, r0+dst->rows) of dense tensor 'base' into dst
 * (dst->rows = shard count; full_rows = on-disk row count for the nbytes check). Reuses the
 * bf16/fp8 -> bf16(_pv) promote, advancing the source by r0 rows. r0 must be 8-aligned
 * (bf16-pv groups) and, for FP8 src, 128-aligned (its 128-row-blocked scale). */
static void ds4f_load_dense_vshard(ds4f_model *m, const ds4f_blob *B, ds4f_tensor *dst,
                                   const char *base, int r0, int full_rows) {
    char wn[256]; snprintf(wn, sizeof wn, "%s.weight", base);
    const ds4f_mani_ent *we = ds4f_mani_find(B, wn);
    if (!we) { fprintf(stderr, "ds4f_load: MISSING tensor '%s'\n", wn); abort(); }
    int K = dst->cols;
    int src_fp8 = (strcmp(we->dtype, "F8_E4M3") == 0), src_bf16 = (strcmp(we->dtype, "BF16") == 0);
    if (!src_fp8 && !src_bf16) { fprintf(stderr, "ds4f_load_dense_vshard: '%s' src %s unsupported\n", wn, we->dtype); abort(); }
    int same = (src_fp8 && dst->type == DS4F_FP8) || (src_bf16 && dst->type == DS4F_BF16);  /* direct copy, no promote */
    if (!same && dst->type != DS4F_BF16 && dst->type != DS4F_BF16_PV) { fprintf(stderr, "ds4f_load_dense_vshard: dst dtype %d NYI\n", dst->type); abort(); }
    size_t fwb = src_fp8 ? ds4f_wbytes(DS4F_FP8, full_rows, K) : ds4f_wbytes(DS4F_BF16, full_rows, K);
    if (we->nbytes != fwb) { fprintf(stderr, "ds4f_load_dense_vshard: '%s' nbytes %llu != full %zu\n", wn, (unsigned long long)we->nbytes, fwb); abort(); }
    const uint8_t *sw, *ss = NULL;
    if (src_fp8) {
        sw = B->blob + we->off + (size_t)r0 * K;                 /* fp8: 1 byte/elem (r0 128-aligned) */
        char sn[256]; snprintf(sn, sizeof sn, "%s.scale", base);
        const ds4f_mani_ent *se = ds4f_need(B, sn, "F8_E8M0", ds4f_sbytes(DS4F_FP8, full_rows, K));
        ss = B->blob + se->off + (size_t)(r0 / 128) * ((K + 127) / 128);
    } else {
        sw = B->blob + we->off + (size_t)r0 * K * 2;             /* bf16: 2 bytes/elem */
    }
    if (same) {                                                 /* FP8->FP8 / bf16->bf16: direct shard copy */
        ds4f_copy_run(m, *dst, sw, ss);
        m->bytes_read += ds4f_wbytes(dst->type, dst->rows, K) + ds4f_sbytes(dst->type, dst->rows, K);
        return;
    }
    ds4f_promote_task T = { *dst, sw, ss, m->fp8_lut, src_fp8 };
    ds4f_pool_run(m->pool, ds4f_promote_worker, &T);             /* fills dst->rows rows from the offset src */
    m->bytes_read += ds4f_wbytes(src_fp8 ? DS4F_FP8 : DS4F_BF16, dst->rows, K);
}

static ds4f_model *ds4f_load_real_opts(const ds4f_runtime_options *opt) {
    ds4f_config cfg = opt->cfg;
    int ep_rank = opt->ep_rank, ep_size = opt->ep_size;
    int n_threads = opt->n_threads, n_cmgs = opt->n_cmgs;
    double t0 = ds4f_wall();
    ds4f_drop_blob = opt->load_drop_blob;
    const char *blob_dir = opt->stage_dir[0] ? opt->stage_dir : "/local/ds4f";
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

    ds4f_mem_pool *mem = ds4f_mem_pool_create();
    ds4f_model *m = mem ? (ds4f_model *)ds4f_mem_calloc(mem, 1, sizeof(*m), 64) : NULL;
    if (!m) { ds4f_blob_close(&B); return NULL; }
    m->mem = mem;
    m->cfg = cfg; m->ep_rank = ep_rank; m->ep_size = ep_size;
    m->n_threads = n_threads; m->n_cmgs = n_cmgs;
    m->gpu_exact_prefill = opt->hip_exact_prefill ? 1 : 0;
    m->mxfp4_w4a8 = opt->mxfp4_w4a8;
    { const char *e = getenv("DS4F_ROUTE_TELEMETRY");
      m->route_telemetry = opt->hip_expert_cache_mb != 0 || opt->hip_expert_cache_stats ||
                           (e && *e && atoi(e) != 0); }
    /* real dtypes: staged dense = FP8(e4m3fn), experts = MXFP4, router/head/embed/
     * norm = BF16 row-major. DS4F_FP8_BF16=1 PROMOTES the replicated dense FP8->bf16
     * at load time (EXACT: e4m3 fits bf16) and auto-enables the pv pair-interleaved
     * layout -> ~1.7x faster matvecs at +~6 GB; DS4F_BF16_PV=0 forces plain bf16, =1
     * forces pv. Mirrors ds4f_alloc_synth's knobs. (DENSE_MXFP4 dense is lossy from
     * real FP8 => NYI; the loader aborts if requested.) */
    ds4f_init_fp8_e4m3fn_lut(m->fp8_lut);
    {   int pre = opt->dense_bf16 ? 1 : 0;
        m->bf16_pv = opt->bf16_pv >= 0 ? (opt->bf16_pv ? 1 : 0) : pre;
        m->dense_qt = pre ? (m->bf16_pv ? DS4F_BF16_PV : DS4F_BF16) : DS4F_FP8;
        m->bf16_mv_qt = m->bf16_pv ? DS4F_BF16_PV : DS4F_BF16; }
    { m->fp8_magic = opt->fp8_magic ? 1 : 0; }
#if !defined(__ARM_FEATURE_SVE)
    m->fp8_magic = 0;      /* the register magic decode is an SVE-only path */
#endif
    { m->mxfp4_gemm_tile = opt->mxfp4_gemm_tile; }
    { m->sparse = opt->sparse ? 1 : 0; }
    { m->mhc = opt->mhc ? 1 : 0; }
    { m->exact = opt->exact ? 1 : 0; }
    { m->tierb2 = opt->tierb2 ? 1 : 0; }
    /* DS4F_Q8_DENSE=1: repack the dominant dense (bf16-pv) -> int8 W8A8 after load
     * (svdot, ~1 B/elem -> halves dense matvec HBM bytes AND reclaims the bf16 src).
     * Was only wired into ds4f_alloc_synth; mirror it here so the REAL path honors it. */
    { m->q8_dense = opt->q8_dense ? 1 : 0; }
    /* DS4F_INT8_KV=1: store the kv latent as int8 + static per-channel scale (S5) -> half
     * the KV footprint at long ctx. Streaming exact decode path only (see ds4f_layer.kv_q). */
    { m->int8_kv = opt->int8_kv ? 1 : 0; }
    { ds4f_int8kv_cal = opt->int8kv_cal > 0 ? opt->int8kv_cal : 256;
      if (ds4f_int8kv_cal < 1) ds4f_int8kv_cal = 1; }
    { m->int8_cmp = opt->int8_cmp ? 1 : 0; }
    { m->int4_cmp = opt->int4_cmp ? 1 : 0;
      if (m->int4_cmp) m->int8_cmp = 1; }   /* int4 cmp implies the int8_cmp infra (cmp_q4 sub-mode) */
    { m->cp = opt->cp ? 1 : 0; }
    { ds4f_int8cmp_cal = opt->int8cmp_cal > 0 ? opt->int8cmp_cal : 64;
      if (ds4f_int8cmp_cal < 1) ds4f_int8cmp_cal = 1; }
    if (m->tierb2) m->exact = 1;   /* Tier-B2 reuses the exact q-norm/RoPE/window path */
    if (m->int8_kv) m->exact = 1;  /* int8 KV uses the exact streaming decode path */
    if (m->int8_cmp) m->exact = 1; /* int8 cmp uses the exact streaming tierb2 path */
    /* Zero-copy routed experts. Needs (a) a DS4F_STAGE_NOCOPY manifest, so the
     * blob IS the mapped safetensors shards, and (b) kernels that read the
     * on-disk MXFP4 nibble order. Default on where both hold: it removes ~147 GB
     * of anonymous arena, which is what makes the full model fit here at all. */
#if !defined(__ARM_FEATURE_SVE) && defined(__AVX2__) && defined(__FMA__)
    {   int want = opt->zero_copy_experts;
        m->mxfp4_raw = (want && B.nocopy && cfg.expert_qt == DS4F_MXFP4) ? 1 : 0;
        if (want && !B.nocopy)
            fprintf(stderr, "ds4f_load_real: zero-copy experts need a DS4F_STAGE_NOCOPY "
                            "manifest; falling back to the repack-into-arena path\n");
    }
#endif
    ds4f_arena_no_experts = m->mxfp4_raw;

    m->pool = ds4f_pool_start_ex(mem, n_threads, n_cmgs);

    m->arena_sz = ds4f_arena_size(&cfg, ep_rank, ep_size,
                                  m->dense_qt == DS4F_BF16 || m->dense_qt == DS4F_BF16_PV,
                                  m->tierb2 && !m->int8_kv, opt);
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
    { int er0, erows; ds4f_tp_embed_shard(cfg.vocab, ep_rank, ep_size, &er0, &erows, opt->tp_embed); m->emb_r0 = er0; m->emb_rows = erows;  /* TP: vocab-shard */
      ds4f_tensor embed = ds4f_new_tensor(m, DS4F_BF16, erows, C); m->embed = (uint16_t *)embed.w; }
    {   int hr0, hrows; ds4f_tp_head_shard(cfg.vocab, ep_rank, ep_size, &hr0, &hrows, opt->tp_head);  /* TP: vocab-shard */
        m->head = ds4f_new_tensor(m, m->bf16_mv_qt, hrows, C); m->head_r0 = hr0; }   /* matvec'd -> pv when promoted */
    ds4f_tp_shared_shard(cfg.shared_inter, ep_rank, ep_size, m->dense_qt == DS4F_FP8 ? 128 : 8, &m->sh_r0, &m->sh_rows, opt->tp_shared);
    ds4f_tp_shared_out_shard(C, ep_rank, ep_size, m->dense_qt == DS4F_FP8 ? 128 : 8, &m->sh2_r0, &m->sh2_rows, opt->tp_shared && opt->tp_shared_full);
    ds4f_tp_attn_shard(cfg.n_heads, ep_rank, ep_size, &m->attn_h0, &m->attn_h1, opt->tp_attn);  /* DS4F_TP_ATTN: head range */
    ds4f_tp_oproj_shard(cfg.o_inter, ep_rank, ep_size, m->dense_qt == DS4F_FP8 ? 128 : 8, &m->oi0, &m->oi_rows, opt->tp_oproj);  /* DS4F_TP_OPROJ: wo_a o_inter shard */
    {   int hc = cfg.hc_mult, hd = hc * C;
        m->hc_head_fn    = (float *)ds4f_bump(m, (size_t)hc * hd * 4, 256);
        m->hc_head_base  = (float *)ds4f_bump(m, (size_t)hc * 4, 64);
        m->hc_head_scale = (float *)ds4f_bump(m, (size_t)4, 64); }

    m->layers = (ds4f_layer *)ds4f_mem_calloc(m->mem, (size_t)cfg.n_layers, sizeof(ds4f_layer), 64);
    int no = ds4f_n_owned(cfg.n_experts, ep_rank, ep_size);
    for (int L = 0; L < cfg.n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        ly->attn_norm = (uint16_t *)ds4f_bump(m, (size_t)C * 2, 64);
        ly->ffn_norm  = (uint16_t *)ds4f_bump(m, (size_t)C * 2, 64);
        ly->q_norm    = (uint16_t *)ds4f_bump(m, (size_t)cfg.q_lora * 2, 64);
        ly->kv_norm   = (uint16_t *)ds4f_bump(m, (size_t)cfg.kv_lora * 2, 64);
        ds4f_qtype dq = m->dense_qt;                            /* FP8 | BF16 | BF16_PV */
        ly->wq_a = ds4f_new_tensor(m, dq, cfg.q_lora, C);
        ly->wq_b = ds4f_new_tensor(m, dq, (m->attn_h1 - m->attn_h0) * cfg.q_head_dim, cfg.q_lora);  /* TP: owned heads */
        ly->wkv  = ds4f_new_tensor(m, dq, cfg.kv_lora, C);
        ly->wo_a = ds4f_new_tensor(m, dq, m->oi_rows, C);       /* TP: o_inter row-shard */
        {   /* wo_b: FP8 o_inter col-shard (pairs w/ TP_OPROJ) */
            int wob_s = (m->oi_rows < cfg.o_inter) && dq == DS4F_FP8 && opt->tp_wob;
            ly->wo_b = ds4f_new_tensor(m, dq, C, wob_s ? m->oi_rows : cfg.o_inter); }
        ly->attn_sink = (float *)ds4f_bump(m, (size_t)cfg.n_heads * 4, 64);
        ly->gate = ds4f_new_tensor(m, m->bf16_mv_qt, cfg.n_experts, C); /* router matvec -> pv */
        /* router selection bias (F32[n_experts]); only non-hash layers have it.
         * Off-arena (tiny, read single-threaded in the exact gate, not a matvec). */
        if (L >= cfg.n_hash_layers) ly->gate_bias = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.n_experts * 4, 64, 1);
        ly->sh_w1 = ds4f_new_tensor(m, dq, m->sh_rows, C);       /* TP: col-shard shared_inter */
        ly->sh_w3 = ds4f_new_tensor(m, dq, m->sh_rows, C);
        ly->sh_w2 = ds4f_new_tensor(m, dq, m->sh2_rows, cfg.shared_inter); /* TP_SHARED_FULL: hidden-row shard */
        ly->ex_w1 = (ds4f_tensor *)ds4f_mem_calloc(m->mem, (size_t)no, sizeof(ds4f_tensor), 64);
        ly->ex_w2 = (ds4f_tensor *)ds4f_mem_calloc(m->mem, (size_t)no, sizeof(ds4f_tensor), 64);
        ly->ex_w3 = (ds4f_tensor *)ds4f_mem_calloc(m->mem, (size_t)no, sizeof(ds4f_tensor), 64);
        ly->owned_eid = (int *)ds4f_mem_calloc(m->mem, (size_t)no, sizeof(int), 64);
        ly->n_owned = no;
        int slot = 0;
        for (int e = 0; e < cfg.n_experts; e++) if (e % ep_size == ep_rank) {
            if (m->mxfp4_raw) {      /* pointers filled in by ds4f_load_q_ref; no arena */
                ds4f_tensor t1 = { NULL, NULL, DS4F_MXFP4, cfg.moe_inter, C, -1 };
                ds4f_tensor t2 = { NULL, NULL, DS4F_MXFP4, C, cfg.moe_inter, -1 };
                ly->ex_w1[slot] = t1; ly->ex_w3[slot] = t1; ly->ex_w2[slot] = t2;
            } else {
                ly->ex_w1[slot] = ds4f_new_tensor(m, DS4F_MXFP4, cfg.moe_inter, C);
                ly->ex_w3[slot] = ds4f_new_tensor(m, DS4F_MXFP4, cfg.moe_inter, C);
                ly->ex_w2[slot] = ds4f_new_tensor(m, DS4F_MXFP4, C, cfg.moe_inter);
            }
            ly->owned_eid[slot] = e; slot++;
        }
        {   int hc = cfg.hc_mult, mix = (2 + hc) * hc, hd = hc * C;
            ly->hc_attn_fn    = (float *)ds4f_bump(m, (size_t)mix * hd * 4, 256);
            ly->hc_attn_base  = (float *)ds4f_bump(m, (size_t)mix * 4, 64);
            ly->hc_attn_scale = (float *)ds4f_bump(m, (size_t)3 * 4, 64);
            ly->hc_ffn_fn     = (float *)ds4f_bump(m, (size_t)mix * hd * 4, 256);
            ly->hc_ffn_base   = (float *)ds4f_bump(m, (size_t)mix * 4, 64);
            ly->hc_ffn_scale  = (float *)ds4f_bump(m, (size_t)3 * 4, 64); }
        /* kv_cache ring buffer: sparse layers need only window_size slots (older history
         * served by cmp_kv); dense layers (0,1) keep max_pos. See ds4f_alloc_synth. */
        ly->kv_slots = (!m->int8_kv && m->tierb2) ? cfg.window_size : cfg.max_pos;
        if (m->int8_kv) ds4f_alloc_int8kv(m, ly, &cfg);            /* int8 KV store (half) + calbuf + scales */
        else ly->kv_cache = (uint16_t *)ds4f_bump(m, (size_t)ly->kv_slots * cfg.kv_lora * sizeof(uint16_t), 256);
    }
    if (m->tierb2) ds4f_alloc_tb2(m, 0);   /* off-arena compressor/indexer (load by name below) */

    /* ---- copy REAL bytes by name (verify dtype/size; abort on any mismatch) ---- */
    int mix = (2 + cfg.hc_mult) * cfg.hc_mult;
    ds4f_load_raw(m, &B, m->out_norm,      "norm.weight",   DS4F_BF16, 1, C);
    if (m->emb_rows < cfg.vocab) {                          /* TP: load only this node's embed vocab shard */
        ds4f_tensor et = { m->embed, NULL, DS4F_BF16, m->emb_rows, C, -1 };
        ds4f_load_dense_vshard(m, &B, &et, "embed", m->emb_r0, cfg.vocab);
    } else ds4f_load_raw(m, &B, m->embed,  "embed.weight",  DS4F_BF16, cfg.vocab, C);
    if (m->head_r0 || m->head.rows < cfg.vocab)                 /* TP: load only this node's vocab shard */
        ds4f_load_dense_vshard(m, &B, &m->head, "head", m->head_r0, cfg.vocab);
    else ds4f_load_dense(m, &B, &m->head,  "head");
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
        if (m->attn_h1 - m->attn_h0 < cfg.n_heads)              /* TP: load only owned heads' rows */
            ds4f_load_dense_vshard(m, &B, &ly->wq_b, DS4F_LN("attn.wq_b"), m->attn_h0 * cfg.q_head_dim, cfg.n_heads * cfg.q_head_dim);
        else ds4f_load_dense(m, &B, &ly->wq_b,   DS4F_LN("attn.wq_b"));
        ds4f_load_dense(m, &B, &ly->wkv,    DS4F_LN("attn.wkv"));
        if (m->oi_rows < cfg.o_inter)                           /* TP: load only this node's o_inter rows */
            ds4f_load_dense_vshard(m, &B, &ly->wo_a, DS4F_LN("attn.wo_a"), m->oi0, cfg.o_inter);
        else ds4f_load_dense(m, &B, &ly->wo_a,   DS4F_LN("attn.wo_a"));
        if (ly->wo_b.cols < cfg.o_inter)                       /* wo_b col-shard: load only o_inter cols [oi0,..) */
            ds4f_load_dense_cshard(m, &B, &ly->wo_b, DS4F_LN("attn.wo_b"), m->oi0, cfg.o_inter);
        else ds4f_load_dense(m, &B, &ly->wo_b, DS4F_LN("attn.wo_b"));
        ds4f_load_raw(m, &B, ly->attn_sink, DS4F_LN("attn.attn_sink"),       DS4F_F32, 1, cfg.n_heads);
        ds4f_load_dense(m, &B, &ly->gate,   DS4F_LN("ffn.gate"));
        if (ly->gate_bias)   /* noaux_tc selection bias (F32[n_experts]); non-hash layers only */
            ds4f_load_raw(m, &B, ly->gate_bias, DS4F_LN("ffn.gate.bias"), DS4F_F32, 1, cfg.n_experts);
        if (m->sh_rows < cfg.shared_inter) {                    /* TP: col-shard sh_w1/sh_w3 */
            ds4f_load_dense_vshard(m, &B, &ly->sh_w1, DS4F_LN("ffn.shared_experts.w1"), m->sh_r0, cfg.shared_inter);
            ds4f_load_dense_vshard(m, &B, &ly->sh_w3, DS4F_LN("ffn.shared_experts.w3"), m->sh_r0, cfg.shared_inter);
        } else {
            ds4f_load_dense(m, &B, &ly->sh_w1,  DS4F_LN("ffn.shared_experts.w1"));
            ds4f_load_dense(m, &B, &ly->sh_w3,  DS4F_LN("ffn.shared_experts.w3"));
        }
        if (m->sh2_rows < C)
            ds4f_load_dense_vshard(m, &B, &ly->sh_w2, DS4F_LN("ffn.shared_experts.w2"), m->sh2_r0, C);
        else ds4f_load_dense(m, &B, &ly->sh_w2, DS4F_LN("ffn.shared_experts.w2"));
        for (int s = 0; s < no; s++) {
            int e = ly->owned_eid[s];
            void (*ldq)(ds4f_model *, const ds4f_blob *, ds4f_tensor *, const char *) =
                m->mxfp4_raw ? ds4f_load_q_ref : ds4f_load_q;
            snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w1", L, e); ldq(m, &B, &ly->ex_w1[s], nm);
            snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w3", L, e); ldq(m, &B, &ly->ex_w3[s], nm);
            snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w2", L, e); ldq(m, &B, &ly->ex_w2[s], nm);
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

    /* DS4F_MTP scaffold: load the mtp.0 block (a dense transformer layer: MLA attn + 256-expert MoE,
     * NO tier-B2 compressor) + the fusion (enorm/hnorm/norm + e_proj/h_proj + own HC-head). Reuses the
     * main embed+head at forward. Experts EP-sharded; dense kept FULL (no TP) for the scaffold. The
     * draft/verify spec-decode loop + the M=K verify rework are the follow-on (env-blocked this session). */
    int mtp_want = opt->mtp;
    int mtp_present = ds4f_mani_find(&B, "mtp.0.attn_norm.weight") != NULL;
    if (mtp_want && !mtp_present)
        fprintf(stderr, "ds4f_load_real: DS4F_MTP requested but checkpoint has no mtp.0 block; disabled\n");
    if (mtp_want && mtp_present) {
        ds4f_layer *mt = &m->mtp; char mn[256]; ds4f_qtype dq = m->dense_qt;
        int C2 = cfg.hidden, hc = cfg.hc_mult, mix = (2 + hc) * hc, no = ds4f_n_owned(cfg.n_experts, ep_rank, ep_size);
        #define MTPN(f) (snprintf(mn, sizeof mn, "mtp.0.%s", f), mn)
        mt->attn_norm=(uint16_t*)ds4f_bump(m,(size_t)C2*2,64); ds4f_load_raw(m,&B,mt->attn_norm,MTPN("attn_norm.weight"),DS4F_BF16,1,C2);
        mt->ffn_norm =(uint16_t*)ds4f_bump(m,(size_t)C2*2,64); ds4f_load_raw(m,&B,mt->ffn_norm, MTPN("ffn_norm.weight"), DS4F_BF16,1,C2);
        mt->q_norm   =(uint16_t*)ds4f_bump(m,(size_t)cfg.q_lora*2,64);  ds4f_load_raw(m,&B,mt->q_norm, MTPN("attn.q_norm.weight"), DS4F_BF16,1,cfg.q_lora);
        mt->kv_norm  =(uint16_t*)ds4f_bump(m,(size_t)cfg.kv_lora*2,64); ds4f_load_raw(m,&B,mt->kv_norm,MTPN("attn.kv_norm.weight"),DS4F_BF16,1,cfg.kv_lora);
        mt->attn_sink=(float*)ds4f_bump(m,(size_t)cfg.n_heads*4,64);    ds4f_load_raw(m,&B,mt->attn_sink,MTPN("attn.attn_sink"),DS4F_F32,1,cfg.n_heads);
        mt->wq_a=ds4f_new_tensor(m,dq,cfg.q_lora,C2);                        ds4f_load_dense(m,&B,&mt->wq_a,MTPN("attn.wq_a"));
        mt->wq_b=ds4f_new_tensor(m,dq,cfg.n_heads*cfg.q_head_dim,cfg.q_lora); ds4f_load_dense(m,&B,&mt->wq_b,MTPN("attn.wq_b"));
        mt->wkv =ds4f_new_tensor(m,dq,cfg.kv_lora,C2);                       ds4f_load_dense(m,&B,&mt->wkv,MTPN("attn.wkv"));
        mt->wo_a=ds4f_new_tensor(m,dq,cfg.o_inter,C2);                       ds4f_load_dense(m,&B,&mt->wo_a,MTPN("attn.wo_a"));
        mt->wo_b=ds4f_new_tensor(m,dq,C2,cfg.o_inter);                       ds4f_load_dense(m,&B,&mt->wo_b,MTPN("attn.wo_b"));
        mt->gate=ds4f_new_tensor(m,m->bf16_mv_qt,cfg.n_experts,C2);          ds4f_load_dense(m,&B,&mt->gate,MTPN("ffn.gate"));
        mt->gate_bias=(float*)ds4f_mem_alloc(m->mem,(size_t)cfg.n_experts*4,64,1);     ds4f_load_raw(m,&B,mt->gate_bias,MTPN("ffn.gate.bias"),DS4F_F32,1,cfg.n_experts);
        /* the MTP ffn has NO shared expert (no mtp.0.ffn.shared_experts.* in the checkpoint) -> sh_w* stay
         * NULL; the block-forward must skip the shared contribution for the MTP layer when wired. */
        mt->ex_w1=(ds4f_tensor*)ds4f_mem_calloc(m->mem,(size_t)no,sizeof(ds4f_tensor),64); mt->ex_w2=(ds4f_tensor*)ds4f_mem_calloc(m->mem,(size_t)no,sizeof(ds4f_tensor),64); mt->ex_w3=(ds4f_tensor*)ds4f_mem_calloc(m->mem,(size_t)no,sizeof(ds4f_tensor),64);
        mt->owned_eid=(int*)ds4f_mem_calloc(m->mem,(size_t)no,sizeof(int),64); mt->n_owned=no;
        { int slot=0; for (int e=0;e<cfg.n_experts;e++) if (e%ep_size==ep_rank) {
            mt->ex_w1[slot]=ds4f_new_tensor(m,DS4F_MXFP4,cfg.moe_inter,C2);
            mt->ex_w3[slot]=ds4f_new_tensor(m,DS4F_MXFP4,cfg.moe_inter,C2);
            mt->ex_w2[slot]=ds4f_new_tensor(m,DS4F_MXFP4,C2,cfg.moe_inter);
            snprintf(mn,sizeof mn,"mtp.0.ffn.experts.%d.w1",e); ds4f_load_q(m,&B,&mt->ex_w1[slot],mn);
            snprintf(mn,sizeof mn,"mtp.0.ffn.experts.%d.w3",e); ds4f_load_q(m,&B,&mt->ex_w3[slot],mn);
            snprintf(mn,sizeof mn,"mtp.0.ffn.experts.%d.w2",e); ds4f_load_q(m,&B,&mt->ex_w2[slot],mn);
            mt->owned_eid[slot]=e; slot++; } }
        mt->hc_attn_fn=(float*)ds4f_bump(m,(size_t)mix*hc*C2*4,256); mt->hc_attn_base=(float*)ds4f_bump(m,(size_t)mix*4,64); mt->hc_attn_scale=(float*)ds4f_bump(m,(size_t)3*4,64);
        mt->hc_ffn_fn =(float*)ds4f_bump(m,(size_t)mix*hc*C2*4,256); mt->hc_ffn_base =(float*)ds4f_bump(m,(size_t)mix*4,64); mt->hc_ffn_scale =(float*)ds4f_bump(m,(size_t)3*4,64);
        ds4f_load_raw(m,&B,mt->hc_attn_fn,MTPN("hc_attn_fn"),DS4F_F32,mix,hc*C2); ds4f_load_raw(m,&B,mt->hc_attn_base,MTPN("hc_attn_base"),DS4F_F32,1,mix); ds4f_load_raw(m,&B,mt->hc_attn_scale,MTPN("hc_attn_scale"),DS4F_F32,1,3);
        ds4f_load_raw(m,&B,mt->hc_ffn_fn, MTPN("hc_ffn_fn"), DS4F_F32,mix,hc*C2); ds4f_load_raw(m,&B,mt->hc_ffn_base, MTPN("hc_ffn_base"), DS4F_F32,1,mix); ds4f_load_raw(m,&B,mt->hc_ffn_scale, MTPN("hc_ffn_scale"), DS4F_F32,1,3);
        m->mtp_enorm=(uint16_t*)ds4f_bump(m,(size_t)C2*2,64); ds4f_load_raw(m,&B,m->mtp_enorm,MTPN("enorm.weight"),DS4F_BF16,1,C2);
        m->mtp_hnorm=(uint16_t*)ds4f_bump(m,(size_t)C2*2,64); ds4f_load_raw(m,&B,m->mtp_hnorm,MTPN("hnorm.weight"),DS4F_BF16,1,C2);
        m->mtp_norm =(uint16_t*)ds4f_bump(m,(size_t)C2*2,64); ds4f_load_raw(m,&B,m->mtp_norm, MTPN("norm.weight"), DS4F_BF16,1,C2);
        m->mtp_e_proj=ds4f_new_tensor(m,dq,C2,C2); ds4f_load_dense(m,&B,&m->mtp_e_proj,MTPN("e_proj"));
        m->mtp_h_proj=ds4f_new_tensor(m,dq,C2,C2); ds4f_load_dense(m,&B,&m->mtp_h_proj,MTPN("h_proj"));
        { int hd=hc*C2; m->mtp_hc_fn=(float*)ds4f_bump(m,(size_t)hc*hd*4,256); m->mtp_hc_base=(float*)ds4f_bump(m,(size_t)hc*4,64); m->mtp_hc_scale=(float*)ds4f_bump(m,(size_t)4,64);
          ds4f_load_raw(m,&B,m->mtp_hc_fn,MTPN("hc_head_fn"),DS4F_F32,hc,hd); ds4f_load_raw(m,&B,m->mtp_hc_base,MTPN("hc_head_base"),DS4F_F32,1,hc); ds4f_load_raw(m,&B,m->mtp_hc_scale,MTPN("hc_head_scale"),DS4F_F32,1,1); }
        mt->kv_slots = cfg.window_size;   /* MTP: dense window attention -> window-size KV ring */
        mt->kv_cache = (uint16_t *)ds4f_mem_calloc(m->mem, (size_t)mt->kv_slots * (size_t)cfg.kv_lora, sizeof(uint16_t), 256);
        mt->sh_w1.w = mt->sh_w3.w = mt->sh_w2.w = NULL;   /* MTP has no shared expert (gate the shared step on sh_w1.w) */
        m->has_mtp=1;
        #undef MTPN
        fprintf(stderr,"ds4f_load_mtp: MTP block loaded (rank %d/%d, %d owned experts)\n", ep_rank, ep_size, no);
    }

    if (m->mxfp4_raw) ds4f_expert_resident(m, opt->expert_resident);

    double loaded_gb = (double)m->bytes_read / 1e9;
    int n_tensors = B.n; uint64_t staged = B.total_bytes;
    m->bytes_read = 0;                 /* runner resets per token; start clean */
    if (m->mxfp4_raw) {
        /* The expert tensors point into this mapping, so it must outlive the
         * load. Hand it to the model and release only the manifest table. */
        m->blob_map = B.blob; m->blob_map_sz = B.blob_sz;
        B.blob = NULL;                 /* stop ds4f_blob_close from unmapping it */
        ds4f_blob_close(&B);
    } else {
        ds4f_blob_close(&B);           /* arena holds the copies; release the blob mmap */
    }

    /* ---- scratch (identical to ds4f_alloc_synth) ---- */
    int H = cfg.n_heads * cfg.q_head_dim;
    m->s_hn    = (float *)ds4f_mem_alloc(m->mem, (size_t)C * 4, 256, 1);
    m->s_qlat  = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.q_lora * 4, 256, 1);
    m->s_q     = (float *)ds4f_mem_alloc(m->mem, (size_t)H * 4, 256, 1);
    m->s_kvlat = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.kv_lora * 4, 256, 1);
    m->s_attn  = (float *)ds4f_mem_alloc(m->mem, (size_t)H * 4, 256, 1);
    m->s_oin   = (float *)ds4f_mem_alloc(m->mem, (size_t)C * 4, 256, 1);
    m->s_o1    = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.o_inter * 4, 256, 1);
    m->s_o     = (float *)ds4f_mem_alloc(m->mem, (size_t)C * 4, 256, 1);
    m->s_h2    = (float *)ds4f_mem_alloc(m->mem, (size_t)C * 4, 256, 1);
    m->s_router= (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.n_experts * 4, 256, 1);
    m->s_shg   = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.shared_inter * 4, 256, 1);
    m->s_shu   = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.shared_inter * 4, 256, 1);
    m->s_exg   = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.moe_inter * 4, 256, 1);
    m->s_exu   = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.moe_inter * 4, 256, 1);
    m->s_exb_g = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.n_active * cfg.moe_inter * 4, 256, 1);
    m->s_exb_u = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.n_active * cfg.moe_inter * 4, 256, 1);
    m->s_exb_o = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.n_active * C * 4, 256, 1);
    m->s_moe   = (float *)ds4f_mem_alloc(m->mem, (size_t)C * 4, 256, 1);
    m->s_route = (float *)ds4f_mem_alloc(m->mem, (size_t)C * 4, 256, 1);
    m->s_logits= (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.vocab * 4, 256, 1);
    m->idx_blk_stride = cfg.max_pos > cfg.index_topk ? cfg.max_pos : cfg.index_topk;
    m->s_idx_scores = (float *)ds4f_mem_alloc(m->mem, (size_t)n_threads * m->idx_blk_stride * 4, 256, 1);
    m->s_idx_sel    = (int   *)ds4f_mem_alloc(m->mem, (size_t)n_threads * cfg.index_topk * 4, 256, 1);
    m->s_x4    = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.hc_mult * C * 4, 256, 1);
    m->s_resid = (float *)ds4f_mem_alloc(m->mem, (size_t)cfg.hc_mult * C * 4, 256, 1);
    m->s_xc    = (float *)ds4f_mem_alloc(m->mem, (size_t)C * 4, 256, 1);

    double el = ds4f_wall() - t0;
    fprintf(stderr,
        "ds4f_load_real rank %d/%d: %d staged tensors, loaded %.2f GB "
        "(FP8 e4m3fn dense + MXFP4 experts %s, %d owned), arena %.2f GB, %.1f s, %.2f GB/s\n",
        ep_rank, ep_size, n_tensors, loaded_gb,
        m->mxfp4_raw ? "referenced in place" : "repacked", no, (double)m->arena_used / 1e9,
        el, el > 0 ? loaded_gb / el : 0.0);
    (void)staged;
    ds4f_build_freqs(m);   /* RoPE/YaRN tables (only when exact) */
    ds4f_q8_promote_dense(m);   /* DS4F_Q8_DENSE: dense bf16-pv -> int8 W8A8 (no-op unless bf16-pv) */
    return m;
}

static ds4f_model *ds4f_load_real(ds4f_config cfg, int ep_rank, int ep_size,
                                  const char *blob_dir, int n_threads, int n_cmgs) {
    ds4f_runtime_options o = ds4f_runtime_options_debug_env(cfg, blob_dir, ep_rank, ep_size,
                                                             n_threads, n_cmgs);
    return ds4f_load_real_opts(&o);
}

/* ---- bf16 KV-cache codec. The model is natively bf16 (torch_dtype=bfloat16);
 * kv latents carry "massive activation" dims (~10^4..10^5) that overflow fp16's
 * 65504 ceiling -> Inf -> NaN (context/token-dependent). bf16 shares f32's 8-bit
 * exponent (no overflow) and matches the model's reference precision (8-bit
 * mantissa), so it is faithful, not lossy, for a bf16 model. Same 2 B/elem as fp16.
 * bf16->f32 is a 16-bit left shift; f32->bf16 is round-to-nearest-even. ---- */
static inline float    ds4f_bf16f(uint16_t b) { union { uint32_t u; float f; } z; z.u = (uint32_t)b << 16; return z.f; }
static inline uint16_t ds4f_f32bf(float x)     { union { uint32_t u; float f; } z; z.f = x; return (uint16_t)((z.u + 0x7fffu + ((z.u >> 16) & 1u)) >> 16); }

/* SVE attn-inner kernels (DS4F_ATTN_SVE). The decode attn worker was four SCALAR
 * loops (dot/axpy over kv_lora=q_head_dim=512) with per-element bf16 decode — ~0.2
 * GFLOP/s/core, scalar-issue/decode-bound, NOT bandwidth-bound (window=128 pos ~128KB
 * + ~512 compressed pos ~1MB, both L2-resident). bf16 widen idiom = svld1uh<<16, which
 * is BIT-EXACT to ds4f_bf16f (b<<16). The axpy keeps j-outer => per-lane accumulation
 * order matches scalar exactly (bit-exact PV); only the dot's horizontal reduction
 * reorders (tiny f32 reorder, argmax-safe; gated A/B confirms). */
#if defined(__ARM_FEATURE_SVE)
static inline float ds4f_sve_dot_bf16(const float *q, const uint16_t *k, int n) {
    svfloat32_t acc = svdup_f32(0.f);
    for (int d = 0; d < n; d += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(d, n);
        svfloat32_t kf = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, k + d), 16));
        acc = svmla_f32_x(pg, acc, svld1_f32(pg, q + d), kf);
    }
    return svaddv_f32(svptrue_b32(), acc);
}
static inline float ds4f_sve_dot_f32(const float *q, const float *k, int n) {
    svfloat32_t acc = svdup_f32(0.f);
    for (int d = 0; d < n; d += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(d, n);
        acc = svmla_f32_x(pg, acc, svld1_f32(pg, q + d), svld1_f32(pg, k + d));
    }
    return svaddv_f32(svptrue_b32(), acc);
}
static inline void ds4f_sve_axpy_bf16(float *out, const uint16_t *k, float w, int n) {
    svfloat32_t wv = svdup_f32(w);
    for (int d = 0; d < n; d += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(d, n);
        svfloat32_t kf = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, k + d), 16));
        svst1_f32(pg, out + d, svmla_f32_x(pg, svld1_f32(pg, out + d), kf, wv));
    }
}
static inline void ds4f_sve_axpy_f32(float *out, const float *k, float w, int n) {
    svfloat32_t wv = svdup_f32(w);
    for (int d = 0; d < n; d += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(d, n);
        svst1_f32(pg, out + d, svmla_f32_x(pg, svld1_f32(pg, out + d), svld1_f32(pg, k + d), wv));
    }
}
#endif /* __ARM_FEATURE_SVE */
static inline float ds4f_attn_dot_bf16(const float *q, const uint16_t *k,
                                       int n, int fast) {
    if (!fast) {
        float s = 0.f;
        for (int d = 0; d < n; d++) s += q[d] * ds4f_bf16f(k[d]);
        return s;
    }
#if defined(__ARM_FEATURE_SVE)
    return ds4f_sve_dot_bf16(q, k, n);
#elif defined(__AVX2__) && defined(__FMA__)
    return ds4f_avx2_dot_bf16(q, k, n);
#else
    float s = 0.f;
    for (int d = 0; d < n; d++) s += q[d] * ds4f_bf16f(k[d]);
    return s;
#endif
}

static inline void ds4f_attn_axpy_bf16(float *out, const uint16_t *k,
                                       float weight, int n, int fast) {
    if (!fast) {
        for (int d = 0; d < n; d++) out[d] += weight * ds4f_bf16f(k[d]);
        return;
    }
#if defined(__ARM_FEATURE_SVE)
    ds4f_sve_axpy_bf16(out, k, weight, n);
#elif defined(__AVX2__) && defined(__FMA__)
    ds4f_avx2_axpy_bf16(out, k, weight, n);
#else
    for (int d = 0; d < n; d++) out[d] += weight * ds4f_bf16f(k[d]);
#endif
}
/* ---- int8 KV (DS4F_INT8_KV) codec: per-channel-scale dequant dot/axpy --------------
 * The int8 store kv_q is dequantized in-lane as (int8 -> f32) * kv_scale[channel]. Same
 * SVE structure as the bf16 widen path (svld1uh<<16) but with svld1sb (sign-extend int8)
 * + svcvt + a per-channel scale multiply. The dot's horizontal reduction reorders (tiny
 * f32 reorder, argmax-safe like the bf16 dot); the axpy keeps j-outer so per-lane
 * accumulation order matches scalar. LOSSY by int8 (~1% rel) on top of that. */
#if defined(__ARM_FEATURE_SVE)
static inline float ds4f_sve_dot_i8s(const float *q, const int8_t *k, const float *sc, int n) {
    svfloat32_t acc = svdup_f32(0.f);
    for (int d = 0; d < n; d += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(d, n);
        svfloat32_t kf = svcvt_f32_s32_x(pg, svld1sb_s32(pg, k + d));
        kf = svmul_f32_x(pg, kf, svld1_f32(pg, sc + d));
        acc = svmla_f32_x(pg, acc, svld1_f32(pg, q + d), kf);
    }
    return svaddv_f32(svptrue_b32(), acc);
}
static inline void ds4f_sve_axpy_i8s(float *out, const int8_t *k, const float *sc, float w, int n) {
    svfloat32_t wv = svdup_f32(w);
    for (int d = 0; d < n; d += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(d, n);
        svfloat32_t kf = svcvt_f32_s32_x(pg, svld1sb_s32(pg, k + d));
        kf = svmul_f32_x(pg, kf, svld1_f32(pg, sc + d));
        svst1_f32(pg, out + d, svmla_f32_x(pg, svld1_f32(pg, out + d), kf, wv));
    }
}
#endif /* __ARM_FEATURE_SVE */
static inline float ds4f_scalar_dot_i8s(const float *q, const int8_t *k, const float *sc, int n) {
    float s = 0.f; for (int d = 0; d < n; d++) s += q[d] * ((float)k[d] * sc[d]); return s;
}
static inline void ds4f_scalar_axpy_i8s(float *out, const int8_t *k, const float *sc, float w, int n) {
    for (int d = 0; d < n; d++) out[d] += w * ((float)k[d] * sc[d]);
}
/* ---- int4 (DS4F_INT4_CMP) packed-nibble codec: 2 signed-4bit channels per byte; channel d in
 * byte d/2 (low nibble if d even, high if d odd), value range +/-7, dequant *cmp_scale[d]. */
#define DS4F_I4_SIGN(nb) (((int)(nb) ^ 8) - 8)   /* 4-bit two's-complement: 0..15 -> -8..7 */
/* unpack the packed byte vector b (vl lanes) -> two f32 vectors in CHANNEL order:
 * *c0 = channels [d..d+vl), *c1 = channels [d+vl..d+2vl). */
#if defined(__ARM_FEATURE_SVE)
static inline void ds4f_i4_unpack(svbool_t pg, svuint32_t b, svfloat32_t *c0, svfloat32_t *c1) {
    svint32_t lo = svsub_n_s32_x(pg, svreinterpret_s32_u32(sveor_n_u32_x(pg, svand_n_u32_x(pg, b, 0x0Fu), 8u)), 8);
    svint32_t hi = svsub_n_s32_x(pg, svreinterpret_s32_u32(sveor_n_u32_x(pg, svlsr_n_u32_x(pg, b, 4),      8u)), 8);
    svfloat32_t lof = svcvt_f32_s32_x(pg, lo);   /* even channels e0,e1,... */
    svfloat32_t hif = svcvt_f32_s32_x(pg, hi);   /* odd  channels o0,o1,... */
    *c0 = svzip1_f32(lof, hif);                  /* [e0,o0,e1,o1,...] = channels d..d+vl-1   */
    *c1 = svzip2_f32(lof, hif);                  /* channels d+vl..d+2vl-1                   */
}
static inline float ds4f_sve_dot_i4s(const float *q, const uint8_t *k4, const float *sc, int n) {
    svbool_t pt = svptrue_b32(); int vl = (int)svcntw();
    svfloat32_t acc = svdup_f32(0.f); int d = 0;
    for (; d + 2*vl <= n; d += 2*vl) {
        svfloat32_t c0, c1; ds4f_i4_unpack(pt, svld1ub_u32(pt, k4 + (d>>1)), &c0, &c1);
        c0 = svmul_f32_x(pt, c0, svld1_f32(pt, sc + d));
        c1 = svmul_f32_x(pt, c1, svld1_f32(pt, sc + d + vl));
        acc = svmla_f32_x(pt, acc, svld1_f32(pt, q + d), c0);
        acc = svmla_f32_x(pt, acc, svld1_f32(pt, q + d + vl), c1);
    }
    float s = svaddv_f32(pt, acc);
    for (; d < n; d++) { int nb = (d & 1) ? (k4[d>>1] >> 4) : (k4[d>>1] & 0xF); s += q[d] * ((float)DS4F_I4_SIGN(nb) * sc[d]); }
    return s;
}
static inline void ds4f_sve_axpy_i4s(float *out, const uint8_t *k4, const float *sc, float w, int n) {
    svbool_t pt = svptrue_b32(); int vl = (int)svcntw(); svfloat32_t wv = svdup_f32(w); int d = 0;
    for (; d + 2*vl <= n; d += 2*vl) {
        svfloat32_t c0, c1; ds4f_i4_unpack(pt, svld1ub_u32(pt, k4 + (d>>1)), &c0, &c1);
        c0 = svmul_f32_x(pt, c0, svld1_f32(pt, sc + d));
        c1 = svmul_f32_x(pt, c1, svld1_f32(pt, sc + d + vl));
        svst1_f32(pt, out + d,      svmla_f32_x(pt, svld1_f32(pt, out + d),      c0, wv));
        svst1_f32(pt, out + d + vl, svmla_f32_x(pt, svld1_f32(pt, out + d + vl), c1, wv));
    }
    for (; d < n; d++) { int nb = (d & 1) ? (k4[d>>1] >> 4) : (k4[d>>1] & 0xF); out[d] += w * ((float)DS4F_I4_SIGN(nb) * sc[d]); }
}
#endif /* __ARM_FEATURE_SVE */
static inline float ds4f_scalar_dot_i4s(const float *q, const uint8_t *k4, const float *sc, int n) {
    float s = 0.f;
    for (int d = 0; d < n; d++) { int nb = (d & 1) ? (k4[d>>1] >> 4) : (k4[d>>1] & 0xF); s += q[d] * ((float)DS4F_I4_SIGN(nb) * sc[d]); }
    return s;
}
static inline void ds4f_scalar_axpy_i4s(float *out, const uint8_t *k4, const float *sc, float w, int n) {
    for (int d = 0; d < n; d++) { int nb = (d & 1) ? (k4[d>>1] >> 4) : (k4[d>>1] & 0xF); out[d] += w * ((float)DS4F_I4_SIGN(nb) * sc[d]); }
}
/* quantize one f32 latent row -> int8 with per-channel iscale (clamp to +/-127). */
static inline void ds4f_kv_quant_row(const float *src, int8_t *dst, const float *iscale, int KV) {
    for (int d = 0; d < KV; d++) {
        int v = (int)lrintf(src[d] * iscale[d]);
        if (v > 127) v = 127; else if (v < -127) v = -127;
        dst[d] = (int8_t)v;
    }
}
/* finalize the per-channel scale from the accumulated absmax and quantize the bf16
 * calibration staging buffer into the int8 store. Called once per layer when the CAL-th
 * position arrives (streaming) or at warm time (all positions present). */
static void ds4f_kv_freeze(ds4f_layer *ly, int KV) {
    for (int d = 0; d < KV; d++) {
        float s = ly->kv_absmax[d] / 127.f; if (s < 1e-12f) s = 1e-12f;
        ly->kv_scale[d] = s; ly->kv_iscale[d] = 1.f / s;
    }
    for (int p = 0; p < ly->kv_caln; p++) {
        const uint16_t *src = ly->kv_calbuf + (size_t)p * KV;
        int8_t *dst = ly->kv_q + (size_t)p * KV;
        for (int d = 0; d < KV; d++) {
            int v = (int)lrintf(ds4f_bf16f(src[d]) * ly->kv_iscale[d]);
            if (v > 127) v = 127; else if (v < -127) v = -127;
            dst[d] = (int8_t)v;
        }
    }
    ly->kv_frozen = 1;
}
/* streaming append of one f32 latent at absolute `pos` into the int8 KV store. Stages to
 * the bf16 calbuf (accumulating per-channel absmax) until CAL positions are seen, then
 * freezes and quantizes directly. Positions arrive in order (pos == kv_caln pre-freeze). */
static inline void ds4f_kv_append_i8(ds4f_layer *ly, const float *lat, int pos, int KV, int CAL) {
    if (ly->kv_frozen) { ds4f_kv_quant_row(lat, ly->kv_q + (size_t)pos * KV, ly->kv_iscale, KV); return; }
    if (pos >= CAL) { ds4f_kv_freeze(ly, KV); ds4f_kv_quant_row(lat, ly->kv_q + (size_t)pos * KV, ly->kv_iscale, KV); return; }
    uint16_t *cb = ly->kv_calbuf + (size_t)pos * KV;
    for (int d = 0; d < KV; d++) {
        float v = lat[d]; cb[d] = ds4f_f32bf(v);
        float a = v < 0 ? -v : v; if (a > ly->kv_absmax[d]) ly->kv_absmax[d] = a;
    }
    ly->kv_caln = pos + 1;
}
/* ---- int8 compressed-latent (DS4F_INT8_CMP) codec: mirrors the int8 KV S5 scheme on
 * cmp_kv (slot-indexed). Reuses ds4f_kv_quant_row + the dot/axpy i8s kernels above. */
static void ds4f_cmp_freeze(ds4f_layer *ly, int KV) {
    for (int d = 0; d < KV; d++) {
        float s = ly->cmp_absmax[d] / 127.f; if (s < 1e-12f) s = 1e-12f;
        ly->cmp_scale[d] = s; ly->cmp_iscale[d] = 1.f / s;
    }
    for (int p = 0; p < ly->cmp_caln; p++) {
        const uint16_t *src = ly->cmp_calbuf + (size_t)p * KV;
        int8_t *dst = ly->cmp_q + (size_t)p * KV;
        for (int d = 0; d < KV; d++) {
            int v = (int)lrintf(ds4f_bf16f(src[d]) * ly->cmp_iscale[d]);
            if (v > 127) v = 127; else if (v < -127) v = -127;
            dst[d] = (int8_t)v;
        }
    }
    ly->cmp_frozen = 1;
}
/* streaming append of one f32 compressed latent at compressed `slot` into cmp_q. Stages to
 * cmp_calbuf (bf16, accumulating per-channel absmax) until CAL slots, then freezes &
 * quantizes directly. Slots arrive in order (slot == cmp_caln pre-freeze). */
static inline void ds4f_cmp_append_i8(ds4f_layer *ly, const float *lat, int slot, int KV, int CAL) {
    if (ly->cmp_frozen) { ds4f_kv_quant_row(lat, ly->cmp_q + (size_t)slot * KV, ly->cmp_iscale, KV); return; }
    if (slot >= CAL) { ds4f_cmp_freeze(ly, KV); ds4f_kv_quant_row(lat, ly->cmp_q + (size_t)slot * KV, ly->cmp_iscale, KV); return; }
    uint16_t *cb = ly->cmp_calbuf + (size_t)slot * KV;
    for (int d = 0; d < KV; d++) {
        float v = lat[d]; cb[d] = ds4f_f32bf(v);
        float a = v < 0 ? -v : v; if (a > ly->cmp_absmax[d]) ly->cmp_absmax[d] = a;
    }
    ly->cmp_caln = slot + 1;
}
/* ---- int4 cmp codec (DS4F_INT4_CMP): mirrors the int8 cmp functions above with +/-7 range,
 * 2 channels/byte packed into cmp_q4 (row stride KV/2). Reuses cmp_calbuf/absmax/caln/frozen. */
static inline void ds4f_cmp_quant_row_i4(const float *src, uint8_t *dst, const float *iscale, int KV) {
    for (int d = 0; d < KV; d += 2) {
        int v0 = (int)lrintf(src[d]   * iscale[d]);   if (v0 > 7) v0 = 7; else if (v0 < -7) v0 = -7;
        int v1 = (int)lrintf(src[d+1] * iscale[d+1]); if (v1 > 7) v1 = 7; else if (v1 < -7) v1 = -7;
        dst[d>>1] = (uint8_t)(((v1 & 0xF) << 4) | (v0 & 0xF));
    }
}
/* dequant one int4 cmp_q4 row -> f32 (for the CP gather: owner dequants its selected slots,
 * ar_cb-SUM assembles the full selected set on every node so attention reads it as f32). */
static inline void ds4f_cmp_deq_row_i4(const uint8_t *q4, const float *scale, int KV, float *out) {
    for (int d = 0; d < KV; d++) {
        int nb = (d & 1) ? (q4[d>>1] >> 4) : (q4[d>>1] & 0xF);
        out[d] = (float)(((nb ^ 8) - 8)) * scale[d];
    }
}
static void ds4f_cmp_freeze_i4(ds4f_layer *ly, int KV) {
    for (int d = 0; d < KV; d++) {
        float s = ly->cmp_absmax[d] / 7.f; if (s < 1e-12f) s = 1e-12f;
        ly->cmp_scale[d] = s; ly->cmp_iscale[d] = 1.f / s;
    }
    if (ly->cmp_caln > ly->cp_nslot) { fprintf(stderr, "[CP-DBG freeze OOB] caln=%d cap=%d\n", ly->cmp_caln, ly->cp_nslot); fflush(stderr); abort(); }
    for (int p = 0; p < ly->cmp_caln; p++) {
        const uint16_t *src = ly->cmp_calbuf + (size_t)p * KV;
        uint8_t *dst = ly->cmp_q4 + (size_t)p * (KV/2);
        for (int d = 0; d < KV; d += 2) {
            int v0 = (int)lrintf(ds4f_bf16f(src[d])   * ly->cmp_iscale[d]);   if (v0 > 7) v0 = 7; else if (v0 < -7) v0 = -7;
            int v1 = (int)lrintf(ds4f_bf16f(src[d+1]) * ly->cmp_iscale[d+1]); if (v1 > 7) v1 = 7; else if (v1 < -7) v1 = -7;
            dst[d>>1] = (uint8_t)(((v1 & 0xF) << 4) | (v0 & 0xF));
        }
    }
    ly->cmp_frozen = 1;
}
static inline void ds4f_cmp_append_i4(ds4f_layer *ly, const float *lat, int slot, int KV, int CAL) {
    /* CP: map global slot -> local cmp_q4 index. [0,CAL) replicated (all nodes, local==slot, for the
     * shared per-channel calibration); tail [CAL,nslot) sharded -> owner writes at CAL+(slot-cp_t0),
     * others skip (loc<0). When cp_on==0 loc==slot (unchanged). The freeze always quantizes [0,CAL). */
    int loc = slot;
    if (ly->cp_on && slot >= CAL) loc = (slot >= ly->cp_t0 && slot < ly->cp_t1) ? CAL + (slot - ly->cp_t0) : -1;
    if (loc >= ly->cp_nslot) { fprintf(stderr, "[CP-DBG append OOB] slot=%d loc=%d cap=%d t0=%d t1=%d CAL=%d caln=%d\n",
        slot, loc, ly->cp_nslot, ly->cp_t0, ly->cp_t1, CAL, ly->cmp_caln); fflush(stderr); abort(); }
    if (ly->cmp_frozen) { if (loc >= 0) ds4f_cmp_quant_row_i4(lat, ly->cmp_q4 + (size_t)loc * (KV/2), ly->cmp_iscale, KV); return; }
    if (slot >= CAL) { ds4f_cmp_freeze_i4(ly, KV); if (loc >= 0) ds4f_cmp_quant_row_i4(lat, ly->cmp_q4 + (size_t)loc * (KV/2), ly->cmp_iscale, KV); return; }
    uint16_t *cb = ly->cmp_calbuf + (size_t)slot * KV;
    for (int d = 0; d < KV; d++) {
        float v = lat[d]; cb[d] = ds4f_f32bf(v);
        float a = v < 0 ? -v : v; if (a > ly->cmp_absmax[d]) ly->cmp_absmax[d] = a;
    }
    ly->cmp_caln = slot + 1;
}
static int ds4f_attn_sve = -1;     /* DS4F_ATTN_SVE (default 1); 0 => scalar reference */

/* ===================== attention (pooled over heads) ===================== */
typedef struct { ds4f_model *m; ds4f_layer *ly; int pos; float scale; int ratio; } ds4f_attn_task;

static void ds4f_attn_worker(void *arg, int tid, int nthr) {
    ds4f_attn_task *T = (ds4f_attn_task *)arg;
    ds4f_model *m = T->m; ds4f_layer *ly = T->ly;
    int HD = m->cfg.q_head_dim, KV = m->cfg.kv_lora, nP = T->pos + 1;
    /* split heads across threads */
    int nh = m->cfg.n_heads; (void)nh;
    int h0, h1; ds4f_head_split(m, nthr, tid, &h0, &h1);        /* TP: only this node's owned heads */
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
                const uint16_t *kc = ly->kv_cache + (size_t)(b*R)*KV;
                float s = 0.f;
                for (int d = 0; d < idim; d++) s += q[d] * ds4f_bf16f(kc[d]);
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
                const uint16_t *kc = ly->kv_cache + (size_t)sel[j]*KV;
                float s = 0.f;
                for (int d = 0; d < KV; d++) s += q[d] * ds4f_bf16f(kc[d]);
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
                const uint16_t *kc = ly->kv_cache + (size_t)sel[j]*KV;
                for (int d = 0; d < KV; d++) out[d] += w * ds4f_bf16f(kc[d]);   /* V = latent */
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
            const uint16_t *kc = ly->kv_cache + (size_t)p*KV;
            float s = 0.f;
            /* score over the kv_lora latent dims (q_head_dim >= kv_lora; use first KV dims) */
            for (int d = 0; d < KV; d++) s += q[d] * ds4f_bf16f(kc[d]);
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
            const uint16_t *kc = ly->kv_cache + (size_t)p*KV;
            for (int d = 0; d < KV; d++) out[d] += w * ds4f_bf16f(kc[d]);   /* V = latent */
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
    int nh = m->cfg.n_heads; (void)nh;
    int h0, h1; ds4f_head_split(m, nthr, tid, &h0, &h1);        /* TP: only this node's owned heads */
    int sve = ds4f_attn_sve;
    int i8 = (m->int8_kv && ly->kv_frozen); const float *kvsc = ly->kv_scale;  /* int8 KV */
    /* pre-freeze (calibration window) the window latents live in the bf16 calbuf, not kv_cache
     * (NULL under int8_kv) nor kv_q (not yet quantized); freeze migrates calbuf->kv_q between
     * tokens, so every written pos is wholly in one store -- no straddle. */
    const uint16_t *kvbf = (m->int8_kv && !ly->kv_frozen) ? ly->kv_calbuf : ly->kv_cache;
    float *sc = (float *)alloca((size_t)nP * 4);
    for (int h = h0; h < h1; h++) {
        const float *q = m->s_q + (size_t)h*HD;
        float mx = -1e30f;
        for (int j = 0; j < nP; j++) {
            float s;                                                      /* HD==KV (512) */
            if (i8) { const int8_t *kc = ly->kv_q + (size_t)((p_lo + j) % ly->kv_slots)*KV;
                s = sve ? ds4f_sve_dot_i8s(q, kc, kvsc, KV) : ds4f_scalar_dot_i8s(q, kc, kvsc, KV); }
            else { const uint16_t *kc = kvbf + (size_t)((p_lo + j) % ly->kv_slots)*KV;
                s = ds4f_attn_dot_bf16(q, kc, KV, sve); }
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
            if (i8) { const int8_t *kc = ly->kv_q + (size_t)((p_lo + j) % ly->kv_slots)*KV;
                if (sve) ds4f_sve_axpy_i8s(out, kc, kvsc, w, HD); else ds4f_scalar_axpy_i8s(out, kc, kvsc, w, HD); }
            else { const uint16_t *kc = kvbf + (size_t)((p_lo + j) % ly->kv_slots)*KV;
                ds4f_attn_axpy_bf16(out, kc, w, HD, sve); }
        }
        ds4f_rope_apply(out + nope, T->rcos, T->rsin, pos, half, 1);  /* de-rotate */
    }
}

/* exact q: per-head RMS-normalize (no weight) over q_head_dim, then RoPE last rd. */
static void ds4f_q_norm_rope(ds4f_model *m, float *q, int pos,
                             const float *rcos, const float *rsin) {
    ds4f_config *c = &m->cfg;
    int HD = c->q_head_dim, rd = c->qk_rope_dim, half = rd/2, nope = HD - rd;
    for (int h = m->attn_h0; h < m->attn_h1; h++) {            /* TP: owned heads (skip q-norm of zero non-owned) */
        float *qh = q + (size_t)h*HD;
        double ss = 0.0; for (int d = 0; d < HD; d++) ss += (double)qh[d]*qh[d];
        float inv = 1.0f/sqrtf((float)(ss/HD) + c->norm_eps);
        for (int d = 0; d < HD; d++) qh[d] *= inv;
        ds4f_rope_apply(qh + nope, rcos, rsin, pos, half, 0);
    }
}
/* Parallel q-norm+RoPE: the n_heads heads are independent and write disjoint qh slices,
 * so splitting heads across threads is BIT-EXACT to the serial loop (same per-head double
 * accumulation order, same scale, same RoPE). Decode runs this serially on tid0 between the
 * wq_b/wkv pool dispatches; it was 7.18 ms/tok = 7.7% of decode @ctx10240 (64 heads * HD=512
 * scalar dot+scale * 43 layers, scalar-issue-bound). One pool_run amortizes that over 48 cores. */
typedef struct { ds4f_model *m; float *q; int pos; const float *rcos, *rsin; } ds4f_qnr_task;
static void ds4f_qnr_worker(void *arg, int tid, int nthr) {
    ds4f_qnr_task *T = (ds4f_qnr_task *)arg;
    ds4f_config *c = &T->m->cfg;
    int HD = c->q_head_dim, rd = c->qk_rope_dim, half = rd/2, nope = HD - rd;
    int h0, h1; ds4f_head_split(T->m, nthr, tid, &h0, &h1);    /* TP: owned heads only */
    for (int h = h0; h < h1; h++) {
        float *qh = T->q + (size_t)h*HD;
        double ss = 0.0; for (int d = 0; d < HD; d++) ss += (double)qh[d]*qh[d];
        float inv = 1.0f/sqrtf((float)(ss/HD) + c->norm_eps);
        for (int d = 0; d < HD; d++) qh[d] *= inv;
        ds4f_rope_apply(qh + nope, T->rcos, T->rsin, T->pos, half, 0);
    }
}
static int ds4f_qnr_par = -1;        /* DS4F_QNR_PAR: 1=pool-parallel (default), 0=serial ref */
static void ds4f_q_norm_rope_par(ds4f_model *m, float *q, int pos,
                                 const float *rcos, const float *rsin) {
    ds4f_qnr_task T = { m, q, pos, rcos, rsin };
    ds4f_pool_run(m->pool, ds4f_qnr_worker, &T);
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
    int nh = m->cfg.n_heads; (void)nh;
    int h0, h1; ds4f_head_split(m, nthr, tid, &h0, &h1);        /* TP: only this node's owned heads */
    int sve = ds4f_attn_sve;
    int i8 = (m->int8_kv && ly->kv_frozen); const float *kvsc = ly->kv_scale;  /* int8 KV */
    /* pre-freeze (calibration window) the window latents live in the bf16 calbuf, not kv_cache
     * (NULL under int8_kv) nor kv_q (not yet quantized); freeze migrates calbuf->kv_q between
     * tokens, so every written pos is wholly in one store -- no straddle. */
    const uint16_t *kvbf = (m->int8_kv && !ly->kv_frozen) ? ly->kv_calbuf : ly->kv_cache;
    int c_i4 = (m->int4_cmp && ly->cmp_frozen);                                       /* int4 cmp (cmp_q4) */
    int c_cp = m->cp_gather;   /* CP: compressed term reads the gathered f32 latents s_cmp_gather[j*KV] */
    const float *cg = m->s_cmp_gather;
    int c_i8 = (m->int8_cmp && !m->int4_cmp && ly->cmp_frozen); const float *cmpsc = ly->cmp_scale;  /* int8 cmp */
    /* pre-freeze the compressed latents live in cmp_calbuf (bf16); freeze migrates
     * calbuf->cmp_q between tokens, so every slot is wholly in one store -- no straddle. */
    const uint16_t *cmpbf = (m->int8_cmp && !ly->cmp_frozen) ? ly->cmp_calbuf : NULL;
    float *sc = (float *)alloca((size_t)(total > 0 ? total : 1) * 4);
    for (int h = h0; h < h1; h++) {
        const float *q = m->s_q + (size_t)h*HD;
        float mx = -1e30f;
        for (int j = 0; j < nP; j++) {                          /* window term (kv_cache) */
            float s;
            if (i8) { const int8_t *kc = ly->kv_q + (size_t)((p_lo + j) % ly->kv_slots)*KV;
                s = sve ? ds4f_sve_dot_i8s(q, kc, kvsc, KV) : ds4f_scalar_dot_i8s(q, kc, kvsc, KV); }
            else { const uint16_t *kc = kvbf + (size_t)((p_lo + j) % ly->kv_slots)*KV;
                s = ds4f_attn_dot_bf16(q, kc, KV, sve); }
            s *= T->scale; sc[j] = s; if (s > mx) mx = s;
        }
        for (int j = 0; j < nsel; j++) {                        /* compressed term (cmp_kv) */
            float s;
            if (c_cp) { const float *kc = cg + (size_t)j*KV;   /* CP: gathered f32 latent (index j, not sel[j]) */
                if (sve) s = ds4f_sve_dot_f32(q, kc, KV); else { s = 0.f; for (int d = 0; d < KV; d++) s += q[d]*kc[d]; } }
            else if (c_i4) { const uint8_t *kc = ly->cmp_q4 + (size_t)sel[j]*(KV/2);
                s = sve ? ds4f_sve_dot_i4s(q, kc, cmpsc, KV) : ds4f_scalar_dot_i4s(q, kc, cmpsc, KV); }
            else if (c_i8) { const int8_t *kc = ly->cmp_q + (size_t)sel[j]*KV;
                s = sve ? ds4f_sve_dot_i8s(q, kc, cmpsc, KV) : ds4f_scalar_dot_i8s(q, kc, cmpsc, KV); }
            else if (cmpbf) { const uint16_t *kc = cmpbf + (size_t)sel[j]*KV;
                s = ds4f_attn_dot_bf16(q, kc, KV, sve); }
            else { const float *kc = cmp + (size_t)sel[j]*KV;
                if (sve) s = ds4f_sve_dot_f32(q, kc, KV);
                else { s = 0.f; for (int d = 0; d < KV; d++) s += q[d]*kc[d]; } }
            s *= T->scale; sc[nP + j] = s; if (s > mx) mx = s;
        }
        float denom = expf(ly->attn_sink[h] - mx);
        for (int j = 0; j < total; j++) { sc[j] = expf(sc[j] - mx); denom += sc[j]; }
        float inv = 1.0f/denom;
        float *out = m->s_attn + (size_t)h*HD;
        for (int d = 0; d < HD; d++) out[d] = 0.f;
        for (int j = 0; j < nP; j++) {
            float w = sc[j]*inv;
            if (i8) { const int8_t *kc = ly->kv_q + (size_t)((p_lo + j) % ly->kv_slots)*KV;
                if (sve) ds4f_sve_axpy_i8s(out, kc, kvsc, w, HD); else ds4f_scalar_axpy_i8s(out, kc, kvsc, w, HD); }
            else { const uint16_t *kc = kvbf + (size_t)((p_lo + j) % ly->kv_slots)*KV;
                ds4f_attn_axpy_bf16(out, kc, w, HD, sve); }
        }
        for (int j = 0; j < nsel; j++) {
            float w = sc[nP + j]*inv;
            if (c_cp) { const float *kc = cg + (size_t)j*KV;
                if (sve) ds4f_sve_axpy_f32(out, kc, w, HD); else for (int d = 0; d < HD; d++) out[d] += w*kc[d]; }
            else if (c_i4) { const uint8_t *kc = ly->cmp_q4 + (size_t)sel[j]*(KV/2);
                if (sve) ds4f_sve_axpy_i4s(out, kc, cmpsc, w, HD); else ds4f_scalar_axpy_i4s(out, kc, cmpsc, w, HD); }
            else if (c_i8) { const int8_t *kc = ly->cmp_q + (size_t)sel[j]*KV;
                if (sve) ds4f_sve_axpy_i8s(out, kc, cmpsc, w, HD); else ds4f_scalar_axpy_i8s(out, kc, cmpsc, w, HD); }
            else if (cmpbf) { const uint16_t *kc = cmpbf + (size_t)sel[j]*KV;
                if (sve) ds4f_sve_axpy_bf16(out, kc, w, HD);
                else for (int d = 0; d < HD; d++) out[d] += w*ds4f_bf16f(kc[d]); }
            else { const float *kc = cmp + (size_t)sel[j]*KV;
                if (sve) ds4f_sve_axpy_f32(out, kc, w, HD);
                else for (int d = 0; d < HD; d++) out[d] += w*kc[d]; }
        }
        ds4f_rope_apply(out + nope, T->rcos, T->rsin, pos, T->half, 1);  /* de-rotate @ query pos */
    }
}

/* DS4F_ATTN_GEMM: reuse each BF16 window/compressed latent across an eight-head
 * block. This is lossless relative to the existing SVE attention math; it
 * changes only the load schedule and keeps the per-head score/value order. */
#define DS4F_ATTN_GEMM_MR 8
#define DS4F_ATTN_GEMM_DBLK 64
#if defined(__ARM_FEATURE_SVE)
static inline void ds4f_score8_bf16(float s[8], const float *q, int qs, const uint16_t *kv, int K) {
    svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
    for (int d=0; d<K; d+=(int)svcntw()) { svbool_t p=svwhilelt_b32(d,K);
        svfloat32_t v=svreinterpret_f32_u32(svlsl_n_u32_x(p,svld1uh_u32(p,kv+d),16));
        a0=svmla_f32_x(p,a0,svld1_f32(p,q+0*qs+d),v); a1=svmla_f32_x(p,a1,svld1_f32(p,q+1*qs+d),v);
        a2=svmla_f32_x(p,a2,svld1_f32(p,q+2*qs+d),v); a3=svmla_f32_x(p,a3,svld1_f32(p,q+3*qs+d),v);
        a4=svmla_f32_x(p,a4,svld1_f32(p,q+4*qs+d),v); a5=svmla_f32_x(p,a5,svld1_f32(p,q+5*qs+d),v);
        a6=svmla_f32_x(p,a6,svld1_f32(p,q+6*qs+d),v); a7=svmla_f32_x(p,a7,svld1_f32(p,q+7*qs+d),v); }
    svbool_t p=svptrue_b32();
    s[0]=svaddv_f32(p,a0); s[1]=svaddv_f32(p,a1); s[2]=svaddv_f32(p,a2); s[3]=svaddv_f32(p,a3);
    s[4]=svaddv_f32(p,a4); s[5]=svaddv_f32(p,a5); s[6]=svaddv_f32(p,a6); s[7]=svaddv_f32(p,a7);
}
static inline void ds4f_axpy8_bf16(float *out, int os, const uint16_t *kv, const float w[8], int n) {
    for (int d=0; d<n; d+=(int)svcntw()) { svbool_t p=svwhilelt_b32(d,n);
        svfloat32_t v=svreinterpret_f32_u32(svlsl_n_u32_x(p,svld1uh_u32(p,kv+d),16));
        for (int h=0; h<8; h++) { float *o=out+(size_t)h*os+d;
            svst1_f32(p,o,svmla_f32_x(p,svld1_f32(p,o),v,svdup_f32(w[h]))); } }
}
static inline void ds4f_score8_f32(float s[8], const float *q, int qs, const float *kv, int K) {
    svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
    for (int d=0; d<K; d+=(int)svcntw()) { svbool_t p=svwhilelt_b32(d,K),v=svwhilelt_b32(d,K);
        svfloat32_t x=svld1_f32(p,kv+d);
        a0=svmla_f32_x(p,a0,svld1_f32(v,q+0*qs+d),x); a1=svmla_f32_x(p,a1,svld1_f32(v,q+1*qs+d),x);
        a2=svmla_f32_x(p,a2,svld1_f32(v,q+2*qs+d),x); a3=svmla_f32_x(p,a3,svld1_f32(v,q+3*qs+d),x);
        a4=svmla_f32_x(p,a4,svld1_f32(v,q+4*qs+d),x); a5=svmla_f32_x(p,a5,svld1_f32(v,q+5*qs+d),x);
        a6=svmla_f32_x(p,a6,svld1_f32(v,q+6*qs+d),x); a7=svmla_f32_x(p,a7,svld1_f32(v,q+7*qs+d),x); }
    svbool_t p=svptrue_b32();
    s[0]=svaddv_f32(p,a0); s[1]=svaddv_f32(p,a1); s[2]=svaddv_f32(p,a2); s[3]=svaddv_f32(p,a3);
    s[4]=svaddv_f32(p,a4); s[5]=svaddv_f32(p,a5); s[6]=svaddv_f32(p,a6); s[7]=svaddv_f32(p,a7);
}
static inline void ds4f_axpy8_f32(float *out, int os, const float *kv, const float w[8], int n) {
    for (int d=0; d<n; d+=(int)svcntw()) { svbool_t p=svwhilelt_b32(d,n),v=svwhilelt_b32(d,n);
        svfloat32_t x=svld1_f32(p,kv+d);
        for (int h=0;h<8;h++) { float *o=out+(size_t)h*os+d;
            svst1_f32(p,o,svmla_f32_x(v,svld1_f32(v,o),x,svdup_f32(w[h]))); } }
}
#endif /* __ARM_FEATURE_SVE */
static int ds4f_attn_gemm = -1;
typedef struct { ds4f_model *m; ds4f_layer *ly; int pos,nP,p_lo,nsel,total,stride;
                 float scale,half; const float *rcos,*rsin; } ds4f_attn_gemm_task;
static void ds4f_attn_gemm_score_worker(void *arg,int tid,int nthr) {
    ds4f_attn_gemm_task *T=(ds4f_attn_gemm_task *)arg; ds4f_model *m=T->m; ds4f_layer *ly=T->ly;
    int HD=m->cfg.q_head_dim,KV=m->cfg.kv_lora,nhb=(m->attn_h1-m->attn_h0)/8;
    int nP=T->nP,total=T->total,stride=T->stride,p_lo=T->p_lo; const int *sel=m->s_tb2_sel;
    long work=(long)nhb*total,per=work/nthr,ex=work%nthr,u0=per*tid+(tid<ex?tid:ex),u1=u0+per+(tid<ex?1:0);
    for(long u=u0;u<u1;u++){int hb=(int)(u/total),j=(int)(u%total),h0=m->attn_h0+hb*8;float s[8];
        if(j<nP) ds4f_score8_bf16(s,m->s_q+(size_t)h0*HD,HD,ly->kv_cache+(size_t)((p_lo+j)%ly->kv_slots)*KV,KV);
        else ds4f_score8_f32(s,m->s_q+(size_t)h0*HD,HD,ly->cmp_kv+(size_t)sel[j-nP]*KV,KV);
        for(int h=0;h<8;h++)m->s_attn_sc[(size_t)(hb*8+h)*stride+j]=s[h]*T->scale;}
}
static void ds4f_attn_gemm_soft_worker(void *arg,int tid,int nthr) {
    ds4f_attn_gemm_task *T=(ds4f_attn_gemm_task *)arg; ds4f_model *m=T->m; ds4f_layer *ly=T->ly;
    int nh=m->attn_h1-m->attn_h0,per=nh/nthr,ex=nh%nthr,h0=per*tid+(tid<ex?tid:ex),h1=h0+per+(tid<ex?1:0);
    for(int hi=h0;hi<h1;hi++){float *sc=m->s_attn_sc+(size_t)hi*T->stride,mx=-1e30f;
        for (int j = 0; j < T->total; j++) if (sc[j] > mx) mx = sc[j];
        float den = expf(ly->attn_sink[m->attn_h0 + hi] - mx);
        for(int j=0;j<T->total;j++){float e=expf(sc[j]-mx);sc[j]=e;den+=e;}float inv=1.0f/den;for(int j=0;j<T->total;j++)sc[j]*=inv;}
}
static void ds4f_attn_gemm_axpy_worker(void *arg,int tid,int nthr) {
    ds4f_attn_gemm_task *T=(ds4f_attn_gemm_task *)arg; ds4f_model *m=T->m; ds4f_layer *ly=T->ly;
    int HD=m->cfg.q_head_dim,KV=m->cfg.kv_lora,nhb=(m->attn_h1-m->attn_h0)/8,nblk=HD/64,nope=HD-m->cfg.qk_rope_dim;
    long work=(long)nhb*nblk,per=work/nthr,ex=work%nthr,u0=per*tid+(tid<ex?tid:ex),u1=u0+per+(tid<ex?1:0);float w[8];const int *sel=m->s_tb2_sel;
    for(long u=u0;u<u1;u++){int hb=(int)(u/nblk),b=(int)(u%nblk),h0=m->attn_h0+hb*8,d0=b*64;float *out=m->s_attn+(size_t)h0*HD+d0;
        for(int h=0;h<8;h++)for(int d=0;d<64;d++)out[(size_t)h*HD+d]=0;
        for(int j=0;j<T->nP;j++){for(int h=0;h<8;h++)w[h]=m->s_attn_sc[(size_t)(h0-m->attn_h0+h)*T->stride+j];
            ds4f_axpy8_bf16(out,HD,ly->kv_cache+(size_t)((T->p_lo+j)%ly->kv_slots)*KV+d0,w,64);}
        for(int j=0;j<T->nsel;j++){for(int h=0;h<8;h++)w[h]=m->s_attn_sc[(size_t)(h0-m->attn_h0+h)*T->stride+T->nP+j];
            ds4f_axpy8_f32(out,HD,ly->cmp_kv+(size_t)sel[j]*KV+d0,w,64);}
        if(d0==nope)for(int h=0;h<8;h++)ds4f_rope_apply(out+(size_t)h*HD,T->rcos,T->rsin,T->pos,T->half,1);}
}
static int ds4f_attn_tb2_gemm(ds4f_model *m,ds4f_attn_ex_task *at) {
    if(ds4f_attn_gemm<0){const char *e=getenv("DS4F_ATTN_GEMM");ds4f_attn_gemm=e?atoi(e):1;}
    ds4f_config *c=&m->cfg;ds4f_layer *ly=at->ly;int HD=c->q_head_dim,nope=HD-c->qk_rope_dim,nh=m->attn_h1-m->attn_h0;
    if(!ds4f_attn_gemm||!ds4f_attn_sve||m->int8_kv||m->int8_cmp||m->int4_cmp||m->cp_gather||ly->kv_frozen||ly->cmp_frozen)return 0;
    if(nh%8||c->qk_rope_dim!=64||HD%64||nope%64)return 0;
    int p=at->pos,lo=p-at->win+1;if(lo<0)lo=0;int nP=p-lo+1,nsel=m->s_tb2_nsel,stride=c->window_size+c->index_topk;
    if(!m->s_attn_sc)m->s_attn_sc=(float *)ds4f_mem_alloc(m->mem,((size_t)c->n_heads*stride*4+255)&~(size_t)255,256,1);
    ds4f_attn_gemm_task T={m,ly,p,nP,lo,nsel,nP+nsel,stride,at->scale,(float)(c->qk_rope_dim/2),at->rcos,at->rsin};
    ds4f_pool_run(m->pool,ds4f_attn_gemm_score_worker,&T);ds4f_pool_run(m->pool,ds4f_attn_gemm_soft_worker,&T);ds4f_pool_run(m->pool,ds4f_attn_gemm_axpy_worker,&T);return 1;
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
    double _tlc0 = ds4f_now();
    if (ds4f_compress_step(m->s_hn, c->hidden, KV, rd, ratio, pos,
                           ly->cmp_wkv, ly->cmp_wgate, 1, ly->cmp_ape, ly->cmp_norm,
                           rcos, rsin, eps, 0,
                           ly->cmp_kv_state, ly->cmp_score_state, m->s_cmp_out, m->pool)) {
        if (m->int4_cmp) ds4f_cmp_append_i4(ly, m->s_cmp_out, pos/ratio, KV, ds4f_int8cmp_cal);
        else if (m->int8_cmp) ds4f_cmp_append_i8(ly, m->s_cmp_out, pos/ratio, KV, ds4f_int8cmp_cal);
        else memcpy(ly->cmp_kv + (size_t)(pos/ratio)*KV, m->s_cmp_out, (size_t)KV*4);
    }
    m->prof[DS4F_P_TB2LCMP] += ds4f_now() - _tlc0;
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
        double _sc_snap = ds4f_g_tb2scan, _qp_snap = ds4f_g_tb2qproj, _rp_snap = ds4f_g_tb2rope,
               _ic_snap = ds4f_g_tb2icmp, _wp_snap = ds4f_g_tb2wproj, _tk_snap = ds4f_g_tb2topk;
        ds4f_index_step(m->s_hn, c->hidden, m->s_qlat, m->s_idx_qpre, c->q_lora,
                        c->index_n_heads, ihd, rd, ratio, pos, offset, k,
                        ly->idx_wq_b, ly->idx_wproj, 1,
                        ly->idx_cmp_wkv, ly->idx_cmp_wgate, ly->idx_cmp_ape, ly->idx_cmp_norm,
                        rcos, rsin, eps,
                        ly->idx_cmp_kv_state, ly->idx_cmp_score_state, ly->idx_kv,
                        ly->idx_kv8, ly->idx_kv8_4, ly->idx_pscale,
                        m->s_idx_q, m->s_idx_score, m->s_tb2_sel, m->pool,
                        ly->idx_cp_on, ly->idx_cp_s0, ly->idx_cp_s1, m->ar_cb, m->ar_ctx,
                        m->ep_rank, m->ep_size, m->s_cp_cand_slot, m->s_cp_cand_score);
        m->prof[DS4F_P_TB2SCAN]  += ds4f_g_tb2scan  - _sc_snap;
        m->prof[DS4F_P_TB2QPROJ] += ds4f_g_tb2qproj - _qp_snap;
        m->prof[DS4F_P_TB2ROPE]  += ds4f_g_tb2rope  - _rp_snap;
        m->prof[DS4F_P_TB2ICMP]  += ds4f_g_tb2icmp  - _ic_snap;
        m->prof[DS4F_P_TB2WPROJ] += ds4f_g_tb2wproj - _wp_snap;
        m->prof[DS4F_P_TB2TOPK]  += ds4f_g_tb2topk  - _tk_snap;
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

static void ds4f_route_observe(ds4f_model *m, int layer, const int *idx, int n) {
    if (!m || !m->route_telemetry) return;
    if (!m->route_hits) {
        size_t nh = (size_t)m->cfg.n_layers * (size_t)m->cfg.n_experts;
        m->route_hits = (uint64_t *)ds4f_mem_alloc(m->mem, nh * sizeof(uint64_t), 256, 1);
        m->route_tokens = (uint64_t *)ds4f_mem_alloc(
            m->mem, (size_t)m->cfg.n_layers * sizeof(uint64_t), 256, 1);
        if (!m->route_hits || !m->route_tokens) return;
    }
    m->route_tokens[layer]++;
    uint64_t *hits = m->route_hits + (size_t)layer * m->cfg.n_experts;
    for (int i = 0; i < n; ++i)
        if (idx[i] >= 0 && idx[i] < m->cfg.n_experts) hits[idx[i]]++;
}

static void ds4f_route_report(const ds4f_model *m, FILE *out) {
    if (!m || !m->route_hits || !m->route_tokens || !out) return;
    int E = m->cfg.n_experts, A = m->cfg.n_active;
    fprintf(out, "route telemetry: layers=%d experts=%d topk=%d\n",
            m->cfg.n_layers, E, A);
    for (int L = 0; L < m->cfg.n_layers; ++L) {
        const uint64_t *h = m->route_hits + (size_t)L * E;
        uint64_t total = m->route_tokens[L] * (uint64_t)A;
        if (!total) continue;
        uint64_t covered = 0;
        int chosen[8] = {0};
        fprintf(out, "  L%02d tokens=%llu hot=", L,
                (unsigned long long)m->route_tokens[L]);
        for (int k = 0; k < 8 && k < E; ++k) {
            int best = -1;
            for (int e = 0; e < E; ++e) {
                int used = 0;
                for (int j = 0; j < k; ++j) if (chosen[j] == e) used = 1;
                if (!used && (best < 0 || h[e] > h[best])) best = e;
            }
            chosen[k] = best; covered += h[best];
            fprintf(out, "%s%d:%llu", k ? "," : "", best,
                    (unsigned long long)h[best]);
        }
        fprintf(out, " hot8_coverage=%.2f%%\n", 100.0 * (double)covered / (double)total);
    }
}

/* ===================== synthetic KV warm (ctx benchmark) =====================
 * Fill every layer's KV cache positions [0,npos) with bounded synthetic latents
 * so decode-attn cost at a large context can be measured WITHOUT running npos
 * real prefill tokens. After this, decode at pos>=npos. Synthetic-only: the
 * latent values are deterministic junk; the point is the position COUNT that
 * the attention loop (dense O(nP) vs sparse O(topk)) iterates over. */
/* RSS in GB (resident pages of this process). For warm-phase per-layer tracing. */
static double ds4f_rss_gb(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    long total = 0, res = 0; if (f) { if (fscanf(f, "%ld %ld", &total, &res) != 2) res = 0; fclose(f); }
    long pg = sysconf(_SC_PAGESIZE); if (pg <= 0) pg = 4096;
    return (double)res * (double)pg / 1e9;
}
/* node-level MemAvailable in GB (whole-node free estimate; catches NUMA/co-tenant/kernel
 * pressure that per-process RSS misses). Returns -1 if unreadable. */
static double ds4f_mem_avail_gb(void) {
    FILE *f = fopen("/proc/meminfo", "r"); if (!f) return -1.0;
    char line[128]; double gb = -1.0;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "MemAvailable:", 13)) { long kb = 0; sscanf(line+13, "%ld", &kb); gb = (double)kb/1e6; break; }
    fclose(f); return gb;
}
/* MemFree = true physical free (vs MemAvailable's reclaim estimate). Distinguishes
 * physically-resident allocations (OOM-relevant) from committed-not-resident ones. */
static double ds4f_mem_free_gb(void) {
    FILE *f = fopen("/proc/meminfo", "r"); if (!f) return -1.0;
    char line[128]; double gb = -1.0;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "MemFree:", 8)) { long kb = 0; sscanf(line+8, "%ld", &kb); gb = (double)kb/1e6; break; }
    fclose(f); return gb;
}

static void ds4f_warm_kv(ds4f_model *m, int npos) {
    int KV = m->cfg.kv_lora;
    if (npos > m->cfg.max_pos) npos = m->cfg.max_pos;
    int CAL = ds4f_int8kv_cal;
    int rss_trace = 0; { const char *e = getenv("DS4F_WARM_RSS_TRACE"); rss_trace = (e && *e && atoi(e)); }
    /* Safe ceiling: when RSS exceeds DS4F_WARM_RSS_STOP_GB (per-layer check, on EVERY rank),
     * exit CLEANLY (_exit(42)) instead of letting the OOM-killer SIGKILL the rank (which
     * degrades PMIx and loses the alloc). One layer adds <=0.14 GB @256k so a 29.5 limit
     * peaks <=29.7 GB << 31.8 node. Reveals how many layers of kv_q fit = real per-layer cost. */
    double rss_stop = 0.0; { const char *e = getenv("DS4F_WARM_RSS_STOP_GB"); if (e && *e) rss_stop = atof(e); }
    double avail_stop = 1.5; { const char *e = getenv("DS4F_WARM_MEMAVAIL_STOP_GB"); if (e && *e) avail_stop = atof(e); }
    int rss_guard = (rss_stop > 0.0) || rss_trace;
    #define DS4F_WARMKV_NEXT() ( s += 0x9E3779B97F4A7C15ull, \
        z = s, z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull, \
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull, z ^= (z >> 31), \
        (float)((double)(z >> 11) / (double)(1ull << 53)) * 2.0f - 1.0f )
    for (int L = 0; L < m->cfg.n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        uint64_t s = 0x4B7700D5F0ull ^ ((uint64_t)L << 40), z;
        if (m->int8_kv) {
            for (int d = 0; d < KV; d++) ly->kv_absmax[d] = 0.f;
            int caln = npos < CAL ? npos : CAL;
            for (int p = 0; p < caln; p++) {           /* stage calibration window (bf16) + absmax */
                uint16_t *row = ly->kv_calbuf + (size_t)p*KV;
                for (int d = 0; d < KV; d++) {
                    float v = DS4F_WARMKV_NEXT(); row[d] = ds4f_f32bf(v);
                    float a = v < 0 ? -v : v; if (a > ly->kv_absmax[d]) ly->kv_absmax[d] = a;
                }
            }
            ly->kv_caln = caln;
            if (npos >= CAL) {                         /* full window -> freeze + quantize the rest int8 */
                ds4f_kv_freeze(ly, KV);
                for (int p = CAL; p < npos; p++) {
                    int8_t *row = ly->kv_q + (size_t)p*KV;
                    for (int d = 0; d < KV; d++) {
                        int iv = (int)lrintf(DS4F_WARMKV_NEXT() * ly->kv_iscale[d]);
                        if (iv > 127) iv = 127; else if (iv < -127) iv = -127; row[d] = (int8_t)iv;
                    }
                }
            }
        } else {
            uint16_t *kc = ly->kv_cache;
            /* ring buffer: sparse layers hold only kv_slots(=window_size) slots; fill just
             * those (the decode at pos=npos reads the window ((pos-w+1..pos) % kv_slots),
             * all of which land in [0,kv_slots)). Dense layers have kv_slots==max_pos => npos. */
            int nfill = (npos < ly->kv_slots) ? npos : ly->kv_slots;
            for (int p = 0; p < nfill; p++) {
                uint16_t *row = kc + (size_t)p*KV;
                for (int d = 0; d < KV; d++) row[d] = ds4f_f32bf(DS4F_WARMKV_NEXT());
            }
        }
        if (rss_guard) {
            double r = ds4f_rss_gb(), avail = ds4f_mem_avail_gb();
            if (rss_trace && m->ep_rank == 0 && (L % 4 == 0 || L == m->cfg.n_layers - 1))
                fprintf(stderr, "[warmkv L=%2d/%d RSS=%.2f GB MemAvail=%.2f GB MemFree=%.2f GB]\n",
                        L, m->cfg.n_layers, r, avail, ds4f_mem_free_gb());
            /* clean-exit on EITHER self-RSS over stop OR node MemAvailable too low (the
             * latter catches whatever consumes the node beyond this rank's RSS). */
            if ((rss_stop > 0.0 && r > rss_stop) || (avail >= 0.0 && avail < avail_stop)) {
                if (m->ep_rank == 0)
                    fprintf(stderr, "[warmkv CEILING HIT L=%d/%d RSS=%.2f MemAvail=%.2f "
                            "(rss_stop=%.2f avail_stop=%.2f) kv_q layers filled=%d => clean exit(42)]\n",
                            L, m->cfg.n_layers, r, avail, rss_stop, avail_stop, L+1);
                fflush(stderr); _exit(42);
            }
        }
    }
    if (rss_trace && m->ep_rank == 0)
        fprintf(stderr, "[warmkv DONE RSS=%.2f GB]\n", ds4f_rss_gb());
    #undef DS4F_WARMKV_NEXT
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
    float *crow = m->int8_cmp ? (float *)alloca((size_t)KV*4) : NULL;  /* int8 cmp staging */
    int rss_trace = 0; { const char *e = getenv("DS4F_WARM_RSS_TRACE"); rss_trace = (e && *e && atoi(e)); }
    double rss_stop = 0.0; { const char *e = getenv("DS4F_WARM_RSS_STOP_GB"); if (e && *e) rss_stop = atof(e); }
    double avail_stop = 1.5; { const char *e = getenv("DS4F_WARM_MEMAVAIL_STOP_GB"); if (e && *e) avail_stop = atof(e); }
    int rss_guard = (rss_stop > 0.0) || rss_trace;
    for (int L = 0; L < c->n_layers; L++) {
        int ratio = c->compress_ratios[L];
        if (ratio == 0) continue;
        ds4f_layer *ly = &m->layers[L];
        if (rss_guard) {
            double r = ds4f_rss_gb(), avail = ds4f_mem_avail_gb();
            if (rss_trace && m->ep_rank == 0 && (L % 4 == 0 || L == c->n_layers - 1))
                fprintf(stderr, "[warmtb2 L=%2d/%d RSS=%.2f GB MemAvail=%.2f GB MemFree=%.2f GB]\n",
                        L, c->n_layers, r, avail, ds4f_mem_free_gb());
            if ((rss_stop > 0.0 && r > rss_stop) || (avail >= 0.0 && avail < avail_stop)) {
                if (m->ep_rank == 0)
                    fprintf(stderr, "[warmtb2 CEILING HIT L=%d/%d RSS=%.2f MemAvail=%.2f (rss_stop=%.2f avail_stop=%.2f) => clean exit(42)]\n",
                            L, c->n_layers, r, avail, rss_stop, avail_stop);
                fflush(stderr); _exit(42);
            }
        }
        int nslot = c->max_pos / ratio;
        int T = (npos + 1) / ratio; if (T > nslot) T = nslot;
        uint64_t s = 0xC0FFEE5F0ull ^ ((uint64_t)L << 40);
        for (int t = 0; t < T; t++) {
            float *kr = m->int8_cmp ? crow : (ly->cmp_kv + (size_t)t*KV);
            for (int d = 0; d < KV; d++) {
                s += 0x9E3779B97F4A7C15ull; uint64_t z = s;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= (z >> 31);
                kr[d] = (float)((double)(z >> 11) / (double)(1ull << 53)) * 2.0f - 1.0f;
            }
            if (m->int4_cmp) ds4f_cmp_append_i4(ly, kr, t, KV, ds4f_int8cmp_cal);
            else if (m->int8_cmp) ds4f_cmp_append_i8(ly, kr, t, KV, ds4f_int8cmp_cal);
            if (ratio == 4) {                                  /* indexer compressed key */
                /* idx_int8: f32 idx_kv is only DS4F_IDX_F32_SLOTS slots -> ring the synthetic
                 * write (content is junk; the warmed decode at high ctx reads int8 idx_kv8). */
                int idx_int = (ly->idx_kv8 || ly->idx_kv8_4);  /* int8 or int4 replacement => f32 idx_kv is ringed */
                float *ir = ly->idx_kv + (size_t)(idx_int ? (t % DS4F_IDX_F32_SLOTS) : t)*ihd;
                for (int d = 0; d < ihd; d++) {
                    s += 0x9E3779B97F4A7C15ull; uint64_t z = s;
                    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= (z >> 31);
                    ir[d] = (float)((double)(z >> 11) / (double)(1ull << 53)) * 2.0f - 1.0f;
                }
                if (ly->idx_kv8_4) { int il = ly->idx_cp_on ? ((t >= ly->idx_cp_s0 && t < ly->idx_cp_s1) ? t - ly->idx_cp_s0 : -1) : t;
                    if (il >= 0) { if (il >= ly->idx_cp_nslot) { fprintf(stderr,"[CP-DBG idxwarm OOB] t=%d il=%d cap=%d\n",t,il,ly->idx_cp_nslot); fflush(stderr); abort(); }
                        ds4f_idx_quant_pos_i4(ir, ihd, ly->idx_kv8_4 + (size_t)il*(ihd/2), &ly->idx_pscale[il]); } }
                else if (ly->idx_kv8)  ds4f_idx_quant_pos(ir, ihd, ly->idx_kv8 + (size_t)t*ihd, &ly->idx_pscale[t]);
            }
        }
    }
    if (rss_trace && m->ep_rank == 0)
        fprintf(stderr, "[warmtb2 DONE RSS=%.2f GB MemAvail=%.2f GB MemFree=%.2f GB int8_cmp=%d]\n",
                ds4f_rss_gb(), ds4f_mem_avail_gb(), ds4f_mem_free_gb(), m->int8_cmp);
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

/* Debug-only prefill trace.  It is intentionally environment-gated so the
 * production/configuration path remains free of per-layer scans. */
static int ds4f_prefill_trace_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("DS4F_PREFILL_TRACE"); on = e && atoi(e) != 0; }
    return on;
}

static float ds4f_trace_maxabs(const float *v, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; ++i) { float a = fabsf(v[i]); if (a > mx) mx = a; }
    return mx;
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

/* DS4F_HC_PAR (WS1): pool-parallelize the mHC collapse/expand loops (the per-d-independent
 * scalar work run on tid0 between pool dispatches, ~35% of real-gen decode). Default off.
 * Splits the hidden dim C into contiguous per-thread ranges -- each output element is an
 * independent reduction over k in fixed order => BIT-EXACT to the serial path (mirrors the
 * validated ds4f_q_norm_rope_par / DS4F_TB2ROPE_PAR pattern). The RMS reduction + matvec +
 * sinkhorn are left untouched (RMS would reassociate; matvec is already a pool_run). */
static int ds4f_hc_par = -1;             /* 1=pool-parallel, 0=serial ref (default) */
static inline int ds4f_hc_par_on(void) {
    if (ds4f_hc_par < 0) { const char *e = getenv("DS4F_HC_PAR"); ds4f_hc_par = e ? atoi(e) : 0; }
    return ds4f_hc_par;
}
/* collapse: y[d] = Σ_k pre[k]·x4[k*C+d]  (used by hc_pre and hc_head_p) */
typedef struct { const float *x4, *pre; float *y; int hc, C; } ds4f_hccol_task;
static void ds4f_hccol_worker(void *arg, int tid, int nthr) {
    ds4f_hccol_task *T = (ds4f_hccol_task *)arg;
    int C = T->C, hc = T->hc;
    int d0 = (int)((long)C*tid/nthr), d1 = (int)((long)C*(tid+1)/nthr);
    for (int d = d0; d < d1; d++) {
        float a = 0.f;
        for (int k = 0; k < hc; k++) a += T->pre[k]*T->x4[(size_t)k*C+d];
        T->y[d] = a;
    }
}
/* expand: x4[k,d] = post[k]·f[d] + Σ_j comb[j,k]·resid[j,d]  (used by hc_post). Loop nesting
 * identical to the serial body below => bit-exact per (k,d). */
typedef struct { float *x4; const float *resid, *f, *post, *comb; int hc, C; } ds4f_hcpost_task;
static void ds4f_hcpost_worker(void *arg, int tid, int nthr) {
    ds4f_hcpost_task *T = (ds4f_hcpost_task *)arg;
    int C = T->C, hc = T->hc;
    int d0 = (int)((long)C*tid/nthr), d1 = (int)((long)C*(tid+1)/nthr);
    for (int k = 0; k < hc; k++) {
        float pk = T->post[k]; float *ok = T->x4 + (size_t)k*C;
        for (int d = d0; d < d1; d++) ok[d] = pk*T->f[d];
        for (int j = 0; j < hc; j++) {
            float cjk = T->comb[j*hc+k]; const float *rj = T->resid + (size_t)j*C;
            for (int d = d0; d < d1; d++) ok[d] += cjk*rj[d];
        }
    }
}

/* WS1b: fuse the mHC RMS sum-of-squares INTO the mixes-matvec dispatch. The serial
 * `ss = Σ x4^2` over hd=hc*hidden was ~35% of the "other" phase (latency-bound double
 * accumulator chain on tid0, 89us/call). This worker computes the IDENTICAL F32 matvec
 * rows (same per-row dot order as ds4f_mv_worker => mixes BIT-EXACT) AND a per-thread
 * partial sum-of-squares of x4 over a disjoint hd slice; the caller combines partials in
 * fixed tid order. The ss REASSOCIATES (parallel vs sequential double) => mixes*rsq and
 * everything downstream is COHERENT, not bit-identical -> argmax gated by real-gen A/B.
 * Gated DS4F_HC_RMSPAR (default off). */
typedef struct { const float *fn, *x4; float *mixes; double *ssp; int rows, hd; } ds4f_hcmix_task;
static void ds4f_hcmix_worker(void *arg, int tid, int nthr) {
    ds4f_hcmix_task *T = (ds4f_hcmix_task *)arg;
    int rows = T->rows, hd = T->hd;
    int per = rows / nthr, extra = rows % nthr;          /* same row split as F32 mv_worker */
    int f0 = per * tid + (tid < extra ? tid : extra);
    int f1 = f0 + per + (tid < extra ? 1 : 0);
    for (int i = f0; i < f1; i++) {
        const float *w = T->fn + (size_t)i * hd;
        float acc = 0.f;
        for (int j = 0; j < hd; j++) acc += w[j] * T->x4[j];   /* identical inner => bit-exact */
        T->mixes[i] = acc;
    }
    int d0 = (int)((long)hd * tid / nthr), d1 = (int)((long)hd * (tid+1) / nthr);
    double s = 0.0;
    for (int j = d0; j < d1; j++) { float v = T->x4[j]; s += (double)v*v; }
    T->ssp[tid] = s;
}
static int ds4f_hc_rmspar = -1;
static inline int ds4f_hc_rmspar_on(void) {
    if (ds4f_hc_rmspar < 0) { const char *e = getenv("DS4F_HC_RMSPAR"); ds4f_hc_rmspar = e ? atoi(e) : 0; }
    return ds4f_hc_rmspar;
}

/* DS4F_HC_SVE: opt-in A64FX mHC fast path.  The scalar F32 dot has only 24 rows,
 * so a normal row split leaves roughly half of the pool idle.  Splitting each
 * row into two 8K-element SVE dots gives the pool 48 independent 32KB reads and
 * keeps both CMGs busy.  The two partial reductions are deliberately combined
 * after the worker returns: this is a reassociation-class (small FP32 error)
 * optimization, unlike DS4F_HC_RMSPAR/HC_PAR.  The default remains 0 so the
 * exact/token-identical path is still the quality reference. */
static int ds4f_hc_sve = -1;
static inline int ds4f_hc_sve_on(void) {
    if (ds4f_hc_sve < 0) { const char *e = getenv("DS4F_HC_SVE"); ds4f_hc_sve = e ? atoi(e) : 0; }
    return ds4f_hc_sve;
}

typedef struct { const float *fn, *x4; float *part; double *ssp; int rows, hd; } ds4f_hcmix_sve_task;
#if defined(__ARM_FEATURE_SVE)
static void ds4f_hcmix_sve_worker(void *arg, int tid, int nthr) {
    ds4f_hcmix_sve_task *T = (ds4f_hcmix_sve_task *)arg;
    int rows = T->rows, hd = T->hd, half = hd / 2;
    int units = rows * 2;
    int per = units / nthr, extra = units % nthr;
    int u0 = per * tid + (tid < extra ? tid : extra);
    int u1 = u0 + per + (tid < extra ? 1 : 0);
    svbool_t pg = svptrue_b32();
    int vl = (int)svcntw();
    for (int u = u0; u < u1; u++) {
        int i = u >> 1, h = u & 1;
        const float *w = T->fn + (size_t)i * hd + (size_t)h * half;
        const float *x = T->x4 + (size_t)h * half;
        svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
        svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
        int j = 0;
        for (; j + 4*vl <= half; j += 4*vl) {
            a0 = svmla_x(pg, a0, svld1_f32(pg, w+j),       svld1_f32(pg, x+j));
            a1 = svmla_x(pg, a1, svld1_f32(pg, w+j+vl),    svld1_f32(pg, x+j+vl));
            a2 = svmla_x(pg, a2, svld1_f32(pg, w+j+2*vl),  svld1_f32(pg, x+j+2*vl));
            a3 = svmla_x(pg, a3, svld1_f32(pg, w+j+3*vl),  svld1_f32(pg, x+j+3*vl));
        }
        for (; j < half; j += vl) {
            svbool_t p = svwhilelt_b32(j, half);
            a0 = svmla_x(p, a0, svld1_f32(p, w+j), svld1_f32(p, x+j));
        }
        T->part[u] = svaddv(pg, svadd_x(pg, svadd_x(pg, a0, a1), svadd_x(pg, a2, a3)));
    }
    int d0 = (int)((long)hd * tid / nthr), d1 = (int)((long)hd * (tid+1) / nthr);
    double ss = 0.0;
    for (int j = d0; j < d1; j++) { float v = T->x4[j]; ss += (double)v*v; }
    T->ssp[tid] = ss;
}
#endif /* __ARM_FEATURE_SVE */

/* SVE divides the 16 mHC comb elements as a single vector.  The row/column
 * sums remain scalar and in the reference order; only the independent element
 * divides are vectorized.  A64FX's 512-bit SVE has exactly 16 FP32 lanes. */
#if defined(__ARM_FEATURE_SVE)
static void ds4f_hc_sinkhorn_sve(const float *mixes, const float *scale, const float *base,
                                 int hc, int iters, float eps,
                                 float *pre, float *post, float *comb) {
    for (int j = 0; j < hc; j++)
        pre[j] = ds4f_sigmoidf(mixes[j]*scale[0] + base[j]) + eps;
    for (int j = 0; j < hc; j++)
        post[j] = 2.0f*ds4f_sigmoidf(mixes[j+hc]*scale[1] + base[j+hc]);
    for (int j = 0; j < hc; j++)
        for (int k = 0; k < hc; k++)
            comb[j*hc+k] = mixes[j*hc+k+2*hc]*scale[2] + base[j*hc+k+2*hc];
    for (int j = 0; j < hc; j++) {
        float mx = comb[j*hc];
        for (int k = 1; k < hc; k++) if (comb[j*hc+k] > mx) mx = comb[j*hc+k];
        float ss = 0.0f;
        for (int k = 0; k < hc; k++) { float e = expf(comb[j*hc+k]-mx); comb[j*hc+k] = e; ss += e; }
        for (int k = 0; k < hc; k++) comb[j*hc+k] = comb[j*hc+k]/ss + eps;
    }
    if (hc == 4) {
        svbool_t pg = svwhilelt_b32(0, 16);
        float den[16];
        for (int k = 0; k < 4; k++) {
            float cs = comb[k] + comb[4+k] + comb[8+k] + comb[12+k] + eps;
            den[k] = den[4+k] = den[8+k] = den[12+k] = cs;
        }
        svst1_f32(pg, comb, svdiv_x(pg, svld1_f32(pg, comb), svld1_f32(pg, den)));
        for (int it = 0; it < iters-1; it++) {
            for (int j = 0; j < 4; j++) {
                float rs = comb[j*4] + comb[j*4+1] + comb[j*4+2] + comb[j*4+3] + eps;
                den[j*4] = den[j*4+1] = den[j*4+2] = den[j*4+3] = rs;
            }
            svst1_f32(pg, comb, svdiv_x(pg, svld1_f32(pg, comb), svld1_f32(pg, den)));
            for (int k = 0; k < 4; k++) {
                float cs = comb[k] + comb[4+k] + comb[8+k] + comb[12+k] + eps;
                den[k] = den[4+k] = den[8+k] = den[12+k] = cs;
            }
            svst1_f32(pg, comb, svdiv_x(pg, svld1_f32(pg, comb), svld1_f32(pg, den)));
        }
    } else {
        for (int k = 0; k < hc; k++) {
            float cs = 0.0f; for (int j = 0; j < hc; j++) cs += comb[j*hc+k];
            cs += eps; for (int j = 0; j < hc; j++) comb[j*hc+k] /= cs;
        }
        for (int it = 0; it < iters-1; it++) {
            for (int j = 0; j < hc; j++) {
                float rs = 0.0f; for (int k = 0; k < hc; k++) rs += comb[j*hc+k];
                rs += eps; for (int k = 0; k < hc; k++) comb[j*hc+k] /= rs;
            }
            for (int k = 0; k < hc; k++) {
                float cs = 0.0f; for (int j = 0; j < hc; j++) cs += comb[j*hc+k];
                cs += eps; for (int j = 0; j < hc; j++) comb[j*hc+k] /= cs;
            }
        }
    }
}
#endif /* __ARM_FEATURE_SVE */

/* hc_pre: x4[hc*C] (4 streams) -> collapsed y[C]; also yields post[hc], comb[hc*hc].
 * mixes = (fn @ flatten(x4)) * rsqrt(mean(x4^2)+norm_eps); sinkhorn; y[d]=Σ_k pre[k]·x4[k,d]. */
static void ds4f_hc_pre(ds4f_model *m, const float *x4, const float *fn,
                        const float *scale, const float *base,
                        float *y, float *post, float *comb) {
    ds4f_config *c = &m->cfg;
    int hc = c->hc_mult, C = c->hidden, hd = hc*C, mix_hc = (2+hc)*hc;
    float mixes[64];                 /* mix_hc <= 24 for hc<=4 */
    float rsq;
#if defined(__ARM_FEATURE_SVE)
    if (ds4f_hc_sve_on()) {
        float part[128]; double ssp[64];
        ds4f_hcmix_sve_task T = { fn, x4, part, ssp, mix_hc, hd };
        m->bytes_read += ds4f_wbytes(DS4F_F32, mix_hc, hd) + ds4f_sbytes(DS4F_F32, mix_hc, hd);
        ds4f_pool_run(m->pool, ds4f_hcmix_sve_worker, &T);
        for (int i = 0; i < mix_hc; i++) mixes[i] = part[2*i] + part[2*i+1];
        double ss = 0.0; for (int t = 0; t < m->pool->nthr; t++) ss += ssp[t];
        rsq = 1.0f/sqrtf((float)(ss/hd) + c->norm_eps);
    } else
#endif
    if (ds4f_hc_rmspar_on()) {       /* WS1b: fold the RMS sum-of-squares INTO the mixes-matvec
                                      * dispatch (kills the 89us serial tid0 reduction). ss is a
                                      * DOUBLE accumulation, so the parallel-vs-sequential reassoc
                                      * (~1e-13) is far below the float rsq's epsilon => rsq, and
                                      * thus the whole result, is bit-identical in practice. */
        double ssp[64];
        ds4f_hcmix_task T = { fn, x4, mixes, ssp, mix_hc, hd };
        m->bytes_read += ds4f_wbytes(DS4F_F32, mix_hc, hd) + ds4f_sbytes(DS4F_F32, mix_hc, hd);
        ds4f_pool_run(m->pool, ds4f_hcmix_worker, &T);
        double ss = 0.0; for (int t = 0; t < m->pool->nthr; t++) ss += ssp[t];   /* fixed tid order */
        rsq = 1.0f/sqrtf((float)(ss/hd) + c->norm_eps);
    } else {
        double ss = 0.0; for (int i = 0; i < hd; i++) { float v = x4[i]; ss += (double)v*v; }
        rsq = 1.0f/sqrtf((float)(ss/hd) + c->norm_eps);
        ds4f_tensor fnt = { (void*)fn, NULL, DS4F_F32, mix_hc, hd, -1 };
        ds4f_matvec(m, mixes, &fnt, x4); /* threaded F32 Linear */
    }
    for (int mm = 0; mm < mix_hc; mm++) mixes[mm] *= rsq;
    float pre[16];
#if defined(__ARM_FEATURE_SVE)
    if (ds4f_hc_sve_on()) ds4f_hc_sinkhorn_sve(mixes, scale, base, hc, c->hc_iters, c->hc_eps, pre, post, comb);
    else
#endif
    ds4f_hc_sinkhorn(mixes, scale, base, hc, c->hc_iters, c->hc_eps, pre, post, comb);
    if (ds4f_hc_par_on()) {
        ds4f_hccol_task T = { x4, pre, y, hc, C };
        ds4f_pool_run(m->pool, ds4f_hccol_worker, &T);
    } else for (int d = 0; d < C; d++) {
        float a = 0.f; for (int k = 0; k < hc; k++) a += pre[k]*x4[(size_t)k*C+d];
        y[d] = a;
    }
}

/* hc_post: expand block output f[C] back to hc streams x4[hc*C], folding the
 * pre-block residual: x4[k,d] = post[k]·f[d] + Σ_j comb[j,k]·resid[j,d]. */
static void ds4f_hc_post(ds4f_model *m, float *x4, const float *resid, const float *f,
                         const float *post, const float *comb) {
    int hc = m->cfg.hc_mult, C = m->cfg.hidden;
    if (ds4f_hc_par_on()) {
        ds4f_hcpost_task T = { x4, resid, f, post, comb, hc, C };
        ds4f_pool_run(m->pool, ds4f_hcpost_worker, &T);
        return;
    }
    for (int k = 0; k < hc; k++) {
        float pk = post[k]; float *ok = x4 + (size_t)k*C;
        for (int d = 0; d < C; d++) ok[d] = pk*f[d];
        for (int j = 0; j < hc; j++) {
            float cjk = comb[j*hc+k]; const float *rj = resid + (size_t)j*C;
            for (int d = 0; d < C; d++) ok[d] += cjk*rj[d];
        }
    }
}

/* Batched mHC helpers used by forward_verify.  The old verify path called the
 * single-token hc_pre/hc_post routines K times, paying a full pool wake and
 * completion barrier for every prompt token.  These workers keep each token's
 * dot-product/reduction order unchanged, but cover all K tokens with one pool
 * dispatch per mHC operation.  The only cross-token reassociation is the
 * independent scheduling of disjoint token buffers. */
typedef struct {
    const float *x4, *fn;
    float *mixes;
    double *ssp;
    int M, mix_hc, hd, nthr_slots;
} ds4f_hcpre_batch_task;

static void ds4f_hcpre_batch_worker(void *arg, int tid, int nthr) {
    ds4f_hcpre_batch_task *T = (ds4f_hcpre_batch_task *)arg;
    int M = T->M, mix_hc = T->mix_hc, hd = T->hd;
    /* Flatten token×mix-row work.  A token-only split gives each of the 8
     * tokens to one of 47 workers and is needlessly fragile around the integer
     * boundaries; this partition keeps every worker busy and guarantees that
     * every token's mix rows are written exactly once. */
    long total = (long)M * mix_hc;
    long u0 = total * tid / nthr, u1 = total * (tid + 1) / nthr;
    for (long u = u0; u < u1; u++) {
        int k = (int)(u / mix_hc), i = (int)(u % mix_hc);
        const float *x = T->x4 + (size_t)k * hd;
        float *mx = T->mixes + (size_t)k * mix_hc;
        const float *w = T->fn + (size_t)i * hd;
        float a = 0.f;
        for (int j = 0; j < hd; j++) a += w[j] * x[j];
        mx[i] = a;
    }
    /* Each worker contributes one fixed-order partial sum for every token. */
    int d0 = (int)((long)hd * tid / nthr), d1 = (int)((long)hd * (tid + 1) / nthr);
    for (int k = 0; k < M; k++) {
        const float *x = T->x4 + (size_t)k * hd;
        double ss = 0.0;
        for (int j = d0; j < d1; j++) { float v = x[j]; ss += (double)v * v; }
        T->ssp[(size_t)k * T->nthr_slots + tid] = ss;
    }
}

typedef struct {
    const float *x4, *pre;
    float *y;
    int M, hc, C;
} ds4f_hccol_batch_task;

static void ds4f_hccol_batch_worker(void *arg, int tid, int nthr) {
    ds4f_hccol_batch_task *T = (ds4f_hccol_batch_task *)arg;
    long total = (long)T->M * T->C;
    long u0 = total * tid / nthr, u1 = total * (tid + 1) / nthr;
    for (long u = u0; u < u1; u++) {
        int k = (int)(u / T->C), d = (int)(u % T->C);
        const float *x = T->x4 + (size_t)k * T->hc * T->C;
        const float *pre = T->pre + (size_t)k * T->hc;
        float a = 0.f;
        for (int j = 0; j < T->hc; j++) a += pre[j] * x[(size_t)j * T->C + d];
        T->y[(size_t)k * T->C + d] = a;
    }
}

typedef struct {
    float *x4;
    const float *resid, *f, *post, *comb;
    int M, hc, C;
} ds4f_hcpost_batch_task;

static void ds4f_hcpost_batch_worker(void *arg, int tid, int nthr) {
    ds4f_hcpost_batch_task *T = (ds4f_hcpost_batch_task *)arg;
    long total = (long)T->M * T->C;
    long u0 = total * tid / nthr, u1 = total * (tid + 1) / nthr;
    for (long u = u0; u < u1; u++) {
        int k = (int)(u / T->C), d = (int)(u % T->C);
        float *x = T->x4 + (size_t)k * T->hc * T->C;
        const float *res = T->resid + (size_t)k * T->hc * T->C;
        const float *f = T->f + (size_t)k * T->C;
        const float *post = T->post + (size_t)k * T->hc;
        const float *comb = T->comb + (size_t)k * T->hc * T->hc;
        for (int j = 0; j < T->hc; j++) {
            float a = post[j] * f[d];
            for (int q = 0; q < T->hc; q++) a += comb[(size_t)q * T->hc + j] * res[(size_t)q * T->C + d];
            x[(size_t)j * T->C + d] = a;
        }
    }
}

static void ds4f_hc_pre_batch_legacy(ds4f_model *m, const float *x4, int M, const float *fn,
                              const float *scale, const float *base,
                              float *y, float *post, float *comb) {
    ds4f_config *c = &m->cfg;
    int hc = c->hc_mult, C = c->hidden, hd = hc * C, mix_hc = (2 + hc) * hc;
    if (M < 1 || M > 8) { fprintf(stderr, "ds4f_hc_pre_batch: M=%d\n", M); abort(); }
    float mixes[8][64], pre[8][16];
    double ssp[8][64];
    memset(ssp, 0, sizeof(ssp));
    ds4f_hcpre_batch_task T = { x4, fn, &mixes[0][0], &ssp[0][0], M, mix_hc, hd, m->pool->nthr };
    m->bytes_read += (size_t)M * ds4f_wbytes(DS4F_F32, mix_hc, hd);
    ds4f_pool_run(m->pool, ds4f_hcpre_batch_worker, &T);
    for (int k = 0; k < M; k++) {
        double ss = 0.0;
        for (int tid = 0; tid < m->pool->nthr; tid++) ss += ssp[k][tid];
        float rsq = 1.0f / sqrtf((float)(ss / hd) + c->norm_eps);
        for (int i = 0; i < mix_hc; i++) mixes[k][i] *= rsq;
        ds4f_hc_sinkhorn(mixes[k], scale, base, hc, c->hc_iters, c->hc_eps,
                         pre[k], post + (size_t)k * hc, comb + (size_t)k * hc * hc);
    }
    /* This is only M*C*hc scalar FMAs (131k at K=8).  Keeping it in one fixed-
     * order loop avoids a pool-dispatch corner on short token tiles and is
     * negligible beside the batched dense GEMMs. */
    for (int k = 0; k < M; k++) {
        const float *px = x4 + (size_t)k*hc*C, *pk = pre[k];
        float *py = y + (size_t)k*C;
        for (int d = 0; d < C; d++) {
            float a = 0.f;
            for (int j = 0; j < hc; j++) a += pk[j] * px[(size_t)j*C+d];
            py[d] = a;
        }
    }
}

static void ds4f_hc_post_batch_legacy(ds4f_model *m, float *x4, const float *resid,
                               const float *f, int M, const float *post, const float *comb) {
    int hc = m->cfg.hc_mult, C = m->cfg.hidden;
    if (M < 1 || M > 8) { fprintf(stderr, "ds4f_hc_post_batch: M=%d\n", M); abort(); }
    for (int k = 0; k < M; k++) {
        float *px = x4 + (size_t)k*hc*C;
        const float *pr = resid + (size_t)k*hc*C, *pf = f + (size_t)k*C;
        const float *pp = post + (size_t)k*hc, *pc = comb + (size_t)k*hc*hc;
        for (int j = 0; j < hc; j++) for (int d = 0; d < C; d++) {
            float a = pp[j] * pf[d];
            for (int q = 0; q < hc; q++) a += pc[(size_t)q*hc+j] * pr[(size_t)q*C+d];
            px[(size_t)j*C+d] = a;
        }
    }
}

/* Validated batched mHC scheduling. Mix rows, RMS sums, sinkhorn positions,
 * and collapse/expand work are separate pool jobs; this avoids making one
 * short token tile depend on a cross-product of partial reductions. */
typedef struct { const float *fn, *x4b; float *mixb; int mix_hc, hd, K; } ds4f_hcmix_b_task;
static void ds4f_hcmix_b_worker(void *arg, int tid, int nthr) {
    ds4f_hcmix_b_task *T = (ds4f_hcmix_b_task *)arg;
    int mh = T->mix_hc, hd = T->hd;
    long tot = (long)T->K * mh, per = tot/nthr, ex = tot%nthr;
    long u0 = per*tid + (tid<ex?tid:ex), u1 = u0 + per + (tid<ex?1:0);
    for (long u = u0; u < u1; u++) {
        int k = (int)(u/mh), i = (int)(u%mh);
        const float *w = T->fn + (size_t)i*hd, *x = T->x4b + (size_t)k*hd;
        float a = 0.f; for (int j = 0; j < hd; j++) a += w[j]*x[j];
        T->mixb[(size_t)k*mh+i] = a;
    }
}
#if defined(__ARM_FEATURE_SVE)
#if defined(__ARM_FEATURE_SVE)
static void ds4f_hcmix_b_sve_worker(void *arg, int tid, int nthr) {
    ds4f_hcmix_b_task *T = (ds4f_hcmix_b_task *)arg;
    int mh = T->mix_hc, hd = T->hd;
    long tot = (long)T->K*mh, per = tot/nthr, ex = tot%nthr;
    long u0 = per*tid + (tid<ex?tid:ex), u1 = u0 + per + (tid<ex?1:0);
    svbool_t pg = svptrue_b32(); int vl = (int)svcntw();
    for (long u = u0; u < u1; u++) {
        int k = (int)(u/mh), i = (int)(u%mh);
        const float *w = T->fn + (size_t)i*hd, *x = T->x4b + (size_t)k*hd;
        svfloat32_t a0=svdup_f32(0), a1=svdup_f32(0), a2=svdup_f32(0), a3=svdup_f32(0);
        int j=0;
        for (; j+4*vl <= hd; j+=4*vl) {
            a0=svmla_x(pg,a0,svld1_f32(pg,w+j),svld1_f32(pg,x+j));
            a1=svmla_x(pg,a1,svld1_f32(pg,w+j+vl),svld1_f32(pg,x+j+vl));
            a2=svmla_x(pg,a2,svld1_f32(pg,w+j+2*vl),svld1_f32(pg,x+j+2*vl));
            a3=svmla_x(pg,a3,svld1_f32(pg,w+j+3*vl),svld1_f32(pg,x+j+3*vl));
        }
        for (; j<hd; j+=vl) { svbool_t p=svwhilelt_b32(j,hd); a0=svmla_x(p,a0,svld1_f32(p,w+j),svld1_f32(p,x+j)); }
        T->mixb[(size_t)k*mh+i]=svaddv(pg,svadd_x(pg,svadd_x(pg,a0,a1),svadd_x(pg,a2,a3)));
    }
}
#endif /* __ARM_FEATURE_SVE */
#endif
typedef struct { const float *mixb,*scale,*base; float *preb,*postb,*combb;
                 int K,mix_hc,hc,iters,pstr,cstr; float eps; } ds4f_sinkb_task;
static void ds4f_sinkb_worker(void *arg, int tid, int nthr) {
    ds4f_sinkb_task *T=(ds4f_sinkb_task *)arg;
    int k0=(int)((long)T->K*tid/nthr), k1=(int)((long)T->K*(tid+1)/nthr);
    for (int k=k0;k<k1;k++) ds4f_hc_sinkhorn(T->mixb+(size_t)k*T->mix_hc,T->scale,T->base,T->hc,T->iters,T->eps,
        T->preb+(size_t)k*T->hc,T->postb+(size_t)k*T->pstr,T->combb+(size_t)k*T->cstr);
}
typedef struct { const float *x4b; float *ssb; int hd,K; } ds4f_hcss_b_task;
static void ds4f_hcss_b_worker(void *arg, int tid, int nthr) {
    ds4f_hcss_b_task *T=(ds4f_hcss_b_task *)arg; int per=T->K/nthr, ex=T->K%nthr;
    int k0=per*tid+(tid<ex?tid:ex), k1=k0+per+(tid<ex?1:0);
    for (int k=k0;k<k1;k++) { const float *x=T->x4b+(size_t)k*T->hd; double s=0;
        for (int j=0;j<T->hd;j++) { float v=x[j]; s+=(double)v*v; } T->ssb[k]=(float)s; }
}
typedef struct { const float *x4b,*preb; float *yb; int hc,C,K; } ds4f_hccol_b_task;
static void ds4f_hccol_b_worker(void *arg, int tid, int nthr) {
    ds4f_hccol_b_task *T=(ds4f_hccol_b_task *)arg; size_t hcC=(size_t)T->hc*T->C;
    long tot=(long)T->K*T->C, per=tot/nthr, ex=tot%nthr;
    long u0=per*tid+(tid<ex?tid:ex), u1=u0+per+(tid<ex?1:0);
    for (long u=u0;u<u1;u++) { int k=(int)(u/T->C), d=(int)(u%T->C);
        const float *x=T->x4b+(size_t)k*hcC,*pre=T->preb+(size_t)k*T->hc; float a=0;
        for (int j = 0; j < T->hc; j++) a += pre[j] * x[(size_t)j * T->C + d];
        T->yb[(size_t)k * T->C + d] = a;
    }
}
typedef struct { float *x4b; const float *residb,*fb,*postb,*combb; int hc,C,K,pstr,cstr; } ds4f_hcpost_b_task;
static void ds4f_hcpost_b_worker(void *arg, int tid, int nthr) {
    ds4f_hcpost_b_task *T=(ds4f_hcpost_b_task *)arg; size_t hcC=(size_t)T->hc*T->C;
    long tot=(long)T->K*T->C, per=tot/nthr, ex=tot%nthr;
    long u0=per*tid+(tid<ex?tid:ex), u1=u0+per+(tid<ex?1:0);
    for (long u=u0;u<u1;u++) { int k=(int)(u/T->C), d=(int)(u%T->C);
        float *x4=T->x4b+(size_t)k*hcC; const float *res=T->residb+(size_t)k*hcC;
        const float *f=T->fb+(size_t)k*T->C,*post=T->postb+(size_t)k*T->pstr,*comb=T->combb+(size_t)k*T->cstr;
        for (int kk=0;kk<T->hc;kk++) { float v=post[kk]*f[d]; for (int j=0;j<T->hc;j++) v+=comb[j*T->hc+kk]*res[(size_t)j*T->C+d]; x4[(size_t)kk*T->C+d]=v; }
    }
}
static void ds4f_hc_pre_batch(ds4f_model *m, const float *x4b, int K, const float *fn,
                              const float *scale, const float *base, float *yb,
                              float *postb, int pstr, float *combb, int cstr) {
    ds4f_config *c=&m->cfg; int hc=c->hc_mult,C=c->hidden,hd=hc*C,mix_hc=(2+hc)*hc;
    if (K<1 || K>128) { fprintf(stderr,"ds4f_hc_pre_batch: K=%d\n",K); abort(); }
    float *mixb=(float *)alloca((size_t)K*mix_hc*4), *ssb=(float *)alloca((size_t)K*4), *preb=(float *)alloca((size_t)K*hc*4);
    ds4f_hcmix_b_task mt={fn,x4b,mixb,mix_hc,hd,K};
#if defined(__ARM_FEATURE_SVE)
#if defined(__ARM_FEATURE_SVE)
    if (ds4f_hc_sve_on()) ds4f_pool_run(m->pool,ds4f_hcmix_b_sve_worker,&mt); else
#endif
#endif
    ds4f_pool_run(m->pool,ds4f_hcmix_b_worker,&mt);
    ds4f_hcss_b_task st={x4b,ssb,hd,K}; ds4f_pool_run(m->pool,ds4f_hcss_b_worker,&st);
    for (int k=0;k<K;k++) { float rsq=1.0f/sqrtf(ssb[k]/hd+c->norm_eps); for (int mm=0;mm<mix_hc;mm++) mixb[(size_t)k*mix_hc+mm]*=rsq; }
    if (ds4f_hc_sve_on() && K>1) { ds4f_sinkb_task kt={mixb,scale,base,preb,postb,combb,K,mix_hc,hc,c->hc_iters,pstr,cstr,c->hc_eps}; ds4f_pool_run(m->pool,ds4f_sinkb_worker,&kt); }
    else for (int k=0;k<K;k++) ds4f_hc_sinkhorn(mixb+(size_t)k*mix_hc,scale,base,hc,c->hc_iters,c->hc_eps,
        preb+(size_t)k*hc,postb+(size_t)k*pstr,combb+(size_t)k*cstr);
    ds4f_hccol_b_task ct={x4b,preb,yb,hc,C,K}; ds4f_pool_run(m->pool,ds4f_hccol_b_worker,&ct);
}
static void ds4f_hc_post_batch(ds4f_model *m, float *x4b, int K, const float *residb, const float *fb,
                               const float *postb, int pstr, const float *combb, int cstr) {
    ds4f_hcpost_b_task pt={x4b,residb,fb,postb,combb,m->cfg.hc_mult,m->cfg.hidden,K,pstr,cstr};
    ds4f_pool_run(m->pool,ds4f_hcpost_b_worker,&pt);
}

/* hc_head: final collapse hc streams x4[hc*C] -> y[C] via per-stream sigmoid gate
 * (sigmoid(mixes*scale+base)+eps), NO sinkhorn. Mirrors ParallelHead.hc_head. */
/* parameterized HC-head collapse (hc streams -> 1) -- the MTP head reuses it with its own
 * hc_fn/base/scale. ds4f_hc_head is the main-model wrapper (m->hc_head_*). */
static void ds4f_hc_head_p(ds4f_model *m, const float *x4, float *y,
                           const float *hc_fn, const float *hc_base, const float *hc_scale) {
    ds4f_config *c = &m->cfg;
    int hc = c->hc_mult, C = c->hidden, hd = hc*C;
    double ss = 0.0; for (int i = 0; i < hd; i++) { float v = x4[i]; ss += (double)v*v; }
    float rsq = 1.0f/sqrtf((float)(ss/hd) + c->norm_eps);
    float mixes[16];
    ds4f_tensor fnt = { (void*)hc_fn, NULL, DS4F_F32, hc, hd, -1 };
    ds4f_matvec(m, mixes, &fnt, x4);
    float pre[16];
    for (int k = 0; k < hc; k++)
        pre[k] = ds4f_sigmoidf(mixes[k]*rsq*hc_scale[0] + hc_base[k]) + c->hc_eps;
    if (ds4f_hc_par_on()) {
        ds4f_hccol_task T = { x4, pre, y, hc, C };
        ds4f_pool_run(m->pool, ds4f_hccol_worker, &T);
    } else for (int d = 0; d < C; d++) {
        float a = 0.f; for (int k = 0; k < hc; k++) a += pre[k]*x4[(size_t)k*C+d];
        y[d] = a;
    }
}
static void ds4f_hc_head(ds4f_model *m, const float *x4, float *y) {
    ds4f_hc_head_p(m, x4, y, m->hc_head_fn, m->hc_head_base, m->hc_head_scale);
}

/* Runs MLA + MoE for one token at position `pos`, hidden state in/out `x`[hidden].
 * Returns next-token argmax of synthetic logits (meaningless). */
/* ===================== batched (M>1) prefill =====================
 * Computes the SAME result as M independent ds4f_forward_token(exact) calls, but
 * the dense projections (wq_a/wq_b/wkv, grouped wo_a + wo_b, shared w1/w3/w2,
 * router gate, head) run as ONE batched ds4f_gemm over the M-token tile -> each
 * weight is read from HBM once instead of once per token, turning the M=1
 * BW-bound matvec into a compute-bound GEMM (the ~10x prefill lever). Attention
 * parallelizes over (token,head); routed experts stay per-token (MXFP4, no M>1
 * quant GEMM yet); the EP all-reduce is ONE [M,C] call per layer (vs M).
 * FP8 dense (the default) batches via a FUSED FP8->bf16 tile-dequant GEMM;
 * DS4F_FP8_BF16=1 predequants the replicated dense bank to resident BF16 for
 * faster projections at the cost of roughly +6 GB. Only
 * DS4F_F32 falls back to per-token (no speedup). mHC / Tier-B2 are NOT supported
 * (die-guarded). The
 * GEMM K-tile reassociation makes the result bit-SIMILAR (~1e-4), not -identical,
 * to the per-token path; argmax matches. */

/* allocate the [m_tile, width] token-major prefill scratch (idempotent). */
static void ds4f_alloc_prefill_batch(ds4f_model *m, int m_tile) {
    if (m_tile > DS4F_MAX_MTILE) m_tile = DS4F_MAX_MTILE;
    if (m->p_x && m->m_tile >= m_tile) return;       /* already big enough */
    ds4f_config *c = &m->cfg;
    int C = c->hidden, H = c->n_heads*c->q_head_dim;
    size_t T = (size_t)m_tile;
    m->m_tile  = m_tile;
    m->p_x     = (float *)ds4f_mem_alloc(m->mem, T*(size_t)C*4, 256, 1);
    m->p_hn    = (float *)ds4f_mem_alloc(m->mem, T*(size_t)C*4, 256, 1);
    m->p_qlat  = (float *)ds4f_mem_alloc(m->mem, T*(size_t)c->q_lora*4, 256, 1);
    m->p_q     = (float *)ds4f_mem_alloc(m->mem, T*(size_t)H*4, 256, 1);
    m->p_kvlat = (float *)ds4f_mem_alloc(m->mem, T*(size_t)c->kv_lora*4, 256, 1);
    m->p_attn  = (float *)ds4f_mem_alloc(m->mem, T*(size_t)H*4, 256, 1);
    m->p_o1    = (float *)ds4f_mem_alloc(m->mem, T*(size_t)c->o_inter*4, 256, 1);
    m->p_o     = (float *)ds4f_mem_alloc(m->mem, T*(size_t)C*4, 256, 1);
    m->p_h2    = (float *)ds4f_mem_alloc(m->mem, T*(size_t)C*4, 256, 1);
    m->p_shg   = (float *)ds4f_mem_alloc(m->mem, T*(size_t)c->shared_inter*4, 256, 1);
    m->p_shu   = (float *)ds4f_mem_alloc(m->mem, T*(size_t)c->shared_inter*4, 256, 1);
    m->p_moe   = (float *)ds4f_mem_alloc(m->mem, T*(size_t)C*4, 256, 1);
    m->p_route = (float *)ds4f_mem_alloc(m->mem, T*(size_t)C*4, 256, 1);
    m->p_router= (float *)ds4f_mem_alloc(m->mem, T*(size_t)c->n_experts*4, 256, 1);
    /* expert-grouping buckets + gather/GEMM scratch.  A token can visit
     * n_active experts, so the staged GEMM slabs are sized for all local
     * assignments; this lets all experts share one gate/up and one down
     * dispatch while each expert retains its original token order. */
    int no = ds4f_n_owned(c->n_experts, m->ep_rank, m->ep_size);
    size_t ET = T * (size_t)c->n_active;
    m->ex_no   = no;
    m->ex_cnt  = (int   *)ds4f_mem_alloc(m->mem, (size_t)no*sizeof(int), 256, 1);
    m->ex_tok  = (int   *)ds4f_mem_alloc(m->mem, (size_t)no*T*sizeof(int), 256, 1);
    m->ex_wt   = (float *)ds4f_mem_alloc(m->mem, (size_t)no*T*4, 256, 1);
    m->p_exX   = (float *)ds4f_mem_alloc(m->mem, ET*(size_t)C*4, 256, 1);
    m->p_exG   = (float *)ds4f_mem_alloc(m->mem, ET*(size_t)c->moe_inter*4, 256, 1);
    m->p_exU   = (float *)ds4f_mem_alloc(m->mem, ET*(size_t)c->moe_inter*4, 256, 1);
    m->p_exO   = (float *)ds4f_mem_alloc(m->mem, ET*(size_t)C*4, 256, 1);
}

/* batched RMSNorm: dst[mm] = rmsnorm(src[mm], w) for mm in [0,M). token-parallel. */
typedef struct { ds4f_model *m; float *dst; const float *src; const uint16_t *w;
                 int n, M, dstride, sstride; } ds4f_pf_rms_task;
static void ds4f_pf_rmsnorm_worker(void *arg, int tid, int nthr) {
    ds4f_pf_rms_task *T = (ds4f_pf_rms_task *)arg;
    int M = T->M, per = M/nthr, extra = M%nthr;
    int m0 = per*tid + (tid<extra?tid:extra), m1 = m0 + per + (tid<extra?1:0);
    for (int mm = m0; mm < m1; mm++)
        ds4f_rmsnorm(T->dst + (size_t)mm*T->dstride, T->src + (size_t)mm*T->sstride,
                     T->w, T->n, 1e-6f);
}

/* batched per-head q-norm + RoPE (matches ds4f_q_norm_rope). over (token,head). */
typedef struct { ds4f_model *m; int pos0, M; const float *rcos, *rsin; } ds4f_pf_qnr_task;
static void ds4f_pf_qnr_worker(void *arg, int tid, int nthr) {
    ds4f_pf_qnr_task *T = (ds4f_pf_qnr_task *)arg;
    ds4f_model *m = T->m; ds4f_config *c = &m->cfg;
    int HD = c->q_head_dim, rd = c->qk_rope_dim, half = rd/2, nope = HD - rd, nh = c->n_heads;
    int H = nh*HD;
    long total = (long)T->M * nh, per = total/nthr, extra = total%nthr;
    long u0 = per*tid + (tid<extra?tid:extra), u1 = u0 + per + (tid<extra?1:0);
    for (long u = u0; u < u1; u++) {
        int mm = (int)(u / nh), h = (int)(u % nh);
        float *qh = m->p_q + (size_t)mm*H + (size_t)h*HD;
        double ss = 0.0; for (int d = 0; d < HD; d++) ss += (double)qh[d]*qh[d];
        float inv = 1.0f/sqrtf((float)(ss/HD) + c->norm_eps);
        for (int d = 0; d < HD; d++) qh[d] *= inv;
        ds4f_rope_apply(qh + nope, T->rcos, T->rsin, T->pos0 + mm, half, 0);
    }
}

/* Widening SVE load: 16 consecutive bf16 latents -> svfloat32_t (shift, no fcvt).
 * bf16 IS the top 16 bits of f32, so widening is just a 16-bit left shift:
 * svld1uh_u32 zero-extends each bf16 into the low 16 bits of a 32-bit lane,
 * lsl #16 moves it into the high half, reinterpret as f32. Lane count matches
 * the b32 predicate, so it drops straight into the f32 kernels. */
#if defined(__ARM_FEATURE_SVE)
static inline svfloat32_t ds4f_ld_bf16x(svbool_t pg, const uint16_t *p) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, p), 16));
}
#endif /* __ARM_FEATURE_SVE */

/* batched kv-latent post: rmsnorm + RoPE(rope dims) + append to kv_cache. token-parallel. */
typedef struct { ds4f_model *m; ds4f_layer *ly; int pos0, M; const float *rcos, *rsin; } ds4f_pf_kv_task;
static void ds4f_pf_kvpost_worker(void *arg, int tid, int nthr) {
    ds4f_pf_kv_task *T = (ds4f_pf_kv_task *)arg;
    ds4f_model *m = T->m; ds4f_layer *ly = T->ly; ds4f_config *c = &m->cfg;
    int KV = c->kv_lora, rd = c->qk_rope_dim;
    int M = T->M, per = M/nthr, extra = M%nthr;
    int m0 = per*tid + (tid<extra?tid:extra), m1 = m0 + per + (tid<extra?1:0);
    for (int mm = m0; mm < m1; mm++) {
        float *kv = m->p_kvlat + (size_t)mm*KV;
        ds4f_rmsnorm(kv, kv, ly->kv_norm, KV, 1e-6f);
        ds4f_rope_apply(kv + (KV - rd), T->rcos, T->rsin, T->pos0 + mm, rd/2, 0);
        uint16_t *dst = ly->kv_cache + (size_t)(T->pos0 + mm)*KV;   /* f32 latent -> bf16 cache */
        for (int d = 0; d < KV; d++) dst[d] = ds4f_f32bf(kv[d]);
    }
}

/* batched sliding-window + sink attention over (token,head). Per token mm the
 * query at absolute pos=pos0+mm attends kv_cache[p_lo..pos] (p_lo=pos-win+1,
 * clamped 0); sink adds exp(sink-max) to the denominator; weighted-V over the
 * latent; output rope dims de-rotated at the query pos. Byte-faithful to
 * ds4f_attn_exact_worker, just reorganized over the M-token tile. */
typedef struct { ds4f_model *m; ds4f_layer *ly; int pos0, M; float scale;
                 int win, half; const float *rcos, *rsin; } ds4f_attn_pf_task;
#if defined(__ARM_FEATURE_SVE)
static void ds4f_attn_prefill_worker(void *arg, int tid, int nthr) {
    ds4f_attn_pf_task *T = (ds4f_attn_pf_task *)arg;
    ds4f_model *m = T->m; ds4f_layer *ly = T->ly;
    int HD = m->cfg.q_head_dim, KV = m->cfg.kv_lora;
    int rd = m->cfg.qk_rope_dim, nope = HD - rd, nh = m->cfg.n_heads;
    int H = nh*HD, win = T->win, pos0 = T->pos0;
    const float scale = T->scale;
    enum { HBLK = 8 };
    /* Attention as two small per-token GEMMs over a head-block of HBLK heads.
     * In MLA all heads share the SAME compressed latent rows kc[j] (= K = V);
     * the per-head path re-streams that latent from L2 once PER HEAD (64x). Here
     * a unit owns (token mm, head-block hb) and loads each kc[j] vector ONCE,
     * reusing it across all HBLK heads in registers: phase-1 scores are
     * j-outer/head-inner (8 dot accumulators share kc[j]); phase-3 V-output is
     * d-block-outer/head-inner (8 column accumulators share kc[j][d]) -> ~HBLK x
     * less latent traffic. Argmax-faithful to the per-head loop: phase-2 softmax
     * is identical; phase-3 keeps j ascending so each out[d] is bit-identical
     * (fmla both sides via -ffp-contract=fast); only the phase-1 dot becomes an
     * SVE tree reduction (bit-similar). A partial trailing block (nh%HBLK!=0)
     * falls back to the scalar per-head path. KV,HD are multiples of 16 here. */
    int nhb = (nh + HBLK - 1) / HBLK;
    long total = (long)T->M * nhb, per = total/nthr, extra = total%nthr;
    long u0 = per*tid + (tid<extra?tid:extra), u1 = u0 + per + (tid<extra?1:0);
    int wpad = (win > 0 ? win : 1);
    float *sc = (float *)alloca((size_t)HBLK*wpad*4);   /* sc[hh][j], nP <= win */
    svbool_t pt = svptrue_b32();
    for (long u = u0; u < u1; u++) {
        int mm = (int)(u / nhb), hb = (int)(u % nhb);
        int h0 = hb*HBLK, nh_b = nh - h0; if (nh_b > HBLK) nh_b = HBLK;
        int pos = pos0 + mm, p_lo = pos - win + 1; if (p_lo < 0) p_lo = 0;
        int nP = pos - p_lo + 1;
        if (nh_b != HBLK) {   /* scalar fallback for a partial head-block */
            for (int h = h0; h < h0 + nh_b; h++) {
                const float *q = m->p_q + (size_t)mm*H + (size_t)h*HD;
                float mx = -1e30f;
                for (int j = 0; j < nP; j++) {
                    const uint16_t *kc = ly->kv_cache + (size_t)((p_lo + j) % ly->kv_slots)*KV;
                    float s = 0.f; for (int d = 0; d < KV; d++) s += q[d]*ds4f_bf16f(kc[d]);
                    s *= scale; sc[j] = s; if (s > mx) mx = s;
                }
                float denom = expf(ly->attn_sink[h] - mx);
                for (int j = 0; j < nP; j++) { sc[j] = expf(sc[j] - mx); denom += sc[j]; }
                float inv = 1.0f/denom;
                float *out = m->p_attn + (size_t)mm*H + (size_t)h*HD;
                for (int d = 0; d < HD; d++) out[d] = 0.f;
                for (int j = 0; j < nP; j++) {
                    float w = sc[j]*inv; const uint16_t *kc = ly->kv_cache + (size_t)((p_lo + j) % ly->kv_slots)*KV;
                    for (int d = 0; d < HD; d++) out[d] += w*ds4f_bf16f(kc[d]);
                }
                ds4f_rope_apply(out + nope, T->rcos, T->rsin, pos, T->half, 1);
            }
            continue;
        }
        const float *qbase = m->p_q + (size_t)mm*H + (size_t)h0*HD;
        float *obase = m->p_attn + (size_t)mm*H + (size_t)h0*HD;
        const uint16_t *kvc = ly->kv_cache + (size_t)p_lo*KV;   /* latent row j = kvc + j*KV */
        #define QH(hh) (qbase + (size_t)(hh)*HD)
        /* ---- Phase 1: sc[hh][j] = scale * (q[hh] . kc[j]); kc[j] reused x HBLK ---- */
        for (int j = 0; j < nP; j++) {
            const uint16_t *kc = kvc + (size_t)j*KV;
            svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f),
                        a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
            for (int d = 0; d < KV; d += 16) {
                svbool_t pg = svwhilelt_b32(d, KV);
                svfloat32_t kv = ds4f_ld_bf16x(pg, kc + d);
                a0 = svmla_f32_x(pg, a0, svld1(pg, QH(0)+d), kv);
                a1 = svmla_f32_x(pg, a1, svld1(pg, QH(1)+d), kv);
                a2 = svmla_f32_x(pg, a2, svld1(pg, QH(2)+d), kv);
                a3 = svmla_f32_x(pg, a3, svld1(pg, QH(3)+d), kv);
                a4 = svmla_f32_x(pg, a4, svld1(pg, QH(4)+d), kv);
                a5 = svmla_f32_x(pg, a5, svld1(pg, QH(5)+d), kv);
                a6 = svmla_f32_x(pg, a6, svld1(pg, QH(6)+d), kv);
                a7 = svmla_f32_x(pg, a7, svld1(pg, QH(7)+d), kv);
            }
            sc[0*wpad+j]=scale*svaddv_f32(pt,a0); sc[1*wpad+j]=scale*svaddv_f32(pt,a1);
            sc[2*wpad+j]=scale*svaddv_f32(pt,a2); sc[3*wpad+j]=scale*svaddv_f32(pt,a3);
            sc[4*wpad+j]=scale*svaddv_f32(pt,a4); sc[5*wpad+j]=scale*svaddv_f32(pt,a5);
            sc[6*wpad+j]=scale*svaddv_f32(pt,a6); sc[7*wpad+j]=scale*svaddv_f32(pt,a7);
        }
        /* ---- Phase 2: per-head softmax (sink) -> sc[hh][j] becomes weight w ---- */
        for (int hh = 0; hh < HBLK; hh++) {
            float *s = sc + (size_t)hh*wpad;
            float mx = -1e30f;
            for (int j = 0; j < nP; j++) if (s[j] > mx) mx = s[j];
            float denom = expf(ly->attn_sink[h0+hh] - mx);
            for (int j = 0; j < nP; j++) { s[j] = expf(s[j] - mx); denom += s[j]; }
            float inv = 1.0f/denom;
            for (int j = 0; j < nP; j++) s[j] *= inv;
        }
        /* ---- Phase 3: out[hh][d] = sum_j w[hh][j]*kc[j][d] (j asc => bit-identical) ---- */
        for (int d = 0; d < HD; d += 16) {
            svbool_t pg = svwhilelt_b32(d, HD);
            svfloat32_t o0=svdup_f32(0.f),o1=svdup_f32(0.f),o2=svdup_f32(0.f),o3=svdup_f32(0.f),
                        o4=svdup_f32(0.f),o5=svdup_f32(0.f),o6=svdup_f32(0.f),o7=svdup_f32(0.f);
            for (int j = 0; j < nP; j++) {
                svfloat32_t kv = ds4f_ld_bf16x(pg, kvc + (size_t)j*KV + d);
                o0 = svmla_f32_x(pg, o0, svdup_f32(sc[0*wpad+j]), kv);
                o1 = svmla_f32_x(pg, o1, svdup_f32(sc[1*wpad+j]), kv);
                o2 = svmla_f32_x(pg, o2, svdup_f32(sc[2*wpad+j]), kv);
                o3 = svmla_f32_x(pg, o3, svdup_f32(sc[3*wpad+j]), kv);
                o4 = svmla_f32_x(pg, o4, svdup_f32(sc[4*wpad+j]), kv);
                o5 = svmla_f32_x(pg, o5, svdup_f32(sc[5*wpad+j]), kv);
                o6 = svmla_f32_x(pg, o6, svdup_f32(sc[6*wpad+j]), kv);
                o7 = svmla_f32_x(pg, o7, svdup_f32(sc[7*wpad+j]), kv);
            }
            svst1(pg, obase + (size_t)0*HD + d, o0); svst1(pg, obase + (size_t)1*HD + d, o1);
            svst1(pg, obase + (size_t)2*HD + d, o2); svst1(pg, obase + (size_t)3*HD + d, o3);
            svst1(pg, obase + (size_t)4*HD + d, o4); svst1(pg, obase + (size_t)5*HD + d, o5);
            svst1(pg, obase + (size_t)6*HD + d, o6); svst1(pg, obase + (size_t)7*HD + d, o7);
        }
        #undef QH
        /* ---- de-rotate each head's output rope dims @ the query pos ---- */
        for (int hh = 0; hh < HBLK; hh++)
            ds4f_rope_apply(obase + (size_t)hh*HD + nope, T->rcos, T->rsin, pos, T->half, 1);
    }
}
#endif /* __ARM_FEATURE_SVE */

#if !defined(__ARM_FEATURE_SVE)
/* x86/portable batched attention worker.  The SVE worker above groups eight
 * heads so one latent row is reused across the group.  Keep the x86 path
 * deliberately simple and reuse the AVX2 BF16 dot/AXPY helpers: this is still
 * materially faster than the old scalar fallback and, importantly, keeps the
 * M>1 prefill API available on AMD hosts.  The numerical order matches the
 * regular exact worker (head-local, key positions in ascending order). */
static void ds4f_attn_prefill_worker_x86(void *arg, int tid, int nthr) {
    ds4f_attn_pf_task *T = (ds4f_attn_pf_task *)arg;
    ds4f_model *m = T->m; ds4f_layer *ly = T->ly;
    int HD = m->cfg.q_head_dim, KV = m->cfg.kv_lora;
    int rd = m->cfg.qk_rope_dim, nope = HD - rd, nh = m->cfg.n_heads;
    int H = nh * HD, win = T->win, pos0 = T->pos0;
    long total = (long)T->M * nh, per = total / nthr, extra = total % nthr;
    long u0 = per * tid + (tid < extra ? tid : extra);
    long u1 = u0 + per + (tid < extra ? 1 : 0);
    int wpad = win > 0 ? win : 1;
#if defined(__AVX2__) && defined(__FMA__)
    /* The ordinary worker assigns one head per task, which reloads the same
     * BF16 latent row for every head. Group eight heads around one KV row on
     * AVX2; this matches the existing SVE prefill shape and keeps the j-order
     * of the softmax/value accumulation unchanged. */
    if ((nh & 7) == 0) {
        int hblocks = nh / 8;
        long grouped = (long)T->M * hblocks;
        per = grouped / nthr; extra = grouped % nthr;
        u0 = per * tid + (tid < extra ? tid : extra);
        u1 = u0 + per + (tid < extra ? 1 : 0);
        float *gsc = (float *)alloca((size_t)8 * (size_t)wpad * sizeof(float));
        float scores[8], weights[8];
        for (long u = u0; u < u1; ++u) {
            int mm = (int)(u / hblocks), h0 = (int)(u % hblocks) * 8;
            int pos = pos0 + mm, p_lo = pos - win + 1;
            if (p_lo < 0) p_lo = 0;
            int nP = pos - p_lo + 1;
            const float *q = m->p_q + (size_t)mm * H + (size_t)h0 * HD;
            for (int j = 0; j < nP; ++j) {
                const uint16_t *kc = ly->kv_cache
                    + (size_t)((p_lo + j) % ly->kv_slots) * KV;
                ds4f_avx2_score8_bf16(scores, q, HD, kc, KV);
                for (int h = 0; h < 8; ++h) gsc[(size_t)h*wpad + j] = scores[h] * T->scale;
            }
            for (int h = 0; h < 8; ++h) {
                float *s = gsc + (size_t)h*wpad;
                float mx = -1e30f;
                for (int j = 0; j < nP; ++j) if (s[j] > mx) mx = s[j];
                float denom = expf(ly->attn_sink[h0 + h] - mx);
                for (int j = 0; j < nP; ++j) { s[j] = expf(s[j] - mx); denom += s[j]; }
                float inv = 1.0f / denom;
                for (int j = 0; j < nP; ++j) s[j] *= inv;
            }
            float *out = m->p_attn + (size_t)mm * H + (size_t)h0 * HD;
            memset(out, 0, (size_t)8 * (size_t)HD * sizeof(float));
            for (int j = 0; j < nP; ++j) {
                const uint16_t *kc = ly->kv_cache
                    + (size_t)((p_lo + j) % ly->kv_slots) * KV;
                for (int h = 0; h < 8; ++h) weights[h] = gsc[(size_t)h*wpad + j];
                ds4f_avx2_axpy8_bf16(out, HD, kc, weights, HD);
            }
            for (int h = 0; h < 8; ++h)
                ds4f_rope_apply(out + (size_t)h*HD + nope, T->rcos, T->rsin, pos, T->half, 1);
        }
        return;
    }
#endif
    float *sc = (float *)alloca((size_t)wpad * sizeof(float));
    for (long u = u0; u < u1; u++) {
        int mm = (int)(u / nh), h = (int)(u % nh);
        int pos = pos0 + mm, p_lo = pos - win + 1;
        if (p_lo < 0) p_lo = 0;
        int nP = pos - p_lo + 1;
        const float *q = m->p_q + (size_t)mm * H + (size_t)h * HD;
        float mx = -1e30f;
        for (int j = 0; j < nP; j++) {
            const uint16_t *kc = ly->kv_cache
                + (size_t)((p_lo + j) % ly->kv_slots) * KV;
            float s = ds4f_attn_dot_bf16(q, kc, KV, 1) * T->scale;
            sc[j] = s;
            if (s > mx) mx = s;
        }
        float denom = expf(ly->attn_sink[h] - mx);
        for (int j = 0; j < nP; j++) {
            sc[j] = expf(sc[j] - mx);
            denom += sc[j];
        }
        float inv = 1.0f / denom;
        float *out = m->p_attn + (size_t)mm * H + (size_t)h * HD;
        memset(out, 0, (size_t)HD * sizeof(float));
        for (int j = 0; j < nP; j++) {
            const uint16_t *kc = ly->kv_cache
                + (size_t)((p_lo + j) % ly->kv_slots) * KV;
            ds4f_attn_axpy_bf16(out, kc, sc[j] * inv, HD, 1);
        }
        ds4f_rope_apply(out + nope, T->rcos, T->rsin, pos, T->half, 1);
    }
}
#endif /* !__ARM_FEATURE_SVE */

/* batched swiglu (exact clamp): g[mm][i] = silu(min(g,lim)) * clamp(u,-lim,lim). */
typedef struct { ds4f_model *m; float *g; const float *u; int n, M, gstride, ustride; float lim; } ds4f_pf_swiglu_task;
static void ds4f_pf_swiglu_worker(void *arg, int tid, int nthr) {
    ds4f_pf_swiglu_task *T = (ds4f_pf_swiglu_task *)arg;
    int M = T->M, n = T->n; float lim = T->lim;
    int per = M/nthr, extra = M%nthr;
    int m0 = per*tid + (tid<extra?tid:extra), m1 = m0 + per + (tid<extra?1:0);
    for (int mm = m0; mm < m1; mm++) {
        float *g = T->g + (size_t)mm*T->gstride; const float *u = T->u + (size_t)mm*T->ustride;
        for (int i = 0; i < n; i++)
            g[i] = ds4f_silu(g[i] > lim ? lim : g[i]) * ds4f_clampf(u[i], -lim, lim);
    }
}

/* Run independent SwiGLU tiles under one pool barrier.  Routed prefill has
 * one tile per active expert; the tiles do not alias, so the same thread
 * partition can process every expert without a barrier between them. */
typedef struct { ds4f_pf_swiglu_task *tasks; int n; } ds4f_pf_swiglu_multi_task;
static void ds4f_pf_swiglu_multi_worker(void *arg, int tid, int nthr) {
    ds4f_pf_swiglu_multi_task *T = (ds4f_pf_swiglu_multi_task *)arg;
    for (int k = 0; k < T->n; k++) {
        ds4f_pf_swiglu_task *A = &T->tasks[k];
        int per = A->M / nthr, extra = A->M % nthr;
        int m0 = per * tid + (tid < extra ? tid : extra);
        int m1 = m0 + per + (tid < extra ? 1 : 0);
        for (int mm = m0; mm < m1; mm++) {
            float *g = A->g + (size_t)mm * A->gstride;
            const float *u = A->u + (size_t)mm * A->ustride;
            for (int i = 0; i < A->n; i++)
                g[i] = ds4f_silu(g[i] > A->lim ? A->lim : g[i]) *
                       ds4f_clampf(u[i], -A->lim, A->lim);
        }
    }
}

/* batched argmax over [M, vocab]; out_tok[mm] = argmax_v logits[mm][v]. */
typedef struct { const float *logits; int *out; int vocab, M; int r0; float *val; } ds4f_pf_argmax_task;
static void ds4f_pf_argmax_worker(void *arg, int tid, int nthr) {
    ds4f_pf_argmax_task *T = (ds4f_pf_argmax_task *)arg;
    int M = T->M, V = T->vocab, per = M/nthr, extra = M%nthr;
    int m0 = per*tid + (tid<extra?tid:extra), m1 = m0 + per + (tid<extra?1:0);
    for (int mm = m0; mm < m1; mm++) {
        const float *lg = T->logits + (size_t)mm*V;       /* V = vocab, or hrows under TP_HEAD shard */
        int best = 0; float bv = lg[0];
        for (int v = 1; v < V; v++) if (lg[v] > bv) { bv = lg[v]; best = v; }
        T->out[mm] = T->r0 + best;                         /* r0 = head_r0 (0 unless TP_HEAD) -> global index */
        if (T->val) T->val[mm] = bv;                       /* local max, merged across shards by the caller */
    }
}

/* X = [M, hidden] input embeddings (token-major); pos0 = absolute position of
 * token 0; out_tok[mm] = argmax logit of token mm (head computed for all M).
 * Requires m->exact, !m->mhc, !m->tierb2, and ds4f_alloc_prefill_batch(>=M). */
static void ds4f_forward_prefill(ds4f_model *m, const float *X, int M, int pos0, int *out_tok) {
    ds4f_config *c = &m->cfg;
    int C = c->hidden, HD = c->q_head_dim, KV = c->kv_lora, H = c->n_heads*HD;
    int og = c->o_groups, gin = H / og;
    if (ds4f_prof_on < 0) { const char *e = getenv("DS4F_PROF"); ds4f_prof_on = e ? atoi(e) : 0; }
    if (!m->exact || m->mhc || m->tierb2) {
        fprintf(stderr, "ds4f_forward_prefill requires exact && !mhc && !tierb2\n"); abort(); }
    if (m->int8_kv) {   /* batched kvpost/attn workers read the bf16 kv_cache (NULL under int8 KV) */
        fprintf(stderr, "ds4f_forward_prefill incompatible with DS4F_INT8_KV (use token-at-a-time)\n"); abort(); }
    if (!m->p_x || m->m_tile < M) { fprintf(stderr, "prefill batch buffers too small (m_tile=%d M=%d)\n", m->m_tile, M); abort(); }
    for (int mm = 0; mm < M; mm++) memcpy(m->p_x + (size_t)mm*C, X + (size_t)mm*C, (size_t)C*4);

    int tps = (m->sh_rows < c->shared_inter);   /* TP shared-expert (sh_w1/w3 col-shard) */
    int tps2 = tps && (m->sh2_rows < C);        /* optional hidden-row shard of sh_w2 */
    for (int L = 0; L < c->n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        if (m->gpu_dense_layer_prefetch && m->gpu_dense_stream_prefill_only &&
            m->gpu_dense_layer_prefetch(m->gpu_dense_ctx, ly) != 0) {
            fprintf(stderr, "ds4f: GPU layer prefetch failed at layer %d\n", L);
            abort();
        }
        if (m->gpu_dense_layer_begin && m->gpu_dense_stream_prefill_only &&
            m->gpu_dense_layer_begin(m->gpu_dense_ctx, ly) != 0) {
            fprintf(stderr, "ds4f: GPU layer residency setup failed at layer %d\n", L);
            abort();
        }
        if (m->gpu_dense_layer_prefetch && m->gpu_dense_stream_prefill_only &&
            L + 1 < c->n_layers &&
            m->gpu_dense_layer_prefetch(m->gpu_dense_ctx, &m->layers[L + 1]) != 0) {
            fprintf(stderr, "ds4f: GPU layer prefetch failed after layer %d\n", L);
            abort();
        }
        int ratio = c->compress_ratios[L];
        const float *rcos = ratio ? m->rope_comp_cos : m->rope_dense_cos;
        const float *rsin = ratio ? m->rope_comp_sin : m->rope_dense_sin;
        /* ---- MLA q/kv projections ---- */
        { DS4F_TIC();
        { ds4f_pf_rms_task t = { m, m->p_hn, m->p_x, ly->attn_norm, C, M, C, C };
          ds4f_pool_run(m->pool, ds4f_pf_rmsnorm_worker, &t); }
        ds4f_gemm_pair(m, m->p_qlat, &ly->wq_a, m->p_kvlat, &ly->wkv,
                       m->p_hn, M, c->q_lora, KV, C);
        { ds4f_pf_rms_task t = { m, m->p_qlat, m->p_qlat, ly->q_norm, c->q_lora, M, c->q_lora, c->q_lora };
          ds4f_pool_run(m->pool, ds4f_pf_rmsnorm_worker, &t); }
        ds4f_gemm(m, m->p_q, &ly->wq_b, m->p_qlat, M, H, c->q_lora);
        { ds4f_pf_qnr_task t = { m, pos0, M, rcos, rsin };
          ds4f_pool_run(m->pool, ds4f_pf_qnr_worker, &t); }
        { ds4f_pf_kv_task t = { m, ly, pos0, M, rcos, rsin };
          ds4f_pool_run(m->pool, ds4f_pf_kvpost_worker, &t); }
        DS4F_TOC(DS4F_P_QKV); }
        /* ---- attention (sliding window + sink), over (token,head) ---- */
        { DS4F_TIC();
        int gpu_attn_ok = 0;
        if (m->gpu_prefill_attn) {
            gpu_attn_ok = m->gpu_prefill_attn(
                m->gpu_dense_ctx, m->p_attn, m->p_q, ly->kv_cache,
                ly->attn_sink, M, pos0, c->n_heads, HD, KV,
                ly->kv_slots, c->window_size, 1.0f / sqrtf((float)HD)) == 0;
            if (gpu_attn_ok) {
                int nope = HD - c->qk_rope_dim;
                for (int mm = 0; mm < M; ++mm)
                    for (int h = 0; h < c->n_heads; ++h)
                        ds4f_rope_apply(m->p_attn + ((size_t)mm*c->n_heads + h)*HD + nope,
                                        rcos, rsin, pos0 + mm, c->qk_rope_dim/2, 1);
            }
        }
        if (!gpu_attn_ok) {
#if defined(__ARM_FEATURE_SVE)
        ds4f_attn_pf_task at = { m, ly, pos0, M, 1.0f/sqrtf((float)HD),
                                 c->window_size, c->qk_rope_dim/2, rcos, rsin };
        ds4f_pool_run(m->pool, ds4f_attn_prefill_worker, &at);
#else
        ds4f_attn_pf_task at = { m, ly, pos0, M, 1.0f/sqrtf((float)HD),
                                 c->window_size, c->qk_rope_dim/2, rcos, rsin };
        ds4f_pool_run(m->pool, ds4f_attn_prefill_worker_x86, &at);
#endif
        }
        DS4F_TOC(DS4F_P_ATTN); }
        /* ---- grouped low-rank o-projection ---- */
        { DS4F_TIC();
        int tpo = (m->oi_rows < c->o_inter);   /* TP_OPROJ: wo_a o_inter row-shard (wo_b full over zero-pad) */
        int gpu_oproj = !tpo && m->gpu_oproj && ly->wo_a.gpu_id >= 0 && ly->wo_b.gpu_id >= 0 &&
            m->gpu_oproj(m->gpu_dense_ctx,m->p_o,&ly->wo_a,&ly->wo_b,m->p_attn,
                         M,og,gin,c->o_lora,H,C,c->o_inter)==0;
        if(!gpu_oproj){
        ds4f_gemm_task *o_tasks = (ds4f_gemm_task *)alloca((size_t)c->o_groups * sizeof(*o_tasks));
        ds4f_tensor *o_views = (ds4f_tensor *)alloca((size_t)c->o_groups * sizeof(*o_views));
        int no_tasks = 0;
        if (tpo) {
            /* Only the owned o_inter rows [oi0, oi0+oi_rows) are computed. wo_a is block-diagonal:
             * o_inter row r belongs to group r/o_lora, contracting that group's gin inputs. The shard
             * may straddle group boundaries, so loop the overlapping groups and slice within the shard
             * (local row = global - oi0). Then full wo_b over the zero-padded p_o1 -> per-node PARTIAL
             * p_o, summed by the [M,C] attention-residual reduce below (mirrors the decode TP_OPROJ). */
            memset(m->p_o1, 0, (size_t)M*c->o_inter*4);
            int olora = c->o_lora, g_lo = m->oi0 / olora, g_hi = (m->oi0 + m->oi_rows - 1) / olora;
            for (int g = g_lo; g <= g_hi; g++) {
                int rlo = g*olora > m->oi0 ? g*olora : m->oi0;
                int rhi = (g+1)*olora < m->oi0 + m->oi_rows ? (g+1)*olora : m->oi0 + m->oi_rows;
                int oi = no_tasks++;
                o_views[oi] = ds4f_row_slice(&ly->wo_a, rlo - m->oi0, rhi - rlo);
                o_tasks[oi] = (ds4f_gemm_task){ m, m->p_o1 + rlo, &o_views[oi],
                    m->p_attn + (size_t)g*gin, M, c->o_inter, H, NULL, NULL, NULL, NULL };
            }
        } else for (int g = 0; g < og; g++) {
            int oi = no_tasks++;
            o_views[oi] = ds4f_row_slice(&ly->wo_a, g * c->o_lora, c->o_lora);
            o_tasks[oi] = (ds4f_gemm_task){ m, m->p_o1 + (size_t)g*c->o_lora, &o_views[oi],
                m->p_attn + (size_t)g*gin, M, c->o_inter, H, NULL, NULL, NULL, NULL };
        }
        ds4f_gemm_multi(m, o_tasks, no_tasks);
        ds4f_gemm(m, m->p_o, &ly->wo_b, m->p_o1, M, C, c->o_inter);   /* p_o = PARTIAL if tpo */
        if (tpo && m->ar_cb) m->ar_cb(m->p_o, C*M, m->ar_ctx);        /* attention-residual reduce (sum partials) */
        }
        for (int mm = 0; mm < M; mm++) { float *x = m->p_x + (size_t)mm*C, *o = m->p_o + (size_t)mm*C;
            for (int i = 0; i < C; i++) x[i] += o[i]; }
        DS4F_TOC(DS4F_P_OPROJ); }
        /* ---- FFN: shared expert ---- */
        { DS4F_TIC();
        { ds4f_pf_rms_task t = { m, m->p_h2, m->p_x, ly->ffn_norm, C, M, C, C };
          ds4f_pool_run(m->pool, ds4f_pf_rmsnorm_worker, &t); }
        /* TP_SHARED: sh_w1/w3 col-shard -> write the [sh_r0, sh_r0+sh_rows) columns of a zero-padded
         * [M, shared_inter] buffer (Ystride=shared_inter); full sh_w2 over the zero-pad -> per-node
         * PARTIAL p_moe, folded into the routed [M,C] reduce below (one reduce, like the decode path). */
        if (tps) { memset(m->p_shg, 0, (size_t)M*c->shared_inter*4); memset(m->p_shu, 0, (size_t)M*c->shared_inter*4); }
        /* Fused device-resident form: the two [M, shared_inter] intermediates
         * never cross host memory.  Only the untiled (non-TP) case qualifies;
         * the backend declines any tensor kind it does not handle. */
        int shared_fused = 0;
        if (!tps && !tps2 && m->gpu_shared_ffn && !m->gpu_exact_prefill &&
            ly->sh_w1.gpu_id >= 0 && ly->sh_w3.gpu_id >= 0 && ly->sh_w2.gpu_id >= 0)
            shared_fused = m->gpu_shared_ffn(
                m->gpu_dense_ctx, m->p_moe, &ly->sh_w1, &ly->sh_w3, &ly->sh_w2,
                m->p_h2, M, c->shared_inter, C, c->swiglu_limit) == 0;
        if (!shared_fused) {
        ds4f_gemm_pair(m, m->p_shg + m->sh_r0, &ly->sh_w1,
                       m->p_shu + m->sh_r0, &ly->sh_w3,
                       m->p_h2, M, c->shared_inter, c->shared_inter, C);
        { ds4f_pf_swiglu_task t = { m, m->p_shg, m->p_shu, c->shared_inter, M,
                                    c->shared_inter, c->shared_inter, c->swiglu_limit };
          ds4f_pool_run(m->pool, ds4f_pf_swiglu_worker, &t); }
        }
        /* Shared down and router are independent after the shared SwiGLU.
         * Keep both GEMMs under one dispatch; the multi-GEMM worker supports
         * their mixed FP8/BF16 tensor types. */
        ds4f_gemm_task shared_router[2];
        if (tps2) shared_router[0] = (ds4f_gemm_task){ m, m->p_exO, &ly->sh_w2,
            m->p_shg, M, m->sh2_rows, c->shared_inter, NULL, NULL, NULL, NULL };
        else shared_router[0] = (ds4f_gemm_task){ m, m->p_moe, &ly->sh_w2,
            m->p_shg, M, C, c->shared_inter, NULL, NULL, NULL, NULL };   /* partial if tps */
        shared_router[1] = (ds4f_gemm_task){ m, m->p_router, &ly->gate,
            m->p_h2, M, c->n_experts, C, NULL, NULL, NULL, NULL };
        if (shared_fused)   /* sh_w2 already done on the device */
            ds4f_gemm(m, m->p_router, &ly->gate, m->p_h2, M, c->n_experts, C);
        else
            ds4f_gemm_multi(m, shared_router, 2);
        if (tps2) for (int mm = 0; mm < M; mm++) {
            float *mo = m->p_moe + (size_t)mm*C;
            memset(mo, 0, (size_t)C * sizeof(float));
            memcpy(mo + m->sh2_r0, m->p_exO + (size_t)mm*m->sh2_rows,
                   (size_t)m->sh2_rows * sizeof(float));
        }
        DS4F_TOC(DS4F_P_SHARED); }
        /* ---- FFN: routed experts (owned-only, EXPERT-GROUPED batched GEMM) ----
         * Bucket the M tokens by their assigned owned expert, then run ONE
         * [Me,*] MXFP4 GEMM per expert weight (weight read from HBM once,
         * amortized over its Me tokens) in place of M*n_active per-token matvecs.
         * The per-token routed sum is reassociated into expert order; bit-similar,
         * argmax-exact. Each expert's GEMM output for a token is itself
         * bit-identical to the old per-token matvec (same kernel, full-K sweep). */
        { DS4F_TIC();
        int no = ly->n_owned;
        for (int s = 0; s < no; s++) m->ex_cnt[s] = 0;
        for (int mm = 0; mm < M; mm++) {
            int idx[8]; float wt[8];
            ds4f_topk_exact(m->p_router + (size_t)mm*c->n_experts, ly->gate_bias,
                            c->n_experts, c->n_active, idx, wt, c->routed_scale);
            ds4f_route_observe(m, L, idx, c->n_active);
            float *route = m->p_route + (size_t)mm*C;
            for (int i = 0; i < C; i++) route[i] = 0.f;
            for (int k = 0; k < c->n_active; k++) {
                int e = idx[k]; if (e < 0) continue;
                if (e % m->ep_size != m->ep_rank) continue;
                int slot = e / m->ep_size, p = m->ex_cnt[slot]++;
                m->ex_tok[(size_t)slot*M + p] = mm;
                m->ex_wt [(size_t)slot*M + p] = wt[k];
            }
        }
        /* Stage every expert's bucket in one contiguous assignment slab.
         * Each expert still sees its original ascending token order, but all
         * gate/up GEMMs, all activations, and all down GEMMs now share one
         * pool barrier apiece. */
        int *ex_off = (int *)alloca((size_t)(no + 1) * sizeof(*ex_off));
        ds4f_gemm_task *gateup = (ds4f_gemm_task *)alloca((size_t)(2 * no) * sizeof(*gateup));
        ds4f_pf_swiglu_task *sw = (ds4f_pf_swiglu_task *)alloca((size_t)no * sizeof(*sw));
        ds4f_gemm_task *down = (ds4f_gemm_task *)alloca((size_t)no * sizeof(*down));
        ex_off[0] = 0;
        for (int s = 0; s < no; s++) ex_off[s + 1] = ex_off[s] + m->ex_cnt[s];
        int ng = 0, ns = 0, nd = 0;
        for (int s = 0; s < no; s++) {
            int cnt = m->ex_cnt[s]; if (cnt == 0) continue;
            int off = ex_off[s];
            for (int p = 0; p < cnt; p++) {                       /* gather h2 of this expert's tokens */
                const float *h2 = m->p_h2 + (size_t)m->ex_tok[(size_t)s*M + p]*C;
                float *xe = m->p_exX + (size_t)(off + p)*C;
                for (int i = 0; i < C; i++) xe[i] = h2[i];
            }
            gateup[ng++] = (ds4f_gemm_task){ m,
                m->p_exG + (size_t)off*c->moe_inter, &ly->ex_w1[s],
                m->p_exX + (size_t)off*C, cnt, c->moe_inter, C, NULL, NULL, NULL, NULL };
            gateup[ng++] = (ds4f_gemm_task){ m,
                m->p_exU + (size_t)off*c->moe_inter, &ly->ex_w3[s],
                m->p_exX + (size_t)off*C, cnt, c->moe_inter, C, NULL, NULL, NULL, NULL };
            sw[ns++] = (ds4f_pf_swiglu_task){ m,
                m->p_exG + (size_t)off*c->moe_inter,
                m->p_exU + (size_t)off*c->moe_inter,
                c->moe_inter, cnt, c->moe_inter, c->moe_inter, c->swiglu_limit };
            down[nd++] = (ds4f_gemm_task){ m,
                m->p_exO + (size_t)off*C, &ly->ex_w2[s],
                m->p_exG + (size_t)off*c->moe_inter, cnt, C, c->moe_inter, NULL, NULL, NULL, NULL };
        }
        ds4f_gemm_multi(m, gateup, ng);
        { ds4f_pf_swiglu_multi_task t = { sw, ns };
          ds4f_pool_run(m->pool, ds4f_pf_swiglu_multi_worker, &t); }
        ds4f_gemm_multi(m, down, nd);
        for (int s = 0; s < no; s++) {
            int cnt = m->ex_cnt[s]; if (cnt == 0) continue;
            int off = ex_off[s];
            for (int p = 0; p < cnt; p++) {                       /* scatter weighted output into route */
                int mm = m->ex_tok[(size_t)s*M + p]; float w = m->ex_wt[(size_t)s*M + p];
                float *route = m->p_route + (size_t)mm*C;
                const float *o = m->p_exO + (size_t)(off + p)*C;
                for (int i = 0; i < C; i++) route[i] += w * o[i];
            }
        }
        DS4F_TOC(DS4F_P_EXPERTS); }
        /* ---- EP combine: ONE [M,C] all-reduce (amortizes the per-op latency by M). Under TP_SHARED
         * the partial shared p_moe is folded into p_route FIRST so it rides the same reduce. ---- */
        if (tps) for (int mm = 0; mm < M; mm++) {
            float *mo = m->p_moe + (size_t)mm*C, *ro = m->p_route + (size_t)mm*C;
            for (int i = 0; i < C; i++) ro[i] += mo[i]; }
        if (m->ar_cb) { DS4F_TIC(); m->ar_cb(m->p_route, C*M, m->ar_ctx); DS4F_TOC(DS4F_P_COMM); }
        /* ---- residual: shared(local, or folded-into-route under tps) + routed(reduced) ---- */
        for (int mm = 0; mm < M; mm++) {
            float *x = m->p_x + (size_t)mm*C, *mo = m->p_moe + (size_t)mm*C, *ro = m->p_route + (size_t)mm*C;
            for (int i = 0; i < C; i++) x[i] += (tps ? 0.f : mo[i]) + ro[i];
        }
        if (ds4f_prefill_trace_on()) {
            double sx = 0.0, sk = 0.0;
            for (int i = 0; i < C; ++i) sx += fabs((double)m->p_x[i]);
            for (int i = 0; i < KV; ++i) sk += fabs((double)m->p_kvlat[i]);
            fprintf(stderr, "prefill-trace %s L%d x_l1=%.9g x_max=%.9g kv_l1=%.9g kv_max=%.9g\n",
                    m->gpu_dense_ctx ? "gpu" : "cpu", L, sx, ds4f_trace_maxabs(m->p_x, C),
                    sk, ds4f_trace_maxabs(m->p_kvlat, KV));
        }
    }
    /* ---- head: out_norm + lm_head. Production prompt ingestion needs only
     * the final token's logits; full-token logits remain the parity-test default. */
    { DS4F_TIC();
    const char *last_e = getenv("DS4F_PREFILL_LAST_LOGITS");
    int last_only = last_e && *last_e && atoi(last_e) != 0 && M > 1;
    int HM = last_only ? 1 : M;
    const float *hx = last_only ? m->p_x + (size_t)(M - 1) * C : m->p_x;
    { ds4f_pf_rms_task t = { m, m->p_hn, hx, m->out_norm, C, HM, C, C };
      ds4f_pool_run(m->pool, ds4f_pf_rmsnorm_worker, &t); }
    /* logits scratch reuses a per-token vocab buffer; allocate lazily once (sized full vocab). */
    if (!m->p_logits) m->p_logits = (float *)ds4f_mem_alloc(m->mem, (size_t)m->m_tile*(size_t)c->vocab*4, 256, 1);
    /* TP_HEAD: head is vocab-sharded (head.rows = hrows). GEMM the owned vocab rows -> p_logits[M, hrows];
     * each token's local argmax (global index head_r0+best, local max value) is merged across the shards by
     * ar_argmax_cb (M small 2-float argmax all-reduces, ONCE -- cheap, unlike a full [M,vocab] logit reduce). */
    int hrows = m->head.rows, tph = (hrows < c->vocab);
    ds4f_gemm(m, m->p_logits, &m->head, m->p_hn, HM, hrows, C);
    if (last_only) for (int mm = 0; mm < M - 1; ++mm) out_tok[mm] = -1;
    int *hout = last_only ? out_tok + M - 1 : out_tok;
    { float *hval = tph ? (float *)alloca((size_t)HM*4) : NULL;
      ds4f_pf_argmax_task t = { m->p_logits, hout, hrows, HM, m->head_r0, hval };
      ds4f_pool_run(m->pool, ds4f_pf_argmax_worker, &t);
      if (tph && m->ar_argmax_cb)
          for (int mm = 0; mm < HM; mm++) { int32_t idx = hout[mm]; float v = hval[mm];
              m->ar_argmax_cb(&v, &idx, m->ar_argmax_ctx); hout[mm] = idx; } }
    DS4F_TOC(DS4F_P_HEAD); }
}

static size_t ds4f_tb2_snap_bytes(ds4f_model *m);                                  /* defined below */
static size_t ds4f_tb2_snap_layer(ds4f_model *m, int L, char *buf, int restore);
static size_t ds4f_tb2_snap_layer_bytes(ds4f_model *m, int L);

/* M2b BATCHED VERIFY: run K positions [pos0, pos0+K) through the 43 layers + head with mHC + tier-B2,
 * BATCHING the dense GEMMs + the per-layer reduce (the amortization) and LOOPING the per-position tier-B2
 * attention in causal order (position k sees k-1's appended KV). COHERENT, not byte-identical to M=1 decode
 * (the GEMM reassociates vs the matvec); the spec loop's committed tokens are the verify's self-consistent
 * output. TP_SHARED is supported: shared up/gate is column-sharded and folded into the existing routed
 * reduction. Other dense TP modes remain disabled because they require additional [K,H] or [K,C] reductions.
 * No CP (off at moderate ctx). Requires exact+mhc+tierb2, !int8_kv, ds4f_alloc_prefill_batch(>=K).
 * out_tok[K] = per-position argmax; out_hc[K*hc*C] = per-position final HC state (for the next draft).
 * snaps (optional, [K-1] x ds4f_tb2_snap_bytes stride): mid-verify compressor-state snapshots -- slot k
 * captures every layer's ring state AFTER position pos0+k advanced it, so a partial accept of the first
 * j+1 positions restores snaps[j] directly (NO redo verify; layout matches ds4f_tb2_snap restore). */
static void ds4f_forward_verify(ds4f_model *m, const float *X, int K, int pos0, int *out_tok, float *out_hc, char *snaps) {
    ds4f_config *c = &m->cfg;
    int C = c->hidden, HD = c->q_head_dim, KV = c->kv_lora, H = c->n_heads*HD, og = c->o_groups, gin = H/og;
    float eps = 1e-6f; int hc = c->hc_mult; size_t hcC = (size_t)hc*C;
    if (K < 1 || K > m->m_tile) {
        fprintf(stderr, "ds4f_forward_verify: K=%d outside allocated tile [1,%d]\n",
                K, m->m_tile);
        abort();
    }
    size_t snap_stride = snaps ? ds4f_tb2_snap_bytes(m) : 0, snap_loff = 0;
    /* Per-position sinkhorn weights.  These used to be fixed [128] stack
     * arrays because this routine was only a speculative-decode verifier.
     * Production prompt prefill uses the same exact causal path in tiles, so
     * size the scratch from the allocated tile instead. */
    float *hc_scratch = (float *)malloc((size_t)K * (16 + 64 + 16 + 64) * sizeof(float));
    if (!hc_scratch) { fprintf(stderr, "ds4f_forward_verify: sinkhorn scratch allocation failed\n"); abort(); }
    float *pa = hc_scratch;
    float *ca = pa + (size_t)K * 16;
    float *pf = ca + (size_t)K * 64;
    float *cf = pf + (size_t)K * 16;
    if (!m->v_x4) { size_t vb = (size_t)m->m_tile*hcC*4;
        m->v_x4 = (float *)ds4f_mem_alloc(m->mem, vb, 256, 1); m->v_resid = (float *)ds4f_mem_alloc(m->mem, vb, 256, 1); }
    for (int k = 0; k < K; k++) for (int s = 0; s < hc; s++)   /* expand each input into hc streams */
        memcpy(m->v_x4 + (size_t)k*hcC + (size_t)s*C, X + (size_t)k*C, (size_t)C*4);
    for (int L = 0; L < c->n_layers; L++) {
        ds4f_layer *ly = &m->layers[L];
        if (m->gpu_dense_layer_prefetch && m->gpu_dense_stream_prefill_only &&
            m->gpu_dense_layer_prefetch(m->gpu_dense_ctx, ly) != 0) {
            fprintf(stderr, "ds4f: GPU layer prefetch failed at layer %d\n", L);
            abort();
        }
        if (m->gpu_dense_layer_begin && m->gpu_dense_stream_prefill_only &&
            m->gpu_dense_layer_begin(m->gpu_dense_ctx, ly) != 0) {
            fprintf(stderr, "ds4f: GPU layer residency setup failed at layer %d\n", L);
            abort();
        }
        if (m->gpu_dense_layer_prefetch && m->gpu_dense_stream_prefill_only &&
            L + 1 < c->n_layers &&
            m->gpu_dense_layer_prefetch(m->gpu_dense_ctx, &m->layers[L + 1]) != 0) {
            fprintf(stderr, "ds4f: GPU layer prefetch failed after layer %d\n", L);
            abort();
        }
        int ratio = c->compress_ratios[L];
        const float *rcos = ratio ? m->rope_comp_cos : m->rope_dense_cos;
        const float *rsin = ratio ? m->rope_comp_sin : m->rope_dense_sin;
        /* mHC pre: one pooled mix/RMS/sinkhorn/collapse sequence for the whole tile. */
        ds4f_hc_pre_batch(m, m->v_x4, K, ly->hc_attn_fn, ly->hc_attn_scale, ly->hc_attn_base,
                          m->p_x, pa, 16, ca, 64);
        memcpy(m->v_resid, m->v_x4, (size_t)K*hcC*4);
        /* batched q/kv projections */
        { ds4f_pf_rms_task t = { m, m->p_hn, m->p_x, ly->attn_norm, C, K, C, C };
          ds4f_pool_run(m->pool, ds4f_pf_rmsnorm_worker, &t); }
        ds4f_gemm(m, m->p_qlat, &ly->wq_a, m->p_hn, K, c->q_lora, C);
        { ds4f_pf_rms_task t = { m, m->p_qlat, m->p_qlat, ly->q_norm, c->q_lora, K, c->q_lora, c->q_lora };
          ds4f_pool_run(m->pool, ds4f_pf_rmsnorm_worker, &t); }
        ds4f_gemm(m, m->p_q, &ly->wq_b, m->p_qlat, K, H, c->q_lora);
        { ds4f_pf_qnr_task t = { m, pos0, K, rcos, rsin };
          ds4f_pool_run(m->pool, ds4f_pf_qnr_worker, &t); }
        ds4f_gemm(m, m->p_kvlat, &ly->wkv, m->p_hn, K, KV, C);
        /* CSA indexer q-projection is independent per prompt position. Reuse
         * the same batched GEMM strategy as the main q projection; the
         * per-position index scan below consumes one row through s_idx_qpre. */
        int idxg_pf = 0, idxHhd = 0;
        if (m->tierb2 && ratio == 4 && ly->idx_wq_b) {
            idxHhd = c->index_n_heads * c->index_head_dim;
            if (!m->v_idxq) m->v_idxq = (float *)ds4f_mem_alloc(m->mem, (size_t)m->m_tile*idxHhd*4, 256, 1);
            ds4f_tensor idxwq = { ly->idx_wq_b, NULL, DS4F_BF16, idxHhd, c->q_lora, -1 };
            ds4f_gemm(m, m->v_idxq, &idxwq, m->p_qlat, K, idxHhd, c->q_lora);
            idxg_pf = 1;
        }
        /* per-position tier-B2 attention (causal: append KV then attend, in order) */
        for (int k = 0; k < K; k++) {
            int pos = pos0 + k;
            float *kvl = m->p_kvlat + (size_t)k*KV;
            ds4f_rmsnorm(kvl, kvl, ly->kv_norm, KV, eps);
            ds4f_rope_apply(kvl + (KV - c->qk_rope_dim), rcos, rsin, pos, c->qk_rope_dim/2, 0);
            { uint16_t *dst = ly->kv_cache + (size_t)(pos % ly->kv_slots)*KV;
              for (int d = 0; d < KV; d++) dst[d] = ds4f_f32bf(kvl[d]); }
            memcpy(m->s_hn, m->p_hn + (size_t)k*C, (size_t)C*4);     /* compressor reads s_hn */
            memcpy(m->s_q,  m->p_q  + (size_t)k*H, (size_t)H*4);     /* indexer + attention read s_q */
            m->s_idx_qpre = idxg_pf ? m->v_idxq + (size_t)k*idxHhd : NULL;
            if (m->tierb2 && ratio) ds4f_tb2_prepare(m, ly, ratio, pos, rcos, rsin);
            m->s_idx_qpre = NULL;
            if (snaps && ratio && k < K-1)                           /* state after THIS position, this layer */
                ds4f_tb2_snap_layer(m, L, snaps + (size_t)k*snap_stride + snap_loff, 0);
            m->cp_gather = 0;
            if (m->tierb2 && ratio) { ds4f_attn_ex_task at = { m, ly, pos, 1.0f/sqrtf((float)HD),
                                          c->window_size, c->qk_rope_dim/2, rcos, rsin };
                if (!ds4f_attn_tb2_gemm(m, &at)) ds4f_pool_run(m->pool, ds4f_attn_tb2_worker, &at);
            } else { ds4f_attn_ex_task at = { m, ly, pos, 1.0f/sqrtf((float)HD),
                                          c->window_size, c->qk_rope_dim/2, rcos, rsin };
                ds4f_pool_run(m->pool, ds4f_attn_exact_worker, &at); }
            memcpy(m->p_attn + (size_t)k*H, m->s_attn, (size_t)H*4);
        }
        if (snaps) snap_loff += ds4f_tb2_snap_layer_bytes(m, L);
        /* batched grouped low-rank o-projection (no-TP) */
        for (int g = 0; g < og; g++) {
            ds4f_tensor vg = ds4f_row_slice(&ly->wo_a, g*c->o_lora, c->o_lora);
            ds4f_gemm(m, m->p_o1 + (size_t)g*c->o_lora, &vg, m->p_attn + (size_t)g*gin, K, c->o_inter, H);
        }
        ds4f_gemm(m, m->p_o, &ly->wo_b, m->p_o1, K, C, c->o_inter);
        ds4f_hc_post_batch(m, m->v_x4, K, m->v_resid, m->p_o, pa, 16, ca, 64);
        /* mHC pre (ffn). */
        ds4f_hc_pre_batch(m, m->v_x4, K, ly->hc_ffn_fn, ly->hc_ffn_scale, ly->hc_ffn_base,
                          m->p_x, pf, 16, cf, 64);
        memcpy(m->v_resid, m->v_x4, (size_t)K*hcC*4);
        { ds4f_pf_rms_task t = { m, m->p_h2, m->p_x, ly->ffn_norm, C, K, C, C };
          ds4f_pool_run(m->pool, ds4f_pf_rmsnorm_worker, &t); }
        /* shared expert. TP_SHARED shards the expensive up/gate projections;
         * TP_SHARED_FULL additionally shards the hidden rows of the down
         * projection. The output partial is folded into the existing routed
         * [K,C] reduction below, so it adds no collective. */
        int tps = (m->sh_rows < c->shared_inter);
        int tps2 = tps && (m->sh2_rows < C);
        if (tps) {
            memset(m->p_shg, 0, (size_t)K * c->shared_inter * sizeof(float));
            memset(m->p_shu, 0, (size_t)K * c->shared_inter * sizeof(float));
        }
        ds4f_gemm(m, m->p_shg + (tps ? m->sh_r0 : 0), &ly->sh_w1,
                  m->p_h2, K, c->shared_inter, C);
        ds4f_gemm(m, m->p_shu + (tps ? m->sh_r0 : 0), &ly->sh_w3,
                  m->p_h2, K, c->shared_inter, C);
        { ds4f_pf_swiglu_task t = { m, m->p_shg, m->p_shu, c->shared_inter, K,
                                    c->shared_inter, c->shared_inter, c->swiglu_limit };
          ds4f_pool_run(m->pool, ds4f_pf_swiglu_worker, &t); }
        if (tps2) {
            ds4f_gemm(m, m->p_exO, &ly->sh_w2, m->p_shg, K, m->sh2_rows, c->shared_inter);
            for (int k = 0; k < K; k++) {
                float *mo = m->p_moe + (size_t)k*C;
                memset(mo, 0, (size_t)C * sizeof(float));
                memcpy(mo + m->sh2_r0, m->p_exO + (size_t)k*m->sh2_rows,
                       (size_t)m->sh2_rows * sizeof(float));
            }
        } else ds4f_gemm(m, m->p_moe, &ly->sh_w2, m->p_shg, K, C, c->shared_inter);
        /* router + routed experts (bucketed batched GEMM, reuse the prefill scheme) */
        ds4f_gemm(m, m->p_router, &ly->gate, m->p_h2, K, c->n_experts, C);
        { int no = ly->n_owned;
          for (int s = 0; s < no; s++) m->ex_cnt[s] = 0;
          for (int k = 0; k < K; k++) {
              int idx[8]; float wt[8];
              ds4f_topk_exact(m->p_router + (size_t)k*c->n_experts, ly->gate_bias,
                              c->n_experts, c->n_active, idx, wt, c->routed_scale);
              ds4f_route_observe(m, L, idx, c->n_active);
              float *route = m->p_route + (size_t)k*C;
              for (int i = 0; i < C; i++) route[i] = 0.f;
              for (int a = 0; a < c->n_active; a++) {
                  int e = idx[a]; if (e < 0 || e % m->ep_size != m->ep_rank) continue;
                  int slot = e / m->ep_size, p = m->ex_cnt[slot]++;
                  m->ex_tok[(size_t)slot*m->m_tile + p] = k; m->ex_wt[(size_t)slot*m->m_tile + p] = wt[a];
              }
          }
          for (int s = 0; s < no; s++) {
              int cnt = m->ex_cnt[s]; if (!cnt) continue;
              for (int p = 0; p < cnt; p++) { const float *h2 = m->p_h2 + (size_t)m->ex_tok[(size_t)s*m->m_tile+p]*C;
                  float *xe = m->p_exX + (size_t)p*C; for (int i = 0; i < C; i++) xe[i] = h2[i]; }
              ds4f_gemm(m, m->p_exG, &ly->ex_w1[s], m->p_exX, cnt, c->moe_inter, C);
              ds4f_gemm(m, m->p_exU, &ly->ex_w3[s], m->p_exX, cnt, c->moe_inter, C);
              { ds4f_pf_swiglu_task t = { m, m->p_exG, m->p_exU, c->moe_inter, cnt,
                                          c->moe_inter, c->moe_inter, c->swiglu_limit };
                ds4f_pool_run(m->pool, ds4f_pf_swiglu_worker, &t); }
              ds4f_gemm(m, m->p_exO, &ly->ex_w2[s], m->p_exG, cnt, C, c->moe_inter);
              for (int p = 0; p < cnt; p++) { int k = m->ex_tok[(size_t)s*m->m_tile+p]; float w = m->ex_wt[(size_t)s*m->m_tile+p];
                  float *route = m->p_route + (size_t)k*C; const float *o = m->p_exO + (size_t)p*C;
                  for (int i = 0; i < C; i++) route[i] += w * o[i]; }
          } }
        if (tps) for (int k = 0; k < K; k++) {
            float *mo = m->p_moe + (size_t)k*C, *ro = m->p_route + (size_t)k*C;
            for (int i = 0; i < C; i++) ro[i] += mo[i];
        }
        if (m->ar_cb) m->ar_cb(m->p_route, C*K, m->ar_ctx);          /* EP combine: one [K,C] reduce */
        for (int k = 0; k < K; k++) {                                /* moe out = shared + routed */
            float *mo = m->p_moe + (size_t)k*C, *ro = m->p_route + (size_t)k*C, *o = m->p_o + (size_t)k*C;
            for (int i = 0; i < C; i++) o[i] = (tps ? 0.f : mo[i]) + ro[i];
        }
        ds4f_hc_post_batch(m, m->v_x4, K, m->v_resid, m->p_o, pf, 16, cf, 64);
    }
    /* head: per-position hc_head collapse -> out_norm (batched) -> lm_head GEMM -> per-position argmax */
    for (int k = 0; k < K; k++) ds4f_hc_head(m, m->v_x4 + (size_t)k*hcC, m->p_x + (size_t)k*C);
    { ds4f_pf_rms_task t = { m, m->p_hn, m->p_x, m->out_norm, C, K, C, C };
      ds4f_pool_run(m->pool, ds4f_pf_rmsnorm_worker, &t); }
    if (!m->p_logits) m->p_logits = (float *)ds4f_mem_alloc(m->mem, (size_t)m->m_tile*(size_t)c->vocab*4, 256, 1);
    int hrows = m->head.rows, tph = (hrows < c->vocab);
    ds4f_gemm(m, m->p_logits, &m->head, m->p_hn, K, hrows, C);
    { static int vdbg = -1; if (vdbg < 0) { const char *e = getenv("DS4F_VRF_DBG"); vdbg = e ? atoi(e) : 0; }
      if (vdbg) { double nx=0, nh=0, nl=0; float mx=m->p_logits[0]; int mi=0;
        for (int i = 0; i < C; i++) { nx += (double)m->p_x[i]*m->p_x[i]; nh += (double)m->p_hn[i]*m->p_hn[i]; }
        for (int v = 0; v < hrows; v++) { float l = m->p_logits[v]; nl += (double)l*l; if (l > mx) { mx = l; mi = v; } }
        fprintf(stderr, "[VRFDBG r%d] pos0=%d K=%d |hc_head|=%.4g |hn|=%.4g |logit|=%.4g max=%.4g@%d hrows=%d r0=%d tph=%d\n",
                m->ep_rank, pos0, K, sqrt(nx), sqrt(nh), sqrt(nl), mx, mi, hrows, m->head_r0, tph); fflush(stderr); } }
    { float *hval = tph ? (float *)alloca((size_t)K*4) : NULL;
      ds4f_pf_argmax_task t = { m->p_logits, out_tok, hrows, K, m->head_r0, hval };
      ds4f_pool_run(m->pool, ds4f_pf_argmax_worker, &t);
      if (tph && m->ar_argmax_cb) for (int k = 0; k < K; k++) { int32_t idx = out_tok[k]; float v = hval[k];
          m->ar_argmax_cb(&v, &idx, m->ar_argmax_ctx); out_tok[k] = idx; } }
    if (out_hc) memcpy(out_hc, m->v_x4, (size_t)K*hcC*4);
    free(hc_scratch);
}

/* DS4F_MTP forward (STUB -- scaffold only). Predicts the token after `next_id` given the main model's
 * last hidden state x[hidden] at `pos`. Reference (MTPBlock.forward, inference/model.py):
 *   e  = enorm(embed(next_id));   h = hnorm(x)
 *   xf = e_proj(e) + h_proj(h)                       // the MTP fusion
 *   xf = <standard transformer block: attn + MoE>(xf, pos)   // m->mtp (reuses the per-layer body)
 *   logits = head(norm=mtp_norm, hc=mtp_hc_*)(xf)    // shares the MAIN head weight (m->head)
 * The block-forward body lives inline in ds4f_forward_token; extracting it (M=1) + the embed lookup +
 * the draft/verify/accept loop (with KV rollback) is the follow-on -- and the spec VERIFY needs the
 * M=K batched-decode rework (the per-token attn/tb2/KV/CP-gather path). Returns -1 until wired. */
/* Compressor ring-state snapshot/restore -- the ONLY per-position decode state that does not roll back
 * trivially (the KV ring + frozen int4 slots self-heal on the next write). Speculative decode snapshots
 * it before a draft position and restores on rejection. Validated in the frozen + no-HCA-slot regime
 * (states-only; the calbuf/caln only change pre-freeze). Sizing via ds4f_tb2_snap_bytes. */
static size_t ds4f_tb2_snap_bytes(ds4f_model *m) {
    ds4f_config *c = &m->cfg; size_t tot = 0;
    for (int L = 0; L < c->n_layers; L++) {
        int ratio = c->compress_ratios[L]; if (!ratio) continue;
        int coff = (ratio == 4) ? 2 : 1, W = coff*c->kv_lora;
        tot += 2*(size_t)coff*ratio*W*4;
        if (ratio == 4) { int iW = 2*c->index_head_dim; tot += 2*(size_t)2*ratio*iW*4; }
    }
    return tot;
}
static size_t ds4f_tb2_snap_layer(ds4f_model *m, int L, char *buf, int restore) {
    ds4f_config *c = &m->cfg;
    int ratio = c->compress_ratios[L]; if (!ratio) return 0;
    ds4f_layer *ly = &m->layers[L];
    int coff = (ratio == 4) ? 2 : 1, W = coff*c->kv_lora; size_t sz = (size_t)coff*ratio*W*4, off = 0;
    char *a = (char *)ly->cmp_kv_state, *b = (char *)ly->cmp_score_state;
    if (restore) memcpy(a, buf+off, sz); else memcpy(buf+off, a, sz); off += sz;
    if (restore) memcpy(b, buf+off, sz); else memcpy(buf+off, b, sz); off += sz;
    if (ratio == 4 && ly->idx_cmp_kv_state) {
        int iW = 2*c->index_head_dim; size_t isz = (size_t)2*ratio*iW*4;
        char *ia = (char *)ly->idx_cmp_kv_state, *ib = (char *)ly->idx_cmp_score_state;
        if (restore) memcpy(ia, buf+off, isz); else memcpy(buf+off, ia, isz); off += isz;
        if (restore) memcpy(ib, buf+off, isz); else memcpy(buf+off, ib, isz); off += isz;
    }
    return off;
}
static size_t ds4f_tb2_snap_layer_bytes(ds4f_model *m, int L) {
    int ratio = m->cfg.compress_ratios[L]; if (!ratio) return 0;
    int coff = (ratio == 4) ? 2 : 1, W = coff*m->cfg.kv_lora;
    size_t sz = 2*(size_t)coff*ratio*W*4;
    if (ratio == 4 && m->layers[L].idx_cmp_kv_state) sz += 2*(size_t)2*ratio*(2*m->cfg.index_head_dim)*4;
    return sz;
}
static void ds4f_tb2_snap(ds4f_model *m, char *buf, int restore) {
    size_t off = 0;
    for (int L = 0; L < m->cfg.n_layers; L++) off += ds4f_tb2_snap_layer(m, L, buf+off, restore);
}
/* skip_head=1: KV-maintenance-only call (backfill an MTP-KV ring hole) -- runs the fusion + layer
 * (appends mt->kv_cache@pos, leaves the MTP hidden in m->s_x4) but skips the lm_head; returns -1. */
static int ds4f_mtp_predict(ds4f_model *m, const float *hc_state, const float *xe, int pos, float *logits_out, int skip_head) {
    if (!m->has_mtp) return -1;
    ds4f_config *c = &m->cfg;
    int C = c->hidden, HD = c->q_head_dim, KV = c->kv_lora, H = c->n_heads*HD;
    float eps = 1e-6f; int hc = c->hc_mult; size_t hcC = (size_t)hc*C;
    ds4f_layer *mt = &m->mtp;
    const float *rcos = m->rope_dense_cos, *rsin = m->rope_dense_sin;
    /* the MTP dense is loaded FULL (no TP) -> override the dense-TP shard ranges to full for its forward;
     * EP-sharded experts + the shared vocab head keep their sharding. Save/restore. */
    int s_h0=m->attn_h0, s_h1=m->attn_h1, s_oi0=m->oi0, s_oir=m->oi_rows, s_shr0=m->sh_r0, s_shr=m->sh_rows;
    m->attn_h0=0; m->attn_h1=c->n_heads; m->oi0=0; m->oi_rows=c->o_inter; m->sh_r0=0; m->sh_rows=c->shared_inter;
    /* ---- fusion: s_x4[k] = e_proj(enorm(xe)) + h_proj(hnorm(hc_state[k])) ---- */
    float *xn = (float *)alloca((size_t)C*4), *eo = (float *)alloca((size_t)C*4);
    ds4f_rmsnorm(xn, xe, m->mtp_enorm, C, eps);
    ds4f_matvec(m, eo, &m->mtp_e_proj, xn);                       /* e_out[C], broadcast over hc */
    for (int k = 0; k < hc; k++) {
        ds4f_rmsnorm(xn, hc_state + (size_t)k*C, m->mtp_hnorm, C, eps);
        ds4f_matvec(m, m->s_xc, &m->mtp_h_proj, xn);
        for (int i = 0; i < C; i++) m->s_x4[(size_t)k*C+i] = eo[i] + m->s_xc[i];
    }
    /* ---- one mHC layer (m->mtp): attn + MoE, NO tier-B2, NO shared expert ---- */
    float post_a[16], comb_a[64], post_f[16], comb_f[64];
    ds4f_hc_pre(m, m->s_x4, mt->hc_attn_fn, mt->hc_attn_scale, mt->hc_attn_base, m->s_xc, post_a, comb_a);
    memcpy(m->s_resid, m->s_x4, hcC*4);
    ds4f_rmsnorm(m->s_hn, m->s_xc, mt->attn_norm, C, eps);
    ds4f_matvec(m, m->s_qlat, &mt->wq_a, m->s_hn);
    ds4f_rmsnorm(m->s_qlat, m->s_qlat, mt->q_norm, c->q_lora, eps);
    ds4f_matvec(m, m->s_q, &mt->wq_b, m->s_qlat);
    ds4f_q_norm_rope_par(m, m->s_q, pos, rcos, rsin);
    ds4f_matvec(m, m->s_kvlat, &mt->wkv, m->s_hn);
    ds4f_rmsnorm(m->s_kvlat, m->s_kvlat, mt->kv_norm, KV, eps);
    ds4f_rope_apply(m->s_kvlat + (KV - c->qk_rope_dim), rcos, rsin, pos, c->qk_rope_dim/2, 0);
    { uint16_t *dst = mt->kv_cache + (size_t)(pos % mt->kv_slots)*KV;
      for (int d = 0; d < KV; d++) dst[d] = ds4f_f32bf(m->s_kvlat[d]); }
    { ds4f_attn_ex_task at = { m, mt, pos, 1.0f/sqrtf((float)HD), c->window_size, c->qk_rope_dim/2, rcos, rsin };
      ds4f_pool_run(m->pool, ds4f_attn_exact_worker, &at); }
    { int og = c->o_groups, gin = H/og;
      ds4f_matvec_blockdiag(m, m->s_o1, &mt->wo_a, m->s_attn, gin, c->o_lora, 0);
      ds4f_matvec(m, m->s_o, &mt->wo_b, m->s_o1); }
    ds4f_hc_post(m, m->s_x4, m->s_resid, m->s_o, post_a, comb_a);
    ds4f_hc_pre(m, m->s_x4, mt->hc_ffn_fn, mt->hc_ffn_scale, mt->hc_ffn_base, m->s_xc, post_f, comb_f);
    memcpy(m->s_resid, m->s_x4, hcC*4);
    ds4f_rmsnorm(m->s_h2, m->s_xc, mt->ffn_norm, C, eps);
    for (int i = 0; i < C; i++) m->s_route[i] = 0.f;
    ds4f_matvec(m, m->s_router, &mt->gate, m->s_h2);
    { int idx[8]; float wt[8];
      ds4f_topk_exact(m->s_router, mt->gate_bias, c->n_experts, c->n_active, idx, wt, c->routed_scale);
      for (int k = 0; k < c->n_active; k++) {
          int e = idx[k]; if (e < 0 || e % m->ep_size != m->ep_rank) continue;
          int slot = e / m->ep_size;
          ds4f_matvec(m, m->s_exg, &mt->ex_w1[slot], m->s_h2);
          ds4f_matvec(m, m->s_exu, &mt->ex_w3[slot], m->s_h2);
          float lim = c->swiglu_limit;
          for (int i = 0; i < c->moe_inter; i++)
              m->s_exg[i] = ds4f_silu(m->s_exg[i] > lim ? lim : m->s_exg[i]) * ds4f_clampf(m->s_exu[i], -lim, lim);
          ds4f_matvec(m, m->s_o, &mt->ex_w2[slot], m->s_exg);
          for (int i = 0; i < C; i++) m->s_route[i] += wt[k] * m->s_o[i];
      } }
    if (m->ar_cb) m->ar_cb(m->s_route, C, m->ar_ctx);            /* EP combine */
    ds4f_hc_post(m, m->s_x4, m->s_resid, m->s_route, post_f, comb_f);
    if (skip_head) {                                             /* KV-only: hidden in s_x4, no head */
        m->attn_h0=s_h0; m->attn_h1=s_h1; m->oi0=s_oi0; m->oi_rows=s_oir; m->sh_r0=s_shr0; m->sh_rows=s_shr;
        return -1;
    }
    /* ---- MTP head: hc_head(mtp params) -> mtp_norm -> shared lm_head -> argmax ---- */
    ds4f_hc_head_p(m, m->s_x4, m->s_xc, m->mtp_hc_fn, m->mtp_hc_base, m->mtp_hc_scale);
    ds4f_rmsnorm(m->s_hn, m->s_xc, m->mtp_norm, C, eps);
    if (m->head.rows < c->vocab && m->ar_cb) {
        memset(m->s_logits, 0, (size_t)c->vocab*4);
        ds4f_matvec(m, m->s_logits + m->head_r0, &m->head, m->s_hn);
        m->ar_cb(m->s_logits, c->vocab, m->ar_ctx);
    } else ds4f_matvec(m, m->s_logits, &m->head, m->s_hn);
    int best = 0; { float bv = m->s_logits[0];
        for (int v = 1; v < c->vocab; v++) if (m->s_logits[v] > bv) { bv = m->s_logits[v]; best = v; } }
    if (logits_out) memcpy(logits_out, m->s_logits, (size_t)c->vocab*4);
    m->attn_h0=s_h0; m->attn_h1=s_h1; m->oi0=s_oi0; m->oi_rows=s_oir; m->sh_r0=s_shr0; m->sh_rows=s_shr;
    return best;
}
/* Finish the shared expert after its two up/gate projections.  The helper is
 * also used after an asynchronous GPU launch has overlapped the CPU router
 * and routed experts. */
static void ds4f_shared_finish(ds4f_model *m, ds4f_layer *ly, int L, int tps2) {
    ds4f_config *c = &m->cfg;
    int C = c->hidden;
    if (m->exact) { float lim = c->swiglu_limit;
        for (int i = 0; i < c->shared_inter; i++)
            m->s_shg[i] = ds4f_silu(m->s_shg[i] > lim ? lim : m->s_shg[i]) *
                          ds4f_clampf(m->s_shu[i], -lim, lim);
    } else {
        for (int i = 0; i < c->shared_inter; i++)
            m->s_shg[i] = ds4f_silu(m->s_shg[i]) * m->s_shu[i];
    }
    ds4f_chk("sh_gu", L, m->s_shg, c->shared_inter);
    if (tps2) {
        memset(m->s_o, 0, (size_t)C * sizeof(float));
        ds4f_matvec(m, m->s_o + m->sh2_r0, &ly->sh_w2, m->s_shg);
    } else {
        ds4f_matvec(m, m->s_o, &ly->sh_w2, m->s_shg);
    }
    ds4f_chk("sh_out", L, m->s_o, C);
    if (tps2) for (int i = m->sh2_r0; i < m->sh2_r0 + m->sh2_rows; i++) m->s_moe[i] += m->s_o[i];
    else      for (int i = 0; i < C; i++) m->s_moe[i] += m->s_o[i];
}

static int ds4f_forward_token(ds4f_model *m, float *x, int pos) {
    ds4f_config *c = &m->cfg;
    int C = c->hidden, HD = c->q_head_dim, KV = c->kv_lora, H = c->n_heads*HD;
    float eps = 1e-6f;
    if (ds4f_prof_on < 0) { const char *e = getenv("DS4F_PROF"); ds4f_prof_on = e ? atoi(e) : 0; }
    if (ds4f_attn_sve < 0) { const char *e = getenv("DS4F_ATTN_SVE"); ds4f_attn_sve = e ? atoi(e) : 1; }
    if (ds4f_oproj_fuse < 0) { const char *e = getenv("DS4F_OPROJ_FUSE"); ds4f_oproj_fuse = e ? atoi(e) : 1; }
    if (ds4f_qnr_par < 0) { const char *e = getenv("DS4F_QNR_PAR"); ds4f_qnr_par = e ? atoi(e) : 1; }
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
            asrc = m->s_xc;
            ds4f_chk("hc_pre_a", L, asrc, C);
            DS4F_TOC(DS4F_P_MHCPRE);
            { DS4F_TIC(); memcpy(m->s_resid, m->s_x4, hcC*4); DS4F_TOC(DS4F_P_MHCCPY); }
        }
        /* ---- MLA: q/kv projections ---- */
        { DS4F_TIC();
        ds4f_rmsnorm(m->s_hn, asrc, ly->attn_norm, C, eps);
        ds4f_chk("attn_norm", L, m->s_hn, C);
        if (ds4f_mv_fuse_on()) {   /* wq_a + wkv both read s_hn -> ONE dispatch (wq_b depends on wq_a, stays separate). s_kvlat is independent of the q-path -> computing it early is bit-exact. */
            DS4F_TIC();
            ds4f_mv1 qkv[2] = { { m->s_qlat, &ly->wq_a, m->s_hn }, { m->s_kvlat, &ly->wkv, m->s_hn } };
            ds4f_matvec_multi(m, qkv, 2);
            DS4F_TOC(DS4F_P_QKV_A);
        } else { DS4F_TIC(); ds4f_matvec(m, m->s_qlat, &ly->wq_a, m->s_hn); DS4F_TOC(DS4F_P_QKV_A); }
        ds4f_rmsnorm(m->s_qlat, m->s_qlat, ly->q_norm, c->q_lora, eps);
        { DS4F_TIC(); ds4f_matvec(m, m->s_q + (size_t)m->attn_h0 * c->q_head_dim, &ly->wq_b, m->s_qlat); DS4F_TOC(DS4F_P_QKV_B); } /* TP: owned heads of [n_heads*q_head_dim] */
        if (m->exact) { DS4F_TIC();
            if (ds4f_qnr_par) ds4f_q_norm_rope_par(m, m->s_q, pos, rcos, rsin);
            else              ds4f_q_norm_rope(m, m->s_q, pos, rcos, rsin);
            DS4F_TOC(DS4F_P_QKV_ROPE); } /* per-head norm + RoPE */
        ds4f_chk("q", L, m->s_q, H);
        if (!ds4f_mv_fuse_on()) { DS4F_TIC(); ds4f_matvec(m, m->s_kvlat, &ly->wkv, m->s_hn); DS4F_TOC(DS4F_P_QKV_KV); }  /* [kv_lora]; fused with wq_a above when MV_FUSE */
        ds4f_rmsnorm(m->s_kvlat, m->s_kvlat, ly->kv_norm, KV, eps);
        if (m->exact)                                            /* RoPE the kv rope dims */
            ds4f_rope_apply(m->s_kvlat + (KV - c->qk_rope_dim), rcos, rsin, pos, c->qk_rope_dim/2, 0);
        ds4f_chk("kvlat", L, m->s_kvlat, KV);
        if (m->int8_kv) ds4f_kv_append_i8(ly, m->s_kvlat, pos, KV, ds4f_int8kv_cal);  /* int8 KV (S5) */
        else { uint16_t *dst = ly->kv_cache + (size_t)(pos % ly->kv_slots)*KV;  /* append latent (f32->bf16); ring for sparse layers */
          for (int d = 0; d < KV; d++) dst[d] = ds4f_f32bf(m->s_kvlat[d]); }
        DS4F_TOC(DS4F_P_QKV); }
        /* ---- Tier-B2 compressor/indexer step (timed apart from the attn worker) ---- */
        if (m->tierb2 && ratio) { DS4F_TIC();
            ds4f_tb2_prepare(m, ly, ratio, pos, rcos, rsin);  /* fills s_tb2_sel/nsel */
            DS4F_TOC(DS4F_P_TB2PREP); }
        /* ---- CP gather-selected (CSA only): dequant the selected cmp_q4 latents to f32 and ar_cb-SUM
         * so every node holds the full selected set even though cmp_q4 is slot-sharded; attention reads
         * s_cmp_gather instead of cmp_q4[sel]. Step A (cmp still replicated): only rank 0 contributes. */
        m->cp_gather = 0;
        if (m->cp && m->tierb2 && ratio == 4 && m->int4_cmp && ly->cmp_frozen && m->ar_cb) {
            int ns = m->s_tb2_nsel, CAL = ds4f_int8cmp_cal;
            memset(m->s_cmp_gather, 0, (size_t)ns*KV*4);
            for (int j = 0; j < ns; j++) {        /* each node dequants the selected slots IT owns; ar_cb-SUM -> all */
                int g = m->s_tb2_sel[j], loc;
                if (!ly->cp_on)        loc = (m->ep_rank == 0) ? g : -1;                      /* Step A: full cmp on rank 0 */
                else if (g < CAL)      loc = (m->ep_rank == 0) ? g : -1;                      /* replicated [0,CAL): rank 0 */
                else                   loc = (g >= ly->cp_t0 && g < ly->cp_t1) ? CAL + (g - ly->cp_t0) : -1;  /* sharded tail: owner */
                if (loc >= ly->cp_nslot) { fprintf(stderr, "[CP-DBG gather OOB] j=%d g=%d loc=%d cap=%d t0=%d t1=%d ns=%d cap_g=%d\n",
                    j, g, loc, ly->cp_nslot, ly->cp_t0, ly->cp_t1, ns, m->cfg.index_topk); fflush(stderr); abort(); }
                if (loc >= 0) ds4f_cmp_deq_row_i4(ly->cmp_q4 + (size_t)loc*(KV/2), ly->cmp_scale, KV,
                                                  m->s_cmp_gather + (size_t)j*KV);
            }
            m->ar_cb(m->s_cmp_gather, ns*KV, m->ar_ctx);
            m->cp_gather = 1;
        }
        /* ---- attention ---- */
        { DS4F_TIC();
        if (m->attn_h1 - m->attn_h0 < c->n_heads) memset(m->s_attn, 0, (size_t)H * sizeof(float));  /* TP: only owned heads filled -> partial s_attn */
        if (m->tierb2 && ratio) {
            /* window + indexer-selected compressed term (prepare ran above). */
            ds4f_attn_ex_task at = { m, ly, pos, 1.0f/sqrtf((float)HD),
                                     c->window_size, c->qk_rope_dim/2, rcos, rsin };
        if (!ds4f_attn_tb2_gemm(m, &at)) ds4f_pool_run(m->pool, ds4f_attn_tb2_worker, &at);
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
            int gin = H / og;                          /* 32768/8 = 4096 == C */
            int tpo = (m->oi_rows < c->o_inter);       /* TP: wo_a o_inter row-shard */
            /* TP_ATTN leaves s_attn head-partial; the o_inter shard needs the FULL s_attn (each
             * owned o_inter row contracts a whole group), so sum it first (disjoint heads + zeros
             * => exact reconstruction). */
            if (tpo && (m->attn_h1 - m->attn_h0 < c->n_heads) && m->ar_cb) m->ar_cb(m->s_attn, H, m->ar_ctx);
            if (tpo) memset(m->s_o1, 0, (size_t)c->o_inter * sizeof(float));  /* only [oi0,oi0+oi_rows) computed */
            /* block-diagonal wo_a (group g: o1[g*o_lora..] = wo_a[g*o_lora..] @ s_attn[g*gin..]);
             * goff carries this node's o_inter offset so sharded rows pick the right group. TP / Q8_PV
             * force the fused path (ds4f_row_slice can't slice the o_inter shard or the 528B Q8 layout). */
            if (ds4f_oproj_fuse || tpo || ly->wo_a.type == DS4F_Q8_PV) {
                ds4f_matvec_blockdiag(m, m->s_o1 + m->oi0, &ly->wo_a, m->s_attn, gin, c->o_lora, m->oi0);
            } else for (int g = 0; g < og; g++) {      /* reference: og separate dispatches */
                ds4f_tensor vg = ds4f_row_slice(&ly->wo_a, g * c->o_lora, c->o_lora);
                ds4f_matvec(m, m->s_o1 + (size_t)g * c->o_lora, &vg, m->s_attn + (size_t)g * gin);
            }
            ds4f_matvec(m, m->s_o, &ly->wo_b, m->s_o1 + (ly->wo_b.cols < c->o_inter ? m->oi0 : 0));  /* col-shard: owned o_inter slice. [hidden], partial under TP */
        } else {
            for (int i = 0; i < C; i++) m->s_oin[i] = 0.f;
            for (int g = 0; g < og; g++) for (int i = 0; i < C; i++) m->s_oin[i] += m->s_attn[g*C + i];
            ds4f_matvec(m, m->s_o1, &ly->wo_a, m->s_oin);       /* [o_inter] */
            for (int i = 0; i < c->o_inter; i++) m->s_o1[i] = ds4f_silu(m->s_o1[i]);  /* stand-in nonlin */
            ds4f_matvec(m, m->s_o, &ly->wo_b, m->s_o1 + (ly->wo_b.cols < c->o_inter ? m->oi0 : 0));  /* col-shard: owned o_inter slice. [hidden] */
        }
        if (((m->attn_h1 - m->attn_h0 < c->n_heads) || (m->oi_rows < c->o_inter)) && m->ar_cb)  /* TP attn/oproj: sum partial s_o -> full hidden */
            m->ar_cb(m->s_o, C, m->ar_ctx);
        ds4f_chk("o", L, m->s_o, C);
        if (m->mhc) { DS4F_TIC();
            ds4f_hc_post(m, m->s_x4, m->s_resid, m->s_o, post_a, comb_a); /* expand 1->4 */
            DS4F_TOC(DS4F_P_MHCPOST);
        }
        else for (int i = 0; i < C; i++) x[i] += m->s_o[i];     /* plain-residual stand-in */
        ds4f_chk("x+attn", L, m->mhc ? m->s_x4 : x, C);
        DS4F_TOC(DS4F_P_OPROJ); }

        /* ---- mHC pre (ffn): collapse 4 streams -> ffn input; save residual ---- */
        float *fsrc = x;
        if (m->mhc) { DS4F_TIC();
            ds4f_hc_pre(m, m->s_x4, ly->hc_ffn_fn, ly->hc_ffn_scale, ly->hc_ffn_base,
                        m->s_xc, post_f, comb_f);
            fsrc = m->s_xc;
            DS4F_TOC(DS4F_P_MHCPRE);
            { DS4F_TIC(); memcpy(m->s_resid, m->s_x4, hcC*4); DS4F_TOC(DS4F_P_MHCCPY); }
        }
        /* ---- MoE: shared expert ---- */
        ds4f_rmsnorm(m->s_h2, fsrc, ly->ffn_norm, C, eps);
        ds4f_chk("ffn_norm", L, m->s_h2, C);
        for (int i = 0; i < C; i++) { m->s_moe[i] = 0.f; m->s_route[i] = 0.f; }
        int tps = (m->sh_rows < c->shared_inter);              /* TP shared-expert (sh_w1/w3 col-shard) */
        int tps2 = tps && (m->sh2_rows < C);                  /* optional sh_w2 hidden-row shard */
        int shared_gpu_pending = 0;
        { DS4F_TIC();
        if (tps) { memset(m->s_shg, 0, (size_t)c->shared_inter*4); memset(m->s_shu, 0, (size_t)c->shared_inter*4); }
        const ds4f_tensor *sh_t[2] = { &ly->sh_w1, &ly->sh_w3 };
        float *sh_y[2] = { m->s_shg + m->sh_r0, m->s_shu + m->sh_r0 };
        const float *sh_x[2] = { m->s_h2, m->s_h2 };
        int sh_gpu = m->gpu_dense_async_multi && m->gpu_dense_wait &&
                     ly->sh_w1.type == DS4F_FP8 && ly->sh_w3.type == DS4F_FP8 &&
                     ly->sh_w1.gpu_id >= 0 && ly->sh_w3.gpu_id >= 0 &&
                     ly->gate.gpu_id < 0; /* a GPU router needs the same bank stream */
        if (sh_gpu) {
            if (m->gpu_dense_async_multi(m->gpu_dense_ctx, sh_y, sh_t, sh_x, 2) != 0) {
                fprintf(stderr, "ds4f: asynchronous shared-expert launch failed\n");
                abort();
            }
            shared_gpu_pending = 1;
        } else if (ds4f_mv_fuse_on()) {   /* sh_w1 + sh_w3 both read s_h2 -> ONE dispatch */
            ds4f_mv1 sh[2] = { { m->s_shg + m->sh_r0, &ly->sh_w1, m->s_h2 },
                               { m->s_shu + m->sh_r0, &ly->sh_w3, m->s_h2 } };
            ds4f_matvec_multi(m, sh, 2);
        } else {
            ds4f_matvec(m, m->s_shg + m->sh_r0, &ly->sh_w1, m->s_h2);  /* col-shard up/gate -> [sh_r0, sh_r0+sh_rows) */
            ds4f_matvec(m, m->s_shu + m->sh_r0, &ly->sh_w3, m->s_h2);
        }
        if (!shared_gpu_pending) ds4f_shared_finish(m, ly, L, tps2);
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
        ds4f_route_observe(m, L, idx, c->n_active);
        if (ds4f_dbg) { fprintf(stderr, "  L%-2d topk wt=", L);
            for (int k=0;k<c->n_active;k++) fprintf(stderr,"%.3f(e%d) ", wt[k], idx[k]);
            fprintf(stderr,"\n"); }
        /* The shared expert normally remains in flight while all routed work
         * runs on the CPU. A resident routed expert uses the same HIP adapter,
         * so retire shared first only when this token has an actual cache hit. */
        int routed_gpu_hit = 0;
        for (int k = 0; k < c->n_active; ++k) {
            int e = idx[k];
            if (e < 0 || e % m->ep_size != m->ep_rank) continue;
            int slot = e / m->ep_size;
            if (ly->ex_w1[slot].gpu_id >= 0 && ly->ex_w3[slot].gpu_id >= 0 &&
                ly->ex_w2[slot].gpu_id >= 0) { routed_gpu_hit = 1; break; }
        }
        if (shared_gpu_pending && routed_gpu_hit) {
            DS4F_TIC();
            if (m->gpu_dense_wait(m->gpu_dense_ctx) != 0) {
                fprintf(stderr, "ds4f: asynchronous shared-expert wait failed\n");
                abort();
            }
            ds4f_shared_finish(m, ly, L, tps2);
            shared_gpu_pending = 0;
            DS4F_TOC(DS4F_P_SHARED);
        }
        /* ---- MoE: routed experts (owned-only) ---- */
        { DS4F_TIC();
        if (ds4f_expert_batch_on()) {
            /* Preserve top-k order while compacting the experts owned by this
             * rank.  EP ranks can have fewer than n_active local experts. */
            int local_k[8], nlocal = 0;
            for (int k = 0; k < c->n_active; k++) {
                int e = idx[k]; if (e < 0) continue;
                if (e % m->ep_size != m->ep_rank) continue;
                local_k[nlocal++] = k;
            }
            if (nlocal > 0) {
                ds4f_mv1 gateup_gpu[16], gateup_cpu[16];
                int ngpu = 0, ncpu = 0;
                for (int j = 0; j < nlocal; j++) {
                    int k = local_k[j], slot = idx[k] / m->ep_size;
                    ds4f_mv1 a = { m->s_exb_g + (size_t)j * c->moe_inter,
                                    &ly->ex_w1[slot], m->s_h2 };
                    ds4f_mv1 b = { m->s_exb_u + (size_t)j * c->moe_inter,
                                    &ly->ex_w3[slot], m->s_h2 };
                    if (a.t->gpu_id >= 0 && b.t->gpu_id >= 0) {
                        gateup_gpu[ngpu++] = a; gateup_gpu[ngpu++] = b;
                    } else {
                        gateup_cpu[ncpu++] = a; gateup_cpu[ncpu++] = b;
                    }
                }
                /* Cache misses stay on the exact CPU path. Cache hits use the
                 * compact raw-MXFP4 HIP matrices and never trigger H2D here. */
                if (ncpu) ds4f_matvec_multi(m, gateup_cpu, ncpu);
                if (ngpu) ds4f_matvec_multi(m, gateup_gpu, ngpu);
                for (int j = 0; j < nlocal; j++) {
                    float *g = m->s_exb_g + (size_t)j * c->moe_inter;
                    const float *u = m->s_exb_u + (size_t)j * c->moe_inter;
                    if (m->exact) { float lim = c->swiglu_limit;
                        for (int i = 0; i < c->moe_inter; i++)
                            g[i] = ds4f_silu(g[i] > lim ? lim : g[i]) * ds4f_clampf(u[i], -lim, lim);
                    } else
                        for (int i = 0; i < c->moe_inter; i++) g[i] = ds4f_silu(g[i]) * u[i];
                }
                ds4f_mv1 down_gpu[8], down_cpu[8];
                ngpu = ncpu = 0;
                for (int j = 0; j < nlocal; j++) {
                    int k = local_k[j], slot = idx[k] / m->ep_size;
                    ds4f_mv1 a = { m->s_exb_o + (size_t)j * C,
                                    &ly->ex_w2[slot],
                                    m->s_exb_g + (size_t)j * c->moe_inter };
                    if (a.t->gpu_id >= 0) down_gpu[ngpu++] = a;
                    else down_cpu[ncpu++] = a;
                }
                if (ncpu) ds4f_matvec_multi(m, down_cpu, ncpu);
                if (ngpu) ds4f_matvec_multi(m, down_gpu, ngpu);
                for (int j = 0; j < nlocal; j++) {
                    const float *y = m->s_exb_o + (size_t)j * C;
                    float w = wt[local_k[j]];
                    for (int i = 0; i < C; i++) m->s_route[i] += w * y[i];
                }
            }
        } else for (int k = 0; k < c->n_active; k++) {
            int e = idx[k]; if (e < 0) continue;
            if (e % m->ep_size != m->ep_rank) continue;          /* owned-only (Stage 1/local) */
            int slot = e / m->ep_size;                           /* dense owned index */
            if (ds4f_mv_fuse_on()) {   /* ex_w1 + ex_w3 both read s_h2 -> ONE dispatch (per expert) */
                ds4f_mv1 ex[2] = { { m->s_exg, &ly->ex_w1[slot], m->s_h2 },
                                   { m->s_exu, &ly->ex_w3[slot], m->s_h2 } };
                ds4f_matvec_multi(m, ex, 2);
            } else {
                ds4f_matvec(m, m->s_exg, &ly->ex_w1[slot], m->s_h2);
                ds4f_matvec(m, m->s_exu, &ly->ex_w3[slot], m->s_h2);
            }
            if (m->exact) { float lim = c->swiglu_limit;
                for (int i = 0; i < c->moe_inter; i++)
                    m->s_exg[i] = ds4f_silu(m->s_exg[i] > lim ? lim : m->s_exg[i]) * ds4f_clampf(m->s_exu[i], -lim, lim);
            } else
                for (int i = 0; i < c->moe_inter; i++) m->s_exg[i] = ds4f_silu(m->s_exg[i]) * m->s_exu[i];
            ds4f_matvec(m, m->s_o, &ly->ex_w2[slot], m->s_exg);
            for (int i = 0; i < C; i++) m->s_route[i] += wt[k] * m->s_o[i];
        }
        DS4F_TOC(DS4F_P_EXPERTS); }
        if (shared_gpu_pending) {
            DS4F_TIC();
            if (m->gpu_dense_wait(m->gpu_dense_ctx) != 0) {
                fprintf(stderr, "ds4f: asynchronous shared-expert wait failed\n");
                abort();
            }
            ds4f_shared_finish(m, ly, L, tps2);
            DS4F_TOC(DS4F_P_SHARED);
        }
        /* EP combine: sum the routed-expert partial across the expert-parallel
         * group (each rank owns a disjoint expert subset). Shared expert is
         * replicated, so it is NOT reduced — added locally below. */
        if (tps) for (int i = 0; i < C; i++) m->s_route[i] += m->s_moe[i];  /* fold partial shared into routed -> ONE reduce */
        if (m->ar_cb) { DS4F_TIC(); m->ar_cb(m->s_route, C, m->ar_ctx); DS4F_TOC(DS4F_P_COMM); }
        ds4f_chk("moe", L, m->s_route, C);
        if (m->mhc) {
            for (int i = 0; i < C; i++) m->s_o[i] = (tps ? 0.f : m->s_moe[i]) + m->s_route[i]; /* tps: shared already in s_route */
            { DS4F_TIC();
              ds4f_hc_post(m, m->s_x4, m->s_resid, m->s_o, post_f, comb_f);         /* expand 1->4 */
              DS4F_TOC(DS4F_P_MHCPOST); }
        } else for (int i = 0; i < C; i++) x[i] += (tps ? 0.f : m->s_moe[i]) + m->s_route[i];  /* shared(local|folded)+routed */
        ds4f_chk("x+moe", L, m->mhc ? m->s_x4 : x, C);
    }
    /* head: mHC collapse 4 streams -> 1 (no sinkhorn), then out_norm + lm_head */
    float *hsrc = x;
    if (m->mhc) { ds4f_hc_head(m, m->s_x4, m->s_xc); hsrc = m->s_xc; ds4f_chk("hc_head", -1, hsrc, C); }
    { DS4F_TIC();
    ds4f_rmsnorm(m->s_hn, hsrc, m->out_norm, C, eps);
    if (m->head.rows < c->vocab && m->ar_cb) {                 /* TP head: shard matvec -> zero-fill -> all-reduce-SUM */
        /* each node computes only its vocab shard at s_logits[head_r0..]; zero the rest, then
         * sum across the TP group. Shards are disjoint so the sum reconstructs the FULL logits
         * BIT-EXACTLY (adding zeros is exact; each row's dot is identical to the replicated head)
         * -> every node has identical full logits -> identical argmax (lockstep), no new collective. */
        memset(m->s_logits, 0, (size_t)c->vocab * sizeof(float));
        ds4f_matvec(m, m->s_logits + m->head_r0, &m->head, m->s_hn);
        m->ar_cb(m->s_logits, c->vocab, m->ar_ctx);
    } else {
        ds4f_matvec(m, m->s_logits, &m->head, m->s_hn);        /* replicated: full head */
    }
    int best = 0; float bv = m->s_logits[0];
    for (int v = 1; v < c->vocab; v++) if (m->s_logits[v] > bv) { bv = m->s_logits[v]; best = v; }
    DS4F_TOC(DS4F_P_HEAD);
    return best; }
}
