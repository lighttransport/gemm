#define _GNU_SOURCE
#include "q38fn_ngram_pipeline.h"
#include <errno.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

enum { SLOT_FREE, SLOT_BUILDING, SLOT_READY, SLOT_RUNNING, SLOT_DONE, SLOT_FAILED,
       SLOT_CONSUMING };
typedef struct {
    q38fn_ngram_request *req, *uniq, *order;
    uint16_t *row_map;
    uint64_t *row_tag;
    uint16_t *data, *scratch;
    uint32_t count, nuniq, state;
    uint64_t generation;
    int error;
} batch_slot;
typedef struct {
    uint64_t tag;
    uint16_t data[Q38FN_NGRAM_HEAD_DIM];
    uint8_t valid;
} cache_entry;
#define CACHE_WAYS 4
#define CACHE_LOCKS 64
#define STAT_ADD(p, field, value) \
    __atomic_fetch_add(&(p)->stats.field, (uint64_t)(value), __ATOMIC_RELAXED)

struct q38fn_ngram_pipeline {
    q38fn_ngram_source *src;
    uint32_t nsrc, depth, workers, started, max_span, max_gap, max_count;
    batch_slot *slots;
    uint32_t *ready_queue;
    uint64_t ready_head, ready_tail;
    pthread_t *threads;
    struct worker_arg *worker_args;
    pthread_mutex_t mu;
    pthread_cond_t ready, free_slot, done;
    _Atomic int stop;
    q38fn_ngram_stats stats;
    cache_entry *cache;
    uint32_t cache_rows;
    uint32_t cache_sets;
    uint32_t cache_shift;
    int cache_power2;
    _Atomic uint32_t local_owner;
    _Atomic int cache_remote_only;
    pthread_mutex_t *cache_mu;
};

struct worker_arg {
    q38fn_ngram_pipeline *p;
    uint32_t index;
};

static void pin_worker(uint32_t index)
{
    const char *enabled = getenv("Q38FN_PIPELINE_PIN_WORKERS");
    if (!enabled || *enabled == '0') return;
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return;
    int cpus[CPU_SETSIZE];
    uint32_t n = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE && n < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &allowed)) cpus[n++] = cpu;
    if (!n) return;
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(cpus[index % n], &one);
    (void)sched_setaffinity(0, sizeof one, &one);
}

static uint64_t mono_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static void *aligned_alloc_bytes(size_t bytes) {
    void *p = NULL;
    return posix_memalign(&p, 256, bytes) == 0 ? p : NULL;
}
static inline void copy_rows(void *restrict dst, const void *restrict src,
                             uint32_t rows) {
#if defined(__ARM_FEATURE_SVE)
    uint16_t *d = (uint16_t *)dst;
    const uint16_t *s = (const uint16_t *)src;
    uint64_t count = (uint64_t)rows * Q38FN_NGRAM_HEAD_DIM;
    uint64_t vl = svcnth();
    /* A64FX has 512-bit SVE, so a row is exactly five full vectors.  Keep
     * this path row-oriented: it avoids constructing a predicate for every
     * vector and preserves the row boundary for the hardware prefetcher.
     * The alignment check matters because this helper is also used for the
     * caller-owned output buffer. */
    if (vl * 5 == Q38FN_NGRAM_HEAD_DIM &&
        ((((uintptr_t)d | (uintptr_t)s) & 63u) == 0)) {
        const svbool_t all = svptrue_b16();
        for (uint32_t row = 0; row < rows; ++row) {
            const size_t base = (size_t)row * Q38FN_NGRAM_HEAD_DIM;
            for (size_t v = 0; v < 5; ++v) {
                const size_t offset = base + v * vl;
                svst1_u16(all, d + offset, svld1_u16(all, s + offset));
            }
        }
        return;
    }
    for (uint64_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b16(i, count);
        svst1_u16(pg, d + i, svld1_u16(pg, s + i));
    }
#else
    memcpy(dst, src, (size_t)rows * Q38FN_NGRAM_ROW_BYTES);
#endif
}
/* The request count is at most 256.  A fixed eight-pass byte radix sort
 * avoids qsort's indirect comparator calls and 64-bit comparison branches in
 * the token-submit hot path.  tmp is a slot-owned buffer whose input contents
 * are dead after deduplication.  Rows are unique here, so row order alone is
 * a complete ordering key. */
static void sort_requests(q38fn_ngram_request *a,
                          q38fn_ngram_request *tmp, uint32_t n)
{
    if (n < 2) return;
    if (n <= 32) {
        /* One token contributes 16 heads.  For this small case, insertion
         * sort avoids eight counter-array passes and has excellent locality. */
        for (uint32_t i = 1; i < n; ++i) {
            q38fn_ngram_request key = a[i];
            uint32_t j = i;
            while (j > 0 && a[j - 1].row > key.row) {
                a[j] = a[j - 1];
                --j;
            }
            a[j] = key;
        }
        return;
    }
    q38fn_ngram_request *src = a, *dst = tmp;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        uint32_t count[256] = {0};
        for (uint32_t i = 0; i < n; ++i)
            ++count[(unsigned)((src[i].row >> shift) & UINT64_C(255))];
        uint32_t offset = 0;
        for (unsigned b = 0; b < 256; ++b) {
            uint32_t c = count[b];
            count[b] = offset;
            offset += c;
        }
        for (uint32_t i = 0; i < n; ++i) {
            unsigned b = (unsigned)((src[i].row >> shift) & UINT64_C(255));
            dst[count[b]++] = src[i];
        }
        q38fn_ngram_request *swap = src;
        src = dst;
        dst = swap;
    }
    if (src != a)
        memcpy(a, src, (size_t)n * sizeof(*a));
}
static int find_free(q38fn_ngram_pipeline *p) {
    for (uint32_t i = 0; i < p->depth; ++i)
        if (p->slots[i].state == SLOT_FREE) return (int)i;
    return -1;
}
static int cache_eligible(q38fn_ngram_pipeline *p, uint64_t row) {
    uint32_t owner = p->src[row / Q38FN_NGRAM_ROWS_PER_SHARD].owner;
    return p->cache_rows && (!atomic_load_explicit(&p->cache_remote_only,
                                                   memory_order_relaxed) ||
        owner != atomic_load_explicit(&p->local_owner, memory_order_relaxed));
}
static inline pthread_mutex_t *cache_lock_for(q38fn_ngram_pipeline *p,
                                               uint32_t set) {
    return &p->cache_mu[set % CACHE_LOCKS];
}
static int cache_get_locked(q38fn_ngram_pipeline *p, uint64_t row, uint16_t *dst) {
    if (!cache_eligible(p, row)) return 0;
    uint32_t set = p->cache_power2 ? (uint32_t)row & (p->cache_sets - 1) :
                                     (uint32_t)(row % p->cache_sets);
    pthread_mutex_lock(cache_lock_for(p, set));
    __builtin_prefetch(&p->cache[set * CACHE_WAYS], 0, 1);
    int hit = 0;
    for (uint32_t w = 0; w < CACHE_WAYS; ++w) {
        cache_entry *e = &p->cache[set * CACHE_WAYS + w];
        if (e->valid && e->tag == row) {
            memcpy(dst, e->data, Q38FN_NGRAM_ROW_BYTES); hit = 1; break;
        }
    }
    pthread_mutex_unlock(cache_lock_for(p, set));
    return hit;
}
static void cache_put_locked(q38fn_ngram_pipeline *p, uint64_t row, const uint16_t *src) {
    if (!cache_eligible(p, row)) return;
    uint32_t set = p->cache_power2 ? (uint32_t)row & (p->cache_sets - 1) :
                                     (uint32_t)(row % p->cache_sets);
    pthread_mutex_lock(cache_lock_for(p, set));
    uint32_t way = p->cache_power2 ? (uint32_t)(row >> p->cache_shift) & (CACHE_WAYS - 1) :
                                     (uint32_t)((row / p->cache_sets) % CACHE_WAYS);
    cache_entry *e = &p->cache[set * CACHE_WAYS + way];
    memcpy(e->data, src, Q38FN_NGRAM_ROW_BYTES); e->tag = row; e->valid = 1;
    pthread_mutex_unlock(cache_lock_for(p, set));
}
static void *worker_main(void *arg) {
    struct worker_arg *wa = (struct worker_arg *)arg;
    q38fn_ngram_pipeline *p = wa->p;
    pin_worker(wa->index);
    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (!p->stop && p->ready_head == p->ready_tail)
            pthread_cond_wait(&p->ready, &p->mu);
        if (p->stop) { pthread_mutex_unlock(&p->mu); return NULL; }
        int si = (int)p->ready_queue[p->ready_head++ % p->depth];
        batch_slot *s = &p->slots[si]; s->state = SLOT_RUNNING;
        pthread_mutex_unlock(&p->mu);

        int err = 0; uint32_t i = 0, nmiss = 0;
        for (uint32_t j = 0; j < s->nuniq; ++j) {
            q38fn_ngram_request *r = &s->order[j];
            if (cache_get_locked(p, r->row, s->data + (size_t)r->unique * Q38FN_NGRAM_HEAD_DIM)) {
                STAT_ADD(p, cache_hits, 1);
            } else {
                s->order[nmiss++] = *r;
                STAT_ADD(p, cache_misses, 1);
            }
        }
        while (i < nmiss) {
            q38fn_ngram_request *a = &s->order[i];
            uint32_t split = (uint32_t)(a->row / Q38FN_NGRAM_ROWS_PER_SHARD);
            uint64_t local = a->row % Q38FN_NGRAM_ROWS_PER_SHARD;
            uint32_t n = 1;
            while (i + n < nmiss && n < p->max_span) {
                q38fn_ngram_request *b = &s->order[i + n];
                uint32_t bs = (uint32_t)(b->row / Q38FN_NGRAM_ROWS_PER_SHARD);
                uint64_t bl = b->row % Q38FN_NGRAM_ROWS_PER_SHARD;
                if (bs != split || bl < local || bl - local > (uint64_t)n + p->max_gap) break;
                ++n;
            }
            if (split >= p->nsrc || !p->src[split].read_span) { err = EINVAL; break; }
            uint64_t span_first = local;
            uint64_t span_last = s->order[i + n - 1].row % Q38FN_NGRAM_ROWS_PER_SHARD;
            uint32_t span_rows = (uint32_t)(span_last - span_first + 1);
            if (span_rows > p->max_span + p->max_gap) { err = E2BIG; break; }
            /* With no cache hits, sorted unique IDs are identical to the
             * worker's span order.  A contiguous span can therefore land
             * directly in the result slab, avoiding a second full-row copy.
             * Gapped/partially-cached spans retain the scratch/reorder path. */
            int direct = nmiss == s->nuniq && span_rows == n;
            for (uint32_t j = 0; direct && j < n; ++j)
                if (s->order[i + j].unique != i + j) direct = 0;
            STAT_ADD(p, direct_spans, direct);
            STAT_ADD(p, reorder_spans, !direct);
            uint16_t *span_dst = direct ? s->data + (size_t)i * Q38FN_NGRAM_HEAD_DIM
                                        : s->scratch;
            if (p->src[split].read_span(p->src[split].opaque, span_first,
                                        span_rows, span_dst) != 0) { err = EIO; break; }
            for (uint32_t j = 0; j < n; ++j) {
                uint32_t u = s->order[i + j].unique;
                uint64_t row = s->order[i + j].row % Q38FN_NGRAM_ROWS_PER_SHARD;
                if (!direct)
                    memcpy(s->data + (size_t)u * Q38FN_NGRAM_HEAD_DIM,
                           s->scratch + (size_t)(row - span_first) * Q38FN_NGRAM_HEAD_DIM,
                           Q38FN_NGRAM_ROW_BYTES);
                cache_put_locked(p, s->order[i + j].row,
                                 s->data + (size_t)u * Q38FN_NGRAM_HEAD_DIM);
            }
            STAT_ADD(p, spans, 1);
            STAT_ADD(p, physical_bytes, (uint64_t)span_rows * Q38FN_NGRAM_ROW_BYTES);
            i += n;
        }
        pthread_mutex_lock(&p->mu);
        s->error = err; s->state = err ? SLOT_FAILED : SLOT_DONE;
        STAT_ADD(p, completed, 1);
        pthread_cond_broadcast(&p->done);
        pthread_mutex_unlock(&p->mu);
    }
}

int q38fn_ngram_pipeline_init(q38fn_ngram_pipeline **out,
                              const q38fn_ngram_source *sources, uint32_t source_count,
                              uint32_t queue_depth, uint32_t workers,
                              uint32_t max_span_rows, uint32_t max_gap_rows) {
    return q38fn_ngram_pipeline_init_ex(out, sources, source_count, queue_depth,
                                        workers, max_span_rows, max_gap_rows, 0);
}
int q38fn_ngram_pipeline_init_ex(q38fn_ngram_pipeline **out,
                              const q38fn_ngram_source *sources, uint32_t source_count,
                              uint32_t queue_depth, uint32_t workers,
                              uint32_t max_span_rows, uint32_t max_gap_rows,
                              uint32_t cache_rows) {
    if (!out || !sources || source_count != Q38FN_NGRAM_SHARDS || !queue_depth || !workers ||
        !max_span_rows || max_span_rows > 256) return EINVAL;
    q38fn_ngram_pipeline *p = (q38fn_ngram_pipeline *)calloc(1, sizeof(*p));
    if (!p) return ENOMEM;
    pthread_mutex_init(&p->mu, NULL); pthread_cond_init(&p->ready, NULL);
    pthread_cond_init(&p->free_slot, NULL); pthread_cond_init(&p->done, NULL);
    p->src = (q38fn_ngram_source *)malloc(sizeof(*p->src) * source_count);
    p->slots = (batch_slot *)calloc(queue_depth, sizeof(*p->slots));
    p->ready_queue = (uint32_t *)malloc(sizeof(*p->ready_queue) * queue_depth);
    p->threads = (pthread_t *)calloc(workers, sizeof(*p->threads));
    p->worker_args = (struct worker_arg *)calloc(workers, sizeof(*p->worker_args));
    if (!p->src || !p->slots || !p->ready_queue || !p->threads || !p->worker_args) { q38fn_ngram_pipeline_destroy(p); return ENOMEM; }
    memcpy(p->src, sources, sizeof(*p->src) * source_count);
    p->nsrc=source_count; p->depth=queue_depth; p->workers=workers;
    p->max_span=max_span_rows; p->max_gap=max_gap_rows; p->max_count=Q38FN_NGRAM_MAX_BATCH;
    p->cache_rows = cache_rows;
    p->cache_sets = cache_rows ? (cache_rows + CACHE_WAYS - 1) / CACHE_WAYS : 0;
    p->cache_power2 = p->cache_sets && !(p->cache_sets & (p->cache_sets - 1));
    if (p->cache_power2)
        while ((UINT32_C(1) << p->cache_shift) < p->cache_sets) ++p->cache_shift;
    p->cache_mu = (pthread_mutex_t *)calloc(CACHE_LOCKS, sizeof(*p->cache_mu));
    if (!p->cache_mu) { q38fn_ngram_pipeline_destroy(p); return ENOMEM; }
    for (uint32_t i = 0; i < CACHE_LOCKS; ++i)
        pthread_mutex_init(&p->cache_mu[i], NULL);
    if (cache_rows) {
        p->cache = (cache_entry *)calloc((size_t)p->cache_sets * CACHE_WAYS,
                                          sizeof(*p->cache));
        if (!p->cache) { q38fn_ngram_pipeline_destroy(p); return ENOMEM; }
    }
    for (uint32_t i=0;i<queue_depth;++i) {
        batch_slot *s=&p->slots[i];
        s->req=aligned_alloc_bytes(sizeof(*s->req)*p->max_count);
        s->uniq=aligned_alloc_bytes(sizeof(*s->uniq)*p->max_count);
        s->order=aligned_alloc_bytes(sizeof(*s->order)*p->max_count);
        s->row_map=aligned_alloc_bytes(sizeof(*s->row_map)*p->max_count*2);
        s->row_tag=aligned_alloc_bytes(sizeof(*s->row_tag)*p->max_count*2);
        s->data=aligned_alloc_bytes((size_t)p->max_count*Q38FN_NGRAM_ROW_BYTES);
        s->scratch=aligned_alloc_bytes((size_t)(max_span_rows+max_gap_rows+1)*Q38FN_NGRAM_ROW_BYTES);
        if(!s->req||!s->uniq||!s->order||!s->row_map||!s->row_tag||!s->data||!s->scratch){q38fn_ngram_pipeline_destroy(p);return ENOMEM;}
        memset(s->row_tag, 0, sizeof(*s->row_tag) * p->max_count * 2);
        s->state=SLOT_FREE;
    }
    for (uint32_t i=0;i<workers;++i) {
        p->worker_args[i] = (struct worker_arg){p, i};
        if (pthread_create(&p->threads[i],NULL,worker_main,&p->worker_args[i]) != 0) {
        pthread_mutex_lock(&p->mu);
        atomic_store_explicit(&p->stop, 1, memory_order_release);
        pthread_cond_broadcast(&p->ready);
        pthread_mutex_unlock(&p->mu);
        q38fn_ngram_pipeline_destroy(p); return EAGAIN;
        } else {
        p->started++;
        }
    }
    *out=p; return 0;
}

void q38fn_ngram_pipeline_destroy(q38fn_ngram_pipeline *p) {
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    atomic_store_explicit(&p->stop, 1, memory_order_release);
    pthread_cond_broadcast(&p->ready);
    pthread_mutex_unlock(&p->mu);
    if(p->threads) for(uint32_t i=0;i<p->started;++i)pthread_join(p->threads[i],NULL);
    if(p->slots) for(uint32_t i=0;i<p->depth;++i){free(p->slots[i].req);free(p->slots[i].uniq);free(p->slots[i].order);free(p->slots[i].row_map);free(p->slots[i].row_tag);free(p->slots[i].data);free(p->slots[i].scratch);}
    pthread_cond_destroy(&p->ready);pthread_cond_destroy(&p->free_slot);pthread_cond_destroy(&p->done);pthread_mutex_destroy(&p->mu);
    if (p->cache_mu) {
        for (uint32_t i = 0; i < CACHE_LOCKS; ++i)
            pthread_mutex_destroy(&p->cache_mu[i]);
        free(p->cache_mu);
    }
    free(p->cache);
    free(p->worker_args);free(p->threads);free(p->ready_queue);free(p->slots);free(p->src);free(p);
}
void q38fn_ngram_set_cache_remote_only(q38fn_ngram_pipeline *p, uint32_t local_owner) {
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    atomic_store_explicit(&p->local_owner, local_owner, memory_order_relaxed);
    atomic_store_explicit(&p->cache_remote_only, 1, memory_order_release);
    pthread_mutex_unlock(&p->mu);
}

int q38fn_ngram_submit(q38fn_ngram_pipeline *p,const uint64_t *rows,uint32_t count,q38fn_ngram_ticket *ticket){
    if(!p||!rows||!ticket||!count||count>p->max_count)return EINVAL;
    for (uint32_t i = 0; i < count; ++i)
        if (rows[i] >= Q38FN_NGRAM_ROWS) return EINVAL;
    pthread_mutex_lock(&p->mu); int si; while((si=find_free(p))<0&&!p->stop)pthread_cond_wait(&p->free_slot,&p->mu);
    if(p->stop){pthread_mutex_unlock(&p->mu);return ECANCELED;} batch_slot*s=&p->slots[si];
    /* Reserve the slot, then release the global lock while doing the
     * producer-side hash-table build and sort.  Workers can now consume
     * earlier READY slots while a large token window is being prepared. */
    s->state=SLOT_BUILDING;s->count=count;s->nuniq=0;s->generation++;s->error=0;
    uint64_t generation = s->generation;
    pthread_mutex_unlock(&p->mu);
    for(uint32_t i=0;i<count;++i){
        s->req[i]=(q38fn_ngram_request){rows[i],(uint16_t)i,0};
        uint32_t pos=(uint32_t)((rows[i] * UINT64_C(11400714819323198485)) >> 55) &
                     (Q38FN_NGRAM_MAX_BATCH * 2 - 1);
        while (s->row_tag[pos] == generation &&
               s->uniq[s->row_map[pos]].row != rows[i])
            pos = (pos + 1) & (Q38FN_NGRAM_MAX_BATCH * 2 - 1);
        uint32_t u = s->row_tag[pos] == generation ? s->row_map[pos] : UINT32_MAX;
        if (u == UINT32_MAX) {
            u = s->nuniq;
            s->row_tag[pos] = generation;
            s->row_map[pos] = (uint16_t)u;
            s->uniq[u] = s->req[i];
            s->uniq[u].unique = (uint16_t)u;
            s->nuniq++;
        }
        s->req[i].unique = (uint16_t)u;
    }
    memcpy(s->order, s->uniq, (size_t)s->nuniq * sizeof(*s->order));
    /* s->uniq is no longer needed after the sorted order is built; preserve
     * s->req because q38fn_ngram_wait uses it to restore logical order. */
    sort_requests(s->order, s->uniq, s->nuniq);
    pthread_mutex_lock(&p->mu);
    if (p->stop) {
        s->state = SLOT_FREE;
        pthread_cond_signal(&p->free_slot);
        pthread_mutex_unlock(&p->mu);
        return ECANCELED;
    }
    STAT_ADD(p, submitted, 1); STAT_ADD(p, logical_rows, count);
    STAT_ADD(p, unique_rows, s->nuniq); STAT_ADD(p, deduplicated_rows, count-s->nuniq);
    STAT_ADD(p, useful_bytes, (uint64_t)s->nuniq*Q38FN_NGRAM_ROW_BYTES); s->state=SLOT_READY;
    p->ready_queue[p->ready_tail++ % p->depth] = (uint32_t)si;
    ticket->slot=(uint32_t)si;ticket->generation=generation;pthread_cond_signal(&p->ready);pthread_mutex_unlock(&p->mu);return 0;
}
int q38fn_ngram_submit_token(q38fn_ngram_pipeline *p, uint64_t current,
                             uint64_t previous, uint64_t previous2,
                             q38fn_ngram_ticket *ticket) {
    uint64_t rows[Q38FN_NGRAM_HEADS];
    q38fn_ngram_rows(current, previous, previous2, rows);
    return q38fn_ngram_submit(p, rows, Q38FN_NGRAM_HEADS, ticket);
}
int q38fn_ngram_submit_token_window(q38fn_ngram_pipeline *p,
                                    const uint64_t *current,
                                    const uint64_t *previous,
                                    const uint64_t *previous2,
                                    uint32_t count,
                                    q38fn_ngram_ticket *ticket) {
    if (!p || !current || !previous || !previous2 || !ticket ||
        !count || count > Q38FN_NGRAM_MAX_TOKEN_WINDOW) return EINVAL;
    uint64_t rows[Q38FN_NGRAM_MAX_BATCH];
    for (uint32_t i = 0; i < count; ++i)
        q38fn_ngram_rows(current[i], previous[i], previous2[i],
                         rows + (size_t)i * Q38FN_NGRAM_HEADS);
    return q38fn_ngram_submit(p, rows,
                              count * Q38FN_NGRAM_HEADS, ticket);
}
int q38fn_ngram_poll(q38fn_ngram_pipeline*p,q38fn_ngram_ticket t){if(!p||t.slot>=p->depth)return EINVAL;pthread_mutex_lock(&p->mu);batch_slot*s=&p->slots[t.slot];int rc=s->generation!=t.generation?EINVAL:(s->state==SLOT_DONE?0:s->state==SLOT_FAILED?s->error:EAGAIN);pthread_mutex_unlock(&p->mu);return rc;}
int q38fn_ngram_wait(q38fn_ngram_pipeline *p, q38fn_ngram_ticket t,
                     void *out, size_t bytes)
{
    if (!p || !out || t.slot >= p->depth) return EINVAL;
    uint64_t begin = mono_ns();
    pthread_mutex_lock(&p->mu);
    batch_slot *s = &p->slots[t.slot];
    if (bytes < (size_t)s->count * Q38FN_NGRAM_ROW_BYTES) {
        pthread_mutex_unlock(&p->mu);
        return EINVAL;
    }
    while (s->generation == t.generation && s->state != SLOT_DONE &&
           s->state != SLOT_FAILED)
        pthread_cond_wait(&p->done, &p->mu);
    if (s->generation != t.generation) {
        pthread_mutex_unlock(&p->mu);
        return EINVAL;
    }
    int rc = s->error;
    /* Reserve the slot while materializing the result, but do not hold the
     * global state mutex across the potentially large copy.  SLOT_CONSUMING
     * is deliberately not considered free by find_free(). */
    s->state = SLOT_CONSUMING;
    pthread_mutex_unlock(&p->mu);

    if (!rc) {
        int identity = 1;
        for (uint32_t i = 0; i < s->count; ++i)
            if (s->req[i].unique != i) { identity = 0; break; }
        if (identity) {
            copy_rows(out, s->data, s->count);
        } else {
            for (uint32_t i = 0; i < s->count; ++i)
                copy_rows((char *)out + (size_t)i * Q38FN_NGRAM_ROW_BYTES,
                          s->data + (size_t)s->req[i].unique * Q38FN_NGRAM_HEAD_DIM,
                          1);
        }
    }

    pthread_mutex_lock(&p->mu);
    s->state = SLOT_FREE;
    STAT_ADD(p, wait_ns, mono_ns() - begin);
    pthread_cond_signal(&p->free_slot);
    pthread_mutex_unlock(&p->mu);
    return rc;
}
int q38fn_ngram_fd_read_span(void *opaque, uint64_t first, uint32_t rows,
                             void *dst)
{
    q38fn_ngram_fd_source *s = (q38fn_ngram_fd_source *)opaque;
    if (!s || s->fd < 0 || !dst || !rows) return EINVAL;
    size_t n = (size_t)rows * Q38FN_NGRAM_ROW_BYTES;
    size_t done = 0;
    off_t off = s->base + (off_t)(first * Q38FN_NGRAM_ROW_BYTES);
    while (done < n) {
        ssize_t got = pread(s->fd, (char *)dst + done, n - done,
                            off + (off_t)done);
        if (got <= 0) return -1;
        done += (size_t)got;
    }
    return 0;
}
void q38fn_ngram_get_stats(const q38fn_ngram_pipeline *p, q38fn_ngram_stats *st) {
    if (!p || !st) return;
#define STAT_LOAD(field) __atomic_load_n(&p->stats.field, __ATOMIC_RELAXED)
    st->submitted = STAT_LOAD(submitted); st->completed = STAT_LOAD(completed);
    st->logical_rows = STAT_LOAD(logical_rows); st->unique_rows = STAT_LOAD(unique_rows);
    st->spans = STAT_LOAD(spans); st->physical_bytes = STAT_LOAD(physical_bytes);
    st->useful_bytes = STAT_LOAD(useful_bytes); st->deduplicated_rows = STAT_LOAD(deduplicated_rows);
    st->local_rows = STAT_LOAD(local_rows); st->remote_rows = STAT_LOAD(remote_rows);
    st->wait_ns = STAT_LOAD(wait_ns); st->cache_hits = STAT_LOAD(cache_hits);
    st->cache_misses = STAT_LOAD(cache_misses);
    st->direct_spans = STAT_LOAD(direct_spans);
    st->reorder_spans = STAT_LOAD(reorder_spans);
#undef STAT_LOAD
}
