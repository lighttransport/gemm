#define _GNU_SOURCE
#include "ngram_utofu_source.h"
#include "../utofu-tests/tofu_demo.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <utofu.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/* Each peer has an independent set of request slots.  The owner services
 * peers concurrently; Put submission is still serialized per VCQ below. */

enum { Q38FN_UTOFU_MAGIC = 0x5133384eUL };
#define Q38FN_UTOFU_XFER_ROWS 32 /* 10,240-byte payload; validate per allocation */
#define Q38FN_UTOFU_CLIENT_STAG 1
#define Q38FN_UTOFU_SERVICE_STAG 2

/* Request and response are written by different CPUs/Tofu engines.  The
 * requester owns request slots and the owner owns response slots: sequence
 * words are monotonic and are never cleared by the reader, avoiding dirty
 * CPU cache lines racing with a later remote Put. */
typedef struct __attribute__((aligned(DEMO_CACHE_LINE))) {
    volatile uint64_t seq;
    uint32_t shard, rows;
    uint64_t first;
    uint32_t magic;
    uint32_t pad;
} q38fn_utofu_req;

typedef struct __attribute__((aligned(DEMO_CACHE_LINE))) {
    volatile uint64_t seq;
    uint32_t rows;
    int32_t status;
    uint32_t magic;
    uint8_t pad[44];
    uint16_t data[Q38FN_UTOFU_XFER_ROWS * Q38FN_NGRAM_HEAD_DIM];
} q38fn_utofu_resp;

typedef struct __attribute__((aligned(DEMO_CACHE_LINE))) {
    volatile uint64_t seq;
} q38fn_utofu_ack;

_Static_assert(sizeof(q38fn_utofu_req) % DEMO_CACHE_LINE == 0,
               "uTofu request slots must not share cache lines");
_Static_assert(sizeof(q38fn_utofu_resp) % DEMO_CACHE_LINE == 0,
               "uTofu response slots must not share cache lines");
_Static_assert(offsetof(q38fn_utofu_resp, data) % 64 == 0,
               "uTofu response payload must be SVE-aligned");
_Static_assert(sizeof(q38fn_utofu_ack) % DEMO_CACHE_LINE == 0,
               "uTofu acknowledgment slots must not share cache lines");

struct q38fn_utofu_source;
struct service_arg {
    struct q38fn_utofu_source *s;
    uint32_t peer;
    uint32_t thread_index;
    uint32_t slot_first;
    uint32_t slot_stride;
};

typedef struct __attribute__((aligned(DEMO_CACHE_LINE))) {
    q38fn_utofu_req req_send[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
    q38fn_utofu_req req_recv[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
    q38fn_utofu_resp resp_send[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
    q38fn_utofu_resp resp_recv[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
    q38fn_utofu_ack ack_send[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
    q38fn_utofu_ack ack_recv[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
} q38fn_utofu_region;

#define Q38FN_REQ_SEND_OFF(i, k)  (offsetof(q38fn_utofu_region, req_send) + \
    ((size_t)(i) * Q38FN_UTOFU_SLOTS + (size_t)(k)) * sizeof(q38fn_utofu_req))
#define Q38FN_REQ_RECV_OFF(i, k)  (offsetof(q38fn_utofu_region, req_recv) + \
    ((size_t)(i) * Q38FN_UTOFU_SLOTS + (size_t)(k)) * sizeof(q38fn_utofu_req))
#define Q38FN_RESP_SEND_OFF(i, k) (offsetof(q38fn_utofu_region, resp_send) + \
    ((size_t)(i) * Q38FN_UTOFU_SLOTS + (size_t)(k)) * sizeof(q38fn_utofu_resp))
#define Q38FN_RESP_RECV_OFF(i, k) (offsetof(q38fn_utofu_region, resp_recv) + \
    ((size_t)(i) * Q38FN_UTOFU_SLOTS + (size_t)(k)) * sizeof(q38fn_utofu_resp))
#define Q38FN_ACK_SEND_OFF(i, k) (offsetof(q38fn_utofu_region, ack_send) + \
    ((size_t)(i) * Q38FN_UTOFU_SLOTS + (size_t)(k)) * sizeof(q38fn_utofu_ack))
#define Q38FN_ACK_RECV_OFF(i, k) (offsetof(q38fn_utofu_region, ack_recv) + \
    ((size_t)(i) * Q38FN_UTOFU_SLOTS + (size_t)(k)) * sizeof(q38fn_utofu_ack))

struct q38fn_utofu_source {
    uint32_t rank, nranks;
    q38fn_utofu_read_fn read_local;
    void *read_opaque;
    uint32_t ack_puts;
    uint64_t request_timeout_ns;
    uint64_t response_retry_ns;
    unsigned response_gap_us;
    q38fn_ngram_source *sources;
    q38fn_ngram_source source_view[Q38FN_NGRAM_SHARDS];
    struct source_arg { struct q38fn_utofu_source *s; uint32_t shard; } args[Q38FN_NGRAM_SHARDS];
    utofu_vcq_hdl_t client_vcq, service_vcq;
    utofu_stadd_t client_base, service_base;
    utofu_stadd_t peer_client_base[Q38FN_UTOFU_MAX_RANKS];
    utofu_stadd_t peer_service_base[Q38FN_UTOFU_MAX_RANKS];
    utofu_vcq_id_t peer_client_vcq[Q38FN_UTOFU_MAX_RANKS];
    utofu_vcq_id_t peer_service_vcq[Q38FN_UTOFU_MAX_RANKS];
    q38fn_utofu_region *region;
    int client_registered, service_registered;
    int mutexes_initialized;
    uint32_t service_started;
    pthread_t service[Q38FN_UTOFU_MAX_SERVICE_THREADS];
    struct service_arg service_args[Q38FN_UTOFU_MAX_SERVICE_THREADS];
    _Atomic int stop;
    pthread_mutex_t *peer_mu;
    pthread_mutex_t *owner_mu;
    uint32_t owner_credits;
    int implicit_ack;
    /* uTofu VCQ handles are not concurrently re-entrant: serialize Put and
     * TCQ polling together, rather than allowing a poll to race a Put. */
    pthread_mutex_t client_vcq_mu, service_vcq_mu, query_mu;
    _Atomic unsigned char peer_base_ready[Q38FN_UTOFU_MAX_RANKS];
    _Atomic unsigned next_slot[Q38FN_UTOFU_MAX_RANKS];
    uint64_t tx_seq[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
    uint64_t rx_seq[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
    uint64_t pending_seq[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
    size_t pending_bytes[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
    unsigned pending_retry[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
    uint64_t pending_started_ns[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
    uint64_t pending_last_send_ns[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
    unsigned char pending[Q38FN_UTOFU_MAX_RANKS][Q38FN_UTOFU_SLOTS];
};

static size_t peer_slot_index(uint32_t peer, uint32_t slot)
{
    return (size_t)peer * Q38FN_UTOFU_SLOTS + slot;
}

static uint64_t monotonic_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static inline void copy_response_rows(void *restrict dst,
                                      const void *restrict src, uint32_t rows)
{
#if defined(__ARM_FEATURE_SVE)
    uint16_t *d = (uint16_t *)dst;
    const uint16_t *s = (const uint16_t *)src;
    uint64_t count = (uint64_t)rows * Q38FN_NGRAM_HEAD_DIM;
    uint64_t vl = svcnth();
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

static void pin_service_thread(uint32_t index)
{
    const char *enabled = getenv("Q38FN_UTOFU_PIN_SERVICE");
    if (enabled && *enabled == '0') return;
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return;
    int selected = -1;
    uint32_t ncpus = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        ncpus += CPU_ISSET(cpu, &allowed) ? 1u : 0u;
    if (!ncpus) return;
    uint32_t cmgs = 4;
    const char *cmg_count = getenv("Q38FN_UTOFU_CMG_COUNT");
    if (cmg_count && *cmg_count) {
        char *end = NULL;
        unsigned long value = strtoul(cmg_count, &end, 10);
        if (end != cmg_count && *end == '\0' && value >= 1 && value <= 16)
            cmgs = (uint32_t)value;
    }
    if (cmgs > ncpus) cmgs = ncpus;
    uint32_t cores_per_cmg = cmgs ? ncpus / cmgs : ncpus;
    if (!cores_per_cmg) cores_per_cmg = 1;
    uint32_t target = (index % cmgs) * cores_per_cmg + index / cmgs;
    if (target >= ncpus) target = index % ncpus;
    uint32_t ordinal = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        if (ordinal++ == target) { selected = cpu; break; }
    }
    if (selected < 0) return;
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(selected, &one);
    (void)sched_setaffinity(0, sizeof one, &one);
}

static int read_topology(const char *path, uint8_t out[][TOFU_NCOORDS], uint32_t want)
{
    FILE *f = fopen(path, "r"); if (!f) return errno;
    char line[256]; uint32_t n = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (n >= want) { fclose(f); return E2BIG; }
        unsigned rank, c[TOFU_NCOORDS];
        if (sscanf(line, "%u %u %u %u %u %u %u", &rank, &c[0], &c[1],
                   &c[2], &c[3], &c[4], &c[5]) != 7 || rank != n) {
            fclose(f); return EINVAL;
        }
        for (unsigned k = 0; k < TOFU_NCOORDS; ++k) out[n][k] = (uint8_t)c[k];
        ++n;
    }
    fclose(f); return n == want ? 0 : EINVAL;
}

static void put_retry(utofu_vcq_hdl_t local_vcq, utofu_vcq_id_t remote_vcq,
                      pthread_mutex_t *vcq_mu,
                      utofu_stadd_t src, utofu_stadd_t dst, size_t bytes,
                      int drain)
{
    void *notice;
    int rc;
    pthread_mutex_lock(vcq_mu);
    do {
        rc = utofu_put(local_vcq, remote_vcq, src, dst, bytes, 0,
                       drain ? UTOFU_ONESIDED_FLAG_TCQ_NOTICE : 0, NULL);
        if (rc == UTOFU_ERR_BUSY) {
            (void)utofu_poll_tcq(local_vcq, 0, &notice);
        }
    } while (rc == UTOFU_ERR_BUSY);
    if (rc != UTOFU_SUCCESS) {
        if (getenv("Q38FN_UTOFU_DEBUG"))
            fprintf(stderr, "q38fn_utofu put failed rc=%d local=%lu remote=%lu bytes=%zu drain=%d\n",
                    rc, (unsigned long)local_vcq, (unsigned long)remote_vcq, bytes, drain);
        pthread_mutex_unlock(vcq_mu); abort();
    }
    if (!drain) {
        pthread_mutex_unlock(vcq_mu);
        return;
    }
    /* Drain notices already available while holding the VCQ lock.  This
     * runtime does not reliably produce a notice for every Put, so waiting
     * indefinitely here would deadlock the request path. */
    for (;;) {
        rc = utofu_poll_tcq(local_vcq, 0, &notice);
        if (rc == UTOFU_ERR_NOT_FOUND) break;
        if (rc != UTOFU_SUCCESS) {
            if (getenv("Q38FN_UTOFU_DEBUG"))
                fprintf(stderr, "q38fn_utofu drain-poll failed rc=%d local=%lu\n",
                        rc, (unsigned long)local_vcq);
            pthread_mutex_unlock(vcq_mu); abort();
        }
    }
    pthread_mutex_unlock(vcq_mu);
}

static int refresh_peer_base(q38fn_utofu_source *s, uint32_t peer, int service)
{
    int rc = UTOFU_ERR_NOT_FOUND;
    /* Querying the same VCQ from multiple workers is not documented as
     * re-entrant.  This is normally cold-path, but multi-credit mode can
     * discover several peer bases concurrently. */
    pthread_mutex_lock(&s->query_mu);
    for (int tries = 0; tries < 200; ++tries) {
        utofu_vcq_id_t remote = service ? s->peer_service_vcq[peer] : s->peer_client_vcq[peer];
        utofu_stadd_t *base = service ? &s->peer_service_base[peer] : &s->peer_client_base[peer];
        rc = utofu_query_stadd(remote, service ? Q38FN_UTOFU_SERVICE_STAG : Q38FN_UTOFU_CLIENT_STAG,
                               base);
        if (rc == UTOFU_SUCCESS) {
            atomic_fetch_or_explicit(&s->peer_base_ready[peer],
                                     (unsigned char)(service ? 2 : 1),
                                     memory_order_release);
            pthread_mutex_unlock(&s->query_mu);
            return 0;
        }
        usleep(50000);
    }
    pthread_mutex_unlock(&s->query_mu);
    return rc;
}

static void *serve_main(void *opaque)
{
    struct service_arg *arg = (struct service_arg *)opaque;
    q38fn_utofu_source *s = arg->s;
    pin_service_thread(arg->thread_index);
    uint32_t poll_round = 0;
    while (!atomic_load_explicit(&s->stop, memory_order_acquire)) {
        /* Each service thread is assigned one peer and a disjoint subset of
         * that peer's slots.  Avoid rescanning every rank on every poll; this
         * loop runs on a pinned progress core and the saved branches directly
         * increase the time available for HBM reads and response puts. */
        uint32_t peer = arg->peer;
        if (peer != s->rank) {
            for (uint32_t slot = arg->slot_first;
                 slot < Q38FN_UTOFU_SLOTS; slot += arg->slot_stride) {
                volatile q38fn_utofu_req *q =
                    (volatile q38fn_utofu_req *)&s->region->req_recv[peer][slot];
                uint64_t seq = q->seq;
                if (s->pending[peer][slot]) {
                    /* Publishing the next request means the requester has
                     * already consumed this response.  This optional path
                     * replaces the separate acknowledgment Put. */
                    if (s->implicit_ack &&
                        seq == s->pending_seq[peer][slot] + 1) {
                        s->pending[peer][slot] = 0;
                        ++s->rx_seq[peer][slot];
                        continue;
                    }
                    if (*(volatile uint64_t *)&s->region->ack_recv[peer][slot].seq ==
                        s->pending_seq[peer][slot]) {
                        s->pending[peer][slot] = 0;
                        ++s->rx_seq[peer][slot];
                    } else {
                        uint64_t now = monotonic_ns();
                        if (now - s->pending_last_send_ns[peer][slot] >= s->response_retry_ns) {
                            ++s->pending_retry[peer][slot];
                            if (now - s->pending_started_ns[peer][slot] >=
                                s->request_timeout_ns) {
                                /* The requester uses the same timeout.
                                 * Abandon the response and consume the old
                                 * request so a dead rank cannot permanently
                                 * occupy this slot. */
                                s->pending[peer][slot] = 0;
                                ++s->rx_seq[peer][slot];
                            } else {
                                put_retry(s->service_vcq, s->peer_client_vcq[peer],
                                          &s->service_vcq_mu,
                                          s->service_base + Q38FN_RESP_SEND_OFF(peer, slot),
                                          s->peer_client_base[peer] + Q38FN_RESP_RECV_OFF(s->rank, slot),
                                          s->pending_bytes[peer][slot], 0);
                                s->pending_last_send_ns[peer][slot] = now;
                                if (s->response_gap_us) usleep(s->response_gap_us);
                }
            }
        }
                    continue;
                }
                if (seq && (q->magic != Q38FN_UTOFU_MAGIC ||
                            q->rows > Q38FN_UTOFU_XFER_ROWS) &&
                    getenv("Q38FN_UTOFU_DEBUG"))
                    fprintf(stderr, "q38fn_utofu rank=%u bad request peer=%u slot=%u seq=%llu magic=0x%08x rows=%u\n",
                            s->rank, peer, slot, (unsigned long long)seq,
                            q->magic, q->rows);
                if (!seq || q->magic != Q38FN_UTOFU_MAGIC || q->rows > Q38FN_UTOFU_XFER_ROWS)
                    continue;
                q38fn_utofu_resp *r = &s->region->resp_send[peer][slot];
                uint64_t expected = s->rx_seq[peer][slot];
                if (seq != expected) continue;
                if (getenv("Q38FN_UTOFU_DEBUG"))
                    fprintf(stderr, "q38fn_utofu rank=%u received peer=%u slot=%u shard=%u seq=%llu\n",
                            s->rank, peer, slot, q->shard, (unsigned long long)seq);
                int rc = (q->shard < Q38FN_NGRAM_SHARDS && s->read_local) ?
                    s->read_local(s->read_opaque, q->shard, q->first, q->rows, r->data) : -1;
                if (rc && getenv("Q38FN_UTOFU_DEBUG"))
                    fprintf(stderr, "q38fn_utofu rank=%u request shard=%u first=%llu rows=%u rc=%d\n",
                            s->rank, q->shard, (unsigned long long)q->first, q->rows, rc);
                r->rows = q->rows; r->status = rc; r->magic = Q38FN_UTOFU_MAGIC;
                __atomic_thread_fence(__ATOMIC_RELEASE);
                r->seq = seq;
                if (!(atomic_load_explicit(&s->peer_base_ready[peer], memory_order_acquire) & 1) &&
                    refresh_peer_base(s, peer, 0)) {
                    if (getenv("Q38FN_UTOFU_DEBUG"))
                        fprintf(stderr, "q38fn_utofu response-base refresh failed rank=%u peer=%u\n",
                                s->rank, peer);
                    abort();
                }
                size_t response_bytes = offsetof(q38fn_utofu_resp, data) +
                    (size_t)q->rows * Q38FN_NGRAM_ROW_BYTES;
                /* uTofu moves cache-line-sized chunks.  A 320-byte row plus
                 * the 16-byte response header would otherwise transfer only
                 * the first 256 bytes (the tail lane is silently absent). */
                response_bytes = (response_bytes + DEMO_CACHE_LINE - 1) &
                                 ~(size_t)(DEMO_CACHE_LINE - 1);
                s->pending_seq[peer][slot] = seq;
                s->pending_bytes[peer][slot] = response_bytes;
                s->pending_retry[peer][slot] = 0;
                s->pending_started_ns[peer][slot] = monotonic_ns();
                s->pending_last_send_ns[peer][slot] = s->pending_started_ns[peer][slot];
                s->pending[peer][slot] = 1;
                put_retry(s->service_vcq, s->peer_client_vcq[peer],
                          &s->service_vcq_mu,
                          s->service_base + Q38FN_RESP_SEND_OFF(peer, slot),
                          s->peer_client_base[peer] + Q38FN_RESP_RECV_OFF(s->rank, slot),
                          response_bytes, 0);
                if (s->response_gap_us) usleep(s->response_gap_us);
            }
        }
        /* Keep the hot path in user-space polling.  Yield periodically so a
         * lightly loaded owner does not monopolize its pinned CPU. */
        if ((++poll_round & 63u) == 0)
            sched_yield();
    }
    return NULL;
}

static int read_remote(void *opaque, uint64_t first, uint32_t rows, void *dst)
{
    struct source_arg *a = (struct source_arg *)opaque;
    q38fn_utofu_source *s = a->s;
    uint32_t owner = a->shard % s->nranks;
    if (owner == s->rank)
        return s->read_local ? s->read_local(s->read_opaque, a->shard, first, rows, dst) : -1;
    if (!rows || rows > Q38FN_UTOFU_MAX_SPAN) return E2BIG;
    if (rows > Q38FN_UTOFU_XFER_ROWS) {
        uint32_t done = 0;
        while (done < rows) {
            uint32_t n = rows - done;
            if (n > Q38FN_UTOFU_XFER_ROWS) n = Q38FN_UTOFU_XFER_ROWS;
            int frc = read_remote(opaque, first + done, n,
                                  (char *)dst + (size_t)done * Q38FN_NGRAM_ROW_BYTES);
            if (frc) return frc;
            done += n;
        }
        return 0;
    }
    /* The credit count is also the number of slots eligible for this owner.
     * Otherwise a setting of two would still cycle through all four slots and
     * silently exceed the requested MRQ bound. */
    uint32_t slot = atomic_fetch_add_explicit(&s->next_slot[owner], 1,
                                              memory_order_relaxed) % s->owner_credits;
    /* Each slot is a bounded credit.  The default two-credit mode overlaps
     * owner transfers without exposing an unrestricted burst; set the
     * environment variable to one for the conservative owner-wide mode. */
    pthread_mutex_t *slot_mu = &s->peer_mu[peer_slot_index(owner, slot)];
    pthread_mutex_t *held_mu = s->owner_credits == 1 ? &s->owner_mu[owner] : slot_mu;
    pthread_mutex_lock(held_mu);
    if (!(atomic_load_explicit(&s->peer_base_ready[owner], memory_order_acquire) & 2) &&
        refresh_peer_base(s, owner, 1)) {
        pthread_mutex_unlock(held_mu);
        return EIO;
    }
    uint64_t seq = ++s->tx_seq[owner][slot];
    q38fn_utofu_req *q = &s->region->req_send[owner][slot];
    q->shard = a->shard; q->rows = rows; q->first = first; q->magic = Q38FN_UTOFU_MAGIC;
    __atomic_thread_fence(__ATOMIC_RELEASE); q->seq = seq;
    put_retry(s->client_vcq, s->peer_service_vcq[owner],
              &s->client_vcq_mu,
              s->client_base + Q38FN_REQ_SEND_OFF(owner, slot),
              s->peer_service_base[owner] + Q38FN_REQ_RECV_OFF(s->rank, slot),
              sizeof(*q), 1);
    if (getenv("Q38FN_UTOFU_DEBUG"))
        fprintf(stderr, "q38fn_utofu rank=%u sent owner=%u slot=%u seq=%llu\n",
                s->rank, owner, slot, (unsigned long long)seq);
    volatile q38fn_utofu_resp *r = &s->region->resp_recv[owner][slot];
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    uint64_t start_ns = (uint64_t)start.tv_sec * 1000000000ull + (uint64_t)start.tv_nsec;
    uint64_t last_request_retry_ns = start_ns;
    uint64_t request_retry_ns = 1000000000ull;
    const char *retry_env = getenv("Q38FN_UTOFU_REQUEST_RETRY_MS");
    if (retry_env && *retry_env) {
        char *end = NULL;
        unsigned long value = strtoul(retry_env, &end, 10);
        if (end != retry_env && *end == '\0' && value >= 1 && value <= 10000)
            request_retry_ns = (uint64_t)value * 1000000ull;
    }
    uint32_t poll_round = 0;
    while (r->seq != seq) {
        void *notice;
        pthread_mutex_lock(&s->client_vcq_mu);
        (void)utofu_poll_tcq(s->client_vcq, 0, &notice);
        pthread_mutex_unlock(&s->client_vcq_mu);
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        uint64_t now_ns = (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
        /* There is no cross-rank barrier in the MPI-free adapter.  A Put
         * issued while the peer is still registering its MR can be dropped;
         * retransmit the same sequence until the response arrives, matching
         * the retry-until-receive rule in tofu_put_demo. */
        if (now_ns - last_request_retry_ns >= request_retry_ns) {
            put_retry(s->client_vcq, s->peer_service_vcq[owner],
                      &s->client_vcq_mu,
                      s->client_base + Q38FN_REQ_SEND_OFF(owner, slot),
                      s->peer_service_base[owner] + Q38FN_REQ_RECV_OFF(s->rank, slot),
                      sizeof(*q), 1);
            last_request_retry_ns = now_ns;
        }
        if (now_ns - start_ns > s->request_timeout_ns) {
            pthread_mutex_unlock(held_mu);
            return ETIMEDOUT;
        }
        if ((++poll_round & 63u) == 0)
            sched_yield();
    }
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    int rc = r->status;
    if (!rc) copy_response_rows(dst, (const uint16_t *)(const void *)r->data, rows);
    if (!s->implicit_ack) {
        q38fn_utofu_ack *ack = &s->region->ack_send[owner][slot];
        ack->seq = seq;
        __atomic_thread_fence(__ATOMIC_RELEASE);
        for (uint32_t retry = 0; retry < s->ack_puts; ++retry)
            put_retry(s->client_vcq, s->peer_service_vcq[owner],
                      &s->client_vcq_mu,
                      s->client_base + Q38FN_ACK_SEND_OFF(owner, slot),
                      s->peer_service_base[owner] + Q38FN_ACK_RECV_OFF(s->rank, slot),
                      sizeof(*ack), 0);
    }
    pthread_mutex_unlock(held_mu);
    return rc;
}

int q38fn_utofu_source_init(q38fn_utofu_source **out, const char *topology,
                            uint32_t rank, uint32_t nranks,
                            q38fn_utofu_read_fn read_local, void *read_opaque,
                            q38fn_ngram_source *sources, uint32_t source_count)
{
    if (!out || !topology || !sources || source_count != Q38FN_NGRAM_SHARDS ||
        !nranks || nranks > Q38FN_UTOFU_MAX_RANKS || rank >= nranks || !read_local)
        return EINVAL;
    q38fn_utofu_source *s = calloc(1, sizeof *s); if (!s) return ENOMEM;
    int rc = 0;
    const char *fail_stage = "alloc";
    uint32_t found = nranks;
    int tries = 0;
    utofu_tni_id_t *tnis = NULL;
    size_t ntni = 0;
    utofu_tni_id_t tni = 0, service_tni = 0;
    s->rank = rank; s->nranks = nranks; s->read_local = read_local;
    const char *response_gap = getenv("Q38FN_UTOFU_RESPONSE_GAP_US");
    if (response_gap && *response_gap) {
        char *end = NULL;
        unsigned long value = strtoul(response_gap, &end, 10);
        if (end != response_gap && *end == '\0' && value <= 1000000)
            s->response_gap_us = (unsigned)value;
    }
    s->ack_puts = 1;
    const char *ack_count = getenv("Q38FN_UTOFU_ACK_PUTS");
    if (ack_count && *ack_count) {
        char *end = NULL;
        unsigned long value = strtoul(ack_count, &end, 10);
        if (end != ack_count && *end == '\0' && value >= 1 && value <= 8)
            s->ack_puts = (uint32_t)value;
    }
    s->response_retry_ns = 100000000ull;
    s->request_timeout_ns = 10000000000ull;
    const char *timeout_env = getenv("Q38FN_UTOFU_TIMEOUT_MS");
    if (timeout_env && *timeout_env) {
        char *end = NULL;
        unsigned long value = strtoul(timeout_env, &end, 10);
        if (end != timeout_env && *end == '\0' && value >= 1000 && value <= 600000)
            s->request_timeout_ns = (uint64_t)value * 1000000ull;
    }
    /* Two credits overlap independent owner-side transfers without exposing
     * an unrestricted burst to the allocation's finite MRQ depth. */
    s->owner_credits = 2;
    const char *credits = getenv("Q38FN_UTOFU_OWNER_CREDITS");
    if (credits && *credits) {
        char *end = NULL;
        unsigned long value = strtoul(credits, &end, 10);
        if (end != credits && *end == '\0' && value >= 1 && value <= Q38FN_UTOFU_SLOTS)
            s->owner_credits = (uint32_t)value;
    }
    /* A single service thread leaves independent HBM reads serialized even
     * when several request credits are available.  Split the slots across a
     * small, bounded number of owner threads; VCQ puts remain serialized by
     * service_vcq_mu and the credit count still caps in-flight transfers. */
    uint32_t service_threads = 2;
    const char *service_env = getenv("Q38FN_UTOFU_SERVICE_THREADS");
    if (service_env && *service_env) {
        char *end = NULL;
        unsigned long value = strtoul(service_env, &end, 10);
        if (end != service_env && *end == '\0' && value >= 1 &&
            value <= Q38FN_UTOFU_SLOTS)
            service_threads = (uint32_t)value;
    }
    if (service_threads > s->owner_credits)
        service_threads = s->owner_credits;
    /* Publishing the next request is a complete acknowledgment because the
     * requester cannot reuse a slot until it has copied the response.  This
     * removes one network Put from the steady state; set the knob to zero to
     * retain the explicit-ack compatibility protocol. */
    s->implicit_ack = 1;
    const char *implicit_ack = getenv("Q38FN_UTOFU_IMPLICIT_ACK");
    if (implicit_ack && *implicit_ack == '0') s->implicit_ack = 0;
    const char *response_retry = getenv("Q38FN_UTOFU_RESPONSE_RETRY_MS");
    if (response_retry && *response_retry) {
        char *end = NULL;
        unsigned long value = strtoul(response_retry, &end, 10);
        if (end != response_retry && *end == '\0' && value >= 1 && value <= 10000)
            s->response_retry_ns = (uint64_t)value * 1000000ull;
    }
    s->read_opaque = read_opaque; s->sources = sources;
    s->peer_mu = calloc((size_t)nranks * Q38FN_UTOFU_SLOTS, sizeof *s->peer_mu);
    s->owner_mu = calloc(nranks, sizeof *s->owner_mu);
    if (posix_memalign((void **)&s->region, DEMO_CACHE_LINE, sizeof *s->region) != 0)
        s->region = NULL;
    if (!s->peer_mu || !s->owner_mu || !s->region) { q38fn_utofu_source_destroy(s); return ENOMEM; }
    memset(s->region, 0, sizeof *s->region);
    for (uint32_t i = 0; i < nranks; ++i)
        for (uint32_t k = 0; k < Q38FN_UTOFU_SLOTS; ++k)
            pthread_mutex_init(&s->peer_mu[peer_slot_index(i, k)], NULL);
    for (uint32_t i = 0; i < nranks; ++i)
        pthread_mutex_init(&s->owner_mu[i], NULL);
    pthread_mutex_init(&s->client_vcq_mu, NULL);
    pthread_mutex_init(&s->service_vcq_mu, NULL);
    pthread_mutex_init(&s->query_mu, NULL);
    s->mutexes_initialized = 1;
    uint8_t topo[Q38FN_UTOFU_MAX_RANKS][TOFU_NCOORDS], mine[TOFU_NCOORDS];
    fail_stage = "read_topology";
    rc = read_topology(topology, topo, nranks);
    if (rc) { q38fn_utofu_source_destroy(s); return rc; }
    fail_stage = "get_onesided_tnis";
    rc = utofu_get_onesided_tnis(&tnis, &ntni);
    if (rc != UTOFU_SUCCESS || ntni < 2) { rc = EIO; free(tnis); goto fail; }
    tni = tnis[0];
    service_tni = tnis[1];
    /* Fujitsu's uTofu runtime publishes the process coordinates only after
     * the one-sided TNI set has been queried.  Keep this ordering aligned with
     * the supported initialization sequence used by the other adapters. */
    fail_stage = "query_my_coords";
    rc = utofu_query_my_coords(mine); if (rc != UTOFU_SUCCESS) { free(tnis); goto fail; }
    for (uint32_t i = 0; i < nranks; ++i) if (!memcmp(mine, topo[i], TOFU_NCOORDS)) found = i;
    if (found != rank) { rc = EINVAL; free(tnis); goto fail; }
    fail_stage = "create_client_vcq";
    rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &s->client_vcq);
    free(tnis); tnis = NULL; if (rc != UTOFU_SUCCESS) goto fail;
    utofu_vcq_id_t client_actual;
    utofu_tni_id_t client_tni_actual;
    utofu_cq_id_t client_cq;
    uint8_t ignored_coords[TOFU_NCOORDS]; uint16_t ignored_extra;
    rc = utofu_query_vcq_id(s->client_vcq, &client_actual); if (rc != UTOFU_SUCCESS) goto fail;
    fail_stage = "create_service_vcq";
    rc = utofu_create_vcq_with_cmp_id(service_tni, DEMO_CMP_ID, 0, &s->service_vcq);
    if (rc != UTOFU_SUCCESS) goto fail;
    utofu_vcq_id_t service_actual;
    utofu_tni_id_t service_tni_actual;
    utofu_cq_id_t service_cq;
    rc = utofu_query_vcq_info(client_actual, ignored_coords, &client_tni_actual, &client_cq, &ignored_extra);
    if (rc != UTOFU_SUCCESS) goto fail;
    rc = utofu_query_vcq_id(s->service_vcq, &service_actual); if (rc != UTOFU_SUCCESS) goto fail;
    rc = utofu_query_vcq_info(service_actual, ignored_coords, &service_tni_actual, &service_cq, &ignored_extra);
    if (rc != UTOFU_SUCCESS) goto fail;
    /* The peer VCQ IDs are reconstructed from the topology.  Validate the
     * convention against the VCQ actually allocated by this process before
     * using any remote STADD; otherwise a wrong CQ convention can silently
     * turn a peer Put into a local/self-directed Put. */
    {
        utofu_vcq_id_t actual, convention;
        rc = utofu_query_vcq_id(s->client_vcq, &actual);
        if (rc != UTOFU_SUCCESS) { fail_stage = "query_vcq_id"; goto fail; }
        rc = utofu_construct_vcq_id(mine, client_tni_actual, client_cq, DEMO_CMP_ID, &convention);
        if (rc != UTOFU_SUCCESS) { fail_stage = "construct_self_vcq_id"; goto fail; }
        utofu_set_vcq_id_path(&actual, NULL);
        utofu_set_vcq_id_path(&convention, NULL);
        if (getenv("Q38FN_UTOFU_DEBUG"))
            fprintf(stderr, "q38fn_utofu rank=%u vcq_actual=0x%016lx vcq_convention=0x%016lx client_tni=%u client_cq=%u service_vcq=0x%016lx service_tni=%u service_cq=%u\n",
                    rank, (unsigned long)actual, (unsigned long)convention,
                    (unsigned)client_tni_actual, (unsigned)client_cq,
                    (unsigned long)service_actual, (unsigned)service_tni_actual,
                    (unsigned)service_cq);
        if (actual != convention) { rc = EINVAL; fail_stage = "vcq_convention"; goto fail; }
    }
    fail_stage = "reg_mem";
    rc = utofu_reg_mem_with_stag(s->client_vcq, s->region, sizeof *s->region,
                                 Q38FN_UTOFU_CLIENT_STAG, 0, &s->client_base);
    if (rc != UTOFU_SUCCESS) goto fail;
    s->client_registered = 1;
    rc = utofu_reg_mem_with_stag(s->service_vcq, s->region, sizeof *s->region,
                                 Q38FN_UTOFU_SERVICE_STAG, 0, &s->service_base);
    if (rc != UTOFU_SUCCESS) goto fail;
    s->service_registered = 1;
    for (uint32_t i = 0; i < nranks; ++i) {
        fail_stage = "construct_client_vcq";
        rc = utofu_construct_vcq_id(topo[i], client_tni_actual, client_cq, DEMO_CMP_ID,
                                    &s->peer_client_vcq[i]);
        if (rc != UTOFU_SUCCESS) goto fail;
        utofu_set_vcq_id_path(&s->peer_client_vcq[i], NULL);
        fail_stage = "construct_service_vcq";
        rc = utofu_construct_vcq_id(topo[i], service_tni_actual, service_cq, DEMO_CMP_ID,
                                    &s->peer_service_vcq[i]);
        if (rc != UTOFU_SUCCESS) goto fail;
        utofu_set_vcq_id_path(&s->peer_service_vcq[i], NULL);
        fail_stage = "query_stadd";
        tries = 0;
        do {
            rc = utofu_query_stadd(s->peer_client_vcq[i], Q38FN_UTOFU_CLIENT_STAG,
                                   &s->peer_client_base[i]);
            if (rc == UTOFU_SUCCESS) break;
            usleep(50000);
        } while (++tries < 200);
        if (rc != UTOFU_SUCCESS) goto fail;
        tries = 0;
        do {
            rc = utofu_query_stadd(s->peer_service_vcq[i], Q38FN_UTOFU_SERVICE_STAG,
                                   &s->peer_service_base[i]);
            if (rc == UTOFU_SUCCESS) break;
            usleep(50000);
        } while (++tries < 200);
        if (rc != UTOFU_SUCCESS) goto fail;
        if (getenv("Q38FN_UTOFU_DEBUG"))
            fprintf(stderr, "q38fn_utofu rank=%u peer=%u client_vcq=0x%016lx service_vcq=0x%016lx client_base=0x%016lx service_base=0x%016lx\n",
                    rank, i, (unsigned long)s->peer_client_vcq[i],
                    (unsigned long)s->peer_service_vcq[i],
                    (unsigned long)s->peer_client_base[i],
                    (unsigned long)s->peer_service_base[i]);
    }
    /* The first successful query may occur before the peer's registration is
     * fully visible to the data path.  Resolve again lazily on first traffic. */
    for (uint32_t i = 0; i < nranks; ++i) {
        atomic_store_explicit(&s->peer_base_ready[i], 0, memory_order_relaxed);
        atomic_store_explicit(&s->next_slot[i], 0, memory_order_relaxed);
        for (uint32_t k = 0; k < Q38FN_UTOFU_SLOTS; ++k) {
            s->tx_seq[i][k] = 0;
            s->rx_seq[i][k] = 1;
        }
    }
    for (uint32_t shard = 0; shard < Q38FN_NGRAM_SHARDS; ++shard) {
        s->args[shard].s = s; s->args[shard].shard = shard;
        s->source_view[shard] = (q38fn_ngram_source){ read_remote, &s->args[shard], shard % nranks };
        sources[shard] = s->source_view[shard];
    }
    for (uint32_t peer = 0; peer < nranks; ++peer) {
        if (peer == rank) continue;
        for (uint32_t thread = 0; thread < service_threads; ++thread) {
            if (s->service_started >= Q38FN_UTOFU_MAX_SERVICE_THREADS) {
                rc = E2BIG;
                goto fail;
            }
            struct service_arg *arg = &s->service_args[s->service_started];
            arg->s = s;
            arg->peer = peer;
            arg->thread_index = s->service_started;
            arg->slot_first = thread;
            arg->slot_stride = service_threads;
            if (pthread_create(&s->service[s->service_started], NULL,
                               serve_main, arg) != 0) {
                rc = EAGAIN;
                goto fail;
            }
            ++s->service_started;
        }
    }
    /* The launcher starts ranks independently.  Let every rank enter its
     * polling loop before callers can issue the first request; otherwise the
     * first one-sided Put can race remote registration/service startup. */
    unsigned startup_wait = 5;
    const char *startup_env = getenv("Q38FN_UTOFU_STARTUP_WAIT");
    if (startup_env && *startup_env) {
        char *end = NULL;
        unsigned long value = strtoul(startup_env, &end, 10);
        if (end != startup_env && *end == '\0' && value <= 60)
            startup_wait = (unsigned)value;
    }
    sleep(startup_wait);
    *out = s; return 0;
fail:
    if (getenv("Q38FN_UTOFU_DEBUG"))
        fprintf(stderr, "q38fn_utofu rank=%u stage=%s rc=%d\n", rank, fail_stage, rc);
    q38fn_utofu_source_destroy(s); return rc;
}

void q38fn_utofu_source_destroy(q38fn_utofu_source *s)
{
    if (!s) return;
    /* There is no MPI barrier in this adapter.  A faster rank may otherwise
     * deregister its response MR while a peer is still completing its final
     * request.  Keep the service endpoint alive for a short, configurable
     * drain interval before stopping its polling threads. */
    unsigned shutdown_wait = 2;
    const char *shutdown_env = getenv("Q38FN_UTOFU_SHUTDOWN_WAIT");
    if (shutdown_env && *shutdown_env) {
        char *end = NULL;
        unsigned long value = strtoul(shutdown_env, &end, 10);
        if (end != shutdown_env && *end == '\0' && value <= 60)
            shutdown_wait = (unsigned)value;
    }
    sleep(shutdown_wait);
    atomic_store_explicit(&s->stop, 1, memory_order_release);
    for (uint32_t i = 0; i < s->service_started; ++i)
        pthread_join(s->service[i], NULL);
    if (s->client_vcq) {
        if (s->client_registered) utofu_dereg_mem(s->client_vcq, s->client_base, 0);
        utofu_free_vcq(s->client_vcq);
    }
    if (s->service_vcq && s->service_vcq != s->client_vcq) {
        if (s->service_registered) utofu_dereg_mem(s->service_vcq, s->service_base, 0);
        utofu_free_vcq(s->service_vcq);
    }
    if (s->peer_mu) {
        if (s->mutexes_initialized)
            for (uint32_t i = 0; i < s->nranks; ++i)
                for (uint32_t k = 0; k < Q38FN_UTOFU_SLOTS; ++k)
                    pthread_mutex_destroy(&s->peer_mu[peer_slot_index(i, k)]);
        free(s->peer_mu);
    }
    if (s->owner_mu) {
        if (s->mutexes_initialized)
            for (uint32_t i = 0; i < s->nranks; ++i)
                pthread_mutex_destroy(&s->owner_mu[i]);
        free(s->owner_mu);
    }
    if (s->mutexes_initialized) {
        pthread_mutex_destroy(&s->client_vcq_mu);
        pthread_mutex_destroy(&s->service_vcq_mu);
        pthread_mutex_destroy(&s->query_mu);
    }
    free(s->region); free(s);
}
