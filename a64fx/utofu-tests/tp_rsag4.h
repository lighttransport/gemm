/* TP4 large-message FP32 all-reduce for prefill.
 * Direct reduce-scatter followed by direct all-gather: every rank sends only
 * three quarters of the tensor in each phase (1.5x tensor bytes total versus
 * 2x for full-buffer recursive doubling). */
#ifndef TP_RSAG4_H
#define TP_RSAG4_H

#include <arm_sve.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <utofu.h>

#define TP_RSAG4_LINE 256
#define TP_RSAG4_STAG 9
#define TP_RSAG4_MAX_TNI 6

typedef struct {
    utofu_vcq_hdl_t vcq[TP_RSAG4_MAX_TNI];
    utofu_vcq_id_t peer_vcq[TP_RSAG4_MAX_TNI][4];
    utofu_stadd_t peer_base[TP_RSAG4_MAX_TNI][4], base[TP_RSAG4_MAX_TNI];
    char *region;
    size_t slot, trailer, region_size;
    int rank, max_count, ntni;
    uint64_t seq;
} tp_rsag4;

static inline size_t tp_rsag4_send(const tp_rsag4 *c, int r) {
    return (size_t)r * c->slot;
}
static inline size_t tp_rsag4_recv(const tp_rsag4 *c, int r) {
    return (size_t)(4 + r) * c->slot;
}
static inline size_t tp_rsag4_reduced(const tp_rsag4 *c) {
    return (size_t)8 * c->slot;
}
static inline size_t tp_rsag4_gather(const tp_rsag4 *c, int r) {
    return (size_t)(9 + r) * c->slot;
}
static inline void tp_rsag4_inval(const volatile void *p) {
    __asm__ __volatile__("dc civac, %0" :: "r"(p) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
}
static inline void tp_rsag4_drain_mrq_one(tp_rsag4 *c, int k) {
    struct utofu_mrq_notice n;
    while (utofu_poll_mrq(c->vcq[k], 0, &n) == UTOFU_SUCCESS) {}
}
static inline int tp_rsag4_put_nb(tp_rsag4 *c, int peer,
        size_t src, size_t dst, size_t bytes, int k) {
    void *cb;
    int rc;
    for (;;) {
        rc = utofu_put(c->vcq[k], c->peer_vcq[k][peer], c->base[k] + src,
                       c->peer_base[k][peer] + dst, bytes, 0,
                       UTOFU_ONESIDED_FLAG_TCQ_NOTICE, NULL);
        if (rc != UTOFU_ERR_BUSY) break;
        utofu_poll_tcq(c->vcq[k], 0, &cb);
    }
    return rc == UTOFU_SUCCESS ? 1 : -1;
}
static inline int tp_rsag4_drain_tcq(tp_rsag4 *c, int issued[]) {
    void *cb;
    for (int k = 0; k < c->ntni; k++) while (issued[k] > 0) {
        int rc = utofu_poll_tcq(c->vcq[k], 0, &cb);
        if (rc == UTOFU_SUCCESS) issued[k]--;
        else if (rc != UTOFU_ERR_NOT_FOUND) return -1;
    }
    for (int k = 0; k < c->ntni; k++) tp_rsag4_drain_mrq_one(c, k);
    return 0;
}
static inline int tp_rsag4_wait(tp_rsag4 *c, size_t off, uint64_t seq) {
    volatile uint64_t *p = (volatile uint64_t *)(c->region + off + c->trailer);
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double start = ts.tv_sec + ts.tv_nsec * 1e-9;
    unsigned spin = 0;
    while (*p < seq) {
        if ((++spin & 7u) == 0) tp_rsag4_inval(p);
        if ((spin & 1023u) == 0)
            for (int k = 0; k < c->ntni; k++) tp_rsag4_drain_mrq_one(c, k);
        if ((spin & 0xfffffu) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &ts);
            if (ts.tv_sec + ts.tv_nsec * 1e-9 - start > 60.0) return -1;
        }
    }
    for (int k = 0; k < c->ntni; k++) tp_rsag4_drain_mrq_one(c, k);
    return 0;
}

static inline int tp_rsag4_init(tp_rsag4 *c, const utofu_vcq_hdl_t vcq[],
        const utofu_vcq_id_t peers[][4], int ntni, int rank, int max_count,
        void (*barrier)(void)) {
    memset(c, 0, sizeof(*c));
    if (rank < 0 || rank >= 4 || max_count < 4 || ntni < 1 || ntni > TP_RSAG4_MAX_TNI) return -1;
    c->rank = rank; c->max_count = max_count; c->ntni = ntni;
    for (int k = 0; k < ntni; k++) { c->vcq[k] = vcq[k];
        for (int r = 0; r < 4; r++) c->peer_vcq[k][r] = peers[k][r]; }
    size_t max_shard = ((size_t)max_count + 3) / 4;
    c->trailer = max_shard * sizeof(float);
    c->slot = (c->trailer + 8 + TP_RSAG4_LINE - 1) & ~(size_t)(TP_RSAG4_LINE - 1);
    c->region_size = 13 * c->slot;
    if (posix_memalign((void **)&c->region, TP_RSAG4_LINE, c->region_size)) return -1;
    memset(c->region, 0, c->region_size);
    for (int k = 0; k < ntni; k++)
        if (utofu_reg_mem_with_stag(c->vcq[k], c->region, c->region_size,
                                    TP_RSAG4_STAG, 0, &c->base[k]) != UTOFU_SUCCESS) {
            free(c->region); c->region = NULL; return -1;
        }
    barrier();
    for (int r = 0; r < 4; r++) {
        for (int k = 0; k < ntni; k++) {
            if (r == rank) c->peer_base[k][r] = c->base[k];
            else if (utofu_query_stadd(c->peer_vcq[k][r], TP_RSAG4_STAG,
                                       &c->peer_base[k][r]) != UTOFU_SUCCESS) return -1;
        }}
    barrier();
    return 0;
}

static inline int tp_rsag4_sum(tp_rsag4 *c, float *buf, int count) {
    if (count < 1 || count > c->max_count) return -1;
    uint64_t seq = ++c->seq;
    int shard = (count + 3) / 4;
    int padded = shard * 4;
    size_t bytes = (size_t)shard * sizeof(float);

    /* Pack the four equal shards, padding only the final one. */
    for (int d = 0; d < 4; d++) {
        float *s = (float *)(c->region + tp_rsag4_send(c, d));
        int start = d * shard, n = count - start;
        if (n > shard) n = shard;
        if (n > 0) memcpy(s, buf + start, (size_t)n * sizeof(float));
        if (n < shard) memset(s + (n > 0 ? n : 0), 0,
                              (size_t)(shard - (n > 0 ? n : 0)) * sizeof(float));
        *(volatile uint64_t *)((char *)s + c->trailer) = seq;
    }
    (void)padded;
    memcpy(c->region + tp_rsag4_recv(c, c->rank),
           c->region + tp_rsag4_send(c, c->rank), bytes);
    *(volatile uint64_t *)(c->region + tp_rsag4_recv(c, c->rank) + c->trailer) = seq;

    int issued[TP_RSAG4_MAX_TNI] = {0};
    int di = 0;
    for (int d = 0; d < 4; d++) if (d != c->rank) {
        int k = di++ % c->ntni;
        int x = tp_rsag4_put_nb(c, d, tp_rsag4_send(c, d),
                                tp_rsag4_recv(c, c->rank), bytes, k);
        if (x < 0) return -1; issued[k] += x;
    }
    if (tp_rsag4_drain_tcq(c, issued)) return -1;
    memset(issued, 0, sizeof(issued)); di = 0;
    for (int d = 0; d < 4; d++) if (d != c->rank) {
        int k = di++ % c->ntni;
        int x = tp_rsag4_put_nb(c, d, tp_rsag4_send(c, d) + c->trailer,
                                tp_rsag4_recv(c, c->rank) + c->trailer, 8, k);
        if (x < 0) return -1; issued[k] += x;
    }
    if (tp_rsag4_drain_tcq(c, issued)) return -1;
    for (int s = 0; s < 4; s++)
        if (tp_rsag4_wait(c, tp_rsag4_recv(c, s), seq)) return -1;

    float *sum = (float *)(c->region + tp_rsag4_reduced(c));
    memcpy(sum, c->region + tp_rsag4_recv(c, 0), bytes);
    for (int s = 1; s < 4; s++) {
        const float *v = (const float *)(c->region + tp_rsag4_recv(c, s));
        for (int i = 0; i < shard; i += (int)svcntw()) {
            svbool_t pg = svwhilelt_b32(i, shard);
            svst1(pg, sum + i, svadd_f32_x(pg, svld1(pg, sum + i), svld1(pg, v + i)));
        }
    }
    *(volatile uint64_t *)(c->region + tp_rsag4_reduced(c) + c->trailer) = seq;
    memcpy(c->region + tp_rsag4_gather(c, c->rank), sum, bytes);
    *(volatile uint64_t *)(c->region + tp_rsag4_gather(c, c->rank) + c->trailer) = seq;

    memset(issued, 0, sizeof(issued)); di = 0;
    for (int d = 0; d < 4; d++) if (d != c->rank) {
        int k = di++ % c->ntni;
        int x = tp_rsag4_put_nb(c, d, tp_rsag4_reduced(c),
                                tp_rsag4_gather(c, c->rank), bytes, k);
        if (x < 0) return -1; issued[k] += x;
    }
    if (tp_rsag4_drain_tcq(c, issued)) return -1;
    memset(issued, 0, sizeof(issued)); di = 0;
    for (int d = 0; d < 4; d++) if (d != c->rank) {
        int k = di++ % c->ntni;
        int x = tp_rsag4_put_nb(c, d, tp_rsag4_reduced(c) + c->trailer,
                                tp_rsag4_gather(c, c->rank) + c->trailer, 8, k);
        if (x < 0) return -1; issued[k] += x;
    }
    if (tp_rsag4_drain_tcq(c, issued)) return -1;
    for (int s = 0; s < 4; s++) {
        if (tp_rsag4_wait(c, tp_rsag4_gather(c, s), seq)) return -1;
        int start = s * shard, n = count - start;
        if (n > shard) n = shard;
        if (n > 0) memcpy(buf + start, c->region + tp_rsag4_gather(c, s),
                          (size_t)n * sizeof(float));
    }
    return 0;
}

/* BF16-wire TP4 all-reduce for BF16-activation prefill. Each rank truncates
 * its FP32 partials before reduce-scatter; the owner widens and sums ranks
 * 0..3 in deterministic order, truncates the reduced shard for all-gather,
 * and every receiver widens the gathered result back into buf. Network bytes
 * are exactly half of tp_rsag4_sum while all additions remain FP32. */
static inline int tp_rsag4_sum_bf16(tp_rsag4 *c, float *buf, int count) {
    if (count < 1 || count > c->max_count) return -1;
    uint64_t seq = ++c->seq;
    int shard = (count + 3) / 4;
    size_t bytes = (size_t)shard * sizeof(uint16_t);

    for (int d = 0; d < 4; d++) {
        uint16_t *s = (uint16_t *)(c->region + tp_rsag4_send(c, d));
        int start = d * shard, n = count - start;
        if (n > shard) n = shard;
        if (n < 0) n = 0;
#if defined(__ARM_FEATURE_SVE)
        int i = 0;
        for (; i < n; i += (int)svcntw()) {
            svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)n);
            svuint32_t u = svlsr_n_u32_x(pg,
                svreinterpret_u32_f32(svld1(pg, buf + start + i)), 16);
            svst1h_u32(pg, s + i, u);
        }
#else
        for (int i = 0; i < n; i++) {
            uint32_t u; memcpy(&u, buf + start + i, sizeof(u));
            s[i] = (uint16_t)(u >> 16);
        }
#endif
        if (n < shard) memset(s + n, 0, (size_t)(shard - n) * sizeof(*s));
        *(volatile uint64_t *)((char *)s + c->trailer) = seq;
    }
    memcpy(c->region + tp_rsag4_recv(c, c->rank),
           c->region + tp_rsag4_send(c, c->rank), bytes);
    *(volatile uint64_t *)(c->region + tp_rsag4_recv(c, c->rank) + c->trailer) = seq;

    int issued[TP_RSAG4_MAX_TNI] = {0};
    int di = 0;
    for (int d = 0; d < 4; d++) if (d != c->rank) {
        int k = di++ % c->ntni;
        int x = tp_rsag4_put_nb(c, d, tp_rsag4_send(c, d),
                                tp_rsag4_recv(c, c->rank), bytes, k);
        if (x < 0) return -1; issued[k] += x;
    }
    if (tp_rsag4_drain_tcq(c, issued)) return -1;
    memset(issued, 0, sizeof(issued)); di = 0;
    for (int d = 0; d < 4; d++) if (d != c->rank) {
        int k = di++ % c->ntni;
        int x = tp_rsag4_put_nb(c, d, tp_rsag4_send(c, d) + c->trailer,
                                tp_rsag4_recv(c, c->rank) + c->trailer, 8, k);
        if (x < 0) return -1; issued[k] += x;
    }
    if (tp_rsag4_drain_tcq(c, issued)) return -1;
    for (int s = 0; s < 4; s++)
        if (tp_rsag4_wait(c, tp_rsag4_recv(c, s), seq)) return -1;

    uint16_t *sum = (uint16_t *)(c->region + tp_rsag4_reduced(c));
#if defined(__ARM_FEATURE_SVE)
    for (int i = 0; i < shard; i += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)shard);
        const uint16_t *v0 = (const uint16_t *)(c->region + tp_rsag4_recv(c, 0)) + i;
        svuint32_t u = svlsl_n_u32_x(pg, svld1uh_u32(pg, v0), 16);
        svfloat32_t acc = svreinterpret_f32_u32(u);
        for (int s = 1; s < 4; s++) {
            const uint16_t *v = (const uint16_t *)(c->region + tp_rsag4_recv(c, s)) + i;
            u = svlsl_n_u32_x(pg, svld1uh_u32(pg, v), 16);
            acc = svadd_f32_x(pg, acc, svreinterpret_f32_u32(u));
        }
        u = svlsr_n_u32_x(pg, svreinterpret_u32_f32(acc), 16);
        svst1h_u32(pg, sum + i, u);
    }
#else
    for (int i = 0; i < shard; i++) {
        float acc = 0.0f;
        for (int s = 0; s < 4; s++) {
            uint16_t b = ((const uint16_t *)(c->region + tp_rsag4_recv(c, s)))[i];
            uint32_t u = (uint32_t)b << 16; float v; memcpy(&v, &u, sizeof(v));
            acc += v;
        }
        uint32_t u; memcpy(&u, &acc, sizeof(u)); sum[i] = (uint16_t)(u >> 16);
    }
#endif
    *(volatile uint64_t *)(c->region + tp_rsag4_reduced(c) + c->trailer) = seq;
    memcpy(c->region + tp_rsag4_gather(c, c->rank), sum, bytes);
    *(volatile uint64_t *)(c->region + tp_rsag4_gather(c, c->rank) + c->trailer) = seq;

    memset(issued, 0, sizeof(issued)); di = 0;
    for (int d = 0; d < 4; d++) if (d != c->rank) {
        int k = di++ % c->ntni;
        int x = tp_rsag4_put_nb(c, d, tp_rsag4_reduced(c),
                                tp_rsag4_gather(c, c->rank), bytes, k);
        if (x < 0) return -1; issued[k] += x;
    }
    if (tp_rsag4_drain_tcq(c, issued)) return -1;
    memset(issued, 0, sizeof(issued)); di = 0;
    for (int d = 0; d < 4; d++) if (d != c->rank) {
        int k = di++ % c->ntni;
        int x = tp_rsag4_put_nb(c, d, tp_rsag4_reduced(c) + c->trailer,
                                tp_rsag4_gather(c, c->rank) + c->trailer, 8, k);
        if (x < 0) return -1; issued[k] += x;
    }
    if (tp_rsag4_drain_tcq(c, issued)) return -1;
    for (int s = 0; s < 4; s++) {
        if (tp_rsag4_wait(c, tp_rsag4_gather(c, s), seq)) return -1;
        int start = s * shard, n = count - start;
        if (n > shard) n = shard;
        if (n <= 0) continue;
        const uint16_t *g = (const uint16_t *)(c->region + tp_rsag4_gather(c, s));
#if defined(__ARM_FEATURE_SVE)
        for (int i = 0; i < n; i += (int)svcntw()) {
            svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)n);
            svuint32_t u = svlsl_n_u32_x(pg, svld1uh_u32(pg, g + i), 16);
            svst1(pg, buf + start + i, svreinterpret_f32_u32(u));
        }
#else
        for (int i = 0; i < n; i++) {
            uint32_t u = (uint32_t)g[i] << 16;
            memcpy(buf + start + i, &u, sizeof(u));
        }
#endif
    }
    return 0;
}

static inline void tp_rsag4_free(tp_rsag4 *c) {
    if (!c || !c->region) return;
    for (int k = 0; k < c->ntni; k++) utofu_dereg_mem(c->vcq[k], c->base[k], 0);
    free(c->region);
    c->region = NULL;
}

#endif
