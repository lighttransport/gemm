/*
 * Multi-TNI striping microbench (A64FX / Fugaku, MPI-free uTofu).
 *
 * Question: the ring-attention decode comm is per-hop software/BW limited
 * (~1.23 us fixed Put overhead + payload at ~6.4 GB/s ~= one TofuD link). A64FX
 * exposes 6 onesided TNIs. Does striping a single hop's payload across K TNIs
 * to the SAME peer raise effective per-hop bandwidth -- i.e. do the TNIs reach
 * the peer over independent link bandwidth, or do they share one link?
 *
 * Method: a 2-rank ping-pong between rank 0 and rank 1 (other ranks exit). Each
 * "send" stripes a payload of TS_BYTES across K = TS_NTNI TNIs as K concurrent
 * Puts (one VCQ per TNI), each chunk carrying a trailing seq the receiver polls.
 * per-hop latency = round-trip / 2. Sweep K via the env var across runs.
 *
 * Mirrors ring_attn_bench's conventions: identical binary on every node, peer
 * VCQ-id reconstructed from coords (no runtime exchange), each recv slot on its
 * own 256 B cache line (the CPU only reads it -- see tofu_demo.h).
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -o tni_stripe_bench tni_stripe_bench.c -ltofucom
 * Run:   TS_NTNI=2 mpiexec -np 12 ./tni_stripe_bench   (needs tofu_topo.txt; rank 0 appends ts_result.txt)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <utofu.h>
#include "tofu_demo.h"

#define MAX_NODES   12
#define BENCH_STAG  1
#define MAX_TNI     6
#define WAIT_TIMEOUT_SEC 30.0

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static void die(const char *m, int rc) { fprintf(stderr, "FATAL: %s (rc=%d)\n", m, rc); exit(1); }
static long envl(const char *n, long d) { const char *v = getenv(n); return v ? atol(v) : d; }

static int read_topo(uint8_t topo[][TOFU_NCOORDS]) {
    FILE *f = fopen(TOPO_PATH, "r");
    if (!f) die("open " TOPO_PATH, -1);
    char line[256];
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        int r; unsigned c[6];
        if (sscanf(line, "%d %u %u %u %u %u %u", &r, &c[0], &c[1], &c[2], &c[3], &c[4], &c[5]) == 7) {
            for (int k = 0; k < 6; k++) topo[r][k] = (uint8_t)c[k];
            if (r + 1 > n) n = r + 1;
        }
    }
    fclose(f);
    return n;
}

int main(void)
{
    int rc;
    long P      = envl("TS_BYTES", 16640);   /* total payload to stripe per hop  */
    int  K      = (int)envl("TS_NTNI", 1);   /* number of TNIs to stripe across  */
    long ITERS  = envl("TS_ITERS", 5000);
    long WARMUP = envl("TS_WARMUP", 500);

    utofu_tni_id_t *tni_ids = NULL;
    size_t num_tnis = 0;
    rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
    if (rc != UTOFU_SUCCESS) die("utofu_get_onesided_tnis", rc);
    if (K < 1 || (size_t)K > num_tnis || K > MAX_TNI) die("TS_NTNI out of range vs available TNIs", -1);

    uint8_t my_coords[TOFU_NCOORDS] = {0};
    rc = utofu_query_my_coords(my_coords);
    if (rc != UTOFU_SUCCESS) die("utofu_query_my_coords", rc);

    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    int nprocs = read_topo(topo);
    int my_rank = -1;
    for (int r = 0; r < nprocs; r++)
        if (memcmp(topo[r], my_coords, TOFU_NCOORDS) == 0) my_rank = r;
    if (my_rank == -1) die("my coords not in topo", -1);

    /* only rank 0 and rank 1 take part; the rest exit cleanly */
    if (my_rank > 1) { free(tni_ids); return 0; }
    int peer = my_rank ^ 1;   /* 0<->1 */

    /* per active TNI: a recv slot and a send slot, each on its own cache line.
     * chunk = bytes one TNI carries; the slot holds chunk + an 8 B trailing seq. */
    size_t chunk = ((size_t)((P + K - 1) / K) + 7) & ~(size_t)7;
    size_t slot  = (chunk + 8 + (DEMO_CACHE_LINE - 1)) & ~(size_t)(DEMO_CACHE_LINE - 1);
    size_t region_sz = (size_t)(2 * K) * slot;   /* [recv 0..K-1][send 0..K-1] */

    void *region = NULL;
    if (posix_memalign(&region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign", -1);
    memset(region, 0, region_sz);
    /* recv slot k at k*slot; send slot k at (K+k)*slot; each slot's seq at +chunk */
#define RECV_SEQ(k) ((volatile uint64_t *)((char *)region + (size_t)(k) * slot + chunk))
#define SEND_SEQ(k) ((uint64_t *)((char *)region + (size_t)(K + (k)) * slot + chunk))

    /* one VCQ per TNI; register the whole region in each (independent stadd space) */
    utofu_vcq_hdl_t vcq[MAX_TNI];
    utofu_vcq_id_t  my_vcq[MAX_TNI];
    utofu_stadd_t   base[MAX_TNI];
    utofu_vcq_id_t  peer_vcq[MAX_TNI];
    utofu_stadd_t   peer_base[MAX_TNI];
    for (int k = 0; k < K; k++) {
        utofu_tni_id_t tni = tni_ids[k];
        rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &vcq[k]);
        if (rc != UTOFU_SUCCESS) die("utofu_create_vcq_with_cmp_id", rc);
        rc = utofu_query_vcq_id(vcq[k], &my_vcq[k]);
        if (rc != UTOFU_SUCCESS) die("utofu_query_vcq_id", rc);
        rc = utofu_reg_mem_with_stag(vcq[k], region, region_sz, BENCH_STAG, 0, &base[k]);
        if (rc != UTOFU_SUCCESS) die("utofu_reg_mem_with_stag", rc);
        rc = utofu_construct_vcq_id(topo[peer], tni, DEMO_CQ_ID, DEMO_CMP_ID, &peer_vcq[k]);
        if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(peer)", rc);
        utofu_set_vcq_id_path(&peer_vcq[k], NULL);
        rc = utofu_query_stadd(peer_vcq[k], BENCH_STAG, &peer_base[k]);
        if (rc != UTOFU_SUCCESS) die("utofu_query_stadd(peer)", rc);
    }

    const unsigned long flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
    void *cb;
    size_t put_len = chunk + 8;   /* chunk payload + trailing seq, one Put per TNI */

    /* stripe token t to the peer: K concurrent Puts, then drain all K local TCQs */
#define STRIPE_SEND(t)                                                          \
    do {                                                                        \
        for (int k = 0; k < K; k++) {                                           \
            *SEND_SEQ(k) = (t);                                                 \
            utofu_stadd_t _s = base[k] + (size_t)(K + k) * slot;                \
            utofu_stadd_t _d = peer_base[k] + (size_t)(k) * slot;               \
            for (;;) {                                                          \
                rc = utofu_put(vcq[k], peer_vcq[k], _s, _d, put_len, 0, flags, NULL); \
                if (rc != UTOFU_ERR_BUSY) break;                               \
                utofu_poll_tcq(vcq[k], 0, &cb);                                 \
            }                                                                   \
            if (rc != UTOFU_SUCCESS) die("utofu_put", rc);                      \
        }                                                                       \
        for (int k = 0; k < K; k++) {                                           \
            do { rc = utofu_poll_tcq(vcq[k], 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND); \
            if (rc != UTOFU_SUCCESS) die("utofu_poll_tcq", rc);                 \
        }                                                                       \
    } while (0)

    /* wait until all K chunks of token t have landed */
#define STRIPE_WAIT(t)                                                          \
    do {                                                                        \
        double _ts = now_sec();                                                 \
        for (int k = 0; k < K; k++)                                             \
            while (*RECV_SEQ(k) < (t)) {                                        \
                if (now_sec() - _ts > WAIT_TIMEOUT_SEC) die("stripe wait timeout", -1); \
            }                                                                   \
    } while (0)

    /* bootstrap: both sides registered & reachable (token 1, barrierless) */
    {
        int got = 0;
        double t0 = now_sec();
        for (int a = 0; a < 400 && !got; a++) {
            STRIPE_SEND(1);
            int all = 1;
            for (int k = 0; k < K; k++) if (*RECV_SEQ(k) < 1) all = 0;
            if (all && a >= 8) got = 1;
            usleep(20000);
            if (now_sec() - t0 > WAIT_TIMEOUT_SEC) break;
        }
        if (!got) die("bootstrap timeout", -1);
    }

    /* ping-pong: rank 0 sends then waits; rank 1 waits then echoes. RTT/2 = hop. */
    uint64_t tok = 2;
#define PINGPONG(count, el)                                                     \
    do {                                                                        \
        double _t0 = now_sec();                                                 \
        for (long _i = 0; _i < (count); _i++) {                                 \
            uint64_t _w = tok++;                                                \
            if (my_rank == 0) { STRIPE_SEND(_w); STRIPE_WAIT(_w); }             \
            else              { STRIPE_WAIT(_w); STRIPE_SEND(_w); }             \
        }                                                                       \
        (el) = now_sec() - _t0;                                                 \
    } while (0)

    double warm = 0, timed = 0;
    PINGPONG(WARMUP, warm);
    PINGPONG(ITERS, timed);

    if (my_rank == 0) {
        double rtt_us = timed / ITERS * 1e6;
        double hop_us = rtt_us / 2.0;
        size_t total  = (size_t)K * chunk;          /* bytes actually moved/hop  */
        double bw     = (double)total / (hop_us * 1e-6) / 1e9;
        char buf[256];
        int n = snprintf(buf, sizeof buf,
            "TNI-stripe: K=%d  payload=%ld B  chunk/TNI=%zu B  moved=%zu B  per-hop=%.3f us  agg_BW=%.2f GB/s\n",
            K, P, chunk, total, hop_us, bw);
        fwrite(buf, 1, n, stdout); fflush(stdout);
        FILE *rf = fopen("ts_result.txt", "a");   /* mpiexec may swallow stdout */
        if (rf) { fwrite(buf, 1, n, rf); fclose(rf); }
    }

    for (int k = 0; k < K; k++) { utofu_dereg_mem(vcq[k], base[k], 0); utofu_free_vcq(vcq[k]); }
    free(tni_ids);
    free(region);
    return 0;
}
