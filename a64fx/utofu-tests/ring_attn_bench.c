/*
 * ring_attn_bench - MPI-free uTofu micro-benchmark estimating the cost of ONE
 * ring-attention DECODE step on A64FX/Fugaku. It models only the two costs that
 * dominate distributed decode attention; NO attention math / GEMM is performed.
 *
 *   1. Memory access: in sequence-parallel (ring/context-parallel) decode, the
 *      KV cache of length S is sharded across N nodes, so each node reads its
 *      own S/N-position KV shard once per decoded token. Attention decode is
 *      memory-bandwidth bound, so we just STREAM a buffer the size of that KV
 *      shard (multi-threaded) and report the achieved bandwidth and the implied
 *      per-step KV-read time.
 *
 *   2. uTofu comm: the single query token attends across all N shards; nodes
 *      combine their partial results (running max m, denominator l, and output
 *      accumulator o per query head) by passing that small reduction payload
 *      around the ring. We circulate a real payload of that size around the
 *      ring with uTofu Put and measure the per-hop latency, then report the
 *      (N-1)-hop ring-reduce cost.
 *
 * Like tofu_put_demo this binary makes ZERO MPI calls: mpiexec only places it,
 * and it reconstructs ring-neighbour VCQ IDs from coordinates in tofu_topo.txt
 * (written once by tofu_topo_helper) via utofu_construct_vcq_id().
 *
 * Build (NO -lmpi). Native A64FX node:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -Wall \
 *       -o ring_attn_bench ring_attn_bench.c -ltofucom
 *
 * Run (after tofu_topo_helper, inside a pjsub node=N alloc, 1 proc/node):
 *   OMP_NUM_THREADS=48 mpiexec -np <N> ./ring_attn_bench
 *
 * Tunables (env, defaults model a ~9B GQA decoder):
 *   RA_SEQ   total context length            (default 16384)
 *   RA_QH    query heads                      (default 32)
 *   RA_KVH   key/value heads (GQA)            (default 8)
 *   RA_HD    head dimension                   (default 128)
 *   RA_KVB   KV cache bytes/element (f16=2)   (default 2)
 *   RA_ITERS timed ring tokens                (default 2000)
 *   RA_WARMUP untimed warmup tokens           (default 200)
 *   RA_MEMGB GiB to stream for the mem bench  (default 8)
 */
#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <utofu.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "tofu_demo.h"

#define MAX_NODES 256
#define BENCH_STAG DEMO_STAG          /* reuse the predictable-STADD convention */
#define WAIT_TIMEOUT_SEC 15.0         /* spin-loop guard so we never hang */

/* MPI launchers redirect rank stdout, so every rank also logs to its own file. */
static FILE *g_log = NULL;

static void logmsg(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    if (g_log) { vfprintf(g_log, fmt, ap); fflush(g_log); }
    vfprintf(stdout, fmt, ap);
    fflush(stdout);
    va_end(ap);
}

static void die(const char *what, int rc)
{
    logmsg("FATAL: %s (rc=%d)\n", what, rc);
    exit(1);
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static long envl(const char *name, long def)
{
    const char *v = getenv(name);
    if (!v || !*v) return def;
    return strtol(v, NULL, 0);
}

/* Parse tofu_topo.txt into coords[][6]; returns node count. */
static int read_topo(uint8_t coords[][TOFU_NCOORDS])
{
    FILE *f = fopen(TOPO_PATH, "r");
    if (!f) {
        perror("cannot open " TOPO_PATH);
        fprintf(stderr, "  (run tofu_topo_helper first)\n");
        exit(1);
    }
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (n >= MAX_NODES) { fprintf(stderr, "too many nodes\n"); exit(1); }
        unsigned r, c[TOFU_NCOORDS];
        if (sscanf(line, "%u %u %u %u %u %u %u",
                   &r, &c[0], &c[1], &c[2], &c[3], &c[4], &c[5]) != 7) {
            fprintf(stderr, "malformed line in %s: %s", TOPO_PATH, line);
            exit(1);
        }
        if ((int)r != n) { fprintf(stderr, "%s ranks out of order\n", TOPO_PATH); exit(1); }
        for (int k = 0; k < TOFU_NCOORDS; k++) coords[n][k] = (uint8_t)c[k];
        n++;
    }
    fclose(f);
    if (n < 2) { fprintf(stderr, "%s lists %d node(s); need >= 2\n", TOPO_PATH, n); exit(1); }
    return n;
}

static volatile uint64_t g_sink;  /* defeats dead-code elimination of the stream */

/* Stream `n8` uint64 words of memory `iters` times with all OpenMP threads and
 * report GB/s. Integer accumulation (associative -> the compiler is free to
 * vectorize/reorder with SVE) keeps this memory-bandwidth bound, unlike an FP
 * reduction whose loop-carried dependency makes it FP-add-latency bound. Four
 * independent accumulators expose enough ILP to hide load latency. */
static double bench_memory_bw(const uint64_t *restrict buf, size_t n8, int iters)
{
    double t0 = now_sec();
    uint64_t acc = 0;
    for (int it = 0; it < iters; it++) {
        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : s0, s1, s2, s3) schedule(static)
#endif
        for (size_t i = 0; i < n8 - 3; i += 4) {
            s0 += buf[i + 0];
            s1 += buf[i + 1];
            s2 += buf[i + 2];
            s3 += buf[i + 3];
        }
        acc += s0 + s1 + s2 + s3;
    }
    double t1 = now_sec();
    g_sink = acc;
    double total_bytes = (double)n8 * sizeof(uint64_t) * (double)iters;
    return total_bytes / (t1 - t0) / 1e9;  /* GB/s (1e9-based) */
}

int main(void)
{
    int rc;

    /* ---- model / bench parameters ---- */
    long S      = envl("RA_SEQ", 16384);
    long QH     = envl("RA_QH", 32);
    long KVH    = envl("RA_KVH", 8);
    long HD     = envl("RA_HD", 128);
    long KVB    = envl("RA_KVB", 2);
    long ITERS  = envl("RA_ITERS", 2000);
    long WARMUP = envl("RA_WARMUP", 200);
    long MEMGB  = envl("RA_MEMGB", 8);

    /* ---- uTofu setup (identical conventions to tofu_put_demo) ---- */
    utofu_tni_id_t *tni_ids = NULL;
    size_t num_tnis = 0;
    rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
    if (rc != UTOFU_SUCCESS || num_tnis <= DEMO_TNI_INDEX) die("utofu_get_onesided_tnis", rc);
    utofu_tni_id_t tni = tni_ids[DEMO_TNI_INDEX];
    free(tni_ids);

    uint8_t my_coords[TOFU_NCOORDS] = {0};
    rc = utofu_query_my_coords(my_coords);
    if (rc != UTOFU_SUCCESS) die("utofu_query_my_coords", rc);

    {
        char name[64];
        snprintf(name, sizeof(name), "ra_log_%u_%u_%u_%u_%u_%u.txt",
                 my_coords[0], my_coords[1], my_coords[2],
                 my_coords[3], my_coords[4], my_coords[5]);
        g_log = fopen(name, "w");
    }

    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    int nprocs = read_topo(topo);
    int my_rank = -1;
    for (int r = 0; r < nprocs; r++)
        if (memcmp(topo[r], my_coords, TOFU_NCOORDS) == 0) my_rank = r;
    if (my_rank == -1) { fprintf(stderr, "my coords not in %s\n", TOPO_PATH); exit(1); }

    int next_rank = (my_rank + 1) % nprocs;          /* we Put to next */
    uint8_t next_coords[TOFU_NCOORDS];
    memcpy(next_coords, topo[next_rank], TOFU_NCOORDS);

    utofu_vcq_hdl_t vcq;
    rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &vcq);
    if (rc != UTOFU_SUCCESS) die("utofu_create_vcq_with_cmp_id", rc);
    utofu_vcq_id_t my_vcq_real;
    rc = utofu_query_vcq_id(vcq, &my_vcq_real);
    if (rc != UTOFU_SUCCESS) die("utofu_query_vcq_id", rc);

    /* VCQ self-check: prove the (tni, cq, cmp) convention reproduces our own
     * VCQ ID, so the same construction applied to a neighbour's coords is sound. */
    {
        utofu_vcq_id_t conv;
        rc = utofu_construct_vcq_id(my_coords, tni, DEMO_CQ_ID, DEMO_CMP_ID, &conv);
        if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(self)", rc);
        utofu_vcq_id_t a = my_vcq_real, b = conv;
        utofu_set_vcq_id_path(&a, NULL);
        utofu_set_vcq_id_path(&b, NULL);
        if (a != b) die("VCQ self-check (cq_id convention wrong)", -1);
    }

    /* ---- comm region: a recv slot and a send slot, each on its own cache
     * line so the CPU never holds the recv line dirty (see tofu_demo.h). Each
     * slot holds the decode reduction payload followed by an 8-byte sequence
     * counter; the receiver polls that counter (written last) to detect arrival. */
    size_t payload = (size_t)QH * (HD + 2) * sizeof(float); /* m,l + o[HD] per head */
    size_t p8      = (payload + 7) & ~(size_t)7;
    size_t slot    = (p8 + 8 + (DEMO_CACHE_LINE - 1)) & ~(size_t)(DEMO_CACHE_LINE - 1);
    size_t region_sz = 2 * slot;

    void *region = NULL;
    if (posix_memalign(&region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign", -1);
    memset(region, 0, region_sz);
    volatile uint64_t *recv_seq = (volatile uint64_t *)((char *)region + p8);
    uint64_t          *send_seq = (uint64_t *)((char *)region + slot + p8);

    utofu_stadd_t base_stadd;
    rc = utofu_reg_mem_with_stag(vcq, region, region_sz, BENCH_STAG, 0, &base_stadd);
    if (rc != UTOFU_SUCCESS) die("utofu_reg_mem_with_stag", rc);
    utofu_stadd_t send_stadd = base_stadd + slot;     /* our send slot */

    utofu_vcq_id_t next_vcq;
    rc = utofu_construct_vcq_id(next_coords, tni, DEMO_CQ_ID, DEMO_CMP_ID, &next_vcq);
    if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(next)", rc);
    utofu_set_vcq_id_path(&next_vcq, NULL);
    utofu_stadd_t next_base;
    rc = utofu_query_stadd(next_vcq, BENCH_STAG, &next_base);
    if (rc != UTOFU_SUCCESS) die("utofu_query_stadd(next)", rc);
    utofu_stadd_t next_recv_stadd = next_base;        /* neighbour's recv slot @ +0 */

    size_t put_len = p8 + 8;                          /* payload + seq counter */
    const unsigned long flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
    void *cb;

    if (my_rank == 0) {
        logmsg("=== ring-attention DECODE cost estimate (no GEMM) ===\n");
        logmsg("nodes=%d  seq S=%ld  pos/node=%ld  q_heads=%ld kv_heads=%ld head_dim=%ld kv_bytes=%ld\n",
               nprocs, S, (S + nprocs - 1) / nprocs, QH, KVH, HD, KVB);
        logmsg("ring payload = %zu B (q_heads*(head_dim+2)*4)\n", payload);
    }

    /* ---- helper: one ring Put of our send slot into the neighbour's recv ---- */
#define RING_PUT()                                                              \
    do {                                                                        \
        for (;;) {                                                              \
            rc = utofu_put(vcq, next_vcq, send_stadd, next_recv_stadd,           \
                           put_len, 0, flags, NULL);                            \
            if (rc != UTOFU_ERR_BUSY) break;                                    \
            utofu_poll_tcq(vcq, 0, &cb);                                        \
        }                                                                       \
        if (rc != UTOFU_SUCCESS) die("utofu_put", rc);                          \
        do { rc = utofu_poll_tcq(vcq, 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND); \
        if (rc != UTOFU_SUCCESS) die("utofu_poll_tcq", rc);                     \
    } while (0)

    /* ---- bootstrap: confirm the whole ring is registered & reachable.
     * Each node Puts seq=1 to its next neighbour until it has seen seq>=1 from
     * its previous neighbour AND issued a minimum number of Puts (covers the
     * barrierless startup race). After this, recv_seq==1 everywhere and the
     * serialized relay below never drops a token. ---- */
    {
        const int MIN_PUTS = 8;
        *send_seq = 1;
        int attempts = 0, got = 0;
        double t0 = now_sec();
        for (; attempts < 400; attempts++) {
            RING_PUT();
            if (*recv_seq >= 1) got = 1;
            if (got && attempts + 1 >= MIN_PUTS) break;
            usleep(20000);
            if (now_sec() - t0 > WAIT_TIMEOUT_SEC) break;
        }
        if (!got) die("ring bootstrap timeout", -1);
    }

    /* ---- ring relay: a single token circulates 0->1->...->N-1->0. Only one
     * message is in flight, so there are no recv-buffer overwrite races. Rank 0
     * injects token `tok`, every other rank waits for it and forwards it; the
     * token's return to rank 0 marks one full N-hop circulation. ---- */
    uint64_t tok = 2;  /* warmup+timed tokens use seq >= 2 (bootstrap used 1) */

    /* relay `count` tokens; rank 0 returns elapsed seconds, others return 0 */
#define RELAY(count, elapsed)                                                   \
    do {                                                                        \
        double _t0 = now_sec();                                                 \
        for (long _i = 0; _i < (count); _i++) {                                 \
            uint64_t _w = tok++;                                                \
            if (my_rank == 0) {                                                 \
                *send_seq = _w; RING_PUT();                                     \
                double _ts = now_sec();                                         \
                while (*recv_seq < _w) {                                        \
                    if (now_sec() - _ts > WAIT_TIMEOUT_SEC) die("relay timeout (rank0)", -1); \
                }                                                               \
            } else {                                                            \
                double _ts = now_sec();                                         \
                while (*recv_seq < _w) {                                        \
                    if (now_sec() - _ts > WAIT_TIMEOUT_SEC) die("relay timeout", -1); \
                }                                                               \
                *send_seq = _w; RING_PUT();                                     \
            }                                                                   \
        }                                                                       \
        (elapsed) = now_sec() - _t0;                                            \
    } while (0)

    double warm_el = 0.0, timed_el = 0.0;
    RELAY(WARMUP, warm_el);
    RELAY(ITERS, timed_el);

    /* ---- memory bench: stream a KV-shard-sized buffer with all threads ---- */
    long pos_per_node = (S + nprocs - 1) / nprocs;
    size_t kv_bytes = (size_t)pos_per_node * KVH * HD * 2 /*K&V*/ * KVB;
    size_t n8 = kv_bytes / sizeof(uint64_t);
    if (n8 < 4) n8 = 4;
    uint64_t *kvbuf = NULL;
    if (posix_memalign((void **)&kvbuf, DEMO_CACHE_LINE, n8 * sizeof(uint64_t)) != 0)
        die("posix_memalign(kv)", -1);
    /* Parallel first-touch with the SAME static schedule the bench uses, so on
     * A64FX's 4 CMGs (NUMA domains) each thread's pages are placed on its own
     * CMG -- otherwise single-thread init pins every page to one CMG and the
     * read streams from one memory controller (~1/4 the bandwidth). */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n8; i++) kvbuf[i] = i + 1;
    int mem_iters = (int)((double)MEMGB * (1024.0 * 1024.0 * 1024.0) / (double)kv_bytes);
    if (mem_iters < 5) mem_iters = 5;
    if (mem_iters > 5000) mem_iters = 5000;
    double meas_gbps = bench_memory_bw(kvbuf, n8, mem_iters);
    /* RA_BW_GBPS lets the user substitute a known/target read bandwidth (e.g. a
     * profiled ~645 GB/s) for the time estimate; 0 (default) uses the measured
     * streaming BW. Measured BW is NUMA-sensitive: launch with
     * `numactl --interleave=all` to spread pages across all 4 CMGs. */
    long bw_override = envl("RA_BW_GBPS", 0);
    double gbps = bw_override > 0 ? (double)bw_override : meas_gbps;
    double kv_read_us = (double)kv_bytes / (gbps * 1e9) * 1e6;

    int omp_threads = 1;
#ifdef _OPENMP
    omp_threads = omp_get_max_threads();
#endif

    /* every rank reports its own memory bandwidth */
    logmsg("rank %d: KV shard = %.2f MiB, mem BW = %.1f GB/s (%d threads), KV read/step = %.1f us\n",
           my_rank, (double)kv_bytes / (1024.0 * 1024.0), gbps, omp_threads, kv_read_us);

    /* rank 0 owns the ring-comm timing and prints the combined estimate */
    if (my_rank == 0) {
        double per_token_us = timed_el / (double)ITERS * 1e6;  /* N-hop round trip */
        double per_hop_us   = per_token_us / (double)nprocs;
        double ring_reduce_us = per_hop_us * (double)(nprocs - 1);
        double hop_bw = (double)payload / (per_hop_us * 1e-6) / 1e9;

        logmsg("\n--- uTofu ring comm (rank 0 timing, %ld tokens) ---\n", ITERS);
        logmsg("per-hop latency   = %.2f us  (payload BW = %.2f GB/s)\n", per_hop_us, hop_bw);
        logmsg("full circulation  = %.2f us  (%d hops)\n", per_token_us, nprocs);
        logmsg("ring-reduce       = %.2f us  (%d hops, the decode comm cost)\n",
               ring_reduce_us, nprocs - 1);

        logmsg("\n--- combined per-decode-step attention estimate ---\n");
        logmsg("KV read (per node) = %.1f us @ %.1f GB/s\n", kv_read_us, gbps);
        logmsg("ring-reduce comm   = %.2f us\n", ring_reduce_us);
        logmsg("serial  (mem+comm) = %.1f us  -> %.1f tok/s (attn-only upper bound)\n",
               kv_read_us + ring_reduce_us, 1e6 / (kv_read_us + ring_reduce_us));
        logmsg("overlap (max)      = %.1f us  -> %.1f tok/s\n",
               (kv_read_us > ring_reduce_us ? kv_read_us : ring_reduce_us),
               1e6 / (kv_read_us > ring_reduce_us ? kv_read_us : ring_reduce_us));
    }

    free(kvbuf);
    utofu_dereg_mem(vcq, base_stadd, 0);
    utofu_free_vcq(vcq);
    free(region);
    return 0;
}
