/*
 * ring_attn_bench - MPI-free uTofu micro-benchmark estimating the cost of ONE
 * ring-attention DECODE step on A64FX/Fugaku. It models only the two costs that
 * dominate distributed decode attention; NO attention math / GEMM is performed.
 *
 *   1. Memory access: in sequence-parallel (ring/context-parallel) decode, the
 *      KV cache of length S is sharded across N nodes, so each node reads its
 *      own S/N-position KV shard once per decoded token. Attention decode is
 *      memory-bandwidth bound, so we measure the node's full streaming read BW
 *      and divide the modeled shard size by it for the per-step KV-read time.
 *      Reaching peak A64FX BW (~200 GB/s x 4 CMG ~ 800 GB/s/node) requires each
 *      CMG to read NUMA-local memory: bench_node_bw gives every CMG its own
 *      buffer pinned to that CMG's node (mbind+MF_MOVE), pins threads, and reads
 *      with an 8-accumulator SVE kernel. (A single interleaved buffer or the
 *      naive 1-accumulator reduction reach only ~1/3 of peak.) The shard itself
 *      is too small to time directly -- it would sit in L2 -- but across a
 *      model's many layers the KV read is a genuine HBM stream.
 *
 *   2. uTofu comm: the single query token attends across all N shards; nodes
 *      combine their partial results (running max m, denominator l, and output
 *      accumulator o per query head) by passing that small reduction payload
 *      around the ring. We circulate a real payload of that size around the
 *      ring with uTofu Put and measure the per-hop latency, then report the
 *      (N-1)-hop ring-reduce cost. We ALSO run a real recursive-doubling tree
 *      all-reduce (Rabenseifner, non-power-of-2 aware) over the same payload and
 *      MEASURE its latency, to check the ceil(log2 N) projection against what
 *      the network actually delivers for the longer-distance XOR partners.
 *
 * From the single run's measured node BW + per-hop latency, rank 0 also prints
 * an N-sweep projection table: KV read (~1/N) vs linear ring-reduce ((N-1) hops)
 * vs recursive-doubling tree (ceil(log2 N) rounds), and the crossover N where
 * comm starts to dominate -- so one 2-node run shows the whole scaling story.
 *
 * The ring relay runs twice -- once over the default rank+1 order and once over
 * a topology-ordered ring that greedily minimises physical link traversals (a
 * near-Hamiltonian cycle on the torus) -- and rank 0 reports both, to test
 * whether fewer physical hops actually lowers the measured per-hop latency.
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
 *   RA_BWMB  per-CMG probe buffer, MiB        (default 512)
 *   RA_BW_GBPS override measured node BW      (default 0 = measure)
 */
#define _GNU_SOURCE
#include <arm_sve.h>
#include <assert.h>
#include <sched.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <utofu.h>
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_thread_num(void) { return 0; }
static int omp_get_max_threads(void) { return 1; }
#endif

#include "tofu_demo.h"

#define MAX_NODES 256
#define BENCH_STAG DEMO_STAG          /* reuse the predictable-STADD convention */
#define WAIT_TIMEOUT_SEC 15.0         /* spin-loop guard so we never hang */
#define TREE_NSTEP 24                 /* distinct recv slots for the tree all-reduce
                                       * (1 pre + log2(pof2) doubling + 1 bcast; 24
                                       * covers pof2 up to 2^22, i.e. any sane N) */

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

/* NUMA node owning logical CPU `cpu`, or -1. Parsed straight from
 * /sys/devices/system/node/nodeN/cpulist so we need no libnuma. */
static int node_of_cpu(int cpu)
{
    for (int nd = 0; nd < 16; nd++) {
        char path[64], buf[256];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", nd);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        char *got = fgets(buf, sizeof(buf), f);
        fclose(f);
        if (!got) continue;
        char *p = buf;
        while (*p && *p != '\n') {
            int a = (int)strtol(p, &p, 10), b = a;
            if (*p == '-') { p++; b = (int)strtol(p, &p, 10); }
            if (cpu >= a && cpu <= b) return nd;
            while (*p == ',') p++;
        }
    }
    return -1;
}

/* SVE 8-accumulator unrolled u64 sum over [lo,hi). hi-lo is a multiple of
 * 8*svcntd(), so no scalar tail. Eight independent vector accumulators expose
 * enough in-flight loads to saturate a CMG's HBM (the naive 1-accumulator
 * reduction is load-latency bound at ~1/3 of peak). */
static inline uint64_t sve_sum(const uint64_t *restrict b, size_t lo, size_t hi)
{
    svbool_t pg = svptrue_b64();
    uint64_t vl = svcntd();
    size_t step = vl * 8;
    svuint64_t a0=svdup_u64(0),a1=svdup_u64(0),a2=svdup_u64(0),a3=svdup_u64(0);
    svuint64_t a4=svdup_u64(0),a5=svdup_u64(0),a6=svdup_u64(0),a7=svdup_u64(0);
    for (size_t i = lo; i + step <= hi; i += step) {
        a0 = svadd_u64_x(pg, a0, svld1_u64(pg, &b[i + 0*vl]));
        a1 = svadd_u64_x(pg, a1, svld1_u64(pg, &b[i + 1*vl]));
        a2 = svadd_u64_x(pg, a2, svld1_u64(pg, &b[i + 2*vl]));
        a3 = svadd_u64_x(pg, a3, svld1_u64(pg, &b[i + 3*vl]));
        a4 = svadd_u64_x(pg, a4, svld1_u64(pg, &b[i + 4*vl]));
        a5 = svadd_u64_x(pg, a5, svld1_u64(pg, &b[i + 5*vl]));
        a6 = svadd_u64_x(pg, a6, svld1_u64(pg, &b[i + 6*vl]));
        a7 = svadd_u64_x(pg, a7, svld1_u64(pg, &b[i + 7*vl]));
    }
    a0 = svadd_u64_x(pg,a0,a1); a2 = svadd_u64_x(pg,a2,a3);
    a4 = svadd_u64_x(pg,a4,a5); a6 = svadd_u64_x(pg,a6,a7);
    a0 = svadd_u64_x(pg,a0,a2); a4 = svadd_u64_x(pg,a4,a6);
    return svaddv_u64(pg, svadd_u64_x(pg, a0, a4));
}

/* Measure aggregate node read bandwidth (GB/s) the way decode actually streams
 * the KV cache. Full A64FX node BW (~200 GB/s x 4 CMG) is only reached when each
 * CMG reads memory *local* to its own NUMA node: a single interleaved buffer
 * caps near 3/8 of peak on cross-CMG traffic. So we split the allowed cores into
 * per-CMG teams, give each CMG its own buffer pinned to that CMG's node, pin
 * each thread, and have every thread read only its CMG's buffer.
 *
 * Reports GB/s; *out_ncmg / *out_cores_per describe the detected layout. */
static double bench_node_bw(size_t per_cmg_mib, double stream_gib,
                            int *out_ncmg, int *out_cores_per)
{
    enum { CMG_CORES = 12 };          /* A64FX: 12 compute cores per CMG */
    cpu_set_t allowed;
    sched_getaffinity(0, sizeof(allowed), &allowed);
    int cores[256], ncore = 0;
    for (int c = 0; c < CPU_SETSIZE && ncore < 256; c++)
        if (CPU_ISSET(c, &allowed)) cores[ncore++] = c;
    int ncmg = ncore / CMG_CORES, cores_per = CMG_CORES;
    if (ncmg < 1) { ncmg = 1; cores_per = ncore < 1 ? 1 : ncore; }
    int nthr = ncmg * cores_per;
    *out_ncmg = ncmg;
    *out_cores_per = cores_per;
    if (ncore < 1) return 0.0;

    /* per-CMG buffer rounded to a multiple of cores_per*(8*svcntd()) so every
     * lane reads a whole number of unrolled SVE blocks (exact byte count). */
    size_t vl = svcntd(), blk = (size_t)cores_per * vl * 8;
    size_t n8 = per_cmg_mib * 1024 * 1024 / 8;
    n8 = (n8 / blk) * blk; if (n8 < blk) n8 = blk;
    size_t bytes = n8 * 8, lane_n8 = n8 / cores_per;

    uint64_t *buf[64] = {0};
    for (int c = 0; c < ncmg; c++) {
        if (posix_memalign((void **)&buf[c], 2*1024*1024, bytes) != 0) die("posix_memalign(bw)", -1);
        int node = node_of_cpu(cores[c * cores_per]);
        if (node >= 0) {
            unsigned long mask = 1UL << node;
            /* MPOL_BIND | MPOL_MF_MOVE: relocate the pages posix_memalign already
             * pre-faulted -- Fugaku huge pages fault in on the allocating node,
             * so without MF_MOVE every buffer would sit on one CMG (~1/4 BW). */
            if (syscall(SYS_mbind, buf[c], bytes, 2L /*MPOL_BIND*/, &mask,
                        (unsigned long)(8 * sizeof(mask)), 2U /*MPOL_MF_MOVE*/) != 0)
                logmsg("warning: mbind CMG %d -> node %d failed (BW may be low)\n", c, node);
        }
    }

    /* Report the BEST of several timed trials: this is a peak-BW probe, and on
     * a shared/interactive node (e.g. the rank running the launcher) background
     * activity steals BW in some windows -- the best trial catches a clean one,
     * giving the achievable ceiling rather than a contention-averaged number. */
    enum { NTRIAL = 6 };
    int iters = (int)(stream_gib * 1073741824.0 / ((double)NTRIAL * ncmg * bytes));
    if (iters < 4) iters = 4; if (iters > 4000) iters = 4000;

    double best_gbps = 0.0, t0 = 0, t1 = 0;
    uint64_t acc = 0;
#pragma omp parallel num_threads(nthr) reduction(+ : acc)
    {
        int t = omp_get_thread_num(), c = t / cores_per, lane = t % cores_per;
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(cores[t], &s);
        sched_setaffinity(0, sizeof(s), &s);
        size_t lo = lane_n8 * lane, hi = lo + lane_n8;
        for (size_t i = lo; i < hi; i++) buf[c][i] = i + 1;   /* local first-touch */
        uint64_t local = 0;
        /* warmup: fault TLB / settle the prefetchers before timing */
        for (int it = 0; it < 3; it++) local += sve_sum(buf[c], lo, hi);
        for (int trial = 0; trial < NTRIAL; trial++) {
#pragma omp barrier
#pragma omp master
            t0 = now_sec();
#pragma omp barrier
            for (int it = 0; it < iters; it++) local += sve_sum(buf[c], lo, hi);
#pragma omp barrier
#pragma omp master
            {
                t1 = now_sec();
                double g = (double)ncmg * bytes * (double)iters / (t1 - t0) / 1e9;
                if (g > best_gbps) best_gbps = g;
            }
        }
        acc += local;
    }
    g_sink = acc;
    for (int c = 0; c < ncmg; c++) free(buf[c]);
    return best_gbps;
}

/* Manhattan distance between two Tofu coord vectors on a torus whose per-axis
 * extent is ext[k] (axes with ext<=1 are fixed, contribute 0; wrap counts as
 * adjacent). Scores a ring ordering by the physical link traversals it costs. */
static int torus_dist(const uint8_t *a, const uint8_t *b, const int *ext)
{
    int d = 0;
    for (int k = 0; k < TOFU_NCOORDS; k++) {
        int dd = a[k] > b[k] ? a[k] - b[k] : b[k] - a[k];
        if (ext[k] > 1 && ext[k] - dd < dd) dd = ext[k] - dd;   /* torus wrap */
        d += dd;
    }
    return d;
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
    long RBYTES = envl("RA_RBYTES", 4);   /* reduce element size: 4=fp32, 2=bf16/fp16 */
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

    /* ---- comm region: slot 0 = ring recv, slot 1 = send staging (shared by the
     * ring relay and the tree all-reduce), slots 2..2+TREE_NSTEP-1 = one recv
     * slot per tree all-reduce step. Each slot is its own cache line so the CPU
     * never holds a recv line dirty (see tofu_demo.h), and holds the decode
     * reduction payload followed by an 8-byte sequence counter the receiver polls
     * (written last) to detect arrival. The tree uses a distinct recv slot per
     * step so a partner advancing to the next step can't overwrite an unread one. */
    size_t payload = (size_t)QH * (HD + 2) * (size_t)RBYTES; /* m,l + o[HD] per head */
    size_t p8      = (payload + 7) & ~(size_t)7;
    size_t slot    = (p8 + 8 + (DEMO_CACHE_LINE - 1)) & ~(size_t)(DEMO_CACHE_LINE - 1);
    size_t region_sz = (size_t)(2 + TREE_NSTEP) * slot;

    void *region = NULL;
    if (posix_memalign(&region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign", -1);
    memset(region, 0, region_sz);
    volatile uint64_t *recv_seq = (volatile uint64_t *)((char *)region + p8);
    uint64_t          *send_seq = (uint64_t *)((char *)region + slot + p8);
    /* tree step `sid` recv-counter lives in slot (2+sid) */
#define TREE_RECV_SEQ(sid) \
    ((volatile uint64_t *)((char *)region + (size_t)(2 + (sid)) * slot + p8))

    utofu_stadd_t base_stadd;
    rc = utofu_reg_mem_with_stag(vcq, region, region_sz, BENCH_STAG, 0, &base_stadd);
    if (rc != UTOFU_SUCCESS) die("utofu_reg_mem_with_stag", rc);
    utofu_stadd_t send_stadd = base_stadd + slot;     /* our send slot */

    /* VCQ id + region base for EVERY peer, so we can Put to any rank (the ring
     * successor in either ordering, and the tree all-reduce's XOR partners).
     * Built by convention from coords + STAG -- no runtime exchange. */
    static utofu_vcq_id_t peer_vcq[MAX_NODES];
    static utofu_stadd_t  peer_base[MAX_NODES];
    for (int r = 0; r < nprocs; r++) {
        if (r == my_rank) { peer_vcq[r] = my_vcq_real; peer_base[r] = base_stadd; continue; }
        rc = utofu_construct_vcq_id(topo[r], tni, DEMO_CQ_ID, DEMO_CMP_ID, &peer_vcq[r]);
        if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(peer)", rc);
        utofu_set_vcq_id_path(&peer_vcq[r], NULL);
        rc = utofu_query_stadd(peer_vcq[r], BENCH_STAG, &peer_base[r]);
        if (rc != UTOFU_SUCCESS) die("utofu_query_stadd(peer)", rc);
    }

    /* ---- ring orderings: the default rank+1 ring vs a topology-ordered ring
     * that greedily minimises physical link traversals (nearest unvisited node
     * by torus distance, starting from rank 0). Per-axis torus extent is taken
     * from the coords present in tofu_topo.txt. We relay over both and compare,
     * to see whether fewer physical hops actually lowers the measured latency or
     * whether (as for the ring's per-hop) it is software-bound and indifferent. */
    int ext[TOFU_NCOORDS];
    for (int k = 0; k < TOFU_NCOORDS; k++) {
        int lo = 255, hi = 0;
        for (int r = 0; r < nprocs; r++) {
            if (topo[r][k] < lo) lo = topo[r][k];
            if (topo[r][k] > hi) hi = topo[r][k];
        }
        ext[k] = hi - lo + 1;
    }
    static int order_opt[MAX_NODES];
    {
        char used[MAX_NODES] = {0};
        order_opt[0] = 0; used[0] = 1;
        for (int i = 1; i < nprocs; i++) {
            int cur = order_opt[i - 1], best = -1, bestd = 1 << 30;
            for (int r = 0; r < nprocs; r++) {
                if (used[r]) continue;
                int dd = torus_dist(topo[cur], topo[r], ext);
                if (dd < bestd) { bestd = dd; best = r; }
            }
            order_opt[i] = best; used[best] = 1;
        }
    }
    int succ_def = (my_rank + 1) % nprocs;            /* default ring successor */
    int succ_opt = succ_def;                          /* topo-ring successor */
    for (int i = 0; i < nprocs; i++)
        if (order_opt[i] == my_rank) { succ_opt = order_opt[(i + 1) % nprocs]; break; }
    int phys_def = 0, phys_opt = 0;                   /* physical links per ring loop */
    for (int i = 0; i < nprocs; i++) {
        phys_def += torus_dist(topo[i], topo[(i + 1) % nprocs], ext);
        phys_opt += torus_dist(topo[order_opt[i]], topo[order_opt[(i + 1) % nprocs]], ext);
    }

    /* current ring successor for RING_PUT; swapped between the two relay phases */
    utofu_vcq_id_t cur_vcq        = peer_vcq[succ_def];
    utofu_stadd_t  cur_recv_stadd = peer_base[succ_def];

    size_t put_len = p8 + 8;                          /* payload + seq counter */
    const unsigned long flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
    void *cb;

    if (my_rank == 0) {
        logmsg("=== ring-attention DECODE cost estimate (no GEMM) ===\n");
        logmsg("nodes=%d  seq S=%ld  pos/node=%ld  q_heads=%ld kv_heads=%ld head_dim=%ld kv_bytes=%ld\n",
               nprocs, S, (S + nprocs - 1) / nprocs, QH, KVH, HD, KVB);
        logmsg("ring payload = %zu B (q_heads*(head_dim+2)*%ld, reduce elem %s)\n",
               payload, RBYTES, RBYTES == 2 ? "bf16/fp16" : RBYTES == 4 ? "fp32" : "?");
    }

    /* ---- helper: one ring Put of our send slot into the neighbour's recv ---- */
#define RING_PUT()                                                              \
    do {                                                                        \
        for (;;) {                                                              \
            rc = utofu_put(vcq, cur_vcq, send_stadd, cur_recv_stadd,             \
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

    /* phase 1: default rank+1 ring (cur_* already point at succ_def) */
    double warm_el = 0.0, timed_def = 0.0;
    RELAY(WARMUP, warm_el);
    RELAY(ITERS, timed_def);

    /* phase 2: topology-ordered ring (same machinery, successor swapped) */
    double timed_opt = 0.0;
    cur_vcq = peer_vcq[succ_opt];
    cur_recv_stadd = peer_base[succ_opt];
    RELAY(WARMUP, warm_el);
    RELAY(ITERS, timed_opt);

    /* ---- tree all-reduce (recursive doubling, the comm pattern that actually
     * scales as ceil(log2 N) instead of the ring's N-1). Standard Rabenseifner
     * non-power-of-2 handling: the lowest 2*rem ranks pair up so the even one
     * folds into the odd one (pre-reduce), recursive doubling runs on the pof2
     * survivors, then results are sent back to the folded-out evens (broadcast).
     * We don't combine values (this benchmark measures comm cost only) but keep
     * the exact dependency chain -- each step waits for the prior step's arrival
     * before issuing the next -- so the measured latency is the true critical
     * path, including the longer-distance XOR partners. Every Put targets the
     * partner's recv slot for that step id; both partners of an exchange agree on
     * the slot id, so there is a unique writer per (slot,token). ---- */
    int pof2 = 1; while (pof2 * 2 <= nprocs) pof2 *= 2;
    int rem = nprocs - pof2;
    int nrounds = 0; for (int x = 1; x < pof2; x <<= 1) nrounds++;   /* log2(pof2) */
    int bcast_sid = nrounds + 1;
    int cl2 = 0; for (int x = 1; x < nprocs; x <<= 1) cl2++;         /* ceil(log2 N) */
    int newrank;                              /* rank within the pof2 doubling group */
    if (my_rank < 2 * rem) newrank = (my_rank % 2 == 0) ? -1 : my_rank / 2;
    else                   newrank = my_rank - rem;

    /* Put our send slot into peer `prank`'s recv slot for step `sid`. */
#define TREE_PEER_PUT(prank, sid)                                               \
    do {                                                                        \
        utofu_stadd_t _tgt = peer_base[prank] + (size_t)(2 + (sid)) * slot;     \
        for (;;) {                                                              \
            rc = utofu_put(vcq, peer_vcq[prank], send_stadd, _tgt,              \
                           put_len, 0, flags, NULL);                            \
            if (rc != UTOFU_ERR_BUSY) break;                                    \
            utofu_poll_tcq(vcq, 0, &cb);                                        \
        }                                                                       \
        if (rc != UTOFU_SUCCESS) die("utofu_put(tree)", rc);                    \
        do { rc = utofu_poll_tcq(vcq, 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND); \
        if (rc != UTOFU_SUCCESS) die("utofu_poll_tcq(tree)", rc);               \
    } while (0)

    /* spin until step `sid`'s recv counter reaches token `w` */
#define TREE_WAIT(sid, w)                                                       \
    do {                                                                        \
        volatile uint64_t *_rs = TREE_RECV_SEQ(sid);                            \
        double _ts = now_sec();                                                 \
        while (*_rs < (uint64_t)(w))                                            \
            if (now_sec() - _ts > WAIT_TIMEOUT_SEC) {                           \
                logmsg("rank %d: tree wait timeout sid=%d want=%lu got=%lu "    \
                       "(slot off=%zu)\n", my_rank, (sid), (uint64_t)(w),       \
                       (uint64_t)*_rs, (size_t)(2 + (sid)) * slot + p8);        \
                exit(1);                                                        \
            }                                                                   \
    } while (0)

    /* one full all-reduce of token seq `w`: pre-reduce, doubling, broadcast */
#define TREE_ALLREDUCE(w)                                                       \
    do {                                                                        \
        *send_seq = (uint64_t)(w);                                              \
        if (my_rank < 2 * rem) {                  /* pre-reduce fold */         \
            if (my_rank % 2 == 0) TREE_PEER_PUT(my_rank + 1, 0);                \
            else                  TREE_WAIT(0, w);                              \
        }                                                                       \
        if (newrank != -1) {                      /* recursive doubling */      \
            for (int _k = 0; _k < nrounds; _k++) {                             \
                int _pnr = newrank ^ (1 << _k);                                \
                int _pr  = (_pnr < rem) ? (_pnr * 2 + 1) : (_pnr + rem);        \
                TREE_PEER_PUT(_pr, _k + 1);                                     \
                TREE_WAIT(_k + 1, w);                                          \
            }                                                                   \
        }                                                                       \
        if (my_rank < 2 * rem) {                  /* broadcast back */          \
            if (my_rank % 2 == 0) TREE_WAIT(bcast_sid, w);                      \
            else                  TREE_PEER_PUT(my_rank - 1, bcast_sid);        \
        }                                                                       \
    } while (0)

    double tree_warm = 0.0, tree_timed = 0.0;
    /* NB: pass a plain variable to TREE_ALLREDUCE -- the macro evaluates its
     * argument many times, so `tok++` here would increment repeatedly. */
    { double _t0 = now_sec(); for (long i = 0; i < WARMUP; i++) { uint64_t w = tok++; TREE_ALLREDUCE(w); } tree_warm  = now_sec() - _t0; }
    { double _t0 = now_sec(); for (long i = 0; i < ITERS;  i++) { uint64_t w = tok++; TREE_ALLREDUCE(w); } tree_timed = now_sec() - _t0; }
    (void)tree_warm;

    /* ---- memory bench: measure full node read BW the way decode streams the
     * KV cache (per-CMG-local buffers; see bench_node_bw), then apply it to the
     * modeled per-node KV-shard size. The shard itself (tens of MiB) is too
     * small to time BW reliably -- it would sit in L2 -- but across a model's
     * many layers the KV read is a genuine HBM stream, so we measure steady BW
     * on large per-CMG buffers and divide. ---- */
    long pos_per_node = (S + nprocs - 1) / nprocs;
    size_t kv_bytes = (size_t)pos_per_node * KVH * HD * 2 /*K&V*/ * KVB;

    long bwmib = envl("RA_BWMB", 512);          /* per-CMG probe buffer (MiB) */
    int ncmg = 1, cores_per = 1;
    double meas_gbps = bench_node_bw((size_t)bwmib, (double)MEMGB, &ncmg, &cores_per);
    /* RA_BW_GBPS substitutes a known/target read BW for the time estimate; 0
     * (default) uses the measured per-CMG-local streaming BW. */
    long bw_override = envl("RA_BW_GBPS", 0);
    double gbps = bw_override > 0 ? (double)bw_override : meas_gbps;
    double kv_read_us = (double)kv_bytes / (gbps * 1e9) * 1e6;

    /* every rank reports its own node bandwidth */
    logmsg("rank %d: KV shard = %.2f MiB, node read BW = %.1f GB/s "
           "(%d CMG x %d cores, %ld MiB/CMG probe), KV read/step = %.1f us\n",
           my_rank, (double)kv_bytes / (1024.0 * 1024.0), gbps,
           ncmg, cores_per, bwmib, kv_read_us);

    /* rank 0 owns the ring-comm timing and prints the combined estimate */
    if (my_rank == 0) {
        double per_token_us = timed_def / (double)ITERS * 1e6;  /* N-hop round trip */
        double per_hop_us   = per_token_us / (double)nprocs;
        double ring_reduce_us = per_hop_us * (double)(nprocs - 1);
        double hop_bw = (double)payload / (per_hop_us * 1e-6) / 1e9;

        logmsg("\n--- uTofu ring comm (rank 0 timing, %ld tokens) ---\n", ITERS);
        logmsg("per-hop latency   = %.2f us  (payload BW = %.2f GB/s)\n", per_hop_us, hop_bw);
        logmsg("full circulation  = %.2f us  (%d hops)\n", per_token_us, nprocs);
        logmsg("ring-reduce       = %.2f us  (%d hops, the decode comm cost)\n",
               ring_reduce_us, nprocs - 1);

        /* topology-ordered ring vs default rank+1: does minimising physical link
         * traversals lower the measured per-hop latency, or is it sw-bound? */
        double per_token_opt = timed_opt / (double)ITERS * 1e6;
        logmsg("\n--- topology-ordered ring vs default rank+1 ---\n");
        logmsg("default rank+1 : %2d physical links (%.2f/hop), circ %.2f us, %.3f us/hop\n",
               phys_def, (double)phys_def / nprocs, per_token_us, per_hop_us);
        logmsg("topo-ordered   : %2d physical links (%.2f/hop), circ %.2f us, %.3f us/hop\n",
               phys_opt, (double)phys_opt / nprocs, per_token_opt, per_token_opt / nprocs);
        logmsg("topo ring saves %d physical links; measured latency change = %+.1f%%\n",
               phys_def - phys_opt, (per_token_opt / per_token_us - 1.0) * 100.0);
        if (nprocs <= 24) {
            logmsg("topo order     :");
            for (int i = 0; i < nprocs; i++) logmsg(" %d", order_opt[i]);
            logmsg("\n");
        }

        double tree_us = tree_timed / (double)ITERS * 1e6;
        int tree_steps = nrounds + (rem > 0 ? 2 : 0);
        logmsg("\n--- uTofu tree all-reduce (recursive doubling, MEASURED) ---\n");
        logmsg("measured all-reduce = %.2f us  (%d steps: %d doubling%s)\n",
               tree_us, tree_steps, nrounds, rem > 0 ? " + pre-reduce + broadcast" : "");
        logmsg("vs linear ring      = %.2f us (%d hops); ceil(log2 N)*hop proj = %.2f us\n",
               ring_reduce_us, nprocs - 1, (double)cl2 * per_hop_us);
        if (nprocs > 2)
            logmsg("tree speedup over linear ring = %.2fx (per-step %.2f us)\n",
                   ring_reduce_us / tree_us, tree_us / (double)tree_steps);

        logmsg("\n--- combined per-decode-step attention estimate ---\n");
        logmsg("KV read (per node) = %.1f us @ %.1f GB/s\n", kv_read_us, gbps);
        logmsg("ring-reduce comm   = %.2f us\n", ring_reduce_us);
        logmsg("serial  (mem+comm) = %.1f us  -> %.1f tok/s (attn-only upper bound)\n",
               kv_read_us + ring_reduce_us, 1e6 / (kv_read_us + ring_reduce_us));
        logmsg("overlap (max)      = %.1f us  -> %.1f tok/s\n",
               (kv_read_us > ring_reduce_us ? kv_read_us : ring_reduce_us),
               1e6 / (kv_read_us > ring_reduce_us ? kv_read_us : ring_reduce_us));

        /* ---- N-sweep projection: how the two costs scale with ring size, from
         * this single run's measured node BW and per-hop latency. KV read/node
         * falls as ~1/N (shard = S/N positions); ring-reduce comm grows as
         * (N-1) hops (linear, reduce-to-one -- same model as the line above),
         * while a recursive-doubling all-reduce needs log2(pof2) doubling rounds
         * plus 2 fold/broadcast steps when N is not a power of 2 (the actual
         * Rabenseifner step depth -- the MEASURED tree section above confirms
         * this count, and that distant XOR partners run a bit above per_hop_us,
         * so the tree column is a lower bound). per_hop_us is the measured
         * single-physical-hop cost. The crossover N is where comm starts to
         * dominate the (overlapped, linear-ring) per-step cost. ---- */
        static const int Nsweep[] = {2,3,4,6,8,12,16,24,32,48,64,96,128};
        logmsg("\n--- scaling projection (S=%ld, BW=%.1f GB/s, hop=%.2f us) ---\n",
               S, gbps, per_hop_us);
        logmsg("   N  pos/node  KV_read(us)  ring_lin(us)  tree(us)  step_max(us)  tok/s\n");
        for (size_t k = 0; k < sizeof(Nsweep) / sizeof(Nsweep[0]); k++) {
            int N = Nsweep[k];
            long pos = (S + N - 1) / N;
            double kvus = (double)pos * KVH * HD * 2 * KVB / (gbps * 1e9) * 1e6;
            double lin  = (double)(N - 1) * per_hop_us;
            int p2 = 1; while (p2 * 2 <= N) p2 *= 2;          /* largest pow2 <= N */
            int depth = 0; for (int x = 1; x < p2; x <<= 1) depth++;  /* log2(pof2) */
            if (N - p2 > 0) depth += 2;                       /* pre-reduce + bcast */
            double tree = (double)depth * per_hop_us;
            double step = kvus > lin ? kvus : lin;                   /* overlapped */
            logmsg("%4d  %8ld  %11.2f  %12.2f  %8.2f  %12.2f  %7.1f\n",
                   N, pos, kvus, lin, tree, step, 1e6 / step);
        }
        int cross = 0;
        for (int N = 2; N <= 4096; N++) {
            long pos = (S + N - 1) / N;
            double kvus = (double)pos * KVH * HD * 2 * KVB / (gbps * 1e9) * 1e6;
            if ((double)(N - 1) * per_hop_us > kvus) { cross = N; break; }
        }
        if (cross)
            logmsg("crossover: linear ring-reduce overtakes KV read at N=%d "
                   "(beyond it comm dominates -- use tree-reduce or fewer/larger shards)\n", cross);
        else
            logmsg("crossover: KV read dominates for all N<=4096 at this context\n");
    }

    utofu_dereg_mem(vcq, base_stadd, 0);
    utofu_free_vcq(vcq);
    free(region);
    return 0;
}
