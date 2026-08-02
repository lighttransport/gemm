/*
 * tp_ar_diag_bench - decompose the DS4F decode all-reduce in-loop cost.
 *
 * WHY: DS4F decode reports comm = 43 all-reduces/token * ~300 us each (DS4F_PROF
 * "comm" phase), yet the SAME tp_allreduce.h recursive-doubling tree measures
 * ~23.5 us standalone at N=12 (summary.md finding #3). The 13x gap is NOT
 * topology (already a tree) and NOT payload (bf16 was refuted at -2.5% and flips
 * argmax). This bench attributes the in-loop cost to four suspects so the fix is
 * chosen from data, not assumed:
 *
 *   (1) warm FLOOR   - tight loop, nothing between reduces. The algorithm cost.
 *   (2) COLD cache   - evict the comm region + trailer lines between reduces
 *                      (decode interleaves heavy matvec/attn/expert compute that
 *                      evicts them). Penalty = cold - floor.
 *   (3) ROBUST       - tp_comm_init reads TP_AR_ROBUST (default 1): drain MRQ +
 *                      `dc civac`+`dsb sy` on EVERY spin. Run this binary with
 *                      TP_AR_ROBUST=0 and =1 and diff every row (robust only
 *                      bites while the spin actually iterates -> see it under skew).
 *   (4) SKEW         - inject a per-rank busy-wait before each reduce (one slow
 *                      rank). The per-layer MoE-combine all-reduce is a BARRIER,
 *                      and ep_ar_callback's timer wraps tp_ar_wait's spin, so
 *                      expert load-imbalance is COUNTED AS COMM. Fit
 *                      reduce_us ~= a + b*skew_max; b~=1 => the 300us is
 *                      straggler-sync, not comm latency (the decision gate).
 *
 * Pure uTofu, NO MPI, NO OpenMP. Bootstrap copied from pp_handoff_bench.c
 * (read_topo -> construct_vcq_id -> reg_mem_with_stag under DEMO_STAG for the
 * barrier region; tp_comm registers its OWN region under TP_AR_STAG=7, no clash).
 *
 * Build (native A64FX node):
 *   make tp_ar_diag_bench CC=fcc        # or:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall \
 *       -o tp_ar_diag_bench tp_ar_diag_bench.c -ltofucom
 *
 * Run (after tofu_topo_helper, 1 proc/node):
 *   mpiexec -np 4 ./tp_ar_diag_bench                       # robust default (1)
 *   TP_AR_ROBUST=0 mpiexec -np 4 ./tp_ar_diag_bench        # robust off A/B
 * stdout is swallowed by mpiexec -> read rank-0's tp_ar_diag_<coords>.txt.
 *
 * Tunables (env):
 *   DIAG_COUNT  floats per reduce      (default 4096 = 16 KiB, the DS4F hidden)
 *   DIAG_ITERS  timed reduces/phase    (default 2000)
 *   DIAG_WARMUP untimed warmup reduces (default 200)
 *   DIAG_EVICT  cold-cache evict bytes (default 33554432 = 32 MiB; 0 disables)
 *   DIAG_SKEWRK rank that runs slow    (default N-1)
 *   plus tp_allreduce.h's TP_AR_ROBUST (default 1), TP_AR_BF16 (default 0).
 */
#define _GNU_SOURCE
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <utofu.h>

#include "tofu_demo.h"
#include "tp_allreduce.h"

#define MAX_NODES 32
#define BENCH_STAG DEMO_STAG          /* barrier region stag (1); tp_comm uses TP_AR_STAG=7 */
#define WAIT_TIMEOUT_SEC 30.0

/* ----- helpers (pp_handoff_bench.c conventions) ----- */
static FILE *g_log = NULL;
static void logmsg(const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    if (g_log) { vfprintf(g_log, fmt, ap); fflush(g_log); }
    vfprintf(stdout, fmt, ap); fflush(stdout);
    va_end(ap);
}
static void die(const char *what, int rc) { logmsg("FATAL: %s (rc=%d)\n", what, rc); exit(1); }
static double now_sec(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static long envl(const char *n, long d) { const char *v = getenv(n); return (v && *v) ? strtol(v, NULL, 0) : d; }

static int read_topo(uint8_t coords[][TOFU_NCOORDS])
{
    FILE *f = fopen(TOPO_PATH, "r");
    if (!f) { perror("cannot open " TOPO_PATH); fprintf(stderr, "  (run tofu_topo_helper first)\n"); exit(1); }
    int n = 0; char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (n >= MAX_NODES) { fprintf(stderr, "too many nodes\n"); exit(1); }
        unsigned r, c[TOFU_NCOORDS];
        if (sscanf(line, "%u %u %u %u %u %u %u", &r, &c[0], &c[1], &c[2], &c[3], &c[4], &c[5]) != 7)
            { fprintf(stderr, "malformed line: %s", line); exit(1); }
        if ((int)r != n) { fprintf(stderr, "%s ranks out of order\n", TOPO_PATH); exit(1); }
        for (int k = 0; k < TOFU_NCOORDS; k++) coords[n][k] = (uint8_t)c[k];
        n++;
    }
    fclose(f);
    if (n < 2) { fprintf(stderr, "%s lists %d node(s); need >= 2\n", TOPO_PATH, n); exit(1); }
    return n;
}

/* ----- file-scope state (single-threaded bench) ----- */
static int             N, MyRank;
static char           *Bar;                  /* barrier region (DEMO_STAG) */
static utofu_vcq_hdl_t Vcq;
static utofu_stadd_t   BarBase;
static utofu_vcq_id_t  PeerVcq[MAX_NODES];
static utofu_stadd_t   PeerBarBase[MAX_NODES];
static const unsigned long FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
static uint64_t        Bt = 1;               /* monotonic barrier token */

/* barrier region layout: per-child fan-in slot + one release slot, each its own line. */
static size_t BarSlot;                        /* DEMO_CACHE_LINE */
static inline size_t bar_send_off(void)    { return 0; }
static inline size_t bar_recv_off(int s)   { return BarSlot + (size_t)s * BarSlot; }
static inline size_t bar_go_off(void)      { return BarSlot + (size_t)N * BarSlot; }

static void put_issue(utofu_vcq_id_t pv, utofu_stadd_t s, utofu_stadd_t d, size_t len)
{
    int rc; void *cb;
    for (;;) { rc = utofu_put(Vcq, pv, s, d, len, 0, FLAGS, NULL);
               if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(Vcq, 0, &cb); }
    if (rc != UTOFU_SUCCESS) die("utofu_put", rc);
    do { rc = utofu_poll_tcq(Vcq, 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND);
    if (rc != UTOFU_SUCCESS) die("utofu_poll_tcq", rc);
    /* drain receiver-side RMT_PUT notices so the barrier never overflows the MRQ */
    { struct utofu_mrq_notice nt; while (utofu_poll_mrq(Vcq, 0, &nt) == UTOFU_SUCCESS) {} }
}
static void wait_seq(volatile uint64_t *q, uint64_t v, const char *what)
{
    double ts = now_sec();
    while (*q < v) if (now_sec() - ts > WAIT_TIMEOUT_SEC) die(what, -1);
}

/* fan-in to rank 0 then fan-out release. robust==1 retries (bootstrap); ==0 tight spin. */
static void barrier_robust(int robust)
{
    uint64_t t = ++Bt;
    if (MyRank == 0) {
        for (int s = 1; s < N; s++)
            wait_seq((volatile uint64_t *)(Bar + bar_recv_off(s)), t, "barrier fan-in");
        for (int s = 1; s < N; s++) {
            *(volatile uint64_t *)(Bar + bar_send_off()) = t;
            put_issue(PeerVcq[s], BarBase + bar_send_off(), PeerBarBase[s] + bar_go_off(), 8);
        }
    } else {
        volatile uint64_t *go = (volatile uint64_t *)(Bar + bar_go_off());
        double ts = now_sec();
        do {
            *(volatile uint64_t *)(Bar + bar_send_off()) = t;
            put_issue(PeerVcq[0], BarBase + bar_send_off(), PeerBarBase[0] + bar_recv_off(MyRank), 8);
            if (!robust) { wait_seq(go, t, "barrier release"); break; }
            for (int a = 0; a < 50 && *go < t; a++) usleep(2000);
            if (now_sec() - ts > WAIT_TIMEOUT_SEC) die("bootstrap barrier timeout", -1);
        } while (*go < t);
    }
}
static void barrier0(void) { barrier_robust(0); }   /* tp_comm_init's barrier_fn */

/* spin-busy-wait for `us` microseconds (the injected straggler delay). */
static void busy_us(double us)
{
    if (us <= 0) return;
    double t0 = now_sec(), end = t0 + us * 1e-6;
    volatile double x = 0;
    while (now_sec() < end) x += 1.0;
    (void)x;
}

int main(void)
{
    int rc;
    long COUNT  = envl("DIAG_COUNT", 4096);          /* 16 KiB f32, the DS4F hidden */
    long ITERS  = envl("DIAG_ITERS", 2000);
    long WARMUP = envl("DIAG_WARMUP", 200);
    long EVICTB = envl("DIAG_EVICT", 32L * 1024 * 1024);

    /* ---- uTofu setup (tofu_put_demo / pp_handoff conventions) ---- */
    utofu_tni_id_t *tni_ids = NULL; size_t num_tnis = 0;
    rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
    if (rc != UTOFU_SUCCESS) die("utofu_get_onesided_tnis", rc);
    if (num_tnis < 1) die("no onesided TNIs", -1);

    uint8_t my_coords[TOFU_NCOORDS] = {0};
    rc = utofu_query_my_coords(my_coords);
    if (rc != UTOFU_SUCCESS) die("utofu_query_my_coords", rc);
    {
        char name[64];
        snprintf(name, sizeof name, "tp_ar_diag_%u_%u_%u_%u_%u_%u.txt",
                 my_coords[0], my_coords[1], my_coords[2], my_coords[3], my_coords[4], my_coords[5]);
        g_log = fopen(name, "w");
    }

    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    N = read_topo(topo);
    MyRank = -1;
    for (int r = 0; r < N; r++) if (memcmp(topo[r], my_coords, TOFU_NCOORDS) == 0) MyRank = r;
    if (MyRank == -1) { fprintf(stderr, "my coords not in %s\n", TOPO_PATH); exit(1); }

    long SKEWRK = envl("DIAG_SKEWRK", N - 1);

    /* ---- barrier region (DEMO_STAG): N fan-in slots + 1 release ---- */
    BarSlot = DEMO_CACHE_LINE;
    size_t bar_sz = (size_t)(N + 2) * BarSlot;
    if (posix_memalign((void **)&Bar, DEMO_CACHE_LINE, bar_sz) != 0) die("posix_memalign(bar)", -1);
    memset(Bar, 0, bar_sz);

    utofu_tni_id_t tni = tni_ids[0];
    rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &Vcq);
    if (rc != UTOFU_SUCCESS) die("utofu_create_vcq_with_cmp_id", rc);
    utofu_vcq_id_t my_real;
    rc = utofu_query_vcq_id(Vcq, &my_real);
    if (rc != UTOFU_SUCCESS) die("utofu_query_vcq_id", rc);
    {                                              /* VCQ self-check (cq_id convention) */
        utofu_vcq_id_t conv;
        rc = utofu_construct_vcq_id(my_coords, tni, DEMO_CQ_ID, DEMO_CMP_ID, &conv);
        if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(self)", rc);
        utofu_vcq_id_t a = my_real, b = conv;
        utofu_set_vcq_id_path(&a, NULL); utofu_set_vcq_id_path(&b, NULL);
        if (a != b) die("VCQ self-check (cq_id convention wrong)", -1);
    }
    rc = utofu_reg_mem_with_stag(Vcq, Bar, bar_sz, BENCH_STAG, 0, &BarBase);
    if (rc != UTOFU_SUCCESS) die("utofu_reg_mem_with_stag(bar)", rc);
    for (int r = 0; r < N; r++) {
        if (r == MyRank) { PeerVcq[r] = my_real; PeerBarBase[r] = BarBase; continue; }
        rc = utofu_construct_vcq_id(topo[r], tni, DEMO_CQ_ID, DEMO_CMP_ID, &PeerVcq[r]);
        if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(peer)", rc);
        utofu_set_vcq_id_path(&PeerVcq[r], NULL);
        rc = utofu_query_stadd(PeerVcq[r], BENCH_STAG, &PeerBarBase[r]);
        if (rc != UTOFU_SUCCESS) die("utofu_query_stadd(peer bar)", rc);
    }
    free(tni_ids);

    barrier_robust(1);   /* robust bootstrap: every rank past barrier-region setup */

    /* ---- the all-reduce communicator (registers its own TP_AR_STAG region) ---- */
    tp_comm c;
    if (tp_comm_init(&c, Vcq, PeerVcq, MyRank, N, (int)COUNT, barrier0) != 0)
        die("tp_comm_init", -1);

    float *buf   = (float *)malloc((size_t)COUNT * sizeof(float));
    char  *evict = (EVICTB > 0) ? (char *)malloc((size_t)EVICTB) : NULL;
    if (!buf) die("malloc(buf)", -1);
    if (EVICTB > 0 && !evict) die("malloc(evict)", -1);
    if (evict) memset(evict, 1, (size_t)EVICTB);

    /* reset buf to all-ones; after a sum-reduce every rank must hold N. */
    #define RESET_BUF() do { for (long i = 0; i < COUNT; i++) buf[i] = 1.0f; } while (0)
    /* stream the evict buffer (read+write) so it displaces the comm region+trailers. */
    #define EVICT() do { if (evict) { for (long i = 0; i < EVICTB; i += DEMO_CACHE_LINE) evict[i]++; } } while (0)

    /* one-time correctness + lockstep check */
    RESET_BUF();
    tp_allreduce_sum(&c, buf, (int)COUNT);
    {
        int bad = 0; for (long i = 0; i < COUNT; i++) if (buf[i] != (float)N) { bad = 1; break; }
        if (bad) die("sum-reduce != N (correctness)", -1);
    }
    barrier0();

    if (MyRank == 0) {
        logmsg("=== tp_allreduce in-loop diagnostic ===\n");
        logmsg("N=%d  count=%ld (%.1f KiB f32)  iters=%ld warmup=%ld  evict=%.0f MiB  robust=%d payload=%s\n",
               N, COUNT, COUNT * 4 / 1024.0, ITERS, WARMUP, EVICTB / 1048576.0,
               c.robust, c.use_bf16 ? "bf16" : "fp32");
        logmsg("skew slow-rank=%ld (rank0 measures)\n\n", SKEWRK);
    }

    /* ---------- (1) WARM FLOOR ---------- */
    for (long i = 0; i < WARMUP; i++) { RESET_BUF(); tp_allreduce_sum(&c, buf, (int)COUNT); }
    double t0 = now_sec();
    for (long i = 0; i < ITERS; i++) { RESET_BUF(); tp_allreduce_sum(&c, buf, (int)COUNT); }
    double floor_us = (now_sec() - t0) / (double)ITERS * 1e6;
    /* RESET_BUF cost alone (subtract to isolate the reduce) */
    t0 = now_sec();
    for (long i = 0; i < ITERS; i++) { RESET_BUF(); }
    double reset_us = (now_sec() - t0) / (double)ITERS * 1e6;
    barrier0();

    /* ---------- (2) COLD CACHE ---------- */
    double cold_us = floor_us, evict_only_us = 0;
    if (evict) {
        for (long i = 0; i < WARMUP; i++) { RESET_BUF(); EVICT(); tp_allreduce_sum(&c, buf, (int)COUNT); }
        t0 = now_sec();
        for (long i = 0; i < ITERS; i++) { RESET_BUF(); EVICT(); tp_allreduce_sum(&c, buf, (int)COUNT); }
        double total = (now_sec() - t0) / (double)ITERS * 1e6;
        /* reset+evict cost without the reduce */
        t0 = now_sec();
        for (long i = 0; i < ITERS; i++) { RESET_BUF(); EVICT(); }
        evict_only_us = (now_sec() - t0) / (double)ITERS * 1e6;
        cold_us = total - evict_only_us;   /* reduce cost with cold comm region */
        barrier0();
    }

    /* ---------- (4) SKEW SWEEP ---------- */
    static const double SKEW[] = {0, 50, 100, 200, 400};
    const int NSKEW = (int)(sizeof SKEW / sizeof SKEW[0]);
    double skew_reduce_us[8];
    for (int s = 0; s < NSKEW; s++) {
        double my_skew = (MyRank == SKEWRK) ? SKEW[s] : 0.0;
        for (long i = 0; i < WARMUP; i++) { RESET_BUF(); busy_us(my_skew); tp_allreduce_sum(&c, buf, (int)COUNT); }
        /* rank0 times only the reduce (busy_us is outside the bracket on each iter). */
        double acc = 0;
        for (long i = 0; i < ITERS; i++) {
            RESET_BUF();
            busy_us(my_skew);
            double r0 = now_sec();
            tp_allreduce_sum(&c, buf, (int)COUNT);
            acc += now_sec() - r0;
        }
        skew_reduce_us[s] = acc / (double)ITERS * 1e6;
        barrier0();
    }

    if (MyRank == 0) {
        logmsg("-- decomposition (us per reduce) --\n");
        logmsg("reset_buf (subtracted)        = %.2f\n", reset_us);
        logmsg("(1) warm FLOOR                = %.2f\n", floor_us);
        if (evict) {
            logmsg("(2) COLD (evict %0.0f MiB/iter)  = %.2f   penalty=%.2f   [evict_only=%.2f]\n",
                   EVICTB / 1048576.0, cold_us, cold_us - floor_us, evict_only_us);
        } else {
            logmsg("(2) COLD                      = skipped (DIAG_EVICT=0)\n");
        }
        logmsg("\n-- (4) STRAGGLER SKEW (rank %ld slow; rank0's reduce time) --\n", SKEWRK);
        logmsg("  skew_us   reduce_us   over_floor\n");
        for (int s = 0; s < NSKEW; s++)
            logmsg("  %7.0f   %9.2f   %+8.2f\n", SKEW[s], skew_reduce_us[s], skew_reduce_us[s] - floor_us);
        /* slope b in reduce_us ~= a + b*skew, from first/last skew points */
        double b = (skew_reduce_us[NSKEW - 1] - skew_reduce_us[0]) / (SKEW[NSKEW - 1] - SKEW[0]);
        logmsg("  skew slope b = %.3f  (b~1 => reduce time tracks the slow rank => straggler-sync-bound)\n", b);
        logmsg("\n-- READ-OUT --\n");
        logmsg("robust=%d  floor=%.2f  cold_penalty=%.2f  skew_slope=%.3f\n",
               c.robust, floor_us, evict ? cold_us - floor_us : 0.0, b);
        logmsg("(diff floor/cold across TP_AR_ROBUST=0 vs 1 runs to size the robust overhead)\n");
    }

    barrier0();
    free(buf); if (evict) free(evict);
    tp_comm_free(&c);
    utofu_dereg_mem(Vcq, BarBase, 0);
    utofu_free_vcq(Vcq);
    free(Bar);
    if (g_log) fclose(g_log);
    return 0;
}
