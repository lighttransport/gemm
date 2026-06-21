/*
 * p2p_latbw_bench - MPI-free uTofu point-to-point latency & bandwidth
 * characterization on A64FX/Fugaku (12-node 2x3x2 in-unit torus). This isolates
 * the raw Tofu link behaviour the rest of the suite's higher-level estimates are
 * built on, and is the only program here that exercises one-sided Get
 * (utofu_get) and atomic RMW (utofu_armw8) -- every other bench is Put + memory
 * sentinel polling.
 *
 * What it measures, all driven from rank 0:
 *
 *   (1) LATENCY SWEEP over ALL peers (rank 0 <-> r, r = 1..N-1), each reported
 *       next to its computed torus hop-distance so distance sensitivity on the
 *       2x3x2 (a,b,c in-unit axes) is visible:
 *         - Put  ping-pong round-trip  -> one-way = RTT/2 (needs the peer to echo)
 *         - Get  one-sided round-trip  -> request+response, timed at the initiator
 *                via the local MRQ (UTOFU_MRQ_TYPE_LCL_GET); the peer is passive
 *         - armw8 fetch-add round-trip -> one-sided atomic, MRQ-timed; peer passive
 *
 *   (2) BANDWIDTH SWEEP vs message size against one target peer (default the
 *       farthest), using a pipelined ASYNC window of W outstanding ops -- the
 *       window is what saturates the link (W=1 collapses to per-op latency):
 *         - Put injection BW : keep W puts outstanding, drain the local TCQ
 *                              (sender-side injection rate)
 *         - Get delivered BW : keep W gets outstanding, drain the MRQ
 *                              (data has fully landed locally -> delivered rate)
 *
 * Like the rest of the suite this binary makes ZERO MPI calls: mpiexec only
 * places it; it reconstructs every peer's VCQ ID + STADD from tofu_topo.txt
 * coordinates via utofu_construct_vcq_id()/utofu_query_stadd() (the file is
 * written once by tofu_topo_helper). A retried fan-in/fan-out bootstrap then a
 * tight barrier (the pp_handoff_bench pattern) separate the phases; idle ranks
 * park at the barrier while rank 0 drives.
 *
 * Build (NO -lmpi, no OpenMP). Native A64FX node:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall \
 *       -o p2p_latbw_bench p2p_latbw_bench.c -ltofucom
 *
 * Run (after tofu_topo_helper, 1 proc/node):
 *   mpiexec -np 12 ./p2p_latbw_bench
 * stdout is swallowed by mpiexec -> read rank-0's p2p_log_<coords>.txt.
 *
 * Tunables (env):
 *   P2P_MAXBYTES  max BW message size, bytes        (default 4 MiB)
 *   P2P_WINDOW    outstanding async ops in BW loop  (default 8)
 *   P2P_LAT_BYTES latency message size, bytes       (default 8)
 *   P2P_ITERS     timed latency ops / peer          (default 2000)
 *   P2P_WARMUP    untimed latency warmup            (default 200)
 *   P2P_BW_ITERS  timed messages per BW size        (default 200)
 *   P2P_BW_WARMUP untimed BW warmup messages        (default 20)
 *   P2P_BW_PEER   target peer rank for the BW sweep (default = farthest)
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

#define MAX_NODES 128
#define BENCH_STAG DEMO_STAG          /* predictable-STADD convention */
#define WAIT_TIMEOUT_SEC 30.0
#define P2P_MAGIC 0x9217EE00u         /* send-buffer head magic = base | sender rank */
#define CHECK_LEN 64                  /* bytes Get-verified in the one-time check */

/* ----- copied helpers (pp_handoff_bench.c / ring_attn_bench.c conventions) ----- */
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

/* A64FX leaves TNI-written (Get-landed) lines stale in cache; invalidate before a
 * CPU read (suite: reference_utofu_cacheline_hazard / assetload flag_inval). Only
 * used in the one-time correctness check -- the timed loops never read payload. */
static inline void inval_range(const volatile void *p, size_t len)
{
    const char *a = (const char *)p;
    for (size_t o = 0; o < len; o += DEMO_CACHE_LINE)
        __asm__ __volatile__("dc civac, %0" :: "r"(a + o) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
}

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

/* ----- file-scope state (single-threaded bench; globals keep loops readable) ----- */
static int             N, MyRank;
static uint8_t         Topo[MAX_NODES][TOFU_NCOORDS];
static int             Ext[TOFU_NCOORDS];        /* per-axis torus extent (ring size proxy) */
static char           *Region;
static size_t          MAXB, WIN, LAT_PLEN, SlotPP;
static size_t          SEND_OFF, PUTRECV_OFF, GETLAND_OFF, PPFWD_OFF, PPBWD_OFF, ARMW_OFF, STAGE_OFF, BAR_BASE;
static utofu_vcq_hdl_t Vcq;
static utofu_stadd_t   Base;
static utofu_vcq_id_t  PeerVcq[MAX_NODES];
static utofu_stadd_t   PeerBase[MAX_NODES];
static const unsigned long PUT_FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
static const unsigned long MRQ_FLAGS = UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE;
static uint64_t        Bt = 1;                   /* monotonic barrier token (bootstrap = 1) */

/* slot offsets (every remote-written slot on its own DEMO_CACHE_LINE).
 *   send      : MAXB staging buffer (CPU writes; Put source; also the remote
 *               source that peers' Gets read; holds the startup magic+coords)
 *   put_recv  : WIN landing slots (MAXB apart) for in-flight Puts from a peer
 *   get_land  : WIN local landing slots for in-flight Gets (rank 0 only used)
 *   pp_fwd    : Put-latency ping-pong forward landing (written only by rank 0)
 *   pp_bwd    : Put-latency ping-pong echo landing     (written only by partner)
 *   armw_cell : 8B atomic fetch-add target (own cache line, 8-byte aligned)
 *   stage     : barrier token scratch (kept OFF the send buffer so the seeded
 *               magic/coords the Get-check reads survive the bootstrap barrier)
 *   bar_*     : rank-0 fan-in/fan-out barrier slots */
static inline size_t putrecv_off(size_t k) { return PUTRECV_OFF + k * MAXB; }
static inline size_t getland_off(size_t k) { return GETLAND_OFF + k * MAXB; }
static inline size_t bar_recv_off(int s)   { return BAR_BASE + (size_t)s * DEMO_CACHE_LINE; }
static inline size_t bar_go_off(void)      { return BAR_BASE + (size_t)N * DEMO_CACHE_LINE; }

/* torus Manhattan distance with ring wraparound (exact for the in-unit 2x3x2). */
static int torus_dist(const uint8_t *a, const uint8_t *b)
{
    int d = 0;
    for (int i = 0; i < TOFU_NCOORDS; i++) {
        int e = Ext[i] > 0 ? Ext[i] : 1;
        int x = abs((int)a[i] - (int)b[i]);
        d += (x < e - x) ? x : (e - x);
    }
    return d;
}

/* one Put, busy-retry idiom (ring_attn_bench); drain==1 polls local TCQ completion. */
static void put_issue(utofu_vcq_id_t pv, utofu_stadd_t s, utofu_stadd_t d, size_t len, int drain)
{
    int rc; void *cb;
    for (;;) { rc = utofu_put(Vcq, pv, s, d, len, 0, PUT_FLAGS, NULL);
               if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(Vcq, 0, &cb); }
    if (rc != UTOFU_SUCCESS) die("utofu_put", rc);
    if (drain) { do { rc = utofu_poll_tcq(Vcq, 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND);
                 if (rc != UTOFU_SUCCESS) die("utofu_poll_tcq", rc); }
}
static void wait_seq(volatile uint64_t *q, uint64_t v, const char *what)
{
    double ts = now_sec();
    while (*q < v) if (now_sec() - ts > WAIT_TIMEOUT_SEC) die(what, -1);
}

/* ---- barrier: fan-in to rank 0 then fan-out release; carries token Bt. ----
 * robust==1 retries the fan-in with usleep (startup bootstrap); robust==0 spins. */
static void barrier(int robust)
{
    uint64_t t = ++Bt;
    char *sb = Region + STAGE_OFF;
    if (MyRank == 0) {
        for (int s = 1; s < N; s++)
            wait_seq((volatile uint64_t *)(Region + bar_recv_off(s)), t, "barrier fan-in");
        for (int s = 1; s < N; s++) {
            *(volatile uint64_t *)sb = t;
            put_issue(PeerVcq[s], Base + STAGE_OFF, PeerBase[s] + bar_go_off(), 8, 1);
        }
    } else {
        volatile uint64_t *go = (volatile uint64_t *)(Region + bar_go_off());
        double ts = now_sec();
        do {
            *(volatile uint64_t *)sb = t;
            put_issue(PeerVcq[0], Base + STAGE_OFF, PeerBase[0] + bar_recv_off(MyRank), 8, 1);
            if (!robust) { wait_seq(go, t, "barrier release"); break; }
            for (int a = 0; a < 50 && *go < t; a++) usleep(2000);
            if (now_sec() - ts > WAIT_TIMEOUT_SEC) die("bootstrap barrier timeout", -1);
        } while (*go < t);
    }
}

/* ---- Put-latency ping-pong: rank 0 -> partner.pp_fwd; partner echoes -> rank 0.pp_bwd.
 * Message = LAT_PLEN payload + trailing 8B seq sentinel (the wake signal). ---- */
static void pingpong(int partner, uint64_t seq)
{
    char *sb = Region + SEND_OFF;
    if (MyRank == 0) {
        *(volatile uint64_t *)(sb + LAT_PLEN) = seq;
        put_issue(PeerVcq[partner], Base + SEND_OFF, PeerBase[partner] + PPFWD_OFF, LAT_PLEN + 8, 1);
        wait_seq((volatile uint64_t *)(Region + PPBWD_OFF + LAT_PLEN), seq, "pingpong pong");
    } else if (MyRank == partner) {
        wait_seq((volatile uint64_t *)(Region + PPFWD_OFF + LAT_PLEN), seq, "pingpong ping");
        *(volatile uint64_t *)(sb + LAT_PLEN) = seq;
        put_issue(PeerVcq[0], Base + SEND_OFF, PeerBase[0] + PPBWD_OFF, LAT_PLEN + 8, 1);
    }
}

/* ---- Get one-sided round-trip (rank 0 only): read peer's send buffer into a
 * local landing slot; completion is the local MRQ LCL_GET notice. ---- */
static void get_once(int r, size_t len)
{
    int rc; void *cb; struct utofu_mrq_notice nt;
    for (;;) { rc = utofu_get(Vcq, PeerVcq[r], Base + getland_off(0), PeerBase[r] + SEND_OFF,
                              len, 0, MRQ_FLAGS, NULL);
               if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(Vcq, 0, &cb); }
    if (rc != UTOFU_SUCCESS) die("utofu_get", rc);
    do { rc = utofu_poll_mrq(Vcq, 0, &nt); } while (rc == UTOFU_ERR_NOT_FOUND);
    if (rc != UTOFU_SUCCESS) die("utofu_poll_mrq(get)", rc);
    if (nt.notice_type != UTOFU_MRQ_TYPE_LCL_GET) die("unexpected MRQ type (get)", (int)nt.notice_type);
}

/* ---- armw8 fetch-add round-trip (rank 0 only): atomically add 1 to peer's
 * armw_cell; the pre-op value is returned in the MRQ notice (nt.rmt_value). ---- */
static uint64_t armw_once(int r)
{
    int rc; void *cb; struct utofu_mrq_notice nt;
    for (;;) { rc = utofu_armw8(Vcq, PeerVcq[r], UTOFU_ARMW_OP_ADD, 1,
                                PeerBase[r] + ARMW_OFF, 0, MRQ_FLAGS, NULL);
               if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(Vcq, 0, &cb); }
    if (rc != UTOFU_SUCCESS) die("utofu_armw8", rc);
    do { rc = utofu_poll_mrq(Vcq, 0, &nt); } while (rc == UTOFU_ERR_NOT_FOUND);
    if (rc != UTOFU_SUCCESS) die("utofu_poll_mrq(armw)", rc);
    if (nt.notice_type != UTOFU_MRQ_TYPE_LCL_ARMW) die("unexpected MRQ type (armw)", (int)nt.notice_type);
    return nt.rmt_value;                              /* pre-operation value */
}

/* ---- bandwidth: keep up to WIN ops of size S outstanding, count completions.
 * Put -> drain local TCQ (injection rate). Returns total time for `msgs`. ---- */
static double put_bw_run(int r, size_t S, long msgs)
{
    int rc; void *cb;
    long issued = 0, done = 0;
    double t0 = now_sec();
    while (done < msgs) {
        while (issued - done < (long)WIN && issued < msgs) {
            size_t slot = (size_t)(issued % (long)WIN);
            rc = utofu_put(Vcq, PeerVcq[r], Base + SEND_OFF, PeerBase[r] + putrecv_off(slot),
                           S, 0, PUT_FLAGS, NULL);
            if (rc == UTOFU_ERR_BUSY) break;
            if (rc != UTOFU_SUCCESS) die("utofu_put(bw)", rc);
            issued++;
        }
        rc = utofu_poll_tcq(Vcq, 0, &cb);
        if (rc == UTOFU_SUCCESS) done++;
        else if (rc != UTOFU_ERR_NOT_FOUND) die("utofu_poll_tcq(bw)", rc);
    }
    return now_sec() - t0;
}
/* Get -> drain MRQ (delivered rate: data has landed locally). */
static double get_bw_run(int r, size_t S, long msgs)
{
    int rc; struct utofu_mrq_notice nt;
    long issued = 0, done = 0;
    double t0 = now_sec();
    while (done < msgs) {
        while (issued - done < (long)WIN && issued < msgs) {
            size_t slot = (size_t)(issued % (long)WIN);
            rc = utofu_get(Vcq, PeerVcq[r], Base + getland_off(slot), PeerBase[r] + SEND_OFF,
                           S, 0, MRQ_FLAGS, NULL);
            if (rc == UTOFU_ERR_BUSY) break;
            if (rc != UTOFU_SUCCESS) die("utofu_get(bw)", rc);
            issued++;
        }
        rc = utofu_poll_mrq(Vcq, 0, &nt);
        if (rc == UTOFU_SUCCESS) {
            if (nt.notice_type != UTOFU_MRQ_TYPE_LCL_GET) die("unexpected MRQ type (get bw)", (int)nt.notice_type);
            done++;
        } else if (rc != UTOFU_ERR_NOT_FOUND) die("utofu_poll_mrq(bw)", rc);
    }
    return now_sec() - t0;
}

int main(void)
{
    int rc;
    MAXB     = (size_t)envl("P2P_MAXBYTES", 4 * 1024 * 1024);
    WIN      = (size_t)envl("P2P_WINDOW", 8);
    long LATB = envl("P2P_LAT_BYTES", 8);
    long ITERS  = envl("P2P_ITERS", 2000);
    long WARMUP = envl("P2P_WARMUP", 200);
    long BW_ITERS  = envl("P2P_BW_ITERS", 200);
    long BW_WARMUP = envl("P2P_BW_WARMUP", 20);

    if (WIN < 1) WIN = 1;
    MAXB = (MAXB + DEMO_CACHE_LINE - 1) & ~(size_t)(DEMO_CACHE_LINE - 1);  /* line-aligned */
    LAT_PLEN = ((size_t)LATB + 7) & ~(size_t)7;                            /* 8B-aligned */
    SlotPP = (LAT_PLEN + 8 + DEMO_CACHE_LINE - 1) & ~(size_t)(DEMO_CACHE_LINE - 1);
    if (LAT_PLEN + 8 > MAXB) die("P2P_LAT_BYTES too large for P2P_MAXBYTES", -1);

    /* ---- uTofu setup (pp_handoff_bench conventions); single VCQ / single TNI ---- */
    utofu_tni_id_t *tni_ids = NULL; size_t num_tnis = 0;
    rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
    if (rc != UTOFU_SUCCESS) die("utofu_get_onesided_tnis", rc);
    if (num_tnis <= DEMO_TNI_INDEX) die("no onesided TNIs", -1);
    utofu_tni_id_t tni = tni_ids[DEMO_TNI_INDEX];
    free(tni_ids);

    uint8_t my_coords[TOFU_NCOORDS] = {0};
    rc = utofu_query_my_coords(my_coords);
    if (rc != UTOFU_SUCCESS) die("utofu_query_my_coords", rc);
    {
        char name[64];
        snprintf(name, sizeof name, "p2p_log_%u_%u_%u_%u_%u_%u.txt",
                 my_coords[0], my_coords[1], my_coords[2], my_coords[3], my_coords[4], my_coords[5]);
        g_log = fopen(name, "w");
    }

    N = read_topo(Topo);
    MyRank = -1;
    for (int r = 0; r < N; r++) if (memcmp(Topo[r], my_coords, TOFU_NCOORDS) == 0) MyRank = r;
    if (MyRank == -1) { fprintf(stderr, "my coords not in %s\n", TOPO_PATH); exit(1); }

    /* per-axis extent = max-min+1 (exact ring size for the in-unit a,b,c=2,3,2;
     * 1 for the fixed x,y,z on a single-unit job). */
    for (int i = 0; i < TOFU_NCOORDS; i++) {
        int lo = 255, hi = 0;
        for (int r = 0; r < N; r++) { int v = Topo[r][i]; if (v < lo) lo = v; if (v > hi) hi = v; }
        Ext[i] = hi - lo + 1;
    }

    /* default BW peer = farthest from rank 0 (overridable). */
    int far = 1, fard = -1;
    for (int r = 1; r < N; r++) { int d = torus_dist(Topo[0], Topo[r]); if (d > fard) { fard = d; far = r; } }
    int bw_peer = (int)envl("P2P_BW_PEER", far);
    if (bw_peer <= 0 || bw_peer >= N) bw_peer = far;

    /* ---- region layout + registration ---- */
    SEND_OFF    = 0;
    PUTRECV_OFF = SEND_OFF + MAXB;
    GETLAND_OFF = PUTRECV_OFF + WIN * MAXB;
    PPFWD_OFF   = GETLAND_OFF + WIN * MAXB;
    PPBWD_OFF   = PPFWD_OFF + SlotPP;
    ARMW_OFF    = PPBWD_OFF + SlotPP;
    STAGE_OFF   = ARMW_OFF + DEMO_CACHE_LINE;
    BAR_BASE    = STAGE_OFF + DEMO_CACHE_LINE;
    size_t region_sz = BAR_BASE + (size_t)(N + 1) * DEMO_CACHE_LINE;
    if (posix_memalign((void **)&Region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign", -1);
    memset(Region, 0, region_sz);

    /* seed my send buffer head+coords so peers' Get-check reads a known pattern. */
    *(volatile uint64_t *)(Region + SEND_OFF) = P2P_MAGIC | (uint64_t)MyRank;
    memcpy(Region + SEND_OFF + 8, my_coords, TOFU_NCOORDS);
    __asm__ __volatile__("dsb sy" ::: "memory");      /* make it visible to the TNI */

    rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &Vcq);
    if (rc != UTOFU_SUCCESS) die("utofu_create_vcq_with_cmp_id", rc);
    utofu_vcq_id_t my_real;
    rc = utofu_query_vcq_id(Vcq, &my_real);
    if (rc != UTOFU_SUCCESS) die("utofu_query_vcq_id", rc);
    {                                                  /* VCQ self-check (cq_id convention) */
        utofu_vcq_id_t conv;
        rc = utofu_construct_vcq_id(my_coords, tni, DEMO_CQ_ID, DEMO_CMP_ID, &conv);
        if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(self)", rc);
        utofu_vcq_id_t a = my_real, b = conv;
        utofu_set_vcq_id_path(&a, NULL); utofu_set_vcq_id_path(&b, NULL);
        if (a != b) die("VCQ self-check (cq_id convention wrong)", -1);
    }
    rc = utofu_reg_mem_with_stag(Vcq, Region, region_sz, BENCH_STAG, 0, &Base);
    if (rc != UTOFU_SUCCESS) die("utofu_reg_mem_with_stag", rc);
    {
        utofu_stadd_t chk;
        rc = utofu_query_stadd(my_real, BENCH_STAG, &chk);
        if (rc != UTOFU_SUCCESS || chk != Base) die("STADD self-check", rc);
    }
    for (int r = 0; r < N; r++) {
        if (r == MyRank) { PeerVcq[r] = my_real; PeerBase[r] = Base; continue; }
        rc = utofu_construct_vcq_id(Topo[r], tni, DEMO_CQ_ID, DEMO_CMP_ID, &PeerVcq[r]);
        if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(peer)", rc);
        utofu_set_vcq_id_path(&PeerVcq[r], NULL);
        rc = utofu_query_stadd(PeerVcq[r], BENCH_STAG, &PeerBase[r]);
        if (rc != UTOFU_SUCCESS) die("utofu_query_stadd(peer)", rc);
    }

    if (MyRank == 0) {
        logmsg("=== uTofu point-to-point latency / bandwidth ===\n");
        logmsg("nodes=%d  axis extents=%d,%d,%d,%d,%d,%d\n", N,
               Ext[0], Ext[1], Ext[2], Ext[3], Ext[4], Ext[5]);
        logmsg("lat: msg=%zuB iters=%ld warmup=%ld   bw: maxbytes=%zu window=%zu peer=rank%d(dist=%d) iters=%ld\n",
               LAT_PLEN, ITERS, WARMUP, MAXB, WIN, bw_peer, torus_dist(Topo[0], Topo[bw_peer]), BW_ITERS);
        logmsg("region=%.1f MiB/node\n", region_sz / 1048576.0);
    }

    barrier(1);                                        /* robust startup bootstrap */

    /* ================= one-time correctness (check peer = rank 1) ================= */
    if (MyRank == 0) {                                 /* (a) Get reads rank1's seeded buffer */
        get_once(1, CHECK_LEN);
        inval_range(Region + getland_off(0), CHECK_LEN);
        uint64_t head = *(volatile uint64_t *)(Region + getland_off(0));
        if (head != (P2P_MAGIC | 1ull)) die("Get verify: head mismatch", (int)head);
        if (memcmp(Region + getland_off(0) + 8, Topo[1], TOFU_NCOORDS) != 0)
            die("Get verify: coords mismatch", -1);
        logmsg("[verify] Get from rank1 OK (head=0x%lx)\n", (unsigned long)head);
    }
    barrier(0);
    if (MyRank == 0) {                                 /* (b) armw8 fetch-add: pre-op must be 0 */
        uint64_t pre = armw_once(1);
        if (pre != 0) die("armw verify: pre-op value != 0", (int)pre);
        logmsg("[verify] armw8 fetch-add on rank1 OK (pre-op=%lu)\n", (unsigned long)pre);
    }
    barrier(0);
    {                                                  /* (c) ping-pong validates Put RTT */
        pingpong(1, 100);
        if (MyRank == 0) logmsg("[verify] Put ping-pong with rank1 OK\n");
    }
    barrier(0);

    /* ================= latency sweep over all peers ================= */
    struct { int dist; double pp_rtt, get_rtt, armw_rtt; } res[MAX_NODES];
    memset(res, 0, sizeof res);
    uint64_t seq = 1000;

    for (int r = 1; r < N; r++) {
        barrier(0);
        /* Put ping-pong: only rank 0 and r act */
        if (MyRank == 0 || MyRank == r) {
            for (long i = 0; i < WARMUP; i++) pingpong(r, seq + i);
            double t0 = now_sec();
            for (long i = 0; i < ITERS; i++) pingpong(r, seq + WARMUP + i);
            if (MyRank == 0) res[r].pp_rtt = (now_sec() - t0) / (double)ITERS * 1e6;
        }
        seq += WARMUP + ITERS + 16;
        barrier(0);
        /* Get + armw: rank 0 only (peer passive) */
        if (MyRank == 0) {
            for (long i = 0; i < WARMUP; i++) get_once(r, LAT_PLEN);
            double t0 = now_sec();
            for (long i = 0; i < ITERS; i++) get_once(r, LAT_PLEN);
            res[r].get_rtt = (now_sec() - t0) / (double)ITERS * 1e6;

            for (long i = 0; i < WARMUP; i++) armw_once(r);
            t0 = now_sec();
            for (long i = 0; i < ITERS; i++) armw_once(r);
            res[r].armw_rtt = (now_sec() - t0) / (double)ITERS * 1e6;
            res[r].dist = torus_dist(Topo[0], Topo[r]);
        }
        barrier(0);
    }

    /* ================= bandwidth sweep vs message size ================= */
    double put_bw[24], get_bw[24]; size_t sizes[24]; int nsz = 0;
    for (size_t S = 1024; S <= MAXB && nsz < 24; S *= 4) sizes[nsz++] = S;
    if (nsz == 0 || sizes[nsz - 1] != MAXB) { if (nsz < 24) sizes[nsz++] = MAXB; }

    barrier(0);
    if (MyRank == 0) {
        for (int j = 0; j < nsz; j++) {
            size_t S = sizes[j];
            put_bw_run(bw_peer, S, BW_WARMUP);
            put_bw[j] = (double)BW_ITERS * (double)S / put_bw_run(bw_peer, S, BW_ITERS) / 1e6;
            get_bw_run(bw_peer, S, BW_WARMUP);
            get_bw[j] = (double)BW_ITERS * (double)S / get_bw_run(bw_peer, S, BW_ITERS) / 1e6;
        }
    }
    barrier(0);

    /* ================= report (rank 0) ================= */
    if (MyRank == 0) {
        logmsg("\n-- point-to-point latency vs torus distance (%ld iters/op) --\n", ITERS);
        logmsg("peer  dist  put_pp_rtt  put_1way   get_rtt  armw_rtt   (us)\n");
        for (int r = 1; r < N; r++)
            logmsg("%3d   %3d   %9.2f  %8.2f  %8.2f  %8.2f\n",
                   r, res[r].dist, res[r].pp_rtt, res[r].pp_rtt / 2.0,
                   res[r].get_rtt, res[r].armw_rtt);
        logmsg("  [put_1way = ping-pong RTT/2; get/armw are one-sided round-trips "
               "timed at the initiator via MRQ]\n");

        logmsg("\n-- bandwidth vs message size (peer rank%d, dist=%d, window=%zu) --\n",
               bw_peer, torus_dist(Topo[0], Topo[bw_peer]), WIN);
        logmsg("  size      put_MB/s   get_MB/s\n");
        for (int j = 0; j < nsz; j++) {
            size_t S = sizes[j];
            char sz[16];
            if (S >= 1048576) snprintf(sz, sizeof sz, "%zuMiB", S / 1048576);
            else              snprintf(sz, sizeof sz, "%zuKiB", S / 1024);
            logmsg("%8s   %9.1f  %9.1f\n", sz, put_bw[j], get_bw[j]);
        }
        logmsg("  [put_MB/s = sender injection rate (local TCQ); get_MB/s = delivered "
               "rate (data landed locally, MRQ)]\n");
    }

    barrier(0);                                        /* keep everyone alive to here */
    utofu_dereg_mem(Vcq, Base, 0);
    utofu_free_vcq(Vcq);
    free(Region);
    if (g_log) fclose(g_log);
    return 0;
}
