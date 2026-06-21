/*
 * reducescatter_bench - MPI-free uTofu micro-benchmark measuring the cost of one
 * REDUCE-SCATTER on A64FX/Fugaku (12-node 2x3x2 in-unit torus).
 *
 * Reduce-scatter is the *other* bandwidth-optimal half of an all-reduce:
 *   all-reduce == reduce-scatter + all-gather.
 * Every rank starts holding a full N-shard vector (its local contribution) and
 * ends owning the SUM over all ranks of just its own shard. In tensor-parallel
 * decode it is the concrete cost of the reduction step that precedes a
 * sequence-parallel re-gather, and it is the time-reverse of allgather_bench's
 * all-gather. tools/decode_estimate.py folds the whole all-reduce into one tree
 * term; this bench measures the reduce-scatter half directly and -- crucially --
 * closes the decomposition empirically in ONE allocation: does decomposing into
 * (multi-TNI reduce-scatter) + (multi-TNI all-gather) beat the fused
 * recursive-doubling tree all-reduce that the estimator assumes?
 *
 * Scope: comm-pattern + roofline only. No reduction arithmetic is performed
 * (like ring_attn_bench / moe_dispatch_bench / allgather_bench). We move
 * realistically sized shards and time the schedules; the per-receive "add" is
 * elided -- this measures the COMMUNICATION term, so reduce-scatter and
 * all-gather come out equal-cost (their wire patterns are exact time-reverses),
 * and the headline is how the decomposed pair compares to the fused tree.
 *
 * Transports over the same scatter (every rank ends owning shard[myrank]):
 *   (a) NAIVE  : single TNI direct scatter -- Put my contribution-to-shard-d to
 *                rank d's recv slot[myrank], for every d. N-1 Puts, drained each.
 *                The receiver collects N-1 contributions to its own shard (would
 *                sum them). Mirror of all-gather's all-broadcast, but the sender
 *                reads N DISTINCT source slices (a real scatter) instead of
 *                broadcasting one shard -- so multi-TNI gets N-1 independent
 *                source buffers, not one shared one.
 *   (b) MULTI  : same direct scatter, N-1 distinct destinations round-robined over
 *                AG_NTNI TNIs, all Puts issued before any drain (finding #10's
 *                distinct-destination lever; ~3x cap on this torus).
 *   (c) RING   : bandwidth-optimal ring reduce-scatter. N-1 sequential steps; each
 *                step forward the chunk I hold to my successor and receive+reduce
 *                the next from my predecessor. Same (N-1)*shard bytes and the same
 *                serial chain as the ring all-gather -- the time-reverse.
 *   (G) AG     : multi-TNI all-gather broadcast over the reduced shards (the AG
 *                half), measured here so the decomposed all-reduce = RS + AG can be
 *                compared to the fused tree in a single self-contained run.
 *   (T) TREE   : recursive-doubling all-reduce over the FULL vector (the fused
 *                decode_estimate.py assumption; reduce-scatter + all-gather fused).
 *
 * Recursive-halving reduce-scatter is deliberately NOT implemented: it is the
 * exact time-reverse of allgather_bench's recursive-doubling all-gather (finding
 * #12), so it would only reproduce that schedule's cost. The ring + direct + tree
 * triad answers the decomposition question.
 *
 * Like the rest of the suite this binary makes ZERO MPI calls: mpiexec only
 * places it; it reconstructs every peer's VCQ ID from tofu_topo.txt coordinates
 * via utofu_construct_vcq_id() (written once by tofu_topo_helper).
 *
 * Build (NO -lmpi, no OpenMP). Native A64FX node:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall \
 *       -o reducescatter_bench reducescatter_bench.c -ltofucom
 *
 * Run (after tofu_topo_helper, 1 proc/node):
 *   AG_BATCH=1   AG_NTNI=6 mpiexec -np 12 ./reducescatter_bench   # decode regime
 *   AG_BATCH=256 AG_NTNI=6 mpiexec -np 12 ./reducescatter_bench   # batched / prefill
 * stdout is swallowed by mpiexec -> read rank-0's rs_log_<coords>.txt.
 *
 * Tunables (env; shared AG_* names with allgather_bench so runs pair up):
 *   AG_HID    hidden size                       (default 6144)
 *   AG_ABYTES activation bytes/elem (bf16=2)     (default 2)
 *   AG_BATCH  tokens reduced this step           (default 1)
 *   AG_SHARD  per-rank shard bytes (0 = derive   (default 0 -> ceil(HID*AB*B / N))
 *             ceil(full/N) rounded to 8)
 *   AG_NTNI   TNIs for the multi-TNI variants    (default 6)
 *   AG_ITERS  timed exchanges                    (default 2000)
 *   AG_WARMUP untimed warmup exchanges           (default 200)
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

#define MAX_NODES 32
#define MAX_TNI   6
#define BENCH_STAG DEMO_STAG          /* predictable-STADD convention */
#define WAIT_TIMEOUT_SEC 30.0
#define TREE_NSTEP 16                 /* distinct recv slots for the tree all-reduce */
#define RS_MAGIC 0x5CA77E00u          /* recv-head magic = base | source/chunk rank */

/* ----- copied helpers (allgather_bench.c / ring_attn_bench.c conventions) ----- */
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

/* ----- file-scope state (single-threaded bench; globals keep loops readable) ----- */
static int            N, NTNI, MyRank;
static char          *Region;                 /* local[N] + recv[N] + tree slots   */
static size_t         SlotG, SlotT, GBASE;    /* shard slot, tree slot, tree base  */
static size_t         Shard, PlenTree;        /* per-rank shard bytes, full vector */
static utofu_vcq_hdl_t Vcq[MAX_TNI];
static utofu_stadd_t   Base[MAX_TNI];
static utofu_vcq_id_t  PeerVcq[MAX_TNI][MAX_NODES];
static utofu_stadd_t   PeerBase[MAX_TNI][MAX_NODES];
static const unsigned long FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
/* tree all-reduce shape (Rabenseifner, non-pow2) */
static int Pof2, Rem, NRounds, BcastSid, NewRank;

/* slot offsets within the region:
 *   local[d]   = my contribution slice destined for shard d (scatter source / my shard)
 *   recv[s]    = landing for the message from source s (== chunk s for the ring)
 *   tree recv/send for the fused all-reduce baseline.  Each slot its own cache line. */
static inline size_t local_off(int d)    { return (size_t)d * SlotG; }
static inline size_t recv_off(int s)      { return (size_t)(N + s) * SlotG; }
static inline size_t tree_recv_off(int s) { return GBASE + (size_t)s * SlotT; }
static inline size_t tree_send_off(void)  { return GBASE + (size_t)TREE_NSTEP * SlotT; }

/* one Put (the ring_attn_bench BUSY-retry idiom); drain==1 polls local completion. */
static void put_issue(utofu_vcq_hdl_t v, utofu_vcq_id_t pv, utofu_stadd_t s,
                      utofu_stadd_t d, size_t len, int drain)
{
    int rc; void *cb;
    for (;;) { rc = utofu_put(v, pv, s, d, len, 0, FLAGS, NULL);
               if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(v, 0, &cb); }
    if (rc != UTOFU_SUCCESS) die("utofu_put", rc);
    if (drain) { do { rc = utofu_poll_tcq(v, 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND);
                 if (rc != UTOFU_SUCCESS) die("utofu_poll_tcq", rc); }
}
static void drain_n(int k, int n)
{
    int rc; void *cb;
    for (int j = 0; j < n; j++) { do { rc = utofu_poll_tcq(Vcq[k], 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND);
                                  if (rc != UTOFU_SUCCESS) die("utofu_poll_tcq(drain)", rc); }
}
/* (a) naive / (b) multi-TNI DIRECT scatter: Put my slice-for-d to dst d's recv[myrank].
 * Each Put reads a DISTINCT source slot local[d] (a real scatter); the receiver lands
 * N-1 contributions to its own shard in recv[*] (which it would sum). */
static void rs_scatter(int multi, uint64_t tok)
{
    int issued[MAX_TNI] = {0}, idx = 0;
    for (int s = 1; s < N; s++) {
        int dst = (MyRank + s) % N;
        char *sb = Region + local_off(dst);
        *(volatile uint64_t *)(sb)         = RS_MAGIC | (uint64_t)MyRank;  /* head: my id  */
        *(volatile uint64_t *)(sb + Shard) = tok;                          /* trailing seq */
        int k = multi ? (idx % NTNI) : 0;
        put_issue(Vcq[k], PeerVcq[k][dst], Base[k] + local_off(dst),
                  PeerBase[k][dst] + recv_off(MyRank), Shard + 8, multi ? 0 : 1);
        if (multi) issued[k]++;
        idx++;
    }
    if (multi) for (int k = 0; k < NTNI; k++) drain_n(k, issued[k]);
}

/* (G) multi-TNI / single-TNI all-gather broadcast over the reduced shard: Put my
 * shard (local[myrank]) to every peer's recv[myrank]. The AG half of all-reduce. */
static void ag_bcast(int multi, uint64_t tok)
{
    char *sb = Region + local_off(MyRank);
    *(volatile uint64_t *)(sb)         = RS_MAGIC | (uint64_t)MyRank;
    *(volatile uint64_t *)(sb + Shard) = tok;
    int issued[MAX_TNI] = {0}, idx = 0;
    for (int s = 1; s < N; s++) {
        int dst = (MyRank + s) % N;
        int k = multi ? (idx % NTNI) : 0;
        put_issue(Vcq[k], PeerVcq[k][dst], Base[k] + local_off(MyRank),
                  PeerBase[k][dst] + recv_off(MyRank), Shard + 8, multi ? 0 : 1);
        if (multi) issued[k]++;
        idx++;
    }
    if (multi) for (int k = 0; k < NTNI; k++) drain_n(k, issued[k]);
}

/* (c) ring reduce-scatter: forward the chunk I hold to my successor, receive the
 * next chunk from my predecessor (would reduce into it). recv[c] (c!=myrank) is
 * single-writer (only my predecessor writes it). Same wire pattern as ring AG. */
static void rs_ring(uint64_t tok)
{
    int succ = (MyRank + 1) % N;
    char *mine = Region + recv_off(MyRank);            /* seed: my own chunk */
    *(volatile uint64_t *)(mine)         = RS_MAGIC | (uint64_t)MyRank;
    *(volatile uint64_t *)(mine + Shard) = tok;
    double ts = now_sec();
    for (int s = 0; s < N - 1; s++) {
        int csend = (MyRank - s + N) % N;              /* chunk I forward this step  */
        int crecv = (MyRank - s - 1 + N) % N;          /* chunk I receive this step  */
        put_issue(Vcq[0], PeerVcq[0][succ], Base[0] + recv_off(csend),
                  PeerBase[0][succ] + recv_off(csend), Shard + 8, 1);
        volatile uint64_t *sq = (volatile uint64_t *)(Region + recv_off(crecv) + Shard);
        while (*sq < tok) if (now_sec() - ts > WAIT_TIMEOUT_SEC) die("ring wait timeout", -1);
    }
}

/* spin until every other recv slot carries seq>=tok (scatter / gather / ring done). */
static void rs_wait(uint64_t tok)
{
    double ts = now_sec();
    for (int s = 0; s < N; s++) {
        if (s == MyRank) continue;
        volatile uint64_t *sq = (volatile uint64_t *)(Region + recv_off(s) + Shard);
        while (*sq < tok) if (now_sec() - ts > WAIT_TIMEOUT_SEC) die("rs_wait timeout", -1);
    }
}
static int rs_verify(void)
{
    for (int s = 0; s < N; s++) {
        if (s == MyRank) continue;
        uint64_t h = *(volatile uint64_t *)(Region + recv_off(s));
        if (h != (RS_MAGIC | (uint64_t)s)) {
            logmsg("verify FAIL src=%d got=0x%lx want=0x%lx\n",
                   s, (unsigned long)h, (unsigned long)(RS_MAGIC | (uint64_t)s));
            return 0;
        }
    }
    return 1;
}

/* (T) recursive-doubling all-reduce over the FULL vector (decode_estimate.py
 * assumption; reduce-scatter + all-gather FUSED). Copied from allgather_bench.c. */
#define TREE_PUT(prank, sid) \
    put_issue(Vcq[0], PeerVcq[0][prank], Base[0] + tree_send_off(), \
              PeerBase[0][prank] + tree_recv_off(sid), PlenTree + 8, 1)
#define TREE_WAIT(sid, w) do { \
        volatile uint64_t *_rs = (volatile uint64_t *)(Region + tree_recv_off(sid) + PlenTree); \
        double _ts = now_sec(); \
        while (*_rs < (uint64_t)(w)) if (now_sec() - _ts > WAIT_TIMEOUT_SEC) die("tree wait timeout", -1); \
    } while (0)
static void tree_allreduce(uint64_t tok)
{
    *(volatile uint64_t *)(Region + tree_send_off() + PlenTree) = tok;
    if (MyRank < 2 * Rem) {
        if (MyRank % 2 == 0) TREE_PUT(MyRank + 1, 0); else TREE_WAIT(0, tok);
    }
    if (NewRank != -1) {
        for (int kk = 0; kk < NRounds; kk++) {
            int pnr = NewRank ^ (1 << kk);
            int pr  = (pnr < Rem) ? (pnr * 2 + 1) : (pnr + Rem);
            TREE_PUT(pr, kk + 1); TREE_WAIT(kk + 1, tok);
        }
    }
    if (MyRank < 2 * Rem) {
        if (MyRank % 2 == 0) TREE_WAIT(BcastSid, tok); else TREE_PUT(MyRank - 1, BcastSid);
    }
}

int main(void)
{
    int rc;
    long HID    = envl("AG_HID", 6144);
    long ABYTES = envl("AG_ABYTES", 2);
    long B      = envl("AG_BATCH", 1);
    long SHARDov= envl("AG_SHARD", 0);
    long ITERS  = envl("AG_ITERS", 2000);
    long WARMUP = envl("AG_WARMUP", 200);
    NTNI        = (int)envl("AG_NTNI", 6);

    /* ---- uTofu setup (tofu_put_demo conventions) ---- */
    utofu_tni_id_t *tni_ids = NULL; size_t num_tnis = 0;
    rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
    if (rc != UTOFU_SUCCESS) die("utofu_get_onesided_tnis", rc);
    if (NTNI < 1) NTNI = 1;
    if ((size_t)NTNI > num_tnis) NTNI = (int)num_tnis;
    if (NTNI > MAX_TNI) NTNI = MAX_TNI;

    uint8_t my_coords[TOFU_NCOORDS] = {0};
    rc = utofu_query_my_coords(my_coords);
    if (rc != UTOFU_SUCCESS) die("utofu_query_my_coords", rc);
    {
        char name[64];
        snprintf(name, sizeof name, "rs_log_%u_%u_%u_%u_%u_%u.txt",
                 my_coords[0], my_coords[1], my_coords[2], my_coords[3], my_coords[4], my_coords[5]);
        g_log = fopen(name, "w");
    }

    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    N = read_topo(topo);
    MyRank = -1;
    for (int r = 0; r < N; r++) if (memcmp(topo[r], my_coords, TOFU_NCOORDS) == 0) MyRank = r;
    if (MyRank == -1) { fprintf(stderr, "my coords not in %s\n", TOPO_PATH); exit(1); }

    /* ---- sizes. full = HID*ABYTES*B; shard = ceil(full/N) (8-aligned) unless
     * AG_SHARD overrides. Tree all-reduce moves the full vector each step. ---- */
    size_t full = (size_t)HID * (size_t)ABYTES * (size_t)B;
    if (SHARDov > 0) Shard = (size_t)SHARDov;
    else             Shard = (full + (size_t)N - 1) / (size_t)N;
    Shard = (Shard + 7) & ~(size_t)7;
    PlenTree = (full + 7) & ~(size_t)7;

    SlotG = (Shard    + 8 + (DEMO_CACHE_LINE - 1)) & ~(size_t)(DEMO_CACHE_LINE - 1);
    SlotT = (PlenTree + 8 + (DEMO_CACHE_LINE - 1)) & ~(size_t)(DEMO_CACHE_LINE - 1);
    GBASE = (size_t)(2 * N) * SlotG;
    size_t region_sz = GBASE + (size_t)(TREE_NSTEP + 1) * SlotT;
    if (posix_memalign((void **)&Region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign", -1);
    memset(Region, 0, region_sz);

    /* tree all-reduce shape (Rabenseifner non-power-of-2) */
    Pof2 = 1; while (Pof2 * 2 <= N) Pof2 *= 2;
    Rem = N - Pof2;
    NRounds = 0; for (int x = 1; x < Pof2; x <<= 1) NRounds++;
    BcastSid = NRounds + 1;
    if (MyRank < 2 * Rem) NewRank = (MyRank % 2 == 0) ? -1 : MyRank / 2;
    else                  NewRank = MyRank - Rem;

    /* one VCQ per TNI; register the region in each (independent stadd space) --
     * the tni_stripe_bench.c pattern generalized to peer arrays for ALL ranks. */
    for (int k = 0; k < NTNI; k++) {
        utofu_tni_id_t tni = tni_ids[k];
        rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &Vcq[k]);
        if (rc != UTOFU_SUCCESS) die("utofu_create_vcq_with_cmp_id", rc);
        utofu_vcq_id_t my_real;
        rc = utofu_query_vcq_id(Vcq[k], &my_real);
        if (rc != UTOFU_SUCCESS) die("utofu_query_vcq_id", rc);
        if (k == 0) {                          /* VCQ self-check on TNI 0 */
            utofu_vcq_id_t conv;
            rc = utofu_construct_vcq_id(my_coords, tni, DEMO_CQ_ID, DEMO_CMP_ID, &conv);
            if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(self)", rc);
            utofu_vcq_id_t a = my_real, b = conv;
            utofu_set_vcq_id_path(&a, NULL); utofu_set_vcq_id_path(&b, NULL);
            if (a != b) die("VCQ self-check (cq_id convention wrong)", -1);
        }
        rc = utofu_reg_mem_with_stag(Vcq[k], Region, region_sz, BENCH_STAG, 0, &Base[k]);
        if (rc != UTOFU_SUCCESS) die("utofu_reg_mem_with_stag", rc);
        utofu_stadd_t chk;
        rc = utofu_query_stadd(my_real, BENCH_STAG, &chk);
        if (rc != UTOFU_SUCCESS || chk != Base[k]) die("STADD self-check", rc);
        for (int r = 0; r < N; r++) {
            if (r == MyRank) { PeerVcq[k][r] = my_real; PeerBase[k][r] = Base[k]; continue; }
            rc = utofu_construct_vcq_id(topo[r], tni, DEMO_CQ_ID, DEMO_CMP_ID, &PeerVcq[k][r]);
            if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(peer)", rc);
            utofu_set_vcq_id_path(&PeerVcq[k][r], NULL);
            rc = utofu_query_stadd(PeerVcq[k][r], BENCH_STAG, &PeerBase[k][r]);
            if (rc != UTOFU_SUCCESS) die("utofu_query_stadd(peer)", rc);
        }
    }
    free(tni_ids);

    if (MyRank == 0) {
        logmsg("=== reduce-scatter (no reduction compute) ===\n");
        logmsg("nodes=%d  HID=%ld abytes=%ld B=%ld  full=%zu B  shard=%zu B\n",
               N, HID, ABYTES, B, full, Shard);
        logmsg("SlotG=%zu SlotT=%zu region=%.1f MiB  NTNI=%d\n",
               SlotG, SlotT, region_sz / 1048576.0, NTNI);
        logmsg("tree: Pof2=%d Rem=%d NRounds=%d (%d steps)\n",
               Pof2, Rem, NRounds, NRounds + (Rem ? 2 : 0));
    }

    /* ---- barrierless bootstrap: direct-scatter seq=1 until seq>=1 from all srcs. ---- */
    {
        double t0 = now_sec(); int got = 0;
        for (int a = 0; a < 400 && !got; a++) {
            rs_scatter(0, 1);
            int all = 1;
            for (int s = 0; s < N; s++) {
                if (s == MyRank) continue;
                volatile uint64_t *sq = (volatile uint64_t *)(Region + recv_off(s) + Shard);
                if (*sq < 1) all = 0;
            }
            if (all && a >= 8) got = 1;
            usleep(20000);
            if (now_sec() - t0 > WAIT_TIMEOUT_SEC) break;
        }
        if (!got) die("bootstrap timeout", -1);
    }
    uint64_t tok = 2;

    /* one-time correctness pass for each variant (heads carry source/chunk ids) */
    { rs_scatter(0, tok); rs_wait(tok); if (!rs_verify()) die("naive verify failed", -1); tok++; }
    { rs_ring(tok);       rs_wait(tok); if (!rs_verify()) die("ring verify failed", -1);  tok++; }
    { ag_bcast(0, tok);   rs_wait(tok); if (!rs_verify()) die("ag verify failed", -1);    tok++; }

    /* ---- timed phases: WARMUP untimed, then ITERS timed. ---- */
#define TIME_SCATTER(fn, multi, out_us)                                          \
    do { for (long i = 0; i < WARMUP; i++) { fn((multi), tok); rs_wait(tok); tok++; } \
         double _t0 = now_sec();                                                 \
         for (long i = 0; i < ITERS;  i++) { fn((multi), tok); rs_wait(tok); tok++; } \
         (out_us) = (now_sec() - _t0) / (double)ITERS * 1e6; } while (0)
#define TIME_FN(fn, out_us)                                                      \
    do { for (long i = 0; i < WARMUP; i++) { fn(tok); tok++; }                   \
         double _t0 = now_sec();                                                 \
         for (long i = 0; i < ITERS;  i++) { fn(tok); tok++; }                   \
         (out_us) = (now_sec() - _t0) / (double)ITERS * 1e6; } while (0)

    double rs_naive = 0, rs_multi = 0, rs_ring_us = 0, ag_multi = 0, tree_us = 0;
    TIME_SCATTER(rs_scatter, 0, rs_naive);
    TIME_SCATTER(rs_scatter, 1, rs_multi);
    TIME_FN(rs_ring, rs_ring_us);
    TIME_SCATTER(ag_bcast, 1, ag_multi);
    TIME_FN(tree_allreduce, tree_us);

    if (MyRank == 0) {
        double recv_bytes = (double)(N - 1) * (double)Shard;       /* reduced into my shard */
        double best = rs_naive;
        if (rs_multi   < best) best = rs_multi;
        if (rs_ring_us < best) best = rs_ring_us;
        const char *bn = (best == rs_ring_us) ? "ring" : (best == rs_multi) ? "multiTNI" : "naive";
        double decomposed = best + ag_multi;                       /* RS + AG = all-reduce  */
        logmsg("\n-- per-reduce-scatter comm (us), %ld iters --\n", ITERS);
        logmsg("naive(1 TNI scatter)   =%.2f\n", rs_naive);
        logmsg("multiTNI scatter       =%.2f  (vs naive x%.2f)\n", rs_multi, rs_naive / rs_multi);
        logmsg("ring reduce-scatter    =%.2f  (vs naive x%.2f)\n", rs_ring_us, rs_naive / rs_ring_us);
        logmsg("BEST reduce-scatter    =%.2f  (%s)\n", best, bn);
        logmsg("multiTNI all-gather    =%.2f  (the AG half)\n", ag_multi);
        logmsg("tree all-reduce (fused, decode_estimate assumption)=%.2f\n", tree_us);
        logmsg("RATIO best_RS / tree_all-reduce = %.2f  [reduce-scatter is the BW-optimal half]\n",
               best / tree_us);
        logmsg("DECOMPOSED all-reduce (best_RS + multiTNI_AG) =%.2f  RATIO vs fused tree x%.2f  "
               "[<1 => decomposing beats the fused tree]\n",
               decomposed, decomposed / tree_us);
        logmsg("ingest BW (per rank, %.0f KiB reduced): naive=%.1f multiTNI=%.1f ring=%.1f GB/s\n",
               recv_bytes / 1024.0,
               recv_bytes / (rs_naive   * 1e-6) / 1e9, recv_bytes / (rs_multi  * 1e-6) / 1e9,
               recv_bytes / (rs_ring_us * 1e-6) / 1e9);
    }

    for (int k = 0; k < NTNI; k++) { utofu_dereg_mem(Vcq[k], Base[k], 0); utofu_free_vcq(Vcq[k]); }
    free(Region);
    if (g_log) fclose(g_log);
    return 0;
}
