/*
 * allgather_bench - MPI-free uTofu micro-benchmark measuring the cost of one
 * ALL-GATHER on A64FX/Fugaku (12-node 2x3x2 in-unit torus).
 *
 * The suite already characterizes the ALL-REDUCE half of distributed decode
 * comm (ring / recursive-doubling tree in ring_attn_*) and the ALL-TO-ALL half
 * (MoE expert dispatch in moe_dispatch_bench). The remaining uncovered primitive
 * is ALL-GATHER -- the bandwidth-optimal *half* of an all-reduce
 * (all-reduce == reduce-scatter + all-gather). In tensor-parallel decode it is
 * the concrete cost of reconstructing a sharded hidden vector after
 * sequence-parallel attention/FFN (each rank owns HIDDEN/N of the vector and must
 * gather the rest), and of gathering KV / sharded weights across TP ranks. It is
 * exactly what tools/decode_estimate.py folds into its all-reduce term without
 * separating; this bench measures it directly and ties it back to the tree
 * all-reduce baseline so the relationship can be checked.
 *
 * Scope: comm-pattern + roofline only. No matmul / no reduction math is performed
 * (like ring_attn_bench and moe_dispatch_bench). We move realistically sized
 * shards and time the schedules.
 *
 * Four transport variants over the same gather result (every rank ends holding
 * all N shards), plus the tree all-reduce baseline:
 *   (a) NAIVE  : single TNI, all-broadcast -- each rank Puts its shard to all N-1
 *                peers' slot[myrank]. N-1 Puts, drained each. The baseline.
 *   (b) MULTI  : same all-broadcast but the N-1 distinct destinations are
 *                round-robined over MOE/AG_NTNI TNIs, all Puts issued before any
 *                TCQ drain -- the suite's recurring distinct-destination lever
 *                (tni_stripe_bench found same-peer striping is a dead end; distinct
 *                destinations parallelize, ~3x cap on this torus per finding #10).
 *   (c) RING   : bandwidth-optimal all-gather. N-1 sequential steps; at step s a
 *                rank forwards the chunk it currently holds to its ring successor
 *                and receives the next chunk from its predecessor. One neighbor per
 *                step, (N-1)*shard bytes moved, but a long serial dependency chain
 *                -- the all-gather analog of the ring all-reduce the suite found
 *                loses to the tree by 1.77x.
 *   (d) RDBL   : recursive-doubling all-gather (Rabenseifner non-pow2). log2(N)
 *                steps with messages that double each step; latency-optimal for
 *                small shards. Direct analog of the tree all-reduce -- the test of
 *                whether the same recursive-doubling-beats-ring result holds for
 *                all-gather, and the latency-bound decode primitive.
 *   (T) TREE   : recursive-doubling all-reduce over the FULL vector (the
 *                decode_estimate.py assumption). All-reduce moves NRounds*full
 *                bytes/rank; all-gather moves ~1*full, so the headline ratio shows
 *                how much cheaper the gather half is than a full all-reduce.
 *
 * Like the rest of the suite this binary makes ZERO MPI calls: mpiexec only
 * places it; it reconstructs every peer's VCQ ID from tofu_topo.txt coordinates
 * via utofu_construct_vcq_id() (written once by tofu_topo_helper).
 *
 * Build (NO -lmpi, no OpenMP). Native A64FX node:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall \
 *       -o allgather_bench allgather_bench.c -ltofucom
 *
 * Run (after tofu_topo_helper, 1 proc/node):
 *   AG_BATCH=1   AG_NTNI=6 mpiexec -np 12 ./allgather_bench   # decode regime
 *   AG_BATCH=256 AG_NTNI=6 mpiexec -np 12 ./allgather_bench   # batched / prefill regime
 * stdout is swallowed by mpiexec -> read rank-0's ag_log_<coords>.txt.
 *
 * Tunables (env, defaults model GLM-5.1 sequence-parallel hidden gather):
 *   AG_HID    hidden size                       (default 6144)
 *   AG_ABYTES activation bytes/elem (bf16=2)     (default 2)
 *   AG_BATCH  tokens gathered this step          (default 1)
 *   AG_SHARD  per-rank shard bytes (0 = derive   (default 0 -> ceil(HID*AB*B / N))
 *             ceil(full/N) rounded to 8)
 *   AG_NTNI   TNIs for the multi-TNI variant     (default 6)
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
#define RD_STAG    2                  /* recursive-doubling region (own stadd space) */
#define WAIT_TIMEOUT_SEC 30.0
#define TREE_NSTEP 16                 /* distinct recv slots for the tree all-reduce */
#define AG_MAGIC 0xA11A7C00u          /* recv-head magic = base | shard origin rank */
#define MAX_PUT (8u * 1024u * 1024u)  /* uTofu single Put caps ~16 MiB -> chunk at 8 */

/* ----- copied helpers (ring_attn_bench.c / moe_dispatch_bench.c conventions) ----- */
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
static char          *Region;                 /* gather slots + tree slots         */
static size_t         SlotG, SlotT, GBASE;    /* gather slot, tree slot, tree base */
static size_t         Shard, PlenTree;        /* per-rank shard bytes, full vector */
static utofu_vcq_hdl_t Vcq[MAX_TNI];
static utofu_stadd_t   Base[MAX_TNI];
static utofu_vcq_id_t  PeerVcq[MAX_TNI][MAX_NODES];
static utofu_stadd_t   PeerBase[MAX_TNI][MAX_NODES];
static const unsigned long FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
/* tree all-reduce / recursive-doubling shape (Rabenseifner, non-pow2) */
static int Pof2, Rem, NRounds, BcastSid, NewRank;

/* recursive-doubling all-gather region: contiguous gather buffer indexed by GLOBAL
 * shard id (so every doubling block stays a contiguous range). Each shard slot
 * carries a TRAILING seq at its last 8 bytes (offset RdStride-8) -- so the very
 * last 8 bytes of any contiguous [lo,hi) block Put are slot(hi-1)'s seq, which by
 * the suite's single-Put "highest address lands last" rule signals the whole block
 * arrived. (A separate flag Put can overtake the data Put and is NOT safe here.) */
static char          *RdReg;
static size_t         RdStride;               /* per-shard slot stride (cache mult) */
static utofu_stadd_t   RdBase[MAX_TNI];
static utofu_stadd_t   RdPeerBase[MAX_TNI][MAX_NODES];

/* slot offsets within the main region: [gather 0..N-1][tree recv 0..NSTEP-1][tree send] */
static inline size_t gather_off(int g)   { return (size_t)g * SlotG; }
static inline size_t tree_recv_off(int s){ return GBASE + (size_t)s * SlotT; }
static inline size_t tree_send_off(void) { return GBASE + (size_t)TREE_NSTEP * SlotT; }
/* recursive-doubling region offsets (seq at the slot's trailing 8 bytes) */
static inline size_t rd_off(int g)       { return (size_t)g * RdStride; }
static inline size_t rd_seq_off(int g)   { return (size_t)g * RdStride + (RdStride - 8); }

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
/* contiguous Put on TNI 0, chunked past the ~16 MiB single-Put cap (chunks on one
 * VCQ to one dst stay ordered), each chunk drained. */
static void put_block(int dst, utofu_stadd_t sbase, utofu_stadd_t dbase, size_t off, size_t len)
{
    size_t o = 0;
    while (o < len) { size_t c = len - o; if (c > MAX_PUT) c = MAX_PUT;
        put_issue(Vcq[0], PeerVcq[0][dst], sbase + off + o, dbase + off + o, c, 1); o += c; }
}

/* (a) naive / (b) multi-TNI all-broadcast: Put my shard to every peer's slot[myrank]. */
static void ag_broadcast(int multi, uint64_t tok)
{
    char *sb = Region + gather_off(MyRank);
    *(volatile uint64_t *)(sb)         = AG_MAGIC | (uint64_t)MyRank;   /* head: my id */
    *(volatile uint64_t *)(sb + Shard) = tok;                          /* trailing seq */
    int issued[MAX_TNI] = {0}, idx = 0;
    for (int s = 1; s < N; s++) {
        int dst = (MyRank + s) % N;
        int k = multi ? (idx % NTNI) : 0;
        put_issue(Vcq[k], PeerVcq[k][dst], Base[k] + gather_off(MyRank),
                  PeerBase[k][dst] + gather_off(MyRank), Shard + 8, multi ? 0 : 1);
        if (multi) issued[k]++;
        idx++;
    }
    if (multi) for (int k = 0; k < NTNI; k++) drain_n(k, issued[k]);
}

/* (c) ring all-gather: forward the chunk I hold to my successor, receive the next
 * from my predecessor. Slot index == global chunk id, stable across steps; my
 * slot[c] (c!=myrank) is single-writer (only my predecessor writes it). */
static void ag_ring(uint64_t tok)
{
    int succ = (MyRank + 1) % N, pred = (MyRank - 1 + N) % N; (void)pred;
    char *mine = Region + gather_off(MyRank);
    *(volatile uint64_t *)(mine)         = AG_MAGIC | (uint64_t)MyRank;
    *(volatile uint64_t *)(mine + Shard) = tok;
    double ts = now_sec();
    for (int s = 0; s < N - 1; s++) {
        int csend = (MyRank - s + N) % N;          /* chunk I forward this step       */
        int crecv = (MyRank - s - 1 + N) % N;      /* chunk I receive this step       */
        put_issue(Vcq[0], PeerVcq[0][succ], Base[0] + gather_off(csend),
                  PeerBase[0][succ] + gather_off(csend), Shard + 8, 1);
        volatile uint64_t *sq = (volatile uint64_t *)(Region + gather_off(crecv) + Shard);
        while (*sq < tok) if (now_sec() - ts > WAIT_TIMEOUT_SEC) die("ring wait timeout", -1);
    }
}

/* spin until every other shard slot carries seq>=tok (broadcast/ring completion). */
static void ag_wait_all(uint64_t tok)
{
    double ts = now_sec();
    for (int s = 0; s < N; s++) {
        if (s == MyRank) continue;
        volatile uint64_t *sq = (volatile uint64_t *)(Region + gather_off(s) + Shard);
        while (*sq < tok) if (now_sec() - ts > WAIT_TIMEOUT_SEC) die("ag_wait_all timeout", -1);
    }
}
static int ag_verify(void)
{
    for (int s = 0; s < N; s++) {
        if (s == MyRank) continue;
        uint64_t h = *(volatile uint64_t *)(Region + gather_off(s));
        if (h != (AG_MAGIC | (uint64_t)s)) {
            logmsg("ag verify FAIL src=%d got=0x%lx want=0x%lx\n",
                   s, (unsigned long)h, (unsigned long)(AG_MAGIC | (uint64_t)s));
            return 0;
        }
    }
    return 1;
}

/* ---- recursive-doubling all-gather (Rabenseifner non-pow2) ----
 * newrank<->global rank: pair-odd nr -> 2*nr+1, lone nr -> nr+Rem (matches the
 * tree all-reduce). The held global-id range stays contiguous throughout, so each
 * step is one contiguous block Put + a tiny flag Put (ordered after the data). */
static void rd_init_range(int nr, int *lo, int *hi)
{
    if (nr < Rem) { *lo = 2 * nr;   *hi = 2 * nr + 2; }   /* folded pair: 2 shards */
    else          { *lo = nr + Rem; *hi = nr + Rem + 1; } /* lone:        1 shard  */
}
/* global-id range a participant holds BEFORE issuing step k (after step k-1). */
static void rd_range(int nr, int k, int *lo, int *hi)
{
    if (k == 0) { rd_init_range(nr, lo, hi); return; }
    int l1, h1, l2, h2;
    rd_range(nr, k - 1, &l1, &h1);
    rd_range(nr ^ (1 << (k - 1)), k - 1, &l2, &h2);
    *lo = l1 < l2 ? l1 : l2; *hi = h1 > h2 ? h1 : h2;
}
/* Put a contiguous [lo,hi) shard block; seq travels as slot(hi-1)'s trailing 8
 * bytes (already tok: either my own shard set at entry, or forwarded with tok). */
static void rd_put(int dst, int lo, int hi)
{
    put_block(dst, RdBase[0], RdPeerBase[0][dst], rd_off(lo), (size_t)(hi - lo) * RdStride);
}
/* wait until the partner's block [plo,phi) has fully landed (its last slot's seq). */
static void rd_wait(int phi, uint64_t tok, double ts)
{
    volatile uint64_t *sq = (volatile uint64_t *)(RdReg + rd_seq_off(phi - 1));
    while (*sq < tok) if (now_sec() - ts > WAIT_TIMEOUT_SEC) die("rd wait timeout", -1);
}
static void ag_rdbl(uint64_t tok)
{
    /* my shard lives at slot[myrank]: head id at offset 0, seq at the trailing 8B.
     * Both travel with the block data as it is forwarded. */
    *(volatile uint64_t *)(RdReg + rd_off(MyRank))     = AG_MAGIC | (uint64_t)MyRank;
    *(volatile uint64_t *)(RdReg + rd_seq_off(MyRank)) = tok;
    double ts = now_sec();
    int folded_even = (MyRank < 2 * Rem) && (MyRank % 2 == 0);
    if (MyRank < 2 * Rem) {                                /* fold-in (single shard) */
        if (folded_even) rd_put(MyRank + 1, MyRank, MyRank + 1);
        else             rd_wait(MyRank, tok, ts);         /* even partner's shard = MyRank-1+1 */
    }
    if (NewRank != -1) {                                   /* recursive doubling */
        for (int kk = 0; kk < NRounds; kk++) {
            int pnr = NewRank ^ (1 << kk);
            int pr  = (pnr < Rem) ? (pnr * 2 + 1) : (pnr + Rem);
            int lo, hi, plo, phi;
            rd_range(NewRank, kk, &lo, &hi);               /* what I hold == what I send */
            rd_range(pnr,     kk, &plo, &phi);             /* what partner sends me      */
            rd_put(pr, lo, hi);
            rd_wait(phi, tok, ts);
        }
    }
    if (MyRank < 2 * Rem) {                                /* fold-out: full buffer back */
        if (folded_even) rd_wait(N, tok, ts);
        else             rd_put(MyRank - 1, 0, N);
    }
}
static int rd_verify(void)
{
    for (int g = 0; g < N; g++) {
        if (g == MyRank) continue;
        uint64_t h = *(volatile uint64_t *)(RdReg + rd_off(g));
        if (h != (AG_MAGIC | (uint64_t)g)) {
            logmsg("rd verify FAIL shard=%d got=0x%lx want=0x%lx\n",
                   g, (unsigned long)h, (unsigned long)(AG_MAGIC | (uint64_t)g));
            return 0;
        }
    }
    return 1;
}

/* (T) recursive-doubling all-reduce over the FULL vector (decode_estimate.py
 * assumption). Structure copied from ring_attn_bench.c / moe_dispatch_bench.c. */
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
        snprintf(name, sizeof name, "ag_log_%u_%u_%u_%u_%u_%u.txt",
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
    GBASE = (size_t)N * SlotG;
    size_t region_sz = GBASE + (size_t)(TREE_NSTEP + 1) * SlotT;
    if (posix_memalign((void **)&Region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign", -1);
    memset(Region, 0, region_sz);

    /* recursive-doubling region: contiguous, indexed by global shard id; each slot
     * holds Shard data + a trailing 8B seq (so a block's last 8 bytes signal it). */
    RdStride = (Shard + 8 + (DEMO_CACHE_LINE - 1)) & ~(size_t)(DEMO_CACHE_LINE - 1);
    Pof2 = 1; while (Pof2 * 2 <= N) Pof2 *= 2;
    Rem = N - Pof2;
    NRounds = 0; for (int x = 1; x < Pof2; x <<= 1) NRounds++;
    BcastSid = NRounds + 1;
    if (MyRank < 2 * Rem) NewRank = (MyRank % 2 == 0) ? -1 : MyRank / 2;
    else                  NewRank = MyRank - Rem;
    size_t rd_region_sz = (size_t)N * RdStride;
    if (posix_memalign((void **)&RdReg, DEMO_CACHE_LINE, rd_region_sz) != 0) die("posix_memalign(rd)", -1);
    memset(RdReg, 0, rd_region_sz);

    /* one VCQ per TNI; register both regions in each (independent stadd spaces) --
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
        rc = utofu_reg_mem_with_stag(Vcq[k], RdReg, rd_region_sz, RD_STAG, 0, &RdBase[k]);
        if (rc != UTOFU_SUCCESS) die("utofu_reg_mem_with_stag(rd)", rc);
        for (int r = 0; r < N; r++) {
            if (r == MyRank) {
                PeerVcq[k][r] = my_real; PeerBase[k][r] = Base[k]; RdPeerBase[k][r] = RdBase[k];
                continue;
            }
            rc = utofu_construct_vcq_id(topo[r], tni, DEMO_CQ_ID, DEMO_CMP_ID, &PeerVcq[k][r]);
            if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(peer)", rc);
            utofu_set_vcq_id_path(&PeerVcq[k][r], NULL);
            rc = utofu_query_stadd(PeerVcq[k][r], BENCH_STAG, &PeerBase[k][r]);
            if (rc != UTOFU_SUCCESS) die("utofu_query_stadd(peer)", rc);
            rc = utofu_query_stadd(PeerVcq[k][r], RD_STAG, &RdPeerBase[k][r]);
            if (rc != UTOFU_SUCCESS) die("utofu_query_stadd(peer rd)", rc);
        }
    }
    free(tni_ids);

    if (MyRank == 0) {
        logmsg("=== all-gather (no compute) ===\n");
        logmsg("nodes=%d  HID=%ld abytes=%ld B=%ld  full=%zu B  shard=%zu B\n",
               N, HID, ABYTES, B, full, Shard);
        logmsg("SlotG=%zu SlotT=%zu region=%.1f MiB  RdStride=%zu rd_region=%.2f MiB  NTNI=%d\n",
               SlotG, SlotT, region_sz / 1048576.0, RdStride, rd_region_sz / 1048576.0, NTNI);
        logmsg("tree: Pof2=%d Rem=%d NRounds=%d (%d steps)  rdbl all-gather: %d doubling steps%s\n",
               Pof2, Rem, NRounds, NRounds + (Rem ? 2 : 0), NRounds, Rem ? " + fold in/out" : "");
    }

    /* ---- barrierless bootstrap: all-broadcast seq=1 until seq>=1 from all srcs. ---- */
    {
        double t0 = now_sec(); int got = 0;
        for (int a = 0; a < 400 && !got; a++) {
            ag_broadcast(0, 1);
            int all = 1;
            for (int s = 0; s < N; s++) {
                if (s == MyRank) continue;
                volatile uint64_t *sq = (volatile uint64_t *)(Region + gather_off(s) + Shard);
                if (*sq < 1) all = 0;
            }
            if (all && a >= 8) got = 1;
            usleep(20000);
            if (now_sec() - t0 > WAIT_TIMEOUT_SEC) break;
        }
        if (!got) die("bootstrap timeout", -1);
    }
    uint64_t tok = 2;

    /* one-time correctness pass for each variant (heads carry shard-origin ids) */
    { ag_broadcast(0, tok); ag_wait_all(tok); if (!ag_verify()) die("naive verify failed", -1); tok++; }
    { ag_ring(tok);         ag_wait_all(tok); if (!ag_verify()) die("ring verify failed", -1);  tok++; }
    { ag_rdbl(tok);         if (!rd_verify()) die("rdbl verify failed", -1); tok++; }

    /* ---- timed phases: WARMUP untimed, then ITERS timed. ---- */
#define TIME_BCAST(multi, out_us)                                                \
    do { for (long i = 0; i < WARMUP; i++) { ag_broadcast((multi), tok); ag_wait_all(tok); tok++; } \
         double _t0 = now_sec();                                                 \
         for (long i = 0; i < ITERS;  i++) { ag_broadcast((multi), tok); ag_wait_all(tok); tok++; } \
         (out_us) = (now_sec() - _t0) / (double)ITERS * 1e6; } while (0)
#define TIME_FN(fn, out_us)                                                      \
    do { for (long i = 0; i < WARMUP; i++) { fn(tok); tok++; }                   \
         double _t0 = now_sec();                                                 \
         for (long i = 0; i < ITERS;  i++) { fn(tok); tok++; }                   \
         (out_us) = (now_sec() - _t0) / (double)ITERS * 1e6; } while (0)

    double ag_naive = 0, ag_multi = 0, ag_ring_us = 0, ag_rd_us = 0, tree_us = 0;
    TIME_BCAST(0, ag_naive);
    TIME_BCAST(1, ag_multi);
    TIME_FN(ag_ring, ag_ring_us);
    TIME_FN(ag_rdbl, ag_rd_us);
    TIME_FN(tree_allreduce, tree_us);

    if (MyRank == 0) {
        double recv_bytes = (double)(N - 1) * (double)Shard;       /* gathered per rank */
        double best = ag_naive;
        if (ag_multi   < best) best = ag_multi;
        if (ag_ring_us < best) best = ag_ring_us;
        if (ag_rd_us   < best) best = ag_rd_us;
        const char *bn = (best == ag_rd_us) ? "rdbl" : (best == ag_ring_us) ? "ring"
                       : (best == ag_multi) ? "multiTNI" : "naive";
        logmsg("\n-- per-all-gather comm (us), %ld iters --\n", ITERS);
        logmsg("naive(1 TNI broadcast) =%.2f\n", ag_naive);
        logmsg("multiTNI broadcast     =%.2f  (vs naive x%.2f)\n", ag_multi, ag_naive / ag_multi);
        logmsg("ring all-gather        =%.2f  (vs naive x%.2f)\n", ag_ring_us, ag_naive / ag_ring_us);
        logmsg("recursive-doubling     =%.2f  (vs naive x%.2f, vs ring x%.2f)\n",
               ag_rd_us, ag_naive / ag_rd_us, ag_ring_us / ag_rd_us);
        logmsg("BEST all-gather        =%.2f  (%s)\n", best, bn);
        logmsg("tree all-reduce (full vector, decode_estimate assumption)=%.2f\n", tree_us);
        logmsg("RATIO best_all-gather / tree_all-reduce = %.2f  "
               "[all-gather is the BW-optimal half; all-reduce = reduce-scatter + all-gather]\n",
               best / tree_us);
        logmsg("ingest BW (per rank, %.0f KiB gathered): naive=%.1f multiTNI=%.1f ring=%.1f rdbl=%.1f GB/s\n",
               recv_bytes / 1024.0,
               recv_bytes / (ag_naive   * 1e-6) / 1e9, recv_bytes / (ag_multi  * 1e-6) / 1e9,
               recv_bytes / (ag_ring_us * 1e-6) / 1e9, recv_bytes / (ag_rd_us  * 1e-6) / 1e9);
    }

    for (int k = 0; k < NTNI; k++) {
        utofu_dereg_mem(Vcq[k], RdBase[k], 0);
        utofu_dereg_mem(Vcq[k], Base[k], 0);
        utofu_free_vcq(Vcq[k]);
    }
    free(RdReg);
    free(Region);
    if (g_log) fclose(g_log);
    return 0;
}
