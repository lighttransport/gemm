/*
 * pp_handoff_bench - MPI-free uTofu micro-benchmark measuring the cost of the
 * PIPELINE-PARALLEL activation handoff on A64FX/Fugaku (12-node 2x3x2 in-unit
 * torus). This is the point-to-point inter-stage send/recv -- the third
 * distributed-decode comm pattern in the suite, alongside the all-to-all
 * (moe_dispatch_bench), all-gather (allgather_bench), reduce-scatter
 * (reducescatter_bench), and tree all-reduce baselines.
 *
 * In pipeline parallelism the model's layers are partitioned into S contiguous
 * stages, one per rank, and a microbatch flows stage 0 -> 1 -> ... -> S-1
 * (forward), then S-1 -> ... -> 0 (backward, in training). The ONLY comm is a
 * single P2P send of the activation tensor (HID*ABYTES*B bytes) across each
 * stage boundary -- no collective. Two facts decide whether PP is worth it on
 * this torus, and this bench measures both:
 *
 *   1. the per-hop handoff latency (and whether it depends on torus distance), and
 *   2. the empty-pipeline critical-path latency = the full S-1-hop chain, which
 *      is exactly what an autoregressive DECODE step pays: with one token in
 *      flight the pipeline can never fill, so the stages run strictly serially.
 *
 * PP is the structural alternative to tensor parallelism: TP pays one all-reduce
 * per layer (decode_estimate.py's 2*num_layers tree terms), whereas PP pays only
 * S-1 tiny point-to-point handoffs for the WHOLE model -- comm that does not
 * scale with depth, traded against the pipeline bubble. The rank-0 report puts
 * the measured chain next to one tree all-reduce so that trade is quantified.
 *
 * Scope: comm-pattern + roofline only -- NO stage compute (mirrors the rest of
 * the suite). Each stage forwards the bytes the moment they arrive; the
 * per-stage GEMM/attention is elided, so the chain measures the pure handoff
 * critical path (the lower bound the bubble formula builds on).
 *
 * Variants:
 *   (P) ping-pong : rank 0 <-> one partner, round-trip / 2 = one isolated hop.
 *                   Run for a NEAR partner (rank 1) and a FAR partner (rank N-1)
 *                   to test torus-distance sensitivity of a single P2P send.
 *   (C) chain     : full forward+backward store-and-forward relay
 *                   0->1->..->(S-1)->..->1->0. Rank 0 times the 2(S-1)-hop round
 *                   trip with no global clock; /2 = the forward decode chain, and
 *                   /(2(S-1)) = the in-chain per-hop (includes each stage's
 *                   wake-on-seq + re-issue store-and-forward cost).
 *   (T) tree      : recursive-doubling all-reduce over the full activation (the
 *                   TP per-layer cost the rest of the suite measures), so the
 *                   PP handoff can be compared to the collective it replaces.
 *
 * Like the rest of the suite this binary makes ZERO MPI calls: mpiexec only
 * places it; it reconstructs every peer's VCQ ID from tofu_topo.txt coordinates
 * via utofu_construct_vcq_id() (written once by tofu_topo_helper). A retried
 * fan-in/fan-out bootstrap then a tight barrier separate the timed phases (only
 * a chain subset, or just two ranks for ping-pong, are active per phase, so the
 * idle ranks must be parked -- hence an explicit barrier, unlike the
 * all-ranks-every-iter collectives).
 *
 * Build (NO -lmpi, no OpenMP). Native A64FX node:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall \
 *       -o pp_handoff_bench pp_handoff_bench.c -ltofucom
 *
 * Run (after tofu_topo_helper, 1 proc/node):
 *   PP_BATCH=1   mpiexec -np 12 ./pp_handoff_bench   # decode regime (~12 KiB hop)
 *   PP_BATCH=256 mpiexec -np 12 ./pp_handoff_bench   # training microbatch (~3 MiB)
 * stdout is swallowed by mpiexec -> read rank-0's pp_log_<coords>.txt.
 *
 * Tunables (env):
 *   PP_HID    hidden size                    (default 6144)
 *   PP_ABYTES activation bytes/elem (bf16=2)  (default 2)
 *   PP_BATCH  tokens handed off per stage     (default 1)
 *   PP_FAR    partner rank for the far hop    (default N-1)
 *   PP_ITERS  timed handoffs                  (default 2000)
 *   PP_WARMUP untimed warmup handoffs         (default 200)
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
#define BENCH_STAG DEMO_STAG          /* predictable-STADD convention */
#define WAIT_TIMEOUT_SEC 30.0
#define TREE_NSTEP 16                 /* distinct recv slots for the tree all-reduce */
#define PP_MAGIC 0x9117EE00u          /* slot-head magic = base | sender rank */

/* ----- copied helpers (reducescatter_bench.c / ring_attn_bench.c conventions) ----- */
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
static int             N, MyRank, FarRank;
static char           *Region;
static size_t          SlotP, SlotB, SlotT;   /* activation slot, barrier slot, tree slot */
static size_t          Plen;                  /* activation bytes = HID*ABYTES*B          */
static size_t          BAR_BASE, TREE_BASE;
static utofu_vcq_hdl_t Vcq;
static utofu_stadd_t   Base;
static utofu_vcq_id_t  PeerVcq[MAX_NODES];
static utofu_stadd_t   PeerBase[MAX_NODES];
static const unsigned long FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
static uint64_t        Bt = 1;                /* monotonic barrier token (bootstrap = 1)  */
/* tree all-reduce shape (Rabenseifner, non-pow2) */
static int Pof2, Rem, NRounds, BcastSid, NewRank;

/* slot offsets within the region. Each remote-written slot sits on its own cache
 * line (DEMO_CACHE_LINE) so concurrent incoming Puts never alias a CPU-written line.
 *   send       : my staging buffer (CPU writes head+trailing seq, then Puts from it)
 *   frecv      : forward activation landing (written ONLY by stage MyRank-1)
 *   brecv      : backward gradient landing  (written ONLY by stage MyRank+1)
 *   bar_recv[s]: rank-0 barrier fan-in slot for child s (written only by s)
 *   bar_go     : barrier release slot       (written only by rank 0)
 *   tree_*     : recursive-doubling all-reduce baseline. */
static inline size_t send_off(void)       { return 0; }
static inline size_t frecv_off(void)      { return SlotP; }
static inline size_t brecv_off(void)      { return 2 * SlotP; }
static inline size_t bar_recv_off(int s)  { return BAR_BASE + (size_t)s * SlotB; }
static inline size_t bar_go_off(void)     { return BAR_BASE + (size_t)N * SlotB; }
static inline size_t tree_recv_off(int s) { return TREE_BASE + (size_t)s * SlotT; }
static inline size_t tree_send_off(void)  { return TREE_BASE + (size_t)TREE_NSTEP * SlotT; }

/* one Put (the ring_attn_bench BUSY-retry idiom); drain==1 polls local completion. */
static void put_issue(utofu_vcq_id_t pv, utofu_stadd_t s, utofu_stadd_t d, size_t len, int drain)
{
    int rc; void *cb;
    for (;;) { rc = utofu_put(Vcq, pv, s, d, len, 0, FLAGS, NULL);
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

/* write head=PP_MAGIC|MyRank and trailing seq into the send slot, then Put it. */
static void send_put(int dst, uint64_t dst_off, uint64_t seq, int drain)
{
    char *sb = Region + send_off();
    *(volatile uint64_t *)(sb)        = PP_MAGIC | (uint64_t)MyRank;
    *(volatile uint64_t *)(sb + Plen) = seq;
    put_issue(PeerVcq[dst], Base + send_off(), PeerBase[dst] + dst_off, Plen + 8, drain);
}

/* ---- barrier: fan-in to rank 0 then fan-out release; carries token Bt. ----
 * robust==1 retries the fan-in with usleep (the startup bootstrap, before all
 * ranks are guaranteed registered+running); robust==0 is a tight spin (warm). */
static void barrier(int robust)
{
    uint64_t t = ++Bt;
    if (MyRank == 0) {
        for (int s = 1; s < N; s++)
            wait_seq((volatile uint64_t *)(Region + bar_recv_off(s)), t, "barrier fan-in");
        for (int s = 1; s < N; s++) {                 /* children confirmed live -> single go */
            *(volatile uint64_t *)(Region + send_off()) = t;
            put_issue(PeerVcq[s], Base + send_off(), PeerBase[s] + bar_go_off(), 8, 1);
        }
    } else {
        volatile uint64_t *go = (volatile uint64_t *)(Region + bar_go_off());
        double ts = now_sec();
        do {
            *(volatile uint64_t *)(Region + send_off()) = t;
            put_issue(PeerVcq[0], Base + send_off(), PeerBase[0] + bar_recv_off(MyRank), 8, 1);
            if (!robust) { wait_seq(go, t, "barrier release"); break; }
            for (int a = 0; a < 50 && *go < t; a++) usleep(2000);
            if (now_sec() - ts > WAIT_TIMEOUT_SEC) die("bootstrap barrier timeout", -1);
        } while (*go < t);
    }
}

/* (P) ping-pong: rank 0 sends to partner.frecv; partner echoes to rank 0.brecv.
 * Only ranks 0 and `partner` act; rank 0 times the round trip. */
static void pingpong(int partner, uint64_t seq)
{
    if (MyRank == 0) {
        send_put(partner, frecv_off(), seq, 1);
        wait_seq((volatile uint64_t *)(Region + brecv_off() + Plen), seq, "pingpong pong");
    } else if (MyRank == partner) {
        wait_seq((volatile uint64_t *)(Region + frecv_off() + Plen), seq, "pingpong ping");
        send_put(0, brecv_off(), seq, 1);
    }
}

/* (C) chain forward+backward store-and-forward relay, no wraparound:
 *   fwd  stage0 -> 1 -> .. -> S-1   (each writes successor's frecv)
 *   bwd  S-1 -> .. -> 1 -> 0        (each writes predecessor's brecv)
 * rank 0 times the full 2(S-1)-hop round trip. Single-writer per slot:
 * frecv[r] only from r-1, brecv[r] only from r+1. */
static void chain_rt(uint64_t seq)
{
    if (MyRank == 0) {
        send_put(1, frecv_off(), seq, 1);                                   /* originate fwd */
        wait_seq((volatile uint64_t *)(Region + brecv_off() + Plen), seq, "chain return");
    } else if (MyRank == N - 1) {
        wait_seq((volatile uint64_t *)(Region + frecv_off() + Plen), seq, "chain sink");
        send_put(N - 2, brecv_off(), seq, 1);                               /* turnaround */
    } else {
        wait_seq((volatile uint64_t *)(Region + frecv_off() + Plen), seq, "chain fwd");
        send_put(MyRank + 1, frecv_off(), seq, 1);                          /* forward fwd */
        wait_seq((volatile uint64_t *)(Region + brecv_off() + Plen), seq, "chain bwd");
        send_put(MyRank - 1, brecv_off(), seq, 1);                          /* forward bwd */
    }
}

/* (T) recursive-doubling all-reduce over the FULL activation (TP per-layer cost;
 * the decode_estimate.py assumption). Copied from reducescatter_bench.c. */
#define TREE_PUT(prank, sid) \
    put_issue(PeerVcq[prank], Base + tree_send_off(), \
              PeerBase[prank] + tree_recv_off(sid), Plen + 8, 1)
#define TREE_WAIT(sid, w) \
    wait_seq((volatile uint64_t *)(Region + tree_recv_off(sid) + Plen), (uint64_t)(w), "tree wait")
static void tree_allreduce(uint64_t tok)
{
    *(volatile uint64_t *)(Region + tree_send_off() + Plen) = tok;
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
    long HID    = envl("PP_HID", 6144);
    long ABYTES = envl("PP_ABYTES", 2);
    long B      = envl("PP_BATCH", 1);
    long ITERS  = envl("PP_ITERS", 2000);
    long WARMUP = envl("PP_WARMUP", 200);

    /* ---- uTofu setup (tofu_put_demo conventions); single TNI -- a stage handoff
     * is one P2P over one torus link, so multi-TNI to the same dst can't help
     * (finding #5). ---- */
    utofu_tni_id_t *tni_ids = NULL; size_t num_tnis = 0;
    rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
    if (rc != UTOFU_SUCCESS) die("utofu_get_onesided_tnis", rc);
    if (num_tnis < 1) die("no onesided TNIs", -1);

    uint8_t my_coords[TOFU_NCOORDS] = {0};
    rc = utofu_query_my_coords(my_coords);
    if (rc != UTOFU_SUCCESS) die("utofu_query_my_coords", rc);
    {
        char name[64];
        snprintf(name, sizeof name, "pp_log_%u_%u_%u_%u_%u_%u.txt",
                 my_coords[0], my_coords[1], my_coords[2], my_coords[3], my_coords[4], my_coords[5]);
        g_log = fopen(name, "w");
    }

    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    N = read_topo(topo);
    MyRank = -1;
    for (int r = 0; r < N; r++) if (memcmp(topo[r], my_coords, TOFU_NCOORDS) == 0) MyRank = r;
    if (MyRank == -1) { fprintf(stderr, "my coords not in %s\n", TOPO_PATH); exit(1); }
    FarRank = (int)envl("PP_FAR", N - 1);
    if (FarRank <= 0 || FarRank >= N) FarRank = N - 1;

    /* ---- sizes. activation handed off = HID*ABYTES*B (NOT sharded -- PP sends
     * the whole hidden state). The tree baseline reduces the same full vector. ---- */
    Plen = (size_t)HID * (size_t)ABYTES * (size_t)B;
    Plen = (Plen + 7) & ~(size_t)7;
    SlotP = (Plen + 8 + (DEMO_CACHE_LINE - 1)) & ~(size_t)(DEMO_CACHE_LINE - 1);
    SlotB = DEMO_CACHE_LINE;
    SlotT = SlotP;
    BAR_BASE  = 3 * SlotP;
    TREE_BASE = BAR_BASE + (size_t)(N + 1) * SlotB;
    size_t region_sz = TREE_BASE + (size_t)(TREE_NSTEP + 1) * SlotT;
    if (posix_memalign((void **)&Region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign", -1);
    memset(Region, 0, region_sz);

    /* tree all-reduce shape (Rabenseifner non-power-of-2) */
    Pof2 = 1; while (Pof2 * 2 <= N) Pof2 *= 2;
    Rem = N - Pof2;
    NRounds = 0; for (int x = 1; x < Pof2; x <<= 1) NRounds++;
    BcastSid = NRounds + 1;
    if (MyRank < 2 * Rem) NewRank = (MyRank % 2 == 0) ? -1 : MyRank / 2;
    else                  NewRank = MyRank - Rem;

    /* single VCQ; register the region; reconstruct every peer's VCQ + STADD by
     * convention (zero runtime exchange) -- the reducescatter_bench pattern. */
    utofu_tni_id_t tni = tni_ids[0];
    rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &Vcq);
    if (rc != UTOFU_SUCCESS) die("utofu_create_vcq_with_cmp_id", rc);
    utofu_vcq_id_t my_real;
    rc = utofu_query_vcq_id(Vcq, &my_real);
    if (rc != UTOFU_SUCCESS) die("utofu_query_vcq_id", rc);
    {                                              /* VCQ self-check */
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
        rc = utofu_construct_vcq_id(topo[r], tni, DEMO_CQ_ID, DEMO_CMP_ID, &PeerVcq[r]);
        if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(peer)", rc);
        utofu_set_vcq_id_path(&PeerVcq[r], NULL);
        rc = utofu_query_stadd(PeerVcq[r], BENCH_STAG, &PeerBase[r]);
        if (rc != UTOFU_SUCCESS) die("utofu_query_stadd(peer)", rc);
    }
    free(tni_ids);

    if (MyRank == 0) {
        logmsg("=== pipeline-parallel handoff (no stage compute) ===\n");
        logmsg("nodes(stages)=%d  HID=%ld abytes=%ld B=%ld  activation=%zu B (%.1f KiB)\n",
               N, HID, ABYTES, B, Plen, Plen / 1024.0);
        logmsg("region=%.1f MiB  near=rank1  far=rank%d  ITERS=%ld WARMUP=%ld\n",
               region_sz / 1048576.0, FarRank, ITERS, WARMUP);
        logmsg("tree: Pof2=%d Rem=%d NRounds=%d (%d steps)\n",
               Pof2, Rem, NRounds, NRounds + (Rem ? 2 : 0));
    }

    barrier(1);                                    /* robust startup bootstrap */

    /* one-time correctness: chain heads carry the forwarding stage's id. */
    chain_rt(100);
    if (MyRank != 0) {
        uint64_t h = *(volatile uint64_t *)(Region + frecv_off());
        if (h != (PP_MAGIC | (uint64_t)(MyRank - 1)))
            die("chain fwd-head verify failed", -1);
    } else {
        uint64_t h = *(volatile uint64_t *)(Region + brecv_off());
        if (h != (PP_MAGIC | 1ull)) die("chain return-head verify failed", -1);
    }
    barrier(0);

    /* ---- timed phases. Each variant runs WARMUP untimed + ITERS timed on a
     * fresh, strictly-increasing seq range; rank 0 brackets the timed loop.
     * Idle ranks fall through the (empty) loop and park at the next barrier. ---- */
    double pp_near = 0, pp_far = 0, chain_us = 0, tree_us = 0;
    uint64_t base = 1000;

#define TIME_PP(partner, out_us)                                                 \
    do { for (long i = 0; i < WARMUP; i++) pingpong((partner), base + i);        \
         double _t0 = now_sec();                                                 \
         for (long i = 0; i < ITERS;  i++) pingpong((partner), base + WARMUP + i); \
         (out_us) = (now_sec() - _t0) / (double)ITERS * 1e6;                      \
         base += WARMUP + ITERS + 16; } while (0)
#define TIME_CHAIN(fn, out_us)                                                   \
    do { for (long i = 0; i < WARMUP; i++) fn(base + i);                         \
         double _t0 = now_sec();                                                 \
         for (long i = 0; i < ITERS;  i++) fn(base + WARMUP + i);                \
         (out_us) = (now_sec() - _t0) / (double)ITERS * 1e6;                     \
         base += WARMUP + ITERS + 16; } while (0)

    TIME_PP(1,       pp_near); barrier(0);
    TIME_PP(FarRank, pp_far);  barrier(0);
    TIME_CHAIN(chain_rt,      chain_us); barrier(0);
    TIME_CHAIN(tree_allreduce, tree_us); barrier(0);

    if (MyRank == 0) {
        double hops      = 2.0 * (double)(N - 1);  /* round-trip hop count   */
        double per_hop_c = chain_us / hops;        /* in-chain store-and-fwd */
        double fwd_chain = chain_us / 2.0;          /* decode (fwd only)      */
        double pp_hop_n  = pp_near / 2.0;           /* isolated hop, near     */
        double pp_hop_f  = pp_far  / 2.0;           /* isolated hop, far      */
        logmsg("\n-- per-handoff comm (us), %ld iters --\n", ITERS);
        logmsg("ping-pong near (rank0<->rank1)   round-trip=%.2f  per-hop=%.2f\n", pp_near, pp_hop_n);
        logmsg("ping-pong far  (rank0<->rank%-2d)  round-trip=%.2f  per-hop=%.2f  (dist x%.2f)\n",
               FarRank, pp_far, pp_hop_f, pp_hop_f / pp_hop_n);
        logmsg("chain round-trip (fwd+bwd, %.0f hops)=%.2f  in-chain per-hop=%.2f\n",
               hops, chain_us, per_hop_c);
        logmsg("chain forward only (DECODE, %d hops)=%.2f\n", N - 1, fwd_chain);
        logmsg("tree all-reduce (full vector, TP per-layer baseline)=%.2f\n", tree_us);
        logmsg("\n-- interpretation --\n");
        logmsg("PP decode handoff (%d-hop fwd chain)=%.2f us  vs ONE tree all-reduce=%.2f us  "
               "-> x%.2f\n", N - 1, fwd_chain, tree_us, fwd_chain / tree_us);
        logmsg("[PP comm is fixed at S-1 handoffs for the WHOLE model; TP pays "
               "2*num_layers all-reduces -- PP trades comm volume for the bubble]\n");
        logmsg("pipeline bubble fraction (S-1)/(M+S-1) with S=%d:  "
               "M=1 -> %.2f   M=8 -> %.2f   M=64 -> %.2f\n",
               N, (double)(N - 1) / (double)(1 + N - 1),
               (double)(N - 1) / (double)(8 + N - 1),
               (double)(N - 1) / (double)(64 + N - 1));
    }

    utofu_dereg_mem(Vcq, Base, 0);
    utofu_free_vcq(Vcq);
    free(Region);
    if (g_log) fclose(g_log);
    return 0;
}
