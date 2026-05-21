/*
 * moe_dispatch_bench - MPI-free uTofu micro-benchmark measuring the cost of one
 * MoE expert-dispatch ALL-TO-ALL on A64FX/Fugaku (12-node 2x3x2 in-unit torus).
 *
 * The ring_attn_* benches characterize the ALL-REDUCE half of distributed decode
 * comm (ring / recursive-doubling tree). The untested half is MoE expert
 * dispatch: with expert-parallelism the routed-expert hidden states must be
 * scattered to the rank owning each selected expert (dispatch) and the results
 * gathered back (combine) -- an all-to-all PERSONALIZED exchange, not an
 * all-reduce. tools/decode_estimate.py currently *assumes* MoE comm costs the
 * same as an attention all-reduce (COLLECTIVES_PER_LAYER=2 x tree-cost). This
 * bench measures the real pattern so that term can be validated or replaced.
 *
 * It is also the first bench here that sends to MANY DISTINCT destinations at
 * once -- the case where tni_stripe_bench's open question flips: striping one
 * hop's payload across 6 TNIs to the SAME peer did nothing (shared link), but
 * TNIs SHOULD parallelize across distinct destinations. All-to-all is the test.
 *
 * Scope: comm-pattern + roofline only. No expert SwiGLU compute is performed
 * (like ring_attn_bench, which performs no attention math). We move realistically
 * sized payloads and time them.
 *
 * Three transport variants over the same routing matrix:
 *   (a) NAIVE  : single TNI, N-1 pairwise Puts (rotate-by-s schedule), drain each.
 *   (b) MULTI  : distinct destinations round-robined over MOE_NTNI TNIs, all Puts
 *                issued before any TCQ drain so the TNIs run concurrently.
 *   (c) TREE   : recursive-doubling all-reduce (copied from ring_attn_bench) over
 *                payload HID*ABYTES*B -- exactly decode_estimate.py's assumption.
 *
 * Like the rest of the suite this binary makes ZERO MPI calls: mpiexec only
 * places it; it reconstructs every peer's VCQ ID from tofu_topo.txt coordinates
 * via utofu_construct_vcq_id() (written once by tofu_topo_helper). Every rank
 * derives the SAME N x N routing matrix from a shared seed, so senders and
 * receivers agree on per-pair byte sizes with no runtime exchange.
 *
 * Build (NO -lmpi, no OpenMP). Native A64FX node:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall \
 *       -o moe_dispatch_bench moe_dispatch_bench.c -ltofucom
 *
 * Run (after tofu_topo_helper, 1 proc/node):
 *   MOE_BATCH=1   MOE_NTNI=6 mpiexec -np 12 ./moe_dispatch_bench   # decode regime
 *   MOE_BATCH=256 MOE_NTNI=6 mpiexec -np 12 ./moe_dispatch_bench   # batched regime
 * stdout is swallowed by mpiexec -> read rank-0's moe_log_<coords>.txt.
 *
 * Tunables (env, defaults model GLM-5.1 GlmMoeDsa):
 *   MOE_E      routed experts                    (default 256)
 *   MOE_K      experts selected per token (top-k)(default 8)
 *   MOE_HID    hidden size                       (default 6144)
 *   MOE_ABYTES activation bytes/elem (bf16=2)    (default 2)
 *   MOE_BATCH  tokens routed this step           (default 1)
 *   MOE_ROUTE  "uniform" or "worst" (imbalance)  (default uniform)
 *   MOE_CAPFAC capacity factor on payload size   (default 1.0)
 *   MOE_NTNI   TNIs for the multi-TNI variant    (default 6)
 *   MOE_LAYERS MoE layers for whole-model scaling(default 75)
 *   MOE_ITERS  timed exchanges                   (default 2000)
 *   MOE_WARMUP untimed warmup exchanges          (default 200)
 *   MOE_SEED   routing PRNG seed                 (default 1)
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
#define TREE_NSTEP 24                 /* distinct recv slots for the tree all-reduce */
#define MOE_MAGIC 0xD15A7C00u         /* recv-head magic = base | source rank */

enum { BANK_DISP = 0, BANK_COMB = 1 };

/* ----- copied helpers (ring_attn_bench.c / tni_stripe_bench.c conventions) ----- */
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

/* ---- deterministic routing: every rank reproduces the SAME N x N count matrix
 * from MOE_SEED, so per-pair byte sizes agree with no runtime exchange.
 * cnt[s][d] = number of (token,expert) pairs source s routes to experts owned by
 * rank d, over B tokens x K top-k picks. ---- */
static uint64_t splitmix64(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void gen_count_matrix(uint64_t seed, int B, int K, int E, int N, int worst,
                             int cnt[][MAX_NODES])
{
    int epn = (E + N - 1) / N;                 /* experts per node (contiguous shard) */
    uint64_t st = seed ? seed : 1;
    for (int s = 0; s < N; s++) for (int d = 0; d < N; d++) cnt[s][d] = 0;
    /* worst-case: concentrate picks on the experts owned by the first ~N/4 ranks */
    int hot_ranks = (N + 3) / 4; if (hot_ranks < 1) hot_ranks = 1;
    int hot_experts = hot_ranks * epn; if (hot_experts > E) hot_experts = E;
    for (int s = 0; s < N; s++) {
        for (int b = 0; b < B; b++) {
            int picked[256], np = 0;           /* K distinct experts per token */
            while (np < K && np < 256) {
                int range = worst ? hot_experts : E;
                int e = (int)(splitmix64(&st) % (uint64_t)range);
                int dup = 0;
                for (int i = 0; i < np; i++) if (picked[i] == e) { dup = 1; break; }
                if (!dup) picked[np++] = e;
            }
            for (int i = 0; i < np; i++) {
                int owner = picked[i] / epn; if (owner >= N) owner = N - 1;
                cnt[s][owner]++;
            }
        }
    }
}

/* ----- file-scope state (single-threaded benchmark; globals keep the inner
 * loops readable, mirroring the macro-heavy style of the sibling benches) ----- */
static int            N, NTNI, MyRank;
static char          *Region;
static size_t         Slot, HBytes, PlenTree;
static utofu_vcq_hdl_t Vcq[MAX_TNI];
static utofu_stadd_t   Base[MAX_TNI];
static utofu_vcq_id_t  PeerVcq[MAX_TNI][MAX_NODES];
static utofu_stadd_t   PeerBase[MAX_TNI][MAX_NODES];
static const unsigned long FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
/* tree all-reduce layout (Rabenseifner, non-pow2), computed in main */
static int Pof2, Rem, NRounds, BcastSid, NewRank;

/* slot offsets within the registered region: [disp recv 0..N-1][comb recv 0..N-1]
 * [tree recv 0..TREE_NSTEP-1][send 0..N-1]; each slot its own cache line(s). */
static inline size_t disp_recv_off(int src) { return (size_t)(src) * Slot; }
static inline size_t comb_recv_off(int src) { return (size_t)(N + src) * Slot; }
static inline size_t tree_recv_off(int sid) { return (size_t)(2 * N + sid) * Slot; }
static inline size_t send_off(int dst)      { return (size_t)(2 * N + TREE_NSTEP + dst) * Slot; }
static inline size_t recv_off(int bank, int src) { return bank == BANK_DISP ? disp_recv_off(src) : comb_recv_off(src); }

/* payload bytes for a per-pair selection count, 8-byte aligned (seq follows). */
static inline size_t plen(int c) { return ((size_t)c * HBytes + 7) & ~(size_t)7; }

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

/* issue this rank's outgoing messages for one all-to-all phase.
 * multi==0: single TNI 0, drain each Put. multi==1: round-robin distinct
 * destinations over NTNI TNIs, issue all then drain per-TNI (concurrent links). */
static void emit(int multi, const int *send_cnt, int bank, uint64_t tok)
{
    int issued[MAX_TNI] = {0}, idx = 0;
    for (int s = 1; s < N; s++) {
        int dst = (MyRank + s) % N;
        if (send_cnt[dst] <= 0) continue;
        size_t L = plen(send_cnt[dst]);
        char *sb = Region + send_off(dst);
        *(volatile uint64_t *)(sb)     = MOE_MAGIC | (uint64_t)MyRank;  /* head: src id */
        *(volatile uint64_t *)(sb + L) = tok;                          /* trailing seq  */
        int k = multi ? (idx % NTNI) : 0;
        put_issue(Vcq[k], PeerVcq[k][dst], Base[k] + send_off(dst),
                  PeerBase[k][dst] + recv_off(bank, MyRank), L + 8, multi ? 0 : 1);
        if (multi) issued[k]++;
        idx++;
    }
    if (multi) for (int k = 0; k < NTNI; k++) drain_n(k, issued[k]);
}

/* spin until every nonzero source's message for token `tok` has landed. */
static void waitall(const int *recv_cnt, int bank, uint64_t tok)
{
    double ts = now_sec();
    for (int s = 0; s < N; s++) {
        if (s == MyRank || recv_cnt[s] <= 0) continue;
        volatile uint64_t *sq = (volatile uint64_t *)(Region + recv_off(bank, s) + plen(recv_cnt[s]));
        while (*sq < tok)
            if (now_sec() - ts > WAIT_TIMEOUT_SEC) die("waitall timeout", -1);
    }
}

/* one-time correctness check: every nonzero source's recv head holds its id. */
static int verify_recv(const int *recv_cnt, int bank)
{
    for (int s = 0; s < N; s++) {
        if (s == MyRank || recv_cnt[s] <= 0) continue;
        uint64_t h = *(volatile uint64_t *)(Region + recv_off(bank, s));
        if (h != (MOE_MAGIC | (uint64_t)s)) {
            logmsg("verify FAIL bank=%d src=%d got=0x%lx want=0x%lx\n",
                   bank, s, (unsigned long)h, (unsigned long)(MOE_MAGIC | (uint64_t)s));
            return 0;
        }
    }
    return 1;
}

/* recursive-doubling all-reduce over PlenTree payload (the decode_estimate.py
 * assumption). Structure copied from ring_attn_bench.c:559-607. */
#define TREE_PUT(prank, sid) \
    put_issue(Vcq[0], PeerVcq[0][prank], Base[0] + send_off(0), \
              PeerBase[0][prank] + tree_recv_off(sid), PlenTree + 8, 1)
#define TREE_WAIT(sid, w) do { \
        volatile uint64_t *_rs = (volatile uint64_t *)(Region + tree_recv_off(sid) + PlenTree); \
        double _ts = now_sec(); \
        while (*_rs < (uint64_t)(w)) if (now_sec() - _ts > WAIT_TIMEOUT_SEC) die("tree wait timeout", -1); \
    } while (0)
static void tree_allreduce(uint64_t tok)
{
    *(volatile uint64_t *)(Region + send_off(0) + PlenTree) = tok;
    if (MyRank < 2 * Rem) {                       /* pre-reduce fold */
        if (MyRank % 2 == 0) TREE_PUT(MyRank + 1, 0); else TREE_WAIT(0, tok);
    }
    if (NewRank != -1) {                          /* recursive doubling */
        for (int kk = 0; kk < NRounds; kk++) {
            int pnr = NewRank ^ (1 << kk);
            int pr  = (pnr < Rem) ? (pnr * 2 + 1) : (pnr + Rem);
            TREE_PUT(pr, kk + 1); TREE_WAIT(kk + 1, tok);
        }
    }
    if (MyRank < 2 * Rem) {                        /* broadcast back */
        if (MyRank % 2 == 0) TREE_WAIT(BcastSid, tok); else TREE_PUT(MyRank - 1, BcastSid);
    }
}

int main(void)
{
    int rc;
    long E      = envl("MOE_E", 256);
    long K      = envl("MOE_K", 8);
    long HID    = envl("MOE_HID", 6144);
    long ABYTES = envl("MOE_ABYTES", 2);
    long B      = envl("MOE_BATCH", 1);
    long CAPx100= (long)(100.0 * (getenv("MOE_CAPFAC") ? atof(getenv("MOE_CAPFAC")) : 1.0));
    long NMOE   = envl("MOE_LAYERS", 75);
    long ITERS  = envl("MOE_ITERS", 2000);
    long WARMUP = envl("MOE_WARMUP", 200);
    long SEED   = envl("MOE_SEED", 1);
    const char *route = getenv("MOE_ROUTE");
    int  worst  = (route && strcmp(route, "worst") == 0);
    NTNI        = (int)envl("MOE_NTNI", 6);

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
        snprintf(name, sizeof name, "moe_log_%u_%u_%u_%u_%u_%u.txt",
                 my_coords[0], my_coords[1], my_coords[2], my_coords[3], my_coords[4], my_coords[5]);
        g_log = fopen(name, "w");
    }

    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    N = read_topo(topo);
    MyRank = -1;
    for (int r = 0; r < N; r++) if (memcmp(topo[r], my_coords, TOFU_NCOORDS) == 0) MyRank = r;
    if (MyRank == -1) { fprintf(stderr, "my coords not in %s\n", TOPO_PATH); exit(1); }

    /* ---- routing matrix (identical on every rank) ---- */
    static int cnt[MAX_NODES][MAX_NODES];
    gen_count_matrix((uint64_t)SEED, (int)B, (int)K, (int)E, N, worst, cnt);
    /* apply capacity factor to per-pair counts (payload over-provisioning) */
    if (CAPx100 != 100)
        for (int s = 0; s < N; s++) for (int d = 0; d < N; d++)
            cnt[s][d] = (int)(((long)cnt[s][d] * CAPx100 + 99) / 100);

    /* dispatch: I send cnt[me][*], receive cnt[*][me].
     * combine:  send back cnt[*][me] (transpose), receive cnt[me][*]. */
    static int disp_send[MAX_NODES], disp_recv[MAX_NODES];
    for (int d = 0; d < N; d++) { disp_send[d] = cnt[MyRank][d]; disp_recv[d] = cnt[d][MyRank]; }
    int *comb_send = disp_recv, *comb_recv = disp_send;

    HBytes  = (size_t)HID * (size_t)ABYTES;
    PlenTree = ((size_t)HID * (size_t)ABYTES * (size_t)B + 7) & ~(size_t)7;
    int maxc = 0;
    for (int s = 0; s < N; s++) for (int d = 0; d < N; d++) if (cnt[s][d] > maxc) maxc = cnt[s][d];
    size_t max_payload = plen(maxc);
    if (PlenTree > max_payload) max_payload = PlenTree;
    Slot = (max_payload + 8 + (DEMO_CACHE_LINE - 1)) & ~(size_t)(DEMO_CACHE_LINE - 1);
    size_t region_sz = (size_t)(3 * N + TREE_NSTEP) * Slot;

    if (posix_memalign((void **)&Region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign", -1);
    memset(Region, 0, region_sz);

    /* one VCQ per TNI; register the same region in each (independent stadd space) --
     * the tni_stripe_bench.c:109-122 pattern, generalized to peer arrays for ALL ranks. */
    for (int k = 0; k < NTNI; k++) {
        utofu_tni_id_t tni = tni_ids[k];
        rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &Vcq[k]);
        if (rc != UTOFU_SUCCESS) die("utofu_create_vcq_with_cmp_id", rc);
        utofu_vcq_id_t my_real;
        rc = utofu_query_vcq_id(Vcq[k], &my_real);
        if (rc != UTOFU_SUCCESS) die("utofu_query_vcq_id", rc);
        /* VCQ self-check on TNI 0 (the cq/cmp convention reproduces our own id) */
        if (k == 0) {
            utofu_vcq_id_t conv;
            rc = utofu_construct_vcq_id(my_coords, tni, DEMO_CQ_ID, DEMO_CMP_ID, &conv);
            if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(self)", rc);
            utofu_vcq_id_t a = my_real, b = conv;
            utofu_set_vcq_id_path(&a, NULL); utofu_set_vcq_id_path(&b, NULL);
            if (a != b) die("VCQ self-check (cq_id convention wrong)", -1);
        }
        rc = utofu_reg_mem_with_stag(Vcq[k], Region, region_sz, BENCH_STAG, 0, &Base[k]);
        if (rc != UTOFU_SUCCESS) die("utofu_reg_mem_with_stag", rc);
        /* STADD self-check */
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

    /* tree all-reduce shape (Rabenseifner non-power-of-2) */
    Pof2 = 1; while (Pof2 * 2 <= N) Pof2 *= 2;
    Rem = N - Pof2;
    NRounds = 0; for (int x = 1; x < Pof2; x <<= 1) NRounds++;
    BcastSid = NRounds + 1;
    if (MyRank < 2 * Rem) NewRank = (MyRank % 2 == 0) ? -1 : MyRank / 2;
    else                  NewRank = MyRank - Rem;

    if (MyRank == 0) {
        long sel = B * K, distinct = 0; for (int d = 0; d < N; d++) if (cnt[0][d] > 0) distinct++;
        logmsg("=== MoE expert-dispatch all-to-all (no FFN compute) ===\n");
        logmsg("nodes=%d  E=%ld K=%ld experts/node=%ld  HID=%ld abytes=%ld  B=%ld  route=%s capfac=%.2f\n",
               N, E, K, (E + N - 1) / N, HID, ABYTES, B, worst ? "worst" : "uniform", CAPx100 / 100.0);
        logmsg("rank0: selections=%ld -> %ld distinct dst, max pair count=%d  H-vec=%zu B\n",
               sel, distinct, maxc, HBytes);
        logmsg("slot=%zu B  region=%.1f MiB  NTNI=%d  tree payload=%zu B (%d steps)\n",
               Slot, region_sz / 1048576.0, NTNI, PlenTree, NRounds + (Rem ? 2 : 0));
    }

    /* ---- bootstrap: every rank repeatedly Puts seq=1 to its nonzero dispatch
     * destinations until it has received seq>=1 from every nonzero source, so the
     * barrierless startup race is covered before timing. ---- */
    {
        double t0 = now_sec(); int got = 0;
        for (int a = 0; a < 400 && !got; a++) {
            emit(0, disp_send, BANK_DISP, 1);
            int all = 1;
            for (int s = 0; s < N; s++) {
                if (s == MyRank || disp_recv[s] <= 0) continue;
                volatile uint64_t *sq = (volatile uint64_t *)(Region + disp_recv_off(s) + plen(disp_recv[s]));
                if (*sq < 1) all = 0;
            }
            if (all && a >= 8) got = 1;
            usleep(20000);
            if (now_sec() - t0 > WAIT_TIMEOUT_SEC) break;
        }
        if (!got) die("dispatch bootstrap timeout", -1);
    }

    uint64_t tok = 2;   /* bootstrap used seq 1 */

    /* one-time correctness pass (dispatch + combine), heads carry source ids */
    { emit(0, disp_send, BANK_DISP, tok); waitall(disp_recv, BANK_DISP, tok);
      if (!verify_recv(disp_recv, BANK_DISP)) die("dispatch verify failed", -1); tok++; }
    { emit(0, comb_send, BANK_COMB, tok); waitall(comb_recv, BANK_COMB, tok);
      if (!verify_recv(comb_recv, BANK_COMB)) die("combine verify failed", -1); tok++; }

    /* ---- timed phases. Each phase: WARMUP untimed then ITERS timed. ---- */
#define TIME_A2A(multi, send_cnt, recv_cnt, bank, out_us)                       \
    do {                                                                        \
        for (long i = 0; i < WARMUP; i++) { emit((multi), (send_cnt), (bank), tok); \
            waitall((recv_cnt), (bank), tok); tok++; }                          \
        double _t0 = now_sec();                                                 \
        for (long i = 0; i < ITERS; i++) { emit((multi), (send_cnt), (bank), tok); \
            waitall((recv_cnt), (bank), tok); tok++; }                          \
        (out_us) = (now_sec() - _t0) / (double)ITERS * 1e6;                     \
    } while (0)

    double disp_naive = 0, disp_tni = 0, comb_naive = 0, comb_tni = 0, tree_us = 0;
    TIME_A2A(0, disp_send, disp_recv, BANK_DISP, disp_naive);
    TIME_A2A(1, disp_send, disp_recv, BANK_DISP, disp_tni);
    TIME_A2A(0, comb_send, comb_recv, BANK_COMB, comb_naive);
    TIME_A2A(1, comb_send, comb_recv, BANK_COMB, comb_tni);

    /* tree all-reduce baseline (variant c) -- pass a plain var (macro multi-evals) */
    { for (long i = 0; i < WARMUP; i++) { uint64_t w = tok++; tree_allreduce(w); }
      double _t0 = now_sec();
      for (long i = 0; i < ITERS;  i++) { uint64_t w = tok++; tree_allreduce(w); }
      tree_us = (now_sec() - _t0) / (double)ITERS * 1e6; }

    if (MyRank == 0) {
        /* total bytes moved per all-to-all phase = sum of all pair payloads */
        double disp_bytes = 0, tree_bytes = (double)N * (PlenTree);
        for (int s = 0; s < N; s++) for (int d = 0; d < N; d++) if (cnt[s][d] > 0) disp_bytes += plen(cnt[s][d]);
        double dc_naive = disp_naive + comb_naive;
        double dc_tni   = disp_tni   + comb_tni;
        double dc_best  = dc_tni < dc_naive ? dc_tni : dc_naive;
        logmsg("\n-- per-MoE-layer comm (us), %ld iters --\n", ITERS);
        logmsg("dispatch  naive=%.2f  multiTNI=%.2f  (speedup x%.2f)\n",
               disp_naive, disp_tni, disp_naive / disp_tni);
        logmsg("combine   naive=%.2f  multiTNI=%.2f  (speedup x%.2f)\n",
               comb_naive, comb_tni, comb_naive / comb_tni);
        logmsg("dispatch+combine  naive=%.2f  multiTNI=%.2f  best=%.2f\n", dc_naive, dc_tni, dc_best);
        logmsg("tree all-reduce (decode_estimate assumption)=%.2f  -> 2x=%.2f\n", tree_us, 2 * tree_us);
        logmsg("RATIO (dispatch+combine_best)/(2*tree) = %.2f  [>1 roofline UNDER-estimates MoE comm]\n",
               dc_best / (2 * tree_us));
        logmsg("agg BW: dispatch naive=%.1f multiTNI=%.1f GB/s  (moved %.2f MiB/phase)\n",
               disp_bytes / (disp_naive * 1e-6) / 1e9, disp_bytes / (disp_tni * 1e-6) / 1e9,
               disp_bytes / 1048576.0);
        (void)tree_bytes;
        logmsg("whole-model MoE comm (x%ld layers): best=%.1f us/token\n", NMOE, dc_best * NMOE);
    }

    for (int k = 0; k < NTNI; k++) { utofu_dereg_mem(Vcq[k], Base[k], 0); utofu_free_vcq(Vcq[k]); }
    free(Region);
    if (g_log) fclose(g_log);
    return 0;
}
