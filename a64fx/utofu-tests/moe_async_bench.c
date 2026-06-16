/*
 * moe_async_bench - MPI-free uTofu + OpenMP micro-benchmark: does overlapping the
 * MoE expert-dispatch ALL-TO-ALL with the expert FFN compute HIDE the comm on
 * A64FX/Fugaku (12-node 2x3x2 in-unit torus)?
 *
 * This is the MoE analogue of ring_attn_async, which showed the across-step async
 * comm driver is the suite's biggest win for the ATTENTION all-reduce half of
 * decode comm (-24..-62% @16K). The OTHER half -- MoE dispatch+combine all-to-all
 * (moe_dispatch_bench, finding #10) -- was never tested for overlap. Here we test
 * it against the expert SwiGLU compute it sandwiches.
 *
 * Crucially the compute is REAL (finding #7's hard lesson: a compute proxy was
 * 15-35x wrong, so the comm/compute ratio that decides overlap MUST be measured,
 * not modelled). Each step the compute threads run a genuine bf16 SwiGLU GEMV over
 * real per-expert weight buffers streamed from HBM -- the decode regime is weight-
 * read-bound, so the expert FFN is a ~56 MB/expert HBM stream, not arithmetic.
 *
 * The MoE pipeline per decode step (one MoE layer) is
 *     dispatch (all-to-all, comm) -> expert FFN (compute) -> combine (all-to-all).
 * The async driver runs, on thread 0 (the comm driver), the combine of the PREVIOUS
 * step and the dispatch of the NEXT step, while threads 1..nct compute the CURRENT
 * step's experts -- steady-state per-step cost = max(dispatch+combine, compute),
 * exactly the ring_attn_async overlap model with comm = dispatch+combine.
 *
 * Unlike the restructured ring-attention kernel, the expert FFN has an INTERNAL
 * cross-thread dependency (gate/up rows -> SwiGLU -> down rows), so it needs a
 * barrier MID-compute. That barrier must be over the COMPUTE THREADS ONLY (it
 * excludes the comm-driver thread 0, which is off issuing Puts) -- the precise
 * constraint finding #9 identified for overlap. We use a dedicated sense-reversing
 * barrier (CBAR) over nct threads for the gu->swiglu->down syncs, and the global
 * STEPBAR (HW libhwb or flat OMP, like ring_attn_async) only at step boundaries.
 *
 * Comm is the multi-TNI distinct-destination dispatch+combine (the finding-#10
 * winner): one VCQ per TNI, distinct dsts round-robined over MOE_NTNI TNIs, all
 * Puts issued before any TCQ drain. Only thread 0 ever touches uTofu.
 *
 * Phases reported per decode step (us), mirroring ring_attn_async:
 *   compute-only   experts only (nct threads), tid0 idle
 *   comm-only      dispatch+combine (multi-TNI) only, compute idle
 *   SERIAL         dispatch -> compute -> combine (3 global barriers)
 *   ASYNC          across-step: compute(cur) || combine(prev)+dispatch(next)
 * plus the model max(compute,comm), the overlap efficiency, and a tree-all-reduce
 * reference (the decode_estimate.py per-collective term).
 *
 * Like the rest of the suite this binary makes ZERO MPI calls (mpiexec only places
 * it); every peer VCQ id is reconstructed from tofu_topo.txt via
 * utofu_construct_vcq_id(), and every rank derives the same N x N routing matrix
 * from MOE_SEED, so per-pair byte sizes agree with no runtime exchange.
 *
 * Build (native A64FX node, OpenMP + libhwb):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -Wall \
 *       -o moe_async_bench moe_async_bench.c -ltofucom -lhwb
 * Run (after tofu_topo_helper, 1 proc/node, 48 threads/proc):
 *   MOE_BATCH=1   TF_HW_BARRIER=1 mpiexec -np 12 ./moe_async_bench   # decode
 *   MOE_BATCH=256 TF_HW_BARRIER=1 mpiexec -np 12 ./moe_async_bench   # batched
 * stdout is swallowed by mpiexec -> read rank-0's moea_log_<coords>.txt.
 *
 * Tunables (env, defaults model a GLM-5.1-class MoE):
 *   MOE_E      routed experts                    (default 256)
 *   MOE_K      experts selected per token        (default 8)
 *   MOE_HID    hidden size H                      (default 6144)
 *   MOE_INTER  expert intermediate size I         (default 1536)
 *   MOE_ABYTES dispatch activation bytes/elem     (default 2, bf16)
 *   MOE_BATCH  tokens routed this step            (default 1)
 *   MOE_ROUTE  "uniform" or "worst"               (default uniform)
 *   MOE_NTNI   TNIs for the multi-TNI all-to-all  (default 6)
 *   MOE_LAYERS MoE layers for whole-model scaling (default 75)
 *   MOE_ITERS  timed steps                        (default 300)
 *   MOE_WARMUP untimed warmup steps               (default 30)
 *   MOE_SEED   routing PRNG seed                  (default 1)
 *   TF_HW_BARRIER 1=force libhwb HW barrier, 0=flat OMP (default auto)
 */
#define _GNU_SOURCE
#include <arm_sve.h>
#include <math.h>
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
#include <omp.h>

#include "tofu_demo.h"

#define MAX_NODES 128   /* was 64; bumped for DS4P 96/112/128-node overlap validation */
#define MAX_TNI   6
#define BENCH_STAG DEMO_STAG
#define WAIT_TIMEOUT_SEC 30.0
#define TREE_NSTEP 24
#define MOE_MAGIC 0xD15A7C00u

enum { BANK_DISP = 0, BANK_COMB = 1 };

/* ---- libhwb (Fujitsu /lib64/libhwb.so): per-CMG EL0 BST hardware barrier ---- */
extern int  vhbm_bar_init(uint64_t core_bitmask);
extern int  vhbm_bar_assign(int bd_mask, void *bb_hint);
extern void vhbm_bar(long bb);
extern int  vhbm_bar_unassign(int bd_mask);

/* ----- copied helpers (ring_attn / moe_dispatch_bench conventions) ----- */
static FILE *g_log = NULL;
static void logmsg(const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    if (g_log) { va_list a2; va_copy(a2, ap); vfprintf(g_log, fmt, a2); fflush(g_log); va_end(a2); }
    vfprintf(stdout, fmt, ap); fflush(stdout);
    va_end(ap);
}
static void die(const char *what, int rc) { logmsg("FATAL: %s (rc=%d)\n", what, rc); exit(1); }
static double now_sec(void)
{ struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9; }
static long envl(const char *n, long d) { const char *v = getenv(n); return (v && *v) ? strtol(v, NULL, 0) : d; }

static int read_topo(uint8_t coords[][TOFU_NCOORDS])
{
    FILE *f = fopen(TOPO_PATH, "r");
    if (!f) { perror("open " TOPO_PATH); fprintf(stderr, "  (run tofu_topo_helper first)\n"); exit(1); }
    int n = 0; char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (n >= MAX_NODES) { fprintf(stderr, "too many nodes\n"); exit(1); }
        unsigned r, c[TOFU_NCOORDS];
        if (sscanf(line, "%u %u %u %u %u %u %u", &r, &c[0],&c[1],&c[2],&c[3],&c[4],&c[5]) != 7) continue;
        if ((int)r != n) { fprintf(stderr, "%s ranks out of order\n", TOPO_PATH); exit(1); }
        for (int k = 0; k < TOFU_NCOORDS; k++) coords[n][k] = (uint8_t)c[k];
        n++;
    }
    fclose(f);
    if (n < 2) { fprintf(stderr, "%s lists %d node(s); need >= 2\n", TOPO_PATH, n); exit(1); }
    return n;
}

/* deterministic routing matrix (identical on every rank), as moe_dispatch_bench. */
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
    int epn = (E + N - 1) / N;
    uint64_t st = seed ? seed : 1;
    for (int s = 0; s < N; s++) for (int d = 0; d < N; d++) cnt[s][d] = 0;
    int hot_ranks = (N + 3) / 4; if (hot_ranks < 1) hot_ranks = 1;
    int hot_experts = hot_ranks * epn; if (hot_experts > E) hot_experts = E;
    for (int s = 0; s < N; s++) {
        for (int b = 0; b < B; b++) {
            int picked[256], np = 0;
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

/* ===================== uTofu transport state (file-global) ===================== */
static int N, NTNI, MyRank;
static char *Region;
static size_t Slot, HBytes, PlenTree;
static utofu_vcq_hdl_t Vcq[MAX_TNI];
static utofu_stadd_t   Base[MAX_TNI];
static utofu_vcq_id_t  PeerVcq[MAX_TNI][MAX_NODES];
static utofu_stadd_t   PeerBase[MAX_TNI][MAX_NODES];
static const unsigned long FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
static int Pof2, Rem, NRounds, BcastSid, NewRank;

/* region: 2 parity generations of [disp recv 0..N-1][comb recv 0..N-1] (4N slots),
 * then [tree recv 0..TREE_NSTEP-1], then [send staging 0..N-1]. Each slot its own
 * cache line(s). Double-buffering lets the async pipeline keep two steps in flight. */
static inline size_t recv_off(int gen, int bank, int src)
{ return (size_t)(gen * 2 * N + bank * N + src) * Slot; }
static inline size_t tree_recv_off(int sid) { return (size_t)(4 * N + sid) * Slot; }
static inline size_t send_off(int dst)       { return (size_t)(4 * N + TREE_NSTEP + dst) * Slot; }

static inline size_t plen(int c) { return ((size_t)c * HBytes + 7) & ~(size_t)7; }

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

/* one all-to-all phase (dispatch or combine). multi-TNI distinct-destination
 * (finding #10): round-robin dsts over NTNI TNIs, issue all then drain per TNI. */
static void emit(const int *send_cnt, int gen, int bank, uint64_t tok)
{
    int issued[MAX_TNI] = {0}, idx = 0;
    for (int s = 1; s < N; s++) {
        int dst = (MyRank + s) % N;
        if (send_cnt[dst] <= 0) continue;
        size_t L = plen(send_cnt[dst]);
        char *sb = Region + send_off(dst);
        *(volatile uint64_t *)(sb)     = MOE_MAGIC | (uint64_t)MyRank;
        *(volatile uint64_t *)(sb + L) = tok;
        int k = idx % NTNI;
        put_issue(Vcq[k], PeerVcq[k][dst], Base[k] + send_off(dst),
                  PeerBase[k][dst] + recv_off(gen, bank, MyRank), L + 8, 0);
        issued[k]++; idx++;
    }
    for (int k = 0; k < NTNI; k++) drain_n(k, issued[k]);
}
static void waitall(const int *recv_cnt, int gen, int bank, uint64_t tok)
{
    double ts = now_sec();
    for (int s = 0; s < N; s++) {
        if (s == MyRank || recv_cnt[s] <= 0) continue;
        volatile uint64_t *sq = (volatile uint64_t *)(Region + recv_off(gen, bank, s) + plen(recv_cnt[s]));
        while (*sq < tok) if (now_sec() - ts > WAIT_TIMEOUT_SEC) die("waitall timeout", -1);
    }
}
static int verify_recv(const int *recv_cnt, int gen, int bank)
{
    for (int s = 0; s < N; s++) {
        if (s == MyRank || recv_cnt[s] <= 0) continue;
        uint64_t h = *(volatile uint64_t *)(Region + recv_off(gen, bank, s));
        if (h != (MOE_MAGIC | (uint64_t)s)) {
            logmsg("verify FAIL bank=%d src=%d got=0x%lx want=0x%lx\n",
                   bank, s, (unsigned long)h, (unsigned long)(MOE_MAGIC | (uint64_t)s));
            return 0;
        }
    }
    return 1;
}

/* recursive-doubling all-reduce over PlenTree (decode_estimate.py reference). */
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
    if (MyRank < 2 * Rem) { if (MyRank % 2 == 0) TREE_PUT(MyRank + 1, 0); else TREE_WAIT(0, tok); }
    if (NewRank != -1) for (int kk = 0; kk < NRounds; kk++) {
        int pnr = NewRank ^ (1 << kk);
        int pr  = (pnr < Rem) ? (pnr * 2 + 1) : (pnr + Rem);
        TREE_PUT(pr, kk + 1); TREE_WAIT(kk + 1, tok);
    }
    if (MyRank < 2 * Rem) { if (MyRank % 2 == 0) TREE_WAIT(BcastSid, tok); else TREE_PUT(MyRank - 1, BcastSid); }
}

/* ===================== expert FFN compute (real bf16 SwiGLU GEMV) ===================== */
/* Per local expert e: Wgu[e] = [2I][H] bf16 (gate rows 0..I-1, up rows I..2I-1),
 * Wdn[e] = [H][I] bf16. Decode is weight-read-bound: y=Wx streams the full weight
 * matrix from HBM per expert, so per-step compute ~ nactive * (2I*H + H*I) * 2 B.
 * Rows are split cyclically across the nct compute threads; first-touched the same
 * way so each thread's rows are NUMA-local to its CMG (-> aggregate node BW). */
static int   HID_g, INTER_g, NLocal, NActive, Nct;
static uint16_t **Wgu, **Wdn;    /* [NLocal][...] */
static float *Gu, *Hh, *Xin;     /* shared: Gu[2I], Hh[I], Xin[H] */
static double *Csink;            /* [nthr] per-thread checksum sink (no false-sharing pad) */
static double g_ct_A = 0, g_ct_C = 0;   /* DIAG: lane-0 expert_compute wall in phase A vs C */
static double g_disp_C = 0, g_comb_C = 0;  /* DIAG: tid0 dispatch/combine spans in phase C */

/* compute-threads-only sense-reversing barrier (excludes comm-driver tid0). */
static volatile int g_ccount = 0; static char _cpad1[60] __attribute__((unused));
static volatile int g_csense = 0; static char _cpad2[60] __attribute__((unused));
#define CBAR(ls) do { \
        int _my = !(*(ls)); \
        if (__sync_add_and_fetch(&g_ccount, 1) == Nct) { g_ccount = 0; __sync_synchronize(); g_csense = _my; } \
        else { while (g_csense != _my) __asm__ volatile("yield"); } \
        *(ls) = _my; \
    } while (0)

/* bf16 (top 16 bits of fp32) dot product against fp32 Xin over `n` elements. */
static inline float bf16_dot(const uint16_t *w, const float *x, int n)
{
    const svbool_t pg = svptrue_b32();
    const int VL = (int)svcntw();
    svfloat32_t acc = svdup_f32(0.0f);
    for (int k = 0; k < n; k += VL) {
        svuint32_t wu = svld1uh_u32(pg, w + k);                 /* zero-extend halfword -> 32b */
        svfloat32_t wf = svreinterpret_f32_u32(svlsl_n_u32_x(pg, wu, 16));
        svfloat32_t xf = svld1_f32(pg, x + k);
        acc = svmla_f32_x(pg, acc, wf, xf);
    }
    return svaddv_f32(pg, acc);
}

/* one decode step's expert FFN, executed by compute thread `lane` (0..Nct-1).
 * Cooperative row-split with two CBARs (gu->swiglu->down). M=1 per active expert
 * (decode is weight-read-bound; the active-expert count, not M, scales with batch). */
static void expert_compute(int lane, int *ls_c)
{
    double sink = 0.0;
    for (int e = 0; e < NActive; e++) {
        const uint16_t *Wg = Wgu[e];
        for (int r = lane; r < 2 * INTER_g; r += Nct)
            Gu[r] = bf16_dot(Wg + (size_t)r * HID_g, Xin, HID_g);
        CBAR(ls_c);
        for (int i = lane; i < INTER_g; i += Nct) {
            float g = Gu[i];
            float s = g / (1.0f + expf(-g));        /* SiLU */
            Hh[i] = s * Gu[INTER_g + i];            /* * up */
        }
        CBAR(ls_c);
        const uint16_t *Wd = Wdn[e];
        for (int r = lane; r < HID_g; r += Nct)
            sink += bf16_dot(Wd + (size_t)r * INTER_g, Hh, INTER_g);
    }
    Csink[lane + 1] += sink;
}

/* ===================== HW barrier (libhwb) state + STEPBAR ===================== */
static int g_hw_enabled = 0;
static int g_hwbar_bd = 0, g_hwbar_tpc = 12, g_hwbar_ncmg = 4;
static volatile int g_lcount = 0; static char _pad1[60] __attribute__((unused));
static volatile int g_lsense = 0; static char _pad2[60] __attribute__((unused));
static volatile int g_join_count = 0, g_assign_failed = 0;
#define STEPBAR(tid, bb, ls4)                                                  \
    do {                                                                       \
        if (g_hw_enabled) {                                                    \
            vhbm_bar(bb);                                                       \
            if ((tid) % g_hwbar_tpc == 0) {                                    \
                int _my = !(*(ls4));                                           \
                if (__sync_add_and_fetch(&g_lcount, 1) == g_hwbar_ncmg) {      \
                    g_lcount = 0; __sync_synchronize(); g_lsense = _my;        \
                } else { while (g_lsense != _my) __asm__ volatile("yield"); }  \
                *(ls4) = _my;                                                  \
            }                                                                  \
            vhbm_bar(bb);                                                       \
        } else { _Pragma("omp barrier") }                                      \
    } while (0)

int main(void)
{
    int rc;
    long E      = envl("MOE_E", 256);
    long K      = envl("MOE_K", 8);
    long HID    = envl("MOE_HID", 6144);
    long INTER  = envl("MOE_INTER", 1536);
    long ABYTES = envl("MOE_ABYTES", 2);
    long B      = envl("MOE_BATCH", 1);
    long NMOE   = envl("MOE_LAYERS", 75);
    long ITERS  = envl("MOE_ITERS", 300);
    long WARMUP = envl("MOE_WARMUP", 30);
    long SEED   = envl("MOE_SEED", 1);
    const char *route = getenv("MOE_ROUTE");
    int  worst  = (route && strcmp(route, "worst") == 0);
    NTNI        = (int)envl("MOE_NTNI", 6);
    if ((HID % (long)svcntw()) || (INTER % (long)svcntw()))
        die("MOE_HID and MOE_INTER must be multiples of svcntw() (16)", -1);

    /* ---- uTofu setup ---- */
    utofu_tni_id_t *tni_ids = NULL; size_t num_tnis = 0;
    rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
    if (rc != UTOFU_SUCCESS) die("utofu_get_onesided_tnis", rc);
    if (NTNI < 1) NTNI = 1;
    if ((size_t)NTNI > num_tnis) NTNI = (int)num_tnis;
    if (NTNI > MAX_TNI) NTNI = MAX_TNI;

    uint8_t my_coords[TOFU_NCOORDS] = {0};
    if (utofu_query_my_coords(my_coords) != UTOFU_SUCCESS) die("query_my_coords", -1);
    { char nm[64];
      snprintf(nm, sizeof nm, "moea_log_%u_%u_%u_%u_%u_%u.txt",
               my_coords[0],my_coords[1],my_coords[2],my_coords[3],my_coords[4],my_coords[5]);
      g_log = fopen(nm, "w"); }

    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    N = read_topo(topo);
    MyRank = -1;
    for (int r = 0; r < N; r++) if (memcmp(topo[r], my_coords, TOFU_NCOORDS) == 0) MyRank = r;
    if (MyRank < 0) die("my coords not in topo", -1);

    /* ---- routing matrix + per-pair counts ---- */
    static int cnt[MAX_NODES][MAX_NODES];
    gen_count_matrix((uint64_t)SEED, (int)B, (int)K, (int)E, N, worst, cnt);
    static int disp_send[MAX_NODES], disp_recv[MAX_NODES];
    for (int d = 0; d < N; d++) { disp_send[d] = cnt[MyRank][d]; disp_recv[d] = cnt[d][MyRank]; }
    int *comb_send = disp_recv, *comb_recv = disp_send;     /* combine = transpose */

    HBytes  = (size_t)HID * (size_t)ABYTES;
    PlenTree = ((size_t)HID * (size_t)ABYTES * (size_t)B + 7) & ~(size_t)7;
    int maxc = 0;
    for (int s = 0; s < N; s++) for (int d = 0; d < N; d++) if (cnt[s][d] > maxc) maxc = cnt[s][d];
    size_t max_payload = plen(maxc);
    if (PlenTree > max_payload) max_payload = PlenTree;
    Slot = (max_payload + 8 + (DEMO_CACHE_LINE - 1)) & ~(size_t)(DEMO_CACHE_LINE - 1);
    size_t region_sz = (size_t)(4 * N + TREE_NSTEP + N) * Slot;
    if (posix_memalign((void **)&Region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign", -1);
    memset(Region, 0, region_sz);

    /* one VCQ per TNI; register the region in each (moe_dispatch_bench pattern). */
    for (int k = 0; k < NTNI; k++) {
        utofu_tni_id_t tni = tni_ids[k];
        rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &Vcq[k]);
        if (rc != UTOFU_SUCCESS) die("utofu_create_vcq_with_cmp_id", rc);
        utofu_vcq_id_t my_real;
        rc = utofu_query_vcq_id(Vcq[k], &my_real);
        if (rc != UTOFU_SUCCESS) die("utofu_query_vcq_id", rc);
        if (k == 0) {
            utofu_vcq_id_t conv;
            rc = utofu_construct_vcq_id(my_coords, tni, DEMO_CQ_ID, DEMO_CMP_ID, &conv);
            if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(self)", rc);
            utofu_vcq_id_t a = my_real, b = conv;
            utofu_set_vcq_id_path(&a, NULL); utofu_set_vcq_id_path(&b, NULL);
            if (a != b) die("VCQ self-check", -1);
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

    /* tree all-reduce shape (Rabenseifner non-power-of-2) */
    Pof2 = 1; while (Pof2 * 2 <= N) Pof2 *= 2;
    Rem = N - Pof2;
    NRounds = 0; for (int x = 1; x < Pof2; x <<= 1) NRounds++;
    BcastSid = NRounds + 1;
    if (MyRank < 2 * Rem) NewRank = (MyRank % 2 == 0) ? -1 : MyRank / 2;
    else                  NewRank = MyRank - Rem;

    /* ---- thread / CMG layout: thread 0 = comm driver, 1..nct compute ---- */
    enum { CMG_CORES = 12 };
    cpu_set_t allowed; sched_getaffinity(0, sizeof allowed, &allowed);
    int cores[256], ncore = 0;
    for (int c = 0; c < CPU_SETSIZE && ncore < 256; c++) if (CPU_ISSET(c, &allowed)) cores[ncore++] = c;
    int ncmg = ncore / CMG_CORES, cores_per = CMG_CORES;
    if (ncmg < 1) { ncmg = 1; cores_per = ncore < 1 ? 1 : ncore; }
    int nthr = ncmg * cores_per;
    int nct = nthr - 1;
    if (nct < 1) die("need >=2 cores", -1);
    Nct = nct; HID_g = (int)HID; INTER_g = (int)INTER;

    /* ---- local expert weights: NLocal experts, each [2I][H] + [H][I] bf16.
     * NActive = experts that receive >=1 token this step (round-robin over the
     * dispatch fan-in R = sum_s cnt[s][me]); decode is weight-read-bound so the
     * HBM stream scales with NActive. ---- */
    int epn = (E + N - 1) / N;
    NLocal = epn;
    long R = 0; for (int s = 0; s < N; s++) R += disp_recv[s];
    NActive = (int)(R < NLocal ? R : NLocal);
    if (NActive < 1 && R > 0) NActive = 1;
    Wgu = calloc(NLocal, sizeof *Wgu);
    Wdn = calloc(NLocal, sizeof *Wdn);
    size_t gu_elems = (size_t)2 * INTER * HID, dn_elems = (size_t)HID * INTER;
    for (int e = 0; e < NLocal; e++) {
        if (posix_memalign((void **)&Wgu[e], 2*1024*1024, gu_elems * sizeof(uint16_t)) != 0) die("memalign(Wgu)", -1);
        if (posix_memalign((void **)&Wdn[e], 2*1024*1024, dn_elems * sizeof(uint16_t)) != 0) die("memalign(Wdn)", -1);
    }
    Gu  = malloc((size_t)2 * INTER * sizeof(float));
    Hh  = malloc((size_t)INTER * sizeof(float));
    Xin = malloc((size_t)HID * sizeof(float));
    for (long i = 0; i < HID; i++) Xin[i] = 0.01f * (float)((i % 17) - 8);
    Csink = calloc(nthr, sizeof *Csink);

    /* ---- HW-barrier master init ---- */
    const char *hbe = getenv("TF_HW_BARRIER");
    int hb_explicit_on  = (hbe && hbe[0] == '1');
    int hb_explicit_off = (hbe && hbe[0] == '0');
    if (!hb_explicit_off && (nthr % ncmg) == 0) {
        uint64_t mask = 0;
        for (int t = 0; t < nthr; t++) mask |= (1ULL << cores[t]);
        int bd = vhbm_bar_init(mask);
        if (bd < 0) { if (hb_explicit_on) fprintf(stderr, "vhbm_bar_init failed (%d)\n", bd); }
        else { g_hwbar_bd = bd; g_hwbar_ncmg = ncmg; g_hwbar_tpc = cores_per; g_hw_enabled = 1; }
    }

    if (MyRank == 0) {
        long sel = B * K, distinct = 0; for (int d = 0; d < N; d++) if (cnt[0][d] > 0) distinct++;
        double wexp = (double)(2*INTER*HID + HID*INTER) * 2.0;     /* bytes/expert */
        logmsg("=== MoE async overlap: dispatch+combine all-to-all || expert SwiGLU ===\n");
        logmsg("nodes=%d threads=%d (1 comm-driver + %d compute, %d CMG x %d)\n", N, nthr, nct, ncmg, cores_per);
        logmsg("E=%ld K=%ld experts/node=%d  HID=%ld INTER=%ld abytes=%ld  B=%ld route=%s\n",
               E, K, epn, HID, INTER, ABYTES, B, worst ? "worst" : "uniform");
        logmsg("rank0: selections=%ld -> %ld distinct dst, max pair=%d  fan-in R=%ld  NActive=%d\n",
               sel, distinct, maxc, R, NActive);
        logmsg("weights/expert=%.1f MiB  active stream=%.1f MiB/step  NTNI=%d  barrier=%s\n",
               wexp/1048576.0, wexp*NActive/1048576.0, NTNI, g_hw_enabled ? "HW(libhwb)" : "flat-OMP");
    }

    /* ---- barrierless bootstrap: Put seq=1 to nonzero dispatch dsts (gen 0) ---- */
    {
        double t0 = now_sec(); int got = 0;
        for (int a = 0; a < 400 && !got; a++) {
            emit(disp_send, 0, BANK_DISP, 1);
            int all = 1;
            for (int s = 0; s < N; s++) {
                if (s == MyRank || disp_recv[s] <= 0) continue;
                volatile uint64_t *sq = (volatile uint64_t *)(Region + recv_off(0, BANK_DISP, s) + plen(disp_recv[s]));
                if (*sq < 1) all = 0;
            }
            if (all && a >= 8) got = 1;
            usleep(20000);
            if (now_sec() - t0 > WAIT_TIMEOUT_SEC) break;
        }
        if (!got) die("dispatch bootstrap timeout", -1);
    }

    uint64_t tok0 = 2;
    /* one-time correctness pass: dispatch then combine, heads carry source ids. */
    { emit(disp_send, 0, BANK_DISP, tok0); waitall(disp_recv, 0, BANK_DISP, tok0);
      if (!verify_recv(disp_recv, 0, BANK_DISP)) die("dispatch verify failed", -1); tok0++; }
    { emit(comb_send, 0, BANK_COMB, tok0); waitall(comb_recv, 0, BANK_COMB, tok0);
      if (!verify_recv(comb_recv, 0, BANK_COMB)) die("combine verify failed", -1); tok0++; }
    /* prime gen 0/1 dispatch so compute-only & serial have landed input. */
    emit(disp_send, 0, BANK_DISP, tok0); waitall(disp_recv, 0, BANK_DISP, tok0); tok0++;
    emit(disp_send, 1, BANK_DISP, tok0); waitall(disp_recv, 1, BANK_DISP, tok0); tok0++;

    double t_compute = 0, t_comm = 0, t_serial = 0, t_async = 0, tree_us = 0;
    volatile double chk = 0;

#pragma omp parallel num_threads(nthr)
    {
        int tid = omp_get_thread_num();
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(cores[tid], &s); sched_setaffinity(0, sizeof s, &s);
        long bb = 0; int ls4 = 0, lsc = 0;
        int lane = tid - 1;                          /* compute lane (tid>=1) */
        if (g_hw_enabled) { bb = vhbm_bar_assign(g_hwbar_bd, NULL); if (bb < 0) { g_assign_failed = 1; bb = 0; } }
        __sync_synchronize();
        __sync_add_and_fetch((int *)&g_join_count, 1);
#pragma omp barrier
#pragma omp master
        if (g_hw_enabled && g_assign_failed) { g_hw_enabled = 0; fprintf(stderr, "HW barrier join failed; flat\n"); }
#pragma omp barrier

        /* first-touch + init each thread's expert rows: NUMA-local placement AND
         * benign bf16 values (0x3c00 ~ 0.0078) so no uninitialized NaN/inf flows
         * through the SwiGLU. Cyclic rows interleave pages across CMGs -> node BW. */
        if (tid != 0) for (int e = 0; e < NLocal; e++) {
            for (int r = lane; r < 2 * INTER_g; r += Nct) {
                uint16_t *w = Wgu[e] + (size_t)r * HID_g;
                for (int k = 0; k < HID_g; k++) w[k] = 0x3c00;
            }
            for (int r = lane; r < HID_g; r += Nct) {
                uint16_t *w = Wdn[e] + (size_t)r * INTER_g;
                for (int k = 0; k < INTER_g; k++) w[k] = 0x3c00;
            }
        }
        STEPBAR(tid, bb, &ls4);

        uint64_t seq = tok0;          /* monotone; only tid0 advances it */

        /* ===== A) compute-only (nct threads); tid0 idle ===== */
        double tA = 0;
        for (long it = 0; it < WARMUP + ITERS; it++) {
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            if (it == WARMUP) tA = now_sec();
            { double _c0 = (tid == 1) ? now_sec() : 0;
              if (tid != 0) expert_compute(lane, &lsc);
              if (tid == 1 && it >= WARMUP) g_ct_A += now_sec() - _c0; }
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            if (it == WARMUP + ITERS - 1) t_compute = (now_sec() - tA) / ITERS;
        }

        /* ===== B) comm-only: dispatch+combine (multi-TNI), compute idle ===== */
        double tB = 0;
        for (long it = 0; it < WARMUP + ITERS; it++) {
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            {
                if (it == WARMUP) tB = now_sec();
                emit(disp_send, 0, BANK_DISP, seq);   waitall(disp_recv, 0, BANK_DISP, seq);   seq++;
                emit(comb_send, 0, BANK_COMB, seq);   waitall(comb_recv, 0, BANK_COMB, seq);   seq++;
            }
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            if (it == WARMUP + ITERS - 1) t_comm = (now_sec() - tB) / ITERS;
        }

        /* ===== C) SERIAL: dispatch -> compute -> combine (3 global barriers) ===== */
        double tC = 0;
        for (long it = 0; it < WARMUP + ITERS; it++) {
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            { if (it == WARMUP) tC = now_sec();
              double _d0 = now_sec();
              emit(disp_send, 0, BANK_DISP, seq); waitall(disp_recv, 0, BANK_DISP, seq); seq++;
              if (it >= WARMUP) g_disp_C += now_sec() - _d0; }
            STEPBAR(tid, bb, &ls4);
            { double _c0 = (tid == 1) ? now_sec() : 0;
              if (tid != 0) expert_compute(lane, &lsc);
              if (tid == 1 && it >= WARMUP) g_ct_C += now_sec() - _c0; }
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            { double _b0 = now_sec();
              emit(comb_send, 0, BANK_COMB, seq); waitall(comb_recv, 0, BANK_COMB, seq); seq++;
              if (it >= WARMUP) g_comb_C += now_sec() - _b0; }
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            if (it == WARMUP + ITERS - 1) t_serial = (now_sec() - tC) / ITERS;
        }

        /* ===== D) ASYNC across-step: compute(cur, gen p) on threads ||
         * combine(prev out) + dispatch(next, gen 1-p) on tid0. Steady-state
         * per-step = max(dispatch+combine, compute). 1 global barrier/step;
         * the gu->down dependency uses the compute-only CBAR (tid0 excluded). ===== */
        double tD = 0;
        for (long it = 0; it < WARMUP + ITERS; it++) {
            int p = (int)(it & 1);
#pragma omp master
            if (it == WARMUP) tD = now_sec();
            if (tid == 0) {
                /* combine the step that compute finished last iter (into gen p
                 * dispatch already landed), then dispatch the next iter (gen 1-p). */
                emit(comb_send, p, BANK_COMB, seq);     waitall(comb_recv, p, BANK_COMB, seq);     seq++;
                emit(disp_send, 1 - p, BANK_DISP, seq); waitall(disp_recv, 1 - p, BANK_DISP, seq); seq++;
            } else {
                expert_compute(lane, &lsc);
            }
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            if (it == WARMUP + ITERS - 1) { t_async = (now_sec() - tD) / ITERS; }
        }

        /* ===== tree all-reduce reference (tid0) ===== */
        double tT = 0;
        for (long it = 0; it < WARMUP + ITERS; it++) {
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            { if (it == WARMUP) tT = now_sec(); tree_allreduce(seq); seq++; }
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            if (it == WARMUP + ITERS - 1) tree_us = (now_sec() - tT) / ITERS;
        }

        if (g_hw_enabled) vhbm_bar_unassign(g_hwbar_bd);
    }
    for (int t = 1; t <= nct; t++) chk += Csink[t];

    /* DIAG: every rank logs its load + compute-only time to expose B=1 imbalance. */
    logmsg("RANK %d: fan-in R=%ld NActive=%d compute-only=%.1f us\n",
           MyRank, R, NActive, t_compute*1e6);

    if (MyRank == 0) {
        double cmp = t_compute*1e6, com = t_comm*1e6, ser = t_serial*1e6, asy = t_async*1e6, tr = tree_us*1e6;
        double mmax = cmp > com ? cmp : com;
        double disp_bytes = 0; for (int s = 0; s < N; s++) for (int d = 0; d < N; d++) if (cnt[s][d] > 0) disp_bytes += plen(cnt[s][d]);
        logmsg("\n--- measured per decode step (us), barrier=%s ---\n", g_hw_enabled ? "HW" : "flat-OMP");
        logmsg("compute-only (expert FFN, %dT) = %.1f\n", nct, cmp);
        logmsg("comm-only (dispatch+combine)   = %.1f   (multi-TNI, %.1f GB/s)\n",
               com, com > 0 ? 2.0*disp_bytes/(com*1e-6)/1e9 : 0.0);
        logmsg("SERIAL  (dispatch+compute+comb)= %.1f\n", ser);
        logmsg("ASYNC   (across-step overlap)  = %.1f\n", asy);
        logmsg("tree all-reduce (estimate ref) = %.1f   (2x = %.1f)\n", tr, 2*tr);
        logmsg("\n--- vs the model ---\n");
        logmsg("model serial = compute+comm    = %.1f\n", cmp + com);
        logmsg("model max()  = max(compute,comm)= %.1f\n", mmax);
        double ideal = ser - mmax;
        logmsg("ASYNC overlap efficiency = (serial-async)/(serial-max) = %.0f%%\n",
               ideal > 0 ? 100.0*(ser - asy)/ideal : 0.0);
        logmsg("ASYNC vs serial = %+.1f%%   ASYNC vs model max() = %+.1f%%\n",
               ser>0?100.0*(asy-ser)/ser:0.0, mmax>0?100.0*(asy-mmax)/mmax:0.0);
        logmsg("regime: %s-bound (compute %s comm)\n", cmp>com?"compute":"comm", cmp>com?">":"<");
        logmsg("whole-model (x%ld MoE layers): SERIAL=%.1f us  ASYNC=%.1f us  saved=%.1f us\n",
               NMOE, ser*NMOE, asy*NMOE, (ser-asy)*NMOE);
        logmsg("DIAG lane0 expert_compute: phaseA=%.1f us  phaseC=%.1f us  (ratio %.2fx)\n",
               g_ct_A/ITERS*1e6, g_ct_C/ITERS*1e6, g_ct_A>0 ? g_ct_C/g_ct_A : 0.0);
        logmsg("DIAG phaseC tid0: dispatch=%.1f us  combine=%.1f us  (barrier slack=%.1f us)\n",
               g_disp_C/ITERS*1e6, g_comb_C/ITERS*1e6,
               ser - g_ct_C/ITERS*1e6 - g_disp_C/ITERS*1e6 - g_comb_C/ITERS*1e6);
        logmsg("chk=%.3e\n", (double)chk);
    }

    for (int k = 0; k < NTNI; k++) { utofu_dereg_mem(Vcq[k], Base[k], 0); utofu_free_vcq(Vcq[k]); }
    for (int e = 0; e < NLocal; e++) { free(Wgu[e]); free(Wdn[e]); }
    free(Wgu); free(Wdn); free(Gu); free(Hh); free(Xin); free(Csink); free(Region);
    if (g_log) fclose(g_log);
    return 0;
}
