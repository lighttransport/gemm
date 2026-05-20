/*
 * Ring-attention DECODE: BARRIER-FREE async comm driver + HW barrier (A64FX).
 *
 * ring_attn_overlap showed the head-group (within-step) pipeline is
 * counterproductive: its per-stage OpenMP barriers (2*(G+1) per step) cost more
 * than the comm they hide, and splitting the reduce into G groups pays the fixed
 * per-hop overhead G times. This benchmark tests the two fixes proposed there:
 *
 *   1. ASYNC across-step driver. Software-pipeline the decode loop: thread 0
 *      drives the FULL ring all-reduce of step k-1's partials while threads
 *      1..47 compute step k. Double-buffered by iteration parity (no swap), so
 *      only 2 barriers/step (compute|comm join, then merge publish). The reduce
 *      is one full-payload collective (no G-way fixed-overhead tax). Step cost
 *      should approach max(compute, comm) instead of the serial sum.
 *
 *   2. A64FX HW barrier (libhwb, per-CMG EL0 BST + 4-way SW leader combine).
 *      ~1.3 us vs the flat OpenMP barrier's ~5-8 us at 48T. Selected by
 *      TF_HW_BARRIER (default on if cores split evenly across CMGs); applied to
 *      EVERY mode, including a re-test of the head-group pipeline -- does the
 *      cheaper barrier rescue it?
 *
 * The comm here is an honest FULL ring all-reduce (reduce 0->..->N-1 then
 * broadcast N-1->0->..->N-2 = 2(N-1) hops, every rank ends with the global),
 * unlike ring_attn_overlap's rank-0 one-hop view.
 *
 * COMPUTE KERNEL: the production qpacked+ktbl QK kernel (transformer.h
 * tf_qpkd_qk_chunk_*). Query heads ride the 16 SVE lanes (q packed [d][h]); per
 * (position,dim) the few GQA kv values are loaded once and broadcast to the
 * head-lanes via svtbl -- NO per-dot svaddv. K is stored K_DP [pos][dim][kv]
 * (bf16, expanded with <<16). QH=32 is processed as 2 lane-groups of 16 (gqa=4,
 * KVH=8). This is ~5-7x faster than ring_attn_overlap's svaddv-per-dot kernel,
 * so 16K compute drops from a pessimistic upper bound (~190us) toward the real
 * model's ~30-40us -- which makes compute COMPARABLE TO the full-ring comm, the
 * regime where across-step overlap matters most. The intra-node parallelisation
 * stays per-thread-partial + online-softmax merge (NOT the production
 * position-parallel softmax) so the comm-driver thread can run the reduce
 * without joining a mid-compute barrier; the qpacked+ktbl QK trick is orthogonal
 * to that choice. (The earlier svaddv-per-dot kernel is in git commit 07c0c67
 * for A/B.) MEASURED: this QK trick is only ~20-30% faster end-to-end here, not
 * production attention's 5-7x -- at 16K per-thread fixed costs (online-softmax
 * merge, 3 barriers, ~29 pos/thread) dominate; at 1M the kernel is DRAM-bound
 * streaming the KV shard; and the bulk of production's win is its position-
 * parallel softmax restructure, which the async overlap precludes. The faster
 * kernel does make 16K compute (~124us @HW) ~= full-ring comm (~150-170us): the
 * balanced regime where across-step overlap is exactly the right tool.
 *
 * Build:  fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *              -o ring_attn_async ring_attn_async.c -ltofucom -lhwb
 * Run:    RA_SEQ=16384 RA_GROUPS=4 TF_HW_BARRIER=1 mpiexec -np 12 ./ring_attn_async
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

#define MAX_NODES 256
#define BENCH_STAG DEMO_STAG
#define WAIT_TIMEOUT_SEC 20.0

/* ---- libhwb (Fujitsu /lib64/libhwb.so): per-CMG EL0 BST hardware barrier ---- */
extern int  vhbm_bar_init(uint64_t core_bitmask);
extern int  vhbm_bar_assign(int bd_mask, void *bb_hint);
extern void vhbm_bar(long bb);
extern int  vhbm_bar_unassign(int bd_mask);

static FILE *g_log = NULL;
static void logmsg(const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    if (g_log) { va_list a2; va_copy(a2, ap); vfprintf(g_log, fmt, a2); fflush(g_log); va_end(a2); }
    vfprintf(stdout, fmt, ap); fflush(stdout);
    va_end(ap);
}
static void die(const char *what, int rc) { logmsg("FATAL: %s (rc=%d)\n", what, rc); exit(1); }
static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec*1e-9; }
static long envl(const char *n, long d) { const char *v = getenv(n); return (v && *v) ? strtol(v, NULL, 0) : d; }

static int read_topo(uint8_t coords[][TOFU_NCOORDS])
{
    FILE *f = fopen(TOPO_PATH, "r");
    if (!f) { perror("open " TOPO_PATH); exit(1); }
    int n = 0; char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        unsigned r, c[TOFU_NCOORDS];
        if (sscanf(line, "%u %u %u %u %u %u %u", &r, &c[0],&c[1],&c[2],&c[3],&c[4],&c[5]) != 7) continue;
        for (int k = 0; k < TOFU_NCOORDS; k++) coords[n][k] = (uint8_t)c[k];
        n++;
    }
    fclose(f);
    if (n < 2) { fprintf(stderr, "%s: need >=2 nodes\n", TOPO_PATH); exit(1); }
    return n;
}

static int node_of_cpu(int cpu)
{
    for (int nd = 0; nd < 16; nd++) {
        char path[64], buf[256];
        snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", nd);
        FILE *f = fopen(path, "r"); if (!f) continue;
        char *got = fgets(buf, sizeof buf, f); fclose(f); if (!got) continue;
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

/* Vectorised expf (Cephes-style, fp32). Inputs here are <= 0 (score - max). */
static inline svfloat32_t expf_sve(svbool_t pg, svfloat32_t x)
{
    const float LOG2EF = 1.44269504088896341f;
    const float C1 = 0.693359375f, C2 = -2.12194440e-4f;
    x = svmax_n_f32_x(pg, x, -87.0f);
    svfloat32_t fx = svmla_n_f32_x(pg, svdup_f32(0.5f), x, LOG2EF);
    svint32_t   e  = svcvt_s32_f32_x(pg, fx);
    svfloat32_t flr = svcvt_f32_s32_x(pg, e);
    svbool_t gt = svcmpgt_f32(pg, flr, fx);
    flr = svsub_n_f32_m(gt, flr, 1.0f);
    x = svmls_n_f32_x(pg, x, flr, C1);
    x = svmls_n_f32_x(pg, x, flr, C2);
    svfloat32_t y = svdup_f32(1.9875691500e-4f);
    y = svmla_f32_x(pg, svdup_f32(1.3981999507e-3f), y, x);
    y = svmla_f32_x(pg, svdup_f32(8.3334519073e-3f), y, x);
    y = svmla_f32_x(pg, svdup_f32(4.1665795894e-2f), y, x);
    y = svmla_f32_x(pg, svdup_f32(1.6666665459e-1f), y, x);
    y = svmla_f32_x(pg, svdup_f32(5.0000001201e-1f), y, x);
    svfloat32_t x2 = svmul_f32_x(pg, x, x);
    y = svmla_f32_x(pg, x, y, x2);
    y = svadd_n_f32_x(pg, y, 1.0f);
    svint32_t n = svadd_n_s32_x(pg, svcvt_s32_f32_x(pg, flr), 127);
    n = svlsl_n_s32_x(pg, n, 23);
    return svmul_f32_x(pg, y, svreinterpret_f32_s32(n));
}

static inline void merge_partial(float *A, const float *B, int HD)
{
    float mA = A[0], lA = A[1], mB = B[0], lB = B[1];
    float m = mA > mB ? mA : mB;
    float sA = expf(mA - m), sB = expf(mB - m);
    A[0] = m;
    A[1] = lA * sA + lB * sB;
    for (int d = 0; d < HD; d++) A[2 + d] = A[2 + d] * sA + B[2 + d] * sB;
}

/* ===================== uTofu transport state (file-global) ===================== */
static utofu_vcq_hdl_t g_vcq;
static utofu_vcq_id_t  g_peer_vcq[MAX_NODES];
static utofu_stadd_t   g_peer_base[MAX_NODES];
static utofu_stadd_t   g_base_stadd;
static size_t g_slot;
static char  *g_recv_buf, *g_send_buf;
static int    g_my_rank, g_N, g_HD;
static const unsigned long g_flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;

/* send nqh heads of `buf` (from head qh0) to peer dst_rank, tagging seq s. */
static void ring_send(const float *buf, int qh0, int nqh, int dst_rank, uint64_t s)
{
    int rc; void *cb;
    size_t fb = (size_t)nqh * (g_HD + 2) * sizeof(float);
    memcpy(g_send_buf, buf + (size_t)qh0 * (g_HD + 2), fb);
    *(volatile uint64_t *)(g_send_buf + fb) = s;
    utofu_stadd_t src = g_base_stadd + g_slot;
    utofu_stadd_t dst = g_peer_base[dst_rank] + 0;
    for (;;) { rc = utofu_put(g_vcq, g_peer_vcq[dst_rank], src, dst, fb + 8, 0, g_flags, NULL);
               if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(g_vcq, 0, &cb); }
    if (rc != UTOFU_SUCCESS) die("put", rc);
    do { rc = utofu_poll_tcq(g_vcq, 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND);
}

/* spin until the recv slot's seq (at offset fb) reaches s; return recv floats. */
static float *ring_wait(uint64_t s, int nqh)
{
    size_t fb = (size_t)nqh * (g_HD + 2) * sizeof(float);
    volatile uint64_t *rseq = (volatile uint64_t *)(g_recv_buf + fb);
    double t0 = now_sec();
    while (*rseq < s) { if (now_sec() - t0 > WAIT_TIMEOUT_SEC) die("allreduce wait timeout", -1); }
    return (float *)g_recv_buf;
}

/* full ring all-reduce of nqh heads (from qh0) in `buf`: reduce 0->..->N-1 then
 * broadcast N-1->0->..->N-2. Every rank ends holding the global online-softmax
 * partial. rs/bs are the reduce/broadcast seq tags (monotonic across calls).
 * Driven entirely by the calling (comm) thread. */
static void full_allreduce(float *buf, int qh0, int nqh, uint64_t rs, uint64_t bs)
{
    int R = g_my_rank, N = g_N, HD = g_HD;
    /* reduce */
    if (R == 0) {
        ring_send(buf, qh0, nqh, 1, rs);
    } else {
        float *rp = ring_wait(rs, nqh);
        for (int h = 0; h < nqh; h++)
            merge_partial(buf + (size_t)(qh0 + h) * (HD + 2), rp + (size_t)h * (HD + 2), HD);
        if (R < N - 1) ring_send(buf, qh0, nqh, R + 1, rs);
    }
    /* broadcast */
    if (R == N - 1) {
        ring_send(buf, qh0, nqh, 0, bs);
    } else {
        float *rp = ring_wait(bs, nqh);
        memcpy(buf + (size_t)qh0 * (HD + 2), rp, (size_t)nqh * (HD + 2) * sizeof(float));
        if (R < N - 2) ring_send(buf, qh0, nqh, R + 1, bs);
    }
}

/* ===================== HW barrier (libhwb) state ===================== */
static int g_hw_enabled = 0;
static int g_hwbar_bd = 0, g_hwbar_tpc = 12, g_hwbar_ncmg = 4;
static volatile int g_lcount = 0; static char _pad1[60] __attribute__((unused));
static volatile int g_lsense = 0; static char _pad2[60] __attribute__((unused));
static volatile int g_join_count = 0, g_assign_failed = 0;

/* Hierarchical 48T barrier (HW intra-CMG arrival + 4-way SW leader combine +
 * HW intra-CMG release) when g_hw_enabled, else a flat OpenMP barrier. The
 * orphaned omp barrier is lexically inside the parallel region at each use. */
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
    long S      = envl("RA_SEQ", 16384);
    long QH     = envl("RA_QH", 32);
    long KVH    = envl("RA_KVH", 8);
    long HD     = envl("RA_HD", 128);
    long G      = envl("RA_GROUPS", 4);
    long ITERS  = envl("RA_ITERS", 300);
    long WARMUP = envl("RA_WARMUP", 30);
    if (KVH % G != 0) die("RA_GROUPS must divide RA_KVH", -1);
    if (HD % (long)svcntw() != 0) die("RA_HD must be a multiple of svcntw()", -1);
    long QPKV = QH / KVH;            /* GQA group size (gqa) */
    long gqa  = QPKV;
    long kvpg = KVH / G;
    long qhpg = QH / G;
    /* qpacked+ktbl: query heads ride 16 SVE lanes (svcntw()==16 for f32); a
     * 16-head lane-group spans <=4 kv-heads so a single quadword broadcast
     * covers it. Requires QH a multiple of svcntw() and gqa>=4 (16/gqa<=4). */
    if ((long)svcntw() != 16) die("qpacked kernel assumes svcntw()==16 (f32)", -1);
    if (QH % 16 != 0)  die("RA_QH must be a multiple of 16 for qpacked+ktbl", -1);
    if (QH % KVH != 0) die("RA_QH must be a multiple of RA_KVH", -1);
    if (gqa < 4 || (16 % gqa) != 0) die("qpacked+ktbl needs gqa in {4,8,16}", -1);
    (void)kvpg;
    float scale = 1.0f / sqrtf((float)HD);

    /* ---- uTofu setup ---- */
    utofu_tni_id_t *tnis = NULL; size_t ntni = 0;
    rc = utofu_get_onesided_tnis(&tnis, &ntni);
    if (rc != UTOFU_SUCCESS || ntni < 1) die("get_onesided_tnis", rc);
    utofu_tni_id_t tni = tnis[0]; free(tnis);

    uint8_t my_coords[TOFU_NCOORDS] = {0};
    if (utofu_query_my_coords(my_coords) != UTOFU_SUCCESS) die("query_my_coords", -1);

    {
        char nm[64];
        snprintf(nm, sizeof nm, "raa_log_%u_%u_%u_%u_%u_%u.txt",
                 my_coords[0],my_coords[1],my_coords[2],my_coords[3],my_coords[4],my_coords[5]);
        g_log = fopen(nm, "w");
    }

    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    int N = read_topo(topo);
    int my_rank = -1;
    for (int r = 0; r < N; r++) if (memcmp(topo[r], my_coords, TOFU_NCOORDS) == 0) my_rank = r;
    if (my_rank < 0) die("my coords not in topo", -1);

    if (utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &g_vcq) != UTOFU_SUCCESS) die("create_vcq", -1);
    utofu_vcq_id_t my_vcq;
    if (utofu_query_vcq_id(g_vcq, &my_vcq) != UTOFU_SUCCESS) die("query_vcq_id", -1);

    size_t max_fbytes = (size_t)QH * (HD + 2) * sizeof(float);
    g_slot = (max_fbytes + 8 + (DEMO_CACHE_LINE - 1)) & ~(size_t)(DEMO_CACHE_LINE - 1);
    size_t region_sz = 2 * g_slot;
    void *region = NULL;
    if (posix_memalign(&region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign(region)", -1);
    memset(region, 0, region_sz);
    g_recv_buf = (char *)region;
    g_send_buf = (char *)region + g_slot;

    if (utofu_reg_mem_with_stag(g_vcq, region, region_sz, BENCH_STAG, 0, &g_base_stadd) != UTOFU_SUCCESS)
        die("reg_mem", -1);

    for (int r = 0; r < N; r++) {
        if (r == my_rank) { g_peer_vcq[r] = my_vcq; g_peer_base[r] = g_base_stadd; continue; }
        if (utofu_construct_vcq_id(topo[r], tni, DEMO_CQ_ID, DEMO_CMP_ID, &g_peer_vcq[r]) != UTOFU_SUCCESS)
            die("construct_vcq_id", -1);
        utofu_set_vcq_id_path(&g_peer_vcq[r], NULL);
        if (utofu_query_stadd(g_peer_vcq[r], BENCH_STAG, &g_peer_base[r]) != UTOFU_SUCCESS)
            die("query_stadd", -1);
    }
    g_my_rank = my_rank; g_N = N; g_HD = (int)HD;

    /* ---- thread / CMG layout: thread 0 = comm driver, 1..nthr-1 compute ---- */
    enum { CMG_CORES = 12 };
    cpu_set_t allowed; sched_getaffinity(0, sizeof allowed, &allowed);
    int cores[256], ncore = 0;
    for (int c = 0; c < CPU_SETSIZE && ncore < 256; c++) if (CPU_ISSET(c, &allowed)) cores[ncore++] = c;
    int ncmg = ncore / CMG_CORES, cores_per = CMG_CORES;
    if (ncmg < 1) { ncmg = 1; cores_per = ncore < 1 ? 1 : ncore; }
    int nthr = ncmg * cores_per;
    int nct = nthr - 1;
    if (nct < 1) die("need >=2 cores", -1);

    long P = (S + N - 1) / N;
    P = (P / nct) * nct; if (P < nct) P = nct;
    long Pt = P / nct;
    long pos_cmg[64];
    for (int c = 0; c < ncmg; c++) pos_cmg[c] = (long)((c == 0) ? cores_per - 1 : cores_per) * Pt;

    uint16_t *Kbuf[64] = {0}, *Vbuf[64] = {0};
    for (int c = 0; c < ncmg; c++) {
        size_t elems = (size_t)KVH * pos_cmg[c] * HD;
        size_t bytes = elems * sizeof(uint16_t);
        if (posix_memalign((void **)&Kbuf[c], 2*1024*1024, bytes) != 0) die("posix_memalign(K)", -1);
        if (posix_memalign((void **)&Vbuf[c], 2*1024*1024, bytes) != 0) die("posix_memalign(V)", -1);
        int node = node_of_cpu(cores[c * cores_per]);
        if (node >= 0) {
            unsigned long mask = 1UL << node;
            syscall(SYS_mbind, Kbuf[c], bytes, 2L, &mask, 8UL*sizeof(mask), 2U);
            syscall(SYS_mbind, Vbuf[c], bytes, 2L, &mask, 8UL*sizeof(mask), 2U);
        }
        for (size_t i = 0; i < elems; i++) {
            float kv = 0.01f * (float)((long)(i % 13) - 6);
            uint32_t b; memcpy(&b, &kv, 4); Kbuf[c][i] = (uint16_t)(b >> 16);
            kv = 0.01f * (float)((long)(i % 11) - 5);
            memcpy(&b, &kv, 4); Vbuf[c][i] = (uint16_t)(b >> 16);
        }
    }

    float *q = malloc((size_t)QH * HD * sizeof(float));
    for (long i = 0; i < QH * HD; i++) q[i] = 0.01f * (float)((i % 17) - 8);
    /* qpacked Q: transpose [h][d] -> [d][h] once (q is constant in this bench). */
    float *q_packed = malloc((size_t)HD * QH * sizeof(float));
    for (long h = 0; h < QH; h++)
        for (long d = 0; d < HD; d++) q_packed[d*QH + h] = q[h*HD + d];
    float **part = calloc(nthr, sizeof(float *));
    float **scr  = calloc(nthr, sizeof(float *));
    for (int t = 0; t < nthr; t++) {
        part[t] = malloc((size_t)QH * (HD + 2) * sizeof(float));
        scr[t]  = malloc((size_t)QH * Pt * sizeof(float));   /* head-major scores */
    }
    /* two node-partial buffers for the across-step pipeline (parity-indexed) */
    float *buf0 = malloc((size_t)QH * (HD + 2) * sizeof(float));
    float *buf1 = malloc((size_t)QH * (HD + 2) * sizeof(float));
    float *bufs[2] = { buf0, buf1 };
    for (long h = 0; h < QH; h++) {           /* prime both buffers (valid partials) */
        buf0[h*(HD+2)] = 0; buf0[h*(HD+2)+1] = 1;
        buf1[h*(HD+2)] = 0; buf1[h*(HD+2)+1] = 1;
        for (long d = 0; d < HD; d++) { buf0[h*(HD+2)+2+d] = 0; buf1[h*(HD+2)+2+d] = 0; }
    }

    /* ---- HW-barrier master init (decide enablement, build core mask) ---- */
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

    if (my_rank == 0) {
        double tot = 0; for (int c = 0; c < ncmg; c++) tot += 2.0 * KVH * pos_cmg[c] * HD * sizeof(uint16_t);
        logmsg("=== ring-attention DECODE: async across-step driver + HW barrier ===\n");
        logmsg("nodes=%d seq S=%ld pos/node=%ld q_heads=%ld kv_heads=%ld head_dim=%ld\n",
               N, S, P, QH, KVH, HD);
        logmsg("groups G=%ld threads=%d (1 comm + %d compute, %d CMG x %d)\n",
               G, nthr, nct, ncmg, cores_per);
        logmsg("barrier = %s   KV shard = %.2f MiB bf16   full-reduce payload = %zu B\n",
               g_hw_enabled ? "HW (libhwb hierarchical)" : "flat OpenMP",
               tot/1048576.0, max_fbytes);
        logmsg("comm = FULL ring all-reduce (reduce + broadcast = %d hops)\n", 2*(N-1));
        logmsg("kernel = qpacked+ktbl (svtbl QK, K_DP [p][d][kv], gqa=%ld, %ld lane-grp/pos)\n",
               gqa, (QH + 15) / 16);
    }

    /* compute compute-thread `tid`'s q-heads [H0, H0+HCNT) into part[tid] using
     * the PRODUCTION qpacked+ktbl QK kernel: query heads ride the 16 SVE lanes,
     * K is K_DP [pos][dim][kv] and the few GQA kv values are broadcast to the
     * head-lanes via svtbl -- no per-dot svaddv. Then the usual per-thread
     * online-softmax + bf16 A.V over this thread's position range. `idx_v`
     * (idx[i]=i/gqa) is the svtbl lane->kv map, built per thread below.
     * QK is done in full 16-lane groups; reading the whole [H0,H0+HCNT) range in
     * one COMPUTE_ALL pass touches each K_DP cache line once at full width. */
#define COMPUTE_RANGE(H0, HCNT, tid)                                              \
    do {                                                                          \
        int _cmg = (tid) / cores_per;                                             \
        int _idx = (_cmg == 0) ? (tid) - 1 : (tid) - _cmg * cores_per;            \
        long _p0 = (long)_idx * Pt, _p1 = _p0 + Pt;                               \
        long _Pc = pos_cmg[_cmg];                                                 \
        const uint16_t *_Kc = Kbuf[_cmg];                                         \
        long _h0 = (H0), _hn = (HCNT);                                            \
        svbool_t _pgd = svptrue_b32();                                            \
        /* -- QK: per position, all _hn heads via qpacked+ktbl (lane-grps) -- */  \
        for (long _p = _p0; _p < _p1; _p++) {                                     \
            const uint16_t *_kp = _Kc + (size_t)_p*HD*KVH;                        \
            long _pl = _p - _p0;                                                  \
            for (long _hc = 0; _hc < _hn; _hc += 16) {                            \
                long _lg = _hn - _hc; if (_lg > 16) _lg = 16;                     \
                long _hbase = _h0 + _hc;                                          \
                long _kvbase = _hbase / gqa;                                      \
                long _nkv = (_lg - 1) / gqa + 1;                                  \
                svbool_t _pgh = svwhilelt_b32((uint64_t)0, (uint64_t)_lg);        \
                svbool_t _pgk = svwhilelt_b32((uint64_t)0, (uint64_t)_nkv);       \
                svfloat32_t _acc = svdup_f32(0.0f);                              \
                for (long _d = 0; _d < HD; _d++) {                               \
                    svfloat32_t _qv = svld1_f32(_pgh, q_packed + _d*QH + _hbase); \
                    svuint32_t _kb = svld1uh_u32(_pgk, _kp + (size_t)_d*KVH + _kvbase); \
                    svfloat32_t _kf = svreinterpret_f32_u32(svlsl_n_u32_x(_pgk,_kb,16)); \
                    _acc = svmla_f32_x(_pgh, _acc, _qv, svtbl_f32(_kf, idx_v));   \
                }                                                                 \
                float _tmp[16];                                                  \
                svst1_f32(_pgh, _tmp, svmul_n_f32_x(_pgh, _acc, scale));          \
                for (long _i = 0; _i < _lg; _i++)                                \
                    scr[tid][(_hc + _i)*Pt + _pl] = _tmp[_i];                    \
            }                                                                     \
        }                                                                         \
        /* -- softmax + A.V per head over this thread's position sub-range -- */  \
        for (long _lh = 0; _lh < _hn; _lh++) {                                    \
            long _hg = _h0 + _lh, _kvh = _hg / gqa;                               \
            const uint16_t *_V = Vbuf[_cmg] + (size_t)_kvh*_Pc*HD;               \
            float *_sc = scr[tid] + _lh*Pt;                                       \
            float _m = -INFINITY;                                                 \
            for (long _i = 0; _i < Pt; _i++) if (_sc[_i] > _m) _m = _sc[_i];      \
            for (long _i = 0; _i < Pt; _i += (long)svcntw()) {                    \
                svbool_t _pg = svwhilelt_b32((uint64_t)_i, (uint64_t)Pt);         \
                svfloat32_t _v = svsub_n_f32_x(_pg, svld1_f32(_pg, _sc+_i), _m);  \
                svst1_f32(_pg, _sc+_i, expf_sve(_pg, _v));                        \
            }                                                                     \
            float _l = 0; for (long _i=0;_i<Pt;_i++) _l += _sc[_i];               \
            float *_o = part[tid] + _hg*(HD+2);                                   \
            _o[0] = _m; _o[1] = _l;                                               \
            for (long _d = 0; _d < HD; _d += (long)svcntw()) svst1_f32(_pgd, _o+2+_d, svdup_f32(0.0f)); \
            for (long _p = _p0; _p < _p1; _p++) {                                 \
                float _w = _sc[_p-_p0];                                           \
                const uint16_t *_vrow = _V + (size_t)_p*HD;                       \
                for (long _d = 0; _d < HD; _d += (long)svcntw()) {               \
                    svuint32_t _vb = svld1uh_u32(_pgd, _vrow+_d);                \
                    svfloat32_t _vf = svreinterpret_f32_u32(svlsl_n_u32_x(_pgd,_vb,16)); \
                    svfloat32_t _of = svld1_f32(_pgd, _o+2+_d);                  \
                    svst1_f32(_pgd, _o+2+_d, svmla_n_f32_x(_pgd,_of,_vf,_w));    \
                }                                                                 \
            }                                                                     \
        }                                                                         \
    } while (0)
/* all heads, single full-width pass (compute-only/serial/async) */
#define COMPUTE_ALL(tid)        COMPUTE_RANGE(0, QH, tid)
/* one head-group (PIPE pipelining only; may run at <16 lanes if qhpg<16) */
#define COMPUTE_GROUP(g, tid)   COMPUTE_RANGE((g)*qhpg, qhpg, tid)

    /* merge group g's per-thread partials into DST (parallel over the group's
     * q-heads; thread `tid` owns heads where (h-qh0) % nthr == tid). */
#define MERGE_GROUP(DST, g, tid)                                                  \
    do {                                                                          \
        long _qh0 = (g)*qhpg;                                                     \
        for (long _h = _qh0 + (tid); _h < _qh0 + qhpg; _h += nthr) {              \
            float *_acc = (DST) + (size_t)_h*(HD+2);                              \
            int _first = 1;                                                       \
            for (int _t = 1; _t < nthr; _t++) {                                   \
                float *_pp = part[_t] + (size_t)_h*(HD+2);                         \
                if (_first) { memcpy(_acc, _pp, (HD+2)*sizeof(float)); _first = 0;}\
                else merge_partial(_acc, _pp, HD);                                \
            }                                                                     \
        }                                                                         \
    } while (0)

    /* prime recv-slot seqs (full-payload offset) so the first timed waits work. */
    {
        uint64_t w = 1;
        for (int a = 0; a < 400; a++) {
            if (my_rank == 0) ring_send(bufs[0], 0, QH, 1, w);
            else { float *rp = NULL; double t0 = now_sec();
                   volatile uint64_t *rs = (volatile uint64_t*)(g_recv_buf + (size_t)QH*(HD+2)*4);
                   while (*rs < w) { if (now_sec()-t0 > 2.0) break; }
                   (void)rp;
                   if (my_rank < N-1) ring_send(bufs[0], 0, QH, my_rank+1, w); }
            break;
        }
    }

    double t_compute=0, t_comm=0, t_serial=0, t_async=0, t_pipe=0;
    volatile double chk = 0;
    uint64_t seq0 = 16;     /* monotone seq base, advanced by tid0 only */

#pragma omp parallel num_threads(nthr)
    {
        int tid = omp_get_thread_num();
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(cores[tid], &s); sched_setaffinity(0, sizeof s, &s);
        long bb = 0; int ls4 = 0;
        if (g_hw_enabled) {
            bb = vhbm_bar_assign(g_hwbar_bd, NULL);
            if (bb < 0) { g_assign_failed = 1; bb = 0; }
        }
        __sync_synchronize();
        __sync_add_and_fetch((int *)&g_join_count, 1);
#pragma omp barrier
#pragma omp master
        if (g_hw_enabled && g_assign_failed) { g_hw_enabled = 0; fprintf(stderr, "HW barrier join failed; flat\n"); }
#pragma omp barrier
        uint64_t seq = seq0;        /* private; only tid0's advances matter */
        /* svtbl lane->kv map for qpacked+ktbl: head-lane i -> kv (i/gqa) */
        uint32_t _idxa[16];
        for (int i = 0; i < 16; i++) _idxa[i] = (uint32_t)(i / gqa);
        svuint32_t idx_v = svld1_u32(svptrue_b32(), _idxa);

        /* ===== A) compute-only (nct threads) ===== */
        double tA = 0;
        for (long it = 0; it < WARMUP + ITERS; it++) {
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            if (it == WARMUP) tA = now_sec();
            if (tid != 0) COMPUTE_ALL(tid);
            STEPBAR(tid, bb, &ls4);
            for (long g = 0; g < G; g++) MERGE_GROUP(bufs[0], g, tid);
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            if (it == WARMUP + ITERS - 1) { t_compute = (now_sec() - tA) / ITERS; chk += bufs[0][1]; }
        }

        /* ===== B) comm-only: one ISOLATED full all-reduce per step (tid0),
         * the other threads idle at the barrier. Per-iteration STEPBARs keep
         * all threads cycling (no marathon block) and measure the all-reduce
         * the way SERIAL/ASYNC see it — isolated, including cross-rank skew. ===== */
        double tB = 0;
        for (long it = 0; it < WARMUP + ITERS; it++) {
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            {
                if (it == WARMUP) tB = now_sec();
                full_allreduce(bufs[0], 0, QH, seq + 1, seq + 2); seq += 2;
            }
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            if (it == WARMUP + ITERS - 1) t_comm = (now_sec() - tB) / ITERS;
        }

        /* ===== C) SERIAL: compute + merge, then ONE full all-reduce ===== */
        double tC = 0;
        for (long it = 0; it < WARMUP + ITERS; it++) {
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            if (it == WARMUP) tC = now_sec();
            if (tid != 0) COMPUTE_ALL(tid);
            STEPBAR(tid, bb, &ls4);
            for (long g = 0; g < G; g++) MERGE_GROUP(bufs[0], g, tid);
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            { full_allreduce(bufs[0], 0, QH, seq + 1, seq + 2); seq += 2; }
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            if (it == WARMUP + ITERS - 1) { t_serial = (now_sec() - tC) / ITERS; chk += bufs[0][1]; }
        }

        /* ===== D) ASYNC across-step: compute step k (1..nct) || all-reduce
         * step k-1 (tid0). Double-buffered by parity; 2 barriers/step. ===== */
        double tD = 0;
        for (long it = 0; it < WARMUP + ITERS; it++) {
            float *cur = bufs[it & 1], *prev = bufs[(it + 1) & 1];
#pragma omp master
            if (it == WARMUP) tD = now_sec();
            /* phase A: compute (compute threads) || reduce prev (comm thread) */
            if (tid != 0) COMPUTE_ALL(tid);
            else          { full_allreduce(prev, 0, QH, seq + 1, seq + 2); seq += 2; }
            STEPBAR(tid, bb, &ls4);                 /* both done */
            /* phase B: publish this step's merged partial into cur */
            for (long g = 0; g < G; g++) MERGE_GROUP(cur, g, tid);
            STEPBAR(tid, bb, &ls4);                 /* cur visible before it becomes prev */
#pragma omp master
            if (it == WARMUP + ITERS - 1) { t_async = (now_sec() - tD) / ITERS; chk += cur[1]; }
        }

        /* ===== E) HEAD-GROUP pipeline (re-test under the selected barrier) =====
         * stage st: compute group st (1..nct) || reduce group st-1 (tid0). */
        double tE = 0;
        for (long it = 0; it < WARMUP + ITERS; it++) {
            STEPBAR(tid, bb, &ls4);
#pragma omp master
            if (it == WARMUP) tE = now_sec();
            for (long st = 0; st <= G; st++) {
                if (st < G && tid != 0) COMPUTE_GROUP(st, tid);
                if (st >= 1 && tid == 0) {
                    long g = st - 1;
                    full_allreduce(bufs[0], g*qhpg, qhpg, seq + 1, seq + 2); seq += 2;
                }
                STEPBAR(tid, bb, &ls4);
                if (st < G) MERGE_GROUP(bufs[0], st, tid);
                STEPBAR(tid, bb, &ls4);
            }
#pragma omp master
            if (it == WARMUP + ITERS - 1) { t_pipe = (now_sec() - tE) / ITERS; chk += bufs[0][1]; }
        }

        if (g_hw_enabled) vhbm_bar_unassign(g_hwbar_bd);
    }

    if (my_rank == 0) {
        double cmp = t_compute*1e6, com = t_comm*1e6, ser = t_serial*1e6,
               asy = t_async*1e6, pip = t_pipe*1e6;
        double mmax = cmp > com ? cmp : com;
        logmsg("\n--- measured per decode step (us), barrier=%s ---\n",
               g_hw_enabled ? "HW" : "flat-OMP");
        logmsg("compute-only (%dT)              = %.1f\n", nct, cmp);
        logmsg("comm-only (full all-reduce)     = %.1f\n", com);
        logmsg("SERIAL  (compute + all-reduce)  = %.1f\n", ser);
        logmsg("ASYNC   (across-step, 2 bar/stp)= %.1f\n", asy);
        logmsg("PIPE    (head-group, %ld groups) = %.1f\n", G, pip);
        logmsg("\n--- vs the model ---\n");
        logmsg("model serial = compute+comm     = %.1f\n", cmp + com);
        logmsg("model max()  = max(compute,comm)= %.1f\n", mmax);
        double ideal = ser - mmax;
        logmsg("ASYNC overlap efficiency = (serial-async)/(serial-max) = %.0f%%\n",
               ideal > 0 ? 100.0*(ser - asy)/ideal : 0.0);
        logmsg("ASYNC vs serial = %+.1f%%   ASYNC vs model max() = %+.1f%%\n",
               ser>0?100.0*(asy-ser)/ser:0.0, mmax>0?100.0*(asy-mmax)/mmax:0.0);
        logmsg("PIPE  vs serial = %+.1f%%\n", ser>0?100.0*(pip-ser)/ser:0.0);
        logmsg("chk=%.3e\n", (double)chk);
    }

    utofu_dereg_mem(g_vcq, g_base_stadd, 0);
    utofu_free_vcq(g_vcq);
    free(region); free(q); free(q_packed); free(buf0); free(buf1);
    for (int t = 0; t < nthr; t++) { free(part[t]); free(scr[t]); }
    free(part); free(scr);
    for (int c = 0; c < ncmg; c++) { free(Kbuf[c]); free(Vbuf[c]); }
    return 0;
}
