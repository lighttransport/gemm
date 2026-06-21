/*
 * Ring-attention DECODE: comm/compute OVERLAP validation (A64FX / Fugaku).
 *
 * ring_attn_bench models the two costs (KV-shard read + uTofu reduce) and prints
 * an "overlap (max)" line = max(compute, comm), ASSUMING comm hides perfectly
 * under compute. This benchmark INTEGRATES the real attention math and measures
 * whether that assumption holds.
 *
 *   - Real flash-decode attention compute over a per-CMG, NUMA-local, bf16 KV
 *     shard (q.K dot, vectorised SVE softmax, A.V), producing real (m,l,o)
 *     online-softmax partials per query head. 47 compute threads, 1 comm driver.
 *   - Real online-softmax merge in a linear ring-reduce over uTofu (N-1 hops,
 *     same payload model as ring_attn_bench: QH*(HD+2)*RBYTES, here fp32 floats).
 *   - HEAD-GROUP PIPELINE overlap: thread 0 drives the ring-reduce of group g-1
 *     while threads 1..47 compute group g. A per-stage OpenMP barrier makes each
 *     stage cost max(compute_g, reduce_{g-1}); G groups => G+1 stages.
 *
 * Measures, per decode step: compute-only, comm-only, SERIAL (compute then one
 * full reduce), and OVERLAP(G). Compares to the model's serial (sum) and max().
 * The catch the model misses: splitting into G groups makes the reduce pay the
 * ~1.23us fixed per-hop overhead G times, so overlap rarely reaches max().
 *
 * Build:  fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *              -o ring_attn_overlap ring_attn_overlap.c -ltofucom
 * Run:    RA_SEQ=16384 RA_GROUPS=4 mpiexec -np 12 ./ring_attn_overlap
 *         (needs tofu_topo.txt; sweep RA_SEQ for 16K/1M, RA_GROUPS over divisors of KVH)
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
    svint32_t   e  = svcvt_s32_f32_x(pg, fx);                 /* trunc */
    svfloat32_t flr = svcvt_f32_s32_x(pg, e);
    svbool_t gt = svcmpgt_f32(pg, flr, fx);
    flr = svsub_n_f32_m(gt, flr, 1.0f);                       /* -> floor */
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

/* q (fp32, HD) . k (bf16, HD); HD a multiple of svcntw(). */
static inline float dot_qk_bf16(const float *q, const uint16_t *k, int HD)
{
    svbool_t pg = svptrue_b32();
    svfloat32_t acc = svdup_f32(0.0f);
    for (int d = 0; d < HD; d += (int)svcntw()) {
        svuint32_t kb = svld1uh_u32(pg, k + d);                       /* bf16 bits */
        svfloat32_t kf = svreinterpret_f32_u32(svlsl_n_u32_x(pg, kb, 16));
        acc = svmla_f32_x(pg, acc, kf, svld1_f32(pg, q + d));
    }
    return svaddv_f32(pg, acc);
}

/* online-softmax merge of partial B (m,l,o[HD]) into A in place. */
static inline void merge_partial(float *A, const float *B, int HD)
{
    float mA = A[0], lA = A[1], mB = B[0], lB = B[1];
    float m = mA > mB ? mA : mB;
    float sA = expf(mA - m), sB = expf(mB - m);
    A[0] = m;
    A[1] = lA * sA + lB * sB;
    for (int d = 0; d < HD; d++) A[2 + d] = A[2 + d] * sA + B[2 + d] * sB;
}

int main(void)
{
    int rc;
    long S      = envl("RA_SEQ", 16384);
    long QH     = envl("RA_QH", 32);
    long KVH    = envl("RA_KVH", 8);
    long HD     = envl("RA_HD", 128);
    long G      = envl("RA_GROUPS", 4);     /* head-groups (must divide KVH)      */
    long ITERS  = envl("RA_ITERS", 300);
    long WARMUP = envl("RA_WARMUP", 30);
    if (KVH % G != 0) die("RA_GROUPS must divide RA_KVH", -1);
    if (HD % (long)svcntw() != 0) die("RA_HD must be a multiple of svcntw()", -1);
    long QPKV = QH / KVH;                    /* GQA: q-heads per kv-head           */
    long kvpg = KVH / G;                     /* kv-heads per group                 */
    long qhpg = QH / G;                      /* q-heads per group                  */
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
        snprintf(nm, sizeof nm, "rao_log_%u_%u_%u_%u_%u_%u.txt",
                 my_coords[0],my_coords[1],my_coords[2],my_coords[3],my_coords[4],my_coords[5]);
        g_log = fopen(nm, "w");
    }

    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    int N = read_topo(topo);
    int my_rank = -1;
    for (int r = 0; r < N; r++) if (memcmp(topo[r], my_coords, TOFU_NCOORDS) == 0) my_rank = r;
    if (my_rank < 0) die("my coords not in topo", -1);

    utofu_vcq_hdl_t vcq;
    if (utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &vcq) != UTOFU_SUCCESS) die("create_vcq", -1);
    utofu_vcq_id_t my_vcq;
    if (utofu_query_vcq_id(vcq, &my_vcq) != UTOFU_SUCCESS) die("query_vcq_id", -1);

    /* reduce transport region: [recv | send-staging], each holds a full-payload
     * float array + trailing seq, on its own cache line (CPU only reads recv). */
    size_t max_fbytes = (size_t)QH * (HD + 2) * sizeof(float);     /* 16640 for 9B */
    size_t slot = (max_fbytes + 8 + (DEMO_CACHE_LINE - 1)) & ~(size_t)(DEMO_CACHE_LINE - 1);
    size_t region_sz = 2 * slot;
    void *region = NULL;
    if (posix_memalign(&region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign(region)", -1);
    memset(region, 0, region_sz);
    char *recv_buf = (char *)region;          /* peer Puts here              */
    char *send_buf_t = (char *)region + slot;  /* we stage the outgoing block */

    utofu_stadd_t base_stadd;
    if (utofu_reg_mem_with_stag(vcq, region, region_sz, BENCH_STAG, 0, &base_stadd) != UTOFU_SUCCESS)
        die("reg_mem", -1);

    static utofu_vcq_id_t peer_vcq[MAX_NODES];
    static utofu_stadd_t  peer_base[MAX_NODES];
    for (int r = 0; r < N; r++) {
        if (r == my_rank) { peer_vcq[r] = my_vcq; peer_base[r] = base_stadd; continue; }
        if (utofu_construct_vcq_id(topo[r], tni, DEMO_CQ_ID, DEMO_CMP_ID, &peer_vcq[r]) != UTOFU_SUCCESS)
            die("construct_vcq_id", -1);
        utofu_set_vcq_id_path(&peer_vcq[r], NULL);
        if (utofu_query_stadd(peer_vcq[r], BENCH_STAG, &peer_base[r]) != UTOFU_SUCCESS)
            die("query_stadd", -1);
    }

    /* ---- thread / CMG layout (mirror bench_node_bw): thread 0 = comm driver,
     * threads 1..nthr-1 compute. Positions split EQUALLY across the nthr-1
     * compute threads, each reading only its own CMG's NUMA-local KV buffer. ---- */
    enum { CMG_CORES = 12 };
    cpu_set_t allowed; sched_getaffinity(0, sizeof allowed, &allowed);
    int cores[256], ncore = 0;
    for (int c = 0; c < CPU_SETSIZE && ncore < 256; c++) if (CPU_ISSET(c, &allowed)) cores[ncore++] = c;
    int ncmg = ncore / CMG_CORES, cores_per = CMG_CORES;
    if (ncmg < 1) { ncmg = 1; cores_per = ncore < 1 ? 1 : ncore; }
    int nthr = ncmg * cores_per;
    int nct = nthr - 1;                          /* compute threads (tid 1..nthr-1) */
    if (nct < 1) die("need >=2 cores", -1);

    long P = (S + N - 1) / N;                    /* positions on this node          */
    P = (P / nct) * nct; if (P < nct) P = nct;   /* even split across compute threads */
    long Pt = P / nct;                           /* positions per compute thread     */
    /* compute threads per CMG: CMG0 has cores_per-1 (tid0 excluded), others cores_per */
    long pos_cmg[64];
    for (int c = 0; c < ncmg; c++) pos_cmg[c] = (long)((c == 0) ? cores_per - 1 : cores_per) * Pt;

    /* per-CMG bf16 K and V buffers, NUMA-local (mbind+MF_MOVE) */
    uint16_t *Kbuf[64] = {0}, *Vbuf[64] = {0};
    for (int c = 0; c < ncmg; c++) {
        size_t elems = (size_t)KVH * pos_cmg[c] * HD;     /* per K, per V */
        size_t bytes = elems * sizeof(uint16_t);
        if (posix_memalign((void **)&Kbuf[c], 2*1024*1024, bytes) != 0) die("posix_memalign(K)", -1);
        if (posix_memalign((void **)&Vbuf[c], 2*1024*1024, bytes) != 0) die("posix_memalign(V)", -1);
        int node = node_of_cpu(cores[c * cores_per]);
        if (node >= 0) {
            unsigned long mask = 1UL << node;
            syscall(SYS_mbind, Kbuf[c], bytes, 2L, &mask, 8UL*sizeof(mask), 2U);
            syscall(SYS_mbind, Vbuf[c], bytes, 2L, &mask, 8UL*sizeof(mask), 2U);
        }
        /* first-touch fill (binds pages to the CMG's node) with small bf16 values
         * so q.K dots stay finite -- uninitialised bf16 bits would yield NaN/Inf. */
        for (size_t i = 0; i < elems; i++) {
            float kv = 0.01f * (float)((long)(i % 13) - 6);
            uint32_t b; memcpy(&b, &kv, 4); Kbuf[c][i] = (uint16_t)(b >> 16);
            kv = 0.01f * (float)((long)(i % 11) - 5);
            memcpy(&b, &kv, 4); Vbuf[c][i] = (uint16_t)(b >> 16);
        }
    }

    float *q = malloc((size_t)QH * HD * sizeof(float));
    for (long i = 0; i < QH * HD; i++) q[i] = 0.01f * (float)((i % 17) - 8);
    /* per-thread partial accumulators [tid][QH*(HD+2)] and score scratch */
    float **part = calloc(nthr, sizeof(float *));
    float **scr  = calloc(nthr, sizeof(float *));
    for (int t = 0; t < nthr; t++) {
        part[t] = malloc((size_t)QH * (HD + 2) * sizeof(float));
        scr[t]  = malloc((size_t)QPKV * Pt * sizeof(float));   /* qh-in-kv x positions */
    }
    float *node_part = malloc((size_t)QH * (HD + 2) * sizeof(float)); /* merged local partial */

    if (my_rank == 0) {
        size_t kvmib = (size_t)ncmg; double tot = 0;
        for (int c = 0; c < ncmg; c++) tot += 2.0 * KVH * pos_cmg[c] * HD * sizeof(uint16_t);
        (void)kvmib;
        logmsg("=== ring-attention DECODE overlap validation (real math) ===\n");
        logmsg("nodes=%d  seq S=%ld  pos/node=%ld  q_heads=%ld kv_heads=%ld head_dim=%ld\n",
               N, S, P, QH, KVH, HD);
        logmsg("groups G=%ld (kv/grp=%ld, qh/grp=%ld)  threads=%d (1 comm + %d compute, %d CMG)\n",
               G, kvpg, qhpg, nthr, nct, ncmg);
        logmsg("KV shard = %.2f MiB bf16  reduce payload(full) = %zu B\n", tot/1048576.0, max_fbytes);
    }

    const unsigned long flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;

    /* ===== shared timing state (written by master) ===== */
    double t_compute = 0, t_comm = 0, t_serial = 0, t_overlap = 0;
    volatile double chk = 0;

    /* ---- ring-reduce of [qh0, qh0+nqh) partials from node_part; thread 0 only.
     * Linear ring 0->1->..->N-1, each rank merges its node_part into the running
     * partial; rank N-1 ends holding the global. fbytes = nqh*(HD+2)*4 (8-aligned,
     * since (HD+2)*4 = 520 = 65*8); the trailing seq sits at +fbytes. ---- */
#define RING_REDUCE(qh0, nqh, token)                                              \
    do {                                                                          \
        size_t _fb = (size_t)(nqh) * (HD + 2) * sizeof(float);                    \
        void *_cb;                                                                \
        if (my_rank == 0) {                                                       \
            memcpy(send_buf_t, node_part + (size_t)(qh0)*(HD+2), _fb);            \
            *(volatile uint64_t *)(send_buf_t + _fb) = (token);                   \
            utofu_stadd_t _src = base_stadd + slot;                               \
            utofu_stadd_t _dst = peer_base[1] + 0;                                \
            for (;;) { rc = utofu_put(vcq, peer_vcq[1], _src, _dst, _fb+8, 0, flags, NULL); \
                       if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(vcq,0,&_cb);} \
            if (rc != UTOFU_SUCCESS) die("put(r0)", rc);                          \
            do { rc = utofu_poll_tcq(vcq,0,&_cb);} while (rc==UTOFU_ERR_NOT_FOUND);\
        } else {                                                                  \
            volatile uint64_t *_rseq = (volatile uint64_t *)(recv_buf + _fb);     \
            double _ts = now_sec();                                               \
            while (*_rseq < (token)) { if (now_sec()-_ts > WAIT_TIMEOUT_SEC) die("reduce wait timeout", -1);} \
            /* merge predecessor's running partial (recv_buf) into our node_part */\
            float *_rp = (float *)recv_buf;                                       \
            for (long _h = 0; _h < (nqh); _h++)                                   \
                merge_partial(node_part + (size_t)((qh0)+_h)*(HD+2), _rp + (size_t)_h*(HD+2), HD); \
            if (my_rank < N - 1) {                                                \
                memcpy(send_buf_t, node_part + (size_t)(qh0)*(HD+2), _fb);        \
                *(volatile uint64_t *)(send_buf_t + _fb) = (token);               \
                utofu_stadd_t _src = base_stadd + slot;                           \
                utofu_stadd_t _dst = peer_base[my_rank+1] + 0;                    \
                for (;;) { rc = utofu_put(vcq, peer_vcq[my_rank+1], _src, _dst, _fb+8, 0, flags, NULL); \
                           if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(vcq,0,&_cb);} \
                if (rc != UTOFU_SUCCESS) die("put(rk)", rc);                      \
                do { rc = utofu_poll_tcq(vcq,0,&_cb);} while (rc==UTOFU_ERR_NOT_FOUND);\
            }                                                                     \
        }                                                                         \
    } while (0)

    /* compute group g's partials for compute-thread `tid` over its position slice;
     * result accumulated into part[tid][qh] for qh in the group. */
#define COMPUTE_GROUP(g, tid)                                                     \
    do {                                                                          \
        int _cmg = (tid) / cores_per;                                             \
        int _ct  = (_cmg == 0) ? cores_per - 1 : cores_per;                       \
        int _idx = (_cmg == 0) ? (tid) - 1 : (tid) - _cmg * cores_per;            \
        long _p0 = (long)_idx * Pt, _p1 = _p0 + Pt;                               \
        long _Pc = pos_cmg[_cmg];                                                 \
        (void)_ct;                                                                \
        for (long _kh = (g)*kvpg; _kh < (g)*kvpg + kvpg; _kh++) {                  \
            const uint16_t *_K = Kbuf[_cmg] + (size_t)_kh*_Pc*HD;                  \
            const uint16_t *_V = Vbuf[_cmg] + (size_t)_kh*_Pc*HD;                  \
            long _q0 = _kh*QPKV;                                                   \
            /* pass 1: scores[qq][p] = scale*q.K ; track running max m[qq] */      \
            float _m[16];                                                         \
            for (long _qq = 0; _qq < QPKV; _qq++) _m[_qq] = -INFINITY;             \
            for (long _p = _p0; _p < _p1; _p++) {                                 \
                const uint16_t *_krow = _K + (size_t)_p*HD;                        \
                for (long _qq = 0; _qq < QPKV; _qq++) {                            \
                    float _s = scale * dot_qk_bf16(q + (_q0+_qq)*HD, _krow, HD);   \
                    scr[tid][_qq*Pt + (_p-_p0)] = _s;                             \
                    if (_s > _m[_qq]) _m[_qq] = _s;                                \
                }                                                                 \
            }                                                                     \
            /* pass 2: w=exp(s-m) (vectorised over p), l=sum w, o += sum w*V */    \
            for (long _qq = 0; _qq < QPKV; _qq++) {                               \
                float *_sc = scr[tid] + _qq*Pt;                                    \
                svbool_t _pg;                                                     \
                for (long _i = 0; _i < Pt; _i += (long)svcntw()) {                \
                    _pg = svwhilelt_b32((uint64_t)_i, (uint64_t)Pt);              \
                    svfloat32_t _v = svsub_n_f32_x(_pg, svld1_f32(_pg, _sc+_i), _m[_qq]); \
                    svst1_f32(_pg, _sc+_i, expf_sve(_pg, _v));                    \
                }                                                                 \
                float _l = 0; for (long _i=0;_i<Pt;_i++) _l += _sc[_i];           \
                float *_o = part[tid] + (_q0+_qq)*(HD+2);                          \
                _o[0] = _m[_qq]; _o[1] = _l;                                       \
                svbool_t _pgd = svptrue_b32();                                     \
                for (long _d = 0; _d < HD; _d += (long)svcntw()) svst1_f32(_pgd, _o+2+_d, svdup_f32(0.0f)); \
                for (long _p = _p0; _p < _p1; _p++) {                             \
                    float _w = _sc[_p-_p0];                                        \
                    const uint16_t *_vrow = _V + (size_t)_p*HD;                    \
                    for (long _d = 0; _d < HD; _d += (long)svcntw()) {            \
                        svuint32_t _vb = svld1uh_u32(_pgd, _vrow+_d);             \
                        svfloat32_t _vf = svreinterpret_f32_u32(svlsl_n_u32_x(_pgd,_vb,16)); \
                        svfloat32_t _of = svld1_f32(_pgd, _o+2+_d);              \
                        svst1_f32(_pgd, _o+2+_d, svmla_n_f32_x(_pgd,_of,_vf,_w)); \
                    }                                                             \
                }                                                                 \
            }                                                                     \
        }                                                                         \
    } while (0)

    /* merge group g's per-thread partials (over compute threads) into node_part.
     * Parallelised over q-heads of the group; thread `tid` owns qh where
     * (qh-qh0) % nthr == tid. Compute threads contributing to qh are those whose
     * CMG holds positions (all of them, every group covers all positions). */
#define MERGE_GROUP(g, tid)                                                       \
    do {                                                                          \
        long _qh0 = (g)*qhpg;                                                     \
        for (long _h = _qh0 + (tid); _h < _qh0 + qhpg; _h += nthr) {              \
            float *_acc = node_part + (size_t)_h*(HD+2);                           \
            int _first = 1;                                                       \
            for (int _t = 1; _t < nthr; _t++) {                                   \
                float *_pp = part[_t] + (size_t)_h*(HD+2);                         \
                if (_first) { memcpy(_acc, _pp, (HD+2)*sizeof(float)); _first = 0;}\
                else merge_partial(_acc, _pp, HD);                                \
            }                                                                     \
        }                                                                         \
    } while (0)

    /* bootstrap the ring (token 1) so every recv seq is primed before timing. */
    {
        for (long h = 0; h < QH; h++) { node_part[h*(HD+2)] = 0; node_part[h*(HD+2)+1] = 1; }
        int got = 0; double t0 = now_sec();
        for (int a = 0; a < 400 && !got; a++) {
            uint64_t w = 1; RING_REDUCE(0, QH, w);
            if (my_rank == 0 || my_rank == N-1) got = 1;          /* r0 sent, last received */
            else { volatile uint64_t *rs=(volatile uint64_t*)(recv_buf + (size_t)QH*(HD+2)*4); if (*rs>=1) got=1; }
            if (got && a >= 4) break;
            usleep(20000);
            if (now_sec()-t0 > WAIT_TIMEOUT_SEC) break;
        }
    }

    uint64_t tok = 2;

    /* ===== A) compute-only (the nct compute threads, full shard) ===== */
    {
        double el = 0;
#pragma omp parallel num_threads(nthr)
        {
            int tid = omp_get_thread_num();
            cpu_set_t s; CPU_ZERO(&s); CPU_SET(cores[tid], &s); sched_setaffinity(0, sizeof s, &s);
            /* tid 0 is the comm driver in the overlap/serial modes, so it does NOT
             * compute here either -- keeps compute-only apples-to-apples (nct threads). */
            for (long it = 0; it < WARMUP + ITERS; it++) {
#pragma omp barrier
#pragma omp master
                if (it == WARMUP) el = now_sec();
                for (long g = 0; g < G; g++) {
                    if (tid != 0) COMPUTE_GROUP(g, tid);
                }
#pragma omp barrier
                for (long g = 0; g < G; g++) MERGE_GROUP(g, tid);
#pragma omp barrier
#pragma omp master
                if (it == WARMUP + ITERS - 1) el = now_sec() - el;
            }
        }
        t_compute = el / ITERS;
        chk += node_part[1];
    }

    /* ===== B) comm-only (thread 0 drives N-1 full reduces, no compute) ===== */
    {
        double t0 = now_sec();
        for (long it = 0; it < WARMUP + ITERS; it++) {
            if (it == WARMUP) t0 = now_sec();
            uint64_t w = tok++; RING_REDUCE(0, QH, w);
        }
        t_comm = (now_sec() - t0) / ITERS;
    }

    /* ===== C) SERIAL: compute (nct threads) then ONE full reduce ===== */
    {
        double el = 0;
#pragma omp parallel num_threads(nthr)
        {
            int tid = omp_get_thread_num();
            cpu_set_t s; CPU_ZERO(&s); CPU_SET(cores[tid], &s); sched_setaffinity(0, sizeof s, &s);
            for (long it = 0; it < WARMUP + ITERS; it++) {
#pragma omp barrier
#pragma omp master
                if (it == WARMUP) el = now_sec();
                for (long g = 0; g < G; g++) if (tid != 0) COMPUTE_GROUP(g, tid);
#pragma omp barrier
                for (long g = 0; g < G; g++) MERGE_GROUP(g, tid);
#pragma omp barrier
#pragma omp master
                { uint64_t w = tok++; RING_REDUCE(0, QH, w); }
#pragma omp barrier
#pragma omp master
                if (it == WARMUP + ITERS - 1) el = now_sec() - el;
            }
        }
        t_serial = el / ITERS;
        chk += node_part[1];
    }

    /* ===== D) OVERLAP: head-group pipeline. Stage g: threads 1..47 compute
     * group g while thread 0 reduces group g-1. ===== */
    {
        double el = 0;
#pragma omp parallel num_threads(nthr)
        {
            int tid = omp_get_thread_num();
            cpu_set_t s; CPU_ZERO(&s); CPU_SET(cores[tid], &s); sched_setaffinity(0, sizeof s, &s);
            for (long it = 0; it < WARMUP + ITERS; it++) {
#pragma omp barrier
#pragma omp master
                if (it == WARMUP) el = now_sec();
                uint64_t base_tok;
#pragma omp master
                base_tok = tok, tok += G;
                for (long st = 0; st <= G; st++) {
                    /* concurrent: compute group st (compute threads) || reduce group st-1 (thread 0) */
                    if (st < G && tid != 0) COMPUTE_GROUP(st, tid);
                    if (st >= 1 && tid == 0) {
                        long g = st - 1;
                        uint64_t w;
#pragma omp atomic read
                        w = base_tok;            /* read once; same value across stages */
                        RING_REDUCE(g*qhpg, qhpg, w + g);
                    }
#pragma omp barrier
                    if (st < G) MERGE_GROUP(st, tid);   /* all threads merge just-computed group */
#pragma omp barrier
                }
#pragma omp master
                if (it == WARMUP + ITERS - 1) el = now_sec() - el;
            }
        }
        t_overlap = el / ITERS;
        chk += node_part[1];
    }

    if (my_rank == 0) {
        double cmp = t_compute*1e6, com = t_comm*1e6, ser = t_serial*1e6, ovl = t_overlap*1e6;
        double model_serial = cmp + com;
        double model_max = cmp > com ? cmp : com;
        double ideal_save = ser - model_max;                 /* if overlap were perfect */
        double real_save  = ser - ovl;
        logmsg("\n--- measured per decode step (us) ---\n");
        logmsg("compute-only (%dT full shard) = %.1f\n", nct, cmp);
        logmsg("comm-only    (N-1 full reduce)= %.1f\n", com);
        logmsg("SERIAL  (compute + full reduce)= %.1f\n", ser);
        logmsg("OVERLAP (head-group pipe, G=%ld)= %.1f\n", G, ovl);
        logmsg("\n--- vs the model ---\n");
        logmsg("model serial  = compute+comm     = %.1f\n", model_serial);
        logmsg("model max()   = max(compute,comm)= %.1f  (the optimistic overlap claim)\n", model_max);
        logmsg("overlap efficiency = (serial-overlap)/(serial-max) = %.0f%%  (100%%=model met)\n",
               ideal_save > 0 ? 100.0*real_save/ideal_save : 0.0);
        logmsg("overlap vs serial = %+.1f%%   overlap vs model max() = %+.1f%%\n",
               ser>0?100.0*(ovl-ser)/ser:0.0, model_max>0?100.0*(ovl-model_max)/model_max:0.0);
        logmsg("note: G groups make the reduce pay the ~1.23us fixed per-hop cost G times\n");
        logmsg("chk=%.3e\n", (double)chk);
    }

    utofu_dereg_mem(vcq, base_stadd, 0);
    utofu_free_vcq(vcq);
    free(region); free(q); free(node_part);
    for (int t = 0; t < nthr; t++) { free(part[t]); free(scr[t]); }
    free(part); free(scr);
    for (int c = 0; c < ncmg; c++) { free(Kbuf[c]); free(Vbuf[c]); }
    return 0;
}
