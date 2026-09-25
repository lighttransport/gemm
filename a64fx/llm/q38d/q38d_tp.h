/* q38d_tp.h - tensor-parallel transport for the q38d decode engine.
 *
 * One process per node (mpiexec places the identical, MPI-free binary);
 * ranks are found by matching the node's uTofu coordinates in the topology
 * file written for this allocation by a64fx/utofu-tests/tofu_topo_helper.
 * Collectives: residual += sum of the ranks' partial vectors (one-round
 * all-to-all, every rank folds in rank order, so all ranks hold bitwise
 * identical residuals), and a global (max, index) for the output head.
 * Build with -DQ38D_TP and link -ltofucom.
 */
#ifndef Q38D_TP_H
#define Q38D_TP_H

static int tp_n = 1, tp_r = 0;   /* ranks, my rank */

#ifdef Q38D_TP
#include <sys/mman.h>
#include <unistd.h>
#include <utofu.h>
#include "../../utofu-tests/tofu_demo.h"
#include "../../utofu-tests/tp_allreduce.h"

#define Q38D_TP_MAXN 4
#define Q38D_TP_BAR_STAG DEMO_STAG
static utofu_vcq_hdl_t tp_vcq;
static utofu_vcq_id_t tp_peer[Q38D_TP_MAXN];
static utofu_stadd_t tp_bar_base, tp_peer_bar[Q38D_TP_MAXN];
static char *tp_bar;
static uint64_t tp_bar_seq;
static tp_comm tp_c;
static float *tp_sendbuf;         /* partial vector handed to the collective */
/* sliced collectives: slice k (1/tp_nsl of the vector) is reduced by lane 0
 * of CMG k through its own VCQ on TNI 1+k and CMG-k-bound regions */
#define Q38D_TP_NSL 4
static int tp_nsl = 4;
static utofu_vcq_hdl_t tp_svcq[Q38D_TP_NSL];
static tp_comm tp_sc[Q38D_TP_NSL];

static void tp_fatal(const char *what, int rc) {
    fprintf(stderr, "q38d_tp rank=%d FATAL %s rc=%d\n", tp_r, what, rc);
    exit(1);
}
static void tp_put_wait(int peer, utofu_stadd_t src, utofu_stadd_t dst, size_t bytes) {
    void *cb;
    int rc;
    do {
        rc = utofu_put(tp_vcq, tp_peer[peer], src, dst, bytes, 0, UTOFU_ONESIDED_FLAG_TCQ_NOTICE, NULL);
        if (rc == UTOFU_ERR_BUSY) utofu_poll_tcq(tp_vcq, 0, &cb);
    } while (rc == UTOFU_ERR_BUSY);
    if (rc != UTOFU_SUCCESS) tp_fatal("utofu_put", rc);
    do { rc = utofu_poll_tcq(tp_vcq, 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND);
    if (rc != UTOFU_SUCCESS) tp_fatal("utofu_poll_tcq", rc);
    struct utofu_mrq_notice nt;
    while (utofu_poll_mrq(tp_vcq, 0, &nt) == UTOFU_SUCCESS) {}
}
/* rank 0 collects arrivals, then releases; slot 0 send, 1..n arrive, n+1 release */
static void tp_barrier(void) {
    uint64_t tok = ++tp_bar_seq;
    size_t L = DEMO_CACHE_LINE;
    if (tp_r == 0) {
        double t0 = tp_ar_now();
        for (int r = 1; r < tp_n; r++) {
            volatile uint64_t *p = (volatile uint64_t *)(tp_bar + L * (1 + r));
            while (*p < tok) { tp_ar_flag_inval(p); if (tp_ar_now() - t0 > 120) tp_fatal("barrier timeout (peer missing)", r); }
        }
        for (int r = 1; r < tp_n; r++) {
            *(uint64_t *)tp_bar = tok;
            tp_put_wait(r, tp_bar_base, tp_peer_bar[r] + L * (1 + tp_n), 8);
        }
    } else {
        *(uint64_t *)tp_bar = tok;
        tp_put_wait(0, tp_bar_base, tp_peer_bar[0] + L * (1 + tp_r), 8);
        volatile uint64_t *p = (volatile uint64_t *)(tp_bar + L * (1 + tp_n));
        double t0 = tp_ar_now();
        while (*p < tok) { tp_ar_flag_inval(p); if (tp_ar_now() - t0 > 120) tp_fatal("barrier timeout (rank 0 missing)", 0); }
    }
}
static void *tp_map_node(size_t bytes, int cmg) {
    size_t n = (bytes + 65535) & ~(size_t)65535;
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) tp_fatal("mmap", errno);
    if (cmg >= 0) {
        unsigned long node = 1ul << (4 + cmg);
        if (syscall(SYS_mbind, p, n, 2 /*MPOL_BIND*/, &node, 64ul, 0ul)) perror("mbind");
    }
    memset(p, 0, n);
    return p;
}
static void *tp_map(size_t bytes) { return tp_map_node(bytes, 0); }

/* Slice collective (recursive doubling over 2 or 4 ranks). Region layout:
 * send[2 gen][2 step][count] | recv[2 gen][2 step][count] | trailer[2][2] |
 * trailer source[2][2] (one 256-byte line each: a CPU-written line must never
 * share a line with a Put target). The payload Put is followed by a STRONG_ORDER
 * trailer Put, and the receiver invalidates the payload lines after seeing
 * the trailer, so a sum never reads a partly landed or stale payload. Slots
 * alternate with the collective's parity: a peer can reuse a parity only
 * after every rank has entered the next collective. */
typedef struct {
    utofu_vcq_hdl_t vcq;
    utofu_vcq_id_t peer[Q38D_TP_MAXN];
    utofu_stadd_t base, pbase[Q38D_TP_MAXN];
    char *reg;
    int count, nr;
    size_t pay;          /* payload slot bytes (256-aligned) */
    uint64_t seq;
} q38d_ar;
static q38d_ar tp_ar[Q38D_TP_NSL];
static int tp_ar_mine = 1;   /* Q38D_TP_AR=0: use the tp_allreduce.h collective */
static size_t q38d_ar_send_off(const q38d_ar *a, int gen, int st) { return ((size_t)gen * 2 + st) * a->pay; }
static size_t q38d_ar_recv_off(const q38d_ar *a, int gen, int st) { return ((size_t)4 + gen * 2 + st) * a->pay; }
static size_t q38d_ar_trl_off(const q38d_ar *a, int gen, int st) { return (size_t)8 * a->pay + ((size_t)gen * 2 + st) * 256; }
static size_t q38d_ar_tsrc_off(const q38d_ar *a, int gen, int st) { return (size_t)8 * a->pay + 1024 + ((size_t)gen * 2 + st) * 256; }
static void q38d_ar_put(q38d_ar *a, int peer, utofu_stadd_t src, utofu_stadd_t dst, size_t len, unsigned long flags) {
    void *cb;
    int rc;
    for (;;) {
        rc = utofu_put(a->vcq, a->peer[peer], src, dst, len, 0, flags | UTOFU_ONESIDED_FLAG_TCQ_NOTICE, NULL);
        if (rc != UTOFU_ERR_BUSY) break;
        utofu_poll_tcq(a->vcq, 0, &cb);
    }
    if (rc != UTOFU_SUCCESS) tp_fatal("slice utofu_put", rc);
}
static void q38d_ar_sum(q38d_ar *a, float *buf) {
    uint64_t tok = ++a->seq;
    int gen = (int)(tok & 1), n = a->count;
    size_t pb = (size_t)n * sizeof(float);
    for (int st = 0; st < a->nr; st++) {
        int partner = tp_r ^ (1 << st);
        char *sb = a->reg + q38d_ar_send_off(a, gen, st);
        memcpy(sb, buf, pb);
        *(volatile uint64_t *)(a->reg + q38d_ar_tsrc_off(a, gen, st)) = tok;   /* trailer source */
        q38d_ar_put(a, partner, a->base + q38d_ar_send_off(a, gen, st), a->pbase[partner] + q38d_ar_recv_off(a, gen, st), pb, 0);
        q38d_ar_put(a, partner, a->base + q38d_ar_tsrc_off(a, gen, st), a->pbase[partner] + q38d_ar_trl_off(a, gen, st), 8,
                    UTOFU_ONESIDED_FLAG_STRONG_ORDER);
        volatile uint64_t *trl = (volatile uint64_t *)(a->reg + q38d_ar_trl_off(a, gen, st));
        double t0 = 0;
        unsigned long spins = 0;
        while (*trl < tok) {
            tp_ar_flag_inval(trl);
            if ((++spins & 0xFFFFF) == 0) {
                if (!t0) t0 = tp_ar_now();
                else if (tp_ar_now() - t0 > 60) tp_fatal("slice collective timeout", st);
            }
        }
        const char *rb = a->reg + q38d_ar_recv_off(a, gen, st);
        for (size_t o = 0; o < pb; o += 256) __asm__ __volatile__("dc civac, %0" :: "r"(rb + o) : "memory");
        __asm__ __volatile__("dsb sy" ::: "memory");
        const float *r = (const float *)rb;
        const svbool_t pf = svptrue_b32();
        for (int i = 0; i < n; i += 16) svst1_f32(pf, buf + i, svadd_f32_x(pf, svld1_f32(pf, buf + i), svld1_f32(pf, r + i)));
        /* both local completions of this step (send slots are per step and parity) */
        void *cb;
        int done = 0, rc;
        while (done < 2) {
            rc = utofu_poll_tcq(a->vcq, 0, &cb);
            if (rc == UTOFU_SUCCESS) done++;
            else if (rc != UTOFU_ERR_NOT_FOUND) tp_fatal("slice poll_tcq", rc);
        }
    }
    struct utofu_mrq_notice nt;
    while (utofu_poll_mrq(a->vcq, 0, &nt) == UTOFU_SUCCESS) {}
}

/* Call from the thread that will run the collectives (pinned to CMG 0). */
static void q38d_tp_init(int n, int max_count) {
    tp_n = n;
    if (n == 1) return;
    if (n > Q38D_TP_MAXN) tp_fatal("too many ranks", n);
    const char *path = getenv("TOFU_TOPO_PATH") ? getenv("TOFU_TOPO_PATH") : TOPO_PATH;
    FILE *f = fopen(path, "r");
    if (!f) tp_fatal("open topology (run tofu_topo_helper in this allocation)", errno);
    uint8_t topo[Q38D_TP_MAXN][TOFU_NCOORDS];
    int cnt = 0;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        unsigned rank, c[TOFU_NCOORDS];
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "%u %u %u %u %u %u %u", &rank, &c[0], &c[1], &c[2], &c[3], &c[4], &c[5]) != 7) continue;
        if (cnt < Q38D_TP_MAXN) for (int k = 0; k < TOFU_NCOORDS; k++) topo[cnt][k] = (uint8_t)c[k];
        cnt++;
    }
    fclose(f);
    if (cnt < n) tp_fatal("topology has fewer ranks than Q38D_TP", cnt);
    uint8_t mine[TOFU_NCOORDS];
    int rc = utofu_query_my_coords(mine);
    if (rc != UTOFU_SUCCESS) tp_fatal("query coords", rc);
    tp_r = -1;
    for (int r = 0; r < n; r++) if (!memcmp(mine, topo[r], TOFU_NCOORDS)) tp_r = r;
    if (tp_r < 0) tp_fatal("node not among the first Q38D_TP topology ranks", -1);
    utofu_tni_id_t *tnis = NULL;
    size_t ntni = 0;
    if ((rc = utofu_get_onesided_tnis(&tnis, &ntni)) != UTOFU_SUCCESS || !ntni) tp_fatal("onesided TNIs", rc);
    utofu_tni_id_t tni = tnis[DEMO_TNI_INDEX];
    if (getenv("Q38D_TP_SLICES")) tp_nsl = atoi(getenv("Q38D_TP_SLICES"));
    if (getenv("Q38D_TP_AR")) tp_ar_mine = atoi(getenv("Q38D_TP_AR"));
    if (tp_nsl < 1 || tp_nsl > Q38D_TP_NSL || (size_t)tp_nsl + 1 > ntni) tp_nsl = 1;
    utofu_tni_id_t stni[Q38D_TP_NSL];
    for (int k = 0; k < tp_nsl; k++) stni[k] = tnis[1 + k];
    free(tnis);
    if ((rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &tp_vcq)) != UTOFU_SUCCESS) tp_fatal("create VCQ", rc);
    utofu_vcq_id_t my;
    if ((rc = utofu_query_vcq_id(tp_vcq, &my)) != UTOFU_SUCCESS) tp_fatal("query VCQ", rc);
    size_t bar_bytes = DEMO_CACHE_LINE * (size_t)(n + 2);
    tp_bar = tp_map(bar_bytes);
    if ((rc = utofu_reg_mem_with_stag(tp_vcq, tp_bar, bar_bytes, Q38D_TP_BAR_STAG, 0, &tp_bar_base)) != UTOFU_SUCCESS)
        tp_fatal("register barrier", rc);
    /* a previous run's registrations on the peers can outlive their processes
     * for a moment and answer the query with dead addresses: let them settle */
    {
        double settle = getenv("Q38D_TP_SETTLE") ? atof(getenv("Q38D_TP_SETTLE")) : 8.0, t0 = tp_ar_now();
        while (tp_ar_now() - t0 < settle) usleep(100000);
    }
    for (int r = 0; r < n; r++) {
        if (r == tp_r) { tp_peer[r] = my; tp_peer_bar[r] = tp_bar_base; continue; }
        if ((rc = utofu_construct_vcq_id(topo[r], tni, DEMO_CQ_ID, DEMO_CMP_ID, &tp_peer[r])) != UTOFU_SUCCESS)
            tp_fatal("construct peer VCQ", rc);
        utofu_set_vcq_id_path(&tp_peer[r], NULL);
        /* the peer may not have registered yet: retry */
        double t0 = tp_ar_now();
        while ((rc = utofu_query_stadd(tp_peer[r], Q38D_TP_BAR_STAG, &tp_peer_bar[r])) != UTOFU_SUCCESS)
            if (tp_ar_now() - t0 > 60) tp_fatal("query peer barrier", rc);
    }
    tp_barrier();
    /* recursive doubling: for 2 or 4 ranks every step adds two partial sums,
     * so all ranks end bitwise identical (IEEE addition commutes) */
    tp_comm_config cfg = tp_comm_env_config();
    cfg.a2a_max = max_count;
    cfg.deterministic = 0;
    size_t need = tp_comm_region_size(n, max_count, &cfg);
    void *region = tp_map(need);
    if (tp_comm_init_external(&tp_c, tp_vcq, tp_peer, tp_r, n, max_count, tp_barrier, &cfg, region,
                              (need + 65535) & ~(size_t)65535) != 0)
        tp_fatal("tp_comm_init", -1);
    tp_sendbuf = tp_map((size_t)max_count * sizeof(float));
    tp_barrier();
    for (int k = 0; k < tp_nsl; k++) {
        if ((rc = utofu_create_vcq_with_cmp_id(stni[k], DEMO_CMP_ID, 0, &tp_svcq[k])) != UTOFU_SUCCESS) tp_fatal("create slice VCQ", rc);
        utofu_vcq_id_t sp[Q38D_TP_MAXN];
        for (int r = 0; r < n; r++) {
            if (r == tp_r) { if ((rc = utofu_query_vcq_id(tp_svcq[k], &sp[r])) != UTOFU_SUCCESS) tp_fatal("query slice VCQ", rc); continue; }
            if ((rc = utofu_construct_vcq_id(topo[r], stni[k], DEMO_CQ_ID, DEMO_CMP_ID, &sp[r])) != UTOFU_SUCCESS)
                tp_fatal("construct slice peer VCQ", rc);
            utofu_set_vcq_id_path(&sp[r], NULL);
        }
        int cnt_k = max_count / tp_nsl;
        size_t need_k = tp_comm_region_size(n, cnt_k, &cfg);
        void *reg = tp_map_node(need_k, k);
        if (tp_comm_init_region_ex(&tp_sc[k], tp_svcq[k], sp, tp_r, n, cnt_k, tp_barrier, 20 + k, &cfg, reg,
                                   (need_k + 65535) & ~(size_t)65535) != 0)
            tp_fatal("slice tp_comm_init", k);
        /* own slice collective on the same VCQ (steering tag 40 + k) */
        q38d_ar *a = &tp_ar[k];
        a->vcq = tp_svcq[k];
        a->count = cnt_k;
        a->nr = n == 4 ? 2 : 1;
        a->pay = ((size_t)cnt_k * sizeof(float) + 255) & ~(size_t)255;
        size_t rbytes = 8 * a->pay + 8 * 256;
        a->reg = tp_map_node(rbytes, k);
        for (size_t o = 0; o < rbytes; o += 256) __asm__ __volatile__("dc civac, %0" :: "r"(a->reg + o) : "memory");
        __asm__ __volatile__("dsb sy" ::: "memory");
        if ((rc = utofu_reg_mem_with_stag(a->vcq, a->reg, (rbytes + 65535) & ~(size_t)65535, 40 + k, 0, &a->base)) != UTOFU_SUCCESS)
            tp_fatal("register slice region", rc);
        for (int r = 0; r < n; r++) a->peer[r] = sp[r];
        tp_barrier();
        for (int r = 0; r < n; r++) {
            if (r == tp_r) { a->pbase[r] = a->base; continue; }
            if ((rc = utofu_query_stadd(a->peer[r], 40 + k, &a->pbase[r])) != UTOFU_SUCCESS) tp_fatal("query slice region", rc);
        }
    }
    tp_barrier();
    fprintf(stderr, "q38d_tp: rank %d/%d ready, %d collective slices\n", tp_r, tp_n, tp_nsl);
}
/* residual[0..count) += sum over ranks of part[0..count) (rank-order fold) */
static void q38d_tp_sum_add(float *part, float *residual, int count) {
    if (tp_n > 1 && tp_c.a2a) {
        uint64_t tok = ++tp_c.seq;
        tp_ar_sum_a2a_add(&tp_c, part, residual, count, tok);
        return;
    }
    if (tp_n > 1) tp_allreduce_sum(&tp_c, part, count);
    const svbool_t pf = svptrue_b32();
    for (int i = 0; i < count; i += 16)
        svst1_f32(pf, residual + i, svadd_f32_x(pf, svld1_f32(pf, residual + i), svld1_f32(pf, part + i)));
}
/* slice k of the vector (called by the slice's own thread) */
static void q38d_tp_sum_add_slice(int k, float *part, float *residual, int count) {
    if (tp_ar_mine) q38d_ar_sum(&tp_ar[k], part);
    else tp_allreduce_sum(&tp_sc[k], part, count);
    const svbool_t pf = svptrue_b32();
    for (int i = 0; i < count; i += 16)
        svst1_f32(pf, residual + i, svadd_f32_x(pf, svld1_f32(pf, residual + i), svld1_f32(pf, part + i)));
}
/* global argmax: (val, global index) -> identical result on every rank */
static void q38d_tp_argmax(float *val, int *idx) {
    if (tp_n == 1) return;
    int32_t i32 = *idx;
    tp_allreduce_argmax(&tp_c, val, &i32);
    *idx = i32;
}
#else
static int tp_nsl = 1;
static void q38d_tp_sum_add_slice(int k, float *part, float *residual, int count) { (void)k; for (int i = 0; i < count; i++) residual[i] += part[i]; }
static void q38d_tp_init(int n, int max_count) { (void)max_count; if (n != 1) { fprintf(stderr, "q38d: built without Q38D_TP\n"); exit(1); } }
static void q38d_tp_sum_add(float *part, float *residual, int count) { for (int i = 0; i < count; i++) residual[i] += part[i]; }

static void q38d_tp_argmax(float *val, int *idx) { (void)val; (void)idx; }
#endif
#endif
