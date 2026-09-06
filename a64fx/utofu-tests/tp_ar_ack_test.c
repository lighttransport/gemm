/* Standalone loss-injection test for the tp_allreduce ack/retransmit reliability layer (TP_AR_ACK).
 *
 * Launches MPI-free on N nodes (1 rank/node) exactly like ds4f_ep_runner: reads tofu_topo.txt, brings
 * up one TNI + VCQ, registers a barrier region, constructs peer VCQs by convention, then tp_comm_init.
 * It then runs REPS sum-all-reduces of a buffer whose rank-r value is (r+1), so every element must
 * equal N(N+1)/2 EXACTLY (fp32-exact for small integers) -- a direct arithmetic check of the reduce,
 * stronger than an end-to-end token compare. A max-all-reduce round is checked too (== N).
 *
 * Purpose: with TP_AR_DROP=M injecting 1-in-M lost payload Puts, prove that
 *   TP_AR_ACK=1  -> completes with 0 mismatches (retransmit recovers every loss),
 *   TP_AR_ACK=0  -> hangs then exit(1) on the first drop (the production failure).
 *
 * Build:  make tp_ar_ack_test            (native fcc, -ltofucom)
 * Run  :  cd a64fx/utofu-tests && mpiexec -np 11 --vcoordfile ../llm/vcoord_ds4f.txt ./tofu_topo_helper
 *         REPS=20000 COUNT=4096 TP_AR_ACK=1 TP_AR_DROP=50 \
 *           mpiexec -np 11 --vcoordfile ../llm/vcoord_ds4f.txt ./tp_ar_ack_test
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <utofu.h>
#include "tofu_demo.h"
#include "tp_allreduce.h"

#define MAX_NODES 128
#define RUN_STAG  DEMO_STAG
#define WAIT_TIMEOUT_SEC 120.0

static int             N, MyRank;
static char           *Region;
static size_t          SEND_OFF, BAR_BASE, SlotSend, SlotB;
static utofu_vcq_hdl_t Vcq;
static utofu_stadd_t   Base;
static utofu_vcq_id_t  PeerVcq[MAX_NODES];
static utofu_stadd_t   PeerBase[MAX_NODES];
static const unsigned long FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
static uint64_t        Bt = 1;

static void die(const char *what, int rc) { fprintf(stderr, "FATAL rank?: %s (rc=%d)\n", what, rc); exit(1); }
static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static int envi(const char *k, int d) { const char *v = getenv(k); return (v && *v) ? atoi(v) : d; }

static inline size_t bar_recv_off(int s) { return BAR_BASE + (size_t)s * SlotB; }
static inline size_t bar_go_off(void)    { return BAR_BASE + (size_t)N * SlotB; }

static void put_issue(utofu_vcq_id_t pv, utofu_stadd_t s, utofu_stadd_t d, size_t len) {
    int rc; void *cb;
    for (;;) { rc = utofu_put(Vcq, pv, s, d, len, 0, FLAGS, NULL);
               if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(Vcq, 0, &cb); }
    if (rc != UTOFU_SUCCESS) die("utofu_put", rc);
    do { rc = utofu_poll_tcq(Vcq, 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND);
    if (rc != UTOFU_SUCCESS) die("utofu_poll_tcq", rc);
}
static void wait_ge(volatile uint64_t *q, uint64_t v, const char *what) {
    double ts = now_sec();
    while (*q < v) if (now_sec() - ts > WAIT_TIMEOUT_SEC) die(what, -1);
}
static void barrier_robust(int robust) {   /* fan-in to rank 0, fan-out release (== ds4f_ep_runner) */
    uint64_t t = ++Bt; char *sb = Region + SEND_OFF;
    if (MyRank == 0) {
        for (int s = 1; s < N; s++) wait_ge((volatile uint64_t *)(Region + bar_recv_off(s)), t, "barrier fan-in");
        for (int s = 1; s < N; s++) { *(volatile uint64_t *)sb = t;
            put_issue(PeerVcq[s], Base + SEND_OFF, PeerBase[s] + bar_go_off(), 8); }
    } else {
        volatile uint64_t *go = (volatile uint64_t *)(Region + bar_go_off()); double ts = now_sec();
        do { *(volatile uint64_t *)sb = t;
            put_issue(PeerVcq[0], Base + SEND_OFF, PeerBase[0] + bar_recv_off(MyRank), 8);
            if (!robust) { wait_ge(go, t, "barrier release"); break; }
            for (int a = 0; a < 50 && *go < t; a++) usleep(2000);
            if (now_sec() - ts > WAIT_TIMEOUT_SEC) die("bootstrap barrier timeout", -1);
        } while (*go < t);
    }
}
static void barrier(void) { barrier_robust(0); }

static int read_topo(uint8_t coords[][TOFU_NCOORDS]) {
    const char *tp = getenv("TOFU_TOPO_PATH"); if (!tp || !*tp) tp = TOPO_PATH;
    FILE *f = fopen(tp, "r"); if (!f) { fprintf(stderr, "cannot open %s (run tofu_topo_helper first)\n", tp); exit(1); }
    int n = 0; char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        unsigned r, c[TOFU_NCOORDS];
        if (sscanf(line, "%u %u %u %u %u %u %u", &r, &c[0], &c[1], &c[2], &c[3], &c[4], &c[5]) != 7) { fprintf(stderr, "bad line\n"); exit(1); }
        for (int k = 0; k < TOFU_NCOORDS; k++) coords[n][k] = (uint8_t)c[k];
        n++;
    }
    fclose(f); return n;
}

int main(void) {
    int rc, COUNT = envi("COUNT", 4096), REPS = envi("REPS", 10000);

    utofu_tni_id_t *tni_ids = NULL; size_t num_tnis = 0;
    rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis); if (rc != UTOFU_SUCCESS) die("get_onesided_tnis", rc);
    if (num_tnis < 1) die("no onesided TNIs", -1);
    uint8_t my_coords[TOFU_NCOORDS] = {0};
    rc = utofu_query_my_coords(my_coords); if (rc != UTOFU_SUCCESS) die("query_my_coords", rc);
    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    N = read_topo(topo);
    MyRank = -1;
    for (int r = 0; r < N; r++) if (memcmp(topo[r], my_coords, TOFU_NCOORDS) == 0) MyRank = r;
    if (MyRank == -1) die("my coords not in topo", -1);
    { char en[64]; snprintf(en, sizeof en, "tp_ar_stderr_rank%02d.txt", MyRank);   /* mpiexec drops rank stderr */
      freopen(en, "w", stderr); setvbuf(stderr, NULL, _IONBF, 0); }
    fprintf(stderr, "rank %d/%d up (COUNT=%d REPS=%d)\n", MyRank, N, COUNT, REPS);

    /* barrier region (own cache line per remote-written slot) */
    SlotSend = DEMO_CACHE_LINE; SlotB = DEMO_CACHE_LINE; SEND_OFF = 0; BAR_BASE = SlotSend;
    size_t region_sz = BAR_BASE + (size_t)(N + 1) * SlotB;
    if (posix_memalign((void **)&Region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign", -1);
    memset(Region, 0, region_sz);
    for (size_t off = 0; off < region_sz; off += DEMO_CACHE_LINE) __asm__ __volatile__("dc civac, %0" :: "r"(Region + off) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");

    utofu_tni_id_t tni = tni_ids[0];
    rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &Vcq); if (rc != UTOFU_SUCCESS) die("create_vcq", rc);
    utofu_vcq_id_t my_real;
    rc = utofu_query_vcq_id(Vcq, &my_real); if (rc != UTOFU_SUCCESS) die("query_vcq_id", rc);
    rc = utofu_reg_mem_with_stag(Vcq, Region, region_sz, RUN_STAG, 0, &Base); if (rc != UTOFU_SUCCESS) die("reg_mem", rc);
    for (int r = 0; r < N; r++) {
        if (r == MyRank) { PeerVcq[r] = my_real; PeerBase[r] = Base; continue; }
        rc = utofu_construct_vcq_id(topo[r], tni, DEMO_CQ_ID, DEMO_CMP_ID, &PeerVcq[r]); if (rc != UTOFU_SUCCESS) die("construct_vcq_id", rc);
        utofu_set_vcq_id_path(&PeerVcq[r], NULL);
        rc = utofu_query_stadd(PeerVcq[r], RUN_STAG, &PeerBase[r]); if (rc != UTOFU_SUCCESS) die("query_stadd", rc);
    }
    free(tni_ids);
    barrier_robust(1);   /* everyone registered before tp_comm_init's internal barrier */

    int ar2d = envi("TP_AR_2D", 0);   /* >0: also time the 2-level AR with A=ar2d groups (must divide N) */
    tp_comm comm;
    if (tp_comm_init(&comm, Vcq, PeerVcq, MyRank, N, COUNT, barrier) != 0) die("tp_comm_init", -1);

    float *buf = (float *)malloc((size_t)COUNT * sizeof(float));
    float exp_sum = (float)(N * (N + 1) / 2);     /* sum_{r=0..N-1}(r+1) */
    float exp_max = (float)N;                     /* max_{r}(r+1) */
    long bad_sum = 0, bad_max = 0;
    double t0 = now_sec();
    for (int it = 0; it < REPS; it++) {
        for (int i = 0; i < COUNT; i++) buf[i] = (float)(MyRank + 1);
        tp_allreduce_sum(&comm, buf, COUNT);
        for (int i = 0; i < COUNT; i++) if (buf[i] != exp_sum) { bad_sum++; break; }
        /* one max-reduce every 8th iter (shares the seq counter -> must run in lockstep on all ranks) */
        if ((it & 7) == 0) {
            for (int i = 0; i < COUNT; i++) buf[i] = (float)(MyRank + 1);
            tp_allreduce_max(&comm, buf, COUNT);
            for (int i = 0; i < COUNT; i++) if (buf[i] != exp_max) { bad_max++; break; }
        }
    }
    double dt = now_sec() - t0;
    barrier_robust(1);

    /* ---- optional: same-allocation comparison against the 2-level (hierarchical) AR ---- */
    double dt2 = -1.0; long bad2 = 0; int A2 = 0, B2 = 0;
    if (ar2d > 0 && N % ar2d == 0) {
        tp_comm_free(&comm);                 /* free STAG7 before the 2D row re-registers it */
        barrier_robust(1);
        tp_comm row, col;
        if (tp_comm_init_2d(&row, &col, Vcq, PeerVcq, MyRank, N, ar2d, COUNT, barrier) != 0) die("tp_comm_init_2d", -1);
        A2 = ar2d; B2 = N / ar2d;
        double t2 = now_sec();
        for (int it = 0; it < REPS; it++) {
            for (int i = 0; i < COUNT; i++) buf[i] = (float)(MyRank + 1);
            tp_allreduce_sum_2d(&row, &col, buf, COUNT);
            for (int i = 0; i < COUNT; i++) if (buf[i] != exp_sum) { bad2++; break; }
        }
        dt2 = now_sec() - t2;
        barrier_robust(1);
        tp_comm_free_2d(&row, &col);
    } else if (ar2d > 0 && MyRank == 0) {
        fprintf(stderr, "tp_ar_ack_test: TP_AR_2D=%d does not divide N=%d; skipping 2D\n", ar2d, N);
    }

    if (MyRank == 0) {
        /* mpiexec does not forward rank stderr, so also write the verdict to a file (== ds4f_ep_runner). */
        const char *pass = (bad_sum == 0 && bad_max == 0 && bad2 == 0) ? "PASS" : "FAIL";
        char line[640];
        int n = snprintf(line, sizeof line,
                 "tp_ar_ack_test N=%d COUNT=%d REPS=%d ack=%d drop=%lu | flat: %.0f reduce/s (%.1f us/reduce) "
                 "sum_mism=%ld max_mism=%ld",
                 N, COUNT, REPS, comm.ack, comm.drop_n, REPS / dt, dt / REPS * 1e6, bad_sum, bad_max);
        if (dt2 >= 0.0)
            n += snprintf(line + n, sizeof line - n,
                 " | 2D A=%dxB=%d: %.0f reduce/s (%.1f us/reduce) sum_mism=%ld speedup=%.2fx",
                 A2, B2, REPS / dt2, dt2 / REPS * 1e6, bad2, dt / dt2);
        snprintf(line + n, sizeof line - n, " | RESULT: %s\n", pass);
        fprintf(stderr, "%s", line);
        FILE *rf = fopen("tp_ar_ack_result.txt", "a"); if (rf) { fputs(line, rf); fclose(rf); }
    }
    free(buf);
    return (bad_sum == 0 && bad_max == 0 && bad2 == 0) ? 0 : 1;
}
