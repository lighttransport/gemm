/*
 * Native uTofu all-reduce benchmark for Qwen3.8-27B decode.
 *
 * Qwen3.8-27B has a 5120-float residual stream and executes 129 reductions per
 * generated token.  Weight storage (Q8_0 or BF16) does not change this FP32
 * activation collective, so each result reports the communication projection
 * for both model formats.  TP2 is measured as two concurrent pairs in the
 * four-rank job; TP4 is measured as one four-rank group.
 *
 * This executable contains no MPI calls.  MPI is needed only once to create
 * tofu_topo.txt and to place one copy of this native binary on each node:
 *
 *   make tofu_topo_helper qwen38_allreduce_bench
 *   mpiexec -np 4 ./tofu_topo_helper
 *   mpiexec -np 4 ./qwen38_allreduce_bench
 *
 * Environment: Q38_AR_ITERS (2000), Q38_AR_WARMUP (200), Q38_AR_COUNT (5120).
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <utofu.h>

#include "tofu_demo.h"
#include "tp_allreduce.h"

#define JOB_NODES 4
#define MODEL_REDUCES 129
#define BARRIER_STAG DEMO_STAG
#define WAIT_SEC 30.0

static int world_rank;
static int world_size;
static int group_first;
static int group_size;
static char *bar_region;
static size_t bar_slot;
static utofu_vcq_hdl_t vcq;
static utofu_stadd_t bar_base;
static utofu_vcq_id_t peer_vcq[JOB_NODES];
static utofu_stadd_t peer_bar_base[JOB_NODES];
static uint64_t barrier_seq = 1;

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static long env_long(const char *name, long fallback)
{
    const char *s = getenv(name);
    return s && *s ? strtol(s, NULL, 0) : fallback;
}

static void fatal(const char *what, int rc)
{
    fprintf(stderr, "qwen38_allreduce rank=%d FATAL %s rc=%d\n",
            world_rank, what, rc);
    exit(1);
}

static int read_topology(uint8_t topo[][TOFU_NCOORDS])
{
    FILE *f = fopen(TOPO_PATH, "r");
    if (!f) {
        perror("open " TOPO_PATH);
        fprintf(stderr, "run tofu_topo_helper for the same four nodes first\n");
        exit(1);
    }
    int n = 0;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        unsigned rank, c[TOFU_NCOORDS];
        if (line[0] == '#' || line[0] == '\n') continue;
        if (n == JOB_NODES) fatal("topology has more than four ranks", -1);
        if (sscanf(line, "%u %u %u %u %u %u %u", &rank, &c[0], &c[1],
                   &c[2], &c[3], &c[4], &c[5]) != 7 || rank != (unsigned)n)
            fatal("malformed or unordered topology", -1);
        for (int k = 0; k < TOFU_NCOORDS; k++) topo[n][k] = (uint8_t)c[k];
        n++;
    }
    fclose(f);
    return n;
}

static size_t send_off(void) { return 0; }
static size_t arrive_off(int rank) { return bar_slot * (size_t)(1 + rank); }
static size_t release_off(void) { return bar_slot * (size_t)(1 + world_size); }

static void put_wait(int peer, utofu_stadd_t src, utofu_stadd_t dst, size_t bytes)
{
    void *cb;
    int rc;
    do {
        rc = utofu_put(vcq, peer_vcq[peer], src, dst, bytes, 0,
                       UTOFU_ONESIDED_FLAG_TCQ_NOTICE, NULL);
        if (rc == UTOFU_ERR_BUSY) utofu_poll_tcq(vcq, 0, &cb);
    } while (rc == UTOFU_ERR_BUSY);
    if (rc != UTOFU_SUCCESS) fatal("utofu_put", rc);
    do { rc = utofu_poll_tcq(vcq, 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND);
    if (rc != UTOFU_SUCCESS) fatal("utofu_poll_tcq", rc);
    struct utofu_mrq_notice notice;
    while (utofu_poll_mrq(vcq, 0, &notice) == UTOFU_SUCCESS) {}
}

static void wait_value(volatile uint64_t *p, uint64_t want)
{
    double start = now_sec();
    while (*p < want) {
        tp_ar_flag_inval(p);
        if (now_sec() - start > WAIT_SEC) fatal("barrier timeout", -1);
    }
}

/* Active-group barrier.  TP2 uses roots 0 and 2 concurrently. */
static void group_barrier(void)
{
    int root = group_first;
    uint64_t token = ++barrier_seq;
    if (world_rank == root) {
        for (int r = root + 1; r < root + group_size; r++)
            wait_value((volatile uint64_t *)(bar_region + arrive_off(r)), token);
        for (int r = root + 1; r < root + group_size; r++) {
            *(uint64_t *)(bar_region + send_off()) = token;
            put_wait(r, bar_base + send_off(), peer_bar_base[r] + release_off(), 8);
        }
    } else {
        *(uint64_t *)(bar_region + send_off()) = token;
        put_wait(root, bar_base + send_off(), peer_bar_base[root] + arrive_off(world_rank), 8);
        wait_value((volatile uint64_t *)(bar_region + release_off()), token);
    }
}

static void run_case(int tp, long count, long warmup, long iters)
{
    group_size = tp;
    group_first = (world_rank / tp) * tp;
    int local_rank = world_rank - group_first;
    utofu_vcq_id_t group_peers[JOB_NODES];
    for (int r = 0; r < tp; r++) group_peers[r] = peer_vcq[group_first + r];

    tp_comm comm;
    if (tp_comm_init(&comm, vcq, group_peers, local_rank, tp, (int)count,
                     group_barrier) != 0)
        fatal("tp_comm_init", -1);

    float *buf = NULL;
    if (posix_memalign((void **)&buf, DEMO_CACHE_LINE,
                       (size_t)count * sizeof(*buf)) != 0)
        fatal("allocate payload", -1);

    float input = (float)(local_rank + 1);
    float expected = (float)(tp * (tp + 1) / 2);
    for (long i = 0; i < count; i++) buf[i] = input;
    tp_allreduce_sum(&comm, buf, (int)count);
    for (long i = 0; i < count; i++)
        if (buf[i] != expected) fatal("all-reduce correctness", -1);

    for (long k = 0; k < warmup; k++) {
        for (long i = 0; i < count; i++) buf[i] = input;
        tp_allreduce_sum(&comm, buf, (int)count);
    }
    group_barrier();
    double start = now_sec();
    for (long k = 0; k < iters; k++) {
        for (long i = 0; i < count; i++) buf[i] = input;
        tp_allreduce_sum(&comm, buf, (int)count);
    }
    double total_usec = (now_sec() - start) * 1e6 / (double)iters;
    group_barrier();

    start = now_sec();
    for (long k = 0; k < iters; k++)
        for (long i = 0; i < count; i++) buf[i] = input;
    double reset_usec = (now_sec() - start) * 1e6 / (double)iters;
    double usec = total_usec - reset_usec;
    group_barrier();

    /* One line per group catches asymmetric pair placement in the TP2 case. */
    if (local_rank == 0) {
        double token_ms = usec * MODEL_REDUCES / 1000.0;
        double gib_s = ((double)count * sizeof(float)) / (usec * 1e-6) /
                       (1024.0 * 1024.0 * 1024.0);
        printf("qwen38_ar tp=%d group=%d-%d ranks=%d payload=fp32 "
               "count=%ld bytes=%ld reduce_us=%.3f loop_us=%.3f effective_GiB/s=%.3f "
               "reduces_per_token=%d comm_ms_per_token=%.3f models=Q8_0,BF16\n",
               tp, group_first, group_first + tp - 1, tp, count,
               count * (long)sizeof(float), usec, total_usec, gib_s,
               MODEL_REDUCES, token_ms);
        fflush(stdout);
    }
    free(buf);
    tp_comm_free(&comm);
}

int main(void)
{
    world_rank = -1;
    long count = env_long("Q38_AR_COUNT", 5120);
    long warmup = env_long("Q38_AR_WARMUP", 200);
    long iters = env_long("Q38_AR_ITERS", 2000);
    if (count < 1 || count > 1048576 || warmup < 0 || iters < 1)
        fatal("invalid benchmark environment", -1);

    utofu_tni_id_t *tnis = NULL;
    size_t ntni = 0;
    int rc = utofu_get_onesided_tnis(&tnis, &ntni);
    if (rc != UTOFU_SUCCESS || ntni == 0) fatal("get onesided TNI", rc);
    uint8_t mine[TOFU_NCOORDS];
    if ((rc = utofu_query_my_coords(mine)) != UTOFU_SUCCESS)
        fatal("query coordinates", rc);
    uint8_t topo[JOB_NODES][TOFU_NCOORDS];
    world_size = read_topology(topo);
    if (world_size != JOB_NODES) fatal("benchmark requires exactly four ranks", world_size);
    for (int r = 0; r < world_size; r++)
        if (!memcmp(mine, topo[r], TOFU_NCOORDS)) world_rank = r;
    if (world_rank < 0) fatal("node absent from topology", -1);

    utofu_tni_id_t tni = tnis[DEMO_TNI_INDEX];
    free(tnis);
    if ((rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &vcq)) != UTOFU_SUCCESS)
        fatal("create VCQ", rc);
    utofu_vcq_id_t my_vcq;
    if ((rc = utofu_query_vcq_id(vcq, &my_vcq)) != UTOFU_SUCCESS)
        fatal("query VCQ", rc);

    bar_slot = DEMO_CACHE_LINE;
    size_t bytes = bar_slot * (size_t)(world_size + 2);
    if (posix_memalign((void **)&bar_region, DEMO_CACHE_LINE, bytes) != 0)
        fatal("allocate barrier", -1);
    memset(bar_region, 0, bytes);
    if ((rc = utofu_reg_mem_with_stag(vcq, bar_region, bytes, BARRIER_STAG, 0,
                                      &bar_base)) != UTOFU_SUCCESS)
        fatal("register barrier", rc);
    for (int r = 0; r < world_size; r++) {
        if (r == world_rank) {
            peer_vcq[r] = my_vcq;
            peer_bar_base[r] = bar_base;
            continue;
        }
        if ((rc = utofu_construct_vcq_id(topo[r], tni, DEMO_CQ_ID, DEMO_CMP_ID,
                                         &peer_vcq[r])) != UTOFU_SUCCESS)
            fatal("construct peer VCQ", rc);
        utofu_set_vcq_id_path(&peer_vcq[r], NULL);
        if ((rc = utofu_query_stadd(peer_vcq[r], BARRIER_STAG,
                                    &peer_bar_base[r])) != UTOFU_SUCCESS)
            fatal("query peer barrier", rc);
    }

    /* TP2 pairs synchronize independently; TP4 then reunites all four ranks. */
    run_case(2, count, warmup, iters);
    group_first = 0;
    group_size = world_size;
    group_barrier();
    run_case(4, count, warmup, iters);

    utofu_dereg_mem(vcq, bar_base, 0);
    utofu_free_vcq(vcq);
    free(bar_region);
    return 0;
}
