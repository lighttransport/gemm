/* tp_ar_8rank.c — N-rank tp_allreduce.h validation under qlair (--ranks).
 *
 * Runs the REAL production all-reduce (a64fx/utofu-tests/tp_allreduce.h,
 * robust path on) with each rank as a pthread: rank = qlair cmg_id =
 * core_id / cores_per_rank (main thread = core 0 = rank 0). Verifies:
 *   1. tp_allreduce_sum   — exact expected sums, bitwise-identical across ranks
 *   2. tp_allreduce_max   — exact
 *   3. tp_allreduce_argmax— exact, tie-break to lower index
 * and prints the simulated ns per all-reduce (decode-sized payload H=6144).
 *
 * Build (x86 host, cross gcc):
 *   aarch64-linux-gnu-gcc -O2 -static -march=armv8.2-a+sve -fno-math-errno \
 *     -I ../../utofu-tests -I . tp_ar_8rank.c utofu_stubs.c -o tp_ar_8rank.elf -lpthread
 * Run:
 *   RANKS=8 qlair --cores 8 --ranks 8 -n 2G tp_ar_8rank.elf
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "utofu.h"
#include "tp_allreduce.h"

#define MAXR 16
#define CNT  6144            /* decode AR payload: [1, hidden] floats */
#define REPS 4

static int R = 8;
static utofu_vcq_id_t   g_vcq_id[MAXR];
static pthread_barrier_t g_bar;
static void bar(void) { pthread_barrier_wait(&g_bar); }

static tp_comm g_comm[MAXR];
static float   g_buf[MAXR][CNT];
static uint64_t g_ar_ns[MAXR];
static volatile int g_fail = 0;

static void *rank_main(void *arg) {
    int rank = (int)(intptr_t)arg;

    utofu_tni_id_t *tnis; size_t ntni;
    if (utofu_get_onesided_tnis(&tnis, &ntni) != UTOFU_SUCCESS || ntni == 0) {
        printf("rank %d: no TNIs (not under qlair?)\n", rank); g_fail = 1; return 0;
    }
    utofu_vcq_hdl_t vcq;
    if (utofu_create_vcq(tnis[rank % ntni], 0, &vcq) != UTOFU_SUCCESS) {
        printf("rank %d: create_vcq failed\n", rank); g_fail = 1; return 0;
    }
    utofu_query_vcq_id(vcq, &g_vcq_id[rank]);
    bar();                                    /* all vcq ids published */

    tp_comm *c = &g_comm[rank];
    if (tp_comm_init(c, vcq, g_vcq_id, rank, R, CNT, bar) != 0) {
        printf("rank %d: tp_comm_init failed\n", rank); g_fail = 1; return 0;
    }

    /* --- 1. sum --- */
    float *buf = g_buf[rank];
    uint64_t t0 = qlair_rd_cyc();
    for (int rep = 0; rep < REPS; rep++) {
        for (int i = 0; i < CNT; i++) buf[i] = (float)((rank + 1) * ((i % 97) + rep + 1));
        tp_allreduce_sum(c, buf, CNT);
        long coef = (long)R * (R + 1) / 2;    /* sum_r (r+1) */
        for (int i = 0; i < CNT; i++) {
            float want = (float)(coef * ((i % 97) + rep + 1));
            if (buf[i] != want) {
                printf("rank %d SUM MISMATCH rep %d i %d got %ld want %ld (x1000)\n",
                       rank, rep, i, (long)(buf[i] * 1000), (long)(want * 1000));
                g_fail = 1; break;
            }
        }
        bar();                                /* all ranks done writing g_buf */
        if (memcmp(g_buf[rank], g_buf[0], sizeof g_buf[0]) != 0) {
            printf("rank %d SUM NOT BITWISE-IDENTICAL to rank 0 (rep %d)\n", rank, rep);
            g_fail = 1;
        }
        bar();                                /* rank 0's buffer read by all */
    }
    g_ar_ns[rank] = (qlair_rd_cyc() - t0) / REPS;

    /* --- 2. max --- */
    for (int i = 0; i < CNT; i++) buf[i] = (float)(((rank * 31 + i) % 113) - 56);
    tp_allreduce_max(c, buf, CNT);
    for (int i = 0; i < CNT; i++) {
        float want = -1e30f;
        for (int r = 0; r < R; r++) {
            float v = (float)(((r * 31 + i) % 113) - 56);
            if (v > want) want = v;
        }
        if (buf[i] != want) {
            printf("rank %d MAX MISMATCH i %d got %ld want %ld\n",
                   rank, i, (long)buf[i], (long)want);
            g_fail = 1; break;
        }
    }

    /* --- 3. argmax (with a deliberate tie: value rank%4, idx rank) --- */
    float v = (float)(rank % 4);
    int32_t idx = rank;
    tp_allreduce_argmax(c, &v, &idx);
    /* winner: max of rank%4 over ranks, tie-break to the LOWEST rank index */
    {
        float wv = 0; int32_t wi = 0;
        for (int r = 0; r < R; r++) { float rv = (float)(r % 4); if (rv > wv) { wv = rv; wi = r; } }
        if (v != wv || idx != wi) {
            printf("rank %d ARGMAX MISMATCH got v=%ld idx=%ld want v=%ld idx=%ld\n",
                   rank, (long)v, (long)idx, (long)wv, (long)wi);
            g_fail = 1;
        }
    }

    bar();
    return 0;
}

int main(void) {
    const char *e = getenv("RANKS");
    if (e) R = atoi(e);
    if (R < 2 || R > MAXR) R = 8;
    printf("tp_ar_8rank: R=%d CNT=%d REPS=%d robust=%s\n", R, CNT, REPS,
           getenv("TP_AR_ROBUST") ? getenv("TP_AR_ROBUST") : "1(default)");

    pthread_barrier_init(&g_bar, NULL, (unsigned)R);
    pthread_t th[MAXR];
    for (int r = 1; r < R; r++)
        pthread_create(&th[r], NULL, rank_main, (void *)(intptr_t)r);
    rank_main((void *)(intptr_t)0);          /* main thread = core 0 = rank 0 */
    for (int r = 1; r < R; r++) pthread_join(th[r], NULL);

    if (g_fail) { printf("FAIL\n"); return 1; }
    printf("per-allreduce simulated time (%d ranks, %d floats):\n", R, CNT);
    for (int r = 0; r < R; r++)
        printf("  rank %d: %llu ns/AR\n", r, (unsigned long long)g_ar_ns[r]);
    printf("PASS\n");
    return 0;
}
