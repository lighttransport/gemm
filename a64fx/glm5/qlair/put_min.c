/* put_min.c — minimal 2-rank utofu_put diagnostic for the qlair Tofu sim.
 * rank0 fills a pattern and puts payload+trailer into rank1's slot; rank1
 * spins on the trailer and dumps the first 12 floats. Isolates the put
 * data path from the tp_allreduce protocol. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "utofu.h"

#define CNT 32
static utofu_vcq_id_t g_id[2];
static utofu_stadd_t  g_base[2];
static pthread_barrier_t g_bar;

/* each rank's region: [payload CNT floats][trailer u64] aligned */
static float          g_region[2][CNT + 4] __attribute__((aligned(256)));

static void *rank_main(void *arg) {
    int rank = (int)(intptr_t)arg;
    utofu_tni_id_t *tnis; size_t ntni;
    utofu_get_onesided_tnis(&tnis, &ntni);
    utofu_vcq_hdl_t vcq;
    utofu_create_vcq(tnis[rank % ntni], 0, &vcq);
    utofu_query_vcq_id(vcq, &g_id[rank]);
    utofu_reg_mem(vcq, g_region[rank], sizeof g_region[0], 0, &g_base[rank]);
    pthread_barrier_wait(&g_bar);

    volatile uint64_t *trl = (volatile uint64_t *)&g_region[rank][CNT];
    if (rank == 0) {
        for (int i = 0; i < CNT; i++) g_region[0][i] = (float)(100 + i);
        *(volatile uint64_t *)&g_region[0][CNT] = 7;
        int rc = utofu_put(vcq, g_id[1], g_base[0], g_base[1],
                           CNT * 4 + 8, 0, 1UL << 14, NULL);
        printf("rank0 put rc=%d\n", rc);
        void *cb; while (utofu_poll_tcq(vcq, 0, &cb) != UTOFU_SUCCESS) {}
        printf("rank0 tcq drained\n");
    } else {
        while (*trl < 7) {}
        printf("rank1 got trailer=%llu payload:", (unsigned long long)*trl);
        for (int i = 0; i < 12; i++) printf(" %d", (int)g_region[1][i]);
        printf("\n");
        int ok = 1;
        for (int i = 0; i < CNT; i++)
            if (g_region[1][i] != (float)(100 + i)) {
                printf("MISMATCH i=%d got=%d\n", i, (int)g_region[1][i]); ok = 0;
            }
        printf(ok ? "PUT PASS\n" : "PUT FAIL\n");
    }
    pthread_barrier_wait(&g_bar);
    return 0;
}

int main(void) {
    pthread_barrier_init(&g_bar, NULL, 2);
    pthread_t t;
    pthread_create(&t, NULL, rank_main, (void *)1);
    rank_main((void *)0);
    pthread_join(t, NULL);
    return 0;
}
