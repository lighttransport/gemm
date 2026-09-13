/* MPI handles placement/bootstrap only; all tested reductions use uTofu. */
#define _GNU_SOURCE
#include <mpi.h>
#include <utofu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "ds41f_expert.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "../utofu-tests/tp_allreduce.h"
#pragma GCC diagnostic pop

static int rank;
static void die(const char *what, int rc)
{
    fprintf(stderr, "FAIL rank=%d %s rc=%d\n", rank, what, rc);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, rc ? rc : 1);
    exit(1);
}
static void barrier(void) { MPI_Barrier(MPI_COMM_WORLD); }

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int n;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    char path[80];
    snprintf(path, sizeof path, "utofu.rank%02d.log", rank);
    if (!freopen(path, "w", stderr)) die("open log", 1);
    setvbuf(stderr, NULL, _IOLBF, 0);
    if (n != 12) die("requires 12 ranks", n);
    uint8_t coords[6], all[12][6];
    int rc = utofu_query_my_coords(coords);
    if (rc != UTOFU_SUCCESS) die("coords", rc);
    MPI_Allgather(coords, 6, MPI_BYTE, all, 6, MPI_BYTE, MPI_COMM_WORLD);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            if (!memcmp(all[i], all[j], 6)) die("multiple ranks on a node", 1);
    char host[256]; gethostname(host, sizeof host);
    fprintf(stderr, "rank=%d host=%s coords=%u,%u,%u,%u,%u,%u\n", rank,
            host, coords[0], coords[1], coords[2], coords[3], coords[4], coords[5]);
    utofu_tni_id_t *tnis;
    size_t nt;
    rc = utofu_get_onesided_tnis(&tnis, &nt);
    if (rc != UTOFU_SUCCESS || !nt) die("TNIs", rc);
    utofu_vcq_hdl_t vcq;
    rc = utofu_create_vcq_with_cmp_id(tnis[0], 0, 0, &vcq);
    free(tnis);
    if (rc != UTOFU_SUCCESS) die("VCQ", rc);
    utofu_vcq_id_t self, peers[12];
    rc = utofu_query_vcq_id(vcq, &self);
    if (rc != UTOFU_SUCCESS) die("VCQ id", rc);
    MPI_Allgather(&self, sizeof self, MPI_BYTE, peers, sizeof self,
                  MPI_BYTE, MPI_COMM_WORLD);
    for (int i = 0; i < n; ++i) {
        rc = utofu_set_vcq_id_path(&peers[i], NULL);
        if (rc != UTOFU_SUCCESS) die("VCQ path", rc);
    }
    tp_comm c;
    tp_comm_config options = {0};
    options.robust = 1; options.poll_spins = 8; options.timeout = 30;
    rc = tp_comm_init_region_ex(&c, vcq, peers, rank, n, 5120,
                                barrier, TP_AR_STAG, &options, NULL, 0);
    if (rc) die("comm init", rc);
    barrier();
    float v[5120];
    const int sizes[] = {1, 2, 19, 5120};
    for (int iter = 0; iter < 100; ++iter) {
        int count = sizes[iter % 4];
        for (int i = 0; i < count; ++i) v[i] = rank * 3 - 7 + i % 17;
        tp_allreduce_sum(&c, v, count);
        for (int i = 0; i < count; ++i)
            if (v[i] != 114 + 12 * (i % 17)) die("sum", iter + 1);
        for (int i = 0; i < count; ++i) v[i] = rank - 4 - i % 17;
        tp_allreduce_max(&c, v, count);
        for (int i = 0; i < count; ++i)
            if (v[i] != 7 - i % 17) die("max", iter + 1);
    }
    barrier();
    fprintf(stderr, "DS41F_UTOFU_TEST PASS rank=%d ranks=12 reductions=200\n", rank);
    if (argc == 3 && !strcmp(argv[1],"--staged-experts")) {
        char stage[4096];
        snprintf(stage,sizeof stage,"%s/rank%d",argv[2],rank);
        ds41f_expert expert;
        rc=ds41f_expert_load(&expert,stage,0,rank);
        if (rc) die("staged expert load",rc);
        float x[5120],expected[5120],sum[5120],scratch[3*2304+5120];
        for (int i=0;i<5120;++i) x[i]=sinf((float)i*.013f);
        /* Six active experts for a deterministic synthetic route. Rank zero
         * through five contribute; remaining ranks participate with zeros. */
        memset(v,0,sizeof v);memset(expected,0,sizeof expected);
        if (rank<6) {
            rc=ds41f_expert_forward(&expert,v,x,.25f,scratch,0);
            rc|=ds41f_expert_forward(&expert,expected,x,.25f,scratch,1);
            if (rc) die("staged expert forward",rc);
        }
        ds41f_expert_free(&expert);
        /* MPI is an independent test oracle, not the inference combine. */
        MPI_Allreduce(expected,sum,5120,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
        tp_allreduce_sum(&c,v,5120);
        float max_error=0;
        for (int i=0;i<5120;++i) {
            float err=fabsf(v[i]-sum[i]);max_error=fmaxf(max_error,err);
            if (!isfinite(v[i]) || !isfinite(sum[i]) || err>1e-3f*(1+fabsf(sum[i])))
                die("staged expert combine",i+1);
        }
        fprintf(stderr,"DS41F_UTOFU_STAGED_MOE PASS rank=%d active_experts=6 max_abs=%g\n",rank,max_error);
    }
    tp_comm_free(&c);
    utofu_free_vcq(vcq);
    MPI_Finalize();
    return 0;
}
