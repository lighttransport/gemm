#define _GNU_SOURCE
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "glm53f_sparse_12n.h"

enum { HIDDEN = 4096, TOKENS = 4 };

int main(int argc, char **argv) {
    int rank, ranks, ok, local_ok = 1;
    int layer = argc > 2 ? atoi(argv[2]) : 3;
    int warm = argc > 3 ? atoi(argv[3]) : 8;
    float *x = malloc((size_t)(warm + TOKENS) * HIDDEN * sizeof(*x));
    float *a = malloc((size_t)TOKENS * HIDDEN * sizeof(*a));
    float *b = malloc((size_t)TOKENS * HIDDEN * sizeof(*b));
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 2 || ranks != 12 || warm < 0 || !x || !a || !b)
        MPI_Abort(MPI_COMM_WORLD, 2);
    int compare_cp = getenv("GLM53F_SPARSE_COMPARE_CP") != NULL;
    if (compare_cp) setenv("GLM53F_SPARSE_CP", "0", 1);
    glm53f_sparse_context_12n *ca = glm53f_sparse_create_12n(
        argv[1], layer, warm + TOKENS);
    if (compare_cp) setenv("GLM53F_SPARSE_CP", "1", 1);
    glm53f_sparse_context_12n *cb = glm53f_sparse_create_12n(
        argv[1], layer, warm + TOKENS);
    if (!ca || !cb) MPI_Abort(MPI_COMM_WORLD, 2);
    for (int t = 0; t < warm + TOKENS; t++)
        for (int i = 0; i < HIDDEN; i++)
            x[(size_t)t * HIDDEN + i] =
                (float)(((i * 17 + t * 31 + 5) % 251) - 125) / 125.0f;
    for (int t = 0; t < warm; t++) {
        local_ok &= !glm53f_sparse_sublayer_12n(
            ca, a, x + (size_t)t * HIDDEN);
        local_ok &= !glm53f_sparse_sublayer_12n(
            cb, b, x + (size_t)t * HIDDEN);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int t = 0; t < TOKENS; t++)
        local_ok &= !glm53f_sparse_sublayer_12n(
            ca, a + (size_t)t * HIDDEN,
            x + (size_t)(warm + t) * HIDDEN);
    double seq = MPI_Wtime() - t0;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    local_ok &= !glm53f_sparse_sublayer_batch_12n(
        cb, b, x + (size_t)warm * HIDDEN, TOKENS);
    double bat = MPI_Wtime() - t0;
    double d2 = 0.0, r2 = 0.0;
    for (int i = 0; i < TOKENS * HIDDEN; i++) {
        double d = (double)a[i] - b[i];
        d2 += d * d;
        r2 += (double)a[i] * a[i];
    }
    double rel = sqrt(d2 / (r2 + 1e-30));
    local_ok &= rel < 3e-6;
    double rollback_d2=0.0,rollback_r2=0.0;
    if(warm>=1){local_ok&=!glm53f_sparse_restore_length_12n(ca,warm+1);
        local_ok&=!glm53f_sparse_restore_length_12n(cb,warm+1);
        for(int t=1;t<TOKENS;t++){local_ok&=!glm53f_sparse_sublayer_12n(ca,a,x+(size_t)(warm+t)*HIDDEN);local_ok&=!glm53f_sparse_sublayer_12n(cb,b,x+(size_t)(warm+t)*HIDDEN);for(int i=0;i<HIDDEN;i++){double d=(double)a[i]-b[i];rollback_d2+=d*d;rollback_r2+=(double)a[i]*a[i];}}}
    double rollback_rel=sqrt(rollback_d2/(rollback_r2+1e-30));
    local_ok &= rollback_rel < 3e-6;
    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    double sm, bm;
    MPI_Reduce(&seq, &sm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bat, &bm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank)
        printf("GLM53F_SPARSE_BATCH mode=%s layer=%d warm=%d tokens=%d rel_l2=%.9g rollback_rel_l2=%.9g "
               "seq_ms=%.3f batch_ms=%.3f speedup=%.3f %s\n",
               compare_cp ? "replicated-vs-cp" : "replicated",
               layer, warm, TOKENS, rel,rollback_rel,sm * 1e3, bm * 1e3, sm / bm,
               ok ? "PASS" : "FAIL");
    if (!rank) {
        const char *report = getenv("GLM53F_SPARSE_REPORT");
        if (report && *report) {
            FILE *rf = fopen(report, "w");
            if (rf) {
                fprintf(rf, "GLM53F_SPARSE_BATCH mode=%s layer=%d warm=%d tokens=%d rel_l2=%.9g rollback_rel_l2=%.9g seq_ms=%.3f batch_ms=%.3f speedup=%.3f %s\n",
                        compare_cp ? "replicated-vs-cp" : "replicated", layer,
                        warm, TOKENS, rel, rollback_rel, sm * 1e3, bm * 1e3,
                        sm / bm, ok ? "PASS" : "FAIL");
                fclose(rf);
            }
        }
    }
    glm53f_sparse_free_12n(cb);
    glm53f_sparse_free_12n(ca);
    free(b); free(a); free(x);
    MPI_Finalize();
    return ok ? 0 : 1;
}
