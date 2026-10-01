#define _GNU_SOURCE
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glm53f_sparse_12n.h"
#include "glm53f_collective_12n.h"

enum { HIDDEN = 4096 };

int main(int argc, char **argv) {
    int rank, ranks, ok, local_ok = 1;
    int layer = argc > 2 ? atoi(argv[2]) : 3;
    int warm = argc > 3 ? atoi(argv[3]) : 8;
    int tokens = argc > 4 ? atoi(argv[4]) : 4;
    int prefill = tokens > 5 || getenv("GLM53F_CHECK_SPARSE_PREFILL");
    if (tokens < 1 || tokens > GLM53F_PREFILL_ATTN_TOKENS || warm < 0 || warm > 1048576)
        return 2;
    float *x = malloc((size_t)(warm + tokens) * HIDDEN * sizeof(*x));
    float *a = malloc((size_t)tokens * HIDDEN * sizeof(*a));
    float *b = malloc((size_t)tokens * HIDDEN * sizeof(*b));
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int async_check = getenv("GLM53F_SPARSE_CHECK_ASYNC") != NULL;
    if (argc < 2 || ranks != 12 || warm < 0 || !x || !a || !b ||
        provided < MPI_THREAD_SERIALIZED || (async_check && !prefill))
        MPI_Abort(MPI_COMM_WORLD, 2);
    if (getenv("GLM53F_UTOFU") && glm53f_collective_init_12n(
            getenv("TOFU_TOPO_PATH"), GLM53F_PREFILL_ATTN_TOKENS * HIDDEN)) MPI_Abort(MPI_COMM_WORLD, 2);
    if (async_check && glm53f_collective_prefill_algorithm_12n(5)) MPI_Abort(MPI_COMM_WORLD, 2);
    int compare_cp = getenv("GLM53F_SPARSE_COMPARE_CP") != NULL;
    if (compare_cp) setenv("GLM53F_SPARSE_CP", "0", 1);
    glm53f_sparse_context_12n *ca = glm53f_sparse_create_12n(
        argv[1], layer, warm + tokens);
    if (compare_cp) setenv("GLM53F_SPARSE_CP", "1", 1);
    glm53f_sparse_context_12n *cb = glm53f_sparse_create_12n(
        argv[1], layer, warm + tokens);
    if (!ca || !cb) MPI_Abort(MPI_COMM_WORLD, 2);
    for (int t = 0; t < warm + tokens; t++)
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
    glm53f_prefill_config config = {GLM53F_PREFILL_FAST, async_check ? 32 : 16, GLM53F_PREFILL_FAST_DEFAULT, NULL, 0};
    config.features &= ~GLM53F_PREFILL_GEMM;
    if (async_check) {
        glm53f_sparse_configure_prefill_12n(ca, &config);
        glm53f_sparse_prefill_workspace_12n *w = glm53f_sparse_prefill_workspace_create_12n();
        local_ok &= w && !glm53f_sparse_prefill_12n(ca, w, a,
            x + (size_t)warm * HIDDEN, tokens);
        glm53f_sparse_prefill_workspace_free_12n(w);
    } else for (int t = 0; t < tokens; t++)
        local_ok &= !glm53f_sparse_sublayer_12n(ca, a + (size_t)t * HIDDEN,
            x + (size_t)(warm + t) * HIDDEN);
    double seq = MPI_Wtime() - t0;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    if (prefill) {
        /* No BF16 GEMM arena is needed for native projection tests. */
        config.features &= ~GLM53F_PREFILL_GEMM;
        glm53f_sparse_configure_prefill_12n(cb, &config);
        glm53f_sparse_prefill_workspace_12n *w = glm53f_sparse_prefill_workspace_create_12n();
        float *partial = async_check ? malloc((size_t)tokens * HIDDEN * sizeof(float)) : NULL;
        if (async_check && (!partial || glm53f_collective_prefill_algorithm_12n(5) ||
                glm53f_async_begin_12n(partial, b, tokens, HIDDEN, 32))) MPI_Abort(MPI_COMM_WORLD, 2);
        if (async_check && !glm53f_async_owner_12n()) MPI_Abort(MPI_COMM_WORLD, 2);
        if (async_check) glm53f_sparse_set_defer_reduce_12n(1);
        local_ok &= w && !glm53f_sparse_prefill_12n(cb, w, async_check ? partial : b,
            x + (size_t)warm * HIDDEN, tokens);
        if (async_check) {
            glm53f_sparse_set_defer_reduce_12n(0);
            glm53f_async_ready_12n(tokens);
            local_ok &= !glm53f_async_finish_12n();
        }
        free(partial);
        glm53f_sparse_prefill_workspace_free_12n(w);
    } else local_ok &= !glm53f_sparse_sublayer_batch_12n(
        cb, b, x + (size_t)warm * HIDDEN, tokens);
    double bat = MPI_Wtime() - t0;
    double d2 = 0.0, r2 = 0.0;
    for (int i = 0; i < tokens * HIDDEN; i++) {
        double d = (double)a[i] - b[i];
        d2 += d * d;
        r2 += (double)a[i] * a[i];
    }
    double rel = sqrt(d2 / (r2 + 1e-30));
    local_ok &= rel < (getenv("GLM53F_SPARSE_CHECK_TOL") ? atof(getenv("GLM53F_SPARSE_CHECK_TOL")) : 3e-6);
    if (async_check) local_ok &= !memcmp(a, b, (size_t)tokens * HIDDEN * sizeof(float));
    double rollback_d2=0.0,rollback_r2=0.0;
    if(warm>=1){local_ok&=!glm53f_sparse_restore_length_12n(ca,warm+1);
        local_ok&=!glm53f_sparse_restore_length_12n(cb,warm+1);
        for(int t=1;t<tokens;t++){local_ok&=!glm53f_sparse_sublayer_12n(ca,a,x+(size_t)(warm+t)*HIDDEN);local_ok&=!glm53f_sparse_sublayer_12n(cb,b,x+(size_t)(warm+t)*HIDDEN);for(int i=0;i<HIDDEN;i++){double d=(double)a[i]-b[i];rollback_d2+=d*d;rollback_r2+=(double)a[i]*a[i];}}}
    double rollback_rel=sqrt(rollback_d2/(rollback_r2+1e-30));
    local_ok &= rollback_rel < 3e-6;
    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    double sm, bm;
    MPI_Reduce(&seq, &sm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bat, &bm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank)
        printf("GLM53F_SPARSE_BATCH mode=%s layer=%d warm=%d tokens=%d rel_l2=%.9g rollback_rel_l2=%.9g "
               "seq_ms=%.3f batch_ms=%.3f speedup=%.3f %s\n",
               async_check ? "prefill-vs-owner-async" : compare_cp ? "replicated-vs-cp" : "replicated",
               layer, warm, tokens, rel,rollback_rel,sm * 1e3, bm * 1e3, sm / bm,
               ok ? "PASS" : "FAIL");
    if (!rank) {
        const char *report = getenv("GLM53F_SPARSE_REPORT");
        if (report && *report) {
            FILE *rf = fopen(report, "w");
            if (rf) {
                fprintf(rf, "GLM53F_SPARSE_BATCH mode=%s layer=%d warm=%d tokens=%d rel_l2=%.9g rollback_rel_l2=%.9g seq_ms=%.3f batch_ms=%.3f speedup=%.3f %s\n",
                        async_check ? "prefill-vs-owner-async" : compare_cp ? "replicated-vs-cp" : "replicated", layer,
                        warm, tokens, rel, rollback_rel, sm * 1e3, bm * 1e3,
                        sm / bm, ok ? "PASS" : "FAIL");
                fclose(rf);
            }
        }
    }
    glm53f_sparse_free_12n(cb);
    glm53f_sparse_free_12n(ca);
    glm53f_collective_free_12n();
    free(b); free(a); free(x);
    MPI_Finalize();
    return ok ? 0 : 1;
}
