/* Full MTP replay versus cache-only replay, including rollback/pool boundaries. */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glm53f_collective_12n.h"
#include "glm53f_mtp_12n.h"

enum { H = 4096 };

static void input(int position, float *hidden) {
    for (int i = 0; i < H; i++)
        hidden[i] = (float)((i * 31 + position * 17) % 257 - 128) / 128.0f;
}

int main(int argc, char **argv) {
    int rank, ranks, ok = 1, all_ok, draft, expected;
    float hidden[H], actual_hidden[H], expected_hidden[H], logit, expected_logit;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 4 || ranks != 12) MPI_Abort(MPI_COMM_WORLD, 2);
    int warm = argc > 4 ? atoi(argv[4]) : 2051;
    if (warm < 4) MPI_Abort(MPI_COMM_WORLD, 2);
    if (getenv("GLM53F_UTOFU") && glm53f_collective_init_12n(
            getenv("TOFU_TOPO_PATH"), 5 * H)) MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_mtp_context_12n *m = glm53f_mtp_create_12n(argv[1], argv[2], argv[3], warm + 8);
    if (!m) MPI_Abort(MPI_COMM_WORLD, 2);
    double elapsed[2];
    for (int mode = 0; mode < 2; mode++) {
        if (glm53f_mtp_restore_length_12n(m, 0)) MPI_Abort(MPI_COMM_WORLD, 2);
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        for (int i = 0; i < warm; i++) {
            input(i, hidden);
            int rc = mode ? glm53f_mtp_cache_append_12n(m, 1 + i % 1000, hidden) :
                glm53f_mtp_forward_12n(m, 1 + i % 1000, hidden, &draft, &logit, NULL);
            if (rc) MPI_Abort(MPI_COMM_WORLD, 2);
        }
        elapsed[mode] = MPI_Wtime() - start;
        /* Also overwrite a rejected suffix spanning a completed pool. */
        if (glm53f_mtp_restore_length_12n(m, warm - 3)) MPI_Abort(MPI_COMM_WORLD, 2);
        for (int i = warm - 3; i < warm + 3; i++) {
            input(i + 100, hidden);
            int rc = mode ? glm53f_mtp_cache_append_12n(m, 31 + i % 1000, hidden) :
                glm53f_mtp_forward_12n(m, 31 + i % 1000, hidden, &draft, &logit, NULL);
            if (rc) MPI_Abort(MPI_COMM_WORLD, 2);
        }
        input(warm + 200, hidden);
        if (glm53f_mtp_forward_12n(m, 17, hidden, &draft, &logit, actual_hidden))
            MPI_Abort(MPI_COMM_WORLD, 2);
        if (!mode) {
            expected = draft; expected_logit = logit;
            memcpy(expected_hidden, actual_hidden, sizeof(expected_hidden));
        } else {
            ok &= draft == expected && logit == expected_logit;
            ok &= !memcmp(expected_hidden, actual_hidden, sizeof(expected_hidden));
            ok &= glm53f_mtp_length_12n(m) == warm + 4;
        }
    }
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    double maximum[2];
    MPI_Reduce(elapsed, maximum, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) printf("GLM53F_MTP_CACHE warm=%d full_ms=%.3f cache_ms=%.3f "
        "draft=%d/%d logit=%.9g/%.9g hidden_bitwise=%s %s\n", warm,
        maximum[0] * 1000 / warm, maximum[1] * 1000 / warm, draft, expected,
        logit, expected_logit, all_ok ? "exact" : "different", all_ok ? "PASS" : "FAIL");
    glm53f_mtp_free_12n(m);
    glm53f_collective_free_12n();
    MPI_Finalize();
    return all_ok ? 0 : 1;
}
