#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "glm53f_collective_12n.h"
#include "glm53f_target_model_12n.h"

int main(int argc, char **argv) {
    int rank, ranks;
    int positions = argc > 4 ? atoi(argv[4]) : 512;
    int chunk = argc > 5 ? atoi(argv[5]) : 5;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 4 || ranks != 12 || positions < 1 || chunk < 1 || chunk > 32)
        MPI_Abort(MPI_COMM_WORLD, 2);
    /* The outer prompt tile is consumed by four-position arithmetic panels;
     * no collective carries the complete tile.  Register only the largest
     * actual payload, avoiding a needlessly large uTofu region at chunk 32. */
    int collective_tokens = chunk < 4 ? chunk : 4;
    if (getenv("GLM53F_UTOFU") &&
        glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"),
                                  collective_tokens * 4096))
        MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_target_model_12n *m = glm53f_target_model_create_12n(
        argv[1], argv[2], argv[3], positions + 1);
    int *input = malloc((size_t)chunk * sizeof(*input));
    if (!m || !input) MPI_Abort(MPI_COMM_WORLD, 2);
    if (getenv("GLM53F_PREFILL_INT8") &&
        atoi(getenv("GLM53F_PREFILL_INT8")) &&
        glm53f_target_model_convert_int8_12n(m))
        MPI_Abort(MPI_COMM_WORLD, 2);
    MPI_Barrier(MPI_COMM_WORLD);
    double begin = MPI_Wtime();
    for (int base = 0; base < positions; base += chunk) {
        int n = positions - base;
        if (n > chunk) n = chunk;
        for (int t = 0; t < n; ++t)
            input[t] = 1 + ((base + t) * 104729) % 154000;
        if (glm53f_target_model_step_batch_12n(
                m, input, n, NULL, NULL, NULL, NULL))
            MPI_Abort(MPI_COMM_WORLD, 3);
    }
    double elapsed = MPI_Wtime() - begin, maximum;
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank)
        printf("GLM53F_PREFILL_12N positions=%d chunk=%d seconds=%.6f tok_s=%.3f\n",
               positions, chunk, maximum, positions / maximum);
    glm53f_target_profile_report_12n(m, "prefill");
    /* Untimed state probe: equal token/logit and hidden checksums across chunk
     * sizes validate the recurrent and sparse-cache boundary reached by the
     * prompt-only scheduler without retaining every intermediate hidden. */
    int probe_token = 0;
    float probe_logit = 0.0f;
    float *probe_hidden = malloc(4096 * sizeof(*probe_hidden));
    if (!probe_hidden || glm53f_target_model_step_12n(
            m, 31415, &probe_token, &probe_logit, probe_hidden))
        MPI_Abort(MPI_COMM_WORLD, 4);
    if (!rank) {
        double sum = 0.0, sumsq = 0.0;
        for (int i = 0; i < 4096; ++i) {
            sum += probe_hidden[i];
            sumsq += (double)probe_hidden[i] * probe_hidden[i];
        }
        printf("GLM53F_PREFILL_PROBE token=%d logit=%.9g hidden_sum=%.17g "
               "hidden_rms=%.17g\n", probe_token, probe_logit, sum,
               sqrt(sumsq / 4096.0));
    }
    free(probe_hidden);
    free(input);
    glm53f_target_model_free_12n(m);
    glm53f_collective_free_12n();
    MPI_Finalize();
    return 0;
}
