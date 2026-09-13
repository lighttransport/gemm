/* Fixed-input full-target quantization probe. Load once, restore zero recurrent
 * state between formats, and compare teacher-forced hidden states/argmaxes.
 * This is an error probe, not a perplexity or long-context quality evaluation. */
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "glm53f_target_model_12n.h"
#include "glm53f_collective_12n.h"

int main(int argc, char **argv) {
    int rank, ranks, count = 0, input[2048], limit = argc > 5 ? atoi(argv[5]) : 256;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 5 || ranks != 12 || limit < 1 || limit > 2048) MPI_Abort(MPI_COMM_WORLD, 2);
    FILE *f = fopen(argv[4], "r");
    if (!f) MPI_Abort(MPI_COMM_WORLD, 2);
    while (count < limit && fscanf(f, "%d", input + count) == 1) {
        if (input[count] < 0 || input[count] >= 154880) MPI_Abort(MPI_COMM_WORLD, 2);
        ++count;
    }
    fclose(f);
    if (!count) MPI_Abort(MPI_COMM_WORLD, 2);
    if (getenv("GLM53F_UTOFU") && glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"), 5 * 4096))
        MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_target_model_12n *m = glm53f_target_model_create_12n(argv[1], argv[2], argv[3], count);
    glm53f_target_snapshot_12n *initial = m ? glm53f_target_snapshot_create_12n(m) : NULL;
    float *reference = malloc((size_t)count * 4096 * sizeof(float)), hidden[4096];
    int *reference_ids = malloc((size_t)count * sizeof(int));
    float *reference_logits = malloc((size_t)count * sizeof(float));
    if (!m || !initial || !reference || !reference_ids || !reference_logits ||
        glm53f_target_snapshot_save_12n(m, initial)) MPI_Abort(MPI_COMM_WORLD, 2);
    const char *names[] = {"fp8", "int8-moe", "int8-moe-kda"};
    for (int mode = 0; mode < 3; ++mode) {
        if (glm53f_target_snapshot_restore_12n(m, initial)) MPI_Abort(MPI_COMM_WORLD, 2);
        if (mode == 1 && glm53f_target_model_convert_int8_12n(m)) MPI_Abort(MPI_COMM_WORLD, 3);
        if (mode == 2 && glm53f_target_model_convert_kda_int8_12n(m)) MPI_Abort(MPI_COMM_WORLD, 3);
        glm53f_target_profile_reset_12n(m);
        MPI_Barrier(MPI_COMM_WORLD);
        double begin = MPI_Wtime(), error = 0, norm = 0, max_abs = 0, logit_error = 0;
        int agree = 0;
        for (int t = 0; t < count; ++t) {
            int next; float logit;
            if (glm53f_target_model_step_12n(m, input[t], &next, &logit, hidden)) MPI_Abort(MPI_COMM_WORLD, 4);
            if (!isfinite(logit)) MPI_Abort(MPI_COMM_WORLD, 5);
            if (!mode) { reference_ids[t] = next; reference_logits[t] = logit; }
            agree += next == reference_ids[t];
            logit_error += fabs((double)logit - reference_logits[t]);
            for (int d = 0; d < 4096; ++d) {
                size_t at = (size_t)t * 4096 + d;
                if (!isfinite(hidden[d])) MPI_Abort(MPI_COMM_WORLD, 5);
                if (!mode) reference[at] = hidden[d];
                double delta = (double)hidden[d] - reference[at];
                error += delta * delta; norm += (double)reference[at] * reference[at];
                if (fabs(delta) > max_abs) max_abs = fabs(delta);
            }
        }
        double elapsed = MPI_Wtime() - begin, maximum;
        MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (!rank) { printf("GLM53F_QUANT_MODEL format=%s teacher_forced=%d tok_s=%.3f argmax_agree=%d/%d hidden_rel_l2=%.9g hidden_max_abs=%.9g mean_argmax_logit_delta=%.9g FINITE\n",
            names[mode], count, count / maximum, agree, count, sqrt(error / (norm + 1e-30)), max_abs, logit_error / count); fflush(stdout); }
        glm53f_target_profile_report_12n(m, names[mode]);
    }
    free(reference_logits); free(reference_ids); free(reference);
    glm53f_target_snapshot_free_12n(initial); glm53f_target_model_free_12n(m);
    glm53f_collective_free_12n(); MPI_Finalize();
    return 0;
}
