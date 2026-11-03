#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "glm53f_collective_12n.h"
#include "glm53f_target_model_12n.h"

enum { TOKENS = 5 };

int main(int argc, char **argv) {
    int rank, ranks, ok, local_ok = 1;
    const int input[TOKENS] = {1, 17, 42, 314, 2718};
    int seq[TOKENS], bat[TOKENS], probe_seq, probe_bat;
    float seq_logit[TOKENS], bat_logit[TOKENS], probe_seq_logit, probe_bat_logit;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 4 || ranks != 12) MPI_Abort(MPI_COMM_WORLD, 2);
    const char *topology = getenv("TOFU_TOPO_PATH");
    if (getenv("GLM53F_UTOFU") &&
            glm53f_collective_init_12n(topology, TOKENS * 4096))
        MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_target_model_12n *m = glm53f_target_model_create_12n(
        argv[1], argv[2], argv[3], 20000);
    if(argc>4){
        glm53f_decode_options o=glm53f_decode_options_default();
        o.expert_top_k=4;o.sparse_top_k=512;o.index_refresh=4;o.kda_int8=1;
        if(glm53f_target_set_decode_options_12n(m,&o))MPI_Abort(MPI_COMM_WORLD,2);
    }
    glm53f_target_snapshot_12n *initial = glm53f_target_snapshot_create_12n(m);
    glm53f_target_snapshot_12n *seq_final = glm53f_target_snapshot_create_12n(m);
    glm53f_target_snapshot_12n *after[TOKENS];
    for (int t = 0; t < TOKENS; t++) after[t] = glm53f_target_snapshot_create_12n(m);
    if (!m || !initial || !seq_final) MPI_Abort(MPI_COMM_WORLD, 2);
    int warmup = getenv("GLM53F_BATCH_WARMUP") ? atoi(getenv("GLM53F_BATCH_WARMUP")) : 0;
    if (warmup < 0 || warmup > 20000) MPI_Abort(MPI_COMM_WORLD, 2);
    int *warm_ids = NULL;
    const char *warm_path = getenv("GLM53F_BATCH_WARMUP_FILE");
    if (warm_path && *warm_path) {
        if (!rank) {
            FILE *wf = fopen(warm_path, "r");
            if (!wf) MPI_Abort(MPI_COMM_WORLD, 2);
            warm_ids = malloc((size_t)20000 * sizeof(*warm_ids));
            while (warmup < 20000 && fscanf(wf, "%d", &warm_ids[warmup]) == 1)
                warmup++;
            fclose(wf);
        }
        MPI_Bcast(&warmup, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank) warm_ids = malloc((size_t)warmup * sizeof(*warm_ids));
        if (!warm_ids) MPI_Abort(MPI_COMM_WORLD, 2);
        MPI_Bcast(warm_ids, warmup, MPI_INT, 0, MPI_COMM_WORLD);
    }
    for (int i = 0; i < warmup; ++i) {
        int ignored; float ignored_logit;
        int warm_token = warm_ids ? warm_ids[i] : input[i % TOKENS];
        if (glm53f_target_model_step_12n(m, warm_token,
                                         &ignored, &ignored_logit, NULL))
            MPI_Abort(MPI_COMM_WORLD, 2);
    }
    free(warm_ids);
    for (int t = 0; t < TOKENS; t++)
        if (!after[t]) MPI_Abort(MPI_COMM_WORLD, 2);
    local_ok &= !glm53f_target_snapshot_save_12n(m, initial);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int t = 0; t < TOKENS; t++)
        local_ok &= !glm53f_target_model_step_12n(
            m, input[t], &seq[t], &seq_logit[t], NULL);
    double seq_sec = MPI_Wtime() - t0;
    local_ok &= !glm53f_target_snapshot_save_12n(m, seq_final);
    local_ok &= !glm53f_target_snapshot_restore_12n(m, initial);
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    local_ok &= !glm53f_target_model_step_batch_12n(
        m, input, TOKENS, bat, bat_logit, NULL, after);
    double bat_sec = MPI_Wtime() - t0;
    for (int t = 0; t < TOKENS; t++) {
        local_ok &= seq[t] == bat[t];
        local_ok &= fabsf(seq_logit[t] - bat_logit[t]) <=
                    2e-5f * fmaxf(1.0f, fabsf(seq_logit[t]));
    }
    local_ok &= !glm53f_target_snapshot_restore_12n(m, seq_final);
    local_ok &= !glm53f_target_model_step_12n(
        m, 1618, &probe_seq, &probe_seq_logit, NULL);
    local_ok &= !glm53f_target_snapshot_restore_12n(m, after[TOKENS - 1]);
    local_ok &= !glm53f_target_model_step_12n(
        m, 1618, &probe_bat, &probe_bat_logit, NULL);
    local_ok &= probe_seq == probe_bat;
    local_ok &= fabsf(probe_seq_logit - probe_bat_logit) <=
                2e-5f * fmaxf(1.0f, fabsf(probe_seq_logit));
    if (getenv("GLM53F_SNAPSHOT_PROBE")) {
        for (int t = 0; t + 1 < TOKENS; ++t) {
            int probe;
            float probe_logit;
            local_ok &= !glm53f_target_snapshot_restore_12n(m, after[t]);
            local_ok &= !glm53f_target_model_step_12n(
                m, input[t + 1], &probe, &probe_logit, NULL);
            local_ok &= probe == seq[t + 1];
            local_ok &= fabsf(probe_logit - seq_logit[t + 1]) <=
                        2e-5f * fmaxf(1.0f, fabsf(seq_logit[t + 1]));
            if (!rank)
                printf("GLM53F_SNAPSHOT_PROBE t=%d got=%d/%d logit=%.9g/%.9g %s\n",
                       t, probe, seq[t + 1], probe_logit, seq_logit[t + 1],
                       (probe == seq[t + 1] &&
                        fabsf(probe_logit - seq_logit[t + 1]) <=
                            2e-5f * fmaxf(1.0f, fabsf(seq_logit[t + 1]))) ?
                       "PASS" : "FAIL");
        }
    }
    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    double seq_max, bat_max;
    MPI_Reduce(&seq_sec, &seq_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bat_sec, &bat_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("GLM53F_TARGET_BATCH tokens=%d seq_ms=%.3f batch_ms=%.3f "
               "speedup=%.3f probe=%d/%d %s\n", TOKENS, seq_max * 1e3,
               bat_max * 1e3, seq_max / bat_max, probe_seq, probe_bat,
               ok ? "PASS" : "FAIL");
        for (int t = 0; t < TOKENS; t++)
            printf(" token[%d]=%d/%d logit=%.9g/%.9g\n", t, seq[t], bat[t],
                   seq_logit[t], bat_logit[t]);
        const char *report = getenv("GLM53F_TARGET_BATCH_REPORT");
        if (report && *report) {
            FILE *rf = fopen(report, "w");
            if (rf) {
                fprintf(rf, "GLM53F_TARGET_BATCH tokens=%d seq_ms=%.3f batch_ms=%.3f speedup=%.3f probe=%d/%d %s\n",
                        TOKENS, seq_max * 1e3, bat_max * 1e3,
                        seq_max / bat_max, probe_seq, probe_bat,
                        ok ? "PASS" : "FAIL");
                fclose(rf);
            }
        }
    }
    for (int t = 0; t < TOKENS; t++) glm53f_target_snapshot_free_12n(after[t]);
    glm53f_target_snapshot_free_12n(seq_final);
    glm53f_target_snapshot_free_12n(initial);
    glm53f_target_profile_report_12n(m, "batch_check");
    glm53f_target_model_free_12n(m);
    glm53f_collective_free_12n();
    MPI_Finalize();
    return ok ? 0 : 1;
}
