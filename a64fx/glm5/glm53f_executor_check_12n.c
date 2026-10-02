#define _GNU_SOURCE
/* Same-model, same-input legacy/persistent comparison with complete recurrent
 * and sparse state traces. This is a correctness run, never a speed run. */
#include "glm53f_target_model_12n.h"
#include "glm53f_collective_12n.h"
#include "glm53f_team.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

enum { H = 4096, LIMIT = 512 };
/* The native runner uses fast-math; isfinite may otherwise disappear. */
static int finite_float(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}
struct check_call {
    glm53f_target_model_12n *model;
    const int *input, *token;
    const float *hidden, *logit;
    int count, failed, bit_mismatches;
    double error, norm;
};
static void persistent_control(void *context) {
    struct check_call *a = context;
    float hidden[H];
    for (int t = 0; t < a->count; ++t) {
        int token; float logit;
        if (glm53f_target_model_step_12n(a->model, a->input[t], &token, &logit, hidden)) {
            a->failed = 1; break;
        }
        a->failed |= token != a->token[t] || !finite_float(logit) || !finite_float(a->logit[t]) ||
            fabsf(logit - a->logit[t]) > 2e-5f * fmaxf(1, fabsf(a->logit[t]));
        for (int i = 0; i < H; ++i) {
            float ref = a->hidden[(size_t)t * H + i];
            a->failed |= !finite_float(hidden[i]) || !finite_float(ref);
            a->bit_mismatches += memcmp(&ref, hidden + i, sizeof(float)) != 0;
            double d = (double)hidden[i] - ref;
            a->error += d * d; a->norm += (double)ref * ref;
        }
    }
}
int main(int argc, char **argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int count = argc > 6 ? atoi(argv[6]) : 32;
    if (argc < 6 || ranks != 12 || provided < MPI_THREAD_SERIALIZED || count < 1 || count > LIMIT ||
        !getenv("GLM53F_Q2_KDA_STAGE") || !getenv("GLM53F_Q2_SPARSE_STAGE")) MPI_Abort(MPI_COMM_WORLD, 2);
    int input[LIMIT], token[LIMIT]; float logit[LIMIT];
    if (!rank) {
        FILE *f = fopen(argv[4], "r");
        if (!f) MPI_Abort(MPI_COMM_WORLD, 2);
        for (int t = 0; t < count; ++t)
            if (fscanf(f, "%d", input + t) != 1 || input[t] < 0 || input[t] >= 154880) MPI_Abort(MPI_COMM_WORLD, 2);
        fclose(f);
    }
    MPI_Bcast(input, count, MPI_INT, 0, MPI_COMM_WORLD);
    if (glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"), 5 * H)) MPI_Abort(MPI_COMM_WORLD, 2);
    const char *index_env = getenv("GLM53F_EXECUTOR_INDEX_KERNEL");
    const char *mla_env = getenv("GLM53F_EXECUTOR_MLA_KERNEL");
    int index_kernel = index_env ? atoi(index_env) : 1;
    int mla_kernel = mla_env ? atoi(mla_env) : 1;
    if (index_kernel < 0 || index_kernel > 4 || mla_kernel < 0 || mla_kernel > 3)
        MPI_Abort(MPI_COMM_WORLD, 2);
    const char *q8_env = getenv("GLM53F_EXECUTOR_Q8_KERNEL");
    int q8_kernel = q8_env ? atoi(q8_env) : 0;
    if (q8_kernel < 0 || q8_kernel > 3) MPI_Abort(MPI_COMM_WORLD, 2);
    const char *mhc_env = getenv("GLM53F_EXECUTOR_MHC_KERNEL");
    int mhc_kernel = mhc_env ? atoi(mhc_env) : 0;
    if (mhc_kernel < 0 || mhc_kernel > 1) MPI_Abort(MPI_COMM_WORLD, 2);
    setenv("GLM53F_MHC_FUSED_SYNC", "0", 1);
    if (q8_kernel) {
        setenv("GLM53F_NATIVE_Q8_ROWS8", "0", 1);
        setenv("GLM53F_NATIVE_Q8_TILE2X8", "0", 1);
        setenv("GLM53F_MLA_FUSED_PROJECTION", "0", 1);
    }
    setenv("GLM53F_MHC_FAST", "1", 1);
    setenv("GLM53F_ROUTER_FUSE", "0", 1);
    setenv("GLM53F_INDEX_HEADS", "0", 1);
    setenv("GLM53F_MLA_REGISTERS", "0", 1);
    glm53f_target_model_12n *m = glm53f_target_model_create_12n(argv[1], argv[2], argv[3], count + 1);
    glm53f_target_snapshot_12n *initial = m ? glm53f_target_snapshot_create_12n(m) : NULL;
    float *hidden = malloc((size_t)count * H * sizeof(float));
    if (!m || !initial || !hidden || glm53f_target_snapshot_save_12n(m, initial) ||
        glm53f_target_trace_open_12n(m, argv[5], 0)) MPI_Abort(MPI_COMM_WORLD, 2);
    for (int t = 0; t < count; ++t)
        if (glm53f_target_model_step_12n(m, input[t], token + t, logit + t, hidden + (size_t)t * H)) MPI_Abort(MPI_COMM_WORLD, 2);
    if (glm53f_target_trace_close_12n(m)) MPI_Abort(MPI_COMM_WORLD, 2);
    char index_value[16], mla_value[16];
    snprintf(index_value, sizeof(index_value), "%d", index_kernel);
    snprintf(mla_value, sizeof(mla_value), "%d", mla_kernel);
    if (q8_kernel) {
        char value[2] = {(char)('0' + q8_kernel), 0};
        setenv("GLM53F_NATIVE_Q8_ROWS8", "1", 1);
        setenv("GLM53F_NATIVE_Q8_TILE2X8", value, 1);
        setenv("GLM53F_MLA_FUSED_PROJECTION", "1", 1);
    }
    setenv("GLM53F_ROUTER_FUSE", "1", 1);
    setenv("GLM53F_MHC_FUSED_SYNC", mhc_kernel ? "1" : "0", 1);
    setenv("GLM53F_INDEX_HEADS", index_value, 1);
    setenv("GLM53F_MLA_REGISTERS", mla_value, 1);
    if (mla_kernel == 3) {
        /* Cache allocation is a startup option. Free the reference before
         * loading a fresh candidate so two resident models never coexist. */
        glm53f_target_snapshot_free_12n(initial); initial = NULL;
        glm53f_target_model_free_12n(m);
        m = glm53f_target_model_create_12n(argv[1], argv[2], argv[3], count + 1);
        if (!m) MPI_Abort(MPI_COMM_WORLD, 2);
    } else if (glm53f_target_snapshot_restore_12n(m, initial)) MPI_Abort(MPI_COMM_WORLD, 2);
    if (glm53f_target_trace_open_12n(m, argv[5], 1)) MPI_Abort(MPI_COMM_WORLD, 2);
    struct check_call call = {m, input, token, hidden, logit, count, 0, 0, 0, 0};
    glm53f_team_run(persistent_control, &call);
    call.failed |= glm53f_target_trace_close_12n(m) != 0;
    double rel = sqrt(call.error / (call.norm + 1e-30)), maximum;
    call.failed |= rel > 1e-3;
    int ok = !call.failed, all;
    MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&rel, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int mismatches;
    MPI_Allreduce(&call.bit_mismatches, &mismatches, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (!rank) printf("GLM53F_EXECUTOR_CHECK tokens=%d index_kernel=%d mla_kernel=%d q8_kernel=%d mhc_kernel=%d hidden_bit_mismatches=%d hidden_rel_l2=%.9g state=%s %s\n",
        count, index_kernel, mla_kernel, q8_kernel, mhc_kernel, mismatches, maximum, all ? "BIT_EXACT" : "UNCHECKED_OR_MISMATCH", all ? "PASS" : "FAIL");
    free(hidden); glm53f_target_snapshot_free_12n(initial);
    glm53f_target_model_free_12n(m); glm53f_collective_free_12n();
    MPI_Finalize(); return all ? 0 : 1;
}
