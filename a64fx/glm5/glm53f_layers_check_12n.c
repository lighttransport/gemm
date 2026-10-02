/* Compare full-model execution with three owned-layer calls on one resident
 * TP12 model. This gates the executor refactoring before PP constructors. */
#include <inttypes.h>
#include <limits.h>
#define GLM53F_TARGET_MODEL_NO_MAIN
#include "glm53f_target_decode_12n.c"
static long guard_memory(long previous) {
    FILE *f = fopen("/proc/meminfo", "r"); char line[256]; long local = -1, minimum;
    if (f) {
        while (fgets(line, sizeof(line), f))
            if (sscanf(line, "MemAvailable: %ld kB", &local) == 1) break;
        fclose(f);
    }
    MPI_Allreduce(&local, &minimum, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
    if (minimum < 6L * 1024 * 1024) MPI_Abort(MPI_COMM_WORLD, 3);
    return minimum < previous ? minimum : previous;
}
static int finite_bits(float value) {
    uint32_t bits; memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}
int main(int argc, char **argv) {
    enum { LIMIT = 32768 };
    int rank, ranks, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 7 || ranks != 12 || provided < MPI_THREAD_SERIALIZED) MPI_Abort(MPI_COMM_WORLD, 2);
    char *end; long batch = strtol(argv[6], &end, 10);
    if (!*argv[6] || *end || batch < 1 || batch > GLM53F_PREFILL_MAX_TOKENS) MPI_Abort(MPI_COMM_WORLD, 2);
    const char *reference_state_prefix = NULL;
    glm53f_prefill_config config = {GLM53F_PREFILL_FAST, 32, 27, NULL, 5};
    for (int i = 7; i < argc; ++i) {
        if (!strcmp(argv[i], "--reference-state-prefix") && i + 1 < argc) { reference_state_prefix = argv[++i]; continue; }
        if (glm53f_prefill_option(&config, argc, argv, &i) != 1) MPI_Abort(MPI_COMM_WORLD, 2);
    }
    int *ids = malloc(LIMIT * sizeof(int)), count = 0;
    if (!ids) MPI_Abort(MPI_COMM_WORLD, 2);
    if (!rank) {
        FILE *f = fopen(argv[4], "r"); int token;
        if (!f) MPI_Abort(MPI_COMM_WORLD, 2);
        while (fscanf(f, "%d", &token) == 1) {
            if (count == LIMIT || token < 0 || token >= 154880) MPI_Abort(MPI_COMM_WORLD, 2);
            ids[count++] = token;
        }
        if (!feof(f) || ferror(f) || fclose(f) || !count) MPI_Abort(MPI_COMM_WORLD, 2);
    }
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Bcast(ids, count, MPI_INT, 0, MPI_COMM_WORLD);
    if (glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"),
            glm53f_prefill_collective_count(count, HIDDEN)) ||
        glm53f_collective_prefill_algorithm_12n(config.collective)) MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_target_model_12n *m = glm53f_target_model_create_12n(argv[1], argv[2], argv[3], count + 1);
    glm53f_target_snapshot_12n *empty = m ? glm53f_target_snapshot_create_12n(m) : NULL;
    float *reference = a256((size_t)count * FLAT * sizeof(float));
    float *streams = a256((size_t)batch * FLAT * sizeof(float));
    if (!m || !empty || !reference || !streams || glm53f_target_model_configure_prefill_12n(m, &config) ||
        glm53f_target_snapshot_save_12n(m, empty)) MPI_Abort(MPI_COMM_WORLD, 2);
    long minimum_kb = guard_memory(LONG_MAX);
    for (int offset = 0; offset < count; offset += (int)batch) {
        int n = count - offset; if (n > batch) n = (int)batch;
        if (glm53f_target_model_step_batch_12n(m, ids + offset, n, NULL, NULL, NULL, NULL)) MPI_Abort(MPI_COMM_WORLD, 2);
        memcpy(reference + (size_t)offset * FLAT, m->batch_streams, (size_t)n * FLAT * sizeof(float));
        minimum_kb = guard_memory(minimum_kb);
    }
    int reference_token, token; float reference_logit, logit;
    if (reference_state_prefix && (glm53f_target_trace_open_12n(m, reference_state_prefix, 1) ||
        glm53f_target_trace_close_12n(m))) MPI_Abort(MPI_COMM_WORLD, 2);
    if (glm53f_target_model_readout_12n(m, &reference_token, &reference_logit) ||
        glm53f_target_trace_open_12n(m, argv[5], 0) || glm53f_target_trace_close_12n(m) ||
        glm53f_target_snapshot_restore_12n(m, empty)) MPI_Abort(MPI_COMM_WORLD, 2);
    int failed = 0; uint64_t mismatches = 0;
    for (int offset = 0; offset < count; offset += (int)batch) {
        int n = count - offset; if (n > batch) n = (int)batch;
        if (glm53f_target_model_embed_batch_12n(m, ids + offset, n, streams) ||
            glm53f_target_model_layers_batch_12n(m, streams, n, 0, 15) ||
            glm53f_target_model_layers_batch_12n(m, streams, n, 15, 30) ||
            glm53f_target_model_layers_batch_12n(m, streams, n, 30, 45)) MPI_Abort(MPI_COMM_WORLD, 2);
        for (size_t j = 0; j < (size_t)n * FLAT; ++j) {
            float *ref = reference + (size_t)offset * FLAT + j;
            mismatches += memcmp(streams + j, ref, sizeof(float)) != 0;
            failed |= !finite_bits(streams[j]) || !finite_bits(*ref);
        }
        minimum_kb = guard_memory(minimum_kb);
    }
    /* Destroy transfer-buffer contents: readout must use model-owned storage. */
    memset(streams, 0xa5, (size_t)batch * FLAT * sizeof(float));
    failed |= glm53f_target_model_readout_12n(m, &token, &logit) != 0;
    failed |= token != reference_token || memcmp(&logit, &reference_logit, sizeof(float)) != 0;
    failed |= glm53f_target_trace_open_12n(m, argv[5], 1) != 0;
    if (m->trace) failed |= glm53f_target_trace_close_12n(m) != 0;
    failed |= mismatches != 0;
    int all_failed; uint64_t maximum;
    MPI_Allreduce(&failed, &all_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&mismatches, &maximum, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    if (!rank) printf("GLM53F_LAYERS_CHECK tokens=%d batch=%ld cuts=15,30 streams_bit_mismatches=%" PRIu64
        " state_and_readout=%s min_MemAvailable_GiB=%.6f %s\n", count, batch, maximum,
        all_failed ? "FAIL" : "BIT_EXACT", minimum_kb / 1048576.0, all_failed ? "FAIL" : "PASS");
    free(streams); free(reference); free(ids); glm53f_target_snapshot_free_12n(empty);
    glm53f_target_model_free_12n(m); glm53f_collective_free_12n(); MPI_Finalize(); return all_failed;
}
