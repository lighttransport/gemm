/* Compare complete prompt endpoints across outer chunk sizes. Including the
 * model implementation exposes its final streams only to this diagnostic. */
#define GLM53F_TARGET_MODEL_NO_MAIN
#include "glm53f_target_decode_12n.c"

static int read_prompt(const char *path, int *ids, int limit) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int count = 0, id;
    while (fscanf(f, "%d", &id) == 1) {
        if (count == limit || id < 0 || id >= 154880) {
            fclose(f);
            return -1;
        }
        ids[count++] = id;
    }
    int failed = !feof(f) || ferror(f);
    fclose(f);
    return failed || !count ? -1 : count;
}

static long available_kb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long value = -1;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) break;
    fclose(f);
    return value;
}

static int run_prompt(glm53f_target_model_12n *m, const int *ids,
        int count, int chunk, long *minimum_kb, float *reference_hidden,
        float *candidate_hidden, int *hidden_mismatches) {
    if (!m->prefill.features && chunk > GLM53F_PREFILL_V5_TOKENS)
        chunk = GLM53F_PREFILL_V5_TOKENS;
    for (int offset = 0; offset < count; offset += chunk) {
        int n = count - offset < chunk ? count - offset : chunk;
        int rc = candidate_hidden ? glm53f_target_model_prefill_hidden_12n(
                m, ids + offset, n, candidate_hidden) :
            glm53f_target_model_step_batch_12n(m, ids + offset, n, NULL, NULL, NULL, NULL);
        if (rc) return -1;
        if (reference_hidden && !candidate_hidden) {
            /* Use the export's flattened OpenMP loop. A serial nested loop
             * can be reassociated differently by FCC under -ffast-math. */
#pragma omp parallel for schedule(static)
            for (int k = 0; k < n * HIDDEN; k++) {
                int t = k / HIDDEN, i = k % HIDDEN;
                const float *stream = m->batch_streams + (size_t)t * FLAT;
                float z = 0;
                for (int s = 0; s < STREAMS; s++)
                    z += stream[(size_t)s * HIDDEN + i];
                reference_hidden[(size_t)(offset + t) * HIDDEN + i] = z / STREAMS;
            }
        } else if (reference_hidden) {
            for (int t = 0; t < n; ++t)
                for (int i = 0; i < HIDDEN; ++i)
                    *hidden_mismatches += memcmp(reference_hidden +
                        (size_t)(offset + t) * HIDDEN + i,
                        candidate_hidden + (size_t)t * HIDDEN + i,
                        sizeof(float)) != 0;
        }
        long local = available_kb(), minimum;
        MPI_Allreduce(&local, &minimum, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
        if (minimum < *minimum_kb) *minimum_kb = minimum;
        if (minimum < 2L * 1024 * 1024) return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    enum { LIMIT = 32768 };
    int rank, ranks, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 7 || ranks != 12 || provided < MPI_THREAD_SERIALIZED) {
        if (!rank) fprintf(stderr, "usage: %s MODEL ROUTED SHARED PROMPT_IDS TRACE_PREFIX CHUNK [--reference-chunk 512] [--capture-hidden] [--compare-moe-prefill-layout] [--compare-kda-prefill-kernel] [prefill/runtime options]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    char *end;
    long chunk = strtol(argv[6], &end, 10);
    if (!*argv[6] || *end || chunk < 1 || chunk > GLM53F_PREFILL_MAX_TOKENS)
        MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_prefill_config config = {GLM53F_PREFILL_FAST, 32, 27, NULL, 5};
    int capture_hidden = 0, compare_layout = 0, compare_kda = 0;
    int reference_chunk = 512;
    for (int i = 7; i < argc; ++i) {
        if (!strcmp(argv[i], "--compare-moe-prefill-layout")) { compare_layout = 1; continue; }
        if (!strcmp(argv[i], "--compare-kda-prefill-kernel")) { compare_kda = 1; continue; }
        if (!strcmp(argv[i], "--capture-hidden")) { capture_hidden = 1; continue; }
        if (!strcmp(argv[i], "--reference-chunk")) {
            if (++i == argc) MPI_Abort(MPI_COMM_WORLD, 2);
            long n = strtol(argv[i], &end, 10);
            if (!*argv[i] || *end || n < 1 || n > GLM53F_PREFILL_MAX_TOKENS)
                MPI_Abort(MPI_COMM_WORLD, 2);
            reference_chunk = (int)n;
            continue;
        }
        if (glm53f_prefill_option(&config, argc, argv, &i) != 1)
            MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (config.mode != GLM53F_PREFILL_FAST) MPI_Abort(MPI_COMM_WORLD, 2);
    if (compare_layout && setenv("GLM53F_MOE_GU_PAD", "0", 1)) MPI_Abort(MPI_COMM_WORLD, 2);
    if (compare_kda && setenv("GLM53F_KDA_PREFILL_COLUMNS", "0", 1)) MPI_Abort(MPI_COMM_WORLD, 2);
    int *ids = malloc(LIMIT * sizeof(int));
    if (!ids) MPI_Abort(MPI_COMM_WORLD, 2);
    int count = rank ? 0 : read_prompt(argv[4], ids, LIMIT);
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (count < 1) MPI_Abort(MPI_COMM_WORLD, 2);
    MPI_Bcast(ids, count, MPI_INT, 0, MPI_COMM_WORLD);
    if (!getenv("GLM53F_NUMA_INTERLEAVE") || atoi(getenv("GLM53F_NUMA_INTERLEAVE"))) {
        unsigned long mask = 0xF0UL;
        long rc = syscall(SYS_set_mempolicy, 3, &mask, 8UL);
        if (!rank) fprintf(stderr, "GLM53F_NUMA_INTERLEAVE %s\n", rc ? "unavailable" : "enabled");
    }
    if (glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"),
            glm53f_prefill_collective_count(count, HIDDEN)) ||
        glm53f_collective_prefill_algorithm_12n(config.collective))
        MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_target_model_12n *m = glm53f_target_model_create_12n(
        argv[1], argv[2], argv[3], count + 1);
    glm53f_target_snapshot_12n *empty = m ? glm53f_target_snapshot_create_12n(m) : NULL;
    float *reference = malloc(FLAT * sizeof(float));
    float *reference_hidden = capture_hidden ? malloc((size_t)count * HIDDEN * sizeof(float)) : NULL;
    float *candidate_hidden = capture_hidden ? malloc((size_t)chunk * HIDDEN * sizeof(float)) : NULL;
    if (capture_hidden && (!reference_hidden || !candidate_hidden)) MPI_Abort(MPI_COMM_WORLD, 2);
    if (!m || !empty || !reference ||
        glm53f_target_model_configure_prefill_12n(m, &config) ||
        glm53f_target_snapshot_save_12n(m, empty)) MPI_Abort(MPI_COMM_WORLD, 2);
    long minimum_kb = LONG_MAX;
    int ref_token, token, local_ok = 1;
    int hidden_mismatches = 0, max_hidden_mismatches;
    float ref_logit, logit;
    if (run_prompt(m, ids, count, reference_chunk, &minimum_kb,
            reference_hidden, NULL, &hidden_mismatches) || !m->last_streams ||
        glm53f_target_model_readout_12n(m, &ref_token, &ref_logit)) MPI_Abort(MPI_COMM_WORLD, 3);
    memcpy(reference, m->last_streams, FLAT * sizeof(float));
    /* Closing an endpoint-only trace writes all persistent KDA/sparse state.
     * It uses bounded I/O and fsync/fadvise, with exclusive reference writes. */
    if (glm53f_target_trace_open_12n(m, argv[5], 0) ||
        glm53f_target_trace_close_12n(m) ||
        glm53f_target_snapshot_restore_12n(m, empty)) MPI_Abort(MPI_COMM_WORLD, 2);
    if (compare_layout && setenv("GLM53F_MOE_GU_PAD", "1", 1)) MPI_Abort(MPI_COMM_WORLD, 2);
    if (compare_kda && setenv("GLM53F_KDA_PREFILL_COLUMNS", "1", 1)) MPI_Abort(MPI_COMM_WORLD, 2);
    if (run_prompt(m, ids, count, (int)chunk, &minimum_kb,
            reference_hidden, candidate_hidden, &hidden_mismatches) || !m->last_streams ||
        glm53f_target_model_readout_12n(m, &token, &logit)) MPI_Abort(MPI_COMM_WORLD, 3);
    int mismatches = 0, max_mismatches;
    for (int i = 0; i < FLAT; ++i)
        mismatches += memcmp(reference + i, m->last_streams + i, sizeof(float)) != 0;
    local_ok &= !mismatches && !hidden_mismatches && token == ref_token && !memcmp(&logit, &ref_logit, sizeof(float));
    int state_ok = !glm53f_target_trace_open_12n(m, argv[5], 1);
    if (state_ok) state_ok = !glm53f_target_trace_close_12n(m);
    local_ok &= state_ok;
    int all_ok, all_state;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&state_ok, &all_state, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&mismatches, &max_mismatches, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&hidden_mismatches, &max_hidden_mismatches, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (!rank) printf("GLM53F_PREFILL_CHUNK_CHECK tokens=%d reference_chunk=%d candidate_chunk=%ld capacity=%d compare_moe_layout=%d compare_kda_columns=%d hidden_bit_mismatches=%d capture_hidden=%d prompt_hidden_bit_mismatches=%d state=%s first_token=%d/%d min_MemAvailable_GiB=%.6f %s\n",
        count, reference_chunk, chunk, GLM53F_PREFILL_MAX_TOKENS, compare_layout, compare_kda, max_mismatches,
        capture_hidden, max_hidden_mismatches,
        all_state ? "BIT_EXACT" : "MISMATCH", ref_token, token, minimum_kb / 1048576.0,
        all_ok ? "PASS" : "FAIL");
    free(reference);
    free(reference_hidden);
    free(candidate_hidden);
    free(ids);
    glm53f_target_snapshot_free_12n(empty);
    glm53f_target_model_free_12n(m);
    glm53f_collective_free_12n();
    MPI_Finalize();
    return all_ok ? 0 : 1;
}
