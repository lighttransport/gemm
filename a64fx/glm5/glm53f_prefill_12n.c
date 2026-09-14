#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "glm53f_collective_12n.h"
#include "glm53f_target_model_12n.h"

static int positive_int(const char *s, int maximum) {
    char *end;
    long n = strtol(s, &end, 10);
    return !*s || *end || n < 1 || n > maximum ? -1 : (int)n;
}

static long available_kb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    char line[256];
    long kb = 0;
    while (f && fgets(line, sizeof(line), f))
        if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    if (f) fclose(f);
    return kb;
}

int main(int argc, char **argv) {
    int rank, ranks;
    int positions = 512, chunk = 5, first_option = 4;
    if (argc > first_option && strncmp(argv[first_option], "--", 2))
        positions = positive_int(argv[first_option++], 524288);
    if (argc > first_option && strncmp(argv[first_option], "--", 2))
        chunk = positive_int(argv[first_option++], GLM53F_PREFILL_MAX_TOKENS);
    int repeats = 1;
    const char *input_path = NULL;
    const char *trace_path = NULL;
    int trace_compare = 0, bad_option = 0, sweep = 0, chunk_sweep = 0, scalar = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    setvbuf(stdout, NULL, _IOLBF, 0);
    for (int a = first_option; a < argc; ++a) {
        if (!strcmp(argv[a], "--input-ids") && a + 1 < argc) input_path = argv[++a];
        else if (!strcmp(argv[a], "--sweep")) sweep = 1;
        else if (!strcmp(argv[a], "--chunk-sweep")) chunk_sweep = 1;
        else if (!strcmp(argv[a], "--scalar-reference")) scalar = 1;
        else if (!strcmp(argv[a], "--repeats") && a + 1 < argc)
            repeats = positive_int(argv[++a], 10);
        else if (!strcmp(argv[a], "--state-out") && a + 1 < argc) {
            trace_path = argv[++a]; trace_compare = 0;
        } else if (!strcmp(argv[a], "--state-check") && a + 1 < argc) {
            trace_path = argv[++a]; trace_compare = 1;
        } else bad_option = 1;
    }
    if (argc < 4 || ranks != 12 || positions < 1 || chunk < 1 || repeats < 1 ||
        bad_option || ((trace_path || sweep) && repeats != 1) ||
        (chunk_sweep && (sweep || trace_path || scalar)) || (scalar && sweep)) {
        if (!rank) fprintf(stderr, "usage: %s MODEL ROUTED SHARED [POSITIONS [CHUNK]] "
            "[--input-ids FILE] [--repeats N] [--sweep | --chunk-sweep | --scalar-reference] "
            "[--state-out PREFIX | --state-check PREFIX] "
            "(12 ranks; chunk <= %d; state diagnostics require repeats=1)\n",
            argv[0], GLM53F_PREFILL_MAX_TOKENS);
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
    int *input = malloc((size_t)positions * sizeof(*input));
    if (!input) MPI_Abort(MPI_COMM_WORLD, 2);
    if (!rank) {
        FILE *file = input_path ? fopen(input_path, "r") : NULL;
        if (input_path && !file) MPI_Abort(MPI_COMM_WORLD, 2);
        for (int t = 0; t < positions; ++t) {
            if (file) {
                if (fscanf(file, "%d", input + t) != 1 ||
                    input[t] < 0 || input[t] >= 154880) {
                    fprintf(stderr, "invalid or missing input token at position %d\n", t);
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }
            } else input[t] = 1 + (int)(((long long)t * 104729) % 154000);
        }
        if (file) fclose(file);
    }
    MPI_Bcast(input, positions, MPI_INT, 0, MPI_COMM_WORLD);
    /* The outer prompt tile is consumed by four-position arithmetic panels;
     * no collective carries the complete tile.  Register only the largest
     * actual payload, avoiding a needlessly large uTofu region at chunk 32. */
    int collective_tokens = sweep || chunk_sweep ? 4 : chunk < 4 ? chunk : 4;
    if (getenv("GLM53F_UTOFU") &&
        glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"),
                                  collective_tokens * 4096))
        MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_target_model_12n *m = glm53f_target_model_create_12n(
        argv[1], argv[2], argv[3], positions + 1);
    if (!m || !input) MPI_Abort(MPI_COMM_WORLD, 2);
    if (getenv("GLM53F_PREFILL_INT8") &&
        atoi(getenv("GLM53F_PREFILL_INT8")) &&
        glm53f_target_model_convert_int8_12n(m))
        MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_target_snapshot_12n *initial = glm53f_target_snapshot_create_12n(m);
    if (!initial || glm53f_target_snapshot_save_12n(m, initial)) MPI_Abort(MPI_COMM_WORLD, 2);
    /* Untimed state probe: equal token/logit and hidden checksums across chunk
     * sizes validate the recurrent and sparse-cache boundary reached by the
     * prompt-only scheduler without retaining every intermediate hidden. */
    int probe_token = 0;
    float probe_logit = 0.0f;
    float *probe_hidden = malloc(4096 * sizeof(*probe_hidden));
    if (!probe_hidden) MPI_Abort(MPI_COMM_WORLD, 2);
    double rates[40];
    const int sweep_chunks[] = {32, 256, 256, 256, 32, 32, 32, 256, 256, 1};
    const int sweep_modes[] = {0, 0, 1, 2, 4, 8, 20, 64, 95, 32};
    const char *sweep_names[] = {"control32", "control256", "int8_grouped", "mhc", "sparse", "kda", "sparse_index", "router", "combined", "scalar_reference"};
    const int chunks[] = {32, 64, 128, 256};
    int passes = sweep ? 10 : chunk_sweep ? 4 * repeats : repeats;
    for (int rep = 0; rep < passes; ++rep) {
        if (sweep) {
            chunk = sweep_chunks[rep];
            setenv("GLM53F_MOE_I8_GROUPED", sweep_modes[rep] & 1 ? "1" : "0", 1);
            setenv("GLM53F_MHC_PREFILL", sweep_modes[rep] & 2 ? "1" : "0", 1);
            setenv("GLM53F_SPARSE_PREFILL", sweep_modes[rep] & 4 ? "1" : "0", 1);
            setenv("GLM53F_KDA_PREFILL", sweep_modes[rep] & 8 ? "1" : "0", 1);
            setenv("GLM53F_SPARSE_INDEX_BATCH", sweep_modes[rep] & 16 ? "1" : "0", 1);
            setenv("GLM53F_MOE_ROUTER_PREFILL", sweep_modes[rep] & 64 ? "1" : "0", 1);
        }
        if (chunk_sweep) chunk = chunks[rep / repeats];
        if (scalar) chunk = 1;
        int scalar_run = scalar || (sweep && (sweep_modes[rep] & 32));
        int checking = trace_compare || (sweep && rep > 0);
        if (trace_path && glm53f_target_trace_open_12n(m, trace_path, checking))
            MPI_Abort(MPI_COMM_WORLD, 2);
        if (rep && glm53f_target_snapshot_restore_12n(m, initial)) MPI_Abort(MPI_COMM_WORLD, 2);
        glm53f_target_profile_reset_12n(m);
        MPI_Barrier(MPI_COMM_WORLD);
        double begin = MPI_Wtime();
        long minimum_kb = LONG_MAX;
        int next_memory_sample = 0;
        for (int base = 0; base < positions; base += chunk) {
            /* Sample at most 511 + chunk positions apart even for a chunk
             * that does not divide 512 (for example the legacy default 5). */
            if (base >= next_memory_sample || base + chunk >= positions) {
                next_memory_sample = base + 512;
                long kb = available_kb();
                if (kb < minimum_kb) minimum_kb = kb;
                if (kb < 2L * 1048576) {
                    fprintf(stderr, "PREFILL_HEADROOM rank=%d position=%d available_kb=%ld\n", rank, base, kb);
                    MPI_Abort(MPI_COMM_WORLD, 5);
                }
            }
            int n = positions - base;
            if (n > chunk) n = chunk;
            int rc = scalar_run ? glm53f_target_model_step_12n(m, input[base],
                &probe_token, &probe_logit, NULL) : glm53f_target_model_step_batch_12n(
                    m, input + base, n, NULL, NULL, NULL, NULL);
            if (rc)
                MPI_Abort(MPI_COMM_WORLD, 3);
        }
        long final_kb = available_kb();
        if (final_kb < minimum_kb) minimum_kb = final_kb;
        if (final_kb < 2L * 1048576) MPI_Abort(MPI_COMM_WORLD, 5);
        double elapsed = MPI_Wtime() - begin, maximum;
        MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        long global_minimum_kb;
        MPI_Reduce(&minimum_kb, &global_minimum_kb, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        if (!rank) {
            rates[rep] = positions / maximum;
            printf("GLM53F_PREFILL_12N positions=%d chunk=%d seconds=%.6f tok_s=%.3f "
                   "repeat=%d source=%s validation_io=%d case=%s min_available_GiB=%.3f\n", positions, chunk, maximum, rates[rep],
                   rep + 1, input_path ? input_path : "synthetic", trace_path != NULL,
                   sweep ? sweep_names[rep] : scalar ? "scalar" : "configured", global_minimum_kb / 1048576.0);
        }
        glm53f_target_profile_report_12n(m, "prefill");
        if (trace_path) {
            int ok = glm53f_target_trace_close_12n(m) == 0, global_ok;
            MPI_Allreduce(&ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (!rank) printf("GLM53F_PREFILL_STATE mode=%s %s\n",
                checking ? "compare" : "write", global_ok ? "PASS" : "FAIL");
            if (!global_ok) MPI_Abort(MPI_COMM_WORLD, 4);
        }
        if (glm53f_target_model_step_12n(
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
    }
    if (!rank && !sweep) {
        for (int group = 0; group < (chunk_sweep ? 4 : 1); ++group) {
            double *group_rates = rates + group * repeats;
            for (int i = 1; i < repeats; ++i)
                for (int j = i; j > 0 && group_rates[j] < group_rates[j - 1]; --j) {
                    double tmp = group_rates[j]; group_rates[j] = group_rates[j - 1]; group_rates[j - 1] = tmp;
                }
            double median = (group_rates[(repeats - 1) / 2] + group_rates[repeats / 2]) / 2;
            double total_seconds_per_token = 0.0;
            for (int i = 0; i < repeats; ++i) total_seconds_per_token += 1.0 / group_rates[i];
            printf("GLM53F_PREFILL_SUMMARY chunk=%d repeats=%d median_tok_s=%.3f min_tok_s=%.3f max_tok_s=%.3f avg_tok_s=%.3f\n",
                   chunk_sweep ? chunks[group] : chunk, repeats, median, group_rates[0],
                   group_rates[repeats - 1], repeats / total_seconds_per_token);
        }
    }
    glm53f_target_snapshot_free_12n(initial);
    free(probe_hidden);
    free(input);
    glm53f_target_model_free_12n(m);
    glm53f_collective_free_12n();
    MPI_Finalize();
    return 0;
}
