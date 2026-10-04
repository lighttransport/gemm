/* Resident, snapshot-reset prefill/decode benchmark. Timing includes the final
 * prompt head, and counts only decode transitions after its first prediction. */
#define _GNU_SOURCE
#include "glm53f_clock.h"
#include "glm53f_target_model_12n.h"
#include "glm53f_lookup_spec_12n.h"
#include "glm53f_mtp_spec_12n.h"
#include "glm53f_collective_12n.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <omp.h>
/* Optional diagnostic sync profile (glm53f_sync_profile.c); absent in normal builds. */
extern void glm53f_sync_profile_reset(void) __attribute__((weak));
extern void glm53f_sync_profile_report(const char *label, long positions) __attribute__((weak));

static int positive(const char *s, int limit) {
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    return !errno && *s && !*end && n > 0 && n <= limit ? (int)n : -1;
}
static long available(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    long n = -1;
    char line[256];
    if (!f) return -1;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "MemAvailable: %ld kB", &n) == 1) break;
    fclose(f);
    return n;
}
static long headroom(void) {
    long n = available(), minimum;
    MPI_Allreduce(&n, &minimum, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
    if (minimum < 2L * 1024 * 1024) {
        fprintf(stderr, "GLM53F_BENCH_HEADROOM minimum_kb=%ld reject\n", minimum);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    return minimum;
}
static double elapsed(double begin) {
    double dt = glm53f_clock() - begin, maximum;
    MPI_Allreduce(&dt, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return maximum;
}
static void check_at(int rc, int line) {
    if (rc) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "GLM53F_BENCH_CHECK_FAIL rank=%d line=%d rc=%d\n", rank, line, rc);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
#define check(rc) check_at((rc), __LINE__)

struct memory_guard { long minimum; int transitions, last_interval; };
static void guard_decode(void *context, int completed) {
    struct memory_guard *a = context;
    if (completed / 32 != a->last_interval || completed == a->transitions) {
        a->last_interval = completed / 32;
        long h = headroom();
        if (h < a->minimum) a->minimum = h;
    }
}

int main(int argc, char **argv) {
    int rank, ranks;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 6 || ranks != 12 || provided < MPI_THREAD_SERIALIZED) {
        if (!rank) fprintf(stderr, "usage: %s MODEL ROUTED SHARED PROMPT_IDS OUTPUT_IDS [--transitions 256] [--repetitions 3] [--prefill-chunk 512] [--speculation none|lookup|mtp] [--mtp-prime-batch 1|64] [--mtp-routed-stage PATH] [--mtp-shared-stage PATH] [--draft-depth 1..4] [--spec-policy adaptive|always] [--decode-state-check TRACE_PREFIX] [prefill/runtime options]\n", argv[0]);
        MPI_Finalize();
        return 2;
    }
    if (!getenv("GLM53F_NUMA_INTERLEAVE") || atoi(getenv("GLM53F_NUMA_INTERLEAVE"))) {
        unsigned long mask = 0xF0UL;
        long rc = syscall(SYS_set_mempolicy, 3, &mask, 8UL);
        if (!rank) fprintf(stderr, "GLM53F_NUMA_INTERLEAVE %s\n", rc == 0 ? "enabled" : "unavailable");
    }
    int transitions = 256, repetitions = 3, chunk = 512;
    int speculation = 0, draft_depth = 4, adaptive = 1, mtp_prime_batch = 1;
    const char *mtp_routed = NULL, *mtp_shared = NULL;
    const char *state_check = NULL;
    glm53f_prefill_config config = {GLM53F_PREFILL_FAST, 32, 27, NULL, 5};
    for (int i = 6; i < argc; ++i) {
        if (!strcmp(argv[i], "--mtp-prime-batch")) {
            if (++i == argc || (strcmp(argv[i], "1") && strcmp(argv[i], "64"))) MPI_Abort(MPI_COMM_WORLD, 2);
            mtp_prime_batch = atoi(argv[i]); continue;
        }
        int rc = glm53f_prefill_option(&config, argc, argv, &i);
        if (rc < 0) MPI_Abort(MPI_COMM_WORLD, 2);
        if (rc) continue;
        if (!strcmp(argv[i], "--speculation") || !strcmp(argv[i], "--spec-policy")) {
            const char *key = argv[i];
            if (++i == argc) MPI_Abort(MPI_COMM_WORLD, 2);
            if (!strcmp(key, "--speculation")) {
                if (!strcmp(argv[i], "lookup")) speculation = 1;
                else if (!strcmp(argv[i], "mtp")) speculation = 2;
                else if (!strcmp(argv[i], "none")) speculation = 0;
                else MPI_Abort(MPI_COMM_WORLD, 2);
            } else {
                if (!strcmp(argv[i], "adaptive")) adaptive = 1;
                else if (!strcmp(argv[i], "always")) adaptive = 0;
                else MPI_Abort(MPI_COMM_WORLD, 2);
            }
            continue;
        }
        if (!strcmp(argv[i], "--mtp-routed-stage") || !strcmp(argv[i], "--mtp-shared-stage")) {
            const char *key = argv[i];
            if (++i == argc || !*argv[i]) MPI_Abort(MPI_COMM_WORLD, 2);
            if (!strcmp(key, "--mtp-routed-stage")) mtp_routed = argv[i];
            else mtp_shared = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "--decode-state-check")) {
            if (++i == argc || !*argv[i]) MPI_Abort(MPI_COMM_WORLD, 2);
            state_check = argv[i];
            continue;
        }
        int *dst = NULL, limit = 0;
        if (!strcmp(argv[i], "--transitions")) { dst = &transitions; limit = 32768; }
        else if (!strcmp(argv[i], "--repetitions")) { dst = &repetitions; limit = 10; }
        else if (!strcmp(argv[i], "--draft-depth")) { dst = &draft_depth; limit = 4; }
        else if (!strcmp(argv[i], "--prefill-chunk")) { dst = &chunk; limit = GLM53F_PREFILL_MAX_TOKENS; }
        if (!dst || i + 1 == argc || (*dst = positive(argv[++i], limit)) < 1)
            MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (config.mode != GLM53F_PREFILL_FAST && chunk > 256)
        MPI_Abort(MPI_COMM_WORLD, 2);
    if ((speculation == 2 && (!mtp_routed || !mtp_shared)) ||
        (speculation != 2 && (mtp_routed || mtp_shared || mtp_prime_batch != 1))) MPI_Abort(MPI_COMM_WORLD, 2);
    FILE *output = NULL;
    if (!rank && !(output = fopen(argv[5], "wx"))) MPI_Abort(MPI_COMM_WORLD, 2);
    if (!rank) {
#ifdef __FAST_MATH__
        const int fast_math = 1;
#else
        const int fast_math = 0;
#endif
        printf("GLM53F_BENCH_CONFIG {\"ranks\":12,\"threads\":%d,\"fast_math\":%d,\"prefill_chunk\":%d,\"prefill_capacity\":%d,\"attention_panel\":%d,\"dense_prefill_tile\":%d,\"prefill_features\":%u,\"collective\":%d,\"persistent\":%d,\"grouped_verify\":%d,\"router_fused\":%d,\"mhc_fused_sync\":%d,\"mhc_batch_team\":%d,\"kda_decode_columns\":%d,\"kda_prefill_columns\":%d,\"iq_scale_words\":%d,\"mla_parallel_softmax\":%d,\"moe_gu_padding\":%d,\"serialized_owner\":%d}\n",
            omp_get_max_threads(), fast_math, chunk, GLM53F_PREFILL_MAX_TOKENS, GLM53F_PREFILL_ATTN_TOKENS, getenv("GLM53F_DENSE_PREFILL_TILE") ? atoi(getenv("GLM53F_DENSE_PREFILL_TILE")) : 4, config.features, config.collective,
            getenv("GLM53F_DECODE_EXECUTOR") ? !!atoi(getenv("GLM53F_DECODE_EXECUTOR")) : 0,
            getenv("GLM53F_VERIFY_GROUPED") ? !!atoi(getenv("GLM53F_VERIFY_GROUPED")) : 0,
            getenv("GLM53F_ROUTER_FUSE") ? !!atoi(getenv("GLM53F_ROUTER_FUSE")) : 0,
            getenv("GLM53F_MHC_FUSED_SYNC") ? !!atoi(getenv("GLM53F_MHC_FUSED_SYNC")) : 0,
            getenv("GLM53F_MHC_BATCH_TEAM") ? !!atoi(getenv("GLM53F_MHC_BATCH_TEAM")) : 0,
            getenv("GLM53F_KDA_DECODE_COLUMNS") ? !!atoi(getenv("GLM53F_KDA_DECODE_COLUMNS")) : 0,
            getenv("GLM53F_KDA_PREFILL_COLUMNS") ? !!atoi(getenv("GLM53F_KDA_PREFILL_COLUMNS")) : 0,
            getenv("GLM53F_IQ_SCALE_WORDS") && atoi(getenv("GLM53F_IQ_SCALE_WORDS")),
            getenv("GLM53F_MLA_PARALLEL_SOFTMAX") && atoi(getenv("GLM53F_MLA_PARALLEL_SOFTMAX")),
            getenv("GLM53F_MOE_GU_PAD") && atoi(getenv("GLM53F_MOE_GU_PAD")) ? 64 : 0,
            getenv("GLM53F_COMM_OWNER") ? !!atoi(getenv("GLM53F_COMM_OWNER")) : 0);
        fflush(stdout);
    }
    int *prompt = NULL, count = 0, capacity = 1024;
    if (!rank) {
        FILE *f = fopen(argv[4], "r");
        prompt = malloc((size_t)capacity * sizeof(int));
        if (!f || !prompt) MPI_Abort(MPI_COMM_WORLD, 2);
        long token;
        int rc;
        while ((rc = fscanf(f, "%ld", &token)) == 1) {
            if (token < 0 || token >= 154880 || count >= 262144)
                MPI_Abort(MPI_COMM_WORLD, 2);
            if (count == capacity) {
                capacity *= 2;
                int *p = realloc(prompt, (size_t)capacity * sizeof(int));
                if (!p) MPI_Abort(MPI_COMM_WORLD, 2);
                prompt = p;
            }
            prompt[count++] = (int)token;
        }
        if (rc != EOF || ferror(f) || !count) MPI_Abort(MPI_COMM_WORLD, 2);
        fclose(f);
    }
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank) prompt = malloc((size_t)count * sizeof(int));
    if (!prompt) MPI_Abort(MPI_COMM_WORLD, 2);
    MPI_Bcast(prompt, count, MPI_INT, 0, MPI_COMM_WORLD);
    if (!strcmp(argv[4], argv[5])) MPI_Abort(MPI_COMM_WORLD, 2);
    /* Sparse prefill packs up to 32 index rows, each with floor(context/4)
     * pooled scores. The output projection's 32*4096 reservation alone only
     * covers contexts through 16K. Keep the original reservation for short
     * prompts, and reserve the packed selector reduction for longer ones. */
    int collective_count = glm53f_prefill_collective_count(count, 4096);
    check(glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"), collective_count));
    check(glm53f_collective_prefill_algorithm_12n(config.collective));
    double load = glm53f_clock();
    glm53f_target_model_12n *m = glm53f_target_model_create_12n(
        argv[1], argv[2], argv[3], count + transitions + 1);
    if (!m) {
        fprintf(stderr, "GLM53F_BENCH_CREATE_FAIL rank=%d phase=target\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    check(glm53f_target_model_configure_prefill_12n(m, &config));
    if (getenv("GLM53F_CMG_PLACE") && atoi(getenv("GLM53F_CMG_PLACE"))) {
        /* Value 2 also places routed experts and selects their CMG-affine row split. */
        if (atoi(getenv("GLM53F_CMG_PLACE")) >= 2) check(setenv("GLM53F_IQ_AFFINE", "1", 1));
        check(glm53f_target_place_weights_12n(m));
    }
    glm53f_mtp_context_12n *mtp = NULL;
    glm53f_mtp_spec_workspace_12n *mtp_workspace = NULL;
    float *prompt_hidden = NULL;
    float parent_hidden[4096];
    if (speculation == 2) {
        /* Layer 45 comes from the checkpoint. Relax the compact target-only
         * requirement during its construction, retaining all target stages. */
        const char *old_requirement = getenv("GLM53F_REPACK_REQUIRE");
        char *saved_requirement = old_requirement ? strdup(old_requirement) : NULL;
        if (old_requirement && !saved_requirement) MPI_Abort(MPI_COMM_WORLD, 2);
        check(setenv("GLM53F_REPACK_REQUIRE", "0", 1));
        mtp = glm53f_mtp_create_12n(argv[1], mtp_routed, mtp_shared,
                                  count + transitions + draft_depth + 1);
        check(saved_requirement ? setenv("GLM53F_REPACK_REQUIRE", saved_requirement, 1) :
                                  unsetenv("GLM53F_REPACK_REQUIRE"));
        free(saved_requirement);
        mtp_workspace = glm53f_mtp_spec_workspace_create_12n(m);
        prompt_hidden = malloc((size_t)chunk * 4096 * sizeof(float));
        if (!mtp || !mtp_workspace || !prompt_hidden) {
            fprintf(stderr, "GLM53F_BENCH_CREATE_FAIL rank=%d phase=mtp context=%d workspace=%d hidden=%d\n",
                    rank, !!mtp, !!mtp_workspace, !!prompt_hidden);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    load = elapsed(load);
    glm53f_target_snapshot_12n *empty = glm53f_target_snapshot_create_12n(m);
    glm53f_target_snapshot_12n *primed = glm53f_target_snapshot_create_12n(m);
    int *reference = malloc((size_t)(transitions + 1) * sizeof(int));
    int *ids = malloc((size_t)(transitions + 1) * sizeof(int));
    if (!empty || !primed || !reference || !ids) {
        fprintf(stderr, "GLM53F_BENCH_CREATE_FAIL rank=%d phase=snapshot empty=%d primed=%d reference=%d ids=%d\n",
                rank, !!empty, !!primed, !!reference, !!ids);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    check(glm53f_target_snapshot_save_12n(m, empty));
    glm53f_lookup_workspace_12n *lookup = speculation == 1 ?
        glm53f_lookup_workspace_create_12n(m, count + transitions + 1) : NULL;
    if (speculation == 1 && !lookup) MPI_Abort(MPI_COMM_WORLD, 2);
    if (!rank) printf("GLM53F_BENCH_MTP_PRIME batch=%d verify_head_shared=%d embedding_packed=%d\n", mtp_prime_batch, getenv("GLM53F_HEAD_VERIFY_SHARED") && atoi(getenv("GLM53F_HEAD_VERIFY_SHARED")), getenv("GLM53F_EMBED_BATCH_PACKED") && atoi(getenv("GLM53F_EMBED_BATCH_PACKED")));
    long minimum = headroom();
    double warmup_prefill = 0, warmup_decode = 0;
    for (int trial = -1; trial < repetitions; ++trial) {
        check(glm53f_target_snapshot_restore_12n(m, empty));
        if (mtp) check(glm53f_mtp_restore_length_12n(mtp, 0));
        glm53f_target_profile_reset_12n(m);
        MPI_Barrier(MPI_COMM_WORLD);
        double begin = glm53f_clock();
        for (int t = 0; t < count; t += chunk) {
            int n = count - t < chunk ? count - t : chunk;
            int rc = mtp ? glm53f_target_model_prefill_hidden_12n(m, prompt + t, n, prompt_hidden) :
                glm53f_target_model_step_batch_12n(m, prompt + t, n, NULL, NULL, NULL, NULL);
            if (rc) fprintf(stderr, "GLM53F_BENCH_PREFILL_FAIL rank=%d trial=%d offset=%d tokens=%d collective_count=%d rc=%d\n",
                            rank, trial, t, n, collective_count, rc);
            check(rc);
            if (mtp) {
                check(glm53f_target_model_normalize_hidden_12n(m, prompt_hidden, n));
                int pairs = n;
                if (t + n == count) --pairs;
                for (int j = 0; j < pairs; j += mtp_prime_batch) {
                    int tile = pairs - j < mtp_prime_batch ? pairs - j : mtp_prime_batch;
                    if (mtp_prime_batch == 1)
                        check(glm53f_mtp_cache_append_12n(mtp, prompt[t + j + 1],
                                                       prompt_hidden + (size_t)j * 4096));
                    else check(glm53f_mtp_cache_append_batch_12n(mtp, prompt + t + j + 1,
                                                               prompt_hidden + (size_t)j * 4096, tile));
                }
                if (t + n == count)
                    memcpy(parent_hidden, prompt_hidden + (size_t)(n - 1) * 4096,
                           sizeof(parent_hidden));
            }
            long h = headroom();
            if (h < minimum) minimum = h;
        }
        float logit;
        check(glm53f_target_model_readout_12n(m, &ids[0], &logit));
        if (mtp) check(glm53f_target_model_head_hidden_12n(m, parent_hidden, 1));
        double prefill = elapsed(begin);
        glm53f_target_profile_report_12n(m, "bench-prefill");
        check(glm53f_target_snapshot_save_12n(m, primed));
        check(glm53f_target_snapshot_restore_12n(m, primed));
        glm53f_target_profile_reset_12n(m);
        if (glm53f_sync_profile_reset) glm53f_sync_profile_reset();
        MPI_Barrier(MPI_COMM_WORLD);
        begin = glm53f_clock();
        struct memory_guard guard = {minimum, transitions, 0};
        glm53f_lookup_stats_12n spec_stats = {0};
        glm53f_mtp_spec_stats_12n mtp_stats = {0};
        if (trial >= 0 && speculation == 2)
            check(glm53f_mtp_spec_decode_12n(m, mtp, mtp_workspace, count, parent_hidden,
                ids[0], transitions, draft_depth, adaptive ? warmup_decode / transitions : 0,
                ids, &mtp_stats, guard_decode, &guard));
        else if (trial >= 0 && speculation == 1)
            check(glm53f_lookup_decode_12n(m, lookup, prompt, count, ids[0], transitions,
                draft_depth, adaptive ? warmup_decode / transitions : 0,
                ids, &spec_stats, guard_decode, &guard));
        else check(glm53f_target_decode_sequence_12n(m, ids[0], transitions, ids, guard_decode, &guard));
        minimum = guard.minimum;
        double decode = elapsed(begin);
        glm53f_target_profile_report_12n(m, "bench-decode");
        if (glm53f_sync_profile_report) glm53f_sync_profile_report("bench-decode", transitions);
        int equal = trial < 0 || !memcmp(ids, reference,
                            (size_t)(transitions + 1) * sizeof(int)), all;
        MPI_Allreduce(&equal, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!all) MPI_Abort(MPI_COMM_WORLD, 4);
        if (state_check) {
            /* Endpoint I/O runs outside timing. Compare complete KDA/sparse
             * state against the plain warmup, including rejected suffixes. */
            check(glm53f_target_trace_open_12n(m, state_check, trial >= 0));
            check(glm53f_target_trace_close_12n(m));
            if (!rank) printf("GLM53F_BENCH_STATE trial=%d state=BIT_EXACT PASS\n", trial);
        }
        if (trial < 0) {
            memcpy(reference, ids, (size_t)(transitions + 1) * sizeof(int));
            warmup_prefill = prefill; warmup_decode = decode;
        } else if (!rank) {
            printf("GLM53F_BENCH_SPEC {\"trial\":%d,\"lookup\":%d,\"depth\":%d,\"adaptive\":%d,\"accepted\":%d,\"proposed\":%d,\"cycles\":%d,\"fallback_tokens\":%d,\"lookup_seconds\":%.9f,\"verify_seconds\":%.9f,\"rollback_seconds\":%.9f,\"plain_seconds\":%.9f}\n",
                trial, speculation == 1, draft_depth, adaptive, spec_stats.accepted, spec_stats.proposed,
                spec_stats.cycles, spec_stats.fallback_tokens, spec_stats.lookup_seconds,
                spec_stats.verify_seconds, spec_stats.rollback_seconds, spec_stats.plain_seconds);
            if (mtp) printf("GLM53F_BENCH_MTP {\"trial\":%d,\"depth\":%d,\"adaptive\":%d,\"accepted\":%d,\"proposed\":%d,\"cycles\":%d,\"fallback_tokens\":%d,\"cache_synchronized\":%d,\"draft_seconds\":%.9f,\"verify_seconds\":%.9f,\"replay_seconds\":%.9f,\"plain_seconds\":%.9f}\n",
                trial, draft_depth, adaptive, mtp_stats.accepted, mtp_stats.proposed,
                mtp_stats.cycles, mtp_stats.fallback_tokens, mtp_stats.cache_synchronized,
                mtp_stats.draft_seconds, mtp_stats.verify_seconds, mtp_stats.replay_seconds,
                mtp_stats.plain_seconds);
            printf("GLM53F_BENCH_TRIAL {\"trial\":%d,\"prompt_tokens\":%d,\"decode_transitions\":%d,\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,\"prefill_tok_s\":%.6f,\"decode_tok_s\":%.6f,\"minimum_available_kb\":%ld,\"tokens_equal\":true}\n",
                   trial, count, transitions, prefill, decode,
                   count / prefill, transitions / decode, minimum);
            fflush(stdout);
        }
    }
    if (!rank) {
        for (int t = 0; t <= transitions; ++t) fprintf(output, "%d\n", reference[t]);
        if (fclose(output)) MPI_Abort(MPI_COMM_WORLD, 2);
        printf("GLM53F_BENCH_COMPLETE {\"repetitions\":%d,\"load_seconds\":%.9f,\"warmup_prefill_seconds\":%.9f,\"warmup_decode_seconds\":%.9f,\"minimum_available_kb\":%ld,\"status\":\"PASS\"}\n",
               repetitions, load, warmup_prefill, warmup_decode, minimum);
    }
    glm53f_lookup_workspace_free_12n(lookup);
    free(prompt_hidden);
    glm53f_mtp_spec_workspace_free_12n(mtp_workspace);
    glm53f_mtp_free_12n(mtp);
    free(ids); free(reference); free(prompt);
    glm53f_target_snapshot_free_12n(primed);
    glm53f_target_snapshot_free_12n(empty);
    glm53f_target_model_free_12n(m);
    glm53f_collective_free_12n();
    MPI_Finalize();
    return 0;
}
