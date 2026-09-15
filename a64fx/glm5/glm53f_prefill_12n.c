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

/* Single resident model, v5 then candidate. Store only this rank's 256 logit
 * shards (~13 MiB), never gather the vocabulary. Reductions use FP64. */
static int qualify(glm53f_target_model_12n *m, glm53f_target_snapshot_12n *empty,
        const int *input, int positions, int chunk, const glm53f_prefill_config *candidate,
        const char *prefix, int rank) {
    enum {STEPS = 256};
    int ids[STEPS+1], top1[STEPS], first, count;
    if (!glm53f_target_model_logits_12n(m,&first,&count)) MPI_Abort(MPI_COMM_WORLD, 4);
    float *reference = malloc((size_t)STEPS*count*sizeof(float));
    if (!reference) MPI_Abort(MPI_COMM_WORLD, 4);
    double mean_kl = 0, nll = 0;
    int matches = 0, evaluated = 0, rc = 0;
    for (int pass=0; pass<2 && !rc; ++pass) {
        glm53f_prefill_config cfg = *candidate;
        if (!pass) cfg.mode = GLM53F_PREFILL_V5;
        if (glm53f_target_snapshot_restore_12n(m,empty) ||
            glm53f_target_model_configure_prefill_12n(m,&cfg) ||
            glm53f_target_trace_open_12n(m,prefix,pass ? 2 : 0)) MPI_Abort(MPI_COMM_WORLD, 4);
        int panel = !pass && chunk > GLM53F_PREFILL_V5_TOKENS ? GLM53F_PREFILL_V5_TOKENS : chunk;
        for (int t=0; t<positions && !rc; t+=panel) {
            if (available_kb() < 2L*1048576) MPI_Abort(MPI_COMM_WORLD, 5);
            int n=positions-t<panel ? positions-t : panel;
            rc=glm53f_target_model_step_batch_12n(m,input+t,n,NULL,NULL,NULL,NULL);
            int bad=rc!=0, any;
            MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
            if (any) rc=-1;
        }
        if (glm53f_target_trace_close_12n(m)) rc=-1;
        int bad=rc!=0, any;
        MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        if (any) { rc=-1; break; }
        for (int t=0; t<STEPS; ++t) {
            int token; float logit;
            if (available_kb() < 2L*1048576) MPI_Abort(MPI_COMM_WORLD, 5);
            int step_rc = t ? glm53f_target_model_step_12n(m,ids[t],&token,&logit,NULL) :
                              glm53f_target_model_readout_12n(m,&token,&logit);
            if (step_rc) MPI_Abort(MPI_COMM_WORLD, 5);
            const float *got=glm53f_target_model_logits_12n(m,&first,&count);
            float *ref=reference+(size_t)t*count;
            if (!pass) {
                memcpy(ref,got,(size_t)count*sizeof(float));
                ids[t+1]=top1[t]=token;
                continue;
            }
            matches += token==top1[t];
            ++evaluated;
            double localmax[2]={-INFINITY,-INFINITY}, maximum[2];
            for (int r=0;r<count;++r) {
                if (!isfinite(ref[r]) || !isfinite(got[r])) rc=-1;
                if (ref[r]>localmax[0]) localmax[0]=ref[r];
                if (got[r]>localmax[1]) localmax[1]=got[r];
            }
            MPI_Allreduce(localmax,maximum,2,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
            double local[5]={0}, sum[5];
            for (int r=0;r<count;++r) {
                double p=exp(ref[r]-maximum[0]);
                local[0]+=p; local[1]+=exp(got[r]-maximum[1]);
                local[2]+=p*((double)ref[r]-got[r]);
                if (first+r==ids[t+1]) { local[3]=ref[r]; local[4]=got[r]; }
            }
            MPI_Allreduce(local,sum,5,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            double logzp=maximum[0]+log(sum[0]), logzq=maximum[1]+log(sum[1]);
            mean_kl += sum[2]/sum[0]+logzq-logzp;
            nll += logzq-sum[4]-(logzp-sum[3]);
        }
    }
    mean_kl=evaluated ? mean_kl/evaluated : NAN;
    nll=evaluated ? nll/evaluated : NAN;
    int failed=rc || matches < 254 || !isfinite(mean_kl) || !isfinite(nll) ||
               mean_kl>1e-4 || fabs(nll)>.01, any;
    MPI_Allreduce(&failed,&any,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if (!rank) printf("GLM53F_PREFILL_QUALIFY positions=%d steps=%d evaluated=%d top1=%d/%d mean_kl=%.9g nll_delta=%.9g %s\n",
        positions,STEPS,evaluated,matches,STEPS,mean_kl,nll,any?"FAIL":"PASS");
    free(reference);
    return any ? -1 : 0;
}

int main(int argc, char **argv) {
    int rank, ranks;
    int positions = 512, chunk = 5, first_option = 4, chunk_given = 0;
    if (argc > first_option && strncmp(argv[first_option], "--", 2))
        positions = positive_int(argv[first_option++], 524288);
    if (argc > first_option && strncmp(argv[first_option], "--", 2)) {
        chunk = positive_int(argv[first_option++], GLM53F_PREFILL_MAX_TOKENS);
        chunk_given = 1;
    }
    int repeats = 1;
    glm53f_prefill_config config = {GLM53F_PREFILL_LEGACY, 32, GLM53F_PREFILL_FAST_DEFAULT, NULL, 0};
    const char *input_path = NULL;
    const char *trace_path = NULL;
    const char *qualify_prefix = NULL;
    int trace_compare = 0, bad_option = 0, sweep = 0, chunk_sweep = 0, scalar = 0, recipe_sweep = 0;
    int weight_format = -1; /* Omitted preserves GLM53F_PREFILL_INT8 behavior. */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    setvbuf(stdout, NULL, _IOLBF, 0);
    for (int a = first_option; a < argc; ++a) {
        int parsed = glm53f_prefill_option(&config, argc, argv, &a);
        if (parsed < 0) { bad_option = 1; break; }
        if (parsed > 0) continue;
        if (!strcmp(argv[a], "--input-ids") && a + 1 < argc) input_path = argv[++a];
        else if (!strcmp(argv[a], "--sweep")) sweep = 1;
        else if (!strcmp(argv[a], "--chunk-sweep")) chunk_sweep = 1;
        else if (!strcmp(argv[a], "--recipe-sweep")) recipe_sweep = 1;
        else if (!strcmp(argv[a], "--weight-format") && a + 1 < argc) {
            const char *value = argv[++a];
            if (!strcmp(value, "int8")) weight_format = 1;
            else if (!strcmp(value, "fp8")) weight_format = 0;
            else bad_option = 1;
        }
        else if (!strcmp(argv[a], "--scalar-reference")) scalar = 1;
        else if (!strcmp(argv[a], "--qualify-prefix") && a + 1 < argc)
            qualify_prefix = argv[++a];
        else if (!strcmp(argv[a], "--repeats") && a + 1 < argc)
            repeats = positive_int(argv[++a], 10);
        else if (!strcmp(argv[a], "--state-out") && a + 1 < argc) {
            trace_path = argv[++a]; trace_compare = 0;
        } else if (!strcmp(argv[a], "--state-check") && a + 1 < argc) {
            trace_path = argv[++a]; trace_compare = 1;
        } else if (!strcmp(argv[a], "--state-check-numeric") && a + 1 < argc) {
            trace_path = argv[++a]; trace_compare = 2;
        } else bad_option = 1;
    }
    if (recipe_sweep) config.mode = GLM53F_PREFILL_FAST;
    if (!chunk_given && config.mode != GLM53F_PREFILL_LEGACY) chunk = 256;
    if (argc < 4 || ranks != 12 || positions < 1 || chunk < 1 || repeats < 1 ||
        bad_option || (chunk > GLM53F_PREFILL_V5_TOKENS && config.mode != GLM53F_PREFILL_FAST) ||
        ((trace_path || sweep) && repeats != 1) ||
        (qualify_prefix && (config.mode != GLM53F_PREFILL_FAST || trace_path || sweep || chunk_sweep || scalar || repeats != 1)) ||
        (recipe_sweep && (qualify_prefix || trace_path || sweep || chunk_sweep || scalar)) ||
        (chunk_sweep && (sweep || trace_path || scalar)) || (scalar && sweep)) {
        if (!rank) fprintf(stderr, "usage: %s MODEL ROUTED SHARED [POSITIONS [CHUNK]] "
            "[--input-ids FILE] [--repeats N] [--sweep | --chunk-sweep | --scalar-reference] "
            "[--recipe-sweep] [--weight-format fp8|int8] "
            "[--state-out PREFIX | --state-check PREFIX] "
            "[--state-check-numeric PREFIX | --qualify-prefix PREFIX] "
            "[--prefill-mode v5|fast] [--prefill-slab 4|8|16|32] [--prefill-features MASK] "
            "[--prefill-collective utofu|mpi-rsag|ring|tree-rsag|tree-packed] "
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
    if (config.mode == GLM53F_PREFILL_FAST) collective_tokens = 32;
    if (getenv("GLM53F_UTOFU") &&
        glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"),
                                  collective_tokens * 4096))
        MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_target_model_12n *m = glm53f_target_model_create_12n(
        argv[1], argv[2], argv[3], positions + (qualify_prefix ? 256 : 1));
    if (!m || !input) MPI_Abort(MPI_COMM_WORLD, 2);
    int use_int8 = weight_format >= 0 ? weight_format :
        (getenv("GLM53F_PREFILL_INT8") && atoi(getenv("GLM53F_PREFILL_INT8")));
    if (use_int8 &&
        glm53f_target_model_convert_int8_12n(m))
        MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_target_snapshot_12n *initial = glm53f_target_snapshot_create_12n(m);
    if (glm53f_target_model_configure_prefill_12n(m, &config)) MPI_Abort(MPI_COMM_WORLD, 2);
    if (!initial || glm53f_target_snapshot_save_12n(m, initial)) MPI_Abort(MPI_COMM_WORLD, 2);
    if (qualify_prefix) {
        int rc = qualify(m,initial,input,positions,chunk,&config,qualify_prefix,rank);
        glm53f_target_snapshot_free_12n(initial);
        free(input); glm53f_target_model_free_12n(m);
        glm53f_collective_free_12n(); MPI_Finalize();
        return rc ? 4 : 0;
    }
    /* Untimed state probe: equal token/logit and hidden checksums across chunk
     * sizes validate the recurrent and sparse-cache boundary reached by the
     * prompt-only scheduler without retaining every intermediate hidden. */
    int probe_token = 0;
    float probe_logit = 0.0f;
    float *probe_hidden = malloc(4096 * sizeof(*probe_hidden));
    if (!probe_hidden) MPI_Abort(MPI_COMM_WORLD, 2);
    double rates[100];
    const int sweep_chunks[] = {32, 256, 256, 256, 32, 32, 32, 256, 256, 1};
    const int sweep_modes[] = {0, 0, 1, 2, 4, 8, 20, 64, 95, 32};
    const char *sweep_names[] = {"control32", "control256", "int8_grouped", "mhc", "sparse", "kda", "sparse_index", "router", "combined", "scalar_reference"};
    const int chunks[] = {32, 64, 128, 256};
    const char *recipe_names[] = {"v5", "comm32", "kda32", "fast8", "fast16", "fast32",
                                  "tree32", "packed32", "tree512", "packed512"};
    const int recipe_slab[] = {4,32,32,8,16,32,32,32,32,32};
    const unsigned recipe_features[] = {0,1,3,11,11,11,11,11,11,11};
    const int recipe_algorithm[] = {0,0,0,0,0,0,3,4,3,4};
    const int recipe_chunk[] = {256,256,256,256,256,256,256,256,512,512};
    int passes = recipe_sweep ? 10 * repeats : sweep ? 10 : chunk_sweep ? 4 * repeats : repeats;
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
        int checking = trace_compare ? trace_compare : (sweep && rep > 0);
        if (trace_path && glm53f_target_trace_open_12n(m, trace_path, checking))
            MPI_Abort(MPI_COMM_WORLD, 2);
        if (rep && glm53f_target_snapshot_restore_12n(m, initial)) MPI_Abort(MPI_COMM_WORLD, 2);
        if (recipe_sweep) {
            int r = rep / repeats;
            glm53f_prefill_config cfg = config;
            cfg.mode = r ? GLM53F_PREFILL_FAST : GLM53F_PREFILL_V5;
            cfg.slab_tokens = recipe_slab[r]; cfg.features = recipe_features[r];
            cfg.collective = recipe_algorithm[r]; chunk = recipe_chunk[r];
            if (glm53f_target_model_configure_prefill_12n(m,&cfg)) MPI_Abort(MPI_COMM_WORLD,2);
        }
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
        printf("GLM53F_PREFILL_MEMORY rank=%d repeat=%d min_available_GiB=%.6f\n",
            rank, rep + 1, minimum_kb / 1048576.0);
        if (!rank) {
            rates[rep] = positions / maximum;
            printf("GLM53F_PREFILL_12N positions=%d chunk=%d seconds=%.6f tok_s=%.3f "
                   "repeat=%d source=%s validation_io=%d case=%s min_available_GiB=%.3f\n", positions, chunk, maximum, rates[rep],
                   rep + 1, input_path ? input_path : "synthetic", trace_path != NULL,
                   recipe_sweep ? recipe_names[rep/repeats] : sweep ? sweep_names[rep] : scalar ? "scalar" : "configured", global_minimum_kb / 1048576.0);
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
        for (int group = 0; group < (recipe_sweep ? 10 : chunk_sweep ? 4 : 1); ++group) {
            double *group_rates = rates + group * repeats;
            for (int i = 1; i < repeats; ++i)
                for (int j = i; j > 0 && group_rates[j] < group_rates[j - 1]; --j) {
                    double tmp = group_rates[j]; group_rates[j] = group_rates[j - 1]; group_rates[j - 1] = tmp;
                }
            double median = (group_rates[(repeats - 1) / 2] + group_rates[repeats / 2]) / 2;
            double total_seconds_per_token = 0.0;
            for (int i = 0; i < repeats; ++i) total_seconds_per_token += 1.0 / group_rates[i];
            printf("GLM53F_PREFILL_SUMMARY case=%s chunk=%d repeats=%d median_tok_s=%.3f min_tok_s=%.3f max_tok_s=%.3f avg_tok_s=%.3f\n",
                   recipe_sweep ? recipe_names[group] : "configured",
                   recipe_sweep ? recipe_chunk[group] : chunk_sweep ? chunks[group] : chunk, repeats, median, group_rates[0],
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
