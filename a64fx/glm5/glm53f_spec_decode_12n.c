/* Stateful greedy MTP draft / full-target verify / rollback loop. */
#include "glm53f_clock.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glm53f_mtp_12n.h"
#include "glm53f_collective_12n.h"
#include "glm53f_target_model_12n.h"
#include <sys/syscall.h>
#include <unistd.h>

enum { HIDDEN = 4096, MAX_DRAFT = 4 };

static long spec_guard(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    char line[256]; long kb = -1, minimum;
    if (f) { while (fgets(line, sizeof(line), f)) if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break; fclose(f); }
    MPI_Allreduce(&kb, &minimum, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
    if (minimum < 2L * 1024 * 1024) MPI_Abort(MPI_COMM_WORLD, 3);
    return minimum;
}

static void *a256(size_t n) {
    void *p = NULL;
    return posix_memalign(&p, 256, n) ? NULL : p;
}

/* Read on rank zero and broadcast so every rank follows the same sequence. */
static int *read_prompt(const char *path, int *count) {
    FILE *f = fopen(path, "r");
    int *ids = NULL, n = 0, cap = 0, id;
    if (!f) return NULL;
    while (fscanf(f, "%d", &id) == 1) {
        if (id < 0 || id >= 154880 || n >= 262144) goto fail;
        if (n == cap) {
            cap = cap ? 2 * cap : 4096;
            int *p = realloc(ids, (size_t)cap * sizeof(*ids));
            if (!p) goto fail;
            ids = p;
        }
        ids[n++] = id;
    }
    if (!feof(f) || !n) goto fail;
    fclose(f); *count = n; return ids;
fail:
    free(ids); fclose(f); return NULL;
}

int main(int argc, char **argv) {
    int rank, ranks, token, cycles, ndraft, warmup, capacity;
    int repetitions = 1, ignore_eos = 0, cli_sweep = 0;
    int draft[MAX_DRAFT], target[MAX_DRAFT + 1], verify_input[MAX_DRAFT + 1];
    float verify_logit[MAX_DRAFT + 1];
    double phase[4] = {0.0, 0.0, 0.0, 0.0};
    long accepted_total = 0, proposed_total = 0, delivered = 0;
    glm53f_target_model_12n *target_model;
    glm53f_mtp_context_12n *mtp;
    glm53f_target_snapshot_12n *snapshot[MAX_DRAFT + 1] = {0};
    float *target_hidden, *verify_hidden, *draft_hidden[2], target_logit, draft_logit;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (!getenv("GLM53F_NUMA_INTERLEAVE") || atoi(getenv("GLM53F_NUMA_INTERLEAVE"))) {
        unsigned long mask = 0xF0UL; /* interleave weights over the four compute CMGs (see glm53f_target_decode_12n.c) */
        syscall(SYS_set_mempolicy, 3L, &mask, 8UL);
    }
    if (argc < 7 || ranks != 12) {
        if (!rank) fprintf(stderr,"usage: %s MODEL TARGET_ROUTED TARGET_SHARED MTP_ROUTED MTP_SHARED [token=1] [cycles=1] [drafts=1] [warmup=128]\n",argv[0]);
        MPI_Abort(MPI_COMM_WORLD,2);
    }
    token=argc>6?atoi(argv[6]):1;cycles=argc>7?atoi(argv[7]):1;
    ndraft=argc>8?atoi(argv[8]):1;
    warmup=argc>9?atoi(argv[9]):128;
    for (int i = 10; i < argc; ++i) {
        if (!strcmp(argv[i], "--repetitions")) {
            if (++i == argc) MPI_Abort(MPI_COMM_WORLD, 2);
            char *end; long n = strtol(argv[i], &end, 10);
            if (!*argv[i] || *end || n < 1 || n > 10) MPI_Abort(MPI_COMM_WORLD, 2);
            repetitions = (int)n;
        } else if (!strcmp(argv[i], "--ignore-eos")) ignore_eos = 1;
        else if (!strcmp(argv[i], "--draft-sweep")) cli_sweep = 1;
        else if (glm53f_runtime_option(argc, argv, &i) != 1) MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (getenv("GLM53F_DECODE_EXECUTOR") && atoi(getenv("GLM53F_DECODE_EXECUTOR"))) {
        if (!rank) fprintf(stderr, "persistent executor is supported by the greedy/lookup runner; MTP currently uses its legacy executor\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    int *prompt = NULL, prompt_count = 0;
    int *reference = NULL, reference_count = 0;
    FILE *output = NULL;
    const char *prompt_path = getenv("GLM53F_SPEC_PROMPT_IDS");
    const char *output_path = getenv("GLM53F_SPEC_OUTPUT_IDS");
    const char *reference_path = getenv("GLM53F_SPEC_REFERENCE_IDS");
    int self_reference = getenv("GLM53F_SPEC_SELF_REFERENCE") &&
                         atoi(getenv("GLM53F_SPEC_SELF_REFERENCE"));
    if (self_reference && reference_path && *reference_path) MPI_Abort(MPI_COMM_WORLD, 2);
    if (!rank && reference_path && *reference_path &&
        !(reference = read_prompt(reference_path, &reference_count)))
        MPI_Abort(MPI_COMM_WORLD, 2);
    int full_replay = getenv("GLM53F_SPEC_FULL_REPLAY") &&
                      atoi(getenv("GLM53F_SPEC_FULL_REPLAY"));
    if (prompt_path && *prompt_path) {
        if (!rank && !(prompt = read_prompt(prompt_path, &prompt_count)))
            MPI_Abort(MPI_COMM_WORLD, 2);
        MPI_Bcast(&prompt_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank) prompt = malloc((size_t)prompt_count * sizeof(*prompt));
        if (!prompt) MPI_Abort(MPI_COMM_WORLD, 2);
        MPI_Bcast(prompt, prompt_count, MPI_INT, 0, MPI_COMM_WORLD);
        warmup = prompt_count - 1;
        token = prompt[0];
    }
    if(token<0||token>=154880||cycles<1||cycles>32768||ndraft<1||ndraft>MAX_DRAFT||warmup<0||warmup>262144)MPI_Abort(MPI_COMM_WORLD,2);
    if(getenv("GLM53F_UTOFU")){const char*topo=getenv("TOFU_TOPO_PATH");if(!topo)topo="../utofu-tests/tofu_topo.txt";if(glm53f_collective_init_12n(topo,5*HIDDEN))MPI_Abort(MPI_COMM_WORLD,2);}
    capacity=warmup+cycles*(ndraft+2)+1;
    target_model=glm53f_target_model_create_12n(argv[1],argv[2],argv[3],capacity);
    mtp=glm53f_mtp_create_12n(argv[1],argv[4],argv[5],capacity);
    target_hidden=a256(HIDDEN*4);verify_hidden=a256((MAX_DRAFT+1)*HIDDEN*4);draft_hidden[0]=a256(HIDDEN*4);draft_hidden[1]=a256(HIDDEN*4);
    if(!target_model||!mtp||!target_hidden||!verify_hidden||!draft_hidden[0]||!draft_hidden[1])MPI_Abort(MPI_COMM_WORLD,2);
    long minimum_available_kb = spec_guard();
    for(int i=0;i<ndraft+1;i++){snapshot[i]=glm53f_target_snapshot_create_12n(target_model);if(!snapshot[i])MPI_Abort(MPI_COMM_WORLD,2);}
    for(int i=0;i<warmup;i++){
        int next,ignored;float next_logit,ignored_logit;
        if(glm53f_target_model_step_12n(target_model,token,&next,&next_logit,target_hidden))MPI_Abort(MPI_COMM_WORLD,2);
        if (prompt) next = prompt[i + 1];
        int rc = full_replay ? glm53f_mtp_forward_12n(mtp,next,target_hidden,
                     &ignored,&ignored_logit,draft_hidden[0]) :
                     glm53f_mtp_cache_append_12n(mtp,next,target_hidden);
        if (rc) MPI_Abort(MPI_COMM_WORLD,2);
        token=next;
        if (!((i + 1) % 512)) {
            long available = spec_guard();
            if (available < minimum_available_kb) minimum_available_kb = available;
        }
        if (!rank && (i + 1) % 512 == 0) {
            printf("GLM53F_SPEC_PREFILL tokens=%d/%d\n", i + 1, warmup);
            fflush(stdout);
        }
    }
    if(!rank)printf("GLM53F_SPEC_WARMUP tokens=%d next_token=%d target_cache=%d mtp_cache=%d\n",warmup,token,warmup,glm53f_mtp_length_12n(mtp));
    int max_draft = ndraft, warm_token = token, all_pass = 1;
    int *output_buffer = !rank && repetitions > 1 ?
        malloc((size_t)cycles * (max_draft + 2) * sizeof(int)) : NULL;
    if (!rank && repetitions > 1 && !output_buffer) MPI_Abort(MPI_COMM_WORLD, 2);
    int sweep = cli_sweep || (getenv("GLM53F_SPEC_DRAFT_SWEEP") && atoi(getenv("GLM53F_SPEC_DRAFT_SWEEP")));
    int compare_batch = getenv("GLM53F_SPEC_COMPARE_BATCH") && atoi(getenv("GLM53F_SPEC_COMPARE_BATCH"));
    if (sweep && compare_batch) MPI_Abort(MPI_COMM_WORLD, 2);
    int variants = compare_batch ? 2 : sweep ? max_draft : 1;
    int trials = repetitions + (repetitions > 1);
    int runs = variants * trials;
    glm53f_target_snapshot_12n *initial = NULL;
    if (runs > 1 || self_reference) {
        initial = glm53f_target_snapshot_create_12n(target_model);
        if (!initial || glm53f_target_snapshot_save_12n(target_model, initial))
            MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (self_reference) {
        int limit = cycles * (max_draft + 2), input = warm_token;
        if (!rank) {
            reference = malloc((size_t)limit * sizeof(*reference));
            if (!reference) MPI_Abort(MPI_COMM_WORLD, 2);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double start = glm53f_clock();
        for (int i = 0; i < limit; i++) {
            int next; float logit;
            if (glm53f_target_model_step_12n(target_model, input, &next, &logit, NULL))
                MPI_Abort(MPI_COMM_WORLD, 2);
            if (!rank) reference[i] = next;
            reference_count++;
            input = next;
            if (!rank && reference_count % 512 == 0) {
                printf("GLM53F_SPEC_GREEDY_PROGRESS tokens=%d\n", reference_count);
                fflush(stdout);
            }
            if (prompt && !ignore_eos && (next == 154820 || next == 154827 || next == 154829)) break;
        }
        double sec = glm53f_clock() - start, maximum;
        MPI_Reduce(&sec, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (!rank) {
            printf("GLM53F_SPEC_GREEDY tokens=%d tok_s=%.3f\n",
                   reference_count, reference_count / maximum);
            if (output_path && *output_path) {
                char path[4096];
                int n = snprintf(path, sizeof(path), "%s.greedy", output_path);
                if (n < 0 || (size_t)n >= sizeof(path)) MPI_Abort(MPI_COMM_WORLD, 2);
                FILE *f = fopen(path, "wx");
                if (!f) MPI_Abort(MPI_COMM_WORLD, 2);
                for (int i = 0; i < reference_count; i++) fprintf(f, "%d\n", reference[i]);
                if (fclose(f)) MPI_Abort(MPI_COMM_WORLD, 2);
            }
        }
        if (glm53f_target_snapshot_restore_12n(target_model, initial))
            MPI_Abort(MPI_COMM_WORLD, 2);
    }
    for (int run = 0; run < runs; run++) {
        int variant = run / trials;
        int trial = run % trials - (repetitions > 1);
        if (compare_batch) {
            setenv("GLM53F_KDA_BATCH_TEAM", variant ? "1" : "0", 1);
            setenv("GLM53F_SPARSE_BATCH_OP", variant ? "1" : "0", 1);
            setenv("GLM53F_Q4_BATCH_SHARED", variant ? "1" : "0", 1);
            full_replay = !variant;
            if (!rank) printf("GLM53F_SPEC_VARIANT %s\n", variant ? "optimized" : "baseline");
        }
        ndraft = sweep ? variant + 1 : max_draft;
        if (run && (glm53f_target_snapshot_restore_12n(target_model, initial) ||
                    glm53f_mtp_restore_length_12n(mtp, warmup))) MPI_Abort(MPI_COMM_WORLD, 2);
        token = warm_token;
        accepted_total = proposed_total = delivered = 0;
        memset(phase, 0, sizeof(phase));
        int executed = 0, reference_ok = 1, eos = 0;
        long checked = 0;
        if (!rank && output_path && *output_path) {
            char path[4096];
            int n = repetitions > 1 ? snprintf(path, sizeof(path), "%s.v%d.d%d.trial%d", output_path, variant, ndraft, trial) :
                    compare_batch ? snprintf(path, sizeof(path), "%s.%s", output_path,
                                             variant ? "optimized" : "baseline") :
                    sweep ? snprintf(path, sizeof(path), "%s.d%d", output_path, ndraft) :
                            snprintf(path, sizeof(path), "%s", output_path);
            if (n < 0 || (size_t)n >= sizeof(path) || !(output = fopen(path, "wx")))
                MPI_Abort(MPI_COMM_WORLD, 2);
        }
        glm53f_target_profile_reset_12n(target_model);
        MPI_Barrier(MPI_COMM_WORLD);double begin=glm53f_clock();
        for(int cycle=0;cycle<cycles;cycle++){
            int first_token;
            double phase_begin=glm53f_clock();
            if(glm53f_target_model_step_12n(target_model,token,&first_token,&target_logit,target_hidden))MPI_Abort(MPI_COMM_WORLD,2);
            phase[0]+=glm53f_clock()-phase_begin;phase_begin=glm53f_clock();
            int mtp_base=glm53f_mtp_length_12n(mtp),input=first_token;
            const float*hidden=target_hidden;
            for(int j=0;j<ndraft;j++){
                float*out_hidden=draft_hidden[j&1];
                if(glm53f_mtp_forward_12n(mtp,input,hidden,&draft[j],&draft_logit,out_hidden))MPI_Abort(MPI_COMM_WORLD,2);
                input=draft[j];hidden=out_hidden;
            }
            phase[1]+=glm53f_clock()-phase_begin;phase_begin=glm53f_clock();
            proposed_total+=ndraft;
            verify_input[0]=first_token;
            for(int j=0;j<ndraft;j++)verify_input[j+1]=draft[j];
            if(glm53f_target_model_step_batch_12n(target_model,verify_input,ndraft+1,
                    target,verify_logit,verify_hidden,snapshot))MPI_Abort(MPI_COMM_WORLD,2);
            int accepted=0,committed=ndraft+1,next_token=target[ndraft];
            for(int j=0;j<ndraft;j++)if(target[j]==draft[j])accepted++;else{committed=j+1;next_token=target[j];break;}
            if(glm53f_target_snapshot_restore_12n(target_model,snapshot[committed-1]))MPI_Abort(MPI_COMM_WORLD,2);
            phase[2]+=glm53f_clock()-phase_begin;phase_begin=glm53f_clock();
            /* Rebuild the committed MTP suffix from target hidden states. Draft
             * hidden states are approximate, and retaining them after rejection
             * leaves the cache one position behind the target sequence. */
            /* The first draft call already used first_token with the exact scalar
             * target hidden state, so retain that cache entry and replay only the
             * suffix whose draft hidden states were approximate. */
            if(glm53f_mtp_restore_length_12n(mtp,mtp_base+1))MPI_Abort(MPI_COMM_WORLD,2);
            int ignored;float ignored_logit;float*replay_hidden=draft_hidden[0];
            for(int j=0;j<committed;j++) {
                int rc = full_replay ? glm53f_mtp_forward_12n(mtp,target[j],
                    verify_hidden+(size_t)j*HIDDEN,&ignored,&ignored_logit,
                    replay_hidden) : glm53f_mtp_cache_append_12n(mtp,target[j],
                    verify_hidden+(size_t)j*HIDDEN);
                if (rc) MPI_Abort(MPI_COMM_WORLD,2);
            }
            phase[3]+=glm53f_clock()-phase_begin;
            int mtp_commit=committed+1;
            accepted_total+=accepted;token=next_token;
            int emitted[MAX_DRAFT + 2], count = accepted + 2;
            emitted[0] = first_token;
            for (int j = 0; j < accepted; j++) emitted[j + 1] = draft[j];
            emitted[count - 1] = next_token;
            for (int j = 0; j < count; j++) {
                int id = emitted[j];
                if (!rank && output_buffer) output_buffer[delivered] = id;
                else if (output) fprintf(output, "%d\n", id);
                if (!rank && reference && delivered < reference_count) {
                    if (reference[delivered] != id && reference_ok) {
                        fprintf(stderr, "GLM53F_SPEC_MISMATCH position=%ld got=%d expected=%d\n",
                                delivered, id, reference[delivered]);
                        reference_ok = 0;
                    }
                    checked++;
                }
                delivered++;
                if (prompt && !ignore_eos && (id == 154820 || id == 154827 || id == 154829)) {
                    eos = 1; token = id; break;
                }
            }
            if (output && !output_buffer) fflush(output);
            executed++;
            if (!(executed % 32)) {
                long available = spec_guard();
                if (available < minimum_available_kb) minimum_available_kb = available;
            }
            if(!rank && repetitions == 1)printf("GLM53F_SPEC_CYCLE cycle=%d first=%d accepted=%d/%d fallback=%d target_steps=%d mtp_steps=%d\n",cycle,first_token,accepted,ndraft,next_token,committed,mtp_commit);
            if (eos) break;
        }
        double sec=glm53f_clock()-begin,max_sec;MPI_Reduce(&sec,&max_sec,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        if (output && output_buffer)
            for (long j = 0; j < delivered; ++j) if (fprintf(output, "%d\n", output_buffer[j]) < 0) MPI_Abort(MPI_COMM_WORLD, 2);
        double max_phase[4];MPI_Reduce(phase,max_phase,4,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        if(!rank)printf("GLM53F_SPEC_PHASE ms_cycle target=%.3f draft=%.3f verify=%.3f rebase=%.3f\n",max_phase[0]*1e3/executed,max_phase[1]*1e3/executed,max_phase[2]*1e3/executed,max_phase[3]*1e3/executed);
        double alpha=proposed_total?(double)accepted_total/proposed_total:0.0;
        MPI_Bcast(&reference_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int pass=reference_ok;const char*gate;
        gate=getenv("GLM53F_SPEC_MIN_ALPHA");if(gate&&*gate&&alpha<strtod(gate,NULL))pass=0;
        gate=getenv("GLM53F_SPEC_EXPECT_ACCEPTED");if(gate&&*gate&&accepted_total!=strtol(gate,NULL,10))pass=0;
        gate=getenv("GLM53F_SPEC_EXPECT_FINAL");if(gate&&*gate&&token!=strtol(gate,NULL,10))pass=0;
        if(!rank)printf("GLM53F_SPEC_DECODE_12N cycles=%d drafts=%d accepted=%ld/%ld alpha=%.6f delivered=%ld tok_s=%.3f final_token=%d %s\n",executed,ndraft,accepted_total,proposed_total,alpha,delivered,delivered/max_sec,token,pass?"PASS":"FAIL");
        if (!rank) printf("GLM53F_SPEC_TRIAL {\"variant\":%d,\"trial\":%d,\"warmup\":%s,\"drafts\":%d,\"delivered\":%ld,\"seconds\":%.9f,\"tok_s\":%.6f,\"accepted\":%ld,\"proposed\":%ld,\"reference_checked\":%ld,\"minimum_available_kb\":%ld,\"status\":\"%s\"}\n",
            variant, trial, trial < 0 ? "true" : "false", ndraft, delivered, max_sec, delivered / max_sec,
            accepted_total, proposed_total, checked, minimum_available_kb, pass ? "PASS" : "FAIL");
        if (!rank && reference) printf("GLM53F_SPEC_REFERENCE checked=%ld delivered=%ld %s\n",
                                       checked, delivered, reference_ok ? "PASS" : "FAIL");
        glm53f_target_profile_report_12n(target_model,"spec");
        if (output && fclose(output)) MPI_Abort(MPI_COMM_WORLD, 2);
        output = NULL;
        all_pass &= pass;
    }
    glm53f_target_snapshot_free_12n(initial);
    free(output_buffer);
    free(reference);
    free(prompt);
    for(int i=0;i<max_draft+1;i++)glm53f_target_snapshot_free_12n(snapshot[i]);
    free(draft_hidden[1]);free(draft_hidden[0]);free(target_hidden);free(verify_hidden);glm53f_mtp_free_12n(mtp);glm53f_target_model_free_12n(target_model);glm53f_collective_free_12n();
    if (!rank) printf("GLM53F_SPEC_COMPLETE {\"variants\":%d,\"repetitions\":%d,\"cycles\":%d,\"status\":\"%s\"}\n",
        variants, repetitions, cycles, all_pass ? "PASS" : "FAIL");
    MPI_Finalize();return all_pass?0:1;
}
