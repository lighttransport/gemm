/* Stateful greedy MTP draft / full-target verify / rollback loop. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "glm53f_mtp_12n.h"
#include "glm53f_collective_12n.h"
#include "glm53f_target_model_12n.h"

enum { HIDDEN = 4096, MAX_DRAFT = 4 };

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
        if (id < 0 || id >= 154880) goto fail;
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

/* Manifest paths cannot contain whitespace. Parse identically on every rank. */
typedef struct { char id[128], prompt[2048], output[2048], policy[32]; int limit; } request_entry;
static request_entry *read_manifest(const char *path, int *count, int *max_prompt) {
    FILE *f=fopen(path,"r"); request_entry *entries=NULL; int n=0;
    char line[4352],extra;
    if(!f){perror(path);return NULL;}
    while(fgets(line,sizeof(line),f)){
        request_entry e;
        memset(&e,0,sizeof(e));
        char limit[32]={0};
        int fields=sscanf(line,"%127s %2047s %2047s %31s %31s %c",e.id,e.prompt,e.output,e.policy,limit,&extra);
        if(fields<3||fields>5){fprintf(stderr,"manifest line %d has %d fields\n",n+1,fields);goto fail;}
        if(fields==5){
            char *end;long value=strtol(limit,&end,10);
            if(*end||value<1||value>INT_MAX)goto fail;
            e.limit=(int)value;
        }
        int np=0;int *ids=read_prompt(e.prompt,&np);
        if(!ids){fprintf(stderr,"manifest cannot read prompt %s\n",e.prompt);goto fail;}
        free(ids);
        if(np>*max_prompt)*max_prompt=np;
        request_entry *next=realloc(entries,(size_t)(n+1)*sizeof(*entries));
        if(!next)goto fail;
        entries=next;entries[n++]=e;
    }
    if(ferror(f)||!n)goto fail;
    fclose(f);*count=n;return entries;
fail:
    free(entries);fclose(f);return NULL;
}

int main(int argc, char **argv) {
    int rank, ranks, token, cycles, ndraft, warmup, capacity;
    int draft[MAX_DRAFT], target[MAX_DRAFT + 1], verify_input[MAX_DRAFT + 1];
    float verify_logit[MAX_DRAFT + 1];
    double phase[4] = {0.0, 0.0, 0.0, 0.0};
    long accepted_total = 0, proposed_total = 0, delivered = 0;
    glm53f_target_model_12n *target_model=NULL;
    glm53f_mtp_context_12n *mtp=NULL;
    glm53f_target_snapshot_12n *snapshot[MAX_DRAFT + 1] = {0};
    float *target_hidden, *cycle_hidden, *verify_hidden, *draft_hidden[2], target_logit, draft_logit;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 7 || ranks != 12) {
        if (!rank) fprintf(stderr,"usage: %s MODEL TARGET_ROUTED TARGET_SHARED MTP_ROUTED MTP_SHARED [token=1] [cycles=1] [drafts=1] [warmup=128] [max_tokens=0]\n",argv[0]);
        MPI_Abort(MPI_COMM_WORLD,2);
    }
    token=argc>6?atoi(argv[6]):1;cycles=argc>7?atoi(argv[7]):1;
    ndraft=argc>8?atoi(argv[8]):1;
    warmup=argc>9?atoi(argv[9]):128;
    long max_tokens = 0;
    if (argc > 10) {
        char *end;
        max_tokens = strtol(argv[10], &end, 10);
        if (!*argv[10] || *end || max_tokens < 0 || max_tokens > INT_MAX)
            MPI_Abort(MPI_COMM_WORLD, 2);
    }
    const char *manifest_path=NULL;
    glm53f_decode_options options=glm53f_decode_options_default();
    for(int i=11;i<argc;i+=2){
        if(i+1==argc)MPI_Abort(MPI_COMM_WORLD,2);
        if(!strcmp(argv[i],"--prompt-list")){manifest_path=argv[i+1];continue;}
        if(!strcmp(argv[i],"--kda-proj")){
            if(strcmp(argv[i+1],"bf16")&&strcmp(argv[i+1],"int8"))MPI_Abort(MPI_COMM_WORLD,2);
            options.kda_int8=!strcmp(argv[i+1],"int8");continue;
        }
        char *end;long value=strtol(argv[i+1],&end,10);
        if(!*argv[i+1]||*end||value<0||value>INT_MAX)MPI_Abort(MPI_COMM_WORLD,2);
        if(!strcmp(argv[i],"--expert-top-k"))options.expert_top_k=(int)value;
        else if(!strcmp(argv[i],"--sparse-top-k"))options.sparse_top_k=(int)value;
        else if(!strcmp(argv[i],"--index-refresh"))options.index_refresh=(int)value;
        else MPI_Abort(MPI_COMM_WORLD,2);
    }
    /* Environment fallback keeps batch launchers from losing option pairs
     * when an MPI wrapper rewrites argv.  CLI values remain authoritative when
     * supplied; these names are intentionally scoped to this runner. */
    const char *env_opt = getenv("GLM53F_EXPERT_TOP_K");
    if (env_opt && *env_opt) {
        char *end; long value = strtol(env_opt, &end, 10);
        if (*end || value < 1 || value > NEXPERTS) MPI_Abort(MPI_COMM_WORLD, 2);
        options.expert_top_k = (int)value;
    }
    env_opt = getenv("GLM53F_SPARSE_TOP_K");
    if (env_opt && *env_opt) {
        char *end; long value = strtol(env_opt, &end, 10);
        if (*end || value < 1 || value > 2048) MPI_Abort(MPI_COMM_WORLD, 2);
        options.sparse_top_k = (int)value;
    }
    env_opt = getenv("GLM53F_INDEX_REFRESH");
    if (env_opt && *env_opt) {
        char *end; long value = strtol(env_opt, &end, 10);
        if (*end || value < 1 || value > 1024) MPI_Abort(MPI_COMM_WORLD, 2);
        options.index_refresh = (int)value;
    }
    env_opt = getenv("GLM53F_KDA_PROJ");
    if (env_opt && *env_opt) {
        if (strcmp(env_opt, "bf16") && strcmp(env_opt, "int8"))
            MPI_Abort(MPI_COMM_WORLD, 2);
        options.kda_int8 = !strcmp(env_opt, "int8");
    }
    if(!glm53f_decode_options_valid(&options))MPI_Abort(MPI_COMM_WORLD,2);
    int request_count=1,max_prompt=0,overall_pass=1;
    request_entry *requests=NULL;
    if(manifest_path){
        requests=read_manifest(manifest_path,&request_count,&max_prompt);
        if(!requests){fprintf(stderr,"invalid manifest %s\n",manifest_path);MPI_Abort(MPI_COMM_WORLD,2);}
        if(getenv("GLM53F_SPEC_REFERENCE_IDS")){fprintf(stderr,"manifest disallows external reference\n");MPI_Abort(MPI_COMM_WORLD,2);}
    }
    const int requested_draft=ndraft;
    const long requested_max_tokens=max_tokens;
    const glm53f_decode_options requested_options=options;
    glm53f_target_snapshot_12n *prefill_snapshot=NULL;
    glm53f_target_snapshot_12n *cycle_snapshot=NULL;
    char cached_prompt[2048]={0};
    for(int request=0;request<request_count;request++){
    ndraft=requested_draft;
    max_tokens=requests&&requests[request].limit?requests[request].limit:requested_max_tokens;
    options=requested_options;
    if(requests&&*requests[request].policy){
        const char *p=requests[request].policy;
        options=glm53f_decode_options_default();
        if(!strcmp(p,"aggressive")){
            options.expert_top_k=2;options.sparse_top_k=256;
            options.index_refresh=8;options.kda_int8=1;
        }else if(!strcmp(p,"all")||!strcmp(p,"no-int8")||!strcmp(p,"no-refresh")){
            options.expert_top_k=4;options.sparse_top_k=512;
            options.index_refresh=strcmp(p,"no-refresh")?4:1;
            options.kda_int8=strcmp(p,"no-int8")!=0;
        }else if(!strcmp(p,"top4"))options.expert_top_k=4;
        else if(!strcmp(p,"sparse512"))options.sparse_top_k=512;
        else if(!strcmp(p,"refresh"))options.index_refresh=4;
        else if(!strcmp(p,"int8"))options.kda_int8=1;
        else if(strcmp(p,"exact"))MPI_Abort(MPI_COMM_WORLD,2);
    }
    int carry = getenv("GLM53F_SPEC_CARRY") && atoi(getenv("GLM53F_SPEC_CARRY"));
    int defer_verify = getenv("GLM53F_SPEC_DEFER_VERIFY") &&
                       atoi(getenv("GLM53F_SPEC_DEFER_VERIFY"));
    int lazy_snapshot = getenv("GLM53F_SPEC_LAZY_SNAPSHOT") &&
                        atoi(getenv("GLM53F_SPEC_LAZY_SNAPSHOT"));
    int *prompt = NULL, prompt_count = 0;
    int *reference = NULL, reference_count = 0;
    FILE *output = NULL;
    const char *prompt_path = getenv("GLM53F_SPEC_PROMPT_IDS");
    const char *output_path = getenv("GLM53F_SPEC_OUTPUT_IDS");
    const char *reference_path = getenv("GLM53F_SPEC_REFERENCE_IDS");
    if(requests){
        prompt_path=requests[request].prompt;output_path=requests[request].output;
        if(!rank){printf("GLM53F_REQUEST index=%d id=%s\n",request,requests[request].id);fflush(stdout);}
    }
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
    if(token<0||token>=154880||cycles<1||ndraft<0||ndraft>MAX_DRAFT||warmup<0)MPI_Abort(MPI_COMM_WORLD,2);
    if ((long long)warmup + (long long)cycles * (ndraft + 2) + 1 > INT_MAX)
        MPI_Abort(MPI_COMM_WORLD, 2);
    if(!request&&getenv("GLM53F_UTOFU")){const char*topo=getenv("TOFU_TOPO_PATH");if(!topo)topo="../utofu-tests/tofu_topo.txt";if(glm53f_collective_init_12n(topo,5*HIDDEN))MPI_Abort(MPI_COMM_WORLD,2);}
    long long needed=(requests?max_prompt-1:warmup)+(long long)cycles*(ndraft+2)+1;
    if(needed>INT_MAX)MPI_Abort(MPI_COMM_WORLD,2);
    capacity=(int)needed;
    int reuse=prompt_path&&prefill_snapshot&&!strcmp(prompt_path,cached_prompt);
    if(!target_model){
    target_model=glm53f_target_model_create_12n(argv[1],argv[2],argv[3],capacity);
    mtp=glm53f_mtp_create_12n(argv[1],argv[4],argv[5],capacity);
    }else if(!reuse){
        glm53f_target_model_reset_12n(target_model);
        if(glm53f_mtp_restore_length_12n(mtp,0))MPI_Abort(MPI_COMM_WORLD,2);
    }
    target_hidden=a256(HIDDEN*4);cycle_hidden=a256(HIDDEN*4);verify_hidden=a256((MAX_DRAFT+1)*HIDDEN*4);draft_hidden[0]=a256(HIDDEN*4);draft_hidden[1]=a256(HIDDEN*4);
    if(!target_model||!mtp||!target_hidden||!cycle_hidden||!verify_hidden||!draft_hidden[0]||!draft_hidden[1])MPI_Abort(MPI_COMM_WORLD,2);
    /* A full-acceptance verifier never restores its final input state. Keep
     * snapshots only for the ndraft possible rejection points; the final
     * verifier position is passed as a NULL destination. */
    /* Deferred one-draft verification advances exactly one scalar target
     * position and never restores a rejected suffix, so it does not need a
     * KDA snapshot.  Keep snapshots for the general batched verifier only;
     * this avoids a second multi-gigabyte allocation on the Q4 route. */
    if (lazy_snapshot && ndraft > 1) {
        cycle_snapshot = glm53f_target_snapshot_create_12n(target_model);
        if (!cycle_snapshot) MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (!(defer_verify && !carry && ndraft == 1) &&
        !(lazy_snapshot && ndraft > 1))
        for(int i=0;i<ndraft;i++){snapshot[i]=glm53f_target_snapshot_create_12n(target_model);if(!snapshot[i])MPI_Abort(MPI_COMM_WORLD,2);}
    if(reuse){
        if(glm53f_target_snapshot_restore_12n(target_model,prefill_snapshot)||glm53f_mtp_restore_length_12n(mtp,warmup))MPI_Abort(MPI_COMM_WORLD,2);
        token=prompt[warmup];
        if(!rank)printf("GLM53F_PREFILL_REUSE tokens=%d\n",warmup);
    }
    for(int i=0;i<(reuse?0:warmup);i++){
        int next,ignored;float next_logit,ignored_logit;
        if(glm53f_target_model_step_12n(target_model,token,&next,&next_logit,target_hidden))MPI_Abort(MPI_COMM_WORLD,2);
        if (prompt) next = prompt[i + 1];
        int rc = full_replay ? glm53f_mtp_forward_12n(mtp,next,target_hidden,
                     &ignored,&ignored_logit,draft_hidden[0]) :
                     glm53f_mtp_cache_append_12n(mtp,next,target_hidden);
        if (rc) MPI_Abort(MPI_COMM_WORLD,2);
        token=next;
        if (!rank && (i + 1) % 512 == 0) {
            printf("GLM53F_SPEC_PREFILL tokens=%d/%d\n", i + 1, warmup);
            fflush(stdout);
        }
    }
    if(requests&&!reuse){
        if(!prefill_snapshot)prefill_snapshot=glm53f_target_snapshot_create_12n(target_model);
        if(!prefill_snapshot||glm53f_target_snapshot_save_12n(target_model,prefill_snapshot))MPI_Abort(MPI_COMM_WORLD,2);
        if(strlen(prompt_path)>=sizeof(cached_prompt))MPI_Abort(MPI_COMM_WORLD,2);
        strcpy(cached_prompt,prompt_path);
    }
    if(glm53f_target_set_decode_options_12n(target_model,&options))MPI_Abort(MPI_COMM_WORLD,2);
    if(!rank)printf("GLM53F_DECODE_POLICY experts=%d sparse=%d refresh=%d kda_int8=%d prefill=exact snapshot=%s\n",options.expert_top_k,options.sparse_top_k,options.index_refresh,options.kda_int8,(lazy_snapshot&&ndraft>1)?"lazy":"per-position");
    if(!rank)printf("GLM53F_SPEC_WARMUP tokens=%d next_token=%d target_cache=%d mtp_cache=%d\n",warmup,token,warmup,glm53f_mtp_length_12n(mtp));
    int max_draft = ndraft, warm_token = token, all_pass = 1;
    int sweep = getenv("GLM53F_SPEC_DRAFT_SWEEP") && atoi(getenv("GLM53F_SPEC_DRAFT_SWEEP"));
    int compare_batch = getenv("GLM53F_SPEC_COMPARE_BATCH") && atoi(getenv("GLM53F_SPEC_COMPARE_BATCH"));
    int opt_sweep=getenv("GLM53F_SPEC_OPT_SWEEP")&&atoi(getenv("GLM53F_SPEC_OPT_SWEEP"));
    int tile_sweep=getenv("GLM53F_SPEC_TILE_SWEEP")&&atoi(getenv("GLM53F_SPEC_TILE_SWEEP"));
    int queue_sweep=getenv("GLM53F_SPEC_QUEUE_SWEEP")?atoi(getenv("GLM53F_SPEC_QUEUE_SWEEP")):0;
    if(queue_sweep<0||queue_sweep>2)MPI_Abort(MPI_COMM_WORLD,2);
    int queue_modes=queue_sweep==2?3:2;
    if((tile_sweep!=0)+(opt_sweep!=0)+(queue_sweep!=0)>1)MPI_Abort(MPI_COMM_WORLD,2);
    if(queue_sweep&&ndraft!=2)MPI_Abort(MPI_COMM_WORLD,2);
    opt_sweep |= tile_sweep | queue_sweep;
    if ((ndraft==0&&sweep)||((sweep!=0)+(compare_batch!=0)+(opt_sweep!=0)>1)) MPI_Abort(MPI_COMM_WORLD, 2);
    const char *old_names[]={"baseline","carry","front","routed","mla","snapshot","kda","op"};
    const char *tile_names[]={"untiled","row","post"};
    const char *queue_names[]={"serial-d1","queue-d1","serial-d2","queue-d2"};
    const char *mhc_queue_names[]={"serial-d1","queue-d1","mhc-queue-d1",
        "serial-d2","queue-d2","mhc-queue-d2"};
    const char **opt_names=queue_sweep==2?mhc_queue_names:queue_sweep?queue_names:tile_sweep?tile_names:old_names;
    const char *opt_flags[]={"GLM53F_SPEC_CARRY","GLM53F_SPARSE_BATCH_FRONT",
        "GLM53F_Q4_BATCH_ROUTED","GLM53F_SPARSE_MLA_REG","GLM53F_SPEC_DIRECT_SNAPSHOT",
        "GLM53F_KDA_FUSED_OP","GLM53F_SPARSE_BATCH_OP"};
    int variants = queue_sweep ? queue_modes*2 : tile_sweep ? 3 : opt_sweep ? 8 : compare_batch ? 2 : sweep ? max_draft : 1;
    int repeats = 1;
    const char *repeat_env = getenv("GLM53F_SPEC_REPEAT");
    if (repeat_env && *repeat_env) {
        char *end;
        long value = strtol(repeat_env, &end, 10);
        if (*end || value < 1 || value > 100) MPI_Abort(MPI_COMM_WORLD, 2);
        repeats = (int)value;
    }
    int runs = variants * repeats;
    glm53f_target_snapshot_12n *initial = NULL;
    if (runs > 1 || self_reference) {
        initial = glm53f_target_snapshot_create_12n(target_model);
        if (!initial || glm53f_target_snapshot_save_12n(target_model, initial))
            MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (self_reference) {
        int limit = cycles * (max_draft + 2), input = warm_token;
        if (max_tokens && max_tokens < limit) limit = (int)max_tokens;
        if (!rank) {
            reference = malloc((size_t)limit * sizeof(*reference));
            if (!reference) MPI_Abort(MPI_COMM_WORLD, 2);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
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
            if (prompt && (next == 154820 || next == 154827 || next == 154829)) break;
        }
        double sec = MPI_Wtime() - start, maximum;
        MPI_Reduce(&sec, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (!rank) {
            printf("GLM53F_SPEC_GREEDY tokens=%d tok_s=%.3f\n",
                   reference_count, reference_count / maximum);
            if (output_path && *output_path) {
                char path[4096];
                int n = snprintf(path, sizeof(path), "%s.greedy", output_path);
                if (n < 0 || (size_t)n >= sizeof(path)) MPI_Abort(MPI_COMM_WORLD, 2);
                FILE *f = fopen(path, "w");
                if (!f) MPI_Abort(MPI_COMM_WORLD, 2);
                for (int i = 0; i < reference_count; i++) fprintf(f, "%d\n", reference[i]);
                if (fclose(f)) MPI_Abort(MPI_COMM_WORLD, 2);
            }
        }
        if (glm53f_target_snapshot_restore_12n(target_model, initial))
            MPI_Abort(MPI_COMM_WORLD, 2);
    }
    for (int trial = 0; trial < runs; trial++) {
        int repetition = trial / variants, run = trial % variants;
        /* Reverse alternate repetitions to expose ordering/warm-cache bias. */
        if (repetition & 1) run = variants - 1 - run;
        if (!rank && repeats > 1)
            printf("GLM53F_SPEC_TRIAL repeat=%d variant=%d\n", repetition + 1, run);
        if(opt_sweep) {
          if(queue_sweep){
            setenv("GLM53F_KDA_PROJECTION_QUEUE",run%queue_modes?"1":"0",1);
            if(queue_sweep==2)setenv("GLM53F_MHC_BATCH_TEAM",run%queue_modes==2?"1":"0",1);
            if(!rank)printf("GLM53F_SPEC_VARIANT %s\n",opt_names[run]);
          }else if(tile_sweep){
            setenv("GLM53F_Q4_ROW_TILE",run>=1?"1":"0",1);
            setenv("GLM53F_MHC_POST_TILES",run>=2?"1":"0",1);
            if(!rank)printf("GLM53F_SPEC_VARIANT %s\n",opt_names[run]);
          }else{
            for(int f=0;f<7;f++)setenv(opt_flags[f],run>f?"1":"0",1);
            carry=run>0;
            setenv("GLM53F_KDA_BATCH_TEAM","1",1);
            setenv("GLM53F_Q4_BATCH_SHARED","1",1);
            full_replay=0;
            if(!rank)printf("GLM53F_SPEC_VARIANT %s\n",opt_names[run]);
          }
        }
        if (compare_batch) {
            setenv("GLM53F_KDA_BATCH_TEAM", run ? "1" : "0", 1);
            setenv("GLM53F_SPARSE_BATCH_OP", run ? "1" : "0", 1);
            setenv("GLM53F_Q4_BATCH_SHARED", run ? "1" : "0", 1);
            full_replay = !run;
            if (!rank) printf("GLM53F_SPEC_VARIANT %s\n", run ? "optimized" : "baseline");
        }
        ndraft = queue_sweep ? run / queue_modes + 1 : sweep ? run + 1 : max_draft;
        if (trial && (glm53f_target_snapshot_restore_12n(target_model, initial) ||
                    glm53f_mtp_restore_length_12n(mtp, warmup))) MPI_Abort(MPI_COMM_WORLD, 2);
        token = warm_token;
        accepted_total = proposed_total = delivered = 0;
        memset(phase, 0, sizeof(phase));
        int executed = 0, reference_ok = 1, eos = 0;
        long checked = 0;
        if (!rank && output_path && *output_path) {
            char path[4096];
            int n = opt_sweep ? snprintf(path,sizeof(path),"%s.%s",output_path,opt_names[run]) :
                    compare_batch ? snprintf(path, sizeof(path), "%s.%s", output_path,
                                             run ? "optimized" : "baseline") :
                    sweep ? snprintf(path, sizeof(path), "%s.d%d", output_path, ndraft) :
                            snprintf(path, sizeof(path), "%s", output_path);
            if (n < 0 || (size_t)n >= sizeof(path)) MPI_Abort(MPI_COMM_WORLD, 2);
            if (repeats > 1) {
                int suffix = snprintf(path + n, sizeof(path) - (size_t)n,
                                      ".r%d", repetition + 1);
                if (suffix < 0 || (size_t)suffix >= sizeof(path) - (size_t)n)
                    MPI_Abort(MPI_COMM_WORLD, 2);
            }
            if (!(output = fopen(path, "w")))
                MPI_Abort(MPI_COMM_WORLD, 2);
        }
        glm53f_target_profile_reset_12n(target_model);
        MPI_Barrier(MPI_COMM_WORLD);double begin=MPI_Wtime();
        for(int cycle=0;cycle<cycles;cycle++){
            int first_token;
            if(ndraft==0){
                double t0=MPI_Wtime();
                if(glm53f_target_model_step_12n(target_model,token,&first_token,&target_logit,target_hidden))MPI_Abort(MPI_COMM_WORLD,2);
                phase[0]+=MPI_Wtime()-t0;token=first_token;
                if(output)fprintf(output,"%d\n",token);
                if(!rank&&reference&&delivered<reference_count){
                    if(reference[delivered]!=token)reference_ok=0;
                    checked++;
                }
                delivered++;executed++;
                eos=prompt&&(token==154820||token==154827||token==154829);
                if(eos||(max_tokens&&delivered>=max_tokens))break;
                continue;
            }
            double phase_begin=MPI_Wtime();
            memcpy(cycle_hidden,target_hidden,HIDDEN*sizeof(float));
            glm53f_sparse_selection_state mtp_selection;
            glm53f_sparse_selection_state mtp_history[8];
            if (glm53f_mtp_save_selection_12n(mtp, &mtp_selection, mtp_history))
                MPI_Abort(MPI_COMM_WORLD, 2);
            if (!carry || cycle == 0) {
                if(glm53f_target_model_step_12n(target_model,token,&first_token,&target_logit,target_hidden))MPI_Abort(MPI_COMM_WORLD,2);
            } else {
                /* The previous verifier produced this pending token and the
                 * hidden state of its predecessor. Neither cache has consumed
                 * the pending token yet; no scalar target step is needed. */
                first_token = token;
            }
            if (lazy_snapshot && ndraft > 1 &&
                glm53f_target_snapshot_save_12n(target_model, cycle_snapshot))
                MPI_Abort(MPI_COMM_WORLD, 2);
            phase[0]+=MPI_Wtime()-phase_begin;phase_begin=MPI_Wtime();
            int mtp_base=glm53f_mtp_length_12n(mtp),input=first_token;
            const float*hidden=target_hidden;
            for(int j=0;j<ndraft;j++){
                float*out_hidden=draft_hidden[j&1];
                if(glm53f_mtp_forward_12n(mtp,input,hidden,&draft[j],&draft_logit,out_hidden))MPI_Abort(MPI_COMM_WORLD,2);
                input=draft[j];hidden=out_hidden;
            }
            phase[1]+=MPI_Wtime()-phase_begin;phase_begin=MPI_Wtime();
            proposed_total+=ndraft;
            verify_input[0]=first_token;
            for(int j=0;j<ndraft;j++)verify_input[j+1]=draft[j];
            int deferred_emit = defer_verify && !carry && ndraft == 1;
            int accepted=0,committed=ndraft+1,next_token=target[ndraft];
            if (deferred_emit) {
                /* One draft only needs the target distribution at the first
                 * verifier position.  If it matches, leave the target model
                 * immediately after first_token and consume draft[0] on the
                 * next cycle.  On rejection, target[0] is already the
                 * committed fallback and the target state is already at the
                 * fallback token's predecessor.  Use the scalar graph here:
                 * a one-position batch pays for the batched scratch/team
                 * setup and is measurably slower, while this path is already
                 * exact and needs only one target position. */
                if (glm53f_target_model_step_12n(target_model,verify_input[0],
                        target,verify_logit,verify_hidden))
                    MPI_Abort(MPI_COMM_WORLD,2);
                accepted = target[0] == draft[0];
                committed = 1;
                next_token = accepted ? draft[0] : target[0];
            } else {
                glm53f_target_snapshot_12n **batch_after =
                    (lazy_snapshot && ndraft > 1) ? NULL : snapshot;
                if(glm53f_target_model_step_batch_12n(target_model,verify_input,ndraft+1,
                        target,verify_logit,verify_hidden,batch_after))MPI_Abort(MPI_COMM_WORLD,2);
                for(int j=0;j<ndraft;j++)if(target[j]==draft[j])accepted++;else{committed=j+1;next_token=target[j];break;}
            }
            int carry_bonus = 0, carry_fallback = -1;
            /* The verifier has already advanced through every committed input
             * on full acceptance. Restore only when a rejected suffix must be
             * discarded; this avoids copying all KDA states on the common path. */
            if (!deferred_emit && accepted != ndraft) {
                if (lazy_snapshot && ndraft > 1) {
                    if (glm53f_target_snapshot_restore_12n(target_model,
                                                           cycle_snapshot))
                        MPI_Abort(MPI_COMM_WORLD, 2);
                    /* Replay only on rejection; accepted cycles avoid the
                     * per-position KDA snapshot copies entirely. */
                    for (int j = 0; j < committed; ++j) {
                        int replay_next;
                        float replay_logit;
                        float *replay_out = verify_hidden + (size_t)j * HIDDEN;
                        if (glm53f_target_model_step_12n(target_model,
                                verify_input[j], &replay_next, &replay_logit,
                                replay_out)) MPI_Abort(MPI_COMM_WORLD, 2);
                        if (replay_next != target[j]) MPI_Abort(MPI_COMM_WORLD, 2);
                    }
                    memcpy(target_hidden,
                           verify_hidden + (size_t)(committed - 1) * HIDDEN,
                           HIDDEN * sizeof(float));
                } else if (glm53f_target_snapshot_restore_12n(
                               target_model, snapshot[committed-1]))
                    MPI_Abort(MPI_COMM_WORLD,2);
                if (carry) {
                    /* Evaluate the fallback immediately.  This leaves the
                     * target at the fallback prefix, so its output can be
                     * carried into the next cycle just like a bonus from a
                     * fully accepted draft. */
                    int fallback=next_token, bonus;
                    float bonus_logit;
                    if (glm53f_target_model_step_12n(target_model,fallback,
                            &bonus,&bonus_logit,target_hidden))
                        MPI_Abort(MPI_COMM_WORLD,2);
                    carry_fallback=fallback;
                    next_token=bonus;
                    carry_bonus=1;
                }
            }
            phase[2]+=MPI_Wtime()-phase_begin;phase_begin=MPI_Wtime();
            /* Rebuild the committed MTP input suffix from target hidden states.
             * mtp_forward(first_token, ...) already advanced a speculative
             * cache, but that state is not authoritative after verification.
             * Restore the pre-cycle length, then append each committed input
             * exactly once after rejection. On full acceptance, the speculative
             * state contains the first `ndraft` inputs; append only the final
             * draft input. The old base+1 restore appended first_token twice,
             * causing long-run drift after rollback cycles. */
            if (deferred_emit) {
                if (accepted) {
                    if (glm53f_mtp_cache_append_12n(mtp,verify_input[1],verify_hidden))
                        MPI_Abort(MPI_COMM_WORLD,2);
                    if (glm53f_mtp_length_12n(mtp) != mtp_base + 2)
                        MPI_Abort(MPI_COMM_WORLD,2);
                } else {
                    if (glm53f_mtp_restore_length_12n(mtp,mtp_base) ||
                        glm53f_mtp_restore_selection_12n(mtp, &mtp_selection, mtp_history) ||
                        glm53f_mtp_cache_append_12n(mtp,verify_input[0],target_hidden))
                        MPI_Abort(MPI_COMM_WORLD,2);
                    if (glm53f_mtp_length_12n(mtp) != mtp_base + 1)
                        MPI_Abort(MPI_COMM_WORLD,2);
                }
            } else if (accepted == ndraft) {
                int rc = glm53f_mtp_cache_append_12n(mtp,verify_input[ndraft],
                    verify_hidden+(size_t)(ndraft-1)*HIDDEN);
                if (rc) MPI_Abort(MPI_COMM_WORLD,2);
            } else {
                if(glm53f_mtp_restore_length_12n(mtp,mtp_base) ||
                   glm53f_mtp_restore_selection_12n(mtp, &mtp_selection, mtp_history))
                    MPI_Abort(MPI_COMM_WORLD,2);
                int ignored; float ignored_logit; float *replay_hidden=draft_hidden[0];
                for(int j=0;j<committed;j++) {
                    const float *committed_hidden = j == 0 ? cycle_hidden :
                        verify_hidden+(size_t)(j-1)*HIDDEN;
                    int rc = full_replay ? glm53f_mtp_forward_12n(mtp,
                        verify_input[j],committed_hidden,&ignored,&ignored_logit,
                        replay_hidden) : glm53f_mtp_cache_append_12n(mtp,
                        verify_input[j],committed_hidden);
                    if (rc) MPI_Abort(MPI_COMM_WORLD,2);
                }
                if (carry_bonus && glm53f_mtp_cache_append_12n(mtp,
                        carry_fallback,verify_hidden+(size_t)(committed-1)*HIDDEN))
                    MPI_Abort(MPI_COMM_WORLD,2);
            }
            if (!deferred_emit && glm53f_mtp_length_12n(mtp) != mtp_base + committed + carry_bonus)
                MPI_Abort(MPI_COMM_WORLD,2);
            phase[3]+=MPI_Wtime()-phase_begin;
            if (carry && !carry_bonus) memcpy(target_hidden, verify_hidden + (size_t)(committed-1)*HIDDEN,
                              HIDDEN*sizeof(float));
            int mtp_commit=committed+(carry ? 0 : 1)+carry_bonus;
            accepted_total+=accepted;token=next_token;
            int emitted[MAX_DRAFT + 3], count = deferred_emit ? 2 : accepted + 2 + carry_bonus;
            emitted[0] = first_token;
            if (deferred_emit) emitted[1] = accepted ? draft[0] : next_token;
            else {
                for (int j = 0; j < accepted; j++) emitted[j + 1] = draft[j];
                if (carry_bonus)
                    emitted[accepted + 1] = carry_fallback;
                emitted[count - 1] = next_token;
            }
            /* Carried first_token was delivered as the previous cycle's bonus. */
            for (int j = carry && cycle > 0 ? 1 : 0; j < count; j++) {
                int id = emitted[j];
                if (output) fprintf(output, "%d\n", id);
                if (!rank && reference && delivered < reference_count) {
                    if (reference[delivered] != id && reference_ok) {
                        fprintf(stderr, "GLM53F_SPEC_MISMATCH position=%ld got=%d expected=%d\n",
                                delivered, id, reference[delivered]);
                        reference_ok = 0;
                    }
                    checked++;
                }
                delivered++;
                token = id;
                if (prompt && (id == 154820 || id == 154827 || id == 154829)) {
                    eos = 1; token = id; break;
                }
                if (max_tokens && delivered >= max_tokens) break;
            }
            if (output) fflush(output);
            executed++;
            if(!rank)printf("GLM53F_SPEC_CYCLE cycle=%d first=%d accepted=%d/%d fallback=%d target_steps=%d mtp_steps=%d\n",cycle,first_token,accepted,ndraft,next_token,committed,mtp_commit);
            if (eos || (max_tokens && delivered >= max_tokens)) break;
        }
        double sec=MPI_Wtime()-begin,max_sec;MPI_Reduce(&sec,&max_sec,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        double max_phase[4];MPI_Reduce(phase,max_phase,4,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        if(!rank)printf("GLM53F_SPEC_PHASE ms_cycle target=%.3f draft=%.3f verify=%.3f rebase=%.3f\n",max_phase[0]*1e3/executed,max_phase[1]*1e3/executed,max_phase[2]*1e3/executed,max_phase[3]*1e3/executed);
        double alpha=proposed_total?(double)accepted_total/proposed_total:0.0;
        if (!rank && reference && checked != delivered) {
            fprintf(stderr, "GLM53F_SPEC_REFERENCE_SHORT checked=%ld delivered=%ld\n",
                    checked, delivered);
            reference_ok = 0;
        }
        MPI_Bcast(&reference_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int pass=reference_ok;const char*gate;
        gate=getenv("GLM53F_SPEC_MIN_ALPHA");if(gate&&*gate&&alpha<strtod(gate,NULL))pass=0;
        gate=getenv("GLM53F_SPEC_EXPECT_ACCEPTED");if(gate&&*gate&&accepted_total!=strtol(gate,NULL,10))pass=0;
        gate=getenv("GLM53F_SPEC_EXPECT_FINAL");if(gate&&*gate&&token!=strtol(gate,NULL,10))pass=0;
        if(!rank)printf("GLM53F_SPEC_DECODE_12N cycles=%d drafts=%d accepted=%ld/%ld alpha=%.6f delivered=%ld tok_s=%.3f final_token=%d %s\n",executed,ndraft,accepted_total,proposed_total,alpha,delivered,delivered/max_sec,token,pass?"PASS":"FAIL");
        if (!rank && reference) printf("GLM53F_SPEC_REFERENCE checked=%ld delivered=%ld %s\n",
                                       checked, delivered, reference_ok ? "PASS" : "FAIL");
        glm53f_target_profile_report_12n(target_model,"spec");
        if (output && fclose(output)) MPI_Abort(MPI_COMM_WORLD, 2);
        output = NULL;
        all_pass &= pass;
    }
    glm53f_target_snapshot_free_12n(cycle_snapshot);
    glm53f_target_snapshot_free_12n(initial);
    free(reference);
    free(prompt);
    for(int i=0;i<max_draft;i++)glm53f_target_snapshot_free_12n(snapshot[i]);
    free(draft_hidden[1]);free(draft_hidden[0]);free(cycle_hidden);free(target_hidden);free(verify_hidden);
    overall_pass &= all_pass;
    }
    free(requests);
    glm53f_target_snapshot_free_12n(prefill_snapshot);
    glm53f_mtp_free_12n(mtp);glm53f_target_model_free_12n(target_model);glm53f_collective_free_12n();
    MPI_Finalize();return overall_pass?0:1;
}
