/* Stateful greedy MTP draft / full-target verify / rollback loop. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glm53f_mtp_12n.h"
#include "glm53f_collective_12n.h"
#include "glm53f_target_model_12n.h"

enum { HIDDEN = 4096, MAX_DRAFT = 4 };

static void *a256(size_t n) {
    void *p = NULL;
    return posix_memalign(&p, 256, n) ? NULL : p;
}

int main(int argc, char **argv) {
    int rank, ranks, token, cycles, ndraft, warmup, capacity;
    int draft[MAX_DRAFT], target[MAX_DRAFT + 1], verify_input[MAX_DRAFT + 1];
    float verify_logit[MAX_DRAFT + 1];
    double phase[4] = {0.0, 0.0, 0.0, 0.0};
    long accepted_total = 0, proposed_total = 0, delivered = 0;
    glm53f_target_model_12n *target_model;
    glm53f_mtp_context_12n *mtp;
    glm53f_target_snapshot_12n *snapshot[MAX_DRAFT + 2] = {0};
    float *target_hidden, *verify_hidden, *draft_hidden[2], target_logit, draft_logit;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 7 || ranks != 12) {
        if (!rank) fprintf(stderr,"usage: %s MODEL TARGET_ROUTED TARGET_SHARED MTP_ROUTED MTP_SHARED [token=1] [cycles=1] [drafts=1] [warmup=128]\n",argv[0]);
        MPI_Abort(MPI_COMM_WORLD,2);
    }
    token=argc>6?atoi(argv[6]):1;cycles=argc>7?atoi(argv[7]):1;
    ndraft=argc>8?atoi(argv[8]):1;
    warmup=argc>9?atoi(argv[9]):128;
    if(token<0||token>=154880||cycles<1||ndraft<1||ndraft>MAX_DRAFT||warmup<0)MPI_Abort(MPI_COMM_WORLD,2);
    if(getenv("GLM53F_UTOFU")){const char*topo=getenv("TOFU_TOPO_PATH");if(!topo)topo="../utofu-tests/tofu_topo.txt";if(glm53f_collective_init_12n(topo,5*HIDDEN))MPI_Abort(MPI_COMM_WORLD,2);}
    capacity=warmup+cycles*(ndraft+2)+1;
    target_model=glm53f_target_model_create_12n(argv[1],argv[2],argv[3],capacity);
    mtp=glm53f_mtp_create_12n(argv[1],argv[4],argv[5],capacity);
    target_hidden=a256(HIDDEN*4);verify_hidden=a256((MAX_DRAFT+1)*HIDDEN*4);draft_hidden[0]=a256(HIDDEN*4);draft_hidden[1]=a256(HIDDEN*4);
    if(!target_model||!mtp||!target_hidden||!verify_hidden||!draft_hidden[0]||!draft_hidden[1])MPI_Abort(MPI_COMM_WORLD,2);
    for(int i=0;i<ndraft+2;i++){snapshot[i]=glm53f_target_snapshot_create_12n(target_model);if(!snapshot[i])MPI_Abort(MPI_COMM_WORLD,2);}
    for(int i=0;i<warmup;i++){
        int next,ignored;float next_logit,ignored_logit;
        if(glm53f_target_model_step_12n(target_model,token,&next,&next_logit,target_hidden))MPI_Abort(MPI_COMM_WORLD,2);
        if(glm53f_mtp_forward_12n(mtp,next,target_hidden,&ignored,&ignored_logit,draft_hidden[0]))MPI_Abort(MPI_COMM_WORLD,2);
        token=next;
    }
    if(!rank)printf("GLM53F_SPEC_WARMUP tokens=%d next_token=%d target_cache=%d mtp_cache=%d\n",warmup,token,warmup,glm53f_mtp_length_12n(mtp));
    glm53f_target_profile_reset_12n(target_model);
    MPI_Barrier(MPI_COMM_WORLD);double begin=MPI_Wtime();
    for(int cycle=0;cycle<cycles;cycle++){
        int first_token;
        double phase_begin=MPI_Wtime();
        if(glm53f_target_model_step_12n(target_model,token,&first_token,&target_logit,target_hidden))MPI_Abort(MPI_COMM_WORLD,2);
        phase[0]+=MPI_Wtime()-phase_begin;phase_begin=MPI_Wtime();
        delivered++;
        if(glm53f_target_snapshot_save_12n(target_model,snapshot[0]))MPI_Abort(MPI_COMM_WORLD,2);
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
        if(glm53f_target_model_step_batch_12n(target_model,verify_input,ndraft+1,
                target,verify_logit,verify_hidden,snapshot+1))MPI_Abort(MPI_COMM_WORLD,2);
        int accepted=0,committed=ndraft+1,next_token=target[ndraft];
        for(int j=0;j<ndraft;j++)if(target[j]==draft[j])accepted++;else{committed=j+1;next_token=target[j];break;}
        if(glm53f_target_snapshot_restore_12n(target_model,snapshot[committed]))MPI_Abort(MPI_COMM_WORLD,2);
        phase[2]+=MPI_Wtime()-phase_begin;phase_begin=MPI_Wtime();
        /* Rebuild the committed MTP suffix from target hidden states. Draft
         * hidden states are approximate, and retaining them after rejection
         * leaves the cache one position behind the target sequence. */
        /* The first draft call already used first_token with the exact scalar
         * target hidden state, so retain that cache entry and replay only the
         * suffix whose draft hidden states were approximate. */
        if(glm53f_mtp_restore_length_12n(mtp,mtp_base+1))MPI_Abort(MPI_COMM_WORLD,2);
        int ignored;float ignored_logit;float*replay_hidden=draft_hidden[0];
        for(int j=0;j<committed;j++)if(glm53f_mtp_forward_12n(mtp,target[j],
                verify_hidden+(size_t)j*HIDDEN,&ignored,&ignored_logit,
                replay_hidden))MPI_Abort(MPI_COMM_WORLD,2);
        phase[3]+=MPI_Wtime()-phase_begin;
        int mtp_commit=committed+1;
        accepted_total+=accepted;delivered+=accepted+1;token=next_token;
        if(!rank)printf("GLM53F_SPEC_CYCLE cycle=%d first=%d accepted=%d/%d fallback=%d target_steps=%d mtp_steps=%d\n",cycle,first_token,accepted,ndraft,next_token,committed,mtp_commit);
    }
    double sec=MPI_Wtime()-begin,max_sec;MPI_Reduce(&sec,&max_sec,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    double max_phase[4];MPI_Reduce(phase,max_phase,4,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(!rank)printf("GLM53F_SPEC_PHASE ms_cycle target=%.3f draft=%.3f verify=%.3f rebase=%.3f\n",max_phase[0]*1e3/cycles,max_phase[1]*1e3/cycles,max_phase[2]*1e3/cycles,max_phase[3]*1e3/cycles);
    double alpha=proposed_total?(double)accepted_total/proposed_total:0.0;
    int pass=1;const char*gate;
    gate=getenv("GLM53F_SPEC_MIN_ALPHA");if(gate&&*gate&&alpha<strtod(gate,NULL))pass=0;
    gate=getenv("GLM53F_SPEC_EXPECT_ACCEPTED");if(gate&&*gate&&accepted_total!=strtol(gate,NULL,10))pass=0;
    gate=getenv("GLM53F_SPEC_EXPECT_FINAL");if(gate&&*gate&&token!=strtol(gate,NULL,10))pass=0;
    if(!rank)printf("GLM53F_SPEC_DECODE_12N cycles=%d drafts=%d accepted=%ld/%ld alpha=%.6f delivered=%ld tok_s=%.3f final_token=%d %s\n",cycles,ndraft,accepted_total,proposed_total,alpha,delivered,delivered/max_sec,token,pass?"PASS":"FAIL");
    glm53f_target_profile_report_12n(target_model,"spec");
    for(int i=0;i<ndraft+2;i++)glm53f_target_snapshot_free_12n(snapshot[i]);
    free(verify_hidden);glm53f_mtp_free_12n(mtp);glm53f_target_model_free_12n(target_model);glm53f_collective_free_12n();
    MPI_Finalize();return pass?0:1;
}
