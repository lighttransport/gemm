/* Stateful greedy MTP draft / full-target verify / rollback loop. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glm53f_mtp_12n.h"
#include "glm53f_target_model_12n.h"

enum { HIDDEN = 4096, MAX_DRAFT = 4 };

static void *a256(size_t n) {
    void *p = NULL;
    return posix_memalign(&p, 256, n) ? NULL : p;
}

int main(int argc, char **argv) {
    int rank, ranks, token, cycles, ndraft, warmup, capacity;
    int draft[MAX_DRAFT], target[MAX_DRAFT + 1];
    long accepted_total = 0, proposed_total = 0, delivered = 0;
    glm53f_target_model_12n *target_model;
    glm53f_mtp_context_12n *mtp;
    glm53f_target_snapshot_12n *snapshot[MAX_DRAFT + 2] = {0};
    float *target_hidden, *draft_hidden[2], target_logit, draft_logit;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 7 || ranks != 12) {
        if (!rank) fprintf(stderr,"usage: %s MODEL TARGET_ROUTED TARGET_SHARED MTP_ROUTED MTP_SHARED [token=1] [cycles=1] [drafts=4] [warmup=128]\n",argv[0]);
        MPI_Abort(MPI_COMM_WORLD,2);
    }
    token=argc>6?atoi(argv[6]):1;cycles=argc>7?atoi(argv[7]):1;
    ndraft=argc>8?atoi(argv[8]):MAX_DRAFT;
    warmup=argc>9?atoi(argv[9]):128;
    if(token<0||token>=154880||cycles<1||ndraft<1||ndraft>MAX_DRAFT||warmup<0)MPI_Abort(MPI_COMM_WORLD,2);
    capacity=warmup+cycles*(ndraft+2)+1;
    target_model=glm53f_target_model_create_12n(argv[1],argv[2],argv[3],capacity);
    mtp=glm53f_mtp_create_12n(argv[1],argv[4],argv[5],capacity);
    target_hidden=a256(HIDDEN*4);draft_hidden[0]=a256(HIDDEN*4);draft_hidden[1]=a256(HIDDEN*4);
    if(!target_model||!mtp||!target_hidden||!draft_hidden[0]||!draft_hidden[1])MPI_Abort(MPI_COMM_WORLD,2);
    for(int i=0;i<ndraft+2;i++){snapshot[i]=glm53f_target_snapshot_create_12n(target_model);if(!snapshot[i])MPI_Abort(MPI_COMM_WORLD,2);}
    for(int i=0;i<warmup;i++){
        int next,ignored;float next_logit,ignored_logit;
        if(glm53f_target_model_step_12n(target_model,token,&next,&next_logit,target_hidden))MPI_Abort(MPI_COMM_WORLD,2);
        if(glm53f_mtp_forward_12n(mtp,next,target_hidden,&ignored,&ignored_logit,draft_hidden[0]))MPI_Abort(MPI_COMM_WORLD,2);
        token=next;
    }
    if(!rank)printf("GLM53F_SPEC_WARMUP tokens=%d next_token=%d target_cache=%d mtp_cache=%d\n",warmup,token,warmup,glm53f_mtp_length_12n(mtp));
    MPI_Barrier(MPI_COMM_WORLD);double begin=MPI_Wtime();
    for(int cycle=0;cycle<cycles;cycle++){
        int first_token;
        if(glm53f_target_model_step_12n(target_model,token,&first_token,&target_logit,target_hidden))MPI_Abort(MPI_COMM_WORLD,2);
        delivered++;
        if(glm53f_target_snapshot_save_12n(target_model,snapshot[0]))MPI_Abort(MPI_COMM_WORLD,2);
        int mtp_base=glm53f_mtp_length_12n(mtp),input=first_token;
        const float*hidden=target_hidden;
        for(int j=0;j<ndraft;j++){
            float*out_hidden=draft_hidden[j&1];
            if(glm53f_mtp_forward_12n(mtp,input,hidden,&draft[j],&draft_logit,out_hidden))MPI_Abort(MPI_COMM_WORLD,2);
            input=draft[j];hidden=out_hidden;
        }
        proposed_total+=ndraft;
        int accepted=0,committed=0,verify_input=first_token,next_token=-1;
        for(int j=0;j<ndraft;j++){
            if(glm53f_target_model_step_12n(target_model,verify_input,&target[j],&target_logit,target_hidden))MPI_Abort(MPI_COMM_WORLD,2);
            if(glm53f_target_snapshot_save_12n(target_model,snapshot[j+1]))MPI_Abort(MPI_COMM_WORLD,2);
            committed=j+1;
            if(target[j]!=draft[j]){next_token=target[j];break;}
            accepted++;verify_input=draft[j];
        }
        if(accepted==ndraft){
            if(glm53f_target_model_step_12n(target_model,draft[ndraft-1],&target[ndraft],&target_logit,target_hidden))MPI_Abort(MPI_COMM_WORLD,2);
            if(glm53f_target_snapshot_save_12n(target_model,snapshot[ndraft+1]))MPI_Abort(MPI_COMM_WORLD,2);
            committed=ndraft+1;next_token=target[ndraft];
        }
        if(glm53f_target_snapshot_restore_12n(target_model,snapshot[committed]))MPI_Abort(MPI_COMM_WORLD,2);
        int mtp_commit=accepted==ndraft?ndraft:accepted+1;
        if(glm53f_mtp_restore_length_12n(mtp,mtp_base+mtp_commit))MPI_Abort(MPI_COMM_WORLD,2);
        accepted_total+=accepted;delivered+=accepted+1;token=next_token;
        if(!rank)printf("GLM53F_SPEC_CYCLE cycle=%d first=%d accepted=%d/%d fallback=%d target_steps=%d mtp_steps=%d\n",cycle,first_token,accepted,ndraft,next_token,committed,mtp_commit);
    }
    double sec=MPI_Wtime()-begin,max_sec;MPI_Reduce(&sec,&max_sec,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(!rank)printf("GLM53F_SPEC_DECODE_12N cycles=%d drafts=%d accepted=%ld/%ld alpha=%.6f delivered=%ld tok_s=%.3f final_token=%d PASS\n",cycles,ndraft,accepted_total,proposed_total,proposed_total?(double)accepted_total/proposed_total:0.0,delivered,delivered/max_sec,token);
    for(int i=0;i<ndraft+2;i++)glm53f_target_snapshot_free_12n(snapshot[i]);
    glm53f_mtp_free_12n(mtp);glm53f_target_model_free_12n(target_model);
    MPI_Finalize();return 0;
}
