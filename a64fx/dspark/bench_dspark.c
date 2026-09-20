#define _POSIX_C_SOURCE 200809L
#include "dspark_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
    size_t profile_context=0;
    if(argc==5&&!strcmp(argv[3],"--profile-context")){
        char*end=NULL;unsigned long long n=strtoull(argv[4],&end,10);
        if(!end||*end||!n||n>262144-DSPARK_BLOCK_SIZE){fprintf(stderr,"invalid profile context: %s\n",argv[4]);return 2;}
        profile_context=(size_t)n;
    }else if(argc!=3){fprintf(stderr,"usage: %s DRAFT_DIR TARGET_DIR [--profile-context N]\n",argv[0]);return 2;}
    char err[512]={0};dspark_model*m=NULL;dspark_load_options lo={DSPARK_BACKEND_AUTO,0};int rc=dspark_model_load(&m,argv[1],argv[2],&lo,err,sizeof(err));if(rc){fprintf(stderr,"load: %s\n",err);return 1;}
    dspark_state*s=NULL;dspark_state_options so={profile_context?profile_context:8192};rc=dspark_state_create(&s,m,&so,err,sizeof(err));if(rc){fprintf(stderr,"state: %s\n",err);dspark_model_free(m);return 1;}
    if(profile_context){
        dspark_proposal proposal;s->cursor=profile_context;m->threads=48;m->backend=DSPARK_BACKEND_SVE;
        double t0=now_sec();rc=dspark_state_propose(s,42,&proposal,err,sizeof(err));double elapsed=now_sec()-t0;
        if(rc)fprintf(stderr,"profile: %s\n",err);else printf("DSPARK_CONTEXT context=%zu threads=48 sve_ms=%.3f tokens=",profile_context,elapsed*1e3);
        if(!rc){for(size_t i=0;i<proposal.count;i++)printf("%s%d",i?",":"",proposal.token_ids[i]);putchar('\n');}
        dspark_state_free(s);dspark_model_free(m);return rc?1:0;
    }
    float *tap[5];for(int j=0;j<5;j++){tap[j]=calloc(5120,sizeof(float));for(int i=0;i<5120;i++)tap[j][i]=(float)((i*17+j*13)%101-50)*1e-4f;}
    const float*ct[5]={tap[0],tap[1],tap[2],tap[3],tap[4]};double t0=now_sec();rc=dspark_state_append_target(s,ct,1,5120,err,sizeof(err));double ta=now_sec()-t0;
    if(!rc)printf("DSPARK_BENCH append_ms=%.3f context=%zu\n",ta*1e3,dspark_state_context_tokens(s));
    const int teams[3]={1,12,48};
    for(int ti=0;ti<3&&!rc;ti++){
        dspark_proposal sve,scalar;m->threads=teams[ti];m->backend=DSPARK_BACKEND_SVE;
        t0=now_sec();rc=dspark_state_propose(s,42,&sve,err,sizeof(err));double ts=now_sec()-t0;
        if(rc)break;
        m->backend=DSPARK_BACKEND_SCALAR;t0=now_sec();rc=dspark_state_propose(s,42,&scalar,err,sizeof(err));double tc=now_sec()-t0;
        int match=1;float max_conf=0,max_logit_rel=0;for(int i=0;i<7;i++){if(sve.token_ids[i]!=scalar.token_ids[i])match=0;float e=fabsf(sve.confidence[i]-scalar.confidence[i]);if(e>max_conf)max_conf=e;e=fabsf(sve.selected_logits[i]-scalar.selected_logits[i])/fmaxf(1.0f,fabsf(scalar.selected_logits[i]));if(e>max_logit_rel)max_logit_rel=e;}
        printf("DSPARK_BENCH threads=%d sve_ms=%.3f scalar_ms=%.3f speedup=%.3f token_match=%d max_conf_diff=%g max_logit_rel=%g\n",teams[ti],ts*1e3,tc*1e3,tc/ts,match,max_conf,max_logit_rel);
        if(!match||max_conf>2e-4f||max_logit_rel>5e-4f)rc=DSPARK_EFORMAT;
    }
    for(int j=0;j<5;j++)free(tap[j]);
    dspark_state_free(s);dspark_model_free(m);return rc?1:0;
}
