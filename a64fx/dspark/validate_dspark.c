#define _POSIX_C_SOURCE 200809L
#include "dspark.h"
#include "../../common/safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint32_t rng_state=1;
static float next_value(void){rng_state=rng_state*1664525u+1013904223u;return (float)((int)(rng_state>>16)-32768)/32768.0f;}

static int compare_golden(const char *path,const dspark_proposal*p){
    FILE*f=fopen(path,"rb");if(!f){perror(path);return 1;}fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);char*s=malloc((size_t)n+1);if(!s||fread(s,1,(size_t)n,f)!=(size_t)n){fclose(f);free(s);return 1;}fclose(f);s[n]='\0';
    json_val*root=json_parse(s,(int)n);free(s);json_val*ids=root?json_obj_get(root,"token_ids"):NULL,*conf=root?json_obj_get(root,"confidence"):NULL,*logits=root?json_obj_get(root,"selected_logits"):NULL;int fail=0;
    if(!ids||!conf||!logits||ids->type!=JSON_ARRAY||conf->type!=JSON_ARRAY||logits->type!=JSON_ARRAY||ids->arr.count!=7||conf->arr.count!=7||logits->arr.count!=7)fail=1;
    for(int i=0;i<7&&!fail;i++){
        int id=(int)ids->arr.items[i].num;float c=(float)conf->arr.items[i].num,l=(float)logits->arr.items[i].num;
        if(id!=p->token_ids[i]||fabsf(c-p->confidence[i])>2e-3f||fabsf(l-p->selected_logits[i])>2e-3f*fmaxf(1.0f,fabsf(l))){fprintf(stderr,"golden mismatch row=%d token=%d/%d confidence=%g/%g logit=%g/%g\n",i,p->token_ids[i],id,p->confidence[i],c,p->selected_logits[i],l);fail=1;}
    }
    json_free(root);printf("golden %s\n",fail?"FAIL":"PASS");return fail;
}

int main(int argc,char**argv){
    const char*golden=NULL;int ai=1;if(argc>2&&!strcmp(argv[1],"--golden")){golden=argv[2];ai=3;}
    if(argc-ai!=3||(strcmp(argv[ai],"--headers")&&strcmp(argv[ai],"--full"))){fprintf(stderr,"usage: %s [--golden FILE] --headers|--full DRAFT_DIR TARGET_DIR\n",argv[0]);return 2;}
    char err[512]={0};double t0=now_sec();int rc=dspark_validate_files(argv[ai+1],argv[ai+2],err,sizeof(err));
    if(rc){fprintf(stderr,"validation failed: %s\n",err);return 1;}
    printf("headers PASS elapsed=%.3f s\n",now_sec()-t0);if(!strcmp(argv[ai],"--headers"))return 0;
    dspark_load_options lo={DSPARK_BACKEND_AUTO,0};dspark_model*m=NULL;t0=now_sec();rc=dspark_model_load(&m,argv[ai+1],argv[ai+2],&lo,err,sizeof(err));
    if(rc){fprintf(stderr,"load failed: %s\n",err);return 1;}printf("load PASS backend=%s elapsed=%.3f s\n",dspark_backend_name(dspark_model_backend(m)),now_sec()-t0);
    dspark_state_options so={16};dspark_state*s=NULL;rc=dspark_state_create(&s,m,&so,err,sizeof(err));if(rc){fprintf(stderr,"state failed: %s\n",err);dspark_model_free(m);return 1;}
    float *tap[5];for(int j=0;j<5;j++){tap[j]=malloc(2*5120*sizeof(float));for(int i=0;i<2*5120;i++)tap[j][i]=next_value()*0.02f;}
    const float*ct[5]={tap[0],tap[1],tap[2],tap[3],tap[4]};rc=dspark_state_append_target(s,ct,2,5120,err,sizeof(err));
    dspark_proposal p;t0=now_sec();if(!rc)rc=dspark_state_propose(s,42,&p,err,sizeof(err));
    if(rc)fprintf(stderr,"forward failed: %s\n",err);else{printf("proposal PASS elapsed=%.3f s tokens=",now_sec()-t0);for(size_t i=0;i<p.count;i++)printf("%s%d@%.6f",i?",":"",p.token_ids[i],p.confidence[i]);putchar('\n');if(golden&&compare_golden(golden,&p))rc=DSPARK_EFORMAT;}
    for(int j=0;j<5;j++)free(tap[j]);
    dspark_state_free(s);dspark_model_free(m);return rc?1:0;
}
