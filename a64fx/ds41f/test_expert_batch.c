#define _POSIX_C_SOURCE 200809L
#include "ds41f_expert.h"
#include "ds41f_fp4_sdot.h"
#include "ds41f_kernels.h"
#include "ds41f_team.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
static void require(int ok,const char *s){if(!ok){fprintf(stderr,"EXPERT_BATCH FAIL %s\n",s);exit(1);}}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void check(void *arg)
{
    (void)arg;size_t values=(size_t)5120*2304;ds41f_expert original={{0},{0},0},packed={{0},{0},1};
    for(int m=0;m<3;++m){original.weight[m]=malloc(values/2);original.scale[m]=malloc(values/32);
        packed.weight[m]=malloc(values/2);packed.scale[m]=malloc(values/32);
        require(original.weight[m]&&original.scale[m]&&packed.weight[m]&&packed.scale[m],"weights");
        for(size_t i=0;i<values/2;++i)original.weight[m][i]=(uint8_t)(i*73+i/17+m*13);
        for(size_t i=0;i<values/32;++i)original.scale[m][i]=(uint8_t)(113+(i+m)%10);
        require(!ds41f_mxfp4_pack_sdot(packed.weight[m],packed.scale[m],original.weight[m],original.scale[m],m==1?5120:2304,m==1?2304:5120),"pack");}
    float *input=malloc(6*5120*4),*q=malloc(6*5120*4),*out=malloc(6*5124*4),*ref=malloc(6*5124*4),*scratch=malloc(6*3*2304*4);
    float route[6]={0,.125f,.25f,.5f,.75f,1.f};require(input&&q&&out&&ref&&scratch,"scratch");
    for(size_t i=0;i<6*5120;++i)input[i]=sinf(i*.13f)*.75f;
    for(size_t i=0;i<6;++i)require(!ds41f_act_quant(q+i*5120,input+i*5120,5120),"prepare");
    for(int mode=0;mode<2;++mode){ds41f_expert *e=mode?&packed:&original;
        for(size_t i=0;i<6;++i)require(!ds41f_expert_forward_fused(e,ref+i*5124,input+i*5120,route[i],scratch,0,1),"control");
        for(size_t i=0;i<6;++i){ds41f_int8_input prepared={0};
            if(mode)require(!ds41f_int8_prepare_input(&prepared,q+i*5120,5120,32),"shared input");
            require(!ds41f_expert_forward_prepared(e,out,q+i*5120,mode?&prepared:NULL,route[i],scratch,0,1),"prepared expert");
            require(!memcmp(out,ref+i*5124,5120*4),"prepared expert bit exact");ds41f_int8_input_free(&prepared);}
        for(size_t n=1;n<=6;++n){for(size_t i=0;i<6*5124;++i)out[i]=12345;
            double start=now();require(!ds41f_expert_batch_prepared(e,out,5124,q,5120,route,scratch,n),"batch");double batch=now()-start;
            for(size_t t=0;t<n;++t)require(!memcmp(out+t*5124,ref+t*5124,5120*4),"bit exact");
            for(size_t t=0;t<6;++t)for(size_t j=t<n?5120:0;j<5124;++j)require(out[t*5124+j]==12345,"canary");
            start=now();for(size_t i=0;i<n;++i)require(!ds41f_expert_forward_fused(e,out+i*5124,input+i*5120,route[i],scratch,0,1),"timing control");
            printf("EXPERT_BATCH packed=%d batch=%zu sequential_ms=%.3f batch_ms=%.3f\n",mode,n,(now()-start)*1e3,batch*1e3);
        }
    }
    ds41f_expert_free(&original);ds41f_expert_free(&packed);free(input);free(q);free(out);free(ref);free(scratch);
    puts("EXPERT_BATCH PASS cases=12 full_expert shared_input bit_exact strides canaries zero_route");
}
int main(void){return ds41f_team_run(check,NULL);}
