#define _POSIX_C_SOURCE 200809L
#include "ds41f_int8.h"
#include "ds41f_team.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static void require(int ok,const char *message)
{if(!ok){fprintf(stderr,"INT8_BATCH FAIL %s\n",message);exit(1);}}
static double now(void)
{struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
typedef struct {float *out;size_t stride,batch,group;const ds41f_int8 *q;const ds41f_int8_input *input;int rc;} team_case;
static void team_check(void *context)
{team_case *c=context;c->rc=ds41f_int8_matmul_prepared(c->out,c->stride,c->q,c->input,c->batch,c->group,0);}
static void check(size_t rows,size_t cols,size_t block,size_t group,int bench)
{
    size_t elements=(rows/group)*cols,istride=elements+7,ostride=rows+11;
    uint8_t *w=malloc(rows*cols),*s=malloc(((rows+31)/32)*(cols/32));
    float *x=malloc(istride*6*4),*got=malloc(ostride*6*4),*ref=malloc(ostride*6*4);
    require(w&&s&&x&&got&&ref,"allocation");
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<rows*cols;++i)w[i]=(uint8_t)((i*17+i/cols*3)%112)|((i%3)?0:128);
    for(size_t i=0;i<((rows+31)/32)*(cols/32);++i)s[i]=(uint8_t)(122+i%7);
    for(size_t i=0;i<istride*6;++i)x[i]=sinf((float)i*.071f)*.75f;
    ds41f_int8 q;require(!ds41f_int8_from_fp8(&q,w,s,rows,cols,block),"convert");free(w);free(s);
    ds41f_int8_input prepared[6]={{0}};
    for(size_t t=0;t<6;++t){
        require(!ds41f_int8_prepare_input(prepared+t,x+t*istride,elements,block),"prepare");
        for(size_t r=0;r<ostride;++r)ref[t*ostride+r]=12345;
        require(!ds41f_int8_matvec(ref+t*ostride,&q,x+t*istride,group,0),"sequential GEMV");
    }
    for(size_t batch=1;batch<=6;++batch){
        for(size_t i=0;i<6*ostride;++i)got[i]=12345;
        require(!ds41f_int8_matmul_prepared(got,ostride,&q,prepared,batch,group,0),"prepared batch");
        require(!memcmp(got,ref,batch*ostride*4),"prepared bit exact and stride canaries");
        for(size_t i=0;i<6*ostride;++i)got[i]=12345;
        team_case c={got,ostride,batch,group,&q,prepared,0};
        require(!ds41f_team_run(team_check,&c)&&!c.rc,"persistent prepared batch");
        require(!memcmp(got,ref,batch*ostride*4),"persistent bit exact and stride canaries");
        for(size_t i=batch*ostride;i<6*ostride;++i)require(got[i]==12345,"inactive token canary");
        for(size_t i=0;i<6*ostride;++i)got[i]=12345;
        require(!ds41f_int8_matmul(got,ostride,&q,x,istride,batch,group,0),"input batch");
        require(!memcmp(got,ref,batch*ostride*4),"input bit exact and stride canaries");
    }
    require(ds41f_int8_matmul(got,ostride,&q,NULL,istride,6,group,0)==EINVAL,"null input");
    require(ds41f_int8_matmul(got,ostride,&q,x,SIZE_MAX,6,group,0)==EINVAL,"overflow stride");
    require(ds41f_int8_matmul_prepared(got,ostride,&q,prepared,7,group,0)==EINVAL,"batch bound");
    require(ds41f_int8_matmul_prepared(got,rows-1,&q,prepared,6,group,0)==EINVAL,"short output stride");
    if(bench){
        size_t n=16*1024*1024;float *flush=malloc(n*4);require(flush!=NULL,"flush allocation");
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<n;++i)flush[i]=(float)(i%7);
        volatile float guard=0;
        for(size_t batch=1;batch<=6;++batch){double elapsed[2]={0,0};
            for(int mode=0;mode<2;++mode)for(int it=0;it<16;++it){float value=0;
                #pragma omp parallel for reduction(+:value) schedule(static)
                for(size_t i=0;i<n;i+=16)value+=flush[i];guard=value;
                double start=now();
                if(mode)require(!ds41f_int8_matmul(got,ostride,&q,x,istride,batch,group,0),"bench batch");
                else for(size_t t=0;t<batch;++t)require(!ds41f_int8_matvec(got+t*ostride,&q,x+t*istride,group,0),"bench GEMV");
                double dt=now()-start;if(it)elapsed[mode]+=dt;
            }
            printf("INT8_BATCH rows=%zu cols=%zu group=%zu batch=%zu cold_gemv_us=%.3f cold_batch_us=%.3f speedup=%.3f guard=%g\n",rows,cols,group,batch,elapsed[0]/15*1e6,elapsed[1]/15*1e6,elapsed[0]/elapsed[1],guard);
        }
        free(flush);
    }
    for(size_t t=0;t<6;++t)ds41f_int8_input_free(prepared+t);
    ds41f_int8_free(&q);free(x);free(got);free(ref);
}
int main(int argc,char **argv)
{
    (void)argv;size_t rows[]={1,3,4,5,32,64},cols[]={32,64,128,5120},cases=0;
    for(size_t r=0;r<6;++r)for(size_t c=0;c<4;++c)for(size_t b=32;b<=128;b*=2){
        if(cols[c]%b)continue;
        check(rows[r],cols[c],b,rows[r],0);cases+=6;
        if(rows[r]==64){check(64,cols[c],b,32,0);cases+=6;}
    }
    printf("INT8_BATCH PASS cases=%zu batches=1..6 prepared strides tails grouped invalid canaries bit_exact\n",cases);
    if(argc>1){check(1280,8192,32,1280,1);check(8192,1280,32,8192,1);check(2048,4096,32,1024,1);}
    return 0;
}
