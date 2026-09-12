#define _POSIX_C_SOURCE 200809L
#include "ds41f_sve.h"
#include "ds41f_fp4_sdot.h"
#include "ds41f_team.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
static void require(int ok,const char *msg)
{if(!ok){fprintf(stderr,"FP4_BATCH FAIL %s\n",msg);exit(1);}}
static double now(void)
{struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
typedef struct {float *out;size_t stride,rows,cols,batch;uint8_t *w,*s;float *x;size_t xs;ds41f_int8_input *in;int packed,rc;} job;
static void run(void *arg)
{job *j=arg;j->rc=j->packed?ds41f_mxfp4_sdot_matmul(j->out,j->stride,j->w,j->s,j->in,j->rows,j->cols,j->batch):ds41f_mxfp4_matmul(j->out,j->stride,j->w,j->s,j->x,j->xs,j->rows,j->cols,j->batch);}
static void check(size_t rows,size_t cols,int bench)
{
    size_t xs=cols+16,os=rows+4,bytes=rows*cols/2,scales=rows*cols/32;
    uint8_t *w=malloc(bytes),*p=malloc(bytes),*s=malloc(scales),*ps=malloc(scales);
    float *x=malloc(6*xs*4),*out=malloc(7*os*4),*ref=malloc(6*os*4);ds41f_int8_input in[6]={{0}};
    require(w&&p&&s&&ps&&x&&out&&ref,"allocation");
    for(size_t i=0;i<bytes;++i)w[i]=(uint8_t)(i*73+i/17);
    for(size_t i=0;i<scales;++i)s[i]=(i%23)?(uint8_t)(117+i%18):0;
    for(size_t t=0;t<6;++t){for(size_t i=0;i<xs;++i)x[t*xs+i]=i>=cols?12345:t==0?0:sinf((i+t*83)*.13f)*.75f;
        require(!ds41f_int8_prepare_input(in+t,x+t*xs,cols,32),"input");}
    require(!ds41f_mxfp4_pack_sdot(p,ps,w,s,rows,cols),"pack");
    for(int packed=0;packed<2;++packed){
        for(size_t t=0;t<6;++t)require(!(packed?ds41f_mxfp4_sdot_prepared(ref+t*os,p,ps,in+t,rows,cols,0):ds41f_mxfp4_matvec(ref+t*os,w,s,x+t*xs,rows,cols)),"control");
        for(size_t n=1;n<=6;++n)for(int team=0;team<2;++team){
            for(size_t i=0;i<7*os;++i)out[i]=12345;
            job j={out,os,rows,cols,n,packed?p:w,packed?ps:s,x,xs,in,packed,0};
            if(team)require(!ds41f_team_run(run,&j),"team");else run(&j);require(!j.rc,"batch");
            for(size_t t=0;t<n;++t)require(!memcmp(out+t*os,ref+t*os,rows*4),"bit exact");
            for(size_t t=0;t<7;++t)for(size_t r=t<n?rows:0;r<os;++r)require(out[t*os+r]==12345,"canary");
        }
        if(bench){size_t nf=16*1024*1024;float *flush=malloc(nf*4);require(flush!=NULL,"flush");
            for(size_t i=0;i<nf;++i)flush[i]=(float)(i%7);volatile float guard=0;
            for(size_t n=2;n<=6;n+=2){double elapsed[2]={0,0};
                for(int mode=0;mode<2;++mode)for(int it=0;it<11;++it){float sum=0;
                    #pragma omp parallel for reduction(+:sum) schedule(static)
                    for(size_t i=0;i<nf;i+=16)sum+=flush[i];guard=sum;double start=now();
                    if(mode){job j={out,os,rows,cols,n,packed?p:w,packed?ps:s,x,xs,in,packed,0};run(&j);require(!j.rc,"bench batch");}
                    else for(size_t t=0;t<n;++t)require(!(packed?ds41f_mxfp4_sdot_prepared(out+t*os,p,ps,in+t,rows,cols,0):ds41f_mxfp4_matvec(out+t*os,w,s,x+t*xs,rows,cols)),"bench control");
                    if(it)elapsed[mode]+=now()-start;
                }
                printf("FP4_BATCH_BENCH rows=%zu cols=%zu packed=%d batch=%zu sequential_us=%.3f batch_us=%.3f speedup=%.3f guard=%g\n",rows,cols,packed,n,elapsed[0]*1e5,elapsed[1]*1e5,elapsed[0]/elapsed[1],guard);
            }free(flush);
        }
    }
    s[0]=255;
    require(!ds41f_mxfp4_matmul(out,os,w,s,x,xs,rows,cols,6),"NaN scale batch");
    for(size_t t=0;t<6;++t){require(!ds41f_mxfp4_matvec(ref+t*os,w,s,x+t*xs,rows,cols),"NaN scale control");
        require(isnan(out[t*os])&&isnan(ref[t*os]),"NaN scale propagation");}
    for(size_t t=0;t<6;++t)ds41f_int8_input_free(in+t);
    free(w);free(p);free(s);free(ps);free(x);free(out);free(ref);
}
int main(int argc,char **argv)
{(void)argv;size_t rows[]={4,8,28,128},cols[]={32,64,160,5120};
    for(size_t r=0;r<4;++r)for(size_t c=0;c<4;++c)check(rows[r],cols[c],0);
    puts("FP4_BATCH PASS cases=384 bit_exact strides canaries zero subnormal_scale NaN_scale persistent_team");
    if(argc>1){check(2304,5120,1);check(5120,2304,1);}return 0;}
