#define _POSIX_C_SOURCE 200809L
#include "ds41f_int8.h"
#include "ds41f_sve.h"
#include "ds41f_tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void)
{struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char **argv)
{
    if(argc!=7){fprintf(stderr,"usage: %s STAGE BASE ROWS COLS BLOCK GROUP_ROWS\n",argv[0]);return 2;}
    size_t rows=strtoul(argv[3],NULL,10),cols=strtoul(argv[4],NULL,10);
    size_t block=strtoul(argv[5],NULL,10),group=strtoul(argv[6],NULL,10);
    if(!rows||!cols||!group||rows%group)return 2;
    char name[256];uint8_t *w=NULL,*s=NULL;double start=now();
    snprintf(name,sizeof name,"%s.weight",argv[2]);
    if(ds41f_tensor_load(argv[1],name,rows*cols,(void **)&w))return 2;
    snprintf(name,sizeof name,"%s.scale",argv[2]);
    if(ds41f_tensor_load(argv[1],name,((rows+31)/32)*((cols+31)/32),(void **)&s))return 2;
    double load=now()-start;ds41f_int8 q;start=now();
    if(ds41f_int8_from_fp8(&q,w,s,rows,cols,block))return 1;
    double convert=now()-start;
    float *x=malloc((rows/group)*cols*sizeof(float)),*ref=malloc(rows*sizeof(float));
    float *got=malloc(rows*sizeof(float)),*integer_ref=malloc(rows*sizeof(float));
    if(!x||!ref||!got||!integer_ref)return 2;
    for(size_t i=0;i<(rows/group)*cols;++i)x[i]=sinf((float)i*.137f)+.3f*cosf((float)i*.071f);
    for(int i=0;i<2;++i)if(ds41f_fp8_grouped_matvec(ref,w,s,x,rows/group,group,cols))return 1;
    start=now();for(int i=0;i<5;++i)if(ds41f_fp8_grouped_matvec(ref,w,s,x,rows/group,group,cols))return 1;
    double fp8=(now()-start)/5;
    for(int i=0;i<2;++i)if(ds41f_int8_matvec(got,&q,x,group,0))return 1;
    start=now();for(int i=0;i<20;++i)if(ds41f_int8_matvec(got,&q,x,group,0))return 1;
    double int8=(now()-start)/20;
    if(ds41f_int8_matvec(integer_ref,&q,x,group,1))return 1;
    double aa=0,bb=0,ab=0,err=0,kerr=0;float maxabs=0;
    for(size_t i=0;i<rows;++i){double a=ref[i],b=got[i],d=a-b;
        aa+=a*a;bb+=b*b;ab+=a*b;err+=d*d;d=b-integer_ref[i];kerr+=d*d;
        maxabs=fmaxf(maxabs,fabsf(got[i]-integer_ref[i]));}
    printf("INT8_BENCH base=%s rows=%zu cols=%zu block=%zu load_s=%.6f convert_s=%.6f fp8_ms=%.6f int8_ms=%.6f speedup=%.3f cosine=%.9f rel_rms=%.9g integer_rel_rms=%.9g integer_max_abs=%.9g bytes=%zu\n",
        argv[2],rows,cols,block,load,convert,fp8*1e3,int8*1e3,fp8/int8,ab/sqrt(aa*bb),sqrt(err/aa),sqrt(kerr/bb),maxabs,q.bytes);
    int rc=!isfinite(err)||!isfinite(kerr)||sqrt(kerr/bb)>1e-5;
    ds41f_int8_free(&q);free(w);free(s);free(x);free(ref);free(got);free(integer_ref);return rc;
}
