#define _POSIX_C_SOURCE 200809L
#include "ds41f_sve.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int check(size_t rows,size_t cols,int bench)
{
    size_t bytes=rows*cols/2,scales=rows*(cols/32);
    uint8_t *w=malloc(bytes*2),*s=malloc(scales*2);float *x=malloc(cols*4),*ref=malloc(rows*8),*out=malloc((rows*2+1)*4);
    if(!w||!s||!x||!ref||!out)return 1;
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<bytes*2;++i)w[i]=(uint8_t)((i*73+i/17)%256);
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<scales*2;++i)s[i]=(uint8_t)(120+i%12);
    for(size_t i=0;i<cols;++i)x[i]=(float)((int)(i%37)-18)*.03125f;
    if(ds41f_mxfp4_matvec(ref,w,s,x,rows,cols)||ds41f_mxfp4_matvec(ref+rows,w+bytes,s+scales,x,rows,cols))return 1;
    for(int tile=1;tile<=2;++tile){out[rows*2]=123456;
        if(ds41f_mxfp4_matvec_pair(out,out+rows,w,s,w+bytes,s+scales,x,rows,cols,tile))return 1;
        if(memcmp(ref,out,rows*8)||out[rows*2]!=123456){fprintf(stderr,"PAIR mismatch rows=%zu cols=%zu tile=%d\n",rows,cols,tile);return 1;}}
    if(bench){
        /* Evict the matrices between timed calls; include both complete
         * projections and retain the original compressed weight layout. */
        size_t n=16*1024*1024;float *flush=malloc(n*4);if(!flush)return 1;
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<n;++i)flush[i]=(float)(i%7);
        volatile float guard=0;
        for(int tile=0;tile<=2;++tile){double sum=0,best=1e9;
            for(int it=0;it<31;++it){float value=0;
                #pragma omp parallel for reduction(+:value) schedule(static)
                for(size_t i=0;i<n;i+=16)value+=flush[i];guard=value;
                double t=now();int rc;
                if(tile)rc=ds41f_mxfp4_matvec_pair(out,out+rows,w,s,w+bytes,s+scales,x,rows,cols,tile);
                else{rc=ds41f_mxfp4_matvec(out,w,s,x,rows,cols);rc|=ds41f_mxfp4_matvec(out+rows,w+bytes,s+scales,x,rows,cols);}
                if(rc)return 1;
                t=now()-t;if(it){sum+=t;if(t<best)best=t;}}
            printf("EXPERT_PAIR tile=%d rows=%zu cols=%zu cold_mean_us=%.3f cold_best_us=%.3f guard=%g\n",tile,rows,cols,sum/30*1e6,best*1e6,guard);
        }
        free(flush);
    }
    free(w);free(s);free(x);free(ref);free(out);return 0;
}
int main(void)
{
    size_t rows[]={1,2,3,7,32,2304},cols[]={32,128,5120};
    for(size_t r=0;r<6;++r)for(size_t c=0;c<3;++c)if(check(rows[r],cols[c],0))return 1;
    puts("EXPERT_PAIR PASS bit_exact cases=18 two_tiles tails canaries");
    return check(2304,5120,1);
}
