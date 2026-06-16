/* m3_bench.c - time the m3_impl.h forward at the current OMP_NUM_THREADS (synthetic,
 * single-process). Run at 1 vs N threads to measure the matvec parallelization.
 *   OMP_NUM_THREADS=1  ./m3_bench
 *   OMP_NUM_THREADS=48 ./m3_bench
 * Env: M3_LAYERS(12) M3_EXPERTS(16) M3_MAXPOS(512) M3_PREFILL(4) M3_DECODE(16). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define M3_IMPL
#include "m3.h"
#include "m3_impl.h"
#ifdef _OPENMP
#include <omp.h>
#endif
static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int envi(const char*k,int d){ const char*v=getenv(k); return (v&&*v)?atoi(v):d; }
int main(void){
    m3_config c=m3_default_config();
    c.n_layers=envi("M3_LAYERS",12); c.n_experts=envi("M3_EXPERTS",16); c.max_pos=envi("M3_MAXPOS",512);
    int prefill=envi("M3_PREFILL",4), decode=envi("M3_DECODE",16);
    int thr=1;
#ifdef _OPENMP
    thr=omp_get_max_threads();
#endif
    int N=envi("M3_MSTREAM",1);   /* >1 -> batched multi-stream decode (aggregate tok/s) */
    m3_model*m=m3_alloc_synth(c,0,1,thr,4);
    int nan=0, argmax=0;
    if(N>1){
        if(m3_alloc_mstream(m,N)){ printf("mstream alloc failed\n"); return 2; }
        int C=c.hidden; float*X=malloc((size_t)N*C*4); int*pos=malloc((size_t)N*sizeof(int)),*out=malloc((size_t)N*sizeof(int));
        m3_sm=0xBEEF; for(int t=0;t<N;t++) pos[t]=0;
        for(int g=0;g<decode;g++){ for(size_t i=0;i<(size_t)N*C;i++) X[i]=(float)(m3_sm_next()*0.2-0.1);
            m3_forward_batch_decode(m,X,N,pos,out); for(int t=0;t<N;t++) pos[t]++; if(g==decode-1) argmax=out[0];
            for(size_t i=0;i<(size_t)N*C;i++) if(!(X[i]==X[i]))nan++; }
        double t0=now_sec();
        for(int g=0;g<decode;g++){ for(size_t i=0;i<(size_t)N*C;i++) X[i]=(float)(m3_sm_next()*0.2-0.1);
            m3_forward_batch_decode(m,X,N,pos,out); for(int t=0;t<N;t++) pos[t]++; }
        double dt=now_sec()-t0;
        printf("threads=%d N=%d layers=%d experts=%d  steps=%d  %.1f ms/step  agg %.2f tok/s  per-stream %.2f  argmax=%d NaNs=%d\n",
               thr,N,c.n_layers,c.n_experts,decode,dt/decode*1e3,(double)decode*N/dt,(double)decode/dt,argmax,nan);
        free(X);free(pos);free(out);m3_free(m); return nan==0?0:1;
    }
    float*x=malloc((size_t)c.hidden*4);
    m3_sm=0xBEEF; for(int p=0;p<prefill;p++){ for(int i=0;i<c.hidden;i++) x[i]=(float)(m3_sm_next()*2-1); argmax=m3_forward_token(m,x,p); }
    double t0=now_sec();
    for(int g=0;g<decode;g++){ int pos=prefill+g; for(int i=0;i<c.hidden;i++) x[i]=(float)(m3_sm_next()*2-1); argmax=m3_forward_token(m,x,pos); for(int i=0;i<c.hidden;i++) if(!(x[i]==x[i]))nan++; }
    double dt=now_sec()-t0;
    printf("threads=%d N=1 layers=%d experts=%d  decode %d tok  %.1f ms/tok  %.2f tok/s  argmax=%d NaNs=%d\n",
           thr,c.n_layers,c.n_experts,decode,dt/decode*1e3,decode/dt,argmax,nan);
    free(x); m3_free(m); return nan==0?0:1;
}
