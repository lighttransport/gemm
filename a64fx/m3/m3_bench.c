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
    m3_model*m=m3_alloc_synth(c,0,1,thr,4);
    float*x=malloc((size_t)c.hidden*4); int nan=0, argmax=0;
    m3_sm=0xBEEF; for(int p=0;p<prefill;p++){ for(int i=0;i<c.hidden;i++) x[i]=(float)(m3_sm_next()*2-1); argmax=m3_forward_token(m,x,p); }
    double t0=now_sec();
    for(int g=0;g<decode;g++){ int pos=prefill+g; for(int i=0;i<c.hidden;i++) x[i]=(float)(m3_sm_next()*2-1); argmax=m3_forward_token(m,x,pos); for(int i=0;i<c.hidden;i++) if(!(x[i]==x[i]))nan++; }
    double dt=now_sec()-t0;
    printf("threads=%d layers=%d experts=%d  decode %d tok  %.1f ms/tok  %.2f tok/s  argmax=%d NaNs=%d\n",
           thr,c.n_layers,c.n_experts,decode,dt/decode*1e3,decode/dt,argmax,nan);
    free(x); m3_free(m); return nan==0?0:1;
}
