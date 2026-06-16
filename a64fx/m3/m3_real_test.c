/*
 * m3_real_test.c - single-node REAL-weight load + forward (no MPI/uTofu).
 *
 * Validates the stager + m3_load_real + forward end-to-end on actual M3 weights,
 * runnable on one native A64FX node (no multi-rank uTofu needed). Use a truncated
 * layer count so the ep_size=1 blob (owns ALL 128 experts) fits /local + HBM.
 *
 *   # stage layers 0..L-1 to /local/m3 (rank 0 of 1; shards 1-3 hold layers 0-3):
 *   M3_EP_RANK=0 M3_EP_SIZE=1 M3_STAGE_LAYERS=4 M3_SHARD_LIMIT=3 M3_STAGE_DIR=/local/m3 ./m3_stage
 *   # load + forward:
 *   M3_LAYERS=4 M3_MAXPOS=256 M3_STAGE_DIR=/local/m3 ./m3_real_test
 *
 * Truncated model => generated text is garbage; the gate is: load OK (all tensors
 * present, shapes match), NaNs=0, a stable argmax. MSA/TP can be toggled via env.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define M3_IMPL
#include "m3.h"
#include "m3_impl.h"

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int envi(const char*k,int d){ const char*v=getenv(k); return (v&&*v)?atoi(v):d; }

int main(void){
    m3_config c=m3_default_config();
    int L=envi("M3_LAYERS",4); if(L>0) c.n_layers=L;
    c.max_pos=envi("M3_MAXPOS",256);
    int prefill=envi("M3_PREFILL",8), decode=envi("M3_DECODE",8);
    const char*dir=getenv("M3_STAGE_DIR");
    printf("=== M3 real-weight test: layers=%d experts=%d max_pos=%d dir=%s ===\n",
           c.n_layers,c.n_experts,c.max_pos,dir?dir:"/local/m3");
    double t0=now_sec();
    m3_model*m=m3_load_real(c,0,1,dir,1,1);
    if(!m){ printf("FAIL: m3_load_real returned NULL (missing/short tensors)\n"); return 1; }
    printf("load OK: %.1fs  arena_used=%.2f GB\n", now_sec()-t0, m->arena_used/1e9);

    float*x=malloc((size_t)c.hidden*4); int nan=0, argmax=0; double xn=0;
    int pchunk=envi("M3_PCHUNK",0);   /* Lever 1: chunked batched prefill (M=pchunk) */
    /* deterministic pseudo-embeddings as input activations (no tokenizer here) */
    m3_sm=0xABCDEF;
    double tp=now_sec();
    if(pchunk>0){
        if(m3_alloc_mstream_ex(m,pchunk,0)){ printf("FAIL: alloc prefill\n"); return 1; }
        float*X=malloc((size_t)prefill*c.hidden*4);
        for(int p=0;p<prefill;p++) for(int i=0;i<c.hidden;i++) X[(size_t)p*c.hidden+i]=(float)(m3_sm_next()*0.2-0.1);
        for(int p0=0;p0<prefill;p0+=pchunk){ int S=prefill-p0; if(S>pchunk)S=pchunk; argmax=m3_forward_prefill_chunk(m,X+(size_t)p0*c.hidden,S,p0); }
        for(size_t i=0;i<(size_t)prefill*c.hidden;i++){ if(!(X[i]==X[i]))nan++; xn+=(double)X[i]*X[i]; }
        free(X); m3_free_mstream(m);
        printf("prefill(chunked M=%d) argmax=%d  ||x||=%.3e\n", pchunk, argmax, sqrt(xn));
    } else
    for(int p=0;p<prefill;p++){ for(int i=0;i<c.hidden;i++) x[i]=(float)(m3_sm_next()*0.2-0.1);
        argmax=m3_forward_token(m,x,p); for(int i=0;i<c.hidden;i++){ if(!(x[i]==x[i]))nan++; xn+=(double)x[i]*x[i]; } }
    if(pchunk<=0) printf("prefill(token-serial) argmax=%d  ||x||=%.3e\n", argmax, sqrt(xn));
    double pf=now_sec()-tp;
    double td=now_sec(); int last=argmax;
    for(int g=0;g<decode;g++){ int pos=prefill+g; for(int i=0;i<c.hidden;i++) x[i]=(float)(m3_sm_next()*0.2-0.1);
        last=m3_forward_token(m,x,pos); for(int i=0;i<c.hidden;i++) if(!(x[i]==x[i]))nan++; }
    double dt=now_sec()-td;
    printf("prefill %d tok %.2f tok/s  decode %d tok %.2f tok/s\n", prefill, prefill/pf, decode, decode/dt);
    printf("last argmax=%d  ||x||=%.3e  NaNs=%d  %s\n", last, sqrt(xn), nan, nan==0?"OK":"FAIL");
    free(x); m3_free(m);
    return nan==0?0:1;
}
