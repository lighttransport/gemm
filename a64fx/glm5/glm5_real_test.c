/*
 * glm5_real_test.c - single-node REAL-weight load + forward (no MPI/uTofu).
 *
 * Validates the stager + glm5_load_real + forward end-to-end on actual GLM5 weights,
 * runnable on one native A64FX node (no multi-rank uTofu needed). Use a truncated
 * layer count so the ep_size=1 blob (owns ALL 256 experts) fits /local + HBM.
 *
 *   # stage layers 0..L-1 to /local/glm5 (rank 0 of 1; shards 1-3 hold layers 0-3):
 *   GLM5_EP_RANK=0 GLM5_EP_SIZE=1 GLM5_STAGE_LAYERS=4 GLM5_SHARD_LIMIT=3 GLM5_STAGE_DIR=/local/glm5 ./glm5_stage
 *   # load + forward:
 *   GLM5_LAYERS=4 GLM5_MAXPOS=256 GLM5_STAGE_DIR=/local/glm5 ./glm5_real_test
 *
 * Truncated model => generated text is garbage; the gate is: load OK (all tensors
 * present, shapes match), NaNs=0, a stable argmax. MSA/TP can be toggled via env.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define GLM5_IMPL
#include "glm5.h"
#include "glm5_impl.h"

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int envi(const char*k,int d){ const char*v=getenv(k); return (v&&*v)?atoi(v):d; }

int main(void){
    glm5_config c=glm5_default_config();
    int L=envi("GLM5_LAYERS",4); if(L>0) c.n_layers=L;
    c.max_pos=envi("GLM5_MAXPOS",256);
    int prefill=envi("GLM5_PREFILL",8), decode=envi("GLM5_DECODE",8);
    const char*dir=getenv("GLM5_STAGE_DIR");
    printf("=== GLM5 real-weight test: layers=%d experts=%d max_pos=%d dir=%s ===\n",
           c.n_layers,c.n_experts,c.max_pos,dir?dir:"/local/glm5");
    double t0=now_sec();
    glm5_model*m=glm5_load_real(c,0,1,dir,1,1);
    if(!m){ printf("FAIL: glm5_load_real returned NULL (missing/short tensors)\n"); return 1; }
    printf("load OK: %.1fs  arena_used=%.2f GB\n", now_sec()-t0, m->arena_used/1e9);

    float*x=glm5_amalloc((size_t)c.hidden*4); int nan=0, argmax=0; double xn=0;
    int pchunk=envi("GLM5_PCHUNK",0);   /* Lever 1: chunked batched prefill (M=pchunk) */
    /* deterministic pseudo-embeddings as input activations (no tokenizer here) */
    glm5_sm=0xABCDEF;
    double tp=now_sec();
    if(pchunk>0){
        if(glm5_alloc_mstream_ex(m,pchunk,0)){ printf("FAIL: alloc prefill\n"); return 1; }
        float*X=glm5_amalloc((size_t)prefill*c.hidden*4);
        for(int p=0;p<prefill;p++) for(int i=0;i<c.hidden;i++) X[(size_t)p*c.hidden+i]=(float)(glm5_sm_next()*0.2-0.1);
        for(int p0=0;p0<prefill;p0+=pchunk){ int S=prefill-p0; if(S>pchunk)S=pchunk; argmax=glm5_forward_prefill_chunk(m,X+(size_t)p0*c.hidden,S,p0); }
        for(size_t i=0;i<(size_t)prefill*c.hidden;i++){ if(!(X[i]==X[i]))nan++; xn+=(double)X[i]*X[i]; }
        glm5_afree(X); glm5_free_mstream(m);
        printf("prefill(chunked M=%d) argmax=%d  ||x||=%.3e\n", pchunk, argmax, sqrt(xn));
    } else
    for(int p=0;p<prefill;p++){ for(int i=0;i<c.hidden;i++) x[i]=(float)(glm5_sm_next()*0.2-0.1);
        argmax=glm5_forward_token(m,x,p); for(int i=0;i<c.hidden;i++){ if(!(x[i]==x[i]))nan++; xn+=(double)x[i]*x[i]; } }
    if(pchunk<=0) printf("prefill(token-serial) argmax=%d  ||x||=%.3e\n", argmax, sqrt(xn));
    double pf=now_sec()-tp;
    double td=now_sec(); int last=argmax;
    for(int g=0;g<decode;g++){ int pos=prefill+g; for(int i=0;i<c.hidden;i++) x[i]=(float)(glm5_sm_next()*0.2-0.1);
        last=glm5_forward_token(m,x,pos); for(int i=0;i<c.hidden;i++) if(!(x[i]==x[i]))nan++; }
    double dt=now_sec()-td;
    printf("prefill %d tok %.2f tok/s  decode %d tok %.2f tok/s\n", prefill, prefill/pf, decode, decode/dt);
    printf("last argmax=%d  ||x||=%.3e  NaNs=%d  %s\n", last, sqrt(xn), nan, nan==0?"OK":"FAIL");
    glm5_afree(x); glm5_free(m);
    return nan==0?0:1;
}
