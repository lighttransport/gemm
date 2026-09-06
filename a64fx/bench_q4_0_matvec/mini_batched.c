/* Batched-decode validation + throughput: greedily generate K tokens via single-token
 * GEMM forwards (reference), reset KV, then verify ONE batched forward (N=K) reproduces
 * the same per-position argmax. Both use the prefill-batch GEMM path so they are
 * M-independent (no GEMM-vs-matvec reassociation). Reports per-position match + speedup. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static int argmax(const float*v,int n){ int b=0; float m=v[0]; for(int i=1;i<n;i++) if(v[i]>m){m=v[i];b=i;} return b; }

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [K]\n",argv[0]); return 1; }
    int K=argc>2?atoi(argv[2]):16;
    int use_mmap=getenv("NO_MMAP")&&atoi(getenv("NO_MMAP"))?0:1;
    gguf_context*g=gguf_open(argv[1],use_mmap); if(!g){fprintf(stderr,"open failed\n");return 1;}
    transformer_model*m=transformer_load(g,4096); if(!m){fprintf(stderr,"load failed\n");return 1;}
    int nthr=getenv("LLM_THREADS")?atoi(getenv("LLM_THREADS")):48;
    transformer_set_threads(m,nthr); transformer_numa_setup(m,g); transformer_build_panels(m);
    extern float *tf_batch_all_logits;

    int prompt[]={2,651,6037,576,6081,603,573,1080};
    int P=sizeof(prompt)/sizeof(prompt[0]);
    int32_t pp[8]; for(int i=0;i<P;i++) pp[i]=prompt[i];

    /* 1. prefill prompt -> KV [0,P), logits predict S[0] */
    float*lg=transformer_prefill_gemm(m,pp,P,0);
    int32_t S[256]; int ref_arg[256];
    S[0]=argmax(lg,m->n_vocab);

    /* 2. reference: K single-token GEMM forwards, record argmax + time */
    double tsingle=now();
    for(int t=0;t<K;t++){ int32_t tok=S[t]; lg=transformer_prefill_gemm(m,&tok,1,P+t);
        ref_arg[t]=argmax(lg,m->n_vocab); if(t+1<256) S[t+1]=ref_arg[t]; }
    tsingle=now()-tsingle;

    /* 3. reset KV to [0,P) by re-prefilling the prompt */
    transformer_prefill_gemm(m,pp,P,0);

    /* 4. batched verify: ONE forward of S[0..K-1] at [P,P+K), all-N logits */
    float*all=(float*)malloc((size_t)K*m->n_vocab*sizeof(float));
    tf_batch_all_logits=all;
    double tbatch=now();
    transformer_prefill_gemm(m,S,K,P);
    tbatch=now()-tbatch;
    tf_batch_all_logits=NULL;

    /* 5. compare per-position argmax */
    int match=0; for(int t=0;t<K;t++){ int ba=argmax(all+(size_t)t*m->n_vocab,m->n_vocab); if(ba==ref_arg[t]) match++;
        if(t<6) fprintf(stderr,"  pos %d: single=%d batched=%d %s\n",P+t,ref_arg[t],ba,ba==ref_arg[t]?"OK":"DIFF"); }
    fprintf(stderr,"\nBATCHED DECODE K=%d: %d/%d positions match\n",K,match,K);
    fprintf(stderr,"  single-token: %.3fs (%.2f tok/s)   batched: %.3fs (%.2f tok/s)   speedup %.2fx\n",
        tsingle,K/tsingle,tbatch,K/tbatch,tsingle/tbatch);
    printf("BATCHED K=%d match=%d/%d single=%.2f batched=%.2f tok/s speedup=%.2fx\n",
        K,match,K,K/tsingle,K/tbatch,tsingle/tbatch);
    return match==K?0:2;
}
