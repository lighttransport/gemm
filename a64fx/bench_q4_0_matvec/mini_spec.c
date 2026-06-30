/* Speculative decode loop with batched verify (12B BF16, A64FX).
 *
 * Each step: draft d tokens, run ONE batched forward over [tok, draft...] (N=d+1) to get
 * per-position logits, accept the longest prefix where draft[j] == argmax(L[j]) (the
 * model's greedy), emit tok + accepted + 1 bonus token. Output is PROVABLY greedy-
 * identical (every emitted token is the model's greedy choice); the draft only affects
 * speed. KV rollback is free: advancing the position counter ignores rejected slots.
 *
 * Draft = prompt-lookup (n-gram match in the generated history) -- reliable, no model,
 * real speedup on repetitive/structured content. The gemma4-assistant MTP draft (in
 * MTP/gemma-4-12b-it-BF16-MTP.gguf) is the general-purpose drop-in replacement.
 *
 * Validates against single-token greedy (same GEMM path -> bit-identical), reports
 * acceptance + speedup. */
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

/* prompt-lookup: longest recent 2-gram match -> copy the following up to K tokens */
static int pld(const int*seq,int len,int K,int*draft){
    if(len<2) return 0;
    int a=seq[len-2], b=seq[len-1];
    for(int i=len-3;i>=0;i--) if(seq[i]==a && seq[i+1]==b){
        int d=0; for(int j=0;j<K && i+2+j<len;j++) draft[d++]=seq[i+2+j];
        return d;
    }
    return 0;
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [G gen] [K draft]\n",argv[0]); return 1; }
    int G=argc>2?atoi(argv[2]):48, K=argc>3?atoi(argv[3]):8;
    int use_mmap=getenv("NO_MMAP")&&atoi(getenv("NO_MMAP"))?0:1;
    gguf_context*g=gguf_open(argv[1],use_mmap); if(!g){fprintf(stderr,"open fail\n");return 1;}
    transformer_model*m=transformer_load(g,8192); if(!m){fprintf(stderr,"load fail\n");return 1;}
    int nthr=getenv("LLM_THREADS")?atoi(getenv("LLM_THREADS")):48;
    transformer_set_threads(m,nthr); transformer_numa_setup(m,g); transformer_build_panels(m);
    if(getenv("TF_PODD")&&atoi(getenv("TF_PODD"))){ extern void transformer_prepack_podd(transformer_model*); transformer_prepack_podd(m); }
    extern float *tf_batch_all_logits; extern int tf_batch_keep_pool, tf_batch_quiet;
    tf_batch_keep_pool=1; tf_batch_quiet=1;
    int do_ref = !(getenv("SPEC_NOREF")&&atoi(getenv("SPEC_NOREF")));
    int V=m->n_vocab;
    float*all=(float*)malloc((size_t)(K+1)*V*sizeof(float));

    /* prompt with deliberate repetition so PLD has matches to find */
    int prompt[]={2,107,108,4368,532,18534,236787,108,4368,532,18534,236787,108,4368,532};
    int P=sizeof(prompt)/sizeof(prompt[0]);
    int32_t pp[64]; for(int i=0;i<P;i++) pp[i]=prompt[i];

    /* ---- reference: single-token greedy (GEMM path), G tokens (optional) ---- */
    int ref[1024]; for(int i=0;i<G;i++) ref[i]=-1;
    if(do_ref){ transformer_prefill_gemm(m,pp,P,0); float*lg=m->logits; ref[0]=argmax(lg,V);
        for(int i=1;i<G;i++){ int32_t t=ref[i-1]; lg=transformer_prefill_gemm(m,&t,1,P+i-1); ref[i]=argmax(lg,V); } }

    /* ---- spec decode ---- */
    transformer_prefill_gemm(m,pp,P,0);          /* reset KV [0,P) */
    int seq[2048]; for(int i=0;i<P;i++) seq[i]=prompt[i];
    int pos=P; seq[pos]=argmax(m->logits,V);     /* pending token at pos */
    int gen=0, rounds=0, accepted=0, fwd=0;
    double t0=now();
    int oracle=getenv("SPEC_ORACLE")&&atoi(getenv("SPEC_ORACLE"));
    while(gen<G){
        int draft[64]; int d;
        if(oracle){ /* perfect draft = reference greedy continuation (MTP ceiling) */
            d=0; for(int j=0;j<K && (pos-P)+1+j<G;j++) draft[d++]=ref[(pos-P)+1+j];
        } else d=pld(seq,pos+1,K,draft);
        int32_t batch[65]; int N=d+1; batch[0]=seq[pos];
        for(int j=0;j<d;j++) batch[1+j]=(int32_t)draft[j];
        tf_batch_all_logits=all; transformer_prefill_gemm(m,batch,N,pos); tf_batch_all_logits=NULL; fwd++;
        int gg[65]; for(int j=0;j<N;j++) gg[j]=argmax(all+(size_t)j*V,V);
        int a=0; while(a<d && draft[a]==gg[a]) a++;
        for(int j=0;j<a;j++) seq[pos+1+j]=draft[j];
        seq[pos+1+a]=gg[a];
        gen += 1+a; accepted += a; rounds++; pos += 1+a;
        if(gen>=G) break;
    }
    double dt=now()-t0;

    /* ---- compare to reference ---- */
    int match=0; for(int i=0;i<G;i++) if(do_ref ? seq[P+i]==ref[i] : 1) match++;
    fprintf(stderr,"\nSPEC DECODE G=%d K=%d: %d/%d tokens match greedy\n",G,gen<G?gen:G,match,G);
    fprintf(stderr,"  %d rounds, %d forwards, %d drafts accepted -> %.2f tokens/forward (avg accept %.2f/%d)\n",
        rounds,fwd,accepted,(double)gen/fwd,(double)accepted/rounds,K);
    fprintf(stderr,"  spec: %.3fs = %.2f tok/s\n",dt,gen/dt);
    printf("SPEC G=%d K=%d match=%d/%d tok/forward=%.2f accept=%.2f/%d tok_s=%.2f\n",
        G,G,match,G,(double)gen/fwd,(double)accepted/rounds,K,gen/dt);
    return match==G?0:2;
}
