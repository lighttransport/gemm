/* gemma4_sn_batched.c - SINGLE-NODE batched-prefill reference for the TP-batched check.
 * Loads the FULL model (NO_MMAP anon, fadvise = memory-safe), batched-prefills a prompt,
 * prints argmax(last-token logits). This is the gold reference the TP-batched path must
 * match. Run on a DEDICATED compute node (24GB load): mpiexec -np 1 ... gemma4_sn_batched.
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *        -I../../common gemma4_sn_batched.c -lm -lpthread -o gemma4_sn_batched
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [M]\n",argv[0]); return 1; }
    int M=argc>2?atoi(argv[2]):2;
    int use_mmap=getenv("NO_MMAP")&&atoi(getenv("NO_MMAP"))?0:1;
    setenv("NUMA_INTERLEAVE","1",0);
    gguf_context*g=gguf_open(argv[1],use_mmap); if(!g){fprintf(stderr,"open failed\n");return 1;}
    transformer_model*m=transformer_load(g,2048); if(!m){fprintf(stderr,"load failed\n");return 1;}
    transformer_set_threads(m,getenv("LLM_THREADS")?atoi(getenv("LLM_THREADS")):48);

    int32_t prompt[]={2,651};
    int P=sizeof(prompt)/sizeof(prompt[0]);
    /* same correctness probe as the TP runner: batched-prefill the prompt, argmax(last) */
    float*lg=transformer_prefill_gemm(m,prompt,P,0);
    if(!lg){ fprintf(stderr,"prefill_gemm NULL (PLE?)\n"); return 1; }
    int a=0; float mx=lg[0]; for(int i=1;i<m->n_vocab;i++) if(lg[i]>mx){mx=lg[i];a=i;}
    printf("SN_BATCHED prompt={2,651} prompt_next_argmax=%d\n",a);

    /* also an M-token batched forward (timing parity w/ the TP runner, value-irrelevant) */
    if(M>2){ int32_t*bt=malloc(M*sizeof(int32_t)); for(int i=0;i<M;i++) bt[i]=prompt[i%P];
             transformer_prefill_gemm(m,bt,M,0); free(bt); }
    return 0;
}
