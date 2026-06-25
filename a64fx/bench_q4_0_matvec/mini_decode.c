/* mini_decode.c - minimal text-only forward harness (NO VLM) to validate +
 * time Gemma-4 31B decode through the real transformer.h forward path.
 * Feeds a fixed token sequence (no tokenizer needed), prefills, then times
 * decode steps and prints argmax tokens — used to A/B the fp32 vs int8 matvec.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *          -D_GNU_SOURCE -I../../common mini_decode.c -lm -o mini_decode
 * Run:   (mirror run_gemma4 env) ./mini_decode model.gguf [n_decode]
 */
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int argmax(const float*v,int n){ int b=0; float m=v[0]; for(int i=1;i<n;i++) if(v[i]>m){m=v[i];b=i;} return b; }

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [n_decode]\n",argv[0]); return 1; }
    int ndec=argc>2?atoi(argv[2]):16;
    int use_mmap = getenv("NO_MMAP")&&atoi(getenv("NO_MMAP"))? 0 : 1;
    fprintf(stderr,"[mini] opening %s (mmap=%d)\n",argv[1],use_mmap);
    gguf_context*g=gguf_open(argv[1],use_mmap); if(!g){ fprintf(stderr,"open failed\n"); return 1; }
    double t0=now();
    transformer_model*m=transformer_load(g,4096); if(!m){ fprintf(stderr,"load failed\n"); return 1; }
    int nthr=getenv("LLM_THREADS")?atoi(getenv("LLM_THREADS")):48;
    transformer_set_threads(m,nthr);
    transformer_numa_setup(m,g);
    if(!(getenv("TF_NO_PANEL")&&atoi(getenv("TF_NO_PANEL")))) transformer_build_panels(m);
    fprintf(stderr,"[mini] loaded in %.1fs, n_vocab=%d, threads=%d\n",now()-t0,m->n_vocab,nthr);

    /* fixed prompt token ids (arbitrary valid ids; we only need a forward path) */
    int prompt[]={2,651,6037,576,6081,603};   /* <bos> + a few tokens */
    int np=sizeof(prompt)/sizeof(prompt[0]);
    int pos=0, tok=0;
    double tp=now();
    for(int i=0;i<np;i++){ float*lg=transformer_forward_logits(m,prompt[i],pos++); if(i==np-1) tok=argmax(lg,m->n_vocab); }
    fprintf(stderr,"[mini] prefill %d tok in %.3fs; first next-tok=%d\n",np,now()-tp,tok);

    /* timed decode */
    double td=now(); int first=tok;
    for(int i=0;i<ndec;i++){ float*lg=transformer_forward_logits(m,tok,pos++); tok=argmax(lg,m->n_vocab); }
    double dt=now()-td;
    fprintf(stderr,"[mini] decode %d tok in %.3fs = %.2f tok/s (first=%d last=%d)\n",ndec,dt,ndec/dt,first,tok);
    printf("DECODE_TOK_S %.3f  first=%d last=%d\n", ndec/dt, first, tok);
    return 0;
}
