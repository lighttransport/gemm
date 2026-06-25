/* mini_prefill.c - exercise the BATCH PREFILL GEMM path (transformer_prefill_gemm
 * -> tf_gemma4_prefill_batch) on a token batch, print the per-section profile +
 * the last-token argmax. Baseline/validation harness for the Q8v2 GEMM integration:
 * argmax must stay stable when TF_Q8V2 is toggled.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *          -D_GNU_SOURCE -I../../common mini_prefill.c -lm -o mini_prefill
 * Run:   ./mini_prefill model.gguf [n_tokens]
 */
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int argmax(const float*v,int n){ int b=0; float m=v[0]; for(int i=1;i<n;i++) if(v[i]>m){m=v[i];b=i;} return b; }

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [n_tokens]\n",argv[0]); return 1; }
    int N=argc>2?atoi(argv[2]):128;
    gguf_context*g=gguf_open(argv[1],1); if(!g){ fprintf(stderr,"open failed\n"); return 1; }
    double t0=now();
    transformer_model*m=transformer_load(g,4096); if(!m){ fprintf(stderr,"load failed\n"); return 1; }
    int nthr=getenv("LLM_THREADS")?atoi(getenv("LLM_THREADS")):48;
    transformer_set_threads(m,nthr);
    transformer_numa_setup(m,g);
    if(!(getenv("TF_NO_PANEL")&&atoi(getenv("TF_NO_PANEL")))) transformer_build_panels(m);
    fprintf(stderr,"[prefill] loaded in %.1fs, n_vocab=%d threads=%d N=%d\n",now()-t0,m->n_vocab,nthr,N);

    /* deterministic token batch (valid ids) */
    int32_t *toks=(int32_t*)malloc((size_t)N*sizeof(int32_t));
    for(int i=0;i<N;i++) toks[i]=2 + (i*131+7)%(m->n_vocab>30000?30000:m->n_vocab-1);
    toks[0]=2;

    double tp=now();
    float*lg=transformer_prefill_gemm(m,toks,N,0);
    double dt=now()-tp;
    if(!lg){ fprintf(stderr,"prefill returned NULL\n"); return 1; }
    int am=argmax(lg,m->n_vocab);
    fprintf(stderr,"[prefill] %d tok in %.3fs = %.1f tok/s\n",N,dt,N/dt);
    printf("PREFILL_ARGMAX %d  logit=%.6f  tok_s=%.2f\n", am, lg[am], N/dt);
    free(toks);
    return 0;
}
