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
    int use_mmap = (getenv("NO_MMAP")&&atoi(getenv("NO_MMAP")))?0:1;
    fprintf(stderr,"[prefill] mmap=%d\n",use_mmap);
    gguf_context*g=gguf_open(argv[1],use_mmap); if(!g){ fprintf(stderr,"open failed\n"); return 1; }
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

    /* A/B in one load: warm-up, then fp32 timed, then Q8v2 timed. tf_q8v2_enable
     * is a transformer.h global we flip between passes (clean single-process compare). */
#ifdef TF_HAVE_Q8V2
    extern int tf_q8v2_enable;
    int ab = getenv("TF_AB")?atoi(getenv("TF_AB")):0;
    if(ab){
        tf_q8v2_enable=0; transformer_prefill_gemm(m,toks,N,0);   /* warm-up (fp32) */
        double t1=now(); float*l0=transformer_prefill_gemm(m,toks,N,0); double d0=now()-t1; int a0=argmax(l0,m->n_vocab);
        fprintf(stderr,">>> AB fp32:  %.3fs argmax=%d\n",d0,a0);
        tf_q8v2_enable=1;
        double t2=now(); float*l1=transformer_prefill_gemm(m,toks,N,0); double d1=now()-t2; int a1=argmax(l1,m->n_vocab);
        fprintf(stderr,">>> AB q8v2:  %.3fs argmax=%d\n",d1,a1);
        printf("AB N=%d fp32=%.3fs q8v2=%.3fs speedup=%.2fx argmax_fp32=%d argmax_q8=%d %s\n",
               N,d0,d1,d0/d1,a0,a1,a0==a1?"MATCH":"DIFFER");
        return 0;
    }
#endif
    int npass = getenv("PREFILL_PASSES")?atoi(getenv("PREFILL_PASSES")):1;
    int am=0; float*lg=NULL;
    for(int p=0;p<npass;p++){
        double tp=now();
        lg=transformer_prefill_gemm(m,toks,N,0);
        double dt=now()-tp;
        if(!lg){ fprintf(stderr,"prefill returned NULL\n"); return 1; }
        am=argmax(lg,m->n_vocab);
        fprintf(stderr,"[prefill pass %d] %d tok in %.3fs = %.1f tok/s  argmax=%d\n",p,N,dt,N/dt,am);
    }
    printf("PREFILL_ARGMAX %d  logit=%.6f\n", am, lg[am]);
    free(toks);
    return 0;
}
