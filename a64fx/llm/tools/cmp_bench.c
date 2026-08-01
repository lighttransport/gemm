/* Tier-B2 compressor matvec-dispatch microbench (single node, no alloc).
 *
 * tb2lcmp is 5.2 ms/tok of decode (biggest tb2prep sub-phase). This attributes it: the
 * layer compressor runs ONE ds4f_cmpmv_bf16 pool dispatch per ratio!=0 layer (41/token),
 * W=2*kv_lora=1024 rows x dim=4096 bf16 (~8.4 MB weight). Cold-HBM streamed (41 distinct
 * weights cycled). Compares: empty dispatch (barrier floor), the real cmpmv dispatch, and
 * a 2-CMG (24T) dispatch (does the tiny W=1024 barrier-starve at 48T?).
 *
 * Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *       -I../../common -o build/cmp_bench tools/cmp_bench.c -lm -lpthread -lhwb
 *   DS4F_FLAGBAR=1 taskset -c 12-59 ./build/cmp_bench [nthr=48] [ntok=64]
 */
#include "ds4f.h"
#include <math.h>
static uint32_t rng=0xA5A5u; static inline uint32_t nr(void){rng=rng*1664525u+1013904223u;return rng;}
static inline float frand(void){return ((float)(nr()>>8)/(float)(1u<<24))*2.f-1.f;}
typedef struct { uint16_t *p; size_t n; } tt; static void tw(void*a,int t,int nt){ tt*T=a; size_t i0=T->n*t/nt,i1=T->n*(t+1)/nt; memset(T->p+i0,0,(i1-i0)*2);}
static void empty_w(void*a,int t,int nt){(void)a;(void)t;(void)nt;}

int main(int argc,char**argv){
    int nthr=argc>1?atoi(argv[1]):48, ntok=argc>2?atoi(argv[2]):64;
    int W=1024, dim=4096, NL=41;   /* 41 ratio!=0 layers, W=2*kv_lora, dim=hidden */
    ds4f_model m; memset(&m,0,sizeof m); m.n_threads=nthr; m.n_cmgs=4; m.pool=ds4f_pool_start(nthr,4);
    printf("cmp_bench nthr=%d ntok=%d W=%d dim=%d NL=%d (%.1f MB/layer weight)\n",nthr,ntok,W,dim,NL,(double)W*dim*2*2/1e6);
    uint16_t **wkv=malloc(NL*sizeof(void*)), **wgate=malloc(NL*sizeof(void*));
    for(int l=0;l<NL;l++){
        wkv[l]=mmap(NULL,(size_t)W*dim*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        wgate[l]=mmap(NULL,(size_t)W*dim*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        tt A={wkv[l],(size_t)W*dim},B={wgate[l],(size_t)W*dim}; ds4f_pool_run(m.pool,tw,&A); ds4f_pool_run(m.pool,tw,&B);
        for(size_t i=0;i<(size_t)W*dim;i++){wkv[l][i]=ds4f_f32_bf16(frand()*.02f); wgate[l][i]=ds4f_f32_bf16(frand()*.02f);}
    }
    float *x=aligned_alloc(256,(size_t)dim*4), *kv=aligned_alloc(256,(size_t)W*4), *sc=aligned_alloc(256,(size_t)W*4);
    for(int i=0;i<dim;i++)x[i]=frand();

    { double t0=ds4f_now(); for(int t=0;t<ntok;t++)for(int l=0;l<NL;l++)ds4f_pool_run(m.pool,empty_w,NULL);
      printf("empty dispatch     : %6.2f us/call  -> %5.2f ms/tok (41 calls)\n",(ds4f_now()-t0)/(ntok*NL)*1e6,(ds4f_now()-t0)/ntok*1e3); }
    { double t0=ds4f_now(); for(int t=0;t<ntok;t++)for(int l=0;l<NL;l++){ ds4f_cmpmv_bf16_task ct={kv,sc,wkv[l],wgate[l],x,W,dim}; ds4f_pool_run(m.pool,ds4f_cmpmv_bf16_worker,&ct);}
      double dt=ds4f_now()-t0; printf("cmpmv 48T          : %6.2f us/call  -> %5.2f ms/tok  (%.1f GB/s weights)\n",dt/(ntok*NL)*1e6,dt/ntok*1e3,(double)ntok*NL*W*dim*2*2/dt/1e9); }
    ds4f_pool_stop(m.pool);
    /* 24T (2-CMG) variant: fewer threads -> cheaper barrier, enough for W=1024? */
    m.pool=ds4f_pool_start(24,2);
    { double t0=ds4f_now(); for(int t=0;t<ntok;t++)for(int l=0;l<NL;l++){ ds4f_cmpmv_bf16_task ct={kv,sc,wkv[l],wgate[l],x,W,dim}; ds4f_pool_run(m.pool,ds4f_cmpmv_bf16_worker,&ct);}
      double dt=ds4f_now()-t0; printf("cmpmv 24T (2CMG)   : %6.2f us/call  -> %5.2f ms/tok  (%.1f GB/s weights)\n",dt/(ntok*NL)*1e6,dt/ntok*1e3,(double)ntok*NL*W*dim*2*2/dt/1e9); }
    ds4f_pool_stop(m.pool);
    return 0;
}
