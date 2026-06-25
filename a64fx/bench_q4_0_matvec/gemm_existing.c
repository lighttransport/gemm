/* gemm_existing.c - benchmark the PRODUCTION prefill GEMM kernel
 * (tf_gemm_q4_0_tm_worker, used by transformer_prefill_gemm) vs the fp32 FLOP
 * peak. It is the fp32 dequant+FMA path, 4-token-blocked (tf_vec_dot_q4_0_f32_4x).
 * Single real weight tensor, NUMA-clean (mmap-anon + per-thread first-touch).
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *          -D_GNU_SOURCE -I../../common gemm_existing.c -lm -o gemm_existing
 * Run:   NUMA_DISTRIBUTE=1 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores ./gemm_existing model.gguf [tensor]
 */
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include <omp.h>
#include <sys/mman.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int find_tensor(gguf_context*g,const char*n){ for(uint64_t i=0;i<g->n_tensors;i++){ const char*s=gguf_tensor_name(g,(int)i); if(s&&!strcmp(s,n))return (int)i;} return -1; }
static void*xmap(size_t n){ void*p=mmap(NULL,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0); if(p==MAP_FAILED){perror("mmap");exit(1);} return p; }

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [tensor]\n",argv[0]); return 1; }
    const char*tname=argc>2?argv[2]:"blk.0.ffn_gate.weight";
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    double fp32_peak=128.0*nt;   /* GFLOPS: 2 FLA/B * 16 fp32 * 2(FMA) * 2GHz */
    gguf_context*g=gguf_open(argv[1],1); if(!g){fprintf(stderr,"open\n");return 1;}
    int idx=find_tensor(g,tname); if(idx<0){fprintf(stderr,"no %s\n",tname);return 1;}
    gguf_tensor_info*ti=&g->tensors[idx];
    int K=(int)ti->dims[0], rows=(int)ti->dims[1], nb=K/32; size_t rb=(size_t)nb*sizeof(block_q4_0);
    fprintf(stderr,"[gemm-existing] %s N(rows)=%d K=%d\n",tname,rows,K);
    block_q4_0*W=(block_q4_0*)xmap((size_t)rows*rb);
    /* per-thread first-touch copy (NUMA-local by row range) */
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r=0;r<rows;r++) memcpy((uint8_t*)W+(size_t)r*rb,(const uint8_t*)gguf_tensor_data(g,idx)+(size_t)r*rb,rb);
    gguf_close(g);
    qtensor mat; memset(&mat,0,sizeof mat); mat.data=W; mat.type=GGML_TYPE_Q4_0; mat.n_rows=rows; mat.n_cols=K;

    int Ms[]={1,8,32,128,256}; int nM=5;
    printf("\n=== %s  N=%d K=%d  PRODUCTION prefill GEMM (tf_gemm_q4_0, fp32 path) @ %d thr ===\n",tname,rows,K,nt);
    printf("  fp32 FLOP peak = %.0f GFLOPS\n",fp32_peak);
    printf("  %-5s %9s %10s %8s\n","M(tok)","ms","GFLOPS","%peak");
    for(int mi=0;mi<nM;mi++){ int M=Ms[mi];
        float*X=(float*)xmap((size_t)M*K*4);          /* X[M,K], X_stride=K */
        float*Y=(float*)xmap((size_t)M*rows*4);       /* Y[M,rows], Y_stride=rows */
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int j=0;j<M;j++) for(int k=0;k<K;k++) X[(size_t)j*K+k]=((j*13+k*7)%101)/50.0f-1.0f;
        double best=1e30; int reps=(M*(long)rows*K>1L<<30)?4:12;
        for(int rep=0;rep<=reps;rep++){ double s=now();
            #pragma omp parallel num_threads(nt)
            { int T=omp_get_num_threads(),id=omp_get_thread_num();
              int per=(rows+T-1)/T, r0=id*per, r1=r0+per; if(r1>rows)r1=rows;
              tf_gemm_qtensor_tm_task task={Y,&mat,X,r0,r1,K,M,rows,K};
              if(r0<r1) tf_gemm_q4_0_tm_worker(&task);
            }
            double e=now()-s; if(rep>0&&e<best)best=e; }  /* rep 0 = warm */
        double gf=2.0*M*rows*K/best/1e9;
        printf("  %-5d %9.3f %10.1f %7.0f%%\n",M,best*1000,gf,gf/fp32_peak*100);
        munmap(X,(size_t)M*K*4); munmap(Y,(size_t)M*rows*4);
    }
    return 0;
}
