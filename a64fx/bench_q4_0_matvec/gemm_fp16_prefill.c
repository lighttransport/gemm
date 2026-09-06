/* gemm_fp16_prefill.c - WS2: fp16 weight x activation prefill GEMM for Gemma-4.
 *
 * The Gemma-4 attention projections (q/k/v/output) and ffn_down are F16. Replace
 * the F16->fp32-dequant + fp32-FMA path (~128 GFLOPS/core) with the proven
 * a64fx/fused-gemm fp16 12x2 micro-kernel (fp16 FMLA, fp32-C epilogue, ~89% of
 * 256 GFLOPS L1-resident). fp32-shadow accumulation = fp16 within each Kc block,
 * fp32 across blocks (init then accum) -> safe for long K (5376 / 21504).
 *
 *   Y[M_tok, N_out] = X[M_tok, K](fp16) @ W[N_out, K]^T(fp16)
 *
 * Build (native A64FX):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *       -I../../common gemm_fp16_prefill.c \
 *       ../fused-gemm/micro_kernel_fp16_12x2_swp.S \
 *       ../fused-gemm/micro_kernel_fp16_12x2_swp_accum.S -lm -o gemm_fp16_prefill
 *   NUMA_DISTRIBUTE=1 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
 *       ./gemm_fp16_prefill MODEL.gguf [tensor]
 */
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include <arm_sve.h>
#include <omp.h>
#include <sys/mman.h>
#include <math.h>

void micro_kernel_fp16_12x2_swp(const _Float16*A,const _Float16*B,float*C,int64_t K,int64_t u,int64_t ldc);
void micro_kernel_fp16_12x2_swp_accum(const _Float16*A,const _Float16*B,float*C,int64_t K,int64_t u,int64_t ldc);

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }
static int find_tensor(gguf_context*g,const char*n){ for(uint64_t i=0;i<g->n_tensors;i++){ const char*s=gguf_tensor_name(g,(int)i); if(s&&!strcmp(s,n))return (int)i;} return -1; }
static void*xmap(size_t n){ void*p=mmap(NULL,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0); if(p==MAP_FAILED){perror("mmap");exit(1);} return p; }

#define MR 12
#define NR 64
#define KC 256    /* fp16 accumulation block; fp32 shadow across blocks */

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [tensor]\n",argv[0]); return 1; }
    const char*tname=argc>2?argv[2]:"blk.0.ffn_down.weight";
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    double peak=256.0*nt;   /* fp16 FMA GFLOPS */
    gguf_context*g=gguf_open(argv[1],1); if(!g){fprintf(stderr,"open fail\n");return 1;}
    int idx=find_tensor(g,tname); if(idx<0){fprintf(stderr,"no tensor %s\n",tname);return 1;}
    gguf_tensor_info*ti=&g->tensors[idx];
    int K=(int)ti->dims[0], N=(int)ti->dims[1];
    int is_f16 = (ti->type==GGML_TYPE_F16);
    fprintf(stderr,"[fp16-gemm] %s type=%d K=%d N=%d %s\n",tname,(int)ti->type,K,N,
            is_f16?"(F16)":"(NOT F16 -> synthetic W on these dims; validates the fp16 12x2 driver for WS3 attention)");
    if(K%KC||N%NR){ fprintf(stderr,"need K%%%d==0,N%%%d==0 (K=%d N=%d)\n",KC,NR,K,N); return 1; }
    int NTn=N/NR, KBn=K/KC;
    _Float16*W=(_Float16*)xmap((size_t)N*K*sizeof(_Float16));
    if(is_f16) memcpy(W,gguf_tensor_data(g,idx),(size_t)N*K*sizeof(_Float16));
    else {
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int n=0;n<N;n++) for(int k=0;k<K;k++) W[(size_t)n*K+k]=(_Float16)(0.02f*(float)(((n*3+k*7)&0x3f)-32));
    }
    gguf_close(g);

    /* pack B: [NTn][K*64] K-major, Bp[n0][k*64+c]=W[(n0*64+c)*K+k] */
    _Float16*Bp=(_Float16*)xmap((size_t)NTn*K*NR*sizeof(_Float16));
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int n0=0;n0<NTn;n0++){ _Float16*b=Bp+(size_t)n0*K*NR;
        for(int k=0;k<K;k++) for(int c=0;c<NR;c++) b[(size_t)k*NR+c]=W[(size_t)(n0*NR+c)*K+k]; }
    munmap(W,(size_t)N*K*sizeof(_Float16));

    int Ms[]={12,48,96,192,384}; int nM=5;
    printf("\n=== %s N=%d K=%d  fp16 12x2 GEMM (fp32 shadow, Kc=%d) @ %d thr (peak %.0f GF) ===\n",tname,N,K,KC,nt,peak);
    printf("  %-6s %9s %10s %8s %10s\n","M(tok)","ms","GFLOPS","%peak","relL2");
    for(int mi=0;mi<nM;mi++){ int M=Ms[mi]; int MTn=M/MR;
        _Float16*Xf=(_Float16*)xmap((size_t)M*K*sizeof(_Float16));
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int i=0;i<M;i++) for(int k=0;k<K;k++) Xf[(size_t)i*K+k]=(_Float16)(0.03f*(float)(((i*5+k*3)&0x3f)-32));
        /* pack A: [MTn][K*12] K-major, Ap[m0][k*12+r]=X[(m0*12+r)*K+k] */
        _Float16*Ap=(_Float16*)xmap((size_t)MTn*K*MR*sizeof(_Float16));
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int m0=0;m0<MTn;m0++){ _Float16*a=Ap+(size_t)m0*K*MR;
            for(int k=0;k<K;k++) for(int r=0;r<MR;r++) a[(size_t)k*MR+r]=Xf[(size_t)(m0*MR+r)*K+k]; }
        float*C=(float*)xmap((size_t)M*N*sizeof(float));

        #define GEMM_PASS() \
          _Pragma("omp parallel for num_threads(nt) schedule(static) collapse(2)") \
          for(int n0=0;n0<NTn;n0++) for(int m0=0;m0<MTn;m0++){ \
            const _Float16*ap=Ap+(size_t)m0*K*MR; const _Float16*bp=Bp+(size_t)n0*K*NR; \
            float*c=C+(size_t)(m0*MR)*N+n0*NR; \
            micro_kernel_fp16_12x2_swp(ap, bp, c, KC, 0, (int64_t)N*4); \
            for(int kb=1;kb<KBn;kb++) \
              micro_kernel_fp16_12x2_swp_accum(ap+(size_t)kb*KC*MR, bp+(size_t)kb*KC*NR, c, KC, 0, (int64_t)N*4); \
          }
        GEMM_PASS();  /* warm */

        /* accuracy vs fp32 reference (few rows, strided cols) */
        double num=0,den=0; int rc=(M<12?M:12);
        for(int m=0;m<rc;m++) for(int cc=0;cc<N;cc+=193){
            int n0=cc/NR, c=cc%NR; const _Float16*bp=Bp+(size_t)n0*K*NR; double ref=0;
            for(int k=0;k<K;k++) ref += (double)(float)Xf[(size_t)m*K+k]*(double)(float)bp[(size_t)k*NR+c];
            double got=C[(size_t)m*N+cc]; double e=got-ref; num+=e*e; den+=ref*ref;
        }
        double rel=den>0?sqrt(num/den):0;

        int reps=(M*(long)N*K>4L<<30)?3:8;
        volatile double sink=0; double freq=(double)rdfreq();
        volatile uint64_t c0=rdcyc();
        for(int rep=0;rep<reps;rep++){ GEMM_PASS(); sink+=C[((size_t)rep*48271)%((size_t)M*N)]; }
        volatile uint64_t c1=rdcyc();
        double per=(double)(c1-c0)/freq/reps; (void)sink;
        double gf=2.0*(double)M*N*K/per/1e9;
        printf("  %-6d %9.3f %10.1f %7.0f%% %10.5f\n",M,per*1000,gf,gf/peak*100,rel);
        #undef GEMM_PASS
        munmap(Xf,(size_t)M*K*sizeof(_Float16)); munmap(Ap,(size_t)MTn*K*MR*sizeof(_Float16)); munmap(C,(size_t)M*N*sizeof(float));
    }
    return 0;
}
