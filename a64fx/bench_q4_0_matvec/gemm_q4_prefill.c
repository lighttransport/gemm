/* gemm_q4_prefill.c - blocked int8 prefill GEMM for Gemma-4 Q4_0, reusing the
 * proven a64fx/int8-new 6x4 SDOT microkernel (85-95% peak @K=256) + packing,
 * extended with a K-blocked driver (gemma K=5376 = 21x256, accumulate).
 *
 *   C[M_tokens, N_rows] = A[M,K] (int8 acts) * B[N,K]^T (int8 dequant'd Q4_0)
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *   -D_GNU_SOURCE -I../../common -I../int8-new gemm_q4_prefill.c \
 *   ../int8-new/gemm_pack.o ../int8-new/kernel_6x4_opt.o -lm -o gemm_q4_prefill
 */
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include "gemm_pack.h"
#include <omp.h>
#include <sys/mman.h>
#include <time.h>
/* ABI-compliant kernel (kernel_6x4_abi.S adds d8-d15 save/restore); call it
 * directly with no clobber wrapper (no per-call spill of the hot loop state). */
void kernel_6x4_opt_256(const int8_t*,const int8_t*,int32_t*,int);
#define k6x4(A,B,C,ldc) kernel_6x4_opt_256((A),(B),(C),(ldc))

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
/* cntvct timer: the asm 6x4 kernel clobbers callee-saved v8-v15, corrupting any
 * double (d8-d15) kept live across it. Use an INTEGER timer in volatile memory. */
static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }
static int find_tensor(gguf_context*g,const char*n){ for(uint64_t i=0;i<g->n_tensors;i++){ const char*s=gguf_tensor_name(g,(int)i); if(s&&!strcmp(s,n))return (int)i;} return -1; }
static void*xmap(size_t n){ void*p=mmap(NULL,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0); if(p==MAP_FAILED){perror("mmap");exit(1);} return p; }
#define KB 256
#define MR 6
#define NR 64

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [tensor]\n",argv[0]); return 1; }
    const char*tname=argc>2?argv[2]:"blk.0.ffn_gate.weight";
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    double peak=512.0*nt;
    gguf_context*g=gguf_open(argv[1],1); if(!g){fprintf(stderr,"open\n");return 1;}
    int idx=find_tensor(g,tname); if(idx<0){fprintf(stderr,"no %s\n",tname);return 1;}
    gguf_tensor_info*ti=&g->tensors[idx];
    int K=(int)ti->dims[0], N=(int)ti->dims[1], nb=K/32; size_t rb=(size_t)nb*sizeof(block_q4_0);
    if(K%KB||N%NR){ fprintf(stderr,"need K%%256==0,N%%64==0 (K=%d N=%d)\n",K,N); return 1; }
    int KBn=K/KB, NTn=N/NR;
    fprintf(stderr,"[q4-gemm] %s N=%d K=%d (Kblocks=%d Ntiles=%d)\n",tname,N,K,KBn,NTn);
    block_q4_0*W=(block_q4_0*)xmap((size_t)N*rb); memcpy(W,gguf_tensor_data(g,idx),(size_t)N*rb); gguf_close(g);

    /* dequant Q4_0 -> int8 B[N,K] row-major (per-tensor scale), NUMA-local */
    float max_d=0; for(size_t i=0;i<(size_t)N*nb;i++){ float a=fabsf(ggml_fp16_to_fp32(W[i].d)); if(a>max_d)max_d=a; }
    float sw=max_d>0?127.0f/(8.0f*max_d):1.0f;
    int8_t*B=(int8_t*)xmap((size_t)N*K);
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r=0;r<N;r++){ const block_q4_0*row=(const block_q4_0*)((const uint8_t*)W+(size_t)r*rb); int8_t*o=B+(size_t)r*K;
        for(int b=0;b<nb;b++){ int di=(int)lrintf(ggml_fp16_to_fp32(row[b].d)*sw); if(di>127)di=127; if(di<-128)di=-128;
            for(int j=0;j<16;j++){ o[b*32+j]=(int8_t)(((row[b].qs[j]&0xf)-8)*di); o[b*32+16+j]=(int8_t)(((row[b].qs[j]>>4)-8)*di); } } }
    /* pre-pack B: [NTn][KBn] tiles of pack_B_64x256, NUMA-local by N-tile */
    int8_t*Bp=(int8_t*)xmap((size_t)N*K);  /* NR*KB per (ntile,kblock) = N*K total */
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int n=0;n<NTn;n++) for(int k=0;k<KBn;k++)
        pack_B_64x256(B+(size_t)n*NR*K + (size_t)k*KB, K, Bp+((size_t)n*KBn+k)*NR*KB, NR);

    int Ms[]={6,48,96,192,384}; int nM=5;
    printf("\n=== %s N=%d K=%d  BLOCKED int8 prefill GEMM (6x4 SDOT, K-blocked) @ %d thr (peak %.0f GIOPS) ===\n",tname,N,K,nt,peak);
    printf("  %-6s %9s %10s %8s\n","M(tok)","ms","GIOPS","%peak");
    for(int mi=0;mi<nM;mi++){ int M=Ms[mi]; int MTn=M/MR;
        int8_t*A=(int8_t*)xmap((size_t)M*K);
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int i=0;i<M;i++) for(int k=0;k<K;k++) A[(size_t)i*K+k]=(int8_t)(((i*7+k*3)&0x3f)-20);
        /* pre-pack A: [MTn][KBn] tiles of pack_A_6x256 */
        int8_t*Ap=(int8_t*)xmap((size_t)MTn*KBn*MR*KB);
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int m=0;m<MTn;m++) for(int k=0;k<KBn;k++) pack_A_6x256(A+(size_t)m*MR*K+(size_t)k*KB,K,Ap+((size_t)m*KBn+k)*MR*KB,MR);
        int32_t*C=(int32_t*)xmap((size_t)M*N*4);
        #define GEMM_PASS() \
            _Pragma("omp parallel for num_threads(nt) schedule(static) collapse(2)") \
            for(int n=0;n<NTn;n++) for(int m=0;m<MTn;m++){ \
                int32_t Cb[MR*NR] __attribute__((aligned(256))); \
                int32_t Ct[MR*NR]; for(int z=0;z<MR*NR;z++) Ct[z]=0; \
                for(int k=0;k<KBn;k++){ \
                    k6x4(Ap+((size_t)m*KBn+k)*MR*KB, Bp+((size_t)n*KBn+k)*NR*KB, Cb, NR*4); \
                    for(int z=0;z<MR*NR;z++) Ct[z]+=Cb[z]; } \
                for(int r=0;r<MR;r++) for(int c=0;c<NR;c++) C[(size_t)(m*MR+r)*N + n*NR+c]=Ct[r*NR+c]; }
        GEMM_PASS();  /* warm */
        /* correctness vs naive int8 (row 0, col 0..3) */
        if(mi==0){ int bad=0; for(int c=0;c<4;c++){ long ref=0; for(int k=0;k<K;k++) ref+=(long)A[k]*B[(size_t)c*K+k]; if(ref!=C[c]) bad++; }
            fprintf(stderr,"  [correctness M=6 row0: %s]\n", bad?"FAIL":"OK"); }
        int reps=(M*(long)N*K>4L<<30)?3:8;
        volatile long sink=0; double freq=(double)rdfreq();
        volatile uint64_t c0=rdcyc();
        for(int rep=0;rep<reps;rep++){ GEMM_PASS(); sink+=C[((size_t)rep*48271)%((size_t)M*N)]; }
        volatile uint64_t c1=rdcyc();
        double per=(double)(c1-c0)/freq/reps; (void)sink;
        double gi=2.0*(double)M*N*K/per/1e9;
        printf("  %-6d %9.3f %10.1f %7.0f%%\n",M,per*1000,gi,gi/peak*100);
        #undef GEMM_PASS
        munmap(A,(size_t)M*K); munmap(Ap,(size_t)MTn*KBn*MR*KB); munmap(C,(size_t)M*N*4);
    }
    return 0;
}
