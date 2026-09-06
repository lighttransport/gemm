/* gemm_layer.c - prefill int8 GEMM on a REAL Gemma-4 weight, per-layer.
 * Prefill is COMPUTE-bound (weight reused across M tokens), so target 80% of
 * the int8 SDOT FLOP peak (512 GIOPS/core; ~460 = 90% register-resident).
 *
 * C[N,M] = W[N,K] (int8, streamed once) * X[M,K] (int8, L2-resident).
 * 4x4 register microkernel: per k-block load 4 W + 4 X vecs, 16 SDOT into
 * C[4][4]. Weight panel [4,K] stays L1-resident -> reused across all M cols ->
 * compute-bound. Sweep M; measure GIOPS, %peak, and accuracy vs fp32 GEMM.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *          -D_GNU_SOURCE -I../../common gemm_layer.c -lm -o gemm_layer
 * Run:   NUMA_DISTRIBUTE=1 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores ./gemm_layer model.gguf [tensor]
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

/* 4 weight-rows x 4 act-cols int8 SDOT microkernel. wK = W rowstride (K int8),
 * xK = X rowstride. Accumulate over K (nbp 64-blocks). cstride = M (C row). */
static inline void mk4x4(const int8_t*w,size_t wK,const int8_t*xx,size_t xK,int nbp,int32_t*C,int M){
    svbool_t pg=svptrue_b8(), p3=svptrue_b32();
    svint32_t c00=svdup_s32(0),c01=svdup_s32(0),c02=svdup_s32(0),c03=svdup_s32(0);
    svint32_t c10=svdup_s32(0),c11=svdup_s32(0),c12=svdup_s32(0),c13=svdup_s32(0);
    svint32_t c20=svdup_s32(0),c21=svdup_s32(0),c22=svdup_s32(0),c23=svdup_s32(0);
    svint32_t c30=svdup_s32(0),c31=svdup_s32(0),c32=svdup_s32(0),c33=svdup_s32(0);
    const int8_t*w0=w,*w1=w+wK,*w2=w+2*wK,*w3=w+3*wK;
    const int8_t*x0=xx,*x1=xx+xK,*x2=xx+2*xK,*x3=xx+3*xK;
    for(int p=0;p<nbp;p++){ size_t o=(size_t)p*64;
        svint8_t a0=svld1_s8(pg,w0+o),a1=svld1_s8(pg,w1+o),a2=svld1_s8(pg,w2+o),a3=svld1_s8(pg,w3+o);
        svint8_t b0=svld1_s8(pg,x0+o),b1=svld1_s8(pg,x1+o),b2=svld1_s8(pg,x2+o),b3=svld1_s8(pg,x3+o);
        c00=svdot_s32(c00,a0,b0);c01=svdot_s32(c01,a0,b1);c02=svdot_s32(c02,a0,b2);c03=svdot_s32(c03,a0,b3);
        c10=svdot_s32(c10,a1,b0);c11=svdot_s32(c11,a1,b1);c12=svdot_s32(c12,a1,b2);c13=svdot_s32(c13,a1,b3);
        c20=svdot_s32(c20,a2,b0);c21=svdot_s32(c21,a2,b1);c22=svdot_s32(c22,a2,b2);c23=svdot_s32(c23,a2,b3);
        c30=svdot_s32(c30,a3,b0);c31=svdot_s32(c31,a3,b1);c32=svdot_s32(c32,a3,b2);c33=svdot_s32(c33,a3,b3);
    }
    int32_t*r0=C,*r1=C+M,*r2=C+2*M,*r3=C+3*M;
    r0[0]=svaddv_s32(p3,c00);r0[1]=svaddv_s32(p3,c01);r0[2]=svaddv_s32(p3,c02);r0[3]=svaddv_s32(p3,c03);
    r1[0]=svaddv_s32(p3,c10);r1[1]=svaddv_s32(p3,c11);r1[2]=svaddv_s32(p3,c12);r1[3]=svaddv_s32(p3,c13);
    r2[0]=svaddv_s32(p3,c20);r2[1]=svaddv_s32(p3,c21);r2[2]=svaddv_s32(p3,c22);r2[3]=svaddv_s32(p3,c23);
    r3[0]=svaddv_s32(p3,c30);r3[1]=svaddv_s32(p3,c31);r3[2]=svaddv_s32(p3,c32);r3[3]=svaddv_s32(p3,c33);
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [tensor]\n",argv[0]); return 1; }
    const char*tname=argc>2?argv[2]:"blk.0.ffn_gate.weight";
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    int ncore=nt; double peak=512.0*ncore; /* int8 GIOPS theoretical */
    gguf_context*g=gguf_open(argv[1],1); if(!g){fprintf(stderr,"open\n");return 1;}
    int idx=find_tensor(g,tname); if(idx<0){fprintf(stderr,"no %s\n",tname);return 1;}
    gguf_tensor_info*t=&g->tensors[idx];
    int cols=(int)t->dims[0], rows=(int)t->dims[1], nb=cols/32, nbp=(cols+63)/64; size_t rb=(size_t)nb*sizeof(block_q4_0);
    fprintf(stderr,"[gemm] %s N=%d K=%d  (prefill weight GEMM)\n",tname,rows,cols);
    block_q4_0*W=(block_q4_0*)xmap((size_t)rows*rb); memcpy(W,gguf_tensor_data(g,idx),(size_t)rows*rb); gguf_close(g);

    /* dequant W -> int8 [N, nbp*64], NUMA-local */
    float max_d=0; for(size_t i=0;i<(size_t)rows*nb;i++){ float a=fabsf(ggml_fp16_to_fp32(W[i].d)); if(a>max_d)max_d=a; }
    float sw=max_d>0?127.0f/(8.0f*max_d):1.0f;
    size_t wK=(size_t)nbp*64; int8_t*WI8=(int8_t*)xmap((size_t)rows*wK);
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r0=0;r0<rows;r0+=8) tf_dequant_q4_0_8row_strided_to_int8((const uint8_t*)W+(size_t)r0*rb,rb,WI8+(size_t)r0*wK,cols,sw);

    int Ms[]={8,32,128,256}; int nM=4;
    printf("\n=== %s  N=%d K=%d  prefill int8 GEMM @ %d threads (peak %.0f GIOPS) ===\n",tname,rows,cols,nt,peak);
    printf("  %-5s %9s %10s %8s\n","M","ms","GIOPS","%peak");
    for(int mi=0;mi<nM;mi++){ int M=Ms[mi];
        /* X[M,K] int8, L2-resident, NUMA-local */
        size_t xK=wK; int8_t*X=(int8_t*)xmap((size_t)M*xK);
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int j=0;j<M;j++) for(size_t k=0;k<xK;k++) X[(size_t)j*xK+k]=(int8_t)(((j*131+k*7)&0x7f)-40);
        int32_t*C=(int32_t*)xmap((size_t)rows*M*4);
        int Mt=M&~3, Nt=rows&~3;
        /* warm */
        #pragma omp parallel for num_threads(nt) schedule(static) collapse(1)
        for(int i=0;i<Nt;i+=4) for(int j=0;j<Mt;j+=4) mk4x4(WI8+(size_t)i*wK,wK,X+(size_t)j*xK,xK,nbp,C+(size_t)i*M+j,M);
        double best=1e30; for(int rep=0;rep<8;rep++){ double s=now();
            #pragma omp parallel for num_threads(nt) schedule(static)
            for(int i=0;i<Nt;i+=4) for(int j=0;j<Mt;j+=4) mk4x4(WI8+(size_t)i*wK,wK,X+(size_t)j*xK,xK,nbp,C+(size_t)i*M+j,M);
            double e=now()-s; if(e<best)best=e; }
        double ops=2.0*Nt*Mt*cols, gi=ops/best/1e9;
        printf("  %-5d %9.3f %10.1f %7.0f%%\n",M,best*1000,gi,gi/peak*100);
        munmap(X,(size_t)M*xK); munmap(C,(size_t)rows*M*4);
    }
    return 0;
}
