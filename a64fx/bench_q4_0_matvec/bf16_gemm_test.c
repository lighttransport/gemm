/* bf16_gemm_test.c - register-blocked token-major BF16 x fp32 GEMM for Gemma-4 12B.
 * The current gemm_bf16_f32_tokmajor (ggml_dequant.h) is matvec-based (~38 GFLOPS):
 * per-token dot products, re-reads activations, no register reuse across (tok,row).
 * This blocks MR tokens x NR weight-rows with shared k-vector loads. BF16 weights
 * widened to fp32 in-register (ld1uh -> lsl#16, zero conversion FLOPs). No packing
 * (weights stay row-major [n_rows][K], activations [N][K] row-major).
 *
 *   Y[tok*Ys + row] = sum_k X[tok][k] * bf16_to_f32(W[row][k])
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *          -I../../common bf16_gemm_test.c -lm -o bf16_gemm_test
 */
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <arm_sve.h>
#include <omp.h>

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }

#define MR 4    /* token block */
#define NR 6    /* weight-row block */

/* widen 16 BF16 (u16) at p -> fp32 vector (top-16-bits trick) */
static inline svfloat32_t bf16w(svbool_t pg, const uint16_t*p){
    return svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, p), 16));
}

/* C[4 tok][6 row] dot-block: each (tok,row) accumulated in a 16-wide reg, reduced at end.
 * Shared k-vector loads: 4 X-loads + 6 W-widens feed 24 FMAs per k-step. */
static void bf16_gemm_blocked(float*Y,const uint16_t*W,const float*X,
        int n_rows,int K,int N,int Ys,int Xs,int nt){
    int MTn=(N+MR-1)/MR, NTn=n_rows/NR;
    #pragma omp parallel for num_threads(nt) schedule(static) collapse(2)
    for(int n0=0;n0<NTn;n0++) for(int m0=0;m0<MTn;m0++){
        int vl=(int)svcntw();
        const uint16_t*w0=W+(size_t)(n0*NR+0)*K,*w1=W+(size_t)(n0*NR+1)*K,*w2=W+(size_t)(n0*NR+2)*K;
        const uint16_t*w3=W+(size_t)(n0*NR+3)*K,*w4=W+(size_t)(n0*NR+4)*K,*w5=W+(size_t)(n0*NR+5)*K;
        int t0=m0*MR,t1=t0+1,t2=t0+2,t3=t0+3;
        const float*x0=X+(size_t)(t0<N?t0:0)*Xs,*x1=X+(size_t)(t1<N?t1:0)*Xs;
        const float*x2=X+(size_t)(t2<N?t2:0)*Xs,*x3=X+(size_t)(t3<N?t3:0)*Xs;
        svfloat32_t a00=svdup_f32(0),a01=svdup_f32(0),a02=svdup_f32(0),a03=svdup_f32(0),a04=svdup_f32(0),a05=svdup_f32(0);
        svfloat32_t a10=svdup_f32(0),a11=svdup_f32(0),a12=svdup_f32(0),a13=svdup_f32(0),a14=svdup_f32(0),a15=svdup_f32(0);
        svfloat32_t a20=svdup_f32(0),a21=svdup_f32(0),a22=svdup_f32(0),a23=svdup_f32(0),a24=svdup_f32(0),a25=svdup_f32(0);
        svfloat32_t a30=svdup_f32(0),a31=svdup_f32(0),a32=svdup_f32(0),a33=svdup_f32(0),a34=svdup_f32(0),a35=svdup_f32(0);
        for(int k=0;k<K;k+=vl){
            svbool_t pg=svwhilelt_b32(k,K);
            svfloat32_t v0=bf16w(pg,w0+k),v1=bf16w(pg,w1+k),v2=bf16w(pg,w2+k),v3=bf16w(pg,w3+k),v4=bf16w(pg,w4+k),v5=bf16w(pg,w5+k);
            svfloat32_t x=svld1_f32(pg,x0+k);
            a00=svmla_f32_m(pg,a00,x,v0);a01=svmla_f32_m(pg,a01,x,v1);a02=svmla_f32_m(pg,a02,x,v2);a03=svmla_f32_m(pg,a03,x,v3);a04=svmla_f32_m(pg,a04,x,v4);a05=svmla_f32_m(pg,a05,x,v5);
            x=svld1_f32(pg,x1+k);
            a10=svmla_f32_m(pg,a10,x,v0);a11=svmla_f32_m(pg,a11,x,v1);a12=svmla_f32_m(pg,a12,x,v2);a13=svmla_f32_m(pg,a13,x,v3);a14=svmla_f32_m(pg,a14,x,v4);a15=svmla_f32_m(pg,a15,x,v5);
            x=svld1_f32(pg,x2+k);
            a20=svmla_f32_m(pg,a20,x,v0);a21=svmla_f32_m(pg,a21,x,v1);a22=svmla_f32_m(pg,a22,x,v2);a23=svmla_f32_m(pg,a23,x,v3);a24=svmla_f32_m(pg,a24,x,v4);a25=svmla_f32_m(pg,a25,x,v5);
            x=svld1_f32(pg,x3+k);
            a30=svmla_f32_m(pg,a30,x,v0);a31=svmla_f32_m(pg,a31,x,v1);a32=svmla_f32_m(pg,a32,x,v2);a33=svmla_f32_m(pg,a33,x,v3);a34=svmla_f32_m(pg,a34,x,v4);a35=svmla_f32_m(pg,a35,x,v5);
        }
        svbool_t pt=svptrue_b32(); float*Yb=Y+(size_t)n0*NR;
        #define ST(tok,i,A0,A1,A2,A3,A4,A5) if((tok)<N){ float*y=Yb+(size_t)(tok)*Ys; \
            y[0]=svaddv_f32(pt,A0);y[1]=svaddv_f32(pt,A1);y[2]=svaddv_f32(pt,A2);y[3]=svaddv_f32(pt,A3);y[4]=svaddv_f32(pt,A4);y[5]=svaddv_f32(pt,A5); }
        ST(t0,0,a00,a01,a02,a03,a04,a05) ST(t1,1,a10,a11,a12,a13,a14,a15)
        ST(t2,2,a20,a21,a22,a23,a24,a25) ST(t3,3,a30,a31,a32,a33,a34,a35)
        #undef ST
    }
    /* remainder rows (n_rows % NR): one row x all tokens, vectorized dot */
    int rdone = NTn*NR;
    if(rdone<n_rows){
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int row=rdone;row<n_rows;row++){ const uint16_t*w=W+(size_t)row*K;
            for(int t=0;t<N;t++){ const float*x=X+(size_t)t*Xs; svfloat32_t a=svdup_f32(0);
                for(int k=0;k<K;k+=(int)svcntw()){ svbool_t pg=svwhilelt_b32(k,K);
                    a=svmla_f32_m(pg,a,svld1_f32(pg,x+k),bf16w(pg,w+k)); }
                Y[(size_t)t*Ys+row]=svaddv_f32(svptrue_b32(),a); } }
    }
}

static uint16_t f2bf16(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):128, K=argc>2?atoi(argv[2]):3840, n_rows=argc>3?atoi(argv[3]):15360;
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;

    uint16_t*W=(uint16_t*)malloc((size_t)n_rows*K*2);
    float*X=(float*)malloc((size_t)N*K*4);
    for(size_t i=0;i<(size_t)n_rows*K;i++) W[i]=f2bf16(0.02f*(float)((int)(i%61)-30)/30.0f);
    for(size_t i=0;i<(size_t)N*K;i++) X[i]=0.02f*(float)((int)(i%53)-26)/26.0f;
    float*Yref=(float*)calloc((size_t)N*n_rows,4);
    float*Ynew=(float*)calloc((size_t)N*n_rows,4);
    double freq=(double)rdfreq();
    /* reference: current matvec-based path */
    gemm_bf16_f32_tokmajor(Yref,W,X,n_rows,K,N,n_rows,K);
    uint64_t c0=rdcyc(); for(int r=0;r<3;r++) gemm_bf16_f32_tokmajor(Yref,W,X,n_rows,K,N,n_rows,K); uint64_t c1=rdcyc();
    double tref=(double)(c1-c0)/freq/3;
    /* new: register-blocked */
    bf16_gemm_blocked(Ynew,W,X,n_rows,K,N,n_rows,K,nt);
    uint64_t c2=rdcyc(); for(int r=0;r<3;r++) bf16_gemm_blocked(Ynew,W,X,n_rows,K,N,n_rows,K,nt); uint64_t c3=rdcyc();
    double tnew=(double)(c3-c2)/freq/3;
    double num=0,den=0; for(size_t i=0;i<(size_t)N*n_rows;i++){ double e=Ynew[i]-Yref[i]; num+=e*e; den+=(double)Yref[i]*Yref[i]; }
    fprintf(stderr,"DBG ref[0..2]=%.4f %.4f %.4f  new[0..2]=%.4f %.4f %.4f  num=%.3g den=%.3g\n",
            Yref[0],Yref[1],Yref[2],Ynew[0],Ynew[1],Ynew[2],num,den);
    double gf_ref=2.0*N*n_rows*K/tref/1e9, gf_new=2.0*N*n_rows*K/tnew/1e9;
    printf("N=%d K=%d n_rows=%d  ref(matvec)=%.1f ms %.0f GF  blocked=%.1f ms %.0f GF  speedup=%.2fx  relL2=%.2e\n",
           N,K,n_rows, tref*1e3,gf_ref, tnew*1e3,gf_new, tref/tnew, sqrt(num/den));
    return 0;
}
