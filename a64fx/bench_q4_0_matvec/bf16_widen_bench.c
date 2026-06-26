/* BF16->fp32 widening technique bench for LLM matvec/GEMM on A64FX SVE.
 *
 * BF16 = upper 16 bits of fp32. Three ways to widen a contiguous BF16 weight
 * row to fp32 inside a dot product:
 *   (A) lsl   : ld1uh (u16->u32, value in LOW 16) + lsl #16            (2 ops / 16 vals)
 *   (B) zip   : ld1h (32 bf16) + zip1/zip2 with a zero vector -> value
 *               lands in the HIGH 16 of each .s lane = fp32, natural order (1 ld + 2 zip / 32 vals)
 *   (C) offset: p_odd predicated ld1h at pointer offsets 0 and -1 (clair trick):
 *               bf16 lands directly in the odd .h lane (= high 16 of .s) -> NO shift;
 *               produces even/odd DEINTERLEAVED, so x is loaded via ld2w (2 ld1h + 1 ld2w / 32 vals)
 *
 * Measures GFLOPS, effective GB/s (is it BW- or issue-bound?), relL2 vs fp32.
 * Real 12B dims: K=3840 (gate/up,qkv,out) and K=15360 (ffn_down), M=1 matvec.
 *
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp bf16_widen_bench.c -lm -o bf16_widen_bench
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <arm_sve.h>

static double wall(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static uint16_t f2bf(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }
static float bf2f(uint16_t h){ uint32_t u=(uint32_t)h<<16; float f; memcpy(&f,&u,4); return f; }

/* (A) lsl baseline */
static inline float dot_lsl(const uint16_t *w,const float *x,int n){
    svfloat32_t acc=svdup_f32(0); int vl=svcntw(),k=0; svbool_t pt=svptrue_b32();
    for(;k+vl<=n;k+=vl){
        svuint32_t wu=svld1uh_u32(pt,w+k);
        svfloat32_t wf=svreinterpret_f32_u32(svlsl_n_u32_x(pt,wu,16));
        acc=svmla_f32_x(pt,acc,wf,svld1_f32(pt,x+k));
    }
    if(k<n){ svbool_t pg=svwhilelt_b32(k,n);
        svuint32_t wu=svld1uh_u32(pg,w+k);
        svfloat32_t wf=svreinterpret_f32_u32(svlsl_n_u32_x(pg,wu,16));
        acc=svmla_f32_m(pg,acc,wf,svld1_f32(pg,x+k)); }
    return svaddv_f32(svptrue_b32(),acc);
}
/* (B) zip-with-zero: load 2*VL bf16, zip with zero -> high-16 = fp32, natural order */
static inline float dot_zip(const uint16_t *w,const float *x,int n){
    svfloat32_t acc0=svdup_f32(0),acc1=svdup_f32(0);
    int vlh=svcnth(),vl=svcntw(),k=0; svbool_t pth=svptrue_b16(),pt=svptrue_b32();
    svuint16_t zero=svdup_u16(0);
    for(;k+vlh<=n;k+=vlh){
        svuint16_t raw=svld1_u16(pth,w+k);
        svfloat32_t wlo=svreinterpret_f32_u16(svzip1_u16(zero,raw));
        svfloat32_t whi=svreinterpret_f32_u16(svzip2_u16(zero,raw));
        acc0=svmla_f32_x(pt,acc0,wlo,svld1_f32(pt,x+k));
        acc1=svmla_f32_x(pt,acc1,whi,svld1_f32(pt,x+k+vl));
    }
    acc0=svadd_f32_x(pt,acc0,acc1);
    for(;k<n;k++){ /* scalar tail */ }
    float s=svaddv_f32(pt,acc0);
    for(int j=(n/vlh)*vlh;j<n;j++) s+=bf2f(w[j])*x[j];
    return s;
}
/* (C) p_odd offset (clair): bf16 directly into high-16 via predicated ld1h, even/odd
 * deinterleaved; x via ld2w. Requires w[k-1] readable -> caller pads front by 1 u16. */
static inline float dot_offset(const uint16_t *w,const float *x,int n){
    svfloat32_t acc_e=svdup_f32(0),acc_o=svdup_f32(0);
    int vlh=svcnth(),k=0;
    /* p_odd: odd .h lanes active (1,3,5,...) */
    svbool_t podd=svnot_b_z(svptrue_b16(), svuzp1_b16(svptrue_b16(),svpfalse_b()));
    /* build p_odd robustly: lane j active iff j odd */
    svuint16_t iota=svindex_u16(0,1);
    podd=svcmpeq_u16(svptrue_b16(), svand_u16_x(svptrue_b16(),iota,svdup_u16(1)), svdup_u16(1));
    for(;k+vlh<=n;k+=vlh){
        svuint32_t wo=svreinterpret_u32_u16(svld1_u16(podd, w+k));     /* bf16[k+1,k+3,..] hi-16 */
        svuint32_t we=svreinterpret_u32_u16(svld1_u16(podd, w+k-1));   /* bf16[k,k+2,..]  hi-16 */
        svfloat32x2_t xv=svld2_f32(svptrue_b32(), x+k);                /* xe=x[k,k+2,..], xo=x[k+1,..] */
        acc_e=svmla_f32_x(svptrue_b32(),acc_e,svreinterpret_f32_u32(we),svget2_f32(xv,0));
        acc_o=svmla_f32_x(svptrue_b32(),acc_o,svreinterpret_f32_u32(wo),svget2_f32(xv,1));
    }
    float s=svaddv_f32(svptrue_b32(),svadd_f32_x(svptrue_b32(),acc_e,acc_o));
    for(int j=(n/vlh)*vlh;j<n;j++) s+=bf2f(w[j])*x[j];
    return s;
}

/* (D) p_odd offset with PRE-SPLIT x (article technique done right): x is deinterleaved
 * ONCE (xe=x[0,2,..], xo=x[1,3,..]); each row's inner loop is just 2 ld1h(p_odd) + 2 fma
 * -> NO shift, NO per-row ld2w. The 2 ld1h replace ld1uh+lsl (4->2 insns, the article's
 * 50%). Requires w[i-1] readable (caller front-pads W by 1 u16). */
static inline float dot_offset_ps(const uint16_t *w,const float *xe,const float *xo,int n,svbool_t p_odd){
    svfloat32_t ae=svdup_f32(0),ao=svdup_f32(0);
    int vlh=svcnth(),vl=svcntw(),i=0; svbool_t pt=svptrue_b32();
    for(;i+vlh<=n;i+=vlh){
        svfloat32_t wo=svreinterpret_f32_u16(svld1_u16(p_odd,w+i));    /* w[i+1,i+3,..] hi-16 */
        svfloat32_t we=svreinterpret_f32_u16(svld1_u16(p_odd,w+i-1));  /* w[i,i+2,..]  hi-16 */
        int j=i/2;
        ae=svmla_x(pt,ae,we,svld1_f32(pt,xe+j));
        ao=svmla_x(pt,ao,wo,svld1_f32(pt,xo+j));
    }
    float s=svaddv(pt,svadd_x(pt,ae,ao));
    for(int k=(n/vlh)*vlh;k<n;k++) s+=bf2f(w[k])*((k&1)?xo[k/2]:xe[k/2]);
    return s;
}

#define MV(NAME,FN) \
static void NAME(float *Y,const uint16_t *W,const float *x,int n_rows,int K){ \
    _Pragma("omp parallel for schedule(static)") \
    for(int r=0;r<n_rows;r++) Y[r]=FN(W+(size_t)r*K,x,K); }
MV(mv_lsl,dot_lsl)
MV(mv_zip,dot_zip)
MV(mv_offset,dot_offset)

static void mv_offset_ps(float *Y,const uint16_t *W,const float *x,int n_rows,int K){
    svuint16_t iota=svindex_u16(0,1);
    svbool_t p_odd=svcmpne_n_u16(svptrue_b16(),svand_n_u16_x(svptrue_b16(),iota,1),0);
    float *xe=(float*)malloc((size_t)(K/2+16)*4),*xo=(float*)malloc((size_t)(K/2+16)*4);
    for(int k=0;k<K/2;k++){ xe[k]=x[2*k]; xo[k]=x[2*k+1]; }
    #pragma omp parallel for schedule(static)
    for(int r=0;r<n_rows;r++) Y[r]=dot_offset_ps(W+(size_t)r*K,xe,xo,K,p_odd);
    free(xe); free(xo);
}

static double rel(const float*a,const float*b,int n){ double nu=0,de=0; for(int i=0;i<n;i++){double e=a[i]-b[i];nu+=e*e;de+=(double)b[i]*b[i];} return sqrt(nu/(de+1e-30)); }

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):3840, n_rows=argc>2?atoi(argv[2]):15360, reps=argc>3?atoi(argv[3]):50;
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    /* W padded front by 1 u16 per the offset trick (point Wp at base+1) */
    uint16_t *Wbase=(uint16_t*)malloc((size_t)n_rows*K*2+2); uint16_t *W=Wbase+1;
    float *x=(float*)malloc((size_t)K*4); float *Yr=(float*)malloc((size_t)n_rows*4),*Y=(float*)malloc((size_t)n_rows*4);
    /* parallel first-touch by row-block (mirror numa_setup) so W is CMG-distributed */
    #pragma omp parallel for schedule(static)
    for(int r=0;r<n_rows;r++) for(int k=0;k<K;k++){ size_t i=(size_t)r*K+k; W[i]=f2bf(0.02f*(float)((int)(i%61)-30)/30.0f); }
    for(int i=0;i<K;i++) x[i]=0.03f*(float)((i%53)-26)/26.0f;
    double flop=2.0*n_rows*K, bytes=(double)n_rows*K*2 + K*4;  /* W bf16 + x */
    printf("BF16 matvec  K=%d n_rows=%d threads=%d  (W=%.1fMB)\n",K,n_rows,nt,(double)n_rows*K*2/1e6);
    /* ref */
    #pragma omp parallel for schedule(static)
    for(int r=0;r<n_rows;r++){ double s=0; for(int k=0;k<K;k++) s+=(double)bf2f(W[(size_t)r*K+k])*x[k]; Yr[r]=(float)s; }
    #define RUN(NAME,FN) do{ FN(Y,W,x,n_rows,K); double e=rel(Y,Yr,n_rows); \
        double t=wall(); for(int r=0;r<reps;r++) FN(Y,W,x,n_rows,K); t=(wall()-t)/reps; \
        printf("  %-8s %7.3f ms  %7.1f GFLOPS  %6.1f GB/s  relL2=%.2e\n",NAME,t*1e3,flop/t/1e9,bytes/t/1e9,e); }while(0)
    RUN("lsl",mv_lsl);
    RUN("zip",mv_zip);
    RUN("offset",mv_offset);
    RUN("offset_ps",mv_offset_ps);
    return 0;
}
