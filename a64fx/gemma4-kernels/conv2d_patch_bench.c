/* Gemma4UV vision patch-embedding (conv2d-as-matmul) kernel bench, A64FX SVE.
 *
 * The "conv2d" is a Conv2d(kernel=stride=patch) over the image = im2col + matmul:
 *   out[tok][r] = sum_k W[r][k] * X[tok][k]      (r in n_rows, k in n_cols)
 * Real dims (mmproj-F32): n_cols=6912 (=48*48*3 patch), n_rows=3840 (embed), N=patches.
 *
 * Current encoder (common/gemma4_vision_encoder.h, g4v_dot #else branch) runs this
 * fully SCALAR on A64FX (no AVX2). This bench compares scalar vs SVE-dot vs an
 * SVE register-blocked F32 GEMM (X reused across rows -> higher AI), at the real dims.
 *
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp conv2d_patch_bench.c -lm -o conv2d_patch_bench
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

static double wall(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

/* ---- scalar reference (mirrors g4v_dot #else) ---- */
static void conv_scalar(float *Y,const float *W,const float *X,int n_rows,int n_cols,int N){
    #pragma omp parallel for schedule(static)
    for(int r=0;r<n_rows;r++){ const float *row=W+(size_t)r*n_cols;
        for(int tok=0;tok<N;tok++){ const float *x=X+(size_t)tok*n_cols;
            float s=0; for(int k=0;k<n_cols;k++) s+=row[k]*x[k];
            Y[(size_t)tok*n_rows+r]=s; } }
}

#if defined(__ARM_FEATURE_SVE)
/* ---- SVE dot (per row,token) ---- */
static inline float sve_dot(const float *a,const float *b,int n){
    svfloat32_t acc=svdup_f32(0); int vl=svcntw();
    int k=0; for(;k+vl<=n;k+=vl) acc=svmla_f32_x(svptrue_b32(),acc,svld1_f32(svptrue_b32(),a+k),svld1_f32(svptrue_b32(),b+k));
    if(k<n){ svbool_t pg=svwhilelt_b32(k,n); acc=svmla_f32_m(pg,acc,svld1_f32(pg,a+k),svld1_f32(pg,b+k)); }
    return svaddv_f32(svptrue_b32(),acc);
}
static void conv_sve_dot(float *Y,const float *W,const float *X,int n_rows,int n_cols,int N){
    #pragma omp parallel for schedule(static)
    for(int r=0;r<n_rows;r++){ const float *row=W+(size_t)r*n_cols;
        for(int tok=0;tok<N;tok++) Y[(size_t)tok*n_rows+r]=sve_dot(row,X+(size_t)tok*n_cols,n_cols); }
}

/* ---- SVE register-blocked F32 GEMM: MR=6 rows x NR=4 tokens, 24 acc vectors,
 * reduce (svaddv) at tile end. X[tok] reused across 6 rows, W[row] across 4 tokens. ---- */
#define MR 6
#define NR 4
static void conv_sve_blk(float *Y,const float *W,const float *X,int n_rows,int n_cols,int N){
    int RB=n_rows/MR, TB=N/NR;
    #pragma omp parallel for schedule(static) collapse(2)
    for(int rb=0;rb<RB;rb++) for(int tb=0;tb<TB;tb++){
        int r0=rb*MR, t0=tb*NR;
        const float *w0=W+(size_t)(r0+0)*n_cols,*w1=W+(size_t)(r0+1)*n_cols,*w2=W+(size_t)(r0+2)*n_cols;
        const float *w3=W+(size_t)(r0+3)*n_cols,*w4=W+(size_t)(r0+4)*n_cols,*w5=W+(size_t)(r0+5)*n_cols;
        const float *x0=X+(size_t)(t0+0)*n_cols,*x1=X+(size_t)(t0+1)*n_cols,*x2=X+(size_t)(t0+2)*n_cols,*x3=X+(size_t)(t0+3)*n_cols;
        svfloat32_t a00=svdup_f32(0),a01=svdup_f32(0),a02=svdup_f32(0),a03=svdup_f32(0);
        svfloat32_t a10=svdup_f32(0),a11=svdup_f32(0),a12=svdup_f32(0),a13=svdup_f32(0);
        svfloat32_t a20=svdup_f32(0),a21=svdup_f32(0),a22=svdup_f32(0),a23=svdup_f32(0);
        svfloat32_t a30=svdup_f32(0),a31=svdup_f32(0),a32=svdup_f32(0),a33=svdup_f32(0);
        svfloat32_t a40=svdup_f32(0),a41=svdup_f32(0),a42=svdup_f32(0),a43=svdup_f32(0);
        svfloat32_t a50=svdup_f32(0),a51=svdup_f32(0),a52=svdup_f32(0),a53=svdup_f32(0);
        int vl=svcntw(),k=0;
        for(;k+vl<=n_cols;k+=vl){ svbool_t pg=svptrue_b32();
            svfloat32_t xv0=svld1_f32(pg,x0+k),xv1=svld1_f32(pg,x1+k),xv2=svld1_f32(pg,x2+k),xv3=svld1_f32(pg,x3+k);
            svfloat32_t wv;
            wv=svld1_f32(pg,w0+k); a00=svmla_f32_x(pg,a00,wv,xv0);a01=svmla_f32_x(pg,a01,wv,xv1);a02=svmla_f32_x(pg,a02,wv,xv2);a03=svmla_f32_x(pg,a03,wv,xv3);
            wv=svld1_f32(pg,w1+k); a10=svmla_f32_x(pg,a10,wv,xv0);a11=svmla_f32_x(pg,a11,wv,xv1);a12=svmla_f32_x(pg,a12,wv,xv2);a13=svmla_f32_x(pg,a13,wv,xv3);
            wv=svld1_f32(pg,w2+k); a20=svmla_f32_x(pg,a20,wv,xv0);a21=svmla_f32_x(pg,a21,wv,xv1);a22=svmla_f32_x(pg,a22,wv,xv2);a23=svmla_f32_x(pg,a23,wv,xv3);
            wv=svld1_f32(pg,w3+k); a30=svmla_f32_x(pg,a30,wv,xv0);a31=svmla_f32_x(pg,a31,wv,xv1);a32=svmla_f32_x(pg,a32,wv,xv2);a33=svmla_f32_x(pg,a33,wv,xv3);
            wv=svld1_f32(pg,w4+k); a40=svmla_f32_x(pg,a40,wv,xv0);a41=svmla_f32_x(pg,a41,wv,xv1);a42=svmla_f32_x(pg,a42,wv,xv2);a43=svmla_f32_x(pg,a43,wv,xv3);
            wv=svld1_f32(pg,w5+k); a50=svmla_f32_x(pg,a50,wv,xv0);a51=svmla_f32_x(pg,a51,wv,xv1);a52=svmla_f32_x(pg,a52,wv,xv2);a53=svmla_f32_x(pg,a53,wv,xv3);
        }
        if(k<n_cols){ svbool_t pg=svwhilelt_b32(k,n_cols);
            svfloat32_t xv0=svld1_f32(pg,x0+k),xv1=svld1_f32(pg,x1+k),xv2=svld1_f32(pg,x2+k),xv3=svld1_f32(pg,x3+k);
            svfloat32_t wv;
            wv=svld1_f32(pg,w0+k); a00=svmla_f32_m(pg,a00,wv,xv0);a01=svmla_f32_m(pg,a01,wv,xv1);a02=svmla_f32_m(pg,a02,wv,xv2);a03=svmla_f32_m(pg,a03,wv,xv3);
            wv=svld1_f32(pg,w1+k); a10=svmla_f32_m(pg,a10,wv,xv0);a11=svmla_f32_m(pg,a11,wv,xv1);a12=svmla_f32_m(pg,a12,wv,xv2);a13=svmla_f32_m(pg,a13,wv,xv3);
            wv=svld1_f32(pg,w2+k); a20=svmla_f32_m(pg,a20,wv,xv0);a21=svmla_f32_m(pg,a21,wv,xv1);a22=svmla_f32_m(pg,a22,wv,xv2);a23=svmla_f32_m(pg,a23,wv,xv3);
            wv=svld1_f32(pg,w3+k); a30=svmla_f32_m(pg,a30,wv,xv0);a31=svmla_f32_m(pg,a31,wv,xv1);a32=svmla_f32_m(pg,a32,wv,xv2);a33=svmla_f32_m(pg,a33,wv,xv3);
            wv=svld1_f32(pg,w4+k); a40=svmla_f32_m(pg,a40,wv,xv0);a41=svmla_f32_m(pg,a41,wv,xv1);a42=svmla_f32_m(pg,a42,wv,xv2);a43=svmla_f32_m(pg,a43,wv,xv3);
            wv=svld1_f32(pg,w5+k); a50=svmla_f32_m(pg,a50,wv,xv0);a51=svmla_f32_m(pg,a51,wv,xv1);a52=svmla_f32_m(pg,a52,wv,xv2);a53=svmla_f32_m(pg,a53,wv,xv3);
        }
        svbool_t p=svptrue_b32();
        float *y0=Y+(size_t)(t0+0)*n_rows+r0,*y1=Y+(size_t)(t0+1)*n_rows+r0,*y2=Y+(size_t)(t0+2)*n_rows+r0,*y3=Y+(size_t)(t0+3)*n_rows+r0;
        y0[0]=svaddv_f32(p,a00);y1[0]=svaddv_f32(p,a01);y2[0]=svaddv_f32(p,a02);y3[0]=svaddv_f32(p,a03);
        y0[1]=svaddv_f32(p,a10);y1[1]=svaddv_f32(p,a11);y2[1]=svaddv_f32(p,a12);y3[1]=svaddv_f32(p,a13);
        y0[2]=svaddv_f32(p,a20);y1[2]=svaddv_f32(p,a21);y2[2]=svaddv_f32(p,a22);y3[2]=svaddv_f32(p,a23);
        y0[3]=svaddv_f32(p,a30);y1[3]=svaddv_f32(p,a31);y2[3]=svaddv_f32(p,a32);y3[3]=svaddv_f32(p,a33);
        y0[4]=svaddv_f32(p,a40);y1[4]=svaddv_f32(p,a41);y2[4]=svaddv_f32(p,a42);y3[4]=svaddv_f32(p,a43);
        y0[5]=svaddv_f32(p,a50);y1[5]=svaddv_f32(p,a51);y2[5]=svaddv_f32(p,a52);y3[5]=svaddv_f32(p,a53);
    }
    /* remainder rows / tokens via scalar dot (small) */
    int RBdone=RB*MR, TBdone=TB*NR;
    for(int r=RBdone;r<n_rows;r++) for(int tok=0;tok<N;tok++){ const float*row=W+(size_t)r*n_cols,*x=X+(size_t)tok*n_cols; float s=0; for(int k=0;k<n_cols;k++)s+=row[k]*x[k]; Y[(size_t)tok*n_rows+r]=s; }
    for(int r=0;r<RBdone;r++) for(int tok=TBdone;tok<N;tok++){ const float*row=W+(size_t)r*n_cols,*x=X+(size_t)tok*n_cols; float s=0; for(int k=0;k<n_cols;k++)s+=row[k]*x[k]; Y[(size_t)tok*n_rows+r]=s; }
}
#endif

static double relL2(const float*A,const float*B,size_t n){ double num=0,den=0; for(size_t i=0;i<n;i++){double e=A[i]-B[i];num+=e*e;den+=(double)B[i]*B[i];} return sqrt(num/(den+1e-30)); }

int main(int argc,char**argv){
    int n_cols=argc>1?atoi(argv[1]):6912, n_rows=argc>2?atoi(argv[2]):3840, N=argc>3?atoi(argv[3]):256;
    int reps=argc>4?atoi(argv[4]):5;
    float *W=(float*)malloc((size_t)n_rows*n_cols*4);
    float *X=(float*)malloc((size_t)N*n_cols*4);
    float *Yr=(float*)malloc((size_t)N*n_rows*4),*Y=(float*)malloc((size_t)N*n_rows*4);
    for(size_t i=0;i<(size_t)n_rows*n_cols;i++) W[i]=0.02f*(float)((int)(i%61)-30)/30.0f;
    for(size_t i=0;i<(size_t)N*n_cols;i++) X[i]=0.03f*(float)((int)(i%53)-26)/26.0f;
    double flop=2.0*n_rows*n_cols*N;
    printf("patch-embed conv2d  n_cols=%d n_rows=%d N=%d\n",n_cols,n_rows,N);

    conv_scalar(Yr,W,X,n_rows,n_cols,N);
    { double t=wall(); for(int r=0;r<reps;r++) conv_scalar(Y,W,X,n_rows,n_cols,N); t=(wall()-t)/reps;
      printf("  scalar    : %7.2f ms  %7.1f GFLOPS\n",t*1e3,flop/t/1e9); }
#if defined(__ARM_FEATURE_SVE)
    conv_sve_dot(Y,W,X,n_rows,n_cols,N);
    { double e=relL2(Y,Yr,(size_t)N*n_rows); double t=wall(); for(int r=0;r<reps;r++) conv_sve_dot(Y,W,X,n_rows,n_cols,N); t=(wall()-t)/reps;
      printf("  sve_dot   : %7.2f ms  %7.1f GFLOPS  relL2=%.2e\n",t*1e3,flop/t/1e9,e); }
    conv_sve_blk(Y,W,X,n_rows,n_cols,N);
    { double e=relL2(Y,Yr,(size_t)N*n_rows); double t=wall(); for(int r=0;r<reps;r++) conv_sve_blk(Y,W,X,n_rows,n_cols,N); t=(wall()-t)/reps;
      printf("  sve_blk6x4: %7.2f ms  %7.1f GFLOPS  relL2=%.2e\n",t*1e3,flop/t/1e9,e); }
#endif
    return 0;
}
