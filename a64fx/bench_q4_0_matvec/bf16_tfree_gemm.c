/* Transpose-free BF16 GEMM (A64FX SVE): dot-structure, reads X contiguous (NO X-pack)
 * and writes Y token-major (NO C-transpose), widen via zip (free-ish). Compare to the
 * podd broadcast kernel which needs X-pack + col-major C + prepacked k-major W.
 *
 * Y[tok][row] = sum_k W[row][k] * X[tok][k].  W row-major bf16, X token-major fp32.
 * Tile MR tokens x NR rows; acc[mr][nr] is an SVE vector accumulating along k (zip lo+hi
 * into the same acc), svaddv at tile end -> token-major store. NO packing of either operand.
 *
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp bf16_tfree_gemm.c -lm -o bf16_tfree_gemm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <arm_sve.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double wall(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static uint16_t f2bf(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }
static float bf2f(uint16_t h){ uint32_t u=(uint32_t)h<<16; float f; memcpy(&f,&u,4); return f; }

/* MR=4 tokens x NR=4 rows, zip-widen W, X contiguous. acc[mr][nr] += wlo*xlo + whi*xhi.
 * 16 acc + 8 x + 2 w = 26 SVE regs. svaddv at end -> token-major store (no transpose). */
#define MR 4
#define NR 4
static void tfree_gemm(float *Y, const uint16_t *W, const float *X,
                       int n_rows, int K, int N, int Ys, int Xs, int nt){
    int RB = n_rows/NR, TB = N/MR;
    svuint16_t zero = svdup_u16(0);
    #pragma omp parallel for num_threads(nt) schedule(static) collapse(2)
    for (int rb=0; rb<RB; rb++) for (int tb=0; tb<TB; tb++){
        int r0=rb*NR, t0=tb*MR;
        const uint16_t *w0=W+(size_t)(r0)*K,*w1=W+(size_t)(r0+1)*K,*w2=W+(size_t)(r0+2)*K,*w3=W+(size_t)(r0+3)*K;
        const float *x0=X+(size_t)(t0)*Xs,*x1=X+(size_t)(t0+1)*Xs,*x2=X+(size_t)(t0+2)*Xs,*x3=X+(size_t)(t0+3)*Xs;
        svfloat32_t a00=svdup_f32(0),a01=svdup_f32(0),a02=svdup_f32(0),a03=svdup_f32(0);
        svfloat32_t a10=svdup_f32(0),a11=svdup_f32(0),a12=svdup_f32(0),a13=svdup_f32(0);
        svfloat32_t a20=svdup_f32(0),a21=svdup_f32(0),a22=svdup_f32(0),a23=svdup_f32(0);
        svfloat32_t a30=svdup_f32(0),a31=svdup_f32(0),a32=svdup_f32(0),a33=svdup_f32(0);
        int vlh=svcnth(), vl=svcntw(); svbool_t pt=svptrue_b32(), pth=svptrue_b16();
        int k=0;
        for(; k+vlh<=K; k+=vlh){
            svfloat32_t xl0=svld1_f32(pt,x0+k),xh0=svld1_f32(pt,x0+k+vl);
            svfloat32_t xl1=svld1_f32(pt,x1+k),xh1=svld1_f32(pt,x1+k+vl);
            svfloat32_t xl2=svld1_f32(pt,x2+k),xh2=svld1_f32(pt,x2+k+vl);
            svfloat32_t xl3=svld1_f32(pt,x3+k),xh3=svld1_f32(pt,x3+k+vl);
            svuint16_t raw; svfloat32_t wl,wh;
            #define TF_ROW(W,A0,A1,A2,A3) raw=svld1_u16(pth,(W)+k); wl=svreinterpret_f32_u16(svzip1_u16(zero,raw)); wh=svreinterpret_f32_u16(svzip2_u16(zero,raw)); \
                A0=svmla_x(pt,A0,wl,xl0);A0=svmla_x(pt,A0,wh,xh0); A1=svmla_x(pt,A1,wl,xl1);A1=svmla_x(pt,A1,wh,xh1); \
                A2=svmla_x(pt,A2,wl,xl2);A2=svmla_x(pt,A2,wh,xh2); A3=svmla_x(pt,A3,wl,xl3);A3=svmla_x(pt,A3,wh,xh3);
            TF_ROW(w0,a00,a10,a20,a30) TF_ROW(w1,a01,a11,a21,a31) TF_ROW(w2,a02,a12,a22,a32) TF_ROW(w3,a03,a13,a23,a33)
            #undef TF_ROW
        }
        float *y0=Y+(size_t)(t0)*Ys+r0,*y1=Y+(size_t)(t0+1)*Ys+r0,*y2=Y+(size_t)(t0+2)*Ys+r0,*y3=Y+(size_t)(t0+3)*Ys+r0;
        y0[0]=svaddv_f32(pt,a00);y0[1]=svaddv_f32(pt,a01);y0[2]=svaddv_f32(pt,a02);y0[3]=svaddv_f32(pt,a03);
        y1[0]=svaddv_f32(pt,a10);y1[1]=svaddv_f32(pt,a11);y1[2]=svaddv_f32(pt,a12);y1[3]=svaddv_f32(pt,a13);
        y2[0]=svaddv_f32(pt,a20);y2[1]=svaddv_f32(pt,a21);y2[2]=svaddv_f32(pt,a22);y2[3]=svaddv_f32(pt,a23);
        y3[0]=svaddv_f32(pt,a30);y3[1]=svaddv_f32(pt,a31);y3[2]=svaddv_f32(pt,a32);y3[3]=svaddv_f32(pt,a33);
    }
    for(int r=(RB*NR);r<n_rows;r++) for(int t=0;t<N;t++){ double s=0; for(int k=0;k<K;k++) s+=(double)bf2f(W[(size_t)r*K+k])*X[(size_t)t*Xs+k]; Y[(size_t)t*Ys+r]=(float)s; }
    for(int r=0;r<(RB*NR);r++) for(int t=(TB*MR);t<N;t++){ double s=0; for(int k=0;k<K;k++) s+=(double)bf2f(W[(size_t)r*K+k])*X[(size_t)t*Xs+k]; Y[(size_t)t*Ys+r]=(float)s; }
}

static double rel(const float*a,const float*b,int n){ double nu=0,de=0; for(int i=0;i<n;i++){double e=a[i]-b[i];nu+=e*e;de+=(double)b[i]*b[i];} return sqrt(nu/(de+1e-30)); }

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):3840, n_rows=argc>2?atoi(argv[2]):15360, N=argc>3?atoi(argv[3]):128, reps=argc>4?atoi(argv[4]):20;
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    uint16_t *W=(uint16_t*)malloc((size_t)n_rows*K*2);
    float *X=(float*)malloc((size_t)N*K*4), *Y=(float*)malloc((size_t)N*n_rows*4), *Yr=(float*)malloc((size_t)N*n_rows*4);
    #pragma omp parallel for schedule(static)
    for(int r=0;r<n_rows;r++) for(int k=0;k<K;k++){ size_t i=(size_t)r*K+k; W[i]=f2bf(0.02f*(float)((int)(i%61)-30)/30.0f); }
    for(size_t i=0;i<(size_t)N*K;i++) X[i]=0.03f*(float)((int)(i%53)-26)/26.0f;
    double flop=2.0*n_rows*K*N;
    printf("transpose-free BF16 GEMM  K=%d n_rows=%d N=%d threads=%d (MR=%d NR=%d)\n",K,n_rows,N,nt,MR,NR);
    /* ref */
    #pragma omp parallel for schedule(static)
    for(int t=0;t<N;t++) for(int r=0;r<n_rows;r++){ double s=0; for(int k=0;k<K;k++) s+=(double)bf2f(W[(size_t)r*K+k])*X[(size_t)t*K+k]; Yr[(size_t)t*n_rows+r]=(float)s; }
    tfree_gemm(Y,W,X,n_rows,K,N,n_rows,K,nt);
    double e=rel(Y,Yr,N*n_rows);
    double t=wall(); for(int r=0;r<reps;r++) tfree_gemm(Y,W,X,n_rows,K,N,n_rows,K,nt); t=(wall()-t)/reps;
    printf("  tfree: %7.2f ms  %7.1f GFLOPS  (%.1f GF/core)  relL2=%.2e\n", t*1e3, flop/t/1e9, flop/t/1e9/nt, e);
    return 0;
}
