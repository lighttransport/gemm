/* laguna_matmat_i8 (prefill linears): per-8-token-tile widen vs widen-once-into-wrow.
 * Shapes are the real ones: q_proj 9216x3072, o_proj 3072x9216, dense_down 3072x12288.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <arm_sve.h>
#include "laguna_s21.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* widen-once-per-row variant (scratch sized for the largest cols we use) */
static void mm_i8_wrow(float *restrict Y, const laguna_w8 *w, const float *restrict X,
                       int rows, int cols, int C) {
    int VL=(int)svcntw(); svbool_t pt=svptrue_b32();
    #pragma omp parallel for schedule(static)
    for (int r=0;r<rows;++r) {
        const int8_t *q=w->q+(size_t)r*cols; svfloat32_t s=svdup_f32(w->s[r]);
        float wrow[LAGUNA_DENSE_INTER];
        for (int c=0;c<cols;c+=VL)
            svst1_f32(pt,wrow+c,svmul_f32_x(pt,svcvt_f32_s32_x(pt,svld1sb_s32(pt,q+c)),s));
        int ct=0;
        for (; ct+8<=C; ct+=8) {
            const float *x0=X+(size_t)(ct+0)*cols,*x1=X+(size_t)(ct+1)*cols,*x2=X+(size_t)(ct+2)*cols,*x3=X+(size_t)(ct+3)*cols;
            const float *x4=X+(size_t)(ct+4)*cols,*x5=X+(size_t)(ct+5)*cols,*x6=X+(size_t)(ct+6)*cols,*x7=X+(size_t)(ct+7)*cols;
            svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
            svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
            for (int c=0;c<cols;c+=VL) {
                svfloat32_t wv=svld1_f32(pt,wrow+c);
                a0=svmla_f32_x(pt,a0,wv,svld1_f32(pt,x0+c)); a1=svmla_f32_x(pt,a1,wv,svld1_f32(pt,x1+c));
                a2=svmla_f32_x(pt,a2,wv,svld1_f32(pt,x2+c)); a3=svmla_f32_x(pt,a3,wv,svld1_f32(pt,x3+c));
                a4=svmla_f32_x(pt,a4,wv,svld1_f32(pt,x4+c)); a5=svmla_f32_x(pt,a5,wv,svld1_f32(pt,x5+c));
                a6=svmla_f32_x(pt,a6,wv,svld1_f32(pt,x6+c)); a7=svmla_f32_x(pt,a7,wv,svld1_f32(pt,x7+c));
            }
            Y[(size_t)(ct+0)*rows+r]=svaddv_f32(pt,a0); Y[(size_t)(ct+1)*rows+r]=svaddv_f32(pt,a1);
            Y[(size_t)(ct+2)*rows+r]=svaddv_f32(pt,a2); Y[(size_t)(ct+3)*rows+r]=svaddv_f32(pt,a3);
            Y[(size_t)(ct+4)*rows+r]=svaddv_f32(pt,a4); Y[(size_t)(ct+5)*rows+r]=svaddv_f32(pt,a5);
            Y[(size_t)(ct+6)*rows+r]=svaddv_f32(pt,a6); Y[(size_t)(ct+7)*rows+r]=svaddv_f32(pt,a7);
        }
        for (; ct<C; ++ct) {
            const float *x=X+(size_t)ct*cols; svfloat32_t a=svdup_f32(0);
            for (int c=0;c<cols;c+=VL) a=svmla_f32_x(pt,a,svld1_f32(pt,wrow+c),svld1_f32(pt,x+c));
            Y[(size_t)ct*rows+r]=svaddv_f32(pt,a);
        }
    }
}

/* Token-blocked: sweep rows once per block of TB tokens so the X block
 * (TB*cols*4 bytes) stays resident in the CMG's 8 MB L2 across the whole row
 * sweep, instead of each row re-streaming all C tokens of X. */
static void mm_i8_tblock(float *restrict Y, const laguna_w8 *w, const float *restrict X,
                         int rows, int cols, int C) {
    int VL=(int)svcntw(); svbool_t pt=svptrue_b32();
    int TB = (int)(4194304u/((unsigned)cols*4u));   /* ~4 MB of X per block */
    if (TB < 8) TB = 8; if (TB > C) TB = C; TB &= ~7; if (TB < 8) TB = 8;
    for (int t0=0; t0<C; t0+=TB) {
        int TN = C-t0 < TB ? C-t0 : TB;
        #pragma omp parallel for schedule(static)
        for (int r=0;r<rows;++r) {
            const int8_t *q=w->q+(size_t)r*cols; float s=w->s[r]; int ct=0;
            for (; ct+8<=TN; ct+=8) {
                const float *x0=X+(size_t)(t0+ct+0)*cols,*x1=X+(size_t)(t0+ct+1)*cols,
                            *x2=X+(size_t)(t0+ct+2)*cols,*x3=X+(size_t)(t0+ct+3)*cols,
                            *x4=X+(size_t)(t0+ct+4)*cols,*x5=X+(size_t)(t0+ct+5)*cols,
                            *x6=X+(size_t)(t0+ct+6)*cols,*x7=X+(size_t)(t0+ct+7)*cols;
                svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
                svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
                for (int c=0;c<cols;c+=VL) {
                    svfloat32_t wv=svcvt_f32_s32_x(pt,svld1sb_s32(pt,q+c));
                    a0=svmla_f32_x(pt,a0,wv,svld1_f32(pt,x0+c)); a1=svmla_f32_x(pt,a1,wv,svld1_f32(pt,x1+c));
                    a2=svmla_f32_x(pt,a2,wv,svld1_f32(pt,x2+c)); a3=svmla_f32_x(pt,a3,wv,svld1_f32(pt,x3+c));
                    a4=svmla_f32_x(pt,a4,wv,svld1_f32(pt,x4+c)); a5=svmla_f32_x(pt,a5,wv,svld1_f32(pt,x5+c));
                    a6=svmla_f32_x(pt,a6,wv,svld1_f32(pt,x6+c)); a7=svmla_f32_x(pt,a7,wv,svld1_f32(pt,x7+c));
                }
                Y[(size_t)(t0+ct+0)*rows+r]=svaddv_f32(pt,a0)*s; Y[(size_t)(t0+ct+1)*rows+r]=svaddv_f32(pt,a1)*s;
                Y[(size_t)(t0+ct+2)*rows+r]=svaddv_f32(pt,a2)*s; Y[(size_t)(t0+ct+3)*rows+r]=svaddv_f32(pt,a3)*s;
                Y[(size_t)(t0+ct+4)*rows+r]=svaddv_f32(pt,a4)*s; Y[(size_t)(t0+ct+5)*rows+r]=svaddv_f32(pt,a5)*s;
                Y[(size_t)(t0+ct+6)*rows+r]=svaddv_f32(pt,a6)*s; Y[(size_t)(t0+ct+7)*rows+r]=svaddv_f32(pt,a7)*s;
            }
            for (; ct<TN; ++ct) {
                const float *x=X+(size_t)(t0+ct)*cols; svfloat32_t a=svdup_f32(0);
                for (int c=0;c<cols;c+=VL)
                    a=svmla_f32_x(pt,a,svcvt_f32_s32_x(pt,svld1sb_s32(pt,q+c)),svld1_f32(pt,x+c));
                Y[(size_t)(t0+ct)*rows+r]=svaddv_f32(pt,a)*s;
            }
        }
    }
}

static void run(const char *tag, int rows, int cols, int C, int iters) {
    int8_t *q=aligned_alloc(256,(size_t)rows*cols);
    float *s=aligned_alloc(256,(size_t)rows*sizeof(float));
    float *X=aligned_alloc(256,(size_t)C*cols*sizeof(float));
    float *Y1=aligned_alloc(256,(size_t)C*rows*sizeof(float));
    float *Y2=aligned_alloc(256,(size_t)C*rows*sizeof(float));
    for (size_t i=0;i<(size_t)rows*cols;++i) q[i]=(int8_t)((int)(i*2654435761u>>24)%255-127);
    for (int i=0;i<rows;++i) s[i]=0.001f;
    for (size_t i=0;i<(size_t)C*cols;++i) X[i]=(float)((int)(i%2000)-1000)/1000.0f;
    laguna_w8 w={q,s};
    laguna_matmat_i8(Y1,&w,X,rows,cols,C); mm_i8_wrow(Y2,&w,X,rows,cols,C);
    double num=0,den=0; for(size_t i=0;i<(size_t)C*rows;++i){double d=(double)Y2[i]-Y1[i];num+=d*d;den+=(double)Y1[i]*Y1[i];}
    double macs=(double)rows*cols*C*iters;
    laguna_matmat_i8(Y1,&w,X,rows,cols,C);
    double t=now(); for(int i=0;i<iters;++i) laguna_matmat_i8(Y1,&w,X,rows,cols,C); double d1=now()-t;
    mm_i8_wrow(Y2,&w,X,rows,cols,C);
    t=now(); for(int i=0;i<iters;++i) mm_i8_wrow(Y2,&w,X,rows,cols,C); double d2=now()-t;
    mm_i8_tblock(Y2,&w,X,rows,cols,C);
    double num3=0; for(size_t i=0;i<(size_t)C*rows;++i){double d=(double)Y2[i]-Y1[i];num3+=d*d;}
    t=now(); for(int i=0;i<iters;++i) mm_i8_tblock(Y2,&w,X,rows,cols,C); double d3=now()-t;
    printf("%-12s %5dx%-6d C=%3d  tiled %7.2f ms (%6.1f)  wrow %7.2f ms (%6.1f)  tblock %7.2f ms (%6.1f GMAC/s) => %.2fx  relerr=%.1e/%.1e\n",
           tag,rows,cols,C,d1/iters*1e3,macs/d1/1e9,d2/iters*1e3,macs/d2/1e9,d3/iters*1e3,macs/d3/1e9,
           d1/d3,sqrt(num/(den>0?den:1)),sqrt(num3/(den>0?den:1)));
    free(q);free(s);free(X);free(Y1);free(Y2);
}

int main(int argc,char**argv){
    int C=argc>1?atoi(argv[1]):256, it=argc>2?atoi(argv[2]):5;
    run("q_proj",  9216,3072,C,it);
    run("o_proj",  3072,9216,C,it);
    run("kv_proj", 1024,3072,C,it);
    run("dense_dn",3072,12288,C,it);
    return 0;
}
