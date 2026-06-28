/* Validate and benchmark the GLM5.2 INT8 (w8a16) matvec/GEMM kernels on A64FX.
 * Kernel-only: no model files, no uTofu, no staging. GS=128 (group) or GS=cols (per-channel). */
#define GLM5_IMPL
#include "glm5.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static double wall_sec(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec + (double)t.tv_nsec*1.0e-9;
}
static uint32_t lcg(uint32_t*s){ *s=*s*1664525u+1013904223u; return *s; }

int main(void){
    int rows=glm5_envi("ROWS",8192);
    int cols=glm5_envi("COLS",6144);
    int reps=glm5_envi("REPS",5);
    int gs=glm5_envi("GS",128);              /* 128 = group-128; set =cols for per-channel */
    if(gs<=0||gs>cols) gs=cols;
    int sb=(cols+gs-1)/gs;
    uint8_t *W=(uint8_t*)glm5_amalloc((size_t)rows*cols);
    float *S=(float*)glm5_amalloc((size_t)rows*sb*4);
    float *x=(float*)glm5_amalloc((size_t)cols*4);
    float *ys=(float*)glm5_amalloc((size_t)rows*4);
    float *yo=(float*)glm5_amalloc((size_t)rows*4);
    if(!W||!S||!x||!ys||!yo) return 2;
    uint32_t st=1;
    for(size_t i=0;i<(size_t)rows*cols;i++) W[i]=(uint8_t)(lcg(&st)&0xff);        /* full int8 range */
    for(int i=0;i<cols;i++) x[i]=(float)((int)(lcg(&st)%2001)-1000)*1.0e-4f;
    for(size_t i=0;i<(size_t)rows*sb;i++) S[i]=2.0e-4f*(float)(1+(lcg(&st)&3));

    /* decode correctness: scalar vs glm5_int8_decode_row_bf16 */
    int decode_err=0, decode_rows=rows<8?rows:8;
    uint16_t *dref=(uint16_t*)glm5_amalloc((size_t)cols*2);
    uint16_t *dopt=(uint16_t*)glm5_amalloc((size_t)cols*2);
    if(!dref||!dopt) return 2;
    for(int r=0;r<decode_rows;r++){
        const uint8_t*wr=W+(size_t)r*cols; const float*sr=S+(size_t)r*sb;
        for(int c=0;c<cols;c++) dref[c]=glm5_f2bf((float)((int)wr[c]-128)*sr[c/gs]);
        glm5_int8_decode_row_bf16(dopt,wr,sr,gs,0,cols);
        for(int c=0;c<cols;c++) if(dref[c]!=dopt[c]) decode_err++;
    }

    double t0=wall_sec();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int r=0;r<rows;r++) ys[r]=glm5_dot_int8_row(W+(size_t)r*cols,S+(size_t)r*sb,gs,x,cols);
    double ts=wall_sec()-t0;

    t0=wall_sec();
#if defined(__ARM_FEATURE_SVE)
    int nb=rows/8;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int bi=0;bi<nb;bi++){
        int r=bi*8; const uint8_t*w=W+(size_t)r*cols; const float*s=S+(size_t)r*sb;
        glm5_matvec_int8_8row(yo+r,
            w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,
            w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,w+7*(size_t)cols,
            s,s+sb,s+2*(size_t)sb,s+3*(size_t)sb,
            s+4*(size_t)sb,s+5*(size_t)sb,s+6*(size_t)sb,s+7*(size_t)sb,
            gs,x,cols);
    }
    for(int r=nb*8;r<rows;r++) yo[r]=glm5_dot_int8_row(W+(size_t)r*cols,S+(size_t)r*sb,gs,x,cols);
#else
    memcpy(yo,ys,(size_t)rows*4);
#endif
    double to=wall_sec()-t0;

    double max_abs=0.0,max_rel=0.0;
    for(int r=0;r<rows;r++){
        double d=fabs((double)ys[r]-(double)yo[r]), den=fabs((double)ys[r])+1.0e-12;
        if(d>max_abs)max_abs=d; if(d/den>max_rel)max_rel=d/den;
    }
    double best_s=ts,best_o=to;
    for(int it=1;it<reps;it++){
        t0=wall_sec();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int r=0;r<rows;r++) ys[r]=glm5_dot_int8_row(W+(size_t)r*cols,S+(size_t)r*sb,gs,x,cols);
        ts=wall_sec()-t0; if(ts<best_s)best_s=ts;
        t0=wall_sec();
#if defined(__ARM_FEATURE_SVE)
        int nb2=rows/8;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int bi=0;bi<nb2;bi++){
            int r=bi*8; const uint8_t*w=W+(size_t)r*cols; const float*s=S+(size_t)r*sb;
            glm5_matvec_int8_8row(yo+r,
                w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,
                w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,w+7*(size_t)cols,
                s,s+sb,s+2*(size_t)sb,s+3*(size_t)sb,
                s+4*(size_t)sb,s+5*(size_t)sb,s+6*(size_t)sb,s+7*(size_t)sb,
                gs,x,cols);
        }
        for(int r=nb2*8;r<rows;r++) yo[r]=glm5_dot_int8_row(W+(size_t)r*cols,S+(size_t)r*sb,gs,x,cols);
#endif
        to=wall_sec()-t0; if(to<best_o)best_o=to;
    }
    double ops=2.0*(double)rows*(double)cols;
    printf("INT8_KERNEL rows=%d cols=%d gs=%d reps=%d decode_err=%d max_abs=%.6g max_rel=%.6g\n",
           rows,cols,gs,reps,decode_err,max_abs,max_rel);
    printf("INT8_KERNEL scalar=%.6f s %.2f Gop/s opt8=%.6f s %.2f Gop/s speedup=%.2fx\n",
           best_s,ops/best_s/1e9,best_o,ops/best_o/1e9,best_s/best_o);

    int gemm_n=glm5_envi("GEMM_N",64);
    int gemm_rows=glm5_envi("GEMM_ROWS",rows<2048?rows:2048);
    if(gemm_rows>rows) gemm_rows=rows;
    float *Xg=(float*)glm5_amalloc((size_t)gemm_n*cols*4);
    float *Yg=(float*)glm5_amalloc((size_t)gemm_n*gemm_rows*4);
    float *Yr=(float*)glm5_amalloc((size_t)gemm_n*gemm_rows*4);
    if(!Xg||!Yg||!Yr) return 2;
    for(size_t i=0;i<(size_t)gemm_n*cols;i++) Xg[i]=(float)((int)(lcg(&st)%2001)-1000)*1.0e-4f;
    glm5_model gm; memset(&gm,0,sizeof gm);
    int check_n=gemm_n<4?gemm_n:4, check_rows=gemm_rows<64?gemm_rows:64;
    for(int t=0;t<check_n;t++) for(int r=0;r<check_rows;r++)
        Yr[(size_t)t*gemm_rows+r]=glm5_dot_int8_row(W+(size_t)r*cols,S+(size_t)r*sb,gs,Xg+(size_t)t*cols,cols);
    glm5_gemm_int8(&gm,Yg,W,S,gs,Xg,check_n,check_rows,cols);
    double gemm_abs=0.0,gemm_rel=0.0;
    for(int t=0;t<check_n;t++) for(int r=0;r<check_rows;r++){
        double d=fabs((double)Yr[(size_t)t*gemm_rows+r]-(double)Yg[(size_t)t*check_rows+r]);
        double den=fabs((double)Yr[(size_t)t*gemm_rows+r])+1.0e-12;
        if(d>gemm_abs)gemm_abs=d; if(d/den>gemm_rel)gemm_rel=d/den;
    }
    double tg_best=1e30;
    for(int it=0;it<reps;it++){
        t0=wall_sec();
        glm5_gemm_int8(&gm,Yg,W,S,gs,Xg,gemm_n,gemm_rows,cols);
        double tg=wall_sec()-t0; if(tg<tg_best)tg_best=tg;
    }
    double gops=2.0*(double)gemm_n*(double)gemm_rows*(double)cols;
    printf("INT8_GEMM N=%d rows=%d cols=%d gs=%d max_abs=%.6g max_rel=%.6g best=%.6f s %.2f Gop/s\n",
           gemm_n,gemm_rows,cols,gs,gemm_abs,gemm_rel,tg_best,gops/tg_best/1e9);

    /* w8a8 SDOT GEMM: speed + accuracy vs the w8a16 reference Yr (activation int8-quant error) */
    float *Ys=(float*)glm5_amalloc((size_t)gemm_n*gemm_rows*4);
    if(!Ys) return 2;
    glm5_gemm_int8_sdot(&gm,Ys,W,S,gs,Xg,check_n,check_rows,cols);
    double sd_abs=0.0,sd_rel=0.0; double refn=0.0;
    for(int t=0;t<check_n;t++) for(int r=0;r<check_rows;r++){
        double ref=Yr[(size_t)t*gemm_rows+r];
        double d=fabs(ref-(double)Ys[(size_t)t*check_rows+r]), den=fabs(ref)+1e-9;
        if(d>sd_abs)sd_abs=d; if(d/den>sd_rel)sd_rel=d/den; refn+=ref*ref;
    }
    /* RMS relative error: more meaningful than worst-case for a quantized dot */
    double sse=0,sref=0;
    for(int t=0;t<check_n;t++) for(int r=0;r<check_rows;r++){
        double ref=Yr[(size_t)t*gemm_rows+r], dd=ref-(double)Ys[(size_t)t*check_rows+r];
        sse+=dd*dd; sref+=ref*ref;
    }
    double tsd=1e30;
    for(int it=0;it<reps;it++){
        t0=wall_sec(); glm5_gemm_int8_sdot(&gm,Ys,W,S,gs,Xg,gemm_n,gemm_rows,cols);
        double tt=wall_sec()-t0; if(tt<tsd)tsd=tt;
    }
    printf("INT8_SDOT N=%d rows=%d cols=%d gs=%d max_rel=%.4g rms_rel=%.4g best=%.6f s %.2f Gop/s (vs w8a16 GEMM %.2f Gop/s)\n",
           gemm_n,gemm_rows,cols,gs,sd_rel,sqrt(sse/(sref+1e-30)),tsd,gops/tsd/1e9,gops/tg_best/1e9);
    return (decode_err==0 && max_rel<1.0e-3 && gemm_rel<5.0e-3)?0:1;  /* gemm bf16-decode rounding ~2e-3 */
}
