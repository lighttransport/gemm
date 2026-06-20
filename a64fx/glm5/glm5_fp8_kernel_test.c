/* Validate and benchmark the GLM5.2 FP8 f32-scale matvec kernel on A64FX.
 * This is a kernel-only test: no model files, no uTofu, no staging. */
#define GLM5_IMPL
#include "glm5.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static double wall_sec(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

static uint32_t lcg(uint32_t *s){
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

int main(void){
    int rows=glm5_envi("ROWS",8192);
    int cols=glm5_envi("COLS",6144);
    int reps=glm5_envi("REPS",5);
    int sb=(cols+127)/128;
    uint8_t *W=(uint8_t*)glm5_amalloc((size_t)rows*cols);
    float *S=(float*)glm5_amalloc((size_t)rows*sb*4);
    float *x=(float*)glm5_amalloc((size_t)cols*4);
    float *ys=(float*)glm5_amalloc((size_t)rows*4);
    float *yo=(float*)glm5_amalloc((size_t)rows*4);
    uint32_t lut[256];
    if(!W||!S||!x||!ys||!yo) return 2;
    glm5_init_fp8_lut(lut);
    uint32_t st=1;
    for(size_t i=0;i<(size_t)rows*cols;i++){
        uint8_t v=(uint8_t)(lcg(&st)&0x7f);
        if((v&0x7f)==0x7f) v=0x3f;
        W[i]=v;
    }
    for(int i=0;i<cols;i++) x[i]=(float)((int)(lcg(&st)%2001)-1000) * 1.0e-4f;
    for(size_t i=0;i<(size_t)rows*sb;i++) S[i]=0.00390625f * (float)(1 + (lcg(&st)&3));

    double t0=wall_sec();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int r=0;r<rows;r++)
        ys[r]=glm5_dot_mxfp8_f32scale_row(W+(size_t)r*cols,S+(size_t)r*sb,x,cols,lut);
    double ts=wall_sec()-t0;

    t0=wall_sec();
#if defined(__ARM_FEATURE_SVE)
    int nb=rows/8;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int bi=0;bi<nb;bi++){
        int r=bi*8; const uint8_t*w=W+(size_t)r*cols; const float*s=S+(size_t)r*sb;
        glm5_matvec_mxfp8_f32scale_8row(yo+r,
            w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,
            w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,w+7*(size_t)cols,
            s,s+sb,s+2*(size_t)sb,s+3*(size_t)sb,
            s+4*(size_t)sb,s+5*(size_t)sb,s+6*(size_t)sb,s+7*(size_t)sb,
            x,cols,lut);
    }
    for(int r=nb*8;r<rows;r++)
        yo[r]=glm5_dot_mxfp8_f32scale_row(W+(size_t)r*cols,S+(size_t)r*sb,x,cols,lut);
#else
    memcpy(yo,ys,(size_t)rows*4);
#endif
    double to=wall_sec()-t0;

    double max_abs=0.0, max_rel=0.0;
    for(int r=0;r<rows;r++){
        double d=fabs((double)ys[r]-(double)yo[r]);
        double den=fabs((double)ys[r])+1.0e-12;
        if(d>max_abs) max_abs=d;
        if(d/den>max_rel) max_rel=d/den;
    }

    double best_s=ts, best_o=to;
    for(int it=1;it<reps;it++){
        t0=wall_sec();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int r=0;r<rows;r++)
            ys[r]=glm5_dot_mxfp8_f32scale_row(W+(size_t)r*cols,S+(size_t)r*sb,x,cols,lut);
        ts=wall_sec()-t0; if(ts<best_s) best_s=ts;

        t0=wall_sec();
#if defined(__ARM_FEATURE_SVE)
        int nb2=rows/8;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int bi=0;bi<nb2;bi++){
            int r=bi*8; const uint8_t*w=W+(size_t)r*cols; const float*s=S+(size_t)r*sb;
            glm5_matvec_mxfp8_f32scale_8row(yo+r,
                w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,
                w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,w+7*(size_t)cols,
                s,s+sb,s+2*(size_t)sb,s+3*(size_t)sb,
                s+4*(size_t)sb,s+5*(size_t)sb,s+6*(size_t)sb,s+7*(size_t)sb,
                x,cols,lut);
        }
        for(int r=nb2*8;r<rows;r++)
            yo[r]=glm5_dot_mxfp8_f32scale_row(W+(size_t)r*cols,S+(size_t)r*sb,x,cols,lut);
#endif
        to=wall_sec()-t0; if(to<best_o) best_o=to;
    }
    double ops=2.0*(double)rows*(double)cols;
    printf("FP8_KERNEL rows=%d cols=%d reps=%d max_abs=%.6g max_rel=%.6g\n",rows,cols,reps,max_abs,max_rel);
    printf("FP8_KERNEL scalar=%.6f s %.2f Gop/s opt8=%.6f s %.2f Gop/s speedup=%.2fx\n",
           best_s,ops/best_s/1e9,best_o,ops/best_o/1e9,best_s/best_o);
    return (max_abs<1.0e-3 && max_rel<1.0e-3) ? 0 : 1;
}
