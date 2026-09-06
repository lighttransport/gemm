/* fapp region-profiling driver for the GLM-5.2 A64FX kernels — decode matvec (M=1) and prefill GEMM
 * (M>=8), for the shipped w8a16 path and the new int16/int8 SDOT paths. Synthetic weights, 1 node.
 * Build:  fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -I../../common \
 *           -I/opt/FJSVxtclanga/tcsds-1.2.43/include glm5_kern_prof.c -lm -o glm5_kern_prof
 * Profile: fapp -C -d prof_kern -Icpupa -Hevent=statistics ./glm5_kern_prof
 *          fapp -A -d prof_kern -Icpupa,nompi -tcsv -o pa.csv
 * Each fapp_start/stop region gets its own CPU-PA row (GFLOPS, mem throughput GB/s, cache). */
#define GLM5_IMPL
#include "glm5.h"
extern void fapp_start(const char*, int, int);   /* fj_tool/fapp.h (declared to avoid the Fujitsu SVE header clash under -Nclang) */
extern void fapp_stop(const char*, int, int);
#ifdef _OPENMP
#include <omp.h>
#endif
static double wall(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint32_t lcg(uint32_t*s){ *s=*s*1664525u+1013904223u; return *s; }

int main(void){
    int rows=glm5_envi("ROWS",8192), cols=glm5_envi("COLS",6144), gs=glm5_envi("GS",128);
    int M=glm5_envi("M",64), mvrep=glm5_envi("MVREP",400), gmrep=glm5_envi("GMREP",30);
    int sb=(cols+gs-1)/gs;
    uint8_t*W=glm5_amalloc((size_t)rows*cols); float*S=glm5_amalloc((size_t)rows*sb*4);
    float*x=glm5_amalloc((size_t)cols*4), *y=glm5_amalloc((size_t)rows*4);
    float*Xg=glm5_amalloc((size_t)M*cols*4), *Yg=glm5_amalloc((size_t)M*rows*4);
    #pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++){ uint32_t s=1u+(uint32_t)r*2654435761u; for(int c=0;c<cols;c++) W[(size_t)r*cols+c]=(uint8_t)(lcg(&s)&0xff); }
    for(int i=0;i<rows*sb;i++) S[i]=2e-4f;
    for(int i=0;i<cols;i++) x[i]=((int)(i%2001)-1000)*1e-4f;
    for(size_t i=0;i<(size_t)M*cols;i++) Xg[i]=((int)(i%4001)-2000)*1e-3f;
    glm5_model m; memset(&m,0,sizeof m);

    struct { const char*name; int gemm; void*fn; } K[] = {
        {"decode_w8a16",0,0},{"decode_int16",0,0},{"prefill_w8a16",1,0},{"prefill_int16",1,0},{"prefill_int8rb",1,0}
    };
    for(int k=0;k<5;k++){
        double t0=wall();
        fapp_start(K[k].name,1,0);
        if(k==0)      for(int it=0;it<mvrep;it++) glm5_mv_int8(&m,y,W,S,gs,x,rows,cols);
        else if(k==1) for(int it=0;it<mvrep;it++) glm5_mv_int16_sdot(&m,y,W,S,gs,x,rows,cols);
        else if(k==2) for(int it=0;it<gmrep;it++) glm5_gemm_int8(&m,Yg,W,S,gs,Xg,M,rows,cols);
        else if(k==3) for(int it=0;it<gmrep;it++) glm5_gemm_int16sdot(&m,Yg,W,S,gs,Xg,M,rows,cols);
        else          for(int it=0;it<gmrep;it++) glm5_gemm_int8sdot_rb(&m,Yg,W,S,gs,Xg,M,rows,cols);
        fapp_stop(K[k].name,1,0);
        double dt=wall()-t0;
        double reps=K[k].gemm?gmrep:mvrep, tokM=K[k].gemm?M:1;
        double ops=2.0*reps*tokM*(double)rows*cols;
        printf("%-15s reps=%.0f M=%.0f  %.4f s  %.1f Gop/s\n",K[k].name,reps,tokM,dt,ops/dt/1e9);
    }
    printf("PROF_DONE rows=%d cols=%d gs=%d M=%d\n",rows,cols,gs,M);
    return 0;
}
