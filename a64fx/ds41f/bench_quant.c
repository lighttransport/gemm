#define _POSIX_C_SOURCE 200809L
#include "ds41f_sve.h"
#include "ds41f_kernels.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
static double now(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int grouped(void)
{
    const size_t groups=8,rows=1024,cols=4096;
    uint8_t *w=malloc(groups*rows*cols),*s=malloc(groups*rows*cols/1024);
    float *x=malloc(groups*cols*4),*y=malloc(groups*rows*4),*ref=malloc(groups*rows*4);
    if(!w||!s||!x||!y||!ref)return 1;
    #pragma omp parallel for
    for(size_t i=0;i<groups*rows*cols;++i)w[i]=(uint8_t)(0x28+i%32);
    for(size_t i=0;i<groups*rows*cols/1024;++i)s[i]=(uint8_t)(124+i%5);
    for(size_t i=0;i<groups*cols;++i)x[i]=sinf((float)i*.013f);
    for(size_t g=0;g<groups;++g)
        if(ds41f_fp8_matvec(ref+g*rows,w+g*rows*cols,s+g*rows*cols/1024,x+g*cols,rows,cols))return 1;
    if(ds41f_fp8_grouped_matvec(y,w,s,x,groups,rows,cols))return 1;
    for(size_t i=0;i<groups*rows;++i)if(!isfinite(y[i])||y[i]!=ref[i])return 1;
    double times[2];
    for(int mode=0;mode<2;++mode){double start=now();
        for(int repeat=0;repeat<50;++repeat){
            if(mode){if(ds41f_fp8_grouped_matvec(y,w,s,x,groups,rows,cols))return 1;}
            else for(size_t g=0;g<groups;++g)
                if(ds41f_fp8_matvec(y+g*rows,w+g*rows*cols,s+g*rows*cols/1024,x+g*cols,rows,cols))return 1;
        }
        times[mode]=(now()-start)/50;
    }
    printf("GROUPED_FP8 PASS groups=8 rows=1024 cols=4096 exact=1 separate_us=%.3f fused_us=%.3f speedup=%.3f\n",times[0]*1e6,times[1]*1e6,times[0]/times[1]);
    free(w);free(s);free(x);free(y);free(ref);return 0;
}
int main(void)
{
    const size_t shapes[][2]={{2304,5120},{5120,2304},{1280,5120},{32768,1280}};
    printf("threads=%d compute_peak_GFLOPs=6144 nominal_HBM_GBs=1024\n",omp_get_max_threads());
    for(int mode=0;mode<2;++mode) for(int sh=0;sh<4;++sh) {
        size_t rows=shapes[sh][0],cols=shapes[sh][1];
        size_t bytes=rows*cols/(mode?1:2);
        size_t scales=mode?((rows+31)/32)*((cols+31)/32):rows*(cols/32);
        uint8_t *w=malloc(bytes),*s=malloc(scales);
        float *x=malloc(cols*4),*y=malloc(rows*4),*ref=malloc(rows*4);
        if(!w||!s||!x||!y||!ref)return 2;
        #pragma omp parallel for
        for(size_t i=0;i<bytes;++i)w[i]=mode?(uint8_t)(0x28+(i%32)):(uint8_t)((i*37+11)%256);
        for(size_t i=0;i<scales;++i)s[i]=(uint8_t)(124+i%5);
        for(size_t i=0;i<cols;++i)x[i]=(float)((int)(i%23)-11)/16;
        if(mode)ds41f_fp8_matvec_ref(ref,w,s,x,rows,cols,32);
        else ds41f_mxfp4_matvec_ref(ref,w,s,x,rows,cols);
        int rc=mode?ds41f_fp8_matvec(y,w,s,x,rows,cols):ds41f_mxfp4_matvec(y,w,s,x,rows,cols);
        double worst=0;
        for(size_t i=0;i<rows;++i){double e=fabs(y[i]-ref[i]);if(!isfinite(y[i])||e>1e-3){fprintf(stderr,"FAIL mode=%d row=%zu y=%g ref=%g\n",mode,i,y[i],ref[i]);return 1;}if(e>worst)worst=e;}
        if(rc)return rc;
        double t=now();int iters=50;
        for(int i=0;i<iters;++i){if(mode)ds41f_fp8_matvec(y,w,s,x,rows,cols);else ds41f_mxfp4_matvec(y,w,s,x,rows,cols);}
        t=(now()-t)/iters;
        printf("PASS format=%s rows=%zu cols=%zu us=%.3f GFLOPs=%.3f effective_GBs=%.3f max_abs=%g\n",mode?"FP8":"MXFP4",rows,cols,t*1e6,2.*rows*cols/t/1e9,(bytes+scales)/t/1e9,worst);
        fflush(stdout);free(w);free(s);free(x);free(y);free(ref);
    }
    return grouped();
}
