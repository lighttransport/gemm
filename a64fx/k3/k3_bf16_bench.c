#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_sve.h>
#include "ggml_dequant.h"

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

/* pure streaming read of the same buffer, 8 accumulators, for the ceiling */
static float stream_read(const uint16_t *w, size_t n16){
    svbool_t pg=svptrue_b16(); svuint16_t a0=svdup_u16(0),a1=a0,a2=a0,a3=a0;
    int vl=(int)svcnth(); size_t i=0;
    for(;i+4*(size_t)vl<=n16;i+=4*vl){
        a0=sveor_u16_x(pg,a0,svld1_u16(pg,&w[i]));
        a1=sveor_u16_x(pg,a1,svld1_u16(pg,&w[i+vl]));
        a2=sveor_u16_x(pg,a2,svld1_u16(pg,&w[i+2*vl]));
        a3=sveor_u16_x(pg,a3,svld1_u16(pg,&w[i+3*vl]));
    }
    a0=sveor_u16_x(pg,sveor_u16_x(pg,a0,a1),sveor_u16_x(pg,a2,a3));
    return (float)svaddv_u16(pg,a0);
}

int main(int argc,char**argv){
    int cols = argc>1?atoi(argv[1]):7168;
    int rows = argc>2?atoi(argv[2]):4224;      /* q_proj shape */
    rows = (rows/8)*8;
    size_t nw = (size_t)rows*cols;
    uint16_t *w = aligned_alloc(256, nw*2);
    float *x = aligned_alloc(256, cols*4), *y = aligned_alloc(256, rows*4);
    for(size_t i=0;i<nw;i++) w[i]=0x3f80^(uint16_t)(i&0x7f);
    for(int i=0;i<cols;i++) x[i]=1.0f/(i+1);
    double mb = nw*2.0/1e9;
    double best=1e9, bs=1e9, bp=1e9; float sink=0;
    for(int rep=0;rep<5;rep++){
        double t=now();
        for(int r=0;r<rows;r+=8){const uint16_t*p=w+(size_t)r*cols;
            matvec_bf16_8row(y+r,p,p+cols,p+2*cols,p+3*cols,p+4*cols,p+5*cols,p+6*cols,p+7*cols,x,cols);}
        double d=now()-t; if(d<best)best=d;
        t=now(); sink+=stream_read(w,nw); d=now()-t; if(d<bs)bs=d;
        t=now();
        for(int r=0;r<rows;r+=8){const uint16_t*p=w+(size_t)r*cols;
            matvec_bf16_8row_pv(y+r,p,p+2*cols,p+4*cols,p+6*cols,x,cols);}
        d=now()-t; if(d<bp)bp=d;
    }
    printf("cols=%d rows=%d bytes=%.2f GB  matvec %.3f ms = %6.1f GB/s | stream %.3f ms = %6.1f GB/s | pv %.3f ms = %6.1f GB/s | ratio %.2fx (sink %g)\n",
           cols,rows,mb,best*1e3,mb/best,bs*1e3,mb/bs,bp*1e3,mb/bp,(mb/bs)/(mb/best),sink);
    return 0;
}
