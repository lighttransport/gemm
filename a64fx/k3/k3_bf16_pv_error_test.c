#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <arm_sve.h>
#include "ggml_dequant.h"
static float b2f(uint16_t h){union{uint32_t u;float f;}x;x.u=(uint32_t)h<<16;return x.f;}
int main(void){
    int cols=7168, rows=8;
    uint16_t *w=aligned_alloc(256,(size_t)rows*cols*2), *pv=aligned_alloc(256,(size_t)rows*cols*2);
    float *x=aligned_alloc(256,cols*4), r[8], o[8], ref[8];
    srand(1);
    for(int i=0;i<rows*cols;i++){ float v=((float)rand()/RAND_MAX-0.5f)*0.1f;
        union{uint32_t u;float f;}c; c.f=v; w[i]=(uint16_t)(c.u>>16); }
    for(int i=0;i<cols;i++) x[i]=((float)rand()/RAND_MAX-0.5f);
    /* pair-interleave: plane p holds rows 2p,2p+1 column-major */
    for(int p=0;p<4;p++) for(int c=0;c<cols;c++){
        pv[(size_t)p*2*cols+2*c]   = w[(size_t)(2*p)*cols+c];
        pv[(size_t)p*2*cols+2*c+1] = w[(size_t)(2*p+1)*cols+c]; }
    for(int q=0;q<8;q++){ double s=0; for(int c=0;c<cols;c++) s+=(double)b2f(w[(size_t)q*cols+c])*x[c]; ref[q]=(float)s; }
    matvec_bf16_8row(r,w,w+cols,w+2*cols,w+3*cols,w+4*cols,w+5*cols,w+6*cols,w+7*cols,x,cols);
    matvec_bf16_8row_pv(o,pv,pv+2*cols,pv+4*cols,pv+6*cols,x,cols);
    double mr=0,mp=0,mrel=0;
    for(int q=0;q<8;q++){
        double dr=fabs(r[q]-ref[q]), dp=fabs(o[q]-ref[q]), dd=fabs(r[q]-o[q]);
        double sc=fabs(ref[q])>1e-6?fabs(ref[q]):1e-6;
        if(dr>mr)mr=dr; if(dp>mp)mp=dp; if(dd/sc>mrel)mrel=dd/sc;
    }
    printf("cols=%d  max_abs_err vs f64 ref: 8row=%.3e  pv=%.3e   max_rel_diff(8row,pv)=%.3e\n",cols,mr,mp,mrel);
    for(int q=0;q<4;q++) printf("  row%d ref=%+.9e  8row=%+.9e  pv=%+.9e\n",q,ref[q],r[q],o[q]);
    return 0;
}
