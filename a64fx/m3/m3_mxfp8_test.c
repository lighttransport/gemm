/* m3_mxfp8_test.c - validate the MXFP8 matvec: SVE kernel vs scalar reference on random
 * FP8 weights + E8M0 scales. (The E8M0 bias/sign convention vs the real model is checked
 * separately against a Python reference.)  Build: fcc/fccpx ... -I common -o m3_mxfp8_test ...
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "m3_mxfp8.h"

static uint64_t sm=0x1234; static double sn(void){ sm+=0x9E3779B97F4A7C15ull; uint64_t z=sm;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull; z^=z>>31; return (double)(z>>11)/(double)(1ull<<53); }

int main(void){
    uint32_t lut[256]; m3_init_fp8_lut(lut);
    int N=256; { const char*e=getenv("N"); if(e&&*e) N=atoi(e); }
    int nb=N/M3_MX_BLK;
    uint8_t *w[8],*s[8]; for(int j=0;j<8;j++){ w[j]=malloc(N); s[j]=malloc(nb);
        for(int c=0;c<N;c++) w[j][c]=(uint8_t)(sn()*256);            /* random fp8 bytes */
        for(int b=0;b<nb;b++) s[j][b]=(uint8_t)(120+sn()*16); }      /* E8M0 ~ 2^(-7..+9) */
    float*x=malloc(N*4); for(int c=0;c<N;c++) x[c]=(float)(sn()*2-1);
    float r[8],v[8];
    m3_matvec_mxfp8_8row_ref(r,w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],x,N,lut);
    m3_matvec_mxfp8_8row    (v,w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],x,N,lut);
    double worst=0; for(int j=0;j<8;j++){ double d=fabs(r[j]-v[j]), den=fabs(r[j])>1e-6?fabs(r[j]):1e-6; double rel=d/den; if(rel>worst)worst=rel;
        printf("row %d  ref=%.6f  sve=%.6f  rel=%.2e\n",j,r[j],v[j],rel); }
    int ok=worst<1e-4;
    printf("N=%d worst_rel=%.2e  %s\n",N,worst,ok?"PASS (SVE == scalar ref)":"FAIL");
    return ok?0:1;
}
