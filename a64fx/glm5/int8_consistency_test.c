/* Standalone consistency check for the INT8 decode kernels (runs natively on A64FX, no job).
 * Compares, on random data:
 *   ref  : brute-force  y[r] = Σ_c (byte[c]-128)*scale[r,c/gs]*x[c]
 *   scal : glm5_dot_int8_row   (the scalar tail path)
 *   k8   : glm5_matvec_int8_8row (the group-factored 8-row decode kernel used by glm5_mv_int8)
 * Tests both group-128 (attn/dense) and per-channel gs=cols (routed experts). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../common/glm5_int8.h"

static double maxrel(const float*a,const float*b,int n){
    double m=0; for(int i=0;i<n;i++){ double d=fabs((double)a[i]-b[i]); double s=fabs((double)a[i])+1e-6; if(d/s>m)m=d/s; }
    return m;
}
static void run(int rows,int cols,int gs){
    int ng=(cols+gs-1)/gs;
    uint8_t*W=malloc((size_t)rows*cols); float*S=malloc((size_t)rows*ng*4); float*x=malloc((size_t)cols*4);
    for(size_t i=0;i<(size_t)rows*cols;i++) W[i]=rand()&0xff;
    for(int i=0;i<rows*ng;i++) S[i]=(float)((rand()%2000-1000)/100000.0);  /* ~bf16-ish scales */
    for(int i=0;i<cols;i++) x[i]=(float)((rand()%4000-2000)/1000.0);
    float*yref=malloc(rows*4),*yscal=malloc(rows*4),*yk8=malloc(rows*4);
    for(int r=0;r<rows;r++){ double a=0; const uint8_t*w=W+(size_t)r*cols; const float*s=S+(size_t)r*ng;
        for(int c=0;c<cols;c++) a+=((int)w[c]-128)*(double)s[c/gs]*(double)x[c]; yref[r]=(float)a; }
    for(int r=0;r<rows;r++) yscal[r]=glm5_dot_int8_row(W+(size_t)r*cols,S+(size_t)r*ng,gs,x,cols);
#if defined(__ARM_FEATURE_SVE)
    for(int bi=0;bi<rows/8;bi++){ int r=bi*8; const uint8_t*w=W+(size_t)r*cols; const float*s=S+(size_t)r*ng;
        glm5_matvec_int8_8row(yk8+r, w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,
            w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,w+7*(size_t)cols,
            s,s+ng,s+2*(size_t)ng,s+3*(size_t)ng,s+4*(size_t)ng,s+5*(size_t)ng,s+6*(size_t)ng,s+7*(size_t)ng,
            gs,x,cols); }
    for(int r=(rows/8)*8;r<rows;r++) yk8[r]=glm5_dot_int8_row(W+(size_t)r*cols,S+(size_t)r*ng,gs,x,cols);
#else
    for(int r=0;r<rows;r++) yk8[r]=yscal[r];
#endif
    printf("rows=%-4d cols=%-5d gs=%-5d ng=%-3d :  scal-vs-ref=%.2e   k8-vs-ref=%.2e   k8-vs-scal=%.2e\n",
           rows,cols,gs,ng, maxrel(yscal,yref,rows), maxrel(yk8,yref,rows), maxrel(yk8,yscal,rows));
    free(W);free(S);free(x);free(yref);free(yscal);free(yk8);
}
int main(void){
    srand(1234);
    printf("=== INT8 decode-kernel consistency (max relative error) ===\n");
    run(256,6144,128);    /* attention/dense group-128 */
    run(256,2048,128);    /* moe_inter group-128 */
    run(256,2048,2048);   /* routed expert per-channel (gs=cols, ng=1) */
    run(64, 512, 128);
    run(8,  6144,128);
    printf("(expect all ~1e-6..1e-4; a large k8-vs-ref => the 8-row decode kernel is the bug)\n");
    return 0;
}
