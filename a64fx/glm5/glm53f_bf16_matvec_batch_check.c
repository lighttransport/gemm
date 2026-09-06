#include <arm_sve.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "glm53f_expert_kern.h"

enum { MAX_TOKENS = 4, REPEAT = 8 };

static void *aligned_buffer(size_t bytes){void*p=NULL;return posix_memalign(&p,256,bytes)?NULL:p;}
static void mv(float*y,const uint16_t*w,const float*x,int rows,int cols){
#pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++)y[r]=glm53f_dot_bf16_sve(w+(size_t)r*cols,x,cols);
}

int main(int argc,char**argv){int rows=argc>1?atoi(argv[1]):4096,cols=argc>2?atoi(argv[2]):384,tokens=argc>3?atoi(argv[3]):4;
    uint16_t*w=aligned_buffer((size_t)rows*cols*2);float*x=aligned_buffer((size_t)MAX_TOKENS*cols*4),*ref=aligned_buffer((size_t)MAX_TOKENS*rows*4),*bat=aligned_buffer((size_t)MAX_TOKENS*rows*4);if(rows<1||cols<1||tokens<1||tokens>MAX_TOKENS||!w||!x||!ref||!bat)return 2;
    for(size_t i=0;i<(size_t)rows*cols;i++)w[i]=(uint16_t)(0x3f00u+((i*13+17)%128));for(int t=0;t<tokens;t++)for(int i=0;i<cols;i++)x[(size_t)t*cols+i]=(float)(((i*17+t*29)%251)-125)/125.0f;
    for(int t=0;t<tokens;t++)mv(ref+(size_t)t*rows,w,x+(size_t)t*cols,rows,cols);glm53f_mv_bf16_batch(bat,w,x,tokens,rows,cols);double t0=omp_get_wtime();for(int z=0;z<REPEAT;z++)for(int t=0;t<tokens;t++)mv(ref+(size_t)t*rows,w,x+(size_t)t*cols,rows,cols);double seq=omp_get_wtime()-t0;t0=omp_get_wtime();for(int z=0;z<REPEAT;z++)glm53f_mv_bf16_batch(bat,w,x,tokens,rows,cols);double batch=omp_get_wtime()-t0,d2=0,r2=0;for(size_t i=0;i<(size_t)tokens*rows;i++){double d=(double)ref[i]-bat[i];d2+=d*d;r2+=(double)ref[i]*ref[i];}double rel=sqrt(d2/(r2+1e-30));int ok=rel<2e-6;printf("GLM53F_BF16_MATVEC_BATCH rows=%d cols=%d tokens=%d rel_l2=%.9g seq_ms=%.3f batch_ms=%.3f speedup=%.3f %s\n",rows,cols,tokens,rel,seq*1e3/REPEAT,batch*1e3/REPEAT,seq/batch,ok?"PASS":"FAIL");free(bat);free(ref);free(x);free(w);return ok?0:1;}
