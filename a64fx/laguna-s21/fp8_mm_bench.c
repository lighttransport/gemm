/* Prefill (batched) comparison: the fp8 "dequant each row once into wrow, then N
 * plain-f32 dots" kernel vs the int8-per-block batched kernel.  The fp8 kernel
 * already amortizes the LUT gather across the N tokens, so the question is
 * whether int8 still wins once the dequant is no longer per-token.
 * Build:  fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *              -DLAGUNA_FP8 -o fp8_mm_bench fp8_mm_bench.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#define LAGUNA_FP8 1
#include "laguna_s21.h"

float laguna_fp8_lut[256];

enum { ROWS = 1024, COLS = 3072 };
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc, char **argv) {
    int N = argc>1 ? atoi(argv[1]) : 10;      /* tokens per expert in a 256-chunk */
    int iters = argc>2 ? atoi(argv[2]) : 40;
    laguna_fp8_init_lut();

    size_t nw=(size_t)ROWS*COLS;
    uint8_t *W=aligned_alloc(256,nw);
    uint16_t *bs=aligned_alloc(256,(size_t)(ROWS/128)*(COLS/128)*sizeof(uint16_t));
    float *X=aligned_alloc(256,(size_t)N*COLS*sizeof(float));
    float *Y1=aligned_alloc(256,(size_t)N*ROWS*sizeof(float));
    float *Y2=aligned_alloc(256,(size_t)N*ROWS*sizeof(float));
    srand(7);
    for (size_t i=0;i<nw;++i){ int b; do{ b=rand()&0xff; }while((b&0x7f)==0x7f||(b&0x78)==0); W[i]=(uint8_t)b; }
    for (size_t i=0;i<(size_t)(ROWS/128)*(COLS/128);++i) bs[i]=laguna_f32_to_bf16(0.0037f);
    for (int i=0;i<N*COLS;++i) X[i]=(float)((rand()%2000)-1000)/1000.0f;

    laguna_w8b q;
    posix_memalign((void**)&q.q,256,nw);
    posix_memalign((void**)&q.s,256,(size_t)ROWS*(COLS/128)*sizeof(float));
    laguna_fp8_to_i8blk(&q,W,bs,ROWS,COLS);

    laguna_matmat_fp8blk(Y1,W,bs,X,ROWS,COLS,N);
    laguna_matmat_i8blk (Y2,&q,X,ROWS,COLS,N);
    double num=0,den=0;
    for (int i=0;i<N*ROWS;++i){ double d=(double)Y2[i]-Y1[i]; num+=d*d; den+=(double)Y1[i]*Y1[i]; }
    printf("N=%d  relerr(i8blk vs fp8) = %.3e\n", N, sqrt(num/(den>0?den:1)));

    double macs=(double)ROWS*COLS*N*iters;
    laguna_matmat_fp8blk(Y1,W,bs,X,ROWS,COLS,N);
    double t0=now(); for(int i=0;i<iters;++i) laguna_matmat_fp8blk(Y1,W,bs,X,ROWS,COLS,N);
    double d1=now()-t0;
    laguna_matmat_i8blk(Y2,&q,X,ROWS,COLS,N);
    t0=now(); for(int i=0;i<iters;++i) laguna_matmat_i8blk(Y2,&q,X,ROWS,COLS,N);
    double d2=now()-t0;
    printf("  fp8 matmat (wrow)  %8.3f ms/iter  %7.2f GMAC/s\n", d1/iters*1e3, macs/d1/1e9);
    printf("  i8blk matmat       %8.3f ms/iter  %7.2f GMAC/s   speedup %.2fx\n",
           d2/iters*1e3, macs/d2/1e9, d1/d2);
    return 0;
}
