/* m3_mxgemm_bench.c - measure the MXFP8 multi-stream "decode-once" speedup, no MPI/uTofu.
 * Times m3_gemm_mxfp8 (the batched kernel used by m3_forward_batch_decode) on a real expert
 * shape (rows=moe_inter, cols=hidden) for N=1 (fused per-token) vs N=8 (decode tile once ->
 * bf16 -> N x matvec_bf16_8row). Reports per-stream ms and the throughput multiple.
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -DM3_IMPL
 *        -D_GNU_SOURCE -I common -I a64fx/llm -o m3_mxgemm_bench a64fx/m3/m3_mxgemm_bench.c -lm
 */
#define M3_IMPL
#include "m3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc,char**argv){
    int rows = argc>1?atoi(argv[1]):3072;   /* moe_inter */
    int cols = argc>2?atoi(argv[2]):6144;   /* hidden */
    int iters= argc>3?atoi(argv[3]):200;
    m3_model m; memset(&m,0,sizeof m); m3_init_fp8_lut(m.fp8_lut); m3_dummy=0;
    int sb=cols/M3_MX_BLK;
    uint8_t*W=malloc((size_t)rows*cols), *S=malloc((size_t)rows*sb);
    for(size_t i=0;i<(size_t)rows*cols;i++) W[i]=(uint8_t)(i*1103515245u>>8);     /* arbitrary fp8 bytes */
    for(size_t i=0;i<(size_t)rows*sb;i++)   S[i]=(uint8_t)(120+(i%16));            /* E8M0 ~2^(-7..+8) */
    int Nmax=8; float*X=malloc((size_t)Nmax*cols*4), *Y=malloc((size_t)Nmax*rows*4);
    for(size_t i=0;i<(size_t)Nmax*cols;i++) X[i]=(float)((i%17)-8)*0.01f;

    for(int N=1;N<=8;N*=2){
        m3_gemm_mxfp8(&m,Y,W,S,X,N,rows,cols);                /* warm */
        double t0=sec(); for(int it=0;it<iters;it++) m3_gemm_mxfp8(&m,Y,W,S,X,N,rows,cols);
        double dt=(sec()-t0)/iters;
        printf("N=%d  %.3f ms/call  %.3f ms/stream  %.1f Gflop/s\n",
               N, dt*1e3, dt*1e3/N, 2.0*N*rows*cols/dt/1e9);
        if(N==1) printf("  (per-stream lower is better; N=8 decode-once should beat N=1)\n");
    }
    return 0;
}
