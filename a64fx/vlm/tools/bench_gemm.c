#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
extern void gemm_fp16_BTP(int M,int K,int N,const float*A,int lda,const uint16_t*BTP,float*C,int ldc);
extern void pack_B_fp16(int K,int N,const uint16_t*BT,int ldb,uint16_t*BTP);
extern size_t packed_B_fp16_size(int K,int N);
// clock_gettime(CLOCK_MONOTONIC) is UNRELIABLE on this A64FX node after SVE asm
// (returns huge deltas); use the CNTVCT_EL0 hardware counter (100 MHz).
static double now(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return (double)v*1e-8; }
/* Simulate 24 transformer blocks: each block has a DIFFERENT W (qkv/o/u/d).
 * nw W matrices of size K×N fp16, rotated per "block". */
static void bench24(const char*name,int M,int K,int N,int nw){
    float*A=malloc((size_t)M*K*4); float*C=malloc((size_t)M*N*4);
    for(size_t i=0;i<(size_t)M*K;i++)A[i]=((i*7)%13)/13.0f-0.5f;
    size_t bts=packed_B_fp16_size(K,N);
    uint16_t**BTP=malloc(nw*sizeof(void*));
    for(int w=0;w<nw;w++){
        BTP[w]=malloc(bts);
        /* distinct data per W */
        uint16_t*BT=malloc((size_t)K*N*2);
        for(size_t i=0;i<(size_t)K*N;i++)BT[i]=(uint16_t)((i+w*97)%400);
        pack_B_fp16(K,N,BT,N,BTP[w]);
        free(BT);
    }
    double flops=2.0*M*K*N;
    double total=0;
    for(int it=0;it<3;it++){
        double t0=now();
        for(int b=0;b<nw;b++) gemm_fp16_BTP(M,K,N,A,K,BTP[b],C,N);
        total+=now()-t0;
    }
    double per_call=total/3.0/nw;
    printf("%-20s M=%d K=%d N=%d nw=%d : %7.3f ms/call  %8.1f GFLOP/s/call\n",
           name,M,K,N,nw,per_call*1000,flops/per_call/1e9);
    for(int w=0;w<nw;w++)free(BTP[w]);
    free(BTP);free(A);free(C);
}
int main(void){
    bench24("ffn_up 1W",96,1024,4096,1);
    bench24("ffn_up 24W",96,1024,4096,24);
    bench24("ffn_down 24W",96,4096,1024,24);
    bench24("qkv 24W",96,1024,3072,24);
    return 0;
}
