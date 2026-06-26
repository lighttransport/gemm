/* bf16_podd_test.c - measure the REALISTIC in-model throughput of the clair p_odd
 * BF16 GEMM (sgemm_bf16_2x12, 137 GF/core standalone) for the Gemma-4 FFN, INCLUDING
 * the packing/transpose overhead that decides whether it beats the simple blocked
 * kernel (53 GF/core, no packing).
 *
 *   Y[tok][feat] = sum_k X[tok][k]*W[feat][k].  Map A=features(MR=32), B=tokens(NR=12):
 *   C[feat][tok] = sum_k Wkmajor[k][feat]*Xkmajor[k][tok], C col-major -> transpose to Y.
 *   W pre-packed k-major-interleaved (one-time, like load); X packed k-major BF16 +
 *   C transposed PER GEMM (the per-call cost). Compare to the blocked BF16 GEMM.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *   -I../../common bf16_podd_test.c ~/work/clair/clair/a64fx/llm-guided-opt/sgemm_bf16_2x12.S -lm -o bf16_podd_test
 */
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <arm_sve.h>
#include <omp.h>

void sgemm_bf16_2x12(int64_t K, const void *A, const void *B, float *C, int64_t ldc);
static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }
static uint16_t f2bf(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }
static float bf2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
#define MR 32
#define NR 12

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):120, K=argc>2?atoi(argv[2]):3840, n_rows=argc>3?atoi(argv[3]):15360;
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    N=(N/NR)*NR; n_rows=(n_rows/MR)*MR;             /* keep divisible for the test */
    int FT=n_rows/MR, TT=N/NR;
    uint16_t*W=(uint16_t*)malloc((size_t)n_rows*K*2);   /* row-major BF16 weight */
    float*X=(float*)malloc((size_t)N*K*4);              /* fp32 activations */
    for(size_t i=0;i<(size_t)n_rows*K;i++) W[i]=f2bf(0.02f*(float)((int)(i%61)-30)/30.0f);
    for(size_t i=0;i<(size_t)N*K;i++) X[i]=0.02f*(float)((int)(i%53)-26)/26.0f;

    uint16_t*Wp=(uint16_t*)malloc((size_t)FT*K*MR*2);   /* W-pack scratch (on-the-fly) */
    float*Y=(float*)malloc((size_t)N*n_rows*4);
    uint16_t*Xall=(uint16_t*)malloc((size_t)TT*K*NR*2);   /* all X tiles k-major BF16 */
    double freq=(double)rdfreq();
    /* TIMED: pack X once per tt-tile (k-major BF16), then kernel + C-transpose per (ft,tt). */
    #define GEMM() do{ \
      _Pragma("omp parallel for num_threads(nt) schedule(static)") \
      for(int ft=0;ft<FT;ft++){ uint16_t*dst=Wp+(size_t)ft*K*MR; \
          for(int k=0;k<K;k++) for(int i=0;i<16;i++){ \
              dst[k*MR+2*i]=W[(size_t)(ft*MR+i)*K+k]; dst[k*MR+2*i+1]=W[(size_t)(ft*MR+16+i)*K+k]; } } \
      _Pragma("omp parallel for num_threads(nt) schedule(static)") \
      for(int tt=0;tt<TT;tt++){ uint16_t*Xb=Xall+(size_t)tt*K*NR; \
          for(int k=0;k<K;k++) for(int n=0;n<NR;n++) Xb[k*NR+n]=f2bf(X[(size_t)(tt*NR+n)*K+k]); } \
      _Pragma("omp parallel num_threads(nt)") { \
        float*Ct=(float*)malloc((size_t)MR*NR*4); \
        _Pragma("omp for schedule(static) collapse(2)") \
        for(int ft=0;ft<FT;ft++) for(int tt=0;tt<TT;tt++){ \
            sgemm_bf16_2x12(K, Wp+(size_t)ft*K*MR, Xall+(size_t)tt*K*NR, Ct, MR); \
            for(int n=0;n<NR;n++) for(int mm=0;mm<MR;mm++) Y[(size_t)(tt*NR+n)*n_rows + ft*MR+mm]=Ct[mm+n*MR]; \
        } free(Ct); } }while(0)
    GEMM(); /* warm */
    uint64_t c0=rdcyc(); for(int r=0;r<3;r++) GEMM(); uint64_t c1=rdcyc();
    double t=(double)(c1-c0)/freq/3, gf=2.0*N*n_rows*K/t/1e9;

    /* reference (fp32) */
    double num=0,den=0;
    for(int tok=0;tok<N;tok+=37) for(int f=0;f<n_rows;f+=131){ double r=0;
        for(int k=0;k<K;k++) r+=(double)X[(size_t)tok*K+k]*bf2f(W[(size_t)f*K+k]);
        double e=Y[(size_t)tok*n_rows+f]-r; num+=e*e; den+=r*r; }
    printf("podd N=%d K=%d n_rows=%d: %.1f ms  %.0f GFLOPS (incl X-pack+transpose)  relL2=%.2e\n",
           N,K,n_rows, t*1e3, gf, sqrt(num/den));
    return 0;
}
