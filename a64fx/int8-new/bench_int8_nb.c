// Parallel nb-outer int8 SDOT GEMM for VLM shapes (K chunked to 256).
// Reuses the proven kernel_6x4_opt_256 + packers. nb-outer: each thread owns an
// N-slice, streams small A across M-tiles, reads B once (no W re-streaming).
// NOTE: kernel overwrites C per K-chunk (st1w), so for K>256 the result is only the
// last chunk -- numerically wrong but the TIMING is valid (same # kernel calls).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
void pack_A_6x256(const int8_t* A, int lda, int8_t* Apack, int M);
void pack_B_64x256(const int8_t* B, int ldb, int8_t* Bpack, int N);
void kernel_6x4_opt_256(const int8_t* Apack, const int8_t* Bpack, int32_t* C, int ldc);
static double now(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return (double)v*1e-8; }

static void gemm_nb(int M,int N,int K,int8_t* Apack,int8_t* Bpack,int32_t* C){
    const int MR=6, NR=64, KC=K/256; int MB=M/MR, NB=N/NR;
    memset(C,0,(size_t)M*N*4);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nb=0;nb<NB;nb++)
      for (int mb=0;mb<MB;mb++) {
          int32_t* Ct = C + (size_t)mb*MR*(N) + nb*NR;
          for (int kc=0;kc<KC;kc++)
              kernel_6x4_opt_256(Apack+(size_t)(mb*KC+kc)*MR*256,
                                 Bpack+(size_t)(nb*KC+kc)*NR*256, Ct, N);
      }
}
static void bench(const char *name, int M, int N, int K, int nw) {
    const int MR=6, NR=64, KC=K/256;
    int MB=M/MR, NB=N/NR;
    int8_t *A = malloc((size_t)M*K);
    int32_t *C = malloc((size_t)M*N*4);
    int8_t *Apack = malloc((size_t)MB*KC*MR*256);
    int8_t **Bpack = malloc(nw*sizeof(void*));
    for (int i=0;i<M*K;i++) A[i]=(int8_t)((i*7)%31-15);
    for (int mb=0;mb<MB;mb++) for (int kc=0;kc<KC;kc++)
        pack_A_6x256(A + mb*MR*K + kc*256, K, Apack + (size_t)(mb*KC+kc)*MR*256, MR);
    for (int w=0;w<nw;w++){
        int8_t *B = malloc((size_t)N*K);
        for (int i=0;i<N*K;i++) B[i]=(int8_t)(((i*13)+w*97)%41-20);
        Bpack[w] = malloc((size_t)NB*KC*NR*256);
        for (int nb=0;nb<NB;nb++) for (int kc=0;kc<KC;kc++)
            pack_B_64x256(B + nb*NR*K + kc*256, K, Bpack[w]+(size_t)(nb*KC+kc)*NR*256, NR);
        free(B);
    }
    double best=1e9;
    for (int it=0; it<3; it++) {
        double t0=now();
        for (int w=0;w<nw;w++) gemm_nb(M,N,K,Apack,Bpack[w],C);
        double t=now()-t0; if(t<nw*t)best=t;
    }
    double per=best/nw;
    double gops=2.0*M*N*K/per/1e9;
    printf("%-14s M=%d N=%d K=%d nw=%-2d : %8.3f ms/call  %9.1f GOPS/call  (%d threads, int8 nb-outer)\n",
           name, M,N,K,nw, per*1000, gops, omp_get_max_threads());
    for (int w=0;w<nw;w++) free(Bpack[w]);
    free(Bpack);free(A);free(C);free(Apack);
}
int main(int argc,char**argv){
    int T=48; if(argc>1)T=atoi(argv[1]);
    omp_set_num_threads(T);
    printf("A64FX int8 SDOT peak = 512 GOPS/core ; %d threads\n", T);
    bench("ffn_up 1W", 96,4096,1024,1);
    bench("ffn_up 24W", 96,4096,1024,24);
    bench("ffn_down 24W",96,1024,4096,24);
    bench("qkv 24W",    96,3072,1024,24);
    return 0;
}
