/* Standalone native CUDA GEMM check against naive double accumulation. */
#include "vhuman_training.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(int argc,char **argv)
{
    if(argc==2&&!strcmp(argv[1],"--compile-only")) {
        if(vht_compile_probe()) { fprintf(stderr,"%s\n",vht_error());return 1; }
        puts("native CUDA training NVRTC compile PASS");return 0;
    }
    if(argc!=1) { fprintf(stderr,"usage: %s [--compile-only]\n",argv[0]);return 2; }
    enum { M=19,N=23,K=37 };
    float a[M*K],b[K*N],out[M*N],parameter=0;
    vht_trainer *t=vht_open(0,&parameter,1,.01,0,64*1048576);
    if(!t) { fprintf(stderr,"%s\n",vht_error());return 1; }
    for(int i=0;i<M*K;++i)a[i]=(float)((i*17%71)-35)/37;
    for(int i=0;i<K*N;++i)b[i]=(float)((i*23%97)-48)/51;
    double max_error=0;
    for(int ta=0;ta<2;++ta)for(int tb=0;tb<2;++tb) {
        if(vht_gemm(t,out,a,b,M,N,K,ta,tb)) { fprintf(stderr,"%s\n",vht_error());vht_close(t);return 1; }
        for(int r=0;r<M;++r)for(int c=0;c<N;++c) {
            double expected=0;
            for(int k=0;k<K;++k)expected+=(double)a[ta?k*M+r:r*K+k]*b[tb?c*K+k:k*N+c];
            double error=fabs(out[r*N+c]-expected);if(error>max_error)max_error=error;
        }
    }
    vht_close(t);
    printf("native CUDA GEMM four transpose/tail cases max_error=%.9g %s\n",max_error,max_error<2e-5?"PASS":"FAIL");
    return max_error<2e-5?0:1;
}
