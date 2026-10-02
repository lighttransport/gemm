/* Depth Anything V2 Small standalone CPU / hybrid CUDA runner. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <omp.h>
#include <time.h>
#ifdef DA2_HAVE_AVX2
#include "../ryzen/gemm_avx2.h"
#endif
static int da2_threads=4, da2_gpu=0;
static void da2_linear(float *,const float *,const float *,const float *,int,int,int,int);
static void da2_conv(float *,const float *,const float *,const float *,int,int,int,int,int,int,int,int);
#define DINOV2_GEMM_F32 da2_linear
#define DA3_CONV2D_OVERRIDE da2_conv
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"
#define DINOV2_IMPLEMENTATION
#define DEPTH_ANYTHING3_IMPLEMENTATION
#define DEPTH_ANYTHING2_IMPLEMENTATION
#include "depth_anything2.h"
#ifdef DA2_CUDA
#include "../cuda/da2/cuda_da2_gemm.h"
#endif

static void *da2_alloc(size_t n)
{
    void *p=malloc(n);
    if (!p) { fprintf(stderr,"da2: allocation failed (%zu bytes)\n",n); exit(3); }
    return p;
}

/* Y[M,N]=X[M,K] W[N,K]^T+b. Preserve F32 weights/activations. */
static void da2_linear(float *y,const float *w,const float *b,const float *x,
                        int m,int n,int k,int threads)
{
#ifdef DA2_CUDA
    if (da2_gpu) {
        if (da2_cuda_linear(y,w,b,x,m,n,k)) { fprintf(stderr,"da2: CUDA GEMM failed\n"); exit(4); }
        return;
    }
#endif
#ifdef DA2_HAVE_AVX2
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        float *xt=da2_alloc((size_t)m*k*sizeof(float)), *yt=da2_alloc((size_t)m*n*sizeof(float));
        for (int i=0; i<m; i++) for (int j=0; j<k; j++) xt[(size_t)j*m+i]=x[(size_t)i*k+j];
        memset(yt,0,(size_t)m*n*sizeof(float));
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int row=0; row<n; row+=64) {
            int rows=n-row<64 ? n-row : 64;
            sgemm_avx2(rows,m,k,1,w+(size_t)row*k,k,xt,m,0,yt+(size_t)row*m,m);
        }
        for (int i=0; i<m; i++) for (int j=0; j<n; j++) y[(size_t)i*n+j]=yt[(size_t)j*m+i]+(b ? b[j] : 0);
        free(xt); free(yt); return;
    }
#endif
    cpu_gemm_f32(y,w,b,x,m,n,k,threads);
}

/* Bounded im2col tiles, using the same GEMM backend as the backbone. */
static void da2_conv(float *dst,const float *src,const float *w,const float *bias,
                      int h,int width,int ci,int co,int kh,int kw,int stride,int pad)
{
    int oh=(h+2*pad-kh)/stride+1, ow=(width+2*pad-kw)/stride+1;
    int total=oh*ow, k=ci*kh*kw, tile=1024;
    float *x=da2_alloc((size_t)tile*k*sizeof(float)), *y=da2_alloc((size_t)tile*co*sizeof(float));
    for (int start=0; start<total; start+=tile) {
        int count=total-start<tile ? total-start : tile;
        #pragma omp parallel for schedule(static) num_threads(da2_threads)
        for (int p=0; p<count; p++) {
            int iy=(start+p)/ow*stride-pad, ix=(start+p)%ow*stride-pad, off=0;
            for (int c=0; c<ci; c++) for (int ky=0; ky<kh; ky++) for (int kx=0; kx<kw; kx++) {
                int sy=iy+ky,sx=ix+kx;
                x[(size_t)p*k+off++]=(sy>=0 && sy<h && sx>=0 && sx<width) ? src[((size_t)c*h+sy)*width+sx] : 0;
            }
        }
        da2_linear(y,w,bias,x,count,co,k,da2_threads);
        for (int p=0; p<count; p++) for (int c=0; c<co; c++) dst[(size_t)c*total+start+p]=y[(size_t)p*co+c];
    }
    free(x); free(y);
}

static int number(const char *s)
{
    char *end; errno=0; long n=strtol(s,&end,10);
    return errno || end==s || *end || n<0 || n>65536 ? -1 : (int)n;
}

int main(int argc,char **argv)
{
    const char *backbone=NULL,*head=NULL,*input=NULL,*output=NULL,*dump=NULL,*backend="cpu";
    int w=0,h=0,ow=0,oh=0,device=0;
    for (int i=1;i<argc;i++) {
        if (i+1>=argc) goto usage;
        const char *flag=argv[i],*v=argv[++i];
        if (!strcmp(flag,"--backbone")) backbone=v;
        else if (!strcmp(flag,"--head")) head=v;
        else if (!strcmp(flag,"--input")) input=v;
        else if (!strcmp(flag,"--output")) output=v;
        else if (!strcmp(flag,"--dump-dir")) dump=v;
        else if (!strcmp(flag,"--backend")) backend=v;
        else if (!strcmp(flag,"--width")) w=number(v);
        else if (!strcmp(flag,"--height")) h=number(v);
        else if (!strcmp(flag,"--output-width")) ow=number(v);
        else if (!strcmp(flag,"--output-height")) oh=number(v);
        else if (!strcmp(flag,"--device")) device=number(v);
        else if (!strcmp(flag,"--threads")) da2_threads=number(v);
        else goto usage;
    }
    if (!backbone || !head || !input || !output || w<14 || h<14 || w%14 || h%14 ||
        (int64_t)(w/14)*(h/14)>4096 || ow<1 || oh<1 || (int64_t)ow*oh>4194304 ||
        da2_threads<1 || da2_threads>128 || device<0 || (strcmp(backend,"cpu") && strcmp(backend,"cuda"))) goto usage;
    da2_gpu=!strcmp(backend,"cuda");
#ifndef DA2_CUDA
    if (da2_gpu) { fprintf(stderr,"Use cuda/da2/da2_depth for CUDA\n"); return 2; }
#endif
    omp_set_num_threads(da2_threads);
    size_t count=(size_t)3*w*h;
    float *chw=da2_alloc(count*sizeof(float));
    FILE *f=fopen(input,"rb");
    if (!f) { free(chw); return 2; }
    int bad=fread(chw,sizeof(float),count,f)!=count || fgetc(f)!=EOF;
    fclose(f);
    for (size_t i=0; !bad && i<count; i++) if (!isfinite(chw[i])) bad=1;
    if (bad) { fprintf(stderr,"da2: invalid finite F32 CHW input\n"); free(chw); return 2; }
    da2_model *m=da2_load(backbone,head);
    if (!m) { free(chw); return 3; }
    int rc=1;
#ifdef DA2_CUDA
    if (da2_gpu && da2_cuda_init(device)) goto done;
#endif
    struct timespec start,end;
    clock_gettime(CLOCK_MONOTONIC,&start);
    float *depth=da2_predict(m,chw,w,h,ow,oh,da2_threads,dump);
    clock_gettime(CLOCK_MONOTONIC,&end);
    if (!depth) goto done;
    bad=0;
    for (size_t i=0;i<(size_t)ow*oh;i++) if (!isfinite(depth[i]) || depth[i]<0) bad=1;
    if (bad) { free(depth); fprintf(stderr,"da2: invalid depth output\n"); goto done; }
    f=fopen(output,"wb");
    if (!f) { free(depth); goto done; }
    bad=fwrite(depth,sizeof(float),(size_t)ow*oh,f)!=(size_t)ow*oh;
    bad |= fclose(f)!=0; free(depth);
    if (bad) goto done;
    printf("{\"backend\":\"%s\",\"execution\":\"%s\",\"seconds\":%.6f,\"width\":%d,\"height\":%d}\n",
           backend,da2_gpu ? "cuda_gemm_cpu_attention" : "native_cpu",
           end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)*1e-9,ow,oh);
    rc=0;
done:
#ifdef DA2_CUDA
    da2_cuda_free();
#endif
    da2_free(m); free(chw); return rc;
usage:
    fprintf(stderr,"da2_depth --backbone FILE --head FILE --input CHW.f32 --width W --height H "
                   "--output FILE --output-width W --output-height H [--backend cpu|cuda] "
                   "[--device N] [--threads N] [--dump-dir DIR]\n");
    return 2;
}
