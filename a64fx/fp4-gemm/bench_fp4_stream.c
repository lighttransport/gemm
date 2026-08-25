#define _POSIX_C_SOURCE 200112L
#include "fp4_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static void*aa(size_t n){void*p=0;return posix_memalign(&p,256,(n+255)&~255ULL)?0:p;}
static unsigned rng=7;static unsigned rnd(void){rng=rng*1664525u+1013904223u;return rng;}
typedef int(*kernel)(float*,const _Float16*,const fp4_matrix*,int,int);
static void run(const char*name,kernel fn,float*c,const _Float16*a,const fp4_matrix*w,int m,int kc,size_t src){
    fn(c,a,w,m,kc);double best=1e9;for(int r=0;r<3;++r){double t=now();fn(c,a,w,m,kc);double d=now()-t;if(d<best)best=d;}
    printf("kernel=%s M=%d kc=%d ms=%.2f gflops=%.2f source_GB/s=%.2f\n",name,m,kc,best*1e3,
        2.0*m*w->n*w->k/best/1e9,src/best/1e9);
}
static void run_omp(float*c,const _Float16*a,const fp4_matrix*w,int m,int kc,size_t src,int threads){
    fp4_gemm_f16_l2_omp(c,a,w,m,kc,threads);double best=1e9;
    for(int r=0;r<3;++r){double t=now();fp4_gemm_f16_l2_omp(c,a,w,m,kc,threads);double d=now()-t;if(d<best)best=d;}
    printf("kernel=l2fused threads=%d M=%d kc=%d ms=%.2f gflops=%.2f source_GB/s=%.2f\n",threads,m,kc,best*1e3,
        2.0*m*w->n*w->k/best/1e9,src/best/1e9);
}
static void run_direct_omp(float*c,const _Float16*a,const fp4_matrix*w,int m,int kc,size_t src,int threads){
    fp4_gemm_f16_n32_omp(c,a,w,m,kc,threads);double best=1e9;
    for(int r=0;r<3;++r){double t=now();fp4_gemm_f16_n32_omp(c,a,w,m,kc,threads);double d=now()-t;if(d<best)best=d;}
    size_t passes=(size_t)(m+5)/6;
    printf("kernel=directfused threads=%d M=%d kc=%d ms=%.2f gflops=%.2f source_GB/s=%.2f\n",threads,m,kc,best*1e3,
        2.0*m*w->n*w->k/best/1e9,src*passes/best/1e9);
}
static void run_u8_omp(float*c,const _Float16*a,const fp4_matrix*w,int kc,size_t src,int threads){
    fp4_gemm_f16_u8tbl_omp(c,a,w,1,kc,threads);double best=1e9;
    for(int r=0;r<3;++r){double t=now();fp4_gemm_f16_u8tbl_omp(c,a,w,1,kc,threads);double d=now()-t;if(d<best)best=d;}
    printf("kernel=u8tbl threads=%d M=1 kc=%d ms=%.2f gflops=%.2f source_GB/s=%.2f\n",threads,kc,best*1e3,
        2.0*w->n*w->k/best/1e9,src/best/1e9);
}
static void run_bitplane_omp(float*c,const _Float16*a,const fp4_matrix*w,int kc,size_t src,int threads){
    double dt[7];fp4_gemm_f16_bitplane_omp(c,a,w,1,kc,threads);
    for(int r=0;r<7;++r){double t=now();fp4_gemm_f16_bitplane_omp(c,a,w,1,kc,threads);dt[r]=now()-t;}
    for(int i=1;i<7;++i){double x=dt[i];int j=i;while(j&&dt[j-1]>x){dt[j]=dt[j-1];--j;}dt[j]=x;}
    double med=dt[3];
    printf("kernel=bitplane threads=%d M=1 kc=%d median_ms=%.2f gflops=%.2f source_GB/s=%.2f\n",threads,kc,med*1e3,
        2.0*w->n*w->k/med/1e9,src/med/1e9);
}
static void run_sdot_omp(float*c,const fp4_i8_activation*a,const fp4_matrix*w,size_t src,int threads){
    double dt[7];fp4_gemv_i8_sdot_omp(c,a,w,threads);
    for(int r=0;r<7;++r){double t=now();fp4_gemv_i8_sdot_omp(c,a,w,threads);dt[r]=now()-t;}
    for(int i=1;i<7;++i){double x=dt[i];int j=i;while(j&&dt[j-1]>x){dt[j]=dt[j-1];--j;}dt[j]=x;}
    double med=dt[3];printf("kernel=fp4_i8_sdot threads=%d A_G=%d median_ms=%.3f gflops=%.2f source_GB/s=%.2f checksum=%.7g\n",
      threads,a->scale_group,med*1e3,2.0*w->n*w->k/med/1e9,src/med/1e9,c[w->n/3]);
}
static double run_pair_omp(float*c,const fp4_pair_activation*a,const fp4_matrix*w,size_t src,int threads){
    double dt[7];fp4_gemv_pair_lut_omp(c,a,w,threads);
    for(int r=0;r<7;++r){double t=now();fp4_gemv_pair_lut_omp(c,a,w,threads);dt[r]=now()-t;}
    for(int i=1;i<7;++i){double x=dt[i];int j=i;while(j&&dt[j-1]>x){dt[j]=dt[j-1];--j;}dt[j]=x;}
    double med=dt[3];printf("kernel=fp4_pair_lut threads=%d A_G=%d median_ms=%.3f gflops=%.2f packed_GB/s=%.2f checksum=%.7g\n",
      threads,a->scale_group,med*1e3,2.0*w->n*w->k/med/1e9,src/med/1e9,c[w->n/3]);return med;
}
int main(int argc,char**argv){int n=argc>1?atoi(argv[1]):32768,k=argc>2?atoi(argv[2]):4096;
    int kc=argc>3?atoi(argv[3]):256;
    if(n%32||k%32)return 2;fp4_matrix w;if(fp4_matrix_alloc(&w,FP4_MX,n,k))return 1;
    for(size_t i=0;i<w.code_bytes;++i)w.codes[i]=(uint8_t)rnd();
    for(size_t i=0;i<w.scale_bytes;++i)w.scales[i]=124; /* 2^-3 */
    double pt=now();if(fp4_matrix_prepare_n32(&w)||fp4_matrix_prepare_u8(&w)||
      fp4_matrix_prepare_bitplane(&w)||fp4_matrix_prepare_sdot(&w)||fp4_matrix_prepare_pair(&w))return 1;
    printf("all_layout_prepare_ms=%.3f\n",(now()-pt)*1e3);
    size_t source=w.code_bytes+w.scales_n32_count*sizeof(_Float16);
    printf("streaming MXFP4 N=%d K=%d packed=%.1f MiB prepared_scales=%.1f MiB source=%.1f MiB\n",n,k,
        w.code_bytes/1048576.,w.scales_n32_count*2/1048576.,source/1048576.);
    void*copy=aa(source);double cbest=1e9;volatile unsigned char guard=0;
    for(int r=0;r<5;++r){double t=now();memcpy(copy,w.codes_n32,w.code_bytes);
      memcpy((char*)copy+w.code_bytes,w.scales_n32,w.scales_n32_count*2);double d=now()-t;
      guard^=((unsigned char*)copy)[(size_t)r*4096];if(d<cbest)cbest=d;}
    printf("pure_memcpy ms=%.2f source_GB/s=%.2f total_read_write_GB/s=%.2f guard=%u\n",
        cbest*1e3,source/cbest/1e9,2.0*source/cbest/1e9,(unsigned)guard);free(copy);
    copy=aa(source);cbest=1e9;
    for(int r=0;r<5;++r){double t=now();
#pragma omp parallel num_threads(12)
      {int id=omp_get_thread_num(),nt=omp_get_num_threads();
       size_t q0=w.code_bytes*id/nt,q1=w.code_bytes*(id+1)/nt;
       size_t sn=w.scales_n32_count*2,s0=sn*id/nt,s1=sn*(id+1)/nt;
       memcpy((char*)copy+q0,w.codes_n32+q0,q1-q0);
       memcpy((char*)copy+w.code_bytes+s0,(char*)w.scales_n32+s0,s1-s0);}
      double d=now()-t;if(d<cbest)cbest=d;}
    guard^=((unsigned char*)copy)[4096];
    printf("pure_memcpy_12c ms=%.2f source_GB/s=%.2f total_read_write_GB/s=%.2f guard=%u\n",
        cbest*1e3,source/cbest/1e9,2.0*source/cbest/1e9,(unsigned)guard);free(copy);
    if(getenv("FP4_SDOT_ONLY")){int ag=argc>4?atoi(argv[4]):4;
      _Float16*ah=aa((size_t)k*2);float*af=aa((size_t)k*4),*c=aa((size_t)n*4);
      fp4_i8_activation qa={0};if(!ah||!af||!c)return 1;
      for(int i=0;i<k;++i){ah[i]=(_Float16)((int)(rnd()&255)-128)/512;af[i]=(float)ah[i];}
      if(fp4_i8_activation_prepare(&qa,af,k,ag))return 1;
      run_sdot_omp(c,&qa,&w,(size_t)n*k+w.scales_n32_count*2,12);
      fp4_i8_activation_free(&qa);free(ah);free(af);free(c);fp4_matrix_free(&w);return 0;}
    if(getenv("FP4_PAIR_ONLY")){int ag=argc>4?atoi(argv[4]):4;
      float*af=aa((size_t)k*4),*c=aa((size_t)n*4);fp4_pair_activation qa={0};
      if(!af||!c)return 1;for(int i=0;i<k;++i)af[i]=(float)((int)(rnd()&255)-128)/512;
      double qt=now();if(fp4_pair_activation_prepare(&qa,af,k,ag))return 1;qt=now()-qt;
      printf("pair_activation_prepare A_G=%d ms=%.3f table_MiB=%.2f\n",ag,qt*1e3,(double)k*256/1048576.);
      double kt=run_pair_omp(c,&qa,&w,source,12);
      printf("pair_end_to_end_once gflops=%.2f\n",2.0*n*k/(qt+kt)/1e9);
      fp4_pair_activation_free(&qa);free(af);free(c);fp4_matrix_free(&w);return 0;}
    int ms[]={1,6,24,128};for(int z=0;z<4;++z){int m=ms[z];
      _Float16*a=aa((size_t)m*k*2);float*c=aa((size_t)m*n*4);if(!a||!c)return 1;
      for(size_t i=0;i<(size_t)m*k;++i)a[i]=(_Float16)((int)(rnd()&255)-128)/512;
      run("direct",fp4_gemm_f16_n32,c,a,&w,m,kc,source*((m+5)/6));
      run("l2full",fp4_gemm_f16_l2,c,a,&w,m,kc,source);
      if(kc)run("l1panel",fp4_gemm_f16_l1panel,c,a,&w,m,kc,source);
      run_direct_omp(c,a,&w,m,kc,source,12);
      if(m==1)run_u8_omp(c,a,&w,kc,(size_t)n*k+w.scales_n32_count*2,12);
      if(m==1)run_bitplane_omp(c,a,&w,kc,source,12);
      if(m==1){int ag=argc>4?atoi(argv[4]):4;float*fa=aa((size_t)k*4);fp4_i8_activation qa={0};
        if(!fa)return 1;for(int i=0;i<k;++i)fa[i]=(float)a[i];
        if(fp4_i8_activation_prepare(&qa,fa,k,ag))return 1;
        run_sdot_omp(c,&qa,&w,(size_t)n*k+w.scales_n32_count*2,12);
        fp4_i8_activation_free(&qa);free(fa);}
      run_omp(c,a,&w,m,kc,source,12);
      free(a);free(c);}fp4_matrix_free(&w);return 0;}
