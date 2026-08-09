#include "../../common/ds4f.h"
#include "cuda_ds4f_dense.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr,"usage: %s STAGE_DIR [M] [wq_a|wq_b|head]\n",argv[0]); return 2; }
    int M=argc>2?atoi(argv[2]):64;
    ds4f_runtime_options o; ds4f_runtime_options_init(&o);
    o.cfg=ds4f_default_config(); o.cfg.n_layers=1; o.ep_size=8; o.ep_rank=0; o.n_threads=32; o.n_cmgs=1;
    snprintf(o.stage_dir,sizeof(o.stage_dir),"%s",argv[1]);
    ds4f_model *m=ds4f_load_real_opts(&o); if(!m)return 3;
    const char *name=argc>3?argv[3]:"wq_a";
    ds4f_tensor *t=!strcmp(name,"wq_b")?&m->layers[0].wq_b:!strcmp(name,"head")?&m->head:&m->layers[0].wq_a;
    int N=t->rows,K=t->cols;
    float *x=malloc((size_t)M*K*4),*ref=malloc((size_t)M*N*4),*got=malloc((size_t)M*N*4);
    if(!x||!ref||!got)return 4;
    for(size_t i=0;i<(size_t)M*K;i++)x[i]=((int)(i*29%101)-50)/37.f;
    double tc=now_s(); ds4f_gemm(m,ref,t,x,M,N,K); tc=now_s()-tc;
    cuda_ds4f_dense *c=cuda_ds4f_dense_create(0,1); if(!c)return 5;
    if(cuda_ds4f_dense_bind_tensor(c,t)<0||cuda_ds4f_dense_gemm_tensor(c,got,t,x,M,N,K))return 6;
    double tg=now_s(); for(int i=0;i<10;i++)if(cuda_ds4f_dense_gemm_tensor(c,got,t,x,M,N,K))return 7; tg=(now_s()-tg)/10;
    double se=0,sr=0;float ma=0;int mmis=0;
    for(int j=0;j<M;j++){int ar=0,ag=0;for(int n=0;n<N;n++){float d=fabsf(ref[(size_t)j*N+n]-got[(size_t)j*N+n]);if(d>ma)ma=d;se+=(double)d*d;sr+=(double)ref[(size_t)j*N+n]*ref[(size_t)j*N+n];if(ref[(size_t)j*N+n]>ref[(size_t)j*N+ar])ar=n;if(got[(size_t)j*N+n]>got[(size_t)j*N+ag])ag=n;}mmis+=ar!=ag;}
    double l2=sqrt(se/(sr+1e-30));
    int pass=l2<4e-2&&(!strcmp(name,"head")?mmis==0:1);
    printf("real CUDA dense %s M=%d N=%d K=%d rel_l2=%.6g max_abs=%.6g argmax=%d/%d cpu=%.3fms cuda=%.3fms %s\n",name,M,N,K,l2,ma,mmis,M,tc*1e3,tg*1e3,pass?"PASS":"FAIL");
    cuda_ds4f_dense_destroy(c);ds4f_free(m);free(got);free(ref);free(x);
    return pass?0:1;
}
