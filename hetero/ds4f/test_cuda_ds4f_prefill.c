#include "../../common/ds4f.h"
#include "cuda_ds4f_dense.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char **argv){
    if(argc<2){fprintf(stderr,"usage: %s STAGE_DIR [M] [THREADS] [CMGS] [ZEROCOPY] [W4A8]\n",argv[0]);return 2;}
    int M=argc>2?atoi(argv[2]):3072,nthr=argc>3?atoi(argv[3]):16,ncmg=argc>4?atoi(argv[4]):2,zc=argc>5?atoi(argv[5]):1,w4a8=argc>6?atoi(argv[6]):1;setenv("DS4F_PREFILL_LAST_LOGITS","1",0);setenv("DS4F_PROF","1",0);
    ds4f_runtime_options o;ds4f_runtime_options_init(&o);o.cfg=ds4f_default_config();o.ep_size=8;o.ep_rank=0;o.n_threads=nthr;o.n_cmgs=ncmg;o.mxfp4_w4a8=w4a8;o.zero_copy_experts=zc;
    snprintf(o.stage_dir,sizeof(o.stage_dir),"%s",argv[1]);ds4f_model *m=ds4f_load_real_opts(&o);if(!m)return 3;
    setenv("DS4F_CUDA_MXFP4_CACHE_MB","512",0);
    cuda_ds4f_dense *c=cuda_ds4f_dense_create(0,0);if(!c)return 4;int nbind=0;
    for(int L=0;L<m->cfg.n_layers;L++){ds4f_layer *z=&m->layers[L];ds4f_tensor *ts[]={&z->wq_a,&z->wq_b,&z->wkv,&z->wo_a,&z->wo_b,&z->sh_w1,&z->sh_w3,&z->sh_w2};
        for(size_t i=0;i<sizeof(ts)/sizeof(ts[0]);i++)if(cuda_ds4f_dense_bind_tensor(c,ts[i])<0){fprintf(stderr,"bind failed L=%d i=%zu\n",L,i);return 5;}else nbind++;}
    if (cuda_ds4f_dense_bind_tensor(c, &m->head) < 0) return 6;
    nbind++;
    for(int L=0;L<m->cfg.n_layers;L++)for(int e=0;e<m->layers[L].n_owned;e++){
        ds4f_layer *z=&m->layers[L];ds4f_tensor *ts[]={&z->ex_w1[e],&z->ex_w3[e],&z->ex_w2[e]};
        for(int i=0;i<3;i++)if(cuda_ds4f_dense_bind_tensor(c,ts[i])<0)return 7;else nbind++;
    }
    m->gpu_dense_ctx=c;m->gpu_dense_gemm=cuda_ds4f_dense_gemm_tensor;m->gpu_shared_ffn=cuda_ds4f_dense_shared_ffn;m->gpu_shared_ffn_begin=cuda_ds4f_dense_shared_ffn_begin;m->gpu_shared_ffn_wait=cuda_ds4f_dense_shared_ffn_wait;m->gpu_oproj=cuda_ds4f_dense_oproj;m->gpu_head_argmax=cuda_ds4f_dense_head_argmax;m->gpu_dense_mixed=1;
    ds4f_alloc_prefill_batch(m,M);float *x=malloc((size_t)M*m->cfg.hidden*4);int *tok=malloc((size_t)M*sizeof(int));
    for(size_t i=0;i<(size_t)M*m->cfg.hidden;i++)x[i]=((int)(i*29%101)-50)/37.f;
    double t0=now_s();ds4f_forward_prefill(m,x,M,0,tok);double dt=now_s()-t0;
    printf("CUDA-only prefill: layers=%d matrices=%d M=%d %.3f tok/s last=%d\n",m->cfg.n_layers,nbind,M,M/dt,tok[M-1]);
    printf("phase ms/token:");for(int i=0;i<DS4F_NPHASE;i++)if(m->prof[i]>0)printf(" %s=%.3f",ds4f_prof_names[i],m->prof[i]*1000/M);printf("\n");
    free(tok);free(x);cuda_ds4f_dense_destroy(c);ds4f_free(m);return 0;
}
