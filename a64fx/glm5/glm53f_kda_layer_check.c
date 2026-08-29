/* Real-weight stateful GLM-5.3F KDA decode-layer correctness/perf check. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"
#include <arm_sve.h>
#include <omp.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { H=4096, NH=64, D=128, QKV=8192, KERNEL=4 };

typedef struct {
    float *conv;                 /* [3,QKV,KERNEL] */
    float *recurrent;            /* [NH,D,D] */
} kda_cache;

typedef struct {
    uint16_t *q, *k, *v, *qconv, *kconv, *vconv;
    uint16_t *fa, *fb, *b, *ga, *gb, *onorm, *op;
    float *alog, *dt;
} kda_weight;

typedef struct {
    float *qkv, *fsmall, *gate, *decay, *beta, *core, *normed, *work;
} kda_scratch;

static double now_sec(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}
static void *a256(size_t n) { void *p=NULL; return posix_memalign(&p,256,n)?NULL:p; }
static inline float dot_bf16_sve(const uint16_t *w,const float *x,int n) {
    svfloat32_t acc=svdup_f32(0.0f); int vl=(int)svcntw();
    for(int i=0;i<n;i+=vl){
        svbool_t pg=svwhilelt_b32(i,n);
        svuint32_t bits=svlsl_n_u32_x(pg,svld1uh_u32(pg,w+i),16);
        acc=svmla_x(pg,acc,svreinterpret_f32_u32(bits),svld1(pg,x+i));
    }
    return svaddv_f32(svptrue_b32(),acc);
}
static inline void dot_bf16_sve_8(float *y,const uint16_t *w,
                                  const float *x,int n) {
    svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
    int vl=(int)svcntw();
    for(int i=0;i<n;i+=vl){
        svbool_t pg=svwhilelt_b32(i,n); svfloat32_t xv=svld1(pg,x+i);
#define ROW(R,A) do { svuint32_t bits=svlsl_n_u32_x(pg, \
            svld1uh_u32(pg,w+(size_t)(R)*n+i),16); \
            A=svmla_x(pg,A,svreinterpret_f32_u32(bits),xv); } while(0)
        ROW(0,a0);ROW(1,a1);ROW(2,a2);ROW(3,a3);
        ROW(4,a4);ROW(5,a5);ROW(6,a6);ROW(7,a7);
#undef ROW
    }
    svbool_t pg=svptrue_b32();
    y[0]=svaddv_f32(pg,a0);y[1]=svaddv_f32(pg,a1);
    y[2]=svaddv_f32(pg,a2);y[3]=svaddv_f32(pg,a3);
    y[4]=svaddv_f32(pg,a4);y[5]=svaddv_f32(pg,a5);
    y[6]=svaddv_f32(pg,a6);y[7]=svaddv_f32(pg,a7);
}
static void *get_tensor(glm53f_st_context *st,const char *name) {
    const st_tensor_info *t=glm53f_st_find(st,name,NULL); void *p;
    if(!t){fprintf(stderr,"missing %s\n",name);return NULL;}
    p=a256(t->nbytes);
    if(!p||glm53f_st_read(st,name,0,p,t->nbytes)){
        fprintf(stderr,"read failed %s\n",name);free(p);return NULL;
    }
    return p;
}
static void mv(float *y,const uint16_t *w,const float *x,int rows,int cols) {
    int blocks=rows/8;
#pragma omp parallel for schedule(static)
    for(int b=0;b<blocks;b++)
        dot_bf16_sve_8(y+b*8,w+(size_t)b*8*cols,x,cols);
    for(int r=blocks*8;r<rows;r++)
        y[r]=dot_bf16_sve(w+(size_t)r*cols,x,cols);
}
static int alloc_cache(kda_cache *c) {
    c->conv=a256((size_t)3*QKV*KERNEL*sizeof(float));
    c->recurrent=a256((size_t)NH*D*D*sizeof(float));
    if(!c->conv||!c->recurrent)return -1;
    memset(c->conv,0,(size_t)3*QKV*KERNEL*sizeof(float));
    memset(c->recurrent,0,(size_t)NH*D*D*sizeof(float));
    return 0;
}
static void copy_cache(kda_cache *dst,const kda_cache *src) {
    memcpy(dst->conv,src->conv,(size_t)3*QKV*KERNEL*sizeof(float));
    memcpy(dst->recurrent,src->recurrent,(size_t)NH*D*D*sizeof(float));
}
static int forward(float *out,const float *x,const kda_weight *w,
                   kda_cache *cache,kda_scratch *s) {
    float *q=s->qkv,*k=q+QKV,*v=k+QKV;
    mv(q,w->q,x,QKV,H); mv(k,w->k,x,QKV,H); mv(v,w->v,x,QKV,H);
    glm53f_causal_conv1d_silu_bf16(q,cache->conv,q,w->qconv,QKV,KERNEL);
    glm53f_causal_conv1d_silu_bf16(k,cache->conv+(size_t)QKV*KERNEL,k,w->kconv,QKV,KERNEL);
    glm53f_causal_conv1d_silu_bf16(v,cache->conv+(size_t)2*QKV*KERNEL,v,w->vconv,QKV,KERNEL);
    mv(s->fsmall,w->fa,x,D,H); mv(s->gate,w->fb,s->fsmall,QKV,D);
    mv(s->beta,w->b,x,NH,H);
    for(int h=0;h<NH;h++){
        glm53f_l2norm(q+(size_t)h*D,D,1e-6f);
        glm53f_l2norm(k+(size_t)h*D,D,1e-6f);
        glm53f_kda_safe_log_decay(s->decay+(size_t)h*D,s->gate+(size_t)h*D,
                                  w->dt+(size_t)h*D,w->alog[h],-5.0f,D);
        s->beta[h]=glm53f_sigmoid(s->beta[h]);
    }
#pragma omp parallel for schedule(static)
    for(int h=0;h<NH;h++)
        glm53f_kda_step_vec_streamed(cache->recurrent+(size_t)h*D*D,
            q+(size_t)h*D,k+(size_t)h*D,v+(size_t)h*D,
            s->decay+(size_t)h*D,s->beta[h],D,D,s->core+(size_t)h*D,
            s->work+(size_t)h*D);
    mv(s->fsmall,w->ga,x,D,H); mv(s->gate,w->gb,s->fsmall,QKV,D);
    glm53f_rmsnorm_gated_bf16(s->normed,s->core,s->gate,w->onorm,NH,D,1e-5f);
    mv(out,w->op,s->normed,H,QKV);
    for(int i=0;i<H;i++)if(!isfinite(out[i]))return -1;
    return 0;
}

int main(int argc,char **argv) {
    glm53f_st_context *st; kda_weight w={0}; kda_cache c={0},checkpoint={0};
    kda_scratch s={0}; float *x,*out,*replay; char name[256]; int layer=44;
    if(argc<2){fprintf(stderr,"usage: %s MODEL_DIR [layer=44]\n",argv[0]);return 2;}
    if(argc>2)layer=atoi(argv[2]);
    st=glm53f_st_open(argv[1]); if(!st)return 2;
#define GET(F,S,T) do { snprintf(name,sizeof name,"model.language_model.layers.%d.self_attn.%s",layer,S); w.F=(T*)get_tensor(st,name); if(!w.F)return 2; } while(0)
    GET(alog,"A_log",float); GET(dt,"dt_bias",float);
    GET(q,"q_proj.weight",uint16_t); GET(k,"k_proj.weight",uint16_t);
    GET(v,"v_proj.weight",uint16_t); GET(qconv,"q_conv1d.weight",uint16_t);
    GET(kconv,"k_conv1d.weight",uint16_t); GET(vconv,"v_conv1d.weight",uint16_t);
    GET(fa,"f_a_proj.weight",uint16_t); GET(fb,"f_b_proj.weight",uint16_t);
    GET(b,"b_proj.weight",uint16_t); GET(ga,"g_a_proj.weight",uint16_t);
    GET(gb,"g_b_proj.weight",uint16_t); GET(onorm,"o_norm.weight",uint16_t);
    GET(op,"o_proj.weight",uint16_t);
#undef GET
    glm53f_st_close(st);
    if(alloc_cache(&c)||alloc_cache(&checkpoint))return 2;
    x=a256(H*sizeof(float));out=a256(H*sizeof(float));replay=a256(H*sizeof(float));
    s.qkv=a256((size_t)3*QKV*sizeof(float));s.fsmall=a256(D*sizeof(float));
    s.gate=a256(QKV*sizeof(float));s.decay=a256(QKV*sizeof(float));
    s.beta=a256(NH*sizeof(float));s.core=a256(QKV*sizeof(float));
    s.normed=a256(QKV*sizeof(float));s.work=a256(QKV*sizeof(float));
    if(!x||!out||!replay||!s.qkv||!s.fsmall||!s.gate||!s.decay||
       !s.beta||!s.core||!s.normed||!s.work)return 2;
    for(int i=0;i<H;i++)x[i]=(float)(((i*17+3)%251)-125)/125.0f;
    double t0=now_sec(); if(forward(out,x,&w,&c,&s))return 1; double first=now_sec()-t0;
    copy_cache(&checkpoint,&c);
    for(int i=0;i<H;i++)x[i]=(float)(((i*29+7)%257)-128)/128.0f;
    t0=now_sec();if(forward(out,x,&w,&c,&s))return 1;double decode=now_sec()-t0;
    copy_cache(&c,&checkpoint);
    if(forward(replay,x,&w,&c,&s))return 1;
    int exact=!memcmp(out,replay,H*sizeof(float)); double ss=0.0;
    for(int i=0;i<H;i++)ss+=(double)out[i]*out[i];
    printf("GLM53F_KDA_LAYER layer=%d first_ms=%.3f decode_ms=%.3f rms=%.9g rollback_replay=%s %s\n",
           layer,first*1e3,decode*1e3,sqrt(ss/H),exact?"BIT_EXACT":"FAIL",exact?"PASS":"FAIL");
    return exact?0:1;
}
