#ifndef Q38FN_TP_RUNTIME_H
#define Q38FN_TP_RUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "q38fn_tp_blob.h"

typedef void (*q38fn_tp_sum_fn)(float *values, int count, void *opaque);
typedef void (*q38fn_tp_argmax_fn)(float *value, int *index, void *opaque);

typedef struct { float *conv,*recurrent; } q38fn_tp_delta_state;
typedef struct { float *keys,*values,*scores;size_t length,capacity; } q38fn_tp_attention_state;
typedef struct { float *conv;uint64_t previous,previous2; } q38fn_tp_ple_state;
typedef struct {
    q38fn_tp_blob blob;
    q38fn_tp_blob hc_blob;
    q38fn_tp_blob q8_blob;
    q38fn_tp_blob aux_blob;
    int has_hc_blob;
    int has_q8_blob;
    int has_aux_blob;
    int rank,ranks;
    q38fn_tp_sum_fn sum;
    q38fn_tp_argmax_fn argmax;
    void *comm;
    float *head_logits;
    size_t head_logits_count;
} q38fn_tp_model;

int q38fn_tp_model_open(q38fn_tp_model*,const char*,int,int,q38fn_tp_sum_fn,q38fn_tp_argmax_fn,void*);
void q38fn_tp_model_close(q38fn_tp_model*);
int q38fn_tp_delta_init(q38fn_tp_delta_state*);void q38fn_tp_delta_close(q38fn_tp_delta_state*);
int q38fn_tp_attention_init(q38fn_tp_attention_state*,size_t);void q38fn_tp_attention_close(q38fn_tp_attention_state*);
int q38fn_tp_ple_init(q38fn_tp_ple_state*);void q38fn_tp_ple_close(q38fn_tp_ple_state*);
int q38fn_tp_embedding(q38fn_tp_model*,int,float*);
int q38fn_tp_ngram(q38fn_tp_model*,uint64_t,uint64_t,uint64_t,float*);
int q38fn_tp_ngram_local(q38fn_tp_model*,uint64_t,uint64_t,uint64_t,float*);
void q38fn_tp_ngram_reduce(q38fn_tp_model*,float*);
int q38fn_tp_linear_layer(q38fn_tp_model*,int,q38fn_tp_delta_state*,float*);
int q38fn_tp_attention_layer(q38fn_tp_model*,int,q38fn_tp_attention_state*,float*);
int q38fn_tp_ple_apply(q38fn_tp_model*,int,q38fn_tp_ple_state*,uint64_t,const float*,float*);
int q38fn_tp_final(q38fn_tp_model*,const float*,float*);
int q38fn_tp_head(q38fn_tp_model*,const float*,int*,float*);
void q38fn_tp_profile_report(FILE*);

#ifdef Q38FN_TP_RUNTIME_IMPLEMENTATION
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__ARM_FEATURE_SVE)
#include "ggml_dequant.h"
#include "../a64fx/k3/k3_dense.h"
#endif

enum { QTP_HC=Q38FN_HC_COUNT*Q38FN_HIDDEN,QTP_PACK=2*Q38FN_EXPERT_INTERMEDIATE,
       QTP_LAYER_MIX=320 };
typedef struct { double hc,hc_norm,hc_down,hc_up,hc_inject,moe,moe_route,moe_route_quant,moe_route_mv,moe_route_select,moe_up,moe_down,moe_reduce,delta,delta_proj,delta_conv,delta_recurrent,delta_out,attention,final_mix,head;uint64_t hc_n,moe_n,delta_n,attention_n,final_n,head_n; } qtp_profile_state;
static qtp_profile_state qtp_profile;
static int qtp_profile_enabled(void){static int init=0,on=0;if(!init){on=getenv("Q38FN_TP_COMPONENT_PROFILE")!=NULL;init=1;}return on;}
#ifdef Q38FN_HAVE_FAPP
static int qtp_fapp_components(void){static int init=0,on=0;if(!init){on=getenv("Q38FN_TP_FAPP_COMPONENTS")!=NULL;init=1;}return on&&fapp_start&&fapp_stop;}
static int qtp_fapp_detail(void){static int init=0,on=0;if(!init){on=getenv("Q38FN_TP_FAPP_DETAIL")!=NULL;init=1;}return on;}
#define QTP_FAPP_START(name) do{if(qtp_fapp_components())fapp_start((name),0,1);}while(0)
#define QTP_FAPP_STOP(name) do{if(qtp_fapp_components())fapp_stop((name),0,1);}while(0)
#else
static int qtp_fapp_detail(void){return 0;}
#define QTP_FAPP_START(name) ((void)0)
#define QTP_FAPP_STOP(name) ((void)0)
#endif
static double qtp_profile_now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static long long qtp_omp_threshold(void){static long long n=-1;if(n<0){const char*s=getenv("Q38FN_TP_OMP_THRESHOLD");n=s?atoll(s):0;if(n<0)n=0;}return n;}
static int qtp_delta_threads(void){static int n=0;if(!n){const char*s=getenv("Q38FN_TP_DELTA_THREADS");n=s?atoi(s):12;if(n<1)n=1;if(n>48)n=48;}return n;}
void q38fn_tp_profile_report(FILE*f){if(!f||!qtp_profile_enabled())return;fprintf(f,
 "components hc=%.9f hc_calls=%llu hc_norm=%.9f hc_down=%.9f hc_up=%.9f hc_inject=%.9f delta=%.9f delta_calls=%llu delta_proj=%.9f delta_conv=%.9f delta_recurrent=%.9f delta_out=%.9f attention=%.9f attention_calls=%llu moe=%.9f moe_calls=%llu moe_route=%.9f moe_route_quant=%.9f moe_route_mv=%.9f moe_route_select=%.9f moe_up=%.9f moe_down=%.9f moe_reduce=%.9f final=%.9f final_calls=%llu head=%.9f head_calls=%llu\n",
 qtp_profile.hc,(unsigned long long)qtp_profile.hc_n,qtp_profile.hc_norm,qtp_profile.hc_down,qtp_profile.hc_up,qtp_profile.hc_inject,qtp_profile.delta,(unsigned long long)qtp_profile.delta_n,
 qtp_profile.delta_proj,qtp_profile.delta_conv,qtp_profile.delta_recurrent,qtp_profile.delta_out,
 qtp_profile.attention,(unsigned long long)qtp_profile.attention_n,qtp_profile.moe,(unsigned long long)qtp_profile.moe_n,
 qtp_profile.moe_route,qtp_profile.moe_route_quant,qtp_profile.moe_route_mv,qtp_profile.moe_route_select,qtp_profile.moe_up,qtp_profile.moe_down,qtp_profile.moe_reduce,
 qtp_profile.final_mix,(unsigned long long)qtp_profile.final_n,qtp_profile.head,(unsigned long long)qtp_profile.head_n);fflush(f);}
static float qtp_bf(uint16_t b){uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f;}
#if defined(__ARM_FEATURE_SVE)
static float qtp_dot_sve8(const uint16_t*w,const float*x,int n){int i=0,vl=(int)svcntw(),vlh=(int)svcnth(),stride=4*vlh;svbool_t pg=svptrue_b32(),pgh=svptrue_b16();svuint16_t zero=svdup_u16(0);svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
 for(;i+stride-1<n;i+=stride){svfloat32_t v0,v1,v2,v3,v4,v5,v6,v7;SVE_BF16_ZIP(svld1_u16(pgh,w+i),zero,v0,v1);SVE_BF16_ZIP(svld1_u16(pgh,w+i+vlh),zero,v2,v3);SVE_BF16_ZIP(svld1_u16(pgh,w+i+2*vlh),zero,v4,v5);SVE_BF16_ZIP(svld1_u16(pgh,w+i+3*vlh),zero,v6,v7);a0=svmla_f32_x(pg,a0,v0,svld1_f32(pg,x+i));a1=svmla_f32_x(pg,a1,v1,svld1_f32(pg,x+i+vl));a2=svmla_f32_x(pg,a2,v2,svld1_f32(pg,x+i+2*vl));a3=svmla_f32_x(pg,a3,v3,svld1_f32(pg,x+i+3*vl));a4=svmla_f32_x(pg,a4,v4,svld1_f32(pg,x+i+4*vl));a5=svmla_f32_x(pg,a5,v5,svld1_f32(pg,x+i+5*vl));a6=svmla_f32_x(pg,a6,v6,svld1_f32(pg,x+i+6*vl));a7=svmla_f32_x(pg,a7,v7,svld1_f32(pg,x+i+7*vl));}
 for(;i<n;i+=vl){svbool_t pt=svwhilelt_b32(i,n);svfloat32_t v=svreinterpret_f32_u32(svlsl_n_u32_x(pt,svld1uh_u32(pt,w+i),16));a0=svmla_f32_m(pt,a0,v,svld1_f32(pt,x+i));}svfloat32_t z=svadd_f32_x(pg,svadd_f32_x(pg,a0,a1),svadd_f32_x(pg,a2,a3));z=svadd_f32_x(pg,z,svadd_f32_x(pg,svadd_f32_x(pg,a4,a5),svadd_f32_x(pg,a6,a7)));return svaddv_f32(pg,z);}
#endif
static float qtp_dot(const uint16_t*w,const float*x,int n){float s=0;
#if defined(__ARM_FEATURE_SVE)
 static int fast_init=0,fast=0;if(!fast_init){fast=getenv("Q38FN_TP_FAST_BF16_DOT8")?2:getenv("Q38FN_TP_NO_FAST_BF16_DOT")?0:1;fast_init=1;}if(fast==2)return qtp_dot_sve8(w,x,n);if(fast==1)return vec_dot_bf16_f32(w,x,n);
#endif
#ifdef _OPENMP
#pragma omp simd reduction(+:s)
#endif
for(int i=0;i<n;i++)s+=qtp_bf(w[i])*x[i];
return s;}
static float qtp_q8_dot(const int8_t*w,float scale,const float*x,int n,float xs){float s=0;
#ifdef _OPENMP
#pragma omp simd reduction(+:s)
#endif
for(int i=0;i<n;i++)s+=(float)w[i]*x[i];
return s*scale*xs;
}
static float qtp_q8_quantize(int8_t*q,const float*x,int n){
#if defined(__ARM_FEATURE_SVE)
 static int sve_init=0,sve_on=0;if(!sve_init){sve_on=getenv("Q38FN_TP_NO_FAST_Q8_QUANT")==NULL;sve_init=1;}if(sve_on){int vl=(int)svcntw();svfloat32_t ma=svdup_f32(0);for(int i=0;i<n;i+=vl){svbool_t pg=svwhilelt_b32(i,n);ma=svmax_f32_x(pg,ma,svabs_f32_x(pg,svld1_f32(pg,x+i)));}float maximum=svmaxv_f32(svptrue_b32(),ma),scale=maximum>0?maximum/127:1,inv=1/scale;for(int i=0;i<n;i+=vl){svbool_t pg=svwhilelt_b32(i,n);svfloat32_t v=svmul_n_f32_x(pg,svld1_f32(pg,x+i),inv);v=svrintn_f32_x(pg,v);svint32_t z=svcvt_s32_f32_x(pg,v);z=svmax_n_s32_x(pg,svmin_n_s32_x(pg,z,127),-127);svst1b_s32(pg,q+i,z);}return scale;}
#endif
    float maximum=0;for(int i=0;i<n;i++)maximum=fmaxf(maximum,fabsf(x[i]));
    float scale=maximum>0?maximum/127:1,inv=1/scale;
    for(int i=0;i<n;i++){long v=lrintf(x[i]*inv);q[i]=(int8_t)(v<-127?-127:v>127?127:v);}return scale;
}
static int qtp_q8_mv_quantized(const q38fn_tp_blob_entry*e,const float*x,const int8_t*qx,float xs,int rows,int cols,float*y){
    if(!e||!e->q8_data||!e->q8_scales||rows<1||cols<1)return-1;
#if defined(__ARM_FEATURE_SVE)
    if(!getenv("Q38FN_TP_Q8_SCALAR")){
#ifdef _OPENMP
        static int q8_max_threads=0;if(!q8_max_threads){const char*v=getenv("Q38FN_TP_Q8_THREADS");q8_max_threads=v?atoi(v):48;if(q8_max_threads<1)q8_max_threads=48;}
        int q8_threads=(rows+7)/8;if(q8_threads>q8_max_threads)q8_threads=q8_max_threads;
#pragma omp parallel for schedule(static) num_threads(q8_threads)
#endif
        for(int r=0;r<rows;r+=8){int32_t dot[8];k3_q8_dot8(dot,e->q8_data+(size_t)r*cols,qx,cols);int count=rows-r<8?rows-r:8;for(int i=0;i<count;i++)y[r+i]=(float)dot[i]*e->q8_scales[r+i]*xs;}
        return 0;
    }
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int r=0;r<rows;r++)y[r]=qtp_q8_dot(e->q8_data+(size_t)r*cols,e->q8_scales[r],x,cols,xs);
    return 0;
}
static int qtp_q8_mv(const q38fn_tp_blob_entry*e,const float*x,int rows,int cols,float*y){
    int8_t qx[cols];float xs=qtp_q8_quantize(qx,x,cols);
    if(getenv("Q38FN_TP_MOE_DEBUG"))for(int r=0;r<rows;r++)if(!e||!e->q8_scales||!isfinite(e->q8_scales[r])){fprintf(stderr,"q38fn q8: invalid scale row=%d\\n",r);break;}
    return qtp_q8_mv_quantized(e,x,qx,xs,rows,cols,y);
}
static int qtp_q8_mv_blocks(const q38fn_tp_blob_entry*e,const size_t*ids,size_t blocks,
                            const int8_t*qx,float xs,int rows,int cols,float*y){
 if(!e||!e->q8_data||!e->q8_block_bytes||!ids||!qx||!blocks||rows<1||cols<1)return-1;
#if defined(__ARM_FEATURE_SVE)
 if(!getenv("Q38FN_TP_Q8_SCALAR")){
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for(size_t b=0;b<blocks;b++)for(int r=0;r<rows;r+=8){
   const int8_t*base=e->q8_data+ids[b]*e->q8_block_bytes;
   int32_t dot[8];k3_q8_dot8(dot,base+(size_t)r*cols,qx,cols);
   int count=rows-r<8?rows-r:8;
   const float*sc=(const float*)(base+(size_t)rows*cols);
   for(int i=0;i<count;i++)y[b*(size_t)rows+(size_t)r+i]=(float)dot[i]*sc[r+i]*xs;
  }
  return 0;
 }
#endif
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
 for(size_t b=0;b<blocks;b++)for(int r=0;r<rows;r++){
  const int8_t*base=e->q8_data+ids[b]*e->q8_block_bytes;
  const float*sc=(const float*)(base+(size_t)rows*cols);
  float sum=0;for(int i=0;i<cols;i++)sum+=(float)base[(size_t)r*cols+i]*(float)qx[i];
  y[b*(size_t)rows+(size_t)r]=sum*sc[r]*xs;
 }
 return 0;
}
static void qtp_mv_rows(const uint16_t*w,const float*x,int rows,int cols,float*y){int groups=rows/8;
#if defined(__ARM_FEATURE_SVE)
 if(rows==4){
  svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0;
  for(int c=0;c<cols;c+=svcntw()){
   svbool_t pg=svwhilelt_b32((uint64_t)c,(uint64_t)cols);
   svfloat32_t xv=svld1_f32(pg,x+c);
   svfloat32_t w0=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,w+c),16));
   svfloat32_t w1=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,w+(size_t)cols+c),16));
   svfloat32_t w2=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,w+2*(size_t)cols+c),16));
   svfloat32_t w3=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,w+3*(size_t)cols+c),16));
   a0=svmla_f32_x(pg,a0,w0,xv);a1=svmla_f32_x(pg,a1,w1,xv);
   a2=svmla_f32_x(pg,a2,w2,xv);a3=svmla_f32_x(pg,a3,w3,xv);
  }
  svbool_t pg=svptrue_b32();y[0]=svaddv_f32(pg,a0);y[1]=svaddv_f32(pg,a1);
  y[2]=svaddv_f32(pg,a2);y[3]=svaddv_f32(pg,a3);return;
 }
 static int exact_init=0,exact=0;if(!exact_init){const char*e=getenv("Q38FN_TP_EXACT_MV");exact=e&&*e&&*e!='0';exact_init=1;}
 /* Tiny row counts leave most threads idle in the eight-row kernel (TP12 HC
  * down has 27 rows and block injection has four).  Distribute individual
  * rows in that case; keep the packed eight-row path for larger matrices. */
 if(exact||rows<=32){
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((long long)rows*cols>=qtp_omp_threshold())
#endif
  for(int r=0;r<rows;r++)y[r]=qtp_dot(w+(size_t)r*cols,x,cols);return;}
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
 for(int g=0;g<groups;g++){const uint16_t*p=w+(size_t)g*8*cols;matvec_bf16_8row(y+g*8,p,p+cols,p+2*cols,p+3*cols,p+4*cols,p+5*cols,p+6*cols,p+7*cols,x,cols);}
#else
 (void)groups;
#endif
#if defined(__ARM_FEATURE_SVE)
 for(int r=groups*8;r<rows;r++)y[r]=qtp_dot(w+(size_t)r*cols,x,cols);}
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
 for(int r=0;r<rows;r++)y[r]=qtp_dot(w+(size_t)r*cols,x,cols);}
#endif
static float qtp_sig(float x){return 1.0f/(1.0f+expf(-x));}
static float qtp_silu(float x){return x*qtp_sig(x);}
#if defined(__ARM_FEATURE_SVE)
static inline svfloat32_t qtp_exp2_fexpa(svbool_t pg,svfloat32_t x){
 const svfloat32_t shift=svdup_f32(204927.0f);svfloat32_t z=svadd_f32_x(pg,x,shift),n=svsub_f32_x(pg,z,shift),r=svsub_f32_x(pg,x,n);svfloat32_t scale=svexpa_f32(svreinterpret_u32_f32(z));return svmul_f32_x(pg,scale,svmla_n_f32_x(pg,svdup_f32(1),r,0.6931471805599453f));}
static inline svfloat32_t qtp_sig_sve(svbool_t pg,svfloat32_t x){
 svfloat32_t t=svmax_n_f32_x(pg,svmin_n_f32_x(pg,svmul_n_f32_x(pg,x,-1.4426950408889634f),80),-80);svfloat32_t den=svadd_n_f32_x(pg,qtp_exp2_fexpa(pg,t),1);svfloat32_t inv=svrecpe_f32(den);return svmul_f32_x(pg,inv,svrecps_f32(den,inv));}
#endif
static float qtp_softplus(float x){return x>20.0f?x:log1pf(expf(x));}
static int qtp_finite_n(const float*x,size_t n){for(size_t i=0;i<n;i++)if(!isfinite(x[i]))return 0;return 1;}
static const q38fn_tp_blob_entry*qtp_find(q38fn_tp_model*m,const char*n){const q38fn_tp_blob_entry*e=m->has_q8_blob?q38fn_tp_blob_find(&m->q8_blob,n):NULL;if(!e&&m->has_aux_blob)e=q38fn_tp_blob_find(&m->aux_blob,n);if(!e&&m->has_hc_blob)e=q38fn_tp_blob_find(&m->hc_blob,n);return e?e:q38fn_tp_blob_find(&m->blob,n);}
static int qtp_entry_mv(const q38fn_tp_blob_entry*e,const float*x,int rows,int cols,float*y){if(!e)return-1;if(e->q8_data)return qtp_q8_mv(e,x,rows,cols,y);if(e->q5_data)return q38fn_q5_matvec(y,e->q5_data,x,(size_t)rows,(size_t)cols);qtp_mv_rows(e->data,x,rows,cols,y);return 0;}
static int qtp_entry_mv_q8_block(const q38fn_tp_blob_entry*e,size_t block,const float*x,int rows,int cols,float*y){if(!e||!e->q8_data||!e->q8_block_bytes)return-1;q38fn_tp_blob_entry part=*e;part.q8_data=e->q8_data+block*e->q8_block_bytes;part.q8_scales=(const float*)((const char*)part.q8_data+(size_t)rows*(size_t)cols);part.q8_block_rows=(size_t)rows;part.q8_block_bytes=e->q8_block_bytes;return qtp_q8_mv(&part,x,rows,cols,y);}
static int qtp_entry_mv_offset(const q38fn_tp_blob_entry*e,size_t row,const float*x,int rows,int cols,float*y){if(!e)return-1;if(e->q8_data){q38fn_tp_blob_entry part=*e;part.q8_data=e->q8_data+row*(size_t)cols;part.q8_scales=e->q8_scales+row;return qtp_q8_mv(&part,x,rows,cols,y);}if(e->q5_data)return q38fn_q5_matvec(y,e->q5_data+row*(size_t)(cols/32),x,(size_t)rows,(size_t)cols);qtp_mv_rows(e->data+row*(size_t)cols,x,rows,cols,y);return 0;}
static int qtp_bf16_mv_blocks(const q38fn_tp_blob_entry*e,const size_t*ids,
                              size_t blocks,const float*x,int rows,int cols,
                              float*y){
 if(!e||!e->data||!ids||!blocks||!x||!y||rows<1||cols<1)return-1;
#if defined(__ARM_FEATURE_SVE)
 int groups=rows/8;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
 for(size_t b=0;b<blocks;b++)for(int g=0;g<groups;g++){
  const uint16_t*w=e->data+(ids[b]*(size_t)rows+(size_t)g*8)*(size_t)cols;
  const float*v=x+b*(size_t)cols;
  matvec_bf16_8row(y+b*(size_t)rows+(size_t)g*8,
      w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,
      w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,
      w+7*(size_t)cols,v,cols);
 }
 for(size_t b=0;b<blocks;b++)for(int r=groups*8;r<rows;r++)
  y[b*(size_t)rows+(size_t)r]=qtp_dot(
      e->data+(ids[b]*(size_t)rows+(size_t)r)*(size_t)cols,
      x+b*(size_t)cols,cols);
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
 for(size_t b=0;b<blocks;b++)for(int r=0;r<rows;r++)
  y[b*(size_t)rows+(size_t)r]=qtp_dot(
      e->data+(ids[b]*(size_t)rows+(size_t)r)*(size_t)cols,
      x+b*(size_t)cols,cols);
#endif
 return 0;
}
static const int8_t *qtp_q8_row(const q38fn_tp_blob_entry*e,size_t row,int cols,const float**scale){size_t block=e->q8_block_rows?row/e->q8_block_rows:0,local=e->q8_block_rows?row%e->q8_block_rows:row;const char*base=(const char*)e->q8_data+block*e->q8_block_bytes;size_t weights=(size_t)e->q8_block_rows*(size_t)cols;*scale=(const float*)(base+weights)+local;return (const int8_t*)base+local*(size_t)cols;}
static float qtp_entry_row_dot(const q38fn_tp_blob_entry*e,size_t row,const float*x,int cols){if(e&&e->q8_data){const float*scale;const int8_t*w=qtp_q8_row(e,row,cols,&scale);return qtp_q8_dot(w,*scale,x,cols,1.0f);}if(e&&e->q5_data){float value=0;q38fn_q5_matvec(&value,e->q5_data+row*(size_t)(cols/32),x,1,(size_t)cols);return value;}return qtp_dot(e->data+row*(size_t)cols,x,cols);}
static float qtp_entry_row_dot_q8(const q38fn_tp_blob_entry*e,size_t row,const int8_t*x,float xs,int cols){const float*scale;const int8_t*w=qtp_q8_row(e,row,cols,&scale);int32_t sum=0;
#if defined(__ARM_FEATURE_SVE)
 sum=k3_q8_dot(w,x,cols);
#else
#ifdef _OPENMP
#pragma omp simd reduction(+:sum)
#endif
 for(int i=0;i<cols;i++)sum+=(int32_t)w[i]*(int32_t)x[i];
#endif
 return(float)sum*(*scale)*xs;}
static int qtp_full_mv(const q38fn_tp_blob_entry*e,const float*x,int rows,int cols,float*y){
 if(!e||e->kind!=Q38FN_TP_FULL)return -1;
 return qtp_entry_mv(e,x,rows,cols,y);}
static int qtp_q5_mv_threads(const q38fn_tp_blob_entry*e,const float*x,int rows,int cols,float*y,int threads){if(!e||!e->q5_data||threads<1)return-1;size_t blocks=(size_t)cols/32;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
 for(int r=0;r<rows;r++)if(q38fn_q5_matvec(y+r,e->q5_data+(size_t)r*blocks,x,1,(size_t)cols))y[r]=NAN;return qtp_finite_n(y,(size_t)rows)?0:-1;}
static int qtp_q5_mv_pair(const q38fn_tp_blob_entry*a,const q38fn_tp_blob_entry*b,const float*x,int rows,int cols,float*ya,float*yb){if(!a||!b||!a->q5_data||!b->q5_data)return-1;size_t blocks=(size_t)cols/32;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
 for(int r=0;r<2*rows;r++){const q38fn_tp_blob_entry*e=r<rows?a:b;float*y=r<rows?ya:yb;int rr=r%rows;if(q38fn_q5_matvec(y+rr,e->q5_data+(size_t)rr*blocks,x,1,(size_t)cols))y[rr]=NAN;}return(qtp_finite_n(ya,(size_t)rows)&&qtp_finite_n(yb,(size_t)rows))?0:-1;}
static int qtp_q5_mv_row_pairs(const q38fn_tp_blob_entry*e,const float*x,
                               int rows,int cols,float*y){
 if(!e||!e->q5_data||rows<2||rows&1)return-1;size_t blocks=(size_t)cols/32;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
 for(int pair=0;pair<rows/2;pair++){size_t row=(size_t)pair*2;q38fn_q5_matvec_pair(y+row,e->q5_data+row*blocks,e->q5_data+(row+1)*blocks,x,(size_t)cols);}
 return qtp_finite_n(y,(size_t)rows)?0:-1;}
static int qtp_axis0_mv(const q38fn_tp_blob_entry*e,const float*x,int cols,float*y){
 if(!e||e->kind!=Q38FN_TP_AXIS0||e->n_ranges!=1)return -1;
 int rows=(int)e->range[0].count;
 if(qtp_entry_mv(e,x,rows,cols,y))return-1;
 return rows;}
static int qtp_axis1_mv(q38fn_tp_model*m,const q38fn_tp_blob_entry*e,const float*x,int rows,float*y){
 if(!e||e->kind!=Q38FN_TP_AXIS1||e->n_ranges!=1)return -1;
 int c=(int)e->range[0].count;
 if(qtp_entry_mv(e,x,rows,c,y))return-1;
 m->sum(y,rows,m->comm);return 0;}
static int qtp_name(char*b,size_t n,int L,const char*s){return snprintf(b,n,"model.language_model.layers.%d.%s",L,s)>=(int)n?-1:0;}
static void qtp_debug_dump(q38fn_tp_model*m,const char*name,const float*x,size_t n){const char*d=getenv("Q38FN_DUMP_DIR");char p[4096];FILE*f;if(!d||!*d||snprintf(p,sizeof(p),"%s/tp-rank-%02d-%s.f32",d,m->rank,name)>=(int)sizeof(p)||(f=fopen(p,"wb"))==NULL)return;fwrite(x,sizeof(*x),n,f);fclose(f);}

int q38fn_tp_model_open(q38fn_tp_model*m,const char*base,int rank,int ranks,q38fn_tp_sum_fn sum,q38fn_tp_argmax_fn argmax,void*comm){
 char d[4096];if(!m||!base||!sum||!argmax||ranks!=Q38FN_TP_RANKS)return-1;memset(m,0,sizeof(*m));
 if(snprintf(d,sizeof(d),"%s/rank-%02d",base,rank)>=(int)sizeof(d)||q38fn_tp_blob_open(&m->blob,d,getenv("Q38FN_TP_VERIFY")!=NULL))return-1;
 const char*hc=getenv("Q38FN_TP_HC_BASE");if(hc&&*hc){if(snprintf(d,sizeof(d),"%s/rank-%02d",hc,rank)>=(int)sizeof(d)||q38fn_tp_blob_open(&m->hc_blob,d,getenv("Q38FN_TP_VERIFY")!=NULL)){q38fn_tp_blob_close(&m->blob);return-1;}m->has_hc_blob=1;}
 const char*q8=getenv("Q38FN_TP_Q8_BASE");if(q8&&*q8){if(snprintf(d,sizeof(d),"%s/rank-%02d",q8,rank)>=(int)sizeof(d)||q38fn_tp_blob_open(&m->q8_blob,d,getenv("Q38FN_TP_VERIFY")!=NULL)){if(m->has_hc_blob)q38fn_tp_blob_close(&m->hc_blob);q38fn_tp_blob_close(&m->blob);return-1;}m->has_q8_blob=1;}
 const char*aux=getenv("Q38FN_TP_AUX_BASE");if(aux&&*aux){if(snprintf(d,sizeof(d),"%s/rank-%02d",aux,rank)>=(int)sizeof(d)||q38fn_tp_blob_open(&m->aux_blob,d,getenv("Q38FN_TP_VERIFY")!=NULL)){if(m->has_q8_blob)q38fn_tp_blob_close(&m->q8_blob);if(m->has_hc_blob)q38fn_tp_blob_close(&m->hc_blob);q38fn_tp_blob_close(&m->blob);return-1;}m->has_aux_blob=1;}
 m->rank=rank;m->ranks=ranks;m->sum=sum;m->argmax=argmax;m->comm=comm;
 const q38fn_tp_blob_entry*head=q38fn_tp_blob_find(&m->blob,"lm_head.weight");if(head&&head->n_ranges>0){m->head_logits_count=(size_t)head->range[0].count;m->head_logits=malloc(m->head_logits_count*sizeof(*m->head_logits));if(!m->head_logits){q38fn_tp_blob_close(&m->blob);return-1;}}return 0;}
void q38fn_tp_model_close(q38fn_tp_model*m){if(m){free(m->head_logits);m->head_logits=NULL;m->head_logits_count=0;if(m->has_aux_blob)q38fn_tp_blob_close(&m->aux_blob);if(m->has_q8_blob)q38fn_tp_blob_close(&m->q8_blob);if(m->has_hc_blob)q38fn_tp_blob_close(&m->hc_blob);q38fn_tp_blob_close(&m->blob);}}
int q38fn_tp_delta_init(q38fn_tp_delta_state*s){if(!s)return-1;s->conv=calloc((size_t)Q38FN_LINEAR_CONV_DIM*4,4);s->recurrent=calloc((size_t)Q38FN_LINEAR_VALUE_HEADS*128*128,4);return s->conv&&s->recurrent?0:-1;}
void q38fn_tp_delta_close(q38fn_tp_delta_state*s){if(s){free(s->conv);free(s->recurrent);memset(s,0,sizeof(*s));}}
int q38fn_tp_attention_init(q38fn_tp_attention_state*s,size_t cap){size_t w=Q38FN_KV_HEADS*Q38FN_HEAD_DIM;if(!s)return-1;memset(s,0,sizeof(*s));s->keys=calloc(cap*w,4);s->values=calloc(cap*w,4);s->scores=malloc(cap*4);s->capacity=cap;return s->keys&&s->values&&s->scores?0:-1;}
void q38fn_tp_attention_close(q38fn_tp_attention_state*s){if(s){free(s->keys);free(s->values);free(s->scores);memset(s,0,sizeof(*s));}}
int q38fn_tp_ple_init(q38fn_tp_ple_state*s){if(!s)return-1;memset(s,0,sizeof(*s));s->conv=calloc((size_t)QTP_HC*9,4);s->previous=s->previous2=Q38FN_EOS;return s->conv?0:-1;}
void q38fn_tp_ple_close(q38fn_tp_ple_state*s){if(s){free(s->conv);memset(s,0,sizeof(*s));}}

int q38fn_tp_embedding(q38fn_tp_model*m,int token,float*out){const q38fn_tp_blob_entry*e=qtp_find(m,"model.language_model.embed_tokens.weight");if(!e||e->kind!=Q38FN_TP_AXIS0)return-1;memset(out,0,Q38FN_HIDDEN*4);uint64_t s=e->range[0].start,n=e->range[0].count;if(token>=(int)s&&token<(int)(s+n)){size_t row=(size_t)(token-(int)s);if(e->q5_data){if(q38fn_q5_dequantize_row(out,e->q5_data+row*(Q38FN_HIDDEN/32),Q38FN_HIDDEN))return-1;}else{const uint16_t*p=e->data+row*Q38FN_HIDDEN;for(int i=0;i<Q38FN_HIDDEN;i++)out[i]=qtp_bf(p[i]);}}m->sum(out,Q38FN_HIDDEN,m->comm);return 0;}
int q38fn_tp_ngram_local(q38fn_tp_model*m,uint64_t token,uint64_t prev,uint64_t prev2,float*out){uint64_t rows[Q38FN_NGRAM_HEADS];q38fn_ngram_rows(token,prev,prev2,rows);memset(out,0,Q38FN_HIDDEN*4);for(int h=0;h<Q38FN_NGRAM_HEADS;h++){uint64_t sh=rows[h]/Q38FN_NGRAM_ROWS_PER_SHARD,row=rows[h]%Q38FN_NGRAM_ROWS_PER_SHARD;if(q38fn_ngram_owner_for_ranks((int)sh,m->ranks)==m->rank){char n[224];snprintf(n,sizeof(n),"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%llu.weight",(unsigned long long)sh);const q38fn_tp_blob_entry*e=qtp_find(m,n);if(!e)return-1;if(e->q5_data){const q38fn_q5_block*p=e->q5_data+row*(Q38FN_NGRAM_HEAD_DIM/32);if(q38fn_q5_dequantize_row(out+h*Q38FN_NGRAM_HEAD_DIM,p,Q38FN_NGRAM_HEAD_DIM))return-1;}else{const uint16_t*p=e->data+row*Q38FN_NGRAM_HEAD_DIM;for(int i=0;i<Q38FN_NGRAM_HEAD_DIM;i++)out[h*Q38FN_NGRAM_HEAD_DIM+i]=qtp_bf(p[i]);}}}return 0;}
void q38fn_tp_ngram_reduce(q38fn_tp_model*m,float*out){m->sum(out,Q38FN_HIDDEN,m->comm);}
int q38fn_tp_ngram(q38fn_tp_model*m,uint64_t token,uint64_t prev,uint64_t prev2,float*out){if(q38fn_tp_ngram_local(m,token,prev,prev2,out))return-1;q38fn_tp_ngram_reduce(m,out);return 0;}

static int qtp_hc(q38fn_tp_model*m,int L,const char*block,const float*h,float*x,float inj[4]){char n[224],p[180];float norm[QTP_HC],lo[QTP_LAYER_MIX],partial[QTP_HC];
 static int rep_init=0,replicated=0;if(!rep_init){const char*e=getenv("Q38FN_TP_REPLICATED_HC");replicated=e&&*e&&*e!='0';rep_init=1;}uint64_t hc_start=0;int prof=qtp_profile_enabled();double pt=prof?qtp_profile_now():0;
 snprintf(p,sizeof(p),"model.language_model.layers.%d.%s_hyper_connection",L,block);snprintf(n,sizeof(n),"%s.hc_norm.weight",p);const q38fn_tp_blob_entry*wn=qtp_find(m,n);if(!wn)return-1;
 int fused_team_norm=getenv("Q38FN_TP_FUSED_TEAM_HC_NORM")!=NULL;
 if(!fused_team_norm){
#if defined(__ARM_FEATURE_SVE)
 if(getenv("Q38FN_TP_TEAM_HC_NORM")){
  float sums[48][4],zs[4];int vl=(int)svcntw();
#ifdef _OPENMP
#pragma omp parallel num_threads(48) shared(sums,zs)
  {
   int tid=omp_get_thread_num(),nth=omp_get_num_threads();svbool_t pg=svptrue_b32();svfloat32_t q0=svdup_f32(0),q1=svdup_f32(0),q2=svdup_f32(0),q3=svdup_f32(0);
   for(int i=tid*vl;i<Q38FN_HIDDEN;i+=nth*vl){svfloat32_t v0=svld1_f32(pg,h+i),v1=svld1_f32(pg,h+Q38FN_HIDDEN+i),v2=svld1_f32(pg,h+2*Q38FN_HIDDEN+i),v3=svld1_f32(pg,h+3*Q38FN_HIDDEN+i);q0=svmla_f32_x(pg,q0,v0,v0);q1=svmla_f32_x(pg,q1,v1,v1);q2=svmla_f32_x(pg,q2,v2,v2);q3=svmla_f32_x(pg,q3,v3,v3);}
   sums[tid][0]=svaddv_f32(pg,q0);sums[tid][1]=svaddv_f32(pg,q1);sums[tid][2]=svaddv_f32(pg,q2);sums[tid][3]=svaddv_f32(pg,q3);
#pragma omp barrier
#pragma omp master
   {for(int s=0;s<4;s++){float q=0;for(int t=0;t<nth;t++)q+=sums[t][s];zs[s]=1.0f/sqrtf(q/Q38FN_HIDDEN+1e-6f);}}
#pragma omp barrier
   for(int j=tid*vl;j<QTP_HC;j+=nth*vl){int s=j/Q38FN_HIDDEN;svfloat32_t v=svld1_f32(pg,h+j),w=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,wn->data+j),16));svst1_f32(pg,norm+j,svmul_n_f32_x(pg,svmul_f32_x(pg,v,svadd_n_f32_x(pg,w,1.0f)),zs[s]));}
  }
#else
  (void)sums;(void)zs;(void)vl;
#endif
 }else if(getenv("Q38FN_TP_PARALLEL_HC_NORM")){
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4)
#endif
  for(int s=0;s<4;s++){svbool_t pg=svptrue_b32();svfloat32_t q=svdup_f32(0);int vl=(int)svcntw(),base=s*Q38FN_HIDDEN;for(int i=0;i<Q38FN_HIDDEN;i+=vl){svfloat32_t v=svld1_f32(pg,h+base+i);q=svmla_f32_x(pg,q,v,v);}float z=1.0f/sqrtf(svaddv_f32(pg,q)/Q38FN_HIDDEN+1e-6f);for(int i=0;i<Q38FN_HIDDEN;i+=vl){int j=base+i;svfloat32_t v=svld1_f32(pg,h+j),w=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,wn->data+j),16));svst1_f32(pg,norm+j,svmul_n_f32_x(pg,svmul_f32_x(pg,v,svadd_n_f32_x(pg,w,1.0f)),z));}}
 }else if(!getenv("Q38FN_TP_NO_FAST_HC_NORM")){
  svbool_t pg=svptrue_b32();svfloat32_t q0=svdup_f32(0),q1=svdup_f32(0),q2=svdup_f32(0),q3=svdup_f32(0);int vl=(int)svcntw();
  for(int i=0;i<Q38FN_HIDDEN;i+=vl){svfloat32_t v0=svld1_f32(pg,h+i),v1=svld1_f32(pg,h+Q38FN_HIDDEN+i),v2=svld1_f32(pg,h+2*Q38FN_HIDDEN+i),v3=svld1_f32(pg,h+3*Q38FN_HIDDEN+i);q0=svmla_f32_x(pg,q0,v0,v0);q1=svmla_f32_x(pg,q1,v1,v1);q2=svmla_f32_x(pg,q2,v2,v2);q3=svmla_f32_x(pg,q3,v3,v3);}
  float z0=1.0f/sqrtf(svaddv_f32(pg,q0)/Q38FN_HIDDEN+1e-6f),z1=1.0f/sqrtf(svaddv_f32(pg,q1)/Q38FN_HIDDEN+1e-6f),z2=1.0f/sqrtf(svaddv_f32(pg,q2)/Q38FN_HIDDEN+1e-6f),z3=1.0f/sqrtf(svaddv_f32(pg,q3)/Q38FN_HIDDEN+1e-6f);
  for(int i=0;i<Q38FN_HIDDEN;i+=vl){for(int s=0;s<4;s++){int j=s*Q38FN_HIDDEN+i;svfloat32_t v=svld1_f32(pg,h+j),w=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,wn->data+j),16));float z=s==0?z0:s==1?z1:s==2?z2:z3;svst1_f32(pg,norm+j,svmul_n_f32_x(pg,svmul_f32_x(pg,v,svadd_n_f32_x(pg,w,1.0f)),z));}}
 }else
#endif
 for(int s=0;s<4;s++){double q=0;for(int i=0;i<Q38FN_HIDDEN;i++){float v=h[s*Q38FN_HIDDEN+i];q+=(double)v*v;}float z=1.0f/sqrtf((float)(q/Q38FN_HIDDEN)+1e-6f);for(int i=0;i<Q38FN_HIDDEN;i++){int j=s*Q38FN_HIDDEN+i;norm[j]=h[j]*z*(1.0f+qtp_bf(wn->data[j]));}}
 }
 if(prof&&!fused_team_norm){qtp_profile.hc_norm+=qtp_profile_now()-pt;pt=qtp_profile_now();}
 snprintf(n,sizeof(n),"%s.input_mix_weight_down.weight",p);const q38fn_tp_blob_entry*wd=qtp_find(m,n);int nr;
 snprintf(n,sizeof(n),"%s.input_mix_weight_up.weight",p);const q38fn_tp_blob_entry*fast_wu=qtp_find(m,n);
 /* The replicated TP12 sidecar stores the 320-row down projection as Q5 and
  * the 8192-row up projection as BF16.  Run down, up, stream mixing and the
  * four injection rows in one persistent OpenMP team.  The former generic
  * path opened four teams per HC block (192 teams/token), which FAPP showed
  * as low IPC and mostly no-commit/load-wait cycles. */
 snprintf(n,sizeof(n),"%s.block_inject_weight.weight",p);const q38fn_tp_blob_entry*fast_wi=qtp_find(m,n);
 if(wd&&fast_wu&&fast_wi&&replicated&&wd->kind==Q38FN_TP_FULL&&
    fast_wu->kind==Q38FN_TP_FULL&&(wd->q5_data||(!wd->q8_data&&wd->data))&&fast_wu->q8_data&&
    fast_wi->data&&wd->shape[0]==QTP_LAYER_MIX&&
    getenv("Q38FN_TP_REPLICATED_Q8_HC")&&
    (wd->q5_data||getenv("Q38FN_TP_PERSISTENT_BF16_Q8_HC"))){
  nr=QTP_LAYER_MIX;int8_t qlo[QTP_LAYER_MIX];float ls=1;double td=pt,tu=pt,ti=pt,norm_done=pt;float norm_sums[48][4],norm_zs[4];
  int pair_down=wd->q5_data&&getenv("Q38FN_TP_HC_Q5_PAIR")!=NULL;
  int down_tasks=pair_down?nr/2:nr;
#ifdef _OPENMP
#pragma omp parallel shared(td,tu,ti,ls,norm_done,norm_sums,norm_zs)
  {
#if defined(__ARM_FEATURE_SVE)
   if(fused_team_norm){int tid=omp_get_thread_num(),nth=omp_get_num_threads(),vl=(int)svcntw();svbool_t pg=svptrue_b32();svfloat32_t q0=svdup_f32(0),q1=svdup_f32(0),q2=svdup_f32(0),q3=svdup_f32(0);for(int i=tid*vl;i<Q38FN_HIDDEN;i+=nth*vl){svfloat32_t v0=svld1_f32(pg,h+i),v1=svld1_f32(pg,h+Q38FN_HIDDEN+i),v2=svld1_f32(pg,h+2*Q38FN_HIDDEN+i),v3=svld1_f32(pg,h+3*Q38FN_HIDDEN+i);q0=svmla_f32_x(pg,q0,v0,v0);q1=svmla_f32_x(pg,q1,v1,v1);q2=svmla_f32_x(pg,q2,v2,v2);q3=svmla_f32_x(pg,q3,v3,v3);}norm_sums[tid][0]=svaddv_f32(pg,q0);norm_sums[tid][1]=svaddv_f32(pg,q1);norm_sums[tid][2]=svaddv_f32(pg,q2);norm_sums[tid][3]=svaddv_f32(pg,q3);
#pragma omp barrier
#pragma omp master
    {for(int s=0;s<4;s++){float q=0;for(int t=0;t<nth;t++)q+=norm_sums[t][s];norm_zs[s]=1.0f/sqrtf(q/Q38FN_HIDDEN+1e-6f);}}
#pragma omp barrier
    for(int j=tid*vl;j<QTP_HC;j+=nth*vl){int s=j/Q38FN_HIDDEN;svfloat32_t v=svld1_f32(pg,h+j),w=svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,wn->data+j),16));svst1_f32(pg,norm+j,svmul_n_f32_x(pg,svmul_f32_x(pg,v,svadd_n_f32_x(pg,w,1.0f)),norm_zs[s]));}
#pragma omp barrier
#pragma omp master
    {norm_done=qtp_profile_now();if(getenv("Q38FN_TP_HC_NORM_DEBUG")&&L==0&&!strcmp(block,"attn")){float mx=0;int bad=0;for(int j=0;j<QTP_HC;j++){if(!isfinite(norm[j]))bad++;else if(fabsf(norm[j])>mx)mx=fabsf(norm[j]);}fprintf(stderr,"hc_norm rank=%d z=%g,%g,%g,%g max=%g bad=%d\\n",m->rank,norm_zs[0],norm_zs[1],norm_zs[2],norm_zs[3],mx,bad);}}
#pragma omp barrier
   }
#endif
#pragma omp for schedule(static)
#endif
   for(int task=0;task<down_tasks+4;task++){
    if(task<down_tasks){
     if(pair_down&&wd->q5_data){int r=task*2;q38fn_q5_matvec_pair(lo+r,wd->q5_data+(size_t)r*(QTP_HC/32),wd->q5_data+(size_t)(r+1)*(QTP_HC/32),norm,QTP_HC);lo[r]=qtp_silu(lo[r]/4.0f);lo[r+1]=qtp_silu(lo[r+1]/4.0f);}
     else lo[task]=qtp_silu(qtp_entry_row_dot(wd,(size_t)task,norm,QTP_HC)/4.0f);
    }else{int ir=task-down_tasks;inj[ir]=2*qtp_sig(qtp_entry_row_dot(fast_wi,(size_t)ir,norm,QTP_HC)/4.0f);}
   }
#ifdef _OPENMP
#pragma omp single
#endif
   {td=qtp_profile_now();ls=qtp_q8_quantize(qlo,lo,nr);}
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
   for(int r=0;r<QTP_HC;r++)partial[r]=qtp_entry_row_dot_q8(fast_wu,(size_t)r,qlo,ls,nr);
#ifdef _OPENMP
#pragma omp master
   {tu=qtp_profile_now();}
#pragma omp barrier
#endif
#if defined(__ARM_FEATURE_SVE)
   if(getenv("Q38FN_TP_FAST_HC_MIX")){
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for(int i=0;i<Q38FN_HIDDEN;i+=(int)svcntw()){svbool_t pg=svwhilelt_b32(i,Q38FN_HIDDEN);svfloat32_t z=svdup_f32(0);for(int s=0;s<4;s++){int j=s*Q38FN_HIDDEN+i;z=svmla_f32_x(pg,z,qtp_sig_sve(pg,svld1_f32(pg,partial+j)),svld1_f32(pg,norm+j));}svst1_f32(pg,x+i,svmul_n_f32_x(pg,z,.25f));}
   }else
#endif
   {
#ifdef _OPENMP
    if(getenv("Q38FN_TP_FLOAT_HC_MIX")){
#pragma omp for schedule(static)
#endif
     for(int i=0;i<Q38FN_HIDDEN;i++){float z=0;for(int s=0;s<4;s++){int j=s*Q38FN_HIDDEN+i;z+=qtp_sig(partial[j])*norm[j];}x[i]=z*.25f;}
#ifdef _OPENMP
    }else{
#pragma omp for schedule(static)
#endif
     for(int i=0;i<Q38FN_HIDDEN;i++){double z=0;for(int s=0;s<4;s++){int j=s*Q38FN_HIDDEN+i;z+=(double)qtp_sig(partial[j])*norm[j];}x[i]=(float)(z/4.0f);}
#ifdef _OPENMP
    }
#endif
   }
#ifdef _OPENMP
#pragma omp master
   {ti=qtp_profile_now();}
  }
#endif
  if(prof){if(fused_team_norm)qtp_profile.hc_norm+=norm_done-pt;qtp_profile.hc_down+=td-(fused_team_norm?norm_done:pt);qtp_profile.hc_up+=tu-td;qtp_profile.hc_inject+=ti-tu;}
  return 0;
 }
 if(wd&&fast_wu&&fast_wi&&replicated&&wd->kind==Q38FN_TP_FULL&&
    fast_wu->kind==Q38FN_TP_FULL&&wd->q8_data&&fast_wu->q8_data&&
    fast_wi->data&&wd->shape[0]==QTP_LAYER_MIX&&
    getenv("Q38FN_TP_REPLICATED_Q8_HC")){
  nr=QTP_LAYER_MIX;int8_t qnorm[QTP_HC],qlo[QTP_LAYER_MIX];
  float ns=qtp_q8_quantize(qnorm,norm,QTP_HC),ls=1;double td=pt,tu=pt,ti=pt;
#ifdef _OPENMP
#pragma omp parallel shared(td,tu,ti,ls)
  {
#pragma omp for schedule(static)
#endif
   for(int r=0;r<nr+4;r++){
    if(r<nr)lo[r]=qtp_silu(qtp_entry_row_dot_q8(wd,(size_t)r,qnorm,ns,QTP_HC)/4.0f);
    else{int ir=r-nr;inj[ir]=2*qtp_sig(qtp_entry_row_dot(fast_wi,(size_t)ir,norm,QTP_HC)/4.0f);}
   }
#ifdef _OPENMP
#pragma omp single
#endif
   {td=qtp_profile_now();ls=qtp_q8_quantize(qlo,lo,nr);}
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
   for(int r=0;r<QTP_HC;r++)partial[r]=qtp_entry_row_dot_q8(fast_wu,(size_t)r,qlo,ls,nr);
#ifdef _OPENMP
#pragma omp master
   {tu=qtp_profile_now();}
#pragma omp barrier
#pragma omp for schedule(static)
#endif
   for(int i=0;i<Q38FN_HIDDEN;i++){double z=0;for(int s=0;s<4;s++){int j=s*Q38FN_HIDDEN+i;z+=(double)qtp_sig(partial[j])*norm[j];}x[i]=(float)(z/4.0f);}
#ifdef _OPENMP
#pragma omp master
   {ti=qtp_profile_now();}
  }
#endif
  if(prof){qtp_profile.hc_down+=td-pt;qtp_profile.hc_up+=tu-td;qtp_profile.hc_inject+=ti-tu;}
  return 0;
 }
 if(wd&&fast_wu&&fast_wi&&replicated&&wd->kind==Q38FN_TP_FULL&&
    fast_wu->kind==Q38FN_TP_FULL&&fast_wi->kind==Q38FN_TP_FULL&&
    !wd->q8_data&&wd->data&&fast_wu->data&&!fast_wu->q5_data&&!fast_wu->q8_data&&
    fast_wi->data&&!fast_wi->q8_data&&wd->shape[0]==QTP_LAYER_MIX&&
    !getenv("Q38FN_TP_NO_FUSED_MIXED_HC")){
  nr=QTP_LAYER_MIX;double td=pt,tu=pt,ti=pt;int packed_up=0,up_tasks=QTP_HC;
#if defined(__ARM_FEATURE_SVE)
  packed_up=getenv("Q38FN_TP_HC_UP_8ROW")!=NULL;if(packed_up)up_tasks=QTP_HC/8;
#endif
#ifdef _OPENMP
#pragma omp parallel shared(td,tu,ti)
  {
#endif
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
   for(int r=0;r<nr+4;r++){
    if(r<nr)lo[r]=qtp_silu(qtp_entry_row_dot(wd,(size_t)r,norm,QTP_HC)/4.0f);
    else{int ir=r-nr;inj[ir]=2*qtp_sig(qtp_entry_row_dot(fast_wi,(size_t)ir,norm,QTP_HC)/4.0f);}
   }
#ifdef _OPENMP
#pragma omp master
   {td=qtp_profile_now();}
#pragma omp barrier
#endif
#ifdef _OPENMP
   if(!getenv("Q38FN_TP_NO_HC_CMG_STRIPE")){
    int tid=omp_get_thread_num(),nth=omp_get_num_threads(),per=nth/4;
    int order=(nth==48)?(tid%per)*4+tid/per:tid;
    int task_rows=packed_up?8:1;
    int tasks_per_block=(int)((64u*1024u)/((size_t)task_rows*(size_t)nr*sizeof(uint16_t)));
    if(tasks_per_block<1)tasks_per_block=1;
    /* Match the measured-fast coarse CMG stripe.  Exact byte-block ownership
     * increased remote traffic in the following OpenMP regions on Fujitsu's
     * runtime, while these contiguous 61,440-byte HC8 groups retain affinity. */
    for(int first=order*tasks_per_block;first<up_tasks;first+=nth*tasks_per_block){
     int end=first+tasks_per_block;if(end>up_tasks)end=up_tasks;
     for(int task=first;task<end;task++){
#if defined(__ARM_FEATURE_SVE)
      if(packed_up){int r=task*8;const uint16_t*w=fast_wu->data+(size_t)r*nr;matvec_bf16_8row(partial+r,w,w+nr,w+2*nr,w+3*nr,w+4*nr,w+5*nr,w+6*nr,w+7*nr,lo,nr);}
      else
#endif
      {int r=task;partial[r]=qtp_dot(fast_wu->data+(size_t)r*nr,lo,nr);}
     }
    }
   }else{
#pragma omp for schedule(static) nowait
#endif
   for(int task=0;task<up_tasks;task++){
#if defined(__ARM_FEATURE_SVE)
    if(packed_up){int r=task*8;const uint16_t*w=fast_wu->data+(size_t)r*nr;matvec_bf16_8row(partial+r,w,w+nr,w+2*nr,w+3*nr,w+4*nr,w+5*nr,w+6*nr,w+7*nr,lo,nr);}
    else
#endif
    {int r=task;partial[r]=qtp_dot(fast_wu->data+(size_t)r*nr,lo,nr);}
   }
#ifdef _OPENMP
   }
#pragma omp barrier
#pragma omp master
   {tu=qtp_profile_now();}
#pragma omp barrier
#pragma omp for schedule(static)
#endif
   for(int i=0;i<Q38FN_HIDDEN;i++){double z=0;for(int s=0;s<4;s++){int j=s*Q38FN_HIDDEN+i;z+=(double)qtp_sig(partial[j])*norm[j];}x[i]=(float)(z/4.0);}
#ifdef _OPENMP
#pragma omp master
   {ti=qtp_profile_now();}
  }
#endif
  if(prof){qtp_profile.hc_down+=td-pt;qtp_profile.hc_up+=tu-td;qtp_profile.hc_inject+=ti-tu;}
  return 0;
 }
 if(wd&&fast_wu&&wd->kind==Q38FN_TP_AXIS0&&fast_wu->kind==Q38FN_TP_AXIS1&&
    wd->q8_data&&fast_wu->q8_data&&wd->n_ranges==1&&fast_wu->n_ranges==1&&
    wd->range[0].count==fast_wu->range[0].count&&getenv("Q38FN_TP_FUSED_HC")){
  nr=(int)wd->range[0].count;int8_t qnorm[QTP_HC],qlo[QTP_LAYER_MIX];
  float ns=qtp_q8_quantize(qnorm,norm,QTP_HC),ls=1;double td=pt,tu=pt,ti=pt;
  snprintf(n,sizeof(n),"%s.block_inject_weight.weight",p);const q38fn_tp_blob_entry*wi=qtp_find(m,n);if(!wi||!wi->q8_data)return-1;
#ifdef _OPENMP
#pragma omp parallel shared(td,tu,ti,ls)
  {
#pragma omp for schedule(static)
#endif
   for(int r=0;r<nr;r++)lo[r]=qtp_silu(qtp_entry_row_dot_q8(wd,(size_t)r,qnorm,ns,QTP_HC)/4.0f);
#ifdef _OPENMP
#pragma omp single
#endif
   {td=qtp_profile_now();ls=qtp_q8_quantize(qlo,lo,nr);}
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
   for(int r=0;r<QTP_HC;r++)partial[r]=qtp_entry_row_dot_q8(fast_wu,(size_t)r,qlo,ls,nr);
#ifdef _OPENMP
#pragma omp master
#endif
   {m->sum(partial,QTP_HC,m->comm);tu=qtp_profile_now();}
#ifdef _OPENMP
#pragma omp barrier
#pragma omp for schedule(static)
#endif
   for(int i=0;i<Q38FN_HIDDEN;i++){double z=0;for(int s=0;s<4;s++){int j=s*Q38FN_HIDDEN+i;z+=(double)qtp_sig(partial[j])*norm[j];}x[i]=(float)(z/4.0);}
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
   for(int r=0;r<4;r++)inj[r]=2*qtp_sig(qtp_entry_row_dot_q8(wi,(size_t)r,qnorm,ns,QTP_HC)/4.0f);
#ifdef _OPENMP
#pragma omp master
#endif
   {ti=qtp_profile_now();}
#ifdef _OPENMP
  }
#endif
  if(prof){qtp_profile.hc_down+=td-pt;qtp_profile.hc_up+=tu-td;qtp_profile.hc_inject+=ti-tu;}return 0;
 }
 if(wd&&fast_wu&&wd->kind==Q38FN_TP_AXIS0&&fast_wu->kind==Q38FN_TP_AXIS1&&
    !wd->q5_data&&!wd->q8_data&&!fast_wu->q5_data&&!fast_wu->q8_data&&
    wd->n_ranges==1&&fast_wu->n_ranges==1&&
    wd->range[0].count==fast_wu->range[0].count&&
    getenv("Q38FN_TP_FUSED_HC")){
  nr=(int)wd->range[0].count;double td=pt,tu=pt;
#ifdef _OPENMP
#pragma omp parallel shared(td,tu)
  {
#pragma omp for schedule(static)
#endif
   for(int r=0;r<nr;r++)lo[r]=qtp_silu(qtp_dot(wd->data+(size_t)r*QTP_HC,norm,QTP_HC)/4.0f);
#ifdef _OPENMP
#pragma omp master
#endif
   {td=qtp_profile_now();}
#ifdef _OPENMP
#pragma omp barrier
#pragma omp for schedule(static)
#endif
   for(int r=0;r<QTP_HC;r++)partial[r]=qtp_dot(fast_wu->data+(size_t)r*nr,lo,nr);
#ifdef _OPENMP
#pragma omp master
#endif
   {m->sum(partial,QTP_HC,m->comm);tu=qtp_profile_now();}
#ifdef _OPENMP
#pragma omp barrier
#pragma omp for schedule(static)
#endif
   for(int i=0;i<Q38FN_HIDDEN;i++){double z=0;for(int s=0;s<4;s++){int j=s*Q38FN_HIDDEN+i;z+=(double)qtp_sig(partial[j])*norm[j];}x[i]=(float)(z/4.0);}
#ifdef _OPENMP
  }
#endif
  if(prof){qtp_profile.hc_down+=td-pt;qtp_profile.hc_up+=tu-td;pt=tu;}
  snprintf(n,sizeof(n),"%s.block_inject_weight.weight",p);const q38fn_tp_blob_entry*wi=qtp_find(m,n);
  if(qtp_full_mv(wi,norm,4,QTP_HC,inj))return-1;for(int i=0;i<4;i++)inj[i]=2*qtp_sig(inj[i]/4);
  if(prof)qtp_profile.hc_inject+=qtp_profile_now()-pt;return 0;
 }
 if(wd&&wd->kind==Q38FN_TP_FULL&&replicated){nr=(int)wd->shape[0];if(nr!=QTP_LAYER_MIX||qtp_full_mv(wd,norm,nr,QTP_HC,lo))return-1;}
 else if(wd&&wd->kind==Q38FN_TP_FULL){uint64_t count;q38fn_tp_split(QTP_LAYER_MIX,m->rank,m->ranks,&hc_start,&count);nr=(int)count;qtp_mv_rows(wd->data+hc_start*QTP_HC,norm,nr,QTP_HC,lo);}
 else {nr=qtp_axis0_mv(wd,norm,QTP_HC,lo);if(nr<0||nr>QTP_LAYER_MIX)return-1;}
 for(int i=0;i<nr;i++)lo[i]=qtp_silu(lo[i]/4.0f);
 if(prof){qtp_profile.hc_down+=qtp_profile_now()-pt;pt=qtp_profile_now();}
 snprintf(n,sizeof(n),"%s.input_mix_weight_up.weight",p);const q38fn_tp_blob_entry*wu=qtp_find(m,n);if(!wu)return-1;
 if(wu->kind==Q38FN_TP_FULL&&replicated){if(qtp_full_mv(wu,lo,QTP_HC,nr,partial))return-1;}
 else if(wu->kind==Q38FN_TP_FULL){
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((long long)QTP_HC*nr>=qtp_omp_threshold())
#endif
  for(int r=0;r<QTP_HC;r++)partial[r]=qtp_dot(wu->data+(size_t)r*QTP_LAYER_MIX+hc_start,lo,nr);
  m->sum(partial,QTP_HC,m->comm);
 }
 else {
  if(wu->kind!=Q38FN_TP_AXIS1||wu->n_ranges!=1||wu->range[0].count!=(uint64_t)nr)return-1;
  if(wu->q8_data){int8_t qlo[QTP_LAYER_MIX];float qlo_scale=qtp_q8_quantize(qlo,lo,nr);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((long long)QTP_HC*nr>=qtp_omp_threshold())
#endif
   for(int r=0;r<QTP_HC;r++)partial[r]=qtp_entry_row_dot_q8(wu,(size_t)r,qlo,qlo_scale,nr);
  }else{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((long long)QTP_HC*nr>=qtp_omp_threshold())
#endif
   for(int r=0;r<QTP_HC;r++)partial[r]=qtp_entry_row_dot(wu,(size_t)r,lo,nr);
  }
  m->sum(partial,QTP_HC,m->comm);
 }
 if(prof){qtp_profile.hc_up+=qtp_profile_now()-pt;pt=qtp_profile_now();}
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
 for(int i=0;i<Q38FN_HIDDEN;i++){double z=0;for(int s=0;s<4;s++){int j=s*Q38FN_HIDDEN+i;z+=(double)qtp_sig(partial[j])*norm[j];}x[i]=(float)(z/4.0);}
 snprintf(n,sizeof(n),"%s.block_inject_weight.weight",p);const q38fn_tp_blob_entry*wi=qtp_find(m,n);if(qtp_full_mv(wi,norm,4,QTP_HC,inj))return-1;for(int i=0;i<4;i++)inj[i]=2*qtp_sig(inj[i]/4);if(prof)qtp_profile.hc_inject+=qtp_profile_now()-pt;return 0;}
static void qtp_residual(float*h,const float*y,const float inj[4]){for(int s=0;s<4;s++)for(int i=0;i<Q38FN_HIDDEN;i++)h[s*Q38FN_HIDDEN+i]+=inj[s]*y[i];}

static int qtp_moe(q38fn_tp_model*m,int L,const float*x,float*y){char n[224];float logits[Q38FN_EXPERTS],sw[Q38FN_ACTIVE_EXPERTS];int sel[Q38FN_ACTIVE_EXPERTS];int prof=qtp_profile_enabled(),fdetail=qtp_fapp_detail();double pt=prof?qtp_profile_now():0,route_quant=0,route_mv=0;if(fdetail)QTP_FAPP_START("moe_route");
 if(qtp_name(n,sizeof(n),L,"mlp.gate.weight"))return-1;const q38fn_tp_blob_entry*gate=qtp_find(m,n);const char*gt=getenv("Q38FN_TP_GATE_THREADS");int gate_threads=gt?atoi(gt):0;if(gate_threads<0)gate_threads=0;
#if defined(__ARM_FEATURE_SVE)
 if(gate&&gate->q8_data&&getenv("Q38FN_TP_SERIAL_Q8_GATE")){
  int8_t qx[Q38FN_HIDDEN];float xs=qtp_q8_quantize(qx,x,Q38FN_HIDDEN);
  for(int r=0;r<Q38FN_EXPERTS;r+=8){int32_t dot[8];k3_q8_dot8(dot,gate->q8_data+(size_t)r*Q38FN_HIDDEN,qx,Q38FN_HIDDEN);int count=Q38FN_EXPERTS-r<8?Q38FN_EXPERTS-r:8;for(int i=0;i<count;i++)logits[r+i]=(float)dot[i]*gate->q8_scales[r+i]*xs;}
 }else if(gate&&gate->q8_data&&gate_threads){
  double qt=prof?qtp_profile_now():0;int8_t qx[Q38FN_HIDDEN];float xs=qtp_q8_quantize(qx,x,Q38FN_HIDDEN);if(prof){route_quant=qtp_profile_now()-qt;qt=qtp_profile_now();}
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(gate_threads)
#endif
  for(int r=0;r<Q38FN_EXPERTS;r+=8){int32_t dot[8];k3_q8_dot8(dot,gate->q8_data+(size_t)r*Q38FN_HIDDEN,qx,Q38FN_HIDDEN);int count=Q38FN_EXPERTS-r<8?Q38FN_EXPERTS-r:8;for(int i=0;i<count;i++)logits[r+i]=(float)dot[i]*gate->q8_scales[r+i]*xs;}
  if(prof)route_mv=qtp_profile_now()-qt;
 }else
#endif
 if((gate&&gate->q5_data&&gate_threads)?qtp_q5_mv_threads(gate,x,Q38FN_EXPERTS,Q38FN_HIDDEN,logits,gate_threads):qtp_full_mv(gate,x,Q38FN_EXPERTS,Q38FN_HIDDEN,logits))return-1;
 for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++)sel[k]=-1;
 for(int e=0;e<Q38FN_EXPERTS;e++){int pos=0;while(pos<Q38FN_ACTIVE_EXPERTS&&sel[pos]>=0&&!(logits[e]>logits[sel[pos]]))pos++;if(pos<Q38FN_ACTIVE_EXPERTS){for(int k=Q38FN_ACTIVE_EXPERTS-1;k>pos;k--)sel[k]=sel[k-1];sel[pos]=e;}}
 float ss=0,mx=logits[sel[0]];for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++){sw[k]=expf(logits[sel[k]]-mx);ss+=sw[k];}for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++)sw[k]/=ss;
 if(fdetail){QTP_FAPP_STOP("moe_route");QTP_FAPP_START("moe_up");}if(prof){double rt=qtp_profile_now()-pt;qtp_profile.moe_route+=rt;qtp_profile.moe_route_quant+=route_quant;qtp_profile.moe_route_mv+=route_mv;qtp_profile.moe_route_select+=rt-route_quant-route_mv;pt=qtp_profile_now();}
 qtp_name(n,sizeof(n),L,"mlp.experts.gate_up_proj");const q38fn_tp_blob_entry*gu=qtp_find(m,n);qtp_name(n,sizeof(n),L,"mlp.experts.down_proj");const q38fn_tp_blob_entry*dw=qtp_find(m,n);if(!gu||!dw)return-1;int d=(int)gu->range[0].count;float packed[Q38FN_ACTIVE_EXPERTS][2*Q38FN_EXPERT_INTERMEDIATE],hidden[Q38FN_ACTIVE_EXPERTS][Q38FN_EXPERT_INTERMEDIATE];int8_t expert_qx[Q38FN_HIDDEN];float expert_xs=gu->q8_data?qtp_q8_quantize(expert_qx,x,Q38FN_HIDDEN):0;
 qtp_name(n,sizeof(n),L,"mlp.shared_expert.gate_proj.weight");const q38fn_tp_blob_entry*sg=qtp_find(m,n);qtp_name(n,sizeof(n),L,"mlp.shared_expert.up_proj.weight");const q38fn_tp_blob_entry*su=qtp_find(m,n);qtp_name(n,sizeof(n),L,"mlp.shared_expert.down_proj.weight");const q38fn_tp_blob_entry*sd=qtp_find(m,n);qtp_name(n,sizeof(n),L,"mlp.shared_expert_gate.weight");const q38fn_tp_blob_entry*se=qtp_find(m,n);float shared_proj[2][Q38FN_EXPERT_INTERMEDIATE],sh[Q38FN_EXPERT_INTERMEDIATE];if(!sg||!su||!sd||!se||d<1||d>Q38FN_EXPERT_INTERMEDIATE||sg->range[0].count!=(uint64_t)d)return-1;int fused_moe_up=gu->q5_data&&sg->q5_data&&su->q5_data&&!getenv("Q38FN_TP_NO_FUSED_MOE_UP");int fused_q8_up=gu->q8_data&&sg->q8_data&&su->q8_data&&!getenv("Q38FN_TP_NO_FUSED_Q8_MOE_UP");
 if(fused_q8_up&&!dw->q5_data&&!dw->q8_data&&dw->data&&sd->data&&!sd->q5_data&&!sd->q8_data&&getenv("Q38FN_TP_PERSISTENT_MOE")){
#if defined(__ARM_FEATURE_SVE)
  int expert_groups=(2*d+7)/8,shared_groups=(d+7)/8;float scale=0;
#ifdef _OPENMP
#pragma omp parallel shared(scale,pt)
  {
#pragma omp for schedule(static)
#endif
   for(int task=0;task<Q38FN_ACTIVE_EXPERTS*expert_groups+2*shared_groups;task++){
    const q38fn_tp_blob_entry*e;size_t row;float*out;int count;
    if(task<Q38FN_ACTIVE_EXPERTS*expert_groups){int k=task/expert_groups,r=(task%expert_groups)*8;e=gu;row=(size_t)sel[k]*2*d+(size_t)r;out=packed[k]+r;count=2*d-r<8?2*d-r:8;}
    else{int t=task-Q38FN_ACTIVE_EXPERTS*expert_groups,which=t/shared_groups,r=(t%shared_groups)*8;e=which?su:sg;row=(size_t)r;out=shared_proj[which]+r;count=d-r<8?d-r:8;}
    const float*sc;const int8_t*w=qtp_q8_row(e,row,Q38FN_HIDDEN,&sc);
    if(count==8){int32_t dot[8];k3_q8_dot8(dot,w,expert_qx,Q38FN_HIDDEN);for(int i=0;i<8;i++)out[i]=(float)dot[i]*sc[i]*expert_xs;}
    else for(int i=0;i<count;i++){const float*si;const int8_t*wi=qtp_q8_row(e,row+(size_t)i,Q38FN_HIDDEN,&si);out[i]=(float)k3_q8_dot(wi,expert_qx,Q38FN_HIDDEN)*(*si)*expert_xs;}
   }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
   for(int task=0;task<(Q38FN_ACTIVE_EXPERTS+1)*d;task++){
    if(task<Q38FN_ACTIVE_EXPERTS*d){int k=task/d,i=task%d;hidden[k][i]=qtp_silu(packed[k][i])*packed[k][d+i];}
    else{int i=task-Q38FN_ACTIVE_EXPERTS*d;sh[i]=qtp_silu(shared_proj[0][i])*shared_proj[1][i];}
   }
#ifdef _OPENMP
#pragma omp master
#endif
   {scale=qtp_sig(qtp_entry_row_dot(se,0,x,Q38FN_HIDDEN));if(fdetail){QTP_FAPP_STOP("moe_up");QTP_FAPP_START("moe_down");}if(prof){qtp_profile.moe_up+=qtp_profile_now()-pt;pt=qtp_profile_now();}}
#ifdef _OPENMP
#pragma omp barrier
#pragma omp for schedule(static)
#endif
   for(int r=0;r<Q38FN_HIDDEN;r+=8){float z[8],v8[8];const uint16_t*w=sd->data+(size_t)r*(size_t)d;matvec_bf16_8row(z,w,w+d,w+2*d,w+3*d,w+4*d,w+5*d,w+6*d,w+7*d,sh,d);for(int i=0;i<8;i++)z[i]*=scale;for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++){w=dw->data+((size_t)sel[k]*Q38FN_HIDDEN+(size_t)r)*(size_t)d;matvec_bf16_8row(v8,w,w+d,w+2*d,w+3*d,w+4*d,w+5*d,w+6*d,w+7*d,hidden[k],d);for(int i=0;i<8;i++)z[i]+=sw[k]*v8[i];}for(int i=0;i<8;i++)y[r+i]=z[i];}
#ifdef _OPENMP
  }
#endif
  if(getenv("Q38FN_TP_MOE_DEBUG")&&!qtp_finite_n(y,Q38FN_HIDDEN))fprintf(stderr,"q38fn moe rank=%d: nonfinite local down projection\n",m->rank);
  if(fdetail){QTP_FAPP_STOP("moe_down");QTP_FAPP_START("moe_reduce");}if(prof){qtp_profile.moe_down+=qtp_profile_now()-pt;pt=qtp_profile_now();}
  if(!getenv("Q38FN_TP_SKIP_REDUCE"))m->sum(y,Q38FN_HIDDEN,m->comm);
  if(fdetail)QTP_FAPP_STOP("moe_reduce");if(prof)qtp_profile.moe_reduce+=qtp_profile_now()-pt;return 0;
#else
  return-1;
#endif
 }
 if(fused_moe_up&&!getenv("Q38FN_TP_NO_Q5_PAIR_MOE_UP")){
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp for schedule(static)
#endif
   for(int task=0;task<(Q38FN_ACTIVE_EXPERTS+1)*d;task++){
    if(task<Q38FN_ACTIVE_EXPERTS*d){int k=task/d,r=2*(task%d);size_t row=(size_t)sel[k]*2*d+(size_t)r;q38fn_q5_matvec_pair(packed[k]+r,gu->q5_data+row*(Q38FN_HIDDEN/32),gu->q5_data+(row+1)*(Q38FN_HIDDEN/32),x,Q38FN_HIDDEN);}
    else{int r=2*(task-Q38FN_ACTIVE_EXPERTS*d);const q38fn_tp_blob_entry*e=r<d?sg:su;int er=r<d?r:r-d;q38fn_q5_matvec_pair((float*)shared_proj+ r,e->q5_data+(size_t)er*(Q38FN_HIDDEN/32),e->q5_data+(size_t)(er+1)*(Q38FN_HIDDEN/32),x,Q38FN_HIDDEN);}
   }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
   for(int task=0;task<(Q38FN_ACTIVE_EXPERTS+1)*d;task++){
    if(task<Q38FN_ACTIVE_EXPERTS*d){int k=task/d,i=task%d;hidden[k][i]=qtp_silu(packed[k][i])*packed[k][d+i];}
    else{int i=task-Q38FN_ACTIVE_EXPERTS*d;sh[i]=qtp_silu(shared_proj[0][i])*shared_proj[1][i];}
   }
#ifdef _OPENMP
  }
#endif
 }else if(fused_moe_up){
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp for schedule(static)
#endif
   for(int task=0;task<(Q38FN_ACTIVE_EXPERTS+1)*2*d;task++){
    if(task<Q38FN_ACTIVE_EXPERTS*2*d){int k=task/(2*d),r=task%(2*d);packed[k][r]=qtp_entry_row_dot(gu,(size_t)sel[k]*2*d+r,x,Q38FN_HIDDEN);}
    else{int r=task-Q38FN_ACTIVE_EXPERTS*2*d;const q38fn_tp_blob_entry*e=r<d?sg:su;shared_proj[r/d][r%d]=qtp_entry_row_dot(e,(size_t)(r%d),x,Q38FN_HIDDEN);}
   }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
   for(int task=0;task<(Q38FN_ACTIVE_EXPERTS+1)*d;task++){
    if(task<Q38FN_ACTIVE_EXPERTS*d){int k=task/d,i=task%d;hidden[k][i]=qtp_silu(packed[k][i])*packed[k][d+i];}
    else{int i=task-Q38FN_ACTIVE_EXPERTS*d;sh[i]=qtp_silu(shared_proj[0][i])*shared_proj[1][i];}
   }
#ifdef _OPENMP
  }
#endif
 }else if(fused_q8_up){
#if defined(__ARM_FEATURE_SVE)
  int expert_groups=(2*d+7)/8,shared_groups=(d+7)/8;
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp for schedule(static)
#endif
   for(int task=0;task<Q38FN_ACTIVE_EXPERTS*expert_groups+2*shared_groups;task++){
    const q38fn_tp_blob_entry*e;size_t row;float*out;int count;
    if(task<Q38FN_ACTIVE_EXPERTS*expert_groups){int k=task/expert_groups,r=(task%expert_groups)*8;e=gu;row=(size_t)sel[k]*2*d+(size_t)r;out=packed[k]+r;count=2*d-r<8?2*d-r:8;}
    else{int t=task-Q38FN_ACTIVE_EXPERTS*expert_groups,which=t/shared_groups,r=(t%shared_groups)*8;e=which?su:sg;row=(size_t)r;out=shared_proj[which]+r;count=d-r<8?d-r:8;}
    const float*sc;const int8_t*w=qtp_q8_row(e,row,Q38FN_HIDDEN,&sc);
    if(count==8){int32_t dot[8];k3_q8_dot8(dot,w,expert_qx,Q38FN_HIDDEN);for(int i=0;i<8;i++)out[i]=(float)dot[i]*sc[i]*expert_xs;}
    else for(int i=0;i<count;i++){const float*si;const int8_t*wi=qtp_q8_row(e,row+(size_t)i,Q38FN_HIDDEN,&si);out[i]=(float)k3_q8_dot(wi,expert_qx,Q38FN_HIDDEN)*(*si)*expert_xs;}
   }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
   for(int task=0;task<(Q38FN_ACTIVE_EXPERTS+1)*d;task++){
    if(task<Q38FN_ACTIVE_EXPERTS*d){int k=task/d,i=task%d;hidden[k][i]=qtp_silu(packed[k][i])*packed[k][d+i];}
    else{int i=task-Q38FN_ACTIVE_EXPERTS*d;sh[i]=qtp_silu(shared_proj[0][i])*shared_proj[1][i];}
   }
#ifdef _OPENMP
  }
#endif
#else
  return-1;
#endif
 }else if(gu->q8_data){size_t ids[Q38FN_ACTIVE_EXPERTS];for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++)ids[k]=(size_t)sel[k];
  if(qtp_q8_mv_blocks(gu,ids,Q38FN_ACTIVE_EXPERTS,expert_qx,expert_xs,2*d,Q38FN_HIDDEN,(float*)packed))return-1;
 }else{
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++)for(int r=0;r<2*d;r++)packed[k][r]=qtp_entry_row_dot(gu,(size_t)sel[k]*2*d+r,x,Q38FN_HIDDEN);
 }
 if(!fused_moe_up&&!fused_q8_up){
#if defined(__ARM_FEATURE_SVE)
  if(gu->q8_data&&sg->q8_data&&su->q8_data&&!getenv("Q38FN_TP_NO_Q8_SHARED_UP")){
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
   for(int task=0;task<2*((d+7)/8);task++){int which=task/((d+7)/8),r=(task%((d+7)/8))*8,count=d-r<8?d-r:8;const q38fn_tp_blob_entry*e=which?su:sg;int32_t dot[8];k3_q8_dot8(dot,e->q8_data+(size_t)r*Q38FN_HIDDEN,expert_qx,Q38FN_HIDDEN);for(int i=0;i<count;i++)shared_proj[which][r+i]=(float)dot[i]*e->q8_scales[r+i]*expert_xs;}
  }else
#endif
  {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
   for(int r=0;r<2*d;r++){const q38fn_tp_blob_entry*e=r<d?sg:su;shared_proj[r/d][r%d]=qtp_entry_row_dot(e,(size_t)(r%d),x,Q38FN_HIDDEN);}
  }
  for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++)for(int i=0;i<d;i++)hidden[k][i]=qtp_silu(packed[k][i])*packed[k][d+i];
 }
 if(getenv("Q38FN_TP_MOE_DEBUG")&&!qtp_finite_n((const float*)hidden,(size_t)Q38FN_ACTIVE_EXPERTS*Q38FN_EXPERT_INTERMEDIATE))fprintf(stderr,"q38fn moe rank=%d: nonfinite expert gate/up\\n",m->rank);
 if(!fused_moe_up&&!fused_q8_up)for(int i=0;i<d;i++)sh[i]=qtp_silu(shared_proj[0][i])*shared_proj[1][i];
 if(getenv("Q38FN_TP_MOE_DEBUG")&&!qtp_finite_n(sh,(size_t)d))fprintf(stderr,"q38fn moe rank=%d: nonfinite shared gate/up\\n",m->rank);
 float scale=qtp_sig(qtp_entry_row_dot(se,0,x,Q38FN_HIDDEN));
 if(fdetail){QTP_FAPP_STOP("moe_up");QTP_FAPP_START("moe_down");}if(prof){qtp_profile.moe_up+=qtp_profile_now()-pt;pt=qtp_profile_now();}
 if(!dw->q5_data&&!dw->q8_data&&dw->data&&sd->data&&!sd->q5_data&&
    !sd->q8_data&&!getenv("Q38FN_TP_NO_FUSED_BF16_MOE_DOWN")){
#if defined(__ARM_FEATURE_SVE)
  /* Produce an eight-row output tile directly instead of materializing ten
   * expert matrices plus the shared-expert matrix.  Each task walks eleven
   * independent contiguous BF16 streams, giving the core enough outstanding
   * HBM requests while retaining the eight-row SVE reuse of the activation.
   * Immediate weighted accumulation also removes 112 KiB of temporary
   * output traffic per layer and its separate reduction pass. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int r=0;r<Q38FN_HIDDEN;r+=8){
   float z[8],v[8];
   const uint16_t*w=sd->data+(size_t)r*(size_t)d;
   matvec_bf16_8row(z,w,w+d,w+2*d,w+3*d,w+4*d,w+5*d,w+6*d,w+7*d,sh,d);
   for(int i=0;i<8;i++)z[i]*=scale;
   for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++){
    w=dw->data+((size_t)sel[k]*Q38FN_HIDDEN+(size_t)r)*(size_t)d;
    matvec_bf16_8row(v,w,w+d,w+2*d,w+3*d,w+4*d,w+5*d,w+6*d,w+7*d,hidden[k],d);
    for(int i=0;i<8;i++)z[i]+=sw[k]*v[i];
   }
   for(int i=0;i<8;i++)y[r+i]=z[i];
  }
#else
  return-1;
#endif
 }else if(getenv("Q38FN_TP_GROUP_DOWN")){
  float expert_down[Q38FN_ACTIVE_EXPERTS][Q38FN_HIDDEN],shared_down[Q38FN_HIDDEN];
#if defined(__ARM_FEATURE_SVE)
  int fused_bf16_down=!dw->q5_data&&!dw->q8_data&&sd->data&&!sd->q5_data&&!sd->q8_data&&!getenv("Q38FN_TP_NO_FUSED_MOE_DOWN");
#else
  int fused_bf16_down=0;
#endif
  if(fused_bf16_down){
#if defined(__ARM_FEATURE_SVE)
   int groups=Q38FN_HIDDEN/8;
#ifdef _OPENMP
#pragma omp parallel
   {
#pragma omp for collapse(2) schedule(static)
#endif
    for(int k=0;k<Q38FN_ACTIVE_EXPERTS+1;k++)for(int g=0;g<groups;g++){
     const uint16_t*w;const float*v;float*out;
     if(k<Q38FN_ACTIVE_EXPERTS){w=dw->data+((size_t)sel[k]*Q38FN_HIDDEN+(size_t)g*8)*(size_t)d;v=hidden[k];out=expert_down[k]+g*8;}
     else{w=sd->data+(size_t)g*8*(size_t)d;v=sh;out=shared_down+g*8;}
     matvec_bf16_8row(out,w,w+d,w+2*d,w+3*d,w+4*d,w+5*d,w+6*d,w+7*d,v,d);
    }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for(int r=0;r<Q38FN_HIDDEN;r++){float z=scale*shared_down[r];for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++)z+=sw[k]*expert_down[k][r];y[r]=z;}
#ifdef _OPENMP
   }
#endif
#endif
  }else if(dw->q5_data){size_t ids[Q38FN_ACTIVE_EXPERTS];for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++)ids[k]=(size_t)sel[k];
   if(q38fn_q5_matvec_indexed((float*)expert_down,dw->q5_data,(const float*)hidden,ids,Q38FN_ACTIVE_EXPERTS,Q38FN_HIDDEN,d))return-1;
  }else if(!dw->q8_data){size_t ids[Q38FN_ACTIVE_EXPERTS];for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++)ids[k]=(size_t)sel[k];
   if(qtp_bf16_mv_blocks(dw,ids,Q38FN_ACTIVE_EXPERTS,(const float*)hidden,
                         Q38FN_HIDDEN,d,(float*)expert_down))return-1;
  }else for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++)
   if(qtp_entry_mv_offset(dw,(size_t)sel[k]*Q38FN_HIDDEN,hidden[k],Q38FN_HIDDEN,d,expert_down[k]))return-1;
  if(!fused_bf16_down&&qtp_entry_mv_offset(sd,0,sh,Q38FN_HIDDEN,d,shared_down))return-1;
  if(!fused_bf16_down){
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
   for(int r=0;r<Q38FN_HIDDEN;r++){float z=scale*shared_down[r];for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++)z+=sw[k]*expert_down[k][r];y[r]=z;}
  }
 }else{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int r=0;r<Q38FN_HIDDEN;r++){float z=0;for(int k=0;k<Q38FN_ACTIVE_EXPERTS;k++){size_t row=(size_t)sel[k]*Q38FN_HIDDEN+r;z+=sw[k]*qtp_entry_row_dot(dw,row,hidden[k],d);}y[r]=z+scale*qtp_entry_row_dot(sd,(size_t)r,sh,d);}
 }
 if(getenv("Q38FN_TP_MOE_DEBUG")&&!qtp_finite_n(y,Q38FN_HIDDEN))fprintf(stderr,"q38fn moe rank=%d: nonfinite local down projection\\n",m->rank);
 if(fdetail){QTP_FAPP_STOP("moe_down");QTP_FAPP_START("moe_reduce");}if(prof){qtp_profile.moe_down+=qtp_profile_now()-pt;pt=qtp_profile_now();}
 if(!getenv("Q38FN_TP_SKIP_REDUCE"))m->sum(y,Q38FN_HIDDEN,m->comm);
 if(fdetail)QTP_FAPP_STOP("moe_reduce");if(prof)qtp_profile.moe_reduce+=qtp_profile_now()-pt;return 0;}

static void qtp_delta_recurrent(float*state,int first_gh,int value_heads,
 const float*qq,const float*kk,const float*v,const float*dec,const float*beta,
 float*core){
 float del[Q38FN_LINEAR_VALUE_HEADS][128];
#if defined(__ARM_FEATURE_SVE)
 int lanes=(int)svcntw(),tiles=128/lanes;
#ifdef _OPENMP
 if(!getenv("Q38FN_TP_NO_DELTA_CMG_STRIPE")){
  /* The TP12 recurrent update has 4 heads x 8 SVE column tiles.  A normal
   * 32-thread close-bound team leaves CMG3 idle, although state was first
   * touched across all four CMGs.  Keep a 48-thread team and permute the 32
   * active task owners as 0,12,24,36,1,13,... so every CMG contributes eight
   * cores and accesses its local quarter of the recurrent state. */
#pragma omp parallel num_threads(48)
  {
   int tid=omp_get_thread_num(),nth=omp_get_num_threads();
   int per=nth/4,order=(nth==48)?(tid%per)*4+tid/per:tid;
   int tasks=value_heads*tiles;
   if(!getenv("Q38FN_TP_NO_FUSED_DELTA_RECURRENT")){for(int task=order;task<tasks;task+=nth){
    int lh=task/tiles,tile=task%tiles,j=tile*lanes,gh=first_gh+lh;float*mat=state+(size_t)gh*128*128;svbool_t pg=svptrue_b32();svfloat32_t mv=svdup_f32(0),dv=svdup_f32(dec[lh]);
    for(int i=0;i<128;i++){float*row=mat+(size_t)i*128+j;svfloat32_t z=svmul_f32_x(pg,svld1_f32(pg,row),dv);svst1_f32(pg,row,z);mv=svmla_n_f32_x(pg,mv,z,kk[(size_t)lh*128+i]);}
    svfloat32_t d=svmul_n_f32_x(pg,svsub_f32_x(pg,svld1_f32(pg,v+(size_t)lh*128+j),mv),beta[lh]),ov=svdup_f32(0);
    for(int i=0;i<128;i++){float*row=mat+(size_t)i*128+j;svfloat32_t z=svmla_n_f32_x(pg,svld1_f32(pg,row),d,kk[(size_t)lh*128+i]);svst1_f32(pg,row,z);ov=svmla_n_f32_x(pg,ov,z,qq[(size_t)lh*128+i]);}
    svst1_f32(pg,core+(size_t)lh*128+j,ov);
   }}else{
   for(int task=order;task<tasks;task+=nth){
    int lh=task/tiles,tile=task%tiles,j=tile*lanes,gh=first_gh+lh;
    float*mat=state+(size_t)gh*128*128;svbool_t pg=svptrue_b32();
    svfloat32_t mv=svdup_f32(0),dv=svdup_f32(dec[lh]);
    for(int i=0;i<128;i++){float*row=mat+(size_t)i*128+j;svfloat32_t z=svmul_f32_x(pg,svld1_f32(pg,row),dv);svst1_f32(pg,row,z);mv=svmla_n_f32_x(pg,mv,z,kk[(size_t)lh*128+i]);}
    svfloat32_t d=svmul_n_f32_x(pg,svsub_f32_x(pg,svld1_f32(pg,v+(size_t)lh*128+j),mv),beta[lh]);svst1_f32(pg,del[lh]+j,d);
   }
#pragma omp barrier
   for(int task=order;task<tasks;task+=nth){
    int lh=task/tiles,tile=task%tiles,j=tile*lanes,gh=first_gh+lh;
    float*mat=state+(size_t)gh*128*128;svbool_t pg=svptrue_b32();
    svfloat32_t d=svld1_f32(pg,del[lh]+j),ov=svdup_f32(0);
    for(int i=0;i<128;i++){float*row=mat+(size_t)i*128+j;svfloat32_t z=svmla_n_f32_x(pg,svld1_f32(pg,row),d,kk[(size_t)lh*128+i]);svst1_f32(pg,row,z);ov=svmla_n_f32_x(pg,ov,z,qq[(size_t)lh*128+i]);}
    svst1_f32(pg,core+(size_t)lh*128+j,ov);
   }
   }
  }
  return;
 }
#endif
#ifdef _OPENMP
#pragma omp parallel num_threads(qtp_delta_threads())
 {
#pragma omp for collapse(2) schedule(static)
#endif
 for(int lh=0;lh<value_heads;lh++)for(int tile=0;tile<tiles;tile++){
  int j=tile*lanes,gh=first_gh+lh;float*mat=state+(size_t)gh*128*128;
  svbool_t pg=svptrue_b32();svfloat32_t mv=svdup_f32(0),dv=svdup_f32(dec[lh]);
  for(int i=0;i<128;i++){float*row=mat+(size_t)i*128+j;svfloat32_t z=svmul_f32_x(pg,svld1_f32(pg,row),dv);svst1_f32(pg,row,z);mv=svmla_n_f32_x(pg,mv,z,kk[(size_t)lh*128+i]);}
  svfloat32_t d=svmul_n_f32_x(pg,svsub_f32_x(pg,svld1_f32(pg,v+(size_t)lh*128+j),mv),beta[lh]);svst1_f32(pg,del[lh]+j,d);
 }
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(static)
#endif
 for(int lh=0;lh<value_heads;lh++)for(int tile=0;tile<tiles;tile++){
  int j=tile*lanes,gh=first_gh+lh;float*mat=state+(size_t)gh*128*128;svbool_t pg=svptrue_b32();svfloat32_t d=svld1_f32(pg,del[lh]+j),ov=svdup_f32(0);
  /* Update and consume each state row while it is resident.  This removes a
   * third 128x128 traversal and its OpenMP barrier. */
  for(int i=0;i<128;i++){float*row=mat+(size_t)i*128+j;svfloat32_t z=svmla_n_f32_x(pg,svld1_f32(pg,row),d,kk[(size_t)lh*128+i]);svst1_f32(pg,row,z);ov=svmla_n_f32_x(pg,ov,z,qq[(size_t)lh*128+i]);}svst1_f32(pg,core+(size_t)lh*128+j,ov);
 }
#ifdef _OPENMP
 }
#endif
 return;
#endif
#ifdef _OPENMP
#pragma omp parallel num_threads(qtp_delta_threads())
 {
#pragma omp for collapse(2) schedule(static)
#endif
 for(int lh=0;lh<value_heads;lh++)for(int j=0;j<128;j++){
  int gh=first_gh+lh;float*mat=state+(size_t)gh*128*128;double mv=0;
  for(int i=0;i<128;i++){float*cell=mat+(size_t)i*128+j;*cell*=dec[lh];mv+=(double)*cell*kk[(size_t)lh*128+i];}
  del[lh][j]=(v[(size_t)lh*128+j]-(float)mv)*beta[lh];
 }
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(static)
#endif
 for(int lh=0;lh<value_heads;lh++)for(int j=0;j<128;j++){
  int gh=first_gh+lh;float*mat=state+(size_t)gh*128*128;double ov=0;
  for(int i=0;i<128;i++){float*z=mat+(size_t)i*128+j;*z+=kk[(size_t)lh*128+i]*del[lh][j];ov+=(double)*z*qq[(size_t)lh*128+i];}
  core[(size_t)lh*128+j]=(float)ov;
 }
#ifdef _OPENMP
 }
#endif
}

static int qtp_delta(q38fn_tp_model*m,int L,q38fn_tp_delta_state*s,const float*x,float*y){char n[224];static _Thread_local float qkv[Q38FN_LINEAR_CONV_DIM];float z[Q38FN_LINEAR_VALUE_DIM],a[Q38FN_LINEAR_VALUE_HEADS],b[Q38FN_LINEAR_VALUE_HEADS];int prof=qtp_profile_enabled(),fdetail=qtp_fapp_detail();double pt=prof?qtp_profile_now():0;if(fdetail)QTP_FAPP_START("delta_proj");
 int value_heads=Q38FN_LINEAR_VALUE_HEADS/m->ranks,first_gh=m->rank*value_heads,first_kh=first_gh/3,last_kh=(first_gh+value_heads-1)/3,qheads=last_kh-first_kh+1,qrows=qheads*128,vrows=value_heads*128,nlocal=2*qrows+vrows;
 qtp_name(n,sizeof(n),L,"linear_attn.in_proj_qkv.weight");const q38fn_tp_blob_entry*qw=qtp_find(m,n);if(!qw)return-1;
 qtp_name(n,sizeof(n),L,"linear_attn.in_proj_z.weight");const q38fn_tp_blob_entry*zw=qtp_find(m,n);
 qtp_name(n,sizeof(n),L,"linear_attn.in_proj_a.weight");const q38fn_tp_blob_entry*aw=qtp_find(m,n);qtp_name(n,sizeof(n),L,"linear_attn.in_proj_b.weight");const q38fn_tp_blob_entry*bw=qtp_find(m,n);
 int fused_mixed_proj=qw->kind==Q38FN_TP_DELTA_QKV&&qw->n_ranges==3&&qw->q8_data&&zw&&zw->kind==Q38FN_TP_AXIS0&&zw->n_ranges==1&&zw->range[0].count==(uint64_t)vrows&&zw->data&&!zw->q5_data&&!zw->q8_data&&aw&&bw&&aw->data&&bw->data&&!aw->q5_data&&!bw->q5_data&&!aw->q8_data&&!bw->q8_data&&!getenv("Q38FN_TP_NO_FUSED_MIXED_DELTA_PROJ");
 int fused_proj=qw->kind==Q38FN_TP_DELTA_QKV&&qw->n_ranges==3&&qw->q5_data&&zw&&zw->kind==Q38FN_TP_AXIS0&&zw->n_ranges==1&&zw->range[0].count==(uint64_t)vrows&&zw->q5_data&&aw&&bw&&aw->q5_data&&bw->q5_data&&getenv("Q38FN_TP_FUSED_DELTA_PROJ");
 if(fused_mixed_proj){
#if defined(__ARM_FEATURE_SVE)
  int8_t qx[Q38FN_HIDDEN];float xs=qtp_q8_quantize(qx,x,Q38FN_HIDDEN);int qg=(nlocal+7)/8,zg=(vrows+7)/8,abg=48/8;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int task=0;task<qg+zg+2*abg;task++){
   if(task<qg){int r=task*8,count=nlocal-r<8?nlocal-r:8;const float*sc;const int8_t*w=qtp_q8_row(qw,(size_t)r,Q38FN_HIDDEN,&sc);if(count==8){int32_t dot[8];k3_q8_dot8(dot,w,qx,Q38FN_HIDDEN);for(int i=0;i<8;i++)qkv[r+i]=(float)dot[i]*sc[i]*xs;}else for(int i=0;i<count;i++){const float*si;const int8_t*wi=qtp_q8_row(qw,(size_t)r+i,Q38FN_HIDDEN,&si);qkv[r+i]=(float)k3_q8_dot(wi,qx,Q38FN_HIDDEN)*(*si)*xs;}}
   else{int t=task-qg;const q38fn_tp_blob_entry*e;float*out;int r;if(t<zg){e=zw;out=z;r=t*8;}else{t-=zg;int which=t/abg;e=which?bw:aw;out=which?b:a;r=(t%abg)*8;}const uint16_t*w=e->data+(size_t)r*Q38FN_HIDDEN;matvec_bf16_8row(out+r,w,w+Q38FN_HIDDEN,w+2*Q38FN_HIDDEN,w+3*Q38FN_HIDDEN,w+4*Q38FN_HIDDEN,w+5*Q38FN_HIDDEN,w+6*Q38FN_HIDDEN,w+7*Q38FN_HIDDEN,x,Q38FN_HIDDEN);}
  }
  if(!qtp_finite_n(qkv,(size_t)nlocal)||!qtp_finite_n(z,(size_t)vrows)||!qtp_finite_n(a,48)||!qtp_finite_n(b,48))return-1;
#else
  return-1;
#endif
 }else if(fused_proj){size_t blocks=Q38FN_HIDDEN/32;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int task=0;task<nlocal+vrows+96;task++){
   const q38fn_tp_blob_entry*e;size_t row;float*out;
   if(task<nlocal){e=qw;row=(size_t)task;out=qkv+task;}
   else if(task<nlocal+vrows){row=(size_t)(task-nlocal);e=zw;out=z+row;}
   else{int t=task-nlocal-vrows;e=t<48?aw:bw;row=(size_t)(t%48);out=(t<48?a:b)+row;}
   if(q38fn_q5_matvec(out,e->q5_data+row*blocks,x,1,Q38FN_HIDDEN))*out=NAN;
  }
  if(!qtp_finite_n(qkv,(size_t)nlocal)||!qtp_finite_n(z,(size_t)vrows)||!qtp_finite_n(a,48)||!qtp_finite_n(b,48))return-1;
 }else{
  if(qw->kind==Q38FN_TP_DELTA_QKV){if(qw->n_ranges!=3||qtp_entry_mv(qw,x,nlocal,Q38FN_HIDDEN,qkv))return-1;}
  else if(qw->kind==Q38FN_TP_FULL){qtp_mv_rows(qw->data+(size_t)first_kh*128*Q38FN_HIDDEN,x,qrows,Q38FN_HIDDEN,qkv);qtp_mv_rows(qw->data+(size_t)(2048+first_kh*128)*Q38FN_HIDDEN,x,qrows,Q38FN_HIDDEN,qkv+qrows);qtp_mv_rows(qw->data+(size_t)(4096+first_gh*128)*Q38FN_HIDDEN,x,vrows,Q38FN_HIDDEN,qkv+2*qrows);}else return-1;
  if(qtp_axis0_mv(zw,x,Q38FN_HIDDEN,z)!=vrows)return-1;
  if((aw&&bw&&aw->q5_data&&bw->q5_data)?qtp_q5_mv_pair(aw,bw,x,48,Q38FN_HIDDEN,a,b):(qtp_full_mv(aw,x,48,Q38FN_HIDDEN,a)||qtp_full_mv(bw,x,48,Q38FN_HIDDEN,b)))return-1;
 }
 qtp_name(n,sizeof(n),L,"linear_attn.conv1d.weight");const q38fn_tp_blob_entry*cw=qtp_find(m,n);qtp_name(n,sizeof(n),L,"linear_attn.A_log");const q38fn_tp_blob_entry*al=qtp_find(m,n);qtp_name(n,sizeof(n),L,"linear_attn.dt_bias");const q38fn_tp_blob_entry*db=qtp_find(m,n);qtp_name(n,sizeof(n),L,"linear_attn.norm.weight");const q38fn_tp_blob_entry*nw=qtp_find(m,n);if(!cw||!al||!db||!nw)return-1;
 if(fdetail){QTP_FAPP_STOP("delta_proj");QTP_FAPP_START("delta_conv");}if(prof){qtp_profile.delta_proj+=qtp_profile_now()-pt;pt=qtp_profile_now();}
 for(int c=0;c<nlocal;c++){int gc=c<qrows?first_kh*128+c:(c<2*qrows?2048+first_kh*128+c-qrows:4096+first_gh*128+c-2*qrows);float*hist=s->conv+(size_t)c*4;for(int k=0;k<3;k++)hist[k]=hist[k+1];hist[3]=qkv[c];float v=0;for(int k=0;k<4;k++)v+=hist[k]*qtp_bf(cw->data[(size_t)gc*4+k]);qkv[c]=qtp_silu(v);}
 qtp_debug_dump(m,"delta-qkv",qkv,(size_t)nlocal);
 if(fdetail){QTP_FAPP_STOP("delta_conv");QTP_FAPP_START("delta_recurrent");}if(prof){qtp_profile.delta_conv+=qtp_profile_now()-pt;pt=qtp_profile_now();}
 float core[Q38FN_LINEAR_VALUE_DIM];
 /* Parallelize by output column, not by stateful head.  Each (head,column)
  * task owns every matrix element it updates and retains the original
  * ascending-row accumulation order.  This avoids the Fujitsu OpenMP issue
  * seen with the old four-iteration head loop while exposing 512 independent
  * TP12 tasks to all 48 A64FX cores. */
 static _Thread_local float qq[Q38FN_LINEAR_VALUE_HEADS][128],kk[Q38FN_LINEAR_VALUE_HEADS][128];
 static _Thread_local float dec[Q38FN_LINEAR_VALUE_HEADS],beta[Q38FN_LINEAR_VALUE_HEADS];
 for(int lh=0;lh<value_heads;lh++){
  int gh=first_gh+lh,kh=gh/3-first_kh;const float*q=qkv+kh*128,*k=qkv+qrows+kh*128;
  double q2=0,k2=0;for(int i=0;i<128;i++){q2+=(double)q[i]*q[i];k2+=(double)k[i]*k[i];}
  float qs=1.0f/(sqrtf((float)q2+1e-6f)*sqrtf(128.0f)),ks=1.0f/sqrtf((float)k2+1e-6f);
  for(int i=0;i<128;i++){qq[lh][i]=q[i]*qs;kk[lh][i]=k[i]*ks;}
  dec[lh]=expf(-expf(qtp_bf(al->data[gh]))*qtp_softplus(a[gh]+qtp_bf(db->data[gh])));beta[lh]=qtp_sig(b[gh]);
 }
 qtp_delta_recurrent(s->recurrent,first_gh,value_heads,(const float*)qq,
                     (const float*)kk,qkv+2*qrows,dec,beta,core);
 for(int lh=0;lh<value_heads;lh++){double sq=0;for(int j=0;j<128;j++){double v=core[lh*128+j];sq+=v*v;}float ns=1.0f/sqrtf((float)(sq/128)+1e-6f);for(int j=0;j<128;j++){int i=lh*128+j;core[i]=core[i]*ns*qtp_bf(nw->data[j])*qtp_sig(z[i]);}}
 qtp_debug_dump(m,"delta-core",core,(size_t)vrows);
 if(fdetail){QTP_FAPP_STOP("delta_recurrent");QTP_FAPP_START("delta_out");}if(prof){qtp_profile.delta_recurrent+=qtp_profile_now()-pt;pt=qtp_profile_now();}
 qtp_name(n,sizeof(n),L,"linear_attn.out_proj.weight");const q38fn_tp_blob_entry*ow=qtp_find(m,n);int rc;
 if(ow&&ow->kind==Q38FN_TP_AXIS1&&ow->n_ranges==1&&ow->q5_data&&!getenv("Q38FN_TP_NO_Q5_PAIR_DELTA_OUT")){rc=qtp_q5_mv_row_pairs(ow,core,Q38FN_HIDDEN,(int)ow->range[0].count,y);if(!rc)m->sum(y,Q38FN_HIDDEN,m->comm);}
 else rc=qtp_axis1_mv(m,ow,core,Q38FN_HIDDEN,y);
 if(fdetail)QTP_FAPP_STOP("delta_out");if(prof)qtp_profile.delta_out+=qtp_profile_now()-pt;return rc;}

int q38fn_tp_linear_layer(q38fn_tp_model*m,int L,q38fn_tp_delta_state*s,float*h){float x[Q38FN_HIDDEN],y[Q38FN_HIDDEN],inj[4];int p=qtp_profile_enabled(),fd=qtp_fapp_detail();double t=p?qtp_profile_now():0;QTP_FAPP_START("hc");int rc=qtp_hc(m,L,"attn",h,x,inj);QTP_FAPP_STOP("hc");if(rc)return-1;if(p){qtp_profile.hc+=qtp_profile_now()-t;qtp_profile.hc_n++;t=qtp_profile_now();}if(!fd)QTP_FAPP_START("delta");rc=qtp_delta(m,L,s,x,y);if(!fd)QTP_FAPP_STOP("delta");if(rc)return-1;if(p){qtp_profile.delta+=qtp_profile_now()-t;qtp_profile.delta_n++;}qtp_residual(h,y,inj);t=p?qtp_profile_now():0;QTP_FAPP_START("hc");rc=qtp_hc(m,L,"mlp",h,x,inj);QTP_FAPP_STOP("hc");if(rc)return-1;if(p){qtp_profile.hc+=qtp_profile_now()-t;qtp_profile.hc_n++;t=qtp_profile_now();}QTP_FAPP_START("moe");rc=qtp_moe(m,L,x,y);QTP_FAPP_STOP("moe");if(rc)return-1;if(p){qtp_profile.moe+=qtp_profile_now()-t;qtp_profile.moe_n++;}qtp_residual(h,y,inj);return 0;}

static void qtp_rope(float*v,size_t pos){for(int i=0;i<32;i++){float a=(float)pos/powf(1e7f,(float)(2*i)/64),c=cosf(a),s=sinf(a),x=v[i],y=v[32+i];v[i]=x*c-y*s;v[32+i]=y*c+x*s;}}
static int qtp_norm_heads(const q38fn_tp_blob_entry*w,float*x,int heads){if(!w)return-1;for(int h=0;h<heads;h++){double q=0;for(int i=0;i<256;i++)q+=(double)x[h*256+i]*x[h*256+i];float z=1.0f/sqrtf((float)(q/256)+1e-6f);for(int i=0;i<256;i++)x[h*256+i]*=z*(1+qtp_bf(w->data[i]));}return 0;}
static int qtp_attention(q38fn_tp_model*m,int L,q38fn_tp_attention_state*s,const float*x,float*y){char n[224];float qg[2*Q38FN_HEADS*Q38FN_HEAD_DIM],q[Q38FN_HEADS*Q38FN_HEAD_DIM],g[Q38FN_HEADS*Q38FN_HEAD_DIM],k[Q38FN_KV_HEADS*Q38FN_HEAD_DIM],v[Q38FN_KV_HEADS*Q38FN_HEAD_DIM],att[Q38FN_HEADS*Q38FN_HEAD_DIM]={0};int local_heads=Q38FN_HEADS/m->ranks,local_width=local_heads*Q38FN_HEAD_DIM;qtp_name(n,sizeof(n),L,"self_attn.q_proj.weight");if(qtp_axis0_mv(qtp_find(m,n),x,Q38FN_HIDDEN,qg)!=2*local_width)return-1;for(int h=0;h<local_heads;h++){memcpy(q+h*Q38FN_HEAD_DIM,qg+h*2*Q38FN_HEAD_DIM,Q38FN_HEAD_DIM*4);memcpy(g+h*Q38FN_HEAD_DIM,qg+h*2*Q38FN_HEAD_DIM+Q38FN_HEAD_DIM,Q38FN_HEAD_DIM*4);}qtp_name(n,sizeof(n),L,"self_attn.k_proj.weight");if(qtp_full_mv(qtp_find(m,n),x,Q38FN_KV_HEADS*Q38FN_HEAD_DIM,Q38FN_HIDDEN,k))return-1;qtp_name(n,sizeof(n),L,"self_attn.v_proj.weight");if(qtp_full_mv(qtp_find(m,n),x,Q38FN_KV_HEADS*Q38FN_HEAD_DIM,Q38FN_HIDDEN,v))return-1;qtp_name(n,sizeof(n),L,"self_attn.q_norm.weight");if(qtp_norm_heads(qtp_find(m,n),q,local_heads))return-1;qtp_name(n,sizeof(n),L,"self_attn.k_norm.weight");if(qtp_norm_heads(qtp_find(m,n),k,Q38FN_KV_HEADS))return-1;for(int h=0;h<local_heads;h++)qtp_rope(q+h*Q38FN_HEAD_DIM,s->length);for(int h=0;h<Q38FN_KV_HEADS;h++)qtp_rope(k+h*Q38FN_HEAD_DIM,s->length);memcpy(s->keys+s->length*Q38FN_KV_HEADS*Q38FN_HEAD_DIM,k,sizeof(k));memcpy(s->values+s->length*Q38FN_KV_HEADS*Q38FN_HEAD_DIM,v,sizeof(v));s->length++;
 for(int lh=0;lh<local_heads;lh++){int gh=m->rank*local_heads+lh,kv=gh/(Q38FN_HEADS/Q38FN_KV_HEADS);float mx=-FLT_MAX,sum=0;for(size_t p=0;p<s->length;p++){double d=0;const float*ck=s->keys+p*Q38FN_KV_HEADS*Q38FN_HEAD_DIM+kv*Q38FN_HEAD_DIM;for(int i=0;i<Q38FN_HEAD_DIM;i++)d+=(double)q[lh*Q38FN_HEAD_DIM+i]*ck[i];s->scores[p]=(float)(d/16.0);if(s->scores[p]>mx)mx=s->scores[p];}for(size_t p=0;p<s->length;p++){s->scores[p]=expf(s->scores[p]-mx);sum+=s->scores[p];}for(size_t p=0;p<s->length;p++){float z=s->scores[p]/sum;const float*cv=s->values+p*Q38FN_KV_HEADS*Q38FN_HEAD_DIM+kv*Q38FN_HEAD_DIM;for(int i=0;i<Q38FN_HEAD_DIM;i++)att[lh*Q38FN_HEAD_DIM+i]+=z*cv[i];}}
 for(int i=0;i<local_width;i++)att[i]*=qtp_sig(g[i]);
 qtp_name(n,sizeof(n),L,"self_attn.o_proj.weight");return qtp_axis1_mv(m,qtp_find(m,n),att,Q38FN_HIDDEN,y);}
int q38fn_tp_attention_layer(q38fn_tp_model*m,int L,q38fn_tp_attention_state*s,float*h){float x[Q38FN_HIDDEN],y[Q38FN_HIDDEN],inj[4];int p=qtp_profile_enabled();double t=p?qtp_profile_now():0;if(qtp_hc(m,L,"attn",h,x,inj))return-1;if(p){qtp_profile.hc+=qtp_profile_now()-t;qtp_profile.hc_n++;t=qtp_profile_now();}if(qtp_attention(m,L,s,x,y))return-1;if(p){qtp_profile.attention+=qtp_profile_now()-t;qtp_profile.attention_n++;}qtp_residual(h,y,inj);t=p?qtp_profile_now():0;if(qtp_hc(m,L,"mlp",h,x,inj))return-1;if(p){qtp_profile.hc+=qtp_profile_now()-t;qtp_profile.hc_n++;t=qtp_profile_now();}if(qtp_moe(m,L,x,y))return-1;if(p){qtp_profile.moe+=qtp_profile_now()-t;qtp_profile.moe_n++;}qtp_residual(h,y,inj);return 0;}

int q38fn_tp_ple_apply(q38fn_tp_model*m,int L,q38fn_tp_ple_state*s,uint64_t tok,const float*emb,float*h){char n[224];float key[QTP_HC],value[Q38FN_HIDDEN],kn[QTP_HC],qn[QTP_HC],gated[QTP_HC];qtp_name(n,sizeof(n),L,"ple.key_proj.weight");if(qtp_full_mv(qtp_find(m,n),emb,QTP_HC,Q38FN_HIDDEN,key))return-1;qtp_name(n,sizeof(n),L,"ple.value_proj.weight");if(qtp_full_mv(qtp_find(m,n),emb,Q38FN_HIDDEN,Q38FN_HIDDEN,value))return-1;qtp_name(n,sizeof(n),L,"ple.norm_key.weight");const q38fn_tp_blob_entry*nk=qtp_find(m,n);qtp_name(n,sizeof(n),L,"ple.norm_query.weight");const q38fn_tp_blob_entry*nq=qtp_find(m,n);if(!nk||!nq)return-1;for(int st=0;st<4;st++){double a=0,b=0;for(int i=0;i<Q38FN_HIDDEN;i++){a+=(double)key[st*Q38FN_HIDDEN+i]*key[st*Q38FN_HIDDEN+i];b+=(double)h[st*Q38FN_HIDDEN+i]*h[st*Q38FN_HIDDEN+i];}float ka=1/sqrtf((float)(a/Q38FN_HIDDEN)+1e-6f),qb=1/sqrtf((float)(b/Q38FN_HIDDEN)+1e-6f);double d=0;for(int i=0;i<Q38FN_HIDDEN;i++){int j=st*Q38FN_HIDDEN+i;kn[j]=key[j]*ka*(1+qtp_bf(nk->data[j]));qn[j]=h[j]*qb*(1+qtp_bf(nq->data[j]));d+=(double)kn[j]*qn[j];}float z=(float)(d/sqrtf(Q38FN_HIDDEN));z=qtp_sig(copysignf(sqrtf(fmaxf(fabsf(z),1e-6f)),z));for(int i=0;i<Q38FN_HIDDEN;i++)gated[st*Q38FN_HIDDEN+i]=z*value[i];}
 qtp_name(n,sizeof(n),L,"ple.norm_conv.weight");const q38fn_tp_blob_entry*nc=qtp_find(m,n);qtp_name(n,sizeof(n),L,"ple.conv1d.weight");const q38fn_tp_blob_entry*cw=qtp_find(m,n);if(!nc||!cw)return-1;for(int st=0;st<4;st++){double q=0;for(int i=0;i<Q38FN_HIDDEN;i++)q+=(double)gated[st*Q38FN_HIDDEN+i]*gated[st*Q38FN_HIDDEN+i];float z=1/sqrtf((float)(q/Q38FN_HIDDEN)+1e-6f);for(int i=0;i<Q38FN_HIDDEN;i++){int c=st*Q38FN_HIDDEN+i;float norm=gated[c]*z*(1+qtp_bf(nc->data[c])),*hist=s->conv+(size_t)c*9;float v=hist[0]*qtp_bf(cw->data[c*4])+hist[3]*qtp_bf(cw->data[c*4+1])+hist[6]*qtp_bf(cw->data[c*4+2])+norm*qtp_bf(cw->data[c*4+3]);for(int j=0;j<8;j++)hist[j]=hist[j+1];hist[8]=norm;h[c]+=gated[c]+qtp_silu(v);}}s->previous2=s->previous;s->previous=tok;return 0;}

int q38fn_tp_final(q38fn_tp_model*m,const float*h,float*out){int prof=qtp_profile_enabled();double pt=prof?qtp_profile_now():0;const char*p="model.language_model.hyper_connection_mixer";char n[180];float norm[QTP_HC],lo[QTP_LAYER_MIX],part[QTP_HC];snprintf(n,sizeof(n),"%s.hc_norm.weight",p);const q38fn_tp_blob_entry*nw=qtp_find(m,n);if(!nw)return-1;for(int st=0;st<4;st++){double q=0;for(int i=0;i<Q38FN_HIDDEN;i++)q+=(double)h[st*Q38FN_HIDDEN+i]*h[st*Q38FN_HIDDEN+i];float z=1/sqrtf((float)(q/Q38FN_HIDDEN)+1e-6f);for(int i=0;i<Q38FN_HIDDEN;i++){int j=st*Q38FN_HIDDEN+i;norm[j]=h[j]*z*(1+qtp_bf(nw->data[j]));}}snprintf(n,sizeof(n),"%s.input_mix_weight_down.weight",p);const q38fn_tp_blob_entry*wd=qtp_find(m,n);int d=qtp_axis0_mv(wd,norm,QTP_HC,lo);if(d<0||d>QTP_LAYER_MIX)return-1;for(int i=0;i<d;i++)lo[i]=qtp_silu(lo[i]/4);snprintf(n,sizeof(n),"%s.input_mix_weight_up.weight",p);const q38fn_tp_blob_entry*wu=qtp_find(m,n);for(int r=0;r<QTP_HC;r++)part[r]=qtp_entry_row_dot(wu,(size_t)r,lo,d);m->sum(part,QTP_HC,m->comm);for(int i=0;i<Q38FN_HIDDEN;i++){double z=0;for(int st=0;st<4;st++){int j=st*Q38FN_HIDDEN+i;z+=(double)qtp_sig(part[j])*norm[j];}out[i]=(float)(z/4);}if(prof){qtp_profile.final_mix+=qtp_profile_now()-pt;qtp_profile.final_n++;}return 0;}
int q38fn_tp_head(q38fn_tp_model*m,const float*x,int*token,float*logit){int prof=qtp_profile_enabled();double pt=prof?qtp_profile_now():0;const q38fn_tp_blob_entry*e=qtp_find(m,"lm_head.weight");if(!e||e->kind!=Q38FN_TP_AXIS0)return-1;int rows=(int)e->range[0].count,best=-1;float bv=-FLT_MAX;
 float*logits=m->head_logits;if(!logits||m->head_logits_count<(size_t)rows||qtp_entry_mv(e,x,rows,Q38FN_HIDDEN,logits))return-1;
#ifdef _OPENMP
#pragma omp parallel
 {int lb=-1;float lv=-FLT_MAX;
#pragma omp for schedule(static)
 for(int r=0;r<rows;r++)if(logits[r]>lv){lv=logits[r];lb=r;}
#pragma omp critical
 if(lv>bv){bv=lv;best=lb;}}
#else
 for(int r=0;r<rows;r++)if(logits[r]>bv){bv=logits[r];best=r;}
#endif
 best+=(int)e->range[0].start;m->argmax(&bv,&best,m->comm);*token=best;if(logit)*logit=bv;if(prof){qtp_profile.head+=qtp_profile_now()-pt;qtp_profile.head_n++;}return 0;}
#endif
#endif
