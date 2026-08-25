#define _POSIX_C_SOURCE 200112L
#include "fp8_gemm.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

typedef struct {const uint8_t*q;const float*a;const float*s;float*out;int k,scale_group,lane_group,pad;const uint32_t*lut;} fp8_m1_args;
extern void fp8_e4m3fn_m1_asm(const fp8_m1_args*);
extern void fp8_e4m3fn_lut_m1_asm(const fp8_m1_args*);
extern void fp8_i8_m1_asm(const fp8_m1_args*);
extern void fp8_i8x4_m1_asm(const fp8_m1_args*);
typedef struct {const int8_t*q,*a;const float*ws,*as;float*out;int k,act_group;} fp8_sdot_args;
extern void fp8_i8_sdot_m1_asm(const fp8_sdot_args*);
static uint32_t fp8_lut[256] __attribute__((aligned(256)));
static int fp8_lut_ready;

int fp8_matrix_alloc(fp8_matrix *w,int n,int k){
    if(!w||n<=0||k<=0||n%128||k%128)return-1;
    memset(w,0,sizeof(*w));w->n=n;w->k=k;
    if(posix_memalign((void**)&w->codes,256,(size_t)n*k)||
       posix_memalign((void**)&w->scales,256,(size_t)(n/128)*(k/128)*sizeof(float))){
        fp8_matrix_free(w);return-1;
    }return 0;
}
void fp8_matrix_free(fp8_matrix*w){if(!w)return;free(w->codes);free(w->scales);
    free(w->exceptions);free(w->exception_offsets);memset(w,0,sizeof(*w));}

float fp8_e4m3fn_decode(uint8_t x){
    int s=x>>7,e=(x>>3)&15,m=x&7;float v;
    if(e==0)v=ldexpf((float)m,-9);
    else if(e==15&&m==7)v=0.0f;
    else v=ldexpf(1.0f+(float)m*.125f,e-7);
    return s?-v:v;
}

int fp8_i8_matrix_alloc(fp8_i8_matrix*w,int n,int k){
    if(!w||n<=0||k<=0||n%128||k%128)return-1;
    memset(w,0,sizeof(*w));w->n=n;w->k=k;
    if(posix_memalign((void**)&w->codes,256,(size_t)n*k)||
       posix_memalign((void**)&w->scales,256,
                      (size_t)(n/32)*k*sizeof(float))){
        fp8_i8_matrix_free(w);return-1;
    }return 0;
}

void fp8_i8_matrix_free(fp8_i8_matrix*w){
    if(!w)return;free(w->codes);free(w->sdot_codes);free(w->scales);memset(w,0,sizeof(*w));
}

static int quant_i8(float x,float inverse){
    long q=lrintf(x*inverse);
    if(q>127)q=127;else if(q< -127)q=-127;
    return(int)q;
}

int fp8_matrix_prepare_i8(const fp8_matrix*src,fp8_i8_matrix*dst,
                          fp8_i8_policy policy){
    return fp8_matrix_prepare_i8_group(src,dst,policy,128);
}

int fp8_matrix_prepare_i8_group(const fp8_matrix*src,fp8_i8_matrix*dst,
                                fp8_i8_policy policy,int scale_group){
    return fp8_matrix_prepare_i8_tile(src,dst,policy,scale_group,128);
}

int fp8_matrix_prepare_i8_tile(const fp8_matrix*src,fp8_i8_matrix*dst,
                               fp8_i8_policy policy,int scale_group,int lane_group){
    if(!src||!dst||!src->codes||!src->scales||
       (policy!=FP8_I8_ABSMAX&&policy!=FP8_I8_MSE)||scale_group<2||
       scale_group>128||(scale_group&(scale_group-1))||src->k%scale_group||
       (lane_group!=32&&lane_group!=128))return-1;
    if((!dst->codes||!dst->scales||dst->n!=src->n||dst->k!=src->k)&&
       fp8_i8_matrix_alloc(dst,src->n,src->k))return-1;
    free(dst->sdot_codes);dst->sdot_codes=NULL;
    dst->scale_group=scale_group;dst->lane_group=lane_group;
    int scale_count=src->k/scale_group,lane_tiles=128/lane_group;
    for(int g=0;g<src->n/128;++g)for(int kb=0;kb<scale_count;++kb)
      for(int lt=0;lt<lane_tiles;++lt){
        uint32_t hist[256]={0};float maxv=0.0f;
        for(int kk=0;kk<scale_group;++kk){
            const uint8_t*q=src->codes+((size_t)g*src->k+kb*scale_group+kk)*128;
            for(int lane=lt*lane_group;lane<(lt+1)*lane_group;++lane)++hist[q[lane]];
        }
        for(int c=0;c<256;++c)if(hist[c]){
            float v=fabsf(fp8_e4m3fn_decode((uint8_t)c));if(v>maxv)maxv=v;
        }
        float threshold=maxv;
        if(policy==FP8_I8_MSE&&maxv>0.0f){
            double best=HUGE_VAL;
            for(int step=0;step<=64;++step){
                float t=maxv*(1.0f-(float)step/128.0f),inv=127.0f/t;
                double error=0.0;
                for(int c=0;c<256;++c)if(hist[c]){
                    float v=fp8_e4m3fn_decode((uint8_t)c);
                    float d=v-(float)quant_i8(v,inv)*(t/127.0f);
                    error+=(double)hist[c]*d*d;
                }
                if(error<best){best=error;threshold=t;}
            }
        }
        float qscale=maxv>0.0f?threshold/127.0f:1.0f;
        float inv=1.0f/qscale;
        dst->scales[((size_t)g*scale_count+kb)*lane_tiles+lt]=
            src->scales[(size_t)g*(src->k/128)+(kb*scale_group)/128]*qscale;
        for(int kk=0;kk<scale_group;++kk){
            size_t off=((size_t)g*src->k+kb*scale_group+kk)*128;
            for(int lane=lt*lane_group;lane<(lt+1)*lane_group;++lane)
                dst->codes[off+lane]=(int8_t)quant_i8(
                    fp8_e4m3fn_decode(src->codes[off+lane]),inv);
        }
    }return 0;
}
static void prepare_lut(void){if(fp8_lut_ready)return;for(int i=0;i<256;++i){float x=fp8_e4m3fn_decode((uint8_t)i);memcpy(fp8_lut+i,&x,4);}fp8_lut_ready=1;}

int fp8_matrix_prepare_fast(fp8_matrix*w){
    if(!w||!w->codes||!w->scales)return-1;free(w->exceptions);free(w->exception_offsets);
    w->exceptions=NULL;w->exception_offsets=NULL;size_t count=0,total=(size_t)w->n*w->k;
    for(size_t i=0;i<total;++i){uint8_t x=w->codes[i];if(!(x&0x78)||(x&0x7f)==0x7f)++count;}
    if(posix_memalign((void**)&w->exception_offsets,256,(size_t)(w->n/128+1)*sizeof(uint32_t))||
       (count&&posix_memalign((void**)&w->exceptions,256,count*sizeof(fp8_exception))))return-1;
    size_t z=0;for(int g=0;g<w->n/128;++g){w->exception_offsets[g]=(uint32_t)z;
      for(int k=0;k<w->k;++k)for(int lane=0;lane<128;++lane){size_t i=((size_t)g*w->k+k)*128+lane;
        uint8_t x=w->codes[i];if(!(x&0x78)||(x&0x7f)==0x7f){uint8_t c=(uint8_t)(8|(x&0x80));
          w->exceptions[z++]=(fp8_exception){(uint32_t)k,(uint8_t)lane,x,c,0};w->codes[i]=c;}}
    }w->exception_offsets[w->n/128]=(uint32_t)z;return 0;
}

static void correct_group(float*out,const float*a,const fp8_matrix*w,int g){
    if(!w->exception_offsets)return;for(uint32_t i=w->exception_offsets[g];i<w->exception_offsets[g+1];++i){
      fp8_exception e=w->exceptions[i];float d=fp8_e4m3fn_decode(e.raw)-fp8_e4m3fn_decode(e.canonical);
      out[g*128+e.lane]+=a[e.k]*w->scales[(size_t)g*(w->k/128)+e.k/128]*d;}
}

int fp8_gemv_reference(float*out,const float*a,const fp8_matrix*w){
    if(!out||!a||!w||!w->codes||!w->scales)return-1;
    for(int r=0;r<w->n;++r){double z=0;int g=r/128,t=(r%128)/32,l=r%32;
      for(int k=0;k<w->k;++k){const uint8_t*q=w->codes+(((size_t)g*w->k+k)*4+t)*32;
        z+=(double)a[k]*fp8_e4m3fn_decode(q[l])*w->scales[(size_t)g*(w->k/128)+k/128];}
      out[r]=(float)z;
    }return 0;
}

int fp8_i8_gemv_reference(float*out,const float*a,const fp8_i8_matrix*w){
    if(!out||!a||!w||!w->codes||!w->scales)return-1;
    int lane_tiles=128/w->lane_group;
    for(int r=0;r<w->n;++r){double z=0.0;int g=r/128,lane=r%128;
      for(int k=0;k<w->k;++k){size_t off=((size_t)g*w->k+k)*128+lane;
        z+=(double)a[k]*w->codes[off]*
           w->scales[((size_t)g*(w->k/w->scale_group)+k/w->scale_group)*lane_tiles+
                     lane/w->lane_group];}
      out[r]=(float)z;
    }return 0;
}

int fp8_i8_matrix_prepare_sdot(fp8_i8_matrix*w){
    if(!w||!w->codes||!w->scales||w->scale_group!=128||w->lane_group!=128)return-1;
    free(w->sdot_codes);w->sdot_codes=NULL;
    if(posix_memalign((void**)&w->sdot_codes,256,(size_t)w->n*w->k))return-1;
    for(int g=0;g<w->n/128;++g)for(int b=0;b<w->k/128;++b)
      for(int p=0;p<32;++p)for(int v=0;v<8;++v)for(int lane=0;lane<16;++lane)
        for(int j=0;j<4;++j){
            int k=b*128+p*4+j,nlane=v*16+lane;
            size_t src=((size_t)g*w->k+k)*128+nlane;
            size_t dst=((((size_t)g*(w->k/128)+b)*32+p)*8+v)*64+lane*4+j;
            w->sdot_codes[dst]=w->codes[src];
        }
    return 0;
}

int fp8_i8_activation_prepare(fp8_i8_activation*q,const float*a,int k){
    return fp8_i8_activation_prepare_group(q,a,k,128);
}

int fp8_i8_activation_prepare_group(fp8_i8_activation*q,const float*a,int k,
                                    int scale_group){
    if(!q||!a||k<=0||k%128||scale_group<4||scale_group>128||
       (scale_group&(scale_group-1)))return-1;
    if((!q->codes||!q->scales||q->k!=k)){
        fp8_i8_activation_free(q);q->k=k;
        if(posix_memalign((void**)&q->codes,256,(size_t)k)||
           posix_memalign((void**)&q->scales,256,(size_t)(k/4)*sizeof(float))){
            fp8_i8_activation_free(q);return-1;
        }
    }
    q->scale_group=scale_group;
    for(int b=0;b<k/scale_group;++b){float m=0.0f;
        for(int j=0;j<scale_group;++j){float x=fabsf(a[b*scale_group+j]);if(x>m)m=x;}
        float scale=m>0.0f?m/127.0f:1.0f,inv=1.0f/scale;q->scales[b]=scale;
        for(int j=0;j<scale_group;++j)q->codes[b*scale_group+j]=
            (int8_t)quant_i8(a[b*scale_group+j],inv);
    }return 0;
}

void fp8_i8_activation_free(fp8_i8_activation*q){
    if(!q)return;free(q->codes);free(q->scales);memset(q,0,sizeof(*q));
}

#if defined(__ARM_FEATURE_SVE)
static inline svfloat32_t fp8_decode_normal(svuint32_t q,svbool_t p){
    svuint32_t misplaced_sign=svlsl_n_u32_x(p,svand_n_u32_x(p,q,0x80),20);
    svuint32_t bits=svadd_n_u32_x(p,svlsl_n_u32_x(p,q,20),0x3c000000u);
    bits=svmla_n_u32_x(p,bits,misplaced_sign,15);
    return svreinterpret_f32_u32(bits);
}
static inline svfloat32_t fp8_decode_exact(svuint32_t q,svbool_t p){
    svfloat32_t normal=fp8_decode_normal(q,p);
    svuint32_t mant=svand_n_u32_x(p,q,7);
    svfloat32_t sub=svmul_n_f32_x(p,svcvt_f32_u32_x(p,mant),0x1p-9f);
    svuint32_t sign=svlsl_n_u32_x(p,svand_n_u32_x(p,q,0x80),24);
    sub=svreinterpret_f32_u32(svorr_u32_x(p,svreinterpret_u32_f32(sub),sign));
    svbool_t exp0=svcmpeq_n_u32(p,svand_n_u32_x(p,q,0x78),0);
    svfloat32_t value=svsel_f32(exp0,sub,normal);
    svbool_t nan=svcmpeq_n_u32(p,svand_n_u32_x(p,q,0x7f),0x7f);
    return svsel_f32(nan,svdup_f32(0),value);
}

static inline void fp8_decode_via_f16(const uint8_t*q,svbool_t ph,svbool_t ps,
        svuint16_t table,svfloat32_t*lo,svfloat32_t*hi){
    svuint16_t raw=svld1ub_u16(ph,q),low=svand_n_u16_x(ph,raw,15);
    svuint16_t high=svlsr_n_u16_x(ph,raw,4);
    svuint16_t bits=svadd_u16_x(ph,svreinterpret_u16_f16(svtbl_f16(
            svreinterpret_f16_u16(table),high)),svlsl_n_u16_x(ph,low,7));
    *lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(bits)));
    *hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(bits)));
}

static inline void tile4_f16decode(float*out,const float*a,const fp8_matrix*w,int g){
    static const uint16_t td[32] __attribute__((aligned(64)))={
      0x2000,0x2800,0x3000,0x3800,0x4000,0x4800,0x5000,0x5800,
      0xa000,0xa800,0xb000,0xb800,0xc000,0xc800,0xd000,0xd800,
      0x2000,0x2800,0x3000,0x3800,0x4000,0x4800,0x5000,0x5800,
      0xa000,0xa800,0xb000,0xb800,0xc000,0xc800,0xd000,0xd800};
    svbool_t ph=svptrue_b16(),ps=svptrue_b32();svuint16_t tab=svld1_u16(ph,td);
    svfloat32_t h0=svdup_f32(0),h1=h0,h2=h0,h3=h0,h4=h0,h5=h0,h6=h0,h7=h0;
    const uint8_t*q=w->codes+(size_t)g*w->k*128;const float*s=w->scales+(size_t)g*(w->k/128);
    for(int b=0;b<w->k/128;++b){float scale=s[b];for(int k=b*128;k<(b+1)*128;++k){
      svfloat32_t x=svdup_f32(a[k]*scale),v0,v1;const uint8_t*r=q+(size_t)k*128;
#define F16_PAIR(T,H0,H1) do{fp8_decode_via_f16(r+(T)*32,ph,ps,tab,&v0,&v1); \
        (H0)=svmla_f32_x(ps,(H0),v0,x);(H1)=svmla_f32_x(ps,(H1),v1,x);}while(0)
      F16_PAIR(0,h0,h1);F16_PAIR(1,h2,h3);F16_PAIR(2,h4,h5);F16_PAIR(3,h6,h7);
#undef F16_PAIR
    }}
    svst1_f32(ps,out+g*128,h0);svst1_f32(ps,out+g*128+16,h1);
    svst1_f32(ps,out+g*128+32,h2);svst1_f32(ps,out+g*128+48,h3);
    svst1_f32(ps,out+g*128+64,h4);svst1_f32(ps,out+g*128+80,h5);
    svst1_f32(ps,out+g*128+96,h6);svst1_f32(ps,out+g*128+112,h7);
}

static inline void tile4(float*out,const float*a,const fp8_matrix*w,int g,int exact){
    svbool_t p=svptrue_b32();svfloat32_t h0=svdup_f32(0),h1=h0,h2=h0,h3=h0;
    svfloat32_t h4=h0,h5=h0,h6=h0,h7=h0;
    const uint8_t*q=w->codes+(size_t)g*w->k*128;const float*s=w->scales+(size_t)g*(w->k/128);
    for(int b=0;b<w->k/128;++b){float scale=s[b];
      for(int k=b*128;k<(b+1)*128;++k){svfloat32_t x=svdup_f32(a[k]*scale);
        const uint8_t*r=q+(size_t)k*128;
#define FP8_PAIR(T,H0,H1) do{svuint32_t q0=svld1ub_u32(p,r+(T)*32); \
          svuint32_t q1=svld1ub_u32(p,r+(T)*32+16); \
          svfloat32_t v0=exact?fp8_decode_exact(q0,p):fp8_decode_normal(q0,p); \
          svfloat32_t v1=exact?fp8_decode_exact(q1,p):fp8_decode_normal(q1,p); \
          (H0)=svmla_f32_x(p,(H0),v0,x);(H1)=svmla_f32_x(p,(H1),v1,x);}while(0)
        FP8_PAIR(0,h0,h1);FP8_PAIR(1,h2,h3);FP8_PAIR(2,h4,h5);FP8_PAIR(3,h6,h7);
#undef FP8_PAIR
      }}
    svst1_f32(p,out+g*128+0,h0);svst1_f32(p,out+g*128+16,h1);
    svst1_f32(p,out+g*128+32,h2);svst1_f32(p,out+g*128+48,h3);
    svst1_f32(p,out+g*128+64,h4);svst1_f32(p,out+g*128+80,h5);
    svst1_f32(p,out+g*128+96,h6);svst1_f32(p,out+g*128+112,h7);
}
#endif

int fp8_gemv_f32_omp(float*out,const float*a,const fp8_matrix*w,int threads,fp8_decode_mode exact){
    if(!out||!a||!w||!w->codes||!w->scales||threads<1)return-1;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
    if(exact==3){
#pragma omp parallel for num_threads(threads) schedule(static)
      for(int g=0;g<w->n/128;++g){tile4_f16decode(out,a,w,g);correct_group(out,a,w,g);}
      return 0;
    }else if(exact==2){prepare_lut();
#pragma omp parallel for num_threads(threads) schedule(static)
      for(int g=0;g<w->n/128;++g){fp8_m1_args x={w->codes+(size_t)g*w->k*128,
          a,w->scales+(size_t)g*(w->k/128),out+g*128,w->k,0,0,0,fp8_lut};fp8_e4m3fn_lut_m1_asm(&x);correct_group(out,a,w,g);}
      return 0;
    }else if(!exact){
#pragma omp parallel for num_threads(threads) schedule(static)
      for(int g=0;g<w->n/128;++g){fp8_m1_args x={w->codes+(size_t)g*w->k*128,
          a,w->scales+(size_t)g*(w->k/128),out+g*128,w->k,0,0,0,NULL};fp8_e4m3fn_m1_asm(&x);correct_group(out,a,w,g);}
      return 0;
    }
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int g=0;g<w->n/128;++g){tile4(out,a,w,g,exact!=0);correct_group(out,a,w,g);}
    return 0;
#else
    (void)exact;(void)threads;return-1;
#endif
}


int fp8_i8_gemv_f32_omp(float*out,const float*a,const fp8_i8_matrix*w,int threads){
    if(!out||!a||!w||!w->codes||!w->scales||threads<1)return-1;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int g=0;g<w->n/128;++g){
        int lane_tiles=128/w->lane_group;
        fp8_m1_args x={(const uint8_t*)w->codes+(size_t)g*w->k*128,
            a,w->scales+(size_t)g*(w->k/w->scale_group)*lane_tiles,out+g*128,w->k,
            w->scale_group,w->lane_group,0,NULL};
        if(w->lane_group==32)fp8_i8x4_m1_asm(&x);else fp8_i8_m1_asm(&x);
    }
    return 0;
#else
    (void)threads;return fp8_i8_gemv_reference(out,a,w);
#endif
}


int fp8_i8_gemv_sdot_omp(float*out,const fp8_i8_activation*a,
                         const fp8_i8_matrix*w,int threads){
    if(!out||!a||!w||!a->codes||!a->scales||!w->sdot_codes||!w->scales||
       a->k!=w->k||w->scale_group!=128||w->lane_group!=128||threads<1)return-1;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int g=0;g<w->n/128;++g){
        fp8_sdot_args x={w->sdot_codes+(size_t)g*w->k*128,a->codes,
            w->scales+(size_t)g*(w->k/128),a->scales,out+g*128,w->k,a->scale_group};
        fp8_i8_sdot_m1_asm(&x);
    }
    return 0;
#else
    (void)threads;return-1;
#endif
}
