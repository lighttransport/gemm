/* 12-way head-parallel real-weight GLM-5.3F sparse MLA core. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef GLM53F_EXTERNAL_ST_IMPLEMENTATION
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#endif
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"
#include "../../common/glm53f_arch.h"
#include <arm_sve.h>
#include <mpi.h>
#include <omp.h>
#ifndef __ARM_FEATURE_SVE
#define __ARM_FEATURE_SVE 1
#endif
#include "glm53f_expert_kern.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { H=4096,NH=64,KD=256,VD=256,LAT=512,QKV=16384 };
static void* a256(size_t n){void*p=NULL;if(posix_memalign(&p,256,n))p=NULL;if(!p)MPI_Abort(MPI_COMM_WORLD,2);return p;}
static void readp(glm53f_st_context*s,const char*n,size_t o,void*p,size_t z,int r){if(glm53f_st_read(s,n,o,p,z)){fprintf(stderr,"rank=%d read failed %s\n",r,n);MPI_Abort(MPI_COMM_WORLD,2);}}
static inline float fp8dot(const uint8_t*w,const float*sc,const float*x,int n){
    svfloat32_t a=svdup_f32(0);int vl=(int)svcntw();
    for(int b=0;b<n;b+=128){int e=b+128<n?b+128:n;for(int i=b;i<e;i+=vl){svbool_t p=svwhilelt_b32(i,e);a=svmla_x(p,a,glm53f_fp8_e4m3_bits(p,w,i),svmul_n_f32_x(p,svld1(p,x+i),sc[b/128]));}}
    return svaddv_f32(svptrue_b32(),a);
}
static inline float f32dot(const float*a,const float*b,int n){svfloat32_t s=svdup_f32(0);int vl=(int)svcntw();for(int i=0;i<n;i+=vl){svbool_t p=svwhilelt_b32(i,n);s=svmla_x(p,s,svld1(p,a+i),svld1(p,b+i));}return svaddv_f32(svptrue_b32(),s);}
static inline float bf16dot(const uint16_t*w,const float*x,int n){svfloat32_t s=svdup_f32(0);int vl=(int)svcntw();for(int i=0;i<n;i+=vl){svbool_t p=svwhilelt_b32(i,n);svuint32_t z=svlsl_n_u32_x(p,svld1uh_u32(p,w+i),16);s=svmla_x(p,s,svreinterpret_f32_u32(z),svld1(p,x+i));}return svaddv_f32(svptrue_b32(),s);}
static int mla_one(float*out,const float*q,const float*cache,const uint16_t*w,
                   const int*sel,int nt){float *ql=a256(LAT*4),*va=a256(LAT*4),*log=a256((size_t)nt*4);const uint16_t*wv=w+(size_t)KD*LAT;int vl=(int)svcntw();memset(ql,0,LAT*4);
    for(int j=0;j<KD;j++){float x=q[j]/sqrtf((float)KD);for(int d=0;d<LAT;d+=vl){svbool_t p=svwhilelt_b32(d,LAT);svuint32_t z=svlsl_n_u32_x(p,svld1uh_u32(p,w+(size_t)j*LAT+d),16);svst1(p,ql+d,svmla_n_f32_x(p,svld1(p,ql+d),svreinterpret_f32_u32(z),x));}}
    float mx=-INFINITY,sum=0;for(int t=0;t<nt;t++){log[t]=f32dot(ql,cache+(size_t)sel[t]*LAT,LAT);if(log[t]>mx)mx=log[t];}for(int t=0;t<nt;t++){log[t]=expf(log[t]-mx);sum+=log[t];}memset(va,0,LAT*4);
    for(int t=0;t<nt;t++){float x=log[t]/sum;const float*z=cache+(size_t)sel[t]*LAT;for(int d=0;d<LAT;d+=vl){svbool_t p=svwhilelt_b32(d,LAT);svst1(p,va+d,svmla_n_f32_x(p,svld1(p,va+d),svld1(p,z+d),x));}}
    for(int j=0;j<VD;j++)out[j]=bf16dot(wv+(size_t)j*LAT,va,LAT);free(log);free(va);free(ql);return 0;}
static int mla_heads(float*out,const float*q,const float*z,const uint16_t*w,
                     const int*sel,int nt,int nh){int fail=0;
#pragma omp parallel for schedule(static) reduction(|:fail)
    for(int h=0;h<nh;h++){
        const uint16_t*wh=w+(size_t)h*(KD+VD)*LAT;
        if(getenv("GLM53F_SPARSE_SCALAR"))fail|=
            glm53f_mla_selected_absorbed_bf16(out+(size_t)h*VD,
                q+(size_t)h*KD,z,wh,sel,nt,1,KD,VD,LAT)!=0;
        else fail|=mla_one(out+(size_t)h*VD,q+(size_t)h*KD,z,wh,sel,nt)!=0;
    }
    return fail?-1:0;
}
int main(int argc,char**argv){
    int rank,nr,layer=argc>3?atoi(argv[3]):43,tokens=argc>2?atoi(argv[2]):512,h0,hn,local;
    char n[256];glm53f_st_context*st;uint16_t*kvb;uint8_t*op;float*ops,*query,*latent,*attn,*partial,*out[2];int*sel;
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);
    if(argc<2||nr!=12||tokens<1||tokens>4096){if(!rank)fprintf(stderr,"usage: mpiexec -np 12 %s MODEL_DIR [tokens=512] [layer=43]\n",argv[0]);MPI_Finalize();return 2;}
    glm53f_balanced_slice(NH,rank,nr,&h0,&hn);local=hn*VD;st=glm53f_st_open(argv[1]);if(!st)MPI_Abort(MPI_COMM_WORLD,2);
    snprintf(n,sizeof n,"model.language_model.layers.%d.self_attn.kv_b_proj.weight",layer);kvb=a256((size_t)hn*(KD+VD)*LAT*2);readp(st,n,(size_t)h0*(KD+VD)*LAT*2,kvb,(size_t)hn*(KD+VD)*LAT*2,rank);
    snprintf(n,sizeof n,"model.language_model.layers.%d.self_attn.o_proj.weight",layer);op=a256((size_t)H*QKV);readp(st,n,0,op,(size_t)H*QKV,rank);
    snprintf(n,sizeof n,"model.language_model.layers.%d.self_attn.o_proj.weight_scale_inv",layer);const st_tensor_info*ti=glm53f_st_find(st,n,NULL);if(!ti){fprintf(stderr,"missing %s\n",n);MPI_Abort(MPI_COMM_WORLD,2);}ops=a256(ti->nbytes);readp(st,n,0,ops,ti->nbytes,rank);glm53f_st_close(st);
    query=a256((size_t)hn*KD*4);latent=a256((size_t)tokens*LAT*4);attn=a256((size_t)local*4);partial=a256(H*4);out[0]=a256(H*4);out[1]=a256(H*4);sel=a256((size_t)tokens*sizeof(int));
    for(int h=0;h<hn;h++)for(int d=0;d<KD;d++)query[(size_t)h*KD+d]=(float)((((h+h0)*31+d*7+5)%251)-125)/125.0f;
    for(int t=0;t<tokens;t++){sel[t]=t;for(int d=0;d<LAT;d++)latent[(size_t)t*LAT+d]=(float)(((t*29+d*11+3)%257)-128)/128.0f;}
    float phase[2][3],elapsed[2];int blocks=QKV/128,col0=h0*VD,b0=col0/128;
    for(int pass=0;pass<2;pass++){
        double t0=MPI_Wtime();if(mla_heads(attn,query,latent,kvb,sel,tokens,hn))MPI_Abort(MPI_COMM_WORLD,2);double t1=MPI_Wtime();
#pragma omp parallel for schedule(static)
        for(int r=0;r<H;r++)partial[r]=fp8dot(op+(size_t)r*QKV+col0,ops+(size_t)(r/128)*blocks+b0,attn,local);
        double t2=MPI_Wtime();MPI_Allreduce(partial,out[pass],H,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);double t3=MPI_Wtime();phase[pass][0]=t1-t0;phase[pass][1]=t2-t1;phase[pass][2]=t3-t2;elapsed[pass]=t3-t0;
    }
    int ok=!memcmp(out[0],out[1],H*4),allok;for(int i=0;i<H;i++)ok&=isfinite(out[0][i]);MPI_Allreduce(&ok,&allok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    float me,mp[3];MPI_Allreduce(&elapsed[1],&me,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);MPI_Allreduce(phase[1],mp,3,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);double ss=0,sum=0;for(int i=0;i<H;i++){ss+=(double)out[0][i]*out[0][i];sum+=out[0][i];}
    if(!rank)printf("GLM53F_SPARSE_CORE_12N layer=%d tokens=%d heads=%d max_ms=%.3f mla_ms=%.3f oproj_ms=%.3f allreduce_ms=%.3f rms=%.9g sum=%.9g repeat=%s %s\n",layer,tokens,NH,me*1e3f,mp[0]*1e3f,mp[1]*1e3f,mp[2]*1e3f,sqrt(ss/H),sum,allok?"BIT_EXACT":"FAIL",allok?"PASS":"FAIL");
    if(!rank){const char*d=getenv("GLM53F_SPARSE_OUTPUT");if(d&&*d){FILE*f=fopen(d,"wb");if(!f||fwrite(out[0],4,H,f)!=(size_t)H)MPI_Abort(MPI_COMM_WORLD,2);fclose(f);}}
    MPI_Finalize();return allok?0:1;
}
